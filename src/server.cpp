#include "server.h"

#include "aof.h"
#include "epoll_compat.h"
#include "storage.h"
#include "third_party/nlohmann/json.hpp"

#include <algorithm>
#include <arpa/inet.h>
#include <cctype>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <netinet/in.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <string>
#include <unistd.h>
#include <utility>

using namespace std;

namespace {

constexpr int kMaxEvents = 64;
constexpr int kReadChunk = 64 * 1024;
constexpr size_t kMaxQueuedChunks = 1024;

bool parse_int(const string& text, int& out) {
    if (text.empty()) {
        return false;
    }
    size_t i = (text[0] == '-' || text[0] == '+') ? 1 : 0;
    if (i == text.size()) {
        return false;
    }
    for (; i < text.size(); ++i) {
        if (!isdigit(static_cast<unsigned char>(text[i]))) {
            return false;
        }
    }
    try {
        out = stoi(text);
        return true;
    } catch (const exception&) {
        return false;
    }
}

string lowercase(const string& text) {
    string out = text;
    transform(out.begin(), out.end(), out.begin(), [](unsigned char c) {
        return static_cast<char>(tolower(c));
    });
    return out;
}

}  // namespace

Server::Server(int port, Storage* storage, AOFLogger* aof, size_t max_value_size,
               int ttl_default_sec, double semantic_threshold,
               const std::string& embedding_worker_url, int shards,
               const std::string& mode)
    : port_(port),
      storage_(storage),
      aof_(aof),
      max_value_size_(max_value_size),
      ttl_default_sec_(ttl_default_sec),
      semantic_threshold_(semantic_threshold),
      embedding_worker_url_(embedding_worker_url),
      shards_(shards),
      mode_(mode) {}

Server::~Server() {
    if (listen_fd_ != -1) {
        close(listen_fd_);
    }
    if (epoll_fd_ != -1) {
        close(epoll_fd_);
    }
    if (wake_pipe_[0] != -1) {
        close(wake_pipe_[0]);
    }
    if (wake_pipe_[1] != -1) {
        close(wake_pipe_[1]);
    }
}

bool Server::call_embedding_worker(const std::string& text,
                                   std::vector<float>& out_embedding) {
    if (embedding_worker_url_.empty()) {
        return false;
    }

    // Parse URL: http://host:port/path
    std::string host;
    int port = 80;
    std::string path = "/embedding";
    std::string url = embedding_worker_url_;
    if (url.compare(0, 7, "http://") == 0) {
        url = url.substr(7);
    }
    auto slash = url.find('/');
    if (slash != std::string::npos) {
        host = url.substr(0, slash);
        path = url.substr(slash);
    } else {
        host = url;
    }
    auto colon = host.find(':');
    if (colon != std::string::npos) {
        port = std::stoi(host.substr(colon + 1));
        host = host.substr(0, colon);
    }

    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) {
        fprintf(stderr, "embed_worker: socket creation failed\n");
        return false;
    }

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<uint16_t>(port));
    if (inet_pton(AF_INET, host.c_str(), &addr.sin_addr) <= 0) {
        close(sock);
        return false;
    }

    // Non-blocking connect with timeout
    int flags = fcntl(sock, F_GETFL, 0);
    fcntl(sock, F_SETFL, flags | O_NONBLOCK);

    if (connect(sock, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
        if (errno != EINPROGRESS) {
            close(sock);
            return false;
        }
        fd_set wfd;
        FD_ZERO(&wfd);
        FD_SET(sock, &wfd);
        timeval tv{5, 0};
        if (select(sock + 1, nullptr, &wfd, nullptr, &tv) <= 0) {
            close(sock);
            return false;
        }
    }

    fcntl(sock, F_SETFL, flags);

    // Build JSON POST body
    // Escape quotes and newlines in text for JSON
    std::string escaped_text = text;
    std::string body = "{\"input\":\"" + escaped_text + "\"}";
    std::string request =
        "POST " + path + " HTTP/1.1\r\n"
        "Host: " + host + "\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(body.size()) + "\r\n"
        "Connection: close\r\n"
        "\r\n" +
        body;

    ssize_t sent = send(sock, request.data(), request.size(), MSG_NOSIGNAL);
    if (sent < 0) {
        close(sock);
        return false;
    }

    // Read response
    std::string response;
    char buf[4096];
    while (true) {
        fd_set rfd;
        FD_ZERO(&rfd);
        FD_SET(sock, &rfd);
        timeval tv{5, 0};
        int r = select(sock + 1, &rfd, nullptr, nullptr, &tv);
        if (r <= 0) {
            break;
        }
        ssize_t n = recv(sock, buf, sizeof(buf), 0);
        if (n <= 0) {
            break;
        }
        response.append(buf, static_cast<size_t>(n));
    }
    close(sock);

    // Strip HTTP headers to find JSON body.
    size_t body_start = response.find("\r\n\r\n");
    size_t skip = 4;
    if (body_start == std::string::npos) {
        body_start = response.find("\n\n");
        skip = 2;
    }
    if (body_start != std::string::npos) {
        body_start += skip;
    } else {
        body_start = 0;
    }
    std::string resp_body = response.substr(body_start);

    // Parse JSON response: [{"index":0,"embedding":[[0.1,0.2,...]]}]
    try {
        auto j = nlohmann::json::parse(resp_body);
        if (!j.is_array() || j.empty()) {
            fprintf(stderr, "embed_worker: response is not a non-empty array\n");
            return false;
        }
        auto& first = j[0];
        if (!first.contains("embedding") || !first["embedding"].is_array()) {
            fprintf(stderr, "embed_worker: no embedding array in response\n");
            return false;
        }
        auto& emb_array = first["embedding"];
        if (!emb_array.is_array() || emb_array.empty()) {
            fprintf(stderr, "embed_worker: embedding is not a non-empty array\n");
            return false;
        }
        auto& emb_vector = emb_array[0];
        if (!emb_vector.is_array()) {
            fprintf(stderr, "embed_worker: embedding[0] is not an array\n");
            return false;
        }
        out_embedding.clear();
        for (const auto& el : emb_vector) {
            out_embedding.push_back(static_cast<float>(el.get<double>()));
        }
        fprintf(stderr, "embed_worker: got %zu-dim embedding\n", out_embedding.size());
        return !out_embedding.empty();
    } catch (const nlohmann::json::exception&) {
        fprintf(stderr, "embed_worker: JSON parse failed\n");
        return false;
    }
}

void Server::start() {
    listen_fd_ = socket(AF_INET, SOCK_STREAM, 0);
    if (listen_fd_ < 0) {
        perror("socket");
        return;
    }
    int flags = fcntl(listen_fd_, F_GETFL, 0);
    fcntl(listen_fd_, F_SETFL, flags | O_NONBLOCK);

    int opt = 1;
    setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    addr.sin_port = htons(static_cast<uint16_t>(port_));
    if (::bind(listen_fd_, reinterpret_cast<const sockaddr*>(&addr), sizeof(addr)) < 0) {
        perror("bind");
        close(listen_fd_);
        listen_fd_ = -1;
        return;
    }
    if (listen(listen_fd_, SOMAXCONN) < 0) {
        perror("listen");
        close(listen_fd_);
        listen_fd_ = -1;
        return;
    }

    epoll_fd_ = epoll_create1(0);
    if (epoll_fd_ < 0) {
        perror("epoll_create1");
        close(listen_fd_);
        listen_fd_ = -1;
        return;
    }

    if (socketpair(AF_UNIX, SOCK_STREAM, 0, wake_pipe_) < 0) {
        perror("socketpair");
        close(epoll_fd_);
        epoll_fd_ = -1;
        close(listen_fd_);
        listen_fd_ = -1;
        return;
    }
    int wake_flags = fcntl(wake_pipe_[0], F_GETFL, 0);
    fcntl(wake_pipe_[0], F_SETFL, wake_flags | O_NONBLOCK);
    wake_flags = fcntl(wake_pipe_[1], F_GETFL, 0);
    fcntl(wake_pipe_[1], F_SETFL, wake_flags | O_NONBLOCK);

    epoll_event ev{};
    ev.events = EPOLLIN;
    ev.data.fd = listen_fd_;
    epoll_ctl(epoll_fd_, EPOLL_CTL_ADD, listen_fd_, &ev);

    ev.data.fd = wake_pipe_[0];
    epoll_ctl(epoll_fd_, EPOLL_CTL_ADD, wake_pipe_[0], &ev);

    running_ = true;
    epoll_event events[kMaxEvents];
    while (running_) {
        const int n = epoll_wait(epoll_fd_, events, kMaxEvents, -1);
        if (n < 0) {
            if (errno == EINTR) {
                continue;
            }
            break;
        }
        for (int i = 0; i < n; ++i) {
            const int fd = events[i].data.fd;
            const uint32_t event_flags = events[i].events;
            if (fd == wake_pipe_[0]) {
                char discard[64];
                while (recv(fd, discard, sizeof(discard), 0) > 0) {
                }
                continue;
            }
            if (fd == listen_fd_) {
                accept_connections();
                continue;
            }
            if ((event_flags & (EPOLLERR | EPOLLHUP)) != 0 &&
                (event_flags & EPOLLIN) == 0) {
                close_client(fd);
                continue;
            }
            if ((event_flags & EPOLLIN) != 0) {
                handle_readable(fd);
                if (parsers_.find(fd) == parsers_.end()) {
                    continue;
                }
            }
            if ((event_flags & EPOLLOUT) != 0) {
                try_send(fd);
            }
            if ((event_flags & (EPOLLERR | EPOLLHUP)) != 0 &&
                parsers_.find(fd) != parsers_.end()) {
                close_client(fd);
            }
        }
    }

    close(listen_fd_);
    listen_fd_ = -1;
    close(epoll_fd_);
    epoll_fd_ = -1;
    close(wake_pipe_[0]);
    close(wake_pipe_[1]);
    wake_pipe_[0] = -1;
    wake_pipe_[1] = -1;
}

void Server::stop() {
    running_ = false;
    if (wake_pipe_[1] != -1) {
        const char byte = 1;
        (void)write(wake_pipe_[1], &byte, 1);
    }
}

void Server::accept_connections() {
    for (;;) {
        const int client_fd = accept(listen_fd_, nullptr, nullptr);
        if (client_fd < 0) {
            if (errno == EINTR) {
                continue;
            }
            return;
        }
        int flags = fcntl(client_fd, F_GETFL, 0);
        fcntl(client_fd, F_SETFL, flags | O_NONBLOCK);
        COMPAT_SET_NOSIGPIPE(client_fd);

        epoll_event ev{};
        ev.events = EPOLLIN;
        ev.data.fd = client_fd;
        epoll_ctl(epoll_fd_, EPOLL_CTL_ADD, client_fd, &ev);

        // Let the parser accept a little more than the configured maximum so
        // an oversized value gets a clean -ERR reply instead of a protocol
        // error, while a hostile declaration far above the limit is still
        // rejected before the allocation. Cap at the memory limit too: a
        // 1MB test server must not allocate a 512MB parser buffer just
        // because --max-value-size defaults high.
        constexpr size_t kParseOvershoot = 1024 * 1024;
        const size_t effective_max = min(max_value_size_, storage_->memory_limit());
        const size_t parse_limit = effective_max + kParseOvershoot;
        parsers_.try_emplace(client_fd, parse_limit);
        watch_flags_[client_fd] = EPOLLIN;
    }
}

void Server::close_client(int fd) {
    epoll_ctl(epoll_fd_, EPOLL_CTL_DEL, fd, nullptr);
    close(fd);
    parsers_.erase(fd);
    outbox_.erase(fd);
    watch_flags_.erase(fd);
}

void Server::handle_readable(int fd) {
    char buf[kReadChunk];
    for (;;) {
        const ssize_t n = recv(fd, buf, sizeof(buf), 0);
        if (n > 0) {
            auto parser_it = parsers_.find(fd);
            if (parser_it == parsers_.end()) {
                return;
            }
            protocol::Parser& parser = parser_it->second;

            const char* cursor = buf;
            size_t left = static_cast<size_t>(n);
            while (left > 0) {
                protocol::Command cmd;
                const protocol::Parser::Status status =
                    parser.consume(cursor, left, cmd);
                if (status == protocol::Parser::Status::NeedMore) {
                    break;
                }
                if (status == protocol::Parser::Status::Error) {
                    const string err =
                        "-ERR Protocol error: " + parser.error() + "\r\n";
                    (void)send(fd, err.data(), err.size(), COMPAT_MSG_NOSIGNAL);
                    close_client(fd);
                    return;
                }
                if (cmd.type == protocol::Command::UNKNOWN && cmd.verb.empty()) {
                    continue;  // blank array / keepalive with no verb
                }
                queue_response(fd, execute_command(std::move(cmd)));
                if (parsers_.find(fd) == parsers_.end()) {
                    return;
                }
            }
            continue;
        }
        if (n == 0) {
            close_client(fd);
            return;
        }
        if (errno == EAGAIN || errno == EWOULDBLOCK) {
            break;
        }
        if (errno == EINTR) {
            continue;
        }
        close_client(fd);
        return;
    }
}

Server::Response Server::execute_command(protocol::Command&& cmd) {
    switch (cmd.type) {
        case protocol::Command::PING: {
            if (cmd.args.empty()) {
                return Response{"+PONG", nullptr, ""};
            }
            try {
                auto blob = make_shared<const string>(std::move(cmd.args[0]));
                return Response{"$" + to_string(blob->size()) + "\r\n", blob, "\r\n"};
            } catch (const bad_alloc&) {
                return Response{"-ERR out of memory", nullptr, ""};
            }
        }

        case protocol::Command::SELECT:
            // Anamesis is single-database; SELECT is accepted and ignored.
            return Response{"+OK", nullptr, ""};

        case protocol::Command::SET: {
            if (cmd.args.size() < 2) {
                return Response{"-ERR wrong number of arguments for SET", nullptr, ""};
            }
            if (cmd.args[0].size() > Storage::kMaxKeySize) {
                return Response{"-ERR key too large", nullptr, ""};
            }
            if (cmd.args[1].size() > max_value_size_) {
                return Response{"-ERR value too large (max-value-size)", nullptr, ""};
            }

            int ttl = 0;
            bool saw_expiry = false;
            for (size_t k = 2; k < cmd.args.size(); ++k) {
                const string opt = lowercase(cmd.args[k]);
                if (opt == "ex" || opt == "px") {
                    if (k + 1 >= cmd.args.size()) {
                        return Response{"-ERR syntax error", nullptr, ""};
                    }
                    int amount = 0;
                    if (!parse_int(cmd.args[k + 1], amount) || amount <= 0) {
                        return Response{"-ERR invalid expire value", nullptr, ""};
                    }
                    // TTL storage is second-granularity; round PX up so a
                    // sub-second TTL never turns into "no expiry".
                    ttl = (opt == "px")
                              ? amount / 1000 + (amount % 1000 != 0 ? 1 : 0)
                              : amount;
                    saw_expiry = true;
                    ++k;
                    continue;
                }
                // Legacy positional TTL ("SET key value 60"); keep it only for
                // the first trailing token and only when numeric. Everything
                // else (NX/XX/GET/KEEPTTL) would silently change semantics, so
                // reject it instead of pretending to support it.
                int legacy_ttl = 0;
                if (k == 2 && !saw_expiry && parse_int(cmd.args[k], legacy_ttl)) {
                    ttl = legacy_ttl;
                    continue;
                }
                return Response{"-ERR unsupported SET option: " + cmd.args[k], nullptr, ""};
            }

            if (!storage_->can_store(cmd.args[0], cmd.args[1])) {
                return Response{"-ERR value too large for memory limit", nullptr, ""};
            }

            try {
                if (aof_ != nullptr) {
                    aof_->log_set(cmd.args[0], cmd.args[1], ttl);
                }
                storage_->set(std::move(cmd.args[0]), std::move(cmd.args[1]), ttl);
            } catch (const bad_alloc&) {
                return Response{"-ERR out of memory", nullptr, ""};
            }
            return Response{"+OK", nullptr, ""};
        }

        case protocol::Command::GET: {
            if (cmd.args.empty()) {
                return Response{"-ERR wrong number of arguments for GET", nullptr, ""};
            }
            Blob blob = storage_->get(cmd.args[0]);
            if (!blob) {
                return Response{"$-1", nullptr, ""};
            }
            return Response{"$" + to_string(blob->size()) + "\r\n", blob, "\r\n"};
        }

        case protocol::Command::DEL: {
            if (cmd.args.empty()) {
                return Response{"-ERR wrong number of arguments for DEL", nullptr, ""};
            }
            const bool removed = storage_->del(cmd.args[0]);
            if (removed && aof_ != nullptr) {
                aof_->log_del(cmd.args[0]);
            }
            return Response{removed ? ":1" : ":0", nullptr, ""};
        }

        case protocol::Command::METRICS: {
            const string json = storage_->metrics().dump();
            return Response{"$" + to_string(json.size()) + "\r\n" + json + "\r\n",
                            nullptr, ""};
        }

        case protocol::Command::SEM_PUT: {
            if (cmd.args.size() < 4) {
                return Response{"-ERR wrong number of arguments for SEM.PUT", nullptr, ""};
            }
            uint64_t namespace_id = 0;
            try {
                namespace_id = stoul(cmd.args[0]);
            } catch (const exception&) {
                return Response{"-ERR invalid namespace id", nullptr, ""};
            }
            if (cmd.args[1].size() > Storage::kMaxKeySize) {
                return Response{"-ERR query too large", nullptr, ""};
            }
            if (cmd.args[3].size() > max_value_size_) {
                return Response{"-ERR value too large (max-value-size)", nullptr, ""};
            }

            // Parse embedding JSON array.
            vector<float> embedding;
            try {
                auto j = nlohmann::json::parse(cmd.args[2]);
                if (!j.is_array()) {
                    return Response{"-ERR embedding must be a JSON array of floats", nullptr, ""};
                }
                for (const auto& el : j) {
                    embedding.push_back(static_cast<float>(el.get<double>()));
                }
            } catch (const nlohmann::json::exception&) {
                return Response{"-ERR invalid embedding JSON", nullptr, ""};
            }

            int ttl = 0;
            if (cmd.args.size() >= 5) {
                try {
                    ttl = stoi(cmd.args[4]);
                    if (ttl < 0) {
                        ttl = 0;
                    }
                } catch (const exception&) {
                    return Response{"-ERR invalid ttl", nullptr, ""};
                }
            }
            if (ttl == 0) {
                ttl = static_cast<int>(ttl_default_sec_);
            }

            try {
                if (aof_ != nullptr) {
                    aof_->log_semput(namespace_id, cmd.args[1], cmd.args[3],
                                     embedding, ttl);
                }
                storage_->semset(namespace_id, std::move(cmd.args[1]),
                                 std::move(cmd.args[3]), std::move(embedding), ttl);
            } catch (const bad_alloc&) {
                return Response{"-ERR out of memory", nullptr, ""};
            }
            return Response{"+OK", nullptr, ""};
        }

        case protocol::Command::SEM_GET: {
            if (cmd.args.size() < 2) {
                return Response{"-ERR wrong number of arguments for SEM.GET", nullptr, ""};
            }
            uint64_t namespace_id = 0;
            try {
                namespace_id = stoul(cmd.args[0]);
            } catch (const exception&) {
                return Response{"-ERR invalid namespace id", nullptr, ""};
            }

            // 1. Exact fingerprint lookup.
            Blob blob = storage_->semget(namespace_id, cmd.args[1]);
            bool exact_hit = (blob != nullptr);

            // 2. No exact hit: orchestrate embedding worker if configured.
            std::vector<float> query_embedding;
            bool semantic_hit = false;
            double semantic_similarity = 0.0;
            std::string semantic_key;
            std::string semantic_value;
            EntryId semantic_id = 0;

            if (!exact_hit && !embedding_worker_url_.empty()) {
                if (!call_embedding_worker(cmd.args[1], query_embedding)) {
                    nlohmann::json j;
                    j["hit"] = false;
                    j["reason"] = "embedding_worker_unavailable";
                    const string body = j.dump();
                    return Response{"$" + to_string(body.size()) + "\r\n" + body + "\r\n",
                                    nullptr, ""};
                }
            } else if (!exact_hit) {
                // No worker configured: client must send embedding via SEM.SEARCH.
                nlohmann::json j;
                j["hit"] = false;
                j["reason"] = "no_embedding_worker";
                const string body = j.dump();
                return Response{"$" + to_string(body.size()) + "\r\n" + body + "\r\n",
                                nullptr, ""};
            }

            // 3. ANN semantic search (only if no exact hit).
            if (!exact_hit) {
                auto result = storage_->semsearch(query_embedding, namespace_id,
                                                  semantic_threshold_, 5);
                if (result) {
                    semantic_hit = true;
                    semantic_similarity = result->similarity;
                    semantic_key = result->key;
                    semantic_value = result->value;
                    semantic_id = result->id;
                }
            }

            bool hit = exact_hit || semantic_hit;

            // Shadow mode: record the would-be hit but always return MISS.
            if (mode_ == "shadow") {
                nlohmann::json j;
                j["hit"] = false;
                j["mode"] = "shadow";
                if (hit) {
                    j["would_hit"] = true;
                    j["exact_hit"] = exact_hit;
                    j["semantic_hit"] = semantic_hit;
                    if (semantic_hit) {
                        j["id"] = to_string(semantic_id);
                        j["key"] = semantic_key;
                        j["similarity"] = semantic_similarity;
                    }
                } else {
                    j["would_hit"] = false;
                }
                const string body = j.dump();
                return Response{"$" + to_string(body.size()) + "\r\n" + body + "\r\n",
                                nullptr, ""};
            }

            // Active mode: return the actual response.
            if (exact_hit && blob) {
                return Response{"$" + to_string(blob->size()) + "\r\n", blob, "\r\n"};
            }
            if (semantic_hit) {
                nlohmann::json j;
                j["hit"] = true;
                j["id"] = to_string(semantic_id);
                j["key"] = semantic_key;
                j["value"] = semantic_value;
                j["similarity"] = semantic_similarity;
                const string body = j.dump();
                return Response{"$" + to_string(body.size()) + "\r\n" + body + "\r\n",
                                nullptr, ""};
            }
            nlohmann::json j;
            j["hit"] = false;
            const string body = j.dump();
            return Response{"$" + to_string(body.size()) + "\r\n" + body + "\r\n",
                            nullptr, ""};
        }

        case protocol::Command::SEM_SEARCH: {
            if (cmd.args.size() < 2) {
                return Response{"-ERR wrong number of arguments for SEM.SEARCH", nullptr, ""};
            }
            uint64_t namespace_id = 0;
            try {
                namespace_id = stoul(cmd.args[0]);
            } catch (const exception&) {
                return Response{"-ERR invalid namespace id", nullptr, ""};
            }
            // Parse query embedding JSON array.
            vector<float> query_embedding;
            try {
                auto j = nlohmann::json::parse(cmd.args[1]);
                if (!j.is_array()) {
                    return Response{"-ERR embedding must be a JSON array of floats", nullptr, ""};
                }
                for (const auto& el : j) {
                    query_embedding.push_back(static_cast<float>(el.get<double>()));
                }
            } catch (const nlohmann::json::exception&) {
                return Response{"-ERR invalid embedding JSON", nullptr, ""};
            }

            double threshold = semantic_threshold_;
            if (cmd.args.size() >= 3) {
                try {
                    threshold = stod(cmd.args[2]);
                } catch (const exception&) {
                    return Response{"-ERR invalid threshold", nullptr, ""};
                }
            }
            size_t top_k = 5;
            if (cmd.args.size() >= 4) {
                try {
                    top_k = stoul(cmd.args[3]);
                } catch (const exception&) {
                    return Response{"-ERR invalid top_k", nullptr, ""};
                }
            }

            auto result = storage_->semsearch(query_embedding, namespace_id, threshold, top_k);
            if (!result) {
                nlohmann::json j;
                j["hit"] = false;
                const string body = j.dump();
                return Response{"$" + to_string(body.size()) + "\r\n" + body + "\r\n",
                                nullptr, ""};
            }
            nlohmann::json j;
            j["hit"] = true;
            j["id"] = to_string(result->id);
            j["key"] = result->key;
            j["value"] = result->value;
            j["similarity"] = result->similarity;
            const string body = j.dump();
            return Response{"$" + to_string(body.size()) + "\r\n" + body + "\r\n",
                            nullptr, ""};
        }

        case protocol::Command::SEM_DEL: {
            if (cmd.args.size() < 2) {
                return Response{"-ERR wrong number of arguments for SEM.DEL", nullptr, ""};
            }
            uint64_t namespace_id = 0;
            try {
                namespace_id = stoul(cmd.args[0]);
            } catch (const exception&) {
                return Response{"-ERR invalid namespace id", nullptr, ""};
            }
            EntryId id = 0;
            try {
                id = stoull(cmd.args[1]);
            } catch (const exception&) {
                return Response{"-ERR invalid entry id", nullptr, ""};
            }
            const bool removed = storage_->semdel(namespace_id, id);
            if (removed && aof_ != nullptr) {
                // Re-canonicalize the query for AOF logging.
                string q = cmd.args[1];
                string canonical;
                for (char c : q) {
                    if (isspace(static_cast<unsigned char>(c))) {
                        if (!canonical.empty() && canonical.back() != ' ')
                            canonical.push_back(' ');
                    } else {
                        canonical.push_back(c);
                    }
                }
                if (!canonical.empty() && canonical.back() == ' ')
                    canonical.pop_back();
                aof_->log_semdel(namespace_id, canonical);
            }
            return Response{removed ? ":1" : ":0", nullptr, ""};
        }

        case protocol::Command::UNKNOWN:
        default:
            if (cmd.verb.empty()) {
                return Response{"", nullptr, ""};
            }
            return Response{"-ERR unknown command '" + cmd.verb + "'", nullptr, ""};
    }
}

void Server::queue_response(int fd, Response&& response) {
    if (!response.blob) {
        if (response.head.size() < 2 ||
            response.head.compare(response.head.size() - 2, 2, "\r\n") != 0) {
            response.head += "\r\n";
        }
    }

    auto& queue = outbox_[fd];
    queue.push_back(OutChunk{std::move(response.head), 0, std::move(response.blob), 0,
                             std::move(response.tail), 0});
    if (queue.size() > kMaxQueuedChunks) {
        close_client(fd);
        return;
    }
    try_send(fd);
}

void Server::try_send(int fd) {
    auto it = outbox_.find(fd);
    if (it == outbox_.end()) {
        return;
    }
    auto& queue = it->second;

    while (!queue.empty()) {
        OutChunk& chunk = queue.front();

        const char* ptr = nullptr;
        size_t left = 0;
        enum class Part { Head, Blob, Tail, Done } part = Part::Done;

        if (chunk.head_sent < chunk.head.size()) {
            ptr = chunk.head.data() + chunk.head_sent;
            left = chunk.head.size() - chunk.head_sent;
            part = Part::Head;
        } else if (chunk.blob && chunk.blob_sent < chunk.blob->size()) {
            ptr = chunk.blob->data() + chunk.blob_sent;
            left = chunk.blob->size() - chunk.blob_sent;
            part = Part::Blob;
        } else if (chunk.tail_sent < chunk.tail.size()) {
            ptr = chunk.tail.data() + chunk.tail_sent;
            left = chunk.tail.size() - chunk.tail_sent;
            part = Part::Tail;
        } else {
            queue.pop_front();
            continue;
        }

        const ssize_t n = send(fd, ptr, left, COMPAT_MSG_NOSIGNAL);
        if (n > 0) {
            const size_t sent = static_cast<size_t>(n);
            if (part == Part::Head) {
                chunk.head_sent += sent;
            } else if (part == Part::Blob) {
                chunk.blob_sent += sent;
            } else {
                chunk.tail_sent += sent;
            }
            continue;
        }
        if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
            break;
        }
        if (n < 0 && errno == EINTR) {
            continue;
        }
        close_client(fd);
        return;
    }

    if (queue.empty()) {
        outbox_.erase(it);
    }
    update_watch(fd);
}

void Server::update_watch(int fd) {
    uint32_t flags = EPOLLIN;
    auto it = outbox_.find(fd);
    if (it != outbox_.end() && !it->second.empty()) {
        flags |= EPOLLOUT;
    }
    auto watch = watch_flags_.find(fd);
    if (watch != watch_flags_.end() && watch->second == flags) {
        return;
    }
    epoll_event ev{};
    ev.events = flags;
    ev.data.fd = fd;
    epoll_ctl(epoll_fd_, EPOLL_CTL_MOD, fd, &ev);
    watch_flags_[fd] = flags;
}
