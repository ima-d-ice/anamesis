#include "admin.h"

#include "epoll_compat.h"
#include "storage.h"

#include <algorithm>
#include <arpa/inet.h>
#include <cmath>
#include <cerrno>
#include <cstdio>
#include <fcntl.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

using namespace std;

namespace {

constexpr int kMaxEvents = 64;
constexpr size_t kMaxRequest = 4096;

string http_response(int status, const string& status_text,
                     const string& body) {
    string resp = "HTTP/1.1 ";
    resp += to_string(status);
    resp += ' ';
    resp += status_text;
    resp += "\r\n";
    resp += "Content-Type: ";
    resp += status == 200 && body.find("# HELP") != string::npos
                ? "text/plain; version=0.0.4; charset=utf-8"
                : "application/json";
    resp += "\r\n";
    resp += "Content-Length: ";
    resp += to_string(body.size());
    resp += "\r\n";
    resp += "Connection: close\r\n";
    resp += "\r\n";
    resp += body;
    return resp;
}

string prometheus_text(const nlohmann::json& m) {
    string out;
    auto gauge = [&](const string& name, const string& help,
                     const string& val) {
        out += "# HELP " + name + " " + help + "\n";
        out += "# TYPE " + name + " gauge\n";
        out += name + " " + val + "\n";
    };
    auto counter = [&](const string& name, const string& help,
                       const string& val) {
        out += "# HELP " + name + " " + help + "\n";
        out += "# TYPE " + name + " counter\n";
        out += name + " " + val + "\n";
    };
    auto histogram = [&](const string& name, const string& help,
                         const nlohmann::json& buckets) {
        out += "# HELP " + name + " " + help + "\n";
        out += "# TYPE " + name + " histogram\n";
        // Convert exclusive buckets to cumulative Prometheus le buckets.
        double cumulative = 0.0;
        for (auto it = buckets.begin(); it != buckets.end(); ++it) {
            string bucket = it.key();
            double val = it.value().get<double>();
            cumulative += val;
            string le = bucket;
            if (bucket == "inf") le = "+Inf";
            out += name + "_bucket{le=\"" + le + "\"} " +
                   to_string(static_cast<long long>(cumulative)) + "\n";
        }
    };

    counter("semcache_requests_total", "Total requests processed",
            to_string(m.value("requests_total", 0)));
    counter("semcache_hits_total", "Total cache hits",
            to_string(m.value("hits", 0)));
    counter("semcache_misses_total", "Total cache misses",
            to_string(m.value("misses", 0)));
    counter("semcache_semantic_hits_total", "Total semantic cache hits",
            to_string(m.value("semantic_hits", 0)));
    counter("semcache_semantic_misses_total", "Total semantic cache misses",
            to_string(m.value("semantic_misses", 0)));
    counter("semcache_semantic_searches_total", "Total semantic searches",
            to_string(m.value("semantic_searches", 0)));
    counter("semcache_evictions_total", "Total evictions",
            to_string(m.value("evictions", 0)));
    counter("semcache_expired_keys_total", "Total expired keys",
            to_string(m.value("expired_keys", 0)));
    counter("semcache_shadow_would_hit_total", "Total shadow mode would-hits",
            to_string(m.value("shadow_would_hit", 0)));
    counter("semcache_shadow_would_miss_total", "Total shadow mode would-misses",
            to_string(m.value("shadow_would_miss", 0)));
    counter("semcache_shadow_false_hits_total", "Total shadow false hits",
            to_string(m.value("shadow_false_hits", 0)));
    counter("semcache_shadow_missed_hits_total", "Total shadow missed hits",
            to_string(m.value("shadow_missed_hits", 0)));
    counter("semcache_probationary_evictions_total", "Total probationary (LRU) evictions",
            to_string(m.value("probationary_evictions", 0)));
    counter("semcache_uniqueness_evictions_total", "Total uniqueness-based protected-segment evictions",
            to_string(m.value("uniqueness_evictions", 0)));
    counter("semcache_semantic_candidates_total", "Total ANN candidates examined across semantic searches",
            to_string(m.value("candidates_examined_total", 0)));
    counter("semcache_embedding_dim_384_total", "Total entries with 384-dim embeddings",
            to_string(m.value("embedding_dim_384", 0)));
    counter("semcache_embedding_dim_768_total", "Total entries with 768-dim embeddings",
            to_string(m.value("embedding_dim_768", 0)));
    counter("semcache_embedding_dim_1024_total", "Total entries with 1024-dim embeddings",
            to_string(m.value("embedding_dim_1024", 0)));
    counter("semcache_embedding_dim_1536_total", "Total entries with 1536-dim embeddings",
            to_string(m.value("embedding_dim_1536", 0)));
    counter("semcache_embedding_dim_other_total", "Total entries with other embedding dims",
            to_string(m.value("embedding_dim_other", 0)));

    gauge("semcache_memory_bytes", "Current memory bytes used",
          to_string(m.value("memory_bytes", 0)));
    gauge("semcache_memory_limit_bytes", "Memory limit in bytes",
          to_string(m.value("memory_limit", 0)));
    gauge("semcache_keys_total", "Total keys in cache",
          to_string(m.value("total_keys", 0)));
    gauge("semcache_up", "Anamesis is up", "1");
    gauge("semcache_avg_item_bytes", "Average item size in bytes",
          to_string(m.value("avg_item_bytes", 0)));
    gauge("semcache_largest_item_bytes", "Largest item size in bytes",
          to_string(m.value("largest_item_bytes", 0)));
    gauge("semcache_total_namespaces", "Total distinct namespaces",
          to_string(m.value("total_namespaces", 0)));

    // Latency histograms (nanoseconds -> milliseconds for Prometheus)
    nlohmann::json sg_buckets;
    sg_buckets["1ms"] = m.value("semget_latency_bucket_1ms", 0);
    sg_buckets["5ms"] = m.value("semget_latency_bucket_5ms", 0);
    sg_buckets["10ms"] = m.value("semget_latency_bucket_10ms", 0);
    sg_buckets["50ms"] = m.value("semget_latency_bucket_50ms", 0);
    sg_buckets["100ms"] = m.value("semget_latency_bucket_100ms", 0);
    sg_buckets["250ms"] = m.value("semget_latency_bucket_250ms", 0);
    sg_buckets["500ms"] = m.value("semget_latency_bucket_500ms", 0);
    sg_buckets["1s"] = m.value("semget_latency_bucket_1s", 0);
    sg_buckets["+Inf"] = m.value("semget_latency_bucket_inf", 0);
    histogram("semcache_semget_latency_nanoseconds", "SEM.GET latency distribution",
              sg_buckets);
    out += "semcache_semget_latency_nanoseconds_sum " +
           to_string(static_cast<long long>(
               m.value("semget_latency_sum_ns", 0.0))) + "\n";
    out += "semcache_semget_latency_nanoseconds_count " +
           to_string(m.value("semget_latency_count", 0)) + "\n";

    nlohmann::json ss_buckets;
    ss_buckets["1ms"] = m.value("semsearch_latency_bucket_1ms", 0);
    ss_buckets["5ms"] = m.value("semsearch_latency_bucket_5ms", 0);
    ss_buckets["10ms"] = m.value("semsearch_latency_bucket_10ms", 0);
    ss_buckets["50ms"] = m.value("semsearch_latency_bucket_50ms", 0);
    ss_buckets["100ms"] = m.value("semsearch_latency_bucket_100ms", 0);
    ss_buckets["250ms"] = m.value("semsearch_latency_bucket_250ms", 0);
    ss_buckets["500ms"] = m.value("semsearch_latency_bucket_500ms", 0);
    ss_buckets["1s"] = m.value("semsearch_latency_bucket_1s", 0);
    ss_buckets["+Inf"] = m.value("semsearch_latency_bucket_inf", 0);
    histogram("semcache_semsearch_latency_nanoseconds",
              "SEM.SEARCH latency distribution", ss_buckets);
    out += "semcache_semsearch_latency_nanoseconds_sum " +
           to_string(static_cast<long long>(
               m.value("semsearch_latency_sum_ns", 0.0))) + "\n";
    out += "semcache_semsearch_latency_nanoseconds_count " +
           to_string(m.value("semsearch_latency_count", 0)) + "\n";

    nlohmann::json emb_buckets;
    emb_buckets["1ms"] = m.value("embedding_latency_bucket_1ms", 0);
    emb_buckets["5ms"] = m.value("embedding_latency_bucket_5ms", 0);
    emb_buckets["10ms"] = m.value("embedding_latency_bucket_10ms", 0);
    emb_buckets["50ms"] = m.value("embedding_latency_bucket_50ms", 0);
    emb_buckets["100ms"] = m.value("embedding_latency_bucket_100ms", 0);
    emb_buckets["250ms"] = m.value("embedding_latency_bucket_250ms", 0);
    emb_buckets["500ms"] = m.value("embedding_latency_bucket_500ms", 0);
    emb_buckets["1s"] = m.value("embedding_latency_bucket_1s", 0);
    emb_buckets["+Inf"] = m.value("embedding_latency_bucket_inf", 0);
    histogram("semcache_embedding_latency_nanoseconds",
              "Embedding worker call latency distribution", emb_buckets);
    out += "semcache_embedding_latency_nanoseconds_sum " +
           to_string(static_cast<long long>(
               m.value("embedding_latency_sum_ns", 0.0))) + "\n";
    out += "semcache_embedding_latency_nanoseconds_count " +
           to_string(m.value("embedding_latency_count", 0)) + "\n";

    gauge("semcache_semget_latency_avg_ns", "Average SEM.GET latency in nanoseconds",
          to_string(m.value("semget_latency_avg_ns", 0)));
    gauge("semcache_semsearch_latency_avg_ns", "Average SEM.SEARCH latency in nanoseconds",
          to_string(m.value("semsearch_latency_avg_ns", 0)));

    // Hit rate gauge
    double hit_rate = m.value("hit_rate", 0.0);
    gauge("semcache_hit_rate", "Cache hit rate (0-1)",
          to_string(hit_rate));
    double miss_rate = m.value("miss_rate", 0.0);
    gauge("semcache_miss_rate", "Cache miss rate (0-1)",
          to_string(miss_rate));
    gauge("semcache_semantic_match_rate", "Fraction of semantic searches with a match",
          to_string(m.value("semantic_match_rate", 0.0)));
    gauge("semcache_semantic_candidates_per_search",
          "Average ANN candidates examined per semantic search",
          to_string(m.value("semantic_candidates_per_search", 0.0)));

    return out;
}

}  // namespace

AdminServer::AdminServer(int port, Storage* storage)
    : port_(port), storage_(storage) {}

AdminServer::~AdminServer() {
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

void AdminServer::start() {
    listen_fd_ = socket(AF_INET, SOCK_STREAM, 0);
    if (listen_fd_ < 0) {
        perror("admin socket");
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
    if (::bind(listen_fd_, reinterpret_cast<const sockaddr*>(&addr),
               sizeof(addr)) < 0) {
        perror("admin bind");
        close(listen_fd_);
        listen_fd_ = -1;
        return;
    }
    if (listen(listen_fd_, SOMAXCONN) < 0) {
        perror("admin listen");
        close(listen_fd_);
        listen_fd_ = -1;
        return;
    }

    epoll_fd_ = epoll_create1(0);
    if (epoll_fd_ < 0) {
        perror("admin epoll_create1");
        close(listen_fd_);
        listen_fd_ = -1;
        return;
    }

    if (socketpair(AF_UNIX, SOCK_STREAM, 0, wake_pipe_) < 0) {
        perror("admin socketpair");
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
                close_conn(fd);
                continue;
            }
            if ((event_flags & EPOLLIN) != 0) {
                handle_connection(fd);
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

void AdminServer::stop() {
    running_ = false;
    if (wake_pipe_[1] != -1) {
        const char byte = 1;
        (void)write(wake_pipe_[1], &byte, 1);
    }
}

void AdminServer::accept_connections() {
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
    }
}

void AdminServer::close_conn(int fd) {
    epoll_ctl(epoll_fd_, EPOLL_CTL_DEL, fd, nullptr);
    close(fd);
}

void AdminServer::handle_connection(int fd) {
    string request;
    char buf[1024];
    for (;;) {
        const ssize_t n = recv(fd, buf, sizeof(buf), 0);
        if (n > 0) {
            request.append(buf, static_cast<size_t>(n));
            if (request.size() > kMaxRequest) {
                send_all(fd, http_response(400, "Bad Request", "{}"));
                close_conn(fd);
                return;
            }
            if (request.find("\r\n\r\n") != string::npos ||
                request.find("\n\n") != string::npos) {
                break;
            }
            continue;
        }
        if (n == 0) {
            if (request.empty()) {
                close_conn(fd);
                return;
            }
            break;
        }
        if (errno == EAGAIN || errno == EWOULDBLOCK) {
            break;
        }
        if (errno == EINTR) {
            continue;
        }
        close_conn(fd);
        return;
    }

    send_all(fd, handle_request(request));
    close_conn(fd);
}

string AdminServer::handle_request(const string& request) {
    const size_t line_end = request.find("\r\n");
    const size_t first_line_len =
        line_end == string::npos ? request.size() : line_end;
    const string first_line = request.substr(0, first_line_len);

    const size_t p1 = first_line.find(' ');
    const size_t p2 = p1 == string::npos ? string::npos
                                         : first_line.find(' ', p1 + 1);
    if (p1 == string::npos || p2 == string::npos) {
        return http_response(400, "Bad Request", "{}");
    }
    const string method = first_line.substr(0, p1);
    const string path = first_line.substr(p1 + 1, p2 - p1 - 1);

    if (method != "GET") {
        return http_response(405, "Method Not Allowed", "{}");
    }
    if (path == "/metrics") {
        const auto m = storage_->metrics();
        return http_response(200, "OK", prometheus_text(m));
    }
    if (path == "/health") {
        return http_response(200, "OK", "{\"status\":\"ok\"}");
    }
    return http_response(404, "Not Found", "{}");
}

void AdminServer::send_all(int fd, const string& response) {
    size_t offset = 0;
    while (offset < response.size()) {
        const ssize_t n = send(fd, response.data() + offset,
                               response.size() - offset,
                               COMPAT_MSG_NOSIGNAL);
        if (n > 0) {
            offset += static_cast<size_t>(n);
            continue;
        }
        if (n < 0 && errno == EAGAIN) {
            pollfd pfd{fd, POLLOUT, 0};
            (void)poll(&pfd, 1, 100);
            continue;
        }
        if (n < 0 && errno == EINTR) {
            continue;
        }
        return;
    }
}
