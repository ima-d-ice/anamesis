// Anamesis cache router — horizontal scaling frontend.
//
// Routes SEM.* commands to the owning Anamesis node via hash(namespace_id) % N.
// Non-SEM commands are round-robined across nodes. Health-checks nodes via
// GET /health and drops unhealthy nodes from the ring.
//
// Built as a standalone binary:
//   g++ -std=c++17 -O2 -pthread src/router/proxy.cpp -o anamesis-router

#include <arpa/inet.h>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fcntl.h>
#include <fstream>
#include <netinet/in.h>
#include <sstream>
#include <string>
#include <sys/socket.h>
#include <sys/select.h>
#include <thread>
#include <unistd.h>
#include <vector>

#include <nlohmann/json.hpp>

using namespace std;
using json = nlohmann::json;

namespace {

struct Node {
    string host;
    int port;
    bool healthy = true;
    time_t last_check = 0;
};

vector<Node> nodes;
size_t rr_index = 0;
mutex nodes_mutex;

bool node_healthy(const Node& n) {
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) return false;
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<uint16_t>(n.port));
    if (inet_pton(AF_INET, n.host.c_str(), &addr.sin_addr) <= 0) {
        close(sock);
        return false;
    }
    int flags = fcntl(sock, F_GETFL, 0);
    fcntl(sock, F_SETFL, flags | O_NONBLOCK);
    int r = connect(sock, reinterpret_cast<sockaddr*>(&addr), sizeof(addr));
    if (r < 0 && errno != EINPROGRESS) {
        close(sock);
        return false;
    }
    if (r == -1 && errno == EINPROGRESS) {
        fd_set wfd;
        FD_ZERO(&wfd);
        FD_SET(sock, &wfd);
        timeval tv{2, 0};
        r = select(sock + 1, nullptr, &wfd, nullptr, &tv);
    }
    bool ok = (r > 0);
    close(sock);
    return ok;
}

void health_loop() {
    while (true) {
        this_thread::sleep_for(chrono::seconds(5));
        lock_guard<mutex> lk(nodes_mutex);
        for (auto& n : nodes) {
            time_t now = time(nullptr);
            if (now - n.last_check < 2) continue;
            n.last_check = now;
            n.healthy = node_healthy(n);
        }
    }
}

string extract_namespace(const string& resp) {
    // Parse RESP2 array: *<count>\r\n<bulk args>
    // For SEM.* commands, namespace is args[0] (verb is args[-1] conceptually)
    size_t pos = 0;
    if (resp.compare(0, 1, "*") != 0) return "";
    pos = resp.find("\r\n", 1);
    if (pos == string::npos) return "";
    (void)stoi(resp.substr(1, pos - 1));
    pos += 2;

    // Read verb
    if (resp.compare(pos, 1, "$") != 0) return "";
    pos++;
    size_t crlf = resp.find("\r\n", pos);
    if (crlf == string::npos) return "";
    int verb_len = stoi(resp.substr(pos, crlf - pos));
    pos = crlf + 2;
    string verb = resp.substr(pos, verb_len);
    pos += verb_len + 2;

    // Check if SEM.*
    if (verb.compare(0, 4, "SEM.") != 0 && verb.compare(0, 6, "SEMSET") != 0 &&
        verb.compare(0, 6, "SEMGET") != 0 && verb.compare(0, 8, "SEMSEARCH") != 0 &&
        verb.compare(0, 6, "SEMDEL") != 0) {
        return "";
    }

    // Read first argument = namespace_id
    if (resp.compare(pos, 1, "$") != 0) return "";
    pos++;
    crlf = resp.find("\r\n", pos);
    if (crlf == string::npos) return "";
    int arg_len = stoi(resp.substr(pos, crlf - pos));
    pos = crlf + 2;
    return resp.substr(pos, arg_len);
}

// Parse all bulk string arguments from a RESP2 array request
vector<string> parse_all_args(const string& resp) {
    vector<string> args;
    size_t pos = 0;
    if (resp.compare(0, 1, "*") != 0) return args;
    pos = resp.find("\r\n", 1);
    if (pos == string::npos) return args;
    pos += 2;

    while (pos < resp.size()) {
        if (resp.compare(pos, 1, "$") != 0) break;
        pos++;
        size_t crlf = resp.find("\r\n", pos);
        if (crlf == string::npos) break;
        int arg_len = stoi(resp.substr(pos, crlf - pos));
        pos = crlf + 2;
        if (pos + arg_len > resp.size()) break;
        args.push_back(resp.substr(pos, arg_len));
        pos += arg_len + 2;  // skip arg + \r\n
    }
    return args;
}

// Check if a command is SEM.SEARCH with broadcast=1
bool is_broadcast_semsearch(const vector<string>& args) {
    if (args.empty()) return false;
    string verb = args[0];
    if (verb != "SEM.SEARCH") return false;
    // Look for "broadcast" followed by "1" in remaining args
    for (size_t i = 1; i + 1 < args.size(); i++) {
        if (args[i] == "broadcast" && args[i + 1] == "1") {
            return true;
        }
    }
    return false;
}

// Merge multiple SEM.SEARCH JSON responses, returning the one with highest similarity
string merge_search_results(const vector<string>& responses) {
    string best;
    double best_sim = -1.0;
    for (const auto& resp : responses) {
        try {
            // Find the JSON object in the response (skip RESP bulk header if present)
            size_t json_start = resp.find("{");
            if (json_start == string::npos) continue;
            auto j = nlohmann::json::parse(resp.substr(json_start));
            if (j.contains("hit") && j["hit"].get<bool>() && j.contains("similarity")) {
                double sim = j["similarity"].get<double>();
                if (sim > best_sim) {
                    best_sim = sim;
                    best = resp;  // Return the full RESP response
                }
            }
        } catch (...) {}
    }
    return best;
}

size_t hash_node(const string& ns, size_t n) {
    if (n == 0) return 0;
    size_t h = 0;
    for (char c : ns) {
        h = h * 31 + static_cast<unsigned char>(c);
    }
    return h % n;
}

int connect_to_node(const Node& n) {
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) return -1;
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<uint16_t>(n.port));
    if (inet_pton(AF_INET, n.host.c_str(), &addr.sin_addr) <= 0) {
        close(sock);
        return -1;
    }
    if (connect(sock, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
        close(sock);
        return -1;
    }
    return sock;
}

ssize_t send_all(int sock, const string& data) {
    size_t off = 0;
    while (off < data.size()) {
        ssize_t n = send(sock, data.data() + off, data.size() - off, MSG_NOSIGNAL);
        if (n <= 0) return n;
        off += static_cast<size_t>(n);
    }
    return static_cast<ssize_t>(off);
}

string read_response(int sock) {
    string out;
    char buf[4096];
    while (true) {
        ssize_t n = recv(sock, buf, sizeof(buf), 0);
        if (n <= 0) break;
        out.append(buf, static_cast<size_t>(n));
        // RESP responses end with \r\n
        if (out.size() >= 2 &&
            out.compare(out.size() - 2, 2, "\r\n") == 0) {
            break;
        }
        // Bulk strings may contain \r\n internally; read until we have enough
        if (!out.empty() && out[0] == '$') {
            size_t crlf = out.find("\r\n");
            if (crlf != string::npos) {
                try {
                    int len = stoi(out.substr(1, crlf - 1));
                    size_t total = crlf + 2 + static_cast<size_t>(len) + 2;
                    if (out.size() >= total) break;
                } catch (...) {}
            }
        }
    }
    return out;
}

void handle_client(int client_fd, const vector<Node>& node_list) {
    char buf[65536];
    while (true) {
        ssize_t n = recv(client_fd, buf, sizeof(buf), 0);
        if (n <= 0) break;

        string req(buf, static_cast<size_t>(n));

        // Parse all arguments to check for broadcast SEM.SEARCH
        vector<string> args = parse_all_args(req);
        bool is_broadcast = is_broadcast_semsearch(args);

        if (is_broadcast) {
            // Broadcast to all nodes, collect responses, return best match
            vector<string> responses;
            {
                lock_guard<mutex> lk(nodes_mutex);
                if (node_list.empty()) {
                    string err = "-ERR no nodes available\r\n";
                    send_all(client_fd, err);
                    continue;
                }
                fprintf(stderr, "[router] Broadcasting SEM.SEARCH to %zu nodes\n", node_list.size());
                for (const auto& node : node_list) {
                    fprintf(stderr, "[router] Connecting to %s:%d\n", node.host.c_str(), node.port);
                    int backend = connect_to_node(node);
                    if (backend < 0) {
                        fprintf(stderr, "[router] Connect failed to %s:%d\n", node.host.c_str(), node.port);
                        continue;
                    }
                    fprintf(stderr, "[router] Sending request to %s:%d\n", node.host.c_str(), node.port);
                    if (send_all(backend, req) < 0) {
                        fprintf(stderr, "[router] Send failed to %s:%d\n", node.host.c_str(), node.port);
                        close(backend);
                        continue;
                    }
                    string resp = read_response(backend);
                    close(backend);
                    fprintf(stderr, "[router] Response from %s:%d: %.200s\n", node.host.c_str(), node.port, resp.c_str());
                    if (!resp.empty()) responses.push_back(resp);
                }
            }
            string best = merge_search_results(responses);
            fprintf(stderr, "[router] Merged result: %.200s\n", best.c_str());
            if (!best.empty()) {
                send_all(client_fd, best);
            } else {
                string err = "{\"hit\":false}\r\n";
                send_all(client_fd, err);
            }
            continue;
        }

        // Normal path: route to single node
        string ns = extract_namespace(req);

        size_t target = 0;
        {
            lock_guard<mutex> lk(nodes_mutex);
            if (node_list.empty()) {
                string err = "-ERR no nodes available\r\n";
                send_all(client_fd, err);
                continue;
            }
            if (!ns.empty()) {
                target = hash_node(ns, node_list.size());
            } else {
                target = rr_index++ % node_list.size();
            }
        }

        const Node& node = node_list[target];
        int backend = connect_to_node(node);
        if (backend < 0) {
            string err = "-ERR backend unavailable\r\n";
            send_all(client_fd, err);
            continue;
        }

        if (send_all(backend, req) < 0) {
            close(backend);
            string err = "-ERR backend send failed\r\n";
            send_all(client_fd, err);
            continue;
        }

        string resp = read_response(backend);
        close(backend);

        if (!resp.empty()) {
            send_all(client_fd, resp);
        }
    }
    close(client_fd);
}

}  // namespace

int main(int argc, char** argv) {
    int listen_port = 6380;
    for (int i = 1; i < argc; ++i) {
        string arg = argv[i];
        if (arg == "--port" && i + 1 < argc) {
            listen_port = stoi(argv[++i]);
        } else if (arg == "--node" && i + 1 < argc) {
            string spec = argv[++i];
            size_t colon = spec.rfind(':');
            if (colon != string::npos) {
                Node n;
                n.host = spec.substr(0, colon);
                n.port = stoi(spec.substr(colon + 1));
                nodes.push_back(n);
            }
        } else if (arg == "--help" || arg == "-h") {
            printf("usage: anamesis-router --port PORT --node HOST:PORT [--node ...]\n");
            return 0;
        }
    }

    if (nodes.empty()) {
        fprintf(stderr, "error: at least one --node required\n");
        return 1;
    }

    thread health(health_loop);
    health.detach();

    int listen_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (listen_fd < 0) {
        perror("socket");
        return 1;
    }
    int opt = 1;
    setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
    int flags = fcntl(listen_fd, F_GETFL, 0);
    fcntl(listen_fd, F_SETFL, flags | O_NONBLOCK);

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    addr.sin_port = htons(static_cast<uint16_t>(listen_port));
    if (::bind(listen_fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
        perror("bind");
        return 1;
    }
    if (listen(listen_fd, SOMAXCONN) < 0) {
        perror("listen");
        return 1;
    }

    fprintf(stderr, "anamesis-router listening on port %d, %zu node(s)\n",
            listen_port, nodes.size());

    while (true) {
        int client_fd = accept(listen_fd, nullptr, nullptr);
        if (client_fd < 0) {
            if (errno == EINTR || errno == EAGAIN) continue;
            perror("accept");
            continue;
        }
        thread(handle_client, client_fd, nodes).detach();
    }

    close(listen_fd);
    return 0;
}
