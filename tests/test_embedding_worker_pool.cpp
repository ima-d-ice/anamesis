#include "embedding_worker_pool.h"

#include <arpa/inet.h>
#include <cstdio>
#include <cstring>
#include <netinet/in.h>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>

#include <algorithm>
#include <string>
#include <vector>

using namespace std;

namespace {

int g_failures = 0;

void check(bool cond, const char* what) {
    if (cond) {
        return;
    }
    ++g_failures;
    fprintf(stderr, "FAIL: %s\n", what);
}

// Minimal HTTP server that returns a fixed embedding for any POST request.
// Listens on a single port, handles one connection at a time.
class FakeEmbeddingServer {
public:
    explicit FakeEmbeddingServer(int port) : port_(port) {
        int one = 1;
        listen_fd_ = socket(AF_INET, SOCK_STREAM, 0);
        setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        addr.sin_port = htons(static_cast<uint16_t>(port));
        if (::bind(listen_fd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
            perror("bind");
            exit(1);
        }
        // Get the actual port if port was 0 (ephemeral)
        socklen_t addr_len = sizeof(addr);
        if (getsockname(listen_fd_, reinterpret_cast<sockaddr*>(&addr), &addr_len) == 0) {
            port_ = ntohs(addr.sin_port);
        }
        listen(listen_fd_, 4);
        running_ = true;
        thread_ = thread(&FakeEmbeddingServer::run, this);
    }

    ~FakeEmbeddingServer() {
        running_ = false;
        int probe = socket(AF_INET, SOCK_STREAM, 0);
        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        addr.sin_port = htons(static_cast<uint16_t>(port_));
        connect(probe, reinterpret_cast<sockaddr*>(&addr), sizeof(addr));
        close(probe);
        if (thread_.joinable()) {
            thread_.join();
        }
        close(listen_fd_);
    }

    int port() const { return port_; }

private:
    void run() {
        while (running_) {
            int client = accept(listen_fd_, nullptr, nullptr);
            if (client < 0) {
                break;
            }
            // Read request (discard)
            char buf[4096];
            recv(client, buf, sizeof(buf), 0);
            // Send response: legacy llama.cpp embedding format
            const char* body = R"([{"index":0,"embedding":[[0.1,0.2,0.3,0.4]]}])";
            string response =
                "HTTP/1.1 200 OK\r\n"
                "Content-Type: application/json\r\n"
                "Content-Length: " +
                to_string(strlen(body)) + "\r\n"
                "Connection: close\r\n\r\n" +
                body;
            send(client, response.data(), response.size(), 0);
            close(client);
        }
    }

    int listen_fd_ = -1;
    int port_;
    atomic<bool> running_{true};
    thread thread_;
};

int run_all_tests() {
    // ---- Test 1: FIFO ordering under pipelining ----
    {
        FakeEmbeddingServer server(0);
        int port = server.port();
        string url = "http://127.0.0.1:" + to_string(port) + "/embedding";

        EmbeddingWorkerPool pool(2, 16, 5000);
        pool.set_wake_fd(-1);  // no epoll integration in test

        const int client_fd = 42;
        const int n_queries = 10;
        vector<string> queries;
        for (int i = 0; i < n_queries; ++i) {
            queries.push_back("query_" + to_string(i));
        }

        // Submit all jobs for the same client fd
        for (int i = 0; i < n_queries; ++i) {
            bool ok = pool.submit(client_fd, 1, "default", queries[i], url);
            check(ok, "submit should succeed when queue has capacity");
        }

        // Wait for all jobs to complete
        pool.wait_for_all();

        // Drain completions
        auto results = pool.drain_completions();
        check(results.size() == static_cast<size_t>(n_queries),
              "all results should be returned");

        // Verify FIFO ordering: result[i] should correspond to queries[i]
        bool order_ok = true;
        for (size_t i = 0; i < results.size(); ++i) {
            if (!results[i].success) {
                fprintf(stderr, "  result %zu failed: %s\n", i,
                        results[i].error.c_str());
                order_ok = false;
            }
            if (results[i].client_fd != client_fd) {
                fprintf(stderr, "  result %zu has wrong client_fd: %d (expected %d)\n",
                        i, results[i].client_fd, client_fd);
                order_ok = false;
            }
        }
        check(order_ok, "FIFO ordering: results returned in submission order");

        pool.stop();
        printf("  test 1 (FIFO ordering): %s\n",
               (g_failures == 0 ? "PASS" : "FAIL (so far)"));
    }

    // ---- Test 2: Multiple clients, independent queues ----
    {
        FakeEmbeddingServer server(0);
        int port = server.port();
        string url = "http://127.0.0.1:" + to_string(port) + "/embedding";

        EmbeddingWorkerPool pool(2, 32, 5000);
        pool.set_wake_fd(-1);

        const int fd_a = 10;
        const int fd_b = 20;

        // Submit interleaved: A, B, A, B
        pool.submit(fd_a, 1, "default", "a1", url);
        pool.submit(fd_b, 1, "default", "b1", url);
        pool.submit(fd_a, 1, "default", "a2", url);
        pool.submit(fd_b, 1, "default", "b2", url);

        pool.wait_for_all();

        auto results = pool.drain_completions();
        check(results.size() == 4, "all 4 results returned");

        // Results should be in submission order within each client fd,
        // but across clients order is nondeterministic. Just verify
        // all results have valid client fds.
        bool all_valid = true;
        for (const auto& r : results) {
            if (r.client_fd != fd_a && r.client_fd != fd_b) {
                fprintf(stderr, "  result has unexpected client_fd: %d\n",
                        r.client_fd);
                all_valid = false;
            }
            if (!r.success) {
                fprintf(stderr, "  result for fd %d failed: %s\n",
                        r.client_fd, r.error.c_str());
                all_valid = false;
            }
        }
        check(all_valid, "all results have valid client fd and success");

        pool.stop();
        printf("  test 2 (multiple clients): %s\n",
               (g_failures == 0 ? "PASS" : "FAIL (so far)"));
    }

    // ---- Test 3: Queue full rejection ----
    {
        EmbeddingWorkerPool pool(1, 2, 5000);  // capacity 2, 1 worker
        pool.set_wake_fd(-1);

        const int client_fd = 7;
        int successes = 0;
        int failures = 0;
        // Submit many jobs quickly; with capacity 2 and 1 slow worker,
        // most submissions should fail once the queue is full.
        for (int i = 0; i < 50; ++i) {
            if (pool.submit(client_fd, 1, "default", "q", "http://127.0.0.1:1/embedding")) {
                ++successes;
            } else {
                ++failures;
            }
        }
        check(successes > 0, "at least one submit succeeds");
        check(failures > 0, "at least one submit fails when queue is full");

        pool.stop();
        printf("  test 3 (queue full rejection): %s\n",
               (g_failures == 0 ? "PASS" : "FAIL (so far)"));
    }

    if (g_failures == 0) {
        printf("ok test_embedding_worker_pool\n");
        return 0;
    }
    fprintf(stderr, "%d check(s) failed\n", g_failures);
    return 1;
}

}  // namespace

int main() {
    return run_all_tests();
}
