#include "aof.h"
#include "admin.h"
#include "server.h"
#include "storage.h"

#include <csignal>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <thread>

using namespace std;

namespace {

Server* g_server = nullptr;
AdminServer* g_admin = nullptr;

void handle_signal(int /*sig*/) {
    if (g_server != nullptr) {
        g_server->stop();
    }
    if (g_admin != nullptr) {
        g_admin->stop();
    }
}

void print_usage(const char* prog) {
    cerr << "usage: " << prog
         << " [--port PORT] [--admin-port PORT] [--memory-limit MB] "
            "[--max-value-size MB] [--aof-file FILE] [--no-aof] "
            "[--policy slru|lru|lfu|sieve] [--shards N] "
            "[--embedding-worker-url URL] [--embedding-timeout-ms MS] "
            "[--embedding-dim N] [--ttl-default-sec SEC] "
            "[--semantic-threshold FLOAT] [--semantic-top-k N] "
            "[--semantic-eviction-weight FLOAT] "
            "[--policy slru|lru|lfu|sieve|semantic_slru] [--mode shadow|active]\n";
}

}  // namespace

int main(int argc, char** argv) {  // NOLINT(bugprone-exception-escape): outermost frame; std::exceptions are caught below, anything else terminating here is the defined behavior
    int port = 6379;
    int admin_port = 8080;
    size_t memory_mb = 1024;
    size_t max_value_mb = 512;
    string aof_file = "anamesis.aof";
    bool use_aof = true;
    string policy = "slru";
    int shards = 0;
    string embedding_worker_url;
    int ttl_default_sec = 86400;
    double semantic_threshold = 0.95;
    double semantic_min_margin = 0.05;
    int semantic_top_k = 5;
    double semantic_eviction_weight = 0.5;
    string mode = "shadow";
    int embedding_timeout_ms = 5000;
    size_t embedding_dim = 0;

    for (int i = 1; i < argc; ++i) {
        const string arg = argv[i];
        auto next_value = [&](const string& name) -> string {
            if (i + 1 >= argc) {
                cerr << "missing value for " << name << "\n";
                exit(1);
            }
            return argv[++i];
        };
        try {
            if (arg == "--port") {
                port = stoi(next_value(arg));
            } else if (arg == "--admin-port") {
                admin_port = stoi(next_value(arg));
            } else if (arg == "--memory-limit") {
                memory_mb = static_cast<size_t>(stoul(next_value(arg)));
            } else if (arg == "--max-value-size") {
                max_value_mb = static_cast<size_t>(stoul(next_value(arg)));
            } else if (arg == "--aof-file") {
                aof_file = next_value(arg);
            } else if (arg == "--no-aof") {
                use_aof = false;
            } else if (arg == "--policy") {
                policy = next_value(arg);
            } else if (arg == "--shards") {
                shards = static_cast<int>(stoi(next_value(arg)));
            } else if (arg == "--embedding-worker-url") {
                embedding_worker_url = next_value(arg);
            } else if (arg == "--ttl-default-sec") {
                ttl_default_sec = static_cast<int>(stol(next_value(arg)));
            } else if (arg == "--semantic-threshold") {
                semantic_threshold = stod(next_value(arg));
            } else if (arg == "--embedding-timeout-ms") {
                embedding_timeout_ms = static_cast<int>(stoi(next_value(arg)));
            } else if (arg == "--embedding-dim") {
                embedding_dim = static_cast<size_t>(stoul(next_value(arg)));
            } else if (arg == "--semantic-top-k") {
                semantic_top_k = static_cast<int>(stoi(next_value(arg)));
            } else if (arg == "--semantic-eviction-weight") {
                semantic_eviction_weight = stod(next_value(arg));
            } else if (arg == "--semantic-min-margin") {
                semantic_min_margin = stod(next_value(arg));
            } else if (arg == "--mode") {
                mode = next_value(arg);
                if (mode != "shadow" && mode != "active") {
                    cerr << "invalid mode: " << mode << " (use shadow or active)" << "\n";
                    return 1;
                }
            } else if (arg == "--help" || arg == "-h") {
                print_usage(argv[0]);
                return 0;
            } else {
                cerr << "unknown option: " << arg << "\n";
                print_usage(argv[0]);
                return 1;
            }
        } catch (const exception&) {
            cerr << "invalid numeric value for " << arg << "\n";
            return 1;
        }
    }

    if (memory_mb == 0 || max_value_mb == 0) {
        cerr << "--memory-limit and --max-value-size must be at least 1MB\n";
        return 1;
    }
    if (embedding_timeout_ms <= 0) {
        cerr << "--embedding-timeout-ms must be positive\n";
        return 1;
    }
    if (semantic_top_k < 1) {
        cerr << "--semantic-top-k must be >= 1\n";
        return 1;
    }
    if (semantic_eviction_weight < 0.0 || semantic_eviction_weight > 1.0) {
        cerr << "--semantic-eviction-weight must be in [0,1]\n";
        return 1;
    }
    if (semantic_min_margin < 0.0 || semantic_min_margin > 1.0) {
        cerr << "--semantic-min-margin must be in [0,1]\n";
        return 1;
    }
    if (max_value_mb > memory_mb) {
        cerr << "warning: --max-value-size (" << max_value_mb
             << "MB) exceeds --memory-limit (" << memory_mb
             << "MB); values larger than the cache will be rejected\n";
    }

    const size_t max_value_bytes = max_value_mb * 1024 * 1024;

    // Resolve the shard count before constructing Storage so both Storage and
    // Server agree on the same value (default: hardware concurrency).
    if (shards <= 0) {
        shards = static_cast<int>(thread::hardware_concurrency());
        if (shards <= 0) {
            shards = 1;
        }
    }

    try {
        Storage storage(memory_mb * 1024 * 1024, max_value_bytes, policy.c_str(),
                      shards, semantic_eviction_weight);
        std::unique_ptr<AOFLogger> aof;
        if (use_aof) {
            aof = std::make_unique<AOFLogger>(aof_file);
            aof->replay(&storage);
        }

        Server server(port, &storage, aof.get(), max_value_bytes,
                      ttl_default_sec, semantic_threshold,
                      embedding_worker_url, shards, mode, embedding_timeout_ms,
                      embedding_dim, semantic_top_k, semantic_min_margin);
        AdminServer admin(admin_port, &storage);

        struct sigaction sa {};
        sa.sa_handler = handle_signal;
        sigemptyset(&sa.sa_mask);
        sigaction(SIGINT, &sa, nullptr);
        sigaction(SIGTERM, &sa, nullptr);

        g_admin = &admin;
        thread admin_thread([&admin] { admin.start(); });

        g_server = &server;
        cout << "Anamesis listening on port " << port << ", admin on port "
             << admin_port << ", memory " << memory_mb << "MB, max value "
             << max_value_mb << "MB, policy: " << policy
             << ", mode: " << mode << '\n'
             << flush;
        server.start();
        g_server = nullptr;

        admin.stop();
        admin_thread.join();
    } catch (const exception& e) {
        cerr << "fatal: " << e.what() << "\n";
        return 1;
    }
    g_admin = nullptr;

    return 0;
}
