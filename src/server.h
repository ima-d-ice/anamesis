#pragma once

#include "embedding_worker_pool.h"
#include "protocol.h"

#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

class Storage;
class AOFLogger;

class Server {
public:
    Server(int port, Storage* storage, AOFLogger* aof,
           std::size_t max_value_size = protocol::Parser::kDefaultMaxBulkLen,
           int ttl_default_sec = 86400, double semantic_threshold = 0.95,
           const std::string& embedding_worker_url = "",
           int shards = 1, const std::string& mode = "shadow",
           int embedding_timeout_ms = 5000, std::size_t embedding_dim = 0,
           int semantic_top_k = 5, double semantic_min_margin = 0.05,
           int embedding_workers = 2, int embedding_queue_capacity = 128);
    ~Server();

    Server(const Server&) = delete;
    Server& operator=(const Server&) = delete;

    void start();
    void stop();

private:
    // A response is either a small owned buffer, or a header + shared blob +
    // trailer. The blob path streams the stored value straight from memory;
    // no copy of a multi-megabyte GET ever lands in the outbox.
    struct Response {
        std::string head;
        std::shared_ptr<const std::string> blob;
        std::string tail;
    };

    struct OutChunk {
        std::string head;
        std::size_t head_sent = 0;
        std::shared_ptr<const std::string> blob;
        std::size_t blob_sent = 0;
        std::string tail;
        std::size_t tail_sent = 0;
    };

    // Pending embedding job for a client that sent SEM.GET and is waiting
    // for the async worker pool to return an embedding.
    struct PendingEmbedding {
        int client_fd = -1;
        uint64_t namespace_id = 0;
        std::string corpus_version;
        std::string query;
    };

    void accept_connections();
    void handle_readable(int fd);
    void queue_response(int fd, Response&& response);
    void try_send(int fd);
    void update_watch(int fd);
    void close_client(int fd);
    Response execute_command(protocol::Command&& cmd, int client_fd);

    // Drain embedding worker completions and send responses.
    void drain_embedding_completions();
    void on_embedding_result(const EmbeddingResult& result);

    // Embedding worker orchestration: async via worker pool.
    bool submit_embedding_job(int client_fd, uint64_t namespace_id,
                              const std::string& corpus_version,
                              const std::string& query);

    int port_;
    Storage* storage_;
    AOFLogger* aof_;
    std::size_t max_value_size_;
    int ttl_default_sec_;
    double semantic_threshold_;
    std::string embedding_worker_url_;
    std::string mode_;
    int embedding_timeout_ms_;
    std::size_t embedding_dim_;
    int semantic_top_k_;
    double semantic_min_margin_;
    int embedding_workers_;
    int embedding_queue_capacity_;

    int listen_fd_ = -1;
    int epoll_fd_ = -1;
    int wake_pipe_[2] = {-1, -1};
    bool running_ = false;

    std::unordered_map<int, protocol::Parser> parsers_;
    std::unordered_map<int, std::deque<OutChunk>> outbox_;
    std::unordered_map<int, std::uint32_t> watch_flags_;
    std::unordered_map<int, PendingEmbedding> pending_embeddings_;

    std::unique_ptr<EmbeddingWorkerPool> embedding_pool_;
};
