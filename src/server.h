#pragma once

#include "protocol.h"

#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <string>
#include <unordered_map>

class Storage;
class AOFLogger;

class Server {
public:
    Server(int port, Storage* storage, AOFLogger* aof,
           std::size_t max_value_size = protocol::Parser::kDefaultMaxBulkLen,
           int ttl_default_sec = 86400, double semantic_threshold = 0.80,
           const std::string& embedding_worker_url = "",
           int shards = 1, const std::string& mode = "shadow");
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

    void accept_connections();
    void handle_readable(int fd);
    void queue_response(int fd, Response&& response);
    void try_send(int fd);
    void update_watch(int fd);
    void close_client(int fd);
    Response execute_command(protocol::Command&& cmd);

    // Embedding worker orchestration: POST text, receive float32 blob.
    bool call_embedding_worker(const std::string& text,
                               std::vector<float>& out_embedding);

    int port_;
    Storage* storage_;
    AOFLogger* aof_;
    std::size_t max_value_size_;
    int ttl_default_sec_;
    double semantic_threshold_;
    std::string embedding_worker_url_;
    int shards_;
    std::string mode_;

    int listen_fd_ = -1;
    int epoll_fd_ = -1;
    int wake_pipe_[2] = {-1, -1};
    bool running_ = false;

    std::unordered_map<int, protocol::Parser> parsers_;
    std::unordered_map<int, std::deque<OutChunk>> outbox_;
    std::unordered_map<int, std::uint32_t> watch_flags_;
};
