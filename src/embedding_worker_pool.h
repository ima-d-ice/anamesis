#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

// A job submitted by the epoll thread for async embedding computation.
struct EmbeddingJob {
    int client_fd = -1;
    uint64_t namespace_id = 0;
    std::string corpus_version;
    std::string query;
    std::string worker_url;
};

// Result of an embedding computation, drained by the epoll thread.
struct EmbeddingResult {
    int client_fd = -1;
    bool success = false;
    std::vector<float> embedding;
    std::string error;
};

// Bounded pool of worker threads that call the llama.cpp embedding HTTP
// endpoint. Workers push results into a thread-safe completion queue and
// wake the epoll thread via a socketpair write. The epoll thread drains
// completions and sends RESP responses in client command order.
//
// No separate coordinator thread: workers own their HTTP connections and
// the epoll thread owns client responses.
class EmbeddingWorkerPool {
public:
    EmbeddingWorkerPool(int num_workers, int queue_capacity, int timeout_ms);
    ~EmbeddingWorkerPool();

    EmbeddingWorkerPool(const EmbeddingWorkerPool&) = delete;
    EmbeddingWorkerPool& operator=(const EmbeddingWorkerPool&) = delete;

    // Submit a job from the epoll thread. Returns false if the queue is full
    // (caller should fail-closed with a miss).
    bool submit(int client_fd, uint64_t namespace_id,
                const std::string& corpus_version,
                const std::string& query,
                const std::string& worker_url);

    // Drain all available completions. Must be called from the epoll thread.
    std::vector<EmbeddingResult> drain_completions();

    // Stop all workers and join. Must be called before destruction.
    void stop();

    // Wait until all submitted jobs have been processed (results pushed to
    // the result queue). Used by tests.
    void wait_for_all();

    // Set the wake pipe write fd (owned by the epoll thread). Workers write
    // here to wake the epoll thread when completions arrive.
    void set_wake_fd(int fd);

private:
    void worker_thread();

    // Job queue (bounded).
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    std::deque<EmbeddingJob> job_queue_;
    int queue_capacity_;

    // Result queue (bounded, drained by epoll thread).
    std::mutex result_mutex_;
    std::condition_variable result_cv_;
    std::deque<EmbeddingResult> result_queue_;
    int result_queue_capacity_;

    // In-flight job counter for waiting on completion.
    std::mutex in_flight_mutex_;
    std::condition_variable in_flight_cv_;
    std::atomic<size_t> in_flight_{0};

    std::vector<std::thread> workers_;
    std::atomic<bool> stop_{false};
    int timeout_ms_;
    int wake_fd_ = -1;  // write end of socketpair to wake epoll thread
};
