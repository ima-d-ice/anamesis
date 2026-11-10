#include "embedding_worker_pool.h"

#include "embedding_client.h"

#include <arpa/inet.h>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <fcntl.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

using namespace std;

EmbeddingWorkerPool::EmbeddingWorkerPool(int num_workers, int queue_capacity,
                                         int timeout_ms)
    : queue_capacity_(queue_capacity),
      result_queue_capacity_(queue_capacity),
      timeout_ms_(timeout_ms),
      in_flight_(0) {
    // Create a socketpair to wake the epoll thread when completions arrive.
    int sp[2];
    if (socketpair(AF_UNIX, SOCK_STREAM, 0, sp) != 0) {
        return;
    }
    // sp[0] is read by epoll thread, sp[1] is written by workers.
    int flags = fcntl(sp[0], F_GETFL, 0);
    fcntl(sp[0], F_SETFL, flags | O_NONBLOCK);
    wake_fd_ = sp[1];

    for (int i = 0; i < num_workers; ++i) {
        workers_.emplace_back(&EmbeddingWorkerPool::worker_thread, this);
    }
}

EmbeddingWorkerPool::~EmbeddingWorkerPool() {
    stop();
    if (wake_fd_ != -1) {
        close(wake_fd_);
        wake_fd_ = -1;
    }
}

void EmbeddingWorkerPool::set_wake_fd(int fd) {
    wake_fd_ = fd;
}

void EmbeddingWorkerPool::stop() {
    if (stop_.exchange(true)) {
        return;
    }
    {
        lock_guard<mutex> lk(queue_mutex_);
        queue_cv_.notify_all();
    }
    for (auto& t : workers_) {
        if (t.joinable()) {
            t.join();
        }
    }
}

void EmbeddingWorkerPool::wait_for_all() {
    unique_lock<mutex> lk(in_flight_mutex_);
    in_flight_cv_.wait(lk, [this] { return in_flight_.load() == 0; });
}

bool EmbeddingWorkerPool::submit(int client_fd, uint64_t namespace_id,
                                 const string& corpus_version,
                                 const string& query,
                                 const string& worker_url) {
    {
        lock_guard<mutex> lk(queue_mutex_);
        if (static_cast<int>(job_queue_.size()) >= queue_capacity_) {
            return false;  // queue full → caller fails closed
        }
        job_queue_.emplace_back(EmbeddingJob{client_fd, namespace_id,
                                              corpus_version, query,
                                              worker_url});
    }
    in_flight_.fetch_add(1, memory_order_relaxed);
    queue_cv_.notify_one();
    return true;
}

vector<EmbeddingResult> EmbeddingWorkerPool::drain_completions() {
    vector<EmbeddingResult> results;
    {
        lock_guard<mutex> lk(result_mutex_);
        results.insert(results.end(), result_queue_.begin(), result_queue_.end());
        result_queue_.clear();
    }
    return results;
}

void EmbeddingWorkerPool::worker_thread() {
    while (true) {
        EmbeddingJob job;
        {
            unique_lock<mutex> lk(queue_mutex_);
            queue_cv_.wait(lk, [this] {
                return stop_.load() || !job_queue_.empty();
            });
            if (stop_.load() && job_queue_.empty()) {
                return;
            }
            job = job_queue_.front();
            job_queue_.pop_front();
        }

        // Perform the HTTP embedding request.
        vector<float> embedding;
        string err;
        bool ok = embedding_client::embed_text(
            job.worker_url, job.query, timeout_ms_,
            embedding, err, 0);

        EmbeddingResult result;
        result.client_fd = job.client_fd;
        if (ok) {
            result.success = true;
            result.embedding = move(embedding);
        } else {
            result.success = false;
            result.error = move(err);
        }

        {
            unique_lock<mutex> lk(result_mutex_);
            result_cv_.wait(lk, [this] {
                return stop_.load() ||
                       result_queue_.size() < result_queue_capacity_;
            });
            if (stop_.load()) {
                return;
            }
            result_queue_.push_back(move(result));
        }
        result_cv_.notify_one();

        in_flight_.fetch_sub(1, memory_order_relaxed);
        in_flight_cv_.notify_one();

        // Wake the epoll thread.
        if (wake_fd_ != -1) {
            uint8_t byte = 1;
            while (true) {
                ssize_t n = write(wake_fd_, &byte, 1);
                if (n > 0) {
                    break;
                }
                if (errno != EINTR && errno != EAGAIN && errno != EWOULDBLOCK) {
                    break;
                }
            }
        }
    }
}
