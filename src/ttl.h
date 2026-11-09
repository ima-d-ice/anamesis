#pragma once

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

class TTLManager {
public:
    using OnExpire = std::function<void(std::uint64_t)>;

    explicit TTLManager(OnExpire on_expire);
    ~TTLManager();

    TTLManager(const TTLManager&) = delete;
    TTLManager& operator=(const TTLManager&) = delete;

    void set_ttl(std::uint64_t id, std::chrono::milliseconds duration);
    bool is_expired(std::uint64_t id);
    void erase(std::uint64_t id);
    void clear();
    std::vector<std::uint64_t> expired_keys();

private:
    void sweep_loop();

    OnExpire on_expire_;
    std::mutex lock_;
    std::unordered_map<std::uint64_t, std::chrono::steady_clock::time_point> expiries_;
    std::thread thread_;
    bool stop_ = false;
    std::condition_variable cv_;
};
