#pragma once

#include "storage.h"

#include <cstdint>
#include <string>
#include <vector>

class AOFLogger {
public:
    explicit AOFLogger(const std::string& filename,
                       int flush_interval_sec = 60);
    ~AOFLogger();

    void log_set(const std::string& key, const std::string& value, int ttl);
    void log_del(const std::string& key);
    void log_semput(std::uint64_t namespace_id, const std::string& query,
                    const std::string& response, const std::vector<float>& embedding,
                    int ttl);
    void log_semdel(std::uint64_t namespace_id, const std::string& canonical_query);
    void replay(Storage* storage);

    // Force a flush of any buffered lines to disk.
    void flush();

private:
    void append_line(const std::string& line);
    void flush_loop();

    std::string filename_;
    int flush_interval_sec_;
    std::FILE* file_ = nullptr;
};
