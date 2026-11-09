#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace protocol {

struct Command {
    enum Type : std::uint8_t {
        SET,
        GET,
        DEL,
        METRICS,
        PING,
        SELECT,
        SEM_PUT,
        SEM_GET,
        SEM_SEARCH,
        SEM_DEL,
        SEM_FLUSH,
        SEM_INFO,
        SEM_STATS,
        UNKNOWN
    };
    Type type = UNKNOWN;
    std::string verb;
    std::vector<std::string> args;
};

class Parser {
public:
    static constexpr std::size_t kDefaultMaxBulkLen = std::size_t{512} * 1024 * 1024;
    static constexpr long kMaxArgs = 1024;

    explicit Parser(std::size_t max_bulk_len = kDefaultMaxBulkLen);

    enum class Status { NeedMore, Ready, Error };

    Status consume(const char*& data, std::size_t& len, Command& out);

    const std::string& error() const { return error_; }
    std::size_t max_bulk_len() const { return max_bulk_len_; }

private:
    enum class State : std::uint8_t {
        ArrayLen,
        BulkLen,
        Payload,
        Trailer,
        TrailerLf,
        Ready,
        Failed
    };

    Status fail(std::string message);

    std::size_t max_bulk_len_;
    State state_ = State::ArrayLen;
    std::string header_;
    std::string error_;
    long argc_ = 0;
    long arg_index_ = 0;
    std::size_t bulk_len_ = 0;
    std::size_t bulk_read_ = 0;
    std::vector<std::string> args_;
};

}  // namespace protocol
