#pragma once

#include <cstddef>
#include <cstdint>

using EntryId = std::uint64_t;

class EvictionPolicy {
public:
    virtual ~EvictionPolicy() = default;

    virtual void touch(EntryId id) = 0;
    virtual void add(EntryId id, std::size_t size) = 0;
    virtual void remove(EntryId id) = 0;
    virtual EntryId evict() = 0;
    virtual std::size_t memory_used() const = 0;
};
