#pragma once

#include "policy.h"

#include <cstddef>
#include <cstdint>
#include <list>
#include <unordered_map>

class SLRU : public EvictionPolicy {
public:
    SLRU(std::size_t max_memory_bytes, double protected_ratio = 0.80);

    void touch(EntryId id) override;
    void add(EntryId id, std::size_t size) override;
    void remove(EntryId id) override;
    EntryId evict() override;
    std::size_t memory_used() const override;

    std::size_t probationary_bytes() const { return probationary_bytes_; }
    std::size_t protected_bytes() const { return protected_bytes_; }
    std::size_t max_memory_bytes() const { return max_memory_bytes_; }

private:
    void promote_to_protected(EntryId id);
    void demote_lru_protected();
    std::size_t target_protected_bytes() const;

    enum class Segment { PROBATION, PROTECTED };

    std::list<EntryId> probationary_;
    std::list<EntryId> protected_;
    struct EntryInfo {
        Segment segment;
        std::list<EntryId>::iterator pos;
        std::size_t size;
    };
    std::unordered_map<EntryId, EntryInfo> entries_;
    std::size_t probationary_bytes_ = 0;
    std::size_t protected_bytes_ = 0;
    std::size_t max_memory_bytes_;
    double protected_ratio_ = 0.80;
};
