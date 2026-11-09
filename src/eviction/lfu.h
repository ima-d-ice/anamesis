#pragma once

#include "policy.h"

#include <cstddef>
#include <cstdint>
#include <list>
#include <unordered_map>

class LFU : public EvictionPolicy {
public:
    void touch(EntryId id) override;
    void add(EntryId id, std::size_t size) override;
    void remove(EntryId id) override;
    EntryId evict() override;
    std::size_t memory_used() const override;

private:
    struct Entry {
        std::size_t size;
        int freq;
        std::list<EntryId>::iterator it;
    };
    std::unordered_map<int, std::list<EntryId>> buckets_;
    std::unordered_map<EntryId, Entry> entries_;
    int min_freq_ = 0;
    std::size_t memory_used_ = 0;
};
