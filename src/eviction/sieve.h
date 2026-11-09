#pragma once

#include "policy.h"

#include <cstddef>
#include <cstdint>
#include <list>
#include <unordered_map>

class SIEVE : public EvictionPolicy {
public:
    void touch(EntryId id) override;
    void add(EntryId id, std::size_t size) override;
    void remove(EntryId id) override;
    EntryId evict() override;
    std::size_t memory_used() const override;

private:
    struct Entry {
        std::size_t size;
        std::list<EntryId>::iterator it;
        bool visited = false;
    };
    std::list<EntryId> fifo_;
    std::unordered_map<EntryId, Entry> entries_;
    std::list<EntryId>::iterator hand_;
    std::size_t memory_used_ = 0;
};
