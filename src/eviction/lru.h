#pragma once

#include "policy.h"

#include <cstddef>
#include <cstdint>
#include <list>
#include <unordered_map>

class LRU : public EvictionPolicy {
public:
    void touch(EntryId id) override;
    void add(EntryId id, std::size_t size) override;
    void remove(EntryId id) override;
    EntryId evict() override;
    std::size_t memory_used() const override;

private:
    std::list<EntryId> order_;
    std::unordered_map<EntryId, std::pair<std::size_t, std::list<EntryId>::iterator>> entries_;
    std::size_t memory_used_ = 0;
};
