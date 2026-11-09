#include "lru.h"

using namespace std;

void LRU::touch(EntryId id) {
    auto it = entries_.find(id);
    if (it == entries_.end()) {
        return;
    }
    order_.erase(it->second.second);
    order_.push_front(id);
    it->second.second = order_.begin();
}

void LRU::add(EntryId id, size_t size) {
    auto it = entries_.find(id);
    if (it != entries_.end()) {
        memory_used_ -= it->second.first;
        order_.erase(it->second.second);
        order_.push_front(id);
        it->second = {size, order_.begin()};
        memory_used_ += size;
        return;
    }
    order_.push_front(id);
    entries_.insert({id, {size, order_.begin()}});
    memory_used_ += size;
}

void LRU::remove(EntryId id) {
    auto it = entries_.find(id);
    if (it == entries_.end()) {
        return;
    }
    order_.erase(it->second.second);
    memory_used_ -= it->second.first;
    entries_.erase(it);
}

EntryId LRU::evict() {
    if (entries_.empty()) {
        return 0;
    }
    EntryId id = order_.back();
    order_.pop_back();
    memory_used_ -= entries_[id].first;
    entries_.erase(id);
    return id;
}

size_t LRU::memory_used() const {
    return memory_used_;
}
