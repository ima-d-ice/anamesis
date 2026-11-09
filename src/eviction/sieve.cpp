#include "sieve.h"

using namespace std;

void SIEVE::touch(EntryId id) {
    auto it = entries_.find(id);
    if (it == entries_.end()) {
        return;
    }
    it->second.visited = true;
}

void SIEVE::add(EntryId id, size_t size) {
    auto it = entries_.find(id);
    if (it != entries_.end()) {
        memory_used_ -= it->second.size;
        it->second.size = size;
        memory_used_ += size;
        return;
    }
    fifo_.push_front(id);
    entries_.insert({id, Entry{size, fifo_.begin(), false}});
    memory_used_ += size;
    if (fifo_.size() == 1) {
        hand_ = fifo_.begin();
    }
}

void SIEVE::remove(EntryId id) {
    auto it = entries_.find(id);
    if (it == entries_.end()) {
        return;
    }
    if (it->second.it == hand_) {
        ++hand_;
    }
    fifo_.erase(it->second.it);
    memory_used_ -= it->second.size;
    entries_.erase(it);
    if (entries_.empty()) {
        hand_ = fifo_.end();
    }
}

EntryId SIEVE::evict() {
    if (entries_.empty()) {
        return 0;
    }
    while (hand_ != fifo_.end() && entries_[*hand_].visited) {
        entries_[*hand_].visited = false;
        ++hand_;
        if (hand_ == fifo_.end()) {
            hand_ = fifo_.begin();
        }
    }
    if (hand_ == fifo_.end()) {
        return 0;
    }
    EntryId id = *hand_;
    auto next = hand_;
    ++next;
    if (next == fifo_.end()) {
        next = fifo_.begin();
    }
    hand_ = next;
    memory_used_ -= entries_[id].size;
    fifo_.erase(entries_[id].it);
    entries_.erase(id);
    if (entries_.empty()) {
        hand_ = fifo_.end();
    }
    return id;
}

size_t SIEVE::memory_used() const {
    return memory_used_;
}
