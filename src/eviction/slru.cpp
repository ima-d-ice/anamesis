#include "slru.h"

using namespace std;

SLRU::SLRU(size_t max_memory_bytes, double protected_ratio)
    : max_memory_bytes_(max_memory_bytes),
      protected_ratio_(protected_ratio) {}

void SLRU::touch(EntryId id) {
    auto it = entries_.find(id);
    if (it == entries_.end()) {
        return;
    }

    EntryInfo& info = it->second;
    if (info.segment == Segment::PROBATION) {
        probationary_.erase(info.pos);
        promote_to_protected(id);
        note_touch(id);
    } else {
        protected_.erase(info.pos);
        protected_.push_front(id);
        info.pos = protected_.begin();
        note_touch(id);
    }
}

void SLRU::promote_to_protected(EntryId id) {
    EntryInfo& info = entries_[id];
    info.segment = Segment::PROTECTED;
    protected_.push_front(id);
    info.pos = protected_.begin();

    probationary_bytes_ -= info.size;
    protected_bytes_ += info.size;

    while (protected_bytes_ > target_protected_bytes() && !protected_.empty()) {
        demote_lru_protected();
    }
}

void SLRU::demote_lru_protected() {
    if (protected_.empty()) {
        return;
    }
    EntryId id = protected_.back();
    protected_.pop_back();

    EntryInfo& info = entries_[id];
    info.segment = Segment::PROBATION;
    probationary_.push_front(id);
    info.pos = probationary_.begin();

    protected_bytes_ -= info.size;
    probationary_bytes_ += info.size;
}

size_t SLRU::target_protected_bytes() const {
    return static_cast<size_t>(static_cast<double>(max_memory_bytes_) * protected_ratio_);
}

void SLRU::add(EntryId id, size_t size) {
    auto it = entries_.find(id);
    if (it != entries_.end()) {
        // Update size in place without changing segment or recency.
        if (it->second.segment == Segment::PROBATION) {
            probationary_bytes_ -= it->second.size;
            probationary_bytes_ += size;
        } else {
            protected_bytes_ -= it->second.size;
            protected_bytes_ += size;
        }
        it->second.size = size;
        return;
    }

    EntryInfo info;
    info.segment = Segment::PROBATION;
    probationary_.push_front(id);
    info.pos = probationary_.begin();
    info.size = size;
    entries_[id] = info;
    probationary_bytes_ += size;
    note_add(id);
}

void SLRU::remove(EntryId id) {
    auto it = entries_.find(id);
    if (it == entries_.end()) {
        return;
    }

    EntryInfo& info = it->second;
    if (info.segment == Segment::PROBATION) {
        probationary_.erase(info.pos);
        probationary_bytes_ -= info.size;
    } else {
        protected_.erase(info.pos);
        protected_bytes_ -= info.size;
    }
    entries_.erase(it);
}

EntryId SLRU::evict() {
    if (!probationary_.empty()) {
        EntryId id = probationary_.back();
        probationary_.pop_back();
        auto it = entries_.find(id);
        if (it != entries_.end()) {
            probationary_bytes_ -= it->second.size;
            entries_.erase(it);
        }
        return id;
    }

    if (!protected_.empty()) {
        EntryId id = protected_.back();
        protected_.pop_back();
        auto it = entries_.find(id);
        if (it != entries_.end()) {
            protected_bytes_ -= it->second.size;
            entries_.erase(it);
        }
        return id;
    }

    return 0;
}

size_t SLRU::memory_used() const {
    return probationary_bytes_ + protected_bytes_;
}
