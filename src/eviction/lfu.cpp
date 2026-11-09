#include "lfu.h"

using namespace std;

void LFU::touch(EntryId id) {
    auto it = entries_.find(id);
    if (it == entries_.end()) {
        return;
    }
    Entry& e = it->second;
    buckets_[e.freq].erase(e.it);
    if (buckets_[e.freq].empty() && e.freq == min_freq_) {
        ++min_freq_;
    }
    ++e.freq;
    buckets_[e.freq].push_front(id);
    e.it = buckets_[e.freq].begin();
}

void LFU::add(EntryId id, size_t size) {
    auto it = entries_.find(id);
    if (it != entries_.end()) {
        memory_used_ -= it->second.size;
        it->second.size = size;
        memory_used_ += size;
        return;
    }
    buckets_[1].push_front(id);
    entries_.insert({id, Entry{size, 1, buckets_[1].begin()}});
    min_freq_ = 1;
    memory_used_ += size;
}

void LFU::remove(EntryId id) {
    auto it = entries_.find(id);
    if (it == entries_.end()) {
        return;
    }
    Entry& e = it->second;
    buckets_[e.freq].erase(e.it);
    if (buckets_[e.freq].empty() && e.freq == min_freq_) {
        ++min_freq_;
    }
    memory_used_ -= e.size;
    entries_.erase(it);
}

EntryId LFU::evict() {
    if (entries_.empty()) {
        return 0;
    }
    while (buckets_[min_freq_].empty()) {
        ++min_freq_;
    }
    list<EntryId>& bucket = buckets_[min_freq_];
    EntryId id = bucket.back();
    bucket.pop_back();
    memory_used_ -= entries_[id].size;
    entries_.erase(id);
    return id;
}

size_t LFU::memory_used() const {
    return memory_used_;
}
