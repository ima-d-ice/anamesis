#pragma once

#include "slru.h"
#include "embedding.h"

#include <cstddef>
#include <cstdint>
#include <list>
#include <unordered_map>
#include <utility>
#include <vector>
#include <cmath>

// SemanticSLRU extends SLRU with uniqueness-aware eviction of the protected
// segment. A victim is chosen among at most kMaxCandidates protected entries
// (rotating cursor), each compared against at most kMaxPeers same-namespace
// peers (rotating cursor). Similarity never crosses namespace boundaries and
// the work per eviction is bounded (~8k cosine evals), so eviction cost stays
// flat as the cache grows.
//
// Victim score = (1 - w) * recency_age + w * (1 - uniqueness), evict max:
//   w = 0.0  -> plain SLRU (oldest protected entry loses)
//   w = 1.0  -> pure uniqueness (least unique entry loses)
class SemanticSLRU : public SLRU {
public:
    SemanticSLRU(std::size_t max_memory_bytes, double protected_ratio = 0.80,
                 double semantic_weight = 0.5);

    void set_embedding(EntryId id, const float* data,
                       std::size_t dimension,
                       uint64_t namespace_id) override;
    void remove(EntryId id) override;
    EntryId evict() override;

    std::size_t probationary_evictions() const override { return probationary_evictions_; }
    std::size_t uniqueness_evictions() const override { return uniqueness_evictions_; }

    void note_add(EntryId id) override;
    void note_touch(EntryId id) override;
    void note_protected_list_changed() override;
    std::size_t memory_used() const override;

private:
    static constexpr std::size_t kMaxCandidates = 32;
    static constexpr std::size_t kMaxPeers = 256;

    struct EmbeddingInfo {
        std::vector<float> vec;
        uint64_t namespace_id = 0;
    };

    struct UniquenessCacheEntry {
        double value;
        uint64_t generation;
    };

    // Touch/update bookkeeping for every entry (recency tracking).
    void note_seq(EntryId id);

    double recency_age(EntryId id) const;
    double uniqueness(EntryId id);                 // computes + caches
    double namespace_uniqueness(EntryId id, uint64_t ns,
                                const std::vector<float>& vec);

    // Select a protected-segment victim using bounded, rotating scans.
    // Returns {victim_id, victim_list_iterator}; {0, end()} if none.
    std::pair<EntryId, std::list<EntryId>::iterator> pick_protected_victim();
    void ensure_protected_cursor();

    void peer_add(uint64_t ns, EntryId id);
    void peer_remove(uint64_t ns, EntryId id);
    void invalidate_namespace(uint64_t ns);

    std::unordered_map<EntryId, EmbeddingInfo> embeddings_;
    std::unordered_map<uint64_t, std::vector<EntryId>> ns_peers_;
    std::unordered_map<EntryId, std::size_t> ns_peer_pos_;
    std::unordered_map<uint64_t, std::size_t> ns_peer_cursor_;
    std::unordered_map<uint64_t, uint64_t> ns_generation_;
    std::unordered_map<EntryId, UniquenessCacheEntry> uniqueness_cache_;
    std::unordered_map<EntryId, uint64_t> entry_seq_;
    std::list<EntryId>::iterator protected_cursor_;
    bool cursor_valid_ = false;

    uint64_t seq_counter_ = 0;
    uint64_t max_seq_ = 0;

    double semantic_weight_ = 0.5;
    std::size_t probationary_evictions_ = 0;  // probationary (LRU) evictions
    std::size_t uniqueness_evictions_ = 0;    // protected-segment evictions
};
