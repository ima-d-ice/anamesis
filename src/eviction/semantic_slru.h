#pragma once

#include "slru.h"
#include "index/vector_index.hpp"
#include <cstddef>
#include <cstdint>
#include <list>
#include <unordered_map>
#include <vector>
#include <cmath>

class SemanticSLRU : public SLRU {
public:
    SemanticSLRU(size_t max_memory_bytes, double protected_ratio = 0.80,
                 double semantic_weight = 0.5);

    // Called when an embedding is available for an entry
    void set_embedding(EntryId id, const float* data,
                       size_t dimension) override;

    // Compute semantic uniqueness for an entry (1 - max similarity to others)
    double semantic_uniqueness(EntryId id) const;

    // Override eviction to consider semantic uniqueness
    EntryId evict() override;

    // Set the IVectorIndex for computing similarities
    void set_vector_index(IVectorIndex* index);

    // Get stats
    size_t semantic_evictions() const { return semantic_evictions_; }
    size_t uniqueness_evictions() const { return uniqueness_evictions_; }

private:
    IVectorIndex* vector_index_ = nullptr;
    std::unordered_map<EntryId, std::vector<float>> embeddings_;
    std::unordered_map<EntryId, double> uniqueness_cache_;
    double semantic_weight_ = 0.5;
    size_t semantic_evictions_ = 0;
    size_t uniqueness_evictions_ = 0;

    // Compute cosine similarity between two vectors
    double cosine_sim(const std::vector<float>& a, const std::vector<float>& b) const;

    // Recompute uniqueness for all entries (expensive, call sparingly)
    void rebuild_uniqueness_cache();

    // Compute uniqueness for a single entry
    double compute_uniqueness(EntryId id) const;
};
