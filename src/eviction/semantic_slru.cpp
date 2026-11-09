#include "semantic_slru.h"
#include <algorithm>
#include <cmath>
#include <limits>

SemanticSLRU::SemanticSLRU(size_t max_memory_bytes, double protected_ratio,
                           double semantic_weight)
    : SLRU(max_memory_bytes, protected_ratio),
      semantic_weight_(semantic_weight) {}

void SemanticSLRU::set_embedding(EntryId id, const float* data,
                                 size_t dimension) {
    if (dimension == 0 || data == nullptr) return;
    std::vector<float> vec(data, data + dimension);
    embeddings_[id] = vec;
    // Invalidate cached uniqueness since embeddings changed
    uniqueness_cache_.erase(id);
}

void SemanticSLRU::set_vector_index(IVectorIndex* index) {
    vector_index_ = index;
}

double SemanticSLRU::cosine_sim(const std::vector<float>& a,
                                const std::vector<float>& b) const {
    if (a.empty() || b.empty() || a.size() != b.size()) return 0.0;
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += static_cast<double>(a[i]) * static_cast<double>(b[i]);
        na += static_cast<double>(a[i]) * static_cast<double>(a[i]);
        nb += static_cast<double>(b[i]) * static_cast<double>(b[i]);
    }
    if (na == 0.0 || nb == 0.0) return 0.0;
    return dot / (std::sqrt(na) * std::sqrt(nb));
}

double SemanticSLRU::compute_uniqueness(EntryId id) const {
    auto it = embeddings_.find(id);
    if (it == embeddings_.end()) return 1.0;  // No embedding = unique

    const auto& vec = it->second;
    double max_sim = 0.0;

    for (const auto& [other_id, other_vec] : embeddings_) {
        if (other_id == id) continue;
        double sim = cosine_sim(vec, other_vec);
        if (sim > max_sim) max_sim = sim;
    }

    return 1.0 - max_sim;  // 1.0 = completely unique, 0.0 = duplicate
}

double SemanticSLRU::semantic_uniqueness(EntryId id) const {
    auto it = uniqueness_cache_.find(id);
    if (it != uniqueness_cache_.end()) return it->second;
    double sim = compute_uniqueness(id);
    return sim;
}

void SemanticSLRU::rebuild_uniqueness_cache() {
    uniqueness_cache_.clear();
    for (const auto& [id, _] : embeddings_) {
        uniqueness_cache_[id] = compute_uniqueness(id);
    }
}

EntryId SemanticSLRU::evict() {
    // First try probationary segment (same as SLRU)
    if (!probationary_.empty()) {
        EntryId id = probationary_.back();
        probationary_.pop_back();
        auto it = entries_.find(id);
        if (it != entries_.end()) {
            probationary_bytes_ -= it->second.size;
            entries_.erase(it);
            uniqueness_cache_.erase(id);
            embeddings_.erase(id);
            ++semantic_evictions_;
        }
        return id;
    }

    // For protected segment, evict the least semantically unique entry
    if (!protected_.empty()) {
        // Find the entry with lowest uniqueness (most similar to others)
        EntryId victim = 0;
        double min_uniqueness = 1.0;
        auto victim_it = protected_.end();

        for (auto it = protected_.begin(); it != protected_.end(); ++it) {
            EntryId id = *it;
            double uniq = semantic_uniqueness(id);
            if (uniq < min_uniqueness) {
                min_uniqueness = uniq;
                victim = id;
                victim_it = it;
            }
        }

        if (victim != 0) {
            protected_.erase(victim_it);
            auto it = entries_.find(victim);
            if (it != entries_.end()) {
                protected_bytes_ -= it->second.size;
                entries_.erase(it);
                uniqueness_cache_.erase(victim);
                embeddings_.erase(victim);
            }
            ++uniqueness_evictions_;
            return victim;
        }
    }

    return 0;
}
