#include "semantic_slru.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_set>

using namespace std;

SemanticSLRU::SemanticSLRU(size_t max_memory_bytes, double protected_ratio,
                           double semantic_weight)
    : SLRU(max_memory_bytes, protected_ratio),
      semantic_weight_(semantic_weight) {
    if (semantic_weight_ < 0.0) {
        semantic_weight_ = 0.0;
    }
    if (semantic_weight_ > 1.0) {
        semantic_weight_ = 1.0;
    }
}

void SemanticSLRU::note_seq(EntryId id) {
    ++seq_counter_;
    entry_seq_[id] = seq_counter_;
    if (seq_counter_ > max_seq_) {
        max_seq_ = seq_counter_;
    }
}

double SemanticSLRU::recency_age(EntryId id) const {
    auto it = entry_seq_.find(id);
    if (it == entry_seq_.end() || max_seq_ == 0) {
        return 0.0;
    }
    double age = 1.0 - static_cast<double>(it->second) /
                         static_cast<double>(max_seq_);
    if (age < 0.0) {
        age = 0.0;
    }
    if (age > 1.0) {
        age = 1.0;
    }
    return age;
}

void SemanticSLRU::peer_add(uint64_t ns, EntryId id) {
    auto& vec = ns_peers_[ns];
    ns_peer_pos_[id] = vec.size();
    vec.push_back(id);
}

void SemanticSLRU::peer_remove(uint64_t ns, EntryId id) {
    auto pit = ns_peers_.find(ns);
    if (pit == ns_peers_.end()) {
        return;
    }
    auto& vec = pit->second;
    auto iit = ns_peer_pos_.find(id);
    if (iit == ns_peer_pos_.end()) {
        return;
    }
    size_t pos = iit->second;
    EntryId last = vec.back();
    vec[pos] = last;
    ns_peer_pos_[last] = pos;
    vec.pop_back();
    ns_peer_pos_.erase(id);
    if (vec.empty()) {
        ns_peers_.erase(pit);
        ns_peer_cursor_.erase(ns);
    }
}

void SemanticSLRU::invalidate_namespace(uint64_t ns) {
    ++ns_generation_[ns];
}

void SemanticSLRU::set_embedding(EntryId id, const float* data,
                                 size_t dimension,
                                 uint64_t namespace_id) {
    if (dimension == 0 || data == nullptr) {
        return;
    }
    vector<float> vec(data, data + dimension);
    auto it = embeddings_.find(id);
    if (it != embeddings_.end()) {
        it->second.vec = std::move(vec);
        uniqueness_cache_.erase(id);
        invalidate_namespace(namespace_id);
        return;
    }
    EmbeddingInfo info;
    info.vec = std::move(vec);
    info.namespace_id = namespace_id;
    embeddings_[id] = info;
    uniqueness_cache_.erase(id);
    peer_add(namespace_id, id);
    invalidate_namespace(namespace_id);
}

void SemanticSLRU::note_add(EntryId id) {
    note_seq(id);
}

void SemanticSLRU::note_touch(EntryId id) {
    note_seq(id);
}

void SemanticSLRU::note_protected_list_changed() {
    cursor_valid_ = false;
}

void SemanticSLRU::remove(EntryId id) {
    uint64_t ns = 0;
    bool had_embedding = false;
    auto it = embeddings_.find(id);
    if (it != embeddings_.end()) {
        ns = it->second.namespace_id;
        had_embedding = true;
    }
    SLRU::remove(id);  // base invalidates the rotating cursor via the hook
    if (!had_embedding) {
        return;
    }
    embeddings_.erase(it);
    entry_seq_.erase(id);
    uniqueness_cache_.erase(id);
    peer_remove(ns, id);
    invalidate_namespace(ns);
}

double SemanticSLRU::namespace_uniqueness(EntryId id, uint64_t ns,
                                          const vector<float>& vec) {
    auto pit = ns_peers_.find(ns);
    if (pit == ns_peers_.end() || pit->second.empty()) {
        return 1.0;
    }
    const vector<EntryId>& peers = pit->second;
    size_t n = peers.size();
    size_t take = std::min(kMaxPeers, n);
    size_t start = ns_peer_cursor_[ns];
    if (start >= n) {
        start = 0;
        ns_peer_cursor_[ns] = 0;
    }

    double max_sim = 0.0;
    size_t examined = 0;
    size_t pos = start;
    while (examined < take) {
        EntryId peer = peers[pos];
        ++pos;
        if (pos >= n) {
            pos = 0;
        }
        ++examined;
        if (peer == id) {
            continue;
        }
        double sim = embedding::cosine_similarity(vec, embeddings_.at(peer).vec);
        if (sim > max_sim) {
            max_sim = sim;
        }
    }
    ns_peer_cursor_[ns] = pos;
    return 1.0 - max_sim;
}

double SemanticSLRU::uniqueness(EntryId id) {
    auto it = embeddings_.find(id);
    if (it == embeddings_.end()) {
        return 1.0;
    }
    uint64_t ns = it->second.namespace_id;
    uint64_t gen = ns_generation_[ns];
    auto cit = uniqueness_cache_.find(id);
    if (cit != uniqueness_cache_.end() && cit->second.generation == gen) {
        return cit->second.value;
    }
    double uniq = namespace_uniqueness(id, ns, it->second.vec);
    uniqueness_cache_[id] = {uniq, gen};
    return uniq;
}

void SemanticSLRU::ensure_protected_cursor() {
    if (!cursor_valid_ || protected_.empty()) {
        protected_cursor_ = protected_.begin();
        cursor_valid_ = !protected_.empty();
        return;
    }
    // cursor_valid_ implies the iterator is still valid: every protected-list
    // mutation invalidates the cursor via note_protected_list_changed().
    auto sit = entries_.find(*protected_cursor_);
    if (sit == entries_.end() || sit->second.segment != Segment::PROTECTED) {
        protected_cursor_ = protected_.begin();
        cursor_valid_ = !protected_.empty();
    }
}

pair<EntryId, list<EntryId>::iterator> SemanticSLRU::pick_protected_victim() {
    ensure_protected_cursor();
    if (protected_.empty()) {
        return {0, protected_.end()};
    }

    size_t n = protected_.size();
    size_t take = std::min(kMaxCandidates, n);
    vector<EntryId> candidates;
    candidates.reserve(take);
    vector<list<EntryId>::iterator> candidate_iters;
    candidate_iters.reserve(take);
    unordered_set<EntryId> seen;
    seen.reserve(take);

    auto pos = protected_cursor_;
    while (candidates.size() < take) {
        EntryId id = *pos;
        if (seen.insert(id).second) {
            candidates.push_back(id);
            candidate_iters.push_back(pos);
        }
        ++pos;
        if (pos == protected_.end()) {
            pos = protected_.begin();
        }
    }
    protected_cursor_ = pos;
    cursor_valid_ = true;

    EntryId victim = 0;
    list<EntryId>::iterator victim_it = protected_.end();
    double best_score = -1.0;
    double w = semantic_weight_;
    for (size_t i = 0; i < candidates.size(); ++i) {
        EntryId id = candidates[i];
        double age = recency_age(id);
        double uniq = uniqueness(id);
        double score = (1.0 - w) * age + w * (1.0 - uniq);
        if (score > best_score) {
            best_score = score;
            victim = id;
            victim_it = candidate_iters[i];
        }
    }
    return {victim, victim_it};
}

EntryId SemanticSLRU::evict() {
    // Probationary segment first: pure LRU tail, O(1), identical to SLRU.
    if (!probationary_.empty()) {
        EntryId id = probationary_.back();
        probationary_.pop_back();
        auto it = entries_.find(id);
        if (it != entries_.end()) {
            probationary_bytes_ -= it->second.size;
            entry_seq_.erase(id);
            uniqueness_cache_.erase(id);
            auto emb = embeddings_.find(id);
            if (emb != embeddings_.end()) {
                peer_remove(emb->second.namespace_id, id);
                invalidate_namespace(emb->second.namespace_id);
                embeddings_.erase(emb);
            }
            entries_.erase(it);
        }
        ++probationary_evictions_;
        return id;
    }

    // Protected segment: bounded, uniqueness-aware victim selection.
    if (!protected_.empty()) {
        auto [victim, victim_it] = pick_protected_victim();
        if (victim != 0) {
            protected_.erase(victim_it);
            cursor_valid_ = false;
            auto it = entries_.find(victim);
            if (it != entries_.end()) {
                protected_bytes_ -= it->second.size;
                entry_seq_.erase(victim);
                uniqueness_cache_.erase(victim);
                auto emb = embeddings_.find(victim);
                if (emb != embeddings_.end()) {
                    peer_remove(emb->second.namespace_id, victim);
                    invalidate_namespace(emb->second.namespace_id);
                    embeddings_.erase(emb);
                }
                entries_.erase(it);
            }
            ++uniqueness_evictions_;
            return victim;
        }
    }

    return 0;
}

size_t SemanticSLRU::memory_used() const {
    size_t bytes = SLRU::memory_used();
    for (const auto& [_, info] : embeddings_) {
        bytes += info.vec.size() * sizeof(float);
    }
    return bytes;
}
