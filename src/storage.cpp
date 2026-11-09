#include "storage.h"

#include "embedding.h"
#include "eviction/lfu.h"
#include "eviction/lru.h"
#include "eviction/sieve.h"
#include "eviction/slru.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <memory>
#include <shared_mutex>
#include <utility>
#include <vector>

using namespace std;

Storage::Storage(size_t memory_limit, size_t max_value_size,
                 const char* policy_name, int shard_count)
    : memory_limit_(memory_limit),
      max_value_size_(max_value_size),
      next_id_{1},
      ttl_([this](uint64_t id) { expire_key(id); }) {
    if (shard_count < 1) {
        shard_count = 1;
    }
    shards_.reserve(static_cast<size_t>(shard_count));
    for (int i = 0; i < shard_count; ++i) {
        auto shard = make_unique<CacheShard>();
        string policy = policy_name ? policy_name : "slru";
        transform(policy.begin(), policy.end(), policy.begin(),
                  [](unsigned char c) { return static_cast<char>(tolower(c)); });
        if (policy == "lru") {
            shard->policy = make_unique<LRU>();
        } else if (policy == "lfu") {
            shard->policy = make_unique<LFU>();
        } else if (policy == "sieve") {
            shard->policy = make_unique<SIEVE>();
        } else {
            shard->policy =
                make_unique<SLRU>(memory_limit / static_cast<size_t>(shard_count),
                                  0.80);
        }
        shards_.push_back(std::move(shard));
    }
}

uint64_t Storage::now_ms() const {
    return static_cast<uint64_t>(
        chrono::duration_cast<chrono::milliseconds>(
            chrono::steady_clock::now().time_since_epoch())
            .count());
}

string Storage::canonicalize(const string& query) const {
    string out;
    out.reserve(query.size());
    bool prev_space = true;
    for (char c : query) {
        if (isspace(static_cast<unsigned char>(c))) {
            if (!prev_space) {
                out.push_back(' ');
                prev_space = true;
            }
        } else {
            out.push_back(c);
            prev_space = false;
        }
    }
    if (!out.empty() && out.back() == ' ') {
        out.pop_back();
    }
    return out;
}

int Storage::shard_for(uint64_t namespace_id) const {
    return static_cast<int>(hash<uint64_t>{}(namespace_id) % shards_.size());
}

CacheShard& Storage::shard_ref(uint64_t namespace_id) {
    return *shards_[static_cast<size_t>(shard_for(namespace_id))];
}

const CacheShard& Storage::shard_ref(uint64_t namespace_id) const {
    return *shards_[static_cast<size_t>(shard_for(namespace_id))];
}

size_t Storage::entry_bytes(const CacheEntry& entry) const {
    size_t emb_bytes = entry.embedding.size() * sizeof(float);
    return entry.original_query.size() + entry.canonical_query.size() +
           entry.response.size() + emb_bytes + kEntryOverhead;
}

bool Storage::can_store(const string& key, const string& value) const {
    if (key.size() > kMaxKeySize || value.size() > max_value_size_) {
        return false;
    }
    const size_t accounted = value.size() + key.size() + kEntryOverhead;
    return accounted <= memory_limit_;
}

bool Storage::set(string key, string value, int ttl_sec) {
    auto& shard = shards_[0];
    unique_lock<shared_mutex> lk(shard->lock);
    if (!can_store(key, value)) {
        return false;
    }

    uint64_t ns = 0; // legacy default namespace
    string canonical = canonicalize(key);
    ExactKey ek{ns, canonical};

    auto existing = shard->exact_index.find(ek);
    if (existing != shard->exact_index.end()) {
        EntryId id = existing->second;
        auto eit = shard->entries.find(id);
        if (eit != shard->entries.end()) {
            shard->value_bytes -= eit->second.response.size();
            eit->second.response = std::move(value);
            eit->second.original_query = key;
            eit->second.canonical_query = canonical;
            eit->second.memory_bytes = entry_bytes(eit->second);
            eit->second.last_access_ms = now_ms();
            eit->second.hit_count++;
            if (ttl_sec > 0) {
                eit->second.expires_at_ms =
                    now_ms() + static_cast<uint64_t>(ttl_sec) * 1000;
                ttl_.set_ttl(id, chrono::milliseconds(static_cast<int64_t>(ttl_sec) * 1000));
            } else {
                eit->second.expires_at_ms = 0;
                ttl_.erase(id);
            }
            shard->value_bytes += eit->second.response.size();
            shard->policy->add(id, eit->second.memory_bytes);
            evict_if_over_limit(*shard);
            return true;
        }
    }

    EntryId id = next_id_.fetch_add(1, memory_order_relaxed);
    uint64_t now = now_ms();
    uint64_t expires = ttl_sec > 0 ? now + static_cast<uint64_t>(ttl_sec) * 1000 : 0;

    CacheEntry entry;
    entry.id = id;
    entry.namespace_id = ns;
    entry.canonical_query = canonical;
    entry.original_query = key;
    entry.response = std::move(value);
    entry.content_type = "text/plain";
    entry.created_at_ms = now;
    entry.expires_at_ms = expires;
    entry.last_access_ms = now;
    entry.memory_bytes = entry_bytes(entry);

    auto result = shard->entries.emplace(id, entry);
    if (!result.second) {
        return false;
    }
    shard->exact_index.emplace(ek, id);

    if (ttl_sec > 0) {
        ttl_.set_ttl(id, chrono::milliseconds(static_cast<int64_t>(ttl_sec) * 1000));
    }

    shard->policy->add(id, entry.memory_bytes);
    shard->value_bytes += entry.response.size();
    shard->largest_item = max(shard->largest_item, entry.response.size());
    evict_if_over_limit(*shard);
    return true;
}

Blob Storage::get(const string& key) {
    uint64_t ns = 0;
    string canonical = canonicalize(key);
    ExactKey ek{ns, canonical};

    auto& shard = shards_[0];
    shared_lock<shared_mutex> lk(shard->lock);
    auto it = shard->exact_index.find(ek);
    if (it == shard->exact_index.end()) {
        misses_.fetch_add(1, memory_order_relaxed);
        return nullptr;
    }
    EntryId id = it->second;
    auto eit = shard->entries.find(id);
    if (eit == shard->entries.end() ||
        (eit->second.expires_at_ms != 0 &&
         eit->second.expires_at_ms <= now_ms())) {
        misses_.fetch_add(1, memory_order_relaxed);
        return nullptr;
    }
    lk.unlock();
    unique_lock<shared_mutex> ulk(shard->lock);
    eit = shard->entries.find(id);
    if (eit == shard->entries.end()) {
        misses_.fetch_add(1, memory_order_relaxed);
        return nullptr;
    }
    eit->second.last_access_ms = now_ms();
    eit->second.hit_count++;
    shard->policy->touch(id);
    hits_.fetch_add(1, memory_order_relaxed);
    return make_shared<const string>(eit->second.response);
}

bool Storage::del(const string& key) {
    uint64_t ns = 0;
    string canonical = canonicalize(key);
    ExactKey ek{ns, canonical};

    auto& shard = shards_[0];
    unique_lock<shared_mutex> lk(shard->lock);
    auto it = shard->exact_index.find(ek);
    if (it == shard->exact_index.end()) {
        return false;
    }
    EntryId id = it->second;
    shard->exact_index.erase(it);
    auto eit = shard->entries.find(id);
    if (eit != shard->entries.end()) {
        shard->value_bytes -= eit->second.response.size();
        shard->policy->remove(id);
        ttl_.erase(id);
        shard->entries.erase(eit);
    }
    return true;
}

EntryId Storage::semset(uint64_t namespace_id, string query, string value,
                        vector<float> embedding, int ttl_sec) {
    auto& shard = shard_ref(namespace_id);
    unique_lock<shared_mutex> lk(shard.lock);

    if (query.size() > kMaxKeySize || value.size() > max_value_size_) {
        return 0;
    }

    string canonical = canonicalize(query);
    ExactKey ek{namespace_id, canonical};
    auto existing = shard.exact_index.find(ek);
    if (existing != shard.exact_index.end()) {
        EntryId id = existing->second;
        auto eit = shard.entries.find(id);
        if (eit != shard.entries.end()) {
            shard.value_bytes -= eit->second.response.size();
            eit->second.response = std::move(value);
            eit->second.embedding = std::move(embedding);
            eit->second.original_query = query;
            eit->second.canonical_query = canonical;
            eit->second.memory_bytes = entry_bytes(eit->second);
            eit->second.last_access_ms = now_ms();
            eit->second.hit_count++;
            if (ttl_sec > 0) {
                eit->second.expires_at_ms =
                    now_ms() + static_cast<uint64_t>(ttl_sec) * 1000;
                ttl_.set_ttl(id, chrono::milliseconds(static_cast<int64_t>(ttl_sec) * 1000));
            } else {
                eit->second.expires_at_ms = 0;
                ttl_.erase(id);
            }
            shard.value_bytes += eit->second.response.size();
            shard.policy->add(id, eit->second.memory_bytes);
            evict_if_over_limit(shard);
            return id;
        }
    }

    EntryId id = next_id_.fetch_add(1, memory_order_relaxed);
    uint64_t now = now_ms();
    uint64_t expires = ttl_sec > 0 ? now + static_cast<uint64_t>(ttl_sec) * 1000 : 0;

    CacheEntry entry;
    entry.id = id;
    entry.namespace_id = namespace_id;
    entry.canonical_query = canonical;
    entry.original_query = std::move(query);
    entry.response = std::move(value);
    entry.content_type = "text/plain";
    entry.embedding = std::move(embedding);
    entry.created_at_ms = now;
    entry.expires_at_ms = expires;
    entry.last_access_ms = now;
    entry.memory_bytes = entry_bytes(entry);

    auto result = shard.entries.emplace(id, entry);
    if (!result.second) {
        return 0;
    }
    shard.exact_index.emplace(ek, id);

    if (ttl_sec > 0) {
        ttl_.set_ttl(id, chrono::milliseconds(static_cast<int64_t>(ttl_sec) * 1000));
    }

    shard.policy->add(id, entry.memory_bytes);
    shard.value_bytes += entry.response.size();
    shard.largest_item = max(shard.largest_item, entry.response.size());

    if (!entry.embedding.empty()) {
        if (!shard.vector_index) {
            shard.vector_index = make_flat_index(entry.embedding.size());
        }
        shard.vector_index->insert(id, entry.embedding.data(),
                                   entry.embedding.size());
    }

    evict_if_over_limit(shard);
    return id;
}

Blob Storage::semget(uint64_t namespace_id, const string& query) {
    auto& shard = shard_ref(namespace_id);
    string canonical = canonicalize(query);
    ExactKey ek{namespace_id, canonical};

    shared_lock<shared_mutex> lk(shard.lock);
    auto it = shard.exact_index.find(ek);
    if (it == shard.exact_index.end()) {
        misses_.fetch_add(1, memory_order_relaxed);
        return nullptr;
    }
    EntryId id = it->second;
    auto eit = shard.entries.find(id);
    if (eit == shard.entries.end() ||
        (eit->second.expires_at_ms != 0 &&
         eit->second.expires_at_ms <= now_ms())) {
        misses_.fetch_add(1, memory_order_relaxed);
        return nullptr;
    }
    lk.unlock();
    unique_lock<shared_mutex> ulk(shard.lock);
    eit = shard.entries.find(id);
    if (eit == shard.entries.end()) {
        misses_.fetch_add(1, memory_order_relaxed);
        return nullptr;
    }
    eit->second.last_access_ms = now_ms();
    eit->second.hit_count++;
    shard.policy->touch(id);
    hits_.fetch_add(1, memory_order_relaxed);
    return make_shared<const string>(eit->second.response);
}

EntryId Storage::resolve(uint64_t namespace_id, const string& query) {
    auto& shard = shard_ref(namespace_id);
    shared_lock<shared_mutex> lk(shard.lock);
    string canonical = canonicalize(query);
    auto it = shard.exact_index.find(ExactKey{namespace_id, canonical});
    if (it == shard.exact_index.end()) {
        return 0;
    }
    return it->second;
}

optional<SemResult> Storage::semsearch(const vector<float>& query_embedding,
                                       uint64_t namespace_id,
                                       double threshold, size_t top_k) {
    auto& shard = shard_ref(namespace_id);
    shared_lock<shared_mutex> lk(shard.lock);
    semantic_searches_.fetch_add(1, memory_order_relaxed);

    if (shard.entries.empty() || !shard.vector_index ||
        shard.vector_index->empty()) {
        semantic_misses_.fetch_add(1, memory_order_relaxed);
        return nullopt;
    }

    size_t search_k = top_k * 10;
    if (search_k < 10) {
        search_k = 10;
    }

    auto candidates = shard.vector_index->search(query_embedding.data(),
                                                 query_embedding.size(),
                                                 search_k);

    uint64_t now = now_ms();
    for (auto& cand : candidates) {
        auto it = shard.entries.find(cand.id);
        if (it == shard.entries.end()) {
            continue;
        }
        auto& entry = it->second;
        if (entry.namespace_id != namespace_id) {
            continue;
        }
        if (entry.expires_at_ms != 0 && entry.expires_at_ms <= now) {
            continue;
        }
        double sim = cand.similarity();
        if (sim < threshold) {
            continue;
        }

        lk.unlock();
        unique_lock<shared_mutex> ulk(shard.lock);
        entry.last_access_ms = now;
        entry.hit_count++;
        shard.policy->touch(cand.id);
        semantic_hits_.fetch_add(1, memory_order_relaxed);
        hits_.fetch_add(1, memory_order_relaxed);
        return SemResult{cand.id, entry.original_query, entry.response, sim};
    }

    semantic_misses_.fetch_add(1, memory_order_relaxed);
    return nullopt;
}

bool Storage::semdel(uint64_t namespace_id, EntryId id) {
    auto& shard = shard_ref(namespace_id);
    unique_lock<shared_mutex> lk(shard.lock);
    auto eit = shard.entries.find(id);
    if (eit == shard.entries.end() || eit->second.namespace_id != namespace_id) {
        return false;
    }
    shard.exact_index.erase(ExactKey{namespace_id, eit->second.canonical_query});
    shard.value_bytes -= eit->second.response.size();
    shard.policy->remove(id);
    ttl_.erase(id);
    if (shard.vector_index) {
        shard.vector_index->remove(id);
    }
    shard.entries.erase(eit);
    return true;
}

nlohmann::json Storage::metrics() const {
    nlohmann::json j;
    size_t total_keys = 0;
    size_t used = 0;
    size_t value_bytes = 0;
    size_t largest = 0;

    for (const auto& shard : shards_) {
        shared_lock<shared_mutex> lk(shard->lock);
        total_keys += shard->entries.size();
        used += shard->policy->memory_used();
        value_bytes += shard->value_bytes;
        largest = max(largest, shard->largest_item);
    }

    j["total_keys"] = total_keys;
    j["memory_bytes"] = used;
    j["used_memory"] = used;
    j["used_memory_peak"] = used;
    j["memory_limit"] = memory_limit_;
    j["max_value_size"] = max_value_size_;
    j["hits"] = hits_.load(memory_order_relaxed);
    j["misses"] = misses_.load(memory_order_relaxed);
    j["evictions"] = evictions_.load(memory_order_relaxed);
    j["expired_keys"] = expired_keys_.load(memory_order_relaxed);
    j["semantic_hits"] = semantic_hits_.load(memory_order_relaxed);
    j["semantic_misses"] = semantic_misses_.load(memory_order_relaxed);
    j["semantic_searches"] = semantic_searches_.load(memory_order_relaxed);
    j["avg_item_bytes"] = total_keys == 0 ? 0 : value_bytes / total_keys;
    j["largest_item_bytes"] = largest;
    double total =
        static_cast<double>(hits_.load(memory_order_relaxed) +
                            misses_.load(memory_order_relaxed));
    j["hit_rate"] = total > 0.0
                        ? static_cast<double>(hits_.load(memory_order_relaxed)) /
                              total
                        : 0.0;
    j["miss_rate"] = total > 0.0
                         ? static_cast<double>(misses_.load(memory_order_relaxed)) /
                               total
                         : 0.0;
    return j;
}

void Storage::evict_if_over_limit(CacheShard& shard) {
    while (!shard.entries.empty() &&
           shard.policy->memory_used() > memory_limit_) {
        EntryId victim = shard.policy->evict();
        if (victim == 0) {
            break;
        }
        auto it = shard.entries.find(victim);
        if (it == shard.entries.end()) {
            break;
        }
        shard.exact_index.erase(
            ExactKey{it->second.namespace_id, it->second.canonical_query});
        shard.value_bytes -= it->second.response.size();
        ttl_.erase(victim);
        if (shard.vector_index) {
            shard.vector_index->remove(victim);
        }
        shard.entries.erase(it);
        evictions_.fetch_add(1, memory_order_relaxed);
    }
}

void Storage::expire_key(EntryId id) {
    for (auto& shard : shards_) {
        unique_lock<shared_mutex> lk(shard->lock);
        auto it = shard->entries.find(id);
        if (it == shard->entries.end()) {
            continue;
        }
        shard->exact_index.erase(
            ExactKey{it->second.namespace_id, it->second.canonical_query});
        shard->value_bytes -= it->second.response.size();
        shard->policy->remove(id);
        if (shard->vector_index) {
            shard->vector_index->remove(id);
        }
        shard->entries.erase(it);
        expired_keys_.fetch_add(1, memory_order_relaxed);
        return;
    }
}
