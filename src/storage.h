#pragma once

#include "eviction/policy.h"
#include "index/vector_index.hpp"
#include "third_party/nlohmann/json.hpp"
#include "ttl.h"

#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <vector>

using EntryId = std::uint64_t;
using Blob = std::shared_ptr<const std::string>;

enum class CacheSegment {
    PROBATION,
    PROTECTED
};

struct CacheEntry {
    EntryId id = 0;
    std::uint64_t namespace_id = 0;
    std::string canonical_query;
    std::string original_query;
    std::string response;
    std::string content_type;
    std::vector<float> embedding;
    std::uint64_t created_at_ms = 0;
    std::uint64_t expires_at_ms = 0;
    std::uint64_t last_access_ms = 0;
    std::uint64_t hit_count = 0;
    std::uint64_t expiry_generation = 0;
    std::size_t memory_bytes = 0;
    CacheSegment segment = CacheSegment::PROBATION;
};

struct ExactKey {
    std::uint64_t namespace_id;
    std::string canonical_query;
    bool operator==(const ExactKey& other) const noexcept {
        return namespace_id == other.namespace_id &&
               canonical_query == other.canonical_query;
    }
};

struct ExactKeyHash {
    std::size_t operator()(const ExactKey& k) const noexcept {
        std::size_t h = std::hash<std::uint64_t>{}(k.namespace_id);
        h ^= std::hash<std::string>{}(k.canonical_query) + 0x9e3779b9 +
             (h << 6) + (h >> 2);
        return h;
    }
};

struct SemResult {
    EntryId id;
    std::string key;
    std::string value;
    double similarity;
};

// One shard of the cache: independent lock, entry map, exact index, eviction
// policy, and vector index. Routes by hash(namespace_id) % shard_count.
struct CacheShard {
    mutable std::shared_mutex lock;
    std::unordered_map<EntryId, CacheEntry> entries;
    std::unordered_map<ExactKey, EntryId, ExactKeyHash> exact_index;
    std::unique_ptr<EvictionPolicy> policy;
    std::unique_ptr<IVectorIndex> vector_index;
    std::size_t memory_bytes = 0;
    std::size_t value_bytes = 0;
    std::size_t largest_item = 0;

    CacheShard() = default;
};

class Storage {
public:
    static constexpr std::size_t kDefaultMemoryLimit = std::size_t{1024} * 1024 * 1024;
    static constexpr std::size_t kDefaultMaxValueSize = std::size_t{512} * 1024 * 1024;
    static constexpr std::size_t kMaxKeySize = std::size_t{1} * 1024 * 1024;
    static constexpr std::size_t kEntryOverhead = 128;

    explicit Storage(std::size_t memory_limit = kDefaultMemoryLimit,
                     std::size_t max_value_size = kDefaultMaxValueSize,
                     const char* policy_name = "slru",
                     int shard_count = 1);
    ~Storage() = default;

    Storage(const Storage&) = delete;
    Storage& operator=(const Storage&) = delete;

    bool set(std::string key, std::string value, int ttl_sec);
    bool can_store(const std::string& key, const std::string& value) const;
    Blob get(const std::string& key);
    bool del(const std::string& key);

    EntryId semset(std::uint64_t namespace_id, std::string query, std::string value,
                   std::vector<float> embedding, int ttl_sec);
    Blob semget(std::uint64_t namespace_id, const std::string& query);
    EntryId resolve(std::uint64_t namespace_id, const std::string& query);
    std::optional<SemResult> semsearch(const std::vector<float>& query_embedding,
                                       std::uint64_t namespace_id,
                                       double threshold, std::size_t top_k);
    bool semdel(std::uint64_t namespace_id, EntryId id);

    nlohmann::json metrics() const;

    std::size_t memory_limit() const { return memory_limit_; }
    std::size_t max_value_size() const { return max_value_size_; }
    EntryId next_entry_id() { return next_id_++; }
    int shard_count() const { return static_cast<int>(shards_.size()); }

private:
    int shard_for(std::uint64_t namespace_id) const;
    CacheShard& shard_ref(std::uint64_t namespace_id);
    const CacheShard& shard_ref(std::uint64_t namespace_id) const;

    std::size_t entry_bytes(const CacheEntry& entry) const;
    void evict_if_over_limit(CacheShard& shard);
    void expire_key(EntryId id);
    std::size_t used_memory() const;
    std::string canonicalize(const std::string& query) const;
    std::uint64_t now_ms() const;

    std::vector<std::unique_ptr<CacheShard>> shards_;
    std::size_t memory_limit_;
    std::size_t max_value_size_;
    std::atomic<EntryId> next_id_{1};

    // Aggregate counters (approximate, lock-free)
    std::atomic<uint64_t> hits_{0};
    std::atomic<uint64_t> misses_{0};
    std::atomic<uint64_t> evictions_{0};
    std::atomic<uint64_t> expired_keys_{0};
    std::atomic<uint64_t> semantic_hits_{0};
    std::atomic<uint64_t> semantic_misses_{0};
    std::atomic<uint64_t> semantic_searches_{0};

    TTLManager ttl_;
};
