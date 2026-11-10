#pragma once

#include <cstddef>
#include <cstdint>

using EntryId = std::uint64_t;

class EvictionPolicy {
public:
    virtual ~EvictionPolicy() = default;

    virtual void touch(EntryId id) = 0;
    virtual void add(EntryId id, std::size_t size) = 0;
    virtual void remove(EntryId id) = 0;
    virtual EntryId evict() = 0;
    virtual std::size_t memory_used() const = 0;

    // Semantic-aware eviction counters (base policies return zero).
    virtual std::size_t semantic_evictions() const { return 0; }
    virtual std::size_t uniqueness_evictions() const { return 0; }

    // Optional: register an embedding for semantic-aware eviction.
    // namespace_id scopes similarity comparisons so entries never compete
    // across namespaces. Base implementation does nothing.
    virtual void set_embedding(EntryId id, const float* data,
                               std::size_t dimension,
                               uint64_t namespace_id = 0) {
        (void)id;
        (void)data;
        (void)dimension;
        (void)namespace_id;
    }
};
