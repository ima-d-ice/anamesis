#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

using EntryId = std::uint64_t;

struct SearchResult {
    EntryId id;
    float distance; // cosine distance: 0 = identical, 1 = opposite, 2 = orthogonal-neg
    float similarity() const { return 1.0f - distance; }
};

class IVectorIndex {
public:
    virtual ~IVectorIndex() = default;

    virtual void insert(EntryId id, const float* vector, std::size_t dimension) = 0;
    virtual void remove(EntryId id) = 0;
    virtual std::vector<SearchResult> search(const float* query,
                                             std::size_t dimension,
                                             std::size_t k) = 0;
    virtual std::size_t size() const = 0;
    virtual bool empty() const = 0;
    virtual void clear() = 0;
    virtual void rebuild() = 0;
    virtual std::size_t memory_bytes() const = 0;
};

std::unique_ptr<IVectorIndex> make_flat_index(std::size_t dimension);
std::unique_ptr<IVectorIndex> make_hnsw_index(std::size_t dimension,
                                              int M = 16,
                                              int ef_construction = 200,
                                              int ef_search = 64);
std::unique_ptr<IVectorIndex> make_vector_index(const std::string& backend,
                                                std::size_t dimension,
                                                int M = 16,
                                                int ef_construction = 200,
                                                int ef_search = 64);
