#include "vector_index.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <unordered_map>
#include <vector>
#include <stdexcept>
#include <string>

#include "hnswlib.h"

using namespace std;

static void normalize_vector(float* data, size_t dim) {
    double norm = 0.0;
    for (size_t i = 0; i < dim; ++i) {
        norm += static_cast<double>(data[i]) * static_cast<double>(data[i]);
    }
    norm = sqrt(norm);
    if (norm > 0.0) {
        double inv = 1.0 / norm;
        for (size_t i = 0; i < dim; ++i) {
            data[i] = static_cast<float>(static_cast<double>(data[i]) * inv);
        }
    }
}

class HNSWVectorIndex : public IVectorIndex {
public:
    HNSWVectorIndex(size_t dimension, int M, int ef_construction,
                    int ef_search)
        : dimension_(dimension),
          space_(new hnswlib::InnerProductSpace(dimension)),
          index_(new hnswlib::HierarchicalNSW<float>(
              space_.get(),
              1000000,  // max elements
              M,
              ef_construction)),
          ef_search_(ef_search) {}

    ~HNSWVectorIndex() override = default;

    void insert(EntryId id, const float* data, size_t dimension) override {
        if (dimension != dimension_) {
            return;
        }
        // Normalize for cosine similarity via inner product space
        vector<float> normalized(data, data + dimension);
        normalize_vector(normalized.data(), dimension);
        // Use EntryId directly as hnswlib label
        hnswlib::labeltype label = static_cast<hnswlib::labeltype>(id);
        index_->addPoint(normalized.data(), label);
    }

    void remove(EntryId id) override {
        hnswlib::labeltype label = static_cast<hnswlib::labeltype>(id);
        index_->markDelete(label);
    }

    vector<SearchResult> search(const float* query, size_t dimension,
                                size_t k) override {
        if (dimension != dimension_ || index_->getCurrentElementCount() == 0) {
            return {};
        }

        index_->setEf(static_cast<size_t>(ef_search_));

        // Normalize query for cosine similarity
        vector<float> normalized_query(query, query + dimension);
        normalize_vector(normalized_query.data(), dimension);

        // Use searchKnn which returns a priority_queue of (dist, label) pairs
        // sorted by distance (max-heap, so top is furthest). We reverse to get
        // closest first.
        auto result = index_->searchKnn(normalized_query.data(), k);

        vector<SearchResult> results;
        results.reserve(result.size());
        while (!result.empty()) {
            auto& [dist, label] = result.top();
            EntryId id = static_cast<EntryId>(label);
            results.push_back({id, static_cast<float>(dist)});
            result.pop();
        }
        // Results come out furthest-first from max-heap; reverse for closest-first
        reverse(results.begin(), results.end());
        return results;
    }

    size_t size() const override {
        return index_->getCurrentElementCount() - index_->getDeletedCount();
    }

    bool empty() const override {
        return (index_->getCurrentElementCount() - index_->getDeletedCount()) == 0;
    }

    void clear() override {
        index_.reset(new hnswlib::HierarchicalNSW<float>(
            space_.get(),
            1000000,
            16,   // M
            200));  // ef_construction
    }

    void rebuild() override {
        // HNSW handles deleted points internally; no explicit rebuild needed
    }

    size_t memory_bytes() const override {
        return index_->getCurrentElementCount() * (dimension_ * sizeof(float) +
                                                   16 * sizeof(hnswlib::labeltype));
    }

private:
    size_t dimension_;
    unique_ptr<hnswlib::SpaceInterface<float>> space_;
    unique_ptr<hnswlib::HierarchicalNSW<float>> index_;
    int ef_search_;
};

unique_ptr<IVectorIndex> make_hnsw_index(size_t dimension, int M,
                                         int ef_construction,
                                         int ef_search) {
    return make_unique<HNSWVectorIndex>(dimension, M, ef_construction,
                                        ef_search);
}
