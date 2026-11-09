#include "vector_index.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <unordered_map>

using namespace std;

namespace {

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

float cosine_similarity(const float* a, const float* b, size_t dim) {
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (size_t i = 0; i < dim; ++i) {
        double av = static_cast<double>(a[i]);
        double bv = static_cast<double>(b[i]);
        dot += av * bv;
        na += av * av;
        nb += bv * bv;
    }
    if (na == 0.0 || nb == 0.0) {
        return 0.0f;
    }
    return static_cast<float>(dot / (sqrt(na) * sqrt(nb)));
}

}  // namespace

class FlatVectorIndex : public IVectorIndex {
public:
    explicit FlatVectorIndex(size_t dimension) : dimension_(dimension) {}

    void insert(EntryId id, const float* data, size_t dimension) override {
        if (dimension != dimension_) {
            return;
        }
        vector<float> v(dimension);
        memcpy(v.data(), data, dimension * sizeof(float));
        normalize_vector(v.data(), dimension);
        vectors_[id] = std::move(v);
    }

    void remove(EntryId id) override {
        vectors_.erase(id);
    }

    vector<SearchResult> search(const float* query, size_t dimension,
                                size_t k) override {
        if (dimension != dimension_ || vectors_.empty()) {
            return {};
        }

        // Normalize query for consistent cosine similarity
        vector<float> normalized_query(query, query + dimension);
        normalize_vector(normalized_query.data(), dimension);

        vector<SearchResult> results;
        results.reserve(vectors_.size());
        for (auto& [id, vec] : vectors_) {
            float sim = cosine_similarity(normalized_query.data(), vec.data(), dimension_);
            results.push_back({id, 1.0f - sim});
        }

        sort(results.begin(), results.end(),
             [](const SearchResult& a, const SearchResult& b) {
                 return a.distance < b.distance;
             });
        if (results.size() > k) {
            results.resize(k);
        }
        return results;
    }

    size_t size() const override { return vectors_.size(); }
    bool empty() const override { return vectors_.empty(); }
    void clear() override { vectors_.clear(); }
    void rebuild() override {}  // flat index is always in sync
    size_t memory_bytes() const override {
        return vectors_.size() * (dimension_ * sizeof(float) + sizeof(EntryId));
    }

private:
    size_t dimension_;
    unordered_map<EntryId, vector<float>> vectors_;
};

unique_ptr<IVectorIndex> make_flat_index(size_t dimension) {
    return make_unique<FlatVectorIndex>(dimension);
}
