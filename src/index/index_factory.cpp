#include "vector_index.hpp"

#include <stdexcept>
#include <string>

using namespace std;

unique_ptr<IVectorIndex> make_vector_index(const string& backend,
                                           size_t dimension,
                                           int M,
                                           int ef_construction,
                                           int ef_search) {
    string b = backend;
    for (auto& c : b) {
        c = static_cast<char>(tolower(static_cast<unsigned char>(c)));
    }
    if (b == "flat") {
        return make_flat_index(dimension);
    }
    if (b == "hnsw" || b == "hnswlib") {
        return make_hnsw_index(dimension, M, ef_construction, ef_search);
    }
    throw runtime_error("unknown vector index backend: " + backend);
}
