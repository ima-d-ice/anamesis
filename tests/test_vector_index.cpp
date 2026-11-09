#include "index/vector_index.hpp"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>

using namespace std;

int main() {
    // Test FlatVectorIndex
    {
        auto index = make_flat_index(3);

        // Insert vectors
        float v1[] = {1.0f, 0.0f, 0.0f};
        float v2[] = {0.0f, 1.0f, 0.0f};
        float v3[] = {0.0f, 0.0f, 1.0f};

        index->insert(1, v1, 3);
        index->insert(2, v2, 3);
        index->insert(3, v3, 3);

        assert(index->size() == 3);
        assert(!index->empty());

        // Search for v1 - should return itself first (similarity = 1.0)
        auto results = index->search(v1, 3, 3);
        assert(results.size() == 3);
        assert(results[0].id == 1);
        assert(results[0].similarity() > 0.99f);  // self-similarity

        // Search for orthogonal vector - should return low similarity
        float v4[] = {1.0f, 1.0f, 0.0f};
        auto results2 = index->search(v4, 3, 2);
        assert(results2.size() == 2);

        // Remove and verify
        index->remove(1);
        assert(index->size() == 2);
        auto results3 = index->search(v1, 3, 2);
        // v1 is removed, should not be in results
        bool found_v1 = false;
        for (auto& r : results3) {
            if (r.id == 1) found_v1 = true;
        }
        assert(!found_v1);

        index->clear();
        assert(index->size() == 0);
        assert(index->empty());
    }

    // Test HNSWVectorIndex
    {
        auto index = make_hnsw_index(3, 16, 200, 64);

        // Insert vectors
        float v1[] = {1.0f, 0.0f, 0.0f};
        float v2[] = {0.0f, 1.0f, 0.0f};
        float v3[] = {0.0f, 0.0f, 1.0f};

        index->insert(1, v1, 3);
        index->insert(2, v2, 3);
        index->insert(3, v3, 3);

        assert(index->size() == 3);
        assert(!index->empty());

        // Search for v1 - should return itself first
        auto results = index->search(v1, 3, 3);
        assert(results.size() == 3);
        assert(results[0].id == 1);
        assert(results[0].similarity() > 0.99f);

        // Remove and verify
        index->remove(1);
        assert(index->size() == 2);
        auto results2 = index->search(v1, 3, 2);
        bool found_v1 = false;
        for (auto& r : results2) {
            if (r.id == 1) found_v1 = true;
        }
        assert(!found_v1);

        index->clear();
        assert(index->size() == 0);
        assert(index->empty());
    }

    // Test similarity ordering: similar vectors rank higher
    {
        auto index = make_flat_index(2);

        float a[] = {1.0f, 0.0f};
        float b[] = {0.9f, 0.1f};  // close to a
        float c[] = {0.0f, 1.0f};  // orthogonal to a

        index->insert(1, a, 2);
        index->insert(2, b, 2);
        index->insert(3, c, 2);

        auto results = index->search(a, 2, 3);
        assert(results.size() == 3);
        assert(results[0].id == 1);  // self
        assert(results[1].id == 2);  // close to self
        assert(results[2].id == 3);  // orthogonal
        assert(results[1].similarity() > results[2].similarity());
    }

    printf("ok test_vector_index\n");
    return 0;
}
