#include "storage.h"
#include "embedding.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace std;

namespace {

void test_cosine_identical() {
    vector<float> a = {1.0f, 0.0f, 0.0f};
    vector<float> b = {1.0f, 0.0f, 0.0f};
    assert(embedding::cosine_similarity(a, b) == 1.0);
}

void test_cosine_opposite() {
    vector<float> a = {1.0f, 0.0f, 0.0f};
    vector<float> b = {-1.0f, 0.0f, 0.0f};
    assert(embedding::cosine_similarity(a, b) == -1.0);
}

void test_cosine_orthogonal() {
    vector<float> a = {1.0f, 0.0f, 0.0f};
    vector<float> b = {0.0f, 1.0f, 0.0f};
    assert(embedding::cosine_similarity(a, b) == 0.0);
}

void test_cosine_zero() {
    vector<float> a = {0.0f, 0.0f, 0.0f};
    vector<float> b = {1.0f, 0.0f, 0.0f};
    assert(embedding::cosine_similarity(a, b) == 0.0);
}

void test_top_k_ordering() {
    vector<vector<float>> vecs = {
        {1.0f, 0.0f, 0.0f},
        {0.0f, 1.0f, 0.0f},
        {-1.0f, 0.0f, 0.0f},
        {0.5f, 0.5f, 0.0f},
    };
    vector<float> query = {1.0f, 0.0f, 0.0f};
    auto scored = embedding::top_k(vecs, query, 3);
    assert(scored.size() == 3);
    assert(scored[0].first == 0);
    assert(scored[0].second == 1.0);
    assert(scored[1].first == 3);
    assert(scored[2].first == 1);
}

void test_semset_semget_roundtrip() {
    Storage storage(8 * 1024 * 1024, 4 * 1024 * 1024);
    vector<float> emb = {0.1f, 0.2f, 0.3f};
    EntryId id = storage.semset(0, "legacy", "q1", "answer1", emb, 0);
    assert(id != 0);
    assert(storage.resolve(0, "legacy", "q1") == id);
    Blob got = storage.semget(0, "legacy", "q1");
    assert(got != nullptr);
    assert(*got == "answer1");
}

void test_semsearch_hit() {
    Storage storage(8 * 1024 * 1024, 4 * 1024 * 1024);
    vector<float> emb1 = {1.0f, 0.0f, 0.0f};
    vector<float> emb2 = {0.0f, 1.0f, 0.0f};
    assert(storage.semset(0, "legacy", "q1", "answer1", emb1, 0) != 0);
    assert(storage.semset(0, "legacy", "q2", "answer2", emb2, 0) != 0);

    auto result = storage.semsearch({1.0f, 0.0f, 0.0f}, 0, "legacy", 0.9, 5);
    assert(result.has_value());
    assert(result->key == "q1");
    assert(result->value == "answer1");
    assert(result->similarity >= 0.9);
}

void test_semsearch_miss_below_threshold() {
    Storage storage(8 * 1024 * 1024, 4 * 1024 * 1024);
    vector<float> emb1 = {1.0f, 0.0f, 0.0f};
    assert(storage.semset(0, "legacy", "q1", "answer1", emb1, 0) != 0);

    auto result = storage.semsearch({0.0f, 1.0f, 0.0f}, 0, "legacy", 0.9, 5);
    assert(!result.has_value());
}

void test_semsearch_empty() {
    Storage storage(8 * 1024 * 1024, 4 * 1024 * 1024);
    auto result = storage.semsearch({1.0f, 0.0f, 0.0f}, 0, "legacy", 0.9, 5);
    assert(!result.has_value());
}

void test_semdel() {
    Storage storage(8 * 1024 * 1024, 4 * 1024 * 1024);
    vector<float> emb = {0.1f, 0.2f, 0.3f};
    EntryId id = storage.semset(0, "legacy", "q1", "answer1", emb, 0);
    assert(id != 0);
    assert(storage.semdel(0, id));
    assert(storage.semget(0, "legacy", "q1") == nullptr);
    assert(!storage.semdel(0, id));
}

void test_semsearch_excludes_empty_embedding() {
    Storage storage(8 * 1024 * 1024, 4 * 1024 * 1024);
    assert(storage.set("plain", "plain answer", 0));
    vector<float> emb = {1.0f, 0.0f, 0.0f};
    assert(storage.semset(0, "legacy", "q1", "answer1", emb, 0) != 0);
    auto result = storage.semsearch({1.0f, 0.0f, 0.0f}, 0, "legacy", 0.9, 5);
    assert(result.has_value());
    assert(result->key == "q1");
}

void test_metrics_semantic() {
    Storage storage(8 * 1024 * 1024, 4 * 1024 * 1024);
    vector<float> emb = {1.0f, 0.0f, 0.0f};
    storage.semset(0, "legacy", "q1", "answer1", emb, 0);
    storage.semsearch({1.0f, 0.0f, 0.0f}, 0, "legacy", 0.9, 5);
    storage.semsearch({0.0f, 1.0f, 0.0f}, 0, "legacy", 0.9, 5);

    auto m = storage.metrics();
    assert(m["semantic_searches"].get<size_t>() == 2);
    assert(m["semantic_hits"].get<size_t>() == 1);
    assert(m["semantic_misses"].get<size_t>() == 1);
}

void test_namespace_isolation() {
    Storage storage(8 * 1024 * 1024, 4 * 1024 * 1024);
    vector<float> emb = {1.0f, 0.0f, 0.0f};
    storage.semset(1, "legacy", "q1", "answer-ns1", emb, 0);
    storage.semset(2, "legacy", "q1", "answer-ns2", emb, 0);

    Blob a = storage.semget(1, "legacy", "q1");
    Blob b = storage.semget(2, "legacy", "q1");
    assert(a != nullptr);
    assert(b != nullptr);
    assert(*a == "answer-ns1");
    assert(*b == "answer-ns2");
    assert(storage.semget(1, "legacy", "q2") == nullptr);
}

void test_semset_overwrite_syncs_index() {
    Storage storage(8 * 1024 * 1024, 4 * 1024 * 1024);
    vector<float> emb1 = {1.0f, 0.0f, 0.0f};
    vector<float> emb2 = {0.0f, 1.0f, 0.0f};  // orthogonal to emb1
    EntryId id = storage.semset(0, "legacy", "q1", "answer1", emb1, 0);
    assert(id != 0);

    // Overwrite q1 with a different embedding and value.
    EntryId id2 = storage.semset(0, "legacy", "q1", "answer1-new", emb2, 0);
    assert(id2 == id);  // same entry, updated in place

    // The ANN index must now return q1 for emb2 (similarity 1.0).
    auto result = storage.semsearch(emb2, 0, "legacy", 0.9, 5);
    assert(result.has_value());
    assert(result->key == "q1");
    assert(result->value == "answer1-new");
    assert(result->similarity >= 0.9);

    // The old embedding must no longer match q1 (now orthogonal).
    auto result2 = storage.semsearch(emb1, 0, "legacy", 0.9, 5);
    assert(!result2.has_value() || result2->key != "q1");
}

void test_semantic_margin() {
    Storage storage(8 * 1024 * 1024, 4 * 1024 * 1024);
    // Two embeddings very close to each other (similarity ~0.99).
    vector<float> emb1 = {1.0f, 0.0f, 0.0f};
    vector<float> emb2 = {0.99f, 0.1f, 0.0f};  // close to emb1
    assert(storage.semset(0, "legacy", "q1", "answer1", emb1, 0) != 0);
    assert(storage.semset(0, "legacy", "q2", "answer2", emb2, 0) != 0);

    // Query equal to emb1: top-1 = q1 (sim=1.0), top-2 = q2 (sim~0.99).
    // Gap ~0.01 < margin 0.05 → fail-closed (miss).
    auto result = storage.semsearch(emb1, 0, "legacy", 0.5, 5, 0.05);
    assert(!result.has_value());

    // With margin 0.0, should hit q1.
    auto result2 = storage.semsearch(emb1, 0, "legacy", 0.5, 5, 0.0);
    assert(result2.has_value());
    assert(result2->key == "q1");

    // Query far from both: no hit.
    vector<float> emb3 = {0.0f, 0.0f, 1.0f};
    auto result3 = storage.semsearch(emb3, 0, "legacy", 0.5, 5, 0.05);
    assert(!result3.has_value());
}

}  // namespace

int main() {
    test_cosine_identical();
    test_cosine_opposite();
    test_cosine_orthogonal();
    test_cosine_zero();
    test_top_k_ordering();
    test_semset_semget_roundtrip();
    test_semsearch_hit();
    test_semsearch_miss_below_threshold();
    test_semsearch_empty();
    test_semdel();
    test_semsearch_excludes_empty_embedding();
    test_metrics_semantic();
    test_namespace_isolation();
    test_semset_overwrite_syncs_index();
    test_semantic_margin();
    printf("ok test_semantic\n");
    return 0;
}
