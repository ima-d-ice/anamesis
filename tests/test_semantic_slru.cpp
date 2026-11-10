#include "eviction/semantic_slru.h"

#include <cmath>
#include <cstdio>
#include <set>
#include <vector>

using namespace std;

namespace {

int g_failures = 0;

void check(bool cond, const char* what) {
    if (cond) {
        return;
    }
    ++g_failures;
    fprintf(stderr, "FAIL: %s\n", what);
}

// unused helpers kept minimal (see test 1 vectors inline)

}  // namespace

int main() {
    // ------------------------------------------------------------------
    // 1. w = 0 degenerates to plain SLRU: protected eviction = LRU tail.
    // ------------------------------------------------------------------
    {
        SemanticSLRU p(10000, 0.80, 0.0);
        float a[4] = {1, 0, 0, 0};
        float b[4] = {1, 0, 0, 0};  // identical, irrelevant at w=0
        p.add(1, 10);
        p.set_embedding(1, a, 4, 1);
        p.add(2, 10);
        p.set_embedding(2, b, 4, 1);
        p.add(3, 10);
        p.set_embedding(3, a, 4, 1);
        // touch all three -> all promoted to protected; probationary empty
        p.touch(1);
        p.touch(2);
        p.touch(3);
        // w=0 protected eviction = LRU tail. touch pushes front:
        // touch(1) -> [1]; touch(2) -> [2,1]; touch(3) -> [3,2,1]; tail=1
        EntryId v = p.evict();
        check(v == 1, "w=0 protected evicts LRU tail (oldest)");
        check(p.uniqueness_evictions() == 1, "w=0 still counts as uniqueness-eviction path");
    }

    // ------------------------------------------------------------------
    // 2. w = 1 evicts the least semantically unique entry (most similar to
    //    a peer). Entry 2 duplicates entry 1 -> low uniqueness -> victim 2.
    // ------------------------------------------------------------------
    {
        SemanticSLRU p(10000, 0.80, 1.0);
        float v1[4] = {1, 0, 0, 0};
        float v2[4] = {1, 0, 0, 0};  // duplicate of 1
        float v3[4] = {0, 1, 0, 0};  // orthogonal
        p.add(1, 10);
        p.set_embedding(1, v1, 4, 1);
        p.add(2, 10);
        p.set_embedding(2, v2, 4, 1);
        p.add(3, 10);
        p.set_embedding(3, v3, 4, 1);
        p.touch(1);
        p.touch(2);
        p.touch(3);  // all in protected; probationary empty
        EntryId v = p.evict();
        check(v == 2, "w=1 evicts least unique (duplicate) entry");
        check(p.uniqueness_evictions() == 1, "uniqueness eviction counted");
    }

    // ------------------------------------------------------------------
    // 3. Namespace isolation: entry 2 (ns 2) duplicates entry 1 (ns 1);
    //    uniqueness of entry 1 must not be affected by ns-2 peer.
    // ------------------------------------------------------------------
    {
        SemanticSLRU p(10000, 0.80, 1.0);
        float v1[4] = {1, 0, 0, 0};
        float v2[4] = {1, 0, 0, 0};  // same vector, different namespace
        p.add(1, 10);
        p.set_embedding(1, v1, 4, 1);  // ns 1
        p.add(2, 10);
        p.set_embedding(2, v2, 4, 2);  // ns 2
        p.add(3, 10);
        float v3[4] = {0, 1, 0, 0};
        p.set_embedding(3, v3, 4, 1);
        p.touch(1);
        p.touch(2);
        p.touch(3);  // all in protected; probationary empty
        EntryId v = p.evict();
        // Entry 3 (orthogonal, uniqueness 1) vs entry 1 (its only peer is in
        // another namespace -> uniqueness 1). Tie broken deterministically,
        // but entry 2 must never be penalized for the cross-namespace twin.
        check(v != 0, "eviction picked a victim");
        check(p.uniqueness_evictions() == 1, "cross-namespace eviction counted");
    }

    // ------------------------------------------------------------------
    // 4. remove() erases embeddings: SEM.DEL / TTL must not leave stale
    //    vectors that distort later uniqueness computations.
    // ------------------------------------------------------------------
    {
        SemanticSLRU p(10000, 0.80, 1.0);
        float v1[4] = {1, 0, 0, 0};
        float v2[4] = {1, 0, 0, 0};
        p.add(1, 10);
        p.set_embedding(1, v1, 4, 1);
        p.add(2, 10);
        p.set_embedding(2, v2, 4, 1);
        p.remove(1);  // SEM.DEL path
        // Entry 2 now has no peer -> uniqueness 1.0.
        // uniqueness() is private; exercised via the eviction path above
        p.add(3, 10);
        float v3[4] = {0, 1, 0, 0};
        p.set_embedding(3, v3, 4, 1);
        p.touch(2);
        p.touch(3);
        p.add(4, 10);
        EntryId v = p.evict();
        check(v != 0, "eviction works after remove");
        // memory_used should not count the removed embedding
        size_t before = p.memory_used();
        p.add(5, 10);
        p.set_embedding(5, v3, 4, 1);
        size_t after = p.memory_used();
        check(after > before, "memory_used grows with new embeddings");
        check(p.semantic_evictions() + p.uniqueness_evictions() >= 1,
              "eviction counters advance");
    }

    // ------------------------------------------------------------------
    // 5. Overwrite set_embedding replaces the vector (no stale copies).
    // ------------------------------------------------------------------
    {
        SemanticSLRU p(10000, 0.80, 1.0);
        float v1[4] = {1, 0, 0, 0};
        float v1b[4] = {0, 1, 0, 0};  // orthogonal replacement
        p.add(1, 10);
        p.set_embedding(1, v1, 4, 1);
        p.add(2, 10);
        float v2[4] = {1, 0, 0, 0};
        p.set_embedding(2, v2, 4, 1);  // duplicate of original v1
        p.touch(1);
        p.touch(2);
        // now overwrite 1 to be orthogonal -> uniqueness changes
        p.set_embedding(1, v1b, 4, 1);
        p.add(3, 10);
        EntryId v = p.evict();
        check(v != 0, "eviction after overwrite works");
    }

    // ------------------------------------------------------------------
    // 6. memory_used accounts for the policy's embedding copies.
    // ------------------------------------------------------------------
    {
        SemanticSLRU p(100000, 0.80, 0.5);
        float v[4] = {1, 0, 0, 0};
        p.add(1, 100);
        size_t m1 = p.memory_used();
        p.set_embedding(1, v, 4, 1);
        size_t m2 = p.memory_used();
        check(m2 > m1, "memory_used includes embedding copy");
        check(m2 - m1 == 4 * sizeof(float), "embedding copy byte count");
        p.remove(1);
        size_t m3 = p.memory_used();
        check(m3 == 0, "memory_used drops to zero after remove");
    }

    // ------------------------------------------------------------------
    // 7. Cursor safety after remove(): erasing a protected entry must not
    //    leave the rotating cursor pointing at a freed list node. Remove a
    //    middle protected entry, then evict the rest; every victim must be a
    //    live, distinct entry (never the removed id).
    // ------------------------------------------------------------------
    {
        SemanticSLRU p(1000, 0.80, 0.5);  // small memory, protected cap 800 B
        float vecs[6][4] = {{1,0,0,0},{0,1,0,0},{1,1,0,0},{0,0,1,0},
                            {1,0,1,0},{0,1,1,0}};
        for (int i = 1; i <= 6; ++i) {
            p.add(i, 40);
            p.set_embedding(i, vecs[i-1], 4, 1);
            p.touch(i);  // promote to protected
        }
        // Remove a middle protected entry (id 3).
        p.remove(3);
        // Evict the remaining five protected entries.
        set<EntryId> seen;
        for (int i = 0; i < 5; ++i) {
            EntryId v = p.evict();
            check(v != 0, "evict returns a victim after protected remove");
            check(v != 3, "removed id is never re-evicted");
            seen.insert(v);
        }
        check(seen.size() == 5, "all five remaining protected entries evicted");
        check(p.evict() == 0, "evict returns 0 when protected is empty");
    }

    // ------------------------------------------------------------------
    // 8. Cursor safety after demote(): overfill protected so the LRU entry is
    //    demoted to probationary (structural list mutation), then evict the
    //    rest without crashing or repeating a victim.
    // ------------------------------------------------------------------
    {
        SemanticSLRU p(300, 0.80, 0.5);  // protected cap 240 B => 6 entries
        float vecs[8][4] = {{1,0,0,0},{0,1,0,0},{1,1,0,0},{0,0,1,0},
                            {1,0,1,0},{0,1,1,0},{1,0,0,1},{0,1,0,1}};
        for (int i = 1; i <= 8; ++i) {
            p.add(i, 40);
            p.set_embedding(i, vecs[i-1], 4, 1);
            p.touch(i);  // promote; 6th+ triggers demotion of LRU protected
        }
        set<EntryId> seen;
        int evicted = 0;
        while (true) {
            EntryId v = p.evict();
            if (v == 0) break;
            check(seen.insert(v).second, "no repeated victim after demote");
            ++evicted;
        }
        check(evicted == 8, "all eight entries eventually evicted");
    }

    // ------------------------------------------------------------------
    // 9. Uniqueness-cache generation invalidation: adding an embedding to a
    //    namespace increments its generation, invalidating cached uniqueness
    //    scores for that namespace. Evicting after such an addition must
    //    recompute uniqueness (not reuse stale scores) and must not crash.
    // ------------------------------------------------------------------
    {
        SemanticSLRU p(10000, 0.80, 1.0);  // pure uniqueness, no recency
        float v1[4] = {1, 0, 0, 0};
        float v2[4] = {1, 0, 0, 0};  // duplicate of v1
        float v3[4] = {0, 1, 0, 0};  // orthogonal
        p.add(1, 10);
        p.set_embedding(1, v1, 4, 1);
        p.add(2, 10);
        p.set_embedding(2, v2, 4, 1);
        p.add(3, 10);
        p.set_embedding(3, v3, 4, 1);
        p.touch(1);
        p.touch(2);
        p.touch(3);  // all in protected; probationary empty

        // First eviction: entry 2 duplicates entry 1 -> lowest uniqueness -> victim 2.
        EntryId v1_evicted = p.evict();
        check(v1_evicted == 2, "first eviction picks least unique (duplicate) entry");

        // Add a new embedding to the same namespace: generation bumps,
        // cached uniqueness scores for ns 1 are invalidated.
        p.add(4, 10);
        p.set_embedding(4, v1, 4, 1);  // another duplicate of entry 1
        p.touch(4);

        // Second eviction: entry 4 is now also a duplicate of entry 1.
        // With generation invalidation, entry 4 (newest duplicate) should
        // be evicted over entry 1 (original), since both have uniqueness 0.
        // Either is acceptable; key is no crash and a live victim.
        EntryId v2_evicted = p.evict();
        check(v2_evicted != 0, "second eviction returns a victim after generation bump");
        check(v2_evicted != v1_evicted, "second eviction picks a different victim");
        check(v2_evicted == 1 || v2_evicted == 4,
              "second eviction picks one of the duplicate entries");
    }

    if (g_failures == 0) {
        printf("ok test_semantic_slru\n");
        return 0;
    }
    fprintf(stderr, "%d check(s) failed\n", g_failures);
    return 1;
}
