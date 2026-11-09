#include "eviction/slru.h"

#include <cassert>
#include <cstdio>
#include <cstdint>

using namespace std;

int main() {
    // Test basic SLRU operations
    {
        SLRU slru(1000, 0.80);

        // Add entries (each add pushes to front, so back is oldest)
        slru.add(1, 100);
        slru.add(2, 200);
        slru.add(3, 300);

        assert(slru.memory_used() == 600);
        assert(slru.probationary_bytes() == 600);
        assert(slru.protected_bytes() == 0);

        // All entries in probationary. Evict from back (oldest = entry 1).
        assert(slru.evict() == 1);  // LRU eviction from probationary (oldest)
        assert(slru.memory_used() == 500);
        assert(slru.probationary_bytes() == 500);
    }

    // Test promotion to protected on touch
    {
        SLRU slru(1000, 0.80);

        slru.add(1, 100);
        slru.add(2, 200);
        slru.add(3, 300);

        // Touch entry 1 - should promote to protected
        slru.touch(1);

        // Entry 1 should now be in protected
        // Probationary should have 2 and 3 (500 bytes)
        // Protected should have 1 (100 bytes)
        assert(slru.probationary_bytes() == 500);
        assert(slru.protected_bytes() == 100);

        // Evict from probationary first (entry 2 is LRU since entry 3 was added last)
        assert(slru.evict() == 2);  // LRU of probationary
        assert(slru.memory_used() == 400);
        assert(slru.probationary_bytes() == 300);
        assert(slru.protected_bytes() == 100);
    }

    // Test remove
    {
        SLRU slru(1000, 0.80);

        slru.add(1, 100);
        slru.add(2, 200);
        slru.remove(1);

        assert(slru.memory_used() == 200);
        assert(slru.evict() == 2);
        assert(slru.memory_used() == 0);
    }

    // Test eviction order: probationary before protected
    {
        SLRU slru(1000, 0.80);

        slru.add(1, 100);
        slru.add(2, 200);
        slru.touch(2);  // promote 2 to protected

        // 1 is in probationary, 2 is in protected
        // Evict from probationary first
        assert(slru.evict() == 1);
        assert(slru.memory_used() == 200);

        // Now only protected remains
        assert(slru.evict() == 2);
        assert(slru.memory_used() == 0);
    }

    // Test update size on re-add
    {
        SLRU slru(1000, 0.80);

        slru.add(1, 100);
        slru.add(1, 200);  // update size

        assert(slru.memory_used() == 200);
    }

    printf("ok test_slru\n");
    return 0;
}
