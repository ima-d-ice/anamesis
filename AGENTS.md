# Anamesis Project Rules

## Current State (read first)

- **Project: anamesis** (Greek, ἀνάμνησις = recollection) — a production-level C++17 in-memory **semantic response cache** for customer-support chatbot / internal-docs RAG workloads. Sits in front of an RAG/LLM pipeline: semantic hit → skip the LLM; miss → RAG → `SEM.PUT` backfill. Target: 40–70% LLM call reduction.
- Folder `/Users/imad/Desktop/anamesis`; binary `./anamesis`; default AOF `anamesis.aof`; router binary `./anamesis-router`. **No git repo.**
- **Input model:** clients send raw **text** queries (`SEM.GET namespace query_text`). Anamesis orchestrates embedding generation via an external llama.cpp worker (`--embedding-worker-url`), then searches by meaning. No embedding inference inside the C++ core.
- **Strings only:** key/query = user query, value = LLM/RAG response. No binary-blob feature work.
- Core implemented + verified: `make clean && make test` → `ok test_storage`, `ok test_protocol`, `ok test_semantic`; `make` builds `./anamesis` (~375KB); live smoke of `PING`/`SEM.PUT`/`SEM.GET`/`SEM.SEARCH`/`SEM.DEL`, `/metrics`, `/health` passed; router + 2-node horizontal smoke passed.
- **Eviction:** SLRU (segmented LRU: probationary + protected) default, selectable at startup via `--policy slru|lru|lfu|sieve`. **No runtime policy switching.** Policy state lives only in the policy object, not `CacheEntry`. Eviction order: TTL-expired → invalidated → probationary LRU → protected LRU.
- **CacheEntry** = `EntryId` (uint64_t), namespace_id, canonical_query, original_query, response, content_type, `std::vector<float> embedding`, created/expires/last-access timestamps, hit_count, segment, memory_bytes. No size field — sizes computed via `entry_bytes()`.
- **Exact lookup:** `unordered_map<ExactKey, EntryId>` (namespace_id + canonical_query) → O(1) byte-identical repeat path, checked before ANN.
- **ANN search:** `IVectorIndex` abstraction. `FlatVectorIndex` (exact linear scan) wired by default; `HNSWVectorIndex` placeholder ready for hnswlib vendor. Per-shard index. Cosine similarity, `similarity = 1 - distance`.
- **Embeddings:** clients send as JSON arrays (`[1.0,0.0,0.0]`) or via gateway flow where Anamesis POSTs text to `--embedding-worker-url` and receives float32 blob. Persisted in AOF as Base64-packed little-endian float32 (`embedding_b64`). No ML deps in C++ core.
- **Threshold:** default 0.80 (`--semantic-threshold`), overridable per `SEM.SEARCH`. To be calibrated empirically per model — never hardcode a "final" threshold.
- **AOF:** JSON lines + base64, `fflush` after every write, `replay` on boot, skips torn lines. Logs `SEM.PUT`/`SEM.DEL` (namespace, query, response, embedding, TTL). No fsync, no compaction yet. Snapshots planned, not implemented.
- **Metrics:** Prometheus text at `GET /metrics`, `GET /health` on the admin port. Metric names keep the **`semcache_*`** prefix (hits, misses, semantic_hits, semantic_misses, semantic_searches, evictions, expired_keys, ann candidates, etc.). Admin binds `INADDR_ANY`.
- **Horizontal scaling:** separate `anamesis-router` binary (`src/router/proxy.cpp`). Consistent hash on `namespace_id` routes `SEM.*` commands to the owning Anamesis node. Non-SEM commands round-robin. Health-checks nodes via `GET /health`, drops unhealthy from ring. No cross-node state, no replication in v1. Router is standalone: `g++ -std=c++17 -O2 -pthread src/router/proxy.cpp -o anamesis-router`.
- **main.cpp:** CLI `--port/--admin-port/--memory-limit/--max-value-size/--aof-file/--no-aof/--policy/--shards/--embedding-worker-url/--ttl-default-sec/--semantic-threshold`; SIGINT/SIGTERM graceful shutdown; banner `Anamesis listening...`. `--shards` defaults to `std::thread::hardware_concurrency()`.
- **Concurrency:** one epoll thread owns the cache; per-shard `std::shared_mutex` (readers = shared, writers = exclusive); `AdminServer` runs on its own worker thread; `Server::stop()`/`AdminServer::stop()` wake loops via socketpair. Policy chosen once at startup; no runtime switching.
- **Protocol:** RESP2 only — incremental `Parser` (`src/protocol.{h,cpp}`), one allocation per bulk, binary-safe, pipelined. Verbs `PING`/`SELECT`/`METRICS`/`SEM.PUT`/`SEM.GET`/`SEM.SEARCH`/`SEM.DEL`/`SEM.FLUSH`/`SEM.INFO`/`SEM.STATS`; unknown → `UNKNOWN`. `GET`/`SET`/`DEL` removed. DoS guards: ≤1024 args, 32-byte header cap, bulk cap = `min(--max-value-size, memory limit) + 1MiB` overshoot.
- **embedding** (`src/embedding.{h,cpp}`): `cosine_similarity` (returns 0 for empty/mismatched/zero vectors) and `top_k` (partial_sort, descending).
- **epoll_compat.h**: `poll`-based epoll shim for macOS with concurrent-instance support (unique atomic epfd counter + global mutex).

## Build, Test & Verify

```sh
make          # builds ./anamesis (g++, -std=c++17 -O2 -Wall -Wextra -pthread -MMD -MP)
make test     # builds + runs tests/test_storage, tests/test_protocol, tests/test_semantic (all must print "ok ...")
make clean    # removes .objs, .objs_test, ./anamesis, test binaries
g++ -std=c++17 -O2 -pthread src/router/proxy.cpp -o anamesis-router   # builds router
python3 -m py_compile semcache_bench.py
```

- Makefile globs `find src -name '*.cpp' ! -path '*/router/*'`: new `.cpp` under `src/` (except router) compiles into `anamesis` automatically. Router is built separately.
- Test binaries live at `tests/test_*` (plus `.d` files); all build artifacts are local, never commit.
- Python: `python3 -m py_compile semcache_bench.py`.

## Implemented Core (verify against these, not the README)

- **Storage** (`src/storage.{h,cpp}`): sharded `std::vector<std::unique_ptr<CacheShard>>`, each shard guarded by `std::shared_mutex`. `CacheShard` holds `unordered_map<EntryId, CacheEntry>`, `unordered_map<ExactKey, EntryId>` exact index, `unique_ptr<EvictionPolicy>`, `unique_ptr<IVectorIndex>`, memory counters. Route by `hash(namespace_id) % shard_count`. `used_memory()` = policy memory_used + bucket bytes. Evicts (via policy) while over `memory_limit_`.
- **Semantic verbs:** `SEM.PUT namespace query embedding response [ttl]`; `SEM.GET namespace query` (exact first, then ANN via embedding worker); `SEM.SEARCH namespace embedding [threshold] [top_k]` → JSON bulk `{"hit":true,"id":...,"key":...,"value":...,"similarity":...}` or `{"hit":false}`; `SEM.DEL namespace entry_id`; `SEM.FLUSH`/`SEM.INFO`/`SEM.STATS`.
- **Gateway flow:** `SEM.GET` text → exact fingerprint lookup (miss) → call embedding worker (HTTP POST text → float32) → ANN search top-K → validate (exists, not expired, namespace match, similarity ≥ threshold) → HIT or MISS. On MISS, client runs RAG → `SEM.PUT` backfill.
- **Eviction** (`src/eviction/`): `EvictionPolicy` = `touch/add/remove/evict/memory_used` using `EntryId`. Called only under shard lock — policies NOT thread-safe. **SLRU default** (probationary → protected on first hit); `lru`/`lfu`/`sieve` selectable at construction. Never add threading inside a policy.
- **TTLManager** (`src/ttl.{h,cpp}`): sweeper thread (~100ms) calls back into `Storage::expire_key` (scans shards, takes shard lock) — the one exception to single-thread storage design; keep it that way. Lazy expiry also checked on `get`/`semget`/`semsearch`.
- **Server** (`src/server.{h,cpp}`): one epoll thread, non-blocking fds, per-client `Parser`/outbox maps, segmented outbox (head + shared blob + tail — no payload copy on GET), `EPOLLOUT` resume of partial sends, wake socketpair, EOF = `recv() == 0`, `epoll_ctl(EPOLL_CTL_DEL)` before `close()`. Embedding worker client (`call_embedding_worker`) does HTTP POST with timeout.
- **AdminServer** (`src/admin.{h,cpp}`): own epoll loop on a worker thread; `GET /metrics` (Prometheus text) + `GET /health`, `Connection: close`, 4KB request cap.
- **AOFLogger** (`src/aof.{h,cpp}`): one JSON line per `SEM.PUT`/`SEM.DEL` (base64 query/response/embedding + ttl + namespace), `fflush` per write, `replay` re-reads on boot and skips torn lines. No fsync, no compaction. Snapshots not yet implemented.
- **main.cpp:** CLI flags as above; graceful shutdown on SIGINT/SIGTERM.

## Benchmarks (Python)

- `semcache_bench.py` (repo root) — sweeps cosine thresholds (0.70–0.95) against a live server on a support-chatbot workload (positives = paraphrases, negatives = unrelated), reporting hit rate, false-positive rate, LLM calls saved, and estimated cost savings.
- **Its `embed_text()` uses deterministic SHA256 hash pseudo-embeddings — NOT real semantics.** Results are a harness smoke test only; real threshold calibration requires a real embedding path.
- No other bench scripts exist (PolyCache eviction/blob benches deleted).

## Architecture Rules

- Storage is sharded; each shard mutex-protected with `std::shared_mutex`; all storage ops under the appropriate lock.
- One epoll thread owns the cache; `AdminServer` runs on its own worker thread; loops woken via socketpair.
- Policy is chosen once at startup (`--policy`); no runtime switching.
- AOF: one JSON line per write, flush per call, replay on boot.
- Protocol is RESP2 only; explicit `-ERR` for unsupported options.

## Tech Stack & Code Style

- C++17 only (no `std::format`/C++20); POSIX; real epoll on Linux with the `poll` shim in `src/epoll_compat.h` for macOS (concurrent instances supported).
- Single-header JSON: vendored `src/third_party/nlohmann/json.hpp`. **Zero ML dependencies in the core** — embeddings are client-supplied or from external worker.
- `#pragma once` in headers; `using namespace std;` allowed in `.cpp`, keep `std::` prefixes in `.h`.
- Prefer `std::optional`, `std::unique_ptr`, `std::shared_ptr`, `std::mutex`/`std::shared_mutex`. Raw pointers only for epoll event data and SIEVE/SLRU linked-list nodes.

## Known Gaps / Next Phase

- **Vendor hnswlib** — `HNSWVectorIndex` is currently a Flat placeholder. Vendoring hnswlib (single header, zero external deps) gives production-grade ANN at scale.
- **Real embeddings** — `--embedding-worker-url` HTTP client exists but no worker binary/server provided yet. Integrate llama.cpp worker or client-side embedding.
- **Threshold calibration** — choose models (e.g. `bge-small-en-v1.5`, `nomic-embed-text-v1.5`, `all-MiniLM-L6-v2`) + dataset (SQuAD + Quora QQP, public support tickets), sweep 0.70–0.95 measuring precision/FPR/recall/latency.
- **Snapshots** — AOF replay only; periodic snapshot + checkpoint not implemented.
- **Router** — modulo hashing; ketama consistent hash for smooth rebalancing is v2. No cross-node `SEM.SEARCH` broadcast, no replication.
- Admin binds `INADDR_ANY`; no auth or TLS.

## Forbidden

- No C++20 features (`std::format`).
- No threading inside eviction policies.
- No runtime policy switching (`SWITCH_POLICY` must not come back); startup `--policy` only.
- No reintroducing "PolyCache", "athena", "CachePilot", or "AdaptiCache" naming, nor the deleted bench/doc files.
- No ML dependencies in the C++ core (no llama.cpp/onnx/torch linked into `anamesis` until an explicit decision says so).
- No committing `venv/`, `.DS_Store`, build artifacts (`.objs/`, test binaries, `*.d`), or `AGENTS.md`.
