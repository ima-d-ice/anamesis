# Anamesis Project Rules

## Current State (read first)

- **Project: anamesis** (Greek, ἀνάμνησις = recollection) — a production-level C++17 in-memory **semantic response cache** for customer-support chatbot / internal-docs RAG workloads. Sits in front of an RAG/LLM pipeline: semantic hit → skip the LLM; miss → RAG → `SEM.PUT` backfill. Target: 40–70% LLM call reduction.
- Folder `/Users/imad/Desktop/anamesis`; binary `./anamesis`; default AOF `anamesis.aof`; router binary `./anamesis-router`. Git repo present.
- **Input model:** clients send raw **text** queries (`SEM.GET namespace query_text`). Anamesis orchestrates embedding generation via an external llama.cpp worker (`--embedding-worker-url`), then searches by meaning. No embedding inference inside the C++ core.
- **Embedding worker:** `llama-server --embedding --model models/nomic-embed-text-v1.5.Q4_K_M.gguf --host 127.0.0.1 --port 8081`; Anamesis POSTs `{"input":"<text>","embd_normalize":2}` to `POST /embedding` (legacy llama.cpp shape `[{"index":0,"embedding":[[...]]}]`, 768-dim floats). Client: `src/embedding_client.{h,cpp}` (JSON escaping, getaddrinfo DNS, bounded retries/timeouts, both legacy and OpenAI `/v1/embeddings` response shapes). No Python worker, no sentence-transformers, no PyTorch, no ML library linked into `anamesis`.
- **Strings only:** key/query = user query, value = LLM/RAG response. No binary-blob feature work.
- Core implemented + verified: `make clean && make test` → 6 binaries print `ok ...` (`test_storage`, `test_protocol`, `test_semantic`, `test_slru`, `test_vector_index`, `test_embedding_client`, `test_semantic_slru`); `make` builds `./anamesis` (~375KB); live smoke of `PING`/`SEM.PUT`/`SEM.GET`/`SEM.SEARCH`/`SEM.DEL`, `/metrics`, `/health` passed; router + 2-node horizontal smoke passed; QQP benchmark through llama.cpp verified end-to-end.
- **Eviction:** `--policy slru|lru|lfu|sieve|semantic_slru` (startup only; **no runtime switching**). `slru` is the plain SLRU baseline; `semantic_slru` adds uniqueness-aware protected-segment eviction with `--semantic-eviction-weight W` (0.0 ≡ plain SLRU, 1.0 ≡ pure uniqueness). Policy state lives only in the policy object, not `CacheEntry`. Eviction order: TTL-expired → invalidated → probationary LRU → protected LRU. Uniqueness comparisons are namespace-scoped and bounded per eviction (≤32 candidates × ≤256 same-namespace peers, deterministic rotating cursors) so eviction cost never scales with cache size.
- **CacheEntry** = `EntryId` (uint64_t), namespace_id, canonical_query, original_query, response, content_type, `std::vector<float> embedding`, created/expires/last-access timestamps, hit_count, segment, memory_bytes. No size field — sizes computed via `entry_bytes()`.
- **Exact lookup:** `unordered_map<ExactKey, EntryId>` (namespace_id + canonical_query) → O(1) byte-identical repeat path, checked before ANN.
- **ANN search:** `IVectorIndex` abstraction. `FlatVectorIndex` (exact linear scan) wired by default; hnswlib vendored (`HNSWVectorIndex` exists but storage uses flat). Per-shard index. Cosine similarity, `similarity = 1 - distance`.
- **Semantic verbs:** `SEM.PUT namespace query embedding response [ttl]`; `SEM.GET namespace query` (exact first, then ANN via embedding worker, `--semantic-top-k`); `SEM.SEARCH namespace embedding [threshold] [top_k]` → JSON bulk `{"hit":true,"id":...,"key":...,"value":...,"similarity":...}` or `{"hit":false}`; `SEM.DEL namespace entry_id`; `SEM.FLUSH`/`SEM.INFO`/`SEM.STATS`.
- **Gateway flow:** `SEM.GET` text → exact fingerprint lookup (miss) → call embedding worker (HTTP POST text → float32) → ANN search top-K → validate (exists, not expired, namespace match, similarity ≥ threshold) → HIT or MISS. On MISS, client runs RAG → `SEM.PUT` backfill.
- **Eviction** (`src/eviction/`): `EvictionPolicy` = `touch/add/remove/evict/memory_used` using `EntryId`, plus optional `set_embedding(id, data, dim, namespace_id)` and semantic counters. Called only under shard lock — policies NOT thread-safe. Never add threading inside a policy.
- **TTLManager** (`src/ttl.{h,cpp}`): sweeper thread (~100ms) calls back into `Storage::expire_key` (scans shards, takes shard lock) — the one exception to single-thread storage design; keep it that way. Lazy expiry also checked on `get`/`semget`/`semsearch`.
- **Server** (`src/server.{h,cpp}`): one epoll thread, non-blocking fds, per-client `Parser`/outbox maps, segmented outbox (head + shared blob + tail — no payload copy on GET), `EPOLLOUT` resume of partial sends, wake socketpair, EOF = `recv() == 0`, `epoll_ctl(EPOLL_CTL_DEL)` before `close()`. Embedding worker client (`call_embedding_worker`, delegated to `embedding_client`) does HTTP POST with timeout.
- **AdminServer** (`src/admin.{h,cpp}`): own epoll loop on a worker thread; `GET /metrics` (Prometheus text) + `GET /health`, `Connection: close`, 4KB request cap.
- **AOFLogger** (`src/aof.{h,cpp}`): one JSON line per `SEM.PUT`/`SEM.DEL` (base64 query/response/embedding + ttl + namespace), `fflush` per write, `replay` re-reads on boot and skips torn lines. No fsync, no compaction. Snapshots not yet implemented.
- **main.cpp:** CLI `--port/--admin-port/--memory-limit/--max-value-size/--aof-file/--no-aof/--policy/--shards/--embedding-worker-url/--embedding-timeout-ms/--embedding-dim/--ttl-default-sec/--semantic-threshold/--semantic-top-k/--semantic-eviction-weight/--mode`; SIGINT/SIGTERM graceful shutdown; banner `Anamesis listening...`. `--shards` defaults to `std::thread::hardware_concurrency()`.
- **Concurrency:** one epoll thread owns the cache; per-shard `std::shared_mutex` (readers = shared, writers = exclusive); `AdminServer` runs on its own worker thread; loops woken via socketpair. Policy chosen once at startup; no runtime switching.
- **Protocol:** RESP2 only — incremental `Parser` (`src/protocol.{h,cpp}`), one allocation per bulk, binary-safe, pipelined. Verbs `PING`/`SELECT`/`METRICS`/`SEM.PUT`/`SEM.GET`/`SEM.SEARCH`/`SEM.DEL`/`SEM.FLUSH`/`SEM.INFO`/`SEM.STATS`; unknown → `UNKNOWN`. `GET`/`SET`/`DEL` removed. DoS guards: ≤1024 args, 32-byte header cap, bulk cap = `min(--max-value-size, memory limit) + 1MiB` overshoot.
- **embedding** (`src/embedding.{h,cpp}`): `cosine_similarity` (returns 0 for empty/mismatched/zero vectors) and `top_k` (partial_sort, descending).
- **epoll_compat.h**: `poll`-based epoll shim for macOS with concurrent-instance support (unique atomic epfd counter + global mutex).

## Build, Test & Verify

```sh
make              # builds ./anamesis (g++, -std=c++17 -O2 -Wall -Wextra -pthread -MMD -MP)
make test         # builds + runs all tests/test_* (all must print "ok ...")
make clean        # removes .objs, .objs_test, ./anamesis, test binaries
g++ -std=c++17 -O2 -pthread -Isrc -Isrc/third_party src/router/proxy.cpp -o anamesis-router   # builds router
python3 -m py_compile semcache_bench.py scripts/prepare_qqp.py scripts/embed_dataset.py tools/compare_policies.py
```

- Makefile globs `find src -name '*.cpp' ! -path '*/router/*'`: new `.cpp` under `src/` (except router) compiles into `anamesis` automatically. Router is built separately.
- Test binaries live at `tests/test_*` (plus `.d` files); all build artifacts are local, never commit.
- Python: venv at `venv/` with `pyarrow` (dataset prep); `python3 -m py_compile` all scripts.

## Implemented Core (verify against these, not the README)

- **Storage** (`src/storage.{h,cpp}`): sharded `std::vector<std::unique_ptr<CacheShard>>`, each shard guarded by `std::shared_mutex`. `CacheShard` holds `unordered_map<EntryId, CacheEntry>`, `unordered_map<ExactKey, EntryId>` exact index, `unique_ptr<EvictionPolicy>`, `unique_ptr<IVectorIndex>`, memory counters. Route by `hash(namespace_id) % shard_count`. `used_memory()` = policy memory_used + bucket bytes (policy `memory_used()` includes its embedding copies). Evicts (via policy) while over per-shard `memory_limit_ / shard_count`.
- **Semantic verbs:** as above.
- **Gateway flow:** as above.
- **Eviction** (`src/eviction/`): as above. `slru` = plain SLRU; `semantic_slru` = SLRU + bounded uniqueness scoring.
- **TTLManager** (`src/ttl.{h,cpp}`): as above.
- **Server** (`src/server.{h,cpp}`): as above.
- **AdminServer** (`src/admin.{h,cpp}`): as above. Metrics: `semcache_*` prefix — hits/misses, semantic hits/misses/searches, semantic match rate, candidates per search, evictions (total / semantic / uniqueness), embedding latency histogram, shadow-mode counters, expired keys, memory gauges, histograms for semget/semsearch/embedding calls.
- **AOFLogger** (`src/aof.{h,cpp}`): as above.
- **main.cpp:** CLI flags as above; graceful shutdown on SIGINT/SIGTERM.

## Benchmarks (Python)

- `semcache_bench.py` (repo root) — QQP-based benchmark against a live server: `search` mode (SEM.SEARCH sweep over thresholds 0.60–0.95), `get` mode (end-to-end SEM.GET through llama.cpp), `stream` mode (interleaved PUT/GET under memory pressure). Reports precision@K, recall@K, F1, FPR, cache hit rate, latency, LLM calls saved, estimated cost savings.
- `scripts/prepare_qqp.py` — downloads nyu-mll/glue QQP parquet, builds 10K cache anchors / 5K positives / 5K negatives (deterministic, seed 42).
- `scripts/embed_dataset.py` — batched embedding of all dataset texts through llama-server, cached to `data/embeddings_cache.jsonl` (every vector produced by llama.cpp; repeat runs reuse the cache).
- `tools/compare_policies.py` — runs `slru` vs `semantic_slru` @ {0.3, 0.5, 0.7} under memory pressure and reports hit rate, evictions, memory.
- All embeddings come from llama.cpp; no pseudo-embedding logic anywhere (the old SHA256 `embed_text()` never existed in code — that was a doc error).

## Architecture Rules

- Storage is sharded; each shard mutex-protected with `std::shared_mutex`; all storage ops under the appropriate lock.
- One epoll thread owns the cache; `AdminServer` runs on its own worker thread; loops woken via socketpair.
- Policy is chosen once at startup (`--policy`); no runtime switching.
- AOF: one JSON line per write, flush per call, replay on boot.
- Protocol is RESP2 only; explicit `-ERR` for unsupported options.

## Tech Stack & Code Style

- C++17 only (no `std::format`/C++20); POSIX; real epoll on Linux with the `poll` shim in `src/epoll_compat.h` for macOS (concurrent instances supported).
- Single-header JSON: vendored `src/third_party/nlohmann/json.hpp`. **Zero ML dependencies in the core** — embeddings are client/external-worker data.
- `#pragma once` headers; `std::` prefixes in `.h`; `using namespace std;` allowed in `.cpp`.
- Prefer `std::optional`, `std::unique_ptr`, `std::shared_ptr`, `std::mutex`/`std::shared_mutex`. Raw pointers only for epoll event data and SIEVE/SLRU linked-list nodes.

## Known Gaps / Next Phase

- **Threshold calibration** — sweep 0.70–0.95 on QQP measuring precision/FPR/recall/latency (semcache_bench.py does this).
- **Snapshots** — AOF replay only; periodic snapshot + checkpoint not implemented.
- **Router** — modulo hashing; ketama consistent hash for smooth rebalancing is v2. No cross-node `SEM.SEARCH` broadcast, no replication.
- Admin binds `INADDR_ANY`; no auth or TLS.

## Forbidden

- No C++20 features (`std::format`).
- No threading inside eviction policies.
- No runtime policy switching (`SWITCH_POLICY` must not come back); startup `--policy` only.
- No reintroducing "PolyCache", "athena", "CachePilot", or "AdaptiCache" naming, nor the deleted bench/doc files.
- No ML dependencies in the C++ core (no llama.cpp/onnx/torch linked into `anamesis` until an explicit decision says so).
- No committing `venv/`, `.DS_Store`, build artifacts (`.objs/`, test binaries, `*.d`), `data/*` datasets, or `AGENTS.md`.
