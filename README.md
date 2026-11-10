# Anamesis

**Anamesis** (Greek, ἀνάμνησις = recollection) is a production-level C++17 in-memory **semantic response cache** for customer-support chatbot / internal-docs RAG workloads. It sits in front of an RAG/LLM pipeline: on a semantic hit it skips the LLM entirely; on a miss the application runs RAG and backfills via `SEM.PUT`. Target: 40–70% LLM call reduction.

## Core idea

Clients send raw **text** queries. Anamesis orchestrates embedding generation via an external worker (e.g. llama.cpp `llama-server --embedding`), then searches by meaning using an HNSW vector index. No embedding inference happens inside the C++ core — zero ML dependencies.

```
Client text query
      │
      ▼
Anamesis (RESP2 TCP:6379)
      │
      ├─▶ Exact fingerprint lookup (namespace + canonical text)  ──hit──▶ return cached response
      │
      ├─▶ Embedding worker (text → float32 vector) ───┐
      │                                               ▼
      │                                     ANN vector search (HNSW)
      │                                               │
      │                                     Validate: TTL, namespace, similarity ≥ threshold
      │                                               │
      │                                     Valid? ──▶ HIT (return response)
      │                                               │
      │                                     Invalid ──▶ MISS
      │                                               │
      │                                     Miss ──▶ Application runs RAG/LLM
      │                                               │
      │                                     Application SEM.PUTs response + embedding
      │                                               │
      │                                     Stored in cache (SLRU eviction if over limit)
```

## Features

- **Semantic search**: `SEM.GET namespace query_text` — exact fingerprint match first, then ANN cosine-similarity search over stored embeddings through an external embedding worker (llama.cpp `llama-server --embedding`)
- **Vector index**: exact flat scan by default (`FlatVectorIndex`); [hnswlib](https://github.com/nmslib/hnswlib) vendored for HNSW approximate search
- **Eviction**: SLRU baseline (`--policy slru`) or uniqueness-aware `SemanticSLRU` (`--policy semantic_slru --semantic-eviction-weight W`) with bounded, namespace-isolated similarity comparisons at eviction time
- **Strings only**: key = user query, value = LLM/RAG response. No binary blobs
- **RESP2 protocol**: incremental parser, pipelined, binary-safe, DoS guards (≤1024 args, 32-byte header cap)
- **AOF persistence**: JSON-lines with Base64-encoded embeddings, `fflush` per write, replay on boot
- **Admin HTTP**: `GET /metrics` (Prometheus text, `semcache_*` prefix) and `GET /health` on a separate worker thread
- **Horizontal scaling**: separate `anamesis-router` binary with consistent hash on `namespace_id`; each node is independent (no cross-node state, no replication in v1)
- **Shadow mode**: `--mode shadow` performs lookups and records would-be hits but returns `MISS`, for safe threshold calibration before production rollout

## Quick start

### 1. Build

```bash
make                       # builds ./anamesis (~375KB)
make test                  # ok test_storage, ok test_protocol, ok test_semantic, ok test_slru, ok test_vector_index
g++ -std=c++17 -O2 -pthread src/router/proxy.cpp -o anamesis-router
```

### 2. Start an embedding worker

Anamesis calls an external HTTP worker to convert text to embeddings. Use
llama.cpp in embedding mode:

```bash
/tmp/opencode/llama.cpp/build/bin/llama-server \
  --embedding --model models/nomic-embed-text-v1.5.Q4_K_M.gguf \
  --host 127.0.0.1 --port 8081
```

Test it:

```bash
curl -s -X POST http://127.0.0.1:8081/embedding \
  -H "Content-Type: application/json" -d '{"input":"how to reset password"}'
```

The response is the legacy llama.cpp embedding shape
`[{"index":0,"embedding":[[...]]}]` (768-dim floats for nomic-embed-text-v1.5).
No Python worker, no sentence-transformers, no PyTorch, and no ML library is
linked into the Anamesis binary.

### 3. Start Anamesis

```bash
./anamesis --port 6379 --admin-port 8080 --memory-limit 1024 \
  --shards 4 --embedding-worker-url http://127.0.0.1:8081/embedding \
  --mode shadow --semantic-threshold 0.80 --no-aof
```

### 4. Use it via RESP2 (or redis-cli)

```
SEM.PUT 1 how to reset password [0.1,0.2,...] reset password instructions 0
SEM.GET 1 how to reset password        # exact match
SEM.GET 1 how can I reset my password? # semantic match (calls embedding worker)
SEM.SEARCH 1 [0.1,0.2,...] 0.8 5       # raw ANN search diagnostics
SEM.DEL 1 <entry_id>
```

Example with `redis-cli`:
```bash
redis-cli -p 6379 SEM.PUT 1 "how to reset password" "[0.1,0.2,0.3]" "reset password instructions" 0
redis-cli -p 6379 SEM.GET 1 "how to reset password"
```

## CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `--port` | 6379 | RESP2 TCP port |
| `--admin-port` | 8080 | Admin HTTP port (/metrics, /health) |
| `--memory-limit` | 1024 | Memory limit in MB |
| `--max-value-size` | 512 | Max value size in MB |
| `--aof-file` | anamesis.aof | AOF file path |
| `--no-aof` | false | Disable AOF |
| `--policy` | slru | Eviction policy: slru\|lru\|lfu\|sieve\|semantic_slru (`slru` is the plain SLRU baseline; `semantic_slru` requires `--semantic-eviction-weight`) |
| `--semantic-eviction-weight` | 0.5 | SemanticSLRU blend weight in [0,1]; 0 degenerates to SLRU |
| `--semantic-top-k` | 5 | ANN candidates returned by SEM.GET / SEM.SEARCH |
| `--embedding-timeout-ms` | 5000 | HTTP embedding-worker request timeout |
| `--embedding-dim` | 0 | Reject worker embeddings that are not this many dimensions (0 = accept any) |
| `--shards` | hardware_concurrency() | Number of cache shards |
| `--ttl-default-sec` | 86400 | Default TTL in seconds |
| `--semantic-threshold` | 0.80 | Cosine similarity threshold |
| `--mode` | shadow | shadow\|active — shadow returns MISS but logs hits |

## Metrics (`GET /metrics`)

Prometheus text format with `semcache_*` prefix:

- `semcache_requests_total`, `semcache_hits_total`, `semcache_misses_total`
- `semcache_semantic_hits_total`, `semcache_semantic_misses_total`, `semcache_semantic_searches_total`
- `semcache_semantic_match_rate`, `semcache_semantic_candidates_per_search`
- `semcache_evictions_total`, `semcache_semantic_evictions_total`, `semcache_uniqueness_evictions_total`
- `semcache_embedding_dim_{384,768,1024,1536,other}_total`
- `semcache_semget_latency_nanoseconds`, `semcache_semsearch_latency_nanoseconds`, `semcache_embedding_latency_nanoseconds` (histograms)
- `semcache_memory_bytes`, `semcache_memory_limit_bytes`, `semcache_keys_total`, `semcache_hit_rate`, `semcache_miss_rate`, `semcache_up`
- `semcache_shadow_*_total` (shadow-mode calibration counters)

## Benchmark

```bash
python3 semcache_bench.py --port 6379 --data-dir data/ \
  --mode search --thresholds 0.60,0.70,0.75,0.80,0.85,0.90,0.95 --llm-cost 0.001
python3 semcache_bench.py --mode get --get-threshold 0.80   # end-to-end via llama.cpp
python3 semcache_bench.py --mode stream --memory-limit 40   # put/get stream under pressure
python3 tools/compare_policies.py --limit 3000 --memory-limit 20  # policy sweep
```

Reports hit rate, precision@K, recall@K, F1, FPR, latency, LLM calls saved, and estimated cost savings per threshold. Dataset: Quora Question Pairs; embeddings via `scripts/embed_dataset.py` through llama-server, cached in `data/embeddings_cache.jsonl`.

## Horizontal scaling

Run multiple Anamesis nodes behind `anamesis-router`:

```bash
./anamesis --port 16379 --admin-port 18080 --memory-limit 256 --shards 4 --no-aof &
./anamesis --port 16380 --admin-port 18081 --memory-limit 256 --shards 4 --no-aof &
./anamesis-router --port 16390 --node 127.0.0.1:16379 --node 127.0.0.1:16380 &
```

Clients connect to the router (port 16390). `namespace_id` is hashed to the owning node. Each node is independent — no cross-node state, no replication in v1.

## Testing

```bash
make test              # 5 test binaries, all print "ok ..."
python3 -m py_compile semcache_bench.py
```

Tests: `test_storage`, `test_protocol`, `test_semantic`, `test_slru`, `test_vector_index`, `test_embedding_client`, `test_semantic_slru`.

## Project structure

```
src/
  main.cpp              CLI, boot, signal handling
  server.h/.cpp         RESP2 event loop, SEM.* dispatch, embedding worker client
  protocol.h/.cpp       Incremental RESP2 parser
  storage.h/.cpp        Sharded entry store, exact index, SLRU + ANN integration
  eviction/             SLRU, LRU, LFU, SIEVE policies (EntryId-based)
  index/                IVectorIndex, FlatVectorIndex, HNSWVectorIndex (hnswlib)
  ttl.h/.cpp            Min-heap TTL sweeper (~100ms)
  aof.h/.cpp            JSON-lines AOF with Base64 embeddings, replay
  admin.h/.cpp          HTTP /metrics + /health
  epoll_compat.h        poll-based epoll shim for macOS
  third_party/          nlohmann/json, hnswlib
tests/
  test_storage.cpp      Entry store, exact index, TTL, eviction
  test_protocol.cpp     RESP2 parser, framing, binary payloads
  test_semantic.cpp     ANN search, namespace isolation, threshold
  test_slru.cpp         Probationary/protected promotion, eviction order
  test_vector_index.cpp Flat + HNSW add/search/remove
scripts/
  embed_client.*    HTTP client for the embedding worker (JSON escaping, llama.cpp legacy + OpenAI shapes)
  prepare_qqp.py    Build QQP benchmark dataset from the nyu-mll/glue parquet split
  embed_dataset.py  Batched embedding of dataset texts through llama-server, cached to data/embeddings_cache.jsonl
data/
  qqp_cache.jsonl   QQP anchor entries cached by the benchmark
  qqp_pos.jsonl / qqp_neg.jsonl  Positive / negative test pairs with ground truth
  embeddings_cache.jsonl  llama.cpp-produced vectors, keyed by text hash (reusable across runs)
```

## Architecture rules

- C++17 only (no `std::format`/C++20)
- Zero ML dependencies in the C++ core (embeddings are client/external-worker data)
- Storage mutex-protected; all ops under `std::shared_mutex`/`std::lock_guard`
- One epoll thread owns the cache; AdminServer on its own worker thread
- Policy chosen once at startup (`--policy`); no runtime switching
- AOF: one JSON line per write, flush per call, replay on boot
- `#pragma once` headers; `std::` prefixes in `.h`

## Roadmap

- [x] SLRU eviction, flat vector index (hnswlib vendored for HNSW)
- [x] SemanticSLRU with bounded, namespace-isolated uniqueness eviction
- [x] Shadow/active mode
- [x] Horizontal router (consistent hash)
- [ ] QQP threshold calibration report (semcache_bench.py)
- [ ] AOF snapshots + compaction
- [ ] Ketama consistent hash for smooth rebalancing
- [ ] Cross-node SEM.SEARCH broadcast (v2)
- [ ] Auth/TLS

## License

MIT. See LICENSE.