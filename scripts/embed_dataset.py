#!/usr/bin/env python3
"""Embed every string in a QQP dataset through llama-server --embedding.

Reads data/qqp_{cache,pos,neg}.jsonl, collects the unique texts, and POSTs
them in batches to the embedding worker. Results are appended to
data/embeddings_cache.jsonl so repeat runs are instant: only texts not yet
cached are (re)embedded. Every vector comes from llama.cpp -- none are
synthesized locally.

Usage:
  venv/bin/python scripts/embed_dataset.py \\
      --worker-url http://127.0.0.1:8081/embedding --batch 32
"""

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO, "data")

BATCH = 32
RETRIES = 3


def norm(text):
    return " ".join(str(text).split())


def text_key(text):
    return hashlib.sha256(norm(text).encode("utf-8")).hexdigest()


def load_texts(*paths):
    texts = {}
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                texts[rec["query"]] = None
    return texts


def load_cached(path):
    done = set()
    if not os.path.exists(path):
        return done
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                done.add(text_key(rec["text"]))
            except (json.JSONDecodeError, KeyError):
                # torn line: ignore, will be re-embedded
                pass
    return done


def post_batch(url, texts, timeout_ms):
    body = json.dumps({"input": list(texts)}).encode("utf-8")
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"})
    last_err = None
    for attempt in range(RETRIES):
        try:
            with urllib.request.urlopen(req, timeout=timeout_ms / 1000.0) as r:
                raw = r.read()
            data = json.loads(raw)
            return parse_batch(data)
        except (urllib.error.URLError, OSError, ValueError, KeyError) as e:
            last_err = e
            time.sleep(min(30, 2 ** attempt))
    raise RuntimeError(f"batch of {len(texts)} failed after {RETRIES} tries: {last_err}")


def parse_batch(data):
    """Extract ordered embeddings from llama legacy or OpenAI response."""
    items = []
    if isinstance(data, list):
        for entry in data:
            idx = entry.get("index", len(items))
            vec = entry["embedding"]
            items.append((idx, vec))
    elif isinstance(data, dict) and "data" in data:
        for entry in data["data"]:
            idx = entry.get("index", len(items))
            vec = entry["embedding"]
            items.append((idx, vec))
    else:
        raise ValueError("unexpected embedding response shape")

    items.sort(key=lambda t: t[0])
    if [i for i, _ in items] != list(range(len(items))):
        raise ValueError("embedding response missing indexes")
    out = []
    for _, vec in items:
        if isinstance(vec[0], list):
            vec = vec[0]  # llama legacy: [[...]]
        out.append([float(x) for x in vec])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=os.path.join(DATA_DIR, "qqp_cache.jsonl"))
    ap.add_argument("--pos", default=os.path.join(DATA_DIR, "qqp_pos.jsonl"))
    ap.add_argument("--neg", default=os.path.join(DATA_DIR, "qqp_neg.jsonl"))
    ap.add_argument("--out", default=os.path.join(DATA_DIR, "embeddings_cache.jsonl"))
    ap.add_argument("--worker-url", default="http://127.0.0.1:8081/embedding")
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--timeout-ms", type=int, default=60000)
    args = ap.parse_args()

    texts = load_texts(args.cache, args.pos, args.neg)
    total = len(texts)
    done = load_cached(args.out)
    missing = [t for t in texts if text_key(t) not in done]
    print(f"[embed] {total} unique texts, {len(done)} already cached, {len(missing)} to embed")

    if not missing:
        print("[embed] nothing to do")
        return

    buf = open(args.out, "a", encoding="utf-8")
    embedded = 0
    failed = 0
    t0 = time.time()
    try:
        for i in range(0, len(missing), args.batch):
            chunk = missing[i:i + args.batch]
            try:
                vecs = post_batch(args.worker_url, chunk, args.timeout_ms)
            except RuntimeError as e:
                print(f"[embed] FAILED batch at offset {i}: {e}")
                failed += len(chunk)
                continue
            if len(vecs) != len(chunk):
                print(f"[embed] short batch: asked {len(chunk)} got {len(vecs)}")
                failed += len(chunk)
                continue
            for text, vec in zip(chunk, vecs):
                buf.write(json.dumps({"text": text, "embedding": vec},
                                     ensure_ascii=False) + "\n")
                embedded += 1
            if (embedded + failed) % 512 < args.batch:
                dt = time.time() - t0
                rate = embedded / dt if dt > 0 else 0
                print(f"[embed] {embedded + failed}/{len(missing)} "
                      f"({embedded} embedded, {failed} failed) {rate:.0f} texts/s")
            buf.flush()
    finally:
        buf.close()

    dt = time.time() - t0
    print(f"[embed] done in {dt:.1f}s: {embedded} embedded, {failed} failed, "
          f"{total} total cached vectors")


if __name__ == "__main__":
    main()
