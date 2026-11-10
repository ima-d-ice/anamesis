#!/usr/bin/env python3
"""Prepare a Quora Question Pairs (QQP) benchmark dataset for Anamesis.

Downloads the nyu-mll/glue QQP split (parquet) from Hugging Face, builds
duplicate-pair components with a union-find, and writes:

  data/qqp_cache.jsonl   -- 10K cached questions (one per line, with a canned
                            response), namespace 1
  data/qqp_pos.jsonl     -- 5K positive test pairs (query paraphrases a cached
                            question; expected cache hit)
  data/qqp_neg.jsonl     -- 5K negative test pairs (query is not a duplicate of
                            any cached question; expected cache miss)

Everything is deterministic (fixed seed). `data/` is gitignored.
"""

import hashlib
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO, "data")
BASE = "https://huggingface.co/datasets/nyu-mll/glue/resolve/main/qqp/"

NFILES = {
    "train": "train-00000-of-00001.parquet",
    "validation": "validation-00000-of-00001.parquet",
}

N_CACHE = 10_000
N_POS = 5_000
N_NEG = 5_000
SEED = 42

CANNED_RESPONSES = [
    "Please check your internet connection and try again.",
    "Contact support at help@example.com for further assistance.",
    "Visit the Account Settings page to manage your subscription.",
    "Use the password reset link on the login page.",
    "Refunds are processed within 5-7 business days.",
]


def norm(text):
    return " ".join(str(text).split())


def download(url, out):
    if os.path.exists(out) and os.path.getsize(out) > 0:
        print(f"[prepare] cached {out} ({os.path.getsize(out)} bytes)")
        return
    tmp = out + ".part"
    req = urllib.request.Request(url, headers={"User-Agent": "anamesis-prepare/1.0"})
    last_err = None
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
            os.replace(tmp, out)
            print(f"[prepare] downloaded {os.path.getsize(out)} bytes -> {out}")
            return
        except (urllib.error.URLError, OSError) as e:
            last_err = e
            time.sleep(2 ** attempt)
    raise RuntimeError(f"failed to download {url}: {last_err}")


class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        parent = self.parent
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra

    def add(self, x):
        self.parent.setdefault(x, x)


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    for name, fname in NFILES.items():
        download(BASE + fname, os.path.join(DATA_DIR, f"qqp_{name}.parquet"))

    import pyarrow.parquet as pq

    rows = []
    for name in ("train", "validation"):
        t = pq.read_table(os.path.join(DATA_DIR, f"qqp_{name}.parquet"))
        for r in t.to_pylist():
            rows.append((norm(r["question1"]), norm(r["question2"]), int(r["label"])))
    print(f"[prepare] {len(rows)} QQP rows loaded")

    uf = UnionFind()
    for a, b, label in rows:
        uf.add(a)
        uf.add(b)
        if label == 1:
            uf.union(a, b)

    comps = {}
    for x in uf.parent:
        comps.setdefault(uf.find(x), set()).add(x)

    dup_texts = {x for x, c in comps.items() if len(c) > 1 for x in c}
    print(f"[prepare] {len(dup_texts)} texts in duplicate components")

    rng = random.Random(SEED)
    anchors = set(rng.sample(sorted(dup_texts), min(N_CACHE, len(dup_texts))))
    print(f"[prepare] {len(anchors)} cache anchors sampled")

    # Positives: anchor -> paraphrase partner not itself an anchor.
    pos_cands = []
    seen_q = set()
    for a, b, label in rows:
        if label == 1 and a in anchors and b not in anchors and b not in seen_q:
            pos_cands.append((a, b))
            seen_q.add(b)
    rng.shuffle(pos_cands)
    pos = pos_cands[:N_POS]
    print(f"[prepare] {len(pos)} positive pairs selected")

    # Negatives: query whose component shares no anchor.
    neg_cands = []
    seen_q = set()
    anchor_roots = {uf.find(a) for a in anchors}
    for a, b, label in rows:
        if label == 0 and uf.find(b) not in anchor_roots and b not in seen_q:
            neg_cands.append((a, b))
            seen_q.add(b)
    rng.shuffle(neg_cands)
    neg = neg_cands[:N_NEG]
    print(f"[prepare] {len(neg)} negative pairs selected")

    def write_cache():
        with open(os.path.join(DATA_DIR, "qqp_cache.jsonl"), "w") as f:
            for i, q in enumerate(sorted(anchors)):
                rec = {"id": f"c{i:05d}", "query": q,
                       "response": CANNED_RESPONSES[i % len(CANNED_RESPONSES)],
                       "namespace": 1}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def write_pairs(path, pairs, is_pos):
        with open(path, "w") as f:
            for i, (a, b) in enumerate(pairs):
                rec = {"id": f"{'p' if is_pos else 'n'}{i:05d}",
                       "query": b, "expected_key": a if is_pos else None,
                       "label": 1 if is_pos else 0}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    write_cache()
    write_pairs(os.path.join(DATA_DIR, "qqp_pos.jsonl"), pos, True)
    write_pairs(os.path.join(DATA_DIR, "qqp_neg.jsonl"), neg, False)

    # Sanity embeddings-ready summary
    texts = set(anchors) | {b for _, b in pos} | {b for _, b in neg}
    print(f"[prepare] {len(texts)} unique texts to embed")
    print("[prepare] done")


if __name__ == "__main__":
    main()
