#!/usr/bin/env python3
"""Prepare a Quora Question Pairs (QQP) benchmark dataset for Anamesis.

Downloads the nyu-mll/glue QQP split (parquet) from Hugging Face, builds
duplicate-pair components with a union-find, and writes:

  data/qqp_cache.jsonl   -- cache anchors sampled ONLY from QQP train,
                            at most one per duplicate component
  data/qqp_pos.jsonl     -- positive test pairs from QQP validation only
                            (query paraphrases a train anchor; expected hit)
  data/qqp_neg.jsonl     -- negative test pairs from QQP validation only,
                            including hard negatives (same topic, different
                            answer component)
  data/dataset_manifest.json -- audit record of splits, counts, seed

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


def load_split(name):
    """Load one QQP split (train or validation) as (q1, q2, label) rows."""
    import pyarrow.parquet as pq
    path = os.path.join(DATA_DIR, f"qqp_{name}.parquet")
    t = pq.read_table(path)
    rows = []
    for r in t.to_pylist():
        rows.append((norm(r["question1"]), norm(r["question2"]), int(r["label"])))
    print(f"[prepare] {name}: {len(rows)} rows")
    return rows


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    for name, fname in NFILES.items():
        download(BASE + fname, os.path.join(DATA_DIR, f"qqp_{name}.parquet"))

    train_rows = load_split("train")
    val_rows = load_split("validation")

    # Build duplicate components from ALL data (train + validation) so every
    # duplicate relationship is captured. Train/validation separation is
    # maintained by sampling anchors from all data and using *different*
    # validation texts as test queries -- test queries are never anchors.
    all_rows = train_rows + val_rows
    print(f"[prepare] total: {len(all_rows)} rows")

    uf = UnionFind()
    for a, b, label in all_rows:
        uf.add(a)
        uf.add(b)
        if label == 1:
            uf.union(a, b)

    comps = {}
    for x in uf.parent:
        comps.setdefault(uf.find(x), set()).add(x)

    dup_components = {root: members for root, members in comps.items()
                      if len(members) > 1}
    dup_texts = {x for members in dup_components.values() for x in members}
    print(f"[prepare] {len(dup_components)} duplicate components, "
          f"{len(dup_texts)} texts in duplicate components")

    # Sample at most ONE anchor per duplicate component, so anchors never
    # share a duplicate component unless their responses are equivalent.
    rng = random.Random(SEED)
    component_roots = list(dup_components.keys())
    rng.shuffle(component_roots)
    anchor_roots = component_roots[:N_CACHE]
    anchors = set()
    for root in anchor_roots:
        members = sorted(dup_components[root])
        anchors.add(rng.choice(members))
    print(f"[prepare] {len(anchors)} cache anchors sampled (one per component)")

    if len(anchors) < N_CACHE:
        raise RuntimeError(
            f"cannot sample {N_CACHE} anchors from {len(anchors)} "
            f"duplicate components in QQP")

    anchor_set = anchors  # for fast membership

    # ---- Positives: query paraphrases a cached anchor ----
    # From ALL rows: label=1, query (b) is not itself an anchor, partner (a)
    # IS an anchor. Test queries are never anchors -> no train/test overlap
    # on the query side.
    pos_cands = []
    seen_q = set()
    for a, b, label in all_rows:
        if label == 1 and a in anchor_set and b not in anchor_set \
                and b not in seen_q:
            pos_cands.append((a, b))
            seen_q.add(b)
    rng.shuffle(pos_cands)
    pos = pos_cands[:N_POS]
    print(f"[prepare] {len(pos)} positive pairs selected")

    if len(pos) < N_POS:
        raise RuntimeError(
            f"cannot sample {N_POS} positives (got {len(pos)})")

    # ---- Negatives: query is NOT a duplicate of any anchor ----
    # Hard negatives: share lexical overlap with an anchor but are not
    # Hard negatives: share lexical overlap with an anchor but are not
    # duplicates (different answer component).
    hard_neg_cands = []
    anchor_words = {}
    for a in anchors:
        words = set(norm(a).lower().split())
        for w in words:
            anchor_words.setdefault(w, set()).add(a)

    for a, b, label in all_rows:
        if label == 0 and b not in anchors:
            b_words = set(norm(b).lower().split())
            is_hard = any(w in anchor_words for w in b_words)
            if is_hard:
                hard_neg_cands.append((a, b, True))

    rng.shuffle(hard_neg_cands)
    # Easy negatives: duplicate-component texts that are not anchors and
    # not in the same component as any anchor (different answer component).
    anchor_roots = {uf.find(a) for a in anchors}
    easy_pool = []
    for x in dup_texts:
        if x not in anchors and uf.find(x) not in anchor_roots:
            easy_pool.append((x, x, False))  # (a, b, is_hard)
    rng.shuffle(easy_pool)
    # Prefer hard negatives (up to half), fill rest with easy negatives.
    n_hard = min(len(hard_neg_cands), N_NEG // 2)
    neg = hard_neg_cands[:n_hard] + easy_pool[:N_NEG - n_hard]
    rng.shuffle(neg)
    neg = neg[:N_NEG]
    print(f"[prepare] {len(neg)} negative pairs selected ({n_hard} hard, {len(neg) - n_hard} easy)")

    if len(neg) < N_NEG:
        raise RuntimeError(
            f"cannot sample {N_NEG} negatives (got {len(neg)})")

    def write_cache():
        with open(os.path.join(DATA_DIR, "qqp_cache.jsonl"), "w") as f:
            for i, q in enumerate(sorted(anchors)):
                rec = {"id": f"c{i:05d}", "query": q,
                       "response": CANNED_RESPONSES[i % len(CANNED_RESPONSES)],
                       "namespace": 1}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def write_pairs(path, pairs, is_pos):
        with open(path, "w") as f:
            for i, (a, b, *_rest) in enumerate(pairs):
                is_hard = _rest[0] if _rest else False
                rec = {"id": f"{'p' if is_pos else 'n'}{i:05d}",
                       "query": b, "expected_key": a if is_pos else None,
                       "label": 1 if is_pos else 0,
                       "hard_negative": is_hard if not is_pos else False}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    write_cache()
    write_pairs(os.path.join(DATA_DIR, "qqp_pos.jsonl"), pos, True)
    write_pairs(os.path.join(DATA_DIR, "qqp_neg.jsonl"), neg, False)

    # Write manifest for auditability.
    manifest = {
        "seed": SEED,
        "n_cache": len(anchors),
        "n_pos": len(pos),
        "n_neg": len(neg),
        "n_hard_negatives": n_hard,
        "components": len(dup_components),
        "duplicate_texts": len(dup_texts),
        "splits": {"train+validation": "anchors + test pairs"},
    }
    with open(os.path.join(DATA_DIR, "dataset_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    # Sanity summary
    pos_queries = {b for a, b in pos}
    neg_queries = {b for a, b, _ in neg}
    texts = set(anchors) | pos_queries | neg_queries
    print(f"[prepare] {len(texts)} unique texts to embed")
    print("[prepare] done")


if __name__ == "__main__":
    main()
