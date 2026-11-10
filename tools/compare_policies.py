#!/usr/bin/env python3
"""Compare Anamesis eviction policies on an identical QQP workload.

Starts llama-server once, then anamesis per policy configuration, runs
semcache_bench.py (search mode, SEM.SEARCH with llama.cpp vectors) against
each, and aggregates precision@K / recall@K / F1 / FPR plus server-side
eviction and memory counters from /metrics.

Usage:
  venv/bin/python tools/compare_policies.py [--limit 1000] [--threshold 0.80]
                                            [--out report.json]

Requires data/qqp_*.jsonl and data/embeddings_cache.jsonl
(scripts/prepare_qqp.py + scripts/embed_dataset.py).
"""

import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.request

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LLAMA_SERVER = "/tmp/opencode/llama.cpp/build/bin/llama-server"
LLAMA_MODEL = os.path.join(REPO, "models", "nomic-embed-text-v1.5.Q4_K_M.gguf")
ANAMESIS = os.path.join(REPO, "anamesis")
BENCH = os.path.join(REPO, "semcache_bench.py")
WORKER_URL = "http://127.0.0.1:8081/embedding"

CONFIGS = [
    ("slru", None),
    ("semantic_slru", 0.3),
    ("semantic_slru", 0.5),
    ("semantic_slru", 0.7),
]

METRIC_RE = re.compile(r"^(\S+)\s+([0-9.eE+-]+)\s*$")


def fetch_metrics(admin_port):
    with urllib.request.urlopen(
            f"http://127.0.0.1:{admin_port}/metrics", timeout=10) as r:
        text = r.read().decode("utf-8", "replace")
    m = {}
    for line in text.splitlines():
        match = METRIC_RE.match(line)
        if match:
            m[match.group(1)] = float(match.group(2))
    return m


def wait_admin(admin_port, timeout=60):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(
                    f"http://127.0.0.1:{admin_port}/health", timeout=2) as r:
                if r.status == 200:
                    return True
        except OSError:
            pass
        time.sleep(0.5)
    return False


class LlamaServer:
    def __init__(self):
        self.proc = None

    def start(self):
        try:
            req = urllib.request.Request(
                WORKER_URL, data=b'{"input":"healthcheck"}',
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=5) as r:
                if r.status == 200:
                    print("[compare] llama-server already running on 8081")
                    self.proc = None
                    return
        except OSError:
            pass
        self.proc = subprocess.Popen(
            [LLAMA_SERVER, "--model", LLAMA_MODEL, "--embedding",
             "--host", "127.0.0.1", "--port", "8081"],
            stdout=open("/tmp/opencode/compare_llama.log", "wb"),
            stderr=subprocess.STDOUT)
        t0 = time.time()
        while time.time() - t0 < 120:
            try:
                req = urllib.request.Request(
                    WORKER_URL,
                    data=b'{"input":"healthcheck"}',
                    headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=10) as r:
                    if r.status == 200:
                        return
            except OSError:
                pass
            if self.proc.poll() is not None:
                raise RuntimeError("llama-server exited early")
            time.sleep(1)
        raise RuntimeError("llama-server did not become ready")

    def stop(self):
        if self.proc is not None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()


class Anamesis:
    def __init__(self, port, admin_port, policy, weight, memory_limit_mb):
        self.port = port
        self.admin_port = admin_port
        cmd = [ANAMESIS, "--port", str(port), "--admin-port", str(admin_port),
               "--embedding-worker-url", WORKER_URL,
               "--semantic-threshold", "0.80",
               "--no-aof", "--memory-limit", str(memory_limit_mb),
               "--mode", "active", "--policy", policy]
        if weight is not None:
            cmd += ["--semantic-eviction-weight", str(weight)]
        self.proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                                     stderr=subprocess.PIPE, text=True)

    def start(self, timeout=60):
        if not wait_admin(self.admin_port, timeout):
            err = self.proc.stderr.read() if self.proc.stderr else ""
            raise RuntimeError(f"anamesis failed: {err}")
        # wait for the cache port to actually accept connections
        import socket as _socket
        t0 = time.time()
        while time.time() - t0 < timeout:
            try:
                s = _socket.create_connection(("127.0.0.1", self.port), timeout=2)
                s.close()
                return self
            except OSError:
                time.sleep(0.2)
        raise RuntimeError(f"anamesis cache port {self.port} not listening")
        return self

    def stop(self):
        if self.proc:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()


def run_bench(limit, memory_limit_mb, tag):
    report = f"/tmp/opencode/compare_report_{tag}.json"
    cmd = [sys.executable, BENCH, "--mode", "stream", "--limit", str(limit),
           "--stream-get-ratio", "4", "--report", report]
    r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"[compare] bench FAILED rc={r.returncode}")
        print(r.stdout[-2000:])
        print(r.stderr[-2000:])
        raise SystemExit(1)
    with open(report, encoding="utf-8") as f:
        return json.load(f)


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=1500)
    ap.add_argument("--threshold", type=float, default=0.80)
    ap.add_argument("--memory-limit", type=int, default=512,
                    help="anamesis --memory-limit in MB")
    ap.add_argument("--out", default="/tmp/opencode/policy_compare.json")
    ap.add_argument("--plot", default=None, help="write matplotlib plot to PNG")
    args = ap.parse_args(argv)

    llama = LlamaServer()
    llama.start()
    print(f"[compare] llama-server ready")

    rows = []
    for policy, weight in CONFIGS:
        tag = f"{policy}_{weight}" if weight is not None else policy
        print(f"[compare] policy={tag} ...")
        anamesis = Anamesis(6379, 8080, policy, weight, args.memory_limit)
        try:
            anamesis.start()
            report = run_bench(args.limit, args.memory_limit, tag)
            metrics = fetch_metrics(8080)
            best = report["results"][0]
            rows.append({
                "policy": policy,
                "weight": weight,
                "semantic_hit_rate": best["hit_rate"],
                "semantic_hits": best["semantic_hits"],
                "gets": best["gets"],
                "puts": best["puts"],
                "exact_hits": best["exact_hits"],
                "lat_p50_ms": best["lat_p50_ms"],
                "lat_p95_ms": best["lat_p95_ms"],
                "evictions": metrics.get("semcache_evictions_total", 0),
                "semantic_evictions": metrics.get("semcache_semantic_evictions_total", 0),
                "uniqueness_evictions": metrics.get("semcache_uniqueness_evictions_total", 0),
                "memory_bytes": metrics.get("semcache_memory_bytes", 0),
            })
            print(f"     semhit={best['hit_rate']:.3f} ({best['semantic_hits']}/{best['gets']}) "
                  f"puts={best['puts']} "
                  f"evictions={metrics.get('semcache_evictions_total', 0)} "
                  f"sem_ev={metrics.get('semcache_semantic_evictions_total', 0)} "
                  f"uniq_ev={metrics.get('semcache_uniqueness_evictions_total', 0)} "
                  f"mem={metrics.get('semcache_memory_bytes', 0):,}")
        finally:
            anamesis.stop()

    llama.stop()

    # ---- markdown report ----
    print()
    print("| Policy | Weight | SemHitRate | Hits/Gets | Puts | LatP50 | LatP95 | Evict | SemEv | UniqEv | MemMB |")
    print("|--------|--------|------------|-----------|------|--------|--------|-------|-------|--------|-------|")
    for r in rows:
        print(f"| {r['policy']} | {r['weight'] if r['weight'] is not None else '-':>6} "
              f"| {r['semantic_hit_rate']:.3f} | {r['semantic_hits']}/{r['gets']} | "
              f"{r['puts']} | {r['lat_p50_ms']:.1f} | {r['lat_p95_ms']:.1f} | "
              f"{r['evictions']:.0f} | {r['semantic_evictions']:.0f} | "
              f"{r['uniqueness_evictions']:.0f} | {r['memory_bytes']/1e6:.1f} |")

    report = {"memory_limit_mb": args.memory_limit, "limit": args.limit, "rows": rows}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\n[compare] JSON report -> {args.out}")

    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            labels = [f"{r['policy']}\n{w}" if w is not None else r['policy']
                      for r, w in zip(rows, [r['weight'] for r in rows])]
            x = range(len(rows))
            plt.figure(figsize=(8, 4))
            plt.bar(x, [r["f1"] for r in rows], label="F1")
            plt.bar(x, [r["precision"] for r in rows], alpha=0.6, label="Precision")
            plt.bar(x, [r["recall"] for r in rows], alpha=0.4, label="Recall")
            plt.xticks(list(x), labels)
            plt.ylabel("score")
            plt.title(f"Policy comparison (threshold={args.threshold}, top_k=5)")
            plt.legend()
            plt.tight_layout()
            plt.savefig(args.plot)
            print(f"[compare] plot -> {args.plot}")
        except ImportError:
            print("[compare] matplotlib not available; skipping plot")

    return 0


if __name__ == "__main__":
    sys.exit(main())
