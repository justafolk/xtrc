#!/usr/bin/env python3
from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import httpx


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark xtrc query latency")
    parser.add_argument("query", help="Natural-language query")
    parser.add_argument("--repo", default=".", help="Repository path")
    parser.add_argument("--host", default="127.0.0.1", help="Server host")
    parser.add_argument("--port", type=int, default=8765, help="Server port")
    parser.add_argument("--runs", type=int, default=50, help="Number of timed runs")
    parser.add_argument("--top-k", type=int, default=8, help="Top-k results")
    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}"
    repo_path = str(Path(args.repo).expanduser().resolve())

    with httpx.Client(timeout=30.0) as client:
        warmup_payload = {"repo_path": repo_path, "query": args.query, "top_k": args.top_k}
        client.post(f"{base_url}/query", json=warmup_payload).raise_for_status()

        elapsed_ms: list[float] = []
        for _ in range(args.runs):
            started = time.perf_counter()
            response = client.post(f"{base_url}/query", json=warmup_payload)
            response.raise_for_status()
            elapsed_ms.append((time.perf_counter() - started) * 1000.0)

    elapsed_ms.sort()
    p50 = statistics.median(elapsed_ms)
    p95 = elapsed_ms[int(len(elapsed_ms) * 0.95) - 1]
    p99 = elapsed_ms[int(len(elapsed_ms) * 0.99) - 1]
    mean = statistics.fmean(elapsed_ms)

    print(f"runs={args.runs}")
    print(f"mean_ms={mean:.3f}")
    print(f"p50_ms={p50:.3f}")
    print(f"p95_ms={p95:.3f}")
    print(f"p99_ms={p99:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
