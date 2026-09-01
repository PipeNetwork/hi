#!/usr/bin/env python3
"""Concurrent-stream decode benchmark for the hi-local CUDA server.

Drives K concurrent OpenAI-API SSE streams with deliberately different prompt
lengths (so sequence positions never align) and reports per-stream TTFT /
inter-token latency plus aggregate decode throughput. Used to track the
vLLM-parity acceptance targets (see .claude plan): aggregate decode should
scale with K instead of staying flat at the single-stream rate.

Usage:
  # Against an already-running server:
  python3 bench/concurrent/bench_concurrent.py --port 8099 --ks 1,2,4,8

  # Launch the server too (kills it on exit):
  python3 bench/concurrent/bench_concurrent.py --launch /home/david/models/qwen25-vl-3b-gguf/model.gguf \
      --ks 1,2,4,8 --label baseline

Results are printed as a table and appended as JSON lines to
bench/concurrent/results.jsonl (one record per (label, K)).
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.jsonl")


def health(port):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=10) as resp:
        return json.loads(resp.read())


def wait_for_server(port, timeout_s=600):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            health(port)
            return
        except (urllib.error.URLError, ConnectionError, OSError):
            time.sleep(1.0)
    raise RuntimeError(f"server on port {port} did not become healthy in {timeout_s}s")


def served_model_id(port):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=10) as resp:
        data = json.loads(resp.read())
    models = data.get("data") or []
    if not models:
        raise RuntimeError("server reports no loaded models")
    return models[0]["id"]


def make_prompt(stream_idx):
    # Different lengths per stream so context lengths never align, and a corpus
    # that is not trivially n-gram-predictable (keeps speculative decode honest).
    words = [
        "river", "granite", "lantern", "orbit", "meadow", "cipher", "violet",
        "harbor", "ember", "thicket", "quartz", "monsoon", "saffron", "glacier",
    ]
    n = 60 + 41 * stream_idx
    body = " ".join(words[(i * 7 + stream_idx) % len(words)] for i in range(n))
    return (
        f"Here are some words: {body}. "
        "Write a short story about a lighthouse keeper. Do not use lists."
    )


def run_stream(port, model_id, prompt, max_tokens, out, idx):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(
            {
                "model": model_id,
                "stream": True,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": 0,
            }
        ).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.monotonic()
    first = None
    last = None
    deltas = 0
    gaps = []
    prev = None
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            for line in resp:
                if not line.startswith(b"data:"):
                    continue
                payload = line[len(b"data:"):].strip()
                if payload == b"[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                if delta.get("content"):
                    now = time.monotonic()
                    if first is None:
                        first = now
                    else:
                        gaps.append(now - prev)
                    prev = now
                    last = now
                    deltas += 1
    except Exception as err:  # noqa: BLE001 - record and keep the run going
        out[idx] = {"error": str(err)}
        return
    out[idx] = {
        "ttft_s": (first - t0) if first else None,
        "deltas": deltas,
        "wall_s": ((last or t0) - t0),
        "decode_wall_s": ((last - first) if (first and last and last > first) else None),
        "itl_p50_ms": statistics.median(gaps) * 1000 if gaps else None,
        "itl_max_ms": max(gaps) * 1000 if gaps else None,
    }


def scheduler_counters(h):
    # /health nests scheduler throughput fields; tolerate shape drift.
    sched = h.get("scheduler") if isinstance(h, dict) else None
    return sched if isinstance(sched, dict) else {}


def run_k(port, model_id, k, max_tokens):
    before = health(port)
    out = [None] * k
    threads = [
        threading.Thread(
            target=run_stream, args=(port, model_id, make_prompt(i), max_tokens, out, i)
        )
        for i in range(k)
    ]
    t0 = time.monotonic()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.monotonic() - t0
    after = health(port)
    ok = [r for r in out if r and "error" not in r]
    errors = [r["error"] for r in out if r and "error" in r]
    total_deltas = sum(r["deltas"] for r in ok)
    return {
        "k": k,
        "streams_ok": len(ok),
        "errors": errors,
        "wall_s": round(wall, 3),
        "aggregate_deltas_per_s": round(total_deltas / wall, 2) if wall > 0 else None,
        "ttft_s": [round(r["ttft_s"], 3) if r["ttft_s"] else None for r in ok],
        "per_stream_decode_tps": [
            round((r["deltas"] - 1) / r["decode_wall_s"], 2)
            if r["decode_wall_s"]
            else None
            for r in ok
        ],
        "itl_p50_ms": [r["itl_p50_ms"] and round(r["itl_p50_ms"], 1) for r in ok],
        "itl_max_ms": [r["itl_max_ms"] and round(r["itl_max_ms"], 1) for r in ok],
        "health_before": scheduler_counters(before),
        "health_after": scheduler_counters(after),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8099)
    parser.add_argument("--ks", default="1,2,4,8")
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--label", default="")
    parser.add_argument(
        "--launch",
        metavar="MODEL_GGUF",
        help="launch target/release/hi-local serve with this model (else assume a running server)",
    )
    parser.add_argument("--warmup", type=int, default=1, help="warmup streams before measuring")
    args = parser.parse_args()

    server = None
    try:
        if args.launch:
            binary = os.path.join(REPO_ROOT, "target", "release", "hi-local")
            server = subprocess.Popen(
                [
                    binary, "serve", "--backend", "cuda", "--execution", "gpu",
                    "--port", str(args.port), args.launch,
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        wait_for_server(args.port)
        model_id = served_model_id(args.port)

        for _ in range(args.warmup):
            out = [None]
            run_stream(args.port, model_id, make_prompt(0), 32, out, 0)
            if out[0] and "error" in out[0]:
                raise RuntimeError(f"warmup stream failed: {out[0]['error']}")

        records = []
        for k in [int(v) for v in args.ks.split(",") if v]:
            record = run_k(args.port, model_id, k, args.max_tokens)
            record["label"] = args.label
            record["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            records.append(record)
            agg = record["aggregate_deltas_per_s"]
            print(
                f"K={k}: aggregate ~{agg} deltas/s | "
                f"TTFT {record['ttft_s']} s | per-stream decode {record['per_stream_decode_tps']} tok/s | "
                f"ITL p50 {record['itl_p50_ms']} ms max {record['itl_max_ms']} ms"
                + (f" | errors: {record['errors']}" if record["errors"] else "")
            )

        with open(RESULTS_PATH, "a", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")
        print(f"appended {len(records)} record(s) to {RESULTS_PATH}")
    finally:
        if server is not None:
            server.terminate()
            try:
                server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                server.kill()


if __name__ == "__main__":
    sys.exit(main())
