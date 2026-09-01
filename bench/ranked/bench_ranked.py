#!/usr/bin/env python3
"""Ranked-sampling decode A/B: single stream, top_k/top_p config, measures
decode tok/s (client-side, after first token). Usage:
  ranked_ab.py <gguf> <label> [K=V env ...]
"""
import json
import os
import subprocess
import sys
import time
import urllib.request

GGUF = sys.argv[1]
LABEL = sys.argv[2]
ENV_EXTRA = dict(kv.split("=", 1) for kv in sys.argv[3:])
PORT = 8098
BASE = f"http://127.0.0.1:{PORT}"


def http_json(path):
    with urllib.request.urlopen(f"{BASE}{path}", timeout=10) as r:
        return json.load(r)


def wait_server(proc, deadline=300):
    start = time.time()
    while time.time() - start < deadline:
        if proc.poll() is not None:
            sys.exit(f"server died: rc={proc.returncode}")
        try:
            http_json("/v1/models")
            return
        except Exception:
            time.sleep(0.5)
    sys.exit("server not ready")


def run_stream(model, max_tokens=256, temperature=0.8, top_p=0.95, top_k=40, seed=1234):
    body = json.dumps(
        {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": "Write a detailed essay about the history of container shipping.",
                }
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "seed": seed,
            "stream": True,
        }
    ).encode()
    req = urllib.request.Request(
        f"{BASE}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    first = None
    last = None
    count = 0
    text = []
    with urllib.request.urlopen(req, timeout=600) as r:
        for line in r:
            if not line.startswith(b"data:"):
                continue
            payload = line[len(b"data:") :].strip()
            if payload == b"[DONE]":
                break
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue
            choices = chunk.get("choices") or []
            if not choices:
                continue
            piece = (choices[0].get("delta") or {}).get("content")
            if piece:
                now = time.monotonic()
                if first is None:
                    first = now
                last = now
                count += 1
                text.append(piece)
    rate = (count - 1) / (last - first) if count > 1 and last > first else 0.0
    return rate, count, "".join(text)


env = os.environ.copy()
env.update(ENV_EXTRA)
proc = subprocess.Popen(
    [
        "./target/release/hi-local",
        "serve",
        "--backend",
        "cuda",
        "--execution",
        "gpu",
        "--port",
        str(PORT),
        GGUF,
    ],
    env=env,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
    cwd="/home/david/hi",
)
try:
    wait_server(proc)
    model = http_json("/v1/models")["data"][0]["id"]
    run_stream(model, max_tokens=32)  # warmup
    rates = []
    text = ""
    for i in range(3):
        rate, count, text = run_stream(model, seed=1234)
        rates.append(rate)
        print(f"[{LABEL}] run {i + 1}: {rate:.1f} tok/s ({count} deltas)", flush=True)
    health = http_json("/health")
    sched = health.get("scheduler") or {}
    print(f"[{LABEL}] median {sorted(rates)[1]:.1f} tok/s", flush=True)
    out = f"/tmp/claude-1000/-home-david-hi/d8b1f159-47bc-4573-969b-e760573d67f9/scratchpad/ranked_ab_{LABEL}.json"
    with open(out, "w") as f:
        json.dump({"label": LABEL, "rates": rates, "last_text": text}, f, indent=1)
    print(f"[{LABEL}] wrote {out}", flush=True)
finally:
    proc.terminate()
    proc.wait(timeout=30)
