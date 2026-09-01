#!/usr/bin/env python3
"""4 concurrent ranked streams: aggregate decode tok/s. Usage: ranked_k4.py <gguf> <label> [K=V ...]"""
import json
import os
import subprocess
import sys
import threading
import time
import urllib.request

GGUF = sys.argv[1]
LABEL = sys.argv[2]
ENV_EXTRA = dict(kv.split("=", 1) for kv in sys.argv[3:])
PORT = 8097
BASE = f"http://127.0.0.1:{PORT}"


def http_json(path):
    with urllib.request.urlopen(f"{BASE}{path}", timeout=10) as r:
        return json.load(r)


def wait_server(proc, deadline=300):
    start = time.time()
    while time.time() - start < deadline:
        if proc.poll() is not None:
            sys.exit(f"server died rc={proc.returncode}")
        try:
            http_json("/v1/models")
            return
        except Exception:
            time.sleep(0.5)
    sys.exit("server not ready")


def stream(model, seed, out):
    body = json.dumps(
        {
            "model": model,
            "messages": [
                {"role": "user", "content": f"Tell me a long story about expedition {seed}."}
            ],
            "max_tokens": 256,
            "temperature": 0.8,
            "top_p": 0.95,
            "top_k": 40,
            "seed": seed,
            "stream": True,
        }
    ).encode()
    req = urllib.request.Request(
        f"{BASE}/v1/chat/completions", data=body, headers={"Content-Type": "application/json"}
    )
    first = last = None
    count = 0
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
            if choices and (choices[0].get("delta") or {}).get("content"):
                now = time.monotonic()
                if first is None:
                    first = now
                last = now
                count += 1
    out.append((first, last, count))


env = os.environ.copy()
env.update(ENV_EXTRA)
proc = subprocess.Popen(
    ["./target/release/hi-local", "serve", "--backend", "cuda", "--execution", "gpu",
     "--port", str(PORT), GGUF],
    env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, cwd="/home/david/hi",
)
try:
    wait_server(proc)
    model = http_json("/v1/models")["data"][0]["id"]
    warm = []
    stream(model, 1, warm)  # warmup
    for rep in range(2):
        results = []
        threads = [threading.Thread(target=stream, args=(model, 100 + i, results)) for i in range(4)]
        t0 = time.monotonic()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        tokens = sum(c for _, _, c in results)
        starts = [f for f, _, _ in results if f]
        ends = [l for _, l, _ in results if l]
        span = max(ends) - min(starts)
        print(f"[{LABEL}] rep {rep + 1}: aggregate {tokens / span:.0f} tok/s ({tokens} tokens / {span:.1f}s)", flush=True)
finally:
    proc.terminate()
    proc.wait(timeout=30)
