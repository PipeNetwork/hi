#!/usr/bin/env python3
"""Agent-loop prefix-cache A/B: 3 growing chat turns, greedy, single stream.

Usage: prefix_ab.py <gguf> <label> [extra env as K=V ...]
Starts hi-local serve itself, runs 3 turns where each request resends the whole
conversation plus a new user message (the agent-loop shape), prints per-turn
TTFT + health reuse counters, dumps completions to JSON for cross-config diff.
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
PORT = 8099
BASE = f"http://127.0.0.1:{PORT}"

FIRST_PROMPT = (
    "You are a meticulous software archaeologist. Analyze the following "
    "changelog and answer questions about it.\n\n"
    + "\n".join(
        f"- v0.{i}: module m{i % 17} refactored; bug #{1000 + i * 7} fixed; "
        f"throughput {'improved' if i % 3 else 'regressed'} by {i % 23}%"
        for i in range(1, 140)
    )
    + "\n\nQuestion 1: Which versions regressed throughput? Answer briefly."
)
FOLLOWUPS = [
    "Question 2: Which module was touched most often? Answer briefly.",
    "Question 3: Summarize the overall trend in one sentence.",
]


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
    sys.exit("server did not become ready")


def stream_turn(model, messages, max_tokens=64):
    body = json.dumps(
        {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0,
            "stream": True,
        }
    ).encode()
    req = urllib.request.Request(
        f"{BASE}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.monotonic()
    ttft = None
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
                if ttft is None:
                    ttft = time.monotonic() - t0
                text.append(piece)
    return ttft, "".join(text), time.monotonic() - t0


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
    messages = [{"role": "user", "content": FIRST_PROMPT}]
    results = []
    for i in range(3):
        ttft, text, total = stream_turn(model, messages)
        results.append(
            {
                "turn": i + 1,
                "conv_chars": sum(len(m["content"]) for m in messages),
                "ttft_s": ttft,
                "total_s": total,
                "text": text,
            }
        )
        print(
            f"[{LABEL}] turn {i + 1}: conversation {results[-1]['conv_chars']} chars, "
            f"ttft {ttft * 1000:.0f} ms, total {total:.2f}s",
            flush=True,
        )
        messages.append({"role": "assistant", "content": text})
        if i < 2:
            messages.append({"role": "user", "content": FOLLOWUPS[i]})
    health = http_json("/health")
    pc = json.dumps(health.get("prefix_cache", {}))
    print(f"[{LABEL}] prefix_cache: {pc}", flush=True)
    out = f"/tmp/claude-1000/-home-david-hi/d8b1f159-47bc-4573-969b-e760573d67f9/scratchpad/prefix_ab_{LABEL}.json"
    with open(out, "w") as f:
        json.dump(
            {"label": LABEL, "results": results, "prefix_cache": health.get("prefix_cache")},
            f,
            indent=1,
        )
    print(f"[{LABEL}] wrote {out}", flush=True)
finally:
    proc.terminate()
    proc.wait(timeout=30)
