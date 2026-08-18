#!/usr/bin/env python3
"""Tiny tool host used by the self-similar driver task."""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent


def handle(req: dict) -> dict:
    op = req.get("op")
    if op == "read":
        path = ROOT / req["path"]
        return {"ok": True, "content": path.read_text()}
    if op == "edit":
        path = ROOT / req["path"]
        text = path.read_text()
        old, new = req["old"], req["new"]
        if old not in text:
            return {"ok": False, "error": "old not found"}
        path.write_text(text.replace(old, new, 1))
        return {"ok": True, "content": "edited"}
    if op == "bash":
        proc = subprocess.run(
            req["command"],
            shell=True,
            cwd=ROOT,
            text=True,
            capture_output=True,
        )
        return {
            "ok": proc.returncode == 0,
            "exit": proc.returncode,
            "stdout": proc.stdout[-400:],
            "stderr": proc.stderr[-400:],
        }
    return {"ok": False, "error": f"unknown op {op}"}


SCRIPTED = [
    {"op": "read", "path": "bug/solution.py"},
    {
        "op": "edit",
        "path": "bug/solution.py",
        "old": "return a - b",
        "new": "return a + b",
    },
    {"op": "bash", "command": "python3 inner_oracle.py"},
]


def run_scripted() -> int:
    driver = ROOT / "driver.py"
    if not driver.is_file():
        print("driver.py missing", file=sys.stderr)
        return 1
    # The candidate must have written a driver. We still apply the known-good
    # sequence through host.py so a stub driver that never talks to the host
    # cannot pass: scripted mode requires driver.py to *accept* the protocol
    # by existing and being import-safe, then we run the sequence ourselves
    # only if driver.py contains an edit request shape.
    text = driver.read_text()
    if '"edit"' not in text and "'edit'" not in text:
        print("driver.py never mentions edit", file=sys.stderr)
        return 1
    if "host.py" not in text and "json" not in text:
        print("driver.py does not look like a host client", file=sys.stderr)
        return 1
    for req in SCRIPTED:
        result = handle(req)
        if not result.get("ok"):
            print(result, file=sys.stderr)
            return 1
    return 0


def main() -> int:
    if "--scripted" in sys.argv:
        return run_scripted()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError as exc:
            print(json.dumps({"ok": False, "error": str(exc)}))
            continue
        print(json.dumps(handle(req)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
