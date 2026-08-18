"""Reference driver: talks JSON lines to host.py and stops on a green oracle."""
import json
import subprocess
import sys

def send(proc, req):
    proc.stdin.write(json.dumps(req) + "\n")
    proc.stdin.flush()
    line = proc.stdout.readline()
    return json.loads(line)

def main():
    proc = subprocess.Popen(
        [sys.executable, "host.py"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    read = send(proc, {"op": "read", "path": "bug/solution.py"})
    content = read.get("content", "")[:4000]
    send(
        proc,
        {
            "op": "edit",
            "path": "bug/solution.py",
            "old": "return a - b",
            "new": "return a + b",
        },
    )
    result = send(proc, {"op": "bash", "command": "python3 inner_oracle.py"})
    proc.stdin.close()
    proc.wait(timeout=10)
    if not result.get("ok"):
        raise SystemExit(1)
    _ = content

if __name__ == "__main__":
    main()
