#!/usr/bin/env python3
"""Verify the child's independent cgroup fence without inference or a VM supervisor."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import uuid

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--system", action="store_true")
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
build = subprocess.run(["cargo", "test", "--locked", "-p", "hi", "--bin", "hi", "--no-run", "--message-format=json"],
                       cwd=root, check=True, stdout=subprocess.PIPE, text=True)
executables = [event["executable"] for line in build.stdout.splitlines()
               if (event := json.loads(line)).get("reason") == "compiler-artifact"
               and event.get("executable") and event["profile"]["test"] and event["target"]["name"] == "hi"]
if len(executables) != 1:
    raise SystemExit("Expected one hi runner test binary")
name = "runner::control::tests::delegated_lease_fence_kills_the_attempt_without_its_supervisor"
listed = subprocess.check_output([executables[0], "--exact", name, "--ignored", "--list"], text=True)
if name + ": test" not in listed.splitlines():
    raise SystemExit("Independent fence test is missing")
command = (["sudo", "-n", "systemd-run", f"--uid={os.getuid()}", f"--gid={os.getgid()}"]
           if args.system else ["systemd-run", "--user"])
subprocess.run(command + ["--collect", "--wait", "--pipe", f"--unit=hi-fence-fixture-{uuid.uuid4().hex}",
                         "-p", "Delegate=yes", executables[0], "--exact", name, "--ignored", "--nocapture"],
               cwd=root, check=True)
