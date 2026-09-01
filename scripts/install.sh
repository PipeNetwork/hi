#!/usr/bin/env bash
# Install the `hi` binary into Cargo's bin dir (usually ~/.cargo/bin).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec cargo install --path "$ROOT/crates/hi-cli" --locked "$@"
