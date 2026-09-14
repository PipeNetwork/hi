#!/usr/bin/env bash
# Install `hi` and `hi-sentinel` into Cargo's bin dir (usually ~/.cargo/bin).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cargo install --path "$ROOT/crates/hi-cli" --locked "$@"
cargo install --path "$ROOT/crates/hi-sentinel" --locked "$@"
