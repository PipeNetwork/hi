#!/usr/bin/env bash
# Build a static Linux `hi` binary for Terminal-Bench task containers.
#
# Builds inside rust:alpine (musl → fully static), so the binary runs in any
# Linux task image regardless of its glibc. Defaults to the host architecture,
# which is also what local Docker task containers run.
#
#   ./build-linux.sh            # host arch
#   ./build-linux.sh x86_64     # or aarch64
set -euo pipefail

cd "$(dirname "$0")"
REPO_ROOT="$(cd ../.. && pwd)"

ARCH="${1:-$(uname -m)}"
case "$ARCH" in
  aarch64 | arm64) PLATFORM=linux/arm64 ;;
  x86_64 | amd64) PLATFORM=linux/amd64 ;;
  *)
    echo "unsupported arch: $ARCH (use aarch64 or x86_64)" >&2
    exit 1
    ;;
esac

# Rust version pinned to the repo toolchain (rust-toolchain.toml).
RUST_VERSION="$(sed -n 's/^channel = "\(.*\)"/\1/p' "$REPO_ROOT/rust-toolchain.toml")"

# Separate target dir so Linux artifacts never collide with host builds;
# named volume caches the registry between runs.
docker run --rm --platform "$PLATFORM" \
  -v "$REPO_ROOT":/src -w /src \
  -v hi-tb-cargo-registry:/usr/local/cargo/registry \
  -e CARGO_TARGET_DIR="/src/bench/terminal-bench/.build-$ARCH" \
  "rust:$RUST_VERSION-alpine" \
  sh -c 'apk add --no-cache musl-dev build-base && cargo build --release -p hi --no-default-features'

mkdir -p dist
cp ".build-$ARCH/release/hi" dist/hi-linux
echo "built: $(pwd)/dist/hi-linux ($ARCH)"
