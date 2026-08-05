#!/usr/bin/env bash
set -euo pipefail

# Build a relocatable macOS Apple-Silicon bundle containing hi and the
# matching MLX sidecar/runtime. The source MLX prefix must match the vendored
# MLX C ABI; see docs/hy_v3-and-prebuilt-mlx.md.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PREFIX="${HI_MLX_SYSTEM_MLX_PREFIX:-}"
OUT="${HI_MLX_PACKAGE_OUT:-$ROOT/target/hi-mlx-bundle}"

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
  echo "MLX bundles require macOS arm64" >&2
  exit 2
fi
if [[ -z "$PREFIX" ]]; then
  echo "set HI_MLX_SYSTEM_MLX_PREFIX to a matching prebuilt MLX prefix" >&2
  exit 2
fi
if [[ ! -f "$PREFIX/lib/libmlx.dylib" || ! -f "$PREFIX/lib/mlx.metallib" ]]; then
  echo "MLX prefix must contain lib/libmlx.dylib and lib/mlx.metallib: $PREFIX" >&2
  exit 2
fi

cd "$ROOT"
cargo build --release -p hi
HI_MLX_SYSTEM_MLX_PREFIX="$PREFIX" HI_MLX_BUNDLE_RPATH=1 \
  cargo build --release -p hi-local --features mlx

rm -rf "$OUT"
mkdir -p "$OUT/bin" "$OUT/lib/mlx"
cp target/release/hi "$OUT/bin/hi"
cp target/release/hi-local "$OUT/bin/hi-local"
cp "$PREFIX/lib/libmlx.dylib" "$OUT/lib/mlx/libmlx.dylib"
cp "$PREFIX/lib/mlx.metallib" "$OUT/lib/mlx/mlx.metallib"

echo "created relocatable MLX bundle at $OUT"
