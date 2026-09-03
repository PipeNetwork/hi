#!/usr/bin/env bash
set -euo pipefail

# Build the external Linux sandbox artifact without making hi depend on an
# unstable Rust library API. The revision is deliberately recorded in a small
# lock file so upstream updates are reviewable and reproducible.
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"
lock_file="$script_dir/pipe-wrap.lock"
source_dir="${PIPE_WRAP_SOURCE_DIR:-$repo_root/.cache/pipe-wrap-src}"
artifact_dir="${PIPE_WRAP_ARTIFACT_DIR:-$repo_root/target/pipe-wrap}"
target_triple="${PIPE_WRAP_TARGET:-x86_64-unknown-linux-gnu}"
repository="$(sed -n 's/^repository=//p' "$lock_file")"
revision="$(sed -n 's/^revision=//p' "$lock_file")"

if [[ ! -d "$source_dir/.git" ]]; then
  mkdir -p "$(dirname "$source_dir")"
  git clone "$repository" "$source_dir"
else
  git -C "$source_dir" fetch --tags origin
fi
git -C "$source_dir" checkout --detach "$revision"
cargo build --manifest-path "$source_dir/Cargo.toml" --locked --release --bin pipe-wrap --target "$target_triple"
mkdir -p "$artifact_dir/$target_triple"
cp "$source_dir/target/$target_triple/release/pipe-wrap" "$artifact_dir/$target_triple/pipe-wrap"
echo "built $artifact_dir/$target_triple/pipe-wrap from pipe-wrap $revision"
