#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
manifest="${HI_CUDA_FIXTURE_MANIFEST:-"$script_dir/cuda-fixtures.tsv"}"
target_root="${HI_CUDA_FIXTURES_DIR:-}"

die() {
  printf 'fetch-cuda-fixtures: %s\n' "$*" >&2
  exit 1
}

if [[ -z "$target_root" ]]; then
  die "set HI_CUDA_FIXTURES_DIR to the directory where fixtures should be stored"
fi
if [[ ! -f "$manifest" ]]; then
  die "manifest not found: $manifest"
fi
if ! command -v sha256sum >/dev/null 2>&1; then
  die "sha256sum is required"
fi

mkdir -p "$target_root"

sha256_of() {
  sha256sum "$1" | awk '{print $1}'
}

verify_sha256() {
  local path="$1"
  local expected="$2"
  local actual
  actual="$(sha256_of "$path")"
  [[ "$actual" == "$expected" ]]
}

download_to() {
  local url="$1"
  local output="$2"
  case "$url" in
    file://*)
      cp "${url#file://}" "$output"
      ;;
    *)
      if ! command -v curl >/dev/null 2>&1; then
        die "curl is required for non-file fixture URLs"
      fi
      curl -L --fail --retry 3 --retry-delay 2 -o "$output" "$url"
      ;;
  esac
}

rows=0
fixture_shas=()
fixture_paths=()

remember_fixture() {
  fixture_shas+=("$1")
  fixture_paths+=("$2")
}

fixture_for_sha() {
  local needle="$1"
  local idx
  for idx in "${!fixture_shas[@]}"; do
    if [[ "${fixture_shas[$idx]}" == "$needle" && -f "${fixture_paths[$idx]}" ]]; then
      printf '%s\n' "${fixture_paths[$idx]}"
      return 0
    fi
  done
  return 1
}

while IFS=$'\t' read -r relative_path url sha256 family architecture quant_type extra || [[ -n "${relative_path:-}" ]]; do
  relative_path="${relative_path%$'\r'}"
  url="${url%$'\r'}"
  sha256="${sha256%$'\r'}"
  family="${family%$'\r'}"
  architecture="${architecture%$'\r'}"
  quant_type="${quant_type%$'\r'}"
  extra="${extra%$'\r'}"

  if [[ -z "$relative_path" || "${relative_path:0:1}" == "#" ]]; then
    continue
  fi
  if [[ -n "${extra:-}" ]]; then
    die "manifest row for $relative_path has too many columns"
  fi
  if [[ -z "$url" || -z "$sha256" || -z "$family" || -z "$architecture" || -z "$quant_type" ]]; then
    die "manifest row for $relative_path is missing required fields"
  fi
  case "$relative_path" in
    /*|../*|*/../*|*/..)
      die "manifest path must stay under HI_CUDA_FIXTURES_DIR: $relative_path"
      ;;
  esac
  if [[ ! "$sha256" =~ ^[0-9a-fA-F]{64}$ ]]; then
    die "manifest row for $relative_path has invalid SHA-256: $sha256"
  fi

  target="$target_root/$relative_path"
  mkdir -p "$(dirname "$target")"
  if [[ -f "$target" ]]; then
    if verify_sha256 "$target" "$sha256"; then
      printf 'ok %s (%s/%s/%s)\n' "$relative_path" "$family" "$architecture" "$quant_type"
      remember_fixture "$sha256" "$target"
      rows=$((rows + 1))
      continue
    fi
    printf 'replace %s: checksum mismatch on existing file\n' "$relative_path" >&2
  fi

  source_for_sha="$(fixture_for_sha "$sha256" || true)"
  if [[ -n "$source_for_sha" ]]; then
    cp "$source_for_sha" "$target"
    if verify_sha256 "$target" "$sha256"; then
      printf 'copied %s (%s/%s/%s)\n' "$relative_path" "$family" "$architecture" "$quant_type"
      rows=$((rows + 1))
      continue
    fi
    rm -f "$target"
    die "internal copy failed checksum verification for $relative_path"
  fi

  part="$target.part.$$"
  rm -f "$part"
  if ! download_to "$url" "$part"; then
    rm -f "$part"
    die "download failed for $relative_path from $url"
  fi
  actual="$(sha256_of "$part")"
  if [[ "$actual" != "$sha256" ]]; then
    rm -f "$part"
    die "checksum mismatch for $relative_path: expected $sha256, got $actual"
  fi
  mv "$part" "$target"
  remember_fixture "$sha256" "$target"
  printf 'fetched %s (%s/%s/%s)\n' "$relative_path" "$family" "$architecture" "$quant_type"
  rows=$((rows + 1))
done < "$manifest"

if [[ "$rows" -eq 0 ]]; then
  die "manifest contained no fixture rows: $manifest"
fi

printf 'fixtures ready in %s (%s file(s))\n' "$target_root" "$rows"
