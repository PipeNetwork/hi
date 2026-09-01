#!/usr/bin/env bash
# Install a verified hi-local-runtime bundle beside the hi executable.
set -Eeuo pipefail

BACKEND="${1:-${HI_LOCAL_RUNTIME_BACKEND:-}}"
ARCHIVE_SOURCE="${HI_LOCAL_RUNTIME_ARCHIVE:-}"
EXPECTED_SHA256="${HI_LOCAL_RUNTIME_SHA256:-}"
DESTINATION="${HI_LOCAL_RUNTIME_DEST:-}"

usage() {
  echo "usage: $0 cpu|mlx|cuda" >&2
  echo "       HI_LOCAL_RUNTIME_ARCHIVE=/path/or/https-url" >&2
  echo "       HI_LOCAL_RUNTIME_SHA256=<64-hex-digest>" >&2
}

[[ "$BACKEND" == mlx || "$BACKEND" == cuda ]] || {
  usage
  exit 2
}
[[ -n "$ARCHIVE_SOURCE" ]] || {
  echo "set HI_LOCAL_RUNTIME_ARCHIVE to a runtime .tar.gz path or HTTPS URL" >&2
  exit 2
}

if [[ -z "$DESTINATION" ]]; then
  HI_EXECUTABLE="${HI_BIN:-}"
  if [[ -z "$HI_EXECUTABLE" ]]; then
    HI_EXECUTABLE="$(command -v hi 2>/dev/null || true)"
  fi
  if [[ -n "$HI_EXECUTABLE" ]]; then
    DESTINATION="$(cd "$(dirname "$HI_EXECUTABLE")" && pwd -P)"
  else
    DESTINATION="$(pwd -P)/target/hi-local-runtime"
  fi
fi

temp_root="$(mktemp -d "${TMPDIR:-/tmp}/hi-local-runtime.XXXXXX")"
cleanup() {
  rm -rf "$temp_root"
}
trap cleanup EXIT

archive="$temp_root/runtime.tar.gz"
case "$ARCHIVE_SOURCE" in
  https://*) curl --fail --location --silent --show-error "$ARCHIVE_SOURCE" --output "$archive" ;;
  http://*)
    echo "runtime bundle URLs must use HTTPS" >&2
    exit 2
    ;;
  *) cp "$ARCHIVE_SOURCE" "$archive" ;;
esac

[[ -s "$archive" ]] || {
  echo "runtime archive is empty: $ARCHIVE_SOURCE" >&2
  exit 1
}

if [[ -n "$EXPECTED_SHA256" ]]; then
  [[ "$EXPECTED_SHA256" =~ ^[0-9A-Fa-f]{64}$ ]] || {
    echo "HI_LOCAL_RUNTIME_SHA256 must be exactly 64 hexadecimal characters" >&2
    exit 2
  }
  if command -v sha256sum >/dev/null 2>&1; then
    actual_sha256="$(sha256sum "$archive" | awk '{print $1}')"
  elif command -v shasum >/dev/null 2>&1; then
    actual_sha256="$(shasum -a 256 "$archive" | awk '{print $1}')"
  else
    echo "need sha256sum or shasum to verify the runtime archive" >&2
    exit 1
  fi
  actual_sha256_lower="$(printf '%s' "$actual_sha256" | tr '[:upper:]' '[:lower:]')"
  expected_sha256_lower="$(printf '%s' "$EXPECTED_SHA256" | tr '[:upper:]' '[:lower:]')"
  [[ "$actual_sha256_lower" == "$expected_sha256_lower" ]] || {
    echo "runtime archive checksum mismatch" >&2
    exit 1
  }
else
  echo "set HI_LOCAL_RUNTIME_SHA256; refusing an unverified runtime archive" >&2
  exit 2
fi

archive_listing="$temp_root/archive.list"
tar -tzf "$archive" > "$archive_listing"
while IFS= read -r entry; do
  case "$entry" in
    /*|../*|*/../*|..)
      echo "runtime archive contains an unsafe path: $entry" >&2
      exit 1
      ;;
  esac
done < "$archive_listing"
tar -xzf "$archive" -C "$temp_root"
bundle_root="$temp_root"
runtime_json="$bundle_root/runtime.json"
if [[ ! -f "$runtime_json" ]]; then
  for candidate in "$temp_root"/*/runtime.json; do
    if [[ -f "$candidate" ]]; then
      runtime_json="$candidate"
      bundle_root="$(dirname "$candidate")"
      break
    fi
  done
fi
sidecar="$bundle_root/bin/hi-local"
[[ -f "$runtime_json" && -f "$sidecar" ]] || {
  echo "runtime archive must contain runtime.json and bin/hi-local" >&2
  exit 1
}

metadata_value() {
  local key="$1"
  sed -n "s/^[[:space:]]*\"${key}\"[[:space:]]*:[[:space:]]*\"\([^\"]*\)\".*$/\1/p" "$runtime_json" | head -n 1
}

protocol="$(metadata_value protocol_version)"
[[ "$protocol" =~ ^1\.[0-9]+$ ]] || {
  echo "runtime protocol ${protocol:-<missing>} is not supported by this hi" >&2
  exit 1
}
advertised_backend="$(metadata_value backend)"
[[ "$advertised_backend" == "$BACKEND" ]] || {
  echo "runtime backend ${advertised_backend:-<missing>} does not match requested $BACKEND" >&2
  exit 1
}
required_platform="$(metadata_value requirements)"
host_platform="$(uname -s)-$(uname -m)"
[[ "$required_platform" == "$host_platform" ]] || {
  echo "runtime requires ${required_platform:-<missing>}; this host is $host_platform" >&2
  exit 1
}

mkdir -p "$DESTINATION"
install -m 0755 "$sidecar" "$DESTINATION/hi-local"
echo "installed verified hi-local $BACKEND sidecar at $DESTINATION/hi-local"
