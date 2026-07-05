#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_ROOT="${HI_MLX_MODELS_DIR:-$ROOT/.hi/models}"
HOST="${HI_MLX_HOST:-127.0.0.1}"
PORT_BASE="${HI_MLX_PORT_BASE:-18080}"
MAX_TOKENS="${HI_MLX_MAX_TOKENS:-64}"
HEALTH_TIMEOUT="${HI_MLX_HEALTH_TIMEOUT:-900}"
ARTIFACT_ROOT="${HI_MLX_ACCEPTANCE_ARTIFACTS:-$ROOT/target/hi-mlx-acceptance}"
RUN_ID="$(date +%Y%m%d-%H%M%S)"
ARTIFACT_DIR="$ARTIFACT_ROOT/$RUN_ID"
BIN="${HI_MLX_BIN:-$ROOT/target/debug/hi-mlx}"

REPOS=(
  "mlx-community/Qwen3-0.6B-4bit"
  "mlx-community/DeepSeek-V3-4bit"
  "mlx-community/DeepSeek-V3.2-4bit"
  "mlx-community/DeepSeek-V4-Flash-4bit"
  "mlx-community/GLM-4.7-Flash-4bit"
  "mlx-community/GLM-4.7-Flash-8bit"
)

usage() {
  cat <<'EOF'
Usage: scripts/hi_mlx_acceptance_matrix.sh [options] [repo ...]

Runs the native hi-mlx acceptance smoke matrix:
  inspect, serve, /health, /v1/models, non-streaming chat, streaming chat,
  and a tool-call prompt for each repo.

Options:
  --no-download     Require model directories to already exist.
  --skip-build      Do not run cargo build -p hi-mlx.
  --skip-unit       Do not run cargo test -p hi-mlx before the matrix.
  --skip-tool       Skip the tool-call smoke check.
  -h, --help        Show this help.

Environment:
  HI_MLX_MODELS_DIR              Default: .hi/models
  HI_MLX_BIN                     Default: target/debug/hi-mlx
  HI_MLX_PORT_BASE               Default: 18080
  HI_MLX_MAX_TOKENS              Default: 64
  HI_MLX_HEALTH_TIMEOUT          Default: 900 seconds
  HI_MLX_ACCEPTANCE_ARTIFACTS    Default: target/hi-mlx-acceptance

Examples:
  scripts/hi_mlx_acceptance_matrix.sh
  HI_MLX_MODELS_DIR=/Volumes/models scripts/hi_mlx_acceptance_matrix.sh
  scripts/hi_mlx_acceptance_matrix.sh --no-download mlx-community/Qwen3-0.6B-4bit
EOF
}

download_missing=1
run_build=1
run_unit=1
run_tool=1
selected_repos=()

while (($#)); do
  case "$1" in
    --no-download)
      download_missing=0
      ;;
    --skip-build)
      run_build=0
      ;;
    --skip-unit)
      run_unit=0
      ;;
    --skip-tool)
      run_tool=0
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    -*)
      echo "unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
    *)
      selected_repos+=("$1")
      ;;
  esac
  shift
done

if ((${#selected_repos[@]})); then
  REPOS=("${selected_repos[@]}")
fi

log() {
  printf '\n[%s] %s\n' "$(date +%H:%M:%S)" "$*"
}

safe_path() {
  local input="$1"
  local out
  out="$(printf '%s' "$input" | sed -E 's/[^A-Za-z0-9._-]+/_/g; s/^_+//; s/_+$//')"
  if [[ -z "$out" ]]; then
    out="download"
  fi
  printf '%.160s' "$out"
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "missing required command: $1" >&2
    exit 1
  fi
}

download_repo() {
  local repo="$1"
  local dir="$2"
  mkdir -p "$dir"
  if command -v hf >/dev/null 2>&1; then
    hf download "$repo" --local-dir "$dir"
  elif command -v huggingface-cli >/dev/null 2>&1; then
    huggingface-cli download "$repo" --local-dir "$dir"
  else
    cat >&2 <<EOF
Model directory is missing and no Hugging Face downloader was found:
  $dir

Install one of:
  python3 -m pip install --user huggingface_hub
  python3 -m pip install --user 'huggingface_hub[cli]'
EOF
    exit 1
  fi
}

wait_for_health() {
  local base_url="$1"
  local out="$2"
  local deadline=$((SECONDS + HEALTH_TIMEOUT))
  local last="$out.health.last.json"
  while ((SECONDS < deadline)); do
    if curl -fsS "$base_url/health" >"$last" 2>"$out.health.err"; then
      if python3 - "$last" <<'PY'
import json, sys
with open(sys.argv[1], "r", encoding="utf-8") as f:
    body = json.load(f)
if body.get("ready") is True:
    sys.exit(0)
print(body)
sys.exit(1)
PY
      then
        cp "$last" "$out.health.json"
        return 0
      fi
    fi
    sleep 2
  done
  echo "timed out waiting for healthy hi-mlx at $base_url" >&2
  if [[ -s "$last" ]]; then
    cat "$last" >&2
    echo >&2
  fi
  return 1
}

post_json() {
  local url="$1"
  local data="$2"
  local output="$3"
  curl -fsS "$url" \
    -H 'content-type: application/json' \
    -d "$data" \
    >"$output"
}

validate_nonstream() {
  local path="$1"
  python3 - "$path" <<'PY'
import json, sys
with open(sys.argv[1], "r", encoding="utf-8") as f:
    body = json.load(f)
choice = body["choices"][0]
content = choice["message"].get("content")
if not isinstance(content, str) or not content.strip():
    raise SystemExit(f"assistant content is empty: {body}")
PY
}

validate_stream() {
  local path="$1"
  python3 - "$path" <<'PY'
import json, sys
done = False
content = []
with open(sys.argv[1], "r", encoding="utf-8") as f:
    for raw in f:
        raw = raw.strip()
        if not raw.startswith("data: "):
            continue
        data = raw[6:]
        if data == "[DONE]":
            done = True
            continue
        body = json.loads(data)
        for choice in body.get("choices", []):
            delta = choice.get("delta", {})
            if isinstance(delta.get("content"), str):
                content.append(delta["content"])
if not done:
    raise SystemExit("stream did not emit [DONE]")
if not "".join(content).strip():
    raise SystemExit("stream did not emit non-empty content delta")
PY
}

validate_tool_call() {
  local path="$1"
  python3 - "$path" <<'PY'
import json, sys
with open(sys.argv[1], "r", encoding="utf-8") as f:
    body = json.load(f)
message = body["choices"][0]["message"]
calls = message.get("tool_calls")
if not isinstance(calls, list) or not calls:
    raise SystemExit(f"tool_calls missing: {body}")
for call in calls:
    if call.get("type") != "function":
        raise SystemExit(f"bad tool call type: {call}")
    fn = call.get("function") or {}
    if not fn.get("name"):
        raise SystemExit(f"tool call missing function name: {call}")
    json.loads(fn.get("arguments") or "{}")
PY
}

cleanup_pid=""
cleanup() {
  if [[ -n "$cleanup_pid" ]] && kill -0 "$cleanup_pid" >/dev/null 2>&1; then
    kill "$cleanup_pid" >/dev/null 2>&1 || true
    wait "$cleanup_pid" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

require_cmd cargo
require_cmd curl
require_cmd python3
mkdir -p "$MODEL_ROOT" "$ARTIFACT_DIR"

log "artifacts: $ARTIFACT_DIR"

if ((run_build)); then
  log "building hi-mlx"
  cargo build -p hi-mlx
fi

if ((run_unit)); then
  log "running native hi-mlx tests"
  cargo test -p hi-mlx
fi

failures=0
for idx in "${!REPOS[@]}"; do
  repo="${REPOS[$idx]}"
  safe="$(safe_path "$repo")"
  model_dir="$MODEL_ROOT/$safe"
  port=$((PORT_BASE + idx))
  base_url="http://$HOST:$port"
  out="$ARTIFACT_DIR/$safe"
  mkdir -p "$out"

  log "repo: $repo"
  log "model dir: $model_dir"

  if [[ ! -f "$model_dir/config.json" ]]; then
    if ((download_missing)); then
      log "downloading $repo"
      if ! download_repo "$repo" "$model_dir" 2>&1 | tee "$out.download.log"; then
        echo "download failed: $repo" >&2
        failures=$((failures + 1))
        continue
      fi
    else
      echo "missing $model_dir/config.json" >&2
      failures=$((failures + 1))
      continue
    fi
  fi

  log "inspect"
  if ! "$BIN" inspect "$model_dir" --model-id "$repo" >"$out/inspect.json" 2>"$out/inspect.err"; then
    cat "$out/inspect.err" >&2
    failures=$((failures + 1))
    continue
  fi

  log "serve on $base_url"
  cleanup
  cleanup_pid=""
  "$BIN" serve "$model_dir" --host "$HOST" --port "$port" --model-id "$repo" \
    >"$out/serve.log" 2>&1 &
  cleanup_pid="$!"

  if ! wait_for_health "$base_url" "$out"; then
    tail -200 "$out/serve.log" >&2 || true
    failures=$((failures + 1))
    cleanup
    cleanup_pid=""
    continue
  fi

  log "models"
  if ! curl -fsS "$base_url/v1/models" >"$out/models.json"; then
    failures=$((failures + 1))
    cleanup
    cleanup_pid=""
    continue
  fi

  log "chat non-streaming"
  nonstream_payload="$(python3 - "$repo" "$MAX_TOKENS" <<'PY'
import json, sys
print(json.dumps({
    "model": sys.argv[1],
    "messages": [{"role": "user", "content": "Say hello in one short sentence."}],
    "max_tokens": int(sys.argv[2]),
    "temperature": 0,
}))
PY
)"
  if ! post_json "$base_url/v1/chat/completions" "$nonstream_payload" "$out/chat.json" ||
    ! validate_nonstream "$out/chat.json"; then
    echo "non-streaming chat failed: $repo" >&2
    failures=$((failures + 1))
    cleanup
    cleanup_pid=""
    continue
  fi

  log "chat streaming"
  stream_payload="$(python3 - "$repo" "$MAX_TOKENS" <<'PY'
import json, sys
print(json.dumps({
    "model": sys.argv[1],
    "stream": True,
    "messages": [{"role": "user", "content": "Say hello in one short sentence."}],
    "max_tokens": int(sys.argv[2]),
    "temperature": 0,
}))
PY
)"
  if ! curl -fsS -N "$base_url/v1/chat/completions" \
    -H 'content-type: application/json' \
    -d "$stream_payload" \
    >"$out/stream.sse" ||
    ! validate_stream "$out/stream.sse"; then
    echo "streaming chat failed: $repo" >&2
    failures=$((failures + 1))
    cleanup
    cleanup_pid=""
    continue
  fi

  if ((run_tool)); then
    log "tool call"
    tool_payload="$(python3 - "$repo" "$MAX_TOKENS" <<'PY'
import json, sys
print(json.dumps({
    "model": sys.argv[1],
    "messages": [{
        "role": "user",
        "content": "Use the get_weather tool for Paris. Return only the tool call."
    }],
    "tools": [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"}
                },
                "required": ["city"]
            }
        }
    }],
    "tool_choice": {
        "type": "function",
        "function": {"name": "get_weather"}
    },
    "max_tokens": int(sys.argv[2]),
    "temperature": 0,
}))
PY
)"
    if ! post_json "$base_url/v1/chat/completions" "$tool_payload" "$out/tool.json" ||
      ! validate_tool_call "$out/tool.json"; then
      echo "tool-call chat failed: $repo" >&2
      failures=$((failures + 1))
      cleanup
      cleanup_pid=""
      continue
    fi
  fi

  log "ok: $repo"
  cleanup
  cleanup_pid=""
done

if ((failures)); then
  log "FAILED: $failures repo(s). Artifacts: $ARTIFACT_DIR"
  exit 1
fi

log "PASS: all repos. Artifacts: $ARTIFACT_DIR"
