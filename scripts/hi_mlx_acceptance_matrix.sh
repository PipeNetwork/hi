#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_ROOT="${HI_MLX_MODELS_DIR:-$ROOT/.hi/models}"
HOST="${HI_MLX_HOST:-127.0.0.1}"
PORT_BASE="${HI_MLX_PORT_BASE:-18080}"
MAX_TOKENS="${HI_MLX_MAX_TOKENS:-64}"
TOOL_MAX_TOKENS="${HI_MLX_TOOL_MAX_TOKENS:-256}"
HEALTH_TIMEOUT="${HI_MLX_HEALTH_TIMEOUT:-900}"
ARTIFACT_ROOT="${HI_MLX_ACCEPTANCE_ARTIFACTS:-$ROOT/target/hi-mlx-acceptance}"
RUN_ID="$(date +%Y%m%d-%H%M%S)"
ARTIFACT_DIR="$ARTIFACT_ROOT/$RUN_ID"
BIN="${HI_MLX_BIN:-$ROOT/target/debug/hi-mlx}"
HI_BIN="${HI_BIN:-$ROOT/target/debug/hi}"
SKIP_OVERSIZE="${HI_MLX_SKIP_OVERSIZE:-1}"
MEMORY_LIMIT_FRACTION="${HI_MLX_MEMORY_LIMIT_FRACTION:-0.85}"

# Default matrix. Small, runnable models across the supported arch families, plus newer variants
# that probe the family-detection edges. Override by passing repos as args.
#
# Verified 2026-07 (family detection is a loose substring match, so "detected" != "runs").
# One coherent generator per supported arch family; override by passing repos as args.
#   works : qwen2, qwen3, qwen3_moe (128-expert MoE, e.g. Qwen3-30B-A3B), qwen3_5 (SSM/gated-delta
#           hybrid dense, e.g. Qwen3.5-27B, Qwen35Like), qwen3_5_moe (SSM hybrid + shared-expert MoE),
#           glm4 (GQA), glm4_moe_lite (MLA), glm_moe_dsa (GLM-5.2), deepseek_v2/v3/v4 (MLA)
REPOS=(
  # --- Qwen / GLM core ---
  "mlx-community/Qwen3-0.6B-4bit"                                          # qwen3
  "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit"                           # qwen2
  "mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit"                         # qwen3_moe    (128-expert MoE; the popular 30B-A3B)
  "Jackrong/MLX-Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled-v2-4bit"    # qwen3_5      (SSM/gated-delta hybrid, dense; Qwen3.5-27B is the same arch)
  "Jackrong/MLX-Qwen3.5-35B-A3B-Claude-4.6-Opus-Reasoning-Distilled-4bit"  # qwen3_5_moe  (SSM hybrid + shared-expert MoE)
  "mlx-community/GLM-4-9B-0414-4bit"                                       # glm4         (GQA, Glm4Like)
  "mlx-community/GLM-4.7-Flash-4bit"                                       # glm4_moe_lite (MLA)
  # --- dense Llama-likes on QwenLike (config-gated) ---
  "mlx-community/granite-3.3-2b-instruct-4bit"                             # granite      (scalar multipliers)
  "mlx-community/SmolLM3-3B-4bit"                                          # smollm3      (per-layer NoPE)
  "mlx-community/exaone-4.0-1.2b-bf16"                                     # exaone4      (post-norm + per-head qk-norm)
  "mlx-community/OLMo-2-1124-7B-Instruct-4bit"                            # olmo2        (post-norm + full qk-norm)
  "mlx-community/Seed-OSS-36B-Instruct-4bit"                               # seed_oss     (drop-in SwiGLU; 19GB)
  # --- dedicated impls (GQA/MoE variants) ---
  "mlx-community/Nemotron-Mini-4B-Instruct-4bit-mlx"                       # nemotron     (LayerNorm1P + squared-ReLU + partial rope)
  "lmstudio-community/gpt-oss-20b-MLX-8bit"                                # gpt_oss      (attention sinks + biased top-k MoE + SwiGLU-OAI; 21GB)
  "mlx-community/gemma-3-1b-it-4bit"                                       # gemma3       (Gemma4TextLike: 1+weight norm, qpa scale, full rope)
  "mlx-community/gemma-2-2b-it-4bit"                                       # gemma2       (Gemma4TextLike minus qk-norm)
  "pipenetwork/Gemma-4-31B-it-MLX-4bit"                                    # gemma4       (sliding/full hybrid, dual RoPE, GeGLU; 16GB)
  "mlx-community/c4ai-command-r7b-12-2024-4bit"                            # cohere2      (LayerNorm + parallel block + NoPE-on-full + logit_scale)
  "lmstudio-community/Llama-4-Scout-17B-16E-MLX-text-4bit"                 # llama4       (iRoPE + L2 qk-norm + llama3 rope + top-1 MoE + shared expert; 57GB)
  # --- MoE variants on QwenLike / dedicated ---
  "mlx-community/OLMoE-1B-7B-0125-Instruct-4bit"                           # olmoe        (Qwen3-MoE + full qk-norm; individual experts auto-stacked)
  "mlx-community/ERNIE-4.5-21B-A3B-PT-4bit"                                # ernie4_5_moe (softmax-topk MoE + shared_experts; 11GB) — NOTE: mlx repo omits tokenizer.json; grab from base baidu/ERNIE-4.5-21B-A3B-PT
  "mlx-community/Phi-3.5-MoE-instruct-4bit"                                # phimoe       (SuScaledRoPE/LongRoPE + LayerNorm + top-2 MoE; 22GB)
  "mlx-community/dots.llm1.inst-mixed-4-6bit"                              # dots1        (per-head qk-norm + DeepSeek aux-free MoE; mixed 4-6bit per-tensor quant; 80GB)
  # --- Nemotron-H (Mamba2 hybrid) + PipeNetwork large MoEs ---
  "pipenetwork/NVIDIA-Nemotron-3-Nano-4B-MLX-8bit"                         # nemotron_h   (Mamba2 + attention + MLP hybrid, dense)
  "pipenetwork/Hy3-REAP50-MLX-4bit"                                        # hy_v3        (Hunyuan-3 REAP50; QwenLike + MoE) — 85GB
  "pipenetwork/MiniMax-M3-MLX-3bit"                                       # minimax_m3   (GQA + SwiGLU-OAI sigmoid-MoE, (1+weight) norm) — 174GB, HI_MLX_MAX_TOKENS=12
  "pipenetwork/LongCat-2.0-REAP75-MLX-4bit"                                 # longcat2     (ScMoE + absorbed-MLA + n-gram embed + YARN) — 282GB, HI_MLX_MAX_TOKENS=12
  "avlp12/GLM-5.2-Alis-MLX-Dynamic-3.5bpw"                                 # glm_moe_dsa  (DeepSeek-V3.2-style: MLA + DSA indexer + MoE) — 310GB, HI_MLX_MAX_TOKENS=8
  # Blocked (not run by the matrix):
  # - kimi_k25: Kimi-K2.7-Code — tiktoken tokenizer, no tokenizer.json (arch-verified on MlaLike)
  # - internlm3: mlx-community/internlm3-8b-instruct-4bit — ships only tokenizer.model, no tokenizer.json
  # - granitemoe: no MLX model published in any quant
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
  HI_MLX_TOOL_MAX_TOKENS         Default: 256
  HI_MLX_HEALTH_TIMEOUT          Default: 900 seconds
  HI_MLX_ACCEPTANCE_ARTIFACTS    Default: target/hi-mlx-acceptance
  HI_MLX_SKIP_OVERSIZE           Default: 1; skip repos above the safe MLX memory budget
  HI_MLX_MEMORY_LIMIT_BYTES      Optional explicit safe memory budget
  HI_MLX_MEMORY_LIMIT_FRACTION   Default: 0.85 of hw.memsize when bytes is unset
  HI_BIN                         Default: target/debug/hi

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
  if [[ -x "$HI_BIN" ]]; then
    "$HI_BIN" hf download "$repo" --keep "$dir"
  else
    cargo run -p hi -- hf download "$repo" --keep "$dir"
  fi
}

wait_for_health() {
  local base_url="$1"
  local out="$2"
  local pid="$3"
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
    if ! kill -0 "$pid" >/dev/null 2>&1; then
      echo "hi-mlx exited before becoming healthy at $base_url" >&2
      return 1
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

oversize_skip_reason() {
  local inspect_json="$1"
  local host_bytes=""
  local fraction="$MEMORY_LIMIT_FRACTION"
  if [[ -n "${HI_MLX_MEMORY_LIMIT_BYTES:-}" ]]; then
    host_bytes="$HI_MLX_MEMORY_LIMIT_BYTES"
    fraction="1.0"
  elif command -v sysctl >/dev/null 2>&1; then
    host_bytes="$(sysctl -n hw.memsize 2>/dev/null || true)"
  fi
  if [[ -z "$host_bytes" ]]; then
    return 1
  fi
  python3 - "$inspect_json" "$host_bytes" "$fraction" <<'PY'
import json, math, sys
path, host_raw, fraction_raw = sys.argv[1:]
try:
    host = int(host_raw)
    fraction = float(fraction_raw)
except ValueError:
    raise SystemExit(1)
if host <= 0 or not math.isfinite(fraction) or fraction <= 0:
    raise SystemExit(1)
with open(path, "r", encoding="utf-8") as f:
    info = json.load(f)
estimate = sum(int(shard.get("bytes") or 0) for shard in info.get("weight_shards") or [])
limit = int(host * min(fraction, 1.0))
if estimate <= limit:
    raise SystemExit(1)
gib = 1024 ** 3
print(
    f"requires {estimate / gib:.2f} GiB of shards; safe MLX budget is {limit / gib:.2f} GiB"
)
PY
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
from collections import Counter
with open(sys.argv[1], "r", encoding="utf-8") as f:
    body = json.load(f)
msg = body["choices"][0]["message"]
# Reasoning models put their answer in `reasoning` until they finish thinking, so accept either.
text = ((msg.get("content") or "") + " " + (msg.get("reasoning") or "")).strip()
if not text:
    raise SystemExit(f"assistant produced no text: {body}")

# Coherence gate: catch degenerate output (repeated tokens/chars) that a broken arch or bad chat
# template produces but a structural "200 OK / non-empty" check would happily pass.
compact = "".join(text.split())
if len(compact) >= 8 and len(set(compact)) <= 2:
    raise SystemExit(f"degenerate output (<=2 distinct chars): {text[:120]!r}")
if compact and Counter(compact).most_common(1)[0][1] / len(compact) > 0.6:
    raise SystemExit(f"degenerate output (one char >60%): {text[:120]!r}")
words = text.split()
if len(words) >= 6 and len(set(words)) <= 2:
    raise SystemExit(f"degenerate output (<=2 distinct words): {text[:120]!r}")
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
  log "building hi and hi-mlx"
  cargo build -p hi -p hi-mlx
fi

if ((run_unit)); then
  log "running native hi-mlx tests"
  cargo test -p hi-mlx
fi

failures=0
skipped=0
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
  model_type="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("model_type",""))' "$out/inspect.json" 2>/dev/null || true)"
  if [[ "$SKIP_OVERSIZE" != "0" ]]; then
    skip_reason="$(oversize_skip_reason "$out/inspect.json" || true)"
    if [[ -n "$skip_reason" ]]; then
      log "skip: $repo ($skip_reason)"
      skipped=$((skipped + 1))
      continue
    fi
  fi

  log "serve on $base_url"
  cleanup
  cleanup_pid=""
  "$BIN" serve "$model_dir" --host "$HOST" --port "$port" --model-id "$repo" \
    >"$out/serve.log" 2>&1 &
  cleanup_pid="$!"

  if ! wait_for_health "$base_url" "$out" "$cleanup_pid"; then
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

  # The tool-call smoke check runs for archs hi-mlx renders + parses tool calls for AND whose matrix test
  # model reliably emits one: the Qwen/GLM hermes path, the chatml-routed expansion archs (olmoe/ernie/
  # dots1), and native-template archs whose tool instruction is folded into a system turn (cohere2/phimoe).
  # The wiring also covers granite/exaone4 (verified on direct prompts) but their 1-2B matrix models are
  # too small to pass reliably at temp 0, so they're skipped here. Also skipped: gemma (small/format),
  # smollm3 (tiny), seed_oss (reasons first), llama4 (Scout-4bit mangles the JSON on some prompts), the model emits an unparsed/unreliable tool format.
  if ((run_tool)) && [[ " qwen2 qwen2_moe qwen3 qwen3_moe qwen3_5 qwen3_5_moe hy_v3 glm4 glm4_moe_lite glm_moe_dsa deepseek_v2 deepseek_v3 deepseek_v4 cohere2 phimoe olmoe ernie4_5_moe dots1 olmo2 gpt_oss " == *" $model_type "* ]]; then
    log "tool call"
    tool_payload="$(python3 - "$repo" "$TOOL_MAX_TOKENS" <<'PY'
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
  elif ((run_tool)); then
    log "tool call: skipped ($model_type is not on the hermes-style tool-call path)"
  fi

  log "ok: $repo"
  cleanup
  cleanup_pid=""
done

if ((failures)); then
  log "FAILED: $failures repo(s). Artifacts: $ARTIFACT_DIR"
  exit 1
fi

if ((skipped)); then
  log "PASS: runnable repos passed; skipped $skipped oversize repo(s). Artifacts: $ARTIFACT_DIR"
else
  log "PASS: all repos. Artifacts: $ARTIFACT_DIR"
fi
