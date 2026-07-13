#!/usr/bin/env bash
set -Eeuo pipefail

# Native CUDA GGUF acceptance matrix for hi-local — the GGUF counterpart of
# scripts/hi_mlx_acceptance_matrix.sh. For each model it downloads (if missing),
# inspects, serves on the CUDA GPU backend, and asserts:
#   /health is ready AND running on the GPU (execution=gpu, backend=cuda) with the
#   expected arch family, /v1/models lists it, a non-streaming chat is coherent
#   (a degenerate-output gate catches a broken arch that a "200 OK" check misses),
#   a streaming chat emits content + [DONE], and (unless skipped) a long-context
#   retrieval prompt is answered — which catches per-layer attention bugs
#   (sliding-window / dual-RoPE) that only surface past the local window.
#
# The models are small enough to run one-at-a-time on an ~8 GB card; each is
# served on its own port and stopped before the next.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_ROOT="${HI_GGUF_MODELS_DIR:-$HOME/.hi/models}"
MATRIX_SUBDIR="${HI_GGUF_MATRIX_SUBDIR:-gguf-matrix}"
HOST="${HI_GGUF_HOST:-127.0.0.1}"
PORT_BASE="${HI_GGUF_PORT_BASE:-18090}"
MAX_TOKENS="${HI_GGUF_MAX_TOKENS:-64}"
TOOL_MAX_TOKENS="${HI_GGUF_TOOL_MAX_TOKENS:-96}"
MAX_BATCHED_TOKENS="${HI_GGUF_MAX_BATCHED_TOKENS:-4096}"
HEALTH_TIMEOUT="${HI_GGUF_HEALTH_TIMEOUT:-300}"
ARTIFACT_ROOT="${HI_GGUF_ACCEPTANCE_ARTIFACTS:-$ROOT/target/hi-gguf-acceptance}"
RUN_STAMP="${HI_GGUF_RUN_STAMP:-$(date +%Y%m%d-%H%M%S)}"
ARTIFACT_DIR="$ARTIFACT_ROOT/$RUN_STAMP"
BIN="${HI_GGUF_BIN:-$ROOT/target/release/hi-local}"

# Prepend a CUDA toolkit bin if nvcc is not already on PATH (the native-cuda
# build needs it); harmless if the directory does not exist.
if ! command -v nvcc >/dev/null 2>&1; then
  for cuda_bin in /usr/local/cuda/bin /opt/cuda/bin; do
    [[ -x "$cuda_bin/nvcc" ]] && export PATH="$cuda_bin:$PATH" && break
  done
fi

# Default matrix: one small, runnable model per supported dense arch family, chosen
# to exercise the CUDA loader/forward-pass edges. Fields are pipe-delimited:
#   id | expected_family | quant | long_ctx | url
# `expected_family` is the /health "family" string (gemma2 and gemma3 both report
# "gemma"). `long_ctx` (1/0) opts a model into the strict long-context *retrieval*
# probe — set 0 for models too weak to retrieve reliably (tiny/degenerate) or that
# emit long <think> preambles, since a retrieval miss there reflects the model, not
# the runtime; those models are still coherence-gated at short context. Override the
# whole matrix by passing `id|family|quant|long_ctx|url` tuples as args.
MODELS=(
  "qwen2.5-0.5b|qwen2|Q4_K_M|1|https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf"  # qwen2 (GPT2-BPE tokenizer)
  "tinyllama-1.1b|llama|Q4_K_M|0|https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"  # llama (SPM tokenizer); 1.1B too weak for retrieval
  "llama-3.2-1b|llama|Q4_K_M|0|https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf"  # llama3 (tiktoken-style BPE + llama3 chat template); 1B too weak for retrieval
  "phi-3-mini|phi|IQ4_NL|1|https://huggingface.co/SixOpen/Phi-3-mini-4k-instruct-IQ4_NL-imat.gguf/resolve/main/phi-3-mini-4k-instruct-iq4_nl-imat.gguf"  # phi3 (fused ffn_up gate+up)
  "qwen3-0.6b|qwen3|Q8_0|0|https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q8_0.gguf"  # qwen3 (QK-norm); thinking mode overruns the short retrieval budget
  "gemma-2-2b|gemma|Q4_K_M|1|https://huggingface.co/bartowski/gemma-2-2b-it-GGUF/resolve/main/gemma-2-2b-it-Q4_K_M.gguf"  # gemma2 (post-norms, GeGLU, softcap)
  "gemma-3-1b|gemma|Q4_K_M|1|https://huggingface.co/unsloth/gemma-3-1b-it-GGUF/resolve/main/gemma-3-1b-it-Q4_K_M.gguf"  # gemma3 (per-layer sliding-window + dual RoPE) — long_ctx probe exercises the fix
  "mistral-7b|llama|Q4_K_M|1|https://huggingface.co/bartowski/Mistral-7B-Instruct-v0.3-GGUF/resolve/main/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf"  # Mistral GGUFs carry the llama arch, so /health reports family "llama"
)

# Larger models (need more than an ~8 GB card) opt in via --large. These cover
# real MoE expert routing that no small dense model exercises.
LARGE_MODELS=(
  "qwen3-30b-a3b|qwen3|Q4_K_M|1|https://huggingface.co/unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF/resolve/main/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf"  # qwen3moe (128-expert MoE, 8 active)
  "mixtral-8x7b|llama|Q4_K_M|0|https://huggingface.co/MaziyarPanahi/Mixtral-8x7B-Instruct-v0.1-GGUF/resolve/main/Mixtral-8x7B-Instruct-v0.1.Q4_K_M.gguf"  # llama-arch MoE (rank-3 ffn_*_exps tensors); long-ctx retrieval currently degenerate — coherence-gated at short context only
  "deepseek-v2-lite|deepseek|Q4_K_M|0|https://huggingface.co/gaianet/DeepSeek-V2-Lite-Chat-GGUF/resolve/main/DeepSeek-V2-Lite-Chat-Q4_K_M.gguf"  # deepseek2 full-Q MLA (kv latent 512 + rope 64, asymmetric qk 192 / v 128) + 64-expert MoE with fused shared experts; massive activations force the f32/f16-activation matmul paths (no int8 dp4a). long_ctx 0: plain-rope CPU-parity config, YARN >4k unverified
  "glm-5.2-reap50|glm-flash|Q3_K_M|0|https://huggingface.co/pipenetwork/GLM-5.2-REAP50-Q3_K_M-GGUF/resolve/main/GLM-5.2-REAP50-Q3_K_M-00001-of-00005.gguf"  # glm_moe_dsa (394B REAP-pruned MoE, 128 experts/layer, MLA q+kv-LoRA, interleaved pe-rope, sigmoid noaux_tc router); 169 GB across 5 shards (split loading pulls siblings); serve with HI_CUDA_EXPERT_STREAMING=1 and a pool sized to the card
)

usage() {
  cat <<'EOF'
Usage: scripts/hi_gguf_acceptance_matrix.sh [options] [id|family|quant|long_ctx|url ...]

Runs the native CUDA GGUF acceptance smoke matrix for hi-local:
  download (if missing), inspect, serve on the CUDA GPU, /health (GPU + family),
  /v1/models, non-streaming chat (coherence-gated), streaming chat, an optional
  long-context retrieval probe, and an optional tool-call prompt.

Options:
  --no-download       Require the .gguf files to already exist.
  --skip-build        Do not build hi-local (--features native-cuda).
  --skip-unit         Do not run cargo test -p hi-gguf before the matrix.
  --skip-long-context Skip the long-context retrieval probe.
  --large             Also run the LARGE_MODELS tier (MoE models needing more
                      than an ~8 GB card: qwen3-30b-a3b, mixtral-8x7b).
  --tool              Also run a (forced) tool-call check (off by default; small
                      GGUF models are unreliable at tool calling).
  -h, --help          Show this help.

Environment:
  HI_GGUF_MODELS_DIR            Default: $HOME/.hi/models
  HI_GGUF_MATRIX_SUBDIR         Default: gguf-matrix (under the models dir)
  HI_GGUF_BIN                   Default: target/release/hi-local
  HI_GGUF_PORT_BASE             Default: 18090
  HI_GGUF_MAX_TOKENS            Default: 64
  HI_GGUF_MAX_BATCHED_TOKENS    Default: 4096 (KV-cache token budget per serve)
  HI_GGUF_HEALTH_TIMEOUT        Default: 300 seconds
  HI_GGUF_ACCEPTANCE_ARTIFACTS  Default: target/hi-gguf-acceptance

Examples:
  scripts/hi_gguf_acceptance_matrix.sh
  scripts/hi_gguf_acceptance_matrix.sh --skip-build --skip-unit
  scripts/hi_gguf_acceptance_matrix.sh --no-download \
    "gemma-3-1b|gemma|Q4_K_M|1|file:///models/gemma-3-1b-it.gguf"
EOF
}

download_missing=1
run_build=1
run_unit=1
run_long=1
run_tool=0
run_large=0
selected=()

while (($#)); do
  case "$1" in
    --no-download) download_missing=0 ;;
    --skip-build) run_build=0 ;;
    --skip-unit) run_unit=0 ;;
    --skip-long-context) run_long=0 ;;
    --large) run_large=1 ;;
    --tool) run_tool=1 ;;
    -h | --help) usage; exit 0 ;;
    -*) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
    *) selected+=("$1") ;;
  esac
  shift
done

if ((${#selected[@]})); then
  MODELS=("${selected[@]}")
fi
if ((run_large)); then
  MODELS+=("${LARGE_MODELS[@]}")
fi

log() { printf '\n[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "missing required command: $1" >&2
    exit 1
  fi
}

download_gguf() {
  local url="$1" dest="$2"
  mkdir -p "$(dirname "$dest")"
  case "$url" in
    file://*) cp "${url#file://}" "$dest" ;;
    *) curl -fL --retry 3 --retry-delay 2 -o "$dest.part" "$url" && mv "$dest.part" "$dest" ;;
  esac
}

# Wait for /health to report ready, then assert it is serving on the GPU with the
# expected arch family. Writes the final health body to "$out/health.json".
wait_for_health() {
  local base_url="$1" out="$2" pid="$3" want_family="$4"
  local deadline=$((SECONDS + HEALTH_TIMEOUT))
  local last="$out/health.last.json"
  while ((SECONDS < deadline)); do
    if curl -fsS "$base_url/health" >"$last" 2>"$out/health.err"; then
      local rc=0
      # exit 0 = ready & on GPU with the right family; 1 = not ready yet (keep
      # polling); 2 = ready but wrong execution/backend/family (hard fail).
      python3 - "$last" "$want_family" <<'PY' || rc=$?
import json, sys
body = json.load(open(sys.argv[1], encoding="utf-8"))
want = sys.argv[2]
if body.get("ready") is not True:
    sys.exit(1)
exe = (body.get("execution") or {}).get("status")
backend = body.get("backend")
family = body.get("family")
problems = []
if exe != "gpu":
    problems.append(f"execution={exe!r} (want gpu)")
if backend != "cuda":
    problems.append(f"backend={backend!r} (want cuda)")
if family != want:
    problems.append(f"family={family!r} (want {want!r})")
if problems:
    print("health assertion failed: " + "; ".join(problems), file=sys.stderr)
    sys.exit(2)
PY
      if [[ $rc -eq 0 ]]; then
        cp "$last" "$out/health.json"
        return 0
      elif [[ $rc -eq 2 ]]; then
        return 1
      fi
    fi
    if ! kill -0 "$pid" >/dev/null 2>&1; then
      echo "hi-local exited before becoming healthy at $base_url" >&2
      return 1
    fi
    sleep 2
  done
  echo "timed out waiting for healthy hi-local at $base_url" >&2
  [[ -s "$last" ]] && { cat "$last" >&2; echo >&2; }
  return 1
}

post_json() {
  curl -fsS "$1" -H 'content-type: application/json' -d "$2" >"$3"
}

# Coherence gate: reject degenerate output (repeated char/word) that a broken arch
# or bad chat template produces but a "200 OK / non-empty" check would pass. Shared
# with the MLX matrix's validator.
validate_nonstream() {
  python3 - "$1" <<'PY'
import json, sys
from collections import Counter
body = json.load(open(sys.argv[1], encoding="utf-8"))
msg = body["choices"][0]["message"]
text = ((msg.get("content") or "") + " " + (msg.get("reasoning") or "")).strip()
if not text:
    raise SystemExit(f"assistant produced no text: {body}")
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
  python3 - "$1" <<'PY'
import json, sys
done = False
content = []
for raw in open(sys.argv[1], encoding="utf-8"):
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

# Long-context probe: bury a distinctive token near the start, pad past the local
# sliding window, then ask the model to recall it. Catches per-layer attention
# bugs (e.g. Gemma-3 dual-RoPE / sliding-window) that only appear past ~512 tokens
# while short prompts stay coherent.
validate_retrieval() {
  python3 - "$1" <<'PY'
import json, sys
body = json.load(open(sys.argv[1], encoding="utf-8"))
msg = body["choices"][0]["message"]
text = ((msg.get("content") or "") + " " + (msg.get("reasoning") or "")).upper()
if "MAGENTA" not in text and "73" not in text:
    raise SystemExit(f"did not recall the code across the context window: {text[:160]!r}")
PY
}

validate_tool_call() {
  python3 - "$1" <<'PY'
import json, sys
body = json.load(open(sys.argv[1], encoding="utf-8"))
calls = body["choices"][0]["message"].get("tool_calls")
if not isinstance(calls, list) or not calls:
    raise SystemExit(f"tool_calls missing: {body}")
for call in calls:
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

require_cmd curl
require_cmd python3
mkdir -p "$MODEL_ROOT/$MATRIX_SUBDIR" "$ARTIFACT_DIR"
log "artifacts: $ARTIFACT_DIR"

if ((run_build)); then
  require_cmd cargo
  log "building hi-local (--features native-cuda)"
  cargo build --release -p hi-local --features native-cuda
fi

if [[ ! -x "$BIN" ]]; then
  echo "hi-local binary not found at $BIN (build it, or set HI_GGUF_BIN / pass --skip-build after building)" >&2
  exit 1
fi

if ((run_unit)); then
  require_cmd cargo
  log "running hi-gguf unit tests"
  cargo test -p hi-gguf
fi

failures=0
passed=0
for idx in "${!MODELS[@]}"; do
  IFS='|' read -r id family quant long_ctx url <<<"${MODELS[$idx]}"
  gguf="$MODEL_ROOT/$MATRIX_SUBDIR/$id.gguf"
  port=$((PORT_BASE + idx))
  base_url="http://$HOST:$port"
  out="$ARTIFACT_DIR/$id"
  mkdir -p "$out"

  log "=== $id ($family, $quant) ==="

  if [[ ! -f "$gguf" ]]; then
    if ((download_missing)); then
      log "downloading $id"
      if ! download_gguf "$url" "$gguf" 2>&1 | tee "$out/download.log"; then
        echo "download failed: $id" >&2
        failures=$((failures + 1)); continue
      fi
    else
      echo "missing $gguf (run without --no-download to fetch it)" >&2
      failures=$((failures + 1)); continue
    fi
  fi

  log "inspect"
  if ! "$BIN" inspect "$gguf" >"$out/inspect.json" 2>"$out/inspect.err"; then
    cat "$out/inspect.err" >&2
    failures=$((failures + 1)); continue
  fi

  log "serve on $base_url"
  cleanup; cleanup_pid=""
  "$BIN" serve "$gguf" --backend cuda --execution gpu --host "$HOST" --port "$port" \
    --model-id "$id" --max-batch-size 1 --max-batched-tokens "$MAX_BATCHED_TOKENS" \
    >"$out/serve.log" 2>&1 &
  cleanup_pid="$!"

  if ! wait_for_health "$base_url" "$out" "$cleanup_pid" "$family"; then
    tail -60 "$out/serve.log" >&2 || true
    failures=$((failures + 1)); cleanup; cleanup_pid=""; continue
  fi

  log "models"
  if ! curl -fsS "$base_url/v1/models" >"$out/models.json"; then
    failures=$((failures + 1)); cleanup; cleanup_pid=""; continue
  fi

  log "chat non-streaming (coherence-gated)"
  nonstream="$(python3 - "$id" "$MAX_TOKENS" <<'PY'
import json, sys
print(json.dumps({"model": sys.argv[1],
  "messages": [{"role": "user", "content": "In one short sentence, what is the capital of France?"}],
  "max_tokens": int(sys.argv[2]), "temperature": 0}))
PY
)"
  if ! post_json "$base_url/v1/chat/completions" "$nonstream" "$out/chat.json" ||
    ! validate_nonstream "$out/chat.json"; then
    echo "non-streaming chat failed: $id" >&2
    failures=$((failures + 1)); cleanup; cleanup_pid=""; continue
  fi

  log "chat streaming"
  stream="$(python3 - "$id" "$MAX_TOKENS" <<'PY'
import json, sys
print(json.dumps({"model": sys.argv[1], "stream": True,
  "messages": [{"role": "user", "content": "Name three colors."}],
  "max_tokens": int(sys.argv[2]), "temperature": 0}))
PY
)"
  if ! curl -fsS -N "$base_url/v1/chat/completions" -H 'content-type: application/json' \
      -d "$stream" >"$out/stream.sse" || ! validate_stream "$out/stream.sse"; then
    echo "streaming chat failed: $id" >&2
    failures=$((failures + 1)); cleanup; cleanup_pid=""; continue
  fi

  if ((run_long)) && [[ "$long_ctx" == "1" ]]; then
    log "long-context retrieval"
    longctx="$(python3 - "$id" <<'PY'
import json, sys
filler = "This is unrelated filler context that should be ignored. " * 90  # ~800+ tokens
prompt = ("Remember this code: MAGENTA-73.\n" + filler +
          "\nWhat was the code I asked you to remember? Answer with just the code.")
print(json.dumps({"model": sys.argv[1],
  "messages": [{"role": "user", "content": prompt}],
  "max_tokens": 20, "temperature": 0}))
PY
)"
    if ! post_json "$base_url/v1/chat/completions" "$longctx" "$out/longctx.json" ||
      ! validate_retrieval "$out/longctx.json"; then
      echo "long-context retrieval failed: $id" >&2
      failures=$((failures + 1)); cleanup; cleanup_pid=""; continue
    fi
  fi

  if ((run_tool)); then
    log "tool call"
    tool="$(python3 - "$id" "$TOOL_MAX_TOKENS" <<'PY'
import json, sys
print(json.dumps({"model": sys.argv[1],
  "messages": [{"role": "user", "content": "Use the get_weather tool for Paris. Return only the tool call."}],
  "tools": [{"type": "function", "function": {"name": "get_weather",
    "description": "Get current weather for a city.",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}],
  "tool_choice": {"type": "function", "function": {"name": "get_weather"}},
  "max_tokens": int(sys.argv[2]), "temperature": 0}))
PY
)"
    if ! post_json "$base_url/v1/chat/completions" "$tool" "$out/tool.json" ||
      ! validate_tool_call "$out/tool.json"; then
      echo "tool-call chat failed: $id" >&2
      failures=$((failures + 1)); cleanup; cleanup_pid=""; continue
    fi
  fi

  log "ok: $id"
  passed=$((passed + 1))
  cleanup; cleanup_pid=""
done

log "matrix: $passed passed, $failures failed of ${#MODELS[@]}. Artifacts: $ARTIFACT_DIR"
((failures == 0))
