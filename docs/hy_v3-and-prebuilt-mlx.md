# hy_v3 (Hunyuan-3) support, prebuilt-MLX linking, and MoE performance notes

Working notes from adding Tencent **Hy3 / Hunyuan-3** (`model_type: hy_v3`,
`HYV3ForCausalLM`) support to `hi-mlx`, getting the Rust MLX backend to build and
generate on a Metal-4 machine, and an honest account of the MoE decode-speed
investigation. Three parts:

1. [Linking a prebuilt MLX (the build/`bfloat16_t` fix)](#1-linking-a-prebuilt-mlx)
2. [hy_v3 model support](#2-hy_v3-model-support)
3. [MoE decode performance — what worked, what didn't, and why](#3-moe-decode-performance)

## Which of this do you actually need?

**These are independent.** Part 1 is an *environment* fix; part 2 is a *model*
feature. Most people need neither.

| You are… | Part 1 (prebuilt MLX) | Part 2 (hy_v3) |
|---|---|---|
| on macOS ≤ 15 (older Metal Toolchain), running any model | **no** — the normal from-source build just works | no |
| on macOS 26 / Metal 4, running **Qwen / DeepSeek / GLM / etc.** | **yes** — otherwise the from-source MLX build fails at runtime with `bfloat16_t`, for *every* model | no |
| running **Hy3 / Hunyuan-3** (any OS) | only if you're also on Metal 4 | **yes** |

Key point: **part 1 is not Hy3-specific.** The `bfloat16_t` failure hits a stock
`Qwen3-0.6B-4bit` exactly the same way — it's about the machine's Metal
Toolchain, not the model. It only came up here because Hy3 was the first thing
tried on a fresh Metal-4 box.

Everything below is on Apple silicon; the Metal-Toolchain issue in part 1 is
specific to macOS 26 (Tahoe) / Metal 4, `hi-mlx` built with the `metal` feature.

---

## 1. Linking a prebuilt MLX

### Symptom
On this machine `hi-mlx` first wouldn't build at all:

```
error: cannot execute tool 'metal' due to missing Metal Toolchain;
use: xcodebuild -downloadComponent MetalToolchain
```

Fixed by installing the Metal Toolchain component (`xcodebuild
-downloadComponent MetalToolchain`, ~700 MB, no auth). After that it built, but
**generation** failed at runtime for *every* model (including a stock
`mlx-community/Qwen3-0.6B-4bit`):

```
[metal::Device] Unable to build metal library from source
mlx/backend/metal/kernels/utils.h:64: unknown type name 'bfloat16_t'
```

### Root cause
`pmetal-mlx-sys` builds MLX **from source** and its runtime Metal-kernel JIT is
incompatible with the very new Metal Toolchain (Metal 4). The **pip `mlx` wheel
works fine** on the same machine because it ships a complete precompiled
`mlx.metallib` and avoids runtime JIT. Bumping the MLX *version* does **not**
fix it — `mlx/backend/metal/kernels/utils.h` is byte-identical across 0.31.1 /
0.31.2, and JIT is already `OFF`. It's a build-provenance problem, not a version
problem.

### Fix — link the prebuilt MLX instead of building from source
Opt-in via `HI_MLX_SYSTEM_MLX_PREFIX` pointing at a prebuilt MLX install (e.g. a
pip `mlx` package dir that contains `lib/libmlx.dylib`, `lib/mlx.metallib`,
`include/`, and `share/cmake/MLX/MLXConfig.cmake`):

```bash
# one-time: install a matching prebuilt MLX (must match the vendored mlx-c, see below)
pip install --target /path/to/mlx-0311 "mlx==0.31.1"

HI_MLX_SYSTEM_MLX_PREFIX=/path/to/mlx-0311/mlx \
  cargo build --release -p hi-mlx
```

This required coordinated changes (all vendored + `[patch.crates-io]`'d from the
workspace `Cargo.toml`):

| Change | Where | Why |
|---|---|---|
| `HI_MLX_SYSTEM_MLX_PREFIX` → `MLX_C_USE_SYSTEM_MLX=ON` + `CMAKE_PREFIX_PATH`, link `dylib=mlx` (+ `-rpath`) instead of `static=mlx`; skip metallib caching | `vendor/pmetal-mlx-sys/build.rs` | use `find_package(MLX)` against the prebuilt install |
| Overlay **mlx-c 0.6.0** C bindings (`mlx/c/*.{cpp,h}`) over the vendored 0.5.0; MLX `GIT_TAG` → **v0.31.1** | `vendor/pmetal-mlx-sys/src/mlx-c/` | mlx-c 0.5.0's `fft`/`ops` signatures don't match MLX 0.31; 0.6.0 pairs with MLX 0.31.1 |
| Add the `global_scale` arg to `mlx_quantize` / `mlx_dequantize` calls | `vendor/pmetal-mlx-rs/src/ops/quantization.rs` | mlx-c 0.6.0 added that parameter |
| Same `global_scale` arg on hi-mlx's own `mlx_dequantize` FFI call | `crates/hi-mlx/src/models.rs` (`dequantize_mode`) | ditto |
| New build script emitting `-rpath` to `$PREFIX/lib` | `crates/hi-mlx/build.rs` | a `-sys` crate's `rustc-link-arg` can't add an rpath to the final binary |

**Version coupling is strict:** mlx-c 0.6.0 ↔ MLX 0.31.1. Using MLX 0.31.2
against mlx-c 0.6.0 fails to compile (`fft` signature changed again between
0.31.1 and 0.31.2). The prebuilt MLX you link at `HI_MLX_SYSTEM_MLX_PREFIX` must
be **0.31.1** to match.

### Result
Stock `Qwen3-0.6B-4bit` (and every other supported model) generates correctly —
the `bfloat16_t` wall is gone. This is a general fix, not hy_v3-specific.

---

## 2. hy_v3 model support

Hy3 is a 295 B / 21 B-active MoE: 80 layers, GQA (64 heads / 8 KV, `head_dim`
128 — note `head_dim ≠ hidden/heads`), `qk_norm`, sigmoid-router MoE with 192
experts top-8 + expert bias + an always-on shared expert, `first_k_dense_replace
= 1`, and a large RoPE base. It routes through hi-mlx's `QwenLike` path (GQA +
MoE); the deltas from Qwen are below.

### Family registration
- `crates/hi-local-core/src/model.rs` — `ModelFamily::Hy3`, `label() = "hy3"`,
  and `from_gguf_architecture` matches `hy_v3` / `hyv3` / `hunyuan`.
- `crates/hi-mlx/src/manifest.rs` and `crates/hi-mlx/src/config.rs` — **both**
  `detect_family`/`model_family` functions match `hy_v3` (inspect uses the
  manifest one, load uses the config one; missing either shows "unsupported
  model_type").
- `crates/hi-mlx/src/weights.rs` — `Hy3` shares the Qwen attention-tensor
  validation arm (`self_attn.q_proj…`).
- `crates/hi-mlx/src/models.rs` `load_model` — `ModelFamily::Hy3` dispatches to
  `QwenLike`.

### Config quirks (both were silent-wrong-output bugs)
- **`rope_theta` is nested** under `rope_parameters.rope_theta` (~11.16 M), not
  top-level. `config.rs` now falls back to `rope_parameters.rope_theta` →
  otherwise defaults to 1 M and RoPE is ~11× off → garbage.
- **`router_scaling_factor`** (Hy3's key, `2.826`) vs hi-mlx's
  `routed_scaling_factor`; `config.rs` reads either.
- `is_qwen_moe_layer` returns `layer_idx >= first_k_dense_replace` for Hy3
  (layer 0 dense, 1+ MoE).

### Weight-key remap — `remap_hy3_moe_weights` (`models.rs`)
Hy3's checkpoint names differ from what the Qwen MoE loader expects:

| Hy3 key | hi-mlx expects | handling |
|---|---|---|
| `mlp.router.gate.{weight,scales,biases}` (8-bit) | `mlp.gate.weight` (dense) | **dequantize** to dense bf16 (the router gate is a plain matmul; also its per-tensor 8-bit quant would be mismatched after rename) |
| `mlp.router.expert_bias` | `mlp.gate.e_score_correction_bias` | rename |
| `mlp.shared_mlp.*` | `mlp.shared_expert.*` (singular) | rename |
| `mlp.switch_mlp.*` (stacked) | `mlp.switch_mlp.*` | already matches (mlx-lm `convert` stacks experts) |

### Routing math — `QwenMoe` (`models.rs`)
Qwen's MoE is softmax + no bias; Hy3 is different. The router now matches the
`mlx_lm` `hy_v3` reference:
`scores = sigmoid(gate(x))` → **add `expert_bias` for top-k *selection* only** →
routed **weights use the bias-free sigmoid scores** → normalize (`route_norm`) →
`× router_scaling_factor (2.826)`. The **shared expert is always-on** (Hy3 has
no `shared_expert_gate`; Qwen gates it with a sigmoid).

### Chat template + stop tokens
Hy3 doesn't use ChatML and ships its template in a separate `chat_template.jinja`
(not in `tokenizer_config.json`), which hi-mlx's family-based prompt builder
doesn't read. Added `build_hy3_prompt` (`crates/hi-local-core/src/prompt.rs`)
emitting the Hunyuan tokens:

```
<｜hy_begin_of_sentence:opensource｜>{system}<｜hy_User:opensource｜>{user}<｜hy_Assistant:opensource｜>
```

EOS is `<｜hy_eos:opensource｜>` (id **120025**), already read from
`config.json`'s `eos_token_id` into `eos_token_ids`. With the correct template
the model reasons in `<think:opensource>…</think:opensource>` and stops cleanly.

### Verified
`hi-mlx inspect` reports `"family": "hy3"`; native generation is correct
(`2+2 → 4` with proper reasoning + clean stop; `hi` agent wrote a correct
`is_prime` via its edit tool). **Correctness is solid.** Speed is the caveat —
see part 3.

---

## 3. MoE decode performance

**Bottom line: native Hy3 decode in hi-mlx is correct but ~0.6 tok/s, and that
ceiling is architectural — not fixable at the MoE/router level.**

### What was tried, and the (repeatable) numbers
4 controlled identical runs sit dead flat at **0.59–0.60 tok/s**, and it's the
same number regardless of router implementation:

| MoE variant | tok/s |
|---|---|
| batched CPU-readback router (`forward_cpu`, the default) | ~0.6 |
| GPU on-device `argpartition` router | ~0.6 |
| compiled MoE, per-call `compile()` | 0.6 |
| compiled MoE, compile-cache kept alive | 0.6 |

Two real optimizations were kept:
- **Batched experts** via `mlx_gather_qmm` (`SwitchMlp::forward_batched`) — run
  all top-k experts in a few kernels instead of one `quantized_matmul` per
  (token, expert). This is the one repeatable win (~2× on the real agent task,
  900 s → 428 s), though absolute tok/s is noisy.
- Router selection on CPU after a single readback of the 192-element score
  vector — cheaper here than an on-device `argpartition` per layer.

### `mx.compile` findings (kept behind `HI_MLX_COMPILE_MOE=1`)
The compiled MoE (`moe_compiled` / `run_moe_compiled`) is **numerically
correct** and proves MLX's compiler handles `gather_qmm` + `argpartition` fused.
Two real discoveries, neither of which yielded a speedup:

1. **`compile()` erases its own cache on drop.** `CompiledState::drop` calls
   `mlx_detail_compile_erase(id)`, so a per-call `compile(f)` re-traces every
   call. Worked around by `mem::forget`-ing the `Compiled` so the TypeId-keyed
   MLX cache entry survives. (A clean cache-at-load is blocked by the API: the
   compiled type isn't `for<'a>`-general enough to box as
   `dyn CallMut<&'a [Array], …>`.)
2. **Fixing the cache didn't help** → re-compilation was never the bottleneck.

### Why it's capped at ~0.6 tok/s
hi-mlx runs the model **eagerly** — every op (matmul, sigmoid, argpartition,
gather, norm, …) is a separate Metal kernel launch, and the sampler reads logits
back to CPU per token. Across 80 layers that's thousands of tiny dispatches per
token; launch/sync overhead dominates. A separate finding underlines this:
adding **one extra `eval()` per layer** (of a tiny constant) collapsed
throughput ~13× — the number of per-layer syncs, not the ops, dominates.

Optimizing the MoE alone can't change the per-token eager structure, which is
why batched / GPU / compiled all land at the same ~0.6.

### The actual fix (not done — it's a rewrite)
Match `mlx_lm`'s architecture: **`mx.compile` the whole generation step** (fuse
attention + MoE + norms into few kernels), a **fixed-shape pre-allocated KV
cache** (so decode shapes are constant and don't re-JIT), and **on-GPU
sampling** (no per-token readback). That's a ground-up rewrite of hi-mlx's
generation loop, not a patch.

### Practical guidance
- For **fast interactive Hy3**, serve via **`mlx_lm.server`** and point
  `hi --provider openai --base-url …` at it — that *is* compiled `mlx_lm`, fast
  and correct, and it serves any of the published MLX builds.
- hi-mlx native generation is best treated as **correctness-complete** for
  hy_v3; the compiled path is validated groundwork for the eventual
  generation-loop rewrite.

---

## Appendix: Qwen3.5 (`qwen3_5`) gated-delta-net port — **DONE, generates coherently**

Qwen3.5 is a **Mamba/SSM hybrid VL** model (24 gated-delta-net layers + 8 gated-attention layers).
Fully ported to `hi-mlx` and validated against coherent output. `Qwen35Like` / `GatedDeltaNet` /
`Qwen35Attention` in `models.rs`; dispatched from the qwen3 family when `linear_num_value_heads` is
present.

### Gotchas that actually cost debugging time
- **Gated output norm order** (the one that produced garbage): Qwen3.5 uses `Qwen3NextRMSNormGated`,
  which norms the SSM output *first* then gates — `silu(z) * rms_norm(out, w)`, **not**
  `rms_norm(silu(z) * out, w)` (the mamba2 order). Wrong order = varied-but-incoherent tokens.
- **Gated attention**: the full-attn `q_proj` is *doubled* — it packs `[queries | gate]`
  (`n_heads × 2 × head_dim`); the attention output is `o_proj(out * sigmoid(gate))` (Qwen3-Next).
  So `n_heads = q_proj_out / (2·head_dim)`, and head counts must be **derived from the weight
  shapes**, not config (checkpoint has 32/… heads where config says 16; head_dim ≠ hidden/heads).
- **VL prefixes**: `language_model.` stripped + `visual.`/`mtp.` dropped in `load_arrays`; the
  catalog validators had to accept the `language_model.`-prefixed keys too.
- SSM runs in **f32** (state `[1,Hv,Dv,Dk]`), cast back to model dtype before `out_proj`.
- **Prefill uses a chunk-parallel scan** (`scan_chunked`, C=64); decode (S==1) uses the per-token
  recurrence (`scan_recurrent`, with a single-token fast path). Both update the same state and were
  verified to agree. The chunked path precomputes the intra-chunk WY/UT quantities batched over all
  chunks — with a Newton-Schulz unit-lower-triangular inverse (`(I+A)⁻¹`, exact in ⌈log₂C⌉ iters
  since A is strictly-lower nilpotent) — then a short sequential scan over chunks. ~2.2× faster
  prefill at 365 tokens, more at longer lengths. **Numerical gotcha**: never form `k/γ` (γ underflows
  → `inf`, then `inf·0` in the triangular mask → `NaN` → garbage on some inputs); instead use decay
  *ratios* `exp(lgₜ−lgⱼ)` from the finite cumulative log-decays, masked additively (`+(−1e9)`)
  *before* the `exp`.
- **Decode throughput** (9B-4bit): ~70 tok/s (Qwen3.5), ~82 tok/s (dense GLM-4-9B) — memory-bandwidth
  bound (reading the 4-bit weights every token), **not** SSM- or CPU-bound. Streamlining the SSM
  decode step changed nothing.

### Why the compiled decode loop is NOT worth building (measured, settled)
The obvious "make decode faster" idea is to `mx.compile` the whole per-token step (mlx_lm does this).
Before building it, measured the actual headroom by comparing against mlx_lm's compiled loop on the
**identical** model (GLM-4-9B-0414-4bit, same machine):

| loop | tok/s |
|---|---|
| hi-mlx (eager, per-token `forward`) | ~82 |
| mlx_lm (compiled) | ~85 |

**~3% gap.** Decode is memory-bandwidth bound — the GPU reads ~4.5 GB of weights per token; compiling
removes CPU/launch overhead that is already mostly hidden behind that read. The rewrite it would take
is large and cross-cutting and would actively fight the design:
- `Cache::update_dense` returns a **dynamic `[..total_len]` slice** that grows each token → a compiled
  `forward` would re-specialize (recompile) every token. Fixing it means masked attention over the
  **full fixed-capacity buffer** in all four attention impls (Qwen/MLA/GLM-4/Qwen3.5) — which *adds*
  wasted compute (attending over unfilled positions), partly cancelling the win.
- The mlx-rs `compile` API is finicky here (the MoE attempt hit `CompiledState::drop` erasing the
  cache; `mem::forget` only half-worked).

Conclusion: a multi-day, high-risk rewrite for ~3%. Not built. hi-mlx's eager decode is already at the
practical memory-bandwidth ceiling. (This also settles the open question from the Hy3 MoE notes.)

### Done (in tree, compiles)
- `config.rs`: hoist `text_config` to top level; parse `linear_num_value_heads`,
  `linear_num_key_heads`, `linear_key_head_dim`, `linear_value_head_dim`, `linear_conv_kernel_dim`,
  `full_attention_interval`; `partial_rotary_factor` also read from `rope_parameters`.
- `weights.rs` `load_arrays`: strip `language_model.` prefix, drop `visual.`/`vision_`/`mtp.`.
- `weights.rs` validation: Qwen3.5 branch requires `…layers.0.linear_attn.conv1d.weight`.
- Detected as `family=qwen3`; dispatch on `config.linear_num_value_heads.is_some()`.

### Remaining: the layers (dispatch qwen3 → Qwen35Like when linear heads present)
9B dims: hidden 4096, 32 layers, `full_attention_interval=4` (layer is linear unless `(idx+1)%4==0`).
Linear: k-heads 16, v-heads 32, k/v head-dim 128 → key_dim 2048, value_dim 4096, conv_dim 8192,
kernel 4. Full-attn: 16 heads / 4 kv, head_dim 256, qk_norm, partial-rotary 0.25, rope θ=1e7.

**GatedDeltaNet.forward(x[1,S,4096])** — hold `conv_state[1,3,8192]` + `ssm_state[1,32,128,128]`
per layer (like the KV cache):
```
qkv = in_proj_qkv(x)            # [1,S,8192]
z   = in_proj_z(x)              # [1,S,4096] -> [1,S,32,128]
b,a = in_proj_b(x), in_proj_a(x)# [1,S,32] each
conv_in = concat(conv_state, qkv, axis=1)          # [1,3+S,8192]
conv_state = conv_in[:, -3:, :]
conv_out = silu(conv1d(conv_in, w[8192,4,1], groups=8192, pad=0))   # [1,S,8192]
q,k,v = split(conv_out, [2048,4096], -1)           # q,k -> [1,S,16,128]; v -> [1,S,32,128]
q = (128**-1.0) * rmsnorm_weightless(q); k = (128**-0.5) * rmsnorm_weightless(k)
beta = sigmoid(b)                                   # [1,S,32]
g = exp(-exp(A_log) * softplus(a + dt_bias))        # [1,S,32]
q,k = repeat_heads(q,2), repeat_heads(k,2)          # -> [1,S,32,128]
for t in 0..S:                                      # per-token delta rule
  st   = state * g[:,t][...,None,None]
  kvm  = sum(st * k[:,t][...,None,:], -1)           # [1,32,128]
  dlt  = (v[:,t] - kvm) * beta[:,t][...,None]
  st   = st + k[:,t][...,None,:] * dlt[...,None]
  y[t] = sum(st * q[:,t][...,None,:], -1); state = st
out = rmsnorm(silu(z) * stack(y), norm_w[128])      # RMSNormGated, per head_v_dim
return out_proj(out.reshape(1,S,4096))
```
Full-attn layer = QwenAttention with `rot_dims = head_dim*0.25 = 64`, rope θ=1e7, qk_norm on.
Block = standard 2-norm (`input_layernorm`→sublayer→+res, `post_attention_layernorm`→mlp→+res);
MLP is the plain `Mlp` (separate gate/up/down). `reset_cache` clears conv+ssm state.

**Validation:** compare layer-0 SSM output against `mlx_lm.models.qwen3_5` for one forward before
trusting decode. The delta-rule sign/scale and the GQA head-repeat are the easy things to get wrong.

---

## Appendix: Greedy speculative decoding (`hi-mlx spec`)

Implemented as the memory-bound decode lever discussed above: a small **draft** model proposes `k`
tokens each round, the **target** verifies them in a single forward (one weight read), accepts the
longest matching prefix, and appends the target's own correction/bonus token. Output is identical to
the target's greedy decode. `speculative_generate` in `models.rs`; `NativeRuntime::from_path` +
`speculative_generate`; CLI `hi-mlx spec <target> <draft> --prompt … --k 4`.

Design notes:
- **Fused, one target read/round**: the correction token is kept OUT of the KV cache and prepended to
  the next round's verify (the "anchor" trick), so there's no separate correction forward.
- **KV-cache rollback** (`Cache::rollback` / `CausalLm::rollback_cache`, gated by `supports_rollback`)
  discards rejected drafts by resetting the fixed-buffer write offset. Only the plain-attention
  models (QwenLike) implement it; SSM state (Qwen3.5) can't roll back, so those can't be targets.
- Argmax runs on the GPU (`argmax_axis`) so only `k` integers cross to the CPU, not `k`×vocab logits.

### Correctness: verified
Self-draft (7B verifying 7B) → **100% acceptance, k+1 tokens/round, output byte-identical to greedy**.
That proves the accept/rollback/anchor bookkeeping is right. (Across *different* draft/target pairs the
output can very occasionally differ from greedy by one token — batched-verify vs single-token matmuls
take different MLX kernels, flipping a near-tie argmax. Rare and still valid target output.)

### Results: draft fidelity + small k are what matter (measured)
| target | draft | k | accept | tok/s (spec / greedy) | speedup |
|---|---|---|---|---|---|
| Qwen2.5-Coder-32B-4bit | 0.5B-**8bit** | **3** | **59%** | **33.4 / 29.1** | **1.15×** ✅ |
| Qwen2.5-Coder-32B-4bit | 0.5B-8bit | 4 | 56% | 32.3 / 28.8 | 1.12× |
| Qwen2.5-Coder-32B-4bit | 0.5B-8bit | 6 | 47% | 29.4 / 28.9 | 1.02× |
| Qwen2.5-Coder-32B-4bit | 0.5B-8bit | 8 | 37% | 24.9 / 28.9 | 0.86× |
| Qwen2.5-Coder-32B-4bit | 1.5B-8bit | 3 | 57% | 29.5 / 29.2 | 1.01× |
| Qwen2.5-Coder-32B-4bit | 0.5B-**4bit** | 6 | 20% | 19 / 29 | 0.65× |
| Qwen2.5-Coder-32B-4bit | 3B-4bit | 4 | 18% | 16 / 29 | 0.56× |
| Qwen2.5-Coder-7B-4bit | 0.5B-8bit | 4 | 33% | 55 / 105 | 0.52× |

All outputs identical to greedy. Every knob that matters showed up:
- **Draft fidelity dominates.** Swapping the 0.5B draft from **4-bit → 8-bit** doubled acceptance on
  the 32B (20% → ~59%) and flipped it from 0.65× to a **1.15× win**. The 4-bit draft's own
  quantization noise was making it disagree with the target on the *fresh* branch token.
- **More draft *capacity* ≠ more agreement.** 1.5B-8bit gave the same ~57% acceptance as 0.5B-8bit but
  is slower, so it's net worse (1.01× vs 1.15×). The tiny 8-bit draft is the sweet spot.
- **Small k wins.** k=3 beat k=6/8 — longer draft chains diverge (acceptance falls) and add verify
  positions + draft forwards. Best = high acceptance × few proposals.
- **Target must be memory-bound.** The 7B (~105 tok/s here) can't win even at 100% self-draft accept —
  the draft cost isn't hidden behind a big enough weight read. The win needs a heavy target (32B) and
  is bounded by this machine's high bandwidth (32B greedy is only ~29 tok/s).

Bottom line: implemented, correct, and it **wins** (1.15× on the 32B) with the right recipe — an
**8-bit** tiny draft and **k=3**. Bigger wins need either a more bandwidth-bound target or an even
higher-agreement draft (e.g. a matched-quantization or distilled draft).

### Wired into `serve`
`hi-mlx serve <target> --draft <draft> --spec-k 3` enables speculative decoding for the OpenAI-compatible API: greedy (`temperature=0`) requests use the draft+verify path, sampling requests fall back to the normal loop, and streaming (SSE) works through it. Measured **1.24×** end-to-end on the 32B (35.9 vs ~29 tok/s) for a prime-check prompt. The draft must share the target's tokenizer and the target must support KV-cache rollback (Qwen2/Qwen3; startup errors otherwise).

### GLM-5.2 MTP self-speculation (built from the paper — no reference exists)
GLM-5.2 ships a DeepSeek-V3-style MTP head (layer 78: `eh_proj`/`enorm`/`hnorm` + a full MLA+MoE
block + `shared_head.norm`), which no MLX loader runs (they all drop it). Implemented it as a
self-speculative "draft": the trunk's pre-final-norm hidden `h_i` + `embed(t_{i+1})` → the MTP head
predicts `t_{i+2}`; the trunk verifies the proposal in one forward, so accepted proposals give two
tokens per trunk read. `MtpHead` + `MlaLike::{forward_hidden, mtp_generate}`; auto-enabled for greedy
requests when the head is present (`HI_MLX_DISABLE_MTP=1` to force the plain loop).

Two bugs found (no reference to check against):
- **Missing per-tensor quant entries.** The dynamic-3.5bpw build's config omits quantization entries
  for the MTP layer, so hi-mlx defaulted its tensors to 4-bit while they're actually 3/4/6-bit →
  `quantized_matmul failed`. Fix: **infer bits from the packing** in `quant_spec_for`
  (`bits = 32·in_packed/(n_groups·group_size)`) rather than trusting the config default. This also
  hardens every dynamic/mixed-bit model.
- **eh_proj concat order.** GLM-5.2 orders the eh_proj input as `[enorm(embed); hnorm(hidden)]` — the
  reverse of the DeepSeek-V3 paper. The paper order gave **0% MTP acceptance** (systematic garbage);
  flipping it gave **55–70%**. (`HI_MTP_HFIRST=1` selects the paper order for other MTP models.)

Result: **~1.38× decode** on GLM-5.2 (0.25 → ~0.35 tok/s), output identical to greedy. MTP acceptance
55–70% by content. Still not interactive at 355B, but the mechanism works and is exact. A fused
verify (fold the rejection-correction into the next round) would push it further.

### Self-calibrating speculation gate
Speculation (draft or MTP) helps a slow/memory-bound target but *hurts* a fast one, and for MoE targets
it's content-dependent (acceptance varies) — a static "big model → on" heuristic mis-fires. So the gate
**measures**: on the first greedy request per model, it warms the trunk, probes the plain decode rate,
and (unless the trunk is clearly slow, <8 tok/s, where speculation always wins and the spec probe is
skipped) probes the spec rate, enabling speculation only if it actually beats plain. The decision is cached per model and
**re-calibrated every `HI_MLX_SPEC_RECAL` greedy requests** (default 64; 0 disables), on that request's
prompt, so it tracks workload shifts — a session that changes topic can flip acceptance and the decision. Crucially it probes on a **prefix of the real request** — acceptance is
content-dependent, so a generic probe under-measures it (e.g. Qwen2.5-Coder-32B + 8-bit draft: a
generic prompt measured spec *slower*, a code prompt measured it faster → enabled). Warmup matters too:
the first forward compiles Metal shaders, so an un-warmed first probe reads ~5-8× slow and skews the
comparison. Verified: 7B → disabled (plain 86 vs spec 63), 32B/code → enabled (22 vs 23.5), GLM-5.2 →
enabled via the slow pre-filter. `HI_MLX_DISABLE_MTP=1` still hard-disables MTP.

### Arch coverage sweep (2026-07) — newest Qwen 27B + 30B-A3B, and a MoE gap
Ran the expanded acceptance matrix across every supported family. **Works (coherent generation):**
`qwen2`, `qwen3`, `qwen3_moe`, `qwen3_5`, `glm4`, `glm4_moe_lite`, `glm_moe_dsa` (GLM-5.2). In
particular, the two most-requested current Qwen models run **out of the box, no code change**:

- **Qwen3-30B-A3B** (`qwen3_moe`, 128-expert MoE) — the popular "30B-A3B". Routes to `QwenLike` +
  `QwenMoe` (`is_qwen_moe_layer` keys off `decoder_sparse_step`/`mlp_only_layers`). "Paris", `sum(my_list)`.
- **Qwen3.5-27B** (`qwen3_5`, SSM/gated-delta hybrid dense) — the newest 27B. Routes to `Qwen35Like`
  (same arch validated earlier on the 9B distill). Coherent reasoning output.

**Known gap — `qwen3_5_moe`** (SSM hybrid *plus* MoE, e.g. the Qwen3.5-35B-A3B distill): fails to load
with `missing tensor model.layers.0.mlp.gate_proj.weight`. `Qwen35Like` builds a dense `Mlp` per layer,
but this variant's layers are MoE (`mlp.gate` + `mlp.switch_mlp` + `mlp.shared_expert` +
`mlp.shared_expert_gate`, 256 experts) — the same structure `QwenMoe` already handles for `qwen3_moe`.
Fix is a FFN branch in the Qwen3.5 layer (dense `Mlp` vs `QwenMoe`), not a new arch.

Aside: the matrix's tool-call smoke step fails for models not tuned for tool-calling (Qwen2.5-Coder-7B
emits a fenced ```json block; GLM-4-9B-0414 rambles) — model capability, not arch support. The core
arch gate (inspect + coherent non-stream + streaming chat) passes for all of the above.

### qwen3_5_moe support — closing the SSM-hybrid MoE gap (two bugs)
The Qwen3.5 *MoE* variant (SSM/gated-delta hybrid **plus** a 256-expert shared-expert MoE, e.g.
Qwen3.5-35B-A3B) now runs. Two bugs, both found via the acceptance matrix + a NaN-localization probe:

1. **Dense MLP where the layer is MoE.** `Qwen35Layer` hard-built a dense `Mlp`, so the variant failed
   to load (`missing model.layers.0.mlp.gate_proj.weight`). Fix: swap it for the existing `QwenFfn`
   enum (`Dense(Mlp)` / `Moe(QwenMoe)`), which dispatches per layer on `is_qwen_moe_layer` — one line,
   and `QwenMoe` already handles the shared-expert + `shared_expert_gate` layout this model uses.

2. **NaN from `log(0)` in the chunk-parallel scan.** After (1) the model loaded but emitted `!!!!`.
   A per-layer + per-op probe pinpointed the SSM mixer: the gated-delta decay `g = exp(neg_a·softplus)`
   **underflows to exactly 0** on this model (`neg_a·softplus ≈ -1000`; the 9B's decays don't), and
   `scan_chunked` took `g.log()` → `-inf`, so the `lg_t - lg_j` decay differences became `-inf-(-inf)
   = NaN`. Fix: clamp `g` to a `1e-30` floor before the log — where `g` underflows the decay is already
   complete, so `exp(-69) ≈ 0` is numerically exact. (The additive-mask trick already handled the
   masked triangle; this handles the diagonal/kept entries whose own `lg` was `-inf`.)

Verified: Qwen3.5-35B-A3B loads and generates coherently (reasoning `<think>` traces, arithmetic, code).
The clamp also hardens the dense qwen3_5 path against any future extreme-decay model.
