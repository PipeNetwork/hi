# hi-mlx architecture coverage

`hi-mlx` (the Apple-silicon MLX inference sidecar) supports models by **architecture** (`model_type`),
so one arch covers every quant / REAP / bit-width variant of a model. Validation is coherence-gated on
device: each arch is served and must answer fixed prompts coherently (capital of France, primary colors,
2+2, a haiku) — see `scripts/hi_mlx_acceptance_matrix.sh`.

This started as PipeNetwork-model coverage (the first 12 archs) and has since expanded to the broader
mlx_lm architecture set. **27 architectures are supported and validated on device**; a few more are
arch-verified or blocked (see the last section).

## The delivery pattern

Most archs are *not* a new forward pass — they ride one of two shared paths with config-gated or
weight-detected quirks:

- **`QwenLike`** — the universal GQA transformer (dense + MoE). Absorbs anything that is "Llama/Qwen with
  a twist": scalar multipliers (Granite), per-layer NoPE (SmolLM3), post-norm residual (OLMo2/EXAONE-4),
  full vs per-head qk-norm (OLMo2/OLMoE), softmax-topk MoE with individual **or** pre-stacked experts,
  shared experts (singular `shared_expert` or plural `shared_experts`), dense/MoE layer splits.
- **`Gemma4TextLike`** — sliding/full hybrid attn, dual RoPE, GeGLU, sandwich norms; parameterized to
  cover Gemma-2/3/4.

Genuinely different shapes get a **small dedicated impl** that still reuses hi-mlx primitives (Linear,
Embedding, RmsNorm/LayerNorm, `rope`, sdpa, Cache, SwitchMlp): NemotronH, MiniMax, LongCat, Gemma4,
Nemotron (LM), GPT-OSS, Cohere2, Llama-4, Phi-MoE.

The hardest arches were cracked by **tensor-diffing every sub-step against the mlx_lm reference on fixed
token ids** until the diverging op is isolated. For an arch the reference can't load, *build* a reference
forward in mlx from the modeling spec and diff against that.

## ✅ Supported — PipeNetwork core (12)

| `model_type` | hi-mlx path | notes |
|---|---|---|
| `qwen2` / `qwen2_moe` | `QwenLike` (+MoE) | foundational dense/MoE Qwen2 |
| `qwen3` / `qwen3_moe` | `QwenLike` (+MoE) | dense + MoE (Qwen3-30B-A3B) |
| `qwen3_5` / `qwen3_5_moe` | `Qwen35Like` | SSM / gated-delta-net hybrid, dense + shared-expert MoE |
| `hy_v3` | `QwenLike` + MoE | Hunyuan-3; sigmoid routing + expert bias |
| `glm_moe_dsa` | `MlaLike` | DeepSeek-V3.2 arch: absorbed MLA + DSA indexer + MoE (GLM-5.2) |
| `kimi_k25` | `MlaLike` | DeepSeek-V3 wrapper — **arch-verified only** (tokenizer, see below) |
| `nemotron_h` | `NemotronHLike` | Mamba2 + attention + ReLU² MLP + MoE hybrid |
| `gemma4` | `Gemma4TextLike` | sliding/full hybrid, dual RoPE, GeGLU, softcap |
| `minimax_m3` | `MiniMaxLike` | GQA (partial RoPE + qk-norm) + sigmoid MoE + shared expert; (1+weight) RMSNorm |
| `longcat2` | `LongCatLike` | ScMoE (double absorbed-MLA + shortcut softmax-MoE) + n-gram embedding + YARN |
| `glm4` / `glm4_moe_lite` | `Glm4Like` / `MlaLike` | GQA GLM-4 + MLA lite |

## ✅ Supported — general architectures (15, added this expansion)

| `model_type` | hi-mlx path | validated on | key quirk |
|---|---|---|---|
| `granite` | `QwenLike` | granite-3.x | embedding/residual/attention/logits scalar multipliers |
| `smollm3` | `QwenLike` | SmolLM3-3B | per-layer NoPE (`no_rope_layers`); think-off render |
| `exaone4` | `QwenLike` (Qwen3) | EXAONE-4 | post-norm residual (sandwich) + per-head qk-norm |
| `olmo2` | `QwenLike` (Qwen3) | OLMo-2 | post-norm residual + full qk-norm |
| `seed_oss` | `QwenLike` | Seed-OSS-36B | drop-in SwiGLU; `<seed:bos>` render w/ thinking_budget 0 |
| `nemotron` | `NemoLm*` (dedicated) | Nemotron-Mini-4B | LayerNorm1P + squared-ReLU MLP + partial rope |
| `gpt_oss` | `GptOssLike` (dedicated) | gpt-oss-20b | **attention sinks** + biased top-k-softmax MoE + SwiGLU-OAI + harmony template |
| `gemma3` / `gemma3_text` | `Gemma4TextLike` | gemma-3-1b | (1+weight) norm + `query_pre_attn_scalar` sdpa scale + full rope + `sliding_window_pattern` |
| `gemma2` | `Gemma4TextLike` | gemma-2-2b | Gemma-3 minus qk-norm (Option) |
| `cohere2` | `CohereLike` (dedicated) | Command-R7B | LayerNorm + parallel attn/MLP block + NoPE on full layers + logit_scale |
| `llama4` / `llama4_text` | `Llama4Like` (dedicated) | Llama-4-Scout text | iRoPE + weightless L2 qk-norm + llama3 rope scaling + top-1 sigmoid MoE + shared expert |
| `olmoe` | `QwenLike` (Qwen3) | OLMoE-1B-7B | Qwen3-MoE + full qk-norm; individual experts auto-stacked |
| `ernie4_5_moe` | `QwenLike` (Qwen2) | ERNIE-4.5-21B-A3B | softmax-topk MoE + `shared_experts` (plural); dense prefix |
| `phimoe` | `PhiMoeLike` (dedicated) | Phi-3.5-MoE | **SuScaledRoPE (LongRoPE)** + LayerNorm + biased attn + top-2 MoE |

Chat-template family (in `hi-local-core/prompt.rs`, detected by marker substrings): gemma turn/standard,
minimax, longcat, granite, llama3, smollm3, seed-oss, gpt-oss harmony, cohere command-r, llama-4, phi-3.
Reasoning models are rendered thinking-off; per-arch chat stop tokens are injected where the eos differs
from the turn-end token (Gemma `<end_of_turn>` 106, Phi `<|end|>` 32007).

## Notable bugs (worth remembering)

- **MiniMax-M3 / Gemma-3** — `(1+weight)` RMSNorm: stored norm weights are deviations from 1; weight-only
  gives input-independent garbage. Gemma-4 (pipenetwork export) folds the +1 in, so the convention is
  per-export — gate it on `model_type`.
- **Gemma-3 sdpa scale** — needs `query_pre_attn_scalar^-0.5`; Gemma-4 folds it into q_norm (scale 1.0).
  The tell: per-layer *magnitudes* matched but the top token was wrong (softmax preserves output scale),
  so only generation exposed it.
- **Llama-4 lm_head** — leaves `tie_word_embeddings` unset (hi-mlx defaults it true) but ships a separate
  lm_head; detect the weight, not the flag. (Tensor-diff: hidden states matched, top token garbage.)
- **ERNIE-4.5 shared expert** — named `shared_experts` (plural); QwenMoe only looked for `shared_expert`,
  silently dropping ~half the FFN → garbage. Diagnosed by elimination (OLMoE also has a quantized gate and
  worked), avoiding a 21B tensor-diff.
- **LongCat-2.0** — `norm_topk_prob` defaults true in hi-mlx but LongCat omits it and needs false.
- **GPT-OSS attention sinks** — free: the mlx-rs sdpa already exposes a `sinks` param (hi-mlx threaded it
  as the 6th arg, always None until now).

## Arch-verified or blocked (not runnable on device)

| `model_type` | status | blocker |
|---|---|---|
| `kimi_k25` | arch-verified | tiktoken tokenizer, **no `tokenizer.json`** (slow tokenizer, can't generate one). Thin DeepSeek-V3 wrapper on the validated `MlaLike` path. *(Correction to an earlier note: the 465 GB quant fits the 550 GB host — the blocker is the tokenizer, not RAM.)* |
| `internlm3` | routed, unvalidatable | mlx-community repo ships only `tokenizer.model`, no `tokenizer.json`. Runs on the Qwen path once a `tokenizer.json` is added. |
| `ernie4_5_moe` | **supported** (see above) | the mlx repo omits `tokenizer.json`; grab it from base `baidu/ERNIE-4.5-21B-A3B-PT` to run. |
| `dots1` (dots.llm1) | not implemented | only MLX model is **mixed-4-6bit**, and hi-mlx does not support mixed quantization. Arch is GQA + per-head qk-norm + DeepSeek-style sigmoid MoE (n_group 1, shared expert) — would ride `QwenLike` + the DeepSeek gate once a uniform-quant model exists (or mixed-quant support is added). |
| `granitemoe` | not implemented | **no MLX model published**. IBM fused-expert format (`block_sparse_moe.input_linear`/`output_linear` → SwitchGLU) + `router.layer` naming + the Granite scalar multipliers (already in config). Ready to add against a real model. |

Both `dots1` and `granitemoe` were left unimplemented **deliberately**: without a runnable model there is
no way to validate a new forward, and shipping unverified arch code would misreport coverage. They are
documented here so the work is scoped when a model appears.

## CI

`cargo test -p hi-mlx` runs on an Apple-silicon runner on every PR (`.github/workflows/ci.yml`), so every
architecture is compiled and unit-tested on each change — not just locally. The CPU crates
(`cargo test --workspace --exclude hi-mlx`) + `cargo fmt` gate on Linux; clippy is advisory.
