# PipeNetwork MLX model coverage in hi-mlx

Support status for the MLX models published under [huggingface.co/pipenetwork](https://huggingface.co/pipenetwork)
(72 MLX repos, ~22 distinct base models). Originally reviewed 2026-07; **every PipeNetwork architecture is
now supported.**

`hi-mlx` support is by **architecture** (`model_type`), so one arch covers every quant / REAP / bit-width
variant of a model. Validation is coherence-gated on device (the acceptance matrix), per-arch below.

## ✅ Supported architectures (12)

| # | `model_type` | PipeNetwork models | hi-mlx path | status & notes |
|---|---|---|---|---|
| 1 | `qwen2` / `qwen2_moe` | *(base arch — PipeNetwork's Qwen line is qwen3+)* | `QwenLike` (+ `QwenMoe`) | foundational dense/MoE Qwen2; the base the rest of the Qwen path builds on |
| 2 | `qwen3` | FrogBoss-32B-2510, FrogMini-14B-2510 | `QwenLike` | dense Qwen3 |
| 3 | `qwen3_moe` | Rio-3.1-Open-30B | `QwenLike` + `QwenMoe` | validated (Qwen3-30B-A3B) |
| 4 | `qwen3_5` | Holo-3.1-4B/9B, VISTA-4B/9B | `Qwen35Like` | SSM / gated-delta-net hybrid, dense |
| 5 | `qwen3_5_moe` | Holo-3.1-35B-A3B, Ornith-1.0-397B, Qwen3.6-35B-A3B | `Qwen35Like` + `QwenFfn` | SSM hybrid + shared-expert MoE; fixed 2026-07 (FFN branch + scan `log(0)` clamp) |
| 6 | `hy_v3` | Hy3-REAP50/62/75, Hy3-2/4/6/8bit | `QwenLike` + MoE | Hunyuan-3; validated 2026-07 on Hy3-REAP50 (85 GB) |
| 7 | `glm_moe_dsa` | GLM-5.2 (4/5/6/8-bit, mixed, nvfp4, REAP25/37/50), Macaron-V1-749B | `MlaLike` | DeepSeek-V3.2 arch: MLA (absorbed) + DSA indexer + MoE |
| 8 | `kimi_k25` | Kimi-K2.7-Code | `MlaLike` | thin DeepSeek-V3 wrapper; routed 2026-07 (arch-verified). **The one supported arch that can't run locally**: Kimi-K2.7-Code-MLX-4bit-hiprec is ~644 GB, over the ~550 GB host RAM — so it's the only arch not exercised by the acceptance matrix. |
| 9 | `nemotron_h` | Nemotron-3-Nano-4B/30B-A3B, Ultra-550B-A55B | `NemotronHLike` | Mamba2 + attention + ReLU² MLP + MoE hybrid; built 2026-07, validated on Nano-4B (dense) & Nano-30B-A3B (MoE). Ultra-550B arch-verified (untested @ 550B). TwoTower is a separate diffusion variant. |
| 10 | `gemma4` | Gemma-4-31B (dense), Gemma-4-26B-A4B (MoE) | `Gemma4TextLike` | sliding/full hybrid attn, dual RoPE, k==v, GeGLU, softcap; built 2026-07, validated on 31B. Reads the chat template from a separate `chat_template.jinja` (channel/turn format). 26B MoE untested. |
| 11 | `minimax_m3` | MiniMax-M3 | `MiniMaxLike` | GQA (partial RoPE + per-head qk-norm) + sigmoid MoE with shared expert + SwiGLU-OAI; **(1+weight) RMSNorm**; built 2026-07, validated on the 3-bit build |
| 12 | `longcat2` | LongCat-2.0 (REAP50/62/75) | `LongCatLike` | ScMoE (2 absorbed-MLA attns + 2 dense MLPs + shortcut softmax-MoE w/ 128 identity zero-experts) + n-gram hash embedding + YARN rope; built 2026-07, validated on REAP75 4-bit |

**Validation legend:** *validated* = ran coherently on device and passes the acceptance matrix (inspect +
chat non-streaming + streaming). *arch-verified* = forward implemented + reviewed against the mlx_lm
reference but not run at full scale (weights too large to load locally).

**Full-matrix run (2026-07):** all 13 runnable model types pass — the 11 validated archs above plus the
two GLM-4 code paths the matrix also exercises (`glm4` GQA, `glm4_moe_lite` MLA). Only `kimi_k25` is left
out, purely because its ~644 GB exceeds host RAM. See `scripts/hi_mlx_acceptance_matrix.sh`.

## How the hardest ones were cracked

The four newest arches (`nemotron_h`, `gemma4`, `minimax_m3`, `longcat2`) each needed a full new forward,
not a routing tweak. The repeatable method: **tensor-diff every sub-step against the mlx_lm reference on
fixed token ids** (embedding → per-layer → attention/MLP/MoE → per-expert → logits) until the diverging op
is isolated. For an arch the reference can't load (MiniMax-M3), *build* a reference forward in mlx from the
modeling spec and diff against that. Bugs this caught, worth remembering:

- **MiniMax-M3** — `(1 + weight)` RMSNorm convention (stored norm weights are deviations from 1); using
  weight-only gave input-independent garbage.
- **LongCat-2.0** — `norm_topk_prob` defaults to `true` in hi-mlx but LongCat omits it and needs `false`;
  normalizing the top-12 weights inflated the MoE output 2.75×. Also: n-gram hashing embedding (i64
  modular), absorbed-MLA + DSA indexer reused wholesale from the GLM/DeepSeek path, YARN rope.
- **Gemma-4** — ships its chat template in a separate `chat_template.jinja` (custom channel/turn format).

## CI

`cargo test -p hi-mlx` runs on an Apple-silicon runner on every PR (`.github/workflows/ci.yml`), so all 12
architectures are compiled and unit-tested on each change — not just locally.
