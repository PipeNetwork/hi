# PipeNetwork MLX model coverage in hi-mlx

Support status for the MLX models published under [huggingface.co/pipenetwork](https://huggingface.co/pipenetwork)
(72 MLX repos, ~22 distinct base models across 11 architectures). Reviewed 2026-07.

`hi-mlx` support is by **architecture** (`model_type`), so one arch covers every quant/REAP variant of a model.

## ✅ Supported (7 archs)

| `model_type` | PipeNetwork models | hi-mlx path | notes |
|---|---|---|---|
| `qwen3` | FrogBoss-32B-2510, FrogMini-14B-2510 | `QwenLike` | dense Qwen3 |
| `qwen3_moe` | Rio-3.1-Open-30B | `QwenLike` + `QwenMoe` | validated (Qwen3-30B-A3B) |
| `qwen3_5` | Holo-3.1-4B/9B, VISTA-4B/9B | `Qwen35Like` | SSM/gated-delta hybrid, dense |
| `qwen3_5_moe` | Holo-3.1-35B-A3B, Ornith-1.0-397B, Qwen3.6-35B-A3B | `Qwen35Like` + `QwenFfn` | SSM hybrid + shared-expert MoE; fixed 2026-07 (FFN branch + scan `log(0)` clamp) |
| `hy_v3` | Hy3-REAP50/62/75 | `QwenLike` + MoE | Hunyuan-3 |
| `glm_moe_dsa` | GLM-5.2 (4/5/6/8-bit, mixed, nvfp4, REAP25/37/50), Macaron-V1-749B | `MlaLike` (DeepSeek) | DeepSeek-V3.2 arch (MLA + DSA indexer + MoE) |
| `kimi_k25` | Kimi-K2.7-Code | `MlaLike` (DeepSeek) | thin DeepSeek-V3 wrapper; routed 2026-07 (arch-verified, untested @ ~1T params) |
| `nemotron_h` | Nemotron-3-Nano-4B/30B-A3B, Ultra-550B-A55B | `NemotronHLike` | Mamba2 + attention + ReLU² MLP + MoE hybrid; built 2026-07, validated on Nano-4B (dense) and Nano-30B-A3B (MoE). Ultra-550B arch-verified (untested @ 550B). TwoTower is a separate diffusion variant. |
| `gemma4` | Gemma-4-31B (dense), Gemma-4-26B-A4B (MoE) | `Gemma4TextLike` | sliding/full hybrid attn, dual RoPE, k==v, GeGLU, softcap; built 2026-07, validated on 31B. Ships its chat template in a separate `chat_template.jinja` (channel/turn format) — hi-mlx now reads it. 26B MoE untested. |
| `minimax_m3` | MiniMax-M3 | `MiniMaxLike` | GQA (partial RoPE + per-head qk-norm) + sigmoid-MoE with shared expert + SwiGLU-OAI; **(1+weight) RMSNorm**; built 2026-07, validated on the 3-bit build |

## ❌ Not yet supported (4 new architectures)

Each is a genuinely new model implementation (not a routing tweak). mlx_lm references live in the
sibling `*-mlx/.venv` checkouts.

| `model_type` | PipeNetwork models | what it is | effort |
|---|---|---|---|
| `longcat2` | LongCat-2.0 (REAP50/62/75/hard) | MLA + custom LongCat MoE + ngram embedding + indexer | large — custom `LongcatFlashMoE` (zero-computation experts) + `NgramEmbedding`; not a DeepSeek wrapper despite the MLA fields |

### Recommended build order
1. **`nemotron_h`** — widest coverage (4 families incl. the 550B/Ultra flagship + TwoTower); the ~2 GB Nano-4B makes iteration cheap.
2. **`gemma4`** — 2 families, standard-ish attention (no SSM) but many Gemma-specific details; both variants downloadable.
3. **`minimax_m3`** — 1 family; smallest reference.
4. **`longcat2`** — 1 family; hardest (custom MoE + ngram).

Each should be built with on-device verification (coherence-gated, like the acceptance matrix) and is a
multi-hour effort — pick per deployment priority.
