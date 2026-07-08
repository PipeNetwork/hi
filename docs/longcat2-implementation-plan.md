# LongCat-2.0 (`longcat2`) implementation plan

Status: **scaffolding landed** (family routing + detect + config detection; `load_model` bails with an
informative message). The four subsystems below remain — best built with the 302 GB REAP75 model
downloaded so each piece can be tensor-diffed against the reference.

Reference: `longcat2.py` (+ `longcat_flash.py`, `longcat_flash_ngram.py`, `mla.py`, `indexer`) in the
`longcat2-mlx` venv. **Unlike MiniMax-M3, the mlx_lm reference should load this model directly** — verify
with `mlx_lm.utils.load_model(Path(D))` and use it for the layer-by-layer tensor diff (the technique that
cracked Gemma-4 and MiniMax-M3).

## Config (REAP75)
`num_layers=38` (→ num_hidden_layers), `hidden_size=8192`, `num_attention_heads=64`, MLA
`q_lora_rank=1536 / kv_lora_rank=512 / qk_nope=128 / qk_rope=64 / v_head_dim=128`, indexer
`index_n_heads=32 / index_head_dim=128 / index_topk=2048 / index_local_tokens=1024 / index_init_tokens=16`,
MoE `n_routed_experts=192 / zero_expert_num=128 (type=identity) / moe_topk=12 / expert_ffn_hidden_size=2048
/ routed_scaling_factor=9`, dense `ffn_hidden_size=12288`, ngram `oe_split_num=4 / oe_neighbor_num=5 /
oe_vocab_size_ratio=100.567`, `mla_scale_q_lora=mla_scale_kv_lora=True`, `rope_theta=1e6`,
`rms_norm_eps=1e-5`, `vocab_size=163840`, `mtp_num_layers=3` (MTP heads — can be dropped like GLM-5.2).
**Check the RMSNorm convention first** (weight-only vs `(1+weight)` — the MiniMax-M3 trap): inspect
`model.norm.weight` mean (≈0 → weight-only; ≈−1 → add 1 at load).

Config fields to add: `oe_split_num`, `oe_neighbor_num`, `oe_vocab_size_ratio`, `zero_expert_num`,
`moe_topk`, `expert_ffn_hidden_size`, `ffn_hidden_size`, `mla_scale_q_lora`, `mla_scale_kv_lora`; map
`num_layers`→`num_hidden_layers`. (MLA + index_* fields already exist from the DeepSeek path.)

## Subsystem 1 — NgramEmbedding (`longcat_flash_ngram.py`) — the input path, highest-risk
`num_embedders = oe_split_num*(oe_neighbor_num−1) = 4*4 = 16`, `emb_dim = hidden/16 = 512`, each embedder
vocab `m + i*2 + 1` where `m = round(oe_vocab_size_ratio*vocab)`. Per token: `h = word_embeddings(ids)`
plus, for each n in 2..=5 and split j in 0..4, an n-gram id = `ids + Σ shifted_ids[k]*vocab_mods[k]`
(modular: `power_mod = (power_mod*vocab) % emb_vocab_dim`), looked up in embedder `(n-2)*k+j`, projected by
`post_projs[idx]` (Linear emb_dim→hidden), summed. EOS-aware `reach` masks n-grams across document
boundaries. **Port the modular hashing exactly** — an off-by-one → garbage. Tensor-diff `ngram_embeddings(ids)`
against the reference on fixed ids first, before anything else.

## Subsystem 2 — MLA attention + Indexer (`longcat2.py::Longcat2Attention`, `Indexer`)
DeepSeek-V3.2 shape: `q_a_proj → q_a_layernorm → q_b_proj`; `kv_a_proj_with_mqa → split → kv_a_layernorm`;
nope/rope split; rope on q_pe/k_pe (θ=1e6); `mla_scale_q_lora/kv_lora` scale the lora latents. The **Indexer**
(when `not skip_topk`) produces `topk_indices` used to gather kv_latent/k_pe (sparse attention, top-2048).
`skip_topk` layers reuse the previous layer's `topk_indices`. **hi-mlx already has `MlaLike` + `MlaIndexer`
(glm_moe_dsa/deepseek_v32)** — reuse them; the deltas are `MultiLinear` (embed_q/unembed_out) and the
per-layer `skip_topk` sharing. For contexts < index_topk the indexer selects everything (= full attention),
so short-prompt validation can start without the sparse path.

## Subsystem 3 — ScMoE decoder (`Longcat2DecoderLayer`) — double attention + shortcut MoE
Each of the 38 layers holds **2 attentions + 2 dense MLPs + 1 MoE**, run as a 2-iteration loop:
```
for i in 0,1:
  h = residual + attn[i](input_ln[i](h), topk_indices)     # topk shared i=0→i=1
  r = post_attn_ln[i](h)
  if i==0: shortcut = moe(r)                                # MoE computed once, at i=0
  h = r_residual + mlps[i](r)                               # dense MLP
  if i==1: h = h + shortcut                                 # add the MoE output
```
Weight prefix: `self_attn.{0,1}`, `mlps.{0,1}`, `input_layernorm.{0,1}`, `post_attention_layernorm.{0,1}`,
`mlp` (the MoE).

## Subsystem 4 — LongCat MoE (`longcat_flash.py::LongcatFlashMoE`) — zero-computation experts
Router: **softmax** over `router_logits` (NOT sigmoid), `+ e_score_correction_bias` for selection,
`argpartition` top-`moe_topk`=12 over `n_routed_experts + zero_expert_num` = 192+128 = 320, gather the
softmax weights, optional `norm_topk_prob`, `* routed_scaling_factor` = 9. Experts 0..191 are real SwiGLU
(`SwitchGLU`, expert_ffn_hidden_size=2048); experts 192..319 are **identity/zero** (`zero_expert_type=identity`
→ contribute the input, or 0 — check the reference `forward`). Dense MLPs use `swiglu(gate,up)` (plain silu,
ffn_hidden_size=12288). Reuse `SwitchMlp` for the real experts; handle the zero-experts by skipping them (they
add `x` or nothing per their type).

## Build order (each step tensor-diffed vs the mlx_lm reference on fixed ids)
1. NgramEmbedding → match `ngram_embeddings(ids)`.
2. One MLA attention (skip indexer) + dense MLP → match a single sublayer.
3. LongCat MoE → match `mlp(r)`.
4. ScMoE decoder layer (full 2-iteration) → match one layer.
5. Full model + indexer → match final logits / top token.
6. Chat template (read `chat_template.jinja`, add a renderer if custom like Gemma-4/MiniMax).

Also drop the MTP head weights (`mtp_num_layers=3`) at load, like GLM-5.2.

---

## Deep-scope findings (2026-07-08) — everything needed to execute

**The hardest subsystem is already done.** hi-mlx's `MlaAttention` + `MlaIndexer` + `MultiLinear`
(built for glm_moe_dsa/deepseek_v32) IS the LongCat absorbed MLA + DSA indexer: same `q_a_proj →
q_a_layernorm → q_b_proj`, `kv_a_proj_with_mqa` split, `embed_q`/`unembed_out` absorption, `pe_scores`
mask, `traditional=true` rope, and the indexer returns `None` (full attention) for prompts ≤ index_topk
(2048) — so short-prompt validation exercises full MLA. **Reuse `MlaAttention`**; it needs three additions
for LongCat:
1. `mla_scale_q_lora`/`mla_scale_kv_lora`: scale `query_latent` by `(hidden/q_lora_rank)^0.5` and
   `kv_latent` by `(hidden/kv_lora_rank)^0.5` (config flags both true). Equivalent to scaling q after
   q_b_proj since attention_bias=false.
2. **YARN rope** (config `rope_scaling: deepseek_yarn`, factor=120, mscale_all_dim=1, orig_max=8192):
   precompute yarn freqs + pass via `rope(..., freqs)`; multiply `scale` by `s²` where
   `s = 0.1*mscale_all_dim*log(factor)+1 ≈ 1.479`. hi-mlx rope() already accepts a `freqs` arg.
3. `topk_indices` sharing: `forward` must accept+return `topk_indices` so ScMoE attn[0] (indexer) feeds
   attn[1] (skip_topk). For prompts ≤2048 the indexer is a no-op, so this can be deferred for first light.

**Weight names confirmed** (match the reference, mlx_lm sanitize keeps them):
`model.ngram_embeddings.{word_embeddings, embedders.{0..15}, post_projs.{0..15}}`;
`model.layers.{L}.self_attn.{0,1}.{q_a_proj,q_a_layernorm,q_b_proj,kv_a_proj_with_mqa,kv_a_layernorm,
embed_q,unembed_out,o_proj,indexer.*}`; `model.layers.{L}.mlps.{0,1}.{gate,up,down}_proj` (dense,
ffn=12288); `model.layers.{L}.mlp.{switch_mlp.{gate,up,down}_proj (192 experts, ffn=2048),
router.classifier (320-out, **8-bit**), router.e_score_correction_bias}`;
`model.layers.{L}.{input_layernorm,post_attention_layernorm}.{0,1}`; `model.norm`, `lm_head`. Quant is
4-bit gs=64 except the per-layer router.classifier (8-bit) — hi-mlx infers bits from shapes.

**NgramEmbedding — decoded + ground-truthed** (`/tmp/lc_ngram_gt.npy` for ids [10,20,30,2000,5000,150,7,8]):
- `m = ngram_vocab_size_ratio(=oe_vocab_size_ratio=100.567) * vocab(163840)`; embedder `idx` vocab =
  `int(m + idx*2 + 1)` (embedder.0 = 16476898, all 4-bit gs=64, emb_dim = hidden/16 = 512); post_projs
  are Linear 512→8192 (4-bit).
- Prefill forward (cache=None): `x = word_emb(ids)`; for `i in 2..=5`, `j in 0..4`, `idx=(i-2)*4+j`,
  `evd=int(m+idx*2+1)`, `shifted[t]=shift_right(ids, t-1)` (zeros fill; EOS-aware `reach` mask only if
  eos_token_id=2 appears), `vocab_mods` via `pm=(pm*vocab)%evd` accumulated, `ngram_id = ids +
  Σ_{t=2..i} shifted[t]*mods[t-2]` (**i64 arithmetic**), `new = (ngram_id % evd)`, `x +=
  post_projs[idx](embedders[idx](new))`. Finally **`x /= 1 + k*(n-1) = 17`**. Do the lookups quantized
  (gather rows then dequantize) — the tables are up to 4 GB each; never dequantize whole.

**LongCat MoE** (`mlp`): softmax(classifier(x)) over 320; `+ e_score_correction_bias` (loaded, may be
~0) → argpartition top-12; gather softmax weights; `norm_topk_prob=false`; `* routed_scaling=9`. Experts
`idx<192` run through `switch_mlp` (silu SwiGLU, reuse `SwitchMlp`); experts `idx>=192` are **identity**:
add `x * Σ(weights of selected zero-experts)`. No shared expert. No group masking (n_group=1).

**ScMoE decoder** (per layer, loop i in 0,1): `h = res + attn[i](input_ln[i](h), topk_shared)`;
`r = post_attn_ln[i](h)`; `if i==0: shortcut = moe(r)`; `h = r_res + mlps[i](r)`; `if i==1: h += shortcut`.
`mlps` are plain-SwiGLU dense (`Mlp`), `mlp` is the LongCat MoE, both attns are `MlaAttention`.

**Model**: `LongCatLike` = NgramEmbedding → 38 ScMoE layers → `norm` (plain RMSNorm — **check mean: is it
weight-only or (1+weight)? the MiniMax trap**) → untied `lm_head`. No embed scale, no logit softcap.
Prompt: read `chat_template.jinja`; add a renderer if the format is custom.

**Build/validate order** (mlx_lm reference doesn't `nn.quantize`-load cleanly, so tensor-diff via a manual
dequant harness like `/tmp/lc_ngram_gt.npy`): ngram → one MLA sublayer → MoE → one ScMoE layer → full
logits. Blocked only on the 302 GB download finishing (was interrupted at 64 GB; resuming).
