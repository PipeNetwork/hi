# DeepSeek-V4-Flash speculative decoding: MTP + DFlash in hi-cuda

Status: plan (2026-07-12). Companion to `deepseek-v4-flash-port-spec.md` (target
forward math) and `deepseek-v4-flash-gpu-bringup.md` (engine architecture).
Sources: vLLM ~v0.23.1rc (`~/vllm`), official HF artifacts (verified byte-level),
DFlash paper arXiv:2602.06036, speculators-format configs.

## Why this wins on our stack

dsv4 decode is expert-miss-H2D bound (~9 tok/s, step floor ~17). A verify step
runs k+1 tokens through each layer with the layer's selected experts fetched
once per unique expert — near the cost of one decode step for small k. Accepted
tokens are free. DFlash reports ~3.9 accepted/step on V4-Flash → a realistic
2.5–3.5× decode. Greedy acceptance is provably lossless when the verify chunk
path is bit-exact with the sequential step path (ours is, by construction).

## Weight artifacts (verified 2026-07-12)

| What | Where | Size | Notes |
|---|---|---|---|
| MTP module `mtp.0.*` (1575 tensors) | `deepseek-ai/DeepSeek-V4-Flash`, entirely in `model-00046-of-00046.safetensors` | 3.59 GB | FP8-e4m3 attention (ue8m0 128×128 block scales), fp4-packed experts (MXFP4-compatible, verify at load), BF16 gate |
| DFlash drafter | `RedHatAI/DeepSeek-V4-Flash-speculator.dflash` (`model.safetensors`) | 3.6 GB BF16 | ~2B params; val: 41.6% full-seq acceptance, ~3.9 accepted/step |
| DSpark (DeepSeek's own, follow-up option) | `deepseek-ai/DeepSeek-V4-Flash-DSpark` shards 46–48 (`mtp.0/1/2`) | ~10.9 GB | 3 chained V4-MoE blocks + rank-256 Markov head, block 5. Not in scope now. |
| GGUF with MTP | none exists | — | unsloth + teamblobfish GGUFs byte-verified MTP-less; upstream llama.cpp deepseek4 conversion structurally drops `mtp.*` |

Local layout: `~/.hi/models/deepseek-v4-flash/mtp/` (shard 46 + index + config),
`~/.hi/models/deepseek-v4-flash/dflash-redhat/`.

## Architecture facts

### V4-Flash MTP module (from shard-46 header + vLLM `models/deepseek_v4/nvidia/mtp.py`)
- One **full V4 decoder layer at compress_ratio = 1**: sliding-window (128,
  `attention.sliding_window`) latent-MQA attention with sinks (`attn_sink[64]`),
  **no compressor, no indexer**, full 256-expert MoE + shared expert,
  hyperconnections with its own hc coefficients. Matches our GGUF's vestigial
  `compress_ratios[43] = 0` slot.
- Inputs: `enorm(embed(token))` and `hnorm(prev_hidden)` combined **additively**
  via split projections: `h_proj(prev) + e_proj(emb)` broadcast across the
  hc_mult streams (V3's concat `eh_proj` does NOT apply). `prev_hidden` is the
  target's **flat pre-hc_head residual, (T, hc_mult·4096)** — the value our
  engine holds immediately before applying `output_hc_{fn,base,scale}` + final
  norm + grouped output head.
- Own `hc_head_{fn,base,scale}` + `norm`; embedding and LM head are genuinely
  shared with the target — shard 46 contains zero non-`mtp.` tensors (census
  2026-07-12), so we bind our token_embd/output weights.
- Recurrence for k>1: feed drafted token + the MTP block's own flat pre-hc_head
  residual back in. `num_nextn_predict_layers = 1` → the module is reused
  (quality decays; default K=1, allow K≤3).
- Tensor names (census-verified, 1575 tensors): `mtp.0.attn.{wq_a,wq_b,wkv,
  wo_a,wo_b}.weight` FP8-E4M3 + ue8m0 `.scale` siblings (128×128 block grid,
  scale is a MULTIPLIER), `mtp.0.attn.{q_norm,kv_norm}.weight` BF16,
  `mtp.0.attn.attn_sink` F32[64], `mtp.0.{e_proj,h_proj}.weight` FP8 [4096,
  4096], `mtp.0.{enorm,hnorm}.weight` BF16[4096],
  `mtp.0.ffn.experts.{0..255}.w{1,3}.weight` I8 fp4-packed [2048,2048]
  (logical [2048,4096]) + `.scale` F8_E8M0 [2048,128], `w2` [4096,1024]
  (logical [4096,2048]) + `.scale` [4096,64] — one e8m0 byte per 32 in-dim
  values, i.e. exactly MXFP4-32; `mtp.0.ffn.gate.weight` BF16 [256,4096] +
  `gate.bias` F32[256] (routing selection bias, values ~10);
  `mtp.0.ffn.shared_experts.w{1,2,3}.weight` FP8 (note PLURAL naming);
  `mtp.0.hc_{attn,ffn}_fn` F32 [24,16384], `_base` [24], `_scale` [3];
  `mtp.0.hc_head_fn` F32 [4,16384], `_base` [4], `_scale` [1];
  `mtp.0.{attn_norm,ffn_norm,norm}.weight` BF16[4096].
  Sizes: non-expert resident ≈ 335 MB f16; experts ≈ 3.42 GB as MXFP4
  (13,369,344 B/expert) — pool as layer 43 via
  `safetensors::fp4_to_gguf_mxfp4` (bit-exact repack, proven).

### DFlash drafter (RedHat config + vLLM `qwen3_dflash.py`, `spec_decode/dflash.py`)
- 5 llama-style decoder layers (census-verified): hidden 4096, 64 Q heads /
  **1 KV head**, head_dim 256 (`q_proj [16384,4096]`, `k_proj/v_proj
  [256,4096]`, `o_proj [4096,16384]`), per-head-dim `q_norm/k_norm [256]`
  (Qwen3 style), SwiGLU intermediate 2048, no biases, RoPE, **all-SWA window
  2048**, block_size 8 → K = 7 drafts/step, `mask_token_id = 1` (no dedicated
  mask_embedding tensor — the mask query embedding is `embed_tokens[1]`),
  `max_anchors 3072`. `embed_tokens [129280,4096]` (full target vocab in),
  `lm_head [32000,4096]` (reduced draft vocab out), final `norm [4096]`, and
  `hidden_norm [4096]` — the norm applied to target hiddens feeding the
  context-KV path. All BF16.
- Vocab map: `d2t` I64[32000] is an OFFSET map — `target_id = draft_id +
  d2t[draft_id]` (monotone, 32000 unique targets); `t2d` BOOL[129280] is the
  membership mask.
- Conditioning: **flat** — `fc.weight [4096, 81920]`, in = 5 aux layers ×
  (hc_mult 4 · 4096). The engine tap must supply per-layer flat hc-stream
  hiddens (16384/position/layer) at layers `[3,13,23,32,42]`, concatenated in
  layer order. fc consumes that raw concat (vLLM `combine_hidden_states`);
  `hidden_norm` applies only on the context-KV path
  (`precompute_and_store_context_kv`), never before fc.
- Context-KV mechanism: per proposal step, the *target's* hidden states for all
  newly verified tokens are normed and projected **by the draft's own per-layer
  KV weights**, RoPE'd, and appended to the draft's per-layer KV cache
  (incremental; rejected positions excluded then overwritten). The draft
  forward itself runs ONLY 1+K query tokens (anchor token + K mask embeddings
  placed at the predicted positions) in one pass — cost independent of context.
- Draft logits: own lm_head over 32000, argmax → map through d2t.

### Acceptance (vLLM `rejection_sampler.py` semantics)
- Greedy (our default, temp 0): accept draft[i] iff draft[i] == argmax(target
  logits at position i); on first mismatch emit the target argmax and stop; if
  all K accepted, also emit the bonus token from the last position. Output is
  bit-identical to sequential greedy decode — verified by construction + tests.
- Sampling: accept iff u < p_target(x)/q_draft(x) with q=1 for greedy drafting;
  on rejection sample from max(p−q, 0). Phase 2 (we serve greedy first; the
  backend rejects spec+sampling combos until then).
- Verify-side KV/state: append k+1 tokens to DsV4State as a chunk, then
  `truncate_state_to_at_most(accepted_prefix_end)` — machinery already exists
  (prefix cache uses it). Prefix-cache snapshots only at accepted boundaries.

## Status (2026-07-13)

All three stages are implemented and green (default suite 119, native 361).
- **Stage A DONE**: `verify_tokens`/`verify_tokens_with_taps` (per-position
  logits, bit-exact with sequential `host_step` on both providers),
  `rewind_state_to` (truncate to compressor boundary + exact re-feed — plain
  truncation cannot land mid-block), `DsV4Taps` (pre-hc-head + aux layers,
  flat and averaged), `Drafter` trait + oracle-tested lossless greedy loop
  (perfect/adversarial/mixed drafts all byte-identical to sequential),
  spec stats in the /health dsv4 segment.
- **Stage B DONE** (`dsv4_mtp.rs`): real-model acceptance **95.8% at K=1**
  (byte-identical output, reproduced twice), 42% overall at K=4 (2.61
  emitted/verify). Experts pinned in the pool as layer 43 (3.19 GiB MXFP4,
  bit-exact repack); 318 MiB dense resident. The (token, hidden) pairing rule
  is documented in the module and pinned by a test that rejects both
  off-by-one hypotheses. Ratio-1 layers use rope base 10000.
- **Stage C DONE** (`dsv4_dflash.rs`): causal=true SWA-2048 confirmed; 2.39
  GiB bf16 resident; ~13-17 ms per propose (K=7). Real-model acceptance
  **~1.04-1.14 accepted/step (16-18%; position-1 68.2%)** vs the checkpoint's
  published 3.9/step — consistent with the Q4_K_XL target's argmax diverging
  from the BF16 teacher, compounding with depth; the port itself is pinned by
  hand-computed tests. **Aux-layer correction (A/B-proven on the real
  model):** vLLM aux ids count "hidden after n layers" (idx+1 convention), so
  config `[3,13,23,32,42]` maps to post-layer tap indices `[2,12,22,31,41]`
  (16.0-18.0% vs 9.9-15.6% accepted with the unshifted taps).
- **Production matrix** (GPU 1 exclusive, 72 GiB pool, greedy 128-token
  generation, identical output text in every row):

  | variant | decode tok/s | acceptance | emitted/verify | ms/verify-step |
  |---|---|---|---|---|
  | baseline | 9.11 | — | 1.00 | 110 |
  | **mtp K=1** | **12.41 (+36%)** | 77.5% | 1.44 | 116 |
  | mtp K=2 | 8.30 | 57.5% | 1.73 | 208 |
  | mtp K=3 | 6.91 | 41.6% | 1.80 | 261 |
  | dflash K=7 | 4.65 | 24.1% | 2.10 | 451 |

  Verify-2 costs ~6% over a single step (the MoE-amortization premise holds),
  but each additional verify token adds ~55 ms on the exact host-GEMV chunk
  path — that marginal cost, not acceptance, is what buries K≥2 and DFlash.
  **Next lever: batched device verify** (the S>1 analog of the
  device-resident decode step) to cut the marginal verify token to ~10-20 ms;
  at that point DFlash's 2.1+ emitted/step and MTP K=2-3 become wins.
- **DEPLOYED (2026-07-13): `HI_DSV4_SPEC=mtp` on hi-local-v4.service**
  (drafter: 318 MiB dense resident + 3.19 GiB experts pool-pinned, k_cap 1).
  First deployment exposed that taps disabled prefix-cache restores (fresh
  sessions re-prefiled ~3 min); fixed by base-aware taps (`DsV4Taps` absolute
  base, accessors return None below it) + drafter cold-start at the restore
  point (MTP ring proven bit-convergent within its 128 window; DFlash context
  floors at base). Verified with spec on: cold 181 s, fresh session 2 s with
  reused=4608, drafter 3/3 accepted. Suites 122 default / 365 native.

## Stages

### Stage A — target-side verify + taps (no draft weights needed)
1. `DsV4Engine::verify_tokens(state, &[u32]) -> Vec<Vec<f32>>`: chunked forward
   returning logits at EVERY position (output head per position), bit-exact
   with sequential `host_step` (assert in tests over the fixture model).
   Route through the existing chunk machinery incl. the prefill expert pool;
   GPU path may batch, but bit-exactness with the step path is the contract
   (GEMV default; GEMM opt-in stays prefill-only for now).
2. Hidden taps (opt-in per request, zero-cost when off):
   - `PreHcHead`: flat (hc_mult·4096) residual per position (MTP input).
   - `AuxLayers([3,13,23,32,42])`: post-layer hc-stream hiddens per position,
     both flat and averaged layouts supported (DFlash input).
   Captured during prefill AND verify so the drafter's context stays complete.
3. `Drafter` trait in dsv4_backend: `propose(&mut self, ctx) -> Vec<u32>` +
   `observe_accepted(...)`; implementations: `None`, test `OracleDrafter`
   (replays a known continuation), later `MtpDrafter`/`DFlashDrafter`.
4. Generation loop: propose K → verify K+1 → greedy-accept prefix + bonus →
   truncate → stats (proposed/accepted/steps in /health dsv4 segment).
5. Tests: oracle-drafter output identical to sequential decode (perfect draft:
   all accepted; adversarial draft: all rejected; mixed); state-truncate
   round-trip across verify boundaries; prefix-cache interaction.

### Stage B — MTP (self-speculation, official weights)
1. `safetensors.rs` (hi-cuda): minimal mmap reader (header JSON + data
   offsets), dequant helpers: FP8-e4m3 + ue8m0 128×128 block scales → f16/f32;
   fp4-packed experts → our MXFP4 arena block layout (expect direct repack —
   same e2m1+e8m0/32 format as the GGUF trunk experts; verify scale-tensor
   shapes at load and requantize only if grouping differs); BF16 → f16.
2. `dsv4_mtp.rs`: MTP module forward on DsV4Engine's layer primitives with a
   compress_ratio-1 attention variant (128-window latent MQA + sinks, no
   compressor/indexer, its own tiny KV ring). Non-expert weights resident f16
   (~350 MB); experts registered in the existing DsV4ExpertPool as layer 43.
3. CPU reference first (fixture + real-shard spot check), then GPU provider,
   CPU==GPU parity gate as always.
4. `MtpDrafter`: K=1 default (`HI_DSV4_SPEC=mtp`, `HI_DSV4_SPEC_K`).
   Env: `HI_DSV4_MTP_PATH` (default `<model_dir>/mtp/model-00046-of-00046.safetensors`).

### Stage C — DFlash (external drafter)
1. `dsv4_dflash.rs`: the 5-layer drafter (dense f16 resident, ~3.6 GB — shave
   the expert pool budget accordingly), fc combiner, mask-token queries,
   context-KV precompute from target taps, SWA-2048 attention, d2t mapping.
2. `DFlashDrafter`: block 8 → K=7 (`HI_DSV4_SPEC=dflash`,
   `HI_DSV4_DFLASH_PATH`). One draft forward per step regardless of K.
3. CPU reference + GPU, parity, then production measurement.

### Order: A → B and the Stage-B loader (safetensors.rs) in parallel with A → C.
DSpark is the documented follow-up (best reported quality; 3 MoE draft blocks).

## Loader (DONE 2026-07-12)
`crates/hi-cuda/src/safetensors.rs`: mmap reader with full header validation
(`SafetensorsFile::open/tensors/bytes/tensor_f32/f16/tensor_i64`), scalar +
bulk dequant (BF16, F16, FP8-E4M3 with ue8m0 block scales via
`fp8_block_scaled_f32/f16`, fp4 groups), and `fp4_to_gguf_mxfp4` /
`repack_fp4_to_gguf_mxfp4` producing the exact GGUF MXFP4 rank-3 packed
layout the expert pool consumes (bit-exact, cross-checked through
hi_gguf::dequantize_tensor_as_f32). `.scale` siblings auto-resolved; scale is
a multiplier. serde_json + memmap2 are now real hi-cuda deps. 97/97 crate
tests green including real-artifact censuses.

## Risks / open questions
- ~~fp4 expert packing~~ RESOLVED: exactly MXFP4-32, direct bit-exact repack.
- ~~DFlash fc width~~ RESOLVED: flat, [4096, 81920].
- Verify-chunk cost: on a cold/small pool an 8-token verify costs ~4.5-5× a
  single step (miss-bound), so DFlash K=7 needs either a warm 72 GiB pool,
  batched verify kernels, or dynamic-K capping to clear break-even. MTP K=1's
  2-token verify is far below that threshold.
- Quantized-target acceptance gap: drafters trained against the BF16 teacher
  lose acceptance against a Q4 target's argmax (DFlash p1 68.2% vs 78.8%
  val). Self-speculation (MTP, drafted from the same quantized weights'
  hidden state) is far less affected — 95.8%.
- Verify-chunk expert traffic: k+1 tokens route to up to 6(k+1) distinct
  experts/layer; prefill-pool dedup keeps this ~1 fetch set per layer. Measure;
  if misses dominate, cap K dynamically (vLLM-style per-batch K schedule).
- MTP hc-stream init for the draft layer's hyperconnections must match vLLM's
  (broadcast combine across streams; mhc_post collapse before hc_head) — port
  exactly from `deepseek_v4/nvidia/mtp.py` and verify against real-shard logits.
- hi's client uses temperature 0 today (greedy) — sampling+spec deferred.

## Acceptance criteria ("fully working")
1. Oracle tests prove lossless verify/rollback (Stage A) — CPU and GPU.
2. MTP: `hi -p v4-flash` output byte-identical with `HI_DSV4_SPEC=mtp` vs off;
   accepted-rate reported; decode tok/s measured and improved.
3. DFlash: same, `HI_DSV4_SPEC=dflash`; target ≥2× decode on real `hi` turns.
4. Suites green (`cargo test -p hi-cuda` native + workspace), service runs with
   spec enabled for a full `hi` tool-mode session, /health exposes spec stats.
