# DeepSeek-V4-Flash GPU bring-up plan (hi-cuda)

Companion to `deepseek-v4-flash-port-spec.md` (the exact math). This file fixes
the *engineering shape* of the CUDA port. Bring-up optimizes for correctness and
"serves tokens end-to-end", not throughput; each stage has a faster follow-up.

## Stage 0 — loader acceptance (hi-gguf)
- `reject_unsupported_mla_layout` must return Ok for `config.is_deepseek4()`
  (V4 owns its own validation; it has q_lora metadata but V2-style checks don't apply).
- V4 layer validation (dispatch before MLA/dense in the per-layer validator):
  require the census tensor set (see spec §GGUF tensor names) with shapes derived
  from config: q_lora=1024, head_dim=key_length=512, rope_tail=rope.dimension_count,
  heads=64, groups/rank from output_group_count/lora_rank, hc_mult from
  hyper_connection.count (fn [hc*embed, hc²+2hc], base [hc²+2hc], scale [3]),
  per-layer compressor (ratio>0) / indexer (ratio==4) / tid2eid (layer<hash_count)
  / exp_probs_b (layer>=hash_count) presence rules; packed rank-3 experts
  [embed, expert_ff, expert_count] and [expert_ff, embed, expert_count].

## Stage 1 — GPU engine, host-orchestrated (the bring-up design)
New module `crates/hi-cuda/src/dsv4_gpu.rs`, mirroring `dsv4_cpu.rs`'s structure
1:1 (same per-layer state machine, same one-token-at-a-time semantics). The ONLY
difference: every large matmul runs on the GPU.

- Hidden state lives on the HOST as f32 (T=1: hc_mult×4096 floats = 64KB). All
  hyperconnection/sinkhorn/rope/sink/compressor-gate/indexer math stays host-side
  f32, copied verbatim from dsv4_cpu (it is tiny per token).
- Matmul primitive: `gpu_mul_vec(matrix_name, &[f32]) -> Vec<f32>` — upload the
  input vector, run the existing project machinery (`project_f32_device` handles
  Q8_0/BF16/F16/F32 and has a fused MXFP4 GEMV for M=1), download the output.
  Per token this moves ~40MB total — negligible.
- Resident GPU weights (~8-10GB): every non-expert tensor (attention mats,
  shexp, router, embeddings/head, compressor/indexer mats). Norms/hc/sinks/ape/
  tid2eid/exp_probs_b stay HOST-side (small F32/I32 vectors).
- Attention proper (q·K over ≤128 raw + compressed blocks) computes on host at
  bring-up: per token per layer it is 64 heads × 512 dims × ~a few hundred keys.
  Follow-up: move to the existing attention kernels once correct.

### Experts (the 140GB problem)
Stage 1a (correctness): experts stay in the mmap'd GGUF on host. Per token per
layer, for the 6 selected experts: slice the packed MXFP4 bytes for expert e
(contiguous: stride = in*ff/32*17 bytes), upload the packed slice to a scratch
DeviceBuffer, run `launch_mxfp4_gemv` (M=1) against it directly. ~3.4GB/token
H2D worst case ≈ 5-10 tok/min. Correct, slow.
Stage 1b (usable): GPU LRU expert pool (~75GB VRAM ≈ 55% of all expert slices
resident; reuse the expert_streaming pointer-table pattern but keyed per
(layer, proj, expert) with MXFP4 slices + per-slice fused GEMV calls). Target
>90% hit rate on real text → several tok/s.

## Stage 2 — serving integration
- `CudaBackend::load`: if `config.is_deepseek4()`, construct a dedicated
  single-request V4 engine implementing `InferenceBackend` (stream_generate =
  sequential greedy/sampled decode loop over dsv4_gpu). Do NOT wire the
  continuous scheduler/paged-KV machinery at bring-up; V4's ring+compressed
  cache is engine-internal. max-batch-size is effectively 1.
- `/health` reports family deepseek, execution gpu; `/v1/models` context 1M
  (advertise less, e.g. 32768, until long-context is validated).
- Chat template: GGUF embeds a 13k jinja (thinking modes). Add a dedicated
  `render_deepseek_v4_template` (detect via template containing "thinking" +
  V4 markers — inspect the saved copy in scratchpad/v4_template.jinja; verify
  against tokenizer vocab special tokens at load). Thinking OFF by default.
- Tokenizer: gpt2 BPE, pre `joyai-llm`, add_bos FALSE, bos 0 eos 1. hi-gguf's
  byte-level BPE (no pretokenizer split) matched deepseek-llm exactly on probes;
  verify with a couple of encode probes against known-good token ids.

## Stage 3 — acceptance
- Tiny synthetic deepseek4 fixture: CPU vs GPU parity (same logits ±1e-3).
- Real model: CPU-oracle first token vs GPU first token IDENTICAL (greedy),
  then 32-64 token coherence probes ("capital of France", small code task).
- Only then: matrix entry (--large tier) + hi.toml profile.

## Verification discipline
At every stage the CPU reference (dsv4_cpu) is the oracle: any GPU/CPU
divergence is a GPU bug by definition. Never debug coherence by eyeballing
GPU output alone (V2-Lite lesson).
