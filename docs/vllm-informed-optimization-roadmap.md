# vLLM-informed optimization roadmap for hi-cuda

Distilled from a review of ~/vllm (v0.23.1rc-class, which ships first-class
DeepSeek-V4 support in `vllm/models/deepseek_v4/` — its design independently
matches our dsv4 port: latent MLA + sinks, compress ratios {4,128}, identical
mHC/Sinkhorn math in `model_executor/kernels/mhc/torch.py`). File references
below are into ~/vllm. Our measured bottlenecks: (A) decode floor ~130ms/token
from ~18 host-synchronous GEMV launches/layer with host-resident hidden state;
(B) prefill 20–32 tok/s with host(rayon) attention; (C) single-snapshot prefix
reuse; (D) long-context YARN semantics.

Ranked (impact × effort):

1. **Stacked experts + one grouped GEMM + device routing (A — biggest win).**
   vLLM runs an entire MoE layer in ~5–7 launches regardless of expert count:
   device sort/pad `moe_align_block_size` (csrc moe_align_sum_kernels.cu) →
   ONE grouped GEMM over stacked `(E,N,K)` weights indexing `expert_ids[pid_m]`
   (`fused_moe/fused_moe.py:294,1592`) → activation → second grouped GEMM →
   top-k reduce. Routing (`fused_moe/router/grouped_topk_router.py`) emits
   device topk_ids — zero host sync. Port: make our LRU pool one contiguous
   `(E_resident, N, K/2)` MXFP4 buffer + device expert_ids indexing pool slots.
   V4 ceiling: 2-kernel MegaMoE (`deepseek_v4/nvidia/model.py:431`).

2. **Device-resident decode step + CUDA graphs (A).** Persistent fixed-address
   buffers for ids/positions/seq_lens/block_table (`gpu_model_runner.py:755+`,
   `CpuGpuBuffer`), sampled token scattered device-side into next input_ids
   (`:1771,:1855`), ONE deferred host sync per step (`AsyncGPUModelRunnerOutput`,
   `:251`), full-step graph capture keyed by batch size
   (`compilation/cuda_graph.py:233`). Requires moving our hidden state, rope,
   rmsnorm, sinkhorn, routing to device kernels.

3. **GPU MLA attention, both formulations (B).** `layers/attention/mla_attention.py:40-188`
   is a spec: absorbed latent-MQA decode (bmm q_nope·W_UK_T → MQA over latent →
   W_UV up-proj) + compute-friendly chunked prefill with `merge_attn_states`
   LSE merges. Replaces our host-rayon prefill attention.

4. **MXFP4 dequant-in-kernel grouped GEMM (A/B).** Marlin lop3/prmt in-register
   nibble unpack (`csrc/.../moe_wna16_utils.h:82-100`) or Cutlass mx_float4_t
   grouped GEMM with per-expert offset precompute
   (`mxfp4_blockwise_moe_kernel.cu:36`); E8M0 scale per 32 elements — matches
   our packed layout exactly.

5. **Copy-stream expert prefetch (A).** `offloader/prefetch.py:127`: static VRAM
   buffers (CUDA-graph safe) + dedicated copy stream prefetching the routed
   experts one layer ahead; sync before the grouped GEMM. Drop-in upgrade for
   our LRU pool misses.

6. **Block-hash prefix cache (C).** Rolling Merkle chain
   `hash(parent_hash, block_token_ids, extra)` at KV-block granularity
   (`v1/core/kv_cache_utils.py:596`) + `HashMap<hash,block>` + LRU free list
   (`block_pool.py`). Replaces our single conversation snapshot with
   multi-conversation sharing + eviction.

7. **V4 fused-kernel specials.** Fuse mHC post+norm+pre into one kernel
   (`deepseek_v4/nvidia/model.py:866,894`; math in `kernels/mhc/torch.py` —
   matches our host implementation), MegaMoE, CuteDSL radix top-k indexer
   (`sparse_attn_indexer.py`), fused compress+rmsnorm+rope+store
   (`deepseek_v4/compressor.py`), fp8_ds_mla (UE8M0) KV cache.

## Correctness note: deepseek2 YARN (D)

vLLM/HF semantics for deepseek_yarn (`models/deepseek_v2.py:428-433,547-554`;
`rotary_embedding/deepseek_scaling_rope.py:56-60`):
- softmax scale = `(qk_nope+qk_rope)^-0.5 · mscale²`, mscale = `0.1·mscale_all_dim·ln(factor)+1`
  → **0.11473 for V2-Lite** (192^-0.5 · 1.5897)
- rope cos/sin scale = ratio of the two mscales = **1.0 for V2-Lite** (table unscaled)
- yarn NTK-by-parts frequency interpolation ON.

Our current V2 config (post dp4a fix) is plain rope + plain scale, which matches
our CPU reference and is empirically correct at short context. The earlier
yarn/mscale sweeps that "failed" ran on top of the (now fixed) int8-dp4a
corruption, so they don't condemn the vLLM package. Follow-up: enable
yarn-freqs + mscale² TOGETHER (as one package) on both CPU ref and GPU, verify
short-context parity holds, and gate long-context (>4k) correctness on it.
