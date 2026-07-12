# DeepSeek-V4-Flash (`deepseek4`) port spec — MLX reference → hi-cuda

Extracted from the working MLX implementation in `crates/hi-mlx/src/models.rs`
(V4 code ~lines 2336-3346; `v4_rope` 8090; masks 8122-8163; `score_v4` 7752;
`forward_expert_limited` 1785). GGUF ground truth from
`unsloth/DeepSeek-V4-Flash-GGUF` (UD-Q4_K_XL). Verified 2026-07-12.

## Real GGUF config (metadata, `deepseek4.` prefix)

43 layers, embed 4096, heads 64, kv_heads 1, vocab 129280, ctx 1,048,576.
`attention.key_length = attention.value_length = 512` (= head_dim, shared KV
latent width), `rope.dimension_count = 64` (rope tail; nope = 448),
`attention.q_lora_rank = 1024`, NO kv_lora_rank key.
`rope.freq_base = 10000`, `attention.compress_rope_freq_base = 160000`,
YARN keys present (factor 16, orig 65536, betas 32/1) but **MLX ignores YARN
for V4 — port starts rope-plain, matching the working reference**.
`attention.sliding_window = 128`. `expert_count = 256`, `expert_used_count = 6`,
`expert_gating_func = 4` (sqrt-softplus), `expert_weights_norm = true`,
`expert_weights_scale = 1.5`, `expert_shared_count = 1`,
`expert_feed_forward_length = 2048`, `hash_layer_count = 3`,
`swiglu_clamp_exp` / `swiglu_clamp_shexp` = per-layer f32 arrays (all 10.0),
`attention.indexer.head_count = 64`, `.key_length = 128`, `.top_k = 512`,
`attention.output_group_count = 8`, `attention.output_lora_rank = 1024`,
`attention.compress_ratios` = per-layer u32 (44 entries = 43 layers + stripped
MTP slot): `[0,0,4,128,4,128,...,4,0]` — ratio 4 on even layers 2..42
(compressor + indexer), 128 on odd layers 3..41 (compressor only), 0 on 0,1.
`hyper_connection.count = 4`, `.sinkhorn_iterations = 20`, `.epsilon ≈ 1e-7`
(hc_eps; MLX default 1e-6 — use the GGUF value).
Tokenizer: gpt2 BPE, pre `joyai-llm`, bos 0, eos 1, add_bos FALSE; chat
template embedded (13k chars, thinking-mode switches).
No MTP/NextN tensors in the GGUF (stripped by unsloth) — do not implement.

## GGUF tensor names (census; MLX name → GGUF name)

Per layer `blk.N.` (GGUF dims are [ne0=in, ne1=out]):
- attn.wq_a → `attn_q_a.weight` [4096,1024] · attn.q_norm → `attn_q_a_norm.weight` [1024]
- attn.wq_b → `attn_q_b.weight` [1024,32768] (64 heads × 512)
- attn.wkv → `attn_kv.weight` [4096,512] · attn.kv_norm → `attn_kv_a_norm.weight` [512]
- attn.attn_sink → `attn_sinks.weight` [64] (per head)
- wo_a → `attn_output_a.weight` [4096,8192] (block-diagonal: 8 groups, each 4096→1024; rows g*1024..(g+1)*1024 serve input slice g)
- wo_b → `attn_output_b.weight` [8192,4096]
- compressor (41 layers, ratio>0): `attn_compressor_gate.weight` [4096,1024], `attn_compressor_kv.weight` [4096,1024] (1024 = 2*head_dim... NO: 2*512=1024 ✓), `attn_compressor_ape.weight` [1024, ratio] (ratio 4 or 128 per layer!), `attn_compressor_norm.weight` [512]
- indexer (even layers 2..42): `indexer.attn_q_b.weight` [1024,8192] (64 idx heads × 128, input = q latent), `indexer.proj.weight` [4096,64] (weights_proj), `indexer_compressor_gate.weight` [4096,256], `indexer_compressor_kv.weight` [4096,256] (2*128), `indexer_compressor_ape.weight` [256, ratio(4)], `indexer_compressor_norm.weight` [128]
- MoE: `ffn_gate_inp.weight` [4096,256] BF16 router; `exp_probs_b.bias` [256] (layers 3..42 only — selection bias); `ffn_gate_tid2eid.weight` [6,129280] I32 (layers 0-2 only; element (i, token) at flat token*6+i); experts packed rank-3 `ffn_{gate,up,down}_exps.weight` [4096,2048,256]/[2048,4096,256] MXFP4; shared `ffn_{gate,up,down}_shexp.weight` [4096,2048]/[2048,4096] Q8_0
- hc: `hc_attn_fn.weight` [16384,24], `hc_attn_base.weight` [24], `hc_attn_scale.weight` [3]; same for `hc_ffn_*`  (24 = hc² + 2·hc with hc=4; 16384 = hc·embed)
- norms: `attn_norm.weight`, `ffn_norm.weight` [4096]

Global: `token_embd.weight`/`output.weight` [4096,129280] Q8_0, `output_norm.weight` [4096], `output_hc_fn.weight` [16384,4], `output_hc_base.weight` [4], `output_hc_scale.weight` [1].

## Forward pass (exact)

Everything below is the MLX reference behavior; f32 for HC math, softmax in f32.

### Model
```
h = embed(ids)                              # [T, D]
H = broadcast h to hc=4 streams             # [T, 4, D]
for layer in 0..43: H = block(H, ids)
h = hyper_head(H)                           # collapse 4→1 (like hc.pre with scale[0]/base only)
h = rms_norm(h, output_norm)
logits = output(h)
```

### Block (hyperconnection wrapper; pre-norm INSIDE the collapsed stream)
```
res = H;  (y, post, comb) = hc_attn.pre(H)
y = attention(rms_norm(y, attn_norm))
H = hc_attn.post(y, res, post, comb)
res = H;  (y, post, comb) = hc_ffn.pre(H)
y = moe(rms_norm(y, ffn_norm), ids)
H = hc_ffn.post(y, res, post, comb)
```

### HyperConnection.pre(x[T,4,D]) → (y[T,D], post[T,4], comb[T,4,4])
```
xf = x.reshape(T, 4D); inv = rsqrt(mean(xf², -1) + rms_eps)
mixes = (xf @ fn) * inv                      # [T, 24]  (fn GGUF [4D,24])
pre_log  = mixes[:, :4]  * scale[0] + base[:4]
post_log = mixes[:, 4:8] * scale[1] + base[4:8]
comb_log = mixes[:, 8:24].reshape(T,4,4) * scale[2] + base[8:24].reshape(4,4)
pre  = sigmoid(pre_log) + hc_eps
post = sigmoid(post_log) * 2.0
comb = softmax(comb_log, -1) + hc_eps
comb /= sum(comb, axis=1(rows), keepdim) + hc_eps          # 1 row-normalize
repeat (sinkhorn_iters - 1) times:                          # 19 more pairs
    comb /= sum(comb, axis=2(cols), keepdim) + hc_eps
    comb /= sum(comb, axis=1(rows), keepdim) + hc_eps
y = Σ_s pre[:,s] * x[:,s,:]                  # weighted stream sum → [T,D]
```
NOTE axis convention: comb[T, i, j]; "axis 1" = i (output-stream index), "axis 2" = j.

### HyperConnection.post(f[T,D], res[T,4,D], post, comb)
```
H_new[t,i,:] = post[t,i] * f[t,:] + Σ_j comb[t,i,j] * res[t,j,:]
```

### HyperHead (output_hc): like pre but only the first 4 coefficients:
```
mixes = (xf @ fn) * inv                      # [T,4]
pre = sigmoid(mixes * scale[0] + base) + hc_eps
y = Σ_s pre[:,s] * x[:,s,:]
```

### V4 Attention (per layer; rope_base = 10000 if ratio==0 else 160000)
```
qr = rms_norm(wq_a(x), q_a_norm)             # [T,1024] q latent
q  = wq_b(qr) → [T, 64, 512]
q  = q * rsqrt(mean(q², per-head-512) + rms_eps)      # UNWEIGHTED per-head RMS
kv = rms_norm(wkv(x), kv_a_norm)             # [T,512] single shared latent; K = V = kv
rope tail (last 64 dims of each 512), INTERLEAVED pairs (x[2i],x[2i+1]), angle = pos / base^(2i/64):
    q[..., 448:512] rotated;  kv[..., 448:512] rotated
raw cache: ring of last 128 positions (sliding window) — RAW k=v=kv per position
raw_mask[q,k] = key_pos <= q_pos AND key_pos >= q_pos-127
if compressor: (ck, cv) = compressor.update(x)         # block-compressed long-range KV
    cmask[q,b] = (b+1)*ratio - 1 <= q_pos              # block fully in past
    if indexer and compressed_len > 512: top-512 block selection (see below)
    K = concat(ck, raw_k); V = concat(cv, raw_v); mask = concat(cmask, raw_mask)
scores = q @ K^T * (512)^-0.5                # scale = head_dim^-0.5 (full 512)
per head: softmax over [keys, sink]: denom += exp(sinks[head]); sink adds no value
out = attn @ V                                # [T,64,512]
out[..., 448:512] = inverse-rope (negated angle) at the QUERY's position
out = wo_b(grouped_wo_a(out.flatten()))       # 32768 →(8×[4096→1024])→ 8192 → 4096
```

### Compressor (per layer with ratio r ∈ {4,128})
```
Only COMPLETE blocks of r tokens are compressed (pending remainder buffered).
gate = wgate(x_block) + ape                  # [r, 1024] + ape[1024, r]ᵀ  (learned additive pos bias)
w = softmax(gate over the r positions, per channel)
kvc = wkv(x_block)                            # [r, 1024]
comp = Σ_r w * kvc                            # [1024] gated block average
k_c = rms_norm(comp[:512], compressor_norm); v_c = rms_norm(comp[512:], compressor_norm)
→ appended to compressed cache (block index b). NO rope on compressed keys.
```

### Indexer (ratio-4 layers; NO rope anywhere in it)
```
(ick, _) = indexer_compressor.update(x)       # own APE compressor, key dim 128, ratio 4
if compressed_len <= 512: attend to all blocks (no selection)
qi = indexer.attn_q_b(qr) → [T, 64, 128]      # from the q latent
s = relu(qi @ ick^T) * 128^-0.5               # [64, T, blocks]
w = indexer.proj(x) * 64^-0.5                 # [T,64] head weights, no activation
score[t,b] = Σ_h w[t,h] * s[h,t,b];  mask block-causal
keep top-512 blocks per query (argpartition; unsorted)
prefill: AND a scatter mask into cmask;  decode: gather the selected blocks
```

### MoE (all 43 layers; batch=1 semantics)
```
logits = router(x)                            # BF16 [256]
scores = sqrt(softplus(logits)) = sqrt(ln(1+e^x))     # gating_func 4, per expert
if layer < 3:  selected = tid2eid[token_id*6 .. token_id*6+6]      # hash routing
else:          selected = top-6 of (scores + exp_probs_b), ties → lower index
weights = scores[selected]                     # bias-free!
if norm (true) and >1 selected: weights /= Σ weights   (skip if scoring==softmax)
weights *= 1.5                                 # expert_weights_scale, AFTER norm
acc = Σ_e weights[e] * expert_e(x)  with swiglu clamp:
    g = gate_proj_e(x);  u = up_proj_e(x)
    g = min(g, 10.0)                           # ceiling only
    u = clamp(u, -10.0, 10.0)                  # both sides
    expert_out = down_proj_e( silu(g) * u )    # silu(g) = g*sigmoid(g)
y = acc + shared_expert(x)                     # plain SwiGLU, NO clamp
```

## Risk notes
- Indexer being rope-free diverges from V3.2's MlaIndexer — flagged by the
  extraction as the highest-risk item; validate against real-model output.
- MLX ignores the GGUF's YARN keys for V4; if long-context output degrades,
  revisit (short-context correctness does not need YARN).
- hc_eps: GGUF `hyper_connection.epsilon` ≈ 1e-7 vs MLX default 1e-6 — use GGUF.
- exp_probs_b missing on hash layers (0-2) — selection there never uses it.
