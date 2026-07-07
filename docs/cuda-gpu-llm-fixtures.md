# CUDA GPU LLM Fixture Guide

Large GGUF files are intentionally kept outside git. Use this guide to populate a local fixture directory for native CUDA smoke runs.

The default path is the checked-in fixture manifest plus downloader:

```sh
HI_CUDA_FIXTURES_DIR=/models/hi-cuda docs/fetch-cuda-fixtures.sh
```

Set `HI_CUDA_FIXTURE_MANIFEST=<path>` to use a local or private manifest with the same tab-separated columns: relative path, URL, SHA-256, expected family, architecture, and quant type.

## Directory Layout

Set `HI_CUDA_FIXTURES_DIR` to a local directory and place models under family-specific names:

```text
$HI_CUDA_FIXTURES_DIR/
  llama3-instruct/model.gguf
  mistral/model.gguf
  gemma/model.gguf
  phi/model.gguf
  mixtral/model.gguf
  deepseek-dense/model.gguf
  glm-dense/model.gguf
  qwen25-vl/model.gguf
  qwen25-vl/mmproj.gguf
  tinyllama/model.gguf
  quant/
    Q4_0/model.gguf
    Q4_1/model.gguf
    Q5_0/model.gguf
    Q5_1/model.gguf
    Q4_K/model.gguf
    Q2_K/model.gguf
    Q3_K/model.gguf
    Q5_K/model.gguf
    Q6_K/model.gguf
    Q8_0/model.gguf
    Q8_1/model.gguf
    Q8_K/model.gguf
    Q1_0/model.gguf
    IQ2_XXS/model.gguf
    IQ2_XS/model.gguf
    IQ3_XXS/model.gguf
    IQ1_S/model.gguf
    IQ2_S/model.gguf
    IQ3_S/model.gguf
    IQ4_NL/model.gguf
    IQ4_XS/model.gguf
    IQ1_M/model.gguf
    MXFP4/model.gguf
    NVFP4/model.gguf
    TQ1_0/model.gguf
    TQ2_0/model.gguf
    Q4_0_4_4/model.gguf
    Q4_0_4_8/model.gguf
    Q4_0_8_8/model.gguf
    IQ4_NL_4_4/model.gguf
    IQ4_NL_4_8/model.gguf
    IQ4_NL_8_8/model.gguf
```

## Representative Sources

Use real GGUFs that match the directory family, model architecture, and quant type under test. `docs/cuda-fixtures.tsv` provides the public default manifest. The smoke suite can also use:

- Llama 3.x instruct GGUF for text, streaming, sampling, cancellation, and dense scheduler batching coverage.
- Mistral dense GGUF for sampled dense scheduler batching coverage.
- Gemma dense GGUF.
- Phi split-Q/K/V bias GGUF.
- Mixtral MoE GGUF for text MoE continuous scheduler batching coverage.
- DeepSeek dense split-attention GGUF.
- GLM dense split-attention GGUF.
- Qwen2.5-VL language GGUF plus matching `mmproj.gguf` for image/video requests.

Synthetic loader fixtures cover every tensor type hi-local accepts: dense numeric tensors, classic Q-quants, K-quants, IQ quants, TQ quants, MXFP4, NVFP4, and the specialized raw GGUF ids `Q4_0_4_4`, `Q4_0_4_8`, `Q4_0_8_8`, `IQ4_NL_4_4`, `IQ4_NL_4_8`, and `IQ4_NL_8_8`. Native CUDA parity tests compare every supported quantized GPU dequantizer against the CPU GGUF dequantizer; the specialized variants are treated as layout aliases of `Q4_0` and `IQ4_NL`.

Dense CUDA matrix fixtures may use FP16, BF16, or F32 storage. Native tests cover F32 matrix loading, F32 token embedding gather, F32 projection parity against the CPU GGUF reference, and matching-dtype cuBLAS projection GEMMs for FP16/BF16 matrix weights.

Health exposes the current GPU feature split as `gpu-features=...`: dense and MoE text requests use continuous paged KV when `--kv-cache-mode paged` is active, recurrent SSM models report `continuous_kv_backend=persistent-recurrent-ssm`, compatible Qwen-VL prompt embeddings use single-request paged KV or batched scheduler-owned page leases, and wide attention reports `wide_kernel=tiled-wide,wide_head_dim_max=512` through `attention-detail` while larger heads keep explicit generic fallback reasons.

## Fixture Environment Variables

The downloader writes the directory layout above under `HI_CUDA_FIXTURES_DIR`. The test harness discovers fixtures from `HI_CUDA_FIXTURES_DIR` first, but every family can also be supplied explicitly:

| Family | Environment variable | Default relative candidates |
| --- | --- | --- |
| Llama 3.x instruct | `HI_CUDA_SMOKE_LLAMA3_GGUF` | `llama3-instruct/model.gguf`, `llama-3-instruct/model.gguf`, `llama-3.1-instruct/model.gguf`, `llama-3.2-instruct/model.gguf` |
| Mistral dense | `HI_CUDA_SMOKE_MISTRAL_GGUF` | `mistral/model.gguf`, `mistral-7b-instruct/model.gguf` |
| Gemma dense | `HI_CUDA_SMOKE_GEMMA_GGUF` | `gemma/model.gguf`, `gemma-instruct/model.gguf` |
| Phi | `HI_CUDA_SMOKE_PHI_GGUF` | `phi/model.gguf`, `phi-3-mini-instruct/model.gguf` |
| Mixtral MoE | `HI_CUDA_SMOKE_MIXTRAL_GGUF` | `mixtral/model.gguf`, `mixtral-instruct/model.gguf` |
| DeepSeek dense | `HI_CUDA_SMOKE_DEEPSEEK_DENSE_GGUF` | `deepseek-dense/model.gguf`, `deepseek/model.gguf` |
| GLM dense | `HI_CUDA_SMOKE_GLM_DENSE_GGUF` | `glm-dense/model.gguf`, `glm4/model.gguf` |
| Default text smoke | `HI_CUDA_SMOKE_TEXT_GGUF` | `llama3-instruct/model.gguf`, `tinyllama/model.gguf`, TinyLlama chat GGUF path |
| Qwen2.5-VL language | `HI_CUDA_SMOKE_QWEN25_VL_GGUF` | `qwen25-vl/model.gguf`, `qwen2.5-vl/model.gguf`, Qwen2.5-VL 3B GGUF path |
| Qwen2.5-VL projector | `HI_CUDA_SMOKE_QWEN25_VL_MMPROJ` | `qwen25-vl/mmproj.gguf`, `qwen2.5-vl/mmproj.gguf`, Qwen2.5-VL 3B mmproj path |

Set `HI_CUDA_REQUIRE_REAL_FIXTURE_MATRIX=1` when you want missing family fixtures to fail the metadata smoke and, when `HI_CUDA_SMOKE_FAMILY_MATRIX=1` is also set, the family HTTP smoke instead of being reported as skipped. Set `HI_CUDA_SMOKE_FAMILY_MATRIX=1` to run one greedy HTTP completion against every discovered family fixture and assert each one is serving with GPU execution, paged KV, continuous scheduling, and a CUDA attention backend; this can be slow for larger Mixtral and DeepSeek models. Set `HI_CUDA_SMOKE_TEXT_STRESS=1` to run the concurrent greedy HTTP smoke against the default text fixture. Add `HI_CUDA_SMOKE_TEXT_SAMPLED_STRESS=1` when you also want a seeded sampled real-model probe. Streaming cancellation is covered by native scheduler and server stream-drop tests rather than by the real HTTP smoke, where small SSE responses can be fully buffered before disconnect propagation.

## Smoke Scenarios

Run native CUDA tests against the fixture directory from a GPU machine:

```sh
HI_CUDA_FIXTURES_DIR=/models/hi-cuda cargo test -p hi-cuda --features native-cuda
HI_CUDA_FIXTURES_DIR=/models/hi-cuda cargo test -p hi-local --features native-cuda
```

The `hi-local` smoke harness also accepts explicit paths:

```bash
HI_CUDA_SMOKE_TEXT_GGUF=/models/tinyllama/model.gguf \
HI_CUDA_SMOKE_QWEN25_VL_GGUF=/models/qwen25-vl/model.gguf \
HI_CUDA_SMOKE_QWEN25_VL_MMPROJ=/models/qwen25-vl/mmproj.gguf \
cargo test -p hi-local --features native-cuda --test cuda_real_smoke
```

By default, the real text smoke runs health and one greedy HTTP completion. A separate default text smoke starts a one-page KV cache and asserts page-exhaustion rejection returns structured `insufficient_gpu_memory`. The Qwen2.5-VL smoke covers both image and frame-list video requests. The opt-in text stress smoke covers seeded sampled concurrent requests and streaming cancellation.
