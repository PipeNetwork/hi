# Optional local runtime

The GPU and GGUF implementation is maintained in the separate
[`PipeNetwork/hi-local-runtime`](https://github.com/PipeNetwork/hi-local-runtime)
repository. The core `hi` workspace does not compile or link CUDA, MLX, GGUF,
or native vendor code.

Install the matching `hi-local` sidecar from that repository or a release
bundle. Core discovers it in this order:

1. `HI_LOCAL_BIN`
2. `hi-local` beside the `hi` executable
3. `hi-local` on `PATH`

To install a release bundle beside the installed `hi` binary, provide its
published SHA-256 digest; the installer rejects unsigned, wrong-platform,
wrong-backend, and unsupported-protocol bundles:

```bash
HI_LOCAL_RUNTIME_ARCHIVE=https://github.com/PipeNetwork/hi-local-runtime/releases/download/v0.1.0/hi-local-mlx-macos-arm64.tar.gz \
HI_LOCAL_RUNTIME_SHA256=<published-sha256> \
scripts/install_hi_local_runtime.sh mlx
```

`HI_LOCAL_BIN` remains the explicit override for development and custom
installations.

The sidecar must answer `hi-local --version` promptly and expose the versioned
OpenAI-compatible API. Core accepts protocol `1.x`, validates `/health`
readiness and the requested backend, then verifies `/v1/models` and chat/tool
compatibility before switching providers. Missing binaries, incompatible
protocols, wrong backends, crashes, and startup timeouts return actionable
errors; core never starts a Cargo build for the sidecar.

See the runtime repository’s [contract](https://github.com/PipeNetwork/hi-local-runtime/blob/main/docs/runtime-contract.md),
[release procedure](https://github.com/PipeNetwork/hi-local-runtime/blob/main/docs/release.md),
and native [MLX](https://github.com/PipeNetwork/hi-local-runtime/blob/main/scripts/hi_mlx_acceptance_matrix.sh)
and [CUDA](https://github.com/PipeNetwork/hi-local-runtime/blob/main/scripts/hi_gguf_acceptance_matrix.sh)
acceptance scripts.
