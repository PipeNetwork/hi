use std::pin::Pin;
use std::process::Command;
use std::sync::Arc;

use anyhow::{Context, Result, anyhow, bail};
use async_trait::async_trait;
use futures_util::Stream;
use tokio::sync::{Mutex, mpsc};

use crate::config::{MlxModelConfig, load_model_config};
use crate::expert_stream;
use crate::generate::TokenizerRuntime;
use crate::manifest::{ModelInfo, inspect_model};
use crate::models::NativeRuntime;
use crate::weights::WeightCatalog;

pub use hi_local_core::backend::{
    BackendHealth, GenerationEvent, GenerationOutput, GenerationRequest, GenerationStream,
    InferenceBackend, SharedBackend,
};

// Self-calibrating speculation gate. `decision` is None until the first greedy request measures
// whether speculation beats the plain loop for this model+hardware; `since` counts greedy requests
// since the last calibration so we can periodically re-measure (workload content shifts acceptance).
#[derive(Default)]
struct SpecGate {
    decision: Option<bool>,
    since: u32,
}

pub struct MlxBackend {
    model: ModelInfo,
    config: MlxModelConfig,
    weights: WeightCatalog,
    runtime: Arc<Mutex<NativeRuntime>>,
    draft: Option<Arc<Mutex<NativeRuntime>>>,
    spec_k: usize,
    spec_gate: Arc<Mutex<SpecGate>>,
    chat_template: Option<String>,
}

// Read the model's chat template: tokenizer_config.json's `chat_template` first, else a separate
// `chat_template.jinja` file (some models — e.g. custom Gemma-4 fine-tunes with a channel/turn format —
// ship it there and leave tokenizer_config empty).
fn load_chat_template(path: &std::path::Path) -> Option<String> {
    if let Ok(text) = std::fs::read_to_string(path.join("tokenizer_config.json")) {
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(&text) {
            if let Some(ct) = v.get("chat_template").and_then(|c| c.as_str()) {
                if !ct.trim().is_empty() {
                    return Some(ct.to_string());
                }
            }
        }
    }
    std::fs::read_to_string(path.join("chat_template.jinja"))
        .ok()
        .filter(|s| !s.trim().is_empty())
}

const OVERSIZE_MODEL_ENV: &str = "HI_MLX_ALLOW_OVERSIZE_MODEL";
const MEMORY_LIMIT_BYTES_ENV: &str = "HI_MLX_MEMORY_LIMIT_BYTES";
const MEMORY_LIMIT_FRACTION_ENV: &str = "HI_MLX_MEMORY_LIMIT_FRACTION";
const DEFAULT_MEMORY_LIMIT_FRACTION: f64 = 0.85;

impl MlxBackend {
    pub fn load(path: impl AsRef<std::path::Path>, model_id: Option<String>) -> Result<Self> {
        Self::load_with_draft(path, model_id, None::<std::path::PathBuf>, 3)
    }

    /// Load a target model, optionally with a draft model for greedy speculative decoding.
    pub fn load_with_draft(
        path: impl AsRef<std::path::Path>,
        model_id: Option<String>,
        draft_path: Option<impl AsRef<std::path::Path>>,
        spec_k: usize,
    ) -> Result<Self> {
        ensure_native_generation_available()?;
        let path = path.as_ref();
        let model = inspect_model(path, model_id)?;
        let config = load_model_config(path)?;
        let weights = WeightCatalog::load(path)?;
        weights.validate_for_config(&config)?;
        // Build the MoE expert-streaming plan and decide whether to stream. When
        // streaming auto-enables (experts + trunk exceed the unified-memory
        // budget), the expert tensors stay out of the resident set, so the
        // admission check only needs to cover the trunk. See `expert_stream`.
        let stream_plan = expert_stream::build_plan(&weights, &config)?;
        let memory_limit = configured_memory_limit_bytes()?;
        let stream_decision = expert_stream::decide(&stream_plan, memory_limit);
        let stream_ctx = if stream_decision.stream {
            eprintln!(
                "hi-mlx: expert streaming enabled — {}; {} expert groups across {} MoE layers, \
                 pool {:.2} GiB",
                stream_decision.reason,
                stream_plan.expert_count_groups(),
                stream_plan.moe_layers,
                stream_decision.pool_bytes as f64 / (1u64 << 30) as f64,
            );
            // When streaming, the expert tensors stay out of the resident set.
            // The trunk (attention, embeddings, shared experts, norms) still
            // loads resident — if even the trunk exceeds the limit, warn but
            // proceed (the user wants to try; the alternative is refusing to
            // load entirely). In practice the trunk is much smaller than the
            // full model, so this only fires on extremely tight budgets.
            if let Err(e) = validate_memory_admission(stream_plan.trunk_bytes) {
                eprintln!(
                    "hi-mlx: warning — trunk alone exceeds memory limit ({e}); proceeding with streaming"
                );
            }
            // Construct the on-demand slab reader + LRU pool, and wrap them in
            // the StreamContext that the model load chain will consume.
            let reader = crate::expert_pool::ExpertSlabReader::new(&stream_plan, path)
                .context("building expert slab reader")?;
            // RAM tier budget: default ~25% of the pool budget, clamped to
            // 1–8 GiB. Override with HI_MLX_EXPERT_RAM_GB (in GiB).
            let ram_tier_bytes = match std::env::var("HI_MLX_EXPERT_RAM_GB") {
                Ok(v) => v.parse::<u64>().unwrap_or(0).saturating_mul(1 << 30),
                Err(_) => {
                    let quarter = stream_decision.pool_bytes / 4;
                    quarter.clamp(1 << 30, 8u64 << 30)
                }
            };
            let mut pool = crate::expert_pool::ExpertPool::new_with_tier(
                reader,
                stream_decision.pool_bytes,
                ram_tier_bytes,
                path.to_path_buf(),
            );
            // Pre-warm the RAM tier with the hottest experts from usage history.
            if ram_tier_bytes > 0 {
                match pool.prewarm_from_usage(512) {
                    Ok(n) if n > 0 => {
                        eprintln!(
                            "hi-mlx: pre-warmed {n} expert slabs into the RAM tier from usage history"
                        );
                    }
                    Ok(_) => {} // no usage history yet — nothing to warm
                    Err(e) => eprintln!("hi-mlx: pre-warm failed (non-fatal): {e:#}"),
                }
            }
            eprintln!(
                "hi-mlx: RAM tier {:.1} GiB, F_NOCACHE direct reads {}",
                ram_tier_bytes as f64 / (1 << 30) as f64,
                if std::env::var("HI_MLX_EXPERT_NOCACHE").as_deref() == Ok("0") {
                    "disabled"
                } else {
                    "enabled"
                },
            );
            Some(crate::models::StreamContext::new(&stream_plan, pool))
        } else {
            validate_memory_admission(weights.estimated_bytes)?;
            None
        };
        let tokenizer = TokenizerRuntime::load(path)?;
        let runtime = NativeRuntime::load(
            config.clone(),
            weights.clone(),
            tokenizer,
            stream_ctx.as_ref(),
        )?;
        let draft = match draft_path {
            Some(dp) => {
                let dp = dp.as_ref();
                let draft = NativeRuntime::from_path(dp)
                    .with_context(|| format!("loading draft model {}", dp.display()))?;
                if !runtime.supports_speculative() {
                    bail!(
                        "speculative decoding target must support KV-cache rollback \
                         (Qwen2/Qwen3 attention); {} does not",
                        path.display()
                    );
                }
                Some(Arc::new(Mutex::new(draft)))
            }
            None => None,
        };
        // Cohere2 (Command-R) ships no usable chat template in the MLX repos; inject a marker so the
        // prompt builder renders the Command-R turn format instead of falling back to ChatML.
        let chat_template = load_chat_template(path).or_else(|| {
            (config.model_type == "cohere2")
                .then(|| "<|START_OF_TURN_TOKEN|><|END_OF_TURN_TOKEN|>".to_string())
        });
        Ok(Self {
            model,
            config,
            weights,
            runtime: Arc::new(Mutex::new(runtime)),
            draft,
            spec_k: spec_k.max(1),
            spec_gate: Arc::new(Mutex::new(SpecGate::default())),
            chat_template,
        })
    }
}

// Measure whether speculation (draft or MTP) actually beats the plain decode loop for this model on
// this hardware, on a short fixed probe. Very slow models are memory/compute-bound and always benefit,
// so a fast plain-rate pre-check short-circuits the (costly) spec probe.
#[cfg(all(target_os = "macos", target_arch = "aarch64", feature = "mlx"))]
fn calibrate_speculation(
    runtime: &mut NativeRuntime,
    mut draft: Option<&mut NativeRuntime>,
    spec_k: usize,
    probe_prompt: &str,
) -> bool {
    use crate::backend::GenerationRequest;
    const N: u32 = 12;
    const SLOW_TPS: f64 = 8.0; // below any spec break-even -> spec always helps
    // Calibrate on a prefix of the real request so draft/MTP acceptance reflects the actual workload
    // (acceptance is content-dependent — a generic probe under-measures it).
    let prompt: String = probe_prompt.chars().take(2000).collect();
    let req = |n: u32| GenerationRequest {
        prompt: prompt.clone(),
        max_tokens: n,
        temperature: 0.0,
        top_p: 1.0,
        top_k: None,
        seed: None,
        stop_sequences: vec![],
        media_inputs: vec![],
    };
    let rate = |out: Option<GenerationOutput>, dt: f64| {
        out.map(|o| o.completion_tokens as f64 / dt.max(1e-6))
            .unwrap_or(0.0)
    };
    // Warm the trunk (first forward compiles Metal shaders / fills caches), then probe the plain rate.
    let _ = runtime.generate(req(2));
    let t0 = std::time::Instant::now();
    let plain_tps = rate(runtime.generate(req(N)).ok(), t0.elapsed().as_secs_f64());
    if plain_tps > 0.0 && plain_tps < SLOW_TPS {
        // Very slow (memory/compute-bound) trunk: speculation always helps; skip the slow spec probe.
        tracing::info!("speculation gate: plain {plain_tps:.1} tok/s (slow) -> ENABLED");
        return true;
    }
    // Warm + probe the spec path.
    match draft.as_deref_mut() {
        Some(d) => {
            let _ = runtime.speculative_generate(d, req(2), spec_k, |_| Ok(()));
        }
        None => {
            let _ = runtime.mtp_generate(req(2), |_| Ok(()));
        }
    }
    let t1 = std::time::Instant::now();
    let spec = match draft.as_deref_mut() {
        Some(d) => runtime
            .speculative_generate(d, req(N), spec_k, |_| Ok(()))
            .map(|(o, _)| o)
            .ok(),
        None => runtime.mtp_generate(req(N), |_| Ok(())).ok(),
    };
    let spec_tps = rate(spec, t1.elapsed().as_secs_f64());
    let enabled = spec_tps > plain_tps * 1.05;
    tracing::info!(
        "speculation gate: plain {plain_tps:.1} vs spec {spec_tps:.1} tok/s -> {}",
        if enabled { "ENABLED" } else { "disabled" }
    );
    enabled
}

fn validate_memory_admission(estimated_bytes: u64) -> Result<()> {
    if env_truthy(OVERSIZE_MODEL_ENV) {
        return Ok(());
    }
    let Some(limit) = configured_memory_limit_bytes()? else {
        return Ok(());
    };
    check_estimated_memory(estimated_bytes, limit)
}

pub fn configured_memory_limit_bytes() -> Result<Option<u64>> {
    if let Some(raw) = std::env::var_os(MEMORY_LIMIT_BYTES_ENV) {
        let raw = raw.to_string_lossy();
        let value = raw
            .trim()
            .parse::<u64>()
            .with_context(|| format!("parsing {MEMORY_LIMIT_BYTES_ENV}={raw:?}"))?;
        return Ok((value > 0).then_some(value));
    }
    let Some(host_bytes) = host_memory_bytes() else {
        return Ok(None);
    };
    let fraction = std::env::var(MEMORY_LIMIT_FRACTION_ENV)
        .ok()
        .map(|raw| {
            raw.parse::<f64>()
                .with_context(|| format!("parsing {MEMORY_LIMIT_FRACTION_ENV}={raw:?}"))
        })
        .transpose()?
        .unwrap_or(DEFAULT_MEMORY_LIMIT_FRACTION);
    if !fraction.is_finite() || fraction <= 0.0 {
        bail!("{MEMORY_LIMIT_FRACTION_ENV} must be a positive finite number");
    }
    Ok(Some(((host_bytes as f64) * fraction.min(1.0)) as u64))
}

fn check_estimated_memory(estimated_bytes: u64, limit_bytes: u64) -> Result<()> {
    if estimated_bytes <= limit_bytes {
        return Ok(());
    }
    bail!(
        "insufficient MLX unified memory: model shards require {} but the configured safe limit is {}; set {OVERSIZE_MODEL_ENV}=1 to override",
        format_bytes(estimated_bytes),
        format_bytes(limit_bytes)
    )
}

fn env_truthy(key: &str) -> bool {
    std::env::var(key)
        .ok()
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(false)
}

fn host_memory_bytes() -> Option<u64> {
    let output = Command::new("sysctl")
        .args(["-n", "hw.memsize"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    std::str::from_utf8(&output.stdout)
        .ok()?
        .trim()
        .parse::<u64>()
        .ok()
}

fn format_bytes(bytes: u64) -> String {
    const GIB: f64 = 1024.0 * 1024.0 * 1024.0;
    format!("{:.2} GiB ({bytes} bytes)", bytes as f64 / GIB)
}

#[async_trait]
impl InferenceBackend for MlxBackend {
    fn model(&self) -> &ModelInfo {
        &self.model
    }

    fn chat_template(&self) -> Option<&str> {
        self.chat_template.as_deref()
    }

    fn health(&self) -> BackendHealth {
        BackendHealth {
            backend: "mlx".to_string(),
            ready: true,
            family: self.config.family.label().to_string(),
            quantization: self.config.quantization_label(),
            context_length: self.config.context_length,
            memory_estimate_bytes: Some(self.weights.estimated_bytes),
        }
    }

    async fn stream_generate(&self, request: GenerationRequest) -> Result<GenerationStream> {
        let runtime = Arc::clone(&self.runtime);
        let draft = self.draft.clone();
        let spec_gate = Arc::clone(&self.spec_gate);
        let spec_k = self.spec_k;
        // Speculative decoding is greedy-only; sampling requests fall back to the normal loop.
        let greedy = request.temperature <= f32::EPSILON;
        let mtp_ok = std::env::var_os("HI_MLX_DISABLE_MTP").is_none();
        let (tx, rx) = mpsc::channel(8);
        tokio::task::spawn_blocking(move || {
            let send = |event| {
                tx.blocking_send(Ok(event))
                    .map_err(|_| anyhow!("generation stream receiver dropped"))
            };
            let result = {
                let mut runtime = runtime.blocking_lock();
                let spec_eligible =
                    greedy && (draft.is_some() || (runtime.supports_mtp() && mtp_ok));
                // Calibrate once per model: measure whether speculation actually beats the plain loop.
                let use_spec = if spec_eligible {
                    let mut gate = spec_gate.blocking_lock();
                    // Calibrate on first use, then re-calibrate every HI_MLX_SPEC_RECAL greedy
                    // requests (default 64; 0 disables) so the decision tracks workload shifts —
                    // acceptance is content-dependent, so a session that changes topic can flip it.
                    let recal = std::env::var("HI_MLX_SPEC_RECAL")
                        .ok()
                        .and_then(|v| v.parse::<u32>().ok())
                        .unwrap_or(64);
                    if gate.decision.is_none() || (recal > 0 && gate.since >= recal) {
                        let mut dguard = draft.as_ref().map(|d| d.blocking_lock());
                        let decision = calibrate_speculation(
                            &mut runtime,
                            dguard.as_deref_mut(),
                            spec_k,
                            &request.prompt,
                        );
                        gate.decision = Some(decision);
                        gate.since = 0;
                    } else {
                        gate.since += 1;
                    }
                    gate.decision.unwrap()
                } else {
                    false
                };
                if use_spec {
                    if let Some(draft) = draft.as_ref() {
                        let mut draft = draft.blocking_lock();
                        runtime
                            .speculative_generate(&mut draft, request, spec_k, send)
                            .map(|(output, _stats)| output)
                    } else {
                        runtime.mtp_generate(request, send)
                    }
                } else {
                    runtime.stream_generate(request, send)
                }
            };
            if let Err(err) = result {
                let _ = tx.blocking_send(Err(err));
            }
        });
        Ok(receiver_stream(rx))
    }
}

pub type NativeBackend = MlxBackend;

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn ensure_native_generation_available() -> Result<()> {
    Ok(())
}

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
fn ensure_native_generation_available() -> Result<()> {
    anyhow::bail!("native MLX inference requires Apple Silicon macOS")
}

pub fn platform_supported() -> bool {
    cfg!(all(target_os = "macos", target_arch = "aarch64"))
}

#[cfg(test)]
pub struct MockBackend {
    model: ModelInfo,
    output: tokio::sync::Mutex<String>,
    last_prompt: tokio::sync::Mutex<Option<String>>,
}

#[cfg(test)]
impl MockBackend {
    pub fn new(model: ModelInfo, output: impl Into<String>) -> Self {
        Self {
            model,
            output: tokio::sync::Mutex::new(output.into()),
            last_prompt: tokio::sync::Mutex::new(None),
        }
    }

    pub async fn last_prompt(&self) -> Option<String> {
        self.last_prompt.lock().await.clone()
    }
}

#[cfg(test)]
#[async_trait]
impl InferenceBackend for MockBackend {
    fn model(&self) -> &ModelInfo {
        &self.model
    }

    fn health(&self) -> BackendHealth {
        BackendHealth {
            backend: "mlx".to_string(),
            ready: true,
            family: self.model.family.label().to_string(),
            quantization: "mock".to_string(),
            context_length: self.model.context_length,
            memory_estimate_bytes: Some(self.model.weight_shards.iter().map(|s| s.bytes).sum()),
        }
    }

    async fn stream_generate(&self, request: GenerationRequest) -> Result<GenerationStream> {
        *self.last_prompt.lock().await = Some(request.prompt.clone());
        let text = self.output.lock().await.clone();
        let prompt_tokens = (request.prompt.len() / 4).max(1) as u64;
        let completion_tokens = (text.len() / 4).max(1) as u64;
        let mut events = split_stream_text(&text)
            .into_iter()
            .map(|piece| {
                Ok(GenerationEvent::TokenDelta {
                    token_id: 0,
                    text: piece,
                })
            })
            .collect::<Vec<_>>();
        events.push(Ok(GenerationEvent::Finished {
            output: GenerationOutput {
                prompt_tokens,
                completion_tokens,
                text,
            },
        }));
        Ok(Box::pin(futures_util::stream::iter(events)))
    }
}

fn receiver_stream<T: Send + 'static>(
    rx: mpsc::Receiver<T>,
) -> Pin<Box<dyn Stream<Item = T> + Send>> {
    Box::pin(futures_util::stream::unfold(rx, |mut rx| async {
        rx.recv().await.map(|item| (item, rx))
    }))
}

#[cfg(test)]
fn split_stream_text(text: &str) -> Vec<String> {
    if text.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::new();
    let mut current = String::new();
    for ch in text.chars() {
        current.push(ch);
        if current.len() >= 512 {
            out.push(std::mem::take(&mut current));
        }
    }
    if !current.is_empty() {
        out.push(current);
    }
    out
}

#[cfg(test)]
mod tests {
    use futures_util::StreamExt;

    use super::*;
    use crate::manifest::{inspect_model, test_support};

    #[cfg(all(target_os = "macos", target_arch = "aarch64", feature = "mlx"))]
    static MLX_TEST_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

    #[tokio::test]
    async fn mock_stream_emits_delta_before_finish() {
        let backend = MockBackend::new(test_model(), "streamed text");
        let mut stream = backend
            .stream_generate(GenerationRequest {
                prompt: "hello".to_string(),
                max_tokens: 4,
                temperature: 0.0,
                top_p: 1.0,
                top_k: None,
                seed: None,
                stop_sequences: Vec::new(),
                media_inputs: Vec::new(),
            })
            .await
            .unwrap();

        let first = stream.next().await.unwrap().unwrap();
        match first {
            GenerationEvent::TokenDelta { text, .. } => assert_eq!(text, "streamed text"),
            GenerationEvent::Finished { .. } => panic!("first stream event must be a delta"),
        }

        let second = stream.next().await.unwrap().unwrap();
        assert!(matches!(second, GenerationEvent::Finished { .. }));
    }

    #[tokio::test]
    async fn generate_collects_the_stream_output() {
        let backend = MockBackend::new(test_model(), "collected text");
        let output = backend
            .generate(GenerationRequest {
                prompt: "hello".to_string(),
                max_tokens: 4,
                temperature: 0.0,
                top_p: 1.0,
                top_k: None,
                seed: None,
                stop_sequences: Vec::new(),
                media_inputs: Vec::new(),
            })
            .await
            .unwrap();

        assert_eq!(output.text, "collected text");
    }

    #[test]
    fn memory_admission_rejects_models_over_limit() {
        let err = check_estimated_memory(120, 100).unwrap_err();

        assert!(err.to_string().contains("insufficient MLX unified memory"));
        assert!(err.to_string().contains(OVERSIZE_MODEL_ENV));
    }

    #[test]
    fn memory_admission_allows_models_under_limit() {
        check_estimated_memory(100, 120).unwrap();
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64", feature = "mlx"))]
    #[tokio::test]
    async fn native_backend_generates_from_tiny_compressed_deepseek_v4_fixture() {
        use std::collections::HashMap;
        use std::fs;
        use std::path::{Path, PathBuf};

        use mlx_rs::Array;
        use tokenizers::Tokenizer;
        use tokenizers::models::wordlevel::WordLevel;

        let _guard = MLX_TEST_LOCK.lock().await;
        let dir = tempfile_path("native-deepseek-v4-compressed");
        fs::create_dir_all(&dir).unwrap();
        fs::write(
            dir.join("config.json"),
            r#"{
              "architectures": ["DeepseekV4ForCausalLM"],
              "model_type": "deepseek_v4",
              "hidden_size": 4,
              "intermediate_size": 8,
              "moe_intermediate_size": 4,
              "num_hidden_layers": 1,
              "num_attention_heads": 1,
              "num_key_value_heads": 1,
              "head_dim": 4,
              "qk_rope_head_dim": 2,
              "q_lora_rank": 4,
              "index_head_dim": 2,
              "index_n_heads": 1,
              "index_topk": 1,
              "o_lora_rank": 4,
              "o_groups": 1,
              "n_routed_experts": 2,
              "n_shared_experts": 0,
              "num_experts_per_tok": 1,
              "num_hash_layers": 0,
              "scoring_func": "sqrtsoftplus",
              "norm_topk_prob": true,
              "routed_scaling_factor": 1.0,
              "swiglu_limit": 0.0,
              "hc_mult": 1,
              "hc_sinkhorn_iters": 1,
              "hc_eps": 1e-6,
              "compress_ratios": [4],
              "compress_rope_theta": 160000,
              "sliding_window": 2,
              "vocab_size": 4,
              "max_position_embeddings": 16,
              "rms_norm_eps": 1e-6,
              "rope_theta": 10000,
              "tie_word_embeddings": false,
              "eos_token_id": 99
            }"#,
        )
        .unwrap();
        write_tokenizer(&dir);
        write_weights(&dir);

        let backend =
            super::MlxBackend::load(&dir, Some("tiny-v4-compressed".to_string())).unwrap();
        let output = super::InferenceBackend::generate(
            &backend,
            super::GenerationRequest {
                prompt: "hello".to_string(),
                max_tokens: 4,
                temperature: 0.0,
                top_p: 1.0,
                top_k: None,
                seed: None,
                stop_sequences: Vec::new(),
                media_inputs: Vec::new(),
            },
        )
        .await
        .unwrap();

        assert_eq!(output.prompt_tokens, 1);
        assert_eq!(output.completion_tokens, 4);
        assert!(!output.text.trim().is_empty());

        fn write_tokenizer(root: &Path) {
            let model = WordLevel::builder()
                .vocab(HashMap::from([
                    ("<unk>".to_string(), 0),
                    ("hello".to_string(), 1),
                    ("</s>".to_string(), 2),
                    ("world".to_string(), 3),
                ]))
                .unk_token("<unk>".to_string())
                .build()
                .unwrap();
            Tokenizer::new(model)
                .save(root.join("tokenizer.json"), false)
                .unwrap();
        }

        fn write_weights(root: &Path) {
            let mut arrays = HashMap::new();
            let vocab = [
                -1.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, -2.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0,
            ];
            arrays.insert(
                "model.embed_tokens.weight".to_string(),
                Array::from_slice(&vocab, &[4, 4]),
            );
            arrays.insert(
                "lm_head.weight".to_string(),
                Array::from_slice(
                    &[
                        -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2.0, 0.0, 0.0, 0.0, 10.0, 0.0,
                        0.0, 0.0,
                    ],
                    &[4, 4],
                ),
            );
            arrays.insert("model.hc_head.fn".to_string(), zeros(&[1, 4]));
            arrays.insert("model.hc_head.base".to_string(), zeros(&[1]));
            arrays.insert("model.hc_head.scale".to_string(), zeros(&[1]));
            arrays.insert("model.norm.weight".to_string(), ones(4));

            let prefix = "model.layers.0";
            arrays.insert(format!("{prefix}.attn_norm.weight"), ones(4));
            let attn = format!("{prefix}.attn");
            arrays.insert(format!("{attn}.wq_a.weight"), zeros(&[4, 4]));
            arrays.insert(format!("{attn}.q_norm.weight"), ones(4));
            arrays.insert(format!("{attn}.wq_b.weight"), zeros(&[4, 4]));
            arrays.insert(format!("{attn}.wkv.weight"), zeros(&[4, 4]));
            arrays.insert(format!("{attn}.kv_norm.weight"), ones(4));
            arrays.insert(format!("{attn}.wo_a.weight"), zeros(&[4, 4]));
            arrays.insert(format!("{attn}.wo_b.weight"), zeros(&[4, 4]));
            arrays.insert(format!("{attn}.compressor.ape"), zeros(&[4, 8]));
            arrays.insert(format!("{attn}.compressor.norm.weight"), ones(4));
            arrays.insert(format!("{attn}.compressor.wgate.weight"), zeros(&[8, 4]));
            arrays.insert(format!("{attn}.compressor.wkv.weight"), zeros(&[8, 4]));
            arrays.insert(format!("{attn}.indexer.compressor.ape"), zeros(&[4, 4]));
            arrays.insert(format!("{attn}.indexer.compressor.norm.weight"), ones(2));
            arrays.insert(
                format!("{attn}.indexer.compressor.wgate.weight"),
                zeros(&[4, 4]),
            );
            arrays.insert(
                format!("{attn}.indexer.compressor.wkv.weight"),
                zeros(&[4, 4]),
            );
            arrays.insert(format!("{attn}.indexer.wq_b.weight"), zeros(&[2, 4]));
            arrays.insert(
                format!("{attn}.indexer.weights_proj.weight"),
                zeros(&[1, 4]),
            );
            arrays.insert(format!("{prefix}.attn_hc.fn"), zeros(&[3, 4]));
            arrays.insert(format!("{prefix}.attn_hc.base"), zeros(&[3]));
            arrays.insert(format!("{prefix}.attn_hc.scale"), zeros(&[3]));

            arrays.insert(format!("{prefix}.ffn_norm.weight"), ones(4));
            arrays.insert(format!("{prefix}.ffn.gate.weight"), zeros(&[2, 4]));
            for name in ["gate_proj", "up_proj", "down_proj"] {
                arrays.insert(
                    format!("{prefix}.ffn.switch_mlp.{name}.weight"),
                    zeros(&[2, 4, 4]),
                );
            }
            arrays.insert(format!("{prefix}.ffn_hc.fn"), zeros(&[3, 4]));
            arrays.insert(format!("{prefix}.ffn_hc.base"), zeros(&[3]));
            arrays.insert(format!("{prefix}.ffn_hc.scale"), zeros(&[3]));

            Array::save_safetensors(&arrays, None, root.join("model.safetensors")).unwrap();
        }

        fn ones(len: usize) -> Array {
            Array::from_slice(&vec![1.0f32; len], &[len as i32])
        }

        fn zeros(shape: &[i32]) -> Array {
            let len = shape.iter().product::<i32>() as usize;
            Array::from_slice(&vec![0.0f32; len], shape)
        }

        fn tempfile_path(name: &str) -> PathBuf {
            let mut path = std::env::temp_dir();
            path.push(format!(
                "hi-mlx-{name}-{}",
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ));
            path
        }
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64", feature = "mlx"))]
    #[tokio::test]
    async fn native_backend_generates_from_tiny_deepseek_v4_fixture() {
        use std::collections::HashMap;
        use std::fs;
        use std::path::{Path, PathBuf};

        use mlx_rs::Array;
        use tokenizers::Tokenizer;
        use tokenizers::models::wordlevel::WordLevel;

        let _guard = MLX_TEST_LOCK.lock().await;
        let dir = tempfile_path("native-deepseek-v4");
        fs::create_dir_all(&dir).unwrap();
        write_config(&dir);
        write_tokenizer(&dir);
        write_weights(&dir);

        let backend = super::MlxBackend::load(&dir, Some("tiny-deepseek-v4".to_string())).unwrap();
        let output = super::InferenceBackend::generate(
            &backend,
            super::GenerationRequest {
                prompt: "hello".to_string(),
                max_tokens: 2,
                temperature: 0.0,
                top_p: 1.0,
                top_k: None,
                seed: None,
                stop_sequences: Vec::new(),
                media_inputs: Vec::new(),
            },
        )
        .await
        .unwrap();

        assert_eq!(output.prompt_tokens, 1);
        assert_eq!(output.completion_tokens, 2);
        assert!(!output.text.trim().is_empty());

        fn write_config(root: &Path) {
            fs::write(
                root.join("config.json"),
                r#"{
                  "architectures": ["DeepseekV4ForCausalLM"],
                  "model_type": "deepseek_v4",
                  "hidden_size": 4,
                  "intermediate_size": 8,
                  "moe_intermediate_size": 4,
                  "num_hidden_layers": 1,
                  "num_attention_heads": 1,
                  "num_key_value_heads": 1,
                  "head_dim": 4,
                  "qk_rope_head_dim": 0,
                  "q_lora_rank": 4,
                  "o_lora_rank": 4,
                  "o_groups": 1,
                  "n_routed_experts": 2,
                  "n_shared_experts": 0,
                  "num_experts_per_tok": 1,
                  "num_hash_layers": 0,
                  "scoring_func": "sqrtsoftplus",
                  "norm_topk_prob": true,
                  "routed_scaling_factor": 1.0,
                  "swiglu_limit": 0.0,
                  "hc_mult": 1,
                  "hc_sinkhorn_iters": 1,
                  "hc_eps": 1e-6,
                  "compress_ratios": [0],
                  "vocab_size": 4,
                  "max_position_embeddings": 16,
                  "rms_norm_eps": 1e-6,
                  "rope_theta": 10000,
                  "tie_word_embeddings": false,
                  "eos_token_id": 99
                }"#,
            )
            .unwrap();
        }

        fn write_tokenizer(root: &Path) {
            let model = WordLevel::builder()
                .vocab(HashMap::from([
                    ("<unk>".to_string(), 0),
                    ("hello".to_string(), 1),
                    ("</s>".to_string(), 2),
                    ("world".to_string(), 3),
                ]))
                .unk_token("<unk>".to_string())
                .build()
                .unwrap();
            Tokenizer::new(model)
                .save(root.join("tokenizer.json"), false)
                .unwrap();
        }

        fn write_weights(root: &Path) {
            let mut arrays = HashMap::new();
            let vocab = [
                -1.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, -2.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0,
            ];
            arrays.insert(
                "model.embed_tokens.weight".to_string(),
                Array::from_slice(&vocab, &[4, 4]),
            );
            arrays.insert(
                "lm_head.weight".to_string(),
                Array::from_slice(
                    &[
                        -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2.0, 0.0, 0.0, 0.0, 10.0, 0.0,
                        0.0, 0.0,
                    ],
                    &[4, 4],
                ),
            );
            arrays.insert("model.hc_head.fn".to_string(), zeros(&[1, 4]));
            arrays.insert("model.hc_head.base".to_string(), zeros(&[1]));
            arrays.insert("model.hc_head.scale".to_string(), zeros(&[1]));
            arrays.insert("model.norm.weight".to_string(), ones(4));

            let prefix = "model.layers.0";
            arrays.insert(format!("{prefix}.attn_norm.weight"), ones(4));
            let attn = format!("{prefix}.attn");
            arrays.insert(format!("{attn}.wq_a.weight"), zeros(&[4, 4]));
            arrays.insert(format!("{attn}.q_norm.weight"), ones(4));
            arrays.insert(format!("{attn}.wq_b.weight"), zeros(&[4, 4]));
            arrays.insert(format!("{attn}.wkv.weight"), zeros(&[4, 4]));
            arrays.insert(format!("{attn}.kv_norm.weight"), ones(4));
            arrays.insert(format!("{attn}.wo_a.weight"), zeros(&[4, 4]));
            arrays.insert(format!("{attn}.wo_b.weight"), zeros(&[4, 4]));
            arrays.insert(format!("{prefix}.attn_hc.fn"), zeros(&[3, 4]));
            arrays.insert(format!("{prefix}.attn_hc.base"), zeros(&[3]));
            arrays.insert(format!("{prefix}.attn_hc.scale"), zeros(&[3]));

            arrays.insert(format!("{prefix}.ffn_norm.weight"), ones(4));
            arrays.insert(format!("{prefix}.ffn.gate.weight"), zeros(&[2, 4]));
            for name in ["gate_proj", "up_proj", "down_proj"] {
                arrays.insert(
                    format!("{prefix}.ffn.switch_mlp.{name}.weight"),
                    zeros(&[2, 4, 4]),
                );
            }
            arrays.insert(format!("{prefix}.ffn_hc.fn"), zeros(&[3, 4]));
            arrays.insert(format!("{prefix}.ffn_hc.base"), zeros(&[3]));
            arrays.insert(format!("{prefix}.ffn_hc.scale"), zeros(&[3]));

            Array::save_safetensors(&arrays, None, root.join("model.safetensors")).unwrap();
        }

        fn ones(len: usize) -> Array {
            Array::from_slice(&vec![1.0f32; len], &[len as i32])
        }

        fn zeros(shape: &[i32]) -> Array {
            let len = shape.iter().product::<i32>() as usize;
            Array::from_slice(&vec![0.0f32; len], shape)
        }

        fn tempfile_path(name: &str) -> PathBuf {
            let mut path = std::env::temp_dir();
            path.push(format!(
                "hi-mlx-{name}-{}",
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ));
            path
        }
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64", feature = "mlx"))]
    #[tokio::test]
    async fn native_backend_generates_from_tiny_qwen_fixture() {
        use std::collections::HashMap;
        use std::fs;
        use std::path::{Path, PathBuf};

        use mlx_rs::Array;
        use tokenizers::Tokenizer;
        use tokenizers::models::wordlevel::WordLevel;

        let _guard = MLX_TEST_LOCK.lock().await;
        let dir = tempfile_path("native-qwen");
        fs::create_dir_all(&dir).unwrap();
        write_config(&dir);
        write_tokenizer(&dir);
        write_weights(&dir);

        let backend = super::MlxBackend::load(&dir, Some("tiny-qwen".to_string())).unwrap();
        let output = super::InferenceBackend::generate(
            &backend,
            super::GenerationRequest {
                prompt: "hello".to_string(),
                max_tokens: 2,
                temperature: 0.0,
                top_p: 1.0,
                top_k: None,
                seed: None,
                stop_sequences: Vec::new(),
                media_inputs: Vec::new(),
            },
        )
        .await
        .unwrap();

        assert_eq!(output.prompt_tokens, 1);
        assert!((1..=2).contains(&output.completion_tokens));
        assert!(!output.text.trim().is_empty());

        fn write_config(root: &Path) {
            fs::write(
                root.join("config.json"),
                r#"{
                  "architectures": ["Qwen3ForCausalLM"],
                  "model_type": "qwen3",
                  "hidden_size": 4,
                  "intermediate_size": 8,
                  "num_hidden_layers": 1,
                  "num_attention_heads": 1,
                  "num_key_value_heads": 1,
                  "head_dim": 4,
                  "vocab_size": 4,
                  "max_position_embeddings": 16,
                  "rms_norm_eps": 1e-6,
                  "rope_theta": 1000000,
                  "tie_word_embeddings": true,
                  "eos_token_id": 2
                }"#,
            )
            .unwrap();
        }

        fn write_tokenizer(root: &Path) {
            let model = WordLevel::builder()
                .vocab(HashMap::from([
                    ("<unk>".to_string(), 0),
                    ("hello".to_string(), 1),
                    ("</s>".to_string(), 2),
                    ("world".to_string(), 3),
                ]))
                .unk_token("<unk>".to_string())
                .build()
                .unwrap();
            Tokenizer::new(model)
                .save(root.join("tokenizer.json"), false)
                .unwrap();
        }

        fn write_weights(root: &Path) {
            let mut arrays = HashMap::new();
            arrays.insert(
                "model.embed_tokens.weight".to_string(),
                Array::from_slice(
                    &[
                        -1.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, -2.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                        0.0, 0.0,
                    ],
                    &[4, 4],
                ),
            );
            arrays.insert("model.norm.weight".to_string(), ones(4));
            arrays.insert("model.layers.0.input_layernorm.weight".to_string(), ones(4));
            arrays.insert(
                "model.layers.0.post_attention_layernorm.weight".to_string(),
                ones(4),
            );
            arrays.insert(
                "model.layers.0.self_attn.q_norm.weight".to_string(),
                ones(4),
            );
            arrays.insert(
                "model.layers.0.self_attn.k_norm.weight".to_string(),
                ones(4),
            );
            for name in [
                "q_proj.weight",
                "k_proj.weight",
                "v_proj.weight",
                "o_proj.weight",
            ] {
                arrays.insert(format!("model.layers.0.self_attn.{name}"), zeros(&[4, 4]));
            }
            arrays.insert(
                "model.layers.0.mlp.gate_proj.weight".to_string(),
                zeros(&[8, 4]),
            );
            arrays.insert(
                "model.layers.0.mlp.up_proj.weight".to_string(),
                zeros(&[8, 4]),
            );
            arrays.insert(
                "model.layers.0.mlp.down_proj.weight".to_string(),
                zeros(&[4, 8]),
            );
            Array::save_safetensors(&arrays, None, root.join("model.safetensors")).unwrap();
        }

        fn ones(len: usize) -> Array {
            Array::from_slice(&vec![1.0f32; len], &[len as i32])
        }

        fn zeros(shape: &[i32]) -> Array {
            let len = shape.iter().product::<i32>() as usize;
            Array::from_slice(&vec![0.0f32; len], shape)
        }

        fn tempfile_path(name: &str) -> PathBuf {
            let mut path = std::env::temp_dir();
            path.push(format!(
                "hi-mlx-{name}-{}",
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ));
            path
        }
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64", feature = "mlx"))]
    #[tokio::test]
    async fn native_backend_generates_from_tiny_qwen_moe_fixture() {
        use std::collections::HashMap;
        use std::fs;
        use std::path::{Path, PathBuf};

        use mlx_rs::Array;
        use tokenizers::Tokenizer;
        use tokenizers::models::wordlevel::WordLevel;

        let _guard = MLX_TEST_LOCK.lock().await;
        let dir = tempfile_path("native-qwen-moe");
        fs::create_dir_all(&dir).unwrap();
        write_config(&dir);
        write_tokenizer(&dir);
        write_weights(&dir);

        let backend = super::MlxBackend::load(&dir, Some("tiny-qwen-moe".to_string())).unwrap();
        let output = super::InferenceBackend::generate(
            &backend,
            super::GenerationRequest {
                prompt: "hello".to_string(),
                max_tokens: 2,
                temperature: 0.0,
                top_p: 1.0,
                top_k: None,
                seed: None,
                stop_sequences: Vec::new(),
                media_inputs: Vec::new(),
            },
        )
        .await
        .unwrap();

        assert_eq!(output.prompt_tokens, 1);
        assert_eq!(output.completion_tokens, 2);
        assert!(!output.text.trim().is_empty());

        fn write_config(root: &Path) {
            fs::write(
                root.join("config.json"),
                r#"{
                  "architectures": ["Qwen3MoeForCausalLM"],
                  "model_type": "qwen3_moe",
                  "hidden_size": 4,
                  "intermediate_size": 8,
                  "moe_intermediate_size": 4,
                  "num_hidden_layers": 1,
                  "num_attention_heads": 1,
                  "num_key_value_heads": 1,
                  "head_dim": 4,
                  "num_experts": 2,
                  "num_experts_per_tok": 1,
                  "decoder_sparse_step": 1,
                  "mlp_only_layers": [],
                  "norm_topk_prob": true,
                  "vocab_size": 4,
                  "max_position_embeddings": 16,
                  "rms_norm_eps": 1e-6,
                  "rope_theta": 1000000,
                  "tie_word_embeddings": true,
                  "eos_token_id": 99
                }"#,
            )
            .unwrap();
        }

        fn write_tokenizer(root: &Path) {
            let model = WordLevel::builder()
                .vocab(HashMap::from([
                    ("<unk>".to_string(), 0),
                    ("hello".to_string(), 1),
                    ("</s>".to_string(), 2),
                    ("world".to_string(), 3),
                ]))
                .unk_token("<unk>".to_string())
                .build()
                .unwrap();
            Tokenizer::new(model)
                .save(root.join("tokenizer.json"), false)
                .unwrap();
        }

        fn write_weights(root: &Path) {
            let mut arrays = HashMap::new();
            arrays.insert(
                "model.embed_tokens.weight".to_string(),
                Array::from_slice(
                    &[
                        -1.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, -2.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                        0.0, 0.0,
                    ],
                    &[4, 4],
                ),
            );
            arrays.insert("model.norm.weight".to_string(), ones(4));
            arrays.insert("model.layers.0.input_layernorm.weight".to_string(), ones(4));
            arrays.insert(
                "model.layers.0.post_attention_layernorm.weight".to_string(),
                ones(4),
            );
            arrays.insert(
                "model.layers.0.self_attn.q_norm.weight".to_string(),
                ones(4),
            );
            arrays.insert(
                "model.layers.0.self_attn.k_norm.weight".to_string(),
                ones(4),
            );
            for name in [
                "q_proj.weight",
                "k_proj.weight",
                "v_proj.weight",
                "o_proj.weight",
            ] {
                arrays.insert(format!("model.layers.0.self_attn.{name}"), zeros(&[4, 4]));
            }
            arrays.insert("model.layers.0.mlp.gate.weight".to_string(), zeros(&[2, 4]));
            for expert in 0..2 {
                for name in ["gate_proj", "up_proj", "down_proj"] {
                    arrays.insert(
                        format!("model.layers.0.mlp.experts.{expert}.{name}.weight"),
                        zeros(&[4, 4]),
                    );
                }
            }
            Array::save_safetensors(&arrays, None, root.join("model.safetensors")).unwrap();
        }

        fn ones(len: usize) -> Array {
            Array::from_slice(&vec![1.0f32; len], &[len as i32])
        }

        fn zeros(shape: &[i32]) -> Array {
            let len = shape.iter().product::<i32>() as usize;
            Array::from_slice(&vec![0.0f32; len], shape)
        }

        fn tempfile_path(name: &str) -> PathBuf {
            let mut path = std::env::temp_dir();
            path.push(format!(
                "hi-mlx-{name}-{}",
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ));
            path
        }
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64", feature = "mlx"))]
    #[tokio::test]
    async fn native_backend_generates_from_tiny_glm_moe_fixture() {
        use std::collections::HashMap;
        use std::fs;
        use std::path::{Path, PathBuf};

        use mlx_rs::Array;
        use tokenizers::Tokenizer;
        use tokenizers::models::wordlevel::WordLevel;

        let _guard = MLX_TEST_LOCK.lock().await;
        let dir = tempfile_path("native-glm-moe");
        fs::create_dir_all(&dir).unwrap();
        write_config(&dir);
        write_tokenizer(&dir);
        write_weights(&dir);

        let backend = super::MlxBackend::load(&dir, Some("tiny-glm".to_string())).unwrap();
        let output = super::InferenceBackend::generate(
            &backend,
            super::GenerationRequest {
                prompt: "hello".to_string(),
                max_tokens: 2,
                temperature: 0.0,
                top_p: 1.0,
                top_k: None,
                seed: None,
                stop_sequences: Vec::new(),
                media_inputs: Vec::new(),
            },
        )
        .await
        .unwrap();

        assert_eq!(output.prompt_tokens, 1);
        assert!((1..=2).contains(&output.completion_tokens));
        assert!(!output.text.trim().is_empty());

        fn write_config(root: &Path) {
            fs::write(
                root.join("config.json"),
                r#"{
                  "architectures": ["Glm4MoeLiteForCausalLM"],
                  "model_type": "glm4_moe_lite",
                  "hidden_size": 4,
                  "intermediate_size": 8,
                  "moe_intermediate_size": 4,
                  "num_hidden_layers": 2,
                  "num_attention_heads": 1,
                  "num_key_value_heads": 1,
                  "qk_nope_head_dim": 2,
                  "qk_rope_head_dim": 2,
                  "v_head_dim": 2,
                  "q_lora_rank": 3,
                  "kv_lora_rank": 3,
                  "vocab_size": 4,
                  "max_position_embeddings": 16,
                  "rms_norm_eps": 1e-6,
                  "rope_theta": 1000000,
                  "tie_word_embeddings": false,
                  "first_k_dense_replace": 1,
                  "moe_layer_freq": 1,
                  "n_routed_experts": 2,
                  "n_shared_experts": 1,
                  "num_experts_per_tok": 1,
                  "n_group": 1,
                  "topk_group": 1,
                  "topk_method": "noaux_tc",
                  "eos_token_id": 99
                }"#,
            )
            .unwrap();
        }

        fn write_tokenizer(root: &Path) {
            let model = WordLevel::builder()
                .vocab(HashMap::from([
                    ("<unk>".to_string(), 0),
                    ("hello".to_string(), 1),
                    ("</s>".to_string(), 2),
                    ("world".to_string(), 3),
                ]))
                .unk_token("<unk>".to_string())
                .build()
                .unwrap();
            Tokenizer::new(model)
                .save(root.join("tokenizer.json"), false)
                .unwrap();
        }

        fn write_weights(root: &Path) {
            let mut arrays = HashMap::new();
            let vocab = [
                -1.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, -2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ];
            arrays.insert(
                "model.embed_tokens.weight".to_string(),
                Array::from_slice(&vocab, &[4, 4]),
            );
            arrays.insert(
                "lm_head.weight".to_string(),
                Array::from_slice(&vocab, &[4, 4]),
            );
            arrays.insert("model.norm.weight".to_string(), ones(4));
            for layer in 0..2 {
                let prefix = format!("model.layers.{layer}");
                arrays.insert(format!("{prefix}.input_layernorm.weight"), ones(4));
                arrays.insert(format!("{prefix}.post_attention_layernorm.weight"), ones(4));
                let attn = format!("{prefix}.self_attn");
                arrays.insert(format!("{attn}.q_a_proj.weight"), zeros(&[3, 4]));
                arrays.insert(format!("{attn}.q_a_layernorm.weight"), ones(3));
                arrays.insert(format!("{attn}.q_b_proj.weight"), zeros(&[4, 3]));
                arrays.insert(format!("{attn}.kv_a_proj_with_mqa.weight"), zeros(&[5, 4]));
                arrays.insert(format!("{attn}.kv_a_layernorm.weight"), ones(3));
                arrays.insert(format!("{attn}.embed_q.weight"), zeros(&[1, 3, 2]));
                arrays.insert(format!("{attn}.unembed_out.weight"), zeros(&[1, 2, 3]));
                arrays.insert(format!("{attn}.o_proj.weight"), zeros(&[4, 2]));
            }
            arrays.insert(
                "model.layers.0.mlp.gate_proj.weight".to_string(),
                zeros(&[8, 4]),
            );
            arrays.insert(
                "model.layers.0.mlp.up_proj.weight".to_string(),
                zeros(&[8, 4]),
            );
            arrays.insert(
                "model.layers.0.mlp.down_proj.weight".to_string(),
                zeros(&[4, 8]),
            );
            arrays.insert("model.layers.1.mlp.gate.weight".to_string(), zeros(&[2, 4]));
            arrays.insert(
                "model.layers.1.mlp.gate.e_score_correction_bias".to_string(),
                zeros(&[2]),
            );
            for name in ["gate_proj", "up_proj"] {
                arrays.insert(
                    format!("model.layers.1.mlp.switch_mlp.{name}.weight"),
                    zeros(&[2, 4, 4]),
                );
            }
            arrays.insert(
                "model.layers.1.mlp.switch_mlp.down_proj.weight".to_string(),
                zeros(&[2, 4, 4]),
            );
            for name in ["gate_proj", "up_proj"] {
                arrays.insert(
                    format!("model.layers.1.mlp.shared_experts.{name}.weight"),
                    zeros(&[4, 4]),
                );
            }
            arrays.insert(
                "model.layers.1.mlp.shared_experts.down_proj.weight".to_string(),
                zeros(&[4, 4]),
            );
            Array::save_safetensors(&arrays, None, root.join("model.safetensors")).unwrap();
        }

        fn ones(len: usize) -> Array {
            Array::from_slice(&vec![1.0f32; len], &[len as i32])
        }

        fn zeros(shape: &[i32]) -> Array {
            let len = shape.iter().product::<i32>() as usize;
            Array::from_slice(&vec![0.0f32; len], shape)
        }

        fn tempfile_path(name: &str) -> PathBuf {
            let mut path = std::env::temp_dir();
            path.push(format!(
                "hi-mlx-{name}-{}",
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ));
            path
        }
    }

    /// The acceptance test for MLX MoE expert streaming: load the same tiny GLM
    /// MoE fixture twice — once resident, once with `HI_MLX_EXPERT_STREAMING=1`
    /// forcing the on-demand pool path — and assert the generated text matches.
    /// This mirrors CUDA's `native_cuda_expert_streaming_matches_resident`.
    #[cfg(all(target_os = "macos", target_arch = "aarch64", feature = "mlx"))]
    #[tokio::test]
    async fn expert_streaming_matches_resident_output() {
        use std::collections::HashMap;
        use std::fs;
        use std::path::Path;

        use mlx_rs::Array;
        use tokenizers::Tokenizer;
        use tokenizers::models::wordlevel::WordLevel;

        let _guard = MLX_TEST_LOCK.lock().await;

        // Helper: build the fixture in a fresh temp dir (we need two independent
        // dirs so the two loads don't share file handles).
        fn build_fixture(dir: &Path) {
            fs::create_dir_all(dir).unwrap();
            write_config(dir);
            write_tokenizer(dir);
            write_weights(dir);

            fn write_config(root: &Path) {
                fs::write(
                    root.join("config.json"),
                    r#"{
                      "architectures": ["Glm4MoeLiteForCausalLM"],
                      "model_type": "glm4_moe_lite",
                      "hidden_size": 4,
                      "intermediate_size": 8,
                      "moe_intermediate_size": 4,
                      "num_hidden_layers": 2,
                      "num_attention_heads": 1,
                      "num_key_value_heads": 1,
                      "qk_nope_head_dim": 2,
                      "qk_rope_head_dim": 2,
                      "v_head_dim": 2,
                      "q_lora_rank": 3,
                      "kv_lora_rank": 3,
                      "vocab_size": 4,
                      "max_position_embeddings": 16,
                      "rms_norm_eps": 1e-6,
                      "rope_theta": 1000000,
                      "tie_word_embeddings": false,
                      "first_k_dense_replace": 1,
                      "moe_layer_freq": 1,
                      "n_routed_experts": 2,
                      "n_shared_experts": 1,
                      "num_experts_per_tok": 1,
                      "n_group": 1,
                      "topk_group": 1,
                      "topk_method": "noaux_tc",
                      "eos_token_id": 99
                    }"#,
                )
                .unwrap();
            }

            fn write_tokenizer(root: &Path) {
                let model = WordLevel::builder()
                    .vocab(HashMap::from([
                        ("<unk>".to_string(), 0),
                        ("hello".to_string(), 1),
                        ("</s>".to_string(), 2),
                        ("world".to_string(), 3),
                    ]))
                    .unk_token("<unk>".to_string())
                    .build()
                    .unwrap();
                Tokenizer::new(model)
                    .save(root.join("tokenizer.json"), false)
                    .unwrap();
            }

            fn write_weights(root: &Path) {
                let mut arrays = HashMap::new();
                let vocab = [
                    -1.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, -2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    0.0,
                ];
                arrays.insert(
                    "model.embed_tokens.weight".to_string(),
                    Array::from_slice(&vocab, &[4, 4]),
                );
                arrays.insert(
                    "lm_head.weight".to_string(),
                    Array::from_slice(&vocab, &[4, 4]),
                );
                arrays.insert("model.norm.weight".to_string(), ones(4));
                for layer in 0..2 {
                    let prefix = format!("model.layers.{layer}");
                    arrays.insert(format!("{prefix}.input_layernorm.weight"), ones(4));
                    arrays.insert(format!("{prefix}.post_attention_layernorm.weight"), ones(4));
                    let attn = format!("{prefix}.self_attn");
                    arrays.insert(format!("{attn}.q_a_proj.weight"), zeros(&[3, 4]));
                    arrays.insert(format!("{attn}.q_a_layernorm.weight"), ones(3));
                    arrays.insert(format!("{attn}.q_b_proj.weight"), zeros(&[4, 3]));
                    arrays.insert(format!("{attn}.kv_a_proj_with_mqa.weight"), zeros(&[5, 4]));
                    arrays.insert(format!("{attn}.kv_a_layernorm.weight"), ones(3));
                    arrays.insert(format!("{attn}.embed_q.weight"), zeros(&[1, 3, 2]));
                    arrays.insert(format!("{attn}.unembed_out.weight"), zeros(&[1, 2, 3]));
                    arrays.insert(format!("{attn}.o_proj.weight"), zeros(&[4, 2]));
                }
                arrays.insert(
                    "model.layers.0.mlp.gate_proj.weight".to_string(),
                    zeros(&[8, 4]),
                );
                arrays.insert(
                    "model.layers.0.mlp.up_proj.weight".to_string(),
                    zeros(&[8, 4]),
                );
                arrays.insert(
                    "model.layers.0.mlp.down_proj.weight".to_string(),
                    zeros(&[4, 8]),
                );
                arrays.insert("model.layers.1.mlp.gate.weight".to_string(), zeros(&[2, 4]));
                arrays.insert(
                    "model.layers.1.mlp.gate.e_score_correction_bias".to_string(),
                    zeros(&[2]),
                );
                for name in ["gate_proj", "up_proj"] {
                    arrays.insert(
                        format!("model.layers.1.mlp.switch_mlp.{name}.weight"),
                        zeros(&[2, 4, 4]),
                    );
                }
                arrays.insert(
                    "model.layers.1.mlp.switch_mlp.down_proj.weight".to_string(),
                    zeros(&[2, 4, 4]),
                );
                for name in ["gate_proj", "up_proj"] {
                    arrays.insert(
                        format!("model.layers.1.mlp.shared_experts.{name}.weight"),
                        zeros(&[4, 4]),
                    );
                }
                arrays.insert(
                    "model.layers.1.mlp.shared_experts.down_proj.weight".to_string(),
                    zeros(&[4, 4]),
                );
                Array::save_safetensors(&arrays, None, root.join("model.safetensors")).unwrap();
            }

            fn ones(len: usize) -> Array {
                Array::from_slice(&vec![1.0f32; len], &[len as i32])
            }

            fn zeros(shape: &[i32]) -> Array {
                let len = shape.iter().product::<i32>() as usize;
                Array::from_slice(&vec![0.0f32; len], shape)
            }
        }

        let gen_req = || super::GenerationRequest {
            prompt: "hello".to_string(),
            max_tokens: 3,
            temperature: 0.0,
            top_p: 1.0,
            top_k: None,
            seed: Some(42),
            stop_sequences: Vec::new(),
            media_inputs: Vec::new(),
        };

        // 1. Resident load (no streaming env var).
        let dir_resident = {
            let mut p = std::env::temp_dir();
            p.push(format!(
                "hi-mlx-stream-vs-resident-{}",
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ));
            p
        };
        build_fixture(&dir_resident);
        // Ensure streaming is off for the resident run.
        unsafe { std::env::remove_var(crate::expert_stream::EXPERT_STREAMING_ENV) };
        unsafe { std::env::remove_var(crate::expert_stream::EXPERT_POOL_BYTES_ENV) };
        let backend_resident =
            super::MlxBackend::load(&dir_resident, Some("tiny-glm-resident".to_string())).unwrap();
        let output_resident = super::InferenceBackend::generate(&backend_resident, gen_req())
            .await
            .unwrap();

        // 2. Streaming load (force on).
        let dir_stream = {
            let mut p = std::env::temp_dir();
            p.push(format!(
                "hi-mlx-stream-vs-streamed-{}",
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ));
            p
        };
        build_fixture(&dir_stream);
        unsafe { std::env::set_var(crate::expert_stream::EXPERT_STREAMING_ENV, "1") };
        let backend_stream =
            super::MlxBackend::load(&dir_stream, Some("tiny-glm-stream".to_string())).unwrap();
        let output_stream = super::InferenceBackend::generate(&backend_stream, gen_req())
            .await
            .unwrap();
        // Clean up env.
        unsafe { std::env::remove_var(crate::expert_stream::EXPERT_STREAMING_ENV) };

        // The streamed output must match the resident output exactly (greedy,
        // temperature=0, fixed seed → deterministic).
        assert_eq!(
            output_resident.text, output_stream.text,
            "streamed MoE output must match resident output"
        );
        assert_eq!(
            output_resident.completion_tokens, output_stream.completion_tokens,
            "streamed MoE token count must match resident"
        );
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64", feature = "mlx"))]
    #[tokio::test]
    async fn native_backend_generates_from_tiny_deepseek_derived_mla_fixture() {
        use std::collections::HashMap;
        use std::fs;
        use std::path::{Path, PathBuf};

        use mlx_rs::Array;
        use tokenizers::Tokenizer;
        use tokenizers::models::wordlevel::WordLevel;

        let _guard = MLX_TEST_LOCK.lock().await;
        let dir = tempfile_path("native-deepseek-mla");
        fs::create_dir_all(&dir).unwrap();
        write_config(&dir);
        write_tokenizer(&dir);
        write_weights(&dir);

        let backend = super::MlxBackend::load(&dir, Some("tiny-deepseek".to_string())).unwrap();
        let output = super::InferenceBackend::generate(
            &backend,
            super::GenerationRequest {
                prompt: "hello".to_string(),
                max_tokens: 2,
                temperature: 0.0,
                top_p: 1.0,
                top_k: None,
                seed: None,
                stop_sequences: Vec::new(),
                media_inputs: Vec::new(),
            },
        )
        .await
        .unwrap();

        assert_eq!(output.prompt_tokens, 1);
        assert_eq!(output.completion_tokens, 2);
        assert!(!output.text.trim().is_empty());

        fn write_config(root: &Path) {
            fs::write(
                root.join("config.json"),
                r#"{
                  "architectures": ["DeepseekV32ForCausalLM"],
                  "model_type": "deepseek_v32",
                  "hidden_size": 4,
                  "intermediate_size": 8,
                  "num_hidden_layers": 1,
                  "num_attention_heads": 1,
                  "num_key_value_heads": 1,
                  "qk_nope_head_dim": 2,
                  "qk_rope_head_dim": 2,
                  "v_head_dim": 2,
                  "q_lora_rank": 3,
                  "kv_lora_rank": 3,
                  "index_head_dim": 2,
                  "index_n_heads": 1,
                  "index_topk": 1,
                  "indexer_rope_interleave": false,
                  "vocab_size": 4,
                  "max_position_embeddings": 16,
                  "rms_norm_eps": 1e-6,
                  "rope_theta": 10000,
                  "tie_word_embeddings": false,
                  "eos_token_id": 99
                }"#,
            )
            .unwrap();
        }

        fn write_tokenizer(root: &Path) {
            let model = WordLevel::builder()
                .vocab(HashMap::from([
                    ("<unk>".to_string(), 0),
                    ("hello".to_string(), 1),
                    ("</s>".to_string(), 2),
                    ("world".to_string(), 3),
                ]))
                .unk_token("<unk>".to_string())
                .build()
                .unwrap();
            Tokenizer::new(model)
                .save(root.join("tokenizer.json"), false)
                .unwrap();
        }

        fn write_weights(root: &Path) {
            let mut arrays = HashMap::new();
            let vocab = [
                -1.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, -2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ];
            arrays.insert(
                "model.embed_tokens.weight".to_string(),
                Array::from_slice(&vocab, &[4, 4]),
            );
            arrays.insert(
                "lm_head.weight".to_string(),
                Array::from_slice(&vocab, &[4, 4]),
            );
            arrays.insert("model.norm.weight".to_string(), ones(4));
            arrays.insert("model.layers.0.input_layernorm.weight".to_string(), ones(4));
            arrays.insert(
                "model.layers.0.post_attention_layernorm.weight".to_string(),
                ones(4),
            );
            let attn = "model.layers.0.self_attn";
            arrays.insert(format!("{attn}.q_a_proj.weight"), zeros(&[3, 4]));
            arrays.insert(format!("{attn}.q_a_layernorm.weight"), ones(3));
            arrays.insert(format!("{attn}.q_b_proj.weight"), zeros(&[4, 3]));
            arrays.insert(format!("{attn}.kv_a_proj_with_mqa.weight"), zeros(&[5, 4]));
            arrays.insert(format!("{attn}.kv_a_layernorm.weight"), ones(3));
            arrays.insert(format!("{attn}.kv_b_proj.weight"), zeros(&[4, 3]));
            arrays.insert(format!("{attn}.o_proj.weight"), zeros(&[4, 2]));
            arrays.insert(format!("{attn}.indexer.wq_b.weight"), zeros(&[2, 3]));
            arrays.insert(format!("{attn}.indexer.wk.weight"), zeros(&[2, 4]));
            arrays.insert(format!("{attn}.indexer.k_norm.weight"), ones(2));
            arrays.insert(format!("{attn}.indexer.k_norm.bias"), zeros(&[2]));
            arrays.insert(
                format!("{attn}.indexer.weights_proj.weight"),
                zeros(&[1, 4]),
            );
            arrays.insert(
                "model.layers.0.mlp.gate_proj.weight".to_string(),
                zeros(&[8, 4]),
            );
            arrays.insert(
                "model.layers.0.mlp.up_proj.weight".to_string(),
                zeros(&[8, 4]),
            );
            arrays.insert(
                "model.layers.0.mlp.down_proj.weight".to_string(),
                zeros(&[4, 8]),
            );
            Array::save_safetensors(&arrays, None, root.join("model.safetensors")).unwrap();
        }

        fn ones(len: usize) -> Array {
            Array::from_slice(&vec![1.0f32; len], &[len as i32])
        }

        fn zeros(shape: &[i32]) -> Array {
            let len = shape.iter().product::<i32>() as usize;
            Array::from_slice(&vec![0.0f32; len], shape)
        }

        fn tempfile_path(name: &str) -> PathBuf {
            let mut path = std::env::temp_dir();
            path.push(format!(
                "hi-mlx-{name}-{}",
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ));
            path
        }
    }

    fn test_model() -> ModelInfo {
        let dir = tempfile_path("backend-model");
        test_support::write_qwen_fixture(&dir);
        inspect_model(&dir, None).unwrap()
    }

    fn tempfile_path(name: &str) -> std::path::PathBuf {
        let mut path = std::env::temp_dir();
        path.push(format!(
            "hi-mlx-{name}-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        path
    }

    /// End-to-end test of MoE expert streaming auto-enable on a "RAM-constrained"
    /// GLM MoE model. We build a small GLM MoE fixture with *non-trivial* expert
    /// weights (distinct per-expert values so routing matters), then artificially
    /// lower the memory budget via `HI_MLX_MEMORY_LIMIT_BYTES` so the planner's
    /// auto-enable heuristic fires (trunk + experts > limit). We then:
    ///
    /// 1. Assert the load succeeds (streaming kicks in instead of OOM).
    /// 2. Assert generation produces coherent output (non-empty, correct tokens).
    /// 3. Assert the streamed output matches the resident output (correctness).
    ///
    /// This mirrors the real-world scenario of a GLM model that won't fit in RAM,
    /// but uses a tiny fixture + a lowered budget so it runs in CI.
    #[cfg(all(target_os = "macos", target_arch = "aarch64", feature = "mlx"))]
    #[tokio::test]
    async fn moe_streaming_auto_enable_on_ram_constrained_glm() {
        use std::collections::HashMap;
        use std::fs;
        use std::path::Path;

        use mlx_rs::Array;
        use tokenizers::Tokenizer;
        use tokenizers::models::wordlevel::WordLevel;

        let _guard = MLX_TEST_LOCK.lock().await;

        // Build two identical fixtures: one for resident load, one for the
        // RAM-constrained (streaming) load. We need separate dirs because the
        // slab reader opens shard files and we don't want handle contention.
        fn build_fixture(root: &Path) {
            fs::create_dir_all(root).unwrap();
            write_config(root);
            write_tokenizer(root);
            write_weights(root);

            fn write_config(root: &Path) {
                fs::write(
                    root.join("config.json"),
                    r#"{
                      "architectures": ["Glm4MoeLiteForCausalLM"],
                      "model_type": "glm4_moe_lite",
                      "hidden_size": 4,
                      "intermediate_size": 8,
                      "moe_intermediate_size": 4,
                      "num_hidden_layers": 2,
                      "num_attention_heads": 1,
                      "num_key_value_heads": 1,
                      "qk_nope_head_dim": 2,
                      "qk_rope_head_dim": 2,
                      "v_head_dim": 2,
                      "q_lora_rank": 3,
                      "kv_lora_rank": 3,
                      "vocab_size": 4,
                      "max_position_embeddings": 16,
                      "rms_norm_eps": 1e-6,
                      "rope_theta": 1000000,
                      "tie_word_embeddings": false,
                      "first_k_dense_replace": 1,
                      "moe_layer_freq": 1,
                      "n_routed_experts": 2,
                      "n_shared_experts": 1,
                      "num_experts_per_tok": 1,
                      "n_group": 1,
                      "topk_group": 1,
                      "topk_method": "noaux_tc",
                      "eos_token_id": 99
                    }"#,
                )
                .unwrap();
            }

            fn write_tokenizer(root: &Path) {
                let model = WordLevel::builder()
                    .vocab(HashMap::from([
                        ("<unk>".to_string(), 0),
                        ("hello".to_string(), 1),
                        ("</s>".to_string(), 2),
                        ("world".to_string(), 3),
                    ]))
                    .unk_token("<unk>".to_string())
                    .build()
                    .unwrap();
                Tokenizer::new(model)
                    .save(root.join("tokenizer.json"), false)
                    .unwrap();
            }

            fn write_weights(root: &Path) {
                let mut arrays = HashMap::new();
                // Embedding: distinct values per token so routing produces
                // different experts for different inputs.
                let vocab = [
                    -1.0f32, 0.0, 0.0, 0.0, // token 0 → negative
                    10.0, 0.0, 0.0, 0.0, // token 1 → large positive
                    -2.0, 0.0, 0.0, 0.0, // token 2
                    0.0, 0.0, 0.0, 0.0, // token 3
                ];
                arrays.insert(
                    "model.embed_tokens.weight".to_string(),
                    Array::from_slice(&vocab, &[4, 4]),
                );
                arrays.insert(
                    "lm_head.weight".to_string(),
                    Array::from_slice(&vocab, &[4, 4]),
                );
                arrays.insert("model.norm.weight".to_string(), ones(4));
                for layer in 0..2 {
                    let prefix = format!("model.layers.{layer}");
                    arrays.insert(format!("{prefix}.input_layernorm.weight"), ones(4));
                    arrays.insert(format!("{prefix}.post_attention_layernorm.weight"), ones(4));
                    let attn = format!("{prefix}.self_attn");
                    arrays.insert(format!("{attn}.q_a_proj.weight"), zeros(&[3, 4]));
                    arrays.insert(format!("{attn}.q_a_layernorm.weight"), ones(3));
                    arrays.insert(format!("{attn}.q_b_proj.weight"), zeros(&[4, 3]));
                    arrays.insert(format!("{attn}.kv_a_proj_with_mqa.weight"), zeros(&[5, 4]));
                    arrays.insert(format!("{attn}.kv_a_layernorm.weight"), ones(3));
                    arrays.insert(format!("{attn}.embed_q.weight"), zeros(&[1, 3, 2]));
                    arrays.insert(format!("{attn}.unembed_out.weight"), zeros(&[1, 2, 3]));
                    arrays.insert(format!("{attn}.o_proj.weight"), zeros(&[4, 2]));
                }
                // Layer 0: dense MLP.
                arrays.insert(
                    "model.layers.0.mlp.gate_proj.weight".to_string(),
                    zeros(&[8, 4]),
                );
                arrays.insert(
                    "model.layers.0.mlp.up_proj.weight".to_string(),
                    zeros(&[8, 4]),
                );
                arrays.insert(
                    "model.layers.0.mlp.down_proj.weight".to_string(),
                    zeros(&[4, 8]),
                );
                // Layer 1: MoE with non-trivial expert weights.
                // Gate: route token 0 → expert 0, token 1 → expert 1.
                arrays.insert(
                    "model.layers.1.mlp.gate.weight".to_string(),
                    // [2 experts, 4 hidden] — expert 0 prefers negative inputs,
                    // expert 1 prefers positive inputs.
                    Array::from_slice(&[-1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], &[2, 4]),
                );
                arrays.insert(
                    "model.layers.1.mlp.gate.e_score_correction_bias".to_string(),
                    zeros(&[2]),
                );
                // Expert weights: distinct per-expert so the routed expert
                // produces a distinguishable output. Expert 0 → output token 2,
                // expert 1 → output token 3.
                for name in ["gate_proj", "up_proj"] {
                    arrays.insert(
                        format!("model.layers.1.mlp.switch_mlp.{name}.weight"),
                        // Expert 0: identity-ish (4→4), expert 1: scaled.
                        Array::from_slice(
                            &[
                                // expert 0: [4, 4] identity
                                1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
                                0.0, 0.0, 1.0, // expert 1: [4, 4] 2× identity
                                2.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0,
                                0.0, 0.0, 2.0,
                            ],
                            &[2, 4, 4],
                        ),
                    );
                }
                arrays.insert(
                    "model.layers.1.mlp.switch_mlp.down_proj.weight".to_string(),
                    // down_proj: [2, 4, 4] — expert 0 maps to token 2, expert 1
                    // maps to token 3.
                    Array::from_slice(
                        &[
                            // expert 0: [4, 4] → maps hidden[0] to output
                            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                            0.0, 0.0, // expert 1: [4, 4] → maps hidden[1] to output
                            0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                            0.0, 0.0,
                        ],
                        &[2, 4, 4],
                    ),
                );
                // Shared expert.
                for name in ["gate_proj", "up_proj"] {
                    arrays.insert(
                        format!("model.layers.1.mlp.shared_experts.{name}.weight"),
                        zeros(&[4, 4]),
                    );
                }
                arrays.insert(
                    "model.layers.1.mlp.shared_experts.down_proj.weight".to_string(),
                    zeros(&[4, 4]),
                );
                Array::save_safetensors(&arrays, None, root.join("model.safetensors")).unwrap();
            }

            fn ones(len: usize) -> Array {
                Array::from_slice(&vec![1.0f32; len], &[len as i32])
            }

            fn zeros(shape: &[i32]) -> Array {
                let len = shape.iter().product::<i32>() as usize;
                Array::from_slice(&vec![0.0f32; len], shape)
            }
        }

        let gen_req = || super::GenerationRequest {
            prompt: "hello".to_string(),
            max_tokens: 3,
            temperature: 0.0,
            top_p: 1.0,
            top_k: None,
            seed: Some(42),
            stop_sequences: Vec::new(),
            media_inputs: Vec::new(),
        };

        // --- 1. Resident load (no streaming, no memory limit) ---
        let dir_resident = tempfile_path("e2e-glm-resident");
        build_fixture(&dir_resident);
        // Ensure streaming env vars are clean for the resident run.
        unsafe { std::env::remove_var(crate::expert_stream::EXPERT_STREAMING_ENV) };
        unsafe { std::env::remove_var(crate::expert_stream::EXPERT_POOL_BYTES_ENV) };
        unsafe { std::env::remove_var(MEMORY_LIMIT_BYTES_ENV) };
        let backend_resident =
            super::MlxBackend::load(&dir_resident, Some("glm-resident".to_string())).unwrap();
        let output_resident = super::InferenceBackend::generate(&backend_resident, gen_req())
            .await
            .unwrap();
        assert!(
            !output_resident.text.trim().is_empty(),
            "resident output should be non-empty"
        );

        // --- 2. RAM-constrained load (auto-enable streaming) ---
        // Set the memory limit to 1 byte — the model (even though tiny) will
        // exceed it, so the auto-enable heuristic fires and streaming kicks in.
        let dir_stream = tempfile_path("e2e-glm-streamed");
        build_fixture(&dir_stream);
        unsafe { std::env::set_var(MEMORY_LIMIT_BYTES_ENV, "1") };
        // Don't force streaming — let auto-enable do its job.
        unsafe { std::env::remove_var(crate::expert_stream::EXPERT_STREAMING_ENV) };
        unsafe { std::env::remove_var(crate::expert_stream::EXPERT_POOL_BYTES_ENV) };
        let backend_stream =
            super::MlxBackend::load(&dir_stream, Some("glm-streamed".to_string())).unwrap();
        let output_stream = super::InferenceBackend::generate(&backend_stream, gen_req())
            .await
            .unwrap();
        // Clean up env.
        unsafe { std::env::remove_var(MEMORY_LIMIT_BYTES_ENV) };

        // --- 3. Assertions ---
        // The streamed output must be non-empty (coherent generation through the
        // streaming pool).
        assert!(
            !output_stream.text.trim().is_empty(),
            "streamed output should be non-empty"
        );
        assert!(
            output_stream.completion_tokens > 0,
            "streamed output should produce tokens"
        );
        // The streamed output must match the resident output exactly (greedy,
        // temperature=0, fixed seed → deterministic; the streaming pool is
        // invisible to the math).
        assert_eq!(
            output_resident.text, output_stream.text,
            "streamed MoE output must match resident output (RAM-constrained auto-enable)"
        );
        assert_eq!(
            output_resident.completion_tokens, output_stream.completion_tokens,
            "streamed MoE token count must match resident"
        );
    }

    /// Verify the **batched gather** MoE path (`SwitchLinear::gather` →
    /// `forward_batched`) produces identical output whether experts are resident
    /// or streamed through the on-demand pool. This covers the streaming branch
    /// added to `gather`: when a `SwitchLinear` is pool-backed, the batched
    /// `gather_qmm_mode` cannot run on the 0-element placeholder weights, so the
    /// streaming path decomposes the gather into per-expert `forward_expert`
    /// calls. The Qwen3-MoE fixture uses quantized expert weights (scales +
    /// biases present), which routes through `forward_batched` rather than the
    /// per-expert loop.
    #[cfg(all(target_os = "macos", target_arch = "aarch64", feature = "mlx"))]
    #[tokio::test]
    async fn expert_streaming_batched_gather_matches_resident_output() {
        use std::collections::HashMap;
        use std::fs;
        use std::path::Path;

        use mlx_rs::Array;
        use tokenizers::Tokenizer;
        use tokenizers::models::wordlevel::WordLevel;

        let _guard = MLX_TEST_LOCK.lock().await;

        // Build a Qwen3-MoE fixture with *quantized* switch experts so the
        // forward pass takes the `forward_batched` → `gather` path (the branch
        // at `scales.is_some()`), not the dense per-expert fallback.
        fn build_fixture(dir: &Path) {
            fs::create_dir_all(dir).unwrap();
            write_config(dir);
            write_tokenizer(dir);
            write_weights(dir);

            fn write_config(root: &Path) {
                fs::write(
                    root.join("config.json"),
                    r#"{
                      "architectures": ["Qwen3MoeForCausalLM"],
                      "model_type": "qwen3_moe",
                      "hidden_size": 32,
                      "intermediate_size": 64,
                      "moe_intermediate_size": 32,
                      "num_hidden_layers": 1,
                      "num_attention_heads": 1,
                      "num_key_value_heads": 1,
                      "head_dim": 32,
                      "num_experts": 2,
                      "num_experts_per_tok": 1,
                      "decoder_sparse_step": 1,
                      "mlp_only_layers": [],
                      "norm_topk_prob": true,
                      "vocab_size": 4,
                      "max_position_embeddings": 16,
                      "rms_norm_eps": 1e-6,
                      "rope_theta": 1000000,
                      "tie_word_embeddings": true,
                      "eos_token_id": 99,
                      "quantization": {
                        "group_size": 32,
                        "bits": 4,
                        "mode": "affine"
                      }
                    }"#,
                )
                .unwrap();
            }

            fn write_tokenizer(root: &Path) {
                let model = WordLevel::builder()
                    .vocab(HashMap::from([
                        ("<unk>".to_string(), 0),
                        ("hello".to_string(), 1),
                        ("</s>".to_string(), 2),
                        ("world".to_string(), 3),
                    ]))
                    .unk_token("<unk>".to_string())
                    .build()
                    .unwrap();
                Tokenizer::new(model)
                    .save(root.join("tokenizer.json"), false)
                    .unwrap();
            }

            fn write_weights(root: &Path) {
                use mlx_rs::ops::quantize;
                let mut arrays = HashMap::new();
                // Embedding / lm_head (tied). Make token 1 ("hello") produce a
                // strong logit for token 2 so generation is deterministic.
                let mut vocab = vec![0.0f32; 4 * 32];
                // token 1 embedding → bias token 2's logit high.
                vocab[1 * 32 + 2] = 10.0;
                arrays.insert(
                    "model.embed_tokens.weight".to_string(),
                    Array::from_slice(&vocab, &[4, 32]),
                );
                arrays.insert("model.norm.weight".to_string(), ones(32));
                // Single MoE layer.
                let prefix = "model.layers.0";
                arrays.insert(format!("{prefix}.input_layernorm.weight"), ones(32));
                arrays.insert(
                    format!("{prefix}.post_attention_layernorm.weight"),
                    ones(32),
                );
                // Attention (GQA, 1 head, head_dim=32).
                let attn = format!("{prefix}.self_attn");
                arrays.insert(format!("{attn}.q_proj.weight"), zeros(&[32, 32]));
                arrays.insert(format!("{attn}.k_proj.weight"), zeros(&[32, 32]));
                arrays.insert(format!("{attn}.v_proj.weight"), zeros(&[32, 32]));
                arrays.insert(format!("{attn}.o_proj.weight"), zeros(&[32, 32]));
                // Router gate: 2 experts.
                arrays.insert(format!("{prefix}.mlp.gate.weight"), zeros(&[2, 32]));
                // Quantized switch experts: f32 [2, 32, 32] → quantize with
                // group_size=32, bits=4. One group per row.
                for name in ["gate_proj", "up_proj", "down_proj"] {
                    let w_f32 = zeros(&[2, 32, 32]);
                    let (w_q, scales, biases) = quantize(&w_f32, 32, 4).unwrap();
                    arrays.insert(format!("{prefix}.mlp.switch_mlp.{name}.weight"), w_q);
                    arrays.insert(format!("{prefix}.mlp.switch_mlp.{name}.scales"), scales);
                    arrays.insert(format!("{prefix}.mlp.switch_mlp.{name}.biases"), biases);
                }
                // Shared expert (dense).
                for name in ["gate_proj", "up_proj"] {
                    arrays.insert(
                        format!("{prefix}.mlp.shared_expert.{name}.weight"),
                        zeros(&[32, 32]),
                    );
                }
                arrays.insert(
                    format!("{prefix}.mlp.shared_expert.down_proj.weight"),
                    zeros(&[32, 32]),
                );
                Array::save_safetensors(&arrays, None, root.join("model.safetensors")).unwrap();
            }

            fn ones(len: usize) -> Array {
                Array::from_slice(&vec![1.0f32; len], &[len as i32])
            }

            fn zeros(shape: &[i32]) -> Array {
                let len = shape.iter().product::<i32>() as usize;
                Array::from_slice(&vec![0.0f32; len], shape)
            }
        }

        let gen_req = || super::GenerationRequest {
            prompt: "hello".to_string(),
            max_tokens: 3,
            temperature: 0.0,
            top_p: 1.0,
            top_k: None,
            seed: Some(42),
            stop_sequences: Vec::new(),
            media_inputs: Vec::new(),
        };

        // 1. Resident load (no streaming env var).
        let dir_resident = {
            let mut p = std::env::temp_dir();
            p.push(format!(
                "hi-mlx-batched-gather-resident-{}",
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ));
            p
        };
        build_fixture(&dir_resident);
        unsafe { std::env::remove_var(crate::expert_stream::EXPERT_STREAMING_ENV) };
        unsafe { std::env::remove_var(crate::expert_stream::EXPERT_POOL_BYTES_ENV) };
        let backend_resident = super::MlxBackend::load(
            &dir_resident,
            Some("tiny-qwen-moe-batched-resident".to_string()),
        )
        .unwrap();
        let output_resident = super::InferenceBackend::generate(&backend_resident, gen_req())
            .await
            .unwrap();

        // 2. Streaming load (force on) — exercises `gather_streaming`.
        let dir_stream = {
            let mut p = std::env::temp_dir();
            p.push(format!(
                "hi-mlx-batched-gather-stream-{}",
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ));
            p
        };
        build_fixture(&dir_stream);
        unsafe { std::env::set_var(crate::expert_stream::EXPERT_STREAMING_ENV, "1") };
        let backend_stream = super::MlxBackend::load(
            &dir_stream,
            Some("tiny-qwen-moe-batched-stream".to_string()),
        )
        .unwrap();
        let output_stream = super::InferenceBackend::generate(&backend_stream, gen_req())
            .await
            .unwrap();
        unsafe { std::env::remove_var(crate::expert_stream::EXPERT_STREAMING_ENV) };

        assert_eq!(
            output_resident.text, output_stream.text,
            "streamed batched-gather MoE output must match resident output"
        );
        assert_eq!(
            output_resident.completion_tokens, output_stream.completion_tokens,
            "streamed batched-gather MoE token count must match resident"
        );
    }
}
