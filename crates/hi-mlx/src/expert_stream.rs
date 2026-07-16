//! MoE expert-streaming planning for the MLX backend.
//!
//! On Apple Silicon, CPU and GPU share unified memory (the MLX memory budget is
//! `sysctl hw.memsize × fraction`, see `backend::configured_memory_limit_bytes`).
//! There is no discrete-VRAM wall like CUDA's, so "expert streaming" here solves
//! a different, softer problem: a MoE whose routed-expert tensors plus the trunk
//! exceed the configured safe memory limit. When that happens we keep the expert
//! tensors out of the resident set and page individual expert slabs in on demand
//! through a bounded host-RAM pool, instead of refusing to load the model.
//!
//! This module is the **planning layer only** — pure Rust, no MLX FFI — so it
//! unit-tests on arm64-macOS without a GPU. It mirrors the CUDA planning layer
//! (`hi-cuda/src/gpu.rs::auto_expert_streaming_budget` + `collect_expert_sources`)
//! but is adapted to MLX's safetensors layout and the unified-memory budget.
//!
//! The on-demand slab loader (lazy MLX arrays over mapped safetensors extents)
//! is a later workstream; this module produces the plan it will consume.

use std::collections::BTreeMap;
use std::path::PathBuf;

use anyhow::{Context, Result, bail};

use crate::config::MlxModelConfig;
use crate::weights::WeightCatalog;

/// Env var: `1` forces streaming on; `0` forces it off (resident load, fails
/// honestly if it doesn't fit). Unset = auto (stream only when it won't fit).
pub const EXPERT_STREAMING_ENV: &str = "HI_MLX_EXPERT_STREAMING";

/// Env var: overrides the auto-derived pool budget, in bytes. Beats the derived
/// size whether or not `HI_MLX_EXPERT_STREAMING` is set (mirrors CUDA).
pub const EXPERT_POOL_BYTES_ENV: &str = "HI_MLX_EXPERT_POOL_BYTES";

/// The three routed-expert projections in a `SwitchMlp`, in execution order.
/// Each has a stacked `[num_experts, out, in]` `weight` (and optional
/// `scales`/`biases` of matching leading dimension).
pub const EXPERT_PROJECTIONS: &[&str] = &["gate_proj", "up_proj", "down_proj"];

/// A streamable expert tensor group: one (layer, projection) pair and the
/// weight/scales/biases tensor names that belong to it, plus the shard file
/// each lives in and the byte extent within that shard.
#[derive(Clone, Debug)]
pub struct ExpertSource {
    pub layer: u32,
    pub projection: &'static str,
    pub weight_name: String,
    pub scales_name: Option<String>,
    pub biases_name: Option<String>,
    /// Which safetensors shard file holds the **weight** tensor (relative to
    /// model root). Scales/biases may live in different shards — see
    /// `scales_shard` / `biases_shard`.
    pub shard_file: String,
    /// Shard file for the scales tensor (if present and in a different shard).
    pub scales_shard: Option<String>,
    /// Shard file for the biases tensor (if present and in a different shard).
    pub biases_shard: Option<String>,
    /// Total bytes across weight + scales + biases for this (layer, projection).
    pub bytes: u64,
}

impl ExpertSource {
    /// A stable key for dedup / indexing: `(layer, projection)`.
    pub fn key(&self) -> (u32, &'static str) {
        (self.layer, self.projection)
    }
}

/// The byte split between the non-expert trunk and the routed-expert tensors,
/// plus the list of streamable expert sources. Produced by a single walk of the
/// safetensors index.
#[derive(Clone, Debug)]
pub struct ExpertStreamPlan {
    /// Bytes of all tensors that are *not* routed-expert weights/scales/biases.
    /// These load resident regardless of streaming.
    pub trunk_bytes: u64,
    /// Bytes of the routed-expert tensors (weight + scales + biases) that
    /// streaming would keep out of the resident set.
    pub expert_bytes: u64,
    /// One entry per (layer, projection) that has a stacked expert tensor.
    pub sources: Vec<ExpertSource>,
    /// Number of MoE layers detected (layers with at least one expert group).
    pub moe_layers: u32,
}

impl ExpertStreamPlan {
    pub fn expert_count_groups(&self) -> usize {
        self.sources.len()
    }

    /// True iff there are streamable expert tensors. When false, streaming is a
    /// no-op and the resident load path should be used.
    pub fn has_experts(&self) -> bool {
        !self.sources.is_empty()
    }
}

/// The auto-enable decision: whether to stream, and if so the pool budget in
/// bytes the on-demand loader should cap itself to.
#[derive(Clone, Debug)]
pub struct StreamingDecision {
    pub stream: bool,
    pub pool_bytes: u64,
    /// Why this decision was reached (for the load-time log line).
    pub reason: String,
}

/// Classify a tensor name as a routed-expert weight/scales/biases tensor and, if
/// so, return `(layer, projection, suffix)` where suffix is "weight" | "scales"
/// | "biases". Returns `None` for non-expert tensors.
///
/// Recognizes the `model.layers.{N}.mlp.switch_mlp.{proj}.{suffix}` layout used
/// by the DeepSeek/GLM/Qwen3-MoE `SwitchMlp` path (`models.rs`). The leading
/// `language_model.` prefix (VL models, stripped by `load_arrays`) is tolerated.
pub fn classify_expert_tensor(name: &str) -> Option<(u32, &'static str, &'static str)> {
    let stripped = name.strip_prefix("language_model.").unwrap_or(name);
    let rest = stripped.strip_prefix("model.layers.")?;
    let (layer_part, tail) = rest.split_once('.')?;
    let layer: u32 = layer_part.parse().ok()?;
    let tail = tail.strip_prefix("mlp.switch_mlp.")?;
    for proj in EXPERT_PROJECTIONS {
        if let Some(suffix) = tail.strip_prefix(&format!("{proj}.")) {
            let suffix_static = match suffix {
                "weight" => "weight",
                "scales" => "scales",
                "biases" => "biases",
                _ => return None,
            };
            return Some((layer, proj_static(proj), suffix_static));
        }
    }
    None
}

/// Resolve a `&str` projection name to its `&'static str` counterpart from
/// [`EXPERT_PROJECTIONS`].
fn proj_static(name: &str) -> &'static str {
    EXPERT_PROJECTIONS
        .iter()
        .copied()
        .find(|p| *p == name)
        .expect("EXPERT_PROJECTIONS contains the matched name")
}

/// Build the stream plan by walking the catalog's tensor index. Each tensor is
/// classified as expert (weight/scales/biases of a `switch_mlp` projection) or
/// trunk. Expert byte extents are summed per (layer, projection) so the
/// on-demand loader knows exactly how many bytes each slab group costs.
///
/// Byte sizes come from the safetensors shard headers (the catalog already
/// reads them for `estimated_bytes`); we re-read the per-tensor extents here to
/// attribute bytes to individual tensors rather than whole shards.
pub fn build_plan(catalog: &WeightCatalog, _config: &MlxModelConfig) -> Result<ExpertStreamPlan> {
    // Per-tensor byte extents + shard file, keyed by tensor name, by reading
    // each shard's safetensors header. This gives us both the byte size and
    // which shard file the tensor lives in (for the on-demand slab reader).
    let tensor_layouts = per_tensor_layout(catalog)?;

    // Accumulate expert sources keyed by (layer, projection).
    let mut sources: BTreeMap<(u32, &'static str), ExpertSource> = BTreeMap::new();
    let mut expert_bytes = 0u64;
    let mut trunk_bytes = 0u64;
    let mut moe_layers: BTreeSet<u32> = BTreeSet::new();

    for name in &catalog.tensors {
        let (bytes, shard_file) = tensor_layouts
            .get(name)
            .map(|(b, s)| (*b, s.clone()))
            .unwrap_or((0, String::new()));
        match classify_expert_tensor(name) {
            Some((layer, proj, suffix)) => {
                moe_layers.insert(layer);
                expert_bytes = expert_bytes.saturating_add(bytes);
                let entry = sources
                    .entry((layer, proj))
                    .or_insert_with(|| ExpertSource {
                        layer,
                        projection: proj,
                        weight_name: String::new(),
                        scales_name: None,
                        biases_name: None,
                        shard_file: String::new(),
                        scales_shard: None,
                        biases_shard: None,
                        bytes: 0,
                    });
                match suffix {
                    "weight" => {
                        entry.weight_name = name.clone();
                        entry.shard_file = shard_file;
                    }
                    "scales" => {
                        entry.scales_name = Some(name.clone());
                        entry.scales_shard = Some(shard_file);
                    }
                    "biases" => {
                        entry.biases_name = Some(name.clone());
                        entry.biases_shard = Some(shard_file);
                    }
                    _ => {}
                }
                entry.bytes = entry.bytes.saturating_add(bytes);
            }
            None => {
                trunk_bytes = trunk_bytes.saturating_add(bytes);
            }
        }
    }

    Ok(ExpertStreamPlan {
        trunk_bytes,
        expert_bytes,
        sources: sources.into_values().collect(),
        moe_layers: moe_layers.len() as u32,
    })
}

/// Read per-tensor byte sizes and shard file names from every shard's safetensors
/// header. Returns a map from tensor name to `(byte length, shard file name)`.
fn per_tensor_layout(catalog: &WeightCatalog) -> Result<BTreeMap<String, (u64, String)>> {
    use std::fs;
    use std::io::Read;

    let mut out = BTreeMap::new();
    for shard in &catalog.shards {
        let path: PathBuf = catalog.root.join(&shard.path);
        let mut file =
            fs::File::open(&path).with_context(|| format!("opening shard {}", path.display()))?;
        let mut len = [0u8; 8];
        file.read_exact(&mut len)?;
        let header_len = u64::from_le_bytes(len);
        let header_len = usize::try_from(header_len).context("safetensors header too large")?;
        let mut header = vec![0u8; header_len];
        file.read_exact(&mut header)?;
        let value: serde_json::Value = serde_json::from_slice(&header)
            .with_context(|| format!("parsing safetensors header {}", path.display()))?;
        let obj = value
            .as_object()
            .ok_or_else(|| anyhow::anyhow!("safetensors header is not an object"))?;
        for (name, info) in obj {
            if name == "__metadata__" {
                continue;
            }
            let offsets = info
                .get("data_offsets")
                .and_then(|v| v.as_array())
                .ok_or_else(|| {
                    anyhow::anyhow!("tensor {name} missing data_offsets in {}", path.display())
                })?;
            if offsets.len() != 2 {
                bail!("tensor {name} has malformed data_offsets");
            }
            let start = offsets[0].as_u64().unwrap_or(0);
            let end = offsets[1].as_u64().unwrap_or(0);
            out.insert(
                name.clone(),
                (end.saturating_sub(start), shard.path.clone()),
            );
        }
    }
    Ok(out)
}

/// Decide whether to stream, and at what pool budget.
///
/// Mirrors CUDA's `auto_expert_streaming_budget`:
/// - `HI_MLX_EXPERT_STREAMING=0` → hard off, resident load (returns `stream:
///   false` even if it won't fit; the caller's admission check will then fail
///   honestly).
/// - `HI_MLX_EXPERT_STREAMING=1` (or any non-"0" value) → force on.
/// - unset → auto: stream iff `trunk + experts > memory_limit`.
/// - `HI_MLX_EXPERT_POOL_BYTES` overrides the derived pool budget.
///
/// `memory_limit_bytes` is the configured safe MLX memory budget from
/// `backend::configured_memory_limit_bytes` (host RAM × fraction). When it is
/// `None` (host RAM unknown), auto falls back to "don't stream" — there's no
/// budget to check against.
pub fn decide(plan: &ExpertStreamPlan, memory_limit_bytes: Option<u64>) -> StreamingDecision {
    let hard_off = streaming_hard_off();
    let forced_on = streaming_enabled();
    let pool_override = expert_pool_budget_bytes();

    // No streamable experts → nothing to do, regardless of env vars.
    if !plan.has_experts() {
        return StreamingDecision {
            stream: false,
            pool_bytes: 0,
            reason: "no routed-expert tensors found".to_string(),
        };
    }

    // Hard off wins over everything: respect the explicit resident request.
    if hard_off {
        return StreamingDecision {
            stream: false,
            pool_bytes: 0,
            reason: format!(
                "HI_MLX_EXPERT_STREAMING=0 forces resident (experts {:.1} GiB may not fit)",
                gib(plan.expert_bytes)
            ),
        };
    }

    // Auto-enable: stream only when the model won't fit the memory budget.
    let need = plan.trunk_bytes.saturating_add(plan.expert_bytes);
    let fits = match memory_limit_bytes {
        Some(limit) => need <= limit,
        None => true, // unknown budget → assume it fits, don't stream.
    };

    let auto_stream = !fits;
    let stream = forced_on || auto_stream;
    if !stream {
        return StreamingDecision {
            stream: false,
            pool_bytes: 0,
            reason: format!(
                "model fits resident (trunk {:.1} GiB + experts {:.1} GiB = {:.1} GiB <= limit {})",
                gib(plan.trunk_bytes),
                gib(plan.expert_bytes),
                gib(need),
                memory_limit_bytes
                    .map(|b| format!("{:.1} GiB", gib(b)))
                    .unwrap_or_else(|| "unknown".to_string())
            ),
        };
    }

    // Derive the pool budget. The override wins; otherwise auto-size from the
    // memory limit minus the trunk (leave room for the resident trunk + a
    // margin). A floor of one expert group keeps the pool useful.
    let pool_bytes = if let Some(b) = pool_override {
        b
    } else {
        auto_pool_budget(plan, memory_limit_bytes)
    };

    let reason = if forced_on {
        format!(
            "HI_MLX_EXPERT_STREAMING forced on; experts {:.1} GiB through a {:.1} GiB pool",
            gib(plan.expert_bytes),
            gib(pool_bytes)
        )
    } else {
        format!(
            "experts {:.1} GiB + trunk {:.1} GiB = {:.1} GiB exceed limit {}; streaming through a {:.1} GiB pool",
            gib(plan.expert_bytes),
            gib(plan.trunk_bytes),
            gib(need),
            memory_limit_bytes
                .map(|b| format!("{:.1} GiB", gib(b)))
                .unwrap_or_else(|| "unknown".to_string()),
            gib(pool_bytes)
        )
    };

    StreamingDecision {
        stream: true,
        pool_bytes,
        reason,
    }
}

/// Auto-size the pool: a fraction of the space left after the trunk, clamped to
/// `[one_expert_group, expert_bytes]`. We use half the leftover (the other half
/// goes to KV cache, activations, and OS/other processes) — the same 0.5
/// fraction CUDA's RAM tier uses for its host budget.
fn auto_pool_budget(plan: &ExpertStreamPlan, memory_limit_bytes: Option<u64>) -> u64 {
    let one_group = plan
        .sources
        .iter()
        .map(|s| s.bytes)
        .min()
        .unwrap_or(0)
        .max(1);
    let Some(limit) = memory_limit_bytes else {
        return one_group;
    };
    let leftover = limit.saturating_sub(plan.trunk_bytes);
    let half = leftover / 2;
    // Clamp: at least one expert group, at most all expert bytes.
    half.clamp(one_group, plan.expert_bytes.max(1))
}

/// `HI_MLX_EXPERT_STREAMING` set and ≠ "0" → force on.
fn streaming_enabled() -> bool {
    std::env::var(EXPERT_STREAMING_ENV).is_ok_and(|v| v != "0")
}

/// `HI_MLX_EXPERT_STREAMING == "0"` → hard off.
fn streaming_hard_off() -> bool {
    std::env::var(EXPERT_STREAMING_ENV).is_ok_and(|v| v == "0")
}

/// `HI_MLX_EXPERT_POOL_BYTES` parsed as bytes, if set and > 0.
fn expert_pool_budget_bytes() -> Option<u64> {
    std::env::var(EXPERT_POOL_BYTES_ENV)
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .filter(|b| *b > 0)
}

fn gib(bytes: u64) -> f64 {
    bytes as f64 / (1u64 << 30) as f64
}

// Use a BTreeSet for moe_layers counting.
use std::collections::BTreeSet;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::parse_model_config;
    use serde_json::json;
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::sync::Mutex;

    // Env-var-touching tests must not run concurrently (set_var/remove_var on
    // shared env vars race under parallel test execution). This lock serializes
    // them.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    fn tempfile_path(name: &str) -> PathBuf {
        let mut path = std::env::temp_dir();
        path.push(format!(
            "hi-mlx-expert-stream-{name}-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        path
    }

    // Nightly 1.98 made `set_var`/`remove_var` unsafe (mutable statics). These
    // tests are single-threaded and each cleans up after itself.
    fn set_env(key: &str, value: &str) {
        // SAFETY: tests run single-threaded; each var is removed before the next test.
        unsafe { std::env::set_var(key, value) }
    }
    fn remove_env(key: &str) {
        // SAFETY: tests run single-threaded.
        unsafe { std::env::remove_var(key) }
    }

    /// Write a safetensors file whose header declares the given tensors with
    /// explicit byte lengths (the data section is zero-filled to the total).
    fn write_safetensors_with_sizes(path: &Path, tensors: &[(&str, u64)]) {
        let mut entries = String::from(r#"{"__metadata__":{"format":"pt"}"#);
        let mut offset = 0u64;
        for (name, size) in tensors {
            entries.push(',');
            entries.push('"');
            entries.push_str(name);
            entries.push('"');
            entries.push_str(&format!(
                r#":{{"dtype":"F32","shape":[{}],"data_offsets":[{},{}]}}"#,
                size,
                offset,
                offset + size
            ));
            offset += size;
        }
        entries.push('}');
        let header = entries.as_bytes();
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        bytes.extend_from_slice(header);
        bytes.resize(bytes.len() + offset as usize, 0);
        fs::write(path, bytes).unwrap();
    }

    fn write_index(dir: &Path, map: &[(&str, &str)]) {
        let mut s = r#"{"metadata":{"total_size":1},"weight_map":{"#.to_string();
        for (i, (k, v)) in map.iter().enumerate() {
            if i > 0 {
                s.push(',');
            }
            s.push_str(&format!("\"{k}\":\"{v}\""));
        }
        s.push_str("}}");
        fs::write(dir.join("model.safetensors.index.json"), s).unwrap();
    }

    fn moe_config(dir: &Path, n_experts: u32, n_layers: u32) -> MlxModelConfig {
        parse_model_config(
            dir,
            json!({
                "architectures": ["DeepseekV2ForCausalLM"],
                "model_type": "deepseek_v2",
                "hidden_size": 256,
                "num_hidden_layers": n_layers,
                "num_attention_heads": 8,
                "num_key_value_heads": 1,
                "n_routed_experts": n_experts,
                "num_experts_per_tok": 2,
                "moe_intermediate_size": 64,
                "vocab_size": 1024
            }),
        )
        .unwrap()
    }

    fn make_moe_catalog(
        dir: &Path,
        expert_bytes_per_proj: u64,
        trunk_bytes: u64,
        n_layers: u32,
    ) -> WeightCatalog {
        // One shard with: trunk tensor + per-layer expert gate/up/down weights.
        let mut tensors: Vec<(&str, u64)> = vec![("model.embed_tokens.weight", trunk_bytes)];
        let mut index: Vec<(&str, &str)> = vec![("model.embed_tokens.weight", "model.safetensors")];
        for layer in 0..n_layers {
            for proj in EXPERT_PROJECTIONS {
                let name = format!("model.layers.{layer}.mlp.switch_mlp.{proj}.weight");
                let leaked: &'static str = Box::leak(name.clone().into_boxed_str());
                tensors.push((leaked, expert_bytes_per_proj));
                index.push((leaked, "model.safetensors"));
            }
        }
        write_safetensors_with_sizes(&dir.join("model.safetensors"), &tensors);
        write_index(dir, &index);
        WeightCatalog::load(dir).unwrap()
    }

    #[test]
    fn classify_recognizes_switch_mlp_expert_tensors() {
        assert_eq!(
            classify_expert_tensor("model.layers.3.mlp.switch_mlp.gate_proj.weight"),
            Some((3, "gate_proj", "weight"))
        );
        assert_eq!(
            classify_expert_tensor("model.layers.0.mlp.switch_mlp.down_proj.scales"),
            Some((0, "down_proj", "scales"))
        );
        assert_eq!(
            classify_expert_tensor("model.layers.12.mlp.switch_mlp.up_proj.biases"),
            Some((12, "up_proj", "biases"))
        );
        // VL prefix tolerated.
        assert_eq!(
            classify_expert_tensor("language_model.model.layers.1.mlp.switch_mlp.gate_proj.weight"),
            Some((1, "gate_proj", "weight"))
        );
    }

    #[test]
    fn classify_rejects_non_expert_tensors() {
        assert_eq!(
            classify_expert_tensor("model.layers.0.self_attn.q_proj.weight"),
            None
        );
        assert_eq!(
            classify_expert_tensor("model.layers.0.mlp.gate_proj.weight"),
            None,
            "dense MLP (no switch_mlp) is not an expert tensor"
        );
        assert_eq!(classify_expert_tensor("model.norm.weight"), None);
        assert_eq!(classify_expert_tensor("model.embed_tokens.weight"), None);
    }

    #[test]
    fn build_plan_splits_trunk_and_expert_bytes() {
        let dir = tempfile_path("plan-split");
        fs::create_dir_all(&dir).unwrap();
        // 2 layers × 3 projections × 100 bytes = 600 expert bytes; 50 trunk.
        let catalog = make_moe_catalog(&dir, 100, 50, 2);
        let config = moe_config(&dir, 4, 2);
        let plan = build_plan(&catalog, &config).unwrap();

        assert_eq!(plan.trunk_bytes, 50);
        assert_eq!(plan.expert_bytes, 600);
        assert_eq!(plan.sources.len(), 6, "2 layers × 3 projections");
        assert_eq!(plan.moe_layers, 2);
        assert!(plan.has_experts());
        // Each source covers one projection: 100 bytes.
        for s in &plan.sources {
            assert_eq!(s.bytes, 100);
            assert!(s.weight_name.contains("switch_mlp"));
            assert!(s.scales_name.is_none());
            assert!(s.biases_name.is_none());
        }
    }

    #[test]
    fn build_plan_includes_scales_and_biases_in_expert_bytes() {
        let dir = tempfile_path("plan-quant");
        fs::create_dir_all(&dir).unwrap();
        // Manually craft a catalog with weight + scales + biases for one layer.
        let tensors: &[(&str, u64)] = &[
            ("model.embed_tokens.weight", 40),
            ("model.layers.0.mlp.switch_mlp.gate_proj.weight", 100),
            ("model.layers.0.mlp.switch_mlp.gate_proj.scales", 20),
            ("model.layers.0.mlp.switch_mlp.gate_proj.biases", 10),
            ("model.layers.0.mlp.switch_mlp.up_proj.weight", 100),
            ("model.layers.0.mlp.switch_mlp.down_proj.weight", 100),
        ];
        write_safetensors_with_sizes(&dir.join("model.safetensors"), tensors);
        write_index(
            &dir,
            &tensors
                .iter()
                .map(|(k, _)| (*k, "model.safetensors"))
                .collect::<Vec<_>>(),
        );
        let catalog = WeightCatalog::load(&dir).unwrap();
        let config = moe_config(&dir, 4, 1);
        let plan = build_plan(&catalog, &config).unwrap();

        // gate_proj group = 100 + 20 + 10 = 130; up = 100; down = 100.
        assert_eq!(plan.expert_bytes, 330);
        assert_eq!(plan.trunk_bytes, 40);
        let gate = plan
            .sources
            .iter()
            .find(|s| s.projection == "gate_proj")
            .unwrap();
        assert_eq!(gate.bytes, 130);
        assert!(gate.scales_name.is_some());
        assert!(gate.biases_name.is_some());
    }

    #[test]
    fn build_plan_no_moe_config_returns_all_trunk() {
        let dir = tempfile_path("plan-no-moe");
        fs::create_dir_all(&dir).unwrap();
        write_safetensors_with_sizes(
            &dir.join("model.safetensors"),
            &[
                ("model.embed_tokens.weight", 100),
                ("model.norm.weight", 50),
            ],
        );
        write_index(
            &dir,
            &[
                ("model.embed_tokens.weight", "model.safetensors"),
                ("model.norm.weight", "model.safetensors"),
            ],
        );
        let catalog = WeightCatalog::load(&dir).unwrap();
        // Dense config (no n_routed_experts).
        let config = parse_model_config(
            &dir,
            json!({
                "architectures": ["Qwen2ForCausalLM"],
                "model_type": "qwen2",
                "hidden_size": 64,
                "num_hidden_layers": 1,
                "num_attention_heads": 4,
                "num_key_value_heads": 1,
                "vocab_size": 256
            }),
        )
        .unwrap();
        let plan = build_plan(&catalog, &config).unwrap();
        assert_eq!(plan.expert_bytes, 0);
        assert!(!plan.has_experts());
        assert_eq!(plan.trunk_bytes, 150);
    }

    #[test]
    fn decide_auto_streams_when_model_exceeds_limit() {
        let _guard = ENV_LOCK.lock().unwrap();
        let dir = tempfile_path("decide-auto-on");
        fs::create_dir_all(&dir).unwrap();
        // 200 expert + 50 trunk = 250 total; limit 100 → must stream.
        let catalog = make_moe_catalog(&dir, 100, 50, 2);
        let config = moe_config(&dir, 4, 2);
        let plan = build_plan(&catalog, &config).unwrap();
        // Clear env so auto path runs.
        remove_env(EXPERT_STREAMING_ENV);
        remove_env(EXPERT_POOL_BYTES_ENV);
        let d = decide(&plan, Some(100));
        assert!(d.stream, "should stream when 250 > 100");
        assert!(d.pool_bytes > 0);
        assert!(d.reason.contains("exceed limit"));
    }

    #[test]
    fn decide_auto_resident_when_model_fits() {
        let _guard = ENV_LOCK.lock().unwrap();
        let dir = tempfile_path("decide-auto-off");
        fs::create_dir_all(&dir).unwrap();
        // 60 expert + 50 trunk = 110; limit 500 → fits, no stream.
        let catalog = make_moe_catalog(&dir, 10, 50, 2);
        let config = moe_config(&dir, 4, 2);
        let plan = build_plan(&catalog, &config).unwrap();
        remove_env(EXPERT_STREAMING_ENV);
        remove_env(EXPERT_POOL_BYTES_ENV);
        let d = decide(&plan, Some(500));
        assert!(!d.stream, "should stay resident when 110 <= 500");
        assert!(d.reason.contains("fits resident"));
    }

    #[test]
    fn decide_hard_off_forces_resident() {
        let _guard = ENV_LOCK.lock().unwrap();
        let dir = tempfile_path("decide-hard-off");
        fs::create_dir_all(&dir).unwrap();
        let catalog = make_moe_catalog(&dir, 100, 50, 2);
        let config = moe_config(&dir, 4, 2);
        let plan = build_plan(&catalog, &config).unwrap();
        set_env(EXPERT_STREAMING_ENV, "0");
        let d = decide(&plan, Some(100));
        assert!(!d.stream, "HI_MLX_EXPERT_STREAMING=0 forces resident");
        assert!(d.reason.contains("forces resident"));
        remove_env(EXPERT_STREAMING_ENV);
    }

    #[test]
    fn decide_forced_on_even_when_it_fits() {
        let _guard = ENV_LOCK.lock().unwrap();
        let dir = tempfile_path("decide-forced-on");
        fs::create_dir_all(&dir).unwrap();
        let catalog = make_moe_catalog(&dir, 10, 50, 2);
        let config = moe_config(&dir, 4, 2);
        let plan = build_plan(&catalog, &config).unwrap();
        set_env(EXPERT_STREAMING_ENV, "1");
        remove_env(EXPERT_POOL_BYTES_ENV);
        let d = decide(&plan, Some(10_000));
        assert!(d.stream, "forced on even when it fits");
        assert!(d.reason.contains("forced on"));
        remove_env(EXPERT_STREAMING_ENV);
    }

    #[test]
    fn decide_pool_override_wins() {
        let _guard = ENV_LOCK.lock().unwrap();
        let dir = tempfile_path("decide-pool-override");
        fs::create_dir_all(&dir).unwrap();
        let catalog = make_moe_catalog(&dir, 100, 50, 2);
        let config = moe_config(&dir, 4, 2);
        let plan = build_plan(&catalog, &config).unwrap();
        set_env(EXPERT_STREAMING_ENV, "1");
        set_env(EXPERT_POOL_BYTES_ENV, "12345");
        let d = decide(&plan, Some(10_000));
        assert!(d.stream);
        assert_eq!(d.pool_bytes, 12345, "override beats auto budget");
        remove_env(EXPERT_STREAMING_ENV);
        remove_env(EXPERT_POOL_BYTES_ENV);
    }

    #[test]
    fn decide_no_experts_is_noop() {
        let _guard = ENV_LOCK.lock().unwrap();
        let plan = ExpertStreamPlan {
            trunk_bytes: 100,
            expert_bytes: 0,
            sources: vec![],
            moe_layers: 0,
        };
        set_env(EXPERT_STREAMING_ENV, "1");
        let d = decide(&plan, Some(10));
        assert!(!d.stream, "no experts → no streaming even if forced");
        remove_env(EXPERT_STREAMING_ENV);
    }

    #[test]
    fn auto_pool_budget_clamps_to_at_least_one_group() {
        let plan = ExpertStreamPlan {
            trunk_bytes: 90,
            expert_bytes: 100,
            sources: vec![ExpertSource {
                layer: 0,
                projection: "gate_proj",
                weight_name: "x".into(),
                scales_name: None,
                biases_name: None,
                shard_file: "f".into(),
                scales_shard: None,
                biases_shard: None,
                bytes: 100,
            }],
            moe_layers: 1,
        };
        // limit 100, trunk 90 → leftover 10, half 5 → clamped up to 100 (one group).
        let pool = auto_pool_budget(&plan, Some(100));
        assert_eq!(pool, 100);
    }

    #[test]
    fn auto_pool_budget_capped_at_expert_bytes() {
        let plan = ExpertStreamPlan {
            trunk_bytes: 10,
            expert_bytes: 50,
            sources: vec![
                ExpertSource {
                    layer: 0,
                    projection: "gate_proj",
                    weight_name: "x".into(),
                    scales_name: None,
                    biases_name: None,
                    shard_file: "f".into(),
                    scales_shard: None,
                    biases_shard: None,
                    bytes: 25,
                },
                ExpertSource {
                    layer: 0,
                    projection: "up_proj",
                    weight_name: "y".into(),
                    scales_name: None,
                    biases_name: None,
                    shard_file: "f".into(),
                    scales_shard: None,
                    biases_shard: None,
                    bytes: 25,
                },
            ],
            moe_layers: 1,
        };
        // limit 1000, trunk 10 → leftover 990, half 495 → capped at 50.
        let pool = auto_pool_budget(&plan, Some(1000));
        assert_eq!(pool, 50);
    }
}
