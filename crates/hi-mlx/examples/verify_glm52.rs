//! Verify the hi-mlx pipeline against a real GLM-5.2 MLX model directory:
//! config parsing, family routing, manifest inspection, weight catalog, and
//! the MoE expert-streaming plan + auto-enable decision.
//!
//! Usage: cargo run -p hi-mlx --example verify_glm52 -- <model-dir>

use std::path::PathBuf;

use anyhow::{Context, Result};

fn main() -> Result<()> {
    let dir = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .context("usage: verify_glm52 <model-dir>")?;

    println!("=== hi-mlx GLM-5.2 end-to-end verification ===");
    println!("model dir: {}", dir.display());

    // 1. Config parsing + family routing.
    let config = hi_mlx::config::load_model_config(&dir)?;
    println!(
        "\n[1] config parsed: model_type={}, family={:?}, arch={:?}",
        config.model_type, config.family, config.architectures
    );
    println!(
        "    hidden_size={}, num_hidden_layers={}, n_routed_experts={:?}, \
         num_experts_per_tok={:?}, moe_intermediate_size={:?}",
        config.hidden_size,
        config.num_hidden_layers,
        config.n_routed_experts,
        config.num_experts_per_tok,
        config.moe_intermediate_size
    );
    println!(
        "    head_dim={:?}, num_attention_heads={}, kv_lora_rank={:?}, q_lora_rank={:?}",
        config.head_dim, config.num_attention_heads, config.kv_lora_rank, config.q_lora_rank
    );
    println!(
        "    quantization label: {}",
        config.quantization_label()
    );
    config.quantization.validate_supported()?;
    println!("    validate_supported: OK");

    // 2. Manifest inspection (requires tokenizer.json + weight shards).
    match hi_mlx::manifest::inspect_model(&dir, None) {
        Ok(info) => {
            println!(
                "\n[2] manifest: family={:?}, model_type={}, context_length={:?}, \
                 max_output_tokens={}, weight_shards={}",
                info.family,
                info.model_type,
                info.context_length,
                info.max_output_tokens,
                info.weight_shards.len()
            );
        }
        Err(e) => {
            println!("\n[2] manifest inspect skipped: {e}");
        }
    }

    // 3. Weight catalog + expert streaming plan.
    match hi_mlx::weights::WeightCatalog::load(&dir) {
        Ok(catalog) => {
            println!(
                "\n[3] weight catalog loaded: {} tensor keys across {} shards ({:.1} GiB on disk)",
                catalog.tensors.len(),
                catalog.shards.len(),
                catalog.estimated_bytes as f64 / (1u64 << 30) as f64
            );
            catalog.validate_for_config(&config)?;
            println!("    validate_for_config: OK");

            let plan = hi_mlx::expert_stream::build_plan(&catalog, &config)?;
            println!(
                "\n[4] expert-stream plan: trunk={:.1} GiB, experts={:.1} GiB, \
                 {} expert groups across {} MoE layers",
                plan.trunk_bytes as f64 / (1u64 << 30) as f64,
                plan.expert_bytes as f64 / (1u64 << 30) as f64,
                plan.expert_count_groups(),
                plan.moe_layers
            );

            // 4. Auto-enable decision against this machine's memory budget.
            let host = hi_mlx::backend::configured_memory_limit_bytes()?;
            println!(
                "    configured memory limit: {}",
                host.map(|b| format!("{:.1} GiB", b as f64 / (1u64 << 30) as f64))
                    .unwrap_or_else(|| "unknown".to_string())
            );
            let decision = hi_mlx::expert_stream::decide(&plan, host);
            println!(
                "    decision: stream={}, pool={:.2} GiB, reason={}",
                decision.stream,
                decision.pool_bytes as f64 / (1u64 << 30) as f64,
                decision.reason
            );
        }
        Err(e) => {
            println!("\n[3] weight catalog skipped (weights not present yet): {e}");
        }
    }

    println!("\n=== verification complete ===");
    Ok(())
}
