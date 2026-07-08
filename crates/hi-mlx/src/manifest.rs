use std::fs;
use std::path::Path;

use anyhow::{Context, Result, anyhow, bail};
use serde_json::Value;

pub use hi_local_core::model::{
    DEFAULT_MAX_OUTPUT_TOKENS, ListedModel, ModelFamily, ModelInfo, TokenizerInfo, WeightShard,
};

pub fn inspect_model(path: impl AsRef<Path>, model_id: Option<String>) -> Result<ModelInfo> {
    let path = path.as_ref();
    if !path.exists() {
        bail!("bad model path {}: does not exist", path.display());
    }
    if !path.is_dir() {
        bail!("bad model path {}: expected a directory", path.display());
    }

    let config_path = path.join("config.json");
    let config = read_json(&config_path)?;
    let model_type = config
        .get("model_type")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    let family = match model_family(&model_type, &config) {
        Some(family) => family,
        None => {
            let found = if model_type.is_empty() {
                "missing".to_string()
            } else {
                format!("'{model_type}'")
            };
            bail!(
                "unsupported model_type {found}; hi-mlx supports these MLX model families: {}",
                supported_model_families()
            );
        }
    };
    let architecture = config
        .get("architectures")
        .and_then(Value::as_array)
        .and_then(|items| items.iter().find_map(Value::as_str))
        .unwrap_or("Qwen2ForCausalLM")
        .to_string();
    let context_length = ["max_position_embeddings", "seq_length", "n_ctx"]
        .iter()
        .find_map(|key| config.get(*key).and_then(Value::as_u64))
        .and_then(|n| u32::try_from(n).ok());

    let generation = read_json_optional(&path.join("generation_config.json"))?;
    let max_output_tokens = generation
        .as_ref()
        .and_then(|value| value.get("max_new_tokens").and_then(Value::as_u64))
        .and_then(|n| u32::try_from(n).ok())
        .unwrap_or(DEFAULT_MAX_OUTPUT_TOKENS);

    let tokenizer_json = path.join("tokenizer.json").exists();
    if !tokenizer_json {
        bail!("bad model path {}: missing tokenizer.json", path.display());
    }
    tokenizers::Tokenizer::from_file(path.join("tokenizer.json"))
        .map_err(|err| anyhow!("loading tokenizer.json: {err}"))?;

    let tokenizer_config_path = path.join("tokenizer_config.json");
    let tokenizer_config = read_json_optional(&tokenizer_config_path)?;
    let chat_template = tokenizer_config
        .as_ref()
        .and_then(|value| value.get("chat_template"))
        .is_some();

    let weight_shards = find_weight_shards(path)?;
    if weight_shards.is_empty() {
        bail!(
            "bad model path {}: no .safetensors weight shards found",
            path.display()
        );
    }

    let id = model_id
        .filter(|id| !id.trim().is_empty())
        .or_else(|| model_id_from_config(&config))
        .unwrap_or_else(|| {
            path.file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("local-mlx-model")
                .to_string()
        });

    Ok(ModelInfo {
        id,
        path: path.to_path_buf(),
        family,
        model_type,
        architecture,
        context_length,
        max_output_tokens,
        tokenizer: TokenizerInfo {
            tokenizer_json,
            tokenizer_config: tokenizer_config_path.exists(),
            special_tokens_map: path.join("special_tokens_map.json").exists(),
        },
        chat_template,
        weight_shards,
    })
}

pub fn list_models(root: impl AsRef<Path>) -> Result<Vec<ListedModel>> {
    let root = root.as_ref();
    if !root.exists() {
        return Ok(Vec::new());
    }
    if !root.is_dir() {
        bail!("{} is not a directory", root.display());
    }
    let mut out = Vec::new();
    for entry in fs::read_dir(root).with_context(|| format!("reading {}", root.display()))? {
        let entry = entry?;
        let path = entry.path();
        if !path.is_dir() || !path.join("config.json").exists() {
            continue;
        }
        match inspect_model(&path, None) {
            Ok(info) => out.push(ListedModel {
                path,
                id: Some(info.id.clone()),
                supported: true,
                summary: format!(
                    "{} · {} · {} · {} shard(s)",
                    info.id,
                    info.family.label(),
                    info.model_type,
                    info.weight_shards.len()
                ),
            }),
            Err(err) => out.push(ListedModel {
                path,
                id: None,
                supported: false,
                summary: err.to_string(),
            }),
        }
    }
    out.sort_by(|a, b| a.path.cmp(&b.path));
    Ok(out)
}

fn read_json(path: &Path) -> Result<Value> {
    let text = fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    serde_json::from_str(&text).with_context(|| format!("parsing {}", path.display()))
}

fn read_json_optional(path: &Path) -> Result<Option<Value>> {
    if !path.exists() {
        return Ok(None);
    }
    read_json(path).map(Some)
}

pub fn supported_model_families() -> &'static str {
    "qwen2/qwen2_moe, qwen3/qwen3_moe/qwen3_next, deepseek_v2/deepseek_v3/deepseek_v32/deepseek_v4/deepseek*, glm4/glm4_moe/glm4_moe_lite Flash, hy_v3 (Hunyuan-3)"
}

fn model_family(model_type: &str, config: &Value) -> Option<ModelFamily> {
    let model_type = model_type.to_ascii_lowercase();
    let architectures = architecture_strings(config);
    let haystack = std::iter::once(model_type.as_str())
        .chain(architectures.iter().map(String::as_str))
        .collect::<Vec<_>>()
        .join(" ");

    if matches!(
        model_type.as_str(),
        "qwen3" | "qwen3_moe" | "qwen3_next" | "qwen3_vl" | "qwen3_vl_moe"
    ) || haystack.contains("qwen3")
    {
        return Some(ModelFamily::Qwen3);
    }
    if matches!(model_type.as_str(), "qwen2" | "qwen2_moe" | "qwen2_5_vl")
        || haystack.contains("qwen2")
    {
        return Some(ModelFamily::Qwen2);
    }
    if model_type.starts_with("deepseek")
        || haystack.contains("deepseek")
        || matches!(
            model_type.as_str(),
            "deepseek_v2" | "deepseek_v3" | "deepseek_v31" | "deepseek_v32" | "deepseek_v4"
        )
        // GLM-5.2 (glm_moe_dsa): DeepSeek-V3.2 arch (MLA + DSA indexer + sigmoid/noaux MoE).
        || matches!(model_type.as_str(), "glm_moe_dsa" | "glm_moe_dsa_mtp")
        || haystack.contains("glm_moe_dsa")
        // Kimi K2/K2.5/K2.7: thin DeepSeek-V3 wrapper.
        || model_type.starts_with("kimi_k2")
        || haystack.contains("kimi_k2")
    {
        return Some(ModelFamily::DeepSeek);
    }
    if model_type.starts_with("glm4")
        || haystack.contains("glm4")
        || haystack.contains("flash")
        || matches!(
            model_type.as_str(),
            "glm4" | "glm4_moe" | "glm4_moe_lite" | "glm4v" | "glm4v_moe"
        )
    {
        return Some(ModelFamily::GlmFlash);
    }
    if matches!(model_type.as_str(), "hy_v3" | "hyv3")
        || haystack.contains("hy_v3")
        || haystack.contains("hyv3")
        || haystack.contains("hunyuan")
    {
        return Some(ModelFamily::Hy3);
    }
    // Nemotron-H: Mamba2 + attention + MLP/MoE hybrid.
    if model_type.starts_with("nemotron_h")
        || haystack.contains("nemotron_h")
        || haystack.contains("nemotronh")
    {
        return Some(ModelFamily::NemotronH);
    }
    // Gemma-4 only (dispatched to Gemma4TextLike in load_model).
    if model_type.starts_with("gemma4") || haystack.contains("gemma4") {
        return Some(ModelFamily::Gemma);
    }
    // MiniMax-M3: GQA + sigmoid/noaux MoE.
    if model_type.starts_with("minimax") || haystack.contains("minimax") {
        return Some(ModelFamily::MiniMax);
    }
    // LongCat-2.0: MLA + DSA indexer + ScMoE double-attention + ngram embedding.
    if model_type.starts_with("longcat") || haystack.contains("longcat") {
        return Some(ModelFamily::LongCat);
    }
    // Dense Llama-like variants that run on the Qwen GQA path (QwenLike).
    if matches!(
        model_type.as_str(),
        "internlm3" | "internlm2" | "granite" | "smollm3"
    ) {
        return Some(ModelFamily::Qwen2);
    }
    // Post-norm Llama-likes (OLMo2 / EXAONE-4): per-head qk-norm; QwenBlock detects post-norm weights.
    if matches!(model_type.as_str(), "exaone4" | "olmo2") {
        return Some(ModelFamily::Qwen3);
    }
    None
}

fn architecture_strings(config: &Value) -> Vec<String> {
    config
        .get("architectures")
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(Value::as_str)
                .map(|item| item.to_ascii_lowercase())
                .collect()
        })
        .unwrap_or_default()
}

fn model_id_from_config(config: &Value) -> Option<String> {
    let raw = config.get("_name_or_path").and_then(Value::as_str)?.trim();
    if raw.is_empty() || raw.starts_with('/') || raw.starts_with('.') {
        return None;
    }
    Some(raw.to_string())
}

fn find_weight_shards(path: &Path) -> Result<Vec<WeightShard>> {
    let mut shards = Vec::new();
    for entry in fs::read_dir(path).with_context(|| format!("reading {}", path.display()))? {
        let entry = entry?;
        let shard_path = entry.path();
        if shard_path.extension().and_then(|s| s.to_str()) != Some("safetensors") {
            continue;
        }
        let meta = entry.metadata()?;
        let rel = shard_path
            .strip_prefix(path)
            .unwrap_or(&shard_path)
            .to_string_lossy()
            .to_string();
        shards.push(WeightShard {
            path: rel,
            bytes: meta.len(),
            tensor_count: safetensors_tensor_count(&shard_path).ok(),
        });
    }
    shards.sort_by(|a, b| a.path.cmp(&b.path));
    Ok(shards)
}

fn safetensors_tensor_count(path: &Path) -> Result<usize> {
    use std::io::Read;

    let mut file = fs::File::open(path)?;
    let mut len = [0u8; 8];
    file.read_exact(&mut len)?;
    let header_len = u64::from_le_bytes(len);
    let header_len = usize::try_from(header_len).context("safetensors header too large")?;
    let mut header = vec![0; header_len];
    file.read_exact(&mut header)?;
    let value: Value = serde_json::from_slice(&header)?;
    let object = value
        .as_object()
        .ok_or_else(|| anyhow!("safetensors header is not an object"))?;
    Ok(object.keys().filter(|key| *key != "__metadata__").count())
}

#[cfg(test)]
pub mod test_support {
    use std::fs;
    use std::path::Path;

    pub fn write_qwen_fixture(root: &Path) {
        write_family_fixture(
            root,
            "qwen2",
            "Qwen2ForCausalLM",
            "Qwen/Qwen2.5-Coder-1.5B-Instruct-MLX",
        );
    }

    pub fn write_family_fixture(
        root: &Path,
        model_type: &str,
        architecture: &str,
        name_or_path: &str,
    ) {
        fs::create_dir_all(root).unwrap();
        fs::write(
            root.join("config.json"),
            format!(
                r#"{{
                  "model_type": "{model_type}",
                  "architectures": ["{architecture}"],
                  "max_position_embeddings": 32768,
                  "_name_or_path": "{name_or_path}"
                }}"#,
            ),
        )
        .unwrap();
        fs::write(
            root.join("tokenizer.json"),
            r#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,"pre_tokenizer":null,"post_processor":null,"decoder":null,"model":{"type":"WordLevel","vocab":{"<unk>":0,"hello":1},"unk_token":"<unk>"}}"#,
        )
        .unwrap();
        fs::write(
            root.join("tokenizer_config.json"),
            r#"{"chat_template":"{% for message in messages %}{{ message['role'] }}: {{ message['content'] }}{% endfor %}"}"#,
        )
        .unwrap();
        fs::write(
            root.join("generation_config.json"),
            r#"{"max_new_tokens": 1024}"#,
        )
        .unwrap();
        write_minimal_safetensors(&root.join("model.safetensors"));
    }

    pub fn write_unsupported_fixture(root: &Path) {
        fs::create_dir_all(root).unwrap();
        fs::write(root.join("config.json"), r#"{"model_type":"llama"}"#).unwrap();
        fs::write(
            root.join("tokenizer.json"),
            r#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,"pre_tokenizer":null,"post_processor":null,"decoder":null,"model":{"type":"WordLevel","vocab":{"<unk>":0},"unk_token":"<unk>"}}"#,
        )
        .unwrap();
        write_minimal_safetensors(&root.join("model.safetensors"));
    }

    fn write_minimal_safetensors(path: &Path) {
        let header = br#"{"__metadata__":{"format":"pt"}}"#;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        bytes.extend_from_slice(header);
        fs::write(path, bytes).unwrap();
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;

    #[test]
    fn inspect_valid_qwen_fixture() {
        let dir = tempfile_path("valid");
        test_support::write_qwen_fixture(&dir);

        let info = inspect_model(&dir, None).unwrap();

        assert_eq!(info.model_type, "qwen2");
        assert_eq!(info.family, ModelFamily::Qwen2);
        assert_eq!(info.architecture, "Qwen2ForCausalLM");
        assert_eq!(info.context_length, Some(32768));
        assert_eq!(info.max_output_tokens, 1024);
        assert!(info.tokenizer.tokenizer_json);
        assert!(info.chat_template);
        assert_eq!(info.weight_shards.len(), 1);
    }

    #[test]
    fn inspect_accepts_qwen3_family() {
        let dir = tempfile_path("qwen3");
        test_support::write_family_fixture(
            &dir,
            "qwen3",
            "Qwen3ForCausalLM",
            "mlx-community/Qwen3-8B-4bit",
        );

        let info = inspect_model(&dir, None).unwrap();

        assert_eq!(info.family, ModelFamily::Qwen3);
        assert_eq!(info.model_type, "qwen3");
    }

    #[test]
    fn inspect_accepts_deepseek_v3_and_v32_families() {
        for (model_type, architecture) in [
            ("deepseek_v3", "DeepseekV3ForCausalLM"),
            ("deepseek_v32", "DeepseekV32ForCausalLM"),
        ] {
            let dir = tempfile_path(model_type);
            test_support::write_family_fixture(
                &dir,
                model_type,
                architecture,
                &format!("mlx-community/{model_type}-4bit"),
            );

            let info = inspect_model(&dir, None).unwrap();

            assert_eq!(info.family, ModelFamily::DeepSeek);
            assert_eq!(info.model_type, model_type);
        }
    }

    #[test]
    fn inspect_accepts_glm_flash_family() {
        let dir = tempfile_path("glm-flash");
        test_support::write_family_fixture(
            &dir,
            "glm4_moe_lite",
            "Glm4MoeForCausalLM",
            "mlx-community/GLM-4.7-Flash-nvfp4",
        );

        let info = inspect_model(&dir, None).unwrap();

        assert_eq!(info.family, ModelFamily::GlmFlash);
        assert_eq!(info.model_type, "glm4_moe_lite");
    }

    #[test]
    fn unsupported_architecture_errors_clearly() {
        let dir = tempfile_path("unsupported");
        test_support::write_unsupported_fixture(&dir);

        let err = inspect_model(&dir, None).unwrap_err();

        assert!(err.to_string().contains("unsupported model_type 'llama'"));
        assert!(err.to_string().contains("qwen3"));
        assert!(err.to_string().contains("deepseek"));
        assert!(err.to_string().contains("glm4"));
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
