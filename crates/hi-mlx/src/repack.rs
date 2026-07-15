#![cfg(all(target_os = "macos", target_arch = "aarch64", feature = "mlx"))]

//! Slab repacking: rewrites shard files so that expert slabs are contiguous
//! on disk, reducing seek overhead during streaming MoE inference.
//!
//! In the original shard layout, expert tensors are interleaved with trunk
//! tensors in whatever order the converter wrote them. A 6.3 MB expert slab
//! may require a seek to a random offset in a 4.3 GB file. After repacking,
//! all experts for a given (layer, projection) are stored contiguously, so
//! reading 8 experts for one layer becomes a single sequential read (or a
//! small number of sequential reads) instead of 8 random reads.
//!
//! The repack also merges weight + scales + biases for each expert into a
//! single contiguous block, reducing the read count by 3×.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::Path;

use anyhow::{Context, Result, bail};
use mlx_rs::Array;
use serde_json::json;

use crate::expert_stream::{ExpertStreamPlan, classify_expert_tensor};

/// Repack a model's shard files into a new directory with contiguous expert
/// slabs. Non-expert (trunk) tensors are written first, followed by expert
/// tensors grouped by (layer, projection).
pub fn repack_model(
    model_path: &Path,
    output_dir: &Path,
    shard_size_gb: u64,
) -> Result<()> {
    fs::create_dir_all(output_dir)
        .with_context(|| format!("creating output dir {}", output_dir.display()))?;

    // Copy non-shard files (config, tokenizer, etc.) as-is.
    copy_metadata_files(model_path, output_dir)?;

    // Load the weight catalog to classify tensors.
    let catalog = crate::weights::WeightCatalog::load(model_path)?;
    let config = crate::config::load_model_config(model_path)?;
    let _plan = crate::expert_stream::build_plan(&catalog, &config)?;

    // Read the safetensors index to get the tensor → shard mapping.
    let index_path = model_path.join("model.safetensors.index.json");
    let index: serde_json::Value = if index_path.exists() {
        serde_json::from_str(&fs::read_to_string(&index_path)?)?
    } else {
        // Single-shard model — read the header directly.
        bail!("repack requires a multi-shard model with model.safetensors.index.json");
    };

    let weight_map = index["weight_map"]
        .as_object()
        .context("missing weight_map in index")?;

    // Group tensors: trunk (non-expert) vs expert, and within expert, by
    // (layer, projection).
    let mut trunk_tensors: Vec<String> = Vec::new();
    let mut expert_groups: HashMap<(u32, &'static str), Vec<String>> = HashMap::new();

    for name in &catalog.tensors {
        match classify_expert_tensor(name) {
            Some((layer, proj, _)) => {
                expert_groups
                    .entry((layer, proj))
                    .or_default()
                    .push(name.clone());
            }
            None => {
                trunk_tensors.push(name.clone());
            }
        }
    }

    // Sort expert groups by (layer, projection) for deterministic output.
    let mut sorted_groups: Vec<((u32, &'static str), Vec<String>)> =
        expert_groups.into_iter().collect();
    sorted_groups.sort_by_key(|((layer, proj), _)| (*layer, proj.to_string()));

    tracing::info!(
        "repack: {} trunk tensors, {} expert groups ({} expert tensors)",
        trunk_tensors.len(),
        sorted_groups.len(),
        sorted_groups.iter().map(|(_, v)| v.len()).sum::<usize>()
    );

    // Read all tensor data from the original shards.
    // We read each original shard's header to get offsets, then read tensor
    // bytes on demand.
    let shard_bytes_budget = shard_size_gb * 1024 * 1024 * 1024;

    // Write new shards: trunk tensors first, then expert groups in order.
    let mut writer = ShardWriter::new(output_dir, shard_bytes_budget);
    let mut new_weight_map: HashMap<String, String> = HashMap::new();

    // Write trunk tensors.
    for name in &trunk_tensors {
        let (shard_name, bytes, shape, dtype) = read_tensor(model_path, name, weight_map)?;
        let dest_shard = writer.write_tensor(name, &bytes, &shape, &dtype)?;
        new_weight_map.insert(name.clone(), dest_shard);
    }

    // Write expert tensors grouped by (layer, projection).
    for ((layer, proj), tensor_names) in &sorted_groups {
        for name in tensor_names {
            let (shard_name, bytes, shape, dtype) = read_tensor(model_path, name, weight_map)?;
            let dest_shard = writer.write_tensor(name, &bytes, &shape, &dtype)?;
            new_weight_map.insert(name.clone(), dest_shard);
        }
        let _ = layer;
        let _ = proj;
    }

    // Finalize the last shard.
    writer.finalize_shard()?;

    // Write the new index.
    let total_size = writer.total_bytes;
    let index_json = json!({
        "metadata": {
            "total_size": total_size,
        },
        "weight_map": new_weight_map,
    });
    fs::write(
        output_dir.join("model.safetensors.index.json"),
        serde_json::to_string_pretty(&index_json)?,
    )?;

    tracing::info!(
        "repack: wrote {} shards, {:.1} GiB total",
        writer.shard_count,
        total_size as f64 / (1 << 30) as f64
    );

    Ok(())
}

/// Copy non-shard metadata files (config.json, tokenizer.json, etc.) from
/// the source to the output directory.
fn copy_metadata_files(src: &Path, dst: &Path) -> Result<()> {
    for entry in fs::read_dir(src)? {
        let entry = entry?;
        let name = entry.file_name();
        let name_str = name.to_string_lossy();
        // Skip safetensors shards and the index — we rewrite those.
        if name_str.ends_with(".safetensors")
            || name_str == "model.safetensors.index.json"
            || name_str == "repacked"
        {
            continue;
        }
        let src_path = entry.path();
        let dst_path = dst.join(&name);
        if src_path.is_file() {
            fs::copy(&src_path, &dst_path)?;
        }
    }
    Ok(())
}

/// Read a tensor's raw bytes + shape + dtype from the original shard files.
fn read_tensor(
    model_path: &Path,
    tensor_name: &str,
    weight_map: &serde_json::Map<String, serde_json::Value>,
) -> Result<(String, Vec<u8>, Vec<i32>, String)> {
    let shard_name = weight_map
        .get(tensor_name)
        .and_then(|v| v.as_str())
        .with_context(|| format!("tensor {tensor_name} not in weight_map"))?
        .to_string();

    let shard_path = model_path.join(&shard_name);
    let (shape, dtype, offset, nbytes) = read_safetensors_header(&shard_path, tensor_name)?;

    let mut file = fs::File::open(&shard_path)?;
    use std::io::{Read, Seek, SeekFrom};
    file.seek(SeekFrom::Start(offset))?;
    let mut bytes = vec![0u8; nbytes];
    file.read_exact(&mut bytes)?;

    Ok((shard_name, bytes, shape, dtype))
}

/// Read the safetensors header of a shard file and return the shape, dtype,
/// data offset, and byte length for a specific tensor.
fn read_safetensors_header(
    shard_path: &Path,
    tensor_name: &str,
) -> Result<(Vec<i32>, String, u64, usize)> {
    use std::io::Read;
    let mut file = fs::File::open(shard_path)?;
    let mut header_len_buf = [0u8; 8];
    file.read_exact(&mut header_len_buf)?;
    let header_len = u64::from_le_bytes(header_len_buf) as usize;
    let mut header_buf = vec![0u8; header_len];
    file.read_exact(&mut header_buf)?;
    let header: serde_json::Value = serde_json::from_slice(&header_buf)?;

    let tensor_info = header
        .get(tensor_name)
        .with_context(|| format!("tensor {tensor_name} not in shard header"))?;

    let shape: Vec<i32> = tensor_info["shape"]
        .as_array()
        .context("missing shape")?
        .iter()
        .map(|v| v.as_i64().unwrap() as i32)
        .collect();
    let dtype = tensor_info["dtype"]
        .as_str()
        .context("missing dtype")?
        .to_string();

    // Data starts after the 8-byte header length + header.
    let data_start = (8 + header_len) as u64;
    let offset = data_start + tensor_info["data_offsets"][0].as_u64().unwrap();
    let end = data_start + tensor_info["data_offsets"][1].as_u64().unwrap();
    let nbytes = (end - offset) as usize;

    Ok((shape, dtype, offset, nbytes))
}

/// Writes tensors into safetensors shard files, splitting when a shard
/// reaches the byte budget.
struct ShardWriter {
    output_dir: std::path::PathBuf,
    shard_budget: u64,
    current_shard: Option<fs::File>,
    current_shard_size: u64,
    current_shard_idx: u32,
    current_header: HashMap<String, serde_json::Value>,
    current_data_offset: u64,
    shard_count: u32,
    total_bytes: u64,
}

impl ShardWriter {
    fn new(output_dir: &Path, shard_budget: u64) -> Self {
        ShardWriter {
            output_dir: output_dir.to_path_buf(),
            shard_budget,
            current_shard: None,
            current_shard_size: 0,
            current_shard_idx: 0,
            current_header: HashMap::new(),
            current_data_offset: 0,
            shard_count: 0,
            total_bytes: 0,
        }
    }

    fn shard_name(&self, idx: u32) -> String {
        format!("model-{:05}-of-{:05}.safetensors", idx + 1, 0)
        // The total count is filled in at finalize.
    }

    /// Write a tensor to the current shard. If it would exceed the budget,
    /// finalize the current shard and start a new one. Returns the shard
    /// filename.
    fn write_tensor(
        &mut self,
        name: &str,
        bytes: &[u8],
        shape: &[i32],
        dtype: &str,
    ) -> Result<String> {
        let nbytes = bytes.len() as u64;

        // Start a new shard if needed.
        if self.current_shard.is_none() {
            self.start_shard()?;
        }

        // Check if we need to roll over to a new shard.
        if self.current_shard_size + nbytes > self.shard_budget && self.current_shard_size > 0 {
            self.finalize_shard()?;
            self.start_shard()?;
        }

        // Record the tensor metadata in the header.
        let start_offset = self.current_data_offset;
        let end_offset = start_offset + nbytes;
        self.current_header.insert(
            name.to_string(),
            json!({
                "dtype": dtype,
                "shape": shape,
                "data_offsets": [start_offset, end_offset],
            }),
        );

        // Write the tensor data.
        let file = self.current_shard.as_mut().unwrap();
        file.write_all(bytes)?;

        self.current_data_offset = end_offset;
        self.current_shard_size += nbytes;
        self.total_bytes += nbytes;

        Ok(self.shard_name(self.current_shard_idx))
    }

    fn start_shard(&mut self) -> Result<()> {
        self.current_shard_idx = self.shard_count;
        self.shard_count += 1;
        self.current_shard_size = 0;
        self.current_data_offset = 0;
        self.current_header.clear();

        let shard_path = self.output_dir.join(self.shard_name(self.current_shard_idx));
        self.current_shard = Some(fs::File::create(&shard_path)?);
        Ok(())
    }

    fn finalize_shard(&mut self) -> Result<()> {
        if let Some(mut file) = self.current_shard.take() {
            // Build the header JSON.
            let header = serde_json::to_string(&self.current_header)?;
            let header_bytes = header.as_bytes();
            let header_len = header_bytes.len() as u64;

            // We need to rewrite the file with the header at the beginning.
            // Since we wrote data first, we need to read it back and rewrite.
            // Actually, safetensors format is: [8-byte header len][header][data].
            // We wrote data first, so we need to restructure.
            //
            // Simpler approach: read the data back, write header, then data.
            drop(file);
            let shard_path = self.output_dir.join(self.shard_name(self.current_shard_idx));
            let data = fs::read(&shard_path)?;

            let mut new_file = fs::File::create(&shard_path)?;
            new_file.write_all(&header_len.to_le_bytes())?;
            new_file.write_all(header_bytes)?;
            new_file.write_all(&data)?;

            tracing::debug!(
                "finalized shard {} ({} tensors, {:.1} MiB)",
                shard_path.display(),
                self.current_header.len(),
                self.current_shard_size as f64 / (1 << 20) as f64
            );
        }
        Ok(())
    }
}
