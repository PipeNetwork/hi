use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use serde_json::Value;

#[test]
fn qwen_cpu_prints_reference_run_json() {
    let path = tempfile_path("qwen-cpu");
    write_reference_qwen(&path);

    let output = Command::new(env!("CARGO_BIN_EXE_hi-local"))
        .arg("qwen-cpu")
        .arg(&path)
        .arg("--tokens")
        .arg("0")
        .arg("--max-tokens")
        .arg("1")
        .arg("--top-k")
        .arg("2")
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let body: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(body["backend"], "cpu-reference");
    assert_eq!(body["input_tokens"], serde_json::json!([0]));
    assert_eq!(body["next_token"], 0);
    assert_eq!(body["next_text"], "a");
    assert_eq!(body["generated_tokens"], serde_json::json!([0]));
    assert_eq!(body["generated_text"], "a");
    assert_eq!(body["top_logits"][0]["token_id"], 0);
    assert_eq!(body["top_logits"][0]["token"], "a");
    assert_eq!(body["top_logits"][1]["token_id"], 2);
    assert!(body.get("logits").is_none());
}

#[test]
fn qwen_cpu_requires_input_tokens_or_prompt() {
    let path = tempfile_path("qwen-cpu-missing-input");
    write_reference_qwen(&path);

    let output = Command::new(env!("CARGO_BIN_EXE_hi-local"))
        .arg("qwen-cpu")
        .arg(&path)
        .output()
        .unwrap();

    assert!(!output.status.success());
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("provide either --tokens or --prompt")
    );
}

fn write_reference_qwen(path: &Path) {
    let tensors = vec![
        tensor_f16(
            "token_embd.weight",
            vec![2, 3],
            &[1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        ),
        tensor_f32("output_norm.weight", vec![2], &[1.0, 1.0]),
        tensor_f32("blk.0.attn_norm.weight", vec![2], &[1.0, 1.0]),
        tensor_f32("blk.0.ffn_norm.weight", vec![2], &[1.0, 1.0]),
        tensor_f16("blk.0.attn_q.weight", vec![2, 2], &[0.0; 4]),
        tensor_f16("blk.0.attn_k.weight", vec![2, 2], &[0.0; 4]),
        tensor_f16("blk.0.attn_v.weight", vec![2, 2], &[0.0; 4]),
        tensor_f16("blk.0.attn_output.weight", vec![2, 2], &[0.0; 4]),
        tensor_f16("blk.0.ffn_gate.weight", vec![2, 2], &[0.0; 4]),
        tensor_f16("blk.0.ffn_up.weight", vec![2, 2], &[0.0; 4]),
        tensor_f16("blk.0.ffn_down.weight", vec![2, 2], &[0.0; 4]),
    ];
    write_qwen_gguf(path, tensors);
}

fn write_qwen_gguf(path: &Path, tensors: Vec<TestTensor>) {
    let mut data = Vec::new();
    let tensors = tensors
        .into_iter()
        .map(|mut tensor| {
            pad_to_alignment(&mut data, 32);
            tensor.offset = data.len() as u64;
            data.extend_from_slice(&tensor.bytes);
            tensor
        })
        .collect::<Vec<_>>();

    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"GGUF");
    write_u32(&mut bytes, 3);
    write_u64(&mut bytes, tensors.len() as u64);
    write_u64(&mut bytes, 14);

    write_kv_string(&mut bytes, "general.architecture", "qwen2");
    write_kv_string(&mut bytes, "general.name", "cpu-reference-qwen");
    write_kv_u32(&mut bytes, "general.alignment", 32);
    write_kv_u32(&mut bytes, "general.file_type", 1);
    write_kv_u32(&mut bytes, "qwen2.context_length", 16);
    write_kv_u32(&mut bytes, "qwen2.embedding_length", 2);
    write_kv_u32(&mut bytes, "qwen2.feed_forward_length", 2);
    write_kv_u32(&mut bytes, "qwen2.block_count", 1);
    write_kv_u32(&mut bytes, "qwen2.attention.head_count", 1);
    write_kv_u32(&mut bytes, "qwen2.attention.head_count_kv", 1);
    write_kv_f32(&mut bytes, "qwen2.attention.layer_norm_rms_epsilon", 1.0e-6);
    write_kv_f32(&mut bytes, "qwen2.rope.freq_base", 1_000_000.0);
    write_kv_u32(&mut bytes, "tokenizer.ggml.eos_token_id", 2);
    write_kv_string_array(&mut bytes, "tokenizer.ggml.tokens", &["a", "b", "c"]);

    for tensor in tensors {
        write_string(&mut bytes, &tensor.name);
        write_u32(&mut bytes, tensor.dims.len() as u32);
        for dim in tensor.dims {
            write_u64(&mut bytes, dim);
        }
        write_u32(&mut bytes, tensor.dtype);
        write_u64(&mut bytes, tensor.offset);
    }

    pad_to_alignment(&mut bytes, 32);
    bytes.extend(data);
    fs::write(path, bytes).unwrap();
}

struct TestTensor {
    name: String,
    dims: Vec<u64>,
    dtype: u32,
    offset: u64,
    bytes: Vec<u8>,
}

fn tensor_f32(name: &str, dims: Vec<u64>, values: &[f32]) -> TestTensor {
    TestTensor {
        name: name.to_string(),
        dims,
        dtype: 0,
        offset: 0,
        bytes: values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>(),
    }
}

fn tensor_f16(name: &str, dims: Vec<u64>, values: &[f32]) -> TestTensor {
    TestTensor {
        name: name.to_string(),
        dims,
        dtype: 1,
        offset: 0,
        bytes: values
            .iter()
            .flat_map(|value| f16_bits(*value).to_le_bytes())
            .collect::<Vec<_>>(),
    }
}

fn f16_bits(value: f32) -> u16 {
    match value {
        0.0 => 0x0000,
        1.0 => 0x3c00,
        -1.0 => 0xbc00,
        2.0 => 0x4000,
        -2.0 => 0xc000,
        _ => panic!("test fixture only supports simple f16 values, got {value}"),
    }
}

fn write_kv_string(bytes: &mut Vec<u8>, key: &str, value: &str) {
    write_string(bytes, key);
    write_u32(bytes, 8);
    write_string(bytes, value);
}

fn write_kv_string_array(bytes: &mut Vec<u8>, key: &str, values: &[&str]) {
    write_string(bytes, key);
    write_u32(bytes, 9);
    write_u32(bytes, 8);
    write_u64(bytes, values.len() as u64);
    for value in values {
        write_string(bytes, value);
    }
}

fn write_kv_u32(bytes: &mut Vec<u8>, key: &str, value: u32) {
    write_string(bytes, key);
    write_u32(bytes, 4);
    write_u32(bytes, value);
}

fn write_kv_f32(bytes: &mut Vec<u8>, key: &str, value: f32) {
    write_string(bytes, key);
    write_u32(bytes, 6);
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn write_string(bytes: &mut Vec<u8>, value: &str) {
    write_u64(bytes, value.len() as u64);
    bytes.extend_from_slice(value.as_bytes());
}

fn write_u32(bytes: &mut Vec<u8>, value: u32) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn write_u64(bytes: &mut Vec<u8>, value: u64) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn pad_to_alignment(bytes: &mut Vec<u8>, alignment: usize) {
    let remainder = bytes.len() % alignment;
    if remainder != 0 {
        bytes.extend(vec![0; alignment - remainder]);
    }
}

fn tempfile_path(name: &str) -> PathBuf {
    let mut path = std::env::temp_dir();
    path.push(format!(
        "hi-local-{name}-{}.gguf",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    path
}
