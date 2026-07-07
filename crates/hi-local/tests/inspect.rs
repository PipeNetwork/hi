use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use serde_json::Value;

#[test]
fn inspect_prints_gguf_summary() {
    let path = tempfile_path("inspect");
    write_tiny_qwen(&path);

    let output = Command::new(env!("CARGO_BIN_EXE_hi-local"))
        .arg("inspect")
        .arg(&path)
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let body: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(body["version"], 3);
    assert_eq!(body["qwen"]["architecture"], "qwen2");
    assert_eq!(body["qwen"]["context_length"], 16);
    assert_eq!(body["tensor_count"], 1);
}

fn write_tiny_qwen(path: &Path) {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"GGUF");
    write_u32(&mut bytes, 3);
    write_u64(&mut bytes, 1);
    write_u64(&mut bytes, 12);

    write_kv_string(&mut bytes, "general.architecture", "qwen2");
    write_kv_u32(&mut bytes, "general.alignment", 32);
    write_kv_u32(&mut bytes, "general.file_type", 1);
    write_kv_u32(&mut bytes, "qwen2.context_length", 16);
    write_kv_u32(&mut bytes, "qwen2.embedding_length", 4);
    write_kv_u32(&mut bytes, "qwen2.feed_forward_length", 8);
    write_kv_u32(&mut bytes, "qwen2.block_count", 1);
    write_kv_u32(&mut bytes, "qwen2.attention.head_count", 1);
    write_kv_u32(&mut bytes, "qwen2.attention.head_count_kv", 1);
    write_kv_f32(&mut bytes, "qwen2.rope.freq_base", 1_000_000.0);
    write_kv_u32(&mut bytes, "tokenizer.ggml.eos_token_id", 1);
    write_kv_string_array(&mut bytes, "tokenizer.ggml.tokens", &["hello", "world"]);

    write_string(&mut bytes, "token_embd.weight");
    write_u32(&mut bytes, 2);
    write_u64(&mut bytes, 2);
    write_u64(&mut bytes, 4);
    write_u32(&mut bytes, 1);
    write_u64(&mut bytes, 0);

    pad_to_alignment(&mut bytes, 32);
    bytes.extend_from_slice(&[0; 16]);
    fs::write(path, bytes).unwrap();
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
