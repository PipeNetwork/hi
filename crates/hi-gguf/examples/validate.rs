//! Validate a GGUF's tensor layout against hi's Qwen-family expectations:
//! `cargo run -p hi-gguf --example validate -- <model.gguf>`
fn main() -> anyhow::Result<()> {
    let path = std::env::args()
        .nth(1)
        .ok_or_else(|| anyhow::anyhow!("usage: validate <model.gguf>"))?;
    let gguf = hi_gguf::GgufFile::open(&path)?;
    let validation = gguf.validate_qwen_tensors()?;
    println!("{validation:#?}");
    Ok(())
}
