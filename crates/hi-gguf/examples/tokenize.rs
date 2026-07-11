//! Encode text with a GGUF's tokenizer and print the ids:
//! `cargo run -p hi-gguf --example tokenize -- <model.gguf> "text"`
fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .ok_or_else(|| anyhow::anyhow!("usage: tokenize <model.gguf> [text]"))?;
    let text = args.next().unwrap_or_else(|| "Hello world".to_string());
    let gguf = hi_gguf::GgufFile::open(&path)?;
    let tokenizer = gguf.tokenizer()?;
    let ids = tokenizer.encode(&text)?;
    println!("{ids:?}");
    Ok(())
}
