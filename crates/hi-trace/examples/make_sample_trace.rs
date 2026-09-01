//! Create a small, valid local trace for the CI `hi trace verify` guard.
//!
//! Usage: `make_sample_trace <state_home>` — writes a two-event trace under
//! `<state_home>/hi/rsi/<id>` and prints the trace id on stdout.

use std::path::PathBuf;

use hi_trace::{TraceMode, TraceWriter};

fn main() -> anyhow::Result<()> {
    let state_home = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."));
    let root = state_home.join("hi").join("rsi");
    std::fs::create_dir_all(&root)?;
    // A 32-char lowercase-hex id so TraceWriter::create adopts it as the
    // trace id (see the file_name filter in create_with_identity).
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_nanos();
    let id = format!("{nanos:032x}");

    let mut writer = TraceWriter::create(root.join(&id), TraceMode::Local, 1 << 20)?;
    writer.record(
        "ci-step",
        "ci",
        1,
        None,
        None,
        serde_json::json!({"step": "sample"}),
    )?;
    writer.record(
        "ci-step",
        "ci",
        1,
        None,
        None,
        serde_json::json!({"step": "verify"}),
    )?;
    let summary = writer.finalize()?;
    println!("{}", summary.trace_id);
    Ok(())
}
