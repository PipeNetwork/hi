//! Soft firewall: turn-loop sources should prefer `hi_tools::protocol` for tool
//! protocol symbols and `hi_tools::infra` for product infrastructure.

use std::fs;
use std::path::PathBuf;

fn turn_rs_files() -> Vec<PathBuf> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/agent/turn");
    let mut out = Vec::new();
    fn walk(dir: &std::path::Path, out: &mut Vec<PathBuf>) {
        for entry in fs::read_dir(dir).unwrap() {
            let entry = entry.unwrap();
            let path = entry.path();
            if path.is_dir() {
                walk(&path, out);
            } else if path.extension().and_then(|s| s.to_str()) == Some("rs") {
                out.push(path);
            }
        }
    }
    walk(&root, &mut out);
    out
}

#[test]
fn turn_loop_prefers_protocol_namespace_for_execute() {
    // New execute/mutation imports in the turn loop should use hi_tools::protocol.
    let mut offenders = Vec::new();
    for path in turn_rs_files() {
        let text = fs::read_to_string(&path).unwrap();
        for (i, line) in text.lines().enumerate() {
            let trimmed = line.trim_start();
            if trimmed.starts_with("//") {
                continue;
            }
            // Disallow root-path execute imports in new turn code.
            if trimmed.contains("use hi_tools::{")
                && (trimmed.contains("execute_in_runtime")
                    || trimmed.contains("prepare_mutation")
                    || trimmed.contains("execute_streaming"))
            {
                offenders.push(format!("{}:{}: {trimmed}", path.display(), i + 1));
            }
        }
    }
    assert!(
        offenders.is_empty(),
        "turn loop should import execute/mutation helpers via hi_tools::protocol::…\n{}",
        offenders.join("\n")
    );
}

#[test]
fn turn_fast_feedback_uses_infra_namespace() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/agent/turn/fast_feedback.rs");
    let text = fs::read_to_string(path).unwrap();
    assert!(
        text.contains("use hi_tools::infra::"),
        "fast_feedback should import via hi_tools::infra"
    );
}
