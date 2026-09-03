//! Regression coverage for unbounded per-session coding-fact recording.

use super::common::*;

#[test]
fn coding_facts_continue_after_the_previous_session_cap() {
    let cfg = config();
    std::fs::write(
        cfg.paths.workspace_root.join("Cargo.toml"),
        "[package]\nname = \"fixture\"\nversion = \"0.1.0\"\n",
    )
    .unwrap();
    std::fs::create_dir_all(cfg.paths.workspace_root.join("src")).unwrap();
    std::fs::write(
        cfg.paths.workspace_root.join("src/lib.rs"),
        "pub fn ok() {}\n",
    )
    .unwrap();

    let mut agent = agent(Vec::new(), cfg);
    agent.subagents.coding_facts_written = 8;
    agent.report.verify = crate::domain::VerifyEvidence::pass(1, "digest".into());
    agent.workspace.last_changed_files = vec!["src/lib.rs".into()];

    agent.record_coding_facts_turn_end(&mut NullUi);

    assert!(
        agent.subagents.coding_facts_written > 8,
        "the old per-session count must remain telemetry, not a stop condition"
    );
    assert!(
        agent
            .decisions
            .entries()
            .iter()
            .any(|decision| decision.summary.contains("Rust")),
        "a fact discovered after the old boundary must still be recorded"
    );
}
