//! Tests for the write-capable `delegate` subagent ([`Agent::handle_delegate`]).
//!
//! The real worktree + subprocess run lives behind the frontend `DelegateRunner`
//! and is covered by live validation. Here a stub runner exercises the dispatch →
//! outcome path, plus the advertisement/depth/budget logic.

use super::common::*;
use super::*;

fn delegate_config() -> AgentConfig {
let mut cfg = config();
    cfg.subagents.write_subagents = crate::WriteSubagentPolicy::On;
    cfg
}

/// A `DelegateRunner` that returns a canned outcome without touching git.
struct StubRunner {
    applied: bool,
}

struct WritingStubRunner {
    root: std::path::PathBuf,
}

#[async_trait::async_trait]
impl crate::DelegateRunner for StubRunner {
    async fn run(&self, task: &str, _verify: Option<&str>) -> crate::DelegateOutcome {
        crate::DelegateOutcome {
            status: if self.applied {
                hi_tools::ToolStatus::Succeeded
            } else {
                hi_tools::ToolStatus::Failed
            },
            applied: self.applied,
            changed_files: vec!["x.rs".to_string()],
            summary: format!("stub outcome for: {task}"),
        }
    }
}

#[async_trait::async_trait]
impl crate::DelegateRunner for WritingStubRunner {
    async fn run(&self, _task: &str, _verify: Option<&str>) -> crate::DelegateOutcome {
        std::fs::write(self.root.join("delegated.rs"), "pub fn delegated() {}\n").unwrap();
        crate::DelegateOutcome {
            status: hi_tools::ToolStatus::Succeeded,
            applied: true,
            changed_files: vec!["delegated.rs".into()],
            summary: "delegate applied one file".into(),
        }
    }
}

#[test]
fn delegate_tool_is_not_read_only_and_not_in_global_set() {
    // Depth ≤ 1 is structural: `delegate` is out of the read-only set and the
    // global set, so it's only ever advertised when explicitly injected.
    assert!(!hi_tools::is_read_only("delegate"));
    assert!(!hi_tools::TOOL_SPECS.iter().any(|t| t.name == "delegate"));
    assert_eq!(hi_tools::delegate_tool_spec().name, "delegate");
}

#[test]
fn delegate_advertised_only_when_enabled_and_not_read_only() {
    // Enabled: offered in an Auto turn...
    let on = agent(Vec::new(), delegate_config());
    assert!(
        on.request_tools_for(hi_ai::ToolMode::Auto)
            .iter()
            .any(|t| t.name == "delegate"),
        "write_subagents should advertise delegate in Auto mode"
    );
    // ...but never in a read-only/review turn (it's a write tool).
    assert!(
        !on.request_tools_for(hi_ai::ToolMode::ReadOnly)
            .iter()
            .any(|t| t.name == "delegate"),
        "delegate must not appear in a read-only turn"
    );
    // Disabled by default.
    let off = agent(Vec::new(), config());
    assert!(
        !off.request_tools_for(hi_ai::ToolMode::Auto)
            .iter()
            .any(|t| t.name == "delegate"),
        "delegate is off by default"
    );
}

#[test]
fn subagent_never_gets_delegate() {
    // Depth ≤ 1: a subagent is never advertised delegate, in any mode.
    let mut cfg = delegate_config();
    cfg.subagents.is_subagent = true;
    let agent = agent(Vec::new(), cfg);
    for mode in [hi_ai::ToolMode::Auto, hi_ai::ToolMode::ReadOnly] {
        assert!(
            !agent
                .request_tools_for(mode)
                .iter()
                .any(|t| t.name == "delegate"),
            "a subagent must never be offered delegate ({mode:?})"
        );
    }
}

#[tokio::test]
async fn delegate_missing_task_errors() {
    // Returns before any checkpoint/git operation.
    let mut agent = agent(Vec::new(), delegate_config());
    let mut ui = NullUi;
    let out = agent.handle_delegate("{}", &mut ui).await;
    assert_eq!(out.status, hi_tools::ToolStatus::Failed);
    assert!(out.content.contains("missing"), "got: {}", out.content);
    assert_eq!(agent.delegate_subagents_used, 0);
}

#[tokio::test]
async fn delegate_respects_session_budget() {
    // At the cap, it returns before touching the working tree.
    let mut agent = agent(Vec::new(), delegate_config());
    agent.delegate_subagents_used = crate::agent::MAX_DELEGATE_SUBAGENTS_PER_SESSION;
    let mut ui = NullUi;
    let out = agent
        .handle_delegate(r#"{"task":"do something"}"#, &mut ui)
        .await;
    assert_eq!(out.status, hi_tools::ToolStatus::Denied);
    assert!(
        out.content.contains("budget exhausted"),
        "got: {}",
        out.content
    );
    assert_eq!(
        agent.delegate_subagents_used,
        crate::agent::MAX_DELEGATE_SUBAGENTS_PER_SESSION
    );
}

#[test]
fn set_write_subagents_toggles_advertisement() {
    // The `/delegate` runtime toggle re-advertises the tool set.
    let mut agent = agent(Vec::new(), config()); // write_subagents off
    let has = |a: &Agent| {
        a.request_tools_for(hi_ai::ToolMode::Auto)
            .iter()
            .any(|t| t.name == "delegate")
    };
    assert!(!has(&agent), "off in test base config");
    agent.set_write_subagents(crate::WriteSubagentPolicy::On);
    assert!(has(&agent), "on after /delegate on");
    agent.set_write_subagents(crate::WriteSubagentPolicy::Off);
    assert!(!has(&agent), "off again after /delegate off");
    agent.set_write_subagents(crate::WriteSubagentPolicy::Risk);
    // Risk without a multi-file task still advertises when tools refresh with no task
    // (startup Full-ish path uses task=None → treated as eligible when enabled).
    assert!(has(&agent), "risk policy still enables the tool family");
}

#[tokio::test]
async fn delegate_without_runner_is_unavailable() {
    // No runner attached (e.g. a frontend that doesn't support write subagents).
    let mut agent = agent(Vec::new(), delegate_config());
    let mut ui = NullUi;
    let out = agent
        .handle_delegate(r#"{"task":"do the thing"}"#, &mut ui)
        .await;
    assert_eq!(out.status, hi_tools::ToolStatus::Denied);
    assert!(out.content.contains("unavailable"), "got: {}", out.content);
    assert_eq!(agent.delegate_subagents_used, 0);
}

#[tokio::test]
async fn delegate_invokes_runner_and_rejects_a_false_applied_claim() {
    let mut agent = agent(Vec::new(), delegate_config());
    agent.set_delegate_runner(std::sync::Arc::new(StubRunner { applied: true }));
    let mut ui = RecUi::default();
    let out = agent
        .handle_delegate(r#"{"task":"do the thing"}"#, &mut ui)
        .await;
    assert_eq!(out.status, hi_tools::ToolStatus::Failed);
    assert!(
        out.content.contains("stub outcome for: do the thing"),
        "got: {}",
        out.content
    );
    assert!(out.effects.mutation_attempted);
    assert!(!out.effects.mutation_applied);
    assert_eq!(agent.delegate_subagents_used, 1);
    assert!(
        ui.statuses.iter().any(|s| s.contains("delegate subagent")),
        "expected a delegate callout; got {:?}",
        ui.statuses
    );
}

#[tokio::test]
async fn rolled_back_delegate_is_failed_in_typed_tool_timeline() {
    let responses = vec![
        completion(
            vec![Content::ToolCall {
                id: "delegate-1".into(),
                name: "delegate".into(),
                arguments: r#"{"task":"make the change"}"#.into(),
            }],
            1,
            1,
        ),
        completion(
            vec![Content::Text("The delegate was rolled back.".into())],
            1,
            1,
        ),
    ];
    let mut agent = agent(responses, delegate_config());
    agent.set_delegate_runner(std::sync::Arc::new(StubRunner { applied: false }));

    agent
        .run_turn("implement the change", &mut NullUi)
        .await
        .unwrap();

    let entry = agent
        .last_turn_telemetry()
        .tool_timeline
        .iter()
        .find(|entry| entry.tool == "delegate")
        .expect("delegate appears in the typed tool timeline");
    assert_eq!(entry.status, hi_tools::ToolStatus::Failed);
    assert!(entry.error);
    assert!(entry.effects.mutation_attempted);
    assert!(!entry.effects.mutation_applied);
    assert!(entry.effects.file_changes.is_empty());
}

#[tokio::test]
async fn applied_delegate_timeline_contains_exact_reconciled_effects() {
    let base = std::env::temp_dir().join(format!(
        "hi-delegate-effects-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let root = base.join("workspace");
    std::fs::create_dir_all(&root).unwrap();
    let mut cfg = delegate_config();
    cfg.paths.workspace_root = root.clone();
    cfg.paths.state_root = base.join("state");
    let responses = vec![
        completion(
            vec![Content::ToolCall {
                id: "delegate-1".into(),
                name: "delegate".into(),
                arguments: r#"{"task":"make the change"}"#.into(),
            }],
            1,
            1,
        ),
        completion(
            vec![Content::Text("The delegated change is ready.".into())],
            1,
            1,
        ),
    ];
    let mut agent = agent(responses, cfg);
    agent.set_delegate_runner(std::sync::Arc::new(WritingStubRunner {
        root: root.clone(),
    }));

    agent
        .run_turn("implement the change", &mut NullUi)
        .await
        .unwrap();

    let entry = agent
        .last_turn_telemetry()
        .tool_timeline
        .iter()
        .find(|entry| entry.tool == "delegate")
        .expect("delegate appears in the typed tool timeline");
    assert_eq!(entry.status, hi_tools::ToolStatus::Succeeded);
    assert!(entry.effects.mutation_attempted);
    assert!(entry.effects.mutation_applied);
    assert_eq!(entry.effects.file_changes.len(), 1);
    let change = &entry.effects.file_changes[0];
    assert_eq!(change.path, "delegated.rs");
    assert_eq!(change.kind, hi_tools::FileChangeKind::Create);
    assert!(change.before_digest.is_none());
    assert!(change.after_digest.is_some());

    let _ = std::fs::remove_dir_all(base);
}
