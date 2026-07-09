//! Tests for the write-capable `delegate` subagent ([`Agent::handle_delegate`]).
//!
//! The real worktree + subprocess run lives behind the frontend `DelegateRunner`
//! and is covered by live validation. Here a stub runner exercises the dispatch →
//! outcome path, plus the advertisement/depth/budget logic.

use super::common::*;
use super::*;

fn delegate_config() -> AgentConfig {
    let mut cfg = config();
    cfg.write_subagents = true;
    cfg
}

/// A `DelegateRunner` that returns a canned outcome without touching git.
struct StubRunner {
    applied: bool,
}

#[async_trait::async_trait]
impl crate::DelegateRunner for StubRunner {
    async fn run(&self, task: &str, _verify: Option<&str>) -> crate::DelegateOutcome {
        crate::DelegateOutcome {
            applied: self.applied,
            changed_files: vec!["x.rs".to_string()],
            summary: format!("stub outcome for: {task}"),
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
    cfg.is_subagent = true;
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
    assert!(out.contains("missing"), "got: {out}");
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
    assert!(out.contains("budget exhausted"), "got: {out}");
    assert_eq!(
        agent.delegate_subagents_used,
        crate::agent::MAX_DELEGATE_SUBAGENTS_PER_SESSION
    );
}

#[tokio::test]
async fn delegate_without_runner_is_unavailable() {
    // No runner attached (e.g. a frontend that doesn't support write subagents).
    let mut agent = agent(Vec::new(), delegate_config());
    let mut ui = NullUi;
    let out = agent
        .handle_delegate(r#"{"task":"do the thing"}"#, &mut ui)
        .await;
    assert!(out.contains("unavailable"), "got: {out}");
    assert_eq!(agent.delegate_subagents_used, 0);
}

#[tokio::test]
async fn delegate_invokes_runner_and_returns_its_summary() {
    let mut agent = agent(Vec::new(), delegate_config());
    agent.set_delegate_runner(std::sync::Arc::new(StubRunner { applied: true }));
    let mut ui = RecUi::default();
    let out = agent
        .handle_delegate(r#"{"task":"do the thing"}"#, &mut ui)
        .await;
    assert!(out.contains("stub outcome for: do the thing"), "got: {out}");
    assert_eq!(agent.delegate_subagents_used, 1);
    assert!(
        ui.statuses.iter().any(|s| s.contains("delegate subagent")),
        "expected a delegate callout; got {:?}",
        ui.statuses
    );
}
