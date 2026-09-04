use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use anyhow::Result;
use async_trait::async_trait;
use hi_workspace::{
    ExecutionDisposition, ExecutionReport, InMemoryWorkspaceController, MutationIntent,
    WorkspaceController, WorkspaceState,
};

use super::common::{agent, config};

#[derive(Clone, Copy)]
enum AdminResult {
    Succeed,
    FailAfterWrite,
}

struct LifecycleMcp {
    root: PathBuf,
    calls: Arc<AtomicUsize>,
    result: AdminResult,
}

#[async_trait]
impl hi_tools::McpBackend for LifecycleMcp {
    async fn search(&self, _: Option<&str>) -> Result<Vec<hi_tools::McpToolInfo>> {
        Ok(Vec::new())
    }

    async fn call(&self, _: &str, _: &str, _: &serde_json::Value) -> Result<String> {
        unreachable!("these tests exercise only workspace MCP administration")
    }

    async fn workspace_admin(&self, args: &str) -> Result<String> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        std::fs::write(self.root.join("mcp-admin-state.txt"), args)?;
        match self.result {
            AdminResult::Succeed => Ok("saved".into()),
            AdminResult::FailAfterWrite => anyhow::bail!("backend failed after write"),
        }
    }
}

struct StageSession {
    records: Arc<Mutex<Vec<crate::WorkspaceTranscriptExecution>>>,
    fail: bool,
}

impl crate::SessionSink for StageSession {
    fn record(&mut self, _: &[hi_ai::Message], _: hi_ai::Usage) -> Result<()> {
        Ok(())
    }

    fn stage_workspace_execution(
        &mut self,
        record: &crate::WorkspaceTranscriptExecution,
    ) -> Result<()> {
        if self.fail {
            anyhow::bail!("stage unavailable");
        }
        self.records.lock().unwrap().push(record.clone());
        Ok(())
    }

    fn record_compaction(&mut self, _: &[hi_ai::Message]) -> Result<()> {
        Ok(())
    }
}

struct Fixture {
    subject: crate::Agent,
    controller: Arc<InMemoryWorkspaceController>,
    calls: Arc<AtomicUsize>,
    records: Arc<Mutex<Vec<crate::WorkspaceTranscriptExecution>>>,
}

fn pipefs_subject(result: AdminResult, fail_stage: bool) -> Fixture {
    let cfg = config();
    let workspace_root = cfg.paths.workspace_root.clone();
    let state_root = cfg.paths.state_root.clone();
    let mut subject = agent(Vec::new(), cfg);
    let controller = Arc::new(InMemoryWorkspaceController::new_pipefs(
        "workspace",
        "session",
        2,
        true,
        &workspace_root,
        &state_root,
    ));
    subject
        .install_workspace_controller(controller.clone())
        .unwrap();
    let calls = Arc::new(AtomicUsize::new(0));
    subject.attach_mcp(Arc::new(LifecycleMcp {
        root: workspace_root,
        calls: calls.clone(),
        result,
    }));
    let records = Arc::new(Mutex::new(Vec::new()));
    subject.set_session(Box::new(StageSession {
        records: records.clone(),
        fail: fail_stage,
    }));
    Fixture {
        subject,
        controller,
        calls,
        records,
    }
}

#[tokio::test]
async fn mcp_admin_stages_exact_pipefs_evidence_before_success() {
    let Fixture {
        mut subject,
        controller,
        calls,
        records,
    } = pipefs_subject(AdminResult::Succeed, false);

    let output = subject
        .mcp_workspace_admin("add docs --http https://example.test")
        .await
        .expect("MCP attached")
        .unwrap();

    assert_eq!(output, "saved");
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert_eq!(controller.status().state, WorkspaceState::Ready);
    let records = records.lock().unwrap();
    assert_eq!(records.len(), 1);
    let record = &records[0];
    assert_eq!(
        record.execution.disposition,
        ExecutionDisposition::Succeeded
    );
    assert!(record.execution.workspace_may_have_changed);
    assert!(record.execution.external_effect_may_have_occurred);
    assert_eq!(record.calls.len(), 1);
    assert_eq!(record.calls[0].name, "workspace_mcp_admin");
    assert_eq!(record.calls[0].result, "saved");
    let hi_ai::Content::ToolCall {
        name, arguments, ..
    } = &record.assistant_content[0]
    else {
        panic!("expected a staged lifecycle call");
    };
    assert_eq!(name, "workspace_mcp_admin");
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(arguments).unwrap()["command"],
        "add docs --http https://example.test"
    );
}

#[tokio::test]
async fn failed_mcp_admin_settles_as_failed_after_reconciling_partial_write() {
    let Fixture {
        mut subject,
        controller,
        calls,
        records,
    } = pipefs_subject(AdminResult::FailAfterWrite, false);

    let error = subject
        .mcp_workspace_admin("docs disable")
        .await
        .expect("MCP attached")
        .unwrap_err();

    assert!(error.to_string().contains("backend failed after write"));
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    // A known backend failure and a successfully reconciled postimage are a
    // durable failed execution, not an ambiguous operation.
    assert_eq!(controller.status().state, WorkspaceState::Ready);
    let records = records.lock().unwrap();
    assert_eq!(records.len(), 1);
    assert_eq!(record_disposition(&records), ExecutionDisposition::Failed);
    assert!(records[0].execution.workspace_may_have_changed);
    assert!(
        records[0].calls[0]
            .result
            .contains("backend failed after write")
    );
}

#[tokio::test]
async fn closed_workspace_admission_prevents_mcp_admin_side_effect() {
    let Fixture {
        mut subject,
        controller,
        calls,
        records,
    } = pipefs_subject(AdminResult::Succeed, false);
    let permit = controller
        .begin(MutationIntent::workspace("existing mutation"))
        .await
        .unwrap();

    let error = subject
        .mcp_workspace_admin("docs disable")
        .await
        .expect("MCP attached")
        .unwrap_err();

    assert!(error.to_string().contains("workspace controller refused"));
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    assert!(records.lock().unwrap().is_empty());
    let outcome = controller
        .settle(permit, ExecutionReport::succeeded(None))
        .await;
    assert!(outcome.receipt.is_some());
}

#[tokio::test]
async fn pipefs_stage_failure_enters_recovery_instead_of_returning_success() {
    let Fixture {
        mut subject,
        controller,
        calls,
        records,
    } = pipefs_subject(AdminResult::Succeed, true);

    let error = subject
        .mcp_workspace_admin("docs disable")
        .await
        .expect("MCP attached")
        .unwrap_err();

    let message = error.to_string();
    assert!(
        message.contains("transcript could not be staged"),
        "{message}"
    );
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert!(records.lock().unwrap().is_empty());
    assert_eq!(controller.status().state, WorkspaceState::RecoveryRequired);
}

fn record_disposition(records: &[crate::WorkspaceTranscriptExecution]) -> ExecutionDisposition {
    records[0].execution.disposition
}
