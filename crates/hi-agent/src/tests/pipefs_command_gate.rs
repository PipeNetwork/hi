use std::sync::{Arc, Mutex};

use anyhow::Result;
use hi_control::{ControlStore, OperationReplayClass, WorkspaceOperationStatus};
use hi_workspace::{
    ExecutionReport, InMemoryWorkspaceController, WorkspaceController, WorkspaceState,
};

use super::common::{IsolatedWorkspace, agent};

struct StageSession {
    records: Arc<Mutex<Vec<crate::WorkspaceTranscriptExecution>>>,
}

impl crate::SessionSink for StageSession {
    fn record(&mut self, _: &[hi_ai::Message], _: hi_ai::Usage) -> Result<()> {
        Ok(())
    }

    fn stage_workspace_execution(
        &mut self,
        record: &crate::WorkspaceTranscriptExecution,
    ) -> Result<()> {
        self.records.lock().unwrap().push(record.clone());
        Ok(())
    }

    fn record_compaction(&mut self, _: &[hi_ai::Message]) -> Result<()> {
        Ok(())
    }
}

#[test]
fn pipefs_controller_blocks_sync_mutation_without_legacy_durability() {
    let workspace = IsolatedWorkspace::new("pipefs-sync-command-gate");
    let source = workspace.path("source-skill/SKILL.md");
    std::fs::create_dir_all(source.parent().unwrap()).unwrap();
    std::fs::write(&source, "---\nname: source-skill\n---\n").unwrap();
    let mut subject = agent(Vec::new(), workspace.config());
    subject
        .activate_pipefs_workspace_controller("command-gate", 1, false)
        .unwrap();
    assert!(
        !subject.workspace_durability_enabled(),
        "fixture must prove authority without the legacy optional adapter"
    );
    assert!(
        subject.pipefs_workspace_active(),
        "typed controller authority must not depend on the legacy adapter"
    );

    let denied = crate::handle_session_command(
        &mut subject,
        &crate::Command::Marketplace(format!("install {}", source.display())),
        &[],
    )
    .expect("marketplace command is handled");
    assert!(
        denied
            .message
            .contains("unavailable while PipeFS is active")
    );
    assert!(
        !workspace.path(".hi/skills/source-skill/SKILL.md").exists(),
        "PipeFS command gate allowed an uncommitted workspace mutation"
    );

    let status = crate::handle_session_command(
        &mut subject,
        &crate::Command::Marketplace("status".into()),
        &[],
    )
    .expect("marketplace status is handled");
    assert!(status.message.contains("marketplace"), "{}", status.message);
    assert!(
        !status
            .message
            .contains("unavailable while PipeFS is active"),
        "read-only command form was incorrectly denied"
    );

    for (command, forbidden_path) in [
        (
            crate::Command::Agents("add reviewer stay read-only".into()),
            Some(workspace.path(".hi/agents/reviewer.md")),
        ),
        (
            crate::Command::Share(String::new()),
            Some(workspace.path(".hi/shares")),
        ),
        (crate::Command::Trust("off".into()), None),
    ] {
        let denied = crate::handle_session_command(&mut subject, &command, &[])
            .expect("session command is handled");
        assert!(
            denied
                .message
                .contains("unavailable while PipeFS is active"),
            "{command:?}: {}",
            denied.message
        );
        if let Some(path) = forbidden_path {
            assert!(
                !path.exists(),
                "PipeFS command gate allowed an uncommitted write to {}",
                path.display()
            );
        }
    }

    for command in [
        crate::Command::Agents("list".into()),
        crate::Command::Trust("status".into()),
    ] {
        let available = crate::handle_session_command(&mut subject, &command, &[])
            .expect("read-only session command is handled");
        assert!(
            !available
                .message
                .contains("unavailable while PipeFS is active"),
            "{command:?}: {}",
            available.message
        );
    }
}

#[tokio::test]
async fn coordinated_pipefs_command_stages_result_before_publication() {
    let workspace = IsolatedWorkspace::new("pipefs-coordinated-command");
    let source = workspace.path("source-skill/SKILL.md");
    std::fs::create_dir_all(source.parent().unwrap()).unwrap();
    std::fs::write(&source, "---\nname: source-skill\n---\n").unwrap();
    let cfg = workspace.config();
    let controller = Arc::new(InMemoryWorkspaceController::new_pipefs(
        "workspace",
        "session",
        2,
        true,
        &cfg.paths.workspace_root,
        &cfg.paths.state_root,
    ));
    let mut subject = agent(Vec::new(), cfg);
    subject
        .install_workspace_controller(controller.clone())
        .unwrap();
    let records = Arc::new(Mutex::new(Vec::new()));
    subject.set_session(Box::new(StageSession {
        records: records.clone(),
    }));

    let effect = crate::handle_session_command_coordinated(
        &mut subject,
        &crate::Command::Marketplace(format!("install {}", source.display())),
        &[],
    )
    .await
    .expect("marketplace command is handled");

    assert!(
        !effect.message.contains("not published"),
        "{}",
        effect.message
    );
    assert!(workspace.path(".hi/skills/source-skill/SKILL.md").exists());
    assert_eq!(controller.status().state, WorkspaceState::Ready);
    let records = records.lock().unwrap();
    assert_eq!(records.len(), 1);
    assert_eq!(records[0].calls.len(), 1);
    assert_eq!(records[0].calls[0].name, "session_command");
    assert!(records[0].calls[0].result.contains("source-skill"));
}

#[tokio::test]
async fn closed_admission_prevents_session_command_write() {
    let workspace = IsolatedWorkspace::new("session-command-closed-admission");
    let mut subject = agent(Vec::new(), workspace.config());
    subject
        .begin_durable_workspace_mutation(None)
        .await
        .unwrap();

    let effect = crate::handle_session_command_coordinated(
        &mut subject,
        &crate::Command::Agents("add reviewer stay read-only".into()),
        &[],
    )
    .await
    .expect("agents command is handled");

    assert!(effect.message.contains("was not run"), "{}", effect.message);
    assert!(!workspace.path(".hi/agents/reviewer.md").exists());
    subject
        .checkpoint_durable_workspace_with_execution(ExecutionReport::succeeded(None))
        .await
        .unwrap();
}

#[tokio::test]
async fn coordinated_local_command_reconciles_and_journals_before_success() {
    let workspace = IsolatedWorkspace::new("local-coordinated-command");
    let cfg = workspace.config();
    let state_root = cfg.paths.state_root.clone();
    let mut subject = agent(Vec::new(), cfg);
    let binding = subject.workspace_controller_binding();

    let effect = crate::handle_session_command_coordinated(
        &mut subject,
        &crate::Command::Agents("add reviewer stay read-only".into()),
        &[],
    )
    .await
    .expect("agents command is handled");

    assert!(
        !effect.message.contains("not published"),
        "{}",
        effect.message
    );
    assert!(workspace.path(".hi/agents/reviewer.md").exists());
    assert_eq!(
        subject.workspace_controller_status().state,
        WorkspaceState::Ready
    );
    let operations = ControlStore::open_for_state(&state_root)
        .unwrap()
        .operations_for_binding(binding.binding_id.as_str())
        .unwrap();
    let command = operations.last().expect("command operation was journaled");
    assert_eq!(command.kind, "live_writer");
    assert_eq!(
        command.replay_class,
        OperationReplayClass::NonReplayableExternal
    );
    assert_eq!(command.status, WorkspaceOperationStatus::Durable);
}

#[tokio::test]
async fn failed_session_command_is_durable_but_never_journaled_as_execution_success() {
    let workspace = IsolatedWorkspace::new("failed-coordinated-command");
    let cfg = workspace.config();
    let state_root = cfg.paths.state_root.clone();
    let mut subject = agent(Vec::new(), cfg);
    let binding = subject.workspace_controller_binding();

    let effect = crate::handle_session_command_coordinated(
        &mut subject,
        &crate::Command::Marketplace("install definitely-missing/SKILL.md".into()),
        &[],
    )
    .await
    .expect("marketplace command is handled");

    assert!(
        effect.message.contains("install failed"),
        "{}",
        effect.message
    );
    assert_eq!(
        subject.workspace_controller_status().state,
        WorkspaceState::Ready
    );
    let operations = ControlStore::open_for_state(&state_root)
        .unwrap()
        .operations_for_binding(binding.binding_id.as_str())
        .unwrap();
    let command = operations.last().expect("failed command was journaled");
    assert_eq!(command.status, WorkspaceOperationStatus::Durable);
    assert!(
        command
            .error
            .as_deref()
            .is_some_and(|detail| detail.contains("execution failed")),
        "failed execution detail was lost: {command:?}"
    );
}
