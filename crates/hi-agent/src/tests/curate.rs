//! Integration tests for verifier-gated skill auto-curation ([`Agent::curate_turn_end`]).
//! A canned provider drives the curation call deterministically, so the full glue
//! (trajectory → model call → parse → write + counter) is exercised without a model.

use super::common::*;
use super::*;
use hi_workspace::{
    ExecutionReport, InMemoryWorkspaceController, MutationIntent, WorkspaceController,
    WorkspaceState,
};

struct CurateStageSession {
    records: std::sync::Arc<Mutex<Vec<crate::WorkspaceTranscriptExecution>>>,
    fail_after_stage: bool,
}

impl crate::SessionSink for CurateStageSession {
    fn record(&mut self, _: &[Message], _: Usage) -> anyhow::Result<()> {
        Ok(())
    }

    fn stage_workspace_execution(
        &mut self,
        record: &crate::WorkspaceTranscriptExecution,
    ) -> anyhow::Result<()> {
        self.records.lock().unwrap().push(record.clone());
        if self.fail_after_stage {
            anyhow::bail!("synthetic lost curation stage response");
        }
        Ok(())
    }

    fn record_compaction(&mut self, _: &[Message]) -> anyhow::Result<()> {
        Ok(())
    }
}

fn unique_dir(tag: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!(
        "hi-curate-{tag}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ))
}

fn verified_turn_agent(response: &str, workspace: &std::path::Path) -> Agent {
    let mut cfg = config();
    cfg.paths.workspace_root = workspace.to_path_buf();
    cfg.paths.state_root = workspace.join(".hi/state");
    cfg.memory.curate_skills = true;
    Agent::resume(
        std::sync::Arc::new(Canned(Mutex::new(vec![completion(
            vec![Content::Text(response.to_string())],
            1,
            1,
        )]))),
        cfg,
        vec![
            Message::user("count_vowels undercounts and ignores uppercase; fix it"),
            Message::assistant(vec![Content::Text("Fixed by lowercasing first.".into())]),
        ],
        Usage::default(),
        Vec::new(),
        None,
        DecisionLog::default(),
    )
    .unwrap()
}

fn install_pipefs_controller(
    agent: &Agent,
    workspace: &std::path::Path,
) -> std::sync::Arc<InMemoryWorkspaceController> {
    let controller = std::sync::Arc::new(InMemoryWorkspaceController::new_pipefs(
        "curate-workspace",
        "curate-session",
        2,
        true,
        workspace,
        workspace.join(".hi/state"),
    ));
    agent
        .install_workspace_controller(controller.clone())
        .unwrap();
    controller
}

fn curated_skill_response(name: &str) -> String {
    format!(
        "---\nname: {name}\ndescription: Preserve an exact publication boundary.\nscope: project\n---\n# {name}\n\nAdmit the write before touching workspace bytes and settle it before success."
    )
}

#[tokio::test]
async fn curate_writes_skill_from_verified_turn() {
    let dir = unique_dir("write");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    let response = "Here is a reusable technique:\n\n\
         ---\n\
         name: Reproduce Before Fixing\n\
         description: Add a failing test first, then make it pass.\n\
         scope: global\n\
         ---\n\
         # Reproduce Before Fixing\n\n\
         Write a failing test that captures the bug, then fix until it passes.";
    let mut agent = verified_turn_agent(response, &dir);

    let mut ui = RecordingUi::default();
    agent.curate_turn_end(0, &mut ui).await;

    assert_eq!(
        agent.subagents.auto_skills_written, 1,
        "a well-formed SKILL.md should be persisted and counted; statuses: {:?}",
        ui.statuses
    );
    let written = dir
        .join(".hi/skills/reproduce-before-fixing")
        .join("SKILL.md");
    assert!(
        written.exists(),
        "curated skill should exist at {written:?}"
    );
    let body = std::fs::read_to_string(&written).unwrap();
    assert!(body.contains("name: Reproduce Before Fixing"));
    assert!(body.contains("scope: project"));
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn curate_stays_silent_when_model_declines() {
    let dir = unique_dir("silent");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    // No frontmatter in the response → the silence path: nothing is written.
    let mut agent = verified_turn_agent("No reusable, general technique here.", &dir);

    let mut ui = NullUi;
    agent.curate_turn_end(0, &mut ui).await;

    assert_eq!(
        agent.subagents.auto_skills_written, 0,
        "a decline must write no skill"
    );
    let skills = dir.join(".hi/skills");
    let empty = std::fs::read_dir(&skills)
        .map(|mut d| d.next().is_none())
        .unwrap_or(true);
    assert!(empty, "no skill dir should be created on the silence path");

    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn curate_continues_after_the_previous_session_cap() {
    let dir = unique_dir("past-old-cap");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    let response = "---\n\
        name: Preserve Long Plans\n\
        description: Keep every distinct plan item until it is settled.\n\
        scope: project\n\
        ---\n\
        # Preserve Long Plans\n\n\
        Do not discard later objectives merely because earlier work was lengthy.";
    let mut agent = verified_turn_agent(response, &dir);
    agent.subagents.auto_skills_written = 3;

    let mut ui = NullUi;
    agent.curate_turn_end(0, &mut ui).await;

    assert_eq!(agent.subagents.auto_skills_written, 4);
    assert!(
        dir.join(".hi/skills/preserve-long-plans/SKILL.md").exists(),
        "curation must not stop solely because three earlier skills were written"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn closed_admission_prevents_curation_from_writing_workspace_bytes() {
    let dir = unique_dir("closed-admission");
    std::fs::create_dir_all(&dir).unwrap();
    let mut agent = verified_turn_agent(&curated_skill_response("Admit Before Curation"), &dir);
    let controller = install_pipefs_controller(&agent, &dir);
    let permit = controller
        .begin(MutationIntent::workspace("existing writer"))
        .await
        .unwrap();
    let mut ui = RecordingUi::default();

    agent.curate_turn_end(0, &mut ui).await;

    assert_eq!(agent.subagents.auto_skills_written, 0);
    assert!(
        !dir.join(".hi/skills/admit-before-curation/SKILL.md")
            .exists()
    );
    assert!(
        ui.statuses
            .iter()
            .all(|status| !status.starts_with("✓ curated skill"))
    );
    assert_eq!(controller.status().state, WorkspaceState::Mutating);
    let settled = controller
        .settle(permit, ExecutionReport::succeeded(None))
        .await;
    assert!(settled.receipt.is_some());

    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn lost_pipefs_stage_response_keeps_curated_bytes_recovery_blocked_and_hides_success() {
    let dir = unique_dir("lost-stage");
    std::fs::create_dir_all(&dir).unwrap();
    let mut agent = verified_turn_agent(&curated_skill_response("Recover Curated Skill"), &dir);
    let controller = install_pipefs_controller(&agent, &dir);
    let records = std::sync::Arc::new(Mutex::new(Vec::new()));
    agent.set_session(Box::new(CurateStageSession {
        records: records.clone(),
        fail_after_stage: true,
    }));
    let mut ui = RecordingUi::default();

    agent.curate_turn_end(0, &mut ui).await;

    assert!(
        dir.join(".hi/skills/recover-curated-skill/SKILL.md")
            .exists(),
        "the test must exercise ambiguity after the filesystem effect"
    );
    assert_eq!(agent.subagents.auto_skills_written, 0);
    assert!(
        ui.statuses
            .iter()
            .all(|status| !status.starts_with("✓ curated skill"))
    );
    assert!(ui.statuses.iter().any(|status| {
        status.contains("skill not saved") && status.contains("transcript staging failed")
    }));
    assert_eq!(controller.status().state, WorkspaceState::RecoveryRequired);
    let records = records.lock().unwrap();
    assert_eq!(records.len(), 1);
    assert_eq!(records[0].calls.len(), 1);
    assert_eq!(records[0].calls[0].name, "auto_curate_skill");
    assert!(
        records[0].calls[0]
            .result
            .contains("workspace://.hi/skills/recover-curated-skill/SKILL.md"),
        "{}",
        records[0].calls[0].result
    );
    assert!(
        !records[0].calls[0]
            .result
            .contains(&dir.display().to_string()),
        "remote transcript must not expose the local materialization path"
    );
    assert_eq!(
        records[0].execution.disposition,
        hi_workspace::ExecutionDisposition::Succeeded
    );

    let _ = std::fs::remove_dir_all(&dir);
}
