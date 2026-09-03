//! Integration tests for verifier-gated skill auto-curation ([`Agent::curate_turn_end`]).
//! A canned provider drives the curation call deterministically, so the full glue
//! (trajectory → model call → parse → write + counter) is exercised without a model.

use super::common::*;
use super::*;

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

    let mut ui = NullUi;
    agent.curate_turn_end(0, &mut ui).await;

    assert_eq!(
        agent.subagents.auto_skills_written, 1,
        "a well-formed SKILL.md should be persisted and counted"
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
