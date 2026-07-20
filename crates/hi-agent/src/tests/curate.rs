//! Integration tests for verifier-gated skill auto-curation ([`Agent::curate_turn_end`]).
//! A canned provider drives the curation call deterministically, so the full glue
//! (trajectory → model call → parse → write + counter) is exercised without a model.

use super::common::*;
use super::*;

// `curate_turn_end` writes through `skills::skill_roots()`, which reads the
// process-global `HI_GLOBAL_SKILLS_DIR`. Serialize the env-mutating tests (async,
// so a tokio mutex — held across the `.await` while the env must stay set).
static ENV_LOCK: std::sync::LazyLock<tokio::sync::Mutex<()>> =
    std::sync::LazyLock::new(|| tokio::sync::Mutex::new(()));

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

fn verified_turn_agent(response: &str, skills_dir: &std::path::Path) -> Agent {
    // SAFETY: serialized by ENV_LOCK; no concurrent reader depends on the default.
    unsafe { std::env::set_var("HI_GLOBAL_SKILLS_DIR", skills_dir) };
    let mut cfg = config();
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
    let _guard = ENV_LOCK.lock().await;
    let dir = unique_dir("write");
    let _ = std::fs::remove_dir_all(&dir);

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
    let written = dir.join("reproduce-before-fixing").join("SKILL.md");
    assert!(
        written.exists(),
        "curated skill should exist at {written:?}"
    );
    let body = std::fs::read_to_string(&written).unwrap();
    assert!(body.contains("name: Reproduce Before Fixing"));
    assert!(body.contains("scope: global"));

    // SAFETY: serialized by ENV_LOCK.
    unsafe { std::env::remove_var("HI_GLOBAL_SKILLS_DIR") };
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn curate_stays_silent_when_model_declines() {
    let _guard = ENV_LOCK.lock().await;
    let dir = unique_dir("silent");
    let _ = std::fs::remove_dir_all(&dir);

    // No frontmatter in the response → the silence path: nothing is written.
    let mut agent = verified_turn_agent("No reusable, general technique here.", &dir);

    let mut ui = NullUi;
    agent.curate_turn_end(0, &mut ui).await;

    assert_eq!(
        agent.subagents.auto_skills_written, 0,
        "a decline must write no skill"
    );
    let empty = std::fs::read_dir(&dir)
        .map(|mut d| d.next().is_none())
        .unwrap_or(true);
    assert!(empty, "no skill dir should be created on the silence path");

    // SAFETY: serialized by ENV_LOCK.
    unsafe { std::env::remove_var("HI_GLOBAL_SKILLS_DIR") };
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn curate_respects_session_cap() {
    let _guard = ENV_LOCK.lock().await;
    let dir = unique_dir("cap");
    let _ = std::fs::remove_dir_all(&dir);

    let mut agent = verified_turn_agent("unused", &dir);
    // Already at the cap: the model is never consulted and nothing is written.
    agent.subagents.auto_skills_written = crate::agent::MAX_AUTO_SKILLS_PER_SESSION;

    let mut ui = NullUi;
    agent.curate_turn_end(0, &mut ui).await;

    assert_eq!(
        agent.subagents.auto_skills_written,
        crate::agent::MAX_AUTO_SKILLS_PER_SESSION
    );
    let empty = std::fs::read_dir(&dir)
        .map(|mut d| d.next().is_none())
        .unwrap_or(true);
    assert!(empty, "capped session must write no further skills");

    // SAFETY: serialized by ENV_LOCK.
    unsafe { std::env::remove_var("HI_GLOBAL_SKILLS_DIR") };
    let _ = std::fs::remove_dir_all(&dir);
}
