//! `/review` on a large workspace with no plan/spec, end to end against
//! `MockPipe`: the default narrows to recent work per git, `all` runs one
//! audit turn per chunk, and a repository with nothing recent refuses.

use super::*;
use crate::pipe::test_support::MockPipe;
use crate::review_drive::{ReviewCommand, ReviewPhase, ReviewStep};
use crate::review_harness_tests::{run, start, test_harness, text};
use crate::review_scope::LARGE_WORKSPACE_FILES;
use std::fs;
use std::path::Path;
use std::process::Command;

fn git(root: &Path, args: &[&str]) {
    let output = Command::new("git")
        .args(["-c", "user.email=t@example.com", "-c", "user.name=t"])
        .args(args)
        .current_dir(root)
        .output()
        .expect("git runs");
    assert!(
        output.status.success(),
        "git {args:?}: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

/// A repository over the size limit with a README and no plan/spec.
fn large_repo(root: &Path) {
    git(root, &["init", "-q"]);
    git(root, &["config", "commit.gpgsign", "false"]);
    fs::create_dir_all(root.join("src")).unwrap();
    for index in 0..=LARGE_WORKSPACE_FILES {
        fs::write(root.join(format!("src/f{index}.rs")), "").unwrap();
    }
    fs::write(root.join("README.md"), "# Chat\n\nGreets with Welcome.\n").unwrap();
    fs::write(root.join(".gitignore"), "/.hi/\n").unwrap();
}

const CLEAN: &str = "Read the two changed files; nothing wrong.\n\n<review>\nverdict: COMPLETE\nfinding: none\n</review>";

#[tokio::test]
async fn large_repo_without_a_spec_audits_uncommitted_work_then_the_last_commit() {
    let dir = tempfile::tempdir().unwrap();
    large_repo(dir.path());
    git(dir.path(), &["add", "-A"]);
    git(dir.path(), &["commit", "-q", "-m", "import"]);
    fs::write(dir.path().join("src/f1.rs"), "pub fn nick() {}\n").unwrap();
    fs::write(dir.path().join("src/new.rs"), "").unwrap();
    let Some(server) = MockPipe::new(vec![text(CLEAN), text(CLEAN)]) else {
        return;
    };
    let mut harness = test_harness(&server.url, dir.path().to_path_buf());

    let (prompt, notes) = start(&mut harness, "audit");
    assert_eq!(
        notes,
        vec![
            "spec review · audit only · 2 uncommitted files (git status) · no plan/spec (defects only)".to_string(),
            "spec review: no plan.md or spec.md found; auditing recent work for defects: 2 uncommitted files (git status) · `/review audit all` audits the whole repo in chunks".to_string(),
        ]
    );
    assert!(
        prompt.starts_with("[hi:review] Defect audit (up to"),
        "{prompt}"
    );
    assert!(
        prompt.contains("Files:\n- src/f1.rs\n- src/new.rs\n"),
        "{prompt}"
    );
    assert!(
        !prompt.contains("readme: README.md"),
        "a README is not a spec for recent work"
    );
    assert!(prompt.contains("write no `coverage:` rows"), "{prompt}");
    let drive = harness.review_drive();
    assert_eq!(drive.inputs.files, vec![]);
    assert_eq!(
        drive.inputs.git.as_ref().map(|git| git.files.clone()),
        Some(vec!["src/f1.rs".to_string(), "src/new.rs".to_string()])
    );

    let mut ui = TestUi::default();
    let outcome = run(&mut harness, &prompt, &mut ui).await;
    assert_eq!(outcome.stop_reason, TurnStopReason::Completed);
    match harness.review_next_step(&outcome, &mut ui) {
        ReviewStep::Done(summary) => assert_eq!(
            summary,
            "spec review complete: 2 uncommitted files (git status) audited, no plan/spec items · no P0/P1 defects"
        ),
        other => panic!("clean defects-only audit finishes, got {other:?}"),
    }
    assert!(harness.review_drive().coverage_complete());
    let report = harness.review_drive().report_lines();
    assert_eq!(
        report[0],
        "coverage: none (no plan/spec items; write plan.md for a coverage audit)"
    );
    assert_eq!(report[1], "findings: none");
    assert!(
        report
            .last()
            .unwrap()
            .starts_with("next: this audited the uncommitted changes only;"),
        "{report:?}"
    );

    // Clean tree: the last commit is the recent work.
    git(dir.path(), &["add", "-A"]);
    git(dir.path(), &["commit", "-q", "-m", "nick parsing"]);
    let (prompt, notes) = start(&mut harness, "audit");
    let short = notes[0]
        .split("last commit ")
        .nth(1)
        .and_then(|rest| rest.split_whitespace().next())
        .expect("status names the commit")
        .to_string();
    assert_eq!(
        notes[0],
        format!(
            "spec review · audit only · last commit {short} (2 files) · no plan/spec (defects only)"
        )
    );
    assert!(
        prompt.contains(&format!(
            "changed by the last commit {short} \"nick parsing\"; the tree is clean"
        )),
        "{prompt}"
    );
    assert!(prompt.contains("Run `git show HEAD`"), "{prompt}");
    let outcome = run(&mut harness, &prompt, &mut ui).await;
    assert!(matches!(
        harness.review_next_step(&outcome, &mut ui),
        ReviewStep::Done(_)
    ));
    assert_eq!(server.bodies.lock().unwrap().len(), 2);

    // Nothing recent: an empty last commit on a clean tree refuses, and
    // says what to pass instead.
    git(
        dir.path(),
        &["commit", "-q", "--allow-empty", "-m", "nothing"],
    );
    let ReviewCommand::Message(refusal) = harness.review_command("audit") else {
        panic!("nothing recent to audit must not start a review");
    };
    assert!(refusal.contains("more than 200 files"), "{refusal}");
    assert!(
        refusal.contains("no commit whose files are still in the tree"),
        "{refusal}"
    );
    assert!(refusal.contains("`/review audit all`"), "{refusal}");
    assert!(
        !harness.review_drive().is_active(),
        "the finished drive stays finished"
    );
    assert_eq!(harness.review_drive().phase, ReviewPhase::Done);
}

#[tokio::test]
async fn review_all_audits_the_repo_one_chunk_per_turn() {
    let dir = tempfile::tempdir().unwrap();
    large_repo(dir.path());
    fs::create_dir_all(dir.path().join("crates/hi-a/src")).unwrap();
    fs::write(dir.path().join("crates/hi-a/src/lib.rs"), "").unwrap();
    fs::create_dir_all(dir.path().join("crates/hi-b")).unwrap();
    fs::write(dir.path().join("crates/hi-b/Cargo.toml"), "").unwrap();
    let chunk_a = "Read crates/hi-a.\n\n<review>\nverdict: INCOMPLETE\n\
finding: P1 | Close the listener on shutdown | crates/hi-a/src/lib.rs:1\n</review>";
    let chunk_b = "Read crates/hi-b.\n\n<review>\nverdict: COMPLETE\n\
finding: P2 | Pin the edition | crates/hi-b/Cargo.toml:1\n</review>";
    let chunk_src = "Read src.\n\n<review>\nverdict: INCOMPLETE\n\
finding: P1 | Close the listener on shutdown | crates/hi-a/src/lib.rs:1\n\
finding: P1 | Reject empty nicknames | src/f1.rs:1\n</review>";
    let top = "Read the top-level files.\n\n<review>\nverdict: COMPLETE\nfinding: none\n</review>";
    let Some(server) = MockPipe::new(vec![
        text(chunk_a),
        text(chunk_b),
        text(chunk_src),
        text(top),
        text("must not be requested"),
    ]) else {
        return;
    };
    let mut harness = test_harness(&server.url, dir.path().to_path_buf());

    let (mut prompt, notes) = start(&mut harness, "audit all");
    assert_eq!(
        notes,
        vec![
            "spec review · audit only · chunk 1/4 · crates/hi-a · no plan/spec (defects only)".to_string(),
            "spec review: no plan.md or spec.md found; auditing the whole workspace for defects in 4 chunks, one turn each: \
crates/hi-a, crates/hi-b, src, top-level files".to_string(),
        ]
    );
    assert_eq!(
        harness.review_drive().inputs.chunks,
        vec![
            "crates/hi-a".to_string(),
            "crates/hi-b".to_string(),
            "src".to_string(),
            ".".to_string()
        ]
    );
    assert!(
        harness.review_drive().inputs.git.is_none(),
        "`all` does not narrow to git"
    );
    let mut ui = TestUi::default();
    let mut labels = Vec::new();
    let summary = loop {
        labels.push(crate::review::transcript_label(&prompt).unwrap());
        let outcome = run(&mut harness, &prompt, &mut ui).await;
        assert_eq!(outcome.stop_reason, TurnStopReason::Completed);
        match harness.review_next_step(&outcome, &mut ui) {
            ReviewStep::RunPrompt(next) => prompt = next,
            ReviewStep::Done(summary) => break summary,
            other => panic!("chunks run back to back, got {other:?}"),
        }
    };
    assert_eq!(
        labels,
        vec![
            "/review · defect audit, chunk 1/4: crates/hi-a",
            "/review · defect audit, chunk 2/4: crates/hi-b",
            "/review · defect audit, chunk 3/4: src",
            "/review · defect audit, chunk 4/4: top-level files",
        ]
    );
    assert_eq!(
        summary,
        "spec review complete: 4 chunks audited, no plan/spec items · 2 P0/P1 defect(s) still open (audit only) · 1 P2/P3 reported"
    );
    let drive = harness.review_drive();
    assert_eq!(drive.phase, ReviewPhase::Done);
    assert_eq!(drive.chunk, 4);
    let titles: Vec<&str> = drive
        .last_verdict
        .as_ref()
        .unwrap()
        .findings
        .iter()
        .map(|finding| finding.title.as_str())
        .collect();
    assert_eq!(
        titles,
        vec![
            "Close the listener on shutdown",
            "Pin the edition",
            "Reject empty nicknames"
        ],
        "one merged verdict; the repeated P1 counts once"
    );
    assert_eq!(harness.review_open_blocking().len(), 2);
    assert!(
        ui.statuses
            .iter()
            .any(|status| status.contains("chunk 3/4 · src")),
        "each chunk turn announces its status line, got {:?}",
        ui.statuses
    );
    assert_eq!(server.bodies.lock().unwrap().len(), 4);
    let bodies = server.bodies.lock().unwrap();
    assert!(
        bodies[2].contains("Scope: chunk 3/4 of the workspace: `src/`"),
        "each chunk turn fences its directory"
    );
    assert!(
        bodies[3].contains("the files directly in the workspace root"),
        "{}",
        bodies[3]
    );
}
