//! `/review` on a workspace with no plan/spec, end to end against
//! `MockPipe`: a large one narrows to recent work per git (with read
//! tracking and a continuation turn), `all` runs one audit turn per chunk,
//! and a repository with nothing recent refuses.

use super::*;
use crate::pipe::test_support::{MockPipe, Scripted};
use crate::review::ReviewSettings;
use crate::review_drive::{ReviewCommand, ReviewPhase, ReviewStep};
use crate::review_harness_tests::{run, start, test_harness, text, tool};
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

fn init_repo(root: &Path) {
    git(root, &["init", "-q"]);
    git(root, &["config", "commit.gpgsign", "false"]);
    fs::write(root.join(".gitignore"), "/.hi/\n").unwrap();
}

/// A repository over the size limit with a README and no plan/spec.
fn large_repo(root: &Path) {
    init_repo(root);
    fs::create_dir_all(root.join("src")).unwrap();
    for index in 0..=LARGE_WORKSPACE_FILES {
        fs::write(root.join(format!("src/f{index}.rs")), "").unwrap();
    }
    fs::write(root.join("README.md"), "# Chat\n\nGreets with Welcome.\n").unwrap();
}

fn read(id: &str, paths: &[&str]) -> Scripted {
    tool(id, "read", &serde_json::json!({ "paths": paths }))
}

const CLEAN: &str = "Read the changed files; nothing wrong.\n\n<review>\nverdict: COMPLETE\nfinding: none\n</review>";

#[tokio::test]
async fn large_repo_without_a_spec_audits_uncommitted_work_then_the_last_commit() {
    let dir = tempfile::tempdir().unwrap();
    large_repo(dir.path());
    git(dir.path(), &["add", "-A"]);
    git(dir.path(), &["commit", "-q", "-m", "import"]);
    fs::write(dir.path().join("src/f1.rs"), "pub fn nick() {}\n").unwrap();
    fs::write(dir.path().join("src/new.rs"), "pub fn topic() {}\n").unwrap();
    let Some(server) = MockPipe::new(vec![
        // Audit of the uncommitted work: one file read, the verdict is
        // partial, the continuation reads the other.
        read("call_r1", &["src/f1.rs"]),
        text(CLEAN),
        read("call_r2", &["src/new.rs"]),
        text(CLEAN),
        // Audit of the last commit: `git show` output names both files.
        tool(
            "call_s",
            "bash",
            &serde_json::json!({ "command": "git show HEAD" }),
        ),
        text(CLEAN),
    ]) else {
        return;
    };
    let mut harness = test_harness(&server.url, dir.path().to_path_buf());

    let (prompt, notes) = start(&mut harness, "audit");
    assert_eq!(
        notes,
        vec![
            "spec review · audit only · 2 uncommitted files (git status, +2 −0) · no plan/spec (defects only)".to_string(),
            "spec review: no plan.md or spec.md found; auditing recent work for defects: 2 uncommitted files (git status, +2 −0) · `/review audit all` audits the whole repo in chunks".to_string(),
        ]
    );
    assert!(
        prompt.starts_with("[hi:review] Defect audit (up to"),
        "{prompt}"
    );
    assert!(
        prompt.contains(&format!(
            "Workspace root: {} (every path below",
            dir.path().display()
        )),
        "{prompt}"
    );
    assert!(
        prompt.contains(
            "Files (read every one):\n- src/f1.rs (+1 −0)\n- src/new.rs (untracked, 1 lines)\n"
        ),
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
    assert!(drive.inputs.chunks.is_empty(), "two files are one turn");

    let mut ui = TestUi::default();
    let outcome = run(&mut harness, &prompt, &mut ui).await;
    assert_eq!(outcome.stop_reason, TurnStopReason::Completed);
    let continuation = match harness.review_next_step(&outcome, &mut ui) {
        ReviewStep::RunPrompt(prompt) => prompt,
        other => panic!("one of two files unread: the audit continues, got {other:?}"),
    };
    assert!(
        continuation.starts_with("[hi:review] Defect audit, continued (1/2). The last turn left 1 of the files under audit unread"),
        "{continuation}"
    );
    assert!(
        continuation
            .contains("Unread files (read every one):\n- src/new.rs (untracked, 1 lines)\n"),
        "{continuation}"
    );
    assert_eq!(
        harness.review_drive().read_files,
        vec!["src/f1.rs".to_string()]
    );
    assert_eq!(
        harness.review_drive().status_line(),
        "spec review · audit only · continued 1/2 · 2 uncommitted files (git status, +2 −0) · no plan/spec (defects only)"
    );
    let outcome = run(&mut harness, &continuation, &mut ui).await;
    match harness.review_next_step(&outcome, &mut ui) {
        ReviewStep::Done(summary) => assert_eq!(
            summary,
            "spec review complete: 2 uncommitted files (git status, +2 −0) audited, no plan/spec items · no P0/P1 defects · read 2/2 files in scope"
        ),
        other => panic!("every file read: the defects-only audit finishes, got {other:?}"),
    }
    assert!(harness.review_drive().coverage_complete());
    assert!(!harness.review_drive().is_partial());
    let report = harness.review_drive().report_lines();
    assert_eq!(
        report,
        vec![
            "coverage: none (no plan/spec items; write plan.md for a coverage audit)".to_string(),
            "findings: none".to_string(),
            "read 2/2 files in scope".to_string(),
            "next: this audited the uncommitted changes only; write plan.md (a `- [ ]` checklist of what should exist) or pass one for a coverage audit: `/review docs/spec.md [dir]`; `/review audit all` audits the whole repo in chunks".to_string(),
        ]
    );

    // Clean tree: the last commit is the recent work; the diff headers of
    // `git show` count as reads of both files.
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
            "spec review · audit only · last commit {short} (2 files, +2 −0) · no plan/spec (defects only)"
        )
    );
    assert!(
        prompt.contains(&format!(
            "changed by the last commit {short} \"nick parsing\"; the tree is clean"
        )),
        "{prompt}"
    );
    assert!(prompt.contains("`git show HEAD -- <path>`"), "{prompt}");
    let outcome = run(&mut harness, &prompt, &mut ui).await;
    match harness.review_next_step(&outcome, &mut ui) {
        ReviewStep::Done(summary) => assert!(
            summary.ends_with("· no P0/P1 defects · read 2/2 files in scope"),
            "{summary}"
        ),
        other => panic!("expected Done, got {other:?}"),
    }
    assert_eq!(
        harness.review_drive().read_files,
        vec!["src/f1.rs".to_string(), "src/new.rs".to_string()]
    );
    assert_eq!(server.bodies.lock().unwrap().len(), 6);

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
    init_repo(dir.path());
    for file in [
        "crates/hi-a/src/lib.rs",
        "crates/hi-a/src/net.rs",
        "crates/hi-b/Cargo.toml",
        "crates/hi-b/src/lib.rs",
        "src/f1.rs",
        "src/f2.rs",
        "src/f3.rs",
        "README.md",
    ] {
        let path = dir.path().join(file);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, "// code\n").unwrap();
    }
    let chunk_a = "Read crates/hi-a.\n\n<review>\nverdict: INCOMPLETE\n\
finding: P1 | Close the listener on shutdown | crates/hi-a/src/lib.rs:1\n</review>";
    let chunk_b = "Read crates/hi-b.\n\n<review>\nverdict: COMPLETE\n\
finding: P2 | Pin the edition | crates/hi-b/Cargo.toml:1\n</review>";
    let chunk_src = "Read src.\n\n<review>\nverdict: INCOMPLETE\n\
finding: P1 | Close the listener on shutdown | crates/hi-a/src/lib.rs:1\n\
finding: P1 | Reject empty nicknames | src/f1.rs:1\n</review>";
    let top = "Read the top-level files.\n\n<review>\nverdict: COMPLETE\nfinding: none\n</review>";
    let Some(server) = MockPipe::new(vec![
        read(
            "call_a",
            &["crates/hi-a/src/lib.rs", "crates/hi-a/src/net.rs"],
        ),
        text(chunk_a),
        read(
            "call_b",
            &["crates/hi-b/Cargo.toml", "crates/hi-b/src/lib.rs"],
        ),
        text(chunk_b),
        read("call_c", &["src/f1.rs", "src/f2.rs", "src/f3.rs"]),
        text(chunk_src),
        read("call_d", &["README.md"]),
        text(top),
        text("must not be requested"),
    ]) else {
        return;
    };
    let mut harness = test_harness(&server.url, dir.path().to_path_buf());
    // `crates/` (4 files) is over the budget and splits per crate; `src/`
    // (3) fits; nothing packs with its neighbour.
    harness.set_review_settings(ReviewSettings {
        chunk_files: 3,
        ..ReviewSettings::default()
    });

    let (mut prompt, notes) = start(&mut harness, "audit all");
    assert_eq!(
        notes,
        vec![
            "spec review · audit only · chunk 1/4 · crates/hi-a · no plan/spec (defects only)".to_string(),
            "spec review: no plan.md or spec.md found; auditing the whole workspace for defects in 4 chunks, one turn each: \
crates/hi-a; crates/hi-b; src; top-level files".to_string(),
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
        "spec review complete: 4 chunks audited, no plan/spec items · 2 P0/P1 defect(s) still open (audit only) · read 8/8 files in scope · 1 P2/P3 reported"
    );
    let drive = harness.review_drive();
    assert_eq!(drive.phase, ReviewPhase::Done);
    assert_eq!(drive.chunk, 4);
    assert_eq!(drive.scope_count, 8, "the chunks' files, summed");
    assert_eq!(drive.unread_count, 0);
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
    let bodies = server.bodies.lock().unwrap();
    assert_eq!(bodies.len(), 8, "a read and a verdict per chunk");
    let src_turn = bodies
        .iter()
        .find(|body| body.contains("Scope: chunk 3/4 of the workspace: `src/`"))
        .expect("each chunk turn fences its directory");
    assert!(
        src_turn.contains(
            "Files in scope (3; the report counts which of them were read):\\n- src/f1.rs\\n- src/f2.rs\\n- src/f3.rs\\n"
        ),
        "{src_turn}"
    );
    assert!(
        bodies
            .iter()
            .any(|body| body.contains("the files directly in the workspace root")),
        "the top-level chunk is described"
    );
}

/// A large workspace with a README and no plan/spec: an explicit small
/// directory is the scope and the README stays the fallback input; a
/// directory bigger than one turn is cut into chunks and the README is
/// dropped (a defects-only audit has nothing to audit against).
#[tokio::test]
async fn explicit_directory_is_the_scope_or_is_chunked_when_big() {
    let dir = tempfile::tempdir().unwrap();
    large_repo(dir.path());
    fs::create_dir_all(dir.path().join("docs")).unwrap();
    fs::write(dir.path().join("docs/notes.md"), "notes\n").unwrap();
    let Some(server) = MockPipe::new(vec![]) else {
        return;
    };
    let mut harness = test_harness(&server.url, dir.path().to_path_buf());
    let (scoped, notes) = start(&mut harness, "audit docs");
    assert!(
        scoped.contains("Scope: limit the code audit to docs"),
        "{scoped}"
    );
    assert!(
        scoped.contains(
            "Files in scope (1; the report counts which of them were read):\n- docs/notes.md\n"
        ),
        "{scoped}"
    );
    assert!(
        scoped.contains("- readme: README.md (fallback: no plan.md or spec.md;"),
        "{scoped}"
    );
    assert_eq!(notes[0], "spec review · audit only · in docs · README.md");
    assert_eq!(
        harness.review_scope_files(),
        Some(vec!["docs/notes.md".to_string()])
    );

    let (chunked, notes) = start(&mut harness, "audit src");
    assert!(
        chunked.contains("Scope: chunk 1/13 of the workspace: `src/` part 1/13."),
        "{chunked}"
    );
    assert!(
        !chunked.contains("readme: README.md"),
        "a chunked defects-only audit has no README to audit against"
    );
    assert_eq!(
        notes[0],
        "spec review · audit only · chunk 1/13 · src (1/13) · no plan/spec (defects only)"
    );
    assert!(
        notes[1].starts_with(
            "spec review: no plan.md or spec.md found; auditing src for defects in 13 chunks, one turn each:"
        ),
        "{}",
        notes[1]
    );
    assert_eq!(
        harness.review_scope_files().map(|files| files.len()),
        Some(16)
    );
    assert!(server.bodies.lock().unwrap().is_empty());
}

#[tokio::test]
async fn review_all_on_a_large_repo_chunks_the_tree_instead_of_narrowing_to_git() {
    let dir = tempfile::tempdir().unwrap();
    large_repo(dir.path());
    let Some(server) = MockPipe::new(vec![]) else {
        return;
    };
    let mut harness = test_harness(&server.url, dir.path().to_path_buf());
    let (prompt, notes) = start(&mut harness, "audit all");
    let drive = harness.review_drive();
    assert!(drive.inputs.git.is_none());
    // 201 files in `src/` at the default budget of 16: 13 slices, the
    // README rides with the last (9-file) slice.
    assert_eq!(drive.inputs.chunks.len(), 13, "{:?}", drive.inputs.chunks);
    assert_eq!(drive.inputs.chunks[0], "src (1/13)");
    assert_eq!(drive.inputs.chunks[12], "src (13/13), .");
    assert_eq!(
        notes[0],
        "spec review · audit only · chunk 1/13 · src (1/13) · no plan/spec (defects only)"
    );
    assert!(
        notes[1].starts_with(
            "spec review: no plan.md or spec.md found; auditing the whole workspace for defects in 13 chunks, one turn each: src (1/13); src (2/13);"
        ),
        "{}",
        notes[1]
    );
    assert!(notes[1].ends_with("(+5 more)"), "{}", notes[1]);
    assert!(
        prompt.contains("Scope: chunk 1/13 of the workspace: `src/` part 1/13."),
        "{prompt}"
    );
    assert!(
        prompt.contains(
            "Files in scope (16; the report counts which of them were read):\n- src/f0.rs\n"
        ),
        "{prompt}"
    );
    assert!(server.bodies.lock().unwrap().is_empty());
}
