//! Prompt text for the `/review` turns: the audit, fix, re-audit, and
//! format re-ask prompts carry the prefix, the inputs, and the verdict
//! contract; a narrowed (git-scoped, chunked, or defects-only) audit
//! names its scope and drops the coverage row.

use super::*;
use crate::review::{ChecklistItem, Finding, InputKind, ReviewInput, ReviewInputs, Severity};
use crate::review_git::{GitNumstat, GitScope, GitSource};

fn git_scope(source: GitSource, files: Vec<String>) -> GitScope {
    GitScope {
        source,
        files,
        numstat: Vec::new(),
        groups: Vec::new(),
    }
}

#[test]
fn prompts_carry_prefix_inputs_and_contract() {
    let inputs = ReviewInputs {
        files: vec![
            ReviewInput {
                path: "plan.md".into(),
                kind: InputKind::Plan,
            },
            ReviewInput {
                path: "docs/spec.md".into(),
                kind: InputKind::Spec,
            },
        ],
        scope: vec!["src".into()],
        ..ReviewInputs::default()
    };
    let checklist = vec![(
        "plan.md".to_string(),
        vec![
            ChecklistItem {
                text: "a".into(),
                checked: Some(true),
            },
            ChecklistItem {
                text: "b".into(),
                checked: Some(false),
            },
        ],
    )];
    let audit = audit_prompt(&inputs, &checklist, 0, 3, None);
    assert!(audit.starts_with(REVIEW_PREFIX));
    assert!(crate::is_harness_injection(&audit));
    assert!(
        audit.contains(
            "plan: plan.md (2 checklist rows: 1 checked, claims to verify; 1 unchecked, known gaps: \
report each as a `missing`/`partial` coverage row, never as a `finding:` row)"
        ),
        "{audit}"
    );
    assert!(audit.contains("spec: docs/spec.md"));
    assert!(
        audit.contains("The block itself is machine-read and shown as a table"),
        "{audit}"
    );

    // Numbered rows are claims, not gaps; a README is never a checklist.
    let numbered = vec![(
        "plan.md".to_string(),
        vec![
            ChecklistItem {
                text: "streams from the API".into(),
                checked: None,
            },
            ChecklistItem {
                text: "done".into(),
                checked: Some(true),
            },
        ],
    )];
    let audit_numbered = audit_prompt(&inputs, &numbered, 0, 3, None);
    assert!(
        audit_numbered
            .contains("plan: plan.md (2 checklist rows: 1 checked, claims to verify; 1 numbered, claims to verify)"),
        "{audit_numbered}"
    );
    assert!(!audit_numbered.contains("known gaps"));
    let readme_inputs = ReviewInputs {
        files: vec![ReviewInput {
            path: "README.md".into(),
            kind: InputKind::Readme,
        }],
        ..ReviewInputs::default()
    };
    let readme_audit = audit_prompt(&readme_inputs, &[], 0, 3, None);
    assert!(
        readme_audit.contains(
            "- readme: README.md (fallback: no plan.md or spec.md; the features this README claims are the plan/spec items, install steps and usage tips are not)"
        ),
        "{readme_audit}"
    );
    assert!(audit.contains("Scope: limit the code audit to src"));
    assert!(audit.contains("<review>") && audit.contains("</review>"));
    assert!(audit.contains("do not call update_plan"));
    assert_eq!(
        transcript_label(&audit).as_deref(),
        Some("/review · spec-coverage audit")
    );

    let finding = Finding {
        severity: Severity::P0,
        title: "Reject empty nicknames".into(),
        location: Some("src/server.rs:88".into()),
        verified: true,
        feature_gap: false,
        from_verify: false,
    };
    let fix = fix_prompt(std::slice::from_ref(&finding), 1, 3);
    assert!(fix.starts_with(REVIEW_PREFIX));
    assert!(fix.contains("1. [P0] Reject empty nicknames — src/server.rs:88"));
    assert!(fix.contains("run the project's tests after the last edit"));
    assert!(crate::completion::user_asked_to_fix(&fix));
    assert_eq!(
        transcript_label(&fix).as_deref(),
        Some("/review · fix pass 1/3")
    );

    let items = vec!["a".to_string(), "b".to_string()];
    let reaudit = reaudit_prompt(
        &inputs,
        &items,
        &["src/server.rs".into(), "src/other.rs".into()],
        &["src/other.rs".into()],
        std::slice::from_ref(&finding),
        1,
        3,
    );
    assert!(reaudit.contains("- src/server.rs\n"));
    assert!(
        reaudit.contains("Of those, files no finding cited")
            && reaudit.contains("- src/other.rs\n"),
        "{reaudit}"
    );
    assert!(reaudit.contains("Prior findings to re-check"));
    assert!(
        reaudit.contains("Plan/spec items (one `coverage:` row each, in this order):\n- a\n- b\n")
    );
    assert!(reaudit.contains("Report; do not plan."));
    assert!(
        reaudit.contains("End your reply with exactly this block, filled in"),
        "{reaudit}"
    );
    assert!(
        reaudit.contains(
            "coverage: <state> | a | <path:line or ->\ncoverage: <state> | b | <path:line or ->\n"
        ),
        "known items pre-fill the re-audit's block: {reaudit}"
    );
    assert!(!reaudit.contains("<plan or spec item>"), "{reaudit}");
    assert!(reaudit.contains("</review>"));
    assert_eq!(
        transcript_label(&reaudit).as_deref(),
        Some("/review · re-audit after fix pass 1/3")
    );
    let bare = reaudit_prompt(&inputs, &[], &[], &[], &[], 2, 3);
    assert!(!bare.contains("Plan/spec items"));
    assert!(bare.contains("The fix pass changed no files."));
    assert!(!bare.contains("files no finding cited"));
    assert!(
        bare.contains(REVIEW_FORMAT_BLOCK),
        "no items: generic contract"
    );
    assert!(
        audit.contains(REVIEW_FORMAT_BLOCK),
        "the first audit names its own rows"
    );

    assert_eq!(
        transcript_label(REVIEW_FORMAT_HINT).as_deref(),
        Some("/review · format re-ask")
    );
    assert_eq!(transcript_label("plain prompt"), None);
    assert!(REVIEW_FORMAT_HINT.contains(REVIEW_FORMAT_BLOCK));
    assert!(REVIEW_FORMAT_HINT.contains("do not write a plan or next steps"));
    assert_eq!(format_reask_prompt(&[], &[], false), REVIEW_FORMAT_HINT);

    let form = format_reask_prompt(&items, std::slice::from_ref(&finding), false);
    assert!(form.starts_with(REVIEW_PREFIX));
    assert!(crate::is_harness_injection(&form));
    assert_eq!(
        transcript_label(&form).as_deref(),
        Some("/review · format re-ask")
    );
    assert!(form.contains("fill in `<state>`"), "{form}");
    assert!(
        form.contains(
            "coverage: <state> | a | <path:line or ->\ncoverage: <state> | b | <path:line or ->\n"
        ),
        "{form}"
    );
    assert!(!form.contains("<plan or spec item>"), "{form}");
    assert!(
        form.contains("- [P0] Reject empty nicknames — src/server.rs:88\n"),
        "{form}"
    );
    assert!(form.ends_with("</review>"), "{form}");
    let first_sentence = REVIEW_FORMAT_HINT.split(". ").next().unwrap();
    assert!(form.starts_with(first_sentence), "{form}");

    let prior_only = format_reask_prompt(&[], std::slice::from_ref(&finding), false);
    assert!(prior_only.contains("one `coverage:` row per plan/spec item"));
    assert!(prior_only.contains("<plan or spec item>"));
    assert!(prior_only.contains("Prior findings"));
}

/// The prompts for a narrowed audit: the git scope names its files and how
/// to see the diff, a chunk turn fences its directory, and a defects-only
/// audit gets the contract without a coverage row.
#[test]
fn narrowed_audit_prompts_name_scope_and_drop_coverage() {
    let mut scope = git_scope(
        GitSource::Uncommitted,
        (0..45).map(|i| format!("src/f{i}.rs")).collect(),
    );
    scope.numstat = vec![
        GitNumstat {
            path: "src/f0.rs".into(),
            added: Some(12),
            removed: Some(3),
            untracked: false,
        },
        GitNumstat {
            path: "src/f1.rs".into(),
            added: Some(80),
            removed: Some(0),
            untracked: true,
        },
        GitNumstat {
            path: "src/f2.rs".into(),
            added: None,
            removed: None,
            untracked: false,
        },
    ];
    let git = ReviewInputs {
        git: Some(scope),
        root: Some("/work/app".into()),
        ..ReviewInputs::default()
    };
    assert!(git.defects_only());
    let audit = audit_prompt(&git, &[], 0, 3, None);
    assert!(
        audit.starts_with("[hi:review] Defect audit (up to 3 fix passes"),
        "{audit}"
    );
    assert_eq!(
        transcript_label(&audit).as_deref(),
        Some("/review · defect audit")
    );
    assert!(
        audit.contains("Workspace root: /work/app (every path below is relative to it; cite paths the same way).\n"),
        "{audit}"
    );
    assert!(
        audit.contains("Inputs: none found (no plan.md or spec.md). This is a defects-only audit"),
        "{audit}"
    );
    assert!(
        audit.contains(
            "Code under audit: the 45 file(s) listed below, uncommitted per `git status`"
        ),
        "{audit}"
    );
    assert!(audit.contains("`git diff HEAD -- <path>`"), "{audit}");
    assert!(
        audit.contains(
            "Files (read every one):\n- src/f0.rs (+12 −3)\n- src/f1.rs (untracked, 80 lines)\n- src/f2.rs (binary)\n- src/f3.rs\n"
        ),
        "{audit}"
    );
    assert!(
        audit.contains("- … and 5 more (`git status --short` lists them)\n"),
        "{audit}"
    );
    assert!(audit.contains(REVIEW_DEFECTS_FORMAT_BLOCK), "{audit}");
    assert!(
        !audit.contains("coverage: implemented|partial|missing"),
        "{audit}"
    );
    assert!(
        audit.contains("`verdict: COMPLETE` when no P0/P1 finding is listed"),
        "{audit}"
    );
    assert!(
        audit.contains("the harness records which files were read and asks again for the rest"),
        "{audit}"
    );
    assert!(!audit.contains("Scope: chunk"), "not chunked");

    // The same scope in turns: each names its own files and the boundary.
    let mut turns = git.clone();
    let scope = turns.git.as_mut().unwrap();
    scope.groups = vec![
        (0..16).map(|i| format!("src/f{i}.rs")).collect(),
        (16..32).map(|i| format!("src/f{i}.rs")).collect(),
        (32..45).map(|i| format!("src/f{i}.rs")).collect(),
    ];
    turns.chunks = vec!["src (1/3)".into(), "src (2/3)".into(), "src (3/3)".into()];
    let second = audit_prompt(&turns, &[], 1, 3, None);
    assert!(
        second.starts_with("[hi:review] Defect audit, chunk 2/3: src (2/3)"),
        "{second}"
    );
    assert!(
        second.contains("Code under audit: chunk 2/3 of the 45 files: the 16 listed below, uncommitted per `git status`"),
        "{second}"
    );
    assert!(second.contains("- src/f16.rs\n"), "{second}");
    assert!(!second.contains("- src/f0.rs"), "{second}");
    assert!(
        second.contains(
            "The other chunks are audited in their own turns, so report nothing outside this one.\n"
        ),
        "{second}"
    );

    let last = ReviewInputs {
        git: Some(git_scope(
            GitSource::LastCommit {
                short_hash: "a1b2c3d".into(),
                subject: "fix nick parsing".into(),
            },
            vec!["src/server.rs".into()],
        )),
        ..ReviewInputs::default()
    };
    let audit = audit_prompt(&last, &[], 0, 3, None);
    assert!(
        audit.contains(
            "Code under audit: the 1 file(s) listed below, changed by the last commit a1b2c3d \"fix nick parsing\"; the tree is clean"
        ),
        "{audit}"
    );
    assert!(audit.contains("`git show HEAD -- <path>`"), "{audit}");
    assert!(
        !audit.contains("Workspace root:"),
        "no root recorded, no line"
    );
    let reaudit = reaudit_prompt(&last, &[], &["src/server.rs".into()], &[], &[], 1, 3);
    assert!(
        reaudit
            .contains("Code under audit: the 1 file(s) listed below, changed by the last commit"),
        "the re-audit keeps the boundary: {reaudit}"
    );
    assert!(reaudit.contains(REVIEW_DEFECTS_FORMAT_BLOCK), "{reaudit}");
    assert!(!reaudit.contains("refresh the coverage rows"), "{reaudit}");

    let chunked = ReviewInputs {
        chunks: vec![
            "crates/hi-a".into(),
            "docs, src/., lib (2/5)".into(),
            ".".into(),
        ],
        ..ReviewInputs::default()
    };
    let first = audit_prompt(&chunked, &[], 0, 3, None);
    assert!(
        first.starts_with("[hi:review] Defect audit, chunk 1/3: crates/hi-a (up to 3 fix passes"),
        "{first}"
    );
    assert_eq!(
        transcript_label(&first).as_deref(),
        Some("/review · defect audit, chunk 1/3: crates/hi-a")
    );
    assert!(
        first.contains(
            "Scope: chunk 1/3 of the workspace: `crates/hi-a/`. Limit the code audit to it (its files are listed below); the other chunks are audited in their own turns, so report nothing outside it.\n"
        ),
        "{first}"
    );
    assert!(
        !first.contains("Coverage rows judge this chunk only"),
        "no spec, no rows"
    );
    let packed = audit_prompt(&chunked, &[], 1, 3, None);
    assert!(
        packed.contains("chunk 2/3: docs, src (own files), lib (2/5)"),
        "{packed}"
    );
    assert!(
        packed.contains(
            "Scope: chunk 2/3 of the workspace: `docs/`, the files directly in `src/` (not its subdirectories) and `lib/` part 2/5."
        ),
        "{packed}"
    );
    let last_chunk = audit_prompt(&chunked, &[], 2, 3, None);
    assert!(
        last_chunk.contains("chunk 3/3: top-level files"),
        "{last_chunk}"
    );
    assert!(
        last_chunk.contains("the files directly in the workspace root (not its subdirectories)"),
        "{last_chunk}"
    );
    let reaudit = reaudit_prompt(&chunked, &[], &["src/main.rs".into()], &[], &[], 1, 3);
    assert!(
        !reaudit.contains("Scope: chunk"),
        "a re-audit is scoped by the fix pass: {reaudit}"
    );

    let files = vec!["src/a.rs".to_string(), "src/b.rs".to_string()];
    let section = scope_files_section(&files);
    assert_eq!(
        section,
        "\nFiles in scope (2; the report counts which of them were read):\n- src/a.rs\n- src/b.rs\n"
    );
    assert_eq!(scope_files_section(&[]), "");
    let many: Vec<String> = (0..50).map(|i| format!("f{i}.rs")).collect();
    assert!(scope_files_section(&many).ends_with("- … and 10 more (`list`/`glob` show them)\n"));

    let chunked_spec = ReviewInputs {
        files: vec![ReviewInput {
            path: "plan.md".into(),
            kind: InputKind::Plan,
        }],
        chunks: vec!["src".into(), "tests".into()],
        ..ReviewInputs::default()
    };
    let with_spec = audit_prompt(&chunked_spec, &[], 1, 3, None);
    assert!(
        with_spec.starts_with("[hi:review] Spec-coverage audit, chunk 2/2: tests"),
        "{with_spec}"
    );
    assert!(
        with_spec.contains("Coverage rows judge this chunk only"),
        "{with_spec}"
    );
    assert!(with_spec.contains(REVIEW_FORMAT_BLOCK), "{with_spec}");

    let reask = format_reask_prompt(&[], &[], true);
    assert!(
        reask.contains("no `coverage:` rows (there are no plan/spec items)"),
        "{reask}"
    );
    assert!(reask.ends_with(REVIEW_DEFECTS_FORMAT_BLOCK), "{reask}");
    assert_eq!(
        transcript_label(&reask).as_deref(),
        Some("/review · format re-ask")
    );
}

/// The seed result rides in the audit and continuation prompts; the
/// continuation lists only the unread files (with their line counts) and
/// the findings so far.
#[test]
fn seed_and_continuation_prompts() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(dir.path().join("src")).unwrap();
    std::fs::write(dir.path().join("src/a.rs"), "").unwrap();
    let failed = VerifySeed::from_output(
        dir.path(),
        "cargo check",
        false,
        "error[E0308]: mismatched types\n --> src/a.rs:4:5\n",
        "audit",
    );
    let mut scope = git_scope(
        GitSource::Uncommitted,
        vec!["src/a.rs".into(), "src/b.rs".into(), "src/c.rs".into()],
    );
    scope.numstat = vec![GitNumstat {
        path: "src/b.rs".into(),
        added: Some(4),
        removed: Some(1),
        untracked: false,
    }];
    let inputs = ReviewInputs {
        git: Some(scope),
        root: Some("/work/app".into()),
        ..ReviewInputs::default()
    };
    let audit = audit_prompt(&inputs, &[], 0, 3, Some(&failed));
    assert!(
        audit.contains("Verify before this turn: `cargo check` FAILED. Output tail:\n```\nerror[E0308]: mismatched types\n --> src/a.rs:4:5\n```"),
        "{audit}"
    );
    assert!(
        audit.contains("The output points at src/a.rs:4."),
        "{audit}"
    );
    let seed_at = audit.find("Verify before this turn").unwrap();
    let procedure_at = audit.find("\nProcedure:\n").unwrap();
    assert!(
        seed_at < procedure_at,
        "the seed comes before the procedure"
    );

    let passed = VerifySeed::from_output(dir.path(), "cargo check", true, "", "re-audit 1");
    let prior = Finding {
        severity: Severity::P2,
        title: "Log unknown commands".into(),
        location: Some("src/a.rs:9".into()),
        verified: true,
        feature_gap: false,
        from_verify: false,
    };
    let unread = vec!["src/b.rs".to_string(), "src/c.rs".to_string()];
    let continued = continuation_prompt(&inputs, 0, &unread, &[prior], 1, 2, Some(&passed));
    assert!(
        continued.starts_with("[hi:review] Defect audit, continued (1/2). The last turn left 2 of the files under audit unread, so its verdict is partial."),
        "{continued}"
    );
    assert_eq!(
        transcript_label(&continued).as_deref(),
        Some("/review · defect audit, continued")
    );
    assert!(
        continued.contains(
            "Workspace root: /work/app (paths below are relative to it; cite paths the same way).\n"
        ),
        "{continued}"
    );
    assert!(
        continued.contains("Unread files (read every one):\n- src/b.rs (+4 −1)\n- src/c.rs\n"),
        "{continued}"
    );
    assert!(!continued.contains("- src/a.rs\n"), "{continued}");
    assert!(
        continued.contains("Findings already recorded from the files read so far (kept; do not repeat them unless one of the files above adds to it):\n- [P2] Log unknown commands — src/a.rs:9\n"),
        "{continued}"
    );
    assert!(
        continued.contains("Verify before this turn: `cargo check` passed. A defect you report must be something it does not cover."),
        "{continued}"
    );
    assert!(
        continued.contains(REVIEW_DEFECTS_FORMAT_BLOCK),
        "{continued}"
    );
    assert!(!continued.contains("Scope: chunk"), "not chunked");

    let chunked = ReviewInputs {
        chunks: vec!["src".into(), "docs".into()],
        ..ReviewInputs::default()
    };
    let in_chunk = continuation_prompt(&chunked, 1, &unread, &[], 2, 2, None);
    assert!(in_chunk.contains("Scope: chunk 2/2: docs.\n"), "{in_chunk}");
    assert!(
        !in_chunk.contains("Findings already recorded"),
        "{in_chunk}"
    );
    assert!(!in_chunk.contains("Verify before this turn"), "{in_chunk}");
    let many: Vec<String> = (0..50).map(|i| format!("src/f{i}.rs")).collect();
    let long = continuation_prompt(&chunked, 0, &many, &[], 1, 2, None);
    assert!(
        long.contains("- … and 10 more; the next continuation lists them\n"),
        "{long}"
    );
}
