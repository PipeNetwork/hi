//! Prompt text for the `/review` turns: the audit, fix, re-audit, and
//! format re-ask prompts carry the prefix, the inputs, and the verdict
//! contract; a narrowed (git-scoped, chunked, or defects-only) audit
//! names its scope and drops the coverage row.

use super::*;
use crate::review::{ChecklistItem, Finding, InputKind, ReviewInput, ReviewInputs, Severity};
use crate::review_scope::{GitScope, GitSource};

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
    let audit = audit_prompt(&inputs, &checklist, 0, 3);
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
    let audit_numbered = audit_prompt(&inputs, &numbered, 0, 3);
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
    let readme_audit = audit_prompt(&readme_inputs, &[], 0, 3);
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
        &["src/server.rs".into()],
        std::slice::from_ref(&finding),
        1,
        3,
    );
    assert!(reaudit.contains("- src/server.rs\n"));
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
    let bare = reaudit_prompt(&inputs, &[], &[], &[], 2, 3);
    assert!(!bare.contains("Plan/spec items"));
    assert!(bare.contains("The fix pass changed no files."));
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
    let git = ReviewInputs {
        git: Some(GitScope {
            source: GitSource::Uncommitted,
            files: (0..45).map(|i| format!("src/f{i}.rs")).collect(),
        }),
        ..ReviewInputs::default()
    };
    assert!(git.defects_only());
    let audit = audit_prompt(&git, &[], 0, 3);
    assert!(
        audit.starts_with("[hi:review] Defect audit (up to 3 fix passes"),
        "{audit}"
    );
    assert_eq!(
        transcript_label(&audit).as_deref(),
        Some("/review · defect audit")
    );
    assert!(
        audit.contains("Inputs: none found (no plan.md or spec.md). This is a defects-only audit"),
        "{audit}"
    );
    assert!(
        audit.contains("Code under audit: the 45 uncommitted file(s) per `git status`"),
        "{audit}"
    );
    assert!(audit.contains("`git diff HEAD`"), "{audit}");
    assert!(audit.contains("- src/f0.rs\n"), "{audit}");
    assert!(
        audit.contains("- src/f39.rs\n- … and 5 more (git lists them)\n"),
        "{audit}"
    );
    assert!(!audit.contains("- src/f40.rs\n"), "{audit}");
    assert!(audit.contains(REVIEW_DEFECTS_FORMAT_BLOCK), "{audit}");
    assert!(
        !audit.contains("coverage: implemented|partial|missing"),
        "{audit}"
    );
    assert!(
        audit.contains("`verdict: COMPLETE` when no P0/P1 finding is listed"),
        "{audit}"
    );
    assert!(!audit.contains("Scope: chunk"), "not chunked");

    let last = ReviewInputs {
        git: Some(GitScope {
            source: GitSource::LastCommit {
                short_hash: "a1b2c3d".into(),
                subject: "fix nick parsing".into(),
            },
            files: vec!["src/server.rs".into()],
        }),
        ..ReviewInputs::default()
    };
    let audit = audit_prompt(&last, &[], 0, 3);
    assert!(
        audit.contains(
            "Code under audit: the 1 file(s) changed by the last commit a1b2c3d \"fix nick parsing\"; the tree is clean"
        ),
        "{audit}"
    );
    assert!(audit.contains("Run `git show HEAD`"), "{audit}");
    let reaudit = reaudit_prompt(&last, &[], &["src/server.rs".into()], &[], 1, 3);
    assert!(
        reaudit.contains("Code under audit: the 1 file(s) changed by the last commit"),
        "the re-audit keeps the boundary: {reaudit}"
    );
    assert!(reaudit.contains(REVIEW_DEFECTS_FORMAT_BLOCK), "{reaudit}");
    assert!(!reaudit.contains("refresh the coverage rows"), "{reaudit}");

    let chunked = ReviewInputs {
        chunks: vec!["crates/hi-a".into(), "docs".into(), ".".into()],
        ..ReviewInputs::default()
    };
    let first = audit_prompt(&chunked, &[], 0, 3);
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
            "Scope: chunk 1/3 of the workspace: `crates/hi-a/`. Limit the code audit to it; the other chunks are audited in their own turns, so report nothing outside it.\n"
        ),
        "{first}"
    );
    assert!(
        !first.contains("Coverage rows judge this chunk only"),
        "no spec, no rows"
    );
    let last_chunk = audit_prompt(&chunked, &[], 2, 3);
    assert!(
        last_chunk.contains("chunk 3/3: top-level files"),
        "{last_chunk}"
    );
    assert!(
        last_chunk.contains("the files directly in the workspace root (not its subdirectories)"),
        "{last_chunk}"
    );
    let reaudit = reaudit_prompt(&chunked, &[], &["src/main.rs".into()], &[], 1, 3);
    assert!(
        !reaudit.contains("Scope: chunk"),
        "a re-audit is scoped by the fix pass: {reaudit}"
    );

    let chunked_spec = ReviewInputs {
        files: vec![ReviewInput {
            path: "plan.md".into(),
            kind: InputKind::Plan,
        }],
        chunks: vec!["src".into(), "tests".into()],
        ..ReviewInputs::default()
    };
    let with_spec = audit_prompt(&chunked_spec, &[], 1, 3);
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
