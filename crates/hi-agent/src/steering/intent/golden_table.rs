use super::*;
use crate::steering::types::ReviewIntent;

/// Frozen prompt → intent pairs. Prefer `/macro` expansions and phrases already
/// proven in `tests/steering.rs` so this table tracks real classifier gates.
#[test]
fn read_only_intent_golden_table() {
    let cases: &[(&str, Option<ReviewIntent>)] = &[
        ("status", None),
        ("fix the unsafe unwraps", None),
        ("review codebase and discuss status and state", None),
        (
            "review this code for auth leaks but do not edit",
            Some(ReviewIntent::Security),
        ),
        (
            "Review this codebase for issues related to ipop/coder-balanced API routing or latency. Use at most 4 file inspections. Do not modify files. Return concise findings only.",
            Some(ReviewIntent::Review),
        ),
    ];
    for (prompt, want) in cases {
        assert_eq!(
            classify_read_only_intent(prompt),
            *want,
            "read-only classify failed for {prompt:?}"
        );
    }
}

#[test]
fn implicit_review_classifier_aligns_bare_review_with_read_only_contract() {
    assert_eq!(
        implicit_read_only_review_intent("review codebase", true),
        Some(ReviewIntent::Review)
    );
    assert_eq!(
        implicit_read_only_review_intent("review codebase and discuss status", true),
        Some(ReviewIntent::Review)
    );
    assert_eq!(
        implicit_read_only_review_intent("review for any major issues", true),
        Some(ReviewIntent::Review)
    );
    assert_eq!(
        implicit_read_only_review_intent("review codebase and fix the bug", false),
        None
    );
    assert_eq!(
        implicit_read_only_review_intent(
            "review we have some kinda issues : models endpoint returned 403 --- and it seems to stayon the screen",
            true,
        ),
        None
    );
    assert_eq!(
        implicit_read_only_review_intent("audit the auth module", true),
        None
    );
    assert_eq!(
        implicit_read_only_review_intent("review the code", true),
        None
    );
}

#[test]
fn bare_codebase_review_is_distinguished_from_deep_or_parallel_review() {
    assert!(is_bare_codebase_review("review codebase"));
    assert!(is_bare_codebase_review(
        "review the repository for major issues"
    ));
    assert!(!is_bare_codebase_review(
        "review codebase using parallel independent investigations"
    ));
    assert!(!is_bare_codebase_review(
        "review crates/hi-agent/src/lib.rs and trace the full request lifecycle"
    ));
}

/// Corpus harness against real-world issue reports (every SWE-bench-style
/// problem statement is an implementation request by construction, so any
/// read-only classification is a false positive). Reporting-only:
/// `HI_INTENT_CORPUS=<prompts.jsonl> cargo test -p hi-agent --lib \
///  intent_corpus -- --ignored --nocapture`
#[test]
#[ignore = "set HI_INTENT_CORPUS to a jsonl of {\"prompt\": …} lines"]
fn intent_corpus_read_only_false_positives() {
    let Some(path) = std::env::var_os("HI_INTENT_CORPUS") else {
        return;
    };
    let text = std::fs::read_to_string(path).expect("corpus file");
    let mut total = 0usize;
    let mut false_positives = Vec::new();
    for line in text.lines() {
        let Ok(value) = serde_json::from_str::<serde_json::Value>(line) else {
            continue;
        };
        let Some(prompt) = value.get("prompt").and_then(|p| p.as_str()) else {
            continue;
        };
        total += 1;
        if let Some(intent) = classify_read_only_intent(prompt) {
            let head: String = prompt.chars().take(120).collect();
            false_positives.push(format!("{intent:?}: {head}"));
        }
    }
    println!(
        "intent corpus: {}/{total} implementation prompts misread as read-only",
        false_positives.len()
    );
    for fp in false_positives.iter().take(10) {
        println!("  FP {fp}");
    }
}

#[test]
fn read_only_prompt_avoids_internal_tool_availability_wording() {
    let prompt = read_only_turn_prompt(
        "review this code for auth leaks but do not edit",
        ReviewIntent::Security,
    );

    assert!(prompt.contains("currently advertised read-only inspection tools"));
    assert!(prompt.contains("advertised read-only inspection tools"));
    assert!(prompt.contains("invent tool names or handles remembered from earlier turns"));
    assert!(
        !prompt.contains("shell execution (`bash`)")
            && !prompt.contains("unavailable for this review")
    );
    assert!(!prompt.contains("run mutating shell commands"));
}

#[test]
fn bounded_exact_review_prompt_prefers_a_best_effort_targeted_pass() {
    let prompt = read_only_turn_prompt(
        "Review only crates/hi-ai/src/openai/request.rs and crates/hi-ai/src/openai/stream.rs for one concrete bug. Use targeted read or grep within those two files only and do not edit files.",
        ReviewIntent::Review,
    );

    assert!(prompt.contains("bounded exact-file review"));
    assert!(prompt.contains("one batched `read` call"));
    assert!(prompt.contains("Do not reread the same content"));
    assert!(prompt.contains("best-effort finding"));
    assert!(prompt.contains("Do not keep paging through a large file"));
}

#[test]
fn broad_review_prompt_keeps_general_inspection_guidance() {
    let prompt =
        read_only_turn_prompt("Review the codebase for major issues", ReviewIntent::Review);

    assert!(!prompt.contains("bounded exact-file review"));
    assert!(prompt.contains("bounded static review"));
    assert!(prompt.contains("Do not repeatedly relist the workspace"));
}

#[test]
fn git_log_piped_to_head_is_search_evidence_not_a_file_read() {
    let git_log = serde_json::json!({
        "command": "git log --oneline -8 2>/dev/null | head -20"
    })
    .to_string();
    assert_eq!(
        evidence_kind_for_bash(&git_log),
        Some(EvidenceKind::TargetedSearch)
    );
    let dump = serde_json::json!({ "command": "sed -n '1,20p' src/web.rs" }).to_string();
    assert_eq!(evidence_kind_for_bash(&dump), Some(EvidenceKind::FileRead));
}

#[test]
fn implementation_intent_golden_table() {
    let build_macro = "Build a small helper.

Implementation requirements
Inspect the workspace before editing.
Expected to edit files and run verification.";
    // Expanded /build macro shape (see expanded_build_macro_request).
    let expanded =
        "build foo implementation requirements inspect the workspace before you edit files";
    assert!(
        classify_implementation_intent(expanded).is_some()
            || classify_implementation_intent(build_macro).is_some()
            || classify_implementation_intent(
                "Implementation task: expected to edit files and run the verification command"
            )
            .is_some(),
        "at least one known implementation shape should classify"
    );
    assert!(
        classify_implementation_intent("keep building the feature").is_some(),
        "natural continuation should classify"
    );
    let chess = classify_implementation_intent(
            "lets build a chess TUI game. lets make a TUI crate that uses similar style as grok and also fully enable mouse usage",
        )
        .expect("ordinary greenfield TUI request should classify");
    assert!(chess.tui, "the chess request should receive TUI guidance");
    let social_app = classify_implementation_intent(
        "we want to build a twitter style app. we also want to seed it with agents.",
    )
    .expect("first-person greenfield app request should classify");
    assert!(!social_app.tui);
    for prompt in [
        "what is the status?",
        "review only, do not change code",
        "discuss the architecture",
        "status",
        "does cargo build?",
        "Build plans the work",
        "review grok-build",
    ] {
        assert_eq!(
            classify_implementation_intent(prompt),
            None,
            "expected no implementation intent for {prompt:?}"
        );
    }
}
