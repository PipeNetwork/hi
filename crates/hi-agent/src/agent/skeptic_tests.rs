use super::*;

#[test]
fn large_diff_prompt_targets_multi_file_holes() {
    assert!(
        LARGE_DIFF_REVIEW_PROMPT.contains("call sites"),
        "should mention missed call sites"
    );
    assert!(
        LARGE_DIFF_REVIEW_PROMPT.contains("APPROVE") && LARGE_DIFF_REVIEW_PROMPT.contains("OBJECT"),
        "must keep the same verdict protocol"
    );
    assert!(
        LARGE_DIFF_REVIEW_PROMPT.contains("LARGE"),
        "should mark large-diff context"
    );
    let gated = crate::skills::gated_review_system_prompt(LARGE_DIFF_REVIEW_PROMPT, false);
    assert!(
        gated.contains("introduced by this change"),
        "completion review must include the code-review Gate: {gated}"
    );
    assert!(
        gated.contains("Line 1 remains exactly APPROVE or OBJECT."),
        "{gated}"
    );
    let skeptic = crate::skills::gated_review_system_prompt(SKEPTIC_PROMPT, true);
    assert!(skeptic.contains("APPROVE, OBJECT, or ESCALATE"));
    assert!(skeptic.contains("introduced by this change"));
}

#[test]
fn approve_variants() {
    assert_eq!(parse_verdict("APPROVE"), SkepticVerdict::Approve);
    assert_eq!(
        parse_verdict("  approve — looks correct\n"),
        SkepticVerdict::Approve
    );
    assert_eq!(parse_verdict("**APPROVE**"), SkepticVerdict::Approve);
    assert!(matches!(
        parse_verdict("   \n\n"),
        SkepticVerdict::Unavailable(_)
    ));
    assert!(matches!(
        parse_verdict("hmm, not sure"),
        SkepticVerdict::Unavailable(_)
    ));
}

#[test]
fn review_diff_context_clips_huge_objective_and_sub_goal() {
    let ctx = review_diff_context(&"O".repeat(20_000), &"S".repeat(20_000), "diff-here");
    assert!(
        ctx.chars().count() < 20_000,
        "side-call context must stay bounded: {}",
        ctx.chars().count()
    );
    assert!(ctx.contains("diff-here"), "{ctx}");
    assert!(
        !ctx.contains(&"O".repeat(MAX_SKEPTIC_SIDE_CHARS + 1)),
        "{ctx}"
    );
    assert!(ctx.contains('…'), "{ctx}");
}

#[test]
fn approve_after_preamble_like_xai_same_model_review() {
    // Session models on xAI/OpenAI often narrate before the keyword.
    assert_eq!(
        parse_verdict("I reviewed the diff and verification evidence.\n\nAPPROVE\n"),
        SkepticVerdict::Approve
    );
    assert_eq!(
        parse_verdict("**Verdict:** APPROVE\nLooks good overall."),
        SkepticVerdict::Approve
    );
    assert_eq!(
        parse_verdict("Summary: the change meets the contract. I APPROVE."),
        SkepticVerdict::Approve
    );
}

#[test]
fn object_after_preamble() {
    assert_eq!(
        parse_verdict("Analysis follows.\nOBJECT\n- missing error path in parser.rs\n"),
        SkepticVerdict::Object(vec!["missing error path in parser.rs".to_string()])
    );
}

#[test]
fn escalate_variants() {
    let v = parse_verdict("ESCALATE\n- the sub-goal contradicts the frozen plan\n");
    assert_eq!(
        v,
        SkepticVerdict::Escalate(vec!["the sub-goal contradicts the frozen plan".to_string()])
    );
    assert_eq!(
        parse_verdict("**Escalate**: needs a user decision on the schema"),
        SkepticVerdict::Escalate(vec!["needs a user decision on the schema".to_string()])
    );
    // An escalation without a reason is unusable — Unavailable (caller policy).
    assert!(matches!(
        parse_verdict("ESCALATE"),
        SkepticVerdict::Unavailable(_)
    ));
}

#[test]
fn negated_approve_does_not_false_pass() {
    assert!(matches!(
        parse_verdict("I do not approve this change"),
        SkepticVerdict::Unavailable(_)
    ));
    assert!(matches!(
        parse_verdict("cannot approve until error handling lands"),
        SkepticVerdict::Unavailable(_)
    ));
    assert!(matches!(
        parse_verdict("I don't approve.\nThe parser still drops empty input."),
        SkepticVerdict::Unavailable(_)
    ));
    assert!(matches!(
        parse_verdict("never approve a stub stand-in"),
        SkepticVerdict::Unavailable(_)
    ));
    // A real OBJECT after a negated-approve preamble still objects.
    assert_eq!(
        parse_verdict("I cannot approve this as-is.\nOBJECT\n- missing error path in parser.rs\n"),
        SkepticVerdict::Object(vec!["missing error path in parser.rs".to_string()])
    );
    // Positive approve still works when not negated.
    assert_eq!(
        parse_verdict("I approve this change."),
        SkepticVerdict::Approve
    );
}

#[test]
fn object_with_listed_objections() {
    let v = parse_verdict("OBJECT\n- the loop is off by one\n- no test for the empty case\n");
    assert_eq!(
        v,
        SkepticVerdict::Object(vec![
            "the loop is off by one".to_string(),
            "no test for the empty case".to_string(),
        ])
    );
}

#[test]
fn object_inline_objection() {
    // Objection on the verdict line after a separator.
    assert_eq!(
        parse_verdict("OBJECT: the sub-goal isn't actually satisfied"),
        SkepticVerdict::Object(vec!["the sub-goal isn't actually satisfied".to_string()])
    );
    // Markdown-wrapped keyword + a following bullet line.
    assert_eq!(
        parse_verdict("**OBJECT**\n* missing error handling"),
        SkepticVerdict::Object(vec!["missing error handling".to_string()])
    );
}

#[test]
fn object_without_anything_actionable_is_unavailable() {
    assert!(matches!(
        parse_verdict("OBJECT"),
        SkepticVerdict::Unavailable(_)
    ));
    assert!(matches!(
        parse_verdict("OBJECT\n\n"),
        SkepticVerdict::Unavailable(_)
    ));
}
