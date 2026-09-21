use super::*;
use crate::review_citations::{citation_path, citation_paths};
use std::fs;

fn write(root: &Path, rel: &str, body: &str) {
    let path = root.join(rel);
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::write(path, body).unwrap();
}

const VERDICT: &str = "Coverage table for the user...\n\n<review>\n\
verdict: INCOMPLETE\n\
coverage: implemented | Welcome banner on connect | src/server.rs:12\n\
coverage: partial | KICK command | src/commands.rs:40\n\
coverage: missing | /topic command | -\n\
finding: P0 | Reject empty nicknames | src/server.rs:88\n\
finding: P2 | Log unknown commands | src/commands.rs:9\n\
finding: P0 | Reject empty nicknames | src/server.rs:90\n\
finding: bogus | not a severity | x\n\
residual: history persistence untested\n\
</review>\n";

#[test]
fn review_args_split_action_and_paths() {
    assert_eq!(
        ReviewArgs::parse(""),
        ReviewArgs {
            action: ReviewAction::Run,
            paths: vec![],
            all: false,
        }
    );
    assert_eq!(
        ReviewArgs::parse("audit docs/spec.md plan.md"),
        ReviewArgs {
            action: ReviewAction::Audit,
            paths: vec!["docs/spec.md".into(), "plan.md".into()],
            all: false,
        }
    );
    assert_eq!(ReviewArgs::parse("STATUS").action, ReviewAction::Status);
    assert_eq!(ReviewArgs::parse("stop").action, ReviewAction::Stop);
    assert_eq!(
        ReviewArgs::parse("src/ docs/spec.md"),
        ReviewArgs {
            action: ReviewAction::Run,
            paths: vec!["src/".into(), "docs/spec.md".into()],
            all: false,
        }
    );
    // `all` is a keyword anywhere after the action, any case; `./all` is a path.
    assert_eq!(
        ReviewArgs::parse("audit all"),
        ReviewArgs {
            action: ReviewAction::Audit,
            paths: vec![],
            all: true,
        }
    );
    assert_eq!(
        ReviewArgs::parse("docs/spec.md ALL ./all"),
        ReviewArgs {
            action: ReviewAction::Run,
            paths: vec!["docs/spec.md".into(), "./all".into()],
            all: true,
        }
    );
    let mut paths = vec!["all".to_string()];
    assert!(take_all(&mut paths));
    assert!(paths.is_empty());
    let mut paths = vec!["src".to_string()];
    assert!(!take_all(&mut paths));
    assert_eq!(paths, vec!["src".to_string()]);
}

#[test]
fn discovery_prefers_explicit_then_root_then_docs_then_hi() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    write(root, ".hi/plan.md", "- [ ] hidden");
    write(root, "docs/SPEC.md", "spec");
    let found = discover_inputs(root, &[]);
    assert_eq!(
        found.files,
        vec![
            ReviewInput {
                path: ".hi/plan.md".into(),
                kind: InputKind::Plan
            },
            ReviewInput {
                path: "docs/SPEC.md".into(),
                kind: InputKind::Spec
            },
        ]
    );
    assert_eq!(found.notice, None);

    write(root, "Plan.md", "- [ ] root wins");
    let found = discover_inputs(root, &[]);
    assert_eq!(found.files[0].path, "Plan.md");
    assert_eq!(found.summary(), "Plan.md + docs/SPEC.md");

    write(root, "notes/other.md", "x");
    fs::create_dir_all(root.join("src")).unwrap();
    let explicit = discover_inputs(root, &["notes/other.md".into(), "src/".into()]);
    assert_eq!(explicit.files.len(), 1);
    assert_eq!(explicit.files[0].kind, InputKind::Other);
    assert_eq!(explicit.scope, vec!["src".to_string()]);

    let missing = discover_inputs(root, &["nope.md".into()]);
    assert_eq!(missing.missing, vec!["nope.md".to_string()]);
    assert!(missing.notice.unwrap().contains("no such file: nope.md"));
}

#[test]
fn discovery_falls_back_to_readme_then_defects_only() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    let none = discover_inputs(root, &[]);
    assert!(none.files.is_empty());
    assert_eq!(none.summary(), "no plan/spec (defects only)");
    assert!(none.notice.unwrap().contains("defects only"));

    write(root, "readme.md", "# app");
    let readme = discover_inputs(root, &[]);
    assert_eq!(readme.files[0].kind, InputKind::Readme);
    assert!(
        readme
            .notice
            .as_deref()
            .unwrap()
            .contains("no plan.md or spec.md found; auditing against readme.md")
    );
}

#[test]
fn checklist_parses_checked_unchecked_numbered_and_skips_fences() {
    let md = "# Plan\n\n- [ ] Welcome banner\n* [x] KICK command\n+ [X] Nick change\n\
```\n- [ ] not a row\n```\n1. Topic command\n2) History\n- plain bullet\n- [ ]   \n";
    let items = parse_checklist(md);
    assert_eq!(
        items,
        vec![
            ChecklistItem {
                text: "Welcome banner".into(),
                checked: Some(false)
            },
            ChecklistItem {
                text: "KICK command".into(),
                checked: Some(true)
            },
            ChecklistItem {
                text: "Nick change".into(),
                checked: Some(true)
            },
            ChecklistItem {
                text: "Topic command".into(),
                checked: None
            },
            ChecklistItem {
                text: "History".into(),
                checked: None
            },
        ]
    );
    assert!(parse_checklist("no rows here").is_empty());
}

#[test]
fn verdict_parses_rows_dedupes_findings_and_skips_malformed() {
    let verdict = ReviewVerdict::parse(VERDICT).expect("block");
    assert!(!verdict.complete);
    assert_eq!(verdict.coverage.len(), 3);
    assert_eq!(verdict.coverage[2].state, CoverageState::Missing);
    assert_eq!(verdict.coverage[2].evidence, None);
    assert_eq!(
        verdict.coverage[0].evidence.as_deref(),
        Some("src/server.rs:12")
    );
    assert_eq!(verdict.findings.len(), 2, "{:?}", verdict.findings);
    assert_eq!(verdict.findings[0].severity, Severity::P0);
    assert_eq!(verdict.findings[1].severity, Severity::P2);
    assert_eq!(
        verdict.residual.as_deref(),
        Some("history persistence untested")
    );
    assert!(verdict.has_blocking());
    assert_eq!(verdict.blocking_findings().len(), 1);
    assert!(!verdict.coverage_complete());
}

#[test]
fn verdict_requires_block_and_a_verdict_or_coverage_row_and_uses_last_block() {
    assert_eq!(ReviewVerdict::parse("no block at all"), None);
    assert_eq!(
        ReviewVerdict::parse("<review>\nfinding: P1 | x | a.rs\n</review>"),
        None,
        "findings alone give nothing to derive a verdict from"
    );
    let rows_only = ReviewVerdict::parse("<review>\ncoverage: implemented | x | -\n</review>")
        .expect("coverage rows carry the verdict even without the word");
    assert!(rows_only.complete);
    assert_eq!(rows_only.stated_complete, None);
    let two = "<review>\nverdict: INCOMPLETE\n</review>\nlater\n<review>\nverdict: complete\nfinding: none\n</review>";
    let verdict = ReviewVerdict::parse(two).unwrap();
    assert!(verdict.complete);
    assert!(verdict.findings.is_empty());
    assert!(verdict.coverage_complete());
    let bullets = "<review>\n- verdict: COMPLETE (all items)\n- coverage: implemented | a | `src/a.rs:1`\n</review>";
    let verdict = ReviewVerdict::parse(bullets).unwrap();
    assert!(verdict.complete);
    assert_eq!(verdict.coverage[0].evidence.as_deref(), Some("src/a.rs:1"));
}

/// Live run: `verdict: INCOMPLETE` over eight `implemented` rows, `finding:
/// none`, and a residual saying everything was implemented; the summary
/// then read "8/8 implemented" under a "coverage (incomplete)" header and
/// headless would have exited 3. The rows are the contract.
#[test]
fn verdict_is_derived_from_the_coverage_rows() {
    let contradicted = ReviewVerdict::parse(
        "<review>\nverdict: INCOMPLETE\ncoverage: implemented | a | x.rs:1\ncoverage: implemented | b | -\nfinding: none\n</review>",
    )
    .unwrap();
    assert!(contradicted.complete);
    assert_eq!(contradicted.stated_complete, Some(false));
    assert_eq!(
        contradicted.verdict_note().as_deref(),
        Some(
            "the model wrote INCOMPLETE with every coverage row implemented; verdict taken from the rows"
        )
    );

    let optimistic = ReviewVerdict::parse(
        "<review>\nverdict: COMPLETE\ncoverage: implemented | a | x.rs:1\ncoverage: missing | b | -\n</review>",
    )
    .unwrap();
    assert!(!optimistic.complete, "a missing row overrides COMPLETE");
    assert_eq!(
        optimistic.verdict_note().as_deref(),
        Some(
            "the model wrote COMPLETE with 1 missing/partial coverage row(s); verdict taken from the rows"
        )
    );

    let dropped = ReviewVerdict::parse(
        "<review>\nverdict: INCOMPLETE\ncoverage: implemented | a | x.rs:1\ncoverage: <state> | b | -\n</review>",
    )
    .unwrap();
    assert!(
        !dropped.complete,
        "a row that failed to parse may be the gap: the model's word stands"
    );
    assert_eq!(dropped.unparsed_coverage_rows, 1);
    assert_eq!(
        dropped.verdict_note().as_deref(),
        Some("1 coverage row(s) could not be parsed and were dropped")
    );

    let word_only =
        ReviewVerdict::parse("<review>\nverdict: INCOMPLETE\nfinding: none\n</review>").unwrap();
    assert!(!word_only.complete);
    assert_eq!(word_only.verdict_note(), None);
    let consistent = ReviewVerdict::parse(VERDICT).unwrap();
    assert_eq!(consistent.verdict_note(), None);
}

#[test]
fn citations_mark_missing_paths_unverified() {
    let dir = tempfile::tempdir().unwrap();
    write(dir.path(), "src/server.rs", "fn main() {}");
    let mut verdict = ReviewVerdict::parse(VERDICT).unwrap();
    let unverified = check_citations(dir.path(), &mut verdict);
    assert!(verdict.findings[0].verified, "src/server.rs exists");
    assert!(!verdict.findings[1].verified, "src/commands.rs does not");
    assert_eq!(
        unverified,
        vec!["src/commands.rs".to_string()],
        "one entry per missing path, however often it is cited"
    );
    assert_eq!(
        citation_path(Some("`crates/a/src/lib.rs:10:4`")).as_deref(),
        Some("crates/a/src/lib.rs")
    );
    assert_eq!(
        citation_path(Some("./src/x.rs:1-9")).as_deref(),
        Some("src/x.rs")
    );
    assert_eq!(citation_path(Some(" - ")), None);
    assert_eq!(citation_path(None), None);
}

/// Live run: evidence cells like `a.rs, b.rs:380-499, c.rs:61` and
/// `lib.rs (CONST_NAME)` were each read as one nonexistent path, so five of
/// eight rows were "unverified" although every file existed.
#[test]
fn citation_cells_may_hold_several_paths_and_a_parenthetical() {
    assert_eq!(
        citation_paths(Some(
            "crates/hi-ai/src/openai/stream.rs, crates/hi-harness/src/turn.rs:380-499, crates/hi-provider-config/src/lib.rs:61"
        )),
        vec![
            "crates/hi-ai/src/openai/stream.rs",
            "crates/hi-harness/src/turn.rs",
            "crates/hi-provider-config/src/lib.rs"
        ]
    );
    assert_eq!(
        citation_paths(Some("crates/p/src/lib.rs (PIPE_DEEPSEEK_MODEL_ID)")),
        vec!["crates/p/src/lib.rs"]
    );
    assert_eq!(
        citation_paths(Some("src/x.rs:10,12; src/x.rs:40")),
        vec!["src/x.rs"],
        "a line list is not a path and repeats collapse"
    );
    assert_eq!(citation_paths(Some("Makefile:3")), vec!["Makefile"]);
    assert!(citation_paths(Some("-")).is_empty());
    assert!(citation_paths(None).is_empty());

    let dir = tempfile::tempdir().unwrap();
    write(dir.path(), "src/a.rs", "");
    write(dir.path(), "src/b.rs", "");
    let mut verdict = ReviewVerdict::parse(
        "<review>\nverdict: COMPLETE\ncoverage: implemented | both | src/a.rs:1, src/b.rs:2-3\n\
finding: P1 | Half real | src/a.rs:4, src/gone.rs:5\n</review>",
    )
    .unwrap();
    assert_eq!(
        check_citations(dir.path(), &mut verdict),
        vec!["src/gone.rs".to_string()]
    );
    assert!(!verdict.findings[0].verified);
}

#[test]
fn fingerprint_is_stable_and_order_independent() {
    let a = Finding {
        severity: Severity::P0,
        title: "Reject empty nicknames".into(),
        location: Some("src/server.rs:88".into()),
        verified: true,
        feature_gap: false,
    };
    let b = Finding {
        severity: Severity::P1,
        title: "Kick requires operator".into(),
        location: Some("src/commands.rs:40".into()),
        verified: true,
        feature_gap: false,
    };
    let mut moved = a.clone();
    moved.location = Some("src/server.rs:120".into());
    moved.title = "reject EMPTY nicknames!".into();
    assert_eq!(
        fingerprint(&[a.clone(), b.clone()]),
        fingerprint(&[b.clone(), a.clone()])
    );
    assert_eq!(
        fingerprint(&[a.clone(), b.clone()]),
        fingerprint(&[moved, b.clone()])
    );
    assert_ne!(fingerprint(std::slice::from_ref(&a)), fingerprint(&[a, b]));
    assert_eq!(fingerprint(&[]), fingerprint(&[]));
}

/// Live run: the audit filed the unchecked `/topic` plan row as
/// `finding: P1 | Implement TOPIC with broadcast and replay on join`, and the
/// fix pass built the feature. Findings that restate an unchecked row are
/// gaps: reported, kept out of the blocking set. A defect in a checked row
/// ("KICK removes the target" vs "Remove kicked target from fan-out") is not
/// matched against unchecked rows, so it stays a defect.
#[test]
fn findings_that_restate_unchecked_plan_rows_are_feature_gaps() {
    let unchecked = vec!["/topic: TOPIC command with broadcast and replay on join".to_string()];
    assert!(restates_item(
        &unchecked[0],
        "Implement TOPIC with broadcast and replay on join"
    ));
    assert!(restates_item(
        &unchecked[0],
        "Add the TOPIC command (broadcast, replay on join)"
    ));
    assert!(!restates_item(
        &unchecked[0],
        "Remove kicked target from channel fan-out"
    ));
    assert!(!restates_item(
        &unchecked[0],
        "Send replies on tx so Welcome/OK/ERROR reach the client"
    ));
    assert!(
        !restates_item(&unchecked[0], "Fix TOPIC"),
        "one shared word is not a restatement"
    );
    assert!(
        !restates_item("Tests", "Add tests for TOPIC"),
        "a one-word row never matches"
    );
    assert!(
        restates_item(
            "KICK removes the target from the channel fan-out",
            "Remove kicked target from channel fan-out"
        ),
        "stemming lines up removes/remove and kicked/kick"
    );

    let mut verdict = ReviewVerdict::parse(
        "<review>\nverdict: INCOMPLETE\n\
coverage: missing | TOPIC command with broadcast and replay on join | -\n\
finding: P0 | Send replies on tx so Welcome/OK/ERROR reach the client | src/main.rs:57\n\
finding: P1 | Remove kicked target from channel fan-out | src/main.rs:158\n\
finding: P1 | Implement TOPIC with broadcast and replay on join | src/main.rs:80\n\
finding: P3 | Add a TOPIC broadcast test | tests/integration.rs:1\n\
</review>",
    )
    .unwrap();
    assert_eq!(
        verdict.blocking_findings().len(),
        3,
        "unmarked: all P0/P1 count"
    );
    verdict.mark_feature_gaps(&unchecked);
    let gaps: Vec<&str> = verdict
        .findings
        .iter()
        .filter(|f| f.feature_gap)
        .map(|f| f.title.as_str())
        .collect();
    assert_eq!(
        gaps,
        vec!["Implement TOPIC with broadcast and replay on join"],
        "the P3 test row shares only 'topic' + 'broadcast' of five words"
    );
    let blocking: Vec<String> = verdict
        .blocking_findings()
        .into_iter()
        .map(|f| f.title)
        .collect();
    assert_eq!(
        blocking,
        vec![
            "Send replies on tx so Welcome/OK/ERROR reach the client".to_string(),
            "Remove kicked target from channel fan-out".to_string()
        ]
    );
    assert!(verdict.has_blocking());
    assert_eq!(verdict.blocking_feature_gaps(), 1);
    assert_eq!(verdict.findings.len(), 4, "gaps stay in the report");

    // No unchecked rows: nothing is a gap.
    verdict.mark_feature_gaps(&[]);
    assert!(verdict.findings.iter().all(|f| !f.feature_gap));
    assert_eq!(verdict.blocking_findings().len(), 3);
}

/// A reply that echoes the template (or leaves skeleton placeholders) must
/// not parse into a verdict: `verdict: COMPLETE | INCOMPLETE` is not
/// COMPLETE and `finding: P0|P1|P2|P3 | …` is not a P0.
#[test]
fn verdict_parse_ignores_template_rows() {
    assert_eq!(ReviewVerdict::parse(REVIEW_FORMAT_HINT), None);
    assert_eq!(ReviewVerdict::parse(REVIEW_FORMAT_BLOCK), None);
    let echoed = format_reask_prompt(&["Welcome".into(), "/topic".into()], &[], false);
    assert_eq!(ReviewVerdict::parse(&echoed), None, "unfilled skeleton");

    // Partly filled: a state and item make a row even with the citation
    // placeholder left in; an unfilled `<state>` row and the template
    // finding row are dropped.
    let partly = "<review>\nverdict: INCOMPLETE\n\
coverage: implemented | Welcome | <path:line or ->\n\
coverage: <state> | /topic | <path:line or ->\n\
finding: P0|P1|P2|P3 | <imperative title> | <path:line>\n\
finding: P1 | Drop kicked members | src/main.rs:150\n\
residual: <one line>\n</review>";
    let verdict = ReviewVerdict::parse(partly).expect("filled rows parse");
    assert!(!verdict.complete);
    assert_eq!(verdict.coverage.len(), 1);
    assert_eq!(verdict.coverage[0].item, "Welcome");
    assert_eq!(verdict.coverage[0].evidence, None);
    assert_eq!(verdict.findings.len(), 1);
    assert_eq!(verdict.findings[0].title, "Drop kicked members");
    assert_eq!(verdict.residual, None);

    // "INCOMPLETE" alone still parses; only the literal alternatives are
    // treated as an echo.
    let plain =
        ReviewVerdict::parse("<review>\nverdict: INCOMPLETE\nfinding: none\n</review>").unwrap();
    assert!(!plain.complete);
}

/// A defects-only audit has no plan/spec items; `coverage: none` says so
/// the way `finding: none` does, and is not a row that failed to parse.
#[test]
fn coverage_none_is_not_an_unparsed_row() {
    let verdict = ReviewVerdict::parse(
        "<review>\nverdict: COMPLETE\ncoverage: none\ncoverage: -\nfinding: none\n</review>",
    )
    .unwrap();
    assert!(verdict.coverage.is_empty());
    assert_eq!(verdict.unparsed_coverage_rows, 0);
    assert_eq!(verdict.verdict_note(), None);
    assert!(verdict.complete, "no rows: the word stands");
    assert_eq!(REVIEW_DEFECTS_FORMAT_BLOCK.matches("coverage:").count(), 0);
    assert_eq!(ReviewVerdict::parse(REVIEW_DEFECTS_FORMAT_BLOCK), None);
}

/// A chunked audit folds one verdict per chunk into the audit's verdict.
#[test]
fn merge_keeps_best_coverage_state_and_dedups_findings() {
    let mut first = ReviewVerdict::parse(
        "<review>\nverdict: INCOMPLETE\n\
coverage: missing | Welcome banner | -\n\
coverage: partial | KICK command | crates/a/src/lib.rs:4\n\
finding: P0 | Reject empty nicknames | crates/a/src/lib.rs:9\n\
finding: P2 | Log unknown commands | crates/a/src/lib.rs:20\n\
residual: chunk a\n</review>",
    )
    .unwrap();
    let second = ReviewVerdict::parse(
        "<review>\nverdict: COMPLETE\n\
coverage: implemented | welcome banner | crates/b/src/lib.rs:1\n\
coverage: missing | KICK command | -\n\
finding: P0 | Reject empty nicknames | crates/a/src/lib.rs:9\n\
finding: P1 | Drop kicked members | crates/b/src/lib.rs:30\n\
residual: chunk b\n</review>",
    )
    .unwrap();
    first.merge(second);
    assert_eq!(first.coverage.len(), 2, "items match case-insensitively");
    assert_eq!(first.coverage[0].state, CoverageState::Implemented);
    assert_eq!(
        first.coverage[0].evidence.as_deref(),
        Some("crates/b/src/lib.rs:1"),
        "the winning row brings its citation"
    );
    assert_eq!(first.coverage[0].item, "welcome banner");
    assert_eq!(first.coverage[1].state, CoverageState::Partial);
    assert_eq!(
        first.coverage[1].evidence.as_deref(),
        Some("crates/a/src/lib.rs:4"),
        "missing does not replace partial"
    );
    let titles: Vec<&str> = first.findings.iter().map(|f| f.title.as_str()).collect();
    assert_eq!(
        titles,
        vec![
            "Reject empty nicknames",
            "Log unknown commands",
            "Drop kicked members"
        ]
    );
    assert_eq!(first.residual.as_deref(), Some("chunk a; chunk b"));
    assert_eq!(first.stated_complete, Some(false), "any INCOMPLETE wins");
    assert!(!first.complete, "a partial row keeps the audit incomplete");
    assert_eq!(first.blocking_findings().len(), 2);

    // Defects-only chunks: no rows; the merged word is COMPLETE only when
    // every chunk said so.
    let mut clean =
        ReviewVerdict::parse("<review>\nverdict: COMPLETE\nfinding: none\n</review>").unwrap();
    clean.merge(
        ReviewVerdict::parse("<review>\nverdict: COMPLETE\nfinding: none\n</review>").unwrap(),
    );
    assert!(clean.complete && clean.findings.is_empty() && clean.residual.is_none());
    clean.merge(
        ReviewVerdict::parse(
            "<review>\nverdict: INCOMPLETE\nfinding: P1 | Close the socket | src/net.rs:4\n</review>",
        )
        .unwrap(),
    );
    assert!(!clean.complete);
    assert_eq!(clean.findings.len(), 1);
}
