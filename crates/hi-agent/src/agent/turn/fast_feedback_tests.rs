use super::*;

/// Corpus harness: do impact notes predict real co-change? For gold
/// multi-file fixes from Multi-SWE-bench, extract definitions from the
/// primary file's pre-image hunks and check whether reverse references
/// land in the other files the maintainers' fix also touched.
/// Reporting-only:
/// `HI_IMPACT_CORPUS=<records.jsonl> cargo test -p hi-agent --lib \
///  impact_corpus -- --ignored --nocapture`
#[tokio::test]
#[ignore = "set HI_IMPACT_CORPUS to a jsonl of {root, file, region, others} records"]
async fn impact_corpus_gold_patch_co_change() {
    let Some(path) = std::env::var_os("HI_IMPACT_CORPUS") else {
        return;
    };
    let text = std::fs::read_to_string(path).expect("corpus file");
    let (mut records, mut with_names, mut hits) = (0usize, 0usize, 0usize);
    let mut misses = Vec::new();
    for line in text.lines() {
        let Ok(value) = serde_json::from_str::<serde_json::Value>(line) else {
            continue;
        };
        let get = |k: &str| value.get(k).and_then(|v| v.as_str()).unwrap_or_default();
        let (root, file, region) = (get("root"), get("file"), get("region"));
        let others: Vec<String> = value
            .get("others")
            .and_then(|v| v.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();
        if root.is_empty() || region.is_empty() || others.is_empty() {
            continue;
        }
        records += 1;
        let names = definition_names_for_edit(Path::new(root), file, region);
        if names.is_empty() {
            misses.push(format!(
                "no definitions extracted: {} {}",
                get("instance"),
                file
            ));
            continue;
        }
        with_names += 1;
        let mut hit = false;
        for name in names.iter().take(MAX_IMPACT_SYMBOLS) {
            let query = hi_tools::references_by_name(Path::new(root), name, Some(file));
            let Ok(Some(locations)) =
                tokio::time::timeout(std::time::Duration::from_secs(20), query).await
            else {
                continue;
            };
            if locations.iter().any(|loc| {
                let loc_file = loc.rsplit_once(':').map_or(loc.as_str(), |(f, _)| f);
                others
                    .iter()
                    .any(|other| loc_file.ends_with(other.as_str()))
            }) {
                hit = true;
                break;
            }
        }
        if hit {
            hits += 1;
        } else {
            misses.push(format!(
                "no co-change hit: {} {} (names {:?} → others {:?})",
                get("instance"),
                file,
                names.iter().take(3).collect::<Vec<_>>(),
                others
            ));
        }
    }
    println!(
        "impact corpus: {records} records · {with_names} with extractable definitions · {hits} co-change hits"
    );
    for miss in misses.iter().take(12) {
        println!("  {miss}");
    }
}

use std::path::Path;

#[test]
fn enclosing_definition_fallback_reads_a_small_file() {
    let root = std::env::temp_dir().join(format!(
        "hi-def-small-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(
        root.join("lib.rs"),
        "fn parse_config() {\n    let parsed = load(path);\n}\n",
    )
    .unwrap();
    let names = definition_names_for_edit(&root, "lib.rs", "    let parsed = load(path);");
    assert_eq!(names, ["parse_config"]);
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn enclosing_definition_fallback_skips_huge_files() {
    let root = std::env::temp_dir().join(format!(
        "hi-def-huge-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&root).unwrap();
    let mut body = String::from("fn parse_config() {\n    let parsed = load(path);\n");
    body.push_str(&"z".repeat(MAX_ENCLOSING_DEF_FILE_BYTES as usize + 64));
    body.push_str("\n}\n");
    std::fs::write(root.join("lib.rs"), &body).unwrap();
    let names = definition_names_for_edit(&root, "lib.rs", "    let parsed = load(path);");
    assert!(
        names.is_empty(),
        "huge file must not be slurped for enclosing defs: {names:?}"
    );
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn definition_names_come_from_definition_lines_only() {
    let region = "\
pub fn parse_config(path: &Path) -> Config {
    let parsed = load(path);
    call_site(parse_other);
}
pub(crate) struct Loader {
export default class Widget extends Base {
def handle_request(self):
    fn_like_variable = 1
";
    let names = extract_definition_names(region);
    assert_eq!(
        names,
        ["parse_config", "Loader", "Widget", "handle_request"],
        "call sites and non-definition lines must not contribute"
    );
    assert!(extract_definition_names("    x += 1;\n").is_empty());
}

#[test]
fn report_combines_failures() {
    let mut report = FastFeedbackReport::default();
    report.failures.push("a".into());
    report.failures.push("b".into());
    assert_eq!(report.combined_feedback().as_deref(), Some("a\n\nb"));
}

#[test]
fn report_replays_passes_before_failures() {
    let mut report = FastFeedbackReport::default();
    report.passes.push("check passed".into());
    report.failures.push("tests failed".into());
    assert_eq!(
        report.combined_feedback().as_deref(),
        Some("check passed\n\ntests failed")
    );
}

#[test]
fn pass_feedback_names_the_checked_packages() {
    let outcome = CargoCommandOutcome::Passed {
        command: "cargo check",
        packages: vec!["crates/hi-agent".into(), "crates/hi-tools".into()],
    };
    assert_eq!(
        format_pass_feedback(&outcome, "check"),
        "✓ fast check passed · cargo check (crates/hi-agent, crates/hi-tools)"
    );
}

#[test]
fn seals_are_revision_sensitive() {
    let mut state = FastFeedbackState::default();
    state.seal_checks_at(&["crates/demo".into(), ".".into()], 3);
    state.seal_tests_at(&["crates/demo".into()], 3);
    assert_eq!(
        state.skippable_check_packages(3),
        BTreeSet::from(["crates/demo".into(), ".".into()])
    );
    assert!(state.skippable_check_packages(4).is_empty());
    assert_eq!(
        state.skippable_test_packages(3),
        BTreeSet::from(["crates/demo".into()])
    );
    // Mutation of demo drops its seals only.
    let mut touched = BTreeSet::new();
    touched.insert("crates/demo".into());
    state.invalidate_packages(&touched);
    assert_eq!(
        state.skippable_check_packages(3),
        BTreeSet::from([".".into()])
    );
    assert!(state.skippable_test_packages(3).is_empty());
}

#[test]
fn package_feedback_is_gated_by_polyglot_source_changes() {
    assert!(!has_polyglot_sources(&[
        "README.md".into(),
        "config.toml".into()
    ]));
    assert!(!has_polyglot_sources(
        &["crates/hi-ai/src/openai.rs".into()]
    ));
    assert!(has_polyglot_sources(&["src/main.py".into()]));
    assert!(has_polyglot_sources(&["web/app.ts".into()]));
    assert!(has_polyglot_sources(&["cmd/main.go".into()]));
    assert_eq!(polyglot_language_label(&["src/main.py".into()]), "Python");
    assert_eq!(
        polyglot_language_label(&["web/app.ts".into()]),
        "JavaScript/TypeScript"
    );
    assert_eq!(polyglot_language_label(&["cmd/main.go".into()]), "Go");
    assert_eq!(
        polyglot_language_label(&["src/main.py".into(), "web/app.ts".into()]),
        "polyglot"
    );
}

struct StatusUi {
    statuses: Vec<String>,
}

impl crate::Ui for StatusUi {
    fn assistant_text(&mut self, _: &str) {}
    fn assistant_reasoning(&mut self, _: &str) {}
    fn assistant_end(&mut self) {}
    fn tool_call(&mut self, _: &str, _: &str) {}
    fn tool_result(&mut self, _: &str, _: &str) {}
    fn status(&mut self, status: &str) {
        self.statuses.push(status.to_string());
    }
    fn turn_end(&mut self, _: &str) {}
}

#[tokio::test]
async fn cargo_fast_feedback_still_runs_when_lsp_is_clean() {
    if std::process::Command::new("cargo")
        .arg("--version")
        .output()
        .map(|o| !o.status.success())
        .unwrap_or(true)
    {
        eprintln!("skipping: cargo not on PATH");
        return;
    }
    let root = std::env::temp_dir().join(format!(
        "hi-ff-cargo-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(root.join("crates/demo/src")).unwrap();
    std::fs::write(
        root.join("Cargo.toml"),
        "[workspace]\nmembers = [\"crates/demo\"]\n",
    )
    .unwrap();
    std::fs::write(
        root.join("crates/demo/Cargo.toml"),
        "[package]\nname = \"demo\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
    )
    .unwrap();
    std::fs::write(root.join("crates/demo/src/lib.rs"), "pub fn ok() {}\n").unwrap();
    let runtime = crate::workspace_runtime::WorkspaceRuntime::new(
        &root,
        root.join(".hi/state"),
        crate::LspMode::On,
    )
    .unwrap();
    runtime.lsp().set_enabled(true).await;
    runtime.lsp().inject_diagnostics(
        runtime.lsp().root().join("crates/demo/src/lib.rs"),
        hi_lsp::DiagnosticState::ConfirmedClean {
            document_version: 1,
        },
    );
    let mut state = FastFeedbackState::default();
    let mut ui = StatusUi {
        statuses: Vec::new(),
    };
    let report = run_fast_feedback(
        &runtime,
        &["crates/demo/src/lib.rs".into()],
        &mut state,
        FastFeedbackOptions::default(),
        &mut ui,
    )
    .await;
    assert!(
        report.cargo_ran || ui.statuses.iter().any(|s| s.contains("cargo check")),
        "expected cargo check after a clean LSP overlay, statuses={:?} report={report:?}",
        ui.statuses
    );
    let _ = std::fs::remove_dir_all(root);
}
