//! Mid-turn fast feedback after a mutating tool batch.
//!
//! Tier 1: LSP diagnostics on changed source files (rs/py/go/js/ts).
//! Tier 2: affected-package `cargo check` when LSP is clean or unavailable (Rust).
//! Tier 3: package-local tests when the task is test-gated:
//!   - Rust: `cargo test` after a green check
//!   - Python / JS / Go: pytest / npm test / go test on affected packages
//!
//! Failures are appended into the transcript tool results so the model sees them
//! before the next reasoning step — not only as a UI status line.

use std::collections::BTreeSet;
use std::path::PathBuf;

use futures_util::future::join_all;
use hi_tools::infra::{
    CargoCommandOutcome, affected_any_package_dirs, format_lsp_error_feedback, go_source_paths,
    has_pending_affected_polyglot_checks, has_pending_affected_polyglot_tests,
    javascript_source_paths, lsp_source_paths, python_source_paths, run_affected_cargo_checks,
    run_affected_cargo_tests, run_affected_polyglot_checks, run_affected_polyglot_tests,
    rust_source_paths,
};

use super::fast_feedback_observations::{input_revision, package_outcome};
use crate::Ui;
use crate::workspace_runtime::WorkspaceRuntime;

/// Mutable turn-local state for fast feedback dedupe and turn-end stage skip.
#[derive(Debug, Default)]
pub(crate) struct FastFeedbackState {
    /// Cargo packages already `cargo check`'d clean this turn (relative labels, or `"."`).
    pub checked_packages: BTreeSet<String>,
    /// Cargo packages already `cargo test`'d clean this turn.
    pub tested_packages: BTreeSet<String>,
    /// Package → ledger revision when last sealed green by mid-turn check.
    /// WorkspaceRepair skips matching `affected-check:` stages when still current.
    pub sealed_checks: std::collections::BTreeMap<String, u64>,
    /// Package → ledger revision when last sealed green by mid-turn test.
    pub sealed_tests: std::collections::BTreeMap<String, u64>,
    /// A fast cargo check hit its time budget this turn (cold target dir).
    /// Further fast cargo checks are skipped for the rest of the turn: each
    /// re-arm eats the whole budget again and returns no evidence — a live
    /// turn burned one full budget per edit this way. Turn-end verification
    /// (with its cold-build retry) still covers the packages.
    pub cargo_timed_out: bool,
}

impl FastFeedbackState {
    /// Packages whose mid-turn `cargo check` is still valid at `ledger_revision`.
    pub fn skippable_check_packages(&self, ledger_revision: u64) -> BTreeSet<String> {
        self.sealed_checks
            .iter()
            .filter(|(_, rev)| **rev == ledger_revision)
            .map(|(pkg, _)| pkg.clone())
            .collect()
    }

    /// Packages whose mid-turn `cargo test` is still valid at `ledger_revision`.
    pub fn skippable_test_packages(&self, ledger_revision: u64) -> BTreeSet<String> {
        self.sealed_tests
            .iter()
            .filter(|(_, rev)| **rev == ledger_revision)
            .map(|(pkg, _)| pkg.clone())
            .collect()
    }

    fn invalidate_packages(&mut self, packages: &BTreeSet<String>) {
        if packages.is_empty() {
            // Root-only / unknown package touch — drop root seals.
            remove_package_keys(&mut self.checked_packages, ".");
            remove_package_keys(&mut self.tested_packages, ".");
            self.sealed_checks.remove(".");
            self.sealed_tests.remove(".");
            return;
        }
        for package in packages {
            remove_package_keys(&mut self.checked_packages, package);
            remove_package_keys(&mut self.tested_packages, package);
            self.sealed_checks.remove(package);
            self.sealed_tests.remove(package);
        }
    }

    fn seal_checks_at(&mut self, packages: &[String], revision: u64) {
        for package in packages {
            self.sealed_checks.insert(package.clone(), revision);
        }
    }

    fn seal_tests_at(&mut self, packages: &[String], revision: u64) {
        for package in packages {
            self.sealed_tests.insert(package.clone(), revision);
        }
    }
}

/// Remove both bare Cargo labels and kind-qualified polyglot labels such as
/// `typecheck::web`. A single directory can contain multiple ecosystems, so
/// the fast-feedback dedupe set must distinguish those jobs while invalidation
/// still needs to clear every check owned by the touched package.
fn remove_package_keys(keys: &mut BTreeSet<String>, package: &str) {
    let suffix = format!("::{package}");
    keys.retain(|key| key != package && !key.ends_with(&suffix));
}

#[derive(Debug, Default)]
pub(crate) struct FastFeedbackReport {
    pub observations: Vec<crate::recovery::ValidationObservation>,
    /// Model-facing success blocks to append onto tool results / nudge. These
    /// let the next reasoning step reuse a real package check instead of
    /// launching a duplicate shell validation.
    pub passes: Vec<String>,
    /// Model-facing failure blocks to append onto tool results / nudge.
    pub failures: Vec<String>,
    pub lsp_errors: u32,
    pub cargo_failed: bool,
    pub cargo_ran: bool,
    pub tests_failed: bool,
    pub tests_ran: bool,
    pub tests_timed_out: bool,
}

/// Keywords that introduce a definition whose signature callers depend on.
const DEFINITION_KEYWORDS: &[&str] = &[
    "fn",
    "struct",
    "enum",
    "trait",
    "type",
    "function",
    "class",
    "def",
    "func",
    "interface",
];
/// Leading modifiers allowed before a definition keyword on its line.
const DEFINITION_MODIFIERS: &[&str] = &[
    "pub",
    "pub(crate)",
    "pub(super)",
    "async",
    "unsafe",
    "export",
    "default",
    "abstract",
    "static",
    "extern",
    "const",
];
/// Skip the enclosing-definition fallback on files larger than this. The
/// names extracted from the edited region itself need no disk read.
const MAX_ENCLOSING_DEF_FILE_BYTES: u64 = 256 * 1024;
/// Definition names queried per batch — impact stays a hint, not a report.
const MAX_IMPACT_SYMBOLS: usize = 3;
/// Referencing files listed per symbol.
const MAX_IMPACT_FILES: usize = 5;
/// Budget for one reverse-reference query; a cold index misses the batch and
/// answers the next one instead of stalling this one.
const IMPACT_QUERY_TIMEOUT_MS: u64 = 2_000;

/// Definition names declared in `region` — lines where a definition keyword
/// (optionally behind modifiers) opens the line. This is the text the model
/// actually touched, so a hit means "an edit landed on this signature's line".
fn extract_definition_names(region: &str) -> Vec<String> {
    let mut names = Vec::new();
    for line in region.lines() {
        let mut tokens = line.split_whitespace().peekable();
        while let Some(token) = tokens.peek() {
            if DEFINITION_MODIFIERS.contains(token) {
                tokens.next();
            } else {
                break;
            }
        }
        let Some(keyword) = tokens.next() else {
            continue;
        };
        if !DEFINITION_KEYWORDS.contains(&keyword) {
            continue;
        }
        let Some(raw) = tokens.next() else { continue };
        let name: String = raw
            .chars()
            .take_while(|c| c.is_alphanumeric() || *c == '_')
            .collect();
        if name.len() > 1 && !names.contains(&name) {
            names.push(name);
        }
    }
    names
}

/// Definition names an edit touches: names declared on the edited lines
/// themselves, else the enclosing definition the region sits inside. Feature
/// changes usually edit bodies and builder chains, not signature lines —
/// measured on gold multi-file patches, 32 of 47 edited regions declared
/// nothing of their own.
pub(crate) fn definition_names_for_edit(
    root: &std::path::Path,
    file: &str,
    region: &str,
) -> Vec<String> {
    let names = extract_definition_names(region);
    if !names.is_empty() {
        return names;
    }
    let path = root.join(file);
    let Ok(metadata) = std::fs::metadata(&path) else {
        return names;
    };
    if !metadata.is_file() || metadata.len() > MAX_ENCLOSING_DEF_FILE_BYTES {
        return names;
    }
    let Ok(handle) = std::fs::File::open(&path) else {
        return names;
    };
    use std::io::Read;
    let mut buf = Vec::new();
    if handle
        .take(MAX_ENCLOSING_DEF_FILE_BYTES)
        .read_to_end(&mut buf)
        .is_err()
    {
        return names;
    }
    let text = String::from_utf8_lossy(&buf);
    // Anchor the region in the file by its first line that appears exactly
    // once, then scan upward for the nearest enclosing definition.
    let file_lines: Vec<&str> = text.lines().collect();
    let mut anchor = None;
    for line in region.lines().map(str::trim).filter(|line| line.len() > 8) {
        let mut matches = file_lines
            .iter()
            .enumerate()
            .filter(|(_, file_line)| file_line.trim() == line);
        if let (Some((index, _)), None) = (matches.next(), matches.next()) {
            anchor = Some(index);
            break;
        }
    }
    let Some(anchor) = anchor else { return names };
    for i in (0..=anchor).rev() {
        let mut found = extract_definition_names(file_lines[i]);
        if !found.is_empty() {
            return vec![found.remove(0)];
        }
    }
    names
}

/// Reverse-reference notes for definitions the batch edited: which *other*
/// files call into what just changed. Proactive — the model updates call
/// sites before the compiler reports them one error at a time.
pub(crate) async fn signature_impact_notes(
    runtime: &WorkspaceRuntime,
    edited_regions: &[(String, String)],
) -> Vec<String> {
    // Finding an enclosing definition may read the entire edited file. Keep
    // that fallback off the agent executor; the reference queries below are
    // already asynchronous and can overlap with the rest of the turn.
    let root = runtime.root().to_path_buf();
    let regions = edited_regions.to_vec();
    let definitions = match tokio::task::spawn_blocking(move || {
        regions
            .iter()
            .flat_map(|(path, region)| {
                definition_names_for_edit(&root, path, region)
                    .into_iter()
                    .map(|name| (path.clone(), name))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>()
    })
    .await
    {
        Ok(definitions) => definitions,
        Err(_) => return Vec::new(),
    };
    let root = runtime.root().to_path_buf();
    let query_results = join_all(definitions.into_iter().take(MAX_IMPACT_SYMBOLS).map(
        |(path, name)| {
            let root = root.clone();
            async move {
                let query = hi_tools::references_by_name(&root, &name, Some(&path));
                let locations = tokio::time::timeout(
                    std::time::Duration::from_millis(IMPACT_QUERY_TIMEOUT_MS),
                    query,
                )
                .await
                .ok()
                .flatten();
                (path, name, locations)
            }
        },
    ))
    .await;
    let mut notes = Vec::new();
    for (path, name, locations) in query_results {
        let Some(locations) = locations else { continue };
        let mut files: Vec<String> = Vec::new();
        for location in &locations {
            let file = location
                .rsplit_once(':')
                .map_or(location.as_str(), |(f, _)| f);
            let file = file.strip_prefix('/').map_or(file, |_| {
                std::path::Path::new(file)
                    .strip_prefix(runtime.root())
                    .ok()
                    .and_then(|p| p.to_str())
                    .unwrap_or(file)
            });
            if file != path && !files.iter().any(|seen| seen == file) {
                files.push(file.to_string());
            }
        }
        if files.is_empty() {
            continue;
        }
        let shown = files
            .iter()
            .take(MAX_IMPACT_FILES)
            .cloned()
            .collect::<Vec<_>>();
        let more = files.len().saturating_sub(shown.len());
        let suffix = if more > 0 {
            format!(" (+{more} more)")
        } else {
            String::new()
        };
        notes.push(format!(
            "signature impact: `{name}` (edited in {path}) is referenced from {} other file(s): {}{suffix} — if its signature or behavior contract changed, update those call sites now.",
            files.len(),
            shown.join(", "),
        ));
    }
    notes
}

impl FastFeedbackReport {
    /// Credit stable successful checks without making the model rerun them.
    /// Compilation alone cannot satisfy a task that explicitly requires tests.
    pub(crate) fn validates_implementation(&self, requires_tests: bool) -> bool {
        !self.observations.is_empty()
            && self
                .observations
                .iter()
                .all(|o| o.status == crate::recovery::ValidationResult::Passed)
            && self.failures.is_empty()
            && self.lsp_errors == 0
            && !self.cargo_failed
            && !self.tests_failed
            && !self.tests_timed_out
            && (!requires_tests || self.tests_ran)
    }

    pub fn combined_feedback(&self) -> Option<String> {
        let mut blocks = Vec::with_capacity(self.passes.len() + self.failures.len());
        blocks.extend(self.passes.iter().cloned());
        blocks.extend(self.failures.iter().cloned());
        if blocks.is_empty() {
            None
        } else {
            Some(blocks.join("\n\n"))
        }
    }
}

/// Options for one mid-turn fast-feedback pass.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct FastFeedbackOptions {
    /// When true (task contract wants tests), run package-local tests after a
    /// clean check (Rust) or on polyglot packages. Never full-workspace suites.
    pub run_tests: bool,
}

/// Run post-batch fast feedback for `changed_paths` (project-relative).
pub(crate) async fn run_fast_feedback(
    runtime: &WorkspaceRuntime,
    changed_paths: &[String],
    state: &mut FastFeedbackState,
    options: FastFeedbackOptions,
    ui: &mut dyn Ui,
) -> FastFeedbackReport {
    let mut report = FastFeedbackReport::default();
    if changed_paths.is_empty() {
        return report;
    }

    let rust_paths = rust_source_paths(changed_paths.iter());
    let diag_paths = lsp_source_paths(changed_paths.iter());
    // Non-source edits do not need mid-turn package feedback. In particular,
    // avoid announcing an empty tsc/go/ruff phase for docs/config/text edits;
    // per-file checks for other languages are handled by the mutation batch.
    if diag_paths.is_empty() {
        return report;
    }
    let workspace_lockfile = runtime.root().join("Cargo.lock");
    let lockfile_preexisting = workspace_lockfile.is_file();
    let has_polyglot_sources = has_polyglot_sources(changed_paths);
    let mut lsp_checked_clean = false;
    let mut lsp_unavailable = true;

    if runtime.lsp_enabled() && !diag_paths.is_empty() {
        let lsp = runtime.lsp();
        // Do not cold-start a language server in the middle of a mutation
        // turn. On a large Rust workspace that startup can dominate the turn
        // and may perform its own project discovery. Explicit LSP queries can
        // still start it; once warm, diagnostics remain a fast tier.
        if lsp.is_enabled().await
            && (runtime.lsp_fast_feedback_cold_start_allowed() || lsp.has_running_server().await)
        {
            lsp_unavailable = false;
            let path_bufs = diag_paths.iter().map(PathBuf::from).collect::<Vec<_>>();
            let mut errors = Vec::new();
            let mut saw_confirmed = false;
            let mut saw_transport_failure = false;
            for (path, diag_state) in lsp.diagnostics_batch(&path_bufs).await {
                match diag_state {
                    hi_lsp::DiagnosticState::ConfirmedClean { .. } => {
                        saw_confirmed = true;
                    }
                    hi_lsp::DiagnosticState::DiagnosticsPresent { diagnostics, .. } => {
                        saw_confirmed = true;
                        let display = path_display(runtime.root(), &path);
                        for d in diagnostics {
                            if d.severity == "error" {
                                errors.push((display.clone(), d.line + 1, d.col + 1, d.message));
                            }
                        }
                    }
                    hi_lsp::DiagnosticState::Failed { error, .. } => {
                        saw_transport_failure = true;
                        // LSP is an optional accelerator. Keep transport or
                        // installation details out of the transcript; the
                        // package-specific fallback below is the useful
                        // user-facing result.
                        let _ = (path, error);
                    }
                    hi_lsp::DiagnosticState::Unavailable { .. } => {}
                }
            }
            if !errors.is_empty() {
                report.lsp_errors = errors.len() as u32;
                let text = format_lsp_error_feedback(&errors);
                ui.status(&text);
                report.failures.push(text);
                remove_new_lsp_lockfile(&workspace_lockfile, lockfile_preexisting);
                // LSP already found compile-level issues — skip cargo this batch.
                return report;
            }
            // Transport death (closed stream, poison) is not a clean bill of
            // health — fall through to cargo check instead of sealing green.
            if saw_transport_failure && !saw_confirmed {
                lsp_unavailable = true;
                // Cargo is only the fallback for Rust edits. Polyglot batches
                // go straight to their package check below; announcing a
                // Cargo fallback for them is misleading and adds noise to an
                // otherwise healthy source-edit turn.
                if !rust_paths.is_empty() {
                    ui.status(
                        "fast check · editor diagnostics unavailable; checking Rust packages…",
                    );
                }
            } else {
                lsp_checked_clean = saw_confirmed;
            }
        }
    }
    // rust-analyzer may create Cargo.lock while discovering a lockfile-less
    // project. That is an editor side effect, not a user edit; clean it before
    // the package check and change ledger can observe it. Existing lockfiles
    // are never removed.
    remove_new_lsp_lockfile(&workspace_lockfile, lockfile_preexisting);

    // Invalidate seals for any language package touched this batch.
    let touched = affected_any_package_dirs(runtime.root(), changed_paths);
    state.invalidate_packages(&touched);

    // Tier 2a: cargo check when LSP is clean or unavailable and Rust files
    // changed — unless a check already timed out this turn (cold build):
    // re-arming would spend the whole budget again for no evidence.
    let should_cargo =
        !rust_paths.is_empty() && (lsp_checked_clean || lsp_unavailable) && !state.cargo_timed_out;
    let mut checks_ok_for_tests = !should_cargo; // non-Rust batches don't need cargo first
    if should_cargo {
        ui.status("fast check · cargo check (affected packages)…");
        let input = input_revision(runtime).await;
        let outcome =
            run_affected_cargo_checks(runtime.root(), changed_paths, &mut state.checked_packages)
                .await;
        let outcome = package_outcome(runtime, &mut report.observations, outcome, input).await;
        report.cargo_ran = matches!(
            outcome,
            CargoCommandOutcome::Passed { .. }
                | CargoCommandOutcome::Failed { .. }
                | CargoCommandOutcome::TimedOut { .. }
        );
        if let Some(status) = outcome.ui_status()
            && !matches!(outcome, CargoCommandOutcome::Passed { .. })
        {
            ui.status(&status);
        }
        if matches!(outcome, CargoCommandOutcome::TimedOut { .. }) {
            // Not evidence about the code — no model-facing failure, no
            // unsealing. Disarm fast cargo checks for the rest of the turn.
            state.cargo_timed_out = true;
        }
        if let Some(failure) = outcome.failure_message() {
            report.cargo_failed = true;
            if let CargoCommandOutcome::Failed { package, .. } = &outcome {
                remove_package_keys(&mut state.checked_packages, package);
                state.sealed_checks.remove(package);
            }
            report.failures.push(failure);
            return report;
        }
        let ledger_revision = runtime.ledger().revision();
        if let CargoCommandOutcome::Passed { packages, .. } = &outcome {
            report.passes.push(format_pass_feedback(&outcome, "check"));
            state.seal_checks_at(packages, ledger_revision);
            checks_ok_for_tests = true;
        } else if matches!(outcome, CargoCommandOutcome::Skipped) {
            checks_ok_for_tests = true;
        }
    }

    // Tier 2b: polyglot typecheck/build/lint (tsc / go build / ruff) — always when
    // those languages changed (not only test-gated). Seals share check namespace.
    let poly_check = if has_polyglot_sources
        && has_pending_affected_polyglot_checks(
            runtime.root(),
            changed_paths,
            &state.checked_packages,
        ) {
        ui.status(&format!(
            "fast check · {} package checks…",
            polyglot_language_label(changed_paths)
        ));
        let input = input_revision(runtime).await;
        let outcome = run_affected_polyglot_checks(
            runtime.root(),
            changed_paths,
            &mut state.checked_packages,
        )
        .await;
        package_outcome(runtime, &mut report.observations, outcome, input).await
    } else {
        CargoCommandOutcome::Skipped
    };
    report.cargo_ran |= matches!(
        poly_check,
        CargoCommandOutcome::Passed { .. }
            | CargoCommandOutcome::Failed { .. }
            | CargoCommandOutcome::TimedOut { .. }
    );
    if let Some(status) = poly_check.ui_status()
        && !matches!(poly_check, CargoCommandOutcome::Passed { .. })
    {
        ui.status(&status);
    }
    if matches!(poly_check, CargoCommandOutcome::TimedOut { .. }) {
        // A slow typecheck/build is infrastructure evidence, not a code
        // failure. Stop re-arming all fast checks for this turn rather than
        // making every subsequent edit pay the same timeout again.
        state.cargo_timed_out = true;
    }
    if let Some(failure) = poly_check.failure_message() {
        report.cargo_failed = true;
        if let CargoCommandOutcome::Failed { package, .. } = &poly_check {
            remove_package_keys(&mut state.checked_packages, package);
            state.sealed_checks.remove(package);
        }
        report.failures.push(failure);
        return report;
    }
    if let CargoCommandOutcome::TimedOut { package, .. } = &poly_check {
        remove_package_keys(&mut state.checked_packages, package);
        state.sealed_checks.remove(package);
        checks_ok_for_tests = false;
    }
    if let CargoCommandOutcome::Passed { packages, .. } = &poly_check {
        report
            .passes
            .push(format_pass_feedback(&poly_check, "check"));
        let revision = runtime.ledger().revision();
        state.seal_checks_at(packages, revision);
        checks_ok_for_tests = true;
    }
    if matches!(poly_check, CargoCommandOutcome::Unavailable { .. }) {
        checks_ok_for_tests = false;
    }
    // Skipped polyglot checks leave checks_ok_for_tests as set by cargo tier.

    // Tier 3: package-local tests when the task is test-gated.
    if !options.run_tests || !checks_ok_for_tests {
        return report;
    }

    // Rust tests (after green check). A cold-build timeout disarms these too:
    // `cargo test` on a cold tree is strictly slower than the check that
    // already failed to finish.
    if !rust_paths.is_empty() && !state.cargo_timed_out {
        ui.status("fast check · cargo test (affected packages)…");
        let input = input_revision(runtime).await;
        let test_outcome =
            run_affected_cargo_tests(runtime.root(), changed_paths, &mut state.tested_packages)
                .await;
        let test_outcome =
            package_outcome(runtime, &mut report.observations, test_outcome, input).await;
        report.tests_ran |= matches!(
            test_outcome,
            CargoCommandOutcome::Passed { .. }
                | CargoCommandOutcome::Failed { .. }
                | CargoCommandOutcome::TimedOut { .. }
        );
        if let Some(status) = test_outcome.ui_status()
            && !matches!(test_outcome, CargoCommandOutcome::Passed { .. })
        {
            ui.status(&status);
        }
        if matches!(test_outcome, CargoCommandOutcome::TimedOut { .. }) {
            state.cargo_timed_out = true;
            report.tests_timed_out = true;
        }
        if let Some(failure) = test_outcome.failure_message() {
            report.tests_failed = true;
            if let CargoCommandOutcome::Failed { package, .. } = &test_outcome {
                remove_package_keys(&mut state.tested_packages, package);
                state.sealed_tests.remove(package);
            }
            report.failures.push(failure);
            return report;
        }
        if let CargoCommandOutcome::Passed { packages, .. } = &test_outcome {
            report
                .passes
                .push(format_pass_feedback(&test_outcome, "tests"));
            let revision = runtime.ledger().revision();
            state.seal_tests_at(packages, revision);
        }
    }

    // Polyglot package tests (pytest / npm test / go test).
    let poly_outcome = if has_polyglot_sources
        && has_pending_affected_polyglot_tests(
            runtime.root(),
            changed_paths,
            &state.tested_packages,
        ) {
        ui.status(&format!(
            "fast check · {} package tests…",
            polyglot_language_label(changed_paths)
        ));
        let input = input_revision(runtime).await;
        let outcome =
            run_affected_polyglot_tests(runtime.root(), changed_paths, &mut state.tested_packages)
                .await;
        package_outcome(runtime, &mut report.observations, outcome, input).await
    } else {
        CargoCommandOutcome::Skipped
    };
    report.tests_ran |= matches!(
        poly_outcome,
        CargoCommandOutcome::Passed { .. }
            | CargoCommandOutcome::Failed { .. }
            | CargoCommandOutcome::TimedOut { .. }
    );
    if let Some(status) = poly_outcome.ui_status()
        && !matches!(poly_outcome, CargoCommandOutcome::Passed { .. })
    {
        ui.status(&status);
    }
    if let Some(failure) = poly_outcome.failure_message() {
        report.tests_failed = true;
        if let CargoCommandOutcome::Failed { package, .. } = &poly_outcome {
            remove_package_keys(&mut state.tested_packages, package);
            state.sealed_tests.remove(package);
        }
        report.failures.push(failure);
        return report;
    }
    if let CargoCommandOutcome::TimedOut { package, .. } = &poly_outcome {
        report.tests_timed_out = true;
        remove_package_keys(&mut state.tested_packages, package);
        state.sealed_tests.remove(package);
    }
    if let CargoCommandOutcome::Passed { packages, .. } = &poly_outcome {
        report
            .passes
            .push(format_pass_feedback(&poly_outcome, "tests"));
        let revision = runtime.ledger().revision();
        state.seal_tests_at(packages, revision);
    }
    report
}

fn format_pass_feedback(outcome: &CargoCommandOutcome, phase: &str) -> String {
    let CargoCommandOutcome::Passed { command, packages } = outcome else {
        unreachable!("only passed outcomes have pass feedback");
    };
    const MAX_SHOWN_PACKAGES: usize = 6;
    let mut shown = packages
        .iter()
        .take(MAX_SHOWN_PACKAGES)
        .map(String::as_str)
        .collect::<Vec<_>>();
    if packages.len() > MAX_SHOWN_PACKAGES {
        shown.push("…");
    }
    format!("✓ fast {phase} passed · {command} ({})", shown.join(", "))
}

fn path_display(root: &std::path::Path, path: &std::path::Path) -> String {
    path.strip_prefix(root)
        .unwrap_or(path)
        .to_string_lossy()
        .replace('\\', "/")
}

fn has_polyglot_sources(changed_paths: &[String]) -> bool {
    !python_source_paths(changed_paths.iter()).is_empty()
        || !javascript_source_paths(changed_paths.iter()).is_empty()
        || !go_source_paths(changed_paths.iter()).is_empty()
}

fn polyglot_language_label(changed_paths: &[String]) -> &'static str {
    let python = !python_source_paths(changed_paths.iter()).is_empty();
    let javascript = !javascript_source_paths(changed_paths.iter()).is_empty();
    let go = !go_source_paths(changed_paths.iter()).is_empty();
    match (python, javascript, go) {
        (true, false, false) => "Python",
        (false, true, false) => "JavaScript/TypeScript",
        (false, false, true) => "Go",
        _ => "polyglot",
    }
}

fn remove_new_lsp_lockfile(path: &std::path::Path, preexisting: bool) {
    if !preexisting && path.is_file() {
        let _ = std::fs::remove_file(path);
    }
}

#[cfg(test)]
#[path = "fast_feedback_tests.rs"]
mod tests;

#[cfg(test)]
mod implementation_validation_tests {
    use super::*;
    use crate::recovery::{ValidationObservation, ValidationResult};

    #[test]
    fn checks_count_but_missing_failed_or_stale_tests_do_not() {
        let mut report = FastFeedbackReport::default();
        assert!(!report.validates_implementation(false));
        report.observations.push(ValidationObservation::command(
            "check".into(),
            "cargo check",
            "revision".into(),
            ValidationResult::Passed,
            "",
            std::path::Path::new("."),
            false,
        ));
        assert!(report.validates_implementation(false));
        assert!(!report.validates_implementation(true));
        report.tests_ran = true;
        assert!(report.validates_implementation(true));
        for status in [
            ValidationResult::Failed,
            ValidationResult::Deferred,
            ValidationResult::Infrastructure,
        ] {
            report.observations[0].status = status;
            assert!(!report.validates_implementation(false));
            assert!(!report.validates_implementation(true));
        }
        report.observations[0].status = ValidationResult::Passed;
        report.tests_timed_out = true;
        assert!(!report.validates_implementation(false));
        report.tests_timed_out = false;
        report.lsp_errors = 1;
        assert!(!report.validates_implementation(false));
    }
}
