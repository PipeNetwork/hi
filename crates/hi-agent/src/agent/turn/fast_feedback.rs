//! Mid-turn fast feedback after a mutating tool batch.
//!
//! Tier 1: LSP diagnostics on changed source files (rs/py/go/js/ts).
//! Tier 2: affected-package `cargo check` when LSP is clean or unavailable (Rust).
//! Tier 3: package-local tests when the task is test-gated:
//!   - Rust: `cargo test` after a green check
//!   - Python / JS / Go: pytest / npm test / go test on affected packages
//! Failures are appended into the transcript tool results so the model sees them
//! before the next reasoning step — not only as a UI status line.

use std::collections::BTreeSet;
use std::path::PathBuf;

use hi_tools::{
    CargoCommandOutcome, affected_any_package_dirs, format_lsp_error_feedback, lsp_source_paths,
    run_affected_cargo_checks, run_affected_cargo_tests, run_affected_polyglot_checks,
    run_affected_polyglot_tests, rust_source_paths,
};

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
            self.checked_packages.remove(".");
            self.tested_packages.remove(".");
            self.sealed_checks.remove(".");
            self.sealed_tests.remove(".");
            return;
        }
        for package in packages {
            self.checked_packages.remove(package);
            self.tested_packages.remove(package);
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

#[derive(Debug, Default)]
pub(crate) struct FastFeedbackReport {
    /// Model-facing failure blocks to append onto tool results / nudge.
    pub failures: Vec<String>,
    pub lsp_errors: u32,
    pub cargo_failed: bool,
    pub cargo_ran: bool,
    pub tests_failed: bool,
    pub tests_ran: bool,
}

impl FastFeedbackReport {
    pub fn combined_failure(&self) -> Option<String> {
        if self.failures.is_empty() {
            None
        } else {
            Some(self.failures.join("\n\n"))
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
    let mut lsp_checked_clean = false;
    let mut lsp_unavailable = true;

    if runtime.lsp_enabled() && !diag_paths.is_empty() {
        let lsp = runtime.lsp();
        if lsp.is_enabled().await {
            lsp_unavailable = false;
            let path_bufs = diag_paths
                .iter()
                .map(PathBuf::from)
                .collect::<Vec<_>>();
            let mut errors = Vec::new();
            let mut saw_confirmed = false;
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
                                errors.push((
                                    display.clone(),
                                    d.line + 1,
                                    d.col + 1,
                                    d.message,
                                ));
                            }
                        }
                    }
                    hi_lsp::DiagnosticState::Failed { error, .. } => {
                        ui.status(&format!(
                            "fast check · LSP failed for {}: {error}",
                            path_display(runtime.root(), &path)
                        ));
                    }
                    hi_lsp::DiagnosticState::Unavailable { .. } => {}
                }
            }
            if !errors.is_empty() {
                report.lsp_errors = errors.len() as u32;
                let text = format_lsp_error_feedback(&errors);
                ui.status(&text);
                report.failures.push(text);
                // LSP already found compile-level issues — skip cargo this batch.
                return report;
            }
            lsp_checked_clean = saw_confirmed;
        }
    }

    // Invalidate seals for any language package touched this batch.
    let touched = affected_any_package_dirs(runtime.root(), changed_paths);
    state.invalidate_packages(&touched);

    // Tier 2a: cargo check when LSP is clean or unavailable and Rust files changed.
    let should_cargo = !rust_paths.is_empty() && (lsp_checked_clean || lsp_unavailable);
    let mut checks_ok_for_tests = !should_cargo; // non-Rust batches don't need cargo first
    if should_cargo {
        ui.status("fast check · cargo check (affected packages)…");
        let outcome = run_affected_cargo_checks(
            runtime.root(),
            changed_paths,
            &mut state.checked_packages,
        )
        .await;
        report.cargo_ran = matches!(
            outcome,
            CargoCommandOutcome::Passed { .. } | CargoCommandOutcome::Failed { .. }
        );
        if let Some(status) = outcome.ui_status() {
            if !matches!(outcome, CargoCommandOutcome::Passed { .. }) {
                ui.status(&status);
            }
        }
        if let Some(failure) = outcome.failure_message() {
            report.cargo_failed = true;
            if let CargoCommandOutcome::Failed { package, .. } = &outcome {
                state.checked_packages.remove(package);
                state.sealed_checks.remove(package);
            }
            report.failures.push(failure);
            return report;
        }
        let ledger_revision = runtime.ledger().revision();
        if let CargoCommandOutcome::Passed { packages, .. } = &outcome {
            state.seal_checks_at(packages, ledger_revision);
            checks_ok_for_tests = true;
        } else if matches!(outcome, CargoCommandOutcome::Skipped) {
            checks_ok_for_tests = true;
        }
    }

    // Tier 2b: polyglot typecheck/build/lint (tsc / go build / ruff) — always when
    // those languages changed (not only test-gated). Seals share check namespace.
    ui.status("fast check · package checks (tsc/go/ruff)…");
    let poly_check = run_affected_polyglot_checks(
        runtime.root(),
        changed_paths,
        &mut state.checked_packages,
    )
    .await;
    report.cargo_ran |= matches!(
        poly_check,
        CargoCommandOutcome::Passed { .. } | CargoCommandOutcome::Failed { .. }
    );
    if let Some(status) = poly_check.ui_status() {
        if !matches!(poly_check, CargoCommandOutcome::Passed { .. }) {
            ui.status(&status);
        }
    }
    if let Some(failure) = poly_check.failure_message() {
        report.cargo_failed = true;
        if let CargoCommandOutcome::Failed { package, .. } = &poly_check {
            state.checked_packages.remove(package);
            state.sealed_checks.remove(package);
        }
        report.failures.push(failure);
        return report;
    }
    if let CargoCommandOutcome::Passed { packages, .. } = &poly_check {
        let revision = runtime.ledger().revision();
        state.seal_checks_at(packages, revision);
        checks_ok_for_tests = true;
    }
    // Skipped polyglot checks leave checks_ok_for_tests as set by cargo tier.

    // Tier 3: package-local tests when the task is test-gated.
    if !options.run_tests || !checks_ok_for_tests {
        return report;
    }

    // Rust tests (after green check).
    if !rust_paths.is_empty() {
        ui.status("fast check · cargo test (affected packages)…");
        let test_outcome = run_affected_cargo_tests(
            runtime.root(),
            changed_paths,
            &mut state.tested_packages,
        )
        .await;
        report.tests_ran |= matches!(
            test_outcome,
            CargoCommandOutcome::Passed { .. } | CargoCommandOutcome::Failed { .. }
        );
        if let Some(status) = test_outcome.ui_status() {
            if !matches!(test_outcome, CargoCommandOutcome::Passed { .. }) {
                ui.status(&status);
            }
        }
        if let Some(failure) = test_outcome.failure_message() {
            report.tests_failed = true;
            if let CargoCommandOutcome::Failed { package, .. } = &test_outcome {
                state.tested_packages.remove(package);
                state.sealed_tests.remove(package);
            }
            report.failures.push(failure);
            return report;
        }
        if let CargoCommandOutcome::Passed { packages, .. } = &test_outcome {
            let revision = runtime.ledger().revision();
            state.seal_tests_at(packages, revision);
        }
    }

    // Polyglot package tests (pytest / npm test / go test).
    ui.status("fast check · package tests (py/js/go)…");
    let poly_outcome = run_affected_polyglot_tests(
        runtime.root(),
        changed_paths,
        &mut state.tested_packages,
    )
    .await;
    report.tests_ran |= matches!(
        poly_outcome,
        CargoCommandOutcome::Passed { .. } | CargoCommandOutcome::Failed { .. }
    );
    if let Some(status) = poly_outcome.ui_status() {
        if !matches!(poly_outcome, CargoCommandOutcome::Passed { .. }) {
            ui.status(&status);
        }
    }
    if let Some(failure) = poly_outcome.failure_message() {
        report.tests_failed = true;
        if let CargoCommandOutcome::Failed { package, .. } = &poly_outcome {
            state.tested_packages.remove(package);
            state.sealed_tests.remove(package);
        }
        report.failures.push(failure);
        return report;
    }
    if let CargoCommandOutcome::Passed { packages, .. } = &poly_outcome {
        let revision = runtime.ledger().revision();
        state.seal_tests_at(packages, revision);
    }
    report
}

fn path_display(root: &std::path::Path, path: &std::path::Path) -> String {
    path.strip_prefix(root)
        .unwrap_or(path)
        .to_string_lossy()
        .replace('\\', "/")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn report_combines_failures() {
        let mut report = FastFeedbackReport::default();
        report.failures.push("a".into());
        report.failures.push("b".into());
        assert_eq!(report.combined_failure().as_deref(), Some("a\n\nb"));
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
}
