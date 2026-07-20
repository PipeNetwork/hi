//! Mid-turn fast feedback after a mutating tool batch.
//!
//! Tier 1: LSP diagnostics on changed Rust files.
//! Tier 2: affected-package `cargo check` when LSP is clean or unavailable.
//! Tier 3: affected-package `cargo test` when the task is test-gated and check is green.
//! Failures are appended into the transcript tool results so the model sees them
//! before the next reasoning step — not only as a UI status line.

use std::collections::BTreeSet;
use std::path::PathBuf;

use hi_tools::{
    CargoCommandOutcome, affected_cargo_package_dirs, format_lsp_error_feedback,
    run_affected_cargo_checks, run_affected_cargo_tests, rust_source_paths,
};

use crate::Ui;
use crate::workspace_runtime::WorkspaceRuntime;

/// Mutable turn-local state for fast feedback dedupe.
#[derive(Debug, Default)]
pub(crate) struct FastFeedbackState {
    /// Cargo packages already `cargo check`'d clean this turn (relative labels, or `"."`).
    pub checked_packages: BTreeSet<String>,
    /// Cargo packages already `cargo test`'d clean this turn.
    pub tested_packages: BTreeSet<String>,
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
    /// When true (task contract wants tests), run affected `cargo test` after a
    /// clean check. Never runs the full workspace suite.
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
    let mut lsp_checked_clean = false;
    let mut lsp_unavailable = true;

    if runtime.lsp_enabled() && !rust_paths.is_empty() {
        let lsp = runtime.lsp();
        if lsp.is_enabled().await {
            lsp_unavailable = false;
            let path_bufs = rust_paths
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

    // Tier 2: cargo check when LSP is clean or unavailable and Rust files changed.
    let should_cargo = !rust_paths.is_empty() && (lsp_checked_clean || lsp_unavailable);
    if !should_cargo {
        return report;
    }

    // A new mutation invalidates prior clean checks/tests for the packages it touches.
    let touched = affected_cargo_package_dirs(runtime.root(), &rust_paths);
    if touched.is_empty() {
        state.checked_packages.remove(".");
        state.tested_packages.remove(".");
    } else {
        for package in &touched {
            state.checked_packages.remove(package);
            state.tested_packages.remove(package);
        }
    }

    ui.status("fast check · cargo check (affected packages)…");
    let outcome =
        run_affected_cargo_checks(runtime.root(), changed_paths, &mut state.checked_packages).await;
    report.cargo_ran = matches!(
        outcome,
        CargoCommandOutcome::Passed { .. } | CargoCommandOutcome::Failed { .. }
    );
    if let Some(status) = outcome.ui_status() {
        // Only surface failures / skips; passes are silent.
        if !matches!(outcome, CargoCommandOutcome::Passed { .. }) {
            ui.status(&status);
        }
    }
    if let Some(failure) = outcome.failure_message() {
        report.cargo_failed = true;
        // A failed package stays out of the "clean" set so a later edit rechecks.
        if let CargoCommandOutcome::Failed { package, .. } = &outcome {
            state.checked_packages.remove(package);
        }
        report.failures.push(failure);
        return report;
    }

    // Tier 3: affected cargo test only when the task is test-gated and check is green.
    if !options.run_tests || !outcome.is_passed() {
        return report;
    }

    ui.status("fast check · cargo test (affected packages)…");
    let test_outcome =
        run_affected_cargo_tests(runtime.root(), changed_paths, &mut state.tested_packages).await;
    report.tests_ran = matches!(
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
        }
        report.failures.push(failure);
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
}
