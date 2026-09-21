//! What a finished (or paused) `/review` shows: the summary line, the
//! coverage/findings report, the status report, and the next step.
//!
//! Split from [`crate::review_drive`] (the state machine) so both stay under
//! the file-size ratchet.

use crate::review::{CoverageState, Finding, GitSource, ReviewVerdict};
use crate::review_drive::{ReviewDrive, ReviewPhase};

impl ReviewDrive {
    /// Blocking findings still open per the last verdict.
    pub fn open_blocking(&self) -> Vec<Finding> {
        self.last_verdict
            .as_ref()
            .map(ReviewVerdict::blocking_findings)
            .unwrap_or_default()
    }

    /// `(implemented, total)` coverage rows from the last verdict.
    pub fn coverage_counts(&self) -> (usize, usize) {
        let rows = self
            .last_verdict
            .as_ref()
            .map(|verdict| verdict.coverage.as_slice())
            .unwrap_or(&[]);
        let implemented = rows
            .iter()
            .filter(|row| row.state == CoverageState::Implemented)
            .count();
        (implemented, rows.len())
    }

    /// True when the last verdict exists and has no partial/missing rows.
    /// With no rows at all there was nothing to cover unless a real
    /// plan/spec was among the inputs (then the model skipped its items).
    pub fn coverage_complete(&self) -> bool {
        self.last_verdict.as_ref().is_some_and(|verdict| {
            if verdict.coverage.is_empty() {
                !self.inputs.has_spec()
            } else {
                verdict.coverage_complete()
            }
        })
    }

    /// What the user should do now, once the loop has ended. `None` when it
    /// ended clean against a real plan/spec: there is nothing left to point
    /// at. (`next_step` is the state machine's transition.)
    pub fn next_action(&self) -> Option<String> {
        let open = self.open_blocking().len();
        if open > 0 {
            return Some(if self.audit_only {
                format!("next: `/review` (without `audit`) fixes the {open} open P0/P1 finding(s)")
            } else {
                format!(
                    "next: {open} P0/P1 finding(s) are still open (above); fix them, then `/review` again"
                )
            });
        }
        if !self.inputs.has_spec() {
            let mut wider = "";
            let audited = match &self.inputs.git {
                Some(git) if matches!(git.source, GitSource::Uncommitted) => {
                    wider = "; `/review audit all` audits the whole repo in chunks";
                    "this audited the uncommitted changes only".to_string()
                }
                Some(_) => {
                    wider = "; `/review audit all` audits the whole repo in chunks";
                    "this audited the last commit only".to_string()
                }
                None if !self.inputs.chunks.is_empty() => {
                    "this audited the whole repo for defects".to_string()
                }
                None if self.inputs.files.is_empty() => "this audited for defects only".to_string(),
                None => format!("this audited {}, not a plan or spec", self.inputs.summary()),
            };
            return Some(format!(
                "next: {audited}; write plan.md (a `- [ ]` checklist of what should exist) or pass one for a coverage audit: `/review docs/spec.md [dir]`{wider}"
            ));
        }
        let (implemented, total) = self.coverage_counts();
        if total > implemented {
            return Some(format!(
                "next: {} plan/spec item(s) are missing or partial (above); build them, then `/review` again",
                total - implemented
            ));
        }
        None
    }

    pub(crate) fn done_summary(&self) -> String {
        let (implemented, total) = self.coverage_counts();
        let open = self.open_blocking().len();
        let minor = self
            .last_verdict
            .as_ref()
            .map(|verdict| {
                verdict
                    .findings
                    .iter()
                    .filter(|finding| !finding.severity.is_blocking())
                    .count()
            })
            .unwrap_or(0);
        let coverage = if total == 0 {
            match self.inputs.scope_summary() {
                Some(scope) => format!("{scope} audited, no plan/spec items"),
                None => "no plan/spec items".to_string(),
            }
        } else if self.coverage_complete() {
            format!("all {total} plan/spec items implemented")
        } else {
            format!("{implemented}/{total} plan/spec items implemented")
        };
        let defects = if open > 0 {
            format!("{open} P0/P1 defect(s) still open (audit only)")
        } else if self.pass > 0 {
            format!("no P0/P1 defects after {} fix pass(es)", self.pass)
        } else {
            "no P0/P1 defects".to_string()
        };
        let mut text = format!("spec review complete: {coverage} · {defects}");
        let gaps = self
            .last_verdict
            .as_ref()
            .map(ReviewVerdict::blocking_feature_gaps)
            .unwrap_or(0);
        if gaps > 0 {
            text.push_str(&format!(
                " · {gaps} P0/P1 row(s) are unchecked plan items, reported not fixed"
            ));
        }
        if minor > 0 {
            text.push_str(&format!(" · {minor} P2/P3 reported"));
        }
        text
    }

    /// Coverage matrix and findings from the last verdict, one line each.
    pub fn report_lines(&self) -> Vec<String> {
        let mut lines = Vec::new();
        let Some(verdict) = &self.last_verdict else {
            return lines;
        };
        if verdict.coverage.is_empty() {
            lines.push(if self.inputs.has_spec() {
                format!(
                    "coverage: no rows reported for {} (the model skipped its items)",
                    self.inputs.summary()
                )
            } else {
                "coverage: none (no plan/spec items; write plan.md for a coverage audit)"
                    .to_string()
            });
        } else {
            lines.push(format!(
                "coverage ({}):",
                if verdict.complete {
                    "complete"
                } else {
                    "incomplete"
                }
            ));
            for row in &verdict.coverage {
                let evidence = row
                    .evidence
                    .as_deref()
                    .map(|e| format!(" — {e}"))
                    .unwrap_or_default();
                lines.push(format!(
                    "  {:<11} {}{evidence}",
                    row.state.as_str(),
                    row.item
                ));
            }
        }
        if verdict.findings.is_empty() {
            lines.push("findings: none".into());
        } else {
            lines.push("findings:".into());
            for finding in &verdict.findings {
                let mut flags = String::new();
                if finding.feature_gap {
                    flags.push_str(" (unchecked plan item: reported, not fixed)");
                }
                if !finding.verified {
                    flags.push_str(" (cited file not found)");
                }
                lines.push(format!("  {}{flags}", finding.label()));
            }
        }
        if let Some(residual) = &verdict.residual {
            lines.push(format!("residual: {residual}"));
        }
        if let Some(note) = verdict.verdict_note() {
            lines.push(format!("note: {note}"));
        }
        if !self.audit_changed_files.is_empty() {
            lines.push(format!(
                "warning: an audit turn changed files (it is meant to be read-only; /undo restores that turn): {}",
                self.audit_changed_files.join(", ")
            ));
        }
        if matches!(self.phase, ReviewPhase::Done | ReviewPhase::Stopped)
            && let Some(next) = self.next_action()
        {
            lines.push(next);
        }
        lines
    }

    pub(crate) fn status_report(&self) -> Vec<String> {
        let mut lines = vec![self.status_line()];
        if let Some(notice) = &self.inputs.notice {
            lines.push(notice.clone());
        }
        lines.extend(self.report_lines());
        if !self.unverified_citations.is_empty() {
            lines.push(format!(
                "unverified citations: {}",
                self.unverified_citations.join(", ")
            ));
        }
        lines
    }
}
