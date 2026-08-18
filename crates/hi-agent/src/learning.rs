//! Append-only learning ledgers under `<state-root>/learning/`.
//!
//! `findings.jsonl` is the automatic intake for harness post-mortems: every
//! turn that ends badly (stalled, verification failed, infrastructure
//! failure) appends one evidence record pointing at what happened, so `hi
//! metrics` can surface failure patterns instead of someone re-deriving them
//! from raw session transcripts by hand — which is how every defect found in
//! live runs had to be recovered before this existed.

use std::path::Path;

use crate::{TurnOutcome, TurnStatus, TurnStopReason};

/// One bad-turn finding. Serialized as a single JSONL line.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct Finding {
    /// Unix seconds when the turn settled (correlates with session mtimes).
    pub ts: u64,
    /// Durable session id (transcript file stem) when the turn ran in a
    /// persisted session — lets `/synth-evals` point at the exact transcript
    /// instead of fuzzy-matching by mtime. Absent on pre-pointer records and
    /// ephemeral (unpersisted) runs.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    /// 0-based turn ordinal within the process run that appended this record.
    /// Resets when a session is resumed; `ts` + `session_id` stay the durable
    /// key, this narrows the search inside one run.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub turn: Option<u32>,
    pub status: TurnStatus,
    pub stop_reason: TurnStopReason,
    pub verification: crate::VerificationStatus,
    pub review: crate::ReviewStatus,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub review_unavailable_reason: Option<String>,
    /// Last stall reason the progress tracker recorded, when there was one.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub last_stall_reason: String,
    pub changed_files: usize,
    pub model: String,
    /// Failure shape of the steering hint that was in the session context
    /// when this turn ran, if one was — the raw material for judging whether
    /// hints help: a shape that keeps recurring under its own hint is a hint
    /// worth deleting.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hint_active: Option<String>,
    /// Compact tool-timeline shape (`write_overwrite`, `root_cargo_test`, …).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub failure_shape: Option<String>,
}

/// Whether a turn outcome warrants a finding record. Completed-and-verified
/// turns are the healthy path; everything that ends incomplete, failed, or
/// with failed verification is post-mortem material.
pub fn outcome_warrants_finding(outcome: &TurnOutcome) -> bool {
    matches!(outcome.status, TurnStatus::Incomplete | TurnStatus::Failed)
        || matches!(
            outcome.stop_reason,
            TurnStopReason::Stalled
                | TurnStopReason::VerificationFailed
                | TurnStopReason::InfrastructureFailure
        )
}

/// Append one finding line. Best-effort: a diagnostics write must never fail
/// a turn that already settled.
pub fn append_finding(state_root: &Path, finding: &Finding) {
    let dir = state_root.join("learning");
    if std::fs::create_dir_all(&dir).is_err() {
        return;
    }
    let Ok(line) = serde_json::to_string(finding) else {
        return;
    };
    let path = dir.join("findings.jsonl");
    if let Ok(mut file) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
    {
        use std::io::Write;
        let _ = writeln!(file, "{line}");
    }
}

/// Compact report of the learning ledgers for `/metrics` in a session:
/// findings by stop reason plus current interventions (latest record per
/// name). The census lives in `hi metrics` (it needs the sessions dir).
pub fn render_report(state_root: &Path) -> String {
    let mut out = String::new();
    let learning = state_root.join("learning");
    if let Some(raw) = read_ledger_capped(&learning.join("findings.jsonl")) {
        let mut by_reason: std::collections::BTreeMap<String, usize> = Default::default();
        let mut total = 0usize;
        for line in raw.lines() {
            if let Ok(finding) = serde_json::from_str::<Finding>(line) {
                *by_reason
                    .entry(format!("{:?}", finding.stop_reason))
                    .or_default() += 1;
                total += 1;
            }
        }
        if total > 0 {
            let parts: Vec<String> = by_reason
                .iter()
                .map(|(reason, count)| format!("{count} {reason}"))
                .collect();
            out.push_str(&format!(
                "findings: {total} bad turn(s) — {}\n",
                parts.join(" · ")
            ));
        }
    }
    if let Some(raw) = read_ledger_capped(&learning.join("interventions.jsonl")) {
        let mut latest: std::collections::BTreeMap<String, serde_json::Value> = Default::default();
        for line in raw.lines() {
            if let Ok(value) = serde_json::from_str::<serde_json::Value>(line)
                && let Some(name) = value.get("name").and_then(|n| n.as_str())
            {
                latest.insert(name.to_string(), value);
            }
        }
        for (name, value) in latest {
            let state = value
                .get("evidence_state")
                .and_then(|s| s.as_str())
                .unwrap_or("?");
            out.push_str(&format!("  [{state}] {name}\n"));
        }
    }
    if out.is_empty() {
        out.push_str("no findings or interventions recorded for this project\n");
    }
    out.push_str("(full report incl. tool census: `hi metrics`)");
    out
}

/// A steering hint derived from recent findings, plus the failure shape it
/// targets — the shape is stamped onto findings recorded while the hint is
/// active, so `hi metrics` can show recurrence-under-hint.
pub struct ContextHint {
    /// Debug-formatted stop reason the hint targets (e.g. "Stalled").
    pub shape: String,
    /// The one-line hint text injected into the session context.
    pub text: String,
}

/// One-line steering hint from recent findings, for the session context
/// block: when the same failure shape ended ≥2 turns in the last 7 days,
/// tell the model so it can adapt (e.g. prefer package-local checks when
/// verification keeps timing out). Returns None when there is no pattern —
/// context space is only spent on evidence.
/// Findings are append-only. Reading the whole file every turn would turn a
/// long-lived project ledger into a startup/token tax.
pub(crate) const MAX_LEDGER_READ_BYTES: usize = 64 * 1024;

pub(crate) fn read_ledger_capped(path: &Path) -> Option<String> {
    use std::io::Read;
    let mut file = std::fs::File::open(path).ok()?;
    let mut buf = vec![0u8; MAX_LEDGER_READ_BYTES];
    let n = file.read(&mut buf).ok()?;
    buf.truncate(n);
    Some(String::from_utf8_lossy(&buf).into_owned())
}

pub fn context_hint(state_root: &Path) -> Option<ContextHint> {
    let raw = read_ledger_capped(&state_root.join("learning").join("findings.jsonl"))?;
    let cutoff = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .ok()?
        .as_secs()
        .saturating_sub(7 * 24 * 3600);
    let mut by_shape: std::collections::BTreeMap<String, usize> = Default::default();
    let mut by_reason: std::collections::BTreeMap<String, usize> = Default::default();
    for line in raw.lines() {
        if let Ok(finding) = serde_json::from_str::<Finding>(line)
            && finding.ts >= cutoff
        {
            *by_reason
                .entry(format!("{:?}", finding.stop_reason))
                .or_default() += 1;
            if let Some(shape) = finding.failure_shape.filter(|s| !s.is_empty()) {
                *by_shape.entry(shape).or_default() += 1;
            }
        }
    }
    if let Some((shape, count)) = by_shape.into_iter().max_by_key(|(_, count)| *count)
        && count >= 2
    {
        let advice = shape_advice(&shape);
        return Some(ContextHint {
            text: format!(
                "Recent harness findings: {count} turn(s) in the last 7 days ended {shape} — {advice}."
            ),
            shape,
        });
    }
    let (reason, count) = by_reason.into_iter().max_by_key(|(_, count)| *count)?;
    if count < 2 {
        return None;
    }
    let advice = match reason.as_str() {
        "VerificationFailed" => "run the affected package-local check yourself before finishing",
        "Stalled" => "act on tool evidence immediately instead of re-polling or re-reading",
        "InfrastructureFailure" => {
            "verification infra has been failing; prefer narrow package-local checks"
        }
        _ => "finish with a concrete verified result",
    };
    Some(ContextHint {
        text: format!(
            "Recent harness findings: {count} turn(s) in the last 7 days ended {reason} — {advice}."
        ),
        shape: reason,
    })
}

fn shape_advice(shape: &str) -> &'static str {
    match shape {
        "write_overwrite" => {
            "use edit/multi_edit/apply_patch instead of rewriting files with write"
        }
        "root_cargo_test" => {
            "run package-local `cargo test -p <crate>` instead of workspace-root cargo test"
        }
        "repeat_inspect" => "act on tool evidence immediately instead of re-reading or re-polling",
        "no_validation" => "run a targeted syntax/build/test command after edits before finishing",
        _ => "finish with a concrete verified result",
    }
}

/// Classify a bad turn's tool timeline into a compact failure shape.
pub fn tool_failure_shape(timeline: &[crate::ToolCallEntry]) -> Option<String> {
    if timeline.iter().any(|entry| {
        entry.tool == "write"
            && entry.effects.mutation_applied
            && entry
                .effects
                .file_changes
                .iter()
                .any(|change| change.kind == hi_tools::FileChangeKind::Modify)
    }) {
        return Some("write_overwrite".into());
    }
    if timeline.iter().any(|entry| {
        entry.tool == "bash" && entry.command.as_deref().is_some_and(is_root_cargo_test)
    }) {
        return Some("root_cargo_test".into());
    }
    let inspect_repeats = timeline
        .iter()
        .filter(|entry| {
            entry.progress_kind == "none"
                && (entry.progress_reason.contains("repeated")
                    || entry.progress_reason.contains("inspection"))
        })
        .count();
    if inspect_repeats >= 2 {
        return Some("repeat_inspect".into());
    }
    let mutated = timeline.iter().any(|entry| entry.effects.mutation_applied);
    let validated = timeline.iter().any(|entry| {
        entry.tool == "bash"
            && entry.status == hi_tools::ToolStatus::Succeeded
            && entry.command.as_deref().is_some_and(|command| {
                let lower = command.to_ascii_lowercase();
                lower.contains("test")
                    || lower.contains("check")
                    || lower.contains("pytest")
                    || lower.contains("cargo")
            })
    });
    if mutated && !validated {
        return Some("no_validation".into());
    }
    None
}

fn is_root_cargo_test(command: &str) -> bool {
    let lower = command.to_ascii_lowercase();
    if !cargo_test_invocation(&lower) {
        return false;
    }
    if cargo_test_is_package_scoped(&lower) {
        return false;
    }
    !directory_changed_before_cargo_test(&lower)
}

fn cargo_test_invocation(lower: &str) -> bool {
    cargo_subcommand_tokens(lower).any(|sub| sub == "test" || sub == "t")
}

fn cargo_test_is_package_scoped(lower: &str) -> bool {
    lower.contains(" -p ")
        || lower.contains(" --package ")
        || lower.contains("--manifest-path")
        || lower.contains("-p=")
        || lower.contains("--package=")
}

fn directory_changed_before_cargo_test(lower: &str) -> bool {
    let Some(idx) = lower.find("cargo") else {
        return false;
    };
    let prefix = &lower[..idx];
    prefix.contains("cd ") || prefix.contains("cd\t") || prefix.contains("pushd ")
}

fn cargo_subcommand_tokens(lower: &str) -> impl Iterator<Item = &str> {
    let mut tokens = tokenize_shell(lower).into_iter();
    std::iter::from_fn(move || {
        while let Some(token) = tokens.next() {
            if token == "cargo" {
                return tokens.next();
            }
        }
        None
    })
}

fn tokenize_shell(input: &str) -> Vec<&str> {
    input
        .split(|ch: char| ch.is_whitespace() || matches!(ch, ';' | '|' | '&' | '(' | ')' | '`'))
        .filter(|token| !token.is_empty())
        .collect()
}

/// Assemble the `/synth-evals` follow-up turn: unprocessed findings past the
/// cursor, grouped by failure signature, as a prompt instructing the session
/// to draft regression evals into `pending-evals/`. The cursor advances at
/// dispatch — each finding gets one synthesis shot. Drafts are a review
/// queue; admission into real suites stays with the human + `bench
/// --validate` (fail-before / pass-after / immutable oracle).
pub fn synth_evals_prompt(state_root: &Path) -> Option<(usize, String)> {
    let learning = state_root.join("learning");
    let raw = read_ledger_capped(&learning.join("findings.jsonl"))?;
    let cursor_path = learning.join("findings.cursor");
    let done: usize = std::fs::read_to_string(&cursor_path)
        .ok()
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(0);
    let all: Vec<Finding> = raw
        .lines()
        .filter_map(|line| serde_json::from_str(line).ok())
        .collect();
    if all.len() <= done {
        return None;
    }
    // Per signature: occurrence count plus up to 3 transcript pointers
    // (session id + turn ordinal) so the drafting turn reads the exact
    // transcripts instead of fuzzy-matching session files by mtime.
    let mut groups: std::collections::BTreeMap<String, (usize, Vec<String>)> = Default::default();
    for finding in &all[done..] {
        let detail = if finding.last_stall_reason.is_empty() {
            finding.review_unavailable_reason.as_deref().unwrap_or("-")
        } else {
            &finding.last_stall_reason
        };
        let entry = groups
            .entry(format!("{:?} / {detail}", finding.stop_reason))
            .or_default();
        entry.0 += 1;
        if let Some(id) = &finding.session_id
            && entry.1.len() < 3
        {
            entry.1.push(match finding.turn {
                Some(turn) => format!("{id} turn {turn}"),
                None => id.clone(),
            });
        }
    }
    let listing: String = groups
        .iter()
        .map(|(sig, (n, sessions))| {
            if sessions.is_empty() {
                format!("- {n}x {sig}\n")
            } else {
                format!("- {n}x {sig} [transcripts: {}]\n", sessions.join(", "))
            }
        })
        .collect();
    let _ = std::fs::write(&cursor_path, all.len().to_string());
    let prompt = format!(
        "Synthesize regression evals from this project's recent harness failures.\n\
         Failure signatures (from {}/learning/findings.jsonl, entries {}..{}):\n{listing}\
         For each signature: read the turn in the transcripts listed with it (session \
         ids are file stems in the sessions dir; turn ordinals are 0-based within one \
         process run), or the newest session transcripts when none are listed, \
         distill the invariant that broke, and write ONE deterministic test draft \
         into pending-evals/ in this repository, named after the signature. Draft \
         contract (checked by `hi bench validate`): a self-contained cargo integration \
         test file that starts with `//! target-crate: <dir under crates/>` (default \
         hi-agent) and `//! pre-fix: <one line naming the behavior it fails against>`, \
         using only that crate's public API; prefer fake-provider/Canned-turn \
         determinism. Each draft must fail against the pre-fix behavior and pass once \
         the fix is in. Do NOT add anything to the real eval suites or CI: \
         pending-evals/ is a review queue; admission happens via `hi bench validate \
         <draft> --before <pre-fix rev>` plus human review.",
        state_root.display(),
        done,
        all.len(),
    );
    Some((groups.len(), prompt))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ReviewStatus, VerificationStatus};

    fn outcome(status: TurnStatus, stop_reason: TurnStopReason) -> TurnOutcome {
        TurnOutcome {
            status,
            verification: VerificationStatus::Passed,
            review: ReviewStatus::NotRequired,
            stop_reason,
            changed_files: Vec::new(),
            verified_workspace_revision: None,
            effective_route: crate::EffectiveModelRoute {
                provider: None,
                model: "m".into(),
            },
            review_same_model: false,
            leftover: None,
            plan_leftover: None,
        }
    }

    #[test]
    fn context_hint_does_not_load_an_unbounded_findings_file() {
        let dir = std::env::temp_dir().join(format!(
            "hi-learning-huge-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("learning")).unwrap();
        let line = r#"{"ts":1,"status":"Incomplete","stop_reason":"Stalled","verification":"Passed","review":"NotRequired","changed_files":0,"model":"m","failure_shape":"repeat_inspect"}"#;
        let mut body = String::new();
        while body.len() < MAX_LEDGER_READ_BYTES + 8_000 {
            body.push_str(line);
            body.push('\n');
        }
        std::fs::write(dir.join("learning").join("findings.jsonl"), &body).unwrap();
        let raw = read_ledger_capped(&dir.join("learning").join("findings.jsonl")).unwrap();
        assert!(
            raw.len() <= MAX_LEDGER_READ_BYTES,
            "findings read must be capped: {}",
            raw.len()
        );
        let _ = context_hint(&dir);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn render_report_does_not_load_an_unbounded_interventions_file() {
        let dir = std::env::temp_dir().join(format!(
            "hi-learning-interventions-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("learning")).unwrap();
        let line = r#"{"name":"cap-test","evidence_state":"shipped"}"#;
        let mut body = String::new();
        while body.len() < MAX_LEDGER_READ_BYTES + 8_000 {
            body.push_str(line);
            body.push('\n');
        }
        std::fs::write(dir.join("learning").join("interventions.jsonl"), &body).unwrap();
        let report = render_report(&dir);
        assert!(
            report.len() < 4_096,
            "metrics report must stay short: {}",
            report.len()
        );
        assert!(report.contains("cap-test"), "{report}");
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn only_bad_outcomes_warrant_findings() {
        assert!(!outcome_warrants_finding(&outcome(
            TurnStatus::Completed,
            TurnStopReason::Completed
        )));
        assert!(outcome_warrants_finding(&outcome(
            TurnStatus::Incomplete,
            TurnStopReason::Stalled
        )));
        assert!(outcome_warrants_finding(&outcome(
            TurnStatus::Failed,
            TurnStopReason::InfrastructureFailure
        )));
    }

    #[test]
    fn findings_append_as_parseable_jsonl() {
        let dir = std::env::temp_dir().join(format!("hi-learning-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let finding = Finding {
            ts: 1,
            session_id: Some("42-refactor".into()),
            turn: Some(5),
            status: TurnStatus::Incomplete,
            stop_reason: TurnStopReason::Stalled,
            verification: VerificationStatus::Passed,
            review: ReviewStatus::Unavailable,
            review_unavailable_reason: Some("provider timed out".into()),
            last_stall_reason: "repeated idempotent tool output".into(),
            changed_files: 3,
            model: "test-model".into(),
            hint_active: Some("Stalled".into()),
            failure_shape: None,
        };
        append_finding(&dir, &finding);
        append_finding(&dir, &finding);
        let raw = std::fs::read_to_string(dir.join("learning/findings.jsonl")).unwrap();
        let parsed: Vec<Finding> = raw
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed[0].stop_reason, TurnStopReason::Stalled);
        assert_eq!(parsed[0].session_id.as_deref(), Some("42-refactor"));
        assert_eq!(parsed[0].turn, Some(5));
        assert_eq!(parsed[0].hint_active.as_deref(), Some("Stalled"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn context_hint_needs_a_repeated_recent_shape_and_names_it() {
        let dir = std::env::temp_dir().join(format!("hi-hint-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("learning")).unwrap();
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let line = |ts: u64| {
            format!(
                r#"{{"ts":{ts},"status":"incomplete","stop_reason":"stalled","verification":"passed","review":"not_required","changed_files":0,"model":"m"}}"#
            )
        };
        // One recent finding plus one outside the 7-day window: no pattern.
        std::fs::write(
            dir.join("learning/findings.jsonl"),
            format!("{}\n{}\n", line(now - 60), line(now - 30 * 24 * 3600)),
        )
        .unwrap();
        assert!(context_hint(&dir).is_none(), "a single instance is noise");
        // A second recent one makes it a pattern; the hint names the shape.
        std::fs::write(
            dir.join("learning/findings.jsonl"),
            format!("{}\n{}\n", line(now - 60), line(now - 120)),
        )
        .unwrap();
        let hint = context_hint(&dir).expect("repeated shape steers");
        assert_eq!(hint.shape, "Stalled");
        assert!(
            hint.text.contains("Stalled"),
            "hint names the shape: {}",
            hint.text
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn synth_prompt_groups_signatures_and_advances_cursor() {
        let dir = std::env::temp_dir().join(format!("hi-synth-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("learning")).unwrap();
        let finding = |reason: &str| {
            format!(
                r#"{{"ts":1,"status":"incomplete","stop_reason":"stalled","verification":"passed","review":"not_required","last_stall_reason":"{reason}","changed_files":0,"model":"m"}}"#
            )
        };
        // Two pointer-less lines (the pre-pointer on-disk format — must still
        // parse) plus one carrying a session pointer.
        std::fs::write(
            dir.join("learning/findings.jsonl"),
            format!(
                "{}\n{}\n{}\n",
                finding("re-poll"),
                finding("re-poll"),
                r#"{"ts":1,"session_id":"7-fix-verify","turn":4,"status":"incomplete","stop_reason":"stalled","verification":"passed","review":"not_required","last_stall_reason":"re-read","changed_files":0,"model":"m"}"#,
            ),
        )
        .unwrap();
        let (signatures, prompt) = synth_evals_prompt(&dir).expect("fresh findings");
        assert_eq!(signatures, 2, "two distinct stall signatures");
        assert!(
            prompt.contains("2x"),
            "duplicate signature counted: {prompt}"
        );
        assert!(
            prompt.contains("[transcripts: 7-fix-verify turn 4]"),
            "session pointer listed with its signature: {prompt}"
        );
        assert!(prompt.contains("pending-evals/"));
        // Cursor advanced at dispatch: nothing left to synthesize.
        assert!(synth_evals_prompt(&dir).is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    fn bash_entry(command: &str) -> crate::ToolCallEntry {
        crate::ToolCallEntry {
            tool: "bash".into(),
            path: String::new(),
            duration_ms: 1,
            queue_delay_ms: 0,
            completion_index: 0,
            status: hi_tools::ToolStatus::Succeeded,
            background: None,
            process: None,
            effects: hi_tools::ToolEffects::default(),
            truncation: hi_tools::TruncationState::Complete,
            error: false,
            progress_kind: "meaningful".into(),
            progress_reason: "ok".into(),
            normalized_signature: None,
            command: Some(command.into()),
            arg_chars: 0,
            result_chars: 0,
            truncated: false,
            kind: "shell".into(),
        }
    }

    #[test]
    fn tool_failure_shape_detects_root_cargo_test() {
        assert_eq!(
            tool_failure_shape(&[bash_entry("cargo test --quiet")]).as_deref(),
            Some("root_cargo_test")
        );
        assert!(tool_failure_shape(&[bash_entry("cargo test -p hi-agent --quiet")]).is_none());
        assert!(
            tool_failure_shape(&[bash_entry("cargo test --manifest-path crates/x/Cargo.toml")])
                .is_none()
        );
        assert!(tool_failure_shape(&[bash_entry("cd pkg && cargo test")]).is_none());
        assert_eq!(
            tool_failure_shape(&[bash_entry("cargo t")]).as_deref(),
            Some("root_cargo_test")
        );
    }

    #[test]
    fn context_hint_prefers_repeated_failure_shape() {
        let dir = std::env::temp_dir().join(format!("hi-hint-shape-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("learning")).unwrap();
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let line = |ts: u64| {
            format!(
                r#"{{"ts":{ts},"status":"incomplete","stop_reason":"stalled","verification":"passed","review":"not_required","changed_files":0,"model":"m","failure_shape":"root_cargo_test"}}"#
            )
        };
        std::fs::write(
            dir.join("learning/findings.jsonl"),
            format!("{}\n{}\n", line(now - 60), line(now - 120)),
        )
        .unwrap();
        let hint = context_hint(&dir).expect("shape pattern");
        assert_eq!(hint.shape, "root_cargo_test");
        assert!(hint.text.contains("cargo test -p"), "{}", hint.text);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
