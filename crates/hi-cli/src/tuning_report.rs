//! Tuning-signal sweep over recent transcripts and journals.
//!
//! `hi metrics` prints this after the orchestration dashboard: the evidence to
//! read when deciding what to tune next — sessions where the repair loop
//! thrashed ("no progress" fired), verification failures the digest parser
//! failed to structure (parser gaps), how often impact notes and escalations
//! actually fire, and the verified-merge rate. Deterministic: it only greps
//! artifacts hi itself wrote.

use std::path::{Path, PathBuf};

/// Most recent transcripts swept.
const MAX_SESSIONS: usize = 20;
/// Transcripts above this are skipped (a runaway session isn't tuning signal).
const MAX_SESSION_BYTES: u64 = 20 * 1024 * 1024;

#[derive(Debug, Default)]
pub(crate) struct TuningSignals {
    pub sessions_swept: usize,
    /// End-of-turn verification failures fed back to the model.
    pub verify_failures: usize,
    /// Of those, how many carried structure: the `── failure digest ──`
    /// banner or a parsed `Likely cause (verify and fix first):` section.
    pub digested_failures: usize,
    /// Repair rounds that made no progress (same failure set persisted).
    pub no_progress: usize,
    /// Repair rounds that regressed (more failures than the previous round).
    pub regressions: usize,
    /// Signature-impact notes injected after definition edits.
    pub impact_notes: usize,
    /// Transcripts where "no progress" fired — the ones worth reading.
    pub thrashing_sessions: Vec<PathBuf>,
    /// Verified delegate merges journaled under `<state-root>/learning/`.
    pub verified_merges: usize,
}

pub(crate) fn sweep(sessions_dir: &Path, state_root: &Path) -> TuningSignals {
    let mut signals = TuningSignals::default();

    let mut transcripts: Vec<(std::time::SystemTime, PathBuf)> = std::fs::read_dir(sessions_dir)
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|entry| {
            let path = entry.path();
            if path.extension().is_none_or(|ext| ext != "jsonl") {
                return None;
            }
            let meta = entry.metadata().ok()?;
            if meta.len() > MAX_SESSION_BYTES {
                return None;
            }
            Some((meta.modified().ok()?, path))
        })
        .collect();
    transcripts.sort_by_key(|entry| std::cmp::Reverse(entry.0));
    transcripts.truncate(MAX_SESSIONS);

    for (_, path) in &transcripts {
        let Ok(text) = std::fs::read_to_string(path) else {
            continue;
        };
        signals.sessions_swept += 1;
        // Markers count only inside user-role messages — the feedback hi
        // itself injected. Whole-file substring matching lied in hi-on-hi
        // sessions: reading feedback source or a test fixture puts every
        // marker into a ToolResult, and the sweep counted those as live
        // repair-loop events (6 "unstructured verify failures" that were
        // mostly the agent reading its own code).
        let mut no_progress = 0usize;
        for line in text.lines() {
            if !line.contains("Verification stage `")
                && !line.contains("No progress since the previous repair attempt")
                && !line.contains("— the last change introduced new breakage")
                && !line.contains("signature impact:")
            {
                continue;
            }
            let Ok(message) = serde_json::from_str::<serde_json::Value>(line) else {
                continue;
            };
            let is_user = message
                .get("role")
                .and_then(|role| role.as_str())
                .is_some_and(|role| role.eq_ignore_ascii_case("user"));
            if !is_user {
                continue;
            }
            let content = message
                .get("content")
                .map(|content| content.to_string())
                .unwrap_or_default();
            if content.contains("Verification stage `") {
                signals.verify_failures += 1;
                if content.contains("── failure digest ──")
                    || content.contains("Likely cause (verify and fix first):")
                {
                    signals.digested_failures += 1;
                }
            }
            if content.contains("No progress since the previous repair attempt") {
                no_progress += 1;
            }
            if content.contains("— the last change introduced new breakage") {
                signals.regressions += 1;
            }
            if content.contains("signature impact:") {
                signals.impact_notes += 1;
            }
        }
        signals.no_progress += no_progress;
        if no_progress > 0 {
            signals.thrashing_sessions.push(path.clone());
        }
    }

    signals.verified_merges =
        std::fs::read_to_string(state_root.join("learning").join("verified-merges.jsonl"))
            .map(|text| text.lines().filter(|line| !line.trim().is_empty()).count())
            .unwrap_or(0);

    signals
}

pub(crate) fn print_tuning_signals(sessions_dir: &Path, state_root: &Path) {
    let signals = sweep(sessions_dir, state_root);
    if signals.sessions_swept == 0 {
        return;
    }
    println!(
        "tuning signals (last {} session(s)):",
        signals.sessions_swept
    );
    if signals.verify_failures > 0 {
        let gap = signals
            .verify_failures
            .saturating_sub(signals.digested_failures);
        let gap_note = if gap > 0 {
            format!(" — {gap} unstructured (digest parser gap: read those stage outputs)")
        } else {
            String::new()
        };
        println!(
            "  verify failures fed back: {} · digested {}{gap_note}",
            signals.verify_failures, signals.digested_failures
        );
    }
    if signals.no_progress > 0 {
        println!(
            "  repair thrashing: \"no progress\" fired {}× — read these transcripts:",
            signals.no_progress
        );
        for path in signals.thrashing_sessions.iter().take(5) {
            println!("    {}", path.display());
        }
    }
    if signals.regressions > 0 {
        println!(
            "  repair regressions flagged: {} (a repair introduced new breakage)",
            signals.regressions
        );
    }
    if signals.impact_notes > 0 {
        println!(
            "  signature-impact notes injected: {}",
            signals.impact_notes
        );
    }
    println!(
        "  verified merges journaled: {} ({})",
        signals.verified_merges,
        state_root.join("learning/verified-merges.jsonl").display()
    );
    if signals.verify_failures == 0 && signals.no_progress == 0 {
        println!("  no repair-loop activity in recent sessions — nothing to tune yet");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sweep_counts_markers_and_flags_thrashing_transcripts() {
        let base = std::env::temp_dir().join(format!("hi-tuning-{}", std::process::id()));
        let sessions = base.join("sessions");
        let state = base.join("state");
        std::fs::create_dir_all(&sessions).unwrap();
        std::fs::create_dir_all(state.join("learning")).unwrap();
        std::fs::write(
            sessions.join("a.jsonl"),
            r#"{"role":"user","content":"Verification stage `check` failed. ── failure digest ── No progress since the previous repair attempt"}"#,
        )
        .unwrap();
        std::fs::write(
            sessions.join("b.jsonl"),
            r#"{"role":"user","content":"Verification stage `test` failed with raw output only. signature impact: `f` is referenced"}"#,
        )
        .unwrap();
        std::fs::write(sessions.join("not-a-transcript.txt"), "ignored").unwrap();
        // The hi-on-hi trap: every marker inside a ToolResult (the agent
        // reading its own feedback source) must count for nothing, while a
        // real nudge whose structure is the `Likely cause` section (not the
        // digest banner) still counts as digested.
        std::fs::write(
            sessions.join("c.jsonl"),
            concat!(
                r#"{"role":"Tool","content":[{"ToolResult":{"call_id":"1","output":"let t = \"Verification stage `x` failed\"; // No progress since the previous repair attempt — the last change introduced new breakage. signature impact: doc"}}]}"#,
                "\n",
                r#"{"role":"User","content":[{"Text":"[hi:nudge:verify]\nVerification stage `affected-test` failed (`cargo test`).\n\nLikely cause (verify and fix first):\n- [test] src/tests/goal.rs:1294 — panicked"}]}"#,
                "\n",
            ),
        )
        .unwrap();
        std::fs::write(
            state.join("learning/verified-merges.jsonl"),
            "{\"task\":\"a\"}\n{\"task\":\"b\"}\n",
        )
        .unwrap();

        let signals = sweep(&sessions, &state);
        assert_eq!(signals.sessions_swept, 3);
        assert_eq!(signals.verify_failures, 3, "tool-result markers excluded");
        assert_eq!(
            signals.digested_failures, 2,
            "banner and likely-cause both count; raw output does not"
        );
        assert_eq!(signals.no_progress, 1, "code mention is not a stall");
        assert_eq!(signals.regressions, 0);
        assert_eq!(signals.impact_notes, 1);
        assert_eq!(signals.thrashing_sessions.len(), 1);
        assert!(signals.thrashing_sessions[0].ends_with("a.jsonl"));
        assert_eq!(signals.verified_merges, 2);
        let _ = std::fs::remove_dir_all(&base);
    }
}
