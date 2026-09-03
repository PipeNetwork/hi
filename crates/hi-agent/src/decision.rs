//! A durable intra-session decision log.
//!
//! When the model commits to a key decision, it records it via the
//! `record_decision` tool. The log is injected into the system prompt each turn
//! (as a bounded view, oldest-first) so the model stays consistent across a
//! long session.
//! Unlike the conversation history, the decision log is **not** subject to
//! compaction summarization — it lives in the system message, which compaction
//! preserves verbatim. This is the *why* of intra-session choices, distinct
//! from cross-session memory and from the task plan (objectives).
//!
//! Distinct from [`crate::memory`]: memory is distilled at session end and
//! reloaded next session; the decision log is authoritative *within* a session
//! and survives the in-session compactions that would otherwise summarize away
//! the reasoning behind earlier decisions.

use serde::{Deserialize, Serialize};

/// One recorded decision: what was decided, why, and the files it bears on.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Decision {
    /// A short one-line title of the decision.
    pub summary: String,
    /// Why this choice — the constraint or tradeoff that drove it.
    pub rationale: String,
    /// Files the decision most affects (may be empty).
    #[serde(default)]
    pub files: Vec<String>,
}

/// Bound only the model-facing view. The authoritative log itself is exact and
/// never evicts decisions: persistence, retry, and resume must not silently
/// discard an early architectural constraint merely because a session ran for
/// a long time.
const MAX_PROMPT_DECISIONS: usize = 12;
const MAX_SUMMARY_CHARS: usize = 200;
const MAX_RATIONALE_CHARS: usize = 400;
const MAX_DECISION_FILES: usize = 8;
const MAX_DECISION_FILE_CHARS: usize = 160;

/// The durable decision log. Injected into the system prompt each turn;
/// compaction-immune (it's not part of the summarizable history).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DecisionLog {
    entries: Vec<Decision>,
}

impl DecisionLog {
    /// Rebuild a log from persisted entries, reusing `record` so duplicate
    /// summaries are normalized the same way as live updates.
    pub fn from_entries(entries: Vec<Decision>) -> Self {
        let mut log = Self::default();
        for entry in entries {
            log.record(entry);
        }
        log
    }

    /// Record a decision. If the summary duplicates an existing entry, the
    /// earlier one is replaced (the model is re-stating/refining a decision,
    /// not adding a duplicate).
    pub fn record(&mut self, decision: Decision) {
        // Replace an existing entry with the same summary rather than
        // accumulating duplicates.
        if let Some(existing) = self
            .entries
            .iter_mut()
            .find(|d| d.summary == decision.summary)
        {
            *existing = decision;
            return;
        }
        self.entries.push(decision);
    }

    /// The complete authoritative decisions, oldest-first, for persistence and
    /// state restoration. [`Self::prompt_section`] renders a bounded copy.
    pub fn entries(&self) -> &[Decision] {
        &self.entries
    }

    /// Whether the log is empty (skip prompt injection when so).
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Render the log as a system-prompt section, or `None` when empty.
    pub fn prompt_section(&self) -> Option<String> {
        if self.is_empty() {
            return None;
        }
        let mut out =
            String::from("\n\n[Key decisions this session — stay consistent with these]\n");
        let omitted = self.entries.len().saturating_sub(MAX_PROMPT_DECISIONS);
        if omitted > 0 {
            out.push_str(&format!(
                "[{omitted} earlier decisions remain retained in durable session state; showing the most recent {MAX_PROMPT_DECISIONS}.]\n"
            ));
        }
        for (i, d) in self
            .entries
            .iter()
            .skip(omitted)
            .cloned()
            .map(clip_decision)
            .enumerate()
        {
            out.push_str(&format!(
                "{}. {}\n   why: {}\n",
                i + 1,
                d.summary,
                d.rationale
            ));
            if !d.files.is_empty() {
                out.push_str(&format!("   files: {}\n", d.files.join(", ")));
            }
        }
        Some(out)
    }
}

fn clip_decision(mut decision: Decision) -> Decision {
    decision.summary = clip_chars(&decision.summary, MAX_SUMMARY_CHARS);
    decision.rationale = clip_chars(&decision.rationale, MAX_RATIONALE_CHARS);
    decision.files.truncate(MAX_DECISION_FILES);
    for file in &mut decision.files {
        *file = clip_chars(file, MAX_DECISION_FILE_CHARS);
    }
    decision
}

fn clip_chars(text: &str, max: usize) -> String {
    let text = text.trim();
    if text.chars().count() <= max {
        return text.to_string();
    }
    let clipped: String = text.chars().take(max.saturating_sub(1)).collect();
    format!("{clipped}…")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dec(summary: &str, rationale: &str) -> Decision {
        Decision {
            summary: summary.into(),
            rationale: rationale.into(),
            files: Vec::new(),
        }
    }

    #[test]
    fn record_keeps_more_than_the_prompt_view_capacity() {
        let mut log = DecisionLog::default();
        for i in 0..20 {
            log.record(dec(&format!("d{i}"), "r"));
        }
        assert_eq!(log.entries().len(), 20);
        assert_eq!(log.entries().first().unwrap().summary, "d0");
        assert_eq!(log.entries().last().unwrap().summary, "d19");

        let resumed = DecisionLog::from_entries(log.entries().to_vec());
        assert_eq!(resumed.entries(), log.entries());

        let section = resumed.prompt_section().unwrap();
        assert!(section.contains("8 earlier decisions remain retained"));
        assert!(
            !section.contains("1. d7\n"),
            "old entries stay out of the bounded view"
        );
        assert!(section.contains("1. d8\n"));
        assert!(section.contains("12. d19\n"));
    }

    #[test]
    fn record_replaces_duplicate_summary() {
        let mut log = DecisionLog::default();
        log.record(dec("use BTreeMap", "ordered iteration"));
        log.record(dec("use BTreeMap", "ordered iteration + we revisited"));
        assert_eq!(log.entries().len(), 1, "duplicate summary replaced");
        assert!(
            log.entries()[0].rationale.contains("revisited"),
            "refined rationale wins: {:?}",
            log.entries()[0]
        );
    }

    #[test]
    fn prompt_section_only_when_nonempty() {
        let mut log = DecisionLog::default();
        assert!(log.prompt_section().is_none());
        log.record(dec("skip Windows", "no CI for it"));
        let section = log.prompt_section().expect("nonempty log renders");
        assert!(section.contains("Key decisions"), "header: {section}");
        assert!(section.contains("skip Windows"), "summary: {section}");
        assert!(section.contains("no CI for it"), "rationale: {section}");
    }

    #[test]
    fn prompt_section_lists_files_when_present() {
        let mut log = DecisionLog::default();
        log.record(Decision {
            summary: "new config layer".into(),
            rationale: "needed per-env overrides".into(),
            files: vec!["src/config.rs".into(), "src/main.rs".into()],
        });
        let section = log.prompt_section().unwrap();
        assert!(
            section.contains("src/config.rs, src/main.rs"),
            "files: {section}"
        );
    }

    #[test]
    fn record_is_exact_while_prompt_view_clips_unbounded_fields() {
        let mut log = DecisionLog::default();
        let summary = "S".repeat(500);
        let rationale = "R".repeat(2_000);
        let files = (0..20)
            .map(|i| format!("crates/hi-agent/src/{i:03}-very-long-path.rs"))
            .collect::<Vec<_>>();
        log.record(Decision {
            summary: summary.clone(),
            rationale: rationale.clone(),
            files: files.clone(),
        });
        let entry = &log.entries()[0];
        assert_eq!(entry.summary, summary);
        assert_eq!(entry.rationale, rationale);
        assert_eq!(entry.files, files);
        let section = log.prompt_section().unwrap();
        assert!(
            section.chars().count() < 4_000,
            "decision prompt section must stay small: {}",
            section.chars().count()
        );
    }
}
