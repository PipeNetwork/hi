//! A durable intra-session decision log.
//!
//! When the model commits to a key decision, it records it via the
//! `record_decision` tool. The log is injected into the system prompt each turn
//! (capped, oldest-first) so the model stays consistent across a long session.
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

/// Cap the log so it can't grow the system prompt without bound. Older
/// decisions age out first — the recent ones are most likely to still govern
/// the work in progress.
const MAX_DECISIONS: usize = 12;

/// The durable decision log. Injected into the system prompt each turn;
/// compaction-immune (it's not part of the summarizable history).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DecisionLog {
    entries: Vec<Decision>,
}

impl DecisionLog {
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
        // Age out the oldest when over capacity.
        if self.entries.len() > MAX_DECISIONS {
            self.entries.remove(0);
        }
    }

    /// The decisions, oldest-first, for system-prompt injection.
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
        let mut out = String::from("\n\n[Key decisions this session — stay consistent with these]\n");
        for (i, d) in self.entries.iter().enumerate() {
            out.push_str(&format!("{}. {}\n   why: {}\n", i + 1, d.summary, d.rationale));
            if !d.files.is_empty() {
                out.push_str(&format!("   files: {}\n", d.files.join(", ")));
            }
        }
        Some(out)
    }
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
    fn record_appends_and_caps() {
        let mut log = DecisionLog::default();
        for i in 0..20 {
            log.record(dec(&format!("d{i}"), "r"));
        }
        assert_eq!(log.entries().len(), MAX_DECISIONS);
        // The oldest (d0..d7) aged out; d8 onward remain.
        assert_eq!(log.entries().first().unwrap().summary, "d8");
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
        assert!(section.contains("src/config.rs, src/main.rs"), "files: {section}");
    }
}
