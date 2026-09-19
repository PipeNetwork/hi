//! Cap identical inspect tools (`grep`/`read`/`list`/`glob`/`repo_map`) in one turn.
//!
//! A live ~/chat "build all of that" turn grepped the same missing
//! `MAX_PASSWORD` pattern eight times after cheap-shrink stubbed the files it
//! had already read. Exact repeats cannot change the tree until a mutation
//! lands; refuse the second call and let the harness stop after two refusals
//! (same as detached probes). `ToolHost` resets this ledger after a successful
//! write/edit so a post-fix `grep`/`read` is not answered with the pre-fix body.

use std::collections::HashMap;

use crate::bash_repeat::is_probe_refusal;

/// First identical inspect runs; the second is a reminder.
pub const MAX_IDENTICAL_INSPECT: u32 = 1;

const MAX_REPLAY_CHARS: usize = 1_500;

pub fn is_inspect_tool(name: &str) -> bool {
    matches!(name, "read" | "grep" | "list" | "glob" | "repo_map")
}

#[derive(Debug, Default)]
pub struct InspectRepeatLedger {
    counts: HashMap<String, u32>,
    last_output: HashMap<String, String>,
}

impl InspectRepeatLedger {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn reset(&mut self) {
        self.counts.clear();
        self.last_output.clear();
    }

    /// Drop one inspect so a re-fetch is allowed after cheap-shrink omitted
    /// that result from the transcript.
    pub fn forget(&mut self, key: &str) {
        self.counts.remove(key);
        self.last_output.remove(key);
    }

    /// Record a completed inspect body so a later identical call can replay it.
    pub fn record_output(&mut self, name: &str, key: &str, output: &str) {
        if !is_inspect_tool(name) || is_probe_refusal(output) {
            return;
        }
        self.last_output
            .insert(key.to_string(), clip_replay(output));
    }

    /// Count this inspect. After [`MAX_IDENTICAL_INSPECT`] matching calls,
    /// returns a refusal that includes the previous body.
    pub fn admit(&mut self, name: &str, key: &str) -> Option<String> {
        if !is_inspect_tool(name) {
            return None;
        }
        let count = self.counts.entry(key.to_string()).or_insert(0);
        *count = count.saturating_add(1);
        if *count <= MAX_IDENTICAL_INSPECT {
            return None;
        }
        let previous = self
            .last_output
            .get(key)
            .map(String::as_str)
            .filter(|text| !text.is_empty())
            .unwrap_or("(previous body was empty or compacted)");
        Some(format!(
            "This exact `{name}` already ran this turn ({count} times) and returned:\n\
             {previous}\n\n\
             Do not repeat it. If there were no matches, the symbol is not in the \
             tree — add it with `edit`/`write`. Change the pattern, path, or offset \
             for a different search."
        ))
    }
}

fn clip_replay(output: &str) -> String {
    let mut out = String::new();
    for ch in output.chars() {
        if out.len() >= MAX_REPLAY_CHARS {
            out.push('…');
            break;
        }
        out.push(ch);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn second_identical_grep_is_refused_with_previous_body() {
        let mut ledger = InspectRepeatLedger::new();
        assert!(ledger.admit("grep", "k").is_none());
        ledger.record_output("grep", "k", "no matches for MAX_PASSWORD");
        let refused = ledger.admit("grep", "k").expect("second grep");
        assert!(refused.contains("already ran this turn"), "{refused}");
        assert!(refused.contains("no matches for MAX_PASSWORD"), "{refused}");
        assert!(is_probe_refusal(&refused));
        assert!(ledger.admit("bash", "k").is_none());
        ledger.reset();
        assert!(ledger.admit("grep", "k").is_none());
    }

    #[test]
    fn forget_allows_the_same_inspect_again() {
        let mut ledger = InspectRepeatLedger::new();
        assert!(ledger.admit("read", "k").is_none());
        ledger.record_output("read", "k", "body");
        assert!(ledger.admit("read", "k").is_some());
        ledger.forget("k");
        assert!(ledger.admit("read", "k").is_none());
    }

    #[test]
    fn mutations_and_distinct_keys_are_not_capped() {
        let mut ledger = InspectRepeatLedger::new();
        assert!(ledger.admit("edit", "k").is_none());
        assert!(ledger.admit("grep", "a").is_none());
        ledger.record_output("grep", "a", "hits");
        assert!(ledger.admit("grep", "b").is_none());
    }
}
