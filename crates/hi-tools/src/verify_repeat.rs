//! Refuse a verify command (`cargo test`, `pytest`, …) whose result cannot
//! have changed.
//!
//! A live drme session ran `cargo test 2>&1 | tail -40` seventeen times in one
//! turn. A plain per-turn cap then refused the fourth run even after the model
//! had edited code, and the model got around it by dropping the `timeout`
//! argument. The key is the command text without its pager, and a rerun is
//! refused only when no file changed since the last identical run.

use std::collections::{HashMap, HashSet};

use crate::bash_repeat::is_probe_refusal;

/// Runaway guard: identical verify runs per turn, even with edits between.
pub const MAX_VERIFY_RUNS_PER_TURN: u32 = 16;

const MAX_REPLAY_CHARS: usize = 1_500;

#[derive(Debug, Default)]
pub struct VerifyRepeatLedger {
    runs: HashMap<String, u32>,
    unchanged: HashSet<String>,
    last_output: HashMap<String, String>,
}

impl VerifyRepeatLedger {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn reset(&mut self) {
        self.runs.clear();
        self.unchanged.clear();
        self.last_output.clear();
    }

    /// A file changed, so every verify command may produce a new result.
    pub fn note_mutation(&mut self) {
        self.unchanged.clear();
    }

    pub fn record_output(&mut self, key: &str, output: &str) {
        if is_probe_refusal(output) {
            return;
        }
        self.last_output
            .insert(key.to_string(), clip_replay(output));
    }

    /// Admit this verify run, or return a refusal that replays the last result.
    /// Refusals say "already ran this turn" so repeated attempts still stop
    /// the turn through the probe-refusal cap.
    pub fn admit(&mut self, key: &str) -> Option<String> {
        let previous = self
            .last_output
            .get(key)
            .map(String::as_str)
            .filter(|text| !text.is_empty())
            .unwrap_or("(previous output was empty)");
        if self.unchanged.contains(key) {
            return Some(format!(
                "`{key}` already ran this turn and no file has changed since, so it \
                 would print the same result:\n{previous}\n\nAct on that result \
                 instead of re-running it: fix failures with edit; if it passed \
                 and the active plan step is complete, call update_plan. Never \
                 make a no-op edit just to run it again."
            ));
        }
        let runs = self.runs.entry(key.to_string()).or_insert(0);
        if *runs >= MAX_VERIFY_RUNS_PER_TURN {
            return Some(format!(
                "`{key}` already ran this turn {runs} times. Stop and report the \
                 last result:\n{previous}"
            ));
        }
        *runs += 1;
        self.unchanged.insert(key.to_string());
        None
    }
}

/// The verify command without a trailing `tail`/`head` pager or extra
/// whitespace, so `cargo test 2>&1 | tail -40` and `cargo test 2>&1 | tail -30`
/// are the same run.
pub fn verify_key(command: &str) -> String {
    let mut text = command.trim();
    if let Some((head, pager)) = text.rsplit_once('|') {
        let words: Vec<&str> = pager.split_whitespace().collect();
        let program = words
            .first()
            .map(|word| word.rsplit('/').next().unwrap_or(word));
        let count_only = words[1.min(words.len())..].iter().all(|word| {
            *word == "-n"
                || word
                    .trim_start_matches("-n")
                    .trim_start_matches('-')
                    .chars()
                    .all(|c| c.is_ascii_digit())
        });
        if matches!(program, Some("tail" | "head")) && count_only && !head.trim().is_empty() {
            text = head.trim_end();
        }
    }
    text.split_whitespace().collect::<Vec<_>>().join(" ")
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
    fn rerun_without_a_change_is_refused_with_the_last_output() {
        let mut ledger = VerifyRepeatLedger::new();
        assert!(ledger.admit("cargo test").is_none());
        ledger.record_output("cargo test", "test result: FAILED. cabac missing [9]");
        let refused = ledger.admit("cargo test").expect("nothing changed");
        assert!(is_probe_refusal(&refused), "{refused}");
        assert!(refused.contains("cabac missing [9]"), "{refused}");
        assert!(refused.contains("no file has changed"), "{refused}");
    }

    #[test]
    fn a_file_change_allows_the_same_command_again() {
        let mut ledger = VerifyRepeatLedger::new();
        for _ in 0..MAX_VERIFY_RUNS_PER_TURN {
            assert!(ledger.admit("cargo test").is_none());
            ledger.note_mutation();
        }
        let capped = ledger.admit("cargo test").expect("runaway guard");
        assert!(is_probe_refusal(&capped), "{capped}");
        ledger.reset();
        assert!(ledger.admit("cargo test").is_none());
    }

    #[test]
    fn pager_and_spacing_variants_share_one_key() {
        assert_eq!(verify_key("cargo test 2>&1 | tail -40"), "cargo test 2>&1");
        assert_eq!(
            verify_key("cargo  test 2>&1 |  tail -n 30"),
            "cargo test 2>&1"
        );
        assert_eq!(verify_key("cargo test 2>&1 | head -20"), "cargo test 2>&1");
        assert_eq!(
            verify_key("cargo test 2>&1 | grep FAILED"),
            "cargo test 2>&1 | grep FAILED"
        );
        assert_eq!(verify_key("cargo test lib::x"), "cargo test lib::x");
    }
}
