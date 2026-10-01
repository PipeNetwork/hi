//! Refuse an edit/write that already failed while the file has not changed.
//!
//! Live drme turns retried the same `old_string` miss (and the same refused
//! large-file `write`) until the harness hit `plan_stall`. The first failure
//! already pointed at the current text; repeating it cannot land.

use std::collections::HashMap;

use crate::bash_repeat::is_probe_refusal;

const MAX_REPLAY_CHARS: usize = 1_200;

/// Fingerprint of a mutation that failed. Survives across turns until a
/// file actually changes so a driven continue cannot replay the miss.
#[derive(Debug, Default)]
pub struct FailedMutationLedger {
    failed: HashMap<String, String>,
}

impl FailedMutationLedger {
    pub fn new() -> Self {
        Self::default()
    }

    /// A file changed, so previous misses may now apply.
    pub fn note_mutation(&mut self) {
        self.failed.clear();
    }

    pub fn note_failure(&mut self, key: &str, error: &str) {
        if key.is_empty() || is_probe_refusal(error) {
            return;
        }
        self.failed.insert(key.to_string(), clip_replay(error));
    }

    /// Admit this mutation, or return a refusal that replays the last miss.
    pub fn admit(&self, key: &str) -> Option<String> {
        let previous = self.failed.get(key)?;
        Some(format!(
            "This exact edit already failed and the file has not changed \
             (already ran this turn):\n{previous}\n\n{guidance}",
            guidance = failure_guidance(previous),
        ))
    }
}

/// The replay must match the failure. A snippet `write` of text that is
/// already in the file is not an `old_string` miss, and telling the model to
/// copy `old_string` sent a lookup turn into a delete-and-restore edit.
fn failure_guidance(previous: &str) -> &'static str {
    if previous.contains("already in this file") {
        "Do not change the file. The text is already there. Answer the user \
         from it. Call update_plan only when a plan step is open and this \
         text finishes that step."
    } else if previous.contains("old_string") {
        "Do not retry the same old_string. Read the target file and copy \
         old_string from the current text. If your new_string is already in \
         the file, call update_plan: mark the active step done and set the \
         next unfinished step active."
    } else {
        "Do not retry this write. Use edit only for a change the user asked \
         for that is not already in the file. If you were answering a \
         question, stop and answer from what you already read."
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

/// True when the replacement text is already in the file, so retrying the
/// miss cannot make progress.
pub fn new_string_already_present(file: &str, new: &str) -> bool {
    let new = collapse_ws(new);
    if new.len() < 40 {
        return false;
    }
    collapse_ws(file).contains(&new)
}

fn collapse_ws(text: &str) -> String {
    let mut out = String::new();
    let mut space = false;
    for ch in text.chars() {
        if ch.is_whitespace() {
            if !space && !out.is_empty() {
                out.push(' ');
                space = true;
            }
        } else {
            out.push(ch);
            space = false;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn second_identical_miss_is_refused() {
        let mut ledger = FailedMutationLedger::new();
        assert!(ledger.admit("edit:foo").is_none());
        ledger.note_failure("edit:foo", "Error: old_string not found near line 12");
        let refused = ledger.admit("edit:foo").expect("replay");
        assert!(is_probe_refusal(&refused), "{refused}");
        assert!(refused.contains("old_string not found"), "{refused}");
        assert!(refused.contains("copy old_string"), "{refused}");
        assert!(refused.contains("update_plan"), "{refused}");
    }

    #[test]
    fn already_present_replay_does_not_ask_for_another_edit() {
        let mut ledger = FailedMutationLedger::new();
        ledger.note_failure(
            "write:wallet",
            "Error: this content is already in this file. Nothing was written.",
        );
        let refused = ledger.admit("write:wallet").expect("replay");
        assert!(is_probe_refusal(&refused), "{refused}");
        assert!(refused.contains("already in this file"), "{refused}");
        assert!(refused.contains("Answer the user"), "{refused}");
        assert!(!refused.contains("old_string"), "{refused}");
        assert!(!refused.contains("copy old_string"), "{refused}");
    }

    #[test]
    fn a_file_change_allows_the_same_edit_again() {
        let mut ledger = FailedMutationLedger::new();
        ledger.note_failure("edit:foo", "old_string not found");
        ledger.note_mutation();
        assert!(ledger.admit("edit:foo").is_none());
    }

    #[test]
    fn already_present_needs_a_substantial_new_string() {
        let file = "fn main() {\n    println!(\"hello world from av1\");\n}\n";
        assert!(new_string_already_present(
            file,
            "fn main() {\n    println!(\"hello world from av1\");\n}"
        ));
        assert!(!new_string_already_present(file, "fn main()"));
    }
}
