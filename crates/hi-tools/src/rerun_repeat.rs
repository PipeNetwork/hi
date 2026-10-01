//! Refuse a shell command that already printed the same output twice while
//! no file changed.
//!
//! A live drme session ran one `python3 << 'EOF'` bit-parse script eight
//! times in a row with byte-identical output and no edit in between. It is
//! neither a read-only probe nor a verify command, so nothing refused it and
//! the identical-tool storm ended the whole turn. Durations and rates are
//! masked before comparing, so a benchmark that only differs in timing
//! still counts as the same output. Commands that `sleep`/`wait`/`watch`
//! poll for outside change and are never refused.

use std::collections::HashMap;

use crate::bash_repeat::is_probe_refusal;

/// Identical outputs in a row, with no file change, before the next run of
/// the same command is refused.
pub const SAME_OUTPUT_RUNS_BEFORE_REFUSAL: u32 = 2;

const MAX_REPLAY_CHARS: usize = 1_500;
const MAX_LABEL_CHARS: usize = 80;

#[derive(Debug, Default)]
pub struct RerunLedger {
    runs: HashMap<String, Rerun>,
}

#[derive(Debug)]
struct Rerun {
    shape: String,
    same: u32,
    last_output: String,
}

impl RerunLedger {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn reset(&mut self) {
        self.runs.clear();
    }

    /// A file changed, so every command may print something new.
    pub fn note_mutation(&mut self) {
        self.runs.clear();
    }

    pub fn record_output(&mut self, key: &str, output: &str) {
        if is_probe_refusal(output) {
            return;
        }
        let shape = output_shape(output);
        let last_output = clip_replay(output);
        match self.runs.get_mut(key) {
            Some(run) if run.shape == shape => {
                run.same = run.same.saturating_add(1);
                run.last_output = last_output;
            }
            _ => {
                self.runs.insert(
                    key.to_string(),
                    Rerun {
                        shape,
                        same: 1,
                        last_output,
                    },
                );
            }
        }
    }

    /// Admit this run of `command`, or return a refusal that replays the
    /// last output. Refusals say "already ran this turn" so repeated
    /// attempts still stop the turn through the probe-refusal cap.
    pub fn admit(&self, key: &str, command: &str) -> Option<String> {
        if polls_for_change(command) {
            return None;
        }
        let run = self.runs.get(key)?;
        if run.same < SAME_OUTPUT_RUNS_BEFORE_REFUSAL {
            return None;
        }
        let previous = if run.last_output.trim().is_empty() {
            "(no output)"
        } else {
            run.last_output.as_str()
        };
        Some(format!(
            "`{label}` already ran this turn {same} times with the same output, and \
             no file has changed since, so it would print this again:\n{previous}\n\n\
             Act on that output instead of re-running it. Answer from it, or run a \
             different command that tells you something new. Do not change a file \
             unless the user asked for a change.",
            label = command_label(command),
            same = run.same,
        ))
    }
}

/// `sleep 30; tail log` waits for something outside the workspace to change.
fn polls_for_change(command: &str) -> bool {
    command
        .split(|ch: char| !(ch.is_ascii_alphanumeric() || ch == '_'))
        .any(|word| matches!(word, "sleep" | "wait" | "watch"))
}

fn command_label(command: &str) -> String {
    let first = command.trim().lines().next().unwrap_or_default().trim();
    let mut label: String = first.chars().take(MAX_LABEL_CHARS).collect();
    if label.len() < first.len() || command.trim().lines().nth(1).is_some() {
        label.push('…');
    }
    label
}

/// The output with durations and rates masked: `finished in 1.23s` and
/// `finished in 0.98s` are the same result.
fn output_shape(output: &str) -> String {
    let chars: Vec<char> = output.chars().collect();
    let mut shape = String::with_capacity(output.len());
    let mut i = 0;
    while i < chars.len() {
        let starts_number =
            chars[i].is_ascii_digit() && (i == 0 || !chars[i - 1].is_ascii_alphanumeric());
        if !starts_number {
            shape.push(chars[i]);
            i += 1;
            continue;
        }
        let mut end = i;
        while end < chars.len() && (chars[end].is_ascii_digit() || chars[end] == '.') {
            end += 1;
        }
        if is_time_unit(&chars[end..]) {
            shape.push('#');
        } else {
            shape.extend(&chars[i..end]);
        }
        i = end;
    }
    shape
}

fn is_time_unit(rest: &[char]) -> bool {
    let rest = match rest.first() {
        Some(' ') => &rest[1..],
        _ => rest,
    };
    let unit: String = rest
        .iter()
        .take_while(|ch| ch.is_alphabetic() || **ch == '/')
        .collect();
    matches!(
        unit.as_str(),
        "ns" | "us"
            | "µs"
            | "ms"
            | "s"
            | "sec"
            | "secs"
            | "second"
            | "seconds"
            | "m"
            | "min"
            | "mins"
            | "minute"
            | "minutes"
            | "h"
            | "fps"
    ) || unit.ends_with("/s")
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

    const SCRIPT: &str = "python3 << 'EOF'\nimport struct\nprint(bits_for(0x4a))\nEOF";
    const OUTPUT: &str =
        "seq_profile=0 still_picture=0\nbits_for(74) = 7\nobu_size mismatch at 0x1f";

    #[test]
    fn third_identical_run_is_refused_with_the_last_output() {
        let mut ledger = RerunLedger::new();
        assert!(ledger.admit("k", SCRIPT).is_none());
        ledger.record_output("k", OUTPUT);
        assert!(ledger.admit("k", SCRIPT).is_none());
        ledger.record_output("k", OUTPUT);
        let refused = ledger.admit("k", SCRIPT).expect("same output twice");
        assert!(is_probe_refusal(&refused), "{refused}");
        assert!(refused.contains("obu_size mismatch at 0x1f"), "{refused}");
        assert!(refused.contains("`python3 << 'EOF'…`"), "{refused}");
        assert!(refused.contains("2 times"), "{refused}");
    }

    #[test]
    fn a_file_change_or_new_turn_allows_the_command_again() {
        let mut ledger = RerunLedger::new();
        ledger.record_output("k", OUTPUT);
        ledger.record_output("k", OUTPUT);
        assert!(ledger.admit("k", SCRIPT).is_some());
        ledger.note_mutation();
        assert!(ledger.admit("k", SCRIPT).is_none());
        ledger.record_output("k", OUTPUT);
        ledger.record_output("k", OUTPUT);
        ledger.reset();
        assert!(ledger.admit("k", SCRIPT).is_none());
    }

    #[test]
    fn changing_output_keeps_the_command_admitted() {
        let mut ledger = RerunLedger::new();
        for pid in 0..10 {
            ledger.record_output("k", &format!("storm {pid}"));
            assert!(ledger.admit("k", "echo storm $$").is_none());
        }
        ledger.record_output("other", OUTPUT);
        ledger.record_output("other", OUTPUT);
        assert!(ledger.admit("k", "echo storm $$").is_none());
    }

    #[test]
    fn timing_noise_is_the_same_output() {
        let mut ledger = RerunLedger::new();
        ledger.record_output(
            "k",
            "decoded 30 frames in 1.23s (24.4 fps)\ntest result: ok. 567 passed; finished in 0.98s",
        );
        ledger.record_output(
            "k",
            "decoded 30 frames in 1.31s (22.9 fps)\ntest result: ok. 567 passed; finished in 1.02s",
        );
        assert!(ledger.admit("k", "./target/release/drme in.ivf").is_some());
        assert_eq!(
            output_shape("took 12ms, 3.5 MB/s, 40 us, 2 files, 567/567, x86_64"),
            "took #ms, # MB/s, # us, 2 files, 567/567, x86_64"
        );
    }

    #[test]
    fn polling_commands_and_refusals_are_never_refused_or_recorded() {
        let mut ledger = RerunLedger::new();
        let poll = "sleep 30; tail -5 /tmp/encode.log";
        ledger.record_output("poll", "frame 12/300");
        ledger.record_output("poll", "frame 12/300");
        assert!(ledger.admit("poll", poll).is_none());
        assert!(
            ledger
                .admit("poll", "python3 -c 'import time; time.sleep(5)'")
                .is_none()
        );
        ledger.record_output("k", "x already ran this turn");
        ledger.record_output("k", "x already ran this turn");
        assert!(ledger.admit("k", SCRIPT).is_none());
    }
}
