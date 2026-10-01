//! `/ladder`: run one task cheap-first. The first attempt uses the cheapest
//! rung (a model, a reasoning effort, or both) and the verify command judges
//! every attempt. A failed check is retried with its output; the ladder
//! climbs a rung when the rung's attempts are spent or an attempt changed
//! nothing, and ends when the check passes or the top rung fails.
//!
//! The model never grades its own work: [`Ladder::after_turn`] reads only
//! the turn's stop reason, its changed files, and the verify result. It is
//! pure, so every transition is unit-tested without a model;
//! [`crate::ladder_harness`] applies rungs and feeds it each turn.

use std::fmt::Write as _;

use hi_ai::{ReasoningEffort, Usage};

use crate::TurnStopReason;
use crate::command::{EffortArg, parse_model_args};
use crate::pipe::reasoning_wire_value;
use crate::usage::fmt_tokens;

/// Attempts on one rung before climbing, when `[ladder] attempts` is unset.
pub const DEFAULT_ATTEMPTS_PER_RUNG: u32 = 1;
/// Cap on attempts per rung, so a config typo cannot buy an unbounded loop.
pub const MAX_ATTEMPTS_PER_RUNG: u32 = 5;
/// Rungs used when none are configured: reasoning effort only, on the model
/// the ladder starts from.
pub const DEFAULT_EFFORT_RUNGS: [ReasoningEffort; 3] = [
    ReasoningEffort::Low,
    ReasoningEffort::Medium,
    ReasoningEffort::High,
];
/// Jev `needs_high_effort` score (0 to 1) at or above which the ladder starts
/// one rung up. The same score maps a turn to `high` effort.
pub const JEV_SKIP_THRESHOLD: f64 = 0.75;
/// Fewest rungs for a Jev skip: on a two-rung ladder a skip would start on
/// the most expensive rung.
pub const MIN_RUNGS_FOR_JEV_SKIP: usize = 3;
/// First token of every retry prompt the ladder writes.
pub const LADDER_PREFIX: &str = "[ladder]";
/// Characters of failing verify output quoted in a retry prompt. The tail is
/// kept: test runners print the failure summary last.
const FAILURE_TAIL_CHARS: usize = 4_000;

/// One step of the ladder: a model, a reasoning effort, or both.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LadderRung {
    /// Model id; `None` keeps the model the ladder started from.
    pub model: Option<String>,
    /// Reasoning effort; `None` leaves effort to the session (a pinned
    /// `/effort`, or Jev's per-turn choice).
    pub effort: Option<EffortArg>,
}

impl LadderRung {
    /// Parse `/model` syntax: `pipe/deepseek-v4-flash low`,
    /// `pipe/gpt-6-astra`, or an effort alone (`high`).
    ///
    /// # Errors
    /// An empty spec, or a model id with whitespace in it (usually a
    /// misspelled effort level).
    pub fn parse(spec: &str) -> Result<Self, String> {
        let args = parse_model_args(spec);
        if args.model.is_none() && args.effort.is_none() {
            return Err("empty ladder rung".into());
        }
        if args
            .model
            .as_deref()
            .is_some_and(|model| model.contains(char::is_whitespace))
        {
            return Err(format!(
                "ladder rung `{}`: expected `model [low|medium|high|xhigh|off]`",
                spec.trim()
            ));
        }
        Ok(Self {
            model: args.model,
            effort: args.effort,
        })
    }

    /// `model · effort`, with `base_model` standing in for a rung without a
    /// model of its own.
    pub fn label(&self, base_model: &str) -> String {
        let model = self.model.as_deref().unwrap_or(base_model);
        match self.effort {
            Some(effort) => format!("{model} · {}", effort.label()),
            None => model.to_string(),
        }
    }
}

/// `[ladder]` knobs. `hi-cli`'s `config/ladder.rs` documents the TOML.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LadderSettings {
    /// Cheapest first. Empty means [`DEFAULT_EFFORT_RUNGS`] on the session
    /// model.
    pub rungs: Vec<LadderRung>,
    /// Attempts on a rung before climbing, clamped to
    /// `1..=`[`MAX_ATTEMPTS_PER_RUNG`] when a ladder starts.
    pub attempts_per_rung: u32,
    /// Check used when the session has no `/verify` command.
    pub verify: Option<String>,
    /// Ask Jev whether the task is hard enough to skip the cheapest rung.
    /// Needs a TypeSafe key; without one the ladder always starts at the
    /// bottom.
    pub jev_start: bool,
    /// Why configured rungs did not parse. A ladder refuses to start until
    /// they are fixed, rather than run a shorter ladder than configured.
    pub invalid_rungs: Vec<String>,
}

impl Default for LadderSettings {
    fn default() -> Self {
        Self {
            rungs: Vec::new(),
            attempts_per_rung: DEFAULT_ATTEMPTS_PER_RUNG,
            verify: None,
            jev_start: true,
            invalid_rungs: Vec::new(),
        }
    }
}

impl LadderSettings {
    /// The configured rungs, or the default effort ladder.
    pub fn resolved_rungs(&self) -> Vec<LadderRung> {
        if !self.rungs.is_empty() {
            return self.rungs.clone();
        }
        DEFAULT_EFFORT_RUNGS
            .iter()
            .map(|&effort| LadderRung {
                model: None,
                effort: Some(EffortArg::Level(effort)),
            })
            .collect()
    }
}

/// Merge consecutive rungs that would send the same request, keeping the
/// later label. Pipe sends DeepSeek `medium`, `high`, and `xhigh` as one
/// effort, so the default low → medium → high ladder on a DeepSeek model has
/// two distinct rungs, not three. Returns the rungs and how many merged.
pub fn distinct_rungs(rungs: Vec<LadderRung>, base_model: &str) -> (Vec<LadderRung>, usize) {
    let mut kept: Vec<LadderRung> = Vec::with_capacity(rungs.len());
    let mut merged = 0;
    for rung in rungs {
        if let Some(last) = kept.last_mut()
            && wire_key(last, base_model) == wire_key(&rung, base_model)
        {
            *last = rung;
            merged += 1;
            continue;
        }
        kept.push(rung);
    }
    (kept, merged)
}

/// The model and effort a rung puts on the wire.
fn wire_key<'a>(rung: &'a LadderRung, base_model: &'a str) -> (&'a str, &'static str) {
    let model = rung.model.as_deref().unwrap_or(base_model);
    let effort = match rung.effort {
        None => "session",
        Some(EffortArg::Off) => "off",
        Some(EffortArg::Level(level)) => reasoning_wire_value(model, level),
    };
    (model, effort)
}

/// Where a ladder is. Every phase but `Running` is final.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LadderPhase {
    #[default]
    Idle,
    Running,
    /// The verify command passed.
    Verified,
    /// The top rung failed verification.
    Exhausted,
    /// Ended early: a cancelled turn, no check ran, or `/ladder stop`.
    Stopped,
}

impl LadderPhase {
    /// Lowercase name for reports.
    pub fn label(self) -> &'static str {
        match self {
            Self::Idle => "idle",
            Self::Running => "running",
            Self::Verified => "verified",
            Self::Exhausted => "exhausted",
            Self::Stopped => "stopped",
        }
    }
}

/// How one attempt ended.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AttemptResult {
    Verified,
    VerifyFailed,
    /// The turn ended with an error before the check could run.
    TurnError(String),
    Cancelled,
    /// The turn completed but no verify command ran.
    Unverified,
}

impl AttemptResult {
    /// The check decides whenever it ran, even after a turn that ended with
    /// a completion-policy error: the model's own account never does.
    fn from_facts(facts: &LadderTurnFacts) -> Self {
        if facts.stop_reason == TurnStopReason::Cancelled
            || facts.verification.as_deref() == Some("cancelled")
        {
            return Self::Cancelled;
        }
        match facts.verification.as_deref() {
            Some("passed") => Self::Verified,
            Some(_) => Self::VerifyFailed,
            None if facts.stop_reason == TurnStopReason::Error => {
                Self::TurnError(facts.error.clone().unwrap_or_else(|| "turn error".into()))
            }
            None => Self::Unverified,
        }
    }

    /// Short label for reports.
    pub fn label(&self) -> String {
        match self {
            Self::Verified => "verified".into(),
            Self::VerifyFailed => "verify failed".into(),
            Self::TurnError(error) => format!("error: {}", first_line(error)),
            Self::Cancelled => "cancelled".into(),
            Self::Unverified => "no check ran".into(),
        }
    }
}

/// One finished attempt, for the report.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LadderAttempt {
    /// Zero-based rung index.
    pub rung: usize,
    /// The rung's `model · effort` label.
    pub label: String,
    pub result: AttemptResult,
    /// Files the attempt changed.
    pub changed_files: usize,
    pub input_tokens: u64,
    pub output_tokens: u64,
}

/// What the ladder reads from a finished turn.
#[derive(Clone, Debug, Default)]
pub struct LadderTurnFacts {
    pub stop_reason: TurnStopReason,
    pub error: Option<String>,
    pub changed_files: Vec<String>,
    /// `TurnOutcome::verification`: `passed`, `failed`, `cancelled`, or
    /// `None` when no check ran.
    pub verification: Option<String>,
    /// The verify command's output when it failed.
    pub verify_output: Option<String>,
    /// Tokens the turn used.
    pub usage: Usage,
}

/// What to do after a turn.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LadderStep {
    /// Run this prompt next, on the current (possibly new) rung.
    RunPrompt(String),
    /// The check passed; the summary line.
    Verified(String),
    /// The ladder ended without a passing check; the summary line.
    Stopped(String),
    /// No ladder is running.
    Idle,
}

/// Result of `/ladder <arg>`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LadderCommand {
    /// A ladder started: print `notes`, then run `prompt` as a turn.
    Start { prompt: String, notes: Vec<String> },
    /// Status, a stop summary, or why the ladder could not start.
    Message(String),
}

/// A running or finished ladder. Session-local and never persisted.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Ladder {
    pub phase: LadderPhase,
    /// The user's task, restated in every retry prompt.
    pub task: String,
    /// The model the ladder started from: rungs without a model use it, and
    /// the session returns to it when the ladder ends.
    pub base_model: String,
    pub rungs: Vec<LadderRung>,
    /// Current rung, zero-based.
    pub rung: usize,
    /// Failed attempts on the current rung.
    pub attempts_on_rung: u32,
    pub attempts_per_rung: u32,
    /// The check that judges every attempt.
    pub verify_command: String,
    /// True when the ladder supplied the session's verify command; the
    /// command is cleared again when the ladder ends.
    pub temporary_verify: bool,
    pub attempts: Vec<LadderAttempt>,
    /// Why the ladder started above the cheapest rung, when it did.
    pub start_note: Option<String>,
    pub stop_reason: Option<String>,
}

impl Ladder {
    /// Start `task` on rung `start_rung` (clamped to the rungs). The first
    /// prompt is the task itself.
    pub fn start(
        task: &str,
        base_model: &str,
        rungs: Vec<LadderRung>,
        attempts_per_rung: u32,
        verify_command: &str,
        start_rung: usize,
    ) -> Self {
        Self {
            phase: LadderPhase::Running,
            task: task.trim().to_string(),
            base_model: base_model.to_string(),
            rung: start_rung.min(rungs.len().saturating_sub(1)),
            rungs,
            attempts_per_rung: attempts_per_rung.clamp(1, MAX_ATTEMPTS_PER_RUNG),
            verify_command: verify_command.trim().to_string(),
            ..Self::default()
        }
    }

    pub fn is_active(&self) -> bool {
        self.phase == LadderPhase::Running
    }

    pub fn current_rung(&self) -> Option<&LadderRung> {
        self.rungs.get(self.rung)
    }

    /// The model the current rung runs on.
    pub fn rung_model(&self) -> Option<&str> {
        self.current_rung()
            .map(|rung| rung.model.as_deref().unwrap_or(&self.base_model))
    }

    /// The current rung's effort while the ladder runs. `None` leaves effort
    /// to the session.
    pub fn rung_effort(&self) -> Option<EffortArg> {
        if !self.is_active() {
            return None;
        }
        self.current_rung()?.effort
    }

    /// `model · effort` for rung `index`.
    pub fn rung_label(&self, index: usize) -> String {
        self.rungs
            .get(index)
            .map(|rung| rung.label(&self.base_model))
            .unwrap_or_default()
    }

    /// One line: where the ladder is, or how it ended.
    pub fn status_line(&self) -> String {
        let rungs = self.rungs.len();
        let attempts = self.attempts.len();
        match self.phase {
            LadderPhase::Idle => "ladder: not running".into(),
            LadderPhase::Running => {
                let mut line = format!(
                    "ladder · rung {}/{rungs} {} · attempt {}",
                    self.rung + 1,
                    self.rung_label(self.rung),
                    attempts + 1,
                );
                if self.attempts_per_rung > 1 {
                    let _ = write!(
                        line,
                        " ({}/{} on this rung)",
                        self.attempts_on_rung + 1,
                        self.attempts_per_rung
                    );
                }
                let _ = write!(line, " · verify `{}`", self.verify_command);
                line
            }
            LadderPhase::Verified => {
                let mut line = format!(
                    "ladder: `{}` passed on rung {}/{rungs} ({}) after {attempts} attempt(s)",
                    self.verify_command,
                    self.rung + 1,
                    self.rung_label(self.rung),
                );
                if self
                    .attempts
                    .iter()
                    .all(|attempt| attempt.changed_files == 0)
                {
                    line.push_str(
                        " · no attempt changed files: the check may have passed before the task",
                    );
                }
                line
            }
            LadderPhase::Exhausted | LadderPhase::Stopped => format!(
                "ladder {}: {} · {attempts} attempt(s)",
                self.phase.label(),
                self.stop_reason.as_deref().unwrap_or("stopped"),
            ),
        }
    }

    /// Feed a finished turn. A pass ends the ladder; a failed check or a
    /// turn error retries on the same rung or climbs; a cancelled turn, or a
    /// turn after which no check ran, stops it.
    pub fn after_turn(&mut self, facts: &LadderTurnFacts) -> LadderStep {
        if !self.is_active() {
            return LadderStep::Idle;
        }
        let result = AttemptResult::from_facts(facts);
        self.attempts.push(LadderAttempt {
            rung: self.rung,
            label: self.rung_label(self.rung),
            result: result.clone(),
            changed_files: facts.changed_files.len(),
            input_tokens: facts.usage.input_tokens,
            output_tokens: facts.usage.output_tokens,
        });
        match result {
            AttemptResult::Verified => {
                self.phase = LadderPhase::Verified;
                LadderStep::Verified(self.status_line())
            }
            AttemptResult::Cancelled => self.stop("the turn was cancelled"),
            AttemptResult::Unverified => self.stop("no verify command ran after the turn"),
            AttemptResult::VerifyFailed | AttemptResult::TurnError(_) => {
                self.retry_or_climb(&result, facts)
            }
        }
    }

    /// End a running ladder early. `Idle` when none is running.
    pub fn stop(&mut self, reason: &str) -> LadderStep {
        if !self.is_active() {
            return LadderStep::Idle;
        }
        self.phase = LadderPhase::Stopped;
        self.stop_reason = Some(reason.to_string());
        LadderStep::Stopped(self.status_line())
    }

    /// One line per attempt, then the token total. Empty before the first
    /// attempt.
    pub fn report_lines(&self) -> Vec<String> {
        if self.attempts.is_empty() {
            return Vec::new();
        }
        let mut lines: Vec<String> = self
            .attempts
            .iter()
            .enumerate()
            .map(|(index, attempt)| {
                format!(
                    "  {}. rung {} {} — {} · {} file(s) · {} in / {} out",
                    index + 1,
                    attempt.rung + 1,
                    attempt.label,
                    attempt.result.label(),
                    attempt.changed_files,
                    fmt_tokens(attempt.input_tokens),
                    fmt_tokens(attempt.output_tokens),
                )
            })
            .collect();
        let (input, output) = self.total_tokens();
        lines.push(format!(
            "  total: {} in / {} out",
            fmt_tokens(input),
            fmt_tokens(output)
        ));
        lines
    }

    /// Input and output tokens across all attempts.
    pub fn total_tokens(&self) -> (u64, u64) {
        self.attempts
            .iter()
            .fold((0, 0), |(input, output), attempt| {
                (input + attempt.input_tokens, output + attempt.output_tokens)
            })
    }

    fn retry_or_climb(&mut self, result: &AttemptResult, facts: &LadderTurnFacts) -> LadderStep {
        self.attempts_on_rung += 1;
        let no_progress = facts.changed_files.is_empty();
        if no_progress || self.attempts_on_rung >= self.attempts_per_rung {
            if self.rung + 1 >= self.rungs.len() {
                self.phase = LadderPhase::Exhausted;
                self.stop_reason = Some(format!(
                    "the top rung ({}) did not get `{}` to pass",
                    self.rung_label(self.rung),
                    self.verify_command
                ));
                return LadderStep::Stopped(self.status_line());
            }
            self.rung += 1;
            self.attempts_on_rung = 0;
        }
        LadderStep::RunPrompt(self.retry_prompt(result, facts, no_progress))
    }

    fn retry_prompt(
        &self,
        result: &AttemptResult,
        facts: &LadderTurnFacts,
        no_progress: bool,
    ) -> String {
        let command = &self.verify_command;
        let mut prompt = format!(
            "{LADDER_PREFIX} attempt {} · rung {}/{} {}\n\n",
            self.attempts.len() + 1,
            self.rung + 1,
            self.rungs.len(),
            self.rung_label(self.rung),
        );
        if let AttemptResult::TurnError(error) = result {
            let _ = writeln!(prompt, "The previous attempt ended with an error: {error}");
        } else {
            let _ = writeln!(prompt, "`{command}` failed after the previous attempt.");
            if let Some(output) = facts
                .verify_output
                .as_deref()
                .map(str::trim)
                .filter(|output| !output.is_empty())
            {
                let (tail, cut) = tail_chars(output, FAILURE_TAIL_CHARS);
                let scope = if cut {
                    format!(" (last {FAILURE_TAIL_CHARS} characters)")
                } else {
                    String::new()
                };
                let _ = writeln!(prompt, "\nOutput{scope}:\n```text\n{tail}\n```");
            }
        }
        if no_progress {
            prompt.push_str("\nThe previous attempt changed no files.\n");
        }
        let _ = write!(
            prompt,
            "\nFix what the check reports and run `{command}` before you finish; the harness runs it again to judge the attempt. Stay within the original task:\n\n{}",
            self.task
        );
        prompt
    }
}

/// The first line of a retry prompt the ladder wrote, for a short transcript
/// echo; `None` for any other prompt.
pub fn transcript_label(prompt: &str) -> Option<String> {
    prompt
        .starts_with(LADDER_PREFIX)
        .then(|| first_line(prompt).to_string())
}

fn first_line(text: &str) -> &str {
    text.lines().next().unwrap_or("").trim()
}

/// The last `max` characters of `text`, and whether anything was cut.
fn tail_chars(text: &str, max: usize) -> (&str, bool) {
    let skip = text.chars().count().saturating_sub(max);
    if skip == 0 {
        return (text, false);
    }
    let start = text
        .char_indices()
        .nth(skip)
        .map_or(text.len(), |(index, _)| index);
    (&text[start..], true)
}

#[cfg(test)]
#[path = "ladder_tests.rs"]
mod tests;
