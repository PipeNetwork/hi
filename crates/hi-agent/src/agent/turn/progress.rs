//! Per-turn progress classification and no-progress tracking.

use std::collections::{BTreeMap, HashSet};

use hi_ai::Content;

use crate::ProgressEvent;
use crate::heuristics::{looks_like_unfinished_step, parse_text_tool_calls};
use crate::steering::{
    EvidenceTracker, ImplementationTracker, ToolLoopGuardrail, bash_no_progress_signature,
    classify_bash_command, evidence_kind_for_tool, implementation_tool_call_validates,
    implementation_tool_result_landed_mutation, implementation_tool_result_landed_substantive_edit,
    inspection_signature,
};

use super::retention::ProgressEventLog;

pub(super) const NO_PROGRESS_FINAL_ANSWER_NUDGE_THRESHOLD: u32 = 2;
pub(super) const NO_PROGRESS_FINAL_ANSWER_NUDGE: &str = "You have not made new progress after repeated tool-use nudges. Stop using tools now and give the best final answer from the evidence already in the conversation. If the task cannot be completed from that evidence, say exactly what is missing.";
/// Sent when a turn reaches its configured step cap: one final tool-free round
/// so the model reports where it left the work instead of the turn dying
/// mid-flight with no answer. Only a deliberate override (`--max-steps`,
/// `/config steps <n>`, or an internal subagent budget) can trigger it.
pub(super) const STEP_LIMIT_WRAP_UP_NUDGE: &str = "You have reached this turn's step limit. Stop using tools now. In a short final answer, report what you completed, what remains unfinished, and the exact state you are leaving the work in (files changed, checks not yet run). Do not claim the task is complete unless it actually is; the user can raise or remove the limit with /config steps.";
pub(super) const TOOL_LIMIT_WRAP_UP_NUDGE: &str = "You have reached this turn's tool-call limit. Stop using tools now. In a short final answer, report what you completed, what remains unfinished, and the exact state you are leaving the work in (files changed, checks not yet run). Do not claim the task is complete unless it actually is; the user can raise the limit with --max-tool-calls.";
/// Progress reason shared between the waiting-round recovery (Steer) and the
/// final-answer acceptance paths: it marks the turn as blocked only on live
/// background work, so a status answer is a valid terminal outcome.
pub(super) const AWAITING_BACKGROUND_REASON: &str = "background process is still running";
/// Consecutive waiting rounds tolerated before the turn is steered to end with
/// a status report. This catches fast completions without allowing unbounded
/// model-driven polling.
pub(super) const WAITING_ROUND_BUDGET: u32 = 3;
pub(super) const REPEATED_VALIDATION_DIAGNOSIS_NUDGE: &str = "The same deterministic validation failure survived another edit-and-test cycle. Stop applying variants of the previous patch. Re-read the failing code and trace the relevant state transition from the assertion backward; if an independent explore tool is available, use it for one focused root-cause diagnosis before editing again. Then make one bounded fix and rerun the narrowest failing validation.";
const FAILED_VALIDATION_LIMIT: usize = 256;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ProgressKind {
    Meaningful,
    Weak,
    None,
}

impl ProgressKind {
    pub(super) fn as_str(self) -> &'static str {
        match self {
            Self::Meaningful => "meaningful",
            Self::Weak => "weak",
            Self::None => "none",
        }
    }
}

#[derive(Clone, Debug)]
pub(super) struct ToolProgressLabel {
    pub(super) kind: ProgressKind,
    pub(super) reason: String,
    pub(super) signature: Option<String>,
    validation_scope: Option<String>,
    validation_selector: Option<String>,
    validation_failure: Option<String>,
}

impl ToolProgressLabel {
    pub(super) fn new(
        kind: ProgressKind,
        reason: impl Into<String>,
        signature: Option<String>,
    ) -> Self {
        Self {
            kind,
            reason: reason.into(),
            signature,
            validation_scope: None,
            validation_selector: None,
            validation_failure: None,
        }
    }

    fn validation(
        kind: ProgressKind,
        reason: impl Into<String>,
        coverage: ValidationCoverage,
        failure: Option<String>,
    ) -> Self {
        Self {
            kind,
            reason: reason.into(),
            signature: failure.clone(),
            validation_scope: Some(coverage.scope),
            validation_selector: coverage.selector,
            validation_failure: failure,
        }
    }
}

#[derive(Clone, Debug)]
struct ValidationFailureProgress {
    scope: String,
    selector: Option<String>,
    signature: String,
    repeats: u32,
    diagnosis_nudged: bool,
    mutation_epoch: u32,
}

#[derive(Clone, Debug, Default)]
pub(super) struct ProgressTracker {
    pub(super) no_progress_streak: u32,
    pub(super) no_progress_nudges: u32,
    pub(super) forced_final_answer_attempts: u32,
    pub(super) last_progress_reason: String,
    pub(super) last_no_progress_reason: String,
    /// Consecutive tool rounds that only watched still-running background work.
    pub(super) waiting_rounds: u32,
    /// Sticky once the waiting budget is spent, until a round does real work.
    pub(super) awaiting_background: bool,
    /// Extra recoveries after a no-progress budget was spent (`max_keep_working`).
    pub(super) keep_working_rounds: u32,
    /// Signature of the no-progress event that last consumed keep-working.
    pub(super) keep_working_blocked_signature: Option<String>,
    /// Whether a tool ran since the last keep-working recovery.
    pub(super) saw_tool_since_keep_working: bool,
    /// Cross-round repeat-loop state lives here so the newer owned turn bag can
    /// retain its construction shape while preserving the established guards.
    pub(super) repeat_sampling_rounds: u32,
    pub(super) force_no_progress_final_answer_next: bool,
    /// The provider repeated a semantically empty completion claim through
    /// the bounded answer-repair budget while durable plan work remained.
    /// Settlement owns this as a typed no-progress outcome; it is not a
    /// provider transport or verification-infrastructure failure.
    pub(super) bounded_plan_answer_recovery_exhausted: bool,
    pub(super) prev_added_no_evidence: bool,
    pub(super) prev_call_sig: Option<Vec<(String, String)>>,
    pub(super) tool_guardrail: ToolLoopGuardrail,
    /// Failed model-authored validation commands keyed by validator family and
    /// any narrowing selector. Entries survive unrelated or narrower green
    /// validators so they cannot erase a still-failing broader trajectory.
    failed_validations: BTreeMap<String, ValidationFailureProgress>,
    #[cfg_attr(not(test), allow(dead_code))]
    failed_validations_dropped: u64,
    pub(super) mutation_epoch: u32,
    /// Bounded diagnostic trail. Correctness-relevant plan-drive evidence is
    /// pinned separately so middle compaction cannot turn productive work into
    /// a false stall.
    pub(super) events: ProgressEventLog,
    plan_drive_progress_event: Option<ProgressEvent>,
    /// Complete hashed read/search identities for cross-turn drive correctness.
    /// Unlike the diagnostic event trail, this set is never head/tail compacted.
    drive_evidence_hashes: HashSet<String>,
}

impl ProgressTracker {
    pub(super) fn push_event(
        &mut self,
        kind: ProgressKind,
        reason: impl Into<String>,
        signature: Option<String>,
    ) {
        let event = ProgressEvent {
            kind: kind.as_str().to_string(),
            reason: reason.into(),
            signature,
        };
        if self.plan_drive_progress_event.is_none()
            && crate::plan_drive::progress_event_counts_as_plan_drive(&event)
        {
            self.plan_drive_progress_event = Some(event.clone());
        }
        if crate::plan_drive::progress_event_is_drive_evidence(&event)
            && let Some(signature) = event.signature.as_deref()
        {
            self.drive_evidence_hashes
                .insert(crate::plan_drive::hash_drive_evidence_signature(signature));
        }
        self.events.push(event);
    }

    /// Materialize the bounded trail for reports/settlement. If the one
    /// correctness-relevant plan-drive event fell in the compacted middle,
    /// reinsert that exact event and evict one non-prefix diagnostic instead.
    pub(super) fn retained_events(&self) -> Vec<ProgressEvent> {
        let mut events = self.events.to_vec();
        let Some(pinned) = self.plan_drive_progress_event.as_ref() else {
            return events;
        };
        if events.iter().any(|event| event == pinned) {
            return events;
        }
        if events.len() >= super::retention::PROGRESS_EVENT_LIMIT {
            events.remove(super::retention::PROGRESS_EVENT_HEAD);
        }
        events.insert(super::retention::PROGRESS_EVENT_HEAD, pinned.clone());
        events
    }

    pub(super) fn retained_events_dropped(&self) -> u64 {
        let retained = self.retained_events().len() as u64;
        self.events.total().saturating_sub(retained)
    }

    pub(super) fn drive_evidence_hashes(&self) -> Vec<String> {
        let mut hashes = self
            .drive_evidence_hashes
            .iter()
            .cloned()
            .collect::<Vec<_>>();
        hashes.sort_unstable();
        hashes
    }

    pub(super) fn record(
        &mut self,
        kind: ProgressKind,
        reason: impl Into<String>,
        signature: Option<String>,
    ) {
        let reason = reason.into();
        match kind {
            ProgressKind::Meaningful | ProgressKind::Weak => {
                self.no_progress_streak = 0;
                self.last_progress_reason = reason.clone();
            }
            ProgressKind::None => {
                self.no_progress_streak = self.no_progress_streak.saturating_add(1);
                self.last_no_progress_reason = reason.clone();
            }
        }
        self.push_event(kind, reason, signature);
    }

    /// Spend one keep-working recovery. Returns false when the budget is
    /// exhausted or disabled (`max == 0`).
    pub(super) fn try_keep_working(&mut self, max: u32) -> bool {
        if max == 0 || self.keep_working_rounds >= max {
            return false;
        }
        self.keep_working_rounds = self.keep_working_rounds.saturating_add(1);
        true
    }

    pub(super) fn last_event_signature(&self) -> Option<String> {
        self.events
            .iter()
            .rev()
            .find_map(|event| event.signature.clone())
    }

    pub(super) fn record_no_progress_nudge(
        &mut self,
        reason: impl Into<String>,
        signature: Option<String>,
    ) -> bool {
        if signature.is_some() {
            self.saw_tool_since_keep_working = true;
        }
        self.no_progress_nudges = self.no_progress_nudges.saturating_add(1);
        self.record(ProgressKind::None, reason, signature);
        self.no_progress_nudges >= NO_PROGRESS_FINAL_ANSWER_NUDGE_THRESHOLD
            && self.forced_final_answer_attempts == 0
    }

    pub(super) fn record_tool(&mut self, label: &ToolProgressLabel) {
        self.saw_tool_since_keep_working = true;
        self.push_event(label.kind, label.reason.clone(), label.signature.clone());
    }

    pub(super) fn record_round_from_tools(&mut self, labels: &[ToolProgressLabel]) {
        self.observe_validation_round(labels);
        if let Some(label) = labels
            .iter()
            .find(|label| label.kind == ProgressKind::Meaningful)
        {
            self.record(
                ProgressKind::Meaningful,
                label.reason.clone(),
                label.signature.clone(),
            );
        } else if labels.iter().all(|label| label.kind == ProgressKind::None) {
            self.record(ProgressKind::None, "tool round made no progress", None);
        } else if let Some(label) = labels.first() {
            self.record(
                ProgressKind::Weak,
                label.reason.clone(),
                label.signature.clone(),
            );
        }
    }

    fn observe_validation_round(&mut self, labels: &[ToolProgressLabel]) {
        if labels.iter().any(|label| {
            matches!(
                label.reason.as_str(),
                "substantive edit" | "successful mutation" | "successful delegated mutation"
            )
        }) {
            self.mutation_epoch = self.mutation_epoch.saturating_add(1);
        }
        for label in labels
            .iter()
            .filter(|label| label.reason == "successful validation after mutation")
        {
            let Some(scope) = label.validation_scope.as_ref() else {
                continue;
            };
            let selector = label.validation_selector.as_ref();
            self.failed_validations.retain(|_, progress| {
                progress.scope != *scope
                    || selector.is_some_and(|selector| {
                        progress
                            .selector
                            .as_ref()
                            .is_none_or(|failed| failed != selector)
                    })
            });
        }
        for label in labels
            .iter()
            .filter(|label| label.reason == "validation command failed")
        {
            let (Some(scope), Some(signature)) = (
                label.validation_scope.as_ref(),
                label.validation_failure.as_ref(),
            ) else {
                continue;
            };
            let selector = label.validation_selector.clone();
            let key = validation_failure_key(scope, selector.as_deref());
            match self.failed_validations.get_mut(&key) {
                Some(progress) if progress.signature == *signature => {
                    if progress.mutation_epoch != self.mutation_epoch {
                        progress.repeats = progress.repeats.saturating_add(1);
                        progress.mutation_epoch = self.mutation_epoch;
                    }
                }
                Some(progress) => {
                    *progress = ValidationFailureProgress {
                        scope: scope.clone(),
                        selector,
                        signature: signature.clone(),
                        repeats: 1,
                        diagnosis_nudged: false,
                        mutation_epoch: self.mutation_epoch,
                    };
                }
                None => {
                    if self.failed_validations.len() >= FAILED_VALIDATION_LIMIT
                        && let Some(evicted) = self.failed_validations.keys().next().cloned()
                    {
                        self.failed_validations.remove(&evicted);
                        self.failed_validations_dropped =
                            self.failed_validations_dropped.saturating_add(1);
                    }
                    self.failed_validations.insert(
                        key,
                        ValidationFailureProgress {
                            scope: scope.clone(),
                            selector,
                            signature: signature.clone(),
                            repeats: 1,
                            diagnosis_nudged: false,
                            mutation_epoch: self.mutation_epoch,
                        },
                    );
                }
            }
        }
    }

    pub(super) fn take_repeated_validation_diagnosis(&mut self) -> bool {
        let Some(progress) = self
            .failed_validations
            .values_mut()
            .find(|progress| progress.repeats >= 2 && !progress.diagnosis_nudged)
        else {
            return false;
        };
        progress.diagnosis_nudged = true;
        true
    }

    pub(super) fn repeated_validation_repair_exhausted(&self) -> Option<String> {
        self.failed_validations
            .iter()
            .find(|(_, progress)| progress.diagnosis_nudged && progress.repeats >= 3)
            .map(|(scope, progress)| format!("{scope}\u{1f}{}", progress.signature))
    }

    pub(super) fn record_final_answer(&mut self) {
        self.record(ProgressKind::Meaningful, "accepted final answer", None);
    }

    pub(super) fn record_forced_final_answer_attempt(&mut self) {
        self.forced_final_answer_attempts = self.forced_final_answer_attempts.saturating_add(1);
    }
}

pub(super) fn no_progress_signature_for_calls(
    calls: &[(String, String, String)],
) -> Option<String> {
    calls.iter().find_map(|(_, name, args)| {
        inspection_signature(name, args)
            .or_else(|| bash_no_progress_signature(args).map(|sig| format!("bash:{sig}")))
    })
}

pub(super) fn forced_final_answer_is_unusable(text: &str, plan_incomplete: bool) -> bool {
    let trimmed = text.trim();
    if trimmed.is_empty() || plan_incomplete || looks_like_unfinished_step(trimmed) {
        return true;
    }
    parse_text_tool_calls(trimmed, 0)
        .iter()
        .any(|content| matches!(content, Content::ToolCall { .. }))
}

pub(super) fn signature_seen(evidence: &EvidenceTracker, signature: &Option<String>) -> bool {
    signature
        .as_ref()
        .is_some_and(|signature| evidence.has_seen_signature(signature))
}

pub(super) fn background_handle_terminal(name: &str, output: &str) -> bool {
    match name {
        "bash_output" => output
            .lines()
            .next()
            .is_some_and(|status| status.contains(": exited") || status.contains(": killed")),
        "bash_kill" => {
            output.starts_with('[')
                && (output.contains("] killed")
                    || output.contains("] already exited")
                    || output.contains("] already killed"))
        }
        _ => false,
    }
}

/// Coarse validator identity used only for cross-edit convergence. Combined
/// commands keep every family so an unrelated green command cannot clear it;
/// narrowing selectors are tracked separately from this family name.
fn contains_command_phrase(command: &str, phrase: &str) -> bool {
    command.match_indices(phrase).any(|(start, matched)| {
        let before = command[..start].chars().next_back();
        let after = command[start + matched.len()..].chars().next();
        before.is_none_or(|character| !character.is_ascii_alphanumeric() && character != '_')
            && after.is_none_or(|character| !character.is_ascii_alphanumeric() && character != '_')
    })
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ValidationCoverage {
    scope: String,
    /// A normalized narrowing selector within `scope`. `None` means the
    /// validator covered the whole family (for example plain `cargo test`).
    selector: Option<String>,
}

fn validation_failure_key(scope: &str, selector: Option<&str>) -> String {
    selector.map_or_else(
        || scope.to_string(),
        |selector| format!("{scope}\u{1e}{selector}"),
    )
}

/// Return the parts of a `cargo test` invocation that narrow which tests are
/// covered. Presentation/execution flags such as `--quiet`, `--release`, and
/// `--nocapture` are deliberately ignored. The result is conservative:
/// unknown flags count as selectors, because retaining a stale failure is
/// safer than letting a narrow green command erase a broader red one.
fn cargo_test_selector(command: &str) -> Option<String> {
    let lower = command.to_ascii_lowercase();
    let (start, matched) = lower.match_indices("cargo test").find(|(start, matched)| {
        let before = lower[..*start].chars().next_back();
        let after = lower[*start + matched.len()..].chars().next();
        before.is_none_or(|character| !character.is_ascii_alphanumeric() && character != '_')
            && after.is_none_or(|character| !character.is_ascii_alphanumeric() && character != '_')
    })?;
    let rest = &lower[start + matched.len()..];
    let end = ["&&", "||", ";", "|", "\n"]
        .into_iter()
        .filter_map(|separator| rest.find(separator))
        .min()
        .unwrap_or(rest.len());
    let words = rest[..end]
        .split_whitespace()
        .map(|word| {
            word.trim_matches(|character| matches!(character, '\'' | '"' | '(' | ')' | '[' | ']'))
        })
        .filter(|word| !word.is_empty())
        .collect::<Vec<_>>();

    let mut selectors = Vec::new();
    let mut index = 0;
    let mut harness_args = false;
    while index < words.len() {
        let word = words[index];
        if word == "--" {
            harness_args = true;
            index += 1;
            continue;
        }

        if harness_args {
            if matches!(word, "--nocapture" | "--show-output") {
                index += 1;
                continue;
            }
            if matches!(word, "--color" | "--format" | "--test-threads") {
                index = (index + 2).min(words.len());
                continue;
            }
            if word.starts_with("--color=")
                || word.starts_with("--format=")
                || word.starts_with("--test-threads=")
            {
                index += 1;
                continue;
            }
            selectors.push(format!("harness:{word}"));
            index += 1;
            continue;
        }

        if matches!(
            word,
            "-p" | "--package"
                | "--exclude"
                | "--manifest-path"
                | "--bin"
                | "--example"
                | "--test"
                | "--bench"
        ) {
            let value = words.get(index + 1).copied().unwrap_or("<missing>");
            selectors.push(format!("{word}={value}"));
            index = (index + 2).min(words.len());
            continue;
        }
        if matches!(
            word,
            "--lib" | "--bins" | "--examples" | "--tests" | "--benches" | "--doc"
        ) || word.starts_with("-p") && word.len() > 2
            || [
                "--package=",
                "--exclude=",
                "--manifest-path=",
                "--bin=",
                "--example=",
                "--test=",
                "--bench=",
            ]
            .iter()
            .any(|prefix| word.starts_with(prefix))
        {
            selectors.push(word.to_string());
            index += 1;
            continue;
        }

        if matches!(
            word,
            "--features"
                | "--target"
                | "--target-dir"
                | "--jobs"
                | "-j"
                | "--profile"
                | "--color"
                | "--message-format"
                | "--config"
                | "-z"
        ) {
            index = (index + 2).min(words.len());
            continue;
        }
        if matches!(
            word,
            "--quiet"
                | "-q"
                | "--verbose"
                | "-v"
                | "--workspace"
                | "--all"
                | "--all-targets"
                | "--all-features"
                | "--no-default-features"
                | "--release"
                | "--locked"
                | "--offline"
                | "--frozen"
                | "--keep-going"
                | "--no-run"
                | "--future-incompat-report"
        ) || word.starts_with("--features=")
            || word.starts_with("--target=")
            || word.starts_with("--target-dir=")
            || word.starts_with("--jobs=")
            || word.starts_with("-j") && word.len() > 2
            || word.starts_with("--profile=")
            || word.starts_with("--color=")
            || word.starts_with("--message-format=")
            || word.starts_with("--config=")
            || word.starts_with("-z") && word.len() > 2
        {
            index += 1;
            continue;
        }

        selectors.push(if word.starts_with('-') {
            format!("flag:{word}")
        } else {
            format!("filter:{word}")
        });
        index += 1;
    }

    (!selectors.is_empty()).then(|| selectors.join(" "))
}

fn validation_coverage(arguments: &str) -> Option<ValidationCoverage> {
    let command = crate::steering::bash_command(arguments)?;
    let compact = command
        .to_ascii_lowercase()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    let mut families = Vec::new();
    for (needle, family) in [
        ("cargo test", "cargo:test"),
        ("cargo check", "cargo:check"),
        ("cargo build", "cargo:build"),
        ("cargo clippy", "cargo:clippy"),
        ("npm run test", "npm:test"),
        ("npm test", "npm:test"),
        ("npm run build", "npm:build"),
        ("npm run check", "npm:check"),
        ("npm run lint", "npm:lint"),
        ("pnpm test", "pnpm:test"),
        ("pnpm build", "pnpm:build"),
        ("pnpm check", "pnpm:check"),
        ("pnpm lint", "pnpm:lint"),
        ("yarn test", "yarn:test"),
        ("yarn build", "yarn:build"),
        ("bun test", "bun:test"),
        ("bun run build", "bun:build"),
        ("pytest", "pytest"),
        ("go test", "go:test"),
        ("make test", "make:test"),
        ("make check", "make:check"),
        ("make build", "make:build"),
        ("just test", "just:test"),
        ("just check", "just:check"),
        ("just build", "just:build"),
        ("cargo run", "cargo:run"),
        ("true # validate", "fixture:validate"),
    ] {
        if contains_command_phrase(&compact, needle) && !families.contains(&family) {
            families.push(family);
        }
    }
    let scope = if families.is_empty() {
        format!("command:{}", hi_policy::normalize_command(&command))
    } else {
        families.join("+")
    };
    let selector = (scope == "cargo:test")
        .then(|| cargo_test_selector(&command))
        .flatten();
    Some(ValidationCoverage { scope, selector })
}

#[cfg(test)]
fn validation_scope(arguments: &str) -> Option<String> {
    validation_coverage(arguments).map(|coverage| coverage.scope)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn classify_tool_progress(
    name: &str,
    arguments: &str,
    output: &str,
    error: bool,
    validation_succeeded: bool,
    signature: Option<String>,
    signature_was_seen: bool,
    repeated_idempotent_result: bool,
    tracker_before: &ImplementationTracker,
    plan_changed: bool,
    workspace_root: &std::path::Path,
) -> ToolProgressLabel {
    if plan_changed {
        return ToolProgressLabel::new(ProgressKind::Meaningful, "changed plan state", signature);
    }
    if repeated_idempotent_result {
        return ToolProgressLabel::new(
            ProgressKind::None,
            "repeated idempotent tool output",
            signature,
        );
    }
    if name == "bash" && bash_no_progress_signature(arguments).is_some() {
        return ToolProgressLabel::new(
            ProgressKind::None,
            "semantic no-op bash command",
            signature,
        );
    }
    if signature_was_seen {
        let reason = if matches!(name, "bash_output" | "bash_kill")
            && background_handle_terminal(name, output)
        {
            "stale background handle"
        } else {
            "repeated inspection signature"
        };
        return ToolProgressLabel::new(ProgressKind::None, reason, signature);
    }
    if error {
        if implementation_tool_call_validates(name, arguments)
            && let Some(coverage) = validation_coverage(arguments)
            && let Some(digest) = crate::verify_digest::digest_failure(workspace_root, output)
        {
            let signature = digest
                .signature
                .into_iter()
                .collect::<Vec<_>>()
                .join("\u{1f}");
            return ToolProgressLabel::validation(
                ProgressKind::Weak,
                "validation command failed",
                coverage,
                Some(signature),
            );
        }
        return ToolProgressLabel::new(ProgressKind::Weak, "tool returned an error", signature);
    }
    if implementation_tool_result_landed_substantive_edit(name, arguments, output) {
        return ToolProgressLabel::new(ProgressKind::Meaningful, "substantive edit", signature);
    }
    if implementation_tool_result_landed_mutation(name, arguments, output) {
        return ToolProgressLabel::new(ProgressKind::Meaningful, "successful mutation", signature);
    }
    if tracker_before.mutation_seen
        && validation_succeeded
        && implementation_tool_call_validates(name, arguments)
        && let Some(coverage) = validation_coverage(arguments)
    {
        return ToolProgressLabel::validation(
            ProgressKind::Meaningful,
            "successful validation after mutation",
            coverage,
            None,
        );
    }
    if let Some(kind) = evidence_kind_for_tool(name, arguments) {
        let (progress_kind, reason) = match kind {
            crate::steering::EvidenceKind::FileRead => {
                (ProgressKind::Meaningful, "new file evidence")
            }
            crate::steering::EvidenceKind::TargetedSearch => {
                (ProgressKind::Meaningful, "new targeted search evidence")
            }
            crate::steering::EvidenceKind::Listing => (ProgressKind::Weak, "new listing evidence"),
        };
        return ToolProgressLabel::new(progress_kind, reason, signature);
    }
    if name == "bash" {
        let Some(command) = crate::steering::bash_command(arguments) else {
            return ToolProgressLabel::new(ProgressKind::Weak, "bash command completed", signature);
        };
        let kind = classify_bash_command(&command);
        let reason = format!("bash {} command completed", kind.as_str());
        return ToolProgressLabel::new(ProgressKind::Weak, reason, signature);
    }
    ToolProgressLabel::new(ProgressKind::Weak, "tool completed", signature)
}

#[cfg(test)]
mod validation_progress_tests {
    use super::*;

    fn label(kind: ProgressKind, reason: &str, signature: Option<&str>) -> ToolProgressLabel {
        ToolProgressLabel::new(kind, reason, signature.map(str::to_string))
    }

    fn failed_validation(scope: &str, signature: &str) -> ToolProgressLabel {
        failed_validation_with_selector(scope, None, signature)
    }

    fn failed_validation_with_selector(
        scope: &str,
        selector: Option<&str>,
        signature: &str,
    ) -> ToolProgressLabel {
        ToolProgressLabel::validation(
            ProgressKind::Weak,
            "validation command failed",
            ValidationCoverage {
                scope: scope.to_string(),
                selector: selector.map(str::to_string),
            },
            Some(signature.to_string()),
        )
    }

    fn passed_validation(scope: &str) -> ToolProgressLabel {
        passed_validation_with_selector(scope, None)
    }

    fn passed_validation_with_selector(scope: &str, selector: Option<&str>) -> ToolProgressLabel {
        ToolProgressLabel::validation(
            ProgressKind::Meaningful,
            "successful validation after mutation",
            ValidationCoverage {
                scope: scope.to_string(),
                selector: selector.map(str::to_string),
            },
            None,
        )
    }

    #[test]
    fn validation_scope_separates_check_from_test_and_tracks_test_filters() {
        assert_eq!(
            validation_scope(r#"{"command":"cargo test moves::checkmate_detected"}"#).as_deref(),
            Some("cargo:test")
        );
        assert_eq!(
            validation_scope(r#"{"command":"cargo check --workspace"}"#).as_deref(),
            Some("cargo:check")
        );
        assert_eq!(
            validation_coverage(r#"{"command":"cargo test --quiet moves::checkmate_detected"}"#)
                .and_then(|coverage| coverage.selector)
                .as_deref(),
            Some("filter:moves::checkmate_detected")
        );
        assert_eq!(
            validation_coverage(r#"{"command":"cargo test --workspace --quiet"}"#)
                .and_then(|coverage| coverage.selector),
            None
        );
    }

    #[test]
    fn plan_drive_progress_is_pinned_across_bounded_middle_compaction() {
        let mut tracker = ProgressTracker::default();
        for index in 0..400 {
            let (kind, reason) = if index == 100 {
                (ProgressKind::Meaningful, "substantive edit".to_string())
            } else {
                (ProgressKind::None, format!("no-progress event {index}"))
            };
            tracker.record(kind, reason, Some(format!("event-{index:03}")));
        }

        let retained = tracker.retained_events();
        assert_eq!(
            retained.len(),
            super::super::retention::PROGRESS_EVENT_LIMIT
        );
        assert_eq!(tracker.retained_events_dropped(), 144);
        assert_eq!(retained[31].signature.as_deref(), Some("event-031"));
        assert_eq!(retained[32].signature.as_deref(), Some("event-100"));
        assert_eq!(retained[33].signature.as_deref(), Some("event-177"));
        assert_eq!(
            retained.last().unwrap().signature.as_deref(),
            Some("event-399")
        );
        assert!(crate::plan_drive_made_progress(
            Some("same step"),
            Some("same step"),
            &retained,
            &[] as &[String],
        ));
    }

    #[test]
    fn drive_evidence_hashes_are_complete_when_diagnostics_compact() {
        let mut tracker = ProgressTracker::default();
        for index in 0..400 {
            tracker.record(
                ProgressKind::Meaningful,
                "new file evidence",
                Some(format!("read:file-{index}:1:default")),
            );
        }

        assert_eq!(
            tracker.retained_events().len(),
            super::super::retention::PROGRESS_EVENT_LIMIT
        );
        assert_eq!(tracker.drive_evidence_hashes().len(), 400);
    }

    #[test]
    fn targeted_validation_pass_does_not_clear_full_suite_failure() {
        let mut tracker = ProgressTracker::default();
        tracker.record_round_from_tools(&[failed_validation("cargo:test", "full-suite-red")]);

        tracker.record_round_from_tools(&[passed_validation_with_selector(
            "cargo:test",
            Some("filter:round_trip"),
        )]);

        assert!(tracker.failed_validations.contains_key("cargo:test"));
    }

    #[test]
    fn full_suite_pass_clears_targeted_validation_failures() {
        let mut tracker = ProgressTracker::default();
        tracker.record_round_from_tools(&[
            failed_validation_with_selector("cargo:test", Some("filter:first_case"), "first-red"),
            failed_validation_with_selector("cargo:test", Some("filter:second_case"), "second-red"),
        ]);

        tracker.record_round_from_tools(&[passed_validation("cargo:test")]);

        assert!(tracker.failed_validations.is_empty());
    }

    #[test]
    fn distinct_failed_validation_selectors_have_bounded_repair_memory() {
        let mut tracker = ProgressTracker::default();
        for index in 0..300 {
            tracker.record_round_from_tools(&[failed_validation_with_selector(
                "cargo:test",
                Some(&format!("filter:test-{index}")),
                &format!("failure-{index}"),
            )]);
        }

        assert_eq!(tracker.failed_validations.len(), FAILED_VALIDATION_LIMIT);
        assert_eq!(tracker.failed_validations_dropped, 44);
        assert!(
            tracker
                .failed_validations
                .contains_key(&validation_failure_key(
                    "cargo:test",
                    Some("filter:test-299")
                )),
            "new validation evidence must still be tracked after diagnostic eviction"
        );
    }

    #[test]
    fn matching_targeted_pass_clears_only_matching_targeted_failure() {
        let mut tracker = ProgressTracker::default();
        tracker.record_round_from_tools(&[
            failed_validation_with_selector("cargo:test", Some("filter:first_case"), "first-red"),
            failed_validation_with_selector("cargo:test", Some("filter:second_case"), "second-red"),
        ]);

        tracker.record_round_from_tools(&[passed_validation_with_selector(
            "cargo:test",
            Some("filter:first_case"),
        )]);

        assert!(
            !tracker
                .failed_validations
                .contains_key(&validation_failure_key(
                    "cargo:test",
                    Some("filter:first_case")
                ))
        );
        assert!(
            tracker
                .failed_validations
                .contains_key(&validation_failure_key(
                    "cargo:test",
                    Some("filter:second_case")
                ))
        );
    }

    #[test]
    fn repeated_validation_failure_survives_intervening_edit_and_nudges_once() {
        let mut tracker = ProgressTracker::default();
        tracker.record_round_from_tools(&[failed_validation(
            "cargo:test",
            "test:moves::checkmate_detected:state-a",
        )]);
        assert!(!tracker.take_repeated_validation_diagnosis());
        tracker.record_round_from_tools(&[passed_validation("cargo:check")]);
        assert!(tracker.failed_validations.contains_key("cargo:test"));
        tracker.record_round_from_tools(&[failed_validation(
            "cargo:test",
            "test:moves::checkmate_detected:state-a",
        )]);
        assert!(
            !tracker.take_repeated_validation_diagnosis(),
            "rerunning the same failure without an intervening edit is not a second repair cycle"
        );

        tracker.record_round_from_tools(&[label(
            ProgressKind::Meaningful,
            "successful mutation",
            None,
        )]);
        tracker.record_round_from_tools(&[failed_validation(
            "cargo:test",
            "test:moves::checkmate_detected:state-a",
        )]);
        assert!(tracker.take_repeated_validation_diagnosis());
        assert!(!tracker.take_repeated_validation_diagnosis());

        tracker.record_round_from_tools(&[label(
            ProgressKind::Meaningful,
            "successful mutation",
            None,
        )]);
        tracker.record_round_from_tools(&[failed_validation(
            "cargo:test",
            "test:moves::checkmate_detected:state-a",
        )]);
        assert!(tracker.repeated_validation_repair_exhausted().is_some());

        tracker.record_round_from_tools(&[label(
            ProgressKind::Meaningful,
            "successful mutation",
            None,
        )]);
        tracker.record_round_from_tools(&[failed_validation(
            "cargo:test",
            "test:moves::checkmate_detected:state-b",
        )]);
        assert_eq!(tracker.failed_validations["cargo:test"].repeats, 1);
        assert!(!tracker.take_repeated_validation_diagnosis());
        assert!(tracker.repeated_validation_repair_exhausted().is_none());

        tracker.record_round_from_tools(&[passed_validation("cargo:test")]);
        assert!(!tracker.failed_validations.contains_key("cargo:test"));
        assert!(tracker.repeated_validation_repair_exhausted().is_none());
    }
}
