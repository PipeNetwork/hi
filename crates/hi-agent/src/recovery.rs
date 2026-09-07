//! One persisted recovery allowance for a user task and its automatic drives.

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use serde::{Deserialize, Serialize};

pub const DEFAULT_RECOVERY_INTERVENTIONS: u32 = 3;
const RECOVERY_SCHEMA_VERSION: u16 = 2;
const FAILED_STATE_HISTORY: usize = 128;
const COMPLETED_OBSERVATIONS: usize = 4096;

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
struct ValidationFrontier {
    best_failures: Option<BTreeSet<String>>,
    passed: bool,
    failed_states: Vec<String>,
    #[serde(default)]
    diagnostic_states: Vec<String>,
    #[serde(default)]
    current_failure: Option<CurrentValidationFailure>,
}

impl<'de> Deserialize<'de> for ValidationFrontier {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        struct StoredFrontier {
            best_failures: Option<BTreeSet<String>>,
            passed: bool,
            failed_states: Vec<String>,
            #[serde(default)]
            diagnostic_states: Vec<String>,
            #[serde(default, deserialize_with = "present")]
            current_failure: Option<Option<CurrentValidationFailure>>,
        }
        fn present<'de, D: serde::Deserializer<'de>>(
            deserializer: D,
        ) -> Result<Option<Option<CurrentValidationFailure>>, D::Error> {
            Option::deserialize(deserializer).map(Some)
        }
        let stored = StoredFrontier::deserialize(deserializer)?;
        let current_failure = match stored.current_failure {
            Some(current) => current,
            None => {
                // V1 retained historical best evidence, but no latest pass order.
                // Ambiguous fail/pass histories require revalidation; keep their
                // transcript usable and do not claim that old bytes still fail.
                let latest = stored
                    .failed_states
                    .last()
                    .map(|state| {
                        serde_json::from_str::<(String, Option<BTreeSet<String>>)>(state)
                            .map_err(<D::Error as serde::de::Error>::custom)
                    })
                    .transpose()?;
                latest
                    .map(|(revision, diagnostics)| CurrentValidationFailure {
                        input_revision: if stored.passed {
                            String::new()
                        } else {
                            revision
                        },
                        diagnostics,
                    })
                    .or_else(|| {
                        stored
                            .best_failures
                            .clone()
                            .map(|diagnostics| CurrentValidationFailure {
                                input_revision: String::new(),
                                diagnostics: Some(diagnostics),
                            })
                    })
            }
        };
        Ok(Self {
            best_failures: stored.best_failures,
            passed: stored.passed,
            failed_states: stored.failed_states,
            diagnostic_states: stored.diagnostic_states,
            current_failure,
        })
    }
}

/// A successful check in another scope cannot discharge this obligation.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
struct CurrentValidationFailure {
    input_revision: String,
    diagnostics: Option<BTreeSet<String>>,
}

/// One observation format for shell tools, fast feedback and final verification.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct ValidationObservation {
    pub execution_id: String,
    pub scope: String,
    pub input_revision: String,
    pub status: ValidationResult,
    pub diagnostics: Option<BTreeSet<String>>,
    pub required_stage: bool,
    /// Proven aliases from native execution only; never accepted from replay wire data.
    #[serde(skip)]
    pub equivalent_scopes: Vec<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) enum ValidationResult {
    Passed,
    Failed,
    Infrastructure,
    Deferred,
}

impl ValidationObservation {
    pub(crate) fn command(
        execution_id: String,
        command: &str,
        input_revision: String,
        status: ValidationResult,
        output: &str,
        root: &std::path::Path,
        required_stage: bool,
    ) -> Self {
        let diagnostics = if status == ValidationResult::Failed {
            crate::verify_digest::digest_failure(root, output).map(|digest| digest.signature)
        } else {
            None
        };
        Self {
            execution_id,
            // Interior whitespace can be data inside a quoted script, pattern,
            // or file name. Distinct checks must never share a passing verdict.
            scope: command.trim().to_owned(),
            input_revision,
            status,
            diagnostics,
            required_stage,
            equivalent_scopes: Vec::new(),
        }
    }
}

/// Remove only incidental source positions, temporary paths and timing lines.
/// Diagnostic values (including assertion numbers) retain their meaning.
pub(crate) fn normalize_diagnostic(value: &str) -> String {
    value
        .lines()
        .filter(|line| {
            !line.contains("finished in ") && !line.trim_start().starts_with("Finished ")
        })
        .flat_map(|line| line.split_whitespace())
        .map(|token| {
            if token.starts_with("/tmp/")
                || token.starts_with("/private/tmp/")
                || token.starts_with("/private/var/folders/")
            {
                return "<temporary-path>".to_owned();
            }
            let mut stable = token.trim_end_matches(':');
            while let Some((prefix, suffix)) = stable.rsplit_once(':') {
                if !suffix.is_empty()
                    && suffix.chars().all(|c| c.is_ascii_digit())
                    && (prefix.contains('/') || prefix.contains('.'))
                {
                    stable = prefix;
                } else {
                    break;
                }
            }
            stable.to_owned()
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// Durable state: changing a prompt marker or resuming a drive cannot buy retries.
#[derive(Clone, Debug)]
pub struct TaskRecoveryState {
    pub schema_version: u16,
    pub objective: String,
    pub limit: u32,
    pub remaining: u32,
    pub exhausted: bool,
    pub interventions: u64,
    pub last_reason: Option<String>,
    pending_validation_correction: bool,
    completed_effects: BTreeSet<String>,
    mutation_credited: bool,
    validations: BTreeMap<String, ValidationFrontier>,
    // Cross-process execution repair is deduplicated by SessionReducer and the
    // workspace receipt. This bounded callback cache is only needed in-process;
    // serializing it in every recovery record would duplicate completed work.
    observed_executions: VecDeque<String>,
    legacy_wire: Option<compat::LegacyRecoveryWire>,
}

// Callback deduplication and old wire bytes are caches, not durable task state.
impl PartialEq for TaskRecoveryState {
    fn eq(&self, other: &Self) -> bool {
        self.schema_version == other.schema_version
            && self.objective == other.objective
            && self.limit == other.limit
            && self.remaining == other.remaining
            && self.exhausted == other.exhausted
            && self.interventions == other.interventions
            && self.last_reason == other.last_reason
            && self.pending_validation_correction == other.pending_validation_correction
            && self.completed_effects == other.completed_effects
            && self.mutation_credited == other.mutation_credited
            && self.validations == other.validations
    }
}
impl Eq for TaskRecoveryState {}

impl Default for TaskRecoveryState {
    fn default() -> Self {
        Self::new(String::new(), DEFAULT_RECOVERY_INTERVENTIONS)
    }
}

impl TaskRecoveryState {
    pub fn new(objective: String, limit: u32) -> Self {
        Self {
            schema_version: RECOVERY_SCHEMA_VERSION,
            objective,
            limit,
            remaining: limit,
            exhausted: false,
            interventions: 0,
            last_reason: None,
            pending_validation_correction: false,
            completed_effects: BTreeSet::new(),
            mutation_credited: false,
            validations: BTreeMap::new(),
            observed_executions: VecDeque::new(),
            legacy_wire: None,
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        if !matches!(self.schema_version, 1 | RECOVERY_SCHEMA_VERSION) {
            return Err(format!(
                "unsupported task recovery schema {}",
                self.schema_version
            ));
        }
        if self.remaining > self.limit {
            return Err("task recovery remaining allowance exceeds its limit".into());
        }
        if self.exhausted && self.remaining != 0 {
            return Err("exhausted task recovery has a remaining allowance".into());
        }
        Ok(())
    }

    /// Call only after any enclosing historical snapshot digest was checked.
    pub(crate) fn migrate(&mut self) {
        if self.schema_version == 1 {
            self.schema_version = RECOVERY_SCHEMA_VERSION;
            self.legacy_wire = None;
            self.observed_executions.clear();
        }
    }

    /// A correction consumes one allowance before its next model dispatch.
    pub(crate) fn intervene(&mut self, reason: impl Into<String>) -> bool {
        self.migrate();
        if self.exhausted {
            return false;
        }
        self.last_reason = Some(reason.into());
        if self.remaining == 0 {
            self.stop("automatic recovery exhausted without objective improvement");
            return false;
        }
        self.remaining -= 1;
        self.interventions = self.interventions.saturating_add(1);
        true
    }

    pub(crate) fn stop(&mut self, reason: impl Into<String>) {
        self.migrate();
        self.pending_validation_correction = false;
        if self.exhausted {
            return;
        }
        self.exhausted = true;
        self.remaining = 0;
        self.last_reason = Some(reason.into());
    }

    fn improved(&mut self) {
        self.migrate();
        // A terminal decision is absorbing. A later callback cannot reopen it.
        if !self.exhausted {
            self.remaining = self.limit;
            self.pending_validation_correction = false;
        }
    }

    pub(crate) fn request_correction(&mut self, reason: &str) {
        self.migrate();
        if self.exhausted {
            return;
        }
        self.pending_validation_correction = true;
        self.last_reason = Some(reason.to_owned());
        if self.remaining == 0 {
            self.stop("automatic recovery exhausted without objective improvement");
        }
    }

    pub(crate) fn observe_required_effect(&mut self, identity: String) {
        self.migrate();
        if self.completed_effects.insert(identity) {
            self.improved();
        }
    }

    pub(crate) fn observe_requested_mutation(&mut self) {
        self.migrate();
        if !self.mutation_credited {
            self.mutation_credited = true;
            self.improved();
        }
    }

    pub(crate) fn observe(&mut self, observation: &ValidationObservation) {
        self.migrate();
        if self.observed_executions.contains(&observation.execution_id) {
            return;
        }
        self.observed_executions
            .push_back(observation.execution_id.clone());
        if self.observed_executions.len() > COMPLETED_OBSERVATIONS {
            self.observed_executions.pop_front();
        }
        if matches!(
            observation.status,
            ValidationResult::Infrastructure | ValidationResult::Deferred
        ) {
            return;
        }
        let improved = self.observe_validation(
            &observation.scope,
            &observation.input_revision,
            observation.diagnostics.clone(),
            observation.status == ValidationResult::Passed,
            observation.required_stage,
        );
        if observation.status == ValidationResult::Passed {
            for alias in &observation.equivalent_scopes {
                self.observe_validation(alias, &observation.input_revision, None, true, false);
            }
        }
        if observation.status == ValidationResult::Failed && !improved {
            self.request_correction("validation repair");
        }
    }

    pub(crate) fn unresolved_validation_status(&self, revision: &str) -> Option<ValidationResult> {
        let mut unresolved = false;
        for failure in self
            .validations
            .values()
            .filter_map(|frontier| frontier.current_failure.as_ref())
        {
            unresolved = true;
            if failure.input_revision == revision {
                return Some(ValidationResult::Failed);
            }
        }
        unresolved.then_some(ValidationResult::Deferred)
    }

    pub(crate) fn unresolved_validation_summary(&self, current_revision: &str) -> Option<String> {
        let lines: Vec<_> = self.validations.iter().filter_map(|(scope, frontier)| {
            let failure = frontier.current_failure.as_ref()?;
            let diagnostics = failure.diagnostics.as_ref()
                .map(|items| items.iter().take(3).map(|item| crate::verify_digest::display_signature_item(item)).collect::<Vec<_>>().join("; "));
            let applicability = if failure.input_revision.is_empty() {
                "Saved evidence cannot establish the latest check result; revalidation is required.".to_owned()
            } else if failure.input_revision != current_revision {
                format!("Earlier failure at workspace revision {}. The workspace changed after this check; the current revision remains unverified.", failure.input_revision.chars().take(12).collect::<String>())
            } else {
                "This check failed for the current workspace revision.".to_owned()
            };
            Some(format!("Unresolved check: {}. {}{}", scope.chars().take(200).collect::<String>(),
                applicability, diagnostics.map(|text| format!(" Diagnostics: {text}")).unwrap_or_default())
                .chars().take(800).collect::<String>())
        }).take(3).collect();
        (!lines.is_empty()).then(|| lines.join("\n"))
    }

    /// Exact scope matching intentionally does not infer that a narrow check
    /// proves a broader one. Unrecognized diagnostics never establish progress.
    pub(crate) fn observe_validation(
        &mut self,
        scope: &str,
        input_revision: &str,
        failures: Option<BTreeSet<String>>,
        passed: bool,
        required_stage: bool,
    ) -> bool {
        self.migrate();
        if passed && !required_stage && !self.validations.contains_key(scope) {
            return false;
        }
        let mut improvement = false;
        let frontier = self.validations.entry(scope.to_owned()).or_default();
        if passed {
            let repaired_failure = frontier.current_failure.take().is_some();
            if !frontier.passed && (required_stage || repaired_failure) {
                frontier.passed = true;
                improvement = true;
            }
        } else {
            frontier.current_failure = Some(CurrentValidationFailure {
                input_revision: input_revision.to_owned(),
                diagnostics: failures.clone(),
            });
            // Bind failure identity to input bytes as well as diagnostic scope.
            let failed_state =
                serde_json::to_string(&(input_revision, &failures)).expect("string state");
            if let Some(failures) = &failures {
                let identity = serde_json::to_string(failures).expect("string set");
                if frontier.diagnostic_states.last() != Some(&identity) {
                    if frontier.diagnostic_states.contains(&identity) {
                        self.stop("verification returned to a previous failure set");
                        return false;
                    }
                    frontier.diagnostic_states.push(identity);
                    if frontier.diagnostic_states.len() > FAILED_STATE_HISTORY {
                        frontier.diagnostic_states.remove(0);
                    }
                }
            }
            if frontier.failed_states.contains(&failed_state) {
                self.stop("verification revisited an unsuccessful workspace state");
                return false;
            }
            frontier.failed_states.push(failed_state);
            if frontier.failed_states.len() > FAILED_STATE_HISTORY {
                frontier.failed_states.remove(0);
            }
            if let Some(failures) = failures.filter(|failures| !failures.is_empty()) {
                if !frontier.passed
                    && frontier
                        .best_failures
                        .as_ref()
                        .is_some_and(|best| failures.len() < best.len() && failures.is_subset(best))
                {
                    improvement = true;
                    frontier.best_failures = Some(failures);
                } else if frontier.best_failures.is_none() {
                    frontier.best_failures = Some(failures);
                }
            }
        }
        if improvement {
            self.improved();
        }
        improvement
    }
}

impl crate::Agent {
    pub(crate) fn start_task_recovery(&mut self, objective: String) -> anyhow::Result<()> {
        let previous = self.task_recovery.clone();
        let limit = if self.config.loop_limits.max_recovery_interventions
            == DEFAULT_RECOVERY_INTERVENTIONS
        {
            self.config.harness.max_recovery_interventions
        } else {
            self.config.loop_limits.max_recovery_interventions
        };
        self.task_recovery = TaskRecoveryState::new(objective, limit);
        if let Err(error) = self.persist_task_recovery() {
            self.task_recovery = previous;
            return Err(error);
        }
        Ok(())
    }

    pub(crate) async fn persist_task_recovery_async(&mut self) -> anyhow::Result<()> {
        let state = self.task_recovery.clone();
        self.write_session(move |sink| sink.record_task_recovery(&state))
            .await
    }

    pub(crate) async fn start_task_recovery_async(
        &mut self,
        objective: String,
    ) -> anyhow::Result<()> {
        let previous = self.task_recovery.clone();
        let limit = if self.config.loop_limits.max_recovery_interventions
            == DEFAULT_RECOVERY_INTERVENTIONS
        {
            self.config.harness.max_recovery_interventions
        } else {
            self.config.loop_limits.max_recovery_interventions
        };
        self.task_recovery = TaskRecoveryState::new(objective, limit);
        if let Err(error) = self.persist_task_recovery_async().await {
            self.task_recovery = previous;
            return Err(error);
        }
        Ok(())
    }

    /// Explicit user retry starts a new allowance while preserving best evidence.
    /// Automatic drives and restoration must never call this.
    pub fn restart_task_recovery(&mut self) -> anyhow::Result<()> {
        let previous = self.task_recovery.clone();
        let state = &mut self.task_recovery;
        state.exhausted = false;
        state.remaining = state.limit;
        state.interventions = 0;
        state.pending_validation_correction = false;
        for frontier in state.validations.values_mut() {
            frontier.failed_states.clear();
            frontier.diagnostic_states.clear();
        }
        if let Err(error) = self.persist_task_recovery() {
            self.task_recovery = previous;
            return Err(error);
        }
        Ok(())
    }

    pub(crate) async fn admit_recovery_request(&mut self) -> anyhow::Result<bool> {
        let nudge = self.messages.take_recovery_request();
        let validation = std::mem::take(&mut self.task_recovery.pending_validation_correction);
        let allowed = if self.task_recovery.exhausted {
            false
        } else if let Some(reason) = nudge {
            self.task_recovery.intervene(reason)
        } else if validation {
            self.task_recovery.intervene(
                self.task_recovery
                    .last_reason
                    .clone()
                    .unwrap_or_else(|| "validation repair".into()),
            )
        } else {
            true
        };
        self.persist_task_recovery_async().await?;
        Ok(allowed)
    }

    pub(crate) async fn observe_validation(
        &mut self,
        observation: ValidationObservation,
    ) -> anyhow::Result<()> {
        self.task_recovery.observe(&observation);
        self.persist_task_recovery_async().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn failures(names: &[&str]) -> Option<BTreeSet<String>> {
        Some(names.iter().map(|name| (*name).to_owned()).collect())
    }

    #[test]
    fn changing_reasons_or_restoring_a_session_cannot_replenish_recovery() {
        let mut state = TaskRecoveryState::default();
        for reason in ["review", "protocol", "verification"] {
            assert!(state.intervene(reason));
        }
        let mut restored: TaskRecoveryState =
            serde_json::from_str(&serde_json::to_string(&state).unwrap()).unwrap();
        assert!(!restored.intervene("a different label"));
        restored.observe_requested_mutation();
        assert!(restored.exhausted);
    }

    #[test]
    fn terminal_observations_retain_the_stop_cause_and_never_schedule_corrections() {
        let mut state = TaskRecoveryState::default();
        state.request_correction("previous repair");
        state.stop("provider terminated recovery: 4/4 sends");
        for revision in ["old", "current", "old"] {
            state.observe(&ValidationObservation::command(
                format!("check-{revision}-{}", state.observed_executions.len()),
                "cargo check",
                revision.into(),
                ValidationResult::Failed,
                "error[E0382]: partial move\n --> src/server.rs:482:4",
                std::path::Path::new("."),
                true,
            ));
            assert_eq!(
                state.unresolved_validation_status(revision),
                Some(ValidationResult::Failed)
            );
            assert!(!state.pending_validation_correction);
            assert!(!state.intervene("late repair"));
            state.stop("later generic exhaustion");
            assert_eq!(
                state.last_reason.as_deref(),
                Some("provider terminated recovery: 4/4 sends")
            );
        }
        let mut restored: TaskRecoveryState =
            serde_json::from_str(&serde_json::to_string(&state).unwrap()).unwrap();
        restored.observe(&ValidationObservation::command(
            "final-pass".into(),
            "cargo check",
            "repaired".into(),
            ValidationResult::Passed,
            "",
            std::path::Path::new("."),
            true,
        ));
        assert_eq!(restored.unresolved_validation_status("repaired"), None);
        assert!(restored.exhausted);
        assert_eq!(restored.remaining, 0);
        assert!(!restored.pending_validation_correction);
        assert_eq!(
            restored.last_reason.as_deref(),
            Some("provider terminated recovery: 4/4 sends")
        );
    }

    #[test]
    fn only_a_new_best_failure_set_replenishes() {
        let mut state = TaskRecoveryState::default();
        state.observe_validation("full", "a", failures(&["x", "y"]), false, true);
        assert!(state.intervene("fix"));
        state.observe_validation("narrow", "b", None, true, false);
        assert_eq!(state.remaining, 2);
        state.observe_validation("full", "b", failures(&["x"]), false, true);
        assert_eq!(state.remaining, 3);
        assert!(state.intervene("fix"));
        state.observe_validation("full", "c", failures(&["x", "y"]), false, true);
        state.observe_validation("full", "d", failures(&["x"]), false, true);
        assert!(
            state.exhausted,
            "regressing and reclaiming a prior best is a failure cycle"
        );
    }

    #[test]
    fn alternating_failed_inputs_stop() {
        let mut state = TaskRecoveryState::default();
        state.observe_validation("test", "a", failures(&["x"]), false, true);
        assert!(state.intervene("fix"));
        state.observe_validation("test", "b", failures(&["x"]), false, true);
        assert!(state.intervene("fix again"));
        state.observe_validation("test", "a", failures(&["x"]), false, true);
        assert!(state.exhausted);
    }

    #[test]
    fn diagnostic_positions_and_output_order_do_not_replenish_recovery() {
        let mut state = TaskRecoveryState::default();
        let observe = |id: &str, revision: &str, output: &str| {
            ValidationObservation::command(
                id.into(),
                "cargo test --workspace",
                revision.into(),
                ValidationResult::Failed,
                output,
                std::path::Path::new("."),
                true,
            )
        };
        state.observe(&observe(
            "first",
            "a",
            "error[E0308]: mismatched types\n --> src/main.rs:10:2",
        ));
        assert!(state.intervene("repair"));
        state.observe(&observe(
            "second",
            "b",
            "error[E0308]: mismatched types\n --> src/main.rs:90:8",
        ));
        assert_eq!(
            state.remaining, 2,
            "moved source positions do not constitute progress"
        );
        state.observe(&observe(
            "second",
            "b",
            "error[E0308]: mismatched types\n --> src/main.rs:90:8",
        ));
        assert!(
            !state.exhausted,
            "duplicate completion callbacks are ignored"
        );
    }

    #[test]
    fn repeated_infrastructure_and_new_unrelated_green_checks_do_not_buy_repairs() {
        let mut state = TaskRecoveryState::default();
        assert!(state.intervene("repair"));
        for (id, status) in [
            ("infra", ValidationResult::Infrastructure),
            ("defer", ValidationResult::Deferred),
            ("pass", ValidationResult::Passed),
        ] {
            state.observe(&ValidationObservation::command(
                id.into(),
                "unrelated narrow check",
                "a".into(),
                status,
                "",
                std::path::Path::new("."),
                false,
            ));
        }
        assert_eq!(state.remaining, 2);
        assert!(!state.exhausted);
    }

    #[test]
    fn pending_corrections_survive_resume_and_do_not_reopen_exhaustion() {
        let mut state = TaskRecoveryState::default();
        for _ in 0..3 {
            state.request_correction("reviewer objection");
            assert!(!state.exhausted);
            assert!(state.intervene("reviewer objection"));
        }
        let mut restored: TaskRecoveryState =
            serde_json::from_str(&serde_json::to_string(&state).unwrap()).unwrap();
        assert!(restored.pending_validation_correction);
        restored.request_correction("different reviewer label");
        assert!(restored.exhausted);
        restored.observe_required_effect("same already terminal goal".into());
        assert!(restored.exhausted);
    }

    #[test]
    fn a_required_effect_is_credited_once() {
        let mut state = TaskRecoveryState::default();
        assert!(state.intervene("repair"));
        state.observe_required_effect("goal:fixed-objective:0".into());
        assert_eq!(state.remaining, 3);
        assert!(state.intervene("repair"));
        state.observe_required_effect("goal:fixed-objective:0".into());
        assert_eq!(state.remaining, 2);
    }

    #[test]
    fn malformed_or_future_recovery_is_rejected() {
        let mut state = TaskRecoveryState::default();
        state.schema_version += 1;
        assert!(state.validate().is_err());
        state = TaskRecoveryState::default();
        state.remaining += 1;
        assert!(state.validate().is_err());
    }
}

/// Presentation state never substitutes for execution or verification evidence.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum AnswerState {
    #[default]
    Missing,
    Commentary,
    Candidate,
    Accepted,
    Deterministic,
}

impl AnswerState {
    pub(crate) fn is_terminal(self) -> bool {
        matches!(self, Self::Accepted | Self::Deterministic)
    }
}

#[cfg(test)]
#[path = "recovery_validation_tests.rs"]
mod validation_tests;

#[path = "recovery_compat.rs"]
mod compat;
