//! Optional TypeSafe (Jev) supervisor.
//!
//! Jev is a typed decision model, not a coding LLM. The harness asks it
//! parallel yes/no (noul) questions and composes the answers in code:
//! next-action flavor after inspect-only rounds, `/permissions auto` hints,
//! and a turn-scoped reasoning-effort override. Fail-open: a missing key,
//! timeout, or parse error leaves the existing harness policy in place.

use std::time::Duration;

use hi_ai::ReasoningEffort;
use serde_json::{Value, json};

use crate::Harness;

pub const API_KEY_ENV: &str = "TYPESAFE_API_KEY";
pub const BASE_URL_ENV: &str = "TYPESAFE_BASE_URL";
pub const MODEL_ENV: &str = "TYPESAFE_DEFAULT_MODEL";
pub const DEFAULT_BASE_URL: &str = "https://api.typesafe.ai";
pub const DEFAULT_MODEL: &str = "jev-latest";
pub const DEFAULT_MIN_CONFIDENCE: f64 = 0.55;
const GATE_TIMEOUT: Duration = Duration::from_millis(2000);
pub(crate) const FAST_GATE_TIMEOUT: Duration = Duration::from_millis(800);
const MAX_PROMPT_CHARS: usize = 240;
const EFFORT_XHIGH: f64 = 0.90;
const EFFORT_HIGH: f64 = 0.75;
const EFFORT_LOW: f64 = 0.25;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NextAction {
    Inspect,
    Edit,
    Verify,
    Verdict,
}

impl NextAction {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Inspect => "inspect",
            Self::Edit => "edit",
            Self::Verify => "verify",
            Self::Verdict => "verdict",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct GateDecision {
    pub action: Option<NextAction>,
    pub needs_high_effort: Option<f64>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct NoulBundle {
    pub specific_edit_known: Option<f64>,
    pub tests_would_inform: Option<f64>,
    pub ready_to_stop: Option<f64>,
    #[allow(dead_code)]
    pub inspect_is_repeating: Option<f64>,
    pub needs_high_effort: Option<f64>,
}

impl NoulBundle {
    fn from_answers(answers: Option<&Value>) -> Self {
        let Some(answers) = answers else {
            return Self::default();
        };
        Self {
            specific_edit_known: noul_answer(answers, "specific_edit_known"),
            tests_would_inform: noul_answer(answers, "tests_would_inform"),
            ready_to_stop: noul_answer(answers, "ready_to_stop"),
            inspect_is_repeating: noul_answer(answers, "inspect_is_repeating"),
            needs_high_effort: noul_answer(answers, "needs_high_effort"),
        }
    }

    fn noul(value: Option<f64>) -> f64 {
        value.filter(|noul| noul.is_finite()).unwrap_or(0.0)
    }
}

#[derive(Clone, Debug)]
pub(crate) struct GateState {
    pub asked_to_fix: bool,
    pub mutated: bool,
    pub ran_verify: bool,
    pub inspect_only: bool,
    pub plan_open: bool,
    pub last_tools: Vec<String>,
    pub user_prompt: String,
}

impl GateState {
    fn encode(&self) -> String {
        let tools = if self.last_tools.is_empty() {
            "(none)".to_string()
        } else {
            self.last_tools.join(", ")
        };
        format!(
            "review-and-fix coding turn\nasked_to_fix={} mutated={} ran_verify={} inspect_only={} plan_open={}\nlast_tools: {tools}\nuser: {}",
            self.asked_to_fix,
            self.mutated,
            self.ran_verify,
            self.inspect_only,
            self.plan_open,
            truncate_prompt(&self.user_prompt),
        )
    }
}

#[derive(Clone)]
pub struct TypesafeSettings {
    pub api_key: Option<String>,
    pub base_url: String,
    pub model: String,
    pub min_confidence: f64,
    /// Jev `/permissions auto` hints. Default on when a key is present.
    pub auto: bool,
    /// Turn-scoped Jev reasoning-effort hints. Default on when a key is present.
    pub effort: bool,
}

impl Default for TypesafeSettings {
    fn default() -> Self {
        Self::disabled()
    }
}

impl std::fmt::Debug for TypesafeSettings {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TypesafeSettings")
            .field("enabled", &self.is_enabled())
            .field("base_url", &self.base_url)
            .field("model", &self.model)
            .field("min_confidence", &self.min_confidence)
            .field("auto", &self.auto)
            .field("effort", &self.effort)
            .finish()
    }
}

impl TypesafeSettings {
    pub const API_KEY_ENV: &'static str = "TYPESAFE_API_KEY";
    pub const BASE_URL_ENV: &'static str = "TYPESAFE_BASE_URL";
    pub const MODEL_ENV: &'static str = "TYPESAFE_DEFAULT_MODEL";
    /// `auth-store://` id for a machine-local TypeSafe key (`~/.config/hi/auth.json`).
    pub const AUTH_STORE_ID: &'static str = "typesafe";

    pub fn disabled() -> Self {
        Self {
            api_key: None,
            base_url: DEFAULT_BASE_URL.to_string(),
            model: DEFAULT_MODEL.to_string(),
            min_confidence: DEFAULT_MIN_CONFIDENCE,
            auto: true,
            effort: true,
        }
    }

    pub fn from_env() -> Self {
        Self::from_env_name(API_KEY_ENV)
    }

    pub fn from_env_name(api_key_env: &str) -> Self {
        let mut settings = Self::disabled();
        let key = std::env::var(api_key_env).ok().and_then(non_empty);
        settings.api_key = key;
        if let Some(url) = std::env::var(BASE_URL_ENV).ok().and_then(non_empty) {
            settings.base_url = url;
        }
        if let Some(model) = std::env::var(MODEL_ENV).ok().and_then(non_empty) {
            settings.model = model;
        }
        settings
    }

    pub fn apply_file(
        &mut self,
        base_url: Option<&str>,
        model: Option<&str>,
        min_confidence: Option<f64>,
        auto: Option<bool>,
        effort: Option<bool>,
    ) {
        if let Some(url) = base_url.map(str::trim).filter(|url| !url.is_empty()) {
            self.base_url = url.to_string();
        }
        if let Some(model) = model.map(str::trim).filter(|model| !model.is_empty()) {
            self.model = model.to_string();
        }
        if let Some(confidence) = min_confidence.filter(|value| value.is_finite()) {
            self.min_confidence = confidence.clamp(0.0, 1.0);
        }
        if let Some(auto) = auto {
            self.auto = auto;
        }
        if let Some(effort) = effort {
            self.effort = effort;
        }
    }

    pub fn is_enabled(&self) -> bool {
        self.api_key
            .as_deref()
            .is_some_and(|key| !key.trim().is_empty())
    }

    pub fn auto_enabled(&self) -> bool {
        self.is_enabled() && self.auto
    }

    pub fn effort_enabled(&self) -> bool {
        self.is_enabled() && self.effort
    }

    pub(crate) fn source(&self) -> NextActionSource {
        match self.gate() {
            Some(gate) => NextActionSource::Typesafe(gate),
            None => NextActionSource::Off,
        }
    }

    pub(crate) fn client(&self) -> Option<TypesafeClient> {
        let api_key = self.api_key.as_deref().and_then(|key| {
            let trimmed = key.trim();
            (!trimmed.is_empty()).then(|| trimmed.to_string())
        })?;
        Some(TypesafeClient::new(
            api_key,
            self.base_url.clone(),
            self.model.clone(),
        ))
    }

    fn gate(&self) -> Option<TypesafeGate> {
        Some(TypesafeGate {
            client: self.client()?,
            min_confidence: self.min_confidence,
        })
    }
}

#[derive(Clone)]
pub(crate) enum NextActionSource {
    Off,
    Typesafe(TypesafeGate),
    #[cfg(test)]
    Forced(NextAction),
}

impl NextActionSource {
    pub(crate) fn doctor_label(&self) -> Option<String> {
        match self {
            Self::Off => None,
            Self::Typesafe(gate) => Some(gate.client.model.clone()),
            #[cfg(test)]
            Self::Forced(action) => Some(format!("forced {}", action.as_str())),
        }
    }

    async fn decide(&self, state: &GateState) -> Option<GateDecision> {
        match self {
            Self::Off => None,
            Self::Typesafe(gate) => gate.decide(state).await,
            #[cfg(test)]
            Self::Forced(action) => Some(GateDecision {
                action: Some(*action),
                needs_high_effort: None,
            }),
        }
    }
}

/// Shared TypeSafe System One client for next-action, auto, effort, and compact.
#[derive(Clone)]
pub(crate) struct TypesafeClient {
    http: reqwest::Client,
    api_key: String,
    base_url: String,
    pub(crate) model: String,
}

impl TypesafeClient {
    pub(crate) fn new(
        api_key: impl Into<String>,
        base_url: impl Into<String>,
        model: impl Into<String>,
    ) -> Self {
        Self {
            http: hi_ai::agent_http_client_quick(),
            api_key: api_key.into(),
            base_url: base_url.into().trim_end_matches('/').to_string(),
            model: model.into(),
        }
    }

    pub(crate) async fn ask(&self, state: Value, questions: Value) -> Result<Value, String> {
        let url = format!("{}/v1/systemone", self.base_url);
        let body = json!({
            "model": self.model,
            "state": state,
            "questions": questions,
        });
        let response = self
            .http
            .post(&url)
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()
            .await
            .map_err(|err| err.to_string())?;
        let status = response.status();
        let value = response
            .json::<Value>()
            .await
            .map_err(|err| err.to_string())?;
        if !status.is_success() {
            return Err(format!("http {status}: {value}"));
        }
        if value.get("answers").is_none() {
            return Err("Jev response is missing answers".into());
        }
        Ok(value)
    }
}

#[derive(Clone)]
pub(crate) struct TypesafeGate {
    client: TypesafeClient,
    min_confidence: f64,
}

impl TypesafeGate {
    #[cfg(test)]
    fn new(
        api_key: impl Into<String>,
        base_url: impl Into<String>,
        model: impl Into<String>,
        min_confidence: f64,
    ) -> Self {
        Self {
            client: TypesafeClient::new(api_key, base_url, model),
            min_confidence,
        }
    }

    #[cfg(test)]
    fn from_env() -> Option<Self> {
        TypesafeSettings::from_env().gate()
    }

    async fn decide(&self, state: &GateState) -> Option<GateDecision> {
        match tokio::time::timeout(GATE_TIMEOUT, self.decide_inner(state)).await {
            Ok(decision) => decision,
            Err(_) => {
                tracing::debug!("typesafe next-action timed out; leaving harness hints in place");
                None
            }
        }
    }

    async fn decide_inner(&self, state: &GateState) -> Option<GateDecision> {
        let (state_value, questions) = request_payload(state);
        let value = self
            .client
            .ask(state_value, questions)
            .await
            .map_err(|err| {
                tracing::debug!("typesafe next-action request failed: {err}");
                err
            })
            .ok()?;
        Some(parse_gate_decision(&value, state, self.min_confidence))
    }
}

impl Harness {
    pub(crate) async fn next_action_decision(&mut self, state: &GateState) -> Option<NextAction> {
        let decision = self.next_action.decide(state).await?;
        self.apply_jev_effort(decision.needs_high_effort);
        decision.action
    }

    pub(crate) fn apply_jev_effort(&mut self, needs_high_effort: Option<f64>) {
        if self.live.effort_pinned() || !self.typesafe.effort_enabled() {
            return;
        }
        let Some(effort) = map_effort(needs_high_effort) else {
            return;
        };
        self.turn_effort = Some(effort);
    }

    pub(crate) fn effective_reasoning_effort(&self) -> Option<ReasoningEffort> {
        if self.live.effort_pinned() {
            return self.live.reasoning_effort();
        }
        self.turn_effort.or_else(|| self.live.reasoning_effort())
    }

    pub(crate) fn pin_user_effort(&mut self) {
        self.live.pin_effort();
        self.turn_effort = None;
    }

    pub(crate) fn spawn_turn_start_effort(
        &self,
        resume: bool,
        input: &str,
    ) -> Option<tokio::task::JoinHandle<Option<f64>>> {
        if resume || self.live.effort_pinned() || !self.typesafe.effort_enabled() {
            return None;
        }
        let client = self.typesafe.client()?;
        let prompt = input.to_string();
        Some(tokio::spawn(async move {
            score_needs_high_effort(&client, &prompt).await
        }))
    }

    #[cfg(test)]
    pub(crate) fn set_next_action_override(&mut self, action: NextAction) {
        self.next_action = NextActionSource::Forced(action);
    }
}

pub(crate) async fn score_needs_high_effort(client: &TypesafeClient, prompt: &str) -> Option<f64> {
    let state = json!(format!(
        "coding agent user request:\n{}",
        truncate_prompt(prompt)
    ));
    let questions = json!({
        "needs_high_effort": {
            "type": "noul",
            "instructions": "This task needs high reasoning effort: architecture, subtle bugs, or high-stakes changes rather than a local lookup, rename, or typo fix"
        }
    });
    match tokio::time::timeout(FAST_GATE_TIMEOUT, client.ask(state, questions)).await {
        Ok(Ok(value)) => noul_from_response(&value, "needs_high_effort"),
        Ok(Err(err)) => {
            tracing::debug!("typesafe effort request failed: {err}");
            None
        }
        Err(_) => {
            tracing::debug!("typesafe effort timed out; leaving live effort in place");
            None
        }
    }
}

pub(crate) fn map_effort(needs_high_effort: Option<f64>) -> Option<ReasoningEffort> {
    let noul = needs_high_effort.filter(|value| value.is_finite())?;
    if noul >= EFFORT_XHIGH {
        Some(ReasoningEffort::Xhigh)
    } else if noul >= EFFORT_HIGH {
        Some(ReasoningEffort::High)
    } else if noul <= EFFORT_LOW {
        Some(ReasoningEffort::Low)
    } else {
        None
    }
}

/// Compose a next-action from parallel nouls. Missing scores count as 0.
/// Inspect is never returned: the completion policy keeps its demand.
/// Unstarted open plans keep that demand too: tests of the current tree
/// and a verdict both skip the checklist (live ~/chat plan_stall).
pub(crate) fn compose_next_action(
    state: &GateState,
    scores: &NoulBundle,
    min_confidence: f64,
) -> Option<NextAction> {
    let tests = NoulBundle::noul(scores.tests_would_inform);
    let edit = NoulBundle::noul(scores.specific_edit_known);
    let stop = NoulBundle::noul(scores.ready_to_stop);
    if state.plan_open && !state.mutated {
        return (edit >= min_confidence).then_some(NextAction::Edit);
    }
    if tests >= min_confidence && state.asked_to_fix && !state.ran_verify {
        return Some(NextAction::Verify);
    }
    if edit >= min_confidence {
        return Some(NextAction::Edit);
    }
    if state.plan_open {
        return None;
    }
    if stop >= min_confidence && (state.ran_verify || !state.asked_to_fix) {
        return Some(NextAction::Verdict);
    }
    None
}

pub(crate) fn noul_answer(answers: &Value, name: &str) -> Option<f64> {
    let noul = answers.get(name)?.get("noul")?.as_f64()?;
    noul.is_finite().then_some(noul)
}

pub(crate) fn noul_from_response(value: &Value, name: &str) -> Option<f64> {
    noul_answer(value.get("answers")?, name)
}

pub(crate) fn truncate_prompt(text: &str) -> String {
    truncate_chars(text, MAX_PROMPT_CHARS)
}

pub(crate) fn truncate_chars(text: &str, max_chars: usize) -> String {
    let trimmed = text.trim();
    if trimmed.chars().count() <= max_chars {
        return trimmed.to_string();
    }
    trimmed.chars().take(max_chars).collect()
}

fn request_payload(state: &GateState) -> (Value, Value) {
    (
        json!(state.encode()),
        json!({
            "specific_edit_known": {
                "type": "noul",
                "instructions": "A specific code change is already known; the agent should call edit/write now rather than inspect more"
            },
            "tests_would_inform": {
                "type": "noul",
                "instructions": "Running the project tests now would inform the next edit more than more grep/read. Prefer this on review-and-fix when tests have not run this turn. Do not prefer this when plan_open=true and mutated=false: that is a feature checklist, not a failing suite"
            },
            "ready_to_stop": {
                "type": "noul",
                "instructions": "The agent can stop with a short user-visible verdict. Prefer this after tests have run, or on a review that needs no code change. Never prefer this when plan_open=true: unfinished plan steps are not a verdict"
            },
            "inspect_is_repeating": {
                "type": "noul",
                "instructions": "Further inspect tools would repeat work already done this turn"
            },
            "needs_high_effort": {
                "type": "noul",
                "instructions": "The next model call needs high reasoning effort rather than a cheap local lookup"
            }
        }),
    )
}

fn parse_gate_decision(value: &Value, state: &GateState, min_confidence: f64) -> GateDecision {
    let scores = NoulBundle::from_answers(value.get("answers"));
    GateDecision {
        action: compose_next_action(state, &scores, min_confidence),
        needs_high_effort: scores.needs_high_effort,
    }
}

fn non_empty(value: String) -> Option<String> {
    let trimmed = value.trim();
    (!trimmed.is_empty()).then(|| trimmed.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    #[test]
    fn compose_verify_wins_over_edit_when_tests_would_inform() {
        let action = compose_next_action(
            &inspect_loop_state(),
            &NoulBundle {
                specific_edit_known: Some(0.9),
                tests_would_inform: Some(0.8),
                ready_to_stop: Some(0.7),
                inspect_is_repeating: Some(0.9),
                needs_high_effort: Some(0.2),
            },
            0.55,
        );
        assert_eq!(action, Some(NextAction::Verify));
    }

    #[test]
    fn compose_edit_when_a_fix_is_known() {
        let action = compose_next_action(
            &inspect_loop_state(),
            &NoulBundle {
                specific_edit_known: Some(0.8),
                tests_would_inform: Some(0.2),
                ready_to_stop: Some(0.1),
                ..NoulBundle::default()
            },
            0.55,
        );
        assert_eq!(action, Some(NextAction::Edit));
    }

    #[test]
    fn compose_review_ready_to_stop_is_verdict() {
        let mut state = inspect_loop_state();
        state.asked_to_fix = false;
        let action = compose_next_action(
            &state,
            &NoulBundle {
                ready_to_stop: Some(0.9),
                tests_would_inform: Some(0.1),
                specific_edit_known: Some(0.1),
                ..NoulBundle::default()
            },
            0.55,
        );
        assert_eq!(action, Some(NextAction::Verdict));
    }

    #[test]
    fn compose_fix_after_verify_can_verdict() {
        let mut state = inspect_loop_state();
        state.ran_verify = true;
        let action = compose_next_action(
            &state,
            &NoulBundle {
                ready_to_stop: Some(0.9),
                tests_would_inform: Some(0.1),
                specific_edit_known: Some(0.1),
                ..NoulBundle::default()
            },
            0.55,
        );
        assert_eq!(action, Some(NextAction::Verdict));
    }

    #[test]
    fn compose_unstarted_plan_does_not_prefer_verify() {
        let action = compose_next_action(
            &unstarted_plan_state(),
            &NoulBundle {
                specific_edit_known: Some(0.9),
                tests_would_inform: Some(0.95),
                ready_to_stop: Some(0.9),
                inspect_is_repeating: Some(0.9),
                needs_high_effort: Some(0.2),
            },
            0.55,
        );
        assert_eq!(action, Some(NextAction::Edit));
    }

    #[test]
    fn compose_unstarted_plan_without_known_edit_keeps_demand() {
        let action = compose_next_action(
            &unstarted_plan_state(),
            &NoulBundle {
                specific_edit_known: Some(0.1),
                tests_would_inform: Some(0.99),
                ready_to_stop: Some(0.99),
                inspect_is_repeating: Some(0.9),
                needs_high_effort: Some(0.2),
            },
            0.55,
        );
        assert_eq!(action, None);
    }

    #[test]
    fn compose_open_plan_after_verify_does_not_verdict() {
        let mut state = unstarted_plan_state();
        state.mutated = true;
        state.ran_verify = true;
        let action = compose_next_action(
            &state,
            &NoulBundle {
                specific_edit_known: Some(0.1),
                tests_would_inform: Some(0.1),
                ready_to_stop: Some(0.99),
                inspect_is_repeating: Some(0.9),
                needs_high_effort: Some(0.2),
            },
            0.55,
        );
        assert_eq!(action, None);
    }

    #[test]
    fn compose_fix_without_verify_cannot_verdict() {
        let action = compose_next_action(
            &inspect_loop_state(),
            &NoulBundle {
                ready_to_stop: Some(0.99),
                tests_would_inform: Some(0.1),
                specific_edit_known: Some(0.1),
                ..NoulBundle::default()
            },
            0.55,
        );
        assert_eq!(action, None);
    }

    #[test]
    fn compose_below_floor_keeps_demand() {
        let action = compose_next_action(
            &inspect_loop_state(),
            &NoulBundle {
                specific_edit_known: Some(0.4),
                tests_would_inform: Some(0.4),
                ready_to_stop: Some(0.4),
                inspect_is_repeating: Some(0.9),
                needs_high_effort: Some(0.9),
            },
            0.55,
        );
        assert_eq!(action, None);
    }

    #[test]
    fn map_effort_thresholds() {
        assert_eq!(map_effort(Some(0.91)), Some(ReasoningEffort::Xhigh));
        assert_eq!(map_effort(Some(0.75)), Some(ReasoningEffort::High));
        assert_eq!(map_effort(Some(0.25)), Some(ReasoningEffort::Low));
        assert_eq!(map_effort(Some(0.5)), None);
        assert_eq!(map_effort(None), None);
    }

    #[test]
    fn user_effort_pin_wins_over_jev() {
        let dir = tempfile::tempdir().unwrap();
        let mut config = crate::HarnessConfig::pipe(dir.path().to_path_buf(), "pk_test");
        config.typesafe.api_key = Some("test-key".into());
        let mut harness = crate::Harness::new(config).unwrap();
        harness.apply_jev_effort(Some(0.95));
        assert_eq!(
            harness.effective_reasoning_effort(),
            Some(ReasoningEffort::Xhigh)
        );
        harness.apply_effort_arg(crate::EffortArg::Level(ReasoningEffort::Low));
        assert_eq!(
            harness.effective_reasoning_effort(),
            Some(ReasoningEffort::Low)
        );
        harness.apply_jev_effort(Some(0.99));
        assert_eq!(
            harness.effective_reasoning_effort(),
            Some(ReasoningEffort::Low)
        );
    }

    #[test]
    fn noul_parse_rejects_non_finite() {
        let answers = json!({"tests_would_inform": {"noul": "nope"}});
        assert!(noul_answer(&answers, "tests_would_inform").is_none());
        let answers = json!({"tests_would_inform": {"noul": 0.7}});
        assert!((noul_answer(&answers, "tests_would_inform").unwrap() - 0.7).abs() < f64::EPSILON);
    }

    #[tokio::test]
    async fn gate_parses_nouls_from_http() {
        let body = r#"{"model":"jev-1.13.0","answers":{"tests_would_inform":{"type":"noul","noul":0.81},"specific_edit_known":{"noul":0.2},"ready_to_stop":{"noul":0.1},"inspect_is_repeating":{"noul":0.9},"needs_high_effort":{"noul":0.4}}}"#;
        let url = serve_json(200, body).await;
        let gate = TypesafeGate::new("test-key", url, "jev-latest", 0.55);
        let decision = gate.decide(&inspect_loop_state()).await.unwrap();
        assert_eq!(decision.action, Some(NextAction::Verify));
        assert!((decision.needs_high_effort.unwrap() - 0.4).abs() < f64::EPSILON);
    }

    #[tokio::test]
    async fn gate_fails_open_on_http_error() {
        let url = serve_json(500, r#"{"error":"nope"}"#).await;
        let gate = TypesafeGate::new("test-key", url, "jev-latest", 0.55);
        assert!(gate.decide(&inspect_loop_state()).await.is_none());
    }

    #[tokio::test]
    #[ignore = "needs TYPESAFE_API_KEY and network"]
    async fn live_inspect_loop_prefers_verify() {
        let Some(gate) = TypesafeGate::from_env() else {
            eprintln!("skipping live TypeSafe test: {API_KEY_ENV} unset");
            return;
        };
        let decision = gate
            .decide(&inspect_loop_state())
            .await
            .expect("TypeSafe should return next-action nouls");
        assert!(
            matches!(decision.action, Some(NextAction::Verify | NextAction::Edit)),
            "inspect-loop review-and-fix should not prefer more inspect, got {:?}",
            decision.action
        );
    }

    fn inspect_loop_state() -> GateState {
        GateState {
            asked_to_fix: true,
            mutated: false,
            ran_verify: false,
            inspect_only: true,
            plan_open: false,
            last_tools: vec!["grep".into(), "grep".into(), "read".into()],
            user_prompt: "review for any major issues and fix".into(),
        }
    }

    fn unstarted_plan_state() -> GateState {
        GateState {
            asked_to_fix: true,
            mutated: false,
            ran_verify: false,
            inspect_only: true,
            plan_open: true,
            last_tools: vec!["update_plan".into(), "read".into()],
            user_prompt: "lets build all of that".into(),
        }
    }

    async fn serve_json(status: u16, body: &'static str) -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let Ok((mut stream, _)) = listener.accept().await else {
                return;
            };
            let mut buf = vec![0u8; 8192];
            let mut data = Vec::new();
            loop {
                let Ok(n) = stream.read(&mut buf).await else {
                    return;
                };
                if n == 0 {
                    break;
                }
                data.extend_from_slice(&buf[..n]);
                let headers_end = data.windows(4).position(|window| window == b"\r\n\r\n");
                let Some(headers_end) = headers_end else {
                    continue;
                };
                let headers = &data[..headers_end];
                let content_length = std::str::from_utf8(headers)
                    .ok()
                    .and_then(|headers| {
                        headers.lines().find_map(|line| {
                            let (name, value) = line.split_once(':')?;
                            name.eq_ignore_ascii_case("content-length")
                                .then(|| value.trim().parse::<usize>().ok())
                                .flatten()
                        })
                    })
                    .unwrap_or(0);
                if data.len() >= headers_end + 4 + content_length {
                    break;
                }
            }
            let reason = if status == 200 { "OK" } else { "Error" };
            let response = format!(
                "HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            );
            let _ = stream.write_all(response.as_bytes()).await;
        });
        format!("http://{addr}")
    }
}
