//! One inference operation's physical sends, recovery allowance and progress clock.
//!
//! Cloning a request shares this ledger. Changing a payload, route, or recovery
//! strategy never creates fresh allowance. Only a newly accepted model round
//! should create a new execution.

use std::collections::{HashMap, VecDeque};
use std::future::Future;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::{ProviderError, ProviderErrorKind, StreamEvent};

mod diagnostics;
pub use diagnostics::{PhysicalAttemptEvidence, RequestFailureEvidence, RequestFailureReason};

#[derive(Clone, Copy, Debug)]
pub struct RequestExecutionPolicy {
    pub max_attempts: u32,
    pub max_backoff: Duration,
    pub no_progress_timeout: Duration,
}

impl Default for RequestExecutionPolicy {
    fn default() -> Self {
        Self {
            max_attempts: 4,
            max_backoff: Duration::from_secs(60),
            no_progress_timeout: Duration::from_secs(300),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ProviderAttemptState {
    Started { replay_attempt: u32 },
    Response { http_status: u16 },
    Retrying { delay_ms: u64 },
    Failed { code: String },
    WaitingForApproval,
    ApprovalCompleted,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ProviderAttemptEvent {
    pub operation_id: String,
    pub request_id: String,
    pub physical_attempt: u32,
    pub provider: String,
    /// A digest of the concrete endpoint, never an endpoint carrying credentials.
    pub route: String,
    pub model: String,
    pub state: ProviderAttemptState,
}

#[derive(Debug, Default)]
struct ExecutionState {
    attempts: u32,
    reserved: u32,
    fanout: Option<FanoutPlan>,
    backoff: Duration,
    identities: HashMap<String, (String, u32)>,
    last: Option<ProviderAttemptEvent>,
    recent_attempts: VecDeque<diagnostics::PhysicalAttemptEvidence>,
}

#[derive(Debug)]
struct FanoutPlan {
    initial_dispatches: u32,
}

impl ExecutionState {
    fn attempt_limit(&self, policy: RequestExecutionPolicy) -> u32 {
        match &self.fanout {
            Some(plan) if policy.max_attempts > 0 => plan
                .initial_dispatches
                .saturating_add(policy.max_attempts.saturating_sub(1)),
            _ => policy.max_attempts,
        }
    }
}

#[derive(Debug)]
pub struct RequestExecution {
    operation_id: String,
    policy: RequestExecutionPolicy,
    state: Mutex<ExecutionState>,
}

impl Default for RequestExecution {
    fn default() -> Self {
        Self::new(RequestExecutionPolicy::default())
    }
}

impl RequestExecution {
    pub fn new(policy: RequestExecutionPolicy) -> Self {
        Self {
            operation_id: format!("hi_{}", uuid::Uuid::new_v4().simple()),
            policy,
            state: Mutex::new(ExecutionState::default()),
        }
    }

    /// Start an accepted next operation with the same resolved policy.
    pub fn fresh_operation(&self) -> Arc<Self> {
        Arc::new(Self::new(self.policy))
    }

    pub fn attempts(&self) -> u32 {
        self.state
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .attempts
    }

    pub fn attempt_limit(&self) -> u32 {
        self.state
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .attempt_limit(self.policy)
    }

    pub fn remaining_attempts(&self) -> u32 {
        let state = self.state.lock().unwrap_or_else(|error| error.into_inner());
        state
            .attempt_limit(self.policy)
            .saturating_sub(state.attempts.saturating_add(state.reserved))
    }

    /// Declare the composite route's initial children plus essential aggregation
    /// once. Their recoveries share the ordinary policy's retry allowance. A
    /// repeated route entry retains all sends and backoff already spent.
    pub(crate) fn plan_fanout(&self, initial_dispatches: u32) -> anyhow::Result<()> {
        let mut state = self.state.lock().unwrap_or_else(|error| error.into_inner());
        anyhow::ensure!(
            initial_dispatches > 0,
            "fanout requires an initial dispatch"
        );
        if let Some(plan) = &state.fanout {
            anyhow::ensure!(
                plan.initial_dispatches == initial_dispatches,
                "cannot change an inference operation's fanout plan"
            );
        } else {
            state.fanout = Some(FanoutPlan { initial_dispatches });
        }
        Ok(())
    }

    pub fn backoff_spent(&self) -> Duration {
        self.state.lock().unwrap_or_else(|e| e.into_inner()).backoff
    }

    // Keep budget failures typed without allocating a second error wrapper.
    #[allow(clippy::result_large_err)]
    pub fn ensure_available(&self) -> Result<(), ProviderError> {
        let state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        if state.attempts
            >= state
                .attempt_limit(self.policy)
                .saturating_sub(state.reserved)
        {
            return Err(self.local_failure(RequestFailureReason::AttemptsExhausted, &state));
        }
        Ok(())
    }

    /// Reserve an essential planned inference (the MoA aggregator) while an
    /// optional child uses the same total allowance. Dropping restores access;
    /// it does not restore spent attempts.
    pub fn reserve(self: &Arc<Self>, count: u32) -> AttemptReservation {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        let count = count.min(
            state
                .attempt_limit(self.policy)
                .saturating_sub(state.attempts.saturating_add(state.reserved)),
        );
        state.reserved += count;
        AttemptReservation {
            execution: self.clone(),
            count,
        }
    }

    /// Backoff is charged before sleeping, so cancellation cannot replenish it.
    pub async fn backoff(&self, delay: Duration) -> anyhow::Result<()> {
        {
            let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
            if delay > self.policy.max_backoff.saturating_sub(state.backoff) {
                return Err(self
                    .local_failure(RequestFailureReason::BackoffExhausted, &state)
                    .into());
            }
            state.backoff += delay;
        }
        tokio::time::sleep(delay).await;
        Ok(())
    }

    pub fn progress(&self) -> RequestProgress {
        RequestProgress {
            timeout: self.policy.no_progress_timeout,
            state: Mutex::new(ProgressState::default()),
            changed: tokio::sync::Notify::new(),
        }
    }

    pub fn note_state(
        &self,
        state: ProviderAttemptState,
        sink: &mut (dyn FnMut(StreamEvent) + Send),
    ) {
        let event = self
            .state
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .last
            .clone();
        if let Some(mut event) = event {
            event.state = state;
            sink(StreamEvent::ProviderAttempt(Box::new(event)));
        }
    }

    /// Dispatch inference HTTP on its original client. Every actual send,
    /// including exact transport replays, acquires allowance here. Build the
    /// request with [`crate::inference_http_client_for_socket`] so reqwest cannot
    /// retry or follow redirects outside this ledger.
    pub async fn dispatch(
        &self,
        builder: reqwest::RequestBuilder,
        provider: &str,
        model: &str,
        sink: &mut (dyn FnMut(StreamEvent) + Send),
    ) -> Result<reqwest::Response, ProviderError> {
        self.dispatch_controlled(builder, provider, model, sink, true)
            .await
    }

    /// Remote harnesses can perform effects before a response is lost. They
    /// must retain accounting without inheriting inference transport replay.
    pub async fn dispatch_once(
        &self,
        builder: reqwest::RequestBuilder,
        provider: &str,
        model: &str,
        sink: &mut (dyn FnMut(StreamEvent) + Send),
    ) -> Result<reqwest::Response, ProviderError> {
        self.dispatch_controlled(builder, provider, model, sink, false)
            .await
    }

    async fn dispatch_controlled(
        &self,
        builder: reqwest::RequestBuilder,
        provider: &str,
        model: &str,
        sink: &mut (dyn FnMut(StreamEvent) + Send),
        allow_transport_replay: bool,
    ) -> Result<reqwest::Response, ProviderError> {
        let budget = crate::http::model_request_timeout()
            .map(crate::http::OperationBudget::new)
            .unwrap_or_else(crate::http::OperationBudget::unlimited);
        let (client, built) = builder.build_split();
        let mut request = built
            .map_err(|_| exhausted("invalid_inference_request", "could not build model request"))?;
        let body = request.body().and_then(|b| b.as_bytes()).ok_or_else(|| {
            exhausted(
                "non_replayable_request",
                "model request body must be buffered before dispatch",
            )
        })?;
        let mut identity = payload_identity(&request, body);
        let initial_id = request
            .headers()
            .get("x-request-id")
            .and_then(|v| v.to_str().ok())
            .filter(|v| {
                !v.is_empty()
                    && v.len() <= 96
                    && v.bytes()
                        .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_'))
            })
            .map(str::to_owned);
        loop {
            let mut attempt_request = request.try_clone().ok_or_else(|| {
                exhausted("non_replayable_request", "model request cannot be replayed")
            })?;
            let event = {
                let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
                if state.attempts
                    >= state
                        .attempt_limit(self.policy)
                        .saturating_sub(state.reserved)
                {
                    return Err(self.local_failure(RequestFailureReason::AttemptsExhausted, &state));
                }
                let first_shape = state.identities.is_empty();
                let entry = state.identities.entry(identity.clone()).or_insert_with(|| {
                    (
                        if first_shape {
                            initial_id
                                .clone()
                                .unwrap_or_else(|| self.operation_id.clone())
                        } else {
                            format!("{}-{}", self.operation_id, &identity[..12])
                        },
                        0,
                    )
                });
                let request_id = entry.0.clone();
                let replay_attempt = entry.1;
                entry.1 += 1;
                state.attempts += 1;
                let event = ProviderAttemptEvent {
                    operation_id: self.operation_id.clone(),
                    request_id,
                    physical_attempt: state.attempts,
                    provider: provider.into(),
                    route: crate::endpoint_capability_route(provider, request.url().as_str()),
                    model: model.into(),
                    state: ProviderAttemptState::Started { replay_attempt },
                };
                state.last = Some(event.clone());
                diagnostics::record_dispatch(&mut state.recent_attempts, &event);
                event
            };
            let replay_attempt = match event.state {
                ProviderAttemptState::Started { replay_attempt } => replay_attempt,
                _ => unreachable!(),
            };
            let headers = attempt_request.headers_mut();
            headers.insert(
                "x-request-id",
                event
                    .request_id
                    .parse()
                    .expect("generated request id is a header"),
            );
            headers.insert(
                "x-request-attempt",
                replay_attempt.to_string().parse().unwrap(),
            );
            headers.insert(
                "idempotency-key",
                format!("{}:{}", event.request_id, &identity[..24])
                    .parse()
                    .unwrap(),
            );
            sink(StreamEvent::ProviderAttempt(Box::new(event.clone())));
            let result = budget
                .run(
                    || "model dispatch exceeded its configured HTTP deadline".into(),
                    client.execute(attempt_request),
                )
                .await
                .map_err(|_| {
                    self.record_attempt_result(event.physical_attempt, None, Some("deadline"));
                    self.failure(RequestFailureReason::DeadlineExceeded)
                })?;
            match result {
                Ok(response) => {
                    self.record_attempt_result(
                        event.physical_attempt,
                        Some(response.status().as_u16()),
                        None,
                    );
                    emit_attempt_state(
                        &event,
                        ProviderAttemptState::Response {
                            http_status: response.status().as_u16(),
                        },
                        sink,
                    );
                    if allow_transport_replay && matches!(response.status().as_u16(), 307 | 308) {
                        let target = response
                            .headers()
                            .get(reqwest::header::LOCATION)
                            .and_then(|location| location.to_str().ok())
                            .and_then(|location| request.url().join(location).ok())
                            .ok_or_else(|| {
                                exhausted(
                                    "invalid_inference_redirect",
                                    "model endpoint returned an invalid redirect",
                                )
                            })?;
                        if crate::http::redirect_leaves_origin(request.url(), &target) {
                            return Err(exhausted(
                                "unsafe_inference_redirect",
                                "refusing model redirect across origins",
                            ));
                        }
                        *request.url_mut() = target;
                        identity =
                            payload_identity(&request, request.body().unwrap().as_bytes().unwrap());
                        continue;
                    }
                    return Ok(response);
                }
                Err(error) => {
                    self.record_attempt_result(
                        event.physical_attempt,
                        None,
                        Some("transport_failure"),
                    );
                    emit_attempt_state(
                        &event,
                        ProviderAttemptState::Failed {
                            code: "transport_failure".into(),
                        },
                        sink,
                    );
                    if !allow_transport_replay
                        || !(error.is_timeout() || error.is_connect() || error.is_request())
                    {
                        return Err(exhausted(
                            "transport_failure",
                            "request to model endpoint failed",
                        ));
                    }
                    self.ensure_available()?;
                    let delay = Duration::from_millis(500 * (1u64 << self.attempts().min(3)));
                    emit_attempt_state(
                        &event,
                        ProviderAttemptState::Retrying {
                            delay_ms: delay.as_millis() as u64,
                        },
                        sink,
                    );
                    budget
                        .run(
                            || "model recovery exceeded its configured HTTP deadline".into(),
                            self.backoff(delay),
                        )
                        .await
                        .map_err(|_| self.failure(RequestFailureReason::DeadlineExceeded))?
                        .map_err(|error| {
                            error.downcast::<ProviderError>().unwrap_or_else(|_| {
                                exhausted(
                                    "request_backoff_exhausted",
                                    "model request recovery wait failed",
                                )
                            })
                        })?;
                }
            }
        }
    }
}

pub struct AttemptReservation {
    execution: Arc<RequestExecution>,
    count: u32,
}

impl Drop for AttemptReservation {
    fn drop(&mut self) {
        self.execution
            .state
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .reserved -= self.count;
    }
}

/// A progress clock belongs to a concrete stream, so progress on a parallel
/// child never keeps a silent sibling alive. Raw bytes/statuses are not progress.
pub struct RequestProgress {
    timeout: Duration,
    state: Mutex<ProgressState>,
    changed: tokio::sync::Notify,
}

#[derive(Default)]
struct ProgressState {
    last_progress: Option<tokio::time::Instant>,
    paused_at: Option<tokio::time::Instant>,
    physical_attempt: Option<(String, u32)>,
}

impl RequestProgress {
    pub fn observe(&self, event: &StreamEvent) {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        let now = tokio::time::Instant::now();
        match event {
            StreamEvent::Text(text) | StreamEvent::Reasoning(text) if !text.is_empty() => {
                state.last_progress = state.last_progress.map(|_| now);
                state.paused_at = state.paused_at.map(|_| now);
            }
            StreamEvent::ToolCallDelta {
                arguments_delta, ..
            } if !arguments_delta.is_empty() => {
                state.last_progress = state.last_progress.map(|_| now);
                state.paused_at = state.paused_at.map(|_| now);
            }
            StreamEvent::ProviderAttempt(event) => match event.state {
                ProviderAttemptState::Started { .. } => {
                    let attempt = (event.operation_id.clone(), event.physical_attempt);
                    if state.physical_attempt.as_ref() == Some(&attempt) {
                        return;
                    }
                    state.physical_attempt = Some(attempt);
                    state.last_progress = Some(now);
                    state.paused_at = None;
                }
                ProviderAttemptState::ApprovalCompleted => {
                    if let Some(paused_at) = state.paused_at.take() {
                        state.last_progress = state
                            .last_progress
                            .map(|last| last + now.saturating_duration_since(paused_at));
                    }
                }
                ProviderAttemptState::WaitingForApproval => {
                    state.paused_at.get_or_insert(now);
                }
                _ => return,
            },
            _ => return,
        }
        self.changed.notify_one();
    }

    pub async fn watch<T>(
        &self,
        future: impl Future<Output = anyhow::Result<T>>,
    ) -> anyhow::Result<T> {
        if self.timeout.is_zero()
            || tokio::time::Instant::now()
                .checked_add(self.timeout)
                .is_none()
        {
            return future.await;
        }
        tokio::pin!(future);
        loop {
            let deadline = {
                let state = self.state.lock().unwrap_or_else(|e| e.into_inner());
                state
                    .last_progress
                    .filter(|_| state.paused_at.is_none())
                    .map(|last| last + self.timeout)
            };
            tokio::select! {
                biased;
                result = &mut future => return result,
                _ = self.changed.notified() => {},
                _ = async {
                    match deadline {
                        Some(deadline) => tokio::time::sleep_until(deadline).await,
                        None => std::future::pending::<()>().await,
                    }
                } => {
                    let state = self.state.lock().unwrap_or_else(|e| e.into_inner());
                    if state.paused_at.is_none() && state.last_progress.is_some_and(|last| last.elapsed() >= self.timeout) {
                        return Err(ProviderError::new(ProviderErrorKind::Outage, "model request stopped making progress")
                            .with_api_contract(Some("request_stalled".into()), Some(true), None).into());
                    }
                }
            }
        }
    }
}

fn payload_identity(request: &reqwest::Request, body: &[u8]) -> String {
    let mut hash = blake3::Hasher::new();
    hash.update(request.url().as_str().as_bytes());
    hash.update(&[0]);
    hash.update(body);
    hash.finalize().to_hex().to_string()
}

fn emit_attempt_state(
    event: &ProviderAttemptEvent,
    state: ProviderAttemptState,
    sink: &mut (dyn FnMut(StreamEvent) + Send),
) {
    let mut event = event.clone();
    event.state = state;
    sink(StreamEvent::ProviderAttempt(Box::new(event)));
}

fn exhausted(code: &str, message: &str) -> ProviderError {
    ProviderError::new(ProviderErrorKind::Outage, message).with_api_contract(
        Some(code.into()),
        Some(false),
        None,
    )
}

pub(crate) fn include_stalled_usage(
    error: anyhow::Error,
    estimated_input: u64,
    previous: crate::Usage,
) -> anyhow::Error {
    let Some(provider) = error
        .downcast_ref::<ProviderError>()
        .filter(|error| error.code.as_deref() == Some("request_stalled"))
    else {
        return error;
    };
    let mut usage = crate::Usage {
        input_tokens: estimated_input,
        context_occupancy: estimated_input,
        input_includes_cache: true,
        estimated: true,
        ..Default::default()
    };
    usage.add(previous);
    provider.clone().with_usage(usage).into()
}

#[cfg(test)]
mod tests;
