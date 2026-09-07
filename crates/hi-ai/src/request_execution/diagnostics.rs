//! Bounded request diagnostics containing no payload, URL, credential, or API text.

use super::*;

const RETAINED_ATTEMPTS: usize = 8;

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RequestFailureReason {
    AttemptsExhausted,
    BackoffExhausted,
    DeadlineExceeded,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct PhysicalAttemptEvidence {
    pub physical_attempt: u32,
    /// Digest of the concrete endpoint; never the endpoint itself.
    pub route: String,
    pub http_status: Option<u16>,
    /// A fixed local classification, never arbitrary provider error text.
    pub failure_kind: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct RequestFailureEvidence {
    pub reason: RequestFailureReason,
    pub operation_id: String,
    pub attempts: u32,
    pub attempt_limit: u32,
    pub backoff_ms: u64,
    pub recent_attempts: Vec<PhysicalAttemptEvidence>,
}

pub(super) fn record_dispatch(
    attempts: &mut VecDeque<PhysicalAttemptEvidence>,
    event: &ProviderAttemptEvent,
) {
    if attempts.len() == RETAINED_ATTEMPTS {
        attempts.pop_front();
    }
    attempts.push_back(PhysicalAttemptEvidence {
        physical_attempt: event.physical_attempt,
        route: event.route.clone(),
        http_status: None,
        failure_kind: None,
    });
}

impl RequestExecution {
    pub(super) fn failure(&self, reason: RequestFailureReason) -> ProviderError {
        self.local_failure(
            reason,
            &self.state.lock().unwrap_or_else(|e| e.into_inner()),
        )
    }

    pub(super) fn local_failure(
        &self,
        reason: RequestFailureReason,
        state: &ExecutionState,
    ) -> ProviderError {
        let evidence = RequestFailureEvidence {
            reason,
            operation_id: self.operation_id.clone(),
            attempts: state.attempts,
            attempt_limit: state.attempt_limit(self.policy),
            backoff_ms: state.backoff.as_millis().min(u128::from(u64::MAX)) as u64,
            recent_attempts: state.recent_attempts.iter().cloned().collect(),
        };
        let (code, message) = match reason {
            RequestFailureReason::AttemptsExhausted => (
                "request_attempts_exhausted",
                "model request exhausted its physical attempt allowance",
            ),
            RequestFailureReason::BackoffExhausted => (
                "request_backoff_exhausted",
                "model request exhausted its recovery wait allowance",
            ),
            RequestFailureReason::DeadlineExceeded => (
                "request_deadline",
                "model request exceeded its configured HTTP deadline",
            ),
        };
        // The short safe summary survives the ordinary persisted failure
        // message even when a frontend did not request detailed telemetry.
        let last = evidence
            .recent_attempts
            .last()
            .map(|attempt| {
                if let Some(kind) = &attempt.failure_kind {
                    format!("; last attempt: {kind}")
                } else if let Some(status) = attempt.http_status {
                    format!("; last HTTP response: {status}")
                } else {
                    "; last attempt had no HTTP response".into()
                }
            })
            .unwrap_or_default();
        let mut error = exhausted(
            code,
            &format!(
                "{message} ({}/{} sends{last})",
                evidence.attempts, evidence.attempt_limit
            ),
        );
        error.request_failure = Some(Box::new(evidence));
        error
    }

    pub(super) fn record_attempt_result(
        &self,
        physical_attempt: u32,
        http_status: Option<u16>,
        failure_kind: Option<&'static str>,
    ) {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(attempt) = state
            .recent_attempts
            .iter_mut()
            .find(|a| a.physical_attempt == physical_attempt)
        {
            attempt.http_status = http_status.or(attempt.http_status);
            if attempt.failure_kind.is_none() {
                attempt.failure_kind = failure_kind.map(str::to_owned);
            }
        }
    }

    /// Record a concrete adapter's decoded failure against its own dispatch.
    /// The explicit ordinal prevents parallel fanout from attributing a late
    /// sibling error to whichever request most recently started.
    pub(crate) fn record_provider_failure(&self, attempt: Option<u32>, error: &anyhow::Error) {
        let Some(attempt) = attempt else {
            return;
        };
        let Some(error) = crate::provider_error_details(error) else {
            return;
        };
        if error.request_failure.is_some() {
            return;
        }
        let kind = if error.code.as_deref() == Some("request_stalled") {
            "no_model_progress"
        } else {
            error.kind.as_str()
        };
        self.record_attempt_result(attempt, error.http_status, Some(kind));
    }
}
