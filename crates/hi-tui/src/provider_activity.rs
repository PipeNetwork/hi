//! Presentation of provider-owned liveness. This state never makes retry decisions.
use std::time::{Duration, Instant};

#[derive(Debug, Default)]
pub(crate) struct ProviderActivity {
    attempt: u32,
    phase: &'static str,
    started: Option<(String, u32)>,
    progress_at: Option<Instant>,
    paused_at: Option<Instant>,
    reason: Option<String>,
}

impl ProviderActivity {
    pub(crate) fn observe(&mut self, event: &hi_ai::ProviderAttemptEvent) {
        use hi_ai::ProviderAttemptState::*;
        self.attempt = event.physical_attempt;
        match &event.state {
            Started { .. } => {
                let identity = (event.operation_id.clone(), event.physical_attempt);
                if self.started.as_ref() == Some(&identity) {
                    return;
                }
                self.started = Some(identity);
                self.phase = "waiting for response";
                self.progress_at = Some(Instant::now());
                self.paused_at = None;
            }
            Response { .. } => self.phase = "streaming",
            Retrying { delay_ms } => {
                self.phase = "retrying";
                self.reason = Some(format!("backoff {}s", delay_ms / 1000));
                self.progress_at = None;
                self.paused_at = None;
            }
            Failed { code } => {
                self.phase = "recovering";
                self.reason = Some(code.clone());
            }
            WaitingForApproval => {
                self.phase = "waiting for approval";
                self.paused_at.get_or_insert_with(Instant::now);
            }
            ApprovalCompleted => {
                self.phase = "waiting for response";
                if let Some(paused) = self.paused_at.take()
                    && let Some(progress) = self.progress_at.as_mut()
                {
                    *progress += paused.elapsed();
                }
            }
        }
    }

    pub(crate) fn progress(&mut self) {
        if self.progress_at.is_some() {
            let now = Instant::now();
            self.progress_at = Some(now);
            if self.paused_at.is_some() {
                self.paused_at = Some(now);
            }
        }
    }

    pub(crate) fn label(&self) -> Option<String> {
        if self.attempt == 0 {
            return None;
        }
        let silence = self
            .progress_at
            .map(|at| {
                self.paused_at
                    .unwrap_or_else(Instant::now)
                    .saturating_duration_since(at)
            })
            .unwrap_or(Duration::ZERO)
            .as_secs();
        Some(format!(
            "{} · attempt {} · quiet {}s{}",
            self.phase,
            self.attempt,
            silence,
            self.reason
                .as_ref()
                .map(|reason| format!(" · {reason}"))
                .unwrap_or_default()
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn status_and_headers_do_not_replenish_silence() {
        let mut activity = ProviderActivity::default();
        let mut event = hi_ai::ProviderAttemptEvent {
            operation_id: "op".into(),
            request_id: "request".into(),
            physical_attempt: 1,
            provider: "fake".into(),
            route: "route".into(),
            model: "model".into(),
            state: hi_ai::ProviderAttemptState::Started { replay_attempt: 0 },
        };
        activity.observe(&event);
        let start = activity.progress_at;
        activity.observe(&event);
        assert_eq!(activity.progress_at, start);
        event.state = hi_ai::ProviderAttemptState::Response { http_status: 200 };
        activity.observe(&event);
        assert_eq!(activity.progress_at, start);
        assert!(activity.label().unwrap().contains("attempt 1"));
        event.state = hi_ai::ProviderAttemptState::WaitingForApproval;
        activity.observe(&event);
        assert_eq!(activity.progress_at, start);
        assert!(activity.paused_at.is_some());
    }

    #[test]
    fn approval_wait_preserves_prior_silence_and_duplicate_events_do_not_reset_it() {
        let now = Instant::now();
        let mut activity = ProviderActivity {
            attempt: 1,
            phase: "waiting for approval",
            started: Some(("op".into(), 1)),
            progress_at: Some(now - Duration::from_secs(850)),
            paused_at: Some(now - Duration::from_secs(600)),
            reason: None,
        };
        let mut event = hi_ai::ProviderAttemptEvent {
            operation_id: "op".into(),
            request_id: "request".into(),
            physical_attempt: 1,
            provider: "fake".into(),
            route: "route".into(),
            model: "model".into(),
            state: hi_ai::ProviderAttemptState::WaitingForApproval,
        };
        activity.observe(&event);
        assert!(activity.label().unwrap().contains("quiet 250s"));
        event.state = hi_ai::ProviderAttemptState::ApprovalCompleted;
        activity.observe(&event);
        assert!(activity.label().unwrap().contains("quiet 250s"));
        let resumed = activity.progress_at;
        activity.observe(&event);
        assert_eq!(activity.progress_at, resumed);
    }
}
