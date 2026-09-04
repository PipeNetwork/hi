use std::sync::{
    Arc, Mutex,
    atomic::{AtomicU32, AtomicU64, Ordering},
};

use anyhow::{Result, anyhow};
use async_trait::async_trait;
use hi_agent::{ConfirmationFuture, ConfirmationRequest, ConfirmationResult, Ui};
use hi_agent::{Observation, ObservationReceipt, ObservationSink};
use hi_ai::{ChatRequest, Completion, Provider, ServedModel, StreamEvent};
use hi_rsi_runtime::{BudgetKind, BudgetReservation, SharedBudgetLedger};
use hi_trace::{TraceSummary, TraceWriter};

struct State {
    writer: Option<TraceWriter>,
    last_hash: Option<String>,
    sequence: u64,
    failure: Option<String>,
    summary: Option<TraceSummary>,
}

pub(crate) struct TraceObservationSink {
    managed: bool,
    full_capture: bool,
    state: Mutex<State>,
}

impl TraceObservationSink {
    pub(crate) fn new(writer: TraceWriter, managed: bool, full_capture: bool) -> Arc<Self> {
        Arc::new(Self {
            managed,
            full_capture,
            state: Mutex::new(State {
                writer: Some(writer),
                last_hash: None,
                sequence: 0,
                failure: None,
                summary: None,
            }),
        })
    }

    pub(crate) fn full_capture(&self) -> bool {
        self.full_capture
    }

    pub(crate) fn finish(&self, terminal: Observation) -> Result<Option<TraceSummary>> {
        self.observe(terminal)?;
        let mut state = self
            .state
            .lock()
            .map_err(|_| anyhow!("RSI trace lock poisoned"))?;
        if state.failure.is_some() {
            return Ok(None);
        }
        let writer = state
            .writer
            .take()
            .ok_or_else(|| anyhow!("RSI trace is unavailable"))?;
        match writer.finalize() {
            Ok(summary) => {
                state.summary = Some(summary.clone());
                Ok(Some(summary))
            }
            Err(error) if self.managed => Err(error),
            Err(error) => {
                state.failure = Some(format!("{error:#}"));
                eprintln!(
                    "\x1b[33mRSI trace warning: {error:#}; this turn is not fully observed\x1b[0m"
                );
                Ok(None)
            }
        }
    }

    #[allow(dead_code)]
    pub(crate) fn failure(&self) -> Option<String> {
        self.state
            .lock()
            .ok()
            .and_then(|state| state.failure.clone())
    }
}

impl ObservationSink for TraceObservationSink {
    fn observe(&self, observation: Observation) -> Result<ObservationReceipt> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| anyhow!("RSI trace lock poisoned"))?;
        if let Some(error) = &state.failure {
            if self.managed {
                return Err(anyhow!(error.clone()));
            }
            return Ok(ObservationReceipt {
                event_hash: String::new(),
                sequence: state.sequence,
            });
        }
        let result = (|| {
            let causation = observation
                .causation_hash
                .or_else(|| state.last_hash.clone());
            let writer = state
                .writer
                .as_mut()
                .ok_or_else(|| anyhow!("RSI trace is unavailable"))?;
            let content = writer.put_blob(&observation.payload, observation.media_type)?;
            let mut data = observation.metadata;
            let object = data
                .as_object_mut()
                .ok_or_else(|| anyhow!("observation metadata must be an object"))?;
            object.insert("content".into(), serde_json::to_value(content)?);
            let hash = writer.record(
                observation.kind,
                observation.stage,
                observation.attempt,
                causation,
                Some(observation.correlation_id),
                data,
            )?;
            state.sequence += 1;
            state.last_hash = Some(hash.clone());
            Ok(ObservationReceipt {
                event_hash: hash,
                sequence: state.sequence,
            })
        })();
        match result {
            Ok(receipt) => Ok(receipt),
            Err(error) if self.managed => Err(error),
            Err(error) => {
                if let Some(writer) = state.writer.as_mut() {
                    let _ = writer.abandon();
                }
                state.failure = Some(format!("{error:#}"));
                eprintln!(
                    "\x1b[33mRSI trace warning: {error:#}; recording stopped for this turn\x1b[0m"
                );
                Ok(ObservationReceipt {
                    event_hash: String::new(),
                    sequence: state.sequence,
                })
            }
        }
    }
}

pub(crate) struct ObservedProvider {
    inner: Arc<dyn Provider>,
    sink: Arc<dyn ObservationSink>,
    attempts: AtomicU32,
    budget: Option<SharedBudgetLedger>,
    full_capture: bool,
}

impl ObservedProvider {
    pub(crate) fn new(
        inner: Arc<dyn Provider>,
        sink: Arc<dyn ObservationSink>,
        budget: Option<SharedBudgetLedger>,
        full_capture: bool,
    ) -> Self {
        Self {
            inner,
            sink,
            attempts: AtomicU32::new(0),
            budget,
            full_capture,
        }
    }

    fn reserve_model(&self, maximum_output: u32) -> Result<Option<ModelReservation>> {
        let Some(budget) = &self.budget else {
            return Ok(None);
        };
        let call = budget.reserve(BudgetKind::ModelCalls, 1)?;
        match budget.reserve(BudgetKind::OutputTokens, u64::from(maximum_output)) {
            Ok(output) => Ok(Some(ModelReservation {
                budget: budget.clone(),
                call: Some(call),
                output: Some(output),
            })),
            Err(error) => {
                budget.release(call)?;
                Err(error)
            }
        }
    }
}

struct ModelReservation {
    budget: SharedBudgetLedger,
    call: Option<BudgetReservation>,
    output: Option<BudgetReservation>,
}

impl ModelReservation {
    /// Commit the call immediately before entering the provider. If the
    /// provider future is later cancelled, that accepted attempt remains
    /// spent while the still-unknown output reservation is released by Drop.
    fn start(&mut self) -> Result<()> {
        if let Some(call) = self.call {
            self.budget.commit(call, 1)?;
            self.call = None;
        }
        Ok(())
    }

    fn complete(&mut self, input_tokens: u64, output_tokens: u64) -> Result<()> {
        if let Some(output) = self.output {
            self.budget.commit(output, output_tokens)?;
            self.output = None;
        }
        self.budget.consume(BudgetKind::InputTokens, input_tokens)
    }

    fn fail(&mut self) -> Result<()> {
        if let Some(output) = self.output {
            self.budget.release(output)?;
            self.output = None;
        }
        Ok(())
    }
}

impl Drop for ModelReservation {
    fn drop(&mut self) {
        if let Some(call) = self.call.take() {
            let _ = self.budget.release(call);
        }
        if let Some(output) = self.output.take() {
            let _ = self.budget.release(output);
        }
    }
}
#[async_trait]
impl Provider for ObservedProvider {
    crate::provider::forward_provider_capabilities!(self, inner);
    async fn stream(
        &self,
        request: ChatRequest,
        sink: &mut (dyn FnMut(StreamEvent) + Send),
    ) -> Result<Completion> {
        let mut reservation = self.reserve_model(request.max_tokens)?;
        let attempt = self.attempts.fetch_add(1, Ordering::Relaxed) + 1;
        let correlation = format!("model-{attempt}");
        let request_payload = if self.full_capture {
            serde_json::json!({
            "model": request.model,
            "messages": request.messages.as_ref(),
            "tools": request.tools.as_ref(),
            "max_tokens": request.max_tokens,
            "output_token_parameter": request.profile.output_token_parameter.label(),
            "native_tools": !request.tools.is_empty()
                && request.profile.tool_mode != hi_ai::ToolMode::ChatOnly,
            "tool_count": request.tools.len(),
            "temperature": request.temperature,
            "top_p": request.top_p,
            "frequency_penalty": request.frequency_penalty,
            "thinking_budget": request.thinking_budget,
            "reasoning_effort": request.reasoning_effort,
            "profile": request.profile,
            })
        } else {
            serde_json::json!({
                "model": request.model,
                "message_count": request.messages.len(),
                "tool_count": request.tools.len(),
                "max_tokens": request.max_tokens,
                "output_token_parameter": request.profile.output_token_parameter.label(),
                "native_tools": !request.tools.is_empty()
                    && request.profile.tool_mode != hi_ai::ToolMode::ChatOnly,
                "temperature": request.temperature,
                "top_p": request.top_p,
                "reasoning_requested": request.thinking_budget.is_some()
                    || request.reasoning_effort.is_some(),
                "reasoning_replayed": request.messages.iter().any(|message| {
                    message.content.iter().any(|content| matches!(
                        content,
                        hi_ai::Content::Thinking { .. }
                    ))
                }),
                "profile": request.profile,
            })
        };
        let request_receipt = self.sink.observe(Observation::json(
            "model_requested",
            "model",
            attempt,
            &correlation,
            &request_payload,
        )?)?;
        let mut observed_sink = |event: StreamEvent| {
            if let StreamEvent::WireAudit(audit) = &event {
                let payload = if self.full_capture {
                    serde_json::to_value(audit).unwrap_or_default()
                } else {
                    serde_json::json!({
                        "provider": audit.provider,
                        "route": audit.route,
                        "model": audit.model,
                        "output_token_parameter": audit.output_token_parameter,
                        "max_output_tokens": audit.max_output_tokens,
                        "temperature": audit.temperature,
                        "top_p": audit.top_p,
                        "reasoning_request": audit.reasoning_request,
                        "reasoning_replay": audit.reasoning_replay,
                        "native_tools_enabled": audit.native_tools_enabled,
                        "tool_count": audit.tool_count,
                        "strict_schema": audit.strict_schema,
                        "tool_choice": audit.tool_choice,
                        "request_attempt": audit.request_attempt,
                        "compatibility_fallback": audit.compatibility_fallback,
                        "accepted": audit.accepted,
                        "response_status": audit.response_status,
                    })
                };
                if let Ok(mut trace_event) = Observation::json(
                    "wire_audit",
                    "model",
                    audit.request_attempt,
                    &correlation,
                    &payload,
                ) {
                    trace_event.causation_hash = Some(request_receipt.event_hash.clone());
                    let _ = self.sink.observe(trace_event);
                }
                if audit.compatibility_fallback.is_some()
                    && let Ok(mut retry_event) = Observation::json(
                        "compatibility_retry",
                        "model",
                        audit.request_attempt,
                        &correlation,
                        &payload,
                    )
                {
                    retry_event.causation_hash = Some(request_receipt.event_hash.clone());
                    let _ = self.sink.observe(retry_event);
                }
            }
            sink(event);
        };
        if let Some(reservation) = reservation.as_mut() {
            reservation.start()?;
        }
        match self.inner.stream(request, &mut observed_sink).await {
            Ok(completion) => {
                if let Some(reservation) = reservation.as_mut() {
                    reservation.complete(
                        completion.usage.input_tokens,
                        completion.usage.output_tokens,
                    )?;
                }
                let completion_payload = if self.full_capture {
                    serde_json::to_value(&completion)?
                } else {
                    serde_json::json!({
                        "usage": completion.usage,
                        "stop_reason": completion.stop_reason,
                        "refusal": completion.refusal.is_some(),
                        "tool_call_count": completion.tool_calls().len(),
                        "tool_call_channel": completion.tool_call_channel,
                    })
                };
                let mut event = Observation::json(
                    "model_completed",
                    "model",
                    attempt,
                    correlation,
                    &completion_payload,
                )?;
                event.causation_hash = Some(request_receipt.event_hash);
                self.sink.observe(event)?;
                Ok(completion)
            }
            Err(error) => {
                if let Some(reservation) = reservation.as_mut() {
                    reservation.fail()?;
                }
                let error_payload = if self.full_capture {
                    serde_json::json!({"error": format!("{error:#}")})
                } else {
                    serde_json::json!({
                        "error_kind": hi_ai::provider_error_kind(&error)
                            .map(|kind| kind.as_str()),
                        "retryable": hi_ai::provider_error_retryable(&error),
                    })
                };
                let mut event = Observation::json(
                    "model_completed",
                    "model",
                    attempt,
                    correlation,
                    &error_payload,
                )?;
                event.causation_hash = Some(request_receipt.event_hash);
                self.sink.observe(event)?;
                Err(error)
            }
        }
    }

    async fn list_models(&self) -> Result<Vec<ServedModel>> {
        self.inner.list_models().await
    }
}

pub(crate) struct ToolObserver {
    sink: Arc<dyn ObservationSink>,
    dispatch: AtomicU64,
    full_capture: bool,
}

struct PendingTool {
    index: u64,
    correlation: String,
    name: String,
    arguments: String,
}

pub(crate) struct ObservedUi<'a> {
    inner: &'a mut dyn Ui,
    tools: Option<Arc<ToolObserver>>,
    approval_store: Option<Arc<dyn hi_policy::ApprovalStore>>,
    pending: Vec<PendingTool>,
    active: Option<PendingTool>,
}

impl<'a> ObservedUi<'a> {
    pub(crate) fn new(
        inner: &'a mut dyn Ui,
        tools: Option<Arc<ToolObserver>>,
        approval_store: Option<Arc<dyn hi_policy::ApprovalStore>>,
    ) -> Self {
        Self {
            inner,
            tools,
            approval_store,
            pending: Vec::new(),
            active: None,
        }
    }

    fn reserve(&self, name: &str, arguments: &str) -> Option<PendingTool> {
        let tools = self.tools.as_ref()?;
        let (index, correlation) = tools.dispatch(name, arguments);
        Some(PendingTool {
            index,
            correlation,
            name: name.to_owned(),
            arguments: arguments.to_owned(),
        })
    }
}

impl Ui for ObservedUi<'_> {
    fn assistant_text(&mut self, text: &str) {
        self.inner.assistant_text(text);
    }
    fn assistant_reasoning(&mut self, text: &str) {
        self.inner.assistant_reasoning(text);
    }
    fn assistant_end(&mut self) {
        self.inner.assistant_end();
    }
    fn tool_started(&mut self, name: &str, arguments: &str) {
        if let Some(call) = self.reserve(name, arguments) {
            self.pending.push(call);
        }
        self.inner.tool_started(name, arguments);
    }
    fn tool_stream(&mut self, name: &str, line: &str) {
        self.inner.tool_stream(name, line);
    }
    fn confirm(&mut self, request: ConfirmationRequest) -> ConfirmationFuture<'_> {
        let details = request.details();
        let tools = self.tools.clone();
        let store = self.approval_store.clone();
        let future = self.inner.confirm(request.clone());
        Box::pin(async move {
            let mut decision = future.await;
            if decision == ConfirmationResult::Unavailable
                && let Some(store) = store.as_ref()
            {
                if hi_agent::try_claim_approved_confirmation(store.as_ref(), &request) {
                    decision = ConfirmationResult::Approved;
                } else if hi_agent::park_confirmation(store.as_ref(), &request).is_ok() {
                    decision = ConfirmationResult::Parked;
                }
            }
            if let Some(tools) = tools {
                let (index, correlation) = tools.dispatch("policy_confirmation", &details);
                tools.result(
                    index,
                    correlation,
                    "policy_confirmation",
                    match decision {
                        ConfirmationResult::Approved => "approved",
                        ConfirmationResult::Rejected => "rejected",
                        ConfirmationResult::Cancelled => "cancelled",
                        ConfirmationResult::Unavailable => "unavailable",
                        ConfirmationResult::Parked => "parked",
                        ConfirmationResult::Answer(_) => "answered",
                    },
                );
            }
            decision
        })
    }
    fn ask_user(&mut self, question: &str, options: &[String]) -> hi_agent::AskUserFuture<'_> {
        self.inner.ask_user(question, options)
    }
    fn tool_call(&mut self, name: &str, arguments: &str) {
        let position = self
            .pending
            .iter()
            .position(|call| call.name == name && call.arguments == arguments);
        self.active = position
            .map(|index| self.pending.remove(index))
            .or_else(|| self.reserve(name, arguments));
        self.inner.tool_call(name, arguments);
    }
    fn tool_result(&mut self, name: &str, result: &str) {
        if let (Some(tools), Some(call)) = (&self.tools, self.active.take()) {
            tools.result(call.index, call.correlation, name, result);
        }
        self.inner.tool_result(name, result);
    }
    fn plan_result_id(
        &mut self,
        id: &str,
        name: &str,
        result: &str,
        status: hi_tools::ToolStatus,
        steps: &[hi_agent::PlanStep],
    ) {
        if let (Some(tools), Some(call)) = (&self.tools, self.active.take()) {
            tools.result(call.index, call.correlation, name, result);
        }
        self.inner.plan_result_id(id, name, result, status, steps);
    }
    fn status(&mut self, text: &str) {
        if let Some(tools) = &self.tools {
            tools.status(text);
        }
        self.inner.status(text);
    }
    fn checkpoint_warning(&mut self, text: &str) {
        self.inner.checkpoint_warning(text);
    }
    fn subagent_note(&mut self, text: &str) {
        self.inner.subagent_note(text);
    }
    fn subagent_sink(&self) -> Option<Arc<dyn hi_agent::SubagentSink>> {
        self.inner.subagent_sink()
    }
    fn plan(&mut self, steps: &[hi_agent::PlanStep]) {
        self.inner.plan(steps);
    }
    fn usage(&mut self, a: u64, b: u64, c: u64, d: Option<u32>, e: bool) {
        self.inner.usage(a, b, c, d, e);
    }
    fn session_usage(&mut self, usage: &hi_ai::Usage) {
        self.inner.session_usage(usage);
    }
    fn rate_limits(&mut self, limits: Option<hi_ai::RateLimitState>) {
        self.inner.rate_limits(limits);
    }
    fn turn_end(&mut self, summary: &str) {
        self.inner.turn_end(summary);
    }
    fn changed_files(&mut self, files: &[String]) {
        self.inner.changed_files(files);
    }
    fn suggested_prompt(&mut self, text: &str) {
        self.inner.suggested_prompt(text);
    }
    fn turn_error(&mut self, kind: &str, message: &str, guidance: &str) {
        self.inner.turn_error(kind, message, guidance);
    }
    fn nudge(&mut self, text: &str) {
        self.inner.nudge(text);
    }
}

impl ToolObserver {
    pub(crate) fn new(sink: Arc<dyn ObservationSink>, full_capture: bool) -> Arc<Self> {
        Arc::new(Self {
            sink,
            dispatch: AtomicU64::new(0),
            full_capture,
        })
    }

    pub(crate) fn dispatch(&self, name: &str, arguments: &str) -> (u64, String) {
        let index = self.dispatch.fetch_add(1, Ordering::Relaxed) + 1;
        let correlation = format!("tool-{index}");
        let payload = if self.full_capture {
            serde_json::json!({"name": name, "arguments": arguments})
        } else {
            serde_json::json!({
                "name": name,
                "argument_bytes": arguments.len(),
                "argument_hash": blake3::hash(arguments.as_bytes()).to_hex().to_string(),
            })
        };
        let mut event =
            match Observation::json("tool_requested", "tools", 1, &correlation, &payload) {
                Ok(event) => event,
                Err(_) => return (index, correlation),
            };
        event.metadata = serde_json::json!({"dispatch_index": index});
        let _ = self.sink.observe(event);
        (index, correlation)
    }

    pub(crate) fn result(&self, index: u64, correlation: String, name: &str, result: &str) {
        let payload = if self.full_capture {
            serde_json::json!({"name": name, "result": result})
        } else {
            serde_json::json!({
                "name": name,
                "result_bytes": result.len(),
                "result_hash": blake3::hash(result.as_bytes()).to_hex().to_string(),
            })
        };
        if let Ok(mut event) =
            Observation::json("tool_completed", "tools", 1, correlation, &payload)
        {
            event.metadata = serde_json::json!({"dispatch_index": index});
            let _ = self.sink.observe(event);
        }
    }

    pub(crate) fn status(&self, text: &str) {
        if !text.to_ascii_lowercase().contains("compact") {
            return;
        }
        if let Ok(event) = Observation::json(
            "context_compacted",
            "transcript",
            1,
            "context-compaction",
            &serde_json::json!({"status": text}),
        ) {
            let _ = self.sink.observe(event);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hi_ai::{Message, RequestProfile};
    use hi_rsi_runtime::RuntimeBudgets;

    struct AcceptingSink;

    impl ObservationSink for AcceptingSink {
        fn observe(&self, _: Observation) -> Result<ObservationReceipt> {
            Ok(ObservationReceipt {
                event_hash: "a".repeat(64),
                sequence: 1,
            })
        }
    }

    struct RejectingSink;

    impl ObservationSink for RejectingSink {
        fn observe(&self, _: Observation) -> Result<ObservationReceipt> {
            Err(anyhow!("trace rejected before provider admission"))
        }
    }

    struct HangingProvider {
        started: Arc<tokio::sync::Notify>,
    }

    #[async_trait]
    impl Provider for HangingProvider {
        async fn stream(
            &self,
            _: ChatRequest,
            _: &mut (dyn FnMut(StreamEvent) + Send),
        ) -> Result<Completion> {
            self.started.notify_one();
            std::future::pending().await
        }
    }

    struct TrackingProvider {
        entered: Arc<std::sync::atomic::AtomicBool>,
    }

    #[async_trait]
    impl Provider for TrackingProvider {
        async fn stream(
            &self,
            _: ChatRequest,
            _: &mut (dyn FnMut(StreamEvent) + Send),
        ) -> Result<Completion> {
            self.entered.store(true, Ordering::SeqCst);
            Ok(Completion::default())
        }
    }

    fn budgets() -> RuntimeBudgets {
        RuntimeBudgets {
            wall_time_seconds: 60,
            cpu_time_seconds: 60,
            memory_bytes: 1,
            disk_bytes: 1,
            input_tokens: 100,
            output_tokens: 100,
            tool_calls: 1,
            cost_microusd: 1,
            model_calls: 1,
            repair_iterations: 1,
            trace_bytes: 1,
        }
    }

    fn request() -> ChatRequest {
        ChatRequest {
            model: "test-model".into(),
            request_id: Some("request-1".into()),
            retry_attempt: 0,
            user_turn: true,
            canonical_objective: Some("test cancellation accounting".into()),
            messages: Arc::new(vec![Message::user("hello")]),
            tools: Vec::new().into(),
            tool_envelope: None,
            max_tokens: 40,
            temperature: None,
            top_p: None,
            frequency_penalty: None,
            thinking_budget: None,
            reasoning_effort: None,
            profile: RequestProfile::default(),
        }
    }

    #[tokio::test]
    async fn cancelled_provider_releases_output_reservation_but_keeps_started_call() {
        let ledger = SharedBudgetLedger::new(&budgets());
        let started = Arc::new(tokio::sync::Notify::new());
        let provider = ObservedProvider::new(
            Arc::new(HangingProvider {
                started: started.clone(),
            }),
            Arc::new(AcceptingSink),
            Some(ledger.clone()),
            false,
        );
        let mut event_sink = |_: StreamEvent| {};
        {
            let stream = provider.stream(request(), &mut event_sink);
            tokio::pin!(stream);
            tokio::select! {
                _ = started.notified() => {}
                result = &mut stream => panic!("provider unexpectedly settled: {result:?}"),
                _ = tokio::time::sleep(std::time::Duration::from_secs(2)) => {
                    panic!("inner provider never started")
                }
            }
        }

        let usage = ledger.usage().unwrap();
        assert_eq!(usage.consumed.get(&BudgetKind::ModelCalls), Some(&1));
        assert_eq!(
            usage
                .consumed
                .get(&BudgetKind::OutputTokens)
                .copied()
                .unwrap_or(0),
            0
        );
        assert!(usage.reserved.values().all(|amount| *amount == 0));
        assert_eq!(ledger.remaining(BudgetKind::ModelCalls).unwrap(), 0);
        assert_eq!(ledger.remaining(BudgetKind::OutputTokens).unwrap(), 100);
    }

    #[tokio::test]
    async fn rejected_request_observation_releases_all_model_reservations() {
        let ledger = SharedBudgetLedger::new(&budgets());
        let entered = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let provider = ObservedProvider::new(
            Arc::new(TrackingProvider {
                entered: entered.clone(),
            }),
            Arc::new(RejectingSink),
            Some(ledger.clone()),
            false,
        );
        let mut event_sink = |_: StreamEvent| {};

        let error = provider
            .stream(request(), &mut event_sink)
            .await
            .expect_err("mandatory observation is scripted to fail");

        assert!(error.to_string().contains("trace rejected"));
        assert!(!entered.load(Ordering::SeqCst));
        let usage = ledger.usage().unwrap();
        assert!(usage.consumed.values().all(|amount| *amount == 0));
        assert!(usage.reserved.values().all(|amount| *amount == 0));
        assert_eq!(ledger.remaining(BudgetKind::ModelCalls).unwrap(), 1);
        assert_eq!(ledger.remaining(BudgetKind::OutputTokens).unwrap(), 100);
    }
}
