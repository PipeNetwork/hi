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
    state: Mutex<State>,
}

impl TraceObservationSink {
    pub(crate) fn new(writer: TraceWriter, managed: bool) -> Arc<Self> {
        Arc::new(Self {
            managed,
            state: Mutex::new(State {
                writer: Some(writer),
                last_hash: None,
                sequence: 0,
                failure: None,
                summary: None,
            }),
        })
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
}

impl ObservedProvider {
    pub(crate) fn new(
        inner: Arc<dyn Provider>,
        sink: Arc<dyn ObservationSink>,
        budget: Option<SharedBudgetLedger>,
    ) -> Self {
        Self {
            inner,
            sink,
            attempts: AtomicU32::new(0),
            budget,
        }
    }

    fn reserve_model(&self, maximum_output: u32) -> Result<Option<ModelReservation>> {
        let Some(budget) = &self.budget else {
            return Ok(None);
        };
        let call = budget.reserve(BudgetKind::ModelCalls, 1)?;
        match budget.reserve(BudgetKind::OutputTokens, u64::from(maximum_output)) {
            Ok(output) => Ok(Some(ModelReservation { call, output })),
            Err(error) => {
                budget.release(call)?;
                Err(error)
            }
        }
    }
}

struct ModelReservation {
    call: BudgetReservation,
    output: BudgetReservation,
}

#[async_trait]
impl Provider for ObservedProvider {
    async fn stream(
        &self,
        request: ChatRequest,
        sink: &mut (dyn FnMut(StreamEvent) + Send),
    ) -> Result<Completion> {
        let reservation = self.reserve_model(request.max_tokens)?;
        let attempt = self.attempts.fetch_add(1, Ordering::Relaxed) + 1;
        let correlation = format!("model-{attempt}");
        let request_payload = serde_json::json!({
            "model": request.model,
            "messages": request.messages.as_ref(),
            "tools": request.tools.as_ref(),
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "frequency_penalty": request.frequency_penalty,
            "thinking_budget": request.thinking_budget,
            "reasoning_effort": request.reasoning_effort,
            "profile": request.profile,
        });
        let request_receipt = self.sink.observe(Observation::json(
            "model_requested",
            "model",
            attempt,
            &correlation,
            &request_payload,
        )?)?;
        match self.inner.stream(request, sink).await {
            Ok(completion) => {
                if let (Some(budget), Some(reservation)) = (&self.budget, reservation) {
                    budget.commit(reservation.call, 1)?;
                    budget.commit(reservation.output, completion.usage.output_tokens)?;
                    budget.consume(BudgetKind::InputTokens, completion.usage.input_tokens)?;
                }
                let mut event = Observation::json(
                    "model_completed",
                    "model",
                    attempt,
                    correlation,
                    &completion,
                )?;
                event.causation_hash = Some(request_receipt.event_hash);
                self.sink.observe(event)?;
                Ok(completion)
            }
            Err(error) => {
                if let (Some(budget), Some(reservation)) = (&self.budget, reservation) {
                    budget.commit(reservation.call, 1)?;
                    budget.release(reservation.output)?;
                }
                let mut event = Observation::json(
                    "model_completed",
                    "model",
                    attempt,
                    correlation,
                    &format!("{error:#}"),
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
    pending: Vec<PendingTool>,
    active: Option<PendingTool>,
}

impl<'a> ObservedUi<'a> {
    pub(crate) fn new(inner: &'a mut dyn Ui, tools: Option<Arc<ToolObserver>>) -> Self {
        Self {
            inner,
            tools,
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
        let future = self.inner.confirm(request);
        Box::pin(async move {
            let decision = future.await;
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
                    },
                );
            }
            decision
        })
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
    fn status(&mut self, text: &str) {
        self.inner.status(text);
    }
    fn checkpoint_warning(&mut self, text: &str) {
        self.inner.checkpoint_warning(text);
    }
    fn subagent_note(&mut self, text: &str) {
        self.inner.subagent_note(text);
    }
    fn plan(&mut self, steps: &[hi_agent::PlanStep]) {
        self.inner.plan(steps);
    }
    fn usage(&mut self, a: u64, b: u64, c: u64, d: Option<u32>, e: bool) {
        self.inner.usage(a, b, c, d, e);
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
    fn turn_error(&mut self, kind: &str, message: &str, guidance: &str) {
        self.inner.turn_error(kind, message, guidance);
    }
    fn nudge(&mut self, text: &str) {
        self.inner.nudge(text);
    }
}

impl ToolObserver {
    pub(crate) fn new(sink: Arc<dyn ObservationSink>) -> Arc<Self> {
        Arc::new(Self {
            sink,
            dispatch: AtomicU64::new(0),
        })
    }

    pub(crate) fn dispatch(&self, name: &str, arguments: &str) -> (u64, String) {
        let index = self.dispatch.fetch_add(1, Ordering::Relaxed) + 1;
        let correlation = format!("tool-{index}");
        let mut event = match Observation::json(
            "tool_requested",
            "tools",
            1,
            &correlation,
            &serde_json::json!({"name": name, "arguments": arguments}),
        ) {
            Ok(event) => event,
            Err(_) => return (index, correlation),
        };
        event.metadata = serde_json::json!({"dispatch_index": index});
        let _ = self.sink.observe(event);
        (index, correlation)
    }

    pub(crate) fn result(&self, index: u64, correlation: String, name: &str, result: &str) {
        if let Ok(mut event) = Observation::json(
            "tool_completed",
            "tools",
            1,
            correlation,
            &serde_json::json!({"name": name, "result": result}),
        ) {
            event.metadata = serde_json::json!({"dispatch_index": index});
            let _ = self.sink.observe(event);
        }
    }
}
