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
        execution: Default::default(),
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
