use crate::{
    ApiCase, ApiExecutor, ApiOutcome, CaseVerdict, CheckpointSink, EquivalenceContract, LocalCase,
    LocalImplementation, ProbeLevel, compare_local, compare_response,
};

/// Run one local case through every implementation and compare all candidates
/// against the first target. Callers can supply a sink per target to persist
/// full checkpoints; passing `None` keeps the hot path summary-only.
pub fn run_local_case(
    case: &LocalCase,
    implementations: &[&dyn LocalImplementation],
    probe: ProbeLevel,
    sinks: &mut [Option<&mut dyn CheckpointSink>],
    contract: &EquivalenceContract,
) -> anyhow::Result<CaseVerdict> {
    anyhow::ensure!(
        implementations.len() >= 2,
        "local differential runs need at least two implementations"
    );
    anyhow::ensure!(
        sinks.len() == implementations.len(),
        "one checkpoint sink slot is required per implementation"
    );
    let mut outcomes = Vec::with_capacity(implementations.len());
    for (implementation, sink) in implementations.iter().zip(sinks.iter_mut()) {
        let mut discard = DiscardSink;
        let sink: &mut dyn CheckpointSink = match sink.as_deref_mut() {
            Some(sink) => sink,
            None => &mut discard,
        };
        outcomes.push(implementation.run_case(case, probe, sink)?);
    }
    let reference = outcomes.remove(0);
    let candidates = implementations
        .iter()
        .skip(1)
        .zip(outcomes)
        .map(|(implementation, outcome)| (implementation.metadata().name, outcome))
        .collect::<Vec<_>>();
    let reference_name = implementations[0].metadata().name;
    Ok(compare_local(
        case.id.clone(),
        &reference_name,
        &reference,
        &candidates,
        contract,
    ))
}

/// Execute the same API case against each target through a host-supplied
/// executor. The executor is responsible for resolving a target to an actual
/// `hi-ai::Provider`; credentials never enter the diff core or artifacts.
pub async fn run_api_case(
    case: &ApiCase,
    targets: &[crate::ApiTarget],
    executor: &dyn ApiExecutor,
    contract: &EquivalenceContract,
) -> anyhow::Result<CaseVerdict> {
    anyhow::ensure!(
        targets.len() >= 2,
        "API differential runs need at least two targets"
    );
    let mut outcomes: Vec<(String, ApiOutcome)> = Vec::with_capacity(targets.len());
    for target in targets {
        match executor.run_response(target, case).await {
            Ok(outcome) => outcomes.push((target.name.clone(), outcome)),
            Err(error) => outcomes.push((
                target.name.clone(),
                ApiOutcome {
                    text: String::new(),
                    json: None,
                    tool_calls: Vec::new(),
                    finish_reason: None,
                    error_category: Some(error.to_string()),
                    input_tokens: 0,
                    output_tokens: 0,
                    latency_ms: 0,
                    schema_valid: None,
                },
            )),
        }
    }
    Ok(compare_response(case.id.clone(), &outcomes, contract))
}

struct DiscardSink;

impl CheckpointSink for DiscardSink {
    fn checkpoint(&mut self, _checkpoint: crate::Checkpoint<'_>) -> anyhow::Result<()> {
        Ok(())
    }
}
