//! Exact reconstruction and independent stochastic replay orchestration.

use std::collections::BTreeMap;

use anyhow::{Result, ensure};
use async_trait::async_trait;
use hi_rsi_runtime::{Checkpoint, ExactReplay, RecordedExchange, ReplayKind, WorkflowGraph};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ReconstructionPackage {
    pub schema_version: u16,
    pub repository_snapshot_hash: String,
    pub manifest_hash: String,
    pub workflow: WorkflowGraph,
    pub checkpoint: Checkpoint,
    pub exchanges: Vec<RecordedExchange>,
    pub response_artifacts: BTreeMap<String, Vec<u8>>,
    pub mutation_artifacts: Vec<String>,
    pub expected_transition_hashes: Vec<String>,
    pub expected_final_budget_hash: String,
}

pub struct ExactReconstruction {
    replay: ExactReplay,
    responses: BTreeMap<String, Vec<u8>>,
    expected_transitions: Vec<String>,
    transition_index: usize,
}

impl ExactReconstruction {
    pub fn new(package: &ReconstructionPackage) -> Result<Self> {
        ensure!(
            package.schema_version == 1,
            "unsupported reconstruction package"
        );
        ensure!(
            package.checkpoint.state.repository.repository_snapshot_hash
                == package.repository_snapshot_hash,
            "reconstruction snapshot mismatch"
        );
        for exchange in &package.exchanges {
            ensure!(
                package
                    .response_artifacts
                    .contains_key(&exchange.response_artifact_hash),
                "recorded response artifact is absent"
            );
        }
        Ok(Self {
            replay: ExactReplay::new(package.exchanges.clone())?,
            responses: package.response_artifacts.clone(),
            expected_transitions: package.expected_transition_hashes.clone(),
            transition_index: 0,
        })
    }

    pub fn resolve(
        &mut self,
        kind: ReplayKind,
        correlation_id: &str,
        request_bytes: &[u8],
        budget_before_hash: &str,
    ) -> Result<Vec<u8>> {
        let request_hash = blake3::hash(request_bytes).to_hex().to_string();
        let exchange =
            self.replay
                .resolve(kind, correlation_id, &request_hash, budget_before_hash)?;
        let response = self
            .responses
            .get(&exchange.response_artifact_hash)
            .cloned()
            .expect("validated response");
        ensure!(
            blake3::hash(&response).to_hex().as_str() == exchange.response_artifact_hash,
            "response artifact hash mismatch"
        );
        Ok(response)
    }

    pub fn observe_transition(&mut self, state_bytes: &[u8]) -> Result<()> {
        let expected = self
            .expected_transitions
            .get(self.transition_index)
            .ok_or_else(|| anyhow::anyhow!("undeclared replay transition"))?;
        ensure!(
            blake3::hash(state_bytes).to_hex().as_str() == expected,
            "replay state transition differs"
        );
        self.transition_index += 1;
        Ok(())
    }

    pub fn finish(self) -> Result<()> {
        ensure!(
            self.transition_index == self.expected_transitions.len(),
            "replay omitted state transitions"
        );
        self.replay.finish()?;
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AttemptResult {
    pub attempt: u32,
    pub snapshot_hash: String,
    pub outcome_hash: String,
    pub model_revision: String,
    pub equivalence_deviations: Vec<String>,
}

#[async_trait]
pub trait StochasticAttemptRunner: Send + Sync {
    async fn run_attempt(&self, snapshot_hash: &str, attempt: u32) -> Result<AttemptResult>;
}

pub async fn stochastic_replay<R: StochasticAttemptRunner>(
    runner: &R,
    snapshot_hash: &str,
    attempts: u32,
) -> Result<Vec<AttemptResult>> {
    ensure!(
        attempts > 0 && attempts <= 1000,
        "invalid stochastic replay attempt count"
    );
    let mut results = Vec::with_capacity(attempts as usize);
    for attempt in 1..=attempts {
        let result = runner.run_attempt(snapshot_hash, attempt).await?;
        ensure!(
            result.attempt == attempt && result.snapshot_hash == snapshot_hash,
            "stochastic attempt did not use the exact snapshot"
        );
        results.push(result);
    }
    Ok(results)
}

/// Interactive event replay kept separate from the exact RSI reconstruction
/// protocol. Events are already redacted at the session boundary; this type
/// only provides deterministic navigation for UIs and tests.
pub struct InteractiveEventReplay {
    events: Vec<hi_agent::AgentEvent>,
    cursor: usize,
}

impl InteractiveEventReplay {
    pub fn new(events: Vec<hi_agent::AgentEvent>) -> Self {
        Self { events, cursor: 0 }
    }

    pub fn from_jsonl(text: &str) -> Result<Self> {
        let events = text
            .lines()
            .filter(|line| !line.trim().is_empty())
            .map(serde_json::from_str)
            .collect::<std::result::Result<Vec<hi_agent::AgentEvent>, _>>()?;
        Ok(Self::new(events))
    }

    pub fn next(&mut self) -> Option<&hi_agent::AgentEvent> {
        let event = self.events.get(self.cursor)?;
        self.cursor += 1;
        Some(event)
    }

    pub fn seek_sequence(&mut self, sequence: u64) {
        self.cursor = self
            .events
            .iter()
            .position(|event| event.sequence >= sequence)
            .unwrap_or(self.events.len());
    }

    pub fn reset(&mut self) {
        self.cursor = 0;
    }
    pub fn finished(&self) -> bool {
        self.cursor >= self.events.len()
    }
    pub fn events(&self) -> &[hi_agent::AgentEvent] {
        &self.events
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    struct Runner;
    #[async_trait]
    impl StochasticAttemptRunner for Runner {
        async fn run_attempt(&self, snapshot: &str, attempt: u32) -> Result<AttemptResult> {
            Ok(AttemptResult {
                attempt,
                snapshot_hash: snapshot.into(),
                outcome_hash: format!("outcome-{attempt}"),
                model_revision: "model-v1".into(),
                equivalence_deviations: vec![],
            })
        }
    }
    #[tokio::test]
    async fn stochastic_attempts_are_independent_and_snapshot_bound() {
        let results = stochastic_replay(&Runner, "snapshot", 3).await.unwrap();
        assert_eq!(results.len(), 3);
        assert_ne!(results[0].outcome_hash, results[1].outcome_hash);
    }

    #[test]
    fn interactive_event_replay_is_seekable() {
        let event = hi_agent::AgentEvent {
            session_id: "s".into(),
            turn_id: None,
            sequence: 4,
            timestamp_ms: 1,
            kind: hi_agent::AgentEventKind::Status {
                text: "ready".into(),
            },
        };
        let encoded = serde_json::to_string(&event).unwrap();
        let mut replay = InteractiveEventReplay::from_jsonl(&encoded).unwrap();
        replay.seek_sequence(4);
        assert_eq!(replay.next().unwrap().sequence, 4);
        assert!(replay.finished());
    }
}
