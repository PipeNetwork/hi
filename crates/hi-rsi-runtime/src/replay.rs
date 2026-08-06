use std::collections::{BTreeMap, VecDeque};

use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ReplayKind {
    Model,
    Tool,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RecordedExchange {
    pub kind: ReplayKind,
    pub correlation_id: String,
    pub request_hash: String,
    pub response_artifact_hash: String,
    pub budget_before_hash: String,
    pub budget_after_hash: String,
}

#[derive(Debug, Error, Eq, PartialEq)]
pub enum ReplayError {
    #[error("replay request {0} was not recorded")]
    Missing(String),
    #[error("replay request hash differs for {0}")]
    RequestMismatch(String),
    #[error("replay budget state differs before {0}")]
    BudgetMismatch(String),
    #[error("replay contains unused recorded exchanges")]
    UnusedExchanges,
    #[error("replay contains duplicate correlation id {0}")]
    Duplicate(String),
}

/// Deterministic response index for orchestration replay. It deliberately
/// stores artifact hashes rather than response bodies; the artifact store owns
/// authorization and content retrieval.
#[derive(Clone, Debug)]
pub struct ExactReplay {
    exchanges: BTreeMap<String, RecordedExchange>,
    order: VecDeque<String>,
}

impl ExactReplay {
    pub fn new(exchanges: Vec<RecordedExchange>) -> Result<Self, ReplayError> {
        let mut indexed = BTreeMap::new();
        let mut order = VecDeque::new();
        for exchange in exchanges {
            let id = exchange.correlation_id.clone();
            if indexed.insert(id.clone(), exchange).is_some() {
                return Err(ReplayError::Duplicate(id));
            }
            order.push_back(id);
        }
        Ok(Self {
            exchanges: indexed,
            order,
        })
    }

    pub fn resolve(
        &mut self,
        kind: ReplayKind,
        correlation_id: &str,
        request_hash: &str,
        budget_before_hash: &str,
    ) -> Result<RecordedExchange, ReplayError> {
        // Validate against the recorded exchange without consuming it: a failed
        // resolution must leave the recording intact so the caller can retry
        // (or audit) the exchange instead of losing it to a transient mismatch.
        let exchange = self
            .exchanges
            .get(correlation_id)
            .ok_or_else(|| ReplayError::Missing(correlation_id.into()))?;
        if exchange.kind != kind || exchange.request_hash != request_hash {
            return Err(ReplayError::RequestMismatch(correlation_id.into()));
        }
        if exchange.budget_before_hash != budget_before_hash {
            return Err(ReplayError::BudgetMismatch(correlation_id.into()));
        }
        if self.order.front().map(String::as_str) != Some(correlation_id) {
            return Err(ReplayError::RequestMismatch(correlation_id.into()));
        }
        self.order.pop_front();
        Ok(self
            .exchanges
            .remove(correlation_id)
            .expect("exchange presence checked above"))
    }

    pub fn finish(self) -> Result<(), ReplayError> {
        if self.exchanges.is_empty() && self.order.is_empty() {
            Ok(())
        } else {
            Err(ReplayError::UnusedExchanges)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn exchange(id: &str) -> RecordedExchange {
        RecordedExchange {
            kind: ReplayKind::Model,
            correlation_id: id.into(),
            request_hash: format!("request-{id}"),
            response_artifact_hash: format!("response-{id}"),
            budget_before_hash: format!("before-{id}"),
            budget_after_hash: format!("after-{id}"),
        }
    }

    #[test]
    fn exact_replay_enforces_request_order_and_budget_state() {
        let mut replay = ExactReplay::new(vec![exchange("one"), exchange("two")]).unwrap();
        assert!(
            replay
                .resolve(ReplayKind::Model, "two", "request-two", "before-two")
                .is_err()
        );
        let mut replay = ExactReplay::new(vec![exchange("one"), exchange("two")]).unwrap();
        replay
            .resolve(ReplayKind::Model, "one", "request-one", "before-one")
            .unwrap();
        replay
            .resolve(ReplayKind::Model, "two", "request-two", "before-two")
            .unwrap();
        replay.finish().unwrap();
    }

    #[test]
    fn failed_resolve_leaves_the_recording_intact() {
        let mut replay = ExactReplay::new(vec![exchange("one"), exchange("two")]).unwrap();

        // Out-of-order: "two" is validated but neither exchange is consumed,
        // so the correct order still resolves afterwards.
        assert_eq!(
            replay.resolve(ReplayKind::Model, "two", "request-two", "before-two"),
            Err(ReplayError::RequestMismatch("two".into()))
        );
        assert_eq!(
            replay.resolve(ReplayKind::Model, "one", "request-one", "before-one"),
            Ok(exchange("one"))
        );
        assert_eq!(
            replay.resolve(ReplayKind::Model, "two", "request-two", "before-two"),
            Ok(exchange("two"))
        );
        replay.finish().unwrap();

        // Mismatched request hash: the exchange stays recorded, and a retry
        // with the correct hash resolves it.
        let mut replay = ExactReplay::new(vec![exchange("one")]).unwrap();
        assert_eq!(
            replay.resolve(ReplayKind::Model, "one", "wrong-request", "before-one"),
            Err(ReplayError::RequestMismatch("one".into()))
        );
        assert_eq!(
            replay.resolve(ReplayKind::Model, "one", "request-one", "before-one"),
            Ok(exchange("one"))
        );
        replay.finish().unwrap();

        // Mismatched budget state: same non-destructive guarantee.
        let mut replay = ExactReplay::new(vec![exchange("one")]).unwrap();
        assert_eq!(
            replay.resolve(ReplayKind::Model, "one", "request-one", "wrong-before"),
            Err(ReplayError::BudgetMismatch("one".into()))
        );
        assert_eq!(
            replay.resolve(ReplayKind::Model, "one", "request-one", "before-one"),
            Ok(exchange("one"))
        );
        replay.finish().unwrap();
    }
}
