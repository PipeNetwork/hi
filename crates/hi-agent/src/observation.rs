//! Event-time evidence seam used by trusted and local observers.
//!
//! The agent deliberately knows nothing about trace layout or retention. It
//! emits complete payloads at the point where they exist; a frontend-owned
//! sink decides how to persist them. Returning an error lets mandatory
//! observers fail a managed turn immediately, while best-effort sinks may
//! disable themselves and keep returning success.

use anyhow::Result;
use serde_json::Value;

#[derive(Clone, Debug)]
pub struct Observation {
    pub kind: String,
    pub stage: String,
    pub attempt: u32,
    pub correlation_id: String,
    pub causation_hash: Option<String>,
    pub media_type: String,
    pub payload: Vec<u8>,
    pub metadata: Value,
}

impl Observation {
    pub fn json(
        kind: impl Into<String>,
        stage: impl Into<String>,
        attempt: u32,
        correlation_id: impl Into<String>,
        payload: &impl serde::Serialize,
    ) -> Result<Self> {
        Ok(Self {
            kind: kind.into(),
            stage: stage.into(),
            attempt,
            correlation_id: correlation_id.into(),
            causation_hash: None,
            media_type: "application/json".into(),
            payload: serde_json::to_vec(payload)?,
            metadata: Value::Object(Default::default()),
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ObservationReceipt {
    pub event_hash: String,
    pub sequence: u64,
}

pub trait ObservationSink: Send + Sync {
    fn observe(&self, observation: Observation) -> Result<ObservationReceipt>;
}
