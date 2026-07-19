//! Strict, bounded local protocol between the immutable bootstrap and candidate.

use std::{io, time::Duration};

use hi_rsi_runtime::{FailureEvidence, StageId, VerificationReport};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use serde_json::Value;
use thiserror::Error;
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::UnixStream,
    time::timeout,
};

pub const PROTOCOL_MAJOR: u16 = 1;
pub const PROTOCOL_MINOR: u16 = 0;
pub const MAX_FRAME_BYTES: usize = 1024 * 1024;

#[derive(Debug, Error)]
pub enum ProtocolError {
    #[error("protocol I/O: {0}")]
    Io(#[from] io::Error),
    #[error("protocol deadline exceeded")]
    Deadline,
    #[error("invalid protocol frame: {0}")]
    Invalid(String),
    #[error("protocol frame is too large")]
    FrameTooLarge,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PeerRole {
    Bootstrap,
    Candidate,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Handshake {
    pub protocol_major: u16,
    pub protocol_minor: u16,
    pub role: PeerRole,
    pub descriptor_hash: String,
    pub nonce: String,
}

impl Handshake {
    pub fn validate_peer(
        &self,
        role: PeerRole,
        descriptor_hash: &str,
        nonce: &str,
    ) -> Result<(), ProtocolError> {
        if self.protocol_major != PROTOCOL_MAJOR
            || self.protocol_minor > PROTOCOL_MINOR
            || self.role != role
            || self.descriptor_hash != descriptor_hash
            || self.nonce != nonce
        {
            return Err(ProtocolError::Invalid("handshake identity mismatch".into()));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Envelope {
    pub request_id: String,
    pub deadline_unix_ms: u64,
    pub message: Message,
}

impl Envelope {
    pub fn validate(&self, now_unix_ms: u64) -> Result<(), ProtocolError> {
        if self.request_id.is_empty()
            || self.request_id.len() > 128
            || !self
                .request_id
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b"-_.".contains(&b))
        {
            return Err(ProtocolError::Invalid("invalid request id".into()));
        }
        if self.deadline_unix_ms <= now_unix_ms {
            return Err(ProtocolError::Deadline);
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "type", content = "payload", rename_all = "snake_case")]
pub enum Message {
    Handshake(Handshake),
    ExecuteStage(StageRequest),
    StageResult(StageResult),
    ToolRequest(ToolRequest),
    ToolResult(ToolResult),
    ModelRequest(ModelRequest),
    ModelResult(ModelResult),
    Checkpoint(CheckpointRequest),
    VerificationProposal(VerificationReport),
    Failure(FailureEvidence),
    Cancel { reason: String },
    Shutdown,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct StageRequest {
    pub stage: StageId,
    pub attempt: u32,
    pub input: Value,
    pub output_schema: Option<Value>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct StageResult {
    pub stage: StageId,
    pub passed: bool,
    pub output: Value,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ToolRequest {
    pub name: String,
    pub stage: StageId,
    pub arguments: Value,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ToolResult {
    pub name: String,
    pub succeeded: bool,
    pub output: Value,
    pub artifact_hashes: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ModelRequest {
    pub role: String,
    pub stage: StageId,
    pub prompt_hash: String,
    pub context_hash: String,
    pub output_schema: Value,
    pub maximum_tokens: u32,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ModelResult {
    pub resolved_route: String,
    pub output: Value,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cost_microusd: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CheckpointRequest {
    pub reason: String,
    pub workspace_tree_hash: String,
}

pub struct FramedUnix {
    stream: UnixStream,
    maximum_frame_bytes: usize,
}

impl FramedUnix {
    pub fn new(stream: UnixStream) -> Self {
        Self {
            stream,
            maximum_frame_bytes: MAX_FRAME_BYTES,
        }
    }

    pub fn with_limit(
        stream: UnixStream,
        maximum_frame_bytes: usize,
    ) -> Result<Self, ProtocolError> {
        if maximum_frame_bytes == 0 || maximum_frame_bytes > MAX_FRAME_BYTES {
            return Err(ProtocolError::Invalid("invalid frame limit".into()));
        }
        Ok(Self {
            stream,
            maximum_frame_bytes,
        })
    }

    pub async fn send<T: Serialize>(
        &mut self,
        value: &T,
        deadline: Duration,
    ) -> Result<(), ProtocolError> {
        let bytes = serde_json::to_vec(value).map_err(|e| ProtocolError::Invalid(e.to_string()))?;
        if bytes.len() > self.maximum_frame_bytes {
            return Err(ProtocolError::FrameTooLarge);
        }
        timeout(deadline, async {
            self.stream.write_u32(bytes.len() as u32).await?;
            self.stream.write_all(&bytes).await?;
            self.stream.flush().await
        })
        .await
        .map_err(|_| ProtocolError::Deadline)??;
        Ok(())
    }

    pub async fn receive<T: DeserializeOwned>(
        &mut self,
        deadline: Duration,
    ) -> Result<T, ProtocolError> {
        let bytes = timeout(deadline, async {
            let length = self.stream.read_u32().await? as usize;
            if length == 0 || length > self.maximum_frame_bytes {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "invalid frame length",
                ));
            }
            let mut bytes = vec![0; length];
            self.stream.read_exact(&mut bytes).await?;
            Ok::<_, io::Error>(bytes)
        })
        .await
        .map_err(|_| ProtocolError::Deadline)??;
        serde_json::from_slice(&bytes).map_err(|e| ProtocolError::Invalid(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn bounded_frames_round_trip_and_reject_unknown_fields() {
        let (left, right) = UnixStream::pair().unwrap();
        let mut tx = FramedUnix::new(left);
        let mut rx = FramedUnix::new(right);
        let handshake = Handshake {
            protocol_major: 1,
            protocol_minor: 0,
            role: PeerRole::Candidate,
            descriptor_hash: "a".repeat(64),
            nonce: "nonce".into(),
        };
        tx.send(&handshake, Duration::from_secs(1)).await.unwrap();
        assert_eq!(
            rx.receive::<Handshake>(Duration::from_secs(1))
                .await
                .unwrap(),
            handshake
        );

        let (left, right) = UnixStream::pair().unwrap();
        let mut tx = FramedUnix::new(left);
        let mut rx = FramedUnix::new(right);
        tx.send(
            &serde_json::json!({
                "protocol_major": 1, "protocol_minor": 0, "role": "candidate",
                "descriptor_hash": "a", "nonce": "n", "forged": true
            }),
            Duration::from_secs(1),
        )
        .await
        .unwrap();
        assert!(matches!(
            rx.receive::<Handshake>(Duration::from_secs(1)).await,
            Err(ProtocolError::Invalid(_))
        ));
    }
}
