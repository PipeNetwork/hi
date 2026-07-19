use std::{
    path::PathBuf,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, Result, ensure};
use hi_protocol::{
    Envelope, FramedUnix, Handshake, Message, PROTOCOL_MAJOR, PROTOCOL_MINOR, PeerRole, StageResult,
};
use hi_rsi_runtime::ManagedRuntimeDescriptor;
use tokio::net::UnixStream;

#[tokio::main]
async fn main() -> Result<()> {
    let descriptor_path = required("HI_RUNTIME_DESCRIPTOR")?;
    let socket_path = required("HI_CANDIDATE_SOCKET")?;
    let descriptor_hash = required("HI_RUNTIME_DESCRIPTOR_HASH")?;
    let nonce = required("HI_PROTOCOL_NONCE")?;
    let descriptor = ManagedRuntimeDescriptor::read(&PathBuf::from(descriptor_path), unix_ms()?)?;
    ensure!(
        descriptor.content_hash()? == descriptor_hash,
        "runtime descriptor hash mismatch"
    );
    let stream = UnixStream::connect(socket_path)
        .await
        .context("connecting to trusted bootstrap")?;
    let mut protocol = FramedUnix::new(stream);
    protocol
        .send(
            &Handshake {
                protocol_major: PROTOCOL_MAJOR,
                protocol_minor: PROTOCOL_MINOR,
                role: PeerRole::Candidate,
                descriptor_hash: descriptor_hash.clone(),
                nonce: nonce.clone(),
            },
            Duration::from_secs(1),
        )
        .await?;
    let peer: Handshake = protocol.receive(Duration::from_secs(1)).await?;
    peer.validate_peer(PeerRole::Bootstrap, &descriptor_hash, &nonce)?;
    loop {
        let envelope: Envelope = protocol.receive(Duration::from_secs(300)).await?;
        envelope.validate(unix_ms()?)?;
        match envelope.message {
            Message::ExecuteStage(request) => {
                let response = Envelope {
                    request_id: envelope.request_id,
                    deadline_unix_ms: envelope.deadline_unix_ms,
                    message: Message::StageResult(StageResult {
                        stage: request.stage,
                        passed: true,
                        output: request.input,
                    }),
                };
                protocol.send(&response, Duration::from_secs(5)).await?;
            }
            Message::Cancel { .. } | Message::Shutdown => break,
            _ => anyhow::bail!("bootstrap sent a message not accepted by the candidate"),
        }
    }
    Ok(())
}

fn required(name: &str) -> Result<String> {
    std::env::var(name).with_context(|| format!("missing {name}"))
}
fn unix_ms() -> Result<u64> {
    Ok(SystemTime::now()
        .duration_since(UNIX_EPOCH)?
        .as_millis()
        .try_into()?)
}
