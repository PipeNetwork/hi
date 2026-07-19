use std::{
    fs,
    os::unix::fs::PermissionsExt,
    path::{Path, PathBuf},
    process::Stdio,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, Result, ensure};
use hi_protocol::{
    Envelope, FramedUnix, Handshake, Message, PROTOCOL_MAJOR, PROTOCOL_MINOR, PeerRole,
};
use hi_rsi_runtime::ManagedRuntimeDescriptor;
use tokio::{net::UnixListener, process::Command, time::timeout};

#[tokio::main]
async fn main() -> Result<()> {
    let mut arguments = std::env::args_os().skip(1);
    let descriptor_path = PathBuf::from(
        arguments
            .next()
            .context("usage: hi-bootstrap DESCRIPTOR CANDIDATE SOCKET")?,
    );
    let candidate = PathBuf::from(
        arguments
            .next()
            .context("usage: hi-bootstrap DESCRIPTOR CANDIDATE SOCKET")?,
    );
    let socket = PathBuf::from(
        arguments
            .next()
            .context("usage: hi-bootstrap DESCRIPTOR CANDIDATE SOCKET")?,
    );
    ensure!(arguments.next().is_none(), "unexpected bootstrap argument");
    let descriptor = ManagedRuntimeDescriptor::read(&descriptor_path, unix_ms()?)?;
    let descriptor_hash = descriptor.content_hash()?;
    ensure!(
        hash_file(&candidate)? == descriptor.identity.agent_artifact_hash,
        "candidate executable hash differs from signed descriptor"
    );
    ensure_regular_executable(&candidate)?;
    if socket.exists() {
        fs::remove_file(&socket)?;
    }
    let listener = UnixListener::bind(&socket).context("binding candidate protocol socket")?;
    fs::set_permissions(&socket, fs::Permissions::from_mode(0o600))?;
    let nonce = format!("{}-{}", std::process::id(), unix_ms()?);
    let mut child = Command::new(&candidate);
    child
        .env_clear()
        .env("HI_RUNTIME_DESCRIPTOR", &descriptor_path)
        .env("HI_CANDIDATE_SOCKET", &socket)
        .env("HI_RUNTIME_DESCRIPTOR_HASH", &descriptor_hash)
        .env("HI_PROTOCOL_NONCE", &nonce)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    let mut child = child.spawn().context("launching candidate executable")?;
    let result = async {
        let (stream, _) = timeout(Duration::from_secs(5), listener.accept())
            .await
            .context("candidate protocol connection deadline")??;
        let mut protocol = FramedUnix::new(stream);
        let peer: Handshake = protocol.receive(Duration::from_secs(1)).await?;
        peer.validate_peer(PeerRole::Candidate, &descriptor_hash, &nonce)?;
        protocol
            .send(
                &Handshake {
                    protocol_major: PROTOCOL_MAJOR,
                    protocol_minor: PROTOCOL_MINOR,
                    role: PeerRole::Bootstrap,
                    descriptor_hash: descriptor_hash.clone(),
                    nonce: nonce.clone(),
                },
                Duration::from_secs(1),
            )
            .await?;
        protocol
            .send(
                &Envelope {
                    request_id: "shutdown-1".into(),
                    deadline_unix_ms: unix_ms()? + 1_000,
                    message: Message::Shutdown,
                },
                Duration::from_secs(1),
            )
            .await?;
        let status = timeout(Duration::from_secs(5), child.wait())
            .await
            .context("candidate cancellation deadline")??;
        ensure!(status.success(), "candidate exited unsuccessfully");
        Ok::<_, anyhow::Error>(())
    }
    .await;
    if result.is_err() {
        let _ = child.kill().await;
    }
    let _ = fs::remove_file(&socket);
    result?;
    println!(
        "{}",
        serde_json::json!({
            "schema_version": 1, "run_id": descriptor.identity.run_id,
            "candidate_id": descriptor.identity.candidate_id, "manifest_hash": descriptor.identity.manifest_hash,
            "runtime_descriptor_hash": descriptor_hash, "candidate_artifact_hash": descriptor.identity.agent_artifact_hash,
            "status": "completed_pending_verification"
        })
    );
    Ok(())
}

fn hash_file(path: &Path) -> Result<String> {
    Ok(blake3::hash(&fs::read(path)?).to_hex().to_string())
}
fn ensure_regular_executable(path: &Path) -> Result<()> {
    let metadata = fs::symlink_metadata(path)?;
    ensure!(
        metadata.is_file() && !metadata.file_type().is_symlink(),
        "candidate executable must be a regular file"
    );
    ensure!(
        metadata.permissions().mode() & 0o111 != 0,
        "candidate artifact is not executable"
    );
    Ok(())
}
fn unix_ms() -> Result<u64> {
    Ok(SystemTime::now()
        .duration_since(UNIX_EPOCH)?
        .as_millis()
        .try_into()?)
}
