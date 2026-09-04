//! Complete, inspectable identity for manifest evaluation comparability.

use std::collections::BTreeMap;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result};
use hi_eval::{EvalProfile, IdentityDetails, RunIdentity};
use hi_eval_adapters::ADAPTER_API_VERSION;

pub(crate) fn build_identity(
    profile_name: &str,
    manifest_digest: &str,
    dataset_digests: BTreeMap<String, String>,
    profile: &EvalProfile,
    workspace_root: &Path,
) -> Result<RunIdentity> {
    let hi_digest = binary_identity_digest()?;
    let digest_value = |value: &serde_json::Value| -> Result<String> {
        Ok(blake3::hash(&serde_json::to_vec(value)?)
            .to_hex()
            .to_string())
    };
    let configuration_digest = blake3::hash(&serde_json::to_vec(profile)?)
        .to_hex()
        .to_string();
    let mcp_configuration_digest = digest_value(&serde_json::to_value(&profile.mcp_servers)?)?;
    let provider_policy_digest = digest_value(&serde_json::json!({
        "policy": profile.provider_policy,
        "sampling": profile.sampling,
        "models": profile.models,
    }))?;
    let scoring_policy_digest = digest_value(&serde_json::to_value(&profile.scoring)?)?;
    let network_digest = digest_value(&serde_json::to_value(&profile.network)?)?;
    let limits_digest = digest_value(&serde_json::json!({
        "resources": profile.resources,
        "trials": profile.trials,
        "evidence": profile.evidence,
    }))?;
    let fixture_digest = digest_value(&serde_json::to_value(&dataset_digests)?)?;
    let secret_configuration_digest = profile
        .secret_configuration_digest
        .clone()
        .unwrap_or_default();
    let runtime_identity = format!(
        "{}-{}-{}",
        std::env::consts::OS,
        std::env::consts::ARCH,
        profile.backend
    );
    let materializer = if matches!(profile.backend.as_str(), "harbor" | "docker") {
        "harbor-docker-copy-v1"
    } else {
        "host-copy-v1"
    };
    let identity_dimensions = BTreeMap::from([
        ("binary".into(), hi_digest.clone()),
        ("fixtures".into(), fixture_digest),
        ("git_state".into(), git_state_digest(workspace_root)?),
        ("limits".into(), limits_digest),
        ("materializer".into(), materializer.into()),
        ("mcp".into(), mcp_configuration_digest.clone()),
        (
            "native_director".into(),
            hi_agent::NATIVE_DIRECTOR_VERSION.to_string(),
        ),
        ("network".into(), network_digest),
        ("os_arch".into(), runtime_identity.clone()),
        ("provider_model".into(), provider_policy_digest.clone()),
        (
            "session_reducer".into(),
            hi_agent::SESSION_REDUCER_VERSION.to_string(),
        ),
        (
            "tool_envelope".into(),
            hi_tools::envelope::TOOL_ENVELOPE_SCHEMA_VERSION.to_string(),
        ),
        ("workspace_backend".into(), profile.backend.clone()),
    ]);
    RunIdentity::new_with_details(
        profile_name,
        manifest_digest,
        dataset_digests,
        profile.models.clone(),
        &profile.backend,
        scoring_policy_digest,
        configuration_digest,
        IdentityDetails {
            adapter_version: ADAPTER_API_VERSION.into(),
            hi_binary_digest: hi_digest,
            provider_policy_digest,
            mcp_configuration_digest,
            secret_configuration_digest,
            runtime_identity,
            identity_dimensions,
        },
    )
}

fn git_state_digest(workspace_root: &Path) -> Result<String> {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"hi.eval.git-state.v1\0");
    let mut inventory = Vec::new();
    for arguments in [
        &["rev-parse", "--verify", "HEAD"] as &[&str],
        &["status", "--porcelain=v2", "--branch", "-z"],
        &[
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "-z",
        ],
    ] {
        let output = Command::new("git")
            .arg("-C")
            .arg(workspace_root)
            .args(arguments)
            .output()
            .context("capturing evaluation Git identity")?;
        hasher.update(&(arguments.len() as u64).to_le_bytes());
        for argument in arguments {
            hasher.update(&(argument.len() as u64).to_le_bytes());
            hasher.update(argument.as_bytes());
        }
        hasher.update(&output.status.code().unwrap_or(-1).to_le_bytes());
        hasher.update(&(output.stdout.len() as u64).to_le_bytes());
        hasher.update(&output.stdout);
        if arguments.first() == Some(&"ls-files") && output.status.success() {
            inventory = output.stdout;
        }
    }
    for raw_path in inventory
        .split(|byte| *byte == 0)
        .filter(|path| !path.is_empty())
    {
        let relative = path_from_git_bytes(raw_path);
        let path = workspace_root.join(&relative);
        let metadata = fs::symlink_metadata(&path)
            .with_context(|| format!("reading Git identity path {}", relative.display()))?;
        hasher.update(&(raw_path.len() as u64).to_le_bytes());
        hasher.update(raw_path);
        hash_file_mode(&mut hasher, &metadata);
        if metadata.file_type().is_symlink() {
            hasher.update(b"symlink\0");
            let target = fs::read_link(&path)?;
            let target = target.to_string_lossy();
            hasher.update(&(target.len() as u64).to_le_bytes());
            hasher.update(target.as_bytes());
        } else if metadata.is_file() {
            hasher.update(b"file\0");
            hasher.update(&metadata.len().to_le_bytes());
            let mut file = fs::File::open(&path)?;
            let mut buffer = [0_u8; 64 * 1024];
            loop {
                let count = file.read(&mut buffer)?;
                if count == 0 {
                    break;
                }
                hasher.update(&buffer[..count]);
            }
        } else {
            hasher.update(b"non-file\0");
        }
    }
    Ok(hasher.finalize().to_hex().to_string())
}

#[cfg(unix)]
fn path_from_git_bytes(bytes: &[u8]) -> PathBuf {
    use std::os::unix::ffi::OsStringExt;

    PathBuf::from(std::ffi::OsString::from_vec(bytes.to_vec()))
}

#[cfg(not(unix))]
fn path_from_git_bytes(bytes: &[u8]) -> PathBuf {
    PathBuf::from(String::from_utf8_lossy(bytes).as_ref())
}

#[cfg(unix)]
fn hash_file_mode(hasher: &mut blake3::Hasher, metadata: &fs::Metadata) {
    use std::os::unix::fs::MetadataExt;

    hasher.update(&metadata.mode().to_le_bytes());
}

#[cfg(not(unix))]
fn hash_file_mode(hasher: &mut blake3::Hasher, metadata: &fs::Metadata) {
    hasher.update(&[u8::from(metadata.permissions().readonly())]);
}

fn binary_identity_digest() -> Result<String> {
    let candidate = std::env::var_os("HI_BIN")
        .map(PathBuf::from)
        .filter(|path| path.is_file())
        .or_else(|| std::env::current_exe().ok().filter(|path| path.is_file()));
    let evaluator = std::env::var_os("HI_EVAL_BIN")
        .map(PathBuf::from)
        .filter(|path| path.is_file())
        .or_else(|| {
            std::env::current_exe()
                .ok()
                .and_then(|current| current.parent().map(|parent| parent.join("hi-eval")))
                .filter(|path| path.is_file())
        });
    let digest = |path: Option<PathBuf>| -> Result<Option<String>> {
        path.map(|path| Ok(blake3::hash(&fs::read(&path)?).to_hex().to_string()))
            .transpose()
    };
    let value = serde_json::json!({
        "candidate": digest(candidate)?,
        "evaluator": digest(evaluator)?,
    });
    Ok(blake3::hash(&serde_json::to_vec(&value)?)
        .to_hex()
        .to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn git_identity_tracks_tracked_and_untracked_content() {
        let root = std::env::temp_dir().join(format!(
            "hi-eval-identity-{}-{}",
            std::process::id(),
            uuid::Uuid::new_v4()
        ));
        fs::create_dir(&root).unwrap();
        let git = |arguments: &[&str]| {
            let status = Command::new("git")
                .arg("-C")
                .arg(&root)
                .args(arguments)
                .status()
                .unwrap();
            assert!(status.success());
        };
        git(&["init", "--quiet"]);
        fs::write(root.join("tracked.txt"), b"one").unwrap();
        git(&["add", "tracked.txt"]);
        git(&[
            "-c",
            "user.name=hi eval",
            "-c",
            "user.email=hi-eval@invalid",
            "commit",
            "--quiet",
            "-m",
            "base",
        ]);
        let base = git_state_digest(&root).unwrap();
        fs::write(root.join("tracked.txt"), b"two").unwrap();
        let tracked = git_state_digest(&root).unwrap();
        assert_ne!(base, tracked);
        fs::write(root.join("new.txt"), b"first").unwrap();
        let untracked = git_state_digest(&root).unwrap();
        fs::write(root.join("new.txt"), b"second").unwrap();
        assert_ne!(untracked, git_state_digest(&root).unwrap());
        fs::remove_dir_all(root).unwrap();
    }
}
