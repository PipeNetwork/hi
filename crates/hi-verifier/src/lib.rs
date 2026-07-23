//! Trusted, post-edit **RSI control-plane** verification with supervisor-owned
//! attestation.
//!
//! Distinct from [`hi_agent`]'s interactive [`RepairVerifier`] (name in the
//! agent crate), which runs compile/test stages inside the turn loop and feeds
//! failures back to the model. This crate produces a hashed
//! [`hi_rsi_runtime::VerificationReport`] and only the supervisor/evaluator
//! attaches an attestation.

use std::{fs, path::Path, process::Stdio, time::Instant};

use anyhow::{Context, Result, ensure};
use hi_rsi_runtime::{ArtifactRef, VerificationCheck, VerificationReport, VerificationStatus};
use tokio::{
    process::Command,
    time::{Duration, timeout},
};

#[derive(Clone, Debug)]
pub struct CheckSpec {
    pub name: String,
    pub program: String,
    pub arguments: Vec<String>,
    pub timeout: Duration,
    pub required: bool,
}

pub trait Attestor: Send + Sync {
    fn attest(&self, report_hash: &[u8; 32]) -> Result<String>;
}

pub struct AttestingVerifier<A> {
    attestor: A,
    environment_hash: String,
    maximum_output_bytes: usize,
}

impl<A: Attestor> AttestingVerifier<A> {
    pub fn new(attestor: A, environment_hash: String) -> Result<Self> {
        ensure_hash(&environment_hash)?;
        Ok(Self {
            attestor,
            environment_hash,
            maximum_output_bytes: 4 * 1024 * 1024,
        })
    }

    pub async fn verify(
        &self,
        workspace: &Path,
        run_id: &str,
        candidate_id: &str,
        specs: &[CheckSpec],
    ) -> Result<VerificationReport> {
        let source_tree_hash = hash_tree(workspace)?;
        let mut checks = Vec::with_capacity(specs.len());
        let mut artifacts = Vec::new();
        let mut policy_violations = Vec::new();
        for spec in specs {
            let started = Instant::now();
            let command_hash = command_hash(spec);
            let mut command = Command::new(&spec.program);
            command
                .args(&spec.arguments)
                .current_dir(workspace)
                .env_clear()
                .env("PATH", "/usr/bin:/bin")
                .env("HOME", "/nonexistent")
                .stdin(Stdio::null())
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .kill_on_drop(true);
            let (status, exit_code, output) = run_check(command, spec.timeout).await;
            let output_ref = artifact(&output, self.maximum_output_bytes);
            artifacts.push(output_ref.clone());
            if spec.required && status != VerificationStatus::Passed {
                policy_violations
                    .push(format!("required verification check failed: {}", spec.name));
            }
            checks.push(VerificationCheck {
                name: spec.name.clone(),
                command_hash,
                status,
                exit_code,
                duration_ms: started.elapsed().as_millis().min(u128::from(u64::MAX)) as u64,
                output: Some(output_ref),
            });
        }
        let passed = checks
            .iter()
            .all(|c| c.status == VerificationStatus::Passed)
            && policy_violations.is_empty();
        let mut report = VerificationReport {
            report_version: 1,
            run_id: run_id.into(),
            candidate_id: candidate_id.into(),
            environment_hash: self.environment_hash.clone(),
            source_tree_hash,
            checks,
            passed,
            policy_violations,
            artifacts,
            supervisor_attestation: None,
        };
        let unsigned = serde_json::to_vec(&report)?;
        report.supervisor_attestation =
            Some(self.attestor.attest(blake3::hash(&unsigned).as_bytes())?);
        report.validate_supervisor_report()?;
        Ok(report)
    }
}

pub fn rust_default_checks(include_workspace_tests: bool) -> Vec<CheckSpec> {
    let mut checks = vec![
        cargo("format", ["fmt", "--all", "--", "--check"], 120),
        cargo(
            "check",
            ["check", "--workspace", "--all-targets", "--all-features"],
            600,
        ),
        cargo(
            "clippy",
            [
                "clippy",
                "--workspace",
                "--all-targets",
                "--all-features",
                "--",
                "-D",
                "warnings",
            ],
            900,
        ),
    ];
    if include_workspace_tests {
        checks.push(cargo(
            "workspace_tests",
            ["test", "--workspace", "--all-features"],
            1200,
        ));
    }
    checks
}

fn cargo<const N: usize>(name: &str, args: [&str; N], seconds: u64) -> CheckSpec {
    CheckSpec {
        name: name.into(),
        program: "cargo".into(),
        arguments: args.into_iter().map(str::to_owned).collect(),
        timeout: Duration::from_secs(seconds),
        required: true,
    }
}

async fn run_check(
    mut command: Command,
    deadline: Duration,
) -> (VerificationStatus, Option<i32>, Vec<u8>) {
    let mut child = match command.spawn() {
        Ok(child) => child,
        Err(error) => {
            return (
                VerificationStatus::InfrastructureError,
                None,
                error.to_string().into_bytes(),
            );
        }
    };

    match timeout(deadline, child.wait()).await {
        Ok(Ok(status)) => {
            let stdout = read_pipe(child.stdout.take()).await;
            let stderr = read_pipe(child.stderr.take()).await;
            let output = match (stdout, stderr) {
                (Ok(stdout), Ok(stderr)) => [stdout, stderr].concat(),
                (Err(error), _) | (_, Err(error)) => {
                    return (
                        VerificationStatus::InfrastructureError,
                        None,
                        error.to_string().into_bytes(),
                    );
                }
            };
            let verification_status = if status.success() {
                VerificationStatus::Passed
            } else {
                VerificationStatus::Failed
            };
            (verification_status, status.code(), output)
        }
        Ok(Err(error)) => (
            VerificationStatus::InfrastructureError,
            None,
            error.to_string().into_bytes(),
        ),
        Err(_) => {
            let _ = child.kill().await;
            let _ = child.wait().await;
            (
                VerificationStatus::Failed,
                None,
                b"verification deadline exceeded".to_vec(),
            )
        }
    }
}

async fn read_pipe<R: tokio::io::AsyncRead + Unpin>(pipe: Option<R>) -> std::io::Result<Vec<u8>> {
    use tokio::io::AsyncReadExt;

    let mut bytes = Vec::new();
    if let Some(mut pipe) = pipe {
        pipe.read_to_end(&mut bytes).await?;
    }
    Ok(bytes)
}

fn command_hash(spec: &CheckSpec) -> String {
    let mut hash = blake3::Hasher::new();
    hash.update(spec.program.as_bytes());
    for argument in &spec.arguments {
        hash.update(&[0]);
        hash.update(argument.as_bytes());
    }
    hash.finalize().to_hex().to_string()
}

fn artifact(output: &[u8], ceiling: usize) -> ArtifactRef {
    ArtifactRef {
        hash: blake3::hash(output).to_hex().to_string(),
        size_bytes: output.len().min(ceiling) as u64,
        media_type: "text/plain".into(),
    }
}

pub fn hash_tree(root: &Path) -> Result<String> {
    let root = root
        .canonicalize()
        .context("canonicalizing verification workspace")?;
    let mut files = Vec::new();
    walk(&root, &root, &mut files)?;
    files.sort();
    let mut hasher = blake3::Hasher::new();
    for relative in files {
        hasher.update(relative.to_string_lossy().as_bytes());
        hasher.update(&[0]);
        let bytes = fs::read(root.join(&relative))?;
        hasher.update(&(bytes.len() as u64).to_le_bytes());
        hasher.update(&bytes);
    }
    Ok(hasher.finalize().to_hex().to_string())
}

fn walk(root: &Path, directory: &Path, files: &mut Vec<std::path::PathBuf>) -> Result<()> {
    for entry in fs::read_dir(directory)? {
        let entry = entry?;
        let metadata = entry.file_type()?;
        ensure!(
            !metadata.is_symlink(),
            "verification workspace contains a symlink"
        );
        if metadata.is_dir() {
            walk(root, &entry.path(), files)?;
        } else if metadata.is_file() {
            files.push(entry.path().strip_prefix(root)?.to_owned());
        } else {
            anyhow::bail!("verification workspace contains a special file");
        }
    }
    Ok(())
}

fn ensure_hash(value: &str) -> Result<()> {
    ensure!(
        value.len() == 64
            && value
                .bytes()
                .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase()),
        "invalid environment hash"
    );
    Ok(())
}

/// Backward-compatible name for [`AttestingVerifier`].
pub type Verifier<A> = AttestingVerifier<A>;

#[cfg(test)]
mod tests {
    use super::*;
    struct TestAttestor;
    impl Attestor for TestAttestor {
        fn attest(&self, hash: &[u8; 32]) -> Result<String> {
            Ok(format!("test:{}", blake3::Hash::from_bytes(*hash).to_hex()))
        }
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn timed_out_check_is_terminated() {
        let workspace = tempfile::tempdir().unwrap();
        fs::write(workspace.path().join("a"), "content").unwrap();
        let marker = workspace.path().join("survived");
        let verifier = AttestingVerifier::new(TestAttestor, "a".repeat(64)).unwrap();
        let report = verifier
            .verify(
                workspace.path(),
                "run",
                "candidate",
                &[CheckSpec {
                    name: "slow".into(),
                    program: "/bin/sh".into(),
                    arguments: vec![
                        "-c".into(),
                        format!("sleep 1; printf survived > {}", marker.display()),
                    ],
                    timeout: Duration::from_millis(20),
                    required: true,
                }],
            )
            .await
            .unwrap();
        assert!(!report.passed);
        tokio::time::sleep(Duration::from_millis(1_200)).await;
        assert!(!marker.exists(), "timed-out verifier command kept running");
    }

    #[tokio::test]
    async fn trusted_report_cannot_claim_a_failed_check_passed() {
        let workspace = tempfile::tempdir().unwrap();
        fs::write(workspace.path().join("a"), "content").unwrap();
        let verifier = AttestingVerifier::new(TestAttestor, "a".repeat(64)).unwrap();
        let report = verifier
            .verify(
                workspace.path(),
                "run",
                "candidate",
                &[CheckSpec {
                    name: "false".into(),
                    program: "false".into(),
                    arguments: vec![],
                    timeout: Duration::from_secs(1),
                    required: true,
                }],
            )
            .await
            .unwrap();
        assert!(!report.passed);
        assert!(report.supervisor_attestation.is_some());
    }
}
