//! Execution backend contracts used by profile-driven evaluation.
//!
//! The host implementation intentionally delegates to the existing hi-eval
//! task runner. Harbor/Docker and the strict Linux runtime share the same
//! package boundary but are kept separate so host execution cannot silently
//! claim container isolation.

use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result, bail};

use crate::platform::{AttemptRecord, EnvironmentSpec, EvalAttempt, EvalBackend, TaskPackage};

/// Existing fixture-copying/schema-v2 execution, exposed through the generic
/// backend seam during migration.
#[derive(Clone, Debug)]
pub struct LegacyHostBackend {
    pub evaluator: PathBuf,
    pub treatments: Vec<String>,
    pub model: Option<String>,
}

impl LegacyHostBackend {
    pub fn new(evaluator: impl Into<PathBuf>) -> Self {
        Self {
            evaluator: evaluator.into(),
            treatments: vec!["baseline".into()],
            model: None,
        }
    }
}

impl EvalBackend for LegacyHostBackend {
    fn name(&self) -> &str {
        "legacy-host"
    }

    fn prepare(&self, task: &TaskPackage) -> Result<()> {
        task.validate()?;
        if !matches!(task.environment, EnvironmentSpec::Host) {
            bail!("legacy host backend cannot execute a non-host environment");
        }
        if !matches!(task.output, crate::EvalOutput::Workspace) {
            bail!("legacy host backend only supports workspace output");
        }
        if !self.evaluator.is_file() {
            bail!(
                "hi-eval binary does not exist: {}",
                self.evaluator.display()
            );
        }
        Ok(())
    }

    fn execute(
        &self,
        task: &TaskPackage,
        attempt: &EvalAttempt,
        root: &Path,
    ) -> Result<AttemptRecord> {
        self.prepare(task)?;
        let artifacts = root.join("evidence");
        std::fs::create_dir_all(&artifacts)?;
        let mut command = Command::new(&self.evaluator);
        command
            .arg(root)
            .arg(format!("--artifacts={}", artifacts.display()))
            .arg("--trials=1")
            .arg(format!("--configs={}", self.treatments.join(",")))
            .current_dir(root);
        if let Some(model) = &self.model {
            command.env("HI_MODEL", model);
        }
        let status = command
            .status()
            .with_context(|| format!("launching legacy evaluator for {}", task.id))?;
        Ok(AttemptRecord {
            profile: String::new(),
            task: attempt.task.clone(),
            arm: attempt.arm.clone(),
            trial: attempt.trial,
            status: if status.success() {
                crate::AttemptStatus::Passed
            } else {
                crate::AttemptStatus::InfrastructureFailed
            },
            identity_digest: attempt.identity_digest.clone(),
            claim_level: task.claim_level,
            score: None,
            evidence: Some(crate::EvalEvidence {
                verifier_log: Some(artifacts.join("summary.json")),
                backend: Some(self.name().into()),
                task_digest: Some(task.source.digest.clone()),
                claim_level: task.claim_level,
                ..crate::EvalEvidence::default()
            }),
        })
    }
}

/// Manifest-driven Docker/Harbor boundary. The concrete Harbor runner can
/// supply its own command and image policy; this type makes the isolation
/// choice explicit and rejects host fallback.
#[derive(Clone, Debug, Default)]
pub struct HarborDockerBackend {
    pub docker_binary: PathBuf,
}

impl EvalBackend for HarborDockerBackend {
    fn name(&self) -> &str {
        "harbor-docker"
    }

    fn prepare(&self, task: &TaskPackage) -> Result<()> {
        task.validate()?;
        match task.environment {
            EnvironmentSpec::Oci { .. } | EnvironmentSpec::Dockerfile { .. } => Ok(()),
            EnvironmentSpec::Host => bail!("Harbor backend requires an OCI image or Dockerfile"),
        }
    }

    fn execute(
        &self,
        task: &TaskPackage,
        _attempt: &EvalAttempt,
        _root: &Path,
    ) -> Result<AttemptRecord> {
        self.prepare(task)?;
        bail!(
            "Harbor/Docker execution is not available in this build; use the host backend or install the Docker runtime"
        )
    }
}

/// Strict OCI + microVM backend. Runtime code is deliberately Linux-only;
/// planning and static validation remain available on every build target.
#[derive(Clone, Debug, Default)]
pub struct NativeOciBackend;

impl EvalBackend for NativeOciBackend {
    fn name(&self) -> &str {
        "native-oci-microvm"
    }

    fn prepare(&self, task: &TaskPackage) -> Result<()> {
        task.validate()?;
        native_runtime_check()?;
        Ok(())
    }

    fn execute(
        &self,
        task: &TaskPackage,
        _attempt: &EvalAttempt,
        _root: &Path,
    ) -> Result<AttemptRecord> {
        self.prepare(task)?;
        bail!("native OCI + microVM execution is not yet enabled for this profile")
    }
}

#[cfg(target_os = "linux")]
fn native_runtime_check() -> Result<()> {
    Ok(())
}

#[cfg(not(target_os = "linux"))]
fn native_runtime_check() -> Result<()> {
    bail!("native OCI + microVM backend is Linux-only")
}
