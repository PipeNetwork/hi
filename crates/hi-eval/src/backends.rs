//! Execution backend contracts used by profile-driven evaluation.
//!
//! The host implementation intentionally delegates to the existing hi-eval
//! task runner. Harbor/Docker and the strict Linux runtime share the same
//! package boundary but are kept separate so host execution cannot silently
//! claim container isolation.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Duration;

use anyhow::{Context, Result, bail};

use crate::platform::{
    AttemptRecord, EnvironmentSpec, EvalAttempt, EvalBackend, NetworkPolicy, ResourceSpec,
    TaskPackage, TimedOutput, command_output_with_timeout,
};

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

/// A bind mount passed to a Docker container.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DockerMount {
    pub source: PathBuf,
    pub destination: String,
    pub read_only: bool,
}

impl DockerMount {
    pub fn new(source: impl Into<PathBuf>, destination: impl Into<String>) -> Self {
        Self {
            source: source.into(),
            destination: destination.into(),
            read_only: false,
        }
    }

    pub fn read_only(mut self) -> Self {
        self.read_only = true;
        self
    }
}

/// A single container invocation. Commands are passed as arguments to
/// `/bin/sh -lc`; no host shell interpolation is involved.
#[derive(Clone, Debug)]
pub struct DockerRunSpec {
    pub image: String,
    pub name: String,
    pub workdir: String,
    pub command: String,
    pub mounts: Vec<DockerMount>,
    pub environment: BTreeMap<String, String>,
    pub network: NetworkPolicy,
    pub resources: ResourceSpec,
    /// Docker Desktop commonly lacks per-container overlay quotas. Storage
    /// enforcement is therefore explicit and opt-in rather than silently
    /// pretending that the requested quota was applied.
    pub enforce_storage: bool,
    pub timeout: Duration,
}

/// Captured result from a candidate or verifier container.
#[derive(Debug)]
pub struct DockerExecution {
    pub output: TimedOutput,
    pub image: String,
}

/// Manifest-driven Docker/Harbor backend. Docker Desktop on macOS is a valid
/// runtime for this backend because the containers themselves run in Docker's
/// Linux VM. The strict OCI + microVM backend remains separate.
#[derive(Clone, Debug)]
pub struct HarborDockerBackend {
    pub docker_binary: PathBuf,
}

impl Default for HarborDockerBackend {
    fn default() -> Self {
        Self {
            docker_binary: PathBuf::from("docker"),
        }
    }
}

impl HarborDockerBackend {
    pub fn from_environment() -> Self {
        Self {
            docker_binary: std::env::var_os("HI_DOCKER_BIN")
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from("docker")),
        }
    }

    /// Verify that Docker is reachable without pulling an image. Prepared
    /// runs must not unexpectedly contact a registry.
    pub fn check_runtime(&self) -> Result<()> {
        let mut command = Command::new(&self.docker_binary);
        command.args(["version", "--format", "{{.Server.Os}}"]);
        let result = command_output_with_timeout(&mut command, Duration::from_secs(30))
            .with_context(|| format!("checking Docker runtime {:?}", self.docker_binary))?;
        if !result.success() {
            bail!(
                "Docker runtime check failed: {}",
                lossy_output(&result.stderr, &result.stdout)
            );
        }
        Ok(())
    }

    /// Resolve an OCI image or build a Dockerfile context into a content-based
    /// local tag. The source directory is already immutable in the import
    /// store, so the tag includes its task digest.
    pub fn ensure_image(&self, task: &TaskPackage, task_root: &Path) -> Result<String> {
        let image = match &task.environment {
            EnvironmentSpec::Oci { image } => image.clone(),
            EnvironmentSpec::Dockerfile { context } => {
                let context = task_root.join(context);
                let context = context.canonicalize().with_context(|| {
                    format!("resolving Dockerfile context {}", context.display())
                })?;
                if !context.starts_with(task_root) {
                    bail!("Dockerfile context escapes the imported task package");
                }
                let (build_context, dockerfile) = if context.is_file() {
                    let parent = context
                        .parent()
                        .context("Dockerfile context has no parent directory")?;
                    (parent.to_path_buf(), Some(context))
                } else if context.is_dir() {
                    (context, None)
                } else {
                    bail!("Dockerfile context is not a file or directory");
                };
                let tag = format!(
                    "hi-eval-{}:local",
                    task.source.digest.chars().take(24).collect::<String>()
                );
                if self.image_is_present(&tag)? {
                    return Ok(tag);
                }
                let mut command = Command::new(&self.docker_binary);
                command
                    .arg("build")
                    .arg("--pull=false")
                    .arg("--tag")
                    .arg(&tag);
                if let Some(dockerfile) = dockerfile {
                    command.arg("--file").arg(dockerfile);
                }
                command.arg(&build_context);
                let result =
                    command_output_with_timeout(&mut command, Duration::from_secs(15 * 60))
                        .with_context(|| format!("building Docker image {tag}"))?;
                if !result.success() {
                    bail!(
                        "Docker image build failed for {}: {}",
                        task.id,
                        lossy_output(&result.stderr, &result.stdout)
                    );
                }
                tag
            }
            EnvironmentSpec::Host => {
                bail!("Docker backend requires an OCI image or Dockerfile")
            }
        };
        self.ensure_image_present(&image)?;
        Ok(image)
    }

    pub fn ensure_image_present(&self, image: &str) -> Result<()> {
        if self.image_is_present(image)? {
            return Ok(());
        }
        bail!(
            "Docker image {image:?} is not available locally; pull or build it before running the prepared evaluation"
        );
    }

    fn image_is_present(&self, image: &str) -> Result<bool> {
        let mut command = Command::new(&self.docker_binary);
        command.args(["image", "inspect", image]);
        let result = command_output_with_timeout(&mut command, Duration::from_secs(30))
            .with_context(|| format!("checking Docker image {image}"))?;
        Ok(result.success())
    }

    pub fn run(&self, spec: DockerRunSpec) -> Result<DockerExecution> {
        validate_docker_run_spec(&spec)?;
        // A deterministic name lets this remove a container left behind if
        // the evaluator was interrupted between Docker and the host process.
        self.remove_container(&spec.name);
        let mut command = Command::new(&self.docker_binary);
        command
            .arg("run")
            .arg("--rm")
            .arg("--init")
            .arg("--name")
            .arg(&spec.name)
            .arg("--entrypoint")
            .arg("/bin/sh")
            .arg("--workdir")
            .arg(&spec.workdir);
        append_network_args(&mut command, &spec.network)?;
        append_resource_args(&mut command, &spec.resources, spec.enforce_storage);
        for mount in &spec.mounts {
            command.arg("--mount").arg(format_mount(mount)?);
        }
        for (key, value) in &spec.environment {
            command.arg("--env").arg(format!("{key}={value}"));
        }
        command.arg(&spec.image).arg("-lc").arg(&spec.command);
        let output = command_output_with_timeout(&mut command, spec.timeout)
            .with_context(|| format!("running Docker container {}", spec.name))?;
        if output.timed_out {
            self.remove_container(&spec.name);
        }
        Ok(DockerExecution {
            output,
            image: spec.image,
        })
    }

    fn remove_container(&self, name: &str) {
        let mut command = Command::new(&self.docker_binary);
        let _ = command
            .args(["rm", "--force", "--volumes", name])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
    }
}

fn validate_docker_run_spec(spec: &DockerRunSpec) -> Result<()> {
    if spec.image.trim().is_empty() {
        bail!("Docker image must be non-empty");
    }
    if spec.name.trim().is_empty() {
        bail!("Docker container name must be non-empty");
    }
    if spec.command.trim().is_empty() {
        bail!("Docker command must be non-empty");
    }
    spec.network.validate("Docker network")?;
    if let NetworkPolicy::Scoped { .. } = spec.network {
        bail!(
            "Docker backend cannot enforce scoped network allowlists yet; use disabled or public network policy"
        );
    }
    if spec.resources.cpus == 0 || spec.resources.memory_mb == 0 || spec.resources.storage_mb == 0 {
        bail!("Docker resources must be greater than zero");
    }
    Ok(())
}

fn append_network_args(command: &mut Command, network: &NetworkPolicy) -> Result<()> {
    match network {
        NetworkPolicy::Disabled => {
            command.args(["--network", "none"]);
        }
        NetworkPolicy::Public => {}
        NetworkPolicy::Scoped { .. } => {
            bail!("scoped Docker networking is not supported")
        }
    }
    Ok(())
}

fn append_resource_args(command: &mut Command, resources: &ResourceSpec, enforce_storage: bool) {
    command
        .arg("--cpus")
        .arg(resources.cpus.to_string())
        .arg("--memory")
        .arg(format!("{}m", resources.memory_mb));
    if enforce_storage {
        command
            .arg("--storage-opt")
            .arg(format!("size={}m", resources.storage_mb));
    }
    if resources.gpus > 0 {
        command
            .arg("--gpus")
            .arg(format!("count={}", resources.gpus));
    }
}

fn format_mount(mount: &DockerMount) -> Result<String> {
    if mount.destination.is_empty() || !mount.destination.starts_with('/') {
        bail!("Docker mount destination must be an absolute path");
    }
    let source = mount
        .source
        .canonicalize()
        .with_context(|| format!("resolving Docker mount source {}", mount.source.display()))?;
    if !source.exists() {
        bail!("Docker mount source does not exist: {}", source.display());
    }
    let mut value = format!(
        "type=bind,source={},destination={}",
        source.display(),
        mount.destination
    );
    if mount.read_only {
        value.push_str(",readonly");
    }
    Ok(value)
}

fn lossy_output(primary: &[u8], secondary: &[u8]) -> String {
    let bytes = if primary.is_empty() {
        secondary
    } else {
        primary
    };
    String::from_utf8_lossy(bytes).trim().to_string()
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
            "Harbor/Docker execution requires an attempt command and workspace context; use HarborDockerBackend::run from the evaluator"
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn docker_mount_formats_read_only_bind() {
        let root = tempfile::tempdir().unwrap();
        let mount = DockerMount::new(root.path(), "/workspace").read_only();
        let value = format_mount(&mount).unwrap();
        assert!(value.contains("type=bind"));
        assert!(value.contains("destination=/workspace"));
        assert!(value.ends_with(",readonly"));
    }

    #[test]
    fn docker_run_rejects_scoped_networking_before_launch() {
        let root = tempfile::tempdir().unwrap();
        let spec = DockerRunSpec {
            image: "example@sha256:abc".into(),
            name: "hi-eval-test".into(),
            workdir: "/workspace".into(),
            command: "true".into(),
            mounts: vec![DockerMount::new(root.path(), "/workspace")],
            environment: BTreeMap::new(),
            network: NetworkPolicy::Scoped {
                hosts: vec!["api.example".into()],
            },
            resources: ResourceSpec::default(),
            enforce_storage: false,
            timeout: Duration::from_secs(1),
        };
        let error = validate_docker_run_spec(&spec).unwrap_err().to_string();
        assert!(error.contains("scoped network"));
    }

    #[test]
    fn docker_storage_quota_is_explicit() {
        let mut command = Command::new("docker");
        append_resource_args(&mut command, &ResourceSpec::default(), false);
        let args = command
            .get_args()
            .map(|arg| arg.to_string_lossy().into_owned())
            .collect::<Vec<_>>();
        assert!(!args.iter().any(|arg| arg == "--storage-opt"));

        let mut command = Command::new("docker");
        append_resource_args(&mut command, &ResourceSpec::default(), true);
        let args = command
            .get_args()
            .map(|arg| arg.to_string_lossy().into_owned())
            .collect::<Vec<_>>();
        assert!(args.iter().any(|arg| arg == "--storage-opt"));
    }

    #[cfg(unix)]
    #[test]
    fn docker_run_builds_an_isolated_command_without_storage_quota_by_default() {
        use std::os::unix::fs::PermissionsExt;

        let root = tempfile::tempdir().unwrap();
        let log = root.path().join("docker.args");
        let script = root.path().join("fake-docker");
        std::fs::write(
            &script,
            format!("#!/bin/sh\nprintf '%s\\n' \"$@\" >> '{}'\n", log.display()),
        )
        .unwrap();
        let mut permissions = std::fs::metadata(&script).unwrap().permissions();
        permissions.set_mode(0o755);
        std::fs::set_permissions(&script, permissions).unwrap();

        let backend = HarborDockerBackend {
            docker_binary: script,
        };
        let result = backend
            .run(DockerRunSpec {
                image: "example:local".into(),
                name: "hi-eval-test".into(),
                workdir: "/workspace".into(),
                command: "echo hi".into(),
                mounts: vec![DockerMount::new(root.path(), "/workspace")],
                environment: BTreeMap::new(),
                network: NetworkPolicy::Disabled,
                resources: ResourceSpec::default(),
                enforce_storage: false,
                timeout: Duration::from_secs(1),
            })
            .unwrap();
        assert!(result.output.success());
        let args = std::fs::read_to_string(log).unwrap();
        assert!(args.contains("--network\nnone"));
        assert!(args.contains("--mount"));
        assert!(!args.contains("--storage-opt"));
    }
}
