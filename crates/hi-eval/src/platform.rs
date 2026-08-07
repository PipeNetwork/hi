//! Benchmark-neutral evaluation contracts and durable preparation state.
//!
//! This module is deliberately independent of any benchmark format. Adapters
//! turn upstream sources into [`DatasetPlan`] values; the evaluator consumes
//! the content-addressed [`ImportedDataset`] produced by [`ImportStore`].

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io;
use std::path::{Component, Path, PathBuf};
use std::process::{Command, ExitStatus, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::config::Task;

pub const PLATFORM_SCHEMA_VERSION: u32 = 1;
static STAGING_COUNTER: AtomicU64 = AtomicU64::new(0);

const MAX_CAPTURED_PROCESS_OUTPUT_BYTES: usize = 4 * 1024 * 1024;

#[derive(Debug)]
pub struct TimedOutput {
    pub status: ExitStatus,
    pub stdout: Vec<u8>,
    pub stderr: Vec<u8>,
    pub timed_out: bool,
}

impl TimedOutput {
    pub fn success(&self) -> bool {
        !self.timed_out && self.status.success()
    }
}

/// Run evaluator-owned commands with bounded output and descendant cleanup.
/// This is intentionally part of the normalized harness contract so external
/// arms and final-message verifiers cannot hang the profile scheduler.
pub fn command_output_with_timeout(
    command: &mut Command,
    timeout: Duration,
) -> io::Result<TimedOutput> {
    command
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        command.process_group(0);
    }
    let started = Instant::now();
    let mut child = command.spawn()?;
    let stdout = child.stdout.take().expect("stdout was piped");
    let stderr = child.stderr.take().expect("stderr was piped");
    let stdout_reader = std::thread::spawn(move || read_process_output(stdout));
    let stderr_reader = std::thread::spawn(move || read_process_output(stderr));
    let mut timed_out = false;
    let status = loop {
        if let Some(status) = child.try_wait()? {
            break status;
        }
        if started.elapsed() >= timeout {
            timed_out = true;
            #[cfg(unix)]
            unsafe {
                libc::kill(-(child.id() as i32), libc::SIGKILL);
            }
            let _ = child.kill();
            break child.wait()?;
        }
        std::thread::sleep(Duration::from_millis(20));
    };
    #[cfg(unix)]
    unsafe {
        libc::kill(-(child.id() as i32), libc::SIGKILL);
    }
    let stdout = stdout_reader
        .join()
        .unwrap_or_else(|_| b"stdout reader panicked".to_vec());
    let stderr = stderr_reader
        .join()
        .unwrap_or_else(|_| b"stderr reader panicked".to_vec());
    Ok(TimedOutput {
        status,
        stdout,
        stderr,
        timed_out,
    })
}

fn read_process_output(mut reader: impl io::Read) -> Vec<u8> {
    let mut output = Vec::new();
    let mut chunk = [0u8; 8192];
    while let Ok(size) = reader.read(&mut chunk) {
        if size == 0 {
            break;
        }
        let remaining = MAX_CAPTURED_PROCESS_OUTPUT_BYTES.saturating_sub(output.len());
        if remaining > 0 {
            output.extend_from_slice(&chunk[..size.min(remaining)]);
        }
    }
    output
}

/// How strongly a result may be described to users.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ClaimLevel {
    #[default]
    Smoke,
    PublicReproduction,
    Official,
    EvidenceOnly,
}

/// Input retained for an evaluation attempt.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum EvalInput {
    Prompt {
        prompt: String,
    },
    Transcript {
        messages: Vec<TranscriptMessage>,
        #[serde(default)]
        final_prompt: Option<String>,
    },
}

/// A role-preserving transcript message. Content is intentionally JSON so
/// adapters can retain text, images, tool calls, and provider-specific blocks.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct TranscriptMessage {
    pub role: String,
    pub content: Value,
}

/// What the verifier consumes from the candidate.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum EvalOutput {
    #[default]
    Workspace,
    FinalMessage,
}

/// Candidate environment declaration.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
#[derive(Default)]
pub enum EnvironmentSpec {
    Oci {
        image: String,
    },
    Dockerfile {
        context: PathBuf,
    },
    #[default]
    Host,
}

/// Candidate/verifier network policy.
#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum NetworkPolicy {
    #[default]
    Disabled,
    Public,
    Scoped {
        hosts: Vec<String>,
    },
}

impl NetworkPolicy {
    pub fn validate(&self, label: &str) -> Result<()> {
        if let Self::Scoped { hosts } = self {
            if hosts.is_empty() {
                bail!("{label} scoped policy must name at least one host");
            }
            if hosts
                .iter()
                .any(|host| host.trim().is_empty() || host.contains('\n'))
            {
                bail!("{label} scoped policy contains an invalid host");
            }
        }
        Ok(())
    }
}

/// Resource requirements used by admission and native backends.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ResourceSpec {
    #[serde(default = "default_cpus")]
    pub cpus: u32,
    #[serde(default = "default_memory_mb")]
    pub memory_mb: u64,
    #[serde(default = "default_storage_mb")]
    pub storage_mb: u64,
    #[serde(default)]
    pub gpus: u32,
}

impl Default for ResourceSpec {
    fn default() -> Self {
        Self {
            cpus: default_cpus(),
            memory_mb: default_memory_mb(),
            storage_mb: default_storage_mb(),
            gpus: 0,
        }
    }
}

const fn default_cpus() -> u32 {
    1
}

const fn default_memory_mb() -> u64 {
    1024
}

const fn default_storage_mb() -> u64 {
    4096
}

/// Candidate output that must be copied into a verifier environment.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ArtifactSpec {
    pub source: PathBuf,
    #[serde(default)]
    pub exclude: Vec<PathBuf>,
}

/// Verifier command and environment placement.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct VerifierSpec {
    pub command: String,
    #[serde(default)]
    pub environment: EnvironmentSpec,
    #[serde(default)]
    pub network: NetworkPolicy,
    #[serde(default)]
    pub artifacts: Vec<ArtifactSpec>,
}

/// Binary classification policy for named raw rewards.
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ScoringPolicy {
    #[default]
    VerifierExit,
    AllRewardsPositive,
    AllRewardsOne,
    NamedReward {
        name: String,
        threshold: f64,
    },
}

impl ScoringPolicy {
    pub fn validate(&self) -> Result<()> {
        if let Self::NamedReward { name, threshold } = self {
            if name.trim().is_empty() {
                bail!("named reward scoring policy requires a non-empty reward name");
            }
            if !threshold.is_finite() {
                bail!("named reward scoring threshold must be finite");
            }
        }
        Ok(())
    }

    pub fn classify(&self, verifier_passed: bool, rewards: &BTreeMap<String, f64>) -> bool {
        match self {
            Self::VerifierExit => verifier_passed,
            Self::AllRewardsPositive => {
                !rewards.is_empty() && rewards.values().all(|reward| *reward > 0.0)
            }
            Self::AllRewardsOne => {
                !rewards.is_empty()
                    && rewards
                        .values()
                        .all(|reward| (*reward - 1.0).abs() <= f64::EPSILON)
            }
            Self::NamedReward { name, threshold } => rewards
                .get(name)
                .is_some_and(|reward| reward.is_finite() && reward >= threshold),
        }
    }
}

/// A complete evaluator-ready task description.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct TaskPackage {
    pub schema_version: u32,
    pub id: String,
    #[serde(default)]
    pub name: Option<String>,
    pub source: SourceIdentity,
    pub input: EvalInput,
    #[serde(default)]
    pub output: EvalOutput,
    #[serde(default)]
    pub environment: EnvironmentSpec,
    #[serde(default)]
    pub verifier: Option<VerifierSpec>,
    #[serde(default)]
    pub resources: ResourceSpec,
    #[serde(default)]
    pub network: NetworkPolicy,
    #[serde(default)]
    pub artifacts: Vec<ArtifactSpec>,
    #[serde(default)]
    pub scoring: ScoringPolicy,
    #[serde(default)]
    pub claim_level: ClaimLevel,
}

/// One backend execution coordinate.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct EvalAttempt {
    pub task: String,
    pub arm: String,
    pub trial: u32,
    pub identity_digest: String,
}

/// A paired arm result for one task/trial coordinate. Arms always point to
/// independent attempt evidence; the comparison itself is the only shared
/// record.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct DifferentialArm {
    pub name: String,
    pub attempt: EvalAttempt,
    pub score: Option<EvalScore>,
    #[serde(default)]
    pub evidence: Vec<PathBuf>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct DifferentialComparison {
    pub schema_version: u32,
    pub task: String,
    pub trial: u32,
    pub identity_digest: String,
    pub arms: Vec<DifferentialArm>,
}

/// Common lifecycle boundary implemented by host, Harbor, and native backends.
/// The first implementation is synchronous because the existing host runner
/// already owns its blocking child-process boundary; async backends can wrap
/// this trait without changing the package contract.
pub trait EvalBackend: Send + Sync {
    fn name(&self) -> &str;
    fn prepare(&self, task: &TaskPackage) -> Result<()>;
    fn start_attempt(&self, task: &TaskPackage, attempt: &EvalAttempt, root: &Path) -> Result<()> {
        let _ = (attempt, root);
        self.prepare(task)
    }
    fn run_hi_agent(
        &self,
        _task: &TaskPackage,
        _attempt: &EvalAttempt,
        _root: &Path,
    ) -> Result<()> {
        bail!(
            "backend {} does not expose a native agent launch operation",
            self.name()
        )
    }
    fn capture_artifacts(
        &self,
        _task: &TaskPackage,
        _attempt: &EvalAttempt,
        _root: &Path,
    ) -> Result<Vec<PathBuf>> {
        bail!("backend {} does not expose artifact capture", self.name())
    }
    fn run_verifier(
        &self,
        _task: &TaskPackage,
        _attempt: &EvalAttempt,
        _root: &Path,
    ) -> Result<Option<EvalScore>> {
        bail!("backend {} does not expose verifier execution", self.name())
    }
    fn execute(
        &self,
        task: &TaskPackage,
        attempt: &EvalAttempt,
        root: &Path,
    ) -> Result<AttemptRecord>;
    fn cleanup_attempt(
        &self,
        _task: &TaskPackage,
        _attempt: &EvalAttempt,
        _root: &Path,
    ) -> Result<()> {
        Ok(())
    }
}

impl TaskPackage {
    pub fn validate(&self) -> Result<()> {
        if self.schema_version != PLATFORM_SCHEMA_VERSION {
            bail!(
                "unsupported task package schema {}; expected {}",
                self.schema_version,
                PLATFORM_SCHEMA_VERSION
            );
        }
        validate_component("task id", &self.id)?;
        if self.source.revision.trim().is_empty() || self.source.digest.trim().is_empty() {
            bail!("task source revision and digest must be non-empty");
        }
        if Path::new(&self.source.revision).is_absolute() {
            bail!("task source revision must not contain a local absolute path");
        }
        match &self.input {
            EvalInput::Prompt { prompt } if prompt.trim().is_empty() => {
                bail!("task evaluation prompt must not be empty")
            }
            EvalInput::Transcript {
                messages,
                final_prompt,
            } => {
                if messages.is_empty()
                    && final_prompt.as_deref().is_none_or(|p| p.trim().is_empty())
                {
                    bail!("transcript evaluation input must contain messages or final_prompt");
                }
                if final_prompt.as_deref().is_some_and(|p| p.trim().is_empty()) {
                    bail!("transcript final_prompt must not be empty");
                }
                for message in messages {
                    if !matches!(
                        message.role.trim().to_ascii_lowercase().as_str(),
                        "system" | "user" | "assistant" | "tool"
                    ) {
                        bail!("unsupported transcript role {:?}", message.role);
                    }
                }
            }
            EvalInput::Prompt { .. } => {}
        }
        if self.resources.cpus == 0
            || self.resources.memory_mb == 0
            || self.resources.storage_mb == 0
        {
            bail!("task resources must be greater than zero");
        }
        if let EnvironmentSpec::Oci { image } = &self.environment
            && image.trim().is_empty()
        {
            bail!("OCI environment image must be non-empty");
        }
        if let EnvironmentSpec::Dockerfile { context } = &self.environment {
            validate_relative_path(context, "Dockerfile context")?;
        }
        self.network.validate("candidate network")?;
        if let Some(verifier) = &self.verifier {
            if verifier.command.trim().is_empty() {
                bail!("verifier command must be non-empty");
            }
            verifier.network.validate("verifier network")?;
        }
        self.scoring.validate()?;
        for artifact in self.artifacts.iter().chain(
            self.verifier
                .as_ref()
                .into_iter()
                .flat_map(|verifier| verifier.artifacts.iter()),
        ) {
            validate_relative_path(&artifact.source, "artifact source")?;
            for excluded in &artifact.exclude {
                validate_relative_path(excluded, "artifact exclusion")?;
            }
        }
        Ok(())
    }

    /// Load either the unchanged schema-v2 task format or a normalized
    /// `package.toml`/`package.json` task. The legacy file is never rewritten.
    pub fn load_from_directory(root: impl AsRef<Path>) -> Result<Self> {
        let root = root.as_ref();
        if root.join("task.toml").is_file() {
            let text = fs::read_to_string(root.join("task.toml"))
                .with_context(|| format!("reading legacy task {}", root.display()))?;
            let task: Task = toml::from_str(&text)
                .with_context(|| format!("parsing legacy task {}", root.display()))?;
            task.validate()?;
            return Self::from_legacy(root, &task);
        }
        let package_path = if root.join("package.toml").is_file() {
            root.join("package.toml")
        } else {
            root.join("package.json")
        };
        let bytes = fs::read(&package_path)
            .with_context(|| format!("reading task package {}", package_path.display()))?;
        let package = if package_path.extension().and_then(|ext| ext.to_str()) == Some("toml") {
            toml::from_str::<Self>(&String::from_utf8(bytes).context("task package is not UTF-8")?)?
        } else {
            serde_json::from_slice::<Self>(&bytes)?
        };
        package.validate()?;
        Ok(package)
    }

    /// Converts an existing schema-v2 local task without changing its files.
    pub fn from_legacy(root: impl Into<PathBuf>, task: &Task) -> Result<Self> {
        let root = root.into();
        let prompt = task.prompt.clone();
        let package = Self {
            schema_version: PLATFORM_SCHEMA_VERSION,
            id: task.name.clone().unwrap_or_else(|| {
                root.file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .into()
            }),
            name: task.name.clone(),
            source: SourceIdentity {
                kind: "hi-legacy".into(),
                revision: "local-task".into(),
                digest: digest_path(&root)?,
            },
            input: EvalInput::Prompt { prompt },
            output: EvalOutput::Workspace,
            environment: EnvironmentSpec::Host,
            verifier: Some(VerifierSpec {
                command: task.final_oracle.command.clone(),
                environment: EnvironmentSpec::Host,
                network: NetworkPolicy::Disabled,
                artifacts: Vec::new(),
            }),
            resources: ResourceSpec::default(),
            network: NetworkPolicy::Disabled,
            artifacts: Vec::new(),
            scoring: ScoringPolicy::VerifierExit,
            claim_level: ClaimLevel::Smoke,
        };
        package.validate()?;
        Ok(package)
    }
}

/// Safe provenance for an imported source.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct SourceIdentity {
    pub kind: String,
    pub revision: String,
    pub digest: String,
}

/// One source package selected by an adapter.
#[derive(Clone, Debug)]
pub struct CasePlan {
    pub id: String,
    pub source_directory: PathBuf,
}

impl CasePlan {
    pub fn new(id: impl Into<String>, source_directory: impl Into<PathBuf>) -> Self {
        Self {
            id: id.into(),
            source_directory: source_directory.into(),
        }
    }
}

/// Adapter output before durable publication.
#[derive(Clone, Debug)]
pub struct DatasetPlan {
    pub name: String,
    pub source: SourceIdentity,
    pub cases: Vec<CasePlan>,
    pub claim_level: ClaimLevel,
}

impl DatasetPlan {
    pub fn new(name: impl Into<String>, source: SourceIdentity) -> Self {
        Self {
            name: name.into(),
            source,
            cases: Vec::new(),
            claim_level: ClaimLevel::Smoke,
        }
    }

    pub fn case(mut self, case: CasePlan) -> Self {
        self.cases.push(case);
        self
    }

    pub fn with_cases(mut self, cases: Vec<CasePlan>) -> Self {
        self.cases = cases;
        self
    }

    pub fn with_claim_level(mut self, claim_level: ClaimLevel) -> Self {
        self.claim_level = claim_level;
        self
    }

    pub fn validate(&self) -> Result<()> {
        validate_component("dataset name", &self.name)?;
        if self.source.kind.trim().is_empty()
            || self.source.revision.trim().is_empty()
            || self.source.digest.trim().is_empty()
        {
            bail!("dataset source kind, revision, and digest must be non-empty");
        }
        if Path::new(&self.source.revision).is_absolute() {
            bail!("dataset source revision must not contain a local absolute path");
        }
        if self.cases.is_empty() {
            bail!("dataset plan must contain at least one case");
        }
        let mut ids = BTreeSet::new();
        for case in &self.cases {
            validate_component("case id", &case.id)?;
            if !ids.insert(&case.id) {
                bail!("duplicate case id {:?}", case.id);
            }
            if !case.source_directory.is_dir() {
                bail!(
                    "case source is not a directory: {}",
                    case.source_directory.display()
                );
            }
        }
        Ok(())
    }
}

/// Durable content-addressed storage for imported datasets.
#[derive(Clone, Debug)]
pub struct ImportStore {
    root: PathBuf,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ImportedDataset {
    pub name: String,
    pub digest: String,
    pub source: SourceIdentity,
    #[serde(default)]
    pub claim_level: ClaimLevel,
    /// Operationally resolved after loading. It is intentionally omitted
    /// from the durable manifest so imported provenance never records a
    /// machine-local absolute path.
    #[serde(skip)]
    pub root: PathBuf,
    pub tasks: Vec<ImportedTask>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ImportedTask {
    pub id: String,
    pub digest: String,
    pub path: PathBuf,
    #[serde(default)]
    pub package: Option<TaskPackage>,
}

impl ImportStore {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn import(&self, plan: &DatasetPlan) -> Result<ImportedDataset> {
        plan.validate()?;
        fs::create_dir_all(&self.root)
            .with_context(|| format!("creating import store {}", self.root.display()))?;
        let digest = plan_digest(plan)?;
        let planned_case_digests = plan
            .cases
            .iter()
            .map(|case| digest_path(&case.source_directory).map(|digest| (case.id.clone(), digest)))
            .collect::<Result<BTreeMap<_, _>>>()?;
        if plan_digest(plan)? != digest {
            bail!("source contents changed while preparing the import; retry preparation");
        }
        let destination = self.root.join(&plan.name).join(&digest);
        let manifest_path = destination.join("dataset.json");
        if manifest_path.is_file() {
            return self.load(&destination);
        }

        let dataset_parent = self.root.join(&plan.name);
        fs::create_dir_all(&dataset_parent)?;
        let staging = self.root.join(format!(
            ".import-{}-{}-{}",
            std::process::id(),
            STAGING_COUNTER.fetch_add(1, Ordering::Relaxed),
            digest
        ));
        if staging.exists() {
            fs::remove_dir_all(&staging)?;
        }
        fs::create_dir_all(staging.join("tasks"))?;

        let mut tasks = Vec::with_capacity(plan.cases.len());
        let result = (|| -> Result<()> {
            let mut sorted = plan.cases.clone();
            sorted.sort_by(|left, right| left.id.cmp(&right.id));
            for case in sorted {
                let source_digest = digest_path(&case.source_directory)?;
                if planned_case_digests.get(&case.id) != Some(&source_digest) {
                    bail!(
                        "source contents changed while importing case {:?}; retry preparation",
                        case.id
                    );
                }
                let package = if case.source_directory.join("task.toml").is_file()
                    || case.source_directory.join("package.toml").is_file()
                    || case.source_directory.join("package.json").is_file()
                {
                    let mut package = TaskPackage::load_from_directory(&case.source_directory)?;
                    package.claim_level = plan.claim_level;
                    Some(package)
                } else {
                    None
                };
                let destination = staging.join("tasks").join(&case.id);
                copy_tree(&case.source_directory, &destination)?;
                if digest_path(&case.source_directory)? != source_digest
                    || digest_path(&destination)? != source_digest
                {
                    bail!(
                        "source contents changed while importing case {:?}; retry preparation",
                        case.id
                    );
                }
                let id = case.id;
                tasks.push(ImportedTask {
                    id: id.clone(),
                    digest: digest_path(&destination)?,
                    path: PathBuf::from("tasks").join(id),
                    package,
                });
            }
            let imported = ImportedDataset {
                name: plan.name.clone(),
                digest: digest.clone(),
                source: plan.source.clone(),
                claim_level: plan.claim_level,
                root: destination.clone(),
                tasks,
            };
            write_json_atomic(&staging.join("dataset.json"), &imported)?;
            Ok(())
        })();
        if let Err(error) = result {
            let _ = fs::remove_dir_all(&staging);
            return Err(error);
        }

        match fs::rename(&staging, &destination) {
            Ok(()) => self.load(&destination),
            Err(error) if destination.exists() => {
                let _ = fs::remove_dir_all(&staging);
                self.load(&destination).with_context(|| {
                    format!("loading concurrently published dataset after {error}")
                })
            }
            Err(error) => {
                let _ = fs::remove_dir_all(&staging);
                Err(error.into())
            }
        }
    }

    pub fn load(&self, root: impl AsRef<Path>) -> Result<ImportedDataset> {
        let root = root.as_ref();
        let manifest = root.join("dataset.json");
        let bytes = fs::read(&manifest)
            .with_context(|| format!("reading imported dataset {}", manifest.display()))?;
        let mut dataset: ImportedDataset = serde_json::from_slice(&bytes)
            .with_context(|| format!("parsing imported dataset {}", manifest.display()))?;
        validate_component("imported dataset name", &dataset.name)?;
        if dataset.name
            != root
                .parent()
                .and_then(Path::file_name)
                .map(|name| name.to_string_lossy().into_owned())
                .unwrap_or_default()
        {
            bail!("imported dataset manifest name does not match its location");
        }
        if dataset.digest
            != root
                .file_name()
                .map(|name| name.to_string_lossy().into_owned())
                .unwrap_or_default()
        {
            bail!("imported dataset manifest digest does not match its location");
        }
        let tasks_root = root.join("tasks");
        if !tasks_root.is_dir() {
            bail!("imported dataset is missing its task directory");
        }
        if dataset.tasks.is_empty() {
            bail!("imported dataset contains no tasks");
        }
        let mut ids = BTreeSet::new();
        for task in &dataset.tasks {
            validate_component("imported task id", &task.id)?;
            validate_relative_path(&task.path, "imported task path")?;
            if !ids.insert(&task.id) {
                bail!("imported dataset contains duplicate task id {:?}", task.id);
            }
            let expected_path = PathBuf::from("tasks").join(&task.id);
            if task.path != expected_path {
                bail!("imported task {:?} has an unexpected path", task.id);
            }
            let task_path = root.join(&task.path);
            let metadata = fs::symlink_metadata(&task_path)
                .with_context(|| format!("reading imported task {}", task_path.display()))?;
            if metadata.file_type().is_symlink() || !metadata.is_dir() {
                bail!(
                    "imported task is not a real directory: {}",
                    task_path.display()
                );
            }
            if digest_path(&task_path)? != task.digest {
                bail!(
                    "imported task {:?} failed its content digest check",
                    task.id
                );
            }
            let has_package = ["task.toml", "package.toml", "package.json"]
                .iter()
                .any(|name| task_path.join(name).is_file());
            match (&task.package, has_package) {
                (Some(package), true) => {
                    if package.id != task.id {
                        bail!("imported task package id does not match {:?}", task.id);
                    }
                    package.validate()?;
                    let mut actual = TaskPackage::load_from_directory(&task_path)?;
                    // The importer applies the dataset claim level after
                    // loading the source package; normalize that one field
                    // before comparing the manifest projection.
                    actual.claim_level = package.claim_level;
                    if serde_json::to_vec(&actual)? != serde_json::to_vec(package)? {
                        bail!(
                            "imported task {:?} package projection was modified",
                            task.id
                        );
                    }
                }
                (None, false) => {}
                (Some(_), false) => {
                    bail!("imported task {:?} is missing its package file", task.id)
                }
                (None, true) => {
                    bail!("imported task {:?} omitted its package projection", task.id)
                }
            }
        }
        let mut on_disk = fs::read_dir(&tasks_root)?.collect::<io::Result<Vec<_>>>()?;
        on_disk.sort_by_key(|entry| entry.file_name());
        for entry in on_disk {
            let metadata = fs::symlink_metadata(entry.path())?;
            let id = entry.file_name().to_string_lossy().into_owned();
            if metadata.file_type().is_symlink() || !metadata.is_dir() || !ids.contains(&id) {
                bail!("imported dataset contains an unexpected task entry {}", id);
            }
        }
        let task_digests = dataset
            .tasks
            .iter()
            .map(|task| (task.id.clone(), task.digest.clone()))
            .collect::<Vec<_>>();
        if dataset_digest(
            &dataset.name,
            &dataset.source,
            dataset.claim_level,
            task_digests,
        )? != dataset.digest
        {
            bail!("imported dataset failed its content-address digest check");
        }
        dataset.root = root.to_path_buf();
        Ok(dataset)
    }
}

/// A profile manifest. Adapter-specific options remain opaque JSON here.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EvalManifest {
    pub schema_version: u32,
    pub name: String,
    #[serde(default)]
    pub datasets: BTreeMap<String, DatasetSource>,
    pub profiles: BTreeMap<String, EvalProfile>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DatasetSource {
    pub adapter: String,
    pub source: PathBuf,
    pub revision: String,
    #[serde(default)]
    pub options: Value,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EvalProfile {
    #[serde(default)]
    pub datasets: Vec<String>,
    #[serde(default)]
    pub models: Vec<String>,
    #[serde(default = "default_trials")]
    pub trials: u32,
    #[serde(default = "default_backend")]
    pub backend: String,
    #[serde(default)]
    pub claim_level: ClaimLevel,
    #[serde(default)]
    pub arms: Vec<DifferentialArmConfig>,
    #[serde(default)]
    pub treatments: Vec<String>,
    /// Optional case selectors. Empty means every imported case.
    #[serde(default)]
    pub selectors: Vec<String>,
    /// Model sampling treatment (temperature, top-p, seeds, or provider
    /// specific knobs) is opaque to the generic scheduler but is part of the
    /// run identity.
    #[serde(default)]
    pub sampling: Value,
    #[serde(default)]
    pub provider_policy: Value,
    #[serde(default)]
    pub resources: ResourceSpec,
    #[serde(default)]
    pub network: NetworkPolicy,
    #[serde(default)]
    pub verifier: Option<VerifierSpec>,
    #[serde(default)]
    pub scoring: ScoringPolicy,
    #[serde(default)]
    pub mcp_servers: Vec<McpServerSpec>,
    #[serde(default)]
    pub evidence: EvidencePolicy,
    /// Operator-provided fingerprint for secret-bearing configuration. The
    /// value is a digest, never the secret itself.
    #[serde(default)]
    pub secret_configuration_digest: Option<String>,
}

const fn default_trials() -> u32 {
    1
}

fn default_backend() -> String {
    "host".into()
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DifferentialArmConfig {
    pub name: String,
    #[serde(default)]
    pub command: Option<PathBuf>,
    #[serde(default)]
    pub environment: BTreeMap<String, String>,
    #[serde(default)]
    pub model: Option<String>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct McpServerSpec {
    pub name: String,
    #[serde(default)]
    pub command: Option<String>,
    #[serde(default)]
    pub args: Vec<String>,
    #[serde(default)]
    pub configuration_digest: Option<String>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EvidencePolicy {
    #[serde(default = "default_true")]
    pub retain_failed: bool,
    #[serde(default)]
    pub diagnostic_roots: Vec<PathBuf>,
}

const fn default_true() -> bool {
    true
}

impl EvalManifest {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let text = fs::read_to_string(path)
            .with_context(|| format!("reading evaluation manifest {}", path.display()))?;
        let manifest: Self = toml::from_str(&text)
            .with_context(|| format!("parsing evaluation manifest {}", path.display()))?;
        manifest.validate()?;
        Ok(manifest)
    }

    pub fn validate(&self) -> Result<()> {
        if self.schema_version != PLATFORM_SCHEMA_VERSION {
            bail!(
                "unsupported evaluation manifest schema {}; expected {}",
                self.schema_version,
                PLATFORM_SCHEMA_VERSION
            );
        }
        validate_component("manifest name", &self.name)?;
        if self.profiles.is_empty() {
            bail!("evaluation manifest must define at least one profile");
        }
        for (name, profile) in &self.profiles {
            validate_component("profile name", name)?;
            if profile.trials == 0 {
                bail!("profile {name:?} must have at least one trial");
            }
            profile.scoring.validate()?;
            let mut arm_names = BTreeSet::new();
            for arm in &profile.arms {
                validate_component("differential arm name", &arm.name)?;
                if !arm_names.insert(&arm.name) {
                    bail!("profile {name:?} contains duplicate arm {:?}", arm.name);
                }
            }
            if profile.datasets.is_empty() {
                bail!("profile {name:?} must select at least one dataset");
            }
            profile
                .network
                .validate(&format!("profile {name:?} network"))?;
            if !matches!(
                profile.backend.as_str(),
                "host" | "legacy-host" | "harbor" | "docker" | "native-oci" | "microvm"
            ) {
                bail!(
                    "profile {name:?} selects unknown backend {:?}",
                    profile.backend
                );
            }
            for treatment in &profile.treatments {
                validate_component("treatment name", treatment)?;
            }
            for model in &profile.models {
                if model.trim().is_empty() {
                    bail!("profile {name:?} contains an empty model id");
                }
            }
            for selector in &profile.selectors {
                if selector.trim().is_empty() {
                    bail!("profile {name:?} contains an empty task selector");
                }
            }
            for dataset in &profile.datasets {
                if !self.datasets.contains_key(dataset) {
                    bail!("profile {name:?} references unknown dataset {dataset:?}");
                }
            }
            let mut selected_datasets = BTreeSet::new();
            for dataset in &profile.datasets {
                if !selected_datasets.insert(dataset) {
                    bail!(
                        "profile {name:?} selects dataset {:?} more than once",
                        dataset
                    );
                }
            }
            for server in &profile.mcp_servers {
                validate_component("MCP server name", &server.name)?;
                if server
                    .configuration_digest
                    .as_deref()
                    .is_some_and(|digest| digest.contains('/') || digest.contains('\\'))
                {
                    bail!("profile {name:?} contains an invalid MCP configuration digest");
                }
            }
        }
        for source in self.datasets.values() {
            if source.revision.trim().is_empty() || source.adapter.trim().is_empty() {
                bail!("dataset adapter and revision must be non-empty");
            }
            if Path::new(&source.revision).is_absolute() {
                bail!("dataset source revision must not contain a local absolute path");
            }
        }
        Ok(())
    }

    pub fn digest(&self) -> Result<String> {
        digest_bytes(&serde_json::to_vec(self)?)
    }

    pub fn profile(&self, name: &str) -> Result<&EvalProfile> {
        self.profiles
            .get(name)
            .with_context(|| format!("unknown evaluation profile {name:?}"))
    }
}

/// Stable identity used to reject stale preparations and resumes.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct RunIdentity {
    pub profile: String,
    pub manifest_digest: String,
    pub dataset_digests: BTreeMap<String, String>,
    pub models: Vec<String>,
    pub backend: String,
    pub scoring_policy_digest: String,
    pub configuration_digest: String,
    #[serde(default)]
    pub adapter_version: String,
    #[serde(default)]
    pub hi_binary_digest: String,
    #[serde(default)]
    pub provider_policy_digest: String,
    #[serde(default)]
    pub mcp_configuration_digest: String,
    #[serde(default)]
    pub secret_configuration_digest: String,
    #[serde(default)]
    pub runtime_identity: String,
    pub digest: String,
}

impl RunIdentity {
    pub fn new(
        profile: impl Into<String>,
        manifest_digest: impl Into<String>,
        dataset_digests: BTreeMap<String, String>,
        models: Vec<String>,
        backend: impl Into<String>,
        scoring_policy_digest: impl Into<String>,
        configuration_digest: impl Into<String>,
    ) -> Result<Self> {
        Self::new_with_details(
            profile,
            manifest_digest,
            dataset_digests,
            models,
            backend,
            scoring_policy_digest,
            configuration_digest,
            IdentityDetails::default(),
        )
    }

    #[allow(clippy::too_many_arguments)] // platform descriptor carries each manifest field 1:1
    pub fn new_with_details(
        profile: impl Into<String>,
        manifest_digest: impl Into<String>,
        dataset_digests: BTreeMap<String, String>,
        models: Vec<String>,
        backend: impl Into<String>,
        scoring_policy_digest: impl Into<String>,
        configuration_digest: impl Into<String>,
        details: IdentityDetails,
    ) -> Result<Self> {
        let mut identity = Self {
            profile: profile.into(),
            manifest_digest: manifest_digest.into(),
            dataset_digests,
            models,
            backend: backend.into(),
            scoring_policy_digest: scoring_policy_digest.into(),
            configuration_digest: configuration_digest.into(),
            adapter_version: details.adapter_version,
            hi_binary_digest: details.hi_binary_digest,
            provider_policy_digest: details.provider_policy_digest,
            mcp_configuration_digest: details.mcp_configuration_digest,
            secret_configuration_digest: details.secret_configuration_digest,
            runtime_identity: details.runtime_identity,
            digest: String::new(),
        };
        identity.digest = digest_bytes(&serde_json::to_vec(&identity)?)?;
        Ok(identity)
    }
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct IdentityDetails {
    pub adapter_version: String,
    pub hi_binary_digest: String,
    pub provider_policy_digest: String,
    pub mcp_configuration_digest: String,
    pub secret_configuration_digest: String,
    pub runtime_identity: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct PreparationReceipt {
    pub schema_version: u32,
    pub profile: String,
    pub manifest_digest: String,
    pub datasets: BTreeMap<String, String>,
    #[serde(default)]
    pub store_root: PathBuf,
    pub identity: RunIdentity,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RunStatus {
    #[default]
    Prepared,
    Running,
    Stopped,
    Failed,
    Completed,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct RunRecord {
    pub schema_version: u32,
    pub profile: String,
    pub identity: RunIdentity,
    pub status: RunStatus,
    pub started_at_unix: u64,
    #[serde(default)]
    pub updated_at_unix: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ProgressEvent {
    pub task: String,
    pub arm: String,
    pub trial: u32,
    pub status: AttemptStatus,
    pub identity_digest: String,
    #[serde(default)]
    pub message: Option<String>,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AttemptStatus {
    #[default]
    Pending,
    Running,
    Passed,
    Failed,
    InfrastructureFailed,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct AttemptRecord {
    pub profile: String,
    pub task: String,
    pub arm: String,
    pub trial: u32,
    pub status: AttemptStatus,
    pub identity_digest: String,
    #[serde(default)]
    pub claim_level: ClaimLevel,
    #[serde(default)]
    pub score: Option<EvalScore>,
    #[serde(default)]
    pub evidence: Option<EvalEvidence>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct EvalScore {
    pub passed: bool,
    #[serde(default)]
    pub rewards: BTreeMap<String, f64>,
    #[serde(default)]
    pub classification: String,
}

impl EvalScore {
    pub fn from_rewards(
        policy: &ScoringPolicy,
        verifier_passed: bool,
        rewards: BTreeMap<String, f64>,
    ) -> Self {
        let passed = policy.classify(verifier_passed, &rewards);
        Self {
            passed,
            rewards,
            classification: if passed { "pass" } else { "fail" }.to_string(),
        }
    }
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct EvalEvidence {
    #[serde(default)]
    pub report: Option<PathBuf>,
    #[serde(default)]
    pub verifier_log: Option<PathBuf>,
    #[serde(default)]
    pub artifacts: Vec<PathBuf>,
    #[serde(default)]
    pub claim_level: ClaimLevel,
    #[serde(default)]
    pub source_digest: Option<String>,
    #[serde(default)]
    pub task_digest: Option<String>,
    #[serde(default)]
    pub backend: Option<String>,
    #[serde(default)]
    pub runtime: Option<String>,
    #[serde(default)]
    pub preparation_seconds: Option<f64>,
    #[serde(default)]
    pub verifier_seconds: Option<f64>,
    #[serde(default)]
    pub transcript_messages: Option<usize>,
    #[serde(default)]
    pub prompt_characters: Option<usize>,
    #[serde(default)]
    pub scoring_policy_digest: Option<String>,
    #[serde(default)]
    pub input_mode: Option<String>,
    #[serde(default)]
    pub output_mode: Option<String>,
    #[serde(default)]
    pub error: Option<String>,
}

/// Filesystem-backed durable evaluation state.
#[derive(Clone, Debug)]
pub struct EvalStateStore {
    root: PathBuf,
}

impl EvalStateStore {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    pub fn profile_root(&self, profile: &str) -> Result<PathBuf> {
        validate_component("profile name", profile)?;
        Ok(self.root.join(profile))
    }

    pub fn write_preparation(&self, receipt: &PreparationReceipt) -> Result<PathBuf> {
        let root = self.profile_root(&receipt.profile)?;
        fs::create_dir_all(&root)?;
        let path = root.join("preparation.json");
        write_json_atomic(&path, receipt)?;
        Ok(path)
    }

    pub fn write_run(&self, run: &RunRecord) -> Result<PathBuf> {
        let root = self.profile_root(&run.profile)?;
        fs::create_dir_all(&root)?;
        let path = root.join("run.json");
        write_json_atomic(&path, run)?;
        Ok(path)
    }

    pub fn read_run(&self, profile: &str) -> Result<Option<RunRecord>> {
        let path = self.profile_root(profile)?.join("run.json");
        if !path.is_file() {
            return Ok(None);
        }
        Ok(Some(read_json(&path)?))
    }

    pub fn append_progress(&self, profile: &str, event: &ProgressEvent) -> Result<()> {
        let root = self.profile_root(profile)?;
        fs::create_dir_all(&root)?;
        let mut line = serde_json::to_vec(event)?;
        line.push(b'\n');
        use std::io::Write;
        let mut file = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(root.join("progress.jsonl"))?;
        file.write_all(&line)?;
        file.sync_data()?;
        Ok(())
    }

    pub fn write_comparison(
        &self,
        profile: &str,
        comparison: &DifferentialComparison,
    ) -> Result<PathBuf> {
        validate_component("task id", &comparison.task)?;
        let root = self
            .profile_root(profile)?
            .join("comparisons")
            .join(&comparison.task);
        fs::create_dir_all(&root)?;
        let path = root.join(format!("trial-{}.json", comparison.trial));
        write_json_atomic(&path, comparison)?;
        Ok(path)
    }

    pub fn read_preparation(&self, profile: &str) -> Result<PreparationReceipt> {
        read_json(&self.profile_root(profile)?.join("preparation.json"))
    }

    pub fn write_report(&self, profile: &str, report: &Value) -> Result<PathBuf> {
        let root = self.profile_root(profile)?;
        fs::create_dir_all(&root)?;
        let path = root.join("report.json");
        write_json_atomic(&path, report)?;
        Ok(path)
    }

    pub fn read_report(&self, profile: &str) -> Result<Option<Value>> {
        let path = self.profile_root(profile)?.join("report.json");
        if !path.is_file() {
            return Ok(None);
        }
        Ok(Some(read_json(&path)?))
    }

    pub fn write_attempt(&self, record: &AttemptRecord) -> Result<PathBuf> {
        validate_component("task id", &record.task)?;
        validate_component("differential arm name", &record.arm)?;
        let root = self
            .profile_root(&record.profile)?
            .join("attempts")
            .join(&record.task)
            .join(&record.arm)
            .join(record.trial.to_string());
        fs::create_dir_all(&root)?;
        let path = root.join("result.json");
        write_json_atomic(&path, record)?;
        Ok(path)
    }

    pub fn read_attempt(
        &self,
        profile: &str,
        task: &str,
        arm: &str,
        trial: u32,
    ) -> Result<Option<AttemptRecord>> {
        validate_component("task id", task)?;
        validate_component("differential arm name", arm)?;
        let root = self
            .profile_root(profile)?
            .join("attempts")
            .join(task)
            .join(arm);
        let path = root.join(trial.to_string()).join("result.json");
        if !path.exists() {
            return Ok(None);
        }
        Ok(Some(read_json(&path)?))
    }

    pub fn cleanup_profile(&self, profile: &str) -> Result<()> {
        let root = self.profile_root(profile)?;
        if root.exists() {
            fs::remove_dir_all(root)?;
        }
        Ok(())
    }
}

fn validate_component(label: &str, value: &str) -> Result<()> {
    if value.is_empty()
        || value == "."
        || value == ".."
        || value.contains('/')
        || value.contains('\\')
    {
        bail!("{label} must be one contained path component: {value:?}");
    }
    Ok(())
}

fn validate_relative_path(path: &Path, label: &str) -> Result<()> {
    if path.as_os_str().is_empty()
        || path.is_absolute()
        || path.components().any(|component| {
            matches!(
                component,
                Component::ParentDir | Component::RootDir | Component::Prefix(_)
            )
        })
    {
        bail!(
            "{label} must be a contained relative path: {}",
            path.display()
        );
    }
    Ok(())
}

fn digest_bytes(bytes: &[u8]) -> Result<String> {
    Ok(blake3::hash(bytes).to_hex().to_string())
}

fn plan_digest(plan: &DatasetPlan) -> Result<String> {
    let cases = plan
        .cases
        .iter()
        .map(|case| Ok((case.id.clone(), digest_path(&case.source_directory)?)))
        .collect::<Result<Vec<_>>>()?;
    dataset_digest(&plan.name, &plan.source, plan.claim_level, cases)
}

fn dataset_digest(
    name: &str,
    source: &SourceIdentity,
    claim_level: ClaimLevel,
    mut cases: Vec<(String, String)>,
) -> Result<String> {
    cases.sort_by(|left, right| left.0.cmp(&right.0));
    let value = serde_json::json!({
        "schema_version": PLATFORM_SCHEMA_VERSION,
        "name": name,
        "source": source,
        "claim_level": claim_level,
        "cases": cases
            .into_iter()
            .map(|(id, digest)| serde_json::json!({ "id": id, "digest": digest }))
            .collect::<Vec<_>>(),
    });
    digest_bytes(&serde_json::to_vec(&value)?)
}

fn digest_path(path: &Path) -> Result<String> {
    let metadata = fs::symlink_metadata(path)
        .with_context(|| format!("reading path for digest {}", path.display()))?;
    if metadata.is_file() {
        let mut bytes = Vec::with_capacity(1 + fs::metadata(path)?.len() as usize);
        bytes.push(b'F');
        bytes.extend_from_slice(&fs::read(path)?);
        return digest_bytes(&bytes);
    }
    if !metadata.is_dir() {
        bail!(
            "cannot digest unsupported filesystem node {}",
            path.display()
        );
    }
    let mut entries = fs::read_dir(path)?.collect::<io::Result<Vec<_>>>()?;
    entries.sort_by_key(|entry| entry.file_name());
    let mut value = Vec::new();
    for entry in entries {
        let name = entry.file_name().to_string_lossy().into_owned();
        value.extend_from_slice(name.as_bytes());
        value.push(0);
        let entry_metadata = fs::symlink_metadata(entry.path())?;
        value.push(if entry_metadata.is_dir() { b'D' } else { b'F' });
        value.extend_from_slice(digest_path(&entry.path())?.as_bytes());
        value.push(0xff);
    }
    digest_bytes(&value)
}

fn copy_tree(source: &Path, destination: &Path) -> Result<()> {
    let metadata = fs::symlink_metadata(source)?;
    if metadata.file_type().is_symlink() {
        bail!(
            "symlinks are not valid imported execution inputs: {}",
            source.display()
        );
    }
    if metadata.is_dir() {
        fs::create_dir_all(destination)?;
        let mut entries = fs::read_dir(source)?.collect::<io::Result<Vec<_>>>()?;
        entries.sort_by_key(|entry| entry.file_name());
        for entry in entries {
            copy_tree(&entry.path(), &destination.join(entry.file_name()))?;
        }
    } else if metadata.is_file() {
        if let Some(parent) = destination.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::copy(source, destination)?;
        fs::set_permissions(destination, metadata.permissions())?;
    } else {
        bail!("unsupported imported filesystem node: {}", source.display());
    }
    Ok(())
}

fn write_json_atomic<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    let parent = path.parent().context("JSON path has no parent")?;
    fs::create_dir_all(parent)?;
    let temporary = parent.join(format!(
        ".{}.tmp-{}-{}",
        path.file_name().unwrap_or_default().to_string_lossy(),
        std::process::id(),
        STAGING_COUNTER.fetch_add(1, Ordering::Relaxed)
    ));
    let result = (|| -> Result<()> {
        fs::write(&temporary, serde_json::to_vec_pretty(value)?)?;
        fs::File::open(&temporary)?.sync_all()?;
        fs::rename(&temporary, path)?;
        #[cfg(unix)]
        fs::File::open(parent)?.sync_all()?;
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    result
}

fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T> {
    let bytes = fs::read(path).with_context(|| format!("reading JSON {}", path.display()))?;
    serde_json::from_slice(&bytes).with_context(|| format!("parsing JSON {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn legacy_task_becomes_workspace_package() {
        let root = std::env::temp_dir().join(format!("hi-eval-platform-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(root.join("fixture")).unwrap();
        fs::write(root.join("fixture/file.txt"), "before").unwrap();
        let task: Task = toml::from_str(
            r#"
schema_version = 2
prompt = "fix it"
allowed_changes = ["**"]
[final_oracle]
command = "test -f file.txt"
"#,
        )
        .unwrap();
        let package = TaskPackage::from_legacy(&root, &task).unwrap();
        assert_eq!(package.output, EvalOutput::Workspace);
        assert!(package.verifier.is_some());
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn import_store_is_content_addressed_and_idempotent() {
        let root = std::env::temp_dir().join(format!("hi-eval-import-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        let source = root.join("source");
        let store = ImportStore::new(root.join("store"));
        fs::create_dir_all(&source).unwrap();
        fs::write(source.join("task.txt"), "hello").unwrap();
        let plan = DatasetPlan::new(
            "fixture",
            SourceIdentity {
                kind: "test".into(),
                revision: "test@1".into(),
                digest: "source".into(),
            },
        )
        .case(CasePlan::new("case-1", &source));
        let first = store.import(&plan).unwrap();
        let second = store.import(&plan).unwrap();
        assert_eq!(first.digest, second.digest);
        assert!(first.root.join("tasks/case-1/task.txt").is_file());
        let manifest = fs::read_to_string(first.root.join("dataset.json")).unwrap();
        assert!(!manifest.contains(source.to_string_lossy().as_ref()));
        fs::write(first.root.join("tasks/case-1/task.txt"), "tampered").unwrap();
        assert!(store.load(&first.root).is_err());
        let _ = fs::remove_dir_all(root);
    }

    #[cfg(unix)]
    #[test]
    fn evaluator_owned_commands_are_timed_and_kill_descendants() {
        let mut command = Command::new("sh");
        command.arg("-c").arg("sleep 10");
        let output = command_output_with_timeout(&mut command, Duration::from_millis(25)).unwrap();
        assert!(output.timed_out);
        assert!(!output.success());
    }

    #[test]
    fn durable_state_rejects_path_traversal_coordinates() {
        let store = EvalStateStore::new(
            std::env::temp_dir().join(format!("hi-eval-state-{}", std::process::id())),
        );
        let record = AttemptRecord {
            profile: "profile".into(),
            task: "../escape".into(),
            arm: "arm".into(),
            trial: 0,
            status: AttemptStatus::Failed,
            identity_digest: "digest".into(),
            claim_level: ClaimLevel::Smoke,
            score: None,
            evidence: None,
        };
        assert!(store.write_attempt(&record).is_err());
    }

    #[test]
    fn manifest_digest_changes_when_profile_changes() {
        let source = DatasetSource {
            adapter: "external".into(),
            source: "data".into(),
            revision: "data@1".into(),
            options: Value::Null,
        };
        let mut datasets = BTreeMap::new();
        datasets.insert("data".into(), source);
        let mut profiles = BTreeMap::new();
        profiles.insert(
            "smoke".into(),
            EvalProfile {
                datasets: vec!["data".into()],
                models: vec!["test/model".into()],
                trials: 1,
                backend: "host".into(),
                claim_level: ClaimLevel::Smoke,
                arms: Vec::new(),
                treatments: Vec::new(),
                selectors: Vec::new(),
                sampling: Value::Null,
                provider_policy: Value::Null,
                resources: ResourceSpec::default(),
                network: NetworkPolicy::default(),
                verifier: None,
                scoring: ScoringPolicy::default(),
                mcp_servers: Vec::new(),
                evidence: EvidencePolicy::default(),
                secret_configuration_digest: None,
            },
        );
        let manifest = EvalManifest {
            schema_version: PLATFORM_SCHEMA_VERSION,
            name: "test".into(),
            datasets,
            profiles,
        };
        let before = manifest.digest().unwrap();
        let mut changed = manifest.clone();
        changed.profiles.get_mut("smoke").unwrap().trials = 2;
        assert_ne!(before, changed.digest().unwrap());
    }

    #[test]
    fn scoring_policy_keeps_continuous_rewards_separate_from_pass() {
        let mut rewards = BTreeMap::new();
        rewards.insert("f1".to_string(), 0.4);
        let score = EvalScore::from_rewards(
            &ScoringPolicy::NamedReward {
                name: "f1".into(),
                threshold: 0.5,
            },
            true,
            rewards,
        );
        assert!(!score.passed);
        assert_eq!(score.rewards["f1"], 0.4);
    }

    #[test]
    fn task_package_rejects_escape_paths_and_empty_images() {
        let package = TaskPackage {
            schema_version: PLATFORM_SCHEMA_VERSION,
            id: "case".into(),
            name: None,
            source: SourceIdentity {
                kind: "test".into(),
                revision: "test@1".into(),
                digest: "digest".into(),
            },
            input: EvalInput::Prompt {
                prompt: "answer".into(),
            },
            output: EvalOutput::Workspace,
            environment: EnvironmentSpec::Oci {
                image: String::new(),
            },
            verifier: None,
            resources: ResourceSpec::default(),
            network: NetworkPolicy::Disabled,
            artifacts: vec![ArtifactSpec {
                source: PathBuf::from("../escape"),
                exclude: Vec::new(),
            }],
            scoring: ScoringPolicy::VerifierExit,
            claim_level: ClaimLevel::Smoke,
        };
        assert!(package.validate().is_err());
    }

    #[test]
    fn transcript_input_round_trips_roles_and_blocks() {
        let input = EvalInput::Transcript {
            messages: vec![
                TranscriptMessage {
                    role: "system".into(),
                    content: Value::String("system".into()),
                },
                TranscriptMessage {
                    role: "assistant".into(),
                    content: serde_json::json!([
                        {"type":"tool_call","id":"call-1","name":"read","arguments":{"path":"a"}}
                    ]),
                },
            ],
            final_prompt: Some("continue".into()),
        };
        let encoded = serde_json::to_vec(&input).unwrap();
        let decoded: EvalInput = serde_json::from_slice(&encoded).unwrap();
        assert_eq!(decoded, input);
    }

    #[cfg(unix)]
    #[test]
    fn import_rejects_symlinked_inputs() {
        let root = std::env::temp_dir().join(format!("hi-eval-symlink-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(root.join("source")).unwrap();
        fs::write(root.join("outside"), "secret").unwrap();
        std::os::unix::fs::symlink(root.join("outside"), root.join("source/link")).unwrap();
        let plan = DatasetPlan::new(
            "symlink",
            SourceIdentity {
                kind: "test".into(),
                revision: "test@1".into(),
                digest: "source".into(),
            },
        )
        .case(CasePlan::new("case", root.join("source")));
        assert!(ImportStore::new(root.join("store")).import(&plan).is_err());
        let _ = fs::remove_dir_all(root);
    }
}
