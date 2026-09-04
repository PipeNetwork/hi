//! Hardened execution for delegate and best-of child `hi` processes.

use std::ffi::OsString;
use std::fs::{File, OpenOptions};
use std::io;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, ensure};

static CHILD_RUNTIME: OnceLock<tokio::runtime::Runtime> = OnceLock::new();

#[derive(Clone, Debug)]
pub(crate) struct CandidateProcessIsolation {
    source_root: PathBuf,
    state_root: PathBuf,
}

impl CandidateProcessIsolation {
    pub(crate) fn new(source_root: &Path, state_root: &Path) -> Self {
        Self {
            source_root: source_root.to_path_buf(),
            state_root: state_root.to_path_buf(),
        }
    }

    fn denied_roots(&self) -> Vec<PathBuf> {
        let mut roots = vec![self.source_root.clone(), self.state_root.clone()];
        if let Some(home) = std::env::var_os("HOME").map(PathBuf::from) {
            roots.extend([
                home.join(".hi"),
                home.join(".config/hi"),
                home.join(".local/share/hi"),
            ]);
        }
        if let Some(path) = std::env::var_os("XDG_CONFIG_HOME").map(PathBuf::from) {
            roots.push(path.join("hi"));
        }
        if let Some(path) = std::env::var_os("XDG_DATA_HOME").map(PathBuf::from) {
            roots.push(path.join("hi"));
        }
        roots = roots
            .into_iter()
            .filter_map(|path| path.canonicalize().ok())
            .collect();
        roots.sort();
        roots.dedup();
        roots
    }
}

#[derive(Clone, Debug)]
pub(crate) struct CandidateChildPaths {
    workspace: PathBuf,
    root: PathBuf,
}

impl CandidateChildPaths {
    pub(crate) fn prepare(
        candidate: &hi_tools::candidate_workspace::CandidateWorkspace,
    ) -> Result<Self> {
        Self::prepare_runtime(candidate.root(), candidate.runtime_root())
    }

    #[cfg(test)]
    pub(crate) fn prepare_test(workspace: &Path, runtime: &Path) -> Result<Self> {
        std::fs::create_dir(runtime)
            .with_context(|| format!("creating test candidate runtime at {}", runtime.display()))?;
        secure_directory(runtime)?;
        Self::prepare_runtime(workspace, runtime)
    }

    fn prepare_runtime(workspace: &Path, runtime: &Path) -> Result<Self> {
        let workspace = canonical_real_directory(workspace, "candidate workspace")?;
        let runtime = canonical_real_directory(runtime, "candidate runtime")?;
        ensure!(
            runtime != workspace
                && !runtime.starts_with(&workspace)
                && !workspace.starts_with(&runtime),
            "candidate runtime {} must be separate from candidate workspace {}",
            runtime.display(),
            workspace.display()
        );

        // This directory is created only by the parent, outside snapshot
        // materialization. Refusing reuse means no source-controlled path or
        // pre-existing symlink can redirect a pre-sandbox write.
        let root = runtime.join("child");
        std::fs::create_dir(&root)
            .with_context(|| format!("creating isolated child runtime at {}", root.display()))?;
        secure_directory(&root)?;
        for name in ["output", "build-cache", "bin"] {
            let path = root.join(name);
            std::fs::create_dir(&path)
                .with_context(|| format!("creating isolated child directory {}", path.display()))?;
            secure_directory(&path)?;
        }
        Ok(Self { workspace, root })
    }

    pub(crate) fn workspace_root(&self) -> &Path {
        &self.workspace
    }

    pub(crate) fn runtime_root(&self) -> &Path {
        &self.root
    }

    pub(crate) fn report(&self) -> PathBuf {
        self.root.join("output/report.json")
    }

    pub(crate) fn events(&self) -> PathBuf {
        self.root.join("output/events.jsonl")
    }

    pub(crate) fn build_cache(&self, name: &str) -> OsString {
        self.root.join("build-cache").join(name).into_os_string()
    }

    pub(crate) fn delegate_environment(&self, api_key: &str) -> Vec<(OsString, OsString)> {
        vec![
            ("HI_FORCE_API_KEY".into(), api_key.into()),
            ("HI_API_KEY".into(), api_key.into()),
            ("CARGO_TARGET_DIR".into(), self.build_cache("cargo-target")),
            ("CARGO_HOME".into(), self.build_cache("cargo-home")),
            ("SCCACHE_DIR".into(), self.build_cache("sccache")),
            // Enforce normal folder trust; never inherit an operator override.
            ("HI_FOLDER_TRUST".into(), "on".into()),
        ]
    }

    pub(crate) fn retain(&self, report: &Path, events: Option<&Path>) {
        let _ = copy_regular_no_follow(&self.report(), report);
        if let Some(events) = events {
            let _ = copy_regular_no_follow(&self.events(), events);
        }
    }
}

fn canonical_real_directory(path: &Path, kind: &str) -> Result<PathBuf> {
    let metadata = std::fs::symlink_metadata(path)
        .with_context(|| format!("reading {kind} metadata at {}", path.display()))?;
    ensure!(
        metadata.file_type().is_dir() && !metadata.file_type().is_symlink(),
        "{kind} must be a real directory, not a symlink: {}",
        path.display()
    );
    path.canonicalize()
        .with_context(|| format!("canonicalizing {kind} {}", path.display()))
}

fn secure_directory(path: &Path) -> Result<()> {
    let metadata = std::fs::symlink_metadata(path)
        .with_context(|| format!("reading private directory metadata at {}", path.display()))?;
    ensure!(
        metadata.file_type().is_dir() && !metadata.file_type().is_symlink(),
        "private candidate path must be a real directory: {}",
        path.display()
    );
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt as _;
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o700))
            .with_context(|| format!("securing private candidate directory {}", path.display()))?;
    }
    Ok(())
}

fn copy_regular_no_follow(source: &Path, destination: &Path) -> Result<()> {
    let mut input = open_read_no_follow(source)?;
    ensure!(
        input.metadata()?.is_file(),
        "candidate child artifact is not a regular file: {}",
        source.display()
    );
    let mut output = File::create(destination)
        .with_context(|| format!("creating retained child artifact {}", destination.display()))?;
    io::copy(&mut input, &mut output).with_context(|| {
        format!(
            "retaining candidate child artifact {} at {}",
            source.display(),
            destination.display()
        )
    })?;
    Ok(())
}

fn open_read_no_follow(path: &Path) -> Result<File> {
    let mut options = OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt as _;
        options.custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC);
    }
    let file = options
        .open(path)
        .with_context(|| format!("opening candidate child artifact {}", path.display()))?;
    #[cfg(not(unix))]
    ensure!(
        !std::fs::symlink_metadata(path)?.file_type().is_symlink(),
        "candidate child artifact must not be a symlink: {}",
        path.display()
    );
    Ok(file)
}

fn stage_executable(source: &Path, destination: &Path) -> Result<()> {
    let mut input = open_read_no_follow(source)?;
    let metadata = input
        .metadata()
        .with_context(|| format!("reading child executable metadata {}", source.display()))?;
    ensure!(
        metadata.is_file(),
        "child executable is not a regular file: {}",
        source.display()
    );
    let parent = destination
        .parent()
        .context("staged child executable has no parent directory")?;
    canonical_real_directory(parent, "staged executable directory")?;
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt as _;
        options.custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC);
    }
    let mut output = options.open(destination).with_context(|| {
        format!(
            "creating isolated child executable {}",
            destination.display()
        )
    })?;
    io::copy(&mut input, &mut output).with_context(|| {
        format!(
            "copying source-contained child executable {} to {}",
            source.display(),
            destination.display()
        )
    })?;
    output
        .set_permissions(metadata.permissions())
        .with_context(|| {
            format!(
                "preserving child executable mode at {}",
                destination.display()
            )
        })?;
    output
        .sync_all()
        .with_context(|| format!("syncing staged child executable {}", destination.display()))?;
    Ok(())
}

fn child_runtime() -> Result<&'static tokio::runtime::Runtime> {
    if let Some(runtime) = CHILD_RUNTIME.get() {
        return Ok(runtime);
    }
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(
            std::thread::available_parallelism()
                .map(usize::from)
                .unwrap_or(2)
                .clamp(2, 8),
        )
        .enable_all()
        .build()
        .context("creating shared child-process runtime")?;
    let _ = CHILD_RUNTIME.set(runtime);
    Ok(CHILD_RUNTIME.get().expect("child runtime initialized"))
}

pub(crate) struct CandidateChildLaunch<'a> {
    pub(crate) workspace_root: &'a Path,
    pub(crate) runtime_root: &'a Path,
    pub(crate) executable: &'a Path,
    pub(crate) arguments: Vec<OsString>,
    pub(crate) environment: Vec<(OsString, OsString)>,
    pub(crate) timeout: Option<Duration>,
    pub(crate) log_path: &'a Path,
    pub(crate) cancellation: Option<hi_agent::TurnCancellation>,
    pub(crate) isolation: CandidateProcessIsolation,
}

pub(crate) fn run_maybe_cancelled(
    launch: CandidateChildLaunch<'_>,
) -> Result<hi_tools::ProcessExecution> {
    let CandidateChildLaunch {
        workspace_root,
        runtime_root,
        executable,
        arguments,
        environment,
        timeout,
        log_path,
        cancellation,
        isolation,
    } = launch;
    let workspace_root = workspace_root.to_path_buf();
    let runtime_root = runtime_root.to_path_buf();
    let executable = executable.to_path_buf();
    let log_path = log_path.to_path_buf();
    if let Some(parent) = log_path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating child log directory {}", parent.display()))?;
    }
    // Establish the durable artifact before launch. The typed result replaces
    // it after completion, including timeout/failure information when present.
    std::fs::write(&log_path, [])
        .with_context(|| format!("creating child log {}", log_path.display()))?;

    let execution = child_runtime()?.block_on(async move {
        let denied_roots = isolation.denied_roots();
        let workspace_root = canonical_real_directory(&workspace_root, "candidate workspace")?;
        let private_root = canonical_real_directory(&runtime_root, "candidate child runtime")?;
        anyhow::ensure!(
            !denied_roots
                .iter()
                .any(|root| workspace_root.starts_with(root)),
            "candidate workspace {} is nested under a denied source/state root",
            workspace_root.display()
        );
        anyhow::ensure!(
            private_root != workspace_root
                && !private_root.starts_with(&workspace_root)
                && !workspace_root.starts_with(&private_root)
                && !denied_roots
                    .iter()
                    .any(|root| private_root.starts_with(root)),
            "candidate child runtime {} is not separate from workspace/source/state roots",
            private_root.display()
        );
        let executable = executable
            .canonicalize()
            .with_context(|| format!("canonicalizing child executable {}", executable.display()))?;
        let isolated_executable = if denied_roots.iter().any(|root| executable.starts_with(root)) {
            let staged = private_root.join("bin/hi-child");
            stage_executable(&executable, &staged)?;
            staged
        } else {
            executable
        };
        let runner = hi_tools::ProcessRunner::new_with_policy_and_config(
            &workspace_root,
            hi_tools::sandbox::SandboxPolicy::Workspace,
            hi_tools::sandbox::SandboxConfig {
                deny_read: denied_roots,
                deny_host_temp: true,
                private_temp: Some(private_root.clone()),
                ..hi_tools::sandbox::SandboxConfig::default()
            },
        )?;
        anyhow::ensure!(
            runner.sandbox_enforced(),
            "candidate child requires an enforced network-capable workspace sandbox"
        );
        let mut environment = environment
            .into_iter()
            .filter(|(name, _)| {
                let name = name.to_string_lossy().to_ascii_uppercase();
                !name.starts_with("HI_SYNC_")
                    && !name.starts_with("HI_PIPEFS_")
                    && !name.contains("LEASE_TOKEN")
            })
            .collect::<Vec<_>>();
        environment.extend([
            (
                OsString::from("XDG_DATA_HOME"),
                private_root.join("data").into_os_string(),
            ),
            (
                OsString::from("XDG_CONFIG_HOME"),
                private_root.join("config").into_os_string(),
            ),
            (
                OsString::from("XDG_CACHE_HOME"),
                private_root.join("cache").into_os_string(),
            ),
        ]);
        let foreground = runner.foreground_registry();
        let started = Instant::now();
        let run = runner.run_program_with_env_maybe_timeout(
            isolated_executable,
            arguments,
            environment,
            timeout,
        );
        tokio::pin!(run);
        match cancellation {
            Some(cancel) => {
                tokio::select! {
                    result = &mut run => result,
                    _ = wait_for_cancel(cancel) => {
                        foreground.kill_current();
                        let reaped = run.await;
                        anyhow::ensure!(
                            foreground.active_count() == 0,
                            "cancelled candidate child returned before its process was reaped"
                        );
                        reaped.map(|_| cancelled_execution(started))
                    },
                }
            }
            None => (&mut run).await,
        }
    })?;

    write_log(&log_path, &execution)?;
    Ok(execution)
}

pub(crate) async fn wait_for_cancel(cancellation: hi_agent::TurnCancellation) {
    while !cancellation.is_cancelled() {
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
}

fn cancelled_execution(started: Instant) -> hi_tools::ProcessExecution {
    hi_tools::ProcessExecution {
        status: hi_tools::ToolStatus::Cancelled,
        outcome: hi_tools::ProcessOutcome {
            exit_code: None,
            stdout_summary: String::new(),
            stderr_summary: "cancelled".into(),
            duration_ms: started.elapsed().as_millis().min(u128::from(u64::MAX)) as u64,
        },
        truncation: hi_tools::TruncationState::Complete,
    }
}

fn write_log(path: &PathBuf, execution: &hi_tools::ProcessExecution) -> Result<()> {
    let text = format!(
        "status: {:?}\nexit_code: {:?}\nduration_ms: {}\ntruncation: {:?}\n\nstdout:\n{}\n\nstderr:\n{}\n",
        execution.status,
        execution.outcome.exit_code,
        execution.outcome.duration_ms,
        execution.truncation,
        execution.outcome.stdout_summary,
        execution.outcome.stderr_summary,
    );
    std::fs::write(path, text).with_context(|| format!("writing child log {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static TEST_ID: AtomicU64 = AtomicU64::new(0);

    fn temp_dir(label: &str) -> PathBuf {
        let id = TEST_ID.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "hi-child-process-{label}-{}-{id}",
            std::process::id()
        ));
        std::fs::create_dir_all(&path).unwrap();
        path
    }

    #[cfg(any(target_os = "macos", target_os = "linux"))]
    #[test]
    fn run_stops_when_cancelled() {
        let owner = temp_dir("cancel");
        let dir = owner.join("candidate");
        let source = owner.join("source");
        let state = owner.join("state");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::create_dir_all(&source).unwrap();
        std::fs::create_dir_all(&state).unwrap();
        let child_paths = CandidateChildPaths::prepare_test(&dir, &owner.join("runtime")).unwrap();
        let pid_path = dir.join("child.pid");
        let cancel = hi_agent::TurnCancellation::new();
        let cancel_thread = cancel.clone();
        let cancel_pid_path = pid_path.clone();
        std::thread::spawn(move || {
            while std::fs::read_to_string(&cancel_pid_path)
                .map(|text| text.trim().is_empty())
                .unwrap_or(true)
            {
                std::thread::sleep(Duration::from_millis(5));
            }
            cancel_thread.cancel();
        });
        let started = Instant::now();
        let execution = run_maybe_cancelled(CandidateChildLaunch {
            workspace_root: &dir,
            runtime_root: child_paths.runtime_root(),
            executable: Path::new("/bin/sh"),
            arguments: vec![
                OsString::from("-c"),
                OsString::from("echo $$ > child.pid; exec sleep 30"),
            ],
            environment: Vec::new(),
            timeout: None,
            log_path: &dir.join("child.log"),
            cancellation: Some(cancel),
            isolation: CandidateProcessIsolation::new(&source, &state),
        })
        .unwrap();
        assert_eq!(execution.status, hi_tools::ToolStatus::Cancelled);
        assert!(
            started.elapsed() < Duration::from_secs(3),
            "cancel must kill the child instead of waiting out the timeout: {:?}",
            started.elapsed()
        );
        #[cfg(unix)]
        {
            let pid = std::fs::read_to_string(&pid_path)
                .unwrap()
                .trim()
                .parse::<i32>()
                .unwrap();
            assert_eq!(unsafe { libc::kill(pid, 0) }, -1, "child must be reaped");
            assert_eq!(
                std::io::Error::last_os_error().raw_os_error(),
                Some(libc::ESRCH)
            );
        }
        let _ = std::fs::remove_dir_all(&owner);
    }

    #[cfg(any(target_os = "macos", target_os = "linux"))]
    #[test]
    fn candidate_sandbox_masks_source_and_state_but_allows_candidate_writes() {
        let owner = temp_dir("isolation");
        let candidate = owner.join("candidate");
        let source = owner.join("source");
        let state = owner.join("state");
        for path in [&candidate, &source, &state] {
            std::fs::create_dir_all(path).unwrap();
        }
        let child_paths =
            CandidateChildPaths::prepare_test(&candidate, &owner.join("runtime")).unwrap();
        let source_secret = source.join("secret.txt");
        let state_secret = state.join("lease-token");
        std::fs::write(&source_secret, "source-secret").unwrap();
        std::fs::write(&state_secret, "lease-secret").unwrap();
        let script = format!(
            "test -z \"${{HI_SYNC_LEASE_TOKEN+x}}\" && test ! -r '{}' && test ! -r '{}' && printf allowed > allowed.txt",
            source_secret.display(),
            state_secret.display()
        );
        let execution = run_maybe_cancelled(CandidateChildLaunch {
            workspace_root: &candidate,
            runtime_root: child_paths.runtime_root(),
            executable: Path::new("/bin/sh"),
            arguments: vec![OsString::from("-c"), OsString::from(script)],
            environment: vec![("HI_SYNC_LEASE_TOKEN".into(), "must-not-pass".into())],
            timeout: Some(Duration::from_secs(5)),
            log_path: &owner.join("child.log"),
            cancellation: None,
            isolation: CandidateProcessIsolation::new(&source, &state),
        })
        .unwrap();
        assert_eq!(execution.status, hi_tools::ToolStatus::Succeeded);
        assert_eq!(
            std::fs::read_to_string(candidate.join("allowed.txt")).unwrap(),
            "allowed"
        );
        let _ = std::fs::remove_dir_all(&owner);
    }

    #[cfg(any(target_os = "macos", target_os = "linux"))]
    #[test]
    fn source_contained_executable_is_staged_before_source_is_masked() {
        use std::os::unix::fs::PermissionsExt as _;

        let owner = temp_dir("source-executable");
        let candidate = owner.join("candidate");
        let source = owner.join("source");
        let state = owner.join("state");
        for path in [&candidate, &source, &state] {
            std::fs::create_dir_all(path).unwrap();
        }
        let child_paths =
            CandidateChildPaths::prepare_test(&candidate, &owner.join("runtime")).unwrap();
        let executable = source.join("child.sh");
        std::fs::write(
            &executable,
            "#!/bin/sh\ntest ! -r \"$1\" && printf staged > staged.txt\n",
        )
        .unwrap();
        std::fs::set_permissions(&executable, std::fs::Permissions::from_mode(0o755)).unwrap();
        let secret = source.join("secret.txt");
        std::fs::write(&secret, "secret").unwrap();

        let execution = run_maybe_cancelled(CandidateChildLaunch {
            workspace_root: &candidate,
            runtime_root: child_paths.runtime_root(),
            executable: &executable,
            arguments: vec![secret.into_os_string()],
            environment: Vec::new(),
            timeout: Some(Duration::from_secs(5)),
            log_path: &owner.join("child.log"),
            cancellation: None,
            isolation: CandidateProcessIsolation::new(&source, &state),
        })
        .unwrap();

        assert_eq!(execution.status, hi_tools::ToolStatus::Succeeded);
        assert_eq!(
            std::fs::read_to_string(candidate.join("staged.txt")).unwrap(),
            "staged"
        );
        let _ = std::fs::remove_dir_all(&owner);
    }

    #[cfg(any(target_os = "macos", target_os = "linux"))]
    #[test]
    fn source_controlled_child_runtime_symlink_cannot_redirect_parent_writes() {
        use std::os::unix::fs::{PermissionsExt as _, symlink};

        let owner = temp_dir("malicious-runtime-symlink");
        let source = owner.join("source");
        let state = owner.join("state");
        let external = owner.join("external");
        for path in [&source, &state, &external] {
            std::fs::create_dir(path).unwrap();
        }
        std::fs::create_dir(source.join(".hi")).unwrap();
        symlink(&external, source.join(".hi/candidate-child")).unwrap();
        let executable = source.join("child.sh");
        std::fs::write(
            &executable,
            "#!/bin/sh\ntest ! -r \"$1\" && printf staged > staged.txt\n",
        )
        .unwrap();
        std::fs::set_permissions(&executable, std::fs::Permissions::from_mode(0o755)).unwrap();
        let secret = source.join("secret.txt");
        std::fs::write(&secret, "secret").unwrap();

        let candidate = hi_tools::candidate_workspace::CandidateWorkspace::create(
            &source,
            &state,
            &owner.join("detached"),
        )
        .unwrap();
        assert!(
            std::fs::symlink_metadata(candidate.root().join(".hi/candidate-child"))
                .unwrap()
                .file_type()
                .is_symlink()
        );
        let child_paths = CandidateChildPaths::prepare(&candidate).unwrap();
        let runtime = child_paths.runtime_root().canonicalize().unwrap();
        assert!(!runtime.starts_with(candidate.root().canonicalize().unwrap()));
        assert!(!runtime.starts_with(source.canonicalize().unwrap()));
        assert!(!source.canonicalize().unwrap().starts_with(&runtime));
        assert!(std::fs::read_dir(&external).unwrap().next().is_none());

        let execution = run_maybe_cancelled(CandidateChildLaunch {
            workspace_root: candidate.root(),
            runtime_root: child_paths.runtime_root(),
            executable: &executable,
            arguments: vec![secret.into_os_string()],
            environment: child_paths.delegate_environment("test-key"),
            timeout: Some(Duration::from_secs(5)),
            log_path: &owner.join("child.log"),
            cancellation: None,
            isolation: CandidateProcessIsolation::new(&source, &state),
        })
        .unwrap();

        assert_eq!(execution.status, hi_tools::ToolStatus::Succeeded);
        assert_eq!(
            std::fs::read_to_string(candidate.root().join("staged.txt")).unwrap(),
            "staged"
        );
        assert!(
            std::fs::read_dir(&external).unwrap().next().is_none(),
            "candidate snapshot symlink redirected a parent runtime write"
        );
        drop(candidate);
        let _ = std::fs::remove_dir_all(&owner);
    }
}
