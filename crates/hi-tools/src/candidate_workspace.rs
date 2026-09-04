//! Detached candidate workspaces for write-capable agents.
//!
//! A candidate owns both its work tree and Git metadata. It never registers a
//! worktree in the source repository and never points `.git` back at source
//! bytes. The captured snapshot ID is the exact-base receipt used by the
//! parent before applying a verified result.

use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::Mutex;

use anyhow::{Context, Result, bail, ensure};
use hi_workspace::{
    CandidateChange, CandidateDestinationVerifier, CandidateFileKind, CandidateFileState,
    CandidateId, CandidatePostimage, CandidateRoute, CandidateVerification, JobId,
    VerifiedCandidate, VerifiedCandidateDraft, WorkspaceBinding,
};
use serde::{Deserialize, Serialize};

#[path = "candidate_artifact.rs"]
mod artifact;
pub use artifact::PersistedDetachedCandidate;
pub(crate) use artifact::read_resource_body as read_candidate_artifact_resource;
#[path = "candidate_publication.rs"]
mod publication;
pub use publication::{
    CandidatePublicationError, CandidatePublicationErrorKind, apply_verified_candidate_and_reverify,
};

static CANDIDATE_APPLY_LOCK: Mutex<()> = Mutex::new(());

const GENERATED_EXCLUDES: &[&str] = &[
    ":(exclude,glob)**/__pycache__/**",
    ":(exclude,glob)**/*.pyc",
    ":(exclude,glob)**/*.pyo",
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CandidateMaterialization {
    /// A private no-local clone supplied history; the work tree came from the
    /// exact internal snapshot.
    PrivateClone,
    /// A non-Git workspace received a new private repository.
    PrivateInit,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DetachedVerifiedCandidate {
    pub candidate: VerifiedCandidate,
    pub source_snapshot_id: String,
}

#[derive(Clone, Debug)]
pub struct CandidateSealContext {
    pub job_id: JobId,
    pub binding: WorkspaceBinding,
    pub route: CandidateRoute,
    pub verification: Vec<CandidateVerification>,
    pub destination_verification: Vec<CandidateDestinationVerifier>,
    pub destination_verification_budget_ms: u64,
}

/// RAII owner for an exact, detached workspace snapshot.
#[derive(Debug)]
pub struct CandidateWorkspace {
    owner: PathBuf,
    root: PathBuf,
    runtime_root: PathBuf,
    source: PathBuf,
    state_root: PathBuf,
    source_snapshot_id: String,
    baseline_commit: String,
    materialization: CandidateMaterialization,
    armed: bool,
}

impl CandidateWorkspace {
    /// Capture `source` and materialize it beneath a newly-created
    /// `destination`. The destination must not exist and becomes wholly owned
    /// by this value until [`CandidateWorkspace::keep`] is called.
    pub fn create(source: &Path, state_root: &Path, destination: &Path) -> Result<Self> {
        ensure!(
            !destination.exists(),
            "candidate destination already exists: {}",
            destination.display()
        );
        let source = source
            .canonicalize()
            .with_context(|| format!("canonicalizing candidate source {}", source.display()))?;
        let state_root = absolute_path(state_root)?;
        let snapshot_id = crate::internal_snapshot::create(&source, &state_root)
            .context("capturing exact candidate base")?;

        fs::create_dir(destination)
            .with_context(|| format!("creating candidate owner {}", destination.display()))?;
        let mut cleanup = OwnedCandidateDirectory {
            path: destination.to_path_buf(),
            armed: true,
        };
        let root = destination.join("workspace");
        crate::internal_snapshot::materialize(&source, &state_root, &snapshot_id, &root)
            .context("materializing exact candidate base")?;
        ensure_no_nested_git_metadata(&root)?;
        let runtime_root = destination.join("runtime");
        fs::create_dir(&runtime_root).context("creating private candidate runtime directory")?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt as _;
            fs::set_permissions(&runtime_root, fs::Permissions::from_mode(0o700))
                .context("securing private candidate runtime directory")?;
        }

        let materialization = if crate::worktree::in_git_repo(&source) {
            let repository_root = git_output(&source, ["rev-parse", "--show-toplevel"])?;
            let repository_root = PathBuf::from(
                String::from_utf8(repository_root.stdout)
                    .context("Git returned a non-UTF-8 repository root")?
                    .trim(),
            )
            .canonicalize()
            .context("canonicalizing Git repository root")?;
            let clone = destination.join("repository");
            let output = Command::new("git")
                .args([
                    "clone",
                    "--no-checkout",
                    "--no-local",
                    "--no-hardlinks",
                    "--",
                ])
                .arg(&repository_root)
                .arg(&clone)
                .output()
                .context("running detached candidate clone")?;
            require_git_success(output, "git clone --no-local")?;
            let private_git = clone.join(".git");
            ensure!(
                private_git.is_dir(),
                "detached candidate clone did not create private Git metadata"
            );
            fs::rename(&private_git, root.join(".git"))
                .context("installing private candidate Git metadata")?;
            remove_owned_directory(destination, &clone)?;
            CandidateMaterialization::PrivateClone
        } else {
            let output = Command::new("git")
                .arg("init")
                .arg("--quiet")
                .arg("--")
                .arg(&root)
                .output()
                .context("initializing private candidate Git repository")?;
            require_git_success(output, "git init")?;
            CandidateMaterialization::PrivateInit
        };

        configure_private_repository(destination, &root)?;
        let baseline_commit = commit_exact_baseline(&root)?;
        ensure_private_git_dir(destination, &root)?;

        cleanup.disarm();
        Ok(Self {
            owner: destination.to_path_buf(),
            root,
            runtime_root,
            source,
            state_root,
            source_snapshot_id: snapshot_id,
            baseline_commit,
            materialization,
            armed: true,
        })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Parent-owned runtime storage that is outside the candidate snapshot.
    /// Child reports, caches, private temp, and staged launchers belong here so
    /// source-controlled symlinks can never redirect pre-sandbox writes.
    pub fn runtime_root(&self) -> &Path {
        &self.runtime_root
    }

    pub fn source_snapshot_id(&self) -> &str {
        &self.source_snapshot_id
    }

    pub fn baseline_commit(&self) -> &str {
        &self.baseline_commit
    }

    pub fn materialization(&self) -> CandidateMaterialization {
        self.materialization
    }

    /// Fail unless the complete authoritative source still matches the exact
    /// snapshot from which this candidate was created.
    pub fn ensure_source_unchanged(&self) -> Result<()> {
        ensure_workspace_matches(&self.source, &self.state_root, &self.source_snapshot_id)
    }

    /// Seal the complete private-repository delta into a bounded typed
    /// candidate. The authoritative source is checked both before and after
    /// candidate inspection so a concurrent parent edit makes this fail stale.
    pub fn seal_verified(
        &self,
        context: CandidateSealContext,
    ) -> Result<DetachedVerifiedCandidate> {
        ensure!(
            context
                .binding
                .workspace_root
                .canonicalize()
                .ok()
                .as_deref()
                == Some(&self.source),
            "candidate binding does not name its authoritative source"
        );
        self.ensure_source_unchanged()?;
        stage_candidate(self.root())?;
        let before_tree = git_text(
            self.root(),
            ["rev-parse", &format!("{}^{{tree}}", self.baseline_commit)],
        )?;
        let after_tree = git_text(self.root(), ["write-tree"])?;
        let paths = changed_paths(self.root(), self.baseline_commit())?;
        let mut changes = Vec::with_capacity(paths.len());
        for path in paths {
            let before = baseline_file(self.root(), self.baseline_commit(), &path)?;
            let after = candidate_file(self.root(), &path)?;
            ensure!(
                before
                    .as_ref()
                    .is_none_or(|(kind, _, _)| *kind == CandidateFileKind::Regular)
                    && after
                        .as_ref()
                        .is_none_or(|(kind, _, _)| *kind == CandidateFileKind::Regular),
                "candidate changes unsupported symlink or special path {}",
                path.display()
            );
            changes.push(CandidateChange {
                path,
                before: before.map(|(kind, mode, bytes)| CandidateFileState {
                    kind,
                    mode,
                    content_digest: content_digest(&bytes),
                }),
                after: after.map(|(kind, mode, bytes)| CandidatePostimage::new(kind, mode, bytes)),
            });
        }
        self.ensure_source_unchanged()?;
        let candidate = VerifiedCandidate::create(VerifiedCandidateDraft {
            candidate_id: CandidateId::new(uuid::Uuid::new_v4().to_string()),
            job_id: context.job_id,
            source_binding_id: context.binding.binding_id,
            source_epoch: context.binding.epoch,
            base_version: context.binding.version,
            before_digest: format!("git:{before_tree}"),
            after_digest: format!("git:{after_tree}"),
            changes,
            verification: context.verification,
            destination_verification: context.destination_verification,
            destination_verification_budget_ms: context.destination_verification_budget_ms,
            artifacts: Vec::new(),
            effective_route: context.route,
        })?;
        Ok(DetachedVerifiedCandidate {
            candidate,
            source_snapshot_id: self.source_snapshot_id.clone(),
        })
    }

    /// Transfer ownership of the candidate directory to the caller.
    pub fn keep(mut self) -> PathBuf {
        self.armed = false;
        self.owner.clone()
    }
}

/// Revalidate a persisted candidate base without recreating an executable
/// candidate capability. Used by the parent's serialized apply path.
pub fn ensure_workspace_matches(root: &Path, state_root: &Path, snapshot_id: &str) -> Result<()> {
    crate::internal_snapshot::ensure_current_matches(root, state_root, snapshot_id)
        .context("candidate base is stale")
}

/// Parent-only, serialized exact-base application. All postimages are first
/// materialized into the shared transaction engine; a stale binding, stale
/// complete snapshot, or preimage race fails before publication.
pub fn apply_verified_candidate(
    detached: &DetachedVerifiedCandidate,
    binding: &WorkspaceBinding,
    state_root: &Path,
) -> Result<Vec<crate::FileChange>> {
    let _serialized = CANDIDATE_APPLY_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    detached.candidate.validate_for_apply(binding)?;
    ensure_workspace_matches(
        &binding.workspace_root,
        state_root,
        &detached.source_snapshot_id,
    )?;
    let mutations = detached
        .candidate
        .changes
        .iter()
        .map(|change| match (&change.before, &change.after) {
            (None, Some(after)) if after.kind == CandidateFileKind::Regular => {
                Ok(crate::PlannedFileMutation::add_with_mode(
                    &change.path,
                    after.bytes.clone(),
                    after.mode,
                ))
            }
            (Some(before), Some(after))
                if before.kind == CandidateFileKind::Regular
                    && after.kind == CandidateFileKind::Regular =>
            {
                Ok(crate::PlannedFileMutation::update_with_mode(
                    &change.path,
                    after.bytes.clone(),
                    after.mode,
                ))
            }
            (Some(before), None) if before.kind == CandidateFileKind::Regular => {
                Ok(crate::PlannedFileMutation::delete(&change.path))
            }
            _ => bail!("candidate contains an unsupported filesystem-node change"),
        })
        .collect::<Result<Vec<_>>>()?;
    let plan = crate::MutationPlan::new_with_state(&binding.workspace_root, state_root, mutations)?;
    ensure!(!plan.is_noop(), "candidate produces no destination changes");
    ensure_workspace_matches(
        &binding.workspace_root,
        state_root,
        &detached.source_snapshot_id,
    )?;
    plan.commit()
}

impl Drop for CandidateWorkspace {
    fn drop(&mut self) {
        if self.armed {
            let _ = remove_owned_root(&self.owner);
        }
    }
}

struct OwnedCandidateDirectory {
    path: PathBuf,
    armed: bool,
}

impl OwnedCandidateDirectory {
    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for OwnedCandidateDirectory {
    fn drop(&mut self) {
        if self.armed {
            let _ = remove_owned_root(&self.path);
        }
    }
}

fn configure_private_repository(owner: &Path, root: &Path) -> Result<()> {
    let hooks = owner.join("disabled-hooks");
    fs::create_dir(&hooks).context("creating inert candidate hooks directory")?;
    for (key, value) in [
        ("core.hooksPath", hooks.to_string_lossy().as_ref()),
        ("commit.gpgsign", "false"),
        ("tag.gpgsign", "false"),
    ] {
        let output = Command::new("git")
            .arg("-C")
            .arg(root)
            .args(["config", "--local", key, value])
            .output()
            .with_context(|| format!("configuring private candidate Git key {key}"))?;
        require_git_success(output, "git config")?;
    }
    // A local clone normally retains an `origin` URL pointing back at the
    // authoritative repository. Even though the candidate process is
    // write-confined on supported platforms, keeping that transport would let
    // an unenforced/disabled sandbox mutate source refs with `git push`.
    // Candidate repositories are deliberately publication-incapable: the
    // parent is the only component allowed to apply their verified bytes.
    let remotes = git_output(root, ["remote"])?;
    let remotes = String::from_utf8(remotes.stdout).context("Git returned non-UTF-8 remotes")?;
    for remote in remotes.lines().filter(|remote| !remote.trim().is_empty()) {
        let output = Command::new("git")
            .arg("-C")
            .arg(root)
            .args(["remote", "remove", remote])
            .output()
            .with_context(|| format!("removing candidate Git remote {remote}"))?;
        require_git_success(output, "git remote remove")?;
    }
    Ok(())
}

fn commit_exact_baseline(root: &Path) -> Result<String> {
    let output = Command::new("git")
        .arg("-C")
        .arg(root)
        .args(["add", "-A", "-f", "--", "."])
        .output()
        .context("staging exact candidate baseline")?;
    require_git_success(output, "git add")?;
    let output = Command::new("git")
        .arg("-C")
        .arg(root)
        .args([
            "-c",
            "user.name=hi candidate",
            "-c",
            "user.email=hi-candidate@invalid",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "--quiet",
            "--no-gpg-sign",
            "--allow-empty",
            "-m",
            "hi detached candidate baseline",
        ])
        .output()
        .context("committing exact candidate baseline")?;
    require_git_success(output, "git commit")?;
    let output = git_output(root, ["rev-parse", "HEAD"])?;
    Ok(String::from_utf8(output.stdout)
        .context("Git returned a non-UTF-8 candidate commit")?
        .trim()
        .to_string())
}

fn ensure_private_git_dir(owner: &Path, root: &Path) -> Result<()> {
    ensure!(
        !root.join(".git/objects/info/alternates").exists(),
        "candidate repository unexpectedly borrows source Git objects"
    );
    let output = git_output(root, ["rev-parse", "--git-common-dir"])?;
    let common = PathBuf::from(
        String::from_utf8(output.stdout)
            .context("Git returned a non-UTF-8 common directory")?
            .trim(),
    );
    let common = if common.is_absolute() {
        common
    } else {
        root.join(common)
    }
    .canonicalize()
    .context("canonicalizing candidate Git common directory")?;
    let owner = owner
        .canonicalize()
        .context("canonicalizing candidate owner")?;
    ensure!(
        common.starts_with(&owner) && common != owner,
        "candidate Git metadata escaped its private owner: {}",
        common.display()
    );
    Ok(())
}

fn ensure_no_nested_git_metadata(root: &Path) -> Result<()> {
    let mut pending = vec![root.to_path_buf()];
    while let Some(directory) = pending.pop() {
        for entry in fs::read_dir(&directory)
            .with_context(|| format!("inspecting candidate directory {}", directory.display()))?
        {
            let entry = entry?;
            let path = entry.path();
            if entry.file_name() == OsStr::new(".git") {
                bail!(
                    "candidate snapshot contains nested Git metadata at {}; submodules require an isolated materializer",
                    path.display()
                );
            }
            if entry.file_type()?.is_dir() {
                pending.push(path);
            }
        }
    }
    Ok(())
}

fn git_output<const N: usize>(root: &Path, args: [&str; N]) -> Result<Output> {
    let output = Command::new("git")
        .arg("-C")
        .arg(root)
        .args(args)
        .output()
        .context("running Git for detached candidate")?;
    require_git_success(output, "git command")
}

fn require_git_success(output: Output, action: &str) -> Result<Output> {
    if output.status.success() {
        return Ok(output);
    }
    bail!(
        "{action} failed: {}",
        String::from_utf8_lossy(&output.stderr).trim()
    )
}

fn git_text<const N: usize>(root: &Path, args: [&str; N]) -> Result<String> {
    let output = git_output(root, args)?;
    Ok(String::from_utf8(output.stdout)
        .context("Git returned non-UTF-8 text")?
        .trim()
        .to_owned())
}

fn stage_candidate(root: &Path) -> Result<()> {
    let output = Command::new("git")
        .arg("-C")
        .arg(root)
        .args(["add", "-A", "-f", "--", "."])
        .args(GENERATED_EXCLUDES)
        .output()
        .context("staging detached candidate")?;
    require_git_success(output, "git add in detached candidate")?;
    Ok(())
}

fn changed_paths(root: &Path, base: &str) -> Result<Vec<PathBuf>> {
    let output = Command::new("git")
        .arg("-C")
        .arg(root)
        .args([
            "diff",
            "--cached",
            "--relative",
            "--no-renames",
            "--name-only",
            "-z",
            base,
            "--",
            ".",
        ])
        .args(GENERATED_EXCLUDES)
        .output()
        .context("listing detached candidate changes")?;
    let output = require_git_success(output, "git diff in detached candidate")?;
    output
        .stdout
        .split(|byte| *byte == 0)
        .filter(|field| !field.is_empty())
        .map(path_from_git_bytes)
        .collect()
}

#[cfg(unix)]
fn path_from_git_bytes(bytes: &[u8]) -> Result<PathBuf> {
    use std::os::unix::ffi::OsStringExt;
    Ok(PathBuf::from(std::ffi::OsString::from_vec(bytes.to_vec())))
}

#[cfg(not(unix))]
fn path_from_git_bytes(bytes: &[u8]) -> Result<PathBuf> {
    Ok(PathBuf::from(String::from_utf8(bytes.to_vec()).context(
        "candidate path is not valid UTF-8 on this platform",
    )?))
}

fn baseline_file(
    root: &Path,
    base: &str,
    path: &Path,
) -> Result<Option<(CandidateFileKind, u32, Vec<u8>)>> {
    let output = Command::new("git")
        .arg("-C")
        .arg(root)
        .args(["ls-tree", "-z", base, "--"])
        .arg(path)
        .output()
        .context("reading detached candidate baseline entry")?;
    let output = require_git_success(output, "git ls-tree in detached candidate")?;
    if output.stdout.is_empty() {
        return Ok(None);
    }
    let header = output
        .stdout
        .split(|byte| *byte == b'\t')
        .next()
        .context("candidate baseline entry omitted metadata")?;
    let fields = std::str::from_utf8(header)
        .context("candidate baseline metadata was not UTF-8")?
        .split_whitespace()
        .collect::<Vec<_>>();
    ensure!(
        fields.len() == 3,
        "candidate baseline metadata is malformed"
    );
    let mode = u32::from_str_radix(fields[0], 8).context("parsing candidate baseline mode")?;
    let kind = match mode {
        0o120000 => CandidateFileKind::Symlink,
        0o100000..=0o100777 => CandidateFileKind::Regular,
        _ => bail!("candidate baseline contains unsupported Git mode {mode:o}"),
    };
    let blob = Command::new("git")
        .arg("-C")
        .arg(root)
        .args(["cat-file", "blob", fields[2]])
        .output()
        .context("reading detached candidate baseline blob")?;
    let blob = require_git_success(blob, "git cat-file in detached candidate")?;
    Ok(Some((kind, mode & 0o7777, blob.stdout)))
}

fn candidate_file(root: &Path, path: &Path) -> Result<Option<(CandidateFileKind, u32, Vec<u8>)>> {
    let absolute = root.join(path);
    let metadata = match fs::symlink_metadata(&absolute) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error.into()),
    };
    let mode = file_mode(&metadata);
    if metadata.is_file() {
        return Ok(Some((
            CandidateFileKind::Regular,
            mode,
            fs::read(absolute)?,
        )));
    }
    if metadata.file_type().is_symlink() {
        let target = fs::read_link(absolute)?;
        return Ok(Some((
            CandidateFileKind::Symlink,
            mode,
            target.to_string_lossy().as_bytes().to_vec(),
        )));
    }
    bail!("candidate contains unsupported path {}", path.display())
}

#[cfg(unix)]
fn file_mode(metadata: &fs::Metadata) -> u32 {
    use std::os::unix::fs::PermissionsExt;
    metadata.permissions().mode() & 0o7777
}

#[cfg(not(unix))]
fn file_mode(_metadata: &fs::Metadata) -> u32 {
    0o644
}

fn content_digest(bytes: &[u8]) -> String {
    format!("blake3:{}", blake3::hash(bytes).to_hex())
}

fn absolute_path(path: &Path) -> Result<PathBuf> {
    if path.is_absolute() {
        return Ok(path.to_path_buf());
    }
    Ok(std::env::current_dir()
        .context("resolving current directory")?
        .join(path))
}

fn remove_owned_directory(owner: &Path, path: &Path) -> Result<()> {
    ensure!(
        path.parent() == Some(owner) && path != owner,
        "refusing to remove a directory outside the candidate owner"
    );
    let metadata = fs::symlink_metadata(path)
        .with_context(|| format!("inspecting owned candidate path {}", path.display()))?;
    ensure!(
        metadata.is_dir() && !metadata.file_type().is_symlink(),
        "owned candidate path is not a real directory"
    );
    fs::remove_dir_all(path)
        .with_context(|| format!("removing owned candidate path {}", path.display()))
}

fn remove_owned_root(path: &Path) -> Result<()> {
    let metadata = match fs::symlink_metadata(path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(error) => return Err(error.into()),
    };
    ensure!(
        metadata.is_dir() && !metadata.file_type().is_symlink(),
        "refusing to recursively remove a non-directory candidate owner"
    );
    fs::remove_dir_all(path).with_context(|| format!("removing candidate owner {}", path.display()))
}

#[cfg(test)]
#[path = "candidate_workspace_tests.rs"]
mod tests;
