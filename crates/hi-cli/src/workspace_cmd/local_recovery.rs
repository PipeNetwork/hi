//! Credential-free inspection and disposition of interrupted local lifecycles.

use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::Read;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail, ensure};
use hi_control::{
    ControlEffectScope, ControlJobKind, ControlJobRecord, ControlStore, WorkspaceAuthority,
    WorkspaceBindingRecord, WorkspaceOperationRecord, WorkspaceOperationStatus,
    WorkspaceProjectionJournal, WorkspaceRecoveryRecord, WorkspaceRecoveryStatus,
};
use hi_workspace::{
    BindingId, JobId, OperationId, RecoveryId, restart_job_recovery_id,
    restart_operation_recovery_id,
};
use ignore::WalkBuilder;
use serde::Serialize;

use super::{RecoveryCommand, WorkspaceCommand};

const SCANNER_VERSION: u16 = 1;
const MAX_ENTRIES: u64 = 200_000;
const MAX_BYTES: u64 = 2 * 1024 * 1024 * 1024;
const RESOLUTION: &str = "Stop every external writer, inspect the current workspace bytes (including VCS metadata), then run `hi workspace recover discard RECOVERY_ID --confirm DIGEST --accept-current-bytes`. This preserves the complete current workspace unchanged and marks only the interrupted lifecycle Failed; it does not infer process reaping, success, cancellation, or rollback.";
const SETTLED_RESOLUTION: &str = "Stop every external writer, inspect the current workspace bytes (including VCS metadata), then run `hi workspace recover discard RECOVERY_ID --confirm DIGEST --accept-current-bytes`. This preserves both the complete current workspace and the already-terminal lifecycle status while resolving only the stale recovery fence.";

#[derive(Clone, Debug, Serialize)]
struct WorkspaceProof {
    scanner_version: u16,
    workspace_digest: Option<String>,
    confirmation_digest: Option<String>,
    entry_count: Option<u64>,
    byte_count: Option<u64>,
    exclusions: Vec<String>,
    scan_error: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
struct LocalRecoveryView {
    schema_version: u16,
    recovery_id: String,
    legacy_recovery_ids: Vec<String>,
    workspace_id: String,
    workspace_root: String,
    state_root: String,
    binding: WorkspaceBindingRecord,
    operation: Option<WorkspaceOperationRecord>,
    job: Option<ControlJobRecord>,
    journal_evidence: Vec<WorkspaceRecoveryRecord>,
    proof: WorkspaceProof,
    retry_safe: bool,
    process_reaping_proven: bool,
    safe_resolution: String,
}

#[derive(Debug, Serialize)]
struct LocalRecoveryList {
    schema_version: u16,
    authority: &'static str,
    workspace_id: String,
    workspace_root: String,
    state_root: String,
    recovery_required: bool,
    recoveries: Vec<LocalRecoveryView>,
}

#[derive(Clone, Debug)]
struct Scan {
    digest: String,
    entry_count: u64,
    byte_count: u64,
    exclusions: Vec<String>,
}

#[derive(Clone)]
struct Target {
    recovery_id: RecoveryId,
    binding: WorkspaceBindingRecord,
    operation: Option<WorkspaceOperationRecord>,
    job: Option<ControlJobRecord>,
    evidence: Vec<WorkspaceRecoveryRecord>,
}

struct LocalRecoveryService {
    workspace_root: PathBuf,
    state_root: PathBuf,
    workspace_id: String,
}

pub(super) async fn run(command: WorkspaceCommand) -> Result<()> {
    let (workspace_root, state_root) = crate::review_target::resolve_runtime_roots()?;
    run_at(command, workspace_root, state_root)
}

fn run_at(command: WorkspaceCommand, workspace_root: PathBuf, state_root: PathBuf) -> Result<()> {
    let service = LocalRecoveryService::new(workspace_root, state_root);
    match command {
        WorkspaceCommand::Status(args) => service.print_list(args.json, true),
        WorkspaceCommand::Recover { command } => match command {
            RecoveryCommand::List(args) => service.print_list(args.json, false),
            RecoveryCommand::Inspect(args) => service.print_inspect(&args.recovery_id, args.json),
            RecoveryCommand::Retry(_) => bail!(
                "local recovery retry is unsafe: persisted state cannot prove the old writer was reaped, so hi will not replay or infer its outcome; use recover inspect and the content-confirmed discard path"
            ),
            RecoveryCommand::Export(_) => bail!(
                "local recovery bytes are already in the current workspace; inspect them in place and use the content-confirmed discard path"
            ),
            RecoveryCommand::Discard(args) => {
                ensure!(
                    args.accept_current_bytes,
                    "local recovery discard requires --accept-current-bytes to acknowledge that all external writers are stopped and the complete current workspace bytes are accepted unchanged"
                );
                let receipt = service.discard(&args.recovery_id, &args.confirm)?;
                if receipt.lifecycle_marked_failed {
                    println!(
                        "Local recovery {} discarded after whole-workspace confirmation {}; current bytes were preserved unchanged and the interrupted lifecycle is Failed. Process reaping, success, cancellation, and rollback were not inferred.",
                        receipt.recovery_id, receipt.confirmation_digest
                    );
                } else if receipt.operation_id.is_none() && receipt.job_id.is_none() {
                    println!(
                        "Local recovery {} discarded after whole-workspace confirmation {}; current bytes were preserved unchanged and no lifecycle status was changed.",
                        receipt.recovery_id, receipt.confirmation_digest
                    );
                } else {
                    println!(
                        "Local recovery {} discarded after whole-workspace confirmation {}; current bytes and the already-terminal lifecycle status were preserved unchanged.",
                        receipt.recovery_id, receipt.confirmation_digest
                    );
                }
                Ok(())
            }
        },
        _ => bail!("this workspace command requires --session ID"),
    }
}

impl LocalRecoveryService {
    fn new(workspace_root: PathBuf, state_root: PathBuf) -> Self {
        let workspace_id = workspace_id(&workspace_root);
        Self {
            workspace_root,
            state_root,
            workspace_id,
        }
    }

    fn print_list(&self, json: bool, status: bool) -> Result<()> {
        let recoveries = self.inventory()?;
        let view = LocalRecoveryList {
            schema_version: 1,
            authority: "local",
            workspace_id: self.workspace_id.clone(),
            workspace_root: self.workspace_root.display().to_string(),
            state_root: self.state_root.display().to_string(),
            recovery_required: !recoveries.is_empty(),
            recoveries,
        };
        if json {
            return super::print_json(&view);
        }
        if status {
            println!("Local workspace status for {}", view.workspace_root);
            println!(
                "recovery required: {}",
                super::yes_no(view.recovery_required)
            );
        } else if view.recoveries.is_empty() {
            println!("Local workspace: no unresolved restart recoveries");
            return Ok(());
        } else {
            println!("Local workspace restart recoveries:");
        }
        for recovery in &view.recoveries {
            print_summary(recovery);
        }
        Ok(())
    }

    fn print_inspect(&self, requested_id: &str, json: bool) -> Result<()> {
        let view = self.find(requested_id)?;
        if json {
            return super::print_json(&view);
        }
        println!("Local workspace recovery {}", view.recovery_id);
        println!("workspace: {}", view.workspace_id);
        println!(
            "binding: {} at epoch {} ({:?}, version {})",
            view.binding.binding_id,
            view.binding.epoch,
            view.binding.state,
            view.binding.workspace_version.as_deref().unwrap_or("none")
        );
        if let Some(operation) = &view.operation {
            println!(
                "operation: {} ({:?}/{:?}, digest {}, idempotency {})",
                operation.operation_id,
                operation.status,
                operation.replay_class,
                operation.operation_digest,
                operation.idempotency_key
            );
        }
        if let Some(job) = &view.job {
            println!(
                "job: {} ({:?}, {:?}/{:?}, artifact {})",
                job.job_id,
                job.state,
                job.kind,
                job.effect_scope,
                job.candidate_ref.as_deref().unwrap_or("none")
            );
        }
        println!(
            "whole-workspace/current-bytes confirmation: {}",
            view.proof
                .confirmation_digest
                .as_deref()
                .unwrap_or("unavailable")
        );
        println!(
            "workspace scan exclusions: {}",
            if view.proof.exclusions.is_empty() {
                "none".to_owned()
            } else {
                view.proof.exclusions.join(", ")
            }
        );
        println!("process reaping proven: no");
        if !view.legacy_recovery_ids.is_empty() {
            println!(
                "legacy journal aliases: {}",
                view.legacy_recovery_ids.join(", ")
            );
        }
        for evidence in &view.journal_evidence {
            println!(
                "journal evidence: {} ({:?}, kind {}, digest {}, artifact {})",
                evidence.recovery_id,
                evidence.status,
                evidence.kind,
                evidence.digest.as_deref().unwrap_or("none"),
                evidence.artifact_ref.as_deref().unwrap_or("none")
            );
        }
        println!("safe resolution: {}", view.safe_resolution);
        if let Some(error) = &view.proof.scan_error {
            println!("scan error: {error}");
        }
        Ok(())
    }

    fn inventory(&self) -> Result<Vec<LocalRecoveryView>> {
        let Some(store) = self.open_store()? else {
            return Ok(Vec::new());
        };
        let targets = self.targets(&store)?;
        if targets.is_empty() {
            return Ok(Vec::new());
        }
        let scan = scan_workspace(&self.workspace_root, &self.state_root);
        Ok(targets
            .into_values()
            .map(|target| self.view(target, scan.as_ref()))
            .collect())
    }

    fn find(&self, requested_id: &str) -> Result<LocalRecoveryView> {
        let mut matches = self.inventory()?.into_iter().filter(|view| {
            view.recovery_id == requested_id
                || view.legacy_recovery_ids.iter().any(|id| id == requested_id)
        });
        let found = matches.next().ok_or_else(|| {
            anyhow::anyhow!("local recovery {requested_id:?} was not found in this workspace")
        })?;
        ensure!(
            matches.next().is_none(),
            "local recovery alias {requested_id:?} is ambiguous"
        );
        Ok(found)
    }

    fn discard(
        &self,
        requested_id: &str,
        confirmation: &str,
    ) -> Result<hi_control::LocalRecoveryDiscardReceipt> {
        let initial = self.find(requested_id)?;
        ensure!(
            initial.proof.confirmation_digest.as_deref() == Some(confirmation),
            "local recovery confirmation does not match the freshly scanned complete current workspace"
        );
        let store = self
            .open_store()?
            .ok_or_else(|| anyhow::anyhow!("local recovery journal disappeared"))?;
        let targets = self.targets(&store)?;
        let stable_id = RecoveryId::new(initial.recovery_id.clone());
        let target = targets.get(&stable_id).ok_or_else(|| {
            anyhow::anyhow!("local recovery changed while validating current workspace bytes")
        })?;
        let fresh_scan = scan_workspace(&self.workspace_root, &self.state_root)?;
        let fresh_confirmation =
            confirmation_digest(&self.workspace_id, target, &fresh_scan.digest);
        ensure!(
            fresh_confirmation == confirmation,
            "workspace bytes or recovery evidence changed during confirmation; inspect again"
        );
        WorkspaceProjectionJournal::from_control_store(store)
            .discard_local_restart_recovery(
                &self.workspace_id,
                &target.binding.binding_id,
                &target.recovery_id,
                target
                    .operation
                    .as_ref()
                    .map(|value| value.operation_id.as_str()),
                target.job.as_ref().map(|value| value.job_id.as_str()),
                confirmation,
            )
            .map_err(Into::into)
    }

    fn open_store(&self) -> Result<Option<ControlStore>> {
        let database = self.state_root.join("events.sqlite3");
        let metadata = match fs::symlink_metadata(&database) {
            Ok(metadata) => metadata,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(error.into()),
        };
        ensure!(
            metadata.is_file() && !metadata.file_type().is_symlink(),
            "local control journal is not a regular file: {}",
            database.display()
        );
        ControlStore::open_for_state(&self.state_root)
            .map(Some)
            .context("opening the local workspace control journal")
    }

    fn targets(&self, store: &ControlStore) -> Result<BTreeMap<RecoveryId, Target>> {
        let mut targets = BTreeMap::new();
        for binding in store.unsettled_workspace_bindings(&self.workspace_id)? {
            for operation in store.operations_for_binding(&binding.binding_id)? {
                if operation_requires_recovery(operation.status) {
                    insert_operation(&mut targets, &binding, operation);
                }
            }
            for job in store.jobs_for_binding(&binding.binding_id)? {
                if !job.state.is_terminal() && is_writer(&job) {
                    insert_job(&mut targets, &binding, job);
                }
            }
        }
        for recovery in store.recoveries_for_workspace(&self.workspace_id)? {
            if recovery.session_id.is_some()
                || matches!(
                    recovery.status,
                    WorkspaceRecoveryStatus::Resolved | WorkspaceRecoveryStatus::Discarded
                )
            {
                continue;
            }
            let Some(binding_id) = recovery.binding_id.as_deref() else {
                continue;
            };
            let Some(binding) = store.get_workspace_binding(binding_id)? else {
                continue;
            };
            if binding.authority != WorkspaceAuthority::Local
                || binding.workspace_id != self.workspace_id
            {
                continue;
            }
            let stable_id = if let Some(operation_id) = recovery.operation_id.as_deref() {
                let Some(operation) = store.get_workspace_operation(operation_id)? else {
                    continue;
                };
                insert_operation(&mut targets, &binding, operation)
            } else if let Some(job_id) = recovery.job_id.as_deref() {
                let Some(job) = store.get_job(job_id)? else {
                    continue;
                };
                if !is_writer(&job) {
                    continue;
                }
                insert_job(&mut targets, &binding, job)
            } else {
                let id = RecoveryId::new(recovery.recovery_id.clone());
                targets.entry(id.clone()).or_insert(Target {
                    recovery_id: id.clone(),
                    binding: binding.clone(),
                    operation: None,
                    job: None,
                    evidence: Vec::new(),
                });
                id
            };
            if let Some(target) = targets.get_mut(&stable_id) {
                target.evidence.push(recovery);
            }
        }
        for target in targets.values_mut() {
            target
                .evidence
                .sort_by(|left, right| left.recovery_id.cmp(&right.recovery_id));
            target
                .evidence
                .dedup_by(|left, right| left.recovery_id == right.recovery_id);
        }
        Ok(targets)
    }

    fn view(&self, target: Target, scan: Result<&Scan, &anyhow::Error>) -> LocalRecoveryView {
        let (digest, entry_count, byte_count, exclusions, error) = match scan {
            Ok(scan) => (
                Some(scan.digest.clone()),
                Some(scan.entry_count),
                Some(scan.byte_count),
                scan.exclusions.clone(),
                None,
            ),
            Err(error) => (
                None,
                None,
                None,
                exclusions(&self.workspace_root, &self.state_root),
                Some(format!("{error:#}")),
            ),
        };
        let confirmation = digest
            .as_deref()
            .map(|digest| confirmation_digest(&self.workspace_id, &target, digest));
        let legacy_recovery_ids = target
            .evidence
            .iter()
            .filter(|record| record.recovery_id != target.recovery_id.as_str())
            .map(|record| record.recovery_id.clone())
            .collect();
        let already_terminal = target
            .operation
            .as_ref()
            .is_some_and(|operation| !operation_requires_recovery(operation.status))
            || target
                .job
                .as_ref()
                .is_some_and(|job| job.state.is_terminal())
            || (target.operation.is_none() && target.job.is_none());
        LocalRecoveryView {
            schema_version: 1,
            recovery_id: target.recovery_id.to_string(),
            legacy_recovery_ids,
            workspace_id: self.workspace_id.clone(),
            workspace_root: self.workspace_root.display().to_string(),
            state_root: self.state_root.display().to_string(),
            binding: target.binding,
            operation: target.operation,
            job: target.job,
            journal_evidence: target.evidence,
            proof: WorkspaceProof {
                scanner_version: SCANNER_VERSION,
                workspace_digest: digest,
                confirmation_digest: confirmation,
                entry_count,
                byte_count,
                exclusions,
                scan_error: error,
            },
            retry_safe: false,
            process_reaping_proven: false,
            safe_resolution: if already_terminal {
                SETTLED_RESOLUTION
            } else {
                RESOLUTION
            }
            .to_owned(),
        }
    }
}

fn insert_operation(
    targets: &mut BTreeMap<RecoveryId, Target>,
    binding: &WorkspaceBindingRecord,
    operation: WorkspaceOperationRecord,
) -> RecoveryId {
    let id = restart_operation_recovery_id(
        &BindingId::new(binding.binding_id.clone()),
        binding.epoch,
        &OperationId::new(operation.operation_id.clone()),
    );
    targets.entry(id.clone()).or_insert(Target {
        recovery_id: id.clone(),
        binding: binding.clone(),
        operation: Some(operation),
        job: None,
        evidence: Vec::new(),
    });
    id
}

fn insert_job(
    targets: &mut BTreeMap<RecoveryId, Target>,
    binding: &WorkspaceBindingRecord,
    job: ControlJobRecord,
) -> RecoveryId {
    let id = restart_job_recovery_id(
        &BindingId::new(binding.binding_id.clone()),
        binding.epoch,
        &JobId::new(job.job_id.clone()),
    );
    targets.entry(id.clone()).or_insert(Target {
        recovery_id: id.clone(),
        binding: binding.clone(),
        operation: None,
        job: Some(job),
        evidence: Vec::new(),
    });
    id
}

fn operation_requires_recovery(status: WorkspaceOperationStatus) -> bool {
    !matches!(
        status,
        WorkspaceOperationStatus::Durable
            | WorkspaceOperationStatus::NoChange
            | WorkspaceOperationStatus::LocalAuditDegraded
            | WorkspaceOperationStatus::Failed
    )
}

fn is_writer(job: &ControlJobRecord) -> bool {
    matches!(
        job.effect_scope,
        ControlEffectScope::CandidateOnly | ControlEffectScope::LiveWriter
    ) || job.kind == ControlJobKind::WriteCandidate
}

fn workspace_id(root: &Path) -> String {
    uuid::Uuid::new_v5(
        &uuid::Uuid::NAMESPACE_URL,
        root.to_string_lossy().as_bytes(),
    )
    .to_string()
}

fn confirmation_digest(workspace_id: &str, target: &Target, workspace_digest: &str) -> String {
    let mut hasher = blake3::Hasher::new_derive_key("hi.local-recovery-confirmation.v1");
    hash_field(&mut hasher, workspace_id.as_bytes());
    hash_field(&mut hasher, target.recovery_id.as_str().as_bytes());
    hash_field(&mut hasher, target.binding.binding_id.as_bytes());
    hash_field(&mut hasher, target.binding.epoch.to_string().as_bytes());
    hash_field(
        &mut hasher,
        target
            .operation
            .as_ref()
            .map(|value| value.operation_id.as_bytes())
            .or_else(|| target.job.as_ref().map(|value| value.job_id.as_bytes()))
            .unwrap_or_default(),
    );
    hash_field(&mut hasher, workspace_digest.as_bytes());
    format!("blake3:{}", hasher.finalize().to_hex())
}

fn scan_workspace(workspace_root: &Path, state_root: &Path) -> Result<Scan> {
    let excluded_state = state_root
        .starts_with(workspace_root)
        .then(|| state_root.to_path_buf());
    let excluded_for_filter = excluded_state.clone();
    let mut builder = WalkBuilder::new(workspace_root);
    builder
        .hidden(false)
        .ignore(false)
        .git_ignore(false)
        .git_global(false)
        .git_exclude(false)
        .follow_links(false)
        .filter_entry(move |entry| {
            excluded_for_filter
                .as_ref()
                .is_none_or(|state| entry.path() != state)
        });
    let mut paths = builder
        .build()
        .filter_map(|entry| match entry {
            Ok(entry) if entry.depth() > 0 => Some(Ok(entry.into_path())),
            Ok(_) => None,
            Err(error) => Some(Err(anyhow::anyhow!(error))),
        })
        .collect::<Result<Vec<_>>>()?;
    paths.sort_by_key(|path| path_bytes(path.strip_prefix(workspace_root).unwrap_or(path)));
    ensure!(
        paths.len() as u64 <= MAX_ENTRIES,
        "workspace scan exceeds {MAX_ENTRIES} entries"
    );
    let mut hasher = blake3::Hasher::new_derive_key("hi.local-workspace-scan.v1");
    let mut byte_count = 0_u64;
    for path in &paths {
        let relative = path.strip_prefix(workspace_root)?;
        let metadata = fs::symlink_metadata(path)?;
        hash_field(&mut hasher, path_bytes(relative).as_slice());
        hash_field(&mut hasher, &mode(&metadata).to_le_bytes());
        if metadata.file_type().is_symlink() {
            hash_field(&mut hasher, b"symlink");
            hash_field(&mut hasher, path_bytes(&fs::read_link(path)?).as_slice());
        } else if metadata.is_dir() {
            hash_field(&mut hasher, b"directory");
        } else if metadata.is_file() {
            hash_field(&mut hasher, b"file");
            hash_field(&mut hasher, &metadata.len().to_le_bytes());
            byte_count = byte_count
                .checked_add(metadata.len())
                .context("workspace byte count overflow")?;
            ensure!(
                byte_count <= MAX_BYTES,
                "workspace scan exceeds {MAX_BYTES} bytes"
            );
            let mut file = open_regular(path)?;
            ensure!(
                file.metadata()?.is_file(),
                "workspace entry changed type during scan: {}",
                path.display()
            );
            let mut buffer = [0_u8; 64 * 1024];
            let mut actual_len = 0_u64;
            loop {
                let read = file.read(&mut buffer)?;
                if read == 0 {
                    break;
                }
                actual_len = actual_len
                    .checked_add(read as u64)
                    .context("workspace file length overflow")?;
                hasher.update(&buffer[..read]);
            }
            ensure!(
                actual_len == metadata.len(),
                "workspace file changed length during scan: {}",
                path.display()
            );
        } else {
            bail!("unsupported workspace entry type: {}", path.display());
        }
    }
    Ok(Scan {
        digest: format!("blake3:{}", hasher.finalize().to_hex()),
        entry_count: paths.len() as u64,
        byte_count,
        exclusions: exclusions(workspace_root, state_root),
    })
}

#[cfg(unix)]
fn open_regular(path: &Path) -> std::io::Result<File> {
    use std::os::unix::fs::OpenOptionsExt;
    fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC)
        .open(path)
}

#[cfg(not(unix))]
fn open_regular(path: &Path) -> std::io::Result<File> {
    File::open(path)
}

fn exclusions(workspace_root: &Path, state_root: &Path) -> Vec<String> {
    let mut values = Vec::new();
    if state_root.starts_with(workspace_root) {
        values.push(format!("runtime state {}", state_root.display()));
    }
    values
}

fn hash_field(hasher: &mut blake3::Hasher, bytes: &[u8]) {
    hasher.update(&(bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
}

#[cfg(unix)]
fn path_bytes(path: &Path) -> Vec<u8> {
    use std::os::unix::ffi::OsStrExt;
    path.as_os_str().as_bytes().to_vec()
}

#[cfg(not(unix))]
fn path_bytes(path: &Path) -> Vec<u8> {
    path.to_string_lossy().as_bytes().to_vec()
}

#[cfg(unix)]
fn mode(metadata: &fs::Metadata) -> u32 {
    use std::os::unix::fs::MetadataExt;
    metadata.mode()
}

#[cfg(not(unix))]
fn mode(_: &fs::Metadata) -> u32 {
    0
}

fn print_summary(recovery: &LocalRecoveryView) {
    let target = if let Some(operation) = &recovery.operation {
        format!("operation {}", operation.operation_id)
    } else if let Some(job) = &recovery.job {
        format!("job {}", job.job_id)
    } else {
        "binding-level recovery".to_owned()
    };
    println!(
        "- {}: binding {} epoch {}, {}; whole-workspace confirmation {}",
        recovery.recovery_id,
        recovery.binding.binding_id,
        recovery.binding.epoch,
        target,
        recovery
            .proof
            .confirmation_digest
            .as_deref()
            .unwrap_or("unavailable")
    );
    println!("  safe resolution: {}", recovery.safe_resolution);
}

#[cfg(test)]
#[path = "local_recovery_tests.rs"]
mod tests;
