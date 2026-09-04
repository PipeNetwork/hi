//! Host integration for the materialized PipeFS workspace.

use std::path::PathBuf;
use std::sync::{Arc, Mutex, RwLock};
use std::{collections::HashMap, time::Duration};

use anyhow::{Context, Result, anyhow, bail, ensure};

use crate::sync::{RemoteSessionSink, SyncConfig};

pub(crate) type SharedSyncHandle = Arc<Mutex<Option<Arc<RemoteSessionSink>>>>;

#[derive(Clone)]
pub(crate) struct PipeFsMcpConfig {
    import_policy: hi_mcp::McpImportPolicy,
    pipe_attach: Option<crate::mcp_host::PipeAttach>,
    server_policies: HashMap<String, hi_mcp::ServerAllowList>,
}

impl PipeFsMcpConfig {
    pub(crate) fn resolve(
        settings: &crate::config::Settings,
        config: &crate::config::Config,
    ) -> Self {
        Self {
            import_policy: config.mcp_import.to_policy(),
            pipe_attach: crate::mcp_host::decide_pipe_attach(
                settings.mcp_pipe_enabled,
                settings.mcp_url.as_deref(),
                &settings.api_key,
                settings.mcp_pipe_allow.clone(),
            )
            .ok(),
            server_policies: config.mcp.server_allowlists(),
        }
    }
}

#[derive(Clone)]
pub(crate) struct PipeFsHost {
    sync_config: SyncConfig,
    session_id: Arc<RwLock<String>>,
    session_path: Arc<RwLock<PathBuf>>,
    sync_handle: SharedSyncHandle,
    original_workspace_root: PathBuf,
    original_state_root: PathBuf,
    mcp: PipeFsMcpConfig,
    active: Arc<tokio::sync::Mutex<Option<hi_pipefs::PipeFsWorkspace>>>,
    active_durability: Arc<tokio::sync::Mutex<Option<Arc<PipeFsDurability>>>>,
    cleanup_pending: Arc<tokio::sync::Mutex<Option<hi_pipefs::PipeFsWorkspace>>>,
}

impl PipeFsHost {
    pub(crate) fn new(
        sync_config: SyncConfig,
        session_id: String,
        session_path: PathBuf,
        sync_handle: SharedSyncHandle,
        original_workspace_root: PathBuf,
        original_state_root: PathBuf,
        mcp: PipeFsMcpConfig,
    ) -> Result<Self> {
        crate::sync::validate_session_id(&session_id)?;
        Ok(Self {
            sync_config,
            session_id: Arc::new(RwLock::new(session_id)),
            session_path: Arc::new(RwLock::new(session_path)),
            sync_handle,
            original_workspace_root,
            original_state_root,
            mcp,
            active: Arc::new(tokio::sync::Mutex::new(None)),
            active_durability: Arc::new(tokio::sync::Mutex::new(None)),
            cleanup_pending: Arc::new(tokio::sync::Mutex::new(None)),
        })
    }

    fn client(&self) -> Result<hi_pipefs::PipeFsClient> {
        hi_pipefs::PipeFsClient::new(hi_pipefs::PipeFsClientConfig::new(
            self.sync_config.base_url.clone(),
            self.sync_config.api_key.clone(),
        ))
        .map_err(Into::into)
    }

    fn session_id(&self) -> String {
        self.session_id
            .read()
            .unwrap_or_else(|error| error.into_inner())
            .clone()
    }

    fn session_path(&self) -> PathBuf {
        self.session_path
            .read()
            .unwrap_or_else(|error| error.into_inner())
            .clone()
    }

    pub(crate) fn sync_handle(&self) -> Option<Arc<RemoteSessionSink>> {
        self.sync_handle
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .clone()
    }

    fn require_sync_handle(&self) -> Result<Arc<RemoteSessionSink>> {
        let sync = self.sync_handle().ok_or_else(|| {
            anyhow!("PipeFS has no active transcript sync session; run /pipefs on to recover")
        })?;
        ensure!(
            sync.session_id() == self.session_id(),
            "PipeFS session identity does not match the active transcript session"
        );
        Ok(sync)
    }

    /// Upgrade an ordinary saved JSONL session to the existing transcript-sync
    /// transport. This is deliberately lazy so PipeFS remains opt-in even when
    /// global sync is off, while `/pipefs on` can still establish the one shared
    /// session identity required for its writer lease.
    async fn ensure_sync(&self, agent: &mut hi_agent::Agent) -> Result<Arc<RemoteSessionSink>> {
        crate::sync_store::SyncStore::open()
            .context("opening transcript sync storage for PipeFS")?
            .set_mode(crate::sync_store::SyncMode::On)
            .context("enabling transcript sync for PipeFS")?;
        if let Some(sync) = self.sync_handle() {
            ensure!(
                sync.session_id() == self.session_id(),
                "PipeFS session identity does not match the active transcript session"
            );
            return Ok(sync);
        }

        let path = self.session_path();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).with_context(|| {
                format!("creating saved session directory {}", parent.display())
            })?;
        }
        std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .with_context(|| format!("creating saved session {}", path.display()))?;

        let remote = RemoteSessionSink::new(self.sync_config.clone(), self.session_id());
        let session =
            crate::sync::SyncSession::new(crate::session::JsonlSession::new(path), remote);
        let sync = session.remote_handle();
        agent.set_session(Box::new(session));
        self.sync_handle
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .replace(sync.clone());
        Ok(sync)
    }

    async fn lease(&self, sync: &RemoteSessionSink) -> Result<hi_pipefs::PipeFsLease> {
        let session_id = self.session_id();
        ensure!(
            sync.session_id() == session_id.as_str(),
            "PipeFS session identity does not match the active transcript session"
        );
        sync.ensure_registered_now()
            .await
            .context("registering the HI session before PipeFS activation")?;
        let token = sync.writer_lease_token().ok_or_else(|| {
            anyhow!("the IPOP server did not issue a writer lease; PipeFS requires lease support")
        })?;
        let generation = sync.writer_lease_generation();
        ensure!(generation > 0, "the IPOP writer lease has no generation");
        Ok(hi_pipefs::PipeFsLease { token, generation })
    }

    fn workspace(
        &self,
        client: hi_pipefs::PipeFsClient,
        lease: hi_pipefs::PipeFsLease,
    ) -> Result<hi_pipefs::PipeFsWorkspace> {
        hi_pipefs::PipeFsWorkspace::new(
            client,
            lease,
            hi_pipefs::PipeFsWorkspaceConfig {
                session_id: self.session_id(),
                original_workspace_root: self.original_workspace_root.clone(),
                original_state_root: self.original_state_root.clone(),
                cache_base: None,
            },
        )
    }

    async fn reconnect_workspace_mcp(
        &self,
        agent: &mut hi_agent::Agent,
        root: &std::path::Path,
        allow_workspace_trust: bool,
    ) {
        let (mcp, _) = crate::mcp_host::connect_workspace_mcp_with_trust(
            root,
            &self.mcp.import_policy,
            self.mcp.pipe_attach.as_ref(),
            allow_workspace_trust,
            &self.mcp.server_policies,
        )
        .await;
        if let Some(mcp) = mcp {
            agent.attach_mcp(mcp);
        }
    }

    /// For an existing session, the server's mode wins over CLI/config. For a
    /// new session, the caller's explicit/configured default decides.
    pub(crate) async fn activate_for_startup(
        &self,
        agent: &mut hi_agent::Agent,
        requested_for_new_session: bool,
        must_resolve_remote_pipefs: bool,
    ) -> Result<bool> {
        let client = self.client()?;
        let session_id = self.session_id();
        let local_recovery_required = hi_pipefs::local_recovery_required(&session_id);
        let must_resolve_remote_pipefs = must_resolve_remote_pipefs || local_recovery_required;
        let remote = client.state(&session_id).await;
        let (remote_is_authoritative, remote_enabled, remote_was_enabled) = match remote {
            Ok(state) => (true, Some(state.enabled), state.enabled),
            Err(error)
                if must_resolve_remote_pipefs
                    && matches!(&error, hi_pipefs::PipeFsError::MissingRevision(_)) =>
            {
                return Err(anyhow!(error).context(
                    "the existing remote session's PipeFS state is unavailable; refusing to continue in the launch directory",
                ));
            }
            Err(error)
                if must_resolve_remote_pipefs
                    && matches!(&error, hi_pipefs::PipeFsError::Disabled(_)) =>
            {
                return Err(anyhow!(error).context(
                    "the saved session is known to use PipeFS, but its remote state is unavailable; refusing to continue in the launch directory",
                ));
            }
            Err(
                hi_pipefs::PipeFsError::MissingRevision(_) | hi_pipefs::PipeFsError::Disabled(_),
            ) => (false, None, false),
            Err(error) => return Err(error.into()),
        };
        let activate = local_recovery_required
            || effective_startup_mode(
                remote_is_authoritative,
                remote_enabled,
                requested_for_new_session,
            );
        if !activate {
            if remote_is_authoritative && remote_enabled == Some(false) && !local_recovery_required
            {
                let _ = hi_pipefs::record_local_mode_hint(&session_id, false);
            }
            // Startup candidates were intentionally constructed without LSP,
            // project hooks, or repository MCP. Once the remote authority says
            // PipeFS is off, activate the deferred integrations in place and
            // only then admit trusted repository integrations. Rebinding the
            // same launch root would reset loaded task/checkpoint state.
            agent.activate_deferred_local_workspace_runtime();
            agent.set_workspace_project_context(crate::project_context::load_project_context_from(
                &self.original_workspace_root,
            ));
            self.reconnect_workspace_mcp(agent, &self.original_workspace_root, true)
                .await;
            return Ok(false);
        }
        let sync = self.ensure_sync(agent).await?;
        sync.set_pipefs_sync_required(true);
        let lease = self.lease(&sync).await?;
        let workspace = self.workspace(client, lease)?;
        self.activate(agent, workspace, sync, remote_was_enabled)
            .await?;
        Ok(true)
    }

    pub(crate) async fn enable(&self, agent: &mut hi_agent::Agent) -> Result<String> {
        if self.active.lock().await.is_some() {
            return Ok(self.status().await);
        }
        ensure!(
            agent.active_background_process_ids().is_empty()
                && agent.active_background_task_ids().await.is_empty(),
            "finish or stop all background jobs before /pipefs on"
        );
        agent
            .ensure_background_processes_quiescent()
            .await
            .context("waiting for stopped background processes before /pipefs on")?;
        if let Some(cleanup) = self.cleanup_pending.lock().await.clone() {
            cleanup
                .finish_disable()
                .await
                .context("removing the previous clean PipeFS cache before enabling")?;
            *self.cleanup_pending.lock().await = None;
        }
        let client = self.client()?;
        let capabilities = client
            .capabilities()
            .await
            .context("checking PipeFS capability and authentication")?;
        ensure!(capabilities.enabled, "the IPOP PipeFS service is disabled");
        ensure!(
            capabilities.archive_version == hi_pipefs::ARCHIVE_VERSION,
            "unsupported PipeFS archive version {}",
            capabilities.archive_version
        );
        let sync = self.ensure_sync(agent).await?;
        sync.set_pipefs_sync_required(true);
        let lease = self.lease(&sync).await?;
        let session_id = self.session_id();
        let remote_was_enabled = match client.state(&session_id).await {
            Ok(state) => state.enabled,
            Err(hi_pipefs::PipeFsError::MissingRevision(_)) => false,
            Err(error) => return Err(error.into()),
        };
        let workspace = self.workspace(client, lease)?;
        self.activate(agent, workspace, sync, remote_was_enabled)
            .await?;
        Ok(self.status().await)
    }

    async fn activate(
        &self,
        agent: &mut hi_agent::Agent,
        workspace: hi_pipefs::PipeFsWorkspace,
        sync: Arc<RemoteSessionSink>,
        remote_was_enabled: bool,
    ) -> Result<()> {
        let activation = match workspace.enable().await {
            Ok(activation) => activation,
            Err(error) => {
                let cleanup = async {
                    let session_id = self.session_id();
                    if !remote_was_enabled {
                        let client = self.client()?;
                        match client.state(&session_id).await {
                            Ok(state) if state.enabled => {
                                let lease = self.lease(&sync).await?;
                                client
                                    .set_enabled(&session_id, &lease, false)
                                    .await
                                    .context("rolling back failed PipeFS activation")?;
                            }
                            Ok(_) | Err(hi_pipefs::PipeFsError::Disabled(_)) => {}
                            Err(state_error) => return Err(anyhow!(state_error)),
                        }
                    }
                    if workspace.status().await.phase == hi_pipefs::WorkspacePhase::Disabled {
                        workspace.finish_disable().await?;
                    }
                    Ok(())
                }
                .await;
                if cleanup.is_ok() && !remote_was_enabled {
                    let _ = workspace.record_mode_hint(false);
                    sync.set_pipefs_sync_required(false);
                }
                return match cleanup {
                    Ok(()) => Err(error),
                    Err(cleanup_error) => Err(error.context(format!(
                        "PipeFS activation failed, and rollback also failed: {cleanup_error:#}"
                    ))),
                };
            }
        };
        // A cache path can be reused across launches. Remove any exact persisted
        // grant defensively, then construct the runtime without consulting folder
        // trust: restored repository bytes must never auto-enable local hooks or
        // stdio MCP commands on this machine.
        let _ = hi_tools::folder_trust::try_revoke_folder_trust(&activation.workspace_root);
        if let Err(error) = agent
            .rebind_portable_workspace(&activation.workspace_root, &activation.state_root)
            .await
        {
            let cleanup = if remote_was_enabled {
                // The remote workspace remains authoritative. Keep the clean
                // materialization and the transport pin for an explicit retry
                // instead of silently resuming work in the launch directory.
                Ok(())
            } else {
                async {
                    workspace.prepare_disable().await?;
                    workspace.finish_disable().await
                }
                .await
            };
            if cleanup.is_ok() && !remote_was_enabled {
                let _ = workspace.record_mode_hint(false);
                sync.set_pipefs_sync_required(false);
            }
            return match cleanup {
                Ok(()) => Err(error.context("rebinding the agent to the materialized PipeFS root")),
                Err(cleanup_error) => Err(error.context(format!(
                    "rebinding the agent to PipeFS failed, and activation rollback also failed: {cleanup_error:#}"
                ))),
            };
        }
        agent.set_workspace_project_context(crate::project_context::load_project_context_from(
            &activation.workspace_root,
        ));
        self.reconnect_workspace_mcp(agent, &activation.workspace_root, false)
            .await;
        let background_processes = agent.background_process_registry();
        let durability = Arc::new(PipeFsDurability {
            workspace: workspace.clone(),
            sync,
            background_processes,
            background_checkpoints: Arc::default(),
        });
        agent.set_workspace_durability(Some(durability.clone()));
        *self.active_durability.lock().await = Some(durability);
        *self.active.lock().await = Some(workspace);
        Ok(())
    }

    pub(crate) async fn disable(&self, agent: &mut hi_agent::Agent) -> Result<String> {
        let sync = self.require_sync_handle()?;
        ensure!(
            agent.active_background_process_ids().is_empty()
                && agent.active_background_task_ids().await.is_empty(),
            "finish or stop all background jobs before /pipefs off"
        );
        agent
            .ensure_background_processes_quiescent()
            .await
            .context("waiting for stopped background processes before /pipefs off")?;
        if let Some(durability) = self.active_durability.lock().await.clone() {
            durability
                .quiesce_background_checkpoints()
                .await
                .context("stopping PipeFS background checkpoints before /pipefs off")?;
        }
        let workspace = self
            .active
            .lock()
            .await
            .clone()
            .ok_or_else(|| anyhow!("PipeFS is not active"))?;
        refresh_pipefs_lease(&workspace, &sync).await?;
        workspace.checkpoint().await?;
        sync.flush_required()
            .await
            .context("flushing transcript before disabling PipeFS")?;
        let activation = workspace.prepare_disable().await?;
        if let Err(error) = agent
            .rebind_workspace(&activation.workspace_root, &activation.state_root)
            .await
        {
            if let Err(rollback_error) = workspace.rollback_disable().await {
                // The local runtime still points at the PipeFS cache, so fail
                // closed and leave a retry hint even when the remote reply is
                // ambiguous. `/pipefs retry` reconciles by idempotently
                // enabling the remote mode again before admitting mutations.
                let _ = workspace.record_mode_hint(true);
                return Err(error.context(format!(
                    "returning to the original local workspace failed, and restoring remote PipeFS also failed: {rollback_error:#}"
                )));
            }
            return Err(error.context("returning to the original local workspace"));
        }
        agent.set_workspace_project_context(crate::project_context::load_project_context_from(
            &activation.workspace_root,
        ));
        self.reconnect_workspace_mcp(agent, &activation.workspace_root, true)
            .await;
        agent.set_workspace_durability(None);
        *self.active_durability.lock().await = None;
        *self.active.lock().await = None;
        sync.set_pipefs_sync_required(false);
        match workspace.finish_disable().await {
            Ok(()) => {
                Ok("PipeFS: off; latest revision committed and local cache removed".to_string())
            }
            Err(error) => {
                *self.cleanup_pending.lock().await = Some(workspace);
                Ok(format!(
                    "PipeFS: off; latest revision committed, but clean cache removal failed ({error:#}). Run /pipefs retry"
                ))
            }
        }
    }

    pub(crate) async fn retry(&self, agent: &mut hi_agent::Agent) -> Result<String> {
        if let Some(workspace) = self.active.lock().await.clone() {
            let sync = self.require_sync_handle()?;
            refresh_pipefs_lease(&workspace, &sync).await?;
            if workspace.status().await.phase == hi_pipefs::WorkspacePhase::Disabled {
                workspace
                    .rollback_disable()
                    .await
                    .context("re-enabling PipeFS after an interrupted disable")?;
            }
            // `checkpoint` deliberately scans even a controller recorded as
            // Clean. This recovers writes from opaque/direct command paths that
            // could otherwise be missed by a no-op retry.
            workspace.checkpoint().await?;
            sync.flush_required()
                .await
                .context("retrying required PipeFS transcript delivery")?;
            return Ok(workspace.status().await.to_string());
        }
        if let Some(cleanup) = self.cleanup_pending.lock().await.clone() {
            cleanup.finish_disable().await?;
            *self.cleanup_pending.lock().await = None;
            return Ok("PipeFS: off; clean local cache removed".to_string());
        }
        if hi_pipefs::local_state_requires_remote_probe(&self.session_id()) {
            return self.enable(agent).await;
        }
        bail!("PipeFS has no pending recovery; use /pipefs on to enable it")
    }

    pub(crate) async fn status(&self) -> String {
        if let Some(workspace) = self.active.lock().await.clone() {
            return workspace.status().await.to_string();
        }
        if self.cleanup_pending.lock().await.is_some() {
            return "PipeFS: off; latest revision is remote, clean cache removal is pending (/pipefs retry)"
                .to_string();
        }
        let session_id = self.session_id();
        match self.client() {
            Ok(client) => match client.state(&session_id).await {
                Ok(remote) => format!(
                    "PipeFS: {} (remote head {}, sequence {}, {} bytes); local materialization inactive",
                    if remote.enabled { "on" } else { "off" },
                    remote
                        .current_head
                        .map_or_else(|| "empty".to_string(), |head| head.to_string()),
                    remote.sequence,
                    remote.logical_size_bytes
                ),
                Err(error) => format!("PipeFS: unavailable ({error})"),
            },
            Err(error) => format!("PipeFS: unavailable ({error:#})"),
        }
    }

    pub(crate) async fn is_active(&self) -> bool {
        self.active.lock().await.is_some()
    }

    /// Validate a TUI session switch before transcript state changes. A
    /// PipeFS-enabled target must be restored as part of startup, so the live
    /// switch path refuses it instead of continuing in the launch directory.
    pub(crate) async fn prepare_session_switch(&self, next_session_id: &str) -> Result<()> {
        crate::sync::validate_session_id(next_session_id)?;
        ensure!(
            self.active.lock().await.is_none(),
            "turn PipeFS off before switching sessions"
        );
        // Ordinary local session switching must remain local/offline when the
        // feature has never been used. A session-level hint survives clean
        // cache removal and fails closed for dirty recovery caches; supported
        // cross-machine resumes go through `--attach --resume-local`, which
        // always resolves the authoritative remote summary before activation.
        if hi_pipefs::local_state_requires_remote_probe(next_session_id) {
            bail!(
                "session {next_session_id} may use PipeFS or has a recovery cache; resume it with `hi --attach {next_session_id} --resume-local` so its remote workspace is resolved before activation"
            );
        }
        Ok(())
    }

    /// Complete a switch already validated by `prepare_session_switch`.
    pub(crate) fn complete_session_switch(
        &self,
        next_session_id: String,
        next_session_path: PathBuf,
    ) {
        *self
            .session_id
            .write()
            .unwrap_or_else(|error| error.into_inner()) = next_session_id;
        *self
            .session_path
            .write()
            .unwrap_or_else(|error| error.into_inner()) = next_session_path;
    }

    /// Call only after the frontend has stopped accepting turns and verified
    /// that background work is gone. Failure intentionally leaves the recovery
    /// cache intact.
    pub(crate) async fn clean_exit(&self, agent: &hi_agent::Agent) -> Result<()> {
        let Some(workspace) = self.active.lock().await.clone() else {
            if let Some(cleanup) = self.cleanup_pending.lock().await.clone() {
                cleanup.finish_disable().await?;
                *self.cleanup_pending.lock().await = None;
            }
            return Ok(());
        };
        if !agent.active_background_process_ids().is_empty()
            || !agent.active_background_task_ids().await.is_empty()
        {
            bail!("PipeFS exit blocked while background jobs are active");
        }
        agent
            .ensure_background_processes_quiescent()
            .await
            .context("waiting for stopped background processes before PipeFS exit")?;
        if let Some(durability) = self.active_durability.lock().await.clone() {
            durability
                .quiesce_background_checkpoints()
                .await
                .context("stopping PipeFS background checkpoints before exit")?;
        }
        let sync = self.require_sync_handle()?;
        refresh_pipefs_lease(&workspace, &sync).await?;
        workspace.checkpoint().await?;
        sync.flush_required()
            .await
            .context("flushing transcript before removing the PipeFS cache")?;
        workspace.finish_clean_exit().await
    }
}

fn effective_startup_mode(
    remote_is_authoritative: bool,
    remote_enabled: Option<bool>,
    requested_for_new_session: bool,
) -> bool {
    if remote_is_authoritative {
        remote_enabled.unwrap_or(false)
    } else {
        requested_for_new_session
    }
}

struct PipeFsDurability {
    workspace: hi_pipefs::PipeFsWorkspace,
    sync: Arc<RemoteSessionSink>,
    background_processes: Arc<hi_tools::BackgroundRegistry>,
    background_checkpoints: Arc<BackgroundCheckpointTasks>,
}

async fn stop_background_processes_after_lease_loss(
    workspace: &hi_pipefs::PipeFsWorkspace,
    sync: &RemoteSessionSink,
    background: &hi_tools::BackgroundRegistry,
) -> bool {
    let lease_lost = sync.writer_lease_is_lost()
        || workspace.status().await.phase == hi_pipefs::WorkspacePhase::LeaseLost;
    if !lease_lost {
        return false;
    }
    // A taken-over writer must not leave an autonomous child modifying its
    // stale cache. Preserve the recovery materialization, but stop and reap all
    // native processes before this durability observer exits.
    background.kill_all();
    let _ = background.ensure_quiescent_and_reaped().await;
    true
}

async fn refresh_pipefs_lease(
    workspace: &hi_pipefs::PipeFsWorkspace,
    sync: &RemoteSessionSink,
) -> Result<()> {
    if sync.writer_lease_is_lost() {
        workspace
            .mark_lease_lost("the shared HI writer lease was taken over by another machine")
            .await?;
        bail!("PipeFS writer lease was taken over; this process cannot accept mutations");
    }
    let token = sync.writer_lease_token().ok_or_else(|| {
        anyhow!("the shared HI writer lease is unavailable; recovery cache retained")
    })?;
    let generation = sync.writer_lease_generation();
    ensure!(
        generation > 0,
        "the shared HI writer lease has no generation"
    );
    workspace
        .update_lease(hi_pipefs::PipeFsLease { token, generation })
        .await
}

#[derive(Default)]
struct BackgroundCheckpointTasks {
    tasks: std::sync::Mutex<HashMap<String, tokio::task::JoinHandle<()>>>,
}

impl Drop for BackgroundCheckpointTasks {
    fn drop(&mut self) {
        let tasks = self
            .tasks
            .get_mut()
            .unwrap_or_else(|error| error.into_inner());
        for (_, task) in tasks.drain() {
            task.abort();
        }
    }
}

impl PipeFsDurability {
    async fn quiesce_background_checkpoints(&self) -> Result<()> {
        let tasks = {
            let mut tasks = self
                .background_checkpoints
                .tasks
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            tasks.drain().collect::<Vec<_>>()
        };
        let mut ids = Vec::with_capacity(tasks.len());
        for (id, task) in tasks {
            ids.push(id);
            task.abort();
            let _ = task.await;
        }
        for id in ids {
            self.workspace
                .background_process_state(&id, false)
                .await
                .with_context(|| format!("clearing background checkpoint state for {id}"))?;
        }
        Ok(())
    }
}

#[async_trait::async_trait]
impl hi_agent::WorkspaceDurability for PipeFsDurability {
    async fn mutation_started(&self, dirty_paths: Option<Vec<String>>) -> Result<()> {
        let result = async {
            refresh_pipefs_lease(&self.workspace, &self.sync).await?;
            self.workspace.mutation_started(dirty_paths).await
        }
        .await;
        if result.is_err() {
            stop_background_processes_after_lease_loss(
                &self.workspace,
                &self.sync,
                &self.background_processes,
            )
            .await;
        }
        result
    }

    async fn checkpoint(&self) -> Result<()> {
        let result = async {
            refresh_pipefs_lease(&self.workspace, &self.sync).await?;
            self.workspace.checkpoint().await.map(|_| ())
        }
        .await;
        if result.is_err() {
            stop_background_processes_after_lease_loss(
                &self.workspace,
                &self.sync,
                &self.background_processes,
            )
            .await;
        }
        result
    }

    async fn background_process_state(&self, id: &str, running: bool) -> Result<()> {
        if !running {
            let task = self
                .background_checkpoints
                .tasks
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .remove(id);
            if let Some(task) = task {
                task.abort();
                let _ = task.await;
            }
        }
        if let Err(error) = self.workspace.background_process_state(id, running).await {
            stop_background_processes_after_lease_loss(
                &self.workspace,
                &self.sync,
                &self.background_processes,
            )
            .await;
            return Err(error);
        }
        if !running {
            return Ok(());
        }
        let mut tasks = self
            .background_checkpoints
            .tasks
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        if tasks.contains_key(id) {
            return Ok(());
        }
        let workspace = self.workspace.clone();
        let sync = self.sync.clone();
        let background_processes = self.background_processes.clone();
        let id_for_task = id.to_string();
        let task = tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(15));
            interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
            // `interval` ticks immediately once; the foreground durability
            // fence already checkpoints process launch, so wait for the first
            // real period before reconciling background writes.
            interval.tick().await;
            loop {
                interval.tick().await;
                if refresh_pipefs_lease(&workspace, &sync).await.is_err() {
                    if stop_background_processes_after_lease_loss(
                        &workspace,
                        &sync,
                        &background_processes,
                    )
                    .await
                    {
                        break;
                    }
                    continue;
                }
                let _ = workspace
                    .mutation_started(Some(vec![format!("<background process {id_for_task}>")]))
                    .await;
                if workspace.checkpoint().await.is_err()
                    && stop_background_processes_after_lease_loss(
                        &workspace,
                        &sync,
                        &background_processes,
                    )
                    .await
                {
                    break;
                }
                if matches!(
                    workspace.status().await.phase,
                    hi_pipefs::WorkspacePhase::Disabled | hi_pipefs::WorkspacePhase::LeaseLost
                ) {
                    break;
                }
            }
        });
        tasks.insert(id.to_string(), task);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::effective_startup_mode;

    #[test]
    fn existing_remote_state_wins_startup_precedence() {
        assert!(!effective_startup_mode(true, Some(false), true));
        assert!(effective_startup_mode(true, Some(true), false));
        assert!(!effective_startup_mode(true, None, true));
    }

    #[test]
    fn new_session_uses_explicit_or_configured_request_then_defaults_off() {
        assert!(effective_startup_mode(false, Some(false), true));
        assert!(!effective_startup_mode(false, Some(true), false));
        assert!(!effective_startup_mode(false, None, false));
    }
}
