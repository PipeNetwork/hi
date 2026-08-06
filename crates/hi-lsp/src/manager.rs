//! Per-session LSP manager: owns one server per language, exposes the
//! query API the tools call, and tracks enabled state for `/lsp`.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::Mutex as StdMutex;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use serde_json::{Value, json};
use tokio::sync::Mutex;

use crate::client::{LspClient, PublishedDiagnostics, path_to_uri, uri_to_path};

/// Lock a `std::sync::Mutex`, recovering the guard if a panic poisoned it.
/// The manager's running/synced bookkeeping is advisory — a producer panic
/// mid-update must not crash the whole LSP manager.
fn lock_recover<T>(mutex: &StdMutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}
use crate::detect::{
    Language, detect_language, detect_project_language, install_hint, language_id_for_path,
    server_available, server_command,
};
use crate::types::{
    Diagnostic, DiagnosticState, Location, diagnostic_state_from_items, file_character_to_utf16,
    file_utf16_to_character, parse_hover, parse_locations,
};

/// Maximum number of synced-document hashes to retain. Beyond this the map
/// is cleared (see `synced` field doc).
const SYNCED_CAP: usize = 256;

/// Status of the LSP subsystem, for `/lsp status`.
#[derive(Clone, Debug)]
pub struct ServerStatus {
    pub language: Language,
    pub available: bool,
    pub running: bool,
}

/// Workspace-owned LSP handle. Callers thread it through the tool runtime so
/// servers, diagnostics, and synced-document state cannot leak across agents.
pub struct LspManager {
    enabled: Mutex<bool>,
    servers: Mutex<HashMap<Language, Arc<LspClient>>>,
    /// Sync mirror of which languages have a live server, so `/lsp status`
    /// can render without entering the async runtime. Best-effort: it's only
    /// updated on explicit insert/remove in `ensure` and on `set_enabled`,
    /// so if a server's child exits on its own, `running` still reports `true`
    /// until the next query triggers a respawn via `is_alive()`. Acceptable
    /// for a status display; `status()` (async) is authoritative.
    running: StdMutex<HashMap<Language, bool>>,
    /// Content hash of the last text synced per URI, so we skip redundant
    /// `didChange` notifications when a query re-reads an unchanged file.
    /// Capped at `SYNCED_CAP` entries; on overflow the whole map is cleared
    /// (the hashes are only a dedup optimization — clearing forces a one-time
    /// re-sync of open files, which is correct, and prevents unbounded growth
    /// in a long session touching many files).
    synced: StdMutex<HashMap<String, u64>>,
    root: PathBuf,
}

impl LspManager {
    pub fn new(root: impl Into<PathBuf>) -> Result<Self> {
        let root = root.into();
        anyhow::ensure!(
            root.is_absolute(),
            "LspManager requires an absolute workspace root, got {}",
            root.display()
        );
        let root = root.canonicalize().unwrap_or(root);
        Ok(Self {
            enabled: Mutex::new(false),
            servers: Mutex::new(HashMap::new()),
            running: StdMutex::new(HashMap::new()),
            synced: StdMutex::new(HashMap::new()),
            root,
        })
    }

    /// The explicit workspace root owned by this manager.
    pub fn root(&self) -> &Path {
        &self.root
    }

    fn workspace_path(&self, path: &Path) -> PathBuf {
        if path.is_absolute() {
            path.to_path_buf()
        } else {
            self.root.join(path)
        }
    }

    /// `/lsp on` / `/lsp off`. Disabling shuts down all running servers.
    /// Enabling is lazy: the first query starts the relevant server. This
    /// keeps startup cheap and avoids letting a language server's project
    /// discovery mutate a lockfile-less workspace before the user edits it.
    pub async fn set_enabled(&self, on: bool) {
        {
            let mut enabled = self.enabled.lock().await;
            *enabled = on;
        }
        // `enabled` is released here in both branches. This is critical for
        // lock ordering: query paths (`locations`, `diagnostics`, `hover`,
        // `sync_document`) acquire `servers` first (via `ensure_for_path`→
        // `ensure`) and then `enabled` (via `is_enabled`). If `set_enabled`
        // held `enabled` while waiting for `servers`, a concurrent query
        // holding `servers` and waiting for `enabled` would deadlock. By
        // always releasing `enabled` before touching `servers`, both paths
        // acquire locks in the same order: `servers` before `enabled`.
        if !on {
            let mut servers = self.servers.lock().await;
            for (_, client) in servers.drain() {
                // `shutdown` takes `&self`, so we can shut the server down
                // through the shared `Arc<LspClient>` even if a long-lived
                // query task still holds a clone. The child is force-killed
                // after a 2s grace window, so the orphaned clone's eventual
                // drop is harmless (its `kill_on_drop` is a no-op on an
                // already-dead child).
                let _ = client.shutdown().await;
            }
            lock_recover(&self.running).clear();
            // The dedup hashes describe documents open on the servers we just
            // shut down. Without clearing, a later `/lsp on` would skip the
            // didOpen for "already synced" files the fresh servers never saw.
            lock_recover(&self.synced).clear();
        } else {
            // The server is started by `ensure_for_path` on demand.
        }
    }

    pub async fn is_enabled(&self) -> bool {
        *self.enabled.lock().await
    }

    /// Whether a language server is already running. Fast-feedback callers use
    /// this to avoid turning a mutation into a cold language-server startup;
    /// explicit LSP queries still start the server on demand.
    pub async fn has_running_server(&self) -> bool {
        let servers = self.servers.lock().await;
        for client in servers.values() {
            if client.is_alive().await && !client.is_poisoned() {
                return true;
            }
        }
        false
    }

    /// Status of each known language, for `/lsp status`.
    pub async fn status(&self) -> Vec<ServerStatus> {
        let servers = self.servers.lock().await;
        let langs = [
            Language::Rust,
            Language::Python,
            Language::Go,
            Language::TypeScript,
        ];
        langs
            .iter()
            .map(|&lang| ServerStatus {
                language: lang,
                available: server_available(lang),
                running: servers.contains_key(&lang),
            })
            .collect()
    }

    /// Sync status for command handlers that aren't async. Uses the sync
    /// `running` mirror rather than entering the async runtime. This is a
    /// best-effort view: it can disagree with `status()` transiently if a
    /// server's child has exited but no query has triggered a respawn yet
    /// (see the `running` field doc). Prefer `status()` when async context
    /// is available.
    pub fn status_sync(&self) -> Vec<ServerStatus> {
        let running = lock_recover(&self.running);
        let langs = [
            Language::Rust,
            Language::Python,
            Language::Go,
            Language::TypeScript,
        ];
        langs
            .iter()
            .map(|&lang| ServerStatus {
                language: lang,
                available: server_available(lang),
                running: running.get(&lang).copied().unwrap_or(false),
            })
            .collect()
    }

    /// Ensure a server for `lang` is running, spawning or restarting if needed.
    /// The servers lock is only held for the map check and insert — the slow
    /// spawn happens outside the lock so concurrent queries for a *different*
    /// language aren't blocked behind a cold start.
    ///
    /// On a cold start with parallel queries, multiple `ensure(lang)` calls
    /// can all pass the "not present" fast path and each spawn a server. The
    /// insert path below handles this deterministically: the first inserter
    /// wins, and later spawners detect the existing live server and explicitly
    /// shut down their duplicate (rather than relying on `kill_on_drop`).
    async fn ensure(&self, lang: Language) -> Result<()> {
        // Fast path: already running, alive, and its stream is intact.
        {
            let servers = self.servers.lock().await;
            if let Some(client) = servers.get(&lang) {
                if client.is_alive().await && !client.is_poisoned() {
                    return Ok(());
                }
                // Crashed or desynced — fall through to respawn. Drop the lock
                // first by removing under a fresh acquisition below.
            } else {
                // Not present — spawn.
            }
        }
        // Remove any dead/poisoned entry, then spawn outside the lock.
        let stale = {
            let mut servers = self.servers.lock().await;
            match servers.get(&lang) {
                Some(client) if client.is_alive().await && !client.is_poisoned() => {
                    return Ok(()); // raced with another ensure; it's healthy
                }
                Some(_) => {
                    let stale = servers.remove(&lang);
                    lock_recover(&self.running).remove(&lang);
                    // The old server had documents open that the replacement
                    // won't know about — drop the dedup hashes so every file
                    // re-syncs (didOpen) on its next query instead of being
                    // skipped as "unchanged" against a server that never saw it.
                    lock_recover(&self.synced).clear();
                    stale
                }
                None => None,
            }
        };
        // A poisoned child may still be alive — reap it deterministically.
        if let Some(stale) = stale {
            let _ = stale.shutdown().await;
        }
        if !server_available(lang) {
            bail!("{}", install_hint(lang));
        }
        let (cmd, args) = server_command(lang);
        let client = LspClient::spawn(cmd, &args, &self.root).await?;
        let mut servers = self.servers.lock().await;
        // If another task raced and inserted a healthy server, keep it and
        // explicitly shut down the duplicate we just spawned (rather than
        // relying on `kill_on_drop` at drop time, which is correct but
        // non-deterministic about *when* the orphaned child is reaped).
        if let Some(existing) = servers.get(&lang) {
            if existing.is_alive().await && !existing.is_poisoned() {
                drop(servers); // release before awaiting shutdown
                let _ = client.shutdown().await;
                return Ok(());
            }
            servers.remove(&lang);
            lock_recover(&self.synced).clear();
        }
        servers.insert(lang, Arc::new(client));
        lock_recover(&self.running).insert(lang, true);
        Ok(())
    }

    /// Resolve the language for a path, ensuring its server is up.
    async fn ensure_for_path(&self, path: &Path) -> Result<Language> {
        let lang = detect_language(path)
            .or_else(|| detect_project_language(&self.root))
            .ok_or_else(|| anyhow::anyhow!("no LSP server for this file type"))?;
        self.ensure(lang).await?;
        Ok(lang)
    }

    /// Push the current file contents to the server (didOpen or didChange).
    /// Skips the round-trip when the text is unchanged since the last sync
    /// for this URI, so repeated queries on the same file don't re-send the
    /// full body each time.
    pub async fn sync_document(&self, path: &Path, text: &str) -> Result<()> {
        if !self.is_enabled().await {
            return Ok(());
        }
        let path = self.workspace_path(path);
        // A file whose extension names no language has no server that can
        // parse it. Opening it anyway means falling back to the *project*
        // language (below and in `ensure_for_path`), which hands `Makefile`,
        // `Cargo.toml` and `Cargo.lock` to rust-analyzer as if they were Rust —
        // it then reports every line as a syntax error. Those bogus
        // diagnostics fail the `lsp` verification stage on a workspace that is
        // perfectly healthy, and no edit the model makes can ever clear them.
        if detect_language(&path).is_none() {
            return Ok(());
        }
        let lang = self.ensure_for_path(&path).await?;
        let uri = path_to_uri(&path);
        let hash = fxhash(text);
        let already_open;
        {
            let mut synced = lock_recover(&self.synced);
            already_open = synced.contains_key(&uri);
            if already_open && synced.get(&uri).copied() == Some(hash) {
                return Ok(()); // unchanged — skip the didChange
            }
            if already_open {
                // New text for an already-open doc: clear stale pushed
                // diagnostics so stale errors don't linger after the server
                // re-publishes (or publishes nothing) for the new content.
                // The didChange below triggers a fresh publishDiagnostics.
            }
            if !already_open && synced.len() >= SYNCED_CAP {
                // Cap reached: clear the dedup map. Open files will re-sync
                // on their next query (a one-time cost), preventing unbounded
                // growth in a long session.
                synced.clear();
            }
            synced.insert(uri.clone(), hash);
        }
        // Clone the Arc handle and drop the servers lock before the
        // didChange/didOpen round-trip (which can take up to the drain
        // timeout), so a sync for one language doesn't block queries for
        // any other language.
        let client = {
            let servers = self.servers.lock().await;
            servers
                .get(&lang)
                .with_context(|| format!("no LSP server for {lang:?} after ensure"))?
                .clone()
        };
        let result = if already_open {
            client.clear_pushed_diagnostics(&uri);
            client.did_change(&uri, text).await
        } else {
            let language_id = language_id_for_path(&path).unwrap_or_else(|| lang.language_id());
            client.did_open(&uri, language_id, text).await
        };
        if result.is_ok() {
            return Ok(());
        }
        let err = result.expect_err("checked is_ok above");
        // The hash was inserted optimistically above, but the server never
        // received the notify — leaving it would make every future sync
        // skip this content as "unchanged".
        lock_recover(&self.synced).remove(&uri);

        // Dead/poisoned servers used to fail every subsequent file in the
        // batch with the same "closed the stream" noise. Respawn once and
        // retry as a fresh didOpen against the replacement.
        if !is_recoverable_transport_error(&err) {
            return Err(err);
        }
        self.ensure(lang).await?;
        let client = {
            let servers = self.servers.lock().await;
            servers
                .get(&lang)
                .with_context(|| format!("no LSP server for {lang:?} after restart"))?
                .clone()
        };
        // After a restart the replacement has never seen this URI, so force
        // didOpen even if we thought the doc was already open on the dead server.
        lock_recover(&self.synced).insert(uri.clone(), hash);
        let language_id = language_id_for_path(&path).unwrap_or_else(|| lang.language_id());
        match client.did_open(&uri, language_id, text).await {
            Ok(()) => Ok(()),
            Err(retry_err) => {
                lock_recover(&self.synced).remove(&uri);
                Err(retry_err)
            }
        }
    }

    /// Close a deleted or no-longer-relevant document and discard all cached
    /// diagnostics for it. This prevents a deleted file's last publication
    /// from surviving in workspace-wide diagnostic results.
    pub async fn close_document(&self, path: &Path) -> Result<()> {
        let path = self.workspace_path(path);
        let Some(lang) = detect_language(&path).or_else(|| detect_project_language(&self.root))
        else {
            return Ok(());
        };
        let uri = path_to_uri(&path);
        lock_recover(&self.synced).remove(&uri);
        let client = self.servers.lock().await.get(&lang).cloned();
        if let Some(client) = client {
            client.did_close(&uri).await?;
        }
        Ok(())
    }

    /// Fetch a versioned diagnostic state. A clean state is returned only
    /// after an empty push publication or a successful pull response for the
    /// current document version.
    pub async fn diagnostic_state(&self, path: &Path) -> DiagnosticState {
        if !self.is_enabled().await {
            return DiagnosticState::Unavailable {
                document_version: None,
                reason: "LSP is disabled".into(),
            };
        }
        let path = self.workspace_path(path);
        let lang = match self.ensure_for_path(&path).await {
            Ok(lang) => lang,
            Err(error) => {
                return DiagnosticState::Unavailable {
                    document_version: None,
                    reason: format!("{error:#}"),
                };
            }
        };
        let uri = path_to_uri(&path);
        let client = self.servers.lock().await.get(&lang).cloned();
        let Some(client) = client else {
            return DiagnosticState::Failed {
                document_version: None,
                error: format!("no LSP server for {lang:?} after startup"),
            };
        };
        self.diagnostic_state_with_client(&client, &path, &uri)
            .await
    }

    async fn diagnostic_state_with_client(
        &self,
        client: &Arc<LspClient>,
        path: &Path,
        uri: &str,
    ) -> DiagnosticState {
        let Some(version) = client.document_version(uri) else {
            return DiagnosticState::Unavailable {
                document_version: None,
                reason: "document has not been synchronized with the language server".into(),
            };
        };
        if let Some(pushed) = client.get_pushed_diagnostics(uri)
            && publication_matches_document(&pushed, version)
        {
            return diagnostic_state_from_items(path, version, &pushed.items);
        }

        // Give push-only servers a bounded opportunity to publish an explicit
        // empty/nonempty result for this version.
        if client.drain_notifications(Duration::from_secs(10)).await
            == crate::client::DrainOutcome::Dead
        {
            return DiagnosticState::Failed {
                document_version: Some(version),
                error: "LSP server closed the stream".into(),
            };
        }
        if let Some(pushed) = client.get_pushed_diagnostics(uri)
            && publication_matches_document(&pushed, version)
        {
            return diagnostic_state_from_items(path, version, &pushed.items);
        }

        if !client.supports_pull_diagnostics() {
            let reason = match client.get_pushed_diagnostics(uri) {
                Some(pushed) if pushed.version.is_none() && version > 0 => {
                    "server published unversioned diagnostics after didChange; freshness cannot be confirmed and diagnostic pull is unsupported"
                }
                Some(_) => {
                    "server published diagnostics for a different document version and diagnostic pull is unsupported"
                }
                None => "server did not publish diagnostics and does not support diagnostic pull",
            };
            return DiagnosticState::Unavailable {
                document_version: Some(version),
                reason: reason.into(),
            };
        }
        match client
            .request(
                "textDocument/diagnostic",
                Some(json!({ "textDocument": { "uri": uri } })),
            )
            .await
        {
            Ok(Value::Array(items)) => diagnostic_state_from_items(path, version, &items),
            Ok(Value::Object(obj)) => match obj.get("items").and_then(Value::as_array) {
                Some(items) => diagnostic_state_from_items(path, version, items),
                None => DiagnosticState::Failed {
                    document_version: Some(version),
                    error: "diagnostic pull response did not contain `items`".into(),
                },
            },
            Ok(other) => DiagnosticState::Failed {
                document_version: Some(version),
                error: format!("unexpected diagnostic pull response: {other}"),
            },
            Err(error) => DiagnosticState::Failed {
                document_version: Some(version),
                error: format!("{error:#}"),
            },
        }
    }

    /// Compatibility helper for callers that want only diagnostics. Unlike
    /// the old API, unavailable/failed servers are surfaced as errors instead
    /// of being translated into a false "no diagnostics" result.
    pub async fn diagnostics(&self, path: &Path) -> Result<Vec<Diagnostic>> {
        match self.diagnostic_state(path).await {
            DiagnosticState::ConfirmedClean { .. } => Ok(Vec::new()),
            DiagnosticState::DiagnosticsPresent { diagnostics, .. } => Ok(diagnostics),
            DiagnosticState::Unavailable { reason, .. } => {
                bail!("LSP diagnostics unavailable: {reason}")
            }
            DiagnosticState::Failed { error, .. } => bail!("LSP diagnostics failed: {error}"),
        }
    }

    /// Versioned states for every currently open document, including clean,
    /// unavailable, and failed states.
    pub async fn diagnostic_states_all(&self) -> Vec<(PathBuf, DiagnosticState)> {
        let uris: Vec<String> = lock_recover(&self.synced).keys().cloned().collect();
        let mut out = Vec::with_capacity(uris.len());
        for uri in uris {
            let path = PathBuf::from(uri_to_path(&uri));
            let state = self.diagnostic_state(&path).await;
            out.push((path, state));
        }
        out
    }

    /// Diagnostics for all open documents. Infrastructure or availability
    /// failures fail the operation; they never collapse to an empty list.
    pub async fn diagnostics_all(&self) -> Result<Vec<(PathBuf, Vec<Diagnostic>)>> {
        let mut out = Vec::new();
        for (path, state) in self.diagnostic_states_all().await {
            match state {
                DiagnosticState::ConfirmedClean { .. } => {}
                DiagnosticState::DiagnosticsPresent { diagnostics, .. } => {
                    out.push((path, diagnostics));
                }
                DiagnosticState::Unavailable { reason, .. } => {
                    bail!(
                        "LSP diagnostics unavailable for {}: {reason}",
                        path.display()
                    )
                }
                DiagnosticState::Failed { error, .. } => {
                    bail!("LSP diagnostics failed for {}: {error}", path.display())
                }
            }
        }
        Ok(out)
    }

    /// Synchronize and diagnose a set of changed files as one logical batch.
    /// Deleted paths are closed so stale publications are removed.
    /// Distinct documents are diagnosed concurrently (bounded) so a large edit
    /// set does not serialize N round-trips on the fast-feedback path.
    pub async fn diagnostics_batch(&self, paths: &[PathBuf]) -> Vec<(PathBuf, DiagnosticState)> {
        use futures_util::stream::{self, StreamExt};

        const CONCURRENCY: usize = 4;
        stream::iter(paths.iter().cloned())
            .map(|original| async move {
                let path = self.workspace_path(&original);
                // Not something any configured server owns — report it as
                // unchecked rather than routing it to the project's server,
                // which would parse it as that language and emit a syntax
                // error per line. `Unavailable` is ignored by the verifier;
                // `Failed` would fail the stage.
                if detect_language(&path).is_none() {
                    return (
                        path,
                        DiagnosticState::Unavailable {
                            document_version: None,
                            reason: "no LSP server handles this file type".into(),
                        },
                    );
                }
                if !path.exists() {
                    let state = match self.close_document(&path).await {
                        Ok(()) => DiagnosticState::Unavailable {
                            document_version: None,
                            reason: "document was deleted".into(),
                        },
                        Err(error) => DiagnosticState::Failed {
                            document_version: None,
                            error: format!("closing deleted document: {error:#}"),
                        },
                    };
                    return (path, state);
                }
                let state = match tokio::fs::read_to_string(&path).await {
                    Ok(text) => {
                        let first = match self.sync_document(&path, &text).await {
                            Ok(()) => self.diagnostic_state(&path).await,
                            Err(error) => DiagnosticState::Failed {
                                document_version: None,
                                error: format!("synchronizing document: {error:#}"),
                            },
                        };
                        // One more recovery pass for transport death discovered
                        // after sync (e.g. closed stream during diagnostic drain).
                        // `sync_document` already respawns on its own notify/drain
                        // failures; this covers the post-sync diagnostic path.
                        match &first {
                            DiagnosticState::Failed { error, .. }
                                if is_recoverable_transport_error(&anyhow::anyhow!("{error}")) =>
                            {
                                match self.sync_document(&path, &text).await {
                                    Ok(()) => self.diagnostic_state(&path).await,
                                    Err(error) => DiagnosticState::Failed {
                                        document_version: None,
                                        error: format!("synchronizing document: {error:#}"),
                                    },
                                }
                            }
                            _ => first,
                        }
                    }
                    Err(error) => DiagnosticState::Failed {
                        document_version: None,
                        error: format!("reading document: {error}"),
                    },
                };
                (path, state)
            })
            .buffer_unordered(CONCURRENCY)
            .collect()
            .await
    }

    /// Goto definition.
    pub async fn definition(&self, path: &Path, line: u32, col: u32) -> Result<Vec<Location>> {
        self.locations("textDocument/definition", path, line, col)
            .await
    }

    /// Find references.
    pub async fn references(&self, path: &Path, line: u32, col: u32) -> Result<Vec<Location>> {
        self.locations("textDocument/references", path, line, col)
            .await
    }

    async fn locations(
        &self,
        method: &str,
        path: &Path,
        line: u32,
        col: u32,
    ) -> Result<Vec<Location>> {
        if !self.is_enabled().await {
            return Ok(Vec::new());
        }
        let path = self.workspace_path(path);
        let lang = self.ensure_for_path(&path).await?;
        let uri = path_to_uri(&path);
        let protocol_col = file_character_to_utf16(&path, line, col).unwrap_or(col);
        // Clone the Arc handle and drop the servers lock before the request
        // round-trip, so a query for one language doesn't block queries for
        // any other language.
        let client = {
            let servers = self.servers.lock().await;
            servers
                .get(&lang)
                .with_context(|| format!("no LSP server for {lang:?} after ensure"))?
                .clone()
        };
        // Retry on "content modified" (-32801): rust-analyzer returns this
        // transient error when a didChange notification is still being
        // processed as the request arrives. A short retry lets it settle.
        let mut last_err: Option<anyhow::Error> = None;
        for attempt in 0..3 {
            let mut params = json!({
                "textDocument": { "uri": uri },
                "position": { "line": line, "character": protocol_col }
            });
            if method == "textDocument/references" {
                params["context"] = json!({ "includeDeclaration": true });
            }
            let result = client.request(method, Some(params)).await;
            match result {
                Ok(v) => {
                    let mut locations = parse_locations(&v);
                    for location in &mut locations {
                        let target = Path::new(&location.path);
                        location.col = file_utf16_to_character(target, location.line, location.col)
                            .unwrap_or(location.col);
                    }
                    return Ok(locations);
                }
                Err(e) => {
                    let msg = format!("{e:#}");
                    if msg.contains("-32801") || msg.contains("content modified") {
                        last_err = Some(e);
                        tokio::time::sleep(Duration::from_millis(150 * (attempt + 1) as u64)).await;
                        continue;
                    }
                    return Err(e);
                }
            }
        }
        Err(last_err.unwrap_or_else(|| anyhow::anyhow!("LSP `{method}` failed")))
    }

    /// Hover info at a position.
    pub async fn hover(&self, path: &Path, line: u32, col: u32) -> Result<Option<String>> {
        if !self.is_enabled().await {
            return Ok(None);
        }
        let path = self.workspace_path(path);
        let lang = self.ensure_for_path(&path).await?;
        let uri = path_to_uri(&path);
        let protocol_col = file_character_to_utf16(&path, line, col).unwrap_or(col);
        // Clone the Arc handle and drop the servers lock before the request
        // round-trip, so a query for one language doesn't block queries for
        // any other language.
        let client = {
            let servers = self.servers.lock().await;
            servers
                .get(&lang)
                .with_context(|| format!("no LSP server for {lang:?} after ensure"))?
                .clone()
        };
        let result = client
            .request(
                "textDocument/hover",
                Some(json!({
                    "textDocument": { "uri": uri },
                    "position": { "line": line, "character": protocol_col }
                })),
            )
            .await?;
        Ok(parse_hover(&result))
    }
}

/// Decide whether a push publication is authoritative for the current text.
///
/// An explicit server version must match exactly. A versionless publication
/// is usable for the initial `didOpen` generation only, where there is no
/// earlier open-document content it could describe. Once `didChange` advances
/// the version, an omitted version cannot cross that freshness boundary; the
/// caller must use diagnostic pull or return `Unavailable`.
fn publication_matches_document(published: &PublishedDiagnostics, document_version: u64) -> bool {
    match published.version {
        Some(published_version) => published_version == document_version,
        None => document_version == 0,
    }
}

/// Transport deaths that should trigger an immediate respawn+retry rather than
/// bubbling a permanent failure for the whole batch.
fn is_recoverable_transport_error(err: &anyhow::Error) -> bool {
    let text = format!("{err:#}").to_ascii_lowercase();
    text.contains("closed the stream")
        || text.contains("lost sync")
        || text.contains("broken pipe")
        || text.contains("connection reset")
}

/// FNV-1a 64-bit hash. Used only to detect unchanged file contents so we
/// can skip redundant `didChange` notifications — not cryptographic.
fn fxhash(s: &str) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in s.as_bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn closing_deleted_document_removes_it_from_workspace_diagnostics() {
        let root = std::env::temp_dir().join(format!(
            "hi-lsp-close-{}-{}",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        let path = root.join("deleted.rs");
        let uri = path_to_uri(&path);
        let manager = LspManager::new(&root).unwrap();
        manager.synced.lock().unwrap().insert(uri, 1);
        manager.close_document(&path).await.unwrap();
        assert!(manager.synced.lock().unwrap().is_empty());
        let _ = std::fs::remove_dir_all(root);
    }

    #[tokio::test]
    async fn non_source_files_are_never_routed_to_the_project_language_server() {
        // The 12-hour-goal failure: in a Cargo workspace, `detect_language`
        // returns None for `Makefile`/`Cargo.toml`/`Cargo.lock`, and the old
        // code fell back to the *project* language — handing them to
        // rust-analyzer, which reported every line as a Rust syntax error.
        // Those diagnostics failed the `lsp` verify stage on a healthy tree,
        // and no edit could ever clear them.
        let root = std::env::temp_dir().join(format!(
            "hi-lsp-nonsource-{}-{}",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        // Marks the project as Rust, which is what the old fallback keyed on.
        std::fs::write(root.join("Cargo.toml"), "[workspace]\n").unwrap();
        std::fs::write(root.join("Makefile"), "all:\n\tcargo build\n").unwrap();
        std::fs::write(root.join("Cargo.lock"), "version = 4\n").unwrap();
        let manager = LspManager::new(&root).unwrap();
        // Enable directly so the guard under test is reached without needing
        // a real language server. `sync_document` returns early while the
        // manager is disabled and the test would otherwise pass trivially.
        *manager.enabled.lock().await = true;

        for name in ["Makefile", "Cargo.toml", "Cargo.lock"] {
            let path = root.join(name);
            // Must be a no-op: nothing opened, and crucially no server spawned
            // (the pre-fix path would route these to rust-analyzer).
            manager.sync_document(&path, "irrelevant").await.unwrap();
            assert!(
                !manager
                    .synced
                    .lock()
                    .unwrap()
                    .contains_key(&path_to_uri(&path)),
                "{name} must not be opened on a language server"
            );
            // And it must read as unchecked, not as a failure: `Failed` would
            // fail the verification stage just as the bogus diagnostics did.
            let states = manager.diagnostics_batch(&[path]).await;
            assert!(
                matches!(states[0].1, DiagnosticState::Unavailable { .. }),
                "{name} should be Unavailable, got {:?}",
                states[0].1
            );
        }
        let _ = std::fs::remove_dir_all(root);
    }

    #[tokio::test]
    async fn enabling_lsp_is_lazy() {
        let root = std::env::temp_dir().join(format!(
            "hi-lsp-lazy-{}-{}",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("Cargo.toml"), "[workspace]\n").unwrap();
        let manager = LspManager::new(&root).unwrap();
        manager.set_enabled(true).await;
        assert!(
            manager
                .status()
                .await
                .into_iter()
                .all(|status| !status.running),
            "enabling LSP must not spawn a server before the first query"
        );
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn recoverable_transport_errors_are_detected() {
        assert!(is_recoverable_transport_error(&anyhow::anyhow!(
            "synchronizing document: LSP server closed the stream"
        )));
        assert!(is_recoverable_transport_error(&anyhow::anyhow!(
            "LSP stream lost sync during `textDocument/diagnostic`; the server will be restarted on the next query"
        )));
        assert!(is_recoverable_transport_error(&anyhow::anyhow!(
            "Broken pipe (os error 32)"
        )));
        assert!(!is_recoverable_transport_error(&anyhow::anyhow!(
            "no LSP server for this file type"
        )));
    }

    #[test]
    fn queued_versionless_push_cannot_confirm_clean_after_did_change() {
        // This models a push queued for version 0 but read only after the
        // client has sent didChange and advanced to version 1. The old code
        // stamped it as version 1 at receipt and falsely confirmed clean.
        let queued = PublishedDiagnostics {
            version: None,
            items: Vec::new(),
        };

        assert!(!publication_matches_document(&queued, 1));
    }

    #[test]
    fn only_the_exact_explicit_document_version_is_authoritative() {
        let stale = PublishedDiagnostics {
            version: Some(3),
            items: Vec::new(),
        };
        let current = PublishedDiagnostics {
            version: Some(4),
            items: Vec::new(),
        };

        assert!(!publication_matches_document(&stale, 4));
        assert!(publication_matches_document(&current, 4));
    }

    #[test]
    fn initial_versionless_push_is_bounded_to_did_open_generation() {
        let initial = PublishedDiagnostics {
            version: None,
            items: Vec::new(),
        };

        assert!(publication_matches_document(&initial, 0));
        assert!(!publication_matches_document(&initial, 1));
    }

    /// Find the workspace root by walking up from CWD until we find a
    /// `Cargo.toml` with `[workspace]`.
    fn workspace_root() -> std::path::PathBuf {
        let mut dir = std::env::current_dir().unwrap();
        loop {
            let cargo_toml = dir.join("Cargo.toml");
            if cargo_toml.exists()
                && let Ok(content) = std::fs::read_to_string(&cargo_toml)
                && content.contains("[workspace]")
            {
                return dir;
            }
            if !dir.pop() {
                return std::env::current_dir().unwrap();
            }
        }
    }

    /// Smoke test: spawn real rust-analyzer on this workspace, open a file
    /// with a deliberate type error, and verify diagnostics come back.
    ///
    /// Uses a temporary *untracked* source file in the crate's `src/` dir, so
    /// no tracked file is ever corrupted (the previous form appended to
    /// `lib.rs`, which a concurrent `cargo check` could read mid-write). The
    /// temp file is deleted on drop. Self-skips unless rust-analyzer both
    /// exists on PATH and actually runs — a bare rustup *shim* passes the PATH
    /// check but fails at spawn when the component isn't installed.
    #[tokio::test]
    async fn rust_analyzer_reports_diagnostics() {
        use crate::detect::{Language, server_available};
        if !server_available(Language::Rust) {
            eprintln!("skipping: rust-analyzer not on PATH");
            return;
        }
        let root = workspace_root();
        let mgr = LspManager::new(&root).unwrap();
        mgr.set_enabled(true).await;

        // A fresh, untracked file inside the crate's source dir: rust-analyzer
        // analyzes any file opened under the workspace root, and an untracked
        // path means the working tree is never left modified. Deleted on drop.
        let target = root.join("crates/hi-lsp/src/__lsp_diag_smoke.rs");
        let broken = "fn _smoke() { let x: u32 = \"bad\"; }\n";
        tokio::fs::write(&target, broken).await.unwrap();
        struct RemoveOnDrop(std::path::PathBuf);
        impl Drop for RemoveOnDrop {
            fn drop(&mut self) {
                let _ = std::fs::remove_file(&self.0);
            }
        }
        let _guard = RemoveOnDrop(target.clone());

        // Sync the broken file so the server analyzes it (the manager no
        // longer re-syncs inside `diagnostics`). A rustup shim whose
        // rust-analyzer component isn't installed closes the stream here —
        // treat that as "server unavailable" and skip.
        if let Err(err) = mgr.sync_document(&target, broken).await {
            eprintln!("skipping: rust-analyzer unavailable ({err})");
            return;
        }
        let diags = mgr.diagnostics(&target).await.unwrap();
        eprintln!("diagnostics ({}): {:?}", diags.len(), diags);

        assert!(
            diags.iter().any(|d| d.severity == "error"),
            "expected an error-severity diagnostic for the type error, got: {diags:?}"
        );
        // `_guard` drops here and deletes the temp file.
    }

    /// Smoke test: definition on a real symbol in this workspace. Read-only
    /// (no source files are mutated). Self-skips unless rust-analyzer both
    /// exists on PATH and actually runs — a bare rustup *shim* passes the PATH
    /// check but fails at spawn when the component isn't installed.
    #[tokio::test]
    async fn rust_analyzer_finds_definition() {
        use crate::detect::{Language, server_available};
        if !server_available(Language::Rust) {
            eprintln!("skipping: rust-analyzer not on PATH");
            return;
        }
        let root = workspace_root();
        let mgr = LspManager::new(&root).unwrap();
        mgr.set_enabled(true).await;

        // `LspManager` is defined in this file. Open it and find the definition
        // of the struct name on its declaration line.
        let path = root.join("crates/hi-lsp/src/manager.rs");
        let text = tokio::fs::read_to_string(&path).await.unwrap();
        let line = text
            .lines()
            .position(|l| l.contains("pub struct LspManager"))
            .unwrap() as u32;
        let col = text
            .lines()
            .nth(line as usize)
            .unwrap()
            .find("LspManager")
            .unwrap() as u32;

        // Sync the document first (the tool layer does this before querying;
        // the manager no longer re-syncs internally to avoid redundant reads).
        // A rustup shim whose rust-analyzer component isn't installed closes
        // the stream here — treat that as "server unavailable" and skip.
        if let Err(err) = mgr.sync_document(&path, &text).await {
            eprintln!("skipping: rust-analyzer unavailable ({err})");
            return;
        }
        let locs = mgr.definition(&path, line, col).await.unwrap();
        eprintln!("definition locations: {locs:?}");
        assert!(
            !locs.is_empty(),
            "expected at least one definition location"
        );
        assert!(
            locs.iter().any(|l| l.path.contains("manager.rs")),
            "expected definition in manager.rs"
        );
    }
}
