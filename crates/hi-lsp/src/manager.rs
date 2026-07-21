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
    /// Enabling proactively warms up the server for the detected project
    /// language so the first query is fast.
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
            self.running.lock().unwrap().clear();
            // The dedup hashes describe documents open on the servers we just
            // shut down. Without clearing, a later `/lsp on` would skip the
            // didOpen for "already synced" files the fresh servers never saw.
            self.synced.lock().unwrap().clear();
        } else {
            // Warm up the server for the project's primary language.
            if let Some(lang) = detect_project_language(&self.root)
                && server_available(lang)
            {
                let _ = self.ensure(lang).await;
            }
        }
    }

    pub async fn is_enabled(&self) -> bool {
        *self.enabled.lock().await
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
        let running = self.running.lock().unwrap();
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
                    self.running.lock().unwrap().remove(&lang);
                    // The old server had documents open that the replacement
                    // won't know about — drop the dedup hashes so every file
                    // re-syncs (didOpen) on its next query instead of being
                    // skipped as "unchanged" against a server that never saw it.
                    self.synced.lock().unwrap().clear();
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
            self.synced.lock().unwrap().clear();
        }
        servers.insert(lang, Arc::new(client));
        self.running.lock().unwrap().insert(lang, true);
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
        let lang = self.ensure_for_path(&path).await?;
        let uri = path_to_uri(&path);
        let hash = fxhash(text);
        let already_open;
        {
            let mut synced = self.synced.lock().unwrap();
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
        let err = result.err().expect("checked is_ok above");
        // The hash was inserted optimistically above, but the server never
        // received the notify — leaving it would make every future sync
        // skip this content as "unchanged".
        self.synced.lock().unwrap().remove(&uri);

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
        self.synced.lock().unwrap().insert(uri.clone(), hash);
        let language_id = language_id_for_path(&path).unwrap_or_else(|| lang.language_id());
        match client.did_open(&uri, language_id, text).await {
            Ok(()) => Ok(()),
            Err(retry_err) => {
                self.synced.lock().unwrap().remove(&uri);
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
        self.synced.lock().unwrap().remove(&uri);
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
        let uris: Vec<String> = self.synced.lock().unwrap().keys().cloned().collect();
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
    pub async fn diagnostics_batch(&self, paths: &[PathBuf]) -> Vec<(PathBuf, DiagnosticState)> {
        let mut out = Vec::with_capacity(paths.len());
        for original in paths {
            let path = self.workspace_path(original);
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
                out.push((path, state));
                continue;
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
            out.push((path, state));
        }
        out
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
    /// Skipped if rust-analyzer isn't on $PATH.
    ///
    /// `#[ignore]`: this mutates a tracked source file (`lib.rs`) and depends
    /// on rust-analyzer being installed, so it must not run in the normal
    /// `cargo test` suite. Run explicitly with `cargo test -p hi-lsp
    /// -- --ignored`.
    #[tokio::test]
    #[ignore]
    async fn rust_analyzer_reports_diagnostics() {
        use crate::detect::{Language, server_available};
        if !server_available(Language::Rust) {
            eprintln!("skipping: rust-analyzer not on PATH");
            return;
        }
        let root = workspace_root();
        let mgr = LspManager::new(&root).unwrap();
        mgr.set_enabled(true).await;

        // Append a type error to lib.rs (which is in the module tree, so
        // rust-analyzer will actually analyze it). A `Drop` guard restores the
        // original content even if the test panics or the assertion fails, so
        // the tracked source file is never left corrupted on disk.
        let target = root.join("crates/hi-lsp/src/lib.rs");
        let original = tokio::fs::read_to_string(&target).await.unwrap();
        let broken = format!("{original}\nfn _smoke() {{ let x: u32 = \"bad\"; }}\n");
        tokio::fs::write(&target, &broken).await.unwrap();
        // Guard restores the file on drop — fires on early return or panic.
        let target_clone = target.clone();
        let original_clone = original.clone();
        struct RestoreOnDrop {
            path: std::path::PathBuf,
            content: String,
        }
        impl Drop for RestoreOnDrop {
            fn drop(&mut self) {
                let _ = std::fs::write(&self.path, &self.content);
            }
        }
        let _guard = RestoreOnDrop {
            path: target_clone,
            content: original_clone,
        };

        // Sync the broken file so the server analyzes it (the manager no
        // longer re-syncs inside `diagnostics`).
        mgr.sync_document(&target, &broken).await.unwrap();
        let diags = mgr.diagnostics(&target).await.unwrap();
        eprintln!("diagnostics ({}): {:?}", diags.len(), diags);

        assert!(
            diags.iter().any(|d| d.severity == "error"),
            "expected an error-severity diagnostic for the type error, got: {diags:?}"
        );
        // `_guard` drops here and restores lib.rs.
    }

    /// Smoke test: definition on a real symbol in this workspace.
    /// `#[ignore]`: depends on rust-analyzer being installed; run with
    /// `cargo test -p hi-lsp -- --ignored`.
    #[tokio::test]
    #[ignore]
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
        mgr.sync_document(&path, &text).await.unwrap();
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
