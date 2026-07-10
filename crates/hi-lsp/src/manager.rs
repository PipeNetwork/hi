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

use crate::client::{LspClient, path_to_uri, uri_to_path};
use crate::detect::{
    Language, detect_language, detect_project_language, install_hint, server_available,
    server_command,
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

/// The per-session LSP handle. Held in a process-global so the tool layer
/// can reach it without threading it through every `execute` call site
/// (mirroring how `READ_CACHE` works in `hi-tools`).
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
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self {
            enabled: Mutex::new(false),
            servers: Mutex::new(HashMap::new()),
            running: StdMutex::new(HashMap::new()),
            synced: StdMutex::new(HashMap::new()),
            root: root.into(),
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
        let lang = self.ensure_for_path(path).await?;
        let uri = path_to_uri(path);
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
            client.did_open(&uri, lang.language_id(), text).await
        };
        if let Err(e) = result {
            // The hash was inserted optimistically above, but the server never
            // received the notify — leaving it would make every future sync
            // skip this content as "unchanged".
            self.synced.lock().unwrap().remove(&uri);
            return Err(e);
        }
        Ok(())
    }

    /// Fetch diagnostics for a file. The server pushes these via
    /// `textDocument/publishDiagnostics` after didOpen/didChange; we request
    /// them with the pull-based `textDocument/diagnostic` if the server
    /// supports it, else return what we last synced.
    ///
    /// The caller is responsible for syncing the document first (so repeated
    /// queries on the same file don't re-read and re-sync here).
    pub async fn diagnostics(&self, path: &Path) -> Result<Vec<Diagnostic>> {
        if !self.is_enabled().await {
            return Ok(Vec::new());
        }
        let lang = self.ensure_for_path(path).await?;
        let uri = path_to_uri(path);
        // Clone the Arc handle and drop the servers lock before awaiting
        // drain_notifications / the diagnostic request, so a query for one
        // language doesn't block queries for any other language.
        let client = {
            let servers = self.servers.lock().await;
            servers
                .get(&lang)
                .with_context(|| format!("no LSP server for {lang:?} after ensure"))?
                .clone()
        };
        // Check pushed diagnostics first — rust-analyzer uses the push model.
        let pushed = client.get_pushed_diagnostics(&uri);
        if !pushed.is_empty() {
            return Ok(pushed.iter().filter_map(parse_diagnostic).collect());
        }
        // No pushed diagnostics yet — the server may still be analyzing.
        // Drain for up to 10s to wait for the publishDiagnostics notification.
        client.drain_notifications(Duration::from_secs(10)).await;
        let pushed = client.get_pushed_diagnostics(&uri);
        if !pushed.is_empty() {
            return Ok(pushed.iter().filter_map(parse_diagnostic).collect());
        }
        // Fallback: try the pull-based `textDocument/diagnostic` request.
        let result = client
            .request(
                "textDocument/diagnostic",
                Some(json!({ "textDocument": { "uri": uri } })),
            )
            .await;
        match result {
            Ok(Value::Array(items)) => Ok(items.iter().filter_map(parse_diagnostic).collect()),
            Ok(Value::Object(obj)) => {
                // The pull model returns { kind: "full", items: [...] }.
                if let Some(items) = obj.get("items").and_then(|i| i.as_array()) {
                    Ok(items.iter().filter_map(parse_diagnostic).collect())
                } else {
                    Ok(Vec::new())
                }
            }
            Ok(_) => Ok(Vec::new()),
            Err(e) => {
                // A transport error here usually means the server crashed or
                // timed out. Surface it under `HI_LSP_DEBUG` so a silently
                // empty result is distinguishable from "no problems found".
                if std::env::var_os("HI_LSP_DEBUG").is_some() {
                    eprintln!("hi-lsp: textDocument/diagnostic for {uri} failed: {e:#}");
                }
                Ok(Vec::new())
            }
        }
    }

    /// Diagnostics for every document that has been synced so far, keyed by
    /// path. Used when the caller asks for diagnostics with no specific file
    /// (empty path) — returns the union across all open files.
    ///
    /// Groups URIs by language and resolves each server once, rather than
    /// re-entering `ensure`/server-lock per file, so N open files cost one
    /// `ensure` per language (not N).
    pub async fn diagnostics_all(&self) -> Result<Vec<(PathBuf, Vec<Diagnostic>)>> {
        if !self.is_enabled().await {
            return Ok(Vec::new());
        }
        // Snapshot the synced URIs under the std lock (no await inside).
        let uris: Vec<String> = self.synced.lock().unwrap().keys().cloned().collect();
        // Resolve each language's client once, up front.
        let mut clients: HashMap<Language, Arc<LspClient>> = HashMap::new();
        for uri in &uris {
            let path = uri_to_path(uri);
            let lang = match detect_language(Path::new(&path))
                .or_else(|| detect_project_language(&self.root))
            {
                Some(l) => l,
                None => continue,
            };
            if let std::collections::hash_map::Entry::Vacant(entry) = clients.entry(lang)
                && self.ensure(lang).await.is_ok()
            {
                let client = {
                    let servers = self.servers.lock().await;
                    servers.get(&lang).cloned()
                };
                if let Some(c) = client {
                    entry.insert(c);
                }
            }
        }
        let mut out = Vec::new();
        for uri in uris {
            let path = uri_to_path(&uri);
            let lang = match detect_language(Path::new(&path))
                .or_else(|| detect_project_language(&self.root))
            {
                Some(l) => l,
                None => continue,
            };
            let Some(client) = clients.get(&lang) else {
                continue;
            };
            let diags = self
                .diagnostics_with_client(client, &path, &uri)
                .await
                .unwrap_or_default();
            if !diags.is_empty() {
                out.push((PathBuf::from(path), diags));
            }
        }
        Ok(out)
    }

    /// Fetch diagnostics for `uri` using an already-resolved `client`, so
    /// `diagnostics_all` doesn't re-acquire the servers lock per file. Mirrors
    /// `diagnostics` but skips the `is_enabled`/`ensure`/lock preamble.
    async fn diagnostics_with_client(
        &self,
        client: &Arc<LspClient>,
        _path: &str,
        uri: &str,
    ) -> Result<Vec<Diagnostic>> {
        let pushed = client.get_pushed_diagnostics(uri);
        if !pushed.is_empty() {
            return Ok(pushed.iter().filter_map(parse_diagnostic).collect());
        }
        client.drain_notifications(Duration::from_secs(10)).await;
        let pushed = client.get_pushed_diagnostics(uri);
        if !pushed.is_empty() {
            return Ok(pushed.iter().filter_map(parse_diagnostic).collect());
        }
        let result = client
            .request(
                "textDocument/diagnostic",
                Some(json!({ "textDocument": { "uri": uri } })),
            )
            .await;
        match result {
            Ok(Value::Array(items)) => Ok(items.iter().filter_map(parse_diagnostic).collect()),
            Ok(Value::Object(obj)) => {
                if let Some(items) = obj.get("items").and_then(|i| i.as_array()) {
                    Ok(items.iter().filter_map(parse_diagnostic).collect())
                } else {
                    Ok(Vec::new())
                }
            }
            Ok(_) => Ok(Vec::new()),
            Err(e) => {
                if std::env::var_os("HI_LSP_DEBUG").is_some() {
                    eprintln!("hi-lsp: textDocument/diagnostic for {uri} failed: {e:#}");
                }
                Ok(Vec::new())
            }
        }
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
        let lang = self.ensure_for_path(path).await?;
        let uri = path_to_uri(path);
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
            let result = client
                .request(
                    method,
                    Some(json!({
                        "textDocument": { "uri": uri },
                        "position": { "line": line, "character": col }
                    })),
                )
                .await;
            match result {
                Ok(v) => return Ok(parse_locations(&v)),
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
        let lang = self.ensure_for_path(path).await?;
        let uri = path_to_uri(path);
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
                    "position": { "line": line, "character": col }
                })),
            )
            .await?;
        Ok(parse_hover(&result))
    }
}

// --- Types and parsers ---

/// One LSP diagnostic (error/warning), flattened for the model.
#[derive(Clone, Debug)]
pub struct Diagnostic {
    pub severity: String,
    pub line: u32,
    pub col: u32,
    pub message: String,
    pub source: Option<String>,
}

/// One location: file + 1-based line/col.
#[derive(Clone, Debug)]
pub struct Location {
    pub path: String,
    pub line: u32,
    pub col: u32,
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

fn severity_label(n: u64) -> String {
    match n {
        1 => "error".into(),
        2 => "warning".into(),
        3 => "info".into(),
        4 => "hint".into(),
        _ => "note".into(),
    }
}

fn parse_diagnostic(v: &Value) -> Option<Diagnostic> {
    let sev = v.get("severity").and_then(|s| s.as_u64()).unwrap_or(0);
    let start = v.get("range")?.get("start")?;
    Some(Diagnostic {
        severity: severity_label(sev),
        line: start.get("line")?.as_u64()? as u32,
        col: start.get("character")?.as_u64()? as u32,
        message: v.get("message")?.as_str()?.to_string(),
        source: v.get("source").and_then(|s| s.as_str()).map(String::from),
    })
}

fn parse_location(v: &Value) -> Option<Location> {
    let uri = v.get("uri")?.as_str()?;
    let start = v.get("range")?.get("start")?;
    Some(Location {
        path: uri_to_path(uri),
        line: start.get("line")?.as_u64()? as u32,
        col: start.get("character")?.as_u64()? as u32,
    })
}

fn parse_locations(v: &Value) -> Vec<Location> {
    match v {
        Value::Array(items) => items.iter().filter_map(parse_location).collect(),
        Value::Object(_) => parse_location(v).into_iter().collect(),
        _ => Vec::new(),
    }
}

fn parse_hover(v: &Value) -> Option<String> {
    let content = v.get("contents").or_else(|| v.get("value"))?;
    if let Some(s) = content.as_str() {
        return Some(s.to_string());
    }
    if let Some(obj) = content.as_object()
        && let Some(s) = obj.get("value").and_then(|v| v.as_str())
    {
        return Some(s.to_string());
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let mgr = LspManager::new(&root);
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
        let mgr = LspManager::new(&root);
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
