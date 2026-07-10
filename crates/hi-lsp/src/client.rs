//! A single LSP server process: spawn, initialize, sync documents, query.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Mutex as StdMutex;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::{Child, ChildStdout};
use tokio::sync::Mutex;

use crate::protocol::{read_message, request_timeout, write_message};

/// Diagnostics pushed by the server via `textDocument/publishDiagnostics`,
/// keyed by document URI. Updated as notifications arrive during requests.
pub type DiagnosticsMap = StdMutex<HashMap<String, Vec<Value>>>;

/// Once data for a message has started arriving, how long a complete frame may
/// take to finish before we declare the stream unrecoverable. Local pipes
/// deliver even multi-MB `publishDiagnostics` payloads in milliseconds, so
/// this only fires on a genuinely wedged server.
const MESSAGE_GRACE: Duration = Duration::from_secs(10);

/// Outcome of one bounded read attempt on the server's stdout.
enum ReadOutcome {
    /// A complete JSON-RPC message.
    Message(Vec<u8>),
    /// No data arrived within the budget; the stream position is untouched.
    Idle,
    /// The stream closed (server exited or pipe broke at a frame boundary).
    Closed,
    /// A read stalled mid-frame — the stream position is unknown, so this
    /// client is unusable and must be respawned (see [`LspClient::is_poisoned`]).
    Poisoned,
}

/// One running language server. Owns the child process and its stdio.
pub struct LspClient {
    child: Mutex<Child>,
    stdin: Mutex<tokio::process::ChildStdin>,
    stdout: Mutex<BufReader<ChildStdout>>,
    /// Serializes whole request/drain round-trips. Without it, two concurrent
    /// requests read from the same stream and each can consume (and drop) the
    /// other's response, leaving the loser to spin until its timeout.
    io: Mutex<()>,
    /// Set when a read stopped mid-frame, leaving the JSON-RPC stream position
    /// unknown: every later read would misparse body bytes as headers. The
    /// manager checks this and respawns the server.
    poisoned: AtomicBool,
    next_id: AtomicU64,
    versions: StdMutex<HashMap<String, u64>>,
    /// Diagnostics pushed by the server, keyed by document URI.
    pub pushed_diagnostics: DiagnosticsMap,
    pub capabilities: Value,
}

impl LspClient {
    /// Spawn `cmd args` and run the LSP `initialize` handshake.
    pub async fn spawn(cmd: &str, args: &[&str], root: &Path) -> Result<Self> {
        let stderr = stderr_setup(cmd);
        let mut child = tokio::process::Command::new(cmd)
            .args(args)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(stderr)
            .kill_on_drop(true)
            .spawn()
            .with_context(|| format!("spawning LSP server `{cmd}`"))?;
        let stdin = child.stdin.take().context("no stdin")?;
        let stdout = child.stdout.take().context("no stdout")?;
        let mut client = Self {
            child: Mutex::new(child),
            stdin: Mutex::new(stdin),
            stdout: Mutex::new(BufReader::new(stdout)),
            io: Mutex::new(()),
            poisoned: AtomicBool::new(false),
            next_id: AtomicU64::new(1),
            versions: StdMutex::new(HashMap::new()),
            pushed_diagnostics: StdMutex::new(HashMap::new()),
            capabilities: Value::Null,
        };
        client.initialize(root).await?;
        Ok(client)
    }

    async fn initialize(&mut self, root: &Path) -> Result<()> {
        let params = json!({
            "processId": std::process::id(),
            "rootUri": path_to_uri(root),
            "capabilities": {
                "textDocument": {
                    "synchronization": { "didSave": true }
                }
            }
        });
        let resp = self.request("initialize", Some(params)).await?;
        self.capabilities = resp;
        self.notify("initialized", json!({})).await?;
        Ok(())
    }

    /// Read one message from stdout, spending at most `budget` *waiting for
    /// data to appear*. Split into two phases because `read_message` is not
    /// cancellation-safe: a timeout that fires mid-frame throws away consumed
    /// bytes and permanently desyncs the stream.
    ///
    /// Phase 1 waits on `fill_buf`, which consumes nothing — cancelling it is
    /// harmless. Once bytes are available, phase 2 commits to reading the full
    /// frame under its own generous grace; only if *that* stalls is the client
    /// marked poisoned (respawned by the manager on the next query).
    async fn read_one(&self, budget: Duration) -> ReadOutcome {
        let mut stdout = self.stdout.lock().await;
        match tokio::time::timeout(budget, stdout.fill_buf()).await {
            Err(_) => return ReadOutcome::Idle,
            Ok(Err(_)) => return ReadOutcome::Closed,
            Ok(Ok([])) => return ReadOutcome::Closed,
            Ok(Ok(_)) => {}
        }
        match tokio::time::timeout(MESSAGE_GRACE, read_message(&mut stdout)).await {
            Ok(Some(msg)) => ReadOutcome::Message(msg),
            Ok(None) => ReadOutcome::Closed,
            Err(_) => {
                self.poisoned.store(true, Ordering::SeqCst);
                ReadOutcome::Poisoned
            }
        }
    }

    /// Record a `publishDiagnostics` notification into `pushed_diagnostics`.
    /// Other notifications are dropped.
    fn capture_notification(&self, v: &Value) {
        if v.get("method").and_then(|m| m.as_str()) == Some("textDocument/publishDiagnostics")
            && let Some(params) = v.get("params")
            && let Some(uri) = params.get("uri").and_then(|u| u.as_str())
        {
            let diags = params
                .get("diagnostics")
                .and_then(|d| d.as_array())
                .cloned()
                .unwrap_or_default();
            self.pushed_diagnostics
                .lock()
                .unwrap()
                .insert(uri.to_string(), diags);
        }
    }

    /// Send a request and wait for the matching response.
    ///
    /// The whole round-trip holds the `io` lock, so concurrent requests (and
    /// drains) on the same client serialize instead of stealing each other's
    /// messages off the shared stream. `LspManager` clones an `Arc` handle and
    /// drops the servers lock before calling, so different languages still run
    /// concurrently.
    pub async fn request(&self, method: &str, params: Option<Value>) -> Result<Value> {
        let _io = self.io.lock().await;
        let id = self.next_id.fetch_add(1, Ordering::SeqCst);
        let body = json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": method,
            "params": params.unwrap_or(Value::Null)
        });
        {
            let mut stdin = self.stdin.lock().await;
            write_message(&mut stdin, &body.to_string()).await?;
        }
        let deadline = Instant::now() + request_timeout();
        loop {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                bail!("LSP request `{method}` timed out");
            }
            match self.read_one(remaining).await {
                ReadOutcome::Idle => continue, // deadline re-checked above
                ReadOutcome::Closed => bail!("LSP server closed the stream"),
                ReadOutcome::Poisoned => bail!(
                    "LSP stream lost sync during `{method}`; the server will be restarted on the next query"
                ),
                ReadOutcome::Message(msg) => {
                    let v: Value = serde_json::from_slice(&msg)?;
                    if v.get("id").and_then(|i| i.as_u64()) == Some(id) {
                        if let Some(err) = v.get("error") {
                            bail!("LSP error on `{method}`: {err}");
                        }
                        return Ok(v["result"].clone());
                    }
                    // Capture pushed diagnostics — the server sends these as
                    // notifications after didOpen/didChange, not as responses.
                    self.capture_notification(&v);
                }
            }
        }
    }

    pub async fn notify(&self, method: &str, params: Value) -> Result<()> {
        let body = json!({ "jsonrpc": "2.0", "method": method, "params": params });
        let mut stdin = self.stdin.lock().await;
        write_message(&mut stdin, &body.to_string()).await?;
        Ok(())
    }

    pub async fn did_open(&self, uri: &str, language_id: &str, text: &str) -> Result<()> {
        self.versions.lock().unwrap().insert(uri.to_string(), 0);
        self.notify("textDocument/didOpen", json!({
            "textDocument": { "uri": uri, "languageId": language_id, "version": 0, "text": text }
        })).await?;
        // Drain pending notifications (the server pushes diagnostics after
        // didOpen). A short budget: most servers publish within a few hundred
        // ms; the diagnostics method does its own longer wait if needed.
        self.drain_notifications(Duration::from_millis(500)).await;
        Ok(())
    }

    pub async fn did_change(&self, uri: &str, text: &str) -> Result<()> {
        let version = {
            let mut v = self.versions.lock().unwrap();
            let n = v.entry(uri.to_string()).or_insert(0);
            *n += 1;
            *n
        };
        self.notify(
            "textDocument/didChange",
            json!({
                "textDocument": { "uri": uri, "version": version },
                "contentChanges": [{ "text": text }]
            }),
        )
        .await?;
        // Short drain — see did_open. The diagnostics method waits longer.
        self.drain_notifications(Duration::from_millis(500)).await;
        Ok(())
    }

    /// Read any pending notifications from stdout, capturing
    /// `publishDiagnostics` into `pushed_diagnostics`. Returns after `wait`
    /// with no data. Holds the `io` lock so it can't race a concurrent
    /// request's read loop and eat its response; the two-phase `read_one`
    /// means an expiring budget can't cancel a frame mid-read (which used to
    /// desync the stream when a large diagnostics payload straddled the
    /// deadline).
    pub async fn drain_notifications(&self, wait: Duration) {
        let _io = self.io.lock().await;
        let deadline = Instant::now() + wait;
        loop {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                return;
            }
            match self.read_one(remaining).await {
                ReadOutcome::Message(msg) => {
                    if let Ok(v) = serde_json::from_slice::<Value>(&msg) {
                        self.capture_notification(&v);
                    }
                }
                ReadOutcome::Idle | ReadOutcome::Closed | ReadOutcome::Poisoned => return,
            }
        }
    }

    /// Shut the server down gracefully, then force-kill if it doesn't exit.
    /// Takes `&self` (not `&mut self`) so a server can be shut down from a
    /// shared `Arc<LspClient>` even when other tasks still hold clones — this
    /// is what `/lsp off` needs, since a long-lived query task may keep a
    /// strong ref. The `child`/`stdin`/`stdout` are already behind `Mutex`es,
    /// so no `&mut` is actually required.
    pub async fn shutdown(&self) -> Result<()> {
        // Skip the graceful JSON-RPC goodbye on a desynced stream — the
        // `shutdown` request would only misread frames until its timeout.
        if !self.is_poisoned() {
            let _ = self.request("shutdown", None).await;
            let _ = self.notify("exit", Value::Null).await;
        }
        // Give the server a moment to exit gracefully, then force-kill so a
        // stubborn server can't hang the shutdown indefinitely. `kill_on_drop`
        // would eventually clean up, but only when the `LspClient` is dropped —
        // making that deterministic here avoids relying on drop ordering.
        let mut child = self.child.lock().await;
        let waited = tokio::time::timeout(Duration::from_secs(2), child.wait()).await;
        if waited.is_err() {
            let _ = child.kill().await;
            let _ = child.wait().await;
        }
        Ok(())
    }

    /// Whether the child process is still running.
    pub async fn is_alive(&self) -> bool {
        matches!(self.child.lock().await.try_wait(), Ok(None))
    }

    /// Whether the JSON-RPC stream has lost sync (a read stalled mid-frame).
    /// A poisoned client's child may still be alive, but no further messages
    /// can be exchanged reliably — the manager respawns it on the next query.
    pub fn is_poisoned(&self) -> bool {
        self.poisoned.load(Ordering::SeqCst)
    }

    /// Get the diagnostics the server has pushed for a URI (via
    /// `publishDiagnostics`), if any. Returns a clone of the raw JSON values.
    pub fn get_pushed_diagnostics(&self, uri: &str) -> Vec<Value> {
        self.pushed_diagnostics
            .lock()
            .unwrap()
            .get(uri)
            .cloned()
            .unwrap_or_default()
    }

    /// Drop any cached pushed diagnostics for a URI. Called before a
    /// `didChange` so stale errors from the previous content don't linger
    /// if the server publishes nothing (or less) for the new content.
    pub fn clear_pushed_diagnostics(&self, uri: &str) {
        self.pushed_diagnostics.lock().unwrap().remove(uri);
    }
}

/// Configure the child's stderr. By default stderr is discarded, but if
/// `HI_LSP_DEBUG` is set in the environment, stderr is piped to a file at
/// `$TMPDIR/hi-lsp-<cmd>-<pid>.log` so server logs and crash output can be
/// inspected when debugging a misbehaving server.
fn stderr_setup(cmd: &str) -> std::process::Stdio {
    if std::env::var_os("HI_LSP_DEBUG").is_none() {
        return std::process::Stdio::null();
    }
    let dir = std::env::temp_dir();
    let path = dir.join(format!("hi-lsp-{cmd}-{}.log", std::process::id()));
    match std::fs::File::create(&path) {
        Ok(f) => std::process::Stdio::from(f),
        Err(_) => std::process::Stdio::null(),
    }
}

/// Percent-encode a single byte for the URI path component. Encodes
/// everything that is not unreserved per RFC 3986 §2.2/§2.3: the reserved
/// set, `%`, and any non-ASCII byte. `/` is left as-is so path separators
/// survive. (`:` is also left as-is; this is harmless on Unix and would only
/// matter for Windows drive letters, which this crate doesn't target.)
fn pct_encode_path(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for &b in s.as_bytes() {
        match b {
            // unreserved
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' | b'/' | b':' => {
                out.push(b as char)
            }
            _ => {
                out.push('%');
                out.push_str(&format!("{b:02X}"));
            }
        }
    }
    out
}

pub(crate) fn path_to_uri(path: &Path) -> String {
    let abs = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir().unwrap_or_default().join(path)
    };
    let s = abs.display().to_string();
    let encoded = pct_encode_path(&s);
    if encoded.starts_with('/') {
        format!("file://{encoded}")
    } else {
        format!("file:///{encoded}")
    }
}

/// Decode percent-encoded triplets (`%XX`) in a URI path back to bytes.
pub fn uri_to_path(uri: &str) -> String {
    let path = uri
        .strip_prefix("file://")
        .or_else(|| uri.strip_prefix("file:///"))
        .unwrap_or(uri);
    let bytes = path.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            let hi = (bytes[i + 1] as char).to_digit(16);
            let lo = (bytes[i + 2] as char).to_digit(16);
            if let (Some(h), Some(l)) = (hi, lo) {
                out.push((h * 16 + l) as u8);
                i += 3;
                continue;
            }
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}
