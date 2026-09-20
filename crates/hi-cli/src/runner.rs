//! Versioned guest-runner interface. All input and detailed output stay on pipes
//! and in the VM's private state directory; never parse the terminal UI.
use anyhow::{Context, Result, ensure};
use hi_harness::{
    Harness, HarnessConfig, JsonlSession, PermissionMode, TurnCancellation, TurnStopReason, Ui,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    fs::OpenOptions,
    io::Write,
    path::PathBuf,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::{SystemTime, UNIX_EPOCH},
};
use tokio::io::{AsyncBufRead, AsyncBufReadExt, AsyncReadExt, BufReader};

const MAX_FRAME: usize = 1_048_576;
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Start {
    version: u32,
    operation: String,
    run_id: uuid::Uuid,
    workspace: PathBuf,
    state_dir: PathBuf,
    credential_file: PathBuf,
    #[serde(default = "base_url")]
    base_url: String,
    prompt: Option<String>,
    #[serde(default = "call_cap")]
    call_micros: u64,
    #[serde(default = "run_cap")]
    run_micros: u64,
    #[serde(default = "execution_seconds")]
    execution_seconds: u64,
    /// The control plane's original deadline; never refreshed by resume.
    #[serde(default)]
    deadline_unix_ms: Option<u64>,
}
fn base_url() -> String {
    hi_harness::DEFAULT_BASE_URL.into()
}
fn call_cap() -> u64 {
    1_000_000
}
fn run_cap() -> u64 {
    20_000_000
}
fn execution_seconds() -> u64 {
    3600
}
fn millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .try_into()
        .unwrap_or(u64::MAX)
}
fn original_deadline(request: &Start, now: u64) -> Result<u64> {
    let maximum = now
        .checked_add(request.execution_seconds * 1000)
        .context("execution deadline overflow")?;
    let deadline = request.deadline_unix_ms.unwrap_or(maximum);
    ensure!(
        deadline > now && deadline <= maximum,
        "invalid original execution deadline"
    );
    Ok(deadline)
}
fn usd(n: u64) -> String {
    format!("{}.{:06}", n / 1_000_000, n % 1_000_000)
}

#[derive(Deserialize, Serialize)]
struct Binding {
    run_id: uuid::Uuid,
    workspace: PathBuf,
    credential_file: PathBuf,
    base_url: String,
    call_micros: u64,
    run_micros: u64,
    deadline: u64,
    execution_seconds: u64,
    #[serde(default)]
    deadline_unix_ms: Option<u64>,
}
async fn frame<R: AsyncBufRead + Unpin>(reader: &mut R) -> Result<Option<Vec<u8>>> {
    let mut bytes = Vec::new();
    let n = reader
        .take((MAX_FRAME + 1) as u64)
        .read_until(b'\n', &mut bytes)
        .await?;
    if n == 0 {
        return Ok(None);
    }
    ensure!(
        n <= MAX_FRAME && bytes.last() == Some(&b'\n'),
        "incomplete or oversized runner frame"
    );
    Ok(Some(bytes))
}
struct JsonUi {
    cancel: TurnCancellation,
}
impl JsonUi {
    fn emit(&self, event: &str, data: Value) {
        let mut out = std::io::stdout().lock();
        if serde_json::to_writer(&mut out, &json!({"version":1,"event":event,"data":data})).is_err()
            || out.write_all(b"\n").is_err()
            || out.flush().is_err()
        {
            self.cancel.cancel();
        }
    }
}
impl Ui for JsonUi {
    fn assistant_text(&mut self, text: &str) {
        self.emit("assistant_text", json!({"text":text}));
    }
    fn assistant_reasoning(&mut self, text: &str) {
        self.emit("assistant_reasoning", json!({"text":text}));
    }
    fn assistant_end(&mut self) {
        self.emit("assistant_end", json!({}));
    }
    fn tool_started_id(&mut self, id: &str, name: &str, arguments: &str) {
        self.emit(
            "tool_started",
            json!({"id":id,"name":name,"arguments":arguments}),
        );
    }
    fn tool_call(&mut self, name: &str, arguments: &str) {
        self.emit("tool_call", json!({"name":name,"arguments":arguments}));
    }
    fn tool_result(&mut self, name: &str, result: &str) {
        self.emit("tool_result", json!({"name":name,"result":result}));
    }
    fn tool_result_id(&mut self, id: &str, name: &str, result: &str, status: hi_tools::ToolStatus) {
        self.emit(
            "tool_result",
            json!({"id":id,"name":name,"result":result,"status":format!("{status:?}")}),
        );
    }
    fn tool_stream(&mut self, name: &str, line: &str) {
        self.emit("tool_stream", json!({"name":name,"text":line}));
    }
    fn status(&mut self, text: &str) {
        self.emit("status", json!({"text":text}));
    }
    fn turn_end(&mut self, summary: &str) {
        self.emit("turn_end", json!({"summary":summary}));
    }
    fn turn_error(&mut self, kind: &str, message: &str, guidance: &str) {
        self.emit(
            "error",
            json!({"kind":kind,"message":message,"guidance":guidance}),
        );
    }
}

fn validate(request: &Start) -> Result<()> {
    ensure!(
        request.version == 1
            && matches!(request.operation.as_str(), "start" | "resume" | "inspect"),
        "unsupported runner protocol"
    );
    ensure!(
        request.call_micros > 0
            && request.call_micros <= 1_000_000
            && request.run_micros >= request.call_micros
            && request.run_micros <= 20_000_000
            && (1..=3600).contains(&request.execution_seconds),
        "invalid runner limits"
    );
    ensure!(
        (request.operation == "start"
            && request
                .prompt
                .as_ref()
                .is_some_and(|p| !p.trim().is_empty()))
            || (matches!(request.operation.as_str(), "resume" | "inspect")
                && request.prompt.is_none()),
        "start requires a prompt; resume uses the original journal"
    );
    let url = url::Url::parse(&request.base_url)?;
    ensure!(
        url.scheme() == "https"
            && url.username().is_empty()
            && url.password().is_none()
            && url.query().is_none()
            && url.fragment().is_none(),
        "managed endpoint requires HTTPS without embedded credentials"
    );
    Ok(())
}

pub async fn run() -> Result<()> {
    let cancel = TurnCancellation::new();
    let mut ui = JsonUi {
        cancel: cancel.clone(),
    };
    let result = execute(&mut ui, cancel).await;
    if result.is_err() {
        ui.emit(
            "terminal",
            json!({"status":"requires_action","safe_error":"runner_protocol_or_recovery_failed"}),
        );
    }
    result
}
async fn execute(ui: &mut JsonUi, cancel: TurnCancellation) -> Result<()> {
    let mut input = BufReader::new(tokio::io::stdin());
    let request: Start = serde_json::from_slice(
        &frame(&mut input)
            .await?
            .context("runner start frame required")?,
    )?;
    validate(&request)?;
    let workspace = request.workspace.canonicalize()?;
    std::fs::create_dir_all(&request.state_dir)?;
    let state = request.state_dir.canonicalize()?;
    let credential = request.credential_file.canonicalize()?;
    ensure!(
        !state.starts_with(&workspace) && !credential.starts_with(&workspace),
        "runner state and credentials must be outside the checkout"
    );
    #[cfg(unix)]
    {
        use std::os::unix::fs::{MetadataExt, PermissionsExt};
        std::fs::set_permissions(&state, std::fs::Permissions::from_mode(0o700))?;
        let info = credential.metadata()?;
        ensure!(
            info.is_file() && info.uid() == unsafe { libc::geteuid() } && info.mode() & 0o077 == 0,
            "credential must be an owner-only file"
        );
    }
    #[cfg(unix)]
    let _run_lock = {
        use std::os::{fd::AsRawFd, unix::fs::OpenOptionsExt};
        let file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(false)
            .mode(0o600)
            .open(state.join("runner.lock"))?;
        ensure!(
            unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } == 0,
            "this run already has a local process"
        );
        file
    };
    let binding = state.join("runner.json");
    let session = state.join("session.jsonl");
    let deadline = if request.operation == "start" {
        ensure!(!session.exists(), "existing session requires resume");
        let deadline = original_deadline(&request, millis())?;
        let record = Binding {
            run_id: request.run_id,
            workspace: workspace.clone(),
            credential_file: credential.clone(),
            base_url: request.base_url.clone(),
            call_micros: request.call_micros,
            run_micros: request.run_micros,
            deadline: deadline / 1000,
            execution_seconds: request.execution_seconds,
            deadline_unix_ms: Some(deadline),
        };
        let mut options = OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        let mut file = options.open(&binding)?;
        serde_json::to_writer(&mut file, &record)?;
        file.sync_all()?;
        std::fs::File::open(&state)?.sync_all()?;
        deadline
    } else {
        ensure!(
            session.is_file() && session.with_extension("managed.json").is_file(),
            "resume requires the original session and managed journal"
        );
        let record: Binding = serde_json::from_reader(std::fs::File::open(binding)?)?;
        ensure!(
            record.run_id == request.run_id
                && record.workspace == workspace
                && record.credential_file == credential
                && record.base_url == request.base_url
                && record.call_micros == request.call_micros
                && record.run_micros == request.run_micros
                && record.execution_seconds == request.execution_seconds,
            "resume cannot change original authority or limits"
        );
        let deadline = record
            .deadline_unix_ms
            .unwrap_or(record.deadline.saturating_mul(1000));
        ensure!(
            request.deadline_unix_ms.is_none_or(|d| d == deadline),
            "resume cannot change the original absolute deadline"
        );
        deadline
    };
    let key = std::fs::read_to_string(credential)?;
    ensure!(!key.trim().is_empty(), "runner credential missing");
    let managed = hi_harness::ManagedSettings {
        call_budget_usd: usd(request.call_micros),
        turn_budget_usd: usd(request.run_micros),
        ..Default::default()
    };
    if request.operation == "inspect" {
        // Read-only inspection is allowed after the deadline for reconciliation.
        // It does not open a turn or query a provider.
        let inspection = hi_harness::inspect_managed_journal(
            &session.with_extension("managed.json"),
            &request.base_url,
            key.trim(),
            &managed,
        )?;
        ui.emit(
            "terminal",
            json!({"status":"inspected","deadline_unix_ms":deadline,
            "deadline_elapsed":deadline<=millis(),"recovery":inspection}),
        );
        return Ok(());
    }
    ensure!(deadline > millis(), "original execution deadline elapsed");
    let mut config = HarnessConfig::pipe(workspace, key.trim());
    config.model = "pipe/auto".into();
    config.base_url = request.base_url.clone();
    config.state_root = state;
    config.session_path = Some(session.clone());
    config.max_tokens = 8192;
    config.managed = managed.clone();
    // The default balanced managed harness requires buffered verification, disables
    // retrieval/ordinary fallback, and journals auxiliary calls in the same budget.
    config.typesafe = hi_harness::TypesafeSettings::disabled();
    config.jev_compact = false;
    let mut harness = Harness::new(config)?;
    harness.set_permission_mode(PermissionMode::Always);
    harness.set_turn_intent_mode(true, true);
    if request.operation == "resume" {
        harness.apply_loaded_session(JsonlSession::load(&session)?);
        ensure!(
            harness.can_resume_incomplete(None),
            "session has no resumable turn"
        );
    }
    ui.emit("ready",json!({"run_id":request.run_id,"model":"pipe/auto","deadline":deadline/1000,"deadline_unix_ms":deadline,"permission_policy":"vm_user_full_authority"}));
    let watcher_cancel = cancel.clone();
    let controls = tokio::spawn(async move {
        let events = JsonUi {
            cancel: watcher_cancel.clone(),
        };
        loop {
            match frame(&mut input).await {
                Ok(Some(bytes))=>match serde_json::from_slice::<Value>(&bytes) {
                    Ok(value) if value["version"]==1 && value["operation"]=="status"=>events.emit("status",json!({"state":if watcher_cancel.is_cancelled(){"cancelling"}else{"running"}})),
                    Ok(value) if value["version"]==1 && value["operation"]=="cancel"=>{watcher_cancel.cancel();events.emit("cancellation_requested",json!({}));break;},
                    _=>{watcher_cancel.cancel();break;}
                },
                _=>{watcher_cancel.cancel();break;}
            }
        }
    });
    let deadline_cancel = cancel.clone();
    let timed_out = Arc::new(AtomicBool::new(false));
    let deadline_reached = timed_out.clone();
    let timer = tokio::spawn(async move {
        tokio::time::sleep(std::time::Duration::from_millis(
            deadline.saturating_sub(millis()),
        ))
        .await;
        deadline_reached.store(true, Ordering::Release);
        deadline_cancel.cancel();
    });
    let signal_cancel = cancel.clone();
    let signal = tokio::spawn(async move {
        #[cfg(unix)]
        if let Ok(mut term) =
            tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
        {
            tokio::select! { _ = term.recv() => {}, _ = tokio::signal::ctrl_c() => {} }
            signal_cancel.cancel();
            return;
        }
        let _ = tokio::signal::ctrl_c().await;
        signal_cancel.cancel();
    });
    let outcome = if request.operation == "start" {
        harness
            .run_turn_cancellable(
                request.prompt.as_deref().context("missing prompt")?,
                ui,
                cancel,
            )
            .await
            .map(Some)
    } else {
        harness.resume_incomplete_turn(ui, cancel).await
    };
    controls.abort();
    timer.abort();
    signal.abort();
    let outcome = outcome?.context("resume produced no terminal result")?;
    // Release the managed journal lease before inspecting its durable receipt.
    drop(harness);
    let inspection = hi_harness::inspect_managed_journal(
        &session.with_extension("managed.json"),
        &request.base_url,
        key.trim(),
        &managed,
    )?;
    ui.emit("terminal",json!({"status":match outcome.stop_reason{TurnStopReason::Completed=>"completed",TurnStopReason::Cancelled if timed_out.load(Ordering::Acquire)=>"timed_out",TurnStopReason::Cancelled=>"cancelled",TurnStopReason::Error=>"requires_action"},"tests":outcome.verification.unwrap_or_else(||"missing".into()),"changed_files":outcome.changed_files,"evidence":"client_reported","recovery":inspection}));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[tokio::test]
    async fn framing_rejects_truncation_and_oversize() {
        assert!(frame(&mut BufReader::new(&b"{}"[..])).await.is_err());
        let huge = vec![b'x'; MAX_FRAME + 1];
        assert!(frame(&mut BufReader::new(huge.as_slice())).await.is_err());
        assert_eq!(
            frame(&mut BufReader::new(&b"{}\n"[..])).await.unwrap(),
            Some(b"{}\n".to_vec())
        );
    }
    #[test]
    fn no_empty_prompt_or_budget_increase() {
        let mut req:Start=serde_json::from_value(json!({"version":1,"operation":"start","run_id":uuid::Uuid::new_v4(),"workspace":"/workspace/repo","state_dir":"/state/run","credential_file":"/state/key","prompt":"fix tests"})).unwrap();
        assert!(validate(&req).is_ok());
        req.prompt = Some(String::new());
        assert!(validate(&req).is_err());
        req.operation = "resume".into();
        req.prompt = None;
        assert!(validate(&req).is_ok());
        req.run_micros = 20_000_001;
        assert!(validate(&req).is_err());
    }
    #[test]
    fn server_deadline_can_only_shorten_the_original_window() {
        let mut req:Start=serde_json::from_value(json!({"version":1,"operation":"start","run_id":uuid::Uuid::new_v4(),"workspace":"/workspace/repo","state_dir":"/state/run","credential_file":"/state/key","prompt":"fix tests","execution_seconds":30})).unwrap();
        assert_eq!(original_deadline(&req, 10_000).unwrap(), 40_000);
        req.deadline_unix_ms = Some(25_000);
        assert_eq!(original_deadline(&req, 10_000).unwrap(), 25_000);
        req.deadline_unix_ms = Some(40_001);
        assert!(original_deadline(&req, 10_000).is_err());
        req.deadline_unix_ms = Some(10_000);
        assert!(original_deadline(&req, 10_000).is_err());
    }
}
