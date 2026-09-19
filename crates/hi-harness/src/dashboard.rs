//! Grok-style agent dashboard runtime: N concurrent `Harness` rows.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, bail};
use hi_ai::Usage;
use hi_dashboard_store::{
    MemberKey, MemberKind, MemberMetadata, MemberOrigin, NewMember, SessionId, WorkspaceStore,
};
use tokio::sync::{Semaphore, mpsc};

use crate::ui::{ConfirmationResult, PermissionMode, Ui};
use crate::{Harness, HarnessConfig, TurnCancellation, TypesafeSettings};

const DEFAULT_MAX_WORKING: usize = 8;
const OPENAI_DEFAULT_BASE: &str = "https://api.openai.com/v1";

/// Pipe Network's gpt-6 id. Dashboard rows use this unless the operator
/// configured an OpenAI profile and dispatched `openai/…`.
pub const PIPE_GPT6: &str = "pipe/gpt-6";

/// True when the model id is an explicit OpenAI route (`openai/…`).
/// gpt-6 / gpt-6-astra without that prefix stay on Pipe.
pub fn is_openai_model(model: &str) -> bool {
    model.trim().to_ascii_lowercase().starts_with("openai/")
}

/// Map shorthand gpt-6 names onto Pipe. Leave `openai/…` and other ids alone.
pub fn normalize_row_model(model: &str) -> String {
    let trimmed = model.trim();
    let lower = trimmed.to_ascii_lowercase();
    if lower.starts_with("openai/") || lower.starts_with("pipe/") {
        return trimmed.to_string();
    }
    match lower.as_str() {
        "gpt-6" | "gpt-6-astra" | "astra" | "gpt6" => PIPE_GPT6.to_string(),
        _ => trimmed.to_string(),
    }
}

/// `/model <id> rest` prefix used on the dashboard dispatch box.
pub fn split_model_prefix(prompt: &str) -> (Option<String>, String) {
    let trimmed = prompt.trim();
    let rest = match trimmed
        .strip_prefix("/model ")
        .or_else(|| trimmed.strip_prefix("/m "))
    {
        Some(rest) => rest.trim(),
        None => return (None, trimmed.to_string()),
    };
    match rest.split_once(char::is_whitespace) {
        Some((id, prompt)) if !id.is_empty() => (Some(id.to_string()), prompt.trim().to_string()),
        _ if !rest.is_empty() => (Some(rest.to_string()), String::new()),
        _ => (None, trimmed.to_string()),
    }
}

/// OpenAI credentials only when the user configured them (profile / knobs).
/// Never required: missing key keeps the row on Pipe.
fn configured_openai(knobs: &DashboardKnobs) -> Option<(String, String)> {
    let key = knobs
        .openai_api_key
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string)?;
    let base = knobs
        .openai_base_url
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .unwrap_or(OPENAI_DEFAULT_BASE)
        .to_string();
    Some((key, base))
}

fn row_credentials(knobs: &DashboardKnobs, model: &str) -> (String, String) {
    if is_openai_model(model)
        && let Some((key, base)) = configured_openai(knobs)
    {
        return (key, base);
    }
    (knobs.api_key.clone(), knobs.base_url.clone())
}

fn add_row_worktree(repo: &Path, id: &str) -> Result<PathBuf> {
    if !hi_tools::worktree::in_git_repo(repo) {
        bail!("not a git repository — worktree dispatch is disabled");
    }
    let dest = std::env::temp_dir().join(format!("hi-dash-{id}"));
    if dest.exists() {
        if dest.join(".git").exists() {
            return Ok(dest);
        }
        remove_row_worktree(repo, &dest);
    }
    let output = std::process::Command::new("git")
        .arg("-C")
        .arg(repo)
        .args(["rev-parse", "HEAD"])
        .output()
        .context("git rev-parse HEAD")?;
    if !output.status.success() {
        bail!(
            "git rev-parse HEAD failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    let head = String::from_utf8_lossy(&output.stdout).trim().to_string();
    hi_tools::worktree::add_worktree(repo, &dest, &head)
        .with_context(|| format!("git worktree add {}", dest.display()))?;
    Ok(dest)
}

fn remove_row_worktree(repo: &Path, path: &Path) {
    let _ = std::process::Command::new("git")
        .arg("-C")
        .arg(repo)
        .args(["worktree", "remove", "--force"])
        .arg(path)
        .status();
    let _ = std::fs::remove_dir_all(path);
}

#[derive(Clone, Debug)]
pub struct DashboardKnobs {
    pub api_key: String,
    pub base_url: String,
    pub model: String,
    pub workspace_root: PathBuf,
    pub sessions_dir: PathBuf,
    pub store_path: PathBuf,
    pub max_working: usize,
    /// Configured OpenAI profile key. `None` or `Some("")` stays on Pipe.
    pub openai_api_key: Option<String>,
    pub openai_base_url: Option<String>,
}

impl DashboardKnobs {
    pub fn for_workspace(
        workspace_root: PathBuf,
        api_key: String,
        base_url: String,
        model: String,
    ) -> Self {
        let data = std::env::var_os("XDG_DATA_HOME")
            .map(PathBuf::from)
            .or_else(|| {
                std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".local/share"))
            })
            .unwrap_or_else(|| PathBuf::from(".").join(".hi-data"));
        let digest = {
            let key =
                std::fs::canonicalize(&workspace_root).unwrap_or_else(|_| workspace_root.clone());
            let mut hash: u64 = 0xcbf29ce484222325;
            for b in key.as_os_str().as_encoded_bytes() {
                hash ^= *b as u64;
                hash = hash.wrapping_mul(0x100000001b3);
            }
            format!("{hash:016x}")
        };
        let hi = data.join("hi");
        let project = hi.join("projects").join(digest).join("dashboard");
        let workspace_root = std::fs::canonicalize(&workspace_root).unwrap_or(workspace_root);
        Self {
            api_key,
            base_url,
            model,
            sessions_dir: project.join("sessions"),
            store_path: project.join("workspace.db"),
            workspace_root,
            max_working: DEFAULT_MAX_WORKING,
            openai_api_key: None,
            openai_base_url: None,
        }
    }

    pub fn max_working(&self) -> usize {
        std::env::var("HI_DASHBOARD_MAX_WORKING")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|n| *n > 0)
            .unwrap_or(self.max_working.max(1))
    }
}

#[derive(Clone, Debug, Default)]
pub struct DispatchOpts {
    pub model: Option<String>,
    pub worktree: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RowState {
    Idle,
    Working,
    Failed,
    Completed,
}

#[derive(Clone, Debug)]
pub struct RowView {
    pub id: String,
    pub title: String,
    pub model: String,
    pub state: RowState,
    pub last_text: String,
    pub usage: Usage,
    pub error: Option<String>,
    pub is_worktree: bool,
    pub started: Option<Instant>,
}

enum RowCmd {
    Run(String),
    Cancel,
    Shutdown,
}

struct RowHandle {
    tx: Option<mpsc::UnboundedSender<RowCmd>>,
    cancel: Arc<Mutex<Option<TurnCancellation>>>,
    view: Arc<Mutex<RowView>>,
    working: Arc<AtomicBool>,
    worktree: Option<PathBuf>,
    cwd: PathBuf,
    model: String,
}

pub struct Dashboard {
    store: WorkspaceStore,
    knobs: DashboardKnobs,
    rows: HashMap<String, RowHandle>,
    seq: AtomicU64,
    slots: Arc<Semaphore>,
}

impl Dashboard {
    pub fn open(knobs: DashboardKnobs) -> Result<Self> {
        std::fs::create_dir_all(&knobs.sessions_dir)
            .with_context(|| format!("sessions dir {}", knobs.sessions_dir.display()))?;
        let store = WorkspaceStore::open(&knobs.store_path)
            .map_err(|e| anyhow::anyhow!("dashboard store: {e}"))?;
        let max = knobs.max_working();
        let mut dash = Self {
            store,
            knobs,
            rows: HashMap::new(),
            seq: AtomicU64::new(1),
            slots: Arc::new(Semaphore::new(max)),
        };
        dash.restore_members();
        Ok(dash)
    }

    pub fn knobs(&self) -> &DashboardKnobs {
        &self.knobs
    }

    /// Refresh credentials/model from the manager session without restarting rows.
    pub fn update_session(&mut self, api_key: String, base_url: String, model: String) {
        self.knobs.api_key = api_key;
        self.knobs.base_url = base_url;
        self.knobs.model = model;
    }

    pub fn roster(&self) -> Vec<RowView> {
        let mut rows: Vec<_> = self
            .rows
            .values()
            .map(|h| h.view.lock().unwrap_or_else(|p| p.into_inner()).clone())
            .collect();
        rows.sort_by(|a, b| {
            fn rank(s: RowState) -> u8 {
                match s {
                    RowState::Working => 0,
                    RowState::Failed => 1,
                    RowState::Idle => 2,
                    RowState::Completed => 3,
                }
            }
            rank(a.state)
                .cmp(&rank(b.state))
                .then_with(|| a.id.cmp(&b.id))
        });
        rows
    }

    pub fn working_count(&self) -> usize {
        self.rows
            .values()
            .filter(|h| h.working.load(Ordering::Acquire))
            .count()
    }

    pub fn dispatch(&mut self, prompt: &str, model: Option<String>) -> Result<String> {
        self.dispatch_with(
            prompt,
            DispatchOpts {
                model,
                worktree: false,
            },
        )
    }

    pub fn dispatch_with(&mut self, prompt: &str, mut opts: DispatchOpts) -> Result<String> {
        let (prefix_model, body) = split_model_prefix(prompt);
        if prefix_model.is_some() {
            opts.model = prefix_model;
        }
        let prompt = body.trim();
        self.spawn_row(
            if prompt.is_empty() {
                None
            } else {
                Some(prompt)
            },
            opts,
        )
    }

    /// Create a roster row. `prompt = None` is an untitled idle agent (Grok + New Agent).
    pub fn spawn_row(&mut self, prompt: Option<&str>, opts: DispatchOpts) -> Result<String> {
        let n = self.seq.fetch_add(1, Ordering::Relaxed);
        let millis = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis();
        let raw = format!("dash-{millis}-{n}");
        let session_id = SessionId::new(&raw).map_err(|e| anyhow::anyhow!("{e}"))?;
        let model = normalize_row_model(
            &opts
                .model
                .clone()
                .unwrap_or_else(|| self.knobs.model.clone()),
        );
        let title: String = prompt
            .map(|p| p.chars().take(72).collect())
            .filter(|s: &String| !s.is_empty())
            .unwrap_or_else(|| "untitled".into());

        let worktree_path = if opts.worktree {
            Some(add_row_worktree(&self.knobs.workspace_root, &raw)?)
        } else {
            None
        };
        let cwd = worktree_path
            .clone()
            .unwrap_or_else(|| self.knobs.workspace_root.clone());
        let cwd = cwd.canonicalize().unwrap_or(cwd);

        let key = MemberKey {
            session_id: session_id.clone(),
            kind: MemberKind::Build,
        };
        let now_ms = millis as i64;
        self.store
            .insert_member(NewMember {
                key,
                origin: MemberOrigin::Local,
                metadata: MemberMetadata {
                    cwd: Some(cwd.to_string_lossy().into_owned()),
                    title: Some(title.clone()),
                    model: Some(model.clone()),
                    last_turn_summary: None,
                    is_worktree: worktree_path.is_some(),
                    last_change_unix_ms: now_ms,
                },
            })
            .map_err(|e| anyhow::anyhow!("insert member: {e}"))?;

        let view = Arc::new(Mutex::new(RowView {
            id: raw.clone(),
            title,
            model: model.clone(),
            state: RowState::Idle,
            last_text: String::new(),
            usage: Usage::default(),
            error: None,
            is_worktree: worktree_path.is_some(),
            started: None,
        }));
        if let Err(err) = self.start_live_row(raw.clone(), cwd, model, view, worktree_path, None) {
            let _ = self.remove(&raw);
            return Err(err);
        }
        if let Some(prompt) = prompt.filter(|p| !p.is_empty()) {
            self.rows
                .get(&raw)
                .and_then(|h| h.tx.as_ref())
                .ok_or_else(|| anyhow::anyhow!("dashboard worker closed"))?
                .send(RowCmd::Run(prompt.to_string()))
                .map_err(|_| anyhow::anyhow!("dashboard worker closed"))?;
        }
        Ok(raw)
    }

    fn restore_members(&mut self) {
        let Ok(snap) = self.store.snapshot() else {
            return;
        };
        for member in snap.members {
            let id = member.session_id.as_ref().to_string();
            if self.rows.contains_key(&id) {
                continue;
            }
            let model = normalize_row_model(member.model.as_deref().unwrap_or(&self.knobs.model));
            let cwd = member
                .cwd
                .as_deref()
                .map(PathBuf::from)
                .unwrap_or_else(|| self.knobs.workspace_root.clone());
            let cwd = cwd.canonicalize().unwrap_or(cwd);
            let session_path = self.knobs.sessions_dir.join(format!("{id}.jsonl"));
            let loaded = crate::JsonlSession::load(&session_path).ok();
            let last_text = loaded
                .as_ref()
                .and_then(|s| {
                    s.messages.iter().rev().find_map(|m| {
                        if matches!(m.role, hi_ai::Role::Assistant) {
                            let t: String = m
                                .content
                                .iter()
                                .filter_map(|c| match c {
                                    hi_ai::Content::Text(text) => Some(text.as_str()),
                                    _ => None,
                                })
                                .collect();
                            (!t.is_empty()).then_some(t)
                        } else {
                            None
                        }
                    })
                })
                .or(member.last_turn_summary.clone())
                .unwrap_or_default();
            let usage = loaded.as_ref().map(|s| s.usage).unwrap_or_default();
            let view = Arc::new(Mutex::new(RowView {
                id: id.clone(),
                title: member.title.clone().unwrap_or_else(|| "untitled".into()),
                model: model.clone(),
                state: RowState::Idle,
                last_text,
                usage,
                error: None,
                is_worktree: member.is_worktree,
                started: None,
            }));
            let worktree = member.is_worktree.then_some(cwd.clone());
            // Dormant until the user replies — restoring 50 sessions must not
            // spawn 50 runtimes on /dashboard.
            self.rows.insert(
                id,
                RowHandle {
                    tx: None,
                    cancel: Arc::new(Mutex::new(None)),
                    view,
                    working: Arc::new(AtomicBool::new(false)),
                    worktree,
                    cwd,
                    model,
                },
            );
        }
    }

    fn start_live_row(
        &mut self,
        id: String,
        cwd: PathBuf,
        model: String,
        view: Arc<Mutex<RowView>>,
        worktree_path: Option<PathBuf>,
        loaded: Option<crate::LoadedSession>,
    ) -> Result<()> {
        let working = Arc::new(AtomicBool::new(false));
        let cancel_slot = Arc::new(Mutex::new(None));
        self.rows.insert(
            id.clone(),
            RowHandle {
                tx: None,
                cancel: cancel_slot,
                view,
                working,
                worktree: worktree_path,
                cwd,
                model,
            },
        );
        self.ensure_worker(&id, loaded)
    }

    fn ensure_worker(&mut self, id: &str, loaded: Option<crate::LoadedSession>) -> Result<()> {
        let row = self
            .rows
            .get(id)
            .ok_or_else(|| anyhow::anyhow!("unknown dashboard row {id}"))?;
        if row.tx.is_some() {
            return Ok(());
        }
        let cwd = row.cwd.clone();
        let model = row.model.clone();
        let view = row.view.clone();
        let working = row.working.clone();
        let cancel_slot = row.cancel.clone();
        let (api_key, base_url) = row_credentials(&self.knobs, &model);
        let session_path = self.knobs.sessions_dir.join(format!("{id}.jsonl"));
        let loaded = loaded.or_else(|| crate::JsonlSession::load(&session_path).ok());
        let mut config = HarnessConfig::pipe(cwd, api_key);
        config.base_url = base_url;
        config.model = model;
        config.session_path = Some(session_path);
        config.typesafe = TypesafeSettings::from_env();
        let mut harness = Harness::new(config)?;
        harness.set_permission_mode(PermissionMode::Always);
        if let Some(loaded) = loaded {
            harness.apply_loaded_session(loaded);
        }
        let (tx, rx) = mpsc::unbounded_channel();
        let slots = self.slots.clone();
        let thread_id = id.to_string();
        std::thread::Builder::new()
            .name(format!("hi-dash-{thread_id}"))
            .spawn(move || {
                let rt = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .expect("dashboard row runtime");
                rt.block_on(row_worker(harness, rx, view, working, cancel_slot, slots));
            })
            .context("spawn dashboard row thread")?;
        if let Some(row) = self.rows.get_mut(id) {
            row.tx = Some(tx);
        }
        Ok(())
    }

    pub fn reply(&mut self, id: &str, prompt: &str) -> Result<()> {
        let prompt = prompt.trim();
        anyhow::ensure!(!prompt.is_empty(), "empty reply");
        self.ensure_worker(id, None)?;
        let row = self
            .rows
            .get(id)
            .ok_or_else(|| anyhow::anyhow!("unknown dashboard row {id}"))?;
        let tx = row
            .tx
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("row {id} is not running"))?;
        tx.send(RowCmd::Run(prompt.to_string()))
            .map_err(|_| anyhow::anyhow!("dashboard worker closed"))?;
        Ok(())
    }

    pub fn cancel(&self, id: &str) -> Result<()> {
        let row = self
            .rows
            .get(id)
            .ok_or_else(|| anyhow::anyhow!("unknown dashboard row {id}"))?;
        if let Some(cancel) = row
            .cancel
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .as_ref()
        {
            cancel.cancel();
        }
        if let Some(tx) = &row.tx {
            let _ = tx.send(RowCmd::Cancel);
        }
        Ok(())
    }

    pub fn remove(&mut self, id: &str) -> Result<()> {
        if let Some(row) = self.rows.remove(id) {
            if let Some(tx) = &row.tx {
                let _ = tx.send(RowCmd::Shutdown);
            }
            if let Some(cancel) = row
                .cancel
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .as_ref()
            {
                cancel.cancel();
            }
            if let Some(path) = row.worktree {
                remove_row_worktree(&self.knobs.workspace_root, &path);
            }
        }
        let key = MemberKey {
            session_id: SessionId::new(id).map_err(|e| anyhow::anyhow!("{e}"))?,
            kind: MemberKind::Build,
        };
        let _ = self.store.remove_member(&key);
        Ok(())
    }
}

impl Drop for Dashboard {
    fn drop(&mut self) {
        for row in self.rows.values() {
            if let Some(tx) = &row.tx {
                let _ = tx.send(RowCmd::Shutdown);
            }
            if let Some(cancel) = row
                .cancel
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .as_ref()
            {
                cancel.cancel();
            }
        }
    }
}

struct RowUi {
    view: Arc<Mutex<RowView>>,
}

impl RowUi {
    fn update(&self, f: impl FnOnce(&mut RowView)) {
        f(&mut self.view.lock().unwrap_or_else(|p| p.into_inner()));
    }
}

impl Ui for RowUi {
    fn assistant_text(&mut self, text: &str) {
        self.update(|v| v.last_text.push_str(text));
    }
    fn assistant_reasoning(&mut self, _text: &str) {}
    fn assistant_end(&mut self) {}
    fn tool_call(&mut self, name: &str, _arguments: &str) {
        self.update(|v| {
            v.last_text = format!("→ {name}");
        });
    }
    fn tool_result(&mut self, _name: &str, _result: &str) {}
    fn status(&mut self, text: &str) {
        self.update(|v| {
            if !text.is_empty() {
                v.last_text = text.to_string();
            }
        });
    }
    fn turn_end(&mut self, summary: &str) {
        self.update(|v| {
            if v.last_text.is_empty() {
                v.last_text = summary.to_string();
            }
            v.state = RowState::Completed;
        });
    }
    fn turn_error(&mut self, error_kind: &str, message: &str, _guidance: &str) {
        self.update(|v| {
            v.state = RowState::Failed;
            v.error = Some(format!("{error_kind}: {message}"));
        });
    }
    fn confirm(&mut self, _request: crate::ConfirmationRequest) -> crate::ConfirmationFuture<'_> {
        Box::pin(async { ConfirmationResult::Approved })
    }
}

async fn row_worker(
    mut harness: Harness,
    mut rx: mpsc::UnboundedReceiver<RowCmd>,
    view: Arc<Mutex<RowView>>,
    working: Arc<AtomicBool>,
    cancel_slot: Arc<Mutex<Option<TurnCancellation>>>,
    slots: Arc<Semaphore>,
) {
    while let Some(cmd) = rx.recv().await {
        match cmd {
            RowCmd::Shutdown => break,
            RowCmd::Cancel => {
                if let Some(cancel) = cancel_slot
                    .lock()
                    .unwrap_or_else(|p| p.into_inner())
                    .as_ref()
                {
                    cancel.cancel();
                }
            }
            RowCmd::Run(prompt) => {
                let cancel = TurnCancellation::new();
                *cancel_slot.lock().unwrap_or_else(|p| p.into_inner()) = Some(cancel.clone());
                let permit = tokio::select! {
                    biased;
                    _ = cancel.cancelled() => None,
                    permit = slots.acquire() => permit.ok(),
                };
                let Some(permit) = permit else {
                    *cancel_slot.lock().unwrap_or_else(|p| p.into_inner()) = None;
                    continue;
                };
                if cancel.is_cancelled() {
                    drop(permit);
                    *cancel_slot.lock().unwrap_or_else(|p| p.into_inner()) = None;
                    continue;
                }
                working.store(true, Ordering::Release);
                {
                    let mut v = view.lock().unwrap_or_else(|p| p.into_inner());
                    v.state = RowState::Working;
                    v.error = None;
                    v.started = Some(Instant::now());
                    v.last_text.clear();
                }
                let mut ui = RowUi { view: view.clone() };
                let result = harness.run_turn_cancellable(&prompt, &mut ui, cancel).await;
                drop(permit);
                *cancel_slot.lock().unwrap_or_else(|p| p.into_inner()) = None;
                working.store(false, Ordering::Release);
                let mut v = view.lock().unwrap_or_else(|p| p.into_inner());
                v.usage = harness.session_usage();
                match result {
                    Ok(_) if v.state != RowState::Failed => v.state = RowState::Idle,
                    Ok(_) => {}
                    Err(err) => {
                        v.state = RowState::Failed;
                        v.error = Some(err.to_string());
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipe::test_support::{MockPipe, Scripted, text_chunk, usage_chunk};

    fn knobs(dir: &Path, url: &str) -> DashboardKnobs {
        DashboardKnobs {
            api_key: "pk_test".into(),
            base_url: url.to_string(),
            model: "pipe/test".into(),
            workspace_root: dir.to_path_buf(),
            sessions_dir: dir.join("sessions"),
            store_path: dir.join("dashboard").join("workspace.db"),
            max_working: DEFAULT_MAX_WORKING,
            openai_api_key: None,
            openai_base_url: None,
        }
    }

    fn git_ok(dir: &Path, args: &[&str]) {
        let status = std::process::Command::new("git")
            .args(args)
            .current_dir(dir)
            .status()
            .unwrap();
        assert!(status.success(), "git {args:?}");
    }

    #[test]
    fn model_prefix_normalizes_gpt6_onto_pipe() {
        assert_eq!(
            split_model_prefix("/model gpt-6-astra review this"),
            (Some("gpt-6-astra".into()), "review this".into())
        );
        assert_eq!(
            split_model_prefix("just a prompt"),
            (None, "just a prompt".into())
        );
        assert_eq!(normalize_row_model("gpt-6"), PIPE_GPT6);
        assert_eq!(normalize_row_model("gpt-6-astra"), PIPE_GPT6);
        assert_eq!(normalize_row_model("pipe/gpt-6"), "pipe/gpt-6");
        assert_eq!(
            normalize_row_model("openai/gpt-6-astra"),
            "openai/gpt-6-astra"
        );
        assert!(!is_openai_model("gpt-6-astra"));
        assert!(!is_openai_model(PIPE_GPT6));
        assert!(is_openai_model("openai/gpt-4o"));
        assert!(!is_openai_model("pipe/deepseek-v4-flash-0731"));
    }

    #[test]
    fn gpt6_without_openai_config_stays_on_pipe() {
        let dir = tempfile::tempdir().unwrap();
        let mut k = knobs(dir.path(), "http://127.0.0.1:9");
        k.openai_api_key = Some(String::new());
        let mut dash = Dashboard::open(k).unwrap();
        let id = dash
            .dispatch_with(
                "review login",
                DispatchOpts {
                    model: Some("gpt-6-astra".into()),
                    worktree: false,
                },
            )
            .unwrap();
        let row = dash.roster().into_iter().find(|r| r.id == id).unwrap();
        assert_eq!(row.state, RowState::Idle);
        assert!(row.error.is_none(), "must not fail without OPENAI_API_KEY");
        assert_eq!(row.model, PIPE_GPT6);
        assert!(dash.reply(&id, "more").is_ok());
        let _ = dash.remove(&id);
    }

    #[test]
    fn configured_openai_is_used_only_for_openai_prefix() {
        let dir = tempfile::tempdir().unwrap();
        let mut k = knobs(dir.path(), "http://pipe.example");
        k.openai_api_key = Some("sk-test".into());
        k.openai_base_url = Some("https://api.openai.com/v1".into());
        let (key, base) = row_credentials(&k, PIPE_GPT6);
        assert_eq!(key, "pk_test");
        assert_eq!(base, "http://pipe.example");
        let (key, base) = row_credentials(&k, "openai/gpt-6-astra");
        assert_eq!(key, "sk-test");
        assert_eq!(base, "https://api.openai.com/v1");
        k.openai_api_key = None;
        let (key, base) = row_credentials(&k, "openai/gpt-6-astra");
        assert_eq!(key, "pk_test");
        assert_eq!(base, "http://pipe.example");
    }

    #[test]
    fn open_restores_store_members() {
        let dir = tempfile::tempdir().unwrap();
        let k = knobs(dir.path(), "http://127.0.0.1:9");
        let mut dash = Dashboard::open(k.clone()).unwrap();
        let id = dash
            .spawn_row(
                None,
                DispatchOpts {
                    model: Some("gpt-6".into()),
                    worktree: false,
                },
            )
            .unwrap();
        drop(dash);
        let dash = Dashboard::open(k).unwrap();
        let row = dash
            .roster()
            .into_iter()
            .find(|r| r.id == id)
            .expect("restored");
        assert_eq!(row.model, PIPE_GPT6);
        assert_eq!(row.state, RowState::Idle);
        // Reboot recovery: restore rehydrates rows as Idle and does not spawn runtimes.
        let _ = dash;
    }

    #[test]
    fn store_path_is_per_workspace() {
        let a = tempfile::tempdir().unwrap();
        let b = tempfile::tempdir().unwrap();
        let ka = DashboardKnobs::for_workspace(
            a.path().to_path_buf(),
            "k".into(),
            "http://x".into(),
            "m".into(),
        );
        let kb = DashboardKnobs::for_workspace(
            b.path().to_path_buf(),
            "k".into(),
            "http://x".into(),
            "m".into(),
        );
        assert_ne!(ka.store_path, kb.store_path);
        assert!(
            ka.store_path.to_string_lossy().contains("projects"),
            "{}",
            ka.store_path.display()
        );
        let path = ka.store_path.to_string_lossy();
        assert!(
            path.contains("/dashboard/workspace.db") || path.ends_with("dashboard/workspace.db"),
            "{path}"
        );
        assert!(
            ka.sessions_dir
                .to_string_lossy()
                .contains("/dashboard/sessions")
                || ka.sessions_dir.ends_with("dashboard/sessions"),
            "{}",
            ka.sessions_dir.display()
        );
    }

    #[tokio::test]
    async fn restored_row_accepts_a_reply() {
        let Some(server) = MockPipe::new(vec![
            Scripted::Sse(vec![text_chunk("ready"), usage_chunk(2, 1)]),
            Scripted::Sse(vec![text_chunk("queued"), usage_chunk(2, 1)]),
        ]) else {
            return;
        };
        let dir = tempfile::tempdir().unwrap();
        let k = knobs(dir.path(), &server.url);
        let mut dash = Dashboard::open(k.clone()).unwrap();
        let id = dash
            .spawn_row(
                None,
                DispatchOpts {
                    model: None,
                    worktree: false,
                },
            )
            .unwrap();
        drop(dash);
        let mut dash = Dashboard::open(k).unwrap();
        dash.reply(&id, "do more").unwrap();
        tokio::time::sleep(std::time::Duration::from_millis(1500)).await;
        let row = dash.roster().into_iter().find(|r| r.id == id).unwrap();
        assert!(
            row.usage.input_tokens > 0 || row.last_text.contains("queued"),
            "restored row should run a reply: {row:?}"
        );
        let _ = dash.remove(&id);
    }

    #[test]
    fn worktree_dispatch_is_isolated() {
        let dir = tempfile::tempdir().unwrap();
        git_ok(dir.path(), &["init"]);
        git_ok(dir.path(), &["config", "user.email", "hi@example.com"]);
        git_ok(dir.path(), &["config", "user.name", "hi"]);
        std::fs::write(dir.path().join("README.md"), "hi\n").unwrap();
        git_ok(dir.path(), &["add", "README.md"]);
        git_ok(dir.path(), &["commit", "-qm", "init"]);
        let mut dash = Dashboard::open(knobs(dir.path(), "http://127.0.0.1:9")).unwrap();
        let id = dash
            .spawn_row(
                None,
                DispatchOpts {
                    model: None,
                    worktree: true,
                },
            )
            .unwrap();
        let row = dash.roster().into_iter().find(|r| r.id == id).unwrap();
        assert!(row.is_worktree, "row should be marked worktree");
        let wt = dash
            .rows
            .get(&id)
            .and_then(|h| h.worktree.clone())
            .expect("worktree path");
        assert_ne!(&wt, dir.path());
        assert!(wt.join("README.md").is_file());
        let _ = dash.remove(&id);
        assert!(!wt.exists(), "worktree cleaned on remove");
    }

    #[tokio::test]
    async fn dispatch_two_rows_and_queue_a_reply() {
        let Some(server) = MockPipe::new(vec![
            Scripted::Sse(vec![text_chunk("one"), usage_chunk(3, 1)]),
            Scripted::Sse(vec![text_chunk("two"), usage_chunk(4, 2)]),
            Scripted::Sse(vec![text_chunk("queued"), usage_chunk(2, 1)]),
        ]) else {
            return;
        };
        let dir = tempfile::tempdir().unwrap();
        let mut dash = Dashboard::open(knobs(dir.path(), &server.url)).unwrap();
        let a = dash.dispatch("fix a", None).unwrap();
        let b = dash.dispatch("fix b", None).unwrap();
        tokio::time::sleep(std::time::Duration::from_millis(800)).await;
        let roster = dash.roster();
        assert_eq!(roster.len(), 2);
        assert!(roster.iter().any(|r| r.id == a));
        assert!(roster.iter().any(|r| r.id == b));
        dash.reply(&a, "do more").unwrap();
        tokio::time::sleep(std::time::Duration::from_millis(800)).await;
        let a_view = dash
            .roster()
            .into_iter()
            .find(|r| r.id == a)
            .expect("row a");
        assert!(
            a_view.usage.input_tokens > 0
                || a_view.last_text.contains("one")
                || a_view.last_text.contains("queued"),
            "row a should have run: {a_view:?}"
        );
        let _ = dash.remove(&a);
        let _ = dash.remove(&b);
    }
}
