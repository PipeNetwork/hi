//! In-memory snapshot the harness updates and the writer thread copies.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError, TryLockError};

use crate::events::EventLog;
use crate::schema::{
    EventCode, HarnessState, Heartbeat, IDENTICAL_TOOL_CONSECUTIVE, IDENTICAL_TOOL_ERROR_REPEAT,
    IDENTICAL_TOOL_IN_TURN, InvariantCode, InvariantViolation, LiveEvent, SCHEMA_VERSION, unix_ms,
};

/// Live child pgids sampled each heartbeat beat: `(child_pgids, current_tool_pgid)`.
pub type PgidSource = Arc<dyn Fn() -> (Vec<i32>, Option<i32>) + Send + Sync>;

#[derive(Clone)]
pub struct Publisher {
    inner: Arc<Mutex<Inner>>,
    last_good: Arc<Mutex<Inner>>,
    pgid_source: Arc<Mutex<Option<PgidSource>>>,
}

#[derive(Clone, Debug)]
struct Inner {
    state: HarnessState,
    last_progress_unix_ms: u64,
    last_event: Option<String>,
    last_tool: Option<String>,
    last_tool_id: Option<String>,
    consecutive_identical_tools: u32,
    identical_tool_count_in_turn: u32,
    turn_index: u32,
    session_path: Option<String>,
    workspace: String,
    pre_checkpoint: Option<String>,
    child_pgids: Vec<i32>,
    current_tool_pgid: Option<i32>,
    invariant: Option<InvariantViolation>,
    open_tool: Option<(String, String)>,
    last_fingerprint: Option<String>,
    in_turn: HashMap<String, u32>,
    last_error: Option<String>,
    error_repeat: u32,
    events: Option<EventLog>,
}

impl Default for Inner {
    fn default() -> Self {
        let now = unix_ms();
        Self {
            state: HarnessState::Starting,
            last_progress_unix_ms: now,
            last_event: None,
            last_tool: None,
            last_tool_id: None,
            consecutive_identical_tools: 0,
            identical_tool_count_in_turn: 0,
            turn_index: 0,
            session_path: None,
            workspace: String::new(),
            pre_checkpoint: None,
            child_pgids: Vec::new(),
            current_tool_pgid: None,
            invariant: None,
            open_tool: None,
            last_fingerprint: None,
            in_turn: HashMap::new(),
            last_error: None,
            error_repeat: 0,
            events: None,
        }
    }
}

impl Default for Publisher {
    fn default() -> Self {
        Self::new()
    }
}

impl Publisher {
    pub fn new() -> Self {
        let inner = Inner::default();
        Self {
            inner: Arc::new(Mutex::new(inner.clone())),
            last_good: Arc::new(Mutex::new(inner)),
            pgid_source: Arc::new(Mutex::new(None)),
        }
    }

    pub fn set_pgid_source(&self, source: PgidSource) {
        *self
            .pgid_source
            .lock()
            .unwrap_or_else(PoisonError::into_inner) = Some(source);
    }

    pub fn set_event_log(&self, log: EventLog) {
        self.lock().events = Some(log);
    }

    pub fn set_state(&self, state: HarnessState) {
        let mut g = self.lock();
        if g.state != state {
            g.state = state;
            touch_progress(&mut g);
            emit_locked(&mut g, EventCode::State, None, None, None);
        }
    }

    pub fn state(&self) -> HarnessState {
        self.lock().state
    }

    pub fn note_progress(&self) {
        touch_progress(&mut self.lock());
    }

    pub fn set_last_event(&self, code: &str) {
        self.lock().last_event = Some(code.to_string());
    }

    pub fn set_workspace(&self, workspace: impl Into<String>) {
        self.lock().workspace = workspace.into();
    }

    pub fn set_session_path(&self, path: Option<String>) {
        self.lock().session_path = path;
    }

    pub fn set_turn_index(&self, turn_index: u32) {
        self.lock().turn_index = turn_index;
    }

    pub fn set_pre_checkpoint(&self, id: Option<String>) {
        self.lock().pre_checkpoint = id;
    }

    pub fn set_child_pgids(&self, pgids: Vec<i32>, current: Option<i32>) {
        let mut g = self.lock();
        g.child_pgids = pgids;
        g.current_tool_pgid = current;
    }

    pub fn note_tool_start(&self, id: &str, name: &str) {
        let mut g = self.lock();
        if g.open_tool.is_some() {
            set_invariant(&mut g, InvariantCode::ToolUnclosed);
        }
        g.open_tool = Some((id.to_string(), name.to_string()));
        g.last_tool = Some(name.to_string());
        g.last_tool_id = Some(id.to_string());
        g.state = HarnessState::ExecutingTool;
        touch_progress(&mut g);
        emit_locked(&mut g, EventCode::ToolStart, Some(name), Some(id), None);
    }

    pub fn note_tool_end(&self, leaked: bool) {
        let mut g = self.lock();
        if leaked {
            set_invariant(&mut g, InvariantCode::ChildLeak);
        }
        let (id, name) = g.open_tool.take().unwrap_or_default();
        touch_progress(&mut g);
        emit_locked(
            &mut g,
            EventCode::ToolEnd,
            Some(name.as_str()).filter(|s| !s.is_empty()),
            Some(id.as_str()).filter(|s| !s.is_empty()),
            None,
        );
    }

    pub fn note_tool_unclosed(&self) {
        set_invariant(&mut self.lock(), InvariantCode::ToolUnclosed);
    }

    pub fn reset_turn_tools(&self) {
        self.finish_turn_tools(false);
    }

    pub fn finish_turn_tools(&self, report_unclosed: bool) {
        let mut g = self.lock();
        if report_unclosed && g.open_tool.is_some() {
            set_invariant(&mut g, InvariantCode::ToolUnclosed);
        }
        g.consecutive_identical_tools = 0;
        g.identical_tool_count_in_turn = 0;
        g.last_fingerprint = None;
        g.in_turn.clear();
        g.last_error = None;
        g.error_repeat = 0;
        g.open_tool = None;
    }

    pub fn note_tool_fingerprint(&self, fingerprint: &str) {
        let mut g = self.lock();
        if g.last_fingerprint.as_deref() == Some(fingerprint) {
            g.consecutive_identical_tools = g.consecutive_identical_tools.saturating_add(1);
        } else {
            g.consecutive_identical_tools = 1;
            g.last_fingerprint = Some(fingerprint.to_string());
        }
        let count = g.in_turn.entry(fingerprint.to_string()).or_insert(0);
        *count = count.saturating_add(1);
        g.identical_tool_count_in_turn = g.in_turn.values().copied().max().unwrap_or(0);
        let storm = g.consecutive_identical_tools >= IDENTICAL_TOOL_CONSECUTIVE
            || g.identical_tool_count_in_turn >= IDENTICAL_TOOL_IN_TURN;
        if storm {
            set_invariant(&mut g, InvariantCode::IdenticalToolStorm);
        }
    }

    pub fn note_tool_error(&self, error: &str) {
        let prefix: String = error.chars().take(200).collect();
        let mut g = self.lock();
        if g.last_error.as_deref() == Some(prefix.as_str()) {
            g.error_repeat = g.error_repeat.saturating_add(1);
        } else {
            g.error_repeat = 1;
            g.last_error = Some(prefix);
        }
        if g.error_repeat >= IDENTICAL_TOOL_ERROR_REPEAT {
            set_invariant(&mut g, InvariantCode::IdenticalToolStorm);
        }
    }

    pub fn set_invariant(&self, code: InvariantCode) {
        set_invariant(&mut self.lock(), code);
    }

    pub fn emit(
        &self,
        code: EventCode,
        tool: Option<&str>,
        tool_id: Option<&str>,
        detail: Option<&str>,
    ) {
        emit_locked(&mut self.lock(), code, tool, tool_id, detail);
    }

    pub fn snapshot(&self) -> Heartbeat {
        let sampled = self.read_pgids();
        let mut g = self.lock();
        if let Some((pgids, current)) = &sampled {
            apply_pgid_delta(&mut g, pgids, *current);
        }
        g.as_heartbeat(0, unix_ms(), std::process::id(), "", 0)
    }

    /// Copy for the writer thread. Never blocks on a contended harness lock:
    /// skip the copy and reuse the last successful snapshot, still allowing
    /// `seq` to advance. Child pgids are sampled from the live source even
    /// when the snapshot mutex is busy.
    pub fn copy_for_write(&self) -> InnerView {
        let sampled = self.read_pgids();
        match self.inner.try_lock() {
            Ok(mut g) => {
                if let Some((pgids, current)) = &sampled {
                    apply_pgid_delta(&mut g, pgids, *current);
                }
                let view = InnerView::from(&*g);
                if let Ok(mut last) = self.last_good.try_lock() {
                    *last = g.clone();
                }
                view
            }
            Err(TryLockError::WouldBlock | TryLockError::Poisoned(_)) => {
                let mut view = InnerView::from(
                    &*self
                        .last_good
                        .lock()
                        .unwrap_or_else(PoisonError::into_inner),
                );
                if let Some((pgids, current)) = sampled {
                    view.child_pgids = pgids;
                    view.current_tool_pgid = current;
                }
                view
            }
        }
    }

    fn read_pgids(&self) -> Option<(Vec<i32>, Option<i32>)> {
        let source = self
            .pgid_source
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()?;
        Some(source())
    }

    fn lock(&self) -> MutexGuard<'_, Inner> {
        self.inner.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

#[derive(Clone, Debug)]
pub struct InnerView {
    pub state: HarnessState,
    pub last_progress_unix_ms: u64,
    pub last_event: Option<String>,
    pub last_tool: Option<String>,
    pub last_tool_id: Option<String>,
    pub consecutive_identical_tools: u32,
    pub identical_tool_count_in_turn: u32,
    pub turn_index: u32,
    pub session_path: Option<String>,
    pub workspace: String,
    pub pre_checkpoint: Option<String>,
    pub child_pgids: Vec<i32>,
    pub current_tool_pgid: Option<i32>,
    pub invariant: Option<InvariantViolation>,
}

impl From<&Inner> for InnerView {
    fn from(g: &Inner) -> Self {
        Self {
            state: g.state,
            last_progress_unix_ms: g.last_progress_unix_ms,
            last_event: g.last_event.clone(),
            last_tool: g.last_tool.clone(),
            last_tool_id: g.last_tool_id.clone(),
            consecutive_identical_tools: g.consecutive_identical_tools,
            identical_tool_count_in_turn: g.identical_tool_count_in_turn,
            turn_index: g.turn_index,
            session_path: g.session_path.clone(),
            workspace: g.workspace.clone(),
            pre_checkpoint: g.pre_checkpoint.clone(),
            child_pgids: g.child_pgids.clone(),
            current_tool_pgid: g.current_tool_pgid,
            invariant: g.invariant.clone(),
        }
    }
}

impl InnerView {
    pub fn into_heartbeat(
        self,
        seq: u64,
        ts_unix_ms: u64,
        pid: u32,
        instance: &str,
        generation: u32,
    ) -> Heartbeat {
        Heartbeat {
            schema_version: SCHEMA_VERSION,
            seq,
            ts_unix_ms,
            pid,
            instance: instance.to_string(),
            generation,
            state: self.state,
            last_progress_unix_ms: self.last_progress_unix_ms,
            last_event: self.last_event,
            last_tool: self.last_tool,
            last_tool_id: self.last_tool_id,
            consecutive_identical_tools: self.consecutive_identical_tools,
            identical_tool_count_in_turn: self.identical_tool_count_in_turn,
            turn_index: self.turn_index,
            session_path: self.session_path,
            workspace: self.workspace,
            pre_checkpoint: self.pre_checkpoint,
            child_pgids: self.child_pgids,
            current_tool_pgid: self.current_tool_pgid,
            invariant: self.invariant,
        }
    }
}

impl Inner {
    fn as_heartbeat(
        &self,
        seq: u64,
        ts_unix_ms: u64,
        pid: u32,
        instance: &str,
        generation: u32,
    ) -> Heartbeat {
        InnerView::from(self).into_heartbeat(seq, ts_unix_ms, pid, instance, generation)
    }
}

fn apply_pgid_delta(g: &mut Inner, pgids: &[i32], current: Option<i32>) {
    let spawned = pgids.iter().any(|pgid| !g.child_pgids.contains(pgid));
    let waited = g.child_pgids.iter().any(|pgid| !pgids.contains(pgid));
    if spawned {
        touch_progress(g);
        emit_locked(g, EventCode::ChildSpawn, None, None, None);
    }
    if waited {
        touch_progress(g);
        emit_locked(g, EventCode::ChildWait, None, None, None);
    }
    g.child_pgids = pgids.to_vec();
    g.current_tool_pgid = current;
}

fn touch_progress(g: &mut Inner) {
    g.last_progress_unix_ms = unix_ms();
}

fn set_invariant(g: &mut Inner, code: InvariantCode) {
    if g.invariant.is_some() {
        return;
    }
    g.invariant = Some(InvariantViolation {
        code,
        ts_unix_ms: unix_ms(),
        detail: None,
    });
    emit_locked(
        &mut *g,
        EventCode::Invariant,
        None,
        None,
        Some(code.as_str()),
    );
}

fn emit_locked(
    g: &mut Inner,
    code: EventCode,
    tool: Option<&str>,
    tool_id: Option<&str>,
    detail: Option<&str>,
) {
    g.last_event = Some(code.as_str().to_string());
    let event = LiveEvent {
        ts_unix_ms: unix_ms(),
        code,
        state: g.state,
        tool: tool.map(str::to_string),
        tool_id: tool_id.map(str::to_string),
        detail: detail.map(str::to_string),
    };
    if let Some(log) = g.events.clone() {
        log.append(&event);
    }
}
