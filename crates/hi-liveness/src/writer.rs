//! Dedicated OS thread that atomically publishes `heartbeat.json`.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use crate::atomic::write_atomic_json;
use crate::events::EventLog;
use crate::publisher::Publisher;
use crate::schema::{
    ENV_EVENTS, ENV_GENERATION, ENV_HEARTBEAT, ENV_INSTANCE, ENV_SUPERVISED, EventCode,
    HEARTBEAT_PERIOD_MS, HarnessState, Heartbeat, env_flag_on, unix_ms, writer_may_publish,
};

#[derive(Clone, Debug)]
pub struct WriterConfig {
    pub heartbeat_path: PathBuf,
    pub events_path: Option<PathBuf>,
    pub instance: String,
    pub generation: u32,
    pub workspace: String,
    pub session_path: Option<String>,
    pub period: Duration,
}

impl WriterConfig {
    pub fn from_env() -> Option<Self> {
        let supervised = std::env::var(ENV_SUPERVISED).ok()?;
        if !env_flag_on(&supervised) {
            return None;
        }
        let heartbeat_path = PathBuf::from(std::env::var_os(ENV_HEARTBEAT)?);
        let events_path = std::env::var_os(ENV_EVENTS).map(PathBuf::from);
        let instance = std::env::var(ENV_INSTANCE).unwrap_or_default();
        let generation = std::env::var(ENV_GENERATION)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0);
        let workspace = std::env::current_dir()
            .ok()
            .map(|p| p.display().to_string())
            .unwrap_or_default();
        Some(Self {
            heartbeat_path,
            events_path,
            instance,
            generation,
            workspace,
            session_path: None,
            period: Duration::from_millis(HEARTBEAT_PERIOD_MS),
        })
    }
}

pub struct WriterHandle {
    shutdown: Arc<(Mutex<bool>, Condvar)>,
    thread: Option<JoinHandle<()>>,
    publisher: Publisher,
}

impl WriterHandle {
    pub fn publisher(&self) -> Publisher {
        self.publisher.clone()
    }
}

impl Drop for WriterHandle {
    fn drop(&mut self) {
        self.publisher.set_state(HarnessState::ShuttingDown);
        self.publisher
            .emit(EventCode::ShuttingDown, None, None, None);
        if let Ok(mut flag) = self.shutdown.0.lock() {
            *flag = true;
        }
        self.shutdown.1.notify_one();
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

pub fn spawn(config: WriterConfig, publisher: Publisher) -> io::Result<WriterHandle> {
    publisher.set_workspace(config.workspace.clone());
    if let Some(path) = &config.session_path {
        publisher.set_session_path(Some(path.clone()));
    }
    if let Some(events) = &config.events_path {
        publisher.set_event_log(EventLog::new(events));
    }

    let shutdown = Arc::new((Mutex::new(false), Condvar::new()));
    let thread_shutdown = Arc::clone(&shutdown);
    let thread_publisher = publisher.clone();
    let pid = std::process::id();
    let thread = thread::Builder::new()
        .name("hi-liveness".into())
        .spawn(move || {
            run_loop(config, thread_publisher, pid, thread_shutdown);
        })?;

    Ok(WriterHandle {
        shutdown,
        thread: Some(thread),
        publisher,
    })
}

fn run_loop(
    config: WriterConfig,
    publisher: Publisher,
    pid: u32,
    shutdown: Arc<(Mutex<bool>, Condvar)>,
) {
    let mut seq = 0u64;
    loop {
        write_once(&config, &publisher, pid, &mut seq);
        let (lock, cvar) = &*shutdown;
        let Ok(guard) = lock.lock() else {
            break;
        };
        if *guard {
            break;
        }
        let (guard, _) = match cvar.wait_timeout(guard, config.period) {
            Ok(pair) => pair,
            Err(poisoned) => poisoned.into_inner(),
        };
        if *guard {
            break;
        }
    }
    publisher.set_state(HarnessState::ShuttingDown);
    write_once(&config, &publisher, pid, &mut seq);
}

fn write_once(config: &WriterConfig, publisher: &Publisher, pid: u32, seq: &mut u64) {
    if !writer_may_publish(pid, &config.instance) {
        return;
    }
    *seq = seq.saturating_add(1);
    let view = publisher.copy_for_write();
    let beat = view.into_heartbeat(*seq, unix_ms(), pid, &config.instance, config.generation);
    let _ = write_atomic_json(&config.heartbeat_path, &beat);
}

pub fn read_heartbeat(path: &Path) -> io::Result<Heartbeat> {
    match read_once(path) {
        Ok(beat) => Ok(beat),
        Err(_) => read_once(path),
    }
}

fn read_once(path: &Path) -> io::Result<Heartbeat> {
    let bytes = fs::read(path)?;
    serde_json::from_slice(&bytes).map_err(|err| io::Error::new(io::ErrorKind::InvalidData, err))
}
