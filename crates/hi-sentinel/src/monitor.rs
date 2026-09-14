//! Dual-clock poll. Seq frozen is liveness; progress stale is a stall signal.

use std::path::Path;
use std::process::ExitStatus;
use std::time::{Duration, Instant};

use hi_liveness::{Heartbeat, unix_ms};

use crate::config::MonitorConfig;

#[derive(Clone, Debug)]
#[allow(dead_code)]
pub enum MonitorSignal {
    ChildExited {
        status: ExitStatus,
        waited: Duration,
    },
    LivenessStall {
        last_seq: u64,
        last_ts_unix_ms: u64,
    },
    ProgressStall {
        heartbeat: Heartbeat,
    },
    Invariant {
        heartbeat: Heartbeat,
    },
    ChildStopped {
        signal: i32,
    },
}

pub struct Monitor {
    cfg: MonitorConfig,
    child_pid: u32,
    instance: String,
    started: Instant,
    last_seq: Option<u64>,
    last_seq_change: Instant,
    last_heartbeat: Option<Heartbeat>,
    progress_signaled: bool,
}

impl Monitor {
    pub fn new(cfg: MonitorConfig, child_pid: u32, instance: String) -> Self {
        let now = Instant::now();
        Self {
            cfg,
            child_pid,
            instance,
            started: now,
            last_seq: None,
            last_seq_change: now,
            last_heartbeat: None,
            progress_signaled: false,
        }
    }

    pub fn last_heartbeat(&self) -> Option<&Heartbeat> {
        self.last_heartbeat.as_ref()
    }

    pub fn poll(&mut self, heartbeat_path: &Path) -> Option<MonitorSignal> {
        self.read_update(heartbeat_path);
        if self.started.elapsed() < self.cfg.start_grace {
            return None;
        }
        if let Some(hb) = &self.last_heartbeat
            && let Some(inv) = &hb.invariant
        {
            let _ = inv;
            return Some(MonitorSignal::Invariant {
                heartbeat: hb.clone(),
            });
        }
        if self.last_seq_change.elapsed() >= self.cfg.liveness_timeout {
            if process_stopped(self.child_pid) {
                return Some(MonitorSignal::ChildStopped {
                    signal: libc::SIGTSTP,
                });
            }
            let last_ts = self
                .last_heartbeat
                .as_ref()
                .map(|h| h.ts_unix_ms)
                .unwrap_or(0);
            return Some(MonitorSignal::LivenessStall {
                last_seq: self.last_seq.unwrap_or(0),
                last_ts_unix_ms: last_ts,
            });
        }
        let hb = self.last_heartbeat.as_ref()?;
        let age = unix_ms().saturating_sub(hb.last_progress_unix_ms);
        if Duration::from_millis(age) >= self.cfg.progress_timeout {
            if self.progress_signaled {
                return None;
            }
            self.progress_signaled = true;
            return Some(MonitorSignal::ProgressStall {
                heartbeat: hb.clone(),
            });
        }
        None
    }

    fn read_update(&mut self, path: &Path) {
        let Ok(hb) = hi_liveness::read_heartbeat(path) else {
            return;
        };
        if hb.pid != self.child_pid || hb.instance != self.instance {
            return;
        }
        if self.last_seq != Some(hb.seq) {
            self.last_seq = Some(hb.seq);
            self.last_seq_change = Instant::now();
        }
        self.last_heartbeat = Some(hb);
    }
}

fn process_stopped(pid: u32) -> bool {
    let output = std::process::Command::new("ps")
        .args(["-o", "stat=", "-p", &pid.to_string()])
        .output();
    let Ok(output) = output else {
        return false;
    };
    let stat = String::from_utf8_lossy(&output.stdout);
    stat.trim_start().starts_with('T')
}
