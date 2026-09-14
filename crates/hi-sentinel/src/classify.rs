//! Fail-closed classifier. Live quiet children are never HarnessBug.

use std::os::unix::process::ExitStatusExt;
use std::path::{Path, PathBuf};
use std::process::ExitStatus;

use hi_liveness::{
    AUTO_REPAIR_SET, HarnessState, Heartbeat, IDENTICAL_TOOL_CONSECUTIVE, IDENTICAL_TOOL_IN_TURN,
    InvariantCode,
};

use crate::checkout::cwd_is_checkout;
use crate::monitor::MonitorSignal;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Class {
    HarnessBug {
        kind: BugKind,
        confidence: Confidence,
    },
    ReportOnly {
        kind: ReportKind,
    },
    NotHarness {
        kind: ExternalKind,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BugKind {
    Crash,
    Stall,
    Invariant,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReportKind {
    AgentLoop,
    LiveChildQuiet,
    CheckoutCwdStall,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExternalKind {
    ProviderSilence,
    UserVerify,
    UserStop,
    UserError,
    ToolFailure,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Confidence {
    High,
    Medium,
}

impl Class {
    pub fn class_slug(&self) -> &'static str {
        match self {
            Self::HarnessBug { .. } => "harness_bug",
            Self::ReportOnly { .. } => "report_only",
            Self::NotHarness { .. } => "not_harness",
        }
    }

    pub fn kind_slug(&self) -> &'static str {
        match self {
            Self::HarnessBug { kind, .. } => match kind {
                BugKind::Crash => "crash",
                BugKind::Stall => "stall",
                BugKind::Invariant => "invariant",
            },
            Self::ReportOnly { kind } => match kind {
                ReportKind::AgentLoop => "agent_loop",
                ReportKind::LiveChildQuiet => "live_child_quiet",
                ReportKind::CheckoutCwdStall => "checkout_cwd_stall",
            },
            Self::NotHarness { kind } => match kind {
                ExternalKind::ProviderSilence => "provider_silence",
                ExternalKind::UserVerify => "user_verify",
                ExternalKind::UserStop => "user_stop",
                ExternalKind::UserError => "user_error",
                ExternalKind::ToolFailure => "tool_failure",
            },
        }
    }

    pub fn is_harness_bug(&self) -> bool {
        matches!(self, Self::HarnessBug { .. })
    }
}

#[derive(Clone, Debug)]
pub struct ClassifyContext {
    pub checkout: Option<PathBuf>,
    pub crash_dir: PathBuf,
    pub panic_file: PathBuf,
    pub last_heartbeat: Option<Heartbeat>,
    pub child_alive: bool,
    /// Child argv started with `--`, so `--session-file` was a prompt.
    pub leaked_end_of_flags: bool,
}

/// `None` means ignored (Idle / AwaitingUser / exit 0).
pub fn classify(signal: &MonitorSignal, ctx: &ClassifyContext) -> Option<Class> {
    match signal {
        MonitorSignal::ChildExited { status, .. } => classify_exit(*status, ctx),
        MonitorSignal::ChildStopped { .. } => Some(Class::NotHarness {
            kind: ExternalKind::UserStop,
        }),
        MonitorSignal::Invariant { heartbeat } => classify_invariant(heartbeat),
        MonitorSignal::LivenessStall { .. } => Some(Class::HarnessBug {
            kind: BugKind::Stall,
            confidence: Confidence::High,
        }),
        MonitorSignal::ProgressStall { heartbeat } => classify_progress(heartbeat, ctx),
    }
}

fn classify_exit(status: ExitStatus, ctx: &ClassifyContext) -> Option<Class> {
    if let Some(sig) = status.signal() {
        if is_fault_signal(sig) || crash_evidence(ctx) {
            return Some(Class::HarnessBug {
                kind: BugKind::Crash,
                confidence: Confidence::High,
            });
        }
        if is_user_stop_signal(sig) {
            return Some(Class::NotHarness {
                kind: ExternalKind::UserStop,
            });
        }
        // SIGABRT without a panic file / crash marker is NotHarness: the
        // crash handler does not register SIGABRT.
    }
    if crash_evidence(ctx) {
        return Some(Class::HarnessBug {
            kind: BugKind::Crash,
            confidence: Confidence::High,
        });
    }
    if let Some(hb) = &ctx.last_heartbeat
        && let Some(class) = classify_invariant(hb)
        && class.is_harness_bug()
    {
        return Some(class);
    }
    match status.code() {
        Some(0) => None,
        _ if ctx.leaked_end_of_flags && startup_death(ctx) => Some(Class::HarnessBug {
            kind: BugKind::Crash,
            confidence: Confidence::High,
        }),
        _ => Some(Class::NotHarness {
            kind: ExternalKind::UserError,
        }),
    }
}

fn startup_death(ctx: &ClassifyContext) -> bool {
    ctx.last_heartbeat
        .as_ref()
        .is_none_or(|hb| hb.turn_index == 0 && hb.last_tool.is_none())
}

fn classify_invariant(heartbeat: &Heartbeat) -> Option<Class> {
    let inv = heartbeat.invariant.as_ref()?;
    if AUTO_REPAIR_SET.contains(&inv.code) {
        Some(Class::HarnessBug {
            kind: BugKind::Invariant,
            confidence: Confidence::High,
        })
    } else {
        Some(Class::ReportOnly {
            kind: ReportKind::AgentLoop,
        })
    }
}

fn classify_progress(heartbeat: &Heartbeat, ctx: &ClassifyContext) -> Option<Class> {
    if identical_storm(heartbeat) {
        return Some(Class::ReportOnly {
            kind: ReportKind::AgentLoop,
        });
    }
    match heartbeat.state {
        HarnessState::AwaitingUser
        | HarnessState::Starting
        | HarnessState::Idle
        | HarnessState::AwaitingConfirmation
        | HarnessState::ShuttingDown => None,
        HarnessState::Verifying => Some(Class::NotHarness {
            kind: ExternalKind::UserVerify,
        }),
        HarnessState::AwaitingModel | HarnessState::Compacting => Some(Class::NotHarness {
            kind: ExternalKind::ProviderSilence,
        }),
        HarnessState::ExecutingTool => {
            if !heartbeat.child_pgids.is_empty() {
                return Some(Class::ReportOnly {
                    kind: ReportKind::LiveChildQuiet,
                });
            }
            if cwd_is_checkout(&heartbeat.workspace, ctx.checkout.as_deref()) {
                return Some(Class::ReportOnly {
                    kind: ReportKind::CheckoutCwdStall,
                });
            }
            Some(Class::HarnessBug {
                kind: BugKind::Stall,
                confidence: Confidence::High,
            })
        }
    }
}

fn identical_storm(heartbeat: &Heartbeat) -> bool {
    heartbeat.consecutive_identical_tools >= IDENTICAL_TOOL_CONSECUTIVE
        || heartbeat.identical_tool_count_in_turn >= IDENTICAL_TOOL_IN_TURN
        || heartbeat
            .invariant
            .as_ref()
            .is_some_and(|inv| inv.code == InvariantCode::IdenticalToolStorm)
}

pub fn crash_evidence(ctx: &ClassifyContext) -> bool {
    nonempty_file(&ctx.panic_file) || crash_dir_has_marker(&ctx.crash_dir)
}

fn nonempty_file(path: &Path) -> bool {
    std::fs::metadata(path).is_ok_and(|meta| meta.is_file() && meta.len() > 0)
}

fn crash_dir_has_marker(dir: &Path) -> bool {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return false;
    };
    entries.flatten().any(|entry| {
        let name = entry.file_name();
        let name = name.to_string_lossy();
        let looks = name.starts_with("last-crash")
            || name.starts_with("crash-")
            || name.ends_with(".bin")
            || name.ends_with(".txt");
        looks && entry.metadata().is_ok_and(|m| m.is_file() && m.len() > 0)
    })
}

fn is_fault_signal(sig: i32) -> bool {
    matches!(
        sig,
        libc::SIGSEGV | libc::SIGBUS | libc::SIGILL | libc::SIGFPE
    )
}

fn is_user_stop_signal(sig: i32) -> bool {
    matches!(
        sig,
        libc::SIGINT | libc::SIGTERM | libc::SIGHUP | libc::SIGTSTP | libc::SIGSTOP
    )
}

#[cfg(test)]
pub(crate) fn sample_beat(state: HarnessState, pgids: Vec<i32>, workspace: &str) -> Heartbeat {
    Heartbeat {
        schema_version: hi_liveness::SCHEMA_VERSION,
        seq: 4,
        ts_unix_ms: 1_000,
        pid: 42,
        instance: "tok".into(),
        generation: 0,
        state,
        last_progress_unix_ms: 1,
        last_event: None,
        last_tool: Some("bash".into()),
        last_tool_id: None,
        consecutive_identical_tools: 0,
        identical_tool_count_in_turn: 0,
        turn_index: 1,
        session_path: None,
        workspace: workspace.into(),
        pre_checkpoint: None,
        child_pgids: pgids,
        current_tool_pgid: None,
        invariant: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hi_liveness::InvariantViolation;
    use std::time::Duration;

    fn ctx() -> ClassifyContext {
        ClassifyContext {
            checkout: None,
            crash_dir: PathBuf::from("/tmp/no-crash-dir-sentinel"),
            panic_file: PathBuf::from("/tmp/no-panic-sentinel"),
            last_heartbeat: None,
            child_alive: true,
            leaked_end_of_flags: false,
        }
    }

    #[test]
    fn executing_tool_empty_pgids_is_harness_stall() {
        let hb = sample_beat(HarnessState::ExecutingTool, vec![], "/tmp/ws");
        let class = classify(&MonitorSignal::ProgressStall { heartbeat: hb }, &ctx());
        assert_eq!(
            class,
            Some(Class::HarnessBug {
                kind: BugKind::Stall,
                confidence: Confidence::High
            })
        );
    }

    #[test]
    fn live_pgids_are_report_only() {
        let hb = sample_beat(HarnessState::ExecutingTool, vec![4242], "/tmp/ws");
        let class = classify(&MonitorSignal::ProgressStall { heartbeat: hb }, &ctx());
        assert_eq!(
            class,
            Some(Class::ReportOnly {
                kind: ReportKind::LiveChildQuiet
            })
        );
    }

    #[test]
    fn verifying_is_not_harness() {
        let hb = sample_beat(HarnessState::Verifying, vec![], "/tmp/ws");
        let class = classify(&MonitorSignal::ProgressStall { heartbeat: hb }, &ctx());
        assert_eq!(
            class,
            Some(Class::NotHarness {
                kind: ExternalKind::UserVerify
            })
        );
    }

    #[test]
    fn awaiting_model_and_compacting_are_provider_silence() {
        for state in [HarnessState::AwaitingModel, HarnessState::Compacting] {
            let hb = sample_beat(state, vec![], "/tmp/ws");
            let class = classify(&MonitorSignal::ProgressStall { heartbeat: hb }, &ctx());
            assert_eq!(
                class,
                Some(Class::NotHarness {
                    kind: ExternalKind::ProviderSilence
                }),
                "{state:?}"
            );
        }
    }

    #[test]
    fn leaked_double_dash_startup_exit_is_harness_bug() {
        let mut ctx = ctx();
        ctx.leaked_end_of_flags = true;
        ctx.child_alive = false;
        let mut hb = sample_beat(HarnessState::Idle, vec![], "/tmp/ws");
        hb.turn_index = 0;
        hb.last_tool = None;
        ctx.last_heartbeat = Some(hb);
        let status = ExitStatus::from_raw(2 << 8);
        let class = classify(
            &MonitorSignal::ChildExited {
                status,
                waited: Duration::from_millis(50),
            },
            &ctx,
        );
        assert_eq!(
            class,
            Some(Class::HarnessBug {
                kind: BugKind::Crash,
                confidence: Confidence::High
            })
        );
    }

    #[test]
    fn exit_1_without_crash_is_not_harness() {
        let status = ExitStatus::from_raw(1 << 8);
        let class = classify(
            &MonitorSignal::ChildExited {
                status,
                waited: Duration::from_millis(10),
            },
            &ctx(),
        );
        assert_eq!(
            class,
            Some(Class::NotHarness {
                kind: ExternalKind::UserError
            })
        );
    }

    #[test]
    fn sigint_is_user_stop() {
        let status = ExitStatus::from_raw(libc::SIGINT);
        let class = classify(
            &MonitorSignal::ChildExited {
                status,
                waited: Duration::from_millis(1),
            },
            &ctx(),
        );
        assert_eq!(
            class,
            Some(Class::NotHarness {
                kind: ExternalKind::UserStop
            })
        );
    }

    #[test]
    fn sigabrt_without_crash_evidence_is_not_harness() {
        let status = ExitStatus::from_raw(libc::SIGABRT);
        let class = classify(
            &MonitorSignal::ChildExited {
                status,
                waited: Duration::from_millis(1),
            },
            &ctx(),
        );
        assert_eq!(
            class,
            Some(Class::NotHarness {
                kind: ExternalKind::UserError
            })
        );
    }

    #[test]
    fn liveness_stall_is_harness_bug() {
        let class = classify(
            &MonitorSignal::LivenessStall {
                last_seq: 3,
                last_ts_unix_ms: 1,
            },
            &ctx(),
        );
        assert_eq!(
            class,
            Some(Class::HarnessBug {
                kind: BugKind::Stall,
                confidence: Confidence::High
            })
        );
    }

    #[test]
    fn idle_progress_is_ignored() {
        let hb = sample_beat(HarnessState::Idle, vec![], "/tmp/ws");
        assert_eq!(
            classify(&MonitorSignal::ProgressStall { heartbeat: hb }, &ctx()),
            None
        );
    }

    #[test]
    fn empty_assistant_after_tools_is_harness_bug_even_when_idle() {
        let mut hb = sample_beat(HarnessState::Idle, vec![], "/tmp/ws");
        hb.invariant = Some(InvariantViolation {
            code: InvariantCode::EmptyAssistantAfterTools,
            ts_unix_ms: 1,
            detail: None,
        });
        let class = classify(&MonitorSignal::Invariant { heartbeat: hb }, &ctx());
        assert_eq!(
            class,
            Some(Class::HarnessBug {
                kind: BugKind::Invariant,
                confidence: Confidence::High
            })
        );
        assert!(
            AUTO_REPAIR_SET.contains(&InvariantCode::EmptyAssistantAfterTools),
            "sentinel must auto-repair this, not leave the live child idle"
        );
    }
}
