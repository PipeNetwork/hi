//! Fake-child classifier tests. Thresholds are shortened; we never wait 180s.

use std::ffi::OsString;
use std::path::PathBuf;
use std::time::Duration;

use hi_liveness::HarnessState;

use crate::classify::{BugKind, Class, ExternalKind, ReportKind};
use crate::config::{MonitorConfig, SupervisorConfig};
use crate::fsutil;
use crate::supervise::supervise;

const SCRIPT: &str = r#"
set -eu
HB="${HI_SENTINEL_HEARTBEAT:?}"
INSTANCE="${HI_SENTINEL_INSTANCE:?}"
STATE="${HI_FAKE_STATE:-executing_tool}"
MODE="${HI_FAKE_MODE:-progress_stall}"
WS="${HI_FAKE_WORKSPACE:-/tmp}"
pgids_json="${HI_FAKE_PGIDS_JSON:-[]}"
write_hb() {
  seq="$1"
  progress="$2"
  ts=$(perl -MTime::HiRes=time -e 'printf("%d", time()*1000)' 2>/dev/null || echo 1)
  tmp="${HB}.tmp"
  printf '%s' "{\"schema_version\":1,\"seq\":${seq},\"ts_unix_ms\":${ts},\"pid\":$$,\"instance\":\"${INSTANCE}\",\"generation\":0,\"state\":\"${STATE}\",\"last_progress_unix_ms\":${progress},\"last_event\":null,\"last_tool\":\"bash\",\"last_tool_id\":null,\"consecutive_identical_tools\":0,\"identical_tool_count_in_turn\":0,\"turn_index\":1,\"session_path\":null,\"workspace\":\"${WS}\",\"pre_checkpoint\":null,\"child_pgids\":${pgids_json},\"current_tool_pgid\":null,\"invariant\":null}" > "$tmp"
  mv "$tmp" "$HB"
}
case "$MODE" in
  progress_stall)
    seq=0
    while true; do
      seq=$((seq+1))
      write_hb "$seq" 1
      sleep 0.02
    done
    ;;
  liveness_stall)
    write_hb 1 1
    while true; do sleep 1; done
    ;;
  idle_then_tool)
    seq=0
    i=0
    while [ "$i" -lt 8 ]; do
      seq=$((seq+1))
      i=$((i+1))
      STATE=idle
      write_hb "$seq" 1
      sleep 0.03
    done
    STATE=executing_tool
    while true; do
      seq=$((seq+1))
      write_hb "$seq" 1
      sleep 0.02
    done
    ;;
  exit1)
    exit 1
    ;;
  sigint)
    exec perl -e '$SIG{INT}="DEFAULT"; kill INT => $$; sleep 5'
    ;;
  sigstop)
    write_hb 1 1
    kill -STOP $$
    sleep 30
    ;;
esac
"#;

fn progress_monitor() -> MonitorConfig {
    MonitorConfig {
        liveness_timeout: Duration::from_secs(5),
        progress_timeout: Duration::from_millis(40),
        live_pgid_report_cap: Duration::from_secs(5),
        start_grace: Duration::from_millis(20),
        poll_interval: Duration::from_millis(10),
        term_grace: Duration::from_millis(200),
        confirmation_exempt: true,
    }
}

fn liveness_monitor() -> MonitorConfig {
    MonitorConfig {
        liveness_timeout: Duration::from_millis(80),
        progress_timeout: Duration::from_secs(5),
        live_pgid_report_cap: Duration::from_secs(5),
        start_grace: Duration::from_millis(20),
        poll_interval: Duration::from_millis(10),
        term_grace: Duration::from_millis(200),
        confirmation_exempt: true,
    }
}

fn cfg(
    dir: &std::path::Path,
    mode: &str,
    state: &str,
    pgids_json: &str,
    monitor: MonitorConfig,
    once: bool,
) -> SupervisorConfig {
    let workspace = dir.join("ws");
    let _ = std::fs::create_dir_all(&workspace);
    SupervisorConfig {
        child_program: PathBuf::from("/bin/sh"),
        child_args: vec![OsString::from("-c"), OsString::from(SCRIPT)],
        hi_binary: PathBuf::from("/bin/sh"),
        original_argv: vec!["hi".into(), "--autoharnessfix".into()],
        checkout: None,
        apply: false,
        monitor,
        state_dir: dir.join("state"),
        workspace: workspace.clone(),
        generation: 0,
        inherit_stdio: false,
        extra_env: vec![
            ("HI_FAKE_MODE".into(), mode.into()),
            ("HI_FAKE_STATE".into(), state.into()),
            ("HI_FAKE_PGIDS_JSON".into(), pgids_json.into()),
            ("HI_FAKE_WORKSPACE".into(), workspace.display().to_string()),
        ],
        once,
        seed_turn_intent: None,
        leaked_end_of_flags: false,
    }
}

async fn run(
    mode: &str,
    state: &str,
    pgids_json: &str,
    monitor: MonitorConfig,
    once: bool,
) -> crate::supervise::SupervisorOutcome {
    let dir = tempfile::tempdir().unwrap();
    let cfg = cfg(dir.path(), mode, state, pgids_json, monitor, once);
    let outcome = tokio::time::timeout(Duration::from_secs(5), supervise(cfg))
        .await
        .expect("supervise timed out")
        .expect("supervise failed");
    // Incident paths live under the temp dir; leak so assertions can read them.
    std::mem::forget(dir);
    outcome
}

#[tokio::test]
async fn executing_tool_empty_pgids_reaps_and_writes_0700_incident() {
    let outcome = run(
        "progress_stall",
        "executing_tool",
        "[]",
        progress_monitor(),
        false,
    )
    .await;
    assert_eq!(
        outcome.class,
        Some(Class::HarnessBug {
            kind: BugKind::Stall,
            confidence: crate::classify::Confidence::High
        })
    );
    assert!(!outcome.child_alive, "HarnessBug must reap the child");
    assert!(!crate::spawn::pid_alive(outcome.child_pid));
    let dir = outcome.incident_dir.expect("incident dir");
    assert_eq!(fsutil::unix_mode(&dir).unwrap(), 0o700);
    assert!(dir.join("incident.json").is_file());
    assert_eq!(
        fsutil::unix_mode(&dir.join("incident.json")).unwrap(),
        0o600
    );
}

#[tokio::test]
async fn live_pgids_are_report_only_and_keep_child_alive() {
    let dir = tempfile::tempdir().unwrap();
    let cfg = cfg(
        dir.path(),
        "progress_stall",
        "executing_tool",
        "[4242]",
        progress_monitor(),
        true,
    );
    let mut outcome = tokio::time::timeout(Duration::from_secs(5), supervise(cfg))
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        outcome.class,
        Some(Class::ReportOnly {
            kind: ReportKind::LiveChildQuiet
        })
    );
    assert!(
        outcome.child_alive,
        "live child_pgids must not SIGTERM the child"
    );
    assert!(crate::spawn::pid_alive(outcome.child_pid));
    let incident = outcome.incident_dir.clone().expect("report-only bundle");
    assert_eq!(fsutil::unix_mode(&incident).unwrap(), 0o700);
    outcome.reap_leftover().await;
}

#[tokio::test]
async fn verifying_progress_stall_is_user_verify_child_alive() {
    let dir = tempfile::tempdir().unwrap();
    let cfg = cfg(
        dir.path(),
        "progress_stall",
        "verifying",
        "[]",
        progress_monitor(),
        true,
    );
    let mut outcome = tokio::time::timeout(Duration::from_secs(5), supervise(cfg))
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        outcome.class,
        Some(Class::NotHarness {
            kind: ExternalKind::UserVerify
        })
    );
    assert!(outcome.child_alive);
    assert!(outcome.incident_dir.is_none());
    outcome.reap_leftover().await;
}

#[tokio::test]
async fn awaiting_model_is_provider_silence_child_alive() {
    let dir = tempfile::tempdir().unwrap();
    let cfg = cfg(
        dir.path(),
        "progress_stall",
        "awaiting_model",
        "[]",
        progress_monitor(),
        true,
    );
    let mut outcome = tokio::time::timeout(Duration::from_secs(5), supervise(cfg))
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        outcome.class,
        Some(Class::NotHarness {
            kind: ExternalKind::ProviderSilence
        })
    );
    assert!(outcome.child_alive);
    outcome.reap_leftover().await;
}

#[tokio::test]
async fn compacting_empty_pgids_is_provider_silence_not_kill() {
    let dir = tempfile::tempdir().unwrap();
    let cfg = cfg(
        dir.path(),
        "progress_stall",
        "compacting",
        "[]",
        progress_monitor(),
        true,
    );
    let mut outcome = tokio::time::timeout(Duration::from_secs(5), supervise(cfg))
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        outcome.class,
        Some(Class::NotHarness {
            kind: ExternalKind::ProviderSilence
        })
    );
    assert!(
        outcome.child_alive,
        "Compacting must not become HarnessBug or SIGTERM"
    );
    assert!(!matches!(outcome.class, Some(Class::HarnessBug { .. })));
    outcome.reap_leftover().await;
}

#[tokio::test]
async fn exit_1_without_crash_marker_is_not_harness() {
    let outcome = run("exit1", "idle", "[]", progress_monitor(), false).await;
    assert_eq!(
        outcome.class,
        Some(Class::NotHarness {
            kind: ExternalKind::UserError
        })
    );
    assert!(!outcome.child_alive);
    assert!(outcome.incident_dir.is_none());
}

#[tokio::test]
async fn sigint_is_user_stop() {
    let outcome = run("sigint", "idle", "[]", progress_monitor(), false).await;
    assert_eq!(
        outcome.class,
        Some(Class::NotHarness {
            kind: ExternalKind::UserStop
        })
    );
    assert!(!outcome.child_alive);
}

#[tokio::test]
async fn stopped_child_exits_the_supervisor() {
    let outcome = run("sigstop", "idle", "[]", liveness_monitor(), false).await;
    assert_eq!(
        outcome.class,
        Some(Class::NotHarness {
            kind: ExternalKind::UserStop
        })
    );
    assert!(
        !outcome.child_alive,
        "a stopped hi must not leave Sentinel running"
    );
    assert!(outcome.relaunch.is_none());
}

#[tokio::test]
async fn idle_ignored_stall_then_executing_tool_reaps() {
    let outcome = run("idle_then_tool", "idle", "[]", progress_monitor(), false).await;
    assert_eq!(
        outcome.class,
        Some(Class::HarnessBug {
            kind: BugKind::Stall,
            confidence: crate::classify::Confidence::High
        })
    );
    assert!(!outcome.child_alive);
    assert!(outcome.incident_dir.is_some());
}

#[tokio::test]
async fn frozen_seq_is_liveness_stall_harness_bug() {
    let outcome = run(
        "liveness_stall",
        "executing_tool",
        "[]",
        liveness_monitor(),
        false,
    )
    .await;
    assert_eq!(
        outcome.class,
        Some(Class::HarnessBug {
            kind: BugKind::Stall,
            confidence: crate::classify::Confidence::High
        })
    );
    assert!(!outcome.child_alive);
    assert!(outcome.incident_dir.is_some());
}

#[test]
fn compacting_state_serde_round_trip() {
    let json = serde_json::to_string(&HarnessState::Compacting).unwrap();
    assert_eq!(json, "\"compacting\"");
}
