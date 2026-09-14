//! Classify-then-act loop. Kill only on HarnessBug or true process death.

use std::ffi::OsString;
use std::path::PathBuf;
use std::process::ExitStatus;
use std::time::Instant;

use anyhow::{Context, Result};
use clap::Parser;
use hi_liveness::{
    ENV_CRASH_DIR, ENV_EVENTS, ENV_GENERATION, ENV_HEARTBEAT, ENV_HI_BINARY, ENV_INSTANCE,
    ENV_PANIC_FILE, ENV_ROLE, ENV_SUPERVISED, ENV_TURN_INTENT,
};
use tokio::process::Child;

use crate::ENV_CHECKOUT;
use crate::ENV_SESSION_TOKEN;
use crate::args::{find_on_path, strip_sentinel_args};
use crate::budget;
use crate::classify::{self, Class, ClassifyContext};
use crate::config::{MonitorConfig, SupervisorConfig, instance_token, peek_machine};
use crate::fsutil;
use crate::incident;
use crate::monitor::{Monitor, MonitorSignal};
use crate::paths;
use crate::repair::{self, RepairOutcome};
use crate::spawn;

#[derive(Parser, Debug)]
#[command(
    name = "hi-sentinel",
    about = "Hi Sentinel supervisor (not hi-bootstrap / RSI)"
)]
struct SentinelCli {
    /// After a verified repair, apply without prompting (still never pushes).
    #[arg(long)]
    apply: bool,
    /// Git checkout of Hi used for repair worktrees.
    #[arg(long, value_name = "PATH")]
    checkout: Option<PathBuf>,
    /// `hi` argv forwarded after `--`.
    #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
    hi_argv: Vec<OsString>,
}

pub struct SupervisorOutcome {
    pub class: Option<Class>,
    pub child_status: Option<ExitStatus>,
    pub child_pid: u32,
    pub child_alive: bool,
    pub incident_dir: Option<PathBuf>,
    pub exit_code: i32,
    pub leftover: Option<Child>,
}

impl SupervisorOutcome {
    pub async fn reap_leftover(&mut self) {
        if let Some(mut child) = self.leftover.take() {
            if let Some(pid) = child.id() {
                spawn::signal_group(pid as i32, libc::SIGKILL);
            }
            let _ = child.wait().await;
            self.child_alive = false;
        }
    }
}

pub fn run() -> Result<i32> {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .context("tokio runtime")?;
    rt.block_on(run_async())
}

async fn run_async() -> Result<i32> {
    let cli = SentinelCli::parse();
    let stripped = strip_sentinel_args(cli.hi_argv);
    let hi_binary = std::env::var_os(ENV_HI_BINARY)
        .map(PathBuf::from)
        .or_else(|| find_on_path("hi"))
        .context("HI_SENTINEL_HI_BINARY / hi not found")?;
    let checkout = cli
        .checkout
        .or(stripped.checkout)
        .or_else(|| std::env::var_os(ENV_CHECKOUT).map(PathBuf::from))
        .or_else(|| peek_machine().and_then(|s| s.checkout));
    let workspace = std::env::current_dir().context("cwd")?;
    let state_dir = paths::state_dir();
    fsutil::mkdir_0700(&state_dir)?;
    let original_argv: Vec<String> = std::env::args().collect();
    let cfg = SupervisorConfig {
        child_program: hi_binary.clone(),
        child_args: stripped.args,
        hi_binary,
        original_argv,
        checkout,
        apply: cli.apply || stripped.apply,
        monitor: MonitorConfig::from_machine_and_env(),
        state_dir,
        workspace,
        generation: std::env::var(ENV_GENERATION)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0),
        inherit_stdio: true,
        extra_env: Vec::new(),
        once: false,
    };
    let outcome = supervise(cfg).await?;
    Ok(outcome.exit_code)
}

pub async fn supervise(cfg: SupervisorConfig) -> Result<SupervisorOutcome> {
    let started = Instant::now();
    fsutil::mkdir_0700(&cfg.state_dir)?;
    let ttl_days = peek_machine()
        .and_then(|s| s.incident_retention_days)
        .unwrap_or(7);
    budget::gc_incidents(
        &paths::incidents_dir(&cfg.state_dir),
        std::time::Duration::from_secs(ttl_days.saturating_mul(86400)),
        50,
    );

    let instance = instance_token();
    let runtime = paths::runtime_dir(&cfg.state_dir, &instance);
    fsutil::mkdir_0700(&runtime)?;
    let crash_dir = runtime.join("crash");
    fsutil::mkdir_0700(&crash_dir)?;
    let heartbeat_path = runtime.join("heartbeat.json");
    let events_path = runtime.join("events.jsonl");
    let panic_file = runtime.join("panic.txt");
    let turn_intent = runtime.join("turn-intent.json");

    let mut env_pairs = vec![
        (ENV_SUPERVISED.to_string(), "1".into()),
        (ENV_GENERATION.to_string(), cfg.generation.to_string()),
        (ENV_ROLE.to_string(), "harness".into()),
        (ENV_INSTANCE.to_string(), instance.clone()),
        (
            ENV_HEARTBEAT.to_string(),
            heartbeat_path.display().to_string(),
        ),
        (ENV_EVENTS.to_string(), events_path.display().to_string()),
        (
            ENV_TURN_INTENT.to_string(),
            turn_intent.display().to_string(),
        ),
        (ENV_PANIC_FILE.to_string(), panic_file.display().to_string()),
        (ENV_CRASH_DIR.to_string(), crash_dir.display().to_string()),
        (
            ENV_HI_BINARY.to_string(),
            cfg.hi_binary.display().to_string(),
        ),
        (ENV_SESSION_TOKEN.to_string(), instance.clone()),
    ];
    env_pairs.extend(cfg.extra_env.iter().cloned());

    spawn::write_supervisor_log(&runtime, "spawning child");
    let terminal = spawn::snapshot_terminal();
    let spawned = spawn::spawn_child(&cfg, &env_pairs, terminal)?;
    let mut child = spawned.child;
    let pid = spawned.pid;
    let pgid = spawned.pgid;
    let terminal = spawned.terminal;
    let mut monitor = Monitor::new(cfg.monitor.clone(), pid, instance);
    let mut report_written: Option<PathBuf> = None;

    loop {
        tokio::select! {
            status = child.wait() => {
                let status = status.context("waiting for supervised child")?;
                terminal.restore();
                let ctx = classify_ctx(&cfg, &crash_dir, &panic_file, monitor.last_heartbeat(), false);
                let signal = MonitorSignal::ChildExited {
                    status,
                    waited: started.elapsed(),
                };
                let class = classify::classify(&signal, &ctx);
                let mut incident_dir = report_written.clone();
                if let Some(class) = &class
                    && (class.is_harness_bug() || matches!(class, Class::ReportOnly { .. }))
                    && incident_dir.is_none()
                {
                    match incident::write_bundle(
                        &cfg,
                        class,
                        &runtime,
                        monitor.last_heartbeat(),
                        started.elapsed(),
                    ) {
                        Ok(bundle) => {
                            incident_dir = Some(bundle.dir);
                        }
                        Err(err) => {
                            spawn::write_supervisor_log(
                                &runtime,
                                &format!("bundle failed: {err:#}"),
                            );
                        }
                    }
                }
                let repair_note = match class.as_ref() {
                    Some(class) if class.is_harness_bug() => {
                        maybe_run_repair(&cfg, class, incident_dir.as_deref(), &runtime).await
                    }
                    _ => None,
                };
                if let Some(class) = &class
                    && (class.is_harness_bug() || matches!(class, Class::ReportOnly { .. }))
                {
                    report_user(class, incident_dir.as_ref(), repair_note.as_deref());
                }
                let exit_code = if class.as_ref().is_some_and(Class::is_harness_bug) {
                    1
                } else {
                    status_code(status)
                };
                return Ok(SupervisorOutcome {
                    class,
                    child_status: Some(status),
                    child_pid: pid,
                    child_alive: false,
                    incident_dir,
                    exit_code,
                    leftover: None,
                });
            }
            _ = tokio::time::sleep(cfg.monitor.poll_interval) => {
                let Some(signal) = monitor.poll(&heartbeat_path) else {
                    continue;
                };
                let ctx = classify_ctx(&cfg, &crash_dir, &panic_file, monitor.last_heartbeat(), true);
                let Some(class) = classify::classify(&signal, &ctx) else {
                    continue;
                };
                if matches!(signal, MonitorSignal::ProgressStall { .. }) {
                    monitor.note_classified_progress_stall();
                }
                spawn::write_supervisor_log(
                    &runtime,
                    &format!("classified {} {}", class.class_slug(), class.kind_slug()),
                );
                if class.is_harness_bug() {
                    let tool = monitor
                        .last_heartbeat()
                        .and_then(|h| h.current_tool_pgid);
                    let status = spawn::abort_harness_child(
                        &mut child,
                        pgid,
                        tool,
                        cfg.monitor.term_grace,
                    )
                    .await?;
                    terminal.restore();
                    let bundle = incident::write_bundle(
                        &cfg,
                        &class,
                        &runtime,
                        monitor.last_heartbeat(),
                        started.elapsed(),
                    );
                    let incident_dir = bundle.ok().map(|b| b.dir);
                    let repair_note =
                        maybe_run_repair(&cfg, &class, incident_dir.as_deref(), &runtime).await;
                    report_user(&class, incident_dir.as_ref(), repair_note.as_deref());
                    return Ok(SupervisorOutcome {
                        class: Some(class),
                        child_status: Some(status),
                        child_pid: pid,
                        child_alive: false,
                        incident_dir,
                        exit_code: 1,
                        leftover: None,
                    });
                }
                if matches!(class, Class::ReportOnly { .. })
                    && report_written.is_none()
                    && let Ok(bundle) = incident::write_bundle(
                        &cfg,
                        &class,
                        &runtime,
                        monitor.last_heartbeat(),
                        started.elapsed(),
                    )
                {
                    report_written = Some(bundle.dir);
                }
                if cfg.once {
                    return Ok(SupervisorOutcome {
                        class: Some(class),
                        child_status: None,
                        child_pid: pid,
                        child_alive: spawn::pid_alive(pid),
                        incident_dir: report_written.clone(),
                        exit_code: 0,
                        leftover: Some(child),
                    });
                }
            }
        }
    }
}

fn classify_ctx(
    cfg: &SupervisorConfig,
    crash_dir: &std::path::Path,
    panic_file: &std::path::Path,
    last: Option<&hi_liveness::Heartbeat>,
    alive: bool,
) -> ClassifyContext {
    ClassifyContext {
        checkout: cfg.checkout.clone(),
        crash_dir: crash_dir.to_path_buf(),
        panic_file: panic_file.to_path_buf(),
        last_heartbeat: last.cloned(),
        child_alive: alive,
    }
}

fn status_code(status: ExitStatus) -> i32 {
    if let Some(code) = status.code() {
        return code;
    }
    #[cfg(unix)]
    {
        use std::os::unix::process::ExitStatusExt;
        if let Some(sig) = status.signal() {
            return 128 + sig;
        }
    }
    1
}

async fn maybe_run_repair(
    cfg: &SupervisorConfig,
    class: &Class,
    incident_dir: Option<&std::path::Path>,
    runtime: &std::path::Path,
) -> Option<String> {
    let dir = incident_dir?;
    let id = dir
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("incident");
    let worktree = paths::worktrees_dir().join(id);
    eprintln!(
        "hi: starting repair for {id}; worktree {}.",
        worktree.display()
    );
    spawn::write_supervisor_log(runtime, "repair starting");
    let note = match repair::maybe_repair(cfg, class, dir).await {
        RepairOutcome::Completed { branch, gate, .. } if gate.passed => {
            format!("Repair available on branch {branch}. Not applied.")
        }
        RepairOutcome::Completed { branch, .. } => {
            format!("Repair produced branch {branch} but verification failed. Not applied.")
        }
        RepairOutcome::Skipped { reason } => reason.user_line(),
    };
    spawn::write_supervisor_log(runtime, &note);
    Some(note)
}

fn report_user(class: &Class, dir: Option<&PathBuf>, repair_note: Option<&str>) {
    let id = dir
        .and_then(|p| p.file_name())
        .and_then(|n| n.to_str())
        .unwrap_or("incident");
    eprintln!(
        "hi: internal harness error recorded as {id} ({}).",
        class.kind_slug()
    );
    if let Some(note) = repair_note {
        eprintln!("    {note}");
    }
}
