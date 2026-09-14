//! Classify-then-act loop. Kill only on HarnessBug or true process death.

use std::ffi::OsString;
use std::io::IsTerminal;
use std::path::PathBuf;
use std::process::ExitStatus;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use clap::Parser;
use hi_liveness::{
    ENV_CRASH_DIR, ENV_EVENTS, ENV_GENERATION, ENV_HEARTBEAT, ENV_HI_BINARY, ENV_INSTANCE,
    ENV_PANIC_FILE, ENV_ROLE, ENV_SUPERVISED, ENV_TURN_INTENT,
};
use tokio::process::Child;

use crate::ENV_CHECKOUT;
use crate::ENV_KNOWN_GOOD;
use crate::ENV_SESSION_TOKEN;
use crate::apply::{self, ApplyOutcome, ApplyRefuse, ApplyRequest};
use crate::args::{find_on_path, strip_sentinel_args};
use crate::budget;
use crate::classify::{self, Class, ClassifyContext};
use crate::config::{MonitorConfig, RepairConfig, SupervisorConfig, instance_token, peek_machine};
use crate::fsutil;
use crate::incident;
use crate::ipc;
use crate::monitor::{Monitor, MonitorSignal};
use crate::paths;
use crate::relaunch::{self, RelaunchPlan};
use crate::repair::{self, RepairOutcome};
use crate::restore::{self, RestoreRequest};
use crate::rollback::{self, RecordInput};
use crate::slash;
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
    /// Manual repair of an existing incident bundle (`/autoharnessfix repair`).
    #[arg(long, value_name = "PATH")]
    incident: Option<PathBuf>,
    /// `hi` argv forwarded after `--`. Leading `repair` selects incident mode.
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
    pub relaunch: Option<RelaunchPlan>,
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
    let mut hi_argv = cli.hi_argv;
    let mut incident = cli.incident;
    if hi_argv.first().is_some_and(|arg| arg == "repair") {
        hi_argv.remove(0);
        if incident.is_none() {
            let mut iter = hi_argv.iter();
            while let Some(arg) = iter.next() {
                if arg == "--incident" {
                    incident = iter.next().cloned().map(PathBuf::from);
                    break;
                }
                if let Some(rest) = arg.to_str().and_then(|s| s.strip_prefix("--incident=")) {
                    incident = Some(PathBuf::from(rest));
                    break;
                }
            }
        }
    }
    if let Some(incident) = incident {
        return run_manual_repair(incident, cli.checkout, cli.apply).await;
    }
    let stripped = strip_sentinel_args(hi_argv);
    let hi_binary = std::env::var_os(ENV_HI_BINARY)
        .map(PathBuf::from)
        .or_else(|| find_on_path("hi"))
        .context("HI_SENTINEL_HI_BINARY / hi not found")?;
    let checkout = cli
        .checkout
        .or(stripped.checkout)
        .or_else(|| std::env::var_os(ENV_CHECKOUT).map(PathBuf::from))
        .or_else(crate::config::ensure_machine_checkout);
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
        apply: cli.apply || stripped.apply || peek_machine().is_some_and(|s| s.apply),
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
        seed_turn_intent: None,
    };
    let outcome = supervise(cfg).await?;
    Ok(outcome.exit_code)
}

pub async fn supervise(mut cfg: SupervisorConfig) -> Result<SupervisorOutcome> {
    fsutil::mkdir_0700(&cfg.state_dir)?;
    let ttl_days = peek_machine()
        .and_then(|s| s.incident_retention_days)
        .unwrap_or(7);
    budget::gc_incidents(
        &paths::incidents_dir(&cfg.state_dir),
        std::time::Duration::from_secs(ttl_days.saturating_mul(86400)),
        50,
    );

    if cfg.hi_binary.is_file() {
        let validated = cfg
            .checkout
            .as_ref()
            .and_then(|path| crate::checkout::validate(path).ok());
        let prev = paths::sidecar_bin_dir(&cfg.state_dir).join("hi.prev");
        let _ = rollback::record(&RecordInput {
            state_dir: &cfg.state_dir,
            checkout_path: validated.as_ref().map(|v| v.path.as_path()),
            checkout_sha: validated.as_ref().map(|v| v.head_sha.as_str()),
            binary_path: &cfg.hi_binary,
            prev_binary_path: prev.is_file().then_some(prev.as_path()),
        });
    }

    loop {
        let outcome = run_generation(&cfg).await?;
        if cfg.once {
            return Ok(outcome);
        }
        let Some(plan) = outcome.relaunch.clone() else {
            return Ok(outcome);
        };
        spawn::prepare_interactive_prompt();
        eprintln!("{}", relaunch::continuing_line(&plan));
        let restore = restore::maybe_restore_user_project(&RestoreRequest {
            hi_binary: cfg.hi_binary.clone(),
            workspace: relaunch::restore_workspace(&plan, &cfg.workspace),
            pre_checkpoint: plan.pre_checkpoint.clone(),
            stdin_is_tty: std::io::stdin().is_terminal(),
            answer: None,
            flag_present: None,
        });
        restore::report_restore(&restore);
        cfg = relaunch::next_generation(&cfg, &plan);
    }
}

async fn run_generation(cfg: &SupervisorConfig) -> Result<SupervisorOutcome> {
    let started = Instant::now();
    let instance = instance_token();
    let runtime = paths::runtime_dir(&cfg.state_dir, &instance);
    fsutil::mkdir_0700(&runtime)?;
    let crash_dir = runtime.join("crash");
    fsutil::mkdir_0700(&crash_dir)?;
    let heartbeat_path = runtime.join("heartbeat.json");
    let events_path = runtime.join("events.jsonl");
    let panic_file = runtime.join("panic.txt");
    let turn_intent = runtime.join("turn-intent.json");
    if let Some(src) = &cfg.seed_turn_intent
        && src.is_file()
    {
        let _ = std::fs::copy(src, &turn_intent);
        let _ = fsutil::chmod_0600(&turn_intent);
    }

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
        (
            ENV_KNOWN_GOOD.to_string(),
            paths::known_good_path(&cfg.state_dir).display().to_string(),
        ),
    ];
    env_pairs.extend(cfg.extra_env.iter().cloned());

    spawn::write_supervisor_log(&runtime, "spawning child");
    let terminal = spawn::snapshot_terminal();
    let spawned = spawn::spawn_child(cfg, &env_pairs, terminal)?;
    let mut child = spawned.child;
    let pid = spawned.pid;
    let pgid = spawned.pgid;
    let terminal = spawned.terminal;
    let mut monitor = Monitor::new(cfg.monitor.clone(), pid, instance);
    let mut report_written: Option<PathBuf> = None;
    let mut repair_task: Option<tokio::task::JoinHandle<String>> = None;

    loop {
        tokio::select! {
            status = child.wait() => {
                let status = status.context("waiting for supervised child")?;
                abort_slash_repair(&mut repair_task, &runtime, "aborted in-flight slash repair (child exited)").await;
                terminal.restore();
                let ctx = classify_ctx(cfg, &crash_dir, &panic_file, monitor.last_heartbeat(), false);
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
                        cfg,
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
                let repair = match class.as_ref() {
                    Some(class) if class.is_harness_bug() => {
                        maybe_run_repair(&cfg, class, incident_dir.as_deref(), &runtime, true).await
                    }
                    _ => RepairReport::default(),
                };
                if repair.relaunch.is_none()
                    && let Some(class) = &class
                    && (class.is_harness_bug() || matches!(class, Class::ReportOnly { .. }))
                {
                    report_user(class, incident_dir.as_ref(), repair.note.as_deref());
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
                    relaunch: repair.relaunch,
                });
            }
            _ = tokio::time::sleep(cfg.monitor.poll_interval) => {
                if ipc::take_request(&runtime, ipc::REQUEST_DIAGNOSE) {
                    match incident::write_diagnose_bundle(
                        &cfg,
                        &runtime,
                        monitor.last_heartbeat(),
                    ) {
                        Ok(bundle) => {
                            let _ = ipc::write_done(
                                &runtime,
                                ipc::REQUEST_DIAGNOSE_DONE,
                                &format!("{}\n", bundle.dir.display()),
                            );
                        }
                        Err(err) => {
                            let _ = ipc::write_done(
                                &runtime,
                                ipc::REQUEST_DIAGNOSE_DONE,
                                &format!("error: {err:#}\n"),
                            );
                            spawn::write_supervisor_log(
                                &runtime,
                                &format!("diagnose bundle failed: {err:#}"),
                            );
                        }
                    }
                }
                if ipc::take_request(&runtime, ipc::REQUEST_REPAIR) {
                    if repair_task.is_some() {
                        let _ = ipc::write_done(
                            &runtime,
                            ipc::REQUEST_REPAIR_DONE,
                            "repair already in progress",
                        );
                    } else {
                        let cfg_repair = cfg.clone();
                        let runtime_repair = runtime.clone();
                        let heartbeat = monitor.last_heartbeat().cloned();
                        repair_task = Some(tokio::spawn(async move {
                            let bundle = incident::write_diagnose_bundle(
                                &cfg_repair,
                                &runtime_repair,
                                heartbeat.as_ref(),
                            );
                            let dir = bundle.ok().map(|b| b.dir);
                            let class = slash::manual_class();
                            maybe_run_repair(
                                &cfg_repair,
                                &class,
                                dir.as_deref(),
                                &runtime_repair,
                                false,
                            )
                            .await
                            .note
                            .unwrap_or_else(|| "repair finished".into())
                        }));
                    }
                }
                if let Some(handle) = repair_task.take() {
                    if handle.is_finished() {
                        match handle.await {
                            Ok(note) => {
                                let _ = ipc::write_done(
                                    &runtime,
                                    ipc::REQUEST_REPAIR_DONE,
                                    &note,
                                );
                                spawn::write_supervisor_log(&runtime, &note);
                            }
                            Err(err) => {
                                spawn::write_supervisor_log(
                                    &runtime,
                                    &format!("repair task failed: {err}"),
                                );
                            }
                        }
                    } else {
                        repair_task = Some(handle);
                    }
                }
                let Some(signal) = monitor.poll(&heartbeat_path) else {
                    continue;
                };
                let ctx = classify_ctx(cfg, &crash_dir, &panic_file, monitor.last_heartbeat(), true);
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
                    abort_slash_repair(
                        &mut repair_task,
                        &runtime,
                        "aborted in-flight slash repair (harness bug)",
                    )
                    .await;
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
                        cfg,
                        &class,
                        &runtime,
                        monitor.last_heartbeat(),
                        started.elapsed(),
                    );
                    let incident_dir = bundle.ok().map(|b| b.dir);
                    let repair =
                        maybe_run_repair(&cfg, &class, incident_dir.as_deref(), &runtime, true)
                            .await;
                    if repair.relaunch.is_none() {
                        report_user(&class, incident_dir.as_ref(), repair.note.as_deref());
                    }
                    return Ok(SupervisorOutcome {
                        class: Some(class),
                        child_status: Some(status),
                        child_pid: pid,
                        child_alive: false,
                        incident_dir,
                        exit_code: 1,
                        leftover: None,
                        relaunch: repair.relaunch,
                    });
                }
                if matches!(class, Class::ReportOnly { .. })
                    && report_written.is_none()
                    && let Ok(bundle) = incident::write_bundle(
                        cfg,
                        &class,
                        &runtime,
                        monitor.last_heartbeat(),
                        started.elapsed(),
                    )
                {
                    report_written = Some(bundle.dir);
                }
                if cfg.once {
                    abort_slash_repair(
                        &mut repair_task,
                        &runtime,
                        "aborted in-flight slash repair (once)",
                    )
                    .await;
                    return Ok(SupervisorOutcome {
                        class: Some(class),
                        child_status: None,
                        child_pid: pid,
                        child_alive: spawn::pid_alive(pid),
                        incident_dir: report_written.clone(),
                        exit_code: 0,
                        leftover: Some(child),
                        relaunch: None,
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

#[derive(Default)]
struct RepairReport {
    note: Option<String>,
    relaunch: Option<RelaunchPlan>,
}

async fn maybe_run_repair(
    cfg: &SupervisorConfig,
    class: &Class,
    incident_dir: Option<&std::path::Path>,
    runtime: &std::path::Path,
    allow_prompt: bool,
) -> RepairReport {
    let Some(dir) = incident_dir else {
        return RepairReport::default();
    };
    let id = dir
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("incident");
    let repair_cfg = RepairConfig::from_machine_and_env();
    if cfg.generation >= repair_cfg.max_repairs_per_session {
        let kg = rollback::load(&cfg.state_dir).ok();
        let current_b3 = rollback::hash_file(&cfg.hi_binary).unwrap_or_default();
        let current_sha = cfg
            .checkout
            .as_ref()
            .and_then(|path| crate::checkout::validate(path).ok())
            .map(|v| v.head_sha);
        let note = apply::halt_user_line(
            cfg.generation,
            id,
            kg.as_ref(),
            current_sha.as_deref(),
            &current_b3,
            cfg.checkout.as_deref(),
            &paths::incidents_dir(&cfg.state_dir),
        );
        spawn::write_supervisor_log(runtime, &note);
        return RepairReport {
            note: Some(note),
            relaunch: None,
        };
    }
    let worktree = paths::worktrees_dir().join(id);
    let start_line = format!(
        "hi: starting repair for {id}; worktree {}.",
        worktree.display()
    );
    if allow_prompt {
        eprintln!("{start_line}");
    }
    spawn::write_supervisor_log(runtime, &start_line);
    let report = match repair::maybe_repair(cfg, class, dir).await {
        RepairOutcome::Completed {
            branch,
            gate,
            worktree,
            ..
        } if gate.passed => {
            if !allow_prompt && !cfg.apply {
                RepairReport {
                    note: Some(format!(
                        "Repair available on branch {branch}. Not applied (session still owns the tty). Enable [autoharnessfix] apply = true, or apply after this session exits. See /autoharnessfix history."
                    )),
                    relaunch: None,
                }
            } else {
                apply_verified(
                    cfg,
                    class,
                    id,
                    &branch,
                    worktree,
                    &repair_cfg,
                    allow_prompt,
                    dir,
                )
            }
        }
        RepairOutcome::Completed { branch, .. } => RepairReport {
            note: Some(format!(
                "Repair produced branch {branch} but verification failed. Not applied."
            )),
            relaunch: None,
        },
        RepairOutcome::Skipped { reason } => RepairReport {
            note: Some(reason.user_line()),
            relaunch: None,
        },
    };
    if let Some(note) = &report.note {
        spawn::write_supervisor_log(runtime, note);
    }
    report
}

fn apply_verified(
    cfg: &SupervisorConfig,
    class: &Class,
    id: &str,
    branch: &str,
    worktree: PathBuf,
    repair_cfg: &RepairConfig,
    allow_prompt: bool,
    incident_dir: &std::path::Path,
) -> RepairReport {
    let validated = cfg
        .checkout
        .as_ref()
        .and_then(|path| crate::checkout::validate(path).ok());
    let outcome = apply::maybe_apply(&ApplyRequest {
        incident_id: id.to_string(),
        worktree,
        checkout: validated
            .as_ref()
            .map(|v| v.path.clone())
            .or_else(|| cfg.checkout.clone())
            .unwrap_or_default(),
        checkout_dirty: validated.as_ref().is_some_and(|v| v.dirty),
        checkout_sha: validated.as_ref().map(|v| v.head_sha.clone()),
        hi_binary: cfg.hi_binary.clone(),
        state_dir: cfg.state_dir.clone(),
        generation: cfg.generation,
        auto_apply: cfg.apply,
        stdin_is_tty: allow_prompt && std::io::stdin().is_terminal(),
        apply_answer: None,
        overwrite_answer: None,
        cargo: apply::cargo_bin(),
        max_repairs_per_session: repair_cfg.max_repairs_per_session,
        max_modifications_per_hour: repair_cfg.max_modifications_per_hour,
        install_timeout: Duration::from_secs(15 * 60),
    });
    match outcome {
        ApplyOutcome::Applied { note } => {
            let plan = relaunch::plan_from_incident(
                incident_dir,
                relaunch::sidecar_or_current(cfg),
                id,
                class.kind_slug(),
                &cfg.child_args,
            );
            let mut out = relaunch::continuing_line(&plan);
            if !note.is_empty() {
                out.push(' ');
                out.push_str(&note);
            }
            RepairReport {
                note: Some(out),
                relaunch: Some(plan),
            }
        }
        ApplyOutcome::Refused {
            reason: ApplyRefuse::GenerationHalt,
            note,
        } => RepairReport {
            note: Some(note),
            relaunch: None,
        },
        ApplyOutcome::Refused { note, .. } => {
            let note = if note.contains("Not applied.") {
                if note.starts_with("hi:") {
                    note
                } else {
                    format!("hi: internal harness error recorded as {id}. {note}")
                }
            } else {
                format!(
                    "hi: internal harness error recorded as {id}. Repair available on branch {branch}. Not applied. {note}"
                )
            };
            RepairReport {
                note: Some(note),
                relaunch: None,
            }
        }
    }
}

async fn run_manual_repair(
    incident: PathBuf,
    checkout: Option<PathBuf>,
    apply: bool,
) -> Result<i32> {
    if !incident.is_dir() {
        anyhow::bail!("incident dir not found: {}", incident.display());
    }
    let hi_binary = std::env::var_os(ENV_HI_BINARY)
        .map(PathBuf::from)
        .or_else(|| find_on_path("hi"))
        .or_else(|| std::env::current_exe().ok())
        .context("hi binary not found")?;
    let checkout = checkout.or_else(|| peek_machine().and_then(|s| s.checkout));
    let state_dir = paths::state_dir();
    fsutil::mkdir_0700(&state_dir)?;
    let cfg = SupervisorConfig {
        child_program: hi_binary.clone(),
        child_args: Vec::new(),
        hi_binary,
        original_argv: std::env::args().collect(),
        checkout,
        apply,
        monitor: MonitorConfig::from_machine_and_env(),
        state_dir: state_dir.clone(),
        workspace: std::env::current_dir().context("cwd")?,
        generation: 0,
        inherit_stdio: true,
        extra_env: Vec::new(),
        once: true,
        seed_turn_intent: None,
    };
    let runtime = paths::runtime_dir(&state_dir, "repair");
    fsutil::mkdir_0700(&runtime)?;
    let class = slash::manual_class();
    let note = maybe_run_repair(&cfg, &class, Some(incident.as_path()), &runtime, true)
        .await
        .note
        .unwrap_or_else(|| "repair finished".into());
    eprintln!("{note}");
    Ok(0)
}

async fn abort_slash_repair(
    task: &mut Option<tokio::task::JoinHandle<String>>,
    runtime: &std::path::Path,
    why: &str,
) {
    let Some(handle) = task.take() else {
        return;
    };
    handle.abort();
    let _ = handle.await;
    spawn::write_supervisor_log(runtime, why);
}

fn report_user(class: &Class, dir: Option<&PathBuf>, repair_note: Option<&str>) {
    if let Some(note) = repair_note
        && note.starts_with("hi:")
    {
        eprintln!("{note}");
        return;
    }
    let id = dir
        .and_then(|p| p.file_name())
        .and_then(|n| n.to_str())
        .unwrap_or("incident");
    eprintln!(
        "hi: internal harness error recorded as {id} ({}).",
        class.kind_slug()
    );
    if let Some(note) = repair_note {
        for line in note.lines() {
            eprintln!("    {line}");
        }
    }
}
