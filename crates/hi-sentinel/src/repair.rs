//! Locked-down repair `hi --plain` in a detached worktree. No apply, no `main`.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;

use anyhow::{Context, Result};
use hi_liveness::{
    ENV_CRASH_DIR, ENV_EVENTS, ENV_GENERATION, ENV_HEARTBEAT, ENV_INSTANCE, ENV_PANIC_FILE,
    ENV_ROLE, ENV_SUPERVISED, ENV_TURN_INTENT,
};

use crate::classify::Class;
use crate::config::{
    PIPE_MODEL_ERROR, RepairConfig, RepairModels, SupervisorConfig, peek_machine,
    resolve_repair_models,
};
use crate::fsutil;
use crate::gate::{self, GateInput, GateReport, GateTimeouts};
use crate::paths;
use crate::spawn;
use crate::worktree::{self, Worktree};

const SKILL_REL: &str = "skills/autoharnessfix/SKILL.md";

#[derive(Clone, Debug)]
pub enum RepairSkip {
    NoCheckout,
    InvalidCheckout(String),
    BudgetGeneration,
    BudgetAttempts,
    NonPipeModel,
    DiagnoseFailed,
    DiagnoseNoRepro,
    DiagnoseTimeout,
    NoFixProduced,
    WorktreeFailed(String),
    PatchFailed,
    PatchTimeout,
}

impl RepairSkip {
    pub fn user_line(&self) -> String {
        match self {
            Self::NoCheckout => "checkout not configured; pass --autoharnessfix-checkout".into(),
            Self::InvalidCheckout(msg) => format!("checkout invalid: {msg}"),
            Self::BudgetGeneration => "Sentinel repair budget exhausted (generation cap).".into(),
            Self::BudgetAttempts => "Sentinel already attempted repair for this incident.".into(),
            Self::NonPipeModel => PIPE_MODEL_ERROR.into(),
            Self::DiagnoseFailed => "diagnose failed; skipping patch".into(),
            Self::DiagnoseNoRepro => "diagnose did not confirm reproduction; skipping patch".into(),
            Self::DiagnoseTimeout => "diagnose timed out; skipping patch".into(),
            Self::NoFixProduced => "no fix produced".into(),
            Self::WorktreeFailed(msg) => format!("worktree failed: {msg}"),
            Self::PatchFailed => "patch agent failed".into(),
            Self::PatchTimeout => "patch timed out".into(),
        }
    }
}

#[derive(Debug)]
pub enum RepairOutcome {
    Skipped {
        reason: RepairSkip,
    },
    Completed {
        worktree: PathBuf,
        branch: String,
        commit: String,
        gate: GateReport,
    },
}

pub struct RepairRequest {
    pub incident_id: String,
    pub incident_dir: PathBuf,
    pub checkout: PathBuf,
    pub hi_binary: PathBuf,
    pub generation: u32,
    pub kind: String,
    pub worktrees_dir: PathBuf,
    pub models: RepairModels,
    pub timeouts: RepairConfig,
    pub gate_timeouts: GateTimeouts,
}

pub async fn maybe_repair(
    cfg: &SupervisorConfig,
    class: &Class,
    incident_dir: &Path,
) -> RepairOutcome {
    if !class.is_harness_bug() {
        return RepairOutcome::Skipped {
            reason: RepairSkip::NoCheckout,
        };
    }
    let Some(checkout) = cfg.checkout.as_ref() else {
        return RepairOutcome::Skipped {
            reason: RepairSkip::NoCheckout,
        };
    };
    let validated = match crate::checkout::validate(checkout) {
        Ok(v) => v,
        Err(err) => {
            return RepairOutcome::Skipped {
                reason: RepairSkip::InvalidCheckout(err.to_string()),
            };
        }
    };
    let models = match resolve_repair_models(peek_machine().as_ref()) {
        Ok(m) => m,
        Err(_) => {
            return RepairOutcome::Skipped {
                reason: RepairSkip::NonPipeModel,
            };
        }
    };
    let timeouts = RepairConfig::from_machine_and_env();
    if cfg.generation >= timeouts.max_repairs_per_session {
        return RepairOutcome::Skipped {
            reason: RepairSkip::BudgetGeneration,
        };
    }
    let id = incident_dir
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("incident")
        .to_string();
    let request = RepairRequest {
        incident_id: id,
        incident_dir: incident_dir.to_path_buf(),
        checkout: validated.path,
        hi_binary: cfg.hi_binary.clone(),
        generation: cfg.generation,
        kind: class.kind_slug().to_string(),
        worktrees_dir: paths::worktrees_dir(),
        models,
        timeouts,
        gate_timeouts: GateTimeouts::default(),
    };
    run_repair(request).await
}

pub async fn run_repair(req: RepairRequest) -> RepairOutcome {
    if let Err(reason) = check_attempts(&req) {
        return RepairOutcome::Skipped { reason };
    }
    let _ = bump_attempts(&req.incident_dir);

    let dest = req.worktrees_dir.join(&req.incident_id);
    worktree::gc(&req.worktrees_dir, &req.checkout, None);
    let wt = match worktree::add_detached(&req.checkout, &dest, "HEAD") {
        Ok(wt) => wt,
        Err(err) => {
            return RepairOutcome::Skipped {
                reason: RepairSkip::WorktreeFailed(err.to_string()),
            };
        }
    };

    let cargo_home = wt.path.join(".hi/cargo-home");
    let cargo_target = wt.path.join("target");
    let state_home = wt.path.join(".hi/state");
    let _ = fsutil::mkdir_0700(&cargo_home);
    let _ = fsutil::mkdir_0700(&cargo_target);
    let _ = fsutil::mkdir_0700(&state_home);
    // Agent writes stay under the worktree: HI_SANDBOX=workspace cannot create
    // files in the incident dir under XDG_STATE_HOME.
    let _ = stage_agent_inputs(&req.incident_dir, &wt.path);

    match run_phase(&req, &wt, Phase::Diagnose).await {
        Ok(()) => {}
        Err(reason) => return RepairOutcome::Skipped { reason },
    }
    let diagnosis = parse_diagnosis(&agent_diagnosis_path(&wt.path));
    let _ = harvest_diagnosis(&wt.path, &req.incident_dir);
    if !diagnosis.reproduced {
        return RepairOutcome::Skipped {
            reason: RepairSkip::DiagnoseNoRepro,
        };
    }

    match run_phase(&req, &wt, Phase::Patch).await {
        Ok(()) => {}
        Err(reason) => return RepairOutcome::Skipped { reason },
    }

    match worktree::has_changes(&wt.path) {
        Ok(true) => {}
        Ok(false) => {
            return RepairOutcome::Skipped {
                reason: RepairSkip::NoFixProduced,
            };
        }
        Err(err) => {
            return RepairOutcome::Skipped {
                reason: RepairSkip::WorktreeFailed(err.to_string()),
            };
        }
    }

    let fix = match worktree::commit_fix(&wt.path, &req.incident_id, &req.kind) {
        Ok(fix) => fix,
        Err(err) => {
            return RepairOutcome::Skipped {
                reason: RepairSkip::WorktreeFailed(err.to_string()),
            };
        }
    };

    let head_dest = req.worktrees_dir.join(format!("{}-head", req.incident_id));
    let head_wt = match worktree::add_detached(&req.checkout, &head_dest, &wt.base_sha) {
        Ok(h) => h,
        Err(err) => {
            return finish_completed(
                &req.incident_dir,
                wt.path,
                fix.branch,
                fix.sha,
                GateReport {
                    passed: false,
                    skip_apply_reason: Some(format!("HEAD worktree failed: {err}")),
                    layers: Vec::new(),
                },
            );
        }
    };
    let head_target = head_wt.path.join("target");
    let _ = fsutil::mkdir_0700(&head_target);

    let gate_input = GateInput {
        worktree: wt.path.clone(),
        head_worktree: head_wt.path.clone(),
        incident_dir: req.incident_dir.clone(),
        cargo_home: cargo_home.clone(),
        cargo_target_dir: cargo_target.clone(),
        head_target_dir: head_target,
        failing_test: diagnosis.failing_test.clone(),
        timeouts: req.gate_timeouts,
        base_sha: wt.base_sha.clone(),
    };
    let gate = match gate::run_layers(&gate_input).await {
        Ok(g) => g,
        Err(err) => GateReport {
            passed: false,
            skip_apply_reason: Some(err.to_string()),
            layers: Vec::new(),
        },
    };
    worktree::remove(&req.checkout, &head_dest);
    finish_completed(&req.incident_dir, wt.path, fix.branch, fix.sha, gate)
}

fn finish_completed(
    incident: &Path,
    worktree: PathBuf,
    branch: String,
    commit: String,
    gate: GateReport,
) -> RepairOutcome {
    let completed = RepairOutcome::Completed {
        worktree,
        branch,
        commit,
        gate,
    };
    write_repair_result(incident, &completed);
    completed
}

fn write_repair_result(incident: &Path, outcome: &RepairOutcome) {
    let body = match outcome {
        RepairOutcome::Completed {
            worktree,
            branch,
            commit,
            gate,
        } => format!(
            "{{\n  \"branch\": {branch:?},\n  \"commit\": {commit:?},\n  \"worktree\": {worktree:?},\n  \"gate_passed\": {}\n}}\n",
            gate.passed
        ),
        RepairOutcome::Skipped { reason } => format!("{{\"skipped\": {:?}}}\n", reason.user_line()),
    };
    let _ = fsutil::write_0600(&incident.join("repair-result.json"), body.as_bytes());
}

#[derive(Clone, Copy)]
enum Phase {
    Diagnose,
    Patch,
}

impl Phase {
    fn name(self) -> &'static str {
        match self {
            Self::Diagnose => "diagnose",
            Self::Patch => "patch",
        }
    }
}

async fn run_phase(req: &RepairRequest, wt: &Worktree, phase: Phase) -> Result<(), RepairSkip> {
    let session = wt
        .path
        .join(".hi")
        .join(format!("{}-session.jsonl", phase.name()));
    let prompt = match phase {
        Phase::Diagnose => diagnose_prompt(req, wt),
        Phase::Patch => patch_prompt(req, wt),
    };
    let _ = fsutil::write_0600(
        &req.incident_dir
            .join(format!("{}-prompt.txt", phase.name())),
        prompt.as_bytes(),
    );
    let model = match phase {
        Phase::Diagnose => req.models.diagnose.as_str(),
        Phase::Patch => req.models.patch.as_str(),
    };
    let timeout = match phase {
        Phase::Diagnose => req.timeouts.diagnose_timeout,
        Phase::Patch => req.timeouts.patch_timeout,
    };
    let argv = repair_argv(model, &session, &wt.path, &prompt);
    let status = spawn_repair(req, wt, &argv, phase.name(), timeout)
        .await
        .map_err(|err| RepairSkip::WorktreeFailed(err.to_string()))?;
    if matches!(status, SpawnStatus::Timeout) {
        return Err(match phase {
            Phase::Diagnose => RepairSkip::DiagnoseTimeout,
            Phase::Patch => RepairSkip::PatchTimeout,
        });
    }
    let _ = harvest_session(&session, &req.incident_dir, phase.name());
    let _ = harvest_diagnosis(&wt.path, &req.incident_dir);
    if !status.success() {
        return Err(match phase {
            Phase::Diagnose => RepairSkip::DiagnoseFailed,
            Phase::Patch => RepairSkip::PatchFailed,
        });
    }
    Ok(())
}

pub fn repair_argv(model: &str, session: &Path, review_target: &Path, prompt: &str) -> Vec<String> {
    vec![
        "--plain".into(),
        "--no-save".into(),
        "--model".into(),
        model.into(),
        "--session-file".into(),
        session.display().to_string(),
        "--review-target".into(),
        review_target.display().to_string(),
        prompt.into(),
    ]
}

enum SpawnStatus {
    Code(i32),
    Timeout,
}

impl SpawnStatus {
    fn success(&self) -> bool {
        matches!(self, Self::Code(0))
    }
}

async fn spawn_repair(
    req: &RepairRequest,
    wt: &Worktree,
    argv: &[String],
    phase: &str,
    timeout: Duration,
) -> Result<SpawnStatus> {
    let log = req.incident_dir.join(format!("{phase}.log"));
    fsutil::write_0600(&log, b"")?;
    let stdout = fs::OpenOptions::new()
        .append(true)
        .open(&log)
        .with_context(|| format!("open {}", log.display()))?;
    let stderr = stdout.try_clone()?;

    let cargo_home = wt.path.join(".hi/cargo-home");
    let cargo_target = wt.path.join("target");
    let state_home = wt.path.join(".hi/state");

    let mut command = tokio::process::Command::new(&req.hi_binary);
    command
        .args(argv)
        .current_dir(&wt.path)
        .kill_on_drop(true)
        .stdin(Stdio::null())
        .stdout(Stdio::from(stdout))
        .stderr(Stdio::from(stderr))
        .env(ENV_ROLE, "repair")
        .env(ENV_SUPERVISED, "1")
        .env(ENV_GENERATION, req.generation.to_string())
        .env("HI_SANDBOX", "workspace")
        .env("HI_SKIP_TUTORIAL", "1")
        .env("CARGO_HOME", &cargo_home)
        .env("CARGO_TARGET_DIR", &cargo_target)
        .env("CARGO_TERM_COLOR", "never")
        .env("XDG_STATE_HOME", &state_home)
        .env_remove(ENV_HEARTBEAT)
        .env_remove(ENV_EVENTS)
        .env_remove(ENV_INSTANCE)
        .env_remove(ENV_TURN_INTENT)
        .env_remove(ENV_PANIC_FILE)
        .env_remove(ENV_CRASH_DIR)
        .env_remove("SSH_AUTH_SOCK")
        .env_remove("GIT_ASKPASS");
    // SAFETY: pre_exec is between fork and exec; setpgid cannot race other threads.
    unsafe {
        command.pre_exec(|| {
            if libc::setpgid(0, 0) != 0 {
                return Err(io::Error::last_os_error());
            }
            Ok(())
        });
    }
    let mut child = command
        .spawn()
        .with_context(|| format!("spawning repair hi {}", req.hi_binary.display()))?;
    let pid = child.id().context("repair child pid")?;
    // Parent setpgid is the fallback if the child has not yet run pre_exec.
    unsafe {
        libc::setpgid(pid as i32, pid as i32);
    }
    tokio::select! {
        status = child.wait() => {
            let status = status.context("waiting for repair hi")?;
            Ok(SpawnStatus::Code(status.code().unwrap_or(1)))
        }
        _ = tokio::time::sleep(timeout) => {
            spawn::signal_group(pid as i32, libc::SIGTERM);
            tokio::select! {
                status = child.wait() => {
                    let _ = status;
                    Ok(SpawnStatus::Timeout)
                }
                _ = tokio::time::sleep(Duration::from_secs(2)) => {
                    spawn::signal_group(pid as i32, libc::SIGKILL);
                    let _ = child.wait().await;
                    Ok(SpawnStatus::Timeout)
                }
            }
        }
    }
}

struct Diagnosis {
    reproduced: bool,
    failing_test: Option<Vec<String>>,
}

fn parse_diagnosis(path: &Path) -> Diagnosis {
    let Ok(text) = fs::read_to_string(path) else {
        return Diagnosis {
            reproduced: false,
            failing_test: None,
        };
    };
    let mut reproduced = false;
    let mut failing_test = None;
    for line in text.lines() {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix("reproduced:") {
            reproduced = rest.trim().eq_ignore_ascii_case("yes");
        }
        if let Some(rest) = line.strip_prefix("failing_test:")
            && let Some(args) = gate::parse_failing_test(rest)
        {
            failing_test = Some(args);
        }
    }
    Diagnosis {
        reproduced,
        failing_test,
    }
}

fn agent_diagnosis_path(wt: &Path) -> PathBuf {
    wt.join(".hi/diagnosis.md")
}

fn stage_agent_inputs(incident: &Path, wt: &Path) -> std::io::Result<()> {
    let dest = wt.join(".hi");
    fsutil::mkdir_0700(&dest)?;
    copy_0600_if_exists(&incident.join("incident.json"), &dest.join("incident.json"))?;
    let repro_src = incident.join("repro");
    if repro_src.is_dir() {
        let repro_dst = dest.join("repro");
        fsutil::mkdir_0700(&repro_dst)?;
        if let Ok(entries) = fs::read_dir(&repro_src) {
            for entry in entries.flatten() {
                let from = entry.path();
                if from.is_file() {
                    let to = repro_dst.join(entry.file_name());
                    copy_0600_if_exists(&from, &to)?;
                    if entry.file_name() == "reproduction.sh" {
                        let _ = fsutil::chmod_0700_file(&to);
                    }
                }
            }
        }
    }
    Ok(())
}

fn harvest_diagnosis(wt: &Path, incident: &Path) -> std::io::Result<()> {
    copy_0600_if_exists(&agent_diagnosis_path(wt), &incident.join("diagnosis.md"))
}

fn harvest_session(session: &Path, incident: &Path, phase: &str) -> std::io::Result<()> {
    copy_0600_if_exists(session, &incident.join(format!("{phase}-session.jsonl")))
}

fn copy_0600_if_exists(from: &Path, to: &Path) -> std::io::Result<()> {
    if !from.is_file() {
        return Ok(());
    }
    let bytes = fs::read(from)?;
    fsutil::write_0600(to, &bytes)
}

fn diagnose_prompt(req: &RepairRequest, wt: &Worktree) -> String {
    let staged = wt.path.join(".hi");
    format!(
        "You are the Hi Sentinel repair agent (diagnose only).\n\
Read {skill} if it exists in the worktree.\n\
Writable worktree: {worktree}\n\
Staged incident files (inside the worktree): {staged}/incident.json and {staged}/repro/\n\
Checkout (do not write): {checkout}\n\
Summary: id={id} kind={kind}. Do not include or re-run any original user prompt.\n\
Tasks: (1) reproduce with cargo test -p … or {staged}/repro/reproduction.sh and HI_BINARY pointing at a worktree-built hi; \
(2) write {diag} with reproduced/failing_test/root_cause/files; (3) stop — do not patch.\n\
Bans: no writes outside the worktree; no git push; no commit to main; no ~/.ssh; no secrets; no live-project prompt replay.\n",
        skill = wt.path.join(SKILL_REL).display(),
        worktree = wt.path.display(),
        staged = staged.display(),
        checkout = req.checkout.display(),
        id = req.incident_id,
        kind = req.kind,
        diag = agent_diagnosis_path(&wt.path).display(),
    )
}

fn patch_prompt(_req: &RepairRequest, wt: &Worktree) -> String {
    let diag = agent_diagnosis_path(&wt.path);
    format!(
        "You are the Hi Sentinel repair agent (patch).\n\
Read {skill} if it exists and {diag}.\n\
Writable worktree: {worktree}\n\
Tasks: (1) patch only the worktree files named in diagnosis.md; \
(2) re-run the same cargo test / reproduction.sh Sentinel will run as the layered gate; (3) stop. Do not commit.\n\
Bans: no writes outside the worktree; no git push; no commit to main; no ~/.ssh; no secrets; no live-project prompt replay.\n",
        skill = wt.path.join(SKILL_REL).display(),
        diag = diag.display(),
        worktree = wt.path.display(),
    )
}

fn check_attempts(req: &RepairRequest) -> Result<(), RepairSkip> {
    let path = req.incident_dir.join("attempts");
    let n = fs::read_to_string(&path)
        .ok()
        .and_then(|s| s.trim().parse::<u32>().ok())
        .unwrap_or(0);
    if n >= req.timeouts.max_attempts_per_incident {
        return Err(RepairSkip::BudgetAttempts);
    }
    Ok(())
}

fn bump_attempts(incident: &Path) -> io::Result<()> {
    let path = incident.join("attempts");
    let n = fs::read_to_string(&path)
        .ok()
        .and_then(|s| s.trim().parse::<u32>().ok())
        .unwrap_or(0)
        .saturating_add(1);
    fsutil::write_0600(&path, format!("{n}\n").as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{DEFAULT_DIAGNOSE_MODEL, DEFAULT_PATCH_MODEL};

    #[test]
    fn argv_is_plain_no_save_model_session_review_only() {
        let argv = repair_argv(
            DEFAULT_DIAGNOSE_MODEL,
            Path::new("/tmp/inc/diagnose-session.jsonl"),
            Path::new("/tmp/wt"),
            "diagnose now",
        );
        assert_eq!(
            argv,
            vec![
                "--plain",
                "--no-save",
                "--model",
                DEFAULT_DIAGNOSE_MODEL,
                "--session-file",
                "/tmp/inc/diagnose-session.jsonl",
                "--review-target",
                "/tmp/wt",
                "diagnose now",
            ]
        );
        assert!(!argv.iter().any(|a| a.starts_with("--confirm-edits")));
        assert!(!argv.iter().any(|a| a == "--worktree" || a == "--provider"));
        assert_eq!(argv.iter().filter(|a| *a == "--model").count(), 1);
    }

    #[test]
    fn patch_argv_uses_resolved_patch_model() {
        let argv = repair_argv(
            DEFAULT_PATCH_MODEL,
            Path::new("/i/repair-session.jsonl"),
            Path::new("/wt"),
            "patch",
        );
        assert_eq!(argv[3], DEFAULT_PATCH_MODEL);
    }

    #[test]
    fn diagnosis_yes_parses_cargo_test_selector() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("diagnosis.md");
        fs::write(
            &path,
            "reproduced: yes\nfailing_test: -p hi-harness invariant_holds\n",
        )
        .unwrap();
        let d = parse_diagnosis(&path);
        assert!(d.reproduced);
        assert_eq!(
            d.failing_test.as_deref(),
            Some(["-p".into(), "hi-harness".into(), "invariant_holds".into()].as_slice())
        );
    }

    #[test]
    fn prompts_omit_original_user_prompt() {
        let req = RepairRequest {
            incident_id: "incident-1-aaaa".into(),
            incident_dir: PathBuf::from("/inc"),
            checkout: PathBuf::from("/co"),
            hi_binary: PathBuf::from("/bin/hi"),
            generation: 0,
            kind: "invariant".into(),
            worktrees_dir: PathBuf::from("/cache"),
            models: RepairModels {
                diagnose: DEFAULT_DIAGNOSE_MODEL.into(),
                patch: DEFAULT_PATCH_MODEL.into(),
            },
            timeouts: RepairConfig::default(),
            gate_timeouts: GateTimeouts::default(),
        };
        let wt = Worktree {
            path: PathBuf::from("/wt"),
            base_sha: "abc".into(),
        };
        let d = diagnose_prompt(&req, &wt);
        let p = patch_prompt(&req, &wt);
        for text in [&d, &p] {
            assert!(!text.contains("fix the parser"));
            assert!(text.contains("do not patch") || text.contains("Do not commit"));
            assert!(text.contains(".hi/diagnosis.md"));
            assert!(!text.contains("/inc/diagnosis.md"));
        }
    }
}
