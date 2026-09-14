//! Repair-agent tests against an isolated fixture checkout — never live `~/hi`.

use std::fs;
use std::path::{Path, PathBuf};
use std::time::Duration;

use crate::classify::{BugKind, Class, Confidence};
use crate::config::{
    DEFAULT_DIAGNOSE_MODEL, DEFAULT_PATCH_MODEL, MonitorConfig, RepairConfig, RepairModels,
    SupervisorConfig,
};
use crate::fsutil;
use crate::gate::GateTimeouts;
use crate::repair::{RepairOutcome, RepairRequest, RepairSkip, maybe_repair, run_repair};
use crate::test_fixture::{self, StubSpec};
use crate::worktree;

fn models() -> RepairModels {
    RepairModels {
        diagnose: DEFAULT_DIAGNOSE_MODEL.into(),
        patch: DEFAULT_PATCH_MODEL.into(),
    }
}

fn timeouts() -> RepairConfig {
    RepairConfig {
        diagnose_timeout: Duration::from_secs(15),
        patch_timeout: Duration::from_secs(15),
        max_repairs_per_session: 2,
        max_attempts_per_incident: 1,
        max_modifications_per_hour: 3,
    }
}

fn gate_timeouts() -> GateTimeouts {
    GateTimeouts {
        layer0: Duration::from_secs(120),
        check: Duration::from_secs(120),
        test: Duration::from_secs(120),
        clippy: Duration::from_secs(120),
    }
}

fn request(
    root: &Path,
    checkout: PathBuf,
    cargo: bool,
) -> (RepairRequest, PathBuf, PathBuf, PathBuf) {
    let incident_id = "incident-1842-a3f1".to_string();
    let incident = root.join("incident");
    fsutil::mkdir_0700(&incident).unwrap();
    fsutil::mkdir_0700(&incident.join("repro")).unwrap();
    if cargo {
        test_fixture::write_hi_binary_repro(&incident.join("repro/reproduction.sh"));
    } else {
        fsutil::write_0600(
            &incident.join("repro/reproduction.sh"),
            b"#!/bin/sh\nexit 2\n",
        )
        .unwrap();
        fsutil::chmod_0700_file(&incident.join("repro/reproduction.sh")).unwrap();
    }
    let log = root.join("stub-log");
    fsutil::mkdir_0700(&log).unwrap();
    let outside = root.join("outside-live-project");
    fs::create_dir_all(&outside).unwrap();
    let stub = root.join("stub-hi");
    test_fixture::write_stub_hi(
        &stub,
        &StubSpec {
            log: log.clone(),
            outside: Some(outside.clone()),
        },
    );
    let worktrees = root.join("autofix-worktrees");
    fsutil::mkdir_0700(&worktrees).unwrap();
    let req = RepairRequest {
        incident_id,
        incident_dir: incident,
        checkout,
        hi_binary: stub,
        generation: 0,
        kind: "invariant".into(),
        worktrees_dir: worktrees,
        models: models(),
        timeouts: timeouts(),
        gate_timeouts: gate_timeouts(),
    };
    (req, log, outside, root.join("autofix-worktrees"))
}

#[tokio::test]
async fn maybe_repair_skips_without_checkout() {
    let dir = tempfile::tempdir().unwrap();
    let cfg = SupervisorConfig {
        child_program: PathBuf::from("/bin/sh"),
        child_args: Vec::new(),
        hi_binary: PathBuf::from("/bin/sh"),
        original_argv: vec!["hi".into()],
        checkout: None,
        apply: false,
        monitor: MonitorConfig::default(),
        state_dir: dir.path().join("state"),
        workspace: dir.path().to_path_buf(),
        generation: 0,
        inherit_stdio: false,
        extra_env: Vec::new(),
        once: true,
    };
    let class = Class::HarnessBug {
        kind: BugKind::Invariant,
        confidence: Confidence::High,
    };
    let incident = dir.path().join("incident-1");
    fsutil::mkdir_0700(&incident).unwrap();
    let outcome = maybe_repair(&cfg, &class, &incident).await;
    match outcome {
        RepairOutcome::Skipped {
            reason: RepairSkip::NoCheckout,
        } => {}
        other => panic!("expected NoCheckout, got {other:?}"),
    }
}

#[tokio::test]
async fn diagnose_without_repro_skips_patch() {
    let root = tempfile::tempdir().unwrap();
    let checkout = test_fixture::minimal_checkout(root.path());
    let (req, log, outside, _) = request(root.path(), checkout, false);
    let incident = req.incident_dir.clone();
    fs::write(log.join("no_repro"), b"1").unwrap();
    let outcome = run_repair(req).await;
    match outcome {
        RepairOutcome::Skipped {
            reason: RepairSkip::DiagnoseNoRepro,
        } => {}
        other => panic!("expected DiagnoseNoRepro, got {other:?}"),
    }
    assert_eq!(fs::read_to_string(log.join("count")).unwrap().trim(), "1");
    assert!(!log.join("spawn-2.argv").exists());
    assert!(incident.join("diagnosis.md").is_file());
    assert!(!outside.join("pwned").exists());
    assert!(log.join("outside").is_file());
    let argv = fs::read_to_string(log.join("spawn-1.argv")).unwrap();
    assert!(argv.lines().any(|l| l == "--plain"));
    assert!(argv.lines().any(|l| l == "--no-save"));
    assert!(argv.lines().any(|l| l == DEFAULT_DIAGNOSE_MODEL));
    assert!(!argv.contains("--confirm-edits"));
    assert!(!argv.contains("fix the parser"));
    let env = fs::read_to_string(log.join("spawn-1.env")).unwrap();
    assert!(env.contains("HI_SANDBOX=workspace"));
    assert!(env.contains("HI_SENTINEL_ROLE=repair"));
    assert!(env.contains("HI_SENTINEL_SUPERVISED=1"));
    assert!(env.contains("CARGO_HOME=") && env.contains("/.hi/cargo-home"));
    assert!(env.contains("CARGO_TARGET_DIR=") && env.contains("/target"));
    assert!(
        env.lines().any(|l| l == "SSH_AUTH_SOCK="),
        "repair child must drop SSH_AUTH_SOCK: {env}"
    );
}

#[tokio::test]
async fn fixture_invariant_repro_commits_only_autofix_branch() {
    let root = tempfile::tempdir().unwrap();
    let checkout = test_fixture::cargo_checkout(root.path());
    let before = worktree::current_branch(&checkout).unwrap();
    let head = std::process::Command::new("git")
        .args(["-C", &checkout.display().to_string(), "rev-parse", "HEAD"])
        .output()
        .unwrap();
    let head_sha = String::from_utf8_lossy(&head.stdout).trim().to_string();
    let (req, log, outside, _wts) = request(root.path(), checkout.clone(), true);
    let incident = req.incident_dir.clone();
    let outcome = run_repair(req).await;
    let RepairOutcome::Completed {
        worktree: wt,
        branch,
        commit,
        gate,
    } = outcome
    else {
        panic!("expected completed repair, got {outcome:?}");
    };
    assert_eq!(branch, "autofix/incident-1842-a3f1");
    assert_eq!(worktree::current_branch(&wt).unwrap(), branch);
    assert_eq!(worktree::current_branch(&checkout).unwrap(), before);
    assert_ne!(commit, head_sha);
    assert!(
        gate.passed,
        "gate failed: {:?}",
        gate.layers
            .iter()
            .map(|l| (&l.command, l.passed, &l.reason))
            .collect::<Vec<_>>()
    );
    assert!(
        gate.layers
            .iter()
            .any(|l| l.layer == 0 && l.passed && !l.skipped)
    );
    assert!(
        gate.layers
            .iter()
            .any(|l| l.layer == 3 && l.passed && !l.skipped),
        "hi-harness diff vs base SHA must run layer 3, got {:?}",
        gate.layers
    );
    assert!(
        wt.join(".hi/diagnosis.md").is_file(),
        "agent diagnosis must land inside the worktree"
    );
    assert!(
        incident.join("diagnosis.md").is_file(),
        "sentinel must copy diagnosis.md into the incident dir"
    );
    assert!(!outside.join("pwned").exists());
    let count: u32 = fs::read_to_string(log.join("count"))
        .unwrap()
        .trim()
        .parse()
        .unwrap();
    assert_eq!(count, 2);
    let argv1 = fs::read_to_string(log.join("spawn-1.argv")).unwrap();
    let argv2 = fs::read_to_string(log.join("spawn-2.argv")).unwrap();
    assert!(argv1.lines().any(|l| l == DEFAULT_DIAGNOSE_MODEL));
    assert!(argv2.lines().any(|l| l == DEFAULT_PATCH_MODEL));
    let src = fs::read_to_string(wt.join("crates/hi-harness/src/lib.rs")).unwrap();
    assert!(src.contains("true"), "{src}");
    let checkout_src = fs::read_to_string(checkout.join("crates/hi-harness/src/lib.rs")).unwrap();
    assert!(
        checkout_src.contains("false"),
        "live checkout must stay unmodified: {checkout_src}"
    );
}
