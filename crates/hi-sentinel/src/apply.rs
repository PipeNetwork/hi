//! Sidecar install of verified repairs. Never moves `main`, never pushes.

use std::fs;
use std::io::{self, Write};
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Duration;

use anyhow::{Context, Result, bail};

use crate::budget;
use crate::fsutil;
use crate::paths;
use crate::rollback::{self, KnownGood, RecordInput};
use crate::spawn;
use hi_liveness::unix_ms;

const INSTALL_TERM_GRACE: Duration = Duration::from_secs(2);

pub struct ApplyRequest {
    pub incident_id: String,
    pub worktree: PathBuf,
    pub checkout: PathBuf,
    pub checkout_dirty: bool,
    pub checkout_sha: Option<String>,
    pub hi_binary: PathBuf,
    pub state_dir: PathBuf,
    pub generation: u32,
    pub auto_apply: bool,
    pub stdin_is_tty: bool,
    pub apply_answer: Option<String>,
    pub overwrite_answer: Option<String>,
    pub cargo: PathBuf,
    pub max_repairs_per_session: u32,
    pub max_modifications_per_hour: u32,
    pub install_timeout: Duration,
}

#[derive(Debug)]
pub enum ApplyOutcome {
    Applied { note: String },
    Refused { reason: ApplyRefuse, note: String },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ApplyRefuse {
    ConservativeNoTty,
    UserDeclined,
    GenerationHalt,
    HourlyCap,
    Blake3Mismatch,
    CorruptKnownGood,
    InstallFailed(String),
}

pub fn cargo_bin() -> PathBuf {
    std::env::var_os("CARGO")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("cargo"))
}

pub fn halt_user_line(
    generation: u32,
    incident_id: &str,
    known_good: Option<&KnownGood>,
    current_sha: Option<&str>,
    current_blake3: &str,
    checkout: Option<&Path>,
    incidents_dir: &Path,
) -> String {
    let kg_sha = known_good
        .map(|k| k.checkout_sha.as_str())
        .filter(|s| !s.is_empty())
        .unwrap_or("unknown");
    let kg_b3 = known_good
        .map(|k| k.binary_blake3.as_str())
        .filter(|s| !s.is_empty())
        .unwrap_or("unknown");
    let cur_sha = current_sha.filter(|s| !s.is_empty()).unwrap_or("unknown");
    let cd = checkout
        .map(|p| p.display().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "~/hi".into());
    format!(
        "hi: Sentinel stopped after generation {generation} failed ({incident_id}).\n    known-good: {kg_sha} / hi blake3 {kg_b3}\n    current:    {cur_sha} / hi blake3 {current_blake3}\n    Manual: cd {cd} && git status ; ls {}",
        incidents_dir.display()
    )
}

pub fn maybe_apply(req: &ApplyRequest) -> ApplyOutcome {
    if req.generation >= req.max_repairs_per_session {
        let kg = rollback::load(&req.state_dir).ok();
        let current_b3 = rollback::hash_file(&req.hi_binary).unwrap_or_default();
        let note = halt_user_line(
            req.generation,
            &req.incident_id,
            kg.as_ref(),
            None,
            &current_b3,
            Some(req.checkout.as_path()).filter(|p| !p.as_os_str().is_empty()),
            &paths::incidents_dir(&req.state_dir),
        );
        return ApplyOutcome::Refused {
            reason: ApplyRefuse::GenerationHalt,
            note,
        };
    }

    let now = unix_ms();
    let log = paths::apply_log_path(&req.state_dir);
    if budget::hourly_cap_reached(&log, req.max_modifications_per_hour, now) {
        return refused(
            ApplyRefuse::HourlyCap,
            format!(
                "hourly apply cap ({}) reached; not applying.",
                req.max_modifications_per_hour
            ),
        );
    }

    if req.checkout_dirty && !req.checkout.as_os_str().is_empty() {
        eprintln!(
            "hi: checkout {} is dirty; apply will not merge or reset main. Repair base is dirty HEAD.",
            req.checkout.display()
        );
    }

    let recorded = match load_or_record_known_good(req) {
        Ok(kg) => kg,
        Err(outcome) => return outcome,
    };
    if !rollback::binary_matches(&recorded, &req.hi_binary) {
        return refused(
            ApplyRefuse::Blake3Mismatch,
            "running hi no longer matches known-good.json; not applying.".into(),
        );
    }

    spawn::prepare_interactive_prompt();
    if !req.auto_apply {
        if !req.stdin_is_tty && req.apply_answer.is_none() {
            eprintln!(
                "hi: repair verified for {}; not applying (no tty).",
                req.incident_id
            );
            return refused(
                ApplyRefuse::ConservativeNoTty,
                format!(
                    "Repair available on branch autofix/{}. Not applied.",
                    req.incident_id
                ),
            );
        }
        let question = format!(
            "Repair succeeded for {}. Apply sidecar binaries from worktree? [y/N] ",
            req.incident_id
        );
        if !confirm(req.stdin_is_tty, req.apply_answer.as_deref(), &question) {
            return refused(
                ApplyRefuse::UserDeclined,
                format!(
                    "Repair available on branch autofix/{}. Not applied.",
                    req.incident_id
                ),
            );
        }
    }

    match install_sidecar(req, recorded) {
        Ok(applied) => applied,
        Err(err) => refused(
            ApplyRefuse::InstallFailed(err.to_string()),
            format!("sidecar install failed: {err}"),
        ),
    }
}

fn install_sidecar(req: &ApplyRequest, mut known_good: KnownGood) -> Result<ApplyOutcome> {
    let root = &req.state_dir;
    let bin_dir = paths::sidecar_bin_dir(root);
    fsutil::mkdir_0700(root).context("sentinel state dir")?;
    fsutil::mkdir_0700(&bin_dir).context("sidecar bin dir")?;

    let sidecar_hi = bin_dir.join("hi");
    let sidecar_sentinel = bin_dir.join("hi-sentinel");
    let prev_hi = bin_dir.join("hi.prev");
    let prev_sentinel = bin_dir.join("hi-sentinel.prev");

    let hi_src = if sidecar_hi.is_file() {
        sidecar_hi.clone()
    } else {
        req.hi_binary.clone()
    };
    rollback::copy_prev(&hi_src, &prev_hi)?;
    let sentinel_src = if sidecar_sentinel.is_file() {
        sidecar_sentinel.clone()
    } else {
        sibling_sentinel(&req.hi_binary)
    };
    rollback::copy_prev(&sentinel_src, &prev_sentinel)?;

    if let Err(err) = cargo_install(req, "crates/hi-cli", root) {
        let _ = rollback::copy_prev(&prev_hi, &sidecar_hi);
        return Err(err);
    }
    if let Err(err) = cargo_install(req, "crates/hi-sentinel", root) {
        let _ = rollback::copy_prev(&prev_hi, &sidecar_hi);
        let _ = rollback::copy_prev(&prev_sentinel, &sidecar_sentinel);
        return Err(err);
    }
    anyhow::ensure!(
        sidecar_hi.is_file() && sidecar_sentinel.is_file(),
        "cargo install did not write sidecar binaries"
    );
    let _ = fsutil::chmod_0700(&bin_dir);
    let _ = fsutil::chmod_0700_file(&sidecar_hi);
    let _ = fsutil::chmod_0700_file(&sidecar_sentinel);

    known_good.prev_binary_path = prev_hi.display().to_string();
    rollback::write_known_good(&req.state_dir, &known_good)?;
    budget::append_apply(
        &paths::apply_log_path(&req.state_dir),
        &req.incident_id,
        unix_ms(),
    )?;

    Ok(ApplyOutcome::Applied {
        note: maybe_overwrite_running(req, &sidecar_hi).unwrap_or_default(),
    })
}

fn cargo_install(req: &ApplyRequest, package_path: &str, root: &Path) -> Result<()> {
    spawn::prepare_interactive_prompt();
    let mut cmd = Command::new(&req.cargo);
    cmd.args([
        "install",
        "--path",
        package_path,
        "--locked",
        "--force",
        "--root",
        &root.display().to_string(),
    ])
    .current_dir(&req.worktree)
    .env("CARGO_TERM_COLOR", "never")
    .stdin(Stdio::null())
    .stdout(Stdio::inherit())
    .stderr(Stdio::inherit());
    // Keep install artifacts inside the worktree; do not inherit the live tree's CARGO_TARGET_DIR.
    let target = req.worktree.join("target");
    let cargo_home = req.worktree.join(".hi/cargo-home");
    let _ = fsutil::mkdir_0700(&target);
    let _ = fsutil::mkdir_0700(&cargo_home);
    cmd.env("CARGO_TARGET_DIR", &target);
    cmd.env("CARGO_HOME", &cargo_home);
    // Own a process group so a timeout can SIGTERM/SIGKILL descendants, not just cargo.
    unsafe {
        cmd.pre_exec(|| {
            if libc::setpgid(0, 0) != 0 {
                return Err(io::Error::last_os_error());
            }
            Ok(())
        });
    }
    let mut child = cmd
        .spawn()
        .with_context(|| format!("running {} install {package_path}", req.cargo.display()))?;
    let pid = child.id();
    unsafe {
        libc::setpgid(pid as i32, pid as i32);
    }
    let status = spawn::wait_install_child(
        &mut child,
        pid as i32,
        req.install_timeout,
        INSTALL_TERM_GRACE,
    )
    .map_err(|err| anyhow::anyhow!("cargo install {package_path}: {err}"))?;
    if !status.success() {
        bail!("cargo install {package_path} failed with {status}");
    }
    Ok(())
}

fn load_or_record_known_good(req: &ApplyRequest) -> Result<KnownGood, ApplyOutcome> {
    match rollback::load(&req.state_dir) {
        Ok(kg) => Ok(kg),
        Err(err) if rollback::is_missing(&err) => record_current(req).map_err(|e| {
            refused(
                ApplyRefuse::InstallFailed(e.to_string()),
                format!("could not record known-good: {e}"),
            )
        }),
        Err(err) if rollback::is_corrupt(&err) => Err(refused(
            ApplyRefuse::CorruptKnownGood,
            "known-good.json is corrupt; not applying.".into(),
        )),
        Err(err) => Err(refused(
            ApplyRefuse::InstallFailed(err.to_string()),
            format!("could not read known-good: {err}"),
        )),
    }
}

fn record_current(req: &ApplyRequest) -> io::Result<KnownGood> {
    let prev = paths::sidecar_bin_dir(&req.state_dir).join("hi.prev");
    rollback::record(&RecordInput {
        state_dir: &req.state_dir,
        checkout_path: Some(req.checkout.as_path()).filter(|p| !p.as_os_str().is_empty()),
        checkout_sha: req.checkout_sha.as_deref(),
        binary_path: &req.hi_binary,
        prev_binary_path: prev.is_file().then_some(prev.as_path()),
    })
}

fn maybe_overwrite_running(req: &ApplyRequest, sidecar_hi: &Path) -> Option<String> {
    if same_path(&req.hi_binary, sidecar_hi) {
        return None;
    }
    // Mixing cargo-install output with a `target/` build can downgrade a dirty debug binary.
    if !req.auto_apply
        && confirm(
            req.stdin_is_tty,
            req.overwrite_answer.as_deref(),
            &format!(
                "Overwrite {} with sidecar hi? [y/N] ",
                req.hi_binary.display()
            ),
        )
    {
        match atomic_replace(sidecar_hi, &req.hi_binary) {
            Ok(()) => return None,
            Err(err) => {
                return Some(format!(
                    "could not replace {}: {err}. cargo install from worktree succeeded; replace {} manually",
                    req.hi_binary.display(),
                    req.hi_binary.display()
                ));
            }
        }
    }
    Some(format!(
        "cargo install from worktree succeeded; replace {} manually",
        req.hi_binary.display()
    ))
}

fn atomic_replace(src: &Path, dest: &Path) -> io::Result<()> {
    let tmp = dest.with_extension("new");
    fs::copy(src, &tmp)?;
    fsutil::chmod_0700_file(&tmp)?;
    fs::rename(&tmp, dest)?;
    Ok(())
}

fn same_path(a: &Path, b: &Path) -> bool {
    match (fs::canonicalize(a), fs::canonicalize(b)) {
        (Ok(aa), Ok(bb)) => aa == bb,
        _ => a == b,
    }
}

fn sibling_sentinel(hi_binary: &Path) -> PathBuf {
    hi_binary
        .parent()
        .map(|p| p.join("hi-sentinel"))
        .unwrap_or_else(|| PathBuf::from("hi-sentinel"))
}

fn confirm(stdin_is_tty: bool, injected: Option<&str>, question: &str) -> bool {
    if injected.is_none() && !stdin_is_tty {
        return false;
    }
    eprint!("{question}");
    let _ = io::stderr().flush();
    let line = match injected {
        Some(answer) => answer.to_string(),
        None => {
            let mut line = String::new();
            if io::stdin().read_line(&mut line).is_err() {
                return false;
            }
            line
        }
    };
    matches!(line.trim(), "y" | "Y" | "yes" | "YES")
}

fn refused(reason: ApplyRefuse, note: String) -> ApplyOutcome {
    ApplyOutcome::Refused { reason, note }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_fixture;
    use crate::worktree;
    use std::time::Instant;

    struct Fixture {
        _root: tempfile::TempDir,
        req: ApplyRequest,
        checkout: PathBuf,
        running_hi: PathBuf,
        cargo_log: PathBuf,
        head: String,
        branch: String,
    }

    fn write_fake_cargo(path: &Path, log: &Path) {
        let log = log.display();
        fs::write(
            path,
            format!(
                r#"#!/bin/sh
set -eu
log='{log}'
root=""
pkg=""
prev=""
n=0
if [ -f "$log/count" ]; then
  n=$(cat "$log/count")
fi
n=$((n+1))
echo "$n" > "$log/count"
for arg in "$@"; do
  if [ "$prev" = "--root" ]; then root=$arg; prev=""; continue; fi
  if [ "$prev" = "--path" ]; then pkg=$arg; prev=""; continue; fi
  case "$arg" in
    --root|--path) prev=$arg ;;
  esac
done
{{
  echo "PWD=$(pwd)"
  echo "CARGO_HOME=${{CARGO_HOME-}}"
  echo "CARGO_TARGET_DIR=${{CARGO_TARGET_DIR-}}"
  echo "pkg=$pkg"
}} > "$log/spawn-$n.env"
mkdir -p "$root/bin"
case "$pkg" in
  crates/hi-cli)
    printf 'sidecar-hi\n' > "$root/bin/hi"
    chmod 700 "$root/bin/hi"
    ;;
  crates/hi-sentinel)
    printf 'sidecar-sentinel\n' > "$root/bin/hi-sentinel"
    chmod 700 "$root/bin/hi-sentinel"
    ;;
  *)
    echo "unexpected path $pkg" >&2
    exit 1
    ;;
esac
exit 0
"#
            ),
        )
        .unwrap();
        fsutil::chmod_0700_file(path).unwrap();
    }

    fn fixture(auto_apply: bool, stdin_is_tty: bool) -> Fixture {
        let root = tempfile::tempdir().unwrap();
        let checkout = test_fixture::minimal_checkout(root.path());
        fs::write(checkout.join("dirty.txt"), b"unstaged").unwrap();
        let head = git_stdout(&checkout, &["rev-parse", "HEAD"]);
        let branch = worktree::current_branch(&checkout).unwrap();
        let state = root.path().join("state");
        fsutil::mkdir_0700(&state).unwrap();
        let running_dir = root.path().join("run");
        fsutil::mkdir_0700(&running_dir).unwrap();
        let running_hi = running_dir.join("hi");
        fs::write(&running_hi, b"old-hi").unwrap();
        fsutil::chmod_0700_file(&running_hi).unwrap();
        fs::write(running_dir.join("hi-sentinel"), b"old-sentinel").unwrap();
        fsutil::chmod_0700_file(&running_dir.join("hi-sentinel")).unwrap();
        let cargo = root.path().join("fake-cargo");
        let cargo_log = root.path().join("cargo-log");
        fsutil::mkdir_0700(&cargo_log).unwrap();
        write_fake_cargo(&cargo, &cargo_log);
        let worktree = root.path().join("wt");
        fsutil::mkdir_0700(&worktree).unwrap();
        rollback::record(&RecordInput {
            state_dir: &state,
            checkout_path: Some(&checkout),
            checkout_sha: Some(&head),
            binary_path: &running_hi,
            prev_binary_path: None,
        })
        .unwrap();
        let req = ApplyRequest {
            incident_id: "incident-1842-a3f1".into(),
            worktree,
            checkout: checkout.clone(),
            checkout_dirty: true,
            checkout_sha: Some(head.clone()),
            hi_binary: running_hi.clone(),
            state_dir: state,
            generation: 0,
            auto_apply,
            stdin_is_tty,
            apply_answer: None,
            overwrite_answer: None,
            cargo,
            max_repairs_per_session: 2,
            max_modifications_per_hour: 3,
            install_timeout: Duration::from_secs(30),
        };
        Fixture {
            _root: root,
            req,
            checkout,
            running_hi,
            cargo_log,
            head,
            branch,
        }
    }

    fn env_path(env: &str, key: &str) -> PathBuf {
        let value = env
            .lines()
            .find_map(|l| l.strip_prefix(&format!("{key}=")))
            .unwrap_or_else(|| panic!("missing {key} in {env}"));
        fs::canonicalize(value).unwrap_or_else(|_| PathBuf::from(value))
    }

    fn git_stdout(dir: &Path, args: &[&str]) -> String {
        let output = Command::new("git")
            .arg("-C")
            .arg(dir)
            .args(args)
            .output()
            .unwrap();
        String::from_utf8_lossy(&output.stdout).trim().to_string()
    }

    fn assert_checkout_untouched(fx: &Fixture) {
        assert_eq!(worktree::current_branch(&fx.checkout).unwrap(), fx.branch);
        assert_eq!(git_stdout(&fx.checkout, &["rev-parse", "HEAD"]), fx.head);
        assert_eq!(
            fs::read_to_string(fx.checkout.join("dirty.txt")).unwrap(),
            "unstaged"
        );
        assert!(
            git_stdout(&fx.checkout, &["status", "--porcelain=v1"]).contains("dirty.txt"),
            "apply must not merge or reset the dirty checkout"
        );
        assert_eq!(fs::read_to_string(&fx.running_hi).unwrap(), "old-hi");
    }

    #[test]
    fn apply_refused_without_tty_in_conservative_mode() {
        let fx = fixture(false, false);
        let outcome = maybe_apply(&fx.req);
        match outcome {
            ApplyOutcome::Refused {
                reason: ApplyRefuse::ConservativeNoTty,
                ..
            } => {}
            other => panic!("expected ConservativeNoTty, got {other:?}"),
        }
        assert!(
            !paths::sidecar_bin_dir(&fx.req.state_dir)
                .join("hi")
                .exists()
        );
        assert_checkout_untouched(&fx);
    }

    #[test]
    fn apply_flag_writes_sidecar() {
        let fx = fixture(true, false);
        let outcome = maybe_apply(&fx.req);
        let ApplyOutcome::Applied { .. } = outcome else {
            panic!("expected Applied, got {outcome:?}");
        };
        let bin = paths::sidecar_bin_dir(&fx.req.state_dir);
        let sidecar_hi = bin.join("hi");
        let prev = bin.join("hi.prev");
        assert_eq!(
            fs::read_to_string(&sidecar_hi).unwrap().trim(),
            "sidecar-hi"
        );
        assert_eq!(
            fs::read_to_string(bin.join("hi-sentinel")).unwrap().trim(),
            "sidecar-sentinel"
        );
        assert_eq!(fs::read_to_string(&prev).unwrap(), "old-hi");
        assert_eq!(
            fs::read_to_string(bin.join("hi-sentinel.prev")).unwrap(),
            "old-sentinel"
        );
        let kg = rollback::load(&fx.req.state_dir).unwrap();
        assert_eq!(kg.prev_binary_path, prev.display().to_string());
        assert_eq!(
            fsutil::unix_mode(&paths::known_good_path(&fx.req.state_dir)).unwrap(),
            0o600
        );
        assert_eq!(
            fsutil::unix_mode(&paths::apply_log_path(&fx.req.state_dir)).unwrap(),
            0o600
        );
        assert_eq!(
            budget::applies_last_hour(&paths::apply_log_path(&fx.req.state_dir), unix_ms()),
            1
        );
        assert_checkout_untouched(&fx);
        assert!(
            kg.binary_blake3 == rollback::hash_file(&fx.running_hi).unwrap(),
            "known-good must stay the pre-apply binary"
        );
        let env1 = fs::read_to_string(fx.cargo_log.join("spawn-1.env")).unwrap();
        let env2 = fs::read_to_string(fx.cargo_log.join("spawn-2.env")).unwrap();
        let want_target = fs::canonicalize(fx.req.worktree.join("target")).unwrap();
        let want_home = fs::canonicalize(fx.req.worktree.join(".hi/cargo-home")).unwrap();
        let want_cwd = fs::canonicalize(&fx.req.worktree).unwrap();
        let checkout = fs::canonicalize(&fx.checkout).unwrap();
        for env in [&env1, &env2] {
            let target = env_path(env, "CARGO_TARGET_DIR");
            let home = env_path(env, "CARGO_HOME");
            let cwd = env_path(env, "PWD");
            assert_eq!(target, want_target, "{env}");
            assert_eq!(home, want_home, "{env}");
            assert_eq!(cwd, want_cwd, "{env}");
            assert!(
                !target.starts_with(&checkout) && !home.starts_with(&checkout),
                "cargo dirs must not be under the checkout: {env}"
            );
        }
    }

    #[test]
    fn generation_2_halt() {
        let mut fx = fixture(true, false);
        fx.req.generation = 2;
        let outcome = maybe_apply(&fx.req);
        match outcome {
            ApplyOutcome::Refused {
                reason: ApplyRefuse::GenerationHalt,
                note,
            } => {
                assert!(note.contains("generation 2 failed"));
                assert!(note.contains("incident-1842-a3f1"));
            }
            other => panic!("expected GenerationHalt, got {other:?}"),
        }
        assert!(
            !paths::sidecar_bin_dir(&fx.req.state_dir)
                .join("hi")
                .exists()
        );
        assert_checkout_untouched(&fx);
    }

    #[test]
    fn hourly_cap() {
        let fx = fixture(true, false);
        let log = paths::apply_log_path(&fx.req.state_dir);
        let now = unix_ms();
        budget::append_apply(&log, "a", now).unwrap();
        budget::append_apply(&log, "b", now).unwrap();
        budget::append_apply(&log, "c", now).unwrap();
        let outcome = maybe_apply(&fx.req);
        match outcome {
            ApplyOutcome::Refused {
                reason: ApplyRefuse::HourlyCap,
                ..
            } => {}
            other => panic!("expected HourlyCap, got {other:?}"),
        }
        assert!(
            !paths::sidecar_bin_dir(&fx.req.state_dir)
                .join("hi")
                .exists()
        );
        assert_checkout_untouched(&fx);
    }

    #[test]
    fn target_debug_hi_is_not_overwritten_without_y() {
        let mut fx = fixture(true, false);
        let target_hi = fx.checkout.join("target").join("debug").join("hi");
        fs::create_dir_all(target_hi.parent().unwrap()).unwrap();
        fs::write(&target_hi, b"debug-build").unwrap();
        fx.req.hi_binary = target_hi.clone();
        rollback::record(&RecordInput {
            state_dir: &fx.req.state_dir,
            checkout_path: Some(&fx.checkout),
            checkout_sha: Some(&fx.head),
            binary_path: &target_hi,
            prev_binary_path: None,
        })
        .unwrap();
        let outcome = maybe_apply(&fx.req);
        let ApplyOutcome::Applied { note, .. } = outcome else {
            panic!("expected Applied, got {outcome:?}");
        };
        assert_eq!(fs::read_to_string(&target_hi).unwrap(), "debug-build");
        assert!(note.contains("replace"));
        assert!(note.contains("manually"));
    }

    #[test]
    fn corrupt_known_good_refuses_without_rewrite() {
        let fx = fixture(true, false);
        let path = paths::known_good_path(&fx.req.state_dir);
        fs::write(&path, b"{not-json").unwrap();
        let outcome = maybe_apply(&fx.req);
        match outcome {
            ApplyOutcome::Refused {
                reason: ApplyRefuse::CorruptKnownGood,
                ..
            } => {}
            other => panic!("expected CorruptKnownGood, got {other:?}"),
        }
        assert_eq!(fs::read(&path).unwrap(), b"{not-json");
        assert!(
            !paths::sidecar_bin_dir(&fx.req.state_dir)
                .join("hi")
                .exists()
        );
    }

    #[test]
    fn cargo_install_times_out_and_kills_the_group() {
        let mut fx = fixture(true, false);
        let hang = fx._root.path().join("hang-cargo");
        fs::write(&hang, "#!/bin/sh\nexec sleep 30\n").unwrap();
        fsutil::chmod_0700_file(&hang).unwrap();
        fx.req.cargo = hang;
        fx.req.install_timeout = Duration::from_millis(300);
        let started = Instant::now();
        let outcome = maybe_apply(&fx.req);
        assert!(
            started.elapsed() < Duration::from_secs(5),
            "timeout must not wait out sleep 30"
        );
        match outcome {
            ApplyOutcome::Refused {
                reason: ApplyRefuse::InstallFailed(msg),
                ..
            } => assert!(msg.contains("timed out"), "{msg}"),
            other => panic!("expected InstallFailed timeout, got {other:?}"),
        }
    }

    #[test]
    fn sigint_kills_cargo_install_group() {
        let mut fx = fixture(true, false);
        let hang = fx._root.path().join("hang-cargo");
        let ready = fx._root.path().join("hang-ready");
        fs::write(
            &hang,
            format!(
                "#!/bin/sh\necho started > '{}'\nexec sleep 30\n",
                ready.display()
            ),
        )
        .unwrap();
        fsutil::chmod_0700_file(&hang).unwrap();
        fx.req.cargo = hang;
        fx.req.install_timeout = Duration::from_secs(30);
        std::thread::spawn(move || {
            for _ in 0..200 {
                if ready.is_file() && spawn::install_pgid() > 1 {
                    spawn::interrupt_install();
                    return;
                }
                std::thread::sleep(Duration::from_millis(20));
            }
        });
        let started = Instant::now();
        let outcome = maybe_apply(&fx.req);
        assert!(
            started.elapsed() < Duration::from_secs(5),
            "interrupt must not wait out sleep 30"
        );
        match outcome {
            ApplyOutcome::Refused {
                reason: ApplyRefuse::InstallFailed(msg),
                ..
            } => assert!(msg.contains("interrupted"), "{msg}"),
            other => panic!("expected InstallFailed interrupted, got {other:?}"),
        }
    }
}
