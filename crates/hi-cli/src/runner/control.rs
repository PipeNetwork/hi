//! Local supervisor fences. Absolute boot-clock leases cannot gain lifetime
//! while a JSONL frame sits in a pipe or either process is suspended.
use anyhow::{Result, ensure};
use hi_harness::TurnCancellation;
use std::sync::{Arc, Mutex};
use tokio::sync::{Notify, oneshot};
use uuid::Uuid;

pub fn boot_millis() -> u64 {
    #[cfg(unix)]
    {
        let mut value = libc::timespec {
            tv_sec: 0,
            tv_nsec: 0,
        };
        #[cfg(target_os = "linux")]
        let clock = libc::CLOCK_BOOTTIME;
        #[cfg(not(target_os = "linux"))]
        let clock = libc::CLOCK_MONOTONIC;
        if unsafe { libc::clock_gettime(clock, &mut value) } == 0 {
            return value.tv_sec as u64 * 1000 + value.tv_nsec as u64 / 1_000_000;
        }
    }
    // Fail closed on a platform without a shared monotonic clock.
    u64::MAX
}
struct Lease {
    until: u64,
    sequence: u64,
}
pub struct Control {
    lease: Mutex<Lease>,
    checkpoint: Mutex<Option<(Uuid, oneshot::Sender<()>)>>,
    changed: Notify,
    cancel: TurnCancellation,
    kill: Option<std::fs::File>,
}
impl Control {
    pub fn new(until: u64, cancel: TurnCancellation) -> Result<Arc<Self>> {
        Self::valid(until)?;
        Ok(Arc::new(Self {
            lease: Mutex::new(Lease { until, sequence: 0 }),
            checkpoint: Mutex::new(None),
            changed: Notify::new(),
            cancel,
            kill: None,
        }))
    }
    #[cfg(unix)]
    pub fn supervised(until: u64, attempt: Uuid, cancel: TurnCancellation) -> Result<Arc<Self>> {
        use std::os::unix::fs::{MetadataExt, OpenOptionsExt};
        let membership = std::fs::read_to_string("/proc/self/cgroup")?;
        let relative = membership
            .lines()
            .find_map(|line| line.strip_prefix("0::"))
            .ok_or_else(|| anyhow::anyhow!("runner requires a delegated cgroup"))?;
        ensure!(
            !relative.split('/').any(|part| part == "..")
                && relative.ends_with(&format!("/pipe-code-{attempt}")),
            "runner is outside its assigned process cgroup"
        );
        let path = std::path::Path::new("/sys/fs/cgroup").join(relative.trim_start_matches('/'));
        ensure!(
            path.symlink_metadata()?.uid() == unsafe { libc::geteuid() },
            "execution cgroup ownership changed"
        );
        let file = std::fs::OpenOptions::new()
            .write(true)
            .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC)
            .open(path.join("cgroup.kill"))?;
        let mut control = Self::new(until, cancel)?;
        Arc::get_mut(&mut control).unwrap().kill = Some(file);
        Ok(control)
    }
    fn expire(&self) {
        self.cancel.cancel();
        self.stop_scope();
    }
    pub fn stop_scope(&self) {
        if let Some(file) = &self.kill {
            // The independent child fence also covers a suspended/crashed
            // supervisor and tools that escaped their original process group.
            // Tool starts/results were fsynced before dispatch/completion.
            use std::io::Write;
            let _ = (&*file).write_all(b"1");
        }
    }
    #[cfg(not(unix))]
    pub fn supervised(_until: u64, _attempt: Uuid, _cancel: TurnCancellation) -> Result<Arc<Self>> {
        anyhow::bail!("guest execution requires a Linux cgroup v2 service")
    }
    fn valid(until: u64) -> Result<()> {
        ensure!(
            (1..=60_000).contains(&until.saturating_sub(boot_millis())),
            "invalid local execution lease"
        );
        Ok(())
    }
    pub fn live(&self) -> bool {
        self.output_live() && !self.cancel.is_cancelled()
    }
    pub fn output_live(&self) -> bool {
        if self.lease.lock().unwrap().until <= boot_millis() {
            self.expire();
            return false;
        }
        true
    }
    pub fn renew(&self, sequence: u64, until: u64) -> Result<()> {
        let mut lease = self.lease.lock().unwrap();
        ensure!(
            !self.cancel.is_cancelled() && lease.until > boot_millis(),
            "expired lease cannot resume execution"
        );
        Self::valid(until)?;
        ensure!(
            sequence == lease.sequence + 1 && until >= lease.until,
            "invalid lease sequence"
        );
        *lease = Lease { until, sequence };
        self.changed.notify_one();
        Ok(())
    }
    pub async fn watch(self: Arc<Self>) {
        loop {
            let remaining = self
                .lease
                .lock()
                .unwrap()
                .until
                .saturating_sub(boot_millis());
            if remaining == 0 {
                self.expire();
                return;
            }
            tokio::select! {
                _ = self.changed.notified() => {},
                _ = tokio::time::sleep(std::time::Duration::from_millis(remaining.min(1000))) => {},
            }
        }
    }
    pub fn checkpoint(&self) -> Result<(Uuid, oneshot::Receiver<()>)> {
        ensure!(self.live(), "execution lease lost before checkpoint");
        let mut pending = self.checkpoint.lock().unwrap();
        ensure!(pending.is_none(), "checkpoint already pending");
        let id = Uuid::new_v4();
        let (send, receive) = oneshot::channel();
        *pending = Some((id, send));
        Ok((id, receive))
    }
    pub fn acknowledge(&self, id: Uuid, sha: &str) -> Result<()> {
        ensure!(self.live(), "execution lease lost during checkpoint");
        ensure!(
            sha.len() == 40
                && sha
                    .bytes()
                    .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)),
            "invalid checkpoint commit"
        );
        let mut pending = self.checkpoint.lock().unwrap();
        ensure!(
            pending
                .as_ref()
                .is_some_and(|(expected, _)| *expected == id),
            "checkpoint receipt does not match pending batch"
        );
        pending
            .take()
            .unwrap()
            .1
            .send(())
            .map_err(|_| anyhow::anyhow!("checkpoint no longer pending"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[tokio::test]
    #[cfg(target_os = "linux")]
    #[ignore = "run explicitly in a delegated systemd service"]
    async fn delegated_lease_fence_kills_the_attempt_without_its_supervisor() {
        use std::{
            fs,
            io::Write,
            os::{fd::AsRawFd, unix::process::ExitStatusExt},
            time::Duration,
        };
        if let Ok(attempt) = std::env::var("HI_FENCE_CHILD_ATTEMPT") {
            let attempt = Uuid::parse_str(&attempt).unwrap();
            let control =
                Control::supervised(boot_millis() + 300, attempt, TurnCancellation::new()).unwrap();
            let mut descendant = tokio::process::Command::new("/usr/bin/setsid");
            descendant.args(["/bin/sleep", "60"]);
            let child = descendant.spawn().unwrap();
            fs::write(
                std::env::var("HI_FENCE_CHILD_PID_FILE").unwrap(),
                child.id().unwrap().to_string(),
            )
            .unwrap();
            control.watch().await;
            panic!("a supervised expired lease must terminate its entire cgroup");
        }
        let membership = fs::read_to_string("/proc/self/cgroup").unwrap();
        let parent = std::path::Path::new("/sys/fs/cgroup").join(
            membership
                .lines()
                .find_map(|s| s.strip_prefix("0::"))
                .unwrap()
                .trim_start_matches('/'),
        );
        let attempt = Uuid::new_v4();
        let scope = parent.join(format!("pipe-code-{attempt}"));
        fs::create_dir(&scope).unwrap();
        struct Cleanup(std::path::PathBuf);
        impl Drop for Cleanup {
            fn drop(&mut self) {
                let _ = fs::OpenOptions::new()
                    .write(true)
                    .open(self.0.join("cgroup.kill"))
                    .and_then(|mut f| f.write_all(b"1"));
                let _ = fs::remove_dir(&self.0);
            }
        }
        let _cleanup = Cleanup(scope.clone());
        let temp = tempfile::tempdir().unwrap();
        let pid_file = temp.path().join("child.pid");
        let membership = fs::OpenOptions::new()
            .write(true)
            .open(scope.join("cgroup.procs"))
            .unwrap();
        let mut child = tokio::process::Command::new(std::env::current_exe().unwrap());
        child.args(["--exact","runner::control::tests::delegated_lease_fence_kills_the_attempt_without_its_supervisor","--ignored"]).env("HI_FENCE_CHILD_ATTEMPT",attempt.to_string()).env("HI_FENCE_CHILD_PID_FILE",&pid_file).kill_on_drop(true);
        unsafe {
            child.pre_exec(move || {
                if libc::write(membership.as_raw_fd(), b"0".as_ptr().cast(), 1) != 1 {
                    return Err(std::io::Error::last_os_error());
                }
                Ok(())
            });
        }
        let mut child = child.spawn().unwrap();
        let status = tokio::time::timeout(Duration::from_secs(5), child.wait())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(status.signal(), Some(libc::SIGKILL));
        let pid = fs::read_to_string(pid_file).unwrap();
        if let Ok(status) = fs::read_to_string(format!("/proc/{pid}/status")) {
            assert!(
                status
                    .lines()
                    .any(|s| s.starts_with("State:") && s.contains("Z (zombie)")),
                "detached tool survived lease expiry"
            );
        }
        assert!(
            fs::read_to_string(scope.join("cgroup.events"))
                .unwrap()
                .lines()
                .any(|s| s == "populated 0")
        );
    }
    #[tokio::test]
    async fn lost_supervisor_expires_and_cannot_revive_a_lease() {
        let cancel = TurnCancellation::new();
        let control = Control::new(boot_millis() + 25, cancel.clone()).unwrap();
        let watch = tokio::spawn(control.clone().watch());
        tokio::time::timeout(std::time::Duration::from_secs(2), cancel.cancelled())
            .await
            .unwrap();
        assert!(!control.live());
        assert!(control.renew(1, boot_millis() + 30_000).is_err());
        watch.await.unwrap();
    }
    #[tokio::test]
    async fn checkpoint_requires_exact_receipt_and_lease_frames_cannot_replay() {
        let control = Control::new(boot_millis() + 10_000, TurnCancellation::new()).unwrap();
        let (id, mut receipt) = control.checkpoint().unwrap();
        assert!(receipt.try_recv().is_err());
        assert!(
            control
                .acknowledge(Uuid::new_v4(), &"a".repeat(40))
                .is_err()
        );
        assert!(control.acknowledge(id, "main").is_err());
        control.acknowledge(id, &"a".repeat(40)).unwrap();
        receipt.await.unwrap();
        assert!(control.acknowledge(id, &"a".repeat(40)).is_err());
        control.renew(1, boot_millis() + 20_000).unwrap();
        assert!(control.renew(1, boot_millis() + 30_000).is_err());
        assert!(Control::new(boot_millis() + 60_001, TurnCancellation::new()).is_err());
    }
}
