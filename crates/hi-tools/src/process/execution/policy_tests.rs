use super::*;
use crate::{ProcessRunner, sandbox::SandboxPolicy};

struct PreservedPolicy;

impl Drop for PreservedPolicy {
    fn drop(&mut self) {
        preserve_detached_descendants(false);
    }
}

struct DetachedProcess(i32);

impl Drop for DetachedProcess {
    fn drop(&mut self) {
        unsafe { libc::kill(self.0, libc::SIGKILL) };
    }
}

#[tokio::test]
async fn adoptable_keep_background_preserves_detached_service() {
    detached_service_obeys_policy(true).await;
}

#[tokio::test]
async fn adoptable_default_policy_kills_detached_service() {
    detached_service_obeys_policy(false).await;
}

async fn detached_service_obeys_policy(preserve: bool) {
    let _lock = crate::background::TEST_LOCK.lock().await;
    let _policy = PreservedPolicy;
    preserve_detached_descendants(preserve);
    let root = tempfile::tempdir().unwrap();
    let runner = ProcessRunner::new_with_policy(root.path(), SandboxPolicy::Off).unwrap();
    let result = runner
        .run_shell_adoptable(
            "sleep 600 >/dev/null 2>&1 & echo $!",
            Duration::from_secs(5),
            &mut |_| {},
        )
        .await
        .unwrap();
    let execution = match result {
        AdoptableOutcome::Completed(execution) => execution,
        AdoptableOutcome::StillRunning(mut child) => {
            kill_process_group(&child.child);
            let _ = child.child.kill().await;
            panic!("a completed launcher must not be adopted");
        }
    };
    let detached = DetachedProcess(execution.outcome.stdout_summary.trim().parse().unwrap());
    assert_eq!(execution.status, ToolStatus::Succeeded);
    tokio::time::sleep(Duration::from_millis(200)).await;
    let alive = unsafe { libc::kill(detached.0, 0) } == 0;
    assert_eq!(alive, preserve, "adoptable Bash must honor keep-background");
}
