use super::*;
use std::time::Duration;

use async_trait::async_trait;
use tokio::sync::Semaphore;

struct ProcessLifecycleGate {
    calls: Mutex<Vec<crate::BackgroundJobTerminal>>,
    entered: Semaphore,
    release: Semaphore,
}

impl Default for ProcessLifecycleGate {
    fn default() -> Self {
        Self {
            calls: Mutex::new(Vec::new()),
            entered: Semaphore::new(0),
            release: Semaphore::new(0),
        }
    }
}

#[async_trait]
impl crate::BackgroundJobLifecycle for ProcessLifecycleGate {
    async fn register(
        &self,
        _registration: crate::BackgroundJobRegistration,
    ) -> Result<(), String> {
        Ok(())
    }

    async fn observe_terminal(
        &self,
        _id: &crate::BackgroundJobId,
        terminal: crate::BackgroundJobTerminal,
        _detail: Option<String>,
    ) -> Result<crate::BackgroundJobPublication, String> {
        self.entered.add_permits(1);
        self.release.acquire().await.unwrap().forget();
        self.calls.lock().unwrap().push(terminal);
        Ok(crate::BackgroundJobPublication::Published)
    }

    async fn pending(&self, _source_id: &str) -> Vec<crate::BackgroundJobId> {
        Vec::new()
    }

    async fn settle_after_workspace(
        &self,
        _pending: &[crate::BackgroundJobId],
    ) -> Result<(), String> {
        Ok(())
    }
}

#[tokio::test]
async fn process_success_is_not_visible_before_lifecycle_settlement() {
    let directory = tempfile::tempdir().unwrap();
    let state = directory.path().join("state");
    std::fs::create_dir_all(&state).unwrap();
    let before = crate::effects::workspace_snapshot(directory.path(), &state)
        .await
        .unwrap();
    let runner =
        crate::ProcessRunner::new_with_policy(directory.path(), crate::sandbox::SandboxPolicy::Off)
            .unwrap();
    let registry = BackgroundRegistry::default();
    let lifecycle = Arc::new(ProcessLifecycleGate::default());
    registry.set_job_lifecycle(lifecycle.clone());
    let id = registry
        .spawn_tracked(&runner, "printf done", directory.path(), &state, before)
        .await
        .unwrap();

    tokio::time::timeout(Duration::from_secs(2), lifecycle.entered.acquire())
        .await
        .unwrap()
        .unwrap()
        .forget();
    assert_eq!(
        registry.outcome(&id).unwrap().state,
        crate::BackgroundState::Running
    );
    lifecycle.release.add_permits(1);
    let deadline = tokio::time::Instant::now() + Duration::from_secs(2);
    while registry.outcome(&id).unwrap().state == crate::BackgroundState::Running {
        assert!(tokio::time::Instant::now() < deadline);
        tokio::task::yield_now().await;
    }
    assert_eq!(
        lifecycle.calls.lock().unwrap().as_slice(),
        [crate::BackgroundJobTerminal::Succeeded]
    );
}

#[test]
fn running_effect_snapshot_is_not_sealed_when_process_exits_during_scan() {
    let inner = BgInner {
        output: String::new(),
        dropped_bytes: 0,
        read_position: 0,
        state: BgState::Exited(Some(0)),
        reaped: true,
        terminal_effects: None,
        empty_polls: 0,
    };
    assert!(should_seal_terminal_effects(&inner, true));
    assert!(
        !should_seal_terminal_effects(&inner, false),
        "a snapshot begun while running must be recomputed after reap"
    );
}

#[test]
fn background_capacity_rejects_new_reservations() {
    let registry = BackgroundRegistry::default();
    registry
        .reserved_slots
        .store(MAX_BG_PROCS, Ordering::Relaxed);

    let error = registry
        .reserve_slot()
        .expect_err("a full registry must reject another child");
    assert!(error.to_string().contains("capacity reached"));
}

/// Poll until the process reports it is no longer running, or time out.
async fn poll_until_done(id: &str) -> String {
    for _ in 0..200 {
        let out = poll(id).unwrap();
        if !out.contains("running") {
            return out;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    panic!("background process {id} never finished");
}

#[tokio::test]
async fn background_captures_output_and_exit_code() {
    let _guard = TEST_LOCK.lock().await;
    let id = spawn("echo hi-bg").unwrap();
    let combined = poll_until_done(&id).await;
    // `poll_until_done` returns the poll that observed the exit; the echoed
    // line should be in that same drain (output is flushed before exit).
    assert!(
        combined.contains("hi-bg") || poll(&id).unwrap().contains("hi-bg"),
        "expected output, got: {combined:?}"
    );
    assert!(combined.contains("exited with code 0"), "got: {combined:?}");
    assert_eq!(outcome(&id).unwrap().state, crate::BackgroundState::Exited);
    assert_eq!(outcome(&id).unwrap().exit_code, Some(0));
}

#[tokio::test]
async fn drained_terminal_poll_restates_output_tail() {
    let _guard = TEST_LOCK.lock().await;
    let id = spawn("echo tail-marker").unwrap();
    let first = poll_until_done(&id).await;
    // Drain any straggling flush so the next poll is genuinely empty.
    if !first.contains("tail-marker") {
        poll(&id).unwrap();
    }
    let drained = poll(&id).unwrap();
    assert!(drained.contains("exited with code 0"), "got: {drained:?}");
    assert!(
        drained.contains("already delivered") && drained.contains("tail-marker"),
        "a drained terminal poll must restate the output tail so the \
             caller can conclude without re-polling: {drained:?}"
    );
}

#[tokio::test]
async fn drained_terminal_poll_of_silent_process_says_so() {
    let _guard = TEST_LOCK.lock().await;
    let id = spawn("true").unwrap();
    poll_until_done(&id).await;
    let drained = poll(&id).unwrap();
    assert!(drained.contains("produced no output"), "got: {drained:?}");
}

#[test]
fn output_tail_bounds_and_aligns_to_line_start() {
    assert_eq!(output_tail("short\n"), "short");
    let long = format!("{}\nlast line", "x".repeat(5000));
    let tail = output_tail(&long);
    assert!(tail.starts_with("… (earlier output elided)\n"));
    assert!(tail.ends_with("last line"));
    assert!(tail.len() < 2100, "tail stays bounded: {}", tail.len());
}

#[test]
fn handle_ids_name_the_job_and_order_by_suffix() {
    // Real names, not opaque `sh_N`: a live run showed the model
    // cold-guessing `sh_1` with nothing running; a slug it cannot
    // predict is not guessable, and polls/kills read as the job.
    assert_eq!(
        handle_id("cargo test --quiet -p hi-tools", 3),
        "cargo-test_3"
    );
    assert_eq!(
        handle_id("RUST_LOG=debug git push origin main", 7),
        "git-push_7"
    );
    assert_eq!(handle_id("sleep 600", 1), "sleep_1");
    // Unparseable → shell_title's generic label, never empty.
    assert_eq!(handle_id("", 2), "shell_2");
    assert_eq!(handle_id("---", 5), "sh_5");
    // A bare `task` command must not mint ids in the task_ namespace.
    assert_eq!(handle_id("task", 4), "sh_4");
    // The slug is bounded and the counter still parses for prune order.
    let long = handle_id("extraordinarily-long-command-name-beyond-any-cap xyz", 12);
    assert!(long.len() <= 28, "bounded: {long}");
    assert_eq!(id_num(&long), 12);
    assert_eq!(id_num("cargo-test_9"), 9);
    assert_eq!(id_num("sh_1"), 1);
    assert_eq!(id_num("bg_5"), 5);
}

#[tokio::test]
async fn background_returns_immediately_for_long_process() {
    let _guard = TEST_LOCK.lock().await;
    // A 600s sleep must not block spawn; it returns an id at once.
    let id = tokio::time::timeout(Duration::from_secs(2), async { spawn("sleep 600") })
        .await
        .expect("spawn must not block")
        .unwrap();
    let out = poll(&id).unwrap();
    assert!(out.contains("running"), "got: {out:?}");
    assert_eq!(outcome(&id).unwrap().state, crate::BackgroundState::Running);
    kill(&id).unwrap();
}

#[tokio::test]
async fn idle_running_poll_does_not_re_echo_command() {
    let _guard = TEST_LOCK.lock().await;
    let id = spawn("sleep 600").unwrap();
    let out = poll(&id).unwrap();
    assert!(
        out.contains("still running — no new output"),
        "idle poll status: {out:?}"
    );
    assert!(
        !out.contains("`sleep 600`") && !out.contains("sleep 600"),
        "idle running polls must not re-echo the full command (looks like a hung UI loop): {out:?}"
    );
    // Auto-name may include the program (`sleep`) — that is the title, not a dump.
    kill(&id).unwrap();
}

#[tokio::test]
async fn snapshot_lists_each_job_with_command_and_status() {
    let _guard = TEST_LOCK.lock().await;
    let registry = BackgroundRegistry::default();
    let runner = crate::ProcessRunner::from_current_dir().unwrap();
    let id = registry.spawn(&runner, "sleep 600").unwrap();

    let snap = registry.snapshot();
    let entry = snap.iter().find(|(eid, _, _)| *eid == id);
    assert!(
        entry.is_some(),
        "snapshot includes the spawned job: {snap:?}"
    );
    let (_, command, status) = entry.unwrap();
    assert_eq!(command, "sleep 600");
    assert_eq!(status, "running");

    // Snapshot is non-consuming: polling afterwards still returns fresh output.
    registry.kill(&id).unwrap();
    let snap_after = registry.snapshot();
    let (_, _, status_after) = snap_after.iter().find(|(eid, _, _)| *eid == id).unwrap();
    assert_eq!(status_after, "killed");
}

#[tokio::test]
async fn quiescent_barrier_rejects_live_children_and_waits_after_kill() {
    let _guard = TEST_LOCK.lock().await;
    let registry = BackgroundRegistry::default();
    let runner = crate::ProcessRunner::from_current_dir().unwrap();
    let id = registry.spawn(&runner, "sleep 600").unwrap();

    let error = registry
        .ensure_quiescent_and_reaped()
        .await
        .expect_err("a running child must block workspace teardown");
    assert!(error.to_string().contains(&id), "{error:#}");

    registry.kill(&id).unwrap();
    registry
        .ensure_quiescent_and_reaped()
        .await
        .expect("a killed child must be fully reaped before teardown returns");
    let (_, _, status) = registry
        .snapshot()
        .into_iter()
        .find(|(candidate, _, _)| candidate == &id)
        .unwrap();
    assert_eq!(status, "killed");
}

#[tokio::test]
async fn background_kill_stops_the_process() {
    let _guard = TEST_LOCK.lock().await;
    let id = spawn("sleep 600").unwrap();
    let killed = kill(&id).unwrap();
    assert!(killed.contains("stopped"), "got: {killed:?}");
    // After the kill propagates, a poll reports it is no longer running.
    let out = poll_until_done(&id).await;
    assert!(out.contains("stopped"), "got: {out:?}");
    // Killing again is idempotent.
    assert!(kill(&id).unwrap().contains("already"), "second kill");
}

#[tokio::test]
async fn kill_started_after_reaps_auto_backgrounded_but_spares_deliberate_jobs() {
    let _guard = TEST_LOCK.lock().await;
    let runner = crate::ProcessRunner::from_current_dir().unwrap();
    let registry = BackgroundRegistry::default();
    let before = registry.ids();
    // Deliberate `run_in_background` work started after the baseline —
    // an 800 GB download must not die because its turn ended.
    let download = registry.spawn(&runner, "sleep 600").unwrap();
    // Auto-backgrounded: a foreground command adopted after outgrowing
    // its budget — incidental turn state, still reaped.
    let mut child = runner.spawn_shell("sleep 600").unwrap();
    let pgid = child.id().map(|p| p as i32);
    let stdout = child.stdout.take();
    let stderr = child.stderr.take();
    let root = std::env::current_dir().unwrap();
    let state = std::env::temp_dir().join(format!("hi-origin-state-{}", std::process::id()));
    let _ = std::fs::create_dir_all(&state);
    let snapshot = crate::effects::workspace_snapshot(&root, &state)
        .await
        .unwrap();
    let adopted = registry
        .adopt(
            "sleep 600",
            child,
            stdout,
            stderr,
            pgid,
            String::new(),
            (root, state, snapshot),
        )
        .await
        .unwrap();

    let killed = registry
        .kill_started_after_and_reap(&before)
        .await
        .expect("turn cleanup must wait for the adopted child to be reaped");

    assert_eq!(killed, 1, "only the auto-backgrounded process is reaped");
    assert_eq!(
        registry.outcome(&download).unwrap().state,
        crate::BackgroundState::Running,
        "a deliberate run_in_background job survives turn-scoped cleanup"
    );
    assert_eq!(
        registry.outcome(&adopted).unwrap().state,
        crate::BackgroundState::Killed
    );
    assert!(
        registry
            .processes
            .lock()
            .unwrap()
            .get(&adopted)
            .unwrap()
            .inner
            .lock()
            .unwrap()
            .reaped,
        "the cancellation barrier returned before native reap"
    );
    registry.kill_all();
}

#[tokio::test]
async fn poll_unknown_id_errors() {
    assert!(poll("sh_does_not_exist").is_err());
    assert!(kill("sh_does_not_exist").is_err());
}

#[tokio::test]
async fn unknown_handles_are_recorded_with_registry_emptiness() {
    let registry = BackgroundRegistry::default();
    // Empty registry: the id cannot have been pruned — it was guessed.
    assert!(registry.poll("ghost_1").is_err());
    let unknown = registry.unknown_handles();
    assert_eq!(unknown.len(), 1);
    assert_eq!(unknown[0].id, "ghost_1");
    assert!(unknown[0].registry_was_empty);

    // A real process makes later misses ambiguous (possibly pruned).
    let runner = crate::ProcessRunner::from_current_dir().unwrap();
    let id = registry.spawn(&runner, "sleep 600").unwrap();
    assert!(registry.poll("ghost_2").is_err());
    let unknown = registry.unknown_handles();
    assert_eq!(unknown.len(), 2);
    assert_eq!(unknown[0].id, "ghost_2");
    assert!(!unknown[0].registry_was_empty);
    assert_eq!(unknown[1].id, "ghost_1");
    assert!(unknown[1].registry_was_empty);
    registry.kill(&id).unwrap();
}

#[tokio::test]
async fn unknown_handle_log_is_bounded() {
    let registry = BackgroundRegistry::default();
    for n in 0..(MAX_UNKNOWN_HANDLES + 5) {
        assert!(registry.poll(&format!("ghost_{n}")).is_err());
    }
    let unknown = registry.unknown_handles();
    assert_eq!(unknown.len(), MAX_UNKNOWN_HANDLES);
    // Oldest misses are dropped first; the most recent is kept.
    assert_eq!(unknown[0].id, format!("ghost_{}", MAX_UNKNOWN_HANDLES + 4));
    assert_eq!(unknown[MAX_UNKNOWN_HANDLES - 1].id, format!("ghost_{}", 5));
}

#[test]
fn default_wait_budget_escalates_and_caps() {
    assert_eq!(
        default_poll_wait_budget(0, Some(15)),
        Duration::from_secs(15)
    );
    assert_eq!(
        default_poll_wait_budget(1, Some(15)),
        Duration::from_secs(30)
    );
    assert_eq!(
        default_poll_wait_budget(2, Some(15)),
        Duration::from_secs(60)
    );
    assert_eq!(
        default_poll_wait_budget(4, Some(15)),
        Duration::from_secs(240)
    );
    assert_eq!(
        default_poll_wait_budget(63, Some(15)),
        Duration::from_secs(240),
        "cap holds for arbitrary streaks"
    );
    assert_eq!(
        default_poll_wait_budget(3, Some(0)),
        Duration::ZERO,
        "0 = instant"
    );
}

#[tokio::test]
async fn default_poll_parks_on_the_watcher_until_output() {
    let _guard = TEST_LOCK.lock().await;
    let registry = BackgroundRegistry::default();
    let runner = crate::ProcessRunner::from_current_dir().unwrap();
    // Quiet for 400ms, then emits: the defaulted poll must park on the
    // change notification and wake with the output — not return an
    // instant "no new output" that costs the caller a round-trip.
    let id = registry
        .spawn(&runner, "sleep 0.4; echo woke-the-watcher; sleep 600")
        .unwrap();

    let started = std::time::Instant::now();
    let out = registry.poll_wait_default(&id).await.unwrap();

    assert!(
        out.contains("woke-the-watcher"),
        "default poll returns the awaited output: {out:?}"
    );
    assert!(
        started.elapsed() >= Duration::from_millis(300),
        "must actually have parked: {:?}",
        started.elapsed()
    );
    assert!(
        started.elapsed() < Duration::from_secs(10),
        "must wake on output, not sleep out the budget: {:?}",
        started.elapsed()
    );
    registry.kill(&id).unwrap();
}

#[tokio::test]
async fn empty_polls_escalate_and_fresh_output_resets() {
    let _guard = TEST_LOCK.lock().await;
    let registry = BackgroundRegistry::default();
    let runner = crate::ProcessRunner::from_current_dir().unwrap();
    let id = registry.spawn(&runner, "echo first; sleep 600").unwrap();
    let strikes = |registry: &BackgroundRegistry, id: &str| {
        let processes = registry.processes.lock().unwrap();
        let proc = processes.get(id).unwrap();
        let inner = proc.inner.lock().unwrap();
        inner.empty_polls
    };

    // Wait for the first line, then drain it: counter resets on output.
    let drained = registry
        .poll_wait(&id, Duration::from_secs(5))
        .await
        .unwrap();
    assert!(drained.contains("first"), "got: {drained:?}");
    assert_eq!(strikes(&registry, &id), 0);

    // Two instant empty peeks escalate the streak.
    let _ = registry.poll(&id).unwrap();
    let _ = registry.poll(&id).unwrap();
    assert_eq!(strikes(&registry, &id), 2);
    registry.kill(&id).unwrap();
}

#[tokio::test]
async fn poll_wait_blocks_until_new_output_arrives() {
    let _guard = TEST_LOCK.lock().await;
    let registry = BackgroundRegistry::default();
    let runner = crate::ProcessRunner::from_current_dir().unwrap();
    let id = registry
        .spawn(&runner, "sleep 0.4; echo late-line; sleep 600")
        .unwrap();

    let started = std::time::Instant::now();
    let out = registry
        .poll_wait(&id, Duration::from_secs(10))
        .await
        .unwrap();

    assert!(
        out.contains("late-line"),
        "the wait should return the fresh output: {out:?}"
    );
    assert!(
        started.elapsed() < Duration::from_secs(8),
        "must wake on output, not sleep out the full wait: {:?}",
        started.elapsed()
    );
    registry.kill(&id).unwrap();
}

#[tokio::test]
async fn poll_wait_streaming_emits_output_before_return() {
    let _guard = TEST_LOCK.lock().await;
    let registry = BackgroundRegistry::default();
    let runner = crate::ProcessRunner::from_current_dir().unwrap();
    let id = registry
        .spawn(&runner, "sleep 0.3; echo streamed-line; sleep 600")
        .unwrap();

    let mut seen = String::new();
    let out = registry
        .poll_wait_streaming(&id, Duration::from_secs(10), &mut |line| {
            seen.push_str(line);
            seen.push('\n');
        })
        .await
        .unwrap();

    assert!(
        seen.contains("streamed-line"),
        "live callback should see output before the poll returns: seen={seen:?} out={out:?}"
    );
    registry.kill(&id).unwrap();
}

#[tokio::test]
async fn poll_wait_times_out_to_idle_status_on_a_quiet_process() {
    let _guard = TEST_LOCK.lock().await;
    let registry = BackgroundRegistry::default();
    let runner = crate::ProcessRunner::from_current_dir().unwrap();
    let id = registry.spawn(&runner, "sleep 600").unwrap();

    let started = std::time::Instant::now();
    let out = registry
        .poll_wait(&id, Duration::from_millis(200))
        .await
        .unwrap();

    assert!(
        out.contains("still running — no new output"),
        "a timed-out wait reports genuine idleness: {out:?}"
    );
    assert!(started.elapsed() >= Duration::from_millis(180));
    registry.kill(&id).unwrap();
}

#[tokio::test]
async fn poll_wait_wakes_promptly_when_the_process_is_killed() {
    let _guard = TEST_LOCK.lock().await;
    let registry = std::sync::Arc::new(BackgroundRegistry::default());
    let runner = crate::ProcessRunner::from_current_dir().unwrap();
    let id = registry.spawn(&runner, "sleep 600").unwrap();

    let waiter = {
        let registry = registry.clone();
        let id = id.clone();
        tokio::spawn(async move { registry.poll_wait(&id, Duration::from_secs(30)).await })
    };
    tokio::time::sleep(Duration::from_millis(100)).await;
    registry.kill(&id).unwrap();

    let out = tokio::time::timeout(Duration::from_secs(5), waiter)
        .await
        .expect("kill must wake the waiter")
        .unwrap()
        .unwrap();
    assert!(out.contains("stopped"), "got: {out:?}");
}

#[tokio::test]
async fn adopt_keeps_child_running_and_seeds_output() {
    let _guard = TEST_LOCK.lock().await;
    let runner = crate::ProcessRunner::from_current_dir().unwrap();
    // Simulate the auto-background handoff: spawn a still-running child and
    // adopt it with a seed capturing the "foreground" output so far.
    let mut child = runner.spawn_shell("sleep 600").unwrap();
    let pgid = child.id().map(|p| p as i32);
    let stdout = child.stdout.take();
    let stderr = child.stderr.take();
    let root = std::env::current_dir().unwrap();
    let state = std::env::temp_dir().join(format!("hi-adopt-state-{}", std::process::id()));
    let _ = std::fs::create_dir_all(&state);
    let snapshot = crate::effects::workspace_snapshot(&root, &state)
        .await
        .unwrap();
    let id = TEST_REGISTRY
        .adopt(
            "sleep 600",
            child,
            stdout,
            stderr,
            pgid,
            "already-printed\n".to_string(),
            (root, state.clone(), snapshot),
        )
        .await
        .unwrap();

    let polled = poll(&id).unwrap();
    assert!(polled.contains("running"), "adopted child runs: {polled:?}");
    assert!(
        polled.contains("already-printed"),
        "seed output is visible on first poll: {polled:?}"
    );
    assert_eq!(outcome(&id).unwrap().state, crate::BackgroundState::Running);
    kill(&id).unwrap();
    let done = poll_until_done(&id).await;
    assert!(done.contains("stopped"), "got: {done:?}");
}

#[test]
fn char_boundary_helper_lands_on_boundaries() {
    let s = "a😀b"; // 😀 is 4 bytes at index 1..5
    assert_eq!(char_boundary_at_or_after(s, 2), 5);
    assert_eq!(char_boundary_at_or_after(s, 1), 1);
    assert_eq!(char_boundary_at_or_after(s, 99), s.len());
}

fn registry_with_retained_output(output: String) -> (BackgroundRegistry, String) {
    let registry = BackgroundRegistry::default();
    let id = "overflow-test_1".to_string();
    let proc = Arc::new(BgProc {
        command: "overflow-test".to_string(),
        title: "overflow-test".to_string(),
        pgid: None,
        origin: BgOrigin::Requested,
        effect_baseline: None,
        managed_job: None,
        inner: Mutex::new(BgInner::running(output)),
        reaped: Notify::new(),
        changed: Notify::new(),
    });
    registry.processes.lock().unwrap().insert(id.clone(), proc);
    (registry, id)
}

#[test]
fn overflow_reports_exact_unread_bytes_once() {
    let excess = 37usize;
    let output = format!("{}{}", "d".repeat(excess), "r".repeat(MAX_BG_BUFFER));
    let (registry, id) = registry_with_retained_output(output);

    let first = registry.poll(&id).unwrap();
    assert!(
        first.contains(&format!("{excess} unread bytes")),
        "first poll must name the exact unavailable span: {first:?}"
    );
    assert!(first.ends_with(&"r".repeat(MAX_BG_BUFFER)));

    let second = registry.poll(&id).unwrap();
    assert!(
        !second.contains("background output omitted"),
        "an acknowledged omission must not repeat: {second:?}"
    );
    assert!(second.contains("still running — no new output"));
}

#[test]
fn overflow_counts_only_unread_bytes() {
    let mut inner = BgInner::running("a".repeat(MAX_BG_BUFFER));
    inner.read_position = output_end(&inner);
    inner.output.push_str(&"b".repeat(73));
    trim_output_to_cap(&mut inner);

    let (omitted, fresh, end) = output_since(&inner, inner.read_position);
    assert_eq!(
        omitted, 0,
        "evicting already-delivered bytes is not data loss"
    );
    assert_eq!(fresh, "b".repeat(73));
    assert_eq!(end, MAX_BG_BUFFER as u64 + 73);
}

#[test]
fn overflow_preserves_utf8_and_reports_actual_boundary_cut() {
    // The nominal two-byte overflow lands inside the leading four-byte
    // scalar. The ring must discard the whole scalar and report four bytes.
    let output = format!("😀{}", "x".repeat(MAX_BG_BUFFER - 2));
    assert_eq!(output.len(), MAX_BG_BUFFER + 2);
    let (registry, id) = registry_with_retained_output(output);

    let first = registry.poll(&id).unwrap();
    assert!(first.contains("4 unread bytes"), "got: {first:?}");
    assert!(first.ends_with(&"x".repeat(MAX_BG_BUFFER - 2)));
}

#[tokio::test]
async fn streaming_poll_surfaces_overflow_to_callback_and_result() {
    let excess = 19usize;
    let output = format!("{}{}", "d".repeat(excess), "r".repeat(MAX_BG_BUFFER));
    let (registry, id) = registry_with_retained_output(output);
    let mut streamed = Vec::new();

    let result = registry
        .poll_wait_streaming(&id, Duration::ZERO, &mut |line| {
            streamed.push(line.to_string());
        })
        .await
        .unwrap();

    assert!(
        streamed
            .iter()
            .any(|line| line.contains(&format!("{excess} unread bytes"))),
        "live view must disclose overflow: {streamed:?}"
    );
    assert!(
        result.contains(&format!("{excess} unread bytes")),
        "model-facing poll result must disclose overflow: {result:?}"
    );
    assert!(
        !registry
            .poll(&id)
            .unwrap()
            .contains("background output omitted")
    );
}
