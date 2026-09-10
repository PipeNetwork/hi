use std::sync::Arc;

use super::common::{IsolatedWorkspace, NullUi, agent, bash_completion, write_completion};
use super::*;

struct CancelAfterResult {
    cancellation: TurnCancellation,
}

impl Ui for CancelAfterResult {
    fn assistant_text(&mut self, _: &str) {}
    fn assistant_reasoning(&mut self, _: &str) {}
    fn assistant_end(&mut self) {}
    fn tool_call(&mut self, _: &str, _: &str) {}
    fn tool_result(&mut self, _: &str, _: &str) {
        self.cancellation.cancel();
    }
    fn status(&mut self, _: &str) {}
    fn turn_end(&mut self, _: &str) {}
}

struct GatedDurability {
    entered: Arc<std::sync::atomic::AtomicUsize>,
    completed: Arc<std::sync::atomic::AtomicUsize>,
}

#[async_trait::async_trait]
impl WorkspaceDurability for GatedDurability {
    async fn mutation_started(&self, _: Option<Vec<String>>) -> anyhow::Result<()> {
        Ok(())
    }

    async fn checkpoint(&self) -> anyhow::Result<()> {
        self.entered
            .fetch_add(1, std::sync::atomic::Ordering::AcqRel);
        tokio::time::sleep(std::time::Duration::from_millis(75)).await;
        self.completed
            .fetch_add(1, std::sync::atomic::Ordering::Release);
        Ok(())
    }
}

#[cfg(unix)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cancelling_foreground_shell_reaps_before_workspace_settlement() {
    let workspace = IsolatedWorkspace::new("cancel-foreground-reap-settlement");
    let mut cfg = workspace.config();
    cfg.gates.verification = VerificationMode::Disabled;
    cfg.memory.finalize = false;
    cfg.memory.suggest_next_prompt = false;
    let mut agent = agent(
        vec![bash_completion(
            "printf started > foreground-started; sleep 600",
        )],
        cfg,
    );
    let foreground = agent.foreground_process_registry();
    let cancellation = TurnCancellation::new();
    let cancel_when_running = {
        let foreground = foreground.clone();
        let cancellation = cancellation.clone();
        tokio::spawn(async move {
            tokio::time::timeout(std::time::Duration::from_secs(3), async {
                while foreground.active_count() == 0 {
                    tokio::task::yield_now().await;
                }
            })
            .await
            .expect("foreground shell never registered");
            cancellation.cancel();
        })
    };

    let outcome = tokio::time::timeout(
        std::time::Duration::from_secs(10),
        agent.run_turn_cancellable("run a long command", &mut NullUi, cancellation),
    )
    .await
    .expect("foreground cancellation exceeded its reap/settlement bound")
    .expect("reaped foreground cancellation should produce a typed outcome");
    cancel_when_running.await.unwrap();

    assert_eq!(outcome.status, TurnStatus::Cancelled);
    assert_eq!(
        foreground.active_count(),
        0,
        "foreground child was not reaped"
    );
    assert!(
        !workspace.path("foreground-started").exists(),
        "rollback raced or skipped the cancelled foreground writer"
    );
    let controller = agent.workspace_controller_status();
    assert_eq!(controller.state, hi_workspace::WorkspaceState::Ready);
    assert!(controller.active_operation.is_none());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cancelled_outcome_waits_for_durability_acknowledgement() {
    let workspace = IsolatedWorkspace::new("cancel-await-durability");
    let mut cfg = workspace.config();
    cfg.gates.verification = VerificationMode::Disabled;
    cfg.memory.finalize = false;
    cfg.memory.suggest_next_prompt = false;
    let mut agent = agent(vec![write_completion("cancelled.txt")], cfg);
    let entered = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let completed = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    agent.set_workspace_durability(Some(Arc::new(GatedDurability {
        entered: entered.clone(),
        completed: completed.clone(),
    })));
    let cancellation = TurnCancellation::new();
    let mut ui = CancelAfterResult {
        cancellation: cancellation.clone(),
    };
    let outcome = agent
        .run_turn_cancellable("write then cancel", &mut ui, cancellation)
        .await
        .expect("acknowledged cancellation should settle");

    assert_eq!(outcome.status, TurnStatus::Cancelled);
    let entered = entered.load(std::sync::atomic::Ordering::Acquire);
    assert!(entered > 0, "cancellation never attempted settlement");
    assert_eq!(
        completed.load(std::sync::atomic::Ordering::Acquire),
        entered,
        "Cancelled was returned with durability acknowledgement still pending"
    );
    assert_eq!(
        agent.workspace_controller_status().state,
        hi_workspace::WorkspaceState::Ready
    );
}

struct HeldDurability {
    entered: tokio::sync::Notify,
    release: tokio::sync::Notify,
}

#[async_trait::async_trait]
impl WorkspaceDurability for HeldDurability {
    async fn mutation_started(&self, _: Option<Vec<String>>) -> anyhow::Result<()> {
        Ok(())
    }
    async fn checkpoint(&self) -> anyhow::Result<()> {
        self.entered.notify_one();
        self.release.notified().await;
        Ok(())
    }
}

#[tokio::test]
async fn selected_settlement_deadline_retains_the_late_durability_owner() {
    let workspace = IsolatedWorkspace::new("selected-settlement-deadline");
    let mut cfg = workspace.config();
    cfg.routing.tool_mode = ToolMode::ChatOnly;
    cfg.gates.verification = VerificationMode::Disabled;
    cfg.memory.suggest_next_prompt = false;
    cfg.loop_limits.turn_timeout = None;
    let mut agent = agent(
        vec![super::common::completion(
            vec![Content::Text("answer".into())],
            1,
            1,
        )],
        cfg,
    );
    let durability = Arc::new(HeldDurability {
        entered: tokio::sync::Notify::new(),
        release: tokio::sync::Notify::new(),
    });
    agent.set_workspace_durability(Some(durability.clone()));
    let cancellation = TurnCancellation::new();
    let mut ui = NullUi;
    let mut turn =
        Box::pin(agent.run_turn_cancellable("answer briefly", &mut ui, cancellation.clone()));
    tokio::select! {
        result = turn.as_mut() => panic!("turn returned before final durability: {result:?}"),
        _ = durability.entered.notified() => {},
    }
    assert!(!cancellation.is_cancelled());
    // Pause only after real setup/journal IO completed and the owned backend
    // operation entered its explicit gate. Auto-advance cannot race SQLite.
    tokio::time::pause();
    tokio::time::advance(std::time::Duration::from_secs(61)).await;
    let error = tokio::time::timeout(std::time::Duration::from_secs(1), turn.as_mut())
        .await
        .expect("selected settlement must have one finite wait")
        .unwrap_err();
    tokio::time::resume();
    drop(turn);
    let failure = crate::TurnFailure::from_error(&error).unwrap();
    assert!(failure.settlement_pending, "{failure:#}");
    assert!(error.to_string().contains("settlement"), "{error:#}");
    assert_ne!(
        agent.workspace_controller_status().state,
        hi_workspace::WorkspaceState::Ready
    );
    durability.release.notify_one();
    tokio::time::timeout(std::time::Duration::from_secs(2), async {
        while agent.workspace_controller_status().state != hi_workspace::WorkspaceState::Ready {
            tokio::time::sleep(std::time::Duration::from_millis(1)).await;
        }
    })
    .await
    .expect("late durability acknowledgement must retain its owner");
}

struct HeldTerminalSession {
    entered: Arc<tokio::sync::Notify>,
    release: Arc<(std::sync::Mutex<bool>, std::sync::Condvar)>,
    outcomes: Arc<std::sync::Mutex<Vec<TurnStatus>>>,
}

struct TerminalEventUi(Arc<std::sync::Mutex<Vec<hi_events::EventKind>>>);
impl Ui for TerminalEventUi {
    fn assistant_text(&mut self, _: &str) {}
    fn assistant_reasoning(&mut self, _: &str) {}
    fn assistant_end(&mut self) {}
    fn tool_call(&mut self, _: &str, _: &str) {}
    fn tool_result(&mut self, _: &str, _: &str) {}
    fn status(&mut self, _: &str) {}
    fn turn_end(&mut self, _: &str) {}
    fn semantic_event(&mut self, event: hi_events::RunEvent) {
        self.0.lock().unwrap().push(event.kind);
    }
}

impl SessionSink for HeldTerminalSession {
    fn record(&mut self, _: &[Message], _: Usage) -> anyhow::Result<()> {
        Ok(())
    }
    fn record_compaction(&mut self, _: &[Message]) -> anyhow::Result<()> {
        Ok(())
    }
    fn record_turn_outcome(
        &mut self,
        outcome: &TurnOutcome,
        _: Option<&str>,
    ) -> anyhow::Result<()> {
        self.entered.notify_one();
        let (lock, ready) = &*self.release;
        let mut released = lock.lock().unwrap();
        while !*released {
            released = ready.wait(released).unwrap();
        }
        self.outcomes.lock().unwrap().push(outcome.status);
        Ok(())
    }
}

#[tokio::test]
async fn cancelled_terminal_wait_retains_one_receipt_without_runtime_blocking() {
    let workspace = IsolatedWorkspace::new("cancel-terminal-session-owner");
    let mut cfg = workspace.config();
    cfg.gates.verification = VerificationMode::Disabled;
    cfg.memory.finalize = false;
    cfg.memory.suggest_next_prompt = false;
    let mut agent = agent(
        vec![super::common::completion(
            vec![Content::Text("answer".into())],
            1,
            1,
        )],
        cfg,
    );
    let entered = Arc::new(tokio::sync::Notify::new());
    let release = Arc::new((std::sync::Mutex::new(false), std::sync::Condvar::new()));
    let outcomes = Arc::new(std::sync::Mutex::new(Vec::new()));
    agent.set_session(Box::new(HeldTerminalSession {
        entered: entered.clone(),
        release: release.clone(),
        outcomes: outcomes.clone(),
    }));
    let cancellation = TurnCancellation::new();
    let events = Arc::new(std::sync::Mutex::new(Vec::new()));
    let mut ui = TerminalEventUi(events.clone());
    let mut turn =
        Box::pin(agent.run_turn_cancellable("answer briefly", &mut ui, cancellation.clone()));
    tokio::select! {
        result = turn.as_mut() => panic!("terminal receipt did not reach owner: {result:?}"),
        _ = entered.notified() => {},
    }
    assert!(
        !events
            .lock()
            .unwrap()
            .contains(&hi_events::EventKind::RunCompleted)
    );
    cancellation.cancel();
    tokio::time::pause();
    tokio::time::advance(std::time::Duration::from_secs(61)).await;
    let error = tokio::time::timeout(std::time::Duration::from_secs(1), turn.as_mut())
        .await
        .expect("blocked session IO must leave the runtime responsive")
        .unwrap_err();
    tokio::time::resume();
    drop(turn);
    let failure = TurnFailure::from_error(&error).unwrap();
    assert!(failure.settlement_pending);
    assert_eq!(failure.outcome.status, TurnStatus::Failed);
    assert_eq!(
        failure.outcome.stop_reason,
        TurnStopReason::InfrastructureFailure
    );
    assert!(outcomes.lock().unwrap().is_empty());
    let rewind = agent
        .rewind_to_snapshot_durable(1, &agent.state_snapshot())
        .unwrap_err();
    assert!(
        rewind
            .to_string()
            .contains("session persistence remains indeterminate")
    );
    {
        let recorded_events = events.lock().unwrap();
        assert!(!recorded_events.contains(&hi_events::EventKind::RunCompleted));
        assert_eq!(
            recorded_events
                .iter()
                .filter(|kind| **kind == hi_events::EventKind::RunFailed)
                .count(),
            1
        );
    }
    *release.0.lock().unwrap() = true;
    release.1.notify_all();
    agent.session_barrier().await.unwrap();
    assert_eq!(
        *outcomes.lock().unwrap(),
        vec![TurnStatus::Completed],
        "cancellation during accepted publication must not append a contradictory outcome"
    );
    let retry = agent.run_turn("retry", &mut NullUi).await.unwrap_err();
    assert!(TurnFailure::from_error(&retry).unwrap().settlement_pending);
    assert!(format!("{retry:#}").contains("reopening the session"));
    assert_eq!(outcomes.lock().unwrap().len(), 1);
}

#[cfg(unix)]
#[tokio::test]
async fn cancelled_no_save_goal_export_is_owned_through_later_session_attachment() {
    use std::os::unix::ffi::OsStrExt;
    let workspace = IsolatedWorkspace::new("cancel-no-save-goal-export");
    let mut cfg = workspace.config();
    cfg.subagents.long_horizon = true;
    let mut agent = agent(Vec::new(), cfg);
    assert!(
        !agent.has_session_io(),
        "ordinary unsaved agents must not start an IO owner"
    );
    agent
        .set_structured_goal(Some(Goal::new(
            "owned goal export",
            vec!["one step".into()],
        )))
        .unwrap();
    let metadata = workspace.path(".hi");
    std::fs::create_dir_all(&metadata).unwrap();
    let fifo = metadata.join("goal-plan.md");
    let path = std::ffi::CString::new(fifo.as_os_str().as_bytes()).unwrap();
    // A FIFO gives the export a real blocking filesystem operation, released
    // only by this test's reader. No arbitrary storage delays are injected.
    // SAFETY: path is NUL-terminated and points into this test's temporary directory.
    assert_eq!(unsafe { libc::mkfifo(path.as_ptr(), 0o600) }, 0);
    let mut ui = NullUi;
    let mut export = Box::pin(agent.persist_goal_async(&mut ui));
    tokio::select! {
        _ = export.as_mut() => panic!("FIFO export unexpectedly completed without a reader"),
        _ = tokio::time::sleep(std::time::Duration::from_millis(20)) => {},
    }
    drop(export);
    assert!(agent.has_session_io());
    // Attaching a real saved session must not forget the prior metadata owner.
    agent.set_session(Box::new(super::common::RecordingSession {
        records: Arc::new(std::sync::Mutex::new(Vec::new())),
    }));
    assert!(
        tokio::time::timeout(
            std::time::Duration::from_millis(20),
            agent.session_barrier()
        )
        .await
        .is_err()
    );
    let reader = tokio::task::spawn_blocking(move || std::fs::read_to_string(fifo).unwrap());
    tokio::time::timeout(std::time::Duration::from_secs(2), agent.session_barrier())
        .await
        .expect("both IO owners must drain after the FIFO is released")
        .unwrap();
    assert!(reader.await.unwrap().contains("owned goal export"));
}

#[tokio::test(flavor = "current_thread")]
async fn blocked_sqlite_cancellation_signals_all_jobs_and_retains_exact_pending_receipts() {
    use hi_tools::{BackgroundTaskOutcome, BackgroundTaskState};
    use std::time::{Duration, Instant};
    struct NotifyDrop(Option<tokio::sync::oneshot::Sender<()>>);
    impl Drop for NotifyDrop {
        fn drop(&mut self) {
            if let Some(sender) = self.0.take() {
                let _ = sender.send(());
            }
        }
    }
    let workspace = IsolatedWorkspace::new("cancel-three-jobs-blocked-sqlite");
    let agent = agent(Vec::new(), workspace.config());
    let registry = agent.background_task_registry();
    let store = hi_control::ControlStore::open_for_state(agent.runtime.state_root()).unwrap();
    let mut ids = Vec::new();
    let mut job_ids = Vec::new();
    let mut stopped = Vec::new();
    for index in 0..3 {
        let (started, running) = tokio::sync::oneshot::channel();
        let (signal, dropped) = tokio::sync::oneshot::channel();
        let id = registry
            .spawn(
                &format!("sqlite cancel {index}"),
                "explore",
                Box::new(move || {
                    Box::pin(async move {
                        let _execution = NotifyDrop(Some(signal));
                        let _ = started.send(());
                        std::future::pending::<BackgroundTaskOutcome>().await
                    })
                }),
            )
            .await
            .unwrap();
        running.await.unwrap();
        job_ids.push(hi_workspace::JobId::new(
            registry.candidate_workspace_job_id(&id).await.unwrap(),
        ));
        ids.push(id);
        stopped.push(dropped);
    }
    let blocker = rusqlite::Connection::open(store.path()).unwrap();
    blocker.execute_batch("BEGIN IMMEDIATE").unwrap();
    let owner = registry.clone();
    let started = Instant::now();
    let drain = tokio::spawn(async move {
        owner
            .kill_all_before(tokio::time::Instant::now() + Duration::from_millis(100))
            .await
    });
    tokio::time::timeout(Duration::from_millis(750), async {
        for execution in stopped {
            execution.await.unwrap();
        }
    })
    .await
    .expect("all three eligible executions must stop while SQLite publication is locked");
    let cancellation_latency = started.elapsed();
    assert!(
        cancellation_latency < Duration::from_millis(750),
        "cancellations must not wait on the blocked SQLite lock: {cancellation_latency:?}"
    );
    eprintln!(
        "blocked SQLite: all 3 execution cancellations acknowledged in {:.3}ms",
        cancellation_latency.as_secs_f64() * 1000.0
    );
    let pending = drain.await.unwrap();
    assert_eq!(pending.len(), 3);
    assert!(
        pending
            .iter()
            .all(|outcome| outcome.state == BackgroundTaskState::Running)
    );
    let mut pending_jobs = agent.workspace_controller_status().active_jobs;
    pending_jobs.sort_by_key(ToString::to_string);
    job_ids.sort_by_key(ToString::to_string);
    assert_eq!(
        pending_jobs, job_ids,
        "all accepted publications retain their exact job identity"
    );
    for job in &job_ids {
        assert_eq!(
            store.get_job(job.as_str()).unwrap().unwrap().state,
            hi_control::ControlJobState::Running
        );
    }
    blocker.execute_batch("ROLLBACK").unwrap();
    let outcomes = registry.wait_all(&ids, Duration::from_secs(2)).await;
    assert!(
        outcomes
            .iter()
            .all(|outcome| outcome.state == BackgroundTaskState::Cancelled)
    );
    for job in &job_ids {
        assert_eq!(
            store.get_job(job.as_str()).unwrap().unwrap().state,
            hi_control::ControlJobState::Cancelled
        );
    }
    let status = agent.workspace_controller_status();
    assert!(status.active_jobs.is_empty());
    assert_eq!(status.state, hi_workspace::WorkspaceState::Ready);
}

/// A native session callback can wait on a file lock indefinitely. Its owner
/// must not postpone signalling running jobs while plan-pause writes queue.
#[tokio::test(flavor = "current_thread")]
async fn plan_cancellation_signals_jobs_before_blocked_session_persistence() {
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::time::{Duration, Instant};

    struct Release(Arc<(std::sync::Mutex<bool>, std::sync::Condvar)>);
    impl Drop for Release {
        fn drop(&mut self) {
            *self.0.0.lock().unwrap() = true;
            self.0.1.notify_all();
        }
    }
    struct HeldRecord {
        armed: Arc<AtomicBool>,
        entered: Arc<tokio::sync::Notify>,
        release: Arc<(std::sync::Mutex<bool>, std::sync::Condvar)>,
        committed: Arc<AtomicUsize>,
    }
    impl SessionSink for HeldRecord {
        fn record(&mut self, _: &[Message], _: Usage) -> anyhow::Result<()> {
            if self.armed.swap(false, Ordering::AcqRel) {
                self.entered.notify_one();
                let (lock, ready) = &*self.release;
                let mut released = lock.lock().unwrap();
                while !*released {
                    released = ready.wait(released).unwrap();
                }
                self.committed.fetch_add(1, Ordering::Release);
            }
            Ok(())
        }
        fn record_compaction(&mut self, _: &[Message]) -> anyhow::Result<()> {
            Ok(())
        }
    }
    struct NotifyDrop(Option<tokio::sync::oneshot::Sender<()>>);
    impl Drop for NotifyDrop {
        fn drop(&mut self) {
            if let Some(send) = self.0.take() {
                let _ = send.send(());
            }
        }
    }

    let workspace = IsolatedWorkspace::new("plan-cancel-blocked-session");
    let requests = Arc::new(Mutex::new(Vec::new()));
    let provider = super::common::ScriptedProvider {
        steps: Mutex::new(vec![super::common::ProviderStep::DelayedCompletion(
            Duration::from_secs(600),
            super::common::completion(vec![Content::Text("eventual answer".into())], 1, 1),
        )]),
        requests: requests.clone(),
        max_tokens: None,
    };
    let mut subject = Agent::new(Arc::new(provider), workspace.config()).unwrap();
    subject.restore_plan(vec![hi_tools::PlanStep {
        title: "finish the requested change".into(),
        status: hi_tools::PlanStatus::Pending,
    }]);
    let armed = Arc::new(AtomicBool::new(false));
    let entered = Arc::new(tokio::sync::Notify::new());
    let release = Release(Arc::new((
        std::sync::Mutex::new(false),
        std::sync::Condvar::new(),
    )));
    let committed = Arc::new(AtomicUsize::new(0));
    subject.set_session(Box::new(HeldRecord {
        armed: armed.clone(),
        entered: entered.clone(),
        release: release.0.clone(),
        committed: committed.clone(),
    }));
    let owner = subject.session.as_ref().unwrap().io_handle().unwrap();
    let registry = subject.background_task_registry();
    let cancellation = TurnCancellation::new();
    let mut ui = NullUi;
    let mut turn = Box::pin(subject.run_turn_cancellable(
        crate::PLAN_DRIVE_PROMPT,
        &mut ui,
        cancellation.clone(),
    ));
    let prepare = async {
        tokio::time::timeout(Duration::from_secs(3), async {
            while requests.lock().unwrap().is_empty() {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("synthetic plan turn did not reach the provider");
        let mut stopped = Vec::new();
        for index in 0..3 {
            let (started, running) = tokio::sync::oneshot::channel();
            let (signal, dropped) = tokio::sync::oneshot::channel();
            registry
                .spawn(
                    &format!("plan job {index}"),
                    "explore",
                    Box::new(move || {
                        Box::pin(async move {
                            let _execution = NotifyDrop(Some(signal));
                            let _ = started.send(());
                            std::future::pending::<hi_tools::BackgroundTaskOutcome>().await
                        })
                    }),
                )
                .await
                .unwrap();
            running.await.unwrap();
            stopped.push(dropped);
        }
        armed.store(true, Ordering::Release);
        let write =
            tokio::spawn(
                async move { owner.write(|sink| sink.record(&[], Usage::default())).await },
            );
        entered.notified().await;
        (stopped, write)
    };
    let (stopped, write) = tokio::select! {
        result = turn.as_mut() => panic!("turn completed before cancellation: {result:?}"),
        prepared = prepare => prepared,
    };
    let started = Instant::now();
    cancellation.cancel();
    tokio::select! {
        result = turn.as_mut() => panic!("turn returned while the accepted write was blocked: {result:?}"),
        stopped = tokio::time::timeout(Duration::from_secs(1), async {
            for execution in stopped {
                execution.await.unwrap();
            }
        }) => stopped.expect("entry must signal all jobs before awaiting plan pause persistence"),
    }
    assert!(started.elapsed() < Duration::from_secs(1));
    assert!(!write.is_finished());
    assert_eq!(committed.load(Ordering::Acquire), 0);

    tokio::time::pause();
    tokio::time::advance(Duration::from_secs(61)).await;
    let error = tokio::time::timeout(Duration::from_secs(1), turn.as_mut())
        .await
        .expect("cancellation must retain one shared settlement deadline")
        .unwrap_err();
    tokio::time::resume();
    drop(turn);
    assert!(
        TurnFailure::from_error(&error).unwrap().settlement_pending,
        "{error:#}"
    );
    assert!(subject.ensure_session_reusable().is_err());
    assert_eq!(committed.load(Ordering::Acquire), 0);
    drop(release);
    write.await.unwrap().unwrap();
    subject.session_barrier().await.unwrap();
    assert_eq!(
        committed.load(Ordering::Acquire),
        1,
        "the accepted write must retain exactly one owner"
    );
}

#[tokio::test(flavor = "current_thread")]
async fn public_cleanup_reuses_the_deadline_and_fences_a_blocked_session_write() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    struct Session;
    impl SessionSink for Session {
        fn record(&mut self, _: &[Message], _: Usage) -> anyhow::Result<()> {
            Ok(())
        }
        fn record_compaction(&mut self, _: &[Message]) -> anyhow::Result<()> {
            Ok(())
        }
    }
    let workspace = IsolatedWorkspace::new("compat-cleanup-session-deadline");
    let mut subject = agent(Vec::new(), workspace.config());
    subject.set_session(Box::new(Session));
    let owner = subject.session.as_ref().unwrap().io_handle().unwrap();
    let entered = Arc::new(tokio::sync::Notify::new());
    let committed = Arc::new(AtomicUsize::new(0));
    let (release, gate) = std::sync::mpsc::channel();
    let write = {
        let entered = entered.clone();
        let committed = committed.clone();
        tokio::spawn(async move {
            owner
                .write(move |_| {
                    entered.notify_one();
                    gate.recv()?;
                    committed.fetch_add(1, Ordering::Release);
                    Ok(())
                })
                .await
        })
    };
    entered.notified().await;
    let cancellation = TurnCancellation::new();
    cancellation.cancel();
    let deadline = cancellation.settlement_deadline();
    subject.turn_cancellation = Some(cancellation);
    tokio::time::pause();
    tokio::time::advance(Duration::from_secs(59)).await;
    let error = tokio::time::timeout(
        Duration::from_secs(2),
        subject.cleanup_turn(crate::TurnCleanupKind::Cancel {
            session: crate::SessionRollback::AlreadyApplied,
        }),
    )
    .await
    .expect("compatibility cleanup must reuse the existing deadline, not restart 60s")
    .unwrap_err();
    assert!(tokio::time::Instant::now() <= deadline + Duration::from_millis(10));
    tokio::time::resume();
    let failure = TurnFailure::from_error(&error).expect("timeout needs a typed pending receipt");
    assert!(failure.settlement_pending);
    assert!(format!("{error:#}").contains("cleanup exceeded"));
    assert!(subject.ensure_session_reusable().is_err());
    assert_eq!(committed.load(Ordering::Acquire), 0);
    release.send(()).unwrap();
    write.await.unwrap().unwrap();
    subject.session_barrier().await.unwrap();
    assert_eq!(committed.load(Ordering::Acquire), 1);
}
