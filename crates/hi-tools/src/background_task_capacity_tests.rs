use super::*;

#[tokio::test]
async fn capacity_pruning_preserves_dependency_entries() {
    let registry = BackgroundTaskRegistry::new();
    let mut completed = Vec::with_capacity(MAX_BG_TASKS);
    for index in 0..MAX_BG_TASKS {
        completed.push(
            registry
                .spawn(
                    &format!("completed-{index}"),
                    "explore",
                    Box::new(|| {
                        Box::pin(async {
                            BackgroundTaskOutcome {
                                id: String::new(),
                                description: String::new(),
                                subagent_type: String::new(),
                                state: BackgroundTaskState::Completed,
                                output: "done".into(),
                                applied: false,
                                changed_files: vec![],
                            }
                        })
                    }),
                )
                .await
                .unwrap(),
        );
    }
    let _ = registry.wait_all(&completed, Duration::from_secs(2)).await;

    // The registry is at capacity, but the first completed task is still
    // a valid dependency. Pruning must not remove it between validation
    // and dependency-gate construction.
    let dependent = registry
        .spawn_after(
            "after-completed",
            "explore",
            std::slice::from_ref(&completed[0]),
            Box::new(|| {
                Box::pin(async {
                    BackgroundTaskOutcome {
                        id: String::new(),
                        description: String::new(),
                        subagent_type: String::new(),
                        state: BackgroundTaskState::Completed,
                        output: "after".into(),
                        applied: false,
                        changed_files: vec![],
                    }
                })
            }),
        )
        .await
        .unwrap();
    let outcome = registry
        .poll(&dependent, Duration::from_secs(1))
        .await
        .unwrap();
    assert_eq!(outcome.state, BackgroundTaskState::Completed);
}

#[tokio::test]
async fn queued_dependency_gate_survives_capacity_pruning() {
    let registry = BackgroundTaskRegistry::new();
    let release = Arc::new(Notify::new());
    let release_in_task = release.clone();
    let prerequisite = registry
        .spawn(
            "queued-prerequisite",
            "explore",
            Box::new(move || {
                Box::pin(async move {
                    release_in_task.notified().await;
                    BackgroundTaskOutcome {
                        id: String::new(),
                        description: String::new(),
                        subagent_type: String::new(),
                        state: BackgroundTaskState::Completed,
                        output: "ready".into(),
                        applied: false,
                        changed_files: Vec::new(),
                    }
                })
            }),
        )
        .await
        .unwrap();
    // Capture exactly the stable gate an already-queued dependent owns,
    // but intentionally do not poll it until after capacity pruning.
    let queued_gate = {
        let tasks = registry.tasks.lock().await;
        let entry = tasks.get(&prerequisite).unwrap();
        DependencyGate {
            id: prerequisite.clone(),
            terminal_outcome: entry.terminal_outcome.clone(),
            notify: entry.notify.clone(),
        }
    };

    let mut fillers = Vec::with_capacity(MAX_BG_TASKS - 1);
    for index in 0..(MAX_BG_TASKS - 1) {
        fillers.push(
            registry
                .spawn(
                    &format!("prune-filler-{index}"),
                    "explore",
                    Box::new(|| {
                        Box::pin(async {
                            BackgroundTaskOutcome {
                                id: String::new(),
                                description: String::new(),
                                subagent_type: String::new(),
                                state: BackgroundTaskState::Completed,
                                output: "done".into(),
                                applied: false,
                                changed_files: Vec::new(),
                            }
                        })
                    }),
                )
                .await
                .unwrap(),
        );
    }
    let filler_results = registry.wait_all(&fillers, Duration::from_secs(2)).await;
    assert!(
        filler_results
            .iter()
            .all(|outcome| outcome.state.is_terminal())
    );

    release.notify_one();
    assert_eq!(
        registry
            .poll(&prerequisite, Duration::from_secs(2))
            .await
            .unwrap()
            .state,
        BackgroundTaskState::Completed
    );

    let replacement = registry
        .spawn(
            "forces-capacity-prune",
            "explore",
            Box::new(|| {
                Box::pin(async {
                    BackgroundTaskOutcome {
                        id: String::new(),
                        description: String::new(),
                        subagent_type: String::new(),
                        state: BackgroundTaskState::Completed,
                        output: "replacement".into(),
                        applied: false,
                        changed_files: Vec::new(),
                    }
                })
            }),
        )
        .await
        .unwrap();
    assert!(
        !registry.tasks.lock().await.contains_key(&prerequisite),
        "capacity pruning should remove the registry entry in this regression"
    );

    let gate_result = tokio::time::timeout(
        Duration::from_millis(250),
        wait_for_dependencies(vec![queued_gate]),
    )
    .await
    .expect("a pruned prerequisite must not strand an existing dependent");
    assert_eq!(gate_result, Ok(()));
    assert_eq!(
        registry
            .poll(&replacement, Duration::from_secs(2))
            .await
            .unwrap()
            .state,
        BackgroundTaskState::Completed
    );
}

#[tokio::test]
async fn capacity_preserves_unobserved_completed_tasks_for_retrieval() {
    let registry = BackgroundTaskRegistry::new();
    let mut ids = Vec::with_capacity(MAX_BG_TASKS);
    for index in 0..MAX_BG_TASKS {
        ids.push(
            registry
                .spawn(
                    &format!("unpolled-{index}"),
                    "explore",
                    Box::new(move || {
                        Box::pin(async move {
                            BackgroundTaskOutcome {
                                id: String::new(),
                                description: String::new(),
                                subagent_type: String::new(),
                                state: BackgroundTaskState::Completed,
                                output: format!("done-{index}"),
                                applied: false,
                                changed_files: vec![],
                            }
                        })
                    }),
                )
                .await
                .unwrap(),
        );
    }
    tokio::time::timeout(Duration::from_secs(2), async {
        loop {
            let terminal = registry
                .outcomes
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .values()
                .filter(|outcome| outcome.state.is_terminal())
                .count();
            if terminal == MAX_BG_TASKS {
                break;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("all tasks should publish terminal results");
    let intermediate = registry.poll_many_inner(&ids, Duration::ZERO, false).await;
    assert!(
        intermediate
            .iter()
            .all(|outcome| outcome.state == BackgroundTaskState::Completed)
    );
    assert!(
        registry
            .tasks
            .lock()
            .await
            .values()
            .all(|entry| !entry.observed),
        "an internal wait snapshot must not acknowledge results before return"
    );

    let error = registry
        .spawn(
            "after-unpolled",
            "explore",
            Box::new(|| {
                Box::pin(async {
                    BackgroundTaskOutcome {
                        id: String::new(),
                        description: String::new(),
                        subagent_type: String::new(),
                        state: BackgroundTaskState::Completed,
                        output: "ran".into(),
                        applied: false,
                        changed_files: vec![],
                    }
                })
            }),
        )
        .await
        .expect_err("unobserved terminal results must retain their slots");
    let capacity = error
        .downcast_ref::<BackgroundTaskCapacityError>()
        .expect("capacity admission should return its typed error");
    assert_eq!(capacity.maximum, MAX_BG_TASKS);
    assert_eq!(capacity.running, 0);
    assert_eq!(capacity.unobserved_terminal, MAX_BG_TASKS);
    assert!(error.to_string().contains("get_task_output or wait_tasks"));

    for (index, id) in ids.iter().enumerate() {
        let outcome = registry.poll(id, Duration::ZERO).await.unwrap();
        assert_eq!(outcome.state, BackgroundTaskState::Completed);
        assert_eq!(outcome.output, format!("done-{index}"));
    }
}

#[tokio::test]
async fn acknowledged_completed_tasks_are_pruned_to_admit_later_work() {
    let registry = BackgroundTaskRegistry::new();
    let mut ids = Vec::with_capacity(MAX_BG_TASKS);
    for index in 0..MAX_BG_TASKS {
        ids.push(
            registry
                .spawn(
                    &format!("observed-{index}"),
                    "explore",
                    Box::new(|| {
                        Box::pin(async {
                            BackgroundTaskOutcome {
                                id: String::new(),
                                description: String::new(),
                                subagent_type: String::new(),
                                state: BackgroundTaskState::Completed,
                                output: "observed".into(),
                                applied: false,
                                changed_files: vec![],
                            }
                        })
                    }),
                )
                .await
                .unwrap(),
        );
    }
    let observed = registry.wait_all(&ids, Duration::from_secs(2)).await;
    assert!(observed.iter().all(|outcome| outcome.state.is_terminal()));

    let next = registry
        .spawn(
            "seventeenth-cumulative-task",
            "explore",
            Box::new(|| {
                Box::pin(async {
                    BackgroundTaskOutcome {
                        id: String::new(),
                        description: String::new(),
                        subagent_type: String::new(),
                        state: BackgroundTaskState::Completed,
                        output: "later".into(),
                        applied: false,
                        changed_files: vec![],
                    }
                })
            }),
        )
        .await
        .expect("acknowledged terminal entries should be reclaimable");
    let outcome = registry.poll(&next, Duration::from_secs(1)).await.unwrap();
    assert_eq!(outcome.state, BackgroundTaskState::Completed);
    assert_eq!(outcome.output, "later");
}

#[tokio::test]
async fn capacity_pruning_reclaims_observed_panicked_futures() {
    let registry = BackgroundTaskRegistry::new();
    let mut ids = Vec::with_capacity(MAX_BG_TASKS);
    for index in 0..MAX_BG_TASKS {
        ids.push(
            registry
                .spawn(
                    &format!("panicked-{index}"),
                    "explore",
                    Box::new(|| Box::pin(async { panic!("future boom") })),
                )
                .await
                .unwrap(),
        );
    }
    // Wait for the workers to publish every caught panic without polling
    // any registry entry. A fixed sleep made this regression sensitive to
    // slow or heavily loaded CI hosts.
    tokio::time::timeout(Duration::from_secs(2), async {
        loop {
            let panicked = registry
                .outcomes
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .values()
                .filter(|outcome| {
                    outcome.state == BackgroundTaskState::Failed
                        && outcome.output.contains("future boom")
                })
                .count();
            if panicked == MAX_BG_TASKS {
                break;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("all panicked futures should publish terminal failures");
    let observed = registry.poll_many(&ids, Duration::ZERO).await;
    assert!(observed.iter().all(|outcome| {
        outcome.state == BackgroundTaskState::Failed && outcome.output.contains("future boom")
    }));

    let next = registry
        .spawn(
            "after-panics",
            "explore",
            Box::new(|| {
                Box::pin(async {
                    BackgroundTaskOutcome {
                        id: String::new(),
                        description: String::new(),
                        subagent_type: String::new(),
                        state: BackgroundTaskState::Completed,
                        output: "ran".into(),
                        applied: false,
                        changed_files: Vec::new(),
                    }
                })
            }),
        )
        .await
        .expect("observed panics should be reclaimable terminal tasks");
    assert_eq!(
        registry
            .poll(&next, Duration::from_secs(1))
            .await
            .unwrap()
            .state,
        BackgroundTaskState::Completed
    );
}
