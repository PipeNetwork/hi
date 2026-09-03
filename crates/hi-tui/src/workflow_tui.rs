//! `/workflow` handling for the TUI.

use crate::theme::UiTone;
use crate::{App, dim, theme};
use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};
use ratatui::text::Line;
use ratatui::text::Span;
use ratatui::widgets::{Paragraph, Wrap};
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum WorkflowOverlayView {
    List,
    Detail,
}

pub(crate) struct WorkflowOverlay {
    pub(crate) runs: Vec<hi_workflow::WorkflowRunSnapshot>,
    pub(crate) selected: usize,
    pub(crate) view: WorkflowOverlayView,
}

impl WorkflowOverlay {
    fn new(mut runs: Vec<hi_workflow::WorkflowRunSnapshot>) -> Self {
        runs.sort_by(|a, b| {
            b.elapsed_ms
                .cmp(&a.elapsed_ms)
                .then_with(|| a.run_id.cmp(&b.run_id))
        });
        Self {
            runs,
            selected: 0,
            view: WorkflowOverlayView::List,
        }
    }

    pub(crate) fn selected(&self) -> Option<&hi_workflow::WorkflowRunSnapshot> {
        self.runs.get(self.selected)
    }
}

pub(crate) enum WorkflowOverlayOutcome {
    Continue,
    Close,
    Command(String),
}

pub(crate) fn handle_overlay_key(
    app: &mut App,
    key: &crossterm::event::KeyEvent,
) -> WorkflowOverlayOutcome {
    use crossterm::event::KeyCode;
    let Some(overlay) = app.workflow_overlay.as_mut() else {
        return WorkflowOverlayOutcome::Continue;
    };
    match key.code {
        KeyCode::Esc | KeyCode::Char('q') if overlay.view == WorkflowOverlayView::List => {
            WorkflowOverlayOutcome::Close
        }
        KeyCode::Esc | KeyCode::Backspace | KeyCode::Char('h')
            if overlay.view == WorkflowOverlayView::Detail =>
        {
            overlay.view = WorkflowOverlayView::List;
            WorkflowOverlayOutcome::Continue
        }
        KeyCode::Up | KeyCode::Char('k') => {
            overlay.selected = overlay.selected.saturating_sub(1);
            WorkflowOverlayOutcome::Continue
        }
        KeyCode::Down | KeyCode::Char('j') => {
            overlay.selected = (overlay.selected + 1).min(overlay.runs.len().saturating_sub(1));
            WorkflowOverlayOutcome::Continue
        }
        KeyCode::Enter | KeyCode::Right | KeyCode::Char('l') => {
            overlay.view = WorkflowOverlayView::Detail;
            WorkflowOverlayOutcome::Continue
        }
        KeyCode::Char('s') => overlay
            .selected()
            .filter(|run| run.status == hi_workflow::WorkflowRunStatus::Active)
            .map(|run| WorkflowOverlayOutcome::Command(format!("/workflow stop {}", run.run_id)))
            .unwrap_or(WorkflowOverlayOutcome::Continue),
        KeyCode::Char('r') => overlay
            .selected()
            .filter(|run| run.status.is_resumable())
            .map(|run| WorkflowOverlayOutcome::Command(format!("/workflow resume {}", run.run_id)))
            .unwrap_or(WorkflowOverlayOutcome::Continue),
        KeyCode::Char('d') => overlay
            .selected()
            .filter(|run| run.status.is_terminal())
            .map(|run| WorkflowOverlayOutcome::Command(format!("/workflow delete {}", run.run_id)))
            .unwrap_or(WorkflowOverlayOutcome::Continue),
        _ => WorkflowOverlayOutcome::Continue,
    }
}

fn stored_snapshot(run: hi_workflow::StoredWorkflowRun) -> hi_workflow::WorkflowRunSnapshot {
    let manifest = run.manifest;
    let elapsed_ms = manifest
        .updated_at_ms
        .saturating_sub(manifest.created_at_ms);
    let status = manifest.status();
    let (pause_message, result_summary) = match manifest.outcome {
        Some(hi_workflow::WorkflowOutcome::Paused { message, .. }) => (Some(message), None),
        Some(hi_workflow::WorkflowOutcome::Completed { result }) => {
            (None, Some(result.to_string()))
        }
        Some(hi_workflow::WorkflowOutcome::BudgetExceeded { message }) => (Some(message), None),
        Some(hi_workflow::WorkflowOutcome::Failed { error }) => (None, Some(error)),
        _ => (None, None),
    };
    hi_workflow::WorkflowRunSnapshot {
        run_id: manifest.run_id,
        revision: 0,
        workflow_name: manifest.workflow_name,
        objective: run
            .args
            .get("input")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("")
            .to_owned(),
        status,
        phases: vec![],
        current_phase: manifest.current_phase,
        agents: vec![],
        agent_budget: manifest.agent_budget,
        agents_used: manifest.agent_spent,
        agents_reserved: 0,
        elapsed_ms,
        pause_message,
        result_summary,
        history: vec![],
    }
}

fn agent_budget_label(agent_budget: Option<u64>) -> String {
    agent_budget
        .map(|budget| budget.to_string())
        .unwrap_or_else(|| "unlimited".to_owned())
}

fn open_workflow_overlay(app: &mut App) {
    let mut runs = runtime_manager()
        .and_then(|manager| manager.list().map_err(anyhow::Error::from))
        .map(|runs| runs.into_iter().map(stored_snapshot).collect::<Vec<_>>())
        .unwrap_or_default();
    for run in app.workflow_runs.values() {
        if let Some(existing) = runs
            .iter_mut()
            .find(|snapshot| snapshot.run_id == run.snapshot.run_id)
        {
            *existing = run.snapshot.clone();
        } else {
            runs.push(run.snapshot.clone());
        }
    }
    app.workflow_overlay = Some(WorkflowOverlay::new(runs));
}

fn status_label(status: hi_workflow::WorkflowRunStatus) -> &'static str {
    use hi_workflow::WorkflowRunStatus::*;
    match status {
        Active => "running",
        UserPaused => "paused",
        BackOffPaused => "backoff",
        NoProgressPaused => "paused — no progress",
        InfraPaused => "infra paused",
        Blocked => "blocked",
        BudgetLimited => "budget",
        Interrupted => "interrupted",
        Complete => "complete",
        Failed => "failed",
        Cancelled => "cancelled",
    }
}

fn elapsed(ms: u64) -> String {
    let seconds = ms / 1000;
    if seconds >= 3600 {
        format!("{}h {:02}m", seconds / 3600, seconds % 3600 / 60)
    } else if seconds >= 60 {
        format!("{}m {:02}s", seconds / 60, seconds % 60)
    } else {
        format!("{seconds}s")
    }
}

pub(crate) fn overlay_lines(overlay: &WorkflowOverlay) -> Vec<Line<'static>> {
    let th = theme::theme();
    let mut lines = Vec::new();
    match overlay.view {
        WorkflowOverlayView::List => {
            lines.push(Line::styled("WORKFLOW RUNS", accent()));
            lines.push(Line::styled(
                "status       name                 phase              agents   elapsed",
                dim(),
            ));
            if overlay.runs.is_empty() {
                lines.push(Line::styled("No workflow runs yet.", dim()));
            }
            for (index, run) in overlay.runs.iter().enumerate() {
                let marker = if index == overlay.selected {
                    "›"
                } else {
                    " "
                };
                let agent_budget = agent_budget_label(run.agent_budget);
                let text = format!(
                    "{marker} {:<12} {:<20} {:<18} {:>2}/{:<9}   {:>7}",
                    status_label(run.status),
                    run.workflow_name,
                    run.current_phase.as_deref().unwrap_or("—"),
                    run.agents_used + run.agents_reserved,
                    agent_budget,
                    elapsed(run.elapsed_ms)
                );
                let style = if index == overlay.selected {
                    Style::default()
                        .fg(th.text_primary)
                        .add_modifier(Modifier::BOLD)
                } else {
                    dim()
                };
                lines.push(Line::styled(text, style));
            }
            lines.push(Line::styled("↑/↓ select · Enter detail · Esc close", dim()));
        }
        WorkflowOverlayView::Detail => {
            if let Some(run) = overlay.selected() {
                lines.push(Line::from(vec![
                    Span::styled(run.workflow_name.clone(), accent()),
                    Span::styled(format!("  {}", status_label(run.status)), dim()),
                ]));
                lines.push(Line::styled(
                    format!("run {} · {}", run.run_id, elapsed(run.elapsed_ms)),
                    dim(),
                ));
                lines.push(Line::styled("Objective", accent()));
                lines.push(Line::raw(if run.objective.is_empty() {
                    "—".into()
                } else {
                    run.objective.clone()
                }));
                lines.push(Line::styled("Phases", accent()));
                lines.push(Line::raw(
                    run.phases
                        .iter()
                        .map(|phase| {
                            format!(
                                "{} {}",
                                if phase.state == "done" {
                                    "✓"
                                } else if phase.state == "active" {
                                    "▸"
                                } else {
                                    "○"
                                },
                                phase.title
                            )
                        })
                        .collect::<Vec<_>>()
                        .join("  "),
                ));
                lines.push(Line::styled(
                    format!(
                        "Agents · {} used + {} reserved / {} budget",
                        run.agents_used,
                        run.agents_reserved,
                        agent_budget_label(run.agent_budget)
                    ),
                    accent(),
                ));
                for agent in &run.agents {
                    lines.push(Line::styled(
                        format!(
                            "  {} · {} · {} tokens · {}",
                            agent.label,
                            agent.state,
                            agent.tokens_used,
                            elapsed(agent.duration_ms)
                        ),
                        dim(),
                    ));
                }
                if let Some(message) = &run.pause_message {
                    lines.push(Line::styled(
                        format!("Pause · {message}"),
                        theme::theme().chrome(UiTone::Active).border,
                    ));
                }
                if let Some(result) = &run.result_summary {
                    lines.push(Line::styled(
                        format!("Result · {result}"),
                        theme::theme().chrome(UiTone::Success).border,
                    ));
                }
                lines.push(Line::styled("Recent history", accent()));
                for event in run.history.iter().rev().take(6).rev() {
                    lines.push(Line::styled(
                        format!(
                            "  {} · {}{}",
                            elapsed(event.at_ms),
                            event.event,
                            event
                                .detail
                                .as_ref()
                                .map(|d| format!(" — {d}"))
                                .unwrap_or_default()
                        ),
                        dim(),
                    ));
                }
                let mut actions = vec!["Esc back"];
                if run.status == hi_workflow::WorkflowRunStatus::Active {
                    actions.push("s stop");
                }
                if run.status.is_resumable() {
                    actions.push("r resume");
                }
                if run.status.is_terminal() {
                    actions.push("d delete");
                }
                lines.push(Line::styled(actions.join(" · "), dim()));
            }
        }
    }
    lines
}

pub(crate) fn render_overlay(frame: &mut ratatui::Frame, area: Rect, overlay: &WorkflowOverlay) {
    let block = theme::theme().panel_block(" Workflows ", UiTone::Assistant);
    frame.render_widget(
        Paragraph::new(overlay_lines(overlay))
            .block(block)
            .wrap(Wrap { trim: false }),
        area,
    );
}

fn run_store() -> Option<hi_workflow::WorkflowRunStore> {
    std::env::var_os("XDG_STATE_HOME")
        .map(std::path::PathBuf::from)
        .or_else(|| {
            std::env::var_os("HOME").map(|home| std::path::PathBuf::from(home).join(".local/state"))
        })
        .map(|base| hi_workflow::WorkflowRunStore::new(base.join("hi/workflow-runs")))
}

fn runtime_manager() -> anyhow::Result<hi_workflow::WorkflowRuntimeManager> {
    let store =
        run_store().ok_or_else(|| anyhow::anyhow!("workflow state directory is unavailable"))?;
    Ok(hi_workflow::WorkflowRuntimeManager::new(store))
}

fn run_id<'a>(app: &'a App, explicit: &'a str) -> Option<&'a str> {
    (!explicit.is_empty())
        .then_some(explicit)
        .or(app.selected_workflow_run.as_deref())
}

fn registry() -> anyhow::Result<hi_workflow::WorkflowRegistry> {
    let root = Path::new(".");
    Ok(hi_workflow::WorkflowRegistry::scan(
        Some(root),
        hi_agent::workspace_trusted(root),
    )?)
}

fn accent() -> Style {
    Style::default()
        .fg(theme::theme().accent_assistant)
        .add_modifier(Modifier::BOLD)
}

pub(crate) async fn start_workflow_run(app: &mut App, arg: &str) -> anyhow::Result<()> {
    let mut parts = arg.splitn(2, char::is_whitespace);
    let name = parts.next().unwrap_or("");
    let args_str = parts.next().unwrap_or("").trim();
    let registry = registry()?;
    let workflow = registry.resolve(name)?;
    let args = if args_str.is_empty() {
        serde_json::json!({})
    } else if args_str.starts_with('{') {
        serde_json::from_str(args_str)
            .map_err(|e| anyhow::anyhow!("invalid workflow JSON arguments: {e}"))?
    } else {
        serde_json::json!({"input": args_str})
    };
    app.push(Line::styled(
        format!("starting workflow '{name}'…"),
        accent(),
    ));
    app.follow();
    crate::dashboard::start_workflow_run(app, workflow.script.clone(), args).await
}

/// `/workflow plan …` — the local plan-objectives engine (`hi workflow run`),
/// spawned as a detached child so the session stays interactive. The child
/// checkpoints under the state root and survives this TUI exiting; `status`
/// tails its log and `stop` terminates it.
pub(crate) fn handle_plan_workflow(
    app: &mut App,
    rest: &str,
    exe: &Path,
    max_steps: Option<u32>,
    max_tool_calls: Option<u32>,
    max_verify_repairs: Option<u32>,
) {
    let error = |app: &mut App, text: String| {
        app.push(Line::styled(
            text,
            theme::theme().chrome(UiTone::Error).border,
        ));
        app.follow();
    };
    let rest = rest.trim();
    let mut parts = rest.split_whitespace();
    match parts.next() {
        None | Some("help") => {
            for line in [
                "/workflow plan — build a plan.md of objectives with the workflow engine",
                "  /workflow plan <plan.md> [--verify CMD] [--parallel N] [--max-steps N] [--max-tool-calls N] [--max-verify-repairs N] [--dry-run]",
                "  /workflow plan resume <plan.md>   continue the latest sealed checkpoint",
                "  /workflow plan status             child liveness + recent output",
                "  /workflow plan stop               terminate the running child",
            ] {
                app.push(Line::styled(
                    line,
                    if line.starts_with("/workflow plan —") {
                        accent()
                    } else {
                        dim()
                    },
                ));
            }
            app.follow();
        }
        Some("status") => {
            let Some((pid, log, plan)) = app.plan_workflow_child.clone() else {
                error(app, "no plan workflow was started in this session".into());
                return;
            };
            let alive = std::process::Command::new("kill")
                .args(["-0", &pid.to_string()])
                .status()
                .is_ok_and(|status| status.success());
            app.push(Line::styled(
                format!(
                    "{plan}: {} (pid {pid}) — log: {}",
                    if alive { "running" } else { "finished" },
                    log.display()
                ),
                accent(),
            ));
            let tail = std::fs::read_to_string(&log).unwrap_or_default();
            for line in tail
                .lines()
                .rev()
                .take(12)
                .collect::<Vec<_>>()
                .into_iter()
                .rev()
            {
                app.push(Line::styled(format!("  {line}"), dim()));
            }
            app.follow();
        }
        Some("stop") => {
            let Some((pid, _, plan)) = app.plan_workflow_child.clone() else {
                error(app, "no plan workflow was started in this session".into());
                return;
            };
            let stopped = std::process::Command::new("kill")
                .arg(pid.to_string())
                .status()
                .is_ok_and(|status| status.success());
            app.push(Line::styled(
                if stopped {
                    format!("{plan}: sent SIGTERM to pid {pid}; `/workflow plan resume` continues from the last sealed checkpoint")
                } else {
                    format!("{plan}: pid {pid} was not running")
                },
                accent(),
            ));
            app.follow();
        }
        Some(first) => {
            let resume = first == "resume";
            let mut arguments: Vec<String> = vec![
                "workflow".into(),
                if resume {
                    "resume".into()
                } else {
                    "run".into()
                },
            ];
            if resume {
                match parts.next() {
                    Some(plan) => arguments.push(plan.into()),
                    None => {
                        error(app, "usage: /workflow plan resume <plan.md>".into());
                        return;
                    }
                }
            } else {
                arguments.push(first.into());
            }
            arguments.extend(parts.map(str::to_owned));
            append_inherited_execution_caps(
                &mut arguments,
                max_steps,
                max_tool_calls,
                max_verify_repairs,
            );
            let plan_label = arguments[2].clone();
            if !resume && !Path::new(&plan_label).is_file() {
                error(app, format!("plan file not found: {plan_label}"));
                return;
            }
            let log = std::env::temp_dir().join(format!(
                "hi-workflow-plan-{}-{}.log",
                std::process::id(),
                plan_label.replace(['/', '.'], "_")
            ));
            let log_file = match std::fs::File::create(&log) {
                Ok(file) => file,
                Err(err) => {
                    error(
                        app,
                        format!("cannot create workflow log {}: {err}", log.display()),
                    );
                    return;
                }
            };
            let stderr_file = match log_file.try_clone() {
                Ok(file) => file,
                Err(err) => {
                    error(app, format!("cannot clone workflow log handle: {err}"));
                    return;
                }
            };
            match std::process::Command::new(exe)
                .args(&arguments)
                .stdin(std::process::Stdio::null())
                .stdout(std::process::Stdio::from(log_file))
                .stderr(std::process::Stdio::from(stderr_file))
                .spawn()
            {
                Ok(child) => {
                    let pid = child.id();
                    drop(child);
                    app.plan_workflow_child = Some((pid, log.clone(), plan_label.clone()));
                    let run_id = format!("plan-{pid}");
                    let snapshot = hi_workflow::WorkflowRunSnapshot {
                        run_id: run_id.clone(),
                        revision: 1,
                        workflow_name: format!("plan:{plan_label}"),
                        objective: plan_label.clone(),
                        status: hi_workflow::WorkflowRunStatus::Active,
                        phases: vec![hi_workflow::WorkflowPhaseSnapshot {
                            title: "Execute plan".into(),
                            state: "active".into(),
                        }],
                        current_phase: Some("Execute plan".into()),
                        agents: vec![],
                        agent_budget: None,
                        agents_used: 0,
                        agents_reserved: 0,
                        elapsed_ms: 0,
                        pause_message: None,
                        result_summary: Some(format!("local-signed · pid {pid}")),
                        history: vec![],
                    };
                    app.apply(crate::event::UiEvent::WorkflowUpdated { snapshot });
                    app.push(Line::styled(
                        format!("▶ workflow {plan_label} started (pid {pid})"),
                        accent(),
                    ));
                    app.push(Line::styled(
                        format!("  log: {} — `/workflow plan status` for progress; it checkpoints every wave and survives this session", log.display()),
                        dim(),
                    ));
                    app.follow();
                }
                Err(err) => error(app, format!("failed to start workflow child: {err}")),
            }
        }
    }
}

fn append_inherited_execution_caps(
    arguments: &mut Vec<String>,
    max_steps: Option<u32>,
    max_tool_calls: Option<u32>,
    max_verify_repairs: Option<u32>,
) {
    let has_steps = arguments.iter().any(|argument| argument == "--max-steps");
    let has_tools = arguments
        .iter()
        .any(|argument| argument == "--max-tool-calls");
    let has_verify_repairs = arguments
        .iter()
        .any(|argument| argument == "--max-verify-repairs");
    for argument in crate::child_execution_cap_args(
        if has_steps { None } else { max_steps },
        if has_tools { None } else { max_tool_calls },
    ) {
        arguments.push(argument.to_string_lossy().into_owned());
    }
    if !has_verify_repairs && let Some(max_verify_repairs) = max_verify_repairs {
        arguments.push("--max-verify-repairs".into());
        arguments.push(max_verify_repairs.to_string());
    }
}

pub(crate) fn handle_workflow_tui(app: &mut App, arg: &str) {
    let arg = arg.trim();
    if arg.is_empty() {
        open_workflow_overlay(app);
        return;
    }
    let mut parts = arg.splitn(2, char::is_whitespace);
    let sub = parts.next().unwrap_or("");
    let rest = parts.next().unwrap_or("").trim();
    match sub {
        "list" | "ls" => match registry() {
            Ok(reg) => {
                for w in reg.list() {
                    app.push(Line::styled(
                        format!("  {:<20} {}", w.name, w.meta.description),
                        dim(),
                    ));
                }
            }
            Err(e) => app.push(Line::styled(
                format!("workflow registry error: {e}"),
                theme::theme().chrome(UiTone::Error).border,
            )),
        },
        "show" => {
            match registry().and_then(|r| r.resolve(rest).cloned().map_err(anyhow::Error::from)) {
                Ok(w) => {
                    app.push(Line::styled(w.meta.name, accent()));
                    app.push(Line::styled(format!("  {}", w.meta.description), dim()));
                    if let Some(usage) = w.meta.when_to_use {
                        app.push(Line::styled(format!("  when: {usage}"), dim()));
                    }
                    for phase in w.meta.phases {
                        app.push(Line::styled(format!("  ○ {}", phase.title), dim()));
                    }
                }
                Err(e) => app.push(Line::styled(
                    e.to_string(),
                    theme::theme().chrome(UiTone::Error).border,
                )),
            }
        }
        "validate" => match std::fs::read_to_string(rest) {
            Ok(source) => match hi_workflow::DeclarativeWorkflow::from_json(&source)
                .map_err(|e| e.to_string())
                .and_then(|workflow| {
                    workflow
                        .validate()
                        .map(|_| workflow)
                        .map_err(|e| e.to_string())
                }) {
                Ok(workflow) => app.push(Line::styled(
                    format!(
                        "VALID — name={}, steps={}",
                        workflow.metadata.name,
                        workflow.steps.len()
                    ),
                    theme::theme().chrome(UiTone::Success).border,
                )),
                Err(e) => app.push(Line::styled(
                    format!("INVALID: {e}"),
                    theme::theme().chrome(UiTone::Error).border,
                )),
            },
            Err(e) => app.push(Line::styled(
                format!("cannot read {rest}: {e}"),
                theme::theme().chrome(UiTone::Error).border,
            )),
        },
        "runs" => match runtime_manager()
            .and_then(|manager| manager.list().map_err(anyhow::Error::from))
        {
            Ok(runs) if runs.is_empty() => {
                app.push(Line::styled("no persisted workflow runs", dim()))
            }
            Ok(runs) => {
                for run in runs {
                    let m = run.manifest;
                    app.push(Line::styled(
                        format!("  {:<28} {:<18} {:?}", m.run_id, m.workflow_name, m.status),
                        dim(),
                    ));
                }
            }
            Err(e) => app.push(Line::styled(
                format!("workflow run list error: {e}"),
                theme::theme().chrome(UiTone::Error).border,
            )),
        },
        "status" | "details" => match run_id(app, rest) {
            Some(id) => match run_store()
                .ok_or_else(|| anyhow::anyhow!("workflow state directory is unavailable"))
                .and_then(|store| store.load(id).map_err(anyhow::Error::from))
            {
                Ok(run) => {
                    let m = run.manifest;
                    app.push(Line::styled(
                        format!("{} — {}", m.workflow_name, m.run_id),
                        accent(),
                    ));
                    app.push(Line::styled(
                        format!(
                            "  status: {:?}  agents: {}/{}",
                            m.status,
                            m.agent_spent,
                            agent_budget_label(m.agent_budget)
                        ),
                        dim(),
                    ));
                    if let Some(phase) = m.current_phase {
                        app.push(Line::styled(format!("  phase: {phase}"), dim()));
                    }
                    if let Some(outcome) = m.outcome {
                        app.push(Line::styled(format!("  outcome: {outcome:?}"), dim()));
                    }
                }
                Err(e) => app.push(Line::styled(
                    format!("workflow status error: {e}"),
                    theme::theme().chrome(UiTone::Error).border,
                )),
            },
            None => app.push(Line::styled(
                "no workflow run selected; use /workflow runs",
                dim(),
            )),
        },
        "stop" => {
            let target = run_id(app, rest).map(str::to_string);
            if target
                .as_deref()
                .is_some_and(|id| crate::dashboard::cancel_workflow_run(app, id))
            {
                app.push(Line::styled("workflow cancellation requested", dim()));
            } else {
                app.push(Line::styled("no active workflow", dim()));
            }
        }
        "resume" => {
            let mut parts = rest.split_whitespace();
            let run_id = parts.next().unwrap_or("");
            let approval_id = parts.next();
            let operation_digest = parts.next();
            if run_id.is_empty() || parts.next().is_some() {
                app.push(Line::styled(
                    "usage: /workflow resume <run-id> [approval-id operation-digest]",
                    dim(),
                ));
            } else if app.workflow_runs.contains_key(run_id) {
                app.push(Line::styled(
                    "a workflow is already active; stop it before resuming another",
                    theme::theme().chrome(UiTone::Error).border,
                ));
            } else {
                let approval_store = app.approval_store.clone();
                match runtime_manager().and_then(|mut manager| {
                    match (approval_id, operation_digest) {
                        (Some(approval_id), Some(operation_digest)) => {
                            let store = approval_store.as_deref().ok_or_else(|| {
                                anyhow::anyhow!("approval store unavailable; resume is fail-closed")
                            })?;
                            manager.resume_with_approval(
                                run_id,
                                store,
                                approval_id,
                                operation_digest,
                                None,
                            )?;
                        }
                        (None, None) => manager.resume(run_id, None)?,
                        _ => anyhow::bail!(
                            "approval resume requires both approval-id and operation-digest"
                        ),
                    }
                    let managed = manager.take_active(run_id)?;
                    let run_store = manager.store().clone();
                    let stored = manager.store().load(run_id)?;
                    let phases = registry()
                        .ok()
                        .and_then(|registry| {
                            registry
                                .resolve(&stored.manifest.workflow_name)
                                .ok()
                                .cloned()
                        })
                        .map(|workflow| {
                            workflow
                                .meta
                                .phases
                                .into_iter()
                                .map(|phase| (phase.title, "pending".to_string()))
                                .collect()
                        })
                        .unwrap_or_default();
                    Ok::<_, anyhow::Error>((
                        managed,
                        run_store,
                        stored.manifest.workflow_name,
                        phases,
                    ))
                }) {
                    Ok((managed, run_store, name, phases)) => {
                        let run = crate::dashboard::WorkflowRun::from_managed(
                            managed,
                            run_store,
                            format!("resumed workflow {name}"),
                            phases,
                        );
                        let snapshot = run.snapshot.clone();
                        let run_id = run.run_id.clone();
                        app.workflow_runs.insert(run_id.clone(), run);
                        app.selected_workflow_run = Some(run_id.clone());
                        app.apply(crate::event::UiEvent::WorkflowUpdated { snapshot });
                        app.push(Line::styled(
                            format!("workflow {run_id} resumed; open /fleet to view its agents"),
                            accent(),
                        ));
                    }
                    Err(e) => app.push(Line::styled(
                        format!("workflow resume error: {e}"),
                        theme::theme().chrome(UiTone::Error).border,
                    )),
                }
            }
        }
        "delete" => match runtime_manager() {
            Ok(manager) if !rest.is_empty() => match manager.delete(rest) {
                Ok(()) => app.push(Line::styled(format!("deleted workflow run {rest}"), dim())),
                Err(e) => app.push(Line::styled(
                    format!("workflow delete error: {e}"),
                    theme::theme().chrome(UiTone::Error).border,
                )),
            },
            Ok(_) => app.push(Line::styled("usage: /workflow delete <run-id>", dim())),
            Err(e) => app.push(Line::styled(
                format!("workflow delete error: {e}"),
                theme::theme().chrome(UiTone::Error).border,
            )),
        },
        "pause" => app.push(Line::styled(
            "pause is cooperative: workflows pause only at a pause step; use stop to cancel",
            dim(),
        )),
        _ => app.push(Line::styled(
            format!("use /workflow {arg} to run — it opens /fleet"),
            dim(),
        )),
    }
    app.follow();
}

#[cfg(test)]
mod tests {
    use super::*;
    use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

    fn snapshot(status: hi_workflow::WorkflowRunStatus) -> hi_workflow::WorkflowRunSnapshot {
        hi_workflow::WorkflowRunSnapshot {
            run_id: "run-1".into(),
            revision: 1,
            workflow_name: "research".into(),
            objective: "compare approaches".into(),
            status,
            phases: vec![hi_workflow::WorkflowPhaseSnapshot {
                title: "Gather".into(),
                state: "active".into(),
            }],
            current_phase: Some("Gather".into()),
            agents: vec![hi_workflow::WorkflowAgentSnapshot {
                agent_id: "a1".into(),
                label: "researcher".into(),
                phase: Some("Gather".into()),
                model: None,
                state: "running".into(),
                tokens_used: 1200,
                duration_ms: 5000,
            }],
            agent_budget: Some(8),
            agents_used: 2,
            agents_reserved: 1,
            elapsed_ms: 65000,
            pause_message: None,
            result_summary: None,
            history: vec![hi_workflow::WorkflowHistoryEntry {
                event: "phase started".into(),
                detail: Some("Gather".into()),
                at_ms: 1000,
            }],
        }
    }

    fn text(lines: Vec<Line<'static>>) -> String {
        lines
            .iter()
            .map(crate::render::line_text)
            .collect::<Vec<_>>()
            .join("\n")
    }

    #[test]
    fn builtins_are_registered() {
        let reg = hi_workflow::WorkflowRegistry::scan_dirs(None, None).unwrap();
        assert!(reg.resolve("deep-research").is_ok());
        assert!(reg.list().count() >= 4);
    }

    #[test]
    fn no_progress_pause_has_a_neutral_label() {
        assert_eq!(
            status_label(hi_workflow::WorkflowRunStatus::NoProgressPaused),
            "paused — no progress"
        );
    }

    #[test]
    fn plan_workflow_children_inherit_caps_without_overriding_explicit_flags() {
        let mut inherited = vec!["workflow".into(), "run".into(), "plan.md".into()];
        append_inherited_execution_caps(&mut inherited, Some(7), Some(0), Some(2));
        assert_eq!(
            inherited,
            [
                "workflow",
                "run",
                "plan.md",
                "--max-steps",
                "7",
                "--max-tool-calls",
                "0",
                "--max-verify-repairs",
                "2",
            ]
        );

        let mut explicit = vec![
            "workflow".into(),
            "run".into(),
            "plan.md".into(),
            "--max-steps".into(),
            "11".into(),
            "--max-tool-calls".into(),
            "13".into(),
            "--max-verify-repairs".into(),
            "3".into(),
        ];
        append_inherited_execution_caps(&mut explicit, Some(7), Some(0), Some(2));
        assert_eq!(
            explicit
                .iter()
                .filter(|argument| argument.as_str() == "--max-steps")
                .count(),
            1
        );
        assert_eq!(
            explicit
                .iter()
                .filter(|argument| argument.as_str() == "--max-tool-calls")
                .count(),
            1
        );
        assert!(
            explicit
                .windows(2)
                .any(|pair| pair == ["--max-steps", "11"])
        );
        assert!(
            explicit
                .windows(2)
                .any(|pair| pair == ["--max-tool-calls", "13"])
        );
        assert!(
            explicit
                .windows(2)
                .any(|pair| pair == ["--max-verify-repairs", "3"])
        );
    }

    #[test]
    fn list_and_detail_render_multi_run_fields() {
        let mut overlay =
            WorkflowOverlay::new(vec![snapshot(hi_workflow::WorkflowRunStatus::Active)]);
        let list = text(overlay_lines(&overlay));
        assert!(list.contains("running") && list.contains("research") && list.contains("Gather"));
        assert!(list.contains("3/8") && list.contains("1m 05s"), "{list}");
        overlay.view = WorkflowOverlayView::Detail;
        let detail = text(overlay_lines(&overlay));
        for expected in [
            "compare approaches",
            "Phases",
            "researcher",
            "8 budget",
            "Recent history",
            "phase started",
            "s stop",
        ] {
            assert!(detail.contains(expected), "missing {expected}: {detail}");
        }
    }

    #[test]
    fn unlimited_budget_is_rendered_explicitly() {
        let mut run = snapshot(hi_workflow::WorkflowRunStatus::Active);
        run.agent_budget = None;
        let mut overlay = WorkflowOverlay::new(vec![run]);
        let list = text(overlay_lines(&overlay));
        assert!(list.contains("3/unlimited"), "{list}");

        overlay.view = WorkflowOverlayView::Detail;
        let detail = text(overlay_lines(&overlay));
        assert!(detail.contains("unlimited budget"), "{detail}");
    }

    #[test]
    fn keys_navigate_and_expose_only_contextual_actions() {
        let mut app = crate::tests::test_app("openai", "gpt-4o");
        app.workflow_overlay = Some(WorkflowOverlay::new(vec![snapshot(
            hi_workflow::WorkflowRunStatus::Failed,
        )]));
        let key = |code| KeyEvent::new(code, KeyModifiers::NONE);
        assert!(matches!(
            handle_overlay_key(&mut app, &key(KeyCode::Enter)),
            WorkflowOverlayOutcome::Continue
        ));
        assert_eq!(
            app.workflow_overlay.as_ref().unwrap().view,
            WorkflowOverlayView::Detail
        );
        assert!(matches!(
            handle_overlay_key(&mut app, &key(KeyCode::Char('s'))),
            WorkflowOverlayOutcome::Continue
        ));
        assert!(
            matches!(handle_overlay_key(&mut app, &key(KeyCode::Char('r'))), WorkflowOverlayOutcome::Command(command) if command == "/workflow resume run-1")
        );
        assert!(
            matches!(handle_overlay_key(&mut app, &key(KeyCode::Char('d'))), WorkflowOverlayOutcome::Command(command) if command == "/workflow delete run-1")
        );
        assert!(matches!(
            handle_overlay_key(&mut app, &key(KeyCode::Esc)),
            WorkflowOverlayOutcome::Continue
        ));
        assert_eq!(
            app.workflow_overlay.as_ref().unwrap().view,
            WorkflowOverlayView::List
        );
        assert!(matches!(
            handle_overlay_key(&mut app, &key(KeyCode::Esc)),
            WorkflowOverlayOutcome::Close
        ));
    }
}
