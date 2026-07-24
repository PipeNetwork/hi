//! `/workflow` handling for the TUI.

use std::path::Path;
use ratatui::style::{Modifier, Style};
use ratatui::text::Line;
use crate::{App, dim, theme};

fn run_store() -> Option<hi_workflow::WorkflowRunStore> {
    std::env::var_os("XDG_STATE_HOME")
        .map(std::path::PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|home| std::path::PathBuf::from(home).join(".local/state")))
        .map(|base| hi_workflow::WorkflowRunStore::new(base.join("hi/workflow-runs")))
}

fn runtime_manager() -> anyhow::Result<hi_workflow::WorkflowRuntimeManager> {
    let store = run_store().ok_or_else(|| anyhow::anyhow!("workflow state directory is unavailable"))?;
    Ok(hi_workflow::WorkflowRuntimeManager::new(store))
}

fn run_id<'a>(app: &'a App, explicit: &'a str) -> Option<&'a str> {
    (!explicit.is_empty()).then_some(explicit).or_else(|| app.workflow_run.as_ref().map(|run| run.run_id.as_str()))
}

fn registry() -> anyhow::Result<hi_workflow::WorkflowRegistry> {
    let root = Path::new(".");
    Ok(hi_workflow::WorkflowRegistry::scan(
        Some(root),
        hi_agent::workspace_trusted(root),
    )?)
}

fn accent() -> Style {
    Style::default().fg(theme::theme().accent_assistant).add_modifier(Modifier::BOLD)
}

pub(crate) async fn start_workflow_run(app: &mut App, arg: &str) -> anyhow::Result<()> {
    let mut parts = arg.splitn(2, char::is_whitespace);
    let name = parts.next().unwrap_or("");
    let args_str = parts.next().unwrap_or("").trim();
    let registry = registry()?;
    let workflow = registry.resolve(name)?;
    let args = if args_str.is_empty() { serde_json::json!({}) }
        else if args_str.starts_with('{') { serde_json::from_str(args_str).map_err(|e| anyhow::anyhow!("invalid workflow JSON arguments: {e}"))? }
        else { serde_json::json!({"input": args_str}) };
    app.push(Line::styled(format!("starting workflow '{name}'…"), accent()));
    app.follow();
    crate::dashboard::start_workflow_run(app, workflow.script.clone(), args).await
}

/// `/workflow plan …` — the local plan-objectives engine (`hi workflow run`),
/// spawned as a detached child so the session stays interactive. The child
/// checkpoints under the state root and survives this TUI exiting; `status`
/// tails its log and `stop` terminates it.
pub(crate) fn handle_plan_workflow(app: &mut App, rest: &str, exe: &Path) {
    let error = |app: &mut App, text: String| {
        app.push(Line::styled(text, Style::default().fg(theme::theme().accent_error)));
        app.follow();
    };
    let rest = rest.trim();
    let mut parts = rest.split_whitespace();
    match parts.next() {
        None | Some("help") => {
            for line in [
                "/workflow plan — build a plan.md of objectives with the workflow engine",
                "  /workflow plan <plan.md> [--verify CMD] [--parallel N] [--dry-run]",
                "  /workflow plan resume <plan.md>   continue the latest sealed checkpoint",
                "  /workflow plan status             child liveness + recent output",
                "  /workflow plan stop               terminate the running child",
            ] {
                app.push(Line::styled(line, if line.starts_with("/workflow plan —") { accent() } else { dim() }));
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
            for line in tail.lines().rev().take(12).collect::<Vec<_>>().into_iter().rev() {
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
                if resume { "resume".into() } else { "run".into() },
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
                    error(app, format!("cannot create workflow log {}: {err}", log.display()));
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

pub(crate) fn handle_workflow_tui(app: &mut App, arg: &str) {
    let arg = arg.trim();
    if arg.is_empty() {
        for line in [
            "/workflow — scripted multi-phase agent orchestration",
            "  /workflow list                  list available workflows",
            "  /workflow show <name>           show workflow metadata",
            "  /workflow validate <file>       dry-run a script",
            "  /workflow runs                  list persisted runs",
            "  /workflow status [run-id]       show run details",
            "  /workflow resume <run-id>       restart an interrupted run",
            "  /workflow delete <run-id>       delete a terminal run",
            "  /workflow stop [run-id]         cancel the active run",
            "  /workflow <name> [args...]      run with live agents",
            "  /workflow plan <plan.md>        build a plan of objectives (see /workflow plan help)",
        ] { app.push(Line::styled(line, if line.starts_with("/workflow —") { accent() } else { dim() })); }
        app.follow(); return;
    }
    let mut parts = arg.splitn(2, char::is_whitespace);
    let sub = parts.next().unwrap_or("");
    let rest = parts.next().unwrap_or("").trim();
    match sub {
        "list" | "ls" => match registry() {
            Ok(reg) => for w in reg.list() { app.push(Line::styled(format!("  {:<20} {}", w.name, w.meta.description), dim())); },
            Err(e) => app.push(Line::styled(format!("workflow registry error: {e}"), Style::default().fg(theme::theme().accent_error))),
        },
        "show" => match registry().and_then(|r| r.resolve(rest).cloned().map_err(anyhow::Error::from)) {
            Ok(w) => {
                app.push(Line::styled(w.meta.name, accent()));
                app.push(Line::styled(format!("  {}", w.meta.description), dim()));
                if let Some(usage) = w.meta.when_to_use { app.push(Line::styled(format!("  when: {usage}"), dim())); }
                for phase in w.meta.phases { app.push(Line::styled(format!("  ○ {}", phase.title), dim())); }
            }
            Err(e) => app.push(Line::styled(e.to_string(), Style::default().fg(theme::theme().accent_error))),
        },
        "validate" => match std::fs::read_to_string(rest) {
            Ok(source) => match hi_workflow::DeclarativeWorkflow::from_json(&source)
                .map_err(|e| e.to_string())
                .and_then(|workflow| workflow.validate().map(|_| workflow).map_err(|e| e.to_string())) {
                Ok(workflow) => app.push(Line::styled(format!("VALID — name={}, steps={}", workflow.metadata.name, workflow.steps.len()), Style::default().fg(theme::theme().accent_success))),
                Err(e) => app.push(Line::styled(format!("INVALID: {e}"), Style::default().fg(theme::theme().accent_error))),
            },
            Err(e) => app.push(Line::styled(format!("cannot read {rest}: {e}"), Style::default().fg(theme::theme().accent_error))),
        },
        "runs" => match runtime_manager().and_then(|manager| manager.list().map_err(anyhow::Error::from)) {
            Ok(runs) if runs.is_empty() => app.push(Line::styled("no persisted workflow runs", dim())),
            Ok(runs) => for run in runs {
                let m = run.manifest;
                app.push(Line::styled(format!("  {:<28} {:<18} {:?}", m.run_id, m.workflow_name, m.status), dim()));
            },
            Err(e) => app.push(Line::styled(format!("workflow run list error: {e}"), Style::default().fg(theme::theme().accent_error))),
        },
        "status" | "details" => match run_id(app, rest) {
            Some(id) => match run_store().ok_or_else(|| anyhow::anyhow!("workflow state directory is unavailable")).and_then(|store| store.load(id).map_err(anyhow::Error::from)) {
                Ok(run) => {
                    let m = run.manifest;
                    app.push(Line::styled(format!("{} — {}", m.workflow_name, m.run_id), accent()));
                    app.push(Line::styled(format!("  status: {:?}  agents: {}/{}", m.status, m.agent_spent, m.agent_budget), dim()));
                    if let Some(phase) = m.current_phase { app.push(Line::styled(format!("  phase: {phase}"), dim())); }
                    if let Some(outcome) = m.outcome { app.push(Line::styled(format!("  outcome: {outcome:?}"), dim())); }
                }
                Err(e) => app.push(Line::styled(format!("workflow status error: {e}"), Style::default().fg(theme::theme().accent_error))),
            },
            None => app.push(Line::styled("no workflow run selected; use /workflow runs", dim())),
        },
        "stop" => {
            if let Some(run) = &app.workflow_run { run.cancel.cancel(); app.push(Line::styled("workflow cancellation requested", dim())); }
            else { app.push(Line::styled("no active workflow", dim())); }
        }
        "resume" => match runtime_manager() {
            Ok(mut manager) if !rest.is_empty() => match manager.resume(rest, None) {
                Ok(()) => app.push(Line::styled(format!("workflow {rest} restarted; open the dashboard to service agent requests"), accent())),
                Err(e) => app.push(Line::styled(format!("workflow resume error: {e}"), Style::default().fg(theme::theme().accent_error))),
            },
            Ok(_) => app.push(Line::styled("usage: /workflow resume <run-id>", dim())),
            Err(e) => app.push(Line::styled(format!("workflow resume error: {e}"), Style::default().fg(theme::theme().accent_error))),
        },
        "delete" => match runtime_manager() {
            Ok(manager) if !rest.is_empty() => match manager.delete(rest) {
                Ok(()) => app.push(Line::styled(format!("deleted workflow run {rest}"), dim())),
                Err(e) => app.push(Line::styled(format!("workflow delete error: {e}"), Style::default().fg(theme::theme().accent_error))),
            },
            Ok(_) => app.push(Line::styled("usage: /workflow delete <run-id>", dim())),
            Err(e) => app.push(Line::styled(format!("workflow delete error: {e}"), Style::default().fg(theme::theme().accent_error))),
        },
        "pause" => app.push(Line::styled("pause is cooperative: workflows pause only at a pause step; use stop to cancel", dim())),
        _ => app.push(Line::styled(format!("use /workflow {arg} to run — it opens the dashboard"), dim())),
    }
    app.follow();
}

#[cfg(test)]
mod tests {
    
    #[test] fn builtins_are_registered() {
        let reg = hi_workflow::WorkflowRegistry::scan_dirs(None, None).unwrap();
        assert!(reg.resolve("deep-research").is_ok());
        assert!(reg.list().count() >= 4);
    }
}
