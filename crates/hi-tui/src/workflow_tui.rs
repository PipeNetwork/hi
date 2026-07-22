//! `/workflow` handling for the TUI — list, show, validate, and run.
//!
//! `list`/`show`/`validate` push styled lines into the transcript. `run`
//! launches the workflow engine in a background thread with a stub host and
//! streams phase/log events into the transcript as they arrive — the full
//! dashboard host bridge (SpawnAgent → FleetRow) is a future extension; for
//! now the TUI run path uses the same stub host as the plain REPL, with live
//! event streaming.

use std::path::PathBuf;

use ratatui::text::Line;
use ratatui::style::{Modifier, Style};

use crate::{App, dim, theme};

const BUILTIN_WORKFLOWS: &[(&str, &str)] = &[
    ("deep-research", include_str!("../../hi-workflow/scripts/deep-research.rhai")),
    ("review-and-fix", include_str!("../../hi-workflow/scripts/review-and-fix.rhai")),
    ("port-feature", include_str!("../../hi-workflow/scripts/port-feature.rhai")),
];

fn workflows_dir() -> Option<PathBuf> {
    let base = std::env::var_os("XDG_DATA_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".local/share")))?;
    Some(base.join("hi").join("workflows"))
}

struct DiscoveredWorkflow {
    name: String,
    script: String,
}

fn find_workflow(name: &str) -> Option<DiscoveredWorkflow> {
    if let Some((_, script)) = BUILTIN_WORKFLOWS.iter().find(|(n, _)| *n == name) {
        return Some(DiscoveredWorkflow {
            name: name.to_string(),
            script: script.to_string(),
        });
    }
    let path = workflows_dir()?.join(format!("{name}.rhai"));
    let script = std::fs::read_to_string(&path).ok()?;
    Some(DiscoveredWorkflow {
        name: name.to_string(),
        script,
    })
}

fn list_workflows() -> Vec<DiscoveredWorkflow> {
    let mut out: Vec<DiscoveredWorkflow> = BUILTIN_WORKFLOWS
        .iter()
        .map(|(name, script)| DiscoveredWorkflow {
            name: name.to_string(),
            script: script.to_string(),
        })
        .collect();
    if let Some(dir) = workflows_dir()
        && let Ok(entries) = std::fs::read_dir(&dir)
    {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().is_some_and(|ext| ext == "rhai") {
                if let Some(name) = path.file_stem().and_then(|s| s.to_str())
                    && !out.iter().any(|w| w.name == name)
                    && let Ok(script) = std::fs::read_to_string(&path)
                {
                    out.push(DiscoveredWorkflow {
                        name: name.to_string(),
                        script,
                    });
                }
            }
        }
    }
    out
}

fn accent() -> Style {
    Style::default().fg(theme::theme().accent_assistant).add_modifier(Modifier::BOLD)
}

/// Start a workflow run and prepare it for the dashboard's live host bridge.
/// Finds the workflow script by name (built-in or `~/.local/share/hi/workflows/`),
/// parses args, and calls `dashboard::start_workflow_run` which launches the
/// engine in a `spawn_blocking` thread. The caller then opens the dashboard so
/// the `select!` loop can service `WorkflowHostRequest`s — `SpawnAgent` creates
/// real `FleetRow`s with worktree-isolated child `hi` turns.
pub(crate) async fn start_workflow_run(
    app: &mut App,
    arg: &str,
    launcher: &crate::FleetLauncher,
) -> anyhow::Result<()> {
    let mut parts = arg.splitn(2, char::is_whitespace);
    let name = parts.next().unwrap_or("");
    let args_str = parts.next().unwrap_or("").trim();

    let Some(w) = find_workflow(name) else {
        anyhow::bail!("no workflow named '{name}' — try /workflow list")
    };

    let args = if args_str.is_empty() {
        serde_json::json!({})
    } else if args_str.starts_with('{') {
        serde_json::from_str(args_str)
            .unwrap_or_else(|_| serde_json::json!({"input": args_str}))
    } else {
        serde_json::json!({"input": args_str})
    };

    app.push(Line::styled(
        format!("starting workflow '{name}'…"),
        accent(),
    ));
    app.follow();

    crate::dashboard::start_workflow_run(app, w.script, args, launcher).await
}

/// Handle `/workflow <arg>` in the TUI. Pushes results into the transcript.
pub(crate) fn handle_workflow_tui(app: &mut App, arg: &str) {
    let arg = arg.trim();
    if arg.is_empty() {
        app.push(Line::styled("/workflow — scripted multi-phase agent orchestration", accent()));
        app.push(Line::styled("  /workflow list                  list available workflows", dim()));
        app.push(Line::styled("  /workflow show <name>           show a workflow's meta and phases", dim()));
        app.push(Line::styled("  /workflow validate <file>       dry-run a script (stub host)", dim()));
        app.push(Line::styled("  /workflow <name> [args...]      run a workflow", dim()));
        app.follow();
        return;
    }
    let mut parts = arg.splitn(2, char::is_whitespace);
    let sub = parts.next().unwrap_or("");
    let rest = parts.next().unwrap_or("").trim();
    match sub {
        "list" | "ls" => {
            let workflows = list_workflows();
            if workflows.is_empty() {
                app.push(Line::styled("no workflows available", dim()));
            } else {
                app.push(Line::styled(format!("available workflows ({}):", workflows.len()), accent()));
                for w in &workflows {
                    let desc = hi_workflow::extract_meta(&w.script)
                        .map(|m| m.description)
                        .unwrap_or_else(|e| format!("invalid: {e}"));
                    app.push(Line::styled(format!("  {}", w.name), accent()));
                    app.push(Line::styled(format!("    {desc}"), dim()));
                }
            }
        }
        "show" => {
            if rest.is_empty() {
                app.push(Line::styled("usage: /workflow show <name>", dim()));
            } else if let Some(w) = find_workflow(rest) {
                match hi_workflow::extract_meta(&w.script) {
                    Ok(meta) => {
                        app.push(Line::styled(w.name.clone(), accent()));
                        app.push(Line::styled(format!("  {}", meta.description), dim()));
                        if !meta.phases.is_empty() {
                            app.push(Line::styled(format!("  phases ({}):", meta.phases.len()), dim()));
                            for (i, phase) in meta.phases.iter().enumerate() {
                                let detail = phase.detail.as_deref().unwrap_or("");
                                app.push(Line::styled(
                                    format!("    {}. {} {}", i + 1, phase.title, detail),
                                    dim(),
                                ));
                            }
                        }
                    }
                    Err(e) => app.push(Line::styled(format!("invalid workflow meta: {e}"), Style::default().fg(theme::theme().accent_error))),
                }
            } else {
                app.push(Line::styled(format!("no workflow named '{rest}' — try /workflow list"), dim()));
            }
        }
        "validate" => {
            if rest.is_empty() {
                app.push(Line::styled("usage: /workflow validate <file>", dim()));
            } else {
                match std::fs::read_to_string(rest) {
                    Ok(script) => match hi_workflow::validate_script(&script, None) {
                        Ok(report) => {
                            app.push(Line::styled(
                                format!("✓ valid — name={}, phases={}", report.name, report.phases),
                                Style::default().fg(theme::theme().accent_success),
                            ));
                            app.push(Line::styled(format!("  {}", report.outcome_summary), dim()));
                        }
                        Err(hi_workflow::ValidationError::Meta(e)) => {
                            app.push(Line::styled(format!("META FAIL: {e}"), Style::default().fg(theme::theme().accent_error)));
                        }
                        Err(hi_workflow::ValidationError::Run(e)) => {
                            app.push(Line::styled(format!("RUN FAIL: {e}"), Style::default().fg(theme::theme().accent_error)));
                        }
                    },
                    Err(e) => app.push(Line::styled(format!("cannot read {rest}: {e}"), Style::default().fg(theme::theme().accent_error))),
                }
            }
        }
        _ => {
            // Run commands are handled by the TUI run loop (which opens the
            // dashboard with a live host bridge). This arm should never fire
            // for run commands — but if it does, point the user to the
            // dashboard.
            app.push(Line::styled(
                format!("use /workflow {arg} to run — it opens the dashboard with live agents"),
                dim(),
            ));
        }
    }
    app.follow();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtin_workflows_are_embedded() {
        assert!(!BUILTIN_WORKFLOWS.is_empty());
        for (name, script) in BUILTIN_WORKFLOWS {
            assert!(!script.is_empty(), "built-in {name} has empty script");
            let meta = hi_workflow::extract_meta(script).unwrap_or_else(|e| panic!("{name}: {e}"));
            assert_eq!(meta.name, *name);
        }
    }

    #[test]
    fn find_builtin_workflow_by_name() {
        let w = find_workflow("deep-research").expect("deep-research should exist");
        assert_eq!(w.name, "deep-research");
        assert!(w.script.contains("let meta"));
    }

    #[test]
    fn find_nonexistent_workflow_returns_none() {
        assert!(find_workflow("does-not-exist-xyz").is_none());
    }

    #[test]
    fn list_includes_all_builtins() {
        let list = list_workflows();
        let names: Vec<&str> = list.iter().map(|w| w.name.as_str()).collect();
        for (name, _) in BUILTIN_WORKFLOWS {
            assert!(names.contains(name), "list missing built-in {name}");
        }
    }
}
