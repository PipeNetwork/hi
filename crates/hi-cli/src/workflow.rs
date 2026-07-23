//! Workflow script discovery, validation, and headless execution.
//!
//! `/workflow list` — list available workflows (built-in + `~/.hi/workflows/`).
//! `/workflow show <name>` — print a workflow's meta (name, description, phases).
//! `/workflow validate <file>` — dry-run a script with a stub host.
//! `/workflow <name> [args...]` — run a workflow headless (plain REPL).

use std::path::PathBuf;

use hi_workflow::{WorkflowOutcome, extract_meta, validate_script};

/// Built-in workflow scripts shipped with the binary.
const BUILTIN_WORKFLOWS: &[(&str, &str)] = &[
    ("deep-research", include_str!("../../hi-workflow/scripts/deep-research.rhai")),
    ("review-and-fix", include_str!("../../hi-workflow/scripts/review-and-fix.rhai")),
    ("port-feature", include_str!("../../hi-workflow/scripts/port-feature.rhai")),
];

/// A discovered workflow: its name, source path (None for built-in), and script
/// text.
struct DiscoveredWorkflow {
    name: String,
    path: Option<PathBuf>,
    script: String,
}

/// The user's workflow directory: `$XDG_DATA_HOME/hi/workflows/` or
/// `~/.local/share/hi/workflows/`.
fn workflows_dir() -> Option<PathBuf> {
    let base = std::env::var_os("XDG_DATA_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".local/share")))?;
    Some(base.join("hi").join("workflows"))
}

/// Find a workflow by name: built-in first, then `~/.hi/workflows/<name>.rhai`.
fn find_workflow(name: &str) -> Option<DiscoveredWorkflow> {
    if let Some((_, script)) = BUILTIN_WORKFLOWS.iter().find(|(n, _)| *n == name) {
        return Some(DiscoveredWorkflow {
            name: name.to_string(),
            path: None,
            script: script.to_string(),
        });
    }
    if !valid_workflow_name(name) {
        return None;
    }
    let path = workflows_dir()?.join(format!("{name}.rhai"));
    let script = std::fs::read_to_string(&path).ok()?;
    Some(DiscoveredWorkflow {
        name: name.to_string(),
        path: Some(path),
        script,
    })
}

fn valid_workflow_name(name: &str) -> bool {
    !name.is_empty()
        && name
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
}

/// List all available workflows: built-ins + any `*.rhai` in `~/.hi/workflows/`.
fn list_workflows() -> Vec<DiscoveredWorkflow> {
    let mut out: Vec<DiscoveredWorkflow> = BUILTIN_WORKFLOWS
        .iter()
        .map(|(name, script)| DiscoveredWorkflow {
            name: name.to_string(),
            path: None,
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
                        path: Some(path),
                        script,
                    });
                }
            }
        }
    }
    out
}

/// Handle `/workflow` in the plain REPL. Returns `true` to quit (never —
/// workflows are non-terminal commands).
pub(crate) fn handle_workflow_command(arg: &str) -> bool {
    let arg = arg.trim();
    if arg.is_empty() {
        print_workflow_help();
        return false;
    }
    let (sub, rest) = split_subcommand(arg);
    match sub {
        "list" | "ls" => print_workflow_list(),
        "show" => {
            let name = rest.trim();
            if name.is_empty() {
                println!("\x1b[33musage: /workflow show <name>\x1b[0m");
            } else {
                print_workflow_meta(name);
            }
        }
        "validate" => {
            let path = rest.trim();
            if path.is_empty() {
                println!("\x1b[33musage: /workflow validate <file>\x1b[0m");
            } else {
                validate_workflow_file(path);
            }
        }
        _ => {
            // Not a subcommand — treat the whole arg as `<name> [args...]`.
            run_workflow_headless(arg);
        }
    }
    false
}

fn split_subcommand(arg: &str) -> (&str, &str) {
    let mut parts = arg.splitn(2, char::is_whitespace);
    let sub = parts.next().unwrap_or("");
    let rest = parts.next().unwrap_or("");
    (sub, rest)
}

fn print_workflow_help() {
    println!("\x1b[1;36m/workflow\x1b[0m — scripted multi-phase agent orchestration");
    println!();
    println!("  \x1b[2m/workflow list\x1b[0m                  list available workflows");
    println!("  \x1b[2m/workflow show <name>\x1b[0m           show a workflow's meta and phases");
    println!("  \x1b[2m/workflow validate <file>\x1b[0m       dry-run a script (stub host)");
    println!("  \x1b[2m/workflow <name> [args...]\x1b[0m      run a workflow headless");
}

fn print_workflow_list() {
    let workflows = list_workflows();
    if workflows.is_empty() {
        println!("\x1b[2mno workflows available\x1b[0m");
        return;
    }
    println!("\x1b[1;35mavailable workflows ({}):\x1b[0m", workflows.len());
    for w in &workflows {
        let source = match &w.path {
            None => "\x1b[2m(built-in)\x1b[0m",
            Some(p) => p.to_str().unwrap_or("?"),
        };
        let desc = extract_meta(&w.script)
            .map(|m| m.description)
            .unwrap_or_else(|e| format!("\x1b[31minvalid: {e}\x1b[0m"));
        println!("  \x1b[1m{:<20}\x1b[0m \x1b[2m{}\x1b[0m", w.name, source);
        println!("    \x1b[2m{desc}\x1b[0m");
    }
}

fn print_workflow_meta(name: &str) {
    let Some(w) = find_workflow(name) else {
        println!("\x1b[33mno workflow named '{name}' — try /workflow list\x1b[0m");
        return;
    };
    match extract_meta(&w.script) {
        Ok(meta) => {
            println!("\x1b[1;36m{name}\x1b[0m");
            println!("  \x1b[2m{}\x1b[0m", meta.description);
            if let Some(when) = &meta.when_to_use {
                println!();
                println!("  \x1b[2mwhen to use:\x1b[0m {when}");
            }
            if !meta.phases.is_empty() {
                println!();
                println!("  \x1b[2mphases ({}):\x1b[0m", meta.phases.len());
                for (i, phase) in meta.phases.iter().enumerate() {
                    let detail = phase.detail.as_deref().unwrap_or("");
                    println!("    {}. {} \x1b[2m{detail}\x1b[0m", i + 1, phase.title);
                }
            }
        }
        Err(e) => println!("\x1b[31minvalid workflow meta: {e}\x1b[0m"),
    }
}

fn validate_workflow_file(path: &str) {
    let script = match std::fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => {
            println!("\x1b[31mcannot read {path}: {e}\x1b[0m");
            return;
        }
    };
    match validate_script(&script, None) {
        Ok(report) => {
            println!("\x1b[32m✓ valid\x1b[0m — name={}, phases={}", report.name, report.phases);
            println!("  \x1b[2m{}\x1b[0m", report.outcome_summary);
        }
        Err(hi_workflow::ValidationError::Meta(e)) => {
            println!("\x1b[31mMETA FAIL: {e}\x1b[0m");
        }
        Err(hi_workflow::ValidationError::Run(e)) => {
            println!("\x1b[31mRUN FAIL: {e}\x1b[0m");
        }
    }
}

/// Run a workflow headless: load the script, start the engine in a background
/// thread with a stub host (agents are not spawned in the plain REPL — this is
/// a validation/dry-run path). For real agent-spawning execution, use the TUI
/// dashboard's `/workflow` which wires the host to fleet rows.
fn run_workflow_headless(arg: &str) {
    let (name, args_str) = split_subcommand(arg);
    let Some(w) = find_workflow(name) else {
        println!("\x1b[33mno workflow named '{name}' — try /workflow list\x1b[0m");
        return;
    };

    // Parse args as JSON: if the args string looks like JSON, use it; otherwise
    // wrap it as a string field.
    let args = if args_str.trim().is_empty() {
        serde_json::json!({})
    } else if args_str.trim().starts_with('{') {
        serde_json::from_str(args_str).unwrap_or_else(|_| serde_json::json!({"input": args_str}))
    } else {
        serde_json::json!({"input": args_str})
    };

    let (host_tx, mut host_rx) = tokio::sync::mpsc::unbounded_channel();
    let cancel = tokio_util::sync::CancellationToken::new();

    // Stub host: logs print, agents return canned success, budget is unlimited.
    let host_thread = std::thread::spawn(move || {
        use hi_workflow::WorkflowHostRequest as R;
        while let Some(req) = host_rx.blocking_recv() {
            match req {
                R::ReserveAgentCalls { reply, .. } | R::ReleaseAgentCalls { reply, .. } => {
                    let _ = reply.send(Ok(()));
                }
                R::SpawnAgent { opts, reply } => {
                    let _ = reply.send(Ok(hi_workflow::AgentResult {
                        agent_id: "stub".into(),
                        success: true,
                        output: serde_json::json!({"summary": format!("stub agent for: {}", opts.prompt)}),
                        cancelled: false,
                        tokens_used: 0,
                        duration_ms: 0,
                    }));
                }
                R::Phase { title, replayed } => {
                    if !replayed {
                        println!("  \x1b[2mphase: {title}\x1b[0m");
                    }
                }
                R::Log { message, replayed } => {
                    if !replayed {
                        println!("  \x1b[2m{message}\x1b[0m");
                    }
                }
                R::Telemetry { name, replayed, .. } => {
                    if !replayed {
                        println!("  \x1b[2mtelemetry: {name}\x1b[0m");
                    }
                }
                R::BudgetQuery { reply } => {
                    let _ = reply.send(Ok(hi_workflow::BudgetState {
                        total: Some(hi_workflow::DEFAULT_AGENT_BUDGET),
                        spent: 0,
                        reserved: 0,
                        remaining: Some(hi_workflow::DEFAULT_AGENT_BUDGET),
                    }));
                }
                R::RenderTemplate { reply, .. } => {
                    let _ = reply.send(Err(hi_workflow::HostError::Unsupported(
                        "render_template not available in headless mode".into(),
                    )));
                }
                R::WriteScratchFile { reply, .. } => {
                    let _ = reply.send(Err(hi_workflow::HostError::Unsupported(
                        "scratch files not available in headless mode".into(),
                    )));
                }
                R::ReadScratchFile { reply, .. } => {
                    let _ = reply.send(Err(hi_workflow::HostError::Unsupported(
                        "scratch files not available in headless mode".into(),
                    )));
                }
                R::GitDiffSince { reply, .. } => {
                    let _ = reply.send(Err(hi_workflow::HostError::Unsupported(
                        "git diff not available in headless mode".into(),
                    )));
                }
            }
        }
    });

    let journal = hi_workflow::Journal::new(None);
    let params = hi_workflow::WorkflowRunParams {
        script: w.script,
        args,
        journal,
        host_tx,
        cancel,
        max_ops: hi_workflow::WorkflowRunParams::DEFAULT_MAX_OPS,
    };

    println!("\x1b[1;36mrunning workflow '{name}' (headless stub host)…\x1b[0m");
    let outcome = hi_workflow::run_workflow(params);
    let _ = host_thread.join();

    match outcome {
        WorkflowOutcome::Completed { result } => {
            println!("\x1b[32m✓ completed\x1b[0m");
            if !result.is_null() {
                println!("  \x1b[2mresult: {result}\x1b[0m");
            }
        }
        WorkflowOutcome::Paused { kind, message } => {
            println!("\x1b[33m⏸ paused ({}): {message}\x1b[0m", kind.as_str());
        }
        WorkflowOutcome::BudgetExceeded { message } => {
            println!("\x1b[33m⏸ budget exceeded: {message}\x1b[0m");
        }
        WorkflowOutcome::Cancelled => {
            println!("\x1b[2m◌ cancelled\x1b[0m");
        }
        WorkflowOutcome::Failed { error } => {
            println!("\x1b[31m✗ failed: {error}\x1b[0m");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtin_workflows_are_embedded() {
        assert!(!BUILTIN_WORKFLOWS.is_empty());
        for (name, script) in BUILTIN_WORKFLOWS {
            assert!(!script.is_empty(), "built-in {name} has empty script");
            // Every built-in must have valid extractable meta.
            let meta = extract_meta(script).unwrap_or_else(|e| panic!("{name}: {e}"));
            assert_eq!(meta.name, *name);
        }
    }

    #[test]
    fn builtin_workflows_pass_validation() {
        for (name, script) in BUILTIN_WORKFLOWS {
            let report = validate_script(script, None)
                .unwrap_or_else(|e| panic!("{name} failed validation: {e}"));
            assert_eq!(report.name, *name);
            assert!(report.outcome_ok, "{name} outcome not ok");
        }
    }

    #[test]
    fn find_builtin_workflow_by_name() {
        let w = find_workflow("deep-research").expect("deep-research should exist");
        assert_eq!(w.name, "deep-research");
        assert!(w.path.is_none(), "built-in should have no path");
        assert!(w.script.contains("let meta"));
    }

    #[test]
    fn find_nonexistent_workflow_returns_none() {
        assert!(find_workflow("does-not-exist-xyz").is_none());
    }

    #[test]
    fn rejects_unsafe_workflow_names() {
        for name in ["../secret", "nested/task", "/tmp/task", "", "."] {
            assert!(!valid_workflow_name(name), "accepted unsafe name: {name}");
            assert!(find_workflow(name).is_none());
        }
    }

    #[test]
    fn list_includes_all_builtins() {
        let list = list_workflows();
        let names: Vec<&str> = list.iter().map(|w| w.name.as_str()).collect();
        for (name, _) in BUILTIN_WORKFLOWS {
            assert!(names.contains(name), "list missing built-in {name}");
        }
    }

    #[test]
    fn split_subcommand_separates_name_and_rest() {
        let (sub, rest) = split_subcommand("show deep-research");
        assert_eq!(sub, "show");
        assert_eq!(rest, "deep-research");
    }

    #[test]
    fn split_subcommand_no_rest() {
        let (sub, rest) = split_subcommand("list");
        assert_eq!(sub, "list");
        assert_eq!(rest, "");
    }

    #[test]
    fn user_workflows_dir_respects_xdg() {
        // With XDG_DATA_HOME set, the dir should be under it.
        let dir = std::env::temp_dir().join("hi-workflow-test-xdg");
        // SAFETY: this test is single-threaded and no other code reads
        // XDG_DATA_HOME concurrently during this test.
        unsafe {
            std::env::set_var("XDG_DATA_HOME", &dir);
        }
        let workflows = workflows_dir().unwrap();
        assert!(workflows.starts_with(&dir), "got {workflows:?}");
        assert!(workflows.ends_with("workflows"));
        // SAFETY: same single-threaded context.
        unsafe {
            std::env::remove_var("XDG_DATA_HOME");
        }
    }
}
