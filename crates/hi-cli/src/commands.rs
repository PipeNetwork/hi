//! Slash-command handler: act on a parsed `/command` for the line REPL.
//!
//! The async commands that drive a turn or run shell work (`/retry`, `/compact`,
//! `/diff`, `/commit`, `/undo`, `/init`) are handled inline in the REPL loop in
//! [`crate::repl`]; this module covers the synchronous remainder.

use hi_agent::Agent;
use hi_ai::Registry;

/// Act on a slash command. Returns true when the session should quit.
pub(crate) fn handle_command(
    agent: &mut Agent,
    command: hi_agent::Command,
    registry: &Registry,
) -> bool {
    use hi_agent::Command;
    match command {
        Command::Quit => return true,
        Command::Help => println!("{}", hi_agent::command::help_text()),
        Command::Status => {
            let t = agent.totals();
            let tel = agent.last_turn_telemetry();
            let ctx = agent
                .context_window()
                .map(|w| {
                    let pct = if w > 0 {
                        agent.context_used() * 100 / w as u64
                    } else {
                        0
                    };
                    format!("{pct}% of {}k", w / 1000)
                })
                .unwrap_or_else(|| "unknown".into());
            println!(
                "\x1b[2mstatus: ready\nmodel: {}\nusage: {} in · {} out · {} total\ncontext: {}\ngoal: {}\nverify: {}\nevidence: {} (reads {}, searches {}, listing_only {}, repair nudges {})\ncheckpoints: {}\x1b[0m",
                agent.model(),
                t.input_tokens,
                t.output_tokens,
                t.total(),
                ctx,
                agent.goal().unwrap_or("off"),
                agent.verify_summary(),
                tel.discovery_depth,
                tel.file_reads,
                tel.targeted_searches,
                tel.listing_only,
                tel.quality_repair_nudges,
                agent.checkpoint_count(),
            );
        }
        Command::Log => {
            let t = agent.totals();
            let body = format!(
                "# hi debug log\n\nmodel: {}\nusage: {} in · {} out · {} total\ngoal: {}\nverify: {}\nlast_error: none\ncheckpoints: {}\n",
                agent.model(),
                t.input_tokens,
                t.output_tokens,
                t.total(),
                agent.goal().unwrap_or("off"),
                agent.verify_summary(),
                agent.checkpoint_count(),
            );
            match std::fs::write(".hi-debug.log", body) {
                Ok(()) => println!("\x1b[2mwrote debug log: .hi-debug.log\x1b[0m"),
                Err(err) => eprintln!("\x1b[33mlog failed: {err}\x1b[0m"),
            }
        }
        Command::Model(id) => {
            if id.is_empty() {
                // The line REPL can't do an arrow-select picker; show the current
                // model + how to switch (the full-screen TUI has a live picker).
                println!(
                    "model: {}\n\x1b[2m{} models known — `/model <id>` to switch (the TUI's /model opens an interactive picker)\x1b[0m",
                    agent.model(),
                    registry.model_ids().len()
                );
            } else {
                let (_price, context_window) = registry.metadata(&id);
                agent.set_model(id.clone(), context_window, None);
                println!("model set to {id}");
            }
        }
        Command::Clear => {
            let count = agent
                .messages()
                .iter()
                .filter(|m| m.role != hi_ai::Role::System)
                .count();
            match agent.clear_history() {
                Ok(()) => println!("\x1b[2mcleared {count} messages — starting fresh\x1b[0m"),
                Err(err) => eprintln!("\x1b[33mclear failed: {err}\x1b[0m"),
            }
        }
        Command::Verify(arg) => match arg.trim() {
            "" if agent.verify_is_on() => {
                println!("\x1b[2mverify: {}\x1b[0m", agent.verify_summary())
            }
            "" => println!("\x1b[2mverify: off (set one with /verify <cmd>)\x1b[0m"),
            "off" | "none" | "clear" | "disable" => {
                agent.set_verify_command(None);
                println!("\x1b[2mverification disabled\x1b[0m");
            }
            cmd => {
                agent.set_verify_command(Some(cmd.to_string()));
                println!(
                    "\x1b[2mverification on: {cmd} — runs after each turn, iterates on failure\x1b[0m"
                );
            }
        },
        // Diff and Commit are handled in the async repl loop.
        Command::Copy(_) => {
            println!("\x1b[33m/copy is only available in the full-screen TUI\x1b[0m");
        }
        Command::Goal(arg) => match arg.trim() {
            "" => {
                // Report whichever goal view is active: the structured
                // long-horizon goal when long_horizon is on, else the
                // transient goal string.
                if let Some(g) = agent.structured_goal() {
                    println!(
                        "\x1b[2mgoal: {} — {}/{} sub-goals done\x1b[0m",
                        g.objective,
                        g.sub_goals
                            .iter()
                            .filter(|s| s.status == hi_agent::GoalStatus::Done)
                            .count(),
                        g.sub_goals.len()
                    );
                } else {
                    match agent.goal() {
                        Some(goal) => println!("\x1b[2mgoal: {goal}\x1b[0m"),
                        None => println!("\x1b[2mgoal: off (set one with /goal <text>)\x1b[0m"),
                    }
                }
            }
            "clear" | "off" | "none" => match agent.set_transient_goal(None) {
                Ok(()) => println!("\x1b[32m✓ goal cleared\x1b[0m"),
                Err(err) => eprintln!("\x1b[33mgoal clear failed: {err:#}\x1b[0m"),
            },
            goal => {
                // When long-horizon agency is on, set a structured goal — a
                // single sub-goal equal to the objective, which the model
                // decomposes as it works (its `update_plan` calls map back to
                // sub-goal statuses). Otherwise fall back to the transient
                // prompt-injected goal string.
                if agent.long_horizon() {
                    match agent.set_structured_goal(Some(hi_agent::Goal::new(
                        goal.to_string(),
                        vec![goal.to_string()],
                    ))) {
                        Ok(true) => {
                            println!(
                                "\x1b[32m✓ long-horizon goal set — drives sub-goals across turns: {goal}\x1b[0m"
                            );
                        }
                        Ok(false) => match agent.set_transient_goal(Some(goal.to_string())) {
                            Ok(()) => println!(
                                "\x1b[32m✓ goal set — steers every turn until cleared: {goal}\x1b[0m"
                            ),
                            Err(err) => eprintln!("\x1b[33mgoal set failed: {err:#}\x1b[0m"),
                        },
                        Err(err) => eprintln!("\x1b[33mgoal set failed: {err:#}\x1b[0m"),
                    }
                } else {
                    match agent.set_transient_goal(Some(goal.to_string())) {
                        Ok(()) => println!(
                            "\x1b[32m✓ goal set — steers every turn until cleared: {goal}\x1b[0m"
                        ),
                        Err(err) => eprintln!("\x1b[33mgoal set failed: {err:#}\x1b[0m"),
                    }
                }
            }
        },
        // Handled in the repl loop (async / runs a turn); never reach here.
        Command::Prompt(_)
        | Command::Moa(_)
        | Command::Compact(_)
        | Command::Retry
        | Command::Edit
        | Command::Undo
        | Command::Init
        | Command::Learn(_)
        | Command::Skill(_)
        | Command::Diff
        | Command::Commit => {}
        Command::Version => {
            println!("hi {}", hi_agent::VERSION);
        }
        Command::Export(arg) => {
            let path = if arg.trim().is_empty() {
                "transcript.md"
            } else {
                arg.trim()
            };
            let content = agent.export_markdown();
            match std::fs::write(path, &content) {
                Ok(()) => println!(
                    "\x1b[2mexported {} messages to {path}\x1b[0m",
                    agent
                        .messages()
                        .iter()
                        .filter(|m| m.role != hi_ai::Role::System)
                        .count()
                ),
                Err(err) => eprintln!("\x1b[33mexport failed: {err}\x1b[0m"),
            }
        }
        Command::Unknown(name) => {
            eprintln!("\x1b[33munknown command /{name}; try /help\x1b[0m");
        }
        Command::Removed(msg) => {
            eprintln!("\x1b[33m/{msg}\x1b[0m");
        }
        Command::Context => {
            print!("{}", agent.context_breakdown());
        }
        Command::Skills => {
            let skills = hi_agent::list_skills();
            if skills.is_empty() {
                println!("\x1b[2mno learned skills found\x1b[0m");
            } else {
                for skill in skills {
                    println!("{}  [{}]  {}", skill.name, skill.scope, skill.description);
                }
            }
        }
        // `/provider` is handled inline by the REPL/TUI (it needs the Config
        // and a provider builder, which this synchronous handler doesn't have).
        // If it reaches here, it's a no-op — the frontend should have
        // intercepted it.
        Command::Provider(_) => {}
        // `/mcp` is handled inline by the REPL/TUI (async + needs settings).
        Command::Mcp => {}
        Command::Lsp(arg) => {
            handle_lsp(agent, &arg);
        }
    }
    false
}

pub(crate) fn handle_lsp(agent: &hi_agent::Agent, arg: &str) {
    let arg = arg.trim();
    match arg {
        "on" => {
            agent.set_lsp_enabled(true);
            println!("\x1b[2mLSP enabled — servers will warm up on first query.\x1b[0m");
        }
        "off" => {
            agent.set_lsp_enabled(false);
            println!("\x1b[2mLSP disabled.\x1b[0m");
        }
        _ => {
            // `/lsp` or `/lsp status` — show enabled state plus per-language
            // server availability and running state.
            let report = hi_tools::lsp_status_report(agent.lsp_enabled());
            for line in report.lines() {
                println!("\x1b[2m{line}\x1b[0m");
            }
        }
    }
}

pub(crate) fn tool_mode_label(mode: hi_ai::ToolMode) -> &'static str {
    match mode {
        hi_ai::ToolMode::Auto => "auto",
        hi_ai::ToolMode::Required => "required",
        hi_ai::ToolMode::ChatOnly => "chat-only",
        hi_ai::ToolMode::ReadOnly => "read-only",
    }
}
