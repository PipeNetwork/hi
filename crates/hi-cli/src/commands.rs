//! Slash-command handler: act on a parsed `/command` for the line REPL.
//!
//! The async commands that drive a turn or run shell work (`/retry`, `/compact`,
//! `/diff`, `/commit`, `/undo`, `/init`) are handled inline in the REPL loop in
//! [`crate::repl`]; this module covers the synchronous remainder.

use hi_agent::Agent;

/// Act on a slash command. Returns true when the session should quit.
pub(crate) fn handle_command(agent: &mut Agent, command: hi_agent::Command) -> bool {
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
                "\x1b[2mstatus: ready\nmodel: {}\nsession usage: {} full-context in · {} generated out · {} total\nlast turn: {} prompt · {} generated\ncontext: {}\ngoal: {}\nverify: {}\nevidence: {} (reads {}, searches {}, listing_only {}, repair nudges {})\ncheckpoints: {}\x1b[0m",
                agent.model(),
                t.input_tokens,
                t.output_tokens,
                t.total(),
                agent.last_user_prompt_tokens(),
                agent.last_turn_usage().output_tokens,
                ctx,
                agent.goal_summary(),
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
                "# hi debug log\n\nmodel: {}\nsession usage: {} full-context in · {} generated out · {} total\nlast turn: {} prompt · {} generated\ngoal: {}\nverify: {}\nlast_error: none\ncheckpoints: {}\n",
                agent.model(),
                t.input_tokens,
                t.output_tokens,
                t.total(),
                agent.last_user_prompt_tokens(),
                agent.last_turn_usage().output_tokens,
                agent.goal_summary(),
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
                // model.
                println!("model: {}", agent.model());
            } else {
                agent.set_model(id.clone(), None, None);
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
        Command::Config(arg) => {
            use hi_agent::command::{ConfigArg, parse_config_arg};
            match parse_config_arg(&arg) {
                ConfigArg::Show => {
                    let r = agent
                        .reasoning_effort()
                        .map(|e| e.as_str().to_string())
                        .unwrap_or_else(|| "off".into());
                    let t = agent
                        .temperature()
                        .map(|t| t.to_string())
                        .unwrap_or_else(|| "default".into());
                    println!("\x1b[2mconfig — reasoning: {r} · temperature: {t}\x1b[0m");
                    println!(
                        "\x1b[2mset: /config reasoning <minimal|low|medium|high|xhigh|off> · /config temp <0.0-2.0|off>\x1b[0m"
                    );
                }
                ConfigArg::Reasoning(effort) => {
                    agent.set_reasoning_effort(effort);
                    match effort {
                        Some(e) => println!(
                            "\x1b[2mreasoning effort → {} (applies next turn; OpenAI-compatible endpoints only)\x1b[0m",
                            e.as_str()
                        ),
                        None => println!(
                            "\x1b[2mreasoning effort → off (no reasoning_effort sent; endpoint default)\x1b[0m"
                        ),
                    }
                }
                ConfigArg::Temperature(temp) => {
                    agent.set_temperature(temp);
                    match temp {
                        Some(t) => println!("\x1b[2mtemperature → {t}\x1b[0m"),
                        None => println!("\x1b[2mtemperature → provider default (cleared)\x1b[0m"),
                    }
                }
                ConfigArg::Invalid(m) => eprintln!("\x1b[33m{m}\x1b[0m"),
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
            s if hi_agent::command::parse_goal_limit(s).is_some() => {
                if let Some(limit) = hi_agent::command::parse_goal_limit(s) {
                    handle_goal_limit(agent, limit);
                }
            }
            s if hi_agent::command::parse_goal_team(s).is_some() => {
                if let Some(team) = hi_agent::command::parse_goal_team(s) {
                    handle_goal_team(agent, team);
                }
            }
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
            "pause" => {
                if agent.set_goal_paused(true) {
                    println!("\x1b[32m✓ goal paused — resume with /goal resume\x1b[0m");
                } else {
                    println!("\x1b[2mno goal to pause\x1b[0m");
                }
            }
            "resume" => {
                if agent.set_goal_paused(false) {
                    println!("\x1b[32m✓ goal resumed — steering turns again\x1b[0m");
                } else {
                    println!("\x1b[2mno goal to resume\x1b[0m");
                }
            }
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
        | Command::Commit
        | Command::Hf(_) => {}
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
        Command::Sync(arg) => match arg.trim() {
            "status" | "" => {
                println!("\x1b[2muse /sync in the TUI, or `hi --sync` on the CLI\x1b[0m");
            }
            _ => {
                println!("\x1b[33m/sync is only available in the full-screen TUI\x1b[0m");
            }
        },
        Command::Sessions(arg) => match arg.trim() {
            "" => {
                let sessions = crate::session::local_sessions();
                if sessions.is_empty() {
                    println!("\x1b[2mno saved sessions in this project\x1b[0m");
                } else {
                    println!("\x1b[2msessions:\x1b[0m");
                    for s in sessions {
                        println!("\x1b[2m  {} ({}, {} lines)\x1b[0m", s.id, s.age, s.lines);
                    }
                }
            }
            value if value == "sync" || value.starts_with("sync ") => {
                println!("\x1b[2muse /sessions sync in the TUI, or start hi with --sync\x1b[0m");
            }
            value if value == "attach" || value.starts_with("attach ") => {
                println!("\x1b[33mattaching requires the TUI or `hi --attach <id>`\x1b[0m");
            }
            value if value == "host" || value.starts_with("host ") => {
                println!("\x1b[33mhosting requires the TUI or `hi --daemon --sync`\x1b[0m");
            }
            _ => {
                println!(
                    "\x1b[33msession switching and renaming require the TUI (run hi without --plain)\x1b[0m"
                );
            }
        },
        Command::Attach(_) => {
            println!(
                "\x1b[33m/attach is only available in the full-screen TUI; or run `hi --attach <id>`\x1b[0m"
            );
        }
        Command::Daemon(_) => {
            println!(
                "\x1b[33m/daemon is only available in the full-screen TUI; or run `hi --daemon --sync`\x1b[0m"
            );
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
        Command::Delegate(arg) => {
            handle_delegate_command(agent, &arg);
        }
        Command::Loop(_) => {
            println!(
                "\x1b[33m/loop is only available in the full-screen TUI (run hi without --plain)\x1b[0m"
            );
        }
        Command::Watch => {
            println!(
                "\x1b[33m/watch is only available in the full-screen TUI (run hi without --plain)\x1b[0m"
            );
        }
        Command::Digest => {
            println!(
                "\x1b[33m/digest is only available in the full-screen TUI (run hi without --plain)\x1b[0m"
            );
        }
        Command::Dashboard(arg) => match arg.trim() {
            "status" | "sessions" | "ls" => {
                let sessions = crate::session::fleet_sessions();
                if sessions.is_empty() {
                    println!("\x1b[2mno fleet sessions in this project yet\x1b[0m");
                } else {
                    println!("\x1b[1;35mfleet sessions ({}):\x1b[0m", sessions.len());
                    for s in sessions.iter().take(20) {
                        println!(
                            "\x1b[2m  {}  {:>8} \u{b7} {:>4} lines \u{b7} {}\x1b[0m",
                            s.id, s.age, s.lines, s.title
                        );
                    }
                    println!("\x1b[2mresume one with: hi --resume <id>\x1b[0m");
                }
            }
            _ => println!(
                "\x1b[33m/dashboard is only available in the full-screen TUI (run hi without --plain); /fleet status works here\x1b[0m"
            ),
        },
    }
    false
}

pub(crate) fn handle_delegate_command(agent: &mut hi_agent::Agent, arg: &str) {
    match arg.trim() {
        "on" => {
            agent.set_write_subagents(true);
            println!(
                "\x1b[2mdelegate enabled — the model can now hand a self-contained subtask to a \
                 worktree-isolated subagent whose changes are kept only if they verify.\x1b[0m"
            );
        }
        "off" => {
            agent.set_write_subagents(false);
            println!("\x1b[2mdelegate disabled.\x1b[0m");
        }
        _ => {
            let state = if agent.write_subagents_enabled() {
                "on"
            } else {
                "off"
            };
            println!(
                "\x1b[2mdelegate is {state} (off by default). `/delegate on` to enable — it's \
                 worktree-isolated and verify-gated; best for large, independently-verifiable \
                 subtasks.\x1b[0m"
            );
        }
    }
}

/// `/goal <objective>` with a planner (the async path driven from the repl):
/// decompose the objective into sub-goals via one bounded planner call, install the
/// structured goal, and echo the checklist. Falls back to a single sub-goal on
/// failure so `/goal` always sets *something*.
pub(crate) async fn handle_goal_planned(agent: &mut hi_agent::Agent, objective: &str) {
    println!("\x1b[2mplanning goal with the planner model…\x1b[0m");
    let sub_goals = match agent.decompose_goal(objective).await {
        Ok(steps) if !steps.is_empty() => steps,
        Ok(_) => vec![objective.to_string()],
        Err(err) => {
            println!(
                "\x1b[2mplanner unavailable ({err:#}); using the objective as one step\x1b[0m"
            );
            vec![objective.to_string()]
        }
    };
    match agent.set_structured_goal(Some(hi_agent::Goal::new(objective.to_string(), sub_goals))) {
        Ok(true) => {
            if let Some(g) = agent.structured_goal() {
                println!(
                    "\x1b[32m✓ long-horizon goal set — {} sub-goal(s):\x1b[0m",
                    g.sub_goals.len()
                );
                for (i, s) in g.sub_goals.iter().enumerate() {
                    println!("\x1b[2m  {}. {}\x1b[0m", i + 1, s.description);
                }
            }
        }
        Ok(false) => match agent.set_transient_goal(Some(objective.to_string())) {
            Ok(()) => {
                println!("\x1b[32m✓ goal set — steers every turn until cleared: {objective}\x1b[0m")
            }
            Err(err) => eprintln!("\x1b[33mgoal set failed: {err:#}\x1b[0m"),
        },
        Err(err) => eprintln!("\x1b[33mgoal set failed: {err:#}\x1b[0m"),
    }
}

fn handle_goal_limit(agent: &mut hi_agent::Agent, limit: hi_agent::command::GoalLimitArg) {
    use hi_agent::command::GoalLimitArg;
    match limit {
        GoalLimitArg::Show => match agent.structured_goal().and_then(|g| g.step_limit) {
            Some(n) => println!("\x1b[2mgoal limit: {n} sub-goals\x1b[0m"),
            None => println!("\x1b[2mgoal limit: none — the plan grows freely\x1b[0m"),
        },
        GoalLimitArg::Set(n) => {
            if agent.set_goal_step_limit(Some(n)) {
                println!("\x1b[32m✓ goal limit set to {n} sub-goals\x1b[0m");
            } else {
                println!("\x1b[2mno goal to limit\x1b[0m");
            }
        }
        GoalLimitArg::Unlimited => {
            if agent.set_goal_step_limit(None) {
                println!("\x1b[32m✓ goal limit removed — the plan grows freely\x1b[0m");
            } else {
                println!("\x1b[2mno goal to limit\x1b[0m");
            }
        }
        GoalLimitArg::Invalid(value) => {
            eprintln!(
                "\x1b[33mgoal limit: '{value}' isn't a number — use /goal limit <n> or 'limit off'\x1b[0m"
            );
        }
    }
}

fn handle_goal_team(agent: &mut hi_agent::Agent, team: hi_agent::command::GoalTeamArg) {
    use hi_agent::command::GoalTeamArg;
    match team {
        GoalTeamArg::Show => match agent.structured_goal() {
            Some(g) if g.team => println!(
                "\x1b[2mgoal team: on — a skeptic reviews each advance ({} objection(s) so far)\x1b[0m",
                g.skeptic_objections
            ),
            Some(_) => println!("\x1b[2mgoal team: off — enable with /goal team on\x1b[0m"),
            None => println!("\x1b[2mno active goal — set one with /goal <text> first\x1b[0m"),
        },
        GoalTeamArg::On => {
            if !agent.has_skeptic() {
                eprintln!(
                    "\x1b[33mgoal team: no skeptic model configured — set HI_SKEPTIC_MODEL (or a profile skeptic_model) first\x1b[0m"
                );
            } else if agent.set_goal_team(true) {
                println!(
                    "\x1b[32m✓ goal team on — a skeptic reviews each turn before it advances a sub-goal\x1b[0m"
                );
            } else {
                println!("\x1b[2mno active goal — set one with /goal <text> first\x1b[0m");
            }
        }
        GoalTeamArg::Off => {
            if agent.set_goal_team(false) {
                println!("\x1b[32m✓ goal team off — single-agent driving\x1b[0m");
            } else {
                println!("\x1b[2mno active goal\x1b[0m");
            }
        }
        GoalTeamArg::Invalid(value) => {
            eprintln!("\x1b[33mgoal team: '{value}' — use /goal team on|off\x1b[0m");
        }
    }
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
