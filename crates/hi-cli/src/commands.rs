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
        Command::Rsi(_) => {
            eprintln!("\x1b[33mRSI recovery command requires an async frontend\x1b[0m")
        }
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
                    format!(
                        "{}{pct}% of {}k",
                        if agent.last_turn_usage().estimated {
                            "~"
                        } else {
                            ""
                        },
                        w / 1000
                    )
                })
                .unwrap_or_else(|| "unknown".into());
            println!(
                "\x1b[2mstatus: ready\nmodel: {}\nsession usage across all model calls: {} input · {} output · {} total{}\nlast turn: user prompt estimate {} · output across all model calls {}{}\ncontext occupancy: {}\ngoal: {}\nverify: {}\nevidence: {} (reads {}, searches {}, listing_only {}, repair nudges {})\ncheckpoints: {}\x1b[0m",
                agent.model(),
                t.input_tokens,
                t.output_tokens,
                t.total(),
                if t.estimated {
                    " (contains estimates)"
                } else {
                    ""
                },
                agent.last_user_prompt_tokens(),
                if agent.last_turn_usage().estimated {
                    "~"
                } else {
                    ""
                },
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
                "# hi debug log (redacted; best-effort secret detection)\n\nmodel: {}\nsession usage across all model calls: {} input · {} output · {} total{}\nlast turn: user prompt estimate {} · output across all model calls {}{}\ngoal: {}\nverify: {}\nlast_error: none\ncheckpoints: {}\n",
                agent.model(),
                t.input_tokens,
                t.output_tokens,
                t.total(),
                if t.estimated {
                    " (contains estimates)"
                } else {
                    ""
                },
                agent.last_user_prompt_tokens(),
                if agent.last_turn_usage().estimated {
                    "~"
                } else {
                    ""
                },
                agent.last_turn_usage().output_tokens,
                agent.goal_summary(),
                agent.verify_summary(),
                agent.checkpoint_count(),
            );
            match hi_agent::ui::write_private_debug_log(
                std::path::Path::new(".hi-debug.log"),
                &body,
            ) {
                Ok(()) => println!("\x1b[2mwrote redacted debug log: .hi-debug.log\x1b[0m"),
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
                    let s = agent.config_snapshot();
                    // Box border + field labels stay dim; values reset to normal
                    // intensity so the actual settings are readable (not gray).
                    println!(
                        "\x1b[2m╭─ config ───────────────────────────────────────────╮\x1b[0m"
                    );
                    println!("\x1b[2m│ model:           \x1b[0m {}", s.model);
                    if !s.provider_route.is_empty() {
                        println!("\x1b[2m│ provider:        \x1b[0m {}", s.provider_route);
                    }
                    println!("\x1b[2m│ max-tokens:      \x1b[0m {}", s.max_tokens);
                    println!("\x1b[2m│ thinking-budget: \x1b[0m {}", s.thinking_budget);
                    println!("\x1b[2m│ reasoning:       \x1b[0m {}", s.reasoning_effort);
                    println!("\x1b[2m│ temperature:     \x1b[0m {}", s.temperature);
                    println!("\x1b[2m│ steps:           \x1b[0m {}", s.max_steps);
                    println!("\x1b[2m│ tool-mode:       \x1b[0m {}", s.tool_mode);
                    println!("\x1b[2m│ compat:          \x1b[0m {}", s.compat);
                    println!("\x1b[2m│ verify:          \x1b[0m {}", s.verify);
                    println!("\x1b[2m│ review:          \x1b[0m {}", s.review);
                    println!("\x1b[2m│ lsp:             \x1b[0m {}", s.lsp);
                    println!("\x1b[2m│ tool-set:        \x1b[0m {}", s.tool_set);
                    println!("\x1b[2m│ auto-compact:    \x1b[0m {}", s.auto_compact);
                    println!("\x1b[2m│ proactive-verify:\x1b[0m {}", s.proactive_verify);
                    println!(
                        "\x1b[2m│ read-only-preflight:\x1b[0m {}",
                        s.read_only_preflight
                    );
                    println!("\x1b[2m│ long-horizon:    \x1b[0m {}", s.long_horizon);
                    println!("\x1b[2m│ confirm-edits:   \x1b[0m {}", s.confirm_edits);
                    println!("\x1b[2m│ curate-skills:   \x1b[0m {}", s.curate_skills);
                    println!("\x1b[2m│ explore-subagents:\x1b[0m {}", s.explore_subagents);
                    println!("\x1b[2m│ write-subagents: \x1b[0m {}", s.write_subagents);
                    println!("\x1b[2m│ planner-model:   \x1b[0m {}", s.planner_model);
                    println!("\x1b[2m│ skeptic-model:   \x1b[0m {}", s.skeptic_model);
                    println!("\x1b[2m│ moe-streaming:   \x1b[0m {}", s.moe_streaming);
                    let (rsi_requested, rsi_mode, rsi_latest) = agent.rsi_status();
                    let rsi_latest =
                        rsi_latest.map_or("none", |value| if value { "yes" } else { "no" });
                    println!("\x1b[2m│ RSI requested:   \x1b[0m {rsi_requested}");
                    println!("\x1b[2m│ RSI active mode: \x1b[0m {rsi_mode}");
                    println!("\x1b[2m│ RSI channel:     \x1b[0m {}", agent.rsi_channel());
                    let rsi_spend = agent
                        .rsi_maximum_cost_microusd()
                        .map(hi_agent::command::format_usd_micros)
                        .unwrap_or_else(|| "unavailable".to_string());
                    println!("\x1b[2m│ RSI spend limit:\x1b[0m {rsi_spend} per run");
                    println!("\x1b[2m│ RSI latest observed:\x1b[0m {rsi_latest}");
                    println!(
                        "\x1b[2m╰────────────────────────────────────────────────────╯\x1b[0m"
                    );
                    println!(
                        "\x1b[2mset: /config reasoning <minimal|low|medium|high|xhigh|off> · /config temp <0.0-2.0|off> · /config steps <1+|auto|off> · /config moe-streaming <on|off|auto> · /config skeptic-local <on|off> · /config rsi [on|off|spend-limit <USD>|channel stable|beta]\x1b[0m"
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
                ConfigArg::MaxSteps(limit) => {
                    agent.set_max_steps_limit(limit);
                    match limit {
                        Some(limit) => {
                            println!("\x1b[2mstep limit → {limit} (applies next turn)\x1b[0m")
                        }
                        None => println!("\x1b[2mstep limit → off (applies next turn)\x1b[0m"),
                    }
                }
                ConfigArg::MaxStepsAuto => {
                    agent.set_max_steps_auto();
                    println!("\x1b[2mstep limit → auto (intent-aware; applies next turn)\x1b[0m");
                }
                ConfigArg::MoeStreaming(mode) => {
                    // Set the env var that the MLX backend reads at model load
                    // time. Takes effect on the next model load (not the current
                    // session's already-loaded model).
                    let env = "HI_MLX_EXPERT_STREAMING";
                    match mode {
                        hi_agent::command::MoeStreamingMode::On => {
                            // SAFETY: single-threaded CLI REPL.
                            unsafe { std::env::set_var(env, "1") };
                            println!(
                                "\x1b[2mMoE streaming → on (applies next model load; MLX backend)\x1b[0m"
                            );
                        }
                        hi_agent::command::MoeStreamingMode::Off => {
                            // SAFETY: single-threaded CLI REPL.
                            unsafe { std::env::set_var(env, "0") };
                            println!(
                                "\x1b[2mMoE streaming → off / resident (applies next model load; MLX backend)\x1b[0m"
                            );
                        }
                        hi_agent::command::MoeStreamingMode::Auto => {
                            // SAFETY: single-threaded CLI REPL.
                            unsafe { std::env::remove_var(env) };
                            println!(
                                "\x1b[2mMoE streaming → auto (applies next model load; streams when model exceeds memory budget)\x1b[0m"
                            );
                        }
                    }
                }
                ConfigArg::SkepticLocal(_) => {
                    // Routed through the async `handle_skeptic_local` from the
                    // REPL loop; only reachable if `/config skeptic-local` is
                    // dispatched outside it.
                    eprintln!(
                        "\x1b[33m/config skeptic-local must be run from the interactive prompt\x1b[0m"
                    );
                }
                ConfigArg::RsiShow => print_rsi_config(agent),
                ConfigArg::Rsi(enabled) => match agent.set_rsi_enabled(enabled) {
                    Ok(()) if enabled => println!(
                        "\x1b[33mRSI candidate channel → on (applies next turn). Repository/context upload and 30-day operational retention apply; training remains off.\x1b[0m"
                    ),
                    Ok(()) => println!("\x1b[2mRSI candidate channel → off\x1b[0m"),
                    Err(error) => eprintln!("\x1b[33mRSI config error: {error}\x1b[0m"),
                },
                ConfigArg::RsiSpendLimit(value) => {
                    match agent.set_rsi_maximum_cost_microusd(value) {
                        Ok(()) => println!(
                            "\x1b[2mRSI spend limit → {} per run (saved)\x1b[0m",
                            hi_agent::command::format_usd_micros(value)
                        ),
                        Err(error) => eprintln!("\x1b[33mRSI config error: {error}\x1b[0m"),
                    }
                }
                ConfigArg::RsiChannel(channel) => match agent.set_rsi_channel(channel) {
                    Ok(()) => println!("\x1b[2mRSI channel → {} (saved)\x1b[0m", channel.as_str()),
                    Err(error) => eprintln!("\x1b[33mRSI config error: {error}\x1b[0m"),
                },
                ConfigArg::Invalid(m) => eprintln!("\x1b[33m{m}\x1b[0m"),
            }
        }
        Command::Verify(arg) => match arg.trim() {
            "" if agent.verify_is_on() => {
                println!("\x1b[2mverify: {}\x1b[0m", agent.verify_summary())
            }
            "" => println!("\x1b[2mverify: off (set one with /verify <cmd>)\x1b[0m"),
            "off" | "none" | "clear" | "disable" => {
                if let Err(error) = agent.set_verify_command(None) {
                    eprintln!("\x1b[33mverification config error: {error}\x1b[0m");
                    return false;
                }
                println!("\x1b[2mverification disabled\x1b[0m");
            }
            cmd => {
                if let Err(error) = agent.set_verify_command(Some(cmd.to_string())) {
                    eprintln!("\x1b[33mverification config error: {error}\x1b[0m");
                    return false;
                }
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
        | Command::Files
        | Command::Review(_)
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
        Command::Theme(_) => {
            println!(
                "\x1b[33m/theme is only available in the full-screen TUI (run hi without --plain)\x1b[0m"
            );
        }
        Command::Mouse(_) => {
            println!(
                "\x1b[33m/mouse is only available in the full-screen TUI (run hi without --plain)\x1b[0m"
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

pub(crate) async fn handle_rsi_config(agent: &mut Agent, arg: hi_agent::command::ConfigArg) {
    match arg {
        hi_agent::command::ConfigArg::RsiShow => match agent.rsi_public_status().await {
            Ok(status) => println!("{status}"),
            Err(error) => {
                print_rsi_config(agent);
                eprintln!("\x1b[33mRSI status unavailable: {error:#}\x1b[0m");
            }
        },
        hi_agent::command::ConfigArg::Rsi(enabled) => {
            match agent.set_rsi_enabled_validated(enabled).await {
                Ok(()) if enabled => println!(
                    "\x1b[33mRSI candidate channel → on (saved). You confirmed repository/context upload, 30-day operational evidence retention, and training off without separate consent.\x1b[0m"
                ),
                Ok(()) => println!("\x1b[2mRSI candidate channel → off (saved)\x1b[0m"),
                Err(error) => eprintln!("\x1b[33mRSI config error: {error:#}\x1b[0m"),
            }
        }
        hi_agent::command::ConfigArg::RsiSpendLimit(value) => {
            match agent.set_rsi_maximum_cost_microusd(value) {
                Ok(()) => println!(
                    "\x1b[2mRSI spend limit → {} per run (saved)\x1b[0m",
                    hi_agent::command::format_usd_micros(value)
                ),
                Err(error) => eprintln!("\x1b[33mRSI config error: {error:#}\x1b[0m"),
            }
        }
        hi_agent::command::ConfigArg::RsiChannel(channel) => match agent.set_rsi_channel(channel) {
            Ok(()) => println!("\x1b[2mRSI channel → {} (saved)\x1b[0m", channel.as_str()),
            Err(error) => eprintln!("\x1b[33mRSI config error: {error:#}\x1b[0m"),
        },
        _ => unreachable!("only RSI config arguments are routed here"),
    }
}

fn print_rsi_config(agent: &Agent) {
    let (requested, mode, _) = agent.rsi_status();
    let spend = agent
        .rsi_maximum_cost_microusd()
        .map(hi_agent::command::format_usd_micros)
        .unwrap_or_else(|| "unavailable".to_string());
    let channel = agent.rsi_channel();
    println!(
        "\x1b[2mRSI candidate channel: {requested} · mode {mode} · channel {channel} · spend limit {spend}/run · gateway https://api.pipenetwork.ai\x1b[0m"
    );
    println!(
        "\x1b[2mset with /config rsi on|off, /config rsi spend-limit <USD>, or /config rsi channel stable|beta\x1b[0m"
    );
}

pub(crate) async fn handle_rsi_command(agent: &Agent, argument: &str) {
    match agent.rsi_command(argument).await {
        Ok(output) => println!("{output}"),
        Err(error) => eprintln!("\x1b[33mRSI command error: {error:#}\x1b[0m"),
    }
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

/// Async handler for `/config skeptic-local <on|off>`. Turning it on detects the
/// machine's local backend, downloads a small review model if needed (progress
/// prints to the terminal), spawns a `hi-local` server, and routes the `/goal`
/// skeptic review to it. Every failure degrades gracefully to the main model.
pub(crate) async fn handle_skeptic_local(agent: &mut Agent, arg: &str) {
    use hi_agent::command::{ConfigArg, parse_config_arg};
    let on = match parse_config_arg(arg) {
        ConfigArg::SkepticLocal(on) => on,
        _ => return,
    };
    if on {
        println!(
            "\x1b[2mlocal skeptic: detecting backend… (first run downloads a small review model)\x1b[0m"
        );
        match agent.enable_local_skeptic(true).await {
            Ok(hi_agent::LocalSkepticOutcome::Ready { endpoint, model_id }) => println!(
                "\x1b[32m✓ local skeptic on\x1b[0m \x1b[2m→ {model_id} at {endpoint} (used for /goal team reviews)\x1b[0m"
            ),
            Ok(hi_agent::LocalSkepticOutcome::NoBackend) => eprintln!(
                "\x1b[33mno local backend found — needs Apple-Silicon MLX or an NVIDIA GPU. Skeptic stays on the main model.\x1b[0m"
            ),
            Ok(hi_agent::LocalSkepticOutcome::NeedsDownload { repo, dir }) => println!(
                "\x1b[2mmodel {repo} isn't cached — fetch it into {} first, then retry\x1b[0m",
                dir.display()
            ),
            Err(err) => eprintln!(
                "\x1b[33mcouldn't start local skeptic: {err:#}\nSkeptic stays on the main model.\x1b[0m"
            ),
        }
    } else if agent.disable_local_skeptic() {
        println!("\x1b[2mlocal skeptic off — skeptic review back on the main model\x1b[0m");
    } else {
        println!("\x1b[2mlocal skeptic was not on\x1b[0m");
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
                "\x1b[2mgoal team: on — skeptic reviews each advance ({} objection(s), {} unavailable; last: {})\x1b[0m",
                g.skeptic_objections,
                g.skeptic_unavailable,
                g.last_skeptic_status
                    .map(|s| format!("{s:?}"))
                    .unwrap_or_else(|| "not run".into())
            ),
            Some(_) => println!("\x1b[2mgoal team: off — enable with /goal team on\x1b[0m"),
            None => println!("\x1b[2mno active goal — set one with /goal <text> first\x1b[0m"),
        },
        GoalTeamArg::On => {
            if agent.set_goal_team(true) {
                println!(
                    "\x1b[32m✓ goal team on — {} reviews each turn before it advances a sub-goal\x1b[0m",
                    agent.effective_skeptic_model()
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
            let report = agent.lsp_status_report();
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
