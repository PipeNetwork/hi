//! Slash-command handler: act on a parsed `/command` for the line REPL.
//!
//! The async commands that drive a turn or run shell work (`/retry`, `/compact`,
//! `/diff`, `/commit`, `/undo`, `/init`) are handled inline in the REPL loop in
//! [`crate::repl`]; this module covers the synchronous remainder.

use hi_agent::Agent;
use std::path::Path;

/// Act on a slash command. Returns true when the session should quit.
///
/// `config`/`active_profile`/`config_path` let settings that the user expects
/// to persist (e.g. `/config reasoning`) be saved to the active profile's
/// owning config layer. They may be omitted by callers with no config access.
pub(crate) fn handle_command(
    agent: &mut Agent,
    command: hi_agent::Command,
    config: Option<&mut crate::config::Config>,
    active_profile: Option<&str>,
    config_path: Option<&Path>,
    approval_store: Option<&dyn hi_policy::ApprovalStore>,
) -> bool {
    use hi_agent::Command;
    // Nested `/config model|lsp|…` rewrites to the bare top-level command.
    let command = hi_agent::command::resolve_command(command);
    match command {
        Command::Quit => return true,
        Command::Rsi(_) => {
            eprintln!("\x1b[33mRSI recovery command requires an async frontend\x1b[0m")
        }
        Command::Help(arg) => println!("{}", hi_agent::command::help_text_for(&arg)),
        Command::Status => {
            let t = agent.totals();
            let tel = agent.last_turn_telemetry();
            let cost = hi_ai::CostEstimate::from_usage(t, agent.usage_pricing())
                .map(hi_ai::CostEstimate::format_usd)
                .unwrap_or_else(|| "n/a".into());
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
                "\x1b[2mstatus: ready\nexecution: {}\nmodel: {}\nsession usage across all model calls: {} input · {} output · {} total{}\nsession $: {}\nlast turn: user prompt estimate {} · output across all model calls {}{}\ncontext occupancy: {}\ngoal: {}\nverify: {}\nevidence: {} (reads {}, searches {}, listing_only {}, repair nudges {})\ncheckpoints: {}\x1b[0m",
                agent.execution_mode().as_str(),
                agent.model(),
                t.input_tokens,
                t.output_tokens,
                t.total(),
                if t.estimated {
                    " (contains estimates)"
                } else {
                    ""
                },
                cost,
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
        Command::Durable(arg) => {
            let value = arg.trim().to_ascii_lowercase();
            match value.as_str() {
                "" | "status" => println!(
                    "\x1b[2mdurable execution: {}\x1b[0m",
                    agent.execution_mode().as_str()
                ),
                "on" | "enable" | "yes" | "true" => {
                    match agent.set_execution_mode(hi_agent::ExecutionMode::Durable) {
                        Ok(()) => println!(
                            "\x1b[2mdurable execution → on (checkpoints prompts and completed tool batches)\x1b[0m"
                        ),
                        Err(error) => eprintln!("\x1b[33mdurable execution: {error:#}\x1b[0m"),
                    }
                }
                "off" | "disable" | "no" | "false" => {
                    match agent.set_execution_mode(hi_agent::ExecutionMode::Ephemeral) {
                        Ok(()) => println!("\x1b[2mdurable execution → off\x1b[0m"),
                        Err(error) => eprintln!("\x1b[33mdurable execution: {error:#}\x1b[0m"),
                    }
                }
                _ => eprintln!(
                    "\x1b[33musage: /durable [on|off|status] — requires a saved session\x1b[0m"
                ),
            }
        }
        Command::Pipefs(_) => {
            eprintln!("\x1b[33mPipeFS command requires an async frontend\x1b[0m")
        }
        Command::Turns(arg) => handle_turns(agent, hi_agent::command::parse_turns_arg(&arg)),
        // `/doctor` needs async settings/MCP probes; handled inline by REPL/TUI.
        Command::Doctor => {}
        Command::Plan(_)
        | Command::ViewPlan
        | Command::Memory
        | Command::Fork(_)
        | Command::Rewind(_)
        | Command::Permissions(_)
        | Command::AlwaysApprove(_)
        | Command::Auto(_)
        | Command::Queue(_)
        | Command::Tasks(_)
        | Command::Plugins(_)
        | Command::Remember(_)
        | Command::UndoMemory
        | Command::ImportClaude(_)
        | Command::Recap
        | Command::Metrics
        | Command::SynthEvals
        | Command::Find(_)
        | Command::Jump(_)
        | Command::History(_)
        | Command::Hooks(_)
        | Command::Trust(_)
        | Command::Marketplace(_)
        | Command::Worktree(_)
        | Command::Inspect(_)
        | Command::Agents(_)
        | Command::Share(_)
        | Command::McpAdmin(_)
        | Command::RewindPicker
        | Command::ScreenMode(_)
        | Command::VimMode(_)
        | Command::Multiline(_)
        | Command::Timeline(_)
        | Command::Timestamps(_)
        | Command::Cd(_) => {
            if let Some(effect) = hi_agent::handle_session_command(agent, &command, &[]) {
                print!("{}", effect.message);
                if !effect.message.ends_with('\n') {
                    println!();
                }
                if effect.follow_up_prompt.is_some() {
                    println!(
                        "\x1b[2m(the follow-up turn runs automatically in the REPL/TUI; paste the request as a normal message here if needed)\x1b[0m"
                    );
                }
            }
        }
        Command::Rename(arg) => {
            let id = crate::session::local_sessions()
                .into_iter()
                .next()
                .map(|s| s.id);
            match id {
                Some(id) if !arg.trim().is_empty() => {
                    match crate::session::rename_session(&id, &arg) {
                        Ok(name) => println!("\x1b[32m✓ session {id} renamed to {name}\x1b[0m"),
                        Err(error) => eprintln!("\x1b[33mrename failed: {error:#}\x1b[0m"),
                    }
                }
                _ => println!(
                    "\x1b[2musage: /rename <name> (or /sessions rename <id> <name>)\x1b[0m"
                ),
            }
        }
        Command::Resume(arg) => {
            if arg.trim().is_empty() {
                println!("\x1b[2muse /sessions to choose a session, or /resume <id>\x1b[0m");
            } else {
                println!("\x1b[2mresume with: hi --resume {}\x1b[0m", arg.trim());
            }
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
            let path = agent.state_root().join("hi-debug.log");
            match hi_agent::ui::write_private_debug_log(&path, &body) {
                Ok(()) => println!("\x1b[2mwrote redacted debug log: {}\x1b[0m", path.display()),
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
                agent.set_usage_pricing(None);
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
        Command::Team(arg) => {
            let parts: Vec<&str> = arg.split_whitespace().collect();
            match parts.as_slice() {
                [] => {
                    for row in agent.team_roles() {
                        let suffix = if row.inherited { "  (driver)" } else { "" };
                        println!(
                            "\x1b[2m  {:<9} {}  @ {}{}\x1b[0m",
                            row.role, row.model, row.route, suffix
                        );
                    }
                    println!(
                        "\x1b[2m  /team <explore|delegate|editor> <model|local|off> · /team planner <model|off>\x1b[0m"
                    );
                    let supported = hi_agent::local_skeptic::SUPPORTED_LOCAL_MODELS
                        .iter()
                        .map(|entry| entry.name)
                        .collect::<Vec<_>>()
                        .join(", ");
                    println!(
                        "\x1b[2m  local models (auto-download + serve): local (auto-size), {supported}\x1b[0m"
                    );
                }
                ["driver", ..] => {
                    println!(
                        "\x1b[2mthe driver is the session model — switch it with /model or /provider\x1b[0m"
                    );
                }
                ["skeptic", ..] => {
                    println!(
                        "\x1b[2mskeptic routing has dedicated commands: /config skeptic-local on|off or HI_SKEPTIC_ENDPOINT\x1b[0m"
                    );
                }
                ["planner", "off"] => {
                    agent.set_planner_model(None);
                    println!("\x1b[2mplanner → driver model\x1b[0m");
                }
                ["planner", model] => {
                    agent.set_planner_model(Some((*model).to_string()));
                    println!("\x1b[2mplanner → {model}\x1b[0m");
                }
                ["auto"] => {
                    // One command wires the whole team: delegate on the best
                    // verified local model, editor/explore on the fast small
                    // one, skeptic riding the same server.
                    let ram = hi_agent::local_skeptic::system_ram_gb();
                    let backend = hi_agent::local_skeptic::detect_backend();
                    let Some(delegate) =
                        hi_agent::local_skeptic::resolve_team_local_model("auto", ram, backend)
                    else {
                        println!(
                            "\x1b[2mno supported local model fits this machine; roles stay on the driver\x1b[0m"
                        );
                        return false;
                    };
                    let fast = hi_agent::local_skeptic::resolve_team_local_model(
                        "nemotron-4b",
                        ram,
                        backend,
                    )
                    .filter(|fast| {
                        fast.entry.fits(ram, backend) && fast.entry.name != delegate.entry.name
                    })
                    .unwrap_or(delegate);
                    println!(
                        "\x1b[2mauto-setup: delegate → {} · editor/explore → {} · skeptic → local\x1b[0m",
                        delegate.display(),
                        fast.display()
                    );
                    for (role, resolved) in
                        [("delegate", delegate), ("editor", fast), ("explore", fast)]
                    {
                        let reuse = resolved
                            .mlx
                            .and_then(|quant| agent.running_local_model_server(quant.model_id))
                            .or_else(|| {
                                resolved.entry.cuda.and_then(|cuda| {
                                    agent.running_local_model_server(cuda.model_id)
                                })
                            });
                        let provisioned = match reuse {
                            Some((endpoint, model_id)) => Ok((endpoint, model_id, String::new())),
                            None => {
                                println!(
                                    "\x1b[2msetting up {} locally for {role}…\x1b[0m",
                                    resolved.display()
                                );
                                let (phase_tx, _phase_rx) = tokio::sync::watch::channel(
                                    hi_agent::local_skeptic::ProvisionPhase::Resolving,
                                );
                                tokio::task::block_in_place(|| {
                                    tokio::runtime::Handle::current().block_on(
                                        hi_agent::local_skeptic::provision_team_local_model(
                                            resolved, phase_tx,
                                        ),
                                    )
                                })
                            }
                        };
                        match provisioned {
                            Ok((endpoint, model_id, process_id)) => {
                                if !process_id.is_empty() {
                                    agent.register_team_local_server(
                                        endpoint.clone(),
                                        model_id.clone(),
                                        process_id,
                                    );
                                }
                                agent.set_team_route(
                                    role,
                                    Some(model_id.clone()),
                                    Some(endpoint),
                                    None,
                                );
                                println!("\x1b[2m✓ {role} → {model_id} @ local\x1b[0m");
                            }
                            Err(error) => {
                                let reason: String =
                                    format!("{error:#}").chars().take(140).collect();
                                println!(
                                    "\x1b[2mcouldn't set up {} ({reason}); {role} stays on the driver\x1b[0m",
                                    resolved.display()
                                );
                                break;
                            }
                        }
                    }
                    if agent.any_team_local_server().is_some() {
                        let outcome = tokio::task::block_in_place(|| {
                            tokio::runtime::Handle::current()
                                .block_on(agent.enable_local_skeptic(false))
                        });
                        if let Ok(hi_agent::LocalSkepticOutcome::Ready { model_id, .. }) = outcome {
                            println!("\x1b[2m✓ skeptic → {model_id} @ local\x1b[0m");
                        }
                    }
                }
                [role @ ("explore" | "delegate" | "editor"), "off"] => {
                    agent.set_team_route(role, None, None, None);
                    println!("\x1b[2m{role} → driver route (applies to new {role} runs)\x1b[0m");
                }
                [role @ ("explore" | "delegate" | "editor"), model, rest @ ..] => {
                    let explicit_endpoint = rest
                        .first()
                        .filter(|value| value.starts_with("http"))
                        .map(|s| (*s).to_string());
                    if let Some(endpoint) = explicit_endpoint {
                        let key = rest.get(1).map(|s| (*s).to_string());
                        agent.set_team_route(
                            role,
                            Some((*model).to_string()),
                            Some(endpoint.clone()),
                            key,
                        );
                        println!("\x1b[2m{role} → {model} @ {endpoint}\x1b[0m");
                    } else if let Some(resolved) = hi_agent::local_skeptic::resolve_team_local_model(
                        model,
                        hi_agent::local_skeptic::system_ram_gb(),
                        hi_agent::local_skeptic::detect_backend(),
                    ) {
                        // Supported local model: reuse a running server or
                        // provision inline (plain mode has no background UI).
                        let reuse = resolved
                            .mlx
                            .and_then(|quant| agent.running_local_model_server(quant.model_id))
                            .or_else(|| {
                                resolved.entry.cuda.and_then(|cuda| {
                                    agent.running_local_model_server(cuda.model_id)
                                })
                            });
                        let provisioned = match reuse {
                            Some((endpoint, model_id)) => Ok((endpoint, model_id, String::new())),
                            None => {
                                println!(
                                    "\x1b[2msetting up {} locally (first run may download weights — this can take a while)…\x1b[0m",
                                    resolved.display()
                                );
                                // Plain mode has no background UI; block the
                                // (synchronous) command loop on the runtime
                                // and print phase transitions as they land.
                                let (phase_tx, mut phase_rx) = tokio::sync::watch::channel(
                                    hi_agent::local_skeptic::ProvisionPhase::Resolving,
                                );
                                let printer = tokio::spawn(async move {
                                    while phase_rx.changed().await.is_ok() {
                                        let phase = phase_rx.borrow().clone();
                                        println!("\x1b[2m  {phase:?}\x1b[0m");
                                    }
                                });
                                let result = tokio::task::block_in_place(|| {
                                    tokio::runtime::Handle::current().block_on(
                                        hi_agent::local_skeptic::provision_team_local_model(
                                            resolved, phase_tx,
                                        ),
                                    )
                                });
                                printer.abort();
                                result
                            }
                        };
                        match provisioned {
                            Ok((endpoint, model_id, process_id)) => {
                                if !process_id.is_empty() {
                                    agent.register_team_local_server(
                                        endpoint.clone(),
                                        model_id.clone(),
                                        process_id,
                                    );
                                }
                                agent.set_team_route(
                                    role,
                                    Some(model_id.clone()),
                                    Some(endpoint),
                                    None,
                                );
                                println!(
                                    "\x1b[2m✓ {role} → {model_id} @ local (applies to new {role} runs)\x1b[0m"
                                );
                            }
                            Err(error) => {
                                let reason: String =
                                    format!("{error:#}").chars().take(140).collect();
                                println!(
                                    "\x1b[2mcouldn't set up {} locally ({reason}); {role} stays on the driver\x1b[0m",
                                    resolved.display()
                                );
                            }
                        }
                    } else {
                        agent.set_team_route(role, Some((*model).to_string()), None, None);
                        println!(
                            "\x1b[2m{role} → {model} (driver route; applies to new {role} runs)\x1b[0m"
                        );
                    }
                }
                [role @ ("explore" | "delegate" | "editor")] => {
                    // Plain mode has no dropdown; print the catalog with the
                    // same per-machine sizing the TUI picker shows.
                    let ram = hi_agent::local_skeptic::system_ram_gb();
                    println!(
                        "\x1b[2mpick a local model for {role}: /team {role} <name>  (or `auto`, `name@quant`, a cloud model id, or an explicit endpoint URL)\x1b[0m"
                    );
                    for entry in hi_agent::local_skeptic::SUPPORTED_LOCAL_MODELS {
                        let fit = match entry.pick_mlx(ram) {
                            Some(quant) if entry.mlx.len() > 1 => format!(
                                "{} fits (needs {}GB RAM) · quants {}",
                                quant.quant,
                                quant.min_ram_gb,
                                entry.quant_summary()
                            ),
                            Some(quant) => format!("needs {}GB RAM · fits", quant.min_ram_gb),
                            None => format!(
                                "needs {}GB+ RAM · too big for this machine",
                                entry.min_ram_gb(None)
                            ),
                        };
                        println!("\x1b[2m  {:<14} {} · {fit}\x1b[0m", entry.name, entry.label);
                    }
                }
                [role, ..] => {
                    println!(
                        "\x1b[2munknown role '{role}' — roles: driver, explore, delegate, editor, skeptic, planner\x1b[0m"
                    );
                }
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
                    println!("\x1b[2m│ execution:       \x1b[0m {}", s.execution);
                    println!("\x1b[2m│ model:           \x1b[0m {}", s.model);
                    if !s.provider_route.is_empty() {
                        println!("\x1b[2m│ provider:        \x1b[0m {}", s.provider_route);
                    }
                    println!("\x1b[2m│ max-tokens:      \x1b[0m {}", s.max_tokens);
                    println!("\x1b[2m│ thinking-budget: \x1b[0m {}", s.thinking_budget);
                    println!("\x1b[2m│ reasoning:       \x1b[0m {}", s.reasoning_effort);
                    println!("\x1b[2m│ temperature:     \x1b[0m {}", s.temperature);
                    println!("\x1b[2m│ top-p:           \x1b[0m {}", s.top_p);
                    println!(
                        "\x1b[2m│ output-tokens:   \x1b[0m {}",
                        s.output_token_parameter
                    );
                    println!(
                        "\x1b[2m│ trace-capture:   \x1b[0m {}",
                        std::env::var("HI_TRACE_CAPTURE")
                            .ok()
                            .filter(|value| !value.trim().is_empty())
                            .unwrap_or_else(|| "metadata".into())
                    );
                    println!("\x1b[2m│ steps:           \x1b[0m {}", s.max_steps);
                    println!("\x1b[2m│ tool-calls:      \x1b[0m {}", s.max_tool_calls);
                    println!("\x1b[2m│ tool-mode:       \x1b[0m {}", s.tool_mode);
                    println!("\x1b[2m│ compat:          \x1b[0m {}", s.compat);
                    println!("\x1b[2m│ deepseek-compat: \x1b[0m {}", s.deepseek_compat);
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
                    println!(
                        "\x1b[2m│ suggest:         \x1b[0m {}",
                        s.suggest_next_prompt
                    );
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
                        "\x1b[2mset: /config reasoning <minimal|low|medium|high|xhigh|off> · /config temp <0.0-2.0|off> · /config steps <1+|auto|off> · /config moe-streaming <on|off|auto> · /config suggest <on|off> · /config skeptic-local <on|off> · /config rsi [on|off|spend-limit <USD>|channel stable|beta]\x1b[0m"
                    );
                }
                ConfigArg::Reasoning(effort) => {
                    agent.set_reasoning_effort(effort);
                    // Always stick the choice on this machine; also mirror onto
                    // the active profile when one is selected.
                    let saved = persist_reasoning(config, active_profile, config_path, effort);
                    match effort {
                        Some(e) => println!(
                            "\x1b[2mreasoning effort → {} (applies next turn; OpenAI-compatible endpoints only){}\x1b[0m",
                            e.as_str(),
                            saved_note(saved),
                        ),
                        None => println!(
                            "\x1b[2mreasoning effort → off (no reasoning_effort sent; endpoint default){}\x1b[0m",
                            saved_note(saved),
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
                    println!(
                        "\x1b[2mstep limit → unlimited (automatic default; applies next turn)\x1b[0m"
                    );
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
                ConfigArg::SuggestNextPrompt(on) => {
                    agent.set_suggest_next_prompt(on);
                    println!(
                        "\x1b[2msuggest next prompt → {} (applies after the next turn)\x1b[0m",
                        if on { "on" } else { "off" }
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
                // Nested settings are rewritten by `resolve_command` before
                // dispatch; if one still reaches here, treat it as a no-op.
                ConfigArg::Model(_)
                | ConfigArg::Provider(_)
                | ConfigArg::Login(_)
                | ConfigArg::Logout(_)
                | ConfigArg::Verify(_)
                | ConfigArg::Lsp(_)
                | ConfigArg::Delegate(_)
                | ConfigArg::Engine(_)
                | ConfigArg::Theme(_)
                | ConfigArg::Density(_)
                | ConfigArg::Mouse(_) => {}
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
        Command::Goal(arg) => handle_goal_command(agent, arg.trim()),
        Command::Race(_) => {
            println!(
                "\x1b[33m/race is available in the full-screen TUI; use `hi --best-of` for headless candidate runs\x1b[0m"
            );
        }
        // Handled in the repl loop (async / runs a turn); never reach here.
        Command::Prompt(_)
        | Command::Btw(_)
        | Command::Moa(_)
        | Command::Compact(_)
        | Command::Retry
        | Command::Edit
        | Command::Undo
        | Command::Init
        | Command::Learn(_)
        | Command::Skill(_)
        | Command::Diff
        | Command::DiffLab(_)
        | Command::Files
        | Command::Review(_)
        | Command::Commit
        | Command::Hf(_) => {}
        Command::Version => {
            println!("hi {}", hi_agent::VERSION);
        }
        Command::Export(arg) => {
            if agent.pipefs_workspace_active() {
                eprintln!(
                    "\x1b[33m/export is unavailable while PipeFS is active because it writes outside the workspace durability fence\x1b[0m"
                );
                return false;
            }
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
        Command::Local(_) => {
            println!(
                "\x1b[33m/local is available in the full-screen TUI (run hi without --plain)\x1b[0m"
            );
        }
        // `/login` and `/logout` are handled inline by the REPL/TUI: the device
        // flow is async and waits on the user's browser.
        Command::Login(_) | Command::Logout(_) | Command::Auth(_) => {}
        // `/mcp` is handled inline by the REPL/TUI (async + needs settings).
        Command::Mcp(_) => {}
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
        Command::Density(_) => {
            println!(
                "\x1b[33m/density is only available in the full-screen TUI (run hi without --plain)\x1b[0m"
            );
        }
        Command::Mouse(_) => {
            println!(
                "\x1b[33m/mouse is only available in the full-screen TUI (run hi without --plain)\x1b[0m"
            );
        }
        Command::Voice(_) => {
            println!(
                "\x1b[33m/voice is only available in the full-screen TUI (run hi without --plain)\x1b[0m"
            );
        }
        Command::Digest => {
            println!(
                "\x1b[33m/digest is only available in the full-screen TUI (run hi without --plain)\x1b[0m"
            );
        }
        Command::Inbox(arg) => handle_inbox(agent, approval_store, &arg),
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
                "\x1b[33m/fleet is only available in the full-screen TUI (run hi without --plain); /fleet status works here\x1b[0m"
            ),
        },
        Command::Engine(arg) => println!("{}", agent.engine_command(&arg)),
        Command::Workflow(arg) => {
            crate::workflow::handle_workflow_command(&arg);
        }
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
            agent.set_write_subagents(hi_agent::WriteSubagentPolicy::On);
            println!(
                "\x1b[2mdelegate on — offered on every mutation turn. Worktree-isolated; \
                 changes kept only if they verify.\x1b[0m"
            );
        }
        "off" => {
            agent.set_write_subagents(hi_agent::WriteSubagentPolicy::Off);
            println!("\x1b[2mdelegate disabled.\x1b[0m");
        }
        "risk" | "auto" => {
            agent.set_write_subagents(hi_agent::WriteSubagentPolicy::Risk);
            println!(
                "\x1b[2mdelegate risk (default) — offered only for multi-file / isolation-shaped \
                 tasks. `/delegate on` for every mutation.\x1b[0m"
            );
        }
        _ => {
            let state = agent.write_subagents_policy().as_str();
            println!(
                "\x1b[2mdelegate is {state} (default risk). `/delegate on|off|risk` — \
                 worktree-isolated and verify-gated; best for large, independently-verifiable \
                 subtasks.\x1b[0m"
            );
        }
    }
}

/// Sync `/goal …` control surface (status/pause/edit/clear/set without planner).
fn handle_goal_command(agent: &mut hi_agent::Agent, arg: &str) {
    use hi_agent::command::{
        parse_goal_budget, parse_goal_edit, parse_goal_limit, parse_goal_objective_flags,
        parse_goal_team, parse_goal_unattended,
    };

    if let Some(limit) = parse_goal_limit(arg) {
        handle_goal_limit(agent, limit);
        return;
    }
    if let Some(budget) = parse_goal_budget(arg) {
        handle_goal_budget(agent, budget);
        return;
    }
    if let Some(team) = parse_goal_team(arg) {
        handle_goal_team(agent, team);
        return;
    }
    if let Some(unattended) = parse_goal_unattended(arg) {
        handle_goal_unattended(agent, unattended);
        return;
    }
    if let Some(edit) = parse_goal_edit(arg) {
        handle_goal_edit(agent, edit);
        return;
    }
    match arg {
        "" | "status" | "show" => {
            if let Some(g) = agent.structured_goal() {
                println!("goal drive: {}", agent.goal_drive_status());
                print!("{}", g.status_report());
            } else {
                match agent.goal() {
                    Some(goal) => println!("\x1b[2mgoal (transient): {goal}\x1b[0m"),
                    None => println!("\x1b[2mgoal: off (set one with /goal <text>)\x1b[0m"),
                }
            }
        }
        "export" | "view" => match agent.export_goal_plan() {
            Ok(Some(path)) => println!("\x1b[32m✓ wrote {}\x1b[0m", path.display()),
            Ok(None) => println!("\x1b[2mno structured goal to export\x1b[0m"),
            Err(err) => eprintln!("\x1b[33mexport failed: {err:#}\x1b[0m"),
        },
        "clear" | "off" | "none" => {
            let n = agent
                .structured_goal()
                .map(|g| g.sub_goals.len())
                .unwrap_or(0);
            let obj = agent
                .structured_goal()
                .map(|g| g.objective.clone())
                .or_else(|| agent.goal().map(|s| s.to_string()));
            match agent.set_transient_goal(None) {
                Ok(()) => {
                    if let Some(o) = obj {
                        println!("\x1b[32m✓ goal cleared — dropped {n} step(s); was: {o}\x1b[0m");
                    } else {
                        println!("\x1b[32m✓ goal cleared\x1b[0m");
                    }
                }
                Err(err) => eprintln!("\x1b[33mgoal clear failed: {err:#}\x1b[0m"),
            }
        }
        "pause" => match agent.try_set_goal_pause_reason(hi_agent::GoalPauseReason::User) {
            Ok(true) => println!("\x1b[32m✓ goal paused (user) — resume with /goal resume\x1b[0m"),
            Ok(false) => println!("\x1b[2mno goal to pause\x1b[0m"),
            Err(err) => eprintln!("\x1b[33mgoal pause failed: {err:#}\x1b[0m"),
        },
        "resume" | "accept" => {
            let was_review = agent
                .structured_goal()
                .is_some_and(|g| g.pause_reason == hi_agent::GoalPauseReason::Review);
            match agent.try_set_goal_pause_reason(hi_agent::GoalPauseReason::None) {
                Ok(true) => {
                    agent.reset_goal_drive_stall();
                    if was_review || arg == "accept" {
                        println!("\x1b[32m✓ plan accepted — goal driving turns again\x1b[0m");
                    } else {
                        println!("\x1b[32m✓ goal resumed — steering turns again\x1b[0m");
                    }
                    if let Some(g) = agent.structured_goal() {
                        print!("{}", g.status_report());
                    }
                }
                Ok(false) => println!("\x1b[2mno goal to resume\x1b[0m"),
                Err(err) => eprintln!("\x1b[33mgoal resume failed: {err:#}\x1b[0m"),
            }
        }
        goal => {
            let flags = parse_goal_objective_flags(goal);
            if flags.workflow {
                handle_goal_workflow(agent, &flags.text);
                return;
            }
            if flags.review && flags.text.is_empty() {
                eprintln!("\x1b[33musage: /goal --review <objective>\x1b[0m");
                return;
            }
            if flags.unattended && flags.text.is_empty() {
                eprintln!("\x1b[33musage: /goal --unattended <objective>\x1b[0m");
                return;
            }
            let text = if flags.text.is_empty() {
                goal.to_string()
            } else {
                flags.text
            };
            if agent.long_horizon() {
                let structured = agent
                    .try_ingest_goal(&text)
                    .unwrap_or_else(|| hi_agent::Goal::new(text.clone(), vec![text.clone()]));
                match agent.set_structured_goal(Some(structured)) {
                    Ok(true) => {
                        echo_planned_goal(agent, flags.review, flags.unattended);
                    }
                    Ok(false) => match agent.set_transient_goal(Some(text.clone())) {
                        Ok(()) => println!(
                            "\x1b[32m✓ goal set — steers every turn until cleared: {text}\x1b[0m"
                        ),
                        Err(err) => eprintln!("\x1b[33mgoal set failed: {err:#}\x1b[0m"),
                    },
                    Err(err) => eprintln!("\x1b[33mgoal set failed: {err:#}\x1b[0m"),
                }
            } else {
                match agent.set_transient_goal(Some(text.clone())) {
                    Ok(()) => println!(
                        "\x1b[32m✓ goal set — steers every turn until cleared: {text}\x1b[0m"
                    ),
                    Err(err) => eprintln!("\x1b[33mgoal set failed: {err:#}\x1b[0m"),
                }
            }
        }
    }
}

fn handle_goal_edit(agent: &mut hi_agent::Agent, edit: hi_agent::command::GoalEditArg) {
    use hi_agent::command::GoalEditArg;
    match edit {
        GoalEditArg::Invalid(msg) => eprintln!("\x1b[33m{msg}\x1b[0m"),
        GoalEditArg::Objective(text) => {
            match agent.update_structured_goal(|g| {
                g.objective = text.clone();
                g.push_event("edit", format!("objective → {text}"));
            }) {
                Ok(true) => {
                    println!("\x1b[32m✓ goal objective updated\x1b[0m");
                    if let Some(g) = agent.structured_goal() {
                        print!("{}", g.status_report());
                    }
                }
                Ok(false) => println!("\x1b[2mno structured goal to edit\x1b[0m"),
                Err(err) => eprintln!("\x1b[33medit failed: {err:#}\x1b[0m"),
            }
        }
        GoalEditArg::Step { index, text } => {
            match agent.update_structured_goal(|g| {
                if let Some(sg) = g.sub_goals.get_mut(index - 1) {
                    sg.description = text.clone();
                    g.push_event("edit", format!("step {index} → {text}"));
                }
            }) {
                Ok(true) => {
                    if agent
                        .structured_goal()
                        .is_some_and(|g| g.sub_goals.len() >= index)
                    {
                        println!("\x1b[32m✓ goal step {index} updated\x1b[0m");
                    } else {
                        println!("\x1b[33mno step {index}\x1b[0m");
                    }
                }
                Ok(false) => println!("\x1b[2mno structured goal to edit\x1b[0m"),
                Err(err) => eprintln!("\x1b[33medit failed: {err:#}\x1b[0m"),
            }
        }
    }
}

/// `/goal <objective>` with a planner (the async path driven from the repl):
/// decompose the objective into sub-goals via one bounded planner call, install the
/// structured goal, and echo the checklist. Falls back to a single sub-goal on
/// failure so `/goal` always sets *something*.
pub(crate) async fn handle_goal_planned(agent: &mut hi_agent::Agent, objective: &str) {
    let flags = hi_agent::command::parse_goal_objective_flags(objective);
    if flags.workflow {
        handle_goal_workflow(agent, &flags.text);
        return;
    }
    let objective = if flags.text.is_empty() {
        objective
    } else {
        flags.text.as_str()
    };
    let sub_goals = if let Some(goal) = agent.try_ingest_goal(objective) {
        match agent.set_structured_goal(Some(goal)) {
            Ok(true) => {
                echo_planned_goal(agent, flags.review, flags.unattended);
                return;
            }
            Ok(false) => {
                echo_transient_goal(agent, objective);
                return;
            }
            Err(err) => {
                eprintln!("\x1b[33mgoal set failed: {err:#}\x1b[0m");
                return;
            }
        }
    } else {
        println!("\x1b[2mplanning goal with the planner model…\x1b[0m");
        match agent.decompose_goal(objective).await {
            Ok(steps) if !steps.is_empty() => steps,
            Ok(_) => vec![objective.to_string()],
            Err(err) => {
                println!(
                    "\x1b[2mplanner unavailable ({err:#}); using the objective as one step\x1b[0m"
                );
                vec![objective.to_string()]
            }
        }
    };
    match agent.set_structured_goal(Some(hi_agent::Goal::new(objective.to_string(), sub_goals))) {
        Ok(true) => echo_planned_goal(agent, flags.review, flags.unattended),
        Ok(false) => echo_transient_goal(agent, objective),
        Err(err) => eprintln!("\x1b[33mgoal set failed: {err:#}\x1b[0m"),
    }
}

fn handle_goal_workflow(agent: &hi_agent::Agent, objective: &str) {
    match hi_agent::goal_workflow_plan_path(false, agent.workspace_root(), objective) {
        Ok(path) => match std::env::current_exe() {
            Ok(exe) => match crate::workflow_cmd::spawn_detached_workflow_run(
                &exe,
                &path,
                agent.max_steps_limit(),
                agent.max_tool_calls_cap(),
                agent.max_verify_repairs_cap(),
            ) {
                Ok((pid, log)) => {
                    println!(
                        "\x1b[32m▶ workflow {path} started (pid {pid})\x1b[0m\n\x1b[2m  log: {} — it checkpoints every wave and survives this session\x1b[0m",
                        log.display()
                    );
                }
                Err(err) => eprintln!("\x1b[33mworkflow start failed: {err:#}\x1b[0m"),
            },
            Err(err) => eprintln!("\x1b[33mcannot resolve hi executable: {err:#}\x1b[0m"),
        },
        Err(err) => eprintln!("\x1b[33m{err}\x1b[0m"),
    }
}

fn echo_planned_goal(agent: &mut hi_agent::Agent, review: bool, unattended: bool) {
    let review_persisted = review
        && match agent.try_set_goal_pause_reason(hi_agent::GoalPauseReason::Review) {
            Ok(true) => true,
            Ok(false) => false,
            Err(err) => {
                eprintln!("\x1b[33mgoal review mode failed: {err:#}\x1b[0m");
                false
            }
        };
    if unattended {
        match agent.try_set_goal_unattended(true) {
            Ok(true) => println!("\x1b[33m{}\x1b[0m", hi_agent::UNATTENDED_DRIVE_WARNING),
            Ok(false) => {}
            Err(err) => eprintln!("\x1b[33mgoal unattended failed: {err:#}\x1b[0m"),
        }
    }
    if review_persisted {
        println!(
            "\x1b[32m✓ long-horizon goal planned (review) — inspect, then /goal accept:\x1b[0m"
        );
    } else {
        println!("\x1b[32m✓ long-horizon goal set — driving sub-goals:\x1b[0m");
    }
    if let Some(g) = agent.structured_goal() {
        for warning in g.actionability_warnings() {
            println!("\x1b[33m  {warning} (driving in-session anyway)\x1b[0m");
        }
        print!("{}", g.status_report());
        if let Ok(path) = g.export_markdown_to(agent.workspace_root()) {
            println!("\x1b[2m  snapshot: {}\x1b[0m", path.display());
        }
    }
}

fn echo_transient_goal(agent: &mut hi_agent::Agent, objective: &str) {
    match agent.set_transient_goal(Some(objective.to_string())) {
        Ok(()) => {
            println!("\x1b[32m✓ goal set — steers every turn until cleared: {objective}\x1b[0m")
        }
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

fn handle_turns(agent: &mut hi_agent::Agent, arg: hi_agent::command::TurnsArg) {
    use hi_agent::command::TurnsArg;
    match arg {
        TurnsArg::Show => match agent.max_turns() {
            Some(n) => println!(
                "\x1b[2mturns: {}/{n} (limit {n})\x1b[0m",
                agent.turn_count()
            ),
            None => println!(
                "\x1b[2mturns: {} (no limit — use /turns <n> to set one)\x1b[0m",
                agent.turn_count()
            ),
        },
        TurnsArg::Set(n) => {
            agent.set_max_turns(Some(n));
            println!("\x1b[32m✓ turn limit set to {n}\x1b[0m");
        }
        TurnsArg::Unlimited => {
            agent.set_max_turns(None);
            println!("\x1b[32m✓ turn limit removed — unlimited turns\x1b[0m");
        }
        TurnsArg::Invalid(value) => {
            eprintln!(
                "\x1b[33mturns: '{value}' isn't a number — use /turns <n> or 'turns off'\x1b[0m"
            );
        }
    }
}

fn handle_goal_limit(agent: &mut hi_agent::Agent, limit: hi_agent::command::GoalLimitArg) {
    use hi_agent::command::GoalLimitArg;
    match limit {
        GoalLimitArg::Show => match agent.structured_goal().and_then(|g| g.step_limit) {
            Some(n) => println!("\x1b[2mgoal limit: {n} sub-goals\x1b[0m"),
            None => println!("\x1b[2mgoal limit: none — the plan grows freely\x1b[0m"),
        },
        GoalLimitArg::Set(n) => match agent.try_set_goal_step_limit(Some(n)) {
            Ok(true) => println!("\x1b[32m✓ goal limit set to {n} sub-goals\x1b[0m"),
            Ok(false) => println!("\x1b[2mno goal to limit\x1b[0m"),
            Err(err) => eprintln!("\x1b[33mgoal limit failed: {err:#}\x1b[0m"),
        },
        GoalLimitArg::Unlimited => match agent.try_set_goal_step_limit(None) {
            Ok(true) => println!("\x1b[32m✓ goal limit removed — the plan grows freely\x1b[0m"),
            Ok(false) => println!("\x1b[2mno goal to limit\x1b[0m"),
            Err(err) => eprintln!("\x1b[33mgoal limit failed: {err:#}\x1b[0m"),
        },
        GoalLimitArg::Invalid(value) => {
            eprintln!(
                "\x1b[33mgoal limit: '{value}' isn't a number — use /goal limit <n> or 'limit off'\x1b[0m"
            );
        }
    }
}

fn handle_goal_budget(agent: &mut hi_agent::Agent, budget: hi_agent::command::GoalBudgetArg) {
    use hi_agent::command::GoalBudgetArg;
    match budget {
        GoalBudgetArg::Show => match agent.structured_goal() {
            Some(goal) => match (goal.turn_budget, goal.turns_remaining()) {
                (Some(budget), Some(left)) => println!(
                    "\x1b[2mgoal budget: {budget} turns · {} spent · {left} left\x1b[0m",
                    goal.turns_spent
                ),
                _ => println!(
                    "\x1b[2mgoal budget: none — runs until done ({} turns so far)\x1b[0m",
                    goal.turns_spent
                ),
            },
            None => println!("\x1b[2mno goal set\x1b[0m"),
        },
        GoalBudgetArg::Set(n) => match agent.try_set_goal_turn_budget(Some(n)) {
            Ok(true) => println!(
                "\x1b[32m✓ goal budget set to {n} drive turns — it will park and report\x1b[0m"
            ),
            Ok(false) => println!("\x1b[2mno goal to budget\x1b[0m"),
            Err(err) => eprintln!("\x1b[33mgoal budget failed: {err:#}\x1b[0m"),
        },
        GoalBudgetArg::Unlimited => match agent.try_set_goal_turn_budget(None) {
            Ok(true) => println!("\x1b[32m✓ goal budget removed — it runs until done\x1b[0m"),
            Ok(false) => println!("\x1b[2mno goal to budget\x1b[0m"),
            Err(err) => eprintln!("\x1b[33mgoal budget failed: {err:#}\x1b[0m"),
        },
        GoalBudgetArg::Invalid(value) => eprintln!(
            "\x1b[33mgoal budget: '{value}' isn't a turn count — use /goal budget <n> or 'budget off'\x1b[0m"
        ),
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
        GoalTeamArg::On => match agent.try_set_goal_team(true) {
            Ok(true) => println!(
                "\x1b[32m✓ goal team on — {} reviews each turn before it advances a sub-goal\x1b[0m",
                agent.effective_skeptic_model()
            ),
            Ok(false) => println!("\x1b[2mno active goal — set one with /goal <text> first\x1b[0m"),
            Err(err) => eprintln!("\x1b[33mgoal team update failed: {err:#}\x1b[0m"),
        },
        GoalTeamArg::Off => match agent.try_set_goal_team(false) {
            Ok(true) => println!("\x1b[32m✓ goal team off — single-agent driving\x1b[0m"),
            Ok(false) => println!("\x1b[2mno active goal\x1b[0m"),
            Err(err) => eprintln!("\x1b[33mgoal team update failed: {err:#}\x1b[0m"),
        },
        GoalTeamArg::Invalid(value) => {
            eprintln!("\x1b[33mgoal team: '{value}' — use /goal team on|off\x1b[0m");
        }
    }
}

fn handle_goal_unattended(agent: &mut hi_agent::Agent, arg: hi_agent::command::GoalUnattendedArg) {
    use hi_agent::command::GoalUnattendedArg;
    match arg {
        GoalUnattendedArg::Show => match agent.structured_goal() {
            Some(g) if g.unattended => {
                println!(
                    "\x1b[2mgoal unattended: on — confirms park in /inbox (Ask/Auto stays)\x1b[0m"
                )
            }
            Some(_) => {
                println!("\x1b[2mgoal unattended: off — enable with /goal unattended on\x1b[0m")
            }
            None => println!("\x1b[2mno active goal — set one with /goal <text> first\x1b[0m"),
        },
        GoalUnattendedArg::On => match agent.try_set_goal_unattended(true) {
            Ok(true) => {
                println!("\x1b[32m✓ goal unattended on\x1b[0m");
                println!("\x1b[33m{}\x1b[0m", hi_agent::UNATTENDED_DRIVE_WARNING);
            }
            Ok(false) => println!("\x1b[2mno active goal — set one with /goal <text> first\x1b[0m"),
            Err(err) => eprintln!("\x1b[33mgoal unattended failed: {err:#}\x1b[0m"),
        },
        GoalUnattendedArg::Off => match agent.try_set_goal_unattended(false) {
            Ok(true) => println!(
                "\x1b[32m✓ goal unattended off — Goal drive uses the session permission mode\x1b[0m"
            ),
            Ok(false) => println!("\x1b[2mno active goal\x1b[0m"),
            Err(err) => eprintln!("\x1b[33mgoal unattended failed: {err:#}\x1b[0m"),
        },
        GoalUnattendedArg::Invalid(value) => {
            eprintln!("\x1b[33mgoal unattended: '{value}' — use /goal unattended on|off\x1b[0m");
        }
    }
}

pub(crate) fn handle_inbox(
    agent: &mut hi_agent::Agent,
    store: Option<&dyn hi_policy::ApprovalStore>,
    arg: &str,
) {
    let Some(store) = store else {
        println!("\x1b[33minbox unavailable (no approval store)\x1b[0m");
        return;
    };
    let action = hi_agent::apply_inbox(store, arg);
    for line in &action.lines {
        println!("{line}");
    }
    if action.resume_goal {
        hi_agent::resume_goal_after_inbox(agent);
    }
    if let Some(id) = action.resume_loop
        && let Some(path) = crate::session::loops_file()
    {
        let _ = hi_tui::set_loop_paused(&path, id, false);
    }
}

pub(crate) fn run_inbox_argv(
    store: Option<&dyn hi_policy::ApprovalStore>,
    args: &[String],
) -> anyhow::Result<()> {
    let arg = args.join(" ");
    let Some(store) = store else {
        anyhow::bail!("inbox unavailable (no approval store)");
    };
    let action = hi_agent::apply_inbox(store, &arg);
    for line in &action.lines {
        println!("{line}");
    }
    if let Some(id) = action.resume_loop
        && let Some(path) = crate::session::loops_file()
    {
        let _ = hi_tui::set_loop_paused(&path, id, false);
    }
    Ok(())
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

/// Persist `reasoning_effort` machine-wide (and to the active profile when one
/// exists). Returns `None` only when there is no config context at all.
fn persist_reasoning(
    config: Option<&mut crate::config::Config>,
    active_profile: Option<&str>,
    config_path: Option<&Path>,
    effort: Option<hi_ai::ReasoningEffort>,
) -> Option<anyhow::Result<bool>> {
    let config = config?;
    Some(crate::config::persist_reasoning_effort(
        config,
        active_profile,
        effort,
        config_path,
    ))
}

/// Render a parenthetical "saved …" / error suffix for the reasoning
/// confirmation line, or an empty string when nothing was persisted.
fn saved_note(saved: Option<anyhow::Result<bool>>) -> String {
    match saved {
        None => String::new(),
        Some(Ok(true)) => String::from(" · saved for this computer and profile"),
        Some(Ok(false)) => String::from(" · saved for this computer"),
        Some(Err(e)) => format!(" · couldn't save: {e:#}"),
    }
}

#[cfg(test)]
mod tests {
    use super::handle_command;
    use hi_agent::{AgentConfig, AgentPaths, Command, Goal};
    use std::sync::Arc;

    #[test]
    #[allow(clippy::field_reassign_with_default)] // test assembles config field-by-field for clarity
    fn cli_goal_budget_is_a_control_command_not_a_new_objective() {
        let root = std::env::temp_dir().join(format!(
            "hi-cli-goal-budget-{}-{}",
            std::process::id(),
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&root).expect("workspace");
        let mut config = AgentConfig::default();
        config.paths = AgentPaths {
            workspace_root: root.clone(),
            state_root: root.join(".hi-state"),
        };
        config.subagents.long_horizon = true;
        let provider = Arc::new(hi_ai::OpenAiProvider::new(
            "http://127.0.0.1:1/v1".into(),
            "test".into(),
        ));
        let mut agent = hi_agent::Agent::new(provider, config).expect("agent");
        agent
            .set_structured_goal(Some(Goal::new("ship it", vec!["implement it".into()])))
            .expect("goal accepted");

        handle_command(
            &mut agent,
            Command::Goal("budget 7".into()),
            None,
            None,
            None,
            None,
        );

        let goal = agent.structured_goal().expect("structured goal remains");
        assert_eq!(goal.objective, "ship it");
        assert_eq!(goal.turn_budget, Some(7));
        assert!(!goal.budget_auto);

        let _ = std::fs::remove_dir_all(root);
    }
}
