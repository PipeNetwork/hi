//! The plain line REPL loop and the animated-spinner turn driver.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use anyhow::{Context, Result, bail};
use hi_agent::Agent;

use crate::commands::handle_command;
use crate::config::{self, Settings};
use crate::goal_drive::pending_drive_prompt;
use crate::provider::provider_label;
use crate::session;
use crate::ui::PlainUi;

const SPINNER: [&str; 10] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

#[allow(clippy::too_many_arguments)]
pub(crate) async fn repl(
    agent: &mut Agent,
    settings: &Settings,
    config: &mut config::Config,
    auto_memory: bool,
    mut active_profile: Option<String>,
    config_path: Option<PathBuf>,
    after_turn: Option<Arc<dyn Fn() + Send + Sync>>,
    approval_store: Option<Arc<dyn hi_policy::ApprovalStore>>,
) -> Result<()> {
    agent.set_interactive_session(true);
    use hi_agent::Command;
    use hi_agent::CompactionKind;
    use rustyline::Editor;
    use rustyline::error::ReadlineError;
    use rustyline::history::DefaultHistory;

    use crate::complete::{ProfileNames, ReplHelper};

    let window = agent
        .context_window()
        .map(|w| format!(" · {}k ctx", w / 1000))
        .unwrap_or_default();
    // Track the live provider label so exit can snapshot it even after /provider.
    let mut active_provider_label = provider_label(settings.provider).to_string();
    println!(
        "hi · {} · {}{} — /help for commands, Ctrl-D to quit.",
        active_provider_label, settings.model, window,
    );

    // Shared, mutable profile-name list the completer reads. We refresh it
    // before each readline so add/edit changes are visible immediately.
    let profiles: ProfileNames =
        std::rc::Rc::new(std::cell::RefCell::new(config::profile_names(config)));
    let helper = ReplHelper::new(hi_agent::command::COMMANDS, profiles.clone());
    let mut editor =
        Editor::<ReplHelper, DefaultHistory>::with_config(rustyline::Config::default())
            .context("initializing line editor")?;
    editor.set_helper(Some(helper));
    let history = session::history_path();
    if let Some(path) = &history {
        let _ = editor.load_history(path);
    }

    // For `/retry`: the last message sent, and the history length just before
    // that turn (so we can drop it before re-running).
    let mut last_prompt: Option<String> = None;
    let mut last_turn_start = 0usize;
    let mut last_turn_snapshot: Option<hi_agent::AgentStateSnapshot> = None;
    let mut hf_state = hi_tools::HfCommandState::default();
    // Auto-drive feeds a synthetic prompt until the goal finishes; Ctrl-C pauses it.
    let mut pending_drive = pending_drive_prompt(agent, None);
    let mut last_outcome: Option<hi_agent::TurnOutcome> = None;

    loop {
        // Refresh profile names for the completer (covers add/edit changes).
        *profiles.borrow_mut() = config::profile_names(config);
        let readline = match pending_drive.take() {
            Some(prompt) => {
                if prompt == hi_agent::GOAL_CONTINUE_PROMPT {
                    if let Some(sg) = agent.structured_goal().and_then(|g| g.active_sub_goal()) {
                        println!("\x1b[2m⟳ goal drive — {}\x1b[0m", sg.description);
                    }
                } else if prompt == hi_agent::PLAN_DRIVE_PROMPT {
                    if let Some(step) = agent.next_plan_step_title() {
                        println!("\x1b[2m⟳ plan drive — {step}\x1b[0m");
                    } else {
                        println!("\x1b[2m⟳ plan drive\x1b[0m");
                    }
                }
                Ok(prompt)
            }
            None => editor.readline("› "),
        };
        match readline {
            Ok(line) => {
                let line = line.trim().to_string();
                if line.is_empty() {
                    if let Some(prompt) =
                        agent.drive_decision(last_outcome.as_ref()).resume_prompt()
                    {
                        pending_drive = Some(prompt.to_string());
                    }
                    continue;
                }
                // Synthetic drive prompts aren't user input — keep them out of
                // the line history.
                if line != hi_agent::GOAL_CONTINUE_PROMPT
                    && line != hi_agent::PLAN_DRIVE_PROMPT
                    && !hi_agent::command::hides_from_history(&line)
                {
                    let _ = editor.add_history_entry(&line);
                }

                // Resolve the line to a prompt to run. Commands either handle
                // themselves (and `continue`) or yield a prompt (`/retry`).
                let mut restore_model_state: Option<hi_agent::AgentModelState> = None;
                let input = if let Some(command) =
                    hi_agent::command::parse(&line).map(hi_agent::command::resolve_command)
                {
                    match command {
                        Command::Quit => break,
                        Command::Prompt(prompt) => prompt,
                        // `/btw` is a mid-turn side channel. The CLI repl is
                        // turn-synchronous (no in-flight inbox), so idle use is
                        // rejected rather than promoted to a full task turn.
                        Command::Btw(question) => {
                            let question = question.trim();
                            if question.is_empty() {
                                println!("\x1b[2musage: /btw <question>\x1b[0m");
                            } else {
                                println!(
                                    "\x1b[2m/btw is mid-turn only (TUI) — start a task, then ask aside\x1b[0m"
                                );
                            }
                            continue;
                        }
                        Command::Moa(prompt) => {
                            let prompt = prompt.trim().to_string();
                            if prompt.is_empty() {
                                println!("\x1b[2musage: /moa <prompt>\x1b[0m");
                                continue;
                            }
                            restore_model_state = Some(agent.model_state());
                            agent.set_model(hi_ai::MOA_MODEL_CONSERVATIVE.to_string(), None, None);
                            prompt
                        }
                        Command::Compact(arg) => {
                            let kind = CompactionKind::from_arg(&arg)
                                .unwrap_or_else(|| agent.compaction_kind());
                            let progress = Arc::new(AtomicBool::new(false));
                            let mut plain = PlainUi::with_progress(progress.clone());
                            let _ = drive_with_spinner(
                                agent.compact_with(kind, &mut plain),
                                &progress,
                                None,
                            )
                            .await;
                            if let Some(callback) = &after_turn {
                                callback();
                            }
                            continue;
                        }
                        Command::Retry => {
                            match (last_prompt.clone(), last_turn_snapshot.as_ref()) {
                                (Some(prompt), Some(snapshot)) => {
                                    if let Err(err) =
                                        agent.rewind_to_snapshot_durable(last_turn_start, snapshot)
                                    {
                                        eprintln!("\x1b[33mretry failed: {err:#}\x1b[0m");
                                        continue;
                                    }
                                    println!("\x1b[2mretrying: {prompt}\x1b[0m");
                                    prompt
                                }
                                _ => {
                                    println!("\x1b[2mnothing to retry yet\x1b[0m");
                                    continue;
                                }
                            }
                        }
                        Command::Edit => {
                            // Load the last user prompt into the line editor
                            // for editing. We use rustyline's `set_line` via
                            // a re-readline with a prefilled buffer.
                            match agent.last_user_message() {
                                Some(prev) => {
                                    // Re-readline with the previous prompt
                                    // pre-filled and the cursor at the end.
                                    let edited = editor.readline_with_initial("› ", (&prev, ""));
                                    match edited {
                                        Ok(line) if line.trim().is_empty() => continue,
                                        Ok(line) => {
                                            let line = line.trim().to_string();
                                            let _ = editor.add_history_entry(&line);
                                            line
                                        }
                                        Err(ReadlineError::Interrupted) => continue,
                                        Err(err) => {
                                            eprintln!("input error: {err}");
                                            continue;
                                        }
                                    }
                                }
                                None => {
                                    println!("\x1b[2mnothing to edit yet\x1b[0m");
                                    continue;
                                }
                            }
                        }
                        Command::Init => {
                            println!("\x1b[2mscanning the project to write HI.md…\x1b[0m");
                            hi_agent::command::INIT_PROMPT.to_string()
                        }
                        Command::Learn(request) => {
                            println!("\x1b[2mlearning a reusable skill…\x1b[0m");
                            hi_agent::build_learn_prompt(&request)
                        }
                        Command::Skill(name) => {
                            let name = name.trim();
                            if name.is_empty() {
                                println!("\x1b[2musage: /skill <name>\x1b[0m");
                                continue;
                            }
                            match hi_agent::read_skill(name) {
                                Ok(skill) => hi_agent::build_skill_use_prompt(
                                    &skill.skill.name,
                                    &skill.content,
                                ),
                                Err(err) => {
                                    println!("\x1b[33m{err}\x1b[0m");
                                    continue;
                                }
                            }
                        }
                        Command::Diff => {
                            let diff = hi_tools::working_tree_diff_in(agent.workspace_root()).await;
                            println!("{diff}");
                            continue;
                        }
                        Command::DiffLab(arg) => {
                            println!(
                                "use `hi diff-lab run --mode {}` from the shell, or open the TUI with /diff-lab",
                                if arg.trim().is_empty() {
                                    "local"
                                } else {
                                    arg.trim()
                                }
                            );
                            continue;
                        }
                        Command::Review(_) => {
                            // The TUI opens a full-screen overlay for /review;
                            // in the plain REPL just print the diff like /diff.
                            let diff = hi_tools::working_tree_diff_in(agent.workspace_root()).await;
                            println!("{diff}");
                            continue;
                        }
                        Command::Files => {
                            let files = agent.session_touched_paths();
                            if files.is_empty() {
                                println!("no files changed this session yet");
                            } else {
                                println!(
                                    "── {} file{} changed ──",
                                    files.len(),
                                    if files.len() == 1 { "" } else { "s" }
                                );
                                for f in files {
                                    println!("  {f}");
                                }
                            }
                            continue;
                        }
                        Command::Commit => {
                            let paths = agent.session_touched_paths();
                            if paths.is_empty() {
                                println!(
                                    "\x1b[2mnothing this session changed — stage files yourself.\x1b[0m"
                                );
                                continue;
                            }
                            let diff =
                                hi_tools::working_tree_diff_plain_in(agent.workspace_root()).await;
                            if diff.trim() != "(no changes)" && !diff.trim().is_empty() {
                                let preview: String =
                                    diff.lines().take(20).collect::<Vec<_>>().join("\n");
                                let total = diff.lines().count();
                                println!(
                                    "\x1b[2m--- committing session paths ({total} line(s) of diff) ---\x1b[0m"
                                );
                                println!("{preview}");
                                if total > 20 {
                                    println!("\x1b[2m  … {} more line(s)\x1b[0m", total - 20);
                                }
                            }
                            let out = hi_tools::commit_in(agent.workspace_root(), &paths).await;
                            for line in out.lines() {
                                println!("\x1b[2m── {line} ──\x1b[0m");
                            }
                            continue;
                        }
                        Command::Undo => {
                            match agent.undo().await {
                                Ok(Some(0)) => println!("\x1b[2mnothing changed to undo\x1b[0m"),
                                Ok(Some(n)) => {
                                    println!(
                                        "\x1b[2m↩ undid the last turn — restored {n} file(s)\x1b[0m"
                                    )
                                }
                                Ok(None) => println!("\x1b[2mnothing to undo\x1b[0m"),
                                Err(err) => eprintln!("\x1b[33mundo failed: {err:#}\x1b[0m"),
                            }
                            continue;
                        }
                        // `/model` with no id: list available live models.
                        Command::Model(id) if id.is_empty() => {
                            match agent.list_models().await {
                                Ok(mut models) if !models.is_empty() => {
                                    models.sort_by(|a, b| a.id.cmp(&b.id));
                                    println!(
                                        "\x1b[2mavailable models (current: {}):\x1b[0m",
                                        agent.model()
                                    );
                                    for m in &models {
                                        let mark = if m.id == agent.model() { "▶" } else { " " };
                                        println!("  {mark} {}", m.id);
                                    }
                                    println!("\x1b[2muse /model <id> to set the model\x1b[0m");
                                }
                                _ => {
                                    println!(
                                        "model: {}\n\x1b[2m(live model list not loaded; use /model <id> to set the model)\x1b[0m",
                                        agent.model()
                                    );
                                }
                            }
                            continue;
                        }
                        Command::Model(id) => {
                            let served = agent
                                .list_models()
                                .await
                                .ok()
                                .and_then(|models| models.into_iter().find(|m| m.id == id));
                            let window = served.as_ref().and_then(|m| m.context_window);
                            let max_output = served.as_ref().and_then(|m| m.max_output_tokens);
                            agent.set_model(id.clone(), window, max_output);
                            agent.set_usage_pricing(served.as_ref().and_then(|m| m.price));
                            // Same guard as the TUI: the active name may be a
                            // provider preset (`/provider xai`) with no profile
                            // behind it, and there is nothing to persist into.
                            if let Some(name) = active_profile
                                .as_deref()
                                .filter(|name| config.profiles.contains_key(*name))
                            {
                                match config::set_profile_model(
                                    config,
                                    name,
                                    &id,
                                    config_path.as_deref(),
                                ) {
                                    Ok(()) => {
                                        println!("model set to {id} (saved to profile {name})");
                                    }
                                    Err(err) => {
                                        println!("model set to {id}");
                                        eprintln!(
                                            "\x1b[33mcouldn't save model to profile '{name}': {err:#}\x1b[0m"
                                        );
                                    }
                                }
                            } else {
                                println!("model set to {id}");
                            }
                            let profile = active_profile
                                .as_deref()
                                .filter(|name| config.profiles.contains_key(*name));
                            let _ = config::remember_session(
                                Path::new("."),
                                profile,
                                &active_provider_label,
                                &id,
                            );
                            continue;
                        }
                        // `/provider` with no arg: list configured profiles.
                        // `/provider <name>`: use that profile, then list live
                        // model metadata so `/model` can set one when needed.
                        // `/provider add`: interactively create a new profile.
                        // `/provider edit [name]`: edit an existing profile.
                        // Subscription sign-in. Handled here rather than in the
                        // synchronous command handler because the device flow
                        // is async and long-lived (it waits on the browser).
                        Command::Login(arg) => {
                            match login_provider_arg(arg.trim()) {
                                Ok(LoginProvider::Xai) => {
                                    let already = hi_ai::xai_auth::has_credential();
                                    if let Err(err) = hi_ai::xai_auth::login().await {
                                        eprintln!("\x1b[33m/login failed: {err:#}\x1b[0m");
                                    } else if !already {
                                        println!(
                                            "\x1b[2mRun /provider xai to use the new credential.\x1b[0m"
                                        );
                                    }
                                }
                                Ok(LoginProvider::Pipenetwork) => {
                                    let already = hi_ai::pipenetwork_auth::has_credential();
                                    if let Err(err) = hi_ai::pipenetwork_auth::login().await {
                                        eprintln!("\x1b[33m/login failed: {err:#}\x1b[0m");
                                    } else if !already {
                                        println!(
                                            "\x1b[2mRun /provider pipenetwork to use the new credential.\x1b[0m"
                                        );
                                    }
                                }
                                Ok(LoginProvider::X402) => {
                                    if let Err(err) =
                                        crate::x402::login(config, config_path.as_deref())
                                    {
                                        eprintln!("\x1b[33m/login failed: {err:#}\x1b[0m");
                                    } else {
                                        println!(
                                            "\x1b[2mRun /provider pipenetwork to use USDC x402.\x1b[0m"
                                        );
                                    }
                                }
                                Err(message) => eprintln!("\x1b[33m{message}\x1b[0m"),
                            }
                            continue;
                        }
                        Command::Logout(arg) => {
                            match login_provider_arg(arg.trim()) {
                                Ok(LoginProvider::Xai) => {
                                    if let Err(err) = hi_ai::xai_auth::logout() {
                                        eprintln!("\x1b[33m/logout failed: {err:#}\x1b[0m");
                                    }
                                }
                                Ok(LoginProvider::Pipenetwork) => {
                                    if let Err(err) = hi_ai::pipenetwork_auth::logout() {
                                        eprintln!("\x1b[33m/logout failed: {err:#}\x1b[0m");
                                    } else if hi_ai::has_credit_token() {
                                        println!(
                                            "x402 credit token still stored — /logout x402 to remove it"
                                        );
                                    }
                                }
                                Ok(LoginProvider::X402) => {
                                    if let Err(err) = crate::x402::logout() {
                                        eprintln!("\x1b[33m/logout failed: {err:#}\x1b[0m");
                                    }
                                }
                                Err(message) => eprintln!("\x1b[33m{message}\x1b[0m"),
                            }
                            continue;
                        }
                        Command::Auth(arg) => {
                            match crate::auth::split_auth_arg(&arg) {
                                Ok((provider, key)) => {
                                    let key = match key {
                                        Some(key) => key,
                                        None => match crate::auth::read_secret_line(&format!(
                                            "Paste your {} API key: ",
                                            provider.as_str()
                                        )) {
                                            Ok(key) => key,
                                            Err(err) => {
                                                eprintln!("\x1b[33m/auth failed: {err:#}\x1b[0m");
                                                continue;
                                            }
                                        },
                                    };
                                    if key.is_empty() {
                                        eprintln!("\x1b[33mno API key entered\x1b[0m");
                                        continue;
                                    }
                                    let path =
                                        config_path.clone().or_else(config::default_config_path);
                                    let Some(path) = path else {
                                        eprintln!(
                                            "\x1b[33mcould not determine config directory\x1b[0m"
                                        );
                                        continue;
                                    };
                                    match crate::auth::apply_pasted_key(
                                        config, provider, &key, None, &path,
                                    )
                                    .await
                                    {
                                        Ok(hi_ai::KeyCheck::Accepted) => {
                                            println!(
                                                "\x1b[2msaved {} — /provider {} to use it\x1b[0m",
                                                provider.as_str(),
                                                provider.as_str()
                                            );
                                        }
                                        Ok(hi_ai::KeyCheck::Unverified(msg)) => {
                                            println!(
                                                "\x1b[33msaved {} (unverified: {msg}) — /provider {}\x1b[0m",
                                                provider.as_str(),
                                                provider.as_str()
                                            );
                                        }
                                        Ok(hi_ai::KeyCheck::Rejected(msg)) => {
                                            eprintln!("\x1b[33mnot saved: {msg}\x1b[0m");
                                        }
                                        Err(err) => {
                                            eprintln!("\x1b[33m/auth failed: {err:#}\x1b[0m");
                                        }
                                    }
                                }
                                Err(message) => eprintln!("\x1b[33m{message}\x1b[0m"),
                            }
                            continue;
                        }
                        Command::Provider(arg) => {
                            let arg = arg.trim();
                            // --- Subcommands ---
                            if arg == "add" {
                                match provider_add_prompt(
                                    config,
                                    config_path.as_deref(),
                                    &mut editor,
                                ) {
                                    Ok(name) => {
                                        println!(
                                            "\x1b[2msaved profile '{name}' — /provider {name} to use\x1b[0m"
                                        );
                                    }
                                    Err(err) => {
                                        eprintln!("\x1b[33m/provider add failed: {err}\x1b[0m");
                                    }
                                }
                                continue;
                            }
                            if let Some(edit_name) = arg.strip_prefix("edit") {
                                let edit_name = edit_name.trim();
                                match provider_edit_prompt(
                                    config,
                                    config_path.as_deref(),
                                    edit_name,
                                    &mut editor,
                                ) {
                                    Ok(name) => {
                                        println!("\x1b[2msaved profile '{name}'\x1b[0m");
                                    }
                                    Err(err) => {
                                        eprintln!("\x1b[33m/provider edit failed: {err}\x1b[0m");
                                    }
                                }
                                continue;
                            }
                            if let Some(rm_name) = arg
                                .strip_prefix("remove")
                                .or_else(|| arg.strip_prefix("rm"))
                            {
                                let rm_name = rm_name.trim();
                                if rm_name.is_empty() {
                                    // Never guess a target — deleting the
                                    // alphabetically-first profile (and its API
                                    // key) because the user typed `/provider
                                    // remove` to see usage is silent data loss.
                                    eprintln!(
                                        "\x1b[33m/provider remove <name> — name the profile to delete (see /provider)\x1b[0m"
                                    );
                                    continue;
                                }
                                let target = rm_name.to_string();
                                let active = config.default_profile.as_ref();
                                if active.map(|a| a.as_str()) == Some(&target) {
                                    eprintln!(
                                        "\x1b[33mcan't remove '{target}' — make a different profile active first\x1b[0m"
                                    );
                                    continue;
                                }
                                match config::remove_profile(
                                    config,
                                    &target,
                                    config_path.as_deref(),
                                ) {
                                    Ok(true) => {
                                        println!("\x1b[2mremoved profile '{target}'\x1b[0m");
                                    }
                                    Ok(false) => {
                                        eprintln!("\x1b[33mno profile named '{target}'\x1b[0m");
                                    }
                                    Err(err) => {
                                        eprintln!("\x1b[33m/provider remove failed: {err}\x1b[0m");
                                    }
                                }
                                continue;
                            }
                            // --- Use / list ---
                            if arg.is_empty() {
                                let names = config::profile_names(config);
                                if names.is_empty() {
                                    println!(
                                        "\x1b[2mno profiles configured — use /provider add, or add [profiles.<name>] to hi.toml\n\
                                         or switch straight to a provider: /provider xai · pipenetwork · anthropic · openai · ollama\x1b[0m"
                                    );
                                } else {
                                    let active = config.default_profile.as_deref();
                                    println!("\x1b[2mconfigured profiles:\x1b[0m");
                                    for name in &names {
                                        let p = config.profiles.get(name);
                                        let prov = p
                                            .and_then(|p| p.provider)
                                            .map(provider_label)
                                            .unwrap_or("openai");
                                        let model = p
                                            .and_then(|p| p.model.as_deref())
                                            .unwrap_or("(not configured)");
                                        let mark = if active == Some(name.as_str()) {
                                            "▶"
                                        } else {
                                            " "
                                        };
                                        let mut row = format!("  {mark} {name} — {prov} · {model}");
                                        if let Some(url) =
                                            p.and_then(|p| p.base_url.as_deref()).filter(|url| {
                                                let default = p
                                                    .and_then(|p| p.provider)
                                                    .map(|prov| prov.default_base_url())
                                                    .unwrap_or("");
                                                url.trim_end_matches('/')
                                                    != default.trim_end_matches('/')
                                            })
                                        {
                                            row.push_str(&format!("  ·  {url}"));
                                        }
                                        println!("\x1b[2m{row}\x1b[0m");
                                    }
                                    println!(
                                        "\x1b[2m/provider <name> — a profile, or a provider (xai, pipenetwork, anthropic, openai, ollama)\n\
                                         /provider add · /provider edit [name] · /provider remove [name]\x1b[0m"
                                    );
                                }
                                continue;
                            }
                            // Resolve the profile and update the provider.
                            if config
                                .profiles
                                .get(arg)
                                .and_then(|profile| profile.runtime.as_ref())
                                .is_some_and(|runtime| runtime.kind == "mlx")
                            {
                                match switch_to_managed_local_profile(agent, config, arg).await {
                                    Ok(Some((label, model))) => {
                                        active_profile = Some(arg.to_string());
                                        active_provider_label = label.to_string();
                                        let _ = config::remember_session(
                                            Path::new("."),
                                            Some(arg),
                                            &active_provider_label,
                                            &model,
                                        );
                                        println!(
                                            "\x1b[2musing {label} (managed local profile: {arg}) — model: {model}\x1b[0m"
                                        );
                                    }
                                    Ok(None) => {}
                                    Err(err) => eprintln!(
                                        "\x1b[33m/provider managed local startup failed: {err:#}\x1b[0m"
                                    ),
                                }
                                continue;
                            }
                            match config::resolve_named_profile(config, arg) {
                                Ok(new_settings) => {
                                    let label = provider_label(new_settings.provider);
                                    let model = new_settings.model.clone();
                                    let provider: std::sync::Arc<dyn hi_ai::Provider> =
                                        crate::build_chain(&new_settings, Vec::new()).into();
                                    agent.clear_driver_local_server();
                                    agent.set_provider(
                                        provider,
                                        model.clone(),
                                        None,
                                        new_settings.max_tokens,
                                        new_settings.max_tokens_explicit,
                                        None,
                                    );
                                    // Track the now-active profile so a later
                                    // `/model` persists into THIS profile, not the
                                    // startup one (which would corrupt a different
                                    // profile's config with a foreign model id).
                                    active_profile = Some(arg.to_string());
                                    active_provider_label = label.to_string();
                                    let profile = active_profile
                                        .as_deref()
                                        .filter(|name| config.profiles.contains_key(*name));
                                    let _ = config::remember_session(
                                        Path::new("."),
                                        profile,
                                        &active_provider_label,
                                        &model,
                                    );
                                    // Only call it a profile when one exists —
                                    // `/provider xai` selects a provider preset.
                                    if config.profiles.contains_key(arg) {
                                        println!(
                                            "\x1b[2musing {label} (profile: {arg}) — model: {model}\x1b[0m"
                                        );
                                    } else {
                                        println!(
                                            "\x1b[2musing {label} — model: {model}  (no profile; /provider add to save these settings)\x1b[0m"
                                        );
                                    }
                                    if model == "__model_not_configured__" {
                                        println!(
                                            "\x1b[2mno model configured for this profile — use /model to view available models\x1b[0m"
                                        );
                                    }
                                    // List available live models for the active profile.
                                    match agent.list_models().await {
                                        Ok(mut models) if !models.is_empty() => {
                                            if let Some(served) =
                                                models.iter().find(|m| m.id == model)
                                            {
                                                let window = served.context_window;
                                                agent.set_model(
                                                    model.clone(),
                                                    window,
                                                    served.max_output_tokens,
                                                );
                                            }
                                            models.sort_by(|a, b| a.id.cmp(&b.id));
                                            println!("\x1b[2mavailable models for {label}:\x1b[0m");
                                            for m in &models {
                                                let mark =
                                                    if m.id == agent.model() { "▶" } else { " " };
                                                println!("  {mark} {}", m.id);
                                            }
                                            println!(
                                                "\x1b[2muse /model <id> to set the model\x1b[0m"
                                            );
                                        }
                                        _ => {
                                            println!(
                                                "\x1b[2m(live model list not loaded; use /model <id> to set the model)\x1b[0m"
                                            );
                                        }
                                    }
                                }
                                Err(err) => {
                                    eprintln!("\x1b[33m/provider failed: {err}\x1b[0m");
                                }
                            }
                            continue;
                        }
                        Command::Mcp(arg) => {
                            let arg = arg.trim();
                            if arg == "pipe" {
                                let Some(url) = settings.mcp_url.as_deref() else {
                                    eprintln!(
                                        "\x1b[33mno MCP URL configured for this provider\x1b[0m"
                                    );
                                    continue;
                                };
                                match crate::orchestration::mcp_inspect(
                                    url,
                                    &settings.api_key,
                                    &settings.model,
                                )
                                .await
                                {
                                    Ok(report) => print!("{report}"),
                                    Err(err) => {
                                        eprintln!("\x1b[33mmcp inspection failed: {err:#}\x1b[0m")
                                    }
                                }
                                continue;
                            }
                            let report = if arg.is_empty() {
                                agent.mcp_workspace_status().await
                            } else {
                                match agent.mcp_workspace_admin(arg).await {
                                    None => None,
                                    Some(Ok(text)) => Some(text),
                                    Some(Err(err)) => Some(format!("mcp: {err:#}")),
                                }
                            };
                            match report {
                                Some(text) => print!("{text}"),
                                None => println!(
                                    "\x1b[2mno workspace MCP servers. Add `.hi/mcp/*.json` or `.mcp.json`. First-party Pipe attaches when mcp_url + API key are set (`/mcp pipe` inspects the endpoint).\x1b[0m"
                                ),
                            }
                            continue;
                        }
                        Command::Doctor => {
                            let cwd = std::env::current_dir()
                                .unwrap_or_else(|_| std::path::PathBuf::from("."));
                            let lsp = agent.lsp_status_report();
                            let report = crate::doctor::run_doctor_for_session(
                                &cwd,
                                settings,
                                crate::doctor::SessionDoctorFacts {
                                    model: agent.model(),
                                    verify_summary: &agent.verify_summary(),
                                    lsp_summary: Some(lsp.as_str()),
                                    checkpoint_count: agent.checkpoint_count(),
                                    workspace_root: Some(agent.workspace_root()),
                                },
                            )
                            .await;
                            print!("{}", crate::doctor::report_text(&report));
                            continue;
                        }
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
                        | Command::Cd(_)
                        | Command::Rename(_)
                        | Command::Resume(_) => {
                            if let Some(effect) =
                                hi_agent::handle_session_command(agent, &command, &[])
                            {
                                print!("{}", effect.message);
                                if !effect.message.ends_with('\n') {
                                    println!();
                                }
                                if let Some(prompt) = effect.follow_up_prompt {
                                    // Run the plan-mode turn immediately in the REPL.
                                    // Fall through by setting last_prompt path via re-binding is awkward;
                                    // just execute here with the same drive helper pattern as compact.
                                    let progress = Arc::new(AtomicBool::new(false));
                                    let mut plain = PlainUi::with_progress(progress.clone());
                                    let cancellation = hi_agent::TurnCancellation::new();
                                    let (driven, _) = drive_with_spinner(
                                        agent.run_turn_cancellable(
                                            &prompt,
                                            &mut plain,
                                            cancellation.clone(),
                                        ),
                                        &progress,
                                        Some(cancellation.clone()),
                                    )
                                    .await;
                                    if driven.as_ref().is_some_and(Result::is_err)
                                        && !(cancellation.is_cancelled()
                                            && agent.last_turn_outcome().is_some_and(|outcome| {
                                                outcome.status == hi_agent::TurnStatus::Cancelled
                                            }))
                                    {
                                        let _ = agent
                                            .cleanup_turn(hi_agent::TurnCleanupKind::Fail)
                                            .await;
                                    }
                                    if let Some(callback) = &after_turn {
                                        callback();
                                    }
                                }
                            }
                            continue;
                        }
                        Command::Hf(arg) => {
                            match hi_tools::handle_hf_command_result(&arg, &mut hf_state).await {
                                Ok(hi_tools::HfCommandResult::Text(text)) => print!("{text}"),
                                Ok(hi_tools::HfCommandResult::MlxReady(run)) => {
                                    print!("{}", run.message);
                                    match switch_to_mlx_profile(
                                        agent,
                                        config,
                                        config_path.as_deref(),
                                        &run,
                                    )
                                    .await
                                    {
                                        Ok(()) => {
                                            active_profile = Some(run.profile_name.clone());
                                            println!(
                                                "\x1b[2musing local MLX profile '{}' — model: {}\x1b[0m",
                                                run.profile_name, run.model_id
                                            );
                                        }
                                        Err(err) => eprintln!(
                                            "\x1b[33m/hf run --mlx profile switch failed: {err:#}\x1b[0m"
                                        ),
                                    }
                                }
                                Err(err) => eprintln!("\x1b[33m/hf failed: {err:#}\x1b[0m"),
                            }
                            continue;
                        }
                        Command::Lsp(arg) => {
                            crate::commands::handle_lsp(agent, &arg);
                            continue;
                        }
                        Command::Rsi(arg) => {
                            crate::commands::handle_rsi_command(agent, &arg).await;
                            continue;
                        }
                        Command::Config(arg)
                            if matches!(
                                hi_agent::command::parse_config_arg(&arg),
                                hi_agent::command::ConfigArg::RsiShow
                                    | hi_agent::command::ConfigArg::Rsi(_)
                                    | hi_agent::command::ConfigArg::RsiSpendLimit(_)
                            ) =>
                        {
                            crate::commands::handle_rsi_config(
                                agent,
                                hi_agent::command::parse_config_arg(&arg),
                            )
                            .await;
                            continue;
                        }
                        // `/config skeptic-local <on|off>` may download a model
                        // and spawn a local server, so it runs on the async path;
                        // every other `/config …` falls through to the sync handler.
                        Command::Config(arg)
                            if hi_agent::command::config_is_skeptic_local(&arg) =>
                        {
                            crate::commands::handle_skeptic_local(agent, &arg).await;
                            continue;
                        }
                        // `/goal <objective>` with a planner: decompose (one bounded
                        // call), then install the structured goal. Control subcommands
                        // (clear/pause/resume/limit) and the no-planner case fall
                        // through to the sync handler.
                        Command::Goal(arg)
                            if agent.has_planner()
                                && hi_agent::command::goal_arg_is_objective(&arg) =>
                        {
                            crate::commands::handle_goal_planned(agent, arg.trim()).await;
                            // A goal is a contract: start pulling toward it now.
                            // Ctrl-C during a drive turn pauses it.
                            if agent
                                .structured_goal()
                                .is_some_and(hi_agent::Goal::should_auto_drive)
                            {
                                agent.reset_goal_drive_stall();
                                pending_drive = Some(hi_agent::GOAL_CONTINUE_PROMPT.to_string());
                            }
                            continue;
                        }
                        other => {
                            // `/goal <objective>` without a planner, and
                            // `/goal resume`, also (re)start the drive.
                            let could_drive = matches!(
                                &other,
                                Command::Goal(a)
                                    if hi_agent::command::goal_arg_is_objective(a)
                                        || a.trim() == "resume"
                            );
                            handle_command(
                                agent,
                                other,
                                Some(config),
                                active_profile.as_deref(),
                                config_path.as_deref(),
                                approval_store.as_deref(),
                            );
                            if could_drive
                                && agent
                                    .structured_goal()
                                    .is_some_and(hi_agent::Goal::should_auto_drive)
                            {
                                agent.reset_goal_drive_stall();
                                pending_drive = Some(hi_agent::GOAL_CONTINUE_PROMPT.to_string());
                            }
                            continue;
                        }
                    }
                } else {
                    hi_tui::expand_file_mentions(&line, agent.workspace_root())
                };

                // Run the turn with an animated "working… Ns" spinner so it's
                // always clear something is happening. Ctrl-C cancels the turn.
                last_prompt = Some(input.clone());
                // Auto-drive bookkeeping: any goal-state change by turn end
                // (advance, retry note, plan growth) is progress; none is a stall.
                let goal_drive_turn = input == hi_agent::GOAL_CONTINUE_PROMPT;
                let plan_drive_turn = input == hi_agent::PLAN_DRIVE_PROMPT;
                let goal_before = agent.structured_goal().cloned();
                let plan_step_before = agent.next_plan_step_title().map(str::to_owned);
                let checkpoint = agent.messages().len();
                last_turn_start = checkpoint;
                let turn_snapshot = agent.state_snapshot();
                last_turn_snapshot = Some(turn_snapshot.clone());
                let progress = Arc::new(AtomicBool::new(false));
                let (driven, interrupt_requested, cancellation_requested) = {
                    let mut plain = PlainUi::with_progress(progress.clone());
                    let cancellation = hi_agent::TurnCancellation::new();
                    let (result, interrupted) = drive_with_spinner(
                        agent.run_turn_cancellable(&input, &mut plain, cancellation.clone()),
                        &progress,
                        Some(cancellation.clone()),
                    )
                    .await;
                    (result, interrupted, cancellation.is_cancelled())
                };
                let cancelled = driven.is_none()
                    || (interrupt_requested
                        && driven.as_ref().is_some_and(|result| {
                            matches!(
                                result,
                                Ok(outcome) if outcome.status == hi_agent::TurnStatus::Cancelled
                            )
                        }));
                if cancelled {
                    // `run_turn_cancellable` has already settled tool results,
                    // rewound durable session state, rolled back checkpoints,
                    // and killed turn-scoped background work. Do not race it
                    // with a second frontend-owned cleanup pass.
                    println!("\x1b[33m^C — interrupted; turn discarded\x1b[0m");
                    // Interrupting a drive turn is an explicit "stop": pause the
                    // goal so the drive doesn't restart on the next message.
                    if goal_drive_turn
                        && agent.set_goal_pause_reason(hi_agent::GoalPauseReason::User)
                    {
                        println!(
                            "\x1b[33mgoal drive interrupted — paused (user); /goal resume to continue\x1b[0m"
                        );
                    }
                    if plan_drive_turn {
                        agent.pause_plan_drive_until_user_input()?;
                        println!(
                            "\x1b[33mplan drive interrupted — paused; reply to steer and resume, or use /plan resume\x1b[0m"
                        );
                    }
                } else if driven.as_ref().is_some_and(Result::is_err) {
                    // A configured hard timeout completes Agent-owned Cancel
                    // cleanup before preserving its deadline error. Avoid
                    // overwriting that terminal Cancel outcome with Fail.
                    let already_cancelled = cancellation_requested
                        && agent.last_turn_outcome().is_some_and(|outcome| {
                            outcome.status == hi_agent::TurnStatus::Cancelled
                        });
                    if !already_cancelled {
                        let _ = agent.cleanup_turn(hi_agent::TurnCleanupKind::Fail).await;
                    }
                    if goal_drive_turn {
                        let _ = agent.set_goal_pause_reason(hi_agent::GoalPauseReason::Infra);
                    }
                } else if cancellation_requested {
                    // Cancellation arrived after the body commit boundary. Keep
                    // the valid completed outcome, but honor the stop request by
                    // not scheduling another synthetic goal/plan drive turn.
                    last_outcome = driven
                        .as_ref()
                        .and_then(|result| result.as_ref().ok())
                        .cloned();
                    pending_drive = None;
                    if goal_drive_turn {
                        let _ = agent.set_goal_pause_reason(hi_agent::GoalPauseReason::User);
                    }
                    if plan_drive_turn {
                        agent.pause_plan_drive_until_user_input()?;
                    }
                } else {
                    // Long-horizon auto-drive: keep pulling toward leftover work.
                    // Drive turns that change nothing count toward a stall park.
                    if goal_drive_turn {
                        let made_progress =
                            agent.goal_drive_turn_made_progress(goal_before.as_ref());
                        match agent.note_goal_drive_progress(made_progress) {
                            hi_agent::GoalDriveProgress::Skipped { failed, next } => {
                                println!(
                                    "\x1b[33m{}\x1b[0m",
                                    hi_agent::goal_drive_skip_message(&failed, next.as_deref())
                                );
                            }
                            hi_agent::GoalDriveProgress::Parked => {
                                println!(
                                    "\x1b[33m{}\x1b[0m",
                                    hi_agent::goal_drive_park_message(
                                        agent.leftover_work().as_deref()
                                    )
                                );
                            }
                            _ => {}
                        }
                    }
                    if let Some(count) = agent.take_goal_requeue_notice() {
                        println!(
                            "\x1b[33m{}\x1b[0m",
                            hi_agent::goal_drive_requeue_message(count)
                        );
                    }
                    if plan_drive_turn {
                        let made_progress =
                            agent.plan_drive_turn_made_progress(plan_step_before.as_deref());
                        agent.note_plan_drive_progress(made_progress);
                        if agent.plan_drive_status() == "parked" {
                            println!(
                                "\x1b[33m{}\x1b[0m",
                                hi_agent::plan_drive_park_message(
                                    agent.plan_leftover_work().as_deref()
                                )
                            );
                        }
                    }
                    let outcome = driven.as_ref().and_then(|r| r.as_ref().ok());
                    last_outcome = outcome.cloned();
                    pending_drive = pending_drive_prompt(agent, outcome);
                }
                if let Some(state) = restore_model_state.take() {
                    agent.restore_model_state(state);
                }
                if let Some(callback) = &after_turn {
                    callback();
                }
            }
            Err(ReadlineError::Interrupted) => continue, // Ctrl-C: discard the line
            Err(ReadlineError::Eof) => break,            // Ctrl-D: quit
            Err(err) => {
                eprintln!("input error: {err}");
                break;
            }
        }
    }

    // Session ending: distill durable lessons into .hi/memory.md (loaded next
    // session). Skip an empty session — only if the model actually did work.
    if hi_agent::should_distill_memory(auto_memory, agent.totals().output_tokens) {
        let progress = Arc::new(AtomicBool::new(false));
        let mut plain = PlainUi::with_progress(progress.clone());
        let memory = async {
            agent.update_memory(&mut plain).await;
            Ok::<(), anyhow::Error>(())
        };
        let _ = drive_with_spinner(memory, &progress, None).await;
    }

    // Don't leave background processes (dev servers, watchers) running after
    // the session ends.
    agent.kill_background_processes();

    if let Some(path) = &history {
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let _ = editor.save_history(path);
    }
    // Snapshot provider/model so the next bare `hi` in this workspace resumes
    // with the same routing (also written on /model and /provider changes).
    let profile = active_profile
        .as_deref()
        .filter(|name| config.profiles.contains_key(*name));
    let _ = config::remember_session(
        Path::new("."),
        profile,
        &active_provider_label,
        agent.model(),
    );
    Ok(())
}

async fn switch_to_mlx_profile(
    agent: &mut Agent,
    config: &mut config::Config,
    config_path: Option<&Path>,
    run: &hi_tools::HfMlxRun,
) -> Result<()> {
    let result = async {
        let profile = config::Profile {
            provider: Some(config::ProviderName::Openai),
            model: Some(run.model_id.clone()),
            base_url: Some(run.base_url.clone()),
            api_key: Some("local".to_string()),
            max_tokens: Some(2048),
            runtime: Some(config::LocalRuntimeProfile {
                kind: "mlx".to_string(),
                repo: run.repo.clone(),
                backend: Some("mlx".to_string()),
                autostart: true,
                model_path: None,
                quantization: None,
                context_window: None,
                tool_mode: Some(hi_ai::ToolMode::ChatOnly),
            }),
            ..Default::default()
        };
        config::upsert_profile_project_local(config, &run.profile_name, profile, config_path)?;
        let settings = config::resolve_named_profile(config, &run.profile_name)?;
        let provider: std::sync::Arc<dyn hi_ai::Provider> =
            crate::build_chain(&settings, Vec::new()).into();
        let mut window: Option<u32> = None;
        agent.set_provider(
            provider,
            settings.model.clone(),
            window,
            settings.max_tokens,
            settings.max_tokens_explicit,
            None,
        );
        agent.register_driver_local_server(
            run.base_url.clone(),
            run.model_id.clone(),
            run.process_id.clone(),
        );
        if let Ok(models) = agent.list_models().await
            && let Some(served) = models.into_iter().find(|model| model.id == settings.model)
        {
            window = served.context_window.or(window);
            agent.set_model(settings.model.clone(), window, served.max_output_tokens);
            agent.set_usage_pricing(served.price);
        }
        Ok(())
    }
    .await;
    if result.is_err() {
        hi_tools::stop_local_server(&run.process_id);
    }
    result
}

async fn switch_to_managed_local_profile(
    agent: &mut Agent,
    config: &config::Config,
    name: &str,
) -> Result<Option<(String, String)>> {
    let settings = config::resolve_named_profile(config, name)?;
    if settings
        .runtime
        .as_ref()
        .is_none_or(|runtime| runtime.kind != "mlx")
    {
        return Ok(None);
    }
    let (settings, runtime) = crate::ensure_managed_local_startup(settings).await?;
    let runtime = runtime
        .ok_or_else(|| anyhow::anyhow!("managed local profile did not produce a runtime"))?;
    let label = provider_label(settings.provider).to_string();
    let model = settings.model.clone();
    let provider: std::sync::Arc<dyn hi_ai::Provider> =
        crate::build_chain(&settings, Vec::new()).into();
    agent.clear_driver_local_server();
    agent.set_provider(
        provider,
        model.clone(),
        None,
        settings.max_tokens,
        settings.max_tokens_explicit,
        None,
    );
    agent.register_driver_local_server(runtime.base_url, runtime.model_id, runtime.process_id);
    if let Ok(models) = agent.list_models().await
        && let Some(served) = models.into_iter().find(|model| model.id == settings.model)
    {
        agent.set_model(
            settings.model.clone(),
            served.context_window,
            served.max_output_tokens,
        );
    }
    Ok(Some((label, model)))
}

/// Drive a model future (a turn or a compaction) to completion, showing an
/// animated spinner until the first output and letting Ctrl-C cancel it.
/// Returns `None` only when an operation without cooperative turn cancellation
/// is interrupted. With a turn token, it keeps polling until Agent-owned
/// rollback completes and returns that terminal result; the boolean separately
/// records whether this driver observed Ctrl-C.
async fn drive_with_spinner<T>(
    fut: impl std::future::Future<Output = Result<T>>,
    progress: &AtomicBool,
    turn_cancellation: Option<hi_agent::TurnCancellation>,
) -> (Option<Result<T>>, bool) {
    use std::io::Write;

    tokio::pin!(fut);
    let started = std::time::Instant::now();
    let mut ticker = tokio::time::interval(Duration::from_millis(90));
    let mut frame = 0usize;
    let mut result = None;
    let mut cancellation_requested = false;
    loop {
        tokio::select! {
            completed = &mut fut => {
                if cancellation_requested {
                    if let Err(error) = &completed {
                        eprintln!("\r\x1b[K\x1b[31mcancellation cleanup failed: {error:#}\x1b[0m");
                    }
                    // The Agent can legitimately cross its commit boundary just
                    // before Ctrl-C, or surface an error whose Fail cleanup the
                    // caller still owns. Preserve the terminal result instead
                    // of collapsing every requested cancellation to `None`.
                    result = Some(completed);
                    break;
                }
                if let Err(err) = &completed {
                    let (kind, guidance) = hi_agent::classify_error(err);
                    let suffix = if guidance.is_empty() {
                        String::new()
                    } else {
                        format!(" — {guidance}")
                    };
                    eprintln!("\r\x1b[K\x1b[31m{kind}: {err:#}{suffix}\x1b[0m");
                }
                result = Some(completed);
                break;
            }
            _ = tokio::signal::ctrl_c(), if !cancellation_requested => {
                cancellation_requested = true;
                if let Some(cancellation) = turn_cancellation.as_ref() {
                    cancellation.cancel();
                } else {
                    break;
                }
            }
            _ = ticker.tick() => {
                if !progress.load(Ordering::Relaxed) {
                    print!(
                        "\r\x1b[2m{} working… {}s\x1b[0m\x1b[K",
                        SPINNER[frame % SPINNER.len()],
                        started.elapsed().as_secs()
                    );
                    let _ = std::io::stdout().flush();
                    frame += 1;
                }
            }
        }
    }
    if !progress.load(Ordering::Relaxed) {
        print!("\r\x1b[K");
        let _ = std::io::stdout().flush();
    }
    (result, cancellation_requested)
}

/// Read a line from the user with a prompt, using rustyline for line editing.
fn rl_prompt(editor: &mut crate::complete::ReplEditor, message: &str) -> Result<String> {
    Ok(editor.readline(message)?.trim().to_string())
}

/// Interactively create a new profile via line prompts and save it to the
/// config file. Returns the profile name.
/// Providers that support interactive `/login` (browser pairing or OAuth).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LoginProvider {
    Xai,
    Pipenetwork,
    X402,
}

/// Validate the provider argument to `/login` / `/logout`.
fn login_provider_arg(arg: &str) -> std::result::Result<LoginProvider, String> {
    match arg {
        "xai" | "grok" => Ok(LoginProvider::Xai),
        "pipenetwork" | "pipe" => Ok(LoginProvider::Pipenetwork),
        "x402" => Ok(LoginProvider::X402),
        "" => Err(
            "usage: /login xai | /login pipenetwork | /login x402 — browser pairing or USDC"
                .to_string(),
        ),
        other => Err(format!(
            "'{other}' has no sign-in. Supported: xai, pipenetwork, x402. \
             Other providers use an API key (see /provider add)."
        )),
    }
}

fn provider_add_prompt(
    config: &mut config::Config,
    config_path: Option<&Path>,
    editor: &mut crate::complete::ReplEditor,
) -> Result<String> {
    use config::{ProfileForm, ProviderName, upsert_profile};

    println!("\x1b[2m— add a provider profile —\x1b[0m");

    // Profile name.
    let name = loop {
        let n = rl_prompt(editor, "Profile name: ")?;
        if n.is_empty() {
            eprintln!("  name can't be empty");
            continue;
        }
        if config.profiles.contains_key(&n) {
            eprintln!(
                "  a profile named '{n}' already exists — use /provider edit {n} to modify it"
            );
            continue;
        }
        break n;
    };

    // Provider type.
    println!("  1) pipenetwork.ai    2) Ollama (local)    3) xAI (Grok)");
    let provider = loop {
        match rl_prompt(editor, "Provider [1-3] (default 1): ")?.as_str() {
            "" | "1" => break ProviderName::Pipenetwork,
            "2" => break ProviderName::Ollama,
            "3" => break ProviderName::Xai,
            other => eprintln!("  '{other}' isn't a choice — pick 1-3."),
        }
    };

    // API key (skip for Ollama).
    let (api_key, store_as_env) = if matches!(provider, ProviderName::Ollama) {
        (String::new(), false)
    } else {
        let key = rl_prompt(
            editor,
            &format!(
                "API key (or env var name like {}_API_KEY): ",
                provider.as_str().to_uppercase()
            ),
        )?;
        if key.is_empty() {
            (String::new(), false)
        } else {
            // Store as env var reference only if it's a plausible env var name
            // AND an env var with that name is actually set — otherwise a real
            // key that happens to be all-caps+digits+underscores would be
            // mistaken for an env var name and fail at resolve time.
            (key.clone(), config::is_env_var_reference(&key))
        }
    };

    // Model (optional; blank keeps the provider default when one exists).
    let default_model = provider.default_model().unwrap_or("");
    let model = if default_model.is_empty() {
        rl_prompt(editor, "Model id (optional): ")?
    } else {
        rl_prompt(editor, &format!("Model id (default {default_model}): "))?.to_string()
    };
    let model = if model.is_empty() {
        default_model.to_string()
    } else {
        model
    };

    // Base URL (optional — uses provider default if blank).
    let base_url = rl_prompt(
        editor,
        &format!("Base URL (blank for {}): ", provider.default_base_url()),
    )?;

    let form = ProfileForm {
        name: name.clone(),
        provider,
        api_key,
        store_as_env,
        model,
        base_url,
    };
    let profile = form.to_profile();

    upsert_profile(config, &name, profile, config_path)?;
    Ok(name)
}

/// Interactively edit an existing profile. `name` may be empty to prompt for it.
fn provider_edit_prompt(
    config: &mut config::Config,
    config_path: Option<&Path>,
    name: &str,
    editor: &mut crate::complete::ReplEditor,
) -> Result<String> {
    use config::{ProfileForm, ProviderName, upsert_profile};

    // Resolve which profile to edit.
    let name = if name.is_empty() {
        let names = config::profile_names(config);
        if names.is_empty() {
            bail!("no profiles configured — use /provider add to create one");
        }
        println!("\x1b[2mconfigured profiles:\x1b[0m");
        for n in &names {
            println!("  {n}");
        }
        loop {
            let n = rl_prompt(editor, "Profile to edit: ")?;
            if config.profiles.contains_key(&n) {
                break n;
            }
            eprintln!("  no profile named '{n}'");
        }
    } else if !config.profiles.contains_key(name) {
        bail!("no profile named '{name}'");
    } else {
        name.to_string()
    };

    let existing = config.profiles.get(&name).unwrap();
    let mut form = ProfileForm::from_profile(&name, existing);

    println!("\x1b[2m— editing profile '{name}' (blank = keep current) —\x1b[0m");

    // Provider type.
    println!(
        "  current: {} (1=pipenetwork.ai 2=Ollama 3=xAI)",
        form.provider.as_str()
    );
    let provider = loop {
        let input = rl_prompt(editor, "Provider [1-3]: ")?;
        if input.is_empty() {
            break form.provider;
        }
        match input.as_str() {
            "1" => break ProviderName::Pipenetwork,
            "2" => break ProviderName::Ollama,
            "3" => break ProviderName::Xai,
            _ => eprintln!("  pick 1-3"),
        }
    };
    form.provider = provider;

    // API key.
    let key_label = if form.store_as_env { "env var" } else { "key" };
    let masked = config::mask_key(&form.api_key);
    let new_key = rl_prompt(
        editor,
        &format!("API key/{key_label} (current: {masked}): "),
    )?;
    if !new_key.is_empty() {
        form.api_key = new_key;
        form.store_as_env = config::is_env_var_reference(&form.api_key);
    }

    // Model.
    let new_model = rl_prompt(editor, &format!("Model (current: {}): ", form.model))?;
    if !new_model.is_empty() {
        form.model = new_model;
    }

    // Base URL.
    let new_url = rl_prompt(editor, &format!("Base URL (current: {}): ", form.base_url))?;
    if !new_url.is_empty() {
        form.base_url = new_url;
    }

    let profile = form.apply_to(existing);
    upsert_profile(config, &name, profile, config_path)?;
    Ok(name)
}

#[cfg(test)]
mod mention_tests {
    #[test]
    fn repl_expands_mentions_as_pointers_not_bodies() {
        let dir = std::env::temp_dir().join(format!(
            "hi-cli-mention-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("a.rs"), "SECRET_BODY").unwrap();
        let out = hi_tui::expand_file_mentions("see @a.rs", &dir);
        assert!(out.contains("a.rs") && out.contains("<file mentions>"));
        assert!(!out.contains("SECRET_BODY"));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
