//! The plain line REPL loop and the animated-spinner turn driver.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use anyhow::{Context, Result, bail};
use hi_agent::Agent;
use hi_ai::Registry;

use crate::commands::handle_command;
use crate::config::{self, Settings};
use crate::ui::PlainUi;
use crate::{provider_label, session};

const SPINNER: [&str; 10] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

pub(crate) async fn repl(
    agent: &mut Agent,
    settings: &Settings,
    config: &mut config::Config,
    registry: &Registry,
    auto_memory: bool,
) -> Result<()> {
    use hi_agent::Command;
    use hi_agent::CompactionKind;
    use rustyline::Editor;
    use rustyline::error::ReadlineError;
    use rustyline::history::DefaultHistory;

    use crate::complete::{ProfileNames, ReplHelper};

    let window = registry
        .metadata(&settings.model)
        .1
        .map(|w| format!(" · {}k ctx", w / 1000))
        .unwrap_or_default();
    println!(
        "hi · {} · {}{} — /help for commands, Ctrl-D to quit.",
        provider_label(settings.provider),
        settings.model,
        window,
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

    loop {
        // Refresh profile names for the completer (covers add/edit changes).
        *profiles.borrow_mut() = config::profile_names(config);
        match editor.readline("› ") {
            Ok(line) => {
                let line = line.trim().to_string();
                if line.is_empty() {
                    continue;
                }
                let _ = editor.add_history_entry(&line);

                // Resolve the line to a prompt to run. Commands either handle
                // themselves (and `continue`) or yield a prompt (`/retry`).
                let input = if let Some(command) = hi_agent::command::parse(&line) {
                    match command {
                        Command::Quit => break,
                        Command::Prompt(prompt) => prompt,
                        Command::Compact(arg) => {
                            let kind = CompactionKind::from_arg(&arg)
                                .unwrap_or_else(|| agent.compaction_kind());
                            let progress = Arc::new(AtomicBool::new(false));
                            let mut plain = PlainUi::with_progress(progress.clone());
                            let _ =
                                drive_with_spinner(agent.compact_with(kind, &mut plain), &progress)
                                    .await;
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
                        Command::Diff => {
                            let diff = hi_tools::working_tree_diff().await;
                            println!("{diff}");
                            continue;
                        }
                        Command::Commit => {
                            let diff = hi_tools::working_tree_diff_plain().await;
                            if diff.trim() == "(no changes)" || diff.trim().is_empty() {
                                println!("\x1b[2mnothing to commit — no changes\x1b[0m");
                                continue;
                            }
                            // Show a preview of what will be committed.
                            let preview: String =
                                diff.lines().take(20).collect::<Vec<_>>().join("\n");
                            let total = diff.lines().count();
                            println!("\x1b[2m--- committing {total} line(s) of changes ---\x1b[0m");
                            println!("{preview}");
                            if total > 20 {
                                println!("\x1b[2m  … {} more line(s)\x1b[0m", total - 20);
                            }
                            let out = hi_tools::commit().await;
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
                        // `/model` with no id: list what the provider actually serves.
                        Command::Model(id) if id.is_empty() => {
                            match agent.list_models().await {
                                Ok(mut models) if !models.is_empty() => {
                                    models.sort_by(|a, b| a.id.cmp(&b.id));
                                    println!(
                                        "\x1b[2mmodels served by this endpoint (current: {}):\x1b[0m",
                                        agent.model()
                                    );
                                    for m in &models {
                                        let mark = if m.id == agent.model() { "▶" } else { " " };
                                        let tag = m
                                            .health()
                                            .map(|h| format!("  ({h})"))
                                            .unwrap_or_default();
                                        println!("  {mark} {}{tag}", m.id);
                                    }
                                    println!("\x1b[2m/model <id> to switch\x1b[0m");
                                }
                                _ => {
                                    println!(
                                        "model: {}\n\x1b[2m(couldn't list endpoint models; /model <id> to switch)\x1b[0m",
                                        agent.model()
                                    );
                                }
                            }
                            continue;
                        }
                        // `/provider` with no arg: list configured profiles.
                        // `/provider <name>`: switch to that profile, then list
                        // the models the new endpoint serves so the user can
                        // `/model` to pick one.
                        // `/provider add`: interactively create a new profile.
                        // `/provider edit [name]`: edit an existing profile.
                        Command::Provider(arg) => {
                            let arg = arg.trim();
                            // --- Subcommands ---
                            if arg == "add" {
                                match provider_add_prompt(config, &mut editor) {
                                    Ok(name) => {
                                        println!(
                                            "\x1b[2msaved profile '{name}' — /provider {name} to switch\x1b[0m"
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
                                match provider_edit_prompt(config, edit_name, &mut editor) {
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
                                let target = if rm_name.is_empty() {
                                    let names = config::profile_names(config);
                                    if names.is_empty() {
                                        eprintln!("\x1b[2mno profiles to remove\x1b[0m");
                                        continue;
                                    }
                                    names[0].clone()
                                } else {
                                    rm_name.to_string()
                                };
                                let active = config.default_profile.as_ref();
                                if active.map(|a| a.as_str()) == Some(&target) {
                                    eprintln!(
                                        "\x1b[33mcan't remove '{target}' — it's the active profile; switch first\x1b[0m"
                                    );
                                    continue;
                                }
                                let path = match config::writable_config_path(None) {
                                    Some(p) => p,
                                    None => {
                                        eprintln!("\x1b[33mcould not determine config path\x1b[0m");
                                        continue;
                                    }
                                };
                                match config::remove_profile(config, &target, &path) {
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
                            // --- Switch / list ---
                            if arg.is_empty() {
                                let names = config::profile_names(config);
                                if names.is_empty() {
                                    println!(
                                        "\x1b[2mno profiles configured — use /provider add, or add [profiles.<name>] to hi.toml\x1b[0m"
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
                                            .unwrap_or("(pick via /model)");
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
                                        "\x1b[2m/provider <name> to switch · /provider add · /provider edit [name] · /provider remove [name]\x1b[0m"
                                    );
                                }
                                continue;
                            }
                            // Resolve the profile and swap the provider.
                            match config::resolve_named_profile(config, arg, registry) {
                                Ok(new_settings) => {
                                    let label = provider_label(new_settings.provider);
                                    let model = new_settings.model.clone();
                                    let provider = crate::build_provider(&new_settings);
                                    let (price, window) = registry.metadata(&model);
                                    agent.set_provider(provider, model.clone(), price, window);
                                    println!(
                                        "\x1b[2mswitched to {label} (profile: {arg}) — model: {model}\x1b[0m"
                                    );
                                    if model == "__pick_via_model__" {
                                        println!(
                                            "\x1b[2mno model configured for this profile — use /model to pick from what this endpoint serves\x1b[0m"
                                        );
                                    }
                                    // List what the new endpoint serves, so the
                                    // user can immediately `/model` to pick.
                                    match agent.list_models().await {
                                        Ok(mut models) if !models.is_empty() => {
                                            models.sort_by(|a, b| a.id.cmp(&b.id));
                                            println!("\x1b[2mmodels served by {label}:\x1b[0m");
                                            for m in &models {
                                                let mark =
                                                    if m.id == agent.model() { "▶" } else { " " };
                                                println!("  {mark} {}", m.id);
                                            }
                                            println!("\x1b[2m/model <id> to switch\x1b[0m");
                                        }
                                        _ => {
                                            let local = matches!(
                                                new_settings.provider,
                                                config::ProviderName::Ollama
                                            );
                                            if local {
                                                println!(
                                                    "\x1b[2m(couldn't reach the local server — is it running and is the base_url correct? /model <id> to switch)\x1b[0m"
                                                );
                                            } else {
                                                println!(
                                                    "\x1b[2m(couldn't list endpoint models; /model <id> to switch)\x1b[0m"
                                                );
                                            }
                                        }
                                    }
                                }
                                Err(err) => {
                                    eprintln!("\x1b[33m/provider failed: {err}\x1b[0m");
                                }
                            }
                            continue;
                        }
                        Command::Mcp => {
                            let Some(url) = settings.mcp_url.as_deref() else {
                                eprintln!("\x1b[33mno MCP URL configured for this provider\x1b[0m");
                                continue;
                            };
                            match crate::mcp_inspect(url, &settings.api_key, &settings.model).await
                            {
                                Ok(report) => print!("{report}"),
                                Err(err) => {
                                    eprintln!("\x1b[33mmcp inspection failed: {err:#}\x1b[0m")
                                }
                            }
                            continue;
                        }
                        Command::Lsp(arg) => {
                            crate::commands::handle_lsp(agent, &arg);
                            continue;
                        }
                        other => {
                            handle_command(agent, other, registry);
                            continue;
                        }
                    }
                } else {
                    line
                };

                // Run the turn with an animated "working… Ns" spinner so it's
                // always clear something is happening. Ctrl-C cancels the turn.
                last_prompt = Some(input.clone());
                let checkpoint = agent.messages().len();
                last_turn_start = checkpoint;
                let turn_snapshot = agent.state_snapshot();
                last_turn_snapshot = Some(turn_snapshot.clone());
                let background_before = hi_tools::background_process_ids();
                let progress = Arc::new(AtomicBool::new(false));
                let cancelled = {
                    let mut plain = PlainUi::with_progress(progress.clone());
                    drive_with_spinner(agent.run_turn(&input, &mut plain), &progress).await
                };
                if cancelled {
                    if let Err(err) = agent.rewind_to_snapshot_durable(checkpoint, &turn_snapshot) {
                        eprintln!(
                            "\x1b[33mcouldn't persist interrupted turn discard: {err:#}\x1b[0m"
                        );
                        agent.truncate_messages(checkpoint);
                        agent.restore_state_snapshot(&turn_snapshot);
                    }
                    let killed =
                        hi_tools::kill_background_processes_started_after(&background_before);
                    if killed > 0 {
                        println!(
                            "\x1b[33m^C — interrupted; turn discarded; killed {killed} background process(es) started by it\x1b[0m"
                        );
                    } else {
                        println!("\x1b[33m^C — interrupted; turn discarded\x1b[0m");
                    }
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
        let _ = drive_with_spinner(memory, &progress).await;
    }

    // Don't leave background processes (dev servers, watchers) running after
    // the session ends.
    hi_tools::kill_background_processes();

    if let Some(path) = &history {
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let _ = editor.save_history(path);
    }
    Ok(())
}

/// Drive a model future (a turn or a compaction) to completion, showing an
/// animated spinner until the first output and letting Ctrl-C cancel it.
/// Returns whether it was cancelled.
async fn drive_with_spinner(
    fut: impl std::future::Future<Output = Result<()>>,
    progress: &AtomicBool,
) -> bool {
    use std::io::Write;

    tokio::pin!(fut);
    let started = std::time::Instant::now();
    let mut ticker = tokio::time::interval(Duration::from_millis(90));
    let mut frame = 0usize;
    let mut cancelled = false;
    loop {
        tokio::select! {
            result = &mut fut => {
                if let Err(err) = result {
                    let (kind, guidance) = hi_agent::classify_error(&err);
                    let suffix = if guidance.is_empty() {
                        String::new()
                    } else {
                        format!(" — {guidance}")
                    };
                    eprintln!("\r\x1b[K\x1b[31m{kind}: {err:#}{suffix}\x1b[0m");
                }
                break;
            }
            _ = tokio::signal::ctrl_c() => { cancelled = true; break; }
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
    cancelled
}

/// Read a line from the user with a prompt, using rustyline for line editing.
fn rl_prompt(editor: &mut crate::complete::ReplEditor, message: &str) -> Result<String> {
    Ok(editor.readline(message)?.trim().to_string())
}

/// Interactively create a new profile via line prompts and save it to the
/// config file. Returns the profile name.
fn provider_add_prompt(
    config: &mut config::Config,
    editor: &mut crate::complete::ReplEditor,
) -> Result<String> {
    use config::{ProfileForm, ProviderName, upsert_profile, writable_config_path};

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
    println!("  1) pipenetwork.ai    2) Ollama (local)");
    let provider = loop {
        match rl_prompt(editor, "Provider [1-2] (default 1): ")?.as_str() {
            "" | "1" => break ProviderName::Pipenetwork,
            "2" => break ProviderName::Ollama,
            other => eprintln!("  '{other}' isn't a choice — pick 1-2."),
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

    // Model (optional — can pick via /model after switching).
    let default_model = provider.default_model().unwrap_or("");
    let model = if default_model.is_empty() {
        rl_prompt(editor, "Model id (optional — blank to pick via /model): ")?
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

    let path = writable_config_path(None).context("could not determine config path")?;
    upsert_profile(config, &name, profile, &path)?;
    Ok(name)
}

/// Interactively edit an existing profile. `name` may be empty to prompt for it.
fn provider_edit_prompt(
    config: &mut config::Config,
    name: &str,
    editor: &mut crate::complete::ReplEditor,
) -> Result<String> {
    use config::{ProfileForm, ProviderName, upsert_profile, writable_config_path};

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
        "  current: {} (1=pipenetwork.ai 2=Ollama)",
        form.provider.as_str()
    );
    let provider = loop {
        let input = rl_prompt(editor, "Provider [1-2]: ")?;
        if input.is_empty() {
            break form.provider;
        }
        match input.as_str() {
            "1" => break ProviderName::Pipenetwork,
            "2" => break ProviderName::Ollama,
            _ => eprintln!("  pick 1-2"),
        }
    };
    form.provider = provider;

    // API key.
    let key_label = if form.store_as_env { "env var" } else { "key" };
    let masked = if form.api_key.len() > 8 {
        format!(
            "{}…{}",
            &form.api_key[..4],
            &form.api_key[form.api_key.len() - 4..]
        )
    } else if form.api_key.is_empty() {
        "(none)".to_string()
    } else {
        "***".to_string()
    };
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

    let mut profile = form.to_profile();
    if profile.mcp_url.is_none() {
        profile.mcp_url = existing.mcp_url.clone();
    }
    let path = writable_config_path(None).context("could not determine config path")?;
    upsert_profile(config, &name, profile, &path)?;
    Ok(name)
}
