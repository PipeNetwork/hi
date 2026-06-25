//! The plain line REPL loop and the animated-spinner turn driver.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use anyhow::{Context, Result};
use hi_agent::Agent;
use hi_ai::Registry;

use crate::commands::handle_command;
use crate::config::Settings;
use crate::ui::PlainUi;
use crate::{provider_label, session};

const SPINNER: [&str; 10] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

pub(crate) async fn repl(
    agent: &mut Agent,
    settings: &Settings,
    registry: &Registry,
    auto_memory: bool,
) -> Result<()> {
    use hi_agent::Command;
    use hi_agent::CompactionKind;
    use rustyline::DefaultEditor;
    use rustyline::error::ReadlineError;

    let window = registry.metadata(&settings.model).1
        .map(|w| format!(" · {}k ctx", w / 1000))
        .unwrap_or_default();
    println!(
        "hi · {} · {}{} — /help for commands, Ctrl-D to quit.",
        provider_label(settings.provider),
        settings.model,
        window,
    );

    let mut editor = DefaultEditor::new().context("initializing line editor")?;
    let history = session::history_path();
    if let Some(path) = &history {
        let _ = editor.load_history(path);
    }

    // For `/retry`: the last message sent, and the history length just before
    // that turn (so we can drop it before re-running).
    let mut last_prompt: Option<String> = None;
    let mut last_turn_start = 0usize;

    loop {
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
                        Command::Retry => match last_prompt.clone() {
                            Some(prompt) => {
                                agent.truncate_messages(last_turn_start);
                                println!("\x1b[2mretrying: {prompt}\x1b[0m");
                                prompt
                            }
                            None => {
                                println!("\x1b[2mnothing to retry yet\x1b[0m");
                                continue;
                            }
                        },
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
                            let preview: String = diff.lines().take(20).collect::<Vec<_>>().join("\n");
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
                let progress = Arc::new(AtomicBool::new(false));
                let cancelled = {
                    let mut plain = PlainUi::with_progress(progress.clone());
                    drive_with_spinner(agent.run_turn(&input, &mut plain), &progress).await
                };
                if cancelled {
                    agent.truncate_messages(checkpoint);
                    println!("\x1b[33m^C — interrupted; turn discarded\x1b[0m");
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
                    let category = hi_ai::provider_error_kind(&err)
                        .map(|k| k.as_str())
                        .unwrap_or("error");
                    let hint = match category {
                        "auth" => " — check your API key (run `hi --show-config`)",
                        "rate_limit" => " — rate limited; wait a moment and /retry",
                        "server_error" => " — provider is having issues; try /retry",
                        "network" => " — can't reach the endpoint; check your connection",
                        _ => "",
                    };
                    eprintln!("\r\x1b[K\x1b[31m{category}: {err:#}{hint}\x1b[0m");
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
