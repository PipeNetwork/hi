//! Shared implementations for session-orchestration slash commands
//! (`/plan`, `/fork`, `/rewind`, `/remember`, `/recap`, …).

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use serde::Serialize;

use crate::Command;
use crate::memory::{global_memory_file, memory_file_at, read_global_memory};
use hi_ai::{Message, Role};

fn msg_text(msg: &Message) -> String {
    msg.text()
}

fn canonical_workspace(workspace: &Path) -> PathBuf {
    hi_tools::folder_trust::workspace_key(workspace)
}

pub fn workspace_trusted(workspace: &Path) -> bool {
    // Every build requires a persisted grant. The only bypass is the explicit
    // operator setting, which `/trust status` surfaces below.
    hi_tools::folder_trust::folder_trust_inert()
        || hi_tools::folder_trust::folder_trust_granted(workspace)
}

pub fn set_workspace_trusted(workspace: &Path, trusted: bool) -> Result<()> {
    if trusted {
        hi_tools::folder_trust::grant_folder_trust(workspace)?;
    } else {
        let _ = hi_tools::folder_trust::try_revoke_folder_trust(workspace)?;
        if hi_tools::folder_trust::folder_trust_granted(workspace) {
            anyhow::bail!(
                "workspace trust is inherited from a trusted ancestor; revoke that ancestor instead"
            );
        }
    }
    Ok(())
}

pub fn trust_command(workspace: &Path, arg: &str) -> String {
    match arg.trim() {
        "" | "status" => {
            let store = hi_tools::folder_trust::trust_store_file()
                .map(|path| path.display().to_string())
                .unwrap_or_else(|| "<unavailable; trust fails closed>".to_string());
            format!(
                "folder trust: {}\n  policy: {}\n  workspace: {}\n  store: {store}",
                if workspace_trusted(workspace) {
                    "trusted"
                } else {
                    "untrusted"
                },
                hi_tools::folder_trust::policy_description(),
                canonical_workspace(workspace).display(),
            )
        }
        "on" | "grant" | "trust" => match set_workspace_trusted(workspace, true) {
            Ok(()) => format!(
                "trusted workspace: {}",
                canonical_workspace(workspace).display()
            ),
            Err(error) => format!("trust failed: {error:#}"),
        },
        "off" | "revoke" | "untrust" => match set_workspace_trusted(workspace, false) {
            Ok(()) => format!(
                "revoked workspace trust: {}",
                canonical_workspace(workspace).display()
            ),
            Err(error) => format!("trust revoke failed: {error:#}"),
        },
        _ => "usage: /trust [status|on|off]".into(),
    }
}

/// Result of handling a session-orchestration slash command.
#[derive(Debug, Clone)]
pub struct SessionCommandEffect {
    /// Text to show the user.
    pub message: String,
    /// Optional follow-up user prompt that should run as a model turn (plan mode).
    pub follow_up_prompt: Option<String>,
}

/// Execute a session command through the same admission, reconciliation, and
/// publication boundary used by tools. Read-only and in-memory commands keep
/// the synchronous fast path; commands which can touch workspace or external
/// state are never run until the always-present controller admits them.
pub async fn handle_session_command_coordinated(
    agent: &mut crate::Agent,
    command: &Command,
    frontend_queue: &[String],
) -> Option<SessionCommandEffect> {
    if !session_command_mutates_workspace(command) {
        return handle_session_command_inner(agent, command, frontend_queue, false);
    }

    let name = command_name(command);
    let intent = hi_workspace::MutationIntent {
        effect_scope: hi_workspace::EffectScope::LiveWriter,
        replay_class: session_command_replay_class(command),
        dirty_paths: None,
        description: Some(format!("session command: /{name}")),
    };
    if let Err(error) = agent.begin_classified_workspace_operation(intent).await {
        return Some(SessionCommandEffect {
            message: format!(
                "/{name} was not run because the workspace controller refused admission: {error:#}"
            ),
            follow_up_prompt: None,
        });
    }

    let ledger_revision = agent.runtime.ledger().revision();
    let handled = handle_session_command_inner(agent, command, frontend_queue, true);
    let raw_result = handled
        .as_ref()
        .map(|effect| effect.message.clone())
        .unwrap_or_else(|| format!("/{name} was not handled"));
    let reconciliation = agent.reconcile_workspace_changes().await;
    let changes = if reconciliation.is_ok() {
        agent.runtime.ledger().changes_since(ledger_revision)
    } else {
        Vec::new()
    };
    let command_disposition = handled
        .as_ref()
        .map_or(hi_workspace::ExecutionDisposition::Failed, |effect| {
            session_command_execution_disposition(command, &effect.message)
        });
    let mut execution = hi_workspace::ExecutionReport {
        disposition: if reconciliation.is_err() {
            hi_workspace::ExecutionDisposition::Indeterminate
        } else {
            command_disposition
        },
        workspace_may_have_changed: reconciliation.is_err() || !changes.is_empty(),
        // Admission precedes the synchronous helper. Even a failure string can
        // follow a partially applied external operation, so settlement records
        // that execution may have happened.
        external_effect_may_have_occurred: true,
        content_digest: reconciliation
            .is_ok()
            .then(|| agent.runtime.ledger().workspace_revision()),
        changed_paths: changes
            .iter()
            .map(|change| PathBuf::from(&change.path))
            .collect(),
        artifacts: Vec::new(),
        detail: reconciliation.as_ref().err().map_or_else(
            || {
                (command_disposition != hi_workspace::ExecutionDisposition::Succeeded)
                    .then(|| format!("/{name} execution failed: {raw_result}"))
            },
            |error| {
                Some(format!(
                    "/{name} effects could not be reconciled: {error:#}"
                ))
            },
        ),
    };

    let operation_id = agent.workspace_coordination.active_parent_operation();
    let call_id = operation_id.map_or_else(
        || format!("session-command:{}", uuid::Uuid::new_v4()),
        |operation| format!("session-command:{operation}"),
    );
    let arguments = serde_json::json!({
        "command": name,
        "parsed": format!("{command:?}"),
    })
    .to_string();
    let calls = [(
        call_id.clone(),
        "session_command".to_owned(),
        arguments.clone(),
    )];
    let assistant_content = [hi_ai::Content::ToolCall {
        id: call_id.clone(),
        name: "session_command".to_owned(),
        arguments,
    }];
    let results = [(call_id, raw_result.clone())];
    let stage_error = agent
        .stage_active_workspace_execution(&calls, &assistant_content, &results, &execution)
        .err();
    if let Some(error) = &stage_error {
        execution.disposition = hi_workspace::ExecutionDisposition::Indeterminate;
        execution.content_digest = None;
        execution.detail = Some(match execution.detail.take() {
            Some(detail) => {
                format!("{detail}; /{name} transcript evidence could not be staged: {error:#}")
            }
            None => format!("/{name} transcript evidence could not be staged: {error:#}"),
        });
    }
    let settlement = agent
        .checkpoint_durable_workspace_with_execution(execution)
        .await;

    match (handled, reconciliation.err(), stage_error, settlement) {
        (Some(effect), None, None, Ok(())) => Some(effect),
        (effect, reconcile, stage, settle) => {
            let mut failures = Vec::new();
            if effect.is_none() {
                failures.push("the command was not handled".to_owned());
            }
            if let Some(error) = reconcile {
                failures.push(format!("workspace reconciliation failed: {error:#}"));
            }
            if let Some(error) = stage {
                failures.push(format!("transcript staging failed: {error:#}"));
            }
            if let Err(error) = settle {
                failures.push(format!("durable settlement failed: {error:#}"));
            }
            Some(SessionCommandEffect {
                message: format!(
                    "/{name} ran, but its result was not published as successful: {}\nheld result: {raw_result}",
                    failures.join("; ")
                ),
                follow_up_prompt: None,
            })
        }
    }
}

/// Handle session-orchestration commands against a live agent.
/// Returns `None` when `command` is not one of these.
pub fn handle_session_command(
    agent: &mut crate::Agent,
    command: &Command,
    frontend_queue: &[String],
) -> Option<SessionCommandEffect> {
    handle_session_command_inner(agent, command, frontend_queue, false)
}

fn handle_session_command_inner(
    agent: &mut crate::Agent,
    command: &Command,
    frontend_queue: &[String],
    coordinated: bool,
) -> Option<SessionCommandEffect> {
    // These commands use synchronous filesystem helpers.  A materialized
    // PipeFS workspace must never be changed outside the async durability
    // fence, because a successful-looking command could otherwise leave bytes
    // behind that have not reached the remote revision.  Keep read-only forms
    // available, but fail closed for the forms that write the workspace (or
    // create/remove a worktree from it) until they can participate in that
    // fence.
    let pipefs_authoritative = matches!(
        agent.workspace_controller_binding().authority,
        hi_workspace::WorkspaceAuthority::PipeFs { .. }
    );
    if pipefs_authoritative && session_command_mutates_workspace(command) && !coordinated {
        return Some(SessionCommandEffect {
            message: format!(
                "/{} is unavailable while PipeFS is active because it cannot yet commit through the workspace durability fence; use normal file tools or /pipefs off first",
                command_name(command)
            ),
            follow_up_prompt: None,
        });
    }
    let message = match command {
        Command::ViewPlan => format_plan(agent.current_plan()),
        Command::Plan(arg) => {
            let arg = arg.trim();
            match arg {
                "off" | "exit" | "done" => {
                    agent.set_plan_mode(false);
                    "plan mode off — normal editing resumed".into()
                }
                "pause" => match agent.try_set_plan_drive_paused(true) {
                    Ok(_) => "plan drive paused — checklist stays pinned; /plan resume to continue"
                        .into(),
                    Err(error) => format!("plan pause failed: {error:#}"),
                },
                "resume" => {
                    if let Err(error) = agent.resume_plan_drive() {
                        return Some(SessionCommandEffect {
                            message: format!("plan resume failed: {error:#}"),
                            follow_up_prompt: None,
                        });
                    }
                    if agent.plan_approval_parked() {
                        return Some(SessionCommandEffect {
                            message: "plan approval is parked — /view-plan to review it".into(),
                            follow_up_prompt: None,
                        });
                    }
                    if agent.plan_incomplete() && !agent.plan_mode() {
                        return Some(SessionCommandEffect {
                            message: "plan drive resumed".into(),
                            follow_up_prompt: Some(crate::PLAN_DRIVE_PROMPT.to_string()),
                        });
                    }
                    "plan drive resumed — no leftover checklist work".into()
                }
                "clear" => {
                    agent.clear_pinned_plan();
                    "plan cleared".into()
                }
                "replace" => {
                    agent.clear_pinned_plan();
                    "plan cleared — next task should update_plan a new checklist".into()
                }
                "show" | "view" | "list" | "status" => {
                    let mode = if agent.plan_mode() { "on" } else { "off" };
                    format!(
                        "plan mode: {mode}\nplan drive: {}\n{}",
                        agent.plan_drive_status(),
                        format_plan(agent.current_plan())
                    )
                }
                "" | "on" => {
                    agent.set_plan_mode(true);
                    return Some(SessionCommandEffect {
                        message: "plan mode on — draft a plan without mutating the tree".into(),
                        // Queue human task text; `run_turn_core` owns the
                        // ephemeral mode wrapper so classifiers and durable
                        // task state never mistake control prose for the task.
                        follow_up_prompt: Some(
                            "Draft a clear implementation plan for the current task.".into(),
                        ),
                    });
                }
                request => {
                    agent.set_plan_mode(true);
                    return Some(SessionCommandEffect {
                        message: format!("plan mode on — planning: {request}"),
                        follow_up_prompt: Some(request.to_string()),
                    });
                }
            }
        }
        Command::Fork(arg) => {
            let (want_wt, directive) = parse_fork_args(arg);
            let wt = if want_wt {
                match fork_worktree(agent.workspace_root(), directive.trim()) {
                    Ok(p) => Some(p),
                    Err(err) => {
                        return Some(SessionCommandEffect {
                            message: format!(
                                "fork worktree failed: {err:#}\n{}",
                                fork_summary(None, &directive)
                            ),
                            follow_up_prompt: None,
                        });
                    }
                }
            } else {
                None
            };
            let mut summary = fork_summary(wt.as_deref(), &directive);
            let fork_dir = agent.workspace_root().join(".hi/forks");
            let fork_path = fork_dir.join(format!("session-{}.md", chrono_like_stamp()));
            match std::fs::create_dir_all(&fork_dir)
                .and_then(|_| std::fs::write(&fork_path, agent.export_markdown()))
            {
                Ok(()) => summary.push_str(&format!(
                    "  session context: {}\n  peer prompt: hi --prompt-file {} (or open and paste)\n",
                    fork_path.display(), fork_path.display()
                )),
                Err(error) => summary.push_str(&format!("  session context export failed: {error}\n")),
            }
            summary
        }
        Command::Rewind(arg) => {
            let arg = arg.trim();
            if arg.is_empty() || arg == "list" || arg == "ls" {
                format_user_turns(&list_user_turns(agent.messages()), 40)
            } else if let Ok(n) = arg.parse::<usize>() {
                match agent.rewind_to_user_turn(n) {
                    Ok(msg) => msg,
                    Err(err) => format!("rewind failed: {err:#}"),
                }
            } else {
                format!("usage: /rewind [n] — got {arg:?}")
            }
        }
        Command::Permissions(arg) => apply_permissions(agent, arg),
        Command::AlwaysApprove(arg) => {
            let a = arg.trim();
            if a.is_empty() || a == "on" {
                apply_permissions(agent, "always")
            } else if a == "off" {
                apply_permissions(agent, "ask")
            } else {
                apply_permissions(agent, a)
            }
        }
        Command::Auto(arg) => {
            let a = arg.trim();
            if a.is_empty() || a == "on" {
                apply_permissions(agent, "auto")
            } else if a == "off" {
                apply_permissions(agent, "ask")
            } else {
                apply_permissions(agent, a)
            }
        }
        Command::Queue(_) | Command::Tasks(_) => {
            let extras: Vec<String> = frontend_queue
                .iter()
                .map(|s| collapse_preview(s, 80))
                .collect();
            format_tasks_report(
                &agent.background_process_ids(),
                &agent.background_task_ids(),
                agent.checkpoint_count(),
                agent.plan_mode(),
                agent.permission_mode(),
                &extras,
            )
        }
        Command::Plugins(_) => plugins_and_hooks_report(agent.workspace_root()),
        Command::Remember(arg) => {
            let (global, text) = parse_remember_args(arg);
            match remember_note(agent.workspace_root(), &text, global) {
                Ok(msg) => msg,
                Err(err) => format!("remember failed: {err:#}"),
            }
        }
        Command::UndoMemory => match crate::memory::undo_memory(agent.workspace_root()) {
            Ok(msg) => msg,
            Err(err) => format!("undo-memory: {err}"),
        },
        Command::Memory => format_memory_dump(agent.workspace_root()),
        Command::ImportClaude(_) => import_claude_report(agent.workspace_root()),
        Command::Hooks(arg) => hooks_command(agent.workspace_root(), arg),
        Command::Trust(arg) => {
            let message = trust_command(agent.workspace_root(), arg);
            update_workspace_context_trust(agent, arg, &message);
            message
        }
        Command::Marketplace(arg) => marketplace_report(agent.workspace_root(), arg),
        Command::Worktree(arg) => worktree_command(agent.workspace_root(), arg),
        Command::Inspect(arg) => {
            let a = arg.trim();
            if matches!(a, "bundle" | "support-bundle") {
                let report = inspect_report(agent, true);
                let path = agent.workspace_root().join(".hi/inspect-report.json");
                match path
                    .parent()
                    .map(std::fs::create_dir_all)
                    .transpose()
                    .and_then(|_| std::fs::write(&path, report.as_bytes()))
                {
                    Ok(()) => format!("wrote redacted inspect bundle: {}", path.display()),
                    Err(error) => format!("inspect bundle failed: {error}"),
                }
            } else {
                inspect_report(agent, matches!(a, "json" | "--json" | "-j"))
            }
        }
        Command::Engine(arg) => agent.engine_command(arg),
        Command::Agents(arg) => agents_report(agent.workspace_root(), arg),
        Command::Share(arg) => share_report(agent, arg),
        Command::McpAdmin(arg) => mcp_admin_report(arg),
        Command::RewindPicker => format_user_turns(&list_user_turns(agent.messages()), 60),
        Command::Cd(arg) => {
            let arg = arg.trim();
            if arg.is_empty() {
                format!("workspace: {}", agent.workspace_root().display())
            } else {
                let path = PathBuf::from(arg);
                let path = if path.is_absolute() {
                    path
                } else {
                    agent.workspace_root().join(path)
                };
                if !path.is_dir() {
                    format!("workspace does not exist: {}", path.display())
                } else {
                    let hint = agent.workspace_root().join(".hi/dashboard-cwd");
                    match hint
                        .parent()
                        .map(std::fs::create_dir_all)
                        .transpose()
                        .and_then(|_| std::fs::write(&hint, path.to_string_lossy().as_bytes()))
                    {
                        Ok(()) => format!(
                            "dashboard workspace → {}\n(saved {}; start/fork a session there for live agent cwd)",
                            path.display(),
                            hint.display()
                        ),
                        Err(error) => format!("could not save dashboard cwd: {error}"),
                    }
                }
            }
        }
        Command::Rename(arg) => format!(
            "session rename requested: {}\n(use /sessions rename <id> <name> in the current frontend)",
            if arg.trim().is_empty() {
                "<name>"
            } else {
                arg.trim()
            }
        ),
        Command::Resume(arg) => format!(
            "resume requested: {}\n(use /sessions switch <id> or `hi --resume <id>`)",
            if arg.trim().is_empty() {
                "list sessions with /sessions"
            } else {
                arg.trim()
            }
        ),
        Command::ScreenMode(_)
        | Command::VimMode(_)
        | Command::Multiline(_)
        | Command::Timeline(_)
        | Command::Timestamps(_) => {
            "this display preference is handled by the active frontend".into()
        }
        Command::Recap => local_recap(agent.messages()),
        Command::Metrics => crate::learning::render_report(agent.runtime.state_root()),
        Command::SynthEvals => {
            return match crate::learning::synth_evals_prompt(agent.runtime.state_root()) {
                Some((signatures, prompt)) => Some(SessionCommandEffect {
                    message: format!(
                        "synthesizing eval drafts for {signatures} failure signature(s) — queued as the next turn"
                    ),
                    follow_up_prompt: Some(prompt),
                }),
                None => Some(SessionCommandEffect {
                    message: "no unprocessed findings — nothing to synthesize".into(),
                    follow_up_prompt: None,
                }),
            };
        }
        Command::Find(arg) => search_messages(agent.messages(), arg),
        Command::Jump(arg) | Command::History(arg) => {
            let arg = arg.trim();
            if arg.is_empty() {
                format_user_turns(&list_user_turns(agent.messages()), 40)
            } else if let Ok(n) = arg.parse::<usize>() {
                let turns = list_user_turns(agent.messages());
                match turns.iter().find(|t| t.n == n) {
                    Some(t) => format!(
                        "jump {}: {}\n(conversation not modified — /rewind {} to truncate)",
                        t.n, t.preview, t.n
                    ),
                    None => format!("no user turn {n}"),
                }
            } else {
                let turns = list_user_turns(agent.messages());
                let q = arg.to_ascii_lowercase();
                let hits: Vec<_> = turns
                    .iter()
                    .filter(|t| t.preview.to_ascii_lowercase().contains(&q))
                    .collect();
                if hits.is_empty() {
                    format!("no prompts matching {arg:?}")
                } else {
                    let mut out = format!("history matching {arg:?}:\n");
                    for t in hits.iter().rev().take(20).rev() {
                        out.push_str(&format!("  {:>3}. {}\n", t.n, t.preview));
                    }
                    out
                }
            }
        }
        _ => return None,
    };
    Some(SessionCommandEffect {
        message,
        follow_up_prompt: None,
    })
}

fn session_command_mutates_workspace(command: &Command) -> bool {
    match command {
        // `/fork` always writes a transcript export and may create a worktree.
        Command::Fork(_) | Command::Remember(_) | Command::UndoMemory => true,
        Command::Marketplace(arg) => arg.trim_start().starts_with("install "),
        Command::Worktree(arg) => {
            matches!(arg.trim(), "gc" | "clean" | "cleanup")
                || arg.trim_start().starts_with("remove ")
        }
        Command::Inspect(arg) => matches!(arg.trim(), "bundle" | "support-bundle"),
        Command::Agents(arg) => {
            let arg = arg.trim_start();
            arg.starts_with("add ") || arg.starts_with("remove ")
        }
        // `/share` always materializes a local bundle before optionally
        // printing a portal URL.
        Command::Share(_) => true,
        // `/cd` saves a dashboard cwd hint below `.hi/`.
        Command::Cd(arg) => !arg.trim().is_empty(),
        // A portable restore starts untrusted.  Do not let a command grant a
        // trust record to the transient materialization.
        Command::Trust(arg) => matches!(
            arg.trim(),
            "on" | "grant" | "trust" | "off" | "revoke" | "untrust"
        ),
        _ => false,
    }
}

fn session_command_replay_class(command: &Command) -> hi_workspace::ReplayClass {
    // Grant/revoke are idempotent local policy writes. Classifying them as
    // such lets a compatibility PipeFS session publish a no-head-change
    // transcript receipt; all other compatibility helpers remain fenced as
    // non-replayable external effects.
    if matches!(
        command,
        Command::Trust(arg)
            if matches!(arg.trim(), "on" | "grant" | "trust" | "off" | "revoke" | "untrust")
    ) {
        return hi_workspace::ReplayClass::IdempotentExternal {
            key: hi_workspace::IdempotencyKey::new(uuid::Uuid::new_v4().to_string()),
        };
    }
    hi_workspace::ReplayClass::NonReplayableExternal
}

fn update_workspace_context_trust(agent: &mut crate::Agent, arg: &str, message: &str) {
    let promote =
        matches!(arg.trim(), "on" | "grant" | "trust") && message.starts_with("trusted workspace:");
    let demote = matches!(arg.trim(), "off" | "revoke" | "untrust")
        && message.starts_with("revoked workspace trust:");
    if !promote && !demote {
        return;
    }
    let context = agent.config.memory.project_context.clone().map(|context| {
        if promote {
            crate::promote_repository_context(context)
        } else {
            crate::mark_repository_context_untrusted(context)
        }
    });
    agent.set_workspace_project_context(context);
}

fn session_command_execution_disposition(
    command: &Command,
    message: &str,
) -> hi_workspace::ExecutionDisposition {
    let succeeded = match command {
        Command::Fork(_) => !message.contains(" failed:"),
        Command::Remember(_) => !message.starts_with("remember failed:"),
        Command::UndoMemory => !message.starts_with("undo-memory:"),
        Command::Marketplace(_) => message.starts_with("installed plugin skill:"),
        Command::Worktree(arg) if matches!(arg.trim(), "gc" | "clean" | "cleanup") => {
            message.starts_with("cleaned ")
        }
        Command::Worktree(arg) if arg.trim_start().starts_with("remove ") => {
            message.starts_with("removed ")
        }
        Command::Inspect(_) => message.starts_with("wrote redacted inspect bundle:"),
        Command::Agents(arg) if arg.trim_start().starts_with("add ") => {
            message.starts_with("added agent persona:")
        }
        Command::Agents(arg) if arg.trim_start().starts_with("remove ") => {
            message.starts_with("removed ")
        }
        Command::Share(_) => {
            message.starts_with("share bundle:") || message.starts_with("{\"path\":")
        }
        Command::Cd(_) => message.starts_with("dashboard workspace "),
        Command::Trust(arg) if matches!(arg.trim(), "on" | "grant" | "trust") => {
            message.starts_with("trusted workspace:")
        }
        Command::Trust(arg) if matches!(arg.trim(), "off" | "revoke" | "untrust") => {
            message.starts_with("revoked workspace trust:")
        }
        _ => false,
    };
    if succeeded {
        hi_workspace::ExecutionDisposition::Succeeded
    } else {
        hi_workspace::ExecutionDisposition::Failed
    }
}

fn command_name(command: &Command) -> &'static str {
    match command {
        Command::Fork(_) => "fork",
        Command::Remember(_) => "remember",
        Command::UndoMemory => "undo-memory",
        Command::Marketplace(_) => "marketplace install",
        Command::Worktree(_) => "worktree",
        Command::Inspect(_) => "inspect bundle",
        Command::Agents(_) => "agents",
        Command::Share(_) => "share",
        Command::Cd(_) => "cd",
        Command::Trust(_) => "trust",
        _ => "command",
    }
}

fn apply_permissions(agent: &mut crate::Agent, arg: &str) -> String {
    let arg = arg.trim();
    if arg.is_empty() || arg == "status" {
        return format!(
            "permissions: {} (confirm_edits flows from this ladder)",
            agent.permission_mode().as_str()
        );
    }
    match PermissionMode::parse(arg) {
        Some(mode) => {
            agent.set_permission_mode(mode);
            format!("permissions → {}", mode.as_str())
        }
        None => format!(
            "usage: /permissions [ask|auto|always] — unknown {arg:?}; current={}",
            agent.permission_mode().as_str()
        ),
    }
}

/// In-session permission ladder (mirrors grok `/always-approve` / `/auto` / ask).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PermissionMode {
    /// Confirm each write/edit when the frontend supports it.
    Ask,
    /// Skip confirmations for routine edits; still block catastrophic shell ops.
    Auto,
    /// Full YOLO: no edit confirms, allow missing checkpoints.
    #[default]
    Always,
}

impl PermissionMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Ask => "ask",
            Self::Auto => "auto",
            Self::Always => "always",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "" | "status" => None,
            "ask" | "confirm" | "on" => Some(Self::Ask),
            "auto" | "safe" => Some(Self::Auto),
            "always" | "yolo" | "off" | "never" => Some(Self::Always),
            _ => None,
        }
    }
}

/// One user turn anchor for `/rewind`, `/jump`, `/history`.
#[derive(Clone, Debug, Serialize)]
pub struct UserTurn {
    /// 1-based index among user turns (oldest = 1).
    pub n: usize,
    /// Index into the full messages vec (including system).
    pub message_index: usize,
    pub preview: String,
}

pub fn list_user_turns(messages: &[Message]) -> Vec<UserTurn> {
    let mut out = Vec::new();
    for (message_index, msg) in messages.iter().enumerate() {
        if msg.role != Role::User {
            continue;
        }
        let text = msg_text(msg);
        if text.trim().is_empty() {
            continue;
        }
        let preview = collapse_preview(text.trim(), 96);
        out.push(UserTurn {
            n: out.len() + 1,
            message_index,
            preview,
        });
    }
    out
}

pub fn format_user_turns(turns: &[UserTurn], limit: usize) -> String {
    if turns.is_empty() {
        return "no user turns yet".into();
    }
    let start = turns.len().saturating_sub(limit);
    let mut out = String::from("user turns (oldest → newest):\n");
    for t in &turns[start..] {
        out.push_str(&format!("  {:>3}. {}\n", t.n, t.preview));
    }
    if start > 0 {
        out.push_str(&format!("  … {} earlier turn(s) omitted\n", start));
    }
    out.push_str("use /rewind <n> to truncate conversation to just before that turn\n");
    out
}

/// Truncate so the chosen user turn becomes the last message kept *before* it
/// is dropped — i.e. rewind discards that turn and everything after.
pub fn rewind_len_before_user_turn(messages: &[Message], turn_n: usize) -> Result<usize> {
    let turns = list_user_turns(messages);
    let Some(t) = turns.iter().find(|t| t.n == turn_n) else {
        bail!(
            "no user turn {turn_n} (have {})",
            turns.last().map(|t| t.n).unwrap_or(0)
        );
    };
    Ok(t.message_index)
}

pub fn search_messages(messages: &[Message], query: &str) -> String {
    let q = query.trim();
    if q.is_empty() {
        return "usage: /find <text>".into();
    }
    let needle = q.to_ascii_lowercase();
    let mut hits = Vec::new();
    for (i, msg) in messages.iter().enumerate() {
        if msg.role == Role::System {
            continue;
        }
        let text = msg_text(msg);
        if text.to_ascii_lowercase().contains(&needle) {
            let role = match msg.role {
                Role::User => "user",
                Role::Assistant => "assistant",
                Role::Tool => "tool",
                Role::System => "system",
            };
            hits.push(format!(
                "  msg#{i} [{role}] {}",
                collapse_preview(&text, 120)
            ));
        }
    }
    if hits.is_empty() {
        format!("no matches for {q:?}")
    } else {
        let total = hits.len();
        let shown: Vec<_> = hits.into_iter().rev().take(30).collect();
        let mut out = format!("find {q:?} — {total} hit(s), newest first:\n");
        for line in shown.into_iter().rev() {
            out.push_str(&line);
            out.push('\n');
        }
        out
    }
}

/// Local side-channel recap (does not mutate history or call the model).
pub fn local_recap(messages: &[Message]) -> String {
    let turns = list_user_turns(messages);
    if turns.is_empty() {
        return "recap: empty session — nothing to summarize.".into();
    }
    let mut out = String::from("recap (local, not added to history):\n");
    out.push_str(&format!("  user turns: {}\n", turns.len()));
    let recent = turns.len().saturating_sub(5);
    out.push_str("  recent prompts:\n");
    for t in &turns[recent..] {
        out.push_str(&format!("    • {}\n", t.preview));
    }
    // Last assistant snippet.
    if let Some(a) = messages.iter().rev().find(|m| m.role == Role::Assistant) {
        let text = msg_text(a);
        if !text.trim().is_empty() {
            out.push_str("  last assistant:\n");
            out.push_str(&format!("    {}\n", collapse_preview(text.trim(), 240)));
        }
    }
    out.push_str("  tip: /compact hybrid shrinks context; /rewind <n> drops later turns\n");
    out
}

pub fn format_plan(steps: &[hi_tools::PlanStep]) -> String {
    if steps.is_empty() {
        return "no active plan — use /plan to enter plan mode, or let the model call update_plan"
            .into();
    }
    let mut out = String::from("plan:\n");
    for (i, step) in steps.iter().enumerate() {
        let mark = match step.status {
            hi_tools::PlanStatus::Done => "x",
            hi_tools::PlanStatus::Active => ">",
            hi_tools::PlanStatus::Pending => " ",
        };
        out.push_str(&format!("  {:>2}. [{mark}] {}\n", i + 1, step.title));
    }
    out
}

/// Append a durable user memory bullet. `global` writes user-level memory.
pub fn remember_note(workspace: &Path, text: &str, global: bool) -> Result<String> {
    let note = text.trim();
    if note.is_empty() {
        bail!("usage: /remember <note>  (add --global for user-level memory)");
    }
    let bullet = if note.starts_with('-') {
        note.to_string()
    } else {
        format!("- {note}")
    };
    let path = if global {
        global_memory_file()
    } else {
        memory_file_at(workspace)
    };
    let existing = if global {
        read_global_memory()
    } else {
        crate::memory::read_layer(&path)
    };
    let mut body = existing.trim().to_string();
    if !body.is_empty() {
        body.push('\n');
    }
    body.push_str(&bullet);
    let body = crate::memory::stamp_layer_ids(workspace, &body);
    let n = crate::memory::write_memory(&path, &body).map_err(|e| anyhow::anyhow!(e))?;
    let stamped = body.lines().last().unwrap_or(&body);
    let id = crate::memory::parse_bullet_line(stamped)
        .and_then(|(id, _)| id)
        .map(|id| format!("[#{id}]"))
        .unwrap_or_default();
    Ok(format!(
        "remembered {id} ({} note{}, {}):\n  {}\n/undo-memory restores the previous file",
        n,
        if n == 1 { "" } else { "s" },
        path.display(),
        stamped
    ))
}

fn format_memory_dump(workspace: &Path) -> String {
    let project = memory_file_at(workspace);
    let global = global_memory_file();
    let read = |path: &Path| {
        let body = crate::memory::read_layer(path);
        if body.trim().is_empty() {
            "(empty)".to_string()
        } else {
            body
        }
    };
    format!(
        "project ({})\n{}\n\nglobal ({})\n{}",
        project.display(),
        read(&project),
        global.display(),
        read(&global)
    )
}

/// Best-effort Claude Code config discovery + migration hints (no silent overwrite).
pub fn import_claude_report(cwd: &Path) -> String {
    let mut out = String::from("claude import scan:\n");
    let mut found = 0usize;

    let home = dirs_home();
    let candidates = [
        home.as_ref()
            .map(|h| h.join(".claude.json"))
            .unwrap_or_default(),
        home.as_ref()
            .map(|h| h.join(".claude").join("settings.json"))
            .unwrap_or_default(),
        cwd.join(".mcp.json"),
        cwd.join(".claude").join("settings.json"),
        cwd.join(".claude").join("settings.local.json"),
        cwd.join("CLAUDE.md"),
    ];

    for path in candidates {
        if path.as_os_str().is_empty() {
            continue;
        }
        if path.is_file() {
            found += 1;
            let kind = classify_claude_path(&path);
            out.push_str(&format!("  found: {} ({kind})\n", path.display()));
            if let Some(hint) = migration_hint(&path, kind) {
                out.push_str(&format!("         → {hint}\n"));
            }
        }
    }

    // MCP servers inside ~/.claude.json
    if let Some(home) = &home {
        let claude_json = home.join(".claude.json");
        if claude_json.is_file() {
            out.push_str(&inspect_claude_mcp_servers(&claude_json));
        }
    }

    if found == 0 {
        out.push_str("  nothing found under ~/.claude*, .mcp.json, or ./CLAUDE.md\n");
    }
    out.push_str(
        "  note: hi does not auto-overwrite hi.toml; apply the hints above, then /doctor\n",
    );
    out
}

fn inspect_claude_mcp_servers(path: &Path) -> String {
    let Some(raw) = crate::learning::read_ledger_capped(path) else {
        return String::new();
    };
    let Ok(value) = serde_json::from_str::<serde_json::Value>(&raw) else {
        return "  mcpServers in ~/.claude.json: (file too large or unreadable; listing skipped)\n"
            .into();
    };
    let Some(servers) = value.get("mcpServers").and_then(|s| s.as_object()) else {
        return String::new();
    };
    let mut out = format!("  mcpServers in ~/.claude.json: {}\n", servers.len());
    for name in servers.keys().take(12) {
        out.push_str(&format!("    - {name}\n"));
    }
    if servers.len() > 12 {
        out.push_str(&format!("    … {} more\n", servers.len() - 12));
    }
    out.push_str(
        "         → copy into hi.toml [profiles.*.mcp] / provider mcp_url, or keep using Claude MCP separately\n",
    );
    out
}

/// List skill packs + optional `.hi/hooks` scripts as a lightweight plugins/hooks view.
pub fn plugins_and_hooks_report(workspace: &Path) -> String {
    let mut out = String::from("plugins / hooks:\n");
    out.push_str("  skills (hi's extension packs):\n");
    let _ = workspace;
    let skills = crate::skills::list_skills();
    if skills.is_empty() {
        out.push_str("    (none discovered — try /learn)\n");
    } else {
        for s in skills.iter().take(30) {
            out.push_str(&format!("    - {} ({})\n", s.name, s.scope));
        }
        if skills.len() > 30 {
            out.push_str(&format!(
                "    … {} more — /skills for full list\n",
                skills.len() - 30
            ));
        }
    }

    let hooks_dir = workspace.join(".hi").join("hooks");
    out.push_str("  hooks (.hi/hooks — executable scripts, convention-only for now):\n");
    if hooks_dir.is_dir() {
        let mut entries: Vec<_> = std::fs::read_dir(&hooks_dir)
            .ok()
            .into_iter()
            .flatten()
            .filter_map(|e| e.ok())
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .collect();
        entries.sort();
        if entries.is_empty() {
            out.push_str("    (directory empty)\n");
        } else {
            for name in entries {
                out.push_str(&format!("    - {name}\n"));
            }
        }
    } else {
        out.push_str(&format!(
            "    (no {} — create scripts named pre-turn, post-turn, …)\n",
            hooks_dir.display()
        ));
    }
    out.push_str(
        "  tip: learned skills via /learn and /skill; lifecycle hook runner is still thin — scripts are inventoried here\n",
    );
    out
}

/// Create an isolated git worktree for `/fork --worktree`.
pub fn fork_worktree(workspace: &Path, label: &str) -> Result<PathBuf> {
    if !hi_tools::worktree::in_git_repo(workspace) {
        bail!("/fork --worktree requires a git repository");
    }
    let _ = label;
    let idx = (chrono_like_stamp() % 10_000) as u32;
    let path = hi_tools::worktree::worktree_path("fork", idx);
    let head = git_head(workspace).unwrap_or_else(|| "HEAD".into());
    hi_tools::worktree::add_worktree(workspace, &path, &head)
        .with_context(|| format!("creating worktree at {}", path.display()))?;
    Ok(path)
}

pub fn fork_summary(worktree: Option<&Path>, directive: &str) -> String {
    let mut out = String::from("fork:\n");
    match worktree {
        Some(p) => {
            out.push_str(&format!("  worktree: {}\n", p.display()));
            out.push_str("  start a peer session with:\n");
            out.push_str(&format!("    cd {} && hi\n", p.display()));
        }
        None => {
            out.push_str("  no worktree (same tree) — open another terminal in this project, or pass --worktree\n");
            out.push_str("  tip: hi --best-of N also explores isolated candidates\n");
        }
    }
    if !directive.trim().is_empty() {
        out.push_str(&format!(
            "  first prompt suggestion:\n    {}\n",
            directive.trim()
        ));
    }
    out
}

fn git_head(workspace: &Path) -> Option<String> {
    let out = std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .current_dir(workspace)
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8(out.stdout).ok()?;
    let s = s.trim();
    if s.is_empty() {
        None
    } else {
        Some(s.to_string())
    }
}

fn chrono_like_stamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() % 1_000_000)
        .unwrap_or(0)
}

fn collapse_preview(text: &str, max: usize) -> String {
    let one = text
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .collect::<Vec<_>>()
        .join(" · ");
    if one.chars().count() <= max {
        one
    } else {
        let clipped: String = one.chars().take(max.saturating_sub(1)).collect();
        format!("{clipped}…")
    }
}

fn dirs_home() -> Option<PathBuf> {
    std::env::var_os("HOME").map(PathBuf::from)
}

fn classify_claude_path(path: &Path) -> &'static str {
    let name = path.file_name().and_then(|s| s.to_str()).unwrap_or("");
    match name {
        ".claude.json" => "claude user config",
        "settings.json" | "settings.local.json" => "claude settings",
        ".mcp.json" => "mcp servers",
        "CLAUDE.md" => "project instructions",
        _ => "claude-related",
    }
}

fn migration_hint(path: &Path, kind: &str) -> Option<String> {
    match kind {
        "project instructions" => Some(format!(
            "merge useful bits into HI.md (run /init) or .hi/memory.md — source {}",
            path.display()
        )),
        "mcp servers" => {
            Some("review servers and configure hi provider mcp_url / external MCP as needed".into())
        }
        "claude user config" => {
            Some("inspect mcpServers / permissions; map API keys into hi profiles".into())
        }
        "claude settings" => Some("map model/permission preferences into hi.toml profiles".into()),
        _ => None,
    }
}

#[path = "session_ops_hooks.rs"]
mod lifecycle_hooks;

pub use lifecycle_hooks::run_hook;
#[cfg(test)]
pub(crate) use lifecycle_hooks::run_hook_process_cancellable_for_test;
#[cfg(test)]
use lifecycle_hooks::{
    HOOK_PIPE_DRAIN_GRACE, MAX_HOOK_OUTPUT_BYTES, hook_timeout_from_value, run_hook_process,
    tolerate_closed_hook_stdin,
};
pub(crate) use lifecycle_hooks::{HookExecution, run_hook_cancellable};

pub fn hooks_command(workspace: &Path, arg: &str) -> String {
    let arg = arg.trim();
    if arg.is_empty() || arg == "list" || arg == "status" {
        return plugins_and_hooks_report(workspace);
    }
    let _ = arg;
    "hook execution is available during turn lifecycle only".into()
}

/// Plugin marketplace implemented as a portable skill-pack index/install path.
pub fn marketplace_report(workspace: &Path, arg: &str) -> String {
    let arg = arg.trim();
    let root = workspace.join(".hi").join("marketplace");
    if arg.is_empty() || matches!(arg, "list" | "status") {
        let mut out = format!("plugin marketplace: {}\n", root.display());
        if !root.is_dir() {
            out.push_str("  (empty)\n  install: /marketplace install <path-to-SKILL.md>\n");
            return out;
        }
        let mut names: Vec<_> = std::fs::read_dir(&root)
            .ok()
            .into_iter()
            .flatten()
            .filter_map(|e| e.ok())
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .collect();
        names.sort();
        for name in names {
            out.push_str(&format!("  - {name}\n"));
        }
        return out;
    }
    if let Some(source) = arg.strip_prefix("install ") {
        let source = PathBuf::from(source.trim());
        let source = if source.is_dir() {
            source.join("SKILL.md")
        } else {
            source
        };
        if !source.is_file() {
            return format!("marketplace install failed: {} not found", source.display());
        }
        let stem = source
            .parent()
            .and_then(|p| p.file_name())
            .or_else(|| source.file_stem())
            .and_then(|s| s.to_str())
            .unwrap_or("plugin");
        let dest = workspace
            .join(".hi")
            .join("skills")
            .join(stem)
            .join("SKILL.md");
        if let Some(parent) = dest.parent()
            && let Err(e) = std::fs::create_dir_all(parent)
        {
            return format!("marketplace install failed: {e}");
        }
        return match std::fs::copy(&source, &dest) {
            Ok(_) => format!("installed plugin skill: {}", dest.display()),
            Err(e) => format!("marketplace install failed: {e}"),
        };
    }
    "usage: /marketplace [list|install <SKILL.md>]".into()
}

pub fn worktree_command(workspace: &Path, arg: &str) -> String {
    let arg = arg.trim();
    let prefix = format!("hi-fork-{}-", std::process::id());
    let temp = std::env::temp_dir();
    let mut trees: Vec<PathBuf> = std::fs::read_dir(&temp)
        .ok()
        .into_iter()
        .flatten()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_name().to_string_lossy().starts_with(&prefix))
        .map(|e| e.path())
        .collect();
    trees.sort();
    match arg {
        "" | "list" | "status" => {
            let mut out = String::from("hi fork worktrees:\n");
            if trees.is_empty() {
                out.push_str("  (none)\n");
            } else {
                for (i, p) in trees.iter().enumerate() {
                    out.push_str(&format!("  {}. {}\n", i + 1, p.display()));
                }
            }
            out.push_str("  cleanup: /worktree gc\n");
            out
        }
        "gc" | "clean" | "cleanup" => {
            hi_tools::worktree::cleanup(workspace, &trees);
            format!("cleaned {} fork worktree(s)", trees.len())
        }
        other if other.starts_with("remove ") => {
            let value = other.trim_start_matches("remove ").trim();
            let selected = value
                .parse::<usize>()
                .ok()
                .and_then(|n| trees.get(n.saturating_sub(1)).cloned())
                .or_else(|| trees.iter().find(|p| p.to_string_lossy() == value).cloned());
            match selected {
                Some(path) => {
                    hi_tools::worktree::cleanup(workspace, std::slice::from_ref(&path));
                    format!("removed {}", path.display())
                }
                None => format!("worktree not found: {value}"),
            }
        }
        _ => "usage: /worktree [list|gc|remove <n|path>]".into(),
    }
}

#[derive(Serialize)]
struct InspectReport {
    workspace: String,
    trusted: bool,
    git: bool,
    model: String,
    provider: String,
    plan_mode: bool,
    permissions: String,
    goal: String,
    checkpoints: usize,
    background_processes: Vec<String>,
    hooks: Vec<String>,
    skills: Vec<String>,
}

pub fn inspect_report(agent: &crate::Agent, json: bool) -> String {
    let workspace = agent.workspace_root();
    let hooks_dir = workspace.join(".hi/hooks");
    let mut hooks: Vec<String> = std::fs::read_dir(&hooks_dir)
        .ok()
        .into_iter()
        .flatten()
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect();
    hooks.sort();
    let skills = crate::skills::list_skills()
        .into_iter()
        .map(|s| format!("{} ({})", s.name, s.scope))
        .collect();
    let report = InspectReport {
        workspace: workspace.display().to_string(),
        trusted: workspace_trusted(workspace),
        git: hi_tools::worktree::in_git_repo(workspace),
        model: agent.model().to_string(),
        provider: agent.provider_route().unwrap_or("unknown").to_string(),
        plan_mode: agent.plan_mode(),
        permissions: agent.permission_mode().as_str().into(),
        goal: agent.goal_summary(),
        checkpoints: agent.checkpoint_count(),
        background_processes: agent.background_process_ids(),
        hooks,
        skills,
    };
    if json {
        return serde_json::to_string_pretty(&report).unwrap_or_else(|_| "{}".into());
    }
    format!(
        "hi inspect\n  workspace: {}\n  trust: {}\n  git: {}\n  model: {}\n  provider: {}\n  plan_mode: {}\n  permissions: {}\n  goal: {}\n  checkpoints: {}\n  background: {}\n  hooks: {}\n  skills: {}\n",
        report.workspace,
        if report.trusted {
            "trusted"
        } else {
            "untrusted"
        },
        report.git,
        report.model,
        report.provider,
        report.plan_mode,
        report.permissions,
        report.goal,
        report.checkpoints,
        if report.background_processes.is_empty() {
            "none".into()
        } else {
            report.background_processes.join(", ")
        },
        if report.hooks.is_empty() {
            "none".into()
        } else {
            report.hooks.join(", ")
        },
        report.skills.len(),
    )
}

pub fn agents_report(workspace: &Path, arg: &str) -> String {
    let dir = workspace.join(".hi/agents");
    let arg = arg.trim();
    if arg.is_empty() || arg == "list" {
        let mut out = format!("agents/personas: {}\n", dir.display());
        let mut files: Vec<_> = std::fs::read_dir(&dir)
            .ok()
            .into_iter()
            .flatten()
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().is_some_and(|x| x == "md"))
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .collect();
        files.sort();
        if files.is_empty() {
            out.push_str("  (none)\n  create: /agents add <name> <instructions>\n");
        } else {
            for f in files {
                out.push_str(&format!("  - {}\n", f.trim_end_matches(".md")));
            }
        }
        return out;
    }
    if let Some(rest) = arg.strip_prefix("add ") {
        let (name, body) = rest.split_once(char::is_whitespace).unwrap_or((rest, ""));
        if name.is_empty() || body.trim().is_empty() {
            return "usage: /agents add <name> <instructions>".into();
        }
        let name: String = name
            .chars()
            .filter(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_'))
            .collect();
        if name.is_empty() {
            return "invalid agent name".into();
        }
        let path = dir.join(format!("{name}.md"));
        if let Err(e) = std::fs::create_dir_all(&dir)
            .and_then(|_| std::fs::write(&path, format!("# {name}\n\n{}\n", body.trim())))
        {
            return format!("agent add failed: {e}");
        }
        return format!("added agent persona: {}", path.display());
    }
    if let Some(name) = arg.strip_prefix("show ") {
        let path = dir.join(format!("{}.md", name.trim()));
        return match crate::learning::read_ledger_capped(&path) {
            Some(raw) => raw,
            None => format!(
                "agent show failed: {}",
                std::fs::metadata(&path)
                    .err()
                    .map(|e| e.to_string())
                    .unwrap_or_else(|| "unreadable".into())
            ),
        };
    }
    if let Some(name) = arg.strip_prefix("remove ") {
        let path = dir.join(format!("{}.md", name.trim()));
        return match std::fs::remove_file(&path) {
            Ok(()) => format!("removed {}", path.display()),
            Err(e) => format!("agent remove failed: {e}"),
        };
    }
    "usage: /agents [list|add <name> <instructions>|show <name>|remove <name>]".into()
}

pub fn share_report(agent: &crate::Agent, arg: &str) -> String {
    let arg = arg.trim();
    let dir = agent.workspace_root().join(".hi/shares");
    let stamp = chrono_like_stamp();
    let path = dir.join(format!("session-{stamp}.md"));
    if let Err(e) =
        std::fs::create_dir_all(&dir).and_then(|_| std::fs::write(&path, agent.export_markdown()))
    {
        return format!("share export failed: {e}");
    }
    let portal = std::env::var("HI_SHARE_BASE_URL").ok();
    let mut out = format!("share bundle: {}\n", path.display());
    if let Some(base) = portal.as_deref() {
        out.push_str(&format!(
            "portal: {}/{}\n",
            base.trim_end_matches('/'),
            path.file_name().unwrap_or_default().to_string_lossy()
        ));
        out.push_str("note: upload is controlled by your portal/sync deployment\n");
    } else {
        out.push_str("set HI_SHARE_BASE_URL to print a review portal URL; bundle is local\n");
    }
    if arg == "json" || arg == "--json" {
        serde_json::json!({"path": path, "portal_base": portal}).to_string()
    } else {
        out
    }
}

pub fn mcp_admin_report(arg: &str) -> String {
    match arg.trim() {
        "" | "list" | "status" => "MCP admin:\n  workspace servers: /mcp\n  provider mcp_url: /mcp pipe\n  /mcp add <name> --stdio … | --http <url>\n  /mcp <name> allow|deny <tool>\n  /doctor probes provider MCP\n  import gating: [mcp_import] in hi.toml\n  per-server lists: [mcp.servers.<name>] in hi.toml or only/exclude in .hi/mcp/<name>.json\n".into(),
        "doctor" => "use /doctor (includes MCP connectivity/tools probe)".into(),
        other if other.starts_with("add ") || other.starts_with("remove ") => {
            "use /mcp add <name> --stdio … | --http <url> (writes .hi/mcp/<name>.json)".into()
        }
        _ => "usage: /mcp-admin [list|doctor|add|remove]".into(),
    }
}

/// Parse `/fork` args into (want_worktree, directive).
pub fn parse_fork_args(arg: &str) -> (bool, String) {
    let mut want_wt = true; // default: isolate when possible
    let mut rest = Vec::new();
    for tok in arg.split_whitespace() {
        match tok {
            "--worktree" | "-w" => want_wt = true,
            "--no-worktree" | "--same-tree" => want_wt = false,
            other => rest.push(other),
        }
    }
    (want_wt, rest.join(" "))
}

/// Parse `/remember` args into (global, text).
pub fn parse_remember_args(arg: &str) -> (bool, String) {
    let mut global = false;
    let mut rest = Vec::new();
    for tok in arg.split_whitespace() {
        match tok {
            "--global" | "-g" => global = true,
            other => rest.push(other),
        }
    }
    // Also allow "/remember global …"
    if rest.first().is_some_and(|t| *t == "global") {
        global = true;
        rest.remove(0);
    }
    (global, rest.join(" "))
}

/// Build a tasks/queue snapshot from agent-visible facts. Frontends may append
/// their own prompt-queue lines before/after.
pub fn format_tasks_report(
    background_pids: &[String],
    bg_task_ids: &[String],
    checkpoint_count: usize,
    plan_mode: bool,
    permission: PermissionMode,
    extra_lines: &[String],
) -> String {
    let mut out = String::from("tasks / queue:\n");
    if extra_lines.is_empty() {
        out.push_str("  prompt queue: (empty or not tracked in this frontend)\n");
    } else {
        out.push_str("  prompt queue:\n");
        for line in extra_lines {
            out.push_str(&format!("    - {line}\n"));
        }
    }
    if background_pids.is_empty() {
        out.push_str("  background processes: none\n");
    } else {
        out.push_str(&format!(
            "  background processes: {}\n",
            background_pids.join(", ")
        ));
    }
    if bg_task_ids.is_empty() {
        out.push_str("  background tasks: none\n");
    } else {
        out.push_str(&format!("  background tasks: {}\n", bg_task_ids.join(", ")));
    }
    out.push_str(&format!("  checkpoints: {checkpoint_count}\n"));
    out.push_str(&format!(
        "  plan_mode: {} · permissions: {}\n",
        if plan_mode { "on" } else { "off" },
        permission.as_str()
    ));
    out.push_str("  tip: /loop list and /fleet show more in the TUI\n");
    out
}

pub fn plan_mode_prompt(user_request: &str) -> String {
    let base = "\
Plan mode is ON for this turn. Do not modify files or run mutating commands.
Produce a clear, ordered implementation plan the user can approve.
Prefer the update_plan tool to record steps. Its checklist represents future
implementation work: keep new or unexecuted steps pending, and mark a step done
only if it was already completed before this planning turn. Finishing the plan
itself does not finish its implementation steps. Ask clarifying questions if needed.
When the plan is ready, stop and wait — the user will leave plan mode to execute.";
    // `/plan <request>` already queues this wrapper as its follow-up prompt;
    // `run_turn_core` also applies it to every turn that starts in plan mode so
    // Shift-Tab has identical semantics. Keep the operation idempotent.
    if user_request
        .trim_start()
        .starts_with(crate::transcript::TURN_CONTROL_START)
    {
        return user_request.to_string();
    }
    if user_request.trim().is_empty() {
        format!(
            "{}\n{base}\nThe user enabled plan mode without a specific request — \
summarize what you know of the task so far and propose next steps.\n{}",
            crate::transcript::TURN_CONTROL_START,
            crate::transcript::TURN_CONTROL_END,
        )
    } else {
        format!(
            "{}\n{base}\n{}\n\nUser request:\n{}",
            crate::transcript::TURN_CONTROL_START,
            crate::transcript::TURN_CONTROL_END,
            user_request.trim()
        )
    }
}

/// Authoritative per-turn counter-control for normal operation. Models that
/// saw a prior planning turn must not infer that its mode restriction remains
/// active after `/plan off` or Shift-Tab restores mutation-capable tools.
pub(crate) fn normal_mode_prompt(user_request: &str) -> String {
    format!(
        "{}\nPlan mode is OFF for this turn. Earlier plan-mode controls are historical and no longer apply. Follow the current user request and the currently advertised tool permissions; do not claim that plan mode blocks an action.\n{}\n\n{}",
        crate::transcript::TURN_CONTROL_START,
        crate::transcript::TURN_CONTROL_END,
        user_request.trim()
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use hi_ai::{Message, Role};

    #[test]
    fn pipefs_write_forms_are_identified_before_sync_helpers_run() {
        for command in [
            Command::Fork(String::new()),
            Command::Remember("note".into()),
            Command::UndoMemory,
            Command::Marketplace("install /tmp/skill.md".into()),
            Command::Worktree("gc".into()),
            Command::Inspect("bundle".into()),
            Command::Cd("nested".into()),
            Command::Trust("on".into()),
        ] {
            assert!(session_command_mutates_workspace(&command), "{command:?}");
        }
        for command in [
            Command::Marketplace("status".into()),
            Command::Worktree("list".into()),
            Command::Inspect("json".into()),
            Command::Cd(String::new()),
            Command::Trust("status".into()),
        ] {
            assert!(!session_command_mutates_workspace(&command), "{command:?}");
        }
        assert!(matches!(
            session_command_replay_class(&Command::Trust("on".into())),
            hi_workspace::ReplayClass::IdempotentExternal { .. }
        ));
        assert!(matches!(
            session_command_replay_class(&Command::Remember("note".into())),
            hi_workspace::ReplayClass::NonReplayableExternal
        ));
    }

    fn u(t: &str) -> Message {
        Message::user(t)
    }
    fn a(t: &str) -> Message {
        Message {
            role: Role::Assistant,
            content: vec![hi_ai::Content::Text(t.into())],
        }
    }

    #[test]
    fn mode_prompts_are_delimited_and_current_turn_authoritative() {
        let plan = plan_mode_prompt("build profiles");
        assert!(plan.starts_with(crate::transcript::TURN_CONTROL_START));
        assert!(plan.contains("Plan mode is ON for this turn"));
        assert!(plan.contains("User request:\nbuild profiles"));
        assert_eq!(plan_mode_prompt(&plan), plan, "plan wrapping is idempotent");

        let normal = normal_mode_prompt("build all of that");
        assert!(normal.starts_with(crate::transcript::TURN_CONTROL_START));
        assert!(normal.contains("Plan mode is OFF for this turn"));
        assert!(normal.contains("Earlier plan-mode controls are historical"));
        assert!(normal.ends_with("build all of that"));
    }

    #[test]
    fn lists_and_rewinds_user_turns() {
        let msgs = vec![
            Message::system("sys"),
            u("one"),
            a("ok1"),
            u("two"),
            a("ok2"),
        ];
        let turns = list_user_turns(&msgs);
        assert_eq!(turns.len(), 2);
        assert_eq!(turns[0].n, 1);
        assert_eq!(rewind_len_before_user_turn(&msgs, 2).unwrap(), 3);
    }

    #[test]
    fn parse_fork_and_remember_flags() {
        assert_eq!(
            parse_fork_args("--no-worktree try rustc"),
            (false, "try rustc".into())
        );
        assert_eq!(
            parse_remember_args("--global prefer pnpm"),
            (true, "prefer pnpm".into())
        );
    }

    #[test]
    fn tasks_report_names_background_task_ids() {
        let report = format_tasks_report(
            &["pid-9".into()],
            &["task_1".into(), "task_2".into()],
            0,
            false,
            PermissionMode::Ask,
            &[],
        );
        assert!(report.contains("task_1"));
        assert!(report.contains("task_2"));
        assert!(report.contains("background tasks:"));
    }

    #[test]
    fn search_finds_assistant_text() {
        let msgs = vec![u("hi"), a("unique-token-xyz")];
        let r = search_messages(&msgs, "unique-token");
        assert!(r.contains("unique-token-xyz"));
    }

    #[test]
    fn agents_show_clips_a_huge_persona() {
        let root = std::env::temp_dir().join(format!(
            "hi-agents-show-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join(".hi/agents")).unwrap();
        std::fs::write(
            root.join(".hi/agents/huge.md"),
            "P".repeat(crate::learning::MAX_LEDGER_READ_BYTES + 8_000),
        )
        .unwrap();
        let out = agents_report(&root, "show huge");
        assert!(
            out.len() <= crate::learning::MAX_LEDGER_READ_BYTES,
            "persona dump must be prefix-capped: {}",
            out.len()
        );
        assert!(out.starts_with('P'), "{out}");
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn claude_mcp_inspect_lists_servers_and_skips_a_huge_file() {
        let dir = std::env::temp_dir().join(format!(
            "hi-claude-json-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let small = dir.join("small.json");
        std::fs::write(
            &small,
            r#"{"mcpServers":{"sqlite":{},"github":{},"extra":{}}}"#,
        )
        .unwrap();
        let listed = inspect_claude_mcp_servers(&small);
        assert!(listed.contains("sqlite"), "{listed}");
        assert!(listed.contains("github"), "{listed}");

        let huge = dir.join("huge.json");
        std::fs::write(
            &huge,
            format!(
                "{{\"mcpServers\":{{}},\"pad\":\"{}\"}}",
                "x".repeat(crate::learning::MAX_LEDGER_READ_BYTES + 8_000)
            ),
        )
        .unwrap();
        let skipped = inspect_claude_mcp_servers(&huge);
        assert!(
            skipped.len() < 4_096,
            "huge claude.json must not be dumped: {}",
            skipped.len()
        );
        assert!(
            skipped.contains("too large") || skipped.contains("mcpServers"),
            "{skipped}"
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn hook_command_lists_missing_directory() {
        let root = std::env::temp_dir().join(format!("hi-hook-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        let report = hooks_command(&root, "list");
        assert!(report.contains("hooks"));
        assert!(report.contains("no "));
        let _ = std::fs::remove_dir_all(root);
    }

    #[tokio::test]
    async fn untrusted_workspace_blocks_hook_execution() {
        let root = std::env::temp_dir().join(format!(
            "hi-untrusted-hook-test-{}-{}",
            std::process::id(),
            chrono_like_stamp()
        ));
        std::fs::create_dir_all(root.join(".hi/hooks")).unwrap();
        std::fs::write(
            root.join(".hi/hooks/pre-turn"),
            "#!/bin/sh\necho should-not-run\n",
        )
        .unwrap();
        let error = run_hook(&root, "pre-turn", "x")
            .await
            .unwrap_err()
            .to_string();
        assert!(error.contains("untrusted"), "{error}");
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn hook_timeout_is_opt_in_and_zero_means_unlimited() {
        assert_eq!(hook_timeout_from_value(None), None);
        assert_eq!(hook_timeout_from_value(Some("")), None);
        assert_eq!(hook_timeout_from_value(Some("0")), None);
        assert_eq!(hook_timeout_from_value(Some("invalid")), None);
        assert_eq!(
            hook_timeout_from_value(Some("17")),
            Some(std::time::Duration::from_secs(17))
        );
    }

    #[test]
    fn closed_hook_stdin_is_benign_but_other_write_errors_are_not() {
        assert!(
            tolerate_closed_hook_stdin(Err(std::io::Error::from(std::io::ErrorKind::BrokenPipe)))
                .is_ok()
        );
        let error = tolerate_closed_hook_stdin(Err(std::io::Error::from(
            std::io::ErrorKind::PermissionDenied,
        )))
        .unwrap_err();
        assert_eq!(error.kind(), std::io::ErrorKind::PermissionDenied);
    }

    #[cfg(unix)]
    fn write_test_hook(root: &Path, body: &str) {
        use std::os::unix::fs::PermissionsExt as _;

        let hooks = root.join(".hi/hooks");
        std::fs::create_dir_all(&hooks).unwrap();
        let path = hooks.join("pre-turn");
        std::fs::write(&path, format!("#!/bin/sh\n{body}\n")).unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
    }

    #[cfg(unix)]
    fn process_is_alive(pid: libc::pid_t) -> bool {
        // SAFETY: signal 0 only probes a PID created by the test and does not
        // alter process state.
        unsafe { libc::kill(pid, 0) == 0 }
    }

    #[cfg(unix)]
    async fn assert_process_exits(pid: libc::pid_t) {
        let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(2);
        while process_is_alive(pid) && tokio::time::Instant::now() < deadline {
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
        assert!(!process_is_alive(pid), "hook descendant {pid} leaked");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn completed_hook_reaps_daemonized_descendants_without_waiting_for_their_pipes() {
        let root = tempfile::tempdir().unwrap();
        write_test_hook(
            root.path(),
            "sleep 30 &\necho $! > child.pid\nprintf '{\"version\":1,\"decision\":\"allow\",\"message\":\"done\"}\\n'",
        );
        // Keep the watchdog strictly outside the implementation's bounded
        // pipe-drain grace. Equal deadlines made this test flaky under load:
        // the outer timer could win before cleanup returned (or before its
        // specific error surfaced), even though the process group was killed.
        let watchdog = HOOK_PIPE_DRAIN_GRACE + std::time::Duration::from_secs(3);
        let report = tokio::time::timeout(
            watchdog,
            run_hook_process(root.path(), "pre-turn", "input", None),
        )
        .await
        .expect("daemonized hook must not wedge output drain")
        .unwrap();
        assert!(report.contains("done"), "{report}");

        let pid: libc::pid_t = std::fs::read_to_string(root.path().join("child.pid"))
            .unwrap()
            .trim()
            .parse()
            .unwrap();
        assert_process_exits(pid).await;
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn explicit_hook_timeout_reaps_the_complete_process_group() {
        let root = tempfile::tempdir().unwrap();
        // Leave enough startup headroom for a heavily loaded test host. The
        // timeout begins after `spawn`, but the OS may not schedule the shell
        // quickly enough to create `child.pid` under concurrent PTY campaigns.
        // Keep the descendant far beyond the deadline so the assertion still
        // proves timeout-driven process-group cleanup rather than natural exit.
        write_test_hook(root.path(), "sleep 30 &\necho $! > child.pid\nwait");
        let error = run_hook_process(
            root.path(),
            "pre-turn",
            "input",
            Some(std::time::Duration::from_secs(5)),
        )
        .await
        .unwrap_err()
        .to_string();
        assert!(error.contains("timed out"), "{error}");

        let pid: libc::pid_t = std::fs::read_to_string(root.path().join("child.pid"))
            .unwrap()
            .trim()
            .parse()
            .unwrap();
        assert_process_exits(pid).await;
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn hook_output_is_drained_concurrently_and_retained_with_a_bound() {
        let root = tempfile::tempdir().unwrap();
        write_test_hook(
            root.path(),
            "i=0\nwhile [ $i -lt 30000 ]; do\n  printf 'stdout-abcdefghijklmnopqrstuvwxyz-0123456789\\n'\n  printf 'stderr-abcdefghijklmnopqrstuvwxyz-0123456789\\n' >&2\n  i=$((i + 1))\ndone",
        );
        let report = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            run_hook_process(root.path(), "pre-turn", "input", None),
        )
        .await
        .expect("noisy hook must not block on full pipes")
        .unwrap();
        assert!(
            report.len() <= MAX_HOOK_OUTPUT_BYTES + 128,
            "{}",
            report.len()
        );
        assert!(report.contains("hook output truncated"));
    }

    #[test]
    fn marketplace_installs_skill_file() {
        let root = std::env::temp_dir().join(format!("hi-market-test-{}", std::process::id()));
        let source_dir = root.join("source-pack");
        std::fs::create_dir_all(&source_dir).unwrap();
        std::fs::write(source_dir.join("SKILL.md"), "---\nname: test\n---\n").unwrap();
        let report = marketplace_report(&root, &format!("install {}", source_dir.display()));
        assert!(report.contains("installed"), "{report}");
        assert!(root.join(".hi/skills/source-pack/SKILL.md").is_file());
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn mcp_admin_has_doctor_route() {
        assert!(mcp_admin_report("doctor").contains("/doctor"));
    }

    #[test]
    fn remember_note_stamps_bullet_id_and_mentions_undo() {
        let root = std::env::temp_dir().join(format!(
            "hi-remember-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join(".hi")).unwrap();
        let msg = remember_note(&root, "prefer unique-remember-token", false).unwrap();
        assert!(msg.contains("[#"), "{msg}");
        assert!(msg.contains("/undo-memory"), "{msg}");
        let body = crate::memory::read_layer(&root.join(".hi/memory.md"));
        assert!(body.contains("[#"), "{body}");
        assert!(body.contains("prefer unique-remember-token"), "{body}");
        let _ = std::fs::remove_dir_all(root);
    }
}
