//! `hi tools` — census-driven trimming of advertised tool schemas.
//!
//! The census in `hi metrics` prices advertised-but-never-called tools in
//! schema tokens per request. This surface turns that evidence into action,
//! deliberately gated:
//!
//! - **explicit** — nothing is ever trimmed automatically; a human runs
//!   `hi tools trim <name>` off the census suggestion;
//! - **evidence-gated** — trim refuses unless the tool went uncalled across
//!   at least [`MIN_TRIM_SESSIONS`] recent sessions (`--force` overrides the
//!   evidence, never the floor);
//! - **floor-protected** — [`hi_tools::PROTECTED_TOOLS`] can never be
//!   trimmed, and the advertisement filter re-checks the floor, so a wrong
//!   list degrades to "no trim", never to a broken agent;
//! - **measured** — applying a trim records a `tool-trim` intervention, so
//!   the before/after effect windows in `hi metrics` cover it.
//!
//! The list lives at `<state-root>/learning/tools-disabled.json` and is
//! loaded into `AgentConfig` at construction; `hi tools keep <name>` undoes.

use std::collections::BTreeSet;
use std::path::Path;

use anyhow::{Context, Result, bail};

/// Evidence bar for a trim without `--force`: the tool must be uncalled
/// across at least this many recent sessions. One quiet week is a fluke;
/// twenty sessions of silence is a pattern.
const MIN_TRIM_SESSIONS: usize = 20;

/// Tools whose advertisement is already gated by a feature flag or mode
/// (subagent flags, long-horizon goals). They cost no schema tokens while
/// their flag is off, and a trim would silently disable the capability the
/// day the flag is turned on — so they are never *suggested* as candidates.
/// An explicit `hi tools trim <name>` still works for someone who means it.
pub(crate) const CONDITIONAL_TOOLS: &[&str] = &[
    "explore",
    "delegate",
    "task",
    "get_task_output",
    "wait_tasks",
    "kill_task",
    "block_step",
];

/// How many recent sessions the evidence sweep reads.
const TRIM_SWEEP_SESSIONS: usize = 100;

fn config_path(state_root: &Path) -> std::path::PathBuf {
    state_root.join("learning").join("tools-disabled.json")
}

/// The current per-project disabled list. Missing or unparseable config is an
/// empty list — the failure mode is always "advertise everything".
pub(crate) fn disabled_tools(state_root: &Path) -> Vec<String> {
    std::fs::read_to_string(config_path(state_root))
        .ok()
        .and_then(|raw| serde_json::from_str::<serde_json::Value>(&raw).ok())
        .and_then(|value| {
            value.get("disabled").and_then(|list| {
                list.as_array().map(|names| {
                    names
                        .iter()
                        .filter_map(|name| name.as_str().map(str::to_string))
                        .collect()
                })
            })
        })
        .unwrap_or_default()
}

fn save_disabled(state_root: &Path, disabled: &BTreeSet<String>) -> Result<()> {
    let path = config_path(state_root);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let value = serde_json::json!({ "disabled": disabled });
    std::fs::write(&path, serde_json::to_string_pretty(&value)?)
        .with_context(|| format!("writing {}", path.display()))
}

/// Every name the advertisement path can produce: the static catalog plus the
/// separately-pushed subagent tools.
fn known_tool_names() -> BTreeSet<String> {
    let mut names: BTreeSet<String> = hi_tools::TOOL_SPECS
        .iter()
        .map(|spec| spec.name.clone())
        .collect();
    for pushed in [
        "explore",
        "delegate",
        "task",
        "get_task_output",
        "wait_tasks",
        "kill_task",
    ] {
        names.insert(pushed.to_string());
    }
    names
}

/// Names worth suggesting for a trim: dead in the sweep, not already
/// trimmed, and neither floor-protected nor flag-gated.
fn trim_candidates(used: &BTreeSet<String>, disabled: &BTreeSet<String>) -> Vec<String> {
    known_tool_names()
        .into_iter()
        .filter(|name| {
            !used.contains(name)
                && !disabled.contains(name)
                && !hi_tools::PROTECTED_TOOLS.contains(&name.as_str())
                && !CONDITIONAL_TOOLS.contains(&name.as_str())
        })
        .collect()
}

/// `hi tools <list|trim|keep> …`
pub(crate) fn run_tools_cli(
    state_root: &Path,
    sessions_dir: Option<&Path>,
    args: &[String],
) -> Result<()> {
    match args.first().map(String::as_str) {
        Some("trim") => {
            let force = args.iter().any(|argument| argument == "--force");
            let names: Vec<&String> = args[1..]
                .iter()
                .filter(|argument| !argument.starts_with('-'))
                .collect();
            if names.is_empty() {
                bail!("usage: hi tools trim <name>… [--force]");
            }
            let known = known_tool_names();
            for name in &names {
                if hi_tools::PROTECTED_TOOLS.contains(&name.as_str()) {
                    bail!(
                        "{name} is on the protected floor (core workspace loop) and can never \
                         be trimmed — not even with --force"
                    );
                }
                if !known.contains(name.as_str()) {
                    bail!(
                        "unknown tool {name:?} — `hi tools list` shows the advertised catalog"
                    );
                }
            }
            let swept = if force {
                0
            } else {
                let Some(sessions_dir) = sessions_dir else {
                    bail!("no session data to base the trim on (or pass --force)");
                };
                let files =
                    crate::learning_ledger::newest_session_files(sessions_dir, TRIM_SWEEP_SESSIONS);
                if files.len() < MIN_TRIM_SESSIONS {
                    bail!(
                        "only {} session(s) of evidence — a trim needs at least \
                         {MIN_TRIM_SESSIONS} (or --force)",
                        files.len()
                    );
                }
                let used = crate::learning_ledger::used_tool_names(&files);
                for name in &names {
                    if used.contains(name.as_str()) {
                        bail!(
                            "{name} was called within the last {} session(s) — not dead, \
                             not trimming (override with --force)",
                            files.len()
                        );
                    }
                }
                files.len()
            };
            let mut disabled: BTreeSet<String> =
                disabled_tools(state_root).into_iter().collect();
            let added: Vec<&String> = names
                .iter()
                .copied()
                .filter(|name| disabled.insert((*name).clone()))
                .collect();
            if added.is_empty() {
                println!("nothing to do — already trimmed");
                return Ok(());
            }
            save_disabled(state_root, &disabled)?;
            let added_list = added
                .iter()
                .map(|name| name.as_str())
                .collect::<Vec<_>>()
                .join(", ");
            let evidence = if force {
                "forced, no census evidence".to_string()
            } else {
                format!("unused across the last {swept} sessions")
            };
            let _ = crate::learning_ledger::record_intervention(
                state_root,
                "tool-trim",
                "tool census + schema tokens per request in hi metrics",
                &format!("trimmed {added_list} ({evidence})"),
            );
            println!("trimmed: {added_list} ({evidence})");
            println!("undo with `hi tools keep <name>`; the protected floor is never trimmed");
        }
        Some("keep") => {
            let names: Vec<&String> = args[1..]
                .iter()
                .filter(|argument| !argument.starts_with('-'))
                .collect();
            if names.is_empty() {
                bail!("usage: hi tools keep <name>…");
            }
            let mut disabled: BTreeSet<String> =
                disabled_tools(state_root).into_iter().collect();
            let mut restored = Vec::new();
            for name in names {
                if disabled.remove(name.as_str()) {
                    restored.push(name.as_str());
                } else {
                    println!("{name} was not trimmed");
                }
            }
            if !restored.is_empty() {
                save_disabled(state_root, &disabled)?;
                println!("restored to advertisement: {}", restored.join(", "));
            }
        }
        Some("list") | None => {
            let disabled: BTreeSet<String> = disabled_tools(state_root).into_iter().collect();
            println!("protected floor (never trimmable): {}", hi_tools::PROTECTED_TOOLS.join(", "));
            if disabled.is_empty() {
                println!("trimmed: none");
            } else {
                println!(
                    "trimmed: {}",
                    disabled.iter().map(String::as_str).collect::<Vec<_>>().join(", ")
                );
            }
            if let Some(sessions_dir) = sessions_dir {
                let files =
                    crate::learning_ledger::newest_session_files(sessions_dir, TRIM_SWEEP_SESSIONS);
                if files.len() >= MIN_TRIM_SESSIONS {
                    let used = crate::learning_ledger::used_tool_names(&files);
                    let candidates = trim_candidates(&used, &disabled);
                    if candidates.is_empty() {
                        println!(
                            "trim candidates: none — every trimmable tool was called in the \
                             last {} session(s)",
                            files.len()
                        );
                    } else {
                        println!(
                            "trim candidates (uncalled in the last {} session(s)): {}",
                            files.len(),
                            candidates.join(", ")
                        );
                        println!("apply with `hi tools trim <name>…`");
                    }
                } else {
                    println!(
                        "trim candidates: not enough evidence yet ({}/{MIN_TRIM_SESSIONS} \
                         sessions swept)",
                        files.len()
                    );
                }
            }
        }
        Some(other) => bail!("unknown tools subcommand {other:?} (list|trim|keep)"),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(tag: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("hi-tooltrim-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn seed_sessions(dir: &Path, count: usize) {
        for index in 0..count {
            std::fs::write(
                dir.join(format!("{index}-s.jsonl")),
                r#"{"role":"Assistant","content":[{"ToolCall":{"id":"1","name":"read","arguments":"{}"}},{"ToolCall":{"id":"2","name":"web_search","arguments":"{}"}}]}
"#,
            )
            .unwrap();
        }
    }

    #[test]
    fn protected_floor_refuses_even_force() {
        let root = scratch("floor");
        let error = run_tools_cli(
            &root,
            None,
            &["trim".into(), "read".into(), "--force".into()],
        )
        .unwrap_err();
        assert!(error.to_string().contains("protected floor"), "{error}");
        assert!(disabled_tools(&root).is_empty());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn trim_needs_evidence_unless_forced() {
        let root = scratch("evidence");
        let sessions = scratch("evidence-sessions");
        // Too few sessions: refused.
        seed_sessions(&sessions, 3);
        let error = run_tools_cli(&root, Some(&sessions), &["trim".into(), "glob".into()])
            .unwrap_err();
        assert!(error.to_string().contains("at least"), "{error}");
        // Enough sessions and the tool is dead: trimmed, and the trim records
        // itself as an intervention for the effect windows.
        seed_sessions(&sessions, MIN_TRIM_SESSIONS);
        run_tools_cli(&root, Some(&sessions), &["trim".into(), "glob".into()]).unwrap();
        assert_eq!(disabled_tools(&root), vec!["glob".to_string()]);
        let ledger =
            std::fs::read_to_string(root.join("learning/interventions.jsonl")).unwrap();
        assert!(ledger.contains("tool-trim"), "{ledger}");
        // A non-floor tool the census saw in use is refused without --force.
        let error = run_tools_cli(&root, Some(&sessions), &["trim".into(), "web_search".into()])
            .unwrap_err();
        assert!(
            error.to_string().contains("was called within"),
            "in-use tool refused: {error}"
        );
        let _ = std::fs::remove_dir_all(&root);
        let _ = std::fs::remove_dir_all(&sessions);
    }

    #[test]
    fn candidates_exclude_floor_and_flag_gated_tools() {
        let used: BTreeSet<String> = ["glob".to_string()].into();
        let disabled: BTreeSet<String> = ["repo_map".to_string()].into();
        let candidates = trim_candidates(&used, &disabled);
        assert!(!candidates.contains(&"read".to_string()), "floor");
        assert!(!candidates.contains(&"explore".to_string()), "flag-gated");
        assert!(!candidates.contains(&"task".to_string()), "flag-gated");
        assert!(!candidates.contains(&"glob".to_string()), "recently used");
        assert!(!candidates.contains(&"repo_map".to_string()), "already trimmed");
        assert!(
            candidates.contains(&"web_search".to_string()),
            "dead unconditional tools remain: {candidates:?}"
        );
    }

    #[test]
    fn keep_restores_and_unknown_names_are_refused() {
        let root = scratch("keep");
        run_tools_cli(&root, None, &["trim".into(), "glob".into(), "--force".into()]).unwrap();
        assert_eq!(disabled_tools(&root), vec!["glob".to_string()]);
        run_tools_cli(&root, None, &["keep".into(), "glob".into()]).unwrap();
        assert!(disabled_tools(&root).is_empty());
        let error = run_tools_cli(
            &root,
            None,
            &["trim".into(), "no_such_tool".into(), "--force".into()],
        )
        .unwrap_err();
        assert!(error.to_string().contains("unknown tool"), "{error}");
        let _ = std::fs::remove_dir_all(&root);
    }
}
