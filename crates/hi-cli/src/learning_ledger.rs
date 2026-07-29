//! `hi intervention` + the learning sections of `hi metrics`.
//!
//! Two append-only ledgers under `<state-root>/learning/`, plus a
//! dead-capability census over recent sessions:
//!
//! - `findings.jsonl` — written by hi-agent when a turn ends badly; rendered
//!   here so failure patterns surface without transcript spelunking.
//! - `interventions.jsonl` — one record per shipped harness change, with the
//!   metric it should move and an evidence state on the ladder
//!   `present → wired → exercised → outcome-supported`. A change starts at
//!   what its tests prove (`exercised`); only a later comparable result
//!   upgrades it (`hi intervention support`). Configured is not used; used
//!   once is not proven better.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use anyhow::{Context, Result, bail};

/// Evidence ladder for an intervention claim, weakest to strongest.
pub(crate) const EVIDENCE_STATES: &[&str] = &["present", "wired", "exercised", "outcome-supported"];

#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub(crate) struct Intervention {
    pub ts: u64,
    pub name: String,
    /// What to watch to judge the effect (free text naming a metric).
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub watch: String,
    pub evidence_state: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub note: String,
}

fn now_ts() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn ledger_path(state_root: &Path) -> std::path::PathBuf {
    state_root.join("learning").join("interventions.jsonl")
}

fn append(state_root: &Path, record: &Intervention) -> Result<()> {
    let path = ledger_path(state_root);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .with_context(|| format!("opening {}", path.display()))?;
    use std::io::Write;
    writeln!(file, "{}", serde_json::to_string(record)?)?;
    Ok(())
}

/// Latest record per intervention name (the ledger is append-only; upgrades
/// re-append under the same name).
pub(crate) fn current(state_root: &Path) -> Vec<Intervention> {
    let Ok(raw) = std::fs::read_to_string(ledger_path(state_root)) else {
        return Vec::new();
    };
    let mut latest: BTreeMap<String, Intervention> = BTreeMap::new();
    for line in raw.lines() {
        if let Ok(record) = serde_json::from_str::<Intervention>(line) {
            latest.insert(record.name.clone(), record);
        }
    }
    latest.into_values().collect()
}

/// `hi intervention <add|support|list> …`
pub(crate) fn run_intervention_cli(state_root: &Path, args: &[String]) -> Result<()> {
    match args.first().map(String::as_str) {
        Some("add") => {
            let name = args.get(1).filter(|n| !n.trim().is_empty());
            let Some(name) = name else {
                bail!(
                    "usage: hi intervention add <name> [--watch <metric note>] [--state <state>]"
                );
            };
            let watch = flag_value(args, "--watch").unwrap_or_default();
            let state = flag_value(args, "--state").unwrap_or_else(|| "exercised".into());
            if !EVIDENCE_STATES.contains(&state.as_str()) {
                bail!("--state must be one of: {}", EVIDENCE_STATES.join(", "));
            }
            append(
                state_root,
                &Intervention {
                    ts: now_ts(),
                    name: name.clone(),
                    watch,
                    evidence_state: state.clone(),
                    note: String::new(),
                },
            )?;
            println!("recorded intervention '{name}' at evidence state '{state}'");
        }
        Some("support") => {
            let Some(name) = args.get(1).filter(|n| !n.trim().is_empty()) else {
                bail!("usage: hi intervention support <name> --note <later comparable evidence>");
            };
            let note = flag_value(args, "--note").unwrap_or_default();
            if note.trim().is_empty() {
                bail!(
                    "outcome-supported requires --note describing the later comparable result — \
                     that evidence is the whole point of the upgrade"
                );
            }
            let known: BTreeSet<String> = current(state_root)
                .into_iter()
                .map(|record| record.name)
                .collect();
            if !known.contains(name) {
                bail!("unknown intervention '{name}' — `hi intervention list` shows the ledger");
            }
            append(
                state_root,
                &Intervention {
                    ts: now_ts(),
                    name: name.clone(),
                    watch: String::new(),
                    evidence_state: "outcome-supported".into(),
                    note,
                },
            )?;
            println!("'{name}' upgraded to outcome-supported");
        }
        Some("list") | None => {
            let records = current(state_root);
            if records.is_empty() {
                println!(
                    "no interventions recorded — `hi intervention add <name> --watch <metric>` \
                     after shipping a harness change"
                );
            }
            for record in records {
                print_intervention(&record);
            }
        }
        Some(other) => bail!("unknown intervention subcommand {other:?} (add|support|list)"),
    }
    Ok(())
}

fn flag_value(args: &[String], flag: &str) -> Option<String> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

fn print_intervention(record: &Intervention) {
    let watch = if record.watch.is_empty() {
        String::new()
    } else {
        format!(" · watch: {}", record.watch)
    };
    let note = if record.note.is_empty() {
        String::new()
    } else {
        format!(" · {}", record.note)
    };
    println!("  [{}] {}{watch}{note}", record.evidence_state, record.name);
}

/// The learning sections of `hi metrics`: findings summary, open
/// interventions, and the advertised-but-unused tool census.
pub(crate) fn print_learning_report(sessions_dir: &Path, state_root: &Path) {
    // Findings: counts by stop reason, newest last.
    let findings_path = state_root.join("learning").join("findings.jsonl");
    if let Ok(raw) = std::fs::read_to_string(&findings_path) {
        let mut by_reason: BTreeMap<String, usize> = BTreeMap::new();
        let mut last = None;
        let mut total = 0usize;
        for line in raw.lines() {
            if let Ok(finding) = serde_json::from_str::<hi_agent::learning::Finding>(line) {
                *by_reason
                    .entry(format!("{:?}", finding.stop_reason))
                    .or_default() += 1;
                total += 1;
                last = Some(finding);
            }
        }
        if total > 0 {
            let summary: Vec<String> = by_reason
                .iter()
                .map(|(reason, count)| format!("{count} {reason}"))
                .collect();
            println!("findings: {total} bad turn(s) — {}", summary.join(" · "));
            if let Some(finding) = last {
                let reason = finding
                    .review_unavailable_reason
                    .as_deref()
                    .unwrap_or(&finding.last_stall_reason);
                let at = match (&finding.session_id, finding.turn) {
                    (Some(id), Some(turn)) => format!(" @ {id} turn {turn}"),
                    (Some(id), None) => format!(" @ {id}"),
                    _ => String::new(),
                };
                println!(
                    "  latest: {:?}/{:?} ({reason}){at}",
                    finding.status, finding.stop_reason
                );
            }
        }
    }

    // Interventions.
    let records = current(state_root);
    if !records.is_empty() {
        println!("interventions ({}):", records.len());
        for record in records {
            print_intervention(&record);
        }
    }

    // Census: advertised tools never called across the recent sessions the
    // tuning sweep also reads. Advertised is not used — unused specs still
    // cost schema tokens on every request.
    let (used, sessions) = used_tool_names(sessions_dir, 20);
    if sessions == 0 {
        return;
    }
    let mut dead: Vec<(String, usize)> = Vec::new();
    for spec in hi_tools::TOOL_SPECS.iter() {
        if !used.contains(spec.name.as_str()) {
            let cost = serde_json::to_string(&spec.parameters)
                .map(|s| (s.len() + spec.description.len()) / 4)
                .unwrap_or(0);
            dead.push((spec.name.clone(), cost));
        }
    }
    if dead.is_empty() {
        println!(
            "tool census: every advertised tool was exercised in the last {sessions} session(s)"
        );
    } else {
        dead.sort_by_key(|(_, cost)| std::cmp::Reverse(*cost));
        let total: usize = dead.iter().map(|(_, cost)| cost).sum();
        let names: Vec<String> = dead
            .iter()
            .map(|(name, cost)| format!("{name} (~{cost} tok)"))
            .collect();
        println!(
            "tool census: {} advertised tool(s) never called in the last {sessions} session(s) — \
             ~{total} schema tokens per request: {}",
            dead.len(),
            names.join(", ")
        );
    }
}

/// Tool names called at least once across the newest `limit` session files.
fn used_tool_names(sessions_dir: &Path, limit: usize) -> (BTreeSet<String>, usize) {
    let mut files: Vec<(std::time::SystemTime, std::path::PathBuf)> = Vec::new();
    if let Ok(entries) = std::fs::read_dir(sessions_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().is_some_and(|ext| ext == "jsonl")
                && let Ok(meta) = entry.metadata()
                && let Ok(modified) = meta.modified()
            {
                files.push((modified, path));
            }
        }
    }
    files.sort_by_key(|(modified, _)| std::cmp::Reverse(*modified));
    files.truncate(limit);
    let mut used = BTreeSet::new();
    let swept = files.len();
    for (_, path) in files {
        let Ok(raw) = std::fs::read_to_string(&path) else {
            continue;
        };
        for line in raw.lines() {
            if !line.contains("ToolCall") {
                continue;
            }
            let Ok(value) = serde_json::from_str::<serde_json::Value>(line) else {
                continue;
            };
            let Some(content) = value.get("content").and_then(|c| c.as_array()) else {
                continue;
            };
            for block in content {
                if let Some(name) = block
                    .get("ToolCall")
                    .and_then(|tc| tc.get("name"))
                    .and_then(|n| n.as_str())
                {
                    used.insert(name.to_string());
                }
            }
        }
    }
    (used, swept)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(tag: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("hi-ledger-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn intervention_ledger_is_append_only_and_latest_wins() {
        let root = scratch("iv");
        run_intervention_cli(
            &root,
            &[
                "add".into(),
                "digest-quiet-format".into(),
                "--watch".into(),
                "verify failures digested".into(),
            ],
        )
        .unwrap();
        run_intervention_cli(
            &root,
            &[
                "support".into(),
                "digest-quiet-format".into(),
                "--note".into(),
                "next 3 verify failures all digested".into(),
            ],
        )
        .unwrap();
        let records = current(&root);
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].evidence_state, "outcome-supported");
        // Two lines on disk — history preserved.
        let raw = std::fs::read_to_string(root.join("learning/interventions.jsonl")).unwrap();
        assert_eq!(raw.lines().count(), 2);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn support_requires_note_and_known_name() {
        let root = scratch("guard");
        assert!(
            run_intervention_cli(&root, &["support".into(), "ghost".into()]).is_err(),
            "outcome claims need evidence"
        );
        run_intervention_cli(&root, &["add".into(), "x".into()]).unwrap();
        assert!(
            run_intervention_cli(
                &root,
                &[
                    "support".into(),
                    "ghost".into(),
                    "--note".into(),
                    "n".into()
                ]
            )
            .is_err(),
            "unknown names are rejected"
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn census_reads_tool_calls_from_session_lines() {
        let dir = scratch("census");
        std::fs::write(
            dir.join("1-a.jsonl"),
            r#"{"role":"Assistant","content":[{"ToolCall":{"id":"1","name":"read","arguments":"{}"}}]}
{"role":"Tool","content":[{"ToolResult":{"call_id":"1","output":"x"}}]}
"#,
        )
        .unwrap();
        let (used, swept) = used_tool_names(&dir, 20);
        assert_eq!(swept, 1);
        assert!(used.contains("read"));
        assert!(!used.contains("bash"));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
