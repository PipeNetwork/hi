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

/// Record an intervention on behalf of another CLI surface — `hi tools trim`
/// logs itself here so the effect windows in `hi metrics` cover it like any
/// hand-recorded change.
pub(crate) fn record_intervention(
    state_root: &Path,
    name: &str,
    watch: &str,
    note: &str,
) -> Result<()> {
    append(
        state_root,
        &Intervention {
            ts: now_ts(),
            name: name.into(),
            watch: watch.into(),
            evidence_state: "exercised".into(),
            note: note.into(),
        },
    )
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

/// Findings vs total turns inside the `window` seconds each side of `pivot`.
/// `None` until both windows contain at least one timestamped turn — a rate
/// over an empty denominator is not evidence, and pre-timestamp records
/// (ts 0) never land in a window.
fn window_rates(
    findings: &[u64],
    outcomes: &[u64],
    pivot: u64,
    window: u64,
) -> Option<((usize, usize), (usize, usize))> {
    let count = |ts_list: &[u64], from: u64, to: u64| {
        ts_list
            .iter()
            .filter(|ts| **ts >= from && **ts < to)
            .count()
    };
    let lo = pivot.saturating_sub(window);
    let hi = pivot.saturating_add(window);
    let before = (count(findings, lo, pivot), count(outcomes, lo, pivot));
    let after = (count(findings, pivot, hi), count(outcomes, pivot, hi));
    if before.1 == 0 || after.1 == 0 {
        return None;
    }
    Some((before, after))
}

/// Effect windows span ±14 days: long enough to accumulate turns on both
/// sides, short enough that unrelated drift doesn't dominate.
const EFFECT_WINDOW_SECS: u64 = 14 * 24 * 3600;

/// The learning sections of `hi metrics`: findings summary (with hint
/// recurrence), interventions with computed before/after effect, and the
/// advertised-but-unused tool census.
pub(crate) fn print_learning_report(sessions_dir: &Path, state_root: &Path) {
    // Findings: counts by stop reason, newest last.
    let findings: Vec<hi_agent::learning::Finding> =
        std::fs::read_to_string(state_root.join("learning").join("findings.jsonl"))
            .map(|raw| {
                raw.lines()
                    .filter_map(|line| serde_json::from_str(line).ok())
                    .collect()
            })
            .unwrap_or_default();
    if !findings.is_empty() {
        let mut by_reason: BTreeMap<String, usize> = BTreeMap::new();
        for finding in &findings {
            *by_reason
                .entry(format!("{:?}", finding.stop_reason))
                .or_default() += 1;
        }
        let summary: Vec<String> = by_reason
            .iter()
            .map(|(reason, count)| format!("{count} {reason}"))
            .collect();
        println!(
            "findings: {} bad turn(s) — {}",
            findings.len(),
            summary.join(" · ")
        );
        if let Some(finding) = findings.last() {
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
        // Hint efficacy: a bad turn whose shape matches the hint that was
        // steering it is direct evidence the hint is not enough.
        let mut recurred: BTreeMap<String, usize> = BTreeMap::new();
        for finding in &findings {
            let stop = format!("{:?}", finding.stop_reason);
            let matches_hint = finding.hint_active.as_ref().is_some_and(|hint| {
                hint == &stop || finding.failure_shape.as_deref() == Some(hint.as_str())
            });
            if matches_hint {
                *recurred.entry(stop).or_default() += 1;
            }
        }
        if !recurred.is_empty() {
            let parts: Vec<String> = recurred
                .iter()
                .map(|(shape, count)| format!("{shape} {count}x"))
                .collect();
            println!(
                "  recurred under own hint: {} — a shape that keeps failing under its \
                 hint needs a structural fix, not steering",
                parts.join(" · ")
            );
        }
    }

    // Interventions, each with its computed before/after effect. The pivot is
    // the intervention's FIRST record (when the change shipped); `support`
    // upgrades re-append later and must not move the measurement window.
    let ledger_raw = std::fs::read_to_string(ledger_path(state_root)).unwrap_or_default();
    let mut latest: BTreeMap<String, Intervention> = BTreeMap::new();
    let mut first_ts: BTreeMap<String, u64> = BTreeMap::new();
    for line in ledger_raw.lines() {
        if let Ok(record) = serde_json::from_str::<Intervention>(line) {
            first_ts.entry(record.name.clone()).or_insert(record.ts);
            latest.insert(record.name.clone(), record);
        }
    }
    let session_files = newest_session_files(sessions_dir, 100);
    if !latest.is_empty() {
        let finding_ts: Vec<u64> = findings.iter().map(|finding| finding.ts).collect();
        let outcome_ts = turn_outcome_timestamps(&session_files);
        println!("interventions ({}):", latest.len());
        for (name, record) in &latest {
            print_intervention(record);
            let pivot = first_ts.get(name).copied().unwrap_or(record.ts);
            match window_rates(&finding_ts, &outcome_ts, pivot, EFFECT_WINDOW_SECS) {
                Some(((bad_before, total_before), (bad_after, total_after))) => println!(
                    "    effect(±14d): before {bad_before}/{total_before} bad turn(s), \
                     after {bad_after}/{total_after}"
                ),
                None => println!(
                    "    effect(±14d): not yet measurable — needs timestamped turns on \
                     both sides of the ship date"
                ),
            }
        }
    }

    // Census: advertised tools never called across the recent sessions the
    // tuning sweep also reads. Advertised is not used — unused specs still
    // cost schema tokens on every request. Already-trimmed tools are out of
    // the request, so they leave the census; floor tools are shown but never
    // suggested for trimming.
    let trimmed = crate::tool_trim::disabled_tools(state_root);
    let census_files: Vec<std::path::PathBuf> = session_files.iter().take(20).cloned().collect();
    let sessions = census_files.len();
    let used = used_tool_names(&census_files);
    if sessions == 0 {
        return;
    }
    let mut dead: Vec<(String, usize)> = Vec::new();
    for spec in hi_tools::TOOL_SPECS.iter() {
        if !used.contains(spec.name.as_str()) && !trimmed.contains(&spec.name) {
            let cost = hi_ai::estimate_tool_schema_tokens(std::slice::from_ref(spec))
                .min(usize::MAX as u64) as usize;
            dead.push((spec.name.clone(), cost));
        }
    }
    if !trimmed.is_empty() {
        println!("tools trimmed from advertisement: {}", trimmed.join(", "));
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
        // Suggest only what the trim gate will accept: dead in the census AND
        // silent across the full deeper sweep — a name the gate would bounce
        // ("called within the last N sessions") is noise here.
        let used_sweep = used_tool_names(&session_files);
        let trimmable: Vec<&str> = dead
            .iter()
            .map(|(name, _)| name.as_str())
            .filter(|name| {
                !hi_tools::PROTECTED_TOOLS.contains(name)
                    && !crate::tool_trim::CONDITIONAL_TOOLS.contains(name)
                    && !used_sweep.contains(*name)
            })
            .collect();
        if !trimmable.is_empty() {
            println!("  apply with: hi tools trim {}", trimmable.join(" "));
        }
    }
}

/// The newest `limit` session JSONL files, most recent first.
pub(crate) fn newest_session_files(sessions_dir: &Path, limit: usize) -> Vec<std::path::PathBuf> {
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
    files.into_iter().map(|(_, path)| path).collect()
}

/// Timestamps of turn-outcome records across `files` — the total-turn
/// denominator for intervention effect windows. Session meta lines are
/// internally tagged (`{"type":"turn_outcome","ts":…}`). Records written
/// before timestamps existed (ts 0 via serde default) are skipped.
fn turn_outcome_timestamps(files: &[std::path::PathBuf]) -> Vec<u64> {
    let mut out = Vec::new();
    for path in files {
        let Ok(raw) = std::fs::read_to_string(path) else {
            continue;
        };
        for line in raw.lines() {
            if !line.contains("\"turn_outcome\"") {
                continue;
            }
            let Ok(value) = serde_json::from_str::<serde_json::Value>(line) else {
                continue;
            };
            if value.get("type").and_then(|tag| tag.as_str()) != Some("turn_outcome") {
                continue;
            }
            if let Some(ts) = value.get("ts").and_then(|ts| ts.as_u64())
                && ts > 0
            {
                out.push(ts);
            }
        }
    }
    out
}

/// Tool names called at least once across `files`.
pub(crate) fn used_tool_names(files: &[std::path::PathBuf]) -> BTreeSet<String> {
    let mut used = BTreeSet::new();
    for path in files {
        let Ok(raw) = std::fs::read_to_string(path) else {
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
    used
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
        let files = newest_session_files(&dir, 20);
        assert_eq!(files.len(), 1);
        let used = used_tool_names(&files);
        assert!(used.contains("read"));
        assert!(!used.contains("bash"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn effect_windows_need_turns_on_both_sides() {
        let outcomes = vec![100, 200, 900, 1100, 1500];
        let findings = vec![200, 1100];
        // Pivot 1000, window 800: before counts ts in [200,1000), after in
        // [1000,1800).
        let ((bad_before, total_before), (bad_after, total_after)) =
            window_rates(&findings, &outcomes, 1000, 800).expect("both windows populated");
        assert_eq!(
            (bad_before, total_before),
            (1, 2),
            "200 and 900; 200 is bad"
        );
        assert_eq!(
            (bad_after, total_after),
            (1, 2),
            "1100 and 1500; 1100 is bad"
        );
        // No timestamped turns after the pivot: no verdict, not a 0% claim.
        assert!(window_rates(&findings, &outcomes, 2000, 300).is_none());
    }

    #[test]
    fn outcome_timestamps_parse_real_session_meta_shape() {
        let dir = scratch("outcomes");
        // Lines mirror the on-disk format (`tag = "type"`, snake_case) —
        // the first scanner shipped matching an externally-tagged shape that
        // never occurs in real files, and its fixture confirmed the bug.
        std::fs::write(
            dir.join("1-a.jsonl"),
            r#"{"type":"turn_outcome","ts":1234,"status":"completed","verification":"not_applicable","review":"not_required","stop_reason":"no_applicable_verification"}
{"type":"turn_outcome","ts":0,"status":"completed","verification":"passed","review":"not_required","stop_reason":"completed"}
{"type":"usage","input_tokens":1,"output_tokens":1,"cache_read_tokens":0,"cache_creation_tokens":0,"estimated":false}
"#,
        )
        .unwrap();
        let files = newest_session_files(&dir, 20);
        assert_eq!(turn_outcome_timestamps(&files), vec![1234]);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
