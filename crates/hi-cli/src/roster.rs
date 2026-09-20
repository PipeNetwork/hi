//! Needs-attention session roster: unfinished plans, pending turns, lock holders.

use std::fs;
use std::io::{self, IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Result, bail};
use hi_harness::{SessionIndex, SessionLockStatus, inspect_session_lock, load_or_refresh_index};

use crate::paths;

/// Printed when several unfinished sessions share a directory and stdin is not a TTY.
#[derive(Debug)]
pub struct AmbiguousResume;

impl std::fmt::Display for AmbiguousResume {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "several unfinished sessions; pick one with hi --resume <id>"
        )
    }
}

impl std::error::Error for AmbiguousResume {}

#[derive(Clone, Debug)]
pub struct RosterEntry {
    pub id: String,
    pub path: PathBuf,
    pub digest: String,
    pub workspace: PathBuf,
    pub age: String,
    pub title: String,
    pub dashboard: bool,
    pub index: SessionIndex,
    pub lock: SessionLockStatus,
}

impl RosterEntry {
    pub fn needs_attention(&self) -> bool {
        self.index.needs_attention() || self.lock.is_held()
    }

    pub fn flags(&self) -> Vec<String> {
        let mut flags = Vec::new();
        if self.dashboard {
            flags.push("dashboard".into());
        }
        if let Some(plan) = self.index.plan_flag() {
            flags.push(plan);
        }
        if let Some(turn) = self.index.pending_turn {
            flags.push(format!("PENDING t{turn}"));
        }
        if self.index.drive_paused {
            flags.push("drive paused".into());
        }
        if let Some(lock) = self.lock.flag_text() {
            flags.push(lock);
        }
        flags
    }

    pub fn resume_command(&self) -> String {
        format!(
            "cd {} && hi --resume {}",
            shell_quote(&self.workspace),
            self.id
        )
    }

    pub fn reason(&self) -> String {
        self.flags().join(" · ")
    }
}

pub fn print_roster(all: bool) -> Result<()> {
    let Some(root) = paths::data_root() else {
        println!("no session directory");
        return Ok(());
    };
    let mut entries = scan_sessions(&root);
    if !all {
        entries.retain(|entry| entry.needs_attention());
    }
    if entries.is_empty() {
        if all {
            println!("no sessions in {}", root.join("projects").display());
        } else {
            println!("no sessions need attention (`hi sessions --all` lists everything)");
        }
        return Ok(());
    }
    print_entries(&entries);
    Ok(())
}

pub fn print_entries(entries: &[RosterEntry]) {
    let mut last_workspace: Option<&Path> = None;
    for entry in entries {
        if last_workspace.map(Path::new) != Some(entry.workspace.as_path()) {
            println!("{}", entry.workspace.display());
            last_workspace = Some(&entry.workspace);
        }
        println!("  {}", entry.resume_command());
        let flags = entry.reason();
        if !flags.is_empty() {
            println!("    {flags}");
        } else if !entry.title.is_empty() {
            println!("    {}", entry.title);
        }
    }
}

pub fn scan_sessions(data_root: &Path) -> Vec<RosterEntry> {
    let projects = data_root.join("projects");
    let mut entries = Vec::new();
    let Ok(buckets) = fs::read_dir(&projects) else {
        return entries;
    };
    for bucket in buckets.flatten() {
        let digest = bucket.file_name().to_string_lossy().into_owned();
        let workspace = fs::read_to_string(bucket.path().join("workspace"))
            .ok()
            .map(|text| PathBuf::from(text.trim()))
            .filter(|path| !path.as_os_str().is_empty())
            .unwrap_or_else(|| PathBuf::from(format!("<digest:{digest}>")));
        collect_jsonl(
            &mut entries,
            &bucket.path().join("sessions"),
            &digest,
            &workspace,
            false,
        );
        collect_jsonl(
            &mut entries,
            &bucket.path().join("dashboard").join("sessions"),
            &digest,
            &workspace,
            true,
        );
    }
    entries.sort_by(|a, b| {
        b.index
            .mtime_unix_ms
            .cmp(&a.index.mtime_unix_ms)
            .then_with(|| a.id.cmp(&b.id))
    });
    entries
}

pub fn unfinished_in_digest(data_root: &Path, digest: &str) -> Vec<RosterEntry> {
    scan_sessions(data_root)
        .into_iter()
        .filter(|entry| entry.digest == digest && entry.needs_attention())
        .collect()
}

pub fn pick_unfinished(
    data_root: &Path,
    digest: &str,
    latest: Option<&Path>,
    stdin_tty: bool,
) -> Result<Option<PathBuf>> {
    let unfinished = unfinished_in_digest(data_root, digest);
    match unfinished.len() {
        0 => Ok(latest.map(Path::to_path_buf)),
        1 => {
            let entry = &unfinished[0];
            if latest.is_none_or(|path| path != entry.path.as_path()) {
                eprintln!(
                    "resuming unfinished session {} ({})",
                    entry.id,
                    entry.reason()
                );
            }
            Ok(Some(entry.path.clone()))
        }
        _ => {
            if stdin_tty && io::stdout().is_terminal() {
                Ok(Some(prompt_picker(&unfinished)?))
            } else {
                eprintln!("pick one with hi --resume <id>");
                bail!(AmbiguousResume);
            }
        }
    }
}

fn prompt_picker(entries: &[RosterEntry]) -> Result<PathBuf> {
    for (i, entry) in entries.iter().enumerate() {
        println!("{:>2}. {}  {}", i + 1, entry.id, entry.reason());
    }
    eprint!("resume which? [1-{}] ", entries.len());
    let _ = io::stderr().flush();
    let mut line = String::new();
    io::stdin().read_line(&mut line)?;
    let choice = line.trim().parse::<usize>().unwrap_or(0);
    entries
        .get(choice.saturating_sub(1))
        .map(|entry| entry.path.clone())
        .ok_or_else(|| anyhow::anyhow!(AmbiguousResume))
}

fn collect_jsonl(
    entries: &mut Vec<RosterEntry>,
    dir: &Path,
    digest: &str,
    workspace: &Path,
    dashboard: bool,
) {
    let Ok(read) = fs::read_dir(dir) else {
        return;
    };
    for entry in read.flatten() {
        let path = entry.path();
        if path.extension().is_none_or(|ext| ext != "jsonl") {
            continue;
        }
        let modified = fs::metadata(&path)
            .and_then(|m| m.modified())
            .unwrap_or(UNIX_EPOCH);
        let index = load_or_refresh_index(&path);
        let lock = inspect_session_lock(&path);
        entries.push(RosterEntry {
            id: hi_harness::session_id_from_path(&path),
            path,
            digest: digest.to_string(),
            workspace: workspace.to_path_buf(),
            age: age_label(modified),
            title: index.title.clone(),
            dashboard,
            index,
            lock,
        });
    }
}

fn age_label(modified: SystemTime) -> String {
    SystemTime::now()
        .duration_since(modified)
        .map(|d| {
            let secs = d.as_secs();
            if secs < 60 {
                format!("{secs}s")
            } else if secs < 3600 {
                format!("{}m", secs / 60)
            } else if secs < 86400 {
                format!("{}h", secs / 3600)
            } else {
                format!("{}d", secs / 86400)
            }
        })
        .unwrap_or_else(|_| "?".into())
}

fn shell_quote(path: &Path) -> String {
    let s = path.display().to_string();
    if s.chars()
        .all(|c| c.is_ascii_alphanumeric() || "/._-".contains(c))
    {
        s
    } else {
        format!("'{}'", s.replace('\'', "'\\''"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hi_tools::{PlanStatus, PlanStep};

    fn write_plan_session(dir: &Path, id: &str, steps: Vec<PlanStep>, user: &str) {
        fs::create_dir_all(dir).unwrap();
        let path = dir.join(format!("{id}.jsonl"));
        let mut session = hi_harness::JsonlSession::create(&path).unwrap();
        session
            .record_messages(&[hi_ai::Message::user(user)])
            .unwrap();
        session.record_plan(&steps).unwrap();
        drop(session);
    }

    fn data_layout() -> (tempfile::TempDir, PathBuf, PathBuf) {
        let tmp = tempfile::tempdir().unwrap();
        let data = tmp.path().join("share").join("hi");
        let digest = "abcdabcdabcdabcd";
        let project = data.join("projects").join(digest);
        fs::create_dir_all(project.join("sessions")).unwrap();
        fs::write(project.join("workspace"), "/Users/david/chat\n").unwrap();
        (tmp, data, project)
    }

    #[test]
    fn default_list_hides_completed_plans() {
        let (_tmp, data, project) = data_layout();
        let sessions = project.join("sessions");
        write_plan_session(
            &sessions,
            "open-plan",
            vec![
                PlanStep {
                    title: "Forward HISTORY pagination".into(),
                    status: PlanStatus::Done,
                },
                PlanStep {
                    title: "Next".into(),
                    status: PlanStatus::Pending,
                },
            ],
            "keep going",
        );
        write_plan_session(
            &sessions,
            "done-plan",
            vec![PlanStep {
                title: "All done".into(),
                status: PlanStatus::Done,
            }],
            "finished",
        );
        let all = scan_sessions(&data);
        assert_eq!(all.len(), 2, "{all:?}");
        let attention: Vec<_> = all.iter().filter(|e| e.needs_attention()).collect();
        assert_eq!(attention.len(), 1);
        assert_eq!(attention[0].id, "open-plan");
        assert!(
            attention[0].reason().contains("PLAN 1/2"),
            "{}",
            attention[0].reason()
        );
        let all_ids: Vec<_> = all.iter().map(|e| e.id.as_str()).collect();
        assert!(all_ids.contains(&"done-plan"));
    }

    #[test]
    fn title_skips_harness_nudges() {
        let (_tmp, data, project) = data_layout();
        let sessions = project.join("sessions");
        fs::create_dir_all(&sessions).unwrap();
        let path = sessions.join("nudged.jsonl");
        let mut session = hi_harness::JsonlSession::create(&path).unwrap();
        session
            .record_messages(&[
                hi_ai::Message::user("[hi:context — session state] blob"),
                hi_ai::Message::user("[hi:nudge] continue"),
                hi_ai::Message::user("real work"),
            ])
            .unwrap();
        drop(session);
        let entry = scan_sessions(&data)
            .into_iter()
            .find(|e| e.id == "nudged")
            .unwrap();
        assert_eq!(entry.title, "real work");
    }

    #[test]
    fn dashboard_rows_are_tagged() {
        let (_tmp, data, project) = data_layout();
        let dash = project.join("dashboard").join("sessions");
        write_plan_session(
            &dash,
            "dash-1",
            vec![PlanStep {
                title: "Fleet row".into(),
                status: PlanStatus::Active,
            }],
            "dashboard task",
        );
        let entry = scan_sessions(&data)
            .into_iter()
            .find(|e| e.id == "dash-1")
            .unwrap();
        assert!(entry.dashboard);
        assert!(entry.flags().iter().any(|f| f == "dashboard"));
        assert!(entry.resume_command().contains("hi --resume dash-1"));
    }
}
