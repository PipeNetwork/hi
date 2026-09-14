//! Relaunch argv after a verified apply: session file, no positional prompt.

use std::path::{Path, PathBuf};

use hi_liveness::{ENV_RESUME_INCOMPLETE, TurnIntent};

use crate::args::relaunch_args;
use crate::config::SupervisorConfig;
use crate::paths;

#[derive(Clone, Debug)]
pub struct RelaunchPlan {
    pub incident_id: String,
    pub kind: String,
    pub session_path: Option<PathBuf>,
    pub pre_checkpoint: Option<String>,
    pub turn_intent: Option<PathBuf>,
    pub sidecar_hi: PathBuf,
}

pub fn plan_from_incident(
    incident_dir: &Path,
    sidecar_hi: PathBuf,
    id: &str,
    kind: &str,
) -> RelaunchPlan {
    let intent = read_turn_intent(&incident_dir.join("turn-intent.json"));
    let incident = read_json_value(&incident_dir.join("incident.json"));
    let session_path = intent
        .as_ref()
        .and_then(|intent| intent.session_path.clone())
        .or_else(|| string_field(incident.as_ref(), "session_path"))
        .filter(|path| !path.is_empty())
        .map(PathBuf::from);
    let pre_checkpoint = intent
        .as_ref()
        .and_then(|intent| intent.pre_checkpoint.clone())
        .or_else(|| string_field(incident.as_ref(), "pre_checkpoint"))
        .filter(|id| !id.is_empty());
    let turn_intent = {
        let path = incident_dir.join("turn-intent.json");
        path.is_file().then_some(path)
    };
    RelaunchPlan {
        incident_id: id.to_string(),
        kind: kind.to_string(),
        session_path,
        pre_checkpoint,
        turn_intent,
        sidecar_hi,
    }
}

pub fn sidecar_or_current(cfg: &SupervisorConfig) -> PathBuf {
    let sidecar = paths::sidecar_bin_dir(&cfg.state_dir).join("hi");
    if sidecar.is_file() {
        sidecar
    } else {
        cfg.hi_binary.clone()
    }
}

pub fn next_generation(cfg: &SupervisorConfig, plan: &RelaunchPlan) -> SupervisorConfig {
    let mut next = cfg.clone();
    next.generation = cfg.generation.saturating_add(1);
    next.child_program = plan.sidecar_hi.clone();
    next.hi_binary = plan.sidecar_hi.clone();
    next.child_args = relaunch_args(&cfg.child_args, plan.session_path.as_deref());
    next.seed_turn_intent = plan.turn_intent.clone();
    next.extra_env
        .retain(|(key, _)| key != ENV_RESUME_INCOMPLETE);
    next.extra_env
        .push((ENV_RESUME_INCOMPLETE.to_string(), "1".into()));
    next
}

pub fn continuing_line(plan: &RelaunchPlan) -> String {
    format!(
        "hi: internal harness error ({}, {}) repaired and verified; continuing.",
        plan.incident_id, plan.kind
    )
}

fn read_turn_intent(path: &Path) -> Option<TurnIntent> {
    let bytes = std::fs::read(path).ok()?;
    serde_json::from_slice(&bytes).ok()
}

fn read_json_value(path: &Path) -> Option<serde_json::Value> {
    let text = std::fs::read_to_string(path).ok()?;
    serde_json::from_str(&text).ok()
}

fn string_field(value: Option<&serde_json::Value>, key: &str) -> Option<String> {
    value
        .and_then(|value| value.get(key))
        .and_then(|value| value.as_str())
        .map(str::to_string)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::args::strip_sentinel_args;
    use crate::fsutil;
    use hi_liveness::{SCHEMA_VERSION, TurnIntent};
    use std::ffi::OsString;

    #[test]
    fn fixture_session_relaunch_drops_prompt_and_keeps_session_file() {
        let dir = tempfile::tempdir().unwrap();
        let session = dir.path().join("pending.jsonl");
        std::fs::write(
            &session,
            concat!(
                r#"{"role":"user","content":[{"type":"text","text":"fix the parser"}]}"#,
                "\n",
                r#"{"type":"pending_turn","turn_index":1,"started_unix_ms":1,"pre_checkpoint":null}"#,
                "\n"
            ),
        )
        .unwrap();
        let stripped = strip_sentinel_args(
            ["--autoharnessfix", "fix the parser"]
                .into_iter()
                .map(OsString::from),
        );
        let args = relaunch_args(&stripped.args, Some(session.as_path()));
        assert!(
            !args.iter().any(|arg| arg == "fix the parser"),
            "relaunch must not keep the positional operand: {args:?}"
        );
        assert_eq!(args[0], "--session-file");
        assert_eq!(args[1], session.as_os_str());
    }

    #[test]
    fn plan_reads_oneshot_session_from_turn_intent() {
        let dir = tempfile::tempdir().unwrap();
        let intent_path = dir.path().join("turn-intent.json");
        let intent = TurnIntent {
            schema_version: SCHEMA_VERSION,
            turn_index: 1,
            prompt: "fix the parser".into(),
            session_path: Some("/tmp/s.jsonl".into()),
            pre_checkpoint: Some("internal:v1:abc".into()),
            started_unix_ms: 1,
            workspace: "/tmp/ws".into(),
            oneshot: true,
            plain: true,
        };
        fsutil::write_0600(&intent_path, &serde_json::to_vec(&intent).unwrap()).unwrap();
        let plan = plan_from_incident(
            dir.path(),
            PathBuf::from("/sidecar/hi"),
            "incident-1",
            "crash",
        );
        assert_eq!(
            plan.session_path.as_deref(),
            Some(Path::new("/tmp/s.jsonl"))
        );
        assert_eq!(plan.pre_checkpoint.as_deref(), Some("internal:v1:abc"));
        assert_eq!(plan.turn_intent.as_deref(), Some(intent_path.as_path()));
    }

    #[test]
    fn next_generation_sets_resume_env_and_drops_prompt() {
        use crate::config::{MonitorConfig, SupervisorConfig};

        let dir = tempfile::tempdir().unwrap();
        let cfg = SupervisorConfig {
            child_program: PathBuf::from("/bin/hi"),
            child_args: vec![OsString::from("--plain"), OsString::from("fix the parser")],
            hi_binary: PathBuf::from("/bin/hi"),
            original_argv: vec![
                "hi".into(),
                "--autoharnessfix".into(),
                "fix the parser".into(),
            ],
            checkout: None,
            apply: true,
            monitor: MonitorConfig::default(),
            state_dir: dir.path().join("state"),
            workspace: dir.path().to_path_buf(),
            generation: 0,
            inherit_stdio: true,
            extra_env: Vec::new(),
            once: false,
            seed_turn_intent: None,
        };
        let plan = RelaunchPlan {
            incident_id: "incident-1".into(),
            kind: "crash".into(),
            session_path: Some(PathBuf::from("/tmp/s.jsonl")),
            pre_checkpoint: None,
            turn_intent: None,
            sidecar_hi: PathBuf::from("/sidecar/hi"),
        };
        let next = next_generation(&cfg, &plan);
        assert_eq!(next.generation, 1);
        assert_eq!(next.hi_binary, PathBuf::from("/sidecar/hi"));
        assert!(
            !next.child_args.iter().any(|arg| arg == "fix the parser"),
            "relaunch argv still has the positional operand: {:?}",
            next.child_args
        );
        assert!(
            next.extra_env
                .iter()
                .any(|(key, value)| key == ENV_RESUME_INCOMPLETE && value == "1")
        );
        assert_eq!(next.child_args[0], "--plain");
    }
}
