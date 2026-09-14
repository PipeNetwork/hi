//! `HI_SENTINEL_RESUME_INCOMPLETE=1` wins over `cli.prompt` / TUI startup_prompt.

use hi_liveness::{ENV_RESUME_INCOMPLETE, ENV_TURN_INTENT, TurnIntent, env_flag_on};

use crate::config::Cli;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ResumePlan {
    pub ignore_prompt: bool,
    pub startup_prompt: Option<String>,
    pub run_incomplete: bool,
    pub exit_after_resume: bool,
}

pub(crate) fn plan(cli: &Cli, prompt: Option<&String>) -> ResumePlan {
    if !resume_incomplete_set() {
        return ResumePlan {
            ignore_prompt: false,
            startup_prompt: prompt.cloned(),
            run_incomplete: false,
            exit_after_resume: false,
        };
    }
    let intent = read_turn_intent();
    let oneshot = intent.as_ref().is_some_and(|intent| intent.oneshot);
    let plain = intent.as_ref().is_some_and(|intent| intent.plain) || cli.plain;
    ResumePlan {
        ignore_prompt: true,
        startup_prompt: None,
        run_incomplete: true,
        exit_after_resume: oneshot || plain,
    }
}

fn resume_incomplete_set() -> bool {
    std::env::var(ENV_RESUME_INCOMPLETE)
        .ok()
        .is_some_and(|value| env_flag_on(&value))
}

fn read_turn_intent() -> Option<TurnIntent> {
    let path = std::env::var_os(ENV_TURN_INTENT)?;
    let bytes = std::fs::read(path).ok()?;
    serde_json::from_slice(&bytes).ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;
    use hi_liveness::SCHEMA_VERSION;

    fn with_env<T>(resume: bool, intent: Option<&TurnIntent>, body: impl FnOnce() -> T) -> T {
        let _lock = crate::CWD_LOCK.lock().unwrap();
        let prev_resume = std::env::var_os(ENV_RESUME_INCOMPLETE);
        let prev_intent = std::env::var_os(ENV_TURN_INTENT);
        let dir = tempfile::tempdir().unwrap();
        unsafe {
            if resume {
                std::env::set_var(ENV_RESUME_INCOMPLETE, "1");
            } else {
                std::env::remove_var(ENV_RESUME_INCOMPLETE);
            }
        }
        if let Some(intent) = intent {
            let path = dir.path().join("turn-intent.json");
            std::fs::write(&path, serde_json::to_vec(intent).unwrap()).unwrap();
            unsafe {
                std::env::set_var(ENV_TURN_INTENT, &path);
            }
        } else {
            unsafe {
                std::env::remove_var(ENV_TURN_INTENT);
            }
        }
        let result = body();
        unsafe {
            match prev_resume {
                Some(value) => std::env::set_var(ENV_RESUME_INCOMPLETE, value),
                None => std::env::remove_var(ENV_RESUME_INCOMPLETE),
            }
            match prev_intent {
                Some(value) => std::env::set_var(ENV_TURN_INTENT, value),
                None => std::env::remove_var(ENV_TURN_INTENT),
            }
        }
        result
    }

    fn intent(oneshot: bool, plain: bool) -> TurnIntent {
        TurnIntent {
            schema_version: SCHEMA_VERSION,
            turn_index: 1,
            prompt: "fix the parser".into(),
            session_path: Some("/tmp/s.jsonl".into()),
            pre_checkpoint: None,
            started_unix_ms: 1,
            workspace: "/tmp".into(),
            oneshot,
            plain,
        }
    }

    #[test]
    fn resume_unsets_startup_prompt_and_ignores_cli_prompt() {
        let cli = Cli::try_parse_from(["hi", "--autoharnessfix", "fix the parser"]).unwrap();
        let prompt = cli.prompt.clone();
        let planned = with_env(true, Some(&intent(false, false)), || {
            plan(&cli, prompt.as_ref())
        });
        assert!(planned.ignore_prompt);
        assert_eq!(planned.startup_prompt, None);
        assert!(planned.run_incomplete);
        assert!(!planned.exit_after_resume);
    }

    #[test]
    fn plain_oneshot_exits_after_resume() {
        let cli = Cli::try_parse_from(["hi", "--plain", "fix the parser"]).unwrap();
        let prompt = cli.prompt.clone();
        let planned = with_env(true, Some(&intent(true, true)), || {
            plan(&cli, prompt.as_ref())
        });
        assert!(planned.ignore_prompt);
        assert_eq!(planned.startup_prompt, None);
        assert!(planned.exit_after_resume);
    }

    #[test]
    fn without_resume_env_keeps_startup_prompt() {
        let cli = Cli::try_parse_from(["hi", "fix the parser"]).unwrap();
        let prompt = cli.prompt.clone();
        let planned = with_env(false, None, || plan(&cli, prompt.as_ref()));
        assert!(!planned.ignore_prompt);
        assert_eq!(planned.startup_prompt.as_deref(), Some("fix the parser"));
        assert!(!planned.run_incomplete);
    }
}
