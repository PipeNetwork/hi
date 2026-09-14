use super::Cli;
use crate::sentinel_exec::{sentinel_flag_error, sentinel_requested, skip_exec_for_mode};
use clap::Parser;

#[test]
fn autoharnessfix_parses_and_conflicts_with_rsi() {
    let cli = Cli::try_parse_from(["hi", "--autoharnessfix"]).unwrap();
    assert!(cli.sentinel.autoharnessfix);
    assert!(Cli::try_parse_from(["hi", "--autoharnessfix", "--rsi"]).is_err());
    assert!(Cli::try_parse_from(["hi", "--autoharnessfix", "--subagent"]).is_err());
}

#[test]
fn repair_role_does_not_exec_sentinel() {
    let _lock = crate::CWD_LOCK.lock().unwrap();
    let previous = std::env::var_os("HI_SENTINEL_ROLE");
    unsafe {
        std::env::set_var("HI_SENTINEL_ROLE", "repair");
    }
    let cli = Cli::try_parse_from(["hi", "--autoharnessfix"]).unwrap();
    let result = crate::sentinel_exec::maybe_exec_into_sentinel(&cli);
    unsafe {
        match previous {
            Some(value) => std::env::set_var("HI_SENTINEL_ROLE", value),
            None => std::env::remove_var("HI_SENTINEL_ROLE"),
        }
    }
    result.expect("ROLE=repair must skip exec rather than wrap a nested supervisor");
}

#[test]
fn show_config_and_daemon_skip_wrap() {
    let show = Cli::try_parse_from(["hi", "--autoharnessfix", "--show-config"]).unwrap();
    assert!(skip_exec_for_mode(&show));
    crate::sentinel_exec::maybe_exec_into_sentinel(&show).unwrap();

    let daemon = Cli::try_parse_from(["hi", "--autoharnessfix", "--daemon"]).unwrap();
    assert!(skip_exec_for_mode(&daemon));
    crate::sentinel_exec::maybe_exec_into_sentinel(&daemon).unwrap();
}

#[test]
fn apply_without_parent_requires_machine_enabled() {
    let _lock = crate::CWD_LOCK.lock().unwrap();
    let dir = tempfile::tempdir().unwrap();
    let previous = std::env::var_os("XDG_CONFIG_HOME");
    unsafe {
        std::env::set_var("XDG_CONFIG_HOME", dir.path());
    }
    let cli = Cli::try_parse_from(["hi", "--autoharnessfix-apply"]).unwrap();
    assert!(sentinel_flag_error(&cli).is_some());
    unsafe {
        match previous {
            Some(value) => std::env::set_var("XDG_CONFIG_HOME", value),
            None => std::env::remove_var("XDG_CONFIG_HOME"),
        }
    }
}

#[test]
fn apply_parses_without_parent_flag() {
    let cli = Cli::try_parse_from(["hi", "--autoharnessfix-apply"]).unwrap();
    assert!(cli.sentinel.autoharnessfix_apply);
    assert!(!cli.sentinel.autoharnessfix);
}

#[test]
fn apply_with_parent_flag_is_legal() {
    let cli = Cli::try_parse_from(["hi", "--autoharnessfix", "--autoharnessfix-apply"]).unwrap();
    assert!(cli.sentinel.autoharnessfix);
    assert!(cli.sentinel.autoharnessfix_apply);
    assert!(sentinel_flag_error(&cli).is_none());
}

#[test]
fn apply_with_machine_enabled_is_legal() {
    let _lock = crate::CWD_LOCK.lock().unwrap();
    let dir = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(dir.path().join("hi")).unwrap();
    std::fs::write(
        dir.path().join("hi/config.toml"),
        "[autoharnessfix]\nenabled = true\n",
    )
    .unwrap();
    let previous = std::env::var_os("XDG_CONFIG_HOME");
    unsafe {
        std::env::set_var("XDG_CONFIG_HOME", dir.path());
    }
    let cli = Cli::try_parse_from(["hi", "--autoharnessfix-apply"]).unwrap();
    let err = sentinel_flag_error(&cli);
    unsafe {
        match previous {
            Some(value) => std::env::set_var("XDG_CONFIG_HOME", value),
            None => std::env::remove_var("XDG_CONFIG_HOME"),
        }
    }
    assert!(err.is_none(), "{err:?}");
    assert!(cli.sentinel.autoharnessfix_apply);
}

#[test]
fn machine_enabled_requests_sentinel_without_cli_flag() {
    let _lock = crate::CWD_LOCK.lock().unwrap();
    let dir = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(dir.path().join("hi")).unwrap();
    std::fs::write(
        dir.path().join("hi/config.toml"),
        "[autoharnessfix]\nenabled = true\n",
    )
    .unwrap();
    let previous = std::env::var_os("XDG_CONFIG_HOME");
    unsafe {
        std::env::set_var("XDG_CONFIG_HOME", dir.path());
    }
    let cli = Cli::try_parse_from(["hi"]).unwrap();
    let requested = sentinel_requested(&cli);
    unsafe {
        match previous {
            Some(value) => std::env::set_var("XDG_CONFIG_HOME", value),
            None => std::env::remove_var("XDG_CONFIG_HOME"),
        }
    }
    assert!(requested);
}

#[test]
fn project_autoharnessfix_is_dropped_on_merge() {
    let mut machine = super::Config::default();
    let project: super::Config =
        toml::from_str("[autoharnessfix]\nenabled = true\ncheckout = \"/tmp/evil\"\n").unwrap();
    assert!(project.autoharnessfix.as_ref().is_some_and(|s| s.enabled));
    super::merge_config(&mut machine, project);
    assert!(
        machine.autoharnessfix.is_none(),
        "project hi.toml must not enable Sentinel"
    );
}
