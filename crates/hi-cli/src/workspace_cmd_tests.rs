use super::*;

fn parse(args: &[&str]) -> std::result::Result<WorkspaceCli, clap::Error> {
    WorkspaceCli::try_parse_from(std::iter::once("hi workspace").chain(args.iter().copied()))
}

#[test]
fn parser_accepts_the_preactivation_recovery_surface() {
    let list = parse(&["recover", "list", "--session", "session-1", "--json"]).unwrap();
    assert!(matches!(
        list.command,
        WorkspaceCommand::Recover {
            command: RecoveryCommand::List(SessionOutputArgs { json: true, .. })
        }
    ));

    let inspect = parse(&["recover", "inspect", "abc123", "--session", "session-1"]).unwrap();
    assert!(matches!(
        inspect.command,
        WorkspaceCommand::Recover {
            command: RecoveryCommand::Inspect(InspectArgs { recovery_id, .. })
        } if recovery_id == "abc123"
    ));

    let export = parse(&[
        "recover",
        "export",
        "abc123",
        "--session",
        "session-1",
        "--to",
        "/tmp/recovery.tar.zst",
    ])
    .unwrap();
    assert!(matches!(
        export.command,
        WorkspaceCommand::Recover {
            command: RecoveryCommand::Export(ExportArgs { recovery_id, .. })
        } if recovery_id == "abc123"
    ));

    let discard = parse(&[
        "recover",
        "discard",
        "abc123",
        "--session",
        "session-1",
        "--confirm",
        "abc123",
    ])
    .unwrap();
    assert!(matches!(
        discard.command,
        WorkspaceCommand::Recover {
            command: RecoveryCommand::Discard(DiscardArgs { recovery_id, confirm, .. })
        } if recovery_id == "abc123" && confirm == "abc123"
    ));
}

#[test]
fn parser_accepts_fresh_remote_export() {
    let export = parse(&[
        "export",
        "--session",
        "session-1",
        "--to",
        "/tmp/exported-workspace",
        "--revision",
        "HEAD",
    ])
    .unwrap();
    assert!(matches!(
        export.command,
        WorkspaceCommand::Export(RemoteExportArgs { revision: Some(value), .. }) if value == "HEAD"
    ));
    assert_eq!(parse_remote_revision(Some("HEAD")).unwrap(), None);
    assert!(parse_remote_revision(Some("not-a-revision")).is_err());
}

#[test]
fn parser_accepts_migration_and_lease_commands() {
    assert!(matches!(
        parse(&["takeover", "--session", "session-1", "--json"])
            .unwrap()
            .command,
        WorkspaceCommand::Takeover(RequiredSessionOutputArgs { json: true, .. })
    ));
    assert!(matches!(
        parse(&["detach", "--session", "session-1", "--if-clean"])
            .unwrap()
            .command,
        WorkspaceCommand::Detach(_)
    ));
    assert!(matches!(
        parse(&[
            "import",
            "--session",
            "session-1",
            "--from",
            "/tmp/source",
            "--preview",
        ])
        .unwrap()
        .command,
        WorkspaceCommand::Import(_)
    ));
    assert!(matches!(
        parse(&["recover", "retry", "cache-1", "--session", "session-1"])
            .unwrap()
            .command,
        WorkspaceCommand::Recover {
            command: RecoveryCommand::Retry(_)
        }
    ));
}

#[test]
fn parser_requires_session_destination_and_confirmation() {
    assert!(parse(&["export", "--to", "/tmp/workspace"]).is_err());
    assert!(parse(&["recover", "export", "abc123", "--session", "session-1"]).is_err());
    assert!(parse(&["recover", "discard", "abc123", "--session", "session-1"]).is_err());
}

#[test]
fn parser_accepts_credential_free_local_status_and_recovery() {
    assert!(matches!(
        parse(&["status", "--json"]).unwrap().command,
        WorkspaceCommand::Status(SessionOutputArgs {
            session: None,
            json: true
        })
    ));
    assert!(matches!(
        parse(&["recover", "list"]).unwrap().command,
        WorkspaceCommand::Recover {
            command: RecoveryCommand::List(SessionOutputArgs { session: None, .. })
        }
    ));
    assert!(matches!(
        parse(&["recover", "inspect", "stable-id"]).unwrap().command,
        WorkspaceCommand::Recover {
            command: RecoveryCommand::Inspect(InspectArgs { session: None, .. })
        }
    ));
    assert!(matches!(
        parse(&[
            "recover",
            "discard",
            "stable-id",
            "--confirm",
            "blake3:proof",
            "--accept-current-bytes",
        ])
        .unwrap()
        .command,
        WorkspaceCommand::Recover {
            command: RecoveryCommand::Discard(DiscardArgs {
                session: None,
                accept_current_bytes: true,
                ..
            })
        }
    ));
    assert!(parse(&["takeover"]).is_err());
}

#[test]
fn interactive_recovery_parser_preserves_legacy_syntax_and_adds_retry() {
    assert!(matches!(
        parse_interactive_recovery("list").unwrap(),
        InteractiveRecoveryCommand::List
    ));
    assert!(matches!(
        parse_interactive_recovery("inspect stable-id").unwrap(),
        InteractiveRecoveryCommand::Inspect("stable-id")
    ));
    assert!(matches!(
        parse_interactive_recovery("retry stable-id").unwrap(),
        InteractiveRecoveryCommand::Retry("stable-id")
    ));
    assert!(matches!(
        parse_interactive_recovery("export cache-alias /tmp/archive with spaces.tar.zst").unwrap(),
        InteractiveRecoveryCommand::Export {
            recovery_id: "cache-alias",
            to: "/tmp/archive with spaces.tar.zst"
        }
    ));
    assert!(matches!(
        parse_interactive_recovery("discard stable-id --confirm blake3:whole-cache").unwrap(),
        InteractiveRecoveryCommand::Discard {
            recovery_id: "stable-id",
            confirm: "blake3:whole-cache"
        }
    ));
    assert!(parse_interactive_recovery("retry one two").is_err());
}

#[tokio::test]
async fn interactive_recovery_keeps_active_workspace_mutations_blocked() {
    let client = hi_pipefs::PipeFsClient::new(hi_pipefs::PipeFsClientConfig::new(
        "https://sync.example",
        "secret",
    ))
    .unwrap();
    for command in [
        "retry stable-id",
        "export cache-alias /tmp/recovery.tar.zst",
        "discard stable-id --confirm blake3:whole-cache",
    ] {
        let sync_config = crate::sync::SyncConfig {
            base_url: "https://sync.example".into(),
            api_key: "secret".into(),
            machine_id: Some("test-machine".into()),
            cwd_digest: None,
        };
        let error = run_pipefs_recovery_alias(
            &client,
            &client.cache_scope(),
            "session-1",
            &sync_config,
            true,
            command,
        )
        .await
        .unwrap_err();
        assert!(error.to_string().contains("turn PipeFS off"));
    }
}

#[test]
fn config_is_global_and_may_appear_after_the_subcommand() {
    let cli = parse(&[
        "status",
        "--session",
        "session-1",
        "--config",
        "/tmp/hi-recovery.toml",
    ])
    .unwrap();
    assert_eq!(cli.config, Some(PathBuf::from("/tmp/hi-recovery.toml")));
}

#[test]
fn authority_environment_values_must_be_a_complete_pair() {
    let error = require_pair(
        Some("https://sync.example"),
        None,
        "HI_SYNC_BASE_URL and HI_SYNC_API_KEY",
    )
    .unwrap_err();
    assert!(error.to_string().contains("must both be set"));
    assert_eq!(
        require_pair(
            Some(" https://sync.example "),
            Some(" secret "),
            "environment",
        )
        .unwrap(),
        ("https://sync.example", "secret")
    );
}

#[test]
fn trusted_sync_config_may_resolve_a_named_machine_credential() {
    let section = crate::config::SyncSection {
        base_url: Some("https://sync.example/v1".to_string()),
        api_key_env: Some("HI_TEST_RECOVERY_KEY".to_string()),
        ..crate::config::SyncSection::default()
    };
    let (base_url, api_key) = authority_from_sync_section(&section, |name| {
        (name == "HI_TEST_RECOVERY_KEY").then(|| "secret".to_string())
    })
    .unwrap();
    assert_eq!(base_url, "https://sync.example/v1");
    assert_eq!(api_key, "secret");
}

#[test]
fn project_sync_config_cannot_select_a_credential_environment_variable() {
    let section = crate::config::SyncSection {
        project_local: true,
        base_url: Some("https://sync.example/v1".to_string()),
        api_key_env: Some("UNTRUSTED_RECOVERY_KEY".to_string()),
        ..crate::config::SyncSection::default()
    };
    let error = authority_from_sync_section(&section, |_| {
        panic!("project-local credential variable must not be read")
    })
    .unwrap_err();
    assert!(error.to_string().contains("project-local"));
}

#[test]
fn pipefs_client_rejects_an_unsafe_recovery_authority() {
    let error = build_cache_authority(
        "http://remote.example/v1",
        "secret",
        AuthoritySource::Config,
    )
    .err()
    .expect("plaintext remote authority must be rejected");
    assert!(
        error
            .to_string()
            .contains("validating the PipeFS cache authority")
    );
}

#[test]
fn recovery_json_never_contains_authority_credentials() {
    let view = RecoveryListView {
        schema_version: OUTPUT_SCHEMA_VERSION,
        session_id: "session-1".to_string(),
        authority_source: AuthoritySource::Environment,
        recovery_caches: vec![RecoveryCacheView {
            id: "cache-id".to_string(),
            confirmation_digest: Some("blake3:abc".to_string()),
            path: "/private/cache".to_string(),
            workspace_root: Some("/private/cache/workspace".to_string()),
            phase: Some(hi_pipefs::WorkspacePhase::Pending),
            logical_size_bytes: 42,
            pending_archive_bytes: 17,
            last_error: None,
        }],
        journal_recoveries: Vec::new(),
    };
    let json = serde_json::to_string(&view).unwrap();
    assert!(json.contains("\"authority_source\":\"environment\""));
    assert!(json.contains("\"phase\":\"pending\""));
    assert!(!json.contains("api_key"));
    assert!(!json.contains("base_url"));
}
