use super::*;

#[test]
fn user_config_load_migrates_all_legacy_environment_credentials_to_refs() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("config.toml");
    std::fs::write(
        &path,
        r#"
[profiles.default]
provider = "openai"
api_key_env = "HI_TEST_PROFILE_KEY"

[sync]
api_key_env = "HI_TEST_SYNC_KEY"

[rsi]
api_key_env = "HI_TEST_RSI_KEY"

[outcome]
api_key_env = "HI_TEST_OUTCOME_KEY"
"#,
    )
    .unwrap();

    let loaded = file::read_config(&path).unwrap();
    assert_eq!(
        loaded.profiles["default"].api_key_env.as_deref(),
        Some("HI_TEST_PROFILE_KEY"),
        "the current run retains backward-compatible legacy projection"
    );

    let persisted = read_config_file(&path).unwrap();
    assert_eq!(
        persisted.profiles["default"].api_key_ref.as_deref(),
        Some("env://HI_TEST_PROFILE_KEY")
    );
    assert_eq!(
        persisted.sync.unwrap().api_key_ref.as_deref(),
        Some("env://HI_TEST_SYNC_KEY")
    );
    assert_eq!(
        persisted.rsi.unwrap().api_key_ref.as_deref(),
        Some("env://HI_TEST_RSI_KEY")
    );
    assert_eq!(
        persisted.outcome.unwrap().api_key_ref.as_deref(),
        Some("env://HI_TEST_OUTCOME_KEY")
    );
    assert!(
        !std::fs::read_to_string(path)
            .unwrap()
            .contains("api_key_env")
    );
}

#[test]
fn a_missing_store_entry_is_a_typed_error_naming_the_reference() {
    let reference = "auth-store://profile-api-key/openai/never-stored-by-anyone";
    let err = resolve_credential_reference(reference, false, true).unwrap_err();
    let missing = err
        .downcast_ref::<MissingStoredCredential>()
        .expect("startup recovery matches on the type");
    assert_eq!(missing.reference, reference);
    assert!(err.to_string().contains("was not found"), "{err}");

    let err = resolve_credential_reference("env://HI_NEVER_SET_KEY_998", false, true).unwrap_err();
    assert!(
        err.downcast_ref::<MissingStoredCredential>().is_none(),
        "an unset environment variable is not a lost store entry: {err}"
    );
}

#[test]
fn repointing_a_profile_replaces_only_its_credential() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("config.toml");
    let mut config = Config {
        default_profile: Some("default".into()),
        profiles: std::collections::HashMap::from([(
            "default".into(),
            Profile {
                provider: Some(ProviderName::Pipenetwork),
                model: Some("ipop/coder-balanced".into()),
                max_tokens: Some(4096),
                api_key_ref: Some("auth-store://profile-api-key/pipenetwork/gone".into()),
                ..Default::default()
            },
        )]),
        ..Default::default()
    };
    save_config_to(&config, &path).unwrap();

    repoint_profile_credential(
        &mut config,
        "default",
        "auth-store://pipenetwork",
        Some(&path),
    )
    .unwrap();

    let profile = &config.profiles["default"];
    assert_eq!(
        profile.api_key_ref.as_deref(),
        Some("auth-store://pipenetwork")
    );
    assert_eq!(profile.model.as_deref(), Some("ipop/coder-balanced"));
    assert_eq!(profile.max_tokens, Some(4096));
    let persisted = read_config_file(&path).unwrap();
    assert_eq!(
        persisted.profiles["default"].api_key_ref.as_deref(),
        Some("auth-store://pipenetwork")
    );
    assert_eq!(persisted.profiles["default"].max_tokens, Some(4096));
    assert_eq!(persisted.default_profile.as_deref(), Some("default"));
    assert!(!std::fs::read_to_string(&path).unwrap().contains("gone"));
}

#[test]
fn set_profile_model_updates_model_and_migrates_legacy_credential() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("config.toml");
    let mut config = Config {
        default_profile: Some("default".into()),
        profiles: std::collections::HashMap::from([(
            "default".into(),
            Profile {
                provider: Some(ProviderName::Pipenetwork),
                model: Some("pipe/auto-coder".into()),
                api_key_env: Some("HI_TEST_PROFILE_KEY".into()),
                ..Default::default()
            },
        )]),
        ..Default::default()
    };

    set_profile_model(&mut config, "default", "ipop/coder-balanced", Some(&path))
        .expect("set model");

    let profile = &config.profiles["default"];
    assert_eq!(profile.model.as_deref(), Some("ipop/coder-balanced"));
    assert!(profile.api_key.is_none());
    assert!(profile.api_key_env.is_none());
    assert_eq!(
        profile.api_key_ref.as_deref(),
        Some("env://HI_TEST_PROFILE_KEY")
    );
    let text = std::fs::read_to_string(path).unwrap();
    assert!(text.contains("model = \"ipop/coder-balanced\""));
    assert!(text.contains("api_key_ref = \"env://HI_TEST_PROFILE_KEY\""));
    assert!(!text.contains("api_key_env"));
}
