//! Credential persistence tests must never use the developer's private store.
use super::*;

fn run_with_isolated_credentials(test: &str) -> bool {
    const CHILD: &str = "HI_TEST_CREDENTIAL_MIGRATION";
    if std::env::var(CHILD).as_deref() == Ok(test) {
        return false;
    }
    let config_home = tempfile::tempdir().unwrap();
    let status = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact", test, "--nocapture"])
        .env(CHILD, test)
        .env("XDG_CONFIG_HOME", config_home.path())
        .status()
        .expect("run credential migration in an isolated process");
    assert!(
        status.success(),
        "isolated credential regression failed: {status}"
    );
    true
}

#[test]
fn to_profile_env_var_name_that_is_not_set_stored_as_literal() {
    if run_with_isolated_credentials(std::thread::current().name().unwrap()) {
        return;
    }
    // An input that looks like an env var name but no such env var is set
    // is treated as a literal key (the user pasted a key, not a var name).
    use crate::config::{Config, ProfileForm, read_config, save_config_to};
    let name = "HI_NEVER_SET_KEY_999";
    assert!(
        std::env::var(name).is_err(),
        "precondition: var must not be set"
    );
    let form = ProfileForm {
        name: "work".into(),
        provider: ProviderName::Openai,
        api_key: name.into(),
        store_as_env: true,
        model: "gpt-4o".into(),
        base_url: String::new(),
    };
    let p = form.to_profile();
    assert_eq!(p.api_key.as_deref(), Some(name));
    assert!(p.api_key_env.is_none());

    // The in-memory form preserves the explicit literal identity. Persistence
    // seals that literal into the private credential store.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("config.toml");
    let mut config = Config::default();
    config.profiles.insert("work".into(), p);
    save_config_to(&config, &path).unwrap();
    let loaded = read_config(&path).unwrap();
    let loaded_profile = loaded.profiles.get("work").unwrap();
    assert_eq!(loaded_profile.api_key.as_deref(), Some(name));
    assert!(loaded_profile.api_key_env.is_none());
    let persisted = read_config_file(&path).unwrap();
    let persisted_profile = persisted.profiles.get("work").unwrap();
    assert!(persisted_profile.api_key.is_none());
    assert!(persisted_profile.api_key_env.is_none());
    let reference = persisted_profile
        .api_key_ref
        .as_deref()
        .expect("credential-store reference");
    let key = reference
        .strip_prefix("auth-store://")
        .expect("private credential-store reference");
    assert_eq!(
        hi_ai::auth_store::load(key).map(|credential| credential.access),
        Some(name.into())
    );
    let text = std::fs::read_to_string(path).unwrap();
    assert!(!text.contains(name), "literal leaked into config: {text}");
    hi_ai::auth_store::delete(key).unwrap();
}

#[test]
fn migrate_moves_bogus_api_key_env_to_literal() {
    if run_with_isolated_credentials(std::thread::current().name().unwrap()) {
        return;
    }
    // Simulate a config written by the old buggy wizard: a literal key
    // stored under api_key_env. Loading repairs the runtime projection and
    // seals the persistent copy into the private credential store.
    use crate::config::{Config, Profile, read_config, save_config_to};
    let dir = std::env::temp_dir().join(format!(
        "hi-migrate-test-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("config.toml");
    let config = Config {
        default_profile: Some("default".into()),
        profiles: {
            let mut m = std::collections::HashMap::new();
            m.insert(
                "default".into(),
                Profile {
                    provider: Some(ProviderName::Pipenetwork),
                    model: Some("ipop/coder-balanced".into()),
                    api_key_env: Some("api_c55ffaeda6574cdb".into()),
                    ..Default::default()
                },
            );
            m
        },
        ..Default::default()
    };
    // No env var named "api_c55ffaeda6574cdb" is set, so this is bogus.
    assert!(std::env::var("api_c55ffaeda6574cdb").is_err());
    save_config_to(&config, &path).unwrap();
    let loaded = read_config(&path).unwrap();
    let p = loaded.profiles.get("default").unwrap();
    assert_eq!(p.api_key.as_deref(), Some("api_c55ffaeda6574cdb"));
    assert!(p.api_key_env.is_none(), "bogus env ref must be cleared");

    let persisted = crate::config::read_config_file(&path).unwrap();
    let p = persisted.profiles.get("default").unwrap();
    assert!(p.api_key.is_none());
    assert!(p.api_key_env.is_none());
    let reference = p
        .api_key_ref
        .as_deref()
        .expect("credential-store reference");
    let key = reference
        .strip_prefix("auth-store://")
        .expect("private credential-store reference");
    assert_eq!(
        hi_ai::auth_store::load(key).map(|credential| credential.access),
        Some("api_c55ffaeda6574cdb".into())
    );
    let text = std::fs::read_to_string(&path).unwrap();
    assert!(
        !text.contains("api_c55ffaeda6574cdb"),
        "literal leaked: {text}"
    );
    hi_ai::auth_store::delete(key).unwrap();
    let _ = std::fs::remove_dir_all(&dir);
}
