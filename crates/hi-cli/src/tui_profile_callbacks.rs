use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use crate::config::{self, Config, ProfileForm, ProviderName};
use crate::landing::profile_infos;
use crate::provider::{agent_provider_route, build_chain};

pub(crate) struct TuiProfileCallbacks {
    pub(crate) profiles: Vec<hi_tui::ProfileInfo>,
    pub(crate) resolver: hi_tui::ProfileResolver,
    pub(crate) saver: hi_tui::ProfileSaver,
    pub(crate) loader: hi_tui::ProfileLoader,
    pub(crate) remover: hi_tui::ProfileRemover,
}

/// Build the TUI profile callbacks over one live config snapshot. A profile
/// created or removed during the session must be visible to every subsequent
/// `/provider` operation without requiring a restart.
pub(crate) fn callbacks(file: Config, config_path: Option<PathBuf>) -> TuiProfileCallbacks {
    let profiles = profile_infos(&file);
    let file = Arc::new(Mutex::new(file));

    let resolver: hi_tui::ProfileResolver = Box::new({
        let file = Arc::clone(&file);
        move |name: &str| {
            let settings = {
                let file = file.lock().unwrap_or_else(|error| error.into_inner());
                config::resolve_named_profile(&file, name)?
            };
            let route = agent_provider_route(&settings);
            let model = settings.model.clone();
            let provider = build_chain(&settings, Vec::new());
            Ok(hi_tui::SwitchedProvider {
                provider,
                model,
                route,
                max_tokens: settings.max_tokens,
                max_tokens_explicit: settings.max_tokens_explicit,
                tool_mode: settings.tool_mode,
                local_runtime: None,
            })
        }
    });
    let saver: hi_tui::ProfileSaver = Box::new({
        let file = Arc::clone(&file);
        let config_path = config_path.clone();
        move |data: &hi_tui::ProfileFormData| {
            let provider = data.provider.parse::<ProviderName>().map_err(|error| {
                anyhow::anyhow!("invalid provider '{}': {error}", data.provider)
            })?;
            let form = ProfileForm {
                name: data.name.clone(),
                provider,
                api_key: data.api_key.clone(),
                store_as_env: data.store_as_env,
                model: data.model.clone(),
                base_url: data.base_url.clone(),
            };
            let mut file = file.lock().unwrap_or_else(|error| error.into_inner());
            // Editing an existing profile must not wipe the fields the form
            // doesn't cover (max_tokens, fallback, tool_mode, …).
            let profile = match file.profiles.get(&data.name) {
                Some(existing) => form.apply_to(existing),
                None => form.to_profile(),
            };
            config::upsert_profile(&mut file, &data.name, profile, config_path.as_deref())?;
            Ok(profile_infos(&file))
        }
    });
    let loader: hi_tui::ProfileLoader = Box::new({
        let file = Arc::clone(&file);
        move |name: &str| {
            let file = file.lock().unwrap_or_else(|error| error.into_inner());
            let profile = file
                .profiles
                .get(name)
                .ok_or_else(|| anyhow::anyhow!("no profile named '{name}'"))?;
            let form = ProfileForm::from_profile(name, profile);
            Ok(hi_tui::ProfileFormData {
                name: form.name,
                provider: form.provider.as_str().to_string(),
                api_key: form.api_key,
                store_as_env: form.store_as_env,
                model: form.model,
                base_url: form.base_url,
            })
        }
    });
    let remover: hi_tui::ProfileRemover = Box::new({
        let file = Arc::clone(&file);
        move |name: &str| {
            let mut file = file.lock().unwrap_or_else(|error| error.into_inner());
            let existed = config::remove_profile(&mut file, name, config_path.as_deref())?;
            if !existed {
                anyhow::bail!("no profile named '{name}'");
            }
            Ok(profile_infos(&file))
        }
    });

    TuiProfileCallbacks {
        profiles,
        resolver,
        saver,
        loader,
        remover,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn save_is_immediately_resolvable_without_ambient_pipe_credentials() {
        const CHILD_ROOT: &str = "HI_TEST_TUI_PROFILE_CALLBACKS_ROOT";

        // Run the credential-writing half in a dedicated process. Redirecting
        // XDG_CONFIG_HOME is process-wide, so doing it in this test thread could
        // make an unrelated parallel test read or overwrite the developer's real
        // credential store.
        let Some(root) = std::env::var_os(CHILD_ROOT) else {
            let root = tempfile::tempdir().unwrap();
            let status = std::process::Command::new(std::env::current_exe().unwrap())
                .arg("--exact")
                .arg("tui_profile_callbacks::tests::save_is_immediately_resolvable_without_ambient_pipe_credentials")
                .arg("--nocapture")
                .env(CHILD_ROOT, root.path())
                .env("HOME", root.path())
                .env("XDG_CONFIG_HOME", root.path())
                .env_remove("PIPENETWORK_API_KEY")
                .env_remove("HI_API_KEY")
                .env_remove("OPENAI_API_KEY")
                .env_remove("HI_X402_KEYPAIR")
                .env_remove("HI_X402_PASTE_SIG")
                .env_remove("HI_EXECUTION_MODE")
                .status()
                .expect("run isolated profile callback regression test");
            assert!(
                status.success(),
                "isolated regression test failed: {status}"
            );
            return;
        };

        let root = PathBuf::from(root);
        let config_path = root.join("config.toml");
        let callbacks = callbacks(Config::default(), Some(config_path.clone()));

        assert!(
            hi_ai::auth_store::load("pipenetwork").is_none(),
            "the isolated store must not contain a direct provider credential"
        );
        assert!(
            (callbacks.resolver)("pipenetwork").is_err(),
            "Pipe must not resolve before the profile credential is saved"
        );

        let profiles = (callbacks.saver)(&hi_tui::ProfileFormData {
            name: "pipenetwork".into(),
            provider: "pipenetwork".into(),
            api_key: "test-only-pipe-key".into(),
            store_as_env: false,
            model: String::new(),
            base_url: "https://api.pipenetwork.ai/v1".into(),
        })
        .expect("save isolated Pipe profile");
        assert!(profiles.iter().any(|profile| profile.name == "pipenetwork"));

        let loaded = (callbacks.loader)("pipenetwork").expect("load just-saved profile");
        assert_eq!(loaded.provider, "pipenetwork");
        assert!(
            loaded
                .api_key
                .starts_with("auth-store://profile-api-key/pipenetwork/"),
            "the form loader must expose an opaque reference, never the stored key"
        );

        let switched =
            (callbacks.resolver)("pipenetwork").expect("resolve just-saved Pipe profile");
        assert_eq!(switched.model, "pipe/deepseek-v4-flash-0731");
        assert_eq!(switched.route.label, "pipenetwork");
        assert!(
            hi_ai::auth_store::load("pipenetwork").is_none(),
            "profile save must not create a provider-wide pairing credential"
        );

        let persisted = config::read_config_file(&config_path).unwrap();
        let persisted = persisted.profiles.get("pipenetwork").unwrap();
        assert!(persisted.api_key.is_none());
        assert!(persisted.api_key_env.is_none());
        assert!(persisted.api_key_ref.as_deref().is_some_and(|reference| {
            reference.starts_with("auth-store://profile-api-key/pipenetwork/")
        }));

        let profiles = (callbacks.remover)("pipenetwork").expect("remove just-saved profile");
        assert!(!profiles.iter().any(|profile| profile.name == "pipenetwork"));
        assert!((callbacks.loader)("pipenetwork").is_err());
        assert!((callbacks.resolver)("pipenetwork").is_err());
    }
}
