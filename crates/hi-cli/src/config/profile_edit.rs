use super::*;

/// The fields needed to create or edit a profile, collected from the user.
/// Used by both the plain REPL prompts and the TUI form.
#[derive(Clone, Debug)]
pub struct ProfileForm {
    pub name: String,
    pub provider: ProviderName,
    pub api_key: String,
    /// Whether to store the key as a literal (`api_key`) or an env var name
    /// (`api_key_env`). The setup wizard uses env vars for cloud providers;
    /// we match that convention.
    pub store_as_env: bool,
    pub model: String,
    pub base_url: String,
}

impl Default for ProfileForm {
    fn default() -> Self {
        Self {
            name: String::new(),
            provider: ProviderName::Openai,
            api_key: String::new(),
            store_as_env: false,
            model: String::new(),
            base_url: String::new(),
        }
    }
}

impl ProfileForm {
    /// Build a `Profile` from the form fields, leaving unused fields as `None`.
    pub fn to_profile(&self) -> Profile {
        let mut p = Profile {
            provider: Some(self.provider),
            ..Default::default()
        };
        if !self.model.is_empty() {
            p.model = Some(self.model.clone());
        }
        if !self.base_url.is_empty() {
            p.base_url = Some(self.base_url.clone());
        }
        if !self.api_key.is_empty() {
            // The API-key field accepts either a literal key or the *name* of an
            // env var that holds the key. Distinguish by checking the environment:
            // if the input is a plausible env var name AND an env var with that
            // name is actually set, store it as an `api_key_env` reference;
            // otherwise treat it as a literal key. This is unambiguous — a real
            // key will never be the name of a set env var, so a pasted key that
            // happens to be all-caps+digits+underscores is stored correctly
            // instead of being mistaken for an env var name (which would fail at
            // resolve time with "env var … is not set").
            if is_env_var_reference(&self.api_key) {
                p.api_key_env = Some(self.api_key.clone());
            } else {
                p.api_key = Some(self.api_key.clone());
            }
        }
        p
    }

    /// Build a `Profile` for an *edit*: start from the existing profile so the
    /// fields the form doesn't cover (max_tokens, thinking_budget, tool_mode,
    /// compat, fallback, subagent/planner settings, mcp_url, …) survive, and
    /// overwrite only what the form actually edits.
    pub fn apply_to(&self, existing: &Profile) -> Profile {
        let form = self.to_profile();
        Profile {
            provider: form.provider,
            model: form.model,
            base_url: form.base_url,
            api_key: form.api_key,
            api_key_env: form.api_key_env,
            ..existing.clone()
        }
    }

    /// Populate the form from an existing profile (for editing).
    pub fn from_profile(name: &str, p: &Profile) -> Self {
        Self {
            name: name.to_string(),
            provider: p.provider.unwrap_or(ProviderName::Openai),
            api_key: p
                .api_key_env
                .clone()
                .or_else(|| p.api_key.clone())
                .unwrap_or_default(),
            store_as_env: p.api_key_env.is_some(),
            model: p.model.clone().unwrap_or_default(),
            base_url: p.base_url.clone().unwrap_or_default(),
        }
    }
}

// --- Layered profile persistence ------------------------------------------
//
// The session's `Config` is the *merge* of the global config and a local
// `hi.toml`, but saves must never serialize that merged view into one file:
// with a local `hi.toml` present that copied every global profile — API keys
// included — into a project file that's easy to commit, and removing a
// globally-defined profile only masked it locally until the next merge
// resurrected it. Instead, every save is a read-modify-write of exactly the
// file(s) that own the data being changed: the explicit `--config` file when
// one was given (the whole session is that single file), else the layer file
// that defines the profile (local first — it wins the merge), else the
// default writable path for brand-new profiles.

/// Read `path` (or start empty if it doesn't exist), apply `mutate` to that
/// file's own contents, and write it back.
pub(crate) fn rmw_config_file(path: &Path, mutate: impl FnOnce(&mut Config)) -> Result<()> {
    let mut file = if path.exists() {
        read_config_file(path)?
    } else {
        Config::default()
    };
    mutate(&mut file);
    save_config_to(&file, path)
}

/// Existing config layer files, highest merge precedence first (local
/// `hi.toml`, then the global config).
pub(crate) fn layer_paths() -> Vec<PathBuf> {
    let mut out = Vec::new();
    let local = local_config_path();
    if local.exists() {
        out.push(local);
    }
    if let Some(global) = default_config_path().filter(|p| p.exists()) {
        out.push(global);
    }
    out
}

/// The layer files (from `layers`, highest precedence first) that define
/// profile `name`. Files that fail to parse are skipped (the profile can't
/// have come from them).
pub(crate) fn layers_defining(layers: &[PathBuf], name: &str) -> Vec<PathBuf> {
    layers
        .iter()
        .filter(|p| {
            read_config_file(p)
                .map(|c| c.profiles.contains_key(name))
                .unwrap_or(false)
        })
        .cloned()
        .collect()
}

/// The highest-precedence layer file that defines profile `name`.
pub(crate) fn owning_path_in(layers: &[PathBuf], name: &str) -> Option<PathBuf> {
    layers_defining(layers, name).into_iter().next()
}

/// Where a change to profile `name` must be written: the explicit `--config`
/// path if given, else the layer file that defines the profile, else (a new
/// profile) the default writable path.
pub(crate) fn profile_save_target(name: &str, explicit: Option<&Path>) -> Result<PathBuf> {
    if let Some(path) = explicit {
        return Ok(path.to_path_buf());
    }
    owning_path_in(&layer_paths(), name)
        .or_else(|| writable_config_path(None))
        .ok_or_else(|| anyhow!("could not determine config path"))
}

/// Add or replace a profile in the config and save it to disk. Only `name`'s
/// entry in the owning file is touched (see module note above).
pub fn upsert_profile(
    config: &mut Config,
    name: &str,
    profile: Profile,
    explicit: Option<&Path>,
) -> Result<()> {
    validate_profile(&profile)?;
    let target = profile_save_target(name, explicit)?;
    rmw_config_file(&target, |file| {
        file.profiles.insert(name.to_string(), profile.clone());
    })?;
    config.profiles.insert(name.to_string(), profile);
    Ok(())
}

/// Add or replace a profile, select it as the default profile, and save.
pub fn upsert_profile_as_default(
    config: &mut Config,
    name: &str,
    profile: Profile,
    explicit: Option<&Path>,
) -> Result<()> {
    upsert_profile(config, name, profile, explicit)?;
    // `default_profile` must land where the merge can't shadow it: the
    // explicit file, else the highest-precedence layer (a local `hi.toml`
    // overrides the global default_profile whenever it sets one), else the
    // default writable path.
    let target = match explicit {
        Some(path) => path.to_path_buf(),
        None => layer_paths()
            .into_iter()
            .next()
            .or_else(|| writable_config_path(None))
            .ok_or_else(|| anyhow!("could not determine config path"))?,
    };
    rmw_config_file(&target, |file| {
        file.default_profile = Some(name.to_string());
    })?;
    config.default_profile = Some(name.to_string());
    Ok(())
}

/// Update only the selected model on an existing profile and save it to disk.
pub fn set_profile_model(
    config: &mut Config,
    name: &str,
    model: &str,
    explicit: Option<&Path>,
) -> Result<()> {
    let profile = config
        .profiles
        .get_mut(name)
        .ok_or_else(|| anyhow!("profile '{name}' not found in config"))?;
    profile.model = Some(model.to_string());
    validate_profile(profile)?;
    let updated = profile.clone();
    let target = profile_save_target(name, explicit)?;
    rmw_config_file(&target, |file| {
        match file.profiles.get_mut(name) {
            // Touch only the model, preserving whatever else that file says
            // about the profile.
            Some(p) => p.model = Some(model.to_string()),
            // The file doesn't define it (deleted mid-session, or a fresh
            // explicit path): write the full in-memory profile.
            None => {
                file.profiles.insert(name.to_string(), updated.clone());
            }
        }
    })
}

/// Remove a profile from the config and save. Returns `false` if the profile
/// didn't exist (caller may treat that as an error or a no-op). Without an
/// explicit path the profile is removed from *every* layer file that defines
/// it — deleting it from just one file would let the merge resurrect it from
/// the other on the next launch.
pub fn remove_profile(config: &mut Config, name: &str, explicit: Option<&Path>) -> Result<bool> {
    let in_memory = config.profiles.remove(name).is_some();
    let targets: Vec<PathBuf> = match explicit {
        Some(path) => vec![path.to_path_buf()],
        None => layers_defining(&layer_paths(), name),
    };
    if !in_memory && targets.is_empty() {
        return Ok(false);
    }
    for path in &targets {
        rmw_config_file(path, |file| {
            file.profiles.remove(name);
        })?;
    }
    Ok(true)
}

/// Sanity-check a profile before saving. Currently validates that the base URL
/// doesn't include an endpoint path (the provider appends `/chat/completions`
/// and `/models` itself — a common copy-paste mistake is to paste the full
/// endpoint URL, which produces 404s).
pub(crate) fn validate_profile(profile: &Profile) -> Result<()> {
    if let Some(url) = &profile.base_url {
        let trimmed = url.trim_end_matches('/');
        for suffix in ["/chat/completions", "/completions", "/messages"] {
            if trimmed.ends_with(suffix) {
                bail!(
                    "base_url looks like a full endpoint path (ends with '{suffix}'). \
                     The provider appends the endpoint path itself — use just the base, \
                     e.g. 'http://localhost:11434/v1' not 'http://localhost:11434/v1{suffix}'."
                );
            }
        }
    }
    Ok(())
}

/// Does `s` look like an env var *name* (not a value)? Env var names are
/// uppercase ASCII letters, digits, and underscores, must contain at least one
/// underscore (single-word names like `PATH` are rare for API-key vars and we
/// err toward treating short all-caps tokens as literal keys), and aren't
/// absurdly long. This is only a pre-filter — the real decision in
/// `to_profile` also requires the var to be set in the environment.
pub(crate) fn looks_like_env_var_name(s: &str) -> bool {
    !s.is_empty()
        && s.len() <= 64
        && s.contains('_')
        && s.chars()
            .all(|c| c.is_ascii_uppercase() || c == '_' || c.is_ascii_digit())
}

/// Does `s` look like an env var *name* and is an env var with that name
/// actually set? Used by the wizards to label the field and pre-set
/// `store_as_env`; `to_profile` makes the final decision the same way.
pub(crate) fn is_env_var_reference(s: &str) -> bool {
    looks_like_env_var_name(s) && std::env::var(s).is_ok_and(|v| !v.is_empty())
}

/// Shown when `hi` is run with nothing configured. Actionable, not terse.
pub(crate) const ONBOARDING: &str = "no model configured. Get started with one of:

  pipenetwork.ai:   PIPENETWORK_API_KEY=...  hi --provider pipenetwork \"...\"
  Local (Ollama):   hi --provider ollama -m qwen2.5-coder \"...\"
  xAI (Grok):       XAI_API_KEY=...  hi --provider xai \"...\"

Or run `hi` on a real terminal for the interactive setup wizard.
Or set HI_MODEL, or add a profile in ~/.config/hi/config.toml (see README).
Tip: interactive sessions use the full-screen interface by default; pass --plain for the line REPL.";

/// Infer a provider + model from API keys present in the environment.
pub(crate) fn auto_select() -> Option<(ProviderName, String)> {
    let set = |name: &str| std::env::var(name).is_ok_and(|v| !v.is_empty());
    if set("PIPENETWORK_API_KEY") {
        Some((ProviderName::Pipenetwork, "ipop/coder-balanced".into()))
    } else if set("ANTHROPIC_API_KEY") {
        Some((ProviderName::Anthropic, "claude-sonnet-4-20250514".into()))
    } else if set("XAI_API_KEY") {
        Some((ProviderName::Xai, "grok-4.3".into()))
    } else {
        None
    }
}

pub(crate) fn resolve_api_key(
    cli: &Cli,
    profile: Option<&Profile>,
    provider: ProviderName,
) -> Result<String> {
    if let Some(key) = &cli.api_key {
        return Ok(key.clone());
    }
    // A launcher (best-of / delegate / fleet child) passes the parent's already
    // resolved key here so the child authenticates with the SAME key the parent
    // used. The child re-resolves config from scratch and would otherwise let a
    // default-profile literal `api_key` shadow the parent's key (e.g. when the
    // parent ran with `--profile alt` or `--api-key`), causing silent auth
    // failures. It's passed in the environment, not argv, so it isn't exposed in
    // `ps` — hence it must win over the profile here rather than being a
    // last-resort candidate like `HI_API_KEY`.
    if let Ok(key) = std::env::var("HI_FORCE_API_KEY")
        && !key.is_empty()
    {
        return Ok(key);
    }
    resolve_api_key_for(profile, provider)
}

/// API key for a profile/provider, independent of CLI flags (used for fallback
/// profiles, whose keys come from their own profile or the environment).
pub(crate) fn resolve_api_key_for(
    profile: Option<&Profile>,
    provider: ProviderName,
) -> Result<String> {
    if let Some(key) = profile.and_then(|p| p.api_key.clone()) {
        return Ok(key);
    }
    if let Some(env_name) = profile.and_then(|p| p.api_key_env.as_ref()) {
        return std::env::var(env_name).map_err(|_| {
            anyhow!(
                "env var {env_name} (from profile) is not set.\n\
                 Fix: either `export {env_name}=your-key` and restart hi, or re-run the\n\
                 setup wizard (`hi` with no config) to store the key directly in the config file."
            )
        });
    }
    // A stored OAuth credential beats ambient env vars: signing in is an
    // explicit act, so it should win over a key that merely happens to be
    // exported. Explicit profile config above still takes precedence over both.
    // Possibly expired — the provider's token source refreshes it in place.
    if let Some(stored) = hi_ai::auth_store::load(provider.as_str()) {
        return Ok(stored.access);
    }
    let candidates = provider.key_envs();
    for name in candidates {
        if let Ok(value) = std::env::var(name)
            && !value.is_empty()
        {
            return Ok(value);
        }
    }
    // A local Ollama server ignores the key, so don't require one.
    if matches!(provider, ProviderName::Ollama) {
        return Ok("ollama".into());
    }
    let names: Vec<String> = candidates.iter().map(|s| s.to_string()).collect();
    let hint = match names.len() {
        0 => "an API key env var".to_string(),
        1 => names[0].clone(),
        _ => format!(
            "{} or {}",
            names[..names.len() - 1].join(", "),
            names[names.len() - 1]
        ),
    };
    if matches!(provider, ProviderName::Xai) {
        bail!(
            "no xAI credential: run `/login xai` to sign in with a grok.com \
             subscription (SuperGrok or X Premium), or set {hint}"
        );
    }
    if matches!(provider, ProviderName::Pipenetwork) {
        bail!(
            "no pipenetwork credential: run `/login pipenetwork` to sign in and \
             receive an API key, or set {hint}"
        );
    }
    bail!("no API key: pass --api-key or set {hint}");
}
