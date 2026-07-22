use super::*;

/// Fully-resolved settings used to build a provider and run the agent.
#[derive(Debug)]
pub struct Settings {
    pub provider: ProviderName,
    pub model: String,
    pub base_url: String,
    pub mcp_url: Option<String>,
    pub api_key: String,
    pub max_tokens: u32,
    pub max_tokens_explicit: bool,
    pub thinking_budget: Option<u32>,
    pub reasoning_effort: Option<ReasoningEffort>,
    pub tool_mode: ToolMode,
    pub compat: CompatMode,
    pub curate_skills: bool,
    pub explore_subagents: bool,
    /// Off / Risk (default) / On — see [`hi_agent::WriteSubagentPolicy`].
    pub write_subagents: hi_agent::WriteSubagentPolicy,
    pub planner_model: Option<String>,
    pub skeptic_model: Option<String>,
    pub moa: hi_ai::MoaConfig,
    pub api_unix_socket: Option<PathBuf>,
}

/// Apply precedence to produce the effective [`Settings`].
pub fn resolve(cli: &Cli, config: &Config) -> Result<Settings> {
    config.moa.validate()?;
    // Workspace last-session restores the provider/model from the previous
    // interactive exit unless the CLI explicitly overrides profile/model/provider.
    let last = if cli.profile.is_none() && cli.model.is_none() && cli.provider.is_none() {
        load_last_session(Path::new("."))
    } else {
        None
    };
    let last_profile = last
        .as_ref()
        .and_then(|s| s.profile.as_deref())
        .filter(|name| config.profiles.contains_key(*name));
    // A last-session snapshot without a profile means the user was on a provider
    // preset (`/provider xai`). Don't fall back to `default_profile` for routing
    // or the preset choice would be silently discarded on the next launch.
    let last_is_preset = last.is_some() && last_profile.is_none() && cli.profile.is_none();

    let profile = match cli
        .profile
        .as_ref()
        .map(|s| s.as_str())
        .or(last_profile)
        .or(if last_is_preset {
            None
        } else {
            config.default_profile.as_deref()
        }) {
        Some(name) => Some(
            config
                .profiles
                .get(name)
                .cloned()
                .ok_or_else(|| anyhow!("profile '{name}' not found in config"))?,
        ),
        None => None,
    };
    let profile = profile.as_ref();

    let provider_explicit =
        cli.provider.is_some() || profile.is_some_and(|p| p.provider.is_some()) || last_is_preset;
    let last_provider = last
        .as_ref()
        .and_then(|s| s.provider.as_deref())
        .and_then(|s| s.parse::<ProviderName>().ok());
    let mut provider = if last_is_preset {
        // Preset path: last provider wins over any residual profile.
        cli.provider
            .or(last_provider)
            .or(profile.and_then(|p| p.provider))
            .unwrap_or(ProviderName::Openai)
    } else {
        cli.provider
            .or(profile.and_then(|p| p.provider))
            .or(last_provider)
            .unwrap_or(ProviderName::Openai)
    };

    // Last-session model beats the profile's stored model so mid-session
    // `/model` picks (also written into the profile when possible) win on
    // restart even if a concurrent edit raced the profile file.
    let mut model = cli
        .model
        .clone()
        .or_else(|| last.as_ref().and_then(|s| s.model.clone()))
        .or_else(|| profile.and_then(|p| p.model.clone()))
        .or_else(|| std::env::var("HI_MODEL").ok())
        .or_else(|| provider.default_model().map(String::from));

    // Bare run with nothing configured: infer a provider+model from the
    // environment so `hi` "just works" when a key is present.
    if model.is_none()
        && !provider_explicit
        && let Some((auto_provider, auto_model)) = auto_select()
    {
        provider = auto_provider;
        model = Some(auto_model);
    }
    let model = model.ok_or_else(|| anyhow!("{ONBOARDING}"))?;

    let base_url = cli
        .base_url
        .clone()
        .or_else(|| profile.and_then(|p| p.base_url.clone()))
        .or_else(|| std::env::var("HI_BASE_URL").ok())
        .unwrap_or_else(|| provider.default_base_url().to_string());

    let mcp_url = cli
        .mcp_url
        .clone()
        .or_else(|| profile.and_then(|p| p.mcp_url.clone()))
        .or_else(|| std::env::var("HI_MCP_URL").ok())
        .or_else(|| provider.default_mcp_url().map(String::from));

    let api_key = resolve_api_key(cli, profile, provider)?;

    let profile_max_tokens = profile.and_then(|p| p.max_tokens);
    let max_tokens = configured_max_tokens(provider, cli.max_tokens, profile_max_tokens);
    let max_tokens_explicit = max_tokens_is_explicit(provider, cli.max_tokens, profile_max_tokens);

    let thinking_budget = cli.thinking.or(profile.and_then(|p| p.thinking_budget));
    let reasoning_effort = cli
        .reasoning_effort
        .map(ReasoningEffort::from)
        .or_else(|| profile.and_then(|p| p.reasoning_effort));
    let tool_mode = cli
        .tool_mode
        .map(ToolMode::from)
        .or_else(|| profile.and_then(|p| p.tool_mode))
        .unwrap_or_default();
    let compat = cli
        .compat
        .map(CompatMode::from)
        .or_else(|| profile.and_then(|p| p.compat))
        .unwrap_or_default();
    let curate_skills = curate_skills_default(provider, profile.and_then(|p| p.curate_skills));
    let explore_subagents = explore_subagents_default(profile.and_then(|p| p.explore_subagents));
    let write_subagents = write_subagents_default(profile.and_then(|p| p.write_subagents));
    let planner_model =
        planner_model_default(provider, profile.and_then(|p| p.planner_model.clone()));
    // Skeptic model: opt-in, no provider default (unlike the planner) — off unless
    // a profile or HI_SKEPTIC_MODEL sets it.
    let skeptic_model = profile.and_then(|p| p.skeptic_model.clone());

    Ok(Settings {
        provider,
        model,
        base_url,
        mcp_url,
        api_key,
        max_tokens,
        max_tokens_explicit,
        thinking_budget,
        reasoning_effort,
        tool_mode,
        compat,
        curate_skills,
        explore_subagents,
        write_subagents,
        planner_model,
        skeptic_model,
        moa: config.moa.clone(),
        api_unix_socket: cli.api_unix_socket.clone(),
    })
}

/// The sorted list of configured profile names, for `/provider` (no arg).
pub fn profile_names(config: &Config) -> Vec<String> {
    let mut names: Vec<String> = config.profiles.keys().cloned().collect();
    names.sort();
    names
}

/// The fallback chain (excluding the primary) — `--fallback` flags first, then
/// the selected profile's `fallback` list, deduped. Profiles that don't resolve
/// (missing key/model) are skipped with a warning rather than blocking startup.
pub fn resolve_fallbacks(cli: &Cli, config: &Config) -> Vec<Settings> {
    let primary_name = cli.profile.as_ref().or(config.default_profile.as_ref());

    let mut names: Vec<String> = cli.fallback.clone();
    if let Some(name) = primary_name
        && let Some(profile) = config.profiles.get(name)
        && let Some(list) = &profile.fallback
    {
        names.extend(list.iter().cloned());
    }

    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    if let Some(name) = primary_name {
        seen.insert(name.clone()); // don't fall back to the primary itself
    }

    let mut out = Vec::new();
    for name in names {
        if !seen.insert(name.clone()) {
            continue;
        }
        match resolve_named_profile(config, &name) {
            Ok(settings) => out.push(settings),
            Err(err) => {
                eprintln!("\x1b[33mwarning: skipping fallback profile '{name}': {err}\x1b[0m")
            }
        }
    }
    out
}

/// Resolve a named profile into [`Settings`] from its own fields + environment
/// (no CLI overrides — those belong to the primary). Used both for fallback
/// profiles at startup and for `/provider` changes mid-session.
///
/// If the profile has no `model` and the provider has no default, a placeholder
/// is used. The placeholder is fine for building the provider and listing
/// models, but a turn can't run with it.
pub fn resolve_named_profile(config: &Config, name: &str) -> Result<Settings> {
    config.moa.validate()?;
    // A bare provider name is accepted when no profile has that name, so
    // `/provider xai` works straight after `/login xai` without first creating
    // a profile. Profiles win on a name clash — they are explicit configuration.
    let profile = config.profiles.get(name);
    let provider = match profile {
        Some(profile) => profile.provider.unwrap_or(ProviderName::Openai),
        None => name.parse::<ProviderName>().map_err(|_| {
            let mut known: Vec<&str> = config.profiles.keys().map(String::as_str).collect();
            known.sort_unstable();
            let profiles = if known.is_empty() {
                "none configured".to_string()
            } else {
                known.join(", ")
            };
            anyhow!(
                "no profile or provider named '{name}'.\n\
                 Profiles: {profiles}\n\
                 Providers: openai, anthropic, pipenetwork, ollama, xai"
            )
        })?,
    };

    let model = profile
        .and_then(|p| p.model.clone())
        .or_else(|| provider.default_model().map(String::from))
        .unwrap_or_else(|| "__model_not_configured__".to_string());
    let base_url = profile
        .and_then(|p| p.base_url.clone())
        .unwrap_or_else(|| provider.default_base_url().to_string());
    let mcp_url = profile
        .and_then(|p| p.mcp_url.clone())
        .or_else(|| std::env::var("HI_MCP_URL").ok())
        .or_else(|| provider.default_mcp_url().map(String::from));
    let api_key = resolve_api_key_for(profile, provider)?;

    let profile_max_tokens = profile.and_then(|p| p.max_tokens);
    let max_tokens = configured_max_tokens(provider, None, profile_max_tokens);
    let max_tokens_explicit = max_tokens_is_explicit(provider, None, profile_max_tokens);

    Ok(Settings {
        provider,
        model,
        base_url,
        mcp_url,
        api_key,
        max_tokens,
        max_tokens_explicit,
        thinking_budget: profile.and_then(|p| p.thinking_budget),
        reasoning_effort: profile.and_then(|p| p.reasoning_effort),
        tool_mode: profile.and_then(|p| p.tool_mode).unwrap_or_default(),
        compat: profile.and_then(|p| p.compat).unwrap_or_default(),
        curate_skills: curate_skills_default(provider, profile.and_then(|p| p.curate_skills)),
        explore_subagents: explore_subagents_default(profile.and_then(|p| p.explore_subagents)),
        write_subagents: write_subagents_default(profile.and_then(|p| p.write_subagents)),
        planner_model: planner_model_default(
            provider,
            profile.and_then(|p| p.planner_model.clone()),
        ),
        skeptic_model: profile.and_then(|p| p.skeptic_model.clone()),
        moa: config.moa.clone(),
        api_unix_socket: None,
    })
}

pub(crate) fn configured_max_tokens(
    provider: ProviderName,
    cli_max_tokens: Option<u32>,
    profile_max_tokens: Option<u32>,
) -> u32 {
    if let Some(value) = cli_max_tokens {
        return value;
    }
    match (provider, profile_max_tokens) {
        // Pipenetwork profiles may carry old wizard defaults. Treat those as
        // implicit so live API limits can size coding-agent turns at runtime;
        // an explicit CLI --max-tokens still wins above.
        (
            ProviderName::Pipenetwork,
            None | Some(DEFAULT_MAX_TOKENS) | Some(LEGACY_PIPENETWORK_DEFAULT_MAX_TOKENS),
        ) => PIPENETWORK_DEFAULT_MAX_TOKENS,
        (_, Some(value)) => value,
        (_, None) => DEFAULT_MAX_TOKENS,
    }
}

/// Whether verifier-gated skill auto-curation is on. An explicit `curate_skills`
/// in the profile always wins; otherwise it defaults on for the pipenetwork
/// provider (its coding-agent models are strong enough for the curator to pay
/// off) and off for every other provider.
pub(crate) fn curate_skills_default(provider: ProviderName, profile_value: Option<bool>) -> bool {
    profile_value.unwrap_or(provider == ProviderName::Pipenetwork)
}

/// Whether the read-only `explore` subagent tool is advertised. On by default for
/// every provider (the tool is read-only, depth-capped at 1, and per-session
/// budgeted, so it's safe to offer broadly); a profile can set `explore_subagents
/// = false` to turn it off (e.g. for a very small local model that would misuse it).
pub(crate) fn explore_subagents_default(profile_value: Option<bool>) -> bool {
    profile_value.unwrap_or(true)
}

/// Write-capable `delegate` policy. Profile `write_subagents = true` → On;
/// `false` → Off; unset → Risk (multi-file / isolation-shaped mutations only).
pub(crate) fn write_subagents_default(
    profile_value: Option<bool>,
) -> hi_agent::WriteSubagentPolicy {
    match profile_value {
        Some(true) => hi_agent::WriteSubagentPolicy::On,
        Some(false) => hi_agent::WriteSubagentPolicy::Off,
        None => hi_agent::WriteSubagentPolicy::Risk,
    }
}

/// The `/goal` planner model. An explicit `planner_model` in the profile always
/// wins; otherwise it defaults to glm-5.2 on pipenetwork (a strong planner served
/// there) and `None` (no decomposition — a single sub-goal) for every other
/// provider, since the id wouldn't route on their endpoint.
pub(crate) fn planner_model_default(
    provider: ProviderName,
    profile_value: Option<String>,
) -> Option<String> {
    profile_value.or_else(|| {
        (provider == ProviderName::Pipenetwork).then(|| "pipe/glm-5.2-fast".to_string())
    })
}

pub(crate) fn max_tokens_is_explicit(
    provider: ProviderName,
    cli_max_tokens: Option<u32>,
    profile_max_tokens: Option<u32>,
) -> bool {
    if cli_max_tokens.is_some() {
        return true;
    }
    match (provider, profile_max_tokens) {
        (
            ProviderName::Pipenetwork,
            None | Some(DEFAULT_MAX_TOKENS) | Some(LEGACY_PIPENETWORK_DEFAULT_MAX_TOKENS),
        ) => false,
        (_, Some(_)) => true,
        (_, None) => false,
    }
}
