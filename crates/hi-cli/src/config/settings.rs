use super::*;

/// Fully-resolved settings used to build a provider and run the agent.
#[derive(Debug)]
pub struct Settings {
    pub execution: hi_agent::ExecutionMode,
    pub provider: ProviderName,
    pub model: String,
    pub base_url: String,
    pub mcp_url: Option<String>,
    pub api_key: String,
    pub max_tokens: u32,
    pub max_tokens_explicit: bool,
    pub top_p: Option<f32>,
    pub output_token_parameter: hi_ai::OutputTokenParameter,
    pub thinking_budget: Option<u32>,
    pub reasoning_effort: Option<ReasoningEffort>,
    pub tool_mode: ToolMode,
    pub compat: CompatMode,
    pub deepseek_compat: hi_ai::DeepSeekCompat,
    pub curate_skills: bool,
    pub explore_subagents: bool,
    /// Claude-style suggested next prompt after turns (ghost text in TUI).
    pub suggest_next_prompt: bool,
    /// Off / Risk (default) / On — see [`hi_agent::WriteSubagentPolicy`].
    pub write_subagents: hi_agent::WriteSubagentPolicy,
    pub planner_model: Option<String>,
    pub skeptic_model: Option<String>,
    pub moa: hi_ai::MoaConfig,
    pub api_unix_socket: Option<PathBuf>,
    /// Optional instructions for recreating a hi-managed local runtime.
    pub runtime: Option<LocalRuntimeProfile>,
    pub x402: X402Settings,
    /// Inject-gated `browser_exec` (default on for page/login/UI-shaped tasks).
    pub browser_enabled: bool,
    pub browser_allow_private: bool,
    /// Auto-attach first-party Pipe `/mcp` as server `pipe`.
    pub mcp_pipe_enabled: bool,
    pub mcp_pipe_allow: Vec<String>,
}

#[derive(Clone, Debug)]
pub struct X402Settings {
    pub keypair: Option<PathBuf>,
    pub rpc: String,
    pub max_usd: f64,
    pub auto_confirm: bool,
    /// `/login x402` without a keypair: paste a signature when quoted.
    pub paste_sig: bool,
}

impl Default for X402Settings {
    fn default() -> Self {
        Self {
            keypair: None,
            rpc: "https://api.mainnet-beta.solana.com".to_string(),
            max_usd: hi_ai::X402_DEFAULT_MAX_USD,
            auto_confirm: false,
            paste_sig: false,
        }
    }
}

impl X402Settings {
    pub fn enabled(&self) -> bool {
        self.keypair.is_some() || self.paste_sig || hi_ai::has_credit_token()
    }
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
        .as_deref()
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
    // `--provider` selects an entire remote route. If it disagrees with the
    // selected/default profile, do not carry that profile's model, endpoints,
    // or credential onto the explicitly selected provider. A profile with no
    // provider is an OpenAI profile, matching the resolver's existing default.
    let route_profile = profile_for_route(profile, cli.provider);

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

    // A matching last-session model beats the profile's stored model so
    // mid-session `/model` picks (also written into the profile when possible)
    // win on restart even if a concurrent edit raced the profile file. Never
    // carry a model across an explicit provider mismatch: a profile may have
    // changed routes since the prior exit, and model ids are provider-scoped.
    //
    // A CLI `--provider` that isn't this profile's provider must not inherit
    // the profile's model: grok-4.3 would otherwise ride onto pipenetwork and
    // ignore `HI_MODEL` / the provider default.
    let profile_model = route_profile.and_then(|p| p.model.clone());
    let last_model = last
        .as_ref()
        .and_then(|session| match session.provider.as_deref() {
            None => session.model.clone(),
            Some(label)
                if label
                    .parse::<ProviderName>()
                    .is_ok_and(|remembered| remembered == provider) =>
            {
                session.model.clone()
            }
            Some(_) => None,
        });
    let mut model = cli
        .model
        .clone()
        .or(last_model)
        .or(profile_model)
        .or_else(|| std::env::var("HI_MODEL").ok())
        .or_else(|| provider.default_model().map(String::from));

    // Bare run with nothing configured: infer a provider+model from the
    // environment so `hi` "just works" when a key is present.
    if model.is_none()
        && !provider_explicit
        && let Some((auto_provider, auto_model)) = auto_select(config)
    {
        provider = auto_provider;
        model = Some(auto_model);
    }
    let mut model = model.ok_or_else(|| anyhow!("{ONBOARDING}"))?;
    hi_provider_config::apply_stale_pipenetwork_default_model(provider.into(), &mut model);

    let base_url = cli
        .base_url
        .clone()
        .or_else(|| route_profile.and_then(|p| p.base_url.clone()))
        .or_else(|| std::env::var("HI_BASE_URL").ok())
        .unwrap_or_else(|| provider.default_base_url().to_string());

    let mcp_url = resolve_mcp_url(cli, route_profile, provider)?;

    let api_key =
        resolve_api_key_with_endpoint(cli, route_profile, provider, &base_url).or_else(|err| {
            // Credit tokens and wallet authorization are ambient Pipe
            // credentials too; never forward them to a profile-controlled
            // custom origin.
            if is_official_provider_url(provider, &base_url) {
                pipenetwork_x402_key(config, provider).ok_or(err)
            } else {
                Err(err)
            }
        })?;

    let profile_max_tokens = route_profile.and_then(|p| p.max_tokens);
    let max_tokens = configured_max_tokens(provider, cli.max_tokens, profile_max_tokens);
    let max_tokens_explicit = max_tokens_is_explicit(provider, cli.max_tokens, profile_max_tokens);

    let top_p = cli.top_p.or_else(|| route_profile.and_then(|p| p.top_p));
    if let Some(top_p) = top_p
        && !(0.0..=1.0).contains(&top_p)
    {
        anyhow::bail!("top_p must be between 0.0 and 1.0");
    }
    let output_token_parameter = cli
        .output_token_parameter
        .map(hi_ai::OutputTokenParameter::from)
        .or_else(|| route_profile.and_then(|p| p.output_token_parameter))
        .unwrap_or_default();

    let thinking_budget = cli
        .thinking
        .or(route_profile.and_then(|p| p.thinking_budget));
    let reasoning_effort = resolve_reasoning_effort(
        cli.reasoning_effort.map(ReasoningEffort::from),
        route_profile,
        provider,
        config.reasoning_effort,
    );
    let tool_mode = cli
        .tool_mode
        .map(ToolMode::from)
        .or_else(|| profile.and_then(|p| p.tool_mode))
        .unwrap_or_default();
    let compat = cli
        .compat
        .map(CompatMode::from)
        .or_else(|| route_profile.and_then(|p| p.compat))
        .unwrap_or_default();
    let deepseek_compat = cli
        .deepseek_compat
        .map(DeepSeekCompat::from)
        .or_else(|| route_profile.and_then(|p| p.deepseek_compat))
        .unwrap_or_default();
    let curate_skills = curate_skills_default(provider, profile.and_then(|p| p.curate_skills));
    let explore_subagents = explore_subagents_default(profile.and_then(|p| p.explore_subagents));
    let suggest_next_prompt =
        suggest_next_prompt_default(profile.and_then(|p| p.suggest_next_prompt));
    let write_subagents = write_subagents_default(profile.and_then(|p| p.write_subagents));
    let planner_model = planner_model_default(
        provider,
        route_profile.and_then(|p| p.planner_model.clone()),
    );
    // Skeptic model: opt-in, no provider default (unlike the planner) — off unless
    // a profile or HI_SKEPTIC_MODEL sets it.
    let skeptic_model = route_profile.and_then(|p| p.skeptic_model.clone());
    let execution = resolve_execution_mode(
        cli,
        profile.and_then(|profile| profile.execution),
        config.execution,
    )?;

    Ok(Settings {
        execution,
        provider,
        model,
        base_url,
        mcp_url,
        api_key,
        max_tokens,
        max_tokens_explicit,
        top_p,
        output_token_parameter,
        thinking_budget,
        reasoning_effort,
        tool_mode,
        compat,
        deepseek_compat,
        curate_skills,
        explore_subagents,
        suggest_next_prompt,
        write_subagents,
        planner_model,
        skeptic_model,
        moa: config.moa.clone(),
        api_unix_socket: cli.api_unix_socket.clone(),
        runtime: route_profile.and_then(|p| p.runtime.clone()),
        x402: resolve_x402_settings(cli.yes, config),
        browser_enabled: config.browser.is_enabled(),
        browser_allow_private: config.browser.allows_private_urls(),
        mcp_pipe_enabled: config.mcp.pipe.is_enabled(),
        mcp_pipe_allow: config.mcp.pipe.allow.clone(),
    })
}

/// Resolve execution persistence without conflating an absent environment
/// override with an explicit `ephemeral`. Saved ordinary sessions default to
/// boundary checkpointing; no-save and measured harness paths retain their
/// end-of-turn-only behavior unless the user explicitly selects a mode.
pub(crate) fn resolve_execution_mode(
    cli: &Cli,
    profile_mode: Option<hi_agent::ExecutionMode>,
    global_mode: Option<hi_agent::ExecutionMode>,
) -> Result<hi_agent::ExecutionMode> {
    if cli.durable {
        return Ok(hi_agent::ExecutionMode::Durable);
    }
    if let Some(mode) = profile_mode.or(global_mode).or(resolve_execution_env()?) {
        return Ok(mode);
    }
    Ok(default_execution_for_cli(cli))
}

fn default_execution_for_cli(cli: &Cli) -> hi_agent::ExecutionMode {
    if cli.no_save || cli.subagent || cli.eval_input.is_some() || cli.report.is_some() {
        hi_agent::ExecutionMode::Ephemeral
    } else {
        hi_agent::ExecutionMode::Durable
    }
}

fn resolve_execution_env() -> Result<Option<hi_agent::ExecutionMode>> {
    match std::env::var("HI_EXECUTION_MODE") {
        Ok(value) => match value.trim().to_ascii_lowercase().as_str() {
            "durable" | "on" | "true" | "1" => Ok(Some(hi_agent::ExecutionMode::Durable)),
            "ephemeral" | "off" | "false" | "0" | "" => {
                Ok(Some(hi_agent::ExecutionMode::Ephemeral))
            }
            other => {
                anyhow::bail!("HI_EXECUTION_MODE must be durable or ephemeral (got {other:?})")
            }
        },
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(error) => anyhow::bail!("reading HI_EXECUTION_MODE: {error}"),
    }
}

/// The sorted list of configured profile names, for `/provider` (no arg).
pub fn profile_names(config: &Config) -> Vec<String> {
    let mut names: Vec<String> = config.profiles.keys().cloned().collect();
    names.sort();
    names
}

/// Keep a selected profile's request-route settings unless an explicit
/// provider selects a different route. Agent behavior and persistence policy
/// continue to read from the selected profile separately.
fn profile_for_route(
    profile: Option<&Profile>,
    explicit_provider: Option<ProviderName>,
) -> Option<&Profile> {
    match explicit_provider {
        Some(explicit_provider)
            if profile.is_some_and(|profile| {
                profile.provider.unwrap_or(ProviderName::Openai) != explicit_provider
            }) =>
        {
            None
        }
        _ => profile,
    }
}

/// The fallback chain (excluding the primary) — `--fallback` flags first, then
/// the selected profile's `fallback` list, deduped. Profiles that don't resolve
/// (missing key/model) are skipped with a warning rather than blocking startup.
pub fn resolve_fallbacks(cli: &Cli, config: &Config) -> Vec<Settings> {
    let primary_name = cli.profile.as_ref().or(config.default_profile.as_ref());
    let primary_route_profile = profile_for_route(
        primary_name.and_then(|name| config.profiles.get(name)),
        cli.provider,
    );

    let mut names: Vec<String> = cli.fallback.clone();
    if let Some(list) = primary_route_profile.and_then(|profile| profile.fallback.as_ref()) {
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
    // Aliases (`pipe` → pipenetwork) are normalized via `ProviderName::from_str`.
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

    // Bare `/provider pipenetwork` (or `pipe`) after `/provider xai` must still
    // see a key stored on e.g. `[profiles.default]` with `provider = "pipenetwork"`.
    // Without this, the preset path only checks auth.json + env and fails even
    // though startup via `default_profile` works fine with the same config.
    let credential_profile = profile.or_else(|| {
        preferred_profile_for_provider(config, provider)
            .and_then(|name| config.profiles.get(name))
            // A key paired with a custom profile endpoint must not be borrowed
            // for the bare provider preset's official endpoint.
            .filter(|profile| {
                profile
                    .base_url
                    .as_deref()
                    .is_none_or(|url| is_official_provider_url(provider, url))
            })
    });

    let mut model = profile
        .and_then(|p| p.model.clone())
        .or_else(|| provider.default_model().map(String::from))
        .unwrap_or_else(|| "__model_not_configured__".to_string());
    hi_provider_config::apply_stale_pipenetwork_default_model(provider.into(), &mut model);
    let base_url = profile
        .and_then(|p| p.base_url.clone())
        .unwrap_or_else(|| provider.default_base_url().to_string());
    let mcp_url = resolve_named_mcp_url(profile, provider)?;
    // Prefer the matching profile's key. If a borrowed profile only references
    // an unset env var, fall through to auth.json / provider env so `/login`
    // still works after `/provider <preset>`.
    let profile_controls_endpoint = profile.is_some_and(|profile| profile.project_local);
    let api_key = resolve_api_key_for_endpoint(
        credential_profile,
        provider,
        &base_url,
        false,
        profile_controls_endpoint,
    )
    .or_else(|err| {
        if profile.is_none() && credential_profile.is_some() {
            resolve_api_key_for_endpoint(None, provider, &base_url, false, false).map_err(|_| err)
        } else {
            Err(err)
        }
    })
    .or_else(|err| {
        if is_official_provider_url(provider, &base_url) {
            pipenetwork_x402_key(config, provider).ok_or(err)
        } else {
            Err(err)
        }
    })?;

    let profile_max_tokens = profile.and_then(|p| p.max_tokens);
    let max_tokens = configured_max_tokens(provider, None, profile_max_tokens);
    let max_tokens_explicit = max_tokens_is_explicit(provider, None, profile_max_tokens);
    let execution = profile
        .and_then(|profile| profile.execution)
        .or(config.execution)
        .or(resolve_execution_env()?)
        .unwrap_or(hi_agent::ExecutionMode::Durable);

    Ok(Settings {
        execution,
        provider,
        model,
        base_url,
        mcp_url,
        api_key,
        max_tokens,
        max_tokens_explicit,
        top_p: profile.and_then(|p| p.top_p),
        output_token_parameter: profile
            .and_then(|p| p.output_token_parameter)
            .unwrap_or_default(),
        thinking_budget: profile.and_then(|p| p.thinking_budget),
        reasoning_effort: resolve_reasoning_effort(
            None,
            profile,
            provider,
            config.reasoning_effort,
        ),
        tool_mode: profile.and_then(|p| p.tool_mode).unwrap_or_default(),
        compat: profile.and_then(|p| p.compat).unwrap_or_default(),
        deepseek_compat: profile.and_then(|p| p.deepseek_compat).unwrap_or_default(),
        curate_skills: curate_skills_default(provider, profile.and_then(|p| p.curate_skills)),
        explore_subagents: explore_subagents_default(profile.and_then(|p| p.explore_subagents)),
        suggest_next_prompt: suggest_next_prompt_default(
            profile.and_then(|p| p.suggest_next_prompt),
        ),
        write_subagents: write_subagents_default(profile.and_then(|p| p.write_subagents)),
        planner_model: planner_model_default(
            provider,
            profile.and_then(|p| p.planner_model.clone()),
        ),
        skeptic_model: profile.and_then(|p| p.skeptic_model.clone()),
        moa: config.moa.clone(),
        api_unix_socket: None,
        runtime: profile.and_then(|p| p.runtime.clone()),
        x402: resolve_x402_settings(false, config),
        browser_enabled: config.browser.is_enabled(),
        browser_allow_private: config.browser.allows_private_urls(),
        mcp_pipe_enabled: config.mcp.pipe.is_enabled(),
        mcp_pipe_allow: config.mcp.pipe.allow.clone(),
    })
}

/// Reasoning effort for this route.
///
/// CLI always wins. A profile's `reasoning_effort` applies only when that
/// profile's provider is the active one — an xAI default profile's `xhigh`
/// must not follow `--provider pipenetwork` onto DeepSeek wrap-up.
/// Machine-wide `/config reasoning` still applies, except `xhigh`, which is
/// an xAI-only wire value and is dropped on every other provider.
pub(crate) fn resolve_reasoning_effort(
    cli: Option<ReasoningEffort>,
    profile: Option<&Profile>,
    provider: ProviderName,
    machine: Option<ReasoningEffort>,
) -> Option<ReasoningEffort> {
    if let Some(effort) = cli {
        return Some(effort);
    }
    if let Some(profile) = profile {
        let profile_provider = profile.provider.unwrap_or(ProviderName::Openai);
        if profile_provider == provider
            && let Some(effort) = profile.reasoning_effort
        {
            return Some(effort);
        }
    }
    match machine {
        Some(ReasoningEffort::Xhigh) if provider != ProviderName::Xai => None,
        other => other,
    }
}

/// Profile to borrow credentials from when resolving a bare provider preset.
///
/// Prefer `default_profile` when it targets `provider`, otherwise the first
/// (sorted) profile that does. Returns the profile *name*, not the profile
/// itself, so callers can look it up once.
fn preferred_profile_for_provider(config: &Config, provider: ProviderName) -> Option<&str> {
    let targets = |p: &Profile| p.provider.unwrap_or(ProviderName::Openai) == provider;
    if let Some(name) = config.default_profile.as_deref()
        && let Some(profile) = config.profiles.get(name)
        && targets(profile)
    {
        return Some(name);
    }
    let mut names: Vec<&str> = config
        .profiles
        .iter()
        .filter(|(_, profile)| targets(profile))
        .map(|(name, _)| name.as_str())
        .collect();
    names.sort_unstable();
    names.first().copied()
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

/// Whether Claude-style suggested next prompts are generated after turns.
/// Profile wins when set; otherwise `HI_SUGGEST_NEXT_PROMPT` can force on/off;
/// default is on (matches Claude Code).
pub(crate) fn suggest_next_prompt_default(profile_value: Option<bool>) -> bool {
    if let Some(value) = profile_value {
        return value;
    }
    match std::env::var("HI_SUGGEST_NEXT_PROMPT") {
        Ok(value) => match value.trim().to_ascii_lowercase().as_str() {
            "0" | "false" | "off" | "no" | "disable" | "disabled" => false,
            "1" | "true" | "on" | "yes" | "enable" | "enabled" => true,
            _ => true,
        },
        Err(_) => true,
    }
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

pub(crate) fn resolve_x402_settings(cli_yes: bool, config: &Config) -> X402Settings {
    let section = config.x402.as_ref();
    let keypair = std::env::var("HI_X402_KEYPAIR")
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
        .or_else(|| section.and_then(|x402| x402.keypair.clone()));
    let rpc = std::env::var("HI_SOLANA_RPC")
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .or_else(|| section.and_then(|x402| x402.rpc.clone()))
        .unwrap_or_else(|| "https://api.mainnet-beta.solana.com".to_string());
    let max_usd = std::env::var("HI_X402_MAX_USD")
        .ok()
        .and_then(|value| value.trim().parse().ok())
        .or_else(|| section.and_then(|x402| x402.max_usd))
        .filter(|value| value.is_finite() && *value > 0.0)
        .unwrap_or(hi_ai::X402_DEFAULT_MAX_USD);
    let auto_confirm = cli_yes
        || env_flag("HI_X402_AUTO_CONFIRM")
        || section.and_then(|x402| x402.auto_confirm).unwrap_or(false);
    let paste_sig = keypair.is_none()
        && (section.and_then(|x402| x402.enabled) == Some(true) || env_flag("HI_X402_PASTE_SIG"));
    X402Settings {
        keypair,
        rpc,
        max_usd,
        auto_confirm,
        paste_sig,
    }
}

fn env_flag(name: &str) -> bool {
    std::env::var(name).is_ok_and(|value| {
        matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "on" | "yes"
        )
    })
}

/// Pairing keys and `PIPENETWORK_API_KEY` already won in `resolve_api_key_for`.
/// Fall back to a stored x402 credit token, or an empty bearer when a keypair /
/// paste-sig flow is configured.
fn pipenetwork_x402_key(config: &Config, provider: ProviderName) -> Option<String> {
    if provider != ProviderName::Pipenetwork {
        return None;
    }
    if let Some(token) = hi_ai::load_credit_token() {
        return Some(token);
    }
    let settings = resolve_x402_settings(false, config);
    if settings.keypair.is_some() || settings.paste_sig {
        Some(String::new())
    } else {
        None
    }
}

#[cfg(test)]
mod execution_default_tests {
    use super::*;

    #[test]
    fn saved_ordinary_sessions_default_durable() {
        let cli = Cli::try_parse_from(["hi"]).unwrap();
        assert_eq!(
            default_execution_for_cli(&cli),
            hi_agent::ExecutionMode::Durable
        );
    }

    #[test]
    fn unsaved_and_measured_sessions_default_ephemeral() {
        for args in [
            vec!["hi", "--no-save"],
            vec!["hi", "--subagent"],
            vec!["hi", "--eval-input", "case.json"],
            vec!["hi", "--report", "report.json"],
        ] {
            let cli = Cli::try_parse_from(args).unwrap();
            assert_eq!(
                default_execution_for_cli(&cli),
                hi_agent::ExecutionMode::Ephemeral
            );
        }
    }

    #[test]
    fn explicit_ephemeral_mode_beats_the_saved_default() {
        let cli = Cli::try_parse_from(["hi"]).unwrap();
        assert_eq!(
            resolve_execution_mode(
                &cli,
                Some(hi_agent::ExecutionMode::Ephemeral),
                Some(hi_agent::ExecutionMode::Durable),
            )
            .unwrap(),
            hi_agent::ExecutionMode::Ephemeral
        );
    }
}
