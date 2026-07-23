use super::*;

#[derive(Clone, Debug, Default, Deserialize)]
pub struct Config {
    pub default_profile: Option<String>,
    #[serde(default)]
    pub moa: hi_ai::MoaConfig,
    #[serde(default)]
    pub profiles: HashMap<String, Profile>,
    #[serde(default)]
    pub sync: Option<SyncSection>,
    #[serde(default)]
    pub rsi: Option<RsiSection>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct RsiSection {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub enabled: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base_url: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub maximum_cost_microusd: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub channel: Option<String>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RsiRequested {
    Off,
    Remote,
    Managed,
}

pub fn resolve_rsi(cli: &Cli, file: &Config) -> anyhow::Result<RsiRequested> {
    if cli.rsi_managed {
        anyhow::ensure!(!cli.no_rsi, "managed RSI cannot be disabled");
        anyhow::ensure!(
            cli.rsi_trace_dir.is_some()
                && cli.rsi_max_bytes.is_some()
                && cli.rsi_runtime_descriptor.is_some(),
            "managed RSI requires its trace and runtime descriptor"
        );
        return Ok(RsiRequested::Managed);
    }
    if cli.rsi {
        return Ok(RsiRequested::Remote);
    }
    if cli.no_rsi {
        return Ok(RsiRequested::Off);
    }
    if let Some(enabled) = file.rsi.as_ref().and_then(|rsi| rsi.enabled) {
        return Ok(if enabled {
            RsiRequested::Remote
        } else {
            RsiRequested::Off
        });
    }
    let environment = std::env::var("HI_RSI_ENABLED").ok();
    match environment
        .as_deref()
        .map(str::trim)
        .map(str::to_ascii_lowercase)
        .as_deref()
    {
        None | Some("") | Some("0" | "false" | "off" | "no") => Ok(RsiRequested::Off),
        Some("1" | "true" | "on" | "yes") => Ok(RsiRequested::Remote),
        Some(_) => anyhow::bail!("HI_RSI_ENABLED must be true or false"),
    }
}

/// The `[sync]` section in `hi.toml` — configures cross-machine session sync.
/// All fields optional; unset fields fall back to env vars or the provider's
/// credentials.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct SyncSection {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub base_url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_key: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_key_env: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub machine_id: Option<String>,
    /// Persisted sync policy. Missing values migrate from legacy `enabled`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mode: Option<crate::sync_store::SyncMode>,
    /// When true, sync is enabled by default (no need for `--sync` on the CLI).
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub enabled: bool,
}

impl serde::Serialize for Config {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut s = serializer.serialize_struct("Config", 5)?;
        if let Some(v) = &self.default_profile {
            s.serialize_field("default_profile", v)?;
        }
        if self.moa != hi_ai::MoaConfig::default() {
            s.serialize_field("moa", &self.moa)?;
        }
        if !self.profiles.is_empty() {
            // BTreeMap serializes as a sorted map → stable, alphabetical output.
            let sorted: BTreeMap<&String, &Profile> = self.profiles.iter().collect();
            s.serialize_field("profiles", &sorted)?;
        }
        if let Some(sync) = &self.sync {
            s.serialize_field("sync", sync)?;
        }
        if let Some(rsi) = &self.rsi {
            s.serialize_field("rsi", rsi)?;
        }
        s.end()
    }
}

// Serialized with `skip_serializing_if` on every field so a saved config omits
// unset keys instead of filling with `model = ""` lines. Keep the attribute on
// each new field: a field missing it is fine, but a field missing from
// serialization entirely (as with the old hand-written `Serialize` impl) is
// silently deleted from the user's config file on every save.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Profile {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider: Option<ProviderName>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub base_url: Option<String>,
    /// MCP endpoint used for metadata discovery, when supported by the provider.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mcp_url: Option<String>,
    /// A literal API key (written by the setup wizard).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_key: Option<String>,
    /// Name of an env var holding the API key for this profile.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_key_env: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking_budget: Option<u32>,
    /// Reasoning effort (`reasoning_effort`) for OpenAI-compatible endpoints
    /// that support it. TOML values: minimal/low/medium/high/xhigh.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<ReasoningEffort>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_mode: Option<ToolMode>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub compat: Option<CompatMode>,
    /// Verifier-gated skill auto-curation: after a verified turn, distill a
    /// reusable technique into a learned skill. Defaults to off.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub curate_skills: Option<bool>,
    /// Advertise the read-only `explore` subagent tool. On by default; set to
    /// false to disable (e.g. for a very small local model).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub explore_subagents: Option<bool>,
    /// Advertise the write-capable `delegate` subagent tool. Off by default (the
    /// riskier tier); set to true to enable.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub write_subagents: Option<bool>,
    /// Model id that decomposes a `/goal <objective>` into sub-goals. Defaults to
    /// `pipe/glm-5.2-fast` on the pipenetwork profile; `None` disables planning.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub planner_model: Option<String>,
    /// Model id for the `/goal team` skeptic gate (reviews a turn before it
    /// advances a sub-goal). `None` (default) disables the gate.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub skeptic_model: Option<String>,
    /// Other profile names to fall back to, in order, when this one returns
    /// nothing or errors.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fallback: Option<Vec<String>>,
}

pub fn load_config(explicit: Option<&Path>) -> Result<Config> {
    if let Some(path) = explicit {
        return read_config(path);
    }

    let mut config = default_config_path()
        .filter(|path| path.exists())
        .map(|path| read_config(&path))
        .transpose()?
        .unwrap_or_default();

    let local_path = local_config_path();
    if local_path.exists() {
        let local = read_config(&local_path)?;
        merge_config(&mut config, local);
    }

    config.moa.validate()?;
    Ok(config)
}

pub(crate) fn read_config(path: &Path) -> Result<Config> {
    let mut config = read_config_file(path)?;
    config
        .moa
        .validate()
        .with_context(|| format!("validating MoA config {}", path.display()))?;
    migrate_api_key_env_to_literal(&mut config, path);
    Ok(config)
}

/// Parse a single config file as-is: no validation, no key migration. Used by
/// the read-modify-write save path, which must reproduce the file's own
/// contents faithfully rather than the session's merged/migrated view.
pub(crate) fn read_config_file(path: &Path) -> Result<Config> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("reading config {}", path.display()))?;
    toml::from_str::<Config>(&text).with_context(|| format!("parsing config {}", path.display()))
}

pub(crate) fn merge_config(base: &mut Config, overlay: Config) {
    if overlay.default_profile.is_some() {
        base.default_profile = overlay.default_profile;
    }
    if overlay.moa != hi_ai::MoaConfig::default() {
        base.moa = overlay.moa;
    }
    base.profiles.extend(overlay.profiles);
    if overlay.rsi.is_some() {
        base.rsi = overlay.rsi;
    }
}

/// Repair profiles whose `api_key_env` holds a literal key instead of an env
/// var name — a bug in the setup wizard before the env-var-reference check was
/// added. The wizard stored a pasted key as `api_key_env` whenever it looked
/// like an env var name (all caps + digits + underscores), so on the next run
/// `resolve_api_key_for` tried to read an env var with that name and failed
/// with "env var … (from profile) is not set".
///
/// Fix: if `api_key_env` is set but no env var with that name exists in the
/// environment, AND the value does NOT look like an env var name (i.e. it's a
/// real key that was misclassified), move it to `api_key` (literal). If the
/// value *does* look like an env var name and the env var isn't set, leave it
/// alone — that's a legitimate but unfulfilled env var reference (the user
/// needs to set the env var), not a misplaced literal key. Moving an env var
/// name like "HI_API_KEY" to `api_key` would authenticate with the literal
/// string "HI_API_KEY" and get a 401.
pub(crate) fn migrate_api_key_env_to_literal(config: &mut Config, path: &Path) {
    let mut changed = false;
    // Set when a repair copies a live secret out of the environment into
    // `api_key` (below). We keep that repair in memory so the session works, but
    // must NOT persist it: writing the resolved secret to disk would leak it into
    // the file — including a project-local `hi.toml` that is routinely committed
    // to git — turning what was an env-var reference into a checked-in credential.
    let mut materialized_env_secret = false;
    for profile in config.profiles.values_mut() {
        // First, repair a bad migration from an earlier version of this fix:
        // the previous migration moved an env var *name* (like "HI_API_KEY")
        // into `api_key` when the env var wasn't set, causing 401s. If `api_key`
        // looks like an env var name and that env var is set, replace the value
        // with the env var's contents. If the env var isn't set, convert it back
        // to an `api_key_env` reference (the user intended an env var, not a
        // literal — they need to set it).
        if let Some(key) = profile.api_key.clone()
            && looks_like_env_var_name(&key)
        {
            if let Ok(val) = std::env::var(&key)
                && !val.is_empty()
            {
                // The env var is set — use its value as the literal key
                // in-memory only (do not persist the resolved secret to disk).
                profile.api_key = Some(val);
                changed = true;
                materialized_env_secret = true;
            } else {
                // Env var not set — this is an env var reference, not a key.
                // Move it back to api_key_env so resolve_api_key_for gives the
                // right error message ("env var … is not set") instead of a 401.
                profile.api_key = None;
                profile.api_key_env = Some(key);
                changed = true;
            }
        }

        let Some(env_name) = profile.api_key_env.clone() else {
            continue;
        };
        // If the env var is actually set, this is a legitimate reference — leave it.
        if std::env::var(&env_name).is_ok_and(|v| !v.is_empty()) {
            continue;
        }
        // If the value looks like an env var name, it could be a legitimate
        // (but unset) env var reference that the user intentionally configured.
        // BUT: the old setup wizard (save_config) always wrote api_key_env =
        // key_envs().first() (e.g. "HI_API_KEY") regardless of what the user
        // entered — it never stored the actual key. That pattern is: api_key_env
        // is one of the provider's standard key env names, the env var isn't set,
        // and there's no literal api_key. In that case, drop the bogus reference
        // so resolve falls through to the env-var candidates and the onboarding
        // error, prompting the user to re-enter their key (the new wizard stores
        // it as a literal api_key).
        if looks_like_env_var_name(&env_name) {
            let provider = profile.provider.unwrap_or(ProviderName::Openai);
            let is_standard = provider.key_envs().iter().any(|n| *n == env_name);
            let has_literal_key = profile.api_key.is_some();
            if is_standard && !has_literal_key {
                // Old buggy save_config output — drop it.
                profile.api_key_env = None;
                changed = true;
            }
            continue;
        }
        // The value doesn't look like an env var name and the env var isn't set:
        // it's a literal key that was misclassified by the old wizard heuristic.
        // Move it to `api_key` and clear `api_key_env`.
        profile.api_key_env = None;
        if profile.api_key.is_none() {
            profile.api_key = Some(env_name);
        }
        changed = true;
    }
    if changed && !materialized_env_secret {
        // Best-effort rewrite; if it fails we've still repaired the in-memory
        // config so this run works, just not the next one. Route through
        // `save_config_to` so the file keeps 0600 (a bare `fs::write` would drop
        // permissions, leaving keys world-readable). Skipped when the repair
        // materialized a live env secret — that stays in memory only (see above).
        let _ = save_config_to(config, path);
    }
}

pub(crate) fn local_config_path() -> PathBuf {
    PathBuf::from("hi.toml")
}

/// Guess a *layered* verification pipeline from marker files in `dir`: a cheap
/// compile/typecheck (and lint, when obviously configured) before tests, so the
/// model gets fast, localizable errors before the slower test stage. Used by
/// automatic verification so the proven verify-loop is zero-config. Empty =
/// unknown project.
#[cfg(test)]
pub fn detect_verify_pipeline(dir: &Path) -> Vec<VerifyStage> {
    hi_agent::detect_verify_pipeline(dir)
}

/// True when a bare `hi` has no model to run — used to trigger the interactive
/// setup wizard on a fresh terminal.
///
/// The test is "nothing *selectable*", not "nothing configured at all". A
/// config that defines profiles but names no `default_profile` (a project-local
/// `hi.toml` is the common case) resolves to no model, so it needs the wizard
/// just as much as an empty config does. This once also required
/// `file.profiles.is_empty()`, to protect existing profiles from a
/// `setup::save_config` that overwrote the whole config file; that made the
/// wizard unreachable in any directory containing a `hi.toml`, and left `hi`
/// printing "run `hi` on a real terminal for the interactive setup wizard" on a
/// real terminal. The save is a read-modify-write of one profile now, so the
/// trigger no longer has to be narrowed to compensate.
pub fn needs_setup(cli: &Cli, file: &Config) -> bool {
    nothing_selected(cli, file) && auto_select().is_none()
}

/// Everything `needs_setup` checks *except* the environment-key inference —
/// i.e. "this run has no model of its own". Split out so [`auto_selected_env`]
/// can ask the same question without duplicating the list.
fn nothing_selected(cli: &Cli, file: &Config) -> bool {
    cli.model.is_none()
        && cli.provider.is_none()
        && cli.profile.is_none()
        && file.default_profile.is_none()
        && std::env::var("HI_MODEL").is_err()
}

/// The env var that is the *only* thing configuring this run — nothing is
/// selected, but `auto_select` found an exported key and `resolve` will infer a
/// provider and model from it. `None` when anything else supplies the model.
///
/// A run in this state works but is invisible: no config is written, the model
/// is a built-in default the user never chose, and the next shell without that
/// variable exported fails. Callers use this to say so once at startup.
pub fn auto_selected_env(cli: &Cli, file: &Config) -> Option<&'static str> {
    if nothing_selected(cli, file) {
        auto_select_env_name()
    } else {
        None
    }
}

/// The default config file path to write the wizard's choices to.
pub fn default_config_path() -> Option<PathBuf> {
    let base = std::env::var_os("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".config")))?;
    Some(base.join("hi").join("config.toml"))
}

/// The path to write config to: an explicit `--config` path, a local `hi.toml`
/// if it exists, or the default global path. Unlike [`config_path`], this
/// returns a path even when the file doesn't exist yet (so we can create it).
pub fn writable_config_path(explicit: Option<&Path>) -> Option<PathBuf> {
    if let Some(path) = explicit {
        return Some(path.to_path_buf());
    }
    let local = PathBuf::from("hi.toml");
    if local.exists() {
        return Some(local);
    }
    default_config_path()
}

/// Mask an API key (or env var name) for display: first and last four
/// characters with an ellipsis. Char-based, so a key containing multi-byte
/// characters (e.g. pasted with a stray curly quote) can't panic a byte slice.
pub fn mask_key(key: &str) -> String {
    if key.is_empty() {
        return "(none)".to_string();
    }
    let chars: Vec<char> = key.chars().collect();
    if chars.len() > 8 {
        let head: String = chars[..4].iter().collect();
        let tail: String = chars[chars.len() - 4..].iter().collect();
        format!("{head}…{tail}")
    } else {
        "***".to_string()
    }
}

/// Serialize `config` to TOML and write it to `path`, creating parent dirs.
/// Sets 0600 permissions on Unix so API keys in the file aren't world-readable.
pub fn save_config_to(config: &Config, path: &Path) -> Result<()> {
    let toml = toml::to_string_pretty(config)
        .with_context(|| format!("serializing config to {}", path.display()))?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating config dir {}", parent.display()))?;
    }
    std::fs::write(path, toml).with_context(|| format!("writing {}", path.display()))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600));
    }
    Ok(())
}

/// Persist the two user-facing public-RSI controls without exposing gateway
/// plumbing. The complete effective section is written to the selected layer
/// so a project-local override does not accidentally discard an inherited
/// setting.
pub fn set_rsi_config(
    config: &mut Config,
    enabled: Option<bool>,
    maximum_cost_microusd: Option<u64>,
    channel: Option<String>,
    explicit: Option<&Path>,
) -> Result<()> {
    if let Some(value) = maximum_cost_microusd {
        anyhow::ensure!(
            (1..=15_000_000).contains(&value),
            "RSI spend limit must be greater than $0 and no more than $15"
        );
    }
    let section = config.rsi.get_or_insert_with(RsiSection::default);
    if let Some(enabled) = enabled {
        section.enabled = Some(enabled);
    }
    if let Some(maximum_cost_microusd) = maximum_cost_microusd {
        section.maximum_cost_microusd = Some(maximum_cost_microusd);
    }
    if let Some(channel) = channel {
        anyhow::ensure!(
            matches!(channel.as_str(), "stable" | "beta"),
            "RSI channel must be stable or beta"
        );
        section.channel = Some(channel);
    }
    let section = section.clone();
    let path =
        writable_config_path(explicit).context("could not determine a writable hi config path")?;
    rmw_config_file(&path, |target| target.rsi = Some(section))
}
