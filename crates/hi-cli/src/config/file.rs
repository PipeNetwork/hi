use super::*;

use std::sync::atomic::{AtomicU64, Ordering};

static CONFIG_WRITE_NONCE: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Debug, Default, Deserialize)]
pub struct Config {
    pub default_profile: Option<String>,
    /// Default execution mode for sessions that do not override it in the
    /// selected profile.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub execution: Option<ExecutionMode>,
    /// Last `/config reasoning` choice for this machine. Applied when the
    /// active profile does not set its own `reasoning_effort`. `None` means
    /// off / endpoint default (same as an explicit `/config reasoning off`).
    #[serde(default)]
    pub reasoning_effort: Option<ReasoningEffort>,
    #[serde(default)]
    pub moa: hi_ai::MoaConfig,
    #[serde(default)]
    pub profiles: HashMap<String, Profile>,
    #[serde(default)]
    pub sync: Option<SyncSection>,
    /// Default for new portable workspaces; restored remote state wins.
    #[serde(default)]
    pub pipefs: PipeFsSection,
    #[serde(default)]
    pub rsi: Option<RsiSection>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub outcome: Option<OutcomeSection>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub x402: Option<X402Section>,
    /// MCP import gating for Claude `.mcp.json` / Codex / `.hi/mcp`.
    #[serde(default)]
    pub mcp_import: McpImportSection,
    /// First-party Pipe MCP attach (`[mcp.pipe]`).
    #[serde(default)]
    pub mcp: McpSection,
    /// Inject-gated `browser_exec` (default on; advertised on page/login/UI tasks).
    #[serde(default)]
    pub browser: BrowserSection,
    #[serde(default)]
    pub harness: HarnessConfig,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct OutcomeSection {
    /// Runtime-only provenance for an automatically merged repository section.
    #[serde(skip)]
    pub(crate) project_local: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mode: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base_url: Option<String>,
    /// Opaque credential-store/environment reference used by new writes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub api_key_ref: Option<String>,
    /// Legacy read-only credential fields; writers migrate them to `api_key_ref`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub api_key: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub api_key_env: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub offer: Option<String>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct RsiSection {
    /// Runtime-only provenance for an automatically merged repository section.
    #[serde(skip)]
    pub(crate) project_local: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub enabled: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base_url: Option<String>,
    /// Opaque credential-store/environment reference used by new writes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub api_key_ref: Option<String>,
    /// Legacy read-only credential fields; writers migrate them to `api_key_ref`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub api_key: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub api_key_env: Option<String>,
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

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct X402Section {
    /// When true, Pipenetwork x402 is an intended auth mode even without a keypair
    /// (paste-signature fallback).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub enabled: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub keypair: Option<PathBuf>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rpc: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_usd: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub auto_confirm: Option<bool>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct McpImportSection {
    #[serde(default)]
    pub claude: McpSourceImport,
    #[serde(default)]
    pub codex: McpSourceImport,
    #[serde(default)]
    pub hi: McpSourceImport,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct McpSourceImport {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub enabled: Option<bool>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub only: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub exclude: Vec<String>,
}

impl McpImportSection {
    pub fn to_policy(&self) -> hi_mcp::McpImportPolicy {
        hi_mcp::McpImportPolicy {
            hi: source_filter(&self.hi, true),
            claude: source_filter(&self.claude, true),
            codex: source_filter(&self.codex, false),
        }
    }
}

/// `[mcp]` — first-party Pipe attach plus optional per-server tool lists.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct McpSection {
    #[serde(default)]
    pub pipe: McpPipeSection,
    /// `[mcp.servers.<name>]` — `only` / `exclude` overlay for imported servers.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub servers: BTreeMap<String, McpServerPolicySection>,
}

/// `[mcp.servers.<name>]` tool visibility. Empty `only` means all tools (minus exclude).
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct McpServerPolicySection {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub only: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub exclude: Vec<String>,
}

impl McpSection {
    pub fn server_allowlists(&self) -> std::collections::HashMap<String, hi_mcp::ServerAllowList> {
        self.servers
            .iter()
            .map(|(name, section)| {
                (
                    name.clone(),
                    hi_mcp::ServerAllowList {
                        only: section.only.clone(),
                        exclude: section.exclude.clone(),
                    },
                )
            })
            .collect()
    }
}

/// `[mcp.pipe]` — auto-attach Pipe `/mcp` into `search_tool` / `use_tool`.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct McpPipeSection {
    /// Default true when omitted.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub enabled: Option<bool>,
    /// Additive extra tools. Nested chat/responses stay code-denied.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub allow: Vec<String>,
}

impl McpPipeSection {
    pub fn is_enabled(&self) -> bool {
        self.enabled.unwrap_or(true)
    }
}

fn source_filter(section: &McpSourceImport, default_enabled: bool) -> hi_mcp::McpSourceFilter {
    hi_mcp::McpSourceFilter {
        enabled: section.enabled.unwrap_or(default_enabled),
        only: section.only.clone(),
        exclude: section.exclude.clone(),
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct BrowserSection {
    /// Inject-gated `browser_exec` (default on). Set `false` to hide the schema.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub enabled: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub allow_private_urls: Option<bool>,
}

impl BrowserSection {
    pub fn is_enabled(&self) -> bool {
        self.enabled.unwrap_or(true)
    }

    pub fn allows_private_urls(&self) -> bool {
        self.allow_private_urls.unwrap_or(false)
    }
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
        if file
            .rsi
            .as_ref()
            .is_some_and(|rsi| rsi.project_local && enabled)
        {
            return Ok(RsiRequested::Off);
        }
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
    /// Runtime-only provenance for an automatically merged repository section.
    #[serde(skip)]
    pub(crate) project_local: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub base_url: Option<String>,
    /// Opaque credential-store/environment reference used by new writes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub api_key_ref: Option<String>,
    /// Legacy read-only credential fields; writers migrate them to `api_key_ref`.
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
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub enabled: bool,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct PipeFsSection {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub enabled: Option<bool>,
}

impl PipeFsSection {
    pub fn is_enabled(&self) -> bool {
        self.enabled.unwrap_or(false)
    }
}

impl serde::Serialize for Config {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut s = serializer.serialize_struct("Config", 14)?;
        if let Some(v) = &self.default_profile {
            s.serialize_field("default_profile", v)?;
        }
        if let Some(v) = &self.execution {
            s.serialize_field("execution", v)?;
        }
        if let Some(v) = &self.reasoning_effort {
            s.serialize_field("reasoning_effort", v)?;
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
        if self.pipefs != PipeFsSection::default() {
            s.serialize_field("pipefs", &self.pipefs)?;
        }
        if let Some(rsi) = &self.rsi {
            s.serialize_field("rsi", rsi)?;
        }
        if let Some(outcome) = &self.outcome {
            s.serialize_field("outcome", outcome)?;
        }
        if let Some(x402) = &self.x402 {
            s.serialize_field("x402", x402)?;
        }
        if self.mcp_import != McpImportSection::default() {
            s.serialize_field("mcp_import", &self.mcp_import)?;
        }
        if self.mcp != McpSection::default() {
            s.serialize_field("mcp", &self.mcp)?;
        }
        if self.browser != BrowserSection::default() {
            s.serialize_field("browser", &self.browser)?;
        }
        if !self.harness.is_empty() {
            s.serialize_field("harness", &self.harness)?;
        }
        s.end()
    }
}

// Omit unset profile fields so saving never materializes defaults.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Profile {
    /// Runtime-only provenance for an automatically merged repository profile.
    /// Such profiles may never name an ambient credential variable.
    #[serde(skip)]
    pub(crate) project_local: bool,
    /// Persisted folder trust for the merged project. Development auto-trust
    /// never authorizes redirecting model or repository data.
    #[serde(skip)]
    pub(crate) project_trusted: bool,
    /// Per-profile execution mode. Durable mode checkpoints progress at task
    /// boundaries and requires a persisted session.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub execution: Option<ExecutionMode>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider: Option<ProviderName>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub base_url: Option<String>,
    /// MCP endpoint used for metadata discovery, when supported by the provider.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mcp_url: Option<String>,
    /// Opaque credential-store or environment reference used by new profile
    /// writes. Examples: `auth-store://profile-api-key/...` and `env://NAME`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub api_key_ref: Option<String>,
    /// Legacy literal API key. It remains readable so existing profiles do not
    /// break, but production profile writers migrate it to `api_key_ref`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_key: Option<String>,
    /// Legacy name of an env var holding the API key for this profile.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_key_env: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_token_parameter: Option<hi_ai::OutputTokenParameter>,
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
    /// DeepSeek-specific OpenAI wire compatibility: auto, on, or off.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub deepseek_compat: Option<hi_ai::DeepSeekCompat>,
    /// Verifier-gated skill auto-curation: after a verified turn, distill a
    /// reusable technique into a learned skill. Defaults to off.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub curate_skills: Option<bool>,
    /// Advertise the read-only `explore` subagent tool. On by default; set to
    /// false to disable (e.g. for a very small local model).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub explore_subagents: Option<bool>,
    /// Claude-style suggested next prompt (ghost text) after turns. On by
    /// default; set to false to disable. Env `HI_SUGGEST_NEXT_PROMPT=0` also
    /// disables.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub suggest_next_prompt: Option<bool>,
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
    /// Metadata for a hi-managed local runtime. The endpoint remains an
    /// OpenAI-compatible profile field, while this describes how hi can
    /// recreate it after a restart without persisting a process id or port.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub runtime: Option<LocalRuntimeProfile>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub harness: HarnessOverrides,
}

/// Persisted intent for a local runtime. `kind` is currently `mlx`; the
/// backend string is kept extensible so future CUDA/GGUF runtimes can use the
/// same profile shape without changing provider semantics.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct LocalRuntimeProfile {
    pub kind: String,
    pub repo: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub backend: Option<String>,
    #[serde(default = "default_runtime_autostart")]
    pub autostart: bool,
    /// Optional machine-local MLX directory. When set, `repo` is retained as
    /// the legacy/display identity and no Hub download is attempted.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_path: Option<PathBuf>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quantization: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_window: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_mode: Option<ToolMode>,
}

fn default_runtime_autostart() -> bool {
    true
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
        // Repository config is untrusted input until the trust decision below.
        // Parse it without running legacy migrations: merely launching in a
        // checkout must never rewrite or reformat that checkout's `hi.toml`.
        let local = read_project_config(&local_path)?;
        let trusted = std::env::current_dir().ok().is_some_and(|cwd| {
            if project_has_sensitive_provider_routes(&local) {
                matches!(
                    hi_tools::folder_trust::resolve_sensitive_config_trust(&cwd),
                    hi_tools::folder_trust::TrustOutcome::Trusted
                )
            } else {
                hi_tools::folder_trust::folder_trust_granted(&cwd)
            }
        });
        merge_config_with_project_trust(&mut config, local, trusted);
    }

    config.moa.validate()?;
    Ok(config)
}

fn project_has_sensitive_provider_routes(config: &Config) -> bool {
    config.profiles.values().any(|profile| {
        let provider = profile.provider.unwrap_or(ProviderName::Openai);
        let api_url = profile
            .base_url
            .as_deref()
            .unwrap_or_else(|| provider.default_base_url());
        let remote_api = !hi_provider_config::is_loopback_endpoint(api_url);
        let remote_mcp = profile
            .mcp_url
            .as_deref()
            .is_some_and(|url| !hi_provider_config::is_loopback_endpoint(url));
        remote_api || remote_mcp
    })
}

pub(crate) fn read_config(path: &Path) -> Result<Config> {
    let mut config = read_config_file(path)?;
    config
        .moa
        .validate()
        .with_context(|| format!("validating MoA config {}", path.display()))?;
    migrate_api_key_env_to_literal(&mut config, path);
    super::credential_refs::migrate_persisted_credentials(&config, path);
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

/// Read repository configuration without any migration or write-back. Keep
/// this seam distinct from [`read_config`]: project input is not authorized to
/// mutate the checkout merely because the CLI inspected it.
pub(super) fn read_project_config(path: &Path) -> Result<Config> {
    read_config_file(path)
}

#[cfg(test)]
pub(crate) fn merge_config(base: &mut Config, overlay: Config) {
    merge_config_with_project_trust(base, overlay, false);
}

pub(crate) fn merge_config_with_project_trust(
    base: &mut Config,
    mut overlay: Config,
    trusted: bool,
) {
    base.harness.merge_project(&mut overlay.harness, trusted);
    for profile in overlay.profiles.values_mut() {
        profile.project_local = true;
        profile.project_trusted = trusted;
    }
    if overlay.default_profile.is_some() {
        base.default_profile = overlay.default_profile;
    }
    if overlay.execution.is_some() {
        base.execution = overlay.execution;
    }
    // Local/project files only override the machine default when they set one;
    // omitting the key keeps the global last `/config reasoning` choice.
    if overlay.reasoning_effort.is_some() {
        base.reasoning_effort = overlay.reasoning_effort;
    }
    // Repository config may turn the multi-model fan-out off, but must not
    // re-enable it or replace its model set. Either would increase billable
    // egress using a machine-level provider credential.
    if !overlay.moa.enabled {
        base.moa.enabled = false;
    }
    base.profiles.extend(overlay.profiles);
    merge_project_sync(&mut base.sync, overlay.sync, trusted);
    // Repository input can opt out but cannot cause a workspace upload. A
    // global/explicit config remains the authority for the default-on choice.
    if overlay.pipefs.enabled == Some(false) {
        base.pipefs.enabled = Some(false);
    }
    merge_project_rsi(&mut base.rsi, overlay.rsi);
    merge_project_outcome(&mut base.outcome, overlay.outcome, trusted);
    merge_project_x402(&mut base.x402, overlay.x402);
    merge_mcp_import_source(&mut base.mcp_import.claude, overlay.mcp_import.claude);
    merge_mcp_import_source(&mut base.mcp_import.codex, overlay.mcp_import.codex);
    merge_mcp_import_source(&mut base.mcp_import.hi, overlay.mcp_import.hi);
    if overlay.mcp != McpSection::default() {
        // Pipe is enabled by default and `allow` is explicitly additive. A
        // repository may disable it, but cannot re-enable it or enlarge the
        // machine allowlist.
        if overlay.mcp.pipe.enabled == Some(false) {
            base.mcp.pipe.enabled = Some(false);
        }
        for (name, section) in overlay.mcp.servers {
            let target = base.mcp.servers.entry(name).or_default();
            merge_restrictive_lists(
                &mut target.only,
                section.only,
                &mut target.exclude,
                section.exclude,
            );
        }
    }
    if overlay.browser.enabled == Some(false) {
        base.browser.enabled = Some(false);
    }
    if overlay.browser.allow_private_urls == Some(false) {
        base.browser.allow_private_urls = Some(false);
    }
}

fn merge_project_x402(base: &mut Option<X402Section>, project: Option<X402Section>) {
    let Some(project) = project else {
        return;
    };
    let tightens = project.enabled == Some(false)
        || project.auto_confirm == Some(false)
        || project.max_usd.is_some();
    if base.is_none() && !tightens {
        return;
    }
    let target = base.get_or_insert_with(X402Section::default);
    if project.enabled == Some(false) {
        target.enabled = Some(false);
    }
    if project.auto_confirm == Some(false) {
        target.auto_confirm = Some(false);
    }
    if let Some(project_max) = project
        .max_usd
        .filter(|value| value.is_finite() && *value > 0.0)
    {
        let current = target.max_usd.unwrap_or(hi_ai::X402_DEFAULT_MAX_USD);
        target.max_usd = Some(current.min(project_max));
    }
    // `keypair` and `rpc` select a wallet and a transaction endpoint; both are
    // machine/user authority and never inherited from repository config.
}

fn merge_project_sync(base: &mut Option<SyncSection>, project: Option<SyncSection>, trusted: bool) {
    let Some(project) = project else {
        return;
    };
    let target = base.get_or_insert_with(SyncSection::default);
    // Repository config may persistently tighten the machine policy, but it
    // must never turn transcript upload on merely because the folder opened.
    if project
        .mode
        .is_some_and(|mode| mode != crate::sync_store::SyncMode::On)
    {
        target.mode = project.mode;
    }
    if !project.enabled {
        target.enabled = false;
    }
    if !trusted {
        return;
    }
    if project.base_url.is_some() {
        target.base_url = project.base_url;
        // Endpoint and credential are coupled: changing the route discards an
        // inherited global credential instead of forwarding it.
        target.api_key = project.api_key;
        target.api_key_env = project.api_key_env;
        target.api_key_ref = project.api_key_ref;
        target.project_local = true;
    } else if project.api_key_ref.is_some()
        || project.api_key.is_some()
        || project.api_key_env.is_some()
    {
        target.api_key = project.api_key;
        target.api_key_env = project.api_key_env;
        target.api_key_ref = project.api_key_ref;
        target.project_local = true;
    }
    if project.machine_id.is_some() {
        target.machine_id = project.machine_id;
    }
}

fn merge_project_rsi(base: &mut Option<RsiSection>, project: Option<RsiSection>) {
    let Some(project) = project else {
        return;
    };
    // Project RSI configuration may only disable the remote path and reduce
    // its budget. Endpoint/channel/auth changes and `enabled = true` are
    // machine/user decisions because RSI uploads the repository.
    let should_create = project.enabled == Some(false) || project.maximum_cost_microusd.is_some();
    if base.is_none() && !should_create {
        return;
    }
    let target = base.get_or_insert_with(RsiSection::default);
    if project.enabled == Some(false) {
        target.enabled = Some(false);
    }
    if let Some(project_max) = project.maximum_cost_microusd {
        let current = target.maximum_cost_microusd.unwrap_or(15_000_000);
        target.maximum_cost_microusd = Some(current.min(project_max));
    }
    target.project_local = true;
}

fn merge_project_outcome(
    base: &mut Option<OutcomeSection>,
    project: Option<OutcomeSection>,
    trusted: bool,
) {
    let Some(project) = project else {
        return;
    };
    let current_rank = base
        .as_ref()
        .and_then(|section| section.mode.as_deref())
        .map(outcome_mode_rank)
        .unwrap_or(0); // OutcomeMode::Chat (ordinary direct-provider default)
    let safe_mode = project
        .mode
        .as_deref()
        .filter(|mode| trusted || outcome_mode_rank(mode) <= current_rank)
        .map(str::to_string);
    if base.is_none() && safe_mode.is_none() && !trusted {
        return;
    }
    let target = base.get_or_insert_with(OutcomeSection::default);
    if safe_mode.is_some() {
        target.mode = safe_mode;
    }
    if !trusted {
        return;
    }
    if project.base_url.is_some() {
        target.base_url = project.base_url;
        target.api_key_ref = project.api_key_ref;
        target.api_key = project.api_key;
        target.api_key_env = project.api_key_env;
        target.project_local = true;
    } else if project.api_key_ref.is_some()
        || project.api_key.is_some()
        || project.api_key_env.is_some()
    {
        target.api_key_ref = project.api_key_ref;
        target.api_key = project.api_key;
        target.api_key_env = project.api_key_env;
        target.project_local = true;
    }
    if project.offer.is_some() {
        target.offer = project.offer;
    }
}

fn outcome_mode_rank(mode: &str) -> u8 {
    // Derive trust ordering from the runtime parser so accepted aliases and
    // fail-closed handling of unknown values cannot drift apart.
    match hi_outcome::OutcomeMode::parse(mode) {
        hi_outcome::OutcomeMode::Chat => 0,
        hi_outcome::OutcomeMode::Auto => 1,
        hi_outcome::OutcomeMode::Tasks => 2,
    }
}

fn merge_mcp_import_source(base: &mut McpSourceImport, overlay: McpSourceImport) {
    if overlay.enabled == Some(false) {
        base.enabled = Some(false);
    }
    merge_restrictive_lists(
        &mut base.only,
        overlay.only,
        &mut base.exclude,
        overlay.exclude,
    );
}

fn merge_restrictive_lists(
    base_only: &mut Vec<String>,
    overlay_only: Vec<String>,
    base_exclude: &mut Vec<String>,
    overlay_exclude: Vec<String>,
) {
    // Empty `only` means all. A project can narrow an inherited allowlist, but
    // never replace it with a broader or disjoint one.
    if !overlay_only.is_empty() {
        if base_only.is_empty() {
            *base_only = overlay_only;
        } else {
            let intersection: Vec<String> = base_only
                .iter()
                .filter(|item| overlay_only.contains(item))
                .cloned()
                .collect();
            // Empty `only` means *all*, not none. When two restrictions are
            // disjoint, preserve the machine allowlist rather than accidentally
            // converting it into an unrestricted list; a project that wants no
            // tools can disable the source or exclude explicit tools.
            if !intersection.is_empty() {
                *base_only = intersection;
            }
        }
    }
    // Exclusions are monotonic: the union is always at least as restrictive.
    for item in overlay_exclude {
        if !base_exclude.contains(&item) {
            base_exclude.push(item);
        }
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

#[cfg(test)]
pub fn detect_verify_pipeline_with(dir: &Path, clippy: bool) -> Vec<VerifyStage> {
    hi_agent::detect_verify_pipeline_with(dir, clippy)
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
    nothing_selected(cli, file) && auto_select(file).is_none()
}

/// True when a local Solana keypair, stored x402 credit token, or `/login x402`
/// is enough to select Pipenetwork without the pairing wizard.
pub fn x402_configured(file: &Config) -> bool {
    env_nonempty("HI_X402_KEYPAIR")
        || hi_ai::has_credit_token()
        || file.x402.as_ref().is_some_and(|section| {
            section.enabled == Some(true)
                || section
                    .keypair
                    .as_ref()
                    .is_some_and(|path| !path.as_os_str().is_empty())
        })
}

fn env_nonempty(name: &str) -> bool {
    std::env::var(name).is_ok_and(|value| !value.trim().is_empty())
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
/// Creates the file with 0600 permissions on Unix so API keys in the file
/// are never world-readable — the mode is set atomically at creation via
/// `OpenOptions`, not chmod'd after the write (which left a readable window
/// and discarded chmod failures). The live path is replaced by rename so a
/// crash or disk-full mid-write cannot truncate an existing key file.
pub fn save_config_to(config: &Config, path: &Path) -> Result<()> {
    let toml = toml::to_string_pretty(config)
        .with_context(|| format!("serializing config to {}", path.display()))?;
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating config dir {}", parent.display()))?;
    }
    write_private_atomic(path, toml.as_bytes())?;
    Ok(())
}

/// Write `bytes` to `path` via a private, exclusively-created same-directory
/// temp file, then rename it into place. Unique create-new names prevent
/// concurrent saves from sharing an inode, and `O_NOFOLLOW` prevents a
/// pre-planted symlink from receiving config secrets.
fn write_private_atomic(path: &Path, bytes: &[u8]) -> Result<()> {
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("config");
    for _ in 0..32 {
        let nonce = CONFIG_WRITE_NONCE.fetch_add(1, Ordering::Relaxed);
        let tmp = parent.join(format!(".{name}.{}.{nonce}.tmp", std::process::id()));
        match write_private(&tmp, bytes) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => {
                return Err(error).with_context(|| format!("writing {}", tmp.display()));
            }
        }
        let result = replace_file(&tmp, path);
        if result.is_err() && path.exists() {
            // The destination still holds the previous key file (or a restored
            // backup), so the private temporary copy is no longer needed. If
            // the destination is missing, retain it for manual recovery.
            let _ = std::fs::remove_file(&tmp);
        }
        return result;
    }
    anyhow::bail!("could not allocate a private config temporary file")
}

/// Replace `to` with `from` atomically, including over an existing file.
#[cfg(not(windows))]
fn replace_file(from: &Path, to: &Path) -> Result<()> {
    std::fs::rename(from, to).with_context(|| format!("replacing {}", to.display()))
}

#[cfg(windows)]
fn replace_file(from: &Path, to: &Path) -> Result<()> {
    use std::os::windows::ffi::OsStrExt as _;

    const MOVEFILE_REPLACE_EXISTING: u32 = 0x0000_0001;
    const MOVEFILE_WRITE_THROUGH: u32 = 0x0000_0008;

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn MoveFileExW(existing: *const u16, replacement: *const u16, flags: u32) -> i32;
    }

    fn wide(path: &Path) -> std::io::Result<Vec<u16>> {
        let mut value = path.as_os_str().encode_wide().collect::<Vec<_>>();
        if value.contains(&0) {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "config path contains an interior NUL",
            ));
        }
        value.push(0);
        Ok(value)
    }

    let from = wide(from)?;
    let to = wide(to)?;
    // SAFETY: both path buffers are NUL-terminated and remain live throughout
    // the Windows atomic-replacement call.
    let result = unsafe {
        MoveFileExW(
            from.as_ptr(),
            to.as_ptr(),
            MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH,
        )
    };
    if result == 0 {
        Err(std::io::Error::last_os_error()).with_context(|| "replacing config atomically")
    } else {
        Ok(())
    }
}

/// Write `bytes` to `path` with owner-only permissions from the start.
/// On Unix the file is created with mode 0600 atomically. Existing paths are
/// rejected rather than opened, so planted symlinks and hard links are never
/// followed or truncated.
#[cfg(unix)]
pub(super) fn write_private(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    use std::io::Write;
    use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC)
        .open(path)?;
    file.set_permissions(std::fs::Permissions::from_mode(0o600))?;
    file.write_all(bytes)?;
    // Durable before rename: without this, a crash after replace can leave
    // an empty/partial key file whose old contents are already gone.
    file.sync_all()?;
    Ok(())
}

#[cfg(not(unix))]
pub(super) fn write_private(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    use std::io::Write as _;
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)?;
    file.write_all(bytes)?;
    file.sync_all()
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
    let mut section = section.clone();
    let path =
        writable_config_path(explicit).context("could not determine a writable hi config path")?;
    (section.api_key_ref, section.api_key, section.api_key_env) =
        super::credential_refs::seal_credential_fields(
            "config-api-key/rsi",
            "rsi",
            &path,
            section.api_key_ref.take(),
            section.api_key.take(),
            section.api_key_env.take(),
        )?;
    config.rsi = Some(section.clone());
    rmw_config_file(&path, |target| target.rsi = Some(section))
}
