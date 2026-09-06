use hi_ai::{CompatMode, DeepSeekCompat, ReasoningEffort, ToolMode};

/// Display and capability-cache identities for a selected provider route.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AgentProviderRoute {
    /// Stable user-facing provider family or route label.
    pub label: String,
    /// Credential-free identity that distinguishes concrete endpoints.
    pub capability_identity: String,
}

impl AgentProviderRoute {
    pub fn new(label: impl Into<String>, capability_identity: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            capability_identity: capability_identity.into(),
        }
    }
}

/// Model identity, sampling, and provider routing.
#[derive(Clone, Debug)]
pub struct AgentRouting {
    pub model: String,
    /// Human-readable effective provider route, when known by the frontend.
    pub provider_route: Option<String>,
    /// Credential-free capability identity; older frontends fall back to `provider_route`.
    pub capability_route: Option<String>,
    /// The user/config requested output-token cap before live model metadata is
    /// applied. Kept separately so `/model` switches can recompute the active
    /// cap without inheriting the previous route's live limit.
    pub requested_max_tokens: u32,
    pub max_tokens: u32,
    /// True when the user deliberately set the cap (CLI or non-default profile).
    /// Explicit caps are honored, only clamped downward to a model's advertised
    /// limit.
    pub max_tokens_explicit: bool,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub output_token_parameter: hi_ai::OutputTokenParameter,
    pub thinking_budget: Option<u32>,
    /// Abstract reasoning level (`reasoning_effort`) applied to every main-turn
    /// request on OpenAI-compatible endpoints that support it; `None` leaves the
    /// endpoint default. See [`hi_ai::ReasoningEffort`]. Housekeeping calls
    /// (compaction/memory/recap) deliberately leave this off. Set via
    /// `--reasoning-effort`, a profile, or `/config reasoning <level>`.
    pub reasoning_effort: Option<ReasoningEffort>,
    pub tool_mode: ToolMode,
    pub compat: CompatMode,
    pub deepseek_compat: DeepSeekCompat,
    /// Model context window, when known — used to show how full it is.
    pub context_window: Option<u32>,
}

impl Default for AgentRouting {
    fn default() -> Self {
        Self {
            model: String::new(),
            provider_route: None,
            capability_route: None,
            requested_max_tokens: 8192,
            max_tokens: 8192,
            max_tokens_explicit: false,
            temperature: None,
            top_p: None,
            output_token_parameter: hi_ai::OutputTokenParameter::Auto,
            thinking_budget: None,
            reasoning_effort: None,
            tool_mode: ToolMode::default(),
            compat: CompatMode::default(),
            deepseek_compat: DeepSeekCompat::default(),
            context_window: None,
        }
    }
}
