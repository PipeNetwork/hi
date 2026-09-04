//! Versioned execution policy attached to catalog entries and tool envelopes.

pub use hi_workspace::EffectScope;
use serde::{Deserialize, Serialize};

use super::{SpeculationClass, ToolCapability};

pub const TOOL_METADATA_SCHEMA_VERSION: u16 = 1;

/// Whether a completed invocation can be repeated after an ambiguous result.
/// Operation-specific idempotency keys are attached by the coordinator.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReplayClass {
    PureWorkspace,
    IdempotentExternal,
    NonReplayableExternal,
}

/// Resources a tool may need at its most permissive supported invocation.
/// Runtime policy may narrow this record, but must never silently widen it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResourceAccess {
    pub workspace_read: bool,
    pub workspace_write: bool,
    pub process: bool,
    pub network: bool,
    pub credentials: bool,
    pub session: bool,
    pub mcp: bool,
}

impl ResourceAccess {
    pub const fn unrestricted() -> Self {
        Self {
            workspace_read: true,
            workspace_write: true,
            process: true,
            network: true,
            credentials: true,
            session: true,
            mcp: true,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OutputOverflowPolicy {
    Truncate,
    ArtifactReference,
    Reject,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct OutputPolicy {
    pub max_inline_bytes: u64,
    pub overflow: OutputOverflowPolicy,
    pub redact_secrets: bool,
}

impl OutputPolicy {
    pub const fn bounded_artifact() -> Self {
        Self {
            max_inline_bytes: 50_000,
            overflow: OutputOverflowPolicy::ArtifactReference,
            redact_secrets: true,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArtifactPolicy {
    pub allowed: bool,
    pub max_count: u16,
    pub max_total_bytes: u64,
    pub digest_required: bool,
}

impl ArtifactPolicy {
    pub const fn bounded() -> Self {
        Self {
            allowed: true,
            max_count: 16,
            max_total_bytes: 64 * 1024 * 1024,
            digest_required: true,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolPolicy {
    pub effect_scope: EffectScope,
    pub replay_class: ReplayClass,
    pub resource_access: ResourceAccess,
    pub output: OutputPolicy,
    pub artifacts: ArtifactPolicy,
}

impl ToolPolicy {
    pub(super) const fn classified(
        capability: ToolCapability,
        speculation: SpeculationClass,
        read_only: bool,
        filesystem_mutating: bool,
    ) -> Self {
        Self {
            effect_scope: effect_scope(capability, read_only, filesystem_mutating),
            replay_class: replay_class(capability, speculation, read_only),
            resource_access: resource_access(capability, read_only, filesystem_mutating),
            output: OutputPolicy::bounded_artifact(),
            artifacts: ArtifactPolicy::bounded(),
        }
    }

    pub(super) const fn program_dispatch() -> Self {
        Self {
            effect_scope: EffectScope::LiveWriter,
            replay_class: ReplayClass::NonReplayableExternal,
            resource_access: ResourceAccess::unrestricted(),
            output: OutputPolicy::bounded_artifact(),
            artifacts: ArtifactPolicy::bounded(),
        }
    }

    pub const fn conservative() -> Self {
        Self::program_dispatch()
    }
}

const fn effect_scope(
    capability: ToolCapability,
    read_only: bool,
    filesystem_mutating: bool,
) -> EffectScope {
    if read_only {
        EffectScope::ReadOnly
    } else if matches!(capability, ToolCapability::Subagent) {
        EffectScope::CandidateOnly
    } else if filesystem_mutating
        || matches!(
            capability,
            ToolCapability::Process
                | ToolCapability::Background
                | ToolCapability::Mcp
                | ToolCapability::Web
                | ToolCapability::Memory
        )
    {
        EffectScope::LiveWriter
    } else {
        EffectScope::ReadOnly
    }
}

const fn replay_class(
    capability: ToolCapability,
    speculation: SpeculationClass,
    read_only: bool,
) -> ReplayClass {
    if matches!(speculation, SpeculationClass::IdempotentExternal) {
        ReplayClass::IdempotentExternal
    } else if !read_only
        && matches!(
            capability,
            ToolCapability::Process
                | ToolCapability::Background
                | ToolCapability::Web
                | ToolCapability::Subagent
                | ToolCapability::Mcp
        )
    {
        ReplayClass::NonReplayableExternal
    } else {
        ReplayClass::PureWorkspace
    }
}

const fn resource_access(
    capability: ToolCapability,
    read_only: bool,
    filesystem_mutating: bool,
) -> ResourceAccess {
    let dynamic = !read_only
        && matches!(
            capability,
            ToolCapability::Process
                | ToolCapability::Background
                | ToolCapability::Subagent
                | ToolCapability::Mcp
        );
    ResourceAccess {
        workspace_read: matches!(
            capability,
            ToolCapability::Repository
                | ToolCapability::Mutation
                | ToolCapability::Process
                | ToolCapability::Background
                | ToolCapability::Lsp
                | ToolCapability::Subagent
                | ToolCapability::Memory
                | ToolCapability::Skill
        ),
        workspace_write: filesystem_mutating || dynamic,
        process: matches!(
            capability,
            ToolCapability::Process | ToolCapability::Background | ToolCapability::Subagent
        ),
        network: dynamic
            || matches!(
                capability,
                ToolCapability::Web | ToolCapability::Subagent | ToolCapability::Mcp
            ),
        credentials: dynamic || matches!(capability, ToolCapability::Web | ToolCapability::Mcp),
        session: matches!(
            capability,
            ToolCapability::Coordination | ToolCapability::Structure | ToolCapability::Memory
        ),
        mcp: matches!(capability, ToolCapability::Mcp),
    }
}
