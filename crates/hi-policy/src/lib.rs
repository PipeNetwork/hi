//! Typed capability and approval contracts for interactive and background hi
//! work. Storage implementations live in the CLI/state layer.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CapabilityKind {
    WorkspaceRead,
    WorkspaceWrite,
    ProcessExecution,
    NetworkAccess,
    DelegateApplication,
    WorkflowExecution,
}

/// Classify the effect of a tool before it crosses the execution boundary.
/// This intentionally errs toward a stronger capability: unknown tools are
/// treated as process-capable by callers that need a fail-closed policy.
pub fn capability_kind_for_tool(tool: &str) -> CapabilityKind {
    match tool {
        "read" | "list" | "grep" | "glob" | "git_diff" | "bash_output" => {
            CapabilityKind::WorkspaceRead
        }
        "write" | "edit" | "multi_edit" | "apply_patch" | "delete" | "move" => {
            CapabilityKind::WorkspaceWrite
        }
        "bash" | "shell" => CapabilityKind::ProcessExecution,
        "web_search" | "web_fetch" | "web_download" | "http_request" => {
            CapabilityKind::NetworkAccess
        }
        "delegate" | "explore" => CapabilityKind::DelegateApplication,
        "workflow" | "run_workflow" => CapabilityKind::WorkflowExecution,
        _ => CapabilityKind::ProcessExecution,
    }
}

pub fn capability_is_read_only(kind: &CapabilityKind) -> bool {
    matches!(kind, CapabilityKind::WorkspaceRead)
}

/// Normalize only shell-insignificant outer whitespace. We deliberately do
/// not collapse internal whitespace because it can change quoted arguments or
/// shell heredoc content.
pub fn normalize_command(command: &str) -> String {
    command.trim().to_string()
}

pub fn normalize_cwd(cwd: &str) -> String {
    cwd.trim().to_string()
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "type")]
pub enum ResourceScope {
    Workspace {
        workspace_id: String,
    },
    Paths {
        workspace_id: String,
        paths: Vec<String>,
    },
    Command {
        workspace_id: String,
        command: String,
        cwd: String,
    },
    Workflow {
        workflow_id: String,
        run_id: String,
    },
    Operation {
        workspace_id: String,
        label: String,
    },
}

/// Stable scope identity shared by policy, resource registries, and audit
/// records. Scope inheritance is explicit and only flows toward descendants.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScopeKind {
    User,
    Workspace,
    Worktree,
    Session,
    Run,
    Attempt,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScopeRef {
    pub scope_id: String,
    pub kind: ScopeKind,
    pub parent_scope_id: Option<String>,
    pub workspace_id: Option<String>,
    pub owner_id: String,
    #[serde(default)]
    pub inherited: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expires_at_ms: Option<u64>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Principal {
    pub id: String,
    pub kind: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Provenance {
    pub principal: Principal,
    pub source: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub run_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub attempt_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_ref: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub correlation_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub policy_version: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PolicySnapshot {
    pub version: String,
    pub digest: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PolicyDecision {
    pub decision: String,
    pub reason: String,
    pub principal: Principal,
    pub scope: Option<ScopeRef>,
    pub policy: PolicySnapshot,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub approval_id: Option<ApprovalId>,
    pub operation_digest: OperationDigest,
    pub decided_at_ms: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AuditRecord {
    pub audit_id: String,
    pub decision: PolicyDecision,
    pub provenance: Provenance,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub route_ref: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub effect_ref: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub event_ref: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct OperationDigest(pub String);

impl OperationDigest {
    pub fn calculate(
        kind: &CapabilityKind,
        tool: &str,
        arguments: &serde_json::Value,
        workspace_id: &str,
        scope: &ResourceScope,
        prepared_mutation: Option<&str>,
    ) -> Self {
        let scope = canonical_scope(scope);
        let material = serde_json::json!({
            "kind": kind,
            "tool": tool,
            "arguments": arguments,
            "workspace_id": workspace_id,
            "scope": scope,
            "prepared_mutation": prepared_mutation,
        });
        Self(
            blake3::hash(material.to_string().as_bytes())
                .to_hex()
                .to_string(),
        )
    }
}

fn canonical_scope(scope: &ResourceScope) -> ResourceScope {
    match scope {
        ResourceScope::Paths {
            workspace_id,
            paths,
        } => {
            let mut paths = paths.clone();
            paths.sort();
            paths.dedup();
            ResourceScope::Paths {
                workspace_id: workspace_id.clone(),
                paths,
            }
        }
        other => other.clone(),
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ApprovalId(pub String);

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ApprovalState {
    Pending,
    Approved,
    Denied,
    Expired,
    Consumed,
    Abandoned,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CapabilityRequest {
    pub approval_id: ApprovalId,
    pub capability: CapabilityKind,
    pub scope: ResourceScope,
    pub operation_digest: OperationDigest,
    pub tool: String,
    pub run_id: Option<String>,
    pub session_id: Option<String>,
    pub title: String,
    pub redacted_detail: String,
    pub created_at_ms: u64,
    pub expires_at_ms: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ApprovalRecord {
    pub request: CapabilityRequest,
    pub state: ApprovalState,
    pub decided_at_ms: Option<u64>,
    pub consumed_at_ms: Option<u64>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ApprovalDecision {
    Approved,
    Denied,
    Cancelled,
    Unavailable,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Authorization {
    Allowed,
    RequiresApproval(CapabilityRequest),
    Denied(String),
}

pub trait ApprovalStore: Send + Sync {
    fn create(&self, request: CapabilityRequest) -> anyhow::Result<ApprovalRecord>;
    fn get(&self, id: &ApprovalId) -> anyhow::Result<Option<ApprovalRecord>>;
    fn decide(&self, id: &ApprovalId, decision: ApprovalDecision)
    -> anyhow::Result<ApprovalRecord>;
    fn claim(&self, id: &ApprovalId, digest: &OperationDigest) -> anyhow::Result<ApprovalRecord>;
    fn abandon_run(&self, run_id: &str) -> anyhow::Result<u64>;
    /// Mark interactive requests from a previous process as abandoned. The
    /// default keeps third-party stores source-compatible; durable local
    /// stores should override it.
    fn abandon_interactive(&self) -> anyhow::Result<u64> {
        Ok(0)
    }
    fn pending(&self) -> anyhow::Result<Vec<ApprovalRecord>>;
}

/// Final-boundary policy hook. Frontends may ask for approval, but execution
/// code is expected to consult this interface immediately before a side
/// effect and fail closed on an unavailable implementation.
pub trait CapabilityAuthorizer: Send + Sync {
    fn authorize(&self, request: &CapabilityRequest) -> anyhow::Result<Authorization>;
}

pub fn new_approval_id() -> ApprovalId {
    ApprovalId(uuid::Uuid::new_v4().to_string())
}

pub fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_millis() as u64)
        .unwrap_or(0)
}

pub fn approval_request(
    capability: CapabilityKind,
    scope: ResourceScope,
    operation_digest: OperationDigest,
    tool: impl Into<String>,
    run_id: Option<String>,
    session_id: Option<String>,
    title: impl Into<String>,
    redacted_detail: impl Into<String>,
) -> CapabilityRequest {
    let created_at_ms = now_ms();
    CapabilityRequest {
        approval_id: new_approval_id(),
        capability,
        scope,
        operation_digest,
        tool: tool.into(),
        run_id,
        session_id,
        title: title.into(),
        redacted_detail: redacted_detail.into(),
        created_at_ms,
        expires_at_ms: created_at_ms.saturating_add(24 * 60 * 60 * 1_000),
    }
}

pub fn canonical_fields(fields: &BTreeMap<String, serde_json::Value>) -> String {
    serde_json::to_string(fields).unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn digest_is_stable_and_changes_with_scope() {
        let scope = ResourceScope::Paths {
            workspace_id: "w".into(),
            paths: vec!["src/lib.rs".into()],
        };
        let first = OperationDigest::calculate(
            &CapabilityKind::WorkspaceWrite,
            "edit",
            &serde_json::json!({"x": 1}),
            "w",
            &scope,
            Some("patch"),
        );
        let second = OperationDigest::calculate(
            &CapabilityKind::WorkspaceWrite,
            "edit",
            &serde_json::json!({"x": 1}),
            "w",
            &scope,
            Some("patch"),
        );
        assert_eq!(first, second);
        let changed = ResourceScope::Paths {
            workspace_id: "w".into(),
            paths: vec!["src/main.rs".into()],
        };
        assert_ne!(
            first,
            OperationDigest::calculate(
                &CapabilityKind::WorkspaceWrite,
                "edit",
                &serde_json::json!({"x": 1}),
                "w",
                &changed,
                Some("patch"),
            )
        );
        let reordered = ResourceScope::Paths {
            workspace_id: "w".into(),
            paths: vec!["src/lib.rs".into(), "src/main.rs".into()],
        };
        let sorted = ResourceScope::Paths {
            workspace_id: "w".into(),
            paths: vec!["src/main.rs".into(), "src/lib.rs".into()],
        };
        assert_eq!(
            OperationDigest::calculate(
                &CapabilityKind::WorkspaceWrite,
                "edit",
                &serde_json::json!({"x": 1}),
                "w",
                &reordered,
                None,
            ),
            OperationDigest::calculate(
                &CapabilityKind::WorkspaceWrite,
                "edit",
                &serde_json::json!({"x": 1}),
                "w",
                &sorted,
                None,
            )
        );
    }
}
