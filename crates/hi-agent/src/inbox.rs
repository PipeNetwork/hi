//! Approval inbox: parked confirms that need a human (`/inbox`, `hi inbox`).
//!
//! `/digest` is "what changed." This is "blocked on you."

use hi_policy::{ApprovalDecision, ApprovalRecord, ApprovalState, ApprovalStore};

/// Parsed `/inbox` / `hi inbox` argument.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum InboxArg {
    List,
    Allow(String),
    Deny(String),
    Usage,
}

/// Result of listing or deciding an inbox item.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InboxAction {
    pub lines: Vec<String>,
    /// Resume a goal parked with [`crate::GoalPauseReason::Approval`].
    pub resume_goal: bool,
    /// Unpause this loop id (`run_id` was `loop:{id}`).
    pub resume_loop: Option<u64>,
}

/// Parse `/inbox` args. Empty lists pending items.
pub fn parse_inbox_arg(arg: &str) -> InboxArg {
    let a = arg.trim();
    if a.is_empty() || a == "list" || a == "ls" {
        return InboxArg::List;
    }
    let mut parts = a.splitn(2, char::is_whitespace);
    let verb = parts.next().unwrap_or("");
    let id = parts.next().unwrap_or("").trim();
    match verb {
        "allow" | "approve" | "yes" if !id.is_empty() => InboxArg::Allow(id.to_string()),
        "deny" | "reject" | "no" if !id.is_empty() => InboxArg::Deny(id.to_string()),
        _ => InboxArg::Usage,
    }
}

/// Loop id encoded as `loop:{n}` on parked records.
pub fn loop_id_from_run_id(run_id: Option<&str>) -> Option<u64> {
    run_id?.strip_prefix("loop:")?.parse().ok()
}

/// List pending items or allow/deny by id (prefix ok when unique).
/// Does not claim or execute the original tool — the next fire retries.
pub fn apply_inbox(store: &dyn ApprovalStore, arg: &str) -> InboxAction {
    match parse_inbox_arg(arg) {
        InboxArg::List => list_inbox(store),
        InboxArg::Allow(id) => decide_inbox(store, &id, true),
        InboxArg::Deny(id) => decide_inbox(store, &id, false),
        InboxArg::Usage => InboxAction {
            lines: vec![
                "usage: /inbox | /inbox allow <id> | /inbox deny <id>".into(),
                "inbox is blocked-on-you; /digest is what changed.".into(),
            ],
            resume_goal: false,
            resume_loop: None,
        },
    }
}

fn pending_only(store: &dyn ApprovalStore) -> Result<Vec<ApprovalRecord>, String> {
    let records = store.pending().map_err(|err| format!("inbox: {err:#}"))?;
    Ok(records
        .into_iter()
        .filter(|r| r.state == ApprovalState::Pending)
        .collect())
}

fn list_inbox(store: &dyn ApprovalStore) -> InboxAction {
    let records = match pending_only(store) {
        Ok(r) => r,
        Err(err) => {
            return InboxAction {
                lines: vec![err],
                resume_goal: false,
                resume_loop: None,
            };
        }
    };
    if records.is_empty() {
        return InboxAction {
            lines: vec!["inbox empty — nothing parked for approval.".into()],
            resume_goal: false,
            resume_loop: None,
        };
    }
    let mut lines = vec![format!(
        "inbox — {} parked (blocked on you; /digest is what changed)",
        records.len()
    )];
    for rec in &records {
        let id = &rec.request.approval_id.0;
        let short = id.get(..8).unwrap_or(id);
        let tool = rec.request.tool.as_str();
        let title = rec.request.title.as_str();
        let detail = rec.request.redacted_detail.as_str();
        let origin = rec.request.run_id.as_deref().unwrap_or("parked");
        lines.push(format!("  {short}  {tool}  {title}"));
        if !detail.is_empty() && detail != title {
            lines.push(format!("          {detail}"));
        }
        lines.push(format!(
            "          {origin}  /inbox allow {short}  |  /inbox deny {short}"
        ));
    }
    InboxAction {
        lines,
        resume_goal: false,
        resume_loop: None,
    }
}

fn decide_inbox(store: &dyn ApprovalStore, token: &str, allow: bool) -> InboxAction {
    let records = match pending_only(store) {
        Ok(r) => r,
        Err(err) => {
            return InboxAction {
                lines: vec![err],
                resume_goal: false,
                resume_loop: None,
            };
        }
    };
    let Some(rec) = resolve_record(&records, token) else {
        return InboxAction {
            lines: vec![format!(
                "no pending inbox item matching {token:?} — /inbox lists ids"
            )],
            resume_goal: false,
            resume_loop: None,
        };
    };
    let id = rec.request.approval_id.clone();
    let decision = if allow {
        ApprovalDecision::Approved
    } else {
        ApprovalDecision::Denied
    };
    if let Err(err) = store.decide(&id, decision) {
        return InboxAction {
            lines: vec![format!("inbox decide failed: {err:#}")],
            resume_goal: false,
            resume_loop: None,
        };
    }
    let verb = if allow { "allowed" } else { "denied" };
    let short = id.0.get(..8).unwrap_or(id.0.as_str());
    InboxAction {
        lines: vec![format!(
            "✓ {verb} {short} ({}) — next fire retries",
            rec.request.tool
        )],
        resume_goal: true,
        resume_loop: loop_id_from_run_id(rec.request.run_id.as_deref()),
    }
}

fn resolve_record<'a>(records: &'a [ApprovalRecord], token: &str) -> Option<&'a ApprovalRecord> {
    let token = token.trim();
    if token.is_empty() {
        return None;
    }
    if let Some(exact) = records.iter().find(|r| r.request.approval_id.0 == token) {
        return Some(exact);
    }
    let hits: Vec<_> = records
        .iter()
        .filter(|r| r.request.approval_id.0.starts_with(token))
        .collect();
    if hits.len() == 1 { Some(hits[0]) } else { None }
}

/// Resume a goal that was paused only for an inbox approval.
pub fn resume_goal_after_inbox(agent: &mut crate::Agent) {
    if agent
        .structured_goal()
        .is_some_and(|g| g.pause_reason == crate::GoalPauseReason::Approval)
    {
        let _ = agent.try_set_goal_pause_reason(crate::GoalPauseReason::None);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hi_policy::{
        ApprovalId, CapabilityKind, CapabilityRequest, OperationDigest, ResourceScope,
        new_approval_id, now_ms,
    };
    use std::collections::HashMap;
    use std::sync::Mutex;

    struct MemStore(Mutex<HashMap<String, ApprovalRecord>>);

    impl MemStore {
        fn new() -> Self {
            Self(Mutex::new(HashMap::new()))
        }
    }

    impl ApprovalStore for MemStore {
        fn create(&self, request: CapabilityRequest) -> anyhow::Result<ApprovalRecord> {
            let rec = ApprovalRecord {
                request,
                state: ApprovalState::Pending,
                decided_at_ms: None,
                consumed_at_ms: None,
            };
            self.0
                .lock()
                .unwrap()
                .insert(rec.request.approval_id.0.clone(), rec.clone());
            Ok(rec)
        }
        fn get(&self, id: &ApprovalId) -> anyhow::Result<Option<ApprovalRecord>> {
            Ok(self.0.lock().unwrap().get(&id.0).cloned())
        }
        fn decide(
            &self,
            id: &ApprovalId,
            decision: ApprovalDecision,
        ) -> anyhow::Result<ApprovalRecord> {
            let mut map = self.0.lock().unwrap();
            let rec = map
                .get_mut(&id.0)
                .ok_or_else(|| anyhow::anyhow!("missing"))?;
            rec.state = match decision {
                ApprovalDecision::Approved => ApprovalState::Approved,
                _ => ApprovalState::Denied,
            };
            rec.decided_at_ms = Some(now_ms());
            Ok(rec.clone())
        }
        fn claim(
            &self,
            id: &ApprovalId,
            digest: &OperationDigest,
        ) -> anyhow::Result<ApprovalRecord> {
            let mut map = self.0.lock().unwrap();
            let rec = map
                .get_mut(&id.0)
                .ok_or_else(|| anyhow::anyhow!("missing"))?;
            anyhow::ensure!(rec.state == ApprovalState::Approved);
            anyhow::ensure!(rec.request.operation_digest == *digest);
            rec.state = ApprovalState::Consumed;
            rec.consumed_at_ms = Some(now_ms());
            Ok(rec.clone())
        }
        fn abandon_run(&self, _run_id: &str) -> anyhow::Result<u64> {
            Ok(0)
        }
        fn pending(&self) -> anyhow::Result<Vec<ApprovalRecord>> {
            Ok(self
                .0
                .lock()
                .unwrap()
                .values()
                .filter(|r| matches!(r.state, ApprovalState::Pending | ApprovalState::Approved))
                .cloned()
                .collect())
        }
    }

    fn sample_request(run_id: Option<&str>) -> CapabilityRequest {
        let created = now_ms();
        CapabilityRequest {
            approval_id: new_approval_id(),
            capability: CapabilityKind::ProcessExecution,
            scope: ResourceScope::Operation {
                workspace_id: "w".into(),
                label: "bash".into(),
            },
            operation_digest: OperationDigest("abc".into()),
            tool: "bash".into(),
            run_id: run_id.map(str::to_string),
            session_id: None,
            title: "Confirm shell mutation".into(),
            redacted_detail: "rm leftover".into(),
            created_at_ms: created,
            expires_at_ms: created + 86_400_000,
        }
    }

    #[test]
    fn parse_inbox_verbs() {
        assert_eq!(parse_inbox_arg(""), InboxArg::List);
        assert_eq!(parse_inbox_arg("list"), InboxArg::List);
        assert_eq!(
            parse_inbox_arg("allow deadbeef"),
            InboxArg::Allow("deadbeef".into())
        );
        assert_eq!(parse_inbox_arg("deny abc"), InboxArg::Deny("abc".into()));
        assert_eq!(parse_inbox_arg("allow"), InboxArg::Usage);
    }

    #[test]
    fn loop_id_parses_run_id() {
        assert_eq!(loop_id_from_run_id(Some("loop:7")), Some(7));
        assert_eq!(loop_id_from_run_id(Some("parked")), None);
        assert_eq!(loop_id_from_run_id(None), None);
    }

    #[test]
    fn list_and_allow_resume_loop() {
        let store = MemStore::new();
        let rec = store.create(sample_request(Some("loop:3"))).unwrap();
        let listed = apply_inbox(&store, "");
        assert!(listed.lines[0].contains("1 parked"), "{:?}", listed.lines);
        assert!(!listed.resume_goal);
        let short = rec.request.approval_id.0.get(..8).unwrap();
        let allowed = apply_inbox(&store, &format!("allow {short}"));
        assert!(allowed.lines[0].contains("allowed"), "{:?}", allowed.lines);
        assert!(allowed.resume_goal);
        assert_eq!(allowed.resume_loop, Some(3));
        let empty = apply_inbox(&store, "");
        assert!(empty.lines[0].contains("empty"), "{:?}", empty.lines);
    }
}
