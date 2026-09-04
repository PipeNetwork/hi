//! Visible and durable audit evidence for explicit safety-policy overrides.

pub(crate) fn record_folder_trust_override(
    workspace_root: &std::path::Path,
    state_root: &std::path::Path,
) {
    if !hi_tools::folder_trust::folder_trust_inert() {
        return;
    }
    eprintln!(
        "\x1b[33mwarning: folder trust is DISABLED by the explicit HI_FOLDER_TRUST override; repository configuration may execute without a stored trust grant\x1b[0m"
    );
    let Ok(control_store) = hi_control::ControlStore::open_for_state(state_root) else {
        return;
    };
    let actor = hi_control::Principal {
        id: "local-process".into(),
        kind: "local_cli".into(),
    };
    let _ = control_store.record_audit(&hi_control::AuditRecord {
        audit_id: uuid::Uuid::new_v4().to_string(),
        decision: "folder_trust_disabled".into(),
        actor: actor.clone(),
        source: "HI_FOLDER_TRUST".into(),
        scope: None,
        provenance: Some(hi_control::Provenance {
            principal: actor,
            source: "operator_override".into(),
            run_id: None,
            attempt_id: None,
            parent_ref: None,
            correlation_id: None,
            policy_version: None,
        }),
        policy_snapshot: None,
        operation_digest: None,
        approval_id: None,
        route: None,
        effect_id: None,
        event_id: None,
        detail: Some(format!("workspace={}", workspace_root.display())),
        created_at_ms: hi_control::now_ms(),
    });
}
