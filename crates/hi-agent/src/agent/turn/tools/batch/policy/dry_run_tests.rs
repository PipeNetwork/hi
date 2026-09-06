use super::*;

#[test]
fn mutating_call_reports_path_and_mutation() {
    let msg = dry_run_message("edit", "src/main.rs", true);
    assert_eq!(
        msg,
        "[dry-run] would run `edit` on src/main.rs (mutating; not executed)"
    );
}

#[test]
fn read_only_call_reports_read_only() {
    let msg = dry_run_message("read", "src/main.rs", false);
    assert_eq!(
        msg,
        "[dry-run] would run `read` on src/main.rs (read-only; not executed)"
    );
}

#[test]
fn call_without_path_omits_target() {
    let msg = dry_run_message("bash", "", true);
    assert_eq!(msg, "[dry-run] would run `bash` (mutating; not executed)");
}

#[test]
fn sealed_mode_can_only_narrow_current_execution_authority() {
    assert!(
        execution_mode_denial(hi_ai::ToolMode::ReadOnly, hi_ai::ToolMode::Auto, "write").is_some()
    );
    assert!(
        execution_mode_denial(hi_ai::ToolMode::ChatOnly, hi_ai::ToolMode::Auto, "read").is_some()
    );
    assert!(
        execution_mode_denial(hi_ai::ToolMode::Auto, hi_ai::ToolMode::ReadOnly, "write").is_some()
    );
    assert!(
        execution_mode_denial(hi_ai::ToolMode::ReadOnly, hi_ai::ToolMode::Auto, "read").is_none()
    );
}

#[test]
fn sealed_call_limit_can_only_narrow_live_turn_budget() {
    assert_eq!(permitted_call_prefix(4, 8, 2), 2);
    assert_eq!(permitted_call_prefix(4, 1, 3), 1);
}
