use super::{
    ConfirmationRequest, classify_error, confirmation_capability, confirmation_for_egress_tool,
    egress_confirm_required, error_counts_as_model_issue, is_live_progress_status,
    redact_debug_text, subagent_activity_label, tool_label, user_facing_status,
    user_visible_tool_result, without_leading_warning_marker, write_private_debug_log,
};
use hi_ai::{ProviderError, ProviderErrorKind};

#[test]
fn auto_classifier_is_conservative() {
    for path in ["", ".", "(unknown)", "(multiple files)", "src/../config"] {
        assert!(
            !ConfirmationRequest::FileEdit {
                path: path.into(),
                diff: "+small edit\n".into(),
            }
            .safe_for_auto(),
            "ambiguous or unnormalized target must require approval: {path}"
        );
    }
    assert!(
        ConfirmationRequest::FileEdit {
            path: "src/lib.rs".into(),
            diff: "+fn ok() {}\n".into(),
        }
        .safe_for_auto()
    );
    assert!(
        !ConfirmationRequest::FileEdit {
            path: ".env".into(),
            diff: "+TOKEN=x\n".into(),
        }
        .safe_for_auto()
    );
    assert!(
        !ConfirmationRequest::ShellMutation {
            command: "npm install".into(),
            cwd: ".".into(),
        }
        .safe_for_auto()
    );
    let git_reset = ConfirmationRequest::ShellMutation {
        command: "git reset --hard HEAD".into(),
        cwd: ".".into(),
    };
    assert!(!git_reset.safe_for_auto());
    assert!(
        git_reset.details().contains("discards worktree files"),
        "{}",
        git_reset.details()
    );
    assert!(
        !ConfirmationRequest::AskUser {
            question: "which API?".into(),
            options: vec!["REST".into(), "gRPC".into()],
        }
        .safe_for_auto()
    );
    assert!(
        !ConfirmationRequest::External {
            tool: "browser_exec".into(),
            operation_arguments: serde_json::json!({
                "script": "goto https://example.com"
            }),
            summary: "goto https://example.com".into(),
            target: String::new(),
            mcp_grant: None,
        }
        .safe_for_auto()
    );
}

#[test]
fn egress_confirm_ladder_matches_ask_auto_always() {
    use crate::PermissionMode;
    assert!(!egress_confirm_required(
        PermissionMode::Always,
        false,
        "browser_exec"
    ));
    assert!(egress_confirm_required(
        PermissionMode::Auto,
        true,
        "browser_exec"
    ));
    assert!(egress_confirm_required(
        PermissionMode::Auto,
        true,
        "use_tool"
    ));
    assert!(!egress_confirm_required(
        PermissionMode::Auto,
        true,
        "web_fetch"
    ));
    assert!(egress_confirm_required(
        PermissionMode::Ask,
        true,
        "web_fetch"
    ));
    assert!(!egress_confirm_required(
        PermissionMode::Ask,
        true,
        "web_search"
    ));
}

#[test]
fn browser_exec_preview_flags_eval() {
    let request = confirmation_for_egress_tool(
        "browser_exec",
        r#"{"script":"goto https://example.com\neval document.cookie"}"#,
    );
    let crate::ConfirmationRequest::External {
        summary, mcp_grant, ..
    } = request
    else {
        panic!("expected External");
    };
    assert!(summary.contains("eval"));
    assert!(summary.contains("warning"));
    assert!(mcp_grant.is_none());
}

#[test]
fn use_tool_preview_offers_mcp_grant() {
    let request = confirmation_for_egress_tool(
        "use_tool",
        r#"{"server":"github","tool":"create_issue","arguments":{"title":"x"}}"#,
    );
    assert_eq!(
        request.mcp_standing_grant(),
        Some(("github", "create_issue"))
    );
}

#[test]
fn external_approval_digest_binds_suffix_beyond_bounded_preview() {
    let shared = "a".repeat(1_300);
    let browser_a = serde_json::json!({
        "script": format!("# {shared}\ngoto https://safe-a.example")
    })
    .to_string();
    let browser_b = serde_json::json!({
        "script": format!("# {shared}\ngoto https://safe-b.example")
    })
    .to_string();
    assert_eq!(&browser_a[..1_200], &browser_b[..1_200]);
    let browser_a = confirmation_for_egress_tool("browser_exec", &browser_a);
    let browser_b = confirmation_for_egress_tool("browser_exec", &browser_b);
    let (
        ConfirmationRequest::External {
            summary: preview_a, ..
        },
        ConfirmationRequest::External {
            summary: preview_b, ..
        },
    ) = (&browser_a, &browser_b)
    else {
        panic!("expected External confirmations");
    };
    assert_eq!(preview_a, preview_b, "display previews stay bounded");
    let (_, digest_a) = confirmation_capability(&browser_a).unwrap();
    let (_, digest_b) = confirmation_capability(&browser_b).unwrap();
    assert_ne!(digest_a, digest_b, "hidden suffix must remain bound");

    let mcp_a = serde_json::json!({
        "server": "mail",
        "tool": "send",
        "arguments": {"body": format!("{shared}recipient-a")}
    })
    .to_string();
    let mcp_b = serde_json::json!({
        "server": "mail",
        "tool": "send",
        "arguments": {"body": format!("{shared}recipient-b")}
    })
    .to_string();
    assert_eq!(&mcp_a[..1_200], &mcp_b[..1_200]);
    let mcp_a = confirmation_for_egress_tool("use_tool", &mcp_a);
    let mcp_b = confirmation_for_egress_tool("use_tool", &mcp_b);
    let (_, digest_a) = confirmation_capability(&mcp_a).unwrap();
    let (_, digest_b) = confirmation_capability(&mcp_b).unwrap();
    assert_ne!(digest_a, digest_b, "MCP suffix must remain bound");
}

#[test]
fn external_approval_digest_canonicalizes_json_arguments() {
    let first = confirmation_for_egress_tool(
        "use_tool",
        r#"{"server":"demo","tool":"echo","arguments":{"b":2,"a":1}}"#,
    );
    let reordered = confirmation_for_egress_tool(
        "use_tool",
        r#"{ "arguments": { "a": 1, "b": 2 }, "tool": "echo", "server": "demo" }"#,
    );
    let (_, first_digest) = confirmation_capability(&first).unwrap();
    let (_, reordered_digest) = confirmation_capability(&reordered).unwrap();
    assert_eq!(first_digest, reordered_digest);
}

#[test]
fn labels_file_tools_by_path() {
    // The bug this fixes: write/edit/read used to dump their whole JSON
    // (content and all) into the header. Show just the path instead.
    assert_eq!(
        tool_label(
            "write",
            r#"{"path":"checkers.rs","content":"use std::fmt;\n…"}"#
        ),
        "write checkers.rs"
    );
    assert_eq!(
        tool_label(
            "edit",
            r#"{"path":"src/cli.rs","old_string":"a","new_string":"b"}"#
        ),
        "edit src/cli.rs"
    );
    assert_eq!(
        tool_label("read", r#"{"path":"Cargo.toml"}"#),
        "read Cargo.toml"
    );
    // Multi-path reads: a one-element array still names the file.
    assert_eq!(
        tool_label("read", r#"{"paths":["Cargo.toml"]}"#),
        "read Cargo.toml"
    );
    // A multi-element array collapses to "N files".
    assert_eq!(
        tool_label("read", r#"{"paths":["a.rs","b.rs","c.rs"]}"#),
        "read 3 files"
    );
}

#[test]
fn labels_bash_by_command_and_grep_by_pattern() {
    assert_eq!(
        tool_label("bash", r#"{"command":"cargo  test\n  --all"}"#),
        "bash cargo test"
    );
    assert_eq!(
        tool_label(
            "bash",
            r#"{"command":"cd /Users/david/chat && cargo clippy --all-targets 2>&1 | grep warning"}"#
        ),
        "bash cargo clippy"
    );
    assert_eq!(
        tool_label("bash_output", r#"{"id":"sh_1"}"#),
        "bash_output sh_1"
    );
    assert_eq!(
        tool_label("grep", r#"{"pattern":"TODO","path":"src"}"#),
        "grep TODO in src"
    );
    assert_eq!(
        tool_label("grep", r#"{"pattern":"fn main"}"#),
        "grep fn main"
    );
    assert_eq!(tool_label("list", "{}"), "list .");
}

#[test]
fn labels_subagent_tools_by_task_not_json() {
    // Subagent calls used to dump raw JSON (prompt and all) into the
    // header: task({"cost":"large","description":"Build…","prompt":"Impl…).
    assert_eq!(
        tool_label(
            "task",
            r#"{"cost":"large","description":"Build workflow run store","prompt":"Implement durable…"}"#
        ),
        "task Build workflow run store"
    );
    assert_eq!(
        tool_label("explore", r#"{"task":"Investigate workflow flags"}"#),
        "explore Investigate workflow flags"
    );
    assert_eq!(
        tool_label("delegate", r#"{"task":"Wire the scheduler"}"#),
        "delegate Wire the scheduler"
    );
    assert_eq!(
        tool_label("get_task_output", r#"{"task_ids":["task_1","task_2"]}"#),
        "get_task_output task_1, task_2"
    );
    assert_eq!(
        tool_label(
            "ask_user",
            r#"{"question":"Which transport should the public API use?"}"#
        ),
        "ask_user Which transport should the public API use?"
    );
}

#[test]
fn internal_statuses_are_hidden_or_humanized() {
    assert!(
        user_facing_status("compat: deepseek profile=gateway protocol=auto strict=false").is_none()
    );
    assert!(user_facing_status("MoA aggregating: coder").is_none());
    assert!(user_facing_status("process_execution capability requested").is_none());
    assert!(user_facing_status("verification started").is_none());
    assert!(user_facing_status("verification finished").is_none());
    assert!(user_facing_status("verification skipped — no files changed this turn").is_none());
    assert!(user_facing_status("Run started").is_none());
    let leftover = user_facing_status("3/9 remaining — wire the scheduler").unwrap();
    assert_eq!(leftover, "3/9 remaining — wire the scheduler");
    assert_eq!(
        user_facing_status(
            "DeepSeek tool arguments failed client validation; retrying once without strict schemas"
        ),
        Some("retrying the tool call with a compatible schema".to_string())
    );
    assert_eq!(
        user_facing_status("⚠ the model returned no response after retrying — try /retry."),
        Some("⚠ no response after retries".to_string())
    );
}

#[test]
fn live_progress_statuses_are_replaceable() {
    assert!(is_live_progress_status("still working — checking"));
    assert!(is_live_progress_status("  still working — retrying"));
    assert!(!is_live_progress_status("working status: complete"));
}

#[test]
fn warning_marker_is_normalized_once() {
    assert_eq!(
        without_leading_warning_marker("⚠ already marked"),
        "already marked"
    );
    assert_eq!(
        without_leading_warning_marker("⚠️ already marked"),
        "already marked"
    );
    assert_eq!(
        without_leading_warning_marker("⚠️ ⚠ already marked"),
        "already marked"
    );
    assert_eq!(
        without_leading_warning_marker("plain notice"),
        "plain notice"
    );
}

#[test]
fn model_only_background_instructions_are_removed_from_display_results() {
    let result = user_visible_tool_result(
        "Started cargo test (sh_1). Use bash_output with id sh_1 for progress; Use bash_kill with id sh_1 to stop.",
    );
    assert_eq!(result, "Started cargo test (sh_1).");

    let missing = user_visible_tool_result(
        "Error: no background process `git-status_1` — no background processes are running at all. Do not call this again; continue the task with other tools.",
    );
    assert_eq!(missing, "background process git-status_1 unavailable");
}

#[test]
fn never_dumps_raw_json_into_labels() {
    // Unknown tools with JSON args: bare name only — no brace soup in the TUI.
    assert_eq!(tool_label("frobnicate", r#"{"x":  1}"#), "frobnicate");
    // Unparsable plain args still show a short plain note.
    assert_eq!(tool_label("write", "not json"), "write not json");
}

#[test]
fn capacity_limit_is_not_a_model_quality_issue() {
    let err: anyhow::Error = ProviderError::new(
        ProviderErrorKind::CapacityUnavailable,
        "API error 409: capacity temporarily unavailable",
    )
    .into();

    let (kind, guidance) = classify_error(&err);

    assert_eq!(kind, "capacity");
    assert!(guidance.contains("capacity is limited"));
    assert!(!error_counts_as_model_issue(&err));
}

#[test]
fn grok_credit_exhaustion_points_at_pipenetwork_not_a_dead_key() {
    let err: anyhow::Error = ProviderError::new(
        ProviderErrorKind::Auth,
        "API error 403 Forbidden: You have run out of credits or need a Grok subscription. Add credits at https://grok.com/?_s=usage",
    )
    .into();
    let (kind, guidance) = classify_error(&err);
    assert_eq!(kind, "auth");
    assert!(guidance.contains("/login pipenetwork"));
    assert!(guidance.contains("/provider pipenetwork"));
    assert!(!guidance.contains("API key may be invalid"));
}

#[test]
fn x402_payment_required_points_at_usdc_login() {
    let err: anyhow::Error = ProviderError::new(
        ProviderErrorKind::PaymentRequired,
        "x402 quote $1.20 exceeds HI_X402_MAX_USD $1.00",
    )
    .into();
    let (kind, guidance) = classify_error(&err);
    assert_eq!(kind, "payment");
    assert!(guidance.contains("/login x402"));
    assert!(guidance.contains("HI_X402_MAX_USD"));
    assert!(!error_counts_as_model_issue(&err));
}

#[test]
fn pipe_external_processing_disabled_explains_account_capability_not_key_failure() {
    let err: anyhow::Error = ProviderError::new(
        ProviderErrorKind::PolicyBlocked,
        "API error 403 Forbidden: external processing is disabled for this request",
    )
    .with_api_contract(
        Some("external_processing_disabled".into()),
        Some(false),
        None,
    )
    .into();

    let (kind, guidance) = classify_error(&err);

    assert_eq!(kind, "policy");
    assert!(guidance.contains("external processing is disabled"));
    assert!(guidance.contains("re-authentication will not change this"));
    assert!(!guidance.contains("API key may be invalid"));
}

#[test]
fn route_rejection_is_not_reported_as_capacity_or_incomplete_turn() {
    let err: anyhow::Error = ProviderError::new(
        ProviderErrorKind::ModelUnavailable,
        "model temporarily unavailable",
    )
    .into();

    let (kind, guidance) = classify_error(&err);

    assert_eq!(kind, "request");
    assert!(!guidance.contains("/model"));
    assert!(!guidance.contains("switch"));
    assert!(!guidance.contains("capacity"));
    assert!(!error_counts_as_model_issue(&err));
}

#[test]
fn explicitly_non_retryable_service_error_does_not_recommend_retrying() {
    let err: anyhow::Error = ProviderError::new(
        ProviderErrorKind::Outage,
        "API rejected the provider payload",
    )
    .with_api_contract(Some("service_unavailable".to_string()), Some(false), None)
    .into();
    let (kind, guidance) = classify_error(&err);
    assert_eq!(kind, "request");
    assert!(guidance.contains("will not succeed unchanged"));
}

#[test]
fn external_processing_policy_code_has_actionable_guidance() {
    let err: anyhow::Error = ProviderError::new(
        ProviderErrorKind::PolicyBlocked,
        "request rejected by account policy",
    )
    .with_api_contract(
        Some("external_processing_disabled".to_string()),
        Some(false),
        None,
    )
    .with_http_status(Some(403))
    .into();

    let (kind, guidance) = classify_error(&err);
    assert_eq!(kind, "policy");
    assert!(guidance.contains("external processing is disabled"));
    assert!(guidance.contains("re-authentication will not change this"));
    assert!(!guidance.contains("API key may be invalid"));
}

#[test]
fn soft_protocol_errors_are_not_model_quality_issues() {
    for (kind, expected_label) in [
        (ProviderErrorKind::QualityRejected, "quality"),
        (ProviderErrorKind::ToolProtocol, "tool_protocol"),
    ] {
        let err: anyhow::Error =
            ProviderError::new(kind, "model output did not satisfy the tool protocol").into();

        let (label, guidance) = classify_error(&err);

        assert_eq!(label, expected_label);
        assert!(!guidance.is_empty());
        assert!(
            !guidance.contains("/retry"),
            "model-glitch guidance must not ask the user to retry: {guidance}"
        );
        assert!(!error_counts_as_model_issue(&err));
    }
}

#[test]
fn empty_and_malformed_guidance_does_not_ask_the_user_to_retry() {
    for kind in [
        ProviderErrorKind::MalformedStream,
        ProviderErrorKind::EmptyCompletion,
        ProviderErrorKind::QualityRejected,
    ] {
        let err: anyhow::Error = ProviderError::new(kind, "glitch").into();
        let (_, guidance) = classify_error(&err);
        assert!(
            !guidance.contains("/retry"),
            "{kind:?} guidance must not ask the user to retry: {guidance}"
        );
        assert!(!guidance.is_empty());
    }
}

#[test]
fn debug_redaction_covers_known_and_structured_secrets() {
    let raw = "Authorization: Bearer abc\napi_key=abc\nlease_token: lease-123\npassword = hunter2\nplain ok";
    let clean = redact_debug_text(raw, &["abc", "lease-123"]);
    assert!(!clean.contains("abc"));
    assert!(!clean.contains("lease-123"));
    assert!(!clean.contains("hunter2"));
    assert!(clean.contains("plain ok"));
}

#[test]
fn debug_redaction_catches_bare_provider_key_shapes() {
    // No `key=`/`Bearer` label — just a credential sitting in output.
    let raw = "loaded key sk-ABCDEF0123456789abcdef for the run\n\
               token ghp_0123456789ABCDEFabcdef and AKIAIOSFODNN7EXAMPLE\n\
               nothing to see on this line";
    let clean = redact_debug_text(raw, &[]);
    assert!(
        !clean.contains("sk-ABCDEF0123456789abcdef"),
        "OpenAI-style: {clean}"
    );
    assert!(
        !clean.contains("ghp_0123456789ABCDEFabcdef"),
        "GitHub PAT: {clean}"
    );
    assert!(
        !clean.contains("AKIAIOSFODNN7EXAMPLE"),
        "AWS key id: {clean}"
    );
    assert!(clean.contains("[REDACTED]"));
    assert!(clean.contains("nothing to see on this line"));
    assert!(
        clean.contains("for the run"),
        "surrounding text preserved: {clean}"
    );
}

#[test]
fn debug_redaction_leaves_ordinary_hyphenated_words_alone() {
    // A word merely starting with a non-credential prefix, or a bare prefix,
    // must not be redacted — only real key-length tokens are.
    let raw = "the well-known sky-blue value and pk- placeholder";
    let clean = redact_debug_text(raw, &[]);
    assert_eq!(clean, raw, "no false positives: {clean}");
}

#[cfg(unix)]
#[test]
fn private_debug_log_is_atomic_and_owner_only() {
    use std::os::unix::fs::PermissionsExt;
    let path = std::env::temp_dir().join(format!(
        "hi-debug-{}-{}.log",
        std::process::id(),
        std::thread::current().name().unwrap_or("test")
    ));
    write_private_debug_log(&path, "first").unwrap();
    write_private_debug_log(&path, "second").unwrap();
    assert_eq!(std::fs::read_to_string(&path).unwrap(), "second");
    assert_eq!(
        std::fs::metadata(&path).unwrap().permissions().mode() & 0o777,
        0o600
    );
    let _ = std::fs::remove_file(path);
}

#[test]
fn subagent_activity_label_maps_read_and_bash() {
    assert_eq!(
        subagent_activity_label("read", r#"{"path":"lib.rs"}"#),
        "Reading lib.rs"
    );
    assert_eq!(
        subagent_activity_label("explore:read", r#"{"path":"lib.rs"}"#),
        "Reading lib.rs"
    );
    let bash = subagent_activity_label("bash", r#"{"command":"cargo test"}"#);
    assert!(bash.starts_with("Run "), "got {bash}");
}

#[test]
fn user_prompt_title_strips_context_and_keeps_the_real_prompt() {
    let dumped = "[hi:context — session state, not instructions]\n\
# Memory (from past sessions; task-ranked)\n\
Prefer bullets that match the current task.\n\
[/hi:context]\n\n\
fix the parser";
    assert_eq!(super::user_prompt_title(dumped, 72), "fix the parser");
    assert_eq!(
        super::user_prompt_title("[hi:context — session state, not instructions] no end", 72),
        ""
    );
    let planned = "[hi:turn-control — current turn only]\n\
Plan mode is ON for this turn.\n[/hi:turn-control]\n\n\
User request:\nbuild profiles";
    assert_eq!(super::user_prompt_title(planned, 72), "build profiles");
    let executing = "[hi:turn-control — current turn only]\n\
Plan mode is OFF for this turn.\n[/hi:turn-control]\n\n\
build all of that\n\n[hi:turn-control — current turn only]\n\
Implementation guard: edit files.\n[/hi:turn-control]";
    assert_eq!(super::user_prompt_title(executing, 72), "build all of that");
    let historical = "review code\n\nRead-only review guard: shell execution (`bash`) and mutation tools are unavailable for this review. Use only the advertised read-only inspection tools; inspect carefully. If only a directory listing is available, keep inspecting before making file-specific findings.";
    assert_eq!(super::user_prompt_title(historical, 72), "review code");
    let mixed = "[hi:turn-control — current turn only]\nPlan mode is OFF for this turn.\n[/hi:turn-control]\n\nreview code\n\nRead-only review guard: do not write, edit, apply patches, run mutating shell commands, or change files. Use read-only inspection before the final answer. If only a directory listing is available, keep inspecting or explicitly say the evidence is insufficient instead of making file-specific findings.";
    assert_eq!(super::user_prompt_title(mixed, 72), "review code");
    let mixed_plan = "[hi:turn-control — current turn only]\nPlan mode is OFF for this turn.\n[/hi:turn-control]\n\nYou are in PLAN MODE. Do not modify files.\n\nProduce a plan.\n\nUser request:\nbuild profiles\n\nRead-only review guard: do not write, edit, apply patches, run mutating shell commands, or change files. Use read-only inspection before the final answer.";
    assert_eq!(super::user_prompt_title(mixed_plan, 72), "build profiles");
    let quoted = "Document this historical prefix exactly:\n\nRead-only review guard: do not write, edit, apply patches, run mutating shell commands, or change files. Use read-only inspection before the final answer. This following sentence is still user-authored.";
    assert_eq!(
        super::user_prompt_title(quoted, 300),
        super::collapse_ws(quoted)
    );
    let exact_quote = "Document this historical guard exactly:\n\nRead-only review guard: do not write, edit, apply patches, run mutating shell commands, or change files. Use read-only inspection before the final answer. If only a directory listing is available, keep inspecting or explicitly say the evidence is insufficient instead of making file-specific findings.";
    assert_eq!(
        super::user_prompt_title(exact_quote, 400),
        super::collapse_ws(exact_quote)
    );
    assert_eq!(super::user_prompt_title("plain prompt", 72), "plain prompt");
}
