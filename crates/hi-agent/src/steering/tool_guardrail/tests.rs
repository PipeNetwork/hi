use super::*;

#[test]
fn distinct_idempotent_results_use_bounded_fifo_repeat_memory() {
    let mut guard = ToolLoopGuardrail::default();
    for index in 0..5_000 {
        let result = guard.record_tool_result(
            "read",
            r#"{"path":"src/lib.rs"}"#,
            &format!("unique output {index}"),
        );
        assert!(!result.repeated_idempotent_result);
    }

    assert_eq!(
        guard.seen_idempotent_result_hashes.len(),
        IDEMPOTENT_RESULT_HASH_LIMIT
    );
    assert_eq!(
        guard.seen_idempotent_result_order.len(),
        IDEMPOTENT_RESULT_HASH_LIMIT
    );
    assert_eq!(guard.evicted_idempotent_result_hashes, 904);
    assert!(
        guard
            .record_tool_result("read", r#"{"path":"src/lib.rs"}"#, "unique output 4999",)
            .repeated_idempotent_result
    );
    assert!(
        !guard
            .record_tool_result("read", r#"{"path":"src/lib.rs"}"#, "unique output 0",)
            .repeated_idempotent_result,
        "evicting an ancient repeat only weakens the loop heuristic; it must not stop work"
    );
}

#[test]
fn repeated_read_result_is_no_progress_even_with_different_args() {
    let mut guard = ToolLoopGuardrail::default();

    let first = guard.record_tool_result("read", r#"{"path":"a.rs"}"#, "same output");
    let second = guard.record_tool_result("read", r#"{"path":"b.rs"}"#, "same output");

    assert!(first.hashable_idempotent);
    assert!(!first.repeated_idempotent_result);
    assert!(second.hashable_idempotent);
    assert!(second.repeated_idempotent_result);
}

#[test]
fn wait_poll_and_validation_bash_are_hash_guarded_but_plain_bash_is_not() {
    let mut guard = ToolLoopGuardrail::default();
    let wait_args = r#"{"command":"sleep 300 && du -sh models/"}"#;

    let first = guard.record_tool_result("bash", wait_args, "90G\t18 shards");
    assert!(first.hashable_idempotent);
    assert!(!first.repeated_idempotent_result);

    let progressed = guard.record_tool_result("bash", wait_args, "124G\t27 shards");
    assert!(progressed.hashable_idempotent);
    assert!(
        !progressed.repeated_idempotent_result,
        "changing output is progress"
    );

    let static_poll = guard.record_tool_result("bash", wait_args, "124G\t27 shards");
    assert!(
        static_poll.repeated_idempotent_result,
        "identical output means the awaited state stopped changing"
    );

    let validation = guard.record_tool_result("bash", r#"{"command":"cargo test"}"#, "1 passed");
    assert!(validation.hashable_idempotent);
    assert!(!validation.repeated_idempotent_result);

    let plain = guard.record_tool_result("bash", r#"{"command":"echo hi"}"#, "hi");
    assert!(!plain.hashable_idempotent, "plain bash is not hash guarded");
}

#[test]
fn varied_bounded_launches_with_the_same_result_are_deduplicated() {
    let mut guard = ToolLoopGuardrail::default();
    let head = r#"{"command":"timeout 10 ./target/debug/app 2>&1 | head -30; echo exit=$?"}"#;
    let tail = r#"{"command":"timeout 20 ./target/debug/app 2>&1 | tail -30; echo exit=$?"}"#;
    let output = "Seeded 20 agents (day 1).\nexit=0";

    let first = guard.record_tool_result_with_effects("bash", head, output, false);
    let repeated = guard.record_tool_result_with_effects("bash", tail, output, false);
    assert!(first.hashable_idempotent);
    assert!(!first.repeated_idempotent_result);
    assert!(repeated.repeated_idempotent_result);

    let mut mutation_guard = ToolLoopGuardrail::default();
    let mutation = mutation_guard.record_tool_result_with_effects("bash", head, output, true);
    assert!(!mutation.hashable_idempotent);
}

#[test]
fn varied_clippy_presentation_queries_with_empty_results_are_deduplicated() {
    let mut guard = ToolLoopGuardrail::default();
    let first = r#"{"command":"cd /Users/david/alovewtf && cargo clippy 2>&1 | grep -B3 -A12 \"src/main.rs:1605\" | head -40"}"#;
    let second = r#"{"command":"cd /Users/david/alovewtf && cargo clippy 2>&1 | grep -B3 -A12 \"src/main.rs:1606\" | head -40"}"#;

    let initial = guard.record_tool_result_with_effects("bash", first, "[no output]", false);
    let repeated = guard.record_tool_result_with_effects("bash", second, "[no output]", false);

    assert!(initial.hashable_idempotent);
    assert!(!initial.repeated_idempotent_result);
    assert!(repeated.hashable_idempotent);
    assert!(repeated.repeated_idempotent_result);
}

#[test]
fn validation_scope_keeps_workspace_context_and_deduplicates_failures() {
    let mut guard = ToolLoopGuardrail::default();
    let api = r#"{"command":"cd crates/api && cargo clippy | grep first"}"#;
    let api_again = r#"{"command":"cd crates/api && cargo clippy | grep second"}"#;
    let web = r#"{"command":"cd crates/web && cargo clippy | grep second"}"#;

    let first = guard.record_tool_result_with_effects("bash", api, "Error: failed", false);
    let repeated = guard.record_tool_result_with_effects("bash", api_again, "Error: failed", false);
    let distinct = guard.record_tool_result_with_effects("bash", web, "Error: failed", false);

    assert!(first.hashable_idempotent);
    assert!(repeated.repeated_idempotent_result);
    assert!(!distinct.repeated_idempotent_result);
}

#[test]
fn filtered_validator_exit_status_is_not_false_green() {
    assert!(validation_exit_status_is_reliable(
        "bash",
        r#"{"command":"cargo clippy --workspace"}"#,
    ));
    assert!(!validation_exit_status_is_reliable(
        "bash",
        r#"{"command":"cargo clippy 2>&1 | grep warning | head -40"}"#,
    ));
    assert!(!validation_exit_status_is_reliable(
        "bash",
        r#"{"command":"set -o pipefail; cargo clippy | tee clippy.log"}"#,
    ));
    assert!(validation_exit_status_is_reliable(
        "bash",
        r#"{"command":"cargo clippy && echo checked"}"#,
    ));
    assert!(!validation_exit_status_is_reliable(
        "bash",
        r#"{"command":"cargo clippy; echo checked"}"#,
    ));
    assert!(!validation_exit_status_is_reliable(
        "bash",
        r#"{"command":"cargo clippy || echo ignored"}"#,
    ));
    for command in [
        "true # cargo clippy",
        "printf '%s' 'cargo clippy'",
        "true || cargo clippy",
        "! cargo clippy",
        "echo \"$(cargo clippy)\"",
        "echo pipefail; cargo clippy | head",
        "set +o pipefail; cargo clippy | head",
    ] {
        let arguments = serde_json::json!({ "command": command }).to_string();
        assert!(
            !validation_exit_status_is_reliable("bash", &arguments),
            "must fail closed when the validator status is ambiguous: {command}"
        );
    }
    for command in [
        "cd crate && cargo clippy",
        "FOO=x cargo clippy",
        "env FOO=x cargo clippy",
        "cargo clippy 2>&1",
        "cargo clippy --message-format='short|json'",
    ] {
        let arguments = serde_json::json!({ "command": command }).to_string();
        assert!(
            validation_exit_status_is_reliable("bash", &arguments),
            "direct validator status should remain usable: {command}"
        );
    }
}

#[test]
fn landed_mutation_re_admits_validation_observation() {
    let mut guard = ToolLoopGuardrail::default();
    let validation = r#"{"command":"cargo clippy --package api"}"#;

    assert!(
        !guard
            .record_tool_result_with_effects("bash", validation, "clean", false)
            .repeated_idempotent_result
    );
    assert!(
        guard
            .record_tool_result_with_effects("bash", validation, "clean", false)
            .repeated_idempotent_result
    );
    guard.record_tool_result_with_effects(
        "edit",
        r#"{"path":"src/lib.rs"}"#,
        "Edited src/lib.rs",
        true,
    );
    assert!(
        !guard
            .record_tool_result_with_effects("bash", validation, "clean", false)
            .repeated_idempotent_result,
        "a landed mutation starts a new validation epoch"
    );
}

#[test]
fn context_result_repeats_are_scoped_to_workspace_revision_and_effects() {
    for (name, arguments) in [
        ("read", r#"{"path":"context.rs"}"#),
        ("grep", r#"{"pattern":"caller"}"#),
        ("list", r#"{"path":"src"}"#),
        ("bash", r#"{"command":"sed -n '1,20p' context.rs"}"#),
    ] {
        let mut guard = ToolLoopGuardrail::default();
        guard.observe_workspace_revision(7);
        assert!(
            !guard
                .record_tool_result(name, arguments, "same context")
                .repeated_idempotent_result
        );
        assert!(
            guard
                .record_tool_result(name, arguments, "same context")
                .repeated_idempotent_result
        );
        guard.observe_workspace_revision(8);
        assert!(
            !guard
                .record_tool_result(name, arguments, "same context")
                .repeated_idempotent_result
        );
        assert!(
            guard
                .record_tool_result(name, arguments, "same context")
                .repeated_idempotent_result
        );
        guard.record_tool_result_with_effects("delegate", "{}", "edited", true);
        assert!(
            !guard
                .record_tool_result(name, arguments, "same context")
                .repeated_idempotent_result
        );
    }
}

#[test]
fn authoritative_workspace_revision_re_admits_validation_observation() {
    let mut guard = ToolLoopGuardrail::default();
    let validation = r#"{"command":"cargo clippy --package api"}"#;
    guard.observe_workspace_revision(7);
    assert!(
        !guard
            .record_tool_result_with_effects("bash", validation, "clean", false)
            .repeated_idempotent_result
    );
    assert!(
        guard
            .record_tool_result_with_effects("bash", validation, "clean", false)
            .repeated_idempotent_result
    );
    guard.observe_workspace_revision(8);
    assert!(
        !guard
            .record_tool_result_with_effects("bash", validation, "clean", false)
            .repeated_idempotent_result,
        "candidate/background reconciliation must re-admit current validation"
    );
}

#[test]
fn distinct_validation_scopes_do_not_share_results() {
    let mut guard = ToolLoopGuardrail::default();
    let api = r#"{"command":"cargo clippy --package api --target aarch64-apple-darwin"}"#;
    let web = r#"{"command":"cargo clippy --package web --target wasm32-unknown-unknown"}"#;

    let first = guard.record_tool_result_with_effects("bash", api, "clean", false);
    let distinct = guard.record_tool_result_with_effects("bash", web, "clean", false);

    assert!(!first.repeated_idempotent_result);
    assert!(!distinct.repeated_idempotent_result);
}

#[test]
fn validation_scope_normalizes_spacing_aliases_and_presentation_flags() {
    let mut guard = ToolLoopGuardrail::default();
    let first = r#"{"command":"cargo clippy -q --color always --message-format short -p api --target wasm32-unknown-unknown | grep first"}"#;
    let cosmetic_variant = r#"{"command":"true && cargo    clippy --target=wasm32-unknown-unknown --package=api --color=never | head -40"}"#;

    assert!(
        !guard
            .record_tool_result_with_effects("bash", first, "clean", false)
            .repeated_idempotent_result
    );
    assert!(
        guard
            .record_tool_result_with_effects("bash", cosmetic_variant, "clean", false)
            .repeated_idempotent_result
    );
}

#[test]
fn direct_script_validation_is_semantically_guarded() {
    let mut guard = ToolLoopGuardrail::default();
    let first = r#"{"command":"python3 check.py | grep first"}"#;
    let variant = r#"{"command":"python3 check.py | grep second"}"#;

    assert!(
        !guard
            .record_tool_result_with_effects("bash", first, "ok", false)
            .repeated_idempotent_result
    );
    assert!(
        guard
            .record_tool_result_with_effects("bash", variant, "ok", false)
            .repeated_idempotent_result
    );
}

#[test]
fn alternating_shell_inspections_cannot_evade_result_deduplication() {
    let mut guard = ToolLoopGuardrail::default();
    let page_a = r#"{"command":"for f in blog_posts/txt/*.txt; do sed -n '1,100p' \"$f\"; done"}"#;
    let page_b =
        r#"{"command":"for f in blog_posts/txt/*.txt; do sed -n '100,150p' \"$f\"; done"}"#;

    let first_a = guard.record_tool_result_with_effects("bash", page_a, "page A", false);
    let first_b = guard.record_tool_result_with_effects("bash", page_b, "page B", false);
    let repeated_a = guard.record_tool_result_with_effects("bash", page_a, "page A", false);

    assert!(first_a.hashable_idempotent);
    assert!(!first_a.repeated_idempotent_result);
    assert!(first_b.hashable_idempotent);
    assert!(!first_b.repeated_idempotent_result);
    assert!(repeated_a.repeated_idempotent_result);

    let same_output_different_page =
        guard.record_tool_result_with_effects("bash", page_b, "page A", false);
    assert!(
        !same_output_different_page.repeated_idempotent_result,
        "a distinct inspection page is new evidence even when its text matches"
    );

    let mutation = guard.record_tool_result_with_effects("bash", page_a, "page A", true);
    assert!(!mutation.hashable_idempotent);
}

#[test]
fn different_wait_polls_with_identical_output_are_distinct_events() {
    // Health checks of two different servers both printing "ready: True"
    // must not read as a static state — the key covers the arguments.
    let mut guard = ToolLoopGuardrail::default();
    let first = guard.record_tool_result(
        "bash",
        r#"{"command":"sleep 30 && curl -fsS http://127.0.0.1:18101/health"}"#,
        "ready: True",
    );
    let second = guard.record_tool_result(
        "bash",
        r#"{"command":"sleep 30 && curl -fsS http://127.0.0.1:18102/health"}"#,
        "ready: True",
    );
    assert!(!first.repeated_idempotent_result);
    assert!(
        !second.repeated_idempotent_result,
        "a different poll is a different event even with identical output"
    );

    let same_again = guard.record_tool_result(
        "bash",
        r#"{"command":"sleep 30 && curl -fsS http://127.0.0.1:18102/health"}"#,
        "ready: True",
    );
    assert!(
        same_again.repeated_idempotent_result,
        "the same poll repeating its own output is static"
    );
}

#[test]
fn idle_bash_output_allows_two_polls_then_flags_tight_loop() {
    let mut guard = ToolLoopGuardrail::default();
    let args = r#"{"id":"sh_1"}"#;
    let idle = "[sh_1: still running — no new output]";

    let first = guard.record_tool_result("bash_output", args, idle);
    let second = guard.record_tool_result("bash_output", args, idle);
    let third = guard.record_tool_result("bash_output", args, idle);

    assert!(first.idle_background_poll && !first.repeated_idempotent_result);
    assert!(second.idle_background_poll && !second.repeated_idempotent_result);
    assert!(
        third.idle_background_poll && third.repeated_idempotent_result,
        "third consecutive idle poll is a tight loop"
    );

    let other = guard.record_tool_result(
        "bash_output",
        r#"{"id":"sh_2"}"#,
        "[sh_2: still running — no new output]",
    );
    assert!(
        !other.repeated_idempotent_result,
        "a different handle starts a fresh idle streak"
    );
}

#[test]
fn fresh_bash_output_resets_idle_streak() {
    let mut guard = ToolLoopGuardrail::default();
    let args = r#"{"id":"sh_1"}"#;
    let idle = "[sh_1: still running — no new output]";

    assert!(
        !guard
            .record_tool_result("bash_output", args, idle)
            .repeated_idempotent_result
    );
    assert!(
        !guard
            .record_tool_result("bash_output", args, idle)
            .repeated_idempotent_result
    );

    let progressed =
        guard.record_tool_result("bash_output", args, "[sh_1: still running]\n== hi-ai ==\n");
    assert!(!progressed.idle_background_poll);
    assert!(!progressed.repeated_idempotent_result);

    // After progress, two more idle polls are allowed again.
    assert!(
        !guard
            .record_tool_result("bash_output", args, idle)
            .repeated_idempotent_result
    );
    assert!(
        !guard
            .record_tool_result("bash_output", args, idle)
            .repeated_idempotent_result
    );
    assert!(
        guard
            .record_tool_result("bash_output", args, idle)
            .repeated_idempotent_result
    );
}

#[test]
fn running_polls_are_flagged_regardless_of_output_novelty() {
    let mut guard = ToolLoopGuardrail::default();
    let args = r#"{"id":"sh_1"}"#;

    // A progress bar delivers fresh bytes on every poll: not idle, but
    // still a poll of a running process — the waiting classifier keys on
    // this, not on output novelty.
    let progressing = guard.record_tool_result(
        "bash_output",
        args,
        "[sh_1: still running]\n42.1 GiB / 767.7 GiB",
    );
    assert!(progressing.running_background_poll);
    assert!(!progressing.idle_background_poll);

    let idle =
        guard.record_tool_result("bash_output", args, "[sh_1: still running — no new output]");
    assert!(idle.running_background_poll && idle.idle_background_poll);

    let exited = guard.record_tool_result("bash_output", args, "[sh_1: exited with code 0]\ndone");
    assert!(!exited.running_background_poll);

    let errored = guard.record_tool_result("bash_output", args, "Error: no background process");
    assert!(!errored.running_background_poll);
}

#[test]
fn mutating_tools_are_not_hash_guarded() {
    let mut guard = ToolLoopGuardrail::default();

    let first = guard.record_tool_result("write", r#"{"path":"a.rs"}"#, "Wrote a.rs");
    let second = guard.record_tool_result("write", r#"{"path":"b.rs"}"#, "Wrote a.rs");

    assert!(!first.hashable_idempotent);
    assert!(!second.repeated_idempotent_result);
}

#[test]
fn error_bearing_running_poll_is_actionable_but_progress_noise_is_not() {
    // The incident this pins: a 600s poll finally surfaced a compile
    // error, and the wait-streak escalation forced a tool-free final
    // answer anyway. Diagnostics in fresh output are work, not waiting.
    let mut guard = ToolLoopGuardrail::default();
    let args = r#"{"id":"cargo-check_1"}"#;
    let noise = guard.record_tool_result(
        "bash_output",
        args,
        "[cargo-check_1 \u{b7} cargo check: still running]\n42.1 GiB / 767.7 GiB",
    );
    assert!(noise.running_background_poll);
    assert!(
        !noise.actionable_background_output,
        "progress noise is not actionable"
    );
    let diag = guard.record_tool_result(
        "bash_output",
        args,
        "[cargo-check_1 \u{b7} cargo check: still running]\nerror[E0107]: enum takes 2 \
             generic arguments but 1 generic argument was supplied",
    );
    assert!(diag.running_background_poll);
    assert!(
        diag.actionable_background_output,
        "compiler errors are actionable"
    );
    // A terminal poll is not a running poll, so the flag stays off.
    let exited = guard.record_tool_result(
        "bash_output",
        args,
        "[cargo-check_1: exited code 101]\nerror: could not compile",
    );
    assert!(!exited.running_background_poll);
    assert!(!exited.actionable_background_output);
}
