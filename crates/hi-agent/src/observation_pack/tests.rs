use super::*;
use hi_ai::Message;

fn tool_call(id: &str, name: &str) -> Message {
    Message {
        role: Role::Assistant,
        content: vec![Content::ToolCall {
            id: id.into(),
            name: name.into(),
            arguments: "{}".into(),
        }],
    }
}

fn tool_result(id: &str, output: impl Into<String>) -> Message {
    Message {
        role: Role::Tool,
        content: vec![Content::ToolResult {
            call_id: id.into(),
            output: output.into(),
        }],
    }
}

fn assistant_text(text: &str) -> Message {
    Message {
        role: Role::Assistant,
        content: vec![Content::Text(text.into())],
    }
}

fn packer(dir: &tempfile::TempDir) -> ObservationPack {
    ObservationPack::with_config(
        dir.path(),
        ObservationPackConfig {
            threshold_bytes: 64,
            full_sends: 2,
            excerpt_bytes: 32,
        },
    )
}

fn large_body() -> String {
    format!("{}\n{}", "alpha-line\n".repeat(20), "omega-line")
}

#[test]
fn obs_recall_is_inject_only() {
    assert!(
        !hi_tools::TOOL_SPECS
            .iter()
            .any(|tool| tool.name == "obs_recall"),
        "obs_recall must not join the always-on catalog"
    );
    assert_eq!(hi_tools::obs_recall_tool_spec().name, "obs_recall");
    assert!(hi_tools::is_read_only("obs_recall"));
    assert!(!hi_tools::is_filesystem_mutating("obs_recall"));
}

#[test]
fn full_sends_then_handle_and_exact_recall() {
    let dir = tempfile::tempdir().unwrap();
    let mut pack = packer(&dir);
    let body = large_body();
    let mut messages = vec![
        Message::user("task"),
        tool_call("c1", "bash"),
        tool_result("c1", body.clone()),
    ];

    let first = pack.project(&messages);
    assert!(
        tool_result_text(&first[2]).unwrap().contains("alpha-line"),
        "first send keeps the full body"
    );
    assert!(!tool_result_text(&first[2]).unwrap().contains("id: obs_"));

    messages.push(assistant_text("saw it"));
    let second = pack.project(&messages);
    assert!(
        tool_result_text(&second[2]).unwrap().contains("alpha-line"),
        "second send still full"
    );

    messages.push(assistant_text("again"));
    let third = pack.project(&messages);
    let packed = tool_result_text(&third[2]).unwrap();
    assert!(
        packed.contains("id: obs_"),
        "later projection carries a handle: {packed}"
    );
    assert!(
        !packed.contains(&body),
        "later projection must not replay the full body"
    );
    assert!(
        messages[2]
            .content
            .iter()
            .any(|block| matches!(block, Content::ToolResult { output, .. } if output == &body)),
        "stored transcript keeps the original"
    );

    let id = packed
        .lines()
        .find_map(|line| line.strip_prefix("id: "))
        .expect("placeholder id");
    let head = pack.recall(id, 0).expect("recall offset 0");
    assert!(
        body.as_bytes()[..head.bytes] == head.text.as_bytes()[..head.bytes]
            || body.starts_with(&head.text),
        "offset 0 must match archive bytes"
    );
    assert_eq!(&body.as_bytes()[..head.bytes], head.text.as_bytes());
    let mid = pack.recall(id, head.next_offset).expect("nonzero offset");
    assert_eq!(
        &body.as_bytes()[head.next_offset..head.next_offset + mid.bytes],
        mid.text.as_bytes()
    );
    assert!(pack.has_packed_handles());
}

#[test]
fn archive_failure_keeps_original() {
    let dir = tempfile::tempdir().unwrap();
    let blocker = dir.path().join("observation-pack");
    std::fs::write(&blocker, "not a directory").unwrap();
    let mut pack = packer(&dir);
    let body = large_body();
    let messages = vec![
        Message::user("task"),
        tool_call("c1", "bash"),
        tool_result("c1", body.clone()),
        assistant_text("one"),
        assistant_text("two"),
    ];
    let projected = pack.project(&messages);
    assert_eq!(tool_result_text(&projected[2]).unwrap(), body);
    assert!(!pack.has_packed_handles());
}

#[test]
fn does_not_pack_verify_digest_or_condensed_or_receipt() {
    let dir = tempfile::tempdir().unwrap();
    let mut pack = packer(&dir);
    let digest = format!("{VERIFY_DIGEST_MARK}\n{}", "error line\n".repeat(40));
    let condensed = format!(
        "head\n… 80{CONDENSE_OMISSION_MARK}…\n{}",
        "tail\n".repeat(20)
    );
    let receipt = format!(
        "{EVIDENCE_RECEIPT_PREFIX}\nstatus=failure\n{}",
        "quote\n".repeat(40)
    );
    for (id, body) in [("d", digest), ("c", condensed), ("r", receipt)] {
        let messages = vec![
            Message::user("task"),
            tool_call(id, "bash"),
            tool_result(id, body.clone()),
            assistant_text("one"),
            assistant_text("two"),
        ];
        let projected = pack.project(&messages);
        assert_eq!(
            tool_result_text(&projected[2]).unwrap(),
            body,
            "protected evidence {id} must not be packed"
        );
    }
    assert!(!pack.has_packed_handles());
}

#[test]
fn does_not_pack_unresolved_compiler_or_fused_failures() {
    let dir = tempfile::tempdir().unwrap();
    let mut pack = packer(&dir);
    let rustc = format!(
        "error[E0425]: cannot find function `register` in this scope\n{}",
        "ok noise\n".repeat(40)
    );
    let fused = format!(
        "{}\nerror[E0425]: cannot find function `register`\n{}",
        hi_tools::FUSED_COMMAND_FAILED,
        "ok noise\n".repeat(40)
    );
    let failed_tests = format!(
        "running 3 tests\ntest result: FAILED. 1 passed; 1 failed\n{}",
        "ok noise\n".repeat(40)
    );
    let gcc = format!(
        "src/web.rs:4:9: error: cannot find function register\n{}",
        "note: noise\n".repeat(40)
    );
    let panic = format!(
        "thread 'tests::it_breaks' panicked at src/lib.rs:4:1:\nassertion failed\n{}",
        "ok noise\n".repeat(40)
    );
    let pytest = format!(
        "===== FAILURES =====\nAssertionError: mismatch\n{}",
        "ok noise\n".repeat(40)
    );
    let go = format!("FAIL: TestRegister\n{}", "ok noise\n".repeat(40));
    let go_summary = format!(
        "--- FAIL: TestRegister (0.00s)\nFAIL\nFAIL\texample.com/pkg\t0.001s\n{}",
        "ok noise\n".repeat(40)
    );
    let tsc = format!(
        "src/web.ts(4,9): error TS2322: Type 'string' is not assignable\n{}",
        "ok noise\n".repeat(40)
    );
    let rustc_summary = format!(
        "error: could not compile `chat` (bin \"chat\") due to 1 previous error\n{}",
        "ok noise\n".repeat(40)
    );
    let pytest_header = format!(
        "===== FAILURES =====\ntest_login failed\n{}",
        "ok noise\n".repeat(40)
    );
    let jest = format!("FAIL src/web.test.js\n{}", "ok noise\n".repeat(40));
    let cargo_one = format!(
        "test tests::it_breaks ... FAILED\n{}",
        "ok noise\n".repeat(40)
    );
    for (id, body) in [
        ("rustc", rustc),
        ("fused", fused),
        ("tests", failed_tests),
        ("gcc", gcc),
        ("panic", panic),
        ("pytest", pytest),
        ("go", go),
        ("go_summary", go_summary),
        ("tsc", tsc),
        ("rustc_summary", rustc_summary),
        ("pytest_header", pytest_header),
        ("jest", jest),
        ("cargo_one", cargo_one),
    ] {
        let messages = vec![
            Message::user("task"),
            tool_call(id, "bash"),
            tool_result(id, body.clone()),
            assistant_text("one"),
            assistant_text("two"),
        ];
        let projected = pack.project(&messages);
        assert_eq!(
            tool_result_text(&projected[2]).unwrap(),
            body,
            "unresolved failure {id} must stay in the request"
        );
    }
    assert!(!pack.has_packed_handles());
}

#[test]
fn discovery_results_are_never_packed() {
    let dir = tempfile::tempdir().unwrap();
    let mut pack = packer(&dir);
    let body = format!("{}\n{}", "hit-line\n".repeat(40), "tail-line");
    for name in ["read", "grep", "glob", "list", "find_symbol", "repo_map"] {
        let messages = vec![
            Message::user("fix the account button"),
            tool_call(name, name),
            tool_result(name, body.clone()),
            assistant_text("one"),
            assistant_text("two"),
            assistant_text("three"),
        ];
        let projected = pack.project(&messages);
        assert_eq!(
            tool_result_text(&projected[2]).unwrap(),
            body,
            "{name} must stay in the request; packing it causes obs_recall stalls"
        );
    }
    assert!(!pack.has_packed_handles());
}

#[test]
fn unresolved_failure_lines_cover_go_jest_and_rustc_summaries() {
    assert!(looks_like_unresolved_failure(
        "--- FAIL: TestRegister (0.00s)\n"
    ));
    assert!(looks_like_unresolved_failure(
        "FAIL\nFAIL\texample.com/pkg\t0.001s\n"
    ));
    assert!(looks_like_unresolved_failure("FAIL src/web.test.js\n"));
    assert!(looks_like_unresolved_failure(
        "error: could not compile `chat` (bin \"chat\") due to 1 previous error\n"
    ));
    assert!(looks_like_unresolved_failure("===== FAILURES =====\n"));
    assert!(looks_like_unresolved_failure(
        "test tests::it_breaks ... FAILED\n"
    ));
    assert!(!looks_like_unresolved_failure(
        "running 3 tests\ntest result: ok. 3 passed; 0 failed\n"
    ));
}

#[test]
fn large_green_test_log_is_still_packed() {
    let dir = tempfile::tempdir().unwrap();
    let mut pack = packer(&dir);
    let body = format!(
        "running 3 tests\ntest result: ok. 3 passed; 0 failed\n{}",
        "ok noise\n".repeat(40)
    );
    let mut messages = vec![
        Message::user("task"),
        tool_call("g", "bash"),
        tool_result("g", body.clone()),
    ];
    let _ = pack.project(&messages);
    messages.push(assistant_text("one"));
    let _ = pack.project(&messages);
    messages.push(assistant_text("two"));
    let projected = pack.project(&messages);
    let packed = tool_result_text(&projected[2]).unwrap();
    assert!(
        packed.contains("id: obs_"),
        "passing logs remain packable: {packed}"
    );
    assert!(pack.has_packed_handles());
}

#[test]
fn recall_does_not_split_utf8_at_the_byte_budget() {
    let dir = tempfile::tempdir().unwrap();
    let mut pack = ObservationPack::with_config(
        dir.path(),
        ObservationPackConfig {
            threshold_bytes: 1,
            full_sends: 0,
            excerpt_bytes: 8,
        },
    );
    let prefix = "a".repeat(RECALL_MAX_BYTES - 1);
    let body = format!("{prefix}émore");
    let messages = vec![
        Message::user("task"),
        tool_call("c1", "bash"),
        tool_result("c1", body.clone()),
    ];
    let projected = pack.project(&messages);
    let packed = tool_result_text(&projected[2]).unwrap();
    let id = packed
        .lines()
        .find_map(|line| line.strip_prefix("id: "))
        .expect("placeholder id");
    let head = pack.recall(id, 0).expect("recall");
    assert!(
        !head.text.contains('\u{FFFD}'),
        "split character must not become U+FFFD: {:?}",
        head.text.chars().rev().take(4).collect::<String>()
    );
    assert_eq!(head.text.as_bytes(), &body.as_bytes()[..head.bytes]);
    assert_eq!(head.next_offset, RECALL_MAX_BYTES - 1);
    let rest = pack.recall(id, head.next_offset).expect("continuation");
    assert!(rest.text.starts_with('é'), "{}", rest.text);
    assert_eq!(
        &body.as_bytes()[head.next_offset..head.next_offset + rest.bytes],
        rest.text.as_bytes()
    );
}

#[test]
fn handle_obs_recall_does_not_clip_pages_over_the_shared_tool_cap() {
    let mut cfg = crate::tests::common::config();
    cfg.memory.observation_pack = true;
    let mut agent = crate::tests::common::agent(Vec::new(), cfg);
    let body: String = (0..400)
        .map(|i| format!("unique-line-{i:04}-payload-xxxx\n"))
        .collect();
    assert!(
        body.chars().count() > 5_000,
        "need a body larger than the default tool-result cap"
    );
    let mid_at = body.len() / 2;
    let middle = &body[mid_at - 24..mid_at + 24];
    let (clipped, _) = hi_tools::bound_tool_content(body.clone());
    assert!(
        !clipped.contains(middle),
        "precondition failed: bound_tool_content kept the middle"
    );

    let messages = vec![
        Message::user("task"),
        tool_call("c1", "bash"),
        tool_result("c1", body.clone()),
        assistant_text("one"),
        assistant_text("two"),
    ];
    let projected = agent.observation_pack.project(&messages);
    let packed = tool_result_text(&projected[2]).unwrap();
    let id = packed
        .lines()
        .find_map(|line| line.strip_prefix("id: "))
        .expect("placeholder id");
    let outcome = agent.handle_obs_recall(&format!(r#"{{"id":"{id}","offset":0}}"#));
    assert_eq!(outcome.status, hi_tools::ToolStatus::Succeeded);
    let spent = agent.handle_obs_recall(&format!(r#"{{"id":"{id}","offset":0}}"#));
    assert_eq!(spent.status, hi_tools::ToolStatus::Failed);
    assert!(
        spent.content.contains("paging budget"),
        "second recall in a turn must stop the live obs_recall loop: {}",
        spent.content
    );
    assert!(
        !outcome.content.contains("truncated"),
        "recall must not insert a truncation marker: {}",
        outcome.content
    );
    assert!(
        outcome.content.contains(middle),
        "handle_obs_recall dropped the middle of a page larger than the shared cap"
    );
    let page = outcome
        .content
        .split_once("use next_offset to continue]\n")
        .map(|(_, rest)| rest)
        .unwrap_or(&outcome.content);
    assert_eq!(
        page.as_bytes(),
        &body.as_bytes()[..page.len()],
        "recall page must be an exact archive prefix"
    );
}
