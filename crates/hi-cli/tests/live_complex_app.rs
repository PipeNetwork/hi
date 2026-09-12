//! Live multi-turn stall hunt against a copy of the chat app.
//!
//! Off unless `HI_LIVE=1`. Uses the operator's configured provider (auth store
//! under `$HOME/.config/hi`). The assertion is the stall we keep shipping:
//! inspection/`obs_recall` until request-limit exhaustion with no file changes.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Duration;

fn live_enabled() -> bool {
    std::env::var("HI_LIVE").ok().as_deref() == Some("1")
}

fn hi_bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_hi"))
}

fn copy_tree(src: &Path, dest: &Path) {
    std::fs::create_dir_all(dest).unwrap();
    for entry in std::fs::read_dir(src).unwrap() {
        let entry = entry.unwrap();
        let name = entry.file_name();
        if matches!(name.to_str(), Some("target" | ".git" | "chat.db" | ".hi")) {
            continue;
        }
        let from = entry.path();
        let to = dest.join(&name);
        if from.is_dir() {
            copy_tree(&from, &to);
        } else {
            std::fs::copy(&from, &to).unwrap();
        }
    }
}

fn run_turn(workspace: &Path, report: &Path, prompt: &str) -> serde_json::Value {
    let output = Command::new(hi_bin())
        .current_dir(workspace)
        .arg("--report")
        .arg(report)
        .arg(prompt)
        .env("HI_SUGGEST_NEXT_PROMPT", "0")
        .output()
        .expect("spawn live hi");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        report.exists(),
        "live hi did not write a report (status={:?})\nstdout:\n{stdout}\nstderr:\n{stderr}",
        output.status
    );
    let raw = std::fs::read_to_string(report).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&raw).unwrap_or_else(|error| {
        panic!("report is not JSON ({error}): {raw}\nstdout:\n{stdout}\nstderr:\n{stderr}")
    });
    parsed
}

fn changed_files(report: &serde_json::Value) -> Vec<String> {
    report
        .pointer("/outcome/changed_files")
        .and_then(|value| value.as_array())
        .into_iter()
        .flatten()
        .filter_map(|value| value.as_str().map(str::to_owned))
        .collect()
}

fn visible_text(report: &serde_json::Value) -> String {
    format!("{report}")
}

fn tool_names(report: &serde_json::Value) -> Vec<String> {
    report
        .pointer("/tools")
        .and_then(|value| value.as_array())
        .into_iter()
        .flatten()
        .filter_map(|value| {
            value
                .get("name")
                .and_then(|name| name.as_str())
                .map(str::to_owned)
        })
        .collect()
}

fn outcome_status(report: &serde_json::Value) -> String {
    report
        .pointer("/outcome/status")
        .and_then(|value| value.as_str())
        .unwrap_or("")
        .to_string()
}

fn outcome_stop(report: &serde_json::Value) -> String {
    report
        .pointer("/outcome/stop_reason")
        .and_then(|value| value.as_str())
        .unwrap_or("")
        .to_string()
}

fn assistant_response(report: &serde_json::Value) -> String {
    report
        .get("assistant_response")
        .and_then(|value| value.as_str())
        .unwrap_or("")
        .to_string()
}

const LEFTOVER_EMPTY_CLOSEOUT: &str = "Automatic recovery stopped. No file changes were made.";
const REVIEW_FIX_PROMPT: &str = "review for any major issues and fix";
const PLANTED_TEST_FN: &str = "fn planted_review_fix_e2e";
const PLANTED_FAILING_ASSERT: &str = "assert_eq!(2 + 2, 5)";
const COMPILE_BREAK: &str = "definitely_not_defined";

struct LiveTurn {
    files: Vec<String>,
    status: String,
    stop: String,
    assistant: String,
    names: Vec<String>,
    visible: String,
    requests: u64,
    elapsed: Duration,
}

struct LiveExpect {
    forbid_dump: bool,
    forbid_obs_recall: bool,
    forbid_recovery_empty_closeout: bool,
    forbid_no_progress_empty: bool,
    forbid_withhold_before_edit: bool,
    forbid_request_limit: bool,
    require_files: bool,
    require_completed_if_empty: bool,
}

impl LiveExpect {
    fn stall_guards() -> Self {
        Self {
            forbid_dump: true,
            forbid_obs_recall: true,
            forbid_recovery_empty_closeout: true,
            forbid_no_progress_empty: true,
            forbid_withhold_before_edit: true,
            forbid_request_limit: true,
            require_files: false,
            require_completed_if_empty: false,
        }
    }
}

fn copy_chat_fixture() -> (tempfile::TempDir, PathBuf) {
    let src = PathBuf::from("/Users/david/chat");
    assert!(
        src.join("src/web/index.html").exists(),
        "chat fixture missing"
    );
    let root = tempfile::TempDir::new().unwrap();
    let workspace = root.path().join("chat");
    copy_tree(&src, &workspace);
    (root, workspace)
}

fn collect_turn(workspace: &Path, report_path: &Path, prompt: &str) -> LiveTurn {
    let started = std::time::Instant::now();
    let report = run_turn(workspace, report_path, prompt);
    let elapsed = started.elapsed();
    assert!(
        elapsed < Duration::from_secs(8 * 60),
        "live turn hung: {elapsed:?}"
    );
    LiveTurn {
        files: changed_files(&report),
        status: outcome_status(&report),
        stop: outcome_stop(&report),
        assistant: assistant_response(&report),
        names: tool_names(&report),
        visible: visible_text(&report),
        requests: report
            .pointer("/model_outcome/model_requests")
            .and_then(|value| value.as_u64())
            .unwrap_or(0),
        elapsed,
    }
}

fn assert_live_turn(label: &str, turn: &LiveTurn, expect: LiveExpect) {
    if expect.forbid_dump {
        assert!(
            !turn.assistant.contains("Insufficient evidence:")
                && !turn.assistant.contains("Evidence summary:"),
            "{label} dumped the gap/roadmap template: status={} stop={} requests={} elapsed={:?} files={:?} tools={:?} assistant={}",
            turn.status,
            turn.stop,
            turn.requests,
            turn.elapsed,
            turn.files,
            turn.names,
            turn.assistant
        );
    }
    if expect.forbid_obs_recall {
        assert!(
            !turn.names.iter().any(|name| name == "obs_recall"),
            "{label} used obs_recall paging: {:?}",
            turn.names
        );
    }
    if expect.forbid_recovery_empty_closeout {
        assert!(
            !turn.assistant.contains(LEFTOVER_EMPTY_CLOSEOUT)
                && !turn.visible.contains(LEFTOVER_EMPTY_CLOSEOUT),
            "{label} leftover-exhausted with no file changes: status={} stop={} requests={} tools={:?} assistant={}",
            turn.status,
            turn.stop,
            turn.requests,
            turn.names,
            turn.assistant
        );
    }
    if expect.forbid_no_progress_empty {
        let no_progress_empty = turn.status.eq_ignore_ascii_case("failed")
            && turn.stop.to_ascii_lowercase().contains("no_progress")
            && turn.files.is_empty();
        assert!(
            !no_progress_empty,
            "{label} failed with no file changes after inspection/cargo test: status={} stop={} requests={} tools={:?} assistant={}",
            turn.status, turn.stop, turn.requests, turn.names, turn.assistant
        );
    }
    if expect.forbid_withhold_before_edit {
        assert!(
            !turn
                .visible
                .contains("withholding inspection tools until a file change lands")
                || !turn.files.is_empty(),
            "{label} withheld inspection before a file change: status={} stop={} requests={} tools={:?}",
            turn.status,
            turn.stop,
            turn.requests,
            turn.names
        );
    }
    if expect.forbid_request_limit {
        assert!(
            !turn.visible.contains("request limit exhausted")
                && !turn
                    .assistant
                    .contains("provider request budget ran out during inspection"),
            "{label} hit request-limit exhaustion: status={} stop={} requests={} assistant={}",
            turn.status,
            turn.stop,
            turn.requests,
            turn.assistant
        );
    }
    if expect.require_files {
        assert!(
            !turn.files.is_empty(),
            "{label} made no file changes: status={} stop={} requests={} elapsed={:?} tools={:?} assistant={}",
            turn.status,
            turn.stop,
            turn.requests,
            turn.elapsed,
            turn.names,
            turn.assistant
        );
    }
    if expect.require_completed_if_empty && turn.files.is_empty() {
        assert!(
            turn.status.eq_ignore_ascii_case("completed"),
            "{label} empty-file turn must complete, not stall: status={} stop={} requests={} tools={:?} assistant={}",
            turn.status,
            turn.stop,
            turn.requests,
            turn.names,
            turn.assistant
        );
    }
}

fn plant_src_visible_failing_test(workspace: &Path) {
    let path = workspace.join("src/main.rs");
    let mut src = std::fs::read_to_string(&path).unwrap();
    src.push_str(
        "\n#[cfg(test)]\nmod planted_review_fix {\n    #[test]\n    fn planted_review_fix_e2e() {\n        assert_eq!(2 + 2, 5);\n    }\n}\n",
    );
    std::fs::write(&path, src).unwrap();
}

fn plant_compile_error(workspace: &Path) {
    let path = workspace.join("src/web.rs");
    let mut src = std::fs::read_to_string(&path).unwrap();
    src.push_str("\npub fn hi_live_e2e_compile_break() { definitely_not_defined(); }\n");
    std::fs::write(&path, src).unwrap();
}

fn cargo_ok(workspace: &Path, args: &[&str]) -> (bool, String) {
    let output = Command::new("cargo")
        .current_dir(workspace)
        .args(args)
        .output()
        .expect("spawn cargo");
    let mut text = String::from_utf8_lossy(&output.stdout).into_owned();
    text.push_str(&String::from_utf8_lossy(&output.stderr));
    (output.status.success(), text)
}

fn copy_irc_fixture() -> (tempfile::TempDir, PathBuf) {
    let src = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../bench/tui-smoke/scenarios/live_review_fix_chat/fixture");
    assert!(
        src.join("src/main.rs").exists() && src.join("tests/integration.rs").exists(),
        "IRC live fixture missing at {}",
        src.display()
    );
    let root = tempfile::TempDir::new().unwrap();
    let workspace = root.path().join("irc");
    copy_tree(&src, &workspace);
    (root, workspace)
}

/// The backend already has `POST /register`. A Create-account button with
/// no fetch, leftover JS after a prior live turn, or the word "register"
/// in `web.rs`, is not a UI fix.
fn html_wires_register(html: &str) -> bool {
    let lower = html.to_ascii_lowercase();
    let has_control = lower.contains("id=\"register\"")
        || lower.contains("id='register'")
        || lower.contains("id=\"registerform\"")
        || lower.contains("id='registerform'")
        || lower.contains("id=\"registerbox\"")
        || lower.contains("id='registerbox'");
    has_control && lower.contains("/register") && lower.contains("fetch")
}

fn break_account_surface(html: &str) -> String {
    let mut out = html.replace(
        "    <button id=\"register\" type=\"button\">Create account</button>\n",
        "",
    );
    if let Some(start) = out.find("$(\"register\").addEventListener")
        && let Some(rel) = out[start..].find("\n$(\"composer\")")
    {
        out.replace_range(start..start + rel, "");
    }
    out
}

fn break_copied_account_surface(workspace: &Path) {
    let path = workspace.join("src/web/index.html");
    let html = std::fs::read_to_string(&path).unwrap_or_default();
    let stripped = break_account_surface(&html);
    if stripped != html {
        std::fs::write(&path, stripped).unwrap();
    }
}

#[test]
#[ignore = "set HI_LIVE=1 to run against the configured provider"]
fn live_chat_account_offline_does_not_stall_without_edits() {
    if !live_enabled() {
        return;
    }
    let (root, workspace) = copy_chat_fixture();
    break_copied_account_surface(&workspace);
    let starting_html =
        std::fs::read_to_string(workspace.join("src/web/index.html")).unwrap_or_default();
    assert!(
        !html_wires_register(&starting_html),
        "live fixture must start unwired so a prior chat fix cannot make this a no-op:\n{starting_html}"
    );

    let prompts = [
        "it shows it offline and i don't see how to make an account. fix",
        "the create account flow is still missing from the web UI. add a visible register form and wire it to POST /register",
        "cargo check and fix any compile errors from the account UI work",
    ];
    let mut any_changes = Vec::new();
    for (index, prompt) in prompts.iter().enumerate() {
        let report_path = root.path().join(format!("turn-{index}.json"));
        let turn = collect_turn(&workspace, &report_path, prompt);
        let mut expect = LiveExpect::stall_guards();
        expect.require_files = index == 0;
        expect.forbid_no_progress_empty = false;
        assert_live_turn(&format!("account-offline turn {index}"), &turn, expect);
        if turn.status.eq_ignore_ascii_case("failed")
            && turn.stop.to_ascii_lowercase().contains("no_progress")
        {
            let html =
                std::fs::read_to_string(workspace.join("src/web/index.html")).unwrap_or_default();
            assert!(
                index > 0 && html_wires_register(&html),
                "turn {index} failed with no_progress: status={} stop={} requests={} files={:?} tools={:?} assistant={}",
                turn.status,
                turn.stop,
                turn.requests,
                turn.files,
                turn.names,
                turn.assistant
            );
        }
        any_changes.extend(turn.files);
    }
    assert!(
        !any_changes.is_empty(),
        "three live turns on the chat app made no file changes"
    );
    let html = std::fs::read_to_string(workspace.join("src/web/index.html")).unwrap();
    assert!(
        html_wires_register(&html),
        "live turns did not wire the UI to POST /register:\n{html}"
    );
}

/// Healthy copy of ~/chat: green `cargo test`, no planted bug. Inspection plus
/// a recap must Complete. The TUI leftover closeout after 9 reads + cargo test
/// is a test failure.
#[test]
#[ignore = "set HI_LIVE=1 to run against the configured provider"]
fn live_review_and_fix_healthy_workspace_does_not_leftover_stall() {
    if !live_enabled() {
        return;
    }
    let (root, workspace) = copy_chat_fixture();
    let report_path = root.path().join("review-fix-healthy.json");
    let turn = collect_turn(&workspace, &report_path, REVIEW_FIX_PROMPT);
    let mut expect = LiveExpect::stall_guards();
    expect.require_completed_if_empty = true;
    assert_live_turn("healthy review-and-fix", &turn, expect);
}

/// Plant a failing unit test in `src/main.rs` so `cargo test --quiet` fails
/// unless that file is edited. A toy `tests/planted_review_fix.rs` is too
/// easy to never open.
#[test]
#[ignore = "set HI_LIVE=1 to run against the configured provider"]
fn live_review_and_fix_repairs_src_visible_failing_test() {
    if !live_enabled() {
        return;
    }
    let (root, workspace) = copy_chat_fixture();
    plant_src_visible_failing_test(&workspace);
    let report_path = root.path().join("review-fix-planted.json");
    let turn = collect_turn(&workspace, &report_path, REVIEW_FIX_PROMPT);
    let mut expect = LiveExpect::stall_guards();
    expect.require_files = true;
    assert_live_turn("planted review-and-fix", &turn, expect);
    let planted = std::fs::read_to_string(workspace.join("src/main.rs")).unwrap_or_default();
    assert!(
        planted.contains(PLANTED_TEST_FN),
        "review-and-fix must not delete the planted test"
    );
    assert!(
        !planted.contains(PLANTED_FAILING_ASSERT),
        "review-and-fix did not repair the planted failing assertion:\n{planted}"
    );
    let (ok, output) = cargo_ok(&workspace, &["test", "--quiet", "--offline"]);
    assert!(
        ok,
        "planted review-and-fix left cargo test failing:\n{output}"
    );
}

#[test]
#[ignore = "set HI_LIVE=1 to run against the configured provider"]
fn live_cargo_check_and_fix_repairs_compile_error() {
    if !live_enabled() {
        return;
    }
    let (root, workspace) = copy_chat_fixture();
    plant_compile_error(&workspace);
    let report_path = root.path().join("cargo-check-fix.json");
    let turn = collect_turn(
        &workspace,
        &report_path,
        "cargo check and fix any compile errors",
    );
    let mut expect = LiveExpect::stall_guards();
    expect.require_files = true;
    assert_live_turn("cargo check and fix", &turn, expect);
    let web = std::fs::read_to_string(workspace.join("src/web.rs")).unwrap_or_default();
    assert!(
        !web.contains(COMPILE_BREAK),
        "compile error still present:\n{web}"
    );
    let (ok, output) = cargo_ok(&workspace, &["check", "--offline", "--quiet"]);
    assert!(ok, "cargo check still fails after the fix turn:\n{output}");
}

/// After a healthy review, a verify-only follow-up whose wording contains
/// "change" must not be treated as a mutation request and leftover-stall.
#[test]
#[ignore = "set HI_LIVE=1 to run against the configured provider"]
fn live_verify_only_follow_up_after_review_does_not_stall() {
    if !live_enabled() {
        return;
    }
    let (root, workspace) = copy_chat_fixture();
    let first = collect_turn(
        &workspace,
        &root.path().join("review-fix.json"),
        REVIEW_FIX_PROMPT,
    );
    let mut first_expect = LiveExpect::stall_guards();
    first_expect.require_completed_if_empty = true;
    assert_live_turn("review before verify-only", &first, first_expect);

    let turn = collect_turn(
        &workspace,
        &root.path().join("verify-only.json"),
        "Run cargo test to verify the review change didn't break anything.",
    );
    let mut expect = LiveExpect::stall_guards();
    expect.require_completed_if_empty = true;
    expect.forbid_withhold_before_edit = false;
    assert_live_turn("verify-only follow-up", &turn, expect);
}

const IRC_REVIEW_PROMPT: &str = "Review this IRC-style chat server for major correctness bugs and fix them.\n\
Read tests/integration.rs and src/main.rs first. The integration tests fail today.\n\
Keep the existing tests; do not weaken or delete them. Make the tests pass by fixing the server.";

#[test]
#[ignore = "set HI_LIVE=1 to run against the configured provider"]
fn live_irc_review_fix_then_verify_only_does_not_stall() {
    if !live_enabled() {
        return;
    }
    let (root, workspace) = copy_irc_fixture();
    let integration = workspace.join("tests/integration.rs");
    let before = std::fs::read_to_string(&integration).unwrap();
    assert!(
        before.contains("read_until(\"Welcome\")") && before.contains("after kick"),
        "IRC fixture tests missing"
    );
    let (pre_ok, _) = cargo_ok(&workspace, &["test", "--offline", "--test", "integration"]);
    assert!(!pre_ok, "buggy IRC fixture already passed cargo test");

    let turn = collect_turn(
        &workspace,
        &root.path().join("irc-review.json"),
        IRC_REVIEW_PROMPT,
    );
    let mut expect = LiveExpect::stall_guards();
    expect.require_files = true;
    assert_live_turn("irc review-and-fix", &turn, expect);
    let after = std::fs::read_to_string(&integration).unwrap();
    assert!(
        after.contains("read_until(\"Welcome\")") && after.contains("after kick"),
        "IRC tests were deleted or weakened:\n{after}"
    );
    let (ok, output) = cargo_ok(&workspace, &["test", "--offline", "--test", "integration"]);
    assert!(ok, "IRC integration tests still fail after hi:\n{output}");

    let verify = collect_turn(
        &workspace,
        &root.path().join("irc-verify.json"),
        "Run cargo test to verify the welcome-message change didn't break anything.",
    );
    let mut verify_expect = LiveExpect::stall_guards();
    verify_expect.require_completed_if_empty = true;
    verify_expect.forbid_withhold_before_edit = false;
    assert_live_turn("irc verify-only follow-up", &verify, verify_expect);
}

#[test]
fn break_account_surface_removes_button_and_click_handler() {
    let html = r#"    <button id="connect" type="submit">Connect</button>
    <button id="register" type="button">Create account</button>
    <span id="status">offline</span>
<script>
$("register").addEventListener("click", async () => {
  const res = await fetch(origin + "/register", { method: "POST" });
});
$("composer").addEventListener("submit", (event) => {
  event.preventDefault();
});
</script>
"#;
    let out = break_account_surface(html);
    assert!(
        !html_wires_register(&out),
        "stripped markup still counts as wired:\n{out}"
    );
    assert!(
        !out.contains("$(\"register\")"),
        "register listener survived:\n{out}"
    );
    assert!(
        out.contains("$(\"composer\")"),
        "composer handler must stay:\n{out}"
    );
    assert!(
        html_wires_register(html),
        "fixture snippet must look wired before strip"
    );
    let form = r#"<form id="registerForm"><button>Register</button></form>
<script>fetch(origin + "/register", { method: "POST" })</script>"#;
    assert!(
        html_wires_register(form),
        "a register form that POSTs /register must count as wired:\n{form}"
    );
}

#[test]
fn copied_chat_fixture_starts_without_register_wiring() {
    let src = PathBuf::from("/Users/david/chat");
    if !src.join("src/web/index.html").exists() {
        return;
    }
    let root = tempfile::TempDir::new().unwrap();
    let workspace = root.path().join("chat");
    copy_tree(&src, &workspace);
    break_copied_account_surface(&workspace);
    let html = std::fs::read_to_string(workspace.join("src/web/index.html")).unwrap();
    assert!(
        !html_wires_register(&html),
        "sanitized copy still looks wired:\n{html}"
    );
    assert!(
        !html.contains("$(\"register\")"),
        "register click handler leaked into the live fixture:\n{html}"
    );
}

#[test]
fn plant_src_visible_failing_test_appends_to_main() {
    let root = tempfile::TempDir::new().unwrap();
    let path = root.path().join("src/main.rs");
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(&path, "fn main() {}\n").unwrap();
    plant_src_visible_failing_test(root.path());
    let src = std::fs::read_to_string(&path).unwrap();
    assert!(src.contains(PLANTED_TEST_FN));
    assert!(src.contains(PLANTED_FAILING_ASSERT));
}

#[test]
fn plant_compile_error_breaks_web_rs() {
    let root = tempfile::TempDir::new().unwrap();
    let path = root.path().join("src/web.rs");
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(&path, "pub fn ready() {}\n").unwrap();
    plant_compile_error(root.path());
    let src = std::fs::read_to_string(&path).unwrap();
    assert!(src.contains(COMPILE_BREAK));
}

#[test]
fn irc_fixture_is_present_and_buggy() {
    let src = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../bench/tui-smoke/scenarios/live_review_fix_chat/fixture");
    if !src.join("src/main.rs").exists() {
        return;
    }
    let main = std::fs::read_to_string(src.join("src/main.rs")).unwrap();
    assert!(
        main.contains("pending.push(line)"),
        "IRC fixture must still drop Welcome on the floor"
    );
}
