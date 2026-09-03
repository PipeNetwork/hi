//! End-to-end: drive the real `hi` binary against a fake OpenAI server and verify
//! the auto-memory loop — a session distills lessons into `.hi/memory.md`, and the
//! next session loads them back into the system prompt. Deterministic and offline
//! (no real model). Uses the `--plain` REPL driven over piped stdin, which fires
//! the same quit hook as the TUI.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use hi_ai::test_support::{FakeOpenAiServer, Response, sse_text};

/// A streamed completion that also reports token usage, so the turn registers as
/// real work (the memory gate requires `output_tokens > 0`).
fn sse_with_usage(text: &str) -> String {
    let content = serde_json::to_string(text).unwrap();
    format!(
        "data: {{\"choices\":[{{\"delta\":{{\"content\":{content}}},\"finish_reason\":null}}]}}\n\n\
         data: {{\"choices\":[{{\"delta\":{{}},\"finish_reason\":\"stop\"}}],\"usage\":{{\"prompt_tokens\":10,\"completion_tokens\":5}}}}\n\n\
         data: [DONE]\n\n"
    )
}

/// A temp dir that removes itself on drop (so a panicking test doesn't leak it).
struct TempDir {
    root: PathBuf,
    workspace: PathBuf,
}
impl TempDir {
    fn new(tag: &str) -> Self {
        let root = std::env::temp_dir().join(format!("hi-e2e-{}-{tag}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let workspace = root.join("workspace");
        std::fs::create_dir_all(&workspace).unwrap();
        std::fs::create_dir_all(root.join("home")).unwrap();
        Self { root, workspace }
    }
    fn path(&self) -> &Path {
        &self.workspace
    }
}
impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.root);
    }
}

fn isolated_home(workspace: &Path) -> PathBuf {
    workspace
        .parent()
        .expect("test workspace should have a temp root")
        .join("home")
}

/// Run the `hi` binary in `dir`, pointed at `server_url`, driving the `--plain`
/// REPL with `stdin_script` (stdin closes → EOF → quit). Blocks until it exits.
fn run_hi(dir: &Path, server_url: &str, extra_args: &[&str], stdin_script: &str) {
    let mut child = Command::new(env!("CARGO_BIN_EXE_hi"))
        .current_dir(dir)
        // Keep user-scoped state outside the workspace. Otherwise trace/session
        // bookkeeping looks like a candidate mutation and creates a spurious
        // verification obligation in these CLI behavior tests.
        .env("HOME", isolated_home(dir))
        .env("HI_MODEL", "fake/model")
        .env("HI_BASE_URL", server_url)
        .env("HI_API_KEY", "test")
        .env("HI_SUGGEST_NEXT_PROMPT", "0")
        .arg("--plain")
        .args(extra_args)
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn hi");
    child
        .stdin
        .take()
        .unwrap()
        .write_all(stdin_script.as_bytes())
        .unwrap(); // handle drops here → stdin closed → EOF
    let status = child.wait().expect("wait for hi");
    assert!(status.code().is_some(), "hi exited cleanly (not killed)");
}

fn run_hi_one_shot_output(
    dir: &Path,
    server_url: &str,
    extra_args: &[&str],
    prompt: &str,
) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_hi"))
        .current_dir(dir)
        .env("HOME", isolated_home(dir))
        .env("HI_MODEL", "fake/model")
        .env("HI_BASE_URL", server_url)
        .env("HI_API_KEY", "test")
        .env("HI_SUGGEST_NEXT_PROMPT", "0")
        .args(extra_args)
        .arg(prompt)
        .output()
        .expect("spawn hi")
}

fn run_hi_one_shot(dir: &Path, server_url: &str, extra_args: &[&str], prompt: &str) {
    let output = run_hi_one_shot_output(dir, server_url, extra_args, prompt);
    assert!(
        output.status.success(),
        "hi failed with {:?}\nstdout:\n{}\nstderr:\n{}",
        output.status.code(),
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn one_shot_report_creates_parent_directories() {
    let Some(server) = FakeOpenAiServer::new(vec![
        Response::sse(sse_with_usage("ok")),
        Response::sse(sse_with_usage("ok")),
    ]) else {
        return;
    };
    let tmp = TempDir::new("report");
    let report = tmp.path().join("reports/nested/run.json");
    let report_arg = report.to_string_lossy().to_string();

    run_hi_one_shot(
        tmp.path(),
        server.url(),
        &["--no-save", "--no-finalize", "--report", &report_arg],
        "say hi",
    );

    let text = std::fs::read_to_string(&report).expect("report should be written");
    let json: serde_json::Value = serde_json::from_str(&text).expect("report json");
    assert_eq!(json["schema_version"], 2);
    assert_eq!(json["outcome"]["effective_route"]["model"], "fake/model");
    assert_eq!(json["outcome"]["status"], "completed");
    assert_eq!(json["outcome"]["verification"], "not_applicable");
    assert!(json["usage"]["session"]["input_tokens"].as_u64().unwrap() >= 10);
    assert!(json["usage"]["session"]["output_tokens"].as_u64().unwrap() >= 5);
    assert!(json["usage"]["session"]["total_tokens"].as_u64().unwrap() >= 15);
    assert!(json["usage"]["turn"]["input_tokens"].as_u64().unwrap() >= 10);
    assert!(json["usage"]["turn"]["input_tokens"].as_u64().unwrap() >= 10);
    assert!(json["usage"]["turn"]["output_tokens"].as_u64().unwrap() >= 5);
    assert!(json["usage"]["turn"]["total_tokens"].as_u64().unwrap() >= 15);
    assert_eq!(json["usage"]["turn"]["user_prompt_estimated_tokens"], 2);
    assert_eq!(json["usage"]["turn"]["raw_user_prompt_estimated_tokens"], 2);
    assert!(json.get("verify_passed").is_none(), "v1 field was emitted");
}

#[test]
fn review_repair_report_settles_without_obsolete_failure_states() {
    let weak_review = "The repository looks healthy and organized.";
    // Surplus responses: the weak review triggers 4 quality-repair nudges
    // plus keep-working recoveries (14 model calls total), all served by this
    // one fake server. Extra responses sit unused but prevent a dead-connection
    // error that would mask the settled review outcome.
    let Some(server) = FakeOpenAiServer::new(
        (0..20)
            .map(|_| Response::sse(sse_with_usage(weak_review)))
            .collect(),
    ) else {
        return;
    };
    let tmp = TempDir::new("review-report");
    let report = tmp.path().join("reports/review.json");
    let report_arg = report.to_string_lossy().to_string();

    let output = run_hi_one_shot_output(
        tmp.path(),
        server.url(),
        &[
            "--no-save",
            "--no-memory",
            "--no-auto-compact",
            "--no-finalize",
            "--report",
            &report_arg,
        ],
        "/status codebase state",
    );
    assert_eq!(
        output.status.code(),
        Some(0),
        "bounded review repair should settle normally\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let visible = format!(
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let visible_lower = visible.to_ascii_lowercase();
    assert!(
        !visible_lower.contains("insufficient evidence")
            && !visible_lower.contains("quality_rejected")
            && !visible_lower.contains("incomplete")
            && !visible_lower.contains("stalled"),
        "review repair text should not leak visibly:\n{visible}"
    );
    let bodies = server.bodies();
    assert!(
        bodies.len() > 1,
        "weak review answer should trigger additional model calls"
    );
    assert!(
        bodies
            .iter()
            .any(|body| body.contains("not a git repository; no git diff available")),
        "non-git preflight diff output should be concise: {bodies:?}"
    );
    assert!(
        !bodies
            .iter()
            .any(|body| body.to_ascii_lowercase().contains("usage: git diff")),
        "non-git preflight diff output should not include verbose git help: {bodies:?}"
    );

    let text = std::fs::read_to_string(&report).expect("report should be written");
    let json: serde_json::Value = serde_json::from_str(&text).expect("report json");
    assert_eq!(json["outcome"]["status"], "completed");
    assert_eq!(json["outcome"]["stop_reason"], "no_applicable_verification");
    let telemetry = &json["telemetry"];
    assert_eq!(telemetry["quality_repair_nudges"], 4);
    assert_eq!(
        telemetry["review_repair_exhaustion_reason"],
        "review_listing_only_exhausted"
    );
    assert_eq!(telemetry["review_repair_counts"]["review_listing_only"], 4);
    assert_eq!(telemetry["review_repair_stopped_by_exhaustion"], true);
    assert_eq!(telemetry["stopped_by_step_cap"], false);
    assert_eq!(telemetry["stopped_by_tool_cap"], false);
    assert!(telemetry.get("stalled_unfinished").is_none());
    assert!(telemetry.get("stalled_repeating").is_none());
    assert_eq!(telemetry["hit_step_cap"], false);
    assert_eq!(telemetry["hit_tool_cap"], false);
    assert!(telemetry["progress_events"].is_array());
    assert!(telemetry["tool_timeline"].is_array());
    assert!(
        telemetry["progress_events"]
            .as_array()
            .unwrap()
            .iter()
            .any(|event| event["reason"] == "review_listing_only_exhausted"),
        "progress events should include review exhaustion reason: {telemetry}"
    );
}

#[test]
fn memory_distills_at_quit_and_reloads_next_session() {
    // Round 1: an explicit user preference remains eligible for memory even
    // without a verifier-backed mutation, followed by quit-time distillation.
    // User-scoped bookkeeping lives outside the workspace, so the turn consumes
    // one model call and quit-time distillation consumes the second.
    let Some(server1) = FakeOpenAiServer::new(vec![
        Response::sse(sse_with_usage("Done — fixed it.")),
        Response::sse(sse_text("- always run cargo fmt before commits")),
    ]) else {
        return; // sandbox can't bind a socket → skip
    };
    let tmp = TempDir::new("mem");
    run_hi(
        tmp.path(),
        server1.url(),
        &[],
        "I prefer always running cargo fmt before commits\n",
    );

    let mem = std::fs::read_to_string(tmp.path().join(".hi/memory.md"))
        .expect("round 1 should write .hi/memory.md");
    assert!(
        mem.contains("always run cargo fmt"),
        "distilled memory saved: {mem}"
    );

    // Round 2 (same dir): the saved memory must load into the system prompt.
    // `--no-memory` so this run doesn't re-distill — one turn, one request to inspect.
    let Some(server2) = FakeOpenAiServer::new(vec![Response::sse(sse_with_usage("ok"))]) else {
        return;
    };
    run_hi(
        tmp.path(),
        server2.url(),
        &["--no-memory"],
        "do something\n",
    );

    let body = server2
        .bodies()
        .into_iter()
        .next()
        .expect("round 2 should make a request");
    assert!(
        // Phase P: live task-ranked section (header includes "; task-ranked").
        (body.contains("Memory (from past sessions") || body.contains("task-ranked"))
            && body.contains("always run cargo fmt"),
        "memory loaded into the system prompt: {body}"
    );
}

/// A streamed completion that emits a single `write` tool call (then stops), so
/// the agent has a planned mutating action to act on.
fn sse_write_tool_call(path: &str, content: &str) -> String {
    let arguments =
        serde_json::to_string(&serde_json::json!({ "path": path, "content": content }).to_string())
            .unwrap();
    format!(
        "data: {{\"choices\":[{{\"delta\":{{\"tool_calls\":[{{\"index\":0,\"id\":\"call_1\",\"function\":{{\"name\":\"write\",\"arguments\":{arguments}}}}}]}}}}]}}\n\n\
         data: {{\"choices\":[{{\"delta\":{{}},\"finish_reason\":\"tool_calls\"}}],\"usage\":{{\"prompt_tokens\":10,\"completion_tokens\":5}}}}\n\n\
         data: [DONE]\n\n"
    )
}

#[test]
fn dry_run_prints_planned_action_without_mutating_workspace() {
    // Turn 1: model asks to write a file. Subsequent requests (the re-prompt
    // after the synthetic dry-run result, plus any suggest/finalize call) all
    // get a terminal prose reply so the turn ends without further tool calls.
    let mut responses = vec![Response::sse(sse_write_tool_call(
        "dry_run_marker.txt",
        "should not be written",
    ))];
    responses.extend((0..8).map(|_| Response::sse(sse_with_usage("done"))));
    let Some(server) = FakeOpenAiServer::new(responses) else {
        return; // sandbox can't bind a socket → skip
    };
    let tmp = TempDir::new("dry-run");

    let output = run_hi_one_shot_output(
        tmp.path(),
        server.url(),
        &["--no-save", "--no-memory", "--no-finalize", "--dry-run"],
        "create a marker file",
    );
    assert!(
        output.status.success(),
        "hi failed with {:?}\nstdout:\n{}\nstderr:\n{}",
        output.status.code(),
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    // The planned action is reported as a dry-run, not executed.
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("[dry-run] would run `write`"),
        "dry-run planned action printed: {stdout}"
    );
    assert!(
        !tmp.path().join("dry_run_marker.txt").exists(),
        "dry-run must not create the file"
    );
}
