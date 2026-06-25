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
         data: {{\"choices\":[{{\"delta\":{{}},\"finish_reason\":\"stop\"}}]}}\n\n\
         data: {{\"choices\":[],\"usage\":{{\"prompt_tokens\":10,\"completion_tokens\":5}}}}\n\n\
         data: [DONE]\n\n"
    )
}

/// A temp dir that removes itself on drop (so a panicking test doesn't leak it).
struct TempDir(PathBuf);
impl TempDir {
    fn new(tag: &str) -> Self {
        let dir = std::env::temp_dir().join(format!("hi-e2e-{}-{tag}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        Self(dir)
    }
    fn path(&self) -> &Path {
        &self.0
    }
}
impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Run the `hi` binary in `dir`, pointed at `server_url`, driving the `--plain`
/// REPL with `stdin_script` (stdin closes → EOF → quit). Blocks until it exits.
fn run_hi(dir: &Path, server_url: &str, extra_args: &[&str], stdin_script: &str) {
    let mut child = Command::new(env!("CARGO_BIN_EXE_hi"))
        .current_dir(dir)
        .env("HOME", dir) // isolate config + session storage from the real home
        .env("HI_MODEL", "fake/model")
        .env("HI_BASE_URL", server_url)
        .env("HI_API_KEY", "test")
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

#[test]
fn memory_distills_at_quit_and_reloads_next_session() {
    // Round 1: one turn (with usage, so work is registered) + the quit-time
    // distillation. "Done — fixed it." is text-only and doesn't look like an
    // unfinished step, so the silent auto-continue doesn't fire — two canned
    // responses suffice (the turn response + the distillation).
    let Some(server1) = FakeOpenAiServer::new(vec![
        Response::sse(sse_with_usage("Done — fixed it.")),
        Response::sse(sse_text("- always run cargo fmt before commits")),
    ]) else {
        return; // sandbox can't bind a socket → skip
    };
    let tmp = TempDir::new("mem");
    run_hi(tmp.path(), server1.url(), &[], "fix the bug\n");

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
        body.contains("Memory (from past sessions)") && body.contains("always run cargo fmt"),
        "memory loaded into the system prompt: {body}"
    );
}
