//! `hi team-bench` — a small, honest coding benchmark for the supported
//! local models, answering "which of these is better at what?" with
//! machine-checked results instead of vibes.
//!
//! Each model is provisioned through the exact `/team` path (download →
//! `hi-local` serve → health), then given verifiable coding tasks:
//!
//! - `codegen`   — write a function from a spec; compiled and run against asserts
//! - `bugfix`    — repair a buggy function; compiled and run against asserts
//! - `edit`      — a precise mechanical edit; textual invariants + compiles
//! - `multiedit` — a rename across two files; both must come back and compile
//! - `repair`    — fix a function given the actual compiler error (the delegate
//!   verify-repair loop in miniature)
//! - `json`      — tool-call fidelity; strict JSON schema + exact values
//!
//! Scores are pass/fail per task plus measured generation speed. Servers are
//! started one at a time and stopped after each model so a 60GB ladder never
//! stacks in RAM. Results print as a table and are saved as JSON under
//! `~/.hi/bench/`.

use anyhow::{Context, Result, bail};
use hi_agent::local_skeptic::{
    LocalBackend, ProvisionPhase, ResolvedLocalModel, SUPPORTED_LOCAL_MODELS,
    detect_backend_offload, provision_team_local_model, resolve_team_local_model, system_ram_gb,
    team_model_spec,
};
use hi_ai::{ChatRequest, Content, Message, OpenAiProvider, Provider, RequestProfile};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// One benchmark task: a prompt and a machine validator for the reply.
struct BenchTask {
    name: &'static str,
    prompt: String,
    check: fn(&str) -> Result<()>,
}

#[derive(serde::Serialize, Clone)]
struct TaskResult {
    task: &'static str,
    pass: bool,
    /// Why the task failed (compile error, wrong value, bad JSON…); None on pass.
    error: Option<String>,
    latency_secs: f64,
    /// Seconds until the server produced its first stream event — the
    /// responsiveness a driver actually feels when delegating.
    ttft_secs: f64,
    output_tokens: u64,
    tokens_per_sec: f64,
}

#[derive(serde::Serialize)]
struct ModelReport {
    model: String,
    model_id: String,
    setup_secs: f64,
    results: Vec<TaskResult>,
}

impl ModelReport {
    fn passed(&self) -> usize {
        self.results.iter().filter(|r| r.pass).count()
    }
    fn median_ttft_secs(&self) -> f64 {
        let mut ttfts: Vec<f64> = self.results.iter().map(|r| r.ttft_secs).collect();
        if ttfts.is_empty() {
            return 0.0;
        }
        ttfts.sort_by(|a, b| a.total_cmp(b));
        ttfts[ttfts.len() / 2]
    }

    fn avg_tokens_per_sec(&self) -> f64 {
        let generating: Vec<&TaskResult> = self
            .results
            .iter()
            .filter(|r| r.output_tokens > 0)
            .collect();
        if generating.is_empty() {
            return 0.0;
        }
        generating.iter().map(|r| r.tokens_per_sec).sum::<f64>() / generating.len() as f64
    }
}

pub(crate) async fn run_team_bench_cli(args: &[String]) -> Result<()> {
    if args.first().map(String::as_str) == Some("--help") {
        println!(
            "usage: hi team-bench [--list] [model…]\n\n\
             Benchmark supported local models on verifiable coding tasks.\n\
             With no arguments, every catalog model already downloaded for this\n\
             machine's backend (and that fits it) is benchmarked. Naming models\n\
             (e.g. `laguna-s@3bit coder-32b`) benchmarks exactly those,\n\
             downloading weights if needed. --list shows the selection and exits."
        );
        return Ok(());
    }
    let list_only = args.first().map(String::as_str) == Some("--list");
    let args = if list_only { &args[1..] } else { args };
    let ram = system_ram_gb();
    let Some(backend) = detect_backend_offload().await else {
        bail!(
            "no local-inference backend detected (needs Apple Silicon MLX or an NVIDIA CUDA runtime)"
        );
    };
    let selections = if args.is_empty() {
        downloaded_selections(ram, backend)
    } else {
        let mut picked = Vec::new();
        for name in args {
            let Some(resolved) = resolve_team_local_model(name, ram, Some(backend)) else {
                bail!("'{name}' is not a supported local model — see /team for the catalog");
            };
            picked.push(resolved);
        }
        picked
    };
    if list_only {
        for resolved in &selections {
            println!("{}", resolved.display());
        }
        return Ok(());
    }
    if selections.is_empty() {
        println!(
            "no local models downloaded yet — name one to fetch and benchmark it, e.g.:\n  hi team-bench local"
        );
        return Ok(());
    }
    if !rustc_available() {
        bail!("team-bench compiles model output with `rustc`, which isn't on PATH");
    }

    let tasks = bench_tasks();
    println!(
        "benchmarking {} model(s) on {} coding task(s); servers run one at a time\n",
        selections.len(),
        tasks.len()
    );
    let mut reports = Vec::new();
    for resolved in selections {
        match bench_model(resolved, &tasks).await {
            Ok(report) => reports.push(report),
            Err(error) => {
                // One broken model must not sink the comparison run.
                println!("  ✗ {}: {error:#}\n", resolved.display());
            }
        }
    }
    if reports.is_empty() {
        bail!("no model completed the benchmark");
    }
    print_summary(&reports, &tasks);
    if let Some(path) = save_report(&reports) {
        println!("\nfull report: {}", path.display());
    }
    Ok(())
}

/// Every catalog selection whose weights are already on disk for `backend`
/// AND fit this machine — for MLX ladders, the highest-quality downloaded
/// quant of each family. A downloaded-but-oversized model (someone's 390GB
/// GLM copy on a 64GB Mac) must be named explicitly to be benched.
fn downloaded_selections(ram: u64, backend: LocalBackend) -> Vec<ResolvedLocalModel> {
    let mut picked = Vec::new();
    for entry in SUPPORTED_LOCAL_MODELS {
        let candidates: Vec<ResolvedLocalModel> = match backend {
            LocalBackend::Mlx => entry
                .mlx
                .iter()
                .filter(|quant| ram >= quant.min_ram_gb)
                .map(|quant| ResolvedLocalModel {
                    entry,
                    mlx: Some(quant),
                })
                .collect(),
            LocalBackend::Cuda => entry
                .cuda
                .filter(|cuda| ram >= cuda.min_ram_gb)
                .map(|_| ResolvedLocalModel {
                    entry,
                    mlx: entry.pick_mlx(ram),
                })
                .into_iter()
                .collect(),
        };
        if let Some(downloaded) = candidates.into_iter().find(|resolved| {
            team_model_spec(*resolved, backend).is_ok_and(|spec| {
                let dir = hi_tools::skeptic_model_dir(&spec.repo);
                hi_agent::local_skeptic::model_present(&dir, &spec)
            })
        }) {
            picked.push(downloaded);
        }
    }
    picked
}

/// Provision one model through the real `/team` path, run every task against
/// its server, and stop the server before returning.
async fn bench_model(resolved: ResolvedLocalModel, tasks: &[BenchTask]) -> Result<ModelReport> {
    let display = resolved.display();
    println!("— {display} —");
    let (phase_tx, mut phase_rx) = tokio::sync::watch::channel(ProvisionPhase::Resolving);
    let narrator = tokio::spawn(async move {
        while phase_rx.changed().await.is_ok() {
            let line = match &*phase_rx.borrow() {
                ProvisionPhase::Resolving => "resolving".to_string(),
                ProvisionPhase::Downloading => "downloading weights (quiet)…".to_string(),
                ProvisionPhase::BuildingServer => "building hi-local…".to_string(),
                ProvisionPhase::LoadingModel { deadline_secs, .. } => {
                    format!("server started — loading weights (up to {deadline_secs}s)…")
                }
            };
            println!("  {line}");
        }
    });
    let setup_started = Instant::now();
    let provisioned = provision_team_local_model(resolved, phase_tx).await;
    narrator.abort();
    let (endpoint, model_id, process_id) =
        provisioned.with_context(|| format!("setting up {display}"))?;
    let setup_secs = setup_started.elapsed().as_secs_f64();
    println!("  ready in {setup_secs:.0}s at {endpoint}");

    let provider = OpenAiProvider::new(endpoint, "local".to_string());
    let mut results = Vec::new();
    for task in tasks {
        let result = run_task(&provider, &model_id, task).await;
        let mark = if result.pass { "PASS" } else { "FAIL" };
        let detail = result
            .error
            .as_deref()
            .map(|e| format!(" — {}", e.lines().next().unwrap_or_default()))
            .unwrap_or_default();
        println!(
            "  {:<9} {mark}  {:>5.1}s (ttft {:>4.1}s)  {:>5.1} tok/s{detail}",
            task.name, result.latency_secs, result.ttft_secs, result.tokens_per_sec
        );
        results.push(result);
    }
    hi_tools::stop_local_server(&process_id);
    println!();
    Ok(ModelReport {
        model: display,
        model_id,
        setup_secs,
        results,
    })
}

/// One prompt → validate round. Task failures (bad output) are results, not
/// errors; only transport-level problems surface as FAIL with the cause.
async fn run_task(provider: &OpenAiProvider, model_id: &str, task: &BenchTask) -> TaskResult {
    let request = ChatRequest {
        model: model_id.to_string(),
        request_id: None,
        retry_attempt: 0,
        user_turn: false,
        canonical_objective: None,
        messages: Arc::new(vec![Message::user(task.prompt.clone())]),
        tools: Arc::new([]),
        // Room for models that reason inline before the code; hi-local's
        // default per-request ceiling is 8192.
        max_tokens: 3500,
        temperature: Some(0.0),
        top_p: None,
        frequency_penalty: None,
        thinking_budget: None,
        reasoning_effort: None,
        profile: RequestProfile::default(),
    };
    let started = Instant::now();
    let mut first_event: Option<Duration> = None;
    let completion = tokio::time::timeout(
        Duration::from_secs(600),
        provider.stream(request, &mut |_event| {
            if first_event.is_none() {
                first_event = Some(started.elapsed());
            }
        }),
    )
    .await;
    let latency_secs = started.elapsed().as_secs_f64();
    let ttft_secs = first_event.map(|d| d.as_secs_f64()).unwrap_or(latency_secs);
    let (text, output_tokens) = match completion {
        Ok(Ok(completion)) => {
            let text: String = completion
                .content
                .iter()
                .filter_map(|content| match content {
                    Content::Text(text) => Some(text.as_str()),
                    _ => None,
                })
                .collect();
            (text, completion.usage.output_tokens)
        }
        Ok(Err(error)) => {
            return TaskResult {
                task: task.name,
                pass: false,
                error: Some(format!("request failed: {error:#}")),
                latency_secs,
                ttft_secs,
                output_tokens: 0,
                tokens_per_sec: 0.0,
            };
        }
        Err(_) => {
            return TaskResult {
                task: task.name,
                pass: false,
                error: Some("timed out after 600s".to_string()),
                latency_secs,
                ttft_secs,
                output_tokens: 0,
                tokens_per_sec: 0.0,
            };
        }
    };
    let tokens_per_sec = if latency_secs > 0.0 {
        output_tokens as f64 / latency_secs
    } else {
        0.0
    };
    match (task.check)(&text) {
        Ok(()) => TaskResult {
            task: task.name,
            pass: true,
            error: None,
            latency_secs,
            ttft_secs,
            output_tokens,
            tokens_per_sec,
        },
        Err(error) => TaskResult {
            task: task.name,
            pass: false,
            error: Some(format!("{error:#}")),
            latency_secs,
            ttft_secs,
            output_tokens,
            tokens_per_sec,
        },
    }
}

// ---- Tasks -----------------------------------------------------------------

const BUGGY_WINDOW_SUM: &str = r#"/// Largest sum over any k consecutive elements; 0 when the slice has
/// fewer than k elements or k is 0.
pub fn max_window_sum(v: &[i64], k: usize) -> i64 {
    if k == 0 || v.len() < k {
        return 0;
    }
    let mut best = i64::MIN;
    for start in 0..v.len() - k {
        let sum: i64 = v[start..start + k].iter().sum();
        if sum > best {
            best = sum;
        }
    }
    best
}"#;

const MULTIEDIT_LIB: &str = r#"mod util;

pub fn describe(id: u64) -> String {
    match util::fetch_user(id) {
        Some(user) => user,
        None => "anonymous".to_string(),
    }
}"#;

const MULTIEDIT_UTIL: &str = r#"/// Fetch a user record by id.
pub fn fetch_user(id: u64) -> Option<String> {
    if id == 0 {
        return None;
    }
    Some(format!("user-{id}"))
}"#;

const REPAIR_SOURCE: &str = r#"pub fn sum_of_evens(v: &[i64]) -> i64 {
    v.iter().filter(|x| x % 2 == 0).sum()
}"#;

const REPAIR_ERROR: &str = r#"error[E0277]: cannot mod `&&i64` by `{integer}`
 --> src/lib.rs:2:25
  |
2 |     v.iter().filter(|x| x % 2 == 0).sum()
  |                         ^ no implementation for `&&i64 % {integer}`"#;

const EDIT_SOURCE: &str = r#"/// Fetch a user record by id.
pub fn fetch_user(id: u64) -> Option<String> {
    if id == 0 {
        return None;
    }
    Some(format!("user-{id}"))
}

pub fn describe(id: u64) -> String {
    match fetch_user(id) {
        Some(user) => user,
        None => "anonymous".to_string(),
    }
}"#;

fn bench_tasks() -> Vec<BenchTask> {
    vec![
        BenchTask {
            name: "codegen",
            prompt:
                "Write a Rust function `pub fn run_length_encode(s: &str) -> Vec<(char, u32)>` \
                     that collapses consecutive repeated characters into (character, count) pairs \
                     in order of appearance. For example \"aab\" becomes [('a', 2), ('b', 1)]. \
                     Return only the function code — no main, no tests, no explanation."
                    .to_string(),
            check: check_codegen,
        },
        BenchTask {
            name: "bugfix",
            prompt: format!(
                "This Rust function has a bug: it can miss a window and can return i64::MIN. \
                 For example max_window_sum(&[1, 2, 3, 4], 2) must be 7. Fix it and return only \
                 the corrected complete function — no explanation.\n\n{BUGGY_WINDOW_SUM}"
            ),
            check: check_bugfix,
        },
        BenchTask {
            name: "edit",
            prompt: format!(
                "In the Rust file below, rename the function `fetch_user` to `load_user` \
                 (including every call site) and add a `#[must_use]` attribute on the line \
                 directly above `pub fn load_user`. Change nothing else. Return the complete \
                 updated file only — no explanation.\n\n{EDIT_SOURCE}"
            ),
            check: check_edit,
        },
        BenchTask {
            name: "multiedit",
            prompt: format!(
                "Rename the function `fetch_user` to `load_user` across BOTH Rust files below, \
                 including the call site. Change nothing else. Return both complete updated \
                 files, each in its own fenced code block whose first line is a comment naming \
                 the file: `// file: src/lib.rs` or `// file: src/util.rs`.\n\n\
                 // file: src/lib.rs\n{MULTIEDIT_LIB}\n\n// file: src/util.rs\n{MULTIEDIT_UTIL}"
            ),
            check: check_multiedit,
        },
        BenchTask {
            name: "repair",
            prompt: format!(
                "This Rust function does not compile. The compiler says:\n\n{REPAIR_ERROR}\n\n\
                 Fix it and return only the corrected complete function — no explanation.\n\n\
                 {REPAIR_SOURCE}"
            ),
            check: check_repair,
        },
        BenchTask {
            name: "json",
            prompt: "Respond with ONLY a JSON object (no prose, no code fences) with exactly \
                     these fields: \"cmd\" (string), \"args\" (array of strings), \
                     \"timeout_secs\" (number). It must describe running the shell command \
                     `cargo nextest run` with a 300 second timeout, where cmd is the executable \
                     and args are its arguments."
                .to_string(),
            check: check_json,
        },
    ]
}

fn check_codegen(reply: &str) -> Result<()> {
    let code = extract_code(reply);
    if !code.contains("fn run_length_encode") {
        bail!("reply has no run_length_encode function");
    }
    let harness = r#"
fn main() {
    assert_eq!(run_length_encode("aaabccd"), vec![('a', 3u32), ('b', 1), ('c', 2), ('d', 1)]);
    assert_eq!(run_length_encode(""), Vec::<(char, u32)>::new());
    assert_eq!(run_length_encode("xx"), vec![('x', 2u32)]);
    println!("ok");
}
"#;
    compile_and_run("codegen", &format!("{code}\n{harness}"))
}

fn check_bugfix(reply: &str) -> Result<()> {
    let code = extract_code(reply);
    if !code.contains("fn max_window_sum") {
        bail!("reply has no max_window_sum function");
    }
    let harness = r#"
fn main() {
    assert_eq!(max_window_sum(&[1, 2, 3, 4], 2), 7);
    assert_eq!(max_window_sum(&[5], 1), 5);
    assert_eq!(max_window_sum(&[2, -1, 2, 3, -9], 3), 4);
    assert_eq!(max_window_sum(&[1, 2], 5), 0);
    assert_eq!(max_window_sum(&[], 0), 0);
    println!("ok");
}
"#;
    compile_and_run("bugfix", &format!("{code}\n{harness}"))
}

fn check_edit(reply: &str) -> Result<()> {
    let code = extract_code(reply);
    if !code.contains("#[must_use]") {
        bail!("missing #[must_use] attribute");
    }
    if !code.contains("pub fn load_user") {
        bail!("fetch_user was not renamed to load_user");
    }
    if code.contains("fetch_user") {
        bail!("a fetch_user reference survived the rename");
    }
    if !code.contains("pub fn describe") {
        bail!("unrelated code was dropped from the file");
    }
    compile_lib("edit", &code)
}

fn check_multiedit(reply: &str) -> Result<()> {
    let files = extract_file_blocks(reply);
    let lib = files
        .iter()
        .find(|(path, _)| path.ends_with("lib.rs"))
        .map(|(_, body)| body)
        .context("no src/lib.rs block in reply")?;
    let util = files
        .iter()
        .find(|(path, _)| path.ends_with("util.rs"))
        .map(|(_, body)| body)
        .context("no src/util.rs block in reply")?;
    if lib.contains("fetch_user") || util.contains("fetch_user") {
        bail!("a fetch_user reference survived the cross-file rename");
    }
    if !util.contains("pub fn load_user") {
        bail!("util.rs does not define load_user");
    }
    if !lib.contains("load_user") {
        bail!("lib.rs call site was not renamed");
    }
    // Both files must still compile together as one crate.
    let dir = bench_scratch("multiedit");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir)?;
    std::fs::write(dir.join("lib.rs"), lib)?;
    std::fs::write(dir.join("util.rs"), util)?;
    let compile = std::process::Command::new("rustc")
        .args(["--edition", "2021", "--crate-type", "lib", "--out-dir"])
        .arg(&dir)
        .arg(dir.join("lib.rs"))
        .output()?;
    if !compile.status.success() {
        bail!("doesn't compile: {}", first_error_line(&compile.stderr));
    }
    Ok(())
}

fn check_repair(reply: &str) -> Result<()> {
    let code = extract_code(reply);
    if !code.contains("fn sum_of_evens") {
        bail!("reply has no sum_of_evens function");
    }
    let harness = r#"
fn main() {
    assert_eq!(sum_of_evens(&[1, 2, 3, 4]), 6);
    assert_eq!(sum_of_evens(&[]), 0);
    assert_eq!(sum_of_evens(&[-2, 5]), -2);
    println!("ok");
}
"#;
    compile_and_run("repair", &format!("{code}\n{harness}"))
}

/// Fenced blocks whose first line is a `// file: <path>` comment, as
/// `(path, body-without-marker)` pairs.
fn extract_file_blocks(reply: &str) -> Vec<(String, String)> {
    let reply = strip_thinking(reply);
    let mut blocks = Vec::new();
    for (index, segment) in reply.split("```").enumerate() {
        if index % 2 == 0 {
            continue;
        }
        let body = match segment.split_once('\n') {
            Some((first, rest)) if first.trim().len() <= 12 && !first.trim().contains(' ') => rest,
            _ => segment,
        };
        if let Some(first_line) = body.lines().next()
            && let Some(path) = first_line.trim().strip_prefix("// file:")
        {
            let rest = body.split_once('\n').map(|(_, rest)| rest).unwrap_or("");
            blocks.push((path.trim().to_string(), rest.to_string()));
        }
    }
    blocks
}

fn check_json(reply: &str) -> Result<()> {
    let value = extract_json(reply)?;
    let cmd = value
        .get("cmd")
        .and_then(|v| v.as_str())
        .unwrap_or_default();
    if cmd != "cargo" {
        bail!("cmd is {cmd:?}, expected \"cargo\"");
    }
    let args: Vec<&str> = value
        .get("args")
        .and_then(|v| v.as_array())
        .map(|a| a.iter().filter_map(|v| v.as_str()).collect())
        .unwrap_or_default();
    if args != ["nextest", "run"] {
        bail!("args are {args:?}, expected [\"nextest\", \"run\"]");
    }
    let timeout = value
        .get("timeout_secs")
        .and_then(|v| v.as_f64())
        .unwrap_or_default();
    if timeout != 300.0 {
        bail!("timeout_secs is {timeout}, expected 300");
    }
    Ok(())
}

// ---- Validation helpers ------------------------------------------------------

/// The model's code answer: thinking stripped, then the longest fenced block,
/// or the whole reply when nothing is fenced.
fn extract_code(reply: &str) -> String {
    let reply = strip_thinking(reply);
    let mut best: Option<String> = None;
    for (index, segment) in reply.split("```").enumerate() {
        // Odd segments sit between fences.
        if index % 2 == 1 {
            let body = match segment.split_once('\n') {
                // Drop a language tag line ("rust", "rs", possibly padded).
                Some((first, rest)) if first.trim().len() <= 12 && !first.trim().contains(' ') => {
                    rest
                }
                _ => segment,
            };
            if best.as_ref().is_none_or(|b| body.len() > b.len()) {
                best = Some(body.to_string());
            }
        }
    }
    best.unwrap_or_else(|| reply.trim().to_string())
}

/// The first balanced JSON object in the reply (fences and prose tolerated).
fn extract_json(reply: &str) -> Result<serde_json::Value> {
    let reply = strip_thinking(reply);
    let start = reply.find('{').context("no JSON object in reply")?;
    let mut depth = 0usize;
    let mut in_string = false;
    let mut escaped = false;
    for (offset, ch) in reply[start..].char_indices() {
        if escaped {
            escaped = false;
            continue;
        }
        match ch {
            '\\' if in_string => escaped = true,
            '"' => in_string = !in_string,
            '{' if !in_string => depth += 1,
            '}' if !in_string => {
                depth -= 1;
                if depth == 0 {
                    let candidate = &reply[start..start + offset + ch.len_utf8()];
                    return serde_json::from_str(candidate)
                        .with_context(|| format!("reply is not valid JSON: {candidate}"));
                }
            }
            _ => {}
        }
    }
    bail!("unbalanced JSON object in reply");
}

/// Remove `<think>…</think>` reasoning that some local models emit inline.
fn strip_thinking(reply: &str) -> &str {
    match (reply.find("<think>"), reply.find("</think>")) {
        (Some(open), Some(close)) if open < close => reply[close + "</think>".len()..].trim(),
        _ => reply.trim(),
    }
}

fn bench_scratch(task: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!("hi-team-bench-{}-{task}", std::process::id()))
}

fn rustc_available() -> bool {
    std::process::Command::new("rustc")
        .arg("--version")
        .output()
        .is_ok_and(|out| out.status.success())
}

/// Compile `source` as a binary and run it; pass = clean exit.
fn compile_and_run(task: &str, source: &str) -> Result<()> {
    let dir = bench_scratch(task);
    std::fs::create_dir_all(&dir)?;
    let main_rs = dir.join("main.rs");
    std::fs::write(&main_rs, source)?;
    let binary = dir.join("bench-bin");
    let compile = std::process::Command::new("rustc")
        .args(["--edition", "2021", "-o"])
        .arg(&binary)
        .arg(&main_rs)
        .output()?;
    if !compile.status.success() {
        bail!("doesn't compile: {}", first_error_line(&compile.stderr));
    }
    let run = std::process::Command::new(&binary).output()?;
    if !run.status.success() {
        bail!("wrong behavior: {}", first_error_line(&run.stderr));
    }
    Ok(())
}

/// Compile `source` as a library; pass = compiles cleanly.
fn compile_lib(task: &str, source: &str) -> Result<()> {
    let dir = bench_scratch(task);
    std::fs::create_dir_all(&dir)?;
    let lib_rs = dir.join("lib.rs");
    std::fs::write(&lib_rs, source)?;
    let compile = std::process::Command::new("rustc")
        .args(["--edition", "2021", "--crate-type", "lib", "--out-dir"])
        .arg(&dir)
        .arg(&lib_rs)
        .output()?;
    if !compile.status.success() {
        bail!("doesn't compile: {}", first_error_line(&compile.stderr));
    }
    Ok(())
}

fn first_error_line(stderr: &[u8]) -> String {
    let text = String::from_utf8_lossy(stderr);
    text.lines()
        .find(|line| line.contains("error") || line.contains("panicked"))
        .unwrap_or_else(|| text.lines().next().unwrap_or("unknown error"))
        .trim()
        .to_string()
}

// ---- Reporting ---------------------------------------------------------------

fn print_summary(reports: &[ModelReport], tasks: &[BenchTask]) {
    println!(
        "results ({} tasks, compiled + asserted locally):",
        tasks.len()
    );
    let name_width = reports
        .iter()
        .map(|r| r.model.len())
        .max()
        .unwrap_or(8)
        .max(8);
    let mut header = format!("{:<name_width$}", "model");
    for task in tasks {
        header.push_str(&format!("  {:<9}", task.name));
    }
    header.push_str("  score   avg tok/s  ttft    setup");
    println!("{header}");
    for report in reports {
        let mut row = format!("{:<name_width$}", report.model);
        for task in tasks {
            let mark = report
                .results
                .iter()
                .find(|r| r.task == task.name)
                .map(|r| if r.pass { "PASS" } else { "FAIL" })
                .unwrap_or("—");
            row.push_str(&format!("  {mark:<9}"));
        }
        row.push_str(&format!(
            "  {}/{}     {:>6.1}   {:>5.1}s  {:>4.0}s",
            report.passed(),
            report.results.len(),
            report.avg_tokens_per_sec(),
            report.median_ttft_secs(),
            report.setup_secs,
        ));
        println!("{row}");
    }
    // Per-task winners: every model that passed, fastest generation first.
    for task in tasks {
        let mut passed: Vec<(&str, f64)> = reports
            .iter()
            .filter_map(|report| {
                report
                    .results
                    .iter()
                    .find(|r| r.task == task.name && r.pass)
                    .map(|r| (report.model.as_str(), r.latency_secs))
            })
            .collect();
        passed.sort_by(|a, b| a.1.total_cmp(&b.1));
        let line = if passed.is_empty() {
            "nobody passed".to_string()
        } else {
            passed
                .iter()
                .map(|(model, secs)| format!("{model} ({secs:.1}s)"))
                .collect::<Vec<_>>()
                .join(", ")
        };
        println!("  {:<9} → {line}", task.name);
    }
}

fn save_report(reports: &[ModelReport]) -> Option<std::path::PathBuf> {
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .ok()?
        .as_secs();
    let dir = std::env::var_os("HOME")
        .map(std::path::PathBuf::from)?
        .join(".hi")
        .join("bench");
    std::fs::create_dir_all(&dir).ok()?;
    let path = dir.join(format!("team-bench-{stamp}.json"));
    let body = serde_json::to_string_pretty(&serde_json::json!({
        "ran_at_unix_secs": stamp,
        "models": reports,
    }))
    .ok()?;
    std::fs::write(&path, body).ok()?;
    Some(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_code_prefers_the_largest_fenced_block_and_strips_thinking() {
        let reply = "<think>plan plan</think>Here you go:\n```rust\npub fn a() {}\n```\nand\n```rust\npub fn bigger() { let x = 1; }\n```";
        assert_eq!(extract_code(reply), "pub fn bigger() { let x = 1; }\n");
        assert_eq!(extract_code("pub fn plain() {}"), "pub fn plain() {}");
    }

    #[test]
    fn extract_json_tolerates_prose_and_fences() {
        let value = extract_json("Sure!\n```json\n{\"cmd\": \"cargo\", \"args\": [\"nextest\", \"run\"], \"timeout_secs\": 300}\n```").unwrap();
        assert_eq!(value["cmd"], "cargo");
        assert!(extract_json("no json here").is_err());
    }

    #[test]
    fn json_check_is_strict_about_values() {
        assert!(
            check_json(
                "{\"cmd\": \"cargo\", \"args\": [\"nextest\", \"run\"], \"timeout_secs\": 300}"
            )
            .is_ok()
        );
        assert!(
            check_json("{\"cmd\": \"cargo nextest run\", \"args\": [], \"timeout_secs\": 300}")
                .is_err()
        );
        assert!(
            check_json(
                "{\"cmd\": \"cargo\", \"args\": [\"nextest\", \"run\"], \"timeout_secs\": 30}"
            )
            .is_err()
        );
    }

    #[test]
    fn multiedit_check_requires_both_files_renamed_and_compiling() {
        let reply = "```rust\n// file: src/lib.rs\nmod util;\n\npub fn describe(id: u64) -> String { match util::load_user(id) { Some(user) => user, None => \"anonymous\".to_string() } }\n```\nand\n```rust\n// file: src/util.rs\npub fn load_user(id: u64) -> Option<String> { if id == 0 { return None; } Some(format!(\"user-{id}\")) }\n```";
        assert!(
            check_multiedit(reply).is_ok(),
            "{:?}",
            check_multiedit(reply)
        );
        let missing = reply.replace("// file: src/util.rs", "// file: src/other.rs");
        assert!(
            check_multiedit(&missing).is_err(),
            "a dropped file is rejected"
        );
        let stale = reply.replace("util::load_user", "util::fetch_user");
        assert!(
            check_multiedit(&stale).is_err(),
            "a stale call site is rejected"
        );
    }

    #[test]
    fn edit_check_requires_a_complete_faithful_rename() {
        let good = "#[must_use]\npub fn load_user(id: u64) -> Option<String> { if id == 0 { return None; } Some(format!(\"user-{id}\")) }\npub fn describe(id: u64) -> String { match load_user(id) { Some(user) => user, None => \"anonymous\".to_string() } }";
        assert!(check_edit(good).is_ok());
        let stale_call_site = good.replace("match load_user", "match fetch_user");
        assert!(check_edit(&stale_call_site).is_err());
    }
}
