use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow, bail, ensure};
use hi_ai::test_support::{
    ChatStep, RequestMatcher, ScriptedOpenAiServer, ScriptedResponse, ScriptedToolCall,
};
use serde::Serialize;
use serde_json::{Value, json};

use crate::cli::RunMode;
use crate::discovery::{self, DiscoveredScenario};
use crate::isolation::{
    IsolationEvidence, IsolationPolicy, IsolationSnapshot, STAGED_CANDIDATE_DIR,
};
use crate::live_route::LiveRoute;
use crate::pty::{PtyProcess, RawTerminal, SpawnSpec, collect_marked_processes};
use crate::scenario::{
    Action, Assertion, GitState, PlanSeedStatus, ProviderResponse, QuiescentSource, RecordSource,
    Scenario, StreamTerminal, normalize_workspace_listing_path, validate_relative_path,
};

const SCRIPTED_FIRST_FRAME: Duration = Duration::from_secs(15);
const LIVE_FIRST_FRAME: Duration = Duration::from_secs(20);
const LIVE_TRANSITION: Duration = Duration::from_secs(30);
const POLL_INTERVAL: Duration = Duration::from_millis(20);

#[derive(Clone, Debug)]
pub(crate) struct SuiteOptions {
    pub hi_bin: PathBuf,
    pub suite: PathBuf,
    pub mode: RunMode,
    pub tags: Vec<String>,
    pub artifacts: PathBuf,
    pub jobs: usize,
    pub keep: bool,
}

#[derive(Clone, Debug)]
pub(crate) struct CaseOptions {
    pub hi_bin: PathBuf,
    pub artifacts: PathBuf,
    pub mode: RunMode,
    /// Recorded route for live replay. `None` resolves the current live route
    /// from the environment. The API key is always read separately.
    pub live_route: Option<LiveRoute>,
    pub keep: bool,
    pub seed: Option<u64>,
    pub sandbox_requirement: SandboxRequirement,
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum SandboxRequirement {
    Enforced,
    #[cfg(test)]
    UnitTestUnenforced,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum CaseStatus {
    Passed,
    Failed,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum CaseFailureKind {
    Scenario,
    TimedOut,
    Cancelled,
    Crashed,
    InfrastructureFailure,
    InfrastructureLoop,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct CaseReport {
    pub name: String,
    pub status: CaseStatus,
    pub failure_kind: Option<CaseFailureKind>,
    pub duration_ms: u64,
    pub failure: Option<String>,
    pub artifact_dir: PathBuf,
    pub provider_request_count: usize,
    pub provider_chat_request_count: usize,
    pub provider_accepted_request_count: usize,
    pub provider_response_status_counts: BTreeMap<u16, usize>,
}

#[derive(Clone, Debug, Default)]
struct ProviderEvidence {
    request_count: usize,
    chat_request_count: usize,
    accepted_request_count: usize,
    response_status_counts: BTreeMap<u16, usize>,
}

pub(crate) fn run_suite(options: SuiteOptions) -> Result<()> {
    let discovered = discovery::discover(&options.suite)?;
    let selected = discovered
        .into_iter()
        .filter(|case| case.scenario.has_tag(&options.tags))
        .collect::<Vec<_>>();
    ensure!(
        !selected.is_empty(),
        "no scenarios under {} matched tag filter {:?}",
        options.suite.display(),
        options.tags
    );
    fs::create_dir_all(&options.artifacts)
        .with_context(|| format!("creating artifact root {}", options.artifacts.display()))?;
    let live_route = match options.mode {
        RunMode::Scripted => None,
        RunMode::Live => Some(resolve_live_route(None)?),
    };

    let queue = Arc::new(Mutex::new(VecDeque::from(selected)));
    let reports = Arc::new(Mutex::new(Vec::<CaseReport>::new()));
    let workers = options.jobs.min(queue.lock().expect("queue lock").len());
    std::thread::scope(|scope| {
        for _ in 0..workers {
            let queue = Arc::clone(&queue);
            let reports = Arc::clone(&reports);
            let case_options = CaseOptions {
                hi_bin: options.hi_bin.clone(),
                artifacts: options.artifacts.clone(),
                mode: options.mode,
                live_route: live_route.clone(),
                keep: options.keep,
                seed: None,
                sandbox_requirement: SandboxRequirement::Enforced,
            };
            scope.spawn(move || {
                loop {
                    let next = queue.lock().expect("queue lock").pop_front();
                    let Some(case) = next else { break };
                    let name = case.scenario.name.clone();
                    eprintln!("[ RUN      ] {name}");
                    let report = run_discovered(case, &case_options);
                    match &report.status {
                        CaseStatus::Passed => {
                            eprintln!("[       OK ] {name} ({} ms)", report.duration_ms)
                        }
                        CaseStatus::Failed => eprintln!(
                            "[  FAILED  ] {name}: {}",
                            report.failure.as_deref().unwrap_or("unknown failure")
                        ),
                    }
                    reports.lock().expect("reports lock").push(report);
                }
            });
        }
    });

    let mut reports = Arc::try_unwrap(reports)
        .map_err(|_| anyhow!("suite report workers did not release state"))?
        .into_inner()
        .map_err(|_| anyhow!("suite report lock was poisoned"))?;
    reports.sort_by(|left, right| left.name.cmp(&right.name));
    let passed = reports
        .iter()
        .filter(|report| matches!(report.status, CaseStatus::Passed))
        .count();
    let total = reports.len();
    let summary = suite_summary(options.mode, live_route.as_ref(), &reports);
    let summary_redactions = vec![
        "hi-smoke-test-key".to_owned(),
        live_value("HI_API_KEY").unwrap_or_default(),
    ];
    crate::artifacts::write_suite_summary(&options.artifacts, &summary, &summary_redactions)?;
    if passed != total {
        bail!(
            "{} of {} TUI smoke scenario(s) failed",
            total - passed,
            total
        );
    }
    println!("{} TUI smoke scenario(s) passed", passed);
    Ok(())
}

fn suite_summary(mode: RunMode, live_route: Option<&LiveRoute>, reports: &[CaseReport]) -> Value {
    let passed = reports
        .iter()
        .filter(|report| matches!(report.status, CaseStatus::Passed))
        .count();
    let total = reports.len();
    let crash_count = reports
        .iter()
        .filter(|report| report.failure_kind == Some(CaseFailureKind::Crashed))
        .count();
    let infrastructure_loop_count = reports
        .iter()
        .filter(|report| report.failure_kind == Some(CaseFailureKind::InfrastructureLoop))
        .count();
    let infrastructure_failure_count = reports
        .iter()
        .filter(|report| report.failure_kind == Some(CaseFailureKind::InfrastructureFailure))
        .count();
    let provider_request_count = reports
        .iter()
        .map(|report| report.provider_request_count)
        .sum::<usize>();
    let provider_chat_request_count = reports
        .iter()
        .map(|report| report.provider_chat_request_count)
        .sum::<usize>();
    let provider_accepted_request_count = reports
        .iter()
        .map(|report| report.provider_accepted_request_count)
        .sum::<usize>();
    let mut provider_response_status_counts = BTreeMap::<u16, usize>::new();
    for report in reports {
        for (status, count) in &report.provider_response_status_counts {
            *provider_response_status_counts.entry(*status).or_default() += count;
        }
    }
    let cases = reports
        .iter()
        .map(|report| {
            let artifact_dir = report
                .artifact_dir
                .file_name()
                .map(|name| name.to_string_lossy().into_owned())
                .unwrap_or_else(|| "case-artifact".to_owned());
            json!({
                "name": &report.name,
                "status": &report.status,
                "failure_kind": report.failure_kind,
                "duration_ms": report.duration_ms,
                "failure": &report.failure,
                "provider_request_count": report.provider_request_count,
                "provider_chat_request_count": report.provider_chat_request_count,
                "provider_accepted_request_count": report.provider_accepted_request_count,
                "provider_response_status_counts": &report.provider_response_status_counts,
                // Case artifacts are siblings of this summary. Never persist
                // the caller's absolute artifact root or host path.
                "artifact_dir": artifact_dir,
            })
        })
        .collect::<Vec<_>>();
    json!({
        "schema_version": 1,
        "mode": format!("{mode:?}").to_ascii_lowercase(),
        "live_route": live_route,
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "scenario_pass_rate": if total == 0 { 0.0 } else { passed as f64 * 100.0 / total as f64 },
        "crash_count": crash_count,
        "infrastructure_loop_count": infrastructure_loop_count,
        "infrastructure_failure_count": infrastructure_failure_count,
        "provider_request_count": provider_request_count,
        "provider_chat_request_count": provider_chat_request_count,
        "provider_accepted_request_count": provider_accepted_request_count,
        "provider_response_status_counts": provider_response_status_counts,
        "cases": cases,
    })
}

pub(crate) fn replay(hi_bin: &Path, replay: &Path, artifacts: &Path, keep: bool) -> Result<()> {
    let scenario = Scenario::parse(replay)?;
    let metadata = replay_metadata(replay)?;
    let mode = metadata.mode;
    let live_route = match mode {
        RunMode::Scripted => None,
        RunMode::Live => Some(resolve_live_route(metadata.live_route.as_ref())?),
    };
    let report = run_discovered(
        DiscoveredScenario {
            path: replay.to_path_buf(),
            scenario,
        },
        &CaseOptions {
            hi_bin: hi_bin.to_path_buf(),
            artifacts: artifacts.to_path_buf(),
            mode,
            live_route,
            keep,
            seed: None,
            sandbox_requirement: SandboxRequirement::Enforced,
        },
    );
    if matches!(report.status, CaseStatus::Failed) {
        bail!(
            "replay failed: {} (evidence: {})",
            report.failure.as_deref().unwrap_or("unknown failure"),
            report.artifact_dir.display()
        );
    }
    println!("replay passed: {}", report.name);
    Ok(())
}

pub(crate) fn run_scenario(scenario: Scenario, options: &CaseOptions) -> CaseReport {
    run_discovered(
        DiscoveredScenario {
            path: scenario.source_dir.join("scenario.toml"),
            scenario,
        },
        options,
    )
}

fn run_discovered(case: DiscoveredScenario, options: &CaseOptions) -> CaseReport {
    let started = Instant::now();
    let source_path = case.path.clone();
    let case_dir_name = unique_case_dir(&case.scenario.name, options.seed);
    let artifact_dir = options.artifacts.join(&case_dir_name);
    let mut provider_evidence = ProviderEvidence::default();
    let result = CaseRuntime::new(case.scenario.clone(), options)
        .and_then(|mut runtime| {
            let execution = runtime.execute();
            // A failed action may leave the PTY leader or one of its tools
            // alive. Stop the full group before taking the final isolation
            // snapshot or strictly parsing the flushed trace so evidence is
            // stable and no late write can race either invariant.
            let cleanup = runtime.prepare_failure_evidence(execution.is_err());
            // Route evidence remains a hard live-mode infrastructure
            // invariant even when an earlier action or assertion failed. It
            // is read after cleanup so a concurrently appended JSONL record
            // cannot create a transient parse failure.
            let result = merge_live_route_invariant(
                execution,
                runtime.check_live_provider_route_invariant(),
            );
            // Forced termination can itself leave an active turn without a
            // terminal settlement or omit normal shutdown. The invariants
            // below deliberately check only evidence that remains valid after
            // a harness kill: strict JSONL state, monotonic/typed trace rows,
            // exact queue accounting, non-overlapping starts, and no autonomous restart
            // after a failed/cancelled settlement.
            let state =
                runtime.check_post_failure_state_invariants(result.is_err() || cleanup.is_err());
            let result = merge_post_failure_state_invariant(result, state);
            let result =
                merge_isolation_invariant(result, runtime.check_isolation_mutation_invariant());
            // Merge cleanup after every later audit so a surviving descendant
            // or failed process inspection remains the authoritative root
            // cause while retaining route, state, isolation, and action
            // context above it.
            let result = merge_failure_cleanup_invariant(result, cleanup);
            provider_evidence = summarize_provider_requests(&runtime.provider_requests());
            runtime.finish_bundle(&artifact_dir, started.elapsed(), result.as_ref().err())?;
            result
        })
        .with_context(|| format!("scenario {}", source_path.display()));
    match result {
        Ok(()) => CaseReport {
            name: case.scenario.name,
            status: CaseStatus::Passed,
            failure_kind: None,
            duration_ms: millis(started.elapsed()),
            failure: None,
            artifact_dir,
            provider_request_count: provider_evidence.request_count,
            provider_chat_request_count: provider_evidence.chat_request_count,
            provider_accepted_request_count: provider_evidence.accepted_request_count,
            provider_response_status_counts: provider_evidence.response_status_counts,
        },
        Err(error) => {
            let failure_kind = classify_failure(&error);
            // Initialization and detailed-bundle failures still get a minimal,
            // self-contained replay. Validate the partial bundle rather than
            // trusting an early summary write as proof it is complete.
            let repair_error = if !minimal_replay_is_complete(&artifact_dir) {
                write_initialization_failure(
                    &artifact_dir,
                    &case.scenario,
                    options,
                    started.elapsed(),
                    &error,
                )
                .err()
            } else {
                None
            };
            let failure = match repair_error {
                Some(repair_error) => format!(
                    "{error:#}; minimal replay bundle could not be repaired: {repair_error:#}"
                ),
                None => format!("{error:#}"),
            };
            CaseReport {
                name: case.scenario.name,
                status: CaseStatus::Failed,
                failure_kind: Some(failure_kind),
                duration_ms: millis(started.elapsed()),
                failure: Some(failure),
                artifact_dir,
                provider_request_count: provider_evidence.request_count,
                provider_chat_request_count: provider_evidence.chat_request_count,
                provider_accepted_request_count: provider_evidence.accepted_request_count,
                provider_response_status_counts: provider_evidence.response_status_counts,
            }
        }
    }
}

fn summarize_provider_requests(requests: &[Value]) -> ProviderEvidence {
    let mut evidence = ProviderEvidence {
        request_count: requests.len(),
        ..ProviderEvidence::default()
    };
    for request in requests {
        let is_chat_request = match request.get("path").and_then(Value::as_str) {
            Some(path) => {
                request.get("method").and_then(Value::as_str) == Some("POST")
                    && path
                        .split('?')
                        .next()
                        .is_some_and(|path| path.ends_with("/chat/completions"))
            }
            None => {
                request.get("provider").and_then(Value::as_str).is_some()
                    && request.get("model").and_then(Value::as_str).is_some()
                    && request
                        .get("request_attempt")
                        .and_then(Value::as_u64)
                        .is_some()
            }
        };
        evidence.chat_request_count += usize::from(is_chat_request);
        evidence.accepted_request_count +=
            usize::from(request.get("accepted").and_then(Value::as_bool) == Some(true));
        if let Some(status) = request
            .get("response_status")
            .and_then(Value::as_u64)
            .and_then(|status| u16::try_from(status).ok())
        {
            *evidence.response_status_counts.entry(status).or_default() += 1;
        }
    }
    evidence
}

struct CaseRuntime {
    scenario: Scenario,
    options: CaseOptions,
    isolation: Option<tempfile::TempDir>,
    workspace: PathBuf,
    initial_workspace: PathBuf,
    home: PathBuf,
    xdg_config: PathBuf,
    xdg_data: PathBuf,
    xdg_state: PathBuf,
    xdg_cache: PathBuf,
    temp_dir: PathBuf,
    session_path: PathBuf,
    events_path: PathBuf,
    event_cursor: usize,
    config_path: PathBuf,
    outer_sandbox: hi_tools::sandbox::SandboxProfile,
    isolation_baseline: IsolationSnapshot,
    isolation_evidence: Option<IsolationEvidence>,
    live_route: Option<LiveRoute>,
    provider: Option<ScriptedOpenAiServer>,
    process: Option<PtyProcess>,
    historical_raw: RawTerminal,
    screens: BTreeMap<String, String>,
    assertions: Vec<Value>,
    timings: BTreeMap<String, u64>,
    exit_code: Option<u32>,
    process_ids: Vec<u32>,
    process_groups: Vec<i32>,
    process_markers: Vec<String>,
    observed_descendant_pids: BTreeSet<i32>,
    observed_descendant_groups: BTreeSet<i32>,
    leaked_processes: Vec<Value>,
    observed_active_turn: Option<(u64, Instant)>,
    scenario_deadline: Instant,
}

impl CaseRuntime {
    fn new(scenario: Scenario, options: &CaseOptions) -> Result<Self> {
        scenario.validate()?;
        fs::create_dir_all(&options.artifacts)?;
        let scratch = options.artifacts.join(".work");
        fs::create_dir_all(&scratch)?;
        let isolation = tempfile::Builder::new()
            .prefix(&format!("{}-", safe_name(&scenario.name)))
            .tempdir_in(&scratch)
            .context("creating isolated smoke workspace")?;
        let root = isolation.path();
        let workspace = root.join("workspace");
        let initial_workspace = root.join("initial-workspace");
        let home = root.join("home");
        let xdg_config = root.join("xdg/config");
        let xdg_data = root.join("xdg/data");
        let xdg_state = root.join("xdg/state");
        let xdg_cache = root.join("xdg/cache");
        let temp_dir = root.join("tmp");
        let session_path = root.join("session/session.jsonl");
        let events_path = root.join("events/tui.jsonl");
        let config_path = root.join("config/hi.toml");
        for directory in [
            &workspace,
            &initial_workspace,
            &home,
            &xdg_config,
            &xdg_data,
            &xdg_state,
            &xdg_cache,
            &temp_dir,
            session_path.parent().expect("session parent"),
            events_path.parent().expect("events parent"),
            config_path.parent().expect("config parent"),
        ] {
            fs::create_dir_all(directory)
                .with_context(|| format!("creating isolation directory {}", directory.display()))?;
        }
        fs::write(
            &config_path,
            "# generated by hi-smoke\n[sync]\nmode = \"off\"\n",
        )?;
        if let Some(fixture) = &scenario.workspace.fixture {
            copy_tree(&scenario.source_dir.join(fixture), &workspace)?;
        }
        copy_tree(&workspace, &initial_workspace)?;
        initialize_git(&workspace, scenario.workspace.git)?;
        initialize_git(&initial_workspace, scenario.workspace.git)?;
        write_session_seed(&session_path, &scenario)?;
        let (outer_sandbox, hi_bin) = match options.sandbox_requirement {
            SandboxRequirement::Enforced => {
                let hi_bin = stage_sandbox_candidate(root, &options.hi_bin)?;
                (smoke_sandbox_profile(root, &hi_bin)?, hi_bin)
            }
            #[cfg(test)]
            SandboxRequirement::UnitTestUnenforced => (
                hi_tools::sandbox::SandboxProfile::new(hi_tools::sandbox::SandboxPolicy::Off, &[]),
                options.hi_bin.clone(),
            ),
        };
        let mut options = options.clone();
        options.hi_bin = hi_bin;
        let isolation_baseline =
            crate::isolation::capture(root).context("capturing pre-run isolation evidence")?;

        let live_route = match options.mode {
            RunMode::Scripted => None,
            RunMode::Live => Some(resolve_live_route(options.live_route.as_ref())?),
        };
        let provider = match options.mode {
            RunMode::Scripted => Some(start_provider(&scenario)?),
            RunMode::Live => None,
        };
        let timeout = match options.mode {
            RunMode::Scripted => Duration::from_millis(scenario.timeout_ms),
            RunMode::Live => {
                Duration::from_millis(scenario.timeout_ms).max(Duration::from_secs(600))
            }
        };
        Ok(Self {
            scenario,
            options,
            isolation: Some(isolation),
            workspace,
            initial_workspace,
            home,
            xdg_config,
            xdg_data,
            xdg_state,
            xdg_cache,
            temp_dir,
            session_path,
            events_path,
            event_cursor: 0,
            config_path,
            outer_sandbox,
            isolation_baseline,
            isolation_evidence: None,
            live_route,
            provider,
            process: None,
            historical_raw: RawTerminal::default(),
            screens: BTreeMap::new(),
            assertions: Vec::new(),
            timings: BTreeMap::new(),
            exit_code: None,
            process_ids: Vec::new(),
            process_groups: Vec::new(),
            process_markers: Vec::new(),
            observed_descendant_pids: BTreeSet::new(),
            observed_descendant_groups: BTreeSet::new(),
            leaked_processes: Vec::new(),
            observed_active_turn: None,
            scenario_deadline: Instant::now() + timeout,
        })
    }

    fn execute(&mut self) -> Result<()> {
        self.spawn_hi()?;
        for (index, action) in self.scenario.actions.clone().into_iter().enumerate() {
            self.ensure_time(format!("before action {index}"))?;
            self.observe_current_descendants()?;
            let started = Instant::now();
            self.execute_action(&action)
                .with_context(|| format!("action {index} ({})", action_name(&action)))?;
            self.observe_current_descendants()?;
            self.timings.insert(
                format!("action_{index}_{}", action_name(&action)),
                millis(started.elapsed()),
            );
        }

        if self.process.is_some() && self.exit_code.is_none() {
            self.quit()?;
        }
        self.check_hard_invariants()?;
        for (index, assertion) in self.scenario.assertions.clone().into_iter().enumerate() {
            let result = self.evaluate_assertion(&assertion);
            self.assertions.push(json!({
                "index": index,
                "kind": assertion_name(&assertion),
                "passed": result.is_ok(),
                "failure": result.as_ref().err().map(|error| format!("{error:#}")),
            }));
            result
                .with_context(|| format!("assertion {index} ({})", assertion_name(&assertion)))?;
        }
        Ok(())
    }

    fn spawn_hi(&mut self) -> Result<()> {
        ensure!(self.process.is_none(), "hi process is already running");
        // A restarted process appends to the same semantic trace. Anchor every
        // transition wait after the records that belonged to prior processes.
        self.event_cursor = read_jsonl(&self.events_path)?.len();
        let (provider_name, model, api_key) = match self.options.mode {
            RunMode::Scripted => (
                "openai".to_string(),
                "test-model".to_string(),
                "hi-smoke-test-key".to_string(),
            ),
            RunMode::Live => {
                let route = self
                    .live_route
                    .as_ref()
                    .ok_or_else(|| anyhow!("live mode route was not resolved"))?;
                (
                    route.provider.clone(),
                    route.model.clone(),
                    live_value("HI_API_KEY")
                        .ok_or_else(|| anyhow!("live mode requires non-empty HI_API_KEY"))?,
                )
            }
        };
        let (mut args, (credential_name, credential_value)) =
            provider_launch_parts(provider_name.as_str(), model, api_key);
        args.extend([
            "--config".into(),
            self.config_path.display().to_string(),
            "--session-file".into(),
            self.session_path.display().to_string(),
            "--tui-events-jsonl".into(),
            self.events_path.display().to_string(),
            "--no-auto-compact".into(),
            "--no-finalize".into(),
            "--no-memory".into(),
            "--no-rsi".into(),
            "--no-tasks".into(),
            "--review".into(),
            "off".into(),
            "--lsp".into(),
            "off".into(),
            "--turn-deadline".into(),
            self.scenario
                .hi
                .turn_deadline_secs
                .unwrap_or(match self.options.mode {
                    RunMode::Scripted => 30,
                    RunMode::Live => 240,
                })
                .to_string(),
        ]);
        let base_url = match &self.provider {
            Some(provider) => provider.v1_url(),
            None => self
                .live_route
                .as_ref()
                .map(|route| route.base_url.clone())
                .ok_or_else(|| anyhow!("live mode route was not resolved"))?,
        };
        args.push("--base-url".into());
        args.push(base_url);
        if !self
            .scenario
            .hi
            .args
            .iter()
            .any(|argument| matches!(argument.as_str(), "--verify" | "--no-verify"))
        {
            args.push("--no-verify".into());
        }
        args.extend(self.scenario.hi.args.iter().cloned());

        let mut env = BTreeMap::from([
            ("HOME".into(), self.home.display().to_string()),
            (
                "XDG_CONFIG_HOME".into(),
                self.xdg_config.display().to_string(),
            ),
            ("XDG_DATA_HOME".into(), self.xdg_data.display().to_string()),
            (
                "XDG_STATE_HOME".into(),
                self.xdg_state.display().to_string(),
            ),
            (
                "XDG_CACHE_HOME".into(),
                self.xdg_cache.display().to_string(),
            ),
            ("TMPDIR".into(), self.temp_dir.display().to_string()),
            (
                "PATH".into(),
                std::env::var("PATH")
                    .unwrap_or_else(|_| "/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin".into()),
            ),
            ("SHELL".into(), "/bin/sh".into()),
            ("LANG".into(), "C".into()),
            ("LC_ALL".into(), "C".into()),
            ("TERM".into(), "xterm-256color".into()),
            ("COLORTERM".into(), "truecolor".into()),
            ("HI_SKIP_TUTORIAL".into(), "1".into()),
            ("HI_SUGGEST_NEXT_PROMPT".into(), "0".into()),
            ("HI_DISABLE_UPDATE_CHECK".into(), "1".into()),
            ("HI_DISABLE_FEEDBACK".into(), "1".into()),
            ("HI_TRACE_CAPTURE".into(), "off".into()),
            // Smoke scenarios exercise agent shell/tool execution under a
            // real write-confining sandbox. Scenarios may not override any
            // HI_* control variable, so these remain authoritative.
            ("HI_SANDBOX".into(), "workspace".into()),
            // The complete TUI process runs inside the harness-owned outer
            // profile. Tool descendants inherit it; nested Seatbelt profiles
            // are unsupported and redundant.
            ("HI_SANDBOXED".into(), "1".into()),
            (
                "HI_STATE_ROOT".into(),
                self.xdg_state.join("hi").display().to_string(),
            ),
            (
                "HI_TRUST_STORE".into(),
                self.xdg_config
                    .join("hi/trusted_folders.toml")
                    .display()
                    .to_string(),
            ),
            (
                "HI_ME_MD".into(),
                self.xdg_config.join("hi/me.md").display().to_string(),
            ),
            ("RUST_BACKTRACE".into(), "1".into()),
            ("RUSTUP_SKIP_UPDATE_CHECK".into(), "1".into()),
            ("CARGO_NET_OFFLINE".into(), "true".into()),
        ]);
        // Keep HOME/XDG isolated, but let rustup resolve the already-installed
        // host toolchain for scenarios that exercise hi's real verification
        // path. This is read-only in smoke runs (update checks and Cargo
        // networking are disabled); build outputs remain under the fixture.
        let rustup_home = std::env::var_os("RUSTUP_HOME")
            .map(PathBuf::from)
            .or_else(|| {
                std::env::var_os("HOME")
                    .map(PathBuf::from)
                    .map(|path| path.join(".rustup"))
            })
            .filter(|path| path.is_dir());
        if let Some(path) = rustup_home {
            env.insert("RUSTUP_HOME".into(), path.display().to_string());
        }
        if let Some(toolchain) = std::env::var_os("RUSTUP_TOOLCHAIN") {
            env.insert(
                "RUSTUP_TOOLCHAIN".into(),
                toolchain.to_string_lossy().into_owned(),
            );
        }
        #[cfg(target_os = "linux")]
        if let Some(pipe_wrap) = std::env::var_os("HI_PIPE_WRAP") {
            let pipe_wrap = PathBuf::from(pipe_wrap)
                .canonicalize()
                .context("canonicalizing operator-provided HI_PIPE_WRAP")?;
            env.insert("HI_PIPE_WRAP".into(), pipe_wrap.display().to_string());
        }
        extend_scenario_env_with_credential(
            &mut env,
            &self.scenario.hi.env,
            (credential_name, credential_value),
        );
        let (wrapped_program, wrapped_args) = self.outer_sandbox.wrap_program_in(
            self.options.hi_bin.as_os_str(),
            args.iter().map(String::as_str),
            &self.workspace,
        );
        let wrapped_program = PathBuf::from(wrapped_program);
        let wrapped_args = wrapped_args
            .into_iter()
            .map(|argument| argument.to_string_lossy().into_owned())
            .collect::<Vec<_>>();
        let process = PtyProcess::spawn(SpawnSpec {
            executable: &wrapped_program,
            args: &wrapped_args,
            cwd: &self.workspace,
            env: &env,
            cols: self.scenario.terminal.cols,
            rows: self.scenario.terminal.rows,
        })?;
        if let Some(pid) = process.process_id() {
            self.process_ids.push(pid);
        }
        self.process_markers.push(process.run_marker().to_owned());
        #[cfg(unix)]
        if let Some(group) = process.process_group() {
            self.process_groups.push(group);
        }
        self.process = Some(process);
        let first_frame = match self.options.mode {
            RunMode::Scripted => SCRIPTED_FIRST_FRAME,
            RunMode::Live => LIVE_FIRST_FRAME,
        };
        let started = Instant::now();
        self.wait_for_event(
            &BTreeMap::from([("/event".into(), json!("ready"))]),
            &BTreeMap::new(),
            first_frame,
        )
        .context("waiting for first full-screen TUI frame")?;
        self.timings
            .insert("first_frame_ms".into(), millis(started.elapsed()));
        Ok(())
    }

    fn execute_action(&mut self, action: &Action) -> Result<()> {
        match action {
            Action::SendLine { text } => self.process_mut()?.send_line(text),
            Action::SendKey { key } => self.process_mut()?.send_bytes(key.bytes()),
            Action::Resize { cols, rows } => {
                self.process_mut()?.resize(*cols, *rows)?;
                // Do not race subsequent keystrokes against SIGWINCH delivery.
                // Waiting on typed evidence keeps this deterministic without a
                // timing sleep and still exposes an unresponsive TUI.
                self.wait_for_event(
                    &BTreeMap::from([
                        ("/event".into(), json!("resized")),
                        ("/data/width".into(), json!(cols)),
                        ("/data/height".into(), json!(rows)),
                    ]),
                    &BTreeMap::new(),
                    self.timeout_for(5_000),
                )
            }
            Action::WaitEvent {
                equals,
                contains,
                timeout_ms,
            } => self.wait_for_event(equals, contains, self.timeout_for(*timeout_ms)),
            Action::WaitEventAbsent {
                equals,
                contains,
                duration_ms,
            } => self.wait_for_event_absence(equals, contains, Duration::from_millis(*duration_ms)),
            Action::WaitProviderRequest { count, timeout_ms } => {
                self.wait_for_provider_requests(*count, self.timeout_for(*timeout_ms))
            }
            Action::WaitFile {
                path,
                exists,
                timeout_ms,
            } => self.wait_for_file(path, *exists, self.timeout_for(*timeout_ms)),
            Action::WaitProcess {
                command_contains,
                at_least,
                timeout_ms,
            } => self.wait_for_marked_process(
                command_contains,
                *at_least,
                self.timeout_for(*timeout_ms),
            ),
            Action::WaitQuiescent {
                source,
                quiet_ms,
                timeout_ms,
            } => self.wait_for_quiescence(
                *source,
                Duration::from_millis(*quiet_ms),
                self.timeout_for(*timeout_ms),
            ),
            Action::ReleaseGate { gate } => {
                let provider = self
                    .provider
                    .as_ref()
                    .ok_or_else(|| anyhow!("release_gate is unavailable in live mode"))?;
                ensure!(
                    provider.release_gate(gate),
                    "provider gate {gate:?} was already released"
                );
                Ok(())
            }
            Action::CaptureScreen { name } => {
                let screen = normalize_screen(&self.process_ref()?.screen()?, &self.workspace);
                self.screens.insert(name.clone(), screen);
                Ok(())
            }
            Action::Restart => self.restart(),
            Action::Quit => self.quit(),
        }
    }

    fn wait_for_event(
        &mut self,
        equals: &BTreeMap<String, Value>,
        contains: &BTreeMap<String, String>,
        timeout: Duration,
    ) -> Result<()> {
        let deadline = self.deadline_after(timeout);
        loop {
            let records = read_jsonl(&self.events_path)?;
            if let Some((index, _)) = records
                .iter()
                .enumerate()
                .skip(self.event_cursor)
                .find(|(_, record)| record_matches(record, equals, contains))
            {
                self.event_cursor = index + 1;
                return Ok(());
            }
            self.ensure_running()?;
            if Instant::now() >= deadline {
                bail!(
                    "timed out after {} ms waiting for TUI event matching equals={equals:?}, contains={contains:?}; saw {} record(s)",
                    timeout.as_millis(),
                    records.len()
                );
            }
            std::thread::sleep(POLL_INTERVAL);
        }
    }

    fn wait_for_provider_requests(&mut self, count: usize, timeout: Duration) -> Result<()> {
        let deadline = self.deadline_after(timeout);
        loop {
            let actual = match self.provider.as_ref() {
                Some(provider) => provider
                    .requests()
                    .iter()
                    .filter(|request| {
                        request
                            .path
                            .split('?')
                            .next()
                            .is_some_and(|path| path.ends_with("/chat/completions"))
                    })
                    .count(),
                None => read_jsonl(&self.events_path)?
                    .iter()
                    .filter(|record| record["event"] == "provider_request")
                    .count(),
            };
            if actual >= count {
                return Ok(());
            }
            self.ensure_running()?;
            if Instant::now() >= deadline {
                bail!("timed out waiting for {count} chat request(s); saw {actual}");
            }
            std::thread::sleep(POLL_INTERVAL);
        }
    }

    fn wait_for_event_absence(
        &mut self,
        equals: &BTreeMap<String, Value>,
        contains: &BTreeMap<String, String>,
        duration: Duration,
    ) -> Result<()> {
        let start = read_jsonl(&self.events_path)?.len();
        let deadline = self.deadline_after(duration);
        loop {
            let records = read_jsonl(&self.events_path)?;
            if let Some((index, record)) = records
                .iter()
                .enumerate()
                .skip(start)
                .find(|(_, record)| record_matches(record, equals, contains))
            {
                bail!(
                    "unexpected TUI event at record {index} while waiting {} ms for absence; equals={equals:?}, contains={contains:?}, record={record}",
                    duration.as_millis()
                );
            }
            self.ensure_running()?;
            if Instant::now() >= deadline {
                // Establish a forward evidence boundary for the next action.
                // This also prevents a later wait from satisfying itself with
                // an event that predates the proven-quiet interval.
                self.event_cursor = records.len();
                return Ok(());
            }
            std::thread::sleep(POLL_INTERVAL);
        }
    }

    fn wait_for_file(&mut self, relative: &str, exists: bool, timeout: Duration) -> Result<()> {
        let relative_path = Path::new(relative);
        let path = self.workspace.join(relative_path);
        let deadline = self.deadline_after(timeout);
        loop {
            let metadata = workspace_file_metadata(&self.workspace, relative_path)?;
            match (metadata, exists) {
                (Some(metadata), true) => {
                    ensure!(
                        metadata.file_type().is_file(),
                        "workspace file wait resolved to a non-file: {}",
                        path.display()
                    );
                    return Ok(());
                }
                (None, false) => return Ok(()),
                (Some(_), false) | (None, true) => {}
            }
            self.ensure_running()?;
            if Instant::now() >= deadline {
                bail!(
                    "timed out waiting for {} to {}",
                    path.display(),
                    if exists { "exist" } else { "be absent" }
                );
            }
            std::thread::sleep(POLL_INTERVAL);
        }
    }

    fn wait_for_marked_process(
        &mut self,
        command_contains: &str,
        at_least: usize,
        timeout: Duration,
    ) -> Result<()> {
        let marker = self.process_ref()?.run_marker().to_owned();
        let leader = self.process_ref()?.process_id();
        #[cfg(unix)]
        let groups = self
            .process_ref()?
            .process_group()
            .into_iter()
            .collect::<Vec<_>>();
        #[cfg(not(unix))]
        let groups = Vec::new();
        let deadline = self.deadline_after(timeout);
        loop {
            let marked = collect_marked_processes(&marker)?;
            let mut matching = marked
                .iter()
                .filter(|process| u32::try_from(process.pid).ok() != leader)
                .filter(|process| process.command.contains(command_contains))
                .filter_map(|process| u32::try_from(process.pid).ok())
                .collect::<BTreeSet<_>>();
            let grouped = collect_process_group_leaks(&groups);
            if let Some(error) = grouped
                .iter()
                .find_map(|process| process.get("inspection_error"))
            {
                bail!("process inspection failed while waiting for descendant: {error}");
            }
            matching.extend(
                grouped
                    .iter()
                    .filter(|process| {
                        process
                            .get("command")
                            .and_then(Value::as_str)
                            .is_some_and(|command| command.contains(command_contains))
                    })
                    .filter_map(|process| {
                        process
                            .get("pid")
                            .and_then(Value::as_str)
                            .and_then(|pid| pid.parse::<u32>().ok())
                    })
                    .filter(|pid| Some(*pid) != leader),
            );
            #[cfg(unix)]
            let descendants = match leader {
                Some(leader) => collect_process_descendants(leader)?,
                None => Vec::new(),
            };
            #[cfg(not(unix))]
            let descendants = Vec::new();
            #[cfg(unix)]
            self.retain_observed_descendants(&descendants, groups.first().copied());
            matching.extend(
                descendants
                    .iter()
                    .filter(|process| process.command.contains(command_contains))
                    .filter_map(|process| u32::try_from(process.pid).ok()),
            );
            if matching.len() >= at_least {
                return Ok(());
            }
            self.ensure_running()?;
            if Instant::now() >= deadline {
                bail!(
                    "timed out waiting for {at_least} harness descendant process(es) containing {command_contains:?}; saw {}; marked processes={marked:?}; process-group members={grouped:?}; PPID descendants={descendants:?}",
                    matching.len()
                );
            }
            std::thread::sleep(POLL_INTERVAL);
        }
    }

    fn wait_for_quiescence(
        &mut self,
        source: QuiescentSource,
        quiet: Duration,
        timeout: Duration,
    ) -> Result<()> {
        ensure!(
            !quiet.is_zero(),
            "quiescence quiet_ms must be greater than zero"
        );
        let deadline = self.deadline_after(timeout);
        let mut prior = self.activity_count(source)?;
        let mut stable_since = Instant::now();
        loop {
            let current = self.activity_count(source)?;
            if current != prior {
                prior = current;
                stable_since = Instant::now();
            } else if stable_since.elapsed() >= quiet {
                return Ok(());
            }
            self.ensure_running()?;
            if Instant::now() >= deadline {
                bail!("timed out waiting for {source:?} quiescence at activity count {current}");
            }
            std::thread::sleep(POLL_INTERVAL);
        }
    }

    fn activity_count(&self, source: QuiescentSource) -> Result<usize> {
        match source {
            QuiescentSource::Events => Ok(read_jsonl(&self.events_path)?.len()),
            QuiescentSource::Provider => self
                .provider
                .as_ref()
                .map(|provider| provider.requests().len())
                .ok_or_else(|| anyhow!("provider quiescence is unavailable in live mode")),
        }
    }

    fn restart(&mut self) -> Result<()> {
        if self.process.is_some() {
            self.observe_current_descendants()?;
            self.process_mut()?.send_line("/quit")?;
            if self
                .wait_for_exit_observing(Instant::now() + Duration::from_secs(5))?
                .is_none()
            {
                self.process_mut()?.terminate_group()?;
                bail!("hi did not exit normally within five seconds of /quit before restart");
            }
            self.archive_exited_process()?;
        }
        self.exit_code = None;
        self.spawn_hi()
    }

    fn quit(&mut self) -> Result<()> {
        let deadline = self.deadline_after(Duration::from_secs(5));
        if self.process.is_none() {
            return Ok(());
        }
        self.observe_current_descendants()?;
        if self.process_mut()?.try_wait()?.is_none() {
            self.process_mut()?.send_line("/quit")?;
        }
        let status = self.wait_for_exit_observing(deadline)?;
        let status = match status {
            Some(status) => status,
            None => {
                self.process_mut()?.terminate_group()?;
                bail!("hi did not exit within five seconds of /quit");
            }
        };
        self.exit_code = Some(status.exit_code());
        self.archive_exited_process()
    }

    fn wait_for_exit_observing(
        &mut self,
        deadline: Instant,
    ) -> Result<Option<portable_pty::ExitStatus>> {
        while Instant::now() < deadline {
            // Sample ancestry before polling the leader. Once it is reaped,
            // escaped descendants may be reparented and become undiscoverable.
            self.observe_current_descendants()?;
            if let Some(status) = self.process_mut()?.try_wait()? {
                return Ok(Some(status));
            }
            std::thread::sleep(POLL_INTERVAL);
        }
        self.observe_current_descendants()?;
        Ok(None)
    }

    fn observe_current_descendants(&mut self) -> Result<()> {
        #[cfg(unix)]
        {
            let Some(process) = self.process.as_ref() else {
                return Ok(());
            };
            let Some(leader) = process.process_id() else {
                return Ok(());
            };
            let leader_group = process.process_group();
            let descendants = collect_process_descendants(leader)?;
            self.retain_observed_descendants(&descendants, leader_group);
        }
        Ok(())
    }

    #[cfg(unix)]
    fn retain_observed_descendants(
        &mut self,
        descendants: &[ObservedProcess],
        leader_group: Option<i32>,
    ) {
        for process in descendants {
            if process.pid > 1 {
                self.observed_descendant_pids.insert(process.pid);
            }
            if process.pgid > 1 && Some(process.pgid) != leader_group {
                self.observed_descendant_groups.insert(process.pgid);
            }
        }
    }

    /// Captures descendants after the `hi` leader has been reaped but before
    /// dropping the PTY process performs its defensive group cleanup.
    fn archive_exited_process(&mut self) -> Result<()> {
        let leak_result = self.record_current_process_leaks();
        let archive_result = self.archive_process();
        leak_result.and(archive_result)
    }

    fn archive_process(&mut self) -> Result<()> {
        if let Some(process) = self.process.take() {
            let raw = process.raw();
            drop(process);
            match raw {
                Ok(raw) => append_raw(&mut self.historical_raw, &raw),
                Err(error) => {
                    let marker = format!(
                        "\n<hi-smoke infrastructure error: raw terminal evidence unavailable: {error:#}>\n"
                    );
                    append_raw(
                        &mut self.historical_raw,
                        &RawTerminal {
                            bytes: marker.as_bytes().to_vec(),
                            truncated: false,
                            total_bytes: marker.len() as u64,
                        },
                    );
                    return Err(error);
                }
            }
        }
        Ok(())
    }

    fn ensure_running(&mut self) -> Result<()> {
        self.ensure_time("while waiting")?;
        self.refresh_turn_watchdog()?;
        self.observe_current_descendants()?;
        if let Some(status) = self.process_mut()?.try_wait()? {
            self.exit_code = Some(status.exit_code());
            if let Err(leak_error) = self.record_current_process_leaks() {
                bail!("hi exited early: {status}; {leak_error:#}");
            }
            bail!("hi exited early: {status}");
        }
        Ok(())
    }

    fn refresh_turn_watchdog(&mut self) -> Result<()> {
        let events = read_jsonl(&self.events_path)?;
        let last_started = events
            .iter()
            .filter(|event| event["event"] == "turn_started")
            .filter_map(|event| event["sequence"].as_u64())
            .max();
        let last_settled = events
            .iter()
            .filter(|event| event["event"] == "turn_settled")
            .filter_map(|event| event["sequence"].as_u64())
            .max();
        let active =
            last_started.filter(|started| last_settled.is_none_or(|settled| *started > settled));
        match active {
            Some(sequence) => {
                if self
                    .observed_active_turn
                    .is_none_or(|(observed, _)| observed != sequence)
                {
                    self.observed_active_turn = Some((sequence, Instant::now()));
                }
                let limit = Duration::from_secs(self.scenario.hi.outer_turn_kill_secs.unwrap_or(
                    match self.options.mode {
                        RunMode::Scripted => 45,
                        RunMode::Live => 300,
                    },
                ));
                if self
                    .observed_active_turn
                    .is_some_and(|(_, started)| started.elapsed() > limit)
                {
                    bail!(
                        "turn {sequence} exceeded the {} second outer kill boundary",
                        limit.as_secs()
                    );
                }
            }
            None => self.observed_active_turn = None,
        }
        Ok(())
    }

    fn ensure_time(&self, context: impl std::fmt::Display) -> Result<()> {
        ensure!(
            Instant::now() < self.scenario_deadline,
            "scenario deadline expired {context}"
        );
        Ok(())
    }

    fn deadline_after(&self, requested: Duration) -> Instant {
        (Instant::now() + requested).min(self.scenario_deadline)
    }

    fn timeout_for(&self, configured_ms: u64) -> Duration {
        match self.options.mode {
            RunMode::Scripted => Duration::from_millis(configured_ms),
            RunMode::Live => Duration::from_millis(configured_ms).max(LIVE_TRANSITION),
        }
    }

    fn process_mut(&mut self) -> Result<&mut PtyProcess> {
        self.process
            .as_mut()
            .ok_or_else(|| anyhow!("hi process is not running"))
    }

    fn process_ref(&self) -> Result<&PtyProcess> {
        self.process
            .as_ref()
            .ok_or_else(|| anyhow!("hi process is not running"))
    }

    fn check_hard_invariants(&mut self) -> Result<()> {
        let events = read_jsonl(&self.events_path)?;
        validate_event_invariants(&events, &self.process_markers)?;
        validate_live_provider_event_route(&events, self.live_route.as_ref())?;
        let session = read_jsonl(&self.session_path)?;
        // Parsing every non-empty line is itself a hard invariant.
        let _ = session;
        if let Some(provider) = &self.provider {
            provider
                .assert_clean()
                .context("strict provider script failed")?;
        }
        self.check_process_groups()?;
        Ok(())
    }

    fn check_live_provider_route_invariant(&self) -> Result<()> {
        if self.live_route.is_none() {
            return Ok(());
        }
        let events = read_jsonl(&self.events_path)
            .context("live provider evidence invariant failed while reading TUI events")?;
        validate_live_provider_event_route(&events, self.live_route.as_ref())
    }

    fn check_process_groups(&mut self) -> Result<()> {
        let observed = self.inspect_owned_processes();
        preserve_process_leaks(&mut self.leaked_processes, observed);
        ensure!(
            self.leaked_processes.is_empty(),
            "leaked descendant processes: {}",
            serde_json::to_string(&self.leaked_processes)?
        );
        Ok(())
    }

    fn check_isolation_mutation_invariant(&mut self) -> Result<()> {
        let final_snapshot = {
            let root = self
                .isolation
                .as_ref()
                .ok_or_else(|| anyhow!("isolation containment invariant lost its root"))?
                .path();
            crate::isolation::capture(root)
                .context("isolation containment invariant could not capture final state")?
        };
        let events = read_jsonl(&self.events_path)
            .context("isolation containment invariant could not read lifecycle evidence")?;
        let process_tool_activity = events.iter().any(|event| {
            event["event"] == "ui_event"
                && matches!(
                    event["data"]["kind"].as_str(),
                    Some("tool_started" | "tool_call" | "tool_result")
                )
                && event["data"]["name"] == "bash"
        });
        let policy = IsolationPolicy::for_workspace(&self.workspace, process_tool_activity)
            .context("isolation containment invariant could not derive per-case identities")?;
        let evidence = crate::isolation::compare_with_policy(
            &self.isolation_baseline,
            &final_snapshot,
            Some(&policy),
        );
        let unexpected = evidence
            .unexpected_paths()
            .into_iter()
            .map(str::to_owned)
            .collect::<Vec<_>>();
        self.isolation_evidence = Some(evidence);
        ensure!(
            unexpected.is_empty(),
            "isolation containment invariant detected {} unexpected mutation(s) outside the workspace: {}",
            unexpected.len(),
            serde_json::to_string(&unexpected)?
        );
        Ok(())
    }

    fn check_post_failure_state_invariants(&self, failed: bool) -> Result<()> {
        if !failed {
            return Ok(());
        }
        let events = read_jsonl(&self.events_path)
            .context("post-failure state invariant could not parse TUI event JSONL")?;
        validate_event_run_ids(&events, &self.process_markers)
            .context("post-failure state invariant rejected TUI process identities")?;
        validate_event_stream_safety(&events)
            .context("post-failure state invariant rejected TUI lifecycle evidence")?;
        read_jsonl(&self.session_path)
            .context("post-failure state invariant could not parse session JSONL")?;
        Ok(())
    }

    fn record_current_process_leaks(&mut self) -> Result<()> {
        // This can run after the leader was reaped, so include every PID and
        // PGID retained at earlier PPID-ancestry checkpoints.
        let observed = self.inspect_owned_processes();
        let leaked = !observed.is_empty();
        preserve_process_leaks(&mut self.leaked_processes, observed);
        ensure!(
            !leaked,
            "leaked descendant processes: {}",
            serde_json::to_string(&self.leaked_processes)?
        );
        Ok(())
    }

    fn inspect_owned_processes(&self) -> Vec<Value> {
        let mut groups = self.process_groups.clone();
        groups.extend(self.observed_descendant_groups.iter().copied());
        groups.sort_unstable();
        groups.dedup();
        let mut observed = collect_process_group_leaks(&groups);
        observed.extend(collect_process_id_leaks(&self.observed_descendant_pids));
        observed.extend(collect_process_marker_leaks(&self.process_markers));
        let mut unique = Vec::new();
        preserve_process_leaks(&mut unique, observed);
        unique
    }

    fn evaluate_assertion(&self, assertion: &Assertion) -> Result<()> {
        match assertion {
            Assertion::Records {
                source,
                equals,
                contains,
                exact,
                at_least,
                at_most,
            } => {
                let records = self.records(*source)?;
                let count = records
                    .iter()
                    .filter(|record| record_matches(record, equals, contains))
                    .count();
                if let Some(expected) = exact {
                    ensure!(
                        count == *expected,
                        "expected exactly {expected} matching record(s), got {count}"
                    );
                }
                if let Some(expected) = at_least {
                    ensure!(
                        count >= *expected,
                        "expected at least {expected} matching record(s), got {count}"
                    );
                }
                if let Some(expected) = at_most {
                    ensure!(
                        count <= *expected,
                        "expected at most {expected} matching record(s), got {count}"
                    );
                }
                Ok(())
            }
            Assertion::RecordSequence {
                source,
                where_equals,
                where_contains,
                pointer,
                values,
            } => {
                let actual = self
                    .records(*source)?
                    .iter()
                    .filter(|record| record_matches(record, where_equals, where_contains))
                    .filter_map(|record| record.pointer(pointer).cloned())
                    .collect::<Vec<_>>();
                ensure!(
                    is_subsequence(&actual, values),
                    "expected sequence {values:?} at pointer {pointer:?} after filtering equals={where_equals:?}, contains={where_contains:?}; got {actual:?}"
                );
                Ok(())
            }
            Assertion::SubstringOccurrences {
                source,
                equals,
                contains,
                pointer,
                substring,
                exact,
            } => {
                let records = self.records(*source)?;
                let matching = records
                    .iter()
                    .filter(|record| record_matches(record, equals, contains))
                    .collect::<Vec<_>>();
                let occurrences = matching
                    .iter()
                    .filter_map(|record| record.pointer(pointer).and_then(Value::as_str))
                    .map(|value| value.match_indices(substring).count())
                    .sum::<usize>();
                ensure!(
                    occurrences == *exact,
                    "expected exactly {exact} occurrence(s) of {substring:?} at pointer {pointer:?} across {} matching record(s), got {occurrences}",
                    matching.len()
                );
                Ok(())
            }
            Assertion::AllRecords {
                source,
                where_equals,
                where_contains,
                equals,
                contains,
                at_least,
            } => {
                let records = self.records(*source)?;
                let matching = records
                    .iter()
                    .enumerate()
                    .filter(|(_, record)| record_matches(record, where_equals, where_contains))
                    .collect::<Vec<_>>();
                ensure!(
                    matching.len() >= *at_least,
                    "expected at least {at_least} selected record(s), got {}",
                    matching.len()
                );
                if let Some((index, _)) = matching
                    .iter()
                    .find(|(_, record)| !record_matches(record, equals, contains))
                {
                    bail!(
                        "expected every selected record to match required fields; source record {index} did not"
                    );
                }
                Ok(())
            }
            Assertion::Screen {
                snapshot,
                contains,
                excludes,
            } => {
                let screen = self
                    .screens
                    .get(snapshot)
                    .ok_or_else(|| anyhow!("screen snapshot {snapshot:?} was not captured"))?;
                for needle in contains {
                    ensure!(
                        screen.contains(needle),
                        "screen {snapshot:?} did not contain {needle:?}"
                    );
                }
                for needle in excludes {
                    ensure!(
                        !screen.contains(needle),
                        "screen {snapshot:?} unexpectedly contained {needle:?}"
                    );
                }
                Ok(())
            }
            Assertion::File {
                path,
                exists,
                contains,
                equals,
            } => {
                let relative = Path::new(path);
                let full = self.workspace.join(relative);
                let metadata = workspace_file_metadata(&self.workspace, relative)?;
                ensure!(
                    metadata.is_some() == *exists,
                    "expected {} to {}",
                    full.display(),
                    if *exists { "exist" } else { "be absent" }
                );
                if let Some(metadata) = metadata {
                    ensure!(
                        metadata.file_type().is_file(),
                        "workspace file assertion resolved to a non-file: {}",
                        full.display()
                    );
                }
                if let Some(expected) = contains {
                    let body = fs::read_to_string(&full)?;
                    ensure!(
                        body.contains(expected),
                        "{} did not contain {expected:?}",
                        full.display()
                    );
                }
                if let Some(expected) = equals {
                    let body = fs::read_to_string(&full)?;
                    ensure!(body == *expected, "{} contents differed", full.display());
                }
                Ok(())
            }
            Assertion::WorkspacePatch {
                contains,
                excludes,
                equals,
            } => {
                let patch = capture_workspace_patch(&self.initial_workspace, &self.workspace)?;
                for needle in contains {
                    let needle = normalize_patch_line_endings(needle);
                    ensure!(
                        patch.contains(&needle),
                        "workspace patch did not contain {needle:?}"
                    );
                }
                for needle in excludes {
                    let needle = normalize_patch_line_endings(needle);
                    ensure!(
                        !patch.contains(&needle),
                        "workspace patch unexpectedly contained {needle:?}"
                    );
                }
                if let Some(expected) = equals {
                    let expected = normalize_patch_line_endings(expected);
                    ensure!(
                        patch == expected,
                        "workspace patch differed from expected text"
                    );
                }
                Ok(())
            }
            Assertion::WorkspaceListing { contains, excludes } => {
                let entries = crate::artifacts::capture_workspace_listing(&self.workspace)?;
                let paths = entries
                    .iter()
                    .map(|entry| entry.path.as_str())
                    .collect::<Vec<_>>();
                for expected in contains {
                    let expected = normalize_workspace_listing_path(expected)?;
                    ensure!(
                        paths.binary_search(&expected.as_str()).is_ok(),
                        "workspace listing did not contain {expected:?}; got {paths:?}"
                    );
                }
                for unexpected in excludes {
                    let unexpected = normalize_workspace_listing_path(unexpected)?;
                    ensure!(
                        paths.binary_search(&unexpected.as_str()).is_err(),
                        "workspace listing unexpectedly contained {unexpected:?}"
                    );
                }
                Ok(())
            }
            Assertion::Exit { code } => {
                ensure!(
                    self.exit_code == Some(*code),
                    "expected exit code {code}, got {:?}",
                    self.exit_code
                );
                Ok(())
            }
            Assertion::ProviderConsumed => self
                .provider
                .as_ref()
                .ok_or_else(|| anyhow!("provider_consumed is unavailable in live mode"))?
                .assert_clean()
                .context("strict provider script was not consumed"),
        }
    }

    fn records(&self, source: RecordSource) -> Result<Vec<Value>> {
        match source {
            RecordSource::Events => read_jsonl(&self.events_path),
            RecordSource::Session => read_jsonl(&self.session_path),
            RecordSource::ProviderRequests => Ok(self.provider_requests()),
        }
    }

    fn provider_requests(&self) -> Vec<Value> {
        match self.provider.as_ref() {
            Some(provider) => provider
                .requests()
                .into_iter()
                .map(|request| serde_json::to_value(request).unwrap_or(Value::Null))
                .collect(),
            None => read_jsonl_lossy(&self.events_path)
                .into_iter()
                .filter(|record| record["event"] == "provider_request")
                .map(|record| record["data"].clone())
                .collect(),
        }
    }

    fn combined_raw(&self) -> Result<RawTerminal> {
        let mut raw = self.historical_raw.clone();
        if let Some(process) = &self.process {
            append_raw(&mut raw, &process.raw()?);
        }
        Ok(raw)
    }

    fn finish_bundle(
        &mut self,
        artifact_dir: &Path,
        duration: Duration,
        failure: Option<&anyhow::Error>,
    ) -> Result<()> {
        // `run_discovered` already merged cleanup failures into the case
        // result. This second call only covers direct unit-test callers of
        // `finish_bundle`; never let it replace the complete rich bundle with
        // a minimal repair after evidence has already been collected.
        let _ = self.prepare_failure_evidence(failure.is_some());
        if self.isolation_evidence.is_none() {
            self.check_isolation_mutation_invariant()?;
        }
        let events = read_jsonl_lossy(&self.events_path);
        let raw = self.combined_raw()?;
        let process = json!({
            "pids": self.process_ids,
            "process_groups": self.process_groups,
            "process_markers": self.process_markers,
            "observed_descendant_pids": self.observed_descendant_pids,
            "observed_descendant_groups": self.observed_descendant_groups,
            "exit_code": self.exit_code,
            "leaked_processes": self.leaked_processes,
        });
        let failure_kind = failure.map(classify_failure);
        let status = failure_kind
            .map(case_failure_kind_label)
            .unwrap_or("passed");
        let result = json!({
            "schema_version": 1,
            "name": self.scenario.name,
            "status": status,
            "mode": format!("{:?}", self.options.mode).to_ascii_lowercase(),
            "seed": self.options.seed,
            "duration_ms": millis(duration),
            "failure": failure.map(|error| format!("{error:#}")),
            "failure_kind": failure_kind,
            "live_route": self.live_route.as_ref(),
        });
        let provider_requests = self.provider_requests();
        let redaction_values = vec![
            "hi-smoke-test-key".into(),
            live_value("HI_API_KEY").unwrap_or_default(),
        ];
        let session_bytes = fs::read(&self.session_path).unwrap_or_default();
        let assertions = json!(self.assertions);
        let timings = serde_json::to_value(&self.timings)?;
        let patch = capture_workspace_patch(&self.initial_workspace, &self.workspace)?;
        let failure_message = failure.map(|error| format!("{error:#}"));
        let isolation_evidence = serde_json::to_value(
            self.isolation_evidence
                .as_ref()
                .ok_or_else(|| anyhow!("isolation containment invariant produced no evidence"))?,
        )?;
        let bundle = crate::artifacts::BundleInput {
            scenario: &self.scenario,
            mode: match self.options.mode {
                RunMode::Scripted => "scripted",
                RunMode::Live => "live",
            },
            live_route: self.live_route.as_ref(),
            status: failure_kind
                .map(bundle_status_for_failure)
                .unwrap_or(crate::artifacts::BundleStatus::Passed),
            seed: self.options.seed,
            duration_ms: millis(duration),
            failure: failure_message.as_deref(),
            tui_events: &events,
            raw_terminal: &raw,
            screens: &self.screens,
            provider_requests: &provider_requests,
            redaction_values: &redaction_values,
            session_jsonl: &session_bytes,
            workspace_root: &self.workspace,
            initial_workspace_root: Some(&self.initial_workspace),
            workspace_patch: &patch,
            isolation_evidence: &isolation_evidence,
            process: &process,
            assertions: &assertions,
            timings: &timings,
            result: &result,
        };
        crate::artifacts::write_case_bundle(
            &self.options.artifacts,
            Path::new(artifact_dir.file_name().unwrap_or_default()),
            &bundle,
        )?;
        if self.options.keep
            && let Some(isolation) = self.isolation.take()
        {
            let kept = isolation.keep();
            eprintln!("kept isolated workspace at {}", kept.display());
        }
        Ok(())
    }

    fn prepare_failure_evidence(&mut self, failed: bool) -> Result<()> {
        if !failed {
            return Ok(());
        }

        let mut cleanup_failures = Vec::new();
        // An action can observe a broken PTY before `ensure_running` gets a
        // chance to report the exited leader. Sample that path here as well,
        // while descendants are still observable.
        if let Err(error) = self.observe_current_descendants() {
            cleanup_failures.push(format!("pre-cleanup descendant discovery: {error:#}"));
        }
        let exited_status = match self.process.as_mut().map(PtyProcess::try_wait) {
            Some(Ok(status)) => status,
            Some(Err(error)) => {
                cleanup_failures.push(format!("polling failed process: {error:#}"));
                None
            }
            None => None,
        };
        if let Some(status) = exited_status {
            self.exit_code = Some(status.exit_code());
            if let Err(error) = self.record_current_process_leaks() {
                cleanup_failures.push(format!("pre-cleanup leak inspection: {error:#}"));
            }
        }
        if let Some(process) = self.process.as_mut() {
            match process.terminate_group() {
                Ok(Some(status)) => self.exit_code = Some(status.exit_code()),
                Ok(None) => {}
                Err(error) => {
                    cleanup_failures.push(format!("terminating process group: {error:#}"))
                }
            }
        }
        if let Some(process) = self.process.as_ref()
            && !self.screens.contains_key("failure")
        {
            match process.screen() {
                Ok(screen) => {
                    self.screens
                        .insert("failure".into(), normalize_screen(&screen, &self.workspace));
                }
                Err(error) => {
                    self.screens.insert(
                        "failure".into(),
                        format!(
                            "<hi-smoke infrastructure error: virtual screen evidence unavailable: {error:#}>\n"
                        ),
                    );
                    cleanup_failures.push(format!("capturing failure virtual screen: {error:#}"));
                }
            }
        }
        if let Err(error) = self.archive_process() {
            cleanup_failures.push(format!("archiving raw terminal evidence: {error:#}"));
        }
        let surviving = self.inspect_owned_processes();
        if !surviving.is_empty() {
            preserve_process_leaks(&mut self.leaked_processes, surviving.clone());
            cleanup_failures.push(format!(
                "post-cleanup leaked descendant processes: {}",
                serde_json::to_string(&surviving)?
            ));
            if let Err(error) = cleanup_observed_processes(
                &self.observed_descendant_pids,
                &self.observed_descendant_groups,
            ) {
                cleanup_failures.push(format!("reaping retained descendants: {error:#}"));
            }
            let remaining = self.inspect_owned_processes();
            if !remaining.is_empty() {
                preserve_process_leaks(&mut self.leaked_processes, remaining.clone());
                cleanup_failures.push(format!(
                    "descendants survived retained-process cleanup: {}",
                    serde_json::to_string(&remaining)?
                ));
            }
        }
        if !self.leaked_processes.is_empty()
            && !cleanup_failures
                .iter()
                .any(|failure| failure.contains("leaked descendant"))
        {
            cleanup_failures.push(format!(
                "previously observed leaked descendant processes: {}",
                serde_json::to_string(&self.leaked_processes)?
            ));
        }
        self.complete_pending_assertion_evidence();

        ensure!(
            cleanup_failures.is_empty(),
            "failure cleanup invariant failed: {}",
            cleanup_failures.join("; ")
        );
        Ok(())
    }

    fn complete_pending_assertion_evidence(&mut self) {
        let completed = self.assertions.len();
        for (index, assertion) in self
            .scenario
            .assertions
            .clone()
            .into_iter()
            .enumerate()
            .skip(completed)
        {
            let result = self.evaluate_assertion(&assertion);
            self.assertions.push(json!({
                "index": index,
                "kind": assertion_name(&assertion),
                "passed": result.is_ok(),
                "failure": result.as_ref().err().map(|error| format!("{error:#}")),
                "evaluated_after_failure": true,
            }));
        }
    }
}

fn merge_live_route_invariant(
    execution: Result<()>,
    live_route_invariant: Result<()>,
) -> Result<()> {
    match (execution, live_route_invariant) {
        (Ok(()), Ok(())) => Ok(()),
        (Err(execution), Ok(())) => Err(execution),
        (Ok(()), Err(invariant)) => Err(invariant),
        (Err(execution), Err(invariant)) => Err(invariant).with_context(|| {
            format!("scenario also failed before route validation: {execution:#}")
        }),
    }
}

fn merge_isolation_invariant(execution: Result<()>, isolation_invariant: Result<()>) -> Result<()> {
    match (execution, isolation_invariant) {
        (Ok(()), Ok(())) => Ok(()),
        (Err(execution), Ok(())) => Err(execution),
        (Ok(()), Err(invariant)) => Err(invariant),
        (Err(execution), Err(invariant)) => Err(invariant).with_context(|| {
            format!("scenario also failed before isolation validation: {execution:#}")
        }),
    }
}

fn merge_failure_cleanup_invariant(
    execution: Result<()>,
    cleanup_invariant: Result<()>,
) -> Result<()> {
    match (execution, cleanup_invariant) {
        (Ok(()), Ok(())) => Ok(()),
        (Err(execution), Ok(())) => Err(execution),
        (Ok(()), Err(invariant)) => Err(invariant),
        (Err(execution), Err(invariant)) => Err(invariant).with_context(|| {
            format!("scenario also failed before cleanup validation: {execution:#}")
        }),
    }
}

fn merge_post_failure_state_invariant(
    execution: Result<()>,
    state_invariant: Result<()>,
) -> Result<()> {
    match (execution, state_invariant) {
        (Ok(()), Ok(())) => Ok(()),
        (Err(execution), Ok(())) => Err(execution),
        (Ok(()), Err(invariant)) => Err(invariant),
        (Err(execution), Err(invariant)) => Err(invariant).with_context(|| {
            format!("scenario also failed before persistent-state validation: {execution:#}")
        }),
    }
}

fn classify_failure(error: &anyhow::Error) -> CaseFailureKind {
    let message = format!("{error:#}").to_ascii_lowercase();
    if message.contains("autonomous")
        || message.contains("prompt queue invariant")
        || (message.contains("started before turn") && message.contains("settled"))
    {
        CaseFailureKind::InfrastructureLoop
    } else if message.contains("live provider evidence invariant")
        || message.contains("isolation containment invariant")
        || message.contains("failure cleanup invariant")
        || message.contains("post-failure state invariant")
    {
        // The route invariant deliberately overrides an earlier action
        // timeout/crash classification. Running a live case against the wrong
        // endpoint or model is harness infrastructure failure, not a scenario
        // quality result.
        CaseFailureKind::InfrastructureFailure
    } else if message.contains("hi exited early")
        || message.contains("did not emit a normal session_ended")
        || message.contains("did not exit normally")
        || message.contains("terminated by signal")
    {
        CaseFailureKind::Crashed
    } else if message.contains("timed out")
        || message.contains("deadline expired")
        || message.contains("outer kill boundary")
    {
        CaseFailureKind::TimedOut
    } else if message.contains("cancelled") || message.contains("canceled") {
        CaseFailureKind::Cancelled
    } else if message.contains("requires an enforced")
        || message.contains("creating isolated smoke workspace")
        || message.contains("starting deterministic openai server")
        || message.contains("minimal replay bundle")
        || message.contains("artifact bundle")
        || message.contains("bundle could not be repaired")
        || message.contains("leaked descendant")
        || message.contains("process inspection failed")
        || message.contains("terminal parser lock was poisoned")
        || message.contains("terminal evidence lock was poisoned")
    {
        CaseFailureKind::InfrastructureFailure
    } else {
        CaseFailureKind::Scenario
    }
}

fn case_failure_kind_label(kind: CaseFailureKind) -> &'static str {
    match kind {
        CaseFailureKind::Scenario => "failed",
        CaseFailureKind::TimedOut => "timed_out",
        CaseFailureKind::Cancelled => "cancelled",
        CaseFailureKind::Crashed => "crashed",
        CaseFailureKind::InfrastructureFailure => "infrastructure_failure",
        CaseFailureKind::InfrastructureLoop => "infrastructure_loop",
    }
}

fn bundle_status_for_failure(kind: CaseFailureKind) -> crate::artifacts::BundleStatus {
    match kind {
        CaseFailureKind::Scenario => crate::artifacts::BundleStatus::Failed,
        CaseFailureKind::TimedOut => crate::artifacts::BundleStatus::TimedOut,
        CaseFailureKind::Cancelled => crate::artifacts::BundleStatus::Cancelled,
        CaseFailureKind::Crashed => crate::artifacts::BundleStatus::Crashed,
        CaseFailureKind::InfrastructureFailure => {
            crate::artifacts::BundleStatus::InfrastructureFailure
        }
        CaseFailureKind::InfrastructureLoop => crate::artifacts::BundleStatus::InfrastructureLoop,
    }
}

fn start_provider(scenario: &Scenario) -> Result<ScriptedOpenAiServer> {
    let steps = scenario
        .provider
        .steps
        .iter()
        .enumerate()
        .map(|(index, step)| {
            let mut matcher = RequestMatcher::any()
                .method("POST")
                .path_suffix("/chat/completions")
                .header("authorization", "Bearer hi-smoke-test-key");
            for needle in &step.expect.body_contains {
                matcher = matcher.body_contains(needle);
            }
            for needle in &step.expect.body_excludes {
                matcher = matcher.body_excludes(needle);
            }
            for (pointer, value) in &step.expect.json_equals {
                matcher = matcher.json_eq(pointer, value.clone());
            }
            let response = scripted_response(&step.response, index);
            ChatStep::expecting(matcher, response).named(&step.id)
        })
        .collect::<Vec<_>>();
    ScriptedOpenAiServer::builder()
        .chat_steps(steps)
        .start()
        .context("starting deterministic OpenAI server")
}

fn scripted_response(response: &ProviderResponse, index: usize) -> ScriptedResponse {
    match response {
        ProviderResponse::Text {
            text,
            gate,
            delay_ms,
            chunk_bytes,
            terminal,
        } => {
            let mut response = ScriptedResponse::text(text);
            if *delay_ms > 0 {
                response = response.delayed(Duration::from_millis(*delay_ms));
            }
            if let Some(chunk_bytes) = chunk_bytes {
                response = response.fragmented(*chunk_bytes, Duration::from_millis(1));
            }
            match terminal {
                StreamTerminal::Done => {}
                StreamTerminal::Eof => response = response.finish_with_eof(),
                StreamTerminal::Reset => response = response.finish_with_reset(),
            }
            gate.as_ref()
                .map_or(response.clone(), |gate| response.wait_for_gate(gate))
        }
        ProviderResponse::ToolCall {
            name,
            arguments,
            gate,
            delay_ms,
        } => {
            let call =
                ScriptedToolCall::new(format!("call-smoke-{index}"), name, arguments.clone());
            let mut response = ScriptedResponse::tool_call(call);
            if *delay_ms > 0 {
                response = response.delayed(Duration::from_millis(*delay_ms));
            }
            gate.as_ref()
                .map_or(response.clone(), |gate| response.wait_for_gate(gate))
        }
        ProviderResponse::HttpError { status, body, gate } => {
            let response = ScriptedResponse::http_error(*status, body);
            gate.as_ref()
                .map_or(response.clone(), |gate| response.wait_for_gate(gate))
        }
        ProviderResponse::RawSse {
            body,
            gate,
            delay_ms,
            chunk_bytes,
            terminal,
        } => {
            let mut response = ScriptedResponse::raw_sse(body);
            if *delay_ms > 0 {
                response = response.delayed(Duration::from_millis(*delay_ms));
            }
            if let Some(chunk_bytes) = chunk_bytes {
                response = response.fragmented(*chunk_bytes, Duration::from_millis(1));
            }
            match terminal {
                StreamTerminal::Done => {}
                StreamTerminal::Eof => response = response.finish_with_eof(),
                StreamTerminal::Reset => response = response.finish_with_reset(),
            }
            gate.as_ref()
                .map_or(response.clone(), |gate| response.wait_for_gate(gate))
        }
        ProviderResponse::Hold { gate } => ScriptedResponse::raw_sse("")
            .hold_open_until(gate.as_deref().unwrap_or("unreachable-hold")),
        ProviderResponse::Reset { gate } => {
            let response = ScriptedResponse::reset();
            gate.as_ref()
                .map_or(response.clone(), |gate| response.wait_for_gate(gate))
        }
    }
}

fn write_session_seed(path: &Path, scenario: &Scenario) -> Result<()> {
    if scenario.session.plan.is_empty()
        && !scenario.session.plan_drive_paused
        && scenario.session.plan_drive_stall == 0
    {
        return Ok(());
    }
    let mut file = OpenOptions::new().create(true).append(true).open(path)?;
    if !scenario.session.plan.is_empty() {
        let steps = scenario
            .session
            .plan
            .iter()
            .map(|step| {
                json!({
                    "title": step.title,
                    "status": match step.status {
                        PlanSeedStatus::Pending => "Pending",
                        PlanSeedStatus::Active => "Active",
                        PlanSeedStatus::Done => "Done",
                    }
                })
            })
            .collect::<Vec<_>>();
        writeln!(file, "{}", json!({"type": "plan", "steps": steps}))?;
    }
    writeln!(
        file,
        "{}",
        json!({
            "type": "plan_drive",
            "paused": scenario.session.plan_drive_paused,
            "resume_on_user_input": scenario.session.plan_drive_resume_on_user_input,
            "stall": scenario.session.plan_drive_stall,
        })
    )?;
    file.flush()?;
    Ok(())
}

fn initialize_git(workspace: &Path, state: GitState) -> Result<()> {
    if matches!(state, GitState::None) {
        return Ok(());
    }
    // Failure replays embed the pre-run repository, including its clean
    // baseline commit. Reuse that exact state instead of attempting a second
    // empty fixture commit.
    if workspace.join(".git").is_dir() {
        return Ok(());
    }
    run_git(workspace, &["init", "--quiet"])?;
    run_git(
        workspace,
        &["config", "user.email", "hi-smoke@example.invalid"],
    )?;
    run_git(workspace, &["config", "user.name", "hi smoke"])?;
    run_git(workspace, &["add", "."])?;
    run_git(workspace, &["commit", "--quiet", "-m", "fixture"])?;
    if matches!(state, GitState::Dirty) {
        fs::write(workspace.join(".hi-smoke-dirty"), "dirty fixture\n")?;
    }
    Ok(())
}

fn run_git(cwd: &Path, args: &[&str]) -> Result<()> {
    let output = std::process::Command::new("git")
        .args(args)
        .current_dir(cwd)
        .output()
        .with_context(|| format!("running git {}", args.join(" ")))?;
    ensure!(
        output.status.success(),
        "git {} failed: {}",
        args.join(" "),
        String::from_utf8_lossy(&output.stderr)
    );
    Ok(())
}

fn validate_event_invariants(events: &[Value], run_ids: &[String]) -> Result<()> {
    ensure!(
        events.iter().any(|event| event["event"] == "ready"),
        "trace did not contain a ready event"
    );
    ensure!(
        events.iter().any(|event| event["event"] == "session_ended"),
        "trace did not contain a normal session_ended event"
    );
    let observed_run_ids = validate_event_run_ids(events, run_ids)?;
    ensure!(
        observed_run_ids == run_ids,
        "trace run order {observed_run_ids:?} did not match spawned run order {run_ids:?}"
    );
    for run_id in run_ids {
        let ready_count = events
            .iter()
            .filter(|event| event["event"] == "ready" && event["run_id"] == run_id.as_str())
            .count();
        ensure!(
            ready_count == 1,
            "TUI run {run_id} emitted {ready_count} ready events instead of exactly one"
        );
        let ended_count = events
            .iter()
            .filter(|event| event["event"] == "session_ended" && event["run_id"] == run_id.as_str())
            .count();
        ensure!(
            ended_count == 1,
            "TUI run {run_id} emitted {ended_count} normal session_ended events instead of exactly one"
        );
        let ready_index = events
            .iter()
            .position(|event| event["event"] == "ready" && event["run_id"] == run_id.as_str())
            .expect("ready count was exactly one");
        let ended_index = events
            .iter()
            .position(|event| {
                event["event"] == "session_ended" && event["run_id"] == run_id.as_str()
            })
            .expect("session_ended count was exactly one");
        ensure!(
            ready_index < ended_index,
            "TUI run {run_id} emitted session_ended before ready"
        );
        let final_index = events
            .iter()
            .rposition(|event| event["run_id"] == run_id.as_str())
            .expect("validated run has trace records");
        ensure!(
            ended_index == final_index,
            "TUI run {run_id} emitted a trace record after session_ended"
        );
    }
    let active_turn = validate_event_stream_safety(events)?;
    ensure!(
        active_turn.is_none(),
        "started turn {active_turn:?} did not emit exactly one terminal settlement"
    );
    Ok(())
}

fn validate_event_run_ids(events: &[Value], expected_run_ids: &[String]) -> Result<Vec<String>> {
    let expected = expected_run_ids
        .iter()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    ensure!(
        expected.len() == expected_run_ids.len(),
        "spawned TUI run ids were not unique: {expected_run_ids:?}"
    );
    let mut observed = Vec::<String>::new();
    let mut process_ids = BTreeMap::<String, u32>::new();
    for (index, event) in events.iter().enumerate() {
        let run_id = event["run_id"]
            .as_str()
            .filter(|run_id| !run_id.is_empty())
            .ok_or_else(|| anyhow!("trace record {index} has no non-empty run_id"))?;
        ensure!(
            expected.contains(run_id),
            "trace record {index} has unknown run_id {run_id:?}"
        );
        let process_id = event["process_id"]
            .as_u64()
            .and_then(|process_id| u32::try_from(process_id).ok())
            .filter(|process_id| *process_id > 0)
            .ok_or_else(|| {
                anyhow!("trace record {index} has no positive u32 process_id for run {run_id:?}")
            })?;
        if let Some(prior) = process_ids.insert(run_id.to_owned(), process_id) {
            ensure!(
                prior == process_id,
                "trace run {run_id:?} changed process_id from {prior} to {process_id} at record {index}"
            );
        }
        if observed.last().is_none_or(|prior| prior != run_id) {
            ensure!(
                !observed.iter().any(|prior| prior == run_id),
                "trace run_id {run_id:?} reappeared noncontiguously at record {index}"
            );
            observed.push(run_id.to_owned());
        }
    }
    ensure!(
        expected_run_ids.starts_with(&observed),
        "trace run order {observed:?} was not a contiguous prefix of spawned run order {expected_run_ids:?}"
    );
    Ok(observed)
}

/// Validate trace properties that remain meaningful when the harness had to
/// terminate `hi` after an earlier failure. An active final turn is permitted
/// here because forced termination can prevent normal settlement; overlapping
/// starts, orphan settlements, inconsistent queue accounting, malformed rows, and autonomous
/// recovery after failure are still production/harness invariant violations.
fn validate_event_stream_safety(events: &[Value]) -> Result<Option<u64>> {
    let mut prior_sequence = None;
    let mut active_turn = None;
    let mut prompt_queues = BTreeMap::<String, PromptQueueTraceState>::new();
    for (index, event) in events.iter().enumerate() {
        ensure!(
            event["schema_version"] == 1,
            "trace record {index} has unsupported schema_version {:?}",
            event["schema_version"]
        );
        let sequence = event["sequence"]
            .as_u64()
            .ok_or_else(|| anyhow!("trace record {index} has no numeric sequence"))?;
        ensure!(
            prior_sequence.is_none_or(|prior| sequence > prior),
            "trace sequence did not increase at record {index}: {prior_sequence:?} then {sequence}"
        );
        prior_sequence = Some(sequence);
        match event["event"].as_str() {
            Some("turn_started") => {
                ensure!(
                    active_turn.is_none(),
                    "turn {sequence} started before turn {:?} settled",
                    active_turn
                );
                active_turn = Some(sequence);
            }
            Some("turn_settled") => {
                ensure!(
                    active_turn.take().is_some(),
                    "turn_settled {sequence} has no matching turn_started"
                );
            }
            _ => {}
        }
        validate_prompt_queue_event(index, event, &mut prompt_queues)?;
    }
    for (index, event) in events.iter().enumerate() {
        if event["event"] != "turn_settled" {
            continue;
        }
        let status = event
            .pointer("/data/outcome/status")
            .and_then(Value::as_str);
        if !matches!(status, Some("failed" | "cancelled")) {
            continue;
        }
        let mut explicit_input = false;
        for next in &events[index + 1..] {
            if next["event"] == "prompt_dequeued"
                && matches!(
                    next.pointer("/data/origin").and_then(Value::as_str),
                    Some("user" | "command_follow_up")
                )
            {
                explicit_input = true;
            }
            if next["event"] == "turn_started" {
                let origin = next.pointer("/data/origin").and_then(Value::as_str);
                ensure!(
                    explicit_input || !matches!(origin, Some("plan_drive" | "goal_drive")),
                    "autonomous {origin:?} turn started after {status:?} settlement without explicit input"
                );
                break;
            }
        }
    }
    Ok(active_turn)
}

#[derive(Default)]
struct PromptQueueTraceState {
    depth: u64,
    pending: BTreeMap<String, u64>,
}

/// Reconcile the redacted prompt multiset from typed queue transitions. Queue
/// depth is deliberately not capped: a long-running turn may accumulate more
/// than 64 legitimate follow-ups. Correctness is instead defined by every
/// enqueue increasing the reported depth by one, every dequeue/removal
/// consuming exactly one previously queued fingerprint, and all other observed
/// depths agreeing with the outstanding multiset.
fn validate_prompt_queue_event(
    index: usize,
    event: &Value,
    queues: &mut BTreeMap<String, PromptQueueTraceState>,
) -> Result<()> {
    let event_name = event["event"].as_str().unwrap_or("<unknown>");
    let is_transition = matches!(
        event_name,
        "prompt_queued" | "prompt_dequeued" | "prompt_removed"
    );
    let Some(depth_value) = event.pointer("/data/queue_depth") else {
        ensure!(
            !is_transition,
            "prompt queue invariant at record {index}: {event_name} has no queue_depth"
        );
        if event_name == "session_ended" {
            let run_id = event["run_id"]
                .as_str()
                .filter(|run_id| !run_id.is_empty())
                .ok_or_else(|| {
                    anyhow!("prompt queue invariant at record {index}: session_ended has no run_id")
                })?;
            if let Some(queue) = queues.get(run_id) {
                ensure!(
                    queue.depth == 0 && queue.pending.is_empty(),
                    "prompt queue invariant at record {index}: run {run_id:?} ended with {} queued prompt(s)",
                    queue.depth
                );
            }
        }
        return Ok(());
    };
    let reported_depth = depth_value.as_u64().ok_or_else(|| {
        anyhow!(
            "prompt queue invariant at record {index}: queue_depth must be a nonnegative integer, got {depth_value}"
        )
    })?;
    let run_id = event["run_id"]
        .as_str()
        .filter(|run_id| !run_id.is_empty())
        .ok_or_else(|| {
            anyhow!(
                "prompt queue invariant at record {index}: queue_depth is present but run_id is missing"
            )
        })?;
    let queue = queues.entry(run_id.to_owned()).or_default();

    match event_name {
        "prompt_queued" => {
            let fingerprint = prompt_fingerprint(index, event, event_name)?;
            queue.depth = queue.depth.checked_add(1).ok_or_else(|| {
                anyhow!("prompt queue invariant at record {index}: tracked depth overflow")
            })?;
            let count = queue.pending.entry(fingerprint.to_string()).or_default();
            *count = count.checked_add(1).ok_or_else(|| {
                anyhow!(
                    "prompt queue invariant at record {index}: fingerprint multiplicity overflow"
                )
            })?;
        }
        "prompt_dequeued" | "prompt_removed" => {
            let fingerprint = prompt_fingerprint(index, event, event_name)?;
            ensure!(
                queue.depth > 0,
                "prompt queue invariant at record {index}: {event_name} consumed {fingerprint:?} from an empty queue"
            );
            let remove_fingerprint = {
                let count = queue.pending.get_mut(fingerprint).ok_or_else(|| {
                    anyhow!(
                        "prompt queue invariant at record {index}: {event_name} consumed unqueued fingerprint {fingerprint:?}"
                    )
                })?;
                ensure!(
                    *count > 0,
                    "prompt queue invariant at record {index}: fingerprint {fingerprint:?} has zero multiplicity"
                );
                *count -= 1;
                *count == 0
            };
            if remove_fingerprint {
                queue.pending.remove(fingerprint);
            }
            queue.depth -= 1;
        }
        _ => {}
    }

    ensure!(
        reported_depth == queue.depth,
        "prompt queue invariant at record {index}: {event_name} reported depth {reported_depth}, but traced prompt multiset has depth {}",
        queue.depth
    );
    if event_name == "session_ended" {
        ensure!(
            queue.depth == 0 && queue.pending.is_empty(),
            "prompt queue invariant at record {index}: run {run_id:?} ended with {} queued prompt(s)",
            queue.depth
        );
    }
    Ok(())
}

fn prompt_fingerprint<'a>(index: usize, event: &'a Value, event_name: &str) -> Result<&'a str> {
    event
        .pointer("/data/prompt_fingerprint")
        .and_then(Value::as_str)
        .filter(|fingerprint| !fingerprint.is_empty())
        .ok_or_else(|| {
            anyhow!(
                "prompt queue invariant at record {index}: {event_name} has no non-empty prompt_fingerprint"
            )
        })
}

fn validate_live_provider_event_route(
    events: &[Value],
    live_route: Option<&LiveRoute>,
) -> Result<()> {
    let Some(route) = live_route else {
        return Ok(());
    };
    let requests = events
        .iter()
        .filter(|record| record["event"] == "provider_request")
        .map(|record| record["data"].clone())
        .collect::<Vec<_>>();
    route.validate_provider_requests(&requests)
}

fn record_matches(
    record: &Value,
    equals: &BTreeMap<String, Value>,
    contains: &BTreeMap<String, String>,
) -> bool {
    equals
        .iter()
        .all(|(pointer, expected)| record.pointer(pointer) == Some(expected))
        && contains.iter().all(|(pointer, needle)| {
            record
                .pointer(pointer)
                .and_then(Value::as_str)
                .is_some_and(|actual| actual.contains(needle))
        })
}

fn is_subsequence(actual: &[Value], expected: &[Value]) -> bool {
    let mut expected = expected.iter();
    let mut next = expected.next();
    for value in actual {
        if next == Some(value) {
            next = expected.next();
        }
    }
    next.is_none()
}

fn read_jsonl(path: &Path) -> Result<Vec<Value>> {
    if !path.exists() {
        return Ok(Vec::new());
    }
    let body = fs::read_to_string(path)
        .with_context(|| format!("reading JSONL evidence {}", path.display()))?;
    body.lines()
        .enumerate()
        .filter(|(_, line)| !line.trim().is_empty())
        .map(|(index, line)| {
            serde_json::from_str(line)
                .with_context(|| format!("parsing {} line {}", path.display(), index + 1))
        })
        .collect()
}

fn read_jsonl_lossy(path: &Path) -> Vec<Value> {
    read_jsonl(path)
        .unwrap_or_else(|error| vec![json!({"evidence_parse_error": format!("{error:#}")})])
}

/// Resolve a scenario-owned workspace file without ever following a symlink.
/// This is used both while `hi` is active (file waits) and after settlement
/// (file assertions). A dangling final symlink is therefore an unsafe present
/// entry, never an absent file, and cannot make `exists = false` pass.
fn workspace_file_metadata(root: &Path, relative: &Path) -> Result<Option<fs::Metadata>> {
    validate_relative_path(
        relative
            .to_str()
            .ok_or_else(|| anyhow!("workspace assertion path is not valid UTF-8"))?,
    )?;
    let components = relative
        .components()
        .filter_map(|component| match component {
            std::path::Component::Normal(part) => Some(part),
            std::path::Component::CurDir => None,
            std::path::Component::ParentDir
            | std::path::Component::RootDir
            | std::path::Component::Prefix(_) => None,
        })
        .collect::<Vec<_>>();
    ensure!(
        !components.is_empty(),
        "workspace file path must name an entry"
    );

    let mut current = root.to_path_buf();
    for (index, component) in components.iter().enumerate() {
        current.push(component);
        let metadata = match fs::symlink_metadata(&current) {
            Ok(metadata) => metadata,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(error) => {
                return Err(error)
                    .with_context(|| format!("reading workspace entry {}", current.display()));
            }
        };
        ensure!(
            !metadata.file_type().is_symlink(),
            "workspace file path traverses a symlink: {}",
            current.display()
        );
        if index + 1 < components.len() {
            ensure!(
                metadata.file_type().is_dir(),
                "workspace file path traverses a non-directory: {}",
                current.display()
            );
        } else {
            return Ok(Some(metadata));
        }
    }
    unreachable!("validated workspace path has at least one component")
}

fn normalize_screen(screen: &str, workspace: &Path) -> String {
    let workspace_text = workspace.to_string_lossy();
    let isolation = workspace.parent();
    let isolation_text = isolation.map(|path| path.to_string_lossy().into_owned());
    let isolation_name = isolation
        .and_then(Path::file_name)
        .map(|name| name.to_string_lossy().into_owned());
    let mut normalized = screen.replace(workspace_text.as_ref(), "<WORKSPACE>");
    if let Some(isolation) = isolation_text {
        normalized = normalized.replace(&isolation, "<ISOLATION>");
    }
    if let Some(isolation) = isolation_name {
        normalized = normalized.replace(&isolation, "<ISOLATION>");
    }
    let mut lines = normalized
        .lines()
        .map(str::trim_end)
        .map(str::to_string)
        .collect::<Vec<_>>();
    while lines.last().is_some_and(|line| line.is_empty()) {
        lines.pop();
    }
    lines.join("\n") + "\n"
}

fn append_raw(target: &mut RawTerminal, next: &RawTerminal) {
    target.total_bytes = target.total_bytes.saturating_add(next.total_bytes);
    let remaining = crate::pty::MAX_RAW_TERMINAL_BYTES.saturating_sub(target.bytes.len());
    let keep = remaining.min(next.bytes.len());
    target.bytes.extend_from_slice(&next.bytes[..keep]);
    target.truncated |= next.truncated || keep < next.bytes.len();
}

fn copy_tree(source: &Path, destination: &Path) -> Result<()> {
    let metadata = fs::symlink_metadata(source)
        .with_context(|| format!("reading fixture metadata {}", source.display()))?;
    ensure!(
        metadata.file_type().is_dir() && !metadata.file_type().is_symlink(),
        "fixture root must be a real directory: {}",
        source.display()
    );
    fs::create_dir_all(destination)?;
    for entry in
        fs::read_dir(source).with_context(|| format!("reading fixture {}", source.display()))?
    {
        let entry = entry?;
        let file_type = entry.file_type()?;
        let target = destination.join(entry.file_name());
        if file_type.is_dir() {
            copy_tree(&entry.path(), &target)?;
        } else if file_type.is_file() {
            fs::copy(entry.path(), &target)?;
        } else if file_type.is_symlink() {
            bail!(
                "fixture symlinks are not allowed: {}",
                entry.path().display()
            );
        }
    }
    Ok(())
}

fn provider_launch_parts(
    provider: &str,
    model: String,
    api_key: String,
) -> (Vec<String>, (String, String)) {
    // Keep provider credentials out of argv and therefore out of process
    // listings and failure-bundle command evidence. Live campaigns can select
    // the real provider implementation with HI_PROVIDER while retaining the
    // same isolated, explicit endpoint. Prefer the provider-specific key name
    // where one exists so official-provider authentication takes its ordinary
    // production path.
    let credential_name = match provider {
        "pipenetwork" | "pipe" => "PIPENETWORK_API_KEY",
        "anthropic" => "ANTHROPIC_API_KEY",
        "xai" => "XAI_API_KEY",
        _ => "HI_API_KEY",
    };
    (
        vec![
            "--provider".into(),
            provider.into(),
            "--model".into(),
            model,
        ],
        (credential_name.into(), api_key),
    )
}

fn extend_scenario_env_with_credential(
    env: &mut BTreeMap<String, String>,
    scenario_env: &BTreeMap<String, String>,
    credential: (String, String),
) {
    env.extend(scenario_env.clone());
    // Schema validation rejects every supported credential alias. Insert the
    // selected credential last as defense in depth so scenario data can never
    // win even if a future caller bypasses validation.
    env.insert(credential.0, credential.1);
}

fn smoke_sandbox_config(hi_bin: &Path) -> hi_tools::sandbox::SandboxConfig {
    let candidate_dir = hi_bin
        .parent()
        .expect("staged TUI candidate always has a parent directory");
    hi_tools::sandbox::SandboxConfig {
        // Overlay the whole staging directory read-only after the broad
        // isolation-root bind. Directory binds are preserved across
        // pipe-wrap's private-root setup, unlike a file-level bind to a
        // candidate on some hosted-runner mounts.
        deny_write: vec![candidate_dir.to_path_buf()],
        deny_host_temp: true,
        // The smoke harness retains the PTY process group, discovers escaped
        // descendants by a run marker, and performs TERM/KILL cleanup. A
        // parent-death signal instead ties pipe-wrap to portable-pty's short
        // launcher lifetime and can kill a healthy sandbox after exec.
        supervisor_owns_lifetime: true,
        ..hi_tools::sandbox::SandboxConfig::default()
    }
}

fn stage_sandbox_candidate(isolation_root: &Path, source: &Path) -> Result<PathBuf> {
    ensure!(
        source.is_file(),
        "TUI smoke candidate is not a file: {}",
        source.display()
    );
    let candidate_dir = isolation_root.join(STAGED_CANDIDATE_DIR);
    fs::create_dir(&candidate_dir).with_context(|| {
        format!(
            "creating staged TUI candidate directory {}",
            candidate_dir.display()
        )
    })?;
    let file_name = source
        .file_name()
        .filter(|name| !name.is_empty())
        .context("TUI smoke candidate path has no file name")?;
    let staged = candidate_dir.join(file_name);
    reflink_copy::reflink_or_copy(source, &staged).with_context(|| {
        format!(
            "staging TUI candidate {} at {}",
            source.display(),
            staged.display()
        )
    })?;
    let permissions = fs::metadata(source)
        .with_context(|| format!("reading TUI candidate metadata {}", source.display()))?
        .permissions();
    fs::set_permissions(&staged, permissions)
        .with_context(|| format!("setting TUI candidate permissions {}", staged.display()))?;
    Ok(staged)
}

fn smoke_sandbox_profile(
    isolation_root: &Path,
    hi_bin: &Path,
) -> Result<hi_tools::sandbox::SandboxProfile> {
    let profile = hi_tools::sandbox::SandboxProfile::with_config(
        hi_tools::sandbox::SandboxPolicy::Workspace,
        &[isolation_root],
        smoke_sandbox_config(hi_bin),
    );
    ensure!(
        profile.is_enforced(),
        "TUI smoke requires an enforced {} sandbox; {}",
        profile.backend_name(),
        hi_tools::sandbox::SandboxProfile::unenforced_warning()
    );
    Ok(profile)
}

fn capture_workspace_patch(initial: &Path, final_root: &Path) -> Result<String> {
    let output = std::process::Command::new("git")
        .args([
            "diff",
            "--no-index",
            "--binary",
            "--",
            ".",
            final_root.to_string_lossy().as_ref(),
        ])
        .current_dir(initial)
        .output()
        .context("capturing workspace patch")?;
    // `git diff --no-index` returns 1 when differences exist.
    ensure!(
        output.status.success() || output.status.code() == Some(1),
        "capturing workspace patch failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let patch = String::from_utf8_lossy(&output.stdout).into_owned();
    Ok(normalize_workspace_patch(&patch, initial, final_root))
}

fn normalize_workspace_patch(patch: &str, initial: &Path, final_root: &Path) -> String {
    // `git diff --no-index` embeds its absolute right-hand operand after the
    // synthetic `a`/`b` prefix. Remove both random isolation roots and collapse
    // the relative left-hand `./` spelling so replay and assertion text is
    // stable across runs. Preserve all other bytes, including trailing spaces
    // in changed lines.
    normalize_patch_line_endings(patch)
        .replace(initial.to_string_lossy().as_ref(), "")
        .replace(final_root.to_string_lossy().as_ref(), "")
        .replace("a/./", "a/")
        .replace("b/./", "b/")
}

fn normalize_patch_line_endings(text: &str) -> String {
    text.replace("\r\n", "\n").replace('\r', "\n")
}

#[cfg(unix)]
#[derive(Clone, Debug, Eq, PartialEq)]
struct ObservedProcess {
    pid: i32,
    ppid: i32,
    pgid: i32,
    command: String,
}

#[cfg(unix)]
fn collect_process_descendants(leader: u32) -> Result<Vec<ObservedProcess>> {
    let processes = collect_process_table()?;
    Ok(process_descendants_from_table(leader, processes))
}

#[cfg(unix)]
fn collect_process_table() -> Result<Vec<ObservedProcess>> {
    let output = std::process::Command::new("ps")
        .args(["-Ao", "pid=,ppid=,pgid=,command="])
        .output()
        .context("launching process inspection for descendant wait")?;
    ensure!(
        output.status.success(),
        "process inspection failed while discovering descendants: {}",
        String::from_utf8_lossy(&output.stderr).trim()
    );
    let mut processes = Vec::new();
    for line in String::from_utf8_lossy(&output.stdout).lines() {
        let fields = line.split_whitespace().collect::<Vec<_>>();
        if fields.len() < 4 {
            continue;
        }
        let (Ok(pid), Ok(ppid), Ok(pgid)) = (
            fields[0].parse::<i32>(),
            fields[1].parse::<i32>(),
            fields[2].parse::<i32>(),
        ) else {
            continue;
        };
        processes.push(ObservedProcess {
            pid,
            ppid,
            pgid,
            command: fields[3..].join(" "),
        });
    }

    Ok(processes)
}

#[cfg(unix)]
fn process_descendants_from_table(
    leader: u32,
    processes: Vec<ObservedProcess>,
) -> Vec<ObservedProcess> {
    let Ok(leader) = i32::try_from(leader) else {
        return Vec::new();
    };
    let mut family = BTreeSet::from([leader]);
    loop {
        let before = family.len();
        for process in &processes {
            if family.contains(&process.ppid) {
                family.insert(process.pid);
            }
        }
        if family.len() == before {
            break;
        }
    }
    processes
        .into_iter()
        .filter(|process| process.pid != leader && family.contains(&process.pid))
        .collect()
}

#[cfg(unix)]
fn collect_process_id_leaks(pids: &BTreeSet<i32>) -> Vec<Value> {
    match collect_process_table() {
        Ok(processes) => processes
            .into_iter()
            .filter(|process| pids.contains(&process.pid))
            .map(|process| {
                json!({
                    "source": "observed_ppid_ancestry",
                    "pid": process.pid,
                    "ppid": process.ppid,
                    "pgid": process.pgid,
                    "command": process.command,
                })
            })
            .collect(),
        Err(error) => vec![json!({
            "source": "observed_ppid_ancestry",
            "inspection_error": format!("{error:#}"),
        })],
    }
}

#[cfg(unix)]
fn cleanup_observed_processes(pids: &BTreeSet<i32>, groups: &BTreeSet<i32>) -> Result<()> {
    let own_pid = unsafe { libc::getpid() };
    let own_group = unsafe { libc::getpgrp() };
    let signal = |signal: i32| -> Result<()> {
        let table = collect_process_table()?;
        for process in table.iter().filter(|process| {
            process.pid > 1 && process.pid != own_pid && pids.contains(&process.pid)
        }) {
            let result = unsafe { libc::kill(process.pid, signal) };
            if result != 0 && std::io::Error::last_os_error().raw_os_error() != Some(libc::ESRCH) {
                return Err(std::io::Error::last_os_error())
                    .with_context(|| format!("signaling retained descendant pid {}", process.pid));
            }
        }
        for group in groups
            .iter()
            .copied()
            .filter(|group| *group > 1 && *group != own_group)
        {
            let result = unsafe { libc::kill(-group, signal) };
            if result != 0 {
                let error = std::io::Error::last_os_error();
                // Some macOS runners deny group-wide signaling even though
                // signaling each retained same-uid PID succeeds. Exact PID
                // cleanup above remains authoritative; the final rescan turns
                // any child it missed into a hard leak failure.
                if !matches!(error.raw_os_error(), Some(libc::ESRCH) | Some(libc::EPERM)) {
                    return Err(error)
                        .with_context(|| format!("signaling retained descendant group {group}"));
                }
            }
        }
        Ok(())
    };

    signal(libc::SIGTERM)?;
    let grace_deadline = Instant::now() + Duration::from_secs(2);
    while Instant::now() < grace_deadline {
        let remaining = collect_process_table()?.into_iter().any(|process| {
            (pids.contains(&process.pid) || groups.contains(&process.pgid))
                && process.pid != own_pid
                && process.pgid != own_group
        });
        if !remaining {
            return Ok(());
        }
        std::thread::sleep(POLL_INTERVAL);
    }
    signal(libc::SIGKILL)?;
    let kill_deadline = Instant::now() + Duration::from_secs(2);
    while Instant::now() < kill_deadline {
        let remaining = collect_process_table()?.into_iter().any(|process| {
            (pids.contains(&process.pid) || groups.contains(&process.pgid))
                && process.pid != own_pid
                && process.pgid != own_group
        });
        if !remaining {
            return Ok(());
        }
        std::thread::sleep(POLL_INTERVAL);
    }
    bail!("retained descendants remained visible after SIGKILL")
}

#[cfg(unix)]
fn collect_process_group_leaks(groups: &[i32]) -> Vec<Value> {
    let output = std::process::Command::new("ps")
        .args(["-Ao", "pid=,ppid=,pgid=,command="])
        .output();
    let Ok(output) = output else {
        return vec![json!({"inspection_error": "ps failed to launch"})];
    };
    process_group_leaks_from_ps_output(
        groups,
        output.status.success(),
        &output.stdout,
        &output.stderr,
    )
}

#[cfg(unix)]
fn process_group_leaks_from_ps_output(
    groups: &[i32],
    success: bool,
    stdout: &[u8],
    stderr: &[u8],
) -> Vec<Value> {
    if !success {
        let detail = String::from_utf8_lossy(stderr)
            .chars()
            .filter(|character| !character.is_control() || *character == '\n')
            .take(1_000)
            .collect::<String>();
        return vec![json!({
            "inspection_error": "process inspection failed: ps exited unsuccessfully",
            "detail": detail.trim(),
        })];
    }
    let mut leaks = Vec::new();
    for line in String::from_utf8_lossy(stdout).lines() {
        let fields = line.split_whitespace().collect::<Vec<_>>();
        if fields.len() < 4 {
            continue;
        }
        let Ok(pgid) = fields[2].parse::<i32>() else {
            continue;
        };
        if groups.contains(&pgid) {
            leaks.push(json!({"pid": fields[0], "ppid": fields[1], "pgid": pgid, "command": fields[3..].join(" ")}));
        }
    }
    leaks
}

fn collect_process_marker_leaks(markers: &[String]) -> Vec<Value> {
    let mut leaks = Vec::new();
    for marker in markers {
        match collect_marked_processes(marker) {
            Ok(processes) => leaks.extend(processes.into_iter().map(|process| {
                json!({
                    "source": "run_marker",
                    "pid": process.pid,
                    "ppid": process.ppid,
                    "pgid": process.pgid,
                    "command": process.command,
                })
            })),
            Err(error) => leaks.push(json!({
                "source": "run_marker",
                "inspection_error": format!("{error:#}"),
            })),
        }
    }
    leaks
}

fn preserve_process_leaks(recorded: &mut Vec<Value>, observed: Vec<Value>) {
    for leak in observed {
        if !recorded.contains(&leak) {
            recorded.push(leak);
        }
    }
}

#[cfg(not(unix))]
fn collect_process_group_leaks(_groups: &[i32]) -> Vec<Value> {
    Vec::new()
}

#[cfg(not(unix))]
fn collect_process_id_leaks(_pids: &BTreeSet<i32>) -> Vec<Value> {
    Vec::new()
}

#[cfg(not(unix))]
fn cleanup_observed_processes(_pids: &BTreeSet<i32>, _groups: &BTreeSet<i32>) -> Result<()> {
    Ok(())
}

fn resolve_live_route(recorded: Option<&LiveRoute>) -> Result<LiveRoute> {
    resolve_live_route_with(recorded, live_value)
}

fn resolve_live_route_with(
    recorded: Option<&LiveRoute>,
    value: impl Fn(&str) -> Option<String>,
) -> Result<LiveRoute> {
    let api_key =
        value("HI_API_KEY").ok_or_else(|| anyhow!("live mode requires non-empty HI_API_KEY"))?;
    let route = match recorded {
        Some(route) => LiveRoute::new(&route.provider, &route.model, &route.base_url)?,
        None => LiveRoute::new(
            value("HI_PROVIDER").as_deref().unwrap_or("openai"),
            &value("HI_MODEL").ok_or_else(|| anyhow!("live mode requires non-empty HI_MODEL"))?,
            &value("HI_BASE_URL")
                .ok_or_else(|| anyhow!("live mode requires non-empty HI_BASE_URL"))?,
        )?,
    };
    route.ensure_excludes_secret(&api_key)?;
    Ok(route)
}

fn live_value(name: &str) -> Option<String> {
    std::env::var(name)
        .ok()
        .filter(|value| !value.trim().is_empty())
}

fn unique_case_dir(name: &str, seed: Option<u64>) -> String {
    let base = match seed {
        Some(seed) => format!("{}-seed-{seed}", safe_name(name)),
        None => safe_name(name),
    };
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or(0);
    format!("{base}-{}-{nonce}", std::process::id())
}

fn safe_name(name: &str) -> String {
    let safe = name
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || matches!(character, '-' | '_') {
                character
            } else {
                '-'
            }
        })
        .collect::<String>()
        .trim_matches('-')
        .to_string();
    if safe.is_empty() {
        "scenario".into()
    } else {
        safe
    }
}

fn millis(duration: Duration) -> u64 {
    duration.as_millis().min(u64::MAX as u128) as u64
}

fn action_name(action: &Action) -> &'static str {
    match action {
        Action::SendLine { .. } => "send_line",
        Action::SendKey { .. } => "send_key",
        Action::Resize { .. } => "resize",
        Action::WaitEvent { .. } => "wait_event",
        Action::WaitEventAbsent { .. } => "wait_event_absent",
        Action::WaitProviderRequest { .. } => "wait_provider_request",
        Action::WaitFile { .. } => "wait_file",
        Action::WaitProcess { .. } => "wait_process",
        Action::WaitQuiescent { .. } => "wait_quiescent",
        Action::ReleaseGate { .. } => "release_gate",
        Action::CaptureScreen { .. } => "capture_screen",
        Action::Restart => "restart",
        Action::Quit => "quit",
    }
}

fn assertion_name(assertion: &Assertion) -> &'static str {
    match assertion {
        Assertion::Records { .. } => "records",
        Assertion::RecordSequence { .. } => "record_sequence",
        Assertion::SubstringOccurrences { .. } => "substring_occurrences",
        Assertion::AllRecords { .. } => "all_records",
        Assertion::Screen { .. } => "screen",
        Assertion::File { .. } => "file",
        Assertion::WorkspacePatch { .. } => "workspace_patch",
        Assertion::WorkspaceListing { .. } => "workspace_listing",
        Assertion::Exit { .. } => "exit",
        Assertion::ProviderConsumed => "provider_consumed",
    }
}

fn write_initialization_failure(
    artifact_dir: &Path,
    scenario: &Scenario,
    options: &CaseOptions,
    duration: Duration,
    error: &anyhow::Error,
) -> Result<()> {
    let mode = match options.mode {
        RunMode::Scripted => "scripted",
        RunMode::Live => "live",
    };
    let fixture = setup_fixture_source(scenario);
    let relative = Path::new(
        artifact_dir
            .file_name()
            .ok_or_else(|| anyhow!("artifact case directory has no file name"))?,
    );
    let failure = format!("{error:#}");
    let redaction_values = vec![
        "hi-smoke-test-key".to_owned(),
        live_value("HI_API_KEY").unwrap_or_default(),
    ];
    crate::artifacts::repair_minimal_failure_bundle(
        &options.artifacts,
        relative,
        &crate::artifacts::MinimalBundleInput {
            scenario,
            mode,
            live_route: options.live_route.as_ref(),
            seed: options.seed,
            duration_ms: millis(duration),
            failure: &failure,
            fixture_root: fixture.as_deref(),
            redaction_values: &redaction_values,
            detailed_bundle_failure: None,
        },
    )?;
    Ok(())
}

fn setup_fixture_source(scenario: &Scenario) -> Option<PathBuf> {
    let fixture = scenario.workspace.fixture.as_deref()?;
    let relative = Path::new(fixture);
    if relative.as_os_str().is_empty()
        || relative.is_absolute()
        || relative.components().any(|component| {
            matches!(
                component,
                std::path::Component::ParentDir
                    | std::path::Component::RootDir
                    | std::path::Component::Prefix(_)
            )
        })
    {
        return None;
    }
    let source = scenario.source_dir.join(relative);
    source.is_dir().then_some(source)
}

fn minimal_replay_is_complete(artifact_dir: &Path) -> bool {
    if !artifact_dir.join("summary.json").is_file() {
        return false;
    }
    let replay = artifact_dir.join("replay.toml");
    if replay_metadata(&replay).is_err() {
        return false;
    }
    let Ok(scenario) = Scenario::parse(&replay) else {
        return false;
    };
    scenario.workspace.fixture.as_ref().is_none_or(|fixture| {
        let relative = Path::new(fixture);
        !relative.is_absolute()
            && !relative.components().any(|component| {
                matches!(
                    component,
                    std::path::Component::ParentDir
                        | std::path::Component::RootDir
                        | std::path::Component::Prefix(_)
                )
            })
            && artifact_dir.join(relative).is_dir()
    })
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ReplayMetadata {
    mode: RunMode,
    live_route: Option<LiveRoute>,
}

fn replay_metadata(path: &Path) -> Result<ReplayMetadata> {
    let text = fs::read_to_string(path)
        .with_context(|| format!("reading replay metadata {}", path.display()))?;
    let mut mode = None;
    let mut provider = None;
    let mut model = None;
    let mut base_url = None;
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if !line.starts_with('#') {
            break;
        }
        if let Some(value) = line
            .strip_prefix("# hi-smoke-replay-mode = ")
            .or_else(|| line.strip_prefix("# mode = "))
        {
            set_replay_metadata(&mut mode, value, "mode", path)?;
        } else if let Some(value) = line.strip_prefix("# hi-smoke-live-provider = ") {
            set_replay_metadata(&mut provider, value, "live provider", path)?;
        } else if let Some(value) = line.strip_prefix("# hi-smoke-live-model = ") {
            set_replay_metadata(&mut model, value, "live model", path)?;
        } else if let Some(value) = line.strip_prefix("# hi-smoke-live-base-url = ") {
            set_replay_metadata(&mut base_url, value, "live base URL", path)?;
        }
    }
    let mode = match mode.as_deref() {
        None | Some("scripted") => RunMode::Scripted,
        Some("live") => RunMode::Live,
        Some(other) => bail!("unsupported replay mode {other:?} in {}", path.display()),
    };
    let live_route = match (provider, model, base_url) {
        (None, None, None) => None,
        (Some(provider), Some(model), Some(base_url)) => {
            Some(LiveRoute::new(&provider, &model, &base_url)?)
        }
        _ => bail!(
            "partial live route metadata in {}; provider, model, and base URL must be recorded together",
            path.display()
        ),
    };
    ensure!(
        live_route.is_none() || mode == RunMode::Live,
        "live route metadata requires live replay mode in {}",
        path.display()
    );
    Ok(ReplayMetadata { mode, live_route })
}

fn set_replay_metadata(
    slot: &mut Option<String>,
    value: &str,
    label: &str,
    path: &Path,
) -> Result<()> {
    ensure!(
        slot.is_none(),
        "duplicate replay {label} metadata in {}",
        path.display()
    );
    let value = value.trim();
    ensure!(
        !value.is_empty(),
        "empty replay {label} metadata in {}",
        path.display()
    );
    *slot = Some(value.to_owned());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(unix)]
    const ESCAPED_DESCENDANT_HELPER_ENV: &str = "HI_SMOKE_ESCAPED_DESCENDANT_HELPER";

    #[cfg(unix)]
    #[test]
    fn escaped_descendant_helper() {
        if std::env::var_os(ESCAPED_DESCENDANT_HELPER_ENV).is_none() {
            return;
        }
        let child = unsafe { libc::fork() };
        assert!(
            child >= 0,
            "fork failed: {}",
            std::io::Error::last_os_error()
        );
        if child == 0 {
            unsafe {
                libc::setsid();
                libc::unsetenv(c"HI_SMOKE_RUN_MARKER".as_ptr());
                libc::execl(
                    c"/bin/sleep".as_ptr(),
                    c"sleep".as_ptr(),
                    c"30".as_ptr(),
                    std::ptr::null::<libc::c_char>(),
                );
                libc::_exit(127);
            }
        }
        std::thread::sleep(Duration::from_secs(30));
    }

    fn test_scenario(source_dir: &Path, assertions: Vec<Assertion>) -> Scenario {
        Scenario {
            schema_version: crate::scenario::SCENARIO_SCHEMA_VERSION,
            name: "runner-evidence-test".into(),
            tags: Vec::new(),
            timeout_ms: 5_000,
            terminal: crate::scenario::TerminalSpec::default(),
            workspace: crate::scenario::WorkspaceSpec::default(),
            session: crate::scenario::SessionSeed::default(),
            hi: crate::scenario::HiSpec::default(),
            provider: crate::scenario::ProviderSpec::default(),
            actions: Vec::new(),
            assertions,
            source_dir: source_dir.to_path_buf(),
        }
    }

    fn scripted_options(root: &Path, hi_bin: PathBuf) -> CaseOptions {
        CaseOptions {
            hi_bin,
            artifacts: root.join("artifacts"),
            mode: RunMode::Scripted,
            live_route: None,
            keep: false,
            seed: None,
            sandbox_requirement: SandboxRequirement::UnitTestUnenforced,
        }
    }

    #[test]
    fn smoke_sandbox_rebinds_the_candidate_directory_read_only() {
        let hi_bin = Path::new("/isolation/.hi-smoke-candidate/hi");
        let config = smoke_sandbox_config(hi_bin);

        assert_eq!(
            config.deny_write,
            [Path::new("/isolation/.hi-smoke-candidate")]
        );
        assert!(config.deny_host_temp);
        assert!(config.supervisor_owns_lifetime);
    }

    #[test]
    fn sandbox_candidate_is_staged_inside_the_isolation_root() {
        let source_dir = tempfile::tempdir().unwrap();
        let isolation = tempfile::tempdir().unwrap();
        let source = source_dir.path().join("candidate-hi");
        fs::write(&source, b"candidate-bytes").unwrap();
        let mut permissions = fs::metadata(&source).unwrap().permissions();
        permissions.set_readonly(true);
        fs::set_permissions(&source, permissions).unwrap();

        let staged = stage_sandbox_candidate(isolation.path(), &source).unwrap();

        assert_eq!(
            staged,
            isolation
                .path()
                .join(STAGED_CANDIDATE_DIR)
                .join("candidate-hi")
        );
        assert_eq!(fs::read(&staged).unwrap(), b"candidate-bytes");
        assert!(fs::metadata(&staged).unwrap().permissions().readonly());
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn sandbox_staged_candidate_child() {
        if std::env::var_os("HI_SMOKE_STAGED_CANDIDATE_CHILD").is_some() {
            let candidate_dir = std::env::current_exe()
                .unwrap()
                .parent()
                .unwrap()
                .to_path_buf();
            let error = fs::write(candidate_dir.join("write-probe"), b"must fail")
                .expect_err("staged candidate directory must be mounted read-only");
            assert!(
                matches!(error.raw_os_error(), Some(libc::EROFS) | Some(libc::EACCES)),
                "unexpected candidate-directory write error: {error}"
            );
            println!("staged-candidate-executed-readonly");
        }
    }

    #[cfg(target_os = "linux")]
    #[test]
    #[ignore = "requires a working operator-provided pipe-wrap sandbox"]
    fn sandbox_staged_candidate_executes_inside_enforced_sandbox() {
        let isolation = tempfile::tempdir().unwrap();
        let source = std::env::current_exe().unwrap();
        let staged = stage_sandbox_candidate(isolation.path(), &source).unwrap();
        let profile = smoke_sandbox_profile(isolation.path(), &staged).unwrap();
        let (program, args) = profile.wrap_program_in(
            staged.as_os_str(),
            [
                "--exact",
                "runner::tests::sandbox_staged_candidate_child",
                "--nocapture",
            ],
            isolation.path(),
        );
        let output = std::process::Command::new(program)
            .args(args)
            .env("HI_SMOKE_STAGED_CANDIDATE_CHILD", "1")
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "status: {}\nstdout:\n{}\nstderr:\n{}",
            output.status,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(
            String::from_utf8_lossy(&output.stdout).contains("staged-candidate-executed-readonly")
        );
    }

    #[cfg(unix)]
    #[test]
    fn raw_sse_transport_options_map_to_delayed_chunked_reset_response() {
        use std::io::Read as _;
        use std::net::{Shutdown, TcpStream};

        let body = "data: first-frame\n\ndata: second-frame\n\n";
        let delay_ms = 20;
        let chunk_bytes = 1;
        let Some(server) = ScriptedOpenAiServer::new(vec![ChatStep::new(scripted_response(
            &ProviderResponse::RawSse {
                body: body.into(),
                gate: None,
                delay_ms,
                chunk_bytes: Some(chunk_bytes),
                terminal: StreamTerminal::Reset,
            },
            0,
        ))]) else {
            return;
        };
        let mut stream = TcpStream::connect(server.url().trim_start_matches("http://")).unwrap();
        stream
            .set_read_timeout(Some(Duration::from_secs(2)))
            .unwrap();
        let request_body = "{}";
        let request = format!(
            "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{request_body}",
            request_body.len()
        );

        let started = Instant::now();
        stream.write_all(request.as_bytes()).unwrap();
        stream.shutdown(Shutdown::Write).unwrap();
        let mut received = Vec::new();
        let error = stream
            .read_to_end(&mut received)
            .expect_err("raw SSE terminal reset must be visible to the client");

        assert_eq!(error.kind(), std::io::ErrorKind::ConnectionReset);
        assert!(
            received.ends_with(body.as_bytes()),
            "scripted frames were not written before reset: {}",
            String::from_utf8_lossy(&received)
        );
        let minimum =
            Duration::from_millis(delay_ms + body.len().saturating_sub(chunk_bytes) as u64 - 5);
        assert!(
            started.elapsed() >= minimum,
            "delay/fragment pacing completed too early: {:?} < {minimum:?}",
            started.elapsed()
        );
        server.assert_clean().unwrap();
    }

    fn lifecycle_event(sequence: u64, event: &str, data: Value) -> Value {
        lifecycle_event_for_run(sequence, "test-run", event, data)
    }

    fn lifecycle_event_for_run(sequence: u64, run_id: &str, event: &str, data: Value) -> Value {
        json!({
            "schema_version": 1,
            "sequence": sequence,
            "process_id": 2,
            "run_id": run_id,
            "event": event,
            "data": data,
        })
    }

    #[test]
    fn matching_supports_json_pointers_and_substrings() {
        let record = json!({"event":"x","data":{"message":"hello world"}});
        assert!(record_matches(
            &record,
            &BTreeMap::from([("/event".into(), json!("x"))]),
            &BTreeMap::from([("/data/message".into(), "world".into())]),
        ));
    }

    #[test]
    fn lifecycle_uses_unique_run_ids_when_pid_namespace_reuses_pid() {
        let events = vec![
            lifecycle_event_for_run(0, "run-a", "ready", json!({"queue_depth": 0})),
            lifecycle_event_for_run(1, "run-a", "session_ended", json!({})),
            lifecycle_event_for_run(2, "run-b", "ready", json!({"queue_depth": 0})),
            lifecycle_event_for_run(3, "run-b", "session_ended", json!({})),
        ];

        validate_event_invariants(&events, &["run-a".into(), "run-b".into()]).unwrap();
        assert!(events.iter().all(|event| event["process_id"] == 2));
    }

    #[test]
    fn lifecycle_rejects_missing_unknown_and_noncontiguous_run_ids() {
        let expected = ["run-a".into(), "run-b".into()];
        let valid = vec![
            lifecycle_event_for_run(0, "run-a", "ready", json!({})),
            lifecycle_event_for_run(1, "run-a", "session_ended", json!({})),
            lifecycle_event_for_run(2, "run-b", "ready", json!({})),
            lifecycle_event_for_run(3, "run-b", "session_ended", json!({})),
        ];

        let mut missing = valid.clone();
        missing[1].as_object_mut().unwrap().remove("run_id");
        assert!(
            validate_event_invariants(&missing, &expected)
                .unwrap_err()
                .to_string()
                .contains("no non-empty run_id")
        );

        let mut unknown = valid.clone();
        unknown[2]["run_id"] = json!("foreign-run");
        assert!(
            validate_event_invariants(&unknown, &expected)
                .unwrap_err()
                .to_string()
                .contains("unknown run_id")
        );

        let noncontiguous = vec![
            valid[0].clone(),
            valid[2].clone(),
            valid[1].clone(),
            valid[3].clone(),
        ];
        assert!(
            validate_event_invariants(&noncontiguous, &expected)
                .unwrap_err()
                .to_string()
                .contains("reappeared noncontiguously")
        );

        let missing_spawn = &valid[..2];
        assert_eq!(
            validate_event_run_ids(missing_spawn, &expected).unwrap(),
            vec!["run-a".to_owned()]
        );
        assert!(
            validate_event_invariants(missing_spawn, &expected)
                .unwrap_err()
                .to_string()
                .contains("did not match spawned run order")
        );

        let duplicate_ready = vec![
            lifecycle_event_for_run(0, "run-a", "ready", json!({})),
            lifecycle_event_for_run(1, "run-a", "ready", json!({})),
            lifecycle_event_for_run(2, "run-a", "session_ended", json!({})),
            lifecycle_event_for_run(3, "run-b", "ready", json!({})),
            lifecycle_event_for_run(4, "run-b", "session_ended", json!({})),
        ];
        assert!(
            validate_event_invariants(&duplicate_ready, &expected)
                .unwrap_err()
                .to_string()
                .contains("2 ready events")
        );
        assert!(
            validate_event_invariants(&valid, &["run-a".into(), "run-a".into()])
                .unwrap_err()
                .to_string()
                .contains("were not unique")
        );

        let missing_middle = vec![valid[0].clone(), valid[1].clone()];
        let missing_middle_expected = ["run-a".into(), "run-b".into(), "run-c".into()];
        let mut run_c = vec![
            lifecycle_event_for_run(2, "run-c", "ready", json!({})),
            lifecycle_event_for_run(3, "run-c", "session_ended", json!({})),
        ];
        let mut missing_middle = missing_middle;
        missing_middle.append(&mut run_c);
        assert!(
            validate_event_run_ids(&missing_middle, &missing_middle_expected)
                .unwrap_err()
                .to_string()
                .contains("not a contiguous prefix")
        );

        let ended_before_ready = vec![
            lifecycle_event_for_run(0, "run-a", "session_ended", json!({})),
            lifecycle_event_for_run(1, "run-a", "ready", json!({})),
        ];
        assert!(
            validate_event_invariants(&ended_before_ready, &["run-a".into()])
                .unwrap_err()
                .to_string()
                .contains("session_ended before ready")
        );

        let row_after_end = vec![
            lifecycle_event_for_run(0, "run-a", "ready", json!({})),
            lifecycle_event_for_run(1, "run-a", "session_ended", json!({})),
            lifecycle_event_for_run(2, "run-a", "ui_event", json!({"kind": "text"})),
        ];
        assert!(
            validate_event_invariants(&row_after_end, &["run-a".into()])
                .unwrap_err()
                .to_string()
                .contains("after session_ended")
        );

        let mut invalid_process_id = valid.clone();
        invalid_process_id[0]["process_id"] = json!(0);
        assert!(
            validate_event_invariants(&invalid_process_id, &expected)
                .unwrap_err()
                .to_string()
                .contains("no positive u32 process_id")
        );
        let mut changed_process_id = valid.clone();
        changed_process_id[1]["process_id"] = json!(3);
        assert!(
            validate_event_invariants(&changed_process_id, &expected)
                .unwrap_err()
                .to_string()
                .contains("changed process_id from 2 to 3")
        );
    }

    #[test]
    fn prompt_queues_are_isolated_by_run_id_when_inner_pid_is_reused() {
        let cross_run_consume = vec![
            lifecycle_event_for_run(0, "run-a", "ready", json!({"queue_depth": 0})),
            lifecycle_event_for_run(
                1,
                "run-a",
                "prompt_queued",
                json!({"prompt_fingerprint": "queued-before-restart", "queue_depth": 1}),
            ),
            lifecycle_event_for_run(
                2,
                "run-b",
                "prompt_dequeued",
                json!({"prompt_fingerprint": "queued-before-restart", "queue_depth": 0}),
            ),
        ];
        let cross_run_error = validate_event_stream_safety(&cross_run_consume).unwrap_err();
        assert!(
            cross_run_error.to_string().contains("from an empty queue"),
            "{cross_run_error:#}"
        );

        let pending_at_end = vec![
            cross_run_consume[0].clone(),
            cross_run_consume[1].clone(),
            lifecycle_event_for_run(2, "run-a", "session_ended", json!({})),
        ];
        let pending_error = validate_event_stream_safety(&pending_at_end).unwrap_err();
        assert!(
            pending_error
                .to_string()
                .contains("ended with 1 queued prompt"),
            "{pending_error:#}"
        );
    }

    #[test]
    fn matching_distinguishes_unlimited_and_finite_step_limit_evidence() {
        let unlimited = json!({
            "event": "turn_settled",
            "data": {"step_limit": {"mode": "unlimited"}},
        });
        let finite = json!({
            "event": "turn_settled",
            "data": {"step_limit": {"mode": "finite", "max_steps": 2}},
        });
        let unlimited_contract = BTreeMap::from([
            ("/event".into(), json!("turn_settled")),
            ("/data/step_limit/mode".into(), json!("unlimited")),
        ]);
        let finite_contract = BTreeMap::from([
            ("/event".into(), json!("turn_settled")),
            ("/data/step_limit/mode".into(), json!("finite")),
            ("/data/step_limit/max_steps".into(), json!(2)),
        ]);

        assert!(record_matches(
            &unlimited,
            &unlimited_contract,
            &BTreeMap::new()
        ));
        assert!(!record_matches(
            &finite,
            &unlimited_contract,
            &BTreeMap::new()
        ));
        assert!(record_matches(&finite, &finite_contract, &BTreeMap::new()));
        assert!(!record_matches(
            &unlimited,
            &finite_contract,
            &BTreeMap::new()
        ));
    }

    #[test]
    fn post_failure_state_parse_error_overrides_timeout_with_context() {
        let temporary = tempfile::tempdir().unwrap();
        let options = scripted_options(temporary.path(), temporary.path().join("unused-hi"));
        let runtime =
            CaseRuntime::new(test_scenario(temporary.path(), Vec::new()), &options).unwrap();
        fs::write(&runtime.session_path, b"{not valid json\n").unwrap();

        let state = runtime
            .check_post_failure_state_invariants(true)
            .unwrap_err();
        let combined = merge_post_failure_state_invariant(
            Err(anyhow!("timed out waiting for expected transition")),
            Err(state),
        )
        .unwrap_err();
        let message = format!("{combined:#}");
        assert!(message.contains("timed out waiting for expected transition"));
        assert!(message.contains("could not parse session JSONL"));
        assert_eq!(
            classify_failure(&combined),
            CaseFailureKind::InfrastructureFailure
        );
    }

    #[test]
    fn post_failure_trace_allows_killed_active_turn_but_rejects_autonomous_restart() {
        let killed_active = [json!({
            "schema_version": 1,
            "sequence": 0,
            "process_id": 7,
            "run_id": "test-run",
            "event": "turn_started",
            "data": {"origin": "user", "queue_depth": 0},
        })];
        assert_eq!(
            validate_event_stream_safety(&killed_active).unwrap(),
            Some(0)
        );

        let autonomous = [
            json!({
                "schema_version": 1,
                "sequence": 0,
                "process_id": 7,
                "run_id": "test-run",
                "event": "turn_started",
                "data": {"origin": "user", "queue_depth": 0},
            }),
            json!({
                "schema_version": 1,
                "sequence": 1,
                "process_id": 7,
                "run_id": "test-run",
                "event": "turn_settled",
                "data": {"outcome": {"status": "failed"}, "queue_depth": 0},
            }),
            json!({
                "schema_version": 1,
                "sequence": 2,
                "process_id": 7,
                "run_id": "test-run",
                "event": "turn_started",
                "data": {"origin": "plan_drive", "queue_depth": 0},
            }),
        ];
        let error = validate_event_stream_safety(&autonomous).unwrap_err();
        assert!(format!("{error:#}").contains("autonomous"));
        assert_eq!(
            classify_failure(&error),
            CaseFailureKind::InfrastructureLoop
        );
    }

    #[test]
    fn prompt_queue_accounting_accepts_more_than_64_balanced_items() {
        let mut events = vec![lifecycle_event(0, "ready", json!({"queue_depth": 0}))];
        let mut sequence = 1_u64;
        for index in 0..65_u64 {
            events.push(lifecycle_event(
                sequence,
                "prompt_queued",
                json!({
                    "prompt_fingerprint": format!("prompt-{index}"),
                    "queue_depth": index + 1,
                }),
            ));
            sequence += 1;
        }
        for index in 0..65_u64 {
            let event = if index % 2 == 0 {
                "prompt_dequeued"
            } else {
                "prompt_removed"
            };
            events.push(lifecycle_event(
                sequence,
                event,
                json!({
                    "prompt_fingerprint": format!("prompt-{index}"),
                    "queue_depth": 64 - index,
                }),
            ));
            sequence += 1;
        }
        events.push(lifecycle_event(
            sequence,
            "session_ended",
            json!({"queue_depth": 0}),
        ));

        assert_eq!(validate_event_stream_safety(&events).unwrap(), None);
    }

    #[test]
    fn prompt_queue_accounting_accepts_startup_enqueue_before_ready() {
        let events = [
            lifecycle_event(
                0,
                "prompt_queued",
                json!({"prompt_fingerprint": "restored-drive", "queue_depth": 1}),
            ),
            lifecycle_event(1, "ready", json!({"queue_depth": 1})),
            lifecycle_event(
                2,
                "prompt_dequeued",
                json!({"prompt_fingerprint": "restored-drive", "queue_depth": 0}),
            ),
            lifecycle_event(3, "session_ended", json!({"queue_depth": 0})),
        ];

        assert_eq!(validate_event_stream_safety(&events).unwrap(), None);
    }

    #[test]
    fn prompt_queue_accounting_preserves_duplicate_fingerprint_multiplicity() {
        let events = [
            lifecycle_event(0, "ready", json!({"queue_depth": 0})),
            lifecycle_event(
                1,
                "prompt_queued",
                json!({"prompt_fingerprint": "same", "queue_depth": 1}),
            ),
            lifecycle_event(
                2,
                "prompt_queued",
                json!({"prompt_fingerprint": "same", "queue_depth": 2}),
            ),
            lifecycle_event(
                3,
                "prompt_dequeued",
                json!({"prompt_fingerprint": "same", "queue_depth": 1}),
            ),
            lifecycle_event(
                4,
                "prompt_removed",
                json!({"prompt_fingerprint": "same", "queue_depth": 0}),
            ),
        ];

        assert_eq!(validate_event_stream_safety(&events).unwrap(), None);
    }

    #[test]
    fn prompt_queue_accounting_rejects_lost_or_unqueued_items() {
        let lost = [
            lifecycle_event(0, "ready", json!({"queue_depth": 0})),
            lifecycle_event(
                1,
                "prompt_queued",
                json!({"prompt_fingerprint": "kept", "queue_depth": 1}),
            ),
            lifecycle_event(2, "session_ended", json!({"queue_depth": 0})),
        ];
        let lost_error = validate_event_stream_safety(&lost).unwrap_err();
        assert!(format!("{lost_error:#}").contains("traced prompt multiset has depth 1"));
        assert_eq!(
            classify_failure(&lost_error),
            CaseFailureKind::InfrastructureLoop
        );

        let unqueued = [
            lifecycle_event(0, "ready", json!({"queue_depth": 0})),
            lifecycle_event(
                1,
                "prompt_queued",
                json!({"prompt_fingerprint": "expected", "queue_depth": 1}),
            ),
            lifecycle_event(
                2,
                "prompt_dequeued",
                json!({"prompt_fingerprint": "different", "queue_depth": 0}),
            ),
        ];
        let unqueued_error = validate_event_stream_safety(&unqueued).unwrap_err();
        assert!(format!("{unqueued_error:#}").contains("consumed unqueued fingerprint"));
        assert_eq!(
            classify_failure(&unqueued_error),
            CaseFailureKind::InfrastructureLoop
        );
    }

    #[test]
    fn prompt_queue_accounting_rejects_negative_or_missing_depth() {
        let negative = [lifecycle_event(0, "ready", json!({"queue_depth": -1}))];
        let negative_error = validate_event_stream_safety(&negative).unwrap_err();
        assert!(format!("{negative_error:#}").contains("nonnegative integer"));

        let missing = [lifecycle_event(
            0,
            "prompt_queued",
            json!({"prompt_fingerprint": "orphan"}),
        )];
        let missing_error = validate_event_stream_safety(&missing).unwrap_err();
        assert!(format!("{missing_error:#}").contains("has no queue_depth"));
        assert_eq!(
            classify_failure(&missing_error),
            CaseFailureKind::InfrastructureLoop
        );
    }

    #[test]
    fn substring_occurrences_counts_inside_and_across_matching_records() {
        let temporary = tempfile::tempdir().unwrap();
        let options = scripted_options(temporary.path(), temporary.path().join("unused-hi"));
        let assertion = Assertion::SubstringOccurrences {
            source: RecordSource::Session,
            equals: BTreeMap::from([("/role".into(), json!("Assistant"))]),
            contains: BTreeMap::new(),
            pointer: "/content/0/Text".into(),
            substring: "TOKEN".into(),
            exact: 3,
        };
        let runtime = CaseRuntime::new(
            test_scenario(temporary.path(), vec![assertion.clone()]),
            &options,
        )
        .unwrap();
        fs::write(
            &runtime.session_path,
            concat!(
                "{\"role\":\"Assistant\",\"content\":[{\"Text\":\"TOKEN TOKEN\"}]}\n",
                "{\"role\":\"User\",\"content\":[{\"Text\":\"TOKEN\"}]}\n",
                "{\"role\":\"Assistant\",\"content\":[{\"Text\":\"TOKEN\"}]}\n",
            ),
        )
        .unwrap();

        runtime.evaluate_assertion(&assertion).unwrap();
        let duplicate_failure = Assertion::SubstringOccurrences {
            source: RecordSource::Session,
            equals: BTreeMap::from([("/role".into(), json!("Assistant"))]),
            contains: BTreeMap::new(),
            pointer: "/content/0/Text".into(),
            substring: "TOKEN".into(),
            exact: 2,
        };
        let message = runtime
            .evaluate_assertion(&duplicate_failure)
            .unwrap_err()
            .to_string();
        assert!(message.contains("got 3"), "{message}");
    }

    #[test]
    fn all_records_checks_every_selected_provider_request() {
        let temporary = tempfile::tempdir().unwrap();
        let options = scripted_options(temporary.path(), temporary.path().join("unused-hi"));
        let assertion = Assertion::AllRecords {
            source: RecordSource::ProviderRequests,
            where_equals: BTreeMap::new(),
            where_contains: BTreeMap::new(),
            equals: BTreeMap::from([
                ("/tool_count".into(), json!(0)),
                ("/native_tools_enabled".into(), json!(false)),
            ]),
            contains: BTreeMap::new(),
            at_least: 1,
        };
        let mut runtime =
            CaseRuntime::new(test_scenario(temporary.path(), Vec::new()), &options).unwrap();
        runtime.provider = None;
        fs::write(
            &runtime.events_path,
            concat!(
                "{\"event\":\"provider_request\",\"data\":{\"tool_count\":0,\"native_tools_enabled\":false}}\n",
                "{\"event\":\"provider_request\",\"data\":{\"tool_count\":0,\"native_tools_enabled\":false}}\n",
            ),
        )
        .unwrap();
        runtime.evaluate_assertion(&assertion).unwrap();

        fs::write(
            &runtime.events_path,
            concat!(
                "{\"event\":\"provider_request\",\"data\":{\"tool_count\":0,\"native_tools_enabled\":false}}\n",
                "{\"event\":\"provider_request\",\"data\":{\"tool_count\":1,\"native_tools_enabled\":true}}\n",
            ),
        )
        .unwrap();
        let message = runtime
            .evaluate_assertion(&assertion)
            .unwrap_err()
            .to_string();
        assert!(message.contains("source record 1"), "{message}");
    }

    #[cfg(unix)]
    #[test]
    fn workspace_file_checks_never_follow_outside_or_dangling_symlinks() {
        use std::os::unix::fs::symlink;

        let temporary = tempfile::tempdir().unwrap();
        let outside = temporary.path().join("outside-secret.txt");
        fs::write(&outside, "host secret\n").unwrap();
        let options = scripted_options(temporary.path(), temporary.path().join("unused-hi"));
        let runtime =
            CaseRuntime::new(test_scenario(temporary.path(), Vec::new()), &options).unwrap();

        symlink(&outside, runtime.workspace.join("proof.txt")).unwrap();
        let outside_read = Assertion::File {
            path: "proof.txt".into(),
            exists: true,
            contains: Some("host secret".into()),
            equals: None,
        };
        let error = runtime.evaluate_assertion(&outside_read).unwrap_err();
        assert!(
            error.to_string().contains("traverses a symlink"),
            "{error:#}"
        );

        symlink(
            temporary.path().join("target-that-does-not-exist"),
            runtime.workspace.join("dangling.txt"),
        )
        .unwrap();
        let false_absence = Assertion::File {
            path: "dangling.txt".into(),
            exists: false,
            contains: None,
            equals: None,
        };
        let error = runtime.evaluate_assertion(&false_absence).unwrap_err();
        assert!(
            error.to_string().contains("traverses a symlink"),
            "{error:#}"
        );
    }

    #[test]
    fn workspace_patch_and_listing_assertions_use_normalized_contained_evidence() {
        let temporary = tempfile::tempdir().unwrap();
        let options = scripted_options(temporary.path(), temporary.path().join("unused-hi"));
        let runtime =
            CaseRuntime::new(test_scenario(temporary.path(), Vec::new()), &options).unwrap();
        fs::write(runtime.initial_workspace.join("changed.txt"), "before\n").unwrap();
        fs::write(runtime.workspace.join("changed.txt"), "after\n").unwrap();
        fs::create_dir(runtime.workspace.join("nested")).unwrap();
        fs::write(runtime.workspace.join("nested/new.txt"), "new\n").unwrap();

        let listing = Assertion::WorkspaceListing {
            contains: vec!["./nested//new.txt".into(), "changed.txt".into()],
            excludes: vec!["missing.txt".into()],
        };
        runtime.evaluate_assertion(&listing).unwrap();

        let patch =
            capture_workspace_patch(&runtime.initial_workspace, &runtime.workspace).unwrap();
        assert!(
            patch.contains("diff --git a/changed.txt b/changed.txt"),
            "{patch}"
        );
        assert!(
            patch.contains("diff --git a/nested/new.txt b/nested/new.txt"),
            "{patch}"
        );
        assert!(!patch.contains(temporary.path().to_string_lossy().as_ref()));
        assert!(!patch.contains("a/./"), "{patch}");
        assert!(!patch.contains("b/./"), "{patch}");

        let patch_assertion = Assertion::WorkspacePatch {
            contains: vec![
                "diff --git a/changed.txt b/changed.txt".into(),
                "+after".into(),
            ],
            excludes: vec!["outside-secret".into()],
            // Expected CRLF is normalized the same way as captured output.
            equals: Some(patch.replace('\n', "\r\n")),
        };
        runtime.evaluate_assertion(&patch_assertion).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn workspace_file_wait_rejects_symlink_intermediates() {
        use std::os::unix::fs::symlink;

        let temporary = tempfile::tempdir().unwrap();
        let outside = temporary.path().join("outside");
        fs::create_dir(&outside).unwrap();
        fs::write(outside.join("proof.txt"), "outside\n").unwrap();
        let options = scripted_options(temporary.path(), temporary.path().join("unused-hi"));
        let mut runtime =
            CaseRuntime::new(test_scenario(temporary.path(), Vec::new()), &options).unwrap();
        symlink(&outside, runtime.workspace.join("linked-dir")).unwrap();

        let error = runtime
            .wait_for_file("linked-dir/proof.txt", true, Duration::from_millis(1))
            .unwrap_err();
        assert!(
            error.to_string().contains("traverses a symlink"),
            "{error:#}"
        );
    }

    #[test]
    fn sequence_is_ordered_but_not_required_to_be_adjacent() {
        assert!(is_subsequence(
            &[json!("a"), json!("noise"), json!("b")],
            &[json!("a"), json!("b")],
        ));
        assert!(!is_subsequence(
            &[json!("b"), json!("a")],
            &[json!("a"), json!("b")],
        ));
    }

    #[test]
    fn record_sequence_filters_before_comparing_values() {
        let temporary = tempfile::tempdir().unwrap();
        let options = scripted_options(temporary.path(), temporary.path().join("unused-hi"));
        let runtime =
            CaseRuntime::new(test_scenario(temporary.path(), Vec::new()), &options).unwrap();
        fs::write(
            &runtime.events_path,
            concat!(
                "{\"event\":\"turn_started\",\"data\":{\"origin\":\"plan_drive\"}}\n",
                "{\"event\":\"prompt_queued\",\"data\":{\"origin\":\"user\"}}\n",
                "{\"event\":\"turn_started\",\"data\":{\"origin\":\"user\"}}\n",
                "{\"event\":\"prompt_dequeued\",\"data\":{\"origin\":\"user\"}}\n",
                "{\"event\":\"turn_started\",\"data\":{\"origin\":\"plan_drive\"}}\n",
            ),
        )
        .unwrap();
        let assertion = Assertion::RecordSequence {
            source: RecordSource::Events,
            where_equals: BTreeMap::from([("/event".into(), json!("turn_started"))]),
            where_contains: BTreeMap::new(),
            pointer: "/data/origin".into(),
            values: vec![json!("plan_drive"), json!("user"), json!("plan_drive")],
        };

        runtime.evaluate_assertion(&assertion).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn absence_wait_rejects_a_matching_event_emitted_during_the_window() {
        let temporary = tempfile::tempdir().unwrap();
        let options = scripted_options(temporary.path(), temporary.path().join("unused-hi"));
        let mut runtime =
            CaseRuntime::new(test_scenario(temporary.path(), Vec::new()), &options).unwrap();
        let args = vec!["-c".to_owned(), "sleep 30".to_owned()];
        runtime.process = Some(
            PtyProcess::spawn(SpawnSpec {
                executable: Path::new("/bin/sh"),
                args: &args,
                cwd: &runtime.workspace,
                env: &BTreeMap::new(),
                cols: 80,
                rows: 24,
            })
            .unwrap(),
        );
        let events_path = runtime.events_path.clone();
        let writer = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(30));
            fs::write(
                events_path,
                "{\"event\":\"turn_started\",\"data\":{\"origin\":\"plan_drive\"}}\n",
            )
            .unwrap();
        });

        let error = runtime
            .wait_for_event_absence(
                &BTreeMap::from([
                    ("/event".into(), json!("turn_started")),
                    ("/data/origin".into(), json!("plan_drive")),
                ]),
                &BTreeMap::new(),
                Duration::from_millis(500),
            )
            .unwrap_err();
        writer.join().unwrap();
        assert!(error.to_string().contains("unexpected TUI event"));
        runtime.process_mut().unwrap().terminate_group().unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn marked_process_wait_observes_a_live_non_leader_descendant() {
        let temporary = tempfile::tempdir().unwrap();
        let options = scripted_options(temporary.path(), temporary.path().join("unused-hi"));
        let mut runtime =
            CaseRuntime::new(test_scenario(temporary.path(), Vec::new()), &options).unwrap();
        let args = vec!["-c".to_owned(), "sleep 30 & wait".to_owned()];
        runtime.process = Some(
            PtyProcess::spawn(SpawnSpec {
                executable: Path::new("/bin/sh"),
                args: &args,
                cwd: &runtime.workspace,
                env: &BTreeMap::new(),
                cols: 80,
                rows: 24,
            })
            .unwrap(),
        );

        runtime
            .wait_for_marked_process("sleep", 1, Duration::from_secs(2))
            .unwrap();
        runtime.process_mut().unwrap().terminate_group().unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn checkpoint_ancestry_retains_and_reaps_an_unmarked_escaped_descendant() {
        let temporary = tempfile::tempdir().unwrap();
        let options = scripted_options(temporary.path(), temporary.path().join("unused-hi"));
        let mut runtime =
            CaseRuntime::new(test_scenario(temporary.path(), Vec::new()), &options).unwrap();
        let executable = std::env::current_exe().unwrap();
        let args = vec![
            "--exact".to_owned(),
            "runner::tests::escaped_descendant_helper".to_owned(),
            "--nocapture".to_owned(),
        ];
        let env = BTreeMap::from([(ESCAPED_DESCENDANT_HELPER_ENV.to_owned(), "1".to_owned())]);
        runtime.process = Some(
            PtyProcess::spawn(SpawnSpec {
                executable: &executable,
                args: &args,
                cwd: &runtime.workspace,
                env: &env,
                cols: 80,
                rows: 24,
            })
            .unwrap(),
        );

        let deadline = Instant::now() + Duration::from_secs(3);
        // The fork is briefly visible in the leader's original process group
        // before the child completes setsid(). Keep sampling until both the
        // ancestry and escaped-group checkpoints are present; seeing only the
        // PID is not yet proof that the transition under test occurred.
        while (runtime.observed_descendant_pids.is_empty()
            || runtime.observed_descendant_groups.is_empty())
            && Instant::now() < deadline
        {
            runtime.observe_current_descendants().unwrap();
            std::thread::sleep(POLL_INTERVAL);
        }
        assert!(!runtime.observed_descendant_pids.is_empty());
        assert!(!runtime.observed_descendant_groups.is_empty());
        runtime.process_mut().unwrap().terminate_group().unwrap();
        runtime.archive_process().unwrap();

        let escaped = runtime.inspect_owned_processes();
        assert!(
            escaped.iter().any(|process| {
                process["source"] == "observed_ppid_ancestry"
                    && process["command"]
                        .as_str()
                        .is_some_and(|command| command.contains("sleep 30"))
            }),
            "escaped descendant was not retained: {escaped:?}"
        );
        cleanup_observed_processes(
            &runtime.observed_descendant_pids,
            &runtime.observed_descendant_groups,
        )
        .unwrap();
        let deadline = Instant::now() + Duration::from_secs(2);
        while !runtime.inspect_owned_processes().is_empty() && Instant::now() < deadline {
            std::thread::sleep(POLL_INTERVAL);
        }
        assert!(runtime.inspect_owned_processes().is_empty());
    }

    #[test]
    fn screen_normalization_removes_workspace_and_trailing_cells() {
        let normalized = normalize_screen("/tmp/ws/a  \n\n", Path::new("/tmp/ws"));
        assert_eq!(normalized, "<WORKSPACE>/a\n");
    }

    #[test]
    fn screen_normalization_removes_the_whole_random_isolation_identity() {
        let workspace = Path::new("/tmp/live-case-r4nd0m/workspace");
        let screen = concat!(
            "/tmp/live-case-r4nd0m/workspace/src/main.rs   \r\n",
            "/tmp/live-case-r4nd0m/home/.config/hi.toml\r\n",
            "run live-case-r4nd0m\r\n\r\n",
        );

        let normalized = normalize_screen(screen, workspace);
        assert_eq!(
            normalized,
            "<WORKSPACE>/src/main.rs\n<ISOLATION>/home/.config/hi.toml\nrun <ISOLATION>\n"
        );
        assert!(!normalized.contains("/tmp/live-case-r4nd0m"));
        assert!(!normalized.contains("live-case-r4nd0m"));
    }

    #[test]
    fn provider_request_wait_uses_immediate_tui_evidence_without_a_fake_server() {
        let temporary = tempfile::tempdir().unwrap();
        let options = scripted_options(temporary.path(), temporary.path().join("unused-hi"));
        let mut runtime =
            CaseRuntime::new(test_scenario(temporary.path(), Vec::new()), &options).unwrap();
        // The no-provider branch is the live-mode path. It consumes the
        // credential-free top-level records emitted before turn settlement.
        runtime.provider = None;
        fs::write(
            &runtime.events_path,
            concat!(
                "{\"event\":\"ui_event\",\"data\":{\"kind\":\"reasoning\"}}\n",
                "{\"event\":\"provider_request\",\"data\":{\"accepted\":true}}\n",
                "{\"event\":\"provider_request\",\"data\":{\"response_status\":200}}\n",
            ),
        )
        .unwrap();

        runtime
            .wait_for_provider_requests(2, Duration::from_millis(10))
            .unwrap();
    }

    #[test]
    fn early_failure_evaluates_every_remaining_assertion_for_evidence() {
        let temporary = tempfile::tempdir().unwrap();
        let assertions = vec![
            Assertion::Exit { code: 0 },
            Assertion::Records {
                source: RecordSource::Events,
                equals: BTreeMap::from([("/event".into(), json!("never"))]),
                contains: BTreeMap::new(),
                exact: Some(0),
                at_least: None,
                at_most: None,
            },
        ];
        let options = scripted_options(temporary.path(), temporary.path().join("unused-hi"));
        let mut runtime =
            CaseRuntime::new(test_scenario(temporary.path(), assertions), &options).unwrap();

        runtime.complete_pending_assertion_evidence();

        assert_eq!(runtime.assertions.len(), 2);
        assert_eq!(runtime.assertions[0]["index"], 0);
        assert_eq!(runtime.assertions[0]["passed"], false);
        assert!(
            runtime.assertions[0]["failure"]
                .as_str()
                .is_some_and(|failure| failure.contains("expected exit code 0"))
        );
        assert_eq!(runtime.assertions[1]["index"], 1);
        assert_eq!(runtime.assertions[1]["passed"], true);
        assert!(
            runtime
                .assertions
                .iter()
                .all(|entry| entry["evaluated_after_failure"] == true)
        );
    }

    #[cfg(unix)]
    #[test]
    fn spawned_hi_receives_trace_capture_off_as_harness_owned_control() {
        use std::os::unix::fs::PermissionsExt;

        let temporary = tempfile::tempdir().unwrap();
        let fake_hi = temporary.path().join("fake-hi");
        fs::write(
            &fake_hi,
            r#"#!/bin/sh
events=
while [ "$#" -gt 0 ]; do
    if [ "$1" = "--tui-events-jsonl" ]; then
        events=$2
        shift 2
    else
        shift
    fi
done
printf '%s' "${HI_TRACE_CAPTURE-unset}" > trace-capture.txt
printf '%s\n' '{"schema_version":1,"sequence":0,"event":"ready","data":{}}' > "$events"
IFS= read -r _
exit 0
"#,
        )
        .unwrap();
        let mut permissions = fs::metadata(&fake_hi).unwrap().permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(&fake_hi, permissions).unwrap();
        let options = scripted_options(temporary.path(), fake_hi);
        let mut runtime =
            CaseRuntime::new(test_scenario(temporary.path(), Vec::new()), &options).unwrap();

        runtime.spawn_hi().unwrap();
        assert_eq!(
            fs::read_to_string(runtime.workspace.join("trace-capture.txt")).unwrap(),
            "off"
        );
        runtime.quit().unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn sibling_write_fails_case_and_is_preserved_as_relative_bundle_evidence() {
        use std::os::unix::fs::PermissionsExt;

        let temporary = tempfile::tempdir().unwrap();
        let fake_hi = temporary.path().join("fake-hi-sibling-write");
        fs::write(
            &fake_hi,
            r#"#!/bin/sh
events=
while [ "$#" -gt 0 ]; do
    if [ "$1" = "--tui-events-jsonl" ]; then
        events=$2
        shift 2
    else
        shift
    fi
done
printf '%s' 'model-owned' > ../home/unexpected-tool-write.txt
printf '%s\n' '{"schema_version":1,"sequence":0,"event":"ready","data":{}}' > "$events"
IFS= read -r _
printf '%s\n' '{"schema_version":1,"sequence":1,"event":"session_ended","data":{}}' >> "$events"
exit 0
"#,
        )
        .unwrap();
        let mut permissions = fs::metadata(&fake_hi).unwrap().permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(&fake_hi, permissions).unwrap();
        let options = scripted_options(temporary.path(), fake_hi);
        let mut runtime =
            CaseRuntime::new(test_scenario(temporary.path(), Vec::new()), &options).unwrap();

        runtime.spawn_hi().unwrap();
        runtime.quit().unwrap();
        let failure = runtime
            .check_isolation_mutation_invariant()
            .expect_err("a tool write beside the workspace must fail containment");
        assert!(
            format!("{failure:#}").contains("home/unexpected-tool-write.txt"),
            "{failure:#}"
        );
        assert_eq!(
            classify_failure(&failure),
            CaseFailureKind::InfrastructureFailure
        );

        let artifact_dir = options.artifacts.join("sibling-write");
        runtime
            .finish_bundle(&artifact_dir, Duration::from_millis(1), Some(&failure))
            .unwrap();
        let evidence: Value = serde_json::from_slice(
            &fs::read(artifact_dir.join("isolation-evidence.json"))
                .expect("isolation evidence artifact"),
        )
        .unwrap();
        assert_eq!(evidence["unexpected_mutation_count"], 1);
        assert_eq!(
            evidence["mutations"]
                .as_array()
                .unwrap()
                .iter()
                .find(|mutation| mutation["path"] == "home/unexpected-tool-write.txt")
                .unwrap()["disposition"],
            "unexpected_outside_workspace"
        );
        let encoded = serde_json::to_string(&evidence).unwrap();
        assert!(!encoded.contains(temporary.path().to_string_lossy().as_ref()));
        let summary: Value = serde_json::from_slice(
            &fs::read(artifact_dir.join("summary.json")).expect("case summary"),
        )
        .unwrap();
        assert_eq!(summary["status"], "infrastructure_failure");
        assert_eq!(summary["unexpected_isolation_mutation_count"], 1);
    }

    #[cfg(unix)]
    #[test]
    fn failure_bundle_captures_exit_status_observed_during_cleanup() {
        let temporary = tempfile::tempdir().unwrap();
        let options = scripted_options(temporary.path(), temporary.path().join("unused-hi"));
        let mut runtime =
            CaseRuntime::new(test_scenario(temporary.path(), Vec::new()), &options).unwrap();
        let args = vec!["-c".to_owned(), "exit 23".to_owned()];
        let mut process = PtyProcess::spawn(SpawnSpec {
            executable: Path::new("/bin/sh"),
            args: &args,
            cwd: &runtime.workspace,
            env: &BTreeMap::new(),
            cols: 80,
            rows: 24,
        })
        .unwrap();
        let status = process
            .wait_until(Instant::now() + Duration::from_secs(2))
            .unwrap()
            .expect("fixture leader should exit");
        assert_eq!(status.exit_code(), 23);
        runtime.process = Some(process);
        assert_eq!(runtime.exit_code, None);

        let artifact_dir = options.artifacts.join("cleanup-exit-evidence");
        let failure = anyhow!("synthetic action failure");
        runtime
            .finish_bundle(&artifact_dir, Duration::from_millis(1), Some(&failure))
            .unwrap();

        let process: Value = serde_json::from_slice(
            &fs::read(artifact_dir.join("process.json")).expect("process evidence"),
        )
        .unwrap();
        assert_eq!(process["exit_code"], 23);
    }

    #[cfg(unix)]
    #[test]
    fn poisoned_terminal_locks_fail_as_infrastructure_with_explicit_evidence_markers() {
        let temporary = tempfile::tempdir().unwrap();
        let options = scripted_options(temporary.path(), temporary.path().join("unused-hi"));
        let mut runtime =
            CaseRuntime::new(test_scenario(temporary.path(), Vec::new()), &options).unwrap();
        let args = vec!["-c".to_owned(), "sleep 30".to_owned()];
        runtime.process = Some(
            PtyProcess::spawn(SpawnSpec {
                executable: Path::new("/bin/sh"),
                args: &args,
                cwd: &runtime.workspace,
                env: &BTreeMap::new(),
                cols: 80,
                rows: 24,
            })
            .unwrap(),
        );
        runtime
            .process_ref()
            .unwrap()
            .poison_evidence_locks_for_test();

        let failure = runtime.prepare_failure_evidence(true).unwrap_err();
        assert_eq!(
            classify_failure(&failure),
            CaseFailureKind::InfrastructureFailure
        );
        assert!(
            String::from_utf8_lossy(&runtime.historical_raw.bytes)
                .contains("raw terminal evidence unavailable")
        );
        assert!(runtime.screens["failure"].contains("virtual screen evidence unavailable"));
    }

    #[cfg(unix)]
    #[test]
    fn emitted_real_failure_bundle_replays_and_reproduces_the_failure() {
        let temporary = tempfile::tempdir().unwrap();
        let false_binary = PathBuf::from("/usr/bin/false");
        assert!(false_binary.is_file());
        let mut scenario = test_scenario(temporary.path(), Vec::new());
        scenario.name = "real-replay-failure".into();
        scenario.timeout_ms = 2_000;
        let options = scripted_options(temporary.path(), false_binary.clone());

        let first = run_scenario(scenario, &options);
        assert!(matches!(first.status, CaseStatus::Failed));
        let replay_path = first.artifact_dir.join("replay.toml");
        assert!(minimal_replay_is_complete(&first.artifact_dir));

        let replay_artifacts = temporary.path().join("replayed");
        let failure = replay(&false_binary, &replay_path, &replay_artifacts, false)
            .expect_err("the deterministic failing executable must fail again on replay");
        assert!(format!("{failure:#}").contains("replay failed"));
        let reproduced = fs::read_dir(&replay_artifacts)
            .unwrap()
            .filter_map(std::result::Result::ok)
            .map(|entry| entry.path())
            .find(|path| path.join("summary.json").is_file())
            .expect("replay must emit its own failure bundle");
        assert!(minimal_replay_is_complete(&reproduced));
        let summary: Value =
            serde_json::from_slice(&fs::read(reproduced.join("summary.json")).unwrap()).unwrap();
        assert_ne!(summary["status"], "passed");
    }

    #[test]
    fn precleanup_leak_evidence_survives_later_empty_scans() {
        let leak = json!({
            "source": "run_marker",
            "pid": 42,
            "ppid": 1,
            "pgid": 42,
            "command": "escaped-child",
        });
        let mut recorded = Vec::new();

        preserve_process_leaks(&mut recorded, vec![leak.clone()]);
        preserve_process_leaks(&mut recorded, Vec::new());
        preserve_process_leaks(&mut recorded, vec![leak.clone()]);

        assert_eq!(recorded, vec![leak]);
    }

    #[test]
    fn early_failure_cleanup_is_root_cause_and_preserves_all_prior_failures() {
        let temporary = tempfile::tempdir().unwrap();
        let options = scripted_options(temporary.path(), temporary.path().join("unused-hi"));
        let mut runtime =
            CaseRuntime::new(test_scenario(temporary.path(), Vec::new()), &options).unwrap();
        runtime.leaked_processes.push(json!({
            "source": "run_marker",
            "pid": 4242,
            "ppid": 1,
            "pgid": 4242,
            "command": "surviving-descendant",
        }));

        let cleanup = runtime.prepare_failure_evidence(true).unwrap_err();
        assert!(format!("{cleanup:#}").contains("surviving-descendant"));
        let before_cleanup = merge_live_route_invariant(
            Err(anyhow!("timed out waiting for turn settlement")),
            Err(anyhow!(
                "live provider evidence invariant failed at request 0"
            )),
        );
        let before_cleanup = merge_post_failure_state_invariant(
            before_cleanup,
            Err(anyhow!(
                "post-failure state invariant failed: malformed session JSONL"
            )),
        );
        let before_cleanup = merge_isolation_invariant(
            before_cleanup,
            Err(anyhow!(
                "isolation containment invariant failed: outside mutation"
            )),
        );
        let combined = merge_failure_cleanup_invariant(before_cleanup, Err(cleanup)).unwrap_err();
        let message = format!("{combined:#}");
        assert!(message.contains("timed out waiting for turn settlement"));
        assert!(message.contains("live provider evidence invariant"));
        assert!(message.contains("post-failure state invariant"));
        assert!(message.contains("isolation containment invariant"));
        assert!(message.contains("failure cleanup invariant failed"));
        assert!(
            combined
                .root_cause()
                .to_string()
                .contains("failure cleanup invariant failed")
        );
        assert_eq!(
            classify_failure(&combined),
            CaseFailureKind::InfrastructureFailure
        );
    }

    #[cfg(unix)]
    #[test]
    fn unsuccessful_ps_is_process_inspection_failure_not_an_empty_scan() {
        let observed =
            process_group_leaks_from_ps_output(&[4242], false, b"", b"permission denied\n");
        assert_eq!(observed.len(), 1);
        assert_eq!(
            observed[0]["inspection_error"],
            "process inspection failed: ps exited unsuccessfully"
        );
        assert_eq!(observed[0]["detail"], "permission denied");
        assert_eq!(
            classify_failure(&anyhow!(
                "leaked descendant processes: {}",
                serde_json::to_string(&observed).unwrap()
            )),
            CaseFailureKind::InfrastructureFailure
        );
    }

    #[test]
    fn failure_classification_separates_crashes_loops_and_timeouts() {
        assert_eq!(
            classify_failure(&anyhow!("hi exited early: terminated by signal 9")),
            CaseFailureKind::Crashed
        );
        assert_eq!(
            classify_failure(&anyhow!(
                "autonomous plan_drive turn started after failed settlement"
            )),
            CaseFailureKind::InfrastructureLoop
        );
        assert_eq!(
            classify_failure(&anyhow!("turn timed out at outer kill boundary")),
            CaseFailureKind::TimedOut
        );
        assert_eq!(
            classify_failure(&anyhow!("expected exactly 1 record, got 2")),
            CaseFailureKind::Scenario
        );
        assert_eq!(
            classify_failure(&anyhow!(
                "live provider evidence invariant failed at request 0: model mismatch"
            )),
            CaseFailureKind::InfrastructureFailure
        );
        assert_eq!(
            classify_failure(&anyhow!("virtual terminal parser lock was poisoned")),
            CaseFailureKind::InfrastructureFailure
        );
        assert_eq!(
            classify_failure(&anyhow!("raw terminal evidence lock was poisoned")),
            CaseFailureKind::InfrastructureFailure
        );
    }

    #[test]
    fn live_route_mismatch_is_a_hard_infrastructure_failure_in_case_and_suite_summaries() {
        let temporary = tempfile::tempdir().unwrap();
        let route = LiveRoute::new(
            "pipenetwork",
            "pipe/deepseek-v4-flash-0731",
            "https://api.pipenetwork.ai/v1",
        )
        .unwrap();
        let request = serde_json::to_value(hi_ai::WireAudit {
            provider: "openai_compatible".into(),
            route: "https://wrong.example/v1".into(),
            model: route.model.clone(),
            output_token_parameter: "max_tokens".into(),
            max_output_tokens: 512,
            request_attempt: 1,
            accepted: true,
            response_status: Some(200),
            ..hi_ai::WireAudit::default()
        })
        .unwrap();
        let events = [json!({"event": "provider_request", "data": request.clone()})];
        let failure = validate_live_provider_event_route(&events, Some(&route)).unwrap_err();
        assert!(format!("{failure:#}").contains("route mismatch"));
        assert_eq!(
            classify_failure(&failure),
            CaseFailureKind::InfrastructureFailure
        );

        let artifacts = temporary.path().join("artifacts");
        fs::create_dir(&artifacts).unwrap();
        let workspace = temporary.path().join("workspace");
        fs::create_dir(&workspace).unwrap();
        let scenario = test_scenario(temporary.path(), Vec::new());
        let raw = RawTerminal::default();
        let screens = BTreeMap::new();
        let empty = json!({});
        let failure_text = format!("{failure:#}");
        let paths = crate::artifacts::write_case_bundle(
            &artifacts,
            Path::new("route-mismatch"),
            &crate::artifacts::BundleInput {
                scenario: &scenario,
                mode: "live",
                live_route: Some(&route),
                status: bundle_status_for_failure(classify_failure(&failure)),
                seed: None,
                duration_ms: 7,
                failure: Some(&failure_text),
                tui_events: &[],
                raw_terminal: &raw,
                screens: &screens,
                provider_requests: std::slice::from_ref(&request),
                redaction_values: &[],
                session_jsonl: &[],
                initial_workspace_root: Some(&workspace),
                workspace_root: &workspace,
                workspace_patch: "",
                isolation_evidence: &empty,
                process: &empty,
                assertions: &empty,
                timings: &empty,
                result: &empty,
            },
        )
        .unwrap();
        let artifact_dir = paths.directory;
        let case_summary: Value =
            serde_json::from_slice(&fs::read(artifact_dir.join("summary.json")).unwrap()).unwrap();
        assert_eq!(case_summary["status"], "infrastructure_failure");
        assert_eq!(case_summary["provider_request_count"], 1);
        assert_eq!(case_summary["provider_chat_request_count"], 1);
        assert_eq!(case_summary["live_route"]["model"], route.model);
        assert!(artifact_dir.join("replay.toml").is_file());

        let report = CaseReport {
            name: "route-mismatch".into(),
            status: CaseStatus::Failed,
            failure_kind: Some(CaseFailureKind::InfrastructureFailure),
            duration_ms: 7,
            failure: Some(format!("{failure:#}")),
            artifact_dir,
            provider_request_count: 1,
            provider_chat_request_count: 1,
            provider_accepted_request_count: 1,
            provider_response_status_counts: BTreeMap::from([(200, 1)]),
        };
        let summary = suite_summary(RunMode::Live, Some(&route), &[report]);
        assert_eq!(summary["passed"], 0);
        assert_eq!(summary["failed"], 1);
        assert_eq!(summary["infrastructure_failure_count"], 1);
    }

    #[test]
    fn live_route_mismatch_overrides_a_preexisting_action_timeout() {
        let route = LiveRoute::new(
            "pipenetwork",
            "pipe/deepseek-v4-flash-0731",
            "https://api.pipenetwork.ai/v1",
        )
        .unwrap();
        let request = serde_json::to_value(hi_ai::WireAudit {
            provider: "openai_compatible".into(),
            route: "https://wrong.example/v1".into(),
            model: route.model.clone(),
            output_token_parameter: "max_tokens".into(),
            max_output_tokens: 512,
            request_attempt: 1,
            accepted: true,
            response_status: Some(200),
            ..hi_ai::WireAudit::default()
        })
        .unwrap();
        let events = [json!({"event": "provider_request", "data": request})];
        let invariant = validate_live_provider_event_route(&events, Some(&route));
        let combined = merge_live_route_invariant(
            Err(anyhow!("action 2 timed out waiting for settlement")),
            invariant,
        )
        .unwrap_err();

        let message = format!("{combined:#}");
        assert!(message.contains("action 2 timed out"));
        assert!(message.contains("route mismatch"));
        assert_eq!(
            classify_failure(&combined),
            CaseFailureKind::InfrastructureFailure
        );
    }

    #[test]
    fn provider_credentials_are_environment_only() {
        let secret = "live-secret-that-must-not-enter-argv";
        let (args, credential) =
            provider_launch_parts("openai", "test-model".into(), secret.into());

        assert!(!args.iter().any(|argument| argument == "--api-key"));
        assert!(args.iter().all(|argument| !argument.contains(secret)));
        assert_eq!(credential, ("HI_API_KEY".into(), secret.into()));

        let (pipe_args, pipe_credential) =
            provider_launch_parts("pipenetwork", "pipe/model".into(), secret.into());
        assert_eq!(pipe_args[1], "pipenetwork");
        assert!(pipe_args.iter().all(|argument| !argument.contains(secret)));
        assert_eq!(
            pipe_credential,
            ("PIPENETWORK_API_KEY".into(), secret.into())
        );
    }

    #[test]
    fn harness_credential_wins_even_if_unvalidated_scenario_env_contains_alias() {
        let mut env = BTreeMap::from([("SAFE_MARKER".into(), "before".into())]);
        let scenario_env = BTreeMap::from([
            ("SAFE_MARKER".into(), "after".into()),
            ("PIPENETWORK_API_KEY".into(), "attacker-value".into()),
        ]);

        extend_scenario_env_with_credential(
            &mut env,
            &scenario_env,
            ("PIPENETWORK_API_KEY".into(), "harness-value".into()),
        );

        assert_eq!(env["SAFE_MARKER"], "after");
        assert_eq!(env["PIPENETWORK_API_KEY"], "harness-value");
    }

    #[test]
    fn suite_summary_retains_the_non_secret_live_route() {
        let route = LiveRoute::new(
            "pipenetwork",
            "pipe/deepseek-v4-flash-0731",
            "https://api.pipenetwork.ai/v1",
        )
        .unwrap();
        let reports = vec![CaseReport {
            name: "live-canary".into(),
            status: CaseStatus::Passed,
            failure_kind: None,
            duration_ms: 12,
            failure: None,
            artifact_dir: PathBuf::from(
                "/private/tmp/host-specific-a81c9e/artifacts/live-canary-run-91",
            ),
            provider_request_count: 2,
            provider_chat_request_count: 2,
            provider_accepted_request_count: 1,
            provider_response_status_counts: BTreeMap::from([(200, 1), (503, 1)]),
        }];

        let summary = suite_summary(RunMode::Live, Some(&route), &reports);
        assert_eq!(summary["live_route"]["provider"], "pipenetwork");
        assert_eq!(
            summary["live_route"]["model"],
            "pipe/deepseek-v4-flash-0731"
        );
        assert_eq!(
            summary["live_route"]["base_url"],
            "https://api.pipenetwork.ai/v1"
        );
        assert_eq!(summary["passed"], 1);
        assert_eq!(summary["provider_request_count"], 2);
        assert_eq!(summary["provider_chat_request_count"], 2);
        assert_eq!(summary["provider_accepted_request_count"], 1);
        assert_eq!(summary["provider_response_status_counts"]["200"], 1);
        assert_eq!(summary["provider_response_status_counts"]["503"], 1);
        assert_eq!(summary["cases"][0]["provider_accepted_request_count"], 1);
        assert_eq!(summary["cases"][0]["artifact_dir"], "live-canary-run-91");
        let encoded = serde_json::to_string(&summary).unwrap();
        assert!(!encoded.contains("API_KEY"));
        assert!(!encoded.contains("/private/tmp/host-specific-a81c9e"));
    }

    #[test]
    fn recorded_live_route_wins_but_api_key_still_comes_from_environment() {
        let recorded = LiveRoute::new(
            "pipenetwork",
            "pipe/deepseek-v4-flash-0731",
            "https://api.pipenetwork.ai/v1",
        )
        .unwrap();
        let requested = std::cell::RefCell::new(Vec::new());
        let resolved = resolve_live_route_with(Some(&recorded), |name| {
            requested.borrow_mut().push(name.to_owned());
            (name == "HI_API_KEY").then(|| "environment-only-secret".to_owned())
        })
        .unwrap();

        assert_eq!(resolved, recorded);
        assert_eq!(requested.into_inner(), vec!["HI_API_KEY"]);
    }

    #[test]
    fn replay_metadata_reads_recorded_route_and_legacy_modes() {
        let temporary = tempfile::tempdir().unwrap();
        let current = temporary.path().join("current.toml");
        fs::write(
            &current,
            "# hi-smoke-replay-mode = live\n# hi-smoke-live-provider = pipenetwork\n# hi-smoke-live-model = pipe/deepseek-v4-flash-0731\n# hi-smoke-live-base-url = https://api.pipenetwork.ai/v1\nschema_version = 1\n",
        )
        .unwrap();
        let metadata = replay_metadata(&current).unwrap();
        assert_eq!(metadata.mode, RunMode::Live);
        assert_eq!(
            metadata.live_route,
            Some(
                LiveRoute::new(
                    "pipenetwork",
                    "pipe/deepseek-v4-flash-0731",
                    "https://api.pipenetwork.ai/v1",
                )
                .unwrap()
            )
        );
        let legacy = temporary.path().join("legacy.toml");
        fs::write(&legacy, "# mode = live\n").unwrap();
        assert_eq!(replay_metadata(&legacy).unwrap().mode, RunMode::Live);
        let plain = temporary.path().join("plain.toml");
        fs::write(&plain, "schema_version = 1\n").unwrap();
        assert_eq!(replay_metadata(&plain).unwrap().mode, RunMode::Scripted);

        let partial = temporary.path().join("partial.toml");
        fs::write(
            &partial,
            "# hi-smoke-replay-mode = live\n# hi-smoke-live-provider = pipenetwork\n",
        )
        .unwrap();
        assert!(replay_metadata(&partial).is_err());
    }

    #[test]
    fn setup_failure_still_writes_self_contained_live_replay() {
        let temporary = tempfile::tempdir().unwrap();
        let source = temporary.path().join("source");
        fs::create_dir(&source).unwrap();
        let fixture = source.join("fixture");
        fs::create_dir(&fixture).unwrap();
        fs::write(fixture.join("input.txt"), "fixture\n").unwrap();
        let scenario_path = source.join("scenario.toml");
        fs::write(
            &scenario_path,
            r#"
schema_version = 1
name = "setup-failure"

[workspace]
fixture = "fixture"
"#,
        )
        .unwrap();
        let scenario = Scenario::parse(&scenario_path).unwrap();
        let artifacts = temporary.path().join("artifacts");
        fs::create_dir(&artifacts).unwrap();
        // Force CaseRuntime setup to fail before the live environment check.
        fs::write(artifacts.join(".work"), "not a directory").unwrap();
        let report = run_scenario(
            scenario,
            &CaseOptions {
                hi_bin: temporary.path().join("unused-hi"),
                artifacts,
                mode: RunMode::Live,
                live_route: None,
                keep: false,
                seed: Some(9),
                sandbox_requirement: SandboxRequirement::UnitTestUnenforced,
            },
        );

        assert!(matches!(report.status, CaseStatus::Failed));
        assert!(minimal_replay_is_complete(&report.artifact_dir));
        let replay = report.artifact_dir.join("replay.toml");
        assert_eq!(replay_metadata(&replay).unwrap().mode, RunMode::Live);
        let replay_scenario = Scenario::parse(&replay).unwrap();
        let embedded = replay_scenario.workspace.fixture.unwrap();
        assert_eq!(
            fs::read_to_string(report.artifact_dir.join(embedded).join("input.txt")).unwrap(),
            "fixture\n"
        );
    }
}
