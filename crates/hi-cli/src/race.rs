//! Provider/configuration boundary for the TUI coding race.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::Arc;

use anyhow::{Context, Result, bail, ensure};
use hi_events::{
    ActivityObject, ActivityState, ActivityVerb, EventContext, EventKind, RunEvent,
    SemanticActivity,
};
use hi_policy::{
    ApprovalDecision, ApprovalId, CapabilityKind, CapabilityRequest, OperationDigest, ResourceScope,
};
use hi_race::{CandidateReport, CandidateState, RaceSnapshot, RaceSpec, RaceStatus, StageResult};
use serde_json::{Value, json};

use crate::bestof::{BestOf, BestOfTarget};
use crate::config::{self, Config};
use crate::event_store::publish_best_effort;
use crate::provider::provider_label;
use crate::report::pipeline_command;

pub(crate) fn build_tui_runner(
    config: Config,
    event_sink: Option<Arc<dyn hi_events::EventSink>>,
    approval_store: Option<Arc<dyn hi_policy::ApprovalStore>>,
) -> hi_tui::RaceRunner {
    Arc::new(move |request| {
        let config = config.clone();
        let event_sink = event_sink.clone();
        let approval_store = approval_store.clone();
        Box::pin(async move { run_request(config, request, event_sink, approval_store).await })
    })
}

pub(crate) fn build_setup_saver(workspace_root: PathBuf) -> hi_tui::RaceSetupSaver {
    Arc::new(move |targets| save_project_race_config(&workspace_root, &targets))
}

fn save_project_race_config(
    workspace_root: &Path,
    targets: &[hi_race::RaceTarget],
) -> Result<String> {
    let config_dir = workspace_root.join(".hi");
    std::fs::create_dir_all(&config_dir)
        .with_context(|| format!("creating project config directory {}", config_dir.display()))?;
    let path = config_dir.join("config.toml");
    let mut document = if path.is_file() {
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("reading project config {}", path.display()))?;
        text.parse::<toml::Value>()
            .with_context(|| format!("parsing project config {}", path.display()))?
    } else {
        toml::Value::Table(toml::map::Map::new())
    };
    let race = document
        .as_table_mut()
        .context("project config root is not a TOML table")?
        .entry("race")
        .or_insert_with(|| toml::Value::Table(toml::map::Map::new()));
    let race = race
        .as_table_mut()
        .context("project race config is not a table")?;
    race.insert("enabled".into(), toml::Value::Boolean(true));
    race.insert("max_candidates".into(), toml::Value::Integer(2));
    race.insert("max_concurrency".into(), toml::Value::Integer(2));
    race.insert(
        "targets".into(),
        toml::Value::Array(
            targets
                .iter()
                .map(|target| {
                    let mut table = toml::map::Map::new();
                    table.insert("name".into(), toml::Value::String(target.name.clone()));
                    table.insert(
                        "profile".into(),
                        toml::Value::String(target.profile.clone()),
                    );
                    table.insert("model".into(), toml::Value::String(target.model.clone()));
                    table.insert(
                        "priority".into(),
                        toml::Value::Integer(target.priority as i64),
                    );
                    toml::Value::Table(table)
                })
                .collect(),
        ),
    );
    let encoded = toml::to_string_pretty(&document).context("serializing project race config")?;
    std::fs::write(&path, encoded)
        .with_context(|| format!("writing project config {}", path.display()))?;
    Ok(path.display().to_string())
}

async fn run_request(
    config: Config,
    request: hi_tui::RaceRunRequest,
    event_sink: Option<Arc<dyn hi_events::EventSink>>,
    approval_store: Option<Arc<dyn hi_policy::ApprovalStore>>,
) -> Result<RaceSnapshot> {
    let (workspace_root, state_root) = crate::review_target::resolve_runtime_roots()?;
    let workspace_snapshot = hi_race::capture_workspace_snapshot(&workspace_root)
        .context("capturing the coding-race workspace")?;
    if request.apply && request.source_run_id.is_some() && request.artifact_root.is_some() {
        let expected = request
            .expected_workspace_digest
            .as_deref()
            .context("reviewed race is missing its workspace digest")?;
        if workspace_snapshot.digest != expected {
            if let Some(run_id) = request.source_run_id.as_deref() {
                publish_race_event(
                    event_sink.as_deref(),
                    EventKind::RaceWorkspaceConflict,
                    run_id,
                    "coding race apply rejected because the workspace changed",
                    ActivityState::Failed,
                );
            }
            bail!(
                "workspace changed since the race snapshot; review the current diff and rerun /race"
            );
        }
        return apply_saved_candidate(
            &workspace_root,
            &state_root,
            &request,
            event_sink,
            approval_store,
        )
        .await;
    }
    let run_id = uuid::Uuid::new_v4().to_string();
    let mut spec = RaceSpec::new(request.task.clone(), request.targets.clone());
    spec.run_id = run_id.clone();
    spec.max_candidates = request.max_candidates;
    spec.max_concurrency = request.max_concurrency;
    spec.verify_commands = request.verify_commands.clone();
    spec.fuzz = request.fuzz.clone();
    spec.workspace_digest = workspace_snapshot.digest.clone();
    spec.validate()?;

    publish_race_event(
        event_sink.as_deref(),
        EventKind::RaceStarted,
        &run_id,
        "coding race started",
        ActivityState::Running,
    );
    for target in &request.targets {
        publish_race_event(
            event_sink.as_deref(),
            EventKind::RaceCandidateStarted,
            &run_id,
            &format!("race candidate {} started", target.name),
            ActivityState::Running,
        );
    }

    let mut targets = Vec::with_capacity(request.targets.len());
    for target in &request.targets {
        let settings = config::resolve_named_profile(&config, &target.profile)
            .with_context(|| format!("resolving race target profile '{}';", target.profile))?;
        targets.push(BestOfTarget {
            name: target.name.clone(),
            provider: provider_label(settings.provider).to_string(),
            model: target.model.clone(),
            base_url: settings.base_url,
            api_key: settings.api_key,
            priority: target.priority,
        });
    }
    let verify = resolved_verify(&request.verify_commands)?;
    let exe = std::env::current_exe().context("locating the hi executable")?;
    let report_path = state_root
        .join("race-artifacts")
        .join(&run_id)
        .join("race.report.json");
    let max_candidates = request.max_candidates.min(targets.len() as u32).max(2);
    let state_root_for_run = state_root.clone();
    let workspace_root_for_run = workspace_root.clone();
    let task = request.task.clone();
    let fuzz = request.fuzz.clone();
    let apply = request.apply;
    let report_path_for_run = report_path.clone();
    let expected_workspace_digest = spec.workspace_digest.clone();
    let output = tokio::task::spawn_blocking(move || {
        let result = crate::bestof::run(&BestOf {
            exe: &exe,
            provider: "configured-roster",
            model: "race",
            base_url: "",
            api_key: "",
            verify: &verify,
            prompt: &task,
            candidates: max_candidates,
            max_steps: None,
            max_verify: 2,
            workspace_root: &workspace_root_for_run,
            state_root: &state_root_for_run,
            report: Some(&report_path_for_run),
            targets: Some(&targets),
            max_concurrency: spec.max_concurrency,
            apply,
            fuzz: fuzz.as_ref(),
            expected_workspace_digest: Some(&expected_workspace_digest),
        })?;
        Ok::<bool, anyhow::Error>(result)
    })
    .await
    .context("coding race worker failed")??;
    let _ = output;
    let report_text = std::fs::read_to_string(&report_path)
        .with_context(|| format!("reading coding race report {}", report_path.display()))?;
    let report: Value = serde_json::from_str(&report_text).context("parsing coding race report")?;
    let snapshot = snapshot_from_report(&report, &spec, &request.targets, report_path);
    for candidate in &snapshot.candidates {
        publish_race_event(
            event_sink.as_deref(),
            EventKind::RaceCandidateCompleted,
            &run_id,
            &format!(
                "race candidate {} {}",
                candidate.target.name,
                if candidate.eligible() {
                    "passed"
                } else {
                    "failed"
                }
            ),
            if candidate.eligible() {
                ActivityState::Succeeded
            } else {
                ActivityState::Failed
            },
        );
        publish_race_event(
            event_sink.as_deref(),
            EventKind::RaceCandidateScored,
            &run_id,
            &format!("race candidate {} scored", candidate.target.name),
            if candidate.eligible() {
                ActivityState::Succeeded
            } else {
                ActivityState::Failed
            },
        );
    }
    publish_race_event(
        event_sink.as_deref(),
        if request.apply {
            EventKind::RaceApplied
        } else {
            EventKind::RaceWinnerReady
        },
        &run_id,
        if request.apply {
            "coding race applied"
        } else {
            "coding race winner ready for review"
        },
        if request.apply {
            ActivityState::Succeeded
        } else {
            ActivityState::Waiting
        },
    );
    Ok(snapshot)
}

fn resolved_verify(commands: &[String]) -> Result<String> {
    pipeline_command(
        &commands
            .iter()
            .enumerate()
            .map(|(index, command)| {
                hi_agent::VerifyStage::new(format!("race-verify-{}", index + 1), command.clone())
            })
            .collect::<Vec<_>>(),
    )
    .context("coding race requires a verification pipeline")
}

async fn apply_saved_candidate(
    workspace_root: &Path,
    state_root: &Path,
    request: &hi_tui::RaceRunRequest,
    event_sink: Option<Arc<dyn hi_events::EventSink>>,
    approval_store: Option<Arc<dyn hi_policy::ApprovalStore>>,
) -> Result<RaceSnapshot> {
    let artifact_root = request
        .artifact_root
        .as_deref()
        .context("reviewed race has no artifact root")?;
    let report_path = artifact_root.join("race.report.json");
    let report_text = std::fs::read_to_string(&report_path)
        .with_context(|| format!("reading reviewed race report {}", report_path.display()))?;
    let report: Value =
        serde_json::from_str(&report_text).context("parsing reviewed race report")?;
    let candidate_id = request
        .selected_candidate
        .as_deref()
        .context("reviewed race has no selected candidate")?;
    let index = candidate_id
        .strip_prefix("candidate-")
        .context("reviewed race selected candidate is malformed")?
        .parse::<usize>()
        .context("reviewed race selected candidate is not numeric")?;
    let candidate = report
        .get("candidates")
        .and_then(Value::as_array)
        .and_then(|candidates| {
            candidates.iter().find(|candidate| {
                candidate.get("index").and_then(Value::as_u64) == Some(index as u64)
            })
        })
        .context("reviewed race candidate is missing")?;
    ensure!(
        candidate.get("eligible").and_then(Value::as_bool) == Some(true),
        "only an eligible race candidate can be applied"
    );
    let patch_path = candidate
        .get("patch_path")
        .and_then(Value::as_str)
        .context("reviewed race candidate has no patch artifact")?;
    let patch = std::fs::read(patch_path)
        .with_context(|| format!("reading reviewed race patch {patch_path}"))?;
    let verify = resolved_verify(&request.verify_commands)?;
    let changed_paths = string_array(candidate.get("actual_changes"));
    consume_race_apply_approval(
        approval_store.as_deref(),
        event_sink.as_deref(),
        request.source_run_id.as_deref(),
        request.expected_workspace_digest.as_deref(),
        workspace_root,
        candidate_id,
        &changed_paths,
        &patch,
    )?;
    let workspace_root = workspace_root.to_path_buf();
    let state_root = state_root.to_path_buf();
    let candidate_id = candidate_id.to_string();
    let applied = tokio::task::spawn_blocking(move || -> Result<()> {
        let repository = crate::candidate_gate::repository_root(&workspace_root)?.canonicalize()?;
        let base = crate::bestof::resolve_revision(&repository, "HEAD")?;
        let worktree = hi_tools::worktree::worktree_path("race-apply", index as u32);
        hi_tools::worktree::add_worktree(&repository, &worktree, &base)?;
        let snapshot = hi_race::capture_workspace_snapshot(&workspace_root)?;
        let workspace_relative = workspace_root.strip_prefix(&repository)?.to_path_buf();
        let candidate_root = worktree.join(&workspace_relative);
        snapshot.materialize_into(&candidate_root)?;
        let candidate_base = crate::bestof::materialize_snapshot_base(&worktree, &base, &snapshot)?;
        let mut child = Command::new("git")
            .current_dir(&candidate_root)
            .args(["apply", "--whitespace=nowarn"])
            .stdin(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .context("starting reviewed race patch application")?;
        child
            .stdin
            .take()
            .context("opening reviewed race patch input")?
            .write_all(&patch)?;
        let output = child.wait_with_output()?;
        if !output.status.success() {
            hi_tools::worktree::cleanup(&repository, &[worktree]);
            bail!(
                "reviewed race patch conflicts with the current workspace: {}",
                String::from_utf8_lossy(&output.stderr).trim()
            );
        }
        let result = crate::candidate_merge::apply_candidate_and_reverify(
            &candidate_root,
            &candidate_base,
            &workspace_root,
            &state_root,
            &verify,
        );
        hi_tools::worktree::cleanup(&repository, &[worktree]);
        result.map(|_| ())
    })
    .await
    .context("reviewed race apply worker failed")?;
    applied?;
    let mut snapshot = snapshot_from_report(
        &report,
        &RaceSpec::new(&request.task, request.targets.clone()),
        &request.targets,
        report_path,
    );
    if let Some(run_id) = request.source_run_id.as_deref() {
        snapshot.run_id = run_id.to_string();
    }
    if let Some(workspace_digest) = request.expected_workspace_digest.as_deref() {
        snapshot.workspace_digest = workspace_digest.to_string();
    }
    snapshot.status = RaceStatus::Applied;
    snapshot.selected_candidate = Some(candidate_id);
    publish_race_event(
        event_sink.as_deref(),
        EventKind::RaceApplied,
        snapshot.run_id.as_str(),
        "coding race winner applied and destination verified",
        ActivityState::Succeeded,
    );
    Ok(snapshot)
}

#[allow(clippy::too_many_arguments)] // race apply approval passes each gate/session handle explicitly
fn consume_race_apply_approval(
    store: Option<&dyn hi_policy::ApprovalStore>,
    sink: Option<&dyn hi_events::EventSink>,
    run_id: Option<&str>,
    workspace_digest: Option<&str>,
    workspace_root: &Path,
    candidate_id: &str,
    changed_paths: &[String],
    patch: &[u8],
) -> Result<ApprovalId> {
    let store = store.context(
        "race apply is unavailable because the local approval store could not be opened",
    )?;
    let workspace_id = workspace_root.display().to_string();
    let patch_digest = blake3::hash(patch).to_hex().to_string();
    let scope = ResourceScope::Paths {
        workspace_id: workspace_id.clone(),
        paths: changed_paths.to_vec(),
    };
    let operation_digest = OperationDigest::calculate(
        &CapabilityKind::WorkspaceWrite,
        "race_apply",
        &json!({
            "candidate_id": candidate_id,
            "run_id": run_id,
            "workspace_digest": workspace_digest,
            "patch_digest": patch_digest,
        }),
        &workspace_id,
        &scope,
        Some(&patch_digest),
    );
    let now = hi_policy::now_ms();
    let request = CapabilityRequest {
        approval_id: ApprovalId(uuid::Uuid::new_v4().to_string()),
        capability: CapabilityKind::WorkspaceWrite,
        scope,
        operation_digest: operation_digest.clone(),
        tool: "race_apply".into(),
        run_id: run_id.map(str::to_string),
        session_id: None,
        title: format!("Apply reviewed race candidate {candidate_id}"),
        redacted_detail: format!(
            "{} changed file(s); prepared patch digest {}",
            changed_paths.len(),
            &patch_digest[..12]
        ),
        created_at_ms: now,
        expires_at_ms: now.saturating_add(24 * 60 * 60 * 1000),
    };
    let record = store.create(request)?;
    publish_approval_event(
        sink,
        EventKind::CapabilityRequested,
        run_id,
        &record.request.approval_id,
        ActivityState::Waiting,
        "race apply approval requested",
    );
    store.decide(&record.request.approval_id, ApprovalDecision::Approved)?;
    publish_approval_event(
        sink,
        EventKind::ApprovalDecided,
        run_id,
        &record.request.approval_id,
        ActivityState::Succeeded,
        "race apply approval decided locally",
    );
    store.claim(&record.request.approval_id, &operation_digest)?;
    publish_approval_event(
        sink,
        EventKind::ApprovalConsumed,
        run_id,
        &record.request.approval_id,
        ActivityState::Succeeded,
        "race apply approval consumed",
    );
    Ok(record.request.approval_id)
}

fn publish_approval_event(
    sink: Option<&dyn hi_events::EventSink>,
    kind: EventKind,
    run_id: Option<&str>,
    approval_id: &ApprovalId,
    state: ActivityState,
    title: &str,
) {
    let Some(sink) = sink else { return };
    let _ = publish_best_effort(
        Some(sink),
        RunEvent::new(
            kind,
            EventContext {
                run_id: run_id.map(str::to_string),
                ..EventContext::default()
            },
            SemanticActivity {
                verb: match state {
                    ActivityState::Waiting => ActivityVerb::Wait,
                    ActivityState::Succeeded => ActivityVerb::Approve,
                    ActivityState::Failed => ActivityVerb::Fail,
                    _ => ActivityVerb::Request,
                },
                object: ActivityObject::Approval,
                state,
                group_key: format!("approval:{}", approval_id.0),
                title: title.into(),
                detail: None,
                refs: vec![],
                progress: None,
            },
        ),
    );
}

fn snapshot_from_report(
    report: &Value,
    spec: &RaceSpec,
    targets: &[hi_race::RaceTarget],
    report_path: PathBuf,
) -> RaceSnapshot {
    let status = match report.get("status").and_then(Value::as_str) {
        Some("completed") => RaceStatus::Applied,
        Some("ready") => RaceStatus::Ready,
        Some("no_winner") => RaceStatus::NoWinner,
        Some("application_failed") => RaceStatus::Failed,
        Some("workspace_conflict") => RaceStatus::Failed,
        _ => RaceStatus::Failed,
    };
    let selected_index = report.get("selected_candidate").and_then(Value::as_u64);
    let candidates = report
        .get("candidates")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default()
        .into_iter()
        .map(|candidate| {
            let index = candidate
                .get("index")
                .and_then(Value::as_u64)
                .unwrap_or_default() as usize;
            let target = targets
                .get(index)
                .cloned()
                .unwrap_or_else(|| hi_race::RaceTarget {
                    name: format!("candidate-{index}"),
                    profile: String::new(),
                    model: String::new(),
                    priority: index as u32,
                });
            let eligible = candidate
                .get("eligible")
                .and_then(Value::as_bool)
                .unwrap_or(false);
            CandidateReport {
                candidate_id: format!("candidate-{index}"),
                target,
                state: if eligible {
                    CandidateState::Passed
                } else {
                    CandidateState::Failed
                },
                process_succeeded: candidate
                    .get("process_succeeded")
                    .and_then(Value::as_bool)
                    .unwrap_or(false),
                report_matches_diff: candidate
                    .get("report_matches_diff")
                    .and_then(Value::as_bool)
                    .unwrap_or(false),
                actual_changes: string_array(candidate.get("actual_changes")),
                changed_lines: candidate
                    .get("changed_lines")
                    .and_then(Value::as_u64)
                    .unwrap_or_default(),
                verify: candidate
                    .get("verify")
                    .cloned()
                    .and_then(|value| serde_json::from_value::<Vec<StageResult>>(value).ok())
                    .unwrap_or_default(),
                fuzz: candidate
                    .get("fuzz")
                    .cloned()
                    .and_then(|value| serde_json::from_value::<StageResult>(value).ok()),
                wall_clock_ms: candidate
                    .get("wall_clock_ms")
                    .and_then(Value::as_u64)
                    .unwrap_or_default() as u128,
                cost_microusd: None,
                artifact_ref: candidate
                    .get("patch_path")
                    .and_then(Value::as_str)
                    .map(str::to_string),
                failure_reason: (!eligible).then(|| {
                    candidate
                        .get("parent_verification")
                        .and_then(Value::as_str)
                        .unwrap_or("candidate failed")
                        .to_string()
                }),
            }
        })
        .collect::<Vec<_>>();
    let selected_candidate = selected_index.map(|index| format!("candidate-{index}"));
    RaceSnapshot {
        schema_version: hi_race::SCHEMA_VERSION,
        run_id: spec.run_id.clone(),
        status,
        workspace_digest: spec.workspace_digest.clone(),
        candidates,
        selected_candidate,
        artifact_root: report_path.parent().map(PathBuf::from),
        error: None,
    }
}

fn string_array(value: Option<&Value>) -> Vec<String> {
    value
        .and_then(Value::as_array)
        .map(|values| {
            values
                .iter()
                .filter_map(Value::as_str)
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default()
}

fn publish_race_event(
    sink: Option<&dyn hi_events::EventSink>,
    kind: EventKind,
    run_id: &str,
    title: &str,
    state: ActivityState,
) {
    let Some(sink) = sink else { return };
    let event = RunEvent::new(
        kind,
        EventContext {
            run_id: Some(run_id.to_string()),
            correlation_id: Some(run_id.to_string()),
            ..EventContext::default()
        },
        SemanticActivity {
            verb: match state {
                ActivityState::Running => ActivityVerb::Start,
                ActivityState::Waiting => ActivityVerb::Wait,
                ActivityState::Succeeded => ActivityVerb::Complete,
                ActivityState::Failed => ActivityVerb::Fail,
                ActivityState::Denied => ActivityVerb::Deny,
                ActivityState::TimedOut => ActivityVerb::Fail,
                ActivityState::Cancelled => ActivityVerb::Cancel,
                ActivityState::Abandoned => ActivityVerb::Fail,
                ActivityState::Pending => ActivityVerb::Request,
            },
            object: ActivityObject::Race,
            state,
            group_key: format!("race:{run_id}"),
            title: title.to_string(),
            detail: None,
            refs: vec![],
            progress: None,
        },
    );
    let _ = publish_best_effort(Some(sink), event);
}
