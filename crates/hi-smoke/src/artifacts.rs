use std::collections::BTreeMap;
use std::fs;
use std::io::{ErrorKind, Write};
use std::path::{Component, Path, PathBuf};

use anyhow::{Context, Result, bail, ensure};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tempfile::NamedTempFile;

use crate::live_route::LiveRoute;
use crate::pty::RawTerminal;
use crate::scenario::Scenario;

pub(crate) const ARTIFACT_SCHEMA_VERSION: u16 = 1;
pub(crate) const RAW_TERMINAL_LIMIT_BYTES: usize = 4 * 1024 * 1024;

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum BundleStatus {
    Passed,
    Failed,
    InfrastructureFailure,
    InfrastructureLoop,
    Crashed,
    TimedOut,
    Cancelled,
}

impl BundleStatus {
    fn is_success(self) -> bool {
        self == Self::Passed
    }
}

#[derive(Debug)]
pub(crate) struct BundleInput<'a> {
    pub scenario: &'a Scenario,
    pub mode: &'a str,
    /// Non-secret live provider route. Credentials are never represented here.
    pub live_route: Option<&'a LiveRoute>,
    pub status: BundleStatus,
    pub seed: Option<u64>,
    pub duration_ms: u64,
    pub failure: Option<&'a str>,
    pub tui_events: &'a [Value],
    pub raw_terminal: &'a RawTerminal,
    pub screens: &'a BTreeMap<String, String>,
    pub provider_requests: &'a [Value],
    /// Exact sensitive values known by the caller, such as a live provider key.
    pub redaction_values: &'a [String],
    pub session_jsonl: &'a [u8],
    /// Pre-run workspace snapshot used to make `replay.toml` self-contained.
    pub initial_workspace_root: Option<&'a Path>,
    /// Final workspace used for post-run listing evidence.
    pub workspace_root: &'a Path,
    pub workspace_patch: &'a str,
    /// Content-addressed pre/post mutation evidence for isolation siblings.
    /// Paths are relative to the ephemeral root and file bodies are omitted.
    pub isolation_evidence: &'a Value,
    /// Includes exit status, descendant cleanup, and leak detection evidence.
    pub process: &'a Value,
    pub assertions: &'a Value,
    pub timings: &'a Value,
    pub result: &'a Value,
}

/// Inputs that remain available even when case setup did not get far enough to
/// construct a [`BundleInput`]. Used to create or repair the smallest complete
/// replay artifact after infrastructure/evidence failures.
#[derive(Debug)]
pub(crate) struct MinimalBundleInput<'a> {
    pub scenario: &'a Scenario,
    pub mode: &'a str,
    /// Non-secret live provider route, when setup resolved it before failing.
    pub live_route: Option<&'a LiveRoute>,
    pub seed: Option<u64>,
    pub duration_ms: u64,
    pub failure: &'a str,
    /// Prefer the initialized pre-run workspace. Setup failures may instead
    /// provide the scenario's source fixture, or `None` for an empty fixture.
    pub fixture_root: Option<&'a Path>,
    /// Exact sensitive values known by the caller, such as a live provider key.
    pub redaction_values: &'a [String],
    pub detailed_bundle_failure: Option<&'a str>,
}

/// One boundary for every user-visible artifact written by the smoke harness.
///
/// Provider credentials must never reach disk even when a model echoes its
/// environment through a tool. Ephemeral workspace/isolation paths are also
/// replaced so bundles remain stable and do not disclose host layout. Exact
/// replacements are byte-safe (and therefore work for the raw PTY capture and
/// binary replay fixtures); structured JSON additionally redacts sensitive
/// field names and common bearer/token spellings.
#[derive(Clone, Debug, Default)]
struct ArtifactSanitizer {
    replacements: Vec<(Vec<u8>, Vec<u8>)>,
}

impl ArtifactSanitizer {
    fn for_bundle(artifact_root: &Path, case_directory: &Path, input: &BundleInput<'_>) -> Self {
        let mut sanitizer = Self::with_secrets(input.redaction_values);
        sanitizer.add_path(case_directory, "<ARTIFACT_CASE>");
        sanitizer.add_path(artifact_root, "<ARTIFACT_ROOT>");
        sanitizer.add_path(input.workspace_root, "<WORKSPACE>");
        if let Some(initial) = input.initial_workspace_root {
            sanitizer.add_path(initial, "<INITIAL_WORKSPACE>");
        }
        if let Some(isolation) = input.workspace_root.parent() {
            sanitizer.add_path(isolation, "<ISOLATION>");
        }
        sanitizer.finish()
    }

    fn for_minimal(
        artifact_root: &Path,
        case_directory: &Path,
        input: &MinimalBundleInput<'_>,
    ) -> Self {
        let mut sanitizer = Self::with_secrets(input.redaction_values);
        sanitizer.add_path(case_directory, "<ARTIFACT_CASE>");
        sanitizer.add_path(artifact_root, "<ARTIFACT_ROOT>");
        if let Some(fixture) = input.fixture_root {
            sanitizer.add_path(fixture, "<REPLAY_FIXTURE>");
        }
        sanitizer.finish()
    }

    fn with_secrets(secrets: &[String]) -> Self {
        let mut sanitizer = Self::default();
        for secret in secrets.iter().filter(|secret| !secret.is_empty()) {
            sanitizer.add_replacement(secret.as_bytes(), b"[REDACTED]");
        }
        sanitizer
    }

    fn add_path(&mut self, path: &Path, replacement: &str) {
        if path.as_os_str().is_empty() {
            return;
        }
        self.add_replacement(path.to_string_lossy().as_bytes(), replacement.as_bytes());
        if let Ok(canonical) = path.canonicalize() {
            self.add_replacement(
                canonical.to_string_lossy().as_bytes(),
                replacement.as_bytes(),
            );
        }
    }

    fn add_replacement(&mut self, needle: &[u8], replacement: &[u8]) {
        if !needle.is_empty() && needle != replacement {
            self.replacements
                .push((needle.to_vec(), replacement.to_vec()));
        }
    }

    fn finish(mut self) -> Self {
        // Longest first prevents an isolation-root replacement from hiding a
        // more precise workspace or case-directory replacement.
        self.replacements
            .sort_by(|left, right| right.0.len().cmp(&left.0.len()).then(left.0.cmp(&right.0)));
        self.replacements.dedup_by(|left, right| left.0 == right.0);
        self
    }

    fn sanitize_bytes(&self, bytes: &[u8]) -> Vec<u8> {
        let mut sanitized = bytes.to_vec();
        for (needle, replacement) in &self.replacements {
            sanitized = replace_bytes(&sanitized, needle, replacement);
        }
        if let Ok(text) = std::str::from_utf8(&sanitized) {
            redact_marker_values(
                text,
                &["bearer ", "api_key=", "api-key=", "access_token=", "token="],
            )
            .into_bytes()
        } else {
            sanitized
        }
    }

    fn sanitize_string(&self, value: &str) -> String {
        // Valid UTF-8 plus valid UTF-8 replacements remains valid UTF-8.
        String::from_utf8(self.sanitize_bytes(value.as_bytes()))
            .expect("artifact string sanitization preserves UTF-8")
    }

    fn sanitize_value(&self, value: &Value) -> Value {
        match value {
            Value::Object(object) => {
                let mut sorted = BTreeMap::new();
                for (key, value) in object {
                    let key = self.sanitize_string(key);
                    let value = if sensitive_key(&key) {
                        Value::String("[REDACTED]".to_owned())
                    } else {
                        self.sanitize_value(value)
                    };
                    sorted.insert(key, value);
                }
                Value::Object(sorted.into_iter().collect())
            }
            Value::Array(values) => Value::Array(
                values
                    .iter()
                    .map(|value| self.sanitize_value(value))
                    .collect(),
            ),
            Value::String(value) => Value::String(self.sanitize_string(value)),
            Value::Null | Value::Bool(_) | Value::Number(_) => value.clone(),
        }
    }
}

fn replace_bytes(haystack: &[u8], needle: &[u8], replacement: &[u8]) -> Vec<u8> {
    if needle.is_empty()
        || !haystack
            .windows(needle.len())
            .any(|window| window == needle)
    {
        return haystack.to_vec();
    }
    let mut output = Vec::with_capacity(haystack.len());
    let mut cursor = 0;
    while cursor < haystack.len() {
        if haystack[cursor..].starts_with(needle) {
            output.extend_from_slice(replacement);
            cursor += needle.len();
        } else {
            output.push(haystack[cursor]);
            cursor += 1;
        }
    }
    output
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct BundlePaths {
    pub directory: PathBuf,
    pub summary: PathBuf,
    pub replay: Option<PathBuf>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub(crate) struct WorkspaceEntry {
    pub path: String,
    pub kind: WorkspaceEntryKind,
    pub bytes: Option<u64>,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum WorkspaceEntryKind {
    Directory,
    File,
    Symlink,
    Other,
}

#[derive(Serialize)]
struct CaseSummary<'a> {
    schema_version: u16,
    generator: &'static str,
    generator_version: &'static str,
    scenario: &'a str,
    mode: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    live_route: Option<&'a LiveRoute>,
    status: BundleStatus,
    seed: Option<u64>,
    duration_ms: u64,
    failure: Option<&'a str>,
    detailed_evidence: bool,
    provider_request_count: usize,
    provider_chat_request_count: usize,
    provider_accepted_request_count: usize,
    provider_response_status_counts: BTreeMap<u16, usize>,
    unexpected_isolation_mutation_count: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    detailed_bundle_failure: Option<&'a str>,
}

#[derive(Serialize)]
struct RawTerminalMetadata {
    schema_version: u16,
    limit_bytes: usize,
    /// Bytes observed before sanitization or the reader-side evidence cap.
    original_bytes: u64,
    /// Sanitized bytes available before the bundle-side cap/truncation marker.
    sanitized_captured_bytes: usize,
    written_bytes: usize,
    truncated: bool,
    /// Hash of the sanitized pre-cap capture; never a fingerprint of secrets.
    captured_blake3: String,
}

fn accepted_provider_request_count(requests: &[Value]) -> usize {
    requests
        .iter()
        .filter(|request| request.get("accepted").and_then(Value::as_bool) == Some(true))
        .count()
}

/// Count model-chat calls consistently across scripted and live evidence.
///
/// Scripted server records include the independently routed `/models` request,
/// while live evidence contains only sanitized `WireAudit` records. Keep the
/// legacy all-route count above for compatibility, but expose this deterministic
/// count for retry-budget assertions and cross-mode summaries.
fn provider_chat_request_count(requests: &[Value]) -> usize {
    requests
        .iter()
        .filter(
            |request| match request.get("path").and_then(Value::as_str) {
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
            },
        )
        .count()
}

fn provider_response_status_counts(requests: &[Value]) -> BTreeMap<u16, usize> {
    let mut counts = BTreeMap::new();
    for status in requests
        .iter()
        .filter_map(|request| request.get("response_status").and_then(Value::as_u64))
        .filter_map(|status| u16::try_from(status).ok())
    {
        *counts.entry(status).or_default() += 1;
    }
    counts
}

/// Write one scenario's evidence beneath a caller-selected, contained relative
/// directory. Passing scenarios always get `summary.json`; non-passing scenarios
/// additionally get the complete replay bundle.
pub(crate) fn write_case_bundle(
    artifact_root: &Path,
    relative_case_dir: &Path,
    input: &BundleInput<'_>,
) -> Result<BundlePaths> {
    match write_case_bundle_detailed(artifact_root, relative_case_dir, input) {
        Ok(paths) => Ok(paths),
        Err(detailed_error) => {
            let detailed_failure = format!("{detailed_error:#}");
            let failure = input.failure.unwrap_or("artifact bundle creation failed");
            let repaired = repair_minimal_failure_bundle(
                artifact_root,
                relative_case_dir,
                &MinimalBundleInput {
                    scenario: input.scenario,
                    mode: input.mode,
                    live_route: input.live_route,
                    seed: input.seed,
                    duration_ms: input.duration_ms,
                    failure,
                    fixture_root: input.initial_workspace_root.or(Some(input.workspace_root)),
                    redaction_values: input.redaction_values,
                    detailed_bundle_failure: Some(&detailed_failure),
                },
            );
            match (input.status.is_success(), repaired) {
                // A failed scenario already has the status the caller needs;
                // a repaired exact replay is sufficient even if rich evidence
                // could not be completed.
                (false, Ok(paths)) => Ok(paths),
                // A passing case whose artifact write failed is an
                // infrastructure failure and must fail the run, but still has
                // a replay bundle explaining the failure.
                (true, Ok(_)) => Err(detailed_error)
                    .context("case passed but its artifact bundle could not be written"),
                (_, Err(repair_error)) => Err(anyhow::anyhow!(
                    "detailed bundle failed: {detailed_error:#}; minimal replay repair failed: {repair_error:#}"
                )),
            }
        }
    }
}

fn write_case_bundle_detailed(
    artifact_root: &Path,
    relative_case_dir: &Path,
    input: &BundleInput<'_>,
) -> Result<BundlePaths> {
    let directory = create_contained_dir(artifact_root, relative_case_dir)?;
    let sanitizer = ArtifactSanitizer::for_bundle(artifact_root, &directory, input);
    let summary = directory.join("summary.json");
    let case_summary = CaseSummary {
        schema_version: ARTIFACT_SCHEMA_VERSION,
        generator: "hi-smoke",
        generator_version: env!("CARGO_PKG_VERSION"),
        scenario: &input.scenario.name,
        mode: input.mode,
        live_route: input.live_route,
        status: input.status,
        seed: input.seed,
        duration_ms: input.duration_ms,
        failure: input.failure,
        detailed_evidence: !input.status.is_success(),
        provider_request_count: input.provider_requests.len(),
        provider_chat_request_count: provider_chat_request_count(input.provider_requests),
        provider_accepted_request_count: accepted_provider_request_count(input.provider_requests),
        provider_response_status_counts: provider_response_status_counts(input.provider_requests),
        unexpected_isolation_mutation_count: input
            .isolation_evidence
            .get("unexpected_mutation_count")
            .and_then(Value::as_u64)
            .unwrap_or(0),
        detailed_bundle_failure: None,
    };
    write_sanitized_json(&summary, &case_summary, &sanitizer)?;

    if input.status.is_success() {
        return Ok(BundlePaths {
            directory,
            summary,
            replay: None,
        });
    }

    let replay_fixture = directory.join("replay-fixture");
    copy_workspace_snapshot_sanitized(
        input.initial_workspace_root.unwrap_or(input.workspace_root),
        &replay_fixture,
        &sanitizer,
    )?;
    let replay = directory.join("replay.toml");
    write_replay(
        &replay,
        input.scenario,
        input.mode,
        input.live_route,
        input.seed,
        Some("replay-fixture"),
        &sanitizer,
    )?;
    write_jsonl(
        &directory.join("tui-events.jsonl"),
        input.tui_events,
        &sanitizer,
    )?;
    write_raw_terminal(&directory, input.raw_terminal, &sanitizer)?;
    write_screens(&directory, input.screens, &sanitizer)?;
    write_jsonl(
        &directory.join("provider-requests.jsonl"),
        input.provider_requests,
        &sanitizer,
    )?;
    atomic_write(
        &directory.join("session.jsonl"),
        &sanitizer.sanitize_bytes(input.session_jsonl),
    )?;

    let listing = capture_workspace_listing(input.workspace_root)?;
    write_sanitized_json(
        &directory.join("workspace-listing.json"),
        &listing,
        &sanitizer,
    )?;
    atomic_write(
        &directory.join("workspace.patch"),
        &sanitizer.sanitize_bytes(input.workspace_patch.as_bytes()),
    )?;
    write_sanitized_json(
        &directory.join("isolation-evidence.json"),
        input.isolation_evidence,
        &sanitizer,
    )?;
    write_sanitized_json(&directory.join("process.json"), input.process, &sanitizer)?;
    write_sanitized_json(
        &directory.join("assertions.json"),
        input.assertions,
        &sanitizer,
    )?;
    write_sanitized_json(&directory.join("timings.json"), input.timings, &sanitizer)?;
    write_sanitized_json(&directory.join("result.json"), input.result, &sanitizer)?;

    Ok(BundlePaths {
        directory,
        summary,
        replay: Some(replay),
    })
}

/// Create or repair a minimal, self-contained failure bundle. The function is
/// deliberately idempotent: if a detailed write left a partial fixture, a new
/// contained repair fixture is selected and `replay.toml` is atomically
/// redirected to it.
pub(crate) fn repair_minimal_failure_bundle(
    artifact_root: &Path,
    relative_case_dir: &Path,
    input: &MinimalBundleInput<'_>,
) -> Result<BundlePaths> {
    let directory = create_contained_dir(artifact_root, relative_case_dir)?;
    let sanitizer = ArtifactSanitizer::for_minimal(artifact_root, &directory, input);
    let fixture_name = if let Some(source) = input.fixture_root {
        let name = available_repair_fixture_name(&directory);
        copy_workspace_snapshot_sanitized(source, &directory.join(&name), &sanitizer)?;
        Some(name)
    } else {
        None
    };
    let replay = directory.join("replay.toml");
    write_replay(
        &replay,
        input.scenario,
        input.mode,
        input.live_route,
        input.seed,
        fixture_name.as_deref(),
        &sanitizer,
    )?;
    let summary = directory.join("summary.json");
    write_sanitized_json(
        &summary,
        &CaseSummary {
            schema_version: ARTIFACT_SCHEMA_VERSION,
            generator: "hi-smoke",
            generator_version: env!("CARGO_PKG_VERSION"),
            scenario: &input.scenario.name,
            mode: input.mode,
            live_route: input.live_route,
            status: BundleStatus::InfrastructureFailure,
            seed: input.seed,
            duration_ms: input.duration_ms,
            failure: Some(input.failure),
            detailed_evidence: false,
            provider_request_count: 0,
            provider_chat_request_count: 0,
            provider_accepted_request_count: 0,
            provider_response_status_counts: BTreeMap::new(),
            unexpected_isolation_mutation_count: 0,
            detailed_bundle_failure: input.detailed_bundle_failure,
        },
        &sanitizer,
    )?;
    if let Some(failure) = input.detailed_bundle_failure {
        atomic_write(
            &directory.join("bundle-write-failure.txt"),
            sanitizer
                .sanitize_string(&single_line_comment(failure))
                .as_bytes(),
        )?;
    }
    Ok(BundlePaths {
        directory,
        summary,
        replay: Some(replay),
    })
}

fn available_repair_fixture_name(directory: &Path) -> String {
    for index in 0u32.. {
        let name = if index == 0 {
            "replay-fixture".to_owned()
        } else {
            format!("replay-fixture-repair-{index}")
        };
        if !directory.join(&name).exists() {
            return name;
        }
    }
    unreachable!("u32 fixture repair namespace exhausted")
}

/// Write the aggregate run summary consumed by CI. This is intentionally
/// independent of case bundles so even an all-passing run uploads a summary.
pub(crate) fn write_suite_summary(
    artifact_root: &Path,
    summary: &Value,
    redaction_values: &[String],
) -> Result<PathBuf> {
    ensure_root_directory(artifact_root)?;
    let path = artifact_root.join("summary.json");
    let mut sanitizer = ArtifactSanitizer::with_secrets(redaction_values);
    sanitizer.add_path(artifact_root, "<ARTIFACT_ROOT>");
    let sanitizer = sanitizer.finish();
    write_sanitized_json(&path, summary, &sanitizer)?;
    Ok(path)
}

/// Capture a deterministic listing without following symlinks.
pub(crate) fn capture_workspace_listing(root: &Path) -> Result<Vec<WorkspaceEntry>> {
    let metadata = fs::symlink_metadata(root)
        .with_context(|| format!("reading workspace metadata {}", root.display()))?;
    ensure!(
        metadata.is_dir(),
        "workspace is not a directory: {}",
        root.display()
    );
    let mut entries = Vec::new();
    collect_workspace_entries(root, root, &mut entries)?;
    entries.sort_by(|left, right| left.path.cmp(&right.path));
    Ok(entries)
}

fn collect_workspace_entries(
    root: &Path,
    directory: &Path,
    entries: &mut Vec<WorkspaceEntry>,
) -> Result<()> {
    let mut children = fs::read_dir(directory)
        .with_context(|| format!("reading workspace directory {}", directory.display()))?
        .collect::<std::result::Result<Vec<_>, _>>()?;
    children.sort_by_key(fs::DirEntry::file_name);

    for child in children {
        let path = child.path();
        let relative = path
            .strip_prefix(root)
            .with_context(|| format!("containing workspace path {}", path.display()))?;
        validate_relative_path(relative)?;
        let metadata = fs::symlink_metadata(&path)
            .with_context(|| format!("reading workspace entry {}", path.display()))?;
        let file_type = metadata.file_type();
        let (kind, bytes) = if file_type.is_dir() {
            (WorkspaceEntryKind::Directory, None)
        } else if file_type.is_file() {
            (WorkspaceEntryKind::File, Some(metadata.len()))
        } else if file_type.is_symlink() {
            (WorkspaceEntryKind::Symlink, None)
        } else {
            (WorkspaceEntryKind::Other, None)
        };
        entries.push(WorkspaceEntry {
            path: portable_relative_path(relative),
            kind,
            bytes,
        });
        if file_type.is_dir() {
            collect_workspace_entries(root, &path, entries)?;
        }
    }
    Ok(())
}

fn portable_relative_path(path: &Path) -> String {
    path.components()
        .filter_map(|component| match component {
            Component::Normal(segment) => Some(segment.to_string_lossy()),
            Component::CurDir => None,
            Component::ParentDir | Component::RootDir | Component::Prefix(_) => None,
        })
        .collect::<Vec<_>>()
        .join("/")
}

fn write_replay(
    path: &Path,
    scenario: &Scenario,
    mode: &str,
    live_route: Option<&LiveRoute>,
    seed: Option<u64>,
    fixture: Option<&str>,
    sanitizer: &ArtifactSanitizer,
) -> Result<()> {
    ensure!(
        live_route.is_none() || mode == "live",
        "live replay route is valid only in live mode"
    );
    let mut replay_scenario = scenario.clone();
    if let Some(fixture) = fixture {
        replay_scenario.workspace.fixture = Some(fixture.to_owned());
    }
    let scenario = sanitizer.sanitize_string(
        &toml::to_string_pretty(&replay_scenario)
            .context("serializing normalized replay scenario")?,
    );
    let mode = sanitizer.sanitize_string(&single_line_comment(mode));
    let seed = seed.map_or_else(|| "none".to_owned(), |seed| seed.to_string());
    let route = live_route.map_or_else(String::new, |route| {
        format!(
            "# hi-smoke-live-provider = {}\n# hi-smoke-live-model = {}\n# hi-smoke-live-base-url = {}\n",
            sanitizer.sanitize_string(&single_line_comment(&route.provider)),
            sanitizer.sanitize_string(&single_line_comment(&route.model)),
            sanitizer.sanitize_string(&single_line_comment(&route.base_url)),
        )
    });
    let text = format!(
        "# Generated by hi-smoke {}\n# hi-smoke-replay-mode = {mode}\n{route}# seed = {seed}\n{scenario}",
        env!("CARGO_PKG_VERSION")
    );
    atomic_write(path, text.as_bytes())
}

fn copy_workspace_snapshot_sanitized(
    source: &Path,
    destination: &Path,
    sanitizer: &ArtifactSanitizer,
) -> Result<()> {
    let metadata = fs::symlink_metadata(source)
        .with_context(|| format!("reading replay fixture {}", source.display()))?;
    ensure!(
        metadata.is_dir(),
        "replay fixture is not a directory: {}",
        source.display()
    );
    ensure!(
        fs::symlink_metadata(destination).is_err_and(|error| error.kind() == ErrorKind::NotFound),
        "replay fixture destination already exists: {}",
        destination.display()
    );
    fs::create_dir(destination)
        .with_context(|| format!("creating replay fixture {}", destination.display()))?;
    copy_snapshot_directory(source, source, destination, sanitizer)?;
    fs::set_permissions(destination, metadata.permissions()).with_context(|| {
        format!(
            "preserving replay fixture permissions {}",
            destination.display()
        )
    })?;
    Ok(())
}

fn copy_snapshot_directory(
    source_root: &Path,
    source: &Path,
    destination: &Path,
    sanitizer: &ArtifactSanitizer,
) -> Result<()> {
    let mut entries = fs::read_dir(source)
        .with_context(|| format!("reading replay fixture directory {}", source.display()))?
        .collect::<std::result::Result<Vec<_>, _>>()?;
    entries.sort_by_key(fs::DirEntry::file_name);
    for entry in entries {
        let source_path = entry.path();
        let relative = source_path
            .strip_prefix(source_root)
            .with_context(|| format!("containing replay fixture path {}", source_path.display()))?;
        validate_relative_path(relative)?;
        let sanitized_name = sanitizer.sanitize_string(&entry.file_name().to_string_lossy());
        ensure!(
            !sanitized_name.is_empty()
                && sanitized_name != "."
                && sanitized_name != ".."
                && !sanitized_name.contains(['/', '\\']),
            "sanitized replay fixture entry has an unsafe name"
        );
        let destination_path = destination.join(sanitized_name);
        let metadata = fs::symlink_metadata(&source_path)
            .with_context(|| format!("reading replay fixture entry {}", source_path.display()))?;
        let file_type = metadata.file_type();
        if file_type.is_dir() {
            fs::create_dir(&destination_path).with_context(|| {
                format!(
                    "creating replay fixture directory {}",
                    destination_path.display()
                )
            })?;
            copy_snapshot_directory(source_root, &source_path, &destination_path, sanitizer)?;
            fs::set_permissions(&destination_path, metadata.permissions()).with_context(|| {
                format!(
                    "preserving replay fixture permissions {}",
                    destination_path.display()
                )
            })?;
        } else if file_type.is_file() {
            let contents = fs::read(&source_path).with_context(|| {
                format!("reading replay fixture file {}", source_path.display())
            })?;
            atomic_write(&destination_path, &sanitizer.sanitize_bytes(&contents)).with_context(
                || {
                    format!(
                        "copying sanitized replay fixture file {} to {}",
                        source_path.display(),
                        destination_path.display()
                    )
                },
            )?;
            fs::set_permissions(&destination_path, metadata.permissions()).with_context(|| {
                format!(
                    "preserving replay fixture permissions {}",
                    destination_path.display()
                )
            })?;
        } else if file_type.is_symlink() {
            copy_snapshot_symlink(
                source_root,
                &source_path,
                relative,
                &destination_path,
                sanitizer,
            )?;
        } else {
            bail!(
                "unsupported special file in replay fixture: {}",
                source_path.display()
            );
        }
    }
    Ok(())
}

fn copy_snapshot_symlink(
    source_root: &Path,
    source_path: &Path,
    relative: &Path,
    destination_path: &Path,
    sanitizer: &ArtifactSanitizer,
) -> Result<()> {
    let target = fs::read_link(source_path)
        .with_context(|| format!("reading replay fixture symlink {}", source_path.display()))?;
    ensure!(
        !target.is_absolute(),
        "absolute symlink is not replay-safe: {} -> {}",
        source_path.display(),
        target.display()
    );
    ensure_link_target_contained(source_root, relative, &target)?;
    let sanitized_target = PathBuf::from(sanitizer.sanitize_string(&target.to_string_lossy()));
    ensure!(
        !sanitized_target.is_absolute(),
        "sanitized replay symlink target must remain relative"
    );
    ensure_link_target_contained(source_root, relative, &sanitized_target)?;
    #[cfg(unix)]
    {
        std::os::unix::fs::symlink(&sanitized_target, destination_path).with_context(|| {
            format!(
                "copying replay fixture symlink {} to {}",
                source_path.display(),
                destination_path.display()
            )
        })?;
        Ok(())
    }
    #[cfg(not(unix))]
    {
        let _ = destination_path;
        bail!("symlink replay fixtures are supported only on Unix")
    }
}

fn ensure_link_target_contained(
    source_root: &Path,
    link_relative: &Path,
    target: &Path,
) -> Result<()> {
    let mut depth = link_relative.parent().map_or(0, |parent| {
        parent
            .components()
            .filter(|part| matches!(part, Component::Normal(_)))
            .count()
    });
    for component in target.components() {
        match component {
            Component::CurDir => {}
            Component::Normal(_) => depth += 1,
            Component::ParentDir => {
                ensure!(
                    depth > 0,
                    "symlink escapes replay fixture {}: {} -> {}",
                    source_root.display(),
                    link_relative.display(),
                    target.display()
                );
                depth -= 1;
            }
            Component::RootDir | Component::Prefix(_) => {
                bail!(
                    "symlink escapes replay fixture {}: {} -> {}",
                    source_root.display(),
                    link_relative.display(),
                    target.display()
                );
            }
        }
    }
    Ok(())
}

fn single_line_comment(value: &str) -> String {
    value.replace(['\r', '\n'], " ")
}

fn write_raw_terminal(
    directory: &Path,
    raw: &RawTerminal,
    sanitizer: &ArtifactSanitizer,
) -> Result<()> {
    let original_bytes = raw.total_bytes.max(raw.bytes.len() as u64);
    let capture_truncated = raw.truncated || raw.total_bytes > raw.bytes.len() as u64;
    let sanitized_capture = sanitizer.sanitize_bytes(&raw.bytes);
    let (bytes, truncated) =
        bounded_terminal(&sanitized_capture, capture_truncated, original_bytes);
    atomic_write(&directory.join("raw-terminal.bin"), &bytes)?;
    let metadata = RawTerminalMetadata {
        schema_version: ARTIFACT_SCHEMA_VERSION,
        limit_bytes: RAW_TERMINAL_LIMIT_BYTES,
        original_bytes,
        sanitized_captured_bytes: sanitized_capture.len(),
        written_bytes: bytes.len(),
        truncated,
        captured_blake3: blake3::hash(&sanitized_capture).to_hex().to_string(),
    };
    write_sanitized_json(
        &directory.join("raw-terminal.meta.json"),
        &metadata,
        sanitizer,
    )
}

fn bounded_terminal(
    sanitized_capture: &[u8],
    capture_truncated: bool,
    original_bytes: u64,
) -> (Vec<u8>, bool) {
    let truncated = capture_truncated || sanitized_capture.len() > RAW_TERMINAL_LIMIT_BYTES;
    if !truncated {
        return (sanitized_capture.to_vec(), false);
    }
    let marker = format!(
        "\n[hi-smoke: raw terminal truncated; original_bytes={}; limit_bytes={}]\n",
        original_bytes, RAW_TERMINAL_LIMIT_BYTES
    );
    let marker = marker.as_bytes();
    let retained = RAW_TERMINAL_LIMIT_BYTES.saturating_sub(marker.len());
    let mut bounded = Vec::with_capacity(RAW_TERMINAL_LIMIT_BYTES);
    bounded.extend_from_slice(&sanitized_capture[..sanitized_capture.len().min(retained)]);
    bounded.extend_from_slice(marker);
    (bounded, true)
}

fn write_screens(
    directory: &Path,
    screens: &BTreeMap<String, String>,
    sanitizer: &ArtifactSanitizer,
) -> Result<()> {
    let normalized = screens
        .iter()
        .map(|(name, screen)| {
            (
                sanitizer.sanitize_string(name),
                sanitizer.sanitize_string(&normalize_screen(screen)),
            )
        })
        .collect::<BTreeMap<_, _>>();
    write_sanitized_json(&directory.join("screens.json"), &normalized, sanitizer)?;

    let screen_dir = create_contained_dir(directory, Path::new("screens"))?;
    for (index, (name, screen)) in normalized.iter().enumerate() {
        let filename = format!("{index:03}-{}.txt", safe_stem(name));
        atomic_write(&screen_dir.join(filename), screen.as_bytes())?;
    }
    Ok(())
}

fn normalize_screen(screen: &str) -> String {
    let unified = screen.replace("\r\n", "\n").replace('\r', "\n");
    let mut lines = unified
        .lines()
        .map(|line| line.trim_end_matches([' ', '\t']))
        .collect::<Vec<_>>();
    while lines.last().is_some_and(|line| line.is_empty()) {
        lines.pop();
    }
    if lines.is_empty() {
        String::new()
    } else {
        let mut normalized = lines.join("\n");
        normalized.push('\n');
        normalized
    }
}

fn safe_stem(name: &str) -> String {
    let mut stem = String::with_capacity(name.len().min(64));
    let mut previous_dash = false;
    for character in name.chars().take(64) {
        if character.is_ascii_alphanumeric() || matches!(character, '-' | '_') {
            stem.push(character);
            previous_dash = false;
        } else if !previous_dash {
            stem.push('-');
            previous_dash = true;
        }
    }
    let stem = stem.trim_matches('-');
    if stem.is_empty() {
        "screen".to_owned()
    } else {
        stem.to_owned()
    }
}

fn write_jsonl(path: &Path, records: &[Value], sanitizer: &ArtifactSanitizer) -> Result<()> {
    let mut output = Vec::new();
    for record in records {
        let record = sanitizer.sanitize_value(record);
        serde_json::to_writer(&mut output, &record).context("serializing JSONL record")?;
        output.push(b'\n');
    }
    atomic_write(path, &output)
}

fn sensitive_key(key: &str) -> bool {
    let normalized = key.to_ascii_lowercase().replace('-', "_");
    matches!(
        normalized.as_str(),
        "authorization"
            | "proxy_authorization"
            | "api_key"
            | "x_api_key"
            | "access_token"
            | "refresh_token"
            | "id_token"
            | "token"
            | "secret"
            | "client_secret"
            | "password"
            | "cookie"
            | "set_cookie"
    ) || normalized.ends_with("_api_key")
        || normalized.ends_with("_secret")
        || normalized.ends_with("_password")
}

fn redact_marker_values(value: &str, markers: &[&str]) -> String {
    let mut output = String::with_capacity(value.len());
    let mut remaining = value;
    while !remaining.is_empty() {
        let lower = remaining.to_ascii_lowercase();
        let found = markers
            .iter()
            .filter_map(|marker| lower.find(marker).map(|index| (index, *marker)))
            .min_by_key(|(index, _)| *index);
        let Some((index, marker)) = found else {
            output.push_str(remaining);
            break;
        };
        let value_start = index + marker.len();
        output.push_str(&remaining[..value_start]);
        output.push_str("[REDACTED]");
        let tail = &remaining[value_start..];
        let value_end = tail
            .find(|character: char| {
                character.is_whitespace()
                    || matches!(character, '&' | '#' | '"' | '\'' | '\\' | ',' | '}')
            })
            .unwrap_or(tail.len());
        remaining = &tail[value_end..];
    }
    output
}

fn write_sanitized_json(
    path: &Path,
    value: &impl Serialize,
    sanitizer: &ArtifactSanitizer,
) -> Result<()> {
    let value = serde_json::to_value(value).context("converting artifact to JSON")?;
    let canonical = sanitizer.sanitize_value(&value);
    let mut bytes = serde_json::to_vec_pretty(&canonical).context("serializing artifact JSON")?;
    bytes.push(b'\n');
    atomic_write(path, &bytes)
}

fn atomic_write(path: &Path, bytes: &[u8]) -> Result<()> {
    let parent = path
        .parent()
        .with_context(|| format!("artifact path has no parent: {}", path.display()))?;
    let mut temporary = NamedTempFile::new_in(parent)
        .with_context(|| format!("creating temporary artifact in {}", parent.display()))?;
    temporary
        .write_all(bytes)
        .with_context(|| format!("writing temporary artifact for {}", path.display()))?;
    temporary
        .flush()
        .with_context(|| format!("flushing temporary artifact for {}", path.display()))?;
    temporary
        .persist(path)
        .map_err(|error| error.error)
        .with_context(|| format!("persisting artifact {}", path.display()))?;
    Ok(())
}

fn create_contained_dir(root: &Path, relative: &Path) -> Result<PathBuf> {
    validate_relative_path(relative)?;
    ensure_root_directory(root)?;
    let mut current = root.to_path_buf();
    for component in relative.components() {
        match component {
            Component::CurDir => {}
            Component::Normal(segment) => {
                current.push(segment);
                ensure_directory(&current)?;
            }
            Component::ParentDir | Component::RootDir | Component::Prefix(_) => {
                bail!("artifact path escapes its root: {}", relative.display());
            }
        }
    }
    Ok(current)
}

fn ensure_root_directory(path: &Path) -> Result<()> {
    fs::create_dir_all(path)
        .with_context(|| format!("creating artifact root {}", path.display()))?;
    let metadata = fs::symlink_metadata(path)
        .with_context(|| format!("reading artifact root {}", path.display()))?;
    ensure!(
        metadata.is_dir() && !metadata.file_type().is_symlink(),
        "artifact root is not a real directory: {}",
        path.display()
    );
    Ok(())
}

fn validate_relative_path(path: &Path) -> Result<()> {
    ensure!(
        !path.as_os_str().is_empty(),
        "artifact path must not be empty"
    );
    ensure!(
        !path.is_absolute(),
        "artifact path must be relative: {}",
        path.display()
    );
    let mut normal_components = 0usize;
    for component in path.components() {
        match component {
            Component::Normal(_) => normal_components += 1,
            Component::CurDir => {}
            Component::ParentDir | Component::RootDir | Component::Prefix(_) => {
                bail!("artifact path escapes its root: {}", path.display());
            }
        }
    }
    ensure!(normal_components > 0, "artifact path must name a directory");
    Ok(())
}

fn ensure_directory(path: &Path) -> Result<()> {
    match fs::symlink_metadata(path) {
        Ok(metadata) => {
            ensure!(
                metadata.is_dir() && !metadata.file_type().is_symlink(),
                "artifact directory is not a real directory: {}",
                path.display()
            );
        }
        Err(error) if error.kind() == ErrorKind::NotFound => {
            fs::create_dir(path)
                .with_context(|| format!("creating artifact directory {}", path.display()))?;
        }
        Err(error) => {
            return Err(error)
                .with_context(|| format!("reading artifact directory {}", path.display()));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scenario::{Action, HiSpec, ProviderSpec, SessionSeed, TerminalSpec, WorkspaceSpec};

    fn scenario(source_dir: &Path) -> Scenario {
        Scenario {
            schema_version: 1,
            name: "artifact-test".to_owned(),
            tags: vec!["pr".to_owned()],
            timeout_ms: 1_000,
            terminal: TerminalSpec::default(),
            workspace: WorkspaceSpec::default(),
            session: SessionSeed::default(),
            hi: HiSpec::default(),
            provider: ProviderSpec::default(),
            actions: Vec::new(),
            assertions: Vec::new(),
            source_dir: source_dir.to_path_buf(),
        }
    }

    #[test]
    fn live_success_summary_records_route_and_safe_request_counts_only() {
        let temporary = tempfile::tempdir().unwrap();
        let artifacts = temporary.path().join("artifacts");
        fs::create_dir(&artifacts).unwrap();
        let workspace = temporary.path().join("workspace");
        fs::create_dir(&workspace).unwrap();
        let scenario = scenario(temporary.path());
        let empty = Value::Object(Default::default());
        let screens = BTreeMap::new();
        let live_route = LiveRoute::new(
            "pipenetwork",
            "pipe/deepseek-v4-flash-0731",
            "https://api.pipenetwork.ai/v1",
        )
        .unwrap();
        let provider_requests = vec![
            serde_json::json!({
                "provider": "openai_compatible",
                "model": "pipe/deepseek-v4-flash-0731",
                "request_attempt": 1,
                "accepted": true,
                "response_status": 200,
                "request_body": "private-success-prompt",
            }),
            serde_json::json!({
                "provider": "openai_compatible",
                "model": "pipe/deepseek-v4-flash-0731",
                "request_attempt": 2,
                "accepted": false,
                "response_status": 503,
                "request_body": "private-retry-prompt",
            }),
        ];
        let raw_terminal = RawTerminal {
            bytes: b"ok".to_vec(),
            truncated: false,
            total_bytes: 2,
        };
        let input = BundleInput {
            scenario: &scenario,
            mode: "live",
            live_route: Some(&live_route),
            status: BundleStatus::Passed,
            seed: None,
            duration_ms: 12,
            failure: None,
            tui_events: &[],
            raw_terminal: &raw_terminal,
            screens: &screens,
            provider_requests: &provider_requests,
            redaction_values: &[],
            session_jsonl: b"",
            initial_workspace_root: Some(&workspace),
            workspace_root: &workspace,
            workspace_patch: "",
            isolation_evidence: &empty,
            process: &empty,
            assertions: &empty,
            timings: &empty,
            result: &empty,
        };

        let paths = write_case_bundle(&artifacts, Path::new("case-1"), &input).unwrap();
        assert!(paths.summary.is_file());
        assert_eq!(paths.replay, None);
        assert!(!paths.directory.join("raw-terminal.bin").exists());
        let summary: Value = serde_json::from_slice(&fs::read(paths.summary).unwrap()).unwrap();
        assert_eq!(summary["status"], "passed");
        assert_eq!(summary["detailed_evidence"], false);
        assert_eq!(summary["live_route"]["provider"], "pipenetwork");
        assert_eq!(
            summary["live_route"]["model"],
            "pipe/deepseek-v4-flash-0731"
        );
        assert_eq!(
            summary["live_route"]["base_url"],
            "https://api.pipenetwork.ai/v1"
        );
        assert_eq!(summary["provider_request_count"], 2);
        assert_eq!(summary["provider_chat_request_count"], 2);
        assert_eq!(summary["provider_accepted_request_count"], 1);
        assert_eq!(summary["provider_response_status_counts"]["200"], 1);
        assert_eq!(summary["provider_response_status_counts"]["503"], 1);
        let encoded = serde_json::to_string(&summary).unwrap();
        assert!(!encoded.contains("private-success-prompt"));
        assert!(!encoded.contains("private-retry-prompt"));
    }

    #[test]
    fn chat_request_count_excludes_models_and_matches_live_wire_audits() {
        let requests = vec![
            serde_json::json!({"method": "GET", "path": "/v1/models"}),
            serde_json::json!({"method": "POST", "path": "/v1/chat/completions"}),
            serde_json::json!({"method": "POST", "path": "/chat/completions?trace=1"}),
            serde_json::json!({"method": "GET", "path": "/v1/chat/completions"}),
            serde_json::json!({
                "provider": "openai_compatible",
                "model": "pipe/deepseek-v4-flash-0731",
                "request_attempt": 1,
            }),
            serde_json::json!({"accepted": true}),
        ];

        assert_eq!(provider_chat_request_count(&requests), 3);
    }

    #[test]
    fn failure_writes_complete_redacted_bounded_replay_bundle() {
        let temporary = tempfile::tempdir().unwrap();
        let artifacts = temporary.path().join("artifacts");
        fs::create_dir(&artifacts).unwrap();
        let initial_workspace = temporary.path().join("initial-workspace");
        fs::create_dir(&initial_workspace).unwrap();
        fs::write(initial_workspace.join("file.txt"), "before\n").unwrap();
        let workspace = temporary.path().join("workspace");
        fs::create_dir(&workspace).unwrap();
        fs::write(workspace.join("file.txt"), "changed\n").unwrap();
        let scenario = scenario(temporary.path());
        let terminal = RawTerminal {
            bytes: vec![b'x'; RAW_TERMINAL_LIMIT_BYTES],
            truncated: true,
            total_bytes: (RAW_TERMINAL_LIMIT_BYTES + 1_024) as u64,
        };
        let mut screens = BTreeMap::new();
        screens.insert("last/frame".to_owned(), "line   \r\n\r\n".to_owned());
        let requests = vec![serde_json::json!({
            "headers": {"Authorization": "Bearer visible"},
            "body": "key=public api_key=secret-value",
            "api_key": "also-visible"
        })];
        let empty = serde_json::json!({});
        let process = serde_json::json!({"exit_status": 1, "leaked_processes": []});
        let secrets = vec!["secret-value".to_owned()];
        let input = BundleInput {
            scenario: &scenario,
            mode: "scripted",
            live_route: None,
            status: BundleStatus::Failed,
            seed: Some(42),
            duration_ms: 50,
            failure: Some("assertion failed"),
            tui_events: &[serde_json::json!({"kind": "turn_started"})],
            raw_terminal: &terminal,
            screens: &screens,
            provider_requests: &requests,
            redaction_values: &secrets,
            session_jsonl: b"{\"kind\":\"session\"}\n",
            initial_workspace_root: Some(&initial_workspace),
            workspace_root: &workspace,
            workspace_patch: "diff --git a/file.txt b/file.txt\n",
            isolation_evidence: &empty,
            process: &process,
            assertions: &serde_json::json!([{"passed": false}]),
            timings: &serde_json::json!({"total_ms": 50}),
            result: &empty,
        };

        let paths = write_case_bundle(&artifacts, Path::new("case-1"), &input).unwrap();
        let replay = paths.replay.unwrap();
        assert!(replay.is_file());
        let replay_scenario = Scenario::parse(&replay).unwrap();
        assert_eq!(
            replay_scenario.workspace.fixture.as_deref(),
            Some("replay-fixture")
        );
        assert_eq!(
            fs::read_to_string(paths.directory.join("replay-fixture/file.txt")).unwrap(),
            "before\n"
        );
        for name in [
            "tui-events.jsonl",
            "raw-terminal.bin",
            "raw-terminal.meta.json",
            "screens.json",
            "provider-requests.jsonl",
            "session.jsonl",
            "workspace-listing.json",
            "workspace.patch",
            "isolation-evidence.json",
            "process.json",
            "assertions.json",
            "timings.json",
            "result.json",
        ] {
            assert!(paths.directory.join(name).is_file(), "missing {name}");
        }

        let raw = fs::read(paths.directory.join("raw-terminal.bin")).unwrap();
        assert_eq!(raw.len(), RAW_TERMINAL_LIMIT_BYTES);
        assert!(String::from_utf8_lossy(&raw).contains("raw terminal truncated"));
        let metadata: Value = serde_json::from_slice(
            &fs::read(paths.directory.join("raw-terminal.meta.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(metadata["truncated"], true);
        assert_eq!(metadata["written_bytes"], RAW_TERMINAL_LIMIT_BYTES);

        let requests = fs::read_to_string(paths.directory.join("provider-requests.jsonl")).unwrap();
        assert!(!requests.contains("secret-value"));
        assert!(!requests.contains("also-visible"));
        assert!(!requests.contains("Bearer visible"));
        assert!(requests.contains("[REDACTED]"));
        assert_eq!(
            fs::read_to_string(paths.directory.join("screens/000-last-frame.txt")).unwrap(),
            "line\n"
        );
    }

    #[test]
    fn shortening_raw_redaction_is_not_reported_as_truncation() {
        let temporary = tempfile::tempdir().unwrap();
        let secret = "a-very-long-live-secret-that-redacts-to-a-short-placeholder".to_owned();
        let original = format!("before {secret} after").into_bytes();
        let raw = RawTerminal {
            bytes: original.clone(),
            truncated: false,
            total_bytes: original.len() as u64,
        };
        let sanitizer = ArtifactSanitizer::with_secrets(std::slice::from_ref(&secret)).finish();

        write_raw_terminal(temporary.path(), &raw, &sanitizer).unwrap();

        let written = fs::read(temporary.path().join("raw-terminal.bin")).unwrap();
        assert_eq!(written, b"before [REDACTED] after");
        assert!(written.len() <= RAW_TERMINAL_LIMIT_BYTES);
        let metadata: Value = serde_json::from_slice(
            &fs::read(temporary.path().join("raw-terminal.meta.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(metadata["original_bytes"], original.len());
        assert_eq!(metadata["sanitized_captured_bytes"], written.len());
        assert_eq!(metadata["written_bytes"], written.len());
        assert_eq!(metadata["truncated"], false);
        assert_eq!(
            metadata["captured_blake3"],
            blake3::hash(&written).to_hex().to_string()
        );
    }

    #[test]
    fn failure_bundle_redacts_known_secret_and_ephemeral_paths_everywhere() {
        const SECRET: &str = "pk_live_SENTINEL_do_not_persist_7f93";

        let temporary = tempfile::tempdir().unwrap();
        let artifacts = temporary.path().join("artifacts");
        fs::create_dir(&artifacts).unwrap();
        let isolation = temporary.path().join("random-isolation-a81c9e");
        let initial_workspace = isolation.join("initial-workspace");
        let workspace = isolation.join("workspace");
        fs::create_dir_all(&initial_workspace).unwrap();
        fs::create_dir_all(&workspace).unwrap();
        let sensitive_fixture_name = format!("seed-{SECRET}.txt");
        let fixture_text = format!(
            "fixture secret={SECRET} initial={} workspace={}",
            initial_workspace.display(),
            workspace.display()
        );
        fs::write(
            initial_workspace.join(&sensitive_fixture_name),
            &fixture_text,
        )
        .unwrap();
        fs::write(
            workspace.join(format!("result-{SECRET}.txt")),
            format!("{SECRET} at {}", workspace.display()),
        )
        .unwrap();

        let mut scenario = scenario(temporary.path());
        scenario.actions.push(Action::SendLine {
            text: format!("replay {SECRET} from {}", workspace.display()),
        });
        scenario.hi.env.insert(
            "EVIDENCE_SENTINEL".to_owned(),
            format!("{SECRET}:{}", isolation.display()),
        );

        let mut terminal_bytes = vec![0xff];
        terminal_bytes.extend_from_slice(
            format!(
                " terminal {SECRET} {} {}",
                workspace.display(),
                isolation.display()
            )
            .as_bytes(),
        );
        let terminal = RawTerminal {
            total_bytes: terminal_bytes.len() as u64,
            bytes: terminal_bytes,
            truncated: false,
        };
        let mut screens = BTreeMap::new();
        screens.insert(
            format!("screen-{SECRET}"),
            format!("{SECRET}\n{}\n{}", workspace.display(), isolation.display()),
        );
        let provider_requests = vec![serde_json::json!({
            "accepted": true,
            "response_status": 200,
            "authorization": format!("Bearer {SECRET}"),
            "diagnostic": format!("{SECRET} {}", workspace.display()),
        })];
        let session = serde_json::to_vec(&serde_json::json!({
            "role": "System",
            "content": format!("key={SECRET}; cwd={}; root={}", workspace.display(), isolation.display()),
        }))
        .unwrap();
        let process = serde_json::json!({
            "command": format!("tool --token={SECRET} --cwd {}", workspace.display()),
            "leaked_processes": [],
        });
        let assertions = serde_json::json!([{
            "passed": false,
            "failure": format!("{SECRET} at {}", workspace.display()),
        }]);
        let timings = serde_json::json!({format!("path:{}", isolation.display()): 1});
        let result = serde_json::json!({
            "failure": format!("{SECRET} at {}", initial_workspace.display()),
        });
        let live_route = LiveRoute::new(
            "pipenetwork",
            "pipe/deepseek-v4-flash-0731",
            "https://api.pipenetwork.ai/v1",
        )
        .unwrap();
        let secrets = vec![SECRET.to_owned()];
        let patch = format!(
            "diff --git a/{SECRET}.txt b/{SECRET}.txt\n+{SECRET} {}\n",
            workspace.display()
        );
        let failure = format!("failure {SECRET} in {}", workspace.display());
        let events = vec![serde_json::json!({
            "event": "diagnostic",
            "data": {"message": format!("{SECRET} {}", isolation.display())},
        })];
        let isolation_evidence = serde_json::json!({
            "unexpected_mutation_count": 1,
            "mutations": [{
                "path": format!("home/{SECRET}-escape"),
                "disposition": "unexpected_outside_workspace",
            }],
        });
        let input = BundleInput {
            scenario: &scenario,
            mode: "live",
            live_route: Some(&live_route),
            status: BundleStatus::Failed,
            seed: Some(91),
            duration_ms: 15,
            failure: Some(&failure),
            tui_events: &events,
            raw_terminal: &terminal,
            screens: &screens,
            provider_requests: &provider_requests,
            redaction_values: &secrets,
            session_jsonl: &session,
            initial_workspace_root: Some(&initial_workspace),
            workspace_root: &workspace,
            workspace_patch: &patch,
            isolation_evidence: &isolation_evidence,
            process: &process,
            assertions: &assertions,
            timings: &timings,
            result: &result,
        };

        let paths = write_case_bundle(&artifacts, Path::new("case-1"), &input).unwrap();
        let mut files = Vec::new();
        collect_regular_files(&paths.directory, &mut files);
        assert!(!files.is_empty());
        let workspace_text = workspace.to_string_lossy().into_owned();
        let initial_workspace_text = initial_workspace.to_string_lossy().into_owned();
        let isolation_text = isolation.to_string_lossy().into_owned();
        let forbidden = [
            SECRET.as_bytes(),
            workspace_text.as_bytes(),
            initial_workspace_text.as_bytes(),
            isolation_text.as_bytes(),
        ];
        for file in &files {
            let relative = file.strip_prefix(&paths.directory).unwrap();
            let relative_text = relative.to_string_lossy();
            let contents = fs::read(file).unwrap();
            for needle in forbidden {
                assert!(
                    !relative_text
                        .as_bytes()
                        .windows(needle.len())
                        .any(|w| w == needle),
                    "sensitive value remained in artifact filename {relative_text}"
                );
                assert!(
                    !contents
                        .windows(needle.len())
                        .any(|window| window == needle),
                    "sensitive value remained in {}",
                    relative.display()
                );
            }
        }

        let replay = paths.replay.as_ref().unwrap();
        let replay_text = fs::read_to_string(replay).unwrap();
        assert!(replay_text.contains("# hi-smoke-replay-mode = live"));
        assert!(replay_text.contains("# hi-smoke-live-provider = pipenetwork"));
        assert!(replay_text.contains("# hi-smoke-live-model = pipe/deepseek-v4-flash-0731"));
        assert!(replay_text.contains("[REDACTED]"));
        let replay_scenario = Scenario::parse(replay).unwrap();
        assert_eq!(
            replay_scenario.workspace.fixture.as_deref(),
            Some("replay-fixture")
        );
        assert!(
            paths
                .directory
                .join("replay-fixture")
                .join("seed-[REDACTED].txt")
                .is_file()
        );
        let sanitized_session = fs::read_to_string(paths.directory.join("session.jsonl")).unwrap();
        let _: Value = serde_json::from_str(&sanitized_session).unwrap();
        assert!(sanitized_session.contains("[REDACTED]"));
        assert!(sanitized_session.contains("<WORKSPACE>"));
        let raw = fs::read(paths.directory.join("raw-terminal.bin")).unwrap();
        let raw = String::from_utf8_lossy(&raw);
        assert!(raw.contains("[REDACTED]"));
        assert!(raw.contains("<WORKSPACE>"));
    }

    fn collect_regular_files(directory: &Path, files: &mut Vec<PathBuf>) {
        let mut entries = fs::read_dir(directory)
            .unwrap()
            .collect::<std::result::Result<Vec<_>, _>>()
            .unwrap();
        entries.sort_by_key(fs::DirEntry::file_name);
        for entry in entries {
            let file_type = entry.file_type().unwrap();
            if file_type.is_dir() {
                collect_regular_files(&entry.path(), files);
            } else if file_type.is_file() {
                files.push(entry.path());
            }
        }
    }

    #[test]
    fn detailed_bundle_failure_is_repaired_with_a_fresh_exact_fixture() {
        let temporary = tempfile::tempdir().unwrap();
        let artifacts = temporary.path().join("artifacts");
        fs::create_dir(&artifacts).unwrap();
        let initial_workspace = temporary.path().join("initial-workspace");
        fs::create_dir(&initial_workspace).unwrap();
        fs::write(initial_workspace.join("original.txt"), "original\n").unwrap();
        let missing_final_workspace = temporary.path().join("missing-workspace");
        let scenario = scenario(temporary.path());
        let terminal = RawTerminal::default();
        let empty = serde_json::json!({});
        let live_route = LiveRoute::new(
            "pipenetwork",
            "pipe/test-model",
            "https://api.pipenetwork.ai/v1",
        )
        .unwrap();
        let input = BundleInput {
            scenario: &scenario,
            mode: "live",
            live_route: Some(&live_route),
            status: BundleStatus::Failed,
            seed: Some(7),
            duration_ms: 25,
            failure: Some("scenario failed first"),
            tui_events: &[],
            raw_terminal: &terminal,
            screens: &BTreeMap::new(),
            provider_requests: &[],
            redaction_values: &[],
            session_jsonl: b"",
            initial_workspace_root: Some(&initial_workspace),
            workspace_root: &missing_final_workspace,
            workspace_patch: "",
            isolation_evidence: &empty,
            process: &empty,
            assertions: &empty,
            timings: &empty,
            result: &empty,
        };

        let paths = write_case_bundle(&artifacts, Path::new("case-1"), &input).unwrap();
        let replay = paths.replay.expect("repaired replay");
        let replay_scenario = Scenario::parse(&replay).unwrap();
        assert_eq!(
            replay_scenario.workspace.fixture.as_deref(),
            Some("replay-fixture-repair-1")
        );
        assert_eq!(
            fs::read_to_string(paths.directory.join("replay-fixture-repair-1/original.txt"))
                .unwrap(),
            "original\n"
        );
        let replay_text = fs::read_to_string(replay).unwrap();
        assert!(replay_text.contains("# hi-smoke-replay-mode = live"));
        assert!(replay_text.contains("# hi-smoke-live-provider = pipenetwork"));
        assert!(replay_text.contains("# hi-smoke-live-model = pipe/test-model"));
        assert!(replay_text.contains("# hi-smoke-live-base-url = https://api.pipenetwork.ai/v1"));
        assert!(!replay_text.contains("api_key"));
        let summary: Value = serde_json::from_slice(&fs::read(paths.summary).unwrap()).unwrap();
        assert_eq!(summary["detailed_evidence"], false);
        assert_eq!(summary["mode"], "live");
        assert!(paths.directory.join("bundle-write-failure.txt").is_file());
    }

    #[test]
    fn minimal_repair_does_not_trust_a_partial_fixture() {
        let temporary = tempfile::tempdir().unwrap();
        let artifacts = temporary.path().join("artifacts");
        fs::create_dir(&artifacts).unwrap();
        let case = artifacts.join("case-1");
        fs::create_dir(&case).unwrap();
        fs::create_dir(case.join("replay-fixture")).unwrap();
        fs::write(case.join("replay-fixture/partial.txt"), "partial").unwrap();
        let fixture = temporary.path().join("fixture");
        fs::create_dir(&fixture).unwrap();
        fs::write(fixture.join("complete.txt"), "complete").unwrap();
        let scenario = scenario(temporary.path());

        let paths = repair_minimal_failure_bundle(
            &artifacts,
            Path::new("case-1"),
            &MinimalBundleInput {
                scenario: &scenario,
                mode: "scripted",
                live_route: None,
                seed: None,
                duration_ms: 1,
                failure: "setup failed",
                fixture_root: Some(&fixture),
                redaction_values: &[],
                detailed_bundle_failure: Some("partial detailed bundle"),
            },
        )
        .unwrap();
        let replay = Scenario::parse(paths.replay.as_ref().unwrap()).unwrap();
        assert_eq!(
            replay.workspace.fixture.as_deref(),
            Some("replay-fixture-repair-1")
        );
        assert!(
            paths
                .directory
                .join("replay-fixture-repair-1/complete.txt")
                .is_file()
        );
    }

    #[test]
    fn rejects_escape_and_symlinked_bundle_paths() {
        let temporary = tempfile::tempdir().unwrap();
        let root = temporary.path().join("artifacts");
        fs::create_dir(&root).unwrap();
        assert!(create_contained_dir(&root, Path::new("../escape")).is_err());
        assert!(create_contained_dir(&root, Path::new(".")).is_err());

        #[cfg(unix)]
        {
            let outside = temporary.path().join("outside");
            fs::create_dir(&outside).unwrap();
            std::os::unix::fs::symlink(&outside, root.join("linked")).unwrap();
            assert!(create_contained_dir(&root, Path::new("linked/case")).is_err());
        }
    }

    #[test]
    fn workspace_listing_is_sorted_and_does_not_follow_symlinks() {
        let temporary = tempfile::tempdir().unwrap();
        let workspace = temporary.path().join("workspace");
        fs::create_dir(&workspace).unwrap();
        fs::create_dir(workspace.join("b")).unwrap();
        fs::write(workspace.join("a"), "123").unwrap();
        #[cfg(unix)]
        std::os::unix::fs::symlink(temporary.path(), workspace.join("c-link")).unwrap();

        let listing = capture_workspace_listing(&workspace).unwrap();
        let paths = listing
            .iter()
            .map(|entry| entry.path.as_str())
            .collect::<Vec<_>>();
        #[cfg(unix)]
        assert_eq!(paths, vec!["a", "b", "c-link"]);
        #[cfg(not(unix))]
        assert_eq!(paths, vec!["a", "b"]);
        assert_eq!(listing[0].bytes, Some(3));
    }

    #[test]
    fn suite_summary_and_json_are_stable() {
        let temporary = tempfile::tempdir().unwrap();
        let root = temporary.path().join("artifacts");
        fs::create_dir(&root).unwrap();
        let first = serde_json::json!({"z": 1, "a": {"y": 2, "b": 3}});
        let path = write_suite_summary(&root, &first, &[]).unwrap();
        let once = fs::read(&path).unwrap();
        write_suite_summary(&root, &first, &[]).unwrap();
        assert_eq!(once, fs::read(path).unwrap());
        assert_eq!(
            String::from_utf8(once).unwrap(),
            "{\n  \"a\": {\n    \"b\": 3,\n    \"y\": 2\n  },\n  \"z\": 1\n}\n"
        );
    }

    #[test]
    fn suite_summary_redacts_secret_and_artifact_root() {
        let temporary = tempfile::tempdir().unwrap();
        let root = temporary.path().join("random-host-root-71/artifacts");
        fs::create_dir_all(&root).unwrap();
        let secret = "pk_live_suite_summary_sentinel".to_owned();
        let summary = serde_json::json!({
            "cases": [{
                "failure": format!("key={secret} evidence={}/case-a", root.display()),
                "artifact_dir": format!("{}/case-a", root.display()),
            }],
        });

        let path = write_suite_summary(&root, &summary, std::slice::from_ref(&secret)).unwrap();
        let text = fs::read_to_string(path).unwrap();
        assert!(!text.contains(&secret));
        assert!(!text.contains(root.to_string_lossy().as_ref()));
        assert!(text.contains("[REDACTED]"));
        assert!(text.contains("<ARTIFACT_ROOT>/case-a"));
    }
}
