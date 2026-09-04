use std::collections::BTreeMap;
use std::path::{Component, Path};

use anyhow::{Context, Result, bail, ensure};
use serde::{Deserialize, Serialize};
use serde_json::Value;

pub(crate) const SCENARIO_SCHEMA_VERSION: u16 = 1;
pub(crate) const MAX_SCENARIO_TIMEOUT_MS: u64 = 15 * 60 * 1_000;

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Scenario {
    pub schema_version: u16,
    pub name: String,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default = "default_timeout_ms")]
    pub timeout_ms: u64,
    #[serde(default)]
    pub terminal: TerminalSpec,
    #[serde(default)]
    pub workspace: WorkspaceSpec,
    #[serde(default)]
    pub session: SessionSeed,
    #[serde(default)]
    pub hi: HiSpec,
    #[serde(default)]
    pub provider: ProviderSpec,
    #[serde(default)]
    pub actions: Vec<Action>,
    #[serde(default)]
    pub assertions: Vec<Assertion>,
    #[serde(skip)]
    pub source_dir: std::path::PathBuf,
}

fn default_timeout_ms() -> u64 {
    120_000
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub(crate) struct TerminalSpec {
    pub cols: u16,
    pub rows: u16,
}

impl Default for TerminalSpec {
    fn default() -> Self {
        Self {
            cols: 160,
            rows: 50,
        }
    }
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub(crate) struct WorkspaceSpec {
    pub fixture: Option<String>,
    pub git: GitState,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum GitState {
    #[default]
    None,
    Clean,
    Dirty,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub(crate) struct SessionSeed {
    pub plan: Vec<PlanSeed>,
    pub plan_drive_paused: bool,
    pub plan_drive_resume_on_user_input: bool,
    pub plan_drive_stall: u32,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct PlanSeed {
    pub title: String,
    pub status: PlanSeedStatus,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum PlanSeedStatus {
    Pending,
    Active,
    Done,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub(crate) struct HiSpec {
    pub args: Vec<String>,
    pub env: BTreeMap<String, String>,
    /// Harness-owned soft deadline passed to `hi --turn-deadline`. `None`
    /// selects the mode default (30s scripted, 240s live); `Some(0)` disables
    /// the product deadline for scenarios that explicitly exercise continual
    /// productive work. The scenario's own outer timeout still bounds cleanup.
    pub turn_deadline_secs: Option<u64>,
    /// Harness watchdog for one active turn. `None` selects the mode default
    /// (45s scripted, 300s live). This is external observation/cleanup policy,
    /// never a `hi` product setting.
    pub outer_turn_kill_secs: Option<u64>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub(crate) struct ProviderSpec {
    pub steps: Vec<ProviderStep>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ProviderStep {
    pub id: String,
    #[serde(default)]
    pub expect: RequestExpectation,
    pub response: ProviderResponse,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub(crate) struct RequestExpectation {
    pub body_contains: Vec<String>,
    pub body_excludes: Vec<String>,
    pub json_equals: BTreeMap<String, Value>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub(crate) enum ProviderResponse {
    Text {
        text: String,
        #[serde(default)]
        gate: Option<String>,
        #[serde(default)]
        delay_ms: u64,
        #[serde(default)]
        chunk_bytes: Option<usize>,
        #[serde(default)]
        terminal: StreamTerminal,
    },
    ToolCall {
        name: String,
        arguments: Value,
        #[serde(default)]
        gate: Option<String>,
        #[serde(default)]
        delay_ms: u64,
    },
    HttpError {
        status: u16,
        #[serde(default)]
        body: String,
        #[serde(default)]
        gate: Option<String>,
    },
    RawSse {
        body: String,
        #[serde(default)]
        gate: Option<String>,
        #[serde(default)]
        delay_ms: u64,
        #[serde(default)]
        chunk_bytes: Option<usize>,
        #[serde(default)]
        terminal: StreamTerminal,
    },
    Hold {
        #[serde(default)]
        gate: Option<String>,
    },
    Reset {
        #[serde(default)]
        gate: Option<String>,
    },
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum StreamTerminal {
    #[default]
    Done,
    Eof,
    Reset,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub(crate) enum Action {
    SendLine {
        text: String,
    },
    SendKey {
        key: Key,
    },
    Resize {
        cols: u16,
        rows: u16,
    },
    WaitEvent {
        #[serde(default)]
        equals: BTreeMap<String, Value>,
        #[serde(default)]
        contains: BTreeMap<String, String>,
        #[serde(default = "default_transition_timeout_ms")]
        timeout_ms: u64,
    },
    WaitEventAbsent {
        #[serde(default)]
        equals: BTreeMap<String, Value>,
        #[serde(default)]
        contains: BTreeMap<String, String>,
        duration_ms: u64,
    },
    WaitProviderRequest {
        count: usize,
        #[serde(default = "default_transition_timeout_ms")]
        timeout_ms: u64,
    },
    WaitFile {
        path: String,
        #[serde(default = "default_true")]
        exists: bool,
        #[serde(default = "default_transition_timeout_ms")]
        timeout_ms: u64,
    },
    WaitProcess {
        command_contains: String,
        #[serde(default = "default_one")]
        at_least: usize,
        #[serde(default = "default_transition_timeout_ms")]
        timeout_ms: u64,
    },
    WaitQuiescent {
        source: QuiescentSource,
        quiet_ms: u64,
        #[serde(default = "default_transition_timeout_ms")]
        timeout_ms: u64,
    },
    ReleaseGate {
        gate: String,
    },
    CaptureScreen {
        name: String,
    },
    Restart,
    Quit,
}

fn default_transition_timeout_ms() -> u64 {
    5_000
}

fn default_true() -> bool {
    true
}

fn default_one() -> usize {
    1
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Key {
    Enter,
    Escape,
    CtrlC,
    CtrlD,
    Tab,
    ShiftTab,
    Up,
    Down,
    Left,
    Right,
}

impl Key {
    pub(crate) fn bytes(self) -> &'static [u8] {
        match self {
            Self::Enter => b"\r",
            Self::Escape => b"\x1b",
            Self::CtrlC => b"\x03",
            Self::CtrlD => b"\x04",
            Self::Tab => b"\t",
            Self::ShiftTab => b"\x1b[Z",
            Self::Up => b"\x1b[A",
            Self::Down => b"\x1b[B",
            Self::Right => b"\x1b[C",
            Self::Left => b"\x1b[D",
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum QuiescentSource {
    Events,
    Provider,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub(crate) enum Assertion {
    Records {
        source: RecordSource,
        #[serde(default)]
        equals: BTreeMap<String, Value>,
        #[serde(default)]
        contains: BTreeMap<String, String>,
        #[serde(default)]
        exact: Option<usize>,
        #[serde(default)]
        at_least: Option<usize>,
        #[serde(default)]
        at_most: Option<usize>,
    },
    RecordSequence {
        source: RecordSource,
        #[serde(default)]
        where_equals: BTreeMap<String, Value>,
        #[serde(default)]
        where_contains: BTreeMap<String, String>,
        pointer: String,
        values: Vec<Value>,
    },
    SubstringOccurrences {
        source: RecordSource,
        #[serde(default)]
        equals: BTreeMap<String, Value>,
        #[serde(default)]
        contains: BTreeMap<String, String>,
        pointer: String,
        substring: String,
        exact: usize,
    },
    AllRecords {
        source: RecordSource,
        #[serde(default)]
        where_equals: BTreeMap<String, Value>,
        #[serde(default)]
        where_contains: BTreeMap<String, String>,
        #[serde(default)]
        equals: BTreeMap<String, Value>,
        #[serde(default)]
        contains: BTreeMap<String, String>,
        #[serde(default = "default_one")]
        at_least: usize,
    },
    Screen {
        snapshot: String,
        #[serde(default)]
        contains: Vec<String>,
        #[serde(default)]
        excludes: Vec<String>,
    },
    File {
        path: String,
        #[serde(default = "default_true")]
        exists: bool,
        #[serde(default)]
        contains: Option<String>,
        #[serde(default)]
        equals: Option<String>,
    },
    WorkspacePatch {
        #[serde(default)]
        contains: Vec<String>,
        #[serde(default)]
        excludes: Vec<String>,
        #[serde(default)]
        equals: Option<String>,
    },
    WorkspaceListing {
        #[serde(default)]
        contains: Vec<String>,
        #[serde(default)]
        excludes: Vec<String>,
    },
    Exit {
        code: u32,
    },
    ProviderConsumed,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum RecordSource {
    Events,
    Session,
    ProviderRequests,
}

impl Scenario {
    pub(crate) fn parse(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("reading scenario {}", path.display()))?;
        let mut scenario: Self = toml::from_str(&text)
            .with_context(|| format!("parsing scenario {}", path.display()))?;
        scenario.source_dir = path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .to_path_buf();
        scenario
            .validate()
            .with_context(|| format!("validating scenario {}", path.display()))?;
        Ok(scenario)
    }

    pub(crate) fn validate(&self) -> Result<()> {
        ensure!(
            self.schema_version == SCENARIO_SCHEMA_VERSION,
            "unsupported schema_version {}; expected {}",
            self.schema_version,
            SCENARIO_SCHEMA_VERSION
        );
        ensure!(!self.name.trim().is_empty(), "name must not be empty");
        ensure!(
            self.timeout_ms > 0 && self.timeout_ms <= MAX_SCENARIO_TIMEOUT_MS,
            "timeout_ms must be in 1..={MAX_SCENARIO_TIMEOUT_MS}"
        );
        ensure!(
            self.terminal.cols >= 20 && self.terminal.rows >= 8,
            "terminal must be at least 20x8"
        );
        if let Some(fixture) = &self.workspace.fixture {
            validate_relative_path(fixture)?;
            let scenario_root = self.source_dir.canonicalize().with_context(|| {
                format!(
                    "canonicalizing scenario directory {}",
                    self.source_dir.display()
                )
            })?;
            let fixture_path = self.source_dir.join(fixture);
            let metadata = std::fs::symlink_metadata(&fixture_path)
                .with_context(|| format!("reading fixture metadata {fixture}"))?;
            ensure!(
                metadata.file_type().is_dir() && !metadata.file_type().is_symlink(),
                "fixture must be a real directory, not a file or symlink: {fixture}"
            );
            let canonical_fixture = fixture_path
                .canonicalize()
                .with_context(|| format!("canonicalizing fixture directory {fixture}"))?;
            ensure!(
                canonical_fixture.starts_with(&scenario_root),
                "fixture directory escapes the scenario root: {fixture}"
            );
        }
        for step in &self.session.plan {
            ensure!(
                !step.title.trim().is_empty(),
                "plan titles must not be empty"
            );
        }
        validate_hi_args(&self.hi.args)?;
        validate_hi_env(&self.hi.env)?;
        if let Some(seconds) = self.hi.turn_deadline_secs {
            ensure!(
                seconds <= MAX_SCENARIO_TIMEOUT_MS / 1_000,
                "hi.turn_deadline_secs must be in 0..={}",
                MAX_SCENARIO_TIMEOUT_MS / 1_000
            );
        }
        if let Some(seconds) = self.hi.outer_turn_kill_secs {
            ensure!(
                seconds > 0 && seconds.saturating_mul(1_000) <= self.timeout_ms,
                "hi.outer_turn_kill_secs must be positive and no greater than the scenario timeout"
            );
        }
        let mut ids = std::collections::BTreeSet::new();
        let mut gates = std::collections::BTreeSet::new();
        for step in &self.provider.steps {
            ensure!(
                !step.id.trim().is_empty(),
                "provider step ids must not be empty"
            );
            ensure!(
                ids.insert(step.id.as_str()),
                "duplicate provider step id: {}",
                step.id
            );
            if let Some(gate) = step.response.gate() {
                ensure!(
                    !gate.trim().is_empty(),
                    "provider gate names must not be empty"
                );
                ensure!(gates.insert(gate), "duplicate provider gate name: {gate}");
            }
            if matches!(step.response, ProviderResponse::Hold { gate: None }) {
                bail!("held provider step {:?} requires a named gate", step.id);
            }
            if matches!(
                &step.response,
                ProviderResponse::Text {
                    chunk_bytes: Some(0),
                    ..
                } | ProviderResponse::RawSse {
                    chunk_bytes: Some(0),
                    ..
                }
            ) {
                bail!("provider step {:?} has zero chunk_bytes", step.id);
            }
            for pointer in step.expect.json_equals.keys() {
                validate_json_pointer(pointer)?;
            }
        }
        for action in &self.actions {
            match action {
                Action::WaitEvent {
                    equals,
                    contains,
                    timeout_ms,
                } => {
                    validate_wait_timeout(*timeout_ms)?;
                    for pointer in equals.keys().chain(contains.keys()) {
                        validate_json_pointer(pointer)?;
                    }
                }
                Action::WaitEventAbsent {
                    equals,
                    contains,
                    duration_ms,
                } => {
                    validate_wait_timeout(*duration_ms)?;
                    ensure!(
                        !equals.is_empty() || !contains.is_empty(),
                        "wait_event_absent needs at least one match field"
                    );
                    for pointer in equals.keys().chain(contains.keys()) {
                        validate_json_pointer(pointer)?;
                    }
                }
                Action::WaitProviderRequest { count, timeout_ms } => {
                    ensure!(
                        *count > 0,
                        "provider request count must be greater than zero"
                    );
                    validate_wait_timeout(*timeout_ms)?;
                }
                Action::WaitQuiescent {
                    quiet_ms,
                    timeout_ms,
                    ..
                } => {
                    ensure!(
                        *quiet_ms > 0,
                        "quiescence quiet_ms must be greater than zero"
                    );
                    validate_wait_timeout(*timeout_ms)?;
                }
                Action::WaitFile {
                    path, timeout_ms, ..
                } => {
                    validate_wait_timeout(*timeout_ms)?;
                    validate_relative_path(path)?;
                }
                Action::WaitProcess {
                    command_contains,
                    at_least,
                    timeout_ms,
                } => {
                    ensure!(
                        !command_contains.trim().is_empty(),
                        "wait_process command_contains must not be empty"
                    );
                    ensure!(*at_least > 0, "wait_process at_least must be positive");
                    validate_wait_timeout(*timeout_ms)?;
                }
                Action::ReleaseGate { gate } => ensure!(
                    gates.contains(gate.as_str()),
                    "action releases undeclared gate {gate:?}"
                ),
                Action::Resize { cols, rows } => ensure!(
                    *cols >= 20 && *rows >= 8,
                    "resized terminal must be at least 20x8"
                ),
                Action::CaptureScreen { name } => {
                    ensure!(
                        !name.trim().is_empty(),
                        "screen snapshot name must not be empty"
                    )
                }
                Action::SendLine { .. }
                | Action::SendKey { .. }
                | Action::Restart
                | Action::Quit => {}
            }
        }
        for assertion in &self.assertions {
            assertion.validate()?;
        }
        Ok(())
    }

    pub(crate) fn has_tag(&self, requested: &[String]) -> bool {
        requested.is_empty()
            || requested
                .iter()
                .all(|tag| self.tags.iter().any(|candidate| candidate == tag))
    }
}

impl ProviderResponse {
    fn gate(&self) -> Option<&str> {
        match self {
            Self::Text { gate, .. }
            | Self::ToolCall { gate, .. }
            | Self::HttpError { gate, .. }
            | Self::RawSse { gate, .. }
            | Self::Hold { gate }
            | Self::Reset { gate } => gate.as_deref(),
        }
    }
}

impl Assertion {
    fn validate(&self) -> Result<()> {
        match self {
            Self::Records {
                equals,
                contains,
                exact,
                at_least,
                at_most,
                ..
            } => {
                let has_range = at_least.is_some() || at_most.is_some();
                ensure!(
                    exact.is_some() ^ has_range,
                    "records assertion needs either exact or a lower/upper count range"
                );
                if let (Some(minimum), Some(maximum)) = (at_least, at_most) {
                    ensure!(
                        minimum <= maximum,
                        "records assertion lower count bound exceeds its upper bound"
                    );
                }
                for pointer in equals.keys().chain(contains.keys()) {
                    validate_json_pointer(pointer)?;
                }
            }
            Self::RecordSequence {
                where_equals,
                where_contains,
                pointer,
                values,
                ..
            } => {
                for pointer in where_equals.keys().chain(where_contains.keys()) {
                    validate_json_pointer(pointer)?;
                }
                validate_json_pointer(pointer)?;
                ensure!(
                    !values.is_empty(),
                    "record_sequence values must not be empty"
                );
            }
            Self::SubstringOccurrences {
                equals,
                contains,
                pointer,
                substring,
                ..
            } => {
                for pointer in equals.keys().chain(contains.keys()) {
                    validate_json_pointer(pointer)?;
                }
                validate_json_pointer(pointer)?;
                ensure!(
                    !substring.is_empty(),
                    "substring_occurrences substring must not be empty"
                );
            }
            Self::AllRecords {
                where_equals,
                where_contains,
                equals,
                contains,
                at_least,
                ..
            } => {
                ensure!(
                    !equals.is_empty() || !contains.is_empty(),
                    "all_records needs at least one required field"
                );
                ensure!(
                    *at_least > 0,
                    "all_records at_least must be greater than zero"
                );
                for pointer in where_equals
                    .keys()
                    .chain(where_contains.keys())
                    .chain(equals.keys())
                    .chain(contains.keys())
                {
                    validate_json_pointer(pointer)?;
                }
            }
            Self::Screen {
                snapshot,
                contains,
                excludes,
            } => {
                ensure!(
                    !snapshot.trim().is_empty(),
                    "screen snapshot must not be empty"
                );
                ensure!(
                    !contains.is_empty() || !excludes.is_empty(),
                    "screen assertion needs contains or excludes"
                );
            }
            Self::File {
                path,
                contains,
                equals,
                ..
            } => {
                validate_relative_path(path)?;
                ensure!(
                    contains.is_none() || equals.is_none(),
                    "file assertion cannot use both contains and equals"
                );
            }
            Self::WorkspacePatch {
                contains,
                excludes,
                equals,
            } => {
                ensure!(
                    equals.is_some() || !contains.is_empty() || !excludes.is_empty(),
                    "workspace_patch assertion needs equals, contains, or excludes"
                );
                ensure!(
                    contains
                        .iter()
                        .chain(excludes)
                        .all(|needle| !needle.is_empty()),
                    "workspace_patch assertion needles must not be empty"
                );
            }
            Self::WorkspaceListing { contains, excludes } => {
                ensure!(
                    !contains.is_empty() || !excludes.is_empty(),
                    "workspace_listing assertion needs contains or excludes"
                );
                for path in contains.iter().chain(excludes) {
                    normalize_workspace_listing_path(path)?;
                }
            }
            Self::Exit { .. } | Self::ProviderConsumed => {}
        }
        Ok(())
    }
}

pub(crate) fn validate_relative_path(path: &str) -> Result<()> {
    let path = Path::new(path);
    ensure!(!path.as_os_str().is_empty(), "path must not be empty");
    ensure!(
        !path.is_absolute(),
        "path must be relative: {}",
        path.display()
    );
    for component in path.components() {
        if matches!(
            component,
            Component::ParentDir | Component::RootDir | Component::Prefix(_)
        ) {
            bail!("path escapes the scenario root: {}", path.display());
        }
    }
    Ok(())
}

/// Normalize a scenario-owned listing path to the portable form emitted by
/// `capture_workspace_listing`. Validation happens before normalization so
/// `..`, roots, and platform prefixes can never be used to name external
/// entries.
pub(crate) fn normalize_workspace_listing_path(path: &str) -> Result<String> {
    validate_relative_path(path)?;
    let normalized = Path::new(path)
        .components()
        .filter_map(|component| match component {
            Component::Normal(segment) => Some(segment.to_string_lossy()),
            Component::CurDir => None,
            Component::ParentDir | Component::RootDir | Component::Prefix(_) => None,
        })
        .collect::<Vec<_>>()
        .join("/");
    ensure!(
        !normalized.is_empty(),
        "workspace listing path must name an entry: {path}"
    );
    Ok(normalized)
}

fn validate_json_pointer(pointer: &str) -> Result<()> {
    ensure!(
        pointer.is_empty() || pointer.starts_with('/'),
        "JSON pointer must be empty or start with '/': {pointer:?}"
    );
    Ok(())
}

fn validate_wait_timeout(timeout_ms: u64) -> Result<()> {
    ensure!(
        timeout_ms > 0 && timeout_ms <= MAX_SCENARIO_TIMEOUT_MS,
        "action timeout must be in 1..={MAX_SCENARIO_TIMEOUT_MS}"
    );
    Ok(())
}

fn validate_hi_args(args: &[String]) -> Result<()> {
    // Scenarios may tune behavior inside the interactive frontend, but they
    // must not replace routing, persistence, tracing, isolation, or the
    // frontend itself. Keep this validation fail-closed: a newly added `hi`
    // argument must be classified here before smoke scenarios can use it.
    const HARNESS_OWNED: &[&str] = &[
        "--profile",
        "--provider",
        "--base-url",
        "--mcp-url",
        "--api-key",
        "--model",
        "--fallback",
        "--session-file",
        "--continue",
        "--resume",
        "--no-save",
        "--sync",
        "--sync-session-id",
        "--attach",
        "--resume-local",
        "--input-token",
        "--subagent",
        "--events-jsonl",
        "--tui-events-jsonl",
        "--trace-full",
        "--trace-capture",
        "--config",
        "--show-config",
        "--workflow",
        "--list-sessions",
        "--loops-daemon",
        "--daemon",
        "--turn-deadline",
        "--plain",
        "--goal",
        "--benchmark-orchestration",
        "--best-of",
        "--judge",
        "--report",
        "--eval-input",
        "--eval-output",
        "--quiet",
        "--skeptic-review",
        "--review-target",
        "--keep-background",
        "--rsi-managed",
        "--rsi-trace-dir",
        "--rsi-max-bytes",
        "--api-unix-socket",
        "--rsi-context-json",
        "--rsi-runtime-descriptor",
    ];
    const VALUE_ARGUMENTS: &[&str] = &[
        "--max-tokens",
        "--temperature",
        "--top-p",
        "--output-token-parameter",
        "--thinking",
        "--reasoning-effort",
        "--tool-mode",
        "--compat",
        "--deepseek-compat",
        "--compaction",
        "--verify",
        "--max-verify-repairs",
        "--review",
        "--lsp",
        "--tool-set",
        "--max-steps",
        "--max-tool-calls",
    ];
    const SWITCH_ARGUMENTS: &[&str] = &[
        "--yes",
        "--durable",
        "--no-auto-compact",
        "--no-finalize",
        "--no-memory",
        "--confirm-edits",
        "--dry-run",
        "--no-verify",
        "--allow-unverified",
        "--skeptic-fail-open",
        "--allow-no-checkpoint",
        "--rsi",
        "--no-rsi",
        "--tasks",
        "--no-tasks",
        "--clippy",
    ];

    let mut index = 0;
    while index < args.len() {
        let argument = &args[index];
        let (flag, inline_value) = argument
            .split_once('=')
            .map_or((argument.as_str(), None), |(flag, value)| {
                (flag, Some(value))
            });

        let owned_short = ["-m", "-p", "-c", "-q"]
            .into_iter()
            .find(|short| argument.starts_with(short));
        if HARNESS_OWNED.contains(&flag) || owned_short.is_some() {
            bail!("scenario cannot override harness-owned argument {flag}");
        }
        if SWITCH_ARGUMENTS.contains(&flag) {
            ensure!(
                inline_value.is_none(),
                "switch argument {flag} does not accept a value"
            );
            index += 1;
            continue;
        }
        if VALUE_ARGUMENTS.contains(&flag) {
            if let Some(value) = inline_value {
                ensure!(!value.is_empty(), "argument {flag} requires a value");
                index += 1;
                continue;
            }
            let value = args
                .get(index + 1)
                .with_context(|| format!("argument {flag} requires a value"))?;
            ensure!(
                !value.starts_with('-'),
                "argument {flag} requires a value before {value}"
            );
            index += 2;
            continue;
        }
        if argument.starts_with('-') {
            bail!("unsupported scenario hi argument {flag}");
        }
        bail!(
            "scenario positional prompts are not allowed because they bypass the full interactive TUI: {argument:?}"
        );
    }
    Ok(())
}

fn validate_hi_env(env: &BTreeMap<String, String>) -> Result<()> {
    const RESERVED: &[&str] = &[
        "HOME",
        "PATH",
        "SHELL",
        "TERM",
        "COLORTERM",
        "TMPDIR",
        "TMP",
        "TEMP",
        "HI_MODEL",
        "HI_API_KEY",
        "HI_BASE_URL",
        "HI_PROVIDER",
        "HI_SKIP_TUTORIAL",
        "HI_SUGGEST_NEXT_PROMPT",
        "HI_DISABLE_UPDATE_CHECK",
        "OPENAI_API_KEY",
        "OPENROUTER_API_KEY",
        "ANTHROPIC_API_KEY",
        "PIPENETWORK_API_KEY",
        "PIPE_API_KEY",
        "OLLAMA_API_KEY",
        "XAI_API_KEY",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "NO_PROXY",
        "RUSTUP_HOME",
        "RUSTUP_TOOLCHAIN",
        "RUSTUP_SKIP_UPDATE_CHECK",
        "CARGO_NET_OFFLINE",
    ];
    for (name, value) in env {
        let normalized_name = name.to_ascii_uppercase();
        ensure!(
            !name.is_empty()
                && name
                    .bytes()
                    .all(|byte| byte == b'_' || byte.is_ascii_alphanumeric()),
            "invalid scenario environment variable name {name:?}"
        );
        ensure!(
            !RESERVED.contains(&normalized_name.as_str())
                && !normalized_name.starts_with("XDG_")
                && !normalized_name.starts_with("HI_"),
            "scenario cannot override harness-owned environment variable {name}"
        );
        ensure!(
            !value.contains('\0'),
            "scenario environment variable {name} contains NUL"
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base() -> Scenario {
        Scenario {
            schema_version: 1,
            name: "x".into(),
            tags: vec![],
            timeout_ms: 1_000,
            terminal: TerminalSpec::default(),
            workspace: WorkspaceSpec::default(),
            session: SessionSeed::default(),
            hi: HiSpec::default(),
            provider: ProviderSpec::default(),
            actions: vec![],
            assertions: vec![],
            source_dir: std::env::temp_dir(),
        }
    }

    #[test]
    fn rejects_escape_paths_and_owned_flags() {
        assert!(validate_relative_path("../oops").is_err());
        assert!(validate_relative_path("fixture/main.rs").is_ok());
        assert!(validate_hi_args(&["--provider=openai".into()]).is_err());
    }

    #[test]
    fn accepts_explicit_unlimited_child_turn_and_rejects_deadline_past_harness_bound() {
        let mut scenario = base();
        scenario.hi.turn_deadline_secs = Some(0);
        scenario.validate().unwrap();

        scenario.hi.turn_deadline_secs = Some(MAX_SCENARIO_TIMEOUT_MS / 1_000 + 1);
        assert!(scenario.validate().is_err());
    }

    #[test]
    fn validates_outer_turn_watchdog_against_scenario_timeout() {
        let mut scenario = base();
        scenario.hi.outer_turn_kill_secs = Some(1);
        scenario.validate().unwrap();

        scenario.hi.outer_turn_kill_secs = Some(0);
        assert!(scenario.validate().is_err());
        scenario.hi.outer_turn_kill_secs = Some(2);
        assert!(scenario.validate().is_err());
    }

    #[test]
    fn raw_sse_transport_options_are_backward_compatible_and_replay_stable() {
        let legacy: ProviderResponse = toml::from_str(
            r#"
kind = "raw_sse"
body = "data: [DONE]\n\n"
"#,
        )
        .unwrap();
        let ProviderResponse::RawSse {
            body,
            gate,
            delay_ms,
            chunk_bytes,
            terminal,
        } = legacy
        else {
            unreachable!()
        };
        assert_eq!(body, "data: [DONE]\n\n");
        assert!(gate.is_none());
        assert_eq!(delay_ms, 0);
        assert_eq!(chunk_bytes, None);
        assert!(matches!(terminal, StreamTerminal::Done));

        let configured: ProviderResponse = toml::from_str(
            r#"
kind = "raw_sse"
body = "data: first\n\ndata: second\n\n"
delay_ms = 17
chunk_bytes = 3
terminal = "reset"
"#,
        )
        .unwrap();
        let replay = toml::to_string_pretty(&configured).unwrap();
        let replayed: ProviderResponse = toml::from_str(&replay).unwrap();
        let ProviderResponse::RawSse {
            body,
            delay_ms,
            chunk_bytes,
            terminal,
            ..
        } = replayed
        else {
            unreachable!()
        };
        assert_eq!(body, "data: first\n\ndata: second\n\n");
        assert_eq!(delay_ms, 17);
        assert_eq!(chunk_bytes, Some(3));
        assert!(matches!(terminal, StreamTerminal::Reset));

        assert!(
            toml::from_str::<ProviderResponse>(
                r#"
kind = "raw_sse"
body = "data: [DONE]\n\n"
frame_delay_ms = 1
"#,
            )
            .is_err()
        );
    }

    #[test]
    fn rejects_zero_raw_sse_chunk_size() {
        let mut scenario = base();
        scenario.provider.steps.push(ProviderStep {
            id: "raw".into(),
            expect: RequestExpectation::default(),
            response: ProviderResponse::RawSse {
                body: "data: [DONE]\n\n".into(),
                gate: None,
                delay_ms: 0,
                chunk_bytes: Some(0),
                terminal: StreamTerminal::Done,
            },
        });

        let error = scenario.validate().unwrap_err().to_string();
        assert!(error.contains("zero chunk_bytes"), "{error}");
    }

    #[test]
    fn rejects_every_owned_argument_spelling_and_short_attached_value() {
        for arguments in [
            vec!["--profile=wrong"],
            vec!["--provider=openai"],
            vec!["--base-url=http://elsewhere.invalid"],
            vec!["--mcp-url=http://elsewhere.invalid"],
            vec!["--api-key=wrong"],
            vec!["--model=wrong"],
            vec!["--fallback=wrong"],
            vec!["--session-file=elsewhere"],
            vec!["--continue"],
            vec!["--resume=wrong"],
            vec!["--no-save"],
            vec!["--sync"],
            vec!["--sync-session-id=wrong"],
            vec!["--attach=wrong"],
            vec!["--resume-local"],
            vec!["--input-token=wrong"],
            vec!["--subagent"],
            vec!["--events-jsonl=elsewhere"],
            vec!["--tui-events-jsonl=elsewhere"],
            vec!["--trace-full"],
            vec!["--trace-capture=full"],
            vec!["--config=elsewhere"],
            vec!["--show-config"],
            vec!["--workflow=wrong"],
            vec!["--list-sessions"],
            vec!["--loops-daemon"],
            vec!["--daemon"],
            vec!["--turn-deadline=1"],
            vec!["--plain"],
            vec!["--goal=wrong"],
            vec!["--benchmark-orchestration"],
            vec!["--best-of=2"],
            vec!["--judge=model"],
            vec!["--report=elsewhere"],
            vec!["--eval-input=elsewhere"],
            vec!["--eval-output=workspace"],
            vec!["--quiet"],
            vec!["--skeptic-review"],
            vec!["--review-target=elsewhere"],
            vec!["--keep-background"],
            vec!["--rsi-managed"],
            vec!["--rsi-trace-dir=elsewhere"],
            vec!["--rsi-max-bytes=1"],
            vec!["--api-unix-socket=elsewhere"],
            vec!["--rsi-context-json=elsewhere"],
            vec!["--rsi-runtime-descriptor=elsewhere"],
            vec!["--model", "wrong"],
            vec!["-m", "wrong"],
            vec!["-mwrong"],
            vec!["-m=wrong"],
            vec!["-p", "wrong"],
            vec!["-pwrong"],
            vec!["-c"],
            vec!["-q"],
            vec!["--"],
        ] {
            let arguments = arguments.into_iter().map(str::to_owned).collect::<Vec<_>>();
            assert!(
                validate_hi_args(&arguments).is_err(),
                "owned spelling was accepted: {arguments:?}"
            );
        }
    }

    #[test]
    fn accepts_only_classified_interactive_tuning_arguments() {
        assert!(
            validate_hi_args(&[
                "--yes".into(),
                "--allow-unverified".into(),
                "--max-tokens=128".into(),
                "--review".into(),
                "off".into(),
                "--verify=cargo test".into(),
                "--max-steps".into(),
                "2".into(),
            ])
            .is_ok()
        );
    }

    #[test]
    fn rejects_positional_prompts_unknown_arguments_and_malformed_tuning_arguments() {
        for arguments in [
            vec!["fix", "the", "bug"],
            vec!["--allow-unverified", "fix the bug"],
            vec!["--max-tokens", "128", "fix the bug"],
            vec!["--unknown-smoke-escape"],
            vec!["--max-tokens"],
            vec!["--max-tokens="],
            vec!["--max-tokens", "--plain"],
            vec!["--allow-unverified=true"],
        ] {
            let arguments = arguments.into_iter().map(str::to_owned).collect::<Vec<_>>();
            assert!(
                validate_hi_args(&arguments).is_err(),
                "non-interactive or unclassified argument was accepted: {arguments:?}"
            );
        }
    }

    #[test]
    fn rejects_all_hi_environment_overrides() {
        for name in [
            "HI_SANDBOX",
            "HI_SANDBOXED",
            "HI_PIPE_WRAP",
            "HI_STATE_ROOT",
            "HI_TRUST_STORE",
            "HI_ME_MD",
            "HI_TRACE_CAPTURE",
            "HI_SMOKE_RUN_MARKER",
            "HI_DISABLE_FEEDBACK",
        ] {
            assert!(
                validate_hi_env(&BTreeMap::from([(name.into(), "unsafe".into())])).is_err(),
                "{name} must remain harness-owned"
            );
        }
        assert!(
            validate_hi_env(&BTreeMap::from([("SCENARIO_MARKER".into(), "ok".into())])).is_ok()
        );
    }

    #[test]
    fn rejects_provider_credentials_and_proxy_route_overrides() {
        for name in [
            "OPENAI_API_KEY",
            "OPENROUTER_API_KEY",
            "ANTHROPIC_API_KEY",
            "PIPENETWORK_API_KEY",
            "PIPE_API_KEY",
            "OLLAMA_API_KEY",
            "XAI_API_KEY",
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "ALL_PROXY",
            "NO_PROXY",
            // Libraries commonly honor lowercase proxy aliases too.
            "https_proxy",
        ] {
            assert!(
                validate_hi_env(&BTreeMap::from([(name.into(), "unsafe".into())])).is_err(),
                "{name} must remain harness-owned"
            );
        }
        assert!(
            validate_hi_env(&BTreeMap::from([("SCENARIO_MARKER".into(), "ok".into())])).is_ok()
        );
    }

    #[cfg(unix)]
    #[test]
    fn rejects_fixture_root_symlinks() {
        use std::os::unix::fs::symlink;

        let root = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        symlink(outside.path(), root.path().join("fixture-link")).unwrap();
        let mut scenario = base();
        scenario.source_dir = root.path().to_path_buf();
        scenario.workspace.fixture = Some("fixture-link".into());

        let error = scenario.validate().unwrap_err().to_string();
        assert!(error.contains("real directory"), "{error}");
    }

    #[test]
    fn rejects_ambiguous_record_counts() {
        let mut scenario = base();
        scenario.assertions.push(Assertion::Records {
            source: RecordSource::Events,
            equals: BTreeMap::new(),
            contains: BTreeMap::new(),
            exact: Some(1),
            at_least: Some(1),
            at_most: None,
        });
        assert!(scenario.validate().is_err());
    }

    #[test]
    fn accepts_bounded_record_count_range_and_rejects_an_inverted_range() {
        let mut scenario = base();
        scenario.assertions.push(Assertion::Records {
            source: RecordSource::Events,
            equals: BTreeMap::new(),
            contains: BTreeMap::new(),
            exact: None,
            at_least: Some(1),
            at_most: Some(2),
        });
        scenario.validate().unwrap();

        let Assertion::Records {
            at_least, at_most, ..
        } = scenario.assertions.last_mut().unwrap()
        else {
            unreachable!()
        };
        *at_least = Some(3);
        *at_most = Some(2);
        assert!(scenario.validate().is_err());
    }

    #[test]
    fn validates_aggregate_record_assertions() {
        let mut invalid_substring = base();
        invalid_substring
            .assertions
            .push(Assertion::SubstringOccurrences {
                source: RecordSource::Session,
                equals: BTreeMap::new(),
                contains: BTreeMap::new(),
                pointer: "/content/0/Text".into(),
                substring: String::new(),
                exact: 1,
            });
        assert!(invalid_substring.validate().is_err());

        let mut invalid_all = base();
        invalid_all.assertions.push(Assertion::AllRecords {
            source: RecordSource::ProviderRequests,
            where_equals: BTreeMap::new(),
            where_contains: BTreeMap::new(),
            equals: BTreeMap::new(),
            contains: BTreeMap::new(),
            at_least: 1,
        });
        assert!(invalid_all.validate().is_err());

        let unknown = r#"
kind = "substring_occurrences"
source = "session"
pointer = "/content/0/Text"
substring = "needle"
exact = 1
unexpected = true
"#;
        assert!(toml::from_str::<Assertion>(unknown).is_err());
    }

    #[test]
    fn validates_workspace_artifact_assertions_and_contains_listing_paths() {
        let mut scenario = base();
        scenario.assertions.extend([
            Assertion::WorkspacePatch {
                contains: vec!["diff --git a/src/main.rs b/src/main.rs".into()],
                excludes: vec!["outside-secret".into()],
                equals: None,
            },
            Assertion::WorkspaceListing {
                contains: vec!["./src//main.rs".into()],
                excludes: vec!["target/debug/app".into()],
            },
        ]);
        scenario.validate().unwrap();
        assert_eq!(
            normalize_workspace_listing_path("./src//main.rs").unwrap(),
            "src/main.rs"
        );

        let Assertion::WorkspaceListing { contains, .. } = scenario.assertions.last_mut().unwrap()
        else {
            unreachable!()
        };
        contains[0] = "../outside".into();
        let error = scenario.validate().unwrap_err().to_string();
        assert!(error.contains("escapes the scenario root"), "{error}");

        let unknown = r#"
kind = "workspace_patch"
contains = ["needle"]
unexpected = true
"#;
        assert!(toml::from_str::<Assertion>(unknown).is_err());
    }

    #[test]
    fn rejects_vacuous_workspace_artifact_assertions() {
        let mut scenario = base();
        scenario.assertions.push(Assertion::WorkspacePatch {
            contains: Vec::new(),
            excludes: Vec::new(),
            equals: None,
        });
        assert!(scenario.validate().is_err());

        scenario.assertions = vec![Assertion::WorkspaceListing {
            contains: Vec::new(),
            excludes: Vec::new(),
        }];
        assert!(scenario.validate().is_err());
    }

    #[test]
    fn rejects_vacuous_absence_and_process_waits() {
        let mut absent = base();
        absent.actions.push(Action::WaitEventAbsent {
            equals: BTreeMap::new(),
            contains: BTreeMap::new(),
            duration_ms: 10,
        });
        assert!(absent.validate().is_err());

        let mut process = base();
        process.actions.push(Action::WaitProcess {
            command_contains: String::new(),
            at_least: 1,
            timeout_ms: 10,
        });
        assert!(process.validate().is_err());
    }
}
