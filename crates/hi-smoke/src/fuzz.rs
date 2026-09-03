use std::collections::{BTreeMap, VecDeque};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use anyhow::{Result, bail, ensure};
use serde_json::json;

use crate::cli::RunMode;
use crate::discovery;
use crate::runner::{CaseFailureKind, CaseOptions, CaseStatus, SandboxRequirement};
use crate::scenario::{
    Action, Assertion, Key, ProviderResponse, ProviderStep, QuiescentSource, RecordSource,
    RequestExpectation, Scenario, StreamTerminal,
};

const FAULT_CAMPAIGN_COUNT: u64 = 12;
const STATE_TEMPLATE_NAMES: [&str; 6] = [
    "plan-approval-approve",
    "plan-approval-request-changes",
    "plan-approval-park-restart",
    "held-request-cancel-resume",
    "session-restart-restores-plan-drive",
    "long-tool-cancellation",
];
const CAMPAIGN_COUNT: u64 = FAULT_CAMPAIGN_COUNT + STATE_TEMPLATE_NAMES.len() as u64;

#[derive(Clone, Debug)]
pub(crate) struct FuzzOptions {
    pub hi_bin: PathBuf,
    pub suite: PathBuf,
    pub seed_start: u64,
    pub seeds: u64,
    pub jobs: usize,
    pub artifacts: PathBuf,
    pub keep: bool,
}

pub(crate) fn run(options: FuzzOptions) -> Result<()> {
    let discovered = discovery::discover(&options.suite)?;
    let templates = discovered
        .iter()
        .filter(|candidate| {
            candidate
                .scenario
                .tags
                .iter()
                .any(|tag| tag == "fuzz_template")
        })
        .collect::<Vec<_>>();
    ensure!(
        templates.len() == 1,
        "suite needs exactly one scenario tagged `fuzz_template`; found {}",
        templates.len()
    );
    let template = templates
        .into_iter()
        .next()
        .expect("one template")
        .scenario
        .clone();
    let state_templates = STATE_TEMPLATE_NAMES
        .into_iter()
        .map(|name| {
            discovered
                .iter()
                .find(|candidate| candidate.scenario.name == name)
                .map(|candidate| candidate.scenario.clone())
                .ok_or_else(|| anyhow::anyhow!("suite is missing fuzz state template {name:?}"))
        })
        .collect::<Result<Vec<_>>>()?;
    std::fs::create_dir_all(&options.artifacts)?;
    let seeds = (options.seed_start..options.seed_start.saturating_add(options.seeds))
        .collect::<VecDeque<_>>();
    ensure!(seeds.len() as u64 == options.seeds, "seed range overflowed");
    let queue = Arc::new(Mutex::new(seeds));
    let reports = Arc::new(Mutex::new(Vec::new()));
    let workers = options.jobs.min(options.seeds as usize);

    std::thread::scope(|scope| {
        for _ in 0..workers {
            let queue = Arc::clone(&queue);
            let reports = Arc::clone(&reports);
            let template = template.clone();
            let state_templates = state_templates.clone();
            let case_options = CaseOptions {
                hi_bin: options.hi_bin.clone(),
                artifacts: options.artifacts.clone(),
                mode: RunMode::Scripted,
                live_route: None,
                keep: options.keep,
                seed: None,
                sandbox_requirement: SandboxRequirement::Enforced,
            };
            scope.spawn(move || {
                loop {
                    let seed = queue.lock().expect("fuzz seed lock").pop_front();
                    let Some(seed) = seed else { break };
                    let scenario = generate(&template, &state_templates, seed);
                    eprintln!("[ FUZZ     ] seed {seed} ({})", scenario.name);
                    let mut options = case_options.clone();
                    options.seed = Some(seed);
                    let report = crate::runner::run_scenario(scenario, &options);
                    reports.lock().expect("fuzz report lock").push(report);
                }
            });
        }
    });

    let mut reports = Arc::try_unwrap(reports)
        .map_err(|_| anyhow::anyhow!("fuzz workers retained report state"))?
        .into_inner()
        .map_err(|_| anyhow::anyhow!("fuzz report lock was poisoned"))?;
    reports.sort_by(|left, right| left.name.cmp(&right.name));
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
    let summary = json!({
        "schema_version": 1,
        "generator": "hi-smoke-state-aware-v1",
        "seed_start": options.seed_start,
        "seeds": options.seeds,
        "passed": passed,
        "failed": total - passed,
        "scenario_pass_rate": passed as f64 * 100.0 / total as f64,
        "crash_count": crash_count,
        "infrastructure_loop_count": infrastructure_loop_count,
        "infrastructure_failure_count": infrastructure_failure_count,
        "cases": reports,
    });
    crate::artifacts::write_suite_summary(
        &options.artifacts,
        &summary,
        &["hi-smoke-test-key".to_owned()],
    )?;
    if passed != total {
        bail!(
            "{} of {total} fuzz seed(s) failed; every failure has an exact replay.toml",
            total - passed
        );
    }
    println!("{passed} deterministic fuzz seed(s) passed");
    Ok(())
}

fn generate(template: &Scenario, state_templates: &[Scenario], seed: u64) -> Scenario {
    let campaign = seed % CAMPAIGN_COUNT;
    if campaign >= FAULT_CAMPAIGN_COUNT {
        return generate_state_scenario(state_templates, seed, campaign);
    }
    let mut scenario = template.clone();
    let mut rng = Generator::new(seed);
    // Contiguous CI seed ranges must cover every fault family before they
    // repeat; pseudo-randomize parameters within the chosen family only.
    scenario.name = format!("fuzz-{}-{seed}", campaign_name(campaign));
    scenario.tags = vec!["generated".into(), "fuzz".into()];
    scenario.timeout_ms = 45_000;
    scenario.provider.steps.clear();
    scenario.actions.clear();
    scenario.assertions.clear();

    // Keep the prompt answer-shaped. Phrases such as "do not modify files" are
    // intentionally classified as review work by `hi`, which adds a read-only
    // preflight and makes a provider-stream campaign exercise the wrong state
    // machine.
    let token = format!("SMOKE-{seed}");
    let prompt =
        format!("Return the exact token {token} and one short sentence confirming receipt.");
    let mut steps = Vec::new();
    let mut actions = vec![Action::SendLine {
        text: prompt.clone(),
    }];
    let expectation = RequestExpectation {
        body_contains: vec![token.clone()],
        body_excludes: vec![],
        json_equals: BTreeMap::new(),
    };

    let mut recovered_after_fault = false;
    let mut settlement_already_waited = false;
    let response = match campaign {
        0 => ProviderResponse::Text {
            text: format!("{token}: the fragmented provider stream arrived intact."),
            gate: None,
            delay_ms: 0,
            chunk_bytes: Some(1 + rng.range(8) as usize),
            terminal: StreamTerminal::Done,
        },
        1 => ProviderResponse::Text {
            text: format!("{token}: the delayed provider stream arrived intact."),
            gate: None,
            delay_ms: 5 + rng.range(80),
            chunk_bytes: Some(1 + rng.range(5) as usize),
            terminal: StreamTerminal::Done,
        },
        2 => {
            recovered_after_fault = true;
            ProviderResponse::RawSse {
                body: "data: {definitely-not-json}\n\ndata: [DONE]\n\n".into(),
                gate: None,
                delay_ms: 0,
                chunk_bytes: None,
                terminal: StreamTerminal::Done,
            }
        }
        3 => {
            recovered_after_fault = true;
            ProviderResponse::RawSse {
                body: "data: [DONE]\n\n".into(),
                gate: None,
                delay_ms: 0,
                chunk_bytes: None,
                terminal: StreamTerminal::Done,
            }
        }
        4 => {
            recovered_after_fault = true;
            ProviderResponse::HttpError {
                status: 503,
                body: r#"{"error":{"message":"seeded unavailable"}}"#.into(),
                gate: None,
            }
        }
        5 => {
            recovered_after_fault = true;
            ProviderResponse::HttpError {
                status: 429,
                body: r#"{"error":{"message":"seeded rate limit"}}"#.into(),
                gate: None,
            }
        }
        6 => {
            recovered_after_fault = true;
            ProviderResponse::Reset { gate: None }
        }
        7 => {
            // EOF after a valid content delta is a truncated-but-usable model
            // response, not a retryable transport failure. Requiring a second
            // provider step here would make a successful bounded settlement
            // fail only because the strict script remained unconsumed.
            ProviderResponse::RawSse {
                body: "data: {\"choices\":[{\"delta\":{\"content\":\"partial\"}}]}\n\n".into(),
                gate: None,
                delay_ms: 0,
                chunk_bytes: None,
                terminal: StreamTerminal::Eof,
            }
        }
        8 => {
            let gate = format!("held-{seed}");
            actions.extend([
                Action::WaitProviderRequest {
                    count: 1,
                    timeout_ms: 5_000,
                },
                Action::SendKey { key: Key::CtrlC },
                Action::WaitEvent {
                    equals: BTreeMap::from([
                        ("/event".into(), json!("turn_settled")),
                        ("/data/outcome/status".into(), json!("cancelled")),
                    ]),
                    contains: BTreeMap::new(),
                    timeout_ms: 15_000,
                },
                Action::ReleaseGate { gate: gate.clone() },
            ]);
            settlement_already_waited = true;
            ProviderResponse::Hold { gate: Some(gate) }
        }
        9 => {
            let gate = format!("queue-{seed}");
            let queued_token = format!("QUEUED-{seed}");
            actions.extend([
                Action::WaitProviderRequest {
                    count: 1,
                    timeout_ms: 5_000,
                },
                Action::SendLine {
                    text: format!(
                        "Also include the exact token {queued_token} in this same reply."
                    ),
                },
                Action::ReleaseGate { gate: gate.clone() },
                Action::WaitProviderRequest {
                    count: 2,
                    timeout_ms: 5_000,
                },
            ]);
            steps.push(ProviderStep {
                id: "queued-follow-up".into(),
                expect: RequestExpectation {
                    body_contains: vec![queued_token.clone()],
                    ..RequestExpectation::default()
                },
                response: ProviderResponse::Text {
                    text: format!(
                        "{token} {queued_token}: the queued input was incorporated exactly once."
                    ),
                    gate: None,
                    delay_ms: 0,
                    chunk_bytes: None,
                    terminal: StreamTerminal::Done,
                },
            });
            ProviderResponse::Text {
                text: format!(
                    "{token}: the initial response arrived before queued input was incorporated."
                ),
                gate: Some(gate),
                delay_ms: 0,
                chunk_bytes: None,
                terminal: StreamTerminal::Done,
            }
        }
        10 => {
            actions.push(Action::Resize {
                cols: 80 + rng.range(100) as u16,
                rows: 24 + rng.range(40) as u16,
            });
            ProviderResponse::Text {
                text: format!("{token}: the resized terminal remained responsive."),
                gate: None,
                delay_ms: 20,
                chunk_bytes: None,
                terminal: StreamTerminal::Done,
            }
        }
        _ => {
            let gate = format!("cancel-{seed}");
            actions.extend([
                Action::WaitProviderRequest {
                    count: 1,
                    timeout_ms: 5_000,
                },
                Action::SendKey { key: Key::CtrlC },
                Action::WaitEvent {
                    equals: BTreeMap::from([
                        ("/event".into(), json!("turn_settled")),
                        ("/data/outcome/status".into(), json!("cancelled")),
                    ]),
                    contains: BTreeMap::new(),
                    timeout_ms: 15_000,
                },
                Action::ReleaseGate { gate: gate.clone() },
            ]);
            settlement_already_waited = true;
            ProviderResponse::Text {
                text: format!(
                    "{token}: this gated response must remain hidden after cancellation."
                ),
                gate: Some(gate),
                delay_ms: 0,
                chunk_bytes: Some(2),
                terminal: StreamTerminal::Done,
            }
        }
    };
    steps.insert(
        0,
        ProviderStep {
            id: format!("seed-{seed}-primary"),
            expect: expectation,
            response,
        },
    );
    if recovered_after_fault {
        steps.push(ProviderStep {
            id: format!("seed-{seed}-recovery"),
            expect: RequestExpectation {
                body_contains: vec![token.clone()],
                ..RequestExpectation::default()
            },
            response: ProviderResponse::Text {
                text: format!(
                    "{token}: the bounded provider retry recovered and produced a concrete answer."
                ),
                gate: None,
                delay_ms: 0,
                chunk_bytes: None,
                terminal: StreamTerminal::Done,
            },
        });
    }
    scenario.provider.steps = steps;
    if !actions
        .iter()
        .any(|action| matches!(action, Action::WaitProviderRequest { .. }))
    {
        actions.push(Action::WaitProviderRequest {
            count: 1,
            timeout_ms: 5_000,
        });
    }
    if !settlement_already_waited {
        actions.push(Action::WaitEvent {
            equals: BTreeMap::from([("/event".into(), json!("turn_settled"))]),
            contains: BTreeMap::new(),
            timeout_ms: 15_000,
        });
    }
    actions.extend([
        Action::WaitQuiescent {
            source: QuiescentSource::Events,
            quiet_ms: 150,
            timeout_ms: 3_000,
        },
        Action::CaptureScreen {
            name: "final".into(),
        },
        Action::Quit,
    ]);
    scenario.actions = actions;
    let (expected_status, expected_settlements) = match campaign {
        8 | 11 => ("cancelled", 1),
        9 => ("completed", 2),
        _ => ("completed", 1),
    };
    scenario.assertions = vec![
        Assertion::Records {
            source: RecordSource::Events,
            equals: BTreeMap::from([("/event".into(), json!("ready"))]),
            contains: BTreeMap::new(),
            exact: Some(1),
            at_least: None,
            at_most: None,
        },
        Assertion::Records {
            source: RecordSource::Events,
            equals: BTreeMap::from([
                ("/event".into(), json!("turn_settled")),
                ("/data/outcome/status".into(), json!(expected_status)),
            ]),
            contains: BTreeMap::new(),
            exact: Some(expected_settlements),
            at_least: None,
            at_most: None,
        },
        Assertion::Exit { code: 0 },
        Assertion::ProviderConsumed,
    ];
    if campaign == 9 {
        scenario.assertions.push(Assertion::Records {
            source: RecordSource::ProviderRequests,
            equals: BTreeMap::new(),
            contains: BTreeMap::from([("/body".into(), format!("QUEUED-{seed}"))]),
            exact: Some(1),
            at_least: None,
            at_most: None,
        });
    }
    scenario
}

fn generate_state_scenario(state_templates: &[Scenario], seed: u64, campaign: u64) -> Scenario {
    let index = (campaign - FAULT_CAMPAIGN_COUNT) as usize;
    let mut scenario = state_templates
        .get(index)
        .expect("validated state template index")
        .clone();
    let mut rng = Generator::new(seed);
    scenario.name = format!("fuzz-{}-{seed}", campaign_name(campaign));
    scenario.tags = vec!["generated".into(), "fuzz".into(), "state_aware".into()];
    scenario.timeout_ms = scenario.timeout_ms.max(45_000);
    scenario.terminal.cols = 90 + rng.range(70) as u16;
    scenario.terminal.rows = 28 + rng.range(25) as u16;
    scenario.actions.insert(
        0,
        Action::Resize {
            cols: 80 + rng.range(80) as u16,
            rows: 24 + rng.range(30) as u16,
        },
    );
    for step in &mut scenario.provider.steps {
        if let ProviderResponse::Text {
            delay_ms,
            chunk_bytes,
            ..
        } = &mut step.response
        {
            *delay_ms = rng.range(25);
            *chunk_bytes = Some(1 + rng.range(12) as usize);
        }
    }
    scenario
}

fn campaign_name(index: u64) -> &'static str {
    match index {
        0 => "fragmented",
        1 => "delayed",
        2 => "malformed-sse",
        3 => "empty",
        4 => "http-503",
        5 => "http-429",
        6 => "reset",
        7 => "eof",
        8 => "held-cancel",
        9 => "queued-input",
        10 => "resize",
        11 => "settlement-cancel",
        12 => "approval-approve",
        13 => "approval-request-changes",
        14 => "approval-park-restart",
        15 => "plan-cancel-resume",
        16 => "session-restart",
        _ => "tool-settlement-cancel",
    }
}

struct Generator(u64);

impl Generator {
    fn new(seed: u64) -> Self {
        Self(seed ^ 0x9e37_79b9_7f4a_7c15)
    }

    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0 = self.0.wrapping_mul(0x2545_f491_4f6c_dd1d);
        self.0
    }

    fn range(&mut self, upper: u64) -> u64 {
        self.next() % upper
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scenario::{HiSpec, ProviderSpec, SessionSeed, TerminalSpec, WorkspaceSpec};

    fn template() -> Scenario {
        Scenario {
            schema_version: 1,
            name: "template".into(),
            tags: vec!["fuzz_template".into()],
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
    fn seed_generation_is_stable_and_replayable() {
        let states = state_templates();
        let first = generate(&template(), &states, 42);
        let second = generate(&template(), &states, 42);
        assert_eq!(
            toml::to_string(&first).unwrap(),
            toml::to_string(&second).unwrap()
        );
        assert!(first.name.ends_with("-42"));
    }

    #[test]
    fn campaign_has_all_fault_families() {
        let names = (0..CAMPAIGN_COUNT).map(campaign_name).collect::<Vec<_>>();
        assert!(names.contains(&"reset"));
        assert!(names.contains(&"held-cancel"));
        assert!(names.contains(&"queued-input"));
        assert!(names.contains(&"approval-park-restart"));
        assert!(names.contains(&"plan-cancel-resume"));
        assert!(names.contains(&"tool-settlement-cancel"));
    }

    #[test]
    fn fault_campaigns_assert_their_typed_terminal_status() {
        let states = state_templates();
        for seed in 0..FAULT_CAMPAIGN_COUNT {
            let scenario = generate(&template(), &states, seed);
            let (expected, exact) = match seed {
                8 | 11 => ("cancelled", 1),
                9 => ("completed", 2),
                _ => ("completed", 1),
            };
            assert!(
                scenario.assertions.iter().any(|assertion| {
                    matches!(
                        assertion,
                        Assertion::Records {
                            source: RecordSource::Events,
                            equals,
                            exact: Some(actual_exact),
                            ..
                        } if equals.get("/event") == Some(&json!("turn_settled"))
                            && equals.get("/data/outcome/status") == Some(&json!(expected))
                            && *actual_exact == exact
                    )
                }),
                "seed {seed} did not assert exactly {exact} terminal status {expected}"
            );
        }
    }

    fn state_templates() -> Vec<Scenario> {
        STATE_TEMPLATE_NAMES
            .into_iter()
            .map(|name| {
                let mut scenario = template();
                scenario.name = name.into();
                scenario
            })
            .collect()
    }
}
