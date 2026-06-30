use super::*;
use super::common::*;

#[tokio::test]
async fn usage_line_separates_cumulative_spend_from_context_fill() {
    // The regression guard: with a window + price set, the done line shows
    // cumulative ↑/↓ session spend (abbreviated, matching the live line), the
    // cost, and a context gauge that is the *last request's* size — distinct
    // from cumulative input and humanized the same way. Pins against mixing
    // raw/abbreviated units, rendering a count two ways, or conflating the two.
    let mut cfg = config();
    cfg.context_window = Some(1_000_000);
    cfg.price = Some((5.0, 15.0)); // $/1M (in, out)
    let responses = vec![
        completion(
            vec![Content::ToolCall {
                id: "1".into(),
                name: "bash".into(),
                arguments: "{\"command\":\"echo hi\"}".into(),
            }],
            8_000,
            100,
        ),
        completion(vec![Content::Text("done".into())], 12_000, 200),
    ];
    let mut agent = agent(responses, cfg);
    let mut ui = RecUi::default();
    agent.run_turn("go", &mut ui).await.unwrap();
    let line = ui.turn_end.expect("turn_end emitted");

    // Cumulative session spend, arrowed + abbreviated (same shape as the live line).
    assert!(line.contains("↑20k"), "cumulative input ↑ (8k+12k): {line}");
    assert!(
        line.contains("↓300"),
        "cumulative output ↓ (100+200): {line}"
    );
    // The context gauge is the LAST request (12k) over the window — NOT the
    // cumulative input (20k), and abbreviated, not raw.
    assert!(
        line.contains("ctx 1% (12k/1.0M)"),
        "point-in-time context: {line}"
    );
    // The old, mixed-unit, misleading format is gone.
    assert!(
        !line.contains(" in ·") && !line.contains("total"),
        "no raw in/out/total wording: {line}"
    );
    assert!(
        !line.contains("20000") && !line.contains("12000"),
        "no raw token counts: {line}"
    );
    // A clean turn (one tool call, no verify/retry/nudge) shows no steer
    // suffix — the trajectory surface is additive, only for noisy turns.
    assert!(
        !line.contains("steer"),
        "clean turn has no steer suffix: {line}"
    );
}

#[test]
fn turn_steer_summarizes_trajectory() {
    // Clean turn → None.
    let mut a = agent(vec![], config());
    assert_eq!(a.turn_steer(), None);

    // Noisy turn → a steer line listing each non-zero component.
    a.last_turn_telemetry = TurnTelemetry {
        verify_rounds: 2,
        recovery_retries: 1,
        repeat_nudges: 0,
        continue_nudges: 0,
        truncation_retries: 0,
        hit_step_cap: false,
        stalled_unfinished: false,
        stalled_repeating: false,
        verify_attributions: Vec::new(),
        tool_calls: 0,
        max_concurrent_batch: 0,
        serial_runs: 0,
        tool_timeline: Vec::new(),
        ..TurnTelemetry::default()
    };
    let steer = a.turn_steer().expect("noisy turn has a steer line");
    assert!(
        steer.contains("2 verify") && steer.contains("1 retry"),
        "lists non-zero components: {steer}"
    );
    assert!(
        !steer.contains("repeat") && !steer.contains("continue"),
        "omits zero components: {steer}"
    );

    // A stall is surfaced even with no rounds.
    a.last_turn_telemetry = TurnTelemetry {
        verify_rounds: 0,
        recovery_retries: 0,
        repeat_nudges: 0,
        continue_nudges: 0,
        truncation_retries: 0,
        hit_step_cap: false,
        stalled_unfinished: true,
        stalled_repeating: false,
        verify_attributions: Vec::new(),
        tool_calls: 0,
        max_concurrent_batch: 0,
        serial_runs: 0,
        tool_timeline: Vec::new(),
        ..TurnTelemetry::default()
    };
    let steer = a.turn_steer().expect("stall has a steer line");
    assert!(steer.contains("stalled"), "stall flagged: {steer}");
}

#[tokio::test]
async fn cost_accumulates_at_price_active_for_each_call() {
    let mut cfg = config();
    cfg.price = Some((1.0, 10.0));
    let responses = vec![
        completion(vec![Content::Text("first".into())], 1_000, 100),
        completion(vec![Content::Text("second".into())], 1_000, 100),
    ];
    let mut agent = agent(responses, cfg);

    agent.run_turn("first", &mut NullUi).await.unwrap();
    agent.set_model("m2".into(), Some((2.0, 20.0)), None);
    agent.run_turn("second", &mut NullUi).await.unwrap();

    assert_eq!(agent.cost_usd(), Some(0.006));
}

#[test]
fn add_usage_uses_normalized_billable_across_provider_semantics() {
    // A session that switches providers mid-run must accrue cost coherently.
    // The `billable` breakdown is provider-computed, so the agent's cost
    // math doesn't have to know whether `input_tokens` includes cached
    // tokens (OpenAI) or excludes them (Anthropic). Pin: an OpenAI-style
    // usage where input_tokens already includes the cached subset must NOT
    // double-count the cached tokens, and an Anthropic-style usage where
    // input excludes cache must still bill the cache portion at a discount.
    let mut cfg = config();
    cfg.price = Some((1.0, 10.0)); // $/1M in, out
    let mut a = agent(vec![], cfg);

    // OpenAI-style: prompt_tokens=1000 includes 400 cached. The normalized
    // breakdown separates them: 600 regular + 400 cached. Cost must bill
    // 600 at full price + 400 at 0.5x — NOT 1000 + 400 (double-count).
    a.add_usage(Usage {
        input_tokens: 1000,
        output_tokens: 0,
        cache_read_tokens: 400,
        cache_creation_tokens: 0,
        input_includes_cache: true,
        context_occupancy: 1000,
        billable: Some(hi_ai::BillableBreakdown {
            regular_input: 600,
            cached_input: 400,
            cache_creation: 0,
            output: 0,
        }),
    });
    let openai_cost = a.cost_usd().unwrap();
    // 600*1 + 400*0.5 = 800 token-units -> $0.0008
    assert!(
        (openai_cost - 0.0008).abs() < 1e-9,
        "openai no double-count: {openai_cost}"
    );

    // Anthropic-style: input_tokens=600 excludes 400 cache_read + 100
    // cache_creation. The breakdown bills 600 regular + 400 at 0.5x + 100
    // at 1.25x. The agent must NOT re-derive (which would wrongly subtract
    // cache_read from input_tokens).
    a.add_usage(Usage {
        input_tokens: 600,
        output_tokens: 50,
        cache_read_tokens: 400,
        cache_creation_tokens: 100,
        input_includes_cache: false,
        context_occupancy: 1100,
        billable: Some(hi_ai::BillableBreakdown {
            regular_input: 600,
            cached_input: 400,
            cache_creation: 100,
            output: 50,
        }),
    });
    let total = a.cost_usd().unwrap();
    // anthropic increment: 600*1 + 400*0.5 + 100*1.25 + 50*10 = 600+200+125+500 = 1425 -> $0.001425
    assert!(
        (total - (0.0008 + 0.001425)).abs() < 1e-9,
        "coherent cumulative across providers: {total}"
    );
}

#[tokio::test]
async fn emits_running_cumulative_usage_each_round() {
    // Two rounds (tool call, then text). The UI should see the cumulative
    // total climb after each round, so it can show usage live mid-turn.
    let responses = vec![
        completion(
            vec![Content::ToolCall {
                id: "1".into(),
                name: "bash".into(),
                arguments: "{\"command\":\"echo hi\"}".into(),
            }],
            5,
            1,
        ),
        completion(vec![Content::Text("done".into())], 6, 2),
    ];
    let mut agent = agent(responses, config());
    let mut ui = RecUi::default();
    agent.run_turn("go", &mut ui).await.unwrap();
    // Cumulative after round 1 = (5,1); after round 2 = (11,3).
    assert_eq!(ui.usages, vec![(5, 1), (11, 3)]);
}

#[tokio::test]
async fn auto_compacts_when_context_fills() {
    let mut cfg = config();
    cfg.auto_compact = true;
    cfg.context_window = Some(100);
    let responses = vec![
        completion(vec![Content::Text("ans1".into())], 90, 1), // fills context to 90%
        completion(vec![Content::Text("CONVO SUMMARY".into())], 5, 5), // the compaction call
        completion(vec![Content::Text("ans2".into())], 5, 1),  // turn two, post-compaction
    ];
    let mut agent = agent(responses, cfg);
    let mut ui = RecUi::default();

    agent.run_turn("q1", &mut ui).await.unwrap(); // starts empty → no compaction
    agent.run_turn("q2", &mut ui).await.unwrap(); // context 90% full → compacts first

    assert!(
        ui.statuses.iter().any(|s| s.contains("compacting")),
        "expected a compaction status, got {:?}",
        ui.statuses
    );
    assert!(
        agent
            .messages()
            .iter()
            .any(|m| m.text().contains("CONVO SUMMARY")),
        "history should be replaced by the summary"
    );
    assert_eq!(agent.messages().last().unwrap().text(), "ans2");
}

#[tokio::test]
async fn elides_old_tool_outputs_before_model_request() {
    let mut cfg = config();
    cfg.auto_compact = true;
    cfg.context_window = Some(100);
    let (mut agent, requests) = scripted_agent(
        vec![ProviderStep::Completion(completion(
            vec![Content::Text("done".into())],
            5,
            1,
        ))],
        cfg,
    );
    agent
        .messages_mut()
        .push(Message::user("existing long turn"));
    for i in 1..=8 {
        let id = format!("c{i}");
        agent
            .messages_mut()
            .push(Message::assistant(vec![Content::ToolCall {
                id: id.clone(),
                name: "read".into(),
                arguments: "{}".into(),
            }]));
        agent.messages_mut().push(Message::tool_result(
            &id,
            format!("{i}\n{}", "x".repeat(500)),
        ));
    }

    let mut ui = RecordingUi::default();
    agent.run_turn("continue", &mut ui).await.unwrap();

    let requests = requests.lock().unwrap();
    let outputs: Vec<String> = requests[0]
        .iter()
        .flat_map(|msg| &msg.content)
        .filter_map(|c| match c {
            Content::ToolResult { output, .. } => Some(output.clone()),
            _ => None,
        })
        .collect();
    assert!(outputs[0].starts_with("[elided"), "{outputs:?}");
    assert!(outputs[1].starts_with("[elided"), "{outputs:?}");
    assert!(outputs[2].starts_with("3\n"), "{outputs:?}");
    assert!(outputs[7].starts_with("8\n"), "{outputs:?}");
    assert!(
        !ui.statuses.iter().any(|s| s.contains("elided old tool")),
        "in-turn elision should stay quiet, got {:?}",
        ui.statuses
    );
}

#[tokio::test]
async fn retry_uses_recovery_sampling() {
    // A content-less first round triggers the silent retry, which must
    // resample hotter and with nucleus + frequency penalty to escape the
    // attractor; the initial (non-retry) call uses the plain configured temp.
    let samples = std::sync::Arc::new(Mutex::new(Vec::new()));
    let provider = RecordTemps {
        responses: Mutex::new(vec![
            completion(vec![], 0, 0), // empty → retry
            completion(vec![Content::Text("recovered".into())], 5, 3),
        ]),
        samples: samples.clone(),
    };
    let mut cfg = config();
    cfg.temperature = Some(0.2);
    let mut agent = Agent::new(Box::new(provider), cfg);
    agent.run_turn("go", &mut NullUi).await.unwrap();

    let samples = samples.lock().unwrap();
    assert_eq!(
        samples.len(),
        2,
        "initial call + one retry, got {:?}",
        *samples
    );
    assert_eq!(
        samples[0],
        (Some(0.2), None, None),
        "first call: configured temp, no recovery overrides"
    );
    let (temp, top_p, freq) = samples[1];
    assert!(temp.unwrap() > 0.2, "retry resamples hotter, got {temp:?}");
    assert_eq!(top_p, Some(0.95), "retry adds nucleus sampling");
    assert!(
        freq.is_some_and(|f| f > 0.0),
        "retry adds a frequency penalty, got {freq:?}"
    );
}

#[tokio::test]
async fn empty_response_recovers_on_retry() {
    // First round comes back content-less; the silent retry succeeds. The
    // dead round is dropped from history, so the retry sees the same context.
    let responses = vec![
        completion(vec![], 0, 0), // empty → retry
        completion(vec![Content::Text("here's the review".into())], 5, 3), // succeeds
    ];
    let mut agent = agent(responses, config());
    let mut ui = RecUi::default();
    agent.run_turn("say hi", &mut ui).await.unwrap();
    assert!(
        ui.statuses.iter().any(|s| s.contains("retrying (1/")),
        "a retry should be shown, got: {:?}",
        ui.statuses
    );
    assert!(
        !ui.statuses.iter().any(|s| s.contains("after retrying")),
        "should not have given up, got: {:?}",
        ui.statuses
    );
    assert_eq!(agent.messages().last().unwrap().text(), "here's the review");
    // Only the successful assistant message is recorded (not the dead round).
    let assistants = agent
        .messages()
        .iter()
        .filter(|m| m.role == Role::Assistant)
        .count();
    assert_eq!(assistants, 1);
}

#[tokio::test]
async fn empty_response_gives_up_after_retries() {
    // Persistent content-less responses (the last is reasoning-only, which the
    // old zero-token check missed): exhaust the budget, then surface it.
    let responses = vec![
        completion(vec![], 0, 0),
        completion(vec![], 0, 0),
        completion(vec![], 0, 42),
    ];
    let mut agent = agent(responses, config());
    let mut ui = RecUi::default();
    agent.run_turn("review codebase", &mut ui).await.unwrap();
    assert!(
        ui.statuses.iter().any(|s| s.contains("after retrying")),
        "exhaustion should be surfaced, got: {:?}",
        ui.statuses
    );
}

#[tokio::test]
async fn normal_final_text_does_not_retry() {
    // A turn that ends with real text must not retry or warn.
    let mut agent = agent(
        vec![completion(
            vec![Content::Text("here's the answer".into())],
            5,
            3,
        )],
        config(),
    );
    let mut ui = RecUi::default();
    agent.run_turn("say hi", &mut ui).await.unwrap();
    assert!(
        !ui.statuses.iter().any(|s| s.contains("no response")),
        "real text should not warn, got: {:?}",
        ui.statuses
    );
}

