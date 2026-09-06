use super::{
    Content, CostEstimate, Message, PromptInput, RateLimitBucket, RateLimitState, ReasoningEffort,
    ToolSpec, Usage, estimate_completion_output_tokens, estimate_request_input_tokens,
    estimate_text_tokens,
};

#[test]
fn effort_escalation_steps_up_and_saturates() {
    assert_eq!(ReasoningEffort::Minimal.next_higher(), ReasoningEffort::Low);
    assert_eq!(ReasoningEffort::Medium.next_higher(), ReasoningEffort::High);
    assert_eq!(ReasoningEffort::High.next_higher(), ReasoningEffort::Xhigh);
    assert_eq!(ReasoningEffort::Xhigh.next_higher(), ReasoningEffort::Xhigh);
}

#[test]
fn add_preserves_last_observed_rate_limits_and_sticks_estimated() {
    let mut totals = Usage {
        input_tokens: 100,
        output_tokens: 10,
        rate_limits: Some(RateLimitState {
            requests_min: RateLimitBucket {
                limit: 10,
                remaining: 8,
                reset_seconds: 1,
            },
            ..RateLimitState::default()
        }),
        ..Usage::default()
    };

    // A booking with no rate-limit snapshot (side-call, error usage,
    // estimate) must not wipe the last observed one.
    totals.add(Usage {
        input_tokens: 50,
        output_tokens: 5,
        estimated: true,
        ..Usage::default()
    });
    assert_eq!(totals.input_tokens, 150);
    assert!(
        totals.rate_limits.is_some(),
        "zero-snapshot add wiped rate limits"
    );
    // Estimated is sticky: once any component was guessed, totals say so.
    assert!(totals.estimated);
    totals.add(Usage {
        input_tokens: 5,
        ..Usage::default()
    });
    assert!(totals.estimated, "estimated must not reset");

    // A booking that carries a fresh snapshot replaces the old one.
    totals.add(Usage {
        rate_limits: Some(RateLimitState {
            requests_min: RateLimitBucket {
                limit: 10,
                remaining: 3,
                reset_seconds: 2,
            },
            ..RateLimitState::default()
        }),
        ..Usage::default()
    });
    assert_eq!(totals.rate_limits.unwrap().requests_min.remaining, 3);
}

#[test]
fn cost_estimate_is_optional_and_uses_micro_usd() {
    let usage = Usage {
        input_tokens: 100,
        output_tokens: 50,
        ..Usage::default()
    };
    let cost = CostEstimate::from_usage(&usage, Some((2.0, 4.0))).unwrap();
    assert_eq!(cost.input_microusd, 200);
    assert_eq!(cost.output_microusd, 200);
    assert_eq!(cost.total_microusd, 400);
    assert!(CostEstimate::from_usage(&usage, None).is_none());
}

#[test]
fn cost_estimate_does_not_bill_cache_read_at_full_input_rate() {
    let usage = Usage {
        input_tokens: 100,
        output_tokens: 50,
        cache_read_tokens: 40,
        input_includes_cache: true,
        ..Usage::default()
    };
    let cost = CostEstimate::from_usage(&usage, Some((2.0, 4.0))).unwrap();
    // Uncached 60 × $2/MTok = 120µUSD; output 50 × $4 = 200µUSD.
    assert_eq!(cost.input_microusd, 120);
    assert_eq!(cost.output_microusd, 200);
    assert_eq!(cost.total_microusd, 320);
    let discounted =
        CostEstimate::from_usage_with_cache(&usage, Some((2.0, 4.0)), Some(0.2)).unwrap();
    // 60×2 + 40×0.2 = 120 + 8.
    assert_eq!(discounted.input_microusd, 128);
    assert_eq!(discounted.format_usd(), "$0.0003");
}

#[test]
fn typed_prompt_preserves_text_and_image_blocks() {
    let input = PromptInput::text("inspect this").image("aGVsbG8=", "image/png");
    let message = input.clone().into_message();
    assert_eq!(input.text_content(), "inspect this");
    assert!(matches!(message.content[0], Content::Text(_)));
    assert!(matches!(message.content[1], Content::Image { .. }));
}

#[test]
fn request_estimate_uses_the_same_content_and_tool_schema_paths() {
    let messages = vec![
        Message::system("rules"),
        Message::assistant(vec![Content::Thinking {
            text: "think".into(),
            signature: Some("sig".into()),
        }]),
        Message::assistant(vec![Content::ToolCall {
            id: "call_1".into(),
            name: "read".into(),
            arguments: r#"{"path":"README.md"}"#.into(),
        }]),
        Message::tool_result("call_1", "contents"),
    ];
    let tools = vec![ToolSpec {
        name: "read".into(),
        description: "read a file".into(),
        parameters: serde_json::json!({
            "type": "object",
            "properties": {"path": {"type": "string"}}
        }),
    }];

    let expected_messages = messages
        .iter()
        .flat_map(|message| &message.content)
        .map(super::estimate_content_tokens)
        .fold(0, u64::saturating_add);
    let expected_tools = estimate_text_tokens("read")
        + estimate_text_tokens("read a file")
        + super::estimate_json_tokens(&tools[0].parameters.to_string());
    assert_eq!(
        estimate_request_input_tokens(&messages, &tools),
        expected_messages.saturating_add(expected_tools)
    );
    assert_eq!(
        estimate_completion_output_tokens(&messages[1].content),
        estimate_text_tokens("think")
    );
}

#[test]
fn estimate_arithmetic_saturates_and_empty_text_is_zero() {
    assert_eq!(estimate_text_tokens(""), 0);
    assert!(
        super::estimate_json_tokens(r#"{"type":"object"}"#)
            > estimate_text_tokens(r#"{"type":"object"}"#),
        "JSON occupancy must be denser than the prose 4-byte heuristic"
    );
    let usage = Usage {
        input_tokens: u64::MAX,
        cache_read_tokens: u64::MAX,
        cache_creation_tokens: u64::MAX,
        input_includes_cache: false,
        ..Usage::default()
    };
    assert_eq!(usage.effective_input_tokens(), u64::MAX);
}
