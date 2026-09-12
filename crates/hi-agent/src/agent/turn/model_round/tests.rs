use super::state::model_step_cap_reached;
use super::*;

#[test]
fn unlimited_step_sentinel_is_never_a_reached_finite_cap() {
    assert!(!model_step_cap_reached(u32::MAX, u32::MAX));
    assert!(!model_step_cap_reached(6, 7));
    assert!(model_step_cap_reached(7, 7));
    assert!(model_step_cap_reached(8, 7));
}

#[test]
fn empty_completion_retry_disables_deepseek_thinking() {
    assert_eq!(deepseek_thinking_for_round(None, false, false, 0), None);
    assert_eq!(
        deepseek_thinking_for_round(None, false, false, 1),
        Some(false)
    );
    assert_eq!(
        deepseek_thinking_for_round(Some(crate::steering::ReviewIntent::Review), true, false, 0),
        Some(true)
    );
}

#[test]
fn collapse_duplicate_inspection_calls_keeps_first_and_preserves_mutations() {
    let read_args = r#"{"path":"src/moves.rs","offset":395,"limit":20}"#;
    let mut content = vec![
        Content::Text("inspect the file".into()),
        Content::ToolCall {
            id: "read-1".into(),
            name: "read".into(),
            arguments: read_args.into(),
        },
        Content::ToolCall {
            id: "read-2".into(),
            name: "read".into(),
            arguments: read_args.into(),
        },
        Content::ToolCall {
            id: "bash-1".into(),
            name: "bash".into(),
            arguments: r#"{"command":"touch marker"}"#.into(),
        },
    ];
    let calls = vec![
        ("read-1".into(), "read".into(), read_args.into()),
        ("read-2".into(), "read".into(), read_args.into()),
        (
            "bash-1".into(),
            "bash".into(),
            r#"{"command":"touch marker"}"#.into(),
        ),
    ];

    let (collapsed, duplicate_count) = collapse_duplicate_inspection_calls(&mut content, calls);

    assert_eq!(duplicate_count, 1);
    assert_eq!(collapsed.len(), 2);
    assert_eq!(collapsed[0].0, "read-1");
    assert_eq!(collapsed[1].0, "bash-1");
    let remaining_ids: Vec<_> = content
        .iter()
        .filter_map(|block| match block {
            Content::ToolCall { id, .. } => Some(id.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(remaining_ids, ["read-1", "bash-1"]);
}
