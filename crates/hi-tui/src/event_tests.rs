use super::*;
use hi_events::{
    ActivityObject, ActivityState, ActivityVerb, EventContext, EventError, EventKind, EventReceipt,
    EventSink, RunEvent, SemanticActivity,
};

#[derive(Default)]
struct RecordingEventSink(std::sync::Mutex<Vec<RunEvent>>);

impl EventSink for RecordingEventSink {
    fn publish(&self, event: RunEvent) -> std::result::Result<EventReceipt, EventError> {
        let event_id = event.event_id.clone();
        let mut events = self.0.lock().unwrap();
        events.push(event);
        Ok(EventReceipt {
            event_id,
            sequence: events.len() as u64,
        })
    }
}

fn event(kind: EventKind, title: &str) -> RunEvent {
    RunEvent::new(
        kind,
        EventContext::default(),
        SemanticActivity {
            verb: ActivityVerb::Verify,
            object: ActivityObject::Verification,
            state: ActivityState::Running,
            group_key: "test".into(),
            title: title.into(),
            detail: None,
            refs: Vec::new(),
            progress: None,
        },
    )
}

#[test]
fn capability_and_verification_lifecycle_is_not_a_status_line() {
    assert!(
        canonical_to_ui_event(&event(
            EventKind::CapabilityRequested,
            "process_execution capability requested"
        ))
        .is_none()
    );
    assert!(
        canonical_to_ui_event(&event(
            EventKind::VerificationStarted,
            "verification started"
        ))
        .is_none()
    );
    assert!(
        canonical_to_ui_event(&event(
            EventKind::VerificationCompleted,
            "verification finished"
        ))
        .is_none()
    );
    assert!(canonical_to_ui_event(&event(EventKind::RunCompleted, "Run finished")).is_none());
    assert!(canonical_to_ui_event(&event(EventKind::RunStarted, "Run started")).is_none());
}

#[test]
fn plan_result_closes_semantic_tool_without_visible_result_row() {
    let (tx, mut events) = mpsc::unbounded_channel();
    let (confirmations, _confirmation_rx) = mpsc::unbounded_channel();
    let sink = Arc::new(RecordingEventSink::default());
    let mut ui = ChannelUi {
        tx,
        confirmations,
        event_sink: Some(sink.clone()),
        approval_store: None,
    };
    let steps = vec![PlanStep {
        title: "verify the harness".into(),
        status: hi_agent::PlanStatus::Active,
    }];

    ui.plan_result_id(
        "plan-call-1",
        "update_plan",
        "Plan recorded: 0/1 done.",
        hi_tools::ToolStatus::Succeeded,
        &steps,
    );

    assert!(matches!(
        events.try_recv(),
        Ok(UiEvent::Plan { steps: received }) if received == steps
    ));
    assert!(matches!(
        events.try_recv(),
        Err(mpsc::error::TryRecvError::Empty)
    ));
    let semantic = sink.0.lock().unwrap();
    assert_eq!(semantic.len(), 1);
    assert_eq!(semantic[0].kind, EventKind::ToolCompleted);
    assert_eq!(semantic[0].activity.object, ActivityObject::Tool);
    assert_eq!(semantic[0].activity.state, ActivityState::Succeeded);
    assert_eq!(semantic[0].activity.group_key, "tool:plan-call-1");
    assert_eq!(semantic[0].activity.title, "update_plan");
}
