use super::{PipeFsError, PipeFsRemoteState};

pub(super) fn validate_remote_state(
    expected_session_id: &str,
    state: PipeFsRemoteState,
) -> Result<PipeFsRemoteState, PipeFsError> {
    if state.session_id != expected_session_id {
        return Err(PipeFsError::Protocol(format!(
            "PipeFS server returned state for session {}, expected {}",
            state.session_id, expected_session_id
        )));
    }
    Ok(state)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn remote_state_is_fenced_to_the_requested_session() {
        let state = PipeFsRemoteState {
            session_id: "other-session".into(),
            enabled: true,
            current_head: None,
            sequence: 0,
            manifest_digest: None,
            logical_size_bytes: 0,
            restore_chain: Vec::new(),
        };
        let error = validate_remote_state("expected-session", state).unwrap_err();
        assert!(error.to_string().contains("other-session"));
        assert!(error.to_string().contains("expected-session"));
    }
}
