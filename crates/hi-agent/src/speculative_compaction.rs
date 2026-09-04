//! Revision-fenced speculative compaction.
//!
//! Production of a candidate is read-only: it owns an immutable message
//! snapshot and never mutates the live reducer. Publication is a single reducer
//! event that succeeds only while the exact source revision is still current.

use hi_ai::Message;
use hi_workspace::{EffectScope, HarnessJobSettings, JobId, JobKind, JobLimits, JobSpec};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{SessionEvent, SessionEventKind, SessionReduceError, SessionReducer, SessionState};

pub const SPECULATIVE_COMPACTION_SCHEMA_VERSION: u16 = 2;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SpeculativeCompaction {
    pub schema_version: u16,
    pub job_id: JobId,
    pub source_revision: u64,
    pub messages: Vec<Message>,
}

impl SpeculativeCompaction {
    pub fn new(job_id: JobId, source_revision: u64, messages: Vec<Message>) -> Self {
        Self {
            schema_version: SPECULATIVE_COMPACTION_SCHEMA_VERSION,
            job_id,
            source_revision,
            messages,
        }
    }

    pub fn job_spec(name: impl Into<String>) -> JobSpec {
        Self::job_spec_with_settings(
            name,
            &HarnessJobSettings {
                queue_timeout: std::time::Duration::from_secs(5 * 60),
                candidate_timeout: std::time::Duration::from_secs(15 * 60),
                verifier_timeout: std::time::Duration::from_secs(2 * 60),
                max_preparations: 4,
                max_active: 16,
            },
        )
    }

    pub fn job_spec_with_settings(
        name: impl Into<String>,
        settings: &HarnessJobSettings,
    ) -> JobSpec {
        JobSpec {
            kind: JobKind::Compaction,
            effect_scope: EffectScope::ReadOnly,
            name: name.into(),
            limits: JobLimits {
                queue_ms: Some(duration_ms(settings.queue_timeout)),
                execution_ms: Some(duration_ms(settings.candidate_timeout)),
                verification_ms: None,
                output_bytes: Some(50_000),
            },
            parent_operation: None,
        }
    }

    /// Publish the compacted projection only when no event landed after the
    /// immutable source snapshot. Stale work is discarded without side effects.
    pub fn commit_if_current(
        self,
        reducer: &mut SessionReducer,
    ) -> Result<SpeculativeCompactionOutcome, SessionReduceError> {
        if reducer.through_sequence() != self.source_revision {
            return Ok(SpeculativeCompactionOutcome::Stale {
                job_id: self.job_id,
                source_revision: self.source_revision,
                current_revision: reducer.through_sequence(),
            });
        }
        if let Err(error) =
            hi_workspace::hit_harness_failpoint(hi_workspace::HarnessFailpoint::CompactionBeforeCas)
        {
            return Ok(SpeculativeCompactionOutcome::Failed {
                job_id: self.job_id,
                detail: error.to_string(),
            });
        }
        let state = reducer.apply(
            SessionEvent::new(SessionEventKind::Compaction {
                messages: self.messages,
            })
            .at_sequence(self.source_revision.saturating_add(1)),
        )?;
        Ok(SpeculativeCompactionOutcome::Committed {
            job_id: self.job_id,
            revision: reducer.through_sequence(),
            state: Box::new(state),
        })
    }
}

/// Content-addressed revision of an immutable provider-facing transcript.
///
/// The live agent is still serialized behind `&mut Agent`, but the digest is
/// deliberately carried with the candidate so future background scheduling
/// cannot accidentally publish work produced from an older message snapshot.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImmutableSessionRevision {
    pub schema_version: u16,
    pub sequence: u64,
    pub digest: String,
    pub message_count: u64,
}

impl ImmutableSessionRevision {
    pub fn capture(messages: &[Message]) -> Result<Self, serde_json::Error> {
        Self::capture_at(0, messages)
    }

    pub fn capture_at(sequence: u64, messages: &[Message]) -> Result<Self, serde_json::Error> {
        let bytes = serde_json::to_vec(messages)?;
        Ok(Self {
            schema_version: SPECULATIVE_COMPACTION_SCHEMA_VERSION,
            sequence,
            digest: format!("sha256:{:x}", Sha256::digest(bytes)),
            message_count: messages.len().try_into().unwrap_or(u64::MAX),
        })
    }
}

/// A compaction result produced entirely from an owned session snapshot.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SpeculativeTranscriptCompaction {
    pub schema_version: u16,
    pub job_id: JobId,
    pub source_revision: ImmutableSessionRevision,
    pub messages: Vec<Message>,
}

impl SpeculativeTranscriptCompaction {
    pub fn new(
        job_id: JobId,
        source_revision: ImmutableSessionRevision,
        messages: Vec<Message>,
    ) -> Self {
        Self {
            schema_version: SPECULATIVE_COMPACTION_SCHEMA_VERSION,
            job_id,
            source_revision,
            messages,
        }
    }

    /// Claim this candidate for synchronous publication. The caller must not
    /// await between this comparison and its durable-boundary/live-state
    /// replacement; `Agent`'s exclusive borrow makes that pair one CAS edge.
    pub fn claim_if_current(
        self,
        current_sequence: u64,
        current: &[Message],
    ) -> Result<TranscriptCompactionClaim, serde_json::Error> {
        let current_revision = ImmutableSessionRevision::capture_at(current_sequence, current)?;
        if current_revision != self.source_revision {
            return Ok(TranscriptCompactionClaim::Stale {
                job_id: self.job_id,
                source_revision: self.source_revision,
                current_revision,
            });
        }
        Ok(TranscriptCompactionClaim::Current {
            job_id: self.job_id,
            source_revision: self.source_revision,
            messages: self.messages,
        })
    }
}

#[derive(Clone, Debug)]
pub enum TranscriptCompactionClaim {
    Current {
        job_id: JobId,
        source_revision: ImmutableSessionRevision,
        messages: Vec<Message>,
    },
    Stale {
        job_id: JobId,
        source_revision: ImmutableSessionRevision,
        current_revision: ImmutableSessionRevision,
    },
}

#[derive(Clone, Debug)]
pub enum SpeculativeCompactionOutcome {
    Committed {
        job_id: JobId,
        revision: u64,
        state: Box<SessionState>,
    },
    Stale {
        job_id: JobId,
        source_revision: u64,
        current_revision: u64,
    },
    Failed {
        job_id: JobId,
        detail: String,
    },
}

fn duration_ms(duration: std::time::Duration) -> u64 {
    duration.as_millis().try_into().unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests {
    use hi_ai::Message;

    use super::*;

    #[test]
    fn compaction_job_is_strictly_read_only() {
        let spec = SpeculativeCompaction::job_spec("compact revision 4");
        assert_eq!(spec.kind, JobKind::Compaction);
        assert_eq!(spec.effect_scope, EffectScope::ReadOnly);
        assert!(spec.limits.execution_ms.is_some());
    }

    #[test]
    fn stale_candidate_never_changes_the_reducer() {
        let mut reducer = SessionReducer::new();
        reducer
            .apply(SessionEvent::new(SessionEventKind::Message {
                message: Message::user("new turn"),
            }))
            .unwrap();
        let candidate = SpeculativeCompaction::new(
            JobId::new("compaction-1"),
            0,
            vec![Message::user("summary")],
        );
        let outcome = candidate.commit_if_current(&mut reducer).unwrap();
        assert!(matches!(
            outcome,
            SpeculativeCompactionOutcome::Stale {
                source_revision: 0,
                current_revision: 1,
                ..
            }
        ));
        assert_eq!(reducer.state().messages[0].text(), "new turn");
    }

    #[test]
    fn exact_revision_candidate_commits_once() {
        let mut reducer = SessionReducer::new();
        let candidate = SpeculativeCompaction::new(
            JobId::new("compaction-1"),
            0,
            vec![Message::user("summary")],
        );
        let outcome = candidate.commit_if_current(&mut reducer).unwrap();
        assert!(matches!(
            outcome,
            SpeculativeCompactionOutcome::Committed { revision: 1, .. }
        ));
        assert_eq!(reducer.state().messages[0].text(), "summary");
    }

    #[test]
    fn transcript_candidate_discards_a_stale_snapshot() {
        let source = vec![Message::system("system"), Message::user("old task")];
        let revision = ImmutableSessionRevision::capture_at(4, &source).unwrap();
        let candidate = SpeculativeTranscriptCompaction::new(
            JobId::new("compaction-1"),
            revision.clone(),
            vec![Message::system("system"), Message::user("summary")],
        );
        let mut current = source;
        current.push(Message::assistant(vec![hi_ai::Content::Text(
            "new event".into(),
        )]));

        let claim = candidate.claim_if_current(5, &current).unwrap();
        assert!(matches!(
            claim,
            TranscriptCompactionClaim::Stale {
                source_revision,
                current_revision,
                ..
            } if source_revision == revision && current_revision != revision
        ));
        assert_eq!(current.last().unwrap().text(), "new event");
    }

    #[test]
    fn transcript_candidate_rejects_an_aba_revision() {
        let source = vec![Message::system("system"), Message::user("old task")];
        let revision = ImmutableSessionRevision::capture_at(7, &source).unwrap();
        let candidate = SpeculativeTranscriptCompaction::new(
            JobId::new("compaction-aba"),
            revision.clone(),
            vec![Message::system("system"), Message::user("summary")],
        );

        let claim = candidate.claim_if_current(9, &source).unwrap();
        assert!(matches!(
            claim,
            TranscriptCompactionClaim::Stale { current_revision, .. }
                if current_revision.sequence == 9 && current_revision.digest == revision.digest
        ));
    }
}
