//! Unified read-only job lifecycle for transcript compaction.
//!
//! Candidate production receives an owned message snapshot. The only live
//! transcript write is the final, synchronous revision claim followed by the
//! durable replacement boundary and in-memory publication.

use std::sync::Arc;

use anyhow::{Context, Result, anyhow};
use hi_ai::Message;
use hi_workspace::{JobCompletion, JobId, JobSealStatus, JobTerminal, WorkspaceController};

use super::compaction_turn::{fold_reference_summary_into_user, reference_summary_block};
use crate::compaction::{self, CompactionKind};
use crate::{
    ImmutableSessionRevision, SpeculativeCompaction, SpeculativeTranscriptCompaction,
    TranscriptCompactionClaim, Ui,
};

enum PreparedCompaction {
    NoChange(&'static str),
    Replace {
        messages: Vec<Message>,
        status: String,
        fresh_window: bool,
    },
}

enum CompactionPublication {
    Committed { fresh_window: bool, status: String },
    Stale { source: String, current: String },
}

struct CompactionJobGuard {
    pending: Option<(Arc<dyn WorkspaceController>, JobId)>,
}

impl CompactionJobGuard {
    fn new(controller: Arc<dyn WorkspaceController>, job_id: JobId) -> Self {
        Self {
            pending: Some((controller, job_id)),
        }
    }

    fn job_id(&self) -> &JobId {
        &self.pending.as_ref().expect("unsealed compaction job").1
    }

    async fn seal(mut self, completion: JobCompletion, detail: Option<String>) -> Result<()> {
        let (controller, job_id) = self.pending.take().expect("unsealed compaction job");
        seal_compaction_job(controller, job_id, completion, detail).await
    }
}

impl Drop for CompactionJobGuard {
    fn drop(&mut self) {
        let Some((controller, job_id)) = self.pending.take() else {
            return;
        };
        let Ok(runtime) = tokio::runtime::Handle::try_current() else {
            tracing::error!(%job_id, "cancelled compaction job could not schedule its terminal callback");
            return;
        };
        runtime.spawn(async move {
            if let Err(error) = seal_compaction_job(
                controller,
                job_id.clone(),
                JobCompletion::Cancelled,
                Some("compaction future was cancelled before publication".into()),
            )
            .await
            {
                tracing::error!(%error, %job_id, "cancelled compaction job did not seal");
            }
        });
    }
}

impl crate::Agent {
    /// Reclaim context using the session's configured strategy. Candidate
    /// production is a bounded, read-only workspace job.
    pub async fn compact(&mut self, ui: &mut dyn Ui) -> Result<()> {
        self.compact_with(self.config.memory.compaction.clone(), ui)
            .await
    }

    /// Reclaim context using a specific strategy (e.g. `/compact <kind>`).
    pub async fn compact_with(&mut self, kind: CompactionKind, ui: &mut dyn Ui) -> Result<()> {
        let source_revision = ImmutableSessionRevision::capture_at(
            self.messages.revision(),
            self.messages.as_slice(),
        )
        .context("capturing the immutable compaction source revision")?;
        let mut source = self.messages.clone();
        // Treat ephemeral turn-control cleanup as part of the candidate. A
        // failed or stale compaction therefore leaves the live transcript
        // byte-for-byte untouched.
        source.strip_previous_turn_blocks();
        let source_messages = source.as_slice().to_vec();
        let cleaned_revision =
            ImmutableSessionRevision::capture_at(source.revision(), &source_messages)
                .context("capturing the cleaned compaction source revision")?;

        let controller = self.workspace_coordination.job_controller();
        let permit = controller
            .register_job(SpeculativeCompaction::job_spec_with_settings(
                format!(
                    "compact {} at {}",
                    kind_label(&kind),
                    source_revision.digest
                ),
                &self.config.harness.jobs,
            ))
            .await
            .context("registering the read-only compaction job")?;
        let job = CompactionJobGuard::new(controller, permit.job_id);
        let job_id = job.job_id().clone();
        let execution_limit = self.config.harness.jobs.candidate_timeout;
        let prepared = tokio::time::timeout(
            execution_limit,
            self.prepare_compaction(kind, &source_messages, ui),
        )
        .await
        .map_err(|_| {
            ui.assistant_end();
            anyhow!(
                "compaction exceeded its {:.1}-second managed execution limit",
                execution_limit.as_secs_f64()
            )
        })
        .and_then(|result| result);

        let result: Result<bool> = match prepared {
            Ok(PreparedCompaction::NoChange(status)) if cleaned_revision == source_revision => {
                ui.status(status);
                Ok(false)
            }
            Ok(PreparedCompaction::NoChange(status)) => self
                .publish_prepared_compaction(
                    &job_id,
                    source_revision,
                    source_messages,
                    status.to_owned(),
                    false,
                )
                .map(|publication| self.finish_compaction_publication(publication, ui)),
            Ok(PreparedCompaction::Replace {
                messages,
                status,
                fresh_window,
            }) => self
                .publish_prepared_compaction(
                    &job_id,
                    source_revision,
                    messages,
                    status,
                    fresh_window,
                )
                .map(|publication| self.finish_compaction_publication(publication, ui)),
            Err(error) => Err(error),
        };

        let completion = match &result {
            Ok(true) => JobCompletion::Stale,
            Ok(false) => JobCompletion::Succeeded,
            Err(_) => JobCompletion::Failed,
        };
        let detail = match &result {
            Ok(true) => Some("source session revision changed; candidate was discarded".into()),
            Ok(false) => Some("compaction job reached a terminal publication boundary".into()),
            Err(error) => Some(format!("compaction failed before publication: {error:#}")),
        };
        let seal = job.seal(completion, detail).await;
        match (result, seal) {
            (Ok(_), Ok(())) => Ok(()),
            (Err(error), Ok(())) => Err(error),
            (Ok(_), Err(seal)) => Err(seal),
            (Err(error), Err(seal)) => {
                Err(error.context(format!("compaction job settlement also failed: {seal:#}")))
            }
        }
    }

    async fn prepare_compaction(
        &mut self,
        kind: CompactionKind,
        source: &[Message],
        ui: &mut dyn Ui,
    ) -> Result<PreparedCompaction> {
        match kind {
            CompactionKind::Summarize => self.prepare_summarize(source, ui).await,
            CompactionKind::Hybrid { keep_recent } => {
                self.prepare_hybrid(source, keep_recent, ui).await
            }
            CompactionKind::ElideToolOutput { keep_recent } => {
                Ok(prepare_elide(source, keep_recent))
            }
            CompactionKind::ElideThenSummarizeTail { keep_recent } => {
                self.prepare_elide_then_summarize_tail(source, keep_recent, ui)
                    .await
            }
            CompactionKind::FreshWindow => Ok(PreparedCompaction::Replace {
                messages: self.prepare_fresh_window_compaction(source),
                status: "fresh context window — conversation dropped, goal/decisions kept".into(),
                fresh_window: true,
            }),
        }
    }

    async fn prepare_summarize(
        &mut self,
        source: &[Message],
        ui: &mut dyn Ui,
    ) -> Result<PreparedCompaction> {
        if source.len() <= 1 {
            return Ok(PreparedCompaction::NoChange("nothing to compact yet"));
        }
        let Some(summary) = self.summarize(&source[1..], ui).await? else {
            return Ok(PreparedCompaction::NoChange(
                "compaction produced no summary; keeping history",
            ));
        };
        Ok(PreparedCompaction::Replace {
            messages: vec![
                self.system_message(),
                Message::user(reference_summary_block(&summary)),
            ],
            status: "✓ compacted — context reset to the summary".into(),
            fresh_window: false,
        })
    }

    async fn prepare_hybrid(
        &mut self,
        source: &[Message],
        keep_recent: usize,
        ui: &mut dyn Ui,
    ) -> Result<PreparedCompaction> {
        if keep_recent == 0 {
            return self.prepare_summarize(source, ui).await;
        }
        let Some(split) = compaction::recent_split(source, keep_recent) else {
            return self.prepare_summarize(source, ui).await;
        };
        let Some(summary) = self.summarize(&source[1..split], ui).await? else {
            return Ok(PreparedCompaction::NoChange(
                "compaction produced no summary; keeping history",
            ));
        };
        let mut recent = source[split..].to_vec();
        fold_reference_summary_into_user(&summary, &mut recent[0]);
        let mut messages = Vec::with_capacity(recent.len() + 1);
        messages.push(self.system_message());
        messages.extend(recent);
        Ok(PreparedCompaction::Replace {
            messages,
            status: "✓ compacted — kept recent turns, summarized the rest".into(),
            fresh_window: false,
        })
    }

    async fn prepare_elide_then_summarize_tail(
        &mut self,
        source: &[Message],
        keep_recent: usize,
        ui: &mut dyn Ui,
    ) -> Result<PreparedCompaction> {
        if keep_recent == 0 {
            return self.prepare_summarize(source, ui).await;
        }
        let Some(split) = compaction::recent_split(source, keep_recent) else {
            return self.prepare_summarize(source, ui).await;
        };
        let mut working = source.to_vec();
        compaction::elide_tool_outputs(&mut working, split);
        let convo = compaction::conversational_tail(&working, split);
        let summary = if convo.is_empty() {
            None
        } else {
            self.summarize(&convo, ui).await?
        };
        let old = compaction::tool_bearing_turns(&working, split);
        let mut recent = working[split..].to_vec();
        let had_summary = summary.is_some();
        if let Some(summary) = summary {
            fold_reference_summary_into_user(&summary, &mut recent[0]);
        }
        let mut messages = Vec::with_capacity(1 + old.len() + recent.len());
        messages.push(self.system_message());
        messages.extend(old);
        messages.extend(recent);
        let status = if had_summary {
            "✓ compacted — elided old tool output, summarized the Q&A tail"
        } else {
            "✓ compacted — elided old tool output (no Q&A tail to summarize)"
        };
        Ok(PreparedCompaction::Replace {
            messages,
            status: status.into(),
            fresh_window: false,
        })
    }

    fn publish_prepared_compaction(
        &mut self,
        job_id: &JobId,
        source_revision: ImmutableSessionRevision,
        messages: Vec<Message>,
        status: String,
        fresh_window: bool,
    ) -> Result<CompactionPublication> {
        hi_workspace::hit_harness_failpoint(hi_workspace::HarnessFailpoint::CompactionBeforeCas)
            .map_err(anyhow::Error::from)?;
        let candidate =
            SpeculativeTranscriptCompaction::new(job_id.clone(), source_revision, messages);
        match candidate
            .claim_if_current(self.messages.revision(), self.messages.as_slice())
            .context("checking the compaction publication revision")?
        {
            TranscriptCompactionClaim::Stale {
                source_revision,
                current_revision,
                ..
            } => Ok(CompactionPublication::Stale {
                source: source_revision.digest,
                current: current_revision.digest,
            }),
            TranscriptCompactionClaim::Current { messages, .. } => {
                // No await is permitted between the revision claim above and
                // these two writes. The exclusive Agent borrow prevents a
                // concurrent transcript event from crossing the boundary.
                if let Some(session) = self.session.as_mut() {
                    session
                        .record_compaction(&messages)
                        .context("persisting the speculative compaction boundary")?;
                }
                self.messages.replace_all(messages);
                self.persisted = self.messages.len();
                Ok(CompactionPublication::Committed {
                    fresh_window,
                    status,
                })
            }
        }
    }

    fn finish_compaction_publication(
        &mut self,
        publication: CompactionPublication,
        ui: &mut dyn Ui,
    ) -> bool {
        match publication {
            CompactionPublication::Committed {
                fresh_window,
                status,
            } => {
                self.runtime.invalidate_context_after_compaction();
                if fresh_window {
                    self.finish_fresh_window_compaction();
                }
                ui.status(&status);
                false
            }
            CompactionPublication::Stale { source, current } => {
                ui.status(&format!(
                    "compaction result discarded because the session advanced ({source} -> {current})"
                ));
                true
            }
        }
    }
}

fn prepare_elide(source: &[Message], keep_recent: usize) -> PreparedCompaction {
    let Some(split) = compaction::recent_split(source, keep_recent) else {
        return PreparedCompaction::NoChange("nothing old to elide");
    };
    let mut messages = source.to_vec();
    let freed = compaction::elide_tool_outputs(&mut messages, split);
    if freed == 0 {
        PreparedCompaction::NoChange("nothing old to elide")
    } else {
        PreparedCompaction::Replace {
            messages,
            status: format!("✓ elided ~{}k chars of old tool output", freed / 1000),
            fresh_window: false,
        }
    }
}

async fn seal_compaction_job(
    controller: Arc<dyn WorkspaceController>,
    job_id: JobId,
    completion: JobCompletion,
    detail: Option<String>,
) -> Result<()> {
    let outcome = controller
        .seal_job(
            job_id.clone(),
            JobTerminal {
                completion,
                detail,
                artifacts: Vec::new(),
            },
        )
        .await;
    match outcome.status {
        JobSealStatus::Sealed => Ok(()),
        status => Err(anyhow!(
            "compaction job {job_id} settlement was rejected ({status:?}): {}",
            outcome.detail.as_deref().unwrap_or("no detail")
        )),
    }
}

fn kind_label(kind: &CompactionKind) -> &'static str {
    match kind {
        CompactionKind::Summarize => "summarize",
        CompactionKind::Hybrid { .. } => "hybrid",
        CompactionKind::ElideToolOutput { .. } => "elide",
        CompactionKind::ElideThenSummarizeTail { .. } => "tail",
        CompactionKind::FreshWindow => "fresh-window",
    }
}
