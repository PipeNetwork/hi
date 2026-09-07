//! Admission and publication ownership for journaled workspace jobs.

use super::*;

impl JournaledWorkspaceController {
    pub(super) async fn register_job_owned(
        &self,
        spec: JobSpec,
    ) -> std::result::Result<JobPermit, AdmissionDenied> {
        if let Some(detail) = self.journal_fence_denial() {
            return Err(self.deny(detail));
        }
        let writer = !matches!(spec.effect_scope, hi_workspace::EffectScope::ReadOnly);
        if writer && !self.writer_jobs_allowed() {
            return Err(self
                .deny("resumable and background writer jobs require a healthy workspace journal"));
        }
        if self.journal_health().state == JournalHealthState::PipeFsFailClosed {
            return Err(self.deny("PipeFS job admission is closed until journal recovery"));
        }

        let reservation = self
            .journal
            .reserve_async()
            .map_err(|error| self.deny(error.to_string()))?;
        let permit = self.inner.register_job(spec).await?;
        lock(&self.permits).insert(permit.job_id.clone(), permit.clone());
        let registered = permit.clone();
        let controller = self.clone();
        if let Err(error) = self
            .journal
            .run_reserved(reservation, move |journal| {
                journal.record_job_with_binding(
                    &controller.inner.binding(),
                    &controller.inner.status(),
                    &controller.inner.capabilities(),
                    &registered,
                    JobState::Running,
                    None,
                    None,
                )?;
                controller.publish_status();
                Ok(())
            })
            .await
        {
            self.note_journal_failure(&error);
            if writer || self.journal_health().policy == JournalFailurePolicy::PipeFsFailClosed {
                let _ = self
                    .inner
                    .seal_job(
                        permit.job_id.clone(),
                        JobTerminal {
                            completion: JobCompletion::Failed,
                            detail: Some(
                                "job admission failed before execution because it was not durably journaled"
                                    .to_owned(),
                            ),
                            artifacts: Vec::new(),
                        },
                    )
                    .await;
                self.publish_status();
                return Err(self.deny("job admission could not be durably journaled"));
            }
        }
        Ok(permit)
    }

    pub(super) async fn seal_job_owned(&self, job: JobId, terminal: JobTerminal) -> JobSealOutcome {
        let permit = lock(&self.permits).get(&job).cloned();
        let local_process_audit_only = permit.as_ref().is_some_and(|permit| {
            self.journal_health().policy == JournalFailurePolicy::LocalContinueForeground
                && permit.spec.kind == hi_workspace::JobKind::Process
                && permit.spec.effect_scope == hi_workspace::EffectScope::LiveWriter
        });
        let must_fence = permit.as_ref().is_some_and(|permit| {
            !matches!(
                permit.spec.effect_scope,
                hi_workspace::EffectScope::ReadOnly
            ) && !local_process_audit_only
        }) || self.journal_health().policy
            == JournalFailurePolicy::PipeFsFailClosed;
        // Every accepted transition retains a publication owner while its
        // acknowledgement is pending. `must_fence` only controls whether an
        // actual journal failure requires recovery or may degrade local audit.
        {
            let mut fences = lock(&self.job_journal_fences);
            match fences.get(&job) {
                Some(JobJournalFence::RecoveryRequired {
                    recovery_id,
                    detail,
                }) => {
                    return JobSealOutcome {
                        job_id: job,
                        status: JobSealStatus::Rejected,
                        state: Some(JobState::RecoveryRequired),
                        recovery_id: Some(recovery_id.clone()),
                        detail: Some(detail.clone()),
                    };
                }
                Some(JobJournalFence::Pending) => {
                    return JobSealOutcome {
                        job_id: job,
                        status: JobSealStatus::Rejected,
                        state: Some(JobState::Settling),
                        recovery_id: None,
                        detail: Some("job publication is already settling".to_owned()),
                    };
                }
                None => {
                    fences.insert(job.clone(), JobJournalFence::Pending);
                }
            }
            drop(fences);
            self.publish_status();
        }

        let outcome = self.inner.seal_job(job.clone(), terminal.clone()).await;
        let fallback = outcome.clone();
        match self
            .journal_work(move |controller| {
                Ok(controller.finish_job_publication(job, terminal, permit, outcome, must_fence))
            })
            .await
        {
            Ok(outcome) => outcome,
            Err(error) => {
                self.note_journal_failure(&error);
                let mut outcome = fallback;
                outcome.status = JobSealStatus::Rejected;
                outcome.state = Some(JobState::RecoveryRequired);
                outcome.detail = Some(error.to_string());
                outcome
            }
        }
    }

    fn finish_job_publication(
        &self,
        job: JobId,
        terminal: JobTerminal,
        permit: Option<JobPermit>,
        outcome: JobSealOutcome,
        must_fence: bool,
    ) -> JobSealOutcome {
        if outcome.status == JobSealStatus::Sealed {
            if let Some(permit) = permit {
                if let Err(error) = self.journal.record_job_with_binding(
                    &self.inner.binding(),
                    &self.inner.status(),
                    &self.inner.capabilities(),
                    &permit,
                    outcome.state.unwrap_or(JobState::RecoveryRequired),
                    outcome.detail.clone(),
                    terminal
                        .artifacts
                        .first()
                        .map(|artifact| artifact.uri.clone()),
                ) {
                    self.note_journal_failure(&error);
                    if must_fence {
                        let binding = self.inner.binding();
                        let recovery_id = journal_job_recovery_id(&binding, &job);
                        let detail = format!(
                            "job reached inner state {:?}, but its lifecycle transition was not durably journaled: {error}",
                            outcome.state.unwrap_or(JobState::RecoveryRequired)
                        );
                        lock(&self.job_journal_fences).insert(
                            job.clone(),
                            JobJournalFence::RecoveryRequired {
                                recovery_id: recovery_id.clone(),
                                detail: detail.clone(),
                            },
                        );
                        if let Err(recovery_error) = self.journal.record_recovery(
                            &binding,
                            &recovery_id,
                            None,
                            Some(job.to_string()),
                            WorkspaceRecoveryStatus::Required,
                            Some(detail.clone()),
                        ) {
                            self.note_journal_failure(&recovery_error);
                        }
                        self.project_binding();
                        return JobSealOutcome {
                            job_id: job,
                            status: JobSealStatus::Rejected,
                            state: Some(JobState::RecoveryRequired),
                            recovery_id: Some(recovery_id),
                            detail: Some(detail),
                        };
                    }
                }
                lock(&self.job_journal_fences).remove(&job);
                if outcome.state.is_some_and(JobState::is_terminal) {
                    lock(&self.permits).remove(&job);
                }
            } else {
                let error = ControlError::Invalid(format!(
                    "missing job permit for lifecycle callback {job}"
                ));
                self.note_journal_failure(&error);
                let binding = self.inner.binding();
                let recovery_id = journal_job_recovery_id(&binding, &job);
                let detail = error.to_string();
                lock(&self.job_journal_fences).insert(
                    job.clone(),
                    JobJournalFence::RecoveryRequired {
                        recovery_id: recovery_id.clone(),
                        detail: detail.clone(),
                    },
                );
                self.project_binding();
                return JobSealOutcome {
                    job_id: job,
                    status: JobSealStatus::Rejected,
                    state: Some(JobState::RecoveryRequired),
                    recovery_id: Some(recovery_id),
                    detail: Some(detail),
                };
            }
        } else {
            lock(&self.job_journal_fences).remove(&job);
        }
        if let Some(recovery_id) = &outcome.recovery_id
            && let Err(error) = self.journal.record_recovery(
                &self.inner.binding(),
                recovery_id,
                None,
                Some(job.to_string()),
                WorkspaceRecoveryStatus::Required,
                outcome.detail.clone(),
            )
        {
            self.note_journal_failure(&error);
        }
        self.publish_status();
        outcome
    }
}
