use std::collections::BTreeSet;
use std::path::{Component, Path, PathBuf};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    ArtifactRef, BindingId, CandidateId, JobId, WORKSPACE_CONTRACT_SCHEMA_VERSION,
    WorkspaceBinding, WorkspaceVersion,
};

pub const MAX_CANDIDATE_CHANGES: usize = 4_096;
pub const MAX_CANDIDATE_POSTIMAGE_BYTES: usize = 128 * 1024 * 1024;
pub const MAX_CANDIDATE_ARTIFACTS: usize = 64;
pub const MAX_CANDIDATE_VERIFICATIONS: usize = 64;
pub const MAX_CANDIDATE_DESTINATION_VERIFIERS: usize = 64;
pub const MAX_CANDIDATE_DESTINATION_VERIFICATION_MS: u64 = 24 * 60 * 60 * 1_000;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CandidateFileKind {
    Regular,
    Symlink,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CandidateFileState {
    pub kind: CandidateFileKind,
    pub mode: u32,
    pub content_digest: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CandidatePostimage {
    pub kind: CandidateFileKind,
    pub mode: u32,
    pub content_digest: String,
    pub bytes: Vec<u8>,
}

impl CandidatePostimage {
    pub fn new(kind: CandidateFileKind, mode: u32, bytes: Vec<u8>) -> Self {
        let content_digest = blake3_prefixed(&bytes);
        Self {
            kind,
            mode,
            content_digest,
            bytes,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CandidateChange {
    pub path: PathBuf,
    pub before: Option<CandidateFileState>,
    pub after: Option<CandidatePostimage>,
}

impl CandidateChange {
    pub fn deletion(path: PathBuf, before: CandidateFileState) -> Self {
        Self {
            path,
            before: Some(before),
            after: None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CandidateVerification {
    pub name: String,
    pub passed: bool,
    pub verifier_digest: String,
    pub detail: Option<String>,
    pub artifacts: Vec<ArtifactRef>,
}

/// One exact verifier the parent must run against the live destination after
/// applying candidate postimages. The command and its finite process deadline
/// are content-bound by [`VerifiedCandidate::candidate_digest`].
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CandidateDestinationVerifier {
    pub name: String,
    pub command: String,
    pub timeout_ms: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CandidateRoute {
    pub provider: String,
    pub model: String,
    pub actual_model_revision: Option<String>,
    pub capability_digest: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct VerifiedCandidateDraft {
    pub candidate_id: CandidateId,
    pub job_id: JobId,
    pub source_binding_id: BindingId,
    pub source_epoch: u64,
    pub base_version: WorkspaceVersion,
    pub before_digest: String,
    pub after_digest: String,
    pub changes: Vec<CandidateChange>,
    pub verification: Vec<CandidateVerification>,
    pub destination_verification: Vec<CandidateDestinationVerifier>,
    pub destination_verification_budget_ms: u64,
    pub artifacts: Vec<ArtifactRef>,
    pub effective_route: CandidateRoute,
}

/// Complete, content-bound evidence produced by a detached write candidate.
/// Construction validates all paths, sizes, digests, and verification results;
/// callers cannot accidentally bless an unchecked draft by setting a digest.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct VerifiedCandidate {
    pub schema_version: u16,
    pub candidate_id: CandidateId,
    pub job_id: JobId,
    pub source_binding_id: BindingId,
    pub source_epoch: u64,
    pub base_version: WorkspaceVersion,
    pub before_digest: String,
    pub after_digest: String,
    pub changes: Vec<CandidateChange>,
    pub verification: Vec<CandidateVerification>,
    /// Empty only for legacy candidate artifacts created before destination
    /// verification was part of the publication contract. Such artifacts can
    /// be inspected/recovered but [`Self::validate_for_apply`] rejects them.
    #[serde(default)]
    pub destination_verification: Vec<CandidateDestinationVerifier>,
    /// Total wall-clock budget shared by the entire destination pipeline.
    /// Zero only on legacy, non-publishable candidates.
    #[serde(default)]
    pub destination_verification_budget_ms: u64,
    pub artifacts: Vec<ArtifactRef>,
    pub candidate_digest: String,
    pub effective_route: CandidateRoute,
}

#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum CandidateError {
    #[error("candidate has no workspace changes")]
    Empty,
    #[error("candidate has too many changes ({actual}; maximum {maximum})")]
    TooManyChanges { actual: usize, maximum: usize },
    #[error("candidate postimages exceed {maximum} bytes")]
    PostimagesTooLarge { maximum: usize },
    #[error("candidate has too many artifact references ({actual}; maximum {maximum})")]
    TooManyArtifacts { actual: usize, maximum: usize },
    #[error("candidate has too many verification records ({actual}; maximum {maximum})")]
    TooManyVerifications { actual: usize, maximum: usize },
    #[error("candidate path is unsafe: {0}")]
    UnsafePath(PathBuf),
    #[error("candidate contains duplicate path: {0}")]
    DuplicatePath(PathBuf),
    #[error("candidate change for {0} has neither a preimage nor a postimage")]
    VacuousChange(PathBuf),
    #[error("candidate postimage digest does not match bytes for {0}")]
    PostimageDigestMismatch(PathBuf),
    #[error("candidate base version is unknown")]
    UnknownBaseVersion,
    #[error("candidate root digest {field} is empty")]
    EmptyRootDigest { field: &'static str },
    #[error("candidate route field {0} is empty")]
    EmptyRoute(&'static str),
    #[error("candidate verification {0} did not pass")]
    VerificationFailed(String),
    #[error("candidate verification name is empty")]
    EmptyVerificationName,
    #[error("candidate verification names must be unique: {0}")]
    DuplicateVerification(String),
    #[error("candidate artifact URI is empty")]
    EmptyArtifactUri,
    #[error("candidate verification {0} has no verifier digest")]
    EmptyVerifierDigest(String),
    #[error("candidate has no executable destination-verification contract")]
    MissingDestinationVerification,
    #[error("candidate has too many destination verifiers ({actual}; maximum {maximum})")]
    TooManyDestinationVerifiers { actual: usize, maximum: usize },
    #[error("candidate destination verifier name is empty")]
    EmptyDestinationVerifierName,
    #[error("candidate destination verifier command is empty: {0}")]
    EmptyDestinationVerifierCommand(String),
    #[error("candidate destination verifier has no finite timeout: {0}")]
    MissingDestinationVerifierTimeout(String),
    #[error("candidate destination verification has no finite total budget")]
    MissingDestinationVerificationBudget,
    #[error("candidate destination verification timeout exceeds {maximum}ms: {actual}ms")]
    DestinationVerificationTimeoutTooLarge { actual: u64, maximum: u64 },
    #[error("candidate digest does not match its payload")]
    DigestMismatch,
    #[error("candidate uses unsupported schema version {0}")]
    UnsupportedSchema(u16),
}

#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum CandidateApplyError {
    #[error(transparent)]
    Invalid(#[from] CandidateError),
    #[error("candidate belongs to a different workspace binding")]
    StaleBinding,
    #[error("candidate belongs to workspace epoch {candidate}, not {current}")]
    StaleEpoch { candidate: u64, current: u64 },
    #[error("candidate base version no longer matches the complete workspace version")]
    StaleBaseVersion,
    #[error("legacy candidate has no destination-verification contract; review or rerun it")]
    MissingDestinationVerification,
}

impl VerifiedCandidate {
    pub fn create(mut draft: VerifiedCandidateDraft) -> Result<Self, CandidateError> {
        if draft.destination_verification.is_empty()
            || draft.destination_verification_budget_ms == 0
        {
            return Err(CandidateError::MissingDestinationVerification);
        }
        draft
            .changes
            .sort_by(|left, right| left.path.cmp(&right.path));
        let mut candidate = Self {
            schema_version: WORKSPACE_CONTRACT_SCHEMA_VERSION,
            candidate_id: draft.candidate_id,
            job_id: draft.job_id,
            source_binding_id: draft.source_binding_id,
            source_epoch: draft.source_epoch,
            base_version: draft.base_version,
            before_digest: draft.before_digest,
            after_digest: draft.after_digest,
            changes: draft.changes,
            verification: draft.verification,
            destination_verification: draft.destination_verification,
            destination_verification_budget_ms: draft.destination_verification_budget_ms,
            artifacts: draft.artifacts,
            candidate_digest: String::new(),
            effective_route: draft.effective_route,
        };
        candidate.validate_payload()?;
        candidate.candidate_digest = candidate.compute_digest();
        Ok(candidate)
    }

    pub fn validate(&self) -> Result<(), CandidateError> {
        if self.schema_version != WORKSPACE_CONTRACT_SCHEMA_VERSION {
            return Err(CandidateError::UnsupportedSchema(self.schema_version));
        }
        self.validate_payload()?;
        if self.candidate_digest != self.compute_digest() {
            return Err(CandidateError::DigestMismatch);
        }
        Ok(())
    }

    pub fn validate_for_apply(
        &self,
        current: &WorkspaceBinding,
    ) -> Result<(), CandidateApplyError> {
        self.validate()?;
        if self.source_binding_id != current.binding_id {
            return Err(CandidateApplyError::StaleBinding);
        }
        if self.source_epoch != current.epoch {
            return Err(CandidateApplyError::StaleEpoch {
                candidate: self.source_epoch,
                current: current.epoch,
            });
        }
        if self.base_version != current.version {
            return Err(CandidateApplyError::StaleBaseVersion);
        }
        if self.destination_verification.is_empty() || self.destination_verification_budget_ms == 0
        {
            return Err(CandidateApplyError::MissingDestinationVerification);
        }
        Ok(())
    }

    pub fn postimage_bytes(&self) -> usize {
        self.changes
            .iter()
            .filter_map(|change| change.after.as_ref())
            .map(|postimage| postimage.bytes.len())
            .sum()
    }

    fn validate_payload(&self) -> Result<(), CandidateError> {
        if self.changes.is_empty() {
            return Err(CandidateError::Empty);
        }
        if self.changes.len() > MAX_CANDIDATE_CHANGES {
            return Err(CandidateError::TooManyChanges {
                actual: self.changes.len(),
                maximum: MAX_CANDIDATE_CHANGES,
            });
        }
        if self.artifacts.len() > MAX_CANDIDATE_ARTIFACTS {
            return Err(CandidateError::TooManyArtifacts {
                actual: self.artifacts.len(),
                maximum: MAX_CANDIDATE_ARTIFACTS,
            });
        }
        if self.verification.len() > MAX_CANDIDATE_VERIFICATIONS {
            return Err(CandidateError::TooManyVerifications {
                actual: self.verification.len(),
                maximum: MAX_CANDIDATE_VERIFICATIONS,
            });
        }
        if matches!(self.base_version, WorkspaceVersion::Unknown) {
            return Err(CandidateError::UnknownBaseVersion);
        }
        if self.before_digest.trim().is_empty() {
            return Err(CandidateError::EmptyRootDigest {
                field: "before_digest",
            });
        }
        if self.after_digest.trim().is_empty() {
            return Err(CandidateError::EmptyRootDigest {
                field: "after_digest",
            });
        }
        validate_route(&self.effective_route)?;
        let mut artifact_count = self.artifacts.len();
        validate_artifacts(&self.artifacts)?;

        let mut paths = BTreeSet::new();
        let mut postimage_bytes = 0usize;
        for change in &self.changes {
            validate_relative_path(&change.path)?;
            if !paths.insert(change.path.clone()) {
                return Err(CandidateError::DuplicatePath(change.path.clone()));
            }
            if change.before.is_none() && change.after.is_none() {
                return Err(CandidateError::VacuousChange(change.path.clone()));
            }
            if let Some(after) = &change.after {
                if after.content_digest != blake3_prefixed(&after.bytes) {
                    return Err(CandidateError::PostimageDigestMismatch(change.path.clone()));
                }
                postimage_bytes = postimage_bytes.checked_add(after.bytes.len()).ok_or(
                    CandidateError::PostimagesTooLarge {
                        maximum: MAX_CANDIDATE_POSTIMAGE_BYTES,
                    },
                )?;
            }
        }
        if postimage_bytes > MAX_CANDIDATE_POSTIMAGE_BYTES {
            return Err(CandidateError::PostimagesTooLarge {
                maximum: MAX_CANDIDATE_POSTIMAGE_BYTES,
            });
        }

        let mut names = BTreeSet::new();
        for result in &self.verification {
            let name = result.name.trim();
            if name.is_empty() {
                return Err(CandidateError::EmptyVerificationName);
            }
            if !names.insert(name.to_owned()) {
                return Err(CandidateError::DuplicateVerification(name.to_owned()));
            }
            if !result.passed {
                return Err(CandidateError::VerificationFailed(name.to_owned()));
            }
            if result.verifier_digest.trim().is_empty() {
                return Err(CandidateError::EmptyVerifierDigest(name.to_owned()));
            }
            validate_artifacts(&result.artifacts)?;
            artifact_count = artifact_count.checked_add(result.artifacts.len()).ok_or(
                CandidateError::TooManyArtifacts {
                    actual: usize::MAX,
                    maximum: MAX_CANDIDATE_ARTIFACTS,
                },
            )?;
        }
        if artifact_count > MAX_CANDIDATE_ARTIFACTS {
            return Err(CandidateError::TooManyArtifacts {
                actual: artifact_count,
                maximum: MAX_CANDIDATE_ARTIFACTS,
            });
        }
        validate_destination_verification(
            &self.destination_verification,
            self.destination_verification_budget_ms,
        )?;
        Ok(())
    }

    fn compute_digest(&self) -> String {
        // Preserve the v1 digest algorithm for inspection of crash evidence
        // written by older binaries. Empty legacy contracts are never
        // publishable; newly-created candidates always use the v2 domain and
        // bind the ordered commands and their finite deadlines.
        let has_destination_verification = !self.destination_verification.is_empty();
        let mut digest = CanonicalDigest::new(if has_destination_verification {
            b"hi.verified-candidate.v2"
        } else {
            b"hi.verified-candidate.v1"
        });
        digest.u16(self.schema_version);
        digest.text(self.candidate_id.as_str());
        digest.text(self.job_id.as_str());
        digest.text(self.source_binding_id.as_str());
        digest.u64(self.source_epoch);
        digest.json(&self.base_version);
        digest.text(&self.before_digest);
        digest.text(&self.after_digest);
        digest.json(&self.changes);
        digest.json(&self.verification);
        if has_destination_verification {
            digest.json(&self.destination_verification);
            digest.u64(self.destination_verification_budget_ms);
        }
        digest.json(&self.artifacts);
        digest.json(&self.effective_route);
        format!("blake3:{}", digest.finish().to_hex())
    }
}

fn validate_destination_verification(
    verifiers: &[CandidateDestinationVerifier],
    total_budget_ms: u64,
) -> Result<(), CandidateError> {
    if verifiers.len() > MAX_CANDIDATE_DESTINATION_VERIFIERS {
        return Err(CandidateError::TooManyDestinationVerifiers {
            actual: verifiers.len(),
            maximum: MAX_CANDIDATE_DESTINATION_VERIFIERS,
        });
    }
    if verifiers.is_empty() {
        return Ok(());
    }
    if total_budget_ms == 0 {
        return Err(CandidateError::MissingDestinationVerificationBudget);
    }
    if total_budget_ms > MAX_CANDIDATE_DESTINATION_VERIFICATION_MS {
        return Err(CandidateError::DestinationVerificationTimeoutTooLarge {
            actual: total_budget_ms,
            maximum: MAX_CANDIDATE_DESTINATION_VERIFICATION_MS,
        });
    }
    for verifier in verifiers {
        let name = verifier.name.trim();
        if name.is_empty() {
            return Err(CandidateError::EmptyDestinationVerifierName);
        }
        if verifier.command.trim().is_empty() {
            return Err(CandidateError::EmptyDestinationVerifierCommand(
                name.to_owned(),
            ));
        }
        if verifier.timeout_ms == 0 {
            return Err(CandidateError::MissingDestinationVerifierTimeout(
                name.to_owned(),
            ));
        }
        if verifier.timeout_ms > MAX_CANDIDATE_DESTINATION_VERIFICATION_MS {
            return Err(CandidateError::DestinationVerificationTimeoutTooLarge {
                actual: verifier.timeout_ms,
                maximum: MAX_CANDIDATE_DESTINATION_VERIFICATION_MS,
            });
        }
    }
    Ok(())
}

fn validate_route(route: &CandidateRoute) -> Result<(), CandidateError> {
    for (name, value) in [
        ("provider", route.provider.as_str()),
        ("model", route.model.as_str()),
        ("capability_digest", route.capability_digest.as_str()),
    ] {
        if value.trim().is_empty() {
            return Err(CandidateError::EmptyRoute(name));
        }
    }
    Ok(())
}

fn validate_artifacts(artifacts: &[ArtifactRef]) -> Result<(), CandidateError> {
    if artifacts.len() > MAX_CANDIDATE_ARTIFACTS {
        return Err(CandidateError::TooManyArtifacts {
            actual: artifacts.len(),
            maximum: MAX_CANDIDATE_ARTIFACTS,
        });
    }
    if artifacts
        .iter()
        .any(|artifact| artifact.uri.trim().is_empty())
    {
        return Err(CandidateError::EmptyArtifactUri);
    }
    Ok(())
}

fn validate_relative_path(path: &Path) -> Result<(), CandidateError> {
    if path.as_os_str().is_empty() || path.is_absolute() {
        return Err(CandidateError::UnsafePath(path.to_path_buf()));
    }
    let Some(portable) = path.to_str() else {
        return Err(CandidateError::UnsafePath(path.to_path_buf()));
    };
    if portable.contains('\\') {
        return Err(CandidateError::UnsafePath(path.to_path_buf()));
    }
    let mut components = path.components();
    if matches!(components.next(), Some(Component::Normal(part)) if part == ".git") {
        return Err(CandidateError::UnsafePath(path.to_path_buf()));
    }
    if path
        .components()
        .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err(CandidateError::UnsafePath(path.to_path_buf()));
    }
    Ok(())
}

fn blake3_prefixed(bytes: &[u8]) -> String {
    format!("blake3:{}", blake3::hash(bytes).to_hex())
}

struct CanonicalDigest(blake3::Hasher);

impl CanonicalDigest {
    fn new(domain: &[u8]) -> Self {
        let mut hasher = blake3::Hasher::new();
        put_bytes(&mut hasher, domain);
        Self(hasher)
    }

    fn u16(&mut self, value: u16) {
        self.0.update(&value.to_le_bytes());
    }

    fn u64(&mut self, value: u64) {
        self.0.update(&value.to_le_bytes());
    }

    fn text(&mut self, value: &str) {
        put_bytes(&mut self.0, value.as_bytes());
    }

    fn json<T: Serialize>(&mut self, value: &T) {
        let encoded = serde_json::to_vec(value).expect("workspace contract values serialize");
        put_bytes(&mut self.0, &encoded);
    }

    fn finish(self) -> blake3::Hash {
        self.0.finalize()
    }
}

fn put_bytes(hasher: &mut blake3::Hasher, value: &[u8]) {
    hasher.update(&(value.len() as u64).to_le_bytes());
    hasher.update(value);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ControllerId, WorkspaceId};

    fn binding() -> WorkspaceBinding {
        WorkspaceBinding::new_local(
            ControllerId::new("controller"),
            WorkspaceId::new("workspace"),
            PathBuf::from("/workspace"),
            PathBuf::from("/state"),
        )
    }

    fn draft(binding: &WorkspaceBinding) -> VerifiedCandidateDraft {
        VerifiedCandidateDraft {
            candidate_id: CandidateId::new("candidate"),
            job_id: JobId::new("job"),
            source_binding_id: binding.binding_id.clone(),
            source_epoch: binding.epoch,
            base_version: binding.version.clone(),
            before_digest: "before".into(),
            after_digest: "after".into(),
            changes: vec![CandidateChange {
                path: PathBuf::from("src/lib.rs"),
                before: None,
                after: Some(CandidatePostimage::new(
                    CandidateFileKind::Regular,
                    0o644,
                    b"new".to_vec(),
                )),
            }],
            verification: vec![CandidateVerification {
                name: "cargo test".into(),
                passed: true,
                verifier_digest: "verify".into(),
                detail: None,
                artifacts: Vec::new(),
            }],
            destination_verification: vec![CandidateDestinationVerifier {
                name: "cargo test".into(),
                command: "cargo test --quiet".into(),
                timeout_ms: 120_000,
            }],
            destination_verification_budget_ms: 120_000,
            artifacts: Vec::new(),
            effective_route: CandidateRoute {
                provider: "openai".into(),
                model: "model".into(),
                actual_model_revision: Some("revision".into()),
                capability_digest: "capabilities".into(),
            },
        }
    }

    #[test]
    fn digest_binds_postimages_and_exact_source_version() {
        let binding = binding();
        let candidate = VerifiedCandidate::create(draft(&binding)).unwrap();
        candidate.validate_for_apply(&binding).unwrap();

        let mut tampered = candidate.clone();
        tampered.changes[0].after.as_mut().unwrap().bytes[0] = b'N';
        assert!(matches!(
            tampered.validate(),
            Err(CandidateError::PostimageDigestMismatch(_))
        ));

        let mut advanced = binding;
        advanced.version = advanced.version.next_local(Some("new-version".into()));
        assert_eq!(
            candidate.validate_for_apply(&advanced),
            Err(CandidateApplyError::StaleBaseVersion)
        );

        let mut changed_verifier = candidate.clone();
        changed_verifier.destination_verification[0].command = "cargo test --all".into();
        assert_eq!(
            changed_verifier.validate(),
            Err(CandidateError::DigestMismatch),
            "destination commands must be bound into the candidate digest"
        );

        let mut changed_timeout = candidate.clone();
        changed_timeout.destination_verification[0].timeout_ms += 1;
        assert_eq!(
            changed_timeout.validate(),
            Err(CandidateError::DigestMismatch),
            "destination deadlines must be bound into the candidate digest"
        );

        let mut changed_total_budget = candidate.clone();
        changed_total_budget.destination_verification_budget_ms += 1;
        assert_eq!(
            changed_total_budget.validate(),
            Err(CandidateError::DigestMismatch),
            "the total destination deadline must be bound into the candidate digest"
        );
    }

    #[test]
    fn legacy_candidate_is_inspectable_but_never_auto_applies() {
        let binding = binding();
        let mut candidate = VerifiedCandidate::create(draft(&binding)).unwrap();
        candidate.destination_verification.clear();
        candidate.destination_verification_budget_ms = 0;
        candidate.candidate_digest = candidate.compute_digest();

        candidate
            .validate()
            .expect("legacy crash evidence should remain inspectable");
        assert_eq!(
            candidate.validate_for_apply(&binding),
            Err(CandidateApplyError::MissingDestinationVerification)
        );
    }

    #[test]
    fn candidate_rejects_git_paths_and_failed_verification() {
        let binding = binding();
        let mut git_path = draft(&binding);
        git_path.changes[0].path = PathBuf::from(".git/config");
        assert!(matches!(
            VerifiedCandidate::create(git_path),
            Err(CandidateError::UnsafePath(_))
        ));

        let mut failed = draft(&binding);
        failed.verification[0].passed = false;
        assert!(matches!(
            VerifiedCandidate::create(failed),
            Err(CandidateError::VerificationFailed(_))
        ));
    }

    #[test]
    fn candidate_digest_is_independent_of_change_input_order() {
        let binding = binding();
        let mut left = draft(&binding);
        left.changes.push(CandidateChange {
            path: PathBuf::from("README.md"),
            before: Some(CandidateFileState {
                kind: CandidateFileKind::Regular,
                mode: 0o644,
                content_digest: "old".into(),
            }),
            after: None,
        });
        let mut right = left.clone();
        right.changes.reverse();
        assert_eq!(
            VerifiedCandidate::create(left).unwrap().candidate_digest,
            VerifiedCandidate::create(right).unwrap().candidate_digest
        );
    }
}
