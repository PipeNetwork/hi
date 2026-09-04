//! Runtime capability resolution with bounded, injectable probes.

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::fmt;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::Result;
use async_trait::async_trait;
use futures_util::future::join_all;
use serde::{Deserialize, Serialize};

use super::ProviderCapabilities;

pub const CAPABILITY_RECORD_SCHEMA_VERSION: u16 = 1;
pub const CAPABILITY_REGISTRY_VERSION: u16 = 1;
pub const DEFAULT_CAPABILITY_CACHE_TTL: Duration = Duration::from_secs(15 * 60);
pub const DEFAULT_CAPABILITY_PROBE_TIMEOUT: Duration = Duration::from_secs(2);
pub const MAX_CAPABILITY_PROBE_TIMEOUT: Duration = Duration::from_secs(10);
pub const DEFAULT_CAPABILITY_PROBE_MEMBERS: usize = 8;
pub const MAX_CAPABILITY_PROBE_MEMBERS: usize = 32;
const DEFAULT_AUDIT_CAPACITY: usize = 128;
const MAX_AUDIT_CAPACITY: usize = 4_096;
const MAX_AUDIT_DETAIL_CHARS: usize = 512;

/// Canonical lookup key. The route is the effective provider route, not merely
/// a provider family, because gateways may serve different wire contracts.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct CapabilityRoute {
    pub route: String,
    pub model: String,
}

impl CapabilityRoute {
    pub fn new(route: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            route: normalized_identity(route.into()),
            model: normalized_identity(model.into()),
        }
    }
}

/// One possible backend for an effective route. Multi-backend routes resolve
/// each member independently and advertise only their intersection.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProviderCapabilityCandidate {
    pub target: CapabilityRoute,
    pub declared: ProviderCapabilities,
}

impl ProviderCapabilityCandidate {
    pub fn new(target: CapabilityRoute, declared: ProviderCapabilities) -> Self {
        Self { target, declared }
    }
}

/// Successful probe output. Revision stays separate so probes can report the
/// backend's concrete revision even when the capability body came from tags.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CapabilityProbeObservation {
    pub capabilities: ProviderCapabilities,
    pub actual_model_revision: Option<String>,
}

#[async_trait]
pub trait CapabilityProbe: Send + Sync {
    /// Perform at most one provider-specific observation. The registry always
    /// wraps this future in its own deadline; implementations must not retry.
    async fn probe(&self, target: &CapabilityRoute) -> Result<Option<CapabilityProbeObservation>>;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CapabilityProbeDisposition {
    NotConfigured,
    Succeeded,
    Unknown,
    Failed,
    TimedOut,
    SkippedLimit,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CapabilityMemberRecord {
    pub target: CapabilityRoute,
    pub disposition: CapabilityProbeDisposition,
    pub declared_digest: String,
    pub observed_digest: Option<String>,
    pub actual_model_revision: Option<String>,
}

/// Exact capability facts used to shape and audit one model request. Wall-clock
/// cache data is excluded, keeping this record replay- and digest-stable.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct EffectiveProviderCapabilities {
    pub schema_version: u16,
    pub registry_version: u16,
    pub target: CapabilityRoute,
    pub capabilities: ProviderCapabilities,
    pub members: Vec<CapabilityMemberRecord>,
}

impl EffectiveProviderCapabilities {
    pub fn conservative(target: CapabilityRoute, capabilities: ProviderCapabilities) -> Self {
        let declared_digest = capabilities.canonical_digest();
        let actual_model_revision = capabilities.actual_model_revision.clone();
        Self {
            schema_version: CAPABILITY_RECORD_SCHEMA_VERSION,
            registry_version: CAPABILITY_REGISTRY_VERSION,
            target: target.clone(),
            capabilities,
            members: vec![CapabilityMemberRecord {
                target,
                disposition: CapabilityProbeDisposition::NotConfigured,
                declared_digest,
                observed_digest: None,
                actual_model_revision,
            }],
        }
    }

    pub fn canonical_digest(&self) -> String {
        canonical_digest(self)
    }

    pub fn actual_model_revision(&self) -> Option<&str> {
        self.capabilities.actual_model_revision.as_deref()
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CapabilityProbeAuditRecord {
    pub schema_version: u16,
    pub registry_version: u16,
    pub target: CapabilityRoute,
    pub member: CapabilityRoute,
    pub disposition: CapabilityProbeDisposition,
    pub cache_hit: bool,
    pub effective_digest: String,
    pub observed_digest: Option<String>,
    pub member_capabilities: ProviderCapabilities,
    pub recorded_at_ms: u64,
    pub expires_at_ms: u64,
    pub detail: Option<String>,
    /// Digest of semantic audit fields; timestamps and diagnostic prose are
    /// deliberately excluded so replays remain comparable.
    pub audit_digest: String,
}

#[derive(Clone, Copy, Debug)]
pub struct CapabilityRegistryConfig {
    pub registry_version: u16,
    pub cache_ttl: Duration,
    pub probe_timeout: Duration,
    pub max_probe_members: usize,
    pub max_audit_records: usize,
}

impl Default for CapabilityRegistryConfig {
    fn default() -> Self {
        Self {
            registry_version: CAPABILITY_REGISTRY_VERSION,
            cache_ttl: DEFAULT_CAPABILITY_CACHE_TTL,
            probe_timeout: DEFAULT_CAPABILITY_PROBE_TIMEOUT,
            max_probe_members: DEFAULT_CAPABILITY_PROBE_MEMBERS,
            max_audit_records: DEFAULT_AUDIT_CAPACITY,
        }
    }
}

#[derive(Clone)]
pub struct ProviderCapabilityRegistry {
    inner: Arc<RegistryInner>,
}

struct RegistryInner {
    config: CapabilityRegistryConfig,
    probe: Option<Arc<dyn CapabilityProbe>>,
    registered: Mutex<HashMap<CapabilityRoute, ProviderCapabilities>>,
    cache: Mutex<HashMap<CapabilityRoute, CachedProbe>>,
    audit: Mutex<VecDeque<CapabilityProbeAuditRecord>>,
}

#[derive(Clone)]
struct CachedProbe {
    registry_version: u16,
    expires_at_ms: u64,
    declared_digest: String,
    capabilities: ProviderCapabilities,
    disposition: CapabilityProbeDisposition,
    observed_digest: Option<String>,
    detail: Option<String>,
}

struct ResolvedMember {
    record: CapabilityMemberRecord,
    capabilities: ProviderCapabilities,
    cache_hit: bool,
    expires_at_ms: u64,
    detail: Option<String>,
}

impl Default for ProviderCapabilityRegistry {
    fn default() -> Self {
        Self::new(CapabilityRegistryConfig::default(), None)
    }
}

impl fmt::Debug for ProviderCapabilityRegistry {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ProviderCapabilityRegistry")
            .field("config", &self.inner.config)
            .field("has_probe", &self.inner.probe.is_some())
            .finish_non_exhaustive()
    }
}

impl ProviderCapabilityRegistry {
    pub fn new(
        mut config: CapabilityRegistryConfig,
        probe: Option<Arc<dyn CapabilityProbe>>,
    ) -> Self {
        config.registry_version = config.registry_version.max(1);
        config.probe_timeout = config.probe_timeout.min(MAX_CAPABILITY_PROBE_TIMEOUT);
        config.max_probe_members = config
            .max_probe_members
            .clamp(1, MAX_CAPABILITY_PROBE_MEMBERS);
        config.max_audit_records = config.max_audit_records.clamp(1, MAX_AUDIT_CAPACITY);
        Self {
            inner: Arc::new(RegistryInner {
                config,
                probe,
                registered: Mutex::new(HashMap::new()),
                cache: Mutex::new(HashMap::new()),
                audit: Mutex::new(VecDeque::new()),
            }),
        }
    }

    /// Install a high-confidence local declaration used when probing is absent,
    /// inconclusive, or fails. Registration never initiates provider I/O.
    pub fn register(&self, target: CapabilityRoute, capabilities: ProviderCapabilities) {
        lock(&self.inner.registered).insert(target.clone(), capabilities);
        lock(&self.inner.cache).remove(&target);
    }

    /// Seed a freshly completed external discovery into the same versioned TTL
    /// cache used by runtime probes. No provider call is made here.
    pub fn seed_observation(
        &self,
        target: CapabilityRoute,
        declared: ProviderCapabilities,
        observation: CapabilityProbeObservation,
    ) {
        self.seed_observation_at(target, declared, observation, unix_time_ms());
    }

    /// Explicit-time form for deterministic cache restoration and tests.
    pub fn seed_observation_at(
        &self,
        target: CapabilityRoute,
        declared: ProviderCapabilities,
        observation: CapabilityProbeObservation,
        observed_at_ms: u64,
    ) {
        let capabilities = normalized_observation(observation);
        let cached = CachedProbe {
            registry_version: self.inner.config.registry_version,
            expires_at_ms: observed_at_ms.saturating_add(duration_ms(self.inner.config.cache_ttl)),
            declared_digest: declared.canonical_digest(),
            observed_digest: Some(capabilities.canonical_digest()),
            capabilities,
            disposition: CapabilityProbeDisposition::Succeeded,
            detail: Some("seeded from bounded live model metadata".to_string()),
        };
        lock(&self.inner.cache).insert(target, cached);
    }

    pub fn audit_records(&self) -> Vec<CapabilityProbeAuditRecord> {
        lock(&self.inner.audit).iter().cloned().collect()
    }

    pub async fn resolve(
        &self,
        target: CapabilityRoute,
        declared: ProviderCapabilities,
    ) -> EffectiveProviderCapabilities {
        self.resolve_candidates(
            target.clone(),
            &[ProviderCapabilityCandidate::new(target, declared)],
        )
        .await
    }

    pub async fn resolve_candidates(
        &self,
        target: CapabilityRoute,
        candidates: &[ProviderCapabilityCandidate],
    ) -> EffectiveProviderCapabilities {
        self.resolve_candidates_at(target, candidates, unix_time_ms())
            .await
    }

    /// Explicit-time form for deterministic replay and TTL tests.
    pub async fn resolve_candidates_at(
        &self,
        target: CapabilityRoute,
        candidates: &[ProviderCapabilityCandidate],
        now_ms: u64,
    ) -> EffectiveProviderCapabilities {
        let candidates = canonical_candidates(&target, candidates);
        let resolved = join_all(
            candidates
                .into_iter()
                .enumerate()
                .map(|(index, candidate)| {
                    self.resolve_member(
                        candidate,
                        now_ms,
                        index < self.inner.config.max_probe_members,
                    )
                }),
        )
        .await;
        let capabilities = resolved
            .iter()
            .map(|member| member.capabilities.clone())
            .reduce(|left, right| left.conservative_intersection(&right))
            .unwrap_or_default();
        let effective = EffectiveProviderCapabilities {
            schema_version: CAPABILITY_RECORD_SCHEMA_VERSION,
            registry_version: self.inner.config.registry_version,
            target: target.clone(),
            capabilities,
            members: resolved
                .iter()
                .map(|member| member.record.clone())
                .collect(),
        };
        let effective_digest = effective.canonical_digest();
        for member in resolved {
            self.record_audit(&target, member, &effective_digest, now_ms);
        }
        effective
    }

    async fn resolve_member(
        &self,
        candidate: ProviderCapabilityCandidate,
        now_ms: u64,
        probe_allowed: bool,
    ) -> ResolvedMember {
        let declared = lock(&self.inner.registered)
            .get(&candidate.target)
            .cloned()
            .unwrap_or(candidate.declared);
        let declared_digest = declared.canonical_digest();
        if let Some(cached) = lock(&self.inner.cache).get(&candidate.target).cloned()
            && cached.registry_version == self.inner.config.registry_version
            && cached.declared_digest == declared_digest
            && now_ms < cached.expires_at_ms
        {
            return resolved_from_cache(candidate.target, cached);
        }
        let expires_at_ms = now_ms.saturating_add(duration_ms(self.inner.config.cache_ttl));
        if !probe_allowed {
            return ResolvedMember {
                record: CapabilityMemberRecord {
                    target: candidate.target,
                    disposition: CapabilityProbeDisposition::SkippedLimit,
                    declared_digest,
                    observed_digest: None,
                    actual_model_revision: declared.actual_model_revision.clone(),
                },
                capabilities: declared,
                cache_hit: false,
                expires_at_ms,
                detail: Some("probe skipped at configured route-member limit".to_string()),
            };
        }
        let Some(probe) = &self.inner.probe else {
            return ResolvedMember {
                record: CapabilityMemberRecord {
                    target: candidate.target,
                    disposition: CapabilityProbeDisposition::NotConfigured,
                    declared_digest,
                    observed_digest: None,
                    actual_model_revision: declared.actual_model_revision.clone(),
                },
                capabilities: declared,
                cache_hit: false,
                expires_at_ms,
                detail: None,
            };
        };
        let result = tokio::time::timeout(
            self.inner.config.probe_timeout,
            probe.probe(&candidate.target),
        )
        .await;
        let (capabilities, disposition, observed_digest, detail) = match result {
            Ok(Ok(Some(observed))) => {
                let capabilities = normalized_observation(observed);
                let digest = capabilities.canonical_digest();
                (
                    capabilities,
                    CapabilityProbeDisposition::Succeeded,
                    Some(digest),
                    None,
                )
            }
            Ok(Ok(None)) => (
                declared.clone(),
                CapabilityProbeDisposition::Unknown,
                None,
                Some("provider returned no capability observation".to_string()),
            ),
            Ok(Err(error)) => (
                declared.clone(),
                CapabilityProbeDisposition::Failed,
                None,
                Some(bounded_detail(format!("{error:#}"))),
            ),
            Err(_) => (
                declared.clone(),
                CapabilityProbeDisposition::TimedOut,
                None,
                Some(format!(
                    "probe exceeded {} ms deadline",
                    duration_ms(self.inner.config.probe_timeout)
                )),
            ),
        };
        let cached = CachedProbe {
            registry_version: self.inner.config.registry_version,
            expires_at_ms,
            declared_digest: declared_digest.clone(),
            capabilities: capabilities.clone(),
            disposition,
            observed_digest: observed_digest.clone(),
            detail: detail.clone(),
        };
        lock(&self.inner.cache).insert(candidate.target.clone(), cached);
        ResolvedMember {
            record: CapabilityMemberRecord {
                target: candidate.target,
                disposition,
                declared_digest,
                observed_digest,
                actual_model_revision: capabilities.actual_model_revision.clone(),
            },
            capabilities,
            cache_hit: false,
            expires_at_ms,
            detail,
        }
    }

    fn record_audit(
        &self,
        target: &CapabilityRoute,
        member: ResolvedMember,
        effective_digest: &str,
        now_ms: u64,
    ) {
        let audit_digest = canonical_digest(&(
            CAPABILITY_RECORD_SCHEMA_VERSION,
            self.inner.config.registry_version,
            target,
            &member.record.target,
            member.record.disposition,
            member.cache_hit,
            effective_digest,
            &member.record.observed_digest,
        ));
        let record = CapabilityProbeAuditRecord {
            schema_version: CAPABILITY_RECORD_SCHEMA_VERSION,
            registry_version: self.inner.config.registry_version,
            target: target.clone(),
            member: member.record.target,
            disposition: member.record.disposition,
            cache_hit: member.cache_hit,
            effective_digest: effective_digest.to_string(),
            observed_digest: member.record.observed_digest,
            member_capabilities: member.capabilities,
            recorded_at_ms: now_ms,
            expires_at_ms: member.expires_at_ms,
            detail: member.detail,
            audit_digest,
        };
        let mut audit = lock(&self.inner.audit);
        while audit.len() >= self.inner.config.max_audit_records {
            audit.pop_front();
        }
        audit.push_back(record);
    }
}

fn resolved_from_cache(target: CapabilityRoute, cached: CachedProbe) -> ResolvedMember {
    ResolvedMember {
        record: CapabilityMemberRecord {
            target,
            disposition: cached.disposition,
            declared_digest: cached.declared_digest,
            observed_digest: cached.observed_digest,
            actual_model_revision: cached.capabilities.actual_model_revision.clone(),
        },
        capabilities: cached.capabilities,
        cache_hit: true,
        expires_at_ms: cached.expires_at_ms,
        detail: cached.detail,
    }
}

fn canonical_candidates(
    target: &CapabilityRoute,
    candidates: &[ProviderCapabilityCandidate],
) -> Vec<ProviderCapabilityCandidate> {
    let mut canonical = BTreeMap::<CapabilityRoute, ProviderCapabilities>::new();
    for candidate in candidates {
        canonical
            .entry(candidate.target.clone())
            .and_modify(|known| *known = known.conservative_intersection(&candidate.declared))
            .or_insert_with(|| candidate.declared.clone());
    }
    if canonical.is_empty() {
        canonical.insert(target.clone(), ProviderCapabilities::default());
    }
    canonical
        .into_iter()
        .map(|(target, declared)| ProviderCapabilityCandidate { target, declared })
        .collect()
}

fn normalized_identity(value: String) -> String {
    let value = value.trim();
    if value.is_empty() {
        "unknown".to_string()
    } else {
        value.to_string()
    }
}

fn normalized_revision(revision: Option<String>) -> Option<String> {
    revision.and_then(|revision| {
        let revision = revision.trim();
        (!revision.is_empty()).then(|| revision.chars().take(256).collect())
    })
}

fn normalized_observation(mut observation: CapabilityProbeObservation) -> ProviderCapabilities {
    observation.capabilities.actual_model_revision =
        normalized_revision(observation.actual_model_revision)
            .or_else(|| normalized_revision(observation.capabilities.actual_model_revision.take()));
    observation.capabilities
}

fn bounded_detail(detail: String) -> String {
    detail.chars().take(MAX_AUDIT_DETAIL_CHARS).collect()
}

fn canonical_digest(value: &impl Serialize) -> String {
    let encoded = serde_json::to_vec(value).expect("capability audit value serializes");
    format!("blake3:{}", blake3::hash(&encoded).to_hex())
}

fn duration_ms(duration: Duration) -> u64 {
    u64::try_from(duration.as_millis()).unwrap_or(u64::MAX)
}

fn unix_time_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, duration_ms)
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}
