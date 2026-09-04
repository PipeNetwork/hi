use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::anyhow;

use super::*;
use crate::{Backend, ChatRequest, Completion, FallbackProvider, Provider, StreamEvent};

#[derive(Clone)]
enum Reply {
    Capabilities(ProviderCapabilities, &'static str),
    Unknown,
    Failure,
    Delay(Duration),
}

struct FakeProbe {
    replies: Mutex<HashMap<CapabilityRoute, Reply>>,
    calls: AtomicUsize,
}

struct StaticProvider(ProviderCapabilities);

#[async_trait::async_trait]
impl Provider for StaticProvider {
    async fn stream(
        &self,
        _request: ChatRequest,
        _sink: &mut (dyn FnMut(StreamEvent) + Send),
    ) -> anyhow::Result<Completion> {
        unreachable!("capability test does not perform inference")
    }

    fn capabilities(&self) -> ProviderCapabilities {
        self.0.clone()
    }
}

impl FakeProbe {
    fn new(replies: impl IntoIterator<Item = (CapabilityRoute, Reply)>) -> Arc<Self> {
        Arc::new(Self {
            replies: Mutex::new(replies.into_iter().collect()),
            calls: AtomicUsize::new(0),
        })
    }
}

#[async_trait::async_trait]
impl CapabilityProbe for FakeProbe {
    async fn probe(
        &self,
        target: &CapabilityRoute,
    ) -> anyhow::Result<Option<CapabilityProbeObservation>> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        let reply = self
            .replies
            .lock()
            .unwrap()
            .get(target)
            .cloned()
            .unwrap_or(Reply::Unknown);
        match reply {
            Reply::Capabilities(capabilities, revision) => Ok(Some(CapabilityProbeObservation {
                capabilities,
                actual_model_revision: Some(revision.to_string()),
            })),
            Reply::Unknown => Ok(None),
            Reply::Failure => Err(anyhow!("bounded fake probe failed")),
            Reply::Delay(delay) => {
                tokio::time::sleep(delay).await;
                Ok(None)
            }
        }
    }
}

fn target(route: &str, model: &str) -> CapabilityRoute {
    CapabilityRoute::new(route, model)
}

fn capable(output_limit: u32) -> ProviderCapabilities {
    let mut capabilities = ProviderCapabilities::native_tools(true);
    capabilities.parallel_tool_calls = true;
    capabilities.tool_choice.automatic = true;
    capabilities.request_limits.max_output_tokens = Some(output_limit);
    capabilities
}

fn config(ttl_ms: u64, timeout_ms: u64) -> CapabilityRegistryConfig {
    CapabilityRegistryConfig {
        cache_ttl: Duration::from_millis(ttl_ms),
        probe_timeout: Duration::from_millis(timeout_ms),
        ..CapabilityRegistryConfig::default()
    }
}

#[tokio::test]
async fn cache_is_isolated_by_effective_route_and_model() {
    let left = target("gateway-a", "shared-name");
    let right = target("gateway-b", "shared-name");
    let other_model = target("gateway-a", "other-name");
    let probe = FakeProbe::new([
        (left.clone(), Reply::Capabilities(capable(8_192), "left@1")),
        (
            right.clone(),
            Reply::Capabilities(capable(16_384), "right@1"),
        ),
        (
            other_model.clone(),
            Reply::Capabilities(capable(32_768), "other@1"),
        ),
    ]);
    let registry = ProviderCapabilityRegistry::new(config(100, 20), Some(probe.clone()));
    registry.register(left.clone(), capable(4_096));

    let left_record = registry
        .resolve_candidates_at(left.clone(), &[], 1_000)
        .await;
    let right_record = registry
        .resolve_candidates_at(right.clone(), &[], 1_000)
        .await;
    let other_record = registry
        .resolve_candidates_at(other_model, &[], 1_000)
        .await;
    let left_cached = registry.resolve_candidates_at(left, &[], 1_001).await;

    assert_eq!(probe.calls.load(Ordering::SeqCst), 3);
    assert_eq!(left_record.actual_model_revision(), Some("left@1"));
    assert_eq!(right_record.actual_model_revision(), Some("right@1"));
    assert_eq!(other_record.actual_model_revision(), Some("other@1"));
    assert_eq!(
        left_record.canonical_digest(),
        left_cached.canonical_digest()
    );
    assert!(registry.audit_records().last().unwrap().cache_hit);
}

#[tokio::test]
async fn cache_ttl_expiry_reprobes_at_the_exact_boundary() {
    let route = target("gateway", "model");
    let probe = FakeProbe::new([(
        route.clone(),
        Reply::Capabilities(capable(8_192), "model@1"),
    )]);
    let registry = ProviderCapabilityRegistry::new(config(10, 20), Some(probe.clone()));

    registry
        .resolve_candidates_at(route.clone(), &[], 100)
        .await;
    registry
        .resolve_candidates_at(route.clone(), &[], 109)
        .await;
    registry.resolve_candidates_at(route, &[], 110).await;

    assert_eq!(probe.calls.load(Ordering::SeqCst), 2);
    let hits = registry
        .audit_records()
        .into_iter()
        .map(|record| record.cache_hit)
        .collect::<Vec<_>>();
    assert_eq!(hits, [false, true, false]);
}

#[tokio::test]
async fn seeded_live_metadata_uses_the_same_ttl_and_conservative_expiry() {
    let route = target("gateway", "model");
    let registry = ProviderCapabilityRegistry::new(config(10, 20), None);
    let declared = ProviderCapabilities::default();
    registry.seed_observation_at(
        route.clone(),
        declared.clone(),
        CapabilityProbeObservation {
            capabilities: capable(8_192),
            actual_model_revision: Some("model@live".into()),
        },
        100,
    );

    let live = registry
        .resolve_candidates_at(
            route.clone(),
            &[ProviderCapabilityCandidate::new(
                route.clone(),
                declared.clone(),
            )],
            109,
        )
        .await;
    let expired = registry
        .resolve_candidates_at(
            route.clone(),
            &[ProviderCapabilityCandidate::new(route, declared.clone())],
            110,
        )
        .await;

    assert_eq!(live.actual_model_revision(), Some("model@live"));
    assert_eq!(expired.capabilities, declared);
    assert_eq!(
        expired.members[0].disposition,
        CapabilityProbeDisposition::NotConfigured
    );
}

#[tokio::test]
async fn unknown_timeout_and_failure_fall_back_to_declared_capabilities() {
    let unknown = target("unknown", "model");
    let slow = target("slow", "model");
    let broken = target("broken", "model");
    let probe = FakeProbe::new([
        (slow.clone(), Reply::Delay(Duration::from_millis(100))),
        (broken.clone(), Reply::Failure),
    ]);
    let registry = ProviderCapabilityRegistry::new(config(100, 2), Some(probe));
    let mut conservative = ProviderCapabilities::default();
    conservative.request_limits.max_output_tokens = Some(4_096);

    let unknown_record = registry
        .resolve_candidates_at(
            unknown.clone(),
            &[ProviderCapabilityCandidate::new(
                unknown,
                conservative.clone(),
            )],
            1,
        )
        .await;
    let timed_out = registry
        .resolve_candidates_at(
            slow.clone(),
            &[ProviderCapabilityCandidate::new(slow, conservative.clone())],
            1,
        )
        .await;
    let failed = registry
        .resolve_candidates_at(
            broken.clone(),
            &[ProviderCapabilityCandidate::new(
                broken,
                conservative.clone(),
            )],
            1,
        )
        .await;

    assert_eq!(unknown_record.capabilities, conservative);
    assert_eq!(timed_out.capabilities, conservative);
    assert_eq!(failed.capabilities, conservative);
    assert_eq!(
        unknown_record.members[0].disposition,
        CapabilityProbeDisposition::Unknown
    );
    assert_eq!(
        timed_out.members[0].disposition,
        CapabilityProbeDisposition::TimedOut
    );
    assert_eq!(
        failed.members[0].disposition,
        CapabilityProbeDisposition::Failed
    );
}

#[tokio::test]
async fn fallback_route_uses_canonical_conservative_intersection() {
    let effective = target("fallback", "requested");
    let primary = target("primary", "model-a");
    let secondary = target("secondary", "model-b");
    let mut left = capable(16_384);
    left.modalities.image_input = true;
    let mut right = capable(8_192);
    right.parallel_tool_calls = false;
    let provider = FallbackProvider::new(vec![
        Backend {
            provider: Box::new(StaticProvider(right)),
            model: secondary.model,
            label: secondary.route,
        },
        Backend {
            provider: Box::new(StaticProvider(left)),
            model: primary.model,
            label: primary.route,
        },
    ])
    .unwrap();
    let candidates = provider.capability_candidates("fallback", "requested");
    let registry = ProviderCapabilityRegistry::default();

    let record = registry
        .resolve_candidates_at(effective, &candidates, 7)
        .await;

    assert!(!record.capabilities.parallel_tool_calls);
    assert!(!record.capabilities.modalities.image_input);
    assert_eq!(
        record.capabilities.request_limits.max_output_tokens,
        Some(8_192)
    );
    assert_eq!(record.members[0].target.route, "primary");
    assert_eq!(record.members[1].target.route, "secondary");
}

#[tokio::test]
async fn effective_and_audit_digests_are_stable_for_equal_observations() {
    async fn resolve_once() -> (EffectiveProviderCapabilities, CapabilityProbeAuditRecord) {
        let route = target("gateway", "model");
        let probe = FakeProbe::new([(
            route.clone(),
            Reply::Capabilities(capable(8_192), "model@sha256:abc"),
        )]);
        let registry = ProviderCapabilityRegistry::new(config(100, 20), Some(probe));
        let record = registry.resolve_candidates_at(route, &[], 42).await;
        let audit = registry.audit_records().pop().unwrap();
        (record, audit)
    }

    let (first, first_audit) = resolve_once().await;
    let (second, second_audit) = resolve_once().await;
    assert_eq!(first, second);
    assert_eq!(first.canonical_digest(), second.canonical_digest());
    assert_eq!(first_audit.audit_digest, second_audit.audit_digest);
    assert_eq!(first_audit.effective_digest, first.canonical_digest());
}

#[tokio::test]
async fn registry_version_is_explicit_and_digest_significant() {
    let route = target("gateway", "model");
    let first = ProviderCapabilityRegistry::new(config(100, 20), None)
        .resolve(route.clone(), capable(8_192))
        .await;
    let second = ProviderCapabilityRegistry::new(
        CapabilityRegistryConfig {
            registry_version: CAPABILITY_REGISTRY_VERSION + 1,
            ..config(100, 20)
        },
        None,
    )
    .resolve(route, capable(8_192))
    .await;

    assert_eq!(first.registry_version, CAPABILITY_REGISTRY_VERSION);
    assert_eq!(second.registry_version, CAPABILITY_REGISTRY_VERSION + 1);
    assert_ne!(first.canonical_digest(), second.canonical_digest());
}
