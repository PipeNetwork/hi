use std::{
    collections::BTreeMap,
    fs::File,
    io::Read,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, bail, ensure};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{LOCAL_PROTOCOL_MAJOR_VERSION, RUNTIME_DESCRIPTOR_SCHEMA_VERSION};

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CandidateIdentity {
    pub run_id: String,
    pub task_id: String,
    pub candidate_id: String,
    pub manifest_hash: String,
    pub agent_artifact_hash: String,
    pub repository_snapshot_hash: String,
    pub source_repository: String,
    pub source_commit: String,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MutationLevel {
    Prompts,
    ModelRouting,
    ContextAndMemory,
    Workflow,
    ToolComposition,
    AgentSource,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum IsolationProfile {
    /// Dedicated unprivileged identity, user/mount/PID/network namespaces,
    /// cgroups v2, seccomp, disposable worktree, and default-deny egress.
    Namespace,
    MicroVm,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeBudgets {
    pub wall_time_seconds: u64,
    pub cpu_time_seconds: u64,
    pub memory_bytes: u64,
    pub disk_bytes: u64,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub tool_calls: u64,
    pub cost_microusd: u64,
    pub model_calls: u32,
    pub repair_iterations: u32,
    pub trace_bytes: u64,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimePolicy {
    pub task_policy_version: String,
    pub mutation_level: MutationLevel,
    pub workflow_entrypoint: String,
    pub model_role: String,
    pub tool_set: String,
    pub tool_mode: String,
    pub filesystem_mode: String,
    pub allowed_tools: Vec<String>,
    pub network_allowlist: Vec<String>,
    pub isolation: IsolationProfile,
    pub trusted_launcher: bool,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ManagedRuntimeDescriptor {
    pub schema_version: u16,
    pub protocol_major: u16,
    pub identity: CandidateIdentity,
    pub budgets: RuntimeBudgets,
    pub policy: RuntimePolicy,
    /// Complete worker-issued, manifest-derived execution package. Protocol-v1
    /// workers omit it and use the one-shot compatibility adapter.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub runtime_package: Option<RuntimePackage>,
    pub issued_at_unix_ms: u64,
    pub expires_at_unix_ms: u64,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimePackage {
    pub workflow: Value,
    pub model_routes: Value,
    pub context_policy: Value,
    pub memory_policy: Value,
    pub tool_policy: Value,
    pub component_hashes: BTreeMap<String, String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EffectiveRuntime<'a> {
    pub model_role: &'a str,
    pub max_model_calls: u32,
    pub max_tool_calls: u32,
    pub max_output_tokens: u32,
    pub max_repair_iterations: u32,
    pub trace_bytes: u64,
    pub tool_set: &'a str,
    pub tool_mode: &'a str,
}

impl ManagedRuntimeDescriptor {
    pub fn read(path: &Path, now_unix_ms: u64) -> Result<Self> {
        let before = std::fs::symlink_metadata(path)
            .with_context(|| format!("inspecting RSI runtime descriptor {}", path.display()))?;
        ensure!(
            before.is_file() && !before.file_type().is_symlink(),
            "RSI runtime descriptor must be a regular file"
        );
        ensure!(
            before.len() <= 256 * 1024,
            "RSI runtime descriptor exceeds 256 KiB"
        );
        let file = File::open(path)
            .with_context(|| format!("opening RSI runtime descriptor {}", path.display()))?;
        let opened = file.metadata()?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            ensure!(
                before.dev() == opened.dev() && before.ino() == opened.ino(),
                "RSI runtime descriptor changed before open"
            );
            ensure!(
                opened.nlink() == 1,
                "linked RSI runtime descriptor rejected"
            );
        }
        let mut bytes = Vec::with_capacity(opened.len() as usize);
        file.take(256 * 1024 + 1).read_to_end(&mut bytes)?;
        ensure!(
            bytes.len() as u64 == opened.len(),
            "RSI runtime descriptor changed while reading"
        );
        let descriptor: Self = serde_json::from_slice(&bytes)
            .context("decoding worker-provided RSI runtime descriptor")?;
        descriptor.validate(now_unix_ms)?;
        Ok(descriptor)
    }

    pub fn validate(&self, now_unix_ms: u64) -> Result<()> {
        ensure!(
            self.schema_version == RUNTIME_DESCRIPTOR_SCHEMA_VERSION,
            "unsupported RSI runtime descriptor schema {}",
            self.schema_version
        );
        ensure!(
            self.protocol_major == LOCAL_PROTOCOL_MAJOR_VERSION,
            "unsupported RSI local protocol major {}",
            self.protocol_major
        );
        ensure!(
            self.issued_at_unix_ms <= now_unix_ms,
            "RSI runtime descriptor is not active yet"
        );
        ensure!(
            self.expires_at_unix_ms > now_unix_ms,
            "RSI runtime descriptor has expired"
        );
        ensure!(
            self.expires_at_unix_ms
                .saturating_sub(self.issued_at_unix_ms)
                <= 24 * 60 * 60 * 1_000,
            "RSI runtime descriptor lifetime exceeds 24 hours"
        );
        for (name, value, maximum) in [
            ("run id", self.identity.run_id.as_str(), 128),
            ("task id", self.identity.task_id.as_str(), 128),
            ("candidate id", self.identity.candidate_id.as_str(), 128),
            (
                "source repository",
                self.identity.source_repository.as_str(),
                512,
            ),
            ("source commit", self.identity.source_commit.as_str(), 128),
            (
                "task policy version",
                self.policy.task_policy_version.as_str(),
                256,
            ),
            (
                "workflow entrypoint",
                self.policy.workflow_entrypoint.as_str(),
                128,
            ),
            ("model role", self.policy.model_role.as_str(), 128),
        ] {
            validate_label(name, value, maximum)?;
        }
        for (name, value) in [
            ("manifest hash", &self.identity.manifest_hash),
            ("agent artifact hash", &self.identity.agent_artifact_hash),
            (
                "repository snapshot hash",
                &self.identity.repository_snapshot_hash,
            ),
        ] {
            ensure!(is_hash(value), "invalid RSI {name}");
        }
        ensure!(self.budgets.wall_time_seconds > 0, "zero wall-time budget");
        ensure!(self.budgets.memory_bytes > 0, "zero memory budget");
        ensure!(self.budgets.disk_bytes > 0, "zero disk budget");
        ensure!(self.budgets.model_calls > 0, "zero model-call budget");
        ensure!(self.budgets.output_tokens > 0, "zero output-token budget");
        ensure!(self.budgets.trace_bytes > 0, "zero trace budget");
        ensure!(
            self.policy.trusted_launcher,
            "managed RSI requires the trusted isolation launcher"
        );
        if self.policy.mutation_level == MutationLevel::AgentSource
            && self.policy.isolation == IsolationProfile::Namespace
        {
            ensure!(
                self.policy.network_allowlist.is_empty(),
                "source candidates in namespace isolation require default-deny egress"
            );
            ensure!(
                self.policy.filesystem_mode == "worktree-write",
                "source candidates require a disposable worktree"
            );
        }
        validate_unique_labels("allowed tool", &self.policy.allowed_tools)?;
        validate_unique_labels("network destination", &self.policy.network_allowlist)?;
        if let Some(package) = &self.runtime_package {
            for required in [
                "workflow",
                "model_routes",
                "context_policy",
                "memory_policy",
                "tool_policy",
            ] {
                let hash = package
                    .component_hashes
                    .get(required)
                    .ok_or_else(|| anyhow::anyhow!("runtime package omits {required} hash"))?;
                ensure!(is_hash(hash), "invalid runtime package {required} hash");
                let value = match required {
                    "workflow" => &package.workflow,
                    "model_routes" => &package.model_routes,
                    "context_policy" => &package.context_policy,
                    "memory_policy" => &package.memory_policy,
                    _ => &package.tool_policy,
                };
                ensure!(
                    blake3::hash(&serde_json::to_vec(value)?).to_hex().as_str() == hash,
                    "runtime package {required} hash mismatch"
                );
            }
        }
        Ok(())
    }

    pub fn bind_effective(&self, effective: &EffectiveRuntime<'_>) -> Result<()> {
        ensure!(
            effective.model_role == self.policy.model_role,
            "managed model role does not match the worker descriptor"
        );
        ensure!(
            u64::from(effective.max_model_calls) <= u64::from(self.budgets.model_calls),
            "managed model-call limit exceeds the worker descriptor"
        );
        ensure!(
            u64::from(effective.max_tool_calls) <= self.budgets.tool_calls,
            "managed tool-call limit exceeds the worker descriptor"
        );
        ensure!(
            u64::from(effective.max_output_tokens) <= self.budgets.output_tokens,
            "managed output-token limit exceeds the worker descriptor"
        );
        ensure!(
            effective.max_repair_iterations <= self.budgets.repair_iterations,
            "managed repair limit exceeds the worker descriptor"
        );
        ensure!(
            effective.trace_bytes == self.budgets.trace_bytes,
            "managed trace limit does not match the worker descriptor"
        );
        ensure!(
            effective.tool_set == self.policy.tool_set
                && effective.tool_mode == self.policy.tool_mode,
            "managed tool profile does not match the worker descriptor"
        );
        Ok(())
    }

    pub fn content_hash(&self) -> Result<String> {
        // Hash the map-sorted JSON value so independent worker and harness
        // implementations do not depend on Rust struct field declaration order.
        let value = serde_json::to_value(self)?;
        Ok(blake3::hash(&serde_json::to_vec(&value)?)
            .to_hex()
            .to_string())
    }

    pub fn canonical_path(path: impl Into<PathBuf>) -> Result<PathBuf> {
        let path = path.into();
        path.canonicalize()
            .with_context(|| format!("canonicalizing RSI runtime descriptor {}", path.display()))
    }
}

fn validate_unique_labels(kind: &str, values: &[String]) -> Result<()> {
    let mut sorted = values.to_vec();
    sorted.sort();
    for value in &sorted {
        validate_label(kind, value, 512)?;
    }
    if sorted.windows(2).any(|pair| pair[0] == pair[1]) {
        bail!("duplicate RSI {kind}");
    }
    Ok(())
}

fn validate_label(name: &str, value: &str, maximum: usize) -> Result<()> {
    ensure!(
        !value.is_empty()
            && value.len() <= maximum
            && value.trim() == value
            && !value.contains(['\0', '\r', '\n']),
        "invalid RSI {name}"
    );
    Ok(())
}

fn is_hash(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn descriptor() -> ManagedRuntimeDescriptor {
        ManagedRuntimeDescriptor {
            schema_version: 1,
            protocol_major: 1,
            identity: CandidateIdentity {
                run_id: "run-1".into(),
                task_id: "task-1".into(),
                candidate_id: "candidate-1".into(),
                manifest_hash: "1".repeat(64),
                agent_artifact_hash: "2".repeat(64),
                repository_snapshot_hash: "3".repeat(64),
                source_repository: "pipe/hi".into(),
                source_commit: "abc123".into(),
            },
            budgets: RuntimeBudgets {
                wall_time_seconds: 60,
                cpu_time_seconds: 60,
                memory_bytes: 1024,
                disk_bytes: 1024,
                input_tokens: 100,
                output_tokens: 100,
                tool_calls: 10,
                cost_microusd: 100,
                model_calls: 5,
                repair_iterations: 2,
                trace_bytes: 4096,
            },
            policy: RuntimePolicy {
                task_policy_version: "task-v1".into(),
                mutation_level: MutationLevel::Workflow,
                workflow_entrypoint: "intake".into(),
                model_role: "implementer".into(),
                tool_set: "minimal".into(),
                tool_mode: "auto".into(),
                filesystem_mode: "worktree-write".into(),
                allowed_tools: vec!["read".into(), "write".into()],
                network_allowlist: vec![],
                isolation: IsolationProfile::Namespace,
                trusted_launcher: true,
            },
            runtime_package: None,
            issued_at_unix_ms: 1_000,
            expires_at_unix_ms: 2_000,
        }
    }

    #[test]
    fn validates_and_binds_effective_policy() {
        let descriptor = descriptor();
        descriptor.validate(1_500).unwrap();
        descriptor
            .bind_effective(&EffectiveRuntime {
                model_role: "implementer",
                max_model_calls: 5,
                max_tool_calls: 10,
                max_output_tokens: 100,
                max_repair_iterations: 2,
                trace_bytes: 4096,
                tool_set: "minimal",
                tool_mode: "auto",
            })
            .unwrap();
    }

    #[test]
    fn rejects_expiry_hash_and_policy_mismatch() {
        let mut descriptor = descriptor();
        assert!(descriptor.validate(2_000).is_err());
        descriptor.expires_at_unix_ms = 3_000;
        descriptor.identity.manifest_hash = "not-a-hash".into();
        assert!(descriptor.validate(1_500).is_err());
        descriptor.identity.manifest_hash = "1".repeat(64);
        assert!(
            descriptor
                .bind_effective(&EffectiveRuntime {
                    model_role: "reviewer",
                    max_model_calls: 1,
                    max_tool_calls: 1,
                    max_output_tokens: 1,
                    max_repair_iterations: 0,
                    trace_bytes: 4096,
                    tool_set: "minimal",
                    tool_mode: "auto",
                })
                .is_err()
        );
    }

    #[test]
    fn source_namespace_profile_fails_closed_without_strict_equivalent() {
        let mut descriptor = descriptor();
        descriptor.policy.mutation_level = MutationLevel::AgentSource;
        descriptor
            .policy
            .network_allowlist
            .push("example.com".into());
        assert!(descriptor.validate(1_500).is_err());
        descriptor.policy.network_allowlist.clear();
        descriptor.policy.trusted_launcher = false;
        assert!(descriptor.validate(1_500).is_err());
    }
}
