use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::platform::RunIdentity;

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct IdentityDetails {
    pub adapter_version: String,
    pub hi_binary_digest: String,
    pub provider_policy_digest: String,
    pub mcp_configuration_digest: String,
    pub secret_configuration_digest: String,
    pub runtime_identity: String,
    /// Named, non-secret inputs that decide whether two harness runs can be
    /// compared. The canonical RunIdentity digest seals this whole map.
    pub identity_dimensions: BTreeMap<String, String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum RunComparability {
    Comparable,
    Incomparable { differing_dimensions: Vec<String> },
}

impl RunComparability {
    pub fn is_comparable(&self) -> bool {
        matches!(self, Self::Comparable)
    }
}

impl RunIdentity {
    /// Compare complete run identities. A difference is evidence that the
    /// measurements are incomparable, never evidence of a regression.
    pub fn comparability_with(&self, other: &Self) -> RunComparability {
        if self.digest == other.digest {
            return RunComparability::Comparable;
        }
        let mut differing = BTreeSet::new();
        for key in self
            .identity_dimensions
            .keys()
            .chain(other.identity_dimensions.keys())
        {
            if self.identity_dimensions.get(key) != other.identity_dimensions.get(key) {
                differing.insert(key.clone());
            }
        }
        if self.profile != other.profile {
            differing.insert("profile".into());
        }
        if self.manifest_digest != other.manifest_digest {
            differing.insert("manifest".into());
        }
        if self.dataset_digests != other.dataset_digests {
            differing.insert("fixtures".into());
        }
        if self.configuration_digest != other.configuration_digest {
            differing.insert("configuration".into());
        }
        if self.models != other.models {
            differing.insert("provider_model".into());
        }
        if self.backend != other.backend {
            differing.insert("workspace_backend".into());
        }
        if self.scoring_policy_digest != other.scoring_policy_digest {
            differing.insert("scoring_policy".into());
        }
        if self.adapter_version != other.adapter_version {
            differing.insert("adapter".into());
        }
        if self.hi_binary_digest != other.hi_binary_digest {
            differing.insert("binary".into());
        }
        if self.provider_policy_digest != other.provider_policy_digest {
            differing.insert("provider_policy".into());
        }
        if self.mcp_configuration_digest != other.mcp_configuration_digest {
            differing.insert("mcp".into());
        }
        if self.secret_configuration_digest != other.secret_configuration_digest {
            differing.insert("secret_configuration".into());
        }
        if self.runtime_identity != other.runtime_identity {
            differing.insert("runtime".into());
        }
        if differing.is_empty() {
            differing.insert("legacy_or_unknown_identity_field".into());
        }
        RunComparability::Incomparable {
            differing_dimensions: differing.into_iter().collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity(dimensions: BTreeMap<String, String>) -> RunIdentity {
        RunIdentity::new_with_details(
            "profile",
            "manifest",
            BTreeMap::from([("fixture".into(), "digest".into())]),
            vec!["model".into()],
            "host",
            "scoring",
            "configuration",
            IdentityDetails {
                identity_dimensions: dimensions,
                ..IdentityDetails::default()
            },
        )
        .unwrap()
    }

    #[test]
    fn identical_identity_is_comparable() {
        let identity = identity(BTreeMap::from([("git_state".into(), "one".into())]));
        assert_eq!(
            identity.comparability_with(&identity),
            RunComparability::Comparable
        );
    }

    #[test]
    fn changed_dimension_is_incomparable_not_regressed() {
        let left = identity(BTreeMap::from([("git_state".into(), "one".into())]));
        let right = identity(BTreeMap::from([("git_state".into(), "two".into())]));
        assert_eq!(
            left.comparability_with(&right),
            RunComparability::Incomparable {
                differing_dimensions: vec!["git_state".into()]
            }
        );
    }
}
