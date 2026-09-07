//! V1 wire retention is limited to integrity checking. The session reducer
//! migrates it after checking the enclosing snapshot; no replay rules live here.
use super::*;
use serde::ser::SerializeMap;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct LegacyRecoveryWire(serde_json::Value);

const FIELDS: &[&str] = &[
    "schema_version",
    "objective",
    "limit",
    "remaining",
    "exhausted",
    "interventions",
    "last_reason",
    "pending_validation_correction",
    "completed_effects",
    "mutation_credited",
    "validations",
    "observed_executions",
];
const FRONTIER_FIELDS: &[&str] = &[
    "best_failures",
    "passed",
    "failed_states",
    "diagnostic_states",
    "current_failure",
];

struct OrderedFields<'a> {
    value: &'a serde_json::Value,
    fields: &'a [&'a str],
}
impl Serialize for OrderedFields<'_> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut map = serializer.serialize_map(None)?;
        for field in self.fields {
            if let Some(value) = self.value.get(field) {
                if *field == "validations" {
                    map.serialize_entry(field, &Frontiers(value))?;
                } else {
                    map.serialize_entry(field, value)?;
                }
            }
        }
        map.end()
    }
}
struct Frontiers<'a>(&'a serde_json::Value);
impl Serialize for Frontiers<'_> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut map = serializer.serialize_map(None)?;
        if let Some(frontiers) = self.0.as_object() {
            for (scope, value) in frontiers {
                map.serialize_entry(
                    scope,
                    &OrderedFields {
                        value,
                        fields: FRONTIER_FIELDS,
                    },
                )?;
            }
        }
        map.end()
    }
}

impl Serialize for TaskRecoveryState {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        if let Some(wire) = &self.legacy_wire {
            let mut map = serializer.serialize_map(None)?;
            for field in FIELDS {
                let Some(original) = wire.0.get(field) else {
                    continue;
                };
                // Public scalar edits must affect the digest even before migration.
                macro_rules! scalar {
                    ($name:ident) => {
                        map.serialize_entry(field, &self.$name)?
                    };
                }
                match *field {
                    "schema_version" => scalar!(schema_version),
                    "objective" => scalar!(objective),
                    "limit" => scalar!(limit),
                    "remaining" => scalar!(remaining),
                    "exhausted" => scalar!(exhausted),
                    "interventions" => scalar!(interventions),
                    "last_reason" => scalar!(last_reason),
                    "validations" => map.serialize_entry(field, &Frontiers(original))?,
                    _ => map.serialize_entry(field, original)?,
                }
            }
            return map.end();
        }
        let mut map = serializer.serialize_map(None)?;
        macro_rules! field {
            ($name:ident) => {
                map.serialize_entry(stringify!($name), &self.$name)?
            };
        }
        field!(schema_version);
        field!(objective);
        field!(limit);
        field!(remaining);
        field!(exhausted);
        field!(interventions);
        field!(last_reason);
        field!(pending_validation_correction);
        field!(completed_effects);
        field!(mutation_credited);
        field!(validations);
        map.end()
    }
}

impl<'de> Deserialize<'de> for TaskRecoveryState {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        struct Fields {
            schema_version: u16,
            objective: String,
            limit: u32,
            remaining: u32,
            exhausted: bool,
            interventions: u64,
            last_reason: Option<String>,
            #[serde(default)]
            pending_validation_correction: bool,
            #[serde(default)]
            completed_effects: BTreeSet<String>,
            mutation_credited: bool,
            validations: BTreeMap<String, ValidationFrontier>,
            #[serde(default)]
            observed_executions: VecDeque<String>,
        }
        let value = serde_json::Value::deserialize(deserializer)?;
        if value
            .get("schema_version")
            .and_then(serde_json::Value::as_u64)
            == Some(2)
            && value
                .get("validations")
                .and_then(serde_json::Value::as_object)
                .is_some_and(|frontiers| {
                    frontiers
                        .values()
                        .any(|frontier| frontier.get("current_failure").is_none())
                })
        {
            return Err(serde::de::Error::custom(
                "task recovery v2 frontier is missing current failure state",
            ));
        }
        let fields: Fields =
            serde_json::from_value(value.clone()).map_err(serde::de::Error::custom)?;
        Ok(Self {
            schema_version: fields.schema_version,
            objective: fields.objective,
            limit: fields.limit,
            remaining: fields.remaining,
            exhausted: fields.exhausted,
            interventions: fields.interventions,
            last_reason: fields.last_reason,
            pending_validation_correction: fields.pending_validation_correction,
            completed_effects: fields.completed_effects,
            mutation_credited: fields.mutation_credited,
            validations: fields.validations,
            observed_executions: fields
                .observed_executions
                .into_iter()
                .take(COMPLETED_OBSERVATIONS)
                .collect(),
            legacy_wire: (fields.schema_version == 1).then_some(LegacyRecoveryWire(value)),
        })
    }
}
