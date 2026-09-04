use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::{OperationId, ResourceUri};

pub const TOOL_DIAGNOSTIC_SCHEMA_VERSION: u16 = 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DiagnosticSeverity {
    Info,
    Warning,
    Error,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DiagnosticRetryability {
    Never,
    Safe,
    AfterRecovery,
    UserAction,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DiagnosticExportPolicy {
    #[default]
    LocalOnly,
    ExplicitlyExportable,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DiagnosticField {
    pub value: String,
    pub sensitive: bool,
}

impl DiagnosticField {
    pub fn public(value: impl Into<String>) -> Self {
        Self {
            value: value.into(),
            sensitive: false,
        }
    }

    pub fn sensitive(value: impl Into<String>) -> Self {
        Self {
            value: value.into(),
            sensitive: true,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolDiagnostic {
    pub schema_version: u16,
    pub code: String,
    pub severity: DiagnosticSeverity,
    pub message: String,
    pub retryability: DiagnosticRetryability,
    pub operation_id: Option<OperationId>,
    pub fields: BTreeMap<String, DiagnosticField>,
    pub artifacts: Vec<ResourceUri>,
    pub export_policy: DiagnosticExportPolicy,
}

impl ToolDiagnostic {
    pub fn new(
        code: impl Into<String>,
        severity: DiagnosticSeverity,
        message: impl Into<String>,
        retryability: DiagnosticRetryability,
    ) -> Self {
        Self {
            schema_version: TOOL_DIAGNOSTIC_SCHEMA_VERSION,
            code: code.into(),
            severity,
            message: message.into(),
            retryability,
            operation_id: None,
            fields: BTreeMap::new(),
            artifacts: Vec::new(),
            export_policy: DiagnosticExportPolicy::LocalOnly,
        }
    }

    pub fn dedupe_key(&self) -> DiagnosticKey {
        DiagnosticKey {
            code: self.code.clone(),
            operation_id: self.operation_id.clone(),
            message: self.message.clone(),
        }
    }

    /// Produce the explicitly requested export form. Sensitive values are
    /// always replaced, even when the diagnostic itself is exportable.
    pub fn redacted_for_export(&self) -> Option<Self> {
        (self.export_policy == DiagnosticExportPolicy::ExplicitlyExportable).then(|| {
            let mut redacted = self.clone();
            for field in redacted.fields.values_mut() {
                if field.sensitive {
                    field.value = "[redacted]".to_owned();
                }
            }
            redacted
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct DiagnosticKey {
    code: String,
    operation_id: Option<OperationId>,
    message: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DiagnosticOccurrence {
    pub diagnostic: ToolDiagnostic,
    pub first_sequence: u64,
    pub last_sequence: u64,
    pub count: u64,
}

#[derive(Clone, Debug, Default)]
pub struct ToolDiagnosticStore {
    next_sequence: u64,
    records: BTreeMap<DiagnosticKey, DiagnosticOccurrence>,
}

impl ToolDiagnosticStore {
    pub fn record(&mut self, diagnostic: ToolDiagnostic) -> &DiagnosticOccurrence {
        self.next_sequence = self.next_sequence.saturating_add(1);
        let sequence = self.next_sequence;
        let key = diagnostic.dedupe_key();
        self.records
            .entry(key.clone())
            .and_modify(|occurrence| {
                occurrence.last_sequence = sequence;
                occurrence.count = occurrence.count.saturating_add(1);
                occurrence.diagnostic = diagnostic.clone();
            })
            .or_insert(DiagnosticOccurrence {
                diagnostic,
                first_sequence: sequence,
                last_sequence: sequence,
                count: 1,
            });
        &self.records[&key]
    }

    pub fn records(&self) -> impl Iterator<Item = &DiagnosticOccurrence> {
        self.records.values()
    }

    pub fn explicitly_exported(&self) -> Vec<ToolDiagnostic> {
        self.records
            .values()
            .filter_map(|occurrence| occurrence.diagnostic.redacted_for_export())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn diagnostic() -> ToolDiagnostic {
        let mut value = ToolDiagnostic::new(
            "workspace.settlement_pending",
            DiagnosticSeverity::Warning,
            "workspace receipt remains pending",
            DiagnosticRetryability::AfterRecovery,
        );
        value.operation_id = Some(OperationId::new("op-1"));
        value
            .fields
            .insert("token".into(), DiagnosticField::sensitive("secret"));
        value
    }

    #[test]
    fn repeated_diagnostics_are_deduplicated_by_operation() {
        let mut store = ToolDiagnosticStore::default();
        store.record(diagnostic());
        let occurrence = store.record(diagnostic());
        assert_eq!(occurrence.count, 2);
        assert_eq!(occurrence.first_sequence, 1);
        assert_eq!(occurrence.last_sequence, 2);
    }

    #[test]
    fn diagnostics_stay_local_without_explicit_export_and_are_redacted() {
        let mut store = ToolDiagnosticStore::default();
        store.record(diagnostic());
        assert!(store.explicitly_exported().is_empty());

        let mut exportable = diagnostic();
        exportable.message = "different".into();
        exportable.export_policy = DiagnosticExportPolicy::ExplicitlyExportable;
        store.record(exportable);
        let exported = store.explicitly_exported();
        assert_eq!(exported.len(), 1);
        assert_eq!(exported[0].fields["token"].value, "[redacted]");
    }
}
