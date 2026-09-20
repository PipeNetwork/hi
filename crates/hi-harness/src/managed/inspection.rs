//! Read-only guest recovery inspection. This never submits or retries inference.
use super::*;
use sha2::{Digest, Sha256};
use std::{io::Read, path::Path};

#[derive(Clone, Debug, Serialize)]
pub struct ManagedInspection {
    pub journal_blake3: String,
    pub journal_sha256: String,
    pub ambiguous_tools: usize,
    pub unretained_responses: usize,
    pub unresolved_calls: usize,
    pub can_resume: bool,
    /// Original idempotency key of the last accepted non-auxiliary final answer.
    /// The control plane must look it up under the exact run credential.
    pub accepted_final_key: Option<String>,
}

pub fn inspect_managed_journal(
    path: &Path,
    endpoint: &str,
    credential: &str,
    settings: &ManagedSettings,
) -> Result<ManagedInspection> {
    let _lease = crate::session_lease::SessionLease::acquire(path)?;
    let mut bytes = Vec::new();
    fs::File::open(path)?
        .take(128 * 1024 * 1024 + 1)
        .read_to_end(&mut bytes)?;
    ensure!(
        bytes.len() <= 128 * 1024 * 1024,
        "managed recovery journal exceeds inspection bound"
    );
    let journal: Journal = serde_json::from_slice(&bytes)?;
    ensure!(
        journal.version == 1
            && journal.endpoint == endpoint
            && journal.credential_hash == blake3::hash(credential.as_bytes()).to_hex().as_str()
            && journal.budget == micros(&settings.turn_budget_usd)?
            && serde_json::to_value(&journal.settings)? == serde_json::to_value(settings)?,
        "managed recovery requires the original credential, endpoint and budgets"
    );
    let ambiguous_tools = journal
        .operations
        .iter()
        .flat_map(|o| o.tools.values())
        .filter(|state| matches!(state, ToolState::Started))
        .count();
    let unretained_responses = journal
        .operations
        .iter()
        .filter(|o| o.completion.is_none() && o.status != "prepared")
        .count();
    let unresolved_calls = journal
        .operations
        .iter()
        .filter(|o| o.unresolved && o.status != "prepared")
        .count();
    let can_resume = ambiguous_tools == 0 && unretained_responses == 0 && unresolved_calls == 0;
    let accepted_final_key = journal
        .operations
        .iter()
        .rev()
        .find(|o| !o.auxiliary)
        .filter(|o| {
            can_resume
                && !o.unresolved
                && o.charge.is_some()
                && o.status == "completed"
                && o.tools.is_empty()
                && o.completion
                    .as_ref()
                    .is_some_and(|c| c.tool_calls.is_empty())
        })
        .map(|o| o.key.clone());
    Ok(ManagedInspection {
        journal_blake3: blake3::hash(&bytes).to_hex().to_string(),
        journal_sha256: format!("{:x}", Sha256::digest(&bytes)),
        ambiguous_tools,
        unretained_responses,
        unresolved_calls,
        can_resume,
        accepted_final_key,
    })
}
