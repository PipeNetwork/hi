//! Local durable managed-turn journal. Content stays on the user's machine.
use crate::{
    TurnCancellation,
    pipe::{PipeClient, PipeCompletion, PipeError, StreamDelta, ToolCall},
};
use anyhow::{Context, Result, bail, ensure};
use hi_ai::{Content, Message, ToolSpec};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    fs::{self, OpenOptions},
    io::Write,
    path::PathBuf,
    sync::Mutex,
    time::Duration,
};

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ManagedProfile {
    Fast,
    #[default]
    Balanced,
    Quality,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ManagedSettings {
    pub profile: ManagedProfile,
    pub call_budget_usd: String,
    pub turn_budget_usd: String,
    pub deadline_ms: u64,
}
impl Default for ManagedSettings {
    fn default() -> Self {
        Self {
            profile: ManagedProfile::Balanced,
            call_budget_usd: "1.000000".into(),
            turn_budget_usd: "20.000000".into(),
            deadline_ms: 180_000,
        }
    }
}
pub fn micros(value: &str) -> Result<i64> {
    let parts: Vec<_> = value.split('.').collect();
    ensure!(
        !parts.is_empty()
            && parts.len() <= 2
            && !parts[0].is_empty()
            && parts[0].bytes().all(|b| b.is_ascii_digit()),
        "invalid exact USD amount"
    );
    let frac = parts.get(1).copied().unwrap_or("");
    ensure!(
        (parts.len() == 1 || !frac.is_empty())
            && frac.len() <= 6
            && frac.bytes().all(|b| b.is_ascii_digit()),
        "USD amounts allow at most six decimal places"
    );
    let whole: i64 = parts[0].parse()?;
    whole
        .checked_mul(1_000_000)
        .and_then(|n| {
            n.checked_add(frac.parse::<i64>().unwrap_or(0) * 10i64.pow(6 - frac.len() as u32))
        })
        .context("USD amount overflow")
}
fn usd(value: i64) -> String {
    format!("{}.{:06}", value / 1_000_000, value % 1_000_000)
}
impl ManagedSettings {
    fn validate(&self) -> Result<()> {
        ensure!(
            micros(&self.call_budget_usd)? > 0
                && micros(&self.turn_budget_usd)? > 0
                && (100..=180000).contains(&self.deadline_ms),
            "managed budgets must be positive and deadline at most 180000ms"
        );
        Ok(())
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
struct Operation {
    #[serde(default)]
    auxiliary: bool,
    key: String,
    input_hash: String,
    payload_hash: String,
    payload: String,
    reserved: i64,
    charge: Option<i64>,
    unresolved: bool,
    status: String,
    completion: Option<PipeCompletion>,
    tools: BTreeMap<String, ToolState>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case")]
enum ToolState {
    Pending,
    Started,
    Completed { outcome: hi_tools::ToolOutcome },
}
#[derive(Clone, Debug, Serialize, Deserialize)]
struct Journal {
    version: u32,
    turn: String,
    endpoint: String,
    credential_hash: String,
    budget: i64,
    settings: ManagedSettings,
    operations: Vec<Operation>,
}
pub(crate) struct ManagedState {
    path: Option<PathBuf>,
    settings: ManagedSettings,
    journal: Mutex<Option<Journal>>,
    lease: Mutex<Option<crate::session_lease::SessionLease>>,
    replay_final: Mutex<Option<PipeCompletion>>,
}
impl Default for ManagedState {
    fn default() -> Self {
        Self {
            path: None,
            settings: ManagedSettings::default(),
            journal: Mutex::new(None),
            lease: Mutex::new(None),
            replay_final: Mutex::new(None),
        }
    }
}
impl ManagedState {
    pub fn configured(path: PathBuf, settings: ManagedSettings) -> Self {
        Self {
            path: Some(path),
            settings,
            journal: Mutex::new(None),
            lease: Mutex::new(None),
            replay_final: Mutex::new(None),
        }
    }
    fn save(&self, journal: &Journal) -> Result<()> {
        let path = self
            .path
            .as_ref()
            .context("managed coding requires a durable local journal")?;
        let parent = path.parent().context("managed journal directory")?;
        fs::create_dir_all(parent)?;
        let tmp = path.with_extension(format!("{}.tmp", uuid::Uuid::new_v4()));
        let mut options = OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        let mut file = options.open(&tmp)?;
        file.write_all(&serde_json::to_vec(journal)?)?;
        file.sync_all()?;
        fs::rename(&tmp, path)?;
        fs::File::open(parent)?.sync_all()?;
        Ok(())
    }
    fn update<T>(&self, f: impl FnOnce(&mut Journal) -> Result<T>) -> Result<T> {
        let mut guard = self
            .journal
            .lock()
            .map_err(|_| anyhow::anyhow!("managed journal lock failed"))?;
        let journal = guard
            .as_mut()
            .context("managed turn has not been initialized")?;
        let result = f(journal)?;
        self.save(journal)?;
        Ok(result)
    }
    fn read<T>(&self, f: impl FnOnce(&Journal) -> Result<T>) -> Result<T> {
        let guard = self
            .journal
            .lock()
            .map_err(|_| anyhow::anyhow!("managed journal lock failed"))?;
        f(guard
            .as_ref()
            .context("managed turn has not been initialized")?)
    }
}
impl PipeClient {
    pub(crate) fn validate_pending_workflow(&self, turn: &str, managed: bool) -> Result<()> {
        if managed {
            return Ok(());
        }
        if let Some(path) = &self.managed.path {
            match fs::read(path) {
                Ok(bytes) => {
                    let old: Journal = serde_json::from_slice(&bytes)
                        .context("reading managed workflow journal")?;
                    ensure!(
                        old.turn != turn,
                        "this pending turn uses managed coding; resume with pipe/auto and reconcile it before changing models"
                    );
                }
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                Err(e) => return Err(e.into()),
            }
        }
        Ok(())
    }
    pub(crate) fn begin_managed_auxiliary(&self) -> Result<()> {
        if self
            .managed
            .journal
            .lock()
            .map_err(|_| anyhow::anyhow!("managed journal lock"))?
            .is_some()
        {
            return Ok(());
        }
        let path = self
            .managed
            .path
            .as_ref()
            .context("managed journal path required")?;
        match fs::read(path) {
            Ok(bytes) => {
                let journal: Journal = serde_json::from_slice(&bytes)
                    .context("reading managed journal for compaction")?;
                self.begin_managed_turn(journal.turn, true)
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                self.begin_managed_turn(format!("compact:{}", uuid::Uuid::new_v4()), false)
            }
            Err(e) => Err(e.into()),
        }
    }
    pub(crate) fn with_managed(mut self, path: PathBuf, settings: ManagedSettings) -> Self {
        self.managed = ManagedState::configured(path, settings);
        self
    }
    pub(crate) fn begin_managed_turn(&self, turn: String, resume: bool) -> Result<()> {
        self.managed.settings.validate()?;
        let path = self
            .managed
            .path
            .as_ref()
            .context("managed journal path required")?;
        let mut lease = self
            .managed
            .lease
            .lock()
            .map_err(|_| anyhow::anyhow!("managed journal lock failed"))?;
        if lease.is_none() {
            *lease = Some(crate::session_lease::SessionLease::acquire(path)?);
        }
        drop(lease);
        let old = match fs::read(path) {
            Ok(bytes) => Some(
                serde_json::from_slice::<Journal>(&bytes)
                    .context("reading managed recovery journal")?,
            ),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => None,
            Err(e) => return Err(e.into()),
        };
        let credential_hash = blake3::hash(self.api_key.as_bytes()).to_hex().to_string();
        let journal = match old {
            Some(old) if old.turn == turn => {
                ensure!(
                    old.version == 1
                        && old.endpoint == self.base_url
                        && old.credential_hash == credential_hash,
                    "managed resume requires the original endpoint and project credential"
                );
                old
            }
            Some(old)
                if old.operations.iter().any(|o| {
                    o.unresolved
                        || o.tools
                            .values()
                            .any(|s| matches!(s, ToolState::Started | ToolState::Pending))
                }) =>
            {
                bail!(
                    "previous managed turn requires reconciliation: inspect {} before starting another turn",
                    path.display()
                );
            }
            _ => {
                ensure!(
                    !resume,
                    "managed turn journal is missing; original spending budget cannot be recovered"
                );
                Journal {
                    version: 1,
                    turn,
                    endpoint: self.base_url.clone(),
                    credential_hash,
                    budget: micros(&self.managed.settings.turn_budget_usd)?,
                    settings: self.managed.settings.clone(),
                    operations: Vec::new(),
                }
            }
        };
        self.managed.save(&journal)?;
        *self
            .managed
            .journal
            .lock()
            .map_err(|_| anyhow::anyhow!("managed journal lock"))? = Some(journal);
        Ok(())
    }
    pub(crate) fn recover_managed_tools(&self, messages: &mut Vec<Message>) -> Result<()> {
        self.managed.read(|j| {
            let compacted_through = j.operations.iter().rposition(|o| o.auxiliary && o.completion.is_some());
            for (index, operation) in j.operations.iter().enumerate() {
                for (id, state) in &operation.tools {
                    ensure!(!matches!(state,ToolState::Started), "tool {id} may have executed before the crash. Reconcile its effects using journal {} before resuming; it will not run again automatically",self.managed.path.as_ref().unwrap().display());
                }
                if compacted_through.is_some_and(|last| index <= last) { continue; }
                if operation.tools.is_empty() || operation.tools.values().any(|s| !matches!(s, ToolState::Completed {..})) { continue; }
                let completion = operation.completion.as_ref().context("accepted tool batch missing")?;
                let present = messages.iter().flat_map(|m| &m.content).any(|c| matches!(c,Content::ToolCall{id,..} if completion.tool_calls.iter().any(|t| &t.id==id)));
                if !present {
                    messages.push(crate::turn::assistant_message(completion));
                }
                for call in &completion.tool_calls {
                    let saved = messages.iter().flat_map(|m| &m.content).any(|c| matches!(c,Content::ToolResult{call_id,..} if call_id==&call.id));
                    if !saved && let ToolState::Completed { outcome } = &operation.tools[&call.id] {
                        messages.push(Message::tool_result(&call.id, reported_outcome(outcome)));
                    }
                }
            }
            if let Some(op) = j.operations.last().filter(|o| !o.auxiliary)
                && let Some(completion) = op.completion.as_ref().filter(|c| c.tool_calls.is_empty()) {
                    let already_saved = messages.iter().rev().take(1).any(|m| m.role == hi_ai::Role::Assistant && m.text() == completion.text);
                    if !already_saved { *self.managed.replay_final.lock().map_err(|_|anyhow::anyhow!("managed replay lock"))? = Some(completion.clone()); }
            }
            Ok(())
        })
    }
    pub(crate) fn managed_tool_start(&self, id: &str) -> Result<Option<hi_tools::ToolOutcome>> {
        self.managed.update(|j| {
            let state = j
                .operations
                .iter_mut()
                .find_map(|o| o.tools.get_mut(id))
                .context("tool is absent from accepted managed batch")?;
            match state {
                ToolState::Completed { outcome } => Ok(Some(outcome.clone())),
                ToolState::Started => {
                    bail!("tool {id} has an ambiguous prior execution; reconcile before retrying")
                }
                ToolState::Pending => {
                    *state = ToolState::Started;
                    Ok(None)
                }
            }
        })
    }
    pub(crate) fn recovered_managed_progress(&self) -> Result<(bool, bool, Vec<String>)> {
        self.managed.read(|j| {
            let (mut mutated, mut verified, mut changed) = (false, false, Vec::new());
            for operation in &j.operations {
                if let Some(completion) = &operation.completion {
                    for call in &completion.tool_calls {
                        if let Some(ToolState::Completed { outcome }) =
                            operation.tools.get(&call.id)
                        {
                            if outcome.effects.mutation_applied {
                                mutated = true;
                                verified = false;
                            }
                            if crate::completion::looks_like_verify(&call.name, &call.arguments) {
                                verified = true;
                            }
                            for change in &outcome.effects.file_changes {
                                if !changed.contains(&change.path) {
                                    changed.push(change.path.clone());
                                }
                            }
                        }
                    }
                }
            }
            Ok((mutated, verified, changed))
        })
    }
    pub(crate) fn managed_tool_complete(
        &self,
        id: &str,
        outcome: &hi_tools::ToolOutcome,
    ) -> Result<()> {
        self.managed.update(|j| {
            let state = j
                .operations
                .iter_mut()
                .find_map(|o| o.tools.get_mut(id))
                .context("accepted managed tool missing")?;
            *state = ToolState::Completed {
                outcome: outcome.clone(),
            };
            Ok(())
        })
    }
    async fn managed_status(&self, key: &str, cancel: bool) -> Result<Value> {
        let url = format!(
            "{}/pipe/requests/by-key{}",
            self.base_url,
            if cancel { "/cancel" } else { "" }
        );
        let request = if cancel {
            self.http.post(url)
        } else {
            self.http.get(url)
        };
        let response = request
            .bearer_auth(&self.api_key)
            .header("Idempotency-Key", key)
            .timeout(Duration::from_secs(8))
            .send()
            .await?;
        ensure!(
            response.status().is_success(),
            "managed status lookup failed (HTTP {})",
            response.status()
        );
        response
            .json()
            .await
            .context("invalid managed recovery status")
    }
    fn settle_managed(&self, key: &str, value: &Value) -> Result<()> {
        let meta = value
            .get("pipe")
            .or_else(|| value.get("metadata"))
            .context("missing managed settlement")?;
        let status = value
            .get("status")
            .or_else(|| meta.get("status"))
            .and_then(Value::as_str)
            .context("missing managed status")?;
        if !matches!(
            status,
            "completed"
                | "cancelled"
                | "insufficient_evidence"
                | "needs_clarification"
                | "budget_exhausted"
                | "deadline_exceeded"
                | "provider_unavailable"
        ) {
            return Ok(());
        }
        let charge = micros(
            meta["total_charge_usd"]
                .as_str()
                .context("missing actual managed charge")?,
        )?;
        let unresolved = meta["unresolved_micros"]
            .as_i64()
            .context("missing unresolved managed cost")?;
        ensure!(unresolved >= 0, "invalid unresolved cost");
        self.managed.update(|j| {
            let op = j
                .operations
                .iter_mut()
                .find(|o| o.key == key)
                .context("managed operation missing")?;
            ensure!(
                charge >= 0 && charge <= op.reserved,
                "managed settlement exceeds reserved call budget"
            );
            op.charge = Some(charge);
            op.unresolved = unresolved > 0;
            op.status = status.into();
            Ok(())
        })
    }
    pub(crate) async fn stream_managed(
        &self,
        mut body: Value,
        tools: &[ToolSpec],
        on_event: &mut dyn FnMut(StreamDelta),
        cancel: &TurnCancellation,
    ) -> Result<PipeCompletion> {
        if let Some(completion) = self
            .managed
            .replay_final
            .lock()
            .map_err(|_| anyhow::anyhow!("managed replay lock"))?
            .take()
        {
            if !completion.text.is_empty() {
                on_event(StreamDelta::Text(completion.text.clone()));
            }
            return Ok(completion);
        }
        if let Some(op) = self.managed.read(|j| {
            Ok(j.operations
                .iter()
                .find(|o| o.completion.is_none() && o.status != "prepared")
                .cloned())
        })? {
            let status = self.managed_status(&op.key, false).await;
            if let Ok(value) = &status {
                let _ = self.settle_managed(&op.key, value);
            }
            bail!(
                "managed request {} has no locally retained accepted response (status {}). Inspect {} and server status before starting a separately budgeted turn; it will not regenerate automatically",
                op.key,
                status
                    .as_ref()
                    .ok()
                    .and_then(|v| v["status"].as_str())
                    .unwrap_or("unknown"),
                self.managed.path.as_ref().unwrap().display()
            );
        }
        // Resume an accepted partial batch before requesting any more inference.
        if let Some(completion) = self.managed.read(|j| {
            Ok(j.operations
                .iter()
                .find(|o| {
                    o.tools
                        .values()
                        .any(|s| matches!(s, ToolState::Pending | ToolState::Started))
                })
                .and_then(|o| o.completion.clone()))
        })? {
            return Ok(completion);
        }
        let discovery = self
            .http
            .get(format!("{}/models", self.base_url))
            .bearer_auth(&self.api_key)
            .timeout(Duration::from_secs(10))
            .send()
            .await?
            .error_for_status()?
            .json::<Value>()
            .await?;
        let cap = discovery["data"]
            .as_array()
            .and_then(|a| a.iter().find(|m| m["id"] == "pipe/auto"))
            .and_then(|m| m.get("managed_coding"))
            .context("managed coding is unavailable: /models did not advertise coding-v1")?;
        ensure!(
            cap["available"] == true && cap["version"] == "coding-v1",
            "managed coding unavailable: {}",
            cap["reason"]
                .as_str()
                .unwrap_or("unsupported coding version")
        );
        let settings = self.managed.read(|j| Ok(j.settings.clone()))?;
        body.as_object_mut()
            .context("request body")?
            .retain(|k, _| {
                [
                    "model",
                    "messages",
                    "tools",
                    "tool_choice",
                    "max_tokens",
                    "stream",
                ]
                .contains(&k.as_str())
            });
        for message in body["messages"].as_array_mut().context("messages")? {
            message
                .as_object_mut()
                .context("message")?
                .remove("reasoning_content");
        }
        let output = cap
            .pointer("/limits/output_tokens")
            .and_then(Value::as_u64)
            .context("coding output limit missing")?
            .min(8192);
        body["max_tokens"] = json!(body["max_tokens"].as_u64().unwrap_or(output).min(output));
        body["pipe"] = json!({"workflow":"coding","profile":settings.profile,"retrieval":{"mode":"off"},"verification":"required","delivery":"buffered_verified","include_metadata":true,"deadline_ms":settings.deadline_ms.min(cap.pointer("/limits/deadline_ms").and_then(Value::as_u64).context("coding deadline missing")?)});
        let input_hash = blake3::hash(&serde_json::to_vec(&body)?)
            .to_hex()
            .to_string();
        let op = self.managed.update(|j| {
            if let Some(old) = j.operations.iter().find(|o| o.input_hash == input_hash) { return Ok(old.clone()); }
            ensure!(!j.operations.iter().any(|o| o.unresolved), "an earlier managed request needs status reconciliation; no new inference will start");
            let spent = j.operations.iter().map(|o| o.charge.unwrap_or(o.reserved)).try_fold(0i64,|a,b| a.checked_add(b)).context("managed budget overflow")?;
            let available = j.budget.checked_sub(spent).context("managed turn budget overflow")?;
            let reserve = micros(&settings.call_budget_usd)?.min(available).min(1_000_000);
            ensure!(reserve > 0,"managed turn budget exhausted");
            body["pipe"]["max_cost_usd"] = json!(usd(reserve));
            let payload = serde_json::to_string(&body)?;
            ensure!(payload.len() <= 1_048_576,"managed coding request exceeds 1 MiB");
            let op = Operation { auxiliary:tools.is_empty(), key:uuid::Uuid::new_v4().to_string(),input_hash,payload_hash:blake3::hash(payload.as_bytes()).to_hex().to_string(),payload,reserved:reserve,charge:None,unresolved:true,status:"prepared".into(),completion:None,tools:BTreeMap::new() };
            j.operations.push(op.clone());Ok(op)
        })?;
        if let Some(completion) = op.completion {
            return Ok(completion);
        }
        if op.status != "prepared" {
            let status = self.managed_status(&op.key, false).await?;
            self.settle_managed(&op.key, &status)?;
            bail!(
                "managed request {} has status {}; its response is not retained by the server. Inspect local journal {} and reconcile before starting a separately budgeted request",
                op.key,
                status["status"],
                self.managed.path.as_ref().unwrap().display()
            );
        }
        self.managed.update(|j| {
            j.operations
                .iter_mut()
                .find(|o| o.key == op.key)
                .unwrap()
                .status = "submitted".into();
            Ok(())
        })?;
        on_event(StreamDelta::Status(format!(
            "managed routing / verifying · {}",
            op.key
        )));
        ensure!(
            blake3::hash(op.payload.as_bytes()).to_hex().as_str() == op.payload_hash,
            "managed payload hash mismatch"
        );
        let execute = async {
            let response = self
                .http
                .post(format!("{}/chat/completions", self.base_url))
                .bearer_auth(&self.api_key)
                .header("Content-Type", "application/json")
                .header("Accept", "text/event-stream")
                .header("Idempotency-Key", &op.key)
                .timeout(Duration::from_millis(settings.deadline_ms + 5000))
                .body(op.payload.clone())
                .send()
                .await?;
            let status = response.status();
            let mut response = response;
            let mut bytes = Vec::new();
            while let Some(chunk) = response.chunk().await? {
                ensure!(
                    bytes.len() + chunk.len() <= 1_048_576,
                    "managed response too large"
                );
                bytes.extend_from_slice(&chunk);
            }
            let value = parse_response(&bytes)?;
            if value.get("pipe").is_some() {
                self.settle_managed(&op.key, &value)?;
            }
            ensure!(
                status.is_success(),
                "managed request {} failed: {}",
                op.key,
                value.get("error").unwrap_or(&value)
            );
            let completion = accepted(&value, tools)?;
            self.managed.update(|j| {
                let op = j.operations.iter_mut().find(|o| o.key == op.key).unwrap();
                ensure!(
                    !op.unresolved && op.charge.is_some(),
                    "managed settlement is incomplete"
                );
                op.tools = completion
                    .tool_calls
                    .iter()
                    .map(|c| (c.id.clone(), ToolState::Pending))
                    .collect();
                op.completion = Some(completion.clone());
                Ok(())
            })?;
            on_event(StreamDelta::Status(format!(
                "managed verified · {} · ${}",
                value["id"].as_str().unwrap_or(&op.key),
                value["pipe"]["total_charge_usd"]
                    .as_str()
                    .unwrap_or("unknown")
            )));
            if !completion.text.is_empty() {
                on_event(StreamDelta::Text(completion.text.clone()));
            }
            Ok::<_, anyhow::Error>(completion)
        };
        let result = tokio::select! {
            result = execute => result,
            _ = cancel.cancelled() => {
                let status = self.managed_status(&op.key,true).await;
                if let Ok(status) = status { let _ = self.settle_managed(&op.key,&status); }
                return Err(PipeError::cancelled().into());
            }
        };
        match result {
            Ok(c) => Ok(c),
            Err(error) => {
                let status = self.managed_status(&op.key, false).await;
                let state = if let Ok(status) = status {
                    let _ = self.settle_managed(&op.key, &status);
                    status["status"].to_string()
                } else {
                    "unknown (budget remains reserved)".into()
                };
                bail!(
                    "managed request {} stopped: {error:#}; server status {state}. Response recovery requires inspecting {}. No automatic regeneration or tool execution occurred",
                    op.key,
                    self.managed.path.as_ref().unwrap().display()
                )
            }
        }
    }
}

mod evidence;
mod inspection;
pub use inspection::{ManagedInspection, inspect_managed_journal};
mod response;
#[cfg(test)]
mod tests;

pub(crate) use evidence::{preserve_result_receipts, retain_execution_evidence};
use response::accepted;
pub(crate) use response::{parse_response, reported_outcome};
