//! Request-time packing of large tool results with exact paged recall.
//!
//! Stored transcripts are not rewritten. Projection replaces a large tool
//! result with a stable handle only in the copy sent to the provider, after a
//! bounded number of full sends. Archive or pack failures leave the original
//! result in that request (fail-open).

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use hi_ai::{Content, Message, Role};
use sha2::{Digest, Sha256};

/// Conservative default: pack payloads larger than this. Not a copy of any
/// external harness cutoff.
pub const DEFAULT_PACK_THRESHOLD_BYTES: usize = 4_096;
/// Full provider sends that still carry the original body.
pub const DEFAULT_FULL_SENDS: u32 = 2;
/// Placeholder excerpt budget, split between head and tail.
pub const DEFAULT_EXCERPT_BYTES: usize = 1_024;
/// Hard recall page size.
pub const RECALL_MAX_BYTES: usize = 16 * 1024;
/// Hard recall line cap.
pub const RECALL_MAX_LINES: usize = 400;

const ID_PREFIX: &str = "obs_";
const VERIFY_DIGEST_MARK: &str = "── failure digest ──";
const CONDENSE_OMISSION_MARK: &str = " lines omitted ";
pub const EVIDENCE_RECEIPT_PREFIX: &str = "hi_evidence_receipt_v1";

#[derive(Clone, Debug)]
pub struct ObservationPackConfig {
    pub threshold_bytes: usize,
    pub full_sends: u32,
    pub excerpt_bytes: usize,
}

impl Default for ObservationPackConfig {
    fn default() -> Self {
        Self {
            threshold_bytes: DEFAULT_PACK_THRESHOLD_BYTES,
            full_sends: DEFAULT_FULL_SENDS,
            excerpt_bytes: DEFAULT_EXCERPT_BYTES,
        }
    }
}

#[derive(Clone, Debug)]
pub struct ObservationPack {
    root: PathBuf,
    config: ObservationPackConfig,
    sent_counts: HashMap<String, u32>,
    packed_ids: HashSet<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RecallChunk {
    pub text: String,
    pub bytes: usize,
    pub lines: usize,
    pub next_offset: usize,
    pub eof: bool,
}

#[derive(Clone, Debug)]
struct PackedObservation {
    id: String,
    text: String,
    bytes: usize,
    lines: usize,
    path: PathBuf,
}

impl ObservationPack {
    pub fn new(state_root: impl Into<PathBuf>) -> Self {
        Self::with_config(state_root, ObservationPackConfig::default())
    }

    pub fn with_config(state_root: impl Into<PathBuf>, config: ObservationPackConfig) -> Self {
        Self {
            root: state_root.into().join("observation-pack"),
            config,
            sent_counts: HashMap::new(),
            packed_ids: HashSet::new(),
        }
    }

    pub fn has_packed_handles(&self) -> bool {
        !self.packed_ids.is_empty()
    }

    /// Project a request copy. The input slice is never mutated.
    pub fn project(&mut self, messages: &[Message]) -> Vec<Message> {
        let mut projected = messages.to_vec();
        let prior_assistant = prior_assistant_counts(messages);
        for (index, message) in projected.iter_mut().enumerate() {
            let Some(output) = tool_result_text(message) else {
                continue;
            };
            let call_id = tool_result_call_id(message).unwrap_or("");
            let tool_name = tool_name_for_result(messages, call_id).unwrap_or("tool");
            if !is_packable(tool_name, &output, self.config.threshold_bytes) {
                continue;
            }
            let observation = match create_observation(&self.root, tool_name, call_id, &output) {
                Some(observation) => observation,
                None => continue,
            };
            if let Err(error) = ensure_stored(&observation) {
                tracing::debug!(
                    error = %error,
                    id = %observation.id,
                    "observation pack failed open; keeping original tool result"
                );
                continue;
            }
            let previous = self
                .sent_counts
                .get(&observation.id)
                .copied()
                .unwrap_or(prior_assistant.get(index).copied().unwrap_or(0));
            if previous < self.config.full_sends {
                self.sent_counts
                    .insert(observation.id.clone(), previous.saturating_add(1));
                continue;
            }
            let placeholder = placeholder_for(&observation, self.config.excerpt_bytes);
            replace_tool_result_text(message, placeholder);
            self.sent_counts
                .insert(observation.id.clone(), previous.saturating_add(1));
            self.packed_ids.insert(observation.id);
        }
        projected
    }

    pub fn recall(&self, id: &str, offset: usize) -> Result<RecallChunk, String> {
        if !is_observation_id(id) {
            return Err(format!("Unknown observation id: {id}"));
        }
        let path = object_path(&self.root, id);
        read_recall_chunk(&path, offset, RECALL_MAX_BYTES, RECALL_MAX_LINES)
    }
}

pub fn is_observation_id(id: &str) -> bool {
    let Some(rest) = id.strip_prefix(ID_PREFIX) else {
        return false;
    };
    rest.len() == 24 && rest.bytes().all(|byte| byte.is_ascii_hexdigit())
}

/// Bash/process logs may pack after full sends. Discovery results (read/grep/
/// list/…) are the working set for the current task — replacing them with an
/// `obs_recall` handle spends model rounds paging archives until the request
/// budget dies, which is the live stall this policy exists to prevent.
pub const OBS_RECALL_TURN_BUDGET: u32 = 1;

pub fn is_process_observation(tool_name: &str) -> bool {
    matches!(tool_name, "bash" | "run_program")
}

pub fn is_packable(tool_name: &str, text: &str, threshold_bytes: usize) -> bool {
    if text.len() <= threshold_bytes {
        return false;
    }
    if !is_process_observation(tool_name) {
        return false;
    }
    !is_protected_evidence(text)
}

pub fn is_protected_evidence(text: &str) -> bool {
    text.contains(VERIFY_DIGEST_MARK)
        || text.contains(CONDENSE_OMISSION_MARK)
        || text.contains(hi_tools::FUSED_COMMAND_FAILED)
        || text.contains(hi_tools::FUSED_COMMAND_SKIPPED)
        || text.lines().any(|line| line == EVIDENCE_RECEIPT_PREFIX)
        || looks_like_unresolved_failure(text)
}

pub(crate) fn looks_like_unresolved_failure(text: &str) -> bool {
    let lower = text.to_ascii_lowercase();
    if lower.contains("error[e")
        || lower.contains(": error:")
        || lower.contains("test result: failed")
        || lower.contains("error: cannot find")
        || lower.contains("panicked at")
        || lower.contains("error ts")
        || lower.contains("assertionerror")
        || lower.contains("===== failures =====")
        || lower.contains("npm err!")
        || lower.contains("\nfail:")
        || lower.starts_with("fail:")
    {
        return true;
    }
    lower.lines().any(failure_evidence_line)
}

fn failure_evidence_line(line: &str) -> bool {
    let line = line.trim_start();
    line.starts_with("error: ")
        || line.starts_with("fail:")
        || line.starts_with("fail\t")
        || line.starts_with("fail ")
        || line == "fail"
        || line.starts_with("failed ")
        || line.starts_with("failed\t")
        || line.starts_with("--- fail")
        || line.ends_with(" ... failed")
}

fn create_observation(
    root: &Path,
    tool_name: &str,
    call_id: &str,
    text: &str,
) -> Option<PackedObservation> {
    if is_protected_evidence(text) {
        return None;
    }
    let bytes = text.len();
    let content_hash = hex_sha256(text.as_bytes());
    let id_src = format!("{tool_name}\0{call_id}\0{content_hash}");
    let id = format!("{ID_PREFIX}{}", &hex_sha256(id_src.as_bytes())[..24]);
    Some(PackedObservation {
        path: object_path(root, &id),
        id,
        text: text.to_string(),
        bytes,
        lines: count_lines(text),
    })
}

fn object_path(root: &Path, id: &str) -> PathBuf {
    root.join("objects").join(format!("{id}.txt"))
}

fn ensure_stored(observation: &PackedObservation) -> std::io::Result<()> {
    let directory = observation
        .path
        .parent()
        .ok_or_else(|| std::io::Error::other("observation path has no parent"))?;
    std::fs::create_dir_all(directory)?;
    match std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&observation.path)
    {
        Ok(_) => std::fs::write(&observation.path, observation.text.as_bytes()),
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
            let existing = std::fs::read(&observation.path)?;
            if existing != observation.text.as_bytes() {
                return Err(std::io::Error::other(
                    "content-addressed observation hash mismatch",
                ));
            }
            Ok(())
        }
        Err(error) => Err(error),
    }
}

fn placeholder_for(observation: &PackedObservation, excerpt_bytes: usize) -> String {
    let head_budget = excerpt_bytes / 2;
    let tail_budget = excerpt_bytes - head_budget;
    let head = complete_line_excerpt(&observation.text, head_budget, false);
    let tail = complete_line_excerpt(&observation.text, tail_budget, true);
    format!(
        "[large tool result replaced after its first full provider sends]\n\
         id: {}\n\
         original_bytes: {}\n\
         original_lines: {}\n\
         retrieve: call obs_recall with {{\"id\":\"{}\",\"offset\":0}}; continue with returned next_offset\n\
         [first complete lines]\n\
         {head}\n\
         [middle omitted; last complete lines]\n\
         {tail}\n\
         [{} original bytes omitted]",
        observation.id, observation.bytes, observation.lines, observation.id, observation.bytes
    )
}

fn complete_line_excerpt(text: &str, budget_bytes: usize, from_end: bool) -> String {
    let lines: Vec<&str> = text.split_inclusive('\n').collect();
    let mut selected = Vec::new();
    let mut used = 0usize;
    if from_end {
        for line in lines.iter().rev() {
            let line_bytes = line.len();
            if used.saturating_add(line_bytes) > budget_bytes {
                break;
            }
            selected.push(*line);
            used += line_bytes;
        }
        selected.reverse();
    } else {
        for line in &lines {
            let line_bytes = line.len();
            if used.saturating_add(line_bytes) > budget_bytes {
                break;
            }
            selected.push(*line);
            used += line_bytes;
        }
    }
    selected.concat()
}

fn read_recall_chunk(
    path: &Path,
    offset: usize,
    max_bytes: usize,
    max_lines: usize,
) -> Result<RecallChunk, String> {
    let bytes = std::fs::read(path).map_err(|_| {
        path.file_stem()
            .and_then(|stem| stem.to_str())
            .map(|id| format!("Unknown observation id: {id}"))
            .unwrap_or_else(|| "Unknown observation id".to_string())
    })?;
    if offset > bytes.len() {
        return Err(format!(
            "Offset {offset} exceeds observation size {}",
            bytes.len()
        ));
    }
    let available = &bytes[offset..];
    let mut end = available.len().min(max_bytes);
    let mut newline_count = 0usize;
    for (index, byte) in available.iter().take(end).enumerate() {
        if *byte == b'\n' {
            newline_count += 1;
            if newline_count == max_lines {
                end = index + 1;
                break;
            }
        }
    }
    end = trim_utf8_end(available, end);
    let chunk = &available[..end];
    let text = std::str::from_utf8(chunk)
        .map_err(|_| "Stored observation is not valid UTF-8 at this offset".to_string())?
        .to_owned();
    let next_offset = offset + chunk.len();
    Ok(RecallChunk {
        bytes: chunk.len(),
        lines: count_lines(&text),
        text,
        next_offset,
        eof: next_offset >= bytes.len(),
    })
}

/// Exclusive end index that does not split a UTF-8 character. If `limit` lands
/// inside a multi-byte sequence, the incomplete character is excluded so
/// `next_offset` stays on a character boundary.
fn trim_utf8_end(buffer: &[u8], mut end: usize) -> usize {
    end = end.min(buffer.len());
    while end > 0 && end < buffer.len() && (buffer[end] & 0xc0) == 0x80 {
        end -= 1;
    }
    end
}

fn prior_assistant_counts(messages: &[Message]) -> Vec<u32> {
    let mut counts = vec![0; messages.len()];
    let mut assistants = 0u32;
    for index in (0..messages.len()).rev() {
        counts[index] = assistants;
        if messages[index].role == Role::Assistant {
            assistants = assistants.saturating_add(1);
        }
    }
    counts
}

fn tool_result_text(message: &Message) -> Option<String> {
    if message.role != Role::Tool {
        return None;
    }
    let mut parts = Vec::new();
    for block in &message.content {
        match block {
            Content::ToolResult { output, .. } => parts.push(output.as_str()),
            Content::Text(text) => parts.push(text.as_str()),
            _ => return None,
        }
    }
    if parts.is_empty() {
        None
    } else {
        Some(parts.join("\n"))
    }
}

fn tool_result_call_id(message: &Message) -> Option<&str> {
    message.content.iter().find_map(|block| match block {
        Content::ToolResult { call_id, .. } => Some(call_id.as_str()),
        _ => None,
    })
}

fn tool_name_for_result<'a>(messages: &'a [Message], call_id: &str) -> Option<&'a str> {
    if call_id.is_empty() {
        return None;
    }
    for message in messages {
        if message.role != Role::Assistant {
            continue;
        }
        for block in &message.content {
            if let Content::ToolCall { id, name, .. } = block
                && id == call_id
            {
                return Some(name.as_str());
            }
        }
    }
    None
}

fn replace_tool_result_text(message: &mut Message, text: String) {
    for block in &mut message.content {
        match block {
            Content::ToolResult { output, .. } => {
                *output = text;
                return;
            }
            Content::Text(existing) => {
                *existing = text;
                return;
            }
            _ => {}
        }
    }
}

fn count_lines(text: &str) -> usize {
    if text.is_empty() {
        return 0;
    }
    let mut lines = if text.ends_with('\n') { 0 } else { 1 };
    lines += text.bytes().filter(|byte| *byte == b'\n').count();
    lines
}

fn hex_sha256(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

impl crate::Agent {
    pub(crate) fn project_messages_for_provider(&mut self) -> std::sync::Arc<Vec<Message>> {
        if !self.config.memory.observation_pack {
            return self.messages.arc();
        }
        std::sync::Arc::new(self.observation_pack.project(self.messages.as_slice()))
    }

    pub(crate) fn handle_obs_recall(&mut self, arguments: &str) -> hi_tools::ToolOutcome {
        self.obs_recall_calls = self.obs_recall_calls.saturating_add(1);
        if self.obs_recall_calls > OBS_RECALL_TURN_BUDGET {
            return hi_tools::ToolOutcome {
                content: "Observation paging budget for this turn is spent. The excerpt already in context is the working set. Implement the change with write/edit/multi_edit/apply_patch; do not page archives.".into(),
                display: None,
                plan: None,
                status: hi_tools::ToolStatus::Failed,
                process: None,
                background: None,
                effects: hi_tools::ToolEffects::default(),
                truncation: hi_tools::TruncationState::Complete,
                images: Vec::new(),
            };
        }
        #[derive(serde::Deserialize)]
        struct Args {
            id: String,
            offset: Option<usize>,
        }
        let parsed = serde_json::from_str::<Args>(arguments);
        let outcome = match parsed {
            Err(error) => (format!("Error: {error}"), hi_tools::ToolStatus::Failed),
            Ok(args) => match self
                .observation_pack
                .recall(&args.id, args.offset.unwrap_or(0))
            {
                Ok(chunk) => {
                    let header = format!(
                        "[obs_recall id={} offset={} next_offset={} eof={}]\n[chunk_bytes={} chunk_lines={}; use next_offset to continue]",
                        args.id,
                        args.offset.unwrap_or(0),
                        chunk.next_offset,
                        chunk.eof,
                        chunk.bytes,
                        chunk.lines
                    );
                    (
                        format!("{header}\n{}", chunk.text),
                        hi_tools::ToolStatus::Succeeded,
                    )
                }
                Err(error) => (format!("Error: {error}"), hi_tools::ToolStatus::Failed),
            },
        };
        // Do not run recall pages through the shared ~5k tool-result cap.
        // next_offset is an archive byte index; clipping the page would drop
        // the middle while still advancing past it. Redact secrets only.
        let content = hi_secrets::redact_secrets(&outcome.0).into_owned();
        hi_tools::ToolOutcome {
            content,
            display: None,
            plan: None,
            status: outcome.1,
            process: None,
            background: None,
            effects: hi_tools::ToolEffects::default(),
            truncation: hi_tools::TruncationState::Complete,
            images: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests;
