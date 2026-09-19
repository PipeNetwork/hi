//! Jev-scored tool prune: drop or truncate stale tool pairs, keep the rest verbatim.
//!
//! User and assistant text is never rewritten. Fail-open: a timeout, HTTP miss,
//! or history that cannot be fitted leaves the transcript unchanged.

use std::collections::{HashMap, HashSet};
use std::time::Duration;

use hi_ai::{Content, Message, Role};
use serde::Serialize;
use serde_json::{Value, json};

use crate::TurnCancellation;
use crate::compact::KEEP_LAST_TOOL_RESULTS;
use crate::typesafe::{TypesafeClient, noul_answer};

const KEEP_THRESHOLD: f64 = 0.5;
const MAX_STATE_TOKENS: u64 = 25_000;
const MAX_REQUEST_TOKENS: u64 = 30_000;
const REQUEST_OVERHEAD_TOKENS: u64 = 20;
const TRUNCATE_HEAD_CHARS: usize = 300;
const PRESERVE_RECENT_MESSAGES: usize = KEEP_LAST_TOOL_RESULTS;
const INPUT_CHARS: [usize; 3] = [1000, 200, 60];
const TEXT_HEAD: usize = 400;
const TEXT_TAIL: usize = 150;
const MAX_CONCURRENT_BATCHES: usize = 4;
const PER_REQUEST_TIMEOUT: Duration = Duration::from_secs(8);
const WALL_TIMEOUT: Duration = Duration::from_secs(12);
const OMITTED_SUFFIX: &str = " · omitted";
const STATE_CONTEXT: &str = "A coding assistant conversation is being compacted to free context. \
`history` is the whole conversation so far, oldest first; tool outputs are replaced by a short \
`result` note and long texts may be abridged. Each question asks whether one tool call, or the \
full output of that call, still needs to stay in the history verbatim. Whatever is not kept is \
deleted permanently, but the assistant can always re-run a tool or re-read a file.";

#[derive(Clone, Debug)]
pub(crate) struct ToolPair {
    pub id: String,
    pub call_id: String,
    pub tool: String,
    pub input: String,
    pub call_index: usize,
    #[allow(dead_code)]
    pub result_index: usize,
    pub result_chars: usize,
    pub is_error: bool,
    pub pinned: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum CallAction {
    Keep,
    DropResult,
    DropCall,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct CallDecision {
    pub id: String,
    pub call_id: String,
    pub tool: String,
    pub action: CallAction,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct PruneStats {
    pub calls: usize,
    pub kept: usize,
    pub results_dropped: usize,
    pub calls_dropped: usize,
    pub pinned: usize,
}

impl PruneStats {
    pub(crate) fn status_line(&self) -> String {
        format!(
            "jev-compact: kept {}/{} tools (dropped {} calls, truncated {} results)",
            self.kept + self.pinned,
            self.calls,
            self.calls_dropped,
            self.results_dropped
        )
    }
}

#[derive(Clone, Debug)]
pub(crate) struct PruneOutcome {
    pub messages: Vec<Message>,
    pub stats: PruneStats,
    pub changed: bool,
}

#[derive(Clone, Debug, Serialize)]
struct HistoryToolCall {
    id: String,
    tool: String,
    input: String,
    result: String,
}

#[derive(Clone, Debug, Serialize)]
#[serde(untagged)]
enum ToolCallsRepr {
    Structured(Vec<HistoryToolCall>),
    Compact(Vec<String>),
}

#[derive(Clone, Debug, Serialize)]
struct HistoryEntry {
    i: usize,
    role: String,
    text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<ToolCallsRepr>,
}

#[derive(Clone, Debug, Serialize)]
struct CompactionState {
    context: String,
    goal: String,
    history: Vec<HistoryEntry>,
}

pub(crate) fn is_pinned(index: usize, total: usize, preserve_recent: usize) -> bool {
    index == 0 || index >= total.saturating_sub(preserve_recent)
}

pub(crate) fn collect_tool_pairs(messages: &[Message], preserve_recent: usize) -> Vec<ToolPair> {
    let mut results = HashMap::new();
    for (index, message) in messages.iter().enumerate() {
        for block in &message.content {
            if let Content::ToolResult { call_id, output } = block {
                results.insert(
                    call_id.clone(),
                    (index, output.len(), looks_like_error(output)),
                );
            }
        }
    }
    let mut pairs = Vec::new();
    for (call_index, message) in messages.iter().enumerate() {
        for block in &message.content {
            let Content::ToolCall {
                id,
                name,
                arguments,
            } = block
            else {
                continue;
            };
            let Some(&(result_index, result_chars, is_error)) = results.get(id) else {
                continue;
            };
            let pinned = is_pinned(call_index, messages.len(), preserve_recent)
                || is_pinned(result_index, messages.len(), preserve_recent);
            pairs.push(ToolPair {
                id: format!("t{}", pairs.len() + 1),
                call_id: id.clone(),
                tool: name.clone(),
                input: arguments.clone(),
                call_index,
                result_index,
                result_chars,
                is_error,
                pinned,
            });
        }
    }
    pairs
}

pub(crate) fn decide_call(pair: &ToolPair, keep_call: f64, keep_result: f64) -> CallDecision {
    let action = if pair.pinned || keep_result >= KEEP_THRESHOLD {
        CallAction::Keep
    } else if keep_call >= KEEP_THRESHOLD {
        CallAction::DropResult
    } else {
        CallAction::DropCall
    };
    CallDecision {
        id: pair.id.clone(),
        call_id: pair.call_id.clone(),
        tool: pair.tool.clone(),
        action,
    }
}

pub(crate) fn apply_decisions(messages: &[Message], decisions: &[CallDecision]) -> Vec<Message> {
    let actions: HashMap<&str, CallAction> = decisions
        .iter()
        .filter(|decision| decision.action != CallAction::Keep)
        .map(|decision| (decision.call_id.as_str(), decision.action))
        .collect();
    if actions.is_empty() {
        return messages.to_vec();
    }
    let mut out = Vec::with_capacity(messages.len());
    for message in messages {
        let rebuilt = rebuild_message(message, &actions);
        if message.role == Role::User {
            out.push(rebuilt.unwrap_or_else(|| message.clone()));
            continue;
        }
        if let Some(rebuilt) = rebuilt {
            out.push(rebuilt);
        }
    }
    out
}

fn rebuild_message(message: &Message, actions: &HashMap<&str, CallAction>) -> Option<Message> {
    let mut content = Vec::with_capacity(message.content.len());
    for block in &message.content {
        match block {
            Content::ToolCall { id, .. } => match actions.get(id.as_str()) {
                Some(CallAction::DropCall) => {}
                _ => content.push(block.clone()),
            },
            Content::ToolResult { call_id, output } => match actions.get(call_id.as_str()) {
                Some(CallAction::DropCall) => {}
                Some(CallAction::DropResult) => {
                    content.push(Content::ToolResult {
                        call_id: call_id.clone(),
                        output: truncated_result(output),
                    });
                }
                _ => content.push(block.clone()),
            },
            _ => content.push(block.clone()),
        }
    }
    if content.is_empty() {
        return None;
    }
    Some(Message {
        role: message.role,
        content,
    })
}

fn truncated_result(output: &str) -> String {
    if output.ends_with(OMITTED_SUFFIX) {
        return output.to_string();
    }
    let head: String = output.chars().take(TRUNCATE_HEAD_CHARS).collect();
    format!("{head}\ntool · {} chars{OMITTED_SUFFIX}", output.len())
}

fn looks_like_error(output: &str) -> bool {
    let lower = output.to_ascii_lowercase();
    lower.contains("error") || lower.contains("failed") || output.starts_with("exit ")
}

fn estimate_tokens(text: &str) -> u64 {
    text.len() as u64 / 4
}

fn truncate(text: &str, limit: usize) -> String {
    if text.chars().count() <= limit {
        return text.to_string();
    }
    let mut out: String = text.chars().take(limit.saturating_sub(1)).collect();
    out.push('…');
    out
}

fn char_prefix(text: &str, n: usize) -> String {
    text.chars().take(n).collect()
}

fn char_suffix(text: &str, n: usize) -> String {
    let count = text.chars().count();
    text.chars().skip(count.saturating_sub(n)).collect()
}

fn abridge(text: &str, head: usize, tail: usize) -> String {
    let count = text.chars().count();
    if count <= head + tail + 40 {
        return text.to_string();
    }
    let omitted = count.saturating_sub(head + tail);
    format!(
        "{}\n[… {omitted} chars omitted …]\n{}",
        char_prefix(text, head),
        char_suffix(text, tail)
    )
}

fn role_name(role: Role) -> &'static str {
    match role {
        Role::System => "system",
        Role::User => "user",
        Role::Assistant => "assistant",
        Role::Tool => "tool",
    }
}

fn result_note(pair: &ToolPair) -> String {
    format!(
        "{}, {} chars (omitted)",
        if pair.is_error { "error" } else { "ok" },
        pair.result_chars
    )
}

fn compact_call(pair: &ToolPair) -> String {
    let input = truncate(&pair.input.replace(['\n', '\r'], " "), INPUT_CHARS[2]);
    format!(
        "{} {} {input} → {} {}ch",
        pair.id,
        pair.tool,
        if pair.is_error { "error" } else { "ok" },
        pair.result_chars
    )
}

fn calls_by_message(pairs: &[ToolPair]) -> HashMap<usize, Vec<&ToolPair>> {
    let mut by_message: HashMap<usize, Vec<&ToolPair>> = HashMap::new();
    for pair in pairs {
        by_message.entry(pair.call_index).or_default().push(pair);
    }
    by_message
}

fn history_entries(
    messages: &[Message],
    pairs: &[ToolPair],
    input_chars: usize,
) -> Vec<HistoryEntry> {
    let by_message = calls_by_message(pairs);
    let mut entries = Vec::new();
    for (i, message) in messages.iter().enumerate() {
        let tool_calls = by_message.get(&i).map(|list| {
            ToolCallsRepr::Structured(
                list.iter()
                    .map(|pair| HistoryToolCall {
                        id: pair.id.clone(),
                        tool: pair.tool.clone(),
                        input: truncate(&pair.input, input_chars),
                        result: result_note(pair),
                    })
                    .collect(),
            )
        });
        let text = message.text();
        if text.trim().is_empty() && tool_calls.is_none() {
            continue;
        }
        entries.push(HistoryEntry {
            i,
            role: role_name(message.role).to_string(),
            text,
            tool_calls,
        });
    }
    entries
}

fn entry_tokens(entry: &HistoryEntry) -> u64 {
    estimate_tokens(&serde_json::to_string(entry).unwrap_or_default()) + 1
}

fn state_of(history: Vec<HistoryEntry>, goal: &str) -> CompactionState {
    CompactionState {
        context: STATE_CONTEXT.to_string(),
        goal: goal.to_string(),
        history,
    }
}

fn base_tokens(goal: &str) -> u64 {
    estimate_tokens(&serde_json::to_string(&state_of(Vec::new(), goal)).unwrap_or_default())
}

fn merge_call_runs(
    history: Vec<HistoryEntry>,
    pinned: impl Fn(&HistoryEntry) -> bool,
) -> Vec<HistoryEntry> {
    let mut merged: Vec<HistoryEntry> = Vec::new();
    for entry in history {
        let foldable = |e: &HistoryEntry| {
            !pinned(e)
                && e.text.is_empty()
                && matches!(e.tool_calls, Some(ToolCallsRepr::Compact(_)))
        };
        if let Some(previous) = merged.last_mut()
            && foldable(previous)
            && foldable(&entry)
            && previous.role == entry.role
            && let (Some(ToolCallsRepr::Compact(left)), Some(ToolCallsRepr::Compact(right))) =
                (previous.tool_calls.as_mut(), entry.tool_calls.clone())
        {
            left.extend(right);
            continue;
        }
        merged.push(entry);
    }
    merged
}

fn fit_state(
    messages: &[Message],
    pairs: &[ToolPair],
    goal: &str,
) -> Result<(CompactionState, u64), String> {
    let mut history = history_entries(messages, pairs, INPUT_CHARS[0]);
    let mut tokens = base_tokens(goal) + history.iter().map(entry_tokens).sum::<u64>();
    if tokens <= MAX_STATE_TOKENS {
        return Ok((state_of(history, goal), tokens));
    }
    for limit in INPUT_CHARS.iter().skip(1) {
        history = history_entries(messages, pairs, *limit);
        tokens = base_tokens(goal) + history.iter().map(entry_tokens).sum::<u64>();
        if tokens <= MAX_STATE_TOKENS {
            return Ok((state_of(history, goal), tokens));
        }
    }

    let pinned =
        |entry: &HistoryEntry| is_pinned(entry.i, messages.len(), PRESERVE_RECENT_MESSAGES);
    let order: Vec<usize> = {
        let mut indices: Vec<usize> = (0..history.len()).collect();
        indices.sort_by_key(|index| pinned(&history[*index]));
        indices
    };

    for index in &order {
        if history[*index].text.len() <= TEXT_HEAD + TEXT_TAIL + 40 {
            continue;
        }
        let abridged = abridge(&history[*index].text, TEXT_HEAD, TEXT_TAIL);
        history[*index].text = abridged;
        tokens = base_tokens(goal) + history.iter().map(entry_tokens).sum::<u64>();
        if tokens <= MAX_STATE_TOKENS {
            return Ok((state_of(history, goal), tokens));
        }
    }

    for index in &order {
        if pinned(&history[*index]) || history[*index].text.is_empty() {
            continue;
        }
        let original = messages
            .get(history[*index].i)
            .map(|message| message.text().len())
            .unwrap_or(history[*index].text.len());
        let collapsed = format!("[… {original} chars omitted …]");
        history[*index].text = collapsed;
        tokens = base_tokens(goal) + history.iter().map(entry_tokens).sum::<u64>();
        if tokens <= MAX_STATE_TOKENS {
            return Ok((state_of(history, goal), tokens));
        }
    }

    let by_message = calls_by_message(pairs);
    for index in &order {
        if pinned(&history[*index]) {
            continue;
        }
        let Some(own) = by_message.get(&history[*index].i) else {
            continue;
        };
        history[*index].tool_calls = Some(ToolCallsRepr::Compact(
            own.iter().map(|pair| compact_call(pair)).collect(),
        ));
        tokens = base_tokens(goal) + history.iter().map(entry_tokens).sum::<u64>();
        if tokens <= MAX_STATE_TOKENS {
            return Ok((state_of(history, goal), tokens));
        }
    }

    let mut left = HashSet::new();
    for index in &order {
        if pinned(&history[*index]) || history[*index].tool_calls.is_some() {
            continue;
        }
        left.insert(*index);
        let remaining: Vec<HistoryEntry> = history
            .iter()
            .enumerate()
            .filter(|(i, _)| !left.contains(i))
            .map(|(_, entry)| entry.clone())
            .collect();
        tokens = base_tokens(goal) + remaining.iter().map(entry_tokens).sum::<u64>();
        if tokens <= MAX_STATE_TOKENS {
            return Ok((state_of(remaining, goal), tokens));
        }
    }

    let remaining: Vec<HistoryEntry> = history
        .into_iter()
        .enumerate()
        .filter(|(i, _)| !left.contains(i))
        .map(|(_, entry)| entry)
        .collect();
    history = merge_call_runs(remaining, pinned);
    tokens = base_tokens(goal) + history.iter().map(entry_tokens).sum::<u64>();
    if tokens <= MAX_STATE_TOKENS {
        return Ok((state_of(history, goal), tokens));
    }
    Err(format!(
        "history too large for Jev (~{tokens} tokens, limit {MAX_STATE_TOKENS})"
    ))
}

fn questions_for(pair: &ToolPair) -> Value {
    json!({
        format!("call_{}", pair.id): {
            "type": "noul",
            "instructions": format!(
                "Tool call {} ({}) should stay in the history: knowing this call was made, with its input, still matters for what the assistant does next",
                pair.id, pair.tool
            ),
        },
        format!("result_{}", pair.id): {
            "type": "noul",
            "instructions": format!(
                "The full output of tool call {} ({}, {} chars) should stay in the history verbatim: the assistant still needs its contents and re-running the tool would not do",
                pair.id, pair.tool, pair.result_chars
            ),
        }
    })
}

fn batch_pairs(pairs: &[ToolPair], state_tokens: u64) -> Result<Vec<Vec<ToolPair>>, String> {
    let budget =
        MAX_REQUEST_TOKENS.saturating_sub(state_tokens.saturating_add(REQUEST_OVERHEAD_TOKENS));
    let mut batches = Vec::new();
    let mut current = Vec::new();
    let mut current_tokens = 0u64;
    for pair in pairs {
        let tokens = estimate_tokens(&questions_for(pair).to_string());
        if !current.is_empty() && current_tokens + tokens > budget {
            batches.push(std::mem::take(&mut current));
            current_tokens = 0;
        }
        if current.is_empty() && tokens > budget {
            return Err(format!(
                "state leaves no room for questions (~{state_tokens} of {MAX_REQUEST_TOKENS} tokens)"
            ));
        }
        current.push(pair.clone());
        current_tokens = current_tokens.saturating_add(tokens);
    }
    if !current.is_empty() {
        batches.push(current);
    }
    Ok(batches)
}

fn goal_from_messages(messages: &[Message], override_goal: Option<&str>) -> String {
    if let Some(goal) = override_goal.map(str::trim).filter(|text| !text.is_empty()) {
        return truncate(goal, 500);
    }
    let prompts: Vec<String> = messages
        .iter()
        .filter(|message| message.role == Role::User && !message.text().trim().is_empty())
        .map(|message| truncate(&message.text(), 500))
        .collect();
    prompts
        .iter()
        .rev()
        .take(3)
        .cloned()
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect::<Vec<_>>()
        .join("\n")
}

fn stats_from(decisions: &[CallDecision], pairs: &[ToolPair]) -> PruneStats {
    let mut stats = PruneStats {
        calls: pairs.len(),
        ..PruneStats::default()
    };
    for (decision, pair) in decisions.iter().zip(pairs) {
        if pair.pinned {
            stats.pinned += 1;
            continue;
        }
        match decision.action {
            CallAction::Keep => stats.kept += 1,
            CallAction::DropResult => stats.results_dropped += 1,
            CallAction::DropCall => stats.calls_dropped += 1,
        }
    }
    stats
}

async fn ask_batch(
    client: &TypesafeClient,
    state: &Value,
    batch: &[ToolPair],
) -> Result<HashMap<String, (f64, f64)>, String> {
    let mut questions = serde_json::Map::new();
    for pair in batch {
        if let Value::Object(map) = questions_for(pair) {
            questions.extend(map);
        }
    }
    let asked = tokio::time::timeout(
        PER_REQUEST_TIMEOUT,
        client.ask(state.clone(), Value::Object(questions)),
    )
    .await
    .map_err(|_| "timed out".to_string())??;
    let answers = asked
        .get("answers")
        .cloned()
        .ok_or_else(|| "Jev response is missing answers".to_string())?;
    let mut out = HashMap::new();
    for pair in batch {
        let keep_call = noul_answer(&answers, &format!("call_{}", pair.id)).unwrap_or(1.0);
        let keep_result = noul_answer(&answers, &format!("result_{}", pair.id)).unwrap_or(1.0);
        out.insert(pair.id.clone(), (keep_call, keep_result));
    }
    Ok(out)
}

/// Score non-pinned tool pairs and apply keep / truncate / drop.
pub(crate) async fn prune_messages(
    messages: &[Message],
    client: &TypesafeClient,
    user_context: Option<&str>,
    cancel: &TurnCancellation,
) -> Result<PruneOutcome, String> {
    let work = prune_messages_inner(messages, client, user_context, cancel);
    tokio::select! {
        biased;
        _ = cancel.cancelled() => Err("cancelled".into()),
        _ = tokio::time::sleep(WALL_TIMEOUT) => Err("timed out".into()),
        result = work => result,
    }
}

async fn prune_messages_inner(
    messages: &[Message],
    client: &TypesafeClient,
    user_context: Option<&str>,
    cancel: &TurnCancellation,
) -> Result<PruneOutcome, String> {
    let pairs = collect_tool_pairs(messages, PRESERVE_RECENT_MESSAGES);
    let candidates: Vec<ToolPair> = pairs.iter().filter(|pair| !pair.pinned).cloned().collect();
    if candidates.is_empty() {
        return Ok(PruneOutcome {
            messages: messages.to_vec(),
            stats: stats_from(&[], &pairs),
            changed: false,
        });
    }
    let goal = goal_from_messages(messages, user_context);
    let (state, state_tokens) = fit_state(messages, &pairs, &goal)?;
    let state_value = serde_json::to_value(&state).map_err(|err| err.to_string())?;
    let batches = batch_pairs(&candidates, state_tokens)?;
    let mut answers = HashMap::new();
    let mut any_ok = false;
    let mut last_err = None;
    for chunk in batches.chunks(MAX_CONCURRENT_BATCHES) {
        if cancel.is_cancelled() {
            return Err("cancelled".into());
        }
        let futs = chunk.iter().map(|batch| {
            let client = client.clone();
            let state_value = state_value.clone();
            let batch = batch.clone();
            async move { ask_batch(&client, &state_value, &batch).await }
        });
        for result in futures_util::future::join_all(futs).await {
            match result {
                Ok(map) => {
                    any_ok = true;
                    answers.extend(map);
                }
                Err(err) => {
                    tracing::debug!("jev-compact batch failed: {err}");
                    last_err = Some(err);
                }
            }
        }
    }
    if !any_ok {
        return Err(last_err.unwrap_or_else(|| "Jev request failed".into()));
    }
    let decisions: Vec<CallDecision> = pairs
        .iter()
        .map(|pair| {
            let (keep_call, keep_result) = answers.get(&pair.id).copied().unwrap_or((1.0, 1.0));
            decide_call(pair, keep_call, keep_result)
        })
        .collect();
    let next = apply_decisions(messages, &decisions);
    let stats = stats_from(&decisions, &pairs);
    let changed = serde_json::to_string(&next).ok() != serde_json::to_string(messages).ok();
    Ok(PruneOutcome {
        messages: next,
        stats,
        changed,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    fn user(text: &str) -> Message {
        Message::user(text)
    }

    fn call(id: &str, name: &str, path: &str) -> Message {
        Message::assistant(vec![Content::ToolCall {
            id: id.into(),
            name: name.into(),
            arguments: format!(r#"{{"path":"{path}"}}"#),
        }])
    }

    fn result(id: &str, body: &str) -> Message {
        Message::tool_result(id, body)
    }

    fn long_history() -> Vec<Message> {
        let mut messages = vec![user("review this")];
        for i in 0..8 {
            let id = format!("c{i}");
            messages.push(call(&id, "read", &format!("src/f{i}.rs")));
            messages.push(result(&id, &format!("body-{i}-{}", "x".repeat(400))));
        }
        messages.push(user("keep going"));
        messages
    }

    #[test]
    fn pins_first_and_newest_six() {
        let messages = long_history();
        let pairs = collect_tool_pairs(&messages, 6);
        let unpinned = pairs.iter().filter(|pair| !pair.pinned).count();
        assert!(unpinned > 0, "middle tool pairs must be candidates");
        assert!(
            !pairs[0].pinned,
            "a tool on message 1 is not pinned by the first user line"
        );
        assert!(pairs.iter().rev().take(2).all(|pair| pair.pinned));
    }

    #[test]
    fn decide_keep_truncate_drop() {
        let pair = ToolPair {
            id: "t1".into(),
            call_id: "c1".into(),
            tool: "read".into(),
            input: "{}".into(),
            call_index: 1,
            result_index: 2,
            result_chars: 10,
            is_error: false,
            pinned: false,
        };
        assert_eq!(decide_call(&pair, 0.9, 0.9).action, CallAction::Keep);
        assert_eq!(decide_call(&pair, 0.9, 0.1).action, CallAction::DropResult);
        assert_eq!(decide_call(&pair, 0.1, 0.1).action, CallAction::DropCall);
        let mut pinned = pair.clone();
        pinned.pinned = true;
        assert_eq!(decide_call(&pinned, 0.0, 0.0).action, CallAction::Keep);
    }

    #[test]
    fn apply_never_drops_user_lines() {
        let messages = vec![
            user("review this"),
            call("c1", "read", "src/a.rs"),
            result("c1", &"x".repeat(500)),
            user("fix it"),
        ];
        let next = apply_decisions(
            &messages,
            &[CallDecision {
                id: "t1".into(),
                call_id: "c1".into(),
                tool: "read".into(),
                action: CallAction::DropCall,
            }],
        );
        assert_eq!(next[0].text(), "review this");
        assert_eq!(next.last().unwrap().text(), "fix it");
        assert!(next.iter().all(|message| message.role != Role::Tool));
    }

    #[test]
    fn truncate_uses_omitted_suffix() {
        let messages = vec![
            user("go"),
            call("c1", "read", "src/a.rs"),
            result("c1", &"y".repeat(800)),
        ];
        let next = apply_decisions(
            &messages,
            &[CallDecision {
                id: "t1".into(),
                call_id: "c1".into(),
                tool: "read".into(),
                action: CallAction::DropResult,
            }],
        );
        let output = next.iter().find_map(|message| {
            message.content.iter().find_map(|block| match block {
                Content::ToolResult { output, .. } => Some(output.as_str()),
                _ => None,
            })
        });
        let output = output.expect("truncated result");
        assert!(output.ends_with(OMITTED_SUFFIX), "{output}");
        assert!(output.contains("yyyy"), "head of the body must remain");
        assert!(next.iter().any(|message| {
            message
                .content
                .iter()
                .any(|block| matches!(block, Content::ToolCall { id, .. } if id == "c1"))
        }));
    }

    #[test]
    fn malformed_noul_defaults_to_keep() {
        let answers = json!({"call_t1": {"noul": "nope"}});
        assert!(noul_answer(&answers, "call_t1").is_none());
        let pair = ToolPair {
            id: "t1".into(),
            call_id: "c1".into(),
            tool: "read".into(),
            input: "{}".into(),
            call_index: 1,
            result_index: 2,
            result_chars: 4,
            is_error: false,
            pinned: false,
        };
        let decision = decide_call(&pair, 1.0, 1.0);
        assert_eq!(decision.action, CallAction::Keep);
    }

    #[tokio::test]
    async fn prune_drops_low_score_calls_over_http() {
        let body = r#"{"answers":{"call_t1":{"type":"noul","noul":0.1},"result_t1":{"type":"noul","noul":0.1},"call_t2":{"noul":0.1},"result_t2":{"noul":0.1},"call_t3":{"noul":0.1},"result_t3":{"noul":0.1},"call_t4":{"noul":0.1},"result_t4":{"noul":0.1},"call_t5":{"noul":0.1},"result_t5":{"noul":0.1},"call_t6":{"noul":0.1},"result_t6":{"noul":0.1},"call_t7":{"noul":0.1},"result_t7":{"noul":0.1},"call_t8":{"noul":0.1},"result_t8":{"noul":0.1}}}"#;
        let url = serve_json_loop(200, body, 4).await;
        let client = TypesafeClient::new("test-key", url, "jev-latest");
        let outcome = prune_messages(&long_history(), &client, None, &TurnCancellation::new())
            .await
            .expect("prune");
        assert!(outcome.changed);
        assert!(outcome.stats.calls_dropped > 0);
        let before = crate::compact::estimate_message_tokens(&long_history());
        let after = crate::compact::estimate_message_tokens(&outcome.messages);
        assert!(
            after < before,
            "prune must shrink occupancy {before} -> {after}"
        );
        assert!(
            outcome
                .messages
                .iter()
                .any(|message| message.role == Role::User && message.text() == "review this")
        );
    }

    #[tokio::test]
    async fn prune_fails_open_on_cancel() {
        let cancel = TurnCancellation::new();
        cancel.cancel();
        let client = TypesafeClient::new("test-key", "http://127.0.0.1:1", "jev-latest");
        let err = prune_messages(&long_history(), &client, None, &cancel)
            .await
            .expect_err("cancelled");
        assert!(err.contains("cancelled"), "{err}");
    }

    #[tokio::test]
    async fn prune_fails_open_on_http_error() {
        let url = serve_json_loop(500, r#"{"error":"nope"}"#, 4).await;
        let client = TypesafeClient::new("test-key", url, "jev-latest");
        let err = prune_messages(&long_history(), &client, None, &TurnCancellation::new())
            .await
            .expect_err("http error");
        assert!(err.contains("http") || err.contains("500"), "{err}");
    }

    async fn serve_json_loop(status: u16, body: &'static str, n: usize) -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            for _ in 0..n {
                let Ok((mut stream, _)) = listener.accept().await else {
                    return;
                };
                let mut buf = vec![0u8; 8192];
                let mut data = Vec::new();
                loop {
                    let Ok(n) = stream.read(&mut buf).await else {
                        return;
                    };
                    if n == 0 {
                        break;
                    }
                    data.extend_from_slice(&buf[..n]);
                    let headers_end = data.windows(4).position(|window| window == b"\r\n\r\n");
                    let Some(headers_end) = headers_end else {
                        continue;
                    };
                    let headers = &data[..headers_end];
                    let content_length = std::str::from_utf8(headers)
                        .ok()
                        .and_then(|headers| {
                            headers.lines().find_map(|line| {
                                let (name, value) = line.split_once(':')?;
                                name.eq_ignore_ascii_case("content-length")
                                    .then(|| value.trim().parse::<usize>().ok())
                                    .flatten()
                            })
                        })
                        .unwrap_or(0);
                    if data.len() >= headers_end + 4 + content_length {
                        break;
                    }
                }
                let reason = if status == 200 { "OK" } else { "Error" };
                let response = format!(
                    "HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                );
                let _ = stream.write_all(response.as_bytes()).await;
            }
        });
        format!("http://{addr}")
    }
}
