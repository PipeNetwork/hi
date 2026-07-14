//! Small heuristics and formatters used by the agent loop.

use hi_ai::{Content, ToolMode};
use hi_tools::{PlanStatus, PlanStep, ToolOutcome};

use crate::transcript::Transcript;
use crate::ui::Ui;

/// Whether a string is a known tool name.
fn is_valid_tool_name(name: &str) -> bool {
    hi_tools::is_known_tool(name)
}

/// Parse a single tool-call JSON object from a substring starting at `{`.
/// Returns `(name, arguments_json_string, end_index)` if the object is a
/// valid tool call, or `None` if it's not. `end_index` is one past the
/// closing `}`.
fn try_parse_tool_call_json(s: &str, start: usize) -> Option<(String, String, usize)> {
    // Walk forward to find the matching closing brace, respecting string
    // literals and nested objects/arrays.
    let bytes = s.as_bytes();
    let mut depth = 0i32;
    let mut in_string = false;
    let mut escape = false;
    let mut i = start;
    while i < bytes.len() {
        let b = bytes[i];
        if in_string {
            if escape {
                escape = false;
            } else if b == b'\\' {
                escape = true;
            } else if b == b'"' {
                in_string = false;
            }
        } else if b == b'"' {
            in_string = true;
        } else if b == b'{' || b == b'[' {
            depth += 1;
        } else if b == b'}' || b == b']' {
            depth -= 1;
            if depth == 0 && b == b'}' {
                break;
            }
        }
        i += 1;
    }
    if i >= bytes.len() {
        return None; // No matching close brace
    }
    let json_str = &s[start..=i];
    let value: serde_json::Value = serde_json::from_str(json_str).ok()?;
    let obj = value.as_object()?;
    // Accept {"name": "...", "arguments": {...}} or
    // {"name": "...", "arguments": "..."} (string form).
    let name = obj.get("name")?.as_str()?;
    if !is_valid_tool_name(name) {
        return None;
    }
    let arguments = match obj.get("arguments") {
        Some(serde_json::Value::String(s)) => s.clone(),
        Some(other) => other.to_string(),
        None => "{}".to_string(),
    };
    Some((name.to_string(), arguments, i + 1))
}

/// Parse the XML-ish text tool protocol some smaller OpenAI-compatible models
/// emit, for example:
/// `<tool_call>bash<arg_key>command</arg_key><arg_value>echo hi</arg_value>`.
///
/// A closing `</tool_call>` is accepted but not required; the call ends after the
/// last complete arg pair. Incomplete arg values are ignored so truncation can be
/// recovered by the normal continue path instead of executing a partial command.
fn try_parse_xml_tool_call(s: &str, start: usize) -> Option<(String, String, usize)> {
    const START: &str = "<tool_call>";
    const END: &str = "</tool_call>";
    const ARG_KEY_START: &str = "<arg_key>";
    const ARG_KEY_END: &str = "</arg_key>";
    const ARG_VALUE_START: &str = "<arg_value>";
    const ARG_VALUE_END: &str = "</arg_value>";

    let rest = s.get(start..)?;
    if !rest.starts_with(START) {
        return None;
    }

    let mut pos = start + START.len();
    while s.as_bytes().get(pos).is_some_and(u8::is_ascii_whitespace) {
        pos += 1;
    }
    let name_start = pos;
    while let Some(ch) = s[pos..].chars().next() {
        if ch == '<' || ch.is_whitespace() {
            break;
        }
        pos += ch.len_utf8();
    }
    let name = s[name_start..pos].trim();
    if !is_valid_tool_name(name) {
        return None;
    }

    let mut args = serde_json::Map::new();
    let mut saw_arg = false;
    loop {
        while s.as_bytes().get(pos).is_some_and(u8::is_ascii_whitespace) {
            pos += 1;
        }
        if s[pos..].starts_with(END) {
            pos += END.len();
            break;
        }
        if !s[pos..].starts_with(ARG_KEY_START) {
            break;
        }
        let key_start = pos + ARG_KEY_START.len();
        let key_rel_end = s[key_start..].find(ARG_KEY_END)?;
        let key_end = key_start + key_rel_end;
        let key = s[key_start..key_end].trim();
        if key.is_empty() {
            return None;
        }
        pos = key_end + ARG_KEY_END.len();

        while s.as_bytes().get(pos).is_some_and(u8::is_ascii_whitespace) {
            pos += 1;
        }
        if !s[pos..].starts_with(ARG_VALUE_START) {
            return None;
        }
        let value_start = pos + ARG_VALUE_START.len();
        let value_rel_end = s[value_start..].find(ARG_VALUE_END)?;
        let value_end = value_start + value_rel_end;
        let value = &s[value_start..value_end];
        args.insert(
            key.to_string(),
            serde_json::Value::String(value.to_string()),
        );
        saw_arg = true;
        pos = value_end + ARG_VALUE_END.len();
    }

    if !saw_arg {
        return None;
    }
    Some((
        name.to_string(),
        serde_json::Value::Object(args).to_string(),
        pos,
    ))
}

/// Scan assistant text for tool-call-like JSON patterns and convert them into
/// `Content::ToolCall` blocks. This is a fallback for local models (Ollama,
/// llama.cpp, etc.) that emit tool calls as text instead of using the
/// structured `tool_calls` API field.
///
/// Recognizes patterns like:
/// - `{"name": "bash", "arguments": {"command": "ls"}}`
/// - `{"name": "read", "arguments": "{\"path\": \"foo.rs\"}"}`
///
/// Returns the parsed tool calls (with generated IDs) and the text with the
/// tool-call JSON removed, so the assistant message recorded in history
/// contains only the prose, not the raw JSON.
///
/// `id_offset` is the number of `textcall_` IDs already present in the
/// transcript — new IDs start at `textcall_{id_offset}` so they're globally
/// unique across the whole conversation, not just this message. Without the
/// offset, every assistant message reuses `textcall_0`, `textcall_1`, … and
/// providers reject the duplicate IDs with a 400 on the next request (e.g.
/// after switching from a local model to a hosted one mid-session).
/// Returns the interleaved content blocks — prose `Text` segments and
/// `ToolCall` blocks in their original emission order — so that trailing
/// prose *after* a tool call stays after it in the recorded message, not
/// merged before it. Each prose segment is trailing-trimmed (newlines left
/// by the JSON sitting on its own line) but leading whitespace is preserved.
/// An empty leading segment (tool call at the very start) is omitted.
pub(crate) fn parse_text_tool_calls(text: &str, id_offset: usize) -> Vec<Content> {
    // Strip ChatML special tokens (<|im_start|>, <|im_end|>, …) that some
    // local models emit as raw text. The streaming layer already strips them,
    // but this is a defense-in-depth for any text that arrives unstripped.
    let text = strip_chatml_tokens(text);
    let mut out = Vec::new();
    let mut call_count = 0usize;
    let bytes = text.as_bytes();
    let mut i = 0;
    let mut search_from = 0;

    while i < bytes.len() {
        let parsed = if bytes[i] == b'{' {
            try_parse_tool_call_json(&text, i)
        } else if bytes[i] == b'<' {
            try_parse_xml_tool_call(&text, i)
        } else {
            None
        };
        if let Some((name, arguments, end)) = parsed {
            // Emit the prose before this tool call as a Text block.
            let prose = text[search_from..i].trim_end();
            if !prose.is_empty() {
                out.push(Content::Text(prose.to_string()));
            }
            let id = format!("textcall_{}", id_offset + call_count);
            out.push(Content::ToolCall {
                id,
                name,
                arguments,
            });
            call_count += 1;
            i = end;
            search_from = end;
            continue;
        }
        i += 1;
    }
    // Emit any trailing prose after the last tool call.
    let trailing = text[search_from..].trim_end();
    if !trailing.is_empty() {
        out.push(Content::Text(trailing.to_string()));
    }

    out
}

/// Count how many `textcall_` tool-call IDs already exist in the transcript, so
/// the next batch of text-parsed tool calls gets globally-unique IDs instead of
/// restarting from `textcall_0` every message. Scans both assistant `ToolCall`
/// blocks and tool `ToolResult` blocks (which carry the matching `call_id`).
pub(crate) fn textcall_id_offset(messages: &Transcript) -> usize {
    messages
        .as_slice()
        .iter()
        .flat_map(|m| m.content.iter())
        .filter_map(|c| match c {
            Content::ToolCall { id, .. } => Some(id.as_str()),
            Content::ToolResult { call_id, .. } => Some(call_id.as_str()),
            _ => None,
        })
        .filter_map(|id| id.strip_prefix("textcall_"))
        .filter_map(|n| n.parse::<usize>().ok())
        .max()
        .map(|m| m + 1)
        .unwrap_or(0)
}

/// Strip ChatML special tokens (`<|…|>`) from text. The token content must be
/// alphanumeric/underscore — this avoids eating literal text like `<|foo bar|>`.
fn strip_chatml_tokens(text: &str) -> String {
    let bytes = text.as_bytes();
    let mut out = String::with_capacity(text.len());
    let mut last = 0;
    let mut i = 0;
    while i < bytes.len() {
        // Search the suffix from after the `<|` prefix so the prefix's own `|`
        // can't match `|>` (the text `<|>` produced a reversed slice range and
        // panicked).
        if bytes[i] == b'<'
            && i + 1 < bytes.len()
            && bytes[i + 1] == b'|'
            && let Some(end) = text[i + 2..].find("|>")
        {
            let inner = &text[i + 2..i + 2 + end];
            if !inner.is_empty() && inner.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
                out.push_str(&text[last..i]);
                last = i + 2 + end + 2;
                i = last;
                continue;
            }
        }
        i += 1;
    }
    out.push_str(&text[last..]);
    out
}

/// Route a tool's output to the right UI surface: a plan update drives the live
/// tracker (in place), everything else renders as a tool result — its richer
/// `display` if present, else the model-facing `content`.
pub(crate) fn emit_tool_output(ui: &mut dyn Ui, name: &str, output: &ToolOutcome) {
    if let Some(plan) = output.plan.as_deref() {
        ui.plan(plan);
    } else {
        ui.tool_result(name, output.display.as_deref().unwrap_or(&output.content));
    }
}

/// Infer within-batch tool-call dependencies so the executor can honor the
/// model's intent rather than relying on emission-order coincidence. Returns,
/// for each call index, the set of earlier call indices it must run *after*.
///
/// Rules (conservative — over-serializing is safe, under-serializing is a bug):
/// - A mutating call (`write`/`edit`/`multi_edit`/`bash`/`apply_patch`) depends
///   on every earlier mutating call, so side effects apply in emission order.
///   (Two independent writes still serialize — file edits aren't commutative
///   and a later write may depend on an earlier write's content.) It also
///   depends on any earlier *read* of the same path (write-after-read): "read
///   a.rs, then write a.rs" must let the read observe the pre-write content, not
///   a file being truncated/rewritten under it. A mutation with an unknown write
///   path (`bash`) conservatively waits for every earlier read.
/// - A read-only call depends on any earlier mutating call whose inferred
///   target path matches the read's target path — so "write a.rs, then read
///   a.rs" reads the post-write state even if a scheduler reorders independent
///   reads. Reads with no path overlap with earlier mutations have no deps and
///   may parallelize freely.
///
/// `calls` is `(id, name, arguments)` per the executor's shape. A call with an
/// unparseable target path is treated as dependent on all earlier mutations
/// (the safe fallback — `target_path` returns `None` for `bash`, so a `bash`
/// edit followed by a read serializes).
pub(crate) fn tool_deps(calls: &[(String, String, String)]) -> Vec<Vec<usize>> {
    let n = calls.len();
    let mut deps = vec![Vec::new(); n];
    // Track, for each prior index, whether it was mutating and its target path.
    let mut prior: Vec<(bool, Option<String>)> = Vec::with_capacity(n);
    for (i, (_, name, arguments)) in calls.iter().enumerate() {
        let mutating = !hi_tools::is_read_only(name);
        let my_path = hi_tools::target_path(name, arguments);
        for (j, (was_mut, their_path)) in prior.iter().enumerate() {
            let must_wait = if mutating {
                // Serialize after all earlier mutations (was_mut), and after an
                // earlier read of the same path (write-after-read: the read must
                // see the pre-write file). A mutation with an unknown write path
                // (bash) conservatively waits for every earlier read, since
                // paths_overlap treats an unknown path as overlapping.
                *was_mut || paths_overlap(their_path.as_deref(), my_path.as_deref())
            } else {
                // Reads wait for an earlier mutation on the same path. If
                // either side has no parseable path, be safe and serialize
                // (covers `bash` edits, which have no path).
                *was_mut && paths_overlap(their_path.as_deref(), my_path.as_deref())
            };
            if must_wait {
                deps[i].push(j);
            }
        }
        prior.push((mutating, my_path));
    }
    deps
}

/// Whether two (possibly-unknown) target paths refer to the same file. `None`
/// on either side means "unknown" — treat as overlapping (the safe choice:
/// serialize rather than risk a read observing a pre-mutation state).
fn paths_overlap(a: Option<&str>, b: Option<&str>) -> bool {
    match (a, b) {
        (Some(a), Some(b)) => a == b,
        // Unknown on either side → conservatively overlap.
        _ => true,
    }
}

/// Whether an execution `order` (a permutation of `0..n`) respects the
/// dependency graph from [`tool_deps`]: every call appears after all of its
/// dependencies. Used as a debug assertion / property-test oracle so a future
/// scheduler change can't regress the "read-after-write observes the write"
/// invariant.
pub(crate) fn respects_deps(deps: &[Vec<usize>], order: &[usize]) -> bool {
    if order.len() != deps.len() {
        return false;
    }
    let mut seen = vec![false; deps.len()];
    for &idx in order {
        let Some(slot) = seen.get_mut(idx) else {
            return false;
        };
        if *slot {
            return false;
        }
        *slot = true;
    }

    let pos = |idx: usize| order.iter().position(|&o| o == idx);
    for (i, ds) in deps.iter().enumerate() {
        let Some(my_pos) = pos(i) else {
            return false;
        };
        for &d in ds {
            let Some(dep_pos) = pos(d) else {
                return false;
            };
            if dep_pos > my_pos {
                return false;
            }
        }
    }
    true
}

/// Humanize a token count compactly and consistently: `991`, `1.2k`, `22k`, `1.0M`.
/// Shared by the live working line and the settled usage summary so they agree.
pub fn humanize_count(n: u64) -> String {
    match n {
        0..=999 => n.to_string(),
        1_000..=9_999 => format!("{:.1}k", n as f64 / 1000.0),
        10_000..=999_999 => format!("{}k", n / 1000),
        _ => format!("{:.1}M", n as f64 / 1_000_000.0),
    }
}

pub(crate) fn tool_mode_label(mode: ToolMode) -> &'static str {
    match mode {
        ToolMode::Auto => "auto",
        ToolMode::Required => "required",
        ToolMode::ChatOnly => "chat-only",
        ToolMode::ReadOnly => "read-only",
    }
}

/// Whether the session `tool_mode` forbids *executing* `name`, returning the
/// synthetic blocked-tool result to feed back to the model if so.
///
/// This enforces the mode at execution time, not just via tool advertisement —
/// which is what closes the text-promoted tool-call hole: a local model can emit
/// a tool call as prose (`{"name":"write",…}`) that never went through the
/// advertised tool list, so a ChatOnly/ReadOnly session (including every
/// `explore` subagent) would otherwise run it. `explore` launches only a
/// read-only child, so it's allowed under ReadOnly (mirroring the advertisement
/// rules and [`crate::steering::nudges::read_only_blocks_tool`]).
pub(crate) fn mode_blocks_tool(mode: ToolMode, name: &str) -> Option<String> {
    match mode {
        ToolMode::Auto | ToolMode::Required => None,
        ToolMode::ChatOnly => Some(format!(
            "Tool `{name}` blocked: this is a discuss-only turn (tool mode chat-only). \
             Answer in text without calling tools."
        )),
        ToolMode::ReadOnly if !hi_tools::is_read_only(name) && name != "explore" => Some(format!(
            "Tool `{name}` blocked: this session is read-only (tool mode read-only). \
             Use read-only inspection tools and do not modify files."
        )),
        ToolMode::ReadOnly => None,
    }
}

pub(crate) fn looks_mutating(input: &str) -> bool {
    let s = input.to_ascii_lowercase();
    [
        "edit",
        "fix",
        "change",
        "update",
        "write",
        "create",
        "delete",
        "remove",
        "rename",
        "implement",
        "add ",
        "modify",
        "refactor",
        "format",
        "run ",
    ]
    .iter()
    .any(|needle| s.contains(needle))
}

/// Heuristic: does the model's final text read like an *announced but unperformed*
/// next step — e.g. "Now let me rewrite main.rs:" or a "Here's my plan:" followed
/// by a numbered to-do list — rather than a finished answer or a past-tense recap?
///
/// It judges the trailing non-empty line, with one twist: when the message trails
/// off into a plan/to-do list, the intent lives in the line that *introduces* the
/// list ("Here's my plan:"), not the last bullet — so it judges that lead-in
/// instead, and only when the lead-in looks forward. That way a proper codex-style
/// recap that ends in a bullet list ("Key changes:\n- …") doesn't read as a stall,
/// while a model that announces a plan and quits without doing it does.
pub(crate) fn looks_like_unfinished_step(text: &str) -> bool {
    let lines: Vec<&str> = text
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .collect();
    let Some(&last) = lines.last() else {
        return false;
    };
    if is_list_item(last) {
        // Trailing plan/to-do list: unfinished only if the line introducing it
        // looks forward ("Here's my plan:"). A past-tense recap list is done.
        let lead = lines
            .iter()
            .rev()
            .find(|l| !is_list_item(l))
            .copied()
            .unwrap_or(last);
        return is_forward_intent(lead);
    }
    // Otherwise judge the trailing line: a dangling colon ("Now let me rewrite
    // main.rs:") or a forward-looking phrase means work was announced, not done.
    last.ends_with(':') || is_forward_intent(last)
}

/// Whether a plan has unfinished work — any step that is `Pending` or `Active`.
/// Used by the continue logic to keep the turn going when the model stops
/// calling tools but the plan isn't complete. The model often writes a
/// finished-looking recap after one sub-task ("I've implemented proof.rs."),
/// which the text-based `looks_like_unfinished_step` heuristic can't catch —
/// but the plan state (2/9 done) is unambiguous.
pub(crate) fn plan_has_pending_steps(steps: &[PlanStep]) -> bool {
    steps
        .iter()
        .any(|s| s.status == PlanStatus::Pending || s.status == PlanStatus::Active)
}

/// Whether a user input looks like a "continue" command — a short prompt
/// asking the agent to keep going, as opposed to a new task. Used to decide
/// whether to persist the plan state across turns: a "continue" on an
/// incomplete plan should keep the plan so the plan-aware continue logic can
/// fire; a new task should clear it so a stale plan doesn't cause spurious
/// nudges.
pub(crate) fn looks_like_continue(input: &str) -> bool {
    let lower = input.trim().to_lowercase();
    if lower.len() > 50 {
        return false; // A continue command is short; a new task is longer.
    }
    const CONTINUE_PHRASES: &[&str] = &[
        "continue",
        "keep going",
        "go on",
        "next",
        "proceed",
        "resume",
        "carry on",
        "finish it",
        "finish up",
        "do the rest",
        "do the remaining",
        "keep working",
    ];
    CONTINUE_PHRASES
        .iter()
        .any(|p| lower == *p || lower.starts_with(p))
}

/// Whether a line expresses *intent to act next* rather than a finished result.
#[allow(dead_code)]
pub(crate) fn is_forward_intent(line: &str) -> bool {
    let lower = line.to_lowercase();
    // Courtesy closings address the *user* ("let me know if…", "I'll be happy
    // to…", "I'll let you know…") — they read like forward phrases but mean the
    // turn is finished, not stalled. Vetoed first so they don't trigger a nudge.
    const CLOSINGS: [&str; 6] = [
        "let me know",
        "i'll be happy",
        "i'll let you",
        "i'll wait",
        "i'm happy to",
        "feel free",
    ];
    if CLOSINGS.iter().any(|c| lower.contains(c)) {
        return false;
    }
    if contains_action_ack(&lower) {
        return true;
    }
    const FORWARD_INTENT: [&str; 12] = [
        "let me ",
        "let's ",
        "i'll ",
        "i will ",
        "i'm going to",
        "i am going to",
        "proceed to ",
        "here's my plan",
        "here is my plan",
        "my plan",
        "i need to ",
        "next, i",
    ];
    FORWARD_INTENT.iter().any(|phrase| lower.contains(phrase))
}

pub(crate) fn contains_action_ack(lower: &str) -> bool {
    const ACKS: [&str; 2] = ["i can do that", "i can help with that"];
    if !ACKS.iter().any(|ack| lower.contains(ack)) {
        return false;
    }
    const ACTION_MARKERS: [&str; 15] = [
        "look into",
        "look at",
        "inspect",
        "scan",
        "check",
        "analyz",
        "review",
        "explore",
        "read",
        "open",
        "run",
        "test",
        "fix",
        "debug",
        "search",
    ];
    ACTION_MARKERS.iter().any(|marker| lower.contains(marker))
}

/// Whether a line is a markdown list item — a bullet (`- `, `* `, `• `) or a
/// numbered item (`1.`, `2)`) — used to spot a trailing plan/to-do list.
#[allow(dead_code)]
pub(crate) fn is_list_item(line: &str) -> bool {
    let l = line.trim_start();
    if l.starts_with("- ") || l.starts_with("* ") || l.starts_with("• ") {
        return true;
    }
    let digits = l.chars().take_while(|c| c.is_ascii_digit()).count();
    digits > 0 && l[digits..].starts_with(['.', ')'])
}

/// Whether recovery sampling (a hotter resample on an empty/garbled retry) is on.
/// Off (`HI_RECOVERY_SAMPLING=0/off/false/no`) re-runs the retry at the configured
/// sampling — the knob for A/B-ing recovery on the eval harness. Read once.
pub(crate) static RECOVERY_SAMPLING: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
    !matches!(
        std::env::var("HI_RECOVERY_SAMPLING").ok().as_deref(),
        Some("0" | "off" | "false" | "no")
    )
});

/// Sampling for a model round, escalating with the count of consecutive
/// content-less rounds (`retries`; 0 = the normal first attempt). Returns
/// `(temperature, top_p, frequency_penalty)`. On a normal round — or when recovery
/// sampling is disabled — it passes the configured temperature through and leaves
/// `top_p`/`frequency_penalty` at the provider default (`None`). On a retry it
/// leads with anti-repetition — nucleus sampling plus a growing frequency penalty
/// — and only gently raises temperature from a ≥0.5 floor, so a repetition/garble
/// loop is broken with less coding-quality risk than a big temperature jump.
pub(crate) fn recovery_sampling(
    retries: u32,
    base_temperature: Option<f32>,
    enabled: bool,
) -> (Option<f32>, Option<f32>, Option<f32>) {
    if !enabled || retries == 0 {
        return (base_temperature, None, None);
    }
    let r = retries as f32;
    let temperature = (base_temperature.unwrap_or(0.7).max(0.5) + 0.15 * r).min(1.0);
    let frequency_penalty = (0.3 * r).min(0.6);
    (Some(temperature), Some(0.95), Some(frequency_penalty))
}

/// Which stall mode fired and triggered recovery sampling. The retry counter
/// (`retries`) is shared across the empty-response path — repeat and continue
/// nudges don't currently escalate sampling, so they surface as `mode == …` with
/// `retries == 0` and produce no telemetry line (see `recovery_telemetry`).
///
/// `Repeat`/`Continue` are modeled but not yet constructed: the plan calls out a
/// separate experiment on whether they should escalate sampling too. They're
/// kept here so the telemetry shape is fixed when that lands.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum StallMode {
    /// A content-less/garbled round (`EmptyCompletion`/`MalformedStream`, or no
    /// text and no tool calls). The only mode recovery sampling escalates today.
    Empty,
    /// The model re-issued the previous round's exact tool calls.
    #[allow(dead_code)]
    Repeat,
    /// The model announced a next step but emitted no tool call to perform it.
    #[allow(dead_code)]
    Continue,
}

impl StallMode {
    pub(crate) fn label(self) -> &'static str {
        match self {
            StallMode::Empty => "empty retry",
            StallMode::Repeat => "repeat nudge",
            StallMode::Continue => "continue nudge",
        }
    }
}

/// Build the recovery-sampling telemetry line, or `None` when there's nothing to
/// report. Emits only when recovery sampling is actually changing params — i.e.
/// `enabled && retries > 0` (a normal first attempt, or recovery disabled, stays
/// silent so ordinary runs aren't noisy). That keeps it behind the
/// `HI_RECOVERY_SAMPLING` A/B knob without needing a separate debug env: the line
/// appears precisely when the knob is on *and* a retry is being resampled, which is
/// the signal the A/B needs to measure rather than just aggregate.
///
/// The line names the stall mode, the retry index out of the per-mode budget, and
/// the applied sampling params, e.g.
/// `recovery sampling: empty retry 1/2 · temp=0.65 top_p=0.95 freq=0.3`.
pub(crate) fn recovery_telemetry(
    mode: StallMode,
    retries: u32,
    budget: u32,
    temperature: Option<f32>,
    top_p: Option<f32>,
    frequency_penalty: Option<f32>,
    enabled: bool,
) -> Option<String> {
    if !enabled || retries == 0 {
        return None;
    }
    let fmt = |v: Option<f32>| v.map(|x| format!("{x:.2}")).unwrap_or_else(|| "—".into());
    Some(format!(
        "recovery sampling: {} {}/{} · temp={} top_p={} freq={}",
        mode.label(),
        retries,
        budget,
        fmt(temperature),
        fmt(top_p),
        fmt(frequency_penalty),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tool_deps_serializes_write_after_read_on_same_path() {
        // read a.rs then write a.rs: the write must wait for the read so the read
        // observes the pre-write file (previously they ran concurrently, so the
        // read could see a torn/post-write file).
        let calls = vec![
            ("r".into(), "read".into(), r#"{"path":"a.rs"}"#.into()),
            (
                "w".into(),
                "write".into(),
                r#"{"path":"a.rs","content":"x"}"#.into(),
            ),
        ];
        let deps = tool_deps(&calls);
        assert!(deps[0].is_empty(), "the read has no deps");
        assert_eq!(deps[1], vec![0], "the write waits for the same-path read");

        // read a.rs then write b.rs: different files, still independent.
        let calls = vec![
            ("r".into(), "read".into(), r#"{"path":"a.rs"}"#.into()),
            (
                "w".into(),
                "write".into(),
                r#"{"path":"b.rs","content":"x"}"#.into(),
            ),
        ];
        let deps = tool_deps(&calls);
        assert!(
            deps[1].is_empty(),
            "a write to a different file is independent of the read"
        );
    }

    #[test]
    fn mode_blocks_tool_enforces_session_mode() {
        // ChatOnly blocks every tool (nothing runs, not even reads).
        assert!(mode_blocks_tool(ToolMode::ChatOnly, "read").is_some());
        assert!(mode_blocks_tool(ToolMode::ChatOnly, "write").is_some());
        // ReadOnly blocks mutating tools but allows inspection + `explore`.
        assert!(mode_blocks_tool(ToolMode::ReadOnly, "write").is_some());
        assert!(mode_blocks_tool(ToolMode::ReadOnly, "bash").is_some());
        assert!(mode_blocks_tool(ToolMode::ReadOnly, "read").is_none());
        assert!(mode_blocks_tool(ToolMode::ReadOnly, "grep").is_none());
        assert!(mode_blocks_tool(ToolMode::ReadOnly, "explore").is_none());
        // Auto/Required never block by mode.
        assert!(mode_blocks_tool(ToolMode::Auto, "write").is_none());
        assert!(mode_blocks_tool(ToolMode::Required, "bash").is_none());
    }

    #[test]
    fn humanize_count_abbreviates_consistently() {
        assert_eq!(humanize_count(0), "0");
        assert_eq!(humanize_count(991), "991");
        assert_eq!(humanize_count(1234), "1.2k");
        assert_eq!(humanize_count(22864), "22k"); // the reported "22864 in"
        assert_eq!(humanize_count(12000), "12k"); // the reported "12k" ctx
        assert_eq!(humanize_count(999_999), "999k"); // last "k" before switching
        assert_eq!(humanize_count(1_000_000), "1.0M"); // a 1M window
        // A long session's cumulative input must read as millions, never a
        // 5-digit "15528k" (the pre-fix formatter that prompted this).
        assert_eq!(humanize_count(15_528_000), "15.5M");
    }

    #[test]
    fn unfinished_step_heuristic() {
        for t in [
            "Now let me rewrite main.rs:",
            "I'll add the struct",
            "Here is the plan:",
            // A "plan:" lead-in followed by a numbered to-do list — the trailing
            // line is a list item, so the lead-in is what's judged. (This is the
            // case the old line-only heuristic missed, ending the turn mid-plan.)
            "Now let me make the fixes. Here's my plan:\n\n1. Remove deps\n2. Fix gitignore\n3. Drop dead code",
        ] {
            assert!(looks_like_unfinished_step(t), "should flag: {t:?}");
        }
        for t in [
            "Done. Run `cargo build`.",
            "The answer is 42.",
            "I changed foo.rs and bar.rs.",
            // A past-tense recap that ends in a bullet list is finished, not a
            // stall — the lead-in ("Key changes:") looks back, not forward.
            "Key changes:\n- Added GOP support in encoder.rs\n- Updated the CLI in main.rs",
            // Courtesy closings address the user — a finished turn, not a stall —
            // even though they contain "let me"/"I'll". These used to false-nudge.
            "All done. Let me know if you'd like any changes.",
            "I'll be happy to help with anything else.",
            "Implemented and tested. I'll let you know if I spot any issues.",
            "Fixed it — feel free to ask if you want more detail.",
        ] {
            assert!(!looks_like_unfinished_step(t), "should not flag: {t:?}");
        }
        for t in [
            "I can do that by reviewing the repo files first.",
            "I can help with that by running the tests.",
        ] {
            assert!(
                looks_like_unfinished_step(t),
                "action ack should flag: {t:?}"
            );
        }
        for t in [
            "I can do that.",
            "I can help with that.",
            "I can help with that if you share more detail.",
        ] {
            assert!(
                !looks_like_unfinished_step(t),
                "plain acknowledgement should not flag: {t:?}"
            );
        }
    }

    #[test]
    fn plan_pending_steps_heuristic() {
        let step = |status: PlanStatus| PlanStep {
            title: "x".into(),
            status,
        };
        // All done → no pending work.
        assert!(!plan_has_pending_steps(&[
            step(PlanStatus::Done),
            step(PlanStatus::Done),
        ]));
        // Has a pending step → unfinished.
        assert!(plan_has_pending_steps(&[
            step(PlanStatus::Done),
            step(PlanStatus::Pending),
        ]));
        // Has an active step → unfinished.
        assert!(plan_has_pending_steps(&[
            step(PlanStatus::Done),
            step(PlanStatus::Active),
            step(PlanStatus::Pending),
        ]));
        // Empty plan → no pending work (no plan to complete).
        assert!(!plan_has_pending_steps(&[]));
    }

    #[test]
    fn looks_like_continue_heuristic() {
        // Short continue commands.
        for s in [
            "continue",
            "Continue",
            "CONTINUE",
            "keep going",
            "go on",
            "next",
            "proceed",
            "resume",
            "carry on",
            "finish it",
            "do the rest",
            "keep working",
            "  continue  ",
        ] {
            assert!(looks_like_continue(s), "should flag as continue: {s:?}");
        }
        // New tasks — should NOT be flagged as continue.
        for s in [
            "fix the bug in parser.rs",
            "implement a new feature for the CLI",
            "review the codebase and suggest improvements",
            "write tests for the auth module",
            "refactor the error handling to use anyhow",
            // Too long even if it starts with "continue".
            "continue working on the feature but also make sure to handle the edge case where the input is empty and the user has not provided a valid path",
        ] {
            assert!(
                !looks_like_continue(s),
                "should NOT flag as continue: {s:?}"
            );
        }
    }

    #[test]
    fn recovery_sampling_escalates_and_toggles() {
        // Normal round: pass the configured temperature through, no overrides.
        assert_eq!(
            recovery_sampling(0, Some(0.2), true),
            (Some(0.2), None, None)
        );
        // First retry: nucleus + frequency penalty lead; temperature rises only
        // gently from the 0.5 floor (to ~0.65, well under the old 0.85).
        let (t1, p1, f1) = recovery_sampling(1, Some(0.2), true);
        assert_eq!((p1, f1), (Some(0.95), Some(0.3)));
        assert!(
            t1.unwrap() > 0.2 && t1.unwrap() < 0.7,
            "temp climbs gently: {t1:?}"
        );
        // Second retry climbs further; temperature and penalty stay bounded.
        let (t2, _, f2) = recovery_sampling(2, Some(0.2), true);
        assert!(t2.unwrap() > t1.unwrap(), "temp keeps climbing");
        assert!(f2.unwrap() > f1.unwrap(), "penalty grows");
        assert!(t2.unwrap() <= 1.0 && f2.unwrap() <= 0.6, "both bounded");
        // Disabled: a retry behaves like a normal round (no overrides).
        assert_eq!(
            recovery_sampling(2, Some(0.2), false),
            (Some(0.2), None, None)
        );
    }

    #[test]
    fn recovery_telemetry_only_when_params_change() {
        // A retry with recovery on names the stall mode, retry index, budget, and
        // the applied sampling params.
        let line = recovery_telemetry(
            StallMode::Empty,
            1,
            2,
            Some(0.65),
            Some(0.95),
            Some(0.3),
            true,
        )
        .expect("retry with recovery on should produce a line");
        assert!(
            line.contains("empty retry 1/2"),
            "expected mode + retry/budget, got {line:?}"
        );
        assert!(
            line.contains("temp=0.65") && line.contains("top_p=0.95") && line.contains("freq=0.30"),
            "expected applied params, got {line:?}"
        );

        // A normal first attempt (retries == 0) is silent regardless of mode or
        // enabled state — ordinary runs must not be noisy.
        assert_eq!(
            recovery_telemetry(StallMode::Empty, 0, 2, Some(0.2), None, None, true),
            None,
            "retries == 0 should not emit"
        );
        // Repeat/continue nudges don't escalate sampling (retries stays 0), so they
        // produce no telemetry line.
        assert_eq!(
            recovery_telemetry(StallMode::Repeat, 0, 3, Some(0.2), None, None, true),
            None,
        );
        assert_eq!(
            recovery_telemetry(StallMode::Continue, 0, 3, Some(0.2), None, None, true),
            None,
        );
        // Recovery disabled: a retry behaves like a normal round, so no line.
        assert_eq!(
            recovery_telemetry(StallMode::Empty, 2, 2, Some(0.2), None, None, false),
            None,
            "disabled recovery should not emit"
        );
    }

    #[test]
    fn tool_deps_serializes_read_after_write_on_same_path() {
        // write a.rs, read a.rs, read b.rs: the read of a.rs depends on the
        // write; the read of b.rs does not (different path) — independent.
        let calls = vec![
            (
                "w".into(),
                "write".into(),
                r#"{"path":"a.rs","content":"x"}"#.into(),
            ),
            ("r1".into(), "read".into(), r#"{"path":"a.rs"}"#.into()),
            ("r2".into(), "read".into(), r#"{"path":"b.rs"}"#.into()),
        ];
        let deps = tool_deps(&calls);
        // write (0) has no deps.
        assert!(deps[0].is_empty(), "write has no deps: {deps:?}");
        // read a.rs (1) depends on the write (0).
        assert!(deps[1].contains(&0), "read a.rs depends on write: {deps:?}");
        // read b.rs (2) is independent of the write on a.rs.
        assert!(
            !deps[2].contains(&0),
            "read b.rs independent of write a.rs: {deps:?}"
        );
    }

    #[test]
    fn tool_deps_serializes_mutating_calls_in_emission_order() {
        // Two writes: the second depends on the first (edits aren't commutative).
        let calls = vec![
            (
                "w1".into(),
                "write".into(),
                r#"{"path":"a.rs","content":"1"}"#.into(),
            ),
            (
                "w2".into(),
                "edit".into(),
                r#"{"path":"a.rs","old_string":"1","new_string":"2"}"#.into(),
            ),
        ];
        let deps = tool_deps(&calls);
        assert!(deps[0].is_empty());
        assert!(
            deps[1].contains(&0),
            "second write depends on first: {deps:?}"
        );
    }

    #[test]
    fn tool_deps_bash_edit_serializes_following_read() {
        // A bash edit has no parseable path, so a following read is conservatively
        // serialized after it (the safe fallback — the read might observe the
        // bash edit's effect).
        let calls = vec![
            (
                "b".into(),
                "bash".into(),
                r#"{"command":"echo x > a.rs"}"#.into(),
            ),
            ("r".into(), "read".into(), r#"{"path":"a.rs"}"#.into()),
        ];
        let deps = tool_deps(&calls);
        assert!(
            deps[1].contains(&0),
            "read after bash edit serializes: {deps:?}"
        );
    }

    #[test]
    fn tool_deps_multi_file_apply_patch_serializes_following_read() {
        // A multi-file apply_patch has no single target path. Treat it as an
        // unknown-path mutation so a same-batch read of any patched file waits
        // for the patch instead of racing and seeing stale content.
        let calls = vec![
            (
                "p".into(),
                "apply_patch".into(),
                serde_json::json!({
                    "patch": "*** Begin Patch\n*** Update File: a.rs\n-old\n+new\n*** Update File: b.rs\n-old\n+new\n*** End Patch"
                })
                .to_string(),
            ),
            ("r".into(), "read".into(), r#"{"path":"b.rs"}"#.into()),
        ];
        let deps = tool_deps(&calls);
        assert!(
            deps[1].contains(&0),
            "read after multi-file apply_patch serializes: {deps:?}"
        );
    }

    #[test]
    fn respects_deps_validates_ordering() {
        // deps: call 1 depends on 0; call 2 depends on 0.
        let deps = vec![vec![], vec![0], vec![0]];
        // Emission order respects deps.
        assert!(respects_deps(&deps, &[0, 1, 2]));
        // Reordering 0 after 1 violates (1 depends on 0).
        assert!(!respects_deps(&deps, &[1, 0, 2]));
        // 2 before 1 is fine (2 doesn't depend on 1).
        assert!(respects_deps(&deps, &[0, 2, 1]));
        // A partial scheduler result is invalid, not a panic.
        assert!(!respects_deps(&deps, &[0, 1]));
        // Duplicating one completed call while omitting another is also invalid.
        assert!(!respects_deps(&deps, &[0, 1, 1]));
        // Duplicates are invalid even when every call appears at least once.
        assert!(!respects_deps(&deps, &[0, 1, 2, 1]));
        // Out-of-range completion indices are invalid.
        assert!(!respects_deps(&deps, &[0, 1, 3]));
    }

    #[test]
    fn emission_order_respects_inferred_deps_for_a_realistic_batch() {
        // The property the executor pins: for a realistic mixed batch, the
        // emission order [0,1,2,...] always respects the inferred deps (since
        // deps only point backward). This is the regression guard.
        let calls = vec![
            ("r0".into(), "read".into(), r#"{"path":"a.rs"}"#.into()),
            (
                "w".into(),
                "write".into(),
                r#"{"path":"a.rs","content":"x"}"#.into(),
            ),
            ("r1".into(), "read".into(), r#"{"path":"a.rs"}"#.into()),
            ("r2".into(), "read".into(), r#"{"path":"b.rs"}"#.into()),
        ];
        let deps = tool_deps(&calls);
        let order: Vec<usize> = (0..calls.len()).collect();
        assert!(
            respects_deps(&deps, &order),
            "emission order respects inferred deps: {deps:?}"
        );
    }

    #[test]
    fn scheduler_allows_independent_read_to_overlap_later_write() {
        // The capability the dep-aware scheduler unlocks: [read a.rs, write b.rs,
        // read c.rs] — none overlap on a path, so the scheduler may complete
        // read c.rs before write b.rs. The dep graph permits any order where
        // each call follows its (here, empty) deps. Pin that such an order
        // respects_deps, while an order that runs a dependent read before its
        // write does not.
        let calls = vec![
            ("r0".into(), "read".into(), r#"{"path":"a.rs"}"#.into()),
            (
                "w".into(),
                "write".into(),
                r#"{"path":"b.rs","content":"x"}"#.into(),
            ),
            ("r2".into(), "read".into(), r#"{"path":"c.rs"}"#.into()),
        ];
        let deps = tool_deps(&calls);
        // No path overlaps → no deps between them → any order respects deps,
        // including overlapping read c.rs ahead of write b.rs.
        assert!(
            deps.iter().all(|d| d.is_empty()),
            "independent batch has no deps: {deps:?}"
        );
        assert!(
            respects_deps(&deps, &[0, 2, 1]),
            "read c.rs may complete before write b.rs: {deps:?}"
        );

        // Contrast: a dependent read (same path as the write) must NOT complete
        // before the write.
        let dep_calls = vec![
            (
                "w".into(),
                "write".into(),
                r#"{"path":"a.rs","content":"x"}"#.into(),
            ),
            ("r".into(), "read".into(), r#"{"path":"a.rs"}"#.into()),
        ];
        let dep = tool_deps(&dep_calls);
        assert!(
            !respects_deps(&dep, &[1, 0]),
            "dependent read before write violates deps: {dep:?}"
        );
        assert!(
            respects_deps(&dep, &[0, 1]),
            "write before dependent read respects deps: {dep:?}"
        );
    }

    // ── parse_text_tool_calls ──

    fn tool_call_names(content: &[Content]) -> Vec<String> {
        content
            .iter()
            .filter_map(|c| match c {
                Content::ToolCall { name, .. } => Some(name.clone()),
                _ => None,
            })
            .collect()
    }

    /// Join all Text blocks in `content` into a single string, for assertions
    /// that check prose preservation.
    fn prose(content: &[Content]) -> String {
        content
            .iter()
            .filter_map(|c| match c {
                Content::Text(t) => Some(t.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    #[test]
    fn parse_text_tool_calls_finds_raw_json() {
        let text = r#"Let me check the files.

{"name": "list", "arguments": {}}

Now I'll read one."#;
        let content = parse_text_tool_calls(text, 0);
        assert_eq!(tool_call_names(&content), vec!["list"]);
        let p = prose(&content);
        assert!(!p.contains("name"), "JSON stripped from text: {p}");
        assert!(p.contains("Let me check"), "prose preserved: {p}");
    }

    #[test]
    fn parse_text_tool_calls_finds_json_with_object_arguments() {
        let text = r#"{"name": "bash", "arguments": {"command": "echo hi"}}"#;
        let content = parse_text_tool_calls(text, 0);
        assert_eq!(tool_call_names(&content), vec!["bash"]);
        assert!(prose(&content).is_empty(), "only JSON → no text blocks");
        if let Content::ToolCall { arguments, .. } = &content[0] {
            assert!(arguments.contains("echo hi"), "args preserved: {arguments}");
        }
    }

    #[test]
    fn parse_text_tool_calls_finds_json_with_string_arguments() {
        let text = r#"{"name": "read", "arguments": "{\"path\": \"foo.rs\"}"}"#;
        let content = parse_text_tool_calls(text, 0);
        assert_eq!(tool_call_names(&content), vec!["read"]);
        if let Content::ToolCall { arguments, .. } = &content[0] {
            assert!(arguments.contains("foo.rs"), "args preserved: {arguments}");
        }
    }

    #[test]
    fn parse_text_tool_calls_finds_xmlish_bash_call() {
        let text = "I'll run it.\n<tool_call>bash<arg_key>command</arg_key><arg_value>echo hi</arg_value></tool_call>\nDone.";
        let content = parse_text_tool_calls(text, 0);
        assert_eq!(tool_call_names(&content), vec!["bash"]);
        if let Content::ToolCall { arguments, .. } = &content[1] {
            assert_eq!(arguments, r#"{"command":"echo hi"}"#);
        } else {
            panic!("expected tool call: {content:?}");
        }
        let p = prose(&content);
        assert!(!p.contains("<tool_call>"), "XML tool call stripped: {p}");
        assert!(p.contains("I'll run it."), "leading prose preserved: {p}");
        assert!(p.contains("Done."), "trailing prose preserved: {p}");
    }

    #[test]
    fn parse_text_tool_calls_finds_xmlish_write_call_with_multiline_content() {
        let text = "<tool_call>write<arg_key>path</arg_key><arg_value>calc.py</arg_value><arg_key>content</arg_key><arg_value>line1\nline2</arg_value>";
        let content = parse_text_tool_calls(text, 0);
        assert_eq!(tool_call_names(&content), vec!["write"]);
        if let Content::ToolCall { arguments, .. } = &content[0] {
            let value: serde_json::Value = serde_json::from_str(arguments).unwrap();
            assert_eq!(value["path"], "calc.py");
            assert_eq!(value["content"], "line1\nline2");
        } else {
            panic!("expected tool call: {content:?}");
        }
    }

    #[test]
    fn parse_text_tool_calls_ignores_incomplete_xmlish_call() {
        let text = "<tool_call>bash<arg_key>command</arg_key><arg_value>echo";
        let content = parse_text_tool_calls(text, 0);
        assert!(tool_call_names(&content).is_empty());
        assert!(prose(&content).contains("<tool_call>"));
    }

    #[test]
    fn parse_text_tool_calls_finds_multiple_calls() {
        let text = r#"Starting now.
{"name": "read", "arguments": {"path": "a.rs"}}
{"name": "read", "arguments": {"path": "b.rs"}}
Done."#;
        let content = parse_text_tool_calls(text, 0);
        assert_eq!(tool_call_names(&content), vec!["read", "read"]);
        let p = prose(&content);
        assert!(p.contains("Starting now"), "prose before first call: {p}");
        assert!(p.contains("Done"), "prose after last call: {p}");
    }

    #[test]
    fn parse_text_tool_calls_ignores_non_tool_json() {
        // Random JSON that isn't a tool call should not be touched.
        let text = r#"The result is {"foo": 42} which is fine."#;
        let content = parse_text_tool_calls(text, 0);
        assert!(
            tool_call_names(&content).is_empty(),
            "no tool calls in random JSON"
        );
        assert_eq!(prose(&content), text, "text unchanged");
    }

    #[test]
    fn parse_text_tool_calls_ignores_unknown_tool_name() {
        let text = r#"{"name": "hack_the_planet", "arguments": {}}"#;
        let content = parse_text_tool_calls(text, 0);
        assert!(
            tool_call_names(&content).is_empty(),
            "unknown tool name rejected"
        );
    }

    #[test]
    fn parse_text_tool_calls_handles_nested_json_arguments() {
        let text =
            r#"{"name": "edit", "arguments": {"path": "a.rs", "old_string": "x\n{\"y\": 1}"}}"#;
        let content = parse_text_tool_calls(text, 0);
        assert_eq!(tool_call_names(&content), vec!["edit"]);
    }

    #[test]
    fn parse_text_tool_calls_preserves_leading_whitespace() {
        // A tool call at the very start leaves the trailing prose as the only
        // text block. Leading whitespace there is the model's actual content
        // (e.g. an indented code block) and must NOT be trimmed — only the
        // trailing newline artifact is removed.
        let text = "{\"name\": \"list\", \"arguments\": {}}\n    code line\n";
        let content = parse_text_tool_calls(text, 0);
        assert_eq!(tool_call_names(&content), vec!["list"]);
        // The trailing prose is a Text block after the ToolCall.
        let p = prose(&content);
        assert!(
            p.contains("    code line"),
            "leading indent preserved: {p:?}"
        );
        assert!(!p.ends_with('\n'), "trailing newline trimmed: {p:?}");
    }

    #[test]
    fn parse_text_tool_calls_interleaves_prose_and_calls() {
        // Trailing prose after a tool call must be a separate Text block
        // AFTER the ToolCall, not merged before it. This is the fix for the
        // "end of the context comes through" bug: previously all prose was
        // merged into one Text block placed before the tool calls, so the
        // model's forward-looking narration ("Now I'll read the file.")
        // appeared before the tool result in history, confusing the model
        // on the next prompt.
        let text =
            "Let me check.\n{\"name\": \"list\", \"arguments\": {}}\nNow I'll read the file.";
        let content = parse_text_tool_calls(text, 0);
        // Expected order: [Text("Let me check."), ToolCall(list), Text("Now I'll read the file.")]
        assert_eq!(content.len(), 3, "expected 3 blocks: {content:?}");
        assert!(
            matches!(content[0], Content::Text(_)),
            "first block is text"
        );
        assert!(
            matches!(content[1], Content::ToolCall { .. }),
            "second block is tool call"
        );
        assert!(
            matches!(content[2], Content::Text(_)),
            "third block is trailing text"
        );
        if let Content::Text(t) = &content[2] {
            assert!(
                t.contains("Now I'll read the file"),
                "trailing prose after call: {t}"
            );
        }
    }

    #[test]
    fn parse_text_tool_calls_no_calls_returns_text_only() {
        let text = "Just a normal message with no tool calls.";
        let content = parse_text_tool_calls(text, 0);
        assert!(tool_call_names(&content).is_empty());
        assert_eq!(prose(&content), text);
    }

    #[test]
    fn parse_text_tool_calls_strips_chatml_tokens() {
        // Models that emit <|im_start|> / <|im_end|> as raw text should have
        // them stripped, and any tool-call JSON between them should still be
        // promoted to a real ToolCall.
        let text = "Let me check.\n<|im_start|>\n{\"name\": \"list\", \"arguments\": {}}\n<|im_end|>\nDone.";
        let content = parse_text_tool_calls(text, 0);
        assert_eq!(tool_call_names(&content), vec!["list"]);
        let p = prose(&content);
        assert!(
            !p.contains("<|im_start|>") && !p.contains("<|im_end|>"),
            "special tokens stripped: {p}"
        );
        assert!(p.contains("Let me check"), "prose preserved: {p}");
        assert!(p.contains("Done"), "trailing prose preserved: {p}");
    }

    #[test]
    fn parse_text_tool_calls_ids_respect_offset() {
        // With offset 0, IDs start at textcall_0.
        let text = r#"{"name": "list", "arguments": {}}
{"name": "read", "arguments": {"path": "a.rs"}}"#;
        let content = parse_text_tool_calls(text, 0);
        let ids: Vec<&str> = content
            .iter()
            .filter_map(|c| match c {
                Content::ToolCall { id, .. } => Some(id.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(ids, vec!["textcall_0", "textcall_1"]);

        // With offset 3 (as if 3 textcall_ IDs already exist in history), IDs
        // start at textcall_3 — globally unique across the conversation.
        let content = parse_text_tool_calls(text, 3);
        let ids: Vec<&str> = content
            .iter()
            .filter_map(|c| match c {
                Content::ToolCall { id, .. } => Some(id.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(ids, vec!["textcall_3", "textcall_4"]);
    }

    #[test]
    fn textcall_id_offset_counts_existing_ids() {
        use crate::transcript::Transcript;
        use hi_ai::{Content, Message};

        // Empty transcript → offset 0.
        let t = Transcript::new(vec![]);
        assert_eq!(textcall_id_offset(&t), 0);

        // An assistant message with textcall_0 and textcall_2 → offset 3.
        let t = Transcript::new(vec![
            Message::system("sys"),
            Message::user("hi"),
            Message::assistant(vec![
                Content::Text("thinking".into()),
                Content::ToolCall {
                    id: "textcall_0".into(),
                    name: "list".into(),
                    arguments: "{}".into(),
                },
                Content::ToolCall {
                    id: "textcall_2".into(),
                    name: "read".into(),
                    arguments: "{}".into(),
                },
            ]),
            Message::tool_result("textcall_0", "ok"),
            Message::tool_result("textcall_2", "ok"),
        ]);
        assert_eq!(textcall_id_offset(&t), 3);

        // Non-textcall IDs (e.g. from a hosted provider) are ignored.
        let t = Transcript::new(vec![Message::assistant(vec![Content::ToolCall {
            id: "call_abc".into(),
            name: "list".into(),
            arguments: "{}".into(),
        }])]);
        assert_eq!(textcall_id_offset(&t), 0);
    }
}
