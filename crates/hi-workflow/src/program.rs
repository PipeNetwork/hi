//! Restricted Rhai programs used by the agent's native `run_program` tool.
//!
//! This is deliberately a sibling of the workflow engine rather than a
//! workflow feature.  Programs are short, per-turn computations: the only
//! effect they can request is a call back to the host through `tool()` (or a
//! batch through `parallel()`).  The host remains responsible for policy,
//! confirmation, cancellation, and the actual tool implementation.

use std::cell::RefCell;
use std::rc::Rc;
use std::time::Duration;

use rhai::{Dynamic, EvalAltResult, ImmutableString, Position};
use tokio::sync::{mpsc, oneshot};
use tokio_util::sync::CancellationToken;

/// Maximum UTF-8 bytes accepted for one program source.
pub const MAX_PROGRAM_SOURCE_BYTES: usize = 256 * 1024;
const HOST_REPLY_POLL_INTERVAL: Duration = Duration::from_millis(5);

/// A tool invocation emitted by a program, in program order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProgramCall {
    pub occurrence: usize,
    pub name: String,
    pub arguments: serde_json::Value,
}

/// Conservatively extract directly resolvable `tool("name", #{...})` calls.
/// This is used only for shadow execution. Any control flow, dynamic value,
/// nested expression, or non-literal map causes the caller to skip the
/// candidate rather than guess.
pub fn extract_safe_literal_calls(source: &str) -> Vec<ProgramCall> {
    let lower = source.to_ascii_lowercase();
    if ["if ", "if(", "for ", "while ", "loop", "switch ", "fn "]
        .iter()
        .any(|token| lower.contains(token))
    {
        return Vec::new();
    }
    let mut calls = Vec::new();
    let mut cursor = 0;
    while let Some(start) = next_tool_call(source, cursor) {
        let before_ok = start == 0 || !source.as_bytes()[start - 1].is_ascii_alphanumeric();
        let after = start + 4;
        let after_ok = source[after..].trim_start().starts_with('(');
        if !before_ok || !after_ok {
            cursor = after;
            continue;
        }
        let open = after + source[after..].find('(').unwrap_or(0) + 1;
        let Some((name, after_name)) = quoted_literal(source, open) else {
            cursor = open;
            continue;
        };
        let Some(comma) = source[after_name..]
            .find(',')
            .map(|index| after_name + index)
        else {
            break;
        };
        let Some(map_start) = source[comma + 1..]
            .find("#{")
            .map(|index| comma + 1 + index)
        else {
            cursor = comma + 1;
            continue;
        };
        let Some(map_end) = balanced_map_end(source, map_start) else {
            break;
        };
        let Some(arguments) = literal_map_to_json(&source[map_start..=map_end]) else {
            cursor = map_end + 1;
            continue;
        };
        calls.push(ProgramCall {
            occurrence: calls.len(),
            name,
            arguments,
        });
        cursor = map_end + 1;
    }
    calls
}

fn next_tool_call(source: &str, start: usize) -> Option<usize> {
    let bytes = source.as_bytes();
    let mut index = start;
    let mut quoted = false;
    while index < bytes.len() {
        if quoted {
            match bytes[index] {
                b'\\' => index = index.saturating_add(2),
                b'"' => {
                    quoted = false;
                    index += 1;
                }
                _ => index += 1,
            }
            continue;
        }
        if bytes[index] == b'"' {
            quoted = true;
            index += 1;
            continue;
        }
        if bytes.get(index..index.saturating_add(2)) == Some(b"//") {
            index = source[index..]
                .find('\n')
                .map(|offset| index + offset + 1)
                .unwrap_or(bytes.len());
            continue;
        }
        if bytes.get(index..index.saturating_add(2)) == Some(b"/*") {
            index = source[index + 2..]
                .find("*/")
                .map(|offset| index + 2 + offset + 2)
                .unwrap_or(bytes.len());
            continue;
        }
        if bytes.get(index..index.saturating_add(4)) == Some(b"tool") {
            let before_ok = index == 0 || !bytes[index - 1].is_ascii_alphanumeric();
            let after = index + 4;
            let after_ok = source[after..].trim_start().starts_with('(');
            if before_ok && after_ok {
                return Some(index);
            }
        }
        index += 1;
    }
    None
}

/// Recover the currently available value of the JSON `source` argument from
/// a streamed native tool-call argument buffer. A missing closing quote is
/// accepted because the returned text is only a shadow-execution candidate;
/// malformed escapes simply produce no candidate.
pub fn extract_partial_program_source(arguments: &str) -> Option<String> {
    let marker = "\"source\"";
    let key = arguments.find(marker)?;
    let after_key = key + marker.len();
    let colon = after_key + arguments[after_key..].find(':')?;
    let (offset, _) = arguments[colon + 1..]
        .char_indices()
        .find(|(_, character)| !character.is_ascii_whitespace())?;
    let index = colon + 1 + offset;
    if arguments[index..].chars().next()? != '"' {
        return None;
    }
    let mut escaped = false;
    let mut end = arguments.len();
    for (offset, character) in arguments[index + 1..].char_indices() {
        if !escaped && character == '"' {
            end = index + 1 + offset;
            break;
        }
        if escaped {
            escaped = false;
        } else if character == '\\' {
            escaped = true;
        }
    }
    if escaped || end == index + 1 {
        return None;
    }
    let raw = &arguments[index + 1..end];
    // Closing an incomplete JSON string is safe for candidate extraction; the
    // JSON parser still rejects incomplete unicode escapes and control bytes.
    serde_json::from_str::<String>(&format!("\"{raw}\"")).ok()
}

fn quoted_literal(source: &str, start: usize) -> Option<(String, usize)> {
    let bytes = source.as_bytes();
    let mut index = start;
    while bytes.get(index).is_some_and(u8::is_ascii_whitespace) {
        index += 1;
    }
    if bytes.get(index) != Some(&b'"') {
        return None;
    }
    let mut escaped = false;
    for end in index + 1..source.len() {
        let byte = bytes[end];
        if byte == b'"' && !escaped {
            let value = serde_json::from_str::<String>(&source[index..=end]).ok()?;
            return Some((value, end + 1));
        }
        escaped = byte == b'\\' && !escaped;
        if byte != b'\\' {
            escaped = false;
        }
    }
    None
}

fn balanced_map_end(source: &str, start: usize) -> Option<usize> {
    let bytes = source.as_bytes();
    let mut depth = 0usize;
    let mut quoted = false;
    let mut escaped = false;
    for (index, byte) in bytes.iter().enumerate().skip(start) {
        if quoted {
            if *byte == b'"' && !escaped {
                quoted = false;
            }
            escaped = *byte == b'\\' && !escaped;
            if *byte != b'\\' {
                escaped = false;
            }
            continue;
        }
        match *byte {
            b'"' => quoted = true,
            b'{' => depth += 1,
            b'}' => {
                depth = depth.saturating_sub(1);
                if depth == 0 {
                    return Some(index);
                }
            }
            _ => {}
        }
    }
    None
}

fn literal_map_to_json(map: &str) -> Option<serde_json::Value> {
    let inner = map.strip_prefix("#{")?.strip_suffix('}')?;
    let mut json = String::from("{");
    for (index, part) in split_top_level(inner, ',').into_iter().enumerate() {
        let (key, value) = part.split_once(':')?;
        if index > 0 {
            json.push(',');
        }
        let key = key.trim();
        if key.starts_with('"') {
            json.push_str(key);
        } else {
            json.push('"');
            json.push_str(key);
            json.push('"');
        }
        json.push(':');
        let value = value.trim();
        if value.starts_with("#{") {
            json.push_str(&serde_json::to_string(&literal_map_to_json(value)?).ok()?);
        } else if value.starts_with('"') {
            let _: String = serde_json::from_str(value).ok()?;
            json.push_str(value);
        } else if matches!(value, "true" | "false" | "null") || value.parse::<f64>().is_ok() {
            json.push_str(value);
        } else {
            return None;
        }
    }
    json.push('}');
    serde_json::from_str(&json).ok()
}

fn runtime_error(message: impl Into<String>) -> Box<EvalAltResult> {
    Box::new(EvalAltResult::ErrorRuntime(
        Dynamic::from(message.into()),
        Position::NONE,
    ))
}

fn split_top_level(input: &str, separator: char) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut start = 0;
    let mut depth: usize = 0;
    let mut quoted = false;
    let mut escaped = false;
    for (index, byte) in input.bytes().enumerate() {
        if quoted {
            if byte == b'"' && !escaped {
                quoted = false;
            }
            escaped = byte == b'\\' && !escaped;
            if byte != b'\\' {
                escaped = false;
            }
            continue;
        }
        match byte {
            b'"' => quoted = true,
            b'{' | b'[' => depth += 1,
            b'}' | b']' => depth = depth.saturating_sub(1),
            value if value == separator as u8 && depth == 0 => {
                parts.push(&input[start..index]);
                start = index + 1;
            }
            _ => {}
        }
    }
    if !input[start..].trim().is_empty() {
        parts.push(&input[start..]);
    }
    parts
}

/// A bounded result returned by the host to a program and included in the
/// aggregate provider result.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ProgramToolResult {
    pub index: usize,
    pub name: String,
    pub status: String,
    pub output: String,
}

/// Requests sent synchronously from the Rhai thread to the async agent host.
pub enum ProgramHostRequest {
    ExecuteTool {
        call: ProgramCall,
        reply: oneshot::Sender<Result<ProgramToolResult, String>>,
    },
    ParallelTools {
        calls: Vec<ProgramCall>,
        reply: oneshot::Sender<Result<Vec<ProgramToolResult>, String>>,
    },
}

/// Inputs for one restricted program evaluation.
pub struct ProgramRunParams {
    pub source: String,
    pub host_tx: mpsc::UnboundedSender<ProgramHostRequest>,
    pub cancel: CancellationToken,
    /// Explicit Rhai operation cap. Zero means unlimited; turn cancellation
    /// remains active through the engine progress callback.
    pub max_ops: u64,
    /// Explicit host-supplied cap, normally the remaining per-turn tool
    /// budget. `None` means the parent turn is unlimited.
    pub max_calls: Option<usize>,
}

impl ProgramRunParams {
    /// Ordinary turn programs inherit the unlimited parent execution model.
    pub const DEFAULT_MAX_OPS: u64 = 0;
}

/// The result of evaluating a program. Calls are retained even on failure so
/// the host can report exactly what ran without manufacturing transcript
/// messages for nested calls.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ProgramOutcome {
    Succeeded {
        result: serde_json::Value,
        calls: Vec<ProgramToolResult>,
    },
    Failed {
        error: String,
        calls: Vec<ProgramToolResult>,
    },
    Cancelled {
        calls: Vec<ProgramToolResult>,
    },
}

struct ProgramContext {
    host_tx: mpsc::UnboundedSender<ProgramHostRequest>,
    cancel: CancellationToken,
    next_occurrence: usize,
    max_calls: Option<usize>,
    calls: Vec<ProgramToolResult>,
}

impl ProgramContext {
    /// Reserve a contiguous occurrence range without partially advancing the
    /// counter. This keeps an unlimited, very long-lived program from wrapping
    /// identities and makes parallel reservation all-or-nothing.
    fn reserve_occurrences(&mut self, count: usize) -> ScriptResult<std::ops::Range<usize>> {
        let start = self.next_occurrence;
        let end = start
            .checked_add(count)
            .ok_or_else(|| runtime_error("program tool-call count overflowed"))?;
        if let Some(max_calls) = self.max_calls
            && end > max_calls
        {
            return Err(runtime_error(format!(
                "program exceeded its remaining budget of {} tool calls",
                max_calls
            )));
        }
        self.next_occurrence = end;
        Ok(start..end)
    }
}

type ScriptResult<T> = Result<T, Box<EvalAltResult>>;

/// Evaluate a program on the calling thread.
pub fn run_program(params: ProgramRunParams) -> ProgramOutcome {
    let ProgramRunParams {
        source,
        host_tx,
        cancel,
        max_ops,
        max_calls,
    } = params;

    if source.len() > MAX_PROGRAM_SOURCE_BYTES {
        return ProgramOutcome::Failed {
            error: format!("program source exceeds the {MAX_PROGRAM_SOURCE_BYTES}-byte limit"),
            calls: Vec::new(),
        };
    }

    let context = Rc::new(RefCell::new(ProgramContext {
        host_tx,
        cancel: cancel.clone(),
        next_occurrence: 0,
        max_calls,
        calls: Vec::new(),
    }));
    let mut engine = rhai::Engine::new();
    engine.set_max_operations(max_ops);
    engine.set_max_call_levels(64);
    engine.set_max_expr_depths(128, 64);
    engine.set_max_string_size(16 * 1024 * 1024);
    engine.set_max_array_size(65_536);
    engine.set_max_map_size(65_536);
    engine.set_module_resolver(rhai::module_resolvers::DummyModuleResolver::new());
    engine.disable_symbol("eval");
    // Programs run inside the harness, including speculative prefix
    // analysis. Keep Rhai's diagnostic hooks from writing directly to the
    // user's terminal or polluting provider-visible output.
    engine.on_print(|_| {});
    engine.on_debug(|_, _, _| {});
    engine.register_fn("timestamp", unavailable("timestamp()"));
    engine.register_fn("sleep", |_seconds: i64| -> ScriptResult<()> {
        Err(runtime_error("sleep() is unavailable in run_program"))
    });
    engine.register_fn("sleep", |_seconds: f64| -> ScriptResult<()> {
        Err(runtime_error("sleep() is unavailable in run_program"))
    });
    engine.register_fn("exit", unavailable("exit()"));

    let progress_cancel = cancel.clone();
    engine.on_progress(move |_ops| {
        if progress_cancel.is_cancelled() {
            Some(Dynamic::from("program cancelled"))
        } else {
            None
        }
    });

    register_program_functions(&mut engine, &context);
    let ast = match engine.compile(&source) {
        Ok(ast) => ast,
        Err(error) => {
            return ProgramOutcome::Failed {
                error: format!(
                    "program failed to compile: {}",
                    crate::with_rhai_hint(error.to_string())
                ),
                calls: Vec::new(),
            };
        }
    };
    if cancel.is_cancelled() {
        return ProgramOutcome::Cancelled { calls: Vec::new() };
    }

    let result = engine.eval_ast::<Dynamic>(&ast);
    let calls = context.borrow().calls.clone();
    match result {
        Ok(value) => ProgramOutcome::Succeeded {
            result: dynamic_to_value(value),
            calls,
        },
        Err(error) if cancel.is_cancelled() || error.to_string().contains("program cancelled") => {
            ProgramOutcome::Cancelled { calls }
        }
        Err(error) => ProgramOutcome::Failed {
            error: crate::with_rhai_hint(error.to_string()),
            calls,
        },
    }
}

fn unavailable(name: &'static str) -> impl Fn() -> ScriptResult<()> {
    move || {
        Err(runtime_error(format!(
            "{name} is unavailable in run_program"
        )))
    }
}

fn register_program_functions(engine: &mut rhai::Engine, context: &Rc<RefCell<ProgramContext>>) {
    let tool_context = context.clone();
    engine.register_fn(
        "tool",
        move |name: ImmutableString, args: rhai::Map| -> ScriptResult<Dynamic> {
            let arguments = map_to_value(args)?;
            let result = host_tool(&tool_context, name.to_string(), arguments)?;
            value_to_dynamic(&serde_json::json!({
                "status": result.status,
                "output": result.output,
                "name": result.name,
                "index": result.index,
            }))
        },
    );

    let parallel_context = context.clone();
    engine.register_fn(
        "parallel",
        move |items: rhai::Array| -> ScriptResult<rhai::Array> {
            let calls = items
                .into_iter()
                .map(parse_parallel_call)
                .collect::<ScriptResult<Vec<_>>>()?;
            if calls.is_empty() {
                return Ok(rhai::Array::new());
            }
            let results = host_parallel(&parallel_context, calls)?;
            results
                .into_iter()
                .map(|result| {
                    value_to_dynamic(&serde_json::json!({
                        "status": result.status,
                        "output": result.output,
                        "name": result.name,
                        "index": result.index,
                    }))
                })
                .collect()
        },
    );
}

fn host_tool(
    context: &Rc<RefCell<ProgramContext>>,
    name: String,
    arguments: serde_json::Value,
) -> ScriptResult<ProgramToolResult> {
    let call = reserve_call(context, name, arguments)?;
    let occurrence = call.occurrence;
    let (reply_tx, reply_rx) = oneshot::channel();
    context
        .borrow()
        .host_tx
        .send(ProgramHostRequest::ExecuteTool {
            call,
            reply: reply_tx,
        })
        .map_err(|_| runtime_error("program host channel closed"))?;
    let result = wait_for_reply(context, reply_rx)?;
    if let Ok(result) = &result {
        if result.index != occurrence {
            return Err(runtime_error("program host returned an invalid call index"));
        }
        context.borrow_mut().calls.push(result.clone());
    }
    result.map_err(runtime_error)
}

fn host_parallel(
    context: &Rc<RefCell<ProgramContext>>,
    mut calls: Vec<ProgramCall>,
) -> ScriptResult<Vec<ProgramToolResult>> {
    {
        let mut context = context.borrow_mut();
        let occurrences = context.reserve_occurrences(calls.len())?;
        for (call, occurrence) in calls.iter_mut().zip(occurrences) {
            call.occurrence = occurrence;
        }
    }
    let expected_occurrences = calls.iter().map(|call| call.occurrence).collect::<Vec<_>>();
    let (reply_tx, reply_rx) = oneshot::channel();
    context
        .borrow()
        .host_tx
        .send(ProgramHostRequest::ParallelTools {
            calls,
            reply: reply_tx,
        })
        .map_err(|_| runtime_error("program host channel closed"))?;
    let result = wait_for_reply(context, reply_rx)?;
    if let Ok(results) = &result {
        if results.len() != expected_occurrences.len()
            || results
                .iter()
                .zip(expected_occurrences)
                .any(|(result, expected)| result.index != expected)
        {
            return Err(runtime_error(
                "program host returned parallel results in an invalid order",
            ));
        }
        context.borrow_mut().calls.extend(results.iter().cloned());
    }
    result.map_err(runtime_error)
}

fn reserve_call(
    context: &Rc<RefCell<ProgramContext>>,
    name: String,
    arguments: serde_json::Value,
) -> ScriptResult<ProgramCall> {
    if name.trim().is_empty() {
        return Err(runtime_error("tool name cannot be empty"));
    }
    let mut context = context.borrow_mut();
    let occurrence = context.reserve_occurrences(1)?.start;
    Ok(ProgramCall {
        occurrence,
        name,
        arguments,
    })
}

fn parse_parallel_call(value: Dynamic) -> ScriptResult<ProgramCall> {
    let map = value
        .try_cast::<rhai::Map>()
        .ok_or_else(|| runtime_error("parallel() expects maps with name and args"))?;
    let name = map
        .get("name")
        .and_then(|value| value.clone().try_cast::<ImmutableString>())
        .ok_or_else(|| runtime_error("parallel() call is missing string field `name`"))?
        .to_string();
    let args = map
        .get("args")
        .cloned()
        .ok_or_else(|| runtime_error("parallel() call is missing map field `args`"))?;
    let arguments = map_to_value(
        args.try_cast::<rhai::Map>()
            .ok_or_else(|| runtime_error("parallel() field `args` must be a map"))?,
    )?;
    // `parallel` assigns the final occurrence identities in the host-facing
    // order. The context counter is reserved here so concurrent calls cannot
    // accidentally share an identity.
    Ok(ProgramCall {
        occurrence: 0,
        name,
        arguments,
    })
}

fn wait_for_reply<T>(
    context: &Rc<RefCell<ProgramContext>>,
    mut receiver: oneshot::Receiver<T>,
) -> ScriptResult<T> {
    let cancel = context.borrow().cancel.clone();
    loop {
        if cancel.is_cancelled() {
            return Err(runtime_error("program cancelled"));
        }
        match receiver.try_recv() {
            Ok(value) => return Ok(value),
            Err(oneshot::error::TryRecvError::Closed) => {
                return Err(runtime_error("program host dropped reply"));
            }
            Err(oneshot::error::TryRecvError::Empty) => {}
        }
        std::thread::sleep(HOST_REPLY_POLL_INTERVAL);
    }
}

fn dynamic_to_value(value: Dynamic) -> serde_json::Value {
    rhai::serde::from_dynamic::<serde_json::Value>(&value).unwrap_or(serde_json::Value::Null)
}

fn value_to_dynamic(value: &serde_json::Value) -> ScriptResult<Dynamic> {
    rhai::serde::to_dynamic(value)
        .map_err(|error| runtime_error(format!("program result conversion failed: {error}")))
}

fn map_to_value(map: rhai::Map) -> ScriptResult<serde_json::Value> {
    rhai::serde::from_dynamic::<serde_json::Value>(&Dynamic::from_map(map))
        .map_err(|error| runtime_error(format!("program arguments must be JSON-like: {error}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_only_literal_direct_calls() {
        let calls = extract_safe_literal_calls(
            r#"let one = tool("read", #{path: "src/lib.rs"}); let two = tool("grep", #{pattern: "TODO", glob: "*.rs"}); #{one: one, two: two}"#,
        );
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "read");
        assert_eq!(calls[0].arguments["path"], "src/lib.rs");
        assert_eq!(calls[1].name, "grep");
        assert_eq!(calls[1].arguments["glob"], "*.rs");
    }

    #[test]
    fn uncertain_control_flow_is_not_shadowed() {
        assert!(
            extract_safe_literal_calls(r#"if ready { tool("read", #{path: "src/lib.rs"}); }"#)
                .is_empty()
        );
        assert!(extract_safe_literal_calls(
            r#"let text = "tool(\"read\", #{path: \"secret\"})"; // tool("read", #{path: "comment"})"#
        )
        .is_empty());
    }

    #[test]
    fn partial_source_preserves_unicode_and_accepts_open_string() {
        assert_eq!(
            extract_partial_program_source(r#"{"source":"let x = \"café\"; x"}"#),
            Some(r#"let x = "café"; x"#.into())
        );
        assert_eq!(
            extract_partial_program_source(r#"{"source":"let x = \"café\"; x"#),
            Some(r#"let x = "café"; x"#.into())
        );
        assert!(extract_partial_program_source(r#"{"source":"let x = \u12"#).is_none());
    }

    #[test]
    fn program_routes_calls_and_returns_one_result() {
        let (host_tx, mut host_rx) = mpsc::unbounded_channel();
        let worker = std::thread::spawn(move || {
            run_program(ProgramRunParams {
                source: r#"let value = tool("read", #{path: "src/lib.rs"}); #{value: value}"#
                    .into(),
                host_tx,
                cancel: CancellationToken::new(),
                max_ops: ProgramRunParams::DEFAULT_MAX_OPS,
                max_calls: None,
            })
        });

        let request = host_rx.blocking_recv().expect("program sent one call");
        match request {
            ProgramHostRequest::ExecuteTool { call, reply } => {
                assert_eq!(call.occurrence, 0);
                assert_eq!(call.name, "read");
                assert_eq!(call.arguments["path"], "src/lib.rs");
                reply
                    .send(Ok(ProgramToolResult {
                        index: 0,
                        name: "read".into(),
                        status: "succeeded".into(),
                        output: "file contents".into(),
                    }))
                    .expect("program is waiting for the reply");
            }
            ProgramHostRequest::ParallelTools { .. } => panic!("expected a direct call"),
        }
        drop(host_rx);
        let outcome = worker.join().expect("program worker did not panic");
        assert!(matches!(
            outcome,
            ProgramOutcome::Succeeded { ref calls, .. } if calls.len() == 1
        ));
    }

    #[test]
    fn parallel_routes_ordered_calls_as_one_host_request() {
        let (host_tx, mut host_rx) = mpsc::unbounded_channel();
        let worker = std::thread::spawn(move || {
            run_program(ProgramRunParams {
                source: r#"parallel([
                    #{name: "read", args: #{path: "a.txt"}},
                    #{name: "grep", args: #{pattern: "TODO"}}
                ])"#
                .into(),
                host_tx,
                cancel: CancellationToken::new(),
                max_ops: ProgramRunParams::DEFAULT_MAX_OPS,
                max_calls: None,
            })
        });

        let request = host_rx
            .blocking_recv()
            .expect("program sent one parallel batch");
        match request {
            ProgramHostRequest::ParallelTools { calls, reply } => {
                assert_eq!(
                    calls
                        .iter()
                        .map(|call| (call.occurrence, call.name.as_str()))
                        .collect::<Vec<_>>(),
                    vec![(0, "read"), (1, "grep")]
                );
                reply
                    .send(Ok(vec![
                        ProgramToolResult {
                            index: 0,
                            name: "read".into(),
                            status: "succeeded".into(),
                            output: "a".into(),
                        },
                        ProgramToolResult {
                            index: 1,
                            name: "grep".into(),
                            status: "succeeded".into(),
                            output: "b".into(),
                        },
                    ]))
                    .expect("program is waiting for the parallel reply");
            }
            ProgramHostRequest::ExecuteTool { .. } => panic!("expected a parallel batch"),
        }
        drop(host_rx);
        let outcome = worker.join().expect("program worker did not panic");
        assert!(matches!(
            outcome,
            ProgramOutcome::Succeeded { ref calls, .. } if calls.len() == 2
        ));
    }

    #[test]
    fn unlimited_program_routes_more_than_the_legacy_48_call_ceiling() {
        const CALLS: usize = 50;
        let source = (0..CALLS)
            .map(|index| format!(r#"tool("read", #{{path: "file-{index}.rs"}});"#))
            .collect::<Vec<_>>()
            .join("\n");
        let (host_tx, mut host_rx) = mpsc::unbounded_channel();
        let worker = std::thread::spawn(move || {
            run_program(ProgramRunParams {
                source,
                host_tx,
                cancel: CancellationToken::new(),
                max_ops: ProgramRunParams::DEFAULT_MAX_OPS,
                max_calls: None,
            })
        });

        for index in 0..CALLS {
            let request = host_rx
                .blocking_recv()
                .expect("unlimited program dispatched every nested call");
            let ProgramHostRequest::ExecuteTool { call, reply } = request else {
                panic!("expected a direct call")
            };
            assert_eq!(call.occurrence, index);
            reply
                .send(Ok(ProgramToolResult {
                    index,
                    name: call.name,
                    status: "succeeded".into(),
                    output: "ok".into(),
                }))
                .expect("program is waiting for the reply");
        }

        drop(host_rx);
        let outcome = worker.join().expect("program worker did not panic");
        assert!(
            matches!(&outcome, ProgramOutcome::Succeeded { calls, .. } if calls.len() == CALLS),
            "unlimited parent budget must survive program execution: {outcome:?}"
        );
    }

    #[test]
    fn cancellation_still_interrupts_a_program_waiting_for_a_host_reply() {
        let (host_tx, mut host_rx) = mpsc::unbounded_channel();
        let cancel = CancellationToken::new();
        let worker_cancel = cancel.clone();
        let worker = std::thread::spawn(move || {
            run_program(ProgramRunParams {
                source: r#"tool("read", #{path: "slow.rs"})"#.into(),
                host_tx,
                cancel: worker_cancel,
                max_ops: ProgramRunParams::DEFAULT_MAX_OPS,
                max_calls: None,
            })
        });

        let request = host_rx
            .blocking_recv()
            .expect("program sent a host request");
        let ProgramHostRequest::ExecuteTool { reply, .. } = request else {
            panic!("expected a direct call")
        };
        cancel.cancel();

        let outcome = worker.join().expect("program worker did not panic");
        drop(reply);
        assert!(
            matches!(&outcome, ProgramOutcome::Cancelled { .. }),
            "cancellation replaces the removed reply deadline: {outcome:?}"
        );
    }

    #[test]
    fn default_operations_are_unlimited_but_an_explicit_cap_still_applies() {
        assert_eq!(ProgramRunParams::DEFAULT_MAX_OPS, 0);

        let (host_tx, _host_rx) = mpsc::unbounded_channel();
        let outcome = run_program(ProgramRunParams {
            source: "let total = 0; for i in 0..1000 { total += i; } total".into(),
            host_tx,
            cancel: CancellationToken::new(),
            max_ops: 10,
            max_calls: None,
        });
        assert!(
            matches!(&outcome, ProgramOutcome::Failed { error, .. } if error.to_ascii_lowercase().contains("too many operations")),
            "an explicit operation cap must remain effective: {outcome:?}"
        );
    }

    #[test]
    fn forbidden_runtime_and_source_limit_fail_closed() {
        let (host_tx, _host_rx) = mpsc::unbounded_channel();
        let outcome = run_program(ProgramRunParams {
            source: "timestamp()".into(),
            host_tx: host_tx.clone(),
            cancel: CancellationToken::new(),
            max_ops: ProgramRunParams::DEFAULT_MAX_OPS,
            max_calls: None,
        });
        assert!(
            matches!(outcome, ProgramOutcome::Failed { error, .. } if error.contains("unavailable"))
        );
        let outcome = run_program(ProgramRunParams {
            source: "x".repeat(MAX_PROGRAM_SOURCE_BYTES + 1),
            host_tx,
            cancel: CancellationToken::new(),
            max_ops: ProgramRunParams::DEFAULT_MAX_OPS,
            max_calls: None,
        });
        assert!(
            matches!(outcome, ProgramOutcome::Failed { error, .. } if error.contains("source exceeds"))
        );
    }

    #[test]
    fn program_cannot_exceed_the_host_supplied_call_budget() {
        let (host_tx, mut host_rx) = mpsc::unbounded_channel();
        let outcome = run_program(ProgramRunParams {
            source: r#"tool("read", #{path: "src/lib.rs"})"#.into(),
            host_tx,
            cancel: CancellationToken::new(),
            max_ops: ProgramRunParams::DEFAULT_MAX_OPS,
            max_calls: Some(0),
        });

        assert!(
            matches!(&outcome, ProgramOutcome::Failed { error, .. } if error.contains("remaining budget of 0")),
            "program should fail before dispatching a nested tool: {outcome:?}"
        );
        assert!(
            host_rx.try_recv().is_err(),
            "a budget-rejected nested call reached the host"
        );
    }

    #[test]
    fn occurrence_reservation_fails_atomically_before_usize_wraparound() {
        let (host_tx, _host_rx) = mpsc::unbounded_channel();
        let mut context = ProgramContext {
            host_tx,
            cancel: CancellationToken::new(),
            next_occurrence: usize::MAX - 1,
            max_calls: None,
            calls: Vec::new(),
        };

        let error = context
            .reserve_occurrences(2)
            .expect_err("occurrence identities must never wrap");
        assert!(error.to_string().contains("tool-call count overflowed"));
        assert_eq!(context.next_occurrence, usize::MAX - 1);
    }
}
