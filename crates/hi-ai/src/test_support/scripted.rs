//! Route-aware, deterministic OpenAI-compatible server for integration tests.
//!
//! Unlike [`super::FakeOpenAiServer`], this server is intended to be shared by
//! integration-test crates. Model discovery is repeatable and independent of
//! the strictly ordered chat script, responses can be synchronized with named
//! gates, and every request/failure is inspectable after a run.

use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::fmt;
use std::io::{self, Read, Write};
use std::net::{Shutdown, TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

const REQUEST_LIMIT_BYTES: usize = 8 * 1024 * 1024;
const ACCEPT_POLL_INTERVAL: Duration = Duration::from_millis(10);
const SOCKET_TIMEOUT: Duration = Duration::from_secs(2);
const REQUEST_READ_DEADLINE: Duration = Duration::from_secs(2);
const REQUEST_READ_POLL_INTERVAL: Duration = Duration::from_millis(50);

/// One HTTP request observed by [`ScriptedOpenAiServer`].
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct RecordedRequest {
    pub sequence: usize,
    pub method: String,
    pub path: String,
    /// Header names are normalized to lowercase; values retain their case.
    pub headers: BTreeMap<String, String>,
    pub body: String,
    pub json: Option<Value>,
}

impl RecordedRequest {
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .get(&name.to_ascii_lowercase())
            .map(String::as_str)
    }
}

/// A declarative JSON-body condition used by [`RequestMatcher`].
#[derive(Clone, Debug)]
pub enum JsonExpectation {
    Equals { pointer: String, value: Value },
    Present { pointer: String },
    Absent { pointer: String },
    StringContains { pointer: String, needle: String },
}

/// Declarative request expectations. An empty matcher accepts every request.
#[derive(Clone, Debug, Default)]
pub struct RequestMatcher {
    method: Option<String>,
    path: Option<String>,
    path_suffix: Option<String>,
    headers: BTreeMap<String, String>,
    body_equals: Option<String>,
    body_contains: Vec<String>,
    body_excludes: Vec<String>,
    json: Vec<JsonExpectation>,
}

impl RequestMatcher {
    pub fn any() -> Self {
        Self::default()
    }

    pub fn method(mut self, method: impl Into<String>) -> Self {
        self.method = Some(method.into().to_ascii_uppercase());
        self
    }

    pub fn path(mut self, path: impl Into<String>) -> Self {
        self.path = Some(path.into());
        self
    }

    pub fn path_suffix(mut self, suffix: impl Into<String>) -> Self {
        self.path_suffix = Some(suffix.into());
        self
    }

    pub fn header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers
            .insert(name.into().to_ascii_lowercase(), value.into());
        self
    }

    pub fn body_equals(mut self, body: impl Into<String>) -> Self {
        self.body_equals = Some(body.into());
        self
    }

    pub fn body_contains(mut self, needle: impl Into<String>) -> Self {
        self.body_contains.push(needle.into());
        self
    }

    pub fn body_excludes(mut self, needle: impl Into<String>) -> Self {
        self.body_excludes.push(needle.into());
        self
    }

    pub fn json_eq(mut self, pointer: impl Into<String>, value: Value) -> Self {
        self.json.push(JsonExpectation::Equals {
            pointer: pointer.into(),
            value,
        });
        self
    }

    pub fn json_present(mut self, pointer: impl Into<String>) -> Self {
        self.json.push(JsonExpectation::Present {
            pointer: pointer.into(),
        });
        self
    }

    pub fn json_absent(mut self, pointer: impl Into<String>) -> Self {
        self.json.push(JsonExpectation::Absent {
            pointer: pointer.into(),
        });
        self
    }

    pub fn json_string_contains(
        mut self,
        pointer: impl Into<String>,
        needle: impl Into<String>,
    ) -> Self {
        self.json.push(JsonExpectation::StringContains {
            pointer: pointer.into(),
            needle: needle.into(),
        });
        self
    }

    /// Returns every mismatch so a failed scenario has useful diagnostics.
    pub fn mismatches(&self, request: &RecordedRequest) -> Vec<String> {
        let mut mismatches = Vec::new();
        if let Some(expected) = &self.method
            && &request.method != expected
        {
            mismatches.push(format!(
                "method: expected {expected:?}, got {:?}",
                request.method
            ));
        }
        if let Some(expected) = &self.path
            && &request.path != expected
        {
            mismatches.push(format!(
                "path: expected {expected:?}, got {:?}",
                request.path
            ));
        }
        if let Some(expected) = &self.path_suffix
            && !request.path.ends_with(expected)
        {
            mismatches.push(format!(
                "path: expected suffix {expected:?}, got {:?}",
                request.path
            ));
        }
        for (name, expected) in &self.headers {
            match request.header(name) {
                Some(actual) if actual == expected => {}
                Some(actual) => mismatches.push(format!(
                    "header {name:?}: expected {expected:?}, got {actual:?}"
                )),
                None => mismatches.push(format!(
                    "header {name:?}: expected {expected:?}, but it was absent"
                )),
            }
        }
        if let Some(expected) = &self.body_equals
            && &request.body != expected
        {
            mismatches.push(format!(
                "body: expected exact body {expected:?}, got {:?}",
                request.body
            ));
        }
        for needle in &self.body_contains {
            if !request.body.contains(needle) {
                mismatches.push(format!("body: missing substring {needle:?}"));
            }
        }
        for needle in &self.body_excludes {
            if request.body.contains(needle) {
                mismatches.push(format!("body: contained forbidden substring {needle:?}"));
            }
        }
        if !self.json.is_empty() && request.json.is_none() {
            mismatches.push("body: expected valid JSON".to_string());
            return mismatches;
        }
        let Some(body) = request.json.as_ref() else {
            return mismatches;
        };
        for expectation in &self.json {
            match expectation {
                JsonExpectation::Equals { pointer, value } => match body.pointer(pointer) {
                    Some(actual) if actual == value => {}
                    Some(actual) => {
                        mismatches.push(format!("json {pointer}: expected {value}, got {actual}"))
                    }
                    None => mismatches.push(format!(
                        "json {pointer}: expected {value}, but the pointer was absent"
                    )),
                },
                JsonExpectation::Present { pointer } => {
                    if body.pointer(pointer).is_none() {
                        mismatches.push(format!("json {pointer}: expected it to be present"));
                    }
                }
                JsonExpectation::Absent { pointer } => {
                    if let Some(actual) = body.pointer(pointer) {
                        mismatches.push(format!(
                            "json {pointer}: expected it to be absent, got {actual}"
                        ));
                    }
                }
                JsonExpectation::StringContains { pointer, needle } => {
                    match body.pointer(pointer).and_then(Value::as_str) {
                        Some(actual) if actual.contains(needle) => {}
                        Some(actual) => mismatches.push(format!(
                            "json {pointer}: expected a string containing {needle:?}, got {actual:?}"
                        )),
                        None => mismatches.push(format!(
                            "json {pointer}: expected a string containing {needle:?}"
                        )),
                    }
                }
            }
        }
        mismatches
    }

    pub fn matches(&self, request: &RecordedRequest) -> Result<(), Vec<String>> {
        let mismatches = self.mismatches(request);
        if mismatches.is_empty() {
            Ok(())
        } else {
            Err(mismatches)
        }
    }
}

/// One fragment of a scripted response, optionally delayed from the previous
/// fragment. Fragmentation happens at actual TCP write boundaries.
#[derive(Clone, Debug)]
pub struct ResponseChunk {
    bytes: Vec<u8>,
    delay_before: Duration,
}

impl ResponseChunk {
    pub fn new(bytes: impl AsRef<[u8]>) -> Self {
        Self {
            bytes: bytes.as_ref().to_vec(),
            delay_before: Duration::ZERO,
        }
    }

    pub fn after(delay: Duration, bytes: impl AsRef<[u8]>) -> Self {
        Self {
            bytes: bytes.as_ref().to_vec(),
            delay_before: delay,
        }
    }

    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub fn delay_before(&self) -> Duration {
        self.delay_before
    }
}

/// One OpenAI-compatible tool call emitted by [`ScriptedResponse::tool_calls`].
#[derive(Clone, Debug)]
pub struct ScriptedToolCall {
    pub id: String,
    pub name: String,
    pub arguments: Value,
}

impl ScriptedToolCall {
    pub fn new(id: impl Into<String>, name: impl Into<String>, arguments: Value) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            arguments,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ResponseTerminal {
    Complete,
    Eof,
    Reset,
}

/// A response plan. Constructors cover common OpenAI responses; modifiers add
/// synchronization, fragmentation, delay, early EOF, or a TCP reset.
#[derive(Clone, Debug)]
pub struct ScriptedResponse {
    status: u16,
    content_type: String,
    headers: Vec<(String, String)>,
    chunks: Vec<ResponseChunk>,
    send_head: bool,
    terminal: ResponseTerminal,
    delay_before: Duration,
    wait_for_gate: Option<String>,
    hold_open_until: Option<String>,
}

impl ScriptedResponse {
    pub fn json(status: u16, body: impl Into<String>) -> Self {
        Self::http(status, "application/json", body)
    }

    pub fn http_error(status: u16, body: impl Into<String>) -> Self {
        Self::json(status, body)
    }

    pub fn http(status: u16, content_type: impl Into<String>, body: impl Into<String>) -> Self {
        Self {
            status,
            content_type: content_type.into(),
            headers: Vec::new(),
            chunks: vec![ResponseChunk::new(body.into())],
            send_head: true,
            terminal: ResponseTerminal::Complete,
            delay_before: Duration::ZERO,
            wait_for_gate: None,
            hold_open_until: None,
        }
    }

    pub fn raw_sse(body: impl Into<String>) -> Self {
        Self::http(200, "text/event-stream", body)
    }

    pub fn text(text: impl AsRef<str>) -> Self {
        Self::raw_sse(super::sse_text(text.as_ref()))
    }

    pub fn tool_call(call: ScriptedToolCall) -> Self {
        Self::tool_calls([call])
    }

    pub fn tool_calls(calls: impl IntoIterator<Item = ScriptedToolCall>) -> Self {
        let calls: Vec<Value> = calls
            .into_iter()
            .enumerate()
            .map(|(index, call)| {
                json!({
                    "index": index,
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": call.arguments.to_string(),
                    }
                })
            })
            .collect();
        let first = json!({
            "choices": [{
                "delta": {"tool_calls": calls},
                "finish_reason": null
            }]
        });
        let finish = json!({
            "choices": [{"delta": {}, "finish_reason": "tool_calls"}]
        });
        Self::raw_sse(format!(
            "data: {first}\n\ndata: {finish}\n\ndata: [DONE]\n\n"
        ))
    }

    /// Close the connection cleanly before sending an HTTP response.
    pub fn eof() -> Self {
        Self::disconnected(ResponseTerminal::Eof)
    }

    /// Reset the connection before sending an HTTP response.
    pub fn reset() -> Self {
        Self::disconnected(ResponseTerminal::Reset)
    }

    fn disconnected(terminal: ResponseTerminal) -> Self {
        Self {
            status: 200,
            content_type: "application/octet-stream".to_string(),
            headers: Vec::new(),
            chunks: Vec::new(),
            send_head: false,
            terminal,
            delay_before: Duration::ZERO,
            wait_for_gate: None,
            hold_open_until: None,
        }
    }

    pub fn with_header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.push((name.into(), value.into()));
        self
    }

    pub fn delayed(mut self, delay: Duration) -> Self {
        self.delay_before = delay;
        self
    }

    /// Wait for a named gate before writing any response bytes.
    pub fn wait_for_gate(mut self, gate: impl Into<String>) -> Self {
        self.wait_for_gate = Some(gate.into());
        self
    }

    /// Write all response bytes, then keep the connection open until the gate
    /// is released. The response omits `content-length` so the open connection
    /// remains observable by the client.
    pub fn hold_open_until(mut self, gate: impl Into<String>) -> Self {
        self.hold_open_until = Some(gate.into());
        self
    }

    /// Replace the body with explicit wire fragments.
    pub fn with_chunks(mut self, chunks: impl IntoIterator<Item = ResponseChunk>) -> Self {
        self.chunks = chunks.into_iter().collect();
        self
    }

    /// Split the current body into fragments, delaying every fragment after
    /// the first by `delay_between`.
    pub fn fragmented(mut self, max_chunk_bytes: usize, delay_between: Duration) -> Self {
        assert!(
            max_chunk_bytes > 0,
            "response fragment size must be non-zero"
        );
        let bytes: Vec<u8> = self
            .chunks
            .iter()
            .flat_map(|chunk| chunk.bytes.iter().copied())
            .collect();
        self.chunks = bytes
            .chunks(max_chunk_bytes)
            .enumerate()
            .map(|(index, bytes)| ResponseChunk {
                bytes: bytes.to_vec(),
                delay_before: if index == 0 {
                    Duration::ZERO
                } else {
                    delay_between
                },
            })
            .collect();
        self
    }

    /// Send the configured head/body, then end before the declared body is
    /// complete. Clients observe an incomplete-message EOF.
    pub fn finish_with_eof(mut self) -> Self {
        self.terminal = ResponseTerminal::Eof;
        self
    }

    /// Send the configured head/body, then reset the TCP connection.
    pub fn finish_with_reset(mut self) -> Self {
        self.terminal = ResponseTerminal::Reset;
        self
    }

    fn body_len(&self) -> usize {
        self.chunks.iter().map(|chunk| chunk.bytes.len()).sum()
    }

    fn http_head(&self) -> Vec<u8> {
        let reason = status_reason(self.status);
        let mut head = format!("HTTP/1.1 {} {}\r\n", self.status, reason);
        let has_content_type = self
            .headers
            .iter()
            .any(|(name, _)| name.eq_ignore_ascii_case("content-type"));
        let has_content_length = self
            .headers
            .iter()
            .any(|(name, _)| name.eq_ignore_ascii_case("content-length"));
        let has_connection = self
            .headers
            .iter()
            .any(|(name, _)| name.eq_ignore_ascii_case("connection"));
        if !has_content_type {
            head.push_str(&format!("content-type: {}\r\n", self.content_type));
        }
        for (name, value) in &self.headers {
            head.push_str(&format!("{name}: {value}\r\n"));
        }
        if !has_content_length && self.hold_open_until.is_none() {
            let declared = match self.terminal {
                ResponseTerminal::Complete => self.body_len(),
                // Deliberately promise one more byte so an orderly close is an
                // observable incomplete response rather than normal EOF.
                ResponseTerminal::Eof => self.body_len().saturating_add(1),
                ResponseTerminal::Reset => self.body_len(),
            };
            head.push_str(&format!("content-length: {declared}\r\n"));
        }
        if !has_connection {
            head.push_str("connection: close\r\n");
        }
        head.push_str("\r\n");
        head.into_bytes()
    }
}

impl From<super::Response> for ScriptedResponse {
    fn from(response: super::Response) -> Self {
        Self {
            status: response.status,
            content_type: response.content_type.to_string(),
            headers: response.headers,
            chunks: vec![ResponseChunk::new(response.body)],
            send_head: true,
            terminal: ResponseTerminal::Complete,
            delay_before: Duration::ZERO,
            wait_for_gate: None,
            hold_open_until: None,
        }
    }
}

/// One strictly ordered `/chat/completions` interaction.
#[derive(Clone, Debug)]
pub struct ChatStep {
    label: Option<String>,
    matcher: RequestMatcher,
    response: ScriptedResponse,
    required: bool,
}

impl ChatStep {
    pub fn new(response: ScriptedResponse) -> Self {
        Self {
            label: None,
            matcher: RequestMatcher::default(),
            response,
            required: true,
        }
    }

    pub fn expecting(matcher: RequestMatcher, response: ScriptedResponse) -> Self {
        Self {
            matcher,
            ..Self::new(response)
        }
    }

    pub fn named(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Optional steps may be skipped when their matcher does not accept the
    /// next request and do not make [`ScriptedOpenAiServer::assert_clean`] fail.
    pub fn optional(mut self) -> Self {
        self.required = false;
        self
    }

    pub fn label(&self) -> Option<&str> {
        self.label.as_deref()
    }

    pub fn matcher(&self) -> &RequestMatcher {
        &self.matcher
    }

    pub fn response(&self) -> &ScriptedResponse {
        &self.response
    }

    pub fn is_required(&self) -> bool {
        self.required
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ScriptedFailureKind {
    RequestRead,
    UnexpectedRoute,
    UnexpectedChatRequest,
    MismatchedChatRequest,
    WrongRouteMethod,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ScriptedFailure {
    pub kind: ScriptedFailureKind,
    pub request_sequence: Option<usize>,
    pub message: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct UnconsumedStep {
    pub position: usize,
    pub label: Option<String>,
    pub required: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ScriptedServerInspection {
    pub requests: Vec<RecordedRequest>,
    pub failures: Vec<ScriptedFailure>,
    pub unconsumed_steps: Vec<UnconsumedStep>,
}

#[derive(Clone, Debug)]
pub struct ScriptedServerError {
    pub failures: Vec<ScriptedFailure>,
    pub unconsumed_required_steps: Vec<UnconsumedStep>,
}

impl fmt::Display for ScriptedServerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "scripted OpenAI server: {} failure(s), {} unconsumed required step(s)",
            self.failures.len(),
            self.unconsumed_required_steps.len()
        )?;
        for failure in &self.failures {
            write!(f, "\n- {:?}: {}", failure.kind, failure.message)?;
        }
        for step in &self.unconsumed_required_steps {
            write!(
                f,
                "\n- unconsumed step {}{}",
                step.position,
                step.label
                    .as_deref()
                    .map(|label| format!(" ({label})"))
                    .unwrap_or_default()
            )?;
        }
        Ok(())
    }
}

impl std::error::Error for ScriptedServerError {}

#[derive(Default)]
struct GateValues {
    released: HashSet<String>,
    waiters: HashMap<String, usize>,
}

#[derive(Default)]
struct Gates {
    values: Mutex<GateValues>,
    changed: Condvar,
}

#[derive(Default)]
struct RequestOrderValues {
    next_to_route: usize,
    abandoned: HashSet<usize>,
    waiting: HashSet<usize>,
}

#[derive(Default)]
struct RequestOrder {
    values: Mutex<RequestOrderValues>,
    changed: Condvar,
}

struct ServerState {
    steps: Mutex<VecDeque<ChatStep>>,
    models_response: ScriptedResponse,
    requests: Mutex<Vec<RecordedRequest>>,
    failures: Mutex<Vec<ScriptedFailure>>,
    next_sequence: AtomicUsize,
    stop: AtomicBool,
    gates: Gates,
    request_order: RequestOrder,
    workers: Mutex<Vec<JoinHandle<()>>>,
}

impl ServerState {
    fn fail(
        &self,
        kind: ScriptedFailureKind,
        request_sequence: Option<usize>,
        message: impl Into<String>,
    ) {
        self.failures.lock().unwrap().push(ScriptedFailure {
            kind,
            request_sequence,
            message: message.into(),
        });
        self.gates.changed.notify_all();
    }

    fn wait_for_gate(&self, name: &str) -> bool {
        let mut values = self.gates.values.lock().unwrap();
        *values.waiters.entry(name.to_string()).or_default() += 1;
        self.gates.changed.notify_all();
        while !values.released.contains(name) && !self.stop.load(Ordering::Acquire) {
            values = self.gates.changed.wait(values).unwrap();
        }
        let count = values.waiters.get_mut(name).expect("gate waiter exists");
        *count -= 1;
        if *count == 0 {
            values.waiters.remove(name);
        }
        !self.stop.load(Ordering::Acquire)
    }

    fn wait_delay(&self, duration: Duration) -> bool {
        if duration.is_zero() {
            return !self.stop.load(Ordering::Acquire);
        }
        let deadline = Instant::now() + duration;
        let mut values = self.gates.values.lock().unwrap();
        while !self.stop.load(Ordering::Acquire) {
            let now = Instant::now();
            if now >= deadline {
                return true;
            }
            let (next, _) = self
                .gates
                .changed
                .wait_timeout(values, deadline.saturating_duration_since(now))
                .unwrap();
            values = next;
        }
        false
    }

    fn abandon_request(&self, sequence: usize) {
        let mut order = self.request_order.values.lock().unwrap();
        if sequence < order.next_to_route {
            return;
        }
        order.abandoned.insert(sequence);
        advance_abandoned_requests(&mut order);
        self.request_order.changed.notify_all();
    }

    fn route_request_in_arrival_order(
        &self,
        request: &RecordedRequest,
    ) -> Option<ScriptedResponse> {
        let mut order = self.request_order.values.lock().unwrap();
        while request.sequence != order.next_to_route && !self.stop.load(Ordering::Acquire) {
            order.waiting.insert(request.sequence);
            self.request_order.changed.notify_all();
            order = self.request_order.changed.wait(order).unwrap();
        }
        order.waiting.remove(&request.sequence);
        if self.stop.load(Ordering::Acquire) {
            return None;
        }

        self.requests.lock().unwrap().push(request.clone());
        self.gates.changed.notify_all();
        let response = route_request(self, request);

        order.next_to_route += 1;
        advance_abandoned_requests(&mut order);
        self.request_order.changed.notify_all();
        Some(response)
    }
}

fn advance_abandoned_requests(order: &mut RequestOrderValues) {
    while order.abandoned.remove(&order.next_to_route) {
        order.next_to_route += 1;
    }
}

pub struct ScriptedOpenAiServerBuilder {
    steps: Vec<ChatStep>,
    models_response: ScriptedResponse,
}

impl Default for ScriptedOpenAiServerBuilder {
    fn default() -> Self {
        Self {
            steps: Vec::new(),
            models_response: ScriptedResponse::json(
                200,
                r#"{"object":"list","data":[{"id":"test-model","object":"model"}]}"#,
            ),
        }
    }
}

impl ScriptedOpenAiServerBuilder {
    pub fn chat_step(mut self, step: ChatStep) -> Self {
        self.steps.push(step);
        self
    }

    pub fn chat_steps(mut self, steps: impl IntoIterator<Item = ChatStep>) -> Self {
        self.steps.extend(steps);
        self
    }

    /// Configure the repeatable `/models` response. Model requests never
    /// consume chat steps.
    pub fn models_response(mut self, response: ScriptedResponse) -> Self {
        self.models_response = response;
        self
    }

    pub fn start(self) -> io::Result<ScriptedOpenAiServer> {
        ScriptedOpenAiServer::start(self.steps, self.models_response)
    }
}

/// Route-aware OpenAI-compatible server with a strict chat-response script.
pub struct ScriptedOpenAiServer {
    url: String,
    state: Arc<ServerState>,
    listener_thread: Mutex<Option<JoinHandle<()>>>,
}

impl ScriptedOpenAiServer {
    pub fn builder() -> ScriptedOpenAiServerBuilder {
        ScriptedOpenAiServerBuilder::default()
    }

    /// Convenience constructor matching [`super::FakeOpenAiServer::new`]: a
    /// sandbox that forbids loopback binding returns `None`; other bind errors
    /// remain test failures.
    pub fn new(steps: Vec<ChatStep>) -> Option<Self> {
        match Self::builder().chat_steps(steps).start() {
            Ok(server) => Some(server),
            Err(error) if error.kind() == io::ErrorKind::PermissionDenied => None,
            Err(error) => panic!("binding scripted OpenAI server: {error}"),
        }
    }

    fn start(steps: Vec<ChatStep>, models_response: ScriptedResponse) -> io::Result<Self> {
        let listener = TcpListener::bind("127.0.0.1:0")?;
        listener.set_nonblocking(true)?;
        let url = format!("http://{}", listener.local_addr()?);
        let state = Arc::new(ServerState {
            steps: Mutex::new(steps.into()),
            models_response,
            requests: Mutex::new(Vec::new()),
            failures: Mutex::new(Vec::new()),
            next_sequence: AtomicUsize::new(0),
            stop: AtomicBool::new(false),
            gates: Gates::default(),
            request_order: RequestOrder::default(),
            workers: Mutex::new(Vec::new()),
        });
        let thread_state = Arc::clone(&state);
        let listener_thread = std::thread::Builder::new()
            .name("scripted-openai-listener".to_string())
            .spawn(move || listener_loop(listener, thread_state))?;
        Ok(Self {
            url,
            state,
            listener_thread: Mutex::new(Some(listener_thread)),
        })
    }

    pub fn url(&self) -> &str {
        &self.url
    }

    pub fn v1_url(&self) -> String {
        format!("{}/v1", self.url)
    }

    pub fn chat_url(&self) -> String {
        format!("{}/v1/chat/completions", self.url)
    }

    pub fn requests(&self) -> Vec<RecordedRequest> {
        self.state.requests.lock().unwrap().clone()
    }

    pub fn failures(&self) -> Vec<ScriptedFailure> {
        self.state.failures.lock().unwrap().clone()
    }

    pub fn unconsumed_steps(&self) -> Vec<UnconsumedStep> {
        self.state
            .steps
            .lock()
            .unwrap()
            .iter()
            .enumerate()
            .map(|(position, step)| UnconsumedStep {
                position,
                label: step.label.clone(),
                required: step.required,
            })
            .collect()
    }

    pub fn inspection(&self) -> ScriptedServerInspection {
        ScriptedServerInspection {
            requests: self.requests(),
            failures: self.failures(),
            unconsumed_steps: self.unconsumed_steps(),
        }
    }

    pub fn assert_clean(&self) -> Result<(), ScriptedServerError> {
        let failures = self.failures();
        let unconsumed_required_steps: Vec<_> = self
            .unconsumed_steps()
            .into_iter()
            .filter(|step| step.required)
            .collect();
        if failures.is_empty() && unconsumed_required_steps.is_empty() {
            Ok(())
        } else {
            Err(ScriptedServerError {
                failures,
                unconsumed_required_steps,
            })
        }
    }

    pub fn release_gate(&self, name: impl Into<String>) -> bool {
        let inserted = self
            .state
            .gates
            .values
            .lock()
            .unwrap()
            .released
            .insert(name.into());
        self.state.gates.changed.notify_all();
        inserted
    }

    pub fn is_gate_waiting(&self, name: &str) -> bool {
        self.state
            .gates
            .values
            .lock()
            .unwrap()
            .waiters
            .get(name)
            .copied()
            .unwrap_or(0)
            > 0
    }

    pub fn wait_for_gate_waiter(&self, name: &str, timeout: Duration) -> bool {
        let deadline = Instant::now() + timeout;
        let mut values = self.state.gates.values.lock().unwrap();
        loop {
            if values.waiters.get(name).copied().unwrap_or(0) > 0 {
                return true;
            }
            let now = Instant::now();
            if now >= deadline {
                return false;
            }
            let (next, result) = self
                .state
                .gates
                .changed
                .wait_timeout(values, deadline.saturating_duration_since(now))
                .unwrap();
            values = next;
            if result.timed_out() {
                return values.waiters.get(name).copied().unwrap_or(0) > 0;
            }
        }
    }

    pub fn wait_for_request_count(&self, count: usize, timeout: Duration) -> bool {
        let deadline = Instant::now() + timeout;
        let mut values = self.state.gates.values.lock().unwrap();
        loop {
            if self.state.requests.lock().unwrap().len() >= count {
                return true;
            }
            let now = Instant::now();
            if now >= deadline {
                return false;
            }
            let wait = deadline
                .saturating_duration_since(now)
                .min(Duration::from_millis(25));
            let (next, _) = self.state.gates.changed.wait_timeout(values, wait).unwrap();
            values = next;
        }
    }

    /// Stop accepting requests, unblock held responses, and join all server
    /// threads. Safe to call more than once.
    pub fn shutdown(&self) {
        self.state.stop.store(true, Ordering::Release);
        self.state.gates.changed.notify_all();
        self.state.request_order.changed.notify_all();
        if let Some(handle) = self.listener_thread.lock().unwrap().take() {
            let _ = handle.join();
        }
        let workers = std::mem::take(&mut *self.state.workers.lock().unwrap());
        for worker in workers {
            let _ = worker.join();
        }
    }
}

impl Drop for ScriptedOpenAiServer {
    fn drop(&mut self) {
        self.shutdown();
    }
}

fn listener_loop(listener: TcpListener, state: Arc<ServerState>) {
    let mut connection_sequence = 0usize;
    while !state.stop.load(Ordering::Acquire) {
        let (stream, _) = match listener.accept() {
            Ok(connection) => connection,
            Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                std::thread::sleep(ACCEPT_POLL_INTERVAL);
                continue;
            }
            Err(error) => {
                if !state.stop.load(Ordering::Acquire) {
                    state.fail(
                        ScriptedFailureKind::RequestRead,
                        None,
                        format!("accepting request: {error}"),
                    );
                }
                break;
            }
        };
        let accepted_sequence = connection_sequence;
        connection_sequence = connection_sequence.wrapping_add(1);
        let worker_state = Arc::clone(&state);
        let worker = std::thread::Builder::new()
            .name(format!("scripted-openai-connection-{accepted_sequence}"))
            .spawn(move || handle_connection(stream, accepted_sequence, worker_state));
        match worker {
            Ok(worker) => state.workers.lock().unwrap().push(worker),
            Err(error) => state.fail(
                ScriptedFailureKind::RequestRead,
                None,
                format!("spawning response worker: {error}"),
            ),
        }
    }
}

fn handle_connection(mut stream: TcpStream, accepted_sequence: usize, state: Arc<ServerState>) {
    // HTTP clients commonly open an idle speculative socket alongside the one
    // carrying the request. Reading on a worker keeps that idle socket from
    // head-of-line blocking the accept loop and every subsequent request.
    let _ = stream.set_read_timeout(Some(SOCKET_TIMEOUT));
    let _ = stream.set_write_timeout(Some(SOCKET_TIMEOUT));
    let mut request_sequence = None;
    let request = match read_request(&mut stream, &state, &mut request_sequence) {
        Ok(request) => request,
        Err(error) if is_benign_empty_connection(&error) => return,
        Err(error)
            if state.stop.load(Ordering::Acquire) && error.kind() == io::ErrorKind::Interrupted =>
        {
            if let Some(sequence) = request_sequence {
                state.abandon_request(sequence);
            }
            return;
        }
        Err(error) => {
            if let Some(sequence) = request_sequence {
                state.abandon_request(sequence);
            }
            state.fail(
                ScriptedFailureKind::RequestRead,
                request_sequence,
                format!("reading connection #{accepted_sequence}: {error}"),
            );
            return;
        }
    };
    let Some(response) = state.route_request_in_arrival_order(&request) else {
        return;
    };
    execute_response(stream, response, state);
}

fn route_request(state: &ServerState, request: &RecordedRequest) -> ScriptedResponse {
    let path = request.path.split('?').next().unwrap_or(&request.path);
    if path.ends_with("/models") {
        if request.method != "GET" {
            state.fail(
                ScriptedFailureKind::WrongRouteMethod,
                Some(request.sequence),
                format!("models route requires GET, got {}", request.method),
            );
        }
        return state.models_response.clone();
    }
    if path.ends_with("/chat/completions") {
        if request.method != "POST" {
            state.fail(
                ScriptedFailureKind::WrongRouteMethod,
                Some(request.sequence),
                format!(
                    "chat completions route requires POST, got {}",
                    request.method
                ),
            );
        }
        let mut steps = state.steps.lock().unwrap();
        loop {
            let Some(step) = steps.pop_front() else {
                state.fail(
                    ScriptedFailureKind::UnexpectedChatRequest,
                    Some(request.sequence),
                    format!(
                        "unexpected chat request #{} after the script was exhausted",
                        request.sequence
                    ),
                );
                return ScriptedResponse::http_error(500, r#"{"error":"unexpected chat request"}"#);
            };
            let mismatches = step.matcher.mismatches(request);
            if mismatches.is_empty() {
                return step.response;
            }
            if !step.required {
                continue;
            }
            let label = step
                .label
                .as_deref()
                .map(|label| format!(" {label:?}"))
                .unwrap_or_default();
            state.fail(
                ScriptedFailureKind::MismatchedChatRequest,
                Some(request.sequence),
                format!(
                    "chat step{label} did not match request #{}: {}",
                    request.sequence,
                    mismatches.join("; ")
                ),
            );
            // Still return the scripted response. This gives the system under
            // test a bounded outcome while preserving the mismatch as the
            // authoritative harness failure.
            return step.response;
        }
    }
    state.fail(
        ScriptedFailureKind::UnexpectedRoute,
        Some(request.sequence),
        format!(
            "unexpected {} request for {:?}",
            request.method, request.path
        ),
    );
    ScriptedResponse::http_error(404, r#"{"error":"unexpected route"}"#)
}

fn execute_response(mut stream: TcpStream, response: ScriptedResponse, state: Arc<ServerState>) {
    if let Some(gate) = response.wait_for_gate.as_deref()
        && !state.wait_for_gate(gate)
    {
        return;
    }
    if !state.wait_delay(response.delay_before) {
        return;
    }
    if !response.send_head {
        finish_stream(&stream, response.terminal);
        return;
    }
    if stream.write_all(&response.http_head()).is_err() {
        return;
    }
    for chunk in &response.chunks {
        if !state.wait_delay(chunk.delay_before) {
            return;
        }
        if stream.write_all(&chunk.bytes).is_err() || stream.flush().is_err() {
            return;
        }
    }
    if let Some(gate) = response.hold_open_until.as_deref()
        && !state.wait_for_gate(gate)
    {
        return;
    }
    finish_stream(&stream, response.terminal);
}

fn finish_stream(stream: &TcpStream, terminal: ResponseTerminal) {
    match terminal {
        ResponseTerminal::Complete | ResponseTerminal::Eof => {
            let _ = stream.shutdown(Shutdown::Both);
        }
        ResponseTerminal::Reset => reset_stream(stream),
    }
}

#[cfg(unix)]
fn reset_stream(stream: &TcpStream) {
    use std::os::fd::AsRawFd;
    let linger = libc::linger {
        l_onoff: 1,
        l_linger: 0,
    };
    // SAFETY: `linger` is a valid value for the duration of this call and the
    // file descriptor is borrowed from a live `TcpStream`.
    unsafe {
        libc::setsockopt(
            stream.as_raw_fd(),
            libc::SOL_SOCKET,
            libc::SO_LINGER,
            std::ptr::from_ref(&linger).cast(),
            std::mem::size_of_val(&linger) as libc::socklen_t,
        );
    }
    // Do not call `shutdown` here. An orderly shutdown queues FIN before the
    // descriptor is closed, which turns the intended reset into a successful
    // EOF on Linux. The stream is dropped immediately after this returns; with
    // zero-linger set, that close emits the required RST.
}

#[cfg(not(unix))]
fn reset_stream(stream: &TcpStream) {
    let _ = stream.shutdown(Shutdown::Both);
}

fn read_request(
    stream: &mut TcpStream,
    state: &ServerState,
    request_sequence: &mut Option<usize>,
) -> io::Result<RecordedRequest> {
    let mut bytes = Vec::new();
    let mut buf = [0u8; 4096];
    let deadline = Instant::now() + REQUEST_READ_DEADLINE;
    let header_end = loop {
        let read = match read_with_deadline(stream, &mut buf, deadline, &state.stop) {
            Ok(read) => read,
            Err(error)
                if matches!(
                    error.kind(),
                    io::ErrorKind::WouldBlock | io::ErrorKind::TimedOut
                ) =>
            {
                let message = if bytes.is_empty() {
                    "connection idle before request headers".to_string()
                } else {
                    format!("request headers timed out after {} bytes", bytes.len())
                };
                return Err(io::Error::new(error.kind(), message));
            }
            Err(error) => return Err(error),
        };
        if read == 0 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "connection closed before request headers",
            ));
        }
        if request_sequence.is_none() {
            *request_sequence = Some(state.next_sequence.fetch_add(1, Ordering::AcqRel));
        }
        bytes.extend_from_slice(&buf[..read]);
        if bytes.len() > REQUEST_LIMIT_BYTES {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "request exceeds 8 MiB limit",
            ));
        }
        if let Some(position) = bytes.windows(4).position(|window| window == b"\r\n\r\n") {
            break position + 4;
        }
    };
    let raw_head = std::str::from_utf8(&bytes[..header_end]).map_err(|error| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("request headers are not UTF-8: {error}"),
        )
    })?;
    let mut lines = raw_head.split("\r\n");
    let request_line = lines
        .next()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing request line"))?;
    let mut request_parts = request_line.split_whitespace();
    let method = request_parts
        .next()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing request method"))?
        .to_ascii_uppercase();
    let path = request_parts
        .next()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing request path"))?
        .to_string();
    let mut headers = BTreeMap::new();
    for line in lines.filter(|line| !line.is_empty()) {
        let Some((name, value)) = line.split_once(':') else {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("malformed request header {line:?}"),
            ));
        };
        headers.insert(name.trim().to_ascii_lowercase(), value.trim().to_string());
    }
    let content_length = headers
        .get("content-length")
        .map(|value| {
            value.parse::<usize>().map_err(|error| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("invalid content-length: {error}"),
                )
            })
        })
        .transpose()?
        .unwrap_or(0);
    if header_end.saturating_add(content_length) > REQUEST_LIMIT_BYTES {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "request exceeds 8 MiB limit",
        ));
    }
    while bytes.len() < header_end + content_length {
        let read = match read_with_deadline(stream, &mut buf, deadline, &state.stop) {
            Ok(read) => read,
            Err(error)
                if matches!(
                    error.kind(),
                    io::ErrorKind::WouldBlock | io::ErrorKind::TimedOut
                ) =>
            {
                return Err(io::Error::new(
                    error.kind(),
                    format!(
                        "request body timed out after {} of {content_length} bytes",
                        bytes.len().saturating_sub(header_end)
                    ),
                ));
            }
            Err(error) => return Err(error),
        };
        if read == 0 {
            break;
        }
        bytes.extend_from_slice(&buf[..read]);
    }
    if bytes.len() < header_end + content_length {
        return Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            "connection closed before request body completed",
        ));
    }
    let body =
        String::from_utf8_lossy(&bytes[header_end..header_end + content_length]).into_owned();
    let json = serde_json::from_str(&body).ok();
    Ok(RecordedRequest {
        sequence: request_sequence.expect("a parsed request observed at least one byte"),
        method,
        path,
        headers,
        body,
        json,
    })
}

fn read_with_deadline(
    stream: &mut TcpStream,
    buf: &mut [u8],
    deadline: Instant,
    stop: &AtomicBool,
) -> io::Result<usize> {
    loop {
        if stop.load(Ordering::Acquire) {
            return Err(io::Error::new(
                io::ErrorKind::Interrupted,
                "server shut down while reading request",
            ));
        }
        let now = Instant::now();
        if now >= deadline {
            return Err(io::Error::new(
                io::ErrorKind::TimedOut,
                "absolute request-read deadline elapsed",
            ));
        }
        stream.set_read_timeout(Some(
            deadline
                .saturating_duration_since(now)
                .min(REQUEST_READ_POLL_INTERVAL),
        ))?;
        match stream.read(buf) {
            Err(error)
                if matches!(
                    error.kind(),
                    io::ErrorKind::WouldBlock | io::ErrorKind::TimedOut
                ) && Instant::now() < deadline => {}
            result => return result,
        }
    }
}

fn is_benign_empty_connection(error: &io::Error) -> bool {
    let message = error.to_string();
    (matches!(
        error.kind(),
        io::ErrorKind::WouldBlock | io::ErrorKind::TimedOut
    ) && message.contains("idle before request headers"))
        || (error.kind() == io::ErrorKind::UnexpectedEof
            && message.contains("closed before request headers"))
}

fn status_reason(status: u16) -> &'static str {
    match status {
        200 => "OK",
        201 => "Created",
        204 => "No Content",
        400 => "Bad Request",
        401 => "Unauthorized",
        402 => "Payment Required",
        403 => "Forbidden",
        404 => "Not Found",
        408 => "Request Timeout",
        409 => "Conflict",
        429 => "Too Many Requests",
        500 => "Internal Server Error",
        502 => "Bad Gateway",
        503 => "Service Unavailable",
        504 => "Gateway Timeout",
        _ => "Status",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::openai::OpenAiProvider;
    use crate::provider::Provider;
    use crate::token::StaticToken;
    use crate::types::{ChatRequest, Content, Message, RequestProfile};

    fn request(model: &str) -> ChatRequest {
        ChatRequest {
            model: model.to_string(),
            request_id: None,
            retry_attempt: 0,
            user_turn: false,
            canonical_objective: None,
            messages: vec![Message::user("hello")].into(),
            tools: Vec::new().into(),
            tool_envelope: None,
            max_tokens: 32,
            temperature: None,
            top_p: None,
            frequency_penalty: None,
            thinking_budget: None,
            reasoning_effort: None,
            profile: RequestProfile::default(),
        }
    }

    fn raw_chat_request(body: &str) -> String {
        format!(
            "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        )
    }

    fn wait_for_request_tickets(
        server: &ScriptedOpenAiServer,
        count: usize,
        timeout: Duration,
    ) -> bool {
        let deadline = Instant::now() + timeout;
        while Instant::now() < deadline {
            if server.state.next_sequence.load(Ordering::Acquire) >= count {
                return true;
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        server.state.next_sequence.load(Ordering::Acquire) >= count
    }

    fn wait_for_order_waiter(
        server: &ScriptedOpenAiServer,
        sequence: usize,
        timeout: Duration,
    ) -> bool {
        let deadline = Instant::now() + timeout;
        let mut order = server.state.request_order.values.lock().unwrap();
        loop {
            if order.waiting.contains(&sequence) {
                return true;
            }
            let now = Instant::now();
            if now >= deadline {
                return false;
            }
            let (next, result) = server
                .state
                .request_order
                .changed
                .wait_timeout(order, deadline.saturating_duration_since(now))
                .unwrap();
            order = next;
            if result.timed_out() {
                return order.waiting.contains(&sequence);
            }
        }
    }

    #[tokio::test]
    async fn model_discovery_is_repeatable_and_does_not_consume_chat_steps() {
        let Some(server) = ScriptedOpenAiServer::new(vec![ChatStep::expecting(
            RequestMatcher::any().json_eq("/model", json!("test-model")),
            ScriptedResponse::text("hello back"),
        )]) else {
            return;
        };
        let provider = OpenAiProvider::with_token_source(
            server.v1_url(),
            Arc::new(StaticToken("test".to_string())),
        );

        let first = provider.list_models().await.unwrap();
        let second = provider.list_models().await.unwrap();
        assert_eq!(first[0].id, "test-model");
        assert_eq!(second[0].id, "test-model");

        let mut sink = |_| {};
        let completion = provider
            .stream(request("test-model"), &mut sink)
            .await
            .unwrap();
        assert!(
            matches!(completion.content.first(), Some(Content::Text(text)) if text == "hello back")
        );
        assert_eq!(server.requests().len(), 3);
        server.assert_clean().unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn idle_speculative_connection_does_not_block_real_requests() {
        let Some(server) = ScriptedOpenAiServer::new(vec![ChatStep::new(ScriptedResponse::text(
            "real request completed",
        ))]) else {
            return;
        };
        let idle = TcpStream::connect(server.url().trim_start_matches("http://")).unwrap();
        tokio::time::sleep(Duration::from_millis(50)).await;

        let client = reqwest::Client::new();
        let response = tokio::time::timeout(
            Duration::from_secs(1),
            client
                .post(server.chat_url())
                .json(&json!({"model": "test-model", "messages": []}))
                .send(),
        )
        .await
        .expect("real request was head-of-line blocked")
        .unwrap();
        assert!(response.status().is_success());

        drop(idle);
        server.assert_clean().unwrap();
    }

    #[test]
    fn concurrent_connections_consume_chat_steps_in_request_arrival_order() {
        let Some(server) = ScriptedOpenAiServer::new(vec![
            ChatStep::expecting(
                RequestMatcher::any().json_eq("/model", json!("first")),
                ScriptedResponse::text("first response"),
            ),
            ChatStep::expecting(
                RequestMatcher::any().json_eq("/model", json!("second")),
                ScriptedResponse::text("second response"),
            ),
        ]) else {
            return;
        };
        let address = server.url().trim_start_matches("http://");

        let first_body = r#"{"model":"first","messages":[]}"#;
        let first_request = raw_chat_request(first_body);
        let first_body_start = first_request.len() - first_body.len();
        let mut first = TcpStream::connect(address).unwrap();
        first
            .set_read_timeout(Some(Duration::from_secs(2)))
            .unwrap();
        first
            .write_all(&first_request.as_bytes()[..first_body_start])
            .unwrap();
        assert!(wait_for_request_tickets(&server, 1, Duration::from_secs(1)));

        let second_body = r#"{"model":"second","messages":[]}"#;
        let mut second = TcpStream::connect(address).unwrap();
        second
            .set_read_timeout(Some(Duration::from_secs(2)))
            .unwrap();
        second
            .write_all(raw_chat_request(second_body).as_bytes())
            .unwrap();
        second.shutdown(Shutdown::Write).unwrap();
        assert!(wait_for_request_tickets(&server, 2, Duration::from_secs(1)));

        // The complete second request must wait for the earlier request body,
        // rather than consuming the first scripted step in its worker thread.
        assert!(wait_for_order_waiter(&server, 1, Duration::from_secs(1)));
        assert!(server.requests().is_empty());

        first
            .write_all(&first_request.as_bytes()[first_body_start..])
            .unwrap();
        first.shutdown(Shutdown::Write).unwrap();
        let mut first_response = String::new();
        first.read_to_string(&mut first_response).unwrap();
        let mut second_response = String::new();
        second.read_to_string(&mut second_response).unwrap();

        assert!(
            first_response.contains("first response"),
            "{first_response}"
        );
        assert!(
            second_response.contains("second response"),
            "{second_response}"
        );
        let requests = server.requests();
        assert_eq!(
            requests
                .iter()
                .map(|request| request.sequence)
                .collect::<Vec<_>>(),
            vec![0, 1]
        );
        assert_eq!(requests[0].json.as_ref().unwrap()["model"], "first");
        assert_eq!(requests[1].json.as_ref().unwrap()["model"], "second");
        server.assert_clean().unwrap();
    }

    #[test]
    fn drip_fed_request_hits_absolute_read_deadline() {
        let Some(server) = ScriptedOpenAiServer::new(Vec::new()) else {
            return;
        };
        let mut stream = TcpStream::connect(server.url().trim_start_matches("http://")).unwrap();
        stream
            .set_write_timeout(Some(Duration::from_millis(250)))
            .unwrap();
        let keep_writing = Arc::new(AtomicBool::new(true));
        let writer_flag = Arc::clone(&keep_writing);
        let writer = std::thread::spawn(move || {
            while writer_flag.load(Ordering::Acquire) {
                if stream.write_all(b"x").is_err() {
                    break;
                }
                std::thread::sleep(Duration::from_millis(25));
            }
        });

        let started = Instant::now();
        let observation_deadline = started + REQUEST_READ_DEADLINE + Duration::from_secs(1);
        let failure = loop {
            if let Some(failure) = server.failures().into_iter().next() {
                break Some(failure);
            }
            if Instant::now() >= observation_deadline {
                break None;
            }
            std::thread::sleep(Duration::from_millis(10));
        };
        keep_writing.store(false, Ordering::Release);
        writer.join().unwrap();

        let failure = failure.expect("drip-fed request outlived its absolute read deadline");
        assert_eq!(failure.kind, ScriptedFailureKind::RequestRead);
        assert_eq!(failure.request_sequence, Some(0));
        assert!(
            started.elapsed() < REQUEST_READ_DEADLINE + Duration::from_secs(1),
            "request failure exceeded its absolute deadline: {:?}",
            started.elapsed()
        );
    }

    #[test]
    fn shutdown_interrupts_partial_request_promptly() {
        let Some(server) = ScriptedOpenAiServer::new(Vec::new()) else {
            return;
        };
        let mut stream = TcpStream::connect(server.url().trim_start_matches("http://")).unwrap();
        stream.write_all(b"POST /v1/chat").unwrap();
        assert!(wait_for_request_tickets(&server, 1, Duration::from_secs(1)));

        let started = Instant::now();
        server.shutdown();
        assert!(
            started.elapsed() < Duration::from_millis(500),
            "shutdown waited too long for a partial request: {:?}",
            started.elapsed()
        );
        assert!(server.failures().is_empty());
    }

    #[test]
    fn request_matcher_reports_all_differences() {
        let request = RecordedRequest {
            sequence: 7,
            method: "POST".to_string(),
            path: "/v1/chat/completions".to_string(),
            headers: BTreeMap::from([("authorization".to_string(), "Bearer real".to_string())]),
            body: r#"{"model":"other","messages":[]}"#.to_string(),
            json: Some(json!({"model": "other", "messages": []})),
        };
        let mismatches = RequestMatcher::any()
            .method("GET")
            .header("authorization", "Bearer expected")
            .body_contains("needle")
            .body_excludes("messages")
            .json_eq("/model", json!("expected"))
            .json_present("/stream")
            .json_absent("/messages")
            .mismatches(&request);
        assert_eq!(mismatches.len(), 7, "{mismatches:#?}");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn named_gate_holds_response_without_blocking_observation() {
        let Some(server) = ScriptedOpenAiServer::new(vec![ChatStep::new(
            ScriptedResponse::text("released").wait_for_gate("model"),
        )]) else {
            return;
        };
        let url = server.chat_url();
        let request = tokio::spawn(async move {
            reqwest::Client::new()
                .post(url)
                .json(&json!({"model": "test-model", "messages": []}))
                .send()
                .await
                .unwrap()
                .text()
                .await
                .unwrap()
        });
        assert!(server.wait_for_gate_waiter("model", Duration::from_secs(2)));
        assert!(!request.is_finished());
        assert!(server.release_gate("model"));
        assert!(request.await.unwrap().contains("released"));
        server.assert_clean().unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn hold_open_gate_waits_after_headers_and_body() {
        let Some(server) = ScriptedOpenAiServer::new(vec![ChatStep::new(
            ScriptedResponse::raw_sse("").hold_open_until("close"),
        )]) else {
            return;
        };
        let response = reqwest::Client::new()
            .post(server.chat_url())
            .json(&json!({"model": "test-model", "messages": []}))
            .send()
            .await
            .unwrap();
        assert!(server.wait_for_gate_waiter("close", Duration::from_secs(2)));
        let body = tokio::spawn(async move { response.text().await.unwrap() });
        tokio::task::yield_now().await;
        assert!(!body.is_finished());
        assert!(server.release_gate("close"));
        assert_eq!(body.await.unwrap(), "");
        server.assert_clean().unwrap();
    }

    #[tokio::test]
    async fn orderly_eof_and_tcp_reset_are_distinct_scripted_failures() {
        let Some(server) = ScriptedOpenAiServer::new(vec![
            ChatStep::new(ScriptedResponse::eof()),
            ChatStep::new(ScriptedResponse::reset()),
        ]) else {
            return;
        };
        let client = reqwest::Client::new();
        let eof = client
            .post(server.chat_url())
            .json(&json!({"model": "test-model"}))
            .send()
            .await;
        assert!(eof.is_err(), "clean EOF before HTTP headers must fail");
        let reset = client
            .post(server.chat_url())
            .json(&json!({"model": "test-model"}))
            .send()
            .await;
        assert!(reset.is_err(), "TCP reset before HTTP headers must fail");
        server.assert_clean().unwrap();
    }

    #[tokio::test]
    async fn mismatch_and_unexpected_requests_are_durable_failures() {
        let Some(server) = ScriptedOpenAiServer::new(vec![ChatStep::expecting(
            RequestMatcher::any().json_eq("/model", json!("expected")),
            ScriptedResponse::text("bounded"),
        )]) else {
            return;
        };
        let client = reqwest::Client::new();
        let first = client
            .post(server.chat_url())
            .json(&json!({"model": "wrong"}))
            .send()
            .await
            .unwrap();
        assert!(first.status().is_success());
        let second = client
            .post(server.chat_url())
            .json(&json!({"model": "extra"}))
            .send()
            .await
            .unwrap();
        assert_eq!(second.status(), 500);
        assert_eq!(
            server
                .failures()
                .iter()
                .map(|failure| &failure.kind)
                .collect::<Vec<_>>(),
            vec![
                &ScriptedFailureKind::MismatchedChatRequest,
                &ScriptedFailureKind::UnexpectedChatRequest,
            ]
        );
        assert!(server.assert_clean().is_err());
    }

    #[test]
    fn unconsumed_required_steps_fail_but_optional_steps_do_not() {
        let Some(server) = ScriptedOpenAiServer::new(vec![
            ChatStep::new(ScriptedResponse::text("optional")).optional(),
            ChatStep::new(ScriptedResponse::text("required")).named("must-run"),
        ]) else {
            return;
        };
        let error = server.assert_clean().unwrap_err();
        assert_eq!(error.unconsumed_required_steps.len(), 1);
        assert_eq!(
            error.unconsumed_required_steps[0].label.as_deref(),
            Some("must-run")
        );
    }

    #[test]
    fn response_builders_cover_text_tools_fragmentation_and_transport_faults() {
        let text = ScriptedResponse::text("hello").fragmented(3, Duration::from_millis(1));
        assert!(text.chunks.len() > 1);
        assert_eq!(text.body_len(), super::super::sse_text("hello").len());

        let tool = ScriptedResponse::tool_call(ScriptedToolCall::new(
            "call-1",
            "bash",
            json!({"command": "true"}),
        ));
        let body = String::from_utf8(
            tool.chunks
                .iter()
                .flat_map(|chunk| chunk.bytes.iter().copied())
                .collect(),
        )
        .unwrap();
        assert!(body.contains("tool_calls"));
        assert!(body.contains("call-1"));
        assert!(body.contains(r#"{\"command\":\"true\"}"#));

        assert!(!ScriptedResponse::eof().send_head);
        assert_eq!(ScriptedResponse::reset().terminal, ResponseTerminal::Reset);
        assert_eq!(
            ScriptedResponse::raw_sse("partial")
                .finish_with_eof()
                .terminal,
            ResponseTerminal::Eof
        );
    }
}
