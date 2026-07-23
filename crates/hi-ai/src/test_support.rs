use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Mutex};

pub struct FakeOpenAiServer {
    url: String,
    bodies: Arc<Mutex<Vec<String>>>,
    authorizations: Arc<Mutex<Vec<Option<String>>>>,
    request_ids: Arc<Mutex<Vec<Option<String>>>>,
    idempotency_keys: Arc<Mutex<Vec<Option<String>>>>,
}

impl FakeOpenAiServer {
    pub fn new(responses: Vec<Response>) -> Option<Self> {
        let listener = match TcpListener::bind("127.0.0.1:0") {
            Ok(listener) => listener,
            Err(err) if err.kind() == std::io::ErrorKind::PermissionDenied => return None,
            Err(err) => panic!("binding fake OpenAI server: {err}"),
        };
        let url = format!("http://{}", listener.local_addr().unwrap());
        let bodies = Arc::new(Mutex::new(Vec::new()));
        let authorizations = Arc::new(Mutex::new(Vec::new()));
        let request_ids = Arc::new(Mutex::new(Vec::new()));
        let idempotency_keys = Arc::new(Mutex::new(Vec::new()));
        let thread_bodies = bodies.clone();
        let thread_authorizations = authorizations.clone();
        let thread_request_ids = request_ids.clone();
        let thread_idempotency_keys = idempotency_keys.clone();
        std::thread::spawn(move || {
            for response in responses {
                let Ok((mut stream, _)) = listener.accept() else {
                    break;
                };
                let (body, authorization, request_id, idempotency_key) =
                    read_http_request(&mut stream);
                thread_bodies.lock().unwrap().push(body);
                thread_authorizations.lock().unwrap().push(authorization);
                thread_request_ids.lock().unwrap().push(request_id);
                thread_idempotency_keys
                    .lock()
                    .unwrap()
                    .push(idempotency_key);
                let _ = stream.write_all(response.to_http().as_bytes());
            }
        });
        Some(Self {
            url,
            bodies,
            authorizations,
            request_ids,
            idempotency_keys,
        })
    }

    pub fn url(&self) -> &str {
        &self.url
    }

    pub fn bodies(&self) -> Vec<String> {
        self.bodies.lock().unwrap().clone()
    }

    /// The `authorization` header of each request received, in order.
    /// Lowercased by the header parse, so compare against lowercase values.
    pub fn authorizations(&self) -> Vec<Option<String>> {
        self.authorizations.lock().unwrap().clone()
    }

    pub fn request_ids(&self) -> Vec<Option<String>> {
        self.request_ids.lock().unwrap().clone()
    }

    pub fn idempotency_keys(&self) -> Vec<Option<String>> {
        self.idempotency_keys.lock().unwrap().clone()
    }
}

pub struct Response {
    status: u16,
    content_type: &'static str,
    headers: Vec<(String, String)>,
    body: String,
}

impl Response {
    pub fn json(status: u16, body: impl Into<String>) -> Self {
        Self {
            status,
            content_type: "application/json",
            headers: Vec::new(),
            body: body.into(),
        }
    }

    pub fn sse(body: impl Into<String>) -> Self {
        Self {
            status: 200,
            content_type: "text/event-stream",
            headers: Vec::new(),
            body: body.into(),
        }
    }

    pub fn with_header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.push((name.into(), value.into()));
        self
    }

    fn to_http(&self) -> String {
        let reason = match self.status {
            200 => "OK",
            400 => "Bad Request",
            401 => "Unauthorized",
            403 => "Forbidden",
            429 => "Too Many Requests",
            500 => "Internal Server Error",
            502 => "Bad Gateway",
            503 => "Service Unavailable",
            _ => "Status",
        };
        let extra_headers: String = self
            .headers
            .iter()
            .map(|(name, value)| format!("{name}: {value}\r\n"))
            .collect();
        format!(
            "HTTP/1.1 {} {}\r\ncontent-type: {}\r\n{}content-length: {}\r\nconnection: close\r\n\r\n{}",
            self.status,
            reason,
            self.content_type,
            extra_headers,
            self.body.len(),
            self.body
        )
    }
}

pub fn sse_text(text: &str) -> String {
    format!(
        "data: {{\"choices\":[{{\"delta\":{{\"content\":{}}},\"finish_reason\":\"stop\"}}]}}\n\ndata: [DONE]\n\n",
        serde_json::to_string(text).unwrap()
    )
}

/// The request body, plus the `authorization` header value if one was sent.
/// Tests that exercise credential rotation need the header to tell a replayed
/// request apart from the original — the bodies are identical.
fn read_http_request(
    stream: &mut TcpStream,
) -> (String, Option<String>, Option<String>, Option<String>) {
    let mut bytes = Vec::new();
    let mut buf = [0u8; 1024];
    let header_end = loop {
        let n = stream.read(&mut buf).unwrap();
        if n == 0 {
            return (String::new(), None, None, None);
        }
        bytes.extend_from_slice(&buf[..n]);
        if let Some(pos) = bytes.windows(4).position(|w| w == b"\r\n\r\n") {
            break pos + 4;
        }
    };
    let headers = String::from_utf8_lossy(&bytes[..header_end]).to_ascii_lowercase();
    let authorization = headers
        .lines()
        .find_map(|line| line.strip_prefix("authorization:"))
        .map(|value| value.trim().to_string());
    let request_id = header_value(&headers, "x-request-id");
    let idempotency_key = header_value(&headers, "idempotency-key");
    let len = headers
        .lines()
        .find_map(|line| line.strip_prefix("content-length:"))
        .and_then(|value| value.trim().parse::<usize>().ok())
        .unwrap_or(0);
    while bytes.len() < header_end + len {
        let n = stream.read(&mut buf).unwrap();
        if n == 0 {
            break;
        }
        bytes.extend_from_slice(&buf[..n]);
    }
    let body =
        String::from_utf8_lossy(&bytes[header_end..bytes.len().min(header_end + len)]).into_owned();
    (body, authorization, request_id, idempotency_key)
}

fn header_value(headers: &str, name: &str) -> Option<String> {
    let prefix = format!("{name}:");
    headers
        .lines()
        .find_map(|line| line.strip_prefix(&prefix))
        .map(|value| value.trim().to_string())
}
