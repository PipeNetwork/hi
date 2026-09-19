//! Minimal HTTP surface for the websocket bridge.
//!
//! The bridge listener is one port for two things: `GET /` serves the bundled
//! single-page chat client, and `/ws/<channel>` is handed to
//! [`crate::ws::handle_ws`] untouched. Routing happens on a *peek* of the
//! request head, so a websocket upgrade is never consumed before tungstenite
//! reads it.

use std::io::IoSlice;
use std::pin::Pin;
use std::task::{Context, Poll};
use std::time::Duration;
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt, ReadBuf};

use crate::state::SharedState;

/// A stream that can be both read and written, used to unify the plaintext
/// (`TcpStream`) and TLS (`TlsStream<TcpStream>`) connection paths behind a
/// single trait object.
pub trait IoStream: AsyncRead + AsyncWrite + Unpin + Send {}
impl<T: AsyncRead + AsyncWrite + Unpin + Send> IoStream for T {}

/// The browser client, embedded so the server stays a single binary with no
/// asset directory to install or ship alongside it.
pub const INDEX_HTML: &str = include_str!("web/index.html");

/// Reply for paths that are neither the page nor a bridge upgrade.
const NOT_FOUND: &str = "not found\n";

/// Reply when a login body is unreadable or malformed.
const BAD_REQUEST: &str = "{\"error\":\"bad request\"}";

/// Reply when the credentials do not check out.
const UNAUTHORIZED: &str = "{\"error\":\"invalid credentials\"}";

/// Reply when a registration collides with an existing username.
const CONFLICT: &str = "{\"error\":\"username taken\"}";

/// Reply when a peer is hammering `/auth` or `/register`.
const RATE_LIMITED: &str = "{\"error\":\"too many attempts, slow down\"}";

/// Largest request head we buffer before answering. A peer that streams an
/// endless head cannot grow our memory without bound.
pub const MAX_HEAD: usize = 8192;

/// Largest body we accept on `POST /auth`. Login forms are tiny; the cap keeps a
/// peer from streaming an unbounded body before we answer.
const MAX_BODY: usize = 4096;

/// How long we wait for a usable request line before deciding the route. A
/// browser sends its request line immediately, so this only bounds the damage
/// from a peer that connects and then says nothing.
const PEEK_TIMEOUT: Duration = Duration::from_secs(2);

/// How long we wait for the rest of a head once we know one is coming.
const READ_TIMEOUT: Duration = Duration::from_secs(2);

/// What an incoming connection on the bridge port is asking for.
#[derive(Debug, PartialEq, Eq)]
pub enum Route {
    /// `/ws/<channel>...`: a bridge upgrade, forwarded to tungstenite.
    Bridge,
    /// `/` or `/index.html`: the embedded browser client.
    Index,
    /// `POST /auth`: exchange credentials for a single-use bridge ticket.
    Auth,
    /// `POST /register`: create an account and return a ticket for it.
    Register,
    /// `GET /health`: liveness/readiness probe for load balancers and
    /// orchestrators. Answers 200 with a small JSON body.
    Health,
    /// Anything else: answer 404 and close.
    NotFound,
}

/// Route a connection from (a prefix of) its request head.
pub fn route(head: &[u8]) -> Route {
    let Some((method, target)) = request_line(head) else {
        return Route::NotFound;
    };
    // Compare the path only: the bridge query string carries its credentials.
    let path = target.split('?').next().unwrap_or_default();
    if path.starts_with("/ws/") {
        Route::Bridge
    } else if path == "/" || path == "/index.html" {
        // Only a GET renders the page; a stray POST is not a page load.
        if method.eq_ignore_ascii_case("GET") {
            Route::Index
        } else {
            Route::NotFound
        }
    } else if path == "/auth" && method.eq_ignore_ascii_case("POST") {
        Route::Auth
    } else if path == "/register" && method.eq_ignore_ascii_case("POST") {
        Route::Register
    } else if path == "/health" && method.eq_ignore_ascii_case("GET") {
        Route::Health
    } else {
        Route::NotFound
    }
}

/// The method and request target at the start of `head`.
///
/// Returns `None` for an incomplete request line, which callers treat as
/// unroutable rather than guessing.
pub fn request_line(head: &[u8]) -> Option<(&str, &str)> {
    // Tolerate bare-LF line endings; browsers send CRLF but tests are terse.
    let line = head.split(|b| *b == b'\n').next()?;
    let line = line.strip_suffix(b"\r").unwrap_or(line);
    let line = std::str::from_utf8(line).ok()?;
    let mut parts = line.split_whitespace();
    let method = parts.next()?;
    let target = parts.next()?;
    // A request line is `<method> <target> <version>`; anything else is noise.
    parts.next()?;
    target.starts_with('/').then_some((method, target))
}

/// The request target at the start of `head`, if the line reads as a method
/// followed by a target. Only exercised by the routing tests; production routing
/// goes through `route`, which needs the method as well.
#[cfg(test)]
pub fn request_target(head: &str) -> Option<&str> {
    request_line(head.as_bytes()).map(|(_, target)| target)
}

/// Read the start of a connection's request *without* consuming it.
///
/// `peek` leaves the bytes queued, so the winning branch still sees the full
/// request: the bridge handshake reads it, and [`serve`] drains it. We wait for
/// a complete request line because a request may arrive in several TCP
/// segments, and routing on a partial first line could send a bridge upgrade to
/// the HTML handler.
///
/// Returns the raw bytes, not a `String`: the head is replayed verbatim into
/// the winning branch, so decoding it here (and re-encoding on replay) would
/// corrupt any non-UTF-8 byte in the request (e.g. a percent-encoded channel
/// name carrying a raw high byte).
pub async fn peek_head<S>(stream: &mut S) -> Vec<u8>
where
    S: AsyncRead + Unpin,
{
    let deadline = tokio::time::Instant::now() + PEEK_TIMEOUT;
    let mut buf = Vec::with_capacity(1024);
    let mut chunk = [0u8; 1024];
    loop {
        match tokio::time::timeout_at(deadline, stream.read(&mut chunk)).await {
            Ok(Ok(n)) if n > 0 => {
                buf.extend_from_slice(&chunk[..n]);
                // Enough to route: a full first line, or a full buffer.
                if buf.contains(&b'\n') || buf.len() >= MAX_HEAD {
                    return buf;
                }
            }
            // Nothing pending, peer gone, or out of time.
            Ok(_) => return Vec::new(),
            Err(_) => return Vec::new(),
        }
        // Partial first line: give the rest of the segment a moment to land.
        tokio::time::sleep(Duration::from_millis(5)).await;
        if tokio::time::Instant::now() >= deadline {
            return Vec::new();
        }
    }
}

/// A stream that replays a buffered prefix before reading from the underlying
/// stream. The bridge route peeks the request head to decide the route; the
/// winning branch must still see those bytes, so the peeked head is wrapped
/// here and replayed before the real request is read.
pub struct PrefixedStream<S> {
    prefix: Vec<u8>,
    inner: S,
}

impl<S> PrefixedStream<S> {
    pub fn new(prefix: Vec<u8>, inner: S) -> Self {
        PrefixedStream { prefix, inner }
    }
}

impl<S: AsyncRead + Unpin> AsyncRead for PrefixedStream<S> {
    fn poll_read(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut ReadBuf<'_>,
    ) -> Poll<std::io::Result<()>> {
        if !self.prefix.is_empty() {
            let n = self.prefix.len().min(buf.remaining());
            buf.put_slice(&self.prefix[..n]);
            self.prefix.drain(..n);
            return Poll::Ready(Ok(()));
        }
        Pin::new(&mut self.inner).poll_read(cx, buf)
    }
}

impl<S: AsyncWrite + Unpin> AsyncWrite for PrefixedStream<S> {
    fn poll_write(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<std::io::Result<usize>> {
        Pin::new(&mut self.inner).poll_write(cx, buf)
    }

    fn poll_flush(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<std::io::Result<()>> {
        Pin::new(&mut self.inner).poll_flush(cx)
    }

    fn poll_shutdown(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<std::io::Result<()>> {
        Pin::new(&mut self.inner).poll_shutdown(cx)
    }

    fn poll_write_vectored(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        bufs: &[IoSlice<'_>],
    ) -> Poll<std::io::Result<usize>> {
        Pin::new(&mut self.inner).poll_write_vectored(cx, bufs)
    }

    fn is_write_vectored(&self) -> bool {
        self.inner.is_write_vectored()
    }
}

/// Answer a routed, non-bridge request and close the connection.
///
/// The head was only peeked, so it is still queued in the socket; each branch
/// consumes it (draining it for the static replies, reading it in full for the
/// login POST) so the stream stays consistent for peers that wait for the
/// connection to close.
pub async fn serve<S>(
    mut stream: S,
    route: Route,
    state: SharedState,
    peer: std::net::IpAddr,
) -> std::io::Result<()>
where
    S: AsyncRead + AsyncWrite + Unpin,
{
    let (status, reason, content_type, body) = match route {
        Route::Index => {
            let _ = drain_head(&mut stream).await;
            (
                200,
                "OK",
                "text/html; charset=utf-8",
                INDEX_HTML.to_string(),
            )
        }
        Route::Auth => match authenticate(&mut stream, &state, peer).await {
            Ok(ticket) => (
                200,
                "OK",
                "application/json",
                format!("{{\"ticket\":\"{ticket}\"}}"),
            ),
            Err(status) => {
                let (reason, body) = match status {
                    401 => ("Unauthorized", UNAUTHORIZED),
                    429 => ("Too Many Requests", RATE_LIMITED),
                    _ => ("Bad Request", BAD_REQUEST),
                };
                (status, reason, "application/json", body.to_string())
            }
        },
        Route::Register => match register(&mut stream, &state, peer).await {
            Ok(ticket) => (
                200,
                "OK",
                "application/json",
                format!("{{\"ticket\":\"{ticket}\"}}"),
            ),
            Err(status) => {
                let (reason, body) = match status {
                    409 => ("Conflict", CONFLICT),
                    429 => ("Too Many Requests", RATE_LIMITED),
                    _ => ("Bad Request", BAD_REQUEST),
                };
                (status, reason, "application/json", body.to_string())
            }
        },
        Route::Health => {
            let _ = drain_head(&mut stream).await;
            // Report the stored message count so a load balancer or operator
            // can see the DB is alive and roughly how much history is retained.
            let db = state.db.clone();
            let count = tokio::task::spawn_blocking(move || db.message_count())
                .await
                .ok()
                .and_then(|r| r.ok())
                .unwrap_or(-1);
            (
                200,
                "OK",
                "application/json",
                format!("{{\"status\":\"ok\",\"messages\":{count}}}"),
            )
        }
        // `Bridge` never reaches here; a caller that passes it gets 404 rather
        // than a hang.
        _ => {
            let _ = drain_head(&mut stream).await;
            (
                404,
                "Not Found",
                "text/plain; charset=utf-8",
                NOT_FOUND.to_string(),
            )
        }
    };
    // Security headers on every response. The CSP is only meaningful for the
    // HTML page, but sending it everywhere is harmless and keeps this in one
    // place. `default-src 'self'` blocks injected scripts; `connect-src` allows
    // ws/wss to the page's own origin (the bridge) plus http(s) for /auth and
    // /register when the server field points elsewhere.
    let csp = "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; \
               connect-src 'self' ws: wss: http: https:; img-src 'self' data:; base-uri 'none'; \
               form-action 'self'";
    let response = format!(
        "HTTP/1.1 {status} {reason}\r\n\
         Content-Type: {content_type}\r\n\
         Content-Length: {}\r\n\
         Cache-Control: no-store\r\n\
         X-Content-Type-Options: nosniff\r\n\
         Content-Security-Policy: {csp}\r\n\
         Referrer-Policy: no-referrer\r\n\
         Connection: close\r\n\
         \r\n\
         {body}",
        body.len()
    );
    stream.write_all(response.as_bytes()).await?;
    stream.flush().await?;
    Ok(())
}

/// Trade a `POST /auth` login for a single-use bridge ticket.
///
/// `Err(status)` is the HTTP status to answer with when the request is
/// malformed (400), the credentials are wrong (401), or the peer is sending
/// too many attempts (429). The password is checked against the same argon2
/// path the IRC `LOGIN` command uses, so the web client cannot be used to
/// bypass that cost or its timing.
async fn authenticate<S>(
    stream: &mut S,
    state: &SharedState,
    peer: std::net::IpAddr,
) -> Result<String, u16>
where
    S: AsyncRead + Unpin,
{
    // Throttle before any parsing or Argon2 work: a credential-stuffing loop
    // should cost the attacker, not us.
    if !state.allow_auth_attempt(peer).await {
        return Err(429);
    }
    let request = read_request(stream).await.map_err(|_| 400u16)?;
    let head_len = find_head_end(&request).ok_or(400u16)?;
    let head = &request[..head_len];
    let body = String::from_utf8_lossy(&request[head_len..]).into_owned();

    // `user` may ride in the query string so the endpoint stays usable with a
    // plain `curl -d password=...`.
    let query = request_line(head)
        .and_then(|(_, target)| target.split_once('?').map(|(_, q)| q.to_string()))
        .unwrap_or_default();
    // The page posts a JSON object; `curl`-style callers post a form. Accept both.
    let user = json_string(&body, "user")
        .or_else(|| form_field(&body, "user"))
        .or_else(|| form_field(&query, "user"))
        .ok_or(400u16)?;
    let password = json_string(&body, "password")
        .or_else(|| form_field(&body, "password"))
        .ok_or(400u16)?;
    if user.is_empty() || password.is_empty() {
        return Err(400);
    }
    // Reject usernames that the IRC path would refuse, so a crafted request
    // cannot register or log in a name that later breaks protocol output.
    if !crate::protocol::valid_username(&user) {
        return Err(400);
    }

    match state.verify_credentials(&user, &password).await {
        Some(account) => Ok(state.mint_ticket(&account.username).await),
        None => Err(401),
    }
}

/// Create an account via `POST /register` and return a ticket for it.
///
/// Mirrors [`authenticate`]: accepts the same JSON or form body, but registers
/// the account first. `Err(status)` is 400 for a malformed request or 409 when
/// the username is already taken, or 429 when the peer is sending too many
/// attempts. On success the caller is logged straight in, so the page can
/// register and connect in one step.
async fn register<S>(
    stream: &mut S,
    state: &SharedState,
    peer: std::net::IpAddr,
) -> Result<String, u16>
where
    S: AsyncRead + Unpin,
{
    if !state.allow_auth_attempt(peer).await {
        return Err(429);
    }
    let request = read_request(stream).await.map_err(|_| 400u16)?;
    let head_len = find_head_end(&request).ok_or(400u16)?;
    let head = &request[..head_len];
    let body = String::from_utf8_lossy(&request[head_len..]).into_owned();

    let query = request_line(head)
        .and_then(|(_, target)| target.split_once('?').map(|(_, q)| q.to_string()))
        .unwrap_or_default();
    let user = json_string(&body, "user")
        .or_else(|| form_field(&body, "user"))
        .or_else(|| form_field(&query, "user"))
        .ok_or(400u16)?;
    let password = json_string(&body, "password")
        .or_else(|| form_field(&body, "password"))
        .ok_or(400u16)?;
    if user.is_empty() || password.is_empty() {
        return Err(400);
    }
    // Reject usernames the IRC path would refuse, so a crafted request cannot
    // create an account that later breaks protocol output.
    if !crate::protocol::valid_username(&user) {
        return Err(400);
    }

    match state.register_account(&user, &password).await {
        Some(account) => Ok(state.mint_ticket(&account.username).await),
        None => Err(409),
    }
}

/// Read a full request head plus whatever body its `Content-Length` declares.
async fn read_request<S>(stream: &mut S) -> std::io::Result<Vec<u8>>
where
    S: AsyncRead + Unpin,
{
    // The head and the body get separate deadlines: a peer that sends the head
    // promptly but then trickles the body should not be able to hold the
    // connection open on the head's clock.
    let head_deadline = tokio::time::Instant::now() + READ_TIMEOUT;
    let mut buf: Vec<u8> = Vec::new();
    let mut chunk = [0u8; 1024];
    let mut head_len = loop {
        if let Some(end) = find_head_end(&buf) {
            break end;
        }
        if buf.len() >= MAX_HEAD {
            // Malformed or over-long head: answer with what we have.
            return Ok(buf);
        }
        match tokio::time::timeout_at(head_deadline, stream.read(&mut chunk)).await {
            Ok(Ok(0)) => return Ok(buf),
            Ok(Ok(n)) => buf.extend_from_slice(&chunk[..n]),
            Ok(Err(e)) => return Err(e),
            Err(_) => return Ok(buf),
        }
    };

    let want = content_length(&buf[..head_len]).min(MAX_BODY);
    // A head that grew past the cap above may already include the body.
    head_len = head_len.min(buf.len());
    // The body gets its own fresh deadline so a slow body cannot ride the
    // head's clock.
    let body_deadline = tokio::time::Instant::now() + READ_TIMEOUT;
    while buf.len() < head_len + want {
        match tokio::time::timeout_at(body_deadline, stream.read(&mut chunk)).await {
            Ok(Ok(0)) => break,
            Ok(Ok(n)) => buf.extend_from_slice(&chunk[..n]),
            Ok(Err(e)) => return Err(e),
            Err(_) => break,
        }
    }
    Ok(buf)
}

/// Index just past the end of the request head, if it has fully arrived.
fn find_head_end(buf: &[u8]) -> Option<usize> {
    // CRLF is what browsers send; bare LF keeps `curl --data-binary`-style and
    // hand-written test requests working.
    let crlf = buf.windows(4).position(|w| w == b"\r\n\r\n").map(|i| i + 4);
    let lf = buf
        .windows(2)
        .position(|w| w == b"\n\n")
        .map(|i| i + 2)
        // A bare-LF head still ends with the CRLF blank line's LF pair.
        .or_else(|| buf.windows(3).position(|w| w == b"\r\n\n").map(|i| i + 3));
    match (crlf, lf) {
        (Some(a), Some(b)) => Some(a.min(b)),
        (Some(a), None) => Some(a),
        (None, Some(b)) => Some(b),
        (None, None) => None,
    }
}

/// The `Content-Length` declared by a request head, in bytes.
fn content_length(head: &[u8]) -> usize {
    head.split(|b| *b == b'\n')
        .find_map(|line| {
            let line = line.strip_suffix(b"\r").unwrap_or(line);
            let colon = line.iter().position(|b| *b == b':')?;
            let name = std::str::from_utf8(&line[..colon]).ok()?;
            let value = std::str::from_utf8(&line[colon + 1..]).ok()?;
            name.trim()
                .eq_ignore_ascii_case("content-length")
                .then(|| value.trim().parse::<usize>().ok())
                .flatten()
        })
        .unwrap_or(0)
        .min(MAX_BODY)
}

/// Value of `key` in an `application/x-www-form-urlencoded` string or query.
fn form_field(input: &str, key: &str) -> Option<String> {
    input.split('&').find_map(|pair| {
        let (name, value) = pair.split_once('=')?;
        // `+` is form encoding for a space, which `url_decode` leaves alone.
        let name = crate::ws::url_decode(name);
        if name != key {
            return None;
        }
        Some(crate::ws::url_decode(&value.replace('+', " ")))
    })
}

/// Value of `"key"` in a flat JSON object, unescaped.
///
/// Deliberately not a general JSON parser: `POST /auth` takes a flat object of
/// two string fields, and the browser controls the encoding. Anything it cannot
/// match returns `None`, which the caller answers as a 400.
fn json_string(input: &str, key: &str) -> Option<String> {
    let needle = format!("\"{key}\"");
    let mut rest = input;
    loop {
        let at = rest.find(&needle)? + needle.len();
        let after = rest[at..].trim_start();
        let Some(after) = after.strip_prefix(':') else {
            // A same-named value elsewhere (or a nested key): keep looking.
            rest = &rest[at..];
            continue;
        };
        let after = after.trim_start();
        let after = after.strip_prefix('"')?;
        let mut out = String::new();
        let mut chars = after.chars();
        while let Some(c) = chars.next() {
            match c {
                '"' => return Some(out),
                '\\' => match chars.next() {
                    Some('n') => out.push('\n'),
                    Some('t') => out.push('\t'),
                    Some('r') => out.push('\r'),
                    Some('u') => {
                        let hex: String = chars.by_ref().take(4).collect();
                        let code = u32::from_str_radix(&hex, 16).ok()?;
                        out.push(char::from_u32(code)?);
                    }
                    // Covers `\"`, `\\`, and `\/`.
                    Some(other) => out.push(other),
                    None => return None,
                },
                _ => out.push(c),
            }
        }
        return None;
    }
}

/// Read up to and including the end of the request head, discarding it.
async fn drain_head<S>(stream: &mut S) -> std::io::Result<()>
where
    S: AsyncRead + Unpin,
{
    let deadline = tokio::time::Instant::now() + READ_TIMEOUT;
    let mut seen = 0usize;
    // Just the tail of what we have read, to spot the head terminator without
    // holding the whole request in memory.
    let mut tail = Vec::with_capacity(8);
    let mut buf = [0u8; 1024];
    loop {
        match tokio::time::timeout_at(deadline, stream.read(&mut buf)).await {
            Ok(Ok(0)) => return Ok(()),
            Ok(Ok(n)) => {
                tail.extend_from_slice(&buf[..n]);
                if tail.ends_with(b"\r\n\r\n") || tail.ends_with(b"\n\n") {
                    return Ok(());
                }
                // Keep only the last 3 bytes: the longest terminator prefix.
                let keep = tail.len().saturating_sub(3);
                tail.drain(..keep);
                seen += n;
                if seen >= MAX_HEAD {
                    return Ok(());
                }
            }
            Ok(Err(e)) => return Err(e),
            // The peer is not finishing its request; answer anyway.
            Err(_) => return Ok(()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn routes_index_paths() {
        assert_eq!(route(b"GET / HTTP/1.1\r\n"), Route::Index);
        assert_eq!(route(b"GET /index.html HTTP/1.1\r\n"), Route::Index);
    }

    #[test]
    fn routes_bridge_paths_with_query() {
        let head = b"GET /ws/%23general?ticket=abc123 HTTP/1.1\r\n";
        assert_eq!(route(head), Route::Bridge);
    }

    #[test]
    fn routes_unknown_paths_to_not_found() {
        assert_eq!(route(b"GET /favicon.ico HTTP/1.1\r\n"), Route::NotFound);
        assert_eq!(route(b"POST / HTTP/1.1\r\n"), Route::NotFound);
        // A bare prefix of a request line is not routable yet.
        assert_eq!(route(b"GET /w"), Route::NotFound);
        assert_eq!(route(b""), Route::NotFound);
    }

    #[test]
    fn routes_auth_post_only() {
        assert_eq!(route(b"POST /auth HTTP/1.1\r\n"), Route::Auth);
        // A GET on /auth is not a login, and must not mint a ticket.
        assert_eq!(route(b"GET /auth HTTP/1.1\r\n"), Route::NotFound);
    }

    #[test]
    fn extracts_json_string_fields() {
        let body = r#"{"user":"alice","password":"p\"w"}"#;
        assert_eq!(json_string(body, "user").as_deref(), Some("alice"));
        assert_eq!(json_string(body, "password").as_deref(), Some("p\"w"));
        assert_eq!(json_string(body, "missing"), None);
    }

    #[test]
    fn extracts_target_from_request_line() {
        assert_eq!(request_target("GET /a?b=c HTTP/1.1\r\n"), Some("/a?b=c"));
        assert_eq!(request_target("GET / HTTP/1.0\n"), Some("/"));
        assert_eq!(request_target("nonsense"), None);
    }

    /// The peek→replay path must preserve non-UTF-8 bytes verbatim. A request
    /// head can carry raw high bytes (e.g. a percent-encoded channel name that
    /// decodes to a non-ASCII byte), and the bridge replays the peeked head into
    /// the winning branch. If peek decoded through `from_utf8_lossy` and
    /// re-encoded, those bytes would be corrupted before tungstenite or the HTTP
    /// handler saw them.
    #[tokio::test]
    async fn peek_head_preserves_non_utf8_bytes() {
        // A request line whose target contains a raw 0xFF byte (invalid UTF-8).
        let raw: Vec<u8> = b"GET /ws/%23general?user=al\xFFce HTTP/1.1\r\n".to_vec();
        let mut stream = std::io::Cursor::new(raw.clone());
        let peeked = peek_head(&mut stream).await;
        assert_eq!(peeked, raw, "peeked head must be byte-identical");

        // Replaying through PrefixedStream must also hand back the same bytes.
        let mut replayed = PrefixedStream::new(peeked, std::io::Cursor::new(Vec::new()));
        let mut out = Vec::new();
        tokio::io::AsyncReadExt::read_to_end(&mut replayed, &mut out)
            .await
            .unwrap();
        assert_eq!(out, raw, "replayed head must be byte-identical");
    }

    /// A request head that arrives in several TCP segments must be reassembled
    /// before routing. Regression test: `peek_head` used to return only the
    /// first read, so a request line split across segments (common on a slow
    /// link or under Nagle) was routed as `NotFound` and the bridge upgrade was
    /// dropped.
    #[tokio::test]
    async fn peek_head_reassembles_a_split_request_line() {
        // A duplex hands out small chunks, so the head is delivered in pieces.
        let (mut client, server) = tokio::io::duplex(4);
        let writer = tokio::spawn(async move {
            client
                .write_all(b"GET /ws/%23general?ticket=abc HTTP/1.1\r\n")
                .await
                .unwrap();
        });
        let mut server = server;
        let peeked = peek_head(&mut server).await;
        writer.await.unwrap();
        assert_eq!(
            peeked,
            b"GET /ws/%23general?ticket=abc HTTP/1.1\r\n".to_vec(),
            "split request head must be reassembled in full"
        );
        // And it must route as a bridge upgrade, not NotFound.
        assert_eq!(route(&peeked), Route::Bridge);
    }

    /// `content_length` must read the header from raw bytes without a lossy
    /// decode, so a non-UTF-8 body or header does not corrupt the count.
    #[test]
    fn content_length_reads_from_raw_bytes() {
        let head = b"POST /auth HTTP/1.1\r\nContent-Length: 12\r\n\r\n";
        assert_eq!(content_length(head), 12);
        // A non-UTF-8 byte elsewhere in the head must not break the parse.
        let head = b"POST /auth HTTP/1.1\r\nX-Garbage: \xFF\r\nContent-Length: 3\r\n\r\n";
        assert_eq!(content_length(head), 3);
        // Missing header defaults to 0.
        assert_eq!(content_length(b"GET / HTTP/1.1\r\n\r\n"), 0);
    }

    /// The page must actually be embedded, and point at the bridge it is served
    /// from — otherwise the UI loads but can never connect.
    #[test]
    fn embedded_page_is_a_self_contained_client() {
        assert!(INDEX_HTML.starts_with("<!DOCTYPE html>"));
        assert!(INDEX_HTML.contains("/ws/"));
        assert!(INDEX_HTML.contains("new WebSocket"));
        // No external assets: the page must work offline on a bare host, so it
        // must not pull scripts, styles, or fonts from anywhere else. (A bare
        // scheme string is fine: the page derives `ws://` vs `wss://` from
        // `location.protocol` rather than hardcoding a host.)
        assert!(!INDEX_HTML.contains("src=\"http"), "external script/image");
        assert!(!INDEX_HTML.contains("href=\"http"), "external stylesheet");
        assert!(!INDEX_HTML.contains("src=\"//"), "protocol-relative asset");
        assert!(!INDEX_HTML.contains("href=\"//"), "protocol-relative asset");
        assert!(!INDEX_HTML.contains("@import"), "external stylesheet");
        assert!(
            INDEX_HTML.contains("location.host"),
            "ws URL must be same-origin"
        );
    }

    #[test]
    fn embedded_page_uses_tickets_and_reconnects() {
        // Credentials go in a POST body, never the websocket URL.
        assert!(INDEX_HTML.contains("/auth"));
        assert!(INDEX_HTML.contains("ticket"));
        assert!(
            !INDEX_HTML.contains("password="),
            "password leaked into a URL"
        );
        // A dropped bridge reconnects with backoff instead of going dead.
        assert!(INDEX_HTML.contains("onclose"));
        assert!(INDEX_HTML.contains("setTimeout"));
        // Past messages are replayed so a reload is not an empty log.
        assert!(INDEX_HTML.contains("history"));
    }
}
