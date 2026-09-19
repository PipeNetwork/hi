use std::io::{BufRead, BufReader, ErrorKind, Read, Write};
use std::net::TcpStream;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

/// Boots the real chat server binary on an ephemeral port and provides a
/// helper to open client connections against it.
struct TestServer {
    child: Child,
    addr: String,
    ws_addr: String,
    _db_dir: tempfile::TempDir,
    _stdout_drain: std::thread::JoinHandle<()>,
}

impl TestServer {
    fn start() -> Self {
        Self::start_with(&[])
    }

    fn start_with(extra_env: &[(&str, &str)]) -> Self {
        let db_dir = tempfile::tempdir().expect("tempdir");
        let db_path = db_dir.path().join("chat.db");

        let mut cmd = Command::new(env!("CARGO_BIN_EXE_chat"));
        cmd.env("CHAT_ADDR", "127.0.0.1:0")
            .env("CHAT_DB", &db_path)
            .env("CHAT_WS_ADDR", "127.0.0.1:0")
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit());
        for (key, value) in extra_env {
            cmd.env(key, value);
        }
        let mut child = cmd.spawn().expect("spawn chat server");

        let stdout = child.stdout.take().expect("child stdout");
        let (addr, ws_addr, stdout_drain) = read_listening_addrs(stdout);
        let server = TestServer {
            child,
            addr,
            ws_addr,
            _db_dir: db_dir,
            _stdout_drain: stdout_drain,
        };
        server.wait_ready();
        server
    }

    fn wait_ready(&self) {
        let deadline = Instant::now() + Duration::from_secs(5);
        while Instant::now() < deadline {
            if TcpStream::connect(&self.addr).is_ok() {
                return;
            }
            std::thread::sleep(Duration::from_millis(50));
        }
        panic!("server did not become ready at {}", self.addr);
    }

    fn connect(&self) -> TestClient {
        let stream = TcpStream::connect(&self.addr).expect("connect to server");
        stream
            .set_read_timeout(Some(Duration::from_secs(5)))
            .expect("set read timeout");
        TestClient {
            reader: BufReader::new(stream),
        }
    }
}

impl Drop for TestServer {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// Read the server's `chat server listening on <addr>` and
/// `websocket bridge listening on <addr>` lines, then keep draining stdout
/// until the child exits. Stopping after the first line closes the pipe and
/// Rust's `println!` panics on Broken pipe, which kills the server before it
/// accepts clients.
fn read_listening_addrs(
    stdout: impl Read + Send + 'static,
) -> (String, String, std::thread::JoinHandle<()>) {
    let (tx, rx) = std::sync::mpsc::channel();
    let handle = std::thread::spawn(move || {
        let mut reader = BufReader::new(stdout);
        let mut line = String::new();
        loop {
            line.clear();
            if reader.read_line(&mut line).is_err() || line.is_empty() {
                break;
            }
            let tagged = line
                .strip_prefix("chat server listening on ")
                .map(|rest| ("tcp", rest))
                .or_else(|| {
                    line.strip_prefix("websocket bridge listening on ")
                        .map(|rest| ("ws", rest))
                });
            if let Some((kind, rest)) = tagged {
                let addr = rest.split_whitespace().next().unwrap_or("").to_string();
                let _ = tx.send((kind, addr));
            }
        }
    });

    let (mut tcp, mut ws) = (None, None);
    while tcp.is_none() || ws.is_none() {
        match rx.recv_timeout(Duration::from_secs(10)) {
            Ok(("tcp", addr)) => tcp = Some(addr),
            Ok(("ws", addr)) => ws = Some(addr),
            Ok(_) => {}
            Err(_) => panic!("server did not print a listening address"),
        }
    }
    (tcp.unwrap(), ws.unwrap(), handle)
}

/// A single client connection with line-oriented send/read helpers.
struct TestClient {
    reader: BufReader<TcpStream>,
}

impl TestClient {
    fn send(&mut self, line: &str) {
        let stream = self.reader.get_mut();
        writeln!(stream, "{line}").expect("write to server");
        stream.flush().expect("flush");
    }

    fn read_line(&mut self) -> String {
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            let mut buf = String::new();
            match self.reader.read_line(&mut buf) {
                Ok(_) => return buf,
                Err(err)
                    if matches!(
                        err.kind(),
                        ErrorKind::WouldBlock | ErrorKind::TimedOut | ErrorKind::Interrupted
                    ) =>
                {
                    if Instant::now() >= deadline {
                        panic!("read line from server: {err}");
                    }
                    std::thread::sleep(Duration::from_millis(20));
                }
                Err(err) => panic!("read line from server: {err}"),
            }
        }
    }

    /// Read lines until one starts with `prefix`, returning it.
    fn read_until(&mut self, prefix: &str) -> String {
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            if Instant::now() >= deadline {
                panic!("timeout waiting for line starting with {prefix:?}");
            }
            let line = self.read_line();
            if line.starts_with(prefix) {
                return line;
            }
        }
    }

    /// Collect everything the server sends within `window` and assert none of
    /// it contains `needle`. Unlike a plain silence check this tolerates
    /// unrelated lines (other broadcasts) arriving during the window, which
    /// keeps the assertion deterministic instead of racy.
    fn assert_nothing_matching(&mut self, needle: &str, window: Duration) {
        // Anything already pulled in by read-ahead counts as received too.
        let mut seen = String::from_utf8_lossy(self.reader.buffer()).into_owned();

        let stream = self.reader.get_mut();
        stream.set_nonblocking(true).expect("nonblocking mode");
        let deadline = Instant::now() + window;
        while Instant::now() < deadline {
            let mut buf = [0u8; 512];
            match stream.read(&mut buf) {
                Ok(0) => break,
                Ok(n) => seen.push_str(&String::from_utf8_lossy(&buf[..n])),
                Err(e) if e.kind() == ErrorKind::WouldBlock => {
                    std::thread::sleep(Duration::from_millis(20));
                }
                Err(e) => {
                    stream.set_nonblocking(false).expect("restore blocking");
                    panic!("read while asserting silence: {e}");
                }
            }
        }
        stream.set_nonblocking(false).expect("restore blocking");

        assert!(
            !seen.contains(needle),
            "unexpected {needle:?} received: {seen:?}"
        );
    }

    /// Accumulate everything the server sends within `window` (including any
    /// buffered read-ahead) and return it. Bounded, so a message that never
    /// arrives fails the test instead of hanging it.
    fn read_for(&mut self, window: Duration) -> String {
        let mut seen = String::from_utf8_lossy(self.reader.buffer()).into_owned();

        let stream = self.reader.get_mut();
        stream.set_nonblocking(true).expect("nonblocking mode");
        let deadline = Instant::now() + window;
        while Instant::now() < deadline {
            let mut buf = [0u8; 512];
            match stream.read(&mut buf) {
                Ok(0) => break,
                Ok(n) => seen.push_str(&String::from_utf8_lossy(&buf[..n])),
                Err(e) if e.kind() == ErrorKind::WouldBlock => {
                    std::thread::sleep(Duration::from_millis(20));
                }
                Err(e) => {
                    stream.set_nonblocking(false).expect("restore blocking");
                    panic!("read_for: {e}");
                }
            }
        }
        stream.set_nonblocking(false).expect("restore blocking");

        seen
    }
}

#[test]
fn register_join_privmsg_history_roundtrip() {
    let server = TestServer::start();

    let mut alice = server.connect();
    alice.read_until("Welcome");

    alice.send("REGISTER alice secret");
    assert!(alice.read_until("OK registered").contains("alice"));

    alice.send("JOIN #general");
    assert!(alice.read_until("OK joined").contains("#general"));

    // Alice sends a message; it is broadcast to all members (including the
    // sender), with no per-sender ack.
    alice.send("PRIVMSG #general :hello bob");
    let got = alice.read_until("PRIVMSG #general");
    assert!(
        got.contains("alice") && got.contains("hello bob"),
        "got: {got}"
    );

    // History is a header line plus indented body lines.
    alice.send("HISTORY #general");
    let header = alice.read_until("HISTORY");
    assert!(header.contains("#general"), "history header: {header}");
    let body = alice.read_line();
    assert!(body.contains("hello bob"), "history body: {body}");
}

#[test]
fn kick_removes_member_and_broadcasts() {
    let server = TestServer::start();

    let mut alice = server.connect();
    alice.read_until("Welcome");
    alice.send("REGISTER alice secret");
    alice.read_until("OK registered");
    alice.send("JOIN #general");
    alice.read_until("OK joined");

    let mut bob = server.connect();
    bob.read_until("Welcome");
    bob.send("REGISTER bob secret");
    bob.read_until("OK registered");
    bob.send("JOIN #general");
    bob.read_until("OK joined");

    // Alice (channel creator) kicks Bob.
    alice.send("KICK #general bob");
    assert!(alice.read_until("OK kicked").contains("bob"));

    // Bob's connection is told it was kicked.
    assert!(bob.read_until("KICKED").contains("#general"));

    // A message after the kick is not delivered to Bob.
    alice.send("PRIVMSG #general :after kick");
    let got = alice.read_until("PRIVMSG #general");
    assert!(got.contains("after kick"), "got: {got}");

    // Bob should not receive the post-kick message. Give it a moment, then
    // confirm nothing arrived. A read-timeout on an empty socket surfaces as
    // WouldBlock rather than an empty buffer, so peek non-blocking.
    std::thread::sleep(Duration::from_millis(300));
    let leftover = bob.reader.buffer();
    assert!(
        leftover.is_empty(),
        "bob received data after being kicked: {:?}",
        String::from_utf8_lossy(leftover)
    );
    let stream = bob.reader.get_mut();
    stream.set_nonblocking(true).expect("nonblocking peek");
    let mut buf = [0u8; 256];
    match stream.peek(&mut buf) {
        Ok(0) => {}
        Ok(n) => panic!(
            "bob received data after being kicked: {:?}",
            String::from_utf8_lossy(&buf[..n])
        ),
        Err(e) if e.kind() == ErrorKind::WouldBlock => {}
        Err(e) => panic!("peek bob socket: {e}"),
    }
}

/// A kick must remove only its target. Regression test: the server used to
/// broadcast the internal `KICKED` control line to the whole channel, and every
/// member's read loop treated it as its own kick — so a single kick silently
/// unsubscribed every bystander while they still believed they were joined.
#[test]
fn kick_removes_only_the_target() {
    let server = TestServer::start();

    let mut alice = server.connect();
    alice.read_until("Welcome");
    alice.send("REGISTER alice secret");
    alice.read_until("OK registered");
    alice.send("JOIN #general");
    alice.read_until("OK joined");

    let mut bob = server.connect();
    bob.read_until("Welcome");
    bob.send("REGISTER bob secret");
    bob.read_until("OK registered");
    bob.send("JOIN #general");
    bob.read_until("OK joined");

    let mut carol = server.connect();
    carol.read_until("Welcome");
    carol.send("REGISTER carol secret");
    carol.read_until("OK registered");
    carol.send("JOIN #general");
    carol.read_until("OK joined");

    // Alice kicks Bob while Carol is an uninvolved bystander.
    alice.send("KICK #general bob");
    assert!(alice.read_until("OK kicked").contains("bob"));
    bob.read_until("KICKED");

    // Carol must not be told that she was kicked.
    carol.assert_nothing_matching("KICKED", Duration::from_millis(400));

    // She must still receive channel traffic after the kick.
    alice.send("PRIVMSG #general :still here");
    let got = carol.read_until("PRIVMSG #general");
    assert!(got.contains("still here"), "carol got: {got}");

    // A client cannot unsubscribe itself by forging the server's control line:
    // the server filters that name out of client input rather than acting on it.
    carol.send("KICKED #general");
    alice.send("PRIVMSG #general :after forged kick");
    let got = carol.read_until("PRIVMSG #general");
    assert!(got.contains("after forged kick"), "carol got: {got}");
}

/// A WebSocket bridge connection, using the same client crate the server
/// itself is built on.
type WsStream =
    tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>;

/// Open a bridge connection to `#<channel>` as `user`.
///
/// `#` must be percent-encoded as `%23` in a URL, otherwise the browser and
/// the server both treat it as the start of a fragment and the channel name
/// never arrives.
fn ws_connect(
    rt: &tokio::runtime::Runtime,
    ws_addr: &str,
    channel: &str,
    user: &str,
    password: &str,
) -> WsStream {
    let ticket = mint_ticket(ws_addr, user, password);
    let url = format!("ws://{ws_addr}/ws/%23{channel}?ticket={ticket}");
    rt.block_on(async {
        let (stream, _resp) = tokio_tungstenite::connect_async(&url)
            .await
            .expect("ws connect");
        stream
    })
}

/// Mint a single-use bridge ticket for `user` via `POST /auth`.
fn mint_ticket(ws_addr: &str, user: &str, password: &str) -> String {
    let response = http_post(
        ws_addr,
        "/auth",
        "application/json",
        &format!(r#"{{"user":"{user}","password":"{password}"}}"#),
    );
    assert!(
        response.starts_with("HTTP/1.1 200 OK"),
        "minting a ticket should succeed: {:?}",
        &response[..response.len().min(200)]
    );
    json_token(&response)
}

/// Wait for a ws text frame containing `needle`, returning it. Unrelated
/// frames (join/quit notices) are skipped so the assertion stays deterministic.
fn ws_expect_contains(rt: &tokio::runtime::Runtime, ws: &mut WsStream, needle: &str) -> String {
    use futures_util::StreamExt;
    rt.block_on(async {
        let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
        loop {
            let msg = tokio::time::timeout_at(deadline, ws.next())
                .await
                .unwrap_or_else(|_| {
                    panic!("timed out waiting for a ws frame containing {needle:?}")
                })
                .expect("ws stream ended")
                .expect("ws error");
            if let Ok(text) = msg.into_text()
                && text.contains(needle)
            {
                return text;
            }
        }
    })
}

/// Collect whatever the bridge sends within `dur`. Unlike `ws_expect_contains`
/// this never waits for a specific line, so it can assert on *absence* of
/// traffic (e.g. that a rejected session receives no broadcasts).
fn ws_drain_for(rt: &tokio::runtime::Runtime, ws: &mut WsStream, dur: Duration) -> String {
    use futures_util::StreamExt;
    rt.block_on(async {
        let deadline = tokio::time::Instant::now() + dur;
        let mut seen = String::new();
        loop {
            let left = deadline.saturating_duration_since(tokio::time::Instant::now());
            if left.is_zero() {
                return seen;
            }
            match tokio::time::timeout(left, ws.next()).await {
                Ok(Some(Ok(msg))) => {
                    if let Ok(text) = msg.into_text() {
                        seen.push_str(&text);
                    }
                }
                Ok(Some(Err(_))) | Ok(None) => return seen,
                Err(_) => return seen,
            }
        }
    })
}

/// Send a text frame over the bridge.
fn ws_send(rt: &tokio::runtime::Runtime, ws: &mut WsStream, text: &str) {
    use futures_util::SinkExt;
    rt.block_on(async {
        ws.send(tokio_tungstenite::tungstenite::Message::text(text))
            .await
            .expect("ws send");
    })
}

/// Block until the bridge is provably subscribed, by having `peer` keep talking
/// until the bridge observes a broadcast.
///
/// The server subscribes *after* completing the handshake, so `ws_connect`
/// returning does not mean the subscription is installed. A client that kills
/// its TCP session immediately after connecting can therefore race ahead of the
/// server and miss the resulting broadcast. Retrying makes that deterministic.
fn ws_wait_subscribed(
    rt: &tokio::runtime::Runtime,
    ws: &mut WsStream,
    peer: &mut TestClient,
) -> String {
    use futures_util::StreamExt;
    rt.block_on(async {
        let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
        let mut n = 0u32;
        loop {
            assert!(
                tokio::time::Instant::now() < deadline,
                "bridge never observed a channel broadcast; subscription looks broken"
            );
            n += 1;
            peer.send(&format!("PRIVMSG #general :sync {n}"));
            match tokio::time::timeout(Duration::from_millis(200), ws.next()).await {
                Ok(Some(Ok(msg))) => {
                    if let Ok(text) = msg.into_text() {
                        // Join/part notices can arrive first; only the sync
                        // message proves this connection is subscribed.
                        if text.contains("sync ") {
                            return text;
                        }
                    }
                }
                Ok(Some(Err(e))) => panic!("ws error while syncing: {e}"),
                Ok(None) => panic!("ws stream ended while syncing"),
                // Nothing yet: the subscription has probably not landed, retry.
                Err(_) => continue,
            }
        }
    })
}

/// The bridge must be usable on its own: a browser client has no IRC session,
/// so requiring prior channel membership made the WebSocket endpoint dead for
/// its only real caller.
#[test]
fn ws_bridge_receives_broadcasts_without_a_tcp_session() {
    let server = TestServer::start();

    // Registration is a TCP-only command; the bridge only needs the account.
    let mut setup = server.connect();
    setup.read_until("Welcome");
    setup.send("REGISTER alice secret");
    setup.read_until("OK registered");
    drop(setup);

    let rt = tokio::runtime::Runtime::new().expect("runtime");
    let mut ws = ws_connect(&rt, &server.ws_addr, "general", "alice", "secret");

    // A different user on TCP joins and speaks; alice never joined over TCP.
    let mut bob = server.connect();
    bob.read_until("Welcome");
    bob.send("REGISTER bob secret");
    bob.read_until("OK registered");
    bob.send("JOIN #general");
    bob.read_until("OK joined");

    // The bridge subscribes after the handshake, so wait until a broadcast
    // actually arrives before asserting on a specific message.
    ws_wait_subscribed(&rt, &mut ws, &mut bob);

    bob.send("PRIVMSG #general :hello over ws");

    let got = ws_expect_contains(&rt, &mut ws, "hello over ws");
    assert!(got.contains("bob"), "ws client got: {got}");
}

/// A TCP disconnect must not strip membership that another live connection
/// for the same user still depends on. Without the guard the bridge goes deaf
/// the moment the user's IRC session ends.
#[test]
fn ws_bridge_keeps_membership_after_tcp_disconnect() {
    let server = TestServer::start();

    let mut alice = server.connect();
    alice.read_until("Welcome");
    alice.send("REGISTER alice secret");
    alice.read_until("OK registered");
    alice.send("JOIN #general");
    alice.read_until("OK joined");

    let rt = tokio::runtime::Runtime::new().expect("runtime");
    let mut ws = ws_connect(&rt, &server.ws_addr, "general", "alice", "secret");

    // A second user gives the bridge a reason to receive channel traffic.
    let mut bob = server.connect();
    bob.read_until("Welcome");
    bob.send("REGISTER bob secret");
    bob.read_until("OK registered");
    bob.send("JOIN #general");
    bob.read_until("OK joined");

    // Prove the subscription is installed before ending the TCP session, so
    // the QUIT broadcast below cannot be missed.
    ws_wait_subscribed(&rt, &mut ws, &mut bob);

    // Alice's IRC session ends while her bridge stays connected.
    alice.send("QUIT");
    drop(alice);

    let quit = ws_expect_contains(&rt, &mut ws, "QUIT #general");
    assert!(quit.contains("alice"), "bridge got: {quit}");

    // The bridge must still be a working member: if the disconnect had stripped
    // the membership, the send loop would have closed the socket.
    ws_send(&rt, &mut ws, "still here over the bridge");
    let seen = bob.read_for(Duration::from_secs(5));
    assert!(
        seen.contains("still here over the bridge"),
        "bridge message never reached bob: {seen:?}"
    );
}

/// Fetch `target` from the bridge port over plain HTTP, returning the raw
/// response as text. `Connection: close` keeps this simple: the server closes
/// when it has answered, so reading to EOF gives the whole message.
fn http_get(ws_addr: &str, target: &str) -> String {
    let mut stream = TcpStream::connect(ws_addr).expect("connect to bridge port");
    stream
        .set_read_timeout(Some(Duration::from_secs(5)))
        .expect("set read timeout");
    let request = format!("GET {target} HTTP/1.1\r\nHost: {ws_addr}\r\nConnection: close\r\n\r\n");
    stream.write_all(request.as_bytes()).expect("write request");
    let mut response = String::new();
    stream.read_to_string(&mut response).expect("read response");
    response
}

/// The page has to be served from the same origin as the bridge: that is what
/// lets the client derive the bridge URL from `location.host` instead of being
/// configured, and it keeps the server a single port and single binary.
#[test]
fn serves_web_ui_on_the_bridge_port() {
    let server = TestServer::start();

    let response = http_get(&server.ws_addr, "/");
    assert!(
        response.starts_with("HTTP/1.1 200 OK"),
        "unexpected status: {:?}",
        &response[..response.len().min(120)]
    );
    assert!(
        response.contains("Content-Type: text/html"),
        "page served with the wrong content type: {:?}",
        &response[..response.len().min(200)]
    );
    assert!(
        response.contains("<!DOCTYPE html>"),
        "response body is not the embedded page"
    );
    // The page must point at this very bridge, or it loads but cannot connect.
    assert!(
        response.contains("/ws/"),
        "page never mentions the bridge path"
    );

    // A path the server does not own is a plain 404, not a hang and not the page.
    let missing = http_get(&server.ws_addr, "/favicon.ico");
    assert!(
        missing.starts_with("HTTP/1.1 404 Not Found"),
        "unexpected status for a missing path: {:?}",
        &missing[..missing.len().min(120)]
    );
}

/// A client that never sends a newline must not be able to grow the server's
/// memory without bound. The oversize line is rejected, and because the reader
/// drains to the next newline the session stays usable instead of desyncing
/// into the middle of the discarded line.
#[test]
fn oversize_line_is_rejected_and_the_session_survives() {
    let server = TestServer::start();

    let mut alice = server.connect();
    alice.read_until("Welcome");
    alice.send("REGISTER alice secret");
    alice.read_until("OK registered");
    alice.send("JOIN #general");
    alice.read_until("OK joined");

    let mut bob = server.connect();
    bob.read_until("Welcome");
    bob.send("REGISTER bob secret");
    bob.read_until("OK registered");
    bob.send("JOIN #general");
    bob.read_until("OK joined");

    // ~200 KB on a single line: far past the per-line cap.
    alice.send(&"A".repeat(200_000));

    let reply = alice.read_for(Duration::from_secs(5));
    assert!(
        reply.contains("ERROR line too long"),
        "oversize line was not rejected: {reply:?}"
    );

    // The connection must still be aligned on a line boundary: this message has
    // to reach bob intact, not be swallowed as the tail of the discarded line.
    alice.send("PRIVMSG #general :after the flood");
    let seen = bob.read_for(Duration::from_secs(5));
    assert!(
        seen.contains("after the flood") && seen.contains("alice"),
        "session did not resync after an oversize line: {seen:?}"
    );
}

/// Same bound on the WebSocket side: a single huge frame must be refused rather
/// than buffered, and it must only cost the offending connection — the listener
/// (and the TCP server) keep serving.
#[test]
fn oversize_ws_frame_closes_only_that_bridge_connection() {
    use futures_util::{SinkExt, StreamExt};
    use tokio_tungstenite::tungstenite::Message;

    let server = TestServer::start();

    // Accounts are created over TCP; the bridge only consumes them.
    let mut setup = server.connect();
    setup.read_until("Welcome");
    setup.send("REGISTER alice secret");
    setup.read_until("OK registered");
    drop(setup);

    let mut bob = server.connect();
    bob.read_until("Welcome");
    bob.send("REGISTER bob secret");
    bob.read_until("OK registered");
    bob.send("JOIN #general");
    bob.read_until("OK joined");

    let rt = tokio::runtime::Runtime::new().expect("runtime");
    let mut ws = ws_connect(&rt, &server.ws_addr, "general", "alice", "secret");

    // 256 KiB in one frame, well past the bridge's cap.
    rt.block_on(async {
        ws.send(Message::text("A".repeat(256 * 1024)))
            .await
            .expect("client must be able to send an oversize frame");
    });

    // The bridge must give up on this connection instead of holding the frame.
    rt.block_on(async {
        let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
        loop {
            match tokio::time::timeout_at(deadline, ws.next()).await {
                // Closed cleanly, or a protocol error: both mean the frame was
                // refused rather than accepted.
                Ok(None) | Ok(Some(Err(_))) => break,
                Ok(Some(Ok(_))) => continue,
                Err(_) => panic!("bridge kept an oversize frame alive"),
            }
        }
    });

    // A new bridge connection still works end to end. The subscription lands
    // after the handshake, so sync on a broadcast before asserting.
    let mut fresh = ws_connect(&rt, &server.ws_addr, "general", "bob", "secret");
    let got = ws_wait_subscribed(&rt, &mut fresh, &mut bob);
    assert!(
        got.contains("sync") && got.contains("bob"),
        "fresh bridge client got: {got}"
    );
}

/// POST a body to the bridge port and return the raw response.
fn http_post(ws_addr: &str, target: &str, content_type: &str, body: &str) -> String {
    let mut stream = TcpStream::connect(ws_addr).expect("connect");
    stream
        .set_read_timeout(Some(Duration::from_secs(5)))
        .expect("set read timeout");
    let request = format!(
        "POST {target} HTTP/1.1\r\nHost: {ws_addr}\r\nContent-Type: {content_type}\r\n\
         Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    stream.write_all(request.as_bytes()).expect("write request");
    let mut response = String::new();
    stream.read_to_string(&mut response).expect("read response");
    response
}

/// Open a bridge connection on an arbitrary path, so tests can exercise the
/// ticket and backfill query parameters.
fn ws_connect_path(rt: &tokio::runtime::Runtime, ws_addr: &str, path: &str) -> WsStream {
    let url = format!("ws://{ws_addr}{path}");
    rt.block_on(async {
        let (stream, _resp) = tokio_tungstenite::connect_async(&url)
            .await
            .expect("ws connect");
        stream
    })
}

/// Extract `"ticket":"…"` from a JSON body.
fn json_token(response: &str) -> String {
    let after = response
        .split("\"ticket\":\"")
        .nth(1)
        .unwrap_or_else(|| panic!("no ticket in response: {response}"));
    after
        .split('"')
        .next()
        .expect("ticket has a closing quote")
        .to_string()
}

/// A page reload cannot replay history over a publish-only bridge, so the bridge
/// replays the last `history=N` messages as frames right after subscribing.
#[test]
fn bridge_backfills_recent_history_on_connect() {
    let server = TestServer::start();

    let mut bob = server.connect();
    bob.read_until("Welcome");
    bob.send("REGISTER bob secret");
    bob.read_until("OK registered");
    bob.send("JOIN #general");
    bob.read_until("OK joined");
    bob.send("PRIVMSG #general :first");
    bob.send("PRIVMSG #general :second");
    // `read_until` matches a line prefix, so wait for the HISTORY header: once it
    // arrives both PRIVMSGs have been stored (they are handled first).
    bob.send("HISTORY #general 5");
    bob.read_until("HISTORY #general");
    assert!(bob.read_line().contains("first"));
    assert!(bob.read_line().contains("second"));

    let rt = tokio::runtime::Runtime::new().expect("runtime");
    let ticket = mint_ticket(&server.ws_addr, "bob", "secret");
    let mut ws = ws_connect_path(
        &rt,
        &server.ws_addr,
        &format!("/ws/%23general?ticket={ticket}&history=2"),
    );

    // Both replayed lines arrive as ordinary frames, oldest first.
    let first = ws_expect_contains(&rt, &mut ws, "first");
    assert!(
        first.contains("bob"),
        "backfilled line should name its author: {first}"
    );
    ws_expect_contains(&rt, &mut ws, "second");
}

/// The bridge must never replay more than it was asked for.
#[test]
fn bridge_backfill_respects_its_limit() {
    let server = TestServer::start();

    let mut bob = server.connect();
    bob.read_until("Welcome");
    bob.send("REGISTER bob secret");
    bob.read_until("OK registered");
    bob.send("JOIN #general");
    bob.read_until("OK joined");
    bob.send("PRIVMSG #general :backfill-me");
    // The gap becomes visible only after the store, so wait for the history reply
    // (message rows are prefixed with their id, so match the header line).
    bob.send("HISTORY #general 5");
    bob.read_until("HISTORY #general");

    let rt = tokio::runtime::Runtime::new().expect("runtime");
    // `history=0` is the default: connect with no replay at all.
    let ticket = mint_ticket(&server.ws_addr, "bob", "secret");
    let mut ws = ws_connect_path(
        &rt,
        &server.ws_addr,
        &format!("/ws/%23general?ticket={ticket}&history=0"),
    );

    ws_send(&rt, &mut ws, "ping after connect");
    let got = ws_expect_contains(&rt, &mut ws, "ping after connect");
    assert!(
        !got.contains("backfill-me"),
        "history=0 must not replay anything: {got}"
    );
}

/// Tickets let the page log in over HTTP (so the password never rides in the
/// bridge URL) and are single-use, so a leaked URL is not a permanent key.
#[test]
fn auth_ticket_authenticates_the_bridge_exactly_once() {
    let server = TestServer::start();

    let mut alice = server.connect();
    alice.read_until("Welcome");
    alice.send("REGISTER alice secret");
    alice.read_until("OK registered");
    drop(alice);

    let response = http_post(
        &server.ws_addr,
        "/auth",
        "application/json",
        r#"{"user":"alice","password":"secret"}"#,
    );
    assert!(
        response.starts_with("HTTP/1.1 200 OK"),
        "login should succeed: {:?}",
        &response[..response.len().min(200)]
    );
    let token = json_token(&response);
    assert!(!token.is_empty(), "empty token in {response:?}");

    // The ticket alone authenticates: no password in this URL.
    let rt = tokio::runtime::Runtime::new().expect("runtime");
    let mut ws = ws_connect_path(
        &rt,
        &server.ws_addr,
        &format!("/ws/%23general?user=alice&ticket={token}"),
    );
    // A raw peer generates traffic, so the bridge can prove it really subscribed.
    let mut bob = server.connect();
    bob.read_until("Welcome");
    bob.send("REGISTER bob secret");
    bob.read_until("OK registered");
    bob.send("JOIN #general");
    bob.read_until("OK joined");

    let sync = ws_wait_subscribed(&rt, &mut ws, &mut bob);
    assert!(sync.contains("bob"), "ticket client got: {sync}");

    // Redeeming it a second time must not authenticate. The bridge completes the
    // websocket handshake and rejects the session, so assert on traffic: a reused
    // ticket must never observe a channel broadcast.
    let mut reused = ws_connect_path(
        &rt,
        &server.ws_addr,
        &format!("/ws/%23general?user=alice&ticket={token}"),
    );
    let sync = ws_wait_subscribed(&rt, &mut ws, &mut bob);
    assert!(sync.contains("bob"), "first bridge lost its subscription");
    let leaked = ws_drain_for(&rt, &mut reused, Duration::from_millis(500));
    assert!(
        !leaked.contains("sync "),
        "a single-use ticket was accepted twice: {leaked:?}"
    );
}

/// Wrong credentials must not mint a ticket.
#[test]
fn auth_rejects_bad_credentials() {
    let server = TestServer::start();

    let mut alice = server.connect();
    alice.read_until("Welcome");
    alice.send("REGISTER alice secret");
    alice.read_until("OK registered");
    drop(alice);

    let response = http_post(
        &server.ws_addr,
        "/auth",
        "application/json",
        r#"{"user":"alice","password":"wrong"}"#,
    );
    assert!(
        response.starts_with("HTTP/1.1 401"),
        "bad password should be unauthorized: {:?}",
        &response[..response.len().min(200)]
    );

    // An unknown user gets the same answer as a wrong password, so the response
    // cannot be used to enumerate accounts.
    let unknown = http_post(
        &server.ws_addr,
        "/auth",
        "application/json",
        r#"{"user":"nobody","password":"wrong"}"#,
    );
    assert!(
        unknown.starts_with("HTTP/1.1 401"),
        "unknown user should be unauthorized: {:?}",
        &unknown[..unknown.len().min(200)]
    );
}

/// A topic is echoed verbatim into `TOPIC <channel> <text>` fan-out, so
/// control characters (notably CR/LF) must be rejected instead of smuggling
/// extra protocol lines into other clients.
#[test]
fn topic_rejects_control_characters() {
    let server = TestServer::start();

    let mut alice = server.connect();
    alice.read_until("Welcome");
    alice.send("REGISTER alice secret");
    alice.read_until("OK registered");
    alice.send("JOIN #general");
    alice.read_until("OK joined");

    alice.send("TOPIC #general hello\u{0001}world");
    let reply = alice.read_until("ERROR");
    assert!(
        reply.contains("invalid topic"),
        "control-character topic must be rejected: {reply}"
    );
}

/// A message body is echoed verbatim into `PRIVMSG <channel> <user> :<body>`
/// fan-out lines, so control characters (notably CR/LF) must be rejected
/// instead of smuggling extra protocol lines into other clients.
#[test]
fn privmsg_rejects_control_characters() {
    let server = TestServer::start();

    let mut alice = server.connect();
    alice.read_until("Welcome");
    alice.send("REGISTER alice secret");
    alice.read_until("OK registered");
    alice.send("JOIN #general");
    alice.read_until("OK joined");

    let mut bob = server.connect();
    bob.read_until("Welcome");
    bob.send("REGISTER bob secret");
    bob.read_until("OK registered");
    bob.send("JOIN #general");
    bob.read_until("OK joined");

    // A control character in a channel message must be rejected, not broadcast.
    alice.send("PRIVMSG #general :hello\u{0001}world");
    let reply = alice.read_until("ERROR");
    assert!(
        reply.contains("invalid message"),
        "control-character message must be rejected: {reply}"
    );

    // A control character in a direct message must be rejected too.
    alice.send("PRIVMSG bob :hello\u{0001}world");
    let reply = alice.read_until("ERROR");
    assert!(
        reply.contains("invalid message"),
        "control-character DM must be rejected: {reply}"
    );

    // Bob must not have received either smuggled line.
    bob.assert_nothing_matching("hello", Duration::from_millis(400));
}

/// `NICK` must re-key every live connection of that user, not only the
/// connection that issued the command. Otherwise a second TCP/WS session
/// keeps listening under the old name and silently stops receiving DMs.
#[test]
fn nick_renames_other_live_connections() {
    let server = TestServer::start();

    let mut alice_a = server.connect();
    alice_a.read_until("Welcome");
    alice_a.send("REGISTER alice secret");
    alice_a.read_until("OK registered");

    let mut alice_b = server.connect();
    alice_b.read_until("Welcome");
    alice_b.send("LOGIN alice secret");
    alice_b.read_until("OK logged in");

    alice_a.send("NICK ali");
    assert!(alice_a.read_until("OK nick").contains("ali"));

    let mut bob = server.connect();
    bob.read_until("Welcome");
    bob.send("REGISTER bob secret");
    bob.read_until("OK registered");
    bob.send("PRIVMSG ali :hello both");
    bob.read_until("OK sent");

    let on_a = alice_a.read_until("PRIVMSG");
    let on_b = alice_b.read_until("PRIVMSG");
    assert!(
        on_a.contains("hello both"),
        "renaming connection missed the DM: {on_a}"
    );
    assert!(
        on_b.contains("hello both"),
        "other live connection was not re-keyed: {on_b}"
    );
}

/// A `/nick` over the bridge must update the bridge's own identity, so the
/// next message is broadcast under the new name rather than the stale one.
/// Regression test: the bridge used to keep its local `username` after a
/// rename, so a browser user's messages kept the old name while the roster
/// showed the new one.
#[test]
fn ws_bridge_nick_updates_its_own_identity() {
    let server = TestServer::start();

    let mut alice = server.connect();
    alice.read_until("Welcome");
    alice.send("REGISTER alice secret");
    alice.read_until("OK registered");
    drop(alice);

    let rt = tokio::runtime::Runtime::new().expect("runtime");
    let mut ws = ws_connect(&rt, &server.ws_addr, "general", "alice", "secret");

    // A peer on TCP joins so the bridge has someone to observe its messages.
    let mut bob = server.connect();
    bob.read_until("Welcome");
    bob.send("REGISTER bob secret");
    bob.read_until("OK registered");
    bob.send("JOIN #general");
    bob.read_until("OK joined");

    // Prove the subscription is installed before renaming.
    ws_wait_subscribed(&rt, &mut ws, &mut bob);

    // Rename over the bridge, then send a message.
    ws_send(&rt, &mut ws, "/nick ali");
    ws_expect_contains(&rt, &mut ws, "nick changed to ali");
    ws_send(&rt, &mut ws, "hello from the new name");

    // Bob must see the message attributed to the *new* name.
    let seen = bob.read_for(Duration::from_secs(5));
    assert!(
        seen.contains("hello from the new name"),
        "bridge message never reached bob: {seen:?}"
    );
    assert!(
        seen.contains("PRIVMSG #general ali :"),
        "message was broadcast under the stale name: {seen:?}"
    );
    assert!(
        !seen.contains("PRIVMSG #general alice :"),
        "message still used the pre-rename name: {seen:?}"
    );
}

/// The bridge must ignore the `user` query parameter and authenticate as the
/// identity the ticket was minted for. A ticket is a capability for a specific
/// account, so the URL cannot be used to impersonate someone else: presenting
/// alice's ticket with `user=bob` must still connect as alice, not bob.
#[test]
fn ws_bridge_ignores_user_param_and_uses_ticket_identity() {
    let server = TestServer::start();

    let mut alice = server.connect();
    alice.read_until("Welcome");
    alice.send("REGISTER alice secret");
    alice.read_until("OK registered");
    drop(alice);

    let mut bob = server.connect();
    bob.read_until("Welcome");
    bob.send("REGISTER bob secret");
    bob.read_until("OK registered");
    drop(bob);

    // Mint a ticket for alice, then present it with a forged `user=bob` param.
    let ticket = mint_ticket(&server.ws_addr, "alice", "secret");
    let rt = tokio::runtime::Runtime::new().expect("runtime");
    let mut ws = ws_connect_path(
        &rt,
        &server.ws_addr,
        &format!("/ws/%23general?user=bob&ticket={ticket}"),
    );

    // A peer on TCP joins and speaks; the bridge must receive the broadcast as
    // alice (the ticket's owner), proving the forged `user` param was ignored.
    let mut carol = server.connect();
    carol.read_until("Welcome");
    carol.send("REGISTER carol secret");
    carol.read_until("OK registered");
    carol.send("JOIN #general");
    carol.read_until("OK joined");

    let sync = ws_wait_subscribed(&rt, &mut ws, &mut carol);
    assert!(
        sync.contains("carol"),
        "bridge with a forged user param never subscribed: {sync:?}"
    );

    carol.send("PRIVMSG #general :hello alice");
    let got = ws_expect_contains(&rt, &mut ws, "hello alice");
    assert!(
        got.contains("alice"),
        "bridge should be authenticated as the ticket's owner (alice), not the forged bob: {got:?}"
    );
}

/// `CHAT_HISTORY_KEEP` is the documented retention knob. The prune task must
/// honour it so a small keep value actually bounds stored history.
#[test]
fn chat_history_keep_env_prunes_to_the_configured_count() {
    let server = TestServer::start_with(&[
        ("CHAT_HISTORY_KEEP", "2"),
        ("CHAT_PRUNE_INTERVAL_SECS", "1"),
    ]);

    let mut alice = server.connect();
    alice.read_until("Welcome");
    alice.send("REGISTER alice secret");
    alice.read_until("OK registered");
    alice.send("JOIN #general");
    alice.read_until("OK joined");

    for body in ["one", "two", "three", "four", "five"] {
        alice.send(&format!("PRIVMSG #general :{body}"));
        let got = alice.read_until("PRIVMSG #general");
        assert!(got.contains(body), "broadcast missing {body}: {got}");
    }

    // First prune tick is at startup (empty). Wait for at least one later
    // pass now that five rows exist.
    std::thread::sleep(Duration::from_millis(1500));

    alice.send("HISTORY #general 20");
    let history = alice.read_for(Duration::from_secs(2));
    assert!(
        history.contains("HISTORY #general"),
        "missing history header: {history:?}"
    );
    assert!(
        history.contains("four") && history.contains("five"),
        "newest two must be kept: {history:?}"
    );
    assert!(
        !history.contains("one") && !history.contains("two") && !history.contains("three"),
        "older rows must be pruned at CHAT_HISTORY_KEEP=2: {history:?}"
    );
}
