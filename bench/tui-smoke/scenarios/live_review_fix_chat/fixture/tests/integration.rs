use std::io::{BufRead, BufReader, ErrorKind, Read, Write};
use std::net::TcpStream;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

struct TestServer {
    child: Child,
    addr: String,
    _stdout_drain: std::thread::JoinHandle<()>,
}

impl TestServer {
    fn start() -> Self {
        let mut child = Command::new(env!("CARGO_BIN_EXE_chat"))
            .env("CHAT_ADDR", "127.0.0.1:0")
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .expect("spawn chat");
        let stdout = child.stdout.take().expect("stdout");
        let (addr, drain) = read_listening_addr(stdout);
        let server = Self {
            child,
            addr,
            _stdout_drain: drain,
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
            std::thread::sleep(Duration::from_millis(20));
        }
        panic!("server did not become ready at {}", self.addr);
    }

    fn connect(&self) -> TestClient {
        let stream = TcpStream::connect(&self.addr).expect("connect");
        stream
            .set_read_timeout(Some(Duration::from_secs(3)))
            .expect("read timeout");
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

fn read_listening_addr(stdout: impl Read + Send + 'static) -> (String, std::thread::JoinHandle<()>) {
    let (tx, rx) = std::sync::mpsc::channel();
    let handle = std::thread::spawn(move || {
        let mut reader = BufReader::new(stdout);
        let mut line = String::new();
        let mut sent = false;
        loop {
            line.clear();
            if reader.read_line(&mut line).is_err() || line.is_empty() {
                break;
            }
            if !sent {
                if let Some(rest) = line.strip_prefix("chat server listening on ") {
                    let addr = rest.split_whitespace().next().unwrap_or("").to_string();
                    let _ = tx.send(addr);
                    sent = true;
                }
            }
        }
    });
    let addr = rx
        .recv_timeout(Duration::from_secs(5))
        .expect("server did not print a listening address");
    (addr, handle)
}

struct TestClient {
    reader: BufReader<TcpStream>,
}

impl TestClient {
    fn send(&mut self, line: &str) {
        let stream = self.reader.get_mut();
        writeln!(stream, "{line}").expect("write");
        stream.flush().expect("flush");
    }

    fn read_line(&mut self) -> String {
        let mut buf = String::new();
        self.reader.read_line(&mut buf).expect("read line");
        buf
    }

    fn read_until(&mut self, prefix: &str) -> String {
        loop {
            let line = self.read_line();
            if line.starts_with(prefix) {
                return line;
            }
        }
    }
}

#[test]
fn register_join_privmsg_roundtrip() {
    let server = TestServer::start();
    let mut alice = server.connect();
    alice.read_until("Welcome");
    alice.send("REGISTER alice secret");
    assert!(alice.read_until("OK registered").contains("alice"));
    alice.send("JOIN #general");
    assert!(alice.read_until("OK joined").contains("#general"));
    alice.send("PRIVMSG #general :hello bob");
    let got = alice.read_until("PRIVMSG #general");
    assert!(got.contains("alice") && got.contains("hello bob"), "got: {got}");
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

    alice.send("KICK #general bob");
    assert!(alice.read_until("OK kicked").contains("bob"));
    assert!(bob.read_until("KICKED").contains("#general"));

    alice.send("PRIVMSG #general :after kick");
    let got = alice.read_until("PRIVMSG #general");
    assert!(got.contains("after kick"), "got: {got}");

    std::thread::sleep(Duration::from_millis(200));
    let leftover = bob.reader.buffer();
    assert!(
        leftover.is_empty(),
        "bob received data after being kicked: {:?}",
        String::from_utf8_lossy(leftover)
    );
    let stream = bob.reader.get_mut();
    stream.set_nonblocking(true).expect("nonblocking");
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
