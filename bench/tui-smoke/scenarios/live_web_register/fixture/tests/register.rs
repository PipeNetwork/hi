use std::io::{Read, Write};
use std::net::TcpStream;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

struct Server {
    child: Child,
    addr: String,
}

impl Server {
    fn start() -> Self {
        let mut child = Command::new(env!("CARGO_BIN_EXE_web_register"))
            .env("APP_ADDR", "127.0.0.1:0")
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .expect("spawn web_register");
        let mut stdout = child.stdout.take().expect("stdout");
        let addr = read_listening_addr(&mut stdout);
        std::thread::spawn(move || {
            let mut sink = Vec::new();
            let _ = stdout.read_to_end(&mut sink);
        });
        let server = Self { child, addr };
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

    fn request(&self, req: &str) -> (u16, String) {
        let mut stream = TcpStream::connect(&self.addr).expect("connect");
        stream
            .set_read_timeout(Some(Duration::from_secs(3)))
            .expect("read timeout");
        stream.write_all(req.as_bytes()).expect("write");
        let _ = stream.flush();
        let mut body = String::new();
        let _ = stream.read_to_string(&mut body);
        let status = body
            .lines()
            .next()
            .and_then(|line| line.split_whitespace().nth(1))
            .and_then(|code| code.parse().ok())
            .unwrap_or(0);
        (status, body)
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn read_listening_addr(stdout: &mut impl Read) -> String {
    let mut buf = Vec::new();
    let deadline = Instant::now() + Duration::from_secs(5);
    while Instant::now() < deadline {
        let mut byte = [0u8; 1];
        if stdout.read(&mut byte).ok() == Some(1) {
            buf.push(byte[0]);
            if byte[0] == b'\n' {
                let line = String::from_utf8_lossy(&buf);
                if let Some(rest) = line.strip_prefix("listening on ") {
                    return rest.trim().to_string();
                }
                buf.clear();
            }
        } else {
            std::thread::sleep(Duration::from_millis(10));
        }
    }
    panic!(
        "did not hear listening address: {}",
        String::from_utf8_lossy(&buf)
    );
}

fn page_has_register_form(html: &str) -> bool {
    let lower = html.to_ascii_lowercase();
    let has_form = lower.contains("<form")
        && (lower.contains("id=\"registerform\"") || lower.contains("id='registerform'"));
    let posts = lower.contains("/register")
        && (lower.contains("fetch") || lower.contains("method=\"post\"") || lower.contains("method='post'"));
    has_form && posts
}

#[test]
fn register_form_is_on_the_page() {
    let html = include_str!("../src/web/index.html");
    assert!(
        page_has_register_form(html),
        "index.html has no register form that POSTs /register:\n{html}"
    );
}

#[test]
fn post_register_creates_an_account() {
    let server = Server::start();
    let (status, body) = server.request(
        "POST /register HTTP/1.1\r\nHost: localhost\r\nContent-Length: 11\r\nConnection: close\r\n\r\nuser=ada\r\n",
    );
    assert_eq!(
        status, 201,
        "POST /register must return 201, got {status}: {body}"
    );
}

#[test]
fn get_page_includes_the_register_form() {
    let server = Server::start();
    let (status, body) = server.request("GET / HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n");
    assert_eq!(status, 200, "GET / failed: {body}");
    let html = body.split("\r\n\r\n").nth(1).unwrap_or(&body);
    assert!(
        page_has_register_form(html),
        "served page has no register form:\n{html}"
    );
}
