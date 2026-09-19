//! Offline status page. POST /register is missing until the live e2e fix.

use std::io::{Read, Write};
use std::net::TcpListener;

const HTML: &str = include_str!("web/index.html");

fn main() {
    let addr = std::env::var("APP_ADDR").unwrap_or_else(|_| "127.0.0.1:0".to_string());
    let listener = TcpListener::bind(&addr).expect("bind");
    let bound = listener.local_addr().expect("local addr");
    println!("listening on {bound}");
    let _ = std::io::stdout().flush();
    for incoming in listener.incoming() {
        let Ok(mut stream) = incoming else { continue };
        let mut buf = [0u8; 8192];
        let n = stream.read(&mut buf).unwrap_or(0);
        let req = String::from_utf8_lossy(&buf[..n]);
        let head = req.lines().next().unwrap_or("");
        let (status, ctype, body) = if head.starts_with("POST /register") {
            // BUG: account creation is not implemented.
            (
                "404 Not Found",
                "text/plain; charset=utf-8",
                "not found".to_string(),
            )
        } else {
            ("200 OK", "text/html; charset=utf-8", HTML.to_string())
        };
        let resp = format!(
            "HTTP/1.1 {status}\r\nContent-Type: {ctype}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        );
        let _ = stream.write_all(resp.as_bytes());
    }
}
