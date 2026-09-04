//! Shared support for sync transport, state, and record-reassembly tests.

use super::*;
use hi_ai::Message;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

async fn read_mock_http_request(socket: &mut tokio::net::TcpStream) -> std::io::Result<Vec<u8>> {
    const MAX_REQUEST_BYTES: usize = 8 * 1024 * 1024;
    let mut request = Vec::new();
    let mut chunk = [0_u8; 16 * 1024];
    loop {
        let read = socket.read(&mut chunk).await?;
        if read == 0 {
            break;
        }
        request.extend_from_slice(&chunk[..read]);
        if request.len() > MAX_REQUEST_BYTES {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "mock request exceeds test limit",
            ));
        }
        let Some(header_end) = request.windows(4).position(|window| window == b"\r\n\r\n") else {
            continue;
        };
        let header_bytes = &request[..header_end];
        let headers = String::from_utf8_lossy(header_bytes);
        let content_length = headers
            .lines()
            .find_map(|line| {
                let (name, value) = line.split_once(':')?;
                name.eq_ignore_ascii_case("content-length")
                    .then(|| value.trim().parse::<usize>().ok())
                    .flatten()
            })
            .unwrap_or(0);
        if request.len() >= header_end + 4 + content_length {
            break;
        }
    }
    Ok(request)
}

/// A minimal mock HTTP server that records received requests.
/// Returns 200 OK for every request and counts POSTs.
struct MockServer {
    base_url: String,
    post_count: Arc<AtomicUsize>,
    _handle: tokio::task::JoinHandle<()>,
}

impl MockServer {
    async fn start() -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let post_count = Arc::new(AtomicUsize::new(0));
        let count_clone = post_count.clone();
        let handle = tokio::spawn(async move {
            loop {
                let (mut sock, _) = match listener.accept().await {
                    Ok(s) => s,
                    Err(_) => break,
                };
                let count = count_clone.clone();
                tokio::spawn(async move {
                    let Ok(request) = read_mock_http_request(&mut sock).await else {
                        return;
                    };
                    let request = String::from_utf8_lossy(&request);
                    if request.starts_with("POST") {
                        count.fetch_add(1, Ordering::SeqCst);
                    }
                    // This mock handles exactly one request per accepted
                    // socket. Advertise that lifecycle explicitly so the
                    // pooled reqwest client never races a follow-up request
                    // against a socket the task has already dropped.
                    let body = if request.contains("/records") {
                        serde_json::json!({
                            "record_count": count.load(Ordering::SeqCst).saturating_mul(1_000)
                        })
                        .to_string()
                    } else {
                        "{}".to_string()
                    };
                    let response = format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                        body.len(),
                        body
                    );
                    let _ = sock.write_all(response.as_bytes()).await;
                    let _ = sock.shutdown().await;
                });
            }
        });
        Self {
            base_url: format!("http://{addr}"),
            post_count,
            _handle: handle,
        }
    }

    fn post_count(&self) -> usize {
        self.post_count.load(Ordering::SeqCst)
    }
}

mod records;
mod remote_state;
mod transport;
