//! JSON-RPC framing over stdio: messages are `Content-Length: N\r\n\r\n`
//! followed by N bytes of JSON. This is the LSP base transport.

use std::time::Duration;
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::process::{ChildStdin, ChildStdout};

/// Upper bound on a single LSP message body. Local language servers can emit
/// large `publishDiagnostics` payloads, but a multi-MB cap is generous for any
/// real notification; a server advertising `Content-Length: 9999999999` is buggy
/// or hostile and must not trigger a multi-GB allocation before the read timeout
/// fires. Mirrors `hi_protocol::MAX_FRAME_BYTES` in spirit.
const MAX_MESSAGE_BYTES: usize = 16 * 1024 * 1024;

/// Read one JSON-RPC message (header + body) from a server's stdout.
///
/// Returns the raw JSON bytes, or `None` at EOF. Server-sent notifications
/// and responses share this framing — the caller dispatches by `method`/`id`.
pub async fn read_message(reader: &mut BufReader<ChildStdout>) -> Option<Vec<u8>> {
    // Parse headers until a blank line.
    let mut content_length: Option<usize> = None;
    let mut line = String::new();
    loop {
        line.clear();
        let n = reader.read_line(&mut line).await.ok()?;
        if n == 0 {
            return None; // EOF
        }
        let trimmed = line.trim_end_matches(['\r', '\n']);
        if trimmed.is_empty() {
            break; // end of headers
        }
        if let Some(rest) = trimmed.strip_prefix("Content-Length:") {
            content_length = rest.trim().parse().ok();
        }
        // Other headers (Content-Type, etc.) are ignored.
    }
    let len = content_length?;
    if len > MAX_MESSAGE_BYTES {
        return None;
    }
    let mut buf = vec![0u8; len];
    reader.read_exact(&mut buf).await.ok()?;
    Some(buf)
}

/// Write one JSON-RPC message to a server's stdin.
pub async fn write_message(stdin: &mut ChildStdin, body: &str) -> std::io::Result<()> {
    let header = format!("Content-Length: {}\r\n\r\n", body.len());
    stdin.write_all(header.as_bytes()).await?;
    stdin.write_all(body.as_bytes()).await?;
    stdin.flush().await?;
    Ok(())
}

/// Default timeout for LSP requests. Servers can be slow (rust-analyzer cold
/// start / indexing a large workspace), but we don't want to hang a turn
/// forever. Overridable via `HI_LSP_TIMEOUT_SECS`.
pub const DEFAULT_REQUEST_TIMEOUT_SECS: u64 = 120;

/// The effective LSP request timeout: `HI_LSP_TIMEOUT_SECS` if set to a
/// positive integer, else [`DEFAULT_REQUEST_TIMEOUT_SECS`]. Read per call so
/// the env var takes effect without a rebuild.
pub fn request_timeout() -> Duration {
    let secs = std::env::var("HI_LSP_TIMEOUT_SECS")
        .ok()
        .and_then(|value| value.trim().parse::<u64>().ok())
        .filter(|secs| *secs > 0)
        .unwrap_or(DEFAULT_REQUEST_TIMEOUT_SECS);
    Duration::from_secs(secs)
}

#[cfg(test)]
mod tests {
    #[tokio::test]
    async fn write_then_read_round_trips() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let (mut tx, mut rx) = tokio::io::duplex(1024);
        let body = r#"{"jsonrpc":"2.0","id":1,"method":"test","params":{}}"#;
        // Simulate writing an LSP message to stdin (write side).
        let header = format!("Content-Length: {}\r\n\r\n", body.len());
        tx.write_all(header.as_bytes()).await.unwrap();
        tx.write_all(body.as_bytes()).await.unwrap();
        // Read it back as raw bytes and verify framing.
        let mut buf = vec![0u8; header.len() + body.len()];
        rx.read_exact(&mut buf).await.unwrap();
        let s = String::from_utf8(buf).unwrap();
        assert!(s.starts_with("Content-Length: "));
        assert!(s.contains(body));
    }

    /// Parse a `Content-Length` header from raw bytes.
    #[test]
    fn content_length_header_parsing() {
        let msg = b"Content-Length: 13\r\n\r\n{\"id\":1}";
        let header_end = msg.windows(4).position(|w| w == b"\r\n\r\n").unwrap();
        let headers = &msg[..header_end];
        let body_len: usize = std::str::from_utf8(headers)
            .unwrap()
            .lines()
            .find_map(|l| l.strip_prefix("Content-Length:"))
            .map(|v| v.trim().parse().unwrap())
            .unwrap();
        assert_eq!(body_len, 13);
        assert_eq!(&msg[header_end + 4..], b"{\"id\":1}");
    }

    /// `uri_to_path` strips `file://` and `file:///` prefixes.
    #[test]
    fn uri_to_path_strips_scheme() {
        use crate::client::uri_to_path;
        assert_eq!(
            uri_to_path("file:///home/user/src/main.rs"),
            "/home/user/src/main.rs"
        );
        assert_eq!(
            uri_to_path("file://home/user/src/main.rs"),
            "home/user/src/main.rs"
        );
        assert_eq!(uri_to_path("not-a-uri"), "not-a-uri");
    }

    /// `path_to_uri` then `uri_to_path` round-trips for paths with spaces and
    /// non-ASCII characters — the encode/decode path is the most bug-prone
    /// part, so this exercises it directly.
    #[test]
    fn path_to_uri_round_trips_spaces_and_unicode() {
        use crate::client::{path_to_uri, uri_to_path};
        let cases = [
            "/home/user/my project/main.rs",
            "/home/user/проект/файл.rs",
            "/tmp/a b/c d.txt",
            "/home/user/名前.rs",
        ];
        for p in cases {
            let uri = path_to_uri(std::path::Path::new(p));
            assert!(uri.starts_with("file://"), "uri was {uri}");
            let back = uri_to_path(&uri);
            assert_eq!(back, *p, "round-trip failed for {p}: uri={uri}");
        }
    }
}
