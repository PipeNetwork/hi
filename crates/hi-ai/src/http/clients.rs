//! Shared agent HTTP construction and inference replay policy.

use super::*;

/// Build a `reqwest::Client` with connection-pool and keep-alive tuned for an
/// agent loop that makes many sequential requests to the same endpoint.
/// Reusing connections avoids a TLS handshake on every model call — the
/// default `Client::new()` does pool internally, but this sets explicit
/// limits and keep-alive so long sessions reuse connections reliably.
///
/// Prefer [`agent_http_client_quick`] for non-streaming calls so those bounded
/// operations retain a finite read deadline.
pub fn agent_http_client() -> reqwest::Client {
    agent_http_client_for_socket(None)
}

/// Like [`agent_http_client`] but with short connect/read timeouts for
/// metadata, auth, MCP, and other non-streaming requests.
pub fn agent_http_client_quick() -> reqwest::Client {
    build_agent_http_client(
        None,
        http_timeout_secs(
            "HI_HTTP_QUICK_CONNECT_TIMEOUT_SECS",
            DEFAULT_QUICK_CONNECT_TIMEOUT_SECS,
        ),
        Some(Duration::from_secs(http_timeout_secs(
            "HI_HTTP_QUICK_READ_TIMEOUT_SECS",
            DEFAULT_QUICK_READ_TIMEOUT_SECS,
        ))),
        false,
    )
}

/// Agent client for non-model-stream transfers that retain the historical
/// finite socket-read deadline.
pub(crate) fn agent_http_client_bounded() -> reqwest::Client {
    build_agent_http_client(
        None,
        http_timeout_secs("HI_HTTP_CONNECT_TIMEOUT_SECS", DEFAULT_CONNECT_TIMEOUT_SECS),
        Some(Duration::from_secs(http_timeout_secs(
            "HI_HTTP_READ_TIMEOUT_SECS",
            DEFAULT_TRANSFER_READ_TIMEOUT_SECS,
        ))),
        false,
    )
}

/// Build the normal agent client while pinning all HTTP transport to one Unix
/// socket. The URL still supplies HTTP paths and Host semantics; no TCP or DNS
/// connection can be made by this client.
pub fn agent_http_client_for_socket(socket: Option<&std::path::Path>) -> reqwest::Client {
    build_agent_http_client(
        socket,
        http_timeout_secs("HI_HTTP_CONNECT_TIMEOUT_SECS", DEFAULT_CONNECT_TIMEOUT_SECS),
        model_stream_read_timeout(),
        false,
    )
}

/// Inference clients leave every replay and body-preserving redirect to the
/// request ledger, so the transport cannot send beyond its physical allowance.
pub fn inference_http_client_for_socket(socket: Option<&std::path::Path>) -> reqwest::Client {
    build_agent_http_client(
        socket,
        http_timeout_secs("HI_HTTP_CONNECT_TIMEOUT_SECS", DEFAULT_CONNECT_TIMEOUT_SECS),
        model_stream_read_timeout(),
        true,
    )
}

fn build_agent_http_client(
    socket: Option<&std::path::Path>,
    connect_timeout_secs: u64,
    read_timeout: Option<Duration>,
    controlled_inference: bool,
) -> reqwest::Client {
    // Identify hi to upstream HTTP services. `User-Agent` is the standard
    // channel; the `AI_AGENT` header mirrors the env-var convention the shell
    // path already uses, so HuggingFace infra sees a consistent `hi` marker on
    // both the subprocess and in-process HTTP surfaces. Additive for existing
    // providers — unknown headers are ignored.
    let mut headers = reqwest::header::HeaderMap::new();
    if let Ok(value) = reqwest::header::HeaderValue::from_str(HF_AGENT_ID) {
        headers.insert(HF_AGENT_HEADER_NAME, value);
    }
    let mut builder = reqwest::Client::builder()
        .user_agent(format!("hi/{}", env!("CARGO_PKG_VERSION")))
        .default_headers(headers)
        // Credentials are attached to requests against a configured base host.
        // reqwest strips `Authorization` on cross-host redirects but NOT custom
        // headers like Anthropic's `x-api-key`, so a same-scheme redirect to a
        // different host would forward the key. A same-host https→http hop
        // would also send the key in the clear. Refuse any origin change
        // (host, port, or scheme); same-origin path/version redirects still work.
        .redirect(if controlled_inference {
            reqwest::redirect::Policy::none()
        } else {
            credential_redirect_policy()
        })
        .retry(reqwest::retry::never())
        .connect_timeout(Duration::from_secs(connect_timeout_secs))
        .pool_idle_timeout(Some(Duration::from_secs(http_timeout_secs(
            "HI_HTTP_POOL_IDLE_TIMEOUT_SECS",
            DEFAULT_POOL_IDLE_TIMEOUT_SECS,
        ))))
        .pool_max_idle_per_host(http_env_usize(
            "HI_HTTP_POOL_MAX_IDLE_PER_HOST",
            DEFAULT_POOL_MAX_IDLE_PER_HOST,
            1,
            128,
        ))
        .tcp_keepalive(Some(Duration::from_secs(http_timeout_secs(
            "HI_HTTP_TCP_KEEPALIVE_SECS",
            DEFAULT_TCP_KEEPALIVE_SECS,
        ))));
    if let Some(read_timeout) = read_timeout {
        builder = builder.read_timeout(read_timeout);
    }
    #[cfg(unix)]
    if let Some(socket) = socket {
        builder = builder.unix_socket(socket);
    }
    builder.build().unwrap_or_else(|_| {
        agent_http_client_fallback(
            connect_timeout_secs,
            read_timeout,
            controlled_inference,
            socket,
        )
    })
}

/// Minimal fallback for agent clients. Unlike [`timed_http_client_fallback`],
/// this preserves an absent productive-stream read deadline rather than
/// silently turning a builder fault into an ordinary-work ceiling.
fn agent_http_client_fallback(
    connect_timeout_secs: u64,
    read_timeout: Option<Duration>,
    controlled_inference: bool,
    socket: Option<&std::path::Path>,
) -> reqwest::Client {
    let mut builder = reqwest::Client::builder()
        .redirect(if controlled_inference {
            reqwest::redirect::Policy::none()
        } else {
            credential_redirect_policy()
        })
        .retry(reqwest::retry::never())
        .connect_timeout(Duration::from_secs(connect_timeout_secs.max(1)));
    if let Some(read_timeout) = read_timeout {
        builder = builder.read_timeout(read_timeout);
    }
    #[cfg(unix)]
    if let Some(socket) = socket {
        builder = builder.unix_socket(socket);
    }
    builder
        .build()
        .expect("failed to build fallback reqwest Client")
}

/// Last-resort client that still carries timeouts — never fall back to an
/// unbounded `Client::new()`. Keeps the same credential redirect policy as
/// the primary agent client so a builder failure cannot silently start
/// forwarding `x-api-key` across hosts or onto http.
pub fn timed_http_client_fallback(
    connect_timeout_secs: u64,
    read_timeout_secs: u64,
) -> reqwest::Client {
    reqwest::Client::builder()
        .redirect(credential_redirect_policy())
        .connect_timeout(Duration::from_secs(connect_timeout_secs.max(1)))
        .read_timeout(Duration::from_secs(read_timeout_secs.max(1)))
        .build()
        .unwrap_or_else(|_| {
            reqwest::Client::builder()
                .redirect(credential_redirect_policy())
                .timeout(Duration::from_secs(
                    read_timeout_secs.max(connect_timeout_secs).max(1),
                ))
                .build()
                .expect("failed to build timed reqwest Client")
        })
}
