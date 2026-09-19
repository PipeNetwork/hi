mod auth;
mod db;
mod metrics;
mod protocol;
mod server;
mod state;
mod tls;
mod web;
mod ws;

use std::sync::Arc;
use std::time::Duration;
use tokio::io::AsyncWriteExt;
use tokio::net::TcpListener;
use tokio::sync::Semaphore;
use tokio::task::JoinSet;

use db::Db;
use state::AppState;

/// How many concurrent connections (line protocol + websocket) the server will
/// accept before rejecting new ones with `ERROR server full`. Bounded so a
/// connection flood cannot pile up unbounded pending sockets or tasks.
const MAX_CONNECTIONS: usize = 256;

/// How often the retention task wakes to prune old messages.
const PRUNE_INTERVAL: Duration = Duration::from_secs(60);

/// How many messages to keep per channel / direct-message pair. Older rows are
/// deleted so the database does not grow without bound. Overridable via the
/// `CHAT_HISTORY_KEEP` environment variable (see `history_keep_from_env`).
const DEFAULT_HISTORY_KEEP: i64 = 1000;

/// Parse a positive integer env value, falling back to `default` when unset
/// or not a positive integer. A non-positive value would make the prune
/// query delete everything (keep) or spin the timer (interval), so it is
/// clamped to a sane floor.
fn parse_positive_u64(raw: Option<&str>, default: u64) -> u64 {
    raw.and_then(|v| v.parse::<u64>().ok())
        .filter(|n| *n > 0)
        .unwrap_or(default)
}

fn parse_history_keep(raw: Option<&str>) -> i64 {
    parse_positive_u64(raw, DEFAULT_HISTORY_KEEP as u64) as i64
}

/// Read the per-channel / per-DM retention count from `CHAT_HISTORY_KEEP`,
/// falling back to [`DEFAULT_HISTORY_KEEP`] when unset or not a positive
/// integer.
fn history_keep_from_env() -> i64 {
    parse_history_keep(std::env::var("CHAT_HISTORY_KEEP").ok().as_deref())
}

fn prune_interval_from_env() -> Duration {
    Duration::from_secs(parse_positive_u64(
        std::env::var("CHAT_PRUNE_INTERVAL_SECS").ok().as_deref(),
        PRUNE_INTERVAL.as_secs(),
    ))
}

/// Bind `addr`, falling back to the same host on port 0 if it is already in
/// use. Docker (and other local services) often occupy 8080, which is the
/// documented default — failing closed there made `cargo run` look like it
/// served the UI when curl was actually talking to whatever else held the port.
async fn bind_listen(addr: &str) -> std::io::Result<TcpListener> {
    match TcpListener::bind(addr).await {
        Ok(listener) => Ok(listener),
        Err(err) if err.kind() == std::io::ErrorKind::AddrInUse => {
            let fallback = ephemeral_on_same_host(addr);
            tracing::warn!("{addr} is in use; falling back to {fallback}");
            eprintln!("chat: {addr} is in use; listening on {fallback} instead");
            TcpListener::bind(&fallback).await
        }
        Err(err) => Err(err),
    }
}

fn ephemeral_on_same_host(addr: &str) -> String {
    match addr.rsplit_once(':') {
        Some((host, _)) if !host.is_empty() => format!("{host}:0"),
        _ => "127.0.0.1:0".into(),
    }
}

#[tokio::main]
async fn main() -> std::io::Result<()> {
    // Structured logging. `RUST_LOG` controls verbosity (default: info).
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let db_path = std::env::var("CHAT_DB").unwrap_or_else(|_| "chat.db".to_string());
    let db = Db::open(&db_path).map_err(|e| std::io::Error::other(format!("open db: {e}")))?;
    let state = Arc::new(AppState::new(db));

    let addr = std::env::var("CHAT_ADDR").unwrap_or_else(|_| "127.0.0.1:8080".to_string());
    let listener = bind_listen(&addr).await?;
    let tcp_bound = listener.local_addr()?;

    // Optional TLS termination. When `CHAT_TLS_CERT` and `CHAT_TLS_KEY` are set,
    // every accepted TCP connection is wrapped in TLS before being handed to the
    // line protocol / websocket handler. Without them the server stays plaintext
    // (suitable for localhost / behind a reverse proxy).
    let tls_acceptor = match tls::tls_acceptor_from_env() {
        Some(Ok(a)) => {
            tracing::info!("TLS enabled");
            Some(a)
        }
        Some(Err(e)) => {
            tracing::error!("failed to load TLS config: {e}");
            return Err(e);
        }
        None => None,
    };

    // The websocket bridge shares the same port as the HTTP page, so the page's
    // own origin *is* the bridge and the client needs no configuration to find
    // it. It is a separate listener task so a slow websocket handshake cannot
    // stall the line protocol.
    let ws_addr = std::env::var("CHAT_WS_ADDR").unwrap_or_else(|_| addr.clone());
    let ws_listener = bind_listen(&ws_addr).await?;
    let ws_bound = ws_listener.local_addr()?;

    // Advertise the *bound* bridge address (so an ephemeral `:0` port resolves
    // to the real one) to clients via the `WS` command. The scheme matches the
    // TLS mode so the advertised URL is actually connectable.
    let ws_scheme = if tls_acceptor.is_some() { "wss" } else { "ws" };
    state.set_ws_base_url(format!("{ws_scheme}://{ws_bound}"));

    // One global cap shared by both the line-protocol and websocket listeners,
    // so a flood on one port cannot push the total past the bound. Each
    // connection (TCP or WS) holds a permit for its lifetime.
    let conn_semaphore = Arc::new(Semaphore::new(MAX_CONNECTIONS));
    let ws_sem = Arc::clone(&conn_semaphore);

    // Track every spawned connection task so shutdown can await them all.
    let conn_tasks = Arc::new(tokio::sync::Mutex::new(JoinSet::new()));
    let conn_tasks_listener = Arc::clone(&conn_tasks);
    let conn_state = Arc::clone(&state);
    let conn_sem = Arc::clone(&conn_semaphore);
    let conn_tls = tls_acceptor.clone();
    let listener_task = tokio::spawn(async move {
        loop {
            let (stream, peer) = match listener.accept().await {
                Ok(x) => x,
                Err(e) => {
                    tracing::error!("accept error: {e}");
                    continue;
                }
            };
            let state = Arc::clone(&conn_state);
            let sem = Arc::clone(&conn_sem);
            let tasks = Arc::clone(&conn_tasks_listener);
            let tls = conn_tls.clone();
            tasks.lock().await.spawn(async move {
                // Reject excess connections rather than queueing them, so a
                // connection flood cannot pile up unbounded pending sockets.
                let Ok(_permit) = sem.try_acquire_owned() else {
                    let mut stream = stream;
                    let _ = stream.write_all(b"ERROR server full\r\n").await;
                    return;
                };
                // Per-IP cap so a single host cannot open an unbounded number
                // of sockets even though the global cap is shared.
                if !state.allow_conn_attempt(peer.ip()).await {
                    let mut stream = stream;
                    let _ = stream.write_all(b"ERROR too many connections\r\n").await;
                    return;
                }
                tracing::info!(%peer, "connection");
                crate::metrics::METRICS
                    .tcp_connections
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                // Optional TLS wrap before the line protocol handler. Both arms
                // are boxed into the same trait object so the plaintext and TLS
                // paths share one handler.
                let stream: Box<dyn crate::web::IoStream> = match &tls {
                    Some(acceptor) => match acceptor.accept(stream).await {
                        Ok(s) => Box::new(s),
                        Err(e) => {
                            tracing::warn!(%peer, "TLS handshake failed: {e}");
                            return;
                        }
                    },
                    None => Box::new(stream),
                };
                server::handle_connection(stream, state, peer.ip()).await;
                tracing::info!(%peer, "connection closed");
            });
        }
    });

    // Track every spawned websocket connection task so shutdown can await them
    // all, mirroring `conn_tasks` for the TCP path. Without this, in-flight
    // bridge connections (and their mid-write broadcasts) would be dropped the
    // moment the process exits. Shared between the listener task (which spawns
    // into it) and the shutdown path (which drains it), so it lives behind a
    // mutex.
    let ws_conn_tasks = Arc::new(tokio::sync::Mutex::new(JoinSet::new()));
    let ws_conn_tasks_listener = Arc::clone(&ws_conn_tasks);
    let ws_state = Arc::clone(&state);
    let ws_sem = Arc::clone(&ws_sem);
    let ws_tls = tls_acceptor.clone();
    let ws_listener_task = tokio::spawn(async move {
        loop {
            let (stream, peer) = match ws_listener.accept().await {
                Ok(x) => x,
                Err(e) => {
                    tracing::error!("ws accept error: {e}");
                    continue;
                }
            };
            let state = Arc::clone(&ws_state);
            let sem = Arc::clone(&ws_sem);
            let tasks = Arc::clone(&ws_conn_tasks_listener);
            let tls = ws_tls.clone();
            tasks.lock().await.spawn(async move {
                // Reject excess WS connections rather than queueing them, so a
                // flood of handshakes cannot spawn unbounded tasks.
                let Ok(_permit) = sem.try_acquire_owned() else {
                    tracing::warn!(%peer, "ws connection rejected (server full)");
                    return;
                };
                // Per-IP cap so a single host cannot open an unbounded number
                // of bridge sockets even though the global cap is shared.
                if !state.allow_conn_attempt(peer.ip()).await {
                    tracing::warn!(%peer, "ws connection rejected (per-IP limit)");
                    return;
                }
                tracing::info!(%peer, "ws connection");
                crate::metrics::METRICS
                    .ws_connections
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                // Optional TLS wrap before peeking the request head. Both arms
                // are boxed into the same trait object so the plaintext and TLS
                // paths share one handler.
                let stream: Box<dyn crate::web::IoStream> = match &tls {
                    Some(acceptor) => match acceptor.accept(stream).await {
                        Ok(s) => Box::new(s),
                        Err(e) => {
                            tracing::warn!(%peer, "ws TLS handshake failed: {e}");
                            return;
                        }
                    },
                    None => Box::new(stream),
                };
                // Route on a peek so a websocket upgrade still reaches
                // tungstenite unconsumed: the page (GET /) is answered here, and
                // only /ws/<channel> is handed to the bridge. Serving both on one
                // port means the page's own origin *is* the bridge, so the client
                // needs no configuration to find it.
                let mut stream = stream;
                let head = web::peek_head(&mut stream).await;
                // `peek_head` reads (consumes) the head bytes, so replay them
                // through a `PrefixedStream` so the winning branch still sees the
                // full request: tungstenite reads the bridge handshake, and
                // `serve` drains the head for the static/login routes.
                let stream = web::PrefixedStream::new(head.clone(), stream);
                match web::route(&head) {
                    web::Route::Bridge => {
                        ws::handle_ws(stream, state, peer.to_string()).await;
                        tracing::info!(%peer, "ws connection closed");
                    }
                    other => {
                        let _ = web::serve(stream, other, state, peer.ip()).await;
                    }
                }
            });
        }
    });

    // Retention: keep the `messages` and `direct_messages` tables bounded (see
    // `DEFAULT_HISTORY_KEEP` / `CHAT_HISTORY_KEEP`).
    let prune_state = Arc::clone(&state);
    let keep = history_keep_from_env();
    let prune_every = prune_interval_from_env();
    let prune_task = tokio::spawn(async move {
        let mut tick = tokio::time::interval(prune_every);
        loop {
            tick.tick().await;
            if let Err(e) = prune_state.prune_old_messages(keep).await {
                tracing::warn!("prune failed: {e}");
            }
        }
    });

    tracing::info!("chat server listening on {tcp_bound}");
    tracing::info!("websocket bridge listening on {ws_bound}");
    // Also print to stdout so the integration tests (which read the child's
    // stdout) can discover the bound addresses. tracing writes to stderr.
    println!("chat server listening on {tcp_bound}");
    println!("websocket bridge listening on {ws_bound}");

    // Graceful shutdown: on Ctrl-C (or SIGTERM on Unix), stop accepting new
    // connections and let in-flight ones finish so a restart does not drop
    // messages mid-write. SIGTERM is what orchestrators and `kill` send, so
    // handling it (not just Ctrl-C) makes containerized deploys shut down
    // cleanly too.
    let shutdown = async {
        #[cfg(unix)]
        {
            let mut sigterm =
                tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
                    .expect("install SIGTERM handler");
            tokio::select! {
                _ = tokio::signal::ctrl_c() => {}
                _ = sigterm.recv() => {}
            }
        }
        #[cfg(not(unix))]
        {
            let _ = tokio::signal::ctrl_c().await;
        }
    };
    tokio::pin!(shutdown);
    tokio::pin!(listener_task);
    tokio::pin!(ws_listener_task);

    tokio::select! {
        _ = &mut shutdown => {
            tracing::info!("shutting down");
        }
        _ = &mut listener_task => {
            // If the TCP listener dies, the whole server should wind down.
        }
        _ = &mut ws_listener_task => {
            // If the websocket bridge listener dies, the whole server should
            // wind down too — otherwise the page keeps being served but the
            // bridge it points at is gone.
        }
    }

    // Stop both listeners and wait for in-flight connections to finish so a
    // restart does not drop messages mid-write. The TCP listener task is an
    // infinite accept loop, so it must be aborted explicitly — otherwise it
    // keeps running (and holding the process) after shutdown. Take the JoinSet
    // locks only here — holding them across the accept loop prevents spawn.
    listener_task.abort();
    ws_listener_task.abort();
    {
        let mut conn_tasks = conn_tasks.lock().await;
        while conn_tasks.join_next().await.is_some() {}
    }
    {
        let mut ws_conn_tasks = ws_conn_tasks.lock().await;
        while ws_conn_tasks.join_next().await.is_some() {}
    }
    // Aborting the prune task mid-prune is safe: each prune runs inside a
    // single SQLite transaction, so an abort between ticks leaves the DB in a
    // consistent state (the in-flight transaction rolls back). We do not wait
    // for it because a prune can take a while on a large history and shutdown
    // should not block on it.
    prune_task.abort();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_history_keep_defaults_and_clamps() {
        assert_eq!(parse_history_keep(None), DEFAULT_HISTORY_KEEP);
        assert_eq!(parse_history_keep(Some("2")), 2);
        assert_eq!(parse_history_keep(Some("5000")), 5000);
        assert_eq!(parse_history_keep(Some("0")), DEFAULT_HISTORY_KEEP);
        assert_eq!(parse_history_keep(Some("-3")), DEFAULT_HISTORY_KEEP);
        assert_eq!(parse_history_keep(Some("nope")), DEFAULT_HISTORY_KEEP);
        assert_eq!(parse_history_keep(Some("")), DEFAULT_HISTORY_KEEP);
    }

    #[test]
    fn parse_prune_interval_defaults_and_clamps() {
        assert_eq!(parse_positive_u64(None, 60), 60);
        assert_eq!(parse_positive_u64(Some("1"), 60), 1);
        assert_eq!(parse_positive_u64(Some("0"), 60), 60);
        assert_eq!(parse_positive_u64(Some("abc"), 60), 60);
    }

    #[test]
    fn ephemeral_on_same_host_keeps_the_host() {
        assert_eq!(ephemeral_on_same_host("127.0.0.1:8080"), "127.0.0.1:0");
        assert_eq!(ephemeral_on_same_host("[::1]:8080"), "[::1]:0");
        assert_eq!(ephemeral_on_same_host("8080"), "127.0.0.1:0");
    }

    #[tokio::test]
    async fn bind_listen_falls_back_when_address_is_in_use() {
        let first = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let taken = first.local_addr().unwrap().to_string();
        let second = bind_listen(&taken).await.unwrap();
        assert_ne!(
            second.local_addr().unwrap(),
            first.local_addr().unwrap(),
            "fallback must pick a free port, not collide"
        );
    }
}
