use futures_util::{SinkExt, StreamExt};
use std::time::Duration;
use tokio::io::{AsyncRead, AsyncWrite};
use tokio_tungstenite::tungstenite::handshake::server::{Request, Response};
use tokio_tungstenite::tungstenite::{Error, Message};

use crate::server::RateLimiter;
use crate::state::SharedState;

/// Maximum accepted websocket message size (bytes). Larger frames are rejected
/// so a single client cannot abuse memory.
pub const MAX_WS_MESSAGE: usize = 4096;

/// Largest number of past messages a bridge may ask to replay on connect. Bounded
/// so a single client cannot force an arbitrarily large read on every reconnect.
/// Kept in sync with the `HISTORY` command's cap (`crate::protocol::MAX_HISTORY`)
/// so both replay paths agree on the largest read a client can force.
pub const MAX_WS_BACKFILL: i64 = crate::protocol::MAX_HISTORY as i64;

/// Idle timeout for the websocket bridge. If a client sends nothing for this
/// long, the bridge is closed. Unlike the TCP path there is no PING/PONG probe
/// (tungstenite has no built-in keepalive and the page is a passive viewer), so
/// an idle bridge is simply torn down. Prevents a client that opened a tab and
/// walked away from holding a task, socket and channel subscription forever.
const WS_IDLE_TIMEOUT: Duration = Duration::from_secs(600);

/// Whether `e` is just a peer that went away mid-connection. These are routine
/// (clients closing tabs, tests dropping sockets) rather than server faults.
fn is_client_disconnect(e: &Error) -> bool {
    matches!(
        e,
        Error::Io(io)
            if matches!(
                io.kind(),
                std::io::ErrorKind::ConnectionReset
                    | std::io::ErrorKind::ConnectionAborted
                    | std::io::ErrorKind::BrokenPipe
                    | std::io::ErrorKind::UnexpectedEof
            )
    )
}

/// Percent-decode a URL component (`%20` -> space, `+` -> space). Returns the
/// decoded string; invalid escape sequences are left as-is.
pub fn url_decode(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'%' if i + 2 < bytes.len() => {
                let hi = (bytes[i + 1] as char).to_digit(16);
                let lo = (bytes[i + 2] as char).to_digit(16);
                match (hi, lo) {
                    (Some(hi), Some(lo)) => {
                        out.push((hi * 16 + lo) as u8);
                        i += 3;
                    }
                    _ => {
                        out.push(bytes[i]);
                        i += 1;
                    }
                }
            }
            b'+' => {
                out.push(b' ');
                i += 1;
            }
            b => {
                out.push(b);
                i += 1;
            }
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// A slash command typed into the web UI's composer, parsed from a line that
/// starts with `/`. The bridge is publish-only, so these are the only way a
/// browser client can drive the protocol (join/part/nick/topic/history/…).
#[derive(Debug, PartialEq, Eq)]
pub enum SlashCommand {
    /// `/join <channel>` — join (or create) a channel.
    Join { channel: String },
    /// `/part [channel]` — leave the current channel, or a named one.
    Part { channel: Option<String> },
    /// `/nick <name>` — change the display name.
    Nick { name: String },
    /// `/topic [text]` — get or set the channel topic.
    Topic { text: Option<String> },
    /// `/history [limit]` — replay recent messages into the log.
    History { limit: usize },
    /// `/help` — list the available commands.
    Help,
    /// `/quit` — close the bridge.
    Quit,
    /// `/me <action>` — an action line (`* nick does something`).
    Me { text: String },
    /// `/msg <user> <text>` — send a direct message.
    Msg { user: String, text: String },
    /// Anything else, kept verbatim so the UI can echo an error.
    Unknown(String),
}

/// Parse a line that begins with `/` into a [`SlashCommand`]. A line that does
/// not start with `/` is not a command and returns `None`.
pub fn parse_slash_command(line: &str) -> Option<SlashCommand> {
    let line = line.trim();
    if !line.starts_with('/') {
        return None;
    }
    let mut parts = line[1..].splitn(2, ' ');
    let verb = parts.next().unwrap_or("").to_ascii_lowercase();
    let rest = parts.next().unwrap_or("").trim();
    match verb.as_str() {
        "join" => {
            let channel = rest.trim().to_string();
            if channel.is_empty() {
                Some(SlashCommand::Unknown(line.to_string()))
            } else {
                Some(SlashCommand::Join { channel })
            }
        }
        "part" => Some(SlashCommand::Part {
            channel: if rest.is_empty() {
                None
            } else {
                Some(rest.to_string())
            },
        }),
        "nick" => {
            let name = rest.trim().to_string();
            if name.is_empty() {
                Some(SlashCommand::Unknown(line.to_string()))
            } else {
                Some(SlashCommand::Nick { name })
            }
        }
        "topic" => Some(SlashCommand::Topic {
            text: if rest.is_empty() {
                None
            } else {
                Some(rest.to_string())
            },
        }),
        "history" => {
            let limit = rest
                .split_whitespace()
                .next()
                .and_then(|n| n.parse::<usize>().ok())
                .unwrap_or(50)
                .clamp(1, crate::protocol::MAX_HISTORY);
            Some(SlashCommand::History { limit })
        }
        "help" => Some(SlashCommand::Help),
        "quit" => Some(SlashCommand::Quit),
        "me" => Some(SlashCommand::Me {
            text: rest.to_string(),
        }),
        "msg" => {
            let mut it = rest.splitn(2, ' ');
            let user = it.next().unwrap_or("").to_string();
            let text = it.next().unwrap_or("").to_string();
            if user.is_empty() || text.is_empty() {
                Some(SlashCommand::Unknown(line.to_string()))
            } else {
                Some(SlashCommand::Msg { user, text })
            }
        }
        _ => Some(SlashCommand::Unknown(line.to_string())),
    }
}

/// Percent-encode a URL path or query component so channel names like
/// `#general` are not parsed as URL fragments.
pub fn url_encode(s: &str) -> String {
    let mut out = String::new();
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'.' | b'_' | b'~' => {
                out.push(b as char);
            }
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

/// Extract the channel name from a websocket request path of the form
/// `/ws/<channel>`. Returns `None` for paths that don't match. The channel is
/// percent-decoded so names containing spaces or reserved characters work.
pub fn parse_ws_path(path: &str) -> Option<String> {
    let path = path.split('?').next().unwrap_or(path);
    let mut parts = path.split('/').filter(|s| !s.is_empty());
    if parts.next()? != "ws" {
        return None;
    }
    let channel = parts.next()?;
    if channel.is_empty() {
        None
    } else {
        Some(url_decode(channel))
    }
}

/// Extract a percent-decoded query parameter from a request path.
fn parse_query_param(path: &str, key: &str) -> Option<String> {
    let query = path.split('?').nth(1)?;
    for pair in query.split('&') {
        let mut kv = pair.splitn(2, '=');
        if kv.next()? == key {
            let v = kv.next()?;
            if !v.is_empty() {
                return Some(url_decode(v));
            }
        }
    }
    None
}

/// Forward channel broadcasts to the websocket sink until the channel is
/// closed or a `KICKED <channel>` control line matching this bridge's channel
/// is received (meaning the user was removed from the channel). Returns when
/// forwarding should stop.
async fn forward_loop<S>(
    sink: &mut S,
    rx: &mut tokio::sync::mpsc::Receiver<String>,
    channel: &str,
    kicked: &std::sync::Arc<std::sync::atomic::AtomicBool>,
) where
    S: futures_util::Sink<Message> + Unpin,
    S::Error: std::fmt::Debug,
{
    while let Some(line) = rx.recv().await {
        // A `KICKED <channel>` control line means this bridge was removed from
        // the channel (e.g. a moderator kicked the user). Stop forwarding and
        // let the connection close so the bridge no longer receives channel
        // traffic.
        if let Some(kicked_ch) = line.strip_prefix("KICKED ")
            && kicked_ch.trim() == channel
        {
            tracing::warn!("forward_loop: KICKED for {channel}");
            kicked.store(true, std::sync::atomic::Ordering::SeqCst);
            break;
        }
        if sink.send(Message::Text(line)).await.is_err() {
            tracing::warn!("forward_loop: sink.send failed for {channel}");
            break;
        }
    }
}

/// Decide whether a bridge client may connect and, if so, which username it is
/// authenticated as. A non-empty `ticket` is redeemed against the in-memory
/// store and yields the username it was minted for. Returns `None` when the
/// credentials do not check out.
///
/// The bridge authenticates *only* via a ticket minted by `POST /auth` (or
/// `POST /register`). A raw `password=` query string is deliberately not
/// accepted: a password in a URL ends up in access logs, proxy logs and
/// `Referer` headers, and the ticket path already covers every legitimate
/// client (the embedded page and any script that does one HTTP POST first).
async fn check_bridge_credentials(state: &SharedState, ticket: &str) -> Option<String> {
    if ticket.is_empty() {
        return None;
    }
    state.consume_ticket(ticket).await
}

/// Outcome of a slash command on the bridge.
enum SlashResult {
    /// Keep the connection; identity unchanged.
    Keep,
    /// Keep the connection; the user was renamed.
    Renamed(String),
    /// Close the connection (`/quit`, or the outbound queue died).
    Close,
}

/// Handle a slash command from a bridge client.
///
/// The bridge is publish-only, so commands that need a reply (history, topic,
/// help) write their answer back through `tx` as ordinary frames the page
/// renders as system lines. Commands that change state (join/part/nick/topic)
/// go through the same DB and broadcast paths the line protocol uses, so a
/// browser client and an IRC client see the same result.
async fn handle_slash_command(
    state: &SharedState,
    cmd: &SlashCommand,
    channel: &str,
    username: &str,
    user_id: i64,
    tx: &tokio::sync::mpsc::Sender<String>,
) -> SlashResult {
    let reply = |s: String| async move { tx.send(s).await.is_ok() };
    match cmd {
        SlashCommand::Join { channel: target } => {
            if !crate::protocol::valid_channel(target) {
                reply(format!("* invalid channel {target}\r\n")).await;
                return SlashResult::Keep;
            }
            let db = state.db.clone();
            let ch = target.clone();
            let uid = user_id;
            let result = tokio::task::spawn_blocking(move || {
                let ch = db.create_channel(&ch)?;
                db.join_channel(ch.id, uid)?;
                Ok::<_, rusqlite::Error>(ch)
            })
            .await;
            match result {
                Ok(Ok(_)) => {
                    state.unsubscribe(target, tx).await;
                    let count = state.subscribe(target, uid, tx.clone(), None).await;
                    reply(format!("* joined {target} ({count} online)\r\n")).await;
                    let line = format!("JOIN {target} {username}\r\n");
                    state.broadcast(target, line).await;
                }
                _ => {
                    reply(format!("* could not join {target}\r\n")).await;
                }
            }
            SlashResult::Keep
        }
        SlashCommand::Part { channel: target } => {
            let target = target.as_deref().unwrap_or(channel);
            // Validate the target before it is used in `unsubscribe` and
            // `broadcast`, so a crafted `/part` cannot inject control
            // characters into the PART line fanned out to subscribers.
            if !crate::protocol::valid_channel(target) {
                reply(format!("* invalid channel {target}\r\n")).await;
                return SlashResult::Keep;
            }
            let db = state.db.clone();
            let ch = target.to_string();
            let uid = user_id;
            let result = tokio::task::spawn_blocking(move || {
                let Some(ch) = db.find_channel(&ch)? else {
                    return Ok::<_, rusqlite::Error>(None);
                };
                db.part_channel(ch.id, uid)?;
                Ok(Some(()))
            })
            .await;
            match result {
                Ok(Ok(Some(()))) => {
                    state.unsubscribe(target, tx).await;
                    reply(format!("* parted {target}\r\n")).await;
                    let line = format!("PART {target} {username}\r\n");
                    state.broadcast(target, line).await;
                }
                _ => {
                    reply(format!("* not in channel {target}\r\n")).await;
                }
            }
            SlashResult::Keep
        }
        SlashCommand::Nick { name } => {
            if !crate::protocol::valid_username(name) {
                reply("* invalid nick\r\n".to_string()).await;
                return SlashResult::Keep;
            }
            let db = state.db.clone();
            let uid = user_id;
            let new_name = name.clone();
            let result = tokio::task::spawn_blocking(move || db.rename_user(uid, &new_name)).await;
            match result {
                Ok(Ok(())) => {
                    let old = username.to_string();
                    state.rename_user_connections(&old, name).await;
                    // Re-key this bridge's own per-user and per-channel rate
                    // limiters to the new name. `rename_user_connections` moves
                    // the *shared* per-user limiter (so the renamed user does not
                    // get a fresh bucket on their other connections), but this
                    // bridge's per-channel limiters are keyed by the local
                    // `username` and would otherwise leak an entry under the old
                    // name and start a fresh bucket under the new one.
                    state.rename_channel_limiters(&old, name).await;
                    reply(format!("* nick changed to {name}\r\n")).await;
                    let db = state.db.clone();
                    let uid = user_id;
                    let old_name = old.clone();
                    let new_name = name.clone();
                    let channels = tokio::task::spawn_blocking(move || db.user_channels(uid)).await;
                    if let Ok(Ok(channels)) = channels {
                        for ch in channels {
                            let line = format!("NICK {old_name} {new_name}\r\n");
                            state.broadcast(&ch, line).await;
                        }
                    }
                    SlashResult::Renamed(name.clone())
                }
                _ => {
                    reply("* nick taken\r\n".to_string()).await;
                    SlashResult::Keep
                }
            }
        }
        SlashCommand::Topic { text } => {
            let db = state.db.clone();
            let ch = channel.to_string();
            let uid = user_id;
            let is_set = text.is_some();
            if let Some(t) = text
                && !crate::protocol::valid_topic(t)
            {
                reply("* invalid topic\r\n".to_string()).await;
                return SlashResult::Keep;
            }
            let text = text.clone();
            let result = tokio::task::spawn_blocking(move || {
                let Some(ch) = db.find_channel(&ch)? else {
                    return Ok::<_, rusqlite::Error>(None);
                };
                if !db.is_member(ch.id, uid)? {
                    return Ok(Some(Err("not in channel".to_string())));
                }
                match text {
                    Some(t) => {
                        if !db.is_op(ch.id, uid)? {
                            return Ok(Some(Err("not an operator".to_string())));
                        }
                        db.set_topic(ch.id, &t)?;
                        Ok(Some(Ok(t.clone())))
                    }
                    None => Ok(Some(Ok(db.get_topic(ch.id)?.unwrap_or_default()))),
                }
            })
            .await;
            match result {
                Ok(Ok(Some(Ok(topic)))) => {
                    reply(format!("* topic: {topic}\r\n")).await;
                    if is_set {
                        let line = format!("TOPIC {channel} {topic}\r\n");
                        state.broadcast(channel, line).await;
                    }
                }
                Ok(Ok(Some(Err(msg)))) => {
                    reply(format!("* {msg}\r\n")).await;
                }
                _ => {
                    reply(format!("* no such channel {channel}\r\n")).await;
                }
            }
            SlashResult::Keep
        }
        SlashCommand::History { limit } => {
            let db = state.db.clone();
            let ch = channel.to_string();
            let uid = user_id;
            let limit = *limit;
            let result = tokio::task::spawn_blocking(move || {
                let Some(ch) = db.find_channel(&ch)? else {
                    return Ok::<_, rusqlite::Error>(None);
                };
                if !db.is_member(ch.id, uid)? {
                    return Ok(Some(Err(())));
                }
                Ok(Some(Ok(db.channel_history_before(
                    ch.id,
                    limit as i64,
                    None,
                )?)))
            })
            .await;
            match result {
                Ok(Ok(Some(Ok(msgs)))) => {
                    for m in msgs {
                        let line = format!("PRIVMSG {channel} {} :{}\r\n", m.username, m.body);
                        if tx.send(line).await.is_err() {
                            return SlashResult::Close;
                        }
                    }
                }
                _ => {
                    reply(format!("* no history for {channel}\r\n")).await;
                }
            }
            SlashResult::Keep
        }
        SlashCommand::Help => {
            reply(
                "* commands: /join <ch> /part [ch] /nick <name> /topic [text] \
                 /history [n] /msg <user> <text> /me <action> /help /quit\r\n"
                    .to_string(),
            )
            .await;
            SlashResult::Keep
        }
        SlashCommand::Quit => SlashResult::Close,
        SlashCommand::Me { text } => {
            if !crate::protocol::valid_message(text) {
                reply("* invalid action\r\n".to_string()).await;
                return SlashResult::Keep;
            }
            let line = format!("ACTION {channel} {username} :{text}\r\n");
            state.broadcast(channel, line).await;
            SlashResult::Keep
        }
        SlashCommand::Msg { user, text } => {
            if !crate::protocol::valid_message(text) {
                reply("* invalid message\r\n".to_string()).await;
                return SlashResult::Keep;
            }
            if user.eq_ignore_ascii_case(username) {
                reply("* cannot message yourself\r\n".to_string()).await;
                return SlashResult::Keep;
            }
            let db = state.db.clone();
            let target = user.clone();
            let result = tokio::task::spawn_blocking(move || db.find_user(&target)).await;
            match result {
                Ok(Ok(Some((recipient, _)))) => {
                    let line = format!("PRIVMSG {user} {username} :{text}\r\n");
                    let delivered = state.dm(&recipient.username, line).await;
                    // Persist the DM so the recipient's history includes it. A
                    // message that was *not* delivered (recipient offline) is
                    // not stored: there is no live recipient to read it, and
                    // storing it would let a sender silently fill the DB with
                    // messages nobody ever sees.
                    if delivered {
                        let db = state.db.clone();
                        let from_id = user_id;
                        let to_id = recipient.id;
                        let body = text.clone();
                        let _ =
                            tokio::task::spawn_blocking(move || db.store_dm(from_id, to_id, &body))
                                .await;
                    }
                    reply(format!("* sent to {user}\r\n")).await;
                }
                _ => {
                    reply(format!("* no such user {user}\r\n")).await;
                }
            }
            SlashResult::Keep
        }
        SlashCommand::Unknown(line) => {
            reply(format!("* unknown command: {line}\r\n")).await;
            SlashResult::Keep
        }
    }
}

#[allow(clippy::result_large_err)] // the handshake closure's error type is fixed by tungstenite
pub async fn handle_ws<S>(stream: S, state: SharedState, peer: String)
where
    S: AsyncRead + AsyncWrite + Unpin + Send + 'static,
{
    let path = std::sync::Mutex::new(String::new());
    // Cap inbound frames while tungstenite reads them. The read loop below drops
    // messages larger than MAX_WS_MESSAGE, but only once the whole frame has been
    // buffered, so a single huge frame could otherwise make the server allocate an
    // arbitrary amount of memory. `max_frame_size` aborts the read instead and the
    // client gets a normal close (1009, "message too big").
    let ws_config = tokio_tungstenite::tungstenite::protocol::WebSocketConfig {
        max_message_size: Some(MAX_WS_MESSAGE),
        max_frame_size: Some(MAX_WS_MESSAGE),
        ..Default::default()
    };
    let ws = match tokio_tungstenite::accept_hdr_async_with_config(
        stream,
        |req: &Request, response: Response| {
            if let Some(pq) = req.uri().path_and_query() {
                *path.lock().unwrap() = pq.as_str().to_string();
            }
            Ok(response)
        },
        Some(ws_config),
    )
    .await
    {
        Ok(ws) => ws,
        Err(e) => {
            tracing::warn!(%peer, "ws handshake failed: {e}");
            return;
        }
    };

    let path = path.into_inner().unwrap_or_default();
    tracing::debug!(%peer, "ws path: {path}");
    let channel = parse_ws_path(&path).unwrap_or_default();
    let ticket = parse_query_param(&path, "ticket").unwrap_or_default();
    // How many past messages to replay on connect. The bridge is publish-only, so
    // without this a page reload would start with an empty log.
    let backfill: i64 = parse_query_param(&path, "history")
        .and_then(|n| n.parse::<i64>().ok())
        .unwrap_or(0)
        .clamp(0, MAX_WS_BACKFILL);
    if channel.is_empty() {
        tracing::warn!(%peer, "ws rejected: missing channel");
        return;
    }
    if ticket.is_empty() {
        tracing::warn!(%peer, "ws rejected: no ticket");
        return;
    }
    // Validate the channel against the same rules the IRC path enforces, so a
    // crafted path cannot inject control characters or ':' into the PRIVMSG
    // lines broadcast to subscribers.
    if !crate::protocol::valid_channel(&channel) {
        tracing::warn!(%peer, "ws rejected: invalid channel name");
        return;
    }

    /// Outcome of the blocking credential/channel check. Kept distinct so the
    /// log says *why* a bridge was refused instead of reporting every case as an
    /// authentication failure.
    enum WsAuth {
        BadCredentials,
        Ok {
            user_id: i64,
            username: String,
            channel_id: i64,
        },
    }

    // Credentials: a minted ticket (from `POST /auth` or `POST /register`) is
    // checked in memory, so the browser path never pays for Argon2. The returned
    // username is the *authenticated* identity: the name the ticket was minted
    // for (not whatever the URL claims), so a ticket minted for `alice` cannot
    // be replayed as `bob`.
    let Some(user) = check_bridge_credentials(&state, &ticket).await else {
        tracing::warn!(%peer, "ws rejected: bad credentials");
        return;
    };

    let db = state.db.clone();
    let uname = user.clone();
    let ch = channel.clone();
    let auth = tokio::task::spawn_blocking(move || {
        let Some((user, _hash)) = db.find_user(&uname)? else {
            return Ok::<_, rusqlite::Error>(WsAuth::BadCredentials);
        };
        // Connecting the bridge is equivalent to `JOIN` on the line protocol:
        // resolve (or create) the channel and add the user to it. This used to
        // require pre-existing membership, which made the bridge unusable on its
        // own -- membership is dropped when the TCP session that created it
        // disconnects, so a browser-only client could never connect. Joining here
        // grants no new privilege: the same user could `JOIN` over TCP with the
        // same credentials, and the password check above is the actual control.
        let ch = db.create_channel(&ch)?;
        db.join_channel(ch.id, user.id)?;
        Ok(WsAuth::Ok {
            user_id: user.id,
            username: user.username,
            channel_id: ch.id,
        })
    })
    .await
    .unwrap_or(Ok(WsAuth::BadCredentials))
    .unwrap_or(WsAuth::BadCredentials);
    let (user_id, mut username, channel_id) = match auth {
        WsAuth::Ok {
            user_id,
            username,
            channel_id,
        } => (user_id, username, channel_id),
        WsAuth::BadCredentials => {
            tracing::warn!(%peer, "ws rejected: bad credentials for user '{user}'");
            return;
        }
    };

    let (sink, mut stream) = ws.split();

    let (tx, mut rx) = tokio::sync::mpsc::channel::<String>(crate::state::OUTBOUND_CAPACITY);
    let writer_alive = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(true));
    let shutdown = std::sync::Arc::new(tokio::sync::Notify::new());

    // Seed the client's member roster with the current membership (and operator
    // status) so the sidebar is correct immediately, before any JOIN/QUIT/NICK
    // broadcast arrives. Sent as `NAMES <channel> <user> <op>` control lines the
    // page renders into the roster.
    {
        let db = state.db.clone();
        let ch_id = channel_id;
        let roster =
            match tokio::task::spawn_blocking(move || db.channel_members_with_ops(ch_id)).await {
                Ok(Ok(rows)) => rows,
                _ => Vec::new(),
            };
        for (name, is_op) in roster {
            let line = format!("NAMES {channel} {name} {}\r\n", if is_op { 1 } else { 0 });
            if tx.try_send(line).is_err() {
                break;
            }
        }
    }

    // Replay recent history into the connection's own queue before the forward
    // loop starts draining it. These frames are plain message bodies, matching
    // what the page appends for live traffic, so a reload no longer starts with
    // an empty log. Bounded by MAX_WS_BACKFILL and skipped when not requested.
    if backfill > 0 {
        let db = state.db.clone();
        let replay = match tokio::task::spawn_blocking(move || {
            db.channel_history_before(channel_id, backfill, None)
        })
        .await
        {
            Ok(Ok(msgs)) => msgs,
            Ok(Err(e)) => {
                tracing::warn!(%peer, "ws backfill failed: {e}");
                Vec::new()
            }
            Err(e) => {
                tracing::warn!(%peer, "ws backfill task failed: {e}");
                Vec::new()
            }
        };
        for msg in replay {
            let line = format!("PRIVMSG {} {} :{}\r\n", channel, msg.username, msg.body);
            // Bounded queue: if the client is already behind, stop replaying
            // rather than dropping live traffic.
            if tx.try_send(line).is_err() {
                break;
            }
        }
    }

    let forward_channel = channel.clone();
    let kicked = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let forward_kicked = kicked.clone();
    let forward_alive = writer_alive.clone();
    let forward = tokio::spawn(async move {
        let mut sink = sink;
        tracing::info!("forward task started for {forward_channel}");
        forward_loop(&mut sink, &mut rx, &forward_channel, &forward_kicked).await;
        tracing::info!("forward task ended for {forward_channel}");
        forward_alive.store(false, std::sync::atomic::Ordering::SeqCst);
    });
    let kill = crate::state::ConnKill::new(
        writer_alive.clone(),
        forward.abort_handle(),
        shutdown.clone(),
    );
    state
        .subscribe(&channel, user_id, tx.clone(), Some(kill.clone()))
        .await;
    // Register this bridge in the per-user map so direct messages (and the
    // server's targeted `KICKED` control line) reach it. The bridge can already
    // *send* PRIVMSG to a user, so without this it could be messaged but never
    // receive a reply. `tx` feeds the same forward loop as channel traffic.
    state
        .register_user(&username, tx.clone(), Some(kill.clone()))
        .await;

    loop {
        if kicked.load(std::sync::atomic::Ordering::SeqCst)
            || !writer_alive.load(std::sync::atomic::Ordering::SeqCst)
        {
            break;
        }
        let notified = shutdown.notified();
        tokio::pin!(notified);
        if !writer_alive.load(std::sync::atomic::Ordering::SeqCst) {
            break;
        }
        let msg = tokio::select! {
            _ = notified => break,
            // Idle timeout: a bridge that sends nothing for WS_IDLE_TIMEOUT is
            // closed. The page is a passive viewer, so there is no PING/PONG
            // probe — an idle tab is simply torn down and the client reconnects
            // on the next interaction.
            _ = tokio::time::sleep(WS_IDLE_TIMEOUT) => {
                tracing::debug!(%peer, "ws idle timeout");
                break;
            }
            msg = stream.next() => msg,
        };
        let Some(msg) = msg else {
            break;
        };
        // If the forward loop saw a KICKED control line for this channel, the
        // user was removed — stop the bridge so it stops receiving channel
        // traffic and closes.
        if kicked.load(std::sync::atomic::Ordering::SeqCst) {
            break;
        }
        match msg {
            Ok(Message::Text(text)) => {
                if text.len() > MAX_WS_MESSAGE {
                    continue;
                }
                // A message body is echoed verbatim into `PRIVMSG <channel>
                // <user> :<body>` fan-out lines, so reject control characters
                // (notably `\r`/`\n`) that would smuggle extra protocol lines
                // into every subscriber's stream. This mirrors the TCP path's
                // `valid_message` check.
                if !crate::protocol::valid_message(&text) {
                    continue;
                }
                // Slash commands let the browser drive the protocol (join/part/
                // nick/topic/history/…). They are handled here rather than sent
                // as channel messages, so a `/join` typed in the composer does
                // not get broadcast as a literal line.
                if let Some(cmd) = parse_slash_command(&text) {
                    match handle_slash_command(&state, &cmd, &channel, &username, user_id, &tx)
                        .await
                    {
                        SlashResult::Renamed(new_name) => {
                            username = new_name;
                            continue;
                        }
                        SlashResult::Keep => continue,
                        SlashResult::Close => break,
                    }
                }
                // Rate limit against the shared per-user limiter so opening
                // many WS connections cannot bypass throttling.
                let allowed = {
                    let mut limiters = state.rate_limiters.lock().await;
                    limiters
                        .entry(username.clone())
                        .or_insert_with(|| RateLimiter::new(20.0, 10.0))
                        .allow()
                };
                if !allowed {
                    continue;
                }
                // Per-channel flood control, mirroring the TCP path: a user can
                // post at the per-user rate across all channels, but this bounds
                // how fast they can flood a single channel.
                let allowed = {
                    let mut limiters = state.channel_limiters.lock().await;
                    limiters
                        .entry((channel.clone(), username.clone()))
                        .or_insert_with(|| RateLimiter::new(10.0, 5.0))
                        .allow()
                };
                if !allowed {
                    continue;
                }
                // Require the user to still be a member of the channel. A
                // kicked user's bridge must not keep posting: the IRC path
                // checks membership before sending, so the WS bridge must too.
                let db = state.db.clone();
                let body = text.to_string();
                let stored = tokio::task::spawn_blocking(move || {
                    if !db.is_member(channel_id, user_id)? {
                        return Ok::<_, rusqlite::Error>(false);
                    }
                    db.store_message(channel_id, user_id, &body)?;
                    Ok(true)
                })
                .await;
                match stored {
                    Ok(Ok(true)) => {}
                    // Not a member anymore (e.g. kicked): stop the bridge.
                    _ => break,
                }
                let line = format!("PRIVMSG {} {} :{}\r\n", channel, username, text);
                state.broadcast(&channel, line).await;
            }
            Ok(Message::Close(_)) => break,
            // Protocol errors, including the oversize frame that `ws_config`
            // rejects mid-read, surface here as `Capacity`.
            Err(e) => {
                // A client that vanishes mid-frame is routine churn, not a
                // server-side fault, so keep it out of the log.
                if !is_client_disconnect(&e) {
                    tracing::debug!(%peer, "ws connection error: {e}");
                }
                break;
            }
            _ => {}
        }
    }

    state.unsubscribe(&channel, &tx).await;
    state.unregister_user(&username, &tx).await;
    // Drop the per-user rate limiter so the map does not leak an entry for
    // every username that ever opened a bridge. Recreated lazily on next message.
    state.remove_rate_limiter(&username).await;
    // Drop the per-channel limiters too, so the map does not leak an entry for
    // every (channel, user) pair that ever posted over a bridge.
    state.remove_channel_limiters(&username).await;
    // Mirror the TCP disconnect cleanup: if no other live connection for this
    // user is still subscribed to the channel, remove the DB membership row so
    // the user does not linger in NAMES/LIST counts or keep HISTORY access after
    // the bridge closes. A still-live connection (e.g. an IRC session) keeps the
    // membership, matching the TCP path's `still_subscribed` guard.
    if !state.still_subscribed(&channel, user_id, &tx).await {
        let db = state.db.clone();
        let ch_name = channel.clone();
        let ch_name_for_task = ch_name.clone();
        if let Err(e) = tokio::task::spawn_blocking(move || {
            if let Some(ch) = db.find_channel(&ch_name_for_task)? {
                db.part_channel(ch.id, user_id)?;
            }
            Ok::<_, rusqlite::Error>(())
        })
        .await
        {
            tracing::warn!("ws part_channel task failed for {ch_name}: {e}");
        }
    }
    forward.abort();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_ws_path() {
        assert_eq!(parse_ws_path("/ws/general"), Some("general".to_string()));
        assert_eq!(parse_ws_path("/ws/"), None);
        assert_eq!(parse_ws_path("/other/general"), None);
        assert_eq!(
            parse_ws_path("/ws/general?user=bob"),
            Some("general".to_string())
        );
    }

    #[test]
    fn url_encodes_channel_sigil() {
        assert_eq!(url_encode("#general"), "%23general");
        assert_eq!(
            parse_ws_path("/ws/%23general"),
            Some("#general".to_string())
        );
    }

    #[test]
    fn url_decodes_channel() {
        // Percent-encoded space and plus sign in the channel name.
        assert_eq!(parse_ws_path("/ws/my%20room"), Some("my room".to_string()));
        assert_eq!(parse_ws_path("/ws/a+b"), Some("a b".to_string()));
    }

    #[test]
    fn url_decode_leaves_invalid_escapes_alone() {
        assert_eq!(url_decode("100%"), "100%");
        assert_eq!(url_decode("%zz"), "%zz");
        assert_eq!(url_decode("a%2"), "a%2");
    }

    #[test]
    fn parses_slash_commands() {
        assert_eq!(
            parse_slash_command("/join #general"),
            Some(SlashCommand::Join {
                channel: "#general".into()
            })
        );
        assert_eq!(
            parse_slash_command("/part"),
            Some(SlashCommand::Part { channel: None })
        );
        assert_eq!(
            parse_slash_command("/part #other"),
            Some(SlashCommand::Part {
                channel: Some("#other".into())
            })
        );
        assert_eq!(
            parse_slash_command("/nick bob"),
            Some(SlashCommand::Nick { name: "bob".into() })
        );
        assert_eq!(
            parse_slash_command("/topic welcome all"),
            Some(SlashCommand::Topic {
                text: Some("welcome all".into())
            })
        );
        assert_eq!(
            parse_slash_command("/topic"),
            Some(SlashCommand::Topic { text: None })
        );
        assert_eq!(
            parse_slash_command("/history 20"),
            Some(SlashCommand::History { limit: 20 })
        );
        assert_eq!(
            parse_slash_command("/history"),
            Some(SlashCommand::History { limit: 50 })
        );
        assert_eq!(parse_slash_command("/help"), Some(SlashCommand::Help));
        assert_eq!(parse_slash_command("/quit"), Some(SlashCommand::Quit));
        assert_eq!(
            parse_slash_command("/me waves"),
            Some(SlashCommand::Me {
                text: "waves".into()
            })
        );
        assert_eq!(
            parse_slash_command("/msg bob hi there"),
            Some(SlashCommand::Msg {
                user: "bob".into(),
                text: "hi there".into()
            })
        );
        // A line that does not start with '/' is not a command.
        assert_eq!(parse_slash_command("hello"), None);
        // An unknown verb is preserved for the UI to echo.
        assert_eq!(
            parse_slash_command("/bogus stuff"),
            Some(SlashCommand::Unknown("/bogus stuff".into()))
        );
    }

    #[test]
    fn slash_command_history_limit_is_clamped() {
        assert_eq!(
            parse_slash_command("/history 99999"),
            Some(SlashCommand::History {
                limit: crate::protocol::MAX_HISTORY
            })
        );
        assert_eq!(
            parse_slash_command("/history 0"),
            Some(SlashCommand::History { limit: 1 })
        );
    }

    /// A minimal sink that records the text payloads of every message sent to
    /// it, so tests can assert on what a websocket bridge forwarded.
    #[derive(Default)]
    struct RecordingSink {
        sent: Vec<String>,
    }

    impl futures_util::Sink<Message> for RecordingSink {
        type Error = std::io::Error;

        fn poll_ready(
            self: std::pin::Pin<&mut Self>,
            _cx: &mut std::task::Context<'_>,
        ) -> std::task::Poll<Result<(), Self::Error>> {
            std::task::Poll::Ready(Ok(()))
        }

        fn start_send(self: std::pin::Pin<&mut Self>, item: Message) -> Result<(), Self::Error> {
            if let Message::Text(text) = item {
                self.get_mut().sent.push(text.to_string());
            }
            Ok(())
        }

        fn poll_flush(
            self: std::pin::Pin<&mut Self>,
            _cx: &mut std::task::Context<'_>,
        ) -> std::task::Poll<Result<(), Self::Error>> {
            std::task::Poll::Ready(Ok(()))
        }

        fn poll_close(
            self: std::pin::Pin<&mut Self>,
            _cx: &mut std::task::Context<'_>,
        ) -> std::task::Poll<Result<(), Self::Error>> {
            std::task::Poll::Ready(Ok(()))
        }
    }

    #[tokio::test]
    async fn forward_loop_stops_after_kicked_for_own_channel() {
        let (tx, mut rx) = tokio::sync::mpsc::channel::<String>(16);
        let mut sink = RecordingSink::default();

        // Broadcast a normal message, then a KICKED control line for this
        // bridge's channel. The loop should forward the first and stop on the
        // second, so nothing after it is forwarded.
        tx.send("hello".to_string()).await.unwrap();
        tx.send("KICKED general".to_string()).await.unwrap();
        tx.send("should-not-arrive".to_string()).await.unwrap();
        drop(tx);

        let kicked = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        forward_loop(&mut sink, &mut rx, "general", &kicked).await;

        assert_eq!(sink.sent, vec!["hello".to_string()]);
        assert!(kicked.load(std::sync::atomic::Ordering::SeqCst));
    }

    #[tokio::test]
    async fn forward_loop_ignores_kicked_for_other_channel() {
        let (tx, mut rx) = tokio::sync::mpsc::channel::<String>(16);
        let mut sink = RecordingSink::default();

        // A KICKED line for a *different* channel must not stop this bridge.
        tx.send("KICKED other".to_string()).await.unwrap();
        tx.send("still-here".to_string()).await.unwrap();
        drop(tx);

        let kicked = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        forward_loop(&mut sink, &mut rx, "general", &kicked).await;

        // A KICKED line for another channel is just a normal broadcast from
        // this bridge's perspective, so it is forwarded along with the rest.
        assert_eq!(
            sink.sent,
            vec!["KICKED other".to_string(), "still-here".to_string()]
        );
        assert!(!kicked.load(std::sync::atomic::Ordering::SeqCst));
    }
}
