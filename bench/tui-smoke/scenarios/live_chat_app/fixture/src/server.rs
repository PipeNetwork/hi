use std::time::{Duration, Instant};
use tokio::io::{AsyncBufRead, AsyncBufReadExt, AsyncRead, AsyncWrite, AsyncWriteExt, BufReader};

use crate::protocol::{Command, parse_line};
use crate::state::{Session, SharedState};

/// Idle timeout: if a client sends nothing for this long, we send a PING and
/// disconnect if it doesn't respond with PONG. Prevents dead connections from
/// holding a task and socket forever.
const IDLE_TIMEOUT: Duration = Duration::from_secs(300);
/// How long to wait for a PONG after sending PING before disconnecting.
const PONG_TIMEOUT: Duration = Duration::from_secs(30);

/// Result of [`read_line_capped`].
enum ReadOutcome {
    /// A complete line sits in the caller's buffer.
    Line,
    /// The peer sent a line longer than the cap. The rest of that line was
    /// drained and discarded, so the next read starts at the next line.
    TooLong,
    /// The peer closed the connection.
    Eof,
}

/// Read one LF-terminated line into `buf` without ever buffering more than
/// `max` bytes plus a single buffered chunk.
///
/// `BufReader::read_line` keeps growing its target until it finds a newline, so
/// a client that simply never sends one can drive the server's memory up without
/// bound. Checking `line.len()` afterwards cannot help: by then the whole line is
/// already allocated. This reads one buffered chunk at a time and, once the cap
/// is exceeded, switches to discarding the remainder so memory stays bounded and
/// the stream stays aligned with the next line boundary.
async fn read_line_capped<R: AsyncBufRead + Unpin>(
    reader: &mut R,
    buf: &mut Vec<u8>,
    max: usize,
) -> std::io::Result<ReadOutcome> {
    buf.clear();
    let mut overflowed = false;
    loop {
        let available = match reader.fill_buf().await {
            Ok(chunk) => chunk,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e),
        };
        if available.is_empty() {
            // EOF. A partial trailing line is still handed to the caller, which
            // matches what `read_line` reported before.
            return Ok(if overflowed || buf.is_empty() {
                ReadOutcome::Eof
            } else {
                ReadOutcome::Line
            });
        }
        let newline = available.iter().position(|b| *b == b'\n');
        let used = newline.map_or(available.len(), |i| i + 1);
        if !overflowed {
            buf.extend_from_slice(&available[..used]);
            if buf.len() > max {
                overflowed = true;
                buf.clear();
            }
        }
        reader.consume(used);
        if newline.is_some() {
            return Ok(if overflowed {
                ReadOutcome::TooLong
            } else {
                ReadOutcome::Line
            });
        }
    }
}

/// The moderation action a connection requested, resolved from the parsed
/// command so it can be moved into blocking closures without re-borrowing.
#[derive(Clone, Copy)]
enum ModAction {
    Kick,
    Op,
    Deop,
}

/// Simple per-connection token bucket for flood control: allows a burst of
/// `burst` lines, then refills at `rate` lines per second.
#[derive(Clone)]
pub struct RateLimiter {
    tokens: f64,
    last: Instant,
    burst: f64,
    rate: f64,
}

impl RateLimiter {
    pub fn new(burst: f64, rate: f64) -> Self {
        RateLimiter {
            tokens: burst,
            last: Instant::now(),
            burst,
            rate,
        }
    }

    /// Returns `true` if a line is allowed, `false` if it should be rejected.
    pub fn allow(&mut self) -> bool {
        let now = Instant::now();
        let elapsed = now.duration_since(self.last).as_secs_f64();
        self.last = now;
        self.tokens = (self.tokens + elapsed * self.rate).min(self.burst);
        if self.tokens >= 1.0 {
            self.tokens -= 1.0;
            true
        } else {
            false
        }
    }

    /// Whether the bucket has refilled to full, i.e. the peer has been idle
    /// long enough that keeping the entry around buys nothing. Used to sweep
    /// shared limiter maps so they stay bounded by active peers.
    pub fn is_full(&mut self) -> bool {
        let now = Instant::now();
        let elapsed = now.duration_since(self.last).as_secs_f64();
        self.tokens + elapsed * self.rate >= self.burst
    }
}

/// Deliver any DMs that were stored while `user` was offline, then mark them
/// delivered. Called right after a successful login so a user sees messages
/// they missed. Runs on the blocking pool since it touches the DB.
async fn deliver_undelivered_dms(
    state: &SharedState,
    user: &crate::db::User,
    tx: &tokio::sync::mpsc::Sender<String>,
) {
    let db = state.db.clone();
    let uid = user.id;
    let dms = match tokio::task::spawn_blocking(move || db.undelivered_dms(uid)).await {
        Ok(Ok(dms)) => dms,
        _ => return,
    };
    for (from, body) in dms {
        // The first field is the *recipient* (the user who just logged in), the
        // second is the sender — matching the live DM format `PRIVMSG <target>
        // <sender> :<body>` used in the Privmsg handler. Using `from` for both
        // would mislabel the target and break the web UI's sender parsing.
        let line = format!("PRIVMSG {} {} :{}\r\n", user.username, from, body);
        if tx.send(line).await.is_err() {
            return;
        }
    }
    let db = state.db.clone();
    let uid = user.id;
    let uname = user.username.clone();
    if let Err(e) = tokio::task::spawn_blocking(move || db.mark_dms_delivered(uid)).await {
        tracing::warn!("mark_dms_delivered task failed for {uname}: {e}");
    }
}

/// Wait for `fut` unless this connection has been killed as a slow consumer.
/// Returns `None` when `kill` trips (or already has), so the handler can
/// break out of a blocking read instead of waiting for the idle timer.
async fn wait_line_or_kill<F, T>(
    shutdown: &tokio::sync::Notify,
    alive: &std::sync::atomic::AtomicBool,
    fut: F,
) -> Option<T>
where
    F: std::future::Future<Output = T>,
{
    let notified = shutdown.notified();
    tokio::pin!(notified);
    if !alive.load(std::sync::atomic::Ordering::SeqCst) {
        return None;
    }
    tokio::select! {
        _ = notified => None,
        result = fut => Some(result),
    }
}

/// Handle a single client connection until it disconnects.
pub async fn handle_connection<S>(stream: S, state: SharedState, peer_ip: std::net::IpAddr)
where
    S: AsyncRead + AsyncWrite + Unpin + Send + 'static,
{
    let (reader, writer) = tokio::io::split(stream);
    let mut reader = BufReader::new(reader);
    let mut line = String::new();
    // Raw bytes for the current line, reused across iterations.
    let mut raw: Vec<u8> = Vec::new();

    // Outbound queue for this connection. The broadcast fan-out writes into
    // this channel; a dedicated task drains it to the socket. Bounded so a
    // slow or disconnected client cannot grow memory without bound.
    let (tx, mut rx) = tokio::sync::mpsc::channel::<String>(crate::state::OUTBOUND_CAPACITY);

    // Spawn a writer task that drains the outbound queue to the socket. This
    // lets the reader loop and the broadcast fan-out both write without
    // interleaving partial lines. The flag flips to false when the writer dies
    // (socket write error), so the reader loop can stop instead of looping
    // forever on a dead connection.
    let writer_alive = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(true));
    let writer_alive_flag = writer_alive.clone();
    let writer_task = tokio::spawn(async move {
        let mut writer = writer;
        while let Some(msg) = rx.recv().await {
            if writer.write_all(msg.as_bytes()).await.is_err() || writer.flush().await.is_err() {
                writer_alive_flag.store(false, std::sync::atomic::Ordering::SeqCst);
                break;
            }
        }
    });
    let shutdown = std::sync::Arc::new(tokio::sync::Notify::new());
    let kill = crate::state::ConnKill::new(
        writer_alive.clone(),
        writer_task.abort_handle(),
        shutdown.clone(),
    );

    let mut session: Option<Session> = None;
    // Channels this connection is currently subscribed to, so we can clean up
    // on disconnect.
    let mut joined: Vec<String> = Vec::new();

    // Flood control: allow a burst of 20 lines, then 10 lines/second. Before
    // login we use a per-connection limiter; once authenticated we switch to a
    // per-user limiter shared across all of that user's connections so opening
    // many sockets cannot bypass throttling.
    let mut anon_limiter = RateLimiter::new(20.0, 10.0);
    let mut rate_key: Option<String> = None;

    // Direct replies go through the same outbound queue so ordering with
    // broadcast messages is preserved. `send().await` enqueues the line and
    // returns `false` if the writer task has died (receiver dropped), so the
    // caller can break out of the loop instead of looping forever on a dead
    // connection.
    let reply = |s: String| async { tx.send(s).await.is_ok() };

    reply("Welcome to chat. Type HELP for commands.\r\n".to_string()).await;

    loop {
        // If the writer task died (socket write error) or a slow-consumer
        // drop tripped `kill`, stop reading: the connection is gone and any
        // further replies would be dropped.
        if !writer_alive.load(std::sync::atomic::Ordering::SeqCst) {
            break;
        }
        line.clear();
        // Idle timeout: if the client sends nothing for IDLE_TIMEOUT, send a
        // PING and wait up to PONG_TIMEOUT for a PONG before disconnecting.
        // `kill.shutdown()` interrupts this wait when the fan-out path closes
        // a slow consumer, so we do not sit on the idle timer after the
        // writer has already been aborted.
        match wait_line_or_kill(
            kill.shutdown(),
            &writer_alive,
            tokio::time::timeout(
                IDLE_TIMEOUT,
                read_line_capped(&mut reader, &mut raw, crate::protocol::MAX_LINE_LEN),
            ),
        )
        .await
        {
            None => break,
            Some(Ok(Ok(ReadOutcome::Line))) => line.push_str(&String::from_utf8_lossy(&raw)),
            Some(Ok(Ok(ReadOutcome::TooLong))) => {
                reply("ERROR line too long\r\n".to_string()).await;
                continue;
            }
            Some(Ok(Ok(ReadOutcome::Eof))) => break,
            Some(Ok(Err(_))) => break,
            Some(Err(_elapsed)) => {
                // Idle: probe liveness with a PING. If the client is gone, the
                // writer task will fail to flush and we break out of the loop.
                reply("PING\r\n".to_string()).await;
                match wait_line_or_kill(
                    kill.shutdown(),
                    &writer_alive,
                    tokio::time::timeout(
                        PONG_TIMEOUT,
                        read_line_capped(&mut reader, &mut raw, crate::protocol::MAX_LINE_LEN),
                    ),
                )
                .await
                {
                    None => break,
                    Some(Ok(Ok(ReadOutcome::Eof))) | Some(Ok(Err(_))) | Some(Err(_)) => break,
                    Some(Ok(Ok(ReadOutcome::TooLong))) => {
                        reply("ERROR line too long\r\n".to_string()).await;
                        continue;
                    }
                    Some(Ok(Ok(ReadOutcome::Line))) => {
                        // Not necessarily a PONG: a client that has been idle can
                        // also have sent a real command that raced with our probe.
                        // Dropping it would lose the message silently, so hand
                        // anything that is not a PONG to the normal path below.
                        let text = String::from_utf8_lossy(&raw);
                        let text = text.trim();
                        if text.is_empty() || text.eq_ignore_ascii_case("PONG") {
                            continue;
                        }
                        line.push_str(text);
                    }
                }
            }
        }
        // Rate limit: use the per-user limiter once authenticated, otherwise
        // the anonymous per-connection limiter.
        let allowed = match &rate_key {
            Some(key) => {
                let mut limiters = state.rate_limiters.lock().await;
                let limiter = limiters
                    .entry(key.clone())
                    .or_insert_with(|| RateLimiter::new(20.0, 10.0));
                limiter.allow()
            }
            None => anon_limiter.allow(),
        };
        if !allowed {
            reply("ERROR rate limited\r\n".to_string()).await;
            continue;
        }

        // A `KICKED <channel>` control line means this connection was removed
        // from the channel by a moderator. Only the local `joined` bookkeeping
        // is updated here so we don't emit a spurious QUIT on disconnect and
        // stop acting as a member. This mirrors the WS bridge's `forward_loop`
        // handling.
        //
        // Crucially this must NOT call `unsubscribe`. The client can type this
        // line just as easily as the server can send it, so acting on it would
        // let any member silently drop its own fan-out entry — going deaf to
        // PRIVMSG while still counted as a member — with one forged line. The
        // real fan-out removal happens in the kick handler on the moderator's
        // connection via `unsubscribe_user`.
        if let Some(kicked) = line.trim().strip_prefix("KICKED ") {
            let ch = kicked.trim().to_string();
            joined.retain(|c| c != &ch);
            continue;
        }

        let cmd = parse_line(&line);
        match cmd {
            Command::Register { username, password } => {
                if session.is_some() {
                    reply("ERROR already logged in\r\n".to_string()).await;
                    continue;
                }
                // Per-IP throttle on the TCP path too, so an attacker cannot
                // open many sockets and spray registrations across them. The
                // Argon2 gate bounds the cost, but this stops the flood before
                // any hashing work is spent.
                if !state.allow_auth_attempt(peer_ip).await {
                    reply("ERROR rate limited\r\n".to_string()).await;
                    continue;
                }
                if !crate::protocol::valid_username(&username) {
                    reply("ERROR invalid username\r\n".to_string()).await;
                    continue;
                }
                // Hashing is expensive on purpose, so it runs on the blocking
                // pool and behind the shared gate that bounds how many Argon2
                // operations a flood of connections can start at once.
                let pw = password.clone();
                let hash = match state
                    .gated_hash(move || crate::auth::hash_password(&pw))
                    .await
                {
                    Ok(h) => h,
                    Err(_) => {
                        reply("ERROR hashing failed\r\n".to_string()).await;
                        continue;
                    }
                };
                let db = state.db.clone();
                let uname = username.clone();
                let result =
                    tokio::task::spawn_blocking(move || db.register_user(&uname, &hash)).await;
                match result {
                    Ok(Ok(user)) => {
                        session = Some(Session {
                            user_id: user.id,
                            username: user.username,
                        });
                        state
                            .register_user(&username, tx.clone(), Some(kill.clone()))
                            .await;
                        rate_key = Some(username.clone());
                        reply(format!("OK registered as {}\r\n", username)).await;
                    }
                    _ => {
                        reply("ERROR username taken or invalid\r\n".to_string()).await;
                    }
                }
            }
            Command::Login { username, password } => {
                if session.is_some() {
                    reply("ERROR already logged in\r\n".to_string()).await;
                    continue;
                }
                // Per-IP throttle on the TCP path too, so an attacker cannot
                // open many sockets and spray login attempts across them.
                if !state.allow_auth_attempt(peer_ip).await {
                    reply("ERROR rate limited\r\n".to_string()).await;
                    continue;
                }
                let db = state.db.clone();
                let uname = username.clone();
                let found = tokio::task::spawn_blocking(move || db.find_user(&uname)).await;
                let stored = match found {
                    Ok(Ok(user)) => user,
                    // A database failure and an unknown user are reported the
                    // same way, so this path stays free of an enumeration oracle.
                    _ => None,
                };
                // Verify even when the user does not exist: a real Argon2 check
                // against a throwaway hash keeps the timing of both outcomes
                // similar, so `LOGIN` cannot be used to probe for usernames.
                let pw = password.clone();
                let target = stored.as_ref().map(|(_, hash)| hash.clone());
                let ok = state
                    .gated_hash(move || {
                        crate::auth::verify_password_or_dummy(&pw, target.as_deref())
                    })
                    .await;
                match stored {
                    Some((user, _)) if ok => {
                        let uid = user.id;
                        let uname = user.username.clone();
                        session = Some(Session {
                            user_id: uid,
                            username: uname.clone(),
                        });
                        state
                            .register_user(&username, tx.clone(), Some(kill.clone()))
                            .await;
                        rate_key = Some(username.clone());
                        reply(format!("OK logged in as {}\r\n", username)).await;
                        // Deliver any DMs that arrived while the user was offline.
                        let dm_user = crate::db::User {
                            id: uid,
                            username: uname,
                        };
                        deliver_undelivered_dms(&state, &dm_user, &tx).await;
                    }
                    _ => {
                        reply("ERROR invalid credentials\r\n".to_string()).await;
                    }
                }
            }
            Command::Join { channel } => {
                let Some(sess) = &session else {
                    reply("ERROR not logged in\r\n".to_string()).await;
                    continue;
                };
                if !crate::protocol::valid_channel(&channel) {
                    reply("ERROR invalid channel\r\n".to_string()).await;
                    continue;
                }
                let db = state.db.clone();
                let ch = channel.clone();
                let uid = sess.user_id;
                let result = tokio::task::spawn_blocking(move || {
                    let ch = db.create_channel(&ch)?;
                    db.join_channel(ch.id, uid)?;
                    Ok::<_, rusqlite::Error>(ch)
                })
                .await;
                match result {
                    Ok(Ok(_)) => {
                        // Always re-bind the subscription. After a kick the
                        // fan-out entry is gone but this connection's `joined`
                        // list is stale, so an "already in" shortcut would
                        // prevent the user from ever receiving again.
                        state.unsubscribe(&channel, &tx).await;
                        let count = state
                            .subscribe(&channel, uid, tx.clone(), Some(kill.clone()))
                            .await;
                        if !joined.contains(&channel) {
                            joined.push(channel.clone());
                        }
                        reply(format!("OK joined {} ({} online)\r\n", channel, count)).await;
                        let line = format!("JOIN {} {}\r\n", channel, sess.username);
                        state.broadcast(&channel, line).await;
                    }
                    _ => {
                        reply("ERROR could not join\r\n".to_string()).await;
                    }
                }
            }
            Command::Part { channel } => {
                let Some(sess) = &session else {
                    reply("ERROR not logged in\r\n".to_string()).await;
                    continue;
                };
                let db = state.db.clone();
                let ch = channel.clone();
                let uid = sess.user_id;
                let result = tokio::task::spawn_blocking(move || {
                    let Some(ch) = db.find_channel(&ch)? else {
                        return Ok::<_, rusqlite::Error>(None);
                    };
                    db.part_channel(ch.id, uid)?;
                    Ok(Some(()))
                })
                .await;
                // Distinguish a DB error from a successful part, and a missing
                // channel from a real part: `result` is the join handle, and its
                // inner value is the rusqlite result. Only treat `Ok(Ok(Some(())))`
                // as success so a DB failure or a non-existent channel isn't
                // silently reported as "parted".
                match result {
                    Ok(Ok(Some(()))) => {
                        state.unsubscribe(&channel, &tx).await;
                        joined.retain(|c| c != &channel);
                        reply(format!("OK parted {}\r\n", channel)).await;
                        let line = format!("PART {} {}\r\n", channel, sess.username);
                        state.broadcast(&channel, line).await;
                    }
                    Ok(Ok(None)) => {
                        reply(format!("ERROR no such channel {}\r\n", channel)).await;
                    }
                    _ => {
                        reply("ERROR could not part\r\n".to_string()).await;
                    }
                }
            }
            Command::Privmsg { channel, message } => {
                let Some(sess) = &session else {
                    reply("ERROR not logged in\r\n".to_string()).await;
                    continue;
                };
                // A message body is echoed verbatim into `PRIVMSG <target>
                // <user> :<body>` fan-out lines, so reject control characters
                // (notably `\r`/`\n`) that would smuggle extra protocol lines
                // into every subscriber's stream. This mirrors the topic
                // validation and the WS bridge's own check.
                if !crate::protocol::valid_message(&message) {
                    reply("ERROR invalid message\r\n".to_string()).await;
                    continue;
                }
                // If the target is not a channel (doesn't start with '#'),
                // treat it as a private message to a user.
                if !channel.starts_with('#') {
                    if channel.eq_ignore_ascii_case(&sess.username) {
                        reply("ERROR cannot message yourself\r\n".to_string()).await;
                        continue;
                    }
                    let db = state.db.clone();
                    let target = channel.clone();
                    let result = tokio::task::spawn_blocking(move || db.find_user(&target)).await;
                    match result {
                        Ok(Ok(Some((user, _)))) => {
                            let line =
                                format!("PRIVMSG {} {} :{}\r\n", channel, sess.username, message);
                            // Deliver live if the recipient is connected; otherwise
                            // store it so it is delivered on their next login.
                            let delivered = state.dm(&user.username, line).await;
                            if !delivered {
                                let db = state.db.clone();
                                let from_id = sess.user_id;
                                let to_id = user.id;
                                let body = message.clone();
                                if let Err(e) = tokio::task::spawn_blocking(move || {
                                    db.store_dm(from_id, to_id, &body)
                                })
                                .await
                                {
                                    tracing::warn!("store_dm task failed: {e}");
                                }
                            }
                            reply(format!("OK sent to {}\r\n", channel)).await;
                        }
                        _ => {
                            reply(format!("ERROR no such user {}\r\n", channel)).await;
                        }
                    }
                    continue;
                }
                let db = state.db.clone();
                let ch = channel.clone();
                let uid = sess.user_id;
                let body = message.clone();
                let uname = sess.username.clone();
                // Per-channel flood control: a user can post at the per-user
                // rate across all channels, but this bounds how fast they can
                // flood a *single* channel. Without it, one user could drown a
                // channel while staying under the global per-user cap.
                let allowed = {
                    let mut limiters = state.channel_limiters.lock().await;
                    limiters
                        .entry((channel.clone(), sess.username.clone()))
                        .or_insert_with(|| RateLimiter::new(10.0, 5.0))
                        .allow()
                };
                if !allowed {
                    reply("ERROR rate limited\r\n".to_string()).await;
                    continue;
                }
                let result = tokio::task::spawn_blocking(move || {
                    let Some(ch) = db.find_channel(&ch)? else {
                        return Ok::<_, rusqlite::Error>(None);
                    };
                    // Require membership before sending.
                    let members = db.channel_members(ch.id)?;
                    if !members.iter().any(|m| m == &uname) {
                        return Ok(Some(false));
                    }
                    db.store_message(ch.id, uid, &body)?;
                    Ok(Some(true))
                })
                .await;
                match result {
                    Ok(Ok(Some(true))) => {
                        let line =
                            format!("PRIVMSG {} {} :{}\r\n", channel, sess.username, message);
                        state.broadcast(&channel, line).await;
                    }
                    Ok(Ok(Some(false))) => {
                        reply(format!("ERROR not in channel {}\r\n", channel)).await;
                    }
                    _ => {
                        reply(format!("ERROR no such channel {}\r\n", channel)).await;
                    }
                }
            }
            Command::List => {
                if session.is_none() {
                    reply("ERROR not logged in\r\n".to_string()).await;
                    continue;
                }
                let db = state.db.clone();
                let result = tokio::task::spawn_blocking(move || db.list_channels()).await;
                match result {
                    Ok(Ok(channels)) => {
                        let mut out = String::from("CHANNELS\r\n");
                        for (name, count) in channels {
                            out.push_str(&format!("  {} ({} members)\r\n", name, count));
                        }
                        reply(out).await;
                    }
                    _ => {
                        reply("ERROR listing channels\r\n".to_string()).await;
                    }
                }
            }
            Command::History {
                channel,
                limit,
                before,
            } => {
                let Some(sess) = &session else {
                    reply("ERROR not logged in\r\n".to_string()).await;
                    continue;
                };
                let db = state.db.clone();
                let ch = channel.clone();
                let uid = sess.user_id;
                let cursor = before;
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
                        cursor,
                    )?)))
                })
                .await;
                match result {
                    Ok(Ok(Some(Ok(msgs)))) => {
                        let mut out = format!("HISTORY {}\r\n", channel);
                        for m in msgs {
                            // The leading id is the cursor for paging backwards.
                            out.push_str(&format!("  {} {}: {}\r\n", m.id, m.username, m.body));
                        }
                        reply(out).await;
                    }
                    Ok(Ok(Some(Err(())))) => {
                        reply(format!("ERROR not in channel {}\r\n", channel)).await;
                    }
                    Ok(Ok(None)) => {
                        reply(format!("ERROR no such channel {}\r\n", channel)).await;
                    }
                    _ => {
                        reply("ERROR reading history\r\n".to_string()).await;
                    }
                }
            }
            Command::Quit => break,
            Command::Ping => {
                reply("PONG\r\n".to_string()).await;
            }
            Command::Names { channel } => {
                if session.is_none() {
                    reply("ERROR not logged in\r\n".to_string()).await;
                    continue;
                }
                let db = state.db.clone();
                let ch = channel.clone();
                let result = tokio::task::spawn_blocking(move || {
                    if let Some(ch) = db.find_channel(&ch)? {
                        db.channel_members(ch.id)
                    } else {
                        Ok(Vec::new())
                    }
                })
                .await;
                match result {
                    Ok(Ok(members)) => {
                        let mut out = format!("NAMES {}\r\n", channel);
                        for m in members {
                            out.push_str(&format!("  {}\r\n", m));
                        }
                        reply(out).await;
                    }
                    _ => {
                        reply("ERROR reading members\r\n".to_string()).await;
                    }
                }
            }
            Command::Topic { channel, text } => {
                let Some(sess) = &session else {
                    reply("ERROR not logged in\r\n".to_string()).await;
                    continue;
                };
                let db = state.db.clone();
                let ch = channel.clone();
                let uid = sess.user_id;
                let is_set = text.is_some();
                // A topic is echoed verbatim into `TOPIC <channel> <text>`
                // broadcast lines, so reject control characters (notably `\r`
                // and `\n`) that would smuggle extra protocol lines into the
                // fan-out. This mirrors the channel/username validation.
                if let Some(t) = &text
                    && !crate::protocol::valid_topic(t)
                {
                    reply("ERROR invalid topic\r\n".to_string()).await;
                    continue;
                }
                let result = tokio::task::spawn_blocking(move || {
                    let Some(ch) = db.find_channel(&ch)? else {
                        return Ok::<_, rusqlite::Error>(None);
                    };
                    if !db.is_member(ch.id, uid)? {
                        return Ok(Some(Err("not in channel".to_string())));
                    }
                    match &text {
                        Some(t) => {
                            if !db.is_op(ch.id, uid)? {
                                return Ok(Some(Err("not an operator".to_string())));
                            }
                            db.set_topic(ch.id, t)?;
                            Ok(Some(Ok(t.clone())))
                        }
                        None => Ok(Some(Ok(db.get_topic(ch.id)?.unwrap_or_default()))),
                    }
                })
                .await;
                match result {
                    Ok(Ok(Some(Ok(topic)))) => {
                        reply(format!("TOPIC {} {}\r\n", channel, topic)).await;
                        if is_set {
                            let line = format!("TOPIC {} {}\r\n", channel, topic);
                            state.broadcast(&channel, line).await;
                        }
                    }
                    Ok(Ok(Some(Err(msg)))) => {
                        reply(format!("ERROR {}\r\n", msg)).await;
                    }
                    Ok(Ok(None)) => {
                        reply(format!("ERROR no such channel {}\r\n", channel)).await;
                    }
                    _ => {
                        reply("ERROR topic failed\r\n".to_string()).await;
                    }
                }
            }
            Command::Help => {
                reply(
                    "Commands:\r\n\
                     REGISTER <name> <password>\r\n\
                     LOGIN <name> <password>\r\n\
                     NICK <name>\r\n\
                     JOIN <channel>\r\n\
                     PART <channel>\r\n\
                     PRIVMSG <channel|user> :<message>\r\n\
                     LIST\r\n\
                     NAMES <channel>\r\n\
                     WHO <user>\r\n\
                     TOPIC <channel> [text]\r\n\
                     HISTORY <channel> [limit]\r\n\
                     KICK <channel> <user>\r\n\
                     OP <channel> <user>\r\n\
                     DEOP <channel> <user>\r\n\
                     WS <channel>\r\n\
                     PING\r\n\
                     QUIT\r\n"
                        .to_string(),
                )
                .await;
            }
            Command::Nick { name } => {
                let Some(sess) = &mut session else {
                    reply("ERROR not logged in\r\n".to_string()).await;
                    continue;
                };
                if !crate::protocol::valid_username(&name) {
                    reply("ERROR invalid nick\r\n".to_string()).await;
                    continue;
                }
                // Persist the rename in the database first; on failure leave the
                // session unchanged.
                let db = state.db.clone();
                let uid = sess.user_id;
                let new_name = name.clone();
                let result =
                    tokio::task::spawn_blocking(move || db.rename_user(uid, &new_name)).await;
                match result {
                    Ok(Ok(())) => {
                        let old = sess.username.clone();
                        sess.username = name.clone();
                        rate_key = Some(name.clone());
                        reply(format!("OK nick changed to {}\r\n", name)).await;
                        // Re-key every live connection of this user (IRC + WS
                        // bridge) to the new name, so the other connections keep
                        // receiving DMs and the `KICKED` control line. The
                        // per-user rate limiter is moved too, so the renamed user
                        // does not get a fresh bucket on their other connections.
                        state.rename_user_connections(&old, &name).await;
                        // Tell everyone in the user's channels about the rename.
                        let db = state.db.clone();
                        let uid = sess.user_id;
                        let old_name = old.clone();
                        let new_name = name.clone();
                        let channels =
                            tokio::task::spawn_blocking(move || db.user_channels(uid)).await;
                        if let Ok(Ok(channels)) = channels {
                            for ch in channels {
                                let line = format!("NICK {} {}\r\n", old_name, new_name);
                                state.broadcast(&ch, line).await;
                            }
                        }
                    }
                    _ => {
                        reply("ERROR nick taken\r\n".to_string()).await;
                    }
                }
            }
            Command::Who { username } | Command::Whois { username } => {
                if session.is_none() {
                    reply("ERROR not logged in\r\n".to_string()).await;
                    continue;
                }
                let db = state.db.clone();
                let target = username.clone();
                let result = tokio::task::spawn_blocking(move || {
                    let Some((user, _)) = db.find_user(&target)? else {
                        return Ok::<_, rusqlite::Error>(None);
                    };
                    let channels = db.user_channels(user.id)?;
                    Ok(Some(channels))
                })
                .await;
                match result {
                    Ok(Ok(Some(channels))) => {
                        let mut out = format!("WHO {}\r\n", username);
                        if channels.is_empty() {
                            out.push_str("  (no channels)\r\n");
                        } else {
                            for c in channels {
                                out.push_str(&format!("  {}\r\n", c));
                            }
                        }
                        reply(out).await;
                    }
                    Ok(Ok(None)) => {
                        reply(format!("ERROR no such user {}\r\n", username)).await;
                    }
                    _ => {
                        reply("ERROR who failed\r\n".to_string()).await;
                    }
                }
            }
            Command::Kick {
                ref channel,
                ref user,
            }
            | Command::Op {
                ref channel,
                ref user,
            }
            | Command::Deop {
                ref channel,
                ref user,
            } => {
                let Some(sess) = &session else {
                    reply("ERROR not logged in\r\n".to_string()).await;
                    continue;
                };
                let db = state.db.clone();
                let ch = channel.clone();
                let target = user.clone();
                let actor_id = sess.user_id;
                // Resolve the moderation action once so we can move it into the
                // blocking closures without fighting the borrow on `cmd`.
                let action = match cmd {
                    Command::Kick { .. } => ModAction::Kick,
                    Command::Op { .. } => ModAction::Op,
                    Command::Deop { .. } => ModAction::Deop,
                    _ => unreachable!(),
                };
                let result = tokio::task::spawn_blocking(move || {
                    let Some(ch) = db.find_channel(&ch)? else {
                        return Ok::<_, rusqlite::Error>(None);
                    };
                    let Some((target_user, _)) = db.find_user(&target)? else {
                        return Ok(None);
                    };
                    if !db.is_op(ch.id, actor_id)? {
                        return Ok(Some(Err("not an operator".to_string())));
                    }
                    if !db.is_member(ch.id, target_user.id)? {
                        return Ok(Some(Err("not a member".to_string())));
                    }
                    Ok(Some(Ok((ch.id, target_user.id, target_user.username))))
                })
                .await;
                match result {
                    Ok(Ok(Some(Ok((ch_id, target_id, target_name))))) => {
                        let db = state.db.clone();
                        let r = tokio::task::spawn_blocking(move || match action {
                            ModAction::Kick => db.part_channel(ch_id, target_id),
                            ModAction::Op => db.set_op(ch_id, target_id, true),
                            ModAction::Deop => db.set_op(ch_id, target_id, false),
                        })
                        .await;
                        // `r` is the join handle; only `Ok(Ok(()))` means the
                        // DB write actually succeeded. A swallowed inner error
                        // would otherwise report success while the kick/op never
                        // persisted.
                        match r {
                            Ok(Ok(())) => {
                                // Use the canonical stored username in broadcasts so
                                // a case-variant target (e.g. `KICK #c Alice` for
                                // stored `alice`) is announced consistently.
                                match action {
                                    ModAction::Kick => {
                                        reply(format!("OK kicked {}\r\n", target_name)).await;
                                        let line = format!("KICK {} {}\r\n", channel, target_name);
                                        state.broadcast(channel, line).await;
                                        // Deliver the control line to the kicked
                                        // user's own connections only. It must not
                                        // go over `broadcast`: every remaining
                                        // member's read loop would treat it as its
                                        // own kick and stop acting as a member for
                                        // this channel. Direct delivery is also
                                        // what lets the target's WS bridge close
                                        // itself proactively.
                                        let kicked_line = format!("KICKED {}\r\n", channel);
                                        state.dm(&target_name, kicked_line).await;
                                        // The kick runs on another connection and
                                        // cannot edit the target's local `joined`
                                        // list, so this is what actually stops
                                        // further PRIVMSG delivery.
                                        state.unsubscribe_user(channel, target_id).await;
                                    }
                                    ModAction::Op => {
                                        reply(format!("OK op granted to {}\r\n", target_name))
                                            .await;
                                        let line = format!("OP {} {}\r\n", channel, target_name);
                                        state.broadcast(channel, line).await;
                                    }
                                    ModAction::Deop => {
                                        reply(format!("OK deop {}\r\n", target_name)).await;
                                        let line = format!("DEOP {} {}\r\n", channel, target_name);
                                        state.broadcast(channel, line).await;
                                    }
                                }
                            }
                            _ => {
                                reply("ERROR moderation failed\r\n".to_string()).await;
                            }
                        }
                    }
                    Ok(Ok(Some(Err(msg)))) => {
                        reply(format!("ERROR {}\r\n", msg)).await;
                    }
                    Ok(Ok(None)) => {
                        reply("ERROR no such channel or user\r\n".to_string()).await;
                    }
                    _ => {
                        reply("ERROR moderation failed\r\n".to_string()).await;
                    }
                }
            }
            Command::Ws { channel } => {
                let Some(sess) = &session else {
                    reply("ERROR not logged in\r\n".to_string()).await;
                    continue;
                };
                // Validate the channel exists before advertising the bridge URL.
                let db = state.db.clone();
                let ch = channel.clone();
                let uid = sess.user_id;
                let result = tokio::task::spawn_blocking(move || {
                    let Some(ch) = db.find_channel(&ch)? else {
                        return Ok::<_, rusqlite::Error>(None);
                    };
                    if !db.is_member(ch.id, uid)? {
                        return Ok(Some(false));
                    }
                    Ok(Some(true))
                })
                .await;
                match result {
                    Ok(Ok(Some(true))) => {
                        let base = state
                            .ws_base_url
                            .get()
                            .cloned()
                            .unwrap_or_else(|| "ws://localhost".to_string());
                        reply(format!(
                            "WS {} {}/ws/{}?user={}\r\n",
                            channel,
                            base,
                            crate::ws::url_encode(&channel),
                            crate::ws::url_encode(&sess.username)
                        ))
                        .await;
                    }
                    Ok(Ok(Some(false))) => {
                        reply(format!("ERROR not in channel {}\r\n", channel)).await;
                    }
                    _ => {
                        reply(format!("ERROR no such channel {}\r\n", channel)).await;
                    }
                }
            }
            Command::Unknown(_) => {
                reply("ERROR unknown command\r\n".to_string()).await;
            }
        }
    }

    // Clean up subscriptions on disconnect. Membership is session-scoped: a
    // user who disconnects is removed from the channel in the DB too, so they
    // don't linger in NAMES/LIST counts or keep receiving HISTORY as a member
    // after reconnecting without re-joining. This mirrors the `PART` command,
    // which also calls `part_channel`.
    for ch in &joined {
        let Some(sess) = &session else {
            state.unsubscribe(ch, &tx).await;
            continue;
        };
        let line = format!("QUIT {} {}\r\n", ch, sess.username);
        state.broadcast(ch, line).await;
        // Drop this connection's subscription before deciding whether to remove
        // DB membership, so the check below cannot see ourselves.
        state.unsubscribe(ch, &tx).await;
        let uid = sess.user_id;
        // Another live connection (e.g. the WS bridge) may still be in this
        // channel. Parting the channel here would remove the membership row its
        // send path re-checks, so it could no longer post.
        if state.still_subscribed(ch, uid, &tx).await {
            continue;
        }
        let db = state.db.clone();
        let ch_name = ch.clone();
        let ch_name_for_task = ch_name.clone();
        if let Err(e) = tokio::task::spawn_blocking(move || {
            if let Some(ch) = db.find_channel(&ch_name_for_task)? {
                db.part_channel(ch.id, uid)?;
            }
            Ok::<_, rusqlite::Error>(())
        })
        .await
        {
            tracing::warn!("part_channel task failed for {ch_name}: {e}");
        }
    }

    // Unregister the user's DM sender so they stop receiving private messages.
    if let Some(sess) = &session {
        state.unregister_user(&sess.username, &tx).await;
        // Drop the per-user rate limiter so the map does not leak an entry for
        // every username that ever connected. Recreated lazily on next message.
        state.remove_rate_limiter(&sess.username).await;
        // Drop the per-channel limiters too, so the map does not leak an entry
        // for every (channel, user) pair that ever posted.
        state.remove_channel_limiters(&sess.username).await;
    }

    // Drop the sender so the writer task finishes.
    drop(tx);
    let _ = writer_task.await;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rate_limiter_allows_burst_then_throttles() {
        let mut limiter = RateLimiter::new(3.0, 1.0);
        assert!(limiter.allow());
        assert!(limiter.allow());
        assert!(limiter.allow());
        // Burst exhausted; no time has passed, so the next line is rejected.
        assert!(!limiter.allow());
    }

    #[test]
    fn rate_limiter_refills_after_sleep() {
        let mut limiter = RateLimiter::new(1.0, 10.0);
        assert!(limiter.allow());
        assert!(!limiter.allow());
        std::thread::sleep(std::time::Duration::from_millis(200));
        // ~2 tokens refilled, so the next line is allowed.
        assert!(limiter.allow());
    }

    #[tokio::test]
    async fn read_line_capped_accepts_lines_at_or_below_the_cap() {
        let mut reader = BufReader::new(std::io::Cursor::new(b"hello\n".to_vec()));
        let mut buf = Vec::new();
        assert!(matches!(
            read_line_capped(&mut reader, &mut buf, 512).await.unwrap(),
            ReadOutcome::Line
        ));
        assert_eq!(String::from_utf8_lossy(&buf), "hello\n");
    }

    #[tokio::test]
    async fn read_line_capped_rejects_oversize_line_and_resyncs() {
        // One absurdly long line followed by a normal one: the long line is
        // rejected, and the next read must start at the *next* line rather than
        // in the middle of the discarded one.
        let mut input = "A".repeat(100_000);
        input.push('\n');
        input.push_str("PRIVMSG #general :still here\n");
        let mut reader = BufReader::new(std::io::Cursor::new(input.into_bytes()));
        let mut buf = Vec::new();

        assert!(matches!(
            read_line_capped(&mut reader, &mut buf, 512).await.unwrap(),
            ReadOutcome::TooLong
        ));
        assert!(
            buf.len() <= 512,
            "oversize line must not be buffered, got {} bytes",
            buf.len()
        );

        assert!(matches!(
            read_line_capped(&mut reader, &mut buf, 512).await.unwrap(),
            ReadOutcome::Line
        ));
        assert_eq!(
            String::from_utf8_lossy(&buf),
            "PRIVMSG #general :still here\n"
        );
    }

    #[tokio::test]
    async fn read_line_capped_reassembles_across_chunks() {
        // A line split across several reads (duplex hands out small chunks) must
        // still come back whole.
        let (mut client, server) = tokio::io::duplex(4);
        let writer = tokio::spawn(async move {
            client.write_all(b"hello ").await.unwrap();
            client.write_all(b"world\n").await.unwrap();
        });
        let mut reader = BufReader::new(server);
        let mut buf = Vec::new();

        assert!(matches!(
            read_line_capped(&mut reader, &mut buf, 512).await.unwrap(),
            ReadOutcome::Line
        ));
        assert_eq!(String::from_utf8_lossy(&buf), "hello world\n");
        writer.await.unwrap();
    }

    #[tokio::test]
    async fn read_line_capped_handles_final_line_without_newline() {
        let mut reader = BufReader::new(std::io::Cursor::new(b"tail".to_vec()));
        let mut buf = Vec::new();
        assert!(matches!(
            read_line_capped(&mut reader, &mut buf, 512).await.unwrap(),
            ReadOutcome::Line
        ));
        assert_eq!(String::from_utf8_lossy(&buf), "tail");

        // Nothing left: the stream reports EOF.
        assert!(matches!(
            read_line_capped(&mut reader, &mut buf, 512).await.unwrap(),
            ReadOutcome::Eof
        ));
    }
}
