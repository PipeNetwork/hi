use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};
use tokio::sync::Mutex;

use crate::db::Db;

/// A connected, authenticated client session.
#[derive(Debug, Clone)]
pub struct Session {
    pub user_id: i64,
    pub username: String,
}

/// Outbound queue capacity per connection. Bounded so a slow or disconnected
/// client cannot grow memory without bound.
pub const OUTBOUND_CAPACITY: usize = 256;

/// Handle used to tear down a connection from the fan-out path when that
/// connection cannot keep up. Dropping the subscriber's `Sender` clone is not
/// enough: the connection handler still holds its own clone, so the writer
/// stays alive and the client is only silently unsubscribed. `trip` marks the
/// handler dead, wakes it, and aborts the writer task so the socket closes.
#[derive(Clone)]
pub struct ConnKill {
    alive: Arc<AtomicBool>,
    abort: tokio::task::AbortHandle,
    shutdown: Arc<tokio::sync::Notify>,
}

impl ConnKill {
    pub fn new(
        alive: Arc<AtomicBool>,
        abort: tokio::task::AbortHandle,
        shutdown: Arc<tokio::sync::Notify>,
    ) -> Self {
        Self {
            alive,
            abort,
            shutdown,
        }
    }

    pub fn trip(&self) {
        self.alive.store(false, Ordering::SeqCst);
        self.shutdown.notify_waiters();
        self.abort.abort();
    }

    pub fn shutdown(&self) -> &tokio::sync::Notify {
        &self.shutdown
    }
}

const SLOW_CONSUMER: &str = "ERROR slow consumer\r\n";

/// How long a minted bridge ticket stays valid. Deliberately short: the client
/// redeems it on the websocket it opens immediately after logging in.
pub const TICKET_TTL: Duration = Duration::from_secs(60);

/// A single-use credential for opening the websocket bridge.
///
/// The bridge used to authenticate only from the query string, which put the
/// password in a URL — and therefore in access logs, `Referer` headers and
/// shell history. A client instead trades its password for one of these over
/// `POST /auth`, and the URL then carries only an opaque, expiring token.
struct Ticket {
    username: String,
    expires_at: Instant,
}

/// One connection subscribed to a channel. `user_id` is stored so a kick can
/// drop every connection belonging to that user without walking sockets.
struct Subscriber {
    user_id: i64,
    tx: tokio::sync::mpsc::Sender<String>,
    kill: Option<ConnKill>,
}

/// One live connection registered for DMs / `KICKED`. Same kill handle as the
/// matching channel subscriber so a slow DM consumer is closed, not just
/// dropped from the map.
struct UserConn {
    tx: tokio::sync::mpsc::Sender<String>,
    kill: Option<ConnKill>,
}

/// How many password hashes may run at once, absent an override.
///
/// Argon2id is deliberately expensive (19 MiB and two passes per operation), so
/// an unauthenticated flood of `LOGIN`/`REGISTER` lines would otherwise pin one
/// blocking-pool thread and ~19 MiB each. Sizing the gate to the machine keeps
/// that cost proportional to the host rather than to the number of clients.
fn default_hash_slots() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .clamp(1, 8)
}

/// Shared application state, cloned into every connection task.
pub struct AppState {
    pub db: Db,
    /// The actual bound websocket bridge address, used by the `WS` command to
    /// advertise a working URL. This is the *bound* address (so an ephemeral
    /// `:0` port resolves to the real one), not the raw `CHAT_WS_ADDR` env var.
    /// Set once at startup; read on every `WS` command.
    pub ws_base_url: std::sync::OnceLock<String>,
    /// channel name -> subscribers (fan-out broadcast).
    channels: Mutex<HashMap<String, Vec<Subscriber>>>,
    /// username -> senders for direct messages to a connected user. A user may
    /// be logged in from multiple connections (IRC + WS bridge, or several
    /// sockets), so each username maps to a list of connections and a DM fans
    /// out to all of them. Registering a second connection must not overwrite
    /// the first, or that connection would silently stop receiving DMs.
    users: Mutex<HashMap<String, Vec<UserConn>>>,
    /// username -> per-user flood-control limiter. Keyed by identity (not
    /// connection) so a user cannot bypass throttling by opening many sockets.
    pub rate_limiters: Mutex<HashMap<String, crate::server::RateLimiter>>,
    /// `(channel, username)` -> per-channel flood-control limiter. A user can
    /// post at the per-user rate across all channels, but this bounds how fast
    /// they can flood a *single* channel, so one user cannot drown a channel
    /// while staying under the global per-user cap. Swept on use so the map
    /// stays bounded by active (channel, user) pairs.
    pub channel_limiters: Mutex<HashMap<(String, String), crate::server::RateLimiter>>,
    /// Bounds how many Argon2 operations are in flight at once. Callers acquire
    /// a permit before hashing or verifying a password; see `default_hash_slots`.
    pub hash_slots: tokio::sync::Semaphore,
    /// Minted bridge tickets, keyed by token. Expiry is enforced on redemption;
    /// see `mint_ticket` for how the map stays bounded.
    tickets: Mutex<HashMap<String, Ticket>>,
    /// peer IP -> limiter for `POST /auth` and `POST /register`. Keyed by IP
    /// (not username) so an attacker spraying many usernames from one address
    /// is throttled before any Argon2 work is spent.
    pub auth_limiters: Mutex<HashMap<std::net::IpAddr, crate::server::RateLimiter>>,
    /// peer IP -> limiter for *connection* attempts (TCP + WS). A single host
    /// behind a NAT or a botnet node cannot open an unbounded number of sockets
    /// even though the global `MAX_CONNECTIONS` cap is shared. Swept on use so
    /// the map stays bounded by the number of distinct IPs seen recently.
    pub conn_limiters: Mutex<HashMap<std::net::IpAddr, crate::server::RateLimiter>>,
}

impl AppState {
    pub fn new(db: Db) -> Self {
        AppState {
            db,
            ws_base_url: std::sync::OnceLock::new(),
            channels: Mutex::new(HashMap::new()),
            users: Mutex::new(HashMap::new()),
            rate_limiters: Mutex::new(HashMap::new()),
            channel_limiters: Mutex::new(HashMap::new()),
            hash_slots: tokio::sync::Semaphore::new(default_hash_slots()),
            tickets: Mutex::new(HashMap::new()),
            auth_limiters: Mutex::new(HashMap::new()),
            conn_limiters: Mutex::new(HashMap::new()),
        }
    }

    /// Set the bridge base URL advertised by the `WS` command. Called once at
    /// startup with the *bound* address (so an ephemeral `:0` port resolves to
    /// the real one) and the scheme matching the TLS mode.
    pub fn set_ws_base_url(&self, url: String) {
        let _ = self.ws_base_url.set(url);
    }

    /// Whether one `POST /auth` or `POST /register` from `ip` is allowed.
    ///
    /// A small burst (5) then a slow refill (1 per 2 s): a human logging in
    /// never notices, but a credential-stuffing loop is throttled before it
    /// spends Argon2 time. The map is swept on use so it stays bounded by the
    /// number of distinct IPs seen recently.
    pub async fn allow_auth_attempt(&self, ip: std::net::IpAddr) -> bool {
        let mut limiters = self.auth_limiters.lock().await;
        // Bound the map: drop entries that have fully refilled (idle peers).
        limiters.retain(|_, l| !l.is_full());
        let allowed = limiters
            .entry(ip)
            .or_insert_with(|| crate::server::RateLimiter::new(5.0, 0.5))
            .allow();
        if !allowed {
            crate::metrics::METRICS
                .auth_rate_limited
                .fetch_add(1, Ordering::Relaxed);
        }
        allowed
    }

    /// Whether one new connection (TCP or WS) from `ip` is allowed.
    ///
    /// A burst of 8 then a slow refill (1 per 2 s): a human opening a few tabs
    /// never notices, but a single host cannot open an unbounded number of
    /// sockets even though the global `MAX_CONNECTIONS` cap is shared. The map
    /// is swept on use so it stays bounded by the number of distinct IPs seen
    /// recently.
    pub async fn allow_conn_attempt(&self, ip: std::net::IpAddr) -> bool {
        let mut limiters = self.conn_limiters.lock().await;
        // Bound the map: drop entries that have fully refilled (idle peers).
        limiters.retain(|_, l| !l.is_full());
        limiters
            .entry(ip)
            .or_insert_with(|| crate::server::RateLimiter::new(8.0, 0.5))
            .allow()
    }

    /// Mint a single-use ticket for `username`, valid for [`TICKET_TTL`].
    ///
    /// Expired tickets are swept here rather than by a timer, so the map is
    /// bounded by the number of logins in the last TTL window.
    pub async fn mint_ticket(&self, username: &str) -> String {
        let token = crate::auth::random_token();
        let now = Instant::now();
        let mut tickets = self.tickets.lock().await;
        tickets.retain(|_, t| t.expires_at > now);
        tickets.insert(
            token.clone(),
            Ticket {
                username: username.to_string(),
                expires_at: now + TICKET_TTL,
            },
        );
        crate::metrics::METRICS
            .tickets_minted
            .fetch_add(1, Ordering::Relaxed);
        token
    }

    /// Redeem a ticket, returning the username it was minted for. Single use:
    /// the ticket is consumed whether or not it had already expired.
    pub async fn consume_ticket(&self, token: &str) -> Option<String> {
        let ticket = self.tickets.lock().await.remove(token)?;
        crate::metrics::METRICS
            .tickets_redeemed
            .fetch_add(1, Ordering::Relaxed);
        (ticket.expires_at > Instant::now()).then_some(ticket.username)
    }

    /// Check a username and password, returning the user on success.
    ///
    /// Spends the same time on an unknown username as on a known one, so it
    /// cannot be used to enumerate accounts.
    pub async fn verify_credentials(
        &self,
        username: &str,
        password: &str,
    ) -> Option<crate::db::User> {
        // The lookup touches SQLite, which blocks, so it runs on the blocking
        // pool rather than stalling the async runtime.
        let db = self.db.clone();
        let uname = username.to_string();
        let found = tokio::task::spawn_blocking(move || db.find_user(&uname))
            .await
            .ok()
            .and_then(|r| r.ok())
            .flatten();
        let stored = found.as_ref().map(|(_, hash)| hash.clone());
        let pw = password.to_string();
        let ok = self
            .gated_hash(move || crate::auth::verify_password_or_dummy(&pw, stored.as_deref()))
            .await;
        match (ok, found) {
            (true, Some((user, _))) => Some(user),
            _ => None,
        }
    }

    /// Create a new account and return it, or `None` if the username is taken.
    ///
    /// Hashing runs through the same bounded pool as login, so registration
    /// cannot be used to bypass the argon2 concurrency cap.
    pub async fn register_account(
        &self,
        username: &str,
        password: &str,
    ) -> Option<crate::db::User> {
        let pw = password.to_string();
        let hash = self
            .gated_hash(move || crate::auth::hash_password(&pw))
            .await
            .ok()?;
        // The insert touches SQLite, which blocks, so it runs on the blocking
        // pool rather than stalling the async runtime.
        let db = self.db.clone();
        let uname = username.to_string();
        tokio::task::spawn_blocking(move || db.register_user(&uname, &hash))
            .await
            .ok()
            .and_then(|r| r.ok())
    }

    /// Run a blocking password operation without letting unauthenticated clients
    /// start more of them at once than the machine can reasonably absorb.
    ///
    /// The permit is held for the duration of `work`, which is expected to call
    /// into `crate::auth` — the only blocking work of this shape in the server.
    /// The closure is dispatched to the blocking pool rather than run inline, so
    /// this is safe on a current-thread runtime and never stalls a worker.
    pub async fn gated_hash<T, F>(&self, work: F) -> T
    where
        F: FnOnce() -> T + Send + 'static,
        T: Send + 'static,
    {
        let _permit = self
            .hash_slots
            .acquire()
            .await
            .expect("hash slot semaphore is never closed");
        tokio::task::spawn_blocking(work)
            .await
            .expect("password hashing task panicked")
    }

    /// Subscribe a connection's sender to a channel. Returns the number of
    /// subscribers now in that channel (including this one).
    ///
    /// `kill` is the connection's teardown handle. Slow-consumer drops call
    /// it so the socket actually closes instead of going silent while the
    /// client still believes it is joined.
    pub async fn subscribe(
        &self,
        channel: &str,
        user_id: i64,
        tx: tokio::sync::mpsc::Sender<String>,
        kill: Option<ConnKill>,
    ) -> usize {
        let mut map = self.channels.lock().await;
        let subs = map.entry(channel.to_string()).or_default();
        subs.push(Subscriber { user_id, tx, kill });
        subs.len()
    }

    /// Unsubscribe a connection's sender from a channel.
    pub async fn unsubscribe(&self, channel: &str, tx: &tokio::sync::mpsc::Sender<String>) {
        let mut map = self.channels.lock().await;
        if let Some(subs) = map.get_mut(channel) {
            subs.retain(|s| !s.tx.same_channel(tx));
            if subs.is_empty() {
                map.remove(channel);
            }
        }
    }

    /// Drop every connection for `user_id` from a channel. Used by KICK so the
    /// target stops receiving broadcasts even though the kick runs on another
    /// connection and cannot touch that connection's local `joined` list.
    pub async fn unsubscribe_user(&self, channel: &str, user_id: i64) {
        let mut map = self.channels.lock().await;
        if let Some(subs) = map.get_mut(channel) {
            subs.retain(|s| s.user_id != user_id);
            if subs.is_empty() {
                map.remove(channel);
            }
        }
    }

    /// True if `user_id` still has a live subscription to `channel` through a
    /// connection other than `tx`. A user can be connected over more than one
    /// transport (the IRC line protocol and the WS bridge), so tearing one
    /// connection down must not remove DB membership that another live
    /// connection still relies on for sending.
    pub async fn still_subscribed(
        &self,
        channel: &str,
        user_id: i64,
        tx: &tokio::sync::mpsc::Sender<String>,
    ) -> bool {
        let map = self.channels.lock().await;
        map.get(channel).is_some_and(|subs| {
            subs.iter()
                .any(|s| s.user_id == user_id && !s.tx.same_channel(tx))
        })
    }

    /// Broadcast a line to every subscriber of a channel. Dead senders (whose
    /// receiver has been dropped) are removed so the map does not leak entries
    /// for disconnected clients. A subscriber whose outbound queue is full (a
    /// slow consumer) is closed: keeping it would silently lose this message
    /// for that client, and merely dropping the map entry would leave the
    /// socket open while the client still believed it was joined.
    pub async fn broadcast(&self, channel: &str, line: String) {
        let mut map = self.channels.lock().await;
        if let Some(subs) = map.get_mut(channel) {
            subs.retain(|s| try_deliver(&s.tx, s.kill.as_ref(), line.clone()));
            if subs.is_empty() {
                map.remove(channel);
            }
            crate::metrics::METRICS
                .messages_broadcast
                .fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Register a connected user's sender for direct messages. Appends to the
    /// user's sender list so a second connection does not overwrite the first.
    pub async fn register_user(
        &self,
        username: &str,
        tx: tokio::sync::mpsc::Sender<String>,
        kill: Option<ConnKill>,
    ) {
        self.users
            .lock()
            .await
            .entry(username.to_string())
            .or_default()
            .push(UserConn { tx, kill });
    }

    /// Unregister a connected user's sender. Removes only this sender; other
    /// connections for the same user are left intact. The entry is dropped when
    /// the last sender is removed.
    pub async fn unregister_user(&self, username: &str, tx: &tokio::sync::mpsc::Sender<String>) {
        let mut users = self.users.lock().await;
        if let Some(senders) = users.get_mut(username) {
            senders.retain(|s| !s.tx.same_channel(tx));
            if senders.is_empty() {
                users.remove(username);
            }
        }
    }

    /// Drop a user's per-user rate limiter. Called on disconnect so the
    /// `rate_limiters` map does not grow without bound for users who log in and
    /// out repeatedly. The limiter is recreated lazily on the next message.
    pub async fn remove_rate_limiter(&self, username: &str) {
        self.rate_limiters.lock().await.remove(username);
    }

    /// Drop every per-channel limiter for a user. Called on disconnect so the
    /// `channel_limiters` map does not grow without bound for users who join and
    /// leave many channels. Recreated lazily on the next message.
    pub async fn remove_channel_limiters(&self, username: &str) {
        self.channel_limiters
            .lock()
            .await
            .retain(|(_, user), _| user != username);
    }

    /// Move a user's per-channel limiters to a new key. Called on `NICK` so the
    /// old username does not leak entries in the map.
    pub async fn rename_channel_limiters(&self, old: &str, new: &str) {
        let mut limiters = self.channel_limiters.lock().await;
        let moved: Vec<_> = limiters
            .iter()
            .filter(|((_, user), _)| user == old)
            .map(|((channel, _), limiter)| (channel.clone(), limiter.clone()))
            .collect();
        limiters.retain(|(_, user), _| user != old);
        for (channel, limiter) in moved {
            limiters.insert((channel, new.to_string()), limiter);
        }
    }

    /// Move a user's per-user rate limiter to a new key. Called on `NICK` so the
    /// old username does not leak an entry in the map. If the old key has no
    /// limiter yet (user never sent a message), this is a no-op.
    pub async fn rename_rate_limiter(&self, old: &str, new: &str) {
        let mut limiters = self.rate_limiters.lock().await;
        if let Some(limiter) = limiters.remove(old) {
            limiters.insert(new.to_string(), limiter);
        }
    }

    /// Re-key every live connection of a user from `old` to `new` after a `NICK`.
    ///
    /// A user can be connected over more than one transport (IRC + WS bridge, or
    /// several sockets), and each connection registers its own DM sender under
    /// the username. Renaming must move *all* of them, not just the connection
    /// that issued the `NICK` — otherwise the other connections keep listening
    /// under the old name and silently stop receiving DMs and the `KICKED`
    /// control line. The per-user rate limiter is moved too, so the renamed user
    /// does not get a fresh bucket (and a throttle bypass) on their other
    /// connections.
    pub async fn rename_user_connections(&self, old: &str, new: &str) {
        let mut users = self.users.lock().await;
        if let Some(senders) = users.remove(old) {
            users.entry(new.to_string()).or_default().extend(senders);
        }
        drop(users);
        self.rename_rate_limiter(old, new).await;
        self.rename_channel_limiters(old, new).await;
    }

    /// Send a line directly to a connected user, fanning out to every one of
    /// their connections. A dead sender (receiver dropped) is removed. A
    /// merely-full queue (slow consumer) is closed, consistent with `broadcast`.
    ///
    /// Returns `true` if the user was connected and the line was queued to at
    /// least one live sender, `false` if the user is offline (so the caller can
    /// store the message for later delivery).
    ///
    /// The `users` lock is taken only to snapshot the live senders, then dropped
    /// before any delivery work. `try_deliver` is strictly non-blocking
    /// (`try_send`, never `send`/`await`), so holding the lock across it would
    /// be safe, but dropping it first keeps the critical section as short as
    /// possible so a user with many connections does not stall every other
    /// user's DM path while its senders are walked.
    pub async fn dm(&self, username: &str, line: String) -> bool {
        let senders = {
            let mut users = self.users.lock().await;
            match users.get_mut(username) {
                Some(senders) => {
                    // Take the vec out so we can iterate it without holding a
                    // borrow on the map entry, then put the survivors back.
                    let taken = std::mem::take(senders);
                    let mut live = Vec::with_capacity(taken.len());
                    for conn in taken {
                        if try_deliver(&conn.tx, conn.kill.as_ref(), line.clone()) {
                            live.push(conn);
                        }
                    }
                    let any_live = !live.is_empty();
                    if any_live {
                        *senders = live;
                    } else {
                        users.remove(username);
                    }
                    any_live
                }
                None => false,
            }
        };
        if senders {
            crate::metrics::METRICS
                .dms_delivered
                .fetch_add(1, Ordering::Relaxed);
        } else {
            crate::metrics::METRICS
                .dms_stored
                .fetch_add(1, Ordering::Relaxed);
        }
        senders
    }

    /// Prune old channel and direct messages, keeping the newest `keep` per
    /// channel / pair. Returns the total number of rows deleted.
    pub async fn prune_old_messages(&self, keep: i64) -> rusqlite::Result<usize> {
        let db = self.db.clone();
        let deleted = tokio::task::spawn_blocking(move || -> rusqlite::Result<usize> {
            let deleted = db.prune_messages(keep)? + db.prune_dms(keep)?;
            Ok(deleted)
        })
        // A join error means the blocking task panicked or was cancelled, not
        // that a value failed to convert to SQL. There is no dedicated
        // "background task failed" variant, so surface it as a conversion
        // failure carrying a descriptive message rather than silently dropping
        // the error.
        .await
        .map_err(|e| {
            rusqlite::Error::ToSqlConversionFailure(Box::new(std::io::Error::other(format!(
                "prune task failed: {e}"
            ))))
        })??;
        crate::metrics::METRICS
            .messages_pruned
            .fetch_add(deleted as u64, Ordering::Relaxed);
        Ok(deleted)
    }
}

/// Convenience alias for the shared state.
pub type SharedState = Arc<AppState>;

/// Queue `line` on `tx`. A closed receiver is a dead connection. A full
/// queue is a slow consumer: try to enqueue a control line, then trip `kill`
/// so the handler and writer actually tear the socket down. Returning `false`
/// drops this entry from the fan-out map.
fn try_deliver(
    tx: &tokio::sync::mpsc::Sender<String>,
    kill: Option<&ConnKill>,
    line: String,
) -> bool {
    match tx.try_send(line) {
        Ok(()) => true,
        Err(tokio::sync::mpsc::error::TrySendError::Closed(_)) => false,
        Err(tokio::sync::mpsc::error::TrySendError::Full(_)) => {
            crate::metrics::METRICS
                .slow_consumers
                .fetch_add(1, Ordering::Relaxed);
            if let Some(kill) = kill {
                let _ = tx.try_send(SLOW_CONSUMER.to_string());
                kill.trip();
            }
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn unsubscribe_user_drops_only_that_user() {
        let db = crate::db::Db::open(":memory:").unwrap();
        let state = AppState::new(db);
        let (tx_a, mut rx_a) = tokio::sync::mpsc::channel(8);
        let (tx_b, mut rx_b) = tokio::sync::mpsc::channel(8);
        state.subscribe("#c", 1, tx_a, None).await;
        state.subscribe("#c", 2, tx_b, None).await;
        state.unsubscribe_user("#c", 1).await;
        state.broadcast("#c", "hi".into()).await;
        assert!(
            rx_a.try_recv().is_err(),
            "kicked user still received a broadcast"
        );
        assert_eq!(rx_b.try_recv().unwrap(), "hi");
    }

    #[tokio::test]
    async fn tickets_are_single_use_and_resolve_the_user() {
        let db = crate::db::Db::open(":memory:").unwrap();
        let state = AppState::new(db);
        let token = state.mint_ticket("alice").await;
        assert_eq!(state.consume_ticket(&token).await.as_deref(), Some("alice"));
        // Second redemption must fail: the ticket is spent.
        assert!(state.consume_ticket(&token).await.is_none());
        assert!(state.consume_ticket("nonsense").await.is_none());
    }

    #[tokio::test]
    async fn verify_credentials_accepts_only_the_right_password() {
        let db = crate::db::Db::open(":memory:").unwrap();
        let hash = crate::auth::hash_password("hunter2").unwrap();
        db.register_user("alice", &hash).unwrap();
        let state = AppState::new(db);
        assert!(state.verify_credentials("alice", "hunter2").await.is_some());
        assert!(state.verify_credentials("alice", "wrong").await.is_none());
        // Unknown user: no user, and no crash from the dummy-hash path.
        assert!(
            state
                .verify_credentials("nobody", "hunter2")
                .await
                .is_none()
        );
    }

    #[tokio::test]
    async fn slow_consumer_is_closed_not_silently_unsubscribed() {
        let db = crate::db::Db::open(":memory:").unwrap();
        let state = AppState::new(db);
        let (tx, _rx) = tokio::sync::mpsc::channel(1);
        tx.try_send("fill".into()).unwrap();

        let alive = Arc::new(AtomicBool::new(true));
        let shutdown = Arc::new(tokio::sync::Notify::new());
        let writer = tokio::spawn(std::future::pending::<()>());
        let kill = ConnKill::new(alive.clone(), writer.abort_handle(), shutdown);
        state.subscribe("#c", 1, tx, Some(kill)).await;
        state.broadcast("#c", "overflow".into()).await;

        assert!(
            !alive.load(Ordering::SeqCst),
            "slow consumer must trip ConnKill so the socket closes"
        );
        // Yield so the aborted writer task can observe the abort.
        tokio::task::yield_now().await;
        assert!(writer.is_finished(), "writer task must be aborted");
    }

    #[tokio::test]
    async fn channel_limiters_are_per_user_and_cleaned_up() {
        let db = crate::db::Db::open(":memory:").unwrap();
        let state = AppState::new(db);

        // A fresh (channel, user) pair gets a full bucket.
        let allowed = {
            let mut limiters = state.channel_limiters.lock().await;
            limiters
                .entry(("#general".into(), "alice".into()))
                .or_insert_with(|| crate::server::RateLimiter::new(2.0, 1.0))
                .allow()
        };
        assert!(allowed);

        // Removing a user's limiters drops every channel they posted to, but
        // leaves another user's entry for the same channel intact.
        {
            let mut limiters = state.channel_limiters.lock().await;
            limiters.insert(
                ("#general".into(), "bob".into()),
                crate::server::RateLimiter::new(2.0, 1.0),
            );
        }
        state.remove_channel_limiters("alice").await;
        let keys: Vec<_> = state
            .channel_limiters
            .lock()
            .await
            .keys()
            .cloned()
            .collect();
        assert_eq!(keys, vec![("#general".into(), "bob".into())]);
    }

    #[tokio::test]
    async fn rename_channel_limiters_moves_all_of_a_users_entries() {
        let db = crate::db::Db::open(":memory:").unwrap();
        let state = AppState::new(db);
        {
            let mut limiters = state.channel_limiters.lock().await;
            limiters.insert(
                ("#a".into(), "alice".into()),
                crate::server::RateLimiter::new(2.0, 1.0),
            );
            limiters.insert(
                ("#b".into(), "alice".into()),
                crate::server::RateLimiter::new(2.0, 1.0),
            );
            limiters.insert(
                ("#a".into(), "bob".into()),
                crate::server::RateLimiter::new(2.0, 1.0),
            );
        }
        state.rename_channel_limiters("alice", "ali").await;
        let mut keys: Vec<_> = state
            .channel_limiters
            .lock()
            .await
            .keys()
            .cloned()
            .collect();
        keys.sort();
        assert_eq!(
            keys,
            vec![
                ("#a".into(), "ali".into()),
                ("#a".into(), "bob".into()),
                ("#b".into(), "ali".into()),
            ]
        );
    }
}
