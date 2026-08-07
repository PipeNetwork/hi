//! Local discovery and control plane for one shared, session-owning runtime.
//!
//! The leader deliberately exposes only registration, status, and attachment.
//! It does not own an agent or provider: the process that already owns the
//! session/sync pipeline runs this server and remains the sole inference owner.

use anyhow::{Context, Result, anyhow, bail};
use hi_protocol::{ClientIdentity, ClientType, FramedUnix};
use hi_tui::event::UiEvent;
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, VecDeque},
    fs::{self, File, OpenOptions},
    io::Write,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
    time::{Duration, SystemTime, UNIX_EPOCH},
};

const DEADLINE: Duration = Duration::from_secs(5);
const JOURNAL_CAPACITY: usize = 512;
/// Registrations are one-shot connections with no disconnect signal, so client
/// records are cumulative; bound them so a long-lived leader cannot grow
/// without limit under repeated register/attach probes.
const CLIENT_RECORD_CAPACITY: usize = 1024;
const PUBLISH_QUEUE_CAPACITY: usize = 1024;
#[cfg(unix)]
const SUBSCRIBER_IDLE_PROBE: Duration = Duration::from_secs(30);

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct SessionEvent {
    pub session_id: String,
    pub sequence: u64,
    pub event: UiEvent,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(tag = "op", rename_all = "snake_case", deny_unknown_fields)]
enum Request {
    Register {
        identity: ClientIdentity,
    },
    Status {
        identity: ClientIdentity,
    },
    Attach {
        identity: ClientIdentity,
    },
    Publish {
        session_id: String,
        event: UiEvent,
    },
    Subscribe {
        session_id: String,
        after_sequence: u64,
    },
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct RuntimeStatus {
    session_id: String,
    pid: u32,
    started_unix: u64,
    clients: usize,
    attached_clients: usize,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(tag = "result", rename_all = "snake_case", deny_unknown_fields)]
enum Response {
    Registered {
        client_id: u64,
        status: RuntimeStatus,
    },
    Status {
        status: RuntimeStatus,
    },
    Attached {
        client_id: u64,
        status: RuntimeStatus,
    },
    Published {
        sequence: u64,
    },
    Event {
        // Boxed: SessionEvent dwarfs the other wire variants; these frames flow
        // on every session event, so keep the enum small.
        envelope: Box<SessionEvent>,
    },
    /// Liveness probe sent on idle subscriptions; carries no data and should
    /// be ignored by subscribers.
    Ping,
    ReplayGap {
        session_id: String,
        requested_after: u64,
        oldest_sequence: u64,
    },
    Error {
        message: String,
    },
}

#[cfg(unix)]
mod unix {
    use super::*;
    use tokio::net::{UnixListener, UnixStream};
    use tokio::sync::broadcast;

    struct ClientRecord {
        _identity: ClientIdentity,
        attached: bool,
    }

    struct State {
        session_id: String,
        started_unix: u64,
        next_client_id: u64,
        clients: BTreeMap<u64, ClientRecord>,
        next_sequence: u64,
        journal: VecDeque<SessionEvent>,
        events: broadcast::Sender<SessionEvent>,
    }

    /// (gap oldest-sequence, journal snapshot, live receiver, resume cursor)
    type Subscription = (
        Option<u64>,
        Vec<SessionEvent>,
        broadcast::Receiver<SessionEvent>,
        u64,
    );

    impl State {
        fn status(&self) -> RuntimeStatus {
            RuntimeStatus {
                session_id: self.session_id.clone(),
                pid: std::process::id(),
                started_unix: self.started_unix,
                clients: self.clients.len(),
                attached_clients: self
                    .clients
                    .values()
                    .filter(|client| client.attached)
                    .count(),
            }
        }

        fn publish(&mut self, session_id: String, event: UiEvent) -> Result<SessionEvent> {
            if session_id != self.session_id {
                bail!("runtime owns session {}", self.session_id);
            }
            let envelope = SessionEvent {
                session_id,
                sequence: self.next_sequence,
                event,
            };
            self.next_sequence += 1;
            if self.journal.len() == JOURNAL_CAPACITY {
                self.journal.pop_front();
            }
            self.journal.push_back(envelope.clone());
            let _ = self.events.send(envelope.clone());
            Ok(envelope)
        }

        fn subscribe_snapshot(
            &self,
            session_id: &str,
            after_sequence: u64,
        ) -> Result<Subscription> {
            if session_id != self.session_id {
                bail!("runtime owns session {}", self.session_id);
            }
            let receiver = self.events.subscribe();
            let oldest = self.journal.front().map(|event| event.sequence);
            // A cursor past everything ever assigned can only come from a
            // previous incarnation of this runtime. Report a gap and restart
            // the cursor at zero — replaying the whole journal — instead of
            // silently starving the subscriber until sequences catch up.
            let latest = self.next_sequence.saturating_sub(1);
            let stale_cursor = after_sequence > latest;
            let gap = if stale_cursor {
                Some(oldest.unwrap_or(self.next_sequence))
            } else {
                oldest.filter(|oldest| after_sequence.saturating_add(1) < *oldest)
            };
            let resume_cursor = if stale_cursor { 0 } else { after_sequence };
            let snapshot: Vec<_> = self
                .journal
                .iter()
                .filter(|event| event.sequence > resume_cursor)
                .cloned()
                .collect();
            if gap.is_some() {
                hi_observability::record(hi_observability::ReliabilityEvent::ReplayGap);
            }
            hi_observability::record(hi_observability::ReliabilityEvent::Replay {
                count: snapshot.len() as u64,
            });
            Ok((gap, snapshot, receiver, resume_cursor))
        }

        fn register(&mut self, identity: ClientIdentity, attached: bool) -> (u64, RuntimeStatus) {
            hi_observability::record(hi_observability::ReliabilityEvent::RuntimeRegistration);
            let id = self.next_client_id;
            self.next_client_id += 1;
            while self.clients.len() >= CLIENT_RECORD_CAPACITY {
                self.clients.pop_first();
            }
            self.clients.insert(
                id,
                ClientRecord {
                    _identity: identity,
                    attached,
                },
            );
            (id, self.status())
        }
    }

    pub(super) struct Leader {
        listener: UnixListener,
        lock: File,
        lock_path: PathBuf,
        socket_path: PathBuf,
        owner_nonce: String,
        state: Arc<Mutex<State>>,
    }

    impl Leader {
        pub(super) fn bind(root: &Path, session_id: &str) -> Result<Self> {
            validate_session_id(session_id)?;
            {
                use std::os::unix::fs::DirBuilderExt;
                // 0700 on creation: the directory holds session sockets, and the
                // tmp fallback lives in a world-writable parent.
                fs::DirBuilder::new()
                    .recursive(true)
                    .mode(0o700)
                    .create(root)
                    .with_context(|| format!("create {}", root.display()))?;
            }
            let socket_path = root.join(format!("{session_id}.sock"));
            let lock_path = root.join(format!("{session_id}.lock"));
            let owner_nonce = format!("{}-{}", std::process::id(), now_nanos());
            let lock = acquire_lock(&lock_path, &socket_path, session_id, &owner_nonce)?;
            if socket_path.exists() {
                if socket_live(&socket_path) {
                    drop(lock);
                    remove_if_owned(&lock_path, &owner_nonce);
                    bail!("runtime {session_id} is already running");
                }
                fs::remove_file(&socket_path).context("remove stale runtime socket")?;
            }
            let listener = UnixListener::bind(&socket_path).context("bind runtime socket")?;
            let (events, _) = broadcast::channel(JOURNAL_CAPACITY);
            Ok(Self {
                listener,
                lock,
                lock_path,
                socket_path,
                owner_nonce,
                state: Arc::new(Mutex::new(State {
                    session_id: session_id.into(),
                    started_unix: now(),
                    next_client_id: 1,
                    clients: BTreeMap::new(),
                    next_sequence: 1,
                    journal: VecDeque::with_capacity(JOURNAL_CAPACITY),
                    events,
                })),
            })
        }

        pub(super) async fn serve(self) -> Result<()> {
            loop {
                match self.listener.accept().await {
                    Ok((stream, _)) => {
                        let state = Arc::clone(&self.state);
                        tokio::spawn(async move {
                            let _ = handle(stream, state).await;
                        });
                    }
                    // Transient accept failures (ECONNABORTED, fd exhaustion)
                    // must not tear down the runtime and unlink its discovery
                    // files; back off briefly and keep accepting.
                    Err(_) => tokio::time::sleep(Duration::from_millis(100)).await,
                }
            }
        }
    }

    impl Drop for Leader {
        fn drop(&mut self) {
            let _ = self.lock.sync_all();
            if lock_owned_by(&self.lock_path, &self.owner_nonce) {
                let _ = fs::remove_file(&self.socket_path);
                remove_if_owned(&self.lock_path, &self.owner_nonce);
            }
        }
    }

    /// Take ownership via an OS advisory lock, which the kernel releases when
    /// the owner dies. This is atomic: there is no window where a contender can
    /// observe a half-written lock file or unlink a live owner's lock, unlike a
    /// create-then-write/read-then-remove protocol.
    fn acquire_lock(
        path: &Path,
        socket_path: &Path,
        session_id: &str,
        nonce: &str,
    ) -> Result<File> {
        loop {
            // `truncate(false)`: wiping the file on open would destroy a live
            // owner's lock contents before we know whether we hold the lock.
            let mut file = OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .truncate(false)
                .open(path)
                .context("open runtime lock")?;
            match file.try_lock() {
                Ok(()) => {
                    // An exiting owner may have unlinked this inode between our
                    // open and lock; only a lock on the file the path still
                    // names counts.
                    if !file_is_at_path(&file, path) {
                        continue;
                    }
                    file.set_len(0).context("truncate runtime lock")?;
                    writeln!(
                        file,
                        "session={session_id}\npid={}\nnonce={nonce}",
                        std::process::id()
                    )
                    .context("write runtime lock")?;
                    file.sync_all().context("sync runtime lock")?;
                    return Ok(file);
                }
                Err(std::fs::TryLockError::WouldBlock) => {
                    if socket_live(socket_path) {
                        bail!("runtime {session_id} is already running");
                    }
                    bail!("runtime {session_id} owner is still alive");
                }
                Err(std::fs::TryLockError::Error(error)) => {
                    return Err(error).context("lock runtime lock");
                }
            }
        }
    }

    fn file_is_at_path(file: &File, path: &Path) -> bool {
        use std::os::unix::fs::MetadataExt;
        match (file.metadata(), fs::metadata(path)) {
            (Ok(held), Ok(named)) => held.ino() == named.ino() && held.dev() == named.dev(),
            _ => false,
        }
    }

    fn lock_owned_by(path: &Path, nonce: &str) -> bool {
        fs::read_to_string(path)
            .map(|contents| {
                contents
                    .lines()
                    .any(|line| line == format!("nonce={nonce}"))
            })
            .unwrap_or(false)
    }

    fn remove_if_owned(path: &Path, nonce: &str) {
        if lock_owned_by(path, nonce) {
            let _ = fs::remove_file(path);
        }
    }

    fn socket_live(path: &Path) -> bool {
        (0..3).any(|attempt| {
            let live = std::os::unix::net::UnixStream::connect(path).is_ok();
            if !live && attempt < 2 {
                std::thread::sleep(Duration::from_millis(20));
            }
            live
        })
    }

    fn validate(identity: &ClientIdentity) -> Result<()> {
        identity.ensure_compatible().map_err(anyhow::Error::from)?;
        if identity.version.is_empty() {
            bail!("client version cannot be empty");
        }
        Ok(())
    }

    async fn handle(stream: UnixStream, state: Arc<Mutex<State>>) -> Result<()> {
        let mut framed = FramedUnix::new(stream);
        let request: Request = framed.receive(DEADLINE).await?;
        let response = match request {
            Request::Register { identity } => match validate(&identity) {
                Ok(()) => {
                    let mut state = state.lock().unwrap();
                    let (client_id, status) = state.register(identity, false);
                    Response::Registered { client_id, status }
                }
                Err(error) => Response::Error {
                    message: error.to_string(),
                },
            },
            Request::Attach { identity } => match validate(&identity) {
                Ok(()) => {
                    let mut state = state.lock().unwrap();
                    let (client_id, status) = state.register(identity, true);
                    Response::Attached { client_id, status }
                }
                Err(error) => Response::Error {
                    message: error.to_string(),
                },
            },
            Request::Status { identity } => match validate(&identity) {
                Ok(()) => Response::Status {
                    status: state.lock().unwrap().status(),
                },
                Err(error) => Response::Error {
                    message: error.to_string(),
                },
            },
            Request::Publish { session_id, event } => {
                match state.lock().unwrap().publish(session_id, event) {
                    Ok(envelope) => Response::Published {
                        sequence: envelope.sequence,
                    },
                    Err(error) => Response::Error {
                        message: error.to_string(),
                    },
                }
            }
            Request::Subscribe {
                session_id,
                after_sequence,
            } => {
                return subscribe(framed, state, session_id, after_sequence).await;
            }
        };
        framed.send(&response, DEADLINE).await?;
        Ok(())
    }

    async fn subscribe(
        mut framed: FramedUnix,
        state: Arc<Mutex<State>>,
        session_id: String,
        after_sequence: u64,
    ) -> Result<()> {
        let subscription = {
            state
                .lock()
                .unwrap()
                .subscribe_snapshot(&session_id, after_sequence)
        };
        let (gap, snapshot, mut receiver, resume_cursor) = match subscription {
            Ok(subscription) => subscription,
            Err(error) => {
                framed
                    .send(
                        &Response::Error {
                            message: error.to_string(),
                        },
                        DEADLINE,
                    )
                    .await?;
                return Ok(());
            }
        };
        let mut cursor = resume_cursor;
        if let Some(oldest_sequence) = gap {
            framed
                .send(
                    &Response::ReplayGap {
                        session_id: session_id.clone(),
                        requested_after: after_sequence,
                        oldest_sequence,
                    },
                    DEADLINE,
                )
                .await?;
        }
        for envelope in snapshot {
            cursor = cursor.max(envelope.sequence);
            framed
                .send(
                    &Response::Event {
                        envelope: Box::new(envelope),
                    },
                    DEADLINE,
                )
                .await?;
        }
        loop {
            match tokio::time::timeout(SUBSCRIBER_IDLE_PROBE, receiver.recv()).await {
                // Idle: probe the connection. Every subscriber task holds the
                // State Arc, so the broadcast sender can never close and an
                // abandoned connection would otherwise park this task (and its
                // fd) until the next publish; the probe's send error reaps it.
                Err(_) => framed.send(&Response::Ping, DEADLINE).await?,
                Ok(Ok(envelope))
                    if envelope.session_id == session_id && envelope.sequence > cursor =>
                {
                    cursor = envelope.sequence;
                    framed
                        .send(
                            &Response::Event {
                                envelope: Box::new(envelope),
                            },
                            DEADLINE,
                        )
                        .await?;
                }
                Ok(Ok(_)) => {}
                Ok(Err(broadcast::error::RecvError::Lagged(_))) => {
                    let (gap, snapshot, new_receiver, resumed) = state
                        .lock()
                        .unwrap()
                        .subscribe_snapshot(&session_id, cursor)?;
                    receiver = new_receiver;
                    cursor = resumed;
                    if let Some(oldest_sequence) = gap {
                        framed
                            .send(
                                &Response::ReplayGap {
                                    session_id: session_id.clone(),
                                    requested_after: cursor,
                                    oldest_sequence,
                                },
                                DEADLINE,
                            )
                            .await?;
                    }
                    for envelope in snapshot {
                        if envelope.sequence > cursor {
                            cursor = envelope.sequence;
                            framed
                                .send(
                                    &Response::Event {
                                        envelope: Box::new(envelope),
                                    },
                                    DEADLINE,
                                )
                                .await?;
                        }
                    }
                }
                Ok(Err(broadcast::error::RecvError::Closed)) => return Ok(()),
            }
        }
    }

    pub(super) async fn request(path: &Path, request: Request) -> Result<Response> {
        // Deadline covers connect too: a wedged (SIGSTOPped) leader accepts
        // the connection at the kernel level and would otherwise stall callers
        // like the doctor sweep indefinitely.
        let stream = tokio::time::timeout(DEADLINE, UnixStream::connect(path))
            .await
            .map_err(|_| anyhow!("timed out connecting to runtime at {}", path.display()))?
            .with_context(|| format!("connect to runtime at {}", path.display()))?;
        let mut framed = FramedUnix::new(stream);
        framed.send(&request, DEADLINE).await?;
        Ok(framed.receive(DEADLINE).await?)
    }
}

#[derive(Clone)]
pub(crate) struct Publisher {
    #[cfg(unix)]
    queue: tokio::sync::mpsc::Sender<UiEvent>,
}

impl Publisher {
    pub(crate) fn for_session(session_id: impl Into<String>) -> Result<Self> {
        let session_id = session_id.into();
        let socket_path = socket_path(&runtime_dir(), &session_id)?;
        #[cfg(unix)]
        {
            // No leader socket → no publisher. This keeps the per-event tap
            // work (clone + queue + a failing connect per event) out of every
            // ordinary TUI session; a leader started mid-session is picked up
            // on the next session switch or restart.
            if !socket_path.exists() {
                bail!("no local runtime for session {session_id}");
            }
            // A single ordered worker: a fire-and-forget task per event would
            // race its siblings, so the leader would assign sequence numbers in
            // arrival order (scrambling streamed text for subscribers) while
            // each in-flight event held its own socket fd.
            let (queue, mut events) = tokio::sync::mpsc::channel::<UiEvent>(PUBLISH_QUEUE_CAPACITY);
            tokio::spawn(async move {
                while let Some(event) = events.recv().await {
                    let _ = unix::request(
                        &socket_path,
                        Request::Publish {
                            session_id: session_id.clone(),
                            event,
                        },
                    )
                    .await;
                }
            });
            Ok(Self { queue })
        }
        #[cfg(not(unix))]
        {
            let _ = socket_path;
            Ok(Self {})
        }
    }

    pub(crate) fn publish_best_effort(&self, event: UiEvent) {
        #[cfg(unix)]
        // Dropping when the queue is full keeps this best-effort under
        // backpressure instead of buffering without bound.
        let _ = self.queue.try_send(event);
        #[cfg(not(unix))]
        let _ = event;
    }
}

fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn now_nanos() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos()
}

pub(crate) fn valid_session_id(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 128
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || b"-_.".contains(&byte))
}

fn validate_session_id(value: &str) -> Result<()> {
    if !valid_session_id(value) {
        bail!("session id must be 1-128 ASCII letters, digits, '-', '_', or '.'");
    }
    Ok(())
}

pub(crate) fn runtime_dir() -> PathBuf {
    if let Some(path) = std::env::var_os("XDG_RUNTIME_DIR") {
        PathBuf::from(path).join("hi")
    } else if let Some(path) = std::env::var_os("HOME") {
        PathBuf::from(path).join(".hi/run")
    } else {
        // Keyed by uid, never pid: the leader and its clients run in different
        // processes and must derive the same discovery directory.
        #[cfg(unix)]
        return std::env::temp_dir().join(format!("hi-{}", unsafe { libc::geteuid() }));
        #[cfg(not(unix))]
        std::env::temp_dir().join(format!("hi-{}", std::process::id()))
    }
}

fn socket_path(root: &Path, session_id: &str) -> Result<PathBuf> {
    validate_session_id(session_id)?;
    Ok(root.join(format!("{session_id}.sock")))
}

fn identity(client_type: ClientType) -> ClientIdentity {
    ClientIdentity::current(client_type, env!("CARGO_PKG_VERSION"))
}

fn print_response(response: Response) -> Result<()> {
    match response {
        Response::Error { message } => bail!("runtime rejected request: {message}"),
        response => println!("{}", serde_json::to_string_pretty(&response)?),
    }
    Ok(())
}

pub(crate) async fn status_check(session_id: &str) -> Result<String> {
    #[cfg(unix)]
    {
        let path = socket_path(&runtime_dir(), session_id)?;
        match unix::request(
            &path,
            Request::Status {
                identity: identity(ClientType::Headless),
            },
        )
        .await?
        {
            Response::Status { status } => Ok(format!(
                "session={} pid={} clients={} attached={}",
                status.session_id, status.pid, status.clients, status.attached_clients
            )),
            Response::Error { message } => bail!("runtime rejected status: {message}"),
            _ => bail!("runtime returned an unexpected status response"),
        }
    }
    #[cfg(not(unix))]
    bail!("local runtime status is unsupported on this platform")
}

pub async fn run_cli(args: &[String]) -> Result<()> {
    let command = args.first().map(String::as_str).unwrap_or("status");
    let session_id = args.get(1).map(String::as_str).unwrap_or("default");
    let root = runtime_dir();
    #[cfg(unix)]
    match command {
        "leader" => {
            let leader = unix::Leader::bind(&root, session_id)?;
            eprintln!(
                "runtime {session_id} listening at {}",
                socket_path(&root, session_id)?.display()
            );
            // Ctrl-C must drop the leader (unlinking socket + lock) rather
            // than kill the process mid-serve: a leaked socket makes every
            // later `hi doctor` fail on "stale or unreachable runtime socket"
            // until a new leader heals it.
            tokio::select! {
                result = leader.serve() => result,
                _ = tokio::signal::ctrl_c() => Ok(()),
            }
        }
        "status" => print_response(
            unix::request(
                &socket_path(&root, session_id)?,
                Request::Status {
                    identity: identity(ClientType::Headless),
                },
            )
            .await?,
        ),
        "attach" => print_response(
            unix::request(
                &socket_path(&root, session_id)?,
                Request::Attach {
                    identity: identity(ClientType::Attach),
                },
            )
            .await?,
        ),
        "register" => print_response(
            unix::request(
                &socket_path(&root, session_id)?,
                Request::Register {
                    identity: identity(ClientType::Headless),
                },
            )
            .await?,
        ),
        _ => Err(anyhow!(
            "usage: hi runtime <leader|status|attach|register> [session-id]"
        )),
    }
    #[cfg(not(unix))]
    {
        let _ = (command, session_id, root);
        Err(anyhow!(
            "local shared runtimes require Unix sockets and are unsupported on this platform"
        ))
    }
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use hi_protocol::PROTOCOL_MAJOR;
    use tempfile::TempDir;

    #[tokio::test]
    async fn registers_attaches_and_reports_status() {
        let dir = TempDir::new().unwrap();
        let leader = unix::Leader::bind(dir.path(), "session-a").unwrap();
        let socket = socket_path(dir.path(), "session-a").unwrap();
        let task = tokio::spawn(leader.serve());
        let registered = unix::request(
            &socket,
            Request::Register {
                identity: identity(ClientType::Tui),
            },
        )
        .await
        .unwrap();
        assert!(matches!(
            registered,
            Response::Registered { client_id: 1, .. }
        ));
        let attached = unix::request(
            &socket,
            Request::Attach {
                identity: identity(ClientType::Attach),
            },
        )
        .await
        .unwrap();
        assert!(matches!(attached, Response::Attached { client_id: 2, .. }));
        let status = unix::request(
            &socket,
            Request::Status {
                identity: identity(ClientType::Headless),
            },
        )
        .await
        .unwrap();
        assert!(matches!(
            status,
            Response::Status {
                status: RuntimeStatus {
                    clients: 2,
                    attached_clients: 1,
                    ..
                }
            }
        ));
        task.abort();
    }

    async fn open_subscription(path: &Path, session_id: &str, after_sequence: u64) -> FramedUnix {
        let stream = tokio::net::UnixStream::connect(path).await.unwrap();
        let mut framed = FramedUnix::new(stream);
        framed
            .send(
                &Request::Subscribe {
                    session_id: session_id.to_string(),
                    after_sequence,
                },
                DEADLINE,
            )
            .await
            .unwrap();
        framed
    }

    fn text(value: &str) -> UiEvent {
        UiEvent::Text {
            text: value.to_string(),
        }
    }

    async fn publish(path: &Path, session_id: &str, value: &str) -> u64 {
        match unix::request(
            path,
            Request::Publish {
                session_id: session_id.to_string(),
                event: text(value),
            },
        )
        .await
        .unwrap()
        {
            Response::Published { sequence } => sequence,
            response => panic!("unexpected response: {response:?}"),
        }
    }

    async fn receive_event(framed: &mut FramedUnix) -> SessionEvent {
        match framed.receive(DEADLINE).await.unwrap() {
            Response::Event { envelope } => *envelope,
            response => panic!("unexpected response: {response:?}"),
        }
    }

    fn mismatched_identity() -> ClientIdentity {
        let mut identity = identity(ClientType::Headless);
        identity.protocol_major = PROTOCOL_MAJOR.saturating_add(1);
        identity
    }

    #[tokio::test]
    async fn subscription_orders_snapshot_then_live_and_reconnects_from_cursor() {
        let dir = TempDir::new().unwrap();
        let leader = unix::Leader::bind(dir.path(), "s").unwrap();
        let socket = socket_path(dir.path(), "s").unwrap();
        let task = tokio::spawn(leader.serve());
        assert_eq!(publish(&socket, "s", "one").await, 1);
        assert_eq!(publish(&socket, "s", "two").await, 2);

        let mut subscriber = open_subscription(&socket, "s", 0).await;
        assert_eq!(receive_event(&mut subscriber).await.sequence, 1);
        assert_eq!(receive_event(&mut subscriber).await.sequence, 2);
        assert_eq!(publish(&socket, "s", "three").await, 3);
        assert_eq!(receive_event(&mut subscriber).await.sequence, 3);
        drop(subscriber);

        let mut resumed = open_subscription(&socket, "s", 2).await;
        assert_eq!(receive_event(&mut resumed).await.sequence, 3);
        task.abort();
    }

    #[tokio::test]
    async fn publish_during_replay_is_delivered_once_after_snapshot() {
        let dir = TempDir::new().unwrap();
        let leader = unix::Leader::bind(dir.path(), "s").unwrap();
        let socket = socket_path(dir.path(), "s").unwrap();
        let task = tokio::spawn(leader.serve());
        assert_eq!(publish(&socket, "s", "replay").await, 1);

        let mut subscriber = open_subscription(&socket, "s", 0).await;
        assert_eq!(publish(&socket, "s", "live").await, 2);
        assert_eq!(receive_event(&mut subscriber).await.sequence, 1);
        assert_eq!(receive_event(&mut subscriber).await.sequence, 2);
        assert!(
            tokio::time::timeout(
                Duration::from_millis(50),
                subscriber.receive::<Response>(DEADLINE)
            )
            .await
            .is_err()
        );
        task.abort();
    }

    #[tokio::test]
    async fn protocol_version_mismatch_is_rejected_without_registering_client() {
        let dir = TempDir::new().unwrap();
        let leader = unix::Leader::bind(dir.path(), "s").unwrap();
        let socket = socket_path(dir.path(), "s").unwrap();
        let task = tokio::spawn(leader.serve());

        assert!(matches!(
            unix::request(
                &socket,
                Request::Register {
                    identity: mismatched_identity(),
                },
            )
            .await
            .unwrap(),
            Response::Error { message } if message.contains("incompatible protocol major version")
        ));
        assert!(matches!(
            unix::request(
                &socket,
                Request::Status {
                    identity: identity(ClientType::Headless),
                },
            )
            .await
            .unwrap(),
            Response::Status {
                status: RuntimeStatus { clients: 0, .. }
            }
        ));
        task.abort();
    }

    #[tokio::test]
    async fn leader_shutdown_closes_subscriber_and_removes_discovery_files() {
        let dir = TempDir::new().unwrap();
        let leader = unix::Leader::bind(dir.path(), "s").unwrap();
        let socket = socket_path(dir.path(), "s").unwrap();
        let lock = dir.path().join("s.lock");
        let task = tokio::spawn(leader.serve());
        let mut subscriber = open_subscription(&socket, "s", 0).await;

        task.abort();
        task.await.unwrap_err();
        assert!(!socket.exists());
        assert!(!lock.exists());
        assert!(subscriber.receive::<Response>(DEADLINE).await.is_err());
    }

    #[tokio::test]
    async fn subscription_filters_sessions_reports_gap_and_disconnects_cleanly() {
        let dir = TempDir::new().unwrap();
        let leader = unix::Leader::bind(dir.path(), "s").unwrap();
        let socket = socket_path(dir.path(), "s").unwrap();
        let task = tokio::spawn(leader.serve());
        assert!(matches!(
            unix::request(
                &socket,
                Request::Publish {
                    session_id: "other".into(),
                    event: text("wrong")
                }
            )
            .await
            .unwrap(),
            Response::Error { .. }
        ));
        for i in 0..=JOURNAL_CAPACITY {
            publish(&socket, "s", &i.to_string()).await;
        }
        let mut subscriber = open_subscription(&socket, "s", 0).await;
        assert!(matches!(
            subscriber.receive::<Response>(DEADLINE).await.unwrap(),
            Response::ReplayGap {
                requested_after: 0,
                oldest_sequence: 2,
                ..
            }
        ));
        assert_eq!(receive_event(&mut subscriber).await.sequence, 2);
        drop(subscriber);
        publish(&socket, "s", "after-disconnect").await;
        assert!(!task.is_finished());
        task.abort();
    }

    #[tokio::test]
    async fn stale_cursor_from_previous_incarnation_gets_gap_and_journal_replay() {
        let dir = TempDir::new().unwrap();
        let leader = unix::Leader::bind(dir.path(), "s").unwrap();
        let socket = socket_path(dir.path(), "s").unwrap();
        let task = tokio::spawn(leader.serve());
        assert_eq!(publish(&socket, "s", "journaled").await, 1);
        // A cursor persisted against a previous leader incarnation is past
        // anything this one has assigned; the subscriber must get a gap, this
        // incarnation's journal, and then live events — not silently starve
        // until sequences catch up.
        let mut subscriber = open_subscription(&socket, "s", 57).await;
        assert!(matches!(
            subscriber.receive::<Response>(DEADLINE).await.unwrap(),
            Response::ReplayGap {
                requested_after: 57,
                oldest_sequence: 1,
                ..
            }
        ));
        assert_eq!(receive_event(&mut subscriber).await.sequence, 1);
        assert_eq!(publish(&socket, "s", "fresh").await, 2);
        assert_eq!(receive_event(&mut subscriber).await.sequence, 2);
        task.abort();
    }

    #[tokio::test]
    async fn stale_socket_and_lock_are_recovered_and_cleaned() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("s.lock"), "pid=0").unwrap();
        fs::write(dir.path().join("s.sock"), "stale").unwrap();
        let leader = unix::Leader::bind(dir.path(), "s").unwrap();
        assert!(dir.path().join("s.sock").exists());
        drop(leader);
        assert!(!dir.path().join("s.sock").exists());
        assert!(!dir.path().join("s.lock").exists());
    }

    #[tokio::test]
    async fn live_leader_excludes_second_owner() {
        let dir = TempDir::new().unwrap();
        let _leader = unix::Leader::bind(dir.path(), "s").unwrap();
        assert!(unix::Leader::bind(dir.path(), "s").is_err());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn concurrent_stale_takeover_elects_exactly_one_owner() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("s.lock"), "pid=0").unwrap();
        fs::write(dir.path().join("s.sock"), "stale").unwrap();
        let root = dir.path().to_path_buf();
        let barrier = Arc::new(std::sync::Barrier::new(8));
        let mut attempts = Vec::new();
        for _ in 0..8 {
            let root = root.clone();
            let barrier = Arc::clone(&barrier);
            attempts.push(tokio::task::spawn_blocking(move || {
                barrier.wait();
                unix::Leader::bind(&root, "s")
            }));
        }
        let mut winners = Vec::new();
        for attempt in attempts {
            if let Ok(leader) = attempt.await.unwrap() {
                winners.push(leader);
            }
        }
        assert_eq!(winners.len(), 1);
        assert!(unix::Leader::bind(dir.path(), "s").is_err());
    }

    #[tokio::test]
    async fn old_owner_drop_does_not_unlink_successor() {
        let dir = TempDir::new().unwrap();
        let old = unix::Leader::bind(dir.path(), "s").unwrap();
        let lock = dir.path().join("s.lock");
        let socket = dir.path().join("s.sock");
        fs::remove_file(&lock).unwrap();
        fs::write(&lock, "session=s\npid=2\nnonce=successor\n").unwrap();
        drop(old);
        assert!(lock.exists());
        assert!(socket.exists());
    }
}
