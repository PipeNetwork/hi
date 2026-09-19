# chat

A minimal IRC-style chat server written in Rust. Users register, create/join
channels, and exchange messages. Data is persisted in SQLite.

## Stack

- **tokio** — async TCP server, one task per connection
- **rusqlite** (bundled SQLite) — storage, accessed via `spawn_blocking`
- **argon2** — password hashing

No `sqlx`; the DB is plain `rusqlite` on a blocking thread pool.

## Build & run

```sh
cargo build --release
cargo run --release            # listens on 127.0.0.1:8080, DB at chat.db
```

Environment: `CHAT_ADDR` (default `127.0.0.1:8080`), `CHAT_DB` (default
`chat.db`), `CHAT_WS_ADDR` (default `127.0.0.1:8080`), `CHAT_HISTORY_KEEP`
(default `1000`, per-channel / per-DM message retention),
`CHAT_PRUNE_INTERVAL_SECS` (default `60`). If the requested port is already in
use (Docker often holds 8080), the server binds `:0` on the same host and
prints the actual port on the `chat server listening on` /
`websocket bridge listening on` lines. Binding `:0` yourself does the same.

The line protocol and the websocket bridge are two separate listeners. The
line protocol (IRC-style commands) listens on `CHAT_ADDR`; the websocket
bridge *and* the embedded web UI share `CHAT_WS_ADDR`. Both default to
`127.0.0.1:8080`, so by default the same port serves the web page, the bridge,
and the line protocol. Set `CHAT_WS_ADDR` to a different port to split them.

Connect with any TCP client:

```sh
nc 127.0.0.1 8080
```

## Wire protocol

Newline-framed text. The server replies with `OK ...`, `ERROR ...`, or
multi-line responses. Lines longer than 512 bytes are rejected.

| Command | Description |
| --- | --- |
| `REGISTER <name> <password>` | Create a user and log in |
| `LOGIN <name> <password>` | Log in as an existing user |
| `NICK <name>` | Rename the current user |
| `JOIN <channel>` | Create-or-join a channel |
| `PART <channel>` | Leave a channel |
| `PRIVMSG <channel\|user> :<message>` | Send a channel or direct message |
| `LIST` | List channels and member counts |
| `NAMES <channel>` | List members of a channel |
| `WHO <user>` | List channels a user has joined |
| `TOPIC <channel>` | Show a channel's topic |
| `TOPIC <channel> <text>` | Set a channel's topic (ops only) |
| `HISTORY <channel> [limit] [BEFORE <id>]` | Replay recent messages (members only) |
| `KICK` / `OP` / `DEOP` | Moderate a channel (ops only) |
| `WS <channel>` | Advertise a websocket bridge URL |
| `HELP` | Show the command list |
| `PING` | Server replies `PONG` |
| `QUIT` | Disconnect |

## Websocket bridge and web UI

A single-page web UI is embedded in the binary and served on the same port as the
bridge: start the server and open `http://127.0.0.1:8080/` in a browser. There is
nothing to build and no assets on disk — `src/web/index.html` is compiled in with
`include_str!`, so the client and the server can never drift apart. Three paths
are served: `GET /` (the page), `GET /index.html` (the same page), and
`POST /auth` (ticket minting); everything else gets a `404`.

The page asks for server, channel, nick and password (server defaults to
`location.host`, so it points at whatever you loaded it from). It does **not**
put the password in the bridge URL: it `POST`s to `/auth` first and connects with
the short-lived one-time ticket that comes back. It renders
`PRIVMSG`/`JOIN`/`QUIT` lines, shows `KICKED` when the server evicts you,
replays recent history on connect, reconnects with exponential backoff if the
link drops, and remembers every field except the password in `localStorage`.

Bridges listen on `CHAT_WS_ADDR` (default `127.0.0.1:8080`). Connect to
`/ws/<channel>?ticket=<ticket>` — the channel name is percent-encoded, since
`#` would otherwise start a URL fragment (`#general` → `%23general`). The
ticket is minted by `POST /auth` (or `POST /register`); a raw `password=` in
the URL is deliberately not accepted, because a password in a query string ends
up in URLs, proxy logs and `Referer` headers. Knowing a username is not enough.

Connecting joins the channel (creating it if needed), so no prior `JOIN` over
TCP is required.

Add `&history=<n>` (max 200) to have the server replay the last `n` messages as
frames right after connecting, so a reloaded page is not blank. Replayed frames
are formatted `user: body` and arrive before any live traffic.

`POST /auth` mints a single-use ticket valid for 60 seconds, which the bridge
redeems instead of re-verifying Argon2 on the socket:

```sh
curl -s -X POST http://127.0.0.1:8080/auth -d 'user=alice&password=hunter2'
# {"ticket":"8f2c…"}
```

Prefer a ticket over a password: a password in the query string ends up in URLs,
proxy logs and `Referer` headers, a ticket does not. The bridge accepts *only*
tickets, so a password can never ride in a bridge URL.

The two directions are asymmetric:

- **Inbound** text frames are taken as the *raw body* of a message and posted
  to the channel. They are not parsed as protocol commands, so sending
  `PRIVMSG #general :hi` posts that whole string as the message text — send
  just `hi`.
- **Outbound** frames are the relayed protocol lines for the channel
  (`PRIVMSG #general alice :hi`, `JOIN ...`, `QUIT ...`). A `KICKED <channel>`
  line closes the bridge.

Channel membership is session-scoped, but is kept while *any* of a user's
connections is still subscribed. A user connected over both TCP and the bridge
therefore keeps posting rights on the bridge after their TCP session quits.

From a browser console (no dependencies):

```js
// Mint a ticket first, then connect with it — never put the password in the URL.
const res = await fetch("http://127.0.0.1:8080/auth", {
  method: "POST",
  headers: { "content-type": "application/json" },
  body: JSON.stringify({ user: "alice", password: "hunter2" }),
});
const { ticket } = await res.json();
const ws = new WebSocket("ws://127.0.0.1:8080/ws/%23general?ticket=" + ticket);
ws.onmessage = (e) => console.log(e.data);
ws.onopen = () => ws.send("hello from the browser");
```

The bridge accepts only tickets, so a script must do one HTTP POST to `/auth`
first (as the browser console example above does).

## Example session

```
REGISTER alice hunter2
OK registered as alice
JOIN #general
OK joined #general (1 online)
PRIVMSG #general :hello world
```

Other members of `#general` receive:

```
PRIVMSG #general alice :hello world
```

## Storage

SQLite schema (created on first run):

- `users(id, username UNIQUE, password_hash, created_at)`
- `channels(id, name UNIQUE, topic, created_at)`
- `channel_members(channel_id, user_id, joined_at)`
- `messages(id, channel_id, user_id, body, sent_at)`, indexed on
  `(channel_id, id)` and pruned so each channel keeps the newest
  `CHAT_HISTORY_KEEP` messages. The `id` doubles as the `HISTORY … BEFORE <id>`
  cursor.

The server also runs a retention pass at startup and every
`CHAT_PRUNE_INTERVAL_SECS` seconds (default 60).

## Tests

```sh
cargo test --quiet
```

`cargo fmt --check` and `cargo clippy --all-targets -- -D warnings` are expected
to be clean; CI (`.github/workflows/ci.yml`) enforces formatting, lints and the
test suite on every push and pull request.