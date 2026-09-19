use rusqlite::{Connection, OptionalExtension, params};
use std::path::Path;
use std::sync::{Arc, Mutex};

/// A thread-safe handle to the SQLite database. SQLite connections are not
/// `Sync`, so we wrap one in a `Mutex` and hand it to `spawn_blocking` tasks.
#[derive(Clone)]
pub struct Db {
    conn: Arc<Mutex<Connection>>,
}

/// A user record as stored in the database.
#[derive(Debug, Clone)]
pub struct User {
    pub id: i64,
    pub username: String,
}

/// A channel record as stored in the database.
#[derive(Debug, Clone)]
pub struct Channel {
    pub id: i64,
    /// The channel's name (e.g. `#general`). Kept so callers that resolve a
    /// channel can render its name without a second query. Currently populated
    /// but not yet read by any caller; it is retained for the upcoming `LIST`
    /// output and to keep the struct self-describing.
    #[allow(dead_code)]
    pub name: String,
}

/// A stored chat message.
#[derive(Debug, Clone)]
pub struct Message {
    /// Row id, used as the history cursor (`HISTORY … BEFORE <id>`).
    pub id: i64,
    pub username: String,
    pub body: String,
}

impl Db {
    /// Lock the connection, recovering from a poisoned mutex.
    ///
    /// A panic while the lock was held (say, an `expect` in a handler) poisons
    /// it. Without this recovery every later request would panic too, turning a
    /// single bad request into a permanently dead server.
    fn lock(&self) -> std::sync::MutexGuard<'_, Connection> {
        self.conn
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Open (or create) the database at `path` and apply the schema.
    pub fn open(path: impl AsRef<Path>) -> rusqlite::Result<Self> {
        let conn = Connection::open(path)?;
        conn.execute_batch(
            "PRAGMA journal_mode = WAL;
             PRAGMA foreign_keys = ON;
             -- Wait for a concurrent writer instead of failing immediately with
             -- SQLITE_BUSY under load.
             PRAGMA busy_timeout = 5000;
             PRAGMA synchronous = NORMAL;

             CREATE TABLE IF NOT EXISTS users (
                 id            INTEGER PRIMARY KEY AUTOINCREMENT,
                 username      TEXT NOT NULL UNIQUE COLLATE NOCASE,
                 password_hash TEXT NOT NULL,
                 created_at    TEXT NOT NULL DEFAULT (datetime('now'))
             );

             CREATE TABLE IF NOT EXISTS channels (
                 id         INTEGER PRIMARY KEY AUTOINCREMENT,
                 name       TEXT NOT NULL UNIQUE,
                 topic      TEXT NOT NULL DEFAULT '',
                 created_at TEXT NOT NULL DEFAULT (datetime('now'))
             );

             CREATE TABLE IF NOT EXISTS channel_members (
                 channel_id INTEGER NOT NULL REFERENCES channels(id) ON DELETE CASCADE,
                 user_id    INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                 is_op      INTEGER NOT NULL DEFAULT 0,
                 joined_at  TEXT NOT NULL DEFAULT (datetime('now')),
                 PRIMARY KEY (channel_id, user_id)
             );

             CREATE TABLE IF NOT EXISTS messages (
                 id         INTEGER PRIMARY KEY AUTOINCREMENT,
                 channel_id INTEGER NOT NULL REFERENCES channels(id) ON DELETE CASCADE,
                 user_id    INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                 body       TEXT NOT NULL,
                 sent_at    TEXT NOT NULL DEFAULT (datetime('now'))
             );

             -- Direct messages, stored so a message to an offline user is not
             -- silently dropped: it is delivered on the recipient's next login.
             -- `delivered` is set once the recipient's connection has been told.
             CREATE TABLE IF NOT EXISTS direct_messages (
                 id         INTEGER PRIMARY KEY AUTOINCREMENT,
                 from_user  INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                 to_user    INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                 body       TEXT NOT NULL,
                 sent_at    TEXT NOT NULL DEFAULT (datetime('now')),
                 delivered  INTEGER NOT NULL DEFAULT 0
             );
             CREATE INDEX IF NOT EXISTS idx_dm_undelivered
                 ON direct_messages(to_user, delivered, id);

             -- Serves both HISTORY and the bridge's on-connect backfill, which
             -- both read the newest rows of a single channel.
             CREATE INDEX IF NOT EXISTS idx_messages_channel_id ON messages(channel_id, id);",
        )?;
        // Migrations for databases created before the topic / is_op columns
        // existed. Guarded so they are safe on both fresh and existing DBs.
        let has_topic: bool = conn
            .query_row(
                "SELECT COUNT(*) FROM pragma_table_info('channels') WHERE name = 'topic'",
                [],
                |r| r.get::<_, i64>(0),
            )
            .map(|n| n != 0)?;
        if !has_topic {
            conn.execute(
                "ALTER TABLE channels ADD COLUMN topic TEXT NOT NULL DEFAULT ''",
                [],
            )?;
        }
        let has_op: bool = conn
            .query_row(
                "SELECT COUNT(*) FROM pragma_table_info('channel_members') WHERE name = 'is_op'",
                [],
                |r| r.get::<_, i64>(0),
            )
            .map(|n| n != 0)?;
        if !has_op {
            conn.execute(
                "ALTER TABLE channel_members ADD COLUMN is_op INTEGER NOT NULL DEFAULT 0",
                [],
            )?;
        }
        Ok(Db {
            conn: Arc::new(Mutex::new(conn)),
        })
    }

    /// Register a new user. Returns `Err` if the username is taken.
    ///
    /// Usernames are unique case-insensitively, so `Bob` and `bob` cannot both
    /// exist — otherwise a case-insensitive `find_user` would be ambiguous.
    pub fn register_user(&self, username: &str, password_hash: &str) -> rusqlite::Result<User> {
        let conn = self.lock();
        conn.execute(
            "INSERT INTO users (username, password_hash) VALUES (?1, ?2)",
            params![username, password_hash],
        )?;
        let id = conn.last_insert_rowid();
        Ok(User {
            id,
            username: username.to_string(),
        })
    }

    /// Look up a user by username, returning the stored password hash too.
    ///
    /// The match is case-insensitive so a client can address a user by any
    /// casing (`WHO Bob` finds `bob`), but the canonical stored username is
    /// returned so broadcasts and DMs use the registered spelling.
    pub fn find_user(&self, username: &str) -> rusqlite::Result<Option<(User, String)>> {
        let conn = self.lock();
        let row = conn
            .query_row(
                "SELECT id, username, password_hash FROM users WHERE username = ?1 COLLATE NOCASE",
                params![username],
                |r| {
                    Ok((
                        User {
                            id: r.get(0)?,
                            username: r.get(1)?,
                        },
                        r.get::<_, String>(2)?,
                    ))
                },
            )
            .optional()?;
        Ok(row)
    }

    /// Rename a user. Returns `Err` if the new name is already taken.
    pub fn rename_user(&self, user_id: i64, new_name: &str) -> rusqlite::Result<()> {
        let conn = self.lock();
        // Check for a name conflict up front so we can report a clear error
        // instead of relying on the UNIQUE constraint firing. Case-insensitive,
        // matching `find_user` and the `users.username` uniqueness.
        let taken: bool = conn.query_row(
            "SELECT EXISTS(
                    SELECT 1 FROM users WHERE username = ?1 COLLATE NOCASE AND id != ?2
                 )",
            params![new_name, user_id],
            |r| r.get(0),
        )?;
        if taken {
            return Err(rusqlite::Error::SqliteFailure(
                rusqlite::ffi::Error::new(19), // SQLITE_CONSTRAINT
                Some(format!("username '{}' is already taken", new_name)),
            ));
        }
        conn.execute(
            "UPDATE users SET username = ?1 WHERE id = ?2",
            params![new_name, user_id],
        )?;
        Ok(())
    }

    /// List the names of all channels a user has joined.
    pub fn user_channels(&self, user_id: i64) -> rusqlite::Result<Vec<String>> {
        let conn = self.lock();
        let mut stmt = conn.prepare(
            "SELECT c.name
             FROM channel_members m
             JOIN channels c ON c.id = m.channel_id
             WHERE m.user_id = ?1
             ORDER BY c.name",
        )?;
        let rows = stmt
            .query_map(params![user_id], |r| r.get(0))?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(rows)
    }

    /// Create a channel if it does not exist, returning its record.
    pub fn create_channel(&self, name: &str) -> rusqlite::Result<Channel> {
        let conn = self.lock();
        conn.execute(
            "INSERT OR IGNORE INTO channels (name) VALUES (?1)",
            params![name],
        )?;
        let id = conn.query_row(
            "SELECT id FROM channels WHERE name = ?1",
            params![name],
            |r| r.get(0),
        )?;
        Ok(Channel {
            id,
            name: name.to_string(),
        })
    }

    /// Add a user to a channel (idempotent). The first member becomes an op.
    pub fn join_channel(&self, channel_id: i64, user_id: i64) -> rusqlite::Result<()> {
        let conn = self.lock();
        let count: i64 = conn.query_row(
            "SELECT COUNT(*) FROM channel_members WHERE channel_id = ?1",
            params![channel_id],
            |r| r.get(0),
        )?;
        let is_op = if count == 0 { 1 } else { 0 };
        conn.execute(
            "INSERT OR IGNORE INTO channel_members (channel_id, user_id, is_op) VALUES (?1, ?2, ?3)",
            params![channel_id, user_id, is_op],
        )?;
        Ok(())
    }

    /// Return whether a user is an operator of a channel. Missing membership
    /// is `false`, not a query error, so callers can treat "not an op" uniformly.
    pub fn is_op(&self, channel_id: i64, user_id: i64) -> rusqlite::Result<bool> {
        let conn = self.lock();
        let is_op: Option<i64> = conn
            .query_row(
                "SELECT is_op FROM channel_members WHERE channel_id = ?1 AND user_id = ?2",
                params![channel_id, user_id],
                |r| r.get(0),
            )
            .optional()?;
        Ok(is_op.unwrap_or(0) != 0)
    }

    /// Return whether a user is a member of a channel.
    pub fn is_member(&self, channel_id: i64, user_id: i64) -> rusqlite::Result<bool> {
        let conn = self.lock();
        let count: i64 = conn.query_row(
            "SELECT COUNT(*) FROM channel_members WHERE channel_id = ?1 AND user_id = ?2",
            params![channel_id, user_id],
            |r| r.get(0),
        )?;
        Ok(count != 0)
    }

    /// Set whether a user is an operator of a channel.
    pub fn set_op(&self, channel_id: i64, user_id: i64, op: bool) -> rusqlite::Result<()> {
        let conn = self.lock();
        conn.execute(
            "UPDATE channel_members SET is_op = ?1 WHERE channel_id = ?2 AND user_id = ?3",
            params![op as i64, channel_id, user_id],
        )?;
        Ok(())
    }

    /// Remove a user from a channel.
    pub fn part_channel(&self, channel_id: i64, user_id: i64) -> rusqlite::Result<()> {
        let conn = self.lock();
        conn.execute(
            "DELETE FROM channel_members WHERE channel_id = ?1 AND user_id = ?2",
            params![channel_id, user_id],
        )?;
        Ok(())
    }

    /// List all channels with their member counts.
    pub fn list_channels(&self) -> rusqlite::Result<Vec<(String, i64)>> {
        let conn = self.lock();
        let mut stmt = conn.prepare(
            "SELECT c.name, COUNT(m.user_id)
             FROM channels c
             LEFT JOIN channel_members m ON m.channel_id = c.id
             GROUP BY c.id
             ORDER BY c.name",
        )?;
        let rows = stmt
            .query_map([], |r| Ok((r.get(0)?, r.get(1)?)))?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(rows)
    }

    /// Store a message and return its record.
    pub fn store_message(
        &self,
        channel_id: i64,
        user_id: i64,
        body: &str,
    ) -> rusqlite::Result<Message> {
        let conn = self.lock();
        conn.execute(
            "INSERT INTO messages (channel_id, user_id, body) VALUES (?1, ?2, ?3)",
            params![channel_id, user_id, body],
        )?;
        let id = conn.last_insert_rowid();
        let (username,) = conn.query_row(
            "SELECT u.username
             FROM messages m
             JOIN users u ON u.id = m.user_id
             WHERE m.id = ?1",
            params![id],
            |r| Ok((r.get(0)?,)),
        )?;
        Ok(Message {
            id,
            username,
            body: body.to_string(),
        })
    }

    /// Store a direct message for `to_user_id`, to be delivered on the
    /// recipient's next login if they are offline now.
    pub fn store_dm(
        &self,
        from_user_id: i64,
        to_user_id: i64,
        body: &str,
    ) -> rusqlite::Result<i64> {
        let conn = self.lock();
        conn.execute(
            "INSERT INTO direct_messages (from_user, to_user, body) VALUES (?1, ?2, ?3)",
            params![from_user_id, to_user_id, body],
        )?;
        Ok(conn.last_insert_rowid())
    }

    /// Mark every stored DM to `user_id` as delivered.
    pub fn mark_dms_delivered(&self, user_id: i64) -> rusqlite::Result<usize> {
        let conn = self.lock();
        conn.execute(
            "UPDATE direct_messages SET delivered = 1 WHERE to_user = ?1 AND delivered = 0",
            params![user_id],
        )
    }

    /// Undelivered DMs for `user_id`, oldest first, with the sender's name.
    pub fn undelivered_dms(&self, user_id: i64) -> rusqlite::Result<Vec<(String, String)>> {
        let conn = self.lock();
        let mut stmt = conn.prepare(
            "SELECT u.username, d.body
             FROM direct_messages d
             JOIN users u ON u.id = d.from_user
             WHERE d.to_user = ?1 AND d.delivered = 0
             ORDER BY d.id",
        )?;
        let rows = stmt
            .query_map(params![user_id], |r| Ok((r.get(0)?, r.get(1)?)))?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(rows)
    }

    /// Keep only the newest `keep` DMs per recipient, mirroring channel
    /// retention so the `direct_messages` table is also bounded.
    ///
    /// Uses a window function to rank each recipient's rows once, so the whole
    /// prune is a single pass over the table (see `prune_messages`).
    pub fn prune_dms(&self, keep: i64) -> rusqlite::Result<usize> {
        if keep <= 0 {
            return Ok(0);
        }
        let conn = self.lock();
        conn.execute(
            "DELETE FROM direct_messages
             WHERE id IN (
                 SELECT id FROM (
                     SELECT id,
                            ROW_NUMBER() OVER (
                                PARTITION BY to_user
                                ORDER BY id DESC
                            ) AS rn
                     FROM direct_messages
                 )
                 WHERE rn > ?1
             )",
            params![keep],
        )
    }

    /// Fetch up to `limit` messages for a channel, newest last. With `before`,
    /// only messages with a smaller id are returned, which lets a client page
    /// backwards instead of re-reading the newest page forever.
    ///
    /// Two separate queries are used (one with a `before` bound, one without)
    /// so both can use the `(channel_id, id)` index. A single query with
    /// `(?2 IS NULL OR m.id < ?2)` cannot use the index efficiently when `?2`
    /// is NULL, because the OR defeats the range scan.
    pub fn channel_history_before(
        &self,
        channel_id: i64,
        limit: i64,
        before: Option<i64>,
    ) -> rusqlite::Result<Vec<Message>> {
        let conn = self.lock();
        let mut rows = match before {
            Some(before) => {
                let mut stmt = conn.prepare(
                    "SELECT m.id, u.username, m.body
                     FROM messages m
                     JOIN users u ON u.id = m.user_id
                     WHERE m.channel_id = ?1 AND m.id < ?2
                     ORDER BY m.id DESC
                     LIMIT ?3",
                )?;
                stmt.query_map(params![channel_id, before, limit], |r| {
                    Ok(Message {
                        id: r.get(0)?,
                        username: r.get(1)?,
                        body: r.get(2)?,
                    })
                })?
                .collect::<rusqlite::Result<Vec<_>>>()?
            }
            None => {
                let mut stmt = conn.prepare(
                    "SELECT m.id, u.username, m.body
                     FROM messages m
                     JOIN users u ON u.id = m.user_id
                     WHERE m.channel_id = ?1
                     ORDER BY m.id DESC
                     LIMIT ?2",
                )?;
                stmt.query_map(params![channel_id, limit], |r| {
                    Ok(Message {
                        id: r.get(0)?,
                        username: r.get(1)?,
                        body: r.get(2)?,
                    })
                })?
                .collect::<rusqlite::Result<Vec<_>>>()?
            }
        };
        rows.reverse();
        Ok(rows)
    }

    /// Keep only the newest `keep` messages of every channel, returning how many
    /// rows were deleted. Without this the `messages` table grows forever, since
    /// nothing else ever removes a message.
    ///
    /// Uses a window function to rank each channel's rows once, so the whole
    /// prune is a single pass over the table. The older `NOT IN (SELECT …
    /// LIMIT)` form re-ran the per-channel subquery for every row, which is
    /// O(n²) on large tables.
    pub fn prune_messages(&self, keep: i64) -> rusqlite::Result<usize> {
        if keep <= 0 {
            return Ok(0);
        }
        let conn = self.lock();
        conn.execute(
            "DELETE FROM messages
             WHERE id IN (
                 SELECT id FROM (
                     SELECT id,
                            ROW_NUMBER() OVER (
                                PARTITION BY channel_id
                                ORDER BY id DESC
                            ) AS rn
                     FROM messages
                 )
                 WHERE rn > ?1
             )",
            params![keep],
        )
    }

    /// Total number of stored messages, for tests and operational checks.
    pub fn message_count(&self) -> rusqlite::Result<i64> {
        self.lock()
            .query_row("SELECT COUNT(*) FROM messages", [], |r| r.get(0))
    }

    /// Resolve a channel by name, if it exists.
    pub fn find_channel(&self, name: &str) -> rusqlite::Result<Option<Channel>> {
        let conn = self.lock();
        let row = conn
            .query_row(
                "SELECT id, name FROM channels WHERE name = ?1",
                params![name],
                |r| {
                    Ok(Channel {
                        id: r.get(0)?,
                        name: r.get(1)?,
                    })
                },
            )
            .optional()?;
        Ok(row)
    }

    /// List the usernames of all members of a channel.
    pub fn channel_members(&self, channel_id: i64) -> rusqlite::Result<Vec<String>> {
        let conn = self.lock();
        let mut stmt = conn.prepare(
            "SELECT u.username
             FROM channel_members m
             JOIN users u ON u.id = m.user_id
             WHERE m.channel_id = ?1
             ORDER BY u.username",
        )?;
        let rows = stmt
            .query_map(params![channel_id], |r| r.get(0))?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(rows)
    }

    /// List the members of a channel with their operator status, so the web UI
    /// can render `@op` markers in the roster. Ordered by username.
    pub fn channel_members_with_ops(
        &self,
        channel_id: i64,
    ) -> rusqlite::Result<Vec<(String, bool)>> {
        let conn = self.lock();
        let mut stmt = conn.prepare(
            "SELECT u.username, m.is_op
             FROM channel_members m
             JOIN users u ON u.id = m.user_id
             WHERE m.channel_id = ?1
             ORDER BY u.username",
        )?;
        let rows = stmt
            .query_map(params![channel_id], |r| {
                Ok((r.get(0)?, r.get::<_, i64>(1)? != 0))
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(rows)
    }

    /// Set a channel's topic.
    pub fn set_topic(&self, channel_id: i64, topic: &str) -> rusqlite::Result<()> {
        let conn = self.lock();
        conn.execute(
            "UPDATE channels SET topic = ?1 WHERE id = ?2",
            params![topic, channel_id],
        )?;
        Ok(())
    }

    /// Get a channel's topic.
    pub fn get_topic(&self, channel_id: i64) -> rusqlite::Result<Option<String>> {
        let conn = self.lock();
        let topic = conn
            .query_row(
                "SELECT topic FROM channels WHERE id = ?1",
                params![channel_id],
                |r| r.get(0),
            )
            .optional()?;
        Ok(topic)
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    fn temp_db() -> Db {
        let path = std::env::temp_dir().join(format!(
            "chat-db-test-{}-{:?}.sqlite",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_file(&path);
        Db::open(&path).expect("open temp db")
    }

    #[test]
    fn history_returns_ids_and_pages_backwards() {
        let db = temp_db();
        let user = db.register_user("alice", "hash").unwrap();
        let ch = db.create_channel("#general").unwrap();
        db.join_channel(ch.id, user.id).unwrap();
        for body in ["one", "two", "three", "four"] {
            db.store_message(ch.id, user.id, body).unwrap();
        }

        let newest = db.channel_history_before(ch.id, 2, None).unwrap();
        assert_eq!(
            newest.iter().map(|m| m.body.as_str()).collect::<Vec<_>>(),
            vec!["three", "four"],
            "newest page should be the last two, oldest-first"
        );

        // Paging before the oldest row of the previous page returns the earlier two.
        let cursor = newest[0].id;
        let older = db.channel_history_before(ch.id, 2, Some(cursor)).unwrap();
        assert_eq!(
            older.iter().map(|m| m.body.as_str()).collect::<Vec<_>>(),
            vec!["one", "two"]
        );
        assert!(
            older.iter().all(|m| m.id < cursor),
            "cursor must exclude the row it points at"
        );
    }

    #[test]
    fn prune_messages_keeps_the_newest_per_channel() {
        let db = temp_db();
        let user = db.register_user("alice", "hash").unwrap();
        let a = db.create_channel("#a").unwrap();
        let b = db.create_channel("#b").unwrap();
        for i in 0..10 {
            db.store_message(a.id, user.id, &format!("a{i}")).unwrap();
            db.store_message(b.id, user.id, &format!("b{i}")).unwrap();
        }

        let deleted = db.prune_messages(3).unwrap();
        assert_eq!(deleted, 14, "7 of 10 removed from each channel");

        // Each channel independently keeps its newest rows.
        let kept_a = db.channel_history_before(a.id, 100, None).unwrap();
        assert_eq!(
            kept_a.iter().map(|m| m.body.as_str()).collect::<Vec<_>>(),
            vec!["a7", "a8", "a9"]
        );
        let kept_b = db.channel_history_before(b.id, 100, None).unwrap();
        assert_eq!(
            kept_b.iter().map(|m| m.body.as_str()).collect::<Vec<_>>(),
            vec!["b7", "b8", "b9"]
        );

        // A second pass finds nothing left to do.
        assert_eq!(db.prune_messages(3).unwrap(), 0);
    }

    #[test]
    fn channel_members_with_ops_reports_operator_status() {
        let db = temp_db();
        let alice = db.register_user("alice", "hash").unwrap();
        let bob = db.register_user("bob", "hash").unwrap();
        let ch = db.create_channel("#general").unwrap();
        // First member becomes an op.
        db.join_channel(ch.id, alice.id).unwrap();
        db.join_channel(ch.id, bob.id).unwrap();

        let members = db.channel_members_with_ops(ch.id).unwrap();
        assert_eq!(
            members,
            vec![("alice".to_string(), true), ("bob".to_string(), false)]
        );

        // Promoting bob flips his flag.
        db.set_op(ch.id, bob.id, true).unwrap();
        let members = db.channel_members_with_ops(ch.id).unwrap();
        assert_eq!(
            members,
            vec![("alice".to_string(), true), ("bob".to_string(), true)]
        );
    }
}
