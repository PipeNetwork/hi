use std::cell::RefCell;
use std::io::{Read as _, Write as _};
use std::path::{Path, PathBuf};

use sha2::Digest as _;

/// Per-segment (and therefore per-record) resource limit, not a journal lifetime limit.
pub const MAX_JOURNAL_BYTES: u64 = 64 * 1024 * 1024;
const SEGMENT_SUFFIX: &str = ".segment-";
#[cfg(unix)]
const PRIVATE_DIR_MODE: u32 = 0o700;
#[cfg(unix)]
const PRIVATE_FILE_MODE: u32 = 0o600;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct JournalEntry {
    pub seq: u64,
    pub kind: String,
    pub req_hash: String,
    pub result: serde_json::Value,
    pub at_ms: u64,
}

#[derive(Debug, thiserror::Error)]
pub enum JournalError {
    #[error("journal io: {0}")]
    Io(#[from] std::io::Error),
    #[error("journal parse at line {line}: {error}")]
    Parse { line: usize, error: String },
    #[error("journal restore rejected (limit {limit}): {reason}")]
    UnsafeRestore { limit: u64, reason: String },
    #[error(
        "journal record at seq {seq} exceeds the {limit}-byte per-record limit; the journal has no \
         lifetime size limit"
    )]
    Full { seq: u64, limit: u64 },
    #[error("journal is not dense at entry {index}: expected sequence {expected}, found {actual}")]
    Sequence {
        index: usize,
        expected: u64,
        actual: u64,
    },
    #[error(
        "replay divergence at seq {seq} ({kind}): the script issued a different call than the \
         recorded run — the workflow script is nondeterministic or was edited mid-run"
    )]
    Divergence { seq: u64, kind: String },
}

#[derive(Debug)]
struct JournalSegment {
    path: PathBuf,
    start_seq: u64,
    entry_count: u64,
    bytes: u64,
    identity: std::fs::Metadata,
}

#[derive(Debug)]
struct CachedSegment {
    index: usize,
    entries: Vec<JournalEntry>,
}

#[derive(Debug)]
pub struct Journal {
    // Entries are retained directly only for deliberately non-persistent journals. Persistent
    // journals replay from a single bounded segment cache instead of retaining the whole run.
    entries: Vec<JournalEntry>,
    path: Option<PathBuf>,
    segments: Vec<JournalSegment>,
    entry_count: u64,
    agent_reservations: u64,
    segment_limit: u64,
    cache: RefCell<Option<CachedSegment>>,
}

impl Default for Journal {
    fn default() -> Self {
        Self::new(None)
    }
}

impl Journal {
    pub fn new(path: Option<PathBuf>) -> Self {
        Self {
            entries: Vec::new(),
            path,
            segments: Vec::new(),
            entry_count: 0,
            agent_reservations: 0,
            segment_limit: MAX_JOURNAL_BYTES,
            cache: RefCell::new(None),
        }
    }

    #[cfg(test)]
    fn with_segment_limit(path: Option<PathBuf>, segment_limit: u64) -> Self {
        assert!(segment_limit > 0);
        Self {
            segment_limit,
            ..Self::new(path)
        }
    }

    pub fn load(path: PathBuf) -> Result<Self, JournalError> {
        let paths = discover_segments(&path).map_err(map_restore_error)?;
        let mut segments = Vec::with_capacity(paths.len());
        let mut entry_count = 0u64;
        let mut agent_reservations = 0u64;
        let mut line_number = 0usize;
        let last = paths.len().saturating_sub(1);
        for (index, segment_path) in paths.into_iter().enumerate() {
            let summary = load_segment(&segment_path, entry_count, &mut line_number, index == last)
                .map_err(|error| match error {
                    JournalError::Io(error) if error.kind() == std::io::ErrorKind::InvalidData => {
                        map_restore_error(error)
                    }
                    other => other,
                })?;
            segments.push(JournalSegment {
                path: segment_path,
                start_seq: entry_count,
                entry_count: summary.entry_count,
                bytes: summary.bytes,
                identity: summary.identity,
            });
            entry_count = entry_count
                .checked_add(summary.entry_count)
                .ok_or_else(|| JournalError::UnsafeRestore {
                    limit: u64::MAX,
                    reason: "journal sequence count overflow".into(),
                })?;
            agent_reservations = agent_reservations.saturating_add(summary.agent_reservations);
        }
        Ok(Self {
            entries: Vec::new(),
            path: Some(path),
            segments,
            entry_count,
            agent_reservations,
            segment_limit: MAX_JOURNAL_BYTES,
            cache: RefCell::new(None),
        })
    }

    pub fn len(&self) -> usize {
        usize::try_from(self.entry_count).unwrap_or(usize::MAX)
    }

    pub fn is_empty(&self) -> bool {
        self.entry_count == 0
    }

    pub fn agent_reservation_count(&self) -> u64 {
        self.agent_reservations
    }

    pub fn covers(&self, seq: u64) -> bool {
        seq < self.entry_count
    }

    pub fn replay(
        &self,
        seq: u64,
        kind: &str,
        req_hash: &str,
    ) -> Result<Option<serde_json::Value>, JournalError> {
        if !self.covers(seq) {
            return Ok(None);
        }
        if self.path.is_none() {
            return replay_entry(&self.entries[seq as usize], seq, kind, req_hash);
        }
        let segment_index = self.segments.partition_point(|segment| {
            segment.start_seq.saturating_add(segment.entry_count) <= seq
        });
        let segment = self
            .segments
            .get(segment_index)
            .ok_or_else(|| JournalError::Sequence {
                index: usize::try_from(seq).unwrap_or(usize::MAX),
                expected: seq,
                actual: self.entry_count,
            })?;
        let needs_load = self
            .cache
            .borrow()
            .as_ref()
            .is_none_or(|cache| cache.index != segment_index);
        if needs_load {
            let entries = read_segment_entries(segment)?;
            *self.cache.borrow_mut() = Some(CachedSegment {
                index: segment_index,
                entries,
            });
        }
        let cache = self.cache.borrow();
        let entry = &cache.as_ref().expect("segment cache was populated").entries
            [usize::try_from(seq - segment.start_seq).unwrap()];
        replay_entry(entry, seq, kind, req_hash)
    }

    pub fn record(
        &mut self,
        seq: u64,
        kind: &str,
        req_hash: String,
        result: serde_json::Value,
    ) -> Result<(), JournalError> {
        let entry = JournalEntry {
            seq,
            kind: kind.to_string(),
            req_hash,
            result,
            at_ms: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis() as u64)
                .unwrap_or(0),
        };
        validate_sequence(self.entry_count, &entry)?;
        let mut line = serde_json::to_string(&entry)
            .map_err(|error| JournalError::Io(std::io::Error::other(error)))?;
        line.push('\n');
        let line_len = line.len() as u64;
        if line_len > self.segment_limit {
            return Err(JournalError::Full {
                seq,
                limit: self.segment_limit,
            });
        }
        if let Some(base_path) = &self.path {
            let rotate = self
                .segments
                .last()
                .is_some_and(|segment| segment.bytes.saturating_add(line_len) > self.segment_limit);
            let segment_index = if rotate {
                self.segments.len()
            } else {
                self.segments.len().saturating_sub(1)
            };
            let target = if self.segments.is_empty() || rotate {
                segment_path(base_path, segment_index)?
            } else {
                self.segments[segment_index].path.clone()
            };
            let expected = if self.segments.is_empty() || rotate {
                None
            } else {
                Some(&self.segments[segment_index].identity)
            };
            let metadata = append_line(&target, &line, expected)?;
            if self.segments.is_empty() || rotate {
                self.segments.push(JournalSegment {
                    path: target,
                    start_seq: seq,
                    entry_count: 1,
                    bytes: line_len,
                    identity: metadata,
                });
            } else {
                let segment = &mut self.segments[segment_index];
                segment.entry_count += 1;
                segment.bytes += line_len;
                segment.identity = metadata;
            }
            *self.cache.get_mut() = None;
        } else {
            self.entries.push(entry);
        }
        self.entry_count += 1;
        if kind == "spawn_agent" {
            self.agent_reservations = self.agent_reservations.saturating_add(1);
        }
        Ok(())
    }
}

fn replay_entry(
    entry: &JournalEntry,
    seq: u64,
    kind: &str,
    req_hash: &str,
) -> Result<Option<serde_json::Value>, JournalError> {
    if entry.seq != seq || entry.kind != kind || entry.req_hash != req_hash {
        return Err(JournalError::Divergence {
            seq,
            kind: kind.to_string(),
        });
    }
    Ok(Some(entry.result.clone()))
}

struct SegmentSummary {
    entry_count: u64,
    agent_reservations: u64,
    bytes: u64,
    identity: std::fs::Metadata,
}

fn map_restore_error(error: std::io::Error) -> JournalError {
    if error.kind() == std::io::ErrorKind::InvalidData {
        JournalError::UnsafeRestore {
            limit: MAX_JOURNAL_BYTES,
            reason: error.to_string(),
        }
    } else {
        JournalError::Io(error)
    }
}

fn discover_segments(base: &Path) -> std::io::Result<Vec<PathBuf>> {
    let Some(file_name) = base.file_name().and_then(|name| name.to_str()) else {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("journal path has no UTF-8 file name: {}", base.display()),
        ));
    };
    let parent = base.parent().filter(|path| !path.as_os_str().is_empty());
    let base_exists = match std::fs::symlink_metadata(base) {
        Ok(_) => true,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => false,
        Err(error) => return Err(error),
    };
    let Some(parent) = parent else {
        return Ok(if base_exists {
            vec![base.to_path_buf()]
        } else {
            Vec::new()
        });
    };
    let parent_metadata = match std::fs::symlink_metadata(parent) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound && !base_exists => {
            return Ok(Vec::new());
        }
        Err(error) => return Err(error),
    };
    if parent_metadata.file_type().is_symlink() || !parent_metadata.is_dir() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("journal parent is not a directory: {}", parent.display()),
        ));
    }
    let prefix = format!("{file_name}{SEGMENT_SUFFIX}");
    let mut indexed = Vec::new();
    for item in std::fs::read_dir(parent)? {
        let item = item?;
        let name = item.file_name();
        let Some(name) = name.to_str() else {
            continue;
        };
        let Some(suffix) = name.strip_prefix(&prefix) else {
            continue;
        };
        if suffix.len() != 16 || !suffix.bytes().all(|byte| byte.is_ascii_digit()) {
            continue;
        }
        let index = suffix.parse::<usize>().map_err(|error| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("invalid journal segment index {suffix}: {error}"),
            )
        })?;
        if index == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "journal segment zero must use the legacy base file name",
            ));
        }
        indexed.push((index, item.path()));
    }
    indexed.sort_unstable_by_key(|(index, _)| *index);
    if !indexed.is_empty() && !base_exists {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "journal continuation exists without its base segment",
        ));
    }
    for (expected, (actual, _)) in (1..).zip(&indexed) {
        if expected != *actual {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("journal segment gap: expected {expected}, found {actual}"),
            ));
        }
    }
    let mut paths = Vec::with_capacity(indexed.len() + usize::from(base_exists));
    if base_exists {
        paths.push(base.to_path_buf());
    }
    paths.extend(indexed.into_iter().map(|(_, path)| path));
    Ok(paths)
}

fn segment_path(base: &Path, index: usize) -> std::io::Result<PathBuf> {
    if index == 0 {
        return Ok(base.to_path_buf());
    }
    let Some(file_name) = base.file_name().and_then(|name| name.to_str()) else {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("journal path has no UTF-8 file name: {}", base.display()),
        ));
    };
    Ok(base.with_file_name(format!("{file_name}{SEGMENT_SUFFIX}{index:016}")))
}

fn load_segment(
    path: &Path,
    start_seq: u64,
    line_number: &mut usize,
    is_final: bool,
) -> Result<SegmentSummary, JournalError> {
    let (content, mut identity) = read_segment_bounded(path, None)?;
    let mut offset = 0usize;
    let mut entry_count = 0u64;
    let mut agent_reservations = 0u64;
    let mut bytes = content.len() as u64;
    while offset < content.len() {
        *line_number = line_number.saturating_add(1);
        let Some(relative_newline) = content[offset..].iter().position(|byte| *byte == b'\n')
        else {
            if !is_final {
                return Err(JournalError::Parse {
                    line: *line_number,
                    error: "unterminated line in a non-final journal segment".into(),
                });
            }
            let tail = &content[offset..];
            if tail.iter().all(u8::is_ascii_whitespace) {
                truncate_tail(path, offset as u64, Some(&identity))?;
                bytes = offset as u64;
            } else {
                match serde_json::from_slice::<JournalEntry>(tail) {
                    Ok(entry) => {
                        validate_sequence(start_seq + entry_count, &entry)?;
                        if entry.kind == "spawn_agent" {
                            agent_reservations = agent_reservations.saturating_add(1);
                        }
                        entry_count += 1;
                        terminate_line(path, Some(&identity))?;
                        bytes += 1;
                    }
                    Err(error) => {
                        tracing::warn!(
                            line = *line_number,
                            %error,
                            "truncating torn workflow journal tail"
                        );
                        truncate_tail(path, offset as u64, Some(&identity))?;
                        bytes = offset as u64;
                    }
                }
            }
            identity = open_existing_journal(path, false, None)?.1;
            break;
        };
        let end = offset + relative_newline;
        let line = &content[offset..end];
        offset = end + 1;
        if line.iter().all(u8::is_ascii_whitespace) {
            continue;
        }
        let entry =
            serde_json::from_slice::<JournalEntry>(line).map_err(|error| JournalError::Parse {
                line: *line_number,
                error: error.to_string(),
            })?;
        validate_sequence(start_seq + entry_count, &entry)?;
        if entry.kind == "spawn_agent" {
            agent_reservations = agent_reservations.saturating_add(1);
        }
        entry_count += 1;
    }
    Ok(SegmentSummary {
        entry_count,
        agent_reservations,
        bytes,
        identity,
    })
}

fn read_segment_entries(segment: &JournalSegment) -> Result<Vec<JournalEntry>, JournalError> {
    let (content, metadata) = read_segment_bounded(&segment.path, Some(&segment.identity))?;
    if metadata.len() != segment.bytes || content.len() as u64 != segment.bytes {
        return Err(JournalError::Io(invalid_journal_file(&segment.path)));
    }
    let capacity = usize::try_from(segment.entry_count)
        .unwrap_or(content.len())
        .min(content.len());
    let mut entries = Vec::with_capacity(capacity);
    for line in content.split(|byte| *byte == b'\n') {
        if line.iter().all(u8::is_ascii_whitespace) {
            continue;
        }
        let entry = serde_json::from_slice::<JournalEntry>(line).map_err(|error| {
            JournalError::Io(std::io::Error::new(std::io::ErrorKind::InvalidData, error))
        })?;
        validate_sequence(segment.start_seq + entries.len() as u64, &entry)?;
        entries.push(entry);
    }
    if entries.len() as u64 != segment.entry_count {
        return Err(JournalError::Io(invalid_journal_file(&segment.path)));
    }
    Ok(entries)
}

fn read_segment_bounded(
    path: &Path,
    expected: Option<&std::fs::Metadata>,
) -> std::io::Result<(Vec<u8>, std::fs::Metadata)> {
    let (file, metadata) = open_existing_journal(path, false, expected)?;
    if metadata.len() > MAX_JOURNAL_BYTES {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("journal exceeds {MAX_JOURNAL_BYTES} bytes"),
        ));
    }
    let mut content = Vec::with_capacity(metadata.len() as usize);
    file.take(MAX_JOURNAL_BYTES.saturating_add(1))
        .read_to_end(&mut content)?;
    if content.len() as u64 > MAX_JOURNAL_BYTES {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("journal exceeds {MAX_JOURNAL_BYTES} bytes"),
        ));
    }
    Ok((content, metadata))
}

fn validate_sequence(expected: u64, entry: &JournalEntry) -> Result<(), JournalError> {
    if entry.seq != expected {
        return Err(JournalError::Sequence {
            index: usize::try_from(expected).unwrap_or(usize::MAX),
            expected,
            actual: entry.seq,
        });
    }
    Ok(())
}

fn truncate_tail(
    path: &Path,
    len: u64,
    expected: Option<&std::fs::Metadata>,
) -> std::io::Result<()> {
    let (file, _) = open_existing_journal(path, true, expected)?;
    file.set_len(len)?;
    file.sync_data()
}

fn terminate_line(path: &Path, expected: Option<&std::fs::Metadata>) -> std::io::Result<()> {
    let (mut file, _) = open_existing_journal(path, true, expected)?;
    file.write_all(b"\n")?;
    file.sync_data()
}

fn append_line(
    path: &Path,
    line: &str,
    expected: Option<&std::fs::Metadata>,
) -> std::io::Result<std::fs::Metadata> {
    ensure_journal_parent(path)?;
    let before = match std::fs::symlink_metadata(path) {
        Ok(metadata) => Some(metadata),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
        Err(error) => return Err(error),
    };
    let mut options = std::fs::OpenOptions::new();
    options.append(true);
    if expected.is_none() {
        options.create_new(true);
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt as _;
        options
            .mode(PRIVATE_FILE_MODE)
            .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC);
    }
    if before.as_ref().is_some_and(|metadata| {
        metadata.file_type().is_symlink()
            || !metadata.is_file()
            || expected.is_none_or(|expected| !same_file(expected, metadata))
    }) || (before.is_none() && expected.is_some())
    {
        return Err(invalid_journal_file(path));
    }
    let mut file = options.open(path)?;
    let opened = file.metadata()?;
    if !opened.is_file()
        || before
            .as_ref()
            .is_some_and(|before| !same_file(before, &opened))
        || expected.is_some_and(|expected| !same_file(expected, &opened))
    {
        return Err(invalid_journal_file(path));
    }
    tighten_private_file(&file)?;
    file.write_all(line.as_bytes())?;
    file.sync_data()?;
    let metadata = file.metadata()?;
    if expected.is_none() {
        sync_parent_dir(path)?;
    }
    Ok(metadata)
}

fn sync_parent_dir(path: &Path) -> std::io::Result<()> {
    let Some(parent) = path.parent().filter(|path| !path.as_os_str().is_empty()) else {
        return Ok(());
    };
    let before = std::fs::symlink_metadata(parent)?;
    if before.file_type().is_symlink() || !before.is_dir() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("journal parent is not a directory: {}", parent.display()),
        ));
    }
    let mut options = std::fs::OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt as _;
        options.custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC | libc::O_DIRECTORY);
    }
    let directory = options.open(parent)?;
    let opened = directory.metadata()?;
    if !opened.is_dir() || !same_file(&before, &opened) {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "journal parent is not a stable directory: {}",
                parent.display()
            ),
        ));
    }
    directory.sync_data()
}

fn open_existing_journal(
    path: &Path,
    write: bool,
    expected: Option<&std::fs::Metadata>,
) -> std::io::Result<(std::fs::File, std::fs::Metadata)> {
    let before = std::fs::symlink_metadata(path)?;
    if before.file_type().is_symlink() || !before.is_file() {
        return Err(invalid_journal_file(path));
    }
    let mut options = std::fs::OpenOptions::new();
    if write {
        options.append(true);
    } else {
        options.read(true);
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt as _;
        options.custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC);
    }
    let file = options.open(path)?;
    let opened = file.metadata()?;
    if !opened.is_file()
        || !same_file(&before, &opened)
        || expected.is_some_and(|expected| !same_file(expected, &opened))
    {
        return Err(invalid_journal_file(path));
    }
    tighten_private_file(&file)?;
    Ok((file, opened))
}

fn invalid_journal_file(path: &Path) -> std::io::Error {
    std::io::Error::new(
        std::io::ErrorKind::InvalidData,
        format!("journal is not a stable regular file: {}", path.display()),
    )
}

#[cfg(unix)]
fn same_file(left: &std::fs::Metadata, right: &std::fs::Metadata) -> bool {
    use std::os::unix::fs::MetadataExt as _;
    left.dev() == right.dev() && left.ino() == right.ino()
}

#[cfg(not(unix))]
fn same_file(_left: &std::fs::Metadata, _right: &std::fs::Metadata) -> bool {
    true
}

fn tighten_private_file(file: &std::fs::File) -> std::io::Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt as _;
        file.set_permissions(std::fs::Permissions::from_mode(PRIVATE_FILE_MODE))?;
    }
    Ok(())
}

fn ensure_journal_parent(path: &Path) -> std::io::Result<()> {
    let Some(parent) = path.parent() else {
        return Ok(());
    };
    if parent.as_os_str().is_empty() {
        return Ok(());
    }
    #[cfg(unix)]
    let mut created = false;
    match std::fs::symlink_metadata(parent) {
        Ok(metadata) if metadata.file_type().is_symlink() || !metadata.is_dir() => {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("journal parent is not a directory: {}", parent.display()),
            ));
        }
        Ok(_) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            let mut builder = std::fs::DirBuilder::new();
            builder.recursive(true);
            #[cfg(unix)]
            {
                use std::os::unix::fs::DirBuilderExt as _;
                builder.mode(PRIVATE_DIR_MODE);
            }
            builder.create(parent)?;
            #[cfg(unix)]
            {
                created = true;
            }
        }
        Err(error) => return Err(error),
    }
    let metadata = std::fs::symlink_metadata(parent)?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("journal parent is not a directory: {}", parent.display()),
        ));
    }
    #[cfg(unix)]
    if created {
        use std::os::unix::fs::PermissionsExt as _;
        std::fs::set_permissions(parent, std::fs::Permissions::from_mode(PRIVATE_DIR_MODE))?;
    }
    Ok(())
}

fn canonical_json(value: &serde_json::Value) -> serde_json::Value {
    match value {
        serde_json::Value::Object(map) => {
            let mut entries: Vec<(&String, &serde_json::Value)> = map.iter().collect();
            entries.sort_unstable_by(|a, b| a.0.cmp(b.0));
            serde_json::Value::Object(
                entries
                    .into_iter()
                    .map(|(k, v)| (k.clone(), canonical_json(v)))
                    .collect(),
            )
        }
        serde_json::Value::Array(items) => {
            serde_json::Value::Array(items.iter().map(canonical_json).collect())
        }
        other => other.clone(),
    }
}

pub fn request_hash(kind: &str, payload: &serde_json::Value) -> String {
    let mut hasher = sha2::Sha256::new();
    hasher.update(kind.as_bytes());
    hasher.update([0u8]);
    hasher.update(canonical_json(payload).to_string().as_bytes());
    let digest = hasher.finalize();
    let mut out = String::with_capacity(32);
    for byte in digest.iter().take(16) {
        out.push_str(&format!("{byte:02x}"));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn record_and_replay_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("journal.jsonl");
        let mut journal = Journal::new(Some(path.clone()));
        let hash = request_hash("spawn_agent", &serde_json::json!({"prompt": "hi"}));
        journal
            .record(
                0,
                "spawn_agent",
                hash.clone(),
                serde_json::json!({"ok": true}),
            )
            .unwrap();

        let loaded = Journal::load(path).unwrap();
        assert_eq!(loaded.len(), 1);
        let replayed = loaded.replay(0, "spawn_agent", &hash).unwrap();
        assert_eq!(replayed, Some(serde_json::json!({"ok": true})));
        assert!(loaded.replay(1, "spawn_agent", &hash).unwrap().is_none());
    }

    #[test]
    fn restore_accepts_more_than_the_legacy_ten_thousand_entries() {
        const ENTRIES: usize = 10_001;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("journal.jsonl");
        let mut content = String::new();
        for seq in 0..ENTRIES {
            let entry = JournalEntry {
                seq: seq as u64,
                kind: "budget".into(),
                req_hash: "hash".into(),
                result: serde_json::Value::Null,
                at_ms: 0,
            };
            content.push_str(&serde_json::to_string(&entry).unwrap());
            content.push('\n');
        }
        std::fs::write(&path, content).unwrap();

        let restored = Journal::load(path).unwrap();

        assert_eq!(restored.len(), ENTRIES);
        assert!(restored.covers(ENTRIES as u64 - 1));
    }

    #[test]
    fn divergence_on_hash_mismatch() {
        let mut journal = Journal::new(None);
        journal
            .record(0, "spawn_agent", "aaaa".into(), serde_json::json!(1))
            .unwrap();
        assert!(matches!(
            journal.replay(0, "spawn_agent", "bbbb"),
            Err(JournalError::Divergence { seq: 0, .. })
        ));
        assert!(matches!(
            journal.replay(0, "budget", "aaaa"),
            Err(JournalError::Divergence { seq: 0, .. })
        ));
    }

    #[test]
    fn torn_tail_is_truncated_before_the_next_append() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("journal.jsonl");
        let first = "{\"seq\":0,\"kind\":\"log\",\"req_hash\":\"x\",\"result\":null,\"at_ms\":1}\n";
        std::fs::write(&path, format!("{first}{{\"seq\":1,\"kind")).unwrap();

        let mut journal = Journal::load(path.clone()).unwrap();
        assert_eq!(journal.len(), 1);
        assert_eq!(std::fs::read_to_string(&path).unwrap(), first);
        journal
            .record(1, "log", "y".into(), serde_json::Value::Null)
            .unwrap();

        assert_eq!(Journal::load(path).unwrap().len(), 2);
    }

    #[test]
    fn valid_unterminated_tail_is_kept_and_terminated() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("journal.jsonl");
        let line = "{\"seq\":0,\"kind\":\"log\",\"req_hash\":\"x\",\"result\":null,\"at_ms\":1}";
        std::fs::write(&path, line).unwrap();

        assert_eq!(Journal::load(path.clone()).unwrap().len(), 1);
        assert_eq!(std::fs::read_to_string(path).unwrap(), format!("{line}\n"));
    }

    #[cfg(unix)]
    #[test]
    fn load_rejects_symlink_journal() {
        use std::os::unix::fs::symlink;

        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("target.jsonl");
        let linked = dir.path().join("journal.jsonl");
        std::fs::write(&target, "").unwrap();
        symlink(&target, &linked).unwrap();
        assert!(matches!(
            Journal::load(linked),
            Err(JournalError::UnsafeRestore { .. })
        ));
    }

    #[cfg(unix)]
    #[test]
    fn journal_is_owner_only_and_existing_mode_is_tightened() {
        use std::os::unix::fs::{MetadataExt as _, PermissionsExt as _};

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("journal.jsonl");
        let mut journal = Journal::new(Some(path.clone()));
        journal
            .record(0, "log", "x".into(), serde_json::Value::Null)
            .unwrap();
        assert_eq!(std::fs::metadata(&path).unwrap().mode() & 0o777, 0o600);

        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o666)).unwrap();
        Journal::load(path.clone()).unwrap();
        assert_eq!(std::fs::metadata(path).unwrap().mode() & 0o777, 0o600);
    }

    #[cfg(unix)]
    #[test]
    fn append_does_not_follow_preplanted_journal_symlink() {
        use std::os::unix::fs::symlink;

        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("target");
        let linked = dir.path().join("journal.jsonl");
        std::fs::write(&target, b"do not append").unwrap();
        symlink(&target, &linked).unwrap();
        let mut journal = Journal::new(Some(linked));

        assert!(matches!(
            journal.record(0, "log", "x".into(), serde_json::Value::Null),
            Err(JournalError::Io(_))
        ));
        assert_eq!(std::fs::read(target).unwrap(), b"do not append");
        assert!(journal.is_empty());
    }

    #[cfg(unix)]
    #[test]
    fn load_rejects_symlinked_continuation_segment() {
        use std::os::unix::fs::symlink;

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("journal.jsonl");
        let target = dir.path().join("target.jsonl");
        std::fs::write(
            &path,
            b"{\"seq\":0,\"kind\":\"log\",\"req_hash\":\"x\",\"result\":null,\"at_ms\":1}\n",
        )
        .unwrap();
        std::fs::write(&target, b"").unwrap();
        symlink(&target, segment_path(&path, 1).unwrap()).unwrap();

        assert!(matches!(
            Journal::load(path),
            Err(JournalError::UnsafeRestore { .. })
        ));
    }

    #[test]
    fn preplanted_next_segment_blocks_rotation_without_advancing_memory() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("journal.jsonl");
        let mut journal = Journal::with_segment_limit(Some(path.clone()), 128);
        journal
            .record(0, "log", "x".into(), serde_json::Value::Null)
            .unwrap();
        let continuation = segment_path(&path, 1).unwrap();
        std::fs::write(&continuation, b"do not append").unwrap();

        assert!(matches!(
            journal.record(1, "log", "y".into(), serde_json::Value::Null),
            Err(JournalError::Io(_))
        ));
        assert_eq!(journal.len(), 1);
        assert_eq!(std::fs::read(continuation).unwrap(), b"do not append");
    }

    #[test]
    fn load_rejects_oversize_journal_before_reading() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("journal.jsonl");
        let file = std::fs::File::create(&path).unwrap();
        file.set_len(MAX_JOURNAL_BYTES + 1).unwrap();
        assert!(matches!(
            Journal::load(path),
            Err(JournalError::UnsafeRestore { .. })
        ));
    }

    #[test]
    fn record_refuses_an_individual_record_over_the_segment_limit() {
        let mut journal = Journal::with_segment_limit(None, 128);
        let hash = request_hash("spawn_agent", &serde_json::json!({}));
        let err = journal
            .record(
                0,
                "spawn_agent",
                hash.clone(),
                serde_json::json!("x".repeat(128)),
            )
            .unwrap_err();
        assert!(matches!(err, JournalError::Full { seq: 0, .. }), "{err}");
        journal
            .record(0, "spawn_agent", hash, serde_json::json!({"ok": true}))
            .unwrap();
        assert_eq!(journal.len(), 1);
    }

    #[test]
    fn rotates_without_a_lifetime_cap_and_replays_across_segments() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("journal.jsonl");
        let mut journal = Journal::with_segment_limit(Some(path.clone()), 160);
        let mut hashes = Vec::new();
        for seq in 0..6 {
            let hash = request_hash("log", &serde_json::json!({"seq": seq}));
            journal
                .record(seq, "log", hash.clone(), serde_json::json!({"seq": seq}))
                .unwrap();
            hashes.push(hash);
        }
        assert!(
            journal.segments.len() > 1,
            "test must cross a segment boundary"
        );
        assert_eq!(journal.len(), 6);
        assert!(segment_path(&path, 1).unwrap().is_file());

        let loaded = Journal::load(path).unwrap();
        assert_eq!(loaded.len(), 6);
        assert!(loaded.segments.len() > 1);
        for (seq, hash) in hashes.iter().enumerate() {
            assert_eq!(
                loaded.replay(seq as u64, "log", hash).unwrap(),
                Some(serde_json::json!({"seq": seq}))
            );
        }
    }

    #[test]
    fn torn_tail_in_final_segment_is_recovered_after_rotation() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("journal.jsonl");
        let mut journal = Journal::with_segment_limit(Some(path.clone()), 160);
        for seq in 0..4 {
            journal
                .record(seq, "log", format!("hash-{seq}"), serde_json::json!(seq))
                .unwrap();
        }
        assert!(
            journal.segments.len() > 1,
            "test must cross a segment boundary"
        );
        let final_path = journal.segments.last().unwrap().path.clone();
        let good_len = std::fs::metadata(&final_path).unwrap().len();
        let mut file = std::fs::OpenOptions::new()
            .append(true)
            .open(&final_path)
            .unwrap();
        file.write_all(b"{\"seq\":4,\"kind\"").unwrap();
        file.sync_data().unwrap();

        let loaded = Journal::load(path).unwrap();
        assert_eq!(loaded.len(), 4);
        assert_eq!(std::fs::metadata(final_path).unwrap().len(), good_len);
        assert_eq!(
            loaded.replay(3, "log", "hash-3").unwrap(),
            Some(serde_json::json!(3))
        );
    }

    #[test]
    fn complete_malformed_line_is_not_treated_as_torn() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("journal.jsonl");
        std::fs::write(&path, b"not-json\n").unwrap();
        assert!(matches!(
            Journal::load(path),
            Err(JournalError::Parse { .. })
        ));
    }

    #[test]
    fn load_and_record_require_dense_sequences() {
        let mut journal = Journal::new(None);
        assert!(matches!(
            journal.record(1, "log", "x".into(), serde_json::Value::Null),
            Err(JournalError::Sequence {
                expected: 0,
                actual: 1,
                ..
            })
        ));

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("journal.jsonl");
        std::fs::write(
            &path,
            "{\"seq\":1,\"kind\":\"log\",\"req_hash\":\"x\",\"result\":null,\"at_ms\":1}\n",
        )
        .unwrap();
        assert!(matches!(
            Journal::load(path),
            Err(JournalError::Sequence {
                expected: 0,
                actual: 1,
                ..
            })
        ));
    }

    #[test]
    fn persistence_error_does_not_advance_memory() {
        let dir = tempfile::tempdir().unwrap();
        let mut journal = Journal::new(Some(dir.path().join("journal.jsonl")));
        std::fs::create_dir(dir.path().join("journal.jsonl")).unwrap();
        assert!(matches!(
            journal.record(0, "log", "x".into(), serde_json::Value::Null),
            Err(JournalError::Io(_))
        ));
        assert!(journal.is_empty());
    }

    #[test]
    fn request_hash_is_stable() {
        let a = request_hash("k", &serde_json::json!({"b": 2, "a": 1}));
        let b = request_hash("k", &serde_json::json!({"a": 1, "b": 2}));
        assert_eq!(a, b, "map key order must not affect the hash");
    }
}
