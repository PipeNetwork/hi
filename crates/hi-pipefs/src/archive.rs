use std::collections::{BTreeMap, HashMap};
use std::fs::{self, File, Metadata, OpenOptions};
use std::io::{Cursor, Read, Write};
use std::path::{Component, Path, PathBuf};
#[cfg(not(unix))]
use std::time::UNIX_EPOCH;

#[cfg(unix)]
use std::ffi::{OsStr, OsString};
#[cfg(unix)]
use std::os::unix::ffi::{OsStrExt, OsStringExt};
#[cfg(unix)]
use std::os::unix::fs::MetadataExt;

#[cfg(unix)]
use rustix::fs::{AtFlags, Dir, FileType as UnixFileType, Mode, OFlags, Stat};

use anyhow::{Context, Result, anyhow, bail, ensure};
use serde::{Deserialize, Serialize};
use unicode_normalization::UnicodeNormalization;
use uuid::Uuid;

pub const ARCHIVE_VERSION: u16 = 1;
const MANIFEST_PATH: &str = "pipefs-manifest.json";
const OBJECT_PREFIX: &str = "objects/";
const MAX_MANIFEST_BYTES: u64 = 64 * 1024 * 1024;
const MAX_DECOMPRESSED_BYTES: u64 = 4 * 1024 * 1024 * 1024;
const STABLE_SCAN_ATTEMPTS: usize = 4;

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RevisionKind {
    Full,
    Delta,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ArchiveEntryKind {
    File,
    Directory,
    Symlink,
    Tombstone,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ArchiveEntry {
    pub path: String,
    pub entry_type: ArchiveEntryKind,
    pub size: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub blake3: Option<String>,
    pub modified_unix: u64,
    pub mode: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub symlink_target: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub payload: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ArchiveManifest {
    pub archive_version: u16,
    pub revision_type: RevisionKind,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub base_manifest_digest: Option<String>,
    pub logical_size_bytes: u64,
    pub entries: Vec<ArchiveEntry>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct SnapshotEntry {
    pub entry_type: ArchiveEntryKind,
    pub size: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub blake3: Option<String>,
    pub modified_unix: u64,
    pub mode: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub symlink_target: Option<String>,
}

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct Snapshot {
    pub entries: BTreeMap<String, SnapshotEntry>,
    pub logical_size_bytes: u64,
    pub manifest_digest: Option<String>,
}

#[derive(Clone, Debug)]
pub struct ArchiveArtifact {
    pub bytes: Vec<u8>,
    pub blake3: String,
    pub manifest_digest: String,
    pub manifest: ArchiveManifest,
    pub snapshot: Snapshot,
}

/// Scan until two complete walks agree. Each regular file is also checked
/// before and after its contents are read. This avoids committing a hybrid of
/// two workspace states when a process is still writing.
pub fn scan_workspace(root: &Path) -> Result<Snapshot> {
    scan_stable(root)
}

pub fn build_revision(
    root: &Path,
    base: Option<&Snapshot>,
    force_full: bool,
) -> Result<ArchiveArtifact> {
    let scanned = scan_stable(root)?;
    let kind = if force_full || base.is_none() {
        RevisionKind::Full
    } else {
        RevisionKind::Delta
    };
    let mut entries = Vec::new();
    if kind == RevisionKind::Delta {
        let base = base.expect("delta has a base snapshot");
        for path in base.entries.keys() {
            if !scanned.entries.contains_key(path) {
                entries.push(ArchiveEntry {
                    path: path.clone(),
                    entry_type: ArchiveEntryKind::Tombstone,
                    size: 0,
                    blake3: None,
                    modified_unix: 0,
                    mode: 0,
                    symlink_target: None,
                    payload: None,
                });
            }
        }
    }

    for (path, entry) in &scanned.entries {
        let changed = kind == RevisionKind::Full
            || base.and_then(|base| base.entries.get(path)) != Some(entry);
        if !changed {
            continue;
        }
        entries.push(ArchiveEntry {
            path: path.clone(),
            entry_type: entry.entry_type,
            size: entry.size,
            blake3: entry.blake3.clone(),
            modified_unix: entry.modified_unix,
            mode: entry.mode,
            symlink_target: entry.symlink_target.clone(),
            payload: None,
        });
    }
    entries.sort_by(|left, right| left.path.cmp(&right.path));
    for (index, entry) in entries.iter_mut().enumerate() {
        if entry.entry_type == ArchiveEntryKind::File {
            entry.payload = Some(format!("{OBJECT_PREFIX}{index:08}"));
        }
    }

    let manifest = ArchiveManifest {
        archive_version: ARCHIVE_VERSION,
        revision_type: kind,
        base_manifest_digest: if kind == RevisionKind::Delta {
            base.and_then(|base| base.manifest_digest.clone())
        } else {
            None
        },
        logical_size_bytes: scanned.logical_size_bytes,
        entries,
    };
    validate_manifest(&manifest)?;
    let manifest_bytes = serde_json::to_vec(&manifest).context("serializing PipeFS manifest")?;
    let manifest_digest = blake3::hash(&manifest_bytes).to_hex().to_string();
    let bytes = encode_archive(root, &manifest, &manifest_bytes, &scanned)?;
    let archive_hash = blake3::hash(&bytes).to_hex().to_string();
    let mut snapshot = scanned;
    snapshot.manifest_digest = Some(manifest_digest.clone());
    Ok(ArchiveArtifact {
        bytes,
        blake3: archive_hash,
        manifest_digest,
        manifest,
        snapshot,
    })
}

fn scan_stable(root: &Path) -> Result<Snapshot> {
    ensure!(
        root.is_dir(),
        "PipeFS workspace is not a directory: {}",
        root.display()
    );
    let mut previous: Option<Snapshot> = None;
    for _ in 0..STABLE_SCAN_ATTEMPTS {
        match scan_once(root) {
            Ok(current) => {
                if previous.as_ref() == Some(&current) {
                    return Ok(current);
                }
                previous = Some(current);
            }
            Err(error) if transient_scan_error(&error) => {
                previous = None;
            }
            Err(error) => return Err(error),
        }
    }
    bail!("workspace_changed_during_scan: files did not reach a stable state")
}

fn transient_scan_error(error: &anyhow::Error) -> bool {
    error.to_string().contains("workspace_changed_during_scan")
        || error.chain().any(|cause| {
            cause
                .downcast_ref::<std::io::Error>()
                .is_some_and(|error| error.kind() == std::io::ErrorKind::NotFound)
        })
}

fn scan_once(root: &Path) -> Result<Snapshot> {
    let mut entries = BTreeMap::new();
    let mut portable_names = HashMap::<String, String>::new();
    #[cfg(unix)]
    scan_workspace_unix(root, &mut entries, &mut portable_names)?;
    #[cfg(not(unix))]
    scan_directory(root, root, &mut entries, &mut portable_names)?;
    let logical_size_bytes = entries
        .values()
        .filter(|entry| entry.entry_type == ArchiveEntryKind::File)
        .try_fold(0_u64, |total, entry| total.checked_add(entry.size))
        .ok_or_else(|| anyhow!("PipeFS workspace logical size overflow"))?;
    Ok(Snapshot {
        entries,
        logical_size_bytes,
        manifest_digest: None,
    })
}

#[cfg(not(unix))]
fn scan_directory(
    root: &Path,
    directory: &Path,
    entries: &mut BTreeMap<String, SnapshotEntry>,
    portable_names: &mut HashMap<String, String>,
) -> Result<()> {
    let mut children = fs::read_dir(directory)
        .with_context(|| format!("reading workspace directory {}", directory.display()))?
        .collect::<std::io::Result<Vec<_>>>()?;
    children.sort_by_key(|left| left.file_name());
    for child in children {
        let path = child.path();
        let relative = path
            .strip_prefix(root)
            .expect("walked path stays beneath root");
        let portable = portable_path(relative)?;
        register_portable_name(&portable, portable_names)?;
        let before = fs::symlink_metadata(&path)
            .with_context(|| format!("reading metadata for {}", path.display()))?;
        let file_type = before.file_type();
        if file_type.is_file() {
            let (digest, size) = hash_file(&path)
                .with_context(|| format!("reading workspace file {}", path.display()))?;
            let after = fs::symlink_metadata(&path)
                .with_context(|| format!("re-reading metadata for {}", path.display()))?;
            ensure!(
                stable_metadata(&before, &after) && size == after.len(),
                "workspace_changed_during_scan: {} changed while being read",
                portable
            );
            entries.insert(
                portable,
                snapshot_entry(&after, ArchiveEntryKind::File, Some(digest), None),
            );
        } else if file_type.is_dir() {
            entries.insert(
                portable,
                snapshot_entry(&before, ArchiveEntryKind::Directory, None, None),
            );
            scan_directory(root, &path, entries, portable_names)?;
        } else if file_type.is_symlink() {
            let target = fs::read_link(&path)
                .with_context(|| format!("reading symlink {}", path.display()))?;
            let target = target.to_str().ok_or_else(|| {
                anyhow!(
                    "path_portability: symlink target is not valid UTF-8: {}",
                    path.display()
                )
            })?;
            validate_symlink_target(&portable, target)?;
            entries.insert(
                portable,
                snapshot_entry(
                    &before,
                    ArchiveEntryKind::Symlink,
                    None,
                    Some(target.to_string()),
                ),
            );
        } else {
            bail!(
                "path_portability: special files, devices, sockets, and FIFOs are not supported: {}",
                portable
            );
        }
    }
    Ok(())
}

#[cfg(not(unix))]
fn hash_file(path: &Path) -> Result<(String, u64)> {
    let mut file = File::open(path)?;
    let mut hasher = blake3::Hasher::new();
    let size = std::io::copy(&mut file, &mut hasher)?;
    Ok((hasher.finalize().to_hex().to_string(), size))
}

#[cfg(unix)]
#[derive(Clone, Debug, Eq, PartialEq)]
struct UnixFileState {
    file_type: UnixFileType,
    device: u64,
    inode: u64,
    mode: u32,
    links: u64,
    size: u64,
    modified_seconds: i64,
    modified_nanoseconds: i64,
    changed_seconds: i64,
    changed_nanoseconds: i64,
}

#[cfg(unix)]
impl UnixFileState {
    fn from_stat(metadata: &Stat) -> Result<Self> {
        Ok(Self {
            file_type: UnixFileType::from_raw_mode(metadata.st_mode),
            device: metadata_number_u64(metadata.st_dev, "device number")?,
            inode: metadata_number_u64(metadata.st_ino, "inode number")?,
            mode: metadata_number_u32(metadata.st_mode, "mode")?,
            links: metadata_number_u64(metadata.st_nlink, "link count")?,
            size: metadata_number_u64(metadata.st_size, "file size")?,
            modified_seconds: metadata_number_i64(metadata.st_mtime, "modification time")?,
            modified_nanoseconds: metadata_number_i64(
                metadata.st_mtime_nsec,
                "modification time nanoseconds",
            )?,
            changed_seconds: metadata_number_i64(metadata.st_ctime, "change time")?,
            changed_nanoseconds: metadata_number_i64(
                metadata.st_ctime_nsec,
                "change time nanoseconds",
            )?,
        })
    }

    fn from_metadata(metadata: &Metadata) -> Self {
        let file_type = metadata.file_type();
        let file_type = if file_type.is_file() {
            UnixFileType::RegularFile
        } else if file_type.is_dir() {
            UnixFileType::Directory
        } else if file_type.is_symlink() {
            UnixFileType::Symlink
        } else {
            UnixFileType::Unknown
        };
        Self {
            file_type,
            device: metadata.dev(),
            inode: metadata.ino(),
            mode: metadata.mode(),
            links: metadata.nlink(),
            size: metadata.size(),
            modified_seconds: metadata.mtime(),
            modified_nanoseconds: metadata.mtime_nsec(),
            changed_seconds: metadata.ctime(),
            changed_nanoseconds: metadata.ctime_nsec(),
        }
    }

    fn file_type(&self) -> UnixFileType {
        self.file_type
    }
}

#[cfg(unix)]
fn metadata_number_u64<T: TryInto<u64>>(value: T, label: &str) -> Result<u64> {
    value
        .try_into()
        .map_err(|_| anyhow!("workspace metadata contains an invalid {label}"))
}

#[cfg(unix)]
fn metadata_number_u32<T: TryInto<u32>>(value: T, label: &str) -> Result<u32> {
    value
        .try_into()
        .map_err(|_| anyhow!("workspace metadata contains an invalid {label}"))
}

#[cfg(unix)]
fn metadata_number_i64<T: TryInto<i64>>(value: T, label: &str) -> Result<i64> {
    value
        .try_into()
        .map_err(|_| anyhow!("workspace metadata contains an invalid {label}"))
}

#[cfg(unix)]
fn scan_workspace_unix(
    root: &Path,
    entries: &mut BTreeMap<String, SnapshotEntry>,
    portable_names: &mut HashMap<String, String>,
) -> Result<()> {
    let directory = open_workspace_root(root)?;
    let before = UnixFileState::from_metadata(&directory.metadata()?);
    ensure!(
        before.file_type() == UnixFileType::Directory,
        "PipeFS workspace is not a real directory: {}",
        root.display()
    );
    scan_directory_unix(&directory, Path::new(""), entries, portable_names)?;
    let after = UnixFileState::from_metadata(&directory.metadata()?);
    ensure!(
        before == after,
        "workspace_changed_during_scan: workspace root changed while being read"
    );
    Ok(())
}

#[cfg(unix)]
fn scan_directory_unix(
    directory: &File,
    relative_directory: &Path,
    entries: &mut BTreeMap<String, SnapshotEntry>,
    portable_names: &mut HashMap<String, String>,
) -> Result<()> {
    let mut reader = Dir::read_from(directory)
        .map_err(std::io::Error::from)
        .context("opening workspace directory descriptor for enumeration")?;
    let mut names = Vec::new();
    while let Some(item) = reader.read() {
        let item = item
            .map_err(std::io::Error::from)
            .context("enumerating workspace directory descriptor")?;
        let bytes = item.file_name().to_bytes();
        if matches!(bytes, b"." | b"..") {
            continue;
        }
        names.push(OsString::from_vec(bytes.to_vec()));
    }
    names.sort_by(|left, right| left.as_bytes().cmp(right.as_bytes()));

    for name in names {
        let relative = relative_directory.join(&name);
        let portable = portable_path(&relative)?;
        register_portable_name(&portable, portable_names)?;
        let before = scan_stat_child(directory, &name, &portable)?;
        match before.file_type() {
            UnixFileType::RegularFile => {
                let file = open_scanned_child(
                    directory,
                    &name,
                    &before,
                    UnixFileType::RegularFile,
                    &portable,
                )?;
                let mut file = file;
                let mut hasher = blake3::Hasher::new();
                let size = std::io::copy(&mut file, &mut hasher).with_context(|| {
                    format!("reading workspace file through descriptor {portable}")
                })?;
                let descriptor_after = UnixFileState::from_metadata(&file.metadata()?);
                let path_after = scan_stat_child(directory, &name, &portable)?;
                ensure!(
                    before == descriptor_after && before == path_after && size == before.size,
                    "workspace_changed_during_scan: {portable} changed while being read"
                );
                entries.insert(
                    portable,
                    snapshot_entry_unix(
                        &before,
                        ArchiveEntryKind::File,
                        Some(hasher.finalize().to_hex().to_string()),
                        None,
                    ),
                );
            }
            UnixFileType::Directory => {
                let child = open_scanned_child(
                    directory,
                    &name,
                    &before,
                    UnixFileType::Directory,
                    &portable,
                )?;
                entries.insert(
                    portable.clone(),
                    snapshot_entry_unix(&before, ArchiveEntryKind::Directory, None, None),
                );
                scan_directory_unix(&child, &relative, entries, portable_names)?;
                let descriptor_after = UnixFileState::from_metadata(&child.metadata()?);
                let path_after = scan_stat_child(directory, &name, &portable)?;
                ensure!(
                    before == descriptor_after && before == path_after,
                    "workspace_changed_during_scan: {portable} changed while being read"
                );
            }
            UnixFileType::Symlink => {
                let target = scan_rustix_result(
                    rustix::fs::readlinkat(directory, &name, Vec::new()),
                    &portable,
                    "reading symlink",
                )?;
                let target = std::str::from_utf8(target.to_bytes()).map_err(|_| {
                    anyhow!("path_portability: symlink target is not valid UTF-8: {portable}")
                })?;
                let after = scan_stat_child(directory, &name, &portable)?;
                ensure!(
                    before == after,
                    "workspace_changed_during_scan: {portable} changed while being read"
                );
                validate_symlink_target(&portable, target)?;
                entries.insert(
                    portable,
                    snapshot_entry_unix(
                        &before,
                        ArchiveEntryKind::Symlink,
                        None,
                        Some(target.to_string()),
                    ),
                );
            }
            _ => {
                bail!(
                    "path_portability: special files, devices, sockets, and FIFOs are not supported: {portable}"
                );
            }
        }
    }
    Ok(())
}

#[cfg(unix)]
fn scan_stat_child(directory: &File, name: &OsStr, portable: &str) -> Result<UnixFileState> {
    let metadata = scan_rustix_result(
        rustix::fs::statat(directory, name, AtFlags::SYMLINK_NOFOLLOW),
        portable,
        "reading metadata",
    )?;
    UnixFileState::from_stat(&metadata)
}

#[cfg(unix)]
fn open_scanned_child(
    directory: &File,
    name: &OsStr,
    before: &UnixFileState,
    expected_type: UnixFileType,
    portable: &str,
) -> Result<File> {
    let mut flags = OFlags::RDONLY | OFlags::CLOEXEC | OFlags::NOFOLLOW | OFlags::NONBLOCK;
    if expected_type == UnixFileType::Directory {
        flags |= OFlags::DIRECTORY;
    }
    let descriptor = scan_rustix_result(
        rustix::fs::openat(directory, name, flags, Mode::empty()),
        portable,
        "opening entry without following symlinks",
    )?;
    let file = File::from(descriptor);
    let actual = UnixFileState::from_metadata(&file.metadata()?);
    ensure!(
        actual.file_type() == expected_type && actual == *before,
        "workspace_changed_during_scan: {portable} changed before it could be opened"
    );
    Ok(file)
}

#[cfg(unix)]
fn scan_rustix_result<T>(
    result: rustix::io::Result<T>,
    portable: &str,
    operation: &str,
) -> Result<T> {
    result.map_err(|error| {
        if matches!(
            error,
            rustix::io::Errno::NOENT | rustix::io::Errno::NOTDIR | rustix::io::Errno::LOOP
        ) {
            anyhow!(
                "workspace_changed_during_scan: {portable} changed while {operation}: {}",
                std::io::Error::from(error)
            )
        } else {
            anyhow!(std::io::Error::from(error)).context(format!("{operation} {portable}"))
        }
    })
}

#[cfg(unix)]
fn snapshot_entry_unix(
    metadata: &UnixFileState,
    entry_type: ArchiveEntryKind,
    blake3: Option<String>,
    symlink_target: Option<String>,
) -> SnapshotEntry {
    SnapshotEntry {
        entry_type,
        size: if entry_type == ArchiveEntryKind::File {
            metadata.size
        } else {
            0
        },
        blake3,
        modified_unix: u64::try_from(metadata.modified_seconds).unwrap_or(0),
        mode: metadata.mode & 0o777,
        symlink_target,
    }
}

#[cfg(unix)]
fn open_workspace_root(root: &Path) -> Result<File> {
    let descriptor = rustix::fs::open(
        root,
        OFlags::RDONLY | OFlags::DIRECTORY | OFlags::NOFOLLOW | OFlags::CLOEXEC | OFlags::NONBLOCK,
        Mode::empty(),
    )
    .map_err(std::io::Error::from)
    .with_context(|| {
        format!(
            "opening PipeFS workspace root without following symlinks: {}",
            root.display()
        )
    })?;
    Ok(File::from(descriptor))
}

#[cfg(not(unix))]
fn stable_metadata(before: &Metadata, after: &Metadata) -> bool {
    before.file_type() == after.file_type()
        && before.len() == after.len()
        && modified_unix(before) == modified_unix(after)
        && portable_mode(before) == portable_mode(after)
}

#[cfg(not(unix))]
fn snapshot_entry(
    metadata: &Metadata,
    entry_type: ArchiveEntryKind,
    blake3: Option<String>,
    symlink_target: Option<String>,
) -> SnapshotEntry {
    SnapshotEntry {
        entry_type,
        size: if entry_type == ArchiveEntryKind::File {
            metadata.len()
        } else {
            0
        },
        blake3,
        modified_unix: modified_unix(metadata),
        mode: portable_mode(metadata),
        symlink_target,
    }
}

#[cfg(not(unix))]
fn modified_unix(metadata: &Metadata) -> u64 {
    metadata
        .modified()
        .ok()
        .and_then(|time| time.duration_since(UNIX_EPOCH).ok())
        .map_or(0, |duration| duration.as_secs())
}

#[cfg(not(unix))]
fn portable_mode(metadata: &Metadata) -> u32 {
    match (metadata.is_dir(), metadata.permissions().readonly()) {
        (true, true) => 0o555,
        (true, false) => 0o755,
        (false, true) => 0o444,
        (false, false) => 0o666,
    }
}

fn encode_archive(
    root: &Path,
    manifest: &ArchiveManifest,
    manifest_bytes: &[u8],
    snapshot: &Snapshot,
) -> Result<Vec<u8>> {
    let workspace = ArchiveWorkspace::open(root)?;
    let mut encoder =
        zstd::stream::Encoder::new(Vec::new(), 9).context("creating PipeFS zstd encoder")?;
    encoder.include_checksum(true)?;
    encoder.include_contentsize(true)?;
    let mut builder = tar::Builder::new(encoder);
    builder.mode(tar::HeaderMode::Deterministic);
    append_regular(
        &mut builder,
        MANIFEST_PATH,
        manifest_bytes.len() as u64,
        Cursor::new(manifest_bytes),
        0o600,
    )?;
    for entry in &manifest.entries {
        let Some(payload) = entry.payload.as_deref() else {
            continue;
        };
        let expected = snapshot.entries.get(&entry.path).ok_or_else(|| {
            anyhow!(
                "archive payload disappeared after stable scan: {}",
                entry.path
            )
        })?;
        let OpenedArchiveFile { file, before } = workspace.open_file(&entry.path, expected)?;
        let mut reader = HashingReader::new(file);
        append_regular(&mut builder, payload, entry.size, &mut reader, 0o600)?;
        let (actual_digest, actual_size) = reader.digest_and_size();
        workspace.verify_file(&entry.path, &before, reader.inner())?;
        ensure!(
            actual_size == entry.size
                && actual_digest == entry.blake3.as_deref().unwrap_or_default(),
            "workspace_changed_during_scan: {} changed during archive encoding",
            entry.path
        );
    }
    workspace.verify_root()?;
    builder.finish().context("finishing PipeFS tar archive")?;
    let encoder = builder
        .into_inner()
        .context("recovering PipeFS zstd encoder")?;
    encoder.finish().context("finishing PipeFS zstd archive")
}

struct ArchiveWorkspace {
    root: PathBuf,
    #[cfg(unix)]
    directory: File,
    #[cfg(unix)]
    root_state: UnixFileState,
}

#[derive(Debug)]
struct OpenedArchiveFile {
    file: File,
    #[cfg(unix)]
    before: UnixFileState,
    #[cfg(not(unix))]
    before: Metadata,
}

#[cfg(unix)]
impl ArchiveWorkspace {
    fn open(root: &Path) -> Result<Self> {
        let directory = open_workspace_root(root)?;
        let root_state = UnixFileState::from_metadata(&directory.metadata()?);
        ensure!(
            root_state.file_type() == UnixFileType::Directory,
            "PipeFS workspace is not a real directory: {}",
            root.display()
        );
        Ok(Self {
            root: root.to_path_buf(),
            directory,
            root_state,
        })
    }

    fn open_file(&self, path: &str, expected: &SnapshotEntry) -> Result<OpenedArchiveFile> {
        let before = stat_relative_nofollow(&self.directory, path).with_context(|| {
            format!(
                "re-reading metadata for workspace file {}",
                self.root.join(path).display()
            )
        })?;
        ensure!(
            before.file_type() == UnixFileType::RegularFile
                && snapshot_entry_unix(
                    &before,
                    ArchiveEntryKind::File,
                    expected.blake3.clone(),
                    None,
                ) == *expected,
            "workspace_changed_during_scan: {path} changed before archive encoding"
        );
        let file = open_relative_regular(&self.directory, path).with_context(|| {
            format!(
                "opening workspace file without following symlinks: {}",
                self.root.join(path).display()
            )
        })?;
        let descriptor_state = UnixFileState::from_metadata(&file.metadata()?);
        ensure!(
            descriptor_state.file_type() == UnixFileType::RegularFile && descriptor_state == before,
            "workspace_changed_during_scan: {path} changed before archive encoding"
        );
        Ok(OpenedArchiveFile { file, before })
    }

    fn verify_file(&self, path: &str, before: &UnixFileState, file: &File) -> Result<()> {
        let descriptor_after = UnixFileState::from_metadata(&file.metadata()?);
        let path_after = stat_relative_nofollow(&self.directory, path).with_context(|| {
            format!(
                "re-reading workspace path after archive encoding: {}",
                self.root.join(path).display()
            )
        })?;
        ensure!(
            descriptor_after == *before && path_after == *before,
            "workspace_changed_during_scan: {path} changed during archive encoding"
        );

        self.verify_root()?;
        Ok(())
    }

    fn verify_root(&self) -> Result<()> {
        let descriptor_after = UnixFileState::from_metadata(&self.directory.metadata()?);
        let reopened = open_workspace_root(&self.root)?;
        let path_after = UnixFileState::from_metadata(&reopened.metadata()?);
        ensure!(
            descriptor_after == self.root_state && path_after == self.root_state,
            "workspace_changed_during_scan: workspace root changed during archive encoding"
        );
        Ok(())
    }
}

#[cfg(not(unix))]
impl ArchiveWorkspace {
    fn open(root: &Path) -> Result<Self> {
        Ok(Self {
            root: root.to_path_buf(),
        })
    }

    fn open_file(&self, path: &str, expected: &SnapshotEntry) -> Result<OpenedArchiveFile> {
        let path = self.root.join(path);
        let before = fs::symlink_metadata(&path)
            .with_context(|| format!("re-reading metadata for {}", path.display()))?;
        ensure!(
            before.is_file()
                && !before.file_type().is_symlink()
                && snapshot_entry(
                    &before,
                    ArchiveEntryKind::File,
                    expected.blake3.clone(),
                    None,
                ) == *expected,
            "workspace_changed_during_scan: {} changed before archive encoding",
            path.display()
        );
        let file = File::open(&path)
            .with_context(|| format!("opening workspace file {}", path.display()))?;
        Ok(OpenedArchiveFile { file, before })
    }

    fn verify_file(&self, path: &str, before: &Metadata, _file: &File) -> Result<()> {
        let path = self.root.join(path);
        let after = fs::symlink_metadata(&path)
            .with_context(|| format!("re-reading metadata for {}", path.display()))?;
        ensure!(
            stable_metadata(before, &after),
            "workspace_changed_during_scan: {} changed during archive encoding",
            path.display()
        );
        Ok(())
    }

    fn verify_root(&self) -> Result<()> {
        Ok(())
    }
}

#[cfg(unix)]
fn open_relative_regular(root: &File, path: &str) -> Result<File> {
    validate_archive_path(path)?;
    let mut current = root.try_clone()?;
    let mut components = Path::new(path).components().peekable();
    while let Some(component) = components.next() {
        let Component::Normal(name) = component else {
            bail!("path_portability: invalid archive path component in {path:?}")
        };
        let final_component = components.peek().is_none();
        let mut flags = OFlags::RDONLY | OFlags::CLOEXEC | OFlags::NOFOLLOW | OFlags::NONBLOCK;
        if !final_component {
            flags |= OFlags::DIRECTORY;
        }
        let descriptor = rustix::fs::openat(&current, name, flags, Mode::empty())
            .map_err(std::io::Error::from)?;
        let opened = File::from(descriptor);
        let state = UnixFileState::from_metadata(&opened.metadata()?);
        let expected_type = if final_component {
            UnixFileType::RegularFile
        } else {
            UnixFileType::Directory
        };
        ensure!(
            state.file_type() == expected_type,
            "workspace_changed_during_scan: {path} contains a symlink or non-directory ancestor"
        );
        current = opened;
    }
    Ok(current)
}

#[cfg(unix)]
fn stat_relative_nofollow(root: &File, path: &str) -> Result<UnixFileState> {
    validate_archive_path(path)?;
    let mut current = root.try_clone()?;
    let mut components = Path::new(path).components().peekable();
    while let Some(component) = components.next() {
        let Component::Normal(name) = component else {
            bail!("path_portability: invalid archive path component in {path:?}")
        };
        if components.peek().is_none() {
            let metadata = rustix::fs::statat(&current, name, AtFlags::SYMLINK_NOFOLLOW)
                .map_err(std::io::Error::from)?;
            return UnixFileState::from_stat(&metadata);
        }
        let descriptor = rustix::fs::openat(
            &current,
            name,
            OFlags::RDONLY
                | OFlags::DIRECTORY
                | OFlags::NOFOLLOW
                | OFlags::CLOEXEC
                | OFlags::NONBLOCK,
            Mode::empty(),
        )
        .map_err(std::io::Error::from)?;
        current = File::from(descriptor);
    }
    bail!("path_portability: empty archive path")
}

fn append_regular<W: Write, R: Read>(
    builder: &mut tar::Builder<W>,
    path: &str,
    size: u64,
    reader: R,
    mode: u32,
) -> Result<()> {
    let mut header = tar::Header::new_gnu();
    header.set_entry_type(tar::EntryType::Regular);
    header.set_mode(mode);
    header.set_uid(0);
    header.set_gid(0);
    header.set_mtime(0);
    header.set_size(size);
    header.set_cksum();
    builder
        .append_data(&mut header, path, reader)
        .with_context(|| format!("adding {path} to PipeFS archive"))
}

struct HashingReader<R> {
    inner: R,
    hasher: blake3::Hasher,
    size: u64,
}

impl<R> HashingReader<R> {
    fn new(inner: R) -> Self {
        Self {
            inner,
            hasher: blake3::Hasher::new(),
            size: 0,
        }
    }

    fn finish(self) -> (String, u64) {
        (self.hasher.finalize().to_hex().to_string(), self.size)
    }

    fn digest_and_size(&self) -> (String, u64) {
        (
            self.hasher.clone().finalize().to_hex().to_string(),
            self.size,
        )
    }

    fn inner(&self) -> &R {
        &self.inner
    }
}

impl<R: Read> Read for HashingReader<R> {
    fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
        let read = self.inner.read(buffer)?;
        self.hasher.update(&buffer[..read]);
        self.size = self.size.saturating_add(read as u64);
        Ok(read)
    }
}

/// Apply one verified archive to an inactive staging tree. Delta application
/// requires the exact snapshot produced by the preceding revision so the
/// manifest's base digest and the on-disk tree can both be verified before
/// any bytes are changed.
pub fn apply_archive(
    root: &Path,
    bytes: &[u8],
    expected_base: Option<&Snapshot>,
) -> Result<Snapshot> {
    fs::create_dir_all(root)
        .with_context(|| format!("creating PipeFS restore root {}", root.display()))?;
    ensure_root_is_not_symlink(root)?;
    let decoder = zstd::stream::read::Decoder::new(Cursor::new(bytes))
        .context("corrupt_archive: invalid zstd stream")?;
    let bounded = BoundedReader::new(decoder, MAX_DECOMPRESSED_BYTES);
    let mut archive = tar::Archive::new(bounded);
    let mut archive_entries = archive.entries().context("reading PipeFS tar entries")?;
    let mut manifest_item = archive_entries
        .next()
        .ok_or_else(|| anyhow!("corrupt_archive: missing manifest"))?
        .context("reading PipeFS manifest tar entry")?;
    ensure_regular_tar_entry(&manifest_item)?;
    ensure!(
        tar_entry_path(&manifest_item)? == MANIFEST_PATH,
        "corrupt_archive: manifest must be the first tar record"
    );
    ensure!(
        manifest_item.size() <= MAX_MANIFEST_BYTES,
        "corrupt_archive: manifest is too large"
    );
    let mut manifest_bytes = Vec::with_capacity(manifest_item.size() as usize);
    manifest_item.read_to_end(&mut manifest_bytes)?;
    drop(manifest_item);
    let manifest: ArchiveManifest = serde_json::from_slice(&manifest_bytes)
        .context("corrupt_archive: invalid manifest JSON")?;
    ensure!(
        serde_json::to_vec(&manifest)? == manifest_bytes,
        "corrupt_archive: manifest JSON is not canonical"
    );
    validate_manifest(&manifest)?;

    let baseline = match manifest.revision_type {
        RevisionKind::Full => {
            ensure!(
                expected_base.is_none(),
                "corrupt_archive: full revision was given an expected base"
            );
            ensure!(
                fs::read_dir(root)?.next().is_none(),
                "full PipeFS revision must be restored into an empty staging directory"
            );
            Snapshot::default()
        }
        RevisionKind::Delta => {
            let expected_base = expected_base.ok_or_else(|| {
                anyhow!("corrupt_archive: delta revision requires an expected base snapshot")
            })?;
            ensure!(
                expected_base.manifest_digest.as_deref()
                    == manifest.base_manifest_digest.as_deref(),
                "corruption_error: delta base manifest digest does not match the restored head"
            );
            let actual = scan_workspace(root)
                .context("verifying the on-disk PipeFS delta base before extraction")?;
            ensure!(
                actual.entries == expected_base.entries
                    && actual.logical_size_bytes == expected_base.logical_size_bytes,
                "corruption_error: on-disk workspace does not match the expected delta base"
            );
            expected_base.clone()
        }
    };
    let mut expected = expected_snapshot(&manifest, &baseline)?;
    make_directories_writable(root, &baseline)?;
    let application = (|| -> Result<()> {
        apply_manifest_structure(root, &manifest)?;
        for entry in manifest
            .entries
            .iter()
            .filter(|entry| entry.entry_type == ArchiveEntryKind::File)
        {
            let mut item = archive_entries
                .next()
                .ok_or_else(|| anyhow!("corrupt_archive: missing payload for {}", entry.path))?
                .context("reading PipeFS payload tar entry")?;
            ensure_regular_tar_entry(&item)?;
            let payload_path = tar_entry_path(&item)?;
            ensure!(
                Some(payload_path.as_str()) == entry.payload.as_deref(),
                "corrupt_archive: payload order or path does not match manifest"
            );
            ensure!(
                item.size() == entry.size,
                "corrupt_archive: size mismatch for {}",
                entry.path
            );
            apply_file_payload(root, entry, &mut item)?;
        }
        ensure!(
            archive_entries.next().is_none(),
            "corrupt_archive: unexpected or duplicate tar payload"
        );
        apply_manifest_symlinks(root, &manifest)?;
        Ok(())
    })();
    if let Err(error) = application {
        let _ = restore_directory_metadata(root, &baseline);
        return Err(error);
    }
    let mut remainder = archive.into_inner();
    if let Err(error) = std::io::copy(&mut remainder, &mut std::io::sink()) {
        let _ = restore_directory_metadata(root, &baseline);
        return Err(anyhow!(error).context("corrupt_archive: decompressing revision"));
    }
    restore_directory_metadata(root, &expected)?;
    let mut snapshot = scan_workspace(root)?;
    let manifest_digest = blake3::hash(&manifest_bytes).to_hex().to_string();
    expected.manifest_digest = Some(manifest_digest.clone());
    snapshot.manifest_digest = Some(manifest_digest);
    ensure!(
        snapshot == expected,
        "corruption_error: restored workspace does not exactly match the revision manifest"
    );
    Ok(snapshot)
}

fn ensure_regular_tar_entry<R: Read>(entry: &tar::Entry<'_, R>) -> Result<()> {
    ensure!(
        entry.header().entry_type().is_file(),
        "corrupt_archive: only regular tar records are allowed"
    );
    Ok(())
}

fn tar_entry_path<R: Read>(entry: &tar::Entry<'_, R>) -> Result<String> {
    entry
        .path()
        .context("reading PipeFS tar path")?
        .to_str()
        .ok_or_else(|| anyhow!("path_portability: archive path is not UTF-8"))
        .map(ToOwned::to_owned)
}

struct BoundedReader<R> {
    inner: R,
    maximum: u64,
    consumed: u64,
}

impl<R> BoundedReader<R> {
    fn new(inner: R, maximum: u64) -> Self {
        Self {
            inner,
            maximum,
            consumed: 0,
        }
    }
}

impl<R: Read> Read for BoundedReader<R> {
    fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
        if buffer.is_empty() {
            return Ok(0);
        }
        if self.consumed == self.maximum {
            let mut probe = [0_u8; 1];
            return match self.inner.read(&mut probe)? {
                0 => Ok(0),
                _ => Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "corrupt_archive: decompressed revision is too large",
                )),
            };
        }
        let available = self.maximum.saturating_sub(self.consumed);
        let limit = usize::try_from(available.min(buffer.len() as u64)).unwrap_or(buffer.len());
        let read = self.inner.read(&mut buffer[..limit])?;
        self.consumed = self.consumed.saturating_add(read as u64);
        Ok(read)
    }
}

fn expected_snapshot(manifest: &ArchiveManifest, baseline: &Snapshot) -> Result<Snapshot> {
    let mut entries = baseline.entries.clone();
    for entry in &manifest.entries {
        let removes_descendants = match entry.entry_type {
            ArchiveEntryKind::Tombstone | ArchiveEntryKind::File | ArchiveEntryKind::Symlink => {
                true
            }
            ArchiveEntryKind::Directory => entries
                .get(&entry.path)
                .is_some_and(|existing| existing.entry_type != ArchiveEntryKind::Directory),
        };
        if removes_descendants {
            let prefix = format!("{}/", entry.path);
            entries.retain(|path, _| path != &entry.path && !path.starts_with(&prefix));
        }
        if entry.entry_type != ArchiveEntryKind::Tombstone {
            entries.insert(
                entry.path.clone(),
                SnapshotEntry {
                    entry_type: entry.entry_type,
                    size: entry.size,
                    blake3: entry.blake3.clone(),
                    modified_unix: entry.modified_unix,
                    mode: entry.mode,
                    symlink_target: entry.symlink_target.clone(),
                },
            );
        }
    }
    validate_snapshot_structure(&entries)?;
    let logical_size_bytes = entries
        .values()
        .filter(|entry| entry.entry_type == ArchiveEntryKind::File)
        .try_fold(0_u64, |total, entry| total.checked_add(entry.size))
        .ok_or_else(|| anyhow!("corrupt_archive: resulting logical size overflow"))?;
    ensure!(
        logical_size_bytes == manifest.logical_size_bytes,
        "corrupt_archive: resulting file sizes do not equal logical workspace size"
    );
    Ok(Snapshot {
        entries,
        logical_size_bytes,
        manifest_digest: None,
    })
}

fn validate_snapshot_structure(entries: &BTreeMap<String, SnapshotEntry>) -> Result<()> {
    for path in entries.keys() {
        let mut parent = Path::new(path).parent();
        while let Some(value) = parent.filter(|value| !value.as_os_str().is_empty()) {
            let portable = value
                .to_str()
                .ok_or_else(|| anyhow!("path_portability: snapshot parent is not UTF-8"))?;
            ensure!(
                entries
                    .get(portable)
                    .is_some_and(|entry| { entry.entry_type == ArchiveEntryKind::Directory }),
                "corrupt_archive: {path:?} has a missing or non-directory parent {portable:?}"
            );
            parent = value.parent();
        }
    }
    Ok(())
}

fn apply_manifest_structure(root: &Path, manifest: &ArchiveManifest) -> Result<()> {
    let mut tombstones = manifest
        .entries
        .iter()
        .filter(|entry| entry.entry_type == ArchiveEntryKind::Tombstone)
        .collect::<Vec<_>>();
    tombstones.sort_by_key(|entry| std::cmp::Reverse(path_depth(&entry.path)));
    for entry in tombstones {
        remove_existing(root, &entry.path)?;
    }

    for entry in manifest
        .entries
        .iter()
        .filter(|entry| entry.entry_type == ArchiveEntryKind::Directory)
    {
        let destination = checked_destination(root, &entry.path)?;
        ensure_safe_parent(root, &destination)?;
        match fs::symlink_metadata(&destination) {
            Ok(metadata) if metadata.is_dir() && !metadata.file_type().is_symlink() => {}
            Ok(metadata) => remove_path(&destination, &metadata)?,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(error.into()),
        }
        fs::create_dir_all(&destination)?;
    }

    Ok(())
}

fn apply_manifest_symlinks(root: &Path, manifest: &ArchiveManifest) -> Result<()> {
    for entry in manifest
        .entries
        .iter()
        .filter(|entry| entry.entry_type == ArchiveEntryKind::Symlink)
    {
        let destination = checked_destination(root, &entry.path)?;
        ensure_safe_parent(root, &destination)?;
        if let Ok(metadata) = fs::symlink_metadata(&destination) {
            remove_path(&destination, &metadata)?;
        }
        create_symlink(
            entry
                .symlink_target
                .as_deref()
                .expect("validated symlink target"),
            &destination,
        )?;
        set_symlink_mtime(&destination, entry.modified_unix)?;
    }
    Ok(())
}

fn apply_file_payload<R: Read>(root: &Path, entry: &ArchiveEntry, payload: R) -> Result<()> {
    let destination = checked_destination(root, &entry.path)?;
    ensure_safe_parent(root, &destination)?;
    if let Ok(metadata) = fs::symlink_metadata(&destination) {
        remove_path(&destination, &metadata)?;
    }
    if let Some(parent) = destination.parent() {
        fs::create_dir_all(parent)?;
        ensure_safe_parent(root, &destination)?;
    }
    write_restored_file(
        &destination,
        payload,
        entry.size,
        entry.blake3.as_deref().expect("validated file digest"),
        entry.mode,
        entry.modified_unix,
    )
}

fn write_restored_file<R: Read>(
    destination: &Path,
    payload: R,
    expected_size: u64,
    expected_digest: &str,
    mode: u32,
    modified_unix: u64,
) -> Result<()> {
    let parent = destination
        .parent()
        .ok_or_else(|| anyhow!("restore destination has no parent"))?;
    let mut temporary = None;
    for _ in 0..32 {
        let candidate = parent.join(format!(".pipefs-restore-{}", Uuid::new_v4().simple()));
        match OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&candidate)
        {
            Ok(file) => {
                temporary = Some((candidate, file));
                break;
            }
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
            Err(error) => return Err(error.into()),
        }
    }
    let (temporary, mut file) = temporary
        .ok_or_else(|| anyhow!("could not allocate a collision-free restore temporary file"))?;
    let mut payload = HashingReader::new(payload);
    let result = (|| -> Result<()> {
        std::io::copy(&mut payload, &mut file)?;
        let (actual_digest, actual_size) = payload.finish();
        ensure!(
            actual_size == expected_size,
            "corrupt_archive: size mismatch while restoring {}",
            destination.display()
        );
        ensure!(
            actual_digest == expected_digest,
            "corrupt_archive: digest mismatch while restoring {}",
            destination.display()
        );
        file.sync_all()?;
        drop(file);
        set_mode(&temporary, mode)?;
        set_mtime(&temporary, modified_unix)?;
        fs::rename(&temporary, destination)?;
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    result
}

fn make_directories_writable(root: &Path, snapshot: &Snapshot) -> Result<()> {
    let mut directories = snapshot
        .entries
        .iter()
        .filter(|(_, entry)| entry.entry_type == ArchiveEntryKind::Directory)
        .collect::<Vec<_>>();
    directories.sort_by_key(|(path, _)| path_depth(path));
    for (path, entry) in directories {
        let destination = checked_destination(root, path)?;
        set_mode(&destination, entry.mode | 0o700).with_context(|| {
            format!(
                "temporarily making PipeFS restore directory writable: {}",
                destination.display()
            )
        })?;
    }
    Ok(())
}

fn restore_directory_metadata(root: &Path, snapshot: &Snapshot) -> Result<()> {
    let mut directories = snapshot
        .entries
        .iter()
        .filter(|(_, entry)| entry.entry_type == ArchiveEntryKind::Directory)
        .collect::<Vec<_>>();
    directories.sort_by_key(|(path, _)| std::cmp::Reverse(path_depth(path)));
    for (path, entry) in directories {
        let destination = checked_destination(root, path)?;
        let metadata = fs::symlink_metadata(&destination)?;
        ensure!(
            metadata.is_dir() && !metadata.file_type().is_symlink(),
            "corruption_error: expected restored directory {}",
            destination.display()
        );
        set_mode(&destination, entry.mode)?;
        set_mtime(&destination, entry.modified_unix)?;
    }
    Ok(())
}

fn validate_manifest(manifest: &ArchiveManifest) -> Result<()> {
    ensure!(
        manifest.archive_version == ARCHIVE_VERSION,
        "unsupported_archive_version: {}",
        manifest.archive_version
    );
    if manifest.revision_type == RevisionKind::Full {
        ensure!(
            manifest.base_manifest_digest.is_none(),
            "corrupt_archive: full revision has a base digest"
        );
    } else {
        let digest = manifest
            .base_manifest_digest
            .as_deref()
            .ok_or_else(|| anyhow!("corrupt_archive: delta revision has no base digest"))?;
        validate_digest(digest, "base_manifest_digest")?;
    }
    let mut previous: Option<&str> = None;
    let mut names = HashMap::new();
    let mut payloads = HashMap::new();
    let mut file_bytes = 0_u64;
    for entry in &manifest.entries {
        validate_archive_path(&entry.path)?;
        register_portable_name(&entry.path, &mut names)?;
        if let Some(previous) = previous {
            ensure!(
                previous < entry.path.as_str(),
                "corrupt_archive: manifest entries are not strictly sorted"
            );
        }
        previous = Some(&entry.path);
        match entry.entry_type {
            ArchiveEntryKind::File => {
                file_bytes = file_bytes
                    .checked_add(entry.size)
                    .ok_or_else(|| anyhow!("corrupt_archive: manifest file sizes overflow"))?;
                let digest = entry
                    .blake3
                    .as_deref()
                    .ok_or_else(|| anyhow!("corrupt_archive: file has no digest"))?;
                validate_digest(digest, "entry.blake3")?;
                let payload = entry
                    .payload
                    .as_deref()
                    .ok_or_else(|| anyhow!("corrupt_archive: file has no payload"))?;
                ensure!(
                    valid_payload_path(payload),
                    "corrupt_archive: invalid payload path"
                );
                ensure!(
                    payloads.insert(payload, &entry.path).is_none(),
                    "corrupt_archive: duplicate payload reference"
                );
                ensure!(
                    entry.symlink_target.is_none(),
                    "corrupt_archive: file has a symlink target"
                );
                ensure!(entry.mode <= 0o777, "corrupt_archive: invalid file mode");
            }
            ArchiveEntryKind::Directory => {
                ensure!(
                    entry.size == 0
                        && entry.blake3.is_none()
                        && entry.payload.is_none()
                        && entry.symlink_target.is_none(),
                    "corrupt_archive: invalid directory metadata"
                );
                ensure!(
                    entry.mode <= 0o777,
                    "corrupt_archive: invalid directory mode"
                );
            }
            ArchiveEntryKind::Symlink => {
                let target = entry
                    .symlink_target
                    .as_deref()
                    .ok_or_else(|| anyhow!("corrupt_archive: symlink has no target"))?;
                validate_symlink_target(&entry.path, target)?;
                ensure!(
                    entry.size == 0 && entry.blake3.is_none() && entry.payload.is_none(),
                    "corrupt_archive: invalid symlink metadata"
                );
                ensure!(entry.mode <= 0o777, "corrupt_archive: invalid symlink mode");
            }
            ArchiveEntryKind::Tombstone => {
                ensure!(
                    manifest.revision_type == RevisionKind::Delta,
                    "corrupt_archive: full revision contains a tombstone"
                );
                ensure!(
                    entry.size == 0
                        && entry.blake3.is_none()
                        && entry.payload.is_none()
                        && entry.symlink_target.is_none()
                        && entry.modified_unix == 0
                        && entry.mode == 0,
                    "corrupt_archive: invalid tombstone metadata"
                );
            }
        }
    }
    if manifest.revision_type == RevisionKind::Full {
        ensure!(
            file_bytes == manifest.logical_size_bytes,
            "corrupt_archive: full revision file sizes do not equal logical size"
        );
    } else {
        ensure!(
            file_bytes <= manifest.logical_size_bytes,
            "corrupt_archive: delta payload exceeds logical workspace size"
        );
    }
    Ok(())
}

fn validate_digest(value: &str, label: &str) -> Result<()> {
    ensure!(
        value.len() == 64
            && value
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase()),
        "corrupt_archive: {label} is not a lowercase BLAKE3 digest"
    );
    Ok(())
}

fn portable_path(path: &Path) -> Result<String> {
    let value = path.to_str().ok_or_else(|| {
        anyhow!(
            "path_portability: filename is not valid UTF-8: {}",
            path.display()
        )
    })?;
    validate_archive_path(value)?;
    Ok(value.to_string())
}

fn validate_archive_path(path: &str) -> Result<()> {
    ensure!(
        !path.is_empty() && path.len() <= 4096,
        "path_portability: invalid path length"
    );
    ensure!(
        !path.starts_with('/') && !path.contains('\\') && !path.contains('\0'),
        "path_portability: absolute, backslash, and NUL paths are forbidden: {path:?}"
    );
    for component in path.split('/') {
        validate_component(component)?;
    }
    Ok(())
}

fn validate_component(component: &str) -> Result<()> {
    ensure!(
        !component.is_empty() && !matches!(component, "." | ".."),
        "path_portability: empty, '.' and '..' components are forbidden"
    );
    ensure!(
        component.len() <= 255,
        "path_portability: filename component exceeds 255 bytes"
    );
    ensure!(
        !component.ends_with([' ', '.']),
        "path_portability: filename cannot end in a space or dot: {component:?}"
    );
    ensure!(
        !component.chars().any(|character| character.is_control()
            || matches!(character, '<' | '>' | ':' | '"' | '|' | '?' | '*')),
        "path_portability: filename contains characters unsafe on a restoring platform: {component:?}"
    );
    let stem = component
        .split('.')
        .next()
        .unwrap_or(component)
        .to_ascii_lowercase();
    let reserved = matches!(stem.as_str(), "con" | "prn" | "aux" | "nul" | "clock$")
        || (stem.len() == 4
            && (stem.starts_with("com") || stem.starts_with("lpt"))
            && matches!(stem.as_bytes()[3], b'1'..=b'9'));
    ensure!(
        !reserved,
        "path_portability: reserved filename {component:?}"
    );
    Ok(())
}

fn register_portable_name(path: &str, names: &mut HashMap<String, String>) -> Result<()> {
    let key = path
        .split('/')
        .map(|component| component.nfc().collect::<String>().to_lowercase())
        .collect::<Vec<_>>()
        .join("/");
    if let Some(existing) = names.insert(key, path.to_string()) {
        bail!(
            "path_portability: {existing:?} and {path:?} collide on a case-insensitive or normalization-insensitive filesystem"
        );
    }
    Ok(())
}

fn validate_symlink_target(link_path: &str, target: &str) -> Result<()> {
    ensure!(
        !target.is_empty() && !target.contains('\0') && !target.contains('\\'),
        "path_portability: invalid symlink target for {link_path:?}"
    );
    let target_path = Path::new(target);
    ensure!(
        !target_path.is_absolute(),
        "path_portability: absolute symlink target for {link_path:?}"
    );
    let mut depth = path_depth(link_path).saturating_sub(1) as i64;
    for component in target_path.components() {
        match component {
            Component::CurDir => {}
            Component::Normal(value) => {
                let value = value.to_str().ok_or_else(|| {
                    anyhow!(
                        "path_portability: symlink target component is not UTF-8 for {link_path:?}"
                    )
                })?;
                validate_component(value).with_context(|| {
                    format!("path_portability: unsafe symlink target component for {link_path:?}")
                })?;
                depth += 1;
            }
            Component::ParentDir => {
                depth -= 1;
                ensure!(
                    depth >= 0,
                    "path_portability: symlink {link_path:?} escapes the workspace"
                );
            }
            Component::RootDir | Component::Prefix(_) => {
                bail!("path_portability: symlink {link_path:?} escapes the workspace")
            }
        }
    }
    Ok(())
}

fn valid_payload_path(path: &str) -> bool {
    path.strip_prefix(OBJECT_PREFIX)
        .is_some_and(|suffix| suffix.len() == 8 && suffix.bytes().all(|byte| byte.is_ascii_digit()))
}

fn checked_destination(root: &Path, relative: &str) -> Result<PathBuf> {
    validate_archive_path(relative)?;
    Ok(root.join(relative))
}

fn ensure_root_is_not_symlink(root: &Path) -> Result<()> {
    let metadata = fs::symlink_metadata(root)?;
    ensure!(
        metadata.is_dir() && !metadata.file_type().is_symlink(),
        "restore root is not a real directory"
    );
    Ok(())
}

fn ensure_safe_parent(root: &Path, destination: &Path) -> Result<()> {
    let relative = destination
        .strip_prefix(root)
        .map_err(|_| anyhow!("extraction_escape: destination escaped restore root"))?;
    let mut current = root.to_path_buf();
    let components = relative.components().collect::<Vec<_>>();
    for component in components.iter().take(components.len().saturating_sub(1)) {
        let Component::Normal(component) = component else {
            bail!("extraction_escape: invalid destination component")
        };
        current.push(component);
        match fs::symlink_metadata(&current) {
            Ok(metadata) => ensure!(
                metadata.is_dir() && !metadata.file_type().is_symlink(),
                "extraction_escape: parent is not a real directory: {}",
                current.display()
            ),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => fs::create_dir(&current)?,
            Err(error) => return Err(error.into()),
        }
    }
    Ok(())
}

fn remove_existing(root: &Path, relative: &str) -> Result<()> {
    let destination = checked_destination(root, relative)?;
    ensure_safe_parent(root, &destination)?;
    match fs::symlink_metadata(&destination) {
        Ok(metadata) => remove_path(&destination, &metadata),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error.into()),
    }
}

fn remove_path(path: &Path, metadata: &Metadata) -> Result<()> {
    if metadata.is_dir() && !metadata.file_type().is_symlink() {
        fs::remove_dir_all(path)?;
    } else {
        fs::remove_file(path)?;
    }
    Ok(())
}

fn path_depth(path: &str) -> usize {
    path.split('/').count()
}

#[cfg(unix)]
fn set_mode(path: &Path, mode: u32) -> Result<()> {
    use std::os::unix::fs::PermissionsExt;
    fs::set_permissions(path, fs::Permissions::from_mode(mode & 0o777))?;
    Ok(())
}

#[cfg(not(unix))]
fn set_mode(path: &Path, mode: u32) -> Result<()> {
    let mut permissions = fs::metadata(path)?.permissions();
    permissions.set_readonly(mode & 0o222 == 0);
    fs::set_permissions(path, permissions)?;
    Ok(())
}

fn set_mtime(path: &Path, modified_unix: u64) -> Result<()> {
    let seconds = i64::try_from(modified_unix).unwrap_or(i64::MAX);
    filetime::set_file_mtime(path, filetime::FileTime::from_unix_time(seconds, 0))?;
    Ok(())
}

fn set_symlink_mtime(path: &Path, modified_unix: u64) -> Result<()> {
    let seconds = i64::try_from(modified_unix).unwrap_or(i64::MAX);
    let time = filetime::FileTime::from_unix_time(seconds, 0);
    filetime::set_symlink_file_times(path, time, time)?;
    Ok(())
}

#[cfg(unix)]
fn create_symlink(target: &str, destination: &Path) -> Result<()> {
    std::os::unix::fs::symlink(target, destination)
        .with_context(|| format!("restoring symlink {}", destination.display()))
}

#[cfg(windows)]
fn create_symlink(target: &str, destination: &Path) -> Result<()> {
    std::os::windows::fs::symlink_file(target, destination).with_context(|| {
        format!(
            "restoring symlink {} (Windows developer mode or symlink privilege is required)",
            destination.display()
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn encode_test_archive(manifest: &ArchiveManifest, payloads: &[(&str, &[u8])]) -> Vec<u8> {
        let manifest_bytes = serde_json::to_vec(manifest).unwrap();
        let mut encoder = zstd::stream::Encoder::new(Vec::new(), 9).unwrap();
        encoder.include_checksum(true).unwrap();
        encoder.include_contentsize(true).unwrap();
        let mut builder = tar::Builder::new(encoder);
        builder.mode(tar::HeaderMode::Deterministic);
        append_regular(
            &mut builder,
            MANIFEST_PATH,
            manifest_bytes.len() as u64,
            Cursor::new(&manifest_bytes),
            0o600,
        )
        .unwrap();
        for (path, bytes) in payloads {
            append_regular(
                &mut builder,
                path,
                bytes.len() as u64,
                Cursor::new(*bytes),
                0o600,
            )
            .unwrap();
        }
        builder.finish().unwrap();
        let encoder = builder.into_inner().unwrap();
        encoder.finish().unwrap()
    }

    #[test]
    fn full_archive_is_deterministic_and_round_trips_git_and_empty_directories() {
        let source = tempfile::tempdir().unwrap();
        fs::create_dir_all(source.path().join("empty/nested")).unwrap();
        fs::create_dir(source.path().join(".git")).unwrap();
        fs::write(source.path().join(".git/HEAD"), "ref: refs/heads/main\n").unwrap();
        fs::write(source.path().join("hello.txt"), "hello").unwrap();
        let first = build_revision(source.path(), None, false).unwrap();
        let second = build_revision(source.path(), None, false).unwrap();
        assert_eq!(first.bytes, second.bytes);
        assert_eq!(first.manifest_digest, second.manifest_digest);

        let restored = tempfile::tempdir().unwrap();
        let snapshot = apply_archive(restored.path(), &first.bytes, None).unwrap();
        assert_eq!(
            fs::read_to_string(restored.path().join("hello.txt")).unwrap(),
            "hello"
        );
        assert_eq!(
            fs::read_to_string(restored.path().join(".git/HEAD")).unwrap(),
            "ref: refs/heads/main\n"
        );
        assert!(restored.path().join("empty/nested").is_dir());
        assert_eq!(snapshot.entries, first.snapshot.entries);
    }

    #[test]
    fn delta_round_trip_handles_append_truncate_rename_and_recursive_delete() {
        let source = tempfile::tempdir().unwrap();
        fs::create_dir(source.path().join("gone")).unwrap();
        fs::write(source.path().join("gone/a"), "delete").unwrap();
        fs::write(source.path().join("append"), "a").unwrap();
        fs::write(source.path().join("truncate"), "long value").unwrap();
        fs::write(source.path().join("rename-old"), "move").unwrap();
        let full = build_revision(source.path(), None, false).unwrap();

        fs::remove_dir_all(source.path().join("gone")).unwrap();
        fs::write(source.path().join("append"), "ab").unwrap();
        fs::write(source.path().join("truncate"), "x").unwrap();
        fs::rename(
            source.path().join("rename-old"),
            source.path().join("rename-new"),
        )
        .unwrap();
        let delta = build_revision(source.path(), Some(&full.snapshot), false).unwrap();
        assert_eq!(delta.manifest.revision_type, RevisionKind::Delta);
        assert!(
            delta
                .manifest
                .entries
                .iter()
                .any(|entry| entry.entry_type == ArchiveEntryKind::Tombstone)
        );

        let restored = tempfile::tempdir().unwrap();
        let restored_base = apply_archive(restored.path(), &full.bytes, None).unwrap();
        let restored_snapshot =
            apply_archive(restored.path(), &delta.bytes, Some(&restored_base)).unwrap();
        assert_eq!(restored_snapshot.entries, delta.snapshot.entries);
        assert!(!restored.path().join("gone").exists());
        assert!(!restored.path().join("rename-old").exists());
        assert_eq!(
            fs::read_to_string(restored.path().join("rename-new")).unwrap(),
            "move"
        );
    }

    #[test]
    fn delta_rejects_a_mismatched_base_manifest_before_mutating() {
        let source = tempfile::tempdir().unwrap();
        fs::write(source.path().join("value"), "old").unwrap();
        let full = build_revision(source.path(), None, false).unwrap();
        fs::write(source.path().join("value"), "new").unwrap();
        let delta = build_revision(source.path(), Some(&full.snapshot), false).unwrap();

        let restored = tempfile::tempdir().unwrap();
        let mut wrong_base = apply_archive(restored.path(), &full.bytes, None).unwrap();
        wrong_base.manifest_digest = Some("0".repeat(64));
        let error = apply_archive(restored.path(), &delta.bytes, Some(&wrong_base)).unwrap_err();
        assert!(error.to_string().contains("base manifest digest"));
        assert_eq!(
            fs::read_to_string(restored.path().join("value")).unwrap(),
            "old"
        );
    }

    #[test]
    fn collision_with_the_old_deterministic_temp_name_round_trips_exactly() {
        let source = tempfile::tempdir().unwrap();
        fs::write(source.path().join("foo.txt"), "value").unwrap();
        let old_temporary = Path::new("foo.txt").with_extension(format!(
            "pipefs-restore-{}",
            blake3::hash(b"foo.txt").to_hex()
        ));
        fs::create_dir(source.path().join(&old_temporary)).unwrap();
        let full = build_revision(source.path(), None, false).unwrap();

        let restored = tempfile::tempdir().unwrap();
        let snapshot = apply_archive(restored.path(), &full.bytes, None).unwrap();
        assert!(restored.path().join(old_temporary).is_dir());
        assert_eq!(snapshot.entries, full.snapshot.entries);
    }

    #[test]
    fn rejects_a_manifest_whose_exact_tree_has_a_non_directory_parent() {
        let manifest = ArchiveManifest {
            archive_version: ARCHIVE_VERSION,
            revision_type: RevisionKind::Full,
            base_manifest_digest: None,
            logical_size_bytes: 0,
            entries: vec![
                ArchiveEntry {
                    path: "parent".to_string(),
                    entry_type: ArchiveEntryKind::Symlink,
                    size: 0,
                    blake3: None,
                    modified_unix: 0,
                    mode: 0o777,
                    symlink_target: Some("target".to_string()),
                    payload: None,
                },
                ArchiveEntry {
                    path: "parent/child".to_string(),
                    entry_type: ArchiveEntryKind::File,
                    size: 0,
                    blake3: Some(blake3::hash(b"").to_hex().to_string()),
                    modified_unix: 0,
                    mode: 0o600,
                    symlink_target: None,
                    payload: Some("objects/00000001".to_string()),
                },
            ],
        };
        let bytes = encode_test_archive(&manifest, &[("objects/00000001", b"")]);
        let restored = tempfile::tempdir().unwrap();
        assert!(apply_archive(restored.path(), &bytes, None).is_err());
        assert!(fs::read_dir(restored.path()).unwrap().next().is_none());
    }

    #[cfg(unix)]
    #[test]
    fn delta_restores_through_a_read_only_directory_and_preserves_its_metadata() {
        use std::os::unix::fs::PermissionsExt;

        let source = tempfile::tempdir().unwrap();
        fs::create_dir(source.path().join("locked")).unwrap();
        fs::write(source.path().join("locked/value"), "old").unwrap();
        fs::set_permissions(
            source.path().join("locked"),
            fs::Permissions::from_mode(0o555),
        )
        .unwrap();
        let full = build_revision(source.path(), None, false).unwrap();
        fs::write(source.path().join("locked/value"), "new").unwrap();
        let delta = build_revision(source.path(), Some(&full.snapshot), false).unwrap();

        let restored = tempfile::tempdir().unwrap();
        let restored_base = apply_archive(restored.path(), &full.bytes, None).unwrap();
        let restored_snapshot =
            apply_archive(restored.path(), &delta.bytes, Some(&restored_base)).unwrap();
        assert_eq!(restored_snapshot.entries, delta.snapshot.entries);
        assert_eq!(
            fs::metadata(restored.path().join("locked"))
                .unwrap()
                .permissions()
                .mode()
                & 0o777,
            0o555
        );
    }

    #[cfg(unix)]
    #[test]
    fn symlinks_and_executable_modes_round_trip() {
        use std::os::unix::fs::PermissionsExt;
        let source = tempfile::tempdir().unwrap();
        fs::write(source.path().join("run"), "#!/bin/sh\n").unwrap();
        fs::set_permissions(source.path().join("run"), fs::Permissions::from_mode(0o755)).unwrap();
        std::os::unix::fs::symlink("run", source.path().join("link")).unwrap();
        let full = build_revision(source.path(), None, false).unwrap();
        let restored = tempfile::tempdir().unwrap();
        let snapshot = apply_archive(restored.path(), &full.bytes, None).unwrap();
        assert_eq!(
            fs::read_link(restored.path().join("link")).unwrap(),
            PathBuf::from("run")
        );
        assert_eq!(
            fs::metadata(restored.path().join("run"))
                .unwrap()
                .permissions()
                .mode()
                & 0o777,
            0o755
        );
        assert_eq!(snapshot.entries, full.snapshot.entries);
    }

    #[test]
    fn rejects_nonportable_and_escaping_paths() {
        assert!(validate_archive_path("../secret").is_err());
        assert!(validate_archive_path("a\\b").is_err());
        assert!(validate_archive_path("CON.txt").is_err());
        assert!(validate_symlink_target("link", "../outside").is_err());
        assert!(validate_symlink_target("link", "safe/CON.txt").is_err());
        assert!(validate_symlink_target("link", "safe/bad:name").is_err());
        assert!(validate_symlink_target("link", "safe/control\u{0001}").is_err());

        let mut names = HashMap::new();
        register_portable_name("Readme", &mut names).unwrap();
        assert!(register_portable_name("README", &mut names).is_err());

        let manifest = ArchiveManifest {
            archive_version: ARCHIVE_VERSION,
            revision_type: RevisionKind::Full,
            base_manifest_digest: None,
            logical_size_bytes: 0,
            entries: vec![ArchiveEntry {
                path: "../escape".to_string(),
                entry_type: ArchiveEntryKind::Directory,
                size: 0,
                blake3: None,
                modified_unix: 0,
                mode: 0o755,
                symlink_target: None,
                payload: None,
            }],
        };
        let bytes = encode_test_archive(&manifest, &[]);
        let restored = tempfile::tempdir().unwrap();
        assert!(apply_archive(restored.path(), &bytes, None).is_err());
        assert!(!restored.path().join("escape").exists());
    }

    #[cfg(unix)]
    #[test]
    fn rejects_special_files_during_scan() {
        let source = tempfile::tempdir().unwrap();
        let socket = source.path().join("agent.sock");
        let _listener = std::os::unix::net::UnixListener::bind(&socket).unwrap();
        let error = scan_workspace(source.path()).unwrap_err();
        assert!(error.to_string().contains("special files"));
    }

    #[cfg(unix)]
    #[test]
    fn descriptor_scan_rejects_directory_swapped_to_an_outside_symlink() {
        let source = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        fs::create_dir(source.path().join("child")).unwrap();
        fs::write(source.path().join("child/inside"), "inside").unwrap();
        fs::write(outside.path().join("secret"), "outside-secret").unwrap();

        let root = open_workspace_root(source.path()).unwrap();
        let before = scan_stat_child(&root, OsStr::new("child"), "child").unwrap();
        fs::rename(source.path().join("child"), source.path().join("saved")).unwrap();
        std::os::unix::fs::symlink(outside.path(), source.path().join("child")).unwrap();

        let error = open_scanned_child(
            &root,
            OsStr::new("child"),
            &before,
            UnixFileType::Directory,
            "child",
        )
        .expect_err("a swapped symlink must not be opened");
        assert!(error.to_string().contains("workspace_changed_during_scan"));
    }

    #[cfg(unix)]
    #[test]
    fn archive_reopen_rejects_a_symlinked_ancestor() {
        let source = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        fs::write(outside.path().join("secret"), "outside-secret").unwrap();
        std::os::unix::fs::symlink(outside.path(), source.path().join("ancestor")).unwrap();

        let workspace = ArchiveWorkspace::open(source.path()).unwrap();
        let digest = blake3::hash(b"outside-secret").to_hex().to_string();
        let expected = SnapshotEntry {
            entry_type: ArchiveEntryKind::File,
            size: 14,
            blake3: Some(digest),
            modified_unix: 0,
            mode: 0o600,
            symlink_target: None,
        };
        let error = workspace
            .open_file("ancestor/secret", &expected)
            .expect_err("a symlinked ancestor must not be followed");
        let message = format!("{error:#}");
        assert!(
            message.contains("opening workspace file without following symlinks")
                || message.contains("re-reading metadata for workspace file")
        );
    }

    #[cfg(unix)]
    #[test]
    fn scan_rejects_a_symlink_workspace_root() {
        let parent = tempfile::tempdir().unwrap();
        let real = parent.path().join("real");
        let link = parent.path().join("link");
        fs::create_dir(&real).unwrap();
        fs::write(real.join("secret"), "do-not-read-through-link").unwrap();
        std::os::unix::fs::symlink(&real, &link).unwrap();

        let error = scan_workspace(&link).unwrap_err();
        assert!(
            format!("{error:#}")
                .contains("opening PipeFS workspace root without following symlinks")
        );
    }

    #[test]
    fn corruption_is_rejected_before_extraction() {
        let source = tempfile::tempdir().unwrap();
        fs::write(source.path().join("file"), "payload").unwrap();
        let mut artifact = build_revision(source.path(), None, false).unwrap();
        let middle = artifact.bytes.len() / 2;
        artifact.bytes[middle] ^= 0x80;
        let restored = tempfile::tempdir().unwrap();
        assert!(apply_archive(restored.path(), &artifact.bytes, None).is_err());
        assert!(fs::read_dir(restored.path()).unwrap().next().is_none());
    }

    #[test]
    fn delta_requires_a_base_manifest_digest() {
        let manifest = ArchiveManifest {
            archive_version: ARCHIVE_VERSION,
            revision_type: RevisionKind::Delta,
            base_manifest_digest: None,
            logical_size_bytes: 0,
            entries: Vec::new(),
        };
        assert!(validate_manifest(&manifest).is_err());
    }
}
