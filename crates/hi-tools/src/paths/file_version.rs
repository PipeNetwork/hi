//! File identity used to keep per-turn reads fresh after external writes.

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct FileVersion {
    size: u64,
    modified: std::time::SystemTime,
    #[cfg(unix)]
    device: u64,
    #[cfg(unix)]
    inode: u64,
    #[cfg(unix)]
    changed: (i64, i64),
}

impl FileVersion {
    pub(crate) fn from_metadata(metadata: &std::fs::Metadata) -> Option<Self> {
        if !metadata.is_file() {
            return None;
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            Some(Self {
                size: metadata.len(),
                modified: metadata.modified().ok()?,
                device: metadata.dev(),
                inode: metadata.ino(),
                changed: (metadata.ctime(), metadata.ctime_nsec()),
            })
        }
        #[cfg(not(unix))]
        {
            // Size + mtime alone can alias an editor's same-size replacement
            // or a write that restores its mtime. Read fresh on platforms where
            // we cannot obtain the stronger identity/change stamp above.
            None
        }
    }
}
