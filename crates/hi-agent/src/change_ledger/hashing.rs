//! Fresh content fingerprints with bounded reads from a validated descriptor.

use std::io::Read;

use super::{CancellationToken, Context, MAX_AUTOMATIC_FILE_BYTES, Path, Result, Sha256};
use super::{Digest, ensure, ensure_scan_active};

pub(super) fn hash_file_streaming(
    path: &Path,
    cancellation: Option<&CancellationToken>,
) -> Result<(String, u64)> {
    ensure_scan_active(cancellation)?;
    let file = open_regular_file(path)?;
    hash_reader(file, path, cancellation)
}

fn open_regular_file(path: &Path) -> Result<std::fs::File> {
    let mut options = std::fs::OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        // A FIFO replacing a previously statted source must not block before
        // we can inspect the descriptor and reject its type.
        options.custom_flags(libc::O_NONBLOCK);
    }
    let file = options
        .open(path)
        .with_context(|| format!("reading {}", path.display()))?;
    let metadata = file
        .metadata()
        .with_context(|| format!("reading metadata for {}", path.display()))?;
    ensure!(
        metadata.is_file(),
        "{} is not a regular file",
        path.display()
    );
    ensure!(
        metadata.len() <= MAX_AUTOMATIC_FILE_BYTES,
        "{} exceeds the workspace hashing limit of {MAX_AUTOMATIC_FILE_BYTES} bytes",
        path.display()
    );
    Ok(file)
}

fn hash_reader(
    reader: impl Read,
    path: &Path,
    cancellation: Option<&CancellationToken>,
) -> Result<(String, u64)> {
    // Read one extra byte to distinguish an exact-limit file from growth.
    // No successful digest may describe only a truncated prefix.
    let mut reader = reader.take(MAX_AUTOMATIC_FILE_BYTES + 1);
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 64 * 1024];
    let mut hashed_len = 0u64;
    loop {
        ensure_scan_active(cancellation)?;
        let n = reader
            .read(&mut buf)
            .with_context(|| format!("reading {}", path.display()))?;
        if n == 0 {
            break;
        }
        hashed_len += n as u64;
        ensure!(
            hashed_len <= MAX_AUTOMATIC_FILE_BYTES,
            "{} grew beyond the workspace hashing limit of {MAX_AUTOMATIC_FILE_BYTES} bytes",
            path.display()
        );
        hasher.update(&buf[..n]);
    }
    ensure_scan_active(cancellation)?;
    Ok((format!("sha256:{:x}", hasher.finalize()), hashed_len))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hashes_exact_bytes_and_accepts_the_size_limit() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("source.rs");
        std::fs::write(&path, b"abc").unwrap();
        assert_eq!(
            hash_file_streaming(&path, None).unwrap(),
            (
                "sha256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad".into(),
                3
            )
        );
        let file = std::fs::File::create(&path).unwrap();
        file.set_len(MAX_AUTOMATIC_FILE_BYTES).unwrap();
        let (_, len) = hash_file_streaming(&path, None).unwrap();
        assert_eq!(len, MAX_AUTOMATIC_FILE_BYTES);
    }

    #[test]
    fn growth_after_descriptor_validation_cannot_seal_a_prefix() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("source.rs");
        std::fs::write(&path, b"before").unwrap();
        let mut reader = open_regular_file(&path).unwrap();
        // A concurrent writer grows the same inode after descriptor admission.
        let writer = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
        writer.set_len(MAX_AUTOMATIC_FILE_BYTES + 4096).unwrap();
        let error = hash_reader(&mut reader, &path, None).unwrap_err();
        assert!(error.to_string().contains("grew beyond"), "{error:#}");
        use std::io::Seek;
        assert_eq!(
            reader.stream_position().unwrap(),
            MAX_AUTOMATIC_FILE_BYTES + 1
        );
        let error = hash_file_streaming(&path, None).unwrap_err();
        assert!(error.to_string().contains("exceeds"), "{error:#}");
    }

    #[cfg(unix)]
    #[test]
    fn fifo_replacement_is_rejected_without_waiting_for_a_writer() {
        use std::os::unix::ffi::OsStrExt;

        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("source.rs");
        std::fs::write(&path, b"before").unwrap();
        assert!(std::fs::metadata(&path).unwrap().is_file());
        std::fs::remove_file(&path).unwrap();
        let c_path = std::ffi::CString::new(path.as_os_str().as_bytes()).unwrap();
        assert_eq!(unsafe { libc::mkfifo(c_path.as_ptr(), 0o600) }, 0);
        let error = hash_file_streaming(&path, None).unwrap_err();
        assert!(
            error.to_string().contains("not a regular file"),
            "{error:#}"
        );
    }

    #[test]
    fn cancellation_during_hashing_cannot_publish_a_digest() {
        struct CancelOnRead(CancellationToken);
        impl Read for CancelOnRead {
            fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
                buffer[0] = b'x';
                self.0.cancel();
                Ok(1)
            }
        }
        let cancellation = CancellationToken::new();
        let error = hash_reader(
            CancelOnRead(cancellation.clone()),
            Path::new("source.rs"),
            Some(&cancellation),
        )
        .unwrap_err();
        assert!(error.to_string().contains("cancelled"), "{error:#}");
    }
}
