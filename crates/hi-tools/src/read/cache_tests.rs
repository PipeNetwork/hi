use super::*;

#[tokio::test]
async fn reread_observes_external_same_length_edit_in_the_same_turn() {
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("source.rs");
    std::fs::write(&path, "before\n").unwrap();
    let cache = std::sync::Mutex::new(ReadCache::new());
    let before = run_read(root.path(), &cache, r#"{"path":"source.rs"}"#)
        .await
        .unwrap();
    assert!(before.content.contains("before"));

    std::fs::write(&path, "after!\n").unwrap();
    let after = run_read(root.path(), &cache, r#"{"path":"source.rs"}"#)
        .await
        .unwrap();
    assert!(after.content.contains("after!"), "{}", after.content);
    assert!(!after.content.contains("before"));
}

#[tokio::test]
async fn reread_does_not_return_cached_contents_of_a_deleted_file() {
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("deleted.rs");
    std::fs::write(&path, "outdated\n").unwrap();
    let cache = std::sync::Mutex::new(ReadCache::new());
    run_read(root.path(), &cache, r#"{"path":"deleted.rs"}"#)
        .await
        .unwrap();
    std::fs::remove_file(&path).unwrap();
    assert!(
        run_read(root.path(), &cache, r#"{"path":"deleted.rs"}"#)
            .await
            .is_err()
    );
}

#[tokio::test]
async fn reread_observes_atomic_replacement_with_preserved_length_and_mtime() {
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("source.rs");
    std::fs::write(&path, "before\n").unwrap();
    let modified = std::fs::metadata(&path).unwrap().modified().unwrap();
    let cache = std::sync::Mutex::new(ReadCache::new());
    run_read(root.path(), &cache, r#"{"path":"source.rs"}"#)
        .await
        .unwrap();

    let replacement = root.path().join("replacement.rs");
    std::fs::write(&replacement, "after!\n").unwrap();
    std::fs::File::options()
        .write(true)
        .open(&replacement)
        .unwrap()
        .set_times(std::fs::FileTimes::new().set_modified(modified))
        .unwrap();
    std::fs::rename(replacement, &path).unwrap();
    let output = run_read(root.path(), &cache, r#"{"path":"source.rs"}"#)
        .await
        .unwrap();
    assert!(output.content.contains("after!"), "{}", output.content);
}

#[tokio::test]
async fn reread_observes_in_place_write_that_restores_mtime() {
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("source.rs");
    std::fs::write(&path, "before\n").unwrap();
    let modified = std::fs::metadata(&path).unwrap().modified().unwrap();
    let cache = std::sync::Mutex::new(ReadCache::new());
    run_read(root.path(), &cache, r#"{"path":"source.rs"}"#)
        .await
        .unwrap();

    std::fs::write(&path, "after!\n").unwrap();
    std::fs::File::options()
        .write(true)
        .open(&path)
        .unwrap()
        .set_times(std::fs::FileTimes::new().set_modified(modified))
        .unwrap();
    let output = run_read(root.path(), &cache, r#"{"path":"source.rs"}"#)
        .await
        .unwrap();
    assert!(output.content.contains("after!"), "{}", output.content);
}

#[cfg(unix)]
#[tokio::test]
async fn cached_read_rejects_replacement_with_a_fifo_without_blocking() {
    use std::os::unix::ffi::OsStrExt;

    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("source.rs");
    std::fs::write(&path, "before\n").unwrap();
    let cache = std::sync::Mutex::new(ReadCache::new());
    run_read(root.path(), &cache, r#"{"path":"source.rs"}"#)
        .await
        .unwrap();
    std::fs::remove_file(&path).unwrap();
    let c_path = std::ffi::CString::new(path.as_os_str().as_bytes()).unwrap();
    assert_eq!(unsafe { libc::mkfifo(c_path.as_ptr(), 0o600) }, 0);
    let result = run_read(root.path(), &cache, r#"{"path":"source.rs"}"#)
        .await
        .unwrap_err();
    assert!(
        result.to_string().contains("not a regular file"),
        "{result:#}"
    );
}
