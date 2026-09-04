use std::sync::{Arc, Barrier};

use super::*;

#[test]
fn parallel_capture_and_materialize_share_one_consistent_store() {
    let temporary = tempfile::tempdir().unwrap();
    let workspace = temporary.path().join("workspace");
    let state = temporary.path().join("state");
    fs::create_dir(&workspace).unwrap();
    fs::create_dir(&state).unwrap();
    for index in 0..32 {
        fs::write(
            workspace.join(format!("file-{index:02}.txt")),
            format!("content-{index}\n"),
        )
        .unwrap();
    }

    let barrier = Arc::new(Barrier::new(9));
    let results = std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for worker in 0..8 {
            let barrier = Arc::clone(&barrier);
            let workspace = &workspace;
            let state = &state;
            let base = temporary.path();
            handles.push(scope.spawn(move || {
                barrier.wait();
                let mut ids = Vec::new();
                for round in 0..8 {
                    let id = create(workspace, state).unwrap();
                    let destination = base.join(format!("materialized-{worker}-{round}"));
                    materialize(workspace, state, &id, &destination).unwrap();
                    assert_eq!(
                        fs::read_to_string(destination.join("file-17.txt")).unwrap(),
                        "content-17\n"
                    );
                    fs::remove_dir_all(destination).unwrap();
                    ids.push(id);
                }
                ids
            }));
        }
        barrier.wait();
        handles
            .into_iter()
            .flat_map(|handle| handle.join().unwrap())
            .collect::<Vec<_>>()
    });

    assert!(
        results.windows(2).all(|pair| pair[0] == pair[1]),
        "an unchanged workspace must always produce one content identity"
    );
}

#[cfg(unix)]
#[test]
fn failed_capture_collects_unreferenced_objects() {
    use std::ffi::CString;
    use std::os::unix::ffi::OsStrExt as _;

    let temporary = tempfile::tempdir().unwrap();
    let workspace = temporary.path().join("workspace");
    let state = temporary.path().join("state");
    fs::create_dir(&workspace).unwrap();
    fs::create_dir(&state).unwrap();
    let store = Store::open(&workspace.canonicalize().unwrap(), &state).unwrap();
    let bytes = b"unreferenced";
    let object = digest(bytes);
    write_object(&store, &object, bytes).unwrap();
    let fifo = workspace.join("unsupported-fifo");
    let fifo_c = CString::new(fifo.as_os_str().as_bytes()).unwrap();
    // SAFETY: fifo_c is a valid NUL-terminated path owned for the call.
    assert_eq!(unsafe { libc::mkfifo(fifo_c.as_ptr(), 0o600) }, 0);

    assert!(capture(&workspace, &state, &store).is_err());
    assert!(
        !store
            .objects_dir
            .join(&object[..2])
            .join(&object[2..])
            .exists(),
        "failed capture left an unreachable object"
    );
}
