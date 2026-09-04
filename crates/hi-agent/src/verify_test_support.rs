use super::Ui;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

pub(super) struct NullUi;

impl Ui for NullUi {
    fn assistant_text(&mut self, _: &str) {}
    fn assistant_reasoning(&mut self, _: &str) {}
    fn assistant_end(&mut self) {}
    fn tool_call(&mut self, _: &str, _: &str) {}
    fn tool_result(&mut self, _: &str, _: &str) {}
    fn status(&mut self, _: &str) {}
    fn turn_end(&mut self, _: &str) {}
}

pub(super) fn roots(label: &str) -> (PathBuf, PathBuf, PathBuf) {
    static N: AtomicU64 = AtomicU64::new(0);
    let base = std::env::temp_dir().join(format!(
        "hi-verifier-{label}-{}-{}",
        std::process::id(),
        N.fetch_add(1, Ordering::Relaxed)
    ));
    let root = base.join("workspace");
    let state = base.join("state");
    std::fs::create_dir_all(&root).unwrap();
    std::fs::create_dir_all(&state).unwrap();
    (base, root, state)
}

pub(super) async fn checkpoint(root: &Path, state: &Path) -> String {
    match hi_tools::checkpoint::create_detailed_with_state(root, state).await {
        hi_tools::checkpoint::CreateResult::Created(id) => id,
        other => panic!("checkpoint failed: {other:?}"),
    }
}
