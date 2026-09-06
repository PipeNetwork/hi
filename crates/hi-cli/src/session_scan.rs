use std::fs::File;
use std::io::{BufReader, Read, Take};
use std::path::Path;

use hi_ai::Role;

/// Open a bounded snapshot so a busy append cannot extend a scan forever.
pub(crate) fn session_snapshot_reader(path: &Path) -> std::io::Result<BufReader<Take<File>>> {
    let file = File::open(path)?;
    let snapshot_len = file.metadata()?.len();
    Ok(BufReader::new(file.take(snapshot_len)))
}

/// Count JSONL records with fixed memory. A final unterminated record counts.
pub(super) fn session_line_count(path: &Path) -> usize {
    let Ok(mut reader) = session_snapshot_reader(path) else {
        return 0;
    };
    let mut buffer = [0_u8; 64 * 1024];
    let mut lines = 0_usize;
    let mut saw_bytes = false;
    let mut ended_with_newline = false;
    loop {
        let Ok(read) = reader.read(&mut buffer) else {
            return 0;
        };
        if read == 0 {
            break;
        }
        saw_bytes = true;
        ended_with_newline = buffer[read - 1] == b'\n';
        lines = lines.saturating_add(buffer[..read].iter().filter(|byte| **byte == b'\n').count());
    }
    lines.saturating_add(usize::from(saw_bytes && !ended_with_newline))
}

/// Concise status shown when a session is resumed.
pub(crate) fn resume_summary(loaded: &super::LoadedSession) -> String {
    let n = loaded
        .messages
        .iter()
        .filter(|message| message.role != Role::System)
        .count();
    let last = loaded
        .messages
        .iter()
        .rev()
        .find(|message| message.role == Role::User)
        .map(|message| hi_agent::ui::clip(&message.text(), 60))
        .unwrap_or_default();
    format!("Resumed: {n} messages, last: '{last}'")
}
