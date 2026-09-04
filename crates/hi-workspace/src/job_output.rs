use serde::{Deserialize, Serialize};

use crate::WorkspaceJobSnapshot;

pub(crate) const DEFAULT_OUTPUT_BYTES: usize = 256 * 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JobOutputStream {
    Stdout,
    Stderr,
    System,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct JobOutputChunk {
    pub sequence: u64,
    pub stream: JobOutputStream,
    pub text: String,
}

pub(crate) fn truncate_output(job: &mut WorkspaceJobSnapshot, limit: usize) {
    while job.output_bytes > limit && job.output.len() > 1 {
        let removed = job.output.remove(0);
        job.output_bytes = job.output_bytes.saturating_sub(removed.text.len());
        job.output_truncated = true;
    }
    if job.output_bytes > limit {
        let chunk = job.output.first_mut().expect("non-zero output has a chunk");
        chunk.text = utf8_suffix(&chunk.text, limit).to_owned();
        job.output_bytes = chunk.text.len();
        job.output_truncated = true;
    }
}

fn utf8_suffix(value: &str, max_bytes: usize) -> &str {
    if value.len() <= max_bytes {
        return value;
    }
    let minimum = value.len().saturating_sub(max_bytes);
    let start = value
        .char_indices()
        .find_map(|(index, _)| (index >= minimum).then_some(index))
        .unwrap_or(value.len());
    &value[start..]
}

#[cfg(test)]
mod tests {
    use super::utf8_suffix;

    #[test]
    fn suffix_respects_utf8_boundaries() {
        assert_eq!(utf8_suffix("aéz", 3), "éz");
        assert_eq!(utf8_suffix("aéz", 2), "z");
        assert_eq!(utf8_suffix("hello", 0), "");
    }
}
