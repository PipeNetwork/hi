use super::*;

/// Keep process/effect receipts when a long tool log is shortened. Content remains
/// explicitly partial; the verifier must not infer facts from omitted log bytes.
fn bounded_result(output: &str, limit: usize) -> String {
    let Ok(mut value) = serde_json::from_str::<Value>(output) else {
        return output.to_owned();
    };
    if value["evidence_source"] != "client_reported" {
        return output.to_owned();
    }
    if let Some(content) = value["content"].as_str().filter(|s| s.len() > limit) {
        let mut end = limit;
        while !content.is_char_boundary(end) {
            end -= 1;
        }
        let total = content.len();
        value["content"] = json!(&content[..end]);
        value["content_omission"] = json!({"retained_bytes":end,"original_bytes":total});
    }
    value.to_string()
}

pub(crate) fn preserve_result_receipts(original: &[Message], shortened: &mut [Message]) {
    for message in shortened {
        for content in &mut message.content {
            if let Content::ToolResult { call_id, output } = content {
                let original = original
                    .iter()
                    .flat_map(|m| &m.content)
                    .find_map(|c| match c {
                        Content::ToolResult {
                            call_id: id,
                            output,
                        } if id == call_id => Some(output),
                        _ => None,
                    });
                if let Some(original) = original.filter(|s| s.as_str() != output) {
                    *output = bounded_result(original, 512);
                }
            }
        }
    }
}

/// A summary is not execution evidence. Retain a bounded, contiguous suffix of
/// original call/result batches so post-compaction claims can still be checked.
pub(crate) fn retain_execution_evidence(original: &[Message], summary: &mut Vec<Message>) {
    let mut batches = Vec::new();
    let mut index = 0;
    while index < original.len() {
        let calls = original[index]
            .content
            .iter()
            .filter(|c| matches!(c, Content::ToolCall { .. }))
            .count();
        if calls == 0 {
            index += 1;
            continue;
        }
        let end = index + 1 + calls;
        if end > original.len()
            || !original[index + 1..end].iter().all(|m| {
                m.content
                    .iter()
                    .any(|c| matches!(c, Content::ToolResult { .. }))
            })
        {
            index += 1;
            continue;
        }
        let mut batch = original[index..end].to_vec();
        for m in &mut batch {
            for c in &mut m.content {
                if let Content::ToolResult { output, .. } = c {
                    *output = bounded_result(output, 1024);
                }
            }
        }
        batches.push(batch);
        index = end;
    }
    let mut kept = Vec::new();
    let mut bytes: usize = 0;
    for batch in batches.into_iter().rev() {
        let size = serde_json::to_vec(&batch)
            .map(|v| v.len())
            .unwrap_or(usize::MAX);
        if bytes.saturating_add(size) > 16_000 {
            break;
        }
        bytes += size;
        kept.push(batch);
    }
    for batch in kept.into_iter().rev() {
        summary.extend(batch);
    }
}
