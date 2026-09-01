//! Chunked remote-record reassembly tests.

use super::*;

/// Helper: build a RemoteRecordResponse with a given type and payload.
fn record(record_type: &str, payload_json: &str, seq: u64) -> RemoteRecordResponse {
    RemoteRecordResponse {
        record_type: record_type.to_string(),
        payload_json: payload_json.to_string(),
        record_seq: Some(seq),
    }
}

/// Helper: build chunk_part + chunk_commit records for a logical payload.
fn chunked_records(
    logical_id: &str,
    record_type: &str,
    payload: &str,
    start_seq: u64,
) -> Vec<RemoteRecordResponse> {
    use sha2::{Digest, Sha256};
    let mut parts = Vec::new();
    let mut start = 0;
    while start < payload.len() {
        let mut end = (start + CHUNK_PART_BYTES).min(payload.len());
        while !payload.is_char_boundary(end) {
            end -= 1;
        }
        parts.push(&payload[start..end]);
        start = end;
    }
    let mut out = Vec::new();
    let mut seq = start_seq;
    for (index, data) in parts.iter().enumerate() {
        out.push(record(
            "chunk_part",
            &serde_json::json!({
                "logical_id": logical_id,
                "index": index,
                "parts": parts.len(),
                "data": data,
            })
            .to_string(),
            seq,
        ));
        seq += 1;
    }
    out.push(record(
        "chunk_commit",
        &serde_json::json!({
            "logical_id": logical_id,
            "record_type": record_type,
            "parts": parts.len(),
            "sha256": format!("{:x}", Sha256::digest(payload.as_bytes())),
            "bytes": payload.len(),
        })
        .to_string(),
        seq,
    ));
    out
}

#[test]
fn reassemble_skips_incomplete_chunk_commit_with_missing_parts() {
    // A chunk_commit arrives but its chunk_part records were never
    // persisted (writer failed mid-way but the commit slipped through).
    // The reader must skip it with a warning, not bail.
    let commit = record(
        "chunk_commit",
        &serde_json::json!({
            "logical_id": "abc123",
            "record_type": "message",
            "parts": 2,
            "sha256": "deadbeef",
            "bytes": 100,
        })
        .to_string(),
        1,
    );
    let normal = record(
        "message",
        &serde_json::to_string(&Message::user("survives")).unwrap(),
        2,
    );
    let records = vec![commit, normal];
    let output = reassemble_remote_records(records).expect("must not bail on incomplete commit");
    assert_eq!(
        output.len(),
        1,
        "incomplete chunk_commit is skipped, normal record survives"
    );
    assert!(output[0].payload_json.contains("survives"));
}

#[test]
fn reassemble_skips_chunk_commit_with_partial_parts() {
    // Two parts expected, only one arrived. Must skip, not bail.
    let part = record(
        "chunk_part",
        &serde_json::json!({
            "logical_id": "xyz789",
            "index": 0,
            "parts": 2,
            "data": "partial",
        })
        .to_string(),
        1,
    );
    let commit = record(
        "chunk_commit",
        &serde_json::json!({
            "logical_id": "xyz789",
            "record_type": "message",
            "parts": 2,
            "sha256": "deadbeef",
            "bytes": 100,
        })
        .to_string(),
        2,
    );
    let records = vec![part, commit];
    let output =
        reassemble_remote_records(records).expect("partial parts must not bail the reader");
    assert!(
        output.is_empty(),
        "incomplete chunk_commit with partial parts is skipped"
    );
}

#[test]
fn reassemble_skips_chunk_commit_with_hash_mismatch() {
    // All parts present but the reassembled hash doesn't match the commit.
    let payload = "hello world";
    let mut records = chunked_records("hashbad", "message", payload, 1);
    // Corrupt the sha256 in the commit record (last record).
    let commit_idx = records.len() - 1;
    let mut commit_value: serde_json::Value =
        serde_json::from_str(&records[commit_idx].payload_json).unwrap();
    commit_value["sha256"] =
        serde_json::json!("0000000000000000000000000000000000000000000000000000000000000000");
    records[commit_idx].payload_json = commit_value.to_string();

    let output =
        reassemble_remote_records(records).expect("hash mismatch must not bail the reader");
    assert!(
        output.is_empty(),
        "chunk_commit with hash mismatch is skipped"
    );
}

#[test]
fn reassemble_tolerates_orphaned_chunk_parts_without_commit() {
    // chunk_part records exist but no chunk_commit ever arrived. The reader
    // must skip them with a warning, not bail.
    let part = record(
        "chunk_part",
        &serde_json::json!({
            "logical_id": "orphan1",
            "index": 0,
            "parts": 2,
            "data": "orphaned",
        })
        .to_string(),
        1,
    );
    let normal = record(
        "message",
        &serde_json::to_string(&Message::user("survives")).unwrap(),
        2,
    );
    let records = vec![part, normal];
    let output =
        reassemble_remote_records(records).expect("orphaned parts must not bail the reader");
    assert_eq!(
        output.len(),
        1,
        "orphaned parts skipped, normal record kept"
    );
    assert!(output[0].payload_json.contains("survives"));
}

#[test]
fn reassemble_complete_chunked_record_round_trips() {
    // Sanity: a well-formed chunked record still reassembles correctly.
    let payload = serde_json::to_string(&Message::user("x".repeat(MAX_RECORD_WIRE_BYTES))).unwrap();
    let records = chunked_records("good1", "message", &payload, 1);
    let output = reassemble_remote_records(records).expect("complete chunked record reassembles");
    assert_eq!(output.len(), 1);
    assert_eq!(output[0].record_type, "message");
    assert_eq!(output[0].payload_json, payload);
}

#[test]
fn reassemble_skips_malformed_chunk_part_payload() {
    // A chunk_part with invalid JSON payload must be skipped, not bailed.
    let bad_part = record("chunk_part", "{not valid json", 1);
    let normal = record(
        "message",
        &serde_json::to_string(&Message::user("survives")).unwrap(),
        2,
    );
    let records = vec![bad_part, normal];
    let output =
        reassemble_remote_records(records).expect("malformed chunk_part must not bail the reader");
    assert_eq!(
        output.len(),
        1,
        "malformed part skipped, normal record kept"
    );
    assert!(output[0].payload_json.contains("survives"));
}

#[test]
fn reassemble_skips_malformed_chunk_commit_payload() {
    // A chunk_commit with invalid JSON payload must be skipped, not bailed.
    let bad_commit = record("chunk_commit", "{not valid json", 1);
    let normal = record(
        "message",
        &serde_json::to_string(&Message::user("survives")).unwrap(),
        2,
    );
    let records = vec![bad_commit, normal];
    let output = reassemble_remote_records(records)
        .expect("malformed chunk_commit must not bail the reader");
    assert_eq!(
        output.len(),
        1,
        "malformed commit skipped, normal record kept"
    );
    assert!(output[0].payload_json.contains("survives"));
}

#[test]
fn reassemble_skips_chunk_commit_missing_required_fields() {
    // A chunk_commit that is valid JSON but missing required fields
    // (e.g. no "parts") must be skipped, not bailed.
    let bad_commit = record(
        "chunk_commit",
        &serde_json::json!({
            "logical_id": "nofields",
            "record_type": "message",
            "sha256": "abc",
        })
        .to_string(),
        1,
    );
    let normal = record(
        "message",
        &serde_json::to_string(&Message::user("survives")).unwrap(),
        2,
    );
    let records = vec![bad_commit, normal];
    let output = reassemble_remote_records(records)
        .expect("chunk_commit missing fields must not bail the reader");
    assert_eq!(
        output.len(),
        1,
        "incomplete commit skipped, normal record kept"
    );
    assert!(output[0].payload_json.contains("survives"));
}
