//! Minimal reference guest for the `hi:engine` Component Model world.
//!
//! The production guest will be produced by the decision-engine build. This
//! intentionally small guest is useful for ABI smoke tests and proves that a
//! replacement module can be compiled independently of the native harness.

wit_bindgen::generate!({
    path: "../hi-engine-api/wit",
});

struct ReferenceEngine;

impl Guest for ReferenceEngine {
    fn step(input_json: String) -> String {
        // The reference guest only demonstrates the protocol handshake. It
        // requests one model round at turn start and waits for host events
        // afterwards; it has no host imports and cannot perform effects.
        if input_json.contains("\"type\":\"turn_started\"") {
            r#"[{"type":"request_model","idempotency_key":"reference:request-model","request_id":"reference-model-1","messages_json":"[]"}]"#.into()
        } else {
            "[{\"type\":\"wait\",\"idempotency_key\":\"reference:wait\"}]".into()
        }
    }

    fn serialize_state() -> Vec<u8> {
        Vec::new()
    }
}

export!(ReferenceEngine);
