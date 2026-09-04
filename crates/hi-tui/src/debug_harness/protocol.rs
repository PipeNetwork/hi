use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::{
    TUI_STDIO_PROTOCOL_VERSION, default_height, default_model, default_provider, default_width,
};
use crate::event::UiEvent;

#[derive(Deserialize)]
pub(super) struct WireRequest {
    #[serde(default)]
    pub(super) id: Option<Value>,
    #[serde(flatten)]
    pub(super) command: Command,
}

#[derive(Deserialize)]
#[serde(tag = "command", rename_all = "snake_case")]
pub(super) enum Command {
    Hello,
    Reset {
        #[serde(default = "default_width")]
        width: u16,
        #[serde(default = "default_height")]
        height: u16,
        #[serde(default = "default_provider")]
        provider: String,
        #[serde(default = "default_model")]
        model: String,
    },
    Resize {
        width: u16,
        height: u16,
    },
    Focus {
        focused: bool,
    },
    Key {
        key: String,
        #[serde(default)]
        ctrl: bool,
        #[serde(default)]
        alt: bool,
        #[serde(default)]
        shift: bool,
    },
    Paste {
        text: String,
    },
    Transcript {
        event: UiEvent,
    },
    ClearTranscript,
    SessionEvent {
        event: hi_agent::SessionEvent,
    },
    SessionPatch {
        patch: hi_agent::SessionProjectionPatch,
    },
    SessionSnapshot {
        snapshot: Box<hi_agent::SessionProjectionSnapshot>,
    },
    Render,
    Inspect,
}

#[derive(Serialize)]
pub(super) struct WireResponse {
    protocol_version: u16,
    #[serde(skip_serializing_if = "Option::is_none")]
    id: Option<Value>,
    ok: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<WireError>,
}

impl WireResponse {
    pub(super) fn success(id: Option<Value>, result: Value) -> Self {
        Self {
            protocol_version: TUI_STDIO_PROTOCOL_VERSION,
            id,
            ok: true,
            result: Some(result),
            error: None,
        }
    }

    pub(super) fn error(id: Option<Value>, code: &'static str, message: &str) -> Self {
        Self {
            protocol_version: TUI_STDIO_PROTOCOL_VERSION,
            id,
            ok: false,
            result: None,
            error: Some(WireError {
                code,
                message: message.to_owned(),
            }),
        }
    }
}

#[derive(Serialize)]
struct WireError {
    code: &'static str,
    message: String,
}

#[derive(Debug)]
pub(super) struct HarnessError {
    pub(super) code: &'static str,
    pub(super) message: String,
}

impl HarnessError {
    pub(super) fn new(code: &'static str, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }
}
