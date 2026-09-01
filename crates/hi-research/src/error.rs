use std::fmt;

/// Classified research API failure.
#[derive(Debug)]
pub struct ResearchError {
    pub kind: ResearchErrorKind,
    pub status: Option<u16>,
    pub code: Option<String>,
    pub message: String,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ResearchErrorKind {
    /// Caller should continue without research (missing key / plane down).
    FailOpen,
    /// Auth, validation, or apply/network failure that is not a missing plane.
    Hard,
}

impl ResearchError {
    pub fn fail_open(message: impl Into<String>) -> Self {
        Self {
            kind: ResearchErrorKind::FailOpen,
            status: None,
            code: None,
            message: message.into(),
        }
    }

    pub fn hard(message: impl Into<String>) -> Self {
        Self {
            kind: ResearchErrorKind::Hard,
            status: None,
            code: None,
            message: message.into(),
        }
    }

    pub fn from_http(status: u16, body: &str) -> Self {
        let code = extract_code(body);
        let message = extract_message(body).unwrap_or_else(|| {
            if body.trim().is_empty() {
                format!("HTTP {status}")
            } else {
                body.chars().take(400).collect()
            }
        });
        let fail_open = matches!(status, 401 | 404 | 503)
            || code.as_deref().is_some_and(|code| {
                code == crate::RESEARCH_UNAVAILABLE_CODE || code.ends_with("_unavailable")
            });
        Self {
            kind: if fail_open {
                ResearchErrorKind::FailOpen
            } else {
                ResearchErrorKind::Hard
            },
            status: Some(status),
            code,
            message,
        }
    }

    pub fn is_fail_open(&self) -> bool {
        self.kind == ResearchErrorKind::FailOpen
    }
}

impl fmt::Display for ResearchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match (&self.code, self.status) {
            (Some(code), Some(status)) => write!(f, "{code} (HTTP {status}): {}", self.message),
            (Some(code), None) => write!(f, "{code}: {}", self.message),
            (None, Some(status)) => write!(f, "HTTP {status}: {}", self.message),
            (None, None) => f.write_str(&self.message),
        }
    }
}

impl std::error::Error for ResearchError {}

fn extract_code(body: &str) -> Option<String> {
    let value: serde_json::Value = serde_json::from_str(body).ok()?;
    value
        .pointer("/error/code")
        .or_else(|| value.get("code"))
        .and_then(serde_json::Value::as_str)
        .map(str::to_owned)
}

fn extract_message(body: &str) -> Option<String> {
    let value: serde_json::Value = serde_json::from_str(body).ok()?;
    value
        .pointer("/error/message")
        .or_else(|| value.get("message"))
        .and_then(serde_json::Value::as_str)
        .map(str::to_owned)
}
