//! Inject-gated browser automation for hi (`browser_exec`).
//!
//! Default on. Call [`configure`] from session setup. SSRF blocks link-local
//! and cloud metadata always; RFC1918/loopback only when `allow_private`.

use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::Result;

mod cdp;
mod install;
mod parser;
mod ssrf;

pub use install::install_extension;
pub use parser::{BrowserCommand, parse_script};
pub use ssrf::{
    BrowserPolicy, check_navigation_url, check_resolved_ips, check_url_with_dns,
    resolve_and_check_host,
};

static ENABLED: AtomicBool = AtomicBool::new(true);
static ALLOW_PRIVATE: AtomicBool = AtomicBool::new(false);

#[derive(Clone, Copy, Debug)]
pub struct BrowserConfig {
    pub enabled: bool,
    pub allow_private: bool,
}

impl Default for BrowserConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            allow_private: false,
        }
    }
}

pub fn configure(config: BrowserConfig) {
    ENABLED.store(config.enabled, Ordering::Relaxed);
    ALLOW_PRIVATE.store(config.allow_private, Ordering::Relaxed);
}

pub fn is_enabled() -> bool {
    ENABLED.load(Ordering::Relaxed)
}

pub fn allow_private() -> bool {
    ALLOW_PRIVATE.load(Ordering::Relaxed)
}

pub fn active_policy() -> BrowserPolicy {
    BrowserPolicy {
        allow_private: allow_private(),
    }
}

/// A PNG (or other) screenshot to attach as model vision after the tool result.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BrowserImage {
    pub data: String,
    pub media_type: String,
}

#[derive(Clone, Debug, Default)]
pub struct BrowserExecResult {
    pub text: String,
    pub images: Vec<BrowserImage>,
}

/// Run a `browser_exec` script. `mode` is `headless` (default) or `dedicated`.
pub async fn run_exec(arguments: &str) -> Result<BrowserExecResult> {
    if !is_enabled() {
        anyhow::bail!(
            "browser_exec is disabled; set [browser] enabled = true in hi.toml (it defaults on)"
        );
    }
    let args: ExecArgs = serde_json::from_str(arguments).unwrap_or(ExecArgs {
        script: arguments.to_string(),
        mode: None,
    });
    let script = args.script.trim();
    if script.is_empty() {
        anyhow::bail!(
            "browser_exec requires a `script` of goto/click/type/screenshot/ax/wait/eval/scroll lines"
        );
    }
    let commands = parse_script(script)?;
    let policy = active_policy();
    for command in &commands {
        if let BrowserCommand::Goto { url } = command {
            check_url_with_dns(url, policy)?;
        }
    }
    let mode = args.mode.as_deref().unwrap_or("headless");
    cdp::run(mode, &commands, policy).await
}

#[derive(serde::Deserialize)]
struct ExecArgs {
    #[serde(default)]
    script: String,
    #[serde(default)]
    mode: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    static CONFIG_TEST_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

    #[tokio::test]
    async fn metadata_goto_fails_before_launching_chrome() {
        let _guard = CONFIG_TEST_LOCK.lock().await;
        configure(BrowserConfig {
            enabled: true,
            allow_private: true,
        });
        let err = run_exec(r#"{"script":"goto http://metadata.google.internal/latest"}"#)
            .await
            .expect_err("metadata host");
        let msg = err.to_string();
        assert!(msg.contains("metadata") || msg.contains("refused"), "{msg}");
        let err = run_exec(r#"{"script":"goto http://169.254.169.254/latest"}"#)
            .await
            .expect_err("link-local metadata");
        assert!(
            err.to_string().contains("blocked") || err.to_string().contains("refused"),
            "{err}"
        );
    }
}
