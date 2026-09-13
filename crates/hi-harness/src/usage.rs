//! Session usage snapshot for `/usage` (grok-build's usage modal).

use std::path::PathBuf;

use hi_ai::{Content, Message, Role, Usage};

use crate::Harness;
use crate::tools::advertised_tools;

/// Pipe's public dashboard. `/usage manage` opens this; there is no grok.com
/// credit ledger on this path.
pub const BILLING_URL: &str = "https://pipenetwork.ai";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UsageTab {
    Limit,
    Context,
    Session,
}

impl UsageTab {
    pub const ALL: [Self; 3] = [Self::Limit, Self::Context, Self::Session];

    pub fn title(self) -> &'static str {
        match self {
            Self::Limit => "Usage limit",
            Self::Context => "Context usage",
            Self::Session => "Session info",
        }
    }

    pub fn next(self) -> Self {
        match self {
            Self::Limit => Self::Context,
            Self::Context => Self::Session,
            Self::Session => Self::Limit,
        }
    }

    pub fn prev(self) -> Self {
        match self {
            Self::Limit => Self::Session,
            Self::Context => Self::Limit,
            Self::Session => Self::Context,
        }
    }

    pub fn from_arg(arg: &str) -> Option<Self> {
        match arg.trim().to_ascii_lowercase().as_str() {
            "" | "show" | "limit" | "cost" => Some(Self::Limit),
            "context" | "ctx" => Some(Self::Context),
            "session" | "info" | "status" => Some(Self::Session),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UsageCategory {
    pub name: &'static str,
    pub tokens: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UsageSnapshot {
    pub model: String,
    pub base_url: String,
    pub permission: String,
    pub reasoning: String,
    pub sandbox: String,
    pub signed_in: bool,
    pub workspace: PathBuf,
    pub session_id: Option<String>,
    pub user_turns: u64,
    pub checkpoints: usize,
    pub window: u64,
    pub occupancy: u64,
    pub usage: Usage,
    pub categories: Vec<UsageCategory>,
    pub info_rows: Vec<UsageCategory>,
}

impl UsageSnapshot {
    pub fn used_pct(&self) -> u64 {
        if self.window == 0 {
            return 0;
        }
        self.occupancy
            .saturating_mul(100)
            .saturating_div(self.window)
            .min(100)
    }

    pub fn free(&self) -> u64 {
        self.window.saturating_sub(self.occupancy)
    }

    /// Plain-text copy of the whole modal (`y` in grok).
    pub fn copy_all(&self) -> String {
        let mut out = String::new();
        for tab in UsageTab::ALL {
            out.push_str(tab.title());
            out.push('\n');
            out.push_str(&self.tab_text(tab));
            out.push('\n');
        }
        out
    }

    pub fn tab_text(&self, tab: UsageTab) -> String {
        match tab {
            UsageTab::Limit => self.limit_text(),
            UsageTab::Context => self.context_text(),
            UsageTab::Session => self.session_text(),
        }
    }

    fn limit_text(&self) -> String {
        let mut lines = vec![
            "Session".into(),
            format!(
                "  input         {:>10}",
                fmt_tokens(self.usage.input_tokens)
            ),
            format!(
                "  output        {:>10}",
                fmt_tokens(self.usage.output_tokens)
            ),
            format!(
                "  cache read    {:>10}",
                fmt_tokens(self.usage.cache_read_tokens)
            ),
            format!("  total         {:>10}", fmt_tokens(self.usage.total())),
        ];
        if self.usage.estimated {
            lines.push("  (contains estimates)".into());
        }
        lines.push(String::new());
        lines.push("Account".into());
        lines.push("  Pipe Network bills this API key.".into());
        lines.push(format!("  /usage manage  {BILLING_URL}"));
        lines.join("\n")
    }

    fn context_text(&self) -> String {
        let mut lines = vec![format!(
            "{} / {}  ({}%)",
            fmt_tokens(self.occupancy),
            fmt_tokens(self.window),
            self.used_pct()
        )];
        lines.push(occupancy_bar(self.used_pct(), 28));
        lines.push(String::new());
        for row in &self.categories {
            lines.push(format!("  {:<22} {:>8}", row.name, fmt_tokens(row.tokens)));
        }
        if !self.info_rows.is_empty() {
            lines.push(String::new());
            for row in &self.info_rows {
                lines.push(format!(
                    "  {:<22} {:>8}  (informational)",
                    row.name,
                    fmt_tokens(row.tokens)
                ));
            }
        }
        lines.join("\n")
    }

    fn session_text(&self) -> String {
        let auth = if self.signed_in {
            "signed in"
        } else {
            "missing — /login pipenetwork"
        };
        [
            format!("  {:<16} {}", "Model", self.model),
            format!("  {:<16} {}", "API", self.base_url),
            format!("  {:<16} {}", "Auth", auth),
            format!("  {:<16} {}", "Permission", self.permission),
            format!("  {:<16} {}", "Reasoning", self.reasoning),
            format!("  {:<16} {}", "Turns", self.user_turns),
            format!(
                "  {:<16} {} / {} ({}%)",
                "Context",
                fmt_tokens(self.occupancy),
                fmt_tokens(self.window),
                self.used_pct()
            ),
            format!("  {:<16} {}", "Sandbox", self.sandbox),
            format!("  {:<16} {}", "Workspace", self.workspace.display()),
            format!(
                "  {:<16} {}",
                "Session",
                self.session_id.as_deref().unwrap_or("(unsaved)")
            ),
            format!("  {:<16} {}", "Checkpoints", self.checkpoints),
        ]
        .join("\n")
    }
}

impl Harness {
    pub fn usage_snapshot(&self) -> UsageSnapshot {
        let window = u64::from(self.context_window().max(1));
        let occupancy = self.current_occupancy().min(window);
        let (categories, info_rows) = context_breakdown(&self.messages, occupancy, window);
        let session_id = self.session.as_ref().map(|session| {
            session
                .path()
                .file_stem()
                .map(|stem| stem.to_string_lossy().into_owned())
                .unwrap_or_else(|| session.path().display().to_string())
        });
        let user_turns = self
            .messages
            .iter()
            .filter(|message| message.role == Role::User)
            .count() as u64;
        let sandbox = if self.sandbox_enforced() {
            self.sandbox_backend_name().to_string()
        } else {
            format!("off ({})", self.sandbox_backend_name())
        };
        UsageSnapshot {
            model: self.model(),
            base_url: self.base_url().to_string(),
            permission: self.permission_mode().label().to_string(),
            reasoning: self
                .reasoning_effort()
                .map(|effort| effort.as_str().to_string())
                .unwrap_or_else(|| "off".into()),
            sandbox,
            signed_in: self.client.has_api_key(),
            workspace: self.workspace_root.clone(),
            session_id,
            user_turns,
            checkpoints: self.checkpoints.len(),
            window,
            occupancy,
            usage: self.session_usage,
            categories,
            info_rows,
        }
    }
}

fn context_breakdown(
    messages: &[Message],
    occupancy: u64,
    window: u64,
) -> (Vec<UsageCategory>, Vec<UsageCategory>) {
    let mut system = 0u64;
    let mut conversation = 0u64;
    let mut reasoning = 0u64;
    for message in messages {
        for block in &message.content {
            match block {
                Content::Thinking { text, .. } => reasoning = reasoning.saturating_add(est(text)),
                Content::Text(text) if message.role == Role::System => {
                    system = system.saturating_add(est(text));
                }
                Content::Text(text) => conversation = conversation.saturating_add(est(text)),
                Content::ToolCall {
                    name, arguments, ..
                } => {
                    conversation = conversation
                        .saturating_add(est(name))
                        .saturating_add(est(arguments));
                }
                Content::ToolResult { output, .. } => {
                    conversation = conversation.saturating_add(est(output));
                }
                Content::Image { .. } | Content::ProviderReplay { .. } => {}
            }
        }
    }
    let accounted = system
        .saturating_add(conversation)
        .saturating_add(reasoning);
    let overhead = occupancy.saturating_sub(accounted);
    let free = window.saturating_sub(occupancy);
    let categories = vec![
        UsageCategory {
            name: "System prompt",
            tokens: system,
        },
        UsageCategory {
            name: "Messages",
            tokens: conversation,
        },
        UsageCategory {
            name: "Reasoning / overhead",
            tokens: reasoning.saturating_add(overhead),
        },
        UsageCategory {
            name: "Free space",
            tokens: free,
        },
    ];
    let tools: u64 = advertised_tools()
        .iter()
        .map(|spec| {
            est(&spec.name)
                .saturating_add(est(&spec.description))
                .saturating_add(est(&spec.parameters.to_string()))
        })
        .sum();
    let info_rows = vec![UsageCategory {
        name: "Tool definitions",
        tokens: tools,
    }];
    (categories, info_rows)
}

fn est(text: &str) -> u64 {
    (text.len() as u64).div_ceil(4)
}

pub fn fmt_tokens(n: u64) -> String {
    match n {
        0..=999 => n.to_string(),
        1_000..=9_999 => format!("{:.1}k", n as f64 / 1_000.0),
        10_000..=999_999 => format!("{}k", n / 1_000),
        _ => format!("{:.1}M", n as f64 / 1_000_000.0),
    }
}

pub fn occupancy_bar(pct: u64, width: usize) -> String {
    let filled = ((pct as usize).saturating_mul(width) / 100).min(width);
    let mut out = String::new();
    for i in 0..width {
        out.push(if i < filled { '#' } else { '-' });
    }
    out
}

pub fn open_billing_url() -> Result<(), String> {
    let url = BILLING_URL;
    let result = if cfg!(target_os = "macos") {
        std::process::Command::new("open").arg(url).spawn()
    } else if cfg!(target_os = "windows") {
        std::process::Command::new("cmd")
            .args(["/C", "start", "", url])
            .spawn()
    } else {
        std::process::Command::new("xdg-open").arg(url).spawn()
    };
    match result {
        Ok(_) => Ok(()),
        Err(err) => Err(format!("open {url}: {err}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hi_ai::Message;

    #[test]
    fn empty_breakdown_is_all_free_space() {
        let window = u64::from(crate::DEFAULT_CONTEXT_WINDOW);
        let (cats, info) = context_breakdown(&[], 0, window);
        assert_eq!(cats[3].name, "Free space");
        assert_eq!(cats[3].tokens, window);
        assert!(info.iter().any(|row| row.name == "Tool definitions"));
    }

    #[test]
    fn messages_count_toward_the_bar() {
        let messages = vec![Message::system("you are hi"), Message::user("hello there")];
        let (cats, _) = context_breakdown(&messages, 50, 128_000);
        assert!(cats[0].tokens > 0, "system prompt");
        assert!(cats[1].tokens > 0, "messages");
        assert_eq!(cats[3].tokens, 128_000 - 50);
    }

    #[test]
    fn tab_args_match_grok() {
        assert_eq!(UsageTab::from_arg(""), Some(UsageTab::Limit));
        assert_eq!(UsageTab::from_arg("show"), Some(UsageTab::Limit));
        assert_eq!(UsageTab::from_arg("context"), Some(UsageTab::Context));
        assert_eq!(UsageTab::from_arg("info"), Some(UsageTab::Session));
        assert_eq!(UsageTab::from_arg("manage"), None);
    }
}
