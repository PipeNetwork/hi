//! Output seam between the turn loop and a frontend (TUI or tests).

use std::future::Future;
use std::pin::Pin;

use hi_ai::Usage;
use hi_tools::{PlanStep, ToolStatus};

/// A mutation that requires an explicit user decision.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ConfirmationRequest {
    FileEdit { path: String, diff: String },
    ShellMutation { command: String, cwd: String },
}

impl ConfirmationRequest {
    pub fn title(&self) -> &'static str {
        match self {
            Self::FileEdit { .. } => "Confirm file edit",
            Self::ShellMutation { .. } => "Confirm shell mutation",
        }
    }

    /// Conservative classifier for `/permissions auto`.
    pub fn safe_for_auto(&self) -> bool {
        match self {
            Self::FileEdit { path, diff } => file_edit_is_safe(path, diff),
            Self::ShellMutation { .. } => false,
        }
    }

    /// Hard floors Jev must not auto-approve: unsafe files, denylisted shell.
    pub fn blocks_auto_expand(&self) -> bool {
        match self {
            Self::FileEdit { .. } => !self.safe_for_auto(),
            Self::ShellMutation { command, .. } => hi_tools::guard::blocked_op(command).is_some(),
        }
    }

    /// Requests Jev may score in Auto: heuristic-safe files (withhold) or
    /// non-denylisted mutating shell (expand).
    pub fn jev_auto_candidate(&self) -> bool {
        match self {
            Self::FileEdit { .. } => self.safe_for_auto(),
            Self::ShellMutation { command, .. } => hi_tools::guard::blocked_op(command).is_none(),
        }
    }

    pub fn details(&self) -> String {
        match self {
            Self::FileEdit { path, diff } => format!("file: {path}\n\n{diff}"),
            Self::ShellMutation { command, cwd } => {
                format!(
                    "working directory: {cwd}\nwarning: this command is likely to mutate the workspace\n\n$ {command}"
                )
            }
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum ConfirmationResult {
    #[default]
    Approved,
    Rejected,
    Cancelled,
    Unavailable,
}

/// Harness-owned Auto decision for one mutating tool call.
///
/// `Heuristic` keeps [`ConfirmationRequest::safe_for_auto`]. `Approve` skips
/// the overlay only when the request does not [`ConfirmationRequest::blocks_auto_expand`].
/// `Confirm` always shows the overlay.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum AutoHint {
    #[default]
    Heuristic,
    Approve,
    Confirm,
}

pub type ConfirmationFuture<'a> = Pin<Box<dyn Future<Output = ConfirmationResult> + Send + 'a>>;

/// Ask / Auto / Always permission ladder (Shift-Tab in the TUI).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PermissionMode {
    #[default]
    Ask,
    Auto,
    Always,
}

impl PermissionMode {
    pub fn label(self) -> &'static str {
        match self {
            Self::Ask => "ask",
            Self::Auto => "auto",
            Self::Always => "always",
        }
    }

    pub fn from_arg(arg: &str) -> Option<Self> {
        match arg.trim().to_ascii_lowercase().as_str() {
            "ask" | "off" => Some(Self::Ask),
            "auto" => Some(Self::Auto),
            "always" | "yolo" | "always-approve" => Some(Self::Always),
            _ => None,
        }
    }

    pub fn describe(self) -> &'static str {
        match self {
            Self::Ask => "ask — confirm file edits and shell mutations",
            Self::Auto => {
                "auto — safe file edits without asking; Jev may auto-approve reversible shell"
            }
            Self::Always => "always — approve mutations this session (yolo)",
        }
    }

    pub fn toggle_always(self) -> Self {
        if self == Self::Always {
            Self::Ask
        } else {
            Self::Always
        }
    }

    pub fn toggle_auto(self) -> Self {
        if self == Self::Auto {
            Self::Ask
        } else {
            Self::Auto
        }
    }

    pub fn cycle(self) -> Self {
        match self {
            Self::Ask => Self::Auto,
            Self::Auto => Self::Always,
            Self::Always => Self::Ask,
        }
    }

    pub(crate) fn as_u8(self) -> u8 {
        match self {
            Self::Ask => 0,
            Self::Auto => 1,
            Self::Always => 2,
        }
    }

    pub(crate) fn from_u8(value: u8) -> Self {
        match value {
            1 => Self::Auto,
            2 => Self::Always,
            _ => Self::Ask,
        }
    }
}

fn file_edit_is_safe(path: &str, diff: &str) -> bool {
    if matches!(path.trim(), "" | "." | "(unknown)" | "(multiple files)")
        || std::path::Path::new(path)
            .components()
            .any(|component| matches!(component, std::path::Component::ParentDir))
    {
        return false;
    }
    let lower = path.to_ascii_lowercase();
    let secretish = [".env", "credential", "secret", "token", "key.pem"]
        .iter()
        .any(|needle| lower.contains(needle));
    let destructive = diff.lines().filter(|line| line.starts_with('-')).count() > 80;
    !secretish && !destructive && diff.len() <= 32 * 1024
}

/// Events the turn loop emits. Frontends decide how to render them.
pub trait Ui: Send {
    fn assistant_text(&mut self, text: &str);
    fn assistant_reasoning(&mut self, text: &str);
    fn assistant_end(&mut self);
    fn tool_started(&mut self, _name: &str, _arguments: &str) {}
    fn tool_started_id(&mut self, _id: &str, name: &str, arguments: &str) {
        self.tool_started(name, arguments);
    }
    fn tool_stream(&mut self, _name: &str, _line: &str) {}
    fn tool_call(&mut self, name: &str, arguments: &str);
    fn tool_call_id(&mut self, _id: &str, name: &str, arguments: &str) {
        self.tool_call(name, arguments);
    }
    fn tool_result(&mut self, name: &str, result: &str);
    fn tool_result_id(&mut self, _id: &str, name: &str, result: &str, _status: ToolStatus) {
        self.tool_result(name, result);
    }
    fn plan(&mut self, _steps: &[PlanStep]) {}
    fn plan_result_id(
        &mut self,
        _id: &str,
        _name: &str,
        _result: &str,
        _status: ToolStatus,
        steps: &[PlanStep],
    ) {
        self.plan(steps);
    }
    fn status(&mut self, text: &str);
    fn top_status(&mut self, text: &str) {
        self.status(text);
    }
    fn checkpoint_warning(&mut self, text: &str) {
        self.status(text);
    }
    fn usage(
        &mut self,
        _prompt: u64,
        _generated: u64,
        _ctx_used: u64,
        _ctx_window: Option<u32>,
        _estimated: bool,
    ) {
    }
    fn session_usage(&mut self, _usage: Usage) {}
    fn turn_end(&mut self, summary: &str);
    fn turn_error(&mut self, error_kind: &str, message: &str, guidance: &str);
    fn changed_files(&mut self, _files: Vec<String>) {}
    fn confirm(&mut self, _request: ConfirmationRequest) -> ConfirmationFuture<'_> {
        Box::pin(async { ConfirmationResult::Unavailable })
    }
}

/// Collecting UI for tests.
#[derive(Default)]
pub struct TestUi {
    pub texts: Vec<String>,
    pub reasoning: Vec<String>,
    pub tool_calls: Vec<(String, String)>,
    pub tool_results: Vec<(String, String)>,
    pub statuses: Vec<String>,
    pub errors: Vec<(String, String)>,
    pub turn_ends: Vec<String>,
    pub plans: Vec<Vec<PlanStep>>,
    pub confirm: ConfirmationResult,
}

impl Ui for TestUi {
    fn assistant_text(&mut self, text: &str) {
        self.texts.push(text.to_string());
    }
    fn assistant_reasoning(&mut self, text: &str) {
        self.reasoning.push(text.to_string());
    }
    fn assistant_end(&mut self) {}
    fn tool_call(&mut self, name: &str, arguments: &str) {
        self.tool_calls
            .push((name.to_string(), arguments.to_string()));
    }
    fn tool_result(&mut self, name: &str, result: &str) {
        self.tool_results
            .push((name.to_string(), result.to_string()));
    }
    fn plan(&mut self, steps: &[PlanStep]) {
        self.plans.push(steps.to_vec());
    }
    fn status(&mut self, text: &str) {
        self.statuses.push(text.to_string());
    }
    fn turn_end(&mut self, summary: &str) {
        self.turn_ends.push(summary.to_string());
    }
    fn turn_error(&mut self, error_kind: &str, message: &str, _guidance: &str) {
        self.errors
            .push((error_kind.to_string(), message.to_string()));
    }
    fn confirm(&mut self, _request: ConfirmationRequest) -> ConfirmationFuture<'_> {
        let result = self.confirm.clone();
        Box::pin(async move { result })
    }
}

#[cfg(test)]
mod tests {
    use super::{ConfirmationRequest, PermissionMode};

    #[test]
    fn yolo_and_auto_toggle_back_to_ask() {
        assert_eq!(PermissionMode::Ask.toggle_always(), PermissionMode::Always);
        assert_eq!(PermissionMode::Always.toggle_always(), PermissionMode::Ask);
        assert_eq!(PermissionMode::Auto.toggle_always(), PermissionMode::Always);
        assert_eq!(PermissionMode::Ask.toggle_auto(), PermissionMode::Auto);
        assert_eq!(PermissionMode::Auto.toggle_auto(), PermissionMode::Ask);
        assert_eq!(PermissionMode::Ask.cycle(), PermissionMode::Auto);
        assert_eq!(PermissionMode::Auto.cycle(), PermissionMode::Always);
        assert_eq!(PermissionMode::Always.cycle(), PermissionMode::Ask);
        assert_eq!(
            PermissionMode::from_arg("yolo"),
            Some(PermissionMode::Always)
        );
        assert_eq!(PermissionMode::from_arg("off"), Some(PermissionMode::Ask));
    }

    #[test]
    fn secret_path_never_auto_expands() {
        let request = ConfirmationRequest::FileEdit {
            path: "secrets/.env".into(),
            diff: "+TOKEN=1\n".into(),
        };
        assert!(!request.safe_for_auto());
        assert!(request.blocks_auto_expand());
        assert!(!request.jev_auto_candidate());
    }

    #[test]
    fn force_push_never_auto_expands() {
        let request = ConfirmationRequest::ShellMutation {
            command: "git push --force origin main".into(),
            cwd: "/tmp/hi".into(),
        };
        assert!(!request.safe_for_auto());
        assert!(request.blocks_auto_expand());
        assert!(!request.jev_auto_candidate());
    }
}
