//! Grouped `/help` text so the default listing stays a front door, not a catalog.

use crate::command::{COMMANDS, CommandSpec};

/// Progressive-disclosure bucket for a slash command.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum HelpSection {
    Core,
    Project,
    Modes,
    Platform,
}

impl HelpSection {
    pub fn title(self) -> &'static str {
        match self {
            Self::Core => "core",
            Self::Project => "project",
            Self::Modes => "modes",
            Self::Platform => "platform",
        }
    }

    /// Parse `/help <section>`. `all` is not a section — it is handled separately.
    pub fn from_arg(arg: &str) -> Option<Self> {
        match arg.trim().to_ascii_lowercase().as_str() {
            "core" => Some(Self::Core),
            "project" | "projects" => Some(Self::Project),
            "mode" | "modes" => Some(Self::Modes),
            "platform" | "power" | "more" => Some(Self::Platform),
            _ => None,
        }
    }
}

/// Canonical everyday commands, in the order `/help` and an empty Ctrl-K list them.
pub const CORE_COMMANDS: &[&str] = &[
    "verify", "undo", "retry", "diff", "sessions", "status", "config", "engine", "compact", "copy",
    "files", "doctor", "clear", "exit", "help",
];

const PROJECT_COMMANDS: &[&str] = &[
    "init",
    "goal",
    "plan",
    "commit",
    "context",
    "learn",
    "skills",
    "skill",
    "window",
    "remember",
    "undo-memory",
    "memory",
    "export",
    "recap",
];

const MODE_COMMANDS: &[&str] = &[
    "fleet", "race", "loop", "watch", "digest", "inbox", "delegate", "local", "team", "workflow",
];

/// Settings aliases plus renamed dual names — parseable, not a primary `/help` row.
pub(crate) fn is_help_alias(name: &str) -> bool {
    matches!(
        name,
        "model"
            | "provider"
            | "login"
            | "logout"
            | "lsp"
            | "theme"
            | "density"
            | "mouse"
            | "dashboard"
    )
}

/// Which help bucket a command belongs in. Aliases return `None`.
pub fn command_section(name: &str) -> Option<HelpSection> {
    if is_help_alias(name) {
        return None;
    }
    if CORE_COMMANDS.contains(&name) {
        return Some(HelpSection::Core);
    }
    if PROJECT_COMMANDS.contains(&name) {
        return Some(HelpSection::Project);
    }
    if MODE_COMMANDS.contains(&name) {
        return Some(HelpSection::Modes);
    }
    Some(HelpSection::Platform)
}

/// Default `/help`: core commands plus pointers to the rest.
pub fn help_text() -> String {
    help_text_for("")
}

/// `/help`, `/help project`, `/help modes`, `/help platform`, `/help all`.
pub fn help_text_for(arg: &str) -> String {
    let arg = arg.trim();
    if arg.is_empty() {
        return render_core();
    }
    if arg.eq_ignore_ascii_case("all") {
        return render_all();
    }
    match HelpSection::from_arg(arg) {
        Some(section) => render_section(section),
        None => {
            format!(
                "unknown help topic '{arg}'. Try /help, /help project, /help modes, /help platform, or /help all.\n"
            )
        }
    }
}

fn spec(name: &str) -> Option<&'static CommandSpec> {
    COMMANDS.iter().find(|c| c.name == name)
}

fn push_spec_row(out: &mut String, spec: &CommandSpec) {
    let left = if spec.args.is_empty() {
        format!("/{}", spec.name)
    } else {
        format!("/{} {}", spec.name, spec.args)
    };
    out.push_str(&format!("  {left:<22} {}\n", spec.help));
}

fn push_named(out: &mut String, names: &[&str]) {
    for name in names {
        if let Some(spec) = spec(name) {
            push_spec_row(out, spec);
        }
    }
}

fn keybindings() -> &'static str {
    "\nkeybindings (TUI):\n  \
     Ctrl-K             command palette (type to search every command)\n  \
     Ctrl-T             toggle reasoning (thinking) collapse\n  \
     Ctrl-D             full-screen diff review (same as Ctrl-G)\n  \
     Ctrl-?             toggle the agent observability panel\n  \
     Ctrl-C             interrupt the running turn; double-press idle to quit\n  \
     Ctrl-R             fuzzy-search input history\n  \
     Ctrl-A / Ctrl-E    move cursor to start / end of the line\n  \
     Ctrl-U             clear the input line\n  \
     Alt-Enter          insert a newline (multi-line prompt)\n  \
     PageUp / PageDown  scroll the transcript\n  \
     Esc                clear input or dismiss panels\n  \
     /quit              quit\n"
}

fn settings_blurb() -> &'static str {
    "\nsettings (also available as bare aliases):\n  \
     /config [key …]   hub for model, provider, auth, reasoning, verify, lsp, ui…\n  \
     /model /provider /login /logout /verify /lsp /delegate\n  \
     /theme /density /mouse   (TUI; also /config ui …)\n\
     aliases: /m /st /cp /redo /revert /new /changes /usage /debug /cfg /set /h /?\n"
}

fn render_core() -> String {
    let mut out = String::from(
        "hi — ask for an outcome; tests decide. /undo takes the last turn back.\n\ncore:\n",
    );
    push_named(&mut out, CORE_COMMANDS);
    out.push_str(
        "\n  /help project       goal, init, commit, …\n  \
         /help modes         fleet, race, watch, local, …\n  \
         /help platform      rsi, mcp, traces, …\n  \
         /help all           every command\n  \
         /tutorial           interactive tour (TUI)\n",
    );
    out.push_str(settings_blurb());
    out.push_str(keybindings());
    out
}

fn render_section(section: HelpSection) -> String {
    let mut out = format!("{}:\n", section.title());
    match section {
        HelpSection::Core => push_named(&mut out, CORE_COMMANDS),
        HelpSection::Project => push_named(&mut out, PROJECT_COMMANDS),
        HelpSection::Modes => push_named(&mut out, MODE_COMMANDS),
        HelpSection::Platform => {
            for spec in COMMANDS {
                if command_section(spec.name) == Some(HelpSection::Platform) {
                    push_spec_row(&mut out, spec);
                }
            }
        }
    }
    out.push_str("\n  /help          core commands\n  /help all      everything\n");
    out
}

fn render_all() -> String {
    let mut out = String::from("core:\n");
    push_named(&mut out, CORE_COMMANDS);
    out.push_str("\nproject:\n");
    push_named(&mut out, PROJECT_COMMANDS);
    out.push_str("\nmodes:\n");
    push_named(&mut out, MODE_COMMANDS);
    out.push_str("\nplatform:\n");
    for spec in COMMANDS {
        if command_section(spec.name) == Some(HelpSection::Platform) {
            push_spec_row(&mut out, spec);
        }
    }
    out.push_str(settings_blurb());
    out.push_str(keybindings());
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_help_is_core_not_a_catalog() {
        let help = help_text();
        assert!(help.contains("core:"));
        assert!(help.contains("/verify"));
        assert!(help.contains("/undo"));
        assert!(help.contains("/help modes"));
        assert!(
            !help.contains("/rsi "),
            "default help must not dump platform commands:\n{help}"
        );
        assert!(
            !help.contains("/moa "),
            "default help must not dump /moa:\n{help}"
        );
        assert!(
            !help.contains("/diff-lab"),
            "default help must not dump /diff-lab:\n{help}"
        );
        assert!(help.contains("settings (also available as bare aliases)"));
        assert!(help.contains("/config [key"));
        assert!(help.contains("/model /provider"));
        assert!(
            !help
                .lines()
                .any(|line| { line.starts_with("  /model ") && line.contains("alias of /config") }),
            "bare /model should not appear as a primary help row"
        );
    }

    #[test]
    fn help_all_includes_platform() {
        let help = help_text_for("all");
        assert!(help.contains("/rsi"));
        assert!(help.contains("platform:"));
        assert!(help.contains("modes:"));
        assert!(help.contains("/fleet"));
    }

    #[test]
    fn help_modes_lists_fleet_not_dashboard_row() {
        let help = help_text_for("modes");
        assert!(help.contains("/fleet"));
        assert!(
            !help
                .lines()
                .any(|line| line.trim_start().starts_with("/dashboard")),
            "dashboard is an alias, not a modes row:\n{help}"
        );
    }

    #[test]
    fn help_project_lists_memory() {
        let help = help_text_for("project");
        assert!(help.contains("/memory"));
        assert!(help.contains("/remember"));
        assert!(help.contains("/window"));
    }

    #[test]
    fn every_command_is_classified() {
        for spec in COMMANDS {
            let section = command_section(spec.name);
            if is_help_alias(spec.name) {
                assert!(section.is_none(), "{} should be an alias", spec.name);
            } else {
                assert!(section.is_some(), "{} needs a help section", spec.name);
            }
        }
    }
}
