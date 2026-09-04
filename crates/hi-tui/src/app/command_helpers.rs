//! Pure parsing and presentation helpers used by the command dispatcher.

use ratatui::text::Line;

pub(super) fn toggle_arg(current: bool, arg: &str) -> bool {
    match arg.trim() {
        "on" | "enable" | "yes" | "true" => true,
        "off" | "disable" | "no" | "false" => false,
        "status" => current,
        _ => !current,
    }
}

pub(super) fn on_off(value: bool) -> &'static str {
    if value { "on" } else { "off" }
}

pub(super) fn tui_mcp_agent_check(
    table: Option<&str>,
    mcp_url: Option<&str>,
    api_key: &str,
) -> hi_agent::DoctorCheck {
    if let Some(table) = table {
        let first_party = table.lines().any(|line| {
            let mut parts = line.split_whitespace();
            parts.next() == Some("pipe") && parts.next() == Some("pipe")
        });
        if first_party {
            return hi_agent::DoctorCheck::pass(
                "mcp agent attach",
                "pipe (allowlist: pipe.models.list, pipe.models.health)",
            );
        }
        let workspace_pipe = table.lines().any(|line| {
            line.split_whitespace()
                .next()
                .is_some_and(|name| name == "pipe")
        });
        if workspace_pipe {
            return hi_agent::DoctorCheck::pass(
                "mcp agent attach",
                "skipped: workspace already defines a server named 'pipe'",
            );
        }
    }
    if mcp_url.map(str::trim).is_none_or(|url| url.is_empty()) {
        return hi_agent::DoctorCheck::pass(
            "mcp agent attach",
            "skipped: no mcp_url for this provider",
        );
    }
    if api_key.trim().is_empty() {
        return hi_agent::DoctorCheck::fail(
            "mcp agent attach",
            "mcp_url is set but no API key",
            "set a project API key, or disable with [mcp.pipe] enabled = false",
        );
    }
    hi_agent::DoctorCheck::pass(
        "mcp agent attach",
        "off ([mcp.pipe] enabled = false, or attach did not register)",
    )
}

pub(super) fn parse_race_arg(arg: &str) -> (String, bool) {
    let mut judge_model = false;
    let mut rest = Vec::new();
    let mut tokens = arg.split_whitespace().peekable();
    while let Some(token) = tokens.next() {
        if token == "--judge" {
            if tokens
                .next()
                .is_some_and(|value| value.eq_ignore_ascii_case("model"))
            {
                judge_model = true;
            }
        } else {
            rest.push(token);
        }
    }
    (rest.join(" "), judge_model)
}

pub(super) fn race_setup_lines(defaults: &crate::RaceDefaults) -> Line<'static> {
    let targets = if defaults.targets.is_empty() {
        "<none configured>".to_string()
    } else {
        defaults
            .targets
            .iter()
            .map(|target| format!("{}={}:{}", target.name, target.profile, target.model))
            .collect::<Vec<_>>()
            .join(", ")
    };
    Line::raw(format!(
        "race targets: {targets} · max candidates {} · fuzz {} · edit .hi/config.toml [race]",
        defaults.max_candidates,
        defaults.fuzz.as_ref().map(|_| "on").unwrap_or("off")
    ))
}
