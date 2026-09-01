//! Fresh-session injection census: what a blank turn injects before any
//! transcript accumulates. Distinct from live occupancy (`/context` history)
//! and from `/doctor` (setup health).

use hi_ai::{Role, estimate_text_tokens, estimate_tool_schema_tokens};
use hi_tools::{TOOL_SPECS, search_tool_tool_spec, use_tool_tool_spec};

use crate::heuristics::humanize_count;
use crate::skills::learned_skills_context;

struct Row {
    label: &'static str,
    tokens: u64,
    note: Option<String>,
}

impl crate::Agent {
    /// Token estimate of what this session injects aside from turn history:
    /// stable system prompt, guides, skills index, advertised vs full tool
    /// schemas, the two MCP gateway tools, and volatile memory/goal blocks.
    pub fn context_injection_census(&self) -> String {
        let mut rows = Vec::new();

        let system = self
            .messages
            .as_slice()
            .iter()
            .find(|message| message.role == Role::System);
        let system_tokens = system
            .map(|message| estimate_text_tokens(&message.text()))
            .unwrap_or(0);
        rows.push(Row {
            label: "stable system prompt",
            tokens: system_tokens,
            note: Some("identity, git rules, cwd — prefix-cached".into()),
        });

        let guides = self.config.memory.project_context.as_deref().unwrap_or("");
        let skills = learned_skills_context().unwrap_or_default();
        let guide_only = if skills.is_empty() {
            guides.to_string()
        } else {
            guides.replace(&skills, "")
        };
        rows.push(Row {
            label: "HI.md / AGENTS.md",
            tokens: estimate_text_tokens(guide_only.trim()),
            note: None,
        });
        rows.push(Row {
            label: "skills index",
            tokens: estimate_text_tokens(skills.trim()),
            note: Some("names + descriptions only; bodies via /skill".into()),
        });

        let standing = self.config.memory.standing_rules.as_deref().unwrap_or("");
        rows.push(Row {
            label: "me.md standing rules",
            tokens: estimate_text_tokens(standing.trim()),
            note: standing
                .is_empty()
                .then(|| "absent — add ~/.config/hi/me.md".into()),
        });

        let advertised = estimate_tool_schema_tokens(&self.tools);
        let full = estimate_tool_schema_tokens(&TOOL_SPECS);
        rows.push(Row {
            label: "advertised tool schemas",
            tokens: advertised,
            note: Some(format!(
                "{} tools this turn; full catalog ~{} tok",
                self.tools.len(),
                humanize_count(full)
            )),
        });

        let gateway = [search_tool_tool_spec(), use_tool_tool_spec()];
        let gateway_advertised = self
            .tools
            .iter()
            .any(|spec| spec.name == "search_tool" || spec.name == "use_tool");
        rows.push(Row {
            label: "MCP gateway schemas",
            tokens: estimate_tool_schema_tokens(&gateway),
            note: Some(if gateway_advertised {
                "search_tool + use_tool only — not every MCP tool JSON".into()
            } else {
                "not advertised this turn".into()
            }),
        });

        let volatile = self.volatile_context_block().unwrap_or_default();
        rows.push(Row {
            label: "volatile memory / goal",
            tokens: estimate_text_tokens(volatile.trim()),
            note: Some("per-turn user block; not prefix-cached".into()),
        });

        let mut ranked = rows;
        ranked.sort_by_key(|row| std::cmp::Reverse(row.tokens));
        let total: u64 = ranked.iter().map(|row| row.tokens).sum();

        let mut out = String::from("injection:\n");
        for row in &ranked {
            out.push_str(&format!(
                "  {:>6}  {}",
                format!("~{}", humanize_count(row.tokens)),
                row.label
            ));
            if let Some(note) = &row.note {
                out.push_str(&format!("  ({note})"));
            }
            out.push('\n');
        }
        out.push_str(&format!("  TOTAL  ~{} tokens\n", humanize_count(total)));
        if let Some(top) = ranked.first() {
            out.push_str(&format!(
                "  trim: largest offender is {} — shorten that file or /compact if history grew\n",
                top.label
            ));
        }
        out
    }
}
