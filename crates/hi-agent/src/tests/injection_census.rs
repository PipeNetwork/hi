use super::common::{config, scripted_agent};

#[test]
fn census_reports_injection_total_and_gateway_note() {
    let (agent, _requests) = scripted_agent(vec![], config());
    let text = agent.context_injection_census();
    assert!(text.contains("injection:"), "{text}");
    assert!(text.contains("TOTAL"), "{text}");
    assert!(text.contains("MCP gateway schemas"), "{text}");
    assert!(
        text.contains("search_tool")
            || text.contains("not every MCP")
            || text.contains("not advertised"),
        "{text}"
    );
    assert!(text.contains("stable system prompt"), "{text}");
}

#[test]
fn census_counts_project_guides() {
    let mut cfg = config();
    cfg.memory.project_context = Some("# Project context (from HI.md)\nUse rustfmt.\n".into());
    let (agent, _requests) = scripted_agent(vec![], cfg);
    let text = agent.context_injection_census();
    assert!(text.contains("HI.md / AGENTS.md"), "{text}");
    assert!(text.contains("injection:"), "{text}");
}

#[test]
fn context_breakdown_includes_injection_census() {
    let (agent, _requests) = scripted_agent(vec![], config());
    let text = agent.context_breakdown();
    assert!(text.contains("injection:"), "{text}");
    assert!(text.contains("TOTAL"), "{text}");
}
