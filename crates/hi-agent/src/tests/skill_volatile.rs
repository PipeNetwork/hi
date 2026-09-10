use super::common::*;

#[test]
fn stack_skill_stays_in_volatile_context_across_turns() {
    let workspace = IsolatedWorkspace::new("stack-skill-repeat");
    std::fs::write(workspace.path("Cargo.toml"), "[workspace]\nmembers = []\n").unwrap();
    let mut cfg = workspace.config();
    cfg.memory.inject_stack_skill = true;
    let agent = agent(vec![], cfg);
    let first = agent.volatile_context_block().unwrap_or_default();
    let second = agent.volatile_context_block().unwrap_or_default();
    assert!(
        first.contains("# Active stack skill"),
        "first turn: {first}"
    );
    assert!(
        second.contains("# Active stack skill"),
        "later turns must keep the pack: {second}"
    );
}
