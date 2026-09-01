use super::*;

/// Canonical capability → side-effect class matrix for interactive tools.
///
/// RSI `hi-tool-host::SideEffect` uses the same vocabulary (None / WorkspaceRead /
/// WorkspaceWrite / Process / Network). Keep these mappings aligned when either
/// catalog changes — see `hi-tool-host` tests for the host-side mirror.
fn expected_side_effect_class(meta: &ToolMetadata) -> &'static str {
    if meta.filesystem_mutating {
        return "workspace_write";
    }
    match meta.capability {
        ToolCapability::Coordination => "none",
        ToolCapability::Repository | ToolCapability::Lsp => "workspace_read",
        ToolCapability::Mutation => "workspace_write",
        ToolCapability::Process | ToolCapability::Background | ToolCapability::Subagent => {
            "process"
        }
        ToolCapability::Mcp => {
            if meta.read_only {
                "none"
            } else {
                "process"
            }
        }
        ToolCapability::Memory => "workspace_read",
        ToolCapability::Skill => "none",
        ToolCapability::Web => {
            // web_download mutates via filesystem_mutating; search/fetch
            // and inject-gated browser_exec are network (page effects).
            "network"
        }
    }
}
#[test]
fn read_only_tools_are_classified() {
    assert!(is_read_only("read"));
    assert!(is_read_only("list"));
    assert!(is_read_only("grep"));
    assert!(is_read_only("diff"));
    assert!(is_read_only("glob"));
    // No filesystem side effects — safe to parallelize and offer in
    // read-only mode.
    assert!(is_read_only("update_plan"));
    assert!(is_read_only("record_decision"));
    assert!(is_read_only("ask_user"));
    assert!(is_read_only("new_context"));
    assert!(is_read_only("research"));
    assert!(is_read_only("research_read"));
    assert!(is_read_only("bash_output"));
    // Mutating / effecting tools are not safe to run concurrently.
    assert!(!is_read_only("write"));
    assert!(!is_read_only("edit"));
    assert!(!is_read_only("multi_edit"));
    assert!(!is_read_only("apply_patch"));
    assert!(!is_read_only("bash"));
    assert!(!is_read_only("bash_kill"));
    assert!(!is_read_only("browser_exec"));
}
#[test]
fn filesystem_mutating_tools_are_classified() {
    // Only tools that write to the working tree.
    assert!(is_filesystem_mutating("write"));
    assert!(is_filesystem_mutating("edit"));
    assert!(is_filesystem_mutating("multi_edit"));
    assert!(is_filesystem_mutating("apply_patch"));
    assert!(is_filesystem_mutating("memory_update"));
    assert!(is_filesystem_mutating("memory_forget"));
    // Everything else — including non-read-only tools like bash — does not
    // directly mutate via the tool layer (bash runs alone; bash_kill stops
    // a process; update_plan/record_decision are in-memory only).
    assert!(!is_filesystem_mutating("bash"));
    assert!(!is_filesystem_mutating("bash_kill"));
    assert!(!is_filesystem_mutating("bash_output"));
    assert!(!is_filesystem_mutating("update_plan"));
    assert!(!is_filesystem_mutating("record_decision"));
    assert!(!is_filesystem_mutating("read"));
    assert!(!is_filesystem_mutating("diff"));
}
#[test]
fn metadata_catalog_covers_every_schema_once() {
    let mut names = std::collections::BTreeSet::new();
    for metadata in TOOL_CATALOG {
        assert!(names.insert(metadata.name), "duplicate {}", metadata.name);
    }
    for spec in TOOL_SPECS.iter() {
        assert!(tool_metadata(&spec.name).is_some(), "missing {}", spec.name);
    }
    for spec in MINIMAL_TOOL_SPECS.iter() {
        assert!(
            tool_metadata(&spec.name).is_some_and(|metadata| metadata.minimal),
            "{} is not marked minimal",
            spec.name
        );
    }
    assert!(is_known_tool("explore"));
    assert!(is_known_tool("delegate"));
    assert!(is_known_tool("ask_user"));
    assert!(is_known_tool("new_context"));
    assert!(is_known_tool("research"));
    assert!(is_known_tool("research_read"));
    assert!(is_known_tool("browser_exec"));
    assert!(!is_known_tool("hallucinated_tool"));
}

fn spec(name: &str) -> &'static ToolSpec {
    TOOL_SPECS
        .iter()
        .find(|spec| spec.name == name)
        .unwrap_or_else(|| panic!("missing {name}"))
}

#[test]
fn web_tools_split_known_url_from_search_and_curl() {
    let bash = spec("bash");
    assert!(
        bash.description.contains("web_fetch") && bash.description.contains("curl"),
        "bash must defer given URLs to web_fetch: {}",
        bash.description
    );
    let search = spec("web_search");
    assert!(
        search.description.contains("web_fetch"),
        "web_search must defer exact URLs to web_fetch: {}",
        search.description
    );
    let fetch = spec("web_fetch");
    assert!(
        fetch.description.contains("curl") || fetch.description.contains("wget"),
        "web_fetch should beat bash curl: {}",
        fetch.description
    );
}

/// ADR 002: every catalog row carries an admission gate + human alternative.
/// There is no `Legacy` variant — the current set is fully classified, so
/// new rows must use Structure/Safety/Reliability (or stay bash/skill).
#[test]
fn every_tool_has_admission_and_alternative() {
    for meta in TOOL_CATALOG {
        assert!(
            !meta.alternative.trim().is_empty(),
            "`{}` needs a non-empty human-protocol alternative \
             (docs/adr/002-tool-admission.md)",
            meta.name
        );
        // Exhaustiveness over the (closed) enum: a future variant added
        // without a meaningful gate is caught here.
        assert!(
            matches!(
                meta.admission,
                ToolAdmission::Structure | ToolAdmission::Safety | ToolAdmission::Reliability
            ),
            "`{}` admission must be Structure, Safety, or Reliability \
             (see docs/adr/002-tool-admission.md)",
            meta.name
        );
    }
    // Spot-check the coding floor so a mistaken reclassify is loud.
    assert_eq!(
        tool_metadata("bash").map(|m| m.admission),
        Some(ToolAdmission::Safety)
    );
    assert_eq!(
        tool_metadata("edit").map(|m| m.admission),
        Some(ToolAdmission::Reliability)
    );
    assert_eq!(
        tool_metadata("use_tool").map(|m| m.admission),
        Some(ToolAdmission::Structure)
    );
}

/// ADR 002: new tools default to inject/capability-gated advertisement,
/// not unconditional membership in the always-on global `TOOL_SPECS`. The
/// global set is therefore a closed allowlist — adding a name to
/// `build_tool_specs` without listing it here fails the test, forcing an
/// explicit admission decision (and ADR update) for any promotion to
/// global.
#[test]
fn global_tool_specs_is_a_closed_allowlist() {
    // The names that `build_tool_specs` unconditionally advertises. Every
    // other catalog row is inject-only (explore/delegate/task family,
    // use_tool/search_tool, memory_search/get/update/forget, skill).
    const ALLOWED_GLOBAL: &[&str] = &[
        "update_plan",
        "record_decision",
        "block_step",
        "read",
        "write",
        "edit",
        "multi_edit",
        "bash",
        "bash_output",
        "bash_kill",
        "list",
        "diff",
        "grep",
        "glob",
        "repo_map",
        "find_symbol",
        "apply_patch",
        "diagnostics",
        "definition",
        "references",
        "hover",
        "web_search",
        "web_fetch",
        "web_download",
    ];
    let allowed: std::collections::BTreeSet<_> = ALLOWED_GLOBAL.iter().copied().collect();
    for spec in TOOL_SPECS.iter() {
        assert!(
            allowed.contains(spec.name.as_str()),
            "`{}` is in the global TOOL_SPECS but not on the ADR-002 allowlist; \
             either keep it inject/capability-gated or add it here with an \
             admission note (docs/adr/002-tool-admission.md)",
            spec.name
        );
    }
    // No allowlist entry may be silently dropped from the global set.
    let global: std::collections::BTreeSet<_> =
        TOOL_SPECS.iter().map(|s| s.name.as_str()).collect();
    for name in allowed {
        assert!(
            global.contains(name),
            "`{}` is on the allowlist but missing from TOOL_SPECS — update \
             build_tool_specs or remove it from ALLOWED_GLOBAL",
            name
        );
    }
}
#[test]
fn target_path_extracts_path_field() {
    assert_eq!(
        target_path("read", r#"{"path":"src/a.rs"}"#),
        Some("src/a.rs".into())
    );
    assert_eq!(
        target_path("write", r#"{"path":"b.rs","content":"x"}"#),
        Some("b.rs".into())
    );
    // list's path is optional → None when absent.
    assert_eq!(target_path("list", r#"{}"#), None);
    assert_eq!(target_path("list", r#"{"path":"sub"}"#), Some("sub".into()));
    // bash has no path → None (the safe-fallback case for dep inference).
    assert_eq!(target_path("bash", r#"{"command":"echo hi"}"#), None);
    // Malformed JSON → None (tolerant).
    assert_eq!(target_path("read", "not json"), None);
    // `read` with `paths`: a one-element array yields that path.
    assert_eq!(
        target_path("read", r#"{"paths":["src/a.rs"]}"#),
        Some("src/a.rs".into())
    );
    // A multi-element array has no single target → None.
    assert_eq!(
        target_path("read", r#"{"paths":["src/a.rs","src/b.rs"]}"#),
        None
    );
    assert_eq!(
        target_paths("read", r#"{"paths":["src/a.rs","src/b.rs"]}"#),
        vec!["src/a.rs".to_string(), "src/b.rs".to_string()]
    );
    // apply_patch: a single file directive's path is extracted.
    let patch =
        r#"{"patch":"*** Begin Patch\n*** Update File: src/a.rs\n-old\n+new\n*** End Patch"}"#;
    assert_eq!(target_path("apply_patch", patch), Some("src/a.rs".into()));
    let add_patch = r#"{"patch":"*** Begin Patch\n*** Add File: new.txt\nhello\n*** End Patch"}"#;
    assert_eq!(
        target_path("apply_patch", add_patch),
        Some("new.txt".into())
    );
    let delete_patch = r#"{"patch":"*** Begin Patch\n*** Delete File: old.txt\n*** End Patch"}"#;
    assert_eq!(
        target_path("apply_patch", delete_patch),
        Some("old.txt".into())
    );
    // Multi-file patches have no single target path. Returning None makes
    // dependency inference serialize later reads conservatively.
    let multi_patch = r#"{"patch":"*** Begin Patch\n*** Update File: src/a.rs\n-old\n+new\n*** Update File: src/b.rs\n-old\n+new\n*** End Patch"}"#;
    assert_eq!(target_path("apply_patch", multi_patch), None);
    // No file directives → None.
    assert_eq!(
        target_path(
            "apply_patch",
            r#"{"patch":"*** Begin Patch\n*** End Patch"}"#
        ),
        None
    );
}

#[test]
fn read_schema_accepts_single_or_multi_path_calls() {
    let read = TOOL_SPECS
        .iter()
        .find(|spec| spec.name == "read")
        .expect("read tool schema");
    assert!(
        hi_ai::validate_client_tool_call(
            "read-single",
            "read",
            r#"{"path":"src/lib.rs"}"#,
            std::slice::from_ref(read),
        )
        .is_ok()
    );
    assert!(
        hi_ai::validate_client_tool_call(
            "read-multi",
            "read",
            r#"{"paths":["src/lib.rs","src/main.rs"],"limit":200}"#,
            std::slice::from_ref(read),
        )
        .is_ok()
    );
    assert!(
        hi_ai::validate_client_tool_call(
            "read-both",
            "read",
            r#"{"path":"src/lib.rs","paths":["src/main.rs"]}"#,
            std::slice::from_ref(read),
        )
        .is_err()
    );
}
#[test]
fn minimal_tool_specs_is_a_lean_subset() {
    let full: Vec<&str> = TOOL_SPECS.iter().map(|s| s.name.as_str()).collect();
    let minimal: Vec<&str> = MINIMAL_TOOL_SPECS.iter().map(|s| s.name.as_str()).collect();
    assert!(minimal.len() < full.len());
    // Every minimal tool exists in the full set, in the same order.
    for name in &minimal {
        assert!(full.contains(name), "{name} missing from full specs");
    }
    // The essentials a small coding agent needs are present.
    for essential in [
        "read",
        "list",
        "grep",
        "repo_map",
        "find_symbol",
        "bash",
        "bash_output",
        "bash_kill",
        "write",
        "edit",
    ] {
        assert!(
            minimal.contains(&essential),
            "{essential} missing from minimal"
        );
    }
}
#[test]
fn capability_matrix_covers_every_catalog_entry() {
    assert!(!TOOL_CATALOG.is_empty());
    let mut names = std::collections::BTreeSet::new();
    for meta in TOOL_CATALOG {
        assert!(names.insert(meta.name), "duplicate tool {}", meta.name);
        let side = expected_side_effect_class(meta);
        // Invariants tying flags to side-effect class.
        match side {
            "none" => {
                assert!(meta.read_only, "{} none must be read_only", meta.name);
                assert!(!meta.filesystem_mutating);
            }
            "workspace_read" => {
                assert!(
                    meta.read_only,
                    "{} workspace_read must be read_only",
                    meta.name
                );
                assert!(!meta.filesystem_mutating);
            }
            "workspace_write" => {
                assert!(
                    meta.filesystem_mutating
                        || matches!(
                            meta.capability,
                            ToolCapability::Mutation | ToolCapability::Web
                        ),
                    "{} workspace_write should mutate fs or be Mutation/Web",
                    meta.name
                );
                assert!(
                    !meta.read_only,
                    "{} workspace_write must not be read_only",
                    meta.name
                );
            }
            "process" => {
                assert!(
                    matches!(
                        meta.capability,
                        ToolCapability::Process
                            | ToolCapability::Background
                            | ToolCapability::Subagent
                            | ToolCapability::Mcp
                    ),
                    "{} process class capability",
                    meta.name
                );
            }
            "network" => {
                assert!(
                    matches!(meta.capability, ToolCapability::Web),
                    "{} network class capability",
                    meta.name
                );
                assert!(
                    meta.read_only || meta.name == "browser_exec",
                    "{} network class is read_only (except browser_exec)",
                    meta.name
                );
            }
            other => panic!("unknown side effect class {other}"),
        }
        // Classifier helpers stay consistent with catalog flags.
        assert_eq!(is_read_only(meta.name), meta.read_only);
        assert_eq!(is_filesystem_mutating(meta.name), meta.filesystem_mutating);
        assert_eq!(
            is_coordination(meta.name),
            meta.capability == ToolCapability::Coordination
        );
        assert!(is_known_tool(meta.name));
    }
}
#[test]
fn capability_matrix_known_tool_side_effects() {
    // Explicit pins so a casual catalog edit fails loudly.
    let pins = [
        ("update_plan", "none"),
        ("record_decision", "none"),
        // Records goal bookkeeping only; touches no file and runs nothing.
        ("block_step", "none"),
        ("ask_user", "none"),
        ("new_context", "none"),
        ("read", "workspace_read"),
        ("list", "workspace_read"),
        ("grep", "workspace_read"),
        ("glob", "workspace_read"),
        ("repo_map", "workspace_read"),
        ("find_symbol", "workspace_read"),
        ("diff", "workspace_read"),
        ("diagnostics", "workspace_read"),
        ("definition", "workspace_read"),
        ("references", "workspace_read"),
        ("hover", "workspace_read"),
        ("write", "workspace_write"),
        ("edit", "workspace_write"),
        ("multi_edit", "workspace_write"),
        ("apply_patch", "workspace_write"),
        ("web_download", "workspace_write"),
        ("bash", "process"),
        ("bash_output", "process"),
        ("bash_kill", "process"),
        ("explore", "process"),
        ("delegate", "process"),
        ("task", "process"),
        ("get_task_output", "process"),
        ("wait_tasks", "process"),
        ("kill_task", "process"),
        ("web_search", "network"),
        ("web_fetch", "network"),
        ("research", "network"),
        ("research_read", "network"),
        ("search_tool", "none"),
        ("use_tool", "process"),
        ("memory_search", "workspace_read"),
        ("memory_get", "workspace_read"),
        ("memory_update", "workspace_write"),
        ("memory_forget", "workspace_write"),
        ("skill", "none"),
        ("browser_exec", "network"),
    ];
    for (name, want) in pins {
        let meta = tool_metadata(name).unwrap_or_else(|| panic!("missing {name}"));
        assert_eq!(
            expected_side_effect_class(meta),
            want,
            "{name} side-effect class drifted"
        );
    }
    // Every catalog entry is pinned (no silent additions).
    // New tools also need an admission note per docs/adr/002-tool-admission.md
    // (human alternative considered; structure/safety/reliability gate;
    // global vs inject vs protected; then spec + TOOL_CATALOG + dispatch).
    let pinned: std::collections::BTreeSet<_> = pins.iter().map(|(n, _)| *n).collect();
    for meta in TOOL_CATALOG {
        assert!(
            pinned.contains(meta.name),
            "add an explicit side-effect pin for new tool `{}` \
             (see docs/adr/002-tool-admission.md — prefer bash/skill unless \
             structure, safety, or reliability requires a first-class tool)",
            meta.name
        );
    }
}
