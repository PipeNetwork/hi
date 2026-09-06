use super::*;

fn roots(project: PathBuf, global: PathBuf) -> SkillRoots {
    SkillRoots {
        project,
        agents: PathBuf::new(),
        global,
    }
}

fn unique_dir(label: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "hi-skills-{label}-{}-{}",
        std::process::id(),
        std::thread::current().name().unwrap_or("anon")
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn disk_skills(roots: &SkillRoots) -> Vec<LearnedSkill> {
    list_skills_in(roots)
        .into_iter()
        .filter(|s| !is_builtin_skill_path(&s.path))
        .collect()
}

fn write_skill(root: &Path, slug: &str, name: &str, description: &str, scope: &str, body: &str) {
    let dir = root.join(slug);
    fs::create_dir_all(&dir).unwrap();
    fs::write(
            dir.join("SKILL.md"),
            format!(
                "---\nname: {name}\ndescription: {description}\nscope: {scope}\n---\n\n# {name}\n\n{body}\n"
            ),
        )
        .unwrap();
}

#[test]
fn scanner_prefers_project_over_global_duplicates() {
    let project = unique_dir("project");
    let global = unique_dir("global");
    write_skill(
        &global,
        "release",
        "release-flow",
        "global flow",
        "global",
        "global body",
    );
    write_skill(
        &global,
        "triage",
        "triage-flow",
        "global triage",
        "global",
        "triage body",
    );
    write_skill(
        &project,
        "release",
        "release-flow",
        "project flow",
        "project",
        "project body",
    );
    let roots = roots(project, global);
    let skills = disk_skills(&roots);
    assert_eq!(skills.len(), 2);
    assert_eq!(skills[0].name, "release-flow");
    assert_eq!(skills[0].description, "project flow");
    assert_eq!(skills[1].name, "triage-flow");
    assert_eq!(skills[1].description, "global triage");
    let skill = read_skill_in(&roots, "release-flow").unwrap();
    assert!(skill.content.contains("project body"));
}

#[test]
fn malformed_frontmatter_is_skipped_without_panic() {
    let project = unique_dir("malformed");
    fs::create_dir_all(project.join("bad")).unwrap();
    fs::write(
        project.join("bad").join("SKILL.md"),
        "# Missing frontmatter\n",
    )
    .unwrap();
    write_skill(&project, "good", "good-skill", "works", "project", "body");
    let roots = SkillRoots {
        agents: PathBuf::new(),
        project,
        global: unique_dir("malformed-global"),
    };
    let skills = disk_skills(&roots);
    assert_eq!(skills.len(), 1);
    assert_eq!(skills[0].name, "good-skill");
}

#[test]
fn learned_context_is_compact_index_only() {
    let project = unique_dir("context");
    write_skill(
        &project,
        "debug",
        "debug-flow",
        "Debug the thing.",
        "project",
        "SECRET FULL BODY",
    );
    let roots = SkillRoots {
        agents: PathBuf::new(),
        project,
        global: unique_dir("context-global"),
    };
    let skills = list_skills_in(&roots);
    let rendered = learned_skills_context_from(&skills).unwrap();
    assert!(rendered.contains("debug-flow"));
    assert!(rendered.contains("Debug the thing."));
    assert!(!rendered.contains("SECRET FULL BODY"));
}

#[test]
fn learned_context_clips_huge_descriptions_and_caps_the_index() {
    let mut skills = Vec::new();
    for i in 0..40 {
        skills.push(LearnedSkill {
            name: format!("skill-{i:02}"),
            description: "D".repeat(2_000),
            scope: "project".into(),
            path: PathBuf::from(format!("/tmp/skill-{i}")),
            disable_model_invocation: false,
        });
    }
    let rendered = learned_skills_context_from(&skills).unwrap();
    assert!(
        rendered.chars().count() <= MAX_SKILLS_CONTEXT_CHARS + 80,
        "skill index must stay bounded: {}",
        rendered.chars().count()
    );
    assert!(
        rendered.contains("truncated") || rendered.contains("omitted"),
        "{rendered}"
    );
    assert!(!rendered.contains(&"D".repeat(500)), "{rendered}");
}

#[test]
fn skill_use_prompt_clips_a_huge_body() {
    let prompt = build_skill_use_prompt("bomb", &"X".repeat(20_000));
    assert!(
        prompt.chars().count()
            <= build_skill_use_prompt("bomb", "").chars().count() + MAX_SKILL_USE_PROMPT_CHARS,
        "{}",
        prompt.chars().count()
    );
    assert!(prompt.contains("bomb"));
}

#[test]
fn builtin_stack_packs_are_listed_and_readable() {
    let roots = SkillRoots {
        agents: PathBuf::new(),
        project: unique_dir("builtin-project-empty"),
        global: unique_dir("builtin-global-empty"),
    };
    let skills = list_skills_in(&roots);
    let names: Vec<_> = skills.iter().map(|s| s.name.as_str()).collect();
    assert!(
        names.contains(&"rust-workspace"),
        "missing rust-workspace in {names:?}"
    );
    assert!(
        names.contains(&"pytest-package"),
        "missing pytest-package in {names:?}"
    );
    assert!(
        names.contains(&"ts-monorepo"),
        "missing ts-monorepo in {names:?}"
    );
    assert!(
        names.contains(&"code-review"),
        "missing code-review in {names:?}"
    );
    assert!(
        names.contains(&"secret-scan"),
        "missing secret-scan in {names:?}"
    );
    assert!(
        names.contains(&"dep-audit"),
        "missing dep-audit in {names:?}"
    );
    for skill in &skills {
        if matches!(
            skill.name.as_str(),
            "rust-workspace"
                | "pytest-package"
                | "ts-monorepo"
                | "code-review"
                | "secret-scan"
                | "dep-audit"
        ) {
            assert_eq!(skill.scope, "global");
            assert!(is_builtin_skill_path(&skill.path), "{:?}", skill.path);
        }
    }
    let body = read_skill_in(&roots, "rust-workspace").unwrap();
    assert!(body.content.contains("cargo test"));
    assert!(body.content.contains("manifest-path"));
    let py = read_skill_in(&roots, "pytest-package").unwrap();
    assert!(py.content.contains("pytest -q"));
    let ts = read_skill_in(&roots, "ts-monorepo").unwrap();
    assert!(ts.content.contains("npm --prefix"));
    let review = read_skill_in(&roots, "code-review").unwrap();
    assert!(review.content.contains("## Gate"));
    assert!(review.content.contains("introduced by this change"));
    assert!(review.content.chars().count() <= MAX_ACTIVE_STACK_SKILL_CHARS);
    let secrets = read_skill_in(&roots, "secret-scan").unwrap();
    assert!(secrets.content.contains("/permissions"));
    assert!(secrets.content.contains("Never print") || secrets.content.contains("Do not"));
    let audit = read_skill_in(&roots, "dep-audit").unwrap();
    assert!(audit.content.contains("/permissions"));
    assert!(audit.content.contains("cargo audit"));
}

#[test]
fn project_skill_shadows_builtin_pack() {
    let project = unique_dir("shadow-project");
    write_skill(
        &project,
        "rust-workspace",
        "rust-workspace",
        "Project override.",
        "project",
        "CUSTOM BODY",
    );
    let roots = SkillRoots {
        agents: PathBuf::new(),
        project,
        global: unique_dir("shadow-global"),
    };
    let skills = list_skills_in(&roots);
    let rust = skills
        .iter()
        .find(|s| s.name == "rust-workspace")
        .expect("rust-workspace present");
    assert_eq!(rust.scope, "project");
    assert!(!is_builtin_skill_path(&rust.path));
    let content = read_skill_in(&roots, "rust-workspace").unwrap();
    assert!(content.content.contains("CUSTOM BODY"));
    assert!(!content.content.contains("manifest-path"));
}

#[test]
fn learn_prompt_empty_defaults_to_current_conversation() {
    let prompt = build_learn_prompt("");
    assert!(prompt.contains("workflow we just went through"));
    assert!(prompt.contains("exactly one file named SKILL.md"));
}

#[test]
fn skill_use_prompt_includes_full_content() {
    let prompt = build_skill_use_prompt("release-flow", "# Release\n\nSteps");
    assert!(prompt.contains("release-flow"));
    assert!(prompt.contains("# Release"));
    assert!(prompt.contains("Steps"));
    assert!(prompt.contains(SKILL_APPLICATION_GUIDANCE));
}

#[test]
fn write_skill_round_trips_and_dedups() {
    let roots = SkillRoots {
        agents: PathBuf::new(),
        project: unique_dir("write-project"),
        global: unique_dir("write-global"),
    };
    // `super::write_skill` is the real writer (the test helper above shadows the name locally).
    let path = super::write_skill(
        &roots,
        "project",
        "Retry Flaky Test",
        "Re-run a flaky test to confirm.",
        "# Retry\n\nsteps here",
    )
    .unwrap();
    assert!(path.is_some());
    let skills = disk_skills(&roots);
    assert_eq!(skills.len(), 1);
    assert_eq!(skills[0].name, "Retry Flaky Test");
    assert_eq!(skills[0].scope, "project");
    assert_eq!(skills[0].description, "Re-run a flaky test to confirm.");
    let content = read_skill_in(&roots, "retry flaky test").unwrap();
    assert!(content.content.contains("steps here"));
    // Same normalized name (different casing) is a de-dup no-op.
    let again = super::write_skill(&roots, "project", "retry flaky test", "dup", "x").unwrap();
    assert!(again.is_none());
    assert_eq!(disk_skills(&roots).len(), 1);
}

#[test]
fn write_skill_oversize_is_rejected() {
    let roots = SkillRoots {
        agents: PathBuf::new(),
        project: unique_dir("oversize-project"),
        global: unique_dir("oversize-global"),
    };
    let huge = "x".repeat(MAX_SKILL_BYTES + 1);
    assert!(super::write_skill(&roots, "project", "big", "big", &huge).is_err());
    assert!(disk_skills(&roots).is_empty());
}

#[cfg(unix)]
#[test]
fn write_skill_refuses_symlinked_project_paths_and_targets() {
    use std::os::unix::fs::symlink;

    let workspace = unique_dir("write-symlink-workspace");
    let escape = unique_dir("write-symlink-escape");
    symlink(&escape, workspace.join(".hi")).unwrap();
    let roots = SkillRoots {
        agents: PathBuf::new(),
        project: workspace.join(".hi/skills"),
        global: unique_dir("write-symlink-global"),
    };
    assert!(super::write_skill(&roots, "project", "Escaped", "desc", "body").is_err());
    assert!(!escape.join("skills/escaped/SKILL.md").exists());

    fs::remove_file(workspace.join(".hi")).unwrap();
    let skill_dir = workspace.join(".hi/skills/planted");
    fs::create_dir_all(&skill_dir).unwrap();
    let victim = workspace.join("victim.txt");
    fs::write(&victim, "keep me").unwrap();
    symlink(&victim, skill_dir.join("SKILL.md")).unwrap();
    assert!(super::write_skill(&roots, "project", "Planted", "desc", "body").is_err());
    assert_eq!(fs::read_to_string(victim).unwrap(), "keep me");
}

#[test]
fn matching_stack_skill_prefers_cargo_then_js_then_python() {
    let roots = SkillRoots {
        agents: PathBuf::new(),
        project: unique_dir("match-project-empty"),
        global: unique_dir("match-global-empty"),
    };
    let cargo = unique_dir("match-cargo");
    fs::write(cargo.join("Cargo.toml"), "[package]\nname = \"x\"\n").unwrap();
    assert_eq!(matching_stack_skill_slug(&cargo), Some("rust-workspace"));
    let skill = matching_stack_skill_in(&cargo, &roots).unwrap();
    assert_eq!(skill.skill.name, "rust-workspace");
    assert!(skill.content.contains("manifest-path"));

    let js = unique_dir("match-js");
    fs::write(js.join("package.json"), "{}\n").unwrap();
    assert_eq!(matching_stack_skill_slug(&js), Some("ts-monorepo"));

    let py = unique_dir("match-py");
    fs::write(py.join("pyproject.toml"), "[project]\nname = \"x\"\n").unwrap();
    assert_eq!(matching_stack_skill_slug(&py), Some("pytest-package"));

    let empty = unique_dir("match-empty");
    assert!(matching_stack_skill_slug(&empty).is_none());
    assert!(active_stack_skill_section_in(&empty, &roots).is_none());

    let mixed = unique_dir("match-mixed");
    fs::write(mixed.join("Cargo.toml"), "[workspace]\nmembers = []\n").unwrap();
    fs::write(mixed.join("package.json"), "{}\n").unwrap();
    assert_eq!(matching_stack_skill_slug(&mixed), Some("rust-workspace"));

    let section = active_stack_skill_section_in(&cargo, &roots).unwrap();
    assert!(section.contains("# Active stack skill (`rust-workspace`)"));
    assert!(section.contains(SKILL_APPLICATION_GUIDANCE));
}

#[test]
fn project_skill_shadows_auto_injected_stack_pack() {
    let project = unique_dir("match-shadow-project");
    write_skill(
        &project,
        "rust-workspace",
        "rust-workspace",
        "Project override.",
        "project",
        "CUSTOM STACK BODY",
    );
    let roots = SkillRoots {
        agents: PathBuf::new(),
        project,
        global: unique_dir("match-shadow-global"),
    };
    let cargo = unique_dir("match-shadow-ws");
    fs::write(cargo.join("Cargo.toml"), "[package]\nname = \"x\"\n").unwrap();
    let content = matching_stack_skill_in(&cargo, &roots).unwrap();
    assert!(content.content.contains("CUSTOM STACK BODY"));
    assert!(!content.content.contains("manifest-path"));
}

#[test]
fn review_skill_injects_without_repo_markers() {
    let roots = SkillRoots {
        agents: PathBuf::new(),
        project: unique_dir("review-empty-project"),
        global: unique_dir("review-empty-global"),
    };
    let empty = unique_dir("review-empty-ws");
    assert!(matching_stack_skill_slug(&empty).is_none());
    let section = active_review_skill_section_in(&roots).unwrap();
    assert!(section.contains("# Active review skill (`code-review`)"));
    assert!(section.contains("Do not follow a coding stack pack"));
    assert!(section.contains(SKILL_APPLICATION_GUIDANCE));
    let gate = review_gate_appendix_in(&roots);
    assert!(
        gate.contains("introduced by this change"),
        "gate excerpt: {gate}"
    );
    assert!(
        !gate.contains("merge-base"),
        "Gate must not include Procedure: {gate}"
    );
    assert!(gate.chars().count() <= MAX_REVIEW_GATE_CHARS);
    let loop_excerpt = review_loop_skill_excerpt_in(&roots);
    assert!(loop_excerpt.contains("merge-base") || loop_excerpt.contains("gh pr diff"));
    assert!(loop_excerpt.contains("[P0]"));
    assert!(!loop_excerpt.contains("When uncertain, APPROVE"));
}

#[test]
fn project_skill_shadows_auto_injected_review_pack() {
    let project = unique_dir("review-shadow-project");
    write_skill(
        &project,
        "code-review",
        "code-review",
        "Project review override.",
        "project",
        "## Gate\nPROJECT GATE BODY introduced by this change.\n\n## Procedure\nmerge-base only here.\n",
    );
    let roots = SkillRoots {
        agents: PathBuf::new(),
        project,
        global: unique_dir("review-shadow-global"),
    };
    let content = read_skill_in(&roots, "code-review").unwrap();
    assert!(content.content.contains("PROJECT GATE BODY"));
    let gate = review_gate_appendix_in(&roots);
    assert!(gate.contains("PROJECT GATE BODY"));
    assert!(!gate.contains("merge-base"));
}

#[test]
fn gated_review_system_prompt_keeps_verdict_contract() {
    let prompt = gated_review_system_prompt("You are a reviewer. APPROVE or OBJECT.", false);
    assert!(prompt.starts_with("You are a reviewer."));
    assert!(prompt.contains("introduced by this change"));
    assert!(prompt.contains("Line 1 remains exactly APPROVE or OBJECT."));
    let escalate = gated_review_system_prompt("You are a reviewer.", true);
    assert!(escalate.contains("APPROVE, OBJECT, or ESCALATE"));
}

#[test]
fn agents_skills_lose_to_project_and_beat_global() {
    let project = unique_dir("agents-project");
    let agents = unique_dir("agents-agents");
    let global = unique_dir("agents-global");
    write_skill(&global, "shared", "shared", "global desc", "global", "g");
    write_skill(&agents, "shared", "shared", "agents desc", "agents", "a");
    write_skill(&project, "shared", "shared", "project desc", "project", "p");
    write_skill(
        &agents,
        "only-agents",
        "only-agents",
        "from agents",
        "agents",
        "x",
    );
    let roots = SkillRoots {
        project,
        agents,
        global,
    };
    let skills = disk_skills(&roots);
    let shared = skills.iter().find(|s| s.name == "shared").unwrap();
    assert_eq!(shared.scope, "project");
    assert!(skills.iter().any(|s| s.name == "only-agents"));
}

#[test]
fn disable_model_invocation_omits_index_but_skill_still_loads() {
    let project = unique_dir("disable-project");
    fs::create_dir_all(project.join("hidden")).unwrap();
    fs::write(
            project.join("hidden/SKILL.md"),
            "---\nname: hidden-flow\ndescription: secret procedure\nscope: project\ndisable-model-invocation: true\n---\n\n# hidden\n\nBODY\n",
        )
        .unwrap();
    let roots = SkillRoots {
        agents: PathBuf::new(),
        project,
        global: unique_dir("disable-global"),
    };
    let skills = list_skills_in(&roots);
    assert!(skills.iter().any(|s| s.name == "hidden-flow"));
    let index = learned_skills_context_from(&skills).unwrap_or_default();
    assert!(
        !index.contains("hidden-flow"),
        "disabled skills stay out of the model index: {index}"
    );
    let loaded = read_skill_in(&roots, "hidden-flow").unwrap();
    assert!(loaded.content.contains("BODY"));
}
