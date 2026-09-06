//! Learned skill discovery and prompt helpers.
//!
//! Learned skills are ordinary Markdown files in project/global directories:
//! `.hi/skills/<slug>/SKILL.md` and `~/.config/hi/skills/<slug>/SKILL.md`.
//! Startup loads only a compact index into the stable system prompt. A matching
//! stack pack (Rust / pytest / TS) is injected into the per-turn volatile
//! context from repo markers. Other full bodies still load via `/skill <name>`.
//!
//! **Built-in packs** (Rust workspace, pytest package, TS monorepo, code-review,
//! secret-scan, dep-audit) ship embedded in the binary and appear as
//! `scope: builtin` when not shadowed by a same-named project/global skill.
//! Source of truth: repo `skills/*/SKILL.md`. The code-review pack auto-injects
//! on review-shaped turns; stack packs auto-inject from repo markers on coding
//! turns. `secret-scan` and `dep-audit` are `/skill` only (not auto-injected).

use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Result, anyhow};
use hi_tools::{MutationPlan, PlannedFileMutation};

const PROJECT_SKILLS_DIR: &str = ".hi/skills";
const MAX_SKILL_BYTES: usize = 64 * 1024;
/// Cap for the auto-injected stack-skill body in the volatile context block.
const MAX_ACTIVE_STACK_SKILL_CHARS: usize = 4_000;
/// Cap for the Gate excerpt appended to chat-only APPROVE/OBJECT reviewers.
const MAX_REVIEW_GATE_CHARS: usize = 900;
const CODE_REVIEW_SKILL: &str = "code-review";
/// The startup skill index lives in the stable system prompt. Unbounded
/// descriptions (or hundreds of skills) would tax every model call.
const MAX_SKILLS_IN_INDEX: usize = 32;
const MAX_SKILL_DESCRIPTION_CHARS: usize = 160;
const MAX_SKILLS_CONTEXT_CHARS: usize = 2_000;
/// `/skill` injects a full body as a user turn; keep it well under a context
/// page even when the on-disk file is at [`MAX_SKILL_BYTES`].
const MAX_SKILL_USE_PROMPT_CHARS: usize = 8_000;
/// Keep workflow preferences subordinate to the current task without changing
/// the harness's enforced authorization or trust boundaries.
const SKILL_APPLICATION_GUIDANCE: &str = "Apply only relevant guidance. Explicit user instructions \
and authorization take precedence over skill workflow guidelines; enforced safety, permission, \
trust, and tool policies still apply. If a skill causes you to pause or diverge from the user's \
task, name the exact SKILL.md source, quote the relevant instruction, and distinguish its explicit \
requirements from your interpretation.";

/// Embedded stack skill packs (Phase N). Order is display order in the index.
const BUILTIN_SKILL_SOURCES: &[(&str, &str)] = &[
    (
        "rust-workspace",
        include_str!("../../../skills/rust-workspace/SKILL.md"),
    ),
    (
        "pytest-package",
        include_str!("../../../skills/pytest-package/SKILL.md"),
    ),
    (
        "ts-monorepo",
        include_str!("../../../skills/ts-monorepo/SKILL.md"),
    ),
    (
        "code-review",
        include_str!("../../../skills/code-review/SKILL.md"),
    ),
    (
        "secret-scan",
        include_str!("../../../skills/secret-scan/SKILL.md"),
    ),
    (
        "dep-audit",
        include_str!("../../../skills/dep-audit/SKILL.md"),
    ),
];

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SkillRoots {
    pub project: PathBuf,
    /// Workspace `.agents/skills` (Agent Skills spec). Shadows global, loses to `.hi/skills`.
    pub agents: PathBuf,
    pub global: PathBuf,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LearnedSkill {
    pub name: String,
    pub description: String,
    pub scope: String,
    pub path: PathBuf,
    /// Agent Skills spec: omit from the model-facing index; `/skill` still loads it.
    pub disable_model_invocation: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SkillContent {
    pub skill: LearnedSkill,
    pub content: String,
}

/// Project and global skill roots. `HI_GLOBAL_SKILLS_DIR` overrides the global
/// root for tests and advanced users.
pub fn skill_roots() -> SkillRoots {
    SkillRoots {
        project: PathBuf::from(PROJECT_SKILLS_DIR),
        agents: PathBuf::from(".agents/skills"),
        global: global_skills_dir(),
    }
}

fn global_skills_dir() -> PathBuf {
    std::env::var_os("HI_GLOBAL_SKILLS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            let base = std::env::var_os("XDG_CONFIG_HOME")
                .map(PathBuf::from)
                .or_else(|| {
                    std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".config"))
                })
                .unwrap_or_else(|| PathBuf::from(".config"));
            base.join("hi").join("skills")
        })
}

/// List learned skills, project first then global. Project skills shadow global
/// skills with the same frontmatter `name`.
pub fn list_skills() -> Vec<LearnedSkill> {
    list_skills_in(&skill_roots())
}

pub fn list_skills_in(roots: &SkillRoots) -> Vec<LearnedSkill> {
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for (root, default_scope) in [
        (&roots.project, "project"),
        (&roots.agents, "agents"),
        (&roots.global, "global"),
    ] {
        for skill in scan_root(root, default_scope) {
            if seen.insert(normalize_name(&skill.name)) {
                out.push(skill);
            }
        }
    }
    // Built-in packs last — never override user/project skills of the same name.
    for skill in builtin_skills() {
        if seen.insert(normalize_name(&skill.name)) {
            out.push(skill);
        }
    }
    out
}

/// Read one learned skill by its frontmatter name. Project wins over global,
/// then built-in packs.
pub fn read_skill(name: &str) -> Result<SkillContent> {
    read_skill_in(&skill_roots(), name)
}

pub fn read_skill_in(roots: &SkillRoots, name: &str) -> Result<SkillContent> {
    let needle = normalize_name(name);
    for skill in list_skills_in(roots) {
        if normalize_name(&skill.name) == needle {
            let content = if is_builtin_skill_path(&skill.path) {
                builtin_skill_body(&skill.name)
                    .ok_or_else(|| anyhow!("builtin skill '{}' body missing", skill.name))?
                    .to_string()
            } else {
                fs::read_to_string(&skill.path)
                    .map_err(|err| anyhow!("failed to read skill '{}': {err}", skill.name))?
            };
            return Ok(SkillContent { skill, content });
        }
    }
    Err(anyhow!("skill '{name}' not found"))
}

/// Virtual path prefix for embedded packs (not on disk).
const BUILTIN_PATH_PREFIX: &str = "builtin://skills/";

fn is_builtin_skill_path(path: &Path) -> bool {
    path.to_string_lossy().starts_with(BUILTIN_PATH_PREFIX)
}

fn builtin_skills() -> Vec<LearnedSkill> {
    let mut out = Vec::new();
    for (slug, raw) in BUILTIN_SKILL_SOURCES {
        if let Some(mut skill) = load_metadata_from_str(raw, "builtin") {
            skill.path = PathBuf::from(format!("{BUILTIN_PATH_PREFIX}{slug}/SKILL.md"));
            out.push(skill);
        }
    }
    out
}

fn builtin_skill_body(name: &str) -> Option<&'static str> {
    let needle = normalize_name(name);
    for (_, raw) in BUILTIN_SKILL_SOURCES {
        let fm = parse_frontmatter(raw)?;
        let n = fm.name?;
        if normalize_name(&n) == needle {
            return Some(raw);
        }
    }
    None
}

fn load_metadata_from_str(raw: &str, default_scope: &str) -> Option<LearnedSkill> {
    if raw.len() > MAX_SKILL_BYTES {
        return None;
    }
    let frontmatter = parse_frontmatter(raw)?;
    let name = frontmatter.name?;
    let description = frontmatter.description.unwrap_or_default();
    let scope = frontmatter
        .scope
        .unwrap_or_else(|| default_scope.to_string());
    Some(LearnedSkill {
        name,
        description,
        scope,
        path: PathBuf::new(),
        disable_model_invocation: frontmatter.disable_model_invocation,
    })
}

/// Write a learned skill to `<root>/<slug>/SKILL.md` with frontmatter that round-trips through
/// [`parse_frontmatter`]. `scope` selects the project or global root. Returns the written path, or
/// `Ok(None)` if a skill with the same normalized `name` already exists (de-dup: never overwrite an
/// existing skill — the auto-curator must not clobber user-authored ones). Errors on oversize/I/O.
pub fn write_skill(
    roots: &SkillRoots,
    scope: &str,
    name: &str,
    description: &str,
    body: &str,
) -> Result<Option<PathBuf>> {
    let name = sanitize_line(name);
    if name.is_empty() {
        return Err(anyhow!("skill name is empty"));
    }
    // De-dup by normalized name across both roots (project shadows global anyway).
    let needle = normalize_name(&name);
    if list_skills_in(roots)
        .iter()
        .any(|s| normalize_name(&s.name) == needle)
    {
        return Ok(None);
    }
    let scope = if scope == "global" {
        "global"
    } else {
        "project"
    };
    let root = if scope == "global" {
        &roots.global
    } else {
        &roots.project
    };
    let description = sanitize_line(description);
    let contents = format!(
        "---\nname: {name}\ndescription: {description}\nscope: {scope}\n---\n\n{}\n",
        body.trim()
    );
    if contents.len() > MAX_SKILL_BYTES {
        return Err(anyhow!("skill '{name}' exceeds {MAX_SKILL_BYTES} bytes"));
    }
    let root = if root.is_absolute() {
        root.clone()
    } else {
        std::env::current_dir()
            .map_err(|err| anyhow!("failed to resolve skill root: {err}"))?
            .join(root)
    };
    let dir = root.join(slugify(&name));
    let file = dir.join("SKILL.md");

    // Learned content is model-authored, so publish it through the same
    // no-follow, preimage-sealed transaction path as workspace edits. Anchor
    // above both `<root>/<slug>` components: a repository cannot pre-plant
    // `.hi`, `skills`, the slug directory, or `SKILL.md` as a symlink and turn
    // automatic curation into an out-of-workspace write. `add_with_mode` also
    // preserves the never-overwrite contract under concurrent writers.
    let mut transaction_root = root
        .parent()
        .and_then(Path::parent)
        .or_else(|| root.parent())
        .unwrap_or(&root);
    while !transaction_root.exists() {
        transaction_root = transaction_root
            .parent()
            .ok_or_else(|| anyhow!("skill root has no existing ancestor: {}", root.display()))?;
    }
    let plan = MutationPlan::new(
        transaction_root,
        vec![PlannedFileMutation::add_with_mode(
            &file,
            contents.into_bytes(),
            0o644,
        )],
    )
    .map_err(|err| anyhow!("failed to prepare skill: {err}"))?;
    plan.commit()
        .map_err(|err| anyhow!("failed to commit skill: {err}"))?;
    Ok(Some(file))
}

/// Collapse a name into a filesystem-safe slug: lowercase, non-alphanumerics become single `-`.
fn slugify(name: &str) -> String {
    let mut slug = String::new();
    let mut prev_dash = false;
    for ch in name.trim().chars() {
        if ch.is_ascii_alphanumeric() {
            slug.push(ch.to_ascii_lowercase());
            prev_dash = false;
        } else if !prev_dash {
            slug.push('-');
            prev_dash = true;
        }
    }
    let slug = slug.trim_matches('-').to_string();
    if slug.is_empty() {
        "skill".to_string()
    } else {
        slug
    }
}

/// Flatten a frontmatter value to a single trimmed line (frontmatter is line-oriented).
fn sanitize_line(value: &str) -> String {
    value.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Slug of the builtin/project stack pack that matches `root`'s markers, if any.
/// Priority matches [`crate::detect_verify_pipeline`]: Cargo.toml, then
/// package.json, then Python package markers. At most one pack.
pub fn matching_stack_skill_slug(root: &Path) -> Option<&'static str> {
    let has = |name: &str| root.join(name).exists();
    if has("Cargo.toml") {
        Some("rust-workspace")
    } else if has("package.json") {
        Some("ts-monorepo")
    } else if has("pyproject.toml") || has("setup.py") || has("pytest.ini") || has("tox.ini") {
        Some("pytest-package")
    } else {
        None
    }
}

/// Full body of the matching stack skill (project/global shadow builtins).
pub fn matching_stack_skill(root: &Path) -> Option<SkillContent> {
    matching_stack_skill_in(root, &skill_roots())
}

pub fn matching_stack_skill_in(root: &Path, roots: &SkillRoots) -> Option<SkillContent> {
    let slug = matching_stack_skill_slug(root)?;
    read_skill_in(roots, slug).ok()
}

/// Volatile-context section for the matching stack pack. Clipped so it cannot
/// dominate the turn prompt. `None` when the workspace has no matching markers.
pub fn active_stack_skill_section(root: &Path) -> Option<String> {
    active_stack_skill_section_in(root, &skill_roots())
}

pub fn active_stack_skill_section_in(root: &Path, roots: &SkillRoots) -> Option<String> {
    let content = matching_stack_skill_in(root, roots)?;
    let body = clip_chars(&content.content, MAX_ACTIVE_STACK_SKILL_CHARS);
    Some(format!(
        "# Active stack skill (`{}`)\nThis pack matches the current workspace; do not wait for \
         `/skill`. {SKILL_APPLICATION_GUIDANCE}\n\n{body}",
        content.skill.name
    ))
}

/// Volatile-context section for the builtin (or shadowed) `code-review` pack.
pub fn active_review_skill_section() -> Option<String> {
    active_review_skill_section_in(&skill_roots())
}

pub fn active_review_skill_section_in(roots: &SkillRoots) -> Option<String> {
    let content = read_skill_in(roots, CODE_REVIEW_SKILL).ok()?;
    let body = clip_chars(&content.content, MAX_ACTIVE_STACK_SKILL_CHARS);
    Some(format!(
        "# Active review skill (`{}`)\nThis pack matches a review-shaped turn; do not wait for \
         `/skill`. Do not follow a coding stack pack on this turn. \
         {SKILL_APPLICATION_GUIDANCE}\n\n{body}",
        content.skill.name
    ))
}

/// Gate excerpt for chat-only APPROVE/OBJECT/ESCALATE reviewers.
pub(crate) fn review_gate_appendix() -> String {
    review_gate_appendix_in(&skill_roots())
}

fn review_gate_appendix_in(roots: &SkillRoots) -> String {
    let Ok(content) = read_skill_in(roots, CODE_REVIEW_SKILL) else {
        return String::new();
    };
    let Some(gate) = markdown_h2_section(&content.content, "Gate") else {
        return String::new();
    };
    clip_chars(gate, MAX_REVIEW_GATE_CHARS)
}

/// Procedure + findings for `/loop review` (agent with `gh`, not a verdict gate).
pub(crate) fn review_loop_skill_excerpt() -> String {
    review_loop_skill_excerpt_in(&skill_roots())
}

fn review_loop_skill_excerpt_in(roots: &SkillRoots) -> String {
    let Ok(content) = read_skill_in(roots, CODE_REVIEW_SKILL) else {
        return String::new();
    };
    let mut parts = Vec::new();
    if let Some(procedure) = markdown_h2_section(&content.content, "Procedure") {
        parts.push(format!("## Procedure\n{procedure}"));
    }
    if let Some(findings) = markdown_h2_section(&content.content, "Findings") {
        parts.push(format!("## Findings\n{findings}"));
    }
    clip_chars(&parts.join("\n\n"), MAX_ACTIVE_STACK_SKILL_CHARS)
}

/// Append the Gate excerpt to a chat-only reviewer system prompt.
pub(crate) fn gated_review_system_prompt(base: &str, allow_escalate: bool) -> String {
    let gate = review_gate_appendix();
    let line1 = if allow_escalate {
        "Line 1 remains exactly APPROVE, OBJECT, or ESCALATE."
    } else {
        "Line 1 remains exactly APPROVE or OBJECT."
    };
    if gate.is_empty() {
        format!("{base}\n\n{line1}")
    } else {
        format!("{base}\n\n{gate}\n{line1}")
    }
}

/// Body of an `## Heading` section until the next `## ` or end of file.
fn markdown_h2_section<'a>(markdown: &'a str, title: &str) -> Option<&'a str> {
    let heading = format!("## {title}");
    let start = markdown.find(&heading)?;
    let after = start + heading.len();
    let rest = markdown[after..].trim_start();
    let end = rest.find("\n## ").unwrap_or(rest.len());
    let body = rest[..end].trim();
    (!body.is_empty()).then_some(body)
}

fn clip_chars(text: &str, max: usize) -> String {
    if text.chars().count() <= max {
        return text.to_string();
    }
    let clipped: String = text.chars().take(max.saturating_sub(1)).collect();
    format!("{clipped}…")
}

/// Render only compact metadata for startup context. Full skill bodies are not
/// included here.
pub fn learned_skills_context() -> Option<String> {
    learned_skills_context_from(&list_skills())
}

/// Render the compact skills index from explicit roots. Callers that already
/// own a workspace boundary should use this instead of relying on the
/// process-global `HI_GLOBAL_SKILLS_DIR` lookup.
pub fn learned_skills_context_in(roots: &SkillRoots) -> Option<String> {
    learned_skills_context_from(&list_skills_in(roots))
}

pub fn learned_skills_context_from(skills: &[LearnedSkill]) -> Option<String> {
    let listed_skills: Vec<&LearnedSkill> = skills
        .iter()
        .filter(|skill| !skill.disable_model_invocation)
        .collect();
    if listed_skills.is_empty() {
        return None;
    }
    let mut out = String::from("# Learned Skills\n");
    out.push_str(
        "A matching stack pack may already be in this turn's context. For other skills, \
         do not assume their full procedure — use `/skill <name>` to load one. Built-in \
         packs (`rust-workspace`, `pytest-package`, `ts-monorepo`) cover package-local \
         check/test loops; `code-review` covers defect-first review turns; `secret-scan` \
         and `dep-audit` are optional `/skill` recipes (not auto-injected) — prefer them \
         over ad-hoc full-repo suites.\n",
    );
    let listed = listed_skills.len().min(MAX_SKILLS_IN_INDEX);
    for skill in listed_skills.iter().take(listed) {
        let line = format!(
            "- {} [{}]: {}\n",
            skill.name,
            skill.scope,
            clip_chars(&skill.description, MAX_SKILL_DESCRIPTION_CHARS)
        );
        if out.chars().count().saturating_add(line.chars().count()) > MAX_SKILLS_CONTEXT_CHARS {
            out.push_str("… (skill index truncated)\n");
            break;
        }
        out.push_str(&line);
    }
    if listed_skills.len() > listed {
        out.push_str(&format!(
            "… ({} more skill(s) omitted from the index — use `/skill <name>`)\n",
            listed_skills.len() - listed
        ));
    }
    Some(out)
}

/// Prompt used by `/learn [request]`. This is a normal agent turn: the model
/// gathers sources with existing tools and writes exactly one `SKILL.md`.
pub fn build_learn_prompt(request: &str) -> String {
    let request = request.trim();
    let task = if request.is_empty() {
        "Learn from the workflow we just went through in this conversation.".to_string()
    } else {
        format!("Learn this reusable workflow: {request}")
    };
    format!(
        "{task}\n\n\
         This saves a reusable procedure as a local skill file; it is not model training.\n\n\
         Requirements:\n\
         - Gather every named source using existing hi tools before writing: list, read, grep, glob, and bash only when appropriate.\n\
         - Write exactly one file named SKILL.md.\n\
         - Default to project scope at `.hi/skills/<slug>/SKILL.md`.\n\
         - Use global scope at `~/.config/hi/skills/<slug>/SKILL.md` only if the request explicitly says global, cross-project, or user-level, or the workflow is clearly repo-independent.\n\
         - The file must start with concise YAML-style frontmatter containing `name`, `description`, and `scope` (`project` or `global`).\n\
         - The body must be practical and reusable, with sections for when to use it, prerequisites, procedure, pitfalls, and verification.\n\
         - Keep it focused on reusable procedure, not a transcript of this session.\n\
         - When learning coding conventions or workspace idiom, also consult the verified-merge journal at `<state-root>/learning/verified-merges.jsonl` if it exists (each line is a delegate merge that passed independent verification: task + files) — read a few of the named files to extract the conventions that repeat across verified changes.\n\
         - After writing the skill, briefly report the path and scope."
    )
}

/// Prompt used by `/skill <name>` to inject the full selected skill body as an
/// explicit user turn.
pub fn build_skill_use_prompt(name: &str, content: &str) -> String {
    let body = clip_chars(content.trim(), MAX_SKILL_USE_PROMPT_CHARS);
    format!(
        "Use the learned skill `{}` for the current task/context.\n\n---\n{body}\n---\n\n\
         {SKILL_APPLICATION_GUIDANCE} Continue with the user's current task.",
        name.trim()
    )
}

fn scan_root(root: &Path, default_scope: &str) -> Vec<LearnedSkill> {
    let Ok(entries) = fs::read_dir(root) else {
        return Vec::new();
    };
    let mut skills = Vec::new();
    for entry in entries.flatten() {
        let Ok(file_type) = entry.file_type() else {
            continue;
        };
        if !file_type.is_dir() {
            continue;
        }
        let path = entry.path().join("SKILL.md");
        if let Some(skill) = load_metadata(&path, default_scope) {
            skills.push(skill);
        }
    }
    skills.sort_by(|left, right| left.name.cmp(&right.name));
    skills
}

fn load_metadata(path: &Path, default_scope: &str) -> Option<LearnedSkill> {
    let metadata = fs::metadata(path).ok()?;
    if metadata.len() as usize > MAX_SKILL_BYTES {
        return None;
    }
    let raw = fs::read_to_string(path).ok()?;
    let frontmatter = parse_frontmatter(&raw)?;
    let name = frontmatter.name?;
    let description = frontmatter.description.unwrap_or_default();
    let scope = frontmatter
        .scope
        .unwrap_or_else(|| default_scope.to_string());
    Some(LearnedSkill {
        name,
        description,
        scope,
        path: path.to_path_buf(),
        disable_model_invocation: frontmatter.disable_model_invocation,
    })
}

#[derive(Default)]
struct SkillFrontmatter {
    name: Option<String>,
    description: Option<String>,
    scope: Option<String>,
    disable_model_invocation: bool,
}

fn parse_frontmatter(raw: &str) -> Option<SkillFrontmatter> {
    let mut lines = raw.lines();
    if lines.next()?.trim() != "---" {
        return None;
    }
    let mut parsed = SkillFrontmatter::default();
    for line in lines {
        let line = line.trim();
        if line == "---" {
            return if parsed.name.is_some() {
                Some(parsed)
            } else {
                None
            };
        }
        let Some((key, value)) = line.split_once(':') else {
            continue;
        };
        let value = clean_frontmatter_value(value);
        match key.trim() {
            "name" if !value.is_empty() => parsed.name = Some(value),
            "description" if !value.is_empty() => parsed.description = Some(value),
            "scope" if !value.is_empty() => parsed.scope = Some(value),
            "disable-model-invocation" => {
                parsed.disable_model_invocation =
                    matches!(value.to_ascii_lowercase().as_str(), "true" | "yes" | "1");
            }
            _ => {}
        }
    }
    None
}

fn clean_frontmatter_value(value: &str) -> String {
    value
        .trim()
        .trim_matches('"')
        .trim_matches('\'')
        .trim()
        .to_string()
}

fn normalize_name(name: &str) -> String {
    name.trim().to_ascii_lowercase()
}

#[cfg(test)]
mod tests;
