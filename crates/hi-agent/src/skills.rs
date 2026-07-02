//! Learned skill discovery and prompt helpers.
//!
//! Learned skills are ordinary Markdown files in project/global directories:
//! `.hi/skills/<slug>/SKILL.md` and `~/.config/hi/skills/<slug>/SKILL.md`.
//! Startup loads only a compact index. Full bodies are read only when the user
//! explicitly invokes `/skill <name>`.

use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Result, anyhow};

const PROJECT_SKILLS_DIR: &str = ".hi/skills";
const MAX_SKILL_BYTES: usize = 64 * 1024;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SkillRoots {
    pub project: PathBuf,
    pub global: PathBuf,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LearnedSkill {
    pub name: String,
    pub description: String,
    pub scope: String,
    pub path: PathBuf,
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
    for (root, default_scope) in [(&roots.project, "project"), (&roots.global, "global")] {
        for skill in scan_root(root, default_scope) {
            if seen.insert(skill.name.clone()) {
                out.push(skill);
            }
        }
    }
    out
}

/// Read one learned skill by its frontmatter name. Project wins over global.
pub fn read_skill(name: &str) -> Result<SkillContent> {
    read_skill_in(&skill_roots(), name)
}

pub fn read_skill_in(roots: &SkillRoots, name: &str) -> Result<SkillContent> {
    let needle = normalize_name(name);
    for skill in list_skills_in(roots) {
        if normalize_name(&skill.name) == needle {
            let content = fs::read_to_string(&skill.path)
                .map_err(|err| anyhow!("failed to read skill '{}': {err}", skill.name))?;
            return Ok(SkillContent { skill, content });
        }
    }
    Err(anyhow!("skill '{name}' not found"))
}

/// Render only compact metadata for startup context. Full skill bodies are not
/// included here.
pub fn learned_skills_context() -> Option<String> {
    learned_skills_context_from(&list_skills())
}

pub fn learned_skills_context_from(skills: &[LearnedSkill]) -> Option<String> {
    if skills.is_empty() {
        return None;
    }
    let mut out = String::from("# Learned Skills\n");
    out.push_str("Available learned skills are indexed below. Do not assume their full procedure; use `/skill <name>` when the user asks to apply one.\n");
    for skill in skills {
        out.push_str("- ");
        out.push_str(&skill.name);
        out.push_str(" [");
        out.push_str(&skill.scope);
        out.push_str("]: ");
        out.push_str(&skill.description);
        out.push('\n');
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
         - After writing the skill, briefly report the path and scope."
    )
}

/// Prompt used by `/skill <name>` to inject the full selected skill body as an
/// explicit user turn.
pub fn build_skill_use_prompt(name: &str, content: &str) -> String {
    format!(
        "Use the learned skill `{}` for the current task/context.\n\n---\n{}\n---\n\nApply this skill only where it is relevant, and continue with the user's current task.",
        name.trim(),
        content.trim()
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
    })
}

#[derive(Default)]
struct SkillFrontmatter {
    name: Option<String>,
    description: Option<String>,
    scope: Option<String>,
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
mod tests {
    use super::*;

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

    fn write_skill(
        root: &Path,
        slug: &str,
        name: &str,
        description: &str,
        scope: &str,
        body: &str,
    ) {
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
        let roots = SkillRoots { project, global };
        let skills = list_skills_in(&roots);
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
            project,
            global: unique_dir("malformed-global"),
        };
        let skills = list_skills_in(&roots);
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
    }
}
