//! Ingest a workspace plan document as structured-goal steps.
//!
//! Checkbox / numbered / bullet lists are the same grammar the workflow runner
//! uses. A solid checklist becomes sub-goals without a planner call; prose
//! documents still go to the planner.

use std::path::Path;

use crate::agent::plan_goal::planner_input;
use crate::goal::{Goal, GoalStatus};

/// Workflow and ingest share this cap — a 40-milestone plan fits; a 600-line
/// dump must be split.
pub const MAX_PLAN_OBJECTIVES: usize = 512;

/// One checklist row, including already-checked items so a rerun does not redo them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlanItem {
    pub description: String,
    pub done: bool,
}

/// A workspace markdown file that parsed as a real checklist.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IngestedPlan {
    /// Workspace-relative path of the document that supplied the checklist.
    pub path: String,
    pub items: Vec<PlanItem>,
}

/// Extract pending objectives: unchecked checkboxes first, then numbered
/// items, then plain bullets. Checked boxes are omitted (already done).
pub fn parse_objectives(markdown: &str) -> Vec<String> {
    parse_plan_items(markdown)
        .into_iter()
        .filter(|item| !item.done)
        .map(|item| item.description)
        .collect()
}

/// Whether the plan contains checked-off checkbox objectives — used to
/// distinguish "everything already done" (success) from "not a plan" (error).
pub fn plan_has_checked_objectives(markdown: &str) -> bool {
    markdown.lines().map(str::trim).any(checkbox_done)
}

/// Parse checklist rows, keeping `- [x]` as `done` when any `- [ ]` is present.
pub fn parse_plan_items(markdown: &str) -> Vec<PlanItem> {
    let lines: Vec<&str> = markdown.lines().map(str::trim).collect();
    let checkboxes: Vec<PlanItem> = lines
        .iter()
        .filter_map(|line| parse_checkbox(line))
        .collect();
    if checkboxes.iter().any(|item| !item.done) {
        return checkboxes
            .into_iter()
            .filter(|item| !item.description.is_empty())
            .collect();
    }
    let numbered = parse_numbered(&lines);
    if !numbered.is_empty() {
        return numbered;
    }
    parse_bullets(&lines)
}

/// A real checklist: any unchecked checkbox, any numbered list, or at least
/// two bullets. A prose PRD or a single stray bullet is not enough.
pub fn is_solid_checklist(markdown: &str) -> bool {
    let lines: Vec<&str> = markdown.lines().map(str::trim).collect();
    if lines
        .iter()
        .any(|line| parse_checkbox(line).is_some_and(|item| !item.done))
    {
        return true;
    }
    if !parse_numbered(&lines).is_empty() {
        return true;
    }
    parse_bullets(&lines).len() >= 2
}

/// Load referenced workspace `.md` files and return the first solid checklist.
pub fn ingest_plan_document(root: &Path, objective: &str) -> Option<IngestedPlan> {
    let input = planner_input(root, objective);
    for (path, body) in input.docs {
        if !is_markdown_path(&path) || !is_solid_checklist(&body) {
            continue;
        }
        let mut items = parse_plan_items(&body);
        items.truncate(MAX_PLAN_OBJECTIVES);
        if items.is_empty() {
            continue;
        }
        return Some(IngestedPlan { path, items });
    }
    None
}

/// Plain one-shot/headless `plan.md` runs go to `hi workflow run`. Fleet
/// `--session-file` children already have a worktree — they ingest in-process.
pub fn one_shot_workflow_plan_path(
    is_fleet_child: bool,
    root: &Path,
    prompt: &str,
    goal: Option<&str>,
) -> Option<String> {
    if is_fleet_child {
        return None;
    }
    ingest_plan_document(root, prompt)
        .or_else(|| goal.and_then(|objective| ingest_plan_document(root, objective)))
        .map(|plan| plan.path)
}

impl crate::Agent {
    /// Ingest a referenced workspace plan as checklist rows, if it is solid.
    pub fn ingest_plan_document(&self, objective: &str) -> Option<IngestedPlan> {
        ingest_plan_document(self.workspace_root(), objective)
    }

    /// Build a structured goal from a referenced checklist, or `None` so the
    /// caller can fall through to the planner / single-step blob.
    pub fn try_ingest_goal(&self, objective: &str) -> Option<Goal> {
        let ingested = self.ingest_plan_document(objective)?;
        Some(Goal::from_plan_items(objective.to_string(), ingested.items))
    }
}

fn is_markdown_path(path: &str) -> bool {
    Path::new(path)
        .extension()
        .is_some_and(|ext| ext.eq_ignore_ascii_case("md"))
}

fn parse_checkbox(line: &str) -> Option<PlanItem> {
    for prefix in ["- [ ]", "* [ ]", "+ [ ]"] {
        if let Some(rest) = line.strip_prefix(prefix) {
            return Some(PlanItem {
                description: rest.trim().to_string(),
                done: false,
            });
        }
    }
    for prefix in ["- [x]", "* [x]", "+ [x]", "- [X]", "* [X]", "+ [X]"] {
        if let Some(rest) = line.strip_prefix(prefix) {
            return Some(PlanItem {
                description: rest.trim().to_string(),
                done: true,
            });
        }
    }
    None
}

fn checkbox_done(line: &str) -> bool {
    parse_checkbox(line).is_some_and(|item| item.done)
}

fn parse_numbered(lines: &[&str]) -> Vec<PlanItem> {
    lines
        .iter()
        .filter_map(|line| {
            let digits = line.chars().take_while(char::is_ascii_digit).count();
            if digits == 0 {
                return None;
            }
            let rest = &line[digits..];
            rest.strip_prefix('.')
                .or_else(|| rest.strip_prefix(')'))
                .map(|text| PlanItem {
                    description: text.trim().to_string(),
                    done: false,
                })
        })
        .filter(|item| !item.description.is_empty())
        .collect()
}

fn parse_bullets(lines: &[&str]) -> Vec<PlanItem> {
    lines
        .iter()
        .filter(|line| !checkbox_done(line))
        .filter_map(|line| {
            ["- ", "* ", "+ "]
                .iter()
                .find_map(|prefix| line.strip_prefix(prefix))
                .map(|text| PlanItem {
                    description: text.trim().to_string(),
                    done: false,
                })
        })
        .filter(|item| !item.description.is_empty() && !item.description.starts_with("[x]"))
        .collect()
}

impl Goal {
    /// Install checklist rows, marking already-checked items `Done` so a rerun
    /// does not redo them. The first not-done row becomes `Active`.
    pub fn from_plan_items(objective: impl Into<String>, items: Vec<PlanItem>) -> Self {
        let done: std::collections::HashSet<String> = items
            .iter()
            .filter(|item| item.done)
            .map(|item| item.description.trim().to_string())
            .collect();
        let descriptions: Vec<String> = items.into_iter().map(|item| item.description).collect();
        let mut goal = Self::new(objective, descriptions);
        for sub_goal in &mut goal.sub_goals {
            if done.contains(&sub_goal.description) {
                sub_goal.status = GoalStatus::Done;
            }
        }
        goal.rederive_status();
        goal
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_root(label: &str) -> std::path::PathBuf {
        let root = std::env::temp_dir().join(format!(
            "hi-plan-ingest-{label}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&root).unwrap();
        root.canonicalize().unwrap()
    }

    #[test]
    fn objectives_prefer_unchecked_checkboxes_then_numbers_then_bullets() {
        let plan = "# Plan\n- [x] done already\n- [ ] first objective\n- [ ] second objective\n";
        assert_eq!(
            parse_objectives(plan),
            vec!["first objective", "second objective"]
        );
        let numbered = "notes\n1. build the loader\n2) train the model\n";
        assert_eq!(
            parse_objectives(numbered),
            vec!["build the loader", "train the model"]
        );
        let bullets = "* add tests\n* fix docs\n";
        assert_eq!(parse_objectives(bullets), vec!["add tests", "fix docs"]);
        assert!(parse_objectives("# just prose\n").is_empty());
    }

    #[test]
    fn mixed_checkboxes_keep_checked_rows_as_done() {
        let items = parse_plan_items(
            "# Plan\n- [x] already shipped\n- [ ] first objective\n- [ ] second objective\n",
        );
        assert_eq!(
            items,
            vec![
                PlanItem {
                    description: "already shipped".into(),
                    done: true,
                },
                PlanItem {
                    description: "first objective".into(),
                    done: false,
                },
                PlanItem {
                    description: "second objective".into(),
                    done: false,
                },
            ]
        );
    }

    #[test]
    fn solid_checklist_requires_real_structure() {
        assert!(is_solid_checklist("- [ ] one\n"));
        assert!(is_solid_checklist("1. only numbered\n"));
        assert!(is_solid_checklist("- first\n- second\n"));
        assert!(!is_solid_checklist("# just prose\n"));
        assert!(!is_solid_checklist("- [x] already done\n"));
        assert!(!is_solid_checklist("- a single stray bullet\n"));
    }

    #[test]
    fn ingest_checkbox_plan_skips_planner_and_marks_done() {
        let root = temp_root("checkbox");
        std::fs::write(
            root.join("plan.md"),
            "- [x] already shipped\n- [ ] wire the CLI\n- [ ] pass tests\n",
        )
        .unwrap();
        let ingested = ingest_plan_document(&root, "implement plan.md").expect("checklist");
        assert_eq!(ingested.path, "plan.md");
        assert_eq!(ingested.items.len(), 3);
        let goal = Goal::from_plan_items("implement plan.md", ingested.items);
        assert_eq!(goal.sub_goals.len(), 3);
        assert_eq!(goal.sub_goals[0].status, GoalStatus::Done);
        assert_eq!(goal.sub_goals[1].status, GoalStatus::Active);
        assert_eq!(goal.sub_goals[1].description, "wire the CLI");
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn ingest_prose_plan_is_none() {
        let root = temp_root("prose");
        std::fs::write(
            root.join("plan.md"),
            "# Design\n\nThis document describes the architecture in prose.\n",
        )
        .unwrap();
        assert!(ingest_plan_document(&root, "implement plan.md").is_none());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn one_shot_checkbox_selects_workflow_and_coding_prompt_does_not() {
        let root = temp_root("oneshot");
        std::fs::write(
            root.join("plan.md"),
            "- [ ] wire the CLI\n- [ ] pass tests\n",
        )
        .unwrap();
        assert_eq!(
            one_shot_workflow_plan_path(false, &root, "implement plan.md", None).as_deref(),
            Some("plan.md")
        );
        assert!(
            one_shot_workflow_plan_path(false, &root, "fix the off-by-one in count()", None)
                .is_none()
        );
        assert!(
            one_shot_workflow_plan_path(true, &root, "implement plan.md", None).is_none(),
            "fleet children must not nest workflow run"
        );
        assert_eq!(
            one_shot_workflow_plan_path(false, &root, "ship it", Some("implement plan.md"))
                .as_deref(),
            Some("plan.md")
        );
        std::fs::remove_dir_all(root).unwrap();
    }
}
