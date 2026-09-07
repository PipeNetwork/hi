//! A narrow opt-in for a check that explicitly names the generated goal view.
//! This is literal path recognition, not command execution or shell coverage.
use std::path::Path;

pub(crate) fn is_referenced(text: &str, root: &Path) -> bool {
    let relative = crate::goal::GOAL_EXPORT_PATH;
    let dotted = format!("./{relative}");
    let absolute = root.join(relative);
    [
        relative,
        dotted.as_str(),
        absolute.to_string_lossy().as_ref(),
    ]
    .iter()
    .any(|path| {
        text.match_indices(*path).any(|(index, matched)| {
            let before = text[..index].chars().next_back();
            let after = text[index + matched.len()..].chars().next();
            before.is_none_or(boundary) && after.is_none_or(boundary)
        })
    })
}

fn boundary(character: char) -> bool {
    !character.is_alphanumeric() && !matches!(character, '_' | '-' | '.' | '/' | '\\')
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_exact_goal_export_paths_in_this_workspace_are_selected() {
        let root = Path::new("/workspace with spaces");
        for text in [
            "cat .hi/goal-plan.md",
            "check './.hi/goal-plan.md'",
            "Path('.hi/goal-plan.md').read_text()",
            "check='/workspace with spaces/.hi/goal-plan.md'",
        ] {
            assert!(is_referenced(text, root), "{text}");
        }
        for text in [
            "cat .hi/goal-plan.md.bak",
            "cat other/.hi/goal-plan.md",
            "cat /other/workspace/.hi/goal-plan.md",
            "check .hi/config.toml",
        ] {
            assert!(!is_referenced(text, root), "{text}");
        }
    }
}
