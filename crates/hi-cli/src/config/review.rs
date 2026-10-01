//! `[review]`: knobs for `/review` and `--spec-review`.
//!
//! ```toml
//! [review]
//! chunk_files = 12                 # files per audit turn (default 16)
//! exclude = ["bench/fixtures", "node_modules"]
//! passes = 2                       # fix passes when the command does not say
//! seed = "cargo test --offline"    # runs before each audit turn; failures are seeded
//! ```
//!
//! A repository's `hi.toml` may make the audit cheaper (more `exclude`s,
//! fewer `passes`, larger `chunk_files`) but never dearer; its `seed`
//! command runs only once the folder is trusted, like any other command a
//! checkout would have the CLI run.

use hi_harness::ReviewSettings;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReviewSection {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chunk_files: Option<usize>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub exclude: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub passes: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<String>,
}

impl ReviewSection {
    /// The harness settings this section describes; unset keys keep the
    /// harness defaults.
    pub fn to_settings(&self) -> ReviewSettings {
        let defaults = ReviewSettings::default();
        ReviewSettings {
            chunk_files: self.chunk_files.unwrap_or(defaults.chunk_files),
            exclude: self.exclude.clone(),
            passes: self.passes.unwrap_or(defaults.passes),
            seed: self
                .seed
                .as_deref()
                .map(str::trim)
                .filter(|seed| !seed.is_empty())
                .map(str::to_owned),
        }
    }
}

/// Fold a repository's `[review]` into the machine config. Audit turns bill
/// a machine credential, so a checkout may only make them fewer: `passes`
/// can go down, `chunk_files` up (fewer, larger turns), and excludes are
/// additive. The seed command is taken only from a trusted folder.
pub(crate) fn merge_project_review(
    base: &mut ReviewSection,
    project: ReviewSection,
    trusted: bool,
) {
    let defaults = ReviewSettings::default();
    if let Some(chunk_files) = project.chunk_files {
        let current = base.chunk_files.unwrap_or(defaults.chunk_files);
        base.chunk_files = Some(current.max(chunk_files));
    }
    if let Some(passes) = project.passes {
        let current = base.passes.unwrap_or(defaults.passes);
        base.passes = Some(current.min(passes));
    }
    for path in project.exclude {
        if !base.exclude.contains(&path) {
            base.exclude.push(path);
        }
    }
    if project.seed.is_some() {
        if trusted {
            base.seed = project.seed;
        } else {
            tracing::debug!("ignoring project [review] seed: folder is not trusted");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn section_parses_and_maps_to_settings() {
        let section: ReviewSection = toml::from_str(
            "chunk_files = 12\nexclude = [\"bench/fixtures\"]\npasses = 2\nseed = \" make test \"\n",
        )
        .unwrap();
        let settings = section.to_settings();
        assert_eq!(settings.chunk_files, 12);
        assert_eq!(settings.exclude, vec!["bench/fixtures".to_string()]);
        assert_eq!(settings.passes, 2);
        assert_eq!(settings.seed.as_deref(), Some("make test"));
        let empty = ReviewSection::default().to_settings();
        assert_eq!(empty, ReviewSettings::default());
        let blank_seed = ReviewSection {
            seed: Some("  ".into()),
            ..ReviewSection::default()
        };
        assert_eq!(blank_seed.to_settings().seed, None);
    }

    #[test]
    fn project_section_narrows_scope_but_seeds_only_when_trusted() {
        let mut base = ReviewSection {
            exclude: vec!["vendor".into()],
            passes: Some(3),
            ..ReviewSection::default()
        };
        let project = ReviewSection {
            chunk_files: Some(32),
            exclude: vec!["vendor".into(), "bench".into()],
            passes: Some(1),
            seed: Some("npm test".into()),
        };
        merge_project_review(&mut base, project.clone(), false);
        assert_eq!(base.chunk_files, Some(32), "larger turns: fewer of them");
        assert_eq!(base.passes, Some(1));
        assert_eq!(
            base.exclude,
            vec!["vendor".to_string(), "bench".to_string()]
        );
        assert_eq!(
            base.seed, None,
            "untrusted checkout cannot pick the command"
        );
        merge_project_review(&mut base, project, true);
        assert_eq!(base.seed.as_deref(), Some("npm test"));
        assert_eq!(base.exclude.len(), 2, "excludes deduplicate");
    }

    #[test]
    fn a_project_section_cannot_make_an_audit_bill_more() {
        let dearer = ReviewSection {
            chunk_files: Some(1),
            passes: Some(50),
            ..ReviewSection::default()
        };
        let mut fresh = ReviewSection::default();
        merge_project_review(&mut fresh, dearer.clone(), true);
        let defaults = ReviewSettings::default();
        assert_eq!(fresh.chunk_files, Some(defaults.chunk_files));
        assert_eq!(
            fresh.passes,
            Some(defaults.passes),
            "the default is the ceiling"
        );

        let mut configured = ReviewSection {
            chunk_files: Some(4),
            passes: Some(5),
            ..ReviewSection::default()
        };
        merge_project_review(&mut configured, dearer, true);
        assert_eq!(
            configured.chunk_files,
            Some(4),
            "the machine config may go lower"
        );
        assert_eq!(configured.passes, Some(5));
    }
}
