//! Generic completion contract derived from a user request.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use crate::{ReviewPolicy, VerificationMode};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskIntent {
    ReadOnly,
    Mutation,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RiskLevel {
    Normal,
    High,
}

/// The facts the turn driver uses to decide which completion gates apply.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TaskContract {
    pub intent: TaskIntent,
    /// True only when the prompt contains an explicit, imperative mutation
    /// request ("fix the login bug"), not for the mutation-capable default
    /// that ambiguous wording ("how do users use it?") or tool/artifact nouns
    /// ("does cargo build build hi-mlx?") fall into. Completion gates that
    /// demand file changes key on this; capability scoping (which tools are
    /// advertised) keys on `intent`, which stays deliberately broad.
    #[serde(default)]
    pub explicit_mutation: bool,
    pub referenced_paths: Vec<String>,
    pub acceptance_text: Vec<String>,
    pub verification: VerificationMode,
    pub risk: RiskLevel,
}

impl TaskContract {
    pub fn derive(prompt: &str, verification: VerificationMode) -> Self {
        let intent = classify_intent(prompt);
        Self {
            intent,
            explicit_mutation: intent == TaskIntent::Mutation
                && explicit_mutation_request(&prompt.to_ascii_lowercase()),
            referenced_paths: referenced_paths(prompt),
            acceptance_text: acceptance_text(prompt),
            verification,
            risk: prompt_risk(prompt),
        }
    }

    /// Mutation observed at runtime always upgrades a read-only classification.
    pub fn observe_mutation(&mut self) {
        self.intent = TaskIntent::Mutation;
    }

    /// Compact requirement digest for checkers (verify-failure nudges, the
    /// independent review): the derived acceptance sentences plus a verbatim
    /// prompt excerpt. Supplying the task's requirements is what lifts a
    /// checker's failure-detection recall — a derived contract alone misses
    /// specification-relative failures. Bounded to ~1.5 KB.
    pub fn requirements_digest(&self, prompt: &str) -> String {
        const MAX_ACCEPTANCE_LINES: usize = 8;
        const MAX_ACCEPTANCE_LINE_CHARS: usize = 200;
        const MAX_PROMPT_CHARS: usize = 700;
        let mut digest = String::new();
        for sentence in self.acceptance_text.iter().take(MAX_ACCEPTANCE_LINES) {
            digest.push_str("- ");
            digest.push_str(truncate_chars(sentence, MAX_ACCEPTANCE_LINE_CHARS));
            digest.push('\n');
        }
        if !digest.is_empty() {
            digest.push('\n');
        }
        digest.push_str("Task (verbatim, may be truncated): ");
        let excerpt = truncate_chars(prompt.trim(), MAX_PROMPT_CHARS);
        digest.push_str(excerpt);
        if excerpt.len() < prompt.trim().len() {
            digest.push('…');
        }
        digest
    }

    pub fn requires_review(
        &self,
        policy: ReviewPolicy,
        changed_files: &[String],
        diff_lines: usize,
        long_horizon_or_delegate: bool,
    ) -> bool {
        match policy {
            ReviewPolicy::Off => false,
            ReviewPolicy::Always => self.intent == TaskIntent::Mutation,
            ReviewPolicy::Risk => {
                self.intent == TaskIntent::Mutation
                    && (self.risk == RiskLevel::High
                        || long_horizon_or_delegate
                        || diff_lines > 300
                        || changed_source_or_config_count(changed_files) >= 3
                        || changed_files.iter().any(|path| risky_path(path)))
            }
        }
    }
}

fn classify_intent(prompt: &str) -> TaskIntent {
    let lower = prompt.to_ascii_lowercase();
    if lower.trim_start().starts_with("read-only ") || lower.trim_start().starts_with("read only ")
    {
        return TaskIntent::ReadOnly;
    }
    let mutating = contains_mutation_request(&lower);
    if mutating {
        return TaskIntent::Mutation;
    }
    let trimmed = lower.trim_start_matches('/').trim_start();
    let clearly_read_only = [
        "analyze",
        "answer",
        "audit",
        "describe",
        "explain",
        "find",
        "hello",
        "hi",
        "inspect",
        "list",
        "review",
        "say",
        "show",
        "status",
        "summarize",
        "tell",
        "thank",
        "thanks",
        "what",
        "where",
        "which",
        "why",
    ]
    .iter()
    .any(|prefix| {
        trimmed == *prefix
            || trimmed
                .strip_prefix(prefix)
                .is_some_and(|rest| rest.starts_with(char::is_whitespace))
    });
    if clearly_read_only {
        TaskIntent::ReadOnly
    } else {
        // Ambiguous ordinary requests are mutation-capable by default.
        TaskIntent::Mutation
    }
}

/// Whether the request contains an action verb that requires workspace
/// mutation. A leading review verb must not erase a later implementation
/// clause (for example, "review plan.md and let's keep building this").
fn contains_mutation_request(lower: &str) -> bool {
    lower
        .split(|character: char| !character.is_ascii_alphanumeric())
        .filter(|word| !word.is_empty())
        .any(is_mutation_verb)
}

fn is_mutation_verb(word: &str) -> bool {
    matches!(
        word,
        "add"
            | "adding"
            | "build"
            | "building"
            | "change"
            | "changing"
            | "create"
            | "creating"
            | "delete"
            | "deleting"
            | "edit"
            | "editing"
            | "finish"
            | "finishing"
            | "fix"
            | "fixing"
            | "implement"
            | "implementing"
            | "migrate"
            | "migrating"
            | "modify"
            | "modifying"
            | "patch"
            | "patching"
            | "refactor"
            | "refactoring"
            | "remove"
            | "removing"
            | "rename"
            | "renaming"
            | "replace"
            | "replacing"
            | "update"
            | "updating"
            | "write"
            | "writing"
    )
}

/// Whether the request uses a mutation verb as an actual instruction to change
/// the workspace, as opposed to a tool/artifact noun ("cargo build", "the
/// build") or a question about behavior ("does that build hi-mlx?"). This is
/// deliberately stricter than [`contains_mutation_request`]: it decides
/// whether a turn that ends with no file changes counts as stalled, not which
/// tools are advertised, so a false negative merely relaxes a completion gate
/// while a false positive brands a correct text-only answer "incomplete ·
/// stalled".
fn explicit_mutation_request(lower: &str) -> bool {
    let mut clause = String::new();
    let mut clauses: Vec<(String, bool)> = Vec::new();
    for character in lower.chars() {
        if matches!(character, '.' | '?' | '!' | ';' | '\n') {
            clauses.push((std::mem::take(&mut clause), character == '?'));
        } else {
            clause.push(character);
        }
    }
    if !clause.trim().is_empty() {
        clauses.push((clause, false));
    }
    clauses
        .iter()
        .any(|(clause, question)| clause_requests_mutation(clause, *question))
}

fn clause_requests_mutation(clause: &str, question: bool) -> bool {
    let words: Vec<&str> = clause
        .split(|character: char| !character.is_ascii_alphanumeric())
        .filter(|word| !word.is_empty())
        .collect();
    words.iter().enumerate().any(|(index, word)| {
        if !is_mutation_verb(word) {
            return false;
        }
        let previous = index.checked_sub(1).map(|position| words[position]);
        if question {
            // Inside a question, a mutation verb only counts as a request when
            // it is directed at the agent ("can you fix …?"), never when it
            // asks about behavior ("does that build hi-mlx?").
            return matches!(previous, Some("you" | "please"));
        }
        // Outside questions, skip tool/artifact-noun usages ("cargo build",
        // "apply the patch") and auxiliary/interrogative frames ("does it
        // build", "will this delete data") that split across a filename dot.
        !matches!(
            previous,
            Some(
                "cargo"
                    | "npm"
                    | "pnpm"
                    | "yarn"
                    | "gradle"
                    | "mvn"
                    | "bazel"
                    | "docker"
                    | "make"
                    | "cmake"
                    | "go"
                    | "rustc"
                    | "the"
                    | "a"
                    | "an"
                    | "this"
                    | "that"
                    | "these"
                    | "those"
                    | "it"
                    | "its"
                    | "my"
                    | "your"
                    | "our"
                    | "their"
                    | "release"
                    | "debug"
                    | "dev"
                    | "ci"
                    | "run"
                    | "still"
                    | "also"
                    | "not"
                    | "dont"
                    | "doesnt"
                    | "didnt"
                    | "wont"
                    | "cant"
                    | "couldnt"
                    | "wouldnt"
                    | "shouldnt"
                    | "never"
                    | "does"
                    | "do"
                    | "did"
                    | "is"
                    | "are"
                    | "was"
                    | "were"
                    | "can"
                    | "could"
                    | "will"
                    | "would"
                    | "should"
                    | "might"
                    | "may"
                    | "must"
            )
        )
    })
}

fn referenced_paths(prompt: &str) -> Vec<String> {
    let mut paths = BTreeSet::new();
    for token in prompt.split_whitespace() {
        let token = token.trim_matches(|character: char| {
            matches!(
                character,
                '`' | '\'' | '"' | '(' | ')' | '[' | ']' | '{' | '}' | ',' | ':' | ';'
            )
        });
        // Strip sentence punctuation after the surrounding-delimiter pass.
        // `plan.md.` should resolve to `plan.md`, while ordinary prose such as
        // `validation.` must not become a fake path with an empty extension.
        let token = token.trim_end_matches(['.', '?', '!']);
        if token.contains('/') || std::path::Path::new(token).extension().is_some() {
            paths.insert(token.trim_start_matches("./").to_string());
        }
    }
    paths.into_iter().collect()
}

/// Clip to at most `max` characters on a char boundary.
fn truncate_chars(text: &str, max: usize) -> &str {
    match text.char_indices().nth(max) {
        Some((byte, _)) => &text[..byte],
        None => text,
    }
}

fn acceptance_text(prompt: &str) -> Vec<String> {
    prompt
        .split(['\n', '.'])
        .map(str::trim)
        .filter(|sentence| {
            let lower = sentence.to_ascii_lowercase();
            [
                "acceptance",
                "must ",
                "should ",
                "success",
                "done when",
                "ensure ",
                "without ",
            ]
            .iter()
            .any(|needle| lower.contains(needle))
        })
        .map(str::to_string)
        .collect()
}

fn prompt_risk(prompt: &str) -> RiskLevel {
    let lower = prompt.to_ascii_lowercase();
    if [
        "auth",
        "permission",
        "security",
        "credential",
        "secret",
        "migration",
        "schema",
        "dependency",
        "lockfile",
        "ci",
        "workflow",
        "deploy",
    ]
    .iter()
    .any(|word| lower.contains(word))
    {
        RiskLevel::High
    } else {
        RiskLevel::Normal
    }
}

fn risky_path(path: &str) -> bool {
    let lower = path.replace('\\', "/").to_ascii_lowercase();
    let name = lower.rsplit('/').next().unwrap_or(&lower);
    is_ci_path(&lower, name)
        || lower.contains("/migrations/")
        || lower.starts_with("migrations/")
        || lower.contains("auth")
        || lower.contains("security")
        || lower.contains("permission")
        || is_dependency_manifest(name)
}

fn is_ci_path(path: &str, name: &str) -> bool {
    path.starts_with(".github/")
        || path.starts_with(".circleci/")
        || path.starts_with(".buildkite/")
        || path.starts_with(".woodpecker/")
        || path.starts_with("ci/")
        || path.contains("/ci/")
        || matches!(
            name,
            ".gitlab-ci.yml"
                | ".gitlab-ci.yaml"
                | ".drone.yml"
                | ".drone.yaml"
                | "azure-pipelines.yml"
                | "azure-pipelines.yaml"
                | "bitbucket-pipelines.yml"
                | "bitbucket-pipelines.yaml"
                | "jenkinsfile"
        )
}

fn is_dependency_manifest(name: &str) -> bool {
    matches!(
        name,
        "cargo.toml"
            | "cargo.lock"
            | "package.json"
            | "package-lock.json"
            | "pnpm-workspace.yaml"
            | "pnpm-lock.yaml"
            | "yarn.lock"
            | "bun.lock"
            | "bun.lockb"
            | "deno.json"
            | "deno.jsonc"
            | "go.mod"
            | "go.sum"
            | "pyproject.toml"
            | "requirements.txt"
            | "pipfile"
            | "pipfile.lock"
            | "poetry.lock"
            | "uv.lock"
            | "gemfile"
            | "gemfile.lock"
            | "composer.json"
            | "composer.lock"
            | "pom.xml"
            | "build.gradle"
            | "build.gradle.kts"
            | "settings.gradle"
            | "settings.gradle.kts"
            | "mix.exs"
            | "mix.lock"
            | "package.swift"
            | "package.resolved"
    ) || (name.starts_with("requirements-") && name.ends_with(".txt"))
        || name.ends_with(".csproj")
}

fn changed_source_or_config_count(paths: &[String]) -> usize {
    paths
        .iter()
        .filter(|path| {
            std::path::Path::new(path)
                .extension()
                .and_then(|extension| extension.to_str())
                .is_some_and(|extension| {
                    matches!(
                        extension.to_ascii_lowercase().as_str(),
                        "rs" | "py"
                            | "go"
                            | "js"
                            | "jsx"
                            | "ts"
                            | "tsx"
                            | "toml"
                            | "json"
                            | "yaml"
                            | "yml"
                    )
                })
        })
        .count()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ordinary_implementation_wording_is_mutating() {
        let contract = TaskContract::derive("implement the parser", VerificationMode::Auto);
        assert_eq!(contract.intent, TaskIntent::Mutation);

        let mixed = TaskContract::derive(
            "review plan.md and lets keep building this",
            VerificationMode::Auto,
        );
        assert_eq!(
            mixed.intent,
            TaskIntent::Mutation,
            "a leading review clause must not remove later mutation intent"
        );
    }

    #[test]
    fn referenced_paths_ignore_sentence_punctuation_and_prose() {
        let contract = TaskContract::derive(
            "Continue the long-horizon goal. Review plan.md. Complete validation.",
            VerificationMode::Auto,
        );
        assert_eq!(contract.referenced_paths, vec!["plan.md"]);
    }

    #[test]
    fn questions_stay_mutation_capable_but_do_not_expect_mutation() {
        for prompt in [
            "how do users use it? does it automatically turn on if theres not enough ram?",
            "if we do cargo build --release on a mac. does that build hi-mlx?",
            "does it build?",
            "review now. does mlx still build?",
            "will this delete my sessions?",
        ] {
            let contract = TaskContract::derive(prompt, VerificationMode::Auto);
            assert_eq!(
                contract.intent,
                TaskIntent::Mutation,
                "capability stays broad for {prompt:?}"
            );
            assert!(
                !contract.explicit_mutation,
                "a question must not expect file changes: {prompt:?}"
            );
        }
    }

    #[test]
    fn requirements_digest_bounds_and_content() {
        let prompt = "Fix the parser. It must handle empty input without panicking. \
                      The CLI should print a warning on malformed lines.";
        let contract = TaskContract::derive(prompt, VerificationMode::Auto);
        let digest = contract.requirements_digest(prompt);
        assert!(
            digest.contains("must handle empty input"),
            "acceptance sentence present: {digest}"
        );
        assert!(
            digest.contains("should print a warning"),
            "second acceptance sentence present: {digest}"
        );
        assert!(
            digest.contains("Task (verbatim, may be truncated): Fix the parser."),
            "verbatim excerpt present: {digest}"
        );

        // A huge prompt with no acceptance keywords still yields a bounded,
        // non-empty digest, truncated on a char boundary (multibyte-safe).
        let long = "é".repeat(5_000);
        let plain = TaskContract::derive(&long, VerificationMode::Auto);
        let digest = plain.requirements_digest(&long);
        assert!(!digest.is_empty());
        assert!(digest.chars().count() < 800, "bounded: {}", digest.len());
        assert!(digest.ends_with('…'), "signals truncation: {digest}");
    }

    #[test]
    fn explicit_requests_expect_mutation() {
        for prompt in [
            "fix the login bug",
            "please update the parser to handle unicode",
            "can you fix the flaky test?",
            "implement the parser",
            "review plan.md and lets keep building this",
        ] {
            let contract = TaskContract::derive(prompt, VerificationMode::Auto);
            assert_eq!(contract.intent, TaskIntent::Mutation, "{prompt:?}");
            assert!(contract.explicit_mutation, "{prompt:?}");
        }
    }

    #[test]
    fn read_only_prefix_never_expects_mutation() {
        let contract = TaskContract::derive("read-only fix report", VerificationMode::Auto);
        assert_eq!(contract.intent, TaskIntent::ReadOnly);
        assert!(!contract.explicit_mutation);
    }

    #[test]
    fn only_clear_questions_are_read_only() {
        assert_eq!(
            TaskContract::derive("explain how src/parser.rs works", VerificationMode::Auto).intent,
            TaskIntent::ReadOnly
        );
        assert_eq!(
            TaskContract::derive("parser behavior", VerificationMode::Auto).intent,
            TaskIntent::Mutation
        );
    }

    #[test]
    fn late_mutation_upgrades_contract() {
        let mut contract = TaskContract::derive("review the parser", VerificationMode::Auto);
        contract.observe_mutation();
        assert_eq!(contract.intent, TaskIntent::Mutation);
    }

    #[test]
    fn risk_review_matrix_matches_contract() {
        let normal = TaskContract::derive("implement parser", VerificationMode::Auto);
        assert!(!normal.requires_review(ReviewPolicy::Risk, &["src/parser.rs".into()], 20, false));
        assert!(normal.requires_review(
            ReviewPolicy::Risk,
            &["src/a.rs".into(), "src/b.rs".into(), "tests/a.rs".into()],
            20,
            false
        ));
        let auth = TaskContract::derive("fix auth permissions", VerificationMode::Auto);
        assert!(auth.requires_review(ReviewPolicy::Risk, &["src/auth.rs".into()], 5, false));
    }

    #[test]
    fn dependency_manifests_and_ci_paths_require_risk_review() {
        let contract = TaskContract::derive("implement parser", VerificationMode::Auto);
        for path in [
            "Cargo.toml",
            "frontend/package.json",
            "service/go.mod",
            "python/pyproject.toml",
            "python/requirements-dev.txt",
            ".github/workflows/ci.yml",
            ".circleci/config.yml",
            ".gitlab-ci.yml",
            "azure-pipelines.yml",
            "ci/release.sh",
        ] {
            assert!(
                contract.requires_review(ReviewPolicy::Risk, &[path.into()], 5, false),
                "expected independent review for {path}"
            );
        }
        assert!(!contract.requires_review(ReviewPolicy::Risk, &["src/parser.rs".into()], 5, false));
    }
}
