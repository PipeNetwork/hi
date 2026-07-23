//! Feasibility preflight for a freshly planned goal.
//!
//! A long-horizon plan routinely contains steps that name infrastructure the
//! machine doesn't have — "integration tests against an embedded PostgreSQL",
//! "verify the OpenTofu modules plan". Nothing checked for those up front, so
//! the drive discovered them one exhausted retry budget at a time, hours in,
//! and reported them as failures of the *work*.
//!
//! [`Agent::block_step`](crate::Agent::handle_block_step) is the robust answer:
//! the model declares a blocker when it actually hits one. This module is the
//! cheap complement — scan the checklist at creation time and tell the user
//! which prerequisites are missing *before* the run starts, while it still
//! costs nothing to fix.
//!
//! Deliberately advisory. It never blocks or reorders a step: the match is a
//! keyword heuristic over prose, and acting on it would risk setting aside work
//! that was perfectly feasible. Being wrong here should cost a line of text.

/// How to decide a prerequisite is actually usable here.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Detect {
    /// The binary resolving on `PATH` is the whole requirement.
    Binary,
    /// A *reachable server*, not just a client. `psql` ships with any Postgres
    /// install and on developer machines is almost always present while no
    /// server runs — which is precisely the case that stalled the run this
    /// module exists for. Checking the binary alone would stay silent on it.
    PostgresServer,
    /// A running daemon, not just the CLI.
    DockerDaemon,
}

/// A prerequisite a plan step names, and how to detect it.
struct Prerequisite {
    /// What to tell the user is missing.
    label: &'static str,
    /// Executable that must resolve on `PATH`.
    binary: &'static str,
    /// Lowercase substrings in a step description that imply this prerequisite.
    /// Kept specific — "test" or "db" would match most of any plan.
    markers: &'static [&'static str],
    detect: Detect,
    /// Extra note narrowing what to do about it.
    caveat: Option<&'static str>,
}

const PREREQUISITES: &[Prerequisite] = &[
    Prerequisite {
        label: "PostgreSQL",
        binary: "psql",
        markers: &["postgres", "postgresql", "sqlx", "pg_"],
        detect: Detect::PostgresServer,
        caveat: Some("no server reachable — start one and set DATABASE_URL"),
    },
    Prerequisite {
        label: "Docker",
        binary: "docker",
        markers: &["docker", "testcontainer", "test container"],
        detect: Detect::DockerDaemon,
        caveat: Some("the daemon must be running, not just the CLI installed"),
    },
    Prerequisite {
        label: "OpenTofu or Terraform",
        binary: "tofu",
        markers: &["opentofu", "terraform", "tofu "],
        detect: Detect::Binary,
        caveat: None,
    },
    Prerequisite {
        label: "Helm",
        binary: "helm",
        markers: &["helm"],
        detect: Detect::Binary,
        caveat: None,
    },
    Prerequisite {
        label: "kubectl",
        binary: "kubectl",
        markers: &["kubectl", "kubernetes", "k8s"],
        detect: Detect::Binary,
        caveat: Some("an actual cluster is also needed for integration steps"),
    },
    Prerequisite {
        label: "the AWS CLI",
        binary: "aws",
        markers: &["aws ", "amazon web services", "eks", "s3 "],
        detect: Detect::Binary,
        caveat: Some("credentials and an account are also needed"),
    },
    Prerequisite {
        label: "the gcloud CLI",
        binary: "gcloud",
        markers: &["gcloud", "google cloud", "gcp"],
        detect: Detect::Binary,
        caveat: None,
    },
];

/// Whether `binary` resolves on `PATH`.
fn on_path(binary: &str) -> bool {
    let Some(path) = std::env::var_os("PATH") else {
        return false;
    };
    std::env::split_paths(&path).any(|dir| {
        let candidate = dir.join(binary);
        // `is_file` alone would accept a non-executable of the same name.
        std::fs::metadata(&candidate).is_ok_and(|meta| meta.is_file()) && is_executable(&candidate)
    })
}

#[cfg(unix)]
fn is_executable(path: &std::path::Path) -> bool {
    use std::os::unix::fs::PermissionsExt;
    std::fs::metadata(path).is_ok_and(|meta| meta.permissions().mode() & 0o111 != 0)
}

#[cfg(not(unix))]
fn is_executable(path: &std::path::Path) -> bool {
    path.is_file()
}

/// Run a probe command, treating anything but a clean exit as unavailable.
///
/// Bounded by the probe itself: these are all local status queries that either
/// answer immediately or fail fast. A probe that cannot be spawned at all (the
/// binary vanished between the `PATH` check and here) reads as unavailable,
/// which is the safe direction — the worst case is one advisory line.
fn probe_succeeds(program: &str, args: &[&str]) -> bool {
    std::process::Command::new(program)
        .args(args)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .is_ok_and(|status| status.success())
}

/// Whether this prerequisite is actually usable here, not merely installed.
fn available(prerequisite: &Prerequisite) -> bool {
    if !on_path(prerequisite.binary) {
        return false;
    }
    match prerequisite.detect {
        Detect::Binary => true,
        // An explicit DATABASE_URL is taken at face value: it may point at a
        // remote server this probe can't see, and second-guessing the user's
        // own configuration would produce a false warning.
        Detect::PostgresServer => {
            std::env::var("DATABASE_URL").is_ok_and(|url| !url.trim().is_empty())
                || probe_succeeds("pg_isready", &["-q"])
        }
        Detect::DockerDaemon => probe_succeeds("docker", &["info"]),
    }
}

/// One missing prerequisite and the 1-based step numbers that name it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MissingPrerequisite {
    pub label: String,
    pub steps: Vec<usize>,
    pub caveat: Option<String>,
}

/// Scan planned step descriptions for prerequisites this machine lacks.
///
/// Returns at most one entry per prerequisite, each listing the steps that need
/// it, ordered as declared so output is stable.
pub fn missing_prerequisites(steps: &[String]) -> Vec<MissingPrerequisite> {
    let lowered: Vec<String> = steps.iter().map(|s| s.to_ascii_lowercase()).collect();
    PREREQUISITES
        .iter()
        .filter_map(|prerequisite| {
            let matched: Vec<usize> = lowered
                .iter()
                .enumerate()
                .filter(|(_, step)| {
                    prerequisite
                        .markers
                        .iter()
                        .any(|marker| step.contains(marker))
                })
                .map(|(i, _)| i + 1)
                .collect();
            // Only report a prerequisite some step actually asks for, and only
            // when it isn't usable here.
            if matched.is_empty() || available(prerequisite) {
                return None;
            }
            Some(MissingPrerequisite {
                label: prerequisite.label.to_string(),
                steps: matched,
                caveat: prerequisite.caveat.map(str::to_string),
            })
        })
        .collect()
}

/// Render the advisory shown after a goal is planned. `None` when nothing is
/// missing — silence is the common case and shouldn't cost a line.
pub fn advisory(steps: &[String]) -> Option<String> {
    let missing = missing_prerequisites(steps);
    if missing.is_empty() {
        return None;
    }
    let mut out = String::from(
        "⚠ some steps name prerequisites this machine doesn't have. They can't complete as written — install these, or expect the drive to report them blocked:",
    );
    for item in missing {
        let steps = item
            .steps
            .iter()
            .map(|n| n.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        out.push_str(&format!("\n  · {} — step(s) {steps}", item.label));
        if let Some(caveat) = item.caveat {
            out.push_str(&format!(" ({caveat})"));
        }
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reports_only_prerequisites_a_step_actually_names() {
        // `ls` stands in for a binary that certainly exists, proving presence
        // suppresses the report; the absent ones are what we assert on.
        let steps = vec!["Write a pure function that formats durations.".to_string()];
        assert!(
            missing_prerequisites(&steps).is_empty(),
            "a plan naming no infrastructure must report nothing"
        );
        assert!(advisory(&steps).is_none());
    }

    #[test]
    fn groups_steps_under_the_prerequisite_they_need() {
        let steps = vec![
            "Implement the persistence crate with SQLx repositories and integration tests against an embedded PostgreSQL.".to_string(),
            "Implement the model catalog tables with sqlx repositories.".to_string(),
            "Write the ADR documents.".to_string(),
        ];
        let missing = missing_prerequisites(&steps);
        // Only assert on the postgres entry — whether `psql` exists varies by
        // machine, so tolerate its absence from the result.
        if let Some(pg) = missing.iter().find(|m| m.label == "PostgreSQL") {
            assert_eq!(pg.steps, vec![1, 2], "both sqlx steps, not the ADR one");
            assert!(pg.caveat.is_some(), "a running server is the real gotcha");
        }
    }

    #[test]
    fn nothing_usable_is_ever_reported_missing() {
        // Whatever this machine has, every entry returned must genuinely be
        // unusable — otherwise the advisory cries wolf and gets ignored.
        let steps = vec![
            "Use docker, kubectl, helm, terraform, aws, gcloud and postgres for everything."
                .to_string(),
        ];
        for item in missing_prerequisites(&steps) {
            let prerequisite = PREREQUISITES
                .iter()
                .find(|p| p.label == item.label)
                .expect("reported label must come from the table");
            assert!(
                !available(prerequisite),
                "{} was reported missing but is usable here",
                item.label
            );
        }
    }

    #[test]
    fn a_client_binary_alone_does_not_count_as_a_running_service() {
        // The case this module exists for: `psql` is installed on nearly every
        // developer machine while no server runs. Checking `PATH` alone would
        // stay silent on exactly the steps that stalled the real run.
        let postgres = PREREQUISITES
            .iter()
            .find(|p| p.label == "PostgreSQL")
            .expect("postgres entry");
        assert_eq!(postgres.detect, Detect::PostgresServer);
        if on_path(postgres.binary)
            && std::env::var("DATABASE_URL").is_err()
            && !probe_succeeds("pg_isready", &["-q"])
        {
            assert!(
                !available(postgres),
                "an installed client with no reachable server must count as missing"
            );
        }
    }

    #[test]
    fn advisory_names_the_steps_so_the_user_can_judge_scope() {
        let steps = vec!["Run the opentofu modules against a mock backend.".to_string()];
        if let Some(text) = advisory(&steps) {
            assert!(text.contains("step(s) 1"), "{text}");
            assert!(text.contains("OpenTofu"), "{text}");
        }
    }
}
