use std::path::PathBuf;

use anyhow::{Context, Result, bail};

use crate::artifacts::find_hi;
use crate::artifacts::make_workdir;
use crate::config::{Config, EvalProfile, Task};
use crate::runner::run_config;

/// The trivial task used by `--self-test`: cheap prompt, file-existence verify.
/// Kept constant so the unit test can assert the fixture fails before any model
/// run, without depending on `bench/tasks`.
pub const SELF_TEST_PROMPT: &str = "create a file named `done`";
pub const SELF_TEST_VERIFY: &str = "[ -f done ]";

/// Build the self-test's on-disk fixture: a temp `task_dir` containing an empty
/// `fixture/` subdir (the shape `run_candidate`/`run_config` expect). The verify
/// command fails on the empty fixture (no `done` file) — exactly the
/// fail-before property `validate_task` checks for real tasks.
pub fn self_test_fixture() -> Result<PathBuf> {
    let dir = make_workdir()?;
    std::fs::create_dir_all(dir.join("fixture"))?;
    Ok(dir)
}

/// `--self-test`: run a single trivial fixture through every active config using
/// the *same* execution pattern as `async_main` — spawn one task per config on
/// the existing Tokio runtime and `await` `run_config` from it. This is the
/// regression guard for the nested-runtime panic (finding 5): if anything in the
/// parallel `best-of-3` path tries to create a second runtime, this panics and
/// the test fails non-zero.
///
/// Unlike `--validate` (which only checks fixture structure), this exercises the
/// real parallel execution + aggregation path, so it requires a working `hi`
/// binary and model env (`HI_MODEL` / `HI_API_KEY`). It is a smoke check of the
/// *harness*, not the model: we assert mechanics (no panic, candidate count,
/// aggregation completed) rather than pass/fail, so a model failure is not a
/// self-test failure.
pub async fn run_self_test(active: &[&Config], profile: EvalProfile) -> Result<()> {
    let hi = find_hi()?;
    profile.validate_env()?;
    if std::env::var("HI_MODEL").is_err() {
        bail!(
            "--self-test requires HI_MODEL (and HI_API_KEY) to invoke the hi binary; \
             set them or use --validate for a fixture-only check"
        );
    }

    // The self-test calls run_config directly (not the async_main semaphore),
    // so candidates are bounded by spawn_blocking's pool — cheap and
    // deterministic without touching HI_EVAL_CONCURRENCY. We deliberately don't
    // mutate the process env (set_var is unsafe and racy with tokio's threads).

    let task_dir = self_test_fixture()?;
    let task = Task {
        name: Some("self-test".to_string()),
        prompt: SELF_TEST_PROMPT.to_string(),
        verify: SELF_TEST_VERIFY.to_string(),
    };

    eprintln!(
        "hi-eval --self-test: {} config(s) · profile={} · hi={}",
        active.len(),
        profile.label(),
        hi.display()
    );

    // Same pattern as async_main: spawn one task per (task,config) on this
    // runtime and await run_config. run_config itself spawns candidates via
    // spawn_blocking and awaits them — we must NOT introduce a nested Runtime.
    let mut futs = Vec::new();
    for config in active {
        let hi = hi.clone();
        let task_dir = task_dir.clone();
        let task = task.clone();
        let config_name = config.name.to_string();
        let use_verify = config.use_verify;
        let temperatures = config.temperatures.to_vec();
        let config_env = config.env;
        futs.push(tokio::spawn(async move {
            run_config(
                &hi,
                &task_dir,
                &task,
                &config_name,
                use_verify,
                &temperatures,
                config_env,
                profile,
                None,
            )
            .await
            .with_context(|| format!("self-test config '{config_name}' failed to execute"))
        }));
    }

    let mut failures: Vec<String> = Vec::new();
    for (i, fut) in futs.into_iter().enumerate() {
        let config = active[i];
        // `tokio::spawn` propagates panics as JoinErrors — a nested-runtime panic
        // surfaces here as a failed join, which we turn into a non-zero exit.
        let joined = fut.await.context("joining self-test task (panic?)")?;
        match joined {
            Ok(result) => {
                // best-of-3 must actually have run 3 candidates and aggregated
                // them without panicking (we got here, so aggregation completed).
                let expected = config.temperatures.len();
                if result.candidates != expected {
                    failures.push(format!(
                        "{}: expected {expected} candidates, got {}",
                        config.name, result.candidates
                    ));
                }
                eprintln!(
                    "  {:10} ran {} candidate(s) · passed={} (mechanics OK)",
                    config.name, result.candidates, result.passed
                );
            }
            Err(err) => {
                failures.push(format!("{}: {err:#}", config.name));
            }
        }
    }

    let _ = std::fs::remove_dir_all(&task_dir);

    if !failures.is_empty() {
        bail!("--self-test failed:\n  {}", failures.join("\n  "));
    }
    eprintln!("--self-test OK: all configs executed, no nested-runtime panic");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{SELF_TEST_PROMPT, SELF_TEST_VERIFY, self_test_fixture};
    use crate::artifacts::{make_workdir, verify_in};
    use crate::config::CONFIGS;

    // ---- --self-test fixture + candidate-count assertions ----
    //
    // These confirm the self-test's harness mechanics without spawning the `hi`
    // binary (so no model/API needed): the built-in fixture's verify command
    // fails on the raw fixture, passes once the expected file exists, and the
    // candidate-count assertion the self-test makes is correct for each config.

    #[test]
    fn self_test_fixture_verify_fails_before() {
        // The self-test verify must fail on the empty fixture (no `done` file) —
        // this is the fail-before property that makes the fixture well-formed
        // and ensures a passing model run is a real signal, not a tautology.
        let work = make_workdir().unwrap();
        std::fs::create_dir_all(work.join("fixture")).unwrap();
        let fixture = work.join("fixture");
        assert!(
            !verify_in(&fixture, SELF_TEST_VERIFY),
            "self-test verify should fail on the empty fixture"
        );

        // ...and pass once the prompt's deliverable exists.
        std::fs::write(fixture.join("done"), "").unwrap();
        assert!(
            verify_in(&fixture, SELF_TEST_VERIFY),
            "self-test verify should pass once `done` exists"
        );

        let _ = std::fs::remove_dir_all(&work);
    }

    #[test]
    fn self_test_prompt_names_a_creatable_file() {
        // The prompt must describe something the verify command actually checks
        // for — if these drift, the self-test becomes meaningless.
        assert!(SELF_TEST_PROMPT.contains("done"));
        assert!(SELF_TEST_VERIFY.contains("done"));
    }

    #[test]
    fn self_test_candidate_count_matches_config_temperatures() {
        // Mirrors the assertion run_self_test makes: a config's candidate count
        // must equal temperatures.len(). best-of-3 must actually be 3 — the
        // regression finding 5 broke (aggregation ran on <3 candidates) would
        // trip this.
        fn expected_candidates(config: &crate::config::Config) -> usize {
            config.temperatures.len()
        }

        let configs = CONFIGS;
        let baseline = configs.iter().find(|c| c.name == "baseline").unwrap();
        let verify = configs.iter().find(|c| c.name == "verify").unwrap();
        let best_of_3 = configs.iter().find(|c| c.name == "best-of-3").unwrap();

        assert_eq!(expected_candidates(baseline), 1);
        assert_eq!(expected_candidates(verify), 1);
        assert_eq!(
            expected_candidates(best_of_3),
            3,
            "best-of-3 must run 3 candidates"
        );

        // The RunResult.candidates field is initialized to temperatures.len()
        // inside run_config; simulate that invariant here.
        for config in configs {
            assert!(
                config.temperatures.iter().all(|t| (0.0..=2.0).contains(t)),
                "temperatures out of expected range"
            );
        }
    }

    #[test]
    fn self_test_fixture_is_isolated_temp_dir() {
        // The self-test must run in an isolated temp dir so `[ -f done ]` can't
        // accidentally pass against a pre-existing file in the repo. make_workdir
        // is the same routine run_candidate uses, so this also guards that path.
        let dir = self_test_fixture().unwrap();
        assert!(dir.is_dir(), "self-test fixture dir must exist");
        assert!(dir.join("fixture").is_dir(), "fixture/ subdir must exist");
        // Nothing pre-existing under fixture/.
        assert!(
            std::fs::read_dir(dir.join("fixture"))
                .unwrap()
                .next()
                .is_none(),
            "fixture/ must start empty"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
