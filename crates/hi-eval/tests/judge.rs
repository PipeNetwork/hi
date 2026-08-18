//! Model-free harness judge: committed tapes must stay green.

use std::path::PathBuf;

#[test]
fn committed_harness_tapes_judge_clean() {
    let suite = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../bench/harness");
    let results = hi_eval::judge::judge_suite(&suite).expect("judge suite");
    assert!(
        results.len() >= 19,
        "expected committed pass/fail tapes, got {}",
        results.len()
    );
}

#[test]
fn fixture_json_tapes_cover_write_image_and_pass() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/judge");
    let rules = hi_eval::judge::load_rules(&dir.join("rules.toml")).unwrap();

    let pass: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("pass.json")).unwrap()).unwrap();
    assert!(hi_eval::judge::judge_report(&pass, &rules).ok());

    let image: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("image-bomb.json")).unwrap())
            .unwrap();
    let image_report = hi_eval::judge::judge_report(&image, &rules);
    assert!(!image_report.ok());
    assert!(
        image_report
            .violations
            .iter()
            .any(|v| v.rule.contains("image")),
        "{:?}",
        image_report.violations
    );

    let write: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("write-overwrite.json")).unwrap())
            .unwrap();
    let write_report = hi_eval::judge::judge_report(&write, &rules);
    assert_eq!(write_report.process, hi_eval::judge::JudgeVerdict::Fail);
}
