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
fn committed_quality_tapes_judge_clean() {
    let suite = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../bench/quality");
    let results = hi_eval::judge::judge_suite(&suite).expect("quality judge suite");
    assert!(
        results.len() >= 14,
        "expected committed quality pass/fail tapes, got {}",
        results.len()
    );
}

#[test]
fn quality_matrix_covers_every_task() {
    let suite = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../bench/quality");
    let raw = std::fs::read_to_string(suite.join("matrix.toml")).expect("matrix.toml");
    let matrix: toml::Value = raw.parse().expect("matrix.toml parses");
    assert_eq!(matrix["schema_version"].as_integer(), Some(1));
    let rows = matrix["row"].as_array().expect("[[row]]");
    let mut matrix_ids = Vec::new();
    for row in rows {
        let id = row["id"].as_str().expect("id").to_string();
        let baseline = row["baseline"].as_str().expect("baseline");
        assert!(
            matches!(baseline, "passing" | "partial" | "known_gap"),
            "{id} baseline={baseline}"
        );
        let coverage = row["coverage"].as_str().expect("coverage");
        assert!(
            matches!(coverage, "tape" | "tape_and_live" | "planned"),
            "{id} coverage={coverage}"
        );
        assert!(
            suite.join(&id).join("task.toml").is_file(),
            "matrix row {id} has no task.toml"
        );
        assert!(
            suite.join(&id).join("judge.toml").is_file(),
            "matrix row {id} has no judge.toml"
        );
        matrix_ids.push(id);
    }
    matrix_ids.sort();
    let mut disk_ids = Vec::new();
    for entry in std::fs::read_dir(&suite).unwrap() {
        let path = entry.unwrap().path();
        if path.join("task.toml").is_file() && path.join("judge.toml").is_file() {
            disk_ids.push(path.file_name().unwrap().to_string_lossy().into_owned());
        }
    }
    disk_ids.sort();
    assert_eq!(
        matrix_ids, disk_ids,
        "bench/quality/matrix.toml must list every quality task once"
    );
}

#[test]
fn quality_live_baseline_is_present() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../eval-baseline/quality.json");
    let raw = std::fs::read_to_string(&path).expect("eval-baseline/quality.json");
    let v: serde_json::Value = serde_json::from_str(&raw).unwrap();
    let process = v["process_pass_rate"].as_f64().expect("process_pass_rate");
    assert!(
        (process - 5.0 / 7.0).abs() < 1e-9,
        "locked 5/7 process after inspect bash allow, got {process}"
    );
    assert_eq!(v["budget_pass_rate"].as_f64(), Some(1.0));
    assert_eq!(v["write_overwrite_violations"].as_u64(), Some(0));
    assert_eq!(v["image_elision_misses"].as_u64(), Some(0));
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
