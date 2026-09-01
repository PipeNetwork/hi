//! Compact UTC date for the per-turn volatile context and `/btw`.
//!
//! Models otherwise invent "today" from training cutoff. One civil date, UTC,
//! no timezone name (that would churn and leak locale). Subagents skip the
//! prompt section so token-estimate tests that empty the volatile block stay
//! stable; `/btw` still sees the date on the parent snapshot.

use std::time::{SystemTime, UNIX_EPOCH};

const WEEKDAYS: [&str; 7] = [
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
    "Monday",
    "Tuesday",
    "Wednesday",
];

/// `[Today]` block for the parent-session volatile context.
pub fn prompt_section() -> String {
    prompt_section_from_unix(unix_now())
}

/// Bullet for the `/btw` session snapshot (`- utc_date: …`).
pub fn snapshot_line() -> String {
    snapshot_line_from_unix(unix_now())
}

pub(crate) fn prompt_section_from_unix(unix_secs: u64) -> String {
    format!("[Today]\n{}", snapshot_line_from_unix(unix_secs))
}

pub(crate) fn snapshot_line_from_unix(unix_secs: u64) -> String {
    let (y, m, d, weekday) = utc_ymd_weekday(unix_secs);
    format!("- utc_date: {y:04}-{m:02}-{d:02} ({weekday})")
}

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// UTC civil date from Unix seconds. Day-of-week uses the Unix epoch
/// (1970-01-01 Thursday). Calendar conversion is Howard Hinnant's
/// `civil_from_days`.
fn utc_ymd_weekday(unix_secs: u64) -> (i32, u8, u8, &'static str) {
    let z = (unix_secs / 86_400) as i64;
    let weekday = WEEKDAYS[z.rem_euclid(7) as usize];
    let (y, m, d) = civil_from_unix_days(z);
    (y, m, d, weekday)
}

fn civil_from_unix_days(z: i64) -> (i32, u8, u8) {
    // Shift to the civil algorithm's epoch (0000-03-01).
    let z = z + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u32;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_097) / 365;
    let y = i64::from(yoe) + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u8;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    (y as i32, m as u8, d)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unix_epoch_is_thursday_1970_01_01() {
        assert_eq!(
            prompt_section_from_unix(0),
            "[Today]\n- utc_date: 1970-01-01 (Thursday)"
        );
        assert_eq!(
            snapshot_line_from_unix(86_400),
            "- utc_date: 1970-01-02 (Friday)"
        );
    }

    #[test]
    fn known_unix_civil_dates() {
        let cases = [
            (1_000_000_000, "2001-09-09 (Sunday)"),
            (1_582_934_400, "2020-02-29 (Saturday)"),
            (1_704_067_200, "2024-01-01 (Monday)"),
            (1_787_212_800, "2026-08-20 (Thursday)"),
        ];
        for (unix, date) in cases {
            let line = snapshot_line_from_unix(unix);
            assert_eq!(line, format!("- utc_date: {date}"), "unix {unix}");
        }
    }

    #[test]
    fn live_clock_emits_iso_date_shape() {
        let section = prompt_section();
        assert!(section.starts_with("[Today]\n- utc_date: "), "{section}");
        let rest = section.trim_start_matches("[Today]\n- utc_date: ");
        assert!(
            rest.len() >= 12 && rest.as_bytes()[4] == b'-' && rest.as_bytes()[7] == b'-',
            "expected YYYY-MM-DD: {section}"
        );
    }
}
