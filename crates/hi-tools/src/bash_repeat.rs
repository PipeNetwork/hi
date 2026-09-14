//! Cap same-turn detached binary probes (`./target/debug/foo & sleep …`).
//!
//! A live ~/chat review spawned `./target/debug/chat > /tmp/out & sleep 1`
//! more than ten times with tiny env tweaks after `recv err: timed out`.
//! Exact-string matching cannot catch that; collapsing those probes to one
//! key can.

use std::collections::HashMap;
use std::sync::Mutex;

pub(crate) const MAX_DETACHED_PROBES: u32 = 2;

/// After this many refused probes in one turn, the harness stops instead of
/// spending the rest of the wall on identical bash calls.
pub const STOP_AFTER_PROBE_REFUSALS: u32 = 2;

pub fn is_probe_refusal(content: &str) -> bool {
    content.contains("already ran this turn")
}

/// Fingerprint used for same-turn repeat detection. Detached
/// `target/debug/<bin> & sleep` probes share one key so env/path tweaks
/// cannot bypass the cap. Other commands are not fingerprinted (so
/// `cargo test` after a fix still runs).
pub fn bash_repeat_key(command: &str) -> Option<&'static str> {
    if is_detached_binary_probe(command) {
        Some("detached-binary-probe")
    } else {
        None
    }
}

fn is_detached_binary_probe(command: &str) -> bool {
    let has_bin = command.contains("target/debug/")
        || command.contains("./target/")
        || command.contains("cargo run");
    if !has_bin {
        return false;
    }
    let has_sleep = command.contains("sleep ")
        || command.contains("sleep\t")
        || command.contains("time.sleep")
        || command.contains("sleep(");
    let has_detach = command.contains(" &")
        || command.contains("&\n")
        || command.contains("&\\n")
        || command.trim_end().ends_with('&')
        || command.contains("Popen")
        || command.contains("subprocess.");
    has_bin && has_sleep && has_detach
}

/// Record this command on a per-runner ledger. After [`MAX_DETACHED_PROBES`]
/// matching probes this turn, returns a refusal instead of spawning another.
pub fn admit_bash_repeat(ledger: &Mutex<HashMap<String, u32>>, command: &str) -> Option<String> {
    let key = bash_repeat_key(command)?;
    let mut ledger = ledger
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let count = ledger.entry(key.to_string()).or_insert(0);
    *count = count.saturating_add(1);
    if *count > MAX_DETACHED_PROBES {
        Some(format!(
            "This detached `target/debug/… & sleep …` probe already ran this turn \
             ({count} times). Do not spawn another copy. Kill leftovers, change \
             the code or test, or stop. Last result was not going to change."
        ))
    } else {
        None
    }
}

pub fn reset_bash_repeats(ledger: &Mutex<HashMap<String, u32>>) {
    ledger
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .clear();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn live_chat_probe_shapes_share_one_key() {
        let a = "cd /Users/david/chat && CHAT_ADDR=127.0.0.1:0 ./target/debug/chat > /tmp/chat-out.txt 2>/tmp/chat-err.txt &\nsleep 1\nADDR=$(grep -o '127.0.0.1:[0-9]*' /tmp/chat-out.txt)";
        let b = "cd /Users/david/chat && RUST_LOG=debug CHAT_ADDR=127.0.0.1:0 ./target/debug/chat > /tmp/chat-out.txt 2>/tmp/chat-err.txt &\nsleep 1\nADDR=$(grep -o '127.0.0.1:[0-9]*' /tmp/chat-out.txt)";
        assert_eq!(bash_repeat_key(a), Some("detached-binary-probe"));
        assert_eq!(bash_repeat_key(a), bash_repeat_key(b));
        assert_eq!(bash_repeat_key("cargo test --offline"), None);
        assert_eq!(bash_repeat_key("sleep 1 & wait"), None);
        let cargo_run = "CHAT_ADDR=127.0.0.1:0 cargo run >/tmp/srv.log 2>/tmp/srv.err &\nSRV=$!\nsleep 2\ncat /tmp/srv.log";
        assert_eq!(bash_repeat_key(cargo_run), Some("detached-binary-probe"));
        let py = r#"cat > /tmp/probe.py <<'EOF'
import subprocess, time
p = subprocess.Popen(["./target/debug/chat"], stdout=subprocess.PIPE)
time.sleep(1.2)
EOF
python3 /tmp/probe.py"#;
        assert_eq!(bash_repeat_key(py), Some("detached-binary-probe"));
        assert!(is_probe_refusal(
            "This detached `target/debug/… & sleep …` probe already ran this turn (3 times)."
        ));
        assert!(!is_probe_refusal("cargo test ok"));
    }

    #[test]
    fn third_detached_probe_is_refused() {
        let ledger = Mutex::new(HashMap::new());
        let cmd =
            "cd /Users/david/chat && ./target/debug/chat > /tmp/out.txt &\nsleep 1\necho PORT=1";
        assert!(admit_bash_repeat(&ledger, cmd).is_none());
        assert!(admit_bash_repeat(&ledger, cmd).is_none());
        let refused = admit_bash_repeat(&ledger, cmd).expect("third probe");
        assert!(refused.contains("already ran this turn"), "{refused}");
        reset_bash_repeats(&ledger);
        assert!(admit_bash_repeat(&ledger, cmd).is_none());
    }
}
