//! Glob-based tool name matching for hooks.

use regex::Regex;

/// A compiled matcher for tool names. Uses glob-to-regex conversion so
/// `Bash` matches `run_terminal_command` is NOT the default — the pattern
/// must explicitly match. `None` means match-all.
#[derive(Debug, Clone)]
pub struct HookMatcher {
    regex: Regex,
}

impl HookMatcher {
    /// Compile a glob pattern into a regex. Supports `*` (any chars) and `?`
    /// (single char). The pattern is case-insensitive.
    pub fn new(pattern: &str) -> Result<Self, String> {
        let regex_str = glob_to_regex(pattern);
        let regex = Regex::new(&format!("(?i)^{regex_str}$"))
            .map_err(|e| format!("invalid regex from pattern '{pattern}': {e}"))?;
        Ok(Self { regex })
    }

    /// Check if a tool name matches this pattern.
    pub fn is_match(&self, tool_name: &str) -> bool {
        self.regex.is_match(tool_name)
    }
}

/// Convert a glob pattern to a regex string. `*` → `.*`, `?` → `.`,
/// other regex metacharacters are escaped.
fn glob_to_regex(pattern: &str) -> String {
    let mut out = String::with_capacity(pattern.len() * 2);
    for ch in pattern.chars() {
        match ch {
            '*' => out.push_str(".*"),
            '?' => out.push('.'),
            c if "\\^$.|+()[]{}".contains(c) => {
                out.push('\\');
                out.push(c);
            }
            c => out.push(c),
        }
    }
    out
}

/// Check if a matcher allows a given tool name. `None` = match-all.
pub fn matcher_allows(matcher: Option<&HookMatcher>, tool_name: Option<&str>) -> bool {
    match matcher {
        None => true,
        Some(m) => match tool_name {
            None => false,
            Some(name) => m.is_match(name),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn match_all_when_no_pattern() {
        assert!(matcher_allows(None, Some("bash")));
        assert!(matcher_allows(None, None));
    }

    #[test]
    fn exact_match() {
        let m = HookMatcher::new("bash").unwrap();
        assert!(m.is_match("bash"));
        assert!(m.is_match("Bash")); // case-insensitive
        assert!(!m.is_match("read"));
    }

    #[test]
    fn glob_match() {
        let m = HookMatcher::new("bash*").unwrap();
        assert!(m.is_match("bash"));
        assert!(m.is_match("bash_output"));
        assert!(!m.is_match("read"));
    }

    #[test]
    fn question_mark_match() {
        let m = HookMatcher::new("bas?").unwrap();
        assert!(m.is_match("bash"));
        assert!(!m.is_match("bash_output"));
    }
}
