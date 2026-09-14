/// Mask an API key (or env var name) for display: first and last four
/// characters with an ellipsis. Char-based, so a key containing multi-byte
/// characters (e.g. pasted with a stray curly quote) can't panic a byte slice.
pub fn mask_key(key: &str) -> String {
    if key.is_empty() {
        return "(none)".to_string();
    }
    let chars: Vec<char> = key.chars().collect();
    if chars.len() > 8 {
        let head: String = chars[..4].iter().collect();
        let tail: String = chars[chars.len() - 4..].iter().collect();
        format!("{head}…{tail}")
    } else {
        "***".to_string()
    }
}
