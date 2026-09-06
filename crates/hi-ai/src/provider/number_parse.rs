pub(super) fn largest_number(text: &str) -> Option<u32> {
    let mut best = None;
    let mut current = String::new();
    for ch in text.chars().chain(std::iter::once(' ')) {
        if ch.is_ascii_digit() {
            current.push(ch);
        } else if matches!(ch, ',' | '_') && !current.is_empty() {
            continue;
        } else if !current.is_empty() {
            if let Ok(value) = current.parse::<u32>() {
                best = Some(best.map_or(value, |prev: u32| prev.max(value)));
            }
            current.clear();
        }
    }
    best
}
