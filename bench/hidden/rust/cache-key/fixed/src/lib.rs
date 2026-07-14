mod normalize;

pub fn cache_key(namespace: &str, key: &str) -> String {
    format!(
        "{}:{}",
        normalize::component(namespace),
        normalize::component(key)
    )
}
