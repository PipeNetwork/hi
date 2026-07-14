mod normalize;

pub fn cache_key(namespace: &str, key: &str) -> String {
    format!("{}/{}", namespace, normalize::component(key))
}
