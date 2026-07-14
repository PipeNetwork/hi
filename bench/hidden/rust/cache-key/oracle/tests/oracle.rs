use subject::cache_key;

#[test]
fn normalizes_both_components_without_changing_the_api() {
    assert_eq!(cache_key(" Users ", " Alice "), "users:alice");
    assert_eq!(cache_key("API", "V1/Thing"), "api:v1/thing");
    assert_eq!(cache_key("x", "y"), "x:y");
}
