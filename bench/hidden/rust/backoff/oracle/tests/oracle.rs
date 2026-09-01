use subject::Backoff;

#[test]
fn exponential_growth_is_capped_and_saturating() {
    let policy = Backoff::new(100, 1_000);
    assert_eq!(policy.delay_ms(0), 100);
    assert_eq!(policy.delay_ms(1), 200);
    assert_eq!(policy.delay_ms(3), 800);
    assert_eq!(policy.delay_ms(4), 1_000);
    assert_eq!(policy.delay_ms(100), 1_000);
    assert_eq!(Backoff::new(u64::MAX, 50).delay_ms(1), 50);
}
