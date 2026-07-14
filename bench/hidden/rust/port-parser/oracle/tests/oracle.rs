use subject::{PortError, parse_port};

#[test]
fn classifies_input_and_preserves_public_errors() {
    assert_eq!(parse_port(" 443 "), Ok(443));
    assert_eq!(parse_port(""), Err(PortError::Empty));
    assert_eq!(parse_port("  "), Err(PortError::Empty));
    assert_eq!(parse_port("12x"), Err(PortError::Invalid));
    assert_eq!(parse_port("0"), Err(PortError::OutOfRange));
    assert_eq!(parse_port("65536"), Err(PortError::OutOfRange));
    assert_eq!(parse_port("999999999999999999999"), Err(PortError::OutOfRange));
}
