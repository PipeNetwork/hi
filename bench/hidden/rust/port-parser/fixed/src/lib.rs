#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PortError {
    Empty,
    Invalid,
    OutOfRange,
}

pub fn parse_port(input: &str) -> Result<u16, PortError> {
    let input = input.trim();
    if input.is_empty() {
        return Err(PortError::Empty);
    }
    if !input.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err(PortError::Invalid);
    }
    let value = input.parse::<u64>().map_err(|_| PortError::OutOfRange)?;
    if !(1..=u16::MAX as u64).contains(&value) {
        return Err(PortError::OutOfRange);
    }
    Ok(value as u16)
}
