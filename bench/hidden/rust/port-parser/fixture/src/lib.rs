#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PortError {
    Empty,
    Invalid,
    OutOfRange,
}

pub fn parse_port(input: &str) -> Result<u16, PortError> {
    input.parse::<u16>().map_err(|_| PortError::Invalid)
}
