//! `hi-sentinel` binary. Unix supervisor; other platforms exit 2.

fn main() {
    #[cfg(not(unix))]
    {
        eprintln!("hi-sentinel is Unix-only");
        std::process::exit(2);
    }
    #[cfg(unix)]
    {
        let code = match hi_sentinel::run() {
            Ok(code) => code,
            Err(err) => {
                eprintln!("hi-sentinel: {err:#}");
                1
            }
        };
        std::process::exit(code);
    }
}
