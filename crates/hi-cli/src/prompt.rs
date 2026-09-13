//! One-shot prompt assembly (stdin folding).

use anyhow::Result;

use crate::config::Cli;

/// The one-shot prompt, with piped stdin folded in as context when present.
pub(crate) fn effective_prompt(cli: &Cli) -> Result<Option<String>> {
    use std::io::IsTerminal;
    let Some(prompt) = cli.prompt.clone() else {
        return Ok(None);
    };
    if std::io::stdin().is_terminal() {
        return Ok(Some(prompt));
    }
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        use std::io::Read;
        let mut stdin = std::io::stdin().lock();
        let mut first = [0u8; 1];
        let piped = match stdin.read(&mut first) {
            Ok(0) | Err(_) => Vec::new(),
            Ok(n) => {
                let mut bytes = first[..n].to_vec();
                let _ = stdin.read_to_end(&mut bytes);
                bytes
            }
        };
        let _ = tx.send(piped);
    });
    let piped = match rx.recv_timeout(std::time::Duration::from_secs(2)) {
        Ok(bytes) => String::from_utf8_lossy(&bytes).into_owned(),
        Err(_) => {
            eprintln!(
                "\x1b[2mstdin is open but silent — proceeding without piped input \
                 (pipe data promptly or redirect stdin from /dev/null)\x1b[0m"
            );
            return Ok(Some(prompt));
        }
    };
    let piped = piped.trim();
    if piped.is_empty() {
        return Ok(Some(prompt));
    }
    Ok(Some(format!("{prompt}\n\nstdin:\n```\n{piped}\n```")))
}
