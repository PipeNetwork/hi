//! Mini-language: one command per line (`goto`, `click`, `type`, `screenshot`,
//! `ax`, `wait`, `eval`, `scroll`).

use anyhow::{Result, bail};

#[derive(Clone, Debug, PartialEq)]
pub enum BrowserCommand {
    Goto {
        url: String,
    },
    Click {
        target: ClickTarget,
    },
    Type {
        text: String,
        target: Option<ClickTarget>,
    },
    Screenshot,
    Ax,
    Wait {
        millis: u64,
    },
    Eval {
        expression: String,
    },
    Scroll {
        dx: i64,
        dy: i64,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub enum ClickTarget {
    Index(u32),
    Point { x: f64, y: f64 },
}

pub fn parse_script(script: &str) -> Result<Vec<BrowserCommand>> {
    let mut out = Vec::new();
    for (i, raw) in script.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        match parse_line(line) {
            Ok(cmd) => out.push(cmd),
            Err(err) => bail!("line {}: {err}", i + 1),
        }
    }
    if out.is_empty() {
        bail!("empty browser script");
    }
    Ok(out)
}

fn parse_line(line: &str) -> Result<BrowserCommand> {
    let (op, rest) = match line.split_once(char::is_whitespace) {
        Some((op, rest)) => (op, rest.trim()),
        None => (line, ""),
    };
    match op.to_ascii_lowercase().as_str() {
        "goto" | "open" | "navigate" => {
            if rest.is_empty() {
                bail!("goto requires a URL");
            }
            Ok(BrowserCommand::Goto {
                url: rest.to_string(),
            })
        }
        "click" => Ok(BrowserCommand::Click {
            target: parse_target(rest)?,
        }),
        "type" => {
            if rest.is_empty() {
                bail!("type requires text");
            }
            if let Some((target, text)) = rest.split_once(char::is_whitespace)
                && target
                    .chars()
                    .all(|c| c.is_ascii_digit() || c == ',' || c == '.')
                && let Ok(parsed) = parse_target(target)
            {
                return Ok(BrowserCommand::Type {
                    text: text.to_string(),
                    target: Some(parsed),
                });
            }
            Ok(BrowserCommand::Type {
                text: rest.to_string(),
                target: None,
            })
        }
        "screenshot" | "shot" => Ok(BrowserCommand::Screenshot),
        "ax" | "accessibility" => Ok(BrowserCommand::Ax),
        "wait" => {
            let millis: u64 = rest
                .parse()
                .map_err(|_| anyhow::anyhow!("wait requires milliseconds"))?;
            Ok(BrowserCommand::Wait { millis })
        }
        "eval" | "js" => {
            if rest.is_empty() {
                bail!("eval requires a JavaScript expression");
            }
            Ok(BrowserCommand::Eval {
                expression: rest.to_string(),
            })
        }
        "scroll" => {
            let mut nums = rest.split_whitespace();
            let dx: i64 = nums
                .next()
                .ok_or_else(|| anyhow::anyhow!("scroll requires dx dy"))?
                .parse()
                .map_err(|_| anyhow::anyhow!("scroll dx must be an integer"))?;
            let dy: i64 = nums
                .next()
                .unwrap_or("0")
                .parse()
                .map_err(|_| anyhow::anyhow!("scroll dy must be an integer"))?;
            Ok(BrowserCommand::Scroll { dx, dy })
        }
        other => bail!("unknown browser command '{other}'"),
    }
}

fn parse_target(rest: &str) -> Result<ClickTarget> {
    let rest = rest.trim();
    if rest.is_empty() {
        bail!("click requires an AX index or x y");
    }
    let mut parts = rest.split_whitespace();
    let a = parts.next().unwrap();
    if let Some(b) = parts.next() {
        let x: f64 = a
            .parse()
            .map_err(|_| anyhow::anyhow!("click x must be a number"))?;
        let y: f64 = b
            .parse()
            .map_err(|_| anyhow::anyhow!("click y must be a number"))?;
        return Ok(ClickTarget::Point { x, y });
    }
    let index: u32 = a
        .parse()
        .map_err(|_| anyhow::anyhow!("click target must be an AX index or x y"))?;
    Ok(ClickTarget::Index(index))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_core_ops() {
        let cmds = parse_script(
            "goto https://example.com\nclick 3\ntype 3 hello\nscreenshot\nax\nwait 250\nscroll 0 400\neval document.title\n",
        )
        .unwrap();
        assert!(matches!(&cmds[0], BrowserCommand::Goto { url } if url == "https://example.com"));
        assert!(matches!(
            &cmds[1],
            BrowserCommand::Click {
                target: ClickTarget::Index(3)
            }
        ));
        assert!(matches!(&cmds[3], BrowserCommand::Screenshot));
        assert!(
            matches!(&cmds[7], BrowserCommand::Eval { expression } if expression == "document.title")
        );
    }

    #[test]
    fn rejects_unknown_op() {
        let err = parse_script("explode").unwrap_err().to_string();
        assert!(err.contains("unknown"), "{err}");
    }
}
