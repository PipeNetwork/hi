//! `hi browser install` — write the unpacked Chrome debugger extension.

use anyhow::{Result, bail};

pub fn run_cli(args: &[String]) -> Result<()> {
    match args.first().map(String::as_str) {
        Some("help") | Some("--help") | Some("-h") => {
            print_usage();
            Ok(())
        }
        Some("install") | None => {
            let dir = hi_tools::install_browser_extension()?;
            println!("Wrote unpacked extension to {}", dir.display());
            println!("Load it in Chrome: chrome://extensions → Developer mode → Load unpacked.");
            println!(
                "Note: browser_exec live attach is disabled because an already-running target \
                 cannot be guarded before it starts network activity; use headless mode."
            );
            Ok(())
        }
        Some(other) => {
            bail!("unknown `hi browser` command '{other}'. Try `hi browser install`.")
        }
    }
}

fn print_usage() {
    println!(
        "\
Usage: hi browser install

Write legacy unpacked MV3 Chrome-extension assets under
~/.config/hi/browser-extension/. Live attach is disabled; browser_exec uses a
fresh guarded headless Chrome. Set [browser] enabled = false in hi.toml to hide
it. Computer-use / AppleScript are not supported."
    );
}
