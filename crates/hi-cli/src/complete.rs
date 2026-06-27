//! Rustyline helper providing tab-completion for `/`-commands and, for
//! `/provider`, profile names and the `add`/`edit` subcommands.
//!
//! Also serves as the single shared line editor for the whole REPL — the
//! provider add/edit prompts reuse it instead of constructing a fresh
//! `DefaultEditor`, which was the cause of paste not working in those fields
//! (a second editor re-initializes the terminal and can drop bracketed-paste
//! mode negotiated by the first).

use std::cell::RefCell;
use std::rc::Rc;

use rustyline::completion::Completer;
use rustyline::highlight::{Highlighter, MatchingBracketHighlighter};
use rustyline::hint::Hinter;
use rustyline::validate::Validator;
use rustyline::{Context, Helper, Result};

use hi_agent::command::CommandSpec;

/// The concrete editor type used throughout the REPL (helper + default
/// history). Provider add/edit prompts take `&mut` on this so they reuse the
/// same terminal state — constructing a second editor was why paste broke in
/// those fields.
pub type ReplEditor = rustyline::Editor<ReplHelper, rustyline::history::DefaultHistory>;

/// Shared, mutable list of profile names the completer consults. Updated by
/// the REPL before each `readline` so newly added/edited profiles appear
/// immediately.
pub type ProfileNames = Rc<RefCell<Vec<String>>>;

/// Rustyline helper: command + profile-name completion, bracket matching,
/// and history-based hints.
pub struct ReplHelper {
    commands: &'static [CommandSpec],
    profiles: ProfileNames,
    brackets: MatchingBracketHighlighter,
}

impl ReplHelper {
    pub fn new(commands: &'static [CommandSpec], profiles: ProfileNames) -> Self {
        Self {
            commands,
            profiles,
            brackets: MatchingBracketHighlighter::new(),
        }
    }
}

impl Completer for ReplHelper {
    type Candidate = String;

    fn complete(&self, line: &str, pos: usize, _ctx: &Context<'_>) -> Result<(usize, Vec<String>)> {
        // Only complete at end-of-line and for lines beginning with '/'.
        if !line.starts_with('/') || pos != line.len() {
            return Ok((0, Vec::new()));
        }
        let rest = &line[1..]; // strip leading '/'

        // If there is a space, we're completing the *argument* to a command.
        if let Some((cmd, arg)) = rest.split_once(' ') {
            if cmd == "provider" {
                return Ok((
                    1 + cmd.len() + 1,
                    provider_arg_completions(arg, &self.profiles),
                ));
            }
            // No other command has completable args yet.
            return Ok((0, Vec::new()));
        }

        // Completing the command name itself.
        let cands: Vec<String> = self
            .commands
            .iter()
            .map(|c| c.name)
            .filter(|name| name.starts_with(rest))
            .map(|name| format!("/{name}"))
            .collect();
        Ok((0, cands))
    }
}

/// Candidates for the argument slot of `/provider`.
fn provider_arg_completions(arg: &str, profiles: &ProfileNames) -> Vec<String> {
    let mut cands: Vec<String> = Vec::new();
    // Subcommands.
    for sub in &["add", "edit"] {
        if sub.starts_with(arg) {
            cands.push((*sub).to_string());
        }
    }
    // Profile names.
    let names = profiles.borrow();
    for name in names.iter() {
        if name.starts_with(arg) {
            cands.push(name.clone());
        }
    }
    cands.sort();
    cands.dedup();
    cands
}

impl Hinter for ReplHelper {
    type Hint = String;

    fn hint(&self, line: &str, pos: usize, ctx: &Context<'_>) -> Option<String> {
        // Offer the rest of a matching command name as a hint.
        if line.starts_with('/') && pos == line.len() {
            let rest = &line[1..];
            if !rest.contains(' ') {
                for c in self.commands {
                    if c.name.starts_with(rest) && c.name.len() > rest.len() {
                        return Some(c.name[rest.len()..].to_string());
                    }
                }
            }
        }
        let _ = ctx;
        None
    }
}

impl Highlighter for ReplHelper {
    fn highlight<'l>(&self, line: &'l str, pos: usize) -> std::borrow::Cow<'l, str> {
        self.brackets.highlight(line, pos)
    }

    fn highlight_hint<'h>(&self, hint: &'h str) -> std::borrow::Cow<'h, str> {
        // Dim the inline hint so it reads as a suggestion, not typed text.
        std::borrow::Cow::Owned(format!("\x1b[2m{hint}\x1b[0m"))
    }

    fn highlight_char(&self, line: &str, pos: usize, forced: bool) -> bool {
        self.brackets.highlight_char(line, pos, forced)
    }
}

impl Validator for ReplHelper {}

impl Helper for ReplHelper {}
