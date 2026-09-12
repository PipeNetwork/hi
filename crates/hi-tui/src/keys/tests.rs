use super::*;

fn key(code: KeyCode, mods: KeyModifiers) -> KeyEvent {
    KeyEvent::new(code, mods)
}

#[test]
fn help_sections_are_non_empty() {
    let rows = help_overlay_rows();
    assert!(
        rows.len() > 10,
        "expected a full cheat sheet, got {}",
        rows.len()
    );
    for ctx in HELP_SECTIONS {
        assert!(
            rows.iter().any(|(k, h)| *k == ctx.title() && h.is_none()),
            "missing section {}",
            ctx.title()
        );
    }
}

#[test]
fn every_in_help_binding_has_nonempty_keys() {
    for b in KEY_BINDINGS.iter().filter(|b| b.in_help) {
        assert!(!b.keys.is_empty());
        assert!(!b.help.is_empty());
    }
}

#[test]
fn every_action_binding_has_matches_or_is_slash_command() {
    for b in KEY_BINDINGS {
        if b.action.is_none() {
            assert!(
                b.matches.is_empty(),
                "help-only binding {} should not list matches",
                b.keys
            );
            continue;
        }
        // Slash commands and Esc-enter-normal are action-tagged for docs but
        // have no physical matches in the table.
        if b.matches.is_empty() {
            assert!(
                b.keys.starts_with('/') || b.keys.starts_with("Esc"),
                "action binding {} needs matches or a slash/Esc exception",
                b.keys
            );
        }
    }
}

#[test]
fn table_resolves_insert_globals() {
    assert_eq!(
        resolve_from_table(
            KeySurface::Insert,
            &key(KeyCode::Char('d'), KeyModifiers::CONTROL)
        ),
        Action::ToggleDiff
    );
    assert_eq!(
        resolve_from_table(
            KeySurface::Insert,
            &key(KeyCode::Char('k'), KeyModifiers::CONTROL)
        ),
        Action::OpenPalette
    );
    assert_eq!(
        resolve_from_table(
            KeySurface::Insert,
            &key(KeyCode::Char('n'), KeyModifiers::CONTROL)
        ),
        Action::JumpPrompt { dir: 1 }
    );
    assert_eq!(
        resolve_from_table(
            KeySurface::Insert,
            &key(KeyCode::Char('f'), KeyModifiers::CONTROL)
        ),
        Action::OpenBlockViewer
    );
    assert_eq!(
        resolve_from_table(
            KeySurface::Normal,
            &key(KeyCode::Char('f'), KeyModifiers::CONTROL)
        ),
        Action::OpenBlockViewer
    );
    assert_eq!(
        resolve_from_table(
            KeySurface::Normal,
            &key(KeyCode::Char('n'), KeyModifiers::CONTROL)
        ),
        Action::None,
        "Ctrl-N is insert-only"
    );
}

#[test]
fn table_resolves_review_and_block_nav() {
    assert_eq!(
        resolve_from_table(KeySurface::Review, &key(KeyCode::Esc, KeyModifiers::NONE)),
        Action::ReviewUnfocus
    );
    assert_eq!(
        resolve_from_table(
            KeySurface::Review,
            &key(KeyCode::Char('n'), KeyModifiers::NONE)
        ),
        Action::ReviewHunk { dir: 1 }
    );
    assert_eq!(
        resolve_from_table(
            KeySurface::BlockNav,
            &key(KeyCode::Enter, KeyModifiers::NONE)
        ),
        Action::BlockNavToggle
    );
    assert_eq!(
        resolve_from_table(
            KeySurface::BlockNav,
            &key(KeyCode::Char('k'), KeyModifiers::NONE)
        ),
        Action::BlockNavUp
    );
    assert_eq!(
        resolve_from_table(
            KeySurface::BlockNav,
            &key(KeyCode::Char('f'), KeyModifiers::CONTROL)
        ),
        Action::OpenBlockViewer
    );
}

#[test]
fn overlay_and_history_search_swallow() {
    assert_eq!(
        resolve_from_table(
            KeySurface::Overlay,
            &key(KeyCode::Char('d'), KeyModifiers::CONTROL)
        ),
        Action::None
    );
    assert_eq!(
        resolve_from_table(
            KeySurface::HistorySearch,
            &key(KeyCode::Char('d'), KeyModifiers::CONTROL)
        ),
        Action::None
    );
}

#[test]
fn action_chords_are_reachable_from_table() {
    // Every physical match row must resolve on an allowed surface.
    for b in KEY_BINDINGS.iter().filter(|b| !b.matches.is_empty()) {
        let surfaces = b.context.surfaces();
        assert!(
            !surfaces.is_empty(),
            "binding {} has matches but no surfaces",
            b.keys
        );
        for m in b.matches {
            let surface = m.only_surface.unwrap_or(surfaces[0]);
            let mut mods = KeyModifiers::NONE;
            if m.ctrl {
                mods |= KeyModifiers::CONTROL;
            }
            if m.alt {
                mods |= KeyModifiers::ALT;
            }
            if m.shift {
                mods |= KeyModifiers::SHIFT;
            }
            let got = resolve_from_table(surface, &key(m.code, mods));
            assert_ne!(
                got,
                Action::None,
                "binding {} match {:?} did not resolve on {:?}",
                b.keys,
                m.code,
                surface
            );
        }
    }
}
