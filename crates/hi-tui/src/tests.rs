use super::*;
use crate::app::{review_next_hunk, search_transcript};
use ratatui::backend::TestBackend;

mod goal;

fn dump(term: &Terminal<TestBackend>) -> String {
    let buf = term.backend().buffer();
    let mut out = String::new();
    for y in 0..buf.area.height {
        for x in 0..buf.area.width {
            out.push_str(buf[(x, y)].symbol());
        }
        out.push('\n');
    }
    out
}

#[test]
fn confirmation_modal_renders_mutation_details() {
    let mut app = test_app("openai", "gpt-4o");
    app.confirmation = Some(hi_agent::ConfirmationRequest::ShellMutation {
        command: "rm generated.txt".into(),
        cwd: "/workspace".into(),
    });
    let mut term = Terminal::new(TestBackend::new(120, 20)).unwrap();
    term.draw(|frame| app.render(frame)).unwrap();
    let screen = dump(&term);
    assert!(screen.contains("Confirm shell mutation"));
    assert!(screen.contains("rm generated.txt"));
    assert!(screen.contains("y approve"));
    assert!(screen.contains("a always allow"));
}

#[tokio::test]
async fn channel_confirmation_uses_local_response_channel() {
    use hi_agent::Ui;
    let (tx, _events) = tokio::sync::mpsc::unbounded_channel();
    let (confirmations, mut controls) = tokio::sync::mpsc::unbounded_channel();
    let mut ui = crate::event::ChannelUi { tx, confirmations };
    let answer = ui.confirm(hi_agent::ConfirmationRequest::FileEdit {
        path: "src/lib.rs".into(),
        diff: "+safe".into(),
    });
    let control = controls.recv().await.unwrap();
    assert!(matches!(
        control.request,
        hi_agent::ConfirmationRequest::FileEdit { .. }
    ));
    control
        .response
        .send(hi_agent::ConfirmationResult::Approved)
        .unwrap();
    assert_eq!(answer.await, hi_agent::ConfirmationResult::Approved);
}

/// A no-op resolver for tests — `/provider` isn't exercised in unit tests.
fn test_resolver() -> ProfileResolver {
    Box::new(|_name| anyhow::bail!("no profiles in tests"))
}

fn test_saver() -> ProfileSaver {
    Box::new(|_form| anyhow::bail!("no profiles in tests"))
}

fn test_loader() -> ProfileLoader {
    Box::new(|_name| anyhow::bail!("no profiles in tests"))
}

fn test_remover() -> ProfileRemover {
    Box::new(|_name| anyhow::bail!("no profiles in tests"))
}

fn test_mlx_switcher() -> MlxProfileSwitcher {
    Box::new(|_run| anyhow::bail!("no mlx profiles in tests"))
}

#[test]
fn selected_model_persists_to_active_profile() {
    let stored = std::sync::Arc::new(std::sync::Mutex::new(ProfileFormData {
        name: "default".into(),
        provider: "pipenetwork".into(),
        api_key: "test-key".into(),
        store_as_env: false,
        model: "pipe/auto-coder".into(),
        base_url: String::new(),
    }));
    let loader_state = stored.clone();
    let saver_state = stored.clone();
    let loader: ProfileLoader = Box::new(move |name| {
        assert_eq!(name, "default");
        Ok(loader_state.lock().unwrap().clone())
    });
    let saver: ProfileSaver = Box::new(move |data| {
        *saver_state.lock().unwrap() = data.clone();
        Ok(vec![ProfileInfo {
            name: data.name.clone(),
            provider: data.provider.clone(),
            model: Some(data.model.clone()),
            base_url: None,
        }])
    });

    let mut app = App::new(
        "pipenetwork",
        "pipe/auto-coder",
        vec![ProfileInfo {
            name: "default".into(),
            provider: "pipenetwork".into(),
            model: Some("pipe/auto-coder".into()),
            base_url: None,
        }],
        Some("default".into()),
        test_resolver(),
        saver,
        loader,
        test_remover(),
        test_mlx_switcher(),
        None,
        String::new(),
    );

    let saved = app
        .persist_active_profile_model("ipop/coder-balanced")
        .expect("persist selected model");

    assert_eq!(saved.as_deref(), Some("default"));
    assert_eq!(
        stored.lock().unwrap().model,
        "ipop/coder-balanced",
        "profile form was rewritten with selected model"
    );
    assert_eq!(
        app.profiles[0].model.as_deref(),
        Some("ipop/coder-balanced")
    );
}

/// `App::new` with empty profiles and dummy callbacks, for tests.
fn test_app(provider: &str, model: &str) -> App {
    App::new(
        provider,
        model,
        Vec::new(),
        None,
        test_resolver(),
        test_saver(),
        test_loader(),
        test_remover(),
        test_mlx_switcher(),
        None,
        String::new(),
    )
}

#[tokio::test]
async fn sessions_switch_replaces_live_agent_and_ui_session() {
    let provider = std::sync::Arc::new(hi_ai::OpenAiProvider::new(
        "http://127.0.0.1:1/v1".into(),
        "test".into(),
    ));
    let mut agent = hi_agent::Agent::new(provider, hi_agent::AgentConfig::default()).unwrap();
    let mut app = test_app("openai", "gpt-4o");
    app.sync_config = Some(SyncConfig {
        base_url: "http://127.0.0.1:1/v1".into(),
        api_key: "test".into(),
        machine_id: None,
        cwd_digest: None,
    });
    let previous_remote = std::sync::Arc::new(crate::sync_tui::RemoteUi::new(
        crate::sync_tui::SyncConfig {
            base_url: "http://127.0.0.1:1/v1".into(),
            api_key: "test".into(),
        },
        "session-1".into(),
    ));
    app.sync_remote_ui = Some(previous_remote.clone());
    app.push(Line::raw("old transcript"));
    app.session_switcher = Some(Box::new(|id, agent| {
        Box::pin(async move {
            agent.apply_loaded_session(
                vec![
                    hi_ai::Message::system("system"),
                    hi_ai::Message::user("resumed prompt"),
                ],
                hi_ai::Usage::default(),
                Vec::new(),
                None,
                hi_agent::DecisionLog::default(),
                Vec::new(),
            );
            Ok(SessionSwitchInfo {
                id: id.to_string(),
                summary: "1 prior message".into(),
            })
        })
    }));

    app.handle_sessions_command(&mut agent, "switch session-2")
        .await;

    assert_eq!(app.sync_session_id.as_deref(), Some("session-2"));
    assert!(!std::sync::Arc::ptr_eq(
        &previous_remote,
        app.sync_remote_ui.as_ref().unwrap()
    ));
    assert!(
        agent
            .messages()
            .iter()
            .any(|m| m.text() == "resumed prompt")
    );
    let transcript = app.transcript_text();
    assert!(transcript.contains("switched to session session-2"));
    assert!(!transcript.contains("old transcript"));
}

#[tokio::test]
async fn sessions_rename_uses_session_manager_callback() {
    let provider = std::sync::Arc::new(hi_ai::OpenAiProvider::new(
        "http://127.0.0.1:1/v1".into(),
        "test".into(),
    ));
    let mut agent = hi_agent::Agent::new(provider, hi_agent::AgentConfig::default()).unwrap();
    let renamed = std::sync::Arc::new(std::sync::Mutex::new(None));
    let observed = renamed.clone();
    let mut app = test_app("openai", "gpt-4o");
    app.session_lister = Some(Box::new(|| {
        vec![LocalSessionInfo {
            id: "session-2".into(),
            title: "Portal work".into(),
            age: "now".into(),
            lines: 1,
        }]
    }));
    app.session_renamer = Some(Box::new(move |id, name| {
        *observed.lock().unwrap() = Some((id.to_string(), name.to_string()));
        Ok(name.to_string())
    }));

    app.handle_sessions_command(&mut agent, "rename session-2 Portal work")
        .await;

    assert_eq!(
        *renamed.lock().unwrap(),
        Some(("session-2".into(), "Portal work".into()))
    );
    assert!(app.transcript_text().contains("session-2 → Portal work"));
}

#[tokio::test]
async fn sessions_list_uses_one_unified_heading() {
    let provider = std::sync::Arc::new(hi_ai::OpenAiProvider::new(
        "http://127.0.0.1:1/v1".into(),
        "test".into(),
    ));
    let mut agent = hi_agent::Agent::new(provider, hi_agent::AgentConfig::default()).unwrap();
    let mut app = test_app("openai", "gpt-4o");
    app.session_lister = Some(Box::new(|| {
        vec![LocalSessionInfo {
            id: "session-2".into(),
            title: "Portal work".into(),
            age: "now".into(),
            lines: 4,
        }]
    }));

    app.handle_sessions_command(&mut agent, "").await;

    let transcript = app.transcript_text();
    assert!(transcript.contains("sessions (1):"));
    assert!(!transcript.contains("local sessions"));
    assert!(!transcript.contains("remote sessions"));
    assert!(transcript.contains("/sessions switch session-2"));
}

#[tokio::test]
async fn sessions_reject_path_like_ids_before_callbacks_or_http() {
    let provider = std::sync::Arc::new(hi_ai::OpenAiProvider::new(
        "http://127.0.0.1:1/v1".into(),
        "test".into(),
    ));
    let mut agent = hi_agent::Agent::new(provider, hi_agent::AgentConfig::default()).unwrap();
    let mut app = test_app("openai", "gpt-4o");
    app.session_switcher = Some(Box::new(|_, _| {
        Box::pin(async { panic!("invalid id reached switch callback") })
    }));

    app.handle_sessions_command(&mut agent, "switch ../../escape")
        .await;
    app.handle_sessions_command(&mut agent, "rename ../../escape bad")
        .await;

    assert_eq!(
        app.transcript_text().matches("invalid session id").count(),
        2
    );
}

#[test]
fn sticky_scroll_unpins_on_scroll_up_and_repins_at_bottom() {
    let mut app = test_app("openai", "gpt-4o");
    // Simulate what render() caches for a transcript taller than the viewport.
    app.view_max_scroll = 100;
    app.view_total = 120;
    assert!(app.following, "starts pinned to the bottom");

    // Scrolling up unpins, holds an absolute offset, and snapshots the count.
    app.scroll_up(10);
    assert!(!app.following, "scroll up unpins");
    assert_eq!(app.scroll, 90, "offset = max_scroll - 10");
    assert_eq!(app.total_when_unpinned, 120);

    // Streaming output below must NOT yank a scrolled-up reader back down.
    app.apply(UiEvent::Text {
        text: "a fresh streamed line\n".into(),
    });
    assert!(
        !app.following,
        "new output leaves the scrolled-up reader put"
    );

    // Scrolling back past the bottom re-pins so output follows again.
    app.scroll_down(1000);
    assert!(app.following, "reaching the bottom re-pins");
}

#[test]
fn mouse_wheel_scrolls_and_repins_the_transcript() {
    let mut app = test_app("openai", "gpt-4o");
    app.view_max_scroll = 30;
    app.view_total = 50;

    let wheel = |kind| crossterm::event::MouseEvent {
        kind,
        column: 0,
        row: 0,
        modifiers: KeyModifiers::NONE,
    };
    app.handle_mouse(wheel(crossterm::event::MouseEventKind::ScrollUp));
    assert!(!app.following);
    assert_eq!(app.scroll, 27);

    app.handle_mouse(wheel(crossterm::event::MouseEventKind::ScrollDown));
    assert!(app.following, "wheel-down at the bottom should re-pin");
}

#[tokio::test]
async fn config_command_sets_disables_and_restores_automatic_step_limit() {
    let provider = std::sync::Arc::new(hi_ai::OpenAiProvider::new(
        "http://127.0.0.1:1/v1".into(),
        "test".into(),
    ));
    let mut agent = hi_agent::Agent::new(provider, hi_agent::AgentConfig::default()).unwrap();
    let mut app = test_app("openai", "gpt-4o");
    assert_eq!(agent.max_steps_setting(), "auto");

    app.handle_command(&mut agent, hi_agent::Command::Config("steps 350".into()))
        .await;
    assert_eq!(agent.max_steps_setting(), "350");

    app.handle_command(&mut agent, hi_agent::Command::Config("steps off".into()))
        .await;
    assert_eq!(agent.max_steps_setting(), "off");

    app.handle_command(&mut agent, hi_agent::Command::Config("steps auto".into()))
        .await;
    assert_eq!(agent.max_steps_setting(), "auto");
    assert!(app.transcript_text().contains("step limit → auto"));
}

#[test]
fn transcript_is_capped_while_following_but_not_while_scrolled_up() {
    let mut app = test_app("openai", "gpt-4o");
    // Following stays bounded and keeps the newest lines.
    for i in 0..(MAX_TRANSCRIPT_LINES + 5_000) {
        app.push(Line::raw(format!("l{i}")));
    }
    assert_eq!(
        app.transcript.len(),
        MAX_TRANSCRIPT_LINES,
        "bounded while following"
    );
    assert_eq!(
        app.transcript.last().unwrap().text(),
        format!("l{}", MAX_TRANSCRIPT_LINES + 5_000 - 1),
        "newest line kept"
    );

    // Scrolled-up pushes are not trimmed because that would shift reader offsets.
    app.view_max_scroll = 50;
    app.view_total = 60;
    app.scroll_up(5);
    assert!(!app.following, "scrolled up");
    let before = app.transcript.len();
    for i in 0..1_000 {
        app.push(Line::raw(format!("m{i}")));
    }
    assert_eq!(
        app.transcript.len(),
        before + 1_000,
        "grows while scrolled up, no trim"
    );
}

#[test]
fn scrolling_moves_the_viewport_through_render_and_repins() {
    let mut app = test_app("openai", "gpt-4o");
    for i in 0..100 {
        app.push(Line::raw(format!("line {i:03}")));
    }
    let mut term = Terminal::new(TestBackend::new(40, 12)).unwrap();
    // Following: the bottom is visible, the top is not.
    term.draw(|f| app.render(f)).unwrap();
    let screen = dump(&term);
    assert!(
        screen.contains("line 099"),
        "bottom visible when following:\n{screen}"
    );
    assert!(
        !screen.contains("line 000"),
        "top hidden when following:\n{screen}"
    );

    // Scroll up: earlier lines appear, the bottom leaves the viewport.
    app.scroll_up(40);
    term.draw(|f| app.render(f)).unwrap();
    let screen = dump(&term);
    assert!(!app.following, "scroll up unpins");
    assert!(
        !screen.contains("line 099"),
        "bottom gone after scroll up:\n{screen}"
    );
    assert!(
        screen.contains("line 0"),
        "older lines now visible:\n{screen}"
    );

    // Scroll back down past the end: re-pins and shows the bottom again.
    app.scroll_down(1000);
    term.draw(|f| app.render(f)).unwrap();
    let screen = dump(&term);
    assert!(app.following, "re-pinned at the bottom");
    assert!(
        screen.contains("line 099"),
        "bottom visible again:\n{screen}"
    );
}

#[test]
fn following_shows_last_line_when_word_wrapping_creates_extra_rows() {
    // Regression: `wrapped_height` counted characters (ceil(len/width)) but
    // ratatui's `WordWrapper` wraps at word boundaries — a word that doesn't
    // fit the remaining space moves to the next line, and a word wider than
    // the line is broken across rows. That makes the real wrapped height
    // LARGER than the char-based estimate, so `max_scroll` was too small
    // and the bottom of a long message was clipped off-screen.
    //
    // Each line below has a 45-char word at width 38: ratatui wraps it to
    // 3 rows, but the old char-based estimate said 2. With 20 such lines
    // the ~20-row undercount pushed the last line entirely off-screen.
    let mut app = test_app("openai", "gpt-4o");
    for i in 0..20 {
        app.push(Line::raw(format!(
            "word{i:02} supercalifragilisticexpialidocious_extras"
        )));
    }
    app.push(Line::raw("LAST_LINE_MARKER_42"));

    let mut term = Terminal::new(TestBackend::new(40, 12)).unwrap();
    term.draw(|f| app.render(f)).unwrap();
    let screen = dump(&term);
    assert!(
        screen.contains("LAST_LINE_MARKER_42"),
        "last line must be visible when following (word-wrap clip bug):\n{screen}"
    );
}

#[test]
fn following_shows_last_line_with_realistic_prose_word_wrapping() {
    // A second regression check with normal prose (no artificially long
    // words). At a narrow width, word-boundary wrapping still produces more
    // rows than char-based `ceil(len/width)` because words that don't fit
    // the remaining space leave the current line short. This is the case
    // that clipped the end of a long assistant message in practice.
    let mut app = test_app("openai", "gpt-4o");
    // 30 lines of prose, each ~70 chars. At width 36 (inner of a 38-wide
    // terminal), char-based says ceil(70/36) = 2 rows per line, but
    // word-wrap often produces 3 because words straddle the boundary.
    for i in 0..30 {
        app.push(Line::raw(format!(
            "The quick brown fox jumps over the lazy dog and then runs {i:02}"
        )));
    }
    app.push(Line::raw("FINAL_ANSWER_99"));

    let mut term = Terminal::new(TestBackend::new(38, 14)).unwrap();
    term.draw(|f| app.render(f)).unwrap();
    let screen = dump(&term);
    assert!(
        screen.contains("FINAL_ANSWER_99"),
        "last line must be visible with prose word-wrapping:\n{screen}"
    );
}

#[test]
fn working_line_names_the_inflight_tool_and_model_phase() {
    let mut app = test_app("openai", "gpt-4o");
    app.set_working(true);
    // Model phase: reasoning then text stream distinctly.
    app.apply(UiEvent::Reasoning { text: "hmm".into() });
    assert!(
        app.activity_line().starts_with("thinking…"),
        "{}",
        app.activity_line()
    );
    app.apply(UiEvent::Text {
        text: "here".into(),
    });
    assert!(
        app.activity_line().starts_with("responding…"),
        "{}",
        app.activity_line()
    );
    // A tool starts → the line names it (with its own timer)…
    app.apply(UiEvent::ToolStarted {
        name: "bash".into(),
        arguments: "{\"command\":\"cargo test\"}".into(),
    });
    assert!(
        app.activity_line().starts_with("running bash cargo test"),
        "{}",
        app.activity_line()
    );
    // …and clears back to the model once the result lands.
    app.apply(UiEvent::ToolResult {
        name: "bash".into(),
        result: "ok".into(),
    });
    assert!(
        app.activity_line().starts_with("Working"),
        "{}",
        app.activity_line()
    );
}

#[test]
fn working_wave_sweeps_one_lit_letter_at_a_time() {
    // "Working" is 7 letters; the lit index sweeps 0..6 then 6..0.
    // At each tick exactly one letter is white/bold and the rest are gray.
    let mut app = test_app("openai", "gpt-4o");
    app.set_working(true);
    let n = "Working".chars().count();
    let cycle = 2 * (n - 1);
    for tick in 0..cycle {
        app.spinner = tick;
        let spans = app.working_spans();
        assert_eq!(spans.len(), n, "one span per letter at tick {tick}");
        let lit_count = spans
            .iter()
            .filter(|s| s.style.fg == Some(crate::theme::theme().accent_running))
            .count();
        assert_eq!(lit_count, 1, "exactly one lit letter at tick {tick}");
        // The lit index matches the forward/back sweep.
        let expected_lit = if tick < n { tick } else { cycle - tick };
        assert_eq!(
            spans[expected_lit].style.fg,
            Some(crate::theme::theme().accent_running),
            "lit index {expected_lit} at tick {tick}"
        );
    }
}

#[test]
fn renders_tool_call_diff_and_spinner() {
    let mut app = test_app("openai", "gpt-4o");
    app.apply(UiEvent::ToolCall {
        name: "edit".into(),
        arguments: "{\"path\":\"src/cli.rs\",\"old_string\":\"a\",\"new_string\":\"b\"}".into(),
    });
    // ANSI-colored diff line (from the edit tool) must render as text.
    app.apply(UiEvent::ToolResult {
        name: "edit".into(),
        result: "\u{1b}[32m+ pub json: bool\u{1b}[0m".into(),
    });
    app.apply(UiEvent::TurnEnd {
        summary: "[1234 in · 56 out · 1290 total]".into(),
    });
    app.working = true;
    app.spinner = 2;

    let mut term = Terminal::new(TestBackend::new(56, 13)).unwrap();
    term.draw(|f| app.render(f)).unwrap();
    let screen = dump(&term);

    // The header reads as "edit <path>", not a raw JSON dump.
    assert!(screen.contains("◆ edit src/cli.rs"), "readable tool header");
    assert!(
        !screen.contains("old_string"),
        "header must not dump JSON args"
    );
    assert!(
        screen.contains("pub json: bool"),
        "ANSI diff rendered as text"
    );
    assert!(screen.contains("1290 total"), "status bar shows usage");
    assert!(
        screen.contains(SPINNER[2]) && screen.contains("0s"),
        "prompt bar shows the spinner + an elapsed timer while working: {screen}"
    );
    assert!(
        screen.contains("Ctrl-C to interrupt"),
        "prompt bar shows the interrupt hint while working"
    );
}

#[test]
fn colorizes_plain_diff_tool_output() {
    let mut app = test_app("openai", "gpt-4o");
    let diff = "--- a/x.rs\n+++ b/x.rs\n@@ -1,2 +1,2 @@\n-old\n+new\n ctx\n";
    app.apply(UiEvent::ToolResult {
        name: "bash".into(),
        result: diff.into(),
    });
    // The content span (after the "  " indent) carries the diff color.
    let colored: Vec<(String, Option<Color>)> = app
        .transcript
        .iter()
        .flat_map(|e| e.flatten(false, true))
        .map(|l| {
            let text: String = l.spans.iter().map(|s| s.content.as_ref()).collect();
            (text, l.spans.last().map(|s| s.style.fg).unwrap_or(None))
        })
        .collect();
    assert!(
        colored
            .iter()
            .any(|(t, fg)| t.contains("+new") && *fg == Some(crate::theme::theme().diff_add)),
        "added line is green: {colored:?}"
    );
    assert!(
        colored
            .iter()
            .any(|(t, fg)| t.contains("-old") && *fg == Some(crate::theme::theme().diff_del)),
        "removed line is red"
    );
    assert!(
        colored
            .iter()
            .any(|(t, fg)| t.contains("@@") && *fg == Some(crate::theme::theme().diff_hunk)),
        "hunk header is cyan"
    );
}

#[test]
fn non_diff_tool_output_is_not_colorized() {
    let mut app = test_app("openai", "gpt-4o");
    app.apply(UiEvent::ToolResult {
        name: "bash".into(),
        result: "- item one\n- item two\n".into(),
    });
    let any_red = app
        .transcript
        .iter()
        .flat_map(|e| e.flatten(false, true))
        .any(|l| l.spans.last().map(|s| s.style.fg) == Some(Some(Color::Red)));
    assert!(!any_red, "a plain list must not be colorized as a diff");
}

#[test]
fn usage_event_keeps_tokens_out_of_compact_working_line() {
    let mut app = test_app("openai", "gpt-4o");
    app.set_working(true);
    app.apply(UiEvent::Usage {
        prompt: 12,
        generated: 340,
        ctx_used: 64_000,
        ctx_window: Some(128_000),
        estimated: false,
    });
    assert_eq!(app.usage, (12, 340));
    assert_eq!(app.context_pct(), Some(50));

    let mut term = Terminal::new(TestBackend::new(72, 8)).unwrap();
    term.draw(|f| app.render(f)).unwrap();
    let screen = dump(&term);
    assert!(screen.contains(SPINNER[0]), "spinner shown: {screen}");
    assert!(
        !screen.contains("prompt↑"),
        "no duplicate prompt tokens: {screen}"
    );
    assert!(
        !screen.contains("gen↓"),
        "no duplicate output tokens: {screen}"
    );
    assert!(screen.contains("50% ctx"), "live context fill: {screen}");
}

#[test]
fn rate_limit_event_updates_working_line() {
    let mut app = test_app("openai", "gpt-4o");
    app.set_working(true);
    app.apply(UiEvent::RateLimits {
        rate_limits: Some(hi_ai::RateLimitState {
            requests_min: hi_ai::RateLimitBucket {
                limit: 60,
                remaining: 58,
                reset_seconds: 12,
            },
            tokens_min: hi_ai::RateLimitBucket {
                limit: 100_000,
                remaining: 88_000,
                reset_seconds: 42,
            },
            ..Default::default()
        }),
    });

    let mut term = Terminal::new(TestBackend::new(100, 8)).unwrap();
    term.draw(|f| app.render(f)).unwrap();
    let screen = dump(&term);
    assert!(
        screen.contains("limits req 58/60"),
        "request bucket: {screen}"
    );
    assert!(screen.contains("tok 88k/100k"), "token bucket: {screen}");
}

#[test]
fn renders_queued_commands_while_working() {
    let mut app = test_app("openai", "gpt-4o");
    app.set_working(true);
    app.queue.push_back("run the tests".into());
    app.queue.push_back("then commit".into());
    app.input.set("typing a third");

    let mut term = Terminal::new(TestBackend::new(60, 14)).unwrap();
    term.draw(|f| app.render(f)).unwrap();
    let screen = dump(&term);

    assert!(screen.contains(SPINNER[0]), "spinner shown while working");
    assert!(
        screen.contains("run the tests"),
        "first queued command shown"
    );
    assert!(
        screen.contains("then commit"),
        "second queued command shown"
    );
    assert!(
        screen.contains("typing a third"),
        "input stays editable while working"
    );
}

#[test]
fn renders_pinned_plan_checklist() {
    use hi_agent::PlanStep;
    let mut app = test_app("openai", "gpt-4o");
    app.apply(UiEvent::Plan {
        steps: vec![
            PlanStep {
                title: "find leak".into(),
                status: PlanStatus::Done,
            },
            PlanStep {
                title: "fix walkers".into(),
                status: PlanStatus::Active,
            },
            PlanStep {
                title: "add tests".into(),
                status: PlanStatus::Pending,
            },
        ],
    });

    let mut term = Terminal::new(TestBackend::new(60, 20)).unwrap();
    term.draw(|f| app.render(f)).unwrap();
    let screen = dump(&term);

    assert!(
        screen.contains("plan · 1/3"),
        "plan header w/ progress:\n{screen}"
    );
    assert!(screen.contains("find leak"), "step titles shown:\n{screen}");
    assert!(screen.contains("fix walkers"));
    assert!(screen.contains("add tests"));
    assert!(screen.contains('✓'), "done glyph:\n{screen}");
    assert!(screen.contains('▸'), "active glyph:\n{screen}");

    // A later update replaces the plan in place — progress advances and the
    // checklist isn't duplicated into the transcript.
    app.apply(UiEvent::Plan {
        steps: vec![
            PlanStep {
                title: "find leak".into(),
                status: PlanStatus::Done,
            },
            PlanStep {
                title: "fix walkers".into(),
                status: PlanStatus::Done,
            },
            PlanStep {
                title: "add tests".into(),
                status: PlanStatus::Active,
            },
        ],
    });
    term.draw(|f| app.render(f)).unwrap();
    let screen2 = dump(&term);
    assert!(
        screen2.contains("plan · 2/3"),
        "progress advanced:\n{screen2}"
    );
    assert!(
        app.transcript.is_empty(),
        "plan must not echo into the transcript"
    );

    app.apply(UiEvent::Plan { steps: Vec::new() });
    term.draw(|f| app.render(f)).unwrap();
    let screen3 = dump(&term);
    assert!(
        !screen3.contains("plan ·"),
        "empty update clears box:\n{screen3}"
    );
}

#[test]
fn long_plan_does_not_break_input_box_border() {
    // Regression: when the plan + status + input is taller than the screen,
    // the input box height used to exceed the terminal height. ratatui's
    // Layout clamps the fixed-Length rect, so the Paragraph content spilled
    // past the bottom border — the `╰` landed mid-content and later steps
    // rendered outside the box. The box must stay closed and fit on screen.
    use hi_agent::PlanStep;
    let mut app = test_app("openai", "gpt-4o");
    app.apply(UiEvent::Plan {
        steps: (0..8)
            .map(|i| PlanStep {
                title: format!("step {i} with a fairly long title to be realistic"),
                status: if i < 3 {
                    PlanStatus::Done
                } else if i == 3 {
                    PlanStatus::Active
                } else {
                    PlanStatus::Pending
                },
            })
            .collect(),
    });
    app.working = true;
    // Tiny height: the full plan (9 lines) + status + input + borders can't fit.
    let mut term = Terminal::new(TestBackend::new(80, 12)).unwrap();
    term.draw(|f| app.render(f)).unwrap();
    let screen = dump(&term);

    // The bottom border must close on its own row, not overlap a plan step.
    let bottom_rows: Vec<&str> = screen.lines().filter(|l| l.contains('╰')).collect();
    assert_eq!(
        bottom_rows.len(),
        1,
        "exactly one bottom-left corner:\n{screen}"
    );
    // The corner must be the first non-space glyph on its row (a closed
    // border), not sitting on top of a `✓`/`▸`/`☐` step glyph.
    let row = bottom_rows[0];
    assert!(
        row.trim_start().starts_with('╰'),
        "bottom border row must start with `╰`, got: {row:?}\n{screen}"
    );
    // The plan is truncated to fit, with a "… +N more" line, rather than
    // overflowing.
    assert!(
        screen.contains("… +3 more"),
        "plan truncated to fit:\n{screen}"
    );
    // The box never exceeds the terminal height.
    assert!(
        screen.lines().filter(|l| !l.trim().is_empty()).count() <= 12,
        "box fits on screen:\n{screen}"
    );

    // A taller terminal shows the whole plan with no truncation.
    let mut term2 = Terminal::new(TestBackend::new(175, 14)).unwrap();
    term2.draw(|f| app.render(f)).unwrap();
    let screen2 = dump(&term2);
    assert!(
        screen2.contains("step 7 with a fairly long title to be realistic"),
        "full plan shown when it fits:\n{screen2}"
    );
    assert!(!screen2.contains("… +"), "no truncation when it fits");

    // Extreme case: a plan so large the box would fill the whole screen.
    // The transcript must still get its Min(1) row and the border must stay
    // closed — the cap reserves a row for the transcript so Layout never
    // clamps the box rect.
    let mut app2 = test_app("openai", "gpt-4o");
    app2.apply(UiEvent::Plan {
        steps: (0..20)
            .map(|i| PlanStep {
                title: format!("step {i}"),
                status: PlanStatus::Pending,
            })
            .collect(),
    });
    app2.working = true;
    let mut term3 = Terminal::new(TestBackend::new(60, 10)).unwrap();
    term3.draw(|f| app2.render(f)).unwrap();
    let screen3 = dump(&term3);
    let bottom3: Vec<&str> = screen3.lines().filter(|l| l.contains('╰')).collect();
    assert_eq!(bottom3.len(), 1, "one bottom corner:\n{screen3}");
    assert!(
        bottom3[0].trim_start().starts_with('╰'),
        "border closed, not overlapping content:\n{screen3}"
    );
    // The transcript title row survives (top border with `hi ·`).
    assert!(
        screen3.contains("hi · openai · gpt-4o"),
        "transcript keeps its row:\n{screen3}"
    );

    // Degenerate tiny terminal: must not panic, and the box border stays closed.
    let mut term4 = Terminal::new(TestBackend::new(60, 3)).unwrap();
    term4.draw(|f| app2.render(f)).unwrap();
    let screen4 = dump(&term4);
    let bottom4: Vec<&str> = screen4.lines().filter(|l| l.contains('╰')).collect();
    assert_eq!(
        bottom4.len(),
        1,
        "one bottom corner on tiny term:\n{screen4}"
    );
}

#[test]
fn startup_notice_does_not_clip_input_line() {
    // On first load, a startup notice (e.g. "model metadata not loaded: …")
    // is pinned above the status line. The box height must
    // account for it, or the input line gets clipped and the cursor lands
    // on the wrong row.
    let mut app = test_app("openai", "gpt-4o");
    app.startup_notice = Some("model metadata not loaded: connection refused".into());
    let mut term = Terminal::new(TestBackend::new(70, 10)).unwrap();
    term.draw(|f| app.render(f)).unwrap();
    let screen = dump(&term);
    assert!(
        screen.contains("model metadata not loaded"),
        "notice shown:\n{screen}"
    );
    // The input prompt must still be visible inside the box (not clipped).
    assert!(screen.contains('❯'), "input prompt visible:\n{screen}");
    // The input box's bottom border closes cleanly (the transcript block
    // also has a `╰`, so check the last one — the input box is at the bottom).
    let bottom: Vec<&str> = screen.lines().filter(|l| l.contains('╰')).collect();
    let input_box_border = bottom.last().expect("input box bottom border");
    assert!(
        input_box_border.trim_start().starts_with('╰'),
        "input box border closed:\n{screen}"
    );
    // The notice, status, and prompt all render inside the input box —
    // i.e. above the input box's bottom border row (the last `╰…─` row).
    let rows: Vec<&str> = screen.lines().collect();
    let border_row_idx = rows
        .iter()
        .rposition(|l| l.trim_start().starts_with('╰') && l.contains('─'))
        .unwrap();
    let above_border: String = rows[..border_row_idx].join("\n");
    assert!(
        above_border.contains("model metadata not loaded") && above_border.contains('❯'),
        "notice + prompt above the border:\n{screen}"
    );
}

#[test]
fn quit_notice_renders_and_does_not_clip_input() {
    // After the first Ctrl-C (idle, empty input), a "Press Ctrl-C again to
    // exit" notice is pinned above the status line. The box height must
    // account for it or the input line clips and the cursor lands wrong.
    let mut app = test_app("openai", "gpt-4o");
    app.quit_notice = Some(Instant::now() + Duration::from_millis(1800));
    let mut term = Terminal::new(TestBackend::new(70, 10)).unwrap();
    term.draw(|f| app.render(f)).unwrap();
    let screen = dump(&term);
    assert!(
        screen.contains("Press Ctrl-C again to exit"),
        "quit notice shown:\n{screen}"
    );
    assert!(screen.contains('❯'), "input prompt visible:\n{screen}");
    // The input box bottom border closes cleanly (not overlapping content).
    let bottom: Vec<&str> = screen.lines().filter(|l| l.contains('╰')).collect();
    let input_box_border = bottom.last().expect("input box bottom border");
    assert!(
        input_box_border.trim_start().starts_with('╰'),
        "input box border closed:\n{screen}"
    );
}

#[test]
fn long_single_line_input_wraps_and_cursor_tracks_it() {
    // The bug: a long single-line prompt didn't wrap — it ran off the right
    // edge and newly typed text was invisible. The input must soft-wrap to the
    // box width, and the cursor must land on the wrapped row/column where the
    // caret actually is.
    let mut app = test_app("openai", "gpt-4o");
    // 40 chars of text on a 28-wide inner area. prefix = 2, so wrap_w = 26
    // → two display lines: first 26 chars, then the remaining 14.
    let long = "abcdefghijklmnopqrstuvwxyz0123456789abcd";
    app.input.set(long);
    // Cursor at the very end (typing position).
    let (lines, cursor_row, cursor_col) = app.input_view(28);
    // First display line holds the first 26 chars; second holds the rest.
    assert!(
        lines.iter().any(|l| l.to_string().contains("abcdefghij")),
        "first wrapped chunk visible"
    );
    assert!(
        lines.iter().any(|l| l.to_string().contains("abcd")),
        "tail wrapped onto a second line: {:?}",
        lines
    );
    // Cursor is on the second wrapped row (index 1), past its last char.
    assert_eq!(cursor_row, 1, "cursor on wrapped row 1");
    // Second chunk is 14 chars + 2-col prefix → cursor at col 16.
    assert_eq!(cursor_col, 16, "cursor col tracks wrap");
}

#[test]
fn long_input_cursor_in_first_wrapped_chunk_stays_on_row_zero() {
    let mut app = test_app("openai", "gpt-4o");
    let long = "abcdefghijklmnopqrstuvwxyz0123456789abcd";
    app.input.set(long);
    // Move cursor to column 5 (within the first wrapped chunk).
    app.input.cursor = 5;
    let (_lines, cursor_row, cursor_col) = app.input_view(28);
    assert_eq!(cursor_row, 0, "cursor on first wrapped row");
    assert_eq!(cursor_col, 2 + 5, "cursor col = prefix + 5");
}

#[test]
fn keybindings_help_does_not_advertise_idle_escape_or_ctrl_d_quit() {
    let mut app = test_app("openai", "gpt-4o");
    app.show_help = true;
    let mut term = Terminal::new(TestBackend::new(80, 40)).unwrap();
    term.draw(|f| app.render(f)).unwrap();
    let screen = dump(&term);

    assert!(
        screen.contains("Ctrl-D") && screen.contains("toggle the working-tree diff panel"),
        "Ctrl-D help should describe diff toggle:\n{screen}"
    );
    assert!(
        screen.contains("/quit") && screen.contains("quit"),
        "explicit quit command should be shown:\n{screen}"
    );
    assert!(
        !screen.contains("Ctrl-D (idle)") && !screen.contains("quit when empty"),
        "help should not advertise the old accidental-exit bindings:\n{screen}"
    );
}

#[test]
fn changed_files_line_shows_what_last_turn_touched() {
    // After a turn that changed files, a compact "changed: …" line sits
    // above the input so the user sees what was touched without scrolling.
    let mut app = test_app("openai", "gpt-4o");
    app.last_changed_files = vec!["src/a.rs".into(), "src/b.rs".into()];
    let mut term = Terminal::new(TestBackend::new(60, 12)).unwrap();
    term.draw(|f| app.render(f)).unwrap();
    let screen = dump(&term);
    assert!(
        screen.contains("changed: src/a.rs, src/b.rs"),
        "changed-files line: {screen}"
    );
    assert!(
        screen.contains("Ctrl-D for diff"),
        "diff toggle hint: {screen}"
    );
}

#[test]
fn ctrl_d_toggles_diff_even_when_input_is_empty() {
    let mut app = test_app("openai", "gpt-4o");
    let ctrl_d = KeyEvent::new(KeyCode::Char('d'), KeyModifiers::CONTROL);

    assert_eq!(app.edit_key(&ctrl_d), None);
    assert!(app.show_diff, "Ctrl-D should open the diff panel");
    assert!(app.diff_text.is_some(), "opening should cache diff text");

    assert_eq!(app.edit_key(&ctrl_d), None);
    assert!(!app.show_diff, "second Ctrl-D should close the diff panel");
    assert!(
        app.diff_text.is_none(),
        "closing should clear cached diff text"
    );
}

#[test]
fn ctrl_d_toggles_the_diff_panel() {
    // Toggling Ctrl-D opens the panel with the cached diff text and a
    // header; toggling again closes it. We set diff_text directly to avoid
    // a real git call in the unit test.
    let mut app = test_app("openai", "gpt-4o");
    app.show_diff = true;
    app.diff_text = Some("--- a/x\n+++ b/x\n@@ -1,1 +1,1 @@\n-old\n+new\n".into());
    let mut term = Terminal::new(TestBackend::new(60, 14)).unwrap();
    term.draw(|f| app.render(f)).unwrap();
    let screen = dump(&term);
    assert!(
        screen.contains("diff (Ctrl-D to close)"),
        "panel header: {screen}"
    );
    assert!(screen.contains("+new"), "diff content rendered: {screen}");

    // Closing drops the panel.
    app.show_diff = false;
    app.diff_text = None;
    term.draw(|f| app.render(f)).unwrap();
    let screen2 = dump(&term);
    assert!(
        !screen2.contains("diff (Ctrl-D to close)"),
        "panel closed: {screen2}"
    );
}

#[test]
fn ctrl_question_toggles_the_observability_panel() {
    // The Ctrl-? agent-observability panel renders the last turn's telemetry
    // counters, the per-turn tool-call count, and session/context numbers.
    let mut app = test_app("openai", "gpt-4o");
    app.show_debug = true;
    let mut repair_counts = std::collections::BTreeMap::new();
    repair_counts.insert("review_listing_only".to_string(), 4);
    repair_counts.insert("review_no_evidence".to_string(), 1);
    app.last_telemetry = Some(hi_agent::TurnTelemetry {
        effective_max_steps: 120,
        verify_rounds: 2,
        recovery_retries: 1,
        repeat_nudges: 0,
        continue_nudges: 1,
        truncation_retries: 0,
        no_progress_streak: 0,
        forced_final_answer_attempts: 0,
        last_progress_reason: "accepted final answer".to_string(),
        last_stall_reason: String::new(),
        hit_step_cap: false,
        stalled_unfinished: false,
        stalled_repeating: false,
        verify_attributions: Vec::new(),
        verification_executions: Vec::new(),
        tool_calls: 7,
        max_concurrent_batch: 3,
        serial_runs: 2,
        tool_timeline: Vec::new(),
        progress_events: Vec::new(),
        file_reads: 2,
        targeted_searches: 1,
        listing_only: false,
        first_tool_kind: "read".to_string(),
        discovery_depth: "mixed".to_string(),
        quality_repair_nudges: 5,
        review_repair_exhaustion_reason: "review_listing_only_exhausted".to_string(),
        review_repair_counts: repair_counts,
        review_repair_stopped_by_exhaustion: true,
        skeptic_unavailable_count: 0,
        skeptic_last_status: None,
        checkpoint_available: None,
        advertised_tools: vec!["read".to_string(), "grep".to_string()],
        tool_schema_tokens: 512,
    });
    app.turn_tool_calls = 7;
    app.apply(UiEvent::Usage {
        prompt: 12,
        generated: 340,
        ctx_used: 64_000,
        ctx_window: Some(128_000),
        estimated: false,
    });
    let mut term = Terminal::new(TestBackend::new(96, 18)).unwrap();
    term.draw(|f| app.render(f)).unwrap();
    let screen = dump(&term);
    assert!(
        screen.contains("agent (Ctrl-? to close)"),
        "panel header: {screen}"
    );
    assert!(
        screen.contains("2 verify") && screen.contains("1 retry") && screen.contains("1 continue"),
        "telemetry counters: {screen}"
    );
    assert!(
        screen.contains("tool calls this turn: 7"),
        "tool-call count: {screen}"
    );
    assert!(
        screen.contains("user prompt estimate 12 · output across all model calls 340 · ctx 50%"),
        "scoped token metrics: {screen}"
    );
    assert!(
        screen.contains("review repair: total 5")
            && screen.contains("top listing=4")
            && screen.contains("exhausted listing"),
        "review repair diagnostics: {screen}"
    );

    // Closing drops the panel.
    app.show_debug = false;
    term.draw(|f| app.render(f)).unwrap();
    let screen2 = dump(&term);
    assert!(
        !screen2.contains("agent (Ctrl-? to close)"),
        "panel closed: {screen2}"
    );
}

#[test]
fn ctrl_question_compacts_long_review_repair_mode_names() {
    let mut app = test_app("openai", "gpt-4o");
    app.show_debug = true;
    let mut repair_counts = std::collections::BTreeMap::new();
    repair_counts.insert("review_security_broad_search".to_string(), 12);
    repair_counts.insert("review_gap_search_overclaim".to_string(), 9);
    app.last_telemetry = Some(hi_agent::TurnTelemetry {
        effective_max_steps: 120,
        quality_repair_nudges: 21,
        review_repair_exhaustion_reason: "review_security_broad_search_exhausted".to_string(),
        review_repair_counts: repair_counts,
        review_repair_stopped_by_exhaustion: true,
        ..hi_agent::TurnTelemetry::default()
    });

    let mut term = Terminal::new(TestBackend::new(96, 14)).unwrap();
    term.draw(|f| app.render(f)).unwrap();
    let screen = dump(&term);
    assert!(
        screen.contains("top security_broad=12")
            && screen.contains("gap_overclaim=9")
            && screen.contains("exhausted security_broad"),
        "compact review-repair labels: {screen}"
    );
    assert!(
        !screen.contains("review_security_broad_search")
            && !screen.contains("review_gap_search_overclaim"),
        "raw long repair keys should not render in Ctrl-?: {screen}"
    );
}

#[test]
fn in_progress_line_is_styled_live() {
    // A heading still streaming (no trailing newline yet) renders styled with
    // its markers stripped — not literally as "## …" until the line commits.
    let mut app = test_app("openai", "gpt-4o");
    app.apply(UiEvent::Text {
        text: "## Hello world".into(),
    });
    let mut term = Terminal::new(TestBackend::new(60, 12)).unwrap();
    term.draw(|f| app.render(f)).unwrap();
    let screen = dump(&term);
    assert!(
        screen.contains("Hello world"),
        "heading text shown:\n{screen}"
    );
    assert!(
        !screen.contains("## Hello"),
        "marker stripped live:\n{screen}"
    );

    // Styling the preview must NOT advance the real fence state: a partial
    // opening fence leaves code_lang untouched until its line commits.
    let mut app2 = test_app("openai", "gpt-4o");
    app2.apply(UiEvent::Text {
        text: "```rust".into(),
    });
    term.draw(|f| app2.render(f)).unwrap();
    assert!(
        app2.code_lang.is_none(),
        "live preview must not mutate the committed fence state"
    );
}

#[test]
fn edit_key_submits_on_enter_and_clears() {
    let mut app = test_app("openai", "gpt-4o");
    app.input.set("queue me");
    let enter = KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE);
    assert_eq!(app.edit_key(&enter).as_deref(), Some("queue me"));
    assert!(app.input.is_empty(), "input cleared after submit");
    // An empty Enter submits nothing.
    assert_eq!(app.edit_key(&enter), None);
}

#[test]
fn block_nav_folds_one_block_independently() {
    use crate::TranscriptEntry;
    let mut app = test_app("openai", "gpt-4o");
    let long =
        || -> Vec<Line<'static>> { (0..40).map(|i| Line::raw(format!("line {i}"))).collect() };
    app.transcript.push(TranscriptEntry::ToolOutput {
        body: long(),
        expanded: false,
    });
    app.transcript.push(TranscriptEntry::ToolOutput {
        body: long(),
        expanded: false,
    });
    assert_eq!(app.tool_block_count(), 2);

    // Ctrl-B enters nav on the most recent block.
    app.edit_key(&KeyEvent::new(KeyCode::Char('b'), KeyModifiers::CONTROL));
    assert!(app.nav_mode, "Ctrl-B enters nav mode");
    assert_eq!(app.selected_block_ord(), 1, "starts on the last block");

    // Enter unfolds just that block.
    app.edit_key(&KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
    let expanded: Vec<bool> = app
        .transcript
        .iter()
        .filter_map(|e| match e {
            TranscriptEntry::ToolOutput { expanded, .. } => Some(*expanded),
            _ => None,
        })
        .collect();
    assert_eq!(
        expanded,
        vec![false, true],
        "only the selected block toggled"
    );

    // k/Up moves to the older block; Space toggles it.
    app.edit_key(&KeyEvent::new(KeyCode::Up, KeyModifiers::NONE));
    assert_eq!(app.selected_block_ord(), 0);
    app.edit_key(&KeyEvent::new(KeyCode::Char(' '), KeyModifiers::NONE));
    let both: Vec<bool> = app
        .transcript
        .iter()
        .filter_map(|e| match e {
            TranscriptEntry::ToolOutput { expanded, .. } => Some(*expanded),
            _ => None,
        })
        .collect();
    assert_eq!(both, vec![true, true]);

    // The cursor never runs past the ends.
    app.edit_key(&KeyEvent::new(KeyCode::Up, KeyModifiers::NONE));
    assert_eq!(app.selected_block_ord(), 0, "clamped at the top");

    // Esc leaves nav mode; keys go back to the input line.
    app.edit_key(&KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE));
    assert!(!app.nav_mode);
    app.edit_key(&KeyEvent::new(KeyCode::Char('j'), KeyModifiers::NONE));
    assert_eq!(
        app.input.text(),
        "j",
        "keys type into the input once nav exits"
    );
}

#[test]
fn streamed_table_commits_aligned_after_it_ends() {
    let mut app = test_app("openai", "gpt-4o");
    // Stream a full pipe table. Every line is a table row, so it accumulates in
    // the buffer and nothing is committed yet.
    app.stream(
        ratatui::style::Style::default(),
        true,
        "| A | Long |\n|---|---|\n| x | y |\n",
    );
    assert!(
        app.transcript.is_empty(),
        "table stays buffered until it ends"
    );
    // A following non-table line flushes the table as an aligned block.
    app.stream(ratatui::style::Style::default(), true, "after\n");
    let texts: Vec<String> = app.transcript.iter().map(|e| e.text()).collect();
    assert_eq!(
        texts.len(),
        4,
        "3 aligned rows + the trailing line: {texts:?}"
    );
    assert_eq!(texts[3], "after");
    // Header (row 0) and data (row 2) are padded to the same width.
    assert_eq!(
        texts[0].chars().count(),
        texts[2].chars().count(),
        "columns aligned across rows: {texts:?}"
    );
    assert!(texts[1].starts_with('├'), "ruled separator: {:?}", texts[1]);
}

#[test]
fn streaming_preview_shows_cursor_during_stream_and_clears_after_flush() {
    let mut app = test_app("openai", "gpt-4o");
    // Stream a partial line (no trailing newline) — the pending preview should
    // be live, and the render should show the block cursor.
    app.stream(ratatui::style::Style::default(), true, "hello wor");
    assert!(app.pending.is_some(), "pending line is live mid-stream");
    let mut term = Terminal::new(TestBackend::new(80, 24)).unwrap();
    term.draw(|f| app.render(f)).unwrap();
    let screen = dump(&term);
    assert!(
        screen.contains("▍"),
        "streaming preview shows a block cursor: {screen}"
    );
    assert!(
        screen.contains("hello wor"),
        "partial line text is visible mid-stream: {screen}"
    );
    // Complete the line — the cursor should disappear once flushed.
    app.stream(ratatui::style::Style::default(), true, "ld\n");
    app.flush_pending();
    term.draw(|f| app.render(f)).unwrap();
    let screen = dump(&term);
    assert!(
        !screen.contains("▍"),
        "no block cursor after the line is committed: {screen}"
    );
    assert!(
        screen.contains("hello world"),
        "completed line is visible: {screen}"
    );
}

#[test]
fn streamed_code_block_is_captured_for_copy_last_code_block() {
    let mut app = test_app("openai", "gpt-4o");
    // Stream a fenced code block with a language tag and two interior lines.
    app.stream(
        ratatui::style::Style::default(),
        true,
        "```rust\nfn main() {}\nlet x = 1;\n```\n",
    );
    // The last code block should hold the two interior lines (no fence markers).
    assert_eq!(
        app.last_code_block.as_deref(),
        Some("fn main() {}\nlet x = 1;"),
        "interior code lines captured without fence markers"
    );
    // A second block replaces the first as the "last" block.
    app.stream(
        ratatui::style::Style::default(),
        true,
        "```python\nprint('hi')\n```\n",
    );
    assert_eq!(
        app.last_code_block.as_deref(),
        Some("print('hi')"),
        "the most recent block is the one Ctrl-Y copies"
    );
}

#[test]
fn copy_last_code_block_falls_back_to_transcript_scan() {
    let mut app = test_app("openai", "gpt-4o");
    // Simulate a resumed session: transcript has rendered code lines (with the
    // `▏ ` gutter) but `last_code_block` was never populated by streaming.
    app.last_code_block = None;
    // Push a non-code line, then a fenced block as markdown_line would render it.
    app.push(ratatui::text::Line::raw("Here is some code:"));
    // Fence-open line: gutter + language tag.
    app.push(crate::render::markdown_line("```rust", &mut None));
    // Interior code lines.
    let mut lang = Some("rust".to_string());
    app.push(crate::render::markdown_line("fn main() {}", &mut lang));
    app.push(crate::render::markdown_line("let x = 1;", &mut lang));
    // Fence-close line.
    app.push(crate::render::markdown_line("```", &mut lang));
    let block = app.scan_transcript_for_last_code_block();
    assert_eq!(
        block.as_deref(),
        Some("fn main() {}\nlet x = 1;"),
        "fallback scan extracts interior code lines without fence markers"
    );
}

#[test]
fn shell_escape_prefix_runs_command_and_pushes_output() {
    let mut app = test_app("openai", "gpt-4o");
    // Use a workspace root that exists (the crate root) so `sh -c` runs there.
    app.workspace_root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    app.run_shell_escape("echo hello-shell-escape");
    // The transcript should contain the `$ echo ...` header and the output line.
    let texts: Vec<String> = app.transcript.iter().map(|e| e.text()).collect();
    let joined = texts.join("\n");
    assert!(
        joined.contains("hello-shell-escape"),
        "shell-escape output should land in the transcript: {joined}"
    );
    assert!(
        joined.contains("$ echo hello-shell-escape"),
        "the command header should be shown: {joined}"
    );
}

#[test]
fn confirmation_modal_colors_file_edit_diff() {
    use hi_agent::ConfirmationRequest;
    let mut app = test_app("openai", "gpt-4o");
    app.confirmation = Some(ConfirmationRequest::FileEdit {
        path: "src/main.rs".to_string(),
        diff: "--- a/src/main.rs\n+++ b/src/main.rs\n@@ -1,2 +1,2 @@\n-old\n+new\n ctx\n"
            .to_string(),
    });
    let mut term = Terminal::new(TestBackend::new(80, 24)).unwrap();
    term.draw(|f| app.render(f)).unwrap();
    let screen = dump(&term);
    // The diff header and the path should appear; the modal title too.
    assert!(
        screen.contains("Confirm file edit"),
        "modal title shown: {screen}"
    );
    assert!(screen.contains("src/main.rs"), "file path shown: {screen}");
    assert!(screen.contains("+new"), "added diff line shown: {screen}");
    assert!(screen.contains("-old"), "removed diff line shown: {screen}");
}

#[test]
fn review_overlay_shows_full_diff_with_title() {
    let mut app = test_app("openai", "gpt-4o");
    app.workspace_root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    app.show_review = true;
    app.diff_text = Some(
        "--- a/src/main.rs\n+++ b/src/main.rs\n@@ -1,2 +1,2 @@\n-old\n+new\n ctx\n".to_string(),
    );
    let mut term = Terminal::new(TestBackend::new(80, 24)).unwrap();
    term.draw(|f| app.render(f)).unwrap();
    let screen = dump(&term);
    assert!(
        screen.contains("Diff review"),
        "overlay title shown: {screen}"
    );
    assert!(screen.contains("+new"), "added line visible: {screen}");
    assert!(screen.contains("-old"), "removed line visible: {screen}");
    assert!(
        screen.contains("n/p hunks"),
        "keybinding footer shown: {screen}"
    );
}

#[test]
fn show_session_files_lists_accumulated_files() {
    let mut app = test_app("openai", "gpt-4o");
    // Simulate two turns touching different files.
    app.last_changed_files = vec!["src/main.rs".to_string(), "src/lib.rs".to_string()];
    app.accumulate_session_files();
    app.last_changed_files = vec!["src/lib.rs".to_string(), "src/render.rs".to_string()];
    app.accumulate_session_files();
    // The session set should be deduplicated, first-seen order.
    assert_eq!(
        app.session_changed_files,
        vec![
            "src/main.rs".to_string(),
            "src/lib.rs".to_string(),
            "src/render.rs".to_string(),
        ],
        "session files accumulate and dedupe across turns"
    );
    // `/files` should render the list into the transcript.
    app.show_session_files();
    let text = app.transcript_text();
    assert!(
        text.contains("3 files changed this session"),
        "header shows count: {text}"
    );
    assert!(
        text.contains("src/main.rs") && text.contains("src/render.rs"),
        "file paths listed: {text}"
    );
}

#[test]
fn show_session_files_handles_empty_session() {
    let mut app = test_app("openai", "gpt-4o");
    app.show_session_files();
    let text = app.transcript_text();
    assert!(
        text.contains("no files changed this session yet"),
        "empty session message: {text}"
    );
}

#[test]
fn normal_mode_renders_banner_and_hides_cursor() {
    let mut app = test_app("openai", "gpt-4o");
    app.normal_mode = true;
    let mut term = Terminal::new(TestBackend::new(80, 24)).unwrap();
    term.draw(|f| app.render(f)).unwrap();
    let screen = dump(&term);
    assert!(
        screen.contains("-- NORMAL --"),
        "normal-mode banner shown: {screen}"
    );
    assert!(
        screen.contains("j/k scroll"),
        "keybinding hint shown: {screen}"
    );
}

#[test]
fn normal_mode_search_banner_shows_query() {
    let mut app = test_app("openai", "gpt-4o");
    app.normal_mode = true;
    app.search_query = Some("render".to_string());
    let mut term = Terminal::new(TestBackend::new(80, 24)).unwrap();
    term.draw(|f| app.render(f)).unwrap();
    let screen = dump(&term);
    assert!(
        screen.contains("-- SEARCH --"),
        "search banner shown: {screen}"
    );
    assert!(screen.contains("/render"), "search query shown: {screen}");
}

#[test]
fn search_transcript_finds_and_scrolls_to_match() {
    let mut app = test_app("openai", "gpt-4o");
    // Push enough lines that the transcript overflows a 24-row terminal, so
    // scrolling to a match is meaningful (view_max_scroll > 0).
    for i in 0..30 {
        app.push(ratatui::text::Line::raw(format!("filler line {i}")));
    }
    app.push(ratatui::text::Line::raw("the render function"));
    for i in 0..10 {
        app.push(ratatui::text::Line::raw(format!("more filler {i}")));
    }
    app.push(ratatui::text::Line::raw("another render here"));
    // view_max_scroll is normally computed during render; set it manually so
    // scroll_to doesn't clamp to 0 in this unit test (no render happens).
    app.view_max_scroll = 50;
    // Compute the transcript text to find the expected line index.
    let text = app.transcript_text();
    let lines: Vec<&str> = text.lines().collect();
    let first_render = lines
        .iter()
        .position(|l| l.contains("render"))
        .expect("first render match exists");
    // Search forward for "render" from the top — should scroll to the first match.
    search_transcript(&mut app, "render", 1);
    assert_eq!(
        app.scroll as usize, first_render,
        "search should scroll to the first match at line {first_render}, got {}",
        app.scroll
    );
    // Search forward again — should advance to the next "render".
    let second_render = lines
        .iter()
        .rposition(|l| l.contains("render"))
        .expect("second render match exists");
    search_transcript(&mut app, "render", 1);
    assert_eq!(
        app.scroll as usize, second_render,
        "n should advance to the next match at line {second_render}"
    );
    // Search backward — should go back to the first match.
    search_transcript(&mut app, "render", -1);
    assert_eq!(
        app.scroll as usize, first_render,
        "N should return to the previous match"
    );
}

#[test]
fn scroll_to_top_and_bottom_set_following_correctly() {
    let mut app = test_app("openai", "gpt-4o");
    for i in 0..50 {
        app.push(ratatui::text::Line::raw(format!("line {i}")));
    }
    // Scroll to top: following=false, scroll=0.
    app.scroll_to_top();
    assert!(!app.following, "scroll_to_top stops following");
    assert_eq!(app.scroll, 0, "scroll_to_top sets scroll to 0");
    // Scroll to bottom: following=true.
    app.scroll_to_bottom();
    assert!(app.following, "scroll_to_bottom resumes following");
}

#[test]
fn review_next_hunk_jumps_between_hunk_headers() {
    let diff = "diff --git a/foo b/foo\n\
                --- a/foo\n\
                +++ b/foo\n\
                @@ -1,1 +1,1 @@\n\
                -a\n\
                +b\n\
                @@ -5,1 +5,1 @@\n\
                -c\n\
                +d\n\
                @@ -10,1 +10,1 @@\n\
                -e\n\
                +f\n";
    // From line 0 (before first hunk), n → first hunk at line 3.
    assert_eq!(review_next_hunk(Some(diff), 0, 1), 3);
    // From line 3 (first hunk), n → second hunk at line 6.
    assert_eq!(review_next_hunk(Some(diff), 3, 1), 6);
    // From line 6 (second hunk), n → third hunk at line 9.
    assert_eq!(review_next_hunk(Some(diff), 6, 1), 9);
    // From line 9 (third hunk), n → clamps to last line (no more hunks).
    assert_eq!(review_next_hunk(Some(diff), 9, 1), 11);
    // From line 9, p → previous hunk at line 6.
    assert_eq!(review_next_hunk(Some(diff), 9, -1), 6);
    // From line 6, p → previous hunk at line 3.
    assert_eq!(review_next_hunk(Some(diff), 6, -1), 3);
    // From line 3, p → clamps to 0 (no earlier hunk).
    assert_eq!(review_next_hunk(Some(diff), 3, -1), 0);
    // None diff → returns `from` unchanged.
    assert_eq!(review_next_hunk(None, 5, 1), 5);
}

#[test]
fn changed_files_entry_deep_links_to_review_on_click() {
    let mut app = test_app("openai", "gpt-4o");
    app.workspace_root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    // Push a ChangedFiles entry — simulates the agent editing two files.
    app.transcript.push(TranscriptEntry::ChangedFiles {
        line: Line::raw("✎ 2 files changed: src/a.rs, src/b.rs"),
        files: vec!["src/a.rs".to_string(), "src/b.rs".to_string()],
    });
    // The ChangedFiles entry is at flattened line 0.
    let files = app.changed_files_at_flat_line(0);
    assert_eq!(
        files.as_deref(),
        Some(&["src/a.rs".to_string(), "src/b.rs".to_string()][..]),
        "click on the changed-files line returns its file list"
    );
    // A line that doesn't fall on a ChangedFiles entry returns None.
    assert_eq!(app.changed_files_at_flat_line(5), None);
    // open_review with a file filter sets up the review overlay.
    app.open_review(Some(&["src/a.rs".to_string()]));
    assert!(app.show_review, "review overlay opened");
    assert!(
        app.diff_text.as_deref().unwrap_or("").contains("src/a.rs")
            || app.diff_text.as_deref().unwrap_or("").is_empty(),
        "filtered diff text set (may be empty if no changes)"
    );
}

#[test]
fn open_review_with_no_files_shows_full_diff() {
    let mut app = test_app("openai", "gpt-4o");
    app.workspace_root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    app.open_review(None);
    assert!(app.show_review, "review overlay opened");
    assert!(app.review_scroll == 0, "scroll reset to top");
}

#[test]
fn external_editor_reads_back_edited_text() {
    // Create a tiny "editor" script that appends to its last argument.
    let script = std::env::temp_dir().join(format!(".hi-test-editor-{}", std::process::id()));
    std::fs::write(&script, "#!/bin/sh\nprintf 'edited' >> \"$1\"\n").unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = std::fs::metadata(&script).unwrap().permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions(&script, perms).unwrap();
    }
    unsafe {
        std::env::set_var("VISUAL", "");
        std::env::set_var("EDITOR", script.to_str().unwrap());
        std::env::set_var("HI_TUI_NO_TERMINAL", "1");
    }
    let mut app = test_app("openai", "gpt-4o");
    app.input.set("original prompt");
    app.edit_in_external_editor();
    let text = app.input.text();
    assert!(
        text.contains("original prompt") && text.contains("edited"),
        "input should contain the edited text: {text}"
    );
    // Clean up.
    unsafe {
        std::env::remove_var("EDITOR");
        std::env::remove_var("HI_TUI_NO_TERMINAL");
    }
    let _ = std::fs::remove_file(&script);
}

#[test]
fn mouse_drag_selects_a_line_range_and_keeps_it() {
    use crossterm::event::{MouseButton, MouseEvent, MouseEventKind};
    let mut app = test_app("openai", "gpt-4o");
    for i in 0..5 {
        app.transcript
            .push(crate::TranscriptEntry::Line(Line::raw(format!("row {i}"))));
    }
    // Geometry the render pass would cache: inner rect at (1,1), no scroll, each
    // line exactly one wrapped row.
    app.view_inner = ratatui::layout::Rect {
        x: 1,
        y: 1,
        width: 80,
        height: 10,
    };
    app.view_scroll = 0;
    app.view_prefix = vec![0, 1, 2, 3, 4, 5];
    app.view_line_texts = (0..5).map(|i| format!("row {i}")).collect();

    let ev = |kind, col, row| MouseEvent {
        kind,
        column: col,
        row,
        modifiers: KeyModifiers::NONE,
    };
    // Press on line 1 (screen row 2 → abs 1); no selection range change yet.
    app.handle_mouse(ev(MouseEventKind::Down(MouseButton::Left), 5, 2));
    assert_eq!(app.selection_range(), Some((1, 1)));
    assert!(!app.select_dragged);
    // Drag down to line 3 (screen row 4 → abs 3).
    app.handle_mouse(ev(MouseEventKind::Drag(MouseButton::Left), 5, 4));
    assert_eq!(app.selection_range(), Some((1, 3)));
    assert!(app.select_dragged);
    // The exact text a release would copy (pure — no real clipboard touched).
    assert_eq!(app.selected_text().as_deref(), Some("row 1\nrow 2\nrow 3"));
    // A drag that runs off the bottom edge clamps to the last visible line.
    app.handle_mouse(ev(MouseEventKind::Down(MouseButton::Left), 5, 3)); // abs 2
    app.handle_mouse(ev(MouseEventKind::Drag(MouseButton::Left), 5, 250));
    assert_eq!(
        app.selection_range(),
        Some((2, 4)),
        "clamped to the last line"
    );
}

#[test]
fn mouse_drag_within_one_line_selects_characters() {
    use crossterm::event::{MouseButton, MouseEvent, MouseEventKind};
    let mut app = test_app("openai", "gpt-4o");
    app.transcript.push(crate::TranscriptEntry::Line(Line::raw(
        "hello world foobar",
    )));
    app.view_inner = ratatui::layout::Rect {
        x: 1,
        y: 1,
        width: 80,
        height: 10,
    };
    app.view_scroll = 0;
    app.view_prefix = vec![0, 1]; // one logical line, one display row
    app.view_line_texts = vec!["hello world foobar".to_string()];
    let ev = |kind, col, row| MouseEvent {
        kind,
        column: col,
        row,
        modifiers: KeyModifiers::NONE,
    };
    // 'w' of "world" is char index 6 → screen col = inner.x(1) + 6 = 7; drag to
    // just past 'd' (index 11) → col 12. The character range 6..11 is "world".
    app.handle_mouse(ev(MouseEventKind::Down(MouseButton::Left), 7, 1));
    app.handle_mouse(ev(MouseEventKind::Drag(MouseButton::Left), 12, 1));
    assert_eq!(app.char_span(), Some((0, 6, 11)), "single-line char range");
    assert_eq!(app.selected_text().as_deref(), Some("world"));

    // Extending across a second line falls back to whole-line selection.
    app.transcript
        .push(crate::TranscriptEntry::Line(Line::raw("second line")));
    app.view_prefix = vec![0, 1, 2];
    app.view_line_texts = vec!["hello world foobar".into(), "second line".into()];
    app.handle_mouse(ev(MouseEventKind::Down(MouseButton::Left), 7, 1)); // line 0
    app.handle_mouse(ev(MouseEventKind::Drag(MouseButton::Left), 5, 2)); // line 1
    assert_eq!(app.char_span(), None, "multi-line → no char span");
    assert_eq!(
        app.selected_text().as_deref(),
        Some("hello world foobar\nsecond line")
    );
}

#[test]
fn mouse_plain_click_folds_and_leaves_no_selection() {
    use crossterm::event::{MouseButton, MouseEvent, MouseEventKind};
    let mut app = test_app("openai", "gpt-4o");
    let body: Vec<Line<'static>> = (0..40).map(|i| Line::raw(format!("l{i}"))).collect();
    app.transcript.push(crate::TranscriptEntry::ToolOutput {
        body,
        expanded: false,
    });
    app.view_inner = ratatui::layout::Rect {
        x: 1,
        y: 1,
        width: 80,
        height: 20,
    };
    app.view_scroll = 0;
    app.view_prefix = (0..=40).collect();
    app.view_line_texts = (0..40).map(|i| format!("l{i}")).collect();
    app.block_row_spans = vec![(0, 17, 0)];

    let ev = |kind, col, row| MouseEvent {
        kind,
        column: col,
        row,
        modifiers: KeyModifiers::NONE,
    };
    // Down then Up at the same spot (no drag) → a fold, not a selection.
    app.handle_mouse(ev(MouseEventKind::Down(MouseButton::Left), 5, 3));
    app.handle_mouse(ev(MouseEventKind::Up(MouseButton::Left), 5, 3));
    assert!(app.selection_range().is_none(), "click leaves no selection");
    let expanded = match &app.transcript[0] {
        crate::TranscriptEntry::ToolOutput { expanded, .. } => *expanded,
        _ => unreachable!(),
    };
    assert!(expanded, "the clicked block folded open");
}

#[test]
fn mouse_click_folds_the_block_under_it() {
    use crate::TranscriptEntry;
    let mut app = test_app("openai", "gpt-4o");
    let long = || -> Vec<Line<'static>> { (0..40).map(|i| Line::raw(format!("l{i}"))).collect() };
    app.transcript.push(TranscriptEntry::ToolOutput {
        body: long(),
        expanded: false,
    });
    app.transcript.push(TranscriptEntry::ToolOutput {
        body: long(),
        expanded: false,
    });
    // Simulate the geometry the render pass caches: inner area at (1,1), no
    // scroll, block 0 spanning wrapped rows 0..17 and block 1 rows 17..34.
    app.view_inner = ratatui::layout::Rect {
        x: 1,
        y: 1,
        width: 80,
        height: 20,
    };
    app.view_scroll = 0;
    app.block_row_spans = vec![(0, 17, 0), (17, 34, 1)];

    let expanded = |app: &crate::App| -> Vec<bool> {
        app.transcript
            .iter()
            .filter_map(|e| match e {
                TranscriptEntry::ToolOutput { expanded, .. } => Some(*expanded),
                _ => None,
            })
            .collect()
    };

    // Screen row 20 → abs row 19 ∈ [17,34) → block 1.
    app.handle_click(5, 20);
    assert_eq!(expanded(&app), vec![false, true], "clicked block toggled");
    assert_eq!(app.block_cursor, 1, "cursor moved to the clicked block");

    // A click below the transcript area is ignored.
    app.handle_click(5, 100);
    assert_eq!(
        expanded(&app),
        vec![false, true],
        "out-of-area click ignored"
    );

    // Screen row 2 → abs row 1 ∈ [0,17) → block 0 toggles open too.
    app.handle_click(5, 2);
    assert_eq!(expanded(&app), vec![true, true]);
}

#[test]
fn block_nav_expanded_block_shows_full_body() {
    use crate::TranscriptEntry;
    let body: Vec<Line<'static>> = (0..40).map(|i| Line::raw(format!("l{i}"))).collect();
    // Folded (default): a preview plus a fold footer.
    let folded = TranscriptEntry::ToolOutput {
        body: body.clone(),
        expanded: false,
    };
    let flat = folded.flatten(false, false);
    assert!(flat.len() < 40, "folded to a preview: {} lines", flat.len());
    assert!(
        flat.iter()
            .any(|l| crate::render::line_text(l).contains("more lines")),
        "fold footer present"
    );
    // Per-block expand shows the whole body without the global toggle.
    let open = TranscriptEntry::ToolOutput {
        body,
        expanded: true,
    };
    let flat = open.flatten(false, false);
    assert!(
        flat.len() >= 40,
        "expanded shows full body: {} lines",
        flat.len()
    );
}

#[test]
fn renders_title_transcript_and_input() {
    let mut app = test_app("openai", "gpt-4o");
    app.push(Line::raw("› hello"));
    app.apply(UiEvent::Text {
        text: "hi there\n".into(),
    });
    app.apply(UiEvent::AssistantEnd);
    app.input.set("next question");

    let mut term = Terminal::new(TestBackend::new(50, 12)).unwrap();
    term.draw(|f| app.render(f)).unwrap();
    let screen = dump(&term);

    assert!(screen.contains("gpt-4o"), "title shows model");
    assert!(screen.contains("hello"), "user line");
    assert!(screen.contains("hi there"), "assistant line");
    assert!(screen.contains("next question"), "input box");
}

fn turn_outcome(
    status: hi_agent::TurnStatus,
    verification: hi_agent::VerificationStatus,
    review: hi_agent::ReviewStatus,
    stop_reason: hi_agent::TurnStopReason,
) -> hi_agent::TurnOutcome {
    hi_agent::TurnOutcome {
        status,
        verification,
        review,
        stop_reason,
        changed_files: vec!["src/lib.rs".to_string()],
        verified_workspace_revision: (verification == hi_agent::VerificationStatus::Passed)
            .then(|| "revision-1".to_string()),
        effective_route: hi_agent::EffectiveModelRoute {
            provider: Some("test".to_string()),
            model: "model".to_string(),
        },
    }
}

#[test]
fn turn_end_is_neutral_until_typed_pass_arrives() {
    let mut app = test_app("openai", "gpt-4o");
    app.set_working(true);
    app.apply(UiEvent::TurnEnd {
        summary: "[10 in · 2 out · 12 total]".into(),
    });

    assert_eq!(app.last_turn_state, TurnState::Running);
    assert_eq!(app.transcript.len(), 1);
    let usage = app.transcript[0].text();
    assert!(usage.contains("usage"), "got: {usage}");
    assert!(
        !usage.contains("✓"),
        "usage must not imply success: {usage}"
    );
    assert!(
        usage.contains("12 total"),
        "historical usage retained: {usage}"
    );

    app.note_turn_outcome(&turn_outcome(
        hi_agent::TurnStatus::Completed,
        hi_agent::VerificationStatus::Passed,
        hi_agent::ReviewStatus::Passed,
        hi_agent::TurnStopReason::Completed,
    ));
    assert_eq!(
        app.last_turn_state,
        TurnState::Done("verified · reviewed".to_string())
    );
    assert!(app.transcript.last().unwrap().text().contains("✓ done"));
}

#[test]
fn usage_summary_content_cannot_override_typed_outcome() {
    let mut app = test_app("openai", "gpt-4o");
    let noisy = "[user prompt estimate 10 · output across all model calls 2 · ctx 5% (500/10k) · steer: 2 verify · 1 retry]";
    app.apply(UiEvent::TurnEnd {
        summary: noisy.into(),
    });
    app.note_turn_outcome(&turn_outcome(
        hi_agent::TurnStatus::Incomplete,
        hi_agent::VerificationStatus::Unverified,
        hi_agent::ReviewStatus::Unavailable,
        hi_agent::TurnStopReason::VerificationUnavailable,
    ));

    assert!(matches!(app.last_turn_state, TurnState::Warning(_)));
    let transcript = app.transcript_text();
    assert!(transcript.contains("steer"), "usage retained: {transcript}");
    assert!(transcript.contains("⚠ incomplete · unverified changes"));
    assert!(!transcript.contains("✓ done"));
}

#[test]
fn unverified_completed_mutation_is_warning_not_done() {
    let mut app = test_app("openai", "gpt-4o");
    app.note_turn_outcome(&turn_outcome(
        hi_agent::TurnStatus::Completed,
        hi_agent::VerificationStatus::Unverified,
        hi_agent::ReviewStatus::NotRequired,
        hi_agent::TurnStopReason::VerificationUnavailable,
    ));

    assert_eq!(
        app.last_turn_state,
        TurnState::Warning("unverified changes".to_string())
    );
    assert!(!app.transcript_text().contains("✓ done"));
}

#[test]
fn deterministic_pass_survives_review_unavailability() {
    let mut app = test_app("openai", "gpt-4o");
    app.note_turn_outcome(&turn_outcome(
        hi_agent::TurnStatus::Completed,
        hi_agent::VerificationStatus::Passed,
        hi_agent::ReviewStatus::Unavailable,
        hi_agent::TurnStopReason::Completed,
    ));

    assert_eq!(
        app.last_turn_state,
        TurnState::Done("verified · review unavailable".to_string())
    );
}

#[test]
fn review_objection_cannot_render_done() {
    let mut app = test_app("openai", "gpt-4o");
    app.note_turn_outcome(&turn_outcome(
        hi_agent::TurnStatus::Incomplete,
        hi_agent::VerificationStatus::Passed,
        hi_agent::ReviewStatus::Objected,
        hi_agent::TurnStopReason::ReviewObjected,
    ));

    assert_eq!(
        app.last_turn_state,
        TurnState::Warning("incomplete · review objected".to_string())
    );
    assert!(!app.transcript_text().contains("✓ done"));
}

#[test]
fn verification_infrastructure_failure_is_failed() {
    let mut app = test_app("openai", "gpt-4o");
    app.note_turn_outcome(&turn_outcome(
        hi_agent::TurnStatus::Failed,
        hi_agent::VerificationStatus::InfrastructureError,
        hi_agent::ReviewStatus::Unavailable,
        hi_agent::TurnStopReason::InfrastructureFailure,
    ));

    assert_eq!(
        app.last_turn_state,
        TurnState::Failed("infrastructure failure".to_string())
    );
    assert!(app.transcript_text().contains("✗ failed"));
}

#[test]
fn typed_cancellation_is_cancelled() {
    let mut app = test_app("openai", "gpt-4o");
    app.note_turn_outcome(&turn_outcome(
        hi_agent::TurnStatus::Cancelled,
        hi_agent::VerificationStatus::Unverified,
        hi_agent::ReviewStatus::Unavailable,
        hi_agent::TurnStopReason::Cancelled,
    ));

    assert_eq!(app.last_turn_state, TurnState::Cancelled);
    assert!(!app.transcript_text().contains("✓ done"));
}

#[test]
fn assistant_text_becomes_copy_target() {
    let mut app = test_app("openai", "gpt-4o");
    app.apply(UiEvent::Text {
        text: "first ".into(),
    });
    app.apply(UiEvent::Text {
        text: "answer\n".into(),
    });
    app.apply(UiEvent::AssistantEnd);
    assert_eq!(app.last_assistant, "first answer");

    app.apply(UiEvent::ToolCall {
        name: "bash".into(),
        arguments: "{\"command\":\"echo noisy\"}".into(),
    });
    app.apply(UiEvent::ToolResult {
        name: "bash".into(),
        result: "noisy output".into(),
    });
    assert_eq!(
        app.last_assistant, "first answer",
        "tool logs are not copied as the assistant response"
    );
}

#[test]
fn transcript_text_serializes_lines() {
    let mut app = test_app("openai", "gpt-4o");
    app.push(Line::raw("one"));
    app.push(Line::from(vec![Span::raw("t"), Span::raw("wo")]));
    assert_eq!(app.transcript_text(), "one\ntwo");
}

#[test]
fn typed_incomplete_outcome_is_visible_after_tool_output_without_usage() {
    let mut app = test_app("openai", "gpt-4o");
    app.apply(UiEvent::ToolCall {
        name: "edit".into(),
        arguments: "{\"path\":\"src/main.rs\"}".into(),
    });
    app.apply(UiEvent::ToolResult {
        name: "edit".into(),
        result: "19 additions, 3 deletions".into(),
    });
    app.note_turn_outcome(&turn_outcome(
        hi_agent::TurnStatus::Incomplete,
        hi_agent::VerificationStatus::Unverified,
        hi_agent::ReviewStatus::NotRequired,
        hi_agent::TurnStopReason::Stalled,
    ));

    let lines: Vec<String> = app.transcript.iter().map(TranscriptEntry::text).collect();
    assert!(
        lines
            .iter()
            .any(|line| line.contains("incomplete · stalled")),
        "transcript: {lines:?}"
    );
    assert!(
        !lines
            .iter()
            .any(|line| line.contains("degraded in-session")),
        "transcript: {lines:?}"
    );
    assert_eq!(app.status, "warning · incomplete · stalled");
}

#[test]
fn failed_turn_is_visible() {
    let mut app = test_app("openai", "gpt-4o");
    app.note_turn_failed("request failed", "request", "wait a moment, then /retry");
    let line = app.transcript.last().unwrap().text();
    assert!(line.contains("✗ failed"), "got: {line}");
    assert!(line.contains("request failed"), "got: {line}");
    assert!(line.contains("request"), "got: {line}");
    assert!(line.contains("💡"), "shows guidance: {line}");
    assert!(
        app.status.contains("request"),
        "status has kind: {}",
        app.status
    );
}

#[test]
fn tool_protocol_failure_does_not_mark_model_degraded() {
    let mut app = test_app("pipenetwork", "pipe/auto-coder");
    let err: anyhow::Error = hi_ai::ProviderError::new(
        hi_ai::ProviderErrorKind::ToolProtocol,
        "model output did not satisfy the tool protocol",
    )
    .into();
    let (kind, guidance) = hi_agent::classify_error(&err);

    app.note_turn_failed(&format!("{err:#}"), kind, guidance);
    if hi_agent::ui::error_counts_as_model_issue(&err) {
        app.record_model_issue();
    }

    let lines: Vec<String> = app.transcript.iter().map(TranscriptEntry::text).collect();
    assert!(
        lines.iter().any(|line| line.contains("tool_protocol")),
        "transcript: {lines:?}"
    );
    assert!(
        !lines
            .iter()
            .any(|line| line.contains("degraded in-session")),
        "transcript: {lines:?}"
    );
    assert_eq!(app.model_issues.get("pipe/auto-coder"), None);
}

#[test]
fn route_rejection_failure_does_not_mark_model_degraded() {
    let mut app = test_app("pipenetwork", "pipe/auto-coder");
    let err: anyhow::Error = hi_ai::ProviderError::new(
        hi_ai::ProviderErrorKind::ModelUnavailable,
        "model temporarily unavailable",
    )
    .into();
    let (kind, guidance) = hi_agent::classify_error(&err);

    app.note_turn_failed(&format!("{err:#}"), kind, guidance);
    if hi_agent::ui::error_counts_as_model_issue(&err) {
        app.record_model_issue();
    }

    let lines: Vec<String> = app.transcript.iter().map(TranscriptEntry::text).collect();
    assert!(
        lines.iter().any(|line| line.contains("request")),
        "transcript: {lines:?}"
    );
    assert!(
        !lines
            .iter()
            .any(|line| line.contains("degraded in-session")),
        "transcript: {lines:?}"
    );
    assert_eq!(app.model_issues.get("pipe/auto-coder"), None);
}

#[test]
fn empty_tool_result_is_visible() {
    let mut app = test_app("openai", "gpt-4o");
    app.apply(UiEvent::ToolCall {
        name: "bash".into(),
        arguments: "{\"command\":\"true\"}".into(),
    });
    app.apply(UiEvent::ToolResult {
        name: "bash".into(),
        result: String::new(),
    });
    let rendered: Vec<String> = app.transcript.iter().map(TranscriptEntry::text).collect();
    assert!(
        rendered.iter().any(|line| line.contains("(no output)")),
        "transcript: {rendered:?}"
    );
}

#[test]
fn explore_tools_collapse_header_and_line_count_into_one_line() {
    let mut app = test_app("openai", "gpt-4o");
    // A read call: header is deferred until the result, then both collapse.
    app.apply(UiEvent::ToolCall {
        name: "read".into(),
        arguments: "{\"path\":\"src/main.rs\"}".into(),
    });
    let lines: Vec<String> = app.transcript.iter().map(TranscriptEntry::text).collect();
    // No header emitted yet — it waits for the result.
    assert!(
        !lines.iter().any(|l| l.contains("◆ read")),
        "no deferred header before result: {lines:?}"
    );
    app.apply(UiEvent::ToolResult {
        name: "read".into(),
        result: "a\nb\nc\n".into(),
    });
    let lines: Vec<String> = app.transcript.iter().map(TranscriptEntry::text).collect();
    // Exactly one line, combining the header and the count.
    assert!(
        lines
            .iter()
            .any(|l| l.contains("◆ read src/main.rs · 3 lines")),
        "collapsed read line: {lines:?}"
    );
    assert_eq!(
        lines.iter().filter(|l| l.contains("◆ read")).count(),
        1,
        "exactly one read header line: {lines:?}"
    );

    // grep with no matches shows "(no output)" in the same collapsed line.
    app.apply(UiEvent::ToolCall {
        name: "grep".into(),
        arguments: "{\"pattern\":\"foo\"}".into(),
    });
    app.apply(UiEvent::ToolResult {
        name: "grep".into(),
        result: String::new(),
    });
    let lines: Vec<String> = app.transcript.iter().map(TranscriptEntry::text).collect();
    assert!(
        lines.iter().any(|l| l.contains("◆ grep foo · (no output)")),
        "collapsed grep empty line: {lines:?}"
    );
}

#[test]
fn consecutive_same_tool_explore_results_merge_into_one_line() {
    let mut app = test_app("openai", "gpt-4o");
    // Three reads in a row should collapse to one summary line.
    app.apply(UiEvent::ToolCall {
        name: "read".into(),
        arguments: "{\"path\":\"a.rs\"}".into(),
    });
    app.apply(UiEvent::ToolResult {
        name: "read".into(),
        result: "a\nb\n".into(),
    });
    app.apply(UiEvent::ToolCall {
        name: "read".into(),
        arguments: "{\"path\":\"b.rs\"}".into(),
    });
    app.apply(UiEvent::ToolResult {
        name: "read".into(),
        result: "c\nd\ne\n".into(),
    });
    app.apply(UiEvent::ToolCall {
        name: "read".into(),
        arguments: "{\"path\":\"c.rs\"}".into(),
    });
    app.apply(UiEvent::ToolResult {
        name: "read".into(),
        result: "f\n".into(),
    });
    let lines: Vec<String> = app.transcript.iter().map(TranscriptEntry::text).collect();
    // Exactly one read line, summarizing all three.
    assert_eq!(
        lines.iter().filter(|l| l.contains("◆ read")).count(),
        1,
        "one merged read line: {lines:?}"
    );
    assert!(
        lines.iter().any(|l| l.contains("◆ read 3 files · 6 lines")),
        "merged summary: {lines:?}"
    );

    // A non-explore tool between reads breaks the run.
    app.apply(UiEvent::ToolCall {
        name: "edit".into(),
        arguments: "{\"path\":\"a.rs\"}".into(),
    });
    app.apply(UiEvent::ToolResult {
        name: "edit".into(),
        result: "ok".into(),
    });
    app.apply(UiEvent::ToolCall {
        name: "read".into(),
        arguments: "{\"path\":\"d.rs\"}".into(),
    });
    app.apply(UiEvent::ToolResult {
        name: "read".into(),
        result: "x\ny\n".into(),
    });
    let lines: Vec<String> = app.transcript.iter().map(TranscriptEntry::text).collect();
    // Now two read lines: the merged 3-file run and a fresh single read.
    assert_eq!(
        lines.iter().filter(|l| l.contains("◆ read")).count(),
        2,
        "run broken by edit: {lines:?}"
    );
    assert!(
        lines.iter().any(|l| l.contains("◆ read d.rs · 2 lines")),
        "fresh read after break: {lines:?}"
    );
}

#[test]
fn renders_fetching_spinner() {
    let mut app = test_app("pipenetwork", "ipop/coder-balanced");
    app.fetching = Some(Instant::now());
    let mut term = Terminal::new(TestBackend::new(60, 10)).unwrap();
    term.draw(|f| app.render(f)).unwrap();
    let screen = dump(&term);
    assert!(
        screen.contains("fetching models from pipenetwork"),
        "fetch spinner: {screen}"
    );
    assert!(screen.contains("Esc to cancel"), "cancel hint: {screen}");
}

#[test]
fn renders_model_picker() {
    let mut app = test_app("openai", "openai/gpt-4o");
    app.picker = Some(ModelPicker::new(
        vec!["anthropic/claude-sonnet-4".into(), "openai/gpt-4o".into()],
        "openai/gpt-4o",
        HashMap::new(),
        &HashMap::new(),
    ));
    let mut term = Terminal::new(TestBackend::new(60, 14)).unwrap();
    term.draw(|f| app.render(f)).unwrap();
    let screen = dump(&term);
    assert!(screen.contains("select a model"), "title: {screen}");
    assert!(screen.contains("filter:"), "filter line: {screen}");
    assert!(screen.contains("claude-sonnet-4"), "lists models: {screen}");
    assert!(screen.contains("▶"), "highlights a selection: {screen}");
    // The active model is marked and pre-selected.
    assert!(
        screen.contains("(current)"),
        "marks current model: {screen}"
    );
}

#[test]
fn picker_hides_health_tag() {
    let mut app = test_app("pipenetwork", "ipop/coder-balanced");
    let tags = HashMap::from([("claude-sonnet-4.6".to_string(), "degraded".to_string())]);
    app.picker = Some(ModelPicker::new(
        vec!["claude-sonnet-4.6".into(), "ipop/coder-balanced".into()],
        "ipop/coder-balanced",
        tags,
        &HashMap::new(),
    ));
    let mut term = Terminal::new(TestBackend::new(60, 14)).unwrap();
    term.draw(|f| app.render(f)).unwrap();
    let screen = dump(&term);
    assert!(
        !screen.contains("[degraded]"),
        "health tag should not be shown: {screen}"
    );
}

#[test]
fn renders_multiline_input() {
    let mut app = test_app("openai", "gpt-4o");
    app.input.insert_str("first\nsecond\nthird");
    let mut term = Terminal::new(TestBackend::new(40, 14)).unwrap();
    term.draw(|f| app.render(f)).unwrap();
    let screen = dump(&term);
    assert!(
        screen.contains("❯ first"),
        "first line with prompt: {screen}"
    );
    assert!(screen.contains("second"), "second line: {screen}");
    assert!(screen.contains("third"), "third line: {screen}");
}

#[test]
fn alt_enter_and_backslash_insert_newline_instead_of_submitting() {
    let mut app = test_app("openai", "gpt-4o");
    app.input.set("line one");
    let alt_enter = KeyEvent::new(KeyCode::Enter, KeyModifiers::ALT);
    assert_eq!(app.edit_key(&alt_enter), None, "alt+enter does not submit");
    assert_eq!(app.input.text(), "line one\n");

    // Trailing backslash + Enter continues the line (universal fallback).
    app.input.set("a\\");
    let enter = KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE);
    assert_eq!(app.edit_key(&enter), None, "backslash continues");
    assert_eq!(app.input.text(), "a\n");

    // A normal Enter still submits.
    app.input.set("go");
    assert_eq!(app.edit_key(&enter).as_deref(), Some("go"));
}

#[test]
fn failed_turn_shows_reason_and_keeps_error() {
    let mut app = test_app("openai", "gpt-4o");
    app.note_turn_failed(
        "API error 401: invalid or expired session",
        "auth",
        "check your API key",
    );
    // record_model_issue runs next in the real flow; it must NOT clobber the
    // real error with a reliability-count message.
    app.record_model_issue();
    assert_eq!(
        app.last_error.as_deref(),
        Some("API error 401: invalid or expired session"),
        "the real error is preserved for /status and /log"
    );
    // The bottom bar shows the reason inline, not a bare "failed".
    let mut term = Terminal::new(TestBackend::new(80, 8)).unwrap();
    term.draw(|f| app.render(f)).unwrap();
    let screen = dump(&term);
    assert!(
        screen.contains("last: failed — API error 401"),
        "reason inline: {screen}"
    );
    assert!(screen.contains("/retry"), "recovery hint: {screen}");
}

#[test]
fn backend_wait_notice_does_not_mark_model_degraded() {
    let mut app = test_app("pipenetwork", "ipop/coder-balanced");
    app.note_backend_waiting(Duration::from_secs(181), Duration::from_secs(180));

    assert_eq!(app.model_issues.get("ipop/coder-balanced"), None);
    assert_eq!(app.last_error, None);
    let mut term = Terminal::new(TestBackend::new(100, 8)).unwrap();
    term.draw(|f| app.render(f)).unwrap();
    let screen = dump(&term);
    assert!(
        screen.contains("Still thinking. Ctrl-C cancels; keep waiting to continue."),
        "soft wait notice shown: {screen}"
    );
    assert!(
        !screen.contains("degraded in-session"),
        "soft wait notice should not surface model health: {screen}"
    );
}

#[test]
fn watchdog_timeout_default_is_longer_than_client_warning_window() {
    assert_eq!(
        watchdog_stuck_timeout_from_value(None),
        Duration::from_secs(180)
    );
    assert_eq!(
        watchdog_stuck_timeout_from_value(Some("5")),
        Duration::from_secs(30)
    );
    assert_eq!(
        watchdog_stuck_timeout_from_value(Some("9999")),
        Duration::from_secs(1_800)
    );
}

#[test]
fn completion_opens_filters_and_closes() {
    let mut app = test_app("openai", "gpt-4o");
    app.input.set("/");
    app.sync_completion();
    assert_eq!(
        app.completion_items().len(),
        hi_agent::command::COMMANDS.len(),
        "bare slash lists every command"
    );
    app.input.set("/co");
    app.sync_completion();
    let labels: Vec<String> = app
        .completion_items()
        .iter()
        .map(|i| i.label.clone())
        .collect();
    assert!(
        labels.contains(&"/copy".to_string()) && labels.contains(&"/compact".to_string()),
        "got {labels:?}"
    );
    assert!(labels.iter().all(|n| n.starts_with("/co")));
    // A space after a command that takes no argument closes the menu.
    app.input.set("/diff ");
    app.sync_completion();
    assert!(app.completion.is_none());
}

#[test]
fn history_recall_of_slash_command_keeps_completion_closed() {
    let mut app = test_app("openai", "gpt-4o");
    app.input.history = vec!["ask first".into(), "/help".into(), "ask last".into()];
    let up = KeyEvent::new(KeyCode::Up, KeyModifiers::NONE);

    assert_eq!(app.edit_key(&up), None);
    app.sync_completion_after_edit_key(&up, false);
    assert_eq!(app.input.text(), "ask last");
    assert!(app.completion.is_none());

    assert_eq!(app.edit_key(&up), None);
    app.sync_completion_after_edit_key(&up, false);
    assert_eq!(app.input.text(), "/help");
    assert!(
        app.completion.is_none(),
        "history recall must not open slash completion"
    );

    assert_eq!(app.edit_key(&up), None);
    app.sync_completion_after_edit_key(&up, false);
    assert_eq!(app.input.text(), "ask first");
    assert!(app.completion.is_none());
}

#[test]
fn history_search_recall_of_slash_command_keeps_completion_closed() {
    let mut app = test_app("openai", "gpt-4o");
    app.input.history = vec!["ask first".into(), "/help".into()];
    let mut search = HistorySearch::default();
    search.refilter(&app.input.history);
    app.history_search = Some(search);

    let esc = KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE);
    let history_search_was_active = app.history_search.is_some();
    assert_eq!(app.edit_key(&esc), None);
    app.sync_completion_after_edit_key(&esc, history_search_was_active);

    assert_eq!(app.input.text(), "/help");
    assert!(
        app.completion.is_none(),
        "loading a slash command from Ctrl-R should leave arrows for history"
    );

    app.sync_completion();
    assert!(
        app.completion.is_some(),
        "normal slash completion remains available outside history recall"
    );
}

#[test]
fn completion_offers_verify_and_goal_keywords() {
    let mut app = test_app("openai", "gpt-4o");
    app.input.set("/verify ");
    app.sync_completion();
    let labels: Vec<String> = app
        .completion_items()
        .iter()
        .map(|i| i.label.clone())
        .collect();
    assert_eq!(labels, vec!["off"], "verify offers its disable keyword");
    app.input.set("/goal cl");
    app.sync_completion();
    let labels: Vec<String> = app
        .completion_items()
        .iter()
        .map(|i| i.label.clone())
        .collect();
    assert_eq!(labels, vec!["clear"], "goal offers its clear keyword");
    assert_eq!(app.accept_completion(true).as_deref(), Some("/goal clear"));
}

#[test]
fn completion_offers_live_model_ids() {
    let mut app = test_app("openai", "gpt-4o");
    app.model_ids = vec!["gpt-4o".into(), "gpt-4o-mini".into(), "claude-opus".into()];
    app.input.set("/model gp");
    app.sync_completion();
    let labels: Vec<String> = app
        .completion_items()
        .iter()
        .map(|i| i.label.clone())
        .collect();
    assert_eq!(
        labels,
        vec!["gpt-4o", "gpt-4o-mini"],
        "filters the catalog by prefix"
    );
    // Accepting a row runs the full command.
    app.completion.as_mut().unwrap().selected = 1;
    assert_eq!(
        app.accept_completion(true).as_deref(),
        Some("/model gpt-4o-mini")
    );

    // With no catalog loaded, there's no inline menu — the picker still
    // handles `/model` (so the feature degrades, it doesn't break).
    let mut bare = test_app("openai", "gpt-4o");
    bare.input.set("/model gp");
    bare.sync_completion();
    assert!(bare.completion.is_none());
}

#[test]
fn sessions_completion_offers_subcommands_then_live_ids() {
    let mut app = test_app("openai", "gpt-4o");
    app.session_lister = Some(Box::new(|| {
        vec![
            LocalSessionInfo {
                id: "1783895144561".into(),
                title: "portal work".into(),
                age: "2m".into(),
                lines: 12,
            },
            LocalSessionInfo {
                id: "1783894593132".into(),
                title: "other work".into(),
                age: "8m".into(),
                lines: 4,
            },
        ]
    }));

    app.input.set("/sessions sw");
    app.sync_completion();
    assert_eq!(app.completion_items()[0].label, "switch");
    assert_eq!(app.accept_completion(true), None);
    assert_eq!(app.input.text(), "/sessions switch ");

    app.sync_completion();
    assert_eq!(app.completion_items().len(), 2);
    assert_eq!(
        app.accept_completion(true).as_deref(),
        Some("/sessions switch 1783895144561")
    );

    app.input.set("/sessions rename 1783894");
    app.sync_completion();
    assert_eq!(app.accept_completion(true), None);
    assert_eq!(app.input.text(), "/sessions rename 1783894593132 ");
}

#[test]
fn session_completion_does_not_rescan_files_for_each_prefix_or_render() {
    let mut app = test_app("openai", "gpt-4o");
    let calls = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let observed = calls.clone();
    app.session_lister = Some(Box::new(move || {
        observed.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        vec![LocalSessionInfo {
            id: "1783895144561".into(),
            title: "portal work".into(),
            age: "2m".into(),
            lines: 12,
        }]
    }));

    app.input.set("/sessions switch ");
    app.sync_completion();
    assert_eq!(app.completion_items().len(), 1);
    app.input.set("/sessions switch 178");
    app.sync_completion();
    for _ in 0..5 {
        assert_eq!(app.completion_items().len(), 1);
    }
    assert_eq!(calls.load(std::sync::atomic::Ordering::Relaxed), 1);
}

#[test]
fn completion_offers_then_fills_compact_kinds() {
    let mut app = test_app("openai", "gpt-4o");
    // The space that used to kill the menu now offers the kinds.
    app.input.set("/compact ");
    app.sync_completion();
    let labels: Vec<String> = app
        .completion_items()
        .iter()
        .map(|i| i.label.clone())
        .collect();
    assert_eq!(labels, vec!["hybrid", "full", "elide"], "offers every kind");
    // Typing narrows by prefix.
    app.input.set("/compact e");
    app.sync_completion();
    let labels: Vec<String> = app
        .completion_items()
        .iter()
        .map(|i| i.label.clone())
        .collect();
    assert_eq!(labels, vec!["elide"]);
    // Accepting a kind fills the whole command and runs it on Enter.
    assert_eq!(
        app.accept_completion(true).as_deref(),
        Some("/compact elide")
    );
    assert!(app.completion.is_none(), "menu closes after accept");
}

#[test]
fn completing_compact_name_opens_its_kind_menu() {
    let mut app = test_app("openai", "gpt-4o");
    app.input.set("/compact");
    app.sync_completion();
    // Tab accepts the command name, leaving `/compact `…
    app.accept_completion(false);
    assert_eq!(app.input.text(), "/compact ");
    // …and the re-sync the Tab handler performs opens the kind menu.
    app.sync_completion();
    let labels: Vec<String> = app
        .completion_items()
        .iter()
        .map(|i| i.label.clone())
        .collect();
    assert!(labels.contains(&"hybrid".to_string()), "got {labels:?}");
}

#[test]
fn completion_navigation_and_accept() {
    let mut app = test_app("openai", "gpt-4o");
    // No-arg command: Enter accepts and submits immediately.
    app.input.set("/hel");
    app.sync_completion();
    let line = app.accept_completion(true);
    assert_eq!(line.as_deref(), Some("/help"));
    assert!(app.completion.is_none(), "menu closes after accept");

    // Arg-taking command: accept leaves a trailing space, does not submit.
    app.input.set("/mod");
    app.sync_completion();
    assert_eq!(
        app.accept_completion(true),
        None,
        "arg command waits for input"
    );
    assert_eq!(app.input.text(), "/model ");

    // Tab never submits, even for a no-arg command.
    app.input.set("/dif");
    app.sync_completion();
    assert_eq!(app.accept_completion(false), None);
    assert_eq!(app.input.text(), "/diff");
}

#[test]
fn completion_move_clamps() {
    let mut app = test_app("openai", "gpt-4o");
    app.input.set("/co"); // [commit, compact, context, copy]
    app.sync_completion();
    let last = app.completion_items().len().saturating_sub(1);
    app.completion_move(-1); // already at 0, stays
    assert_eq!(app.completion.as_ref().unwrap().selected, 0);
    app.completion_move(1);
    assert_eq!(app.completion.as_ref().unwrap().selected, 1);
    // Move past the end to verify clamping.
    for _ in 0..last + 1 {
        app.completion_move(1);
    }
    assert_eq!(app.completion.as_ref().unwrap().selected, last);
}

#[test]
fn renders_completion_menu() {
    let mut app = test_app("openai", "gpt-4o");
    app.input.set("/");
    app.sync_completion();
    let mut term = Terminal::new(TestBackend::new(72, 20)).unwrap();
    term.draw(|f| app.render(f)).unwrap();
    let screen = dump(&term);
    assert!(screen.contains("/help"), "lists help: {screen}");
    assert!(screen.contains("/model"), "lists model: {screen}");
    assert!(screen.contains("▶"), "highlights a row: {screen}");
}

/// `splash_lines` now leads with a ~2x block-letter "PipeNetwork.AI"
/// banner (5 figlet rows, all orange bold), then the dim model line, the
/// cwd, and a trailing blank. Verifies banner shape, orange+bold styling
/// on every banner row, the model/cwd content, and the line count.
#[test]
fn splash_shows_full_pipenetwork_wordmark_in_orange() {
    let lines = splash_lines("openai", "gpt-4o", Some(128_000));

    // 5 banner rows + model line + cwd line + trailing blank = 8.
    assert_eq!(
        lines.len(),
        8,
        "expected 8 lines (5 banner + model + cwd + blank)"
    );

    // Banner: rows 0..5, each one span styled orange + bold, carrying
    // figlet strokes (pipes / underscores).
    for (i, line) in lines[0..5].iter().enumerate() {
        assert_eq!(
            line.spans.len(),
            1,
            "banner row {i} should be a single span, got {} spans",
            line.spans.len()
        );
        let span = &line.spans[0];
        assert_eq!(
            span.style.fg,
            Some(Color::Rgb(255, 140, 0)),
            "banner row {i} should be orange"
        );
        assert!(
            span.style.add_modifier.contains(Modifier::BOLD),
            "banner row {i} should be bold"
        );
        let text = span.content.to_string();
        assert!(
            text.contains('|') || text.contains('_'),
            "banner row {i} should carry figlet strokes, got: {text:?}"
        );
    }

    // Row 5: dim model line (model + provider + context window).
    let model_line: String = lines[5]
        .spans
        .iter()
        .map(|s| s.content.to_string())
        .collect();
    assert!(
        model_line.contains("gpt-4o"),
        "model line missing model: {model_line:?}"
    );
    assert!(
        model_line.contains("openai"),
        "model line missing provider: {model_line:?}"
    );
    assert!(
        model_line.contains("128K context"),
        "model line missing context window: {model_line:?}"
    );

    // Row 6: cwd — non-empty.
    let cwd_line: String = lines[6]
        .spans
        .iter()
        .map(|s| s.content.to_string())
        .collect();
    assert!(
        !cwd_line.is_empty(),
        "cwd line should be non-empty, got: {cwd_line:?}"
    );

    // Row 7: the blank breathing-room line.
    assert!(
        lines[7].spans.is_empty(),
        "last line should be the blank breathing-room line"
    );
}

#[test]
fn uievent_serializes_and_deserializes_roundtrip() {
    use crate::event::UiEvent;
    use hi_agent::{PlanStatus, PlanStep};

    // Every variant must round-trip through serde JSON.
    let cases = vec![
        UiEvent::Text {
            text: "hello".to_string(),
        },
        UiEvent::Reasoning {
            text: "thinking...".to_string(),
        },
        UiEvent::AssistantEnd,
        UiEvent::ToolStarted {
            name: "bash".to_string(),
            arguments: r#"{"command":"ls"}"#.to_string(),
        },
        UiEvent::ToolCall {
            name: "edit".to_string(),
            arguments: r#"{"path":"a.rs"}"#.to_string(),
        },
        UiEvent::ToolResult {
            name: "bash".to_string(),
            result: "ok".to_string(),
        },
        UiEvent::ToolStream {
            name: "bash".to_string(),
            line: "compiling...".to_string(),
        },
        UiEvent::Status {
            text: "running".to_string(),
        },
        UiEvent::Plan {
            steps: vec![
                PlanStep {
                    title: "step 1".to_string(),
                    status: PlanStatus::Done,
                },
                PlanStep {
                    title: "step 2".to_string(),
                    status: PlanStatus::Active,
                },
            ],
        },
        UiEvent::Usage {
            prompt: 100,
            generated: 50,
            ctx_used: 1000,
            ctx_window: Some(8000),
            estimated: false,
        },
        UiEvent::RateLimits { rate_limits: None },
        UiEvent::TurnEnd {
            summary: "[100 in · 50 out]".to_string(),
        },
        UiEvent::TurnError {
            error_kind: "rate_limit".to_string(),
            message: "too many requests".to_string(),
            guidance: "wait and retry".to_string(),
        },
        UiEvent::ChangedFiles {
            files: vec!["a.rs".to_string(), "b.rs".to_string()],
        },
    ];

    for original in &cases {
        let json = serde_json::to_string(original).unwrap();
        let decoded: UiEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(
            serde_json::to_string(&decoded).unwrap(),
            json,
            "round-trip mismatch for {json}"
        );
    }

    // Verify the tagged format: each event has a "kind" field.
    let text_json = serde_json::to_string(&UiEvent::Text {
        text: "hi".to_string(),
    })
    .unwrap();
    assert!(
        text_json.contains(r#""kind":"text""#),
        "text event should have kind tag: {text_json}"
    );
    assert!(
        text_json.contains(r#""text":"hi""#),
        "text event should have text field: {text_json}"
    );

    // Verify the TurnError uses error_kind (not kind, which conflicts with the tag).
    let error_json = serde_json::to_string(&UiEvent::TurnError {
        error_kind: "auth".to_string(),
        message: "bad key".to_string(),
        guidance: "check key".to_string(),
    })
    .unwrap();
    assert!(
        error_json.contains(r#""error_kind":"auth""#),
        "turn_error should use error_kind field: {error_json}"
    );
    assert!(
        !error_json.contains(r#""kind":"auth""#),
        "turn_error must not use kind for the error type (conflicts with tag): {error_json}"
    );
}

/// A visual smoke of the Phase-1 transcript grammar: prints the rendered screen
/// (run with `--nocapture`) and asserts the block-accent markers are present.
#[test]
fn phase1_visual_grammar_smoke() {
    let mut app = test_app("pipe", "glm-5.2");
    app.push(ratatui::text::Line::styled(
        "❯ port the parser to the new API",
        ratatui::style::Style::default().fg(crate::theme::theme().accent_user),
    ));
    app.apply(UiEvent::ToolCall {
        name: "read".into(),
        arguments: "{\"path\":\"src/parser.rs\"}".into(),
    });
    app.apply(UiEvent::ToolResult {
        name: "read".into(),
        result: "a\nb\nc\n".into(),
    });
    app.apply(UiEvent::ToolCall {
        name: "bash".into(),
        arguments: "{\"command\":\"cargo test\"}".into(),
    });
    app.apply(UiEvent::ToolResult {
        name: "bash".into(),
        result: "running 3 tests\ntest result: ok".into(),
    });
    app.apply(UiEvent::Status {
        text: "🔍 skeptic approved — advancing".into(),
    });
    app.apply(UiEvent::Text {
        text: "Done. The parser now uses the new API.\n".into(),
    });
    app.apply(UiEvent::AssistantEnd);
    app.apply(UiEvent::ChangedFiles {
        files: vec!["src/parser.rs".into()],
    });

    // Exercise the chrome: context chip + input.
    app.context_used = 42000;
    app.context_window = Some(128000);
    let mut term = Terminal::new(TestBackend::new(72, 22)).unwrap();
    term.draw(|f| app.render(f)).unwrap();
    let screen = dump(&term);
    println!("\n{screen}");

    assert!(screen.contains("❯ port the parser"), "user prompt band");
    assert!(
        screen.contains("◆ read src/parser.rs"),
        "read header with ◆"
    );
    assert!(screen.contains("◆ bash"), "bash header with ◆");
    assert!(screen.contains("skeptic approved"), "status line present");
    assert!(screen.contains("┃"), "accent gutter present");
    assert!(screen.contains("✎ 1 file changed"), "changed-files line");
}

#[test]
fn long_tool_output_folds_to_preview_and_expands_on_ctrl_o() {
    let mut app = test_app("pipe", "glm-5.2");
    // 40 lines of bash output — well over the preview cap.
    let output = (0..40)
        .map(|i| format!("line {i}"))
        .collect::<Vec<_>>()
        .join("\n");
    app.apply(UiEvent::ToolCall {
        name: "bash".into(),
        arguments: "{\"command\":\"seq 40\"}".into(),
    });
    app.apply(UiEvent::ToolResult {
        name: "bash".into(),
        result: output,
    });

    // Collapsed (default): preview lines + a fold footer, not all 40.
    let collapsed: Vec<String> = app
        .transcript
        .iter()
        .flat_map(|e| e.flatten(false, false))
        .map(|l| crate::render::line_text(&l))
        .collect();
    assert!(
        collapsed.iter().any(|l| l.contains("line 0")),
        "preview shows the head: {collapsed:?}"
    );
    assert!(
        !collapsed.iter().any(|l| l.contains("line 39")),
        "the tail is folded away when collapsed"
    );
    assert!(
        collapsed
            .iter()
            .any(|l| l.contains("more lines · Ctrl-O to expand")),
        "a fold footer names the hidden lines: {collapsed:?}"
    );

    // Expanded (Ctrl-O / show_tool_output): the full body, no footer.
    let expanded: Vec<String> = app
        .transcript
        .iter()
        .flat_map(|e| e.flatten(false, true))
        .map(|l| crate::render::line_text(&l))
        .collect();
    assert!(
        expanded.iter().any(|l| l.contains("line 39")),
        "expanded shows the whole output"
    );
    assert!(
        !expanded.iter().any(|l| l.contains("Ctrl-O to expand")),
        "no fold footer when expanded"
    );

    // Short output (≤ preview) is never folded — no regression from the old
    // inline behavior.
    let mut app2 = test_app("pipe", "glm-5.2");
    app2.apply(UiEvent::ToolResult {
        name: "bash".into(),
        result: "just one line".into(),
    });
    let short: Vec<String> = app2
        .transcript
        .iter()
        .flat_map(|e| e.flatten(false, false))
        .map(|l| crate::render::line_text(&l))
        .collect();
    assert!(short.iter().any(|l| l.contains("just one line")));
    assert!(
        !short.iter().any(|l| l.contains("Ctrl-O")),
        "short output isn't folded"
    );

    // Full text (for /copy and /export) always has everything, regardless of fold.
    let full = app
        .transcript
        .iter()
        .map(TranscriptEntry::text)
        .collect::<Vec<_>>()
        .join("\n");
    assert!(full.contains("line 39"), "copy/export keeps the full body");
}

#[test]
fn tool_output_body_carries_panel_background_when_theme_paints() {
    // Mutation-free: flatten reads the active theme; assert the body-line bg is
    // consistent with whatever palette is active (panel on truecolor, none on
    // ansi). This never touches the global mode, so it can't race other tests.
    let th = crate::theme::theme();
    let body: Vec<Line<'static>> = vec![Line::raw("a line of output")];
    let entry = TranscriptEntry::ToolOutput {
        body,
        expanded: false,
    };
    let flat = entry.flatten(false, true);
    let bg = flat[0].style.bg;
    if th.paints_backgrounds() {
        assert_eq!(
            bg,
            Some(th.panel),
            "truecolor themes sink the body into a panel"
        );
    } else {
        assert_eq!(
            bg, None,
            "ansi theme leaves the body background at terminal default"
        );
    }
}

#[test]
fn sticky_prompt_header_pins_when_scrolled_past() {
    let mut app = test_app("pipe", "glm-5.2");
    app.push_user_prompt(Line::styled(
        "❯ first question about the parser",
        Style::default().fg(crate::theme::theme().accent_user),
    ));
    // A long block of output so the prompt scrolls off the top.
    for i in 0..60 {
        app.push(Line::raw(format!("output line {i}")));
    }
    let mut term = Terminal::new(TestBackend::new(60, 12)).unwrap();

    // First render (following = bottom-pinned): NO sticky, top row is the border.
    term.draw(|f| app.render(f)).unwrap();
    let bottom_pinned = dump(&term);
    let first_content_row = bottom_pinned.lines().nth(1).unwrap_or("");
    assert!(
        !first_content_row.contains("first question"),
        "while following, the prompt is not pinned: {first_content_row:?}"
    );

    // Scroll up to the top: the prompt is now above the viewport → pinned.
    app.following = false;
    app.scroll = 0; // top of transcript
    // At scroll 0 the prompt IS visible (offset 0 == scroll), so it should NOT
    // pin. Now scroll down past it.
    app.scroll = 30;
    term.draw(|f| app.render(f)).unwrap();
    let scrolled = dump(&term);
    let top_content_row = scrolled.lines().nth(1).unwrap_or("");
    assert!(
        top_content_row.contains("first question"),
        "the governing prompt pins to the top when scrolled past: {top_content_row:?}"
    );
}

/// `/provider xai` switches to a provider preset without creating a profile,
/// so the active name need not name one. Selecting a model then had nothing to
/// persist into and surfaced "couldn't save model to active profile: no profile
/// named 'xai'" over an otherwise successful switch.
#[test]
fn selecting_a_model_on_a_provider_preset_does_not_error_about_a_missing_profile() {
    let mut app = test_app("xai", "grok-4.3");
    // No profiles configured; the active name is a provider preset.
    app.active_profile = Some("xai".to_string());

    let saved = app
        .persist_active_profile_model("grok-4.5")
        .expect("a preset with no profile must not be an error");
    assert_eq!(saved, None, "nothing to save into, so no profile name back");
}

/// The guard must not be over-broad: a name that IS a configured profile still
/// reaches the loader/saver. The test scaffolding's loader always errors, so
/// reaching it at all is the signal — an `Ok(None)` here would mean the guard
/// had swallowed a real profile and silently stopped persisting model choices.
#[test]
fn a_configured_profile_still_reaches_the_persist_path() {
    let mut app = test_app("xai", "grok-4.3");
    app.profiles = vec![crate::ProfileInfo {
        name: "work".into(),
        provider: "xai".into(),
        model: Some("grok-4.3".into()),
        base_url: None,
    }];
    app.active_profile = Some("work".to_string());

    assert!(
        app.persist_active_profile_model("grok-4.5").is_err(),
        "a configured profile must go through the loader, not be skipped"
    );
}
