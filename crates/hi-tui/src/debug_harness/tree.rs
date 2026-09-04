use std::collections::BTreeMap;

use ratatui::layout::Rect;
use serde::Serialize;
use serde_json::{Value, json};

use super::TUI_COMPONENT_TREE_SCHEMA_VERSION;
use crate::activity_feed::ActivityKind;
use crate::mode::UiMode;
use crate::{App, TranscriptEntry};

pub(super) fn build(app: &App, width: u16, height: u16, revision: u64) -> ComponentTree {
    let mut transcript = ComponentNode::new("transcript", "transcript");
    transcript.bounds = rect(app.view_inner);
    let selected = app.mode.is_block_nav().then(|| app.selected_block_ord());
    let projected = app.projected_transcript_identities();
    let mut foldable = 0usize;
    for (index, entry) in app.transcript.iter().enumerate() {
        let is_foldable = entry.is_foldable();
        let identity = projected.get(index).and_then(Option::as_ref);
        let node_id = identity.map_or_else(
            || format!("transcript.block.{index}"),
            |identity| format!("transcript.block.{}", identity.id),
        );
        let mut node = ComponentNode::new(node_id, transcript_kind(entry));
        node.text = Some(entry.text());
        node.focused = is_foldable && selected == Some(foldable);
        node.attributes
            .insert("foldable".into(), json!(is_foldable));
        if let Some(identity) = identity {
            node.attributes
                .insert("stable_id".into(), json!(identity.id));
            node.attributes.insert(
                "lifecycle".into(),
                json!(identity.terminal.unwrap_or("open")),
            );
        }
        if let Some(expanded) = entry_expanded(entry) {
            node.attributes.insert("expanded".into(), json!(expanded));
        }
        transcript.children.push(node);
        foldable += usize::from(is_foldable);
    }

    let mut composer = ComponentNode::new("composer", "composer");
    composer.focused = app.focused && !app.mode.is_normal();
    composer.text = Some(app.input.text());
    composer
        .attributes
        .insert("cursor".into(), json!(app.input.cursor()));
    composer
        .attributes
        .insert("mode".into(), json!(mode_name(&app.mode)));

    let mut root = ComponentNode::new("app", "app");
    root.bounds = Some(RectSnapshot {
        x: 0,
        y: 0,
        width,
        height,
    });
    root.focused = app.focused;
    root.children.push(transcript);
    root.children.push(composer);
    push_rect_node(&mut root, "status.context", app.ctx_chip_rect);
    push_rect_node(&mut root, "status.turn", app.turn_status_rect);
    push_rect_node(&mut root, "timeline", app.timeline_rect);
    push_rect_node(&mut root, "changed_files", app.changed_files_rect);
    push_rect_node(&mut root, "overlay.btw", app.last_btw_area);
    if app.show_help {
        root.children
            .push(ComponentNode::new("overlay.help", "help"));
    }
    if app.palette.is_some() {
        root.children
            .push(ComponentNode::new("overlay.palette", "palette"));
    }
    ComponentTree {
        schema_version: TUI_COMPONENT_TREE_SCHEMA_VERSION,
        revision,
        root,
    }
}

#[derive(Serialize)]
pub(super) struct ComponentTree {
    schema_version: u16,
    revision: u64,
    root: ComponentNode,
}

#[derive(Serialize)]
struct ComponentNode {
    id: String,
    kind: String,
    visible: bool,
    focused: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    bounds: Option<RectSnapshot>,
    #[serde(skip_serializing_if = "Option::is_none")]
    text: Option<String>,
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    attributes: BTreeMap<String, Value>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    children: Vec<ComponentNode>,
}

impl ComponentNode {
    fn new(id: impl Into<String>, kind: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            kind: kind.into(),
            visible: true,
            focused: false,
            bounds: None,
            text: None,
            attributes: BTreeMap::new(),
            children: Vec::new(),
        }
    }
}

#[derive(Clone, Copy, Serialize)]
struct RectSnapshot {
    x: u16,
    y: u16,
    width: u16,
    height: u16,
}

fn transcript_kind(entry: &TranscriptEntry) -> &'static str {
    match entry {
        TranscriptEntry::Line(_) => "line",
        TranscriptEntry::UserPrompt { .. } => "user_prompt",
        TranscriptEntry::Assistant(_) => "assistant",
        TranscriptEntry::AssistantMessage { .. } => "assistant_message",
        TranscriptEntry::Reasoning { .. } => "reasoning",
        TranscriptEntry::Btw { .. } => "btw",
        TranscriptEntry::Workflow { .. } => "workflow",
        TranscriptEntry::Activity(block) => match &block.kind {
            ActivityKind::VerbGroup(_) => "activity_explore",
            ActivityKind::Edit { .. } => "activity_edit",
            ActivityKind::Run { .. } => "activity_run",
            ActivityKind::Other { .. } => "activity_other",
            ActivityKind::Subagent { .. } => "activity_subagent",
        },
        TranscriptEntry::ToolOutput { .. } => "tool_output",
    }
}

fn entry_expanded(entry: &TranscriptEntry) -> Option<bool> {
    match entry {
        TranscriptEntry::Btw { expanded, .. } | TranscriptEntry::ToolOutput { expanded, .. } => {
            Some(*expanded)
        }
        TranscriptEntry::Activity(block) if block.is_foldable() => Some(block.expanded),
        _ => None,
    }
}

fn mode_name(mode: &UiMode) -> &'static str {
    match mode {
        UiMode::Insert => "insert",
        UiMode::Normal { search: Some(_) } => "normal_search",
        UiMode::Normal { search: None } => "normal",
        UiMode::BlockNav => "block_nav",
        UiMode::HistorySearch(_) => "history_search",
        UiMode::Review => "review",
    }
}

fn rect(value: Rect) -> Option<RectSnapshot> {
    (value.width > 0 && value.height > 0).then_some(RectSnapshot {
        x: value.x,
        y: value.y,
        width: value.width,
        height: value.height,
    })
}

fn push_rect_node(root: &mut ComponentNode, id: &'static str, bounds: Rect) {
    if let Some(bounds) = rect(bounds) {
        let mut node = ComponentNode::new(id, id);
        node.bounds = Some(bounds);
        root.children.push(node);
    }
}
