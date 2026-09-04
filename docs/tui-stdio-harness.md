# Deterministic TUI stdio harness

`hi debug tui --stdio` drives Hi's production `App` without taking over a
terminal. It reads newline-delimited JSON commands from stdin, applies them to
the real TUI state, renders through ratatui's `TestBackend`, and writes exactly
one JSON response per non-empty input line.

This is intended for deterministic UI tests, agent-driven inspection, and
reproducible bug reports. It does not load provider configuration, contact a
model, or execute tools. Interactive `hi` behavior is unchanged.

## Framing and responses

The protocol version is `1`. A request may include any JSON `id`; a parsed
request echoes it in the response.

```json
{"id":1,"command":"hello"}
```

Successful responses have `ok: true` and a `result`. Failures have `ok: false`
and a typed `error`. A malformed line does not terminate the stream, so a
long-running test driver can report an error and continue.

```json
{"protocol_version":1,"id":1,"ok":true,"result":{"commands":["hello"]}}
{"protocol_version":1,"ok":false,"error":{"code":"invalid_json","message":"..."}}
```

Input lines are capped at 1 MiB. Render dimensions must be between `1x1` and
`512x256` inclusive.

## Commands

### Handshake and reset

`hello` reports supported commands and the component-tree, session-projection,
session-event, and reducer schema versions.

`reset` creates a fresh App and session projection. Its optional defaults are
`width: 80`, `height: 24`, `provider: "debug"`, and
`model: "debug-model"`.

```json
{"command":"reset","width":100,"height":30,"provider":"openai","model":"gpt-test"}
```

### Synthetic terminal input

`resize` changes the next `TestBackend` dimensions. `focus` applies the same
focus state used by terminal focus events.

```json
{"command":"resize","width":64,"height":18}
{"command":"focus","focused":false}
```

`paste` inserts text through the production multiline composer. `key` routes a
synthetic key through the production action/editing path. A submitted prompt is
returned in `result.submitted` and becomes a real user-prompt transcript block.

```json
{"command":"paste","text":"review src/lib.rs"}
{"command":"key","key":"left"}
{"command":"key","key":"enter"}
{"command":"key","key":"t","ctrl":true}
```

Named keys are `enter`, `esc`, `backspace`, `delete`, `left`, `right`, `up`,
`down`, `home`, `end`, `page_up`, `page_down`, `tab`, `back_tab`, and `space`.
A one-character string or `char:<character>` represents a character key.
`ctrl`, `alt`, and `shift` default to false.

Clipboard, external-editor, and voice chords are rejected because they require
external devices or processes and would make a stdio test nondeterministic.

### Live transcript lifecycle

`transcript` accepts the existing serialized `hi_tui::event::UiEvent`. This
exercises the same streaming and transcript-block reducer as the live agent UI.
For example, a complete assistant/tool lifecycle is:

```json
{"command":"transcript","event":{"kind":"reasoning","text":"checking"}}
{"command":"transcript","event":{"kind":"text","text":"I found it."}}
{"command":"transcript","event":{"kind":"assistant_end"}}
{"command":"transcript","event":{"kind":"tool_call","name":"bash","arguments":"{\"command\":\"pwd\"}"}}
{"command":"transcript","event":{"kind":"tool_result","name":"bash","result":"/workspace"}}
```

Other `UiEvent` variants use their ordinary tagged JSON representation. The
`clear_transcript` command clears presentation blocks and in-flight stream
buffers but intentionally leaves the durable session projection alone.

### Versioned session projection

The harness uses `hi_agent::SessionProjection`, the same versioned transport
around the deterministic session reducer used by other presentation clients.
There is no harness-specific session model.

`session_event` applies one `SessionEvent` by preparing and atomically applying
an exact-base projection patch. `session_patch` consumes a serialized
`SessionProjectionPatch`; stale bases, revision drift, unsupported versions,
and digest mismatches are typed `invalid_session_projection` errors.
`session_snapshot` validates and installs a complete
`SessionProjectionSnapshot`.

```json
{"command":"session_event","event":{"schema_version":1,"sequence":1,"kind":{"type":"message","message":{"role":"User","content":[{"Text":"hello"}]}}}}
```

After any projection command, the production App is rebuilt from the projected
messages, goal, plan, pause state, and usage. Projection mutations return the
new integrity-sealed snapshot. `inspect` also emits that snapshot, so a client
can transfer it into another harness with `session_snapshot` or prepare an
exact-base tail patch using the public projection contract.

### Render and inspect

`render` invokes `App::render` with a fresh `TestBackend`. It returns the fixed
dimensions, cursor position, trailing-space-trimmed screen rows, and a BLAKE3
digest over the dimensions and rows. Time-derived live fields are normalized
to zero-duration before rendering.

```json
{"command":"render"}
```

`inspect` renders first, then returns:

- `render_digest`, matching the current textual render;
- a versioned semantic component tree rooted at `app`, including transcript
  blocks, composer state, focus, actual cached hit-target rectangles, and
  visible overlays; and
- the current integrity-sealed `SessionProjectionSnapshot`.

```json
{"command":"inspect"}
```

Transcript node IDs are stable ordinals such as `transcript.block.0`. Their
`kind` values distinguish user prompts, assistant messages, reasoning,
workflows, tool output, and typed activity rows (`activity_run`,
`activity_edit`, `activity_explore`, `activity_subagent`, or `activity_other`).
Foldable blocks expose `foldable`, `expanded`, and selection focus.

## Shell example

```sh
printf '%s\n' \
  '{"command":"paste","text":"hello"}' \
  '{"command":"key","key":"enter"}' \
  '{"command":"transcript","event":{"kind":"text","text":"Hi."}}' \
  '{"command":"transcript","event":{"kind":"assistant_end"}}' \
  '{"command":"inspect"}' \
  | hi debug tui --stdio
```
