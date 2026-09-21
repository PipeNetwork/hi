# `/review` — spec-coverage audit + bounded fix loop

`/review` reads your `plan.md` and `spec.md`, audits the codebase against
them, tells you what is implemented, partial, or missing, and then fixes the
**P0/P1 defects** it found in a bounded audit -> fix -> re-audit loop.
Missing plan or spec items are reported, never auto-implemented.

Everything user-facing says "spec review"; the Ctrl-G diff review pane keeps
the bare word.

```
❯ /review
spec review · audit · up to 3 fix passes · plan.md + docs/spec.md
❯ spec review · audit                                  (transcript echo)
  … the model reads plan/spec, repo_map, cited files, runs the tests …
spec review · fix pass 1/3 · 2 P0/P1 · plan.md + docs/spec.md
  [P0] Reject empty nicknames — src/server.rs:120     (plan panel fills)
  [P1] KICK ignores operator status — src/server.rs:188
  … edits, cargo test, plan steps close …
spec review · re-audit after pass 1/3 · plan.md + docs/spec.md
spec review complete: 5/6 plan/spec items implemented · no P0/P1 defects after 1 fix pass(es) · 1 P2/P3 reported
coverage (incomplete):
  implemented  Welcome banner              src/server.rs:41
  missing      /topic command              -
findings:
  [P2] Log unknown commands — src/server.rs:163
```

## Usage

| command | does |
|---|---|
| `/review` | audit, then fix P0/P1 and re-audit until clean, paused, or capped (3 passes) |
| `/review audit` (`report`) | audit and report only; no fix loop |
| `/review status` | current phase, pass, inputs; after a run, the coverage matrix and findings |
| `/review stop` (`cancel`, `off`) | end the loop and drop its fix checklist |
| `/review [audit] <path>…` | explicit plan/spec files and/or scope directories |
| `/review [audit] all` | audit the whole workspace in chunks, one turn per top-level directory (or crate) |
| Esc during a review turn | pauses the loop; empty Enter on the ghost `/review` resumes the same phase |

Paths may be documents (`docs/spec-v2.md plan.md`) or directories
(`crates/hi-tui`). A directory scopes the audit; a file is classified by name
(`plan`, `spec`, `readme`, otherwise `doc`). A path that does not exist
refuses to start: `spec review: no such file: docs/nope.md`. `all` is a
keyword anywhere after the action (a directory literally named `all` is
`./all`).

### Input discovery

Without paths, `/review` looks for `plan.md` and `spec.md` (any case) in the
workspace root, then `docs/`, then `.hi/`, taking the first hit for each.

- Neither found, small workspace (200 non-ignored files or fewer): audits
  the whole tree against `README.md` with the notice `no plan.md or spec.md
  found; auditing against README.md`. The README's feature claims become
  the coverage rows; its numbered lists are not read as a checklist (install
  steps and usage tips are not plan items). No README either: `auditing for
  defects only`.
- **Neither found, large workspace: audits recent work.** One audit turn
  cannot cover a big tree and has nothing to compare it against (a live run
  on this repo read six of 1175 files against the README and reported
  nothing), so the audit narrows to what git says changed: the uncommitted
  files (modified, added, renamed, untracked; per `git status`), or, on a
  clean tree, the files the last commit touched. The README is dropped and
  the audit is **defects only**: the prompt lists the files, says how to see
  the diff (`git diff HEAD`, or `git show HEAD`), and says to report defects
  in those changes, not elsewhere. The status line shows the scope:

  ```
  spec review · audit only · 47 uncommitted files (git status) · no plan/spec (defects only)
  spec review · audit · up to 3 fix passes · last commit a1b2c3d (3 files) · no plan/spec (defects only)
  ```

  Paths are workspace-relative even when the workspace is a subdirectory of
  the repository, and only files still on disk are listed.
- **Nothing recent: refuses to start.** Not a git repository, or a clean
  tree whose last commit touched no file still present (an empty commit, a
  pure deletion). The message says why and what to pass: `/review audit
  all`, `/review audit <dir>`, or write `plan.md` as a `- [ ]` checklist.
- **`all`: the whole workspace in chunks.** Every non-ignored, non-hidden
  top-level directory is one audit turn; a directory with no files of its
  own (a container such as `crates/` or `packages/`) contributes its
  subdirectories instead, so a Cargo workspace is one turn per crate; the
  workspace's own top-level files are a last chunk. Directories with no
  non-ignored files are skipped. The start notice lists the chunks, the
  status line counts them (`chunk 3/7 · crates/hi-cli`), and each turn's
  prompt fences its directory ("report nothing outside it"). The chunk
  verdicts are merged into one: findings appended (duplicates by title and
  file dropped), a coverage item keeping the best state any chunk saw, and
  the fix loop, if any, runs once on the merged P0/P1 set; the re-audit is
  scoped by the fix pass's files as usual. Without a plan/spec `all` is
  defects only (the README is dropped); with one, each chunk turn also fills
  the coverage rows for what it holds. `all` with directories (`/review all
  src tests`) uses those directories as the chunks. Esc pauses between or
  during chunks; `/review` resumes with the chunk that was running.

A defects-only audit gets the verdict contract without a `coverage:` row;
`verdict: COMPLETE` then means "no P0/P1 found", the report shows
`coverage: none (no plan/spec items; write plan.md for a coverage audit)`,
and headless coverage counts as complete (exit 0 when no P0/P1 is open, not
3). With a real plan/spec and no rows, the model skipped its items: the
report says so and the run exits 3.

Checklist rows (`- [ ]`, `- [x]`, `1.`) in a plan or spec are counted and
handed to the model so the coverage table has one row per item. Checked and
numbered rows are claims to verify; unchecked rows are known gaps, reported
as `missing`/`partial` coverage and never as findings.

## What the loop does

```
/review ──> Audit ──(no P0/P1 or audit-only)──> Done
              │
              ├─(P0/P1)──> Fix ──(turn completed or policy error)──> Re-audit ──(clean)──> Done
              │             │                                          │
              │             └─(Esc)──> Paused                           ├─(new P0/P1, pass < cap)──> Fix
              │                                                        └─(same findings | cap)──> Stopped
              └─(no <review> block twice | turn error)──> Stopped
```

- **Audit turn.** Intent is pinned to `Review`, so a prompt that mentions
  "fix" never demands edits mid-audit, and the "you described a fix, apply
  it" nudge is off (an audit reply naturally says what should change). The
  model reads the plan/spec, starts from `repo_map`, reads only the files it
  cites, may run the test command, and ends with a tagged verdict block
  (below). Reasoning effort is raised to `high` for audit turns unless you
  pinned `/effort`.
- **Feature gaps are not defects.** A P0/P1 finding whose title restates an
  unchecked plan row (`- [ ] /topic …` vs `finding: P1 | Implement TOPIC …`)
  is flagged `feature_gap`: it stays in the report (marked "unchecked plan
  item: reported, not fixed") but never enters the fix loop or the
  `open_blocking` count. The match is deterministic (at least half of the
  row's content words, two or more, appear in the title after stemming),
  so a defect in a *checked* row — "KICK removes the target" claimed done
  but broken — is still fixed. Without a checklist nothing is flagged and
  the prompt rules alone keep missing features out of the findings.
- **Fix turn.** One plan step per P0/P1 finding (`[P0] title — path:line`),
  all pending, posted to the plan panel. The existing completion policy then
  demands edits while steps remain and `cargo test` (or your `/verify`
  command) after edits; the usual work credits stop bulk-closing steps
  without edits. `/undo` still restores the last fix turn's checkpoint.
  A fix turn that ends on a completion-policy error (`plan_stall` because
  the defect turned out to be fixed already, `unverified_stop`, a tool
  storm) still goes to the re-audit: the tree it left is what gets judged,
  and the status line says `fix turn ended early: …` until a verdict is in.
- **Re-audit turn.** Scoped to the files the fix pass changed plus the prior
  findings; same verdict contract.
- **Stops.** Clean verdict; audit-only; pass cap (default 3, `--spec-review-passes`
  headless); the same P0/P1 set surviving a fix pass (fingerprint on
  severity + normalized title, so a moved line does not count as progress);
  an audit or re-audit turn ending in a harness error; no parseable verdict
  after one format re-ask. `/review status` shows the stop reason. When the
  loop ends after a fix pass, its checklist is removed from the plan panel
  either way; the report lines list whatever is still open.
- **Pause.** Esc/Ctrl-C during any review turn sets `paused`; the drive is
  persisted in the session, so `/review` (or the ghost prompt) resumes the
  same phase, even after `hi --resume`.

Review prompts are injected with the `[hi:review]` prefix, so they never
become the session title or a later turn's inherited intent, and the
transcript echoes a short label (`spec review · fix pass 1/3`) instead of the
instruction block.

### Verdict contract

The model must end every audit with:

```text
<review>
verdict: COMPLETE | INCOMPLETE
coverage: implemented|partial|missing | <plan or spec item> | <path:line or ->
finding: P0|P1|P2|P3 | <imperative title> | <path:line>
residual: <one line>
</review>
```

**The coverage rows are the verdict.** The `verdict:` word is advisory: a
`missing` or `partial` row makes the result incomplete whatever the model
wrote, and rows that are all `implemented` make it complete (a live audit
wrote `INCOMPLETE` over eight implemented rows and `finding: none`, then
said in its residual that everything was implemented). When the word and the
rows disagree the report adds a `note:` line saying so; if a coverage row
failed to parse, the dropped row may be the gap, so the word stands. A block
with coverage rows but no `verdict:` row still parses.

Malformed rows are skipped and duplicate findings merged. Every path cited
in a `path:line` cell is checked against the workspace; a cell may list
several (`a.rs:10, b.rs:20-31; c.rs`) or add a parenthetical (`lib.rs
(CONST)`), each path is checked on its own, and citations that do not resolve
are kept but marked unverified (`unverified_citations` in the report lists
each missing path once). Rows echoed from the template itself (`verdict:
COMPLETE | INCOMPLETE`, `finding: P0|P1|P2|P3 | …`) are not answers and are
ignored, so a copied template cannot read as a clean verdict. Severities
follow the `code-review` skill: P0 release blocker, P1 urgent defect, P2
ordinary, P3 worth fixing. Only P0/P1 enter the fix loop.

In the transcript the block itself is hidden from audit and re-audit replies
(the model is told it is machine-read); the drive renders it once as the
report, which ends with a `next:` line: fix the open P0/P1 (or run `/review`
without `audit`), build the missing items, or write a `plan.md` when the
audit ran against a README or nothing. A clean result against a real
plan/spec has no next line.

A reply with no parseable block gets one format re-ask. When the plan/spec
items are known (checklist rows, or the previous verdict's coverage rows)
the re-ask is a form: one `coverage: <state> | <item> | <path:line or ->`
row per item with the item column already filled, plus the prior findings
to confirm or drop. The re-audit prompt lists the same items and says
"report; do not plan": a missing feature is a `missing` row, not a plan for
building it (a live run answered a re-audit with a "Build Next" plan for
the unimplemented item instead of the block).

## Headless: `hi --spec-review`

`--review` is the independent-review policy flag, so the headless entry is
`--spec-review`.

```bash
hi --spec-review --verify "cargo test --offline" --report out.json --no-save
hi --spec-review docs/spec.md plan.md crates/hi-tui --spec-review-passes 1
hi --spec-review --spec-review-audit-only --report out.json
hi --spec-review all --spec-review-audit-only --report out.json
```

| flag | meaning |
|---|---|
| `--spec-review [PATH…]` | run the loop headless with `StdoutUi`; conflicts with a prompt, `--goal`, and `--eval-input` |
| `--spec-review-passes N` | fix-pass cap (default 3) |
| `--spec-review-audit-only` | report only |

Values follow the slash command: plan/spec files, scope directories, and the
`all` keyword; with none on a large repo without a plan/spec, the run audits
the uncommitted changes or the last commit (a pre-commit or CI hook can run
`hi --spec-review --spec-review-audit-only` over exactly what changed).

The final summary, coverage matrix, and findings are printed to stdout;
status lines go to stderr. A nonzero exit prints the summary in place of the
usual `error:` line.

| exit | meaning |
|---|---|
| 0 | coverage complete and no P0/P1 findings |
| 1 | harness error before the loop settled (request failure, missing key) |
| 2 | usage error: an explicit plan/spec path does not exist, or a large workspace has no plan/spec, no scope, and nothing recent in git |
| 3 | loop finished; plan/spec items are partial or missing (or the model wrote no coverage rows for a real plan/spec) |
| 4 | loop finished; P0/P1 findings still open (audit-only, or open after the cap) |
| 5 | stopped early: no progress, pass cap, unparseable verdict, turn error |

Open P0/P1 outranks incomplete coverage (a run with both exits 4). Findings
flagged `feature_gap` (an unchecked plan row filed as a defect) do not count
as open P0/P1; the missing row already exits 3.

### Report object

`--report` adds a `review` object next to the usual turn report:

```json
"review": {
  "inputs": { "files": [{ "path": "plan.md", "kind": "plan" }], "scope": [], "notice": null, "git": null, "chunks": [] },
  "phase": "done",
  "audit_only": false,
  "passes": 1,
  "max_passes": 3,
  "verdict": "incomplete",
  "coverage_complete": false,
  "coverage": [{ "state": "missing", "item": "/topic command", "location": null }],
  "findings": [{ "severity": "P2", "title": "Log unknown commands", "location": "src/server.rs:163", "verified": true, "feature_gap": false }],
  "open_blocking": 0,
  "residual": "/topic is a missing feature, not a defect",
  "unverified_citations": [],
  "changed_files": ["src/server.rs"],
  "audit_changed_files": [],
  "stop_reason": null,
  "fix_turn_error": null,
  "summary": "spec review · done · plan.md + docs/spec.md",
  "exit_code": 3
}
```

`coverage` and `findings` are from the **last** verdict (the final re-audit
in a fixed run); `passes` and `changed_files` tell you whether a fix loop
ran. `audit_changed_files` is normally empty: audit turns are read-only and
write tools are denied in them; anything listed there got through a shell
redirection and the verdict was formed on a tree that moved. `fix_turn_error`
is the completion-policy error of the last fix turn when the run ended before
a re-audit verdict cleared it (otherwise `null`). `inputs.git` is the recent
work a large repo without a plan/spec narrowed to, `{ "source": { "kind":
"uncommitted" }, "files": [...] }` or `{ "source": { "kind": "last_commit",
"short_hash": "a1b2c3d", "subject": "..." }, "files": [...] }`, else `null`;
`inputs.chunks` lists an `all` run's chunk directories (`"."` is the
top-level files), else `[]`. In both cases `coverage` is `[]` and
`coverage_complete` is `true` unless a plan/spec was also given.

## Plain REPL

`hi --plain` accepts the same `/review …` commands and runs the fix ->
re-audit chain after each turn. There is no Esc pause there; the drive is
saved with the session, so if the process ends mid-loop, `/review` in the
resumed session picks up the same phase (a fix phase re-seeds its checklist).

## Limits

- `pipe/auto` has a 64k context and an 8k output cap. On large repos scope
  the audit (`/review crates/hi-tui docs/spec.md`), or let the default
  narrow it to recent work, or use `all`; the prompt tells the model to
  start with `repo_map` and read only cited files. A chunk is still one
  turn: a crate of several thousand lines is skimmed, not read.
- Recent-work scoping trusts git: files ignored or committed with a broken
  hook are not seen, and the last commit of a squash-merged branch is the
  whole branch. It does not diff the changes for the model; the model runs
  `git diff HEAD` / `git show HEAD` itself, so a huge diff is read in part.
- The re-audit is model-graded. The fingerprint and pass cap are the real
  loop guards; citation checking is deterministic.
- `/review implement` (build the missing items) and a `[review]` config
  section for default passes are follow-ups, not shipped.

## Verification

- Unit: `crates/hi-harness/src/review_tests.rs`, `review_drive_tests.rs`,
  `command.rs` (`/review` parsing), `hi-cli/src/review_cli_tests.rs`
  (flags, exit codes, report).
- Harness loop against `MockPipe`: `crates/hi-harness/src/review_harness_tests.rs`
  (audit verdict, format re-ask, fix turn seeds plan and requires tests,
  identical-findings stop, pass cap, cancel/resume across a reload, README
  fallback) and `review_harness_scope_tests.rs` (large repo without a spec:
  uncommitted files, then the last commit, then the refusal; `all` runs one
  turn per chunk and merges the verdicts).
- CLI one-shot against the scripted server: `crates/hi-cli/tests/spec_review_headless.rs`
  (exit 0/2/3/4/5 and the report object).
- Stream filter and workspace guard: `review_stream.rs` (block hidden from
  the transcript, split tags, truncated blocks shown) and `review_scope.rs`
  (file count stops at the limit, honours `.gitignore`; against temp git
  repos: status parsing with renames and untracked files, workspace-relative
  paths inside a larger repository, chunking with containers split and
  empty directories skipped, the refusal outside git and on an empty last
  commit). `review_drive_scope_tests.rs` covers chunk progression, verdict
  merging, the status line, and chunk state across a session reload.
- Live: `scripts/review_slash_live_e2e.sh` copies the buggy chat fixture,
  adds a `SPEC.md`/`plan.md` with an unimplemented `/topic` item, and expects
  the two bugs fixed, `cargo test --offline` green, `/topic` still missing,
  exit 3. Needs a Pipe credential; exit 2 means skipped.

  Last recorded run (2026-09-20, `pipe/deepseek-v4-flash-0731` via the
  `pipenetwork` provider): audit read `plan.md` + `SPEC.md`, filed the two
  bugs as P0/P1 and `/topic` as `missing`; fix pass 1 seeded a two-step plan,
  edited `src/main.rs`, ran `cargo test --offline` green; the re-audit needed
  one format re-ask, then reported 5/6 items implemented and no P0/P1;
  `tests/integration.rs` untouched; exit 3. Earlier runs of the same script
  drove the fixes described under "What the loop guards against" below.

### What the loop guards against

Live runs of the script surfaced these model behaviours; each has a
deterministic guard in the harness now:

- The model edits during the read-only audit after writing "let me fix
  that": the promised-work nudge in `turn.rs` is off for review-pinned audit
  turns.
- A fix turn stalls because the findings were already fixed and the model
  answers with a `<review>` block instead of `update_plan`: the policy error
  is carried into the re-audit (`fix_turn_error`) instead of stopping the
  loop.
- A re-audit answers with a "Missing / Build Next" plan instead of the
  block: the re-ask (and every re-audit) sends the block as a form with the
  item column pre-filled, and template echoes are not parsed as rows.
- The model files an unchecked plan item as a P1 and the fix pass builds it:
  findings that restate an unchecked row are `feature_gap`, reported and
  never seeded into the fix plan.
- The model calls `write` with identical content during a re-audit: the
  tool reports "(no changes)" and `changed_files` stays empty.
- A re-audit rewrote `src/main.rs` with the missing `/topic` feature, added
  a test, closed a plan, and then reported `/topic` as missing (its prompt
  already said "read-only" and "missing items are not work for this turn"):
  audit and re-audit turns now deny `write`, `edit`, `multi_edit`,
  `apply_patch` and `update_plan`, and any `bash` command the shell policy
  proves mutating (`sed -i`, `rm`, `git checkout`, …). The denial is the tool
  result, telling the model to file a `coverage:` or `finding:` row instead
  and end with the block; the third denial in one turn stops it
  (`audit_writes`, which stops the loop like any audit error). Test runs
  such as `cargo test … | tail` are not provably read-only and stay allowed.
  A change that still slips through a redirection is recorded in
  `audit_changed_files` and the report carries a `warning:` line naming the
  files (`/undo` restores that turn).

A `/review audit` on this repo (1175 files, README only) added these:

- `verdict: INCOMPLETE` over eight `implemented` rows and `finding: none`:
  the verdict is derived from the rows and the report notes the
  contradiction, so the summary and the exit code agree with the table.
- The reply was the raw block and nothing else, then the same rows again as
  the report: the block is hidden from streamed audit text and rendered once,
  and the prompt asks for a few lines of prose instead of a copy of the rows.
- Five of eight rows "unverified" because a cell held several paths or a
  parenthetical: each path in a cell is checked on its own.
- The README's numbered "how it works" list counted as three unchecked plan
  rows: a README is never parsed as a checklist, and numbered rows in a real
  plan are claims, not gaps.
- Six of 1175 files read, 106k tokens, nothing found, no next step: over
  200 files with no plan/spec and no scope the command now audits the
  uncommitted changes (else the last commit) for defects, `all` walks the
  repo in chunks, and it refuses only when git has nothing recent; every
  finished report ends with a `next:` line.

### Manual TUI checklist

Run in this repo with a small scoped spec, e.g. `/review docs/review-command.md crates/hi-harness/src/review.rs`:

1. `/review` prints `spec review · audit · up to 3 fix passes · <inputs>` and
   the transcript echoes `spec review · audit`, not the prompt block.
2. `/review status` while the audit runs answers from the cached status line.
3. If P0/P1s are found, the plan panel fills with `[P0] …` / `[P1] …` steps
   and the status flips to `fix pass 1/3`; the turn is asked for tests after
   edits.
4. Esc mid-turn: `spec review paused during <phase>`, the input shows a ghost
   `/review`; empty Enter resumes the same phase.
5. `/review stop` ends it and clears the checklist; `/review stop` again says
   nothing to stop.
6. `/undo` after a fix pass restores the files from that turn's checkpoint.
7. Quit, `hi --resume`: a paused review suggests `/review` on startup and
   `/review status` still shows the phase.
