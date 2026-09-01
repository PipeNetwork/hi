---
name: code-review
description: Defect-first read-only review of a diff, PR, or named area. Findings first; do not edit.
scope: global
---

# Code review

## When to use
`/review <topic>`, `/security`, `/gaps`, `/roadmap`, `/status`, a bare "review the codebase", `/loop review` (PR comments), or a post-verify completion/skeptic/trio gate. Not for implementation turns.

## Procedure
1. Resolve the target: working-tree diff for an open review; `git merge-base` then `git diff <merge-base>` for a branch; `gh pr diff <n>` for a PR. Do not review a branch tip against itself.
2. Read the changed paths plus call sites and tests that the diff actually affects.
3. Do not write, edit, patch, or run mutating shell. `/loop review` may post a `gh pr review --comment` only — never approve or request-changes.
4. Keep inspecting until findings are grounded, or say the evidence is insufficient.

## Findings
Lead with findings, highest severity first. One issue per entry:

`[P0] Imperative title — path/to/file.rs:line`

Then one short paragraph: the scenario, why it is wrong, and that the reviewed change introduced it. Cite the smallest overlapping range.

- `P0` release blocker. `P1` urgent defect. `P2` ordinary defect. `P3` still worth fixing.
- Flag only correctness, security, performance, or maintainability that is discrete, introduced by this change, and that the author would likely fix.
- Do not flag style nits, pre-existing issues, intentional behavior, or speculation.
- If none qualify, write exactly `No findings.` then a brief residual-risk note.

## Gate
Chat-only reviewers (completion review, large-diff, goal skeptic, trio) have no tools. OBJECT only for a concrete correctness, security, compatibility, migration, or acceptance defect **introduced by this change**, with the affected file or behavior named. Missing tests OBJECT only when the contract or sub-goal demands them. Do not OBJECT over style, naming, or approach. When uncertain, APPROVE. ESCALATE only when retry cannot fix it (contradicts the objective, or needs a user decision). The bar does not rise on re-review.

## Pitfalls
- The rust/pytest/ts stack pack is for coding loops. Do not `cargo test -p` or rewrite files on a review turn.
- A listing is not evidence. A guessed URL is not a PR diff.
