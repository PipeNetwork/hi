//! Prompts the `/review` drive injects (audit, fix pass, re-audit, format
//! re-ask) and the verdict-block contract they all quote.
//!
//! Split from [`crate::review`] (inputs, checklist, verdict parsing) so both
//! stay under the file-size ratchet; `review` re-exports every public item
//! here, so callers keep using `crate::review::audit_prompt` and friends.

use crate::review::{
    ChecklistItem, Finding, GitSource, InputKind, REVIEW_PREFIX, ReviewInputs, chunk_label,
};

/// The verdict contract, quoted in every audit prompt and the format re-ask.
pub const REVIEW_FORMAT_BLOCK: &str = "<review>\n\
verdict: COMPLETE | INCOMPLETE\n\
coverage: implemented|partial|missing | <plan or spec item> | <path:line or ->\n\
finding: P0|P1|P2|P3 | <imperative title> | <path:line>\n\
residual: <one line>\n\
</review>";

/// The contract for a defects-only audit (no plan/spec, so no coverage
/// rows): the verdict word stands in for "no P0/P1 found".
pub const REVIEW_DEFECTS_FORMAT_BLOCK: &str = "<review>\n\
verdict: COMPLETE | INCOMPLETE\n\
finding: P0|P1|P2|P3 | <imperative title> | <path:line>\n\
residual: <one line>\n\
</review>";

/// Scope files named in a prompt before `… and N more`.
const SCOPE_FILES_LISTED: usize = 40;

const REVIEW_REASK_HEAD: &str = "[hi:review] Format re-ask. Your last reply had no parseable <review> block, so no verdict was recorded. \
Do not repeat that reply and do not write a plan or next steps for missing features. \
Reply with the block below and nothing else, one row per line, `|` separated: ";

const REVIEW_REASK_TAIL: &str = "one `finding:` row per defect still present (or `finding: none`), \
and `verdict: COMPLETE` only when every coverage row is `implemented`.\n";

const REVIEW_REASK_DEFECTS_TAIL: &str = "no `coverage:` rows (there are no plan/spec items), \
one `finding:` row per defect still present (or `finding: none`), \
and `verdict: COMPLETE` only when no P0/P1 finding is listed.\n";

/// Re-ask sent once when an audit reply has no parseable block and no
/// plan/spec items are known; see [`format_reask_prompt`].
pub const REVIEW_FORMAT_HINT: &str = "[hi:review] Format re-ask. Your last reply had no parseable <review> block, so no verdict was recorded. \
Do not repeat that reply and do not write a plan or next steps for missing features. \
Reply with the block below and nothing else, one row per line, `|` separated: \
one `coverage:` row per plan/spec item, \
one `finding:` row per defect still present (or `finding: none`), \
and `verdict: COMPLETE` only when every coverage row is `implemented`.\n\
<review>\n\
verdict: COMPLETE | INCOMPLETE\n\
coverage: implemented|partial|missing | <plan or spec item> | <path:line or ->\n\
finding: P0|P1|P2|P3 | <imperative title> | <path:line>\n\
residual: <one line>\n\
</review>";

/// The format re-ask, as a form when the plan/spec `items` are known (from
/// the checklist or the last verdict): one pre-filled `coverage:` row per
/// item with only the state and citation left to fill. A small model copies
/// a skeleton far more reliably than it applies a grammar, and an item
/// column that is already there leaves nothing to plan. `prior` findings
/// are listed so "still present" has something to refer to. With no items
/// and no prior findings this is [`REVIEW_FORMAT_HINT`]; a `defects_only`
/// audit gets the block without a coverage row.
pub fn format_reask_prompt(items: &[String], prior: &[Finding], defects_only: bool) -> String {
    if items.is_empty() && prior.is_empty() && !defects_only {
        return REVIEW_FORMAT_HINT.to_string();
    }
    let mut out = String::from(REVIEW_REASK_HEAD);
    if defects_only {
        out.push_str(REVIEW_REASK_DEFECTS_TAIL);
    } else {
        if items.is_empty() {
            out.push_str("one `coverage:` row per plan/spec item, ");
        } else {
            out.push_str(
                "fill in `<state>` (implemented, partial, or missing) and `<path:line or ->` on every row already listed, ",
            );
        }
        out.push_str(REVIEW_REASK_TAIL);
    }
    if !prior.is_empty() {
        out.push_str("Prior findings, each listed again only if it is still present:\n");
        for finding in prior {
            out.push_str(&format!("- {}\n", finding.label()));
        }
    }
    out.push_str(&skeleton_block(items, defects_only));
    out
}

/// The verdict block as a form: one `coverage:` row per known item with
/// only the state and citation left to fill. With no items this is
/// [`REVIEW_FORMAT_BLOCK`], or [`REVIEW_DEFECTS_FORMAT_BLOCK`] when there
/// is no plan/spec to cover.
fn skeleton_block(items: &[String], defects_only: bool) -> String {
    if defects_only {
        return REVIEW_DEFECTS_FORMAT_BLOCK.to_string();
    }
    if items.is_empty() {
        return REVIEW_FORMAT_BLOCK.to_string();
    }
    let mut out = String::from("<review>\nverdict: COMPLETE | INCOMPLETE\n");
    for item in items {
        out.push_str(&format!("coverage: <state> | {item} | <path:line or ->\n"));
    }
    out.push_str(
        "finding: P0|P1|P2|P3 | <imperative title> | <path:line>\nresidual: <one line>\n</review>",
    );
    out
}

fn push_items(out: &mut String, items: &[String]) {
    if items.is_empty() {
        return;
    }
    out.push_str("\nPlan/spec items (one `coverage:` row each, in this order):\n");
    for item in items {
        out.push_str(&format!("- {item}\n"));
    }
}

/// Inputs, then what code the audit covers: an explicit scope, the git
/// scope, or (in an audit turn) the current chunk. A re-audit passes
/// `chunk: None`: it is scoped by the fix pass's files instead.
fn push_inputs(
    out: &mut String,
    inputs: &ReviewInputs,
    checklists: &[(String, Vec<ChecklistItem>)],
    chunk: Option<usize>,
) {
    if inputs.files.is_empty() {
        out.push_str(
            "Inputs: none found (no plan.md or spec.md). This is a defects-only audit: there are no plan/spec items, \
so write no `coverage:` rows.\n",
        );
    } else {
        out.push_str("Inputs (read each in full first):\n");
        for input in &inputs.files {
            let rows = checklists
                .iter()
                .find(|(path, _)| path == &input.path)
                .map(|(_, items)| items)
                .filter(|items| !items.is_empty());
            out.push_str(&format!("- {}: {}", input.kind.label(), input.path));
            if input.kind == InputKind::Readme {
                out.push_str(
                    " (fallback: no plan.md or spec.md; the features this README claims are the plan/spec items, \
install steps and usage tips are not)",
                );
            } else if let Some(items) = rows {
                out.push_str(&format!(" ({})", describe_rows(items)));
            }
            out.push('\n');
        }
    }
    if !inputs.scope.is_empty() {
        out.push_str(&format!(
            "Scope: limit the code audit to {} (still read the inputs above in full).\n",
            inputs.scope.join(", ")
        ));
    }
    if let Some(git) = &inputs.git {
        push_git_scope(out, git);
    }
    if let Some(index) = chunk
        && let Some(dir) = inputs.chunks.get(index)
    {
        let where_ = if dir == "." {
            "the files directly in the workspace root (not its subdirectories)".to_string()
        } else {
            format!("`{dir}/`")
        };
        out.push_str(&format!(
            "Scope: chunk {}/{} of the workspace: {where_}. Limit the code audit to it; the other chunks are audited \
in their own turns, so report nothing outside it.",
            index + 1,
            inputs.chunks.len()
        ));
        if !inputs.files.is_empty() {
            out.push_str(
                " Coverage rows judge this chunk only: an item this chunk holds none of is `missing` here with `-` \
as its citation; rows are merged across chunks (an item implemented in any chunk counts as implemented).",
            );
        }
        out.push('\n');
    }
}

/// The recent-work scope: how it was chosen, how to see the changes, and
/// the files (capped; git has the rest).
fn push_git_scope(out: &mut String, git: &crate::review::GitScope) {
    let count = git.files.len();
    match &git.source {
        GitSource::Uncommitted => out.push_str(&format!(
            "\nCode under audit: the {count} uncommitted file(s) per `git status` (no plan/spec was found, so this audit \
covers recent work, not the whole repo). Run `git status --short` and `git diff HEAD` to see the changes; \
untracked files have no diff, read them in full. Read surrounding code only as needed to judge the changes, \
and report defects in these changes, not elsewhere.\n"
        )),
        GitSource::LastCommit { short_hash, subject } => out.push_str(&format!(
            "\nCode under audit: the {count} file(s) changed by the last commit {short_hash} \"{subject}\"; the tree is clean \
(no plan/spec was found, so this audit covers recent work, not the whole repo). Run `git show HEAD` to see the change. \
Read surrounding code only as needed to judge it, and report defects in this change, not elsewhere.\n"
        )),
    }
    out.push_str("Files:\n");
    for file in git.files.iter().take(SCOPE_FILES_LISTED) {
        out.push_str(&format!("- {file}\n"));
    }
    if count > SCOPE_FILES_LISTED {
        out.push_str(&format!(
            "- … and {} more (git lists them)\n",
            count - SCOPE_FILES_LISTED
        ));
    }
}

/// `3 checklist rows: 1 checked, claims to verify; 2 unchecked, known gaps…`.
/// Numbered rows carry no checkbox, so they are claims like checked ones,
/// not gaps (a README's numbered "how it works" list once read as three
/// unchecked gaps).
fn describe_rows(items: &[ChecklistItem]) -> String {
    let checked = items
        .iter()
        .filter(|item| item.checked == Some(true))
        .count();
    let unchecked = items
        .iter()
        .filter(|item| item.checked == Some(false))
        .count();
    let numbered = items.len() - checked - unchecked;
    let mut parts = Vec::new();
    if checked > 0 {
        parts.push(format!("{checked} checked, claims to verify"));
    }
    if unchecked > 0 {
        parts.push(format!(
            "{unchecked} unchecked, known gaps: report each as a `missing`/`partial` coverage row, never as a `finding:` row"
        ));
    }
    if numbered > 0 {
        parts.push(format!("{numbered} numbered, claims to verify"));
    }
    format!("{} checklist rows: {}", items.len(), parts.join("; "))
}

/// The verdict contract. With `items` the block is pre-filled (the
/// re-audit knows its rows from the last verdict); a live model answered
/// the generic contract with a "Build Next" plan and the form with the
/// block. A `defects_only` audit gets the block without a coverage row and
/// a verdict word that means "no P0/P1".
fn push_format_contract(out: &mut String, items: &[String], defects_only: bool) {
    if items.is_empty() || defects_only {
        out.push_str(
            "\nEnd your reply with exactly one block in this format (one row per line, `|` separated):\n",
        );
    } else {
        out.push_str(
            "\nEnd your reply with exactly this block, filled in (one row per line, `|` separated; \
replace `<state>` with implemented, partial, or missing and `<path:line or ->` with the citation or `-`; keep every row):\n",
        );
    }
    out.push_str(&skeleton_block(items, defects_only));
    if defects_only {
        out.push_str(
            "\nRules: there are no plan/spec items, so write no `coverage:` rows; `verdict: COMPLETE` when no P0/P1 finding is listed, \
`INCOMPLETE` otherwise; `finding:` rows are defects in the code (not missing features or style), highest severity first, or `finding: none`; \
P0 = release blocker or data loss, P1 = urgent defect, P2 = ordinary defect, P3 = minor. \
Cite `path:line` only for files you actually read.\n",
        );
        return;
    }
    out.push_str(
        "\nRules: one `coverage:` row per plan/spec item; `verdict: COMPLETE` only when every row is `implemented`; \
a `missing` or `partial` row is the whole report for an unimplemented item (do not write a plan or next steps for it); \
`finding:` rows are defects in the code (not missing features), highest severity first, or `finding: none`; \
P0 = release blocker or data loss, P1 = urgent defect, P2 = ordinary defect, P3 = minor. \
Cite `path:line` only for files you actually read.\n",
    );
}

/// Audit turn. Read-only; the harness forces `Intent::Review` because this
/// text mentions fixing. `chunk` indexes `inputs.chunks` in a chunked audit
/// (ignored otherwise); each chunk turn carries the whole contract, since
/// the model has no memory of the previous chunk's turn.
pub fn audit_prompt(
    inputs: &ReviewInputs,
    checklists: &[(String, Vec<ChecklistItem>)],
    chunk: usize,
    max_passes: u32,
) -> String {
    let defects_only = inputs.defects_only();
    let mut out = format!("{REVIEW_PREFIX} ");
    if defects_only {
        out.push_str("Defect audit");
    } else {
        out.push_str("Spec-coverage audit");
    }
    if let Some(dir) = inputs.chunks.get(chunk) {
        out.push_str(&format!(
            ", chunk {}/{}: {}",
            chunk + 1,
            inputs.chunks.len(),
            chunk_label(dir)
        ));
    }
    out.push_str(&format!(
        " (up to {max_passes} fix passes follow if you report P0/P1 defects). \
This turn is read-only: do not edit files, do not run mutating shell commands, do not call update_plan.\n\n"
    ));
    push_inputs(&mut out, inputs, checklists, Some(chunk));
    out.push_str("\nProcedure:\n");
    if defects_only {
        out.push_str(
            "1. Call repo_map (or `git status`/`git diff` for a recent-work scope), then read the code under audit and only \
the surrounding files needed to judge it.\n\
2. Look for P0/P1 defects: crashes, data loss, wrong results, security holes, broken contracts between modules. \
Running the project's existing tests read-only (for example `cargo test`) is allowed when it grounds a finding.\n\
3. Do not fix anything in this turn; report (write, edit, multi_edit, apply_patch and update_plan are denied in an audit turn). \
Before the block, write a few lines for the user: what you read, and for each finding why it is a defect. \
The block itself is machine-read and shown as a table, so do not repeat its rows as prose.\n",
        );
    } else {
        out.push_str(
            "1. Read every input file above in full.\n\
2. Call repo_map, then read only the source files needed to ground each plan/spec item and each finding.\n\
3. For each plan/spec item decide implemented, partial, or missing, and cite the code that implements it.\n\
4. Look for P0/P1 defects in what is implemented: crashes, data loss, wrong results, security holes, spec violations. \
Running the project's existing tests read-only (for example `cargo test`) is allowed when it grounds a finding.\n\
5. Do not fix anything in this turn; report (write, edit, multi_edit, apply_patch and update_plan are denied in an audit turn). \
Before the block, write a few lines for the user: what you read, and for each finding why it is a defect. \
The block itself is machine-read and shown as a table, so do not repeat its rows as prose.\n",
        );
    }
    push_format_contract(&mut out, &[], defects_only);
    out
}

/// Fix turn for the blocking findings of the last audit. The harness seeds
/// one plan step per finding and forces `Intent::Fix`.
pub fn fix_prompt(findings: &[Finding], pass: u32, max_passes: u32) -> String {
    let mut out = format!(
        "{REVIEW_PREFIX} Fix pass {pass}/{max_passes}. Fix these P0/P1 defects from the spec review. \
The checklist below is already posted as the plan: work through it in order, mark each step done with update_plan as you fix it, \
and run the project's tests after the last edit.\n\n"
    );
    for (index, finding) in findings.iter().enumerate() {
        out.push_str(&format!("{}. {}\n", index + 1, finding.label()));
    }
    out.push_str(
        "\nRules: keep existing tests (do not weaken or delete them); add a regression test when a fix is testable; \
only fix the listed defects, do not build missing plan/spec features in this pass; \
if a finding is wrong or already fixed when you inspect it, say so in one line and do not make cosmetic edits \
(the re-audit confirms it); this pass ends with a short summary, not a <review> block.\n",
    );
    out
}

/// Re-audit after a fix pass, scoped to the changed files plus the prior
/// findings. Same verdict contract as the first audit; `items` are the
/// plan/spec items the coverage rows must cover (from the last verdict, or
/// the checklist).
pub fn reaudit_prompt(
    inputs: &ReviewInputs,
    items: &[String],
    changed_files: &[String],
    prior: &[Finding],
    pass: u32,
    max_passes: u32,
) -> String {
    let mut out = format!(
        "{REVIEW_PREFIX} Re-audit after fix pass {pass}/{max_passes}. This turn is read-only: write tools and update_plan are denied. \
Report; do not plan. Items still missing from the plan/spec stay `missing` coverage rows, they are not work for this turn.\n\n"
    );
    push_inputs(&mut out, inputs, &[], None);
    push_items(&mut out, items);
    if changed_files.is_empty() {
        out.push_str("\nThe fix pass changed no files.\n");
    } else {
        out.push_str("\nFiles changed by the fix pass (read each in full):\n");
        for file in changed_files {
            out.push_str(&format!("- {file}\n"));
        }
    }
    if !prior.is_empty() {
        out.push_str(
            "\nPrior findings to re-check (list a finding again only if it is still present):\n",
        );
        for finding in prior {
            out.push_str(&format!("- {}\n", finding.label()));
        }
    }
    if inputs.defects_only() {
        out.push_str(
            "\nProcedure: confirm each prior finding is resolved or still present, check the changed files for new defects, \
and run the project's tests read-only if useful.\n",
        );
    } else {
        out.push_str(
            "\nProcedure: confirm each prior finding is resolved or still present, check the changed files for new defects, \
run the project's tests read-only if useful, then refresh the coverage rows for every plan/spec item.\n",
        );
    }
    push_format_contract(&mut out, items, inputs.defects_only());
    out
}

/// One-line label for the transcript instead of echoing a long injected prompt.
pub fn transcript_label(prompt: &str) -> Option<String> {
    let rest = prompt.trim_start().strip_prefix(REVIEW_PREFIX)?;
    let first = rest.trim().lines().next().unwrap_or("").trim();
    let head = first
        .split_once(['.', '('])
        .map(|(h, _)| h)
        .unwrap_or(first);
    Some(format!("/review · {}", head.trim().to_ascii_lowercase()))
}

#[cfg(test)]
#[path = "review_prompts_tests.rs"]
mod tests;
