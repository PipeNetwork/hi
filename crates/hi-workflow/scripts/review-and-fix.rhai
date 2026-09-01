// review-and-fix — parallel skeptical review, triage, serial fix, then verify.
let meta = #{
    name: "review-and-fix",
    description: "Review changes with parallel skeptics, triage the findings, fix issues serially, then verify the result.",
    when_to_use: "When you have a diff or changeset that needs adversarial review before merging — multiple skeptics find issues, then fixes are applied and verified.",
    phases: [
        #{ title: "Review", detail: "parallel skeptical review of the diff" },
        #{ title: "Triage", detail: "deduplicate and prioritize findings" },
        #{ title: "Fix", detail: "apply fixes serially" },
        #{ title: "Verify", detail: "run the test suite to confirm" },
    ],
};

let diff = args.diff_summary;
let test_cmd = args.test_command;

phase("Review");
let reviewers = [
    #{ prompt: "Review this diff for correctness bugs — logic errors, off-by-one, null derefs, race conditions. List each issue with file and line. Diff:\n" + diff, label: "correctness", phase: "Review" },
    #{ prompt: "Review this diff for security issues — injection, unsafe input handling, missing validation. List each issue with file and line. Diff:\n" + diff, label: "security", phase: "Review" },
    #{ prompt: "Review this diff for performance regressions — unnecessary allocations, O(n²) loops, missing indexes. List each issue with file and line. Diff:\n" + diff, label: "performance", phase: "Review" },
];
let reviews = parallel(reviewers);

phase("Triage");
let all_findings = [];
for r in reviews {
    let output_text = if type_of(r.output) == "map" {
        json_encode(r.output)
    } else {
        r.output
    };
    all_findings.push(output_text);
}
// Manual join with separator (Rhai arrays don't have .join()).
let combined_findings = "";
let first = true;
for f in all_findings {
    if !first {
        combined_findings += "\n---\n";
    }
    combined_findings += f;
    first = false;
}
let triage = agent("Triage these review findings. Deduplicate, rank by severity (critical > major > minor), and produce a JSON array of fix items with {file, issue, severity, fix_description}.\n\nFindings:\n" + combined_findings);
let triage_text = if type_of(triage.output) == "map" {
    triage.output.summary
} else {
    json_encode(triage.output)
};
let fixes = if type_of(triage.output) == "array" {
    triage.output
} else {
    // Fallback: no fixes to apply during validation.
    []
};

phase("Fix");
for fix in fixes {
    agent("Fix this issue in " + fix.file + ": " + fix.issue + ". Approach: " + fix.fix_description + ". Make the minimal change needed.");
}

phase("Verify");
let verify_result = agent("Run the test suite (" + test_cmd + ") and report whether it passes. If it fails, describe the failure.");
if !verify_result.success {
    pause("verification", "Tests failed after fixes");
}
complete(verify_result.output);
