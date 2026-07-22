// port-feature — plan, implement in a worktree, verify, then merge.
let meta = #{
    name: "port-feature",
    description: "Port or implement a feature: plan the work, implement it in an isolated worktree, verify, then prepare for merge.",
    when_to_use: "When you need to implement a non-trivial feature that should be isolated, verified, and merged — the workflow creates a worktree, implements, tests, and reports the diff.",
    phases: [
        #{ title: "Plan", detail: "decompose the feature into steps" },
        #{ title: "Implement", detail: "worktree-isolated implementation" },
        #{ title: "Verify", detail: "run tests in the worktree" },
        #{ title: "Report", detail: "summarize the diff for merge" },
    ],
};

let objective = args.objective;
let test_cmd = args.test_command;

phase("Plan");
let plan_result = agent("Plan the implementation of: " + objective + ". Break it into ordered steps. Return a JSON array of step descriptions.");
let steps = if type_of(plan_result.output) == "array" {
    plan_result.output
} else {
    // Fallback: treat the output as a single-step plan.
    [json_encode(plan_result.output)]
};

phase("Implement");
let impl_result = agent("Implement this feature in the worktree: " + objective + ". Follow these steps:\n" + json_encode(steps) + ". Make all necessary code changes. Do not run tests yet.", #{
    isolation_worktree: true,
    phase: "Implement",
});

phase("Verify");
let verify_result = agent("Run the test suite (" + test_cmd + ") in the worktree and report the result. If tests fail, fix the issues and re-run.", #{
    isolation_worktree: true,
    phase: "Verify",
});
if !verify_result.success {
    pause("verification", "Tests failed");
}

phase("Report");
let diff = git_diff_since(args.baseline_commit);
let report = agent("Summarize the changes made for: " + objective + ". The diff is:\n" + diff + "\n\nWrite a concise merge-ready summary: what changed, why, and any risks.");
complete(report.output);
