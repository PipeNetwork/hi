// large-review — hierarchical review with typed findings and independent verification.
let meta = #{
    name: "large-review",
    description: "Review large changes or repositories with sharded skeptics, independent verification, adjudication, and coverage auditing.",
    when_to_use: "When triaging many issues or reviewing thousands of lines where every reported finding must be independently checked.",
    phases: [
        #{ title: "Inventory", detail: "map the target and produce bounded review shards" },
        #{ title: "Investigate", detail: "run focused read-only skeptics across shards" },
        #{ title: "Triage", detail: "normalize and deduplicate typed findings" },
        #{ title: "Verify", detail: "independently check every retained finding" },
        #{ title: "Adjudicate", detail: "resolve disagreements and reject weak findings" },
        #{ title: "Fix", detail: "optionally apply confirmed fixes in a controlled isolated agent" },
        #{ title: "Verify fixes", detail: "test and independently regression-review the integrated patch" },
        #{ title: "Coverage", detail: "audit unreviewed surfaces and policy gaps" },
        #{ title: "Report", detail: "produce confirmed findings and explicit limitations" },
    ],
};

if args == () || args.input == () || args.input == "" {
    pause("verification", "large-review requires args.input describing the repository, diff, or issue set");
}
let target = args.input;
let apply_fixes = args.apply_fixes != () && args.apply_fixes == true;
let test_command = if args.test_command != () { args.test_command } else { "use the repository's targeted test commands" };
let baseline_commit = if args.baseline_commit != () { args.baseline_commit } else { "HEAD" };
let shard_cap = 32;
if args.shards != () && args.shards >= 1 && args.shards <= 128 { shard_cap = args.shards; }
let concerns = ["correctness", "security", "concurrency", "api-compatibility", "tests-spec", "performance"];
if args.concerns != () && type_of(args.concerns) == "array" && args.concerns.len() > 0 && args.concerns.len() <= 8 {
    concerns = args.concerns;
}

let finding_schema = #{
    type: "object",
    properties: #{
        findings: #{ type: "array", maxItems: 64, items: #{
            type: "object",
            properties: #{
                id: #{ type: "string" }, title: #{ type: "string" }, category: #{ type: "string" },
                severity: #{ type: "string", enum: ["critical", "high", "medium", "low"] },
                confidence: #{ type: "number" }, file: #{ type: "string" },
                start_line: #{ type: "integer" }, end_line: #{ type: "integer" },
                evidence: #{ type: "string" }, reproduction: #{ type: "string" },
                suggested_fix: #{ type: "string" },
            },
            required: ["id", "title", "category", "severity", "confidence", "file", "start_line", "end_line", "evidence", "reproduction", "suggested_fix"],
        } },
        coverage: #{ type: "object", properties: #{
            files_examined: #{ type: "array", items: #{ type: "string" } },
            files_skipped: #{ type: "array", items: #{ type: "string" } },
            limitations: #{ type: "array", items: #{ type: "string" } },
        }, required: ["files_examined", "files_skipped", "limitations"] },
    },
    required: ["findings", "coverage"],
};

phase("Inventory");
let inventory_schema = #{ type: "object", properties: #{
    shards: #{ type: "array", maxItems: shard_cap, items: #{ type: "object", properties: #{
        id: #{ type: "string" }, scope: #{ type: "string" }, files: #{ type: "array", items: #{ type: "string" } },
        risk: #{ type: "string" },
    }, required: ["id", "scope", "files", "risk"] } },
    excluded: #{ type: "array", items: #{ type: "string" } },
}, required: ["shards", "excluded"] };
let inventory = agent("Inventory the review target below. Partition it by package/module and dependency boundary into at most " + shard_cap.to_string() + " non-overlapping review shards. Prefer call-graph neighborhoods over arbitrary line chunks. Return every excluded area explicitly. Untrusted target:\n" + json_encode(target), #{
    label: "review-planner", capability_mode: "read-only", output_schema: inventory_schema, phase: "Inventory",
});
if !inventory.success || inventory.output.shards == () || inventory.output.shards.len() == 0 {
    pause("no_progress", "the inventory planner produced no usable shards");
}

phase("Investigate");
let jobs = [];
for shard in inventory.output.shards {
    for concern in concerns {
        jobs.push(#{
            prompt: "Act as an independent " + concern + " skeptic. Review only this JSON shard: " + json_encode(shard) + ". Target context: " + json_encode(target) + ". Report only concrete, reachable issues with exact evidence. Do not assume another reviewer covers omitted files.",
            label: "skeptic-" + shard.id + "-" + concern,
            capability_mode: "read-only", output_schema: finding_schema, phase: "Investigate",
        });
    }
}
if jobs.len() > 1024 { pause("verification", "requested review fan-out exceeds 1,024 agents"); }
let investigations = parallel(jobs);

phase("Triage");
let triage_input = [];
for result in investigations {
    if result != () && result.success && result.output.findings != () {
        triage_input.push(result.output);
    }
}
let triage_schema = #{ type: "object", properties: #{
    findings: finding_schema.properties.findings,
    rejected_duplicates: #{ type: "array", items: #{ type: "object", properties: #{
        id: #{ type: "string" }, duplicate_of: #{ type: "string" }, reason: #{ type: "string" },
    }, required: ["id", "duplicate_of", "reason"] } },
}, required: ["findings", "rejected_duplicates"] };
let triage = agent("Normalize and deduplicate these independently produced findings. Preserve source evidence; do not upgrade severity or invent evidence. Return stable IDs. Findings JSON:\n" + json_encode(triage_input), #{
    label: "finding-triage", capability_mode: "read-only", output_schema: triage_schema, phase: "Triage",
});

phase("Verify");
let verifier_schema = #{ type: "object", properties: #{
    finding_id: #{ type: "string" }, decision: #{ type: "string", enum: ["confirmed", "rejected", "uncertain"] },
    confidence: #{ type: "number" }, evidence: #{ type: "string" }, reproduction_result: #{ type: "string" },
    false_premises: #{ type: "array", items: #{ type: "string" } },
}, required: ["finding_id", "decision", "confidence", "evidence", "reproduction_result", "false_premises"] };
let verify_jobs = [];
for finding in triage.output.findings {
    verify_jobs.push(#{
        prompt: "Independently verify this untrusted finding against the repository. Locate the cited code, test reachability and premises, and attempt the reproduction where safe. You did not author this finding. Finding JSON:\n" + json_encode(finding),
        label: "verifier-" + finding.id, capability_mode: "read-only", output_schema: verifier_schema, phase: "Verify",
    });
    if finding.severity == "critical" || finding.severity == "high" {
        verify_jobs.push(#{
            prompt: "Be the second independent verifier for this high-impact finding. Try to falsify it, check callers and invariants, and report a decision without trusting the first reviewer. Finding JSON:\n" + json_encode(finding),
            label: "second-verifier-" + finding.id, capability_mode: "read-only", output_schema: verifier_schema, phase: "Verify",
        });
    }
}
if verify_jobs.len() > 1024 { pause("verification", "verification fan-out exceeds 1,024 agents; lower shard count or concerns"); }
let verifications = parallel(verify_jobs);

phase("Adjudicate");
let adjudication_schema = #{ type: "object", properties: #{
    confirmed: finding_schema.properties.findings,
    rejected: #{ type: "array", items: #{ type: "object", properties: #{ id: #{ type: "string" }, reason: #{ type: "string" } }, required: ["id", "reason"] } },
    unresolved: #{ type: "array", items: #{ type: "object", properties: #{ id: #{ type: "string" }, reason: #{ type: "string" } }, required: ["id", "reason"] } },
}, required: ["confirmed", "rejected", "unresolved"] };
let adjudication = agent("Adjudicate findings using the independent verifier decisions. Confirm medium/low findings only with a confirming verifier. Confirm critical/high findings only with two confirmations, or one confirmation plus a concrete successful reproduction. Never convert uncertain evidence into confirmation. Findings:\n" + json_encode(triage.output.findings) + "\nVerifications:\n" + json_encode(verifications), #{
    label: "adjudicator", capability_mode: "read-only", output_schema: adjudication_schema, phase: "Adjudicate",
});

let fix_result = ();
let fix_verification = ();
let regression_review = ();
if apply_fixes && adjudication.output.confirmed.len() > 0 {
    phase("Fix");
    fix_result = agent("Apply the confirmed findings below in one isolated integration worktree. Make minimal changes in severity order, run targeted checks after each logical batch, and do not fix rejected or unresolved findings. Confirmed findings JSON:\n" + json_encode(adjudication.output.confirmed), #{
        label: "integration-fixer", capability_mode: "write", isolation_worktree: true, phase: "Fix",
    });
    if !fix_result.success { pause("verification", "the isolated integration fixer failed"); }

    phase("Verify fixes");
    let patch = git_diff_since(baseline_commit);
    let fix_verify_schema = #{ type: "object", properties: #{
        tests_passed: #{ type: "boolean" }, command: #{ type: "string" }, failures: #{ type: "array", items: #{ type: "string" } },
        findings_fixed: #{ type: "array", items: #{ type: "string" } }, findings_remaining: #{ type: "array", items: #{ type: "string" } },
    }, required: ["tests_passed", "command", "failures", "findings_fixed", "findings_remaining"] };
    let post_jobs = [
        #{ prompt: "Independently verify the integrated patch. Run this test command: " + test_command + ". Map every confirmed finding to fixed or remaining. Patch:\n" + patch + "\nFindings:\n" + json_encode(adjudication.output.confirmed), label: "fix-verifier", capability_mode: "read-only", output_schema: fix_verify_schema, phase: "Verify fixes" },
        #{ prompt: "Perform a fresh regression review of this integrated patch without trusting the fixer. Look for semantic regressions, incomplete fixes, API breakage, and newly introduced security or concurrency issues. Patch:\n" + patch, label: "regression-skeptic", capability_mode: "read-only", output_schema: finding_schema, phase: "Verify fixes" },
    ];
    let post_results = parallel(post_jobs);
    fix_verification = post_results[0];
    regression_review = post_results[1];
    if fix_verification == () || !fix_verification.success || !fix_verification.output.tests_passed {
        pause("verification", "integrated fixes did not pass independent verification");
    }
}

phase("Coverage");
let coverage_schema = #{ type: "object", properties: #{
    reviewed: #{ type: "array", items: #{ type: "string" } }, missed: #{ type: "array", items: #{ type: "string" } },
    policy_gaps: #{ type: "array", items: #{ type: "string" } }, confidence: #{ type: "number" },
}, required: ["reviewed", "missed", "policy_gaps", "confidence"] };
let coverage = agent("Audit review coverage independently. Cross-check every inventory shard and excluded area against successful investigation outputs. Check tests, enum/policy variants, cross-module interactions, and failure/cancellation paths. Do not create findings; identify coverage gaps. Inventory:\n" + json_encode(inventory.output) + "\nResults:\n" + json_encode(triage_input), #{
    label: "coverage-auditor", capability_mode: "read-only", output_schema: coverage_schema, phase: "Coverage",
});

phase("Report");
complete(#{
    confirmed_findings: adjudication.output.confirmed,
    rejected_findings: adjudication.output.rejected,
    unresolved_findings: adjudication.output.unresolved,
    duplicate_findings: triage.output.rejected_duplicates,
    coverage: coverage.output,
    inventory_exclusions: inventory.output.excluded,
    fixes_requested: apply_fixes,
    fix_result: fix_result,
    fix_verification: fix_verification,
    regression_review: regression_review,
    partial: triage_input.len() < investigations.len(),
});
