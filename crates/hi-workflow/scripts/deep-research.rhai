// deep-research — scan, analyze in parallel, then synthesize.
let meta = #{
    name: "deep-research",
    description: "Research a topic by scanning sources, analyzing them in parallel, then synthesizing a report.",
    when_to_use: "When you need breadth-first research with a synthesized summary — multiple sources analyzed concurrently, then merged into one answer.",
    phases: [
        #{ title: "Scan", detail: "identify and list relevant sources" },
        #{ title: "Analyze", detail: "parallel deep-dive into each source" },
        #{ title: "Synthesize", detail: "merge analyses into a single report" },
    ],
};

phase("Scan");
let scan_result = agent("Scan the codebase or topic for: " + args.input + ". List the key areas, files, or sources worth investigating. Return a JSON array of strings, each a concise subject to analyze.");
// The agent's output is a JSON value. In validation (stub host) it's a map
// with fixed fields; in real execution it's whatever the agent returned.
// Extract a list of subjects defensively.
let subjects = if type_of(scan_result.output) == "array" {
    scan_result.output
} else if type_of(scan_result.output) == "map" && scan_result.output.contains("subjects") {
    scan_result.output.subjects
} else {
    // Fallback: treat the output as a single-subject list.
    [json_encode(scan_result.output)]
};

phase("Analyze");
let jobs = [];
for subject in subjects {
    jobs.push(#{
        prompt: "Analyze this subject in depth: " + subject + ". Context: " + args.input + ". Provide findings, key facts, and notable patterns.",
        label: subject,
        phase: "Analyze",
    });
}
let analyses = parallel(jobs);

phase("Synthesize");
let summaries = [];
for a in analyses {
    let summary_text = if type_of(a.output) == "map" {
        a.output.summary
    } else {
        json_encode(a.output)
    };
    summaries.push("# " + a.label + "\n" + summary_text);
}
// Manual join with separator (Rhai arrays don't have .join()).
let combined = "";
let first = true;
for s in summaries {
    if !first {
        combined += "\n\n---\n\n";
    }
    combined += s;
    first = false;
}
let synth = agent("Synthesize these analyses into a single coherent report. Identify consensus, contradictions, and gaps. Write a clear summary with citations to the analyzed areas.\n\n" + combined);

complete(synth.output);
