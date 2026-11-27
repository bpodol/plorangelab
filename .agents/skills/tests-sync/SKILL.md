---
name: tests-sync
description: Synchronize tests with developer changes. Invoke explicitly with $tests-sync.
---

# Test synchronization

Maintain behavioral test guarantees after developer changes.

## Boundaries

Edit only approved tests. Preserve developer changes,
production code, testing configuration, and Git index.
Leave all changes uncommitted.

Preserve existing assertion intent. Coverage and mutation
scores are evidence, not targets.

Wait for commands to complete. Request execution approval
for known sandbox restrictions. Report unresolved blockers.

## 1. Intent

Inspect staged, unstaged, and untracked changes against HEAD.

Account for every modified production module. Read relevant
implementation, callers, documentation, and tests.

Summarize:
- Intended behavior and interface changes.
- Behavior that must remain unchanged.
- Affected components and ambiguities.

Proceed after explicit developer approval.

## 2. Regression gate

Run the complete test suite with coverage:

```sh
uv run --no-sync pytest --cov --cov-context=test --cov-report=term-missing --cov-report=json --cov-report=xml
```

Account for every failure:

- Regression: report and stop.
- Approved behavior change: propose a test update that
  preserves the original assertion intent.
- Infrastructure failure: resolve through supported
  execution approval or report a blocker.
- Ambiguity: ask the developer.

Proceed only when all failures are explained by approved
changes, or the suite passes.

## 3. Test gaps

Run diff coverage:

```sh
uv run --no-sync diff-cover coverage.xml --compare-branch=HEAD --include-untracked
```

Inspect coverage.json for affected modules, missing
statements, branches, and relevant test contexts.

Identify behavioral gaps, including edge cases,
error paths, and untested approved changes.

For Numba-compiled kernels, assess behavioral scenarios
directly rather than relying on coverage or mutation scores.

Propose test additions and updates, stating the behavior
each verifies. Obtain approval before editing.

## 4. Feedback loop

Implement approved test changes.

Repeat the full-suite and coverage checks from steps 2–3.

Run mutation testing on modified production modules and
other affected modules where justified:

```sh
uv run --no-sync pytest -q --gremlins --gremlin-targets=PATH_TO_MODULE --gremlin-report=console --gremlin-report=json
```

Use the complete suite and configured Gremlins settings.
Inspect surviving mutants for meaningful behavioral gaps.
Treat equivalent mutations and unsupported Numba mutations
as limitations, not reasons to manufacture tests.

Filter mutation reports before reading them into context.

Read aggregate and per-file statistics, skipped files,
and non-killed mutants (ID, path, line, description,
status). Retrieve complete mutant details only when
necessary for investigation.

Use shell JSON filtering rather than printing the
complete report.

Propose further edits when meaningful gaps remain.
Obtain approval before implementing them.

If a production regression emerges, undo only your own
test edits when safe, report the regression, and stop.

Finish when the full suite passes and meaningful test
gaps are addressed or explicitly reported.

## 5. Review

Review the final Git diff against the initial changes.
Verify the boundaries and ensure reported results reflect
the final test state.

Report concisely:
- Approved intent.
- Tests changed and behaviors verified.
- Full-suite results.
- Statement, branch, and diff coverage.
- Mutation results and surviving mutants.
- Unresolved gaps and blockers.