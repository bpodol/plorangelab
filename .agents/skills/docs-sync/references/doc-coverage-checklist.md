
# Documentation Coverage Checklist

Apply this checklist only within the selected scope.

## Implementation inventory

- Functions, classes, modules, and entry points.
- Parameters, types, defaults, and return values.
- Exceptions and validation requirements.
- Responsibilities and component relationships.
- State transitions and data ownership.
- Mutation, side effects, and invariants.
- Algorithms and their assumptions.
- Concurrency, cancellation, and resource management.
- Removed or renamed functionality.

## Docstrings

- Does the summary describe the actual responsibility?
- Are documented parameters and return values accurate?
- Are relevant exceptions documented?
- Are important assumptions and constraints explained?
- Are mutation and side effects described where relevant?
- Does the docstring follow existing project conventions?

## Project documentation

- Do component descriptions match implementation?
- Are documented dependencies still correct?
- Do API descriptions match actual contracts?
- Are workflows and state transitions accurate?
- Are examples and symbol references valid?
- Is important new functionality documented?

## Evidence

- Record relevant file paths and symbol names.
- Inspect callers and tests when necessary.
- Separate observable behavior from design intent.
- Do not infer guarantees from naming alone.

## Patch guidance

- Prefer minimal, targeted changes.
- Avoid duplicating information.
- Preserve useful design rationale.
- Update references to renamed or removed components.
- Do not rewrite documentation that remains accurate.
- Escalate conflicting evidence to the developer.
