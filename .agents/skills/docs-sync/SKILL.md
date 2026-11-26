---
name: docs-sync
description: Audit and synchronize project documentation and docstrings with implementation changes. Use Git diffs as the default scope.
---

# Docs Sync

## Overview

Identify outdated, incorrect, and missing documentation by comparing the current implementation with existing project documentation.

## Authorization and scope

For audit-only requests, report findings without editing.

For update requests, including explicit invocation without additional instructions, apply unambiguous documentation changes automatically.

Ask the developer only when scope, intended behavior, architectural rationale, or correctness is uncertain.

Modify documentation and docstrings only. Preserve existing developer changes. Do not modify executable code, stage files, or create commits.

## Workflow

### 1. Determine scope

Unless the developer specifies another scope, analyze all uncommitted changes relative to HEAD.

Use:

    git status --short
    git diff --name-status -M HEAD
    git ls-files --others --exclude-standard

Inspect individual file diffs as needed.

For branch-scoped requests, compare against the specified base branch.

Do not switch branches or discard local changes.

### 2. Build an implementation inventory

Identify added, modified, renamed, and removed components within the selected scope.

Consider:

- Functions, classes, and modules.
- Public interfaces and API contracts.
- Parameters, return values, and exceptions.
- Component responsibilities and dependencies.
- Algorithms and observable behavior.
- State management, concurrency, and invariants.

Read relevant implementation, callers, and tests to establish the actual behavior.

Record evidence using file paths and symbol names.

Distinguish behavioral changes from internal refactoring.

### 3. Documentation-first review

Locate existing documentation relevant to the changes.

Use targeted searches based on changed symbols, module names, responsibilities, and domain terminology.

Review affected:

- Module, class, function, and method docstrings.
- README files.
- Architecture and component documentation.
- API documentation.
- Usage examples.

Identify descriptions that no longer match implementation.

### 4. Implementation-first review

Map changed components and behavior to the existing documentation structure.

Identify important behavior, contracts, or responsibilities that are missing from documentation.

Prefer updating existing documents over creating new ones.

Preserve existing terminology, structure, and style.

Do not document implementation details that provide no useful information to developers or users.

### 5. Identify documentation issues

Classify findings as:

- Missing documentation.
- Incorrect or outdated documentation.
- Structural issues.
- No documentation change required.

Use references/doc-coverage-checklist.md when additional review guidance is needed.

### 6. Resolve findings

Automatically update documentation when the correct description is directly supported by available evidence.

Ask the developer when:

- Implementation contradicts documented requirements.
- Tests contradict implementation.
- A behavior change may be an unintended regression.
- Design intent or architectural rationale is unknown.
- Guarantees about correctness, concurrency, or performance
  cannot be established.
- Multiple materially different interpretations are plausible.

Do not guess or silently resolve conflicting evidence.

Complete independent, unambiguous updates before asking about unresolved findings.

Leave ambiguous sections unchanged until clarified.

### 7. Apply and validate changes

Keep edits focused on the selected scope.

Follow existing documentation and docstring conventions.

Review the resulting diff against the initial working tree state.

Ensure that:

- Only documentation and docstrings were changed.
- Existing developer changes were preserved.
- Updated descriptions are supported by implementation.
- References and examples remain consistent.

Run applicable existing documentation checks and:

    git diff --check

Report any validation limitations.

## Output format

Provide a concise Docs Sync Report containing:

- Updated documentation:
  File, symbol or section, and reason for the update.

- Missing documentation addressed:
  Behavior or component and its documentation location.

- Unresolved questions:
  Conflicting or missing evidence and the required decision.

- Validation:
  Checks performed and their results.

Omit empty sections.

## References

- references/doc-coverage-checklist.md
