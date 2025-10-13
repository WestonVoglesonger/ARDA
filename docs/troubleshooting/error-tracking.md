# Error Tracking Log

This document tracks errors, bugs, and issues encountered during development, along with their resolution status.

## Format
- **Date**: When the error was first encountered
- **Error ID**: Unique identifier (e.g., ERR-001)
- **Description**: What happened and symptoms
- **Root Cause**: Why it occurred
- **Impact**: What was affected
- **Resolution**: How it was fixed
- **Status**: [OPEN, RESOLVED, IN_PROGRESS, DEFERRED]
- **Files Modified**: Relevant files changed
- **Notes**: Additional context

---

## ERR-001: Feedback Agent Missing retry_architecture Action

**Date**: 2025-10-13
**Error ID**: ERR-001
**Description**:
Pipeline execution failed when architecture stage encountered validation errors. The feedback agent attempted to suggest `retry_architecture` but this action was not defined in the `FeedbackDecision` model, causing a Pydantic validation error:

```
FAIL [feedback] stage_failed error=1 validation error for FeedbackDecision
action
  Input should be 'continue', 'retry_spec', 'retry_quant', 'retry_microarch', 'retry_rtl', 'retry_verify', 'retry_lint', 'retry_test_generation', 'retry_simulation', 'retry_synth', 'retry_evaluate', 'tune_microarch' or 'abort' [type=literal_error, input_value='retry_architecture', input_type=str]
```

**Root Cause**:
The `FeedbackDecision.action` field used a `Literal` type that included retry actions for all pipeline stages except `architecture`. While the agent configurations included `retry_architecture`, the Pydantic model validation was missing this action.

**Impact**:
- Pipeline execution halts when architecture stage fails validation
- Feedback loop cannot suggest retrying architecture stage
- Blocks automated recovery from architecture validation failures

**Resolution**:
Added `"retry_architecture"` to the `Literal` type in `FeedbackDecision.action` field in `/Users/westonvoglesonger/Projects/ALG2SV/ardagen/domain/feedback.py`.

**Status**: RESOLVED
**Files Modified**:
- `ardagen/domain/feedback.py` (added `"retry_architecture"` to action Literal)

**Notes**:
- Agent configs in `agent_configs.json` already included this action
- The architecture stage exists in the pipeline but retry action was missing from validation
- This was likely an oversight during initial implementation

---

## Template for New Errors

**Date**: YYYY-MM-DD
**Error ID**: ERR-XXX
**Description**:

**Root Cause**:

**Impact**:

**Resolution**:

**Status**: [OPEN, RESOLVED, IN_PROGRESS, DEFERRED]
**Files Modified**:

**Notes**:

