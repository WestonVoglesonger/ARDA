# RTL Tool Call Fix Summary

## Problem Identified

The RTL stage was failing with "Agent response for stage 'rtl' did not include textual output" because:

1. **Empty Tool Schema**: The tool schema sent to OpenAI had empty `properties: {}` and `required: []`
2. **No JSON Response After Tools**: The agent would call tools but not return the final JSON summary
3. **Unclear Instructions**: The RTL agent instructions were too verbose with Python code examples

## Root Cause

Looking at the dumped response (`/tmp/arda_openai_rtl_no_textual_output_*.json`):
```json
"tools": [{
  "name": "write_artifact",
  "parameters": {
    "additionalProperties": "False",
    "type": "object",
    "properties": {},      // EMPTY!
    "required": []         // EMPTY!
  }
}]
```

The agent thought `write_artifact` takes no arguments, so it called it with `{"arguments": "{\n \t\t}"}`.

## Fixes Implemented

### 1. Enhanced Tool Schema Debugging (`ardagen/agents/openai_runner.py`)

**Location**: `_build_tool_definitions()` method (lines 781-832)

**Changes**:
- Added detailed debug logging to inspect schema structure
- Added validation warnings for empty properties/required fields
- Improved parameter extraction to preserve full schema structure

**What it does**:
- When `ARDA_DEBUG_EXTRACTION=1`, prints schema keys, properties, and required fields for each tool
- Warns if a tool has empty or missing properties/required fields
- Helps diagnose schema loading issues

### 2. Better Error Messages for Empty Tool Arguments

**Location**: `_handle_required_actions()` method (lines 237-255)

**Changes**:
- Expanded detection of empty argument patterns
- Include required parameter schema in error messages
- Show expected parameter types

**Example error message**:
```
ERROR: Tool 'write_artifact' called with empty or whitespace-only arguments.
You must provide all required parameters.

Required parameters:
  - workspace_token: string
  - path: string
  - content: string

Raw arguments received: '{\n \t\t}'
```

### 3. Two-Phase Response Handling

**Location**: `_run_agent_sync()` method (lines 189-197)

**Changes**:
- Added `_response_is_empty()` check after tool execution
- Added `_prompt_for_final_response()` to request JSON after tools complete
- Automatic recovery when agent forgets to return JSON

**How it works**:
1. Agent calls tools (write_artifact)
2. System detects no JSON response
3. System sends follow-up prompt: "You have successfully completed the tool calls. Now provide the final JSON response..."
4. Agent returns required RTLConfig JSON

### 4. Simplified RTL Agent Instructions (`agent_configs.json`)

**Location**: RTL agent instructions (lines 387-427)

**Changes**:
- Removed verbose Python code examples
- Clear two-phase workflow:
  - Phase 1: Call write_artifact for each file
  - Phase 2: Return JSON summary
- Explicit requirements for workspace_token parameter
- Example JSON structure

**Before**: 200+ lines of Python code and verbose instructions
**After**: ~40 lines of clear, actionable instructions

## Testing

To test the fixes, run:

```bash
ARDA_DEBUG_EXTRACTION=1 ARDA_DUMP_OPENAI_RESPONSE=1 python -m ardagen.cli \
  /Users/westonvoglesonger/Projects/ALG2SV/examples/complex_adaptive_filter_bundle.txt \
  --synthesis-backend vivado \
  --fpga-family xc7a100t \
  --verbose \
  --extract-rtl generated_rtl/run1/
```

**Expected behavior**:
1. Debug output shows tool schema with all properties
2. If tool calls are empty, clear error with required parameters
3. After tools complete, agent returns JSON or is prompted for it
4. RTL files are written to workspace
5. Stage completes with valid RTLConfig

## Files Modified

1. **ardagen/agents/openai_runner.py** (lines 781-832, 237-255, 189-383)
   - Enhanced tool schema debugging
   - Better error messages for empty tool arguments
   - Two-phase response handling with automatic recovery

2. **agent_configs.json** (lines 387-427)
   - Simplified RTL agent instructions
   - Clear two-phase workflow
   - Explicit parameter requirements

## Next Steps If Issues Persist

1. **Check Debug Output**: Look for "WARNING: Tool 'write_artifact' has empty or missing properties"
2. **Inspect Schema Loading**: The debug output will show if `get_function_tool_schema()` is returning empty schemas
3. **Review agent_configs.json**: Ensure `function_tools.write_artifact.schema.parameters` has properties and required fields
4. **Check Follow-up Prompt**: Debug output will show if follow-up prompt is triggered and succeeds

