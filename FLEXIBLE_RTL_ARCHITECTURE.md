# Flexible RTL Architecture: Removing File Structure Constraints

**Date:** October 10, 2025  
**Priority:** HIGH (Phase 0 - Prerequisite for verification improvements)  
**Implementation Time:** 1-2 days  
**Expected Impact:** 40-60% reduction in architectural bugs

---

## Executive Summary

**Problem:** The current RTL generation pipeline forces all designs into a fixed 3-file template (`params.svh`, `algorithm_core.sv`, `algorithm_top.sv`), causing monolithic code that is:
- Hard to verify
- Prone to timing violations
- Difficult to debug
- Forces unnatural architectures

**Evidence:** Two completely different algorithms (Adaptive Filter and Conv2D) both generated identical file structures, leading to similar classes of bugs.

**Solution:** Remove file structure constraints and let the agent generate 3-15 files based on natural architectural decomposition.

**Benefits:**
- ✅ Better modularity → easier verification
- ✅ Smaller modules → better timing
- ✅ Reusable components (FIFOs, MACs, FSMs)
- ✅ More natural architectures
- ✅ Fewer fatal bugs

---

## Problem Statement: Evidence from Two Designs

### Design Comparison

| Aspect | Adaptive Filter | Conv2D | Observation |
|--------|----------------|--------|-------------|
| **Algorithm** | 16-tap FIR + LMS adaptation | 3×3 convolution + ReLU | Completely different |
| **Files Generated** | 3 | 3 | **Identical count** |
| **File Names** | params.svh, algorithm_core.sv, algorithm_top.sv | **params.svh, algorithm_core.sv, algorithm_top.sv** | **Identical names!** |
| **Core Module Size** | 272 lines | 137 lines | Both monolithic |
| **Fatal Bugs** | 4 | 2 | Both have critical issues |
| **Verification** | ✅ Passed (6/6) | ✅ Passed (50/50) | Both false positives |

### What Should Have Been Generated

#### Adaptive Filter (Natural Architecture):
```
rtl/
├── fir_params.sv                 # Parameters
├── fir_mac_pipeline.sv           # FIR computation (16 taps)
├── fir_adder_tree.sv             # Reduction tree
├── lms_update_unit.sv            # Coefficient adaptation logic (separate!)
├── tap_delay_buffer.sv           # Properly aligned tap storage
├── error_computation.sv          # Error signal generation
└── adaptive_filter_top.sv        # Top-level integration

7 files, modular, each <100 lines
```

**Benefits of this structure:**
- LMS tap alignment bug would be isolated in `lms_update_unit.sv`
- Could unit test each module separately
- Reusable `fir_mac_pipeline.sv` for other designs

#### Conv2D (Natural Architecture):
```
rtl/
├── conv2d_params.sv              # Parameters
├── conv2d_pe.sv                  # Processing Element (1 MAC)
├── conv2d_pe_array.sv            # 16×PE array
├── conv2d_fifo.sv                # Buffering (separate!)
├── conv2d_control_fsm.sv         # Control state machine
└── conv2d_top.sv                 # Top-level integration

6 files, modular, each <80 lines
```

**Benefits of this structure:**
- FIFO count bug would be in isolated 50-line module
- Could write targeted FIFO tests (simultaneous read/write)
- Reusable `conv2d_pe.sv` for other convolution sizes

### What Was Actually Generated

#### Both Designs (Current):
```
rtl/
├── params.svh                    # Parameters (OK)
├── algorithm_core.sv             # EVERYTHING CRAMMED HERE
│   ├── Computation logic
│   ├── Pipeline management
│   ├── FIFO / buffering
│   ├── Control logic
│   ├── Adaptation logic (adaptive filter)
│   └── All in 137-272 lines!
└── algorithm_top.sv              # Thin wrapper

3 files, monolithic
```

**Problems:**
- ❌ Impossible to unit test individual components
- ❌ Massive combinational blocks (timing violations)
- ❌ Bugs hidden in complexity
- ❌ Can't reuse components
- ❌ Verification can't isolate failures

---

## Root Cause Analysis

### Current Implementation

**File:** `agent_configs.json` (Lines 56-72)
```
REQUIRED FILES TO GENERATE:

1. params.svh - Parameter package with:
   - COEFF_WIDTH, DATA_WIDTH, PIPELINE_DEPTH
   - Coefficient ROM initialization
   - Type definitions

2. algorithm_core.sv - Main computation module:
   - FIR filter logic with adaptive coefficients
   - Pipeline registers per microarch.pipeline_depth
   - Fixed-point arithmetic

3. algorithm_top.sv - Top-level wrapper:
   - I/O ports matching spec formats
   - Handshake logic (ready_valid or axis per microarch)
   - Instantiates algorithm_core
```

**File:** `ardagen/core/stages/rtl_stage.py` (Lines 56-62)
```python
file_map = {
    "params_svh": "rtl/params.svh",
    "algorithm_core_sv": "rtl/algorithm_core.sv", 
    "algorithm_top_sv": "rtl/algorithm_top.sv"
}
```

**File:** `agent_configs.json` (Lines 199-204)
```json
"properties": {
  "params_svh": {"type": "string"},
  "algorithm_core_sv": {"type": "string"},
  "algorithm_top_sv": {"type": "string"}
},
"required": ["params_svh", "algorithm_core_sv", "algorithm_top_sv"]
```

### The Constraint Chain

```
Instructions            Schema                  Post-Processing
     ↓                    ↓                           ↓
"Generate these      "required:              file_map = {
 3 files"             [params_svh,             "params_svh": ...,
                      core_sv,                 "core_sv": ...,
                      top_sv]"                 "top_sv": ...
                                              }

        ↓                ↓                          ↓
    Agent forced     Agent can't         Only 3 files written
    into template    add files           to workspace
```

**Result:** Agent has no choice but to cram everything into `algorithm_core.sv`.

---

## Proposed Solution

### Phase 1: Flexible Schema

**File:** `agent_configs.json` → `rtl_agent.output_schema`

**From (Fixed):**
```json
"generated_files": {
  "type": "object",
  "properties": {
    "params_svh": {"type": "string"},
    "algorithm_core_sv": {"type": "string"},
    "algorithm_top_sv": {"type": "string"}
  },
  "required": ["params_svh", "algorithm_core_sv", "algorithm_top_sv"]
}
```

**To (Flexible):**
```json
"generated_files": {
  "type": "object",
  "description": "Generated RTL files keyed by logical name (e.g., 'conv2d_pe_sv', 'fir_pipeline_sv'). Keys should be descriptive snake_case names ending in _sv or _svh.",
  "additionalProperties": {
    "type": "string",
    "minLength": 100,
    "maxLength": 50000,
    "description": "Complete SystemVerilog file content"
  },
  "minProperties": 3,
  "maxProperties": 15
}
```

**Key Changes:**
- ✅ No fixed property names
- ✅ Agent chooses file names
- ✅ 3-15 files allowed (prevents both under and over-modularization)
- ✅ Size validation (100 bytes - 50KB per file)

### Phase 2: Updated Instructions

**File:** `agent_configs.json` → `rtl_agent.instructions`

**Remove:**
```
REQUIRED FILES TO GENERATE:

1. params.svh - Parameter package...
2. algorithm_core.sv - Main computation module...
3. algorithm_top.sv - Top-level wrapper...
```

**Add:**
```
FILE ORGANIZATION:

Generate a modular RTL design with 3-15 SystemVerilog files.
Consider natural architectural decomposition:

RECOMMENDED STRUCTURE:
- Parameters/constants package (1 file)
- Reusable computational units (1-5 files)
  * Processing elements (MAC, ALU, etc.)
  * Memory structures (FIFOs, buffers, RAM controllers)
  * State machines (FSM controllers)
- Integration/hierarchy (1-3 files)
  * Array instantiation
  * Interconnect
  * Top-level wrapper

NAMING CONVENTION:
- Use descriptive snake_case names
- End with _sv for .sv files, _svh for .svh files
- Examples:
  * "conv2d_params_svh" → rtl/conv2d_params.svh
  * "fir_mac_pipeline_sv" → rtl/fir_mac_pipeline.sv
  * "adaptive_filter_top_sv" → rtl/adaptive_filter_top.sv

MODULARITY GUIDELINES:
- Each module should have ONE primary responsibility
- Target 50-150 lines per module (except parameters)
- Separate computation from control
- Isolate complex logic (FIFOs, FSMs) into dedicated modules
- Make components reusable when possible

FILE FIELDS:
Return JSON with:
{
  "generated_files": {
    "descriptive_name_sv": "<complete SystemVerilog code>",
    "another_module_sv": "<complete SystemVerilog code>",
    ...
  },
  "file_paths": ["rtl/descriptive_name.sv", "rtl/another_module.sv", ...],
  "top_module": "name_of_top_module",
  ...
}

CRITICAL:
- Generate actual synthesizable SystemVerilog
- One complete module per file
- Include all module instantiations in file_paths
- Top-level module must instantiate all sub-modules
```

### Phase 3: Dynamic File Writing

**File:** `ardagen/core/stages/rtl_stage.py`

**From (Fixed Mapping):**
```python
def _write_rtl_files(self, workspace_token: str, rtl_config: RTLConfig) -> None:
    """Write generated RTL files to workspace."""
    workspace = workspace_manager.get_workspace(workspace_token)
    if not workspace:
        return
    
    # Map logical names to file paths
    file_map = {
        "params_svh": "rtl/params.svh",
        "algorithm_core_sv": "rtl/algorithm_core.sv", 
        "algorithm_top_sv": "rtl/algorithm_top.sv"
    }
    
    for logical_name, content in rtl_config.generated_files.items():
        if logical_name in file_map:
            path = file_map[logical_name]
            workspace.add_file(path, content)
            print(f"✓ Wrote {path} ({len(content)} bytes)")
```

**To (Dynamic Mapping):**
```python
def _write_rtl_files(self, workspace_token: str, rtl_config: RTLConfig) -> None:
    """Write generated RTL files to workspace with dynamic naming."""
    workspace = workspace_manager.get_workspace(workspace_token)
    if not workspace:
        print(f"⚠️  Could not find workspace with token {workspace_token}")
        return
    
    if not rtl_config.generated_files:
        print(f"⚠️  No files in generated_files")
        return
    
    # Convert logical names to file paths
    files_written = 0
    for logical_name, content in rtl_config.generated_files.items():
        # Convert snake_case to file path
        # e.g., "conv2d_pe_sv" → "rtl/conv2d_pe.sv"
        #       "fir_params_svh" → "rtl/fir_params.svh"
        path = self._logical_to_physical_path(logical_name)
        
        # Validate content
        if not self._validate_rtl_content(logical_name, content):
            print(f"⚠️  Skipping {logical_name}: validation failed")
            continue
        
        # Write to workspace
        workspace.add_file(path, content)
        print(f"✓ Wrote {path} ({len(content)} bytes)")
        files_written += 1
    
    if files_written == 0:
        print(f"⚠️  No valid files were written!")
    else:
        print(f"✅ Successfully wrote {files_written} RTL files")

def _logical_to_physical_path(self, logical_name: str) -> str:
    """Convert logical name to physical file path.
    
    Examples:
        "conv2d_pe_sv" → "rtl/conv2d_pe.sv"
        "fir_params_svh" → "rtl/fir_params.svh"
        "adaptive_filter_top_sv" → "rtl/adaptive_filter_top.sv"
    """
    # Remove trailing type suffix and add proper extension
    if logical_name.endswith("_svh"):
        filename = logical_name[:-4] + ".svh"
    elif logical_name.endswith("_sv"):
        filename = logical_name[:-3] + ".sv"
    else:
        # Default to .sv if no suffix
        filename = logical_name + ".sv"
    
    return f"rtl/{filename}"

def _validate_rtl_content(self, logical_name: str, content: str) -> bool:
    """Basic validation of RTL file content."""
    if not content or len(content) < 100:
        print(f"  → {logical_name}: Content too short ({len(content)} bytes)")
        return False
    
    # Check for basic SystemVerilog structure
    has_module = "module " in content
    has_endmodule = "endmodule" in content
    
    # .svh files (packages) may not have modules
    if logical_name.endswith("_svh"):
        # Check for package or parameters
        if not any(kw in content for kw in ["package ", "parameter ", "typedef "]):
            print(f"  → {logical_name}: No package/parameter declarations found")
            return False
    else:
        # .sv files must have modules
        if not (has_module and has_endmodule):
            print(f"  → {logical_name}: Missing module/endmodule pair")
            return False
        
        # Count matches
        module_count = content.count("module ")
        endmodule_count = content.count("endmodule")
        if module_count != endmodule_count:
            print(f"  → {logical_name}: Mismatched module/endmodule ({module_count} vs {endmodule_count})")
            return False
    
    return True
```

### Phase 4: Update Domain Model

**File:** `ardagen/domain/rtl_artifacts.py`

**Add validation:**
```python
from pydantic import BaseModel, Field, validator

class RTLConfig(BaseModel):
    """RTL generation configuration and artifacts."""

    # Flexible file contents (no fixed keys)
    generated_files: Dict[str, str] = Field(
        description="Generated SystemVerilog file contents keyed by logical name"
    )
    
    # File paths must match generated_files keys
    file_paths: List[str] = Field(description="Paths where files will be written")
    top_module: str
    estimated_resources: Dict[str, int]
    confidence: float = Field(default=80.0, ge=0, le=100)
    
    # Optional fields
    lint_passed: bool = False
    params_file: Optional[str] = None
    
    # Deprecated
    rtl_files: Optional[List[str]] = None
    
    @validator('generated_files')
    def validate_file_count(cls, v):
        """Ensure reasonable file count."""
        if len(v) < 3:
            raise ValueError(f"Must generate at least 3 files, got {len(v)}")
        if len(v) > 15:
            raise ValueError(f"Too many files (max 15), got {len(v)}")
        return v
    
    @validator('generated_files')
    def validate_file_sizes(cls, v):
        """Ensure reasonable file sizes."""
        for name, content in v.items():
            if len(content) < 100:
                raise ValueError(f"File {name} too short: {len(content)} bytes")
            if len(content) > 50000:
                raise ValueError(f"File {name} too large: {len(content)} bytes")
        return v
    
    @validator('file_paths')
    def validate_paths_match_files(cls, v, values):
        """Ensure file_paths matches generated_files count."""
        if 'generated_files' in values:
            gen_count = len(values['generated_files'])
            path_count = len(v)
            if gen_count != path_count:
                raise ValueError(
                    f"Mismatch: {gen_count} files generated but {path_count} paths provided"
                )
        return v

    class Config:
        extra = "allow"
```

---

## Expected Benefits

### 1. Better Verification

**Before:** Monolithic `algorithm_core.sv` (137-272 lines)
- Hard to isolate failures
- Can't unit test components
- Bugs hide in complexity

**After:** Modular files (50-150 lines each)
- ✅ Unit test each module independently
- ✅ Pin down exact location of failures
- ✅ Clear module boundaries

**Example:**
```systemverilog
// Before: FIFO embedded in algorithm_core.sv
// Bug: Simultaneous read/write corrupts count
// Test: Must test entire algorithm to see bug

// After: Separate conv2d_fifo.sv (50 lines)
// Bug: Same issue but isolated
// Test: Write targeted FIFO testbench
module tb_conv2d_fifo;
  // Test simultaneous read/write
  initial begin
    fifo_write = 1; fifo_read = 1;
    @(posedge clk);
    assert(count == initial_count) else $error("Count bug!");
  end
endmodule
```

### 2. Better Timing

**Before:** Massive combinational blocks
- Conv2D: 432 MACs in single always_comb
- Timing path: 5.8ns > 5.0ns @ 200MHz
- Can't meet timing

**After:** Natural pipeline boundaries
```systemverilog
// Before: All in always_comb
always_comb begin
  for (c = 0; c < 16; c++)
    for (i = 0; i < 27; i++)
      temp_acc += din[i] * coeff[c*27+i];
end

// After: Separate pipeline stages
// conv2d_pe.sv - single MAC (1 cycle)
// conv2d_pe_array.sv - 16 parallel MACs + register (1 cycle)
// conv2d_accumulator.sv - tree reduction (2 cycles)
// conv2d_activation.sv - ReLU + saturate (1 cycle)
// Total: 5 cycles, each <2ns → 500 MHz capable!
```

### 3. Reusable Components

**Before:** Everything coupled, can't reuse

**After:** Library of verified components
```
rtl/
├── common/
│   ├── fifo_sync.sv          # Generic FIFO
│   ├── mac_unit.sv            # Generic MAC
│   └── handshake_buffer.sv    # Generic buffer
└── designs/
    ├── conv2d/
    │   ├── conv2d_pe.sv       # Uses mac_unit
    │   ├── conv2d_fifo.sv     # Instantiates fifo_sync
    │   └── conv2d_top.sv
    └── adaptive_filter/
        ├── fir_mac.sv         # Uses mac_unit
        ├── fir_buffer.sv      # Instantiates fifo_sync
        └── adaptive_top.sv
```

### 4. Easier Debugging

**Before:**
```
ERROR: Synthesis failed, timing violation in algorithm_core.sv
  Critical path: Line 46 → Line 86 (5.8ns)
  
Where exactly? 🤷 (40 lines of nested loops)
```

**After:**
```
ERROR: Synthesis failed, timing violation in conv2d_accumulator.sv
  Critical path: Line 23 → Line 45 (3.2ns)
  
Clear: It's the adder tree in the accumulator module!
Fix: Pipeline the tree over 2 cycles
```

---

## Migration Path

### Phase 0: Preparation (2 hours)

1. **Backup current config:**
   ```bash
   cp agent_configs.json agent_configs.json.backup
   cp ardagen/core/stages/rtl_stage.py ardagen/core/stages/rtl_stage.py.backup
   ```

2. **Create feature branch:**
   ```bash
   git checkout -b feature/flexible-rtl-architecture
   ```

### Phase 1: Schema & Instructions (4 hours)

1. ✅ Update `agent_configs.json`:
   - Modify `rtl_agent.output_schema` (flexible schema)
   - Update `rtl_agent.instructions` (new guidance)
   - Remove file name requirements

2. ✅ Update `ardagen/domain/rtl_artifacts.py`:
   - Add validators for file count/size
   - Add path matching validation

### Phase 2: File Writing (4 hours)

1. ✅ Update `ardagen/core/stages/rtl_stage.py`:
   - Implement `_logical_to_physical_path()`
   - Implement `_validate_rtl_content()`
   - Update `_write_rtl_files()` with dynamic mapping

2. ✅ Add logging for debugging:
   - File count
   - Validation results
   - Path mappings

### Phase 3: Testing (4 hours)

1. ✅ Unit tests:
```python
# tests/test_flexible_rtl.py
def test_logical_to_physical_path():
    stage = RTLStage()
    assert stage._logical_to_physical_path("conv2d_pe_sv") == "rtl/conv2d_pe.sv"
    assert stage._logical_to_physical_path("params_svh") == "rtl/params.svh"

def test_rtl_config_validates_file_count():
    # Too few files
    with pytest.raises(ValidationError):
        RTLConfig(generated_files={"a_sv": "...", "b_sv": "..."}, ...)
    
    # Too many files  
    with pytest.raises(ValidationError):
        RTLConfig(generated_files={f"file{i}_sv": "..." for i in range(20)}, ...)

def test_file_writing_dynamic():
    config = RTLConfig(
        generated_files={
            "conv2d_params_svh": "package ...",
            "conv2d_pe_sv": "module ...",
            "conv2d_top_sv": "module ..."
        },
        file_paths=["rtl/conv2d_params.svh", "rtl/conv2d_pe.sv", "rtl/conv2d_top.sv"],
        ...
    )
    stage._write_rtl_files(token, config)
    # Verify 3 files written
```

2. ✅ Integration test:
```bash
# Test with existing design
python -m ardagen.cli examples/conv2d_bundle.txt --verbose

# Expected: Agent generates more modular design
# Should see: 4-8 files instead of 3
```

### Phase 4: Validation (2 hours)

1. ✅ Run full pipeline on multiple designs
2. ✅ Compare modularity metrics:
   - Average file size
   - Lines per module
   - Module count
3. ✅ Check for improvement in bug detection

### Phase 5: Rollout (2 hours)

1. ✅ Update tests for new file structure
2. ✅ Update documentation
3. ✅ Merge to main
4. ✅ Monitor first production runs

**Total: 1-2 days**

---

## Validation Criteria

### Success Metrics

| Metric | Current | Target | Measurement |
|--------|---------|--------|-------------|
| **Files per design** | 3 (fixed) | 4-8 (variable) | Count in generated_files |
| **Lines per file** | 137-272 | 50-150 | Average LoC |
| **Module reusability** | 0% | 30%+ | Same module in 2+ designs |
| **Timing violations** | 50% | <20% | Synthesis reports |
| **Bug isolation** | Hard | Easy | Time to locate bug |
| **Verification granularity** | Coarse | Fine | Unit test count |

### Test Cases

1. **Conv2D Redesign:**
   - Should generate 5-7 files
   - Should have separate FIFO module
   - Should have separate PE module
   - FIFO bug should be easier to spot

2. **Adaptive Filter Redesign:**
   - Should generate 6-8 files
   - Should have separate LMS update module
   - Should have separate tap delay module
   - Tap alignment bug should be isolated

3. **New Design (FIR Filter):**
   - Should reuse MAC/FIFO modules
   - Should generate 4-5 files
   - Should follow modular pattern

---

## Risk Assessment

### Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| **Agent generates too many files** | Low | Medium | Schema limits to 15 max |
| **Agent generates wrong file names** | Medium | Low | Validation catches, falls back |
| **Breaking existing tests** | High | Medium | Update tests before merge |
| **Agent ignores new guidance** | Medium | High | Test with multiple designs first |
| **File path conflicts** | Low | Medium | Validation detects duplicates |

### Rollback Plan

If issues occur:
1. Revert `agent_configs.json` from backup
2. Revert `rtl_stage.py` from backup
3. Existing 3-file structure still works
4. No data loss (all changes in code only)

---

## Integration with Verification Improvements

This change is **Phase 0** for `PIPELINE_VERIFICATION_IMPROVEMENTS.md`.

### How They Connect:

**Flexible Architecture (This Doc):**
- Enables better modular design
- Creates natural test boundaries
- Makes components reusable

**↓ Feeds Into ↓**

**Verification Improvements (Main Doc):**
- Unit test individual modules
- Protocol compliance per module
- Targeted bug detection

### Updated Timeline:

```
Week 0 (Phase 0): Flexible Architecture  ← THIS DOCUMENT
  - Remove file constraints
  - Enable modular design
  
Week 1 (Phase 1): Critical Bug Detection
  - Enhanced instructions
  - Post-generation validation
  - Bug pattern checking
  
Week 2 (Phase 2): Enhanced Verification
  - Protocol tests
  - Convergence tests
  - Stress tests
  
Week 3 (Phase 3): Expert Review
  - Automated review stage
  - Timing analysis
  - Resource checks
  
Week 4 (Phase 4): Feedback Tuning
  - Better error messages
  - Fix suggestions
  - Continuous improvement
```

**Impact:** Flexible architecture makes weeks 1-4 MORE effective because bugs are easier to isolate!

---

## Examples: Before & After

### Example 1: Conv2D

**Before (Current):**
```json
{
  "generated_files": {
    "params_svh": "...",
    "algorithm_core_sv": "... 137 lines, everything mixed ...",
    "algorithm_top_sv": "... thin wrapper ..."
  },
  "file_paths": ["rtl/params.svh", "rtl/algorithm_core.sv", "rtl/algorithm_top.sv"]
}
```

**After (Flexible):**
```json
{
  "generated_files": {
    "conv2d_params_svh": "... parameters ...",
    "conv2d_pe_sv": "... 45 lines, single MAC unit ...",
    "conv2d_pe_array_sv": "... 67 lines, 16×PE instantiation ...",
    "conv2d_fifo_sv": "... 52 lines, FIFO with count fix ...",
    "conv2d_control_fsm_sv": "... 78 lines, state machine ...",
    "conv2d_top_sv": "... 58 lines, integration ..."
  },
  "file_paths": [
    "rtl/conv2d_params.svh",
    "rtl/conv2d_pe.sv",
    "rtl/conv2d_pe_array.sv",
    "rtl/conv2d_fifo.sv",
    "rtl/conv2d_control_fsm.sv",
    "rtl/conv2d_top.sv"
  ]
}
```

**Improvements:**
- ✅ FIFO bug isolated in 52-line module
- ✅ PE can be unit tested independently
- ✅ Natural pipeline boundaries in array module
- ✅ Control separate from datapath

### Example 2: Adaptive Filter

**Before (Current):**
```json
{
  "generated_files": {
    "params_svh": "...",
    "algorithm_core_sv": "... 272 lines, FIR+LMS+pipeline ...",
    "algorithm_top_sv": "... thin wrapper ..."
  }
}
```

**After (Flexible):**
```json
{
  "generated_files": {
    "fir_params_svh": "... parameters ...",
    "fir_mac_pipeline_sv": "... 83 lines, FIR computation ...",
    "tap_delay_buffer_sv": "... 56 lines, aligned tap storage ...",
    "error_computation_sv": "... 42 lines, error calculation ...",
    "lms_update_unit_sv": "... 91 lines, coefficient adaptation ...",
    "fir_adder_tree_sv": "... 67 lines, reduction tree ...",
    "adaptive_filter_top_sv": "... 72 lines, integration ..."
  },
  "file_paths": [
    "rtl/fir_params.svh",
    "rtl/fir_mac_pipeline.sv",
    "rtl/tap_delay_buffer.sv",
    "rtl/error_computation.sv",
    "rtl/lms_update_unit.sv",
    "rtl/fir_adder_tree.sv",
    "rtl/adaptive_filter_top.sv"
  ]
}
```

**Improvements:**
- ✅ Tap alignment bug isolated in `tap_delay_buffer.sv`
- ✅ LMS update can be tested separately
- ✅ FIR pipeline reusable for other designs
- ✅ Error computation clear and testable

---

## Conclusion

**Problem:** Fixed 3-file template forces monolithic designs with hidden bugs.

**Solution:** Flexible 3-15 file architecture with agent-chosen modularity.

**Benefits:**
- 40-60% reduction in architectural bugs (estimated)
- Better timing (smaller critical paths)
- Easier verification (unit testable components)
- Reusable components across designs

**Cost:** 1-2 days implementation, minimal risk.

**Priority:** HIGH - Prerequisite for verification improvements.

**Status:** Ready to implement.

---

## Next Steps

1. ✅ Review this document
2. ✅ Get approval for changes
3. ✅ Create feature branch
4. ✅ Implement Phase 1 (Schema & Instructions)
5. ✅ Implement Phase 2 (File Writing)
6. ✅ Implement Phase 3 (Testing)
7. ✅ Validate with existing designs
8. ✅ Merge and deploy

**Estimated completion:** 2 business days from approval.

**Then:** Proceed with full verification improvements (4 weeks).

