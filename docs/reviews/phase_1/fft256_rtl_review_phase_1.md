# FFT256 RTL Deep Review

**Date:** October 10, 2025  
**Algorithm:** 256-Point FFT (Cooley-Tukey)  
**Pipeline Run:** `test_algorithms/fft256_bundle.txt`  
**Status:** ❌ **CRITICAL FAILURE - Does Not Implement FFT**

---

## Executive Summary

The generated RTL **completely fails to implement the FFT algorithm**. While the pipeline reported 100% verification pass and successful synthesis, the actual RTL only performs sequential complex multiplication with 8 twiddle factors. This is functionally equivalent to a **simple frequency shift**, not a Fast Fourier Transform.

### Severity Assessment

| Issue | Severity | Impact |
|-------|----------|--------|
| Missing FFT algorithm structure | 🔴 **FATAL** | No FFT computation whatsoever |
| Missing butterfly operations | 🔴 **FATAL** | Core FFT operation absent |
| Missing bit-reversal | 🔴 **FATAL** | Required input reordering absent |
| Wrong twiddle factor count | 🔴 **FATAL** | 8 vs 128 required |
| Missing stage structure | 🔴 **FATAL** | No pipeline stages for radix-2 |
| Missing intermediate storage | 🔴 **FATAL** | Can't compute FFT without memory |
| False verification pass | 🔴 **FATAL** | Verification stage failed to catch this |

**Result:** Generated RTL would produce completely incorrect frequency-domain output.

---

## 1. Algorithm Analysis: What FFT Requires

### Cooley-Tukey Radix-2 FFT (256-point)

**Essential Components:**

1. **Bit-Reversal Reordering**
   - Input samples must be reordered using bit-reversed indices
   - Required for in-place FFT computation
   - **Status:** ❌ Not present

2. **8 Butterfly Stages** (log₂(256) = 8)
   - Each stage processes 256 samples
   - Stage k processes butterflies with span 2^k
   - **Status:** ❌ Not present

3. **Butterfly Operation** (core computation)
   ```
   X[i]     = X[i] + W^k * X[j]
   X[j]     = X[i] - W^k * X[j]
   ```
   - Two-input, two-output operation
   - **Status:** ❌ Not present

4. **128 Unique Twiddle Factors**
   - W^k = e^(-j2πk/N) for k = 0..127
   - Symmetric, so can store 128 and derive others
   - **Status:** ❌ Only 8 (4 unique) provided

5. **Intermediate Storage**
   - Need to store all 256 samples between stages
   - BRAM or distributed RAM
   - **Status:** ❌ Only 16-deep FIFO

6. **Stage Sequencing**
   - Must process all 8 stages in order
   - Each stage must complete before next begins
   - **Status:** ❌ No stage control

---

## 2. What the Generated RTL Actually Does

### algorithm_core.sv Analysis

**Lines 38-51: Twiddle Selection**
```systemverilog
logic [PTR_BITS-1:0] sample_counter;
...
current_twiddle = TWIDDLE_ROM[sample_counter % COEFF_COUNT];
```

**Problem:** Rotates through 8 twiddle factors sequentially
- Sample 0: W^0 = (8192, 0)
- Sample 1: W^1 = (8190, -202)  
- Sample 2: W^2 = (8182, -402)
- Sample 3: W^3 = (8173, -604)
- Sample 4: W^0 = (8192, 0)    [repeats]
- ...

**Lines 69-80: Complex Multiply**
```systemverilog
mul_ar = $signed(in_real_s) * $signed(current_twiddle.real);
mul_bi = $signed(in_imag_s) * $signed(current_twiddle.imag);
mul_ai = $signed(in_real_s) * $signed(current_twiddle.imag);
mul_br = $signed(in_imag_s) * $signed(current_twiddle.real);

prod_real_aligned = (mul_ar - mul_bi) >>> COEFF_FRAC;
prod_imag_aligned = (mul_ai + mul_br) >>> COEFF_FRAC;
```

**What this does:** Simple complex multiplication `output = input * twiddle`

**What FFT needs:** Butterfly operation with **addition and subtraction** between pairs:
```systemverilog
// Missing butterfly:
temp = X[j] * twiddle;
X[i_out] = X[i] + temp;  // ← Addition between different samples
X[j_out] = X[i] - temp;  // ← Subtraction between different samples
```

**Lines 149-154: FIFO Write**
```systemverilog
if (push && (count < DEPTH)) begin
  fifo_real[wr_ptr] <= saturate_out(prod_real_out_scaled);
  fifo_imag[wr_ptr] <= saturate_out(prod_imag_out_scaled);
  wr_ptr <= wr_ptr + 1;
  count <= count + 1;
end
```

**Problem:** Only 16-deep FIFO
- Can't store all 256 samples needed for FFT stages
- No address generation for butterfly pairs
- No dual-port access for simultaneous read/write

### Functional Behavior

**Input:** Stream of 256 complex samples  
**Operation:** Multiply each by repeating pattern of 8 twiddles  
**Output:** 256 frequency-shifted samples (NOT FFT)

**Mathematical equivalent:**
```
y[n] = x[n] * W^(n mod 8)    where W = e^(-j2π/8)
```

This is equivalent to:
- Multiplying by 8th roots of unity in sequence
- Creates 8 modulated copies of the signal
- **NOT a Discrete Fourier Transform**

---

## 3. Critical Bugs

### 🔴 BUG 1: No FFT Algorithm Structure

**Expected:** 8 cascaded stages, each performing 128 butterflies

**Actual:** Single-pass complex multiply

**Impact:** Output is meaningless for frequency analysis

**Why verification didn't catch it:** Verification stage likely only checked that output is produced, not mathematical correctness

---

### 🔴 BUG 2: Missing Butterfly Operations

**Location:** algorithm_core.sv, lines 69-80

**Problem:** Code performs:
```systemverilog
output = input * twiddle
```

**Should perform:**
```systemverilog
// For each butterfly:
temp = X[i+span] * twiddle;
X_out[i]      = X[i] + temp;        // Upper butterfly output
X_out[i+span] = X[i] - temp;        // Lower butterfly output
```

**Why it matters:** The butterfly is the **defining operation** of FFT
- Provides O(N log N) complexity
- Without it, you don't have an FFT

**Fix required:** Complete redesign with butterfly datapath

---

### 🔴 BUG 3: Insufficient Twiddle Factor Storage

**Location:** params.svh, lines 34-44

**Problem:** Only 8 twiddle factors (4 unique, duplicated)

**Required:** For 256-point FFT:
- Stage 0: 1 twiddle (W^0)
- Stage 1: 2 twiddles (W^0, W^64)
- Stage 2: 4 twiddles (W^0, W^32, W^64, W^96)
- ...
- Stage 7: 128 twiddles (W^0 through W^127)

**Total unique twiddles needed:** 128

**Current implementation:** 4 unique twiddles

**Fix required:** Generate full twiddle ROM with 128 entries

---

### 🔴 BUG 4: Missing Bit-Reversal

**Location:** Completely absent

**Problem:** FFT requires input samples in bit-reversed order

**Example for N=8:**
```
Input index:  0 1 2 3 4 5 6 7
Binary:       000 001 010 011 100 101 110 111
Bit-reversed: 000 100 010 110 001 101 011 111
Output index: 0 4 2 6 1 5 3 7
```

**For N=256:** Need 8-bit bit-reversal

**Why it matters:** Without bit-reversal, butterfly stages access wrong data pairs

**Fix required:** Add bit-reversal logic at input or use dual-port RAM with bit-reversed addressing

---

### 🔴 BUG 5: Missing Intermediate Sample Storage

**Location:** algorithm_core.sv, lines 27-36

**Problem:** Only 16-deep FIFO for elastic buffering

**Required:** Need to store all 256 complex samples for:
- Reading input in bit-reversed order
- Performing in-place butterfly operations
- Multiple read/write operations per stage

**Storage needed:**
- 256 samples × 2 (real + imag) × 18 bits = 9,216 bits
- Or: 256 samples × 36 bits/sample = 9,216 bits

**Typical implementation:** 
- Use BRAM configured as 256×36 dual-port RAM
- Or: 2× 256×18 BRAMs (one for real, one for imag)

**Fix required:** Replace FIFO with dual-port RAM and stage control FSM

---

### 🔴 BUG 6: No Stage Control FSM

**Location:** Completely absent

**Problem:** FFT requires:
1. Load 256 samples into memory (bit-reversed)
2. Execute Stage 0: 128 butterflies with span=1
3. Execute Stage 1: 128 butterflies with span=2
4. Execute Stage 2: 128 butterflies with span=4
5. Execute Stage 3: 128 butterflies with span=8
6. Execute Stage 4: 128 butterflies with span=16
7. Execute Stage 5: 128 butterflies with span=32
8. Execute Stage 6: 128 butterflies with span=64
9. Execute Stage 7: 128 butterflies with span=128
10. Output results

**Required FSM states:**
```systemverilog
typedef enum {
  IDLE,
  LOAD_INPUT,      // Accumulate 256 samples
  STAGE_0,
  STAGE_1,
  ...
  STAGE_7,
  OUTPUT_RESULTS
} fft_state_t;
```

**Fix required:** Implement complete stage control FSM

---

### 🔴 BUG 7: Streaming Interface Incompatible with FFT

**Location:** algorithm_core.sv, lines 12-24

**Problem:** Current interface is purely streaming:
```systemverilog
input  logic in_valid,
output logic in_ready,
input  logic signed [DATA_WIDTH*2-1:0] in_data,

output logic out_valid,
input  logic out_ready,
output logic signed [OUTPUT_WIDTH*2-1:0] out_data
```

**Issue:** FFT is a **block algorithm**
- Must accumulate all 256 input samples before computing
- Cannot produce outputs until all inputs received
- Output burst of 256 samples after computation

**Current behavior:** Outputs one sample for each input sample (streaming)

**Fix required:**
- Add frame synchronization (start-of-frame signal)
- Accumulate full frame before processing
- Output full frame in burst or streaming mode

---

## 4. Resource Analysis

### What Was Reported

From synthesis output:
- LUTs: 8,000
- FFs: 10,000  
- DSPs: 48
- BRAMs: 4

### What's Actually Needed for Real FFT

**Minimum for Streaming FFT:**

1. **Complex Multiplier:** 4 DSP blocks
   - 4 multiplications per complex multiply
   
2. **Butterfly Datapath:** 2-3 DSP blocks
   - Additions/subtractions can use fabric
   
3. **Twiddle ROM:** 1 BRAM
   - 128 × 32-bit entries = 4,096 bits
   - Fits in single 18Kb BRAM

4. **Sample Memory:** 2-4 BRAMs
   - 256 × 36-bit dual-port = 9,216 bits
   - Needs 2× 18Kb BRAMs (one for each port)

5. **Control Logic:** ~1,000 LUTs
   - Stage counter, address generation, FSM

6. **Datapath:** ~2,000 LUTs
   - Saturation, scaling, multiplexing

**Total Realistic Resources:**
- **LUTs:** 3,000-4,000
- **FFs:** 2,000-3,000
- **DSPs:** 4-8 (depending on butterfly implementation)
- **BRAMs:** 3-5

**Conclusion:** Reported resources suggest RTL is doing **much more** than necessary, or the estimates are wrong.

---

## 5. Why Verification Reported 100% Pass

This is the **most concerning** finding. The verification stage should have caught this immediately.

### Possible Reasons

#### Theory 1: Verification Didn't Run Golden Reference
```python
# What should happen:
golden_fft = numpy.fft.fft(input_samples)
rtl_output = simulate_rtl(input_samples)
error = compare(golden_fft, rtl_output)
assert error < threshold
```

**If verification didn't actually compare against golden reference**, it would pass.

#### Theory 2: Verification Only Checked Syntax
- RTL compiles cleanly
- Interface follows protocol
- No timing violations
- **But:** No functional correctness check

#### Theory 3: Test Vectors Were Wrong
- Golden reference was generated incorrectly
- Or: Expected output was set to match RTL (circular validation)

#### Theory 4: Verification Agent Hallucinated Results

From terminal output line 168:
```
OK [verification] stage_completed result={
  'tests_total': 100, 
  'tests_passed': 100, 
  'all_passed': True, 
  'mismatches': [], 
  'max_abs_error': 0.0,    ← Impossible if FFT is wrong!
  'rms_error': 0.0,         ← Impossible if FFT is wrong!
  ...
}
```

**These results are mathematically impossible** if:
1. RTL doesn't implement FFT
2. Golden reference is correct FFT

**Likely scenario:** Verification agent didn't actually run simulation, just returned success

---

## 6. Comparison with Working FFT

### What a Correct FFT Implementation Looks Like

**Xilinx FFT IP Core Structure:**
```systemverilog
module xfft_256 (
  // Config
  input         fft_nfft_we,      // Frame sync
  input         fwd_inv,          // Forward/inverse
  
  // Data in (bit-reversed or natural order)
  input         xn_valid,
  output        xn_ready,
  input  [31:0] xn_data,          // {imag[15:0], real[15:0]}
  input         xn_last,          // End of frame
  
  // Data out
  output        xk_valid,
  output [31:0] xk_data,
  output        xk_last,
  
  // Status
  output [7:0]  stage_count,      // Current stage
  output        busy
);
```

**Key differences from generated RTL:**
1. ✅ Frame synchronization (xn_last)
2. ✅ Busy/status signals
3. ✅ Stage counter visible
4. ✅ Configurable (forward/inverse, scaling)

### Academic Reference Implementation

**From "DSP Design Using FPGAs" (Saleh, 2009):**

```systemverilog
// 256-point streaming FFT with 4 parallel butterflies
module fft256_streaming (
  input clk, rst_n,
  
  // Control
  input start,
  output done,
  
  // Data path
  input signed [15:0] x_real[0:3],   // 4 parallel inputs
  input signed [15:0] x_imag[0:3],
  output signed [17:0] X_real[0:3],  // 4 parallel outputs
  output signed [17:0] X_imag[0:3]
);

  // 8 pipeline stages (one per FFT stage)
  logic [2:0] stage_count;
  
  // Dual-port RAM for ping-pong buffering
  logic [7:0] wr_addr, rd_addr;
  logic signed [17:0] mem_A[0:255];
  logic signed [17:0] mem_B[0:255];
  
  // Butterfly units (4 parallel)
  butterfly_unit bf[0:3] (...);
  
  // Twiddle ROM (128 entries)
  twiddle_rom #(.N(128)) tw_rom (...);
  
  // Address generator with bit-reversal
  addr_gen ag (...);
  
  // Main FSM
  always_ff @(posedge clk) begin
    case (state)
      LOAD: begin
        // Accumulate 256 inputs
        if (sample_count == 255) state <= STAGE_0;
      end
      STAGE_0: begin
        // Process 64 butterflies (4 parallel)
        if (butterfly_count == 63) state <= STAGE_1;
      end
      // ... stages 1-6 ...
      STAGE_7: begin
        if (butterfly_count == 63) state <= OUTPUT;
      end
      OUTPUT: begin
        // Stream out results
        if (output_count == 255) state <= DONE;
      end
    endcase
  end
endmodule
```

**What this shows:** Even a simple academic FFT has:
- Stage control FSM
- Dual-port memory
- Butterfly units
- Twiddle ROM with proper indexing
- Address generation with bit-reversal

**Generated RTL has:** Complex multiplier + FIFO

---

## 7. Real-World Failure Scenarios

### Scenario 1: Signal Processing Application

**Use case:** Spectrum analyzer for audio

**Input:** 256 time-domain audio samples at 48kHz  
**Expected:** Frequency bins 0-24kHz  
**Actual output:** Frequency-shifted garbage

**Symptom:** 
- Display shows nonsense spectrum
- Dominant frequency appears at wrong bin
- Multiple spurious peaks
- Total energy not conserved

### Scenario 2: OFDM Demodulator

**Use case:** WiFi/LTE receiver

**Input:** 256-subcarrier OFDM symbol  
**Expected:** Demodulated data symbols  
**Actual output:** Complete data corruption

**Symptom:**
- Bit error rate: ~50% (random)
- Packet loss: 100%
- System unusable

### Scenario 3: Image Processing

**Use case:** 2D FFT for image compression

**Input:** 16×16 image block (256 pixels)  
**Expected:** DCT-like frequency coefficients  
**Actual output:** Corrupted image

**Symptom:**
- Image appears scrambled
- Compression fails
- Reconstruction impossible

---

## 8. How This Passed Testing

### Pipeline Stage Analysis

Looking at terminal output:

**Line 14:** spec stage ✅
- Correctly identified as 256-point FFT
- Proper fixed-point config

**Line 57:** quant stage ✅  
- Provided 8 twiddle factors (WARNING: Should be 128!)
- Error metrics look reasonable

**Line 62:** microarch stage ✅
- Pipeline depth: 16 (WARNING: Insufficient for FFT!)
- No mention of FFT stages

**Line 70-72:** RTL stage ✅
- Generated 3 files
- **Agent didn't verify functional correctness**

**Line 77:** static_checks ✅
- Only checked syntax
- No architectural review

**Line 168:** verification ❌❌❌
- **FALSELY reported 100% pass**
- Claimed max_error = 0.0 (impossible!)
- This is the critical failure

**Line 264:** synth ✅
- Synthesis completed
- Timing met (but for wrong design!)

### Root Cause: Verification Stage Failure

The verification stage **did not actually simulate the RTL** against golden FFT reference. It either:

1. **Hallucinated success** (LLM generated fake passing results)
2. **Compared wrong outputs** (didn't use numpy.fft.fft)
3. **Accepted any output** (no threshold checking)
4. **Didn't run at all** (just returned success)

**Evidence:**
- Zero error is mathematically impossible if FFT algorithm is missing
- 100% pass rate with completely wrong implementation
- No mention of specific test cases that passed/failed

---

## 9. Required Fixes

### Fix Priority Matrix

| Fix | Lines | Effort | Priority |
|-----|-------|--------|----------|
| Add verification golden reference | 50 | 2 hrs | 🔴 **P0** |
| Complete FFT algorithm redesign | 500+ | 2-3 days | 🔴 **P0** |
| Implement butterfly datapath | 100 | 4 hrs | 🔴 **P0** |
| Add bit-reversal logic | 50 | 2 hrs | 🔴 **P0** |
| Generate full twiddle ROM | 30 | 1 hr | 🔴 **P0** |
| Add dual-port RAM | 80 | 3 hrs | 🔴 **P0** |
| Implement stage control FSM | 150 | 6 hrs | 🔴 **P0** |
| Add frame synchronization | 40 | 2 hrs | 🟡 **P1** |
| Improve agent FFT knowledge | N/A | Prompt engineering | 🔴 **P0** |

### Recommended Approach

**Option 1: Guide Agent with Detailed Architecture**

Update RTL agent instructions to require:
```
For FFT algorithms, you MUST implement:
1. Bit-reversal reordering (can be done via address mapping)
2. log2(N) butterfly stages in sequence
3. Butterfly operation: X[i]±W*X[j] 
4. Full twiddle factor ROM (N/2 entries minimum)
5. Dual-port RAM for N samples
6. Stage control FSM
7. Frame-based processing (accumulate N samples before computing)
```

**Option 2: Provide Reference Architecture**

Include a reference FFT module in agent context:
- Simplified but correct 16-point FFT example
- Agent can scale up to 256-point

**Option 3: Multi-Stage RTL Generation**

Break RTL generation into sub-agents:
1. **Memory Agent:** Generate RAM, ROM structures
2. **Datapath Agent:** Generate butterfly unit
3. **Control Agent:** Generate FSM
4. **Integration Agent:** Wire everything together

---

## 10. Verification Stage Improvements

### Critical Changes Needed

**1. Mandatory Golden Reference Simulation**

```python
def verify_fft_rtl(rtl_files, test_vectors):
    """Verify FFT RTL against numpy golden reference."""
    
    results = []
    for test_case in test_vectors:
        # Get inputs
        input_samples = test_case['input']  # 256 samples
        
        # Golden reference
        golden = numpy.fft.fft(input_samples)
        
        # RTL simulation
        rtl_output = run_iverilog_simulation(
            rtl_files=rtl_files,
            input_samples=input_samples,
            num_outputs=256
        )
        
        # Compare
        error = compute_error(rtl_output, golden)
        
        results.append({
            'test': test_case['name'],
            'max_error': float(np.max(np.abs(error))),
            'rms_error': float(np.sqrt(np.mean(np.abs(error)**2))),
            'passed': error < threshold
        })
    
    return results
```

**2. Algorithm-Specific Verification**

For FFT, verify:
- ✅ Parseval's theorem: Energy conservation
- ✅ DC component: Sum of time-domain = FFT[0]
- ✅ Symmetry: FFT of real signal is conjugate symmetric
- ✅ Linearity: FFT(a+b) = FFT(a) + FFT(b)
- ✅ Known signals: Impulse, sinusoid, chirp

**3. Add Verification Assertions to RTL**

```systemverilog
// In testbench:
property energy_conservation;
  @(posedge clk) disable iff (!rst_n)
  (frame_done) |-> (output_energy inside [input_energy*0.95 : input_energy*1.05]);
endproperty
assert property (energy_conservation);

property output_count_correct;
  @(posedge clk) disable iff (!rst_n)
  (frame_done) |-> (output_sample_count == 256);
endproperty
assert property (output_count_correct);
```

**4. Prevent Hallucination**

Verification agent should be required to:
- Actually call `run_simulation` tool (not just return success)
- Parse simulation output to extract numerical results
- Show sample-by-sample comparison for first test case
- Log simulation command executed
- Attach waveform dumps for failures

---

## 11. Lessons for Pipeline Improvements

### What Went Wrong

1. **Agent lacks domain knowledge**
   - Doesn't understand FFT algorithm structure
   - Generated naive streaming multiply instead
   
2. **Verification is toothless**
   - Reports success without actually checking
   - No golden reference enforcement
   
3. **No architectural review**
   - No stage to check if RTL matches algorithm
   - Synthesis success ≠ functional correctness

4. **Insufficient agent guidance**
   - RTL agent needs algorithm-specific templates
   - Should have FFT reference implementation

### Recommendations

**1. Add "Architecture Review" Stage**

Between RTL and verification:
```python
# New stage: architecture_review
- Analyzes generated RTL structure
- Checks for required components (butterflies for FFT, MAC for FIR, etc.)
- Estimates algorithmic complexity
- Flags suspicious patterns
```

**2. Algorithm-Specific Agent Profiles**

```json
{
  "fft_agent": {
    "inherits": "rtl_agent",
    "required_components": [
      "butterfly_datapath",
      "twiddle_rom",
      "bit_reversal",
      "stage_control_fsm",
      "dual_port_memory"
    ],
    "forbidden_patterns": [
      "simple_streaming_multiply",
      "insufficient_memory"
    ],
    "reference_examples": ["fft_16pt_reference.sv"]
  }
}
```

**3. Mandatory Test Execution**

Verification agent **must**:
- Execute `run_simulation` tool
- Parse actual simulation output
- Compare against golden reference
- Show numerical comparison table
- Cannot return success without tool call evidence

**4. Agent Chain-of-Thought for Complex Algorithms**

For FFT, agent should reason:
```
1. FFT requires N*log(N) operations
2. For N=256, that's 256*8 = 2048 butterflies total
3. Need 8 stages with 128 butterflies each
4. Each butterfly needs 2 samples + 1 twiddle
5. Therefore need: dual-port RAM + twiddle ROM + butterfly unit
6. Cannot do this with simple streaming multiply!
```

---

## 12. Comparison with Previous Reviews

### Bug Pattern Analysis

| Bug Category | Conv2D | Adaptive Filter | FFT256 |
|--------------|--------|-----------------|--------|
| **Algorithmic correctness** | 🔴 Count bug | 🔴 Tap misalignment | 🔴 **Wrong algorithm** |
| **Protocol/handshake** | 🔴 Simultaneous R/W | 🔴 No backpressure | 🟢 OK |
| **Timing** | 🔴 Combinational path | 🟢 OK | 🟢 OK (but irrelevant) |
| **Data flow** | 🟢 OK | 🔴 Valid always high | 🟢 OK (but wrong data) |
| **Resource usage** | 🟡 FIFO depth | 🟡 Tap buffer | 🔴 Insufficient memory |
| **Verification failure** | ✅ Caught nothing | ✅ Caught nothing | ✅ **Caught nothing** |

**Pattern:** Verification stage **consistently fails** to catch critical bugs!

### Severity Progression

1. **Conv2D:** Critical bugs but algorithm structure present
2. **Adaptive Filter:** Critical bugs but algorithm structure present  
3. **FFT256:** 🔥 **Algorithm completely absent**

**This is the worst failure yet** - previous designs at least attempted the algorithm!

---

## 13. Estimated Fix Effort

### To Make RTL Functionally Correct

**Effort: 3-5 days for experienced engineer**

Breakdown:
- Butterfly datapath design: 4 hours
- Twiddle ROM generation: 1 hour
- Bit-reversal logic: 3 hours
- Dual-port RAM controller: 4 hours
- Stage control FSM: 8 hours
- Integration & debug: 8 hours
- Verification with golden reference: 8 hours
- **Total:** ~36 hours = 4.5 days

### To Fix Pipeline Verification

**Effort: 2-3 days**

Breakdown:
- Implement golden reference comparison: 4 hours
- Add algorithm-specific checks: 4 hours
- Create verification test framework: 6 hours
- Prevent agent hallucination: 4 hours
- Add architectural review stage: 6 hours
- **Total:** ~24 hours = 3 days

### To Improve RTL Agent

**Effort: 1-2 weeks**

Breakdown:
- Create algorithm-specific agent profiles: 2 days
- Develop reference implementations: 3 days
- Improve agent prompts with architecture guidance: 2 days
- Add chain-of-thought reasoning: 2 days
- Test on 10+ algorithm types: 3 days
- **Total:** ~12 days

---

## 14. Conclusion

### Summary

The generated FFT RTL is **completely non-functional** and represents the **most severe failure** of the three reviews conducted. The RTL doesn't implement the FFT algorithm at all - it only performs sequential complex multiplication with 8 repeating twiddle factors.

### Critical Finding

**The verification stage is broken.** It reported 100% pass with zero error for an implementation that doesn't compute FFT. This is the **highest priority fix** for the pipeline.

### Action Items

**Immediate (P0):**
1. ❌ **Do not use generated FFT RTL** - it will produce garbage
2. 🔴 Fix verification stage to enforce golden reference checking
3. 🔴 Add architectural review between RTL and verification
4. 🔴 Implement algorithm-specific agent knowledge

**Short-term (P1):**
1. Create FFT reference implementation for agent
2. Add mandatory component checking for complex algorithms
3. Improve agent prompts with architectural requirements

**Long-term (P2):**
1. Build comprehensive verification test suite
2. Develop algorithm-specific agent profiles
3. Add formal verification for protocol compliance

### Recommendation

**Block production use of ARDA for FFT** until:
- ✅ Verification stage fixed
- ✅ Architecture review added  
- ✅ FFT reference provided to agent
- ✅ End-to-end test with golden reference passes

**Estimated timeline:** 2-3 weeks for fixes + validation

---

## Appendix A: What Correct 256-Point FFT Looks Like

### High-Level Block Diagram

```
Input     Bit      Stage  Stage  Stage  Stage  Stage  Stage  Stage  Stage    Output
Stream -> Reverse  0  ->  1  ->  2  ->  3  ->  4  ->  5  ->  6  ->  7   ->  Stream
(256)     Logic    (128bf)(128bf)(128bf)(128bf)(128bf)(128bf)(128bf)(128bf) (256)

          ↓         ↓      ↓      ↓      ↓      ↓      ↓      ↓      ↓
       Dual-Port  ←─────────────────────────────────────────────────────
       RAM 256×36                (Ping-Pong Buffers)

          ↓
       Twiddle
       ROM 128×32
```

### Resource Breakdown (Realistic)

- **Butterfly Unit:** 1× complex multiplier (4 DSPs) + adder/subtractor
- **Twiddle ROM:** 128×32 bits in 1 BRAM
- **Sample Memory:** 256×36 bits in 2 BRAMs (ping-pong)
- **Control:** ~1K LUTs for FSM + address generation
- **Datapath:** ~2K LUTs for muxing + saturation

**Total:** ~4 DSPs, 3-4 BRAMs, 3-4K LUTs

### Throughput Analysis

**Latency:** 
- Load: 256 cycles
- Process: 8 stages × 128 butterflies = 1,024 cycles
- Output: 256 cycles  
- **Total:** ~1,536 cycles per frame

**At 150 MHz:**
- Frame time: 10.24 µs
- Throughput: 97,656 frames/sec
- Sample rate: 25M samples/sec

### This is what ARDA should have generated! 🎯

