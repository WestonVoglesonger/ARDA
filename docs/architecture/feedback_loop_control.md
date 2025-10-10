# Feedback Loop Control and Retry Limits

**Date:** October 10, 2025  
**Issue Type:** Critical Pipeline Reliability  
**Status:** 🔴 Active Problem - Requires Immediate Fix

---

## Executive Summary

The ARDA pipeline contains a **critical flaw** in its feedback mechanism that allows **infinite retry loops**. During a Conv2D test run, the quantization stage failed repeatedly (7+ attempts) with the feedback agent continuously requesting retries without convergence detection or exit conditions. The user had to manually terminate the process with Ctrl-C.

**Impact:** Pipeline can hang indefinitely, wasting compute resources and API credits.

**Priority:** 🔴 **P0** - Must fix before production use

---

## Problem Description

### What Happened

**Test Case:** Conv2D algorithm (test_algorithms/conv2d_bundle.txt)  
**Stage:** Quantization (quant)  
**Duration:** 7+ retry attempts before manual termination  
**Result:** Infinite loop requiring keyboard interrupt

### Retry Sequence

| Attempt | Result | Feedback Decision |
|---------|--------|-------------------|
| 1 | `max_abs_error: 0.8` (exceeds 0.1 tolerance) | `retry_quant` |
| 2 | `quantized_coefficients: [0.0]` (suspicious single zero) | `retry_quant` |
| 3 | **Validation error** (returned schema instead of data) | `retry_quant` |
| 4 | `quantized_coefficients: [0.0, 0.0, ...]` (all zeros) | `retry_quant` |
| 5 | `quantized_coefficients: []` (empty array) | `retry_quant` |
| 6 | `quantized_coefficients: []` (empty array, no change) | `retry_quant` |
| 7 | `quantized_coefficients: [0.0]` (back to single zero) | `retry_quant` |
| 8+ | **User terminated with Ctrl-C** | N/A |

### Evidence from Terminal Output

**Attempt 3 (Lines 151-163):**
```python
FAIL [quant] stage_failed error=4 validation errors for QuantConfig
fixed_point_config: Field required
error_metrics: Field required
quantized_coefficients: Field required
fxp_model_path: Field required
```

**Critical observation:** The agent returned the Pydantic schema definition instead of actual data, indicating severe confusion.

**Attempt 5 vs Attempt 6:**
Both returned empty `quantized_coefficients: []`, but feedback agent didn't recognize the pattern and just kept saying "retry".

**Feedback at Attempt 7 (Lines 303-304):**
```json
{
  "action": "retry_quant",
  "guidance": "Retry the quant pass with explicit output of quantized coefficients..."
}
```

No indication that this was the 7th attempt or that previous retries weren't helping.

---

## Root Causes

### 1. No Retry Limit Enforcement 🔴

**Current behavior:**
```python
# pipeline.py - simplified conceptual flow
while True:
    result = await stage.run()
    feedback = await self._apply_feedback(stage, result)
    
    if feedback == "retry":
        continue  # ← No limit!
    elif feedback == "continue":
        break
    elif feedback == "abort":
        return failure
```

**Problem:** Infinite loop possible if feedback always returns "retry"

**Location:** `ardagen/pipeline.py`, lines ~120-210

---

### 2. Feedback Agent Lacks Pattern Recognition 🔴

**What feedback agent should detect:**

1. **Repeated identical failures:**
   - Attempts 5 and 6 both had empty arrays
   - Should recognize: "Same error twice → agent is stuck"

2. **Thrashing between error modes:**
   - Attempt 1: High error
   - Attempt 2: Suspicious zeros
   - Attempt 3: Schema error
   - Attempt 4: All zeros
   - Attempt 5: Empty array
   - Should recognize: "Different error each time → agent is confused"

3. **Degrading quality:**
   - Attempt 2: `[0.0]` (1 coefficient)
   - Attempt 4: `[0.0, 0.0, ...]` (8 coefficients, all zero)
   - Attempt 5: `[]` (0 coefficients)
   - Should recognize: "Quality not improving → strategy failing"

**Current behavior:** Feedback agent only looks at current attempt, not history.

**Location:** Feedback agent instructions in `agent_configs.json`, lines ~941-982

---

### 3. Quant Agent Hallucinating/Unstable 🔴

**Symptom:** Each retry produces a different type of failure:

1. Numerical error (tolerance violation)
2. Degenerate output (zeros only)
3. Schema confusion (returned structure instead of data)
4. Empty output
5. Random variation in same failure modes

**Why this happens:**

1. **Non-deterministic LLM behavior:**
   - Each retry is a fresh API call
   - No seed or temperature control
   - Agent doesn't learn from previous failures

2. **Insufficient context:**
   - Feedback messages aren't detailed enough
   - Agent doesn't see previous attempt results
   - No examples of successful quantization

3. **Task complexity:**
   - Quantization requires domain knowledge
   - Agent might not understand fixed-point arithmetic
   - Hallucinating plausible-looking but wrong outputs

**Location:** 
- Quant agent instructions: `agent_configs.json`, lines ~198-270
- Quant agent runner: `ardagen/agents/openai_runner.py`

---

### 4. No Convergence Detection 🔴

**Missing checks:**

1. **Same error twice in a row:** Should abort or escalate
2. **Error oscillation:** If error bounces up and down, strategy isn't working
3. **Attempt budget exhausted:** Hard limit on retries per stage
4. **Time limit:** Don't spend >X minutes on one stage
5. **API cost limit:** Don't burn $Y on retries

**Current state:** None of these checks exist

---

## Impact Assessment

### Immediate Impacts

| Impact | Severity | Description |
|--------|----------|-------------|
| **User Experience** | 🔴 Critical | User must manually kill process |
| **Resource Waste** | 🔴 Critical | Unnecessary API calls and compute time |
| **Cost** | 🟡 High | Each retry costs money (OpenAI API) |
| **Development Velocity** | 🟡 High | Can't iterate quickly if runs hang |
| **Reliability** | 🔴 Critical | Pipeline is unreliable for production |

### Affected Stages

**Any stage can hit this issue:**

- ✅ **Confirmed:** quant stage (Conv2D run)
- 🟡 **Possible:** spec, microarch, rtl, verification, synth stages
- 🟡 **Possible:** Any stage that uses feedback-driven retries

**Likelihood:** Medium-High
- Feedback agent is aggressive (prefers retry over continue)
- Agents are non-deterministic
- Complex tasks more likely to fail

---

## Proposed Solutions

### Solution 1: Hard Retry Limits (P0 - Immediate)

**Implementation:**

```python
# ardagen/pipeline.py

MAX_RETRIES_PER_STAGE = 3  # Configurable constant

async def run(self, algorithm_bundle: str) -> Dict[str, Any]:
    # ... existing setup ...
    
    for stage in stages:
        attempts = 0
        max_attempts = MAX_RETRIES_PER_STAGE
        
        while attempts < max_attempts:
            try:
                result = await stage.run()
                attempts += 1
                
                feedback = await self._apply_feedback(stage, result, attempts)
                
                if feedback.action == "continue":
                    break
                elif feedback.action == "abort":
                    return self._build_error_response("Aborted by feedback")
                elif feedback.action.startswith("retry_"):
                    if attempts >= max_attempts:
                        # Force abort on max attempts
                        return self._build_error_response(
                            f"Stage {stage.name} failed after {max_attempts} attempts"
                        )
                    continue
            except Exception as e:
                attempts += 1
                if attempts >= max_attempts:
                    return self._build_error_response(str(e))
```

**Benefits:**
- ✅ Prevents infinite loops
- ✅ Simple to implement (1 hour)
- ✅ No API changes needed
- ✅ Configurable per-stage if needed

**Tradeoffs:**
- ⚠️ Might abort on genuinely recoverable failures
- ⚠️ Hard limit might be too low for some algorithms

**Effort:** 1-2 hours  
**Risk:** Low  
**Priority:** 🔴 **P0**

---

### Solution 2: Convergence Detection (P1)

**Implementation:**

```python
class RetryHistory:
    """Track retry attempts for convergence detection."""
    
    def __init__(self):
        self.attempts = []
        self.error_hashes = []
    
    def add_attempt(self, result: Dict[str, Any]):
        """Record an attempt result."""
        self.attempts.append(result)
        
        # Hash key fields to detect identical failures
        key_fields = ("error", "coefficients", "validation_errors")
        error_hash = hash(tuple(result.get(k) for k in key_fields))
        self.error_hashes.append(error_hash)
    
    def is_stuck(self) -> bool:
        """Detect if retries aren't making progress."""
        if len(self.attempts) < 2:
            return False
        
        # Check for identical last two attempts
        if len(self.error_hashes) >= 2:
            if self.error_hashes[-1] == self.error_hashes[-2]:
                return True  # Same error twice
        
        # Check for oscillation
        if len(self.error_hashes) >= 3:
            if self.error_hashes[-1] == self.error_hashes[-3]:
                return True  # Error pattern repeating
        
        return False
    
    def should_abort(self) -> tuple[bool, str]:
        """Determine if retry should be aborted."""
        if self.is_stuck():
            return True, "No progress after multiple retries (same error repeating)"
        
        if len(self.attempts) >= MAX_RETRIES_PER_STAGE:
            return True, f"Maximum retry limit ({MAX_RETRIES_PER_STAGE}) reached"
        
        return False, ""
```

**Usage:**

```python
retry_history = RetryHistory()

for attempt in range(MAX_RETRIES_PER_STAGE):
    result = await stage.run()
    retry_history.add_attempt(result)
    
    should_abort, reason = retry_history.should_abort()
    if should_abort:
        return self._build_error_response(reason)
    
    feedback = await self._apply_feedback(stage, result, retry_history)
```

**Benefits:**
- ✅ Smarter than hard limits
- ✅ Detects oscillation and thrashing
- ✅ Provides informative abort reasons

**Effort:** 3-4 hours  
**Risk:** Low  
**Priority:** 🟡 **P1**

---

### Solution 3: Enhanced Feedback Agent (P1)

**Update feedback agent instructions to include history:**

```json
{
  "feedback": {
    "instructions": "...\n\nWhen evaluating retry decisions:\n
      1. Check if the same error occurred in the last attempt (attempt_history provided)\n
      2. If errors are oscillating (A -> B -> A), recommend abort\n
      3. If error severity is increasing, recommend abort\n
      4. Consider the attempt number - after 3 attempts, strongly prefer abort or continue\n
      5. Provide detailed guidance on WHY retry might succeed\n
      6. If you recommend retry, suggest specific changes to the approach\n
      ...",
    "output_schema": {
      "action": "...",
      "confidence": "number",  // NEW: 0-100 confidence that retry will succeed
      "attempt_analysis": "string",  // NEW: Why retry might/might not work
      "specific_guidance": "string"  // NEW: What should change in retry
    }
  }
}
```

**Provide attempt history in context:**

```python
feedback_context = self._build_feedback_context(
    stage_name=stage_name,
    result=result,
    attempt=attempt,
    previous_attempts=self.stage_history.get(stage_name, [])  # NEW
)
```

**Benefits:**
- ✅ Feedback agent makes informed decisions
- ✅ Can detect patterns (same error, oscillation)
- ✅ More actionable guidance

**Tradeoffs:**
- ⚠️ More tokens per feedback call (higher cost)
- ⚠️ Feedback agent might still fail to detect patterns

**Effort:** 2-3 hours  
**Risk:** Medium  
**Priority:** 🟡 **P1**

---

### Solution 4: Exponential Backoff (P2)

**For transient failures (API errors, timeouts):**

```python
import time

backoff_delay = 1.0  # Start with 1 second

for attempt in range(MAX_RETRIES_PER_STAGE):
    try:
        result = await stage.run()
        break  # Success
    except TransientError as e:
        if attempt < MAX_RETRIES_PER_STAGE - 1:
            time.sleep(backoff_delay)
            backoff_delay *= 2  # Exponential: 1s, 2s, 4s, 8s...
        else:
            raise
```

**Benefits:**
- ✅ Handles transient failures gracefully
- ✅ Reduces API rate limit issues

**Effort:** 1 hour  
**Risk:** Low  
**Priority:** 🟢 **P2**

---

### Solution 5: Per-Stage Retry Configuration (P2)

**Allow different retry limits per stage:**

```python
STAGE_RETRY_LIMITS = {
    "spec": 2,        # Spec usually works first try
    "quant": 5,       # Quantization might need tuning
    "microarch": 3,
    "rtl": 3,
    "verification": 2,
    "synth": 1,       # Synthesis deterministic
}

max_retries = STAGE_RETRY_LIMITS.get(stage.name, DEFAULT_MAX_RETRIES)
```

**Benefits:**
- ✅ Flexibility for different stage characteristics
- ✅ Can tune based on observed failure rates

**Effort:** 30 minutes  
**Risk:** Low  
**Priority:** 🟢 **P2**

---

## Recommended Implementation Plan

### Phase 1: Emergency Fix (Today - 2 hours)

**Goal:** Stop infinite loops immediately

1. ✅ Implement hard retry limits (Solution 1)
   - Add `MAX_RETRIES_PER_STAGE = 3` constant
   - Add attempt counter to stage execution
   - Force abort on max attempts
   - Test with Conv2D to verify it aborts

2. ✅ Add attempt count to feedback context
   - Pass `attempt` number to feedback agent
   - Update feedback instructions to consider attempt count

**Validation:**
```bash
# Should abort after 3 attempts instead of hanging
python -m ardagen.cli test_algorithms/conv2d_bundle.txt --verbose

# Should complete or abort in reasonable time
timeout 5m python -m ardagen.cli test_algorithms/fft256_bundle.txt
```

---

### Phase 2: Smart Detection (This Week - 4 hours)

**Goal:** Detect and handle stuck retries intelligently

1. ✅ Implement convergence detection (Solution 2)
   - Create `RetryHistory` class
   - Track attempt results
   - Detect same error twice
   - Detect oscillation patterns

2. ✅ Enhance feedback agent context (Solution 3)
   - Include previous attempt results
   - Add confidence scoring
   - Require specific guidance for retries

**Validation:**
```python
# Unit test for convergence detection
def test_retry_history_detects_stuck():
    history = RetryHistory()
    history.add_attempt({"error": "E1"})
    history.add_attempt({"error": "E2"})
    history.add_attempt({"error": "E1"})  # Oscillation
    assert history.is_stuck()

def test_retry_history_detects_repeat():
    history = RetryHistory()
    history.add_attempt({"error": "E1"})
    history.add_attempt({"error": "E1"})  # Duplicate
    assert history.is_stuck()
```

---

### Phase 3: Polish (Next Week - 3 hours)

**Goal:** Fine-tune retry behavior

1. ✅ Add exponential backoff (Solution 4)
2. ✅ Implement per-stage configuration (Solution 5)
3. ✅ Add retry metrics to observability
4. ✅ Add warnings at 50% and 75% of retry budget

**Validation:**
- Run on 10+ algorithms
- Monitor retry rates per stage
- Tune limits based on observed patterns

---

## Success Metrics

### Before Fix (Current State)

- ❌ Pipeline can hang indefinitely
- ❌ No limit on retries
- ❌ User must manually terminate
- ❌ Wasted API calls and compute
- ❌ No pattern detection

### After Phase 1 (Emergency Fix)

- ✅ Pipeline aborts after N attempts
- ✅ No infinite loops possible
- ✅ Clear error messages on abort
- ⚠️ Might abort prematurely

### After Phase 2 (Smart Detection)

- ✅ Detects repeated failures
- ✅ Detects oscillation
- ✅ Feedback agent sees history
- ✅ Fewer unnecessary retries

### After Phase 3 (Polish)

- ✅ Optimized retry limits per stage
- ✅ Exponential backoff for transients
- ✅ Retry metrics in observability
- ✅ Production-ready reliability

---

## Testing Strategy

### Unit Tests

```python
# tests/test_retry_limits.py

def test_hard_retry_limit_enforced():
    """Verify pipeline aborts after max retries."""
    pipeline = Pipeline()
    
    # Mock stage that always fails
    with patch('stage.run', side_effect=Exception("Fail")):
        result = await pipeline.run(bundle)
    
    assert result['success'] == False
    assert 'after 3 attempts' in result['error']

def test_retry_history_convergence():
    """Verify convergence detection works."""
    history = RetryHistory()
    
    # Add repeated failures
    history.add_attempt({"error": "E1"})
    history.add_attempt({"error": "E1"})
    
    should_abort, reason = history.should_abort()
    assert should_abort
    assert "same error" in reason.lower()
```

### Integration Tests

```python
# tests/test_feedback_loops.py

@pytest.mark.slow
def test_conv2d_doesnt_hang():
    """Verify Conv2D completes or aborts (not hang)."""
    import signal
    
    # Set 10-minute timeout
    signal.alarm(600)
    
    try:
        result = await pipeline.run("test_algorithms/conv2d_bundle.txt")
        # Should either succeed or fail, but not hang
        assert 'success' in result
    finally:
        signal.alarm(0)
```

### Manual Tests

1. **Intentionally broken algorithm:**
   - Create algorithm with invalid quantization config
   - Verify pipeline aborts after 3 attempts
   - Check error message is informative

2. **Transient failure simulation:**
   - Mock API failures
   - Verify exponential backoff works
   - Verify eventually succeeds or aborts

3. **Resource monitoring:**
   - Monitor API call count during retries
   - Verify not making excessive calls
   - Check token usage stays reasonable

---

## Migration Notes

### Breaking Changes

**None** - This is purely additive:
- Existing behavior unchanged if retries succeed
- Only new behavior is aborting after N attempts (vs hanging forever)

### Backward Compatibility

✅ **Fully backward compatible:**
- Hard limit of 3 is generous (most stages succeed in 1-2 attempts)
- Can increase limit if needed
- No API changes
- No config changes required

### Rollout Strategy

1. **Deploy to development first**
   - Test with known problematic algorithms
   - Monitor retry rates
   - Tune limits if needed

2. **Gradual limit increase**
   - Start with limit=3 for most stages
   - Increase to 5 if too aggressive
   - Collect metrics to inform final values

3. **Production deployment**
   - Deploy with monitoring
   - Alert on high retry rates
   - Quick rollback plan if issues

---

## Related Issues

### Connection to Other Problems

1. **Quant agent instability** (seen in Conv2D run)
   - Agent returned schema instead of data (attempt 3)
   - Needs better prompting or examples
   - Related: Agent domain knowledge improvement

2. **Verification stage weakness** (seen in FFT256, Conv2D, Adaptive Filter)
   - Verification reports success despite bugs
   - Allows bad results to pass through
   - Related: `docs/architecture/pipeline_verification_improvements.md`

3. **Flexible RTL architecture** (planned next step)
   - Won't fix retry loops
   - But might reduce RTL stage retries
   - Related: `docs/architecture/flexible_rtl_architecture.md`

### Priority Relative to Other Work

| Work Item | Priority | Blocks What | Effort |
|-----------|----------|-------------|--------|
| **Retry limits (This)** | 🔴 P0 | Everything | 2-4 hours |
| Flexible RTL architecture | 🟡 P1 | RTL quality | 1-2 days |
| Verification improvements | 🟡 P1 | Bug detection | 1-2 weeks |
| Agent domain knowledge | 🟢 P2 | Algorithm quality | Ongoing |

**Reasoning:** Retry limits are quick to implement and prevent pipeline hangs. Should be done first or in parallel with flexible RTL.

---

## Appendix A: Full Terminal Output Analysis

### Attempt Progression

**Attempt 1 (Line 14-17):**
```
spec: confidence 75%
feedback: "continue"
quant attempt 1: max_abs_error 0.8
feedback: "retry_quant" (error exceeds tolerance)
```
✅ **Reasonable:** Retry makes sense

**Attempt 2 (Line 59-62):**
```
quant: quantized_coefficients [0.0], error 0.08
feedback: "retry_quant" (suspicious single coefficient)
```
⚠️ **Questionable:** Error within tolerance but coefficients wrong

**Attempt 3 (Line 151-165):**
```
quant: Validation error - returned schema instead of data
feedback: "retry_quant"
```
🔴 **Bad:** Agent is hallucinating, should abort

**Attempts 4-7 (Lines 208-347):**
```
All zeros → retry
Empty array → retry  
Empty array again → retry
Single zero → retry
```
🔴 **Terrible:** Clear infinite loop, no progress

### Cost Analysis

**Assuming:**
- Each attempt = 1 agent call (quant) + 1 agent call (feedback)
- Average 2000 tokens per call
- GPT-4 pricing: $0.03/1K input tokens, $0.06/1K output tokens
- Estimate: $0.10 per attempt (conservative)

**For this Conv2D run:**
- 7 attempts × $0.10 = **$0.70 wasted**
- Plus initial spec/feedback: **~$0.90 total**
- **Pipeline never completed** (would have continued indefinitely)

**At scale:**
- 100 algorithms × 3 retries each = **$30/run**
- If pipeline hangs 50% of time: **$60/run** (doubled due to aborts)

---

## Appendix B: Recommended Config Values

### Conservative (Default)

```python
MAX_RETRIES_PER_STAGE = 3

STAGE_RETRY_LIMITS = {
    "spec": 2,
    "quant": 3,
    "microarch": 3,
    "rtl": 3,
    "verification": 2,
    "synth": 1,
}

RETRY_BACKOFF_SECONDS = 1.0
RETRY_BACKOFF_MULTIPLIER = 2.0
RETRY_TIMEOUT_MINUTES = 10
```

### Aggressive (Testing)

```python
MAX_RETRIES_PER_STAGE = 5

STAGE_RETRY_LIMITS = {
    "spec": 3,
    "quant": 5,
    "microarch": 5,
    "rtl": 5,
    "verification": 3,
    "synth": 2,
}

RETRY_BACKOFF_SECONDS = 0.5
RETRY_BACKOFF_MULTIPLIER = 1.5
RETRY_TIMEOUT_MINUTES = 20
```

### Production (Recommended after tuning)

```python
MAX_RETRIES_PER_STAGE = 3

STAGE_RETRY_LIMITS = {
    "spec": 2,        # Spec rarely needs retries
    "quant": 4,       # Quantization often needs tuning
    "microarch": 3,   # Moderate complexity
    "rtl": 3,         # Complex but should work
    "verification": 2, # Either works or doesn't
    "synth": 1,       # Deterministic tool
}

RETRY_BACKOFF_SECONDS = 2.0
RETRY_BACKOFF_MULTIPLIER = 2.0
RETRY_TIMEOUT_MINUTES = 15
```

---

## Summary

**Problem:** Pipeline can hang indefinitely in retry loops  
**Root Cause:** No retry limits, aggressive feedback, hallucinating agents  
**Impact:** Critical reliability issue, wasted resources  
**Solution:** Hard limits + convergence detection + smarter feedback  
**Effort:** 2-4 hours for emergency fix, 7 hours total for complete solution  
**Priority:** 🔴 **P0** - Fix immediately  

**Recommendation:** Implement Phase 1 (hard limits) **today** before proceeding with flexible RTL architecture work.

