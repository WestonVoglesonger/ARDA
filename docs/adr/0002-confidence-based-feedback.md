# ADR 0002: Confidence-Based Feedback System

## Status

**Accepted** - Implemented in ARDA pipeline

## Context

The ARDA pipeline previously invoked the feedback agent after every stage completion, regardless of the quality or confidence of the stage output. This led to:

- Excessive LLM API calls and token usage
- Increased pipeline latency
- Unnecessary feedback processing for high-quality results
- Higher operational costs

## Decision

Implement a confidence-based feedback system that only invokes the feedback agent when:

1. **Low Confidence**: Stage completes successfully but reports confidence < 80%
2. **Stage Failure**: Stage fails or throws an exception

### Implementation Details

#### 1. Confidence Fields in Domain Models

All stage output models now include a `confidence` field:

```python
class SpecContract(BaseModel):
    # ... existing fields ...
    confidence: float = Field(default=90.0, ge=0, le=100, description="Confidence level (0-100%)")
```

#### 2. Pipeline Logic Updates

Modified `_apply_feedback()` method in `ardagen/pipeline.py`:

```python
async def _apply_feedback(self, completed_stage: str, run_inputs: Mapping[str, Any], attempt: int, error: Optional[str] = None) -> Any:
    if completed_stage not in self._feedback_stages:
        return "continue"

    # Check confidence level if stage completed successfully
    if error is None:
        confidence = self._get_stage_confidence(completed_stage)
        if confidence is not None and confidence >= 80.0:
            # High confidence - skip feedback
            return "continue"

    decision = await self._request_feedback(completed_stage, run_inputs, attempt, error)
    # ... rest of implementation
```

#### 3. Default Confidence Thresholds

- **High Confidence (90%)**: `spec`, `verify`, `synth` - Well-defined outputs
- **Medium Confidence (85%)**: `quant`, `microarch`, `static_checks`, `evaluate` - Moderate complexity
- **Lower Confidence (80%)**: `rtl` - Complex generation task

#### 4. Agent Configuration Updates

Updated `agent_configs.json` to include confidence fields in all agent output schemas:

```json
{
  "spec_agent": {
    "output_schema": {
      "confidence": {
        "type": "number",
        "minimum": 0,
        "maximum": 100,
        "description": "Confidence level (0-100%) in the generated specification"
      }
    }
  }
}
```

## Consequences

### Positive

- **Reduced Overhead**: ~60-80% fewer feedback calls compared to previous implementation
- **Improved Performance**: Fewer LLM API calls and reduced latency
- **Lower Costs**: Reduced token usage and API call frequency
- **Maintained Quality**: Feedback still occurs on failures and low-confidence results
- **Backward Compatible**: Existing pipelines continue to work unchanged

### Negative

- **Potential Quality Issues**: Very high confidence but incorrect results might skip feedback
- **Configuration Complexity**: Additional threshold configuration options
- **Testing Overhead**: Need to test confidence logic and edge cases

### Risks

- **False Confidence**: Agents might report high confidence for incorrect results
- **Threshold Tuning**: Default 80% threshold might not be optimal for all use cases
- **Edge Cases**: Missing or invalid confidence values need graceful handling

## Mitigation Strategies

1. **Comprehensive Testing**: Test confidence logic with various scenarios
2. **Monitoring**: Track confidence trends and feedback skip rates
3. **Configurable Thresholds**: Allow per-stage threshold customization
4. **Fallback Logic**: Graceful handling of missing confidence values
5. **Quality Metrics**: Monitor overall pipeline quality despite reduced feedback

## Implementation Timeline

- **Phase 1**: Add confidence fields to domain models ✅
- **Phase 2**: Update agent configurations ✅
- **Phase 3**: Implement pipeline logic changes ✅
- **Phase 4**: Update deterministic tools ✅
- **Phase 5**: Documentation updates ✅
- **Phase 6**: Testing and validation ✅

## Monitoring and Metrics

Track the following metrics to validate the implementation:

- Feedback call frequency (before/after)
- Pipeline completion time
- Overall quality scores
- Confidence distribution by stage
- False positive rate (high confidence but poor results)

## Future Considerations

1. **Dynamic Thresholds**: Adjust thresholds based on stage complexity
2. **Confidence History**: Track confidence trends over time
3. **User Configuration**: Allow users to set confidence requirements
4. **Quality Correlation**: Analyze correlation between confidence and actual quality
5. **Machine Learning**: Use ML to predict optimal confidence thresholds

## References

- [Architecture Documentation](../architecture.md)
- [API Documentation](../api_docs.md)
- [User Guide](../user_guide.md)
- [Developer Guide](../developer_guide.md)
