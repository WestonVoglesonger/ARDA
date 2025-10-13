# OpenAI Stage Testing Guide

This guide explains how to set up and run the OpenAI stage testing system for the ARDA pipeline.

## Overview

The OpenAI stage testing system provides comprehensive testing for all 8 ARDA pipeline stages using real OpenAI API calls. Tests are independently executable with fixture support, detailed logging, and interactive retry logic on failures.

## Setup

### 1. Environment Configuration

1. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```

2. Add your OpenAI API key to `.env`:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ARDA_TEST_MODE=live
   ARDA_TEST_LOG_DIR=tests/logs
   ```

3. Install dependencies:
   ```bash
   pip install -e .
   ```

### 2. Test Modes

The system supports three test modes:

- **`live`**: Uses real OpenAI API calls (default)
- **`fixture`**: Uses pre-generated test data only
- **`hybrid`**: Uses fixtures for dependencies, live calls for the stage being tested

Set `ARDA_TEST_MODE` in your `.env` file to control this behavior.

## Running Tests

### All Stages (Costly!)

Run all OpenAI stage tests:
```bash
pytest tests/test_openai_stages.py
```

### Single Stage

Run tests for a specific stage:
```bash
# By test name
pytest tests/test_openai_stages.py -k "test_spec_stage"

# By marker
pytest -m openai_spec
pytest -m openai_quant
pytest -m openai_architecture
```

### Single Algorithm

Run tests for a specific algorithm:
```bash
# By test name
pytest tests/test_openai_stages.py -k "conv2d"

# By marker
pytest -m openai_conv2d
```

### Selective Execution

Combine markers for precise control:
```bash
# Only spec stage for Conv2D
pytest -m "openai_spec and openai_conv2d"

# All Conv2D tests except synthesis
pytest -m "openai_conv2d and not openai_synth"

# Skip expensive stages
pytest -m "openai and not openai_rtl and not openai_synth"
```

### Dry Run (Use Fixtures)

Run tests without API calls:
```bash
ARDA_TEST_MODE=fixture pytest tests/test_openai_stages.py
```

## Test Output

### Logs

- **Location**: `tests/logs/{stage}/{status}/`
- **Format**: Each test creates `{algorithm}_{timestamp}.json`
- **Content**: Inputs, outputs, reasoning, tokens, errors, duration
- **Organization**: Logs are organized by stage and pass/fail status:
  ```
  tests/logs/
  ├── spec/
  │   ├── passed/
  │   │   └── conv2d_20251012_223716.json
  │   └── failed/
  │       └── conv2d_20251012_223800.json
  ├── quant/
  │   ├── passed/
  │   └── failed/
  ├── microarch/
  │   ├── passed/
  │   └── failed/
  └── ... (other stages)
  ```

### Log Viewer Utility

Use the included log viewer to easily navigate and summarize test results:

```bash
# Show summary of all logs
python tests/utils/log_viewer.py

# Show details for a specific log file
python tests/utils/log_viewer.py --show tests/logs/spec/passed/conv2d_20251012_223716.json

# Filter by stage
python tests/utils/log_viewer.py --stage spec

# Filter by status
python tests/utils/log_viewer.py --status failed
```

### Log Structure

Each test execution saves comprehensive data:

```json
{
  "test_name": "test_spec_stage_conv2d",
  "algorithm": "conv2d",
  "stage": "spec",
  "timestamp": "2025-10-13T10:30:00",
  "status": "passed",
  "inputs": {
    "bundle": "...",
    "context": {...}
  },
  "outputs": {
    "name": "Conv2D",
    "clock_mhz_target": 200.0,
    ...
  },
  "reasoning": "Agent's reasoning if captured",
  "token_usage": {
    "prompt_tokens": 1523,
    "completion_tokens": 342,
    "total_tokens": 1865,
    "cost_usd": 0.0234
  },
  "duration_ms": 4523,
  "retries": 0,
  "errors": []
}
```

### Token Usage Tracking

Token usage is automatically tracked and integrated with the existing `token_usage.json` file. Each test run appends data to the project-level tracking.

## Interactive Retry

### On Failure

When a test fails, you'll be prompted:
```
⚠️  Test 'test_spec_stage_conv2d' failed
   Algorithm: conv2d
   Stage: spec
   Attempt: 1/4
   Error: OpenAI API rate limit exceeded
   Category: Rate Limit - Wait before retrying

   Retry? [y/n]: 
```

### Retry Configuration

Configure retry behavior in `.env`:
```
ARDA_MAX_RETRIES=3
ARDA_RETRY_DELAY=1.0
```

## Available Stages

The testing system covers all 8 ARDA pipeline stages:

1. **Spec Stage** (`openai_spec`): Hardware contract generation
2. **Quant Stage** (`openai_quant`): Fixed-point quantization
3. **MicroArch Stage** (`openai_microarch`): Micro-architecture design
4. **Architecture Stage** (`openai_architecture`): RTL architecture planning
5. **RTL Stage** (`openai_rtl`): SystemVerilog code generation
6. **Verification Stage** (`openai_verification`): Test generation and simulation
7. **Synth Stage** (`openai_synth`): Synthesis and timing analysis
8. **Evaluate Stage** (`openai_evaluate`): Performance evaluation

## Available Algorithms

Currently supported algorithms:

- **Conv2D** (`openai_conv2d`): 2D Convolutional Neural Network Layer
- **FFT256** (`openai_fft256`): 256-point Fast Fourier Transform
- **BPF16** (`openai_bpf16`): 16-tap Bandpass Filter
- **Adaptive Filter** (`openai_adaptive_filter`): Complex Adaptive Filter

## Fixtures

### Fixture Structure

Fixtures are stored in `tests/fixtures/{algorithm}/`:
- `{algorithm}_fixtures.json`: Stage output data
- `{algorithm}_bundle.txt`: Algorithm bundle input

### Using Fixtures

Fixtures provide realistic dependency data for stage tests. They are automatically loaded and validated against Pydantic schemas.

### Adding New Algorithms

To add a new algorithm:

1. Create directory: `tests/fixtures/{algorithm}/`
2. Add `{algorithm}_fixtures.json` with stage data
3. Add `{algorithm}_bundle.txt` with algorithm bundle
4. Update test parametrization in `test_openai_stages.py`

## Configuration Options

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | - | OpenAI API key (required for live mode) |
| `ARDA_TEST_MODE` | `live` | Test mode: `live`, `fixture`, or `hybrid` |
| `ARDA_TEST_LOG_DIR` | `tests/logs` | Directory for test logs |
| `ARDA_MAX_RETRIES` | `3` | Maximum retry attempts |
| `ARDA_RETRY_DELAY` | `1.0` | Delay between retries (seconds) |
| `ARDA_MIN_CONFIDENCE` | `70.0` | Minimum confidence score threshold |
| `ARDA_DEBUG_EXTRACTION` | `false` | Enable debug extraction logging |
| `ARDA_DUMP_OPENAI_RESPONSE` | `false` | Dump OpenAI responses for debugging |

### Model Configuration

Models are configured per stage in `tests/openai_test_config.py`:

- **Spec/Quant/MicroArch**: `gpt-4o-mini` (cost-effective)
- **Architecture/RTL**: `gpt-4o` (more capable for complex tasks)
- **Verification/Synth/Evaluate**: `gpt-4o-mini` (analysis tasks)

## Troubleshooting

### Common Issues

1. **API Key Not Set**
   ```
   Skipping OpenAI test - no API key provided
   ```
   Solution: Set `OPENAI_API_KEY` in `.env` file

2. **Rate Limit Exceeded**
   ```
   OpenAI API rate limit exceeded
   ```
   Solution: Wait and retry, or reduce test frequency

3. **Fixture Not Found**
   ```
   Skipping test - Conv2D fixtures not available
   ```
   Solution: Ensure fixture files exist in `tests/fixtures/conv2d/`

4. **Schema Validation Failed**
   ```
   Schema validation failed for stage 'spec'
   ```
   Solution: Check agent configuration and output schemas

### Debug Mode

Enable debug mode for detailed logging:
```bash
ARDA_DEBUG_EXTRACTION=true pytest tests/test_openai_stages.py -v
```

### Response Dumping

Dump OpenAI responses for analysis:
```bash
ARDA_DUMP_OPENAI_RESPONSE=true pytest tests/test_openai_stages.py
```

Responses are saved to `/tmp/arda_openai_{stage}_{reason}_{timestamp}.json`

## Best Practices

### Cost Management

1. **Use selective execution**: Run only the stages you're testing
2. **Use fixture mode**: For development and debugging
3. **Monitor token usage**: Check `token_usage.json` regularly
4. **Set reasonable retry limits**: Avoid excessive API calls

### Test Development

1. **Start with fixtures**: Develop tests using fixture mode first
2. **Test incrementally**: Run individual stages before full pipeline
3. **Validate schemas**: Ensure outputs match expected formats
4. **Log everything**: Use the comprehensive logging for debugging

### CI/CD Integration

For continuous integration:

```bash
# Run only critical stages
pytest -m "openai_spec or openai_quant or openai_architecture"

# Use fixture mode to avoid API costs
ARDA_TEST_MODE=fixture pytest tests/test_openai_stages.py

# Run with timeout to prevent hanging
pytest --timeout=300 tests/test_openai_stages.py
```

## Examples

### Quick Test Run

```bash
# Test just the spec stage
pytest -m openai_spec -v

# Test Conv2D algorithm
pytest -m openai_conv2d -v

# Dry run all tests
ARDA_TEST_MODE=fixture pytest tests/test_openai_stages.py -v
```

### Development Workflow

```bash
# 1. Develop with fixtures
ARDA_TEST_MODE=fixture pytest tests/test_openai_stages.py::TestSpecStage -v

# 2. Test single stage with live API
pytest tests/test_openai_stages.py::TestSpecStage::test_spec_stage_schema -v

# 3. Run full pipeline test
pytest tests/test_openai_stages.py -v
```

### Debugging Failed Tests

```bash
# Enable debug mode
ARDA_DEBUG_EXTRACTION=true pytest tests/test_openai_stages.py::TestSpecStage -v -s

# Dump responses
ARDA_DUMP_OPENAI_RESPONSE=true pytest tests/test_openai_stages.py::TestSpecStage -v

# Check logs
ls tests/logs/
cat tests/logs/2025-10-13_10-30-00/spec_conv2d_1697123456.json
```
