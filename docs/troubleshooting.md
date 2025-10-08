# ARDA Troubleshooting Guide

Common issues, error messages, and solutions for ARDA users and developers.

## Table of Contents

- [Installation Issues](#installation-issues)
- [Bundle Creation Problems](#bundle-creation-problems)
- [OpenAI API Issues](#openai-api-issues)
- [Pipeline Execution Errors](#pipeline-execution-errors)
- [Synthesis Issues](#synthesis-issues)
- [RTL Generation Problems](#rtl-generation-problems)
- [Performance Issues](#performance-issues)
- [Development Issues](#development-issues)

## Installation Issues

### Python Version Compatibility

**Error:**
```
Python 3.8+ is required, but you have Python 3.7
```

**Solution:**
```bash
# Check Python version
python --version

# Install Python 3.8+ if needed
# On Ubuntu/Debian:
sudo apt-get install python3.8 python3.8-venv

# On macOS:
brew install python@3.8

# On Windows: Download from python.org
```

### Package Installation Fails

**Error:**
```
ERROR: Could not find a version that satisfies the requirement ardagen
```

**Solution:**
ARDA is not yet published to PyPI. Install from source:

```bash
# Clone the repository
git clone https://github.com/WestonVoglesonger/ARDA.git
cd ARDA

# Install in development mode
pip install -e .
```

### Import Errors

**Error:**
```
ImportError: No module named 'ardagen'
```

**Solution:**
```bash
# Ensure package is installed
pip show ardagen

# If not installed, install it
pip install -e .

# Check Python path
import sys
print(sys.path)
```

## Bundle Creation Problems

### No Python Files Found

**Error:**
```
Error: No Python files found in directory
```

**Solution:**
- Ensure the directory contains `.py` files
- Check file permissions
- Verify the path is correct

```bash
# Check directory contents
ls -la my_project/

# Check for hidden files
ls -la my_project/ | grep "\.py$"

# Test with single file
arda --create-bundle my_algorithm.py bundle.txt
```

### Bundle Parsing Errors

**Error:**
```
BundleParseError: Invalid bundle format
```

**Solution:**
- Check bundle file encoding (should be UTF-8)
- Ensure proper `path=` prefixes for files
- Verify bundle structure

```bash
# Check bundle content
head -5 my_bundle.txt

# Validate format
grep "^path=" my_bundle.txt
```

### Algorithm Detection Issues

**Error:**
```
Warning: No algorithm patterns detected in file
```

**Solution:**
- Add `step()` method to your algorithm class
- Use class names with "Filter", "Algorithm", or "DSP"
- Ensure the file contains algorithm logic

```python
class MyAlgorithm:
    def __init__(self, params):
        self.params = params

    def step(self, input_data):  # This triggers detection
        # Algorithm logic here
        return output_data
```

## OpenAI API Issues

### Missing API Key

**Error:**
```
RuntimeError: OPENAI_API_KEY environment variable is required
```

**Solution:**
```bash
# Set API key
export OPENAI_API_KEY="your-api-key-here"

# Or set permanently
echo 'export OPENAI_API_KEY="your-key-here"' >> ~/.bashrc

# Verify it's set
echo $OPENAI_API_KEY
```

### Tool Format Errors

**Error:**
```
Missing required parameter: 'tools[0].name'
```

**Solution:**
This was fixed in recent versions. Ensure you're using the latest ARDA:

```bash
pip install --upgrade arda
```

### Rate Limiting

**Error:**
```
Rate limit exceeded. Try again in X minutes.
```

**Solution:**
```bash
# Use deterministic agents (no API calls)
arda algorithm.txt --agent-runner deterministic

# Or reduce request frequency by using smaller models
# Edit arda/model_config.py to use gpt-3.5-turbo instead of gpt-4
```

### API Quota Exceeded

**Error:**
```
You exceeded your current quota
```

**Solution:**
- Check OpenAI usage dashboard
- Upgrade billing plan
- Use `--agent-runner deterministic` for development
- Implement caching for repeated operations

## Pipeline Execution Errors

### Stage Dependency Errors

**Error:**
```
Stage 'rtl' missing dependencies: spec, quant, microarch
```

**Solution:**
- Ensure all previous stages completed successfully
- Check pipeline execution order
- Verify stage outputs are properly saved

```bash
# Run with verbose output to see stage progress
arda algorithm.txt --verbose

# Check intermediate results
arda algorithm.txt --extract-rtl rtl_output/ --output results.json
```

### Memory Issues

**Error:**
```
MemoryError: Out of memory during RTL generation
```

**Solution:**
```bash
# Reduce model complexity
# Edit arda/model_config.py to use smaller models

# Process in chunks for large algorithms
# Modify stage implementations to handle large datasets

# Enable memory debugging
export PYTHONTRACEMALLOC=1
```

### Timeout Errors

**Error:**
```
TimeoutError: Stage execution timed out
```

**Solution:**
```bash
# Increase timeout in agent configurations
# Edit agent_configs.json timeout settings

# Use faster models for development
# Modify arda/model_config.py

# Enable async processing where possible
```

## Synthesis Issues

### Vivado Not Found

**Error:**
```
SynthesisError: Vivado executable not found
```

**Solution:**
```bash
# Install Xilinx Vivado
# Add to PATH
export PATH=$PATH:/opt/Xilinx/Vivado/2023.2/bin

# Or use open-source alternatives
arda algorithm.txt --synthesis-backend yosys
```

### Timing Not Met

**Error:**
```
SynthesisError: Timing not met (fmax=80MHz, target=100MHz)
```

**Solution:**
- The feedback agent will automatically suggest improvements
- Check synthesis constraints and clock definitions
- Review RTL for timing-critical paths

```bash
# Run multiple iterations for optimization
arda algorithm.txt --verbose  # Shows feedback decisions

# Check synthesis reports
arda algorithm.txt --extract-rtl rtl_output/
# Look for timing reports in rtl_output/
```

### Resource Overutilization

**Error:**
```
SynthesisError: LUT usage exceeds device capacity
```

**Solution:**
- Review resource estimates in microarchitecture stage
- Adjust resource budgets in algorithm specification
- Enable resource optimization in synthesis settings

```bash
# Adjust resource constraints
arda algorithm.txt --max-luts 8000 --max-dsps 40
```

## RTL Generation Problems

### Syntax Errors in Generated RTL

**Error:**
```
SyntaxError: Invalid SystemVerilog syntax
```

**Solution:**
- Check RTL files for syntax issues
- Review agent instructions for RTL generation
- Use deterministic agents for debugging

```bash
# Extract and inspect RTL
arda algorithm.txt --extract-rtl rtl_output/
cat rtl_output/rtl/*.sv | head -20

# Use deterministic agents for comparison
arda algorithm.txt --agent-runner deterministic --extract-rtl rtl_output_det/
```

### Incorrect Interface Generation

**Error:**
```
RTL interface doesn't match expected format
```

**Solution:**
- Verify algorithm bundle specifies correct interfaces
- Check AXI-Stream interface requirements
- Review generated RTL for interface compliance

### Coefficient Issues

**Error:**
```
quantized_coefficients is None in RTL generation
```

**Solution:**
- Ensure quantization stage completed successfully
- Check that quantized coefficients are properly formatted
- Add defensive programming in RTL agent

```bash
# Debug quantization stage
arda algorithm.txt --verbose | grep -A 10 -B 5 "quant"
```

## Performance Issues

### Slow Pipeline Execution

**Problem:** Pipeline takes too long to complete

**Solution:**
```bash
# Use faster models
# Edit arda/model_config.py

# Enable caching for repeated operations
# Implement LRU cache in expensive computations

# Profile performance
python -m cProfile -s time arda algorithm.txt > profile.txt
```

### High Memory Usage

**Problem:** Pipeline consumes excessive memory

**Solution:**
```bash
# Process large datasets in chunks
# Modify stage implementations

# Monitor memory usage
export PYTHONTRACEMALLOC=1
arda algorithm.txt

# Check for memory leaks
python -m tracemalloc --traceback arda algorithm.txt
```

### Network Issues

**Problem:** API calls fail due to network problems

**Solution:**
```bash
# Test network connectivity
curl -I https://api.openai.com

# Use local caching for repeated requests
# Implement request retry logic

# Use deterministic mode for offline development
arda algorithm.txt --agent-runner deterministic
```

## Development Issues

### Test Failures

**Error:**
```
FAILED tests/test_pipeline.py::test_rtl_generation
```

**Solution:**
- Check test environment setup
- Ensure all dependencies are installed
- Review test data and expectations

```bash
# Run tests with verbose output
pytest tests/ -v -s

# Run specific failing test
pytest tests/test_pipeline.py::test_rtl_generation -v

# Check test environment
python -c "import ardagen; print('Import successful')"
```

### Type Checking Errors

**Error:**
```
error: Incompatible types in assignment
```

**Solution:**
- Add proper type annotations
- Use mypy for type checking
- Fix type mismatches

```bash
# Install mypy
pip install mypy

# Run type checking
mypy arda/

# Fix identified issues
```

### Import Path Issues

**Error:**
```
ModuleNotFoundError: No module named 'arda.agents'
```

**Solution:**
- Ensure package is properly installed
- Check PYTHONPATH
- Verify package structure

```bash
# Check installation
pip list | grep arda

# Check Python path
python -c "import sys; print('\n'.join(sys.path))"

# Reinstall if needed
pip uninstall arda
pip install -e .
```

## Getting Help

### Debug Information Collection

When reporting issues, include:

1. **ARDA version**: `arda --version`
2. **Python version**: `python --version`
3. **Operating system**: `uname -a`
4. **Full error traceback**: Run with `--verbose`
5. **Bundle file** (if applicable)
6. **Configuration files** (if modified)

### Reporting Issues

- **GitHub Issues**: [https://github.com/WestonVoglesonger/ARDA/issues](https://github.com/WestonVoglesonger/ARDA/issues)
- **Discussions**: [https://github.com/WestonVoglesonger/ARDA/discussions](https://github.com/WestonVoglesonger/ARDA/discussions)

### Common Workarounds

#### Development Mode
```bash
# Use deterministic agents for development
arda algorithm.txt --agent-runner deterministic

# Enable detailed logging
arda algorithm.txt --verbose --debug

# Save intermediate results
arda algorithm.txt --output debug_results.json
```

#### Production Mode
```bash
# Use optimized settings
arda algorithm.txt --agent-runner openai --synthesis-backend vivado

# Enable error recovery
arda algorithm.txt --max-retries 3

# Monitor performance
arda algorithm.txt --verbose | tee execution.log
```

## Version-Specific Issues

### v1.0.0 Issues

- **OpenAI API compatibility**: Fixed in this release
- **Bundle creation**: Enhanced with auto-detection
- **RTL simulation**: Added iverilog/verilator support

### Migration from v0.9.x

```bash
# Update package
pip install --upgrade arda

# Regenerate bundles if needed
arda --create-bundle old_algorithm.py new_bundle.txt

# Test with new version
arda new_bundle.txt --verbose
```

## Performance Tuning

### Agent Performance

```bash
# Use faster models for development
# Edit arda/model_config.py

# Enable caching for expensive operations
@functools.lru_cache(maxsize=128)
def expensive_computation(params):
    return result
```

### Memory Optimization

```bash
# Process large RTL files in chunks
def process_rtl_in_chunks(files, chunk_size=10):
    for i in range(0, len(files), chunk_size):
        chunk = files[i:i + chunk_size]
        yield process_chunk(chunk)
```

### Network Optimization

```bash
# Implement retry logic for API calls
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def call_openai_api(request):
    return client.responses.create(**request)
```

## Advanced Debugging

### Pipeline State Inspection

```python
# Debug pipeline state at each stage
def debug_pipeline_state(stage_name, context):
    print(f"=== {stage_name} Debug ===")
    print(f"Context keys: {list(context.__dict__.keys())}")
    print(f"Results keys: {list(context.results.keys())}")

    for key, value in context.results.items():
        print(f"  {key}: {type(value).__name__}")
        if hasattr(value, 'model_dump'):
            data = value.model_dump()
            print(f"    Fields: {list(data.keys())}")
```

### Agent Tool Debugging

```python
# Patch tools for debugging
original_tool = FUNCTION_MAP["write_artifact"]

def debug_tool(*args, **kwargs):
    print(f"Tool called: write_artifact({args}, {kwargs})")
    result = original_tool(*args, **kwargs)
    print(f"Tool result: {result}")
    return result

FUNCTION_MAP["write_artifact"] = debug_tool
```

### Memory Leak Detection

```bash
# Enable memory tracing
export PYTHONTRACEMALLOC=1

# Run pipeline
arda algorithm.txt

# Analyze memory usage
python -c "
import tracemalloc
tracemalloc.start()
# Your code here
current, peak = tracemalloc.get_traced_memory()
print(f'Current: {current / 1024 / 1024:.2f} MB')
print(f'Peak: {peak / 1024 / 1024:.2f} MB')
"
```

## Emergency Fixes

### Reset to Working State

```bash
# If pipeline gets stuck, reset workspace
rm -rf workspace_* generated_rtl/*

# Clear Python cache
find . -name "*.pyc" -delete
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# Reinstall package
pip uninstall arda
pip install -e .
```

### Quick Debugging Mode

```bash
# Minimal pipeline for debugging
arda algorithm.txt --agent-runner deterministic --synthesis-backend none --verbose
```

---

*This troubleshooting guide is regularly updated based on user reports and development experience. For issues not covered here, please open a GitHub issue with detailed information about your setup and the problem you're experiencing.*
