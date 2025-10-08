# ARDA User Guide

A comprehensive guide for using ARDA to convert Python algorithms to SystemVerilog RTL.

## Table of Contents

- [Getting Started](#getting-started)
- [Bundle Format](#bundle-format)
- [Command Line Interface](#command-line-interface)
- [Pipeline Stages](#pipeline-stages)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)
- [Advanced Usage](#advanced-usage)

## Getting Started

### Prerequisites

- Python 3.8+
- OpenAI API key (for AI-powered agents)
- Optional: Icarus Verilog or Verilator (for real RTL simulation)

### Installation

**Note:** ARDA is not yet published to PyPI. Install from source:

```bash
# Clone the repository
git clone https://github.com/WestonVoglesonger/ARDA.git
cd ARDA

# Install in development mode
pip install -e .
```

### First Algorithm Conversion

1. **Create a Python algorithm file:**
```python
# my_fir.py
import numpy as np

class StreamingFIR:
    def __init__(self, coeffs):
        self.coeffs = coeffs
        self.history = [0] * len(coeffs)

    def step(self, x):
        # Shift history
        self.history = [x] + self.history[:-1]
        # Compute FIR output
        return sum(h * c for h, c in zip(self.history, self.coeffs))

# Example usage
coeffs = [0.1, 0.2, 0.3, 0.2, 0.1]
fir = StreamingFIR(coeffs)
```

2. **Create bundle:**
```bash
arda --create-bundle my_fir.py fir_bundle.txt
```

3. **Generate RTL:**
```bash
arda fir_bundle.txt --verbose --agent-runner openai
```

4. **Extract RTL files:**
```bash
arda fir_bundle.txt --extract-rtl rtl_output/
```

## Bundle Format

ARDA uses a text-based bundle format to encapsulate algorithm code and metadata.

### Bundle Structure

```text
path=algorithms/my_algorithm.py
# Optional algorithm description and metadata
# Performance requirements: 200MHz target, streaming interface
# Input format: 12-bit signed integers
# Output format: 16-bit signed integers

import numpy as np

class MyAlgorithm:
    def __init__(self, params):
        self.params = params

    def step(self, input_sample):
        # Your algorithm logic here
        return output_sample
```

### Bundle Creation Options

#### From Single File
```bash
arda --create-bundle algorithm.py bundle.txt
```

#### From Directory
```bash
arda --create-bundle my_project/ project_bundle.txt
```

#### Bundle Features
- **Auto-detection**: Recognizes algorithm patterns (`step()` methods, Filter classes)
- **Metadata extraction**: Adds helpful comments about algorithm interfaces
- **Multiple files**: Supports bundling multiple related Python files
- **Rich context**: Includes algorithm descriptions and requirements

## Command Line Interface

### Basic Usage

```bash
arda <bundle_file> [options]
```

### Options

| Option | Description | Example |
|--------|-------------|---------|
| `--verbose` | Detailed output | `--verbose` |
| `--agent-runner` | Agent backend | `--agent-runner openai` |
| `--create-bundle` | Create bundle from files | `--create-bundle src/ bundle.txt` |
| `--extract-rtl` | Extract RTL to directory | `--extract-rtl rtl/` |
| `--output` | Save results to JSON | `--output results.json` |
| `--synthesis-backend` | Synthesis tool | `--synthesis-backend vivado` |

### Agent Runners

- **`auto`** (default): Uses OpenAI if available, falls back to deterministic
- **`openai`**: Force OpenAI Agents SDK (requires API key)
- **`deterministic`**: Use rule-based agents (no AI, for testing)

### Synthesis Backends

- **`auto`**: Automatically detect best available backend
- **`yosys`**: Open-source synthesis for iCE40, ECP5, etc.
- **`vivado`**: Xilinx Vivado (requires installation)

## Pipeline Stages

ARDA processes algorithms through 8 specialized stages:

### 1. Specification Stage (`spec`)
**Purpose**: Analyzes algorithm and generates hardware requirements
**Input**: Algorithm bundle
**Output**: Hardware contract (clock frequency, data formats, resource budget)
**Agent**: Spec Agent

### 2. Quantization Stage (`quant`)
**Purpose**: Converts floating-point to fixed-point arithmetic
**Input**: Spec contract
**Output**: Fixed-point configuration and quantized coefficients
**Agent**: Quant Agent

### 3. Microarchitecture Stage (`microarch`)
**Purpose**: Designs hardware architecture (pipeline depth, unrolling, memory)
**Input**: Quant configuration
**Output**: Microarchitecture decisions
**Agent**: MicroArch Agent

### 4. RTL Generation Stage (`rtl`)
**Purpose**: Generates synthesizable SystemVerilog code
**Input**: Microarchitecture and quantization data
**Output**: Complete RTL implementation with AXI-Stream interfaces
**Agent**: RTL Agent

### 5. Static Checks Stage (`static_checks`)
**Purpose**: Lint and style analysis of generated RTL
**Input**: RTL files
**Output**: Quality metrics and issue reports

### 6. Verification Stage (`verification`)
**Purpose**: Functional verification against golden reference
**Input**: RTL and test vectors
**Output**: Pass/fail results and error metrics

### 7. Synthesis Stage (`synth`)
**Purpose**: FPGA synthesis and timing analysis
**Input**: RTL files
**Output**: Timing, area, and power reports

### 8. Evaluation Stage (`evaluate`)
**Purpose**: Aggregate results into scorecard
**Input**: All stage outputs
**Output**: Overall quality assessment and recommendations

## Examples

### FIR Filter Example

**Algorithm:**
```python
class StreamingFIR:
    def __init__(self, coeffs):
        self.coeffs = coeffs
        self.history = [0] * len(coeffs)

    def step(self, x):
        self.history = [x] + self.history[:-1]
        return sum(h * c for h, c in zip(self.history, self.coeffs))
```

**Command:**
```bash
arda --create-bundle fir.py fir_bundle.txt
arda fir_bundle.txt --verbose --agent-runner openai --extract-rtl rtl_output/
```

**Results:**
```
RTL generated: 4 different architectural variants
Synthesis: 102.5MHz achieved (target: 100MHz)
Verification: 100% pass rate (1024/1024 tests)
Resources: 450 LUTs, 520 FFs, 3 DSPs
```

### Advanced Algorithm Example

**Multi-stage Algorithm:**
```python
class MultiStageProcessor:
    def __init__(self):
        self.stage1_state = 0
        self.stage2_buffer = []

    def step(self, input_data):
        # Stage 1: Preprocessing
        processed = self._preprocess(input_data)

        # Stage 2: Core algorithm
        result = self._core_processing(processed)

        # Stage 3: Postprocessing
        output = self._postprocess(result)

        return output
```

## Troubleshooting

### Common Issues

#### OpenAI API Errors
```
Error: Missing required parameter: 'tools[0].name'
```
**Solution**: This was fixed in recent versions. Ensure you're using the latest ARDA.

#### Bundle Creation Issues
```
Error: No Python files found in directory
```
**Solution**: Ensure the directory contains `.py` files with algorithm code.

#### Synthesis Failures
```
Error: Timing not met (fmax=80MHz, target=100MHz)
```
**Solution**: The feedback agent will suggest microarchitecture adjustments and retry.

### Debugging

#### Verbose Output
```bash
arda algorithm.txt --verbose
```
Shows detailed progress through each pipeline stage.

#### Agent Debugging
```bash
arda algorithm.txt --verbose --agent-runner openai
```
Shows AI agent decisions and tool calls.

#### RTL Inspection
```bash
arda algorithm.txt --extract-rtl rtl_output/
ls -la rtl_output/
```
Examine generated RTL files.

## Advanced Usage

### Custom Agent Configuration

Create `custom_agents.json`:
```json
{
  "agents": {
    "custom_rtl_agent": {
      "name": "Custom RTL Agent",
      "instructions": "Generate RTL with specific coding style...",
      "model": "gpt-4",
      "tools": [...]
    }
  }
}
```

### Integration with FPGA Tools

#### Vivado Integration
```bash
arda algorithm.txt --synthesis-backend vivado --fpga-family xc7a100t
```

#### Yosys Integration
```bash
arda algorithm.txt --synthesis-backend yosys --fpga-family ice40hx8k
```

### Performance Optimization

#### Target-Specific Optimization
```bash
arda algorithm.txt --target-clock 250MHz --target-device xc7a200t
```

#### Resource Constraints
```bash
arda algorithm.txt --max-luts 10000 --max-dsps 50
```

## Architecture Overview

ARDA uses a modular architecture with specialized AI agents:

- **Orchestrator**: Coordinates pipeline execution and feedback loops
- **Agent Registry**: Manages different AI agent implementations
- **Tool Adapters**: Interface with external EDA tools
- **Domain Models**: Structured data models for each pipeline stage
- **Observability**: Comprehensive logging and metrics collection

## Contributing

See [DEVELOPER_GUIDE.md](developer_guide.md) for contribution guidelines.

## Support

- **Issues**: [GitHub Issues](https://github.com/WestonVoglesonger/ARDA/issues)
- **Discussions**: [GitHub Discussions](https://github.com/WestonVoglesonger/ARDA/discussions)
- **Documentation**: This user guide and [API docs](api_docs.md)

## License

MIT License - see [LICENSE](LICENSE) file for details.
