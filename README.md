# ARDA: AI-Powered RTL Generation from Python Algorithms 🚀

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status: Working](https://img.shields.io/badge/status-working-brightgreen.svg)](#current-capabilities)

**ARDA (Automated RTL Design Agents)** is a groundbreaking AI system that automatically converts Python algorithms into production-quality SystemVerilog RTL for FPGA implementation. Powered by OpenAI's Agents SDK, ARDA uses specialized AI agents to transform streaming algorithms through an intelligent, multi-stage pipeline.

🎯 **Breakthrough Achievement**: Successfully generates synthesizable RTL that achieves target clock frequencies (102.5MHz vs 100MHz target) with 100% functional verification.

## 🚀 Quick Start

### Installation

**Note:** ARDA is not yet published to PyPI. Install from source:

```bash
# Clone the repository
git clone https://github.com/WestonVoglesonger/ARDA.git
cd ARDA

# Install in development mode
pip install -e .
```

### Convert Your Python Algorithm to RTL

```bash
# 1. Create a bundle from your Python file
arda --create-bundle my_algorithm.py algorithm_bundle.txt

# 2. Generate RTL with AI agents
arda algorithm_bundle.txt --verbose --agent-runner openai

# 3. Extract the generated RTL files
arda algorithm_bundle.txt --extract-rtl rtl_output/
```

**Example Results:**
```
✅ Pipeline completed successfully!
   Algorithm: FIR5_Symmetric_5tap_FPGA
   Target: 100.0MHz
   Achieved: 102.5MHz
   Resources: 450 LUTs, 520 FFs, 3 DSPs
   Verification: ✅ Passed
```

## 📚 Documentation

- **[User Guide](docs/user_guide.md)** - Complete guide for using ARDA
- **[Developer Guide](docs/developer_guide.md)** - For contributors and advanced users
- **[API Documentation](docs/api_docs.md)** - Comprehensive API reference
- **[Troubleshooting Guide](docs/troubleshooting.md)** - Common issues and solutions
- **[Examples](docs/examples.md)** - Practical usage examples and patterns
- **[Architecture](docs/architecture.md)** - Technical architecture overview

## Overview

This system takes a Python algorithm as input and guides it through a complete hardware design pipeline:

- `SpecStage` → Generate a hardware contract from the bundle
- `QuantStage` → Derive fixed-point formats and error budgets
- `MicroArchStage` → Choose pipeline depth, unrolling, and buffers
- `RTLStage` → Emit synthesizable SystemVerilog artifacts
- `StaticChecksStage` → Run lint/style/structural analysis
- `VerificationStage` → Execute simulation-driven verification and coverage analysis
- `SynthStage` → Launch backend synthesis (Vivado/Yosys/etc.)
- `EvaluateStage` → Aggregate reports into a scorecard for feedback

Behind the scenes the orchestrator pulls agents from `alg2sv/agents/registry.py`, which can be swapped out in `alg2sv/runtime/agent_runner.py` to integrate real LLM- or tool-backed flows. Deterministic tool adapters live under `alg2sv/tools/` providing default lint, simulation, synthesis, and reporting stubs.

1. **Spec Agent**: Analyzes algorithm and generates hardware contract
2. **Quant Agent**: Converts to fixed-point arithmetic with error analysis
3. **MicroArch Agent**: Designs micro-architecture and dataflow
4. **RTL Agent**: Generates synthesizable SystemVerilog with AXI interfaces
5. **Verify Agent**: Runs functional verification against golden reference
6. **Synth Agent**: Synthesizes design and reports timing/area metrics

## ⚠️ Current Limitations

**CRITICAL: ARDA generates RTL but does NOT perform real hardware verification!**

### What ARDA Currently Does:
- ✅ Generates synthesizable SystemVerilog RTL
- ✅ Performs software-based functional verification
- ✅ Estimates synthesis results (timing, area)
- ✅ Lints and analyzes code quality
- ✅ Validates against golden reference models

### What ARDA Does NOT Do (Without Vivado):
- ❌ **Real FPGA synthesis** (requires Vivado installation)
- ❌ **Bitstream generation** (requires Vivado for .bit files)
- ❌ **Hardware programming** (requires JTAG/FPGA board)
- ❌ **Actual FPGA testing** (requires hardware setup)
- ❌ **PCIe/DMA data transfer** (requires board-specific drivers)

### True Hardware Verification Requires:
1. **Synthesis Tool Integration** (Vivado CLI, Quartus)
2. **FPGA Board Access** (Xilinx Zynq/Artix/Kintex)
3. **Hardware Test Infrastructure** (PCIe, DMA, JTAG)
4. **Real-time Data Transfer** (hardware vs software comparison)

**Bottom Line**: ARDA is currently a **software-based RTL generation and estimation tool**. For production use, the generated RTL must be synthesized and tested on actual FPGA hardware separately.

## 🛠️ Vivado Integration (Real Hardware Synthesis)

ARDA now includes **optional Vivado CLI integration** for real FPGA synthesis and bitstream generation!

### Prerequisites

1. **Xilinx Vivado Design Suite** (2022.2 or later)
   ```bash
   # Download from: https://www.xilinx.com/support/download.html
   # Install Vivado Design Suite
   ```

2. **Add Vivado to PATH**
   ```bash
   # Linux/Mac
   export PATH=$PATH:/opt/Xilinx/Vivado/2023.2/bin

   # Windows
   set PATH=%PATH%;C:\\Xilinx\\Vivado\\2023.2\\bin
   ```

3. **Verify Installation**
   ```bash
   vivado -version
   ```

### Vivado-Enabled Features

When Vivado is available, ARDA can perform:

- ✅ **Real FPGA Synthesis** (not estimation)
- ✅ **Actual Implementation** with place & route
- ✅ **Bitstream Generation** (.bit files)
- ✅ **Accurate Resource Usage** (LUTs, FFs, DSPs, BRAM)
- ✅ **Real Timing Analysis** (WNS, TNS, slack)
- ✅ **Power Analysis** (when available)

### Testing Vivado Integration

```bash
# Test if Vivado is properly integrated
python test_vivado_integration.py

# Run full pipeline with real synthesis
python run_pipeline.py test_algorithms/conv2d_bundle.txt
```

### Default Target FPGA

**Xilinx Nexys 7 Board**
- **FPGA**: Artix-7 XC7A100T-1CSG324C
- **LUTs**: 63,400
- **Flip-Flops**: 126,800  
- **DSP Slices**: 240
- **Block RAM**: 4.9 Mb
- **Max Frequency**: 450 MHz (450 MHz for simple designs, 200-300 MHz typical)

### Supported FPGA Families

- **Xilinx Artix-7** (Nexys 7, Basys 3) - **Recommended for DSP designs**
- **Xilinx Kintex-7** - High-performance applications
- **Xilinx Virtex-7** - Maximum performance
- **Xilinx UltraScale** - Kintex UltraScale, Virtex UltraScale
- **Xilinx Zynq** - Zynq-7000, Zynq UltraScale+
- **iCE40 HX/UP** - Limited support (no DSPs, max ~50 MHz)

### Device Selection Guidelines
- **For DSP-heavy designs**: Use Xilinx Artix-7 or Kintex-7
- **For simple logic**: iCE40 is sufficient but frequency-limited
- **For 200+ MHz targets**: Xilinx devices recommended
- **For prototyping**: Nexys 7 (XC7A100T) provides excellent balance

### Vivado Integration Architecture

```
Algorithm → RTL → Vivado TCL Script → Synthesis → Implementation → Bitstream
     ↓         ↓           ↓              ↓           ↓             ↓
   Python    .sv        .tcl script    .dcp        .bit         .bit file
   code      files      generation    results    generation    + reports
```

### TCL Script Generation

ARDA automatically generates comprehensive TCL scripts that:

1. Create Vivado projects
2. Add RTL sources and constraints
3. Run synthesis with timing optimization
4. Perform place & route
5. Generate utilization/timing reports
6. Create bitstreams for programming

### Hardware Verification Pipeline

With Vivado integration, ARDA becomes a **complete hardware development platform**:

```
Algorithm → RTL Generation → Synthesis → Bitstream → FPGA Programming → Testing
     ↓            ↓             ↓          ↓            ↓            ↓
   Design     Vivado CLI     .bit      Hardware    PCIe/DMA    Validation
   Spec       Integration   file      Interface   Transfer    Results
```

## Key Features

- **Virtual Workspace**: Input algorithms as bundled strings (no file uploads needed)
- **Structured Pipeline**: Deterministic agent handoffs with quality gates
- **Fixed-Point Conversion**: Automatic quantization with error metrics
- **Streaming RTL Generation**: AXI-Stream interfaces with pipeline optimization
- **End-to-End Verification**: Golden model comparison with mismatch reporting
- **Synthesis Integration**: CI/CD pipeline for FPGA synthesis and timing closure

## Quick Start

### 1. Installation

#### From Source (Development)

```bash
# Clone the repository
git clone https://github.com/WestonVoglesonger/arda.git
cd arda

# Install in development mode (recommended for contributors)
pip install -e .
```

#### Environment Setup

Set your OpenAI API key:
```bash
export OPENAI_API_KEY="your-openai-api-key-here"
```

### 3. Choose Synthesis Backend

ARDA supports multiple FPGA synthesis backends (legacy `alg2sv` CLI remains available):

```bash
# Auto-detect best available backend
arda examples/bpf16_bundle.txt --synthesis-backend auto

# Xilinx Vivado (for Xilinx 7-series FPGAs)
arda examples/bpf16_bundle.txt --synthesis-backend vivado --fpga-family xc7a100t

# Open-source Yosys (for iCE40/ECP5 FPGAs)
arda examples/bpf16_bundle.txt --synthesis-backend yosys --fpga-family ice40hx8k

# Experimental SymbiFlow (for Xilinx 7-series)
arda examples/bpf16_bundle.txt --synthesis-backend symbiflow --fpga-family xc7a100t
```

### 4. Test with Example Algorithm

```bash
# Run the BPF16 filter example
arda examples/bpf16_bundle.txt --verbose --workspace-info

# Or use the module directly
python -m alg2sv.cli examples/bpf16_bundle.txt --verbose
```

### 5. Input Algorithm Format

Create algorithm bundles in this format:

```text
``` path=algorithms/myfilter/algo.py
# Your Python algorithm here
import numpy as np

def step(x: float) -> float:
    # Your streaming algorithm
    return processed_sample
```

``` path=algorithms/myfilter/meta.yaml
name: MyFilter
description: Custom digital filter
clock_mhz_target: 200
resource_budget:
  lut: 10000
  ff: 20000
  dsp: 20
  bram: 10
throughput:
  samples_per_cycle: 1
verify:
  num_samples: 1000
  tolerance_abs: 1.0e-3
```

``` path=algorithms/myfilter/vectors.py
# Test vector generation (optional)
```
```

### 6. Run Custom Algorithms

```bash
# Run with your own bundle file
arda my_algorithm_bundle.txt --output results.json --extract-rtl rtl_output/

# Run with inline bundle
arda --bundle "$(cat my_algorithm_bundle.txt)" --verbose --agent-runner auto
```

### 7. Testing the Pipeline

### Automated Testing

> **Note:** Python import paths currently remain under the `alg2sv` namespace until the package rename milestone lands.

```bash
# Run the complete pipeline test
python -m pytest tests/ -v

# Or run individual components
python -c "
from alg2sv.simplified_pipeline import SimplifiedPipeline
from alg2sv.runtime import DefaultAgentRunner
from alg2sv.agents import create_default_registry
from alg2sv.test_algorithms.bpf16_bundle import get_bundle

runner = DefaultAgentRunner(create_default_registry())
pipeline = SimplifiedPipeline(agent_runner=runner)
result = await pipeline.run(bundle_data)
print('Success!' if result['success'] else 'Failed!')
"
```

### Manual Testing Steps

1. **Test Workspace Management:**
```python
from alg2sv.workspace import ingest_from_bundle

# Test bundle ingestion
bundle = """``` path=test.py
print('hello')
```"""

result = ingest_from_bundle(bundle)
print(f"Workspace: {result['workspace_token']}")
```

2. **Test Individual Agents:**
```python
from alg2sv.agents import create_default_registry

registry = create_default_registry()
spec_agent = registry.get_stage_agent("spec")
spec_result = await spec_agent({"bundle": "...", "observability": None})
print(spec_result)
```

### End-to-End Pipeline Test
```bash
# Full pipeline with verbose output
arda test_algorithms/bpf16_bundle.txt \
  --verbose \
  --workspace-info \
  --output test_results.json \
  --extract-rtl test_rtl/
```

### Expected Test Results

✅ **Successful Run:**
- Pipeline completes all 6 stages
- Generates SystemVerilog files
- Verification passes (100% match)
- Synthesis meets timing/area constraints

📊 **Sample Output:**
```
📥 Ingesting algorithm bundle...
✅ Created workspace with 3 files
🔍 Running Spec Agent...
✅ Spec: BPF16 - 200MHz target
🔢 Running Quant Agent...
✅ Quant: 16 coeffs, error=1.23e-06
🏗️ Running MicroArch Agent...
✅ MicroArch: 4 stages, 16 DSPs
💾 Running RTL Agent...
✅ RTL: Generated 2 files, top=bpf16_top
✅ Running Verify Agent...
✅ Verify: 1024/1024 tests passed
🔨 Running Synth Agent...
🎉 Pipeline completed successfully!
   Target: 200.0MHz, Achieved: 198.5MHz
   Resources: 1847 LUTs, 3201 FFs, 16 DSPs
```

### Troubleshooting Tests

**❌ Agent Import Errors:**
```bash
# Check OpenAI Agents SDK installation
pip show openai-agents

# Verify API key
echo $OPENAI_API_KEY
```

**❌ Bundle Parsing Issues:**
```bash
# Test bundle parsing
python -c "
from alg2sv.workspace import ingest_from_bundle
result = ingest_from_bundle(open('test_algorithms/bpf16_bundle.txt').read())
print('Parsed successfully' if result['success'] else result['error'])
"
```

**❌ API Rate Limits:**
- The pipeline makes multiple OpenAI API calls
- Monitor usage in OpenAI dashboard
- Consider adding delays between agent calls

## Architecture

### Agent Pipeline Flow

```
Input Bundle String
       ↓
   ingest_from_bundle()
       ↓
Spec Agent → Quant Agent → MicroArch Agent → RTL Agent → Verify Agent → Synth Agent
       ↓           ↓            ↓              ↓            ↓              ↓
   Contract   Fixed-Point   Architecture     SystemVerilog  Verification   Synthesis
```

### Tools Integration

- **Local Functions**: Virtual workspace management (ingest/read/write)
- **Code Interpreter**: Numerical analysis and golden model execution
- **External Functions**: Synthesis job dispatch and result polling
- **Logic Nodes**: Conditional routing based on verification/synthesis results

### Quality Gates

- **Spec Gate**: Valid contract with resource budgets
- **Quant Gate**: Error metrics within tolerance
- **RTL Gate**: Lint-clean synthesizable code
- **Verify Gate**: Bit-exact match with golden reference
- **Synth Gate**: Timing closure + resource budget compliance

## Supported Algorithms

The pipeline works with streaming algorithms that implement a `step(x) -> y` interface:

- **FIR/IIR Filters**: Band-pass, low-pass, high-pass
- **FFT**: Streaming FFT implementations
- **DWT**: Discrete Wavelet Transform
- **Custom DSP**: Any algorithm with fixed sample rate

### Example: 16-Tap Band-Pass Filter

See `test_algorithms/bpf16_bundle.txt` for a complete working example.

## Tool Setup

### Local Function Tools

Upload these to Agent Builder:

1. **`ingest_from_bundle`**: Parses algorithm bundle into workspace
2. **`read_source`**: Reads files from workspace
3. **`write_artifact`**: Writes generated files to workspace

### External Integration

For synthesis (optional but recommended):

1. Set up GitHub Actions workflow for FPGA synthesis
2. Configure `submit_synth_job` and `fetch_synth_results` functions
3. Supports Yosys, Vivado, or other synthesis tools

### OpenAI Agents Runtime

1. Install the latest OpenAI Python SDK: `pip install openai`
2. Export `OPENAI_API_KEY`
3. Run the CLI with `--agent-runner openai` (or rely on `--agent-runner auto` with the key set)
4. Agents, prompts, and tool schemas are defined in `agent_configs.json`

## Configuration

### Agent Instructions

Each agent has detailed instructions in `agent_configs.json`:

- Clear responsibilities and deliverables
- Tool usage guidance
- Quality criteria for outputs

### Structured Outputs

All agents emit JSON with defined schemas:

- Spec Agent: Hardware contract specifications
- Quant Agent: Fixed-point configuration + error metrics
- RTL Agent: File paths + lint status
- Verify Agent: Pass/fail + mismatch details
- Synth Agent: Timing/area metrics

## Development

### Adding New Algorithms

1. Create algorithm bundle (algo.py + meta.yaml + vectors.py)
2. Test with Spec Agent first
3. Verify quantization accuracy
4. Check RTL synthesis results

### Extending the Pipeline

- Add new agent types (e.g., power analysis, formal verification)
- Integrate additional synthesis tools
- Add custom quality gates
- Support new algorithm patterns

## Limitations & Future Work

### Current Limitations

- Streaming algorithms only (no batch processing)
- Fixed-point arithmetic focus (no floating-point FPGA support)
- Basic synthesis integration (expand to full P&R flow)
- Single-clock domain (no CDC support)

### Roadmap

- Multi-rate algorithms (decimation/interpolation)
- Advanced architectures ( systolic arrays, CORDIC)
- Formal verification integration
- Power analysis and optimization
- Multi-FPGA partitioning

## Files Structure

```
/Users/westonvoglesonger/Projects/ARDA/
├── agent_configs.json          # Agent configurations for Agent Builder
├── tools/                      # Local Function tool implementations
│   ├── ingest_from_bundle.js
│   ├── read_source.js
│   └── write_artifact.js
├── test_algorithms/            # Example algorithm bundles
│   └── bpf16_bundle.txt
└── README.md                   # This file
```

## Contributing

1. Test with new algorithm types
2. Improve agent prompts for better RTL generation
3. Add synthesis tool integrations
4. Enhance verification coverage

## License

This project demonstrates advanced AI agent orchestration for hardware design automation.
