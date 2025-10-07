# ALG2SV: Algorithm to SystemVerilog Pipeline

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A sophisticated multi-agent AI system that automatically converts Python algorithms into synthesizable SystemVerilog RTL for FPGA implementation. Built with OpenAI's Agents SDK, ALG2SV transforms streaming algorithms through a structured pipeline of specialized AI agents.

## Overview

This system takes a Python algorithm as input and guides it through a complete hardware design pipeline:

1. **Spec Agent**: Analyzes algorithm and generates hardware contract
2. **Quant Agent**: Converts to fixed-point arithmetic with error analysis
3. **MicroArch Agent**: Designs micro-architecture and dataflow
4. **RTL Agent**: Generates synthesizable SystemVerilog with AXI interfaces
5. **Verify Agent**: Runs functional verification against golden reference
6. **Synth Agent**: Synthesizes design and reports timing/area metrics

## ⚠️ Current Limitations

**CRITICAL: ALG2SV generates RTL but does NOT perform real hardware verification!**

### What ALG2SV Currently Does:
- ✅ Generates synthesizable SystemVerilog RTL
- ✅ Performs software-based functional verification
- ✅ Estimates synthesis results (timing, area)
- ✅ Lints and analyzes code quality
- ✅ Validates against golden reference models

### What ALG2SV Does NOT Do (Without Vivado):
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

**Bottom Line**: ALG2SV is currently a **software-based RTL generation and estimation tool**. For production use, the generated RTL must be synthesized and tested on actual FPGA hardware separately.

## 🛠️ Vivado Integration (Real Hardware Synthesis)

ALG2SV now includes **optional Vivado CLI integration** for real FPGA synthesis and bitstream generation!

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

When Vivado is available, ALG2SV can perform:

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

### Supported FPGA Families

- **Xilinx 7 Series**: Artix-7, Kintex-7, Virtex-7
- **Xilinx UltraScale**: Kintex UltraScale, Virtex UltraScale
- **Xilinx Zynq**: Zynq-7000, Zynq UltraScale+

### Vivado Integration Architecture

```
Algorithm → RTL → Vivado TCL Script → Synthesis → Implementation → Bitstream
     ↓         ↓           ↓              ↓           ↓             ↓
   Python    .sv        .tcl script    .dcp        .bit         .bit file
   code      files      generation    results    generation    + reports
```

### TCL Script Generation

ALG2SV automatically generates comprehensive TCL scripts that:

1. Create Vivado projects
2. Add RTL sources and constraints
3. Run synthesis with timing optimization
4. Perform place & route
5. Generate utilization/timing reports
6. Create bitstreams for programming

### Hardware Verification Pipeline

With Vivado integration, ALG2SV becomes a **complete hardware development platform**:

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
git clone https://github.com/your-repo/alg2sv.git
cd alg2sv

# Install in development mode (recommended for contributors)
pip install -e .
```

#### Environment Setup

Set your OpenAI API key:
```bash
export OPENAI_API_KEY="your-openai-api-key-here"
```


### 2. Test with Example Algorithm

```bash
# Run the BPF16 filter example
alg2sv test_algorithms/bpf16_bundle.txt --verbose --workspace-info

# Or use the module directly
python -m alg2sv.cli test_algorithms/bpf16_bundle.txt --verbose
```

### 3. Input Algorithm Format

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

### 4. Run Custom Algorithms

```bash
# Run with your own bundle file
alg2sv my_algorithm_bundle.txt --output results.json --extract-rtl rtl_output/

# Run with inline bundle
alg2sv --bundle "$(cat my_algorithm_bundle.txt)" --verbose
```

### 5. Testing the Pipeline

### Automated Testing

```bash
# Run the complete pipeline test
python -m pytest tests/ -v

# Or run individual components
python -c "
from alg2sv.pipeline import run_pipeline_sync
from alg2sv.test_algorithms.bpf16_bundle import get_bundle

result = run_pipeline_sync(get_bundle())
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
from alg2sv.agents import SpecAgent
from alg2sv.workspace import ingest_from_bundle

# Create agent and test
agent = SpecAgent()
# ... test agent functionality
```

### End-to-End Pipeline Test
```bash
# Full pipeline with verbose output
alg2sv test_algorithms/bpf16_bundle.txt \
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
/Users/westonvoglesonger/ALG2SV/
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
