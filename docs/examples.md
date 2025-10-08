# ARDA Examples

Practical examples showing how to use ARDA to convert various types of algorithms to SystemVerilog RTL.

## Table of Contents

- [Basic FIR Filter](#basic-fir-filter)
- [Advanced DSP Pipeline](#advanced-dsp-pipeline)
- [Multi-Stage Algorithm](#multi-stage-algorithm)
- [Custom Algorithm Types](#custom-algorithm-types)
- [Performance Optimization](#performance-optimization)
- [Hardware Integration](#hardware-integration)

## Basic FIR Filter

### Algorithm Implementation

```python
# fir_filter.py
import numpy as np

class StreamingFIR:
    """5-tap symmetric FIR filter for streaming applications."""

    def __init__(self, coeffs=None):
        # Default coefficients (low-pass filter)
        if coeffs is None:
            coeffs = [0.1, 0.2, 0.3, 0.2, 0.1]

        self.coeffs = np.array(coeffs, dtype=np.float64)
        self.history = np.zeros(len(coeffs), dtype=np.float64)

    def step(self, input_sample):
        """Process one input sample and return one output sample."""
        # Shift history (newest sample at index 0)
        self.history = np.roll(self.history, 1)
        self.history[0] = input_sample

        # Compute FIR output
        output = np.sum(self.history * self.coeffs)
        return float(output)

    def reset(self):
        """Reset filter state."""
        self.history.fill(0)

# Example usage
if __name__ == "__main__":
    fir = StreamingFIR()
    test_input = 1.0
    output = fir.step(test_input)
    print(f"Input: {test_input}, Output: {output}")
```

### Bundle Creation

```bash
# Create bundle from the Python file
arda --create-bundle fir_filter.py fir_filter_bundle.txt

# Bundle content preview
head -10 fir_filter_bundle.txt
```

### RTL Generation

```bash
# Generate RTL with AI agents
arda fir_filter_bundle.txt --verbose --agent-runner openai --extract-rtl rtl_output/

# Check results
echo "Generated RTL files:"
ls -la rtl_output/rtl/
echo
echo "Synthesis results:"
cat rtl_output/synth_results.json | jq '.fmax_mhz, .lut_usage, .timing_met'
```

### Expected Results

```
✅ spec: Generated hardware contract (100MHz target, Q1.15 fixed-point)
✅ quant: Applied fixed-point quantization with error analysis
✅ microarch: Designed 4-stage pipeline with 3 DSPs
✅ rtl: Generated symmetric FIR with AXI-Stream interfaces
✅ static_checks: Passed linting (95% quality score)
✅ verification: 100% functional verification (1024 tests)
✅ synth: Achieved 102.5MHz (exceeding 100MHz target)
✅ evaluate: Overall score 94.5%
```

## Advanced DSP Pipeline

### Complex Algorithm with Multiple Stages

```python
# dsp_pipeline.py
import numpy as np

class AdvancedDSPPipeline:
    """Multi-stage DSP pipeline with preprocessing and postprocessing."""

    def __init__(self, filter_coeffs, gain=1.0):
        self.filter = StreamingFIR(filter_coeffs)
        self.gain = gain
        self.dc_blocker = DCBlocker()
        self.output_scaler = OutputScaler()

    def step(self, input_sample):
        """Complete DSP pipeline processing."""
        # Stage 1: DC blocking
        dc_blocked = self.dc_blocker.step(input_sample)

        # Stage 2: FIR filtering
        filtered = self.filter.step(dc_blocked)

        # Stage 3: Gain adjustment
        gained = filtered * self.gain

        # Stage 4: Output scaling
        scaled = self.output_scaler.step(gained)

        return scaled

class DCBlocker:
    """Simple DC blocking filter."""

    def __init__(self, alpha=0.95):
        self.alpha = alpha
        self.previous_input = 0.0
        self.previous_output = 0.0

    def step(self, x):
        # High-pass filter: y[n] = x[n] - x[n-1] + alpha * y[n-1]
        output = x - self.previous_input + self.alpha * self.previous_output

        self.previous_input = x
        self.previous_output = output

        return output

class OutputScaler:
    """Output scaling and clipping."""

    def __init__(self, scale_factor=0.5, clip_min=-1.0, clip_max=1.0):
        self.scale_factor = scale_factor
        self.clip_min = clip_min
        self.clip_max = clip_max

    def step(self, x):
        # Scale and clip output
        scaled = x * self.scale_factor
        clipped = max(self.clip_min, min(self.clip_max, scaled))
        return clipped
```

### Pipeline Execution

```bash
# Create bundle
arda --create-bundle dsp_pipeline.py dsp_bundle.txt

# Run with performance optimization
arda dsp_bundle.txt \
  --verbose \
  --agent-runner openai \
  --target-clock 150MHz \
  --max-luts 5000 \
  --extract-rtl advanced_dsp_rtl/

# Results will show multi-stage pipeline optimization
```

## Multi-Stage Algorithm

### Algorithm with Complex State Management

```python
# multi_stage_processor.py
import numpy as np

class MultiStageProcessor:
    """Processor with multiple internal stages and state."""

    def __init__(self, window_size=16, threshold=0.5):
        self.window_size = window_size
        self.threshold = threshold

        # Stage 1: Input buffering
        self.input_buffer = []

        # Stage 2: Window processing
        self.window_buffer = np.zeros(window_size)

        # Stage 3: Feature extraction
        self.feature_history = []

        # Stage 4: Decision making
        self.decision_state = "IDLE"

    def step(self, input_sample):
        """Process one sample through all stages."""
        # Stage 1: Buffer input
        self.input_buffer.append(input_sample)
        if len(self.input_buffer) > self.window_size:
            self.input_buffer.pop(0)

        # Stage 2: Update sliding window
        if len(self.input_buffer) == self.window_size:
            self.window_buffer = np.array(self.input_buffer)

        # Stage 3: Extract features (when window is full)
        if len(self.window_buffer) == self.window_size:
            features = self._extract_features(self.window_buffer)
            self.feature_history.append(features)

            # Keep only recent features
            if len(self.feature_history) > 10:
                self.feature_history.pop(0)

        # Stage 4: Make decision based on features
        decision = self._make_decision()

        return {
            "processed_sample": input_sample * 0.8,  # Simple processing
            "window_full": len(self.window_buffer) == self.window_size,
            "decision": decision,
            "feature_count": len(self.feature_history)
        }

    def _extract_features(self, window):
        """Extract features from window."""
        return {
            "mean": np.mean(window),
            "std": np.std(window),
            "max": np.max(window),
            "min": np.min(window),
            "energy": np.sum(window ** 2)
        }

    def _make_decision(self):
        """Make decision based on feature history."""
        if len(self.feature_history) < 3:
            return "INSUFFICIENT_DATA"

        recent_features = self.feature_history[-3:]

        # Simple decision logic
        avg_energy = np.mean([f["energy"] for f in recent_features])

        if avg_energy > self.threshold:
            return "HIGH_ACTIVITY"
        else:
            return "NORMAL"
```

### Advanced Usage

```bash
# Create bundle with metadata
arda --create-bundle multi_stage_processor.py complex_bundle.txt

# Run with detailed analysis
arda complex_bundle.txt \
  --verbose \
  --agent-runner openai \
  --output detailed_results.json \
  --extract-rtl multi_stage_rtl/

# The AI agents will analyze the multi-stage nature and design
# appropriate pipelining and state management in hardware
```

## Custom Algorithm Types

### Matrix Operations

```python
# matrix_processor.py
import numpy as np

class MatrixProcessor:
    """Hardware-accelerated matrix operations."""

    def __init__(self, matrix_size=4):
        self.size = matrix_size
        self.matrix_a = np.eye(matrix_size)  # Identity matrix
        self.matrix_b = np.eye(matrix_size)  # Identity matrix

    def step(self, input_vector):
        """Process input vector through matrix operations."""
        # Convert to numpy array
        vec = np.array(input_vector, dtype=np.float64)

        # Matrix-vector multiplication: result = A * B * input
        temp = self.matrix_a @ vec
        result = self.matrix_b @ temp

        return result.tolist()

    def update_matrix_a(self, new_matrix):
        """Update matrix A (expensive operation)."""
        self.matrix_a = np.array(new_matrix, dtype=np.float64)

    def update_matrix_b(self, new_matrix):
        """Update matrix B (expensive operation)."""
        self.matrix_b = np.array(new_matrix, dtype=np.float64)
```

### FFT Implementation

```python
# fft_processor.py
import numpy as np

class FFTProcessor:
    """Streaming FFT processor for frequency domain analysis."""

    def __init__(self, fft_size=256):
        self.fft_size = fft_size
        self.input_buffer = np.zeros(fft_size, dtype=np.complex128)
        self.output_buffer = np.zeros(fft_size, dtype=np.complex128)

    def step(self, real_sample, imag_sample=0.0):
        """Process one complex sample."""
        # Add to input buffer
        sample = complex(real_sample, imag_sample)
        self.input_buffer = np.roll(self.input_buffer, 1)
        self.input_buffer[0] = sample

        # Perform FFT when buffer is full
        if np.all(self.input_buffer != 0):
            self.output_buffer = np.fft.fft(self.input_buffer)

        # Return magnitude of first bin (example)
        return abs(self.output_buffer[0])
```

## Performance Optimization

### High-Performance Algorithm

```python
# high_perf_algorithm.py
import numpy as np

class HighPerformanceDSP:
    """Optimized DSP algorithm for high-throughput applications."""

    def __init__(self, num_channels=8, filter_order=32):
        self.num_channels = num_channels
        self.filter_order = filter_order

        # Pre-allocate arrays for performance
        self.channel_buffers = [
            np.zeros(filter_order) for _ in range(num_channels)
        ]
        self.coefficients = np.random.randn(filter_order) * 0.1

    def step(self, input_samples):
        """Process multiple channels in parallel."""
        if len(input_samples) != self.num_channels:
            raise ValueError(f"Expected {self.num_channels} samples")

        outputs = []

        for i, sample in enumerate(input_samples):
            # Update buffer
            buffer = self.channel_buffers[i]
            buffer = np.roll(buffer, 1)
            buffer[0] = sample
            self.channel_buffers[i] = buffer

            # Compute filtered output
            output = np.sum(buffer * self.coefficients)
            outputs.append(float(output))

        return outputs
```

### Usage with Performance Targets

```bash
# Create bundle
arda --create-bundle high_perf_algorithm.py perf_bundle.txt

# Run with specific performance targets
arda perf_bundle.txt \
  --verbose \
  --agent-runner openai \
  --target-clock 250MHz \
  --throughput 8 \
  --max-luts 15000 \
  --max-dsps 64 \
  --extract-rtl high_perf_rtl/

# Results will show optimized multi-channel architecture
```

## Hardware Integration

### FPGA Board Interface

```python
# fpga_interface.py
import numpy as np

class FPGAInterface:
    """Interface for FPGA board communication."""

    def __init__(self, pcie_address="0000:01:00.0"):
        self.pcie_address = pcie_address
        self.dma_buffer_size = 4096

    def step(self, input_data):
        """Send data to FPGA and receive processed results."""
        # This would interface with actual FPGA hardware
        # For demonstration, simulate processing delay

        # Simulate FPGA processing time
        processed_data = input_data * 0.95  # Simple attenuation

        return {
            "input_samples": input_data,
            "processed_samples": processed_data,
            "processing_time_ms": 1.2,
            "throughput_mbps": 800.0
        }
```

### Real-Time Processing

```bash
# Create bundle for hardware interface
arda --create-bundle fpga_interface.py hw_bundle.txt

# Run with hardware integration focus
arda hw_bundle.txt \
  --agent-runner openai \
  --synthesis-backend vivado \
  --fpga-family xc7a200t \
  --extract-rtl hardware_interface_rtl/
```

## Algorithm Bundle Examples

### Minimal Algorithm

```text
path=minimal_algo.py
# Simple streaming algorithm
class SimpleAlgo:
    def step(self, x):
        return x * 0.5
```

### Rich Metadata Bundle

```text
path=dsp_algorithm.py
# Advanced DSP Algorithm: Adaptive FIR Filter
#
# Hardware Requirements:
# - Target clock: 200MHz
# - Input format: 16-bit signed integers (Q1.15)
# - Output format: 16-bit signed integers (Q1.15)
# - Resource budget: 10k LUTs, 5k FFs, 20 DSPs
# - Interface: AXI-Stream ready/valid
#
# Algorithm Description:
# Implements an adaptive FIR filter that adjusts coefficients
# based on input signal characteristics for optimal filtering.

import numpy as np

class AdaptiveFIR:
    def __init__(self, num_taps=16, adaptation_rate=0.01):
        self.num_taps = num_taps
        self.adaptation_rate = adaptation_rate

        # Initialize filter state
        self.coefficients = np.zeros(num_taps)
        self.history = np.zeros(num_taps)
        self.error_history = []

    def step(self, input_sample, desired_output=None):
        """Process one sample with optional adaptation."""
        # Shift input history
        self.history = np.roll(self.history, 1)
        self.history[0] = input_sample

        # Compute filter output
        output = np.sum(self.history * self.coefficients)

        # Adaptive algorithm (LMS)
        if desired_output is not None:
            error = desired_output - output
            self.error_history.append(error)

            # Update coefficients
            gradient = error * self.history * self.adaptation_rate
            self.coefficients += gradient

            # Normalize coefficients
            coeff_sum = np.sum(np.abs(self.coefficients))
            if coeff_sum > 0:
                self.coefficients /= coeff_sum

        return float(output)
```

## Command Examples

### Development Workflow

```bash
# 1. Create algorithm
echo "Creating algorithm file..."
cat > my_algorithm.py << 'EOF'
class MyAlgorithm:
    def step(self, x):
        return x * 0.7
EOF

# 2. Create bundle
arda --create-bundle my_algorithm.py algorithm_bundle.txt

# 3. Test with deterministic agents
arda algorithm_bundle.txt --agent-runner deterministic --verbose

# 4. Generate RTL with AI
arda algorithm_bundle.txt --agent-runner openai --extract-rtl rtl_output/

# 5. Check results
echo "RTL files generated:"
ls rtl_output/rtl/
echo
echo "Synthesis results:"
cat rtl_output/synth_results.json | jq '.'
```

### Production Deployment

```bash
# High-performance algorithm processing
arda production_algorithm.txt \
  --agent-runner openai \
  --synthesis-backend vivado \
  --fpga-family xc7a200t \
  --target-clock 300MHz \
  --max-luts 20000 \
  --extract-rtl production_rtl/ \
  --output production_results.json

# Verify deployment readiness
echo "Deployment checklist:"
echo "✅ RTL generated: $(ls production_rtl/rtl/ | wc -l) files"
echo "✅ Timing met: $(cat production_results.json | jq '.synth.timing_met')"
echo "✅ Resources within budget: $(cat production_results.json | jq '.synth.lut_usage < 20000')"
```

### Batch Processing

```bash
# Process multiple algorithms
for algo in algorithms/*.py; do
    bundle=$(basename $algo .py)_bundle.txt
    arda --create-bundle $algo $bundle
    arda $bundle --extract-rtl rtl_output/$(basename $algo .py)/
done

# Generate summary report
echo "Batch processing complete:"
find rtl_output/ -name "*.sv" | wc -l | xargs echo "RTL files generated:"
```

## Example Output Analysis

### Successful FIR Filter Generation

```
Pipeline Stages:
✅ spec: Algorithm analysis complete (5-tap FIR, 100MHz target)
✅ quant: Fixed-point conversion (Q1.15, SNR=85dB)
✅ microarch: Pipeline design (4 stages, 3 DSPs)
✅ rtl: RTL generation complete (symmetric optimization)
✅ static_checks: Quality verification (95% score)
✅ verification: Functional testing (100% pass rate)
✅ synth: FPGA synthesis (102.5MHz achieved)
✅ evaluate: Performance evaluation (94.5% overall score)

Hardware Results:
- Clock frequency: 102.5MHz (target: 100MHz) ✅
- Resource usage: 450 LUTs, 520 FFs, 3 DSPs
- Power consumption: 120mW
- Verification: All 1024 test vectors passed
```

### Optimization Example

When synthesis doesn't meet timing targets, ARDA automatically suggests improvements:

```
Initial synthesis: 85MHz (target: 100MHz) ❌

Feedback analysis:
- Increase pipeline depth from 4 to 6 stages
- Add register balancing for critical paths
- Optimize DSP placement

Retry synthesis: 105MHz achieved ✅

Final results:
- Clock frequency: 105MHz (5% over target)
- Resource usage: 480 LUTs (+7% from baseline)
- Quality score: 96.2% (improved from 91.5%)
```

## Algorithm Patterns

### Recognized Patterns

ARDA automatically detects and optimizes for common algorithm patterns:

#### Streaming Filters
- FIR filters (tapped delay lines)
- IIR filters (recursive structures)
- Adaptive filters (coefficient updating)

#### Signal Processing
- FFT processors (butterfly networks)
- Matrix operations (parallel processing)
- Convolution engines (sliding window)

#### Control Systems
- PID controllers (state machines)
- State machines (sequential logic)
- Feedback loops (recursive computation)

### Pattern-Specific Optimizations

**FIR Filters:**
- Symmetric coefficient exploitation (fewer multipliers)
- Tapped delay line optimization
- Parallel multiplier implementation

**Matrix Operations:**
- Systolic array architectures
- Parallel processing units
- Memory bandwidth optimization

**Control Systems:**
- State encoding optimization
- Transition logic minimization
- Timing constraint satisfaction

## Performance Benchmarks

### Algorithm Complexity vs Performance

| Algorithm Type | Input Size | RTL Generation | Synthesis Time | Achieved Freq |
|----------------|------------|-----------------|----------------|---------------|
| Simple FIR | 5 taps | 15s | 2min | 102MHz |
| Complex FIR | 32 taps | 45s | 5min | 95MHz |
| Matrix Ops | 4x4 | 25s | 3min | 98MHz |
| FFT | 256-point | 60s | 8min | 85MHz |

### Resource Utilization

| Algorithm | LUTs | FFs | DSPs | BRAMs | Power (mW) |
|-----------|------|-----|------|-------|------------|
| FIR-5 | 450 | 520 | 3 | 0 | 120 |
| FIR-32 | 2,800 | 3,200 | 16 | 0 | 450 |
| Matrix-4x4 | 1,200 | 1,800 | 8 | 2 | 280 |
| FFT-256 | 8,500 | 12,000 | 32 | 8 | 850 |

## Best Practices

### Algorithm Design for Hardware

1. **Use streaming interfaces** with `step()` methods
2. **Minimize state** between processing steps
3. **Consider fixed-point arithmetic** early in design
4. **Document performance requirements** in comments
5. **Test with representative data** before RTL generation

### Bundle Organization

1. **Single responsibility** per file
2. **Clear algorithm descriptions** in comments
3. **Performance specifications** in metadata
4. **Interface documentation** for complex algorithms

### Optimization Strategies

1. **Start with moderate targets** (100-150MHz)
2. **Use feedback iterations** for timing optimization
3. **Monitor resource utilization** trends
4. **Consider parallel processing** for high-throughput applications

## Troubleshooting Examples

### Common Issues and Solutions

#### Algorithm Not Detected
```
Warning: No algorithm patterns detected
```
**Solution:** Add `step()` method or use class names with "Filter"/"Algorithm"

#### Timing Not Met
```
Synthesis: 85MHz achieved (target: 100MHz)
```
**Solution:** The feedback agent will suggest pipeline adjustments

#### Resource Overutilization
```
LUT usage: 12,000 (budget: 10,000)
```
**Solution:** Review microarchitecture decisions or adjust resource budget

## Integration Examples

### Vivado Integration

```bash
# Generate RTL for Xilinx FPGA
arda algorithm.txt \
  --synthesis-backend vivado \
  --fpga-family xc7a200t \
  --target-clock 200MHz \
  --extract-rtl vivado_project/rtl/

# Import into Vivado project
# Add generated RTL files to Vivado sources
# Run synthesis and implementation
```

### Custom FPGA Board

```bash
# Generate RTL for custom board
arda algorithm.txt \
  --agent-runner openai \
  --custom-constraints board_constraints.xdc \
  --extract-rtl custom_board_rtl/
```

## Algorithm Gallery

### FIR Filter Family

```python
# Low-pass FIR
class LowPassFIR:
    def __init__(self):
        self.coeffs = [0.1, 0.2, 0.3, 0.2, 0.1]  # Low-pass
        self.history = [0] * 5

    def step(self, x):
        self.history = [x] + self.history[:-1]
        return sum(h * c for h, c in zip(self.history, self.coeffs))

# High-pass FIR
class HighPassFIR:
    def __init__(self):
        self.coeffs = [-0.1, -0.2, 0.6, -0.2, -0.1]  # High-pass
        self.history = [0] * 5

    def step(self, x):
        self.history = [x] + self.history[:-1]
        return sum(h * c for h, c in zip(self.history, self.coeffs))
```

### Signal Processing Pipeline

```python
# Complete DSP chain
class DSPSignalChain:
    def __init__(self):
        self.pre_filter = LowPassFIR()
        self.main_filter = BandPassFIR()
        self.post_filter = HighPassFIR()
        self.gain_stage = GainStage()

    def step(self, input_sample):
        # Pre-filtering
        pre_filtered = self.pre_filter.step(input_sample)

        # Main processing
        main_processed = self.main_filter.step(pre_filtered)

        # Post-filtering
        post_filtered = self.post_filter.step(main_processed)

        # Gain adjustment
        output = self.gain_stage.step(post_filtered)

        return output
```

## Advanced Features

### Custom Agent Behaviors

```python
# Custom RTL generation strategy
arda algorithm.txt \
  --custom-agent-config custom_rtl_agent.json \
  --agent-runner openai \
  --extract-rtl custom_rtl/
```

### Performance Profiling

```bash
# Profile pipeline performance
arda algorithm.txt --profile --output profile_results.json

# Analyze bottlenecks
python analyze_profile.py profile_results.json
```

### Batch Algorithm Processing

```python
# Process multiple algorithms
from ardagen.bundle_utils import create_bundle

algorithms = ["fir1.py", "fir2.py", "iir1.py"]
for algo in algorithms:
    bundle = create_bundle(algo)
    result = arda_process(bundle)
    save_results(algo, result)
```

This examples guide provides practical, real-world usage patterns for ARDA. Each example demonstrates different algorithm types and showcases the system's capabilities across various application domains.
