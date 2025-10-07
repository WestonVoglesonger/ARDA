"""
256-Point FFT Algorithm - Cooley-Tukey Implementation
Complex fixed-point arithmetic for FPGA acceleration
"""

import numpy as np
from typing import List, Tuple, Optional
import math
import cmath

class ComplexFP:
    """Fixed-point complex number representation."""
    def __init__(self, real: float, imag: float, total_bits: int = 16, frac_bits: int = 12):
        self.total_bits = total_bits
        self.frac_bits = frac_bits
        self.int_bits = total_bits - frac_bits - 1  # -1 for sign bit

        # Convert to fixed-point
        scale = 2 ** frac_bits
        self.real_fp = int(real * scale)
        self.imag_fp = int(imag * scale)

        # Clamp to range
        max_val = (1 << (total_bits - 1)) - 1
        min_val = -(1 << (total_bits - 1))
        self.real_fp = max(min_val, min(max_val, self.real_fp))
        self.imag_fp = max(min_val, min(max_val, self.imag_fp))

    def to_float(self) -> complex:
        scale = 2 ** self.frac_bits
        return complex(self.real_fp / scale, self.imag_fp / scale)

    def __add__(self, other):
        return ComplexFP(
            (self.real_fp + other.real_fp) / (2 ** self.frac_bits),
            (self.imag_fp + other.imag_fp) / (2 ** self.frac_bits),
            self.total_bits, self.frac_bits
        )

    def __sub__(self, other):
        return ComplexFP(
            (self.real_fp - other.real_fp) / (2 ** self.frac_bits),
            (self.imag_fp - other.imag_fp) / (2 ** self.frac_bits),
            self.total_bits, self.frac_bits
        )

    def __mul__(self, other):
        # Complex multiplication: (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
        a, b = self.real_fp, self.imag_fp
        c, d = other.real_fp, other.imag_fp

        real_result = (a * c - b * d) >> self.frac_bits
        imag_result = (a * d + b * c) >> self.frac_bits

        return ComplexFP(
            real_result / (2 ** self.frac_bits),
            imag_result / (2 ** self.frac_bits),
            self.total_bits, self.frac_bits
        )

def generate_twiddle_factors(N: int, fp_config: dict) -> List[ComplexFP]:
    """Generate twiddle factors for FFT."""
    twiddles = []
    for k in range(N // 2):
        angle = -2 * math.pi * k / N
        twiddle = cmath.exp(1j * angle)
        twiddles.append(ComplexFP(
            twiddle.real, twiddle.imag,
            fp_config['total_bits'], fp_config['frac_bits']
        ))
    return twiddles

def bit_reverse_permutation(n: int, size: int) -> int:
    """Bit reversal for FFT input ordering."""
    result = 0
    for i in range(size):
        result = (result << 1) | (n & 1)
        n >>= 1
    return result

def fft_256(input_data: List[complex], fp_config: dict) -> List[complex]:
    """
    256-point FFT implementation using Cooley-Tukey algorithm.

    Args:
        input_data: 256 complex input samples
        fp_config: Fixed-point configuration

    Returns:
        256 complex frequency domain samples
    """
    N = 256
    stages = 8  # log2(256) = 8

    # Convert input to fixed-point complex numbers
    x = [ComplexFP(sample.real, sample.imag,
                   fp_config['total_bits'], fp_config['frac_bits'])
         for sample in input_data]

    # Bit reversal permutation
    for i in range(N):
        j = bit_reverse_permutation(i, stages)
        if j > i:
            x[i], x[j] = x[j], x[i]

    # FFT computation stages
    for stage in range(stages):
        span = 1 << stage  # 1, 2, 4, 8, 16, 32, 64, 128
        for start in range(0, N, 2 * span):
            # Twiddle factor for this stage
            k = start // (2 * span)
            twiddle = generate_twiddle_factors(2 * span, fp_config)[k]

            for offset in range(span):
                i = start + offset
                j = start + offset + span

                # Butterfly operation
                temp = x[j] * twiddle
                x[j] = x[i] - temp
                x[i] = x[i] + temp

    # Convert back to floating point
    return [sample.to_float() for sample in x]

def fft_step_function(input_samples: List[float], config: dict) -> List[float]:
    """
    FFT processing step function - processes 256 real samples into frequency domain.

    Args:
        input_samples: 256 real input samples
        config: Configuration dictionary with fp_config

    Returns:
        512 values: 256 real + 256 imaginary frequency components
    """
    if len(input_samples) != 256:
        raise ValueError("FFT requires exactly 256 input samples")

    # Convert real input to complex (imaginary part = 0)
    complex_input = [complex(sample, 0.0) for sample in input_samples]

    # Perform FFT
    fp_config = config.get('fp_config', {
        'total_bits': 16,
        'frac_bits': 12
    })

    result = fft_256(complex_input, fp_config)

    # Return as interleaved real/imaginary
    output = []
    for sample in result:
        output.extend([sample.real, sample.imag])

    return output

# Test function for validation
def test_fft():
    """Test FFT with known input."""
    # Generate test signal: sum of two sinusoids
    N = 256
    t = np.linspace(0, 1, N, endpoint=False)
    f1, f2 = 10, 40  # Hz
    signal = np.sin(2 * np.pi * f1 * t) + 0.5 * np.sin(2 * np.pi * f2 * t)

    fp_config = {'total_bits': 16, 'frac_bits': 12}

    # Perform FFT
    result = fft_256([complex(s, 0) for s in signal], fp_config)
    result_float = [r.to_float() for r in result]

    # Check that we get peaks at f1 and f2
    magnitudes = [abs(r) for r in result_float]

    # Find peaks (should be at indices f1 and f2)
    peak1_idx = np.argmax(magnitudes[:N//2])
    peak2_idx = np.argmax(magnitudes[f2-5:f2+5]) + f2 - 5

    print(f"Peak 1 at frequency index: {peak1_idx} (expected: {f1})")
    print(f"Peak 2 at frequency index: {peak2_idx} (expected: {f2})")

    return result_float

if __name__ == "__main__":
    test_fft()
