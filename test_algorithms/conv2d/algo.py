"""
Conv2D Neural Network Layer - 2D Convolution Implementation
Fixed-point arithmetic for FPGA acceleration

Architecture:
- Input: 8x8x3 feature map (H=8, W=8, C=3)
- Kernel: 3x3 convolution
- Output: 6x6x16 feature map (with padding, stride=1)
- Activation: ReLU
- Quantization: INT8 weights, INT16 accumulators, INT8 outputs
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import math

class FixedPoint:
    """Fixed-point number representation for neural networks."""
    def __init__(self, value: float, total_bits: int = 8, frac_bits: int = 6):
        self.total_bits = total_bits
        self.frac_bits = frac_bits
        self.int_bits = total_bits - frac_bits - 1  # -1 for sign bit

        # Convert to fixed-point
        scale = 2 ** frac_bits
        self.fp_value = int(value * scale)

        # Clamp to range
        max_val = (1 << (total_bits - 1)) - 1
        min_val = -(1 << (total_bits - 1))
        self.fp_value = max(min_val, min(max_val, self.fp_value))

    def to_float(self) -> float:
        scale = 2 ** self.frac_bits
        return self.fp_value / scale

    def __add__(self, other):
        result = self.fp_value + other.fp_value
        # Clamp result
        max_val = (1 << (self.total_bits - 1)) - 1
        min_val = -(1 << (self.total_bits - 1))
        result = max(min_val, min(max_val, result))
        return FixedPoint(result / (2 ** self.frac_bits), self.total_bits, self.frac_bits)

    def __mul__(self, other):
        # Fixed-point multiplication with proper scaling
        result = (self.fp_value * other.fp_value) >> self.frac_bits
        # Clamp result
        max_val = (1 << (self.total_bits - 1)) - 1
        min_val = -(1 << (self.total_bits - 1))
        result = max(min_val, min(max_val, result))
        return FixedPoint(result / (2 ** self.frac_bits), self.total_bits, self.frac_bits)

    def relu(self):
        """Apply ReLU activation function."""
        if self.fp_value < 0:
            return FixedPoint(0.0, self.total_bits, self.frac_bits)
        return FixedPoint(self.to_float(), self.total_bits, self.frac_bits)

def generate_conv2d_weights(input_channels: int, output_channels: int,
                          kernel_size: int, seed: int = 42) -> np.ndarray:
    """Generate random weights for Conv2D layer."""
    np.random.seed(seed)
    # Initialize with Xavier/Glorot initialization
    scale = math.sqrt(2.0 / (input_channels * kernel_size * kernel_size))
    weights = np.random.normal(0, scale, (output_channels, input_channels, kernel_size, kernel_size))
    return weights.astype(np.float32)

def conv2d_layer(input_fm: np.ndarray, weights: np.ndarray, bias: np.ndarray,
                stride: int = 1, padding: int = 1, activation: str = 'relu') -> np.ndarray:
    """
    Perform 2D convolution operation.

    Args:
        input_fm: Input feature map (H, W, C_in)
        weights: Convolution weights (C_out, C_in, K, K)
        bias: Bias terms (C_out,)
        stride: Convolution stride
        padding: Zero padding
        activation: Activation function ('relu' or 'none')

    Returns:
        Output feature map (H_out, W_out, C_out)
    """
    H, W, C_in = input_fm.shape
    C_out, C_in_weights, K, _ = weights.shape

    # Calculate output dimensions
    H_out = (H + 2 * padding - K) // stride + 1
    W_out = (W + 2 * padding - K) // stride + 1

    # Initialize output
    output = np.zeros((H_out, W_out, C_out), dtype=np.float32)

    # Add padding to input
    if padding > 0:
        input_padded = np.pad(input_fm, ((padding, padding), (padding, padding), (0, 0)), 'constant')
    else:
        input_padded = input_fm

    # Perform convolution
    for h_out in range(H_out):
        for w_out in range(W_out):
            for c_out in range(C_out):
                # Extract patch
                h_start = h_out * stride
                w_start = w_out * stride
                patch = input_padded[h_start:h_start+K, w_start:w_start+K, :]

                # Compute convolution for this output pixel
                conv_sum = bias[c_out]
                for c_in in range(C_in):
                    for kh in range(K):
                        for kw in range(K):
                            conv_sum += patch[kh, kw, c_in] * weights[c_out, c_in, kh, kw]

                # Apply activation
                if activation == 'relu':
                    conv_sum = max(0.0, conv_sum)

                output[h_out, w_out, c_out] = conv_sum

    return output

def conv2d_fixed_point(input_fm: np.ndarray, weights: np.ndarray, bias: np.ndarray,
                      fp_config: dict) -> np.ndarray:
    """
    Conv2D with fixed-point arithmetic.

    Args:
        input_fm: Float input feature map
        weights: Float weights
        bias: Float bias
        fp_config: Fixed-point configuration

    Returns:
        Fixed-point output feature map
    """
    # Convert to fixed-point
    input_fp = np.array([[[FixedPoint(val,
                                      fp_config['input_bits'],
                                      fp_config['input_frac_bits'])
                          for val in row] for row in channel]
                        for channel in input_fm])

    weights_fp = np.array([[[[FixedPoint(val,
                                          fp_config['weight_bits'],
                                          fp_config['weight_frac_bits'])
                             for val in kw_row] for kw_row in kh_row]
                           for kh_row in cin_row] for cin_row in weights])

    bias_fp = np.array([FixedPoint(val,
                                   fp_config['bias_bits'],
                                   fp_config['bias_frac_bits'])
                       for val in bias])

    H, W, C_in = input_fm.shape
    C_out, _, K, _ = weights.shape

    # Calculate output dimensions (same padding, stride=1)
    H_out = H
    W_out = W

    # Initialize output
    output_fp = np.zeros((H_out, W_out, C_out), dtype=object)

    # Perform fixed-point convolution
    for h_out in range(H_out):
        for w_out in range(W_out):
            for c_out in range(C_out):
                # Extract patch with padding
                h_start = max(0, h_out - 1)
                w_start = max(0, w_out - 1)
                h_end = min(H, h_out + 2)
                w_end = min(W, w_out + 2)

                # Compute convolution
                accumulator = bias_fp[c_out]
                for c_in in range(C_in):
                    for kh in range(max(0, h_out-1), min(H, h_out+2)):
                        for kw in range(max(0, w_out-1), min(W, w_out+2)):
                            # Calculate relative kernel position
                            kh_rel = kh - (h_out - 1)
                            kw_rel = kw - (w_out - 1)

                            if 0 <= kh_rel < K and 0 <= kw_rel < K:
                                input_val = input_fp[kh, kw, c_in]
                                weight_val = weights_fp[c_out, c_in, kh_rel, kw_rel]
                                accumulator = accumulator + (input_val * weight_val)

                # Apply ReLU
                output_fp[h_out, w_out, c_out] = accumulator.relu()

    # Convert back to float for return
    output_float = np.array([[[val.to_float() for val in row] for row in channel]
                           for channel in output_fp])

    return output_float

def conv2d_step_function(input_data: List[float], config: dict) -> List[float]:
    """
    Conv2D processing function for ALG2SV pipeline.

    Args:
        input_data: Flattened input feature map (8x8x3 = 192 elements)
        config: Configuration dictionary

    Returns:
        Flattened output feature map (6x6x16 = 576 elements)
    """
    # Reshape input to 8x8x3
    input_array = np.array(input_data).reshape(8, 8, 3)

    # Generate weights and bias (deterministic for testing)
    weights = generate_conv2d_weights(3, 16, 3, seed=42)
    bias = np.zeros(16, dtype=np.float32)  # Simple bias for testing

    # Get fixed-point config
    fp_config = config.get('fp_config', {
        'input_bits': 8, 'input_frac_bits': 6,
        'weight_bits': 8, 'weight_frac_bits': 6,
        'bias_bits': 16, 'bias_frac_bits': 12
    })

    # Perform convolution
    output = conv2d_fixed_point(input_array, weights, bias, fp_config)

    # Flatten output
    return output.flatten().tolist()

# Test function
def test_conv2d():
    """Test Conv2D implementation."""
    print("Testing Conv2D Layer...")

    # Create test input (8x8x3)
    np.random.seed(42)
    input_fm = np.random.randn(8, 8, 3).astype(np.float32)

    # Generate weights
    weights = generate_conv2d_weights(3, 16, 3, seed=42)
    bias = np.zeros(16, dtype=np.float32)

    # Test float version
    output_float = conv2d_layer(input_fm, weights, bias, stride=1, padding=1)

    print(f"Input shape: {input_fm.shape}")
    print(f"Output shape: {output_float.shape}")
    print(f"Weights shape: {weights.shape}")
    print(f"Output range: [{output_float.min():.3f}, {output_float.max():.3f}]")

    # Test fixed-point version
    fp_config = {
        'input_bits': 8, 'input_frac_bits': 6,
        'weight_bits': 8, 'weight_frac_bits': 6,
        'bias_bits': 16, 'bias_frac_bits': 12
    }
    output_fp = conv2d_fixed_point(input_fm, weights, bias, fp_config)

    print(f"Fixed-point output range: [{output_fp.min():.3f}, {output_fp.max():.3f}]")

    return output_float, output_fp

if __name__ == "__main__":
    test_conv2d()
