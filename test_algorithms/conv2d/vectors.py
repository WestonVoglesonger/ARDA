"""
Test vectors for Conv2D neural network layer validation
"""

import numpy as np
import json
from typing import Dict, List, Any

def generate_test_vectors() -> Dict[str, Any]:
    """Generate comprehensive test vectors for Conv2D validation."""

    test_vectors = {}

    # Test Case 1: Random input (normal case)
    np.random.seed(42)
    random_input = np.random.randn(8, 8, 3).astype(np.float32)
    test_vectors['random_input'] = {
        'input': random_input.flatten().tolist(),
        'description': 'Random normal distributed input',
        'expected_properties': ['non_zero_output', 'relu_activation']
    }

    # Test Case 2: Zero input (should produce bias-only output)
    zero_input = np.zeros((8, 8, 3), dtype=np.float32)
    test_vectors['zero_input'] = {
        'input': zero_input.flatten().tolist(),
        'description': 'All-zero input (tests bias handling)',
        'expected_properties': ['minimal_output', 'bias_only']
    }

    # Test Case 3: Saturated input (high values)
    saturated_input = np.full((8, 8, 3), 2.0, dtype=np.float32)
    test_vectors['saturated_input'] = {
        'input': saturated_input.flatten().tolist(),
        'description': 'High-value input (tests saturation handling)',
        'expected_properties': ['high_output', 'saturation_handling']
    }

    # Test Case 4: Edge detection pattern
    edge_input = np.zeros((8, 8, 3), dtype=np.float32)
    edge_input[2:6, 2:6, :] = 1.0  # Square in center
    test_vectors['edge_pattern'] = {
        'input': edge_input.flatten().tolist(),
        'description': 'Edge detection pattern (square in center)',
        'expected_properties': ['edge_response', 'spatial_features']
    }

    # Test Case 5: Checkerboard pattern
    checkerboard = np.zeros((8, 8, 3), dtype=np.float32)
    checkerboard[::2, ::2, :] = 1.0
    checkerboard[1::2, 1::2, :] = 1.0
    test_vectors['checkerboard'] = {
        'input': checkerboard.flatten().tolist(),
        'description': 'Checkerboard pattern for frequency testing',
        'expected_properties': ['frequency_response', 'alternating_output']
    }

    # Test Case 6: Gradient input
    gradient = np.zeros((8, 8, 3), dtype=np.float32)
    for i in range(8):
        for j in range(8):
            gradient[i, j, :] = (i + j) / 14.0  # 0 to 1 gradient
    test_vectors['gradient'] = {
        'input': gradient.flatten().tolist(),
        'description': 'Smooth gradient input',
        'expected_properties': ['smooth_response', 'gradient_preservation']
    }

    # Test Case 7: Single pixel activation
    single_pixel = np.zeros((8, 8, 3), dtype=np.float32)
    single_pixel[4, 4, 1] = 1.0  # Single pixel in center, green channel
    test_vectors['single_pixel'] = {
        'input': single_pixel.flatten().tolist(),
        'description': 'Single activated pixel',
        'expected_properties': ['localized_response', 'kernel_spread']
    }

    # Test Case 8: Noise input
    np.random.seed(123)
    noise_input = np.random.normal(0, 0.1, (8, 8, 3)).astype(np.float32)
    test_vectors['noise_input'] = {
        'input': noise_input.flatten().tolist(),
        'description': 'Low-amplitude noise input',
        'expected_properties': ['noise_suppression', 'low_output']
    }

    return test_vectors

def generate_golden_reference(input_signal: List[float]) -> List[float]:
    """Generate golden reference using numpy/scipy convolution."""
    # This would use scipy.signal.convolve2d for accurate reference
    # For now, we'll use a simplified implementation

    # Reshape input
    input_array = np.array(input_signal).reshape(8, 8, 3)

    # Simple convolution with fixed kernel for testing
    # In practice, this would match the exact algorithm implementation
    kernel = np.ones((3, 3, 3, 16)) * 0.1  # Simple kernel
    bias = np.zeros(16)

    # Simplified 2D convolution (would be more complex in real implementation)
    output = np.zeros((6, 6, 16))

    # Very basic convolution for testing
    for h in range(6):
        for w in range(6):
            for c_out in range(16):
                # Simple average of 3x3 region
                region = input_array[h:h+3, w:w+3, :]
                output[h, w, c_out] = np.mean(region) + bias[c_out]
                # Apply ReLU
                output[h, w, c_out] = max(0, output[h, w, c_out])

    return output.flatten().tolist()

def validate_conv2d_output(output: List[float], reference: List[float],
                          tolerance: float = 0.1) -> Dict[str, Any]:
    """Validate Conv2D output against golden reference."""

    if len(output) != len(reference):
        return {
            'valid': False,
            'error': f'Length mismatch: {len(output)} vs {len(reference)}'
        }

    output_array = np.array(output)
    ref_array = np.array(reference)

    # Calculate error metrics
    error = output_array - ref_array
    max_error = np.max(np.abs(error))
    rms_error = np.sqrt(np.mean(error**2))
    if np.max(np.abs(ref_array)) > 1e-12 and rms_error > 1e-12:
        snr_db = 20 * np.log10(np.max(np.abs(ref_array)) / rms_error)
    else:
        snr_db = 100.0  # Perfect SNR if no error or signal

    # Check if output is reasonable (not all zeros, has variation)
    output_variation = np.std(output_array)
    has_activation = np.any(output_array > 0.01)

    return {
        'valid': bool(max_error < tolerance and has_activation and output_variation > 0.001),
        'snr_db': float(snr_db),
        'max_error': float(max_error),
        'rms_error': float(rms_error),
        'output_std': float(output_variation),
        'has_activation': bool(has_activation),
        'tolerance_met': bool(max_error < tolerance)
    }

if __name__ == "__main__":
    # Generate and save test vectors
    test_vectors = generate_test_vectors()

    # Generate golden references for each test case
    for test_name, test_data in test_vectors.items():
        reference = generate_golden_reference(test_data['input'])
        test_vectors[test_name]['golden_reference'] = reference

        # Validate the golden reference against itself (should be perfect)
        validation = validate_conv2d_output(reference, reference)
        test_vectors[test_name]['self_validation'] = validation

    # Save to JSON
    with open('conv2d_test_vectors.json', 'w') as f:
        json.dump(test_vectors, f, indent=2)

    print(f"Generated {len(test_vectors)} test vectors for Conv2D")
    print("Test vectors saved to conv2d_test_vectors.json")
