"""
Test vectors for Complex Adaptive Filter with Kalman-like State Estimation

This file contains comprehensive test vectors that will challenge ARDA's ability to:
- Handle complex mathematical operations
- Process multiple data streams
- Manage state transitions
- Handle edge cases and boundary conditions
"""

import numpy as np
import json
from typing import List, Dict, Any


def generate_complex_test_vectors() -> Dict[str, Any]:
    """
    Generate comprehensive test vectors for the complex adaptive filter.
    
    Returns:
        Dictionary containing various test scenarios
    """
    
    # Test Case 1: Mixed Frequency Signal
    t1 = np.linspace(0, 2, 200)
    mixed_signal = (np.sin(2 * np.pi * 0.5 * t1) + 
                   0.5 * np.sin(2 * np.pi * 2.0 * t1) + 
                   0.3 * np.sin(2 * np.pi * 5.0 * t1) + 
                   0.1 * np.random.normal(0, 1, 200))
    
    # Test Case 2: Chirp Signal (frequency sweep)
    t2 = np.linspace(0, 4, 400)
    chirp_signal = np.sin(2 * np.pi * (0.1 + 0.2 * t2) * t2)
    
    # Test Case 3: Impulse Response
    impulse_signal = np.zeros(100)
    impulse_signal[25] = 1.0
    impulse_signal[50] = -0.5
    impulse_signal[75] = 0.3
    
    # Test Case 4: Step Response
    step_signal = np.concatenate([
        np.zeros(50),
        np.ones(100),
        np.zeros(50)
    ])
    
    # Test Case 5: Random Noise
    noise_signal = np.random.normal(0, 0.5, 300)
    
    # Test Case 6: Sparse Signal (many zeros)
    sparse_signal = np.zeros(200)
    sparse_indices = [20, 45, 67, 89, 123, 156, 178]
    for idx in sparse_indices:
        sparse_signal[idx] = np.random.uniform(-1, 1)
    
    # Test Case 7: High Dynamic Range
    t7 = np.linspace(0, 1, 150)
    high_dr_signal = np.concatenate([
        0.001 * np.sin(2 * np.pi * 10 * t7[:50]),  # Low amplitude
        10.0 * np.sin(2 * np.pi * 10 * t7[50:100]),  # High amplitude
        0.1 * np.sin(2 * np.pi * 10 * t7[100:])  # Medium amplitude
    ])
    
    # Test Case 8: Edge Cases
    edge_cases = {
        "zeros": np.zeros(50).tolist(),
        "ones": np.ones(50).tolist(),
        "alternating": [1, -1] * 25,
        "single_spike": [0] * 49 + [1.0] + [0] * 50,
        "overflow_values": [1e6, -1e6, 0, 1e-6, -1e-6] * 20
    }
    
    # Test Case 9: Complex Parameter Sets
    parameter_sets = [
        {
            "name": "conservative",
            "filter_length": 16,
            "state_dim": 4,
            "learning_rate": 0.001,
            "adaptation_threshold": 0.01
        },
        {
            "name": "aggressive", 
            "filter_length": 64,
            "state_dim": 12,
            "learning_rate": 0.05,
            "adaptation_threshold": 0.2
        },
        {
            "name": "balanced",
            "filter_length": 32,
            "state_dim": 8,
            "learning_rate": 0.01,
            "adaptation_threshold": 0.1
        },
        {
            "name": "minimal",
            "filter_length": 8,
            "state_dim": 2,
            "learning_rate": 0.005,
            "adaptation_threshold": 0.05
        }
    ]
    
    # Test Case 10: Expected Results (Golden Reference)
    # These are computed using the reference implementation
    expected_results = {
        "mixed_signal": {
            "final_snr_range": [15.0, 25.0],
            "mean_error_range": [0.01, 0.1],
            "convergence_expected": True,
            "adaptation_events_range": [50, 150]
        },
        "chirp_signal": {
            "final_snr_range": [10.0, 20.0],
            "mean_error_range": [0.05, 0.2],
            "convergence_expected": False,  # Chirp doesn't converge
            "adaptation_events_range": [100, 300]
        },
        "impulse_signal": {
            "final_snr_range": [5.0, 15.0],
            "mean_error_range": [0.1, 0.5],
            "convergence_expected": True,
            "adaptation_events_range": [3, 10]
        }
    }
    
    # Compile all test vectors
    test_vectors = {
        "test_cases": {
            "mixed_frequency": {
                "input": mixed_signal.tolist(),
                "description": "Mixed frequency signal with noise",
                "expected_behavior": "Should adapt to multiple frequencies"
            },
            "chirp_signal": {
                "input": chirp_signal.tolist(),
                "description": "Frequency sweep signal",
                "expected_behavior": "Should track changing frequency"
            },
            "impulse_response": {
                "input": impulse_signal.tolist(),
                "description": "Impulse response test",
                "expected_behavior": "Should show transient response"
            },
            "step_response": {
                "input": step_signal.tolist(),
                "description": "Step response test",
                "expected_behavior": "Should adapt to step change"
            },
            "noise_only": {
                "input": noise_signal.tolist(),
                "description": "Pure noise input",
                "expected_behavior": "Should suppress noise"
            },
            "sparse_signal": {
                "input": sparse_signal.tolist(),
                "description": "Sparse signal with few non-zero values",
                "expected_behavior": "Should handle sparse data efficiently"
            },
            "high_dynamic_range": {
                "input": high_dr_signal.tolist(),
                "description": "Signal with high dynamic range",
                "expected_behavior": "Should handle amplitude variations"
            }
        },
        "edge_cases": edge_cases,
        "parameter_sets": parameter_sets,
        "expected_results": expected_results,
        "metadata": {
            "total_test_cases": 7,
            "total_edge_cases": 5,
            "parameter_combinations": 4,
            "description": "Comprehensive test suite for complex adaptive filter",
            "complexity_level": "high",
            "challenges": [
                "Multiple mathematical operations",
                "State management",
                "Adaptive behavior",
                "Nonlinear transformations",
                "Matrix operations",
                "Conditional logic",
                "Memory management"
            ]
        }
    }
    
    return test_vectors


def create_validation_vectors() -> Dict[str, Any]:
    """
    Create validation vectors for algorithm correctness verification.
    
    Returns:
        Dictionary containing validation test cases
    """
    
    # Known input-output pairs for validation
    validation_cases = {
        "unit_impulse": {
            "input": [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            "expected_output_range": [-0.5, 0.5],  # Should be small
            "tolerance": 0.1
        },
        "dc_signal": {
            "input": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            "expected_output_range": [0.8, 1.2],  # Should track DC
            "tolerance": 0.2
        },
        "sinusoid": {
            "input": [0, 0.5, 1, 0.5, 0, -0.5, -1, -0.5, 0, 0.5],
            "expected_output_range": [-1.2, 1.2],  # Should track sinusoid
            "tolerance": 0.3
        }
    }
    
    # Performance benchmarks
    performance_benchmarks = {
        "convergence_time": {
            "max_samples": 1000,
            "tolerance": 0.01,
            "description": "Algorithm should converge within 1000 samples"
        },
        "snr_improvement": {
            "min_improvement": 3.0,  # dB
            "description": "Should improve SNR by at least 3dB"
        },
        "stability": {
            "max_coefficient_variation": 0.1,
            "description": "Coefficients should remain stable"
        }
    }
    
    return {
        "validation_cases": validation_cases,
        "performance_benchmarks": performance_benchmarks
    }


if __name__ == "__main__":
    # Generate test vectors
    test_vectors = generate_complex_test_vectors()
    validation_vectors = create_validation_vectors()
    
    # Combine all test data
    all_test_data = {
        "test_vectors": test_vectors,
        "validation_vectors": validation_vectors,
        "generation_info": {
            "timestamp": "2024-01-01T00:00:00Z",
            "version": "1.0",
            "description": "Complex adaptive filter test vectors"
        }
    }
    
    # Save to file
    with open("complex_adaptive_filter_test_vectors.json", "w") as f:
        json.dump(all_test_data, f, indent=2)
    
    print("Test vectors generated successfully!")
    print(f"Total test cases: {test_vectors['metadata']['total_test_cases']}")
    print(f"Edge cases: {len(test_vectors['edge_cases'])}")
    print(f"Parameter sets: {len(test_vectors['parameter_sets'])}")
    print(f"Validation cases: {len(validation_vectors['validation_cases'])}")
