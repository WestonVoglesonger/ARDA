"""
Test vectors for 256-point FFT validation
"""

import numpy as np
import json
from typing import Dict, List, Any

def generate_test_vectors() -> Dict[str, Any]:
    """Generate comprehensive test vectors for FFT validation."""

    N = 256
    test_vectors = {}

    # Test Case 1: DC signal (should have peak at frequency 0)
    dc_signal = np.ones(N)
    test_vectors['dc_signal'] = {
        'input': dc_signal.tolist(),
        'description': 'DC signal - all ones',
        'expected_peaks': [{'frequency': 0, 'magnitude': N}]
    }

    # Test Case 2: Single tone at frequency 10
    t = np.linspace(0, 1, N, endpoint=False)
    tone_10hz = np.sin(2 * np.pi * 10 * t)
    test_vectors['tone_10hz'] = {
        'input': tone_10hz.tolist(),
        'description': 'Single tone at frequency index 10',
        'expected_peaks': [{'frequency': 10, 'magnitude': N/2}]
    }

    # Test Case 3: Two tones
    tone_mixed = np.sin(2 * np.pi * 15 * t) + 0.7 * np.sin(2 * np.pi * 45 * t)
    test_vectors['two_tones'] = {
        'input': tone_mixed.tolist(),
        'description': 'Two tones at frequencies 15 and 45',
        'expected_peaks': [
            {'frequency': 15, 'magnitude': N/2},
            {'frequency': 45, 'magnitude': 0.7 * N/2}
        ]
    }

    # Test Case 4: Impulse response
    impulse = np.zeros(N)
    impulse[0] = 1.0
    test_vectors['impulse'] = {
        'input': impulse.tolist(),
        'description': 'Unit impulse at index 0',
        'expected_flat_spectrum': True
    }

    # Test Case 5: Nyquist frequency (fs/2)
    nyquist = np.sin(2 * np.pi * 128 * t)  # Frequency index 128
    test_vectors['nyquist'] = {
        'input': nyquist.tolist(),
        'description': 'Nyquist frequency (fs/2)',
        'expected_peaks': [{'frequency': 128, 'magnitude': N/2}]
    }

    # Test Case 6: White noise (should have flat spectrum)
    np.random.seed(42)  # For reproducible results
    noise = np.random.normal(0, 0.5, N)
    test_vectors['white_noise'] = {
        'input': noise.tolist(),
        'description': 'White Gaussian noise',
        'expected_flat_spectrum': True
    }

    # Test Case 7: Rectangular window
    rect_window = np.ones(N)
    rect_window[N//4:3*N//4] = 0.5
    test_vectors['rect_window'] = {
        'input': rect_window.tolist(),
        'description': 'Rectangular window function',
        'expected_sinc_response': True
    }

    # Test Case 8: Chirp signal (linear frequency sweep)
    chirp = np.sin(2 * np.pi * (10 + 0.1 * np.arange(N)) * t)
    test_vectors['chirp'] = {
        'input': chirp.tolist(),
        'description': 'Linear frequency chirp from 10 to 26 Hz',
        'expected_chirp_response': True
    }

    return test_vectors

def generate_golden_reference(input_signal: List[float]) -> List[float]:
    """Generate golden reference FFT using numpy."""
    complex_input = [complex(x, 0) for x in input_signal]
    fft_result = np.fft.fft(complex_input)

    # Return as interleaved real/imaginary
    output = []
    for sample in fft_result:
        output.extend([sample.real, sample.imag])

    return output

def validate_fft_output(output: List[float], reference: List[float],
                       tolerance: float = 1e-3) -> Dict[str, Any]:
    """Validate FFT output against golden reference."""

    if len(output) != len(reference):
        return {
            'valid': False,
            'error': f'Length mismatch: {len(output)} vs {len(reference)}'
        }

    # Calculate SNR
    output_complex = []
    ref_complex = []
    for i in range(0, len(output), 2):
        output_complex.append(complex(output[i], output[i+1]))
        ref_complex.append(complex(reference[i], reference[i+1]))

    # Compute error
    error = np.array(output_complex) - np.array(ref_complex)
    signal_power = np.mean(np.abs(np.array(ref_complex))**2)
    noise_power = np.mean(np.abs(error)**2)

    if signal_power > 1e-12 and noise_power > 1e-12:
        snr_db = 10 * np.log10(signal_power / noise_power)
    else:
        snr_db = 100.0  # Perfect SNR if no noise or signal

    max_error = np.max(np.abs(error))
    rms_error = np.sqrt(np.mean(np.abs(error)**2))

    return {
        'valid': bool(snr_db > 40),  # Require 40dB SNR
        'snr_db': float(snr_db),
        'max_error': float(max_error),
        'rms_error': float(rms_error),
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
        validation = validate_fft_output(reference, reference)
        test_vectors[test_name]['self_validation'] = validation

    # Save to JSON
    with open('fft256_test_vectors.json', 'w') as f:
        json.dump(test_vectors, f, indent=2)

    print(f"Generated {len(test_vectors)} test vectors for FFT256")
    print("Test vectors saved to fft256_test_vectors.json")
