"""
Complex Adaptive Filter with Kalman-like State Estimation

This algorithm implements a sophisticated adaptive filtering system that combines:
- Multi-tap FIR filtering with adaptive coefficients
- State estimation using Kalman-like updates
- Nonlinear activation functions
- Real-time parameter adaptation
- Complex mathematical operations

This will test ARDA's ability to handle:
- Complex state machines
- Matrix operations
- Conditional logic
- Multiple data streams
- Nonlinear transformations
"""

import numpy as np
from typing import List, Tuple, Dict, Any
import math


class ComplexAdaptiveFilter:
    """
    A complex adaptive filter that combines FIR filtering with state estimation.
    
    This algorithm processes streaming data through multiple stages:
    1. Input preprocessing with normalization
    2. Multi-tap FIR filtering with adaptive coefficients
    3. State estimation using Kalman-like updates
    4. Nonlinear post-processing
    5. Output generation with feedback
    """
    
    def __init__(self, 
                 filter_length: int = 32,
                 state_dim: int = 8,
                 learning_rate: float = 0.01,
                 adaptation_threshold: float = 0.1):
        """
        Initialize the complex adaptive filter.
        
        Args:
            filter_length: Number of FIR filter taps
            state_dim: Dimension of state vector
            learning_rate: Adaptation rate for coefficients
            adaptation_threshold: Threshold for triggering adaptation
        """
        self.filter_length = filter_length
        self.state_dim = state_dim
        self.learning_rate = learning_rate
        self.adaptation_threshold = adaptation_threshold
        
        # Initialize filter coefficients (adaptive)
        self.coefficients = np.random.normal(0, 0.1, filter_length)
        self.coefficient_history = []
        
        # Initialize state vector and covariance
        self.state_vector = np.zeros(state_dim)
        self.state_covariance = np.eye(state_dim) * 0.1
        
        # Input/output buffers
        self.input_buffer = np.zeros(filter_length)
        self.output_buffer = np.zeros(filter_length)
        
        # Adaptation parameters
        self.error_history = []
        self.adaptation_active = False
        self.adaptation_counter = 0
        
        # Performance metrics
        self.snr_estimate = 0.0
        self.convergence_rate = 0.0
        self.stability_factor = 1.0
        
        # Nonlinear processing parameters
        self.sigmoid_gain = 2.0
        self.tanh_scaling = 1.5
        
    def _normalize_input(self, x: float) -> float:
        """Normalize input using adaptive scaling."""
        # Adaptive normalization based on recent statistics
        if len(self.error_history) > 10:
            recent_std = np.std(self.error_history[-10:])
            if recent_std > 0:
                return x / (recent_std * 2.0)
        return x
    
    def _apply_nonlinear_transform(self, x: float, transform_type: str = "sigmoid") -> float:
        """Apply nonlinear transformation to input."""
        if transform_type == "sigmoid":
            return 1.0 / (1.0 + math.exp(-self.sigmoid_gain * x))
        elif transform_type == "tanh":
            return math.tanh(self.tanh_scaling * x)
        elif transform_type == "relu":
            return max(0.0, x)
        else:
            return x
    
    def _update_state_estimation(self, measurement: float, innovation: float) -> None:
        """Update state vector using Kalman-like estimation."""
        # Innovation covariance
        innovation_var = abs(innovation) + 1e-6
        
        # Kalman gain (simplified)
        kalman_gain = self.state_covariance[0, 0] / (self.state_covariance[0, 0] + innovation_var)
        
        # State update
        state_innovation = kalman_gain * innovation
        self.state_vector[0] += state_innovation
        
        # Covariance update
        self.state_covariance[0, 0] *= (1.0 - kalman_gain)
        
        # Propagate state updates to other dimensions
        for i in range(1, self.state_dim):
            coupling_factor = 0.1 * math.exp(-i * 0.2)
            self.state_vector[i] += coupling_factor * state_innovation
            self.state_covariance[i, i] *= (1.0 - coupling_factor * kalman_gain)
    
    def _adapt_coefficients(self, error: float) -> None:
        """Adapt filter coefficients using LMS-like algorithm."""
        if not self.adaptation_active:
            return
            
        # LMS adaptation with momentum
        momentum = 0.9
        if len(self.coefficient_history) > 0:
            prev_coeffs = self.coefficient_history[-1]
            momentum_term = momentum * (self.coefficients - prev_coeffs)
        else:
            momentum_term = np.zeros_like(self.coefficients)
        
        # Gradient-based update
        gradient = -error * self.input_buffer
        self.coefficients += self.learning_rate * gradient + momentum_term
        
        # Coefficient constraints
        self.coefficients = np.clip(self.coefficients, -1.0, 1.0)
        
        # Store coefficient history
        self.coefficient_history.append(self.coefficients.copy())
        if len(self.coefficient_history) > 100:
            self.coefficient_history.pop(0)
    
    def _compute_performance_metrics(self) -> None:
        """Compute performance metrics for monitoring."""
        if len(self.error_history) > 20:
            recent_errors = self.error_history[-20:]
            
            # SNR estimation
            signal_power = np.var(self.output_buffer)
            noise_power = np.var(recent_errors)
            if noise_power > 0:
                self.snr_estimate = 10 * math.log10(signal_power / noise_power)
            
            # Convergence rate
            if len(self.error_history) > 40:
                old_errors = self.error_history[-40:-20]
                new_errors = self.error_history[-20:]
                self.convergence_rate = np.mean(np.abs(old_errors)) - np.mean(np.abs(new_errors))
            
            # Stability factor
            coeff_variation = np.std(self.coefficients)
            self.stability_factor = 1.0 / (1.0 + coeff_variation)
    
    def process_sample(self, input_sample: float) -> Dict[str, Any]:
        """
        Process a single input sample through the complex adaptive filter.
        
        Args:
            input_sample: Input sample to process
            
        Returns:
            Dictionary containing output and metadata
        """
        # Step 1: Input preprocessing
        normalized_input = self._normalize_input(input_sample)
        
        # Step 2: Update input buffer
        self.input_buffer = np.roll(self.input_buffer, 1)
        self.input_buffer[0] = normalized_input
        
        # Step 3: FIR filtering
        filtered_output = np.dot(self.coefficients, self.input_buffer)
        
        # Step 4: Nonlinear transformation
        nonlinear_output = self._apply_nonlinear_transform(filtered_output, "tanh")
        
        # Step 5: State estimation update
        innovation = nonlinear_output - self.state_vector[0]
        self._update_state_estimation(nonlinear_output, innovation)
        
        # Step 6: Generate final output
        state_contribution = np.sum(self.state_vector[:4]) * 0.25
        final_output = nonlinear_output + state_contribution
        
        # Step 7: Update output buffer
        self.output_buffer = np.roll(self.output_buffer, 1)
        self.output_buffer[0] = final_output
        
        # Step 8: Compute error and adaptation
        error = input_sample - final_output
        self.error_history.append(error)
        if len(self.error_history) > 1000:
            self.error_history.pop(0)
        
        # Step 9: Adaptive coefficient update
        error_magnitude = abs(error)
        if error_magnitude > self.adaptation_threshold:
            self.adaptation_active = True
            self.adaptation_counter += 1
        else:
            self.adaptation_active = False
        
        self._adapt_coefficients(error)
        
        # Step 10: Performance monitoring
        self._compute_performance_metrics()
        
        # Step 11: Generate output metadata
        output_metadata = {
            "output": final_output,
            "error": error,
            "state_vector": self.state_vector.copy(),
            "coefficients": self.coefficients.copy(),
            "snr_estimate": self.snr_estimate,
            "convergence_rate": self.convergence_rate,
            "stability_factor": self.stability_factor,
            "adaptation_active": self.adaptation_active,
            "adaptation_counter": self.adaptation_counter,
            "innovation": innovation,
            "nonlinear_output": nonlinear_output,
            "filtered_output": filtered_output
        }
        
        return output_metadata


def process_streaming_data(input_stream: List[float], 
                          filter_params: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Process a stream of input data through the complex adaptive filter.
    
    Args:
        input_stream: List of input samples
        filter_params: Filter configuration parameters
        
    Returns:
        List of output dictionaries with metadata
    """
    # Initialize filter
    filter_length = filter_params.get("filter_length", 32)
    state_dim = filter_params.get("state_dim", 8)
    learning_rate = filter_params.get("learning_rate", 0.01)
    adaptation_threshold = filter_params.get("adaptation_threshold", 0.1)
    
    adaptive_filter = ComplexAdaptiveFilter(
        filter_length=filter_length,
        state_dim=state_dim,
        learning_rate=learning_rate,
        adaptation_threshold=adaptation_threshold
    )
    
    # Process each sample
    outputs = []
    for sample in input_stream:
        output = adaptive_filter.process_sample(sample)
        outputs.append(output)
    
    return outputs


def generate_test_signal(length: int = 1000, 
                        signal_type: str = "mixed") -> List[float]:
    """
    Generate a complex test signal for algorithm validation.
    
    Args:
        length: Length of test signal
        signal_type: Type of signal to generate
        
    Returns:
        List of signal samples
    """
    t = np.linspace(0, 10, length)
    
    if signal_type == "mixed":
        # Mixed signal with multiple components
        signal = (np.sin(2 * np.pi * 0.5 * t) +  # Low frequency
                 0.5 * np.sin(2 * np.pi * 2.0 * t) +  # Medium frequency
                 0.3 * np.sin(2 * np.pi * 5.0 * t) +  # High frequency
                 0.1 * np.random.normal(0, 1, length))  # Noise
    elif signal_type == "chirp":
        # Chirp signal with varying frequency
        signal = np.sin(2 * np.pi * (0.1 + 0.4 * t) * t)
    elif signal_type == "impulse":
        # Impulse response test
        signal = np.zeros(length)
        signal[length//4] = 1.0
        signal[length//2] = -0.5
        signal[3*length//4] = 0.3
    else:
        # Default: sinusoidal
        signal = np.sin(2 * np.pi * t)
    
    return signal.tolist()


# Main processing function for ARDA
def process_algorithm(input_data: List[float], 
                    parameters: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main algorithm processing function for ARDA integration.
    
    Args:
        input_data: Input data stream
        parameters: Algorithm parameters
        
    Returns:
        Dictionary containing results and metadata
    """
    # Process the input stream
    outputs = process_streaming_data(input_data, parameters)
    
    # Extract key metrics
    final_outputs = [out["output"] for out in outputs]
    errors = [out["error"] for out in outputs]
    snr_values = [out["snr_estimate"] for out in outputs]
    
    # Compute summary statistics
    final_snr = snr_values[-1] if snr_values else 0.0
    mean_error = np.mean(np.abs(errors))
    convergence_achieved = abs(errors[-1]) < 0.01 if errors else False
    
    # Final state information
    final_state = outputs[-1] if outputs else {}
    
    return {
        "output_stream": final_outputs,
        "error_stream": errors,
        "final_snr": final_snr,
        "mean_error": mean_error,
        "convergence_achieved": convergence_achieved,
        "final_state_vector": final_state.get("state_vector", []),
        "final_coefficients": final_state.get("coefficients", []),
        "adaptation_counter": final_state.get("adaptation_counter", 0),
        "stability_factor": final_state.get("stability_factor", 1.0),
        "processing_metadata": {
            "input_length": len(input_data),
            "filter_length": parameters.get("filter_length", 32),
            "state_dimension": parameters.get("state_dim", 8),
            "learning_rate": parameters.get("learning_rate", 0.01)
        }
    }


if __name__ == "__main__":
    # Test the algorithm
    test_signal = generate_test_signal(500, "mixed")
    params = {
        "filter_length": 16,
        "state_dim": 6,
        "learning_rate": 0.005,
        "adaptation_threshold": 0.05
    }
    
    result = process_algorithm(test_signal, params)
    print(f"Final SNR: {result['final_snr']:.2f} dB")
    print(f"Mean Error: {result['mean_error']:.4f}")
    print(f"Convergence: {result['convergence_achieved']}")
    print(f"Adaptation Events: {result['adaptation_counter']}")
