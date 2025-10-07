"""
Mock agent implementation for testing ALG2SV pipeline.
This simulates the OpenAI Agents SDK functionality.
"""

import asyncio
import json
from typing import Dict, Any, List, Optional, Callable
from pydantic import BaseModel


class MockAgent:
    """Mock agent that simulates OpenAI Agents SDK behavior."""

    def __init__(self, name: str, instructions: str, tools: Optional[List] = None, output_type=None):
        self.name = name
        self.instructions = instructions
        self.tools = tools or []
        self.output_type = output_type

    async def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Mock agent execution that simulates the pipeline stages."""
        print(f"🤖 Running {self.name}...")

        # Simulate processing time
        await asyncio.sleep(0.1)

        # Mock responses based on agent type
        if "Spec" in self.name:
            return self._mock_spec_agent(input_data)
        elif "Quant" in self.name:
            return self._mock_quant_agent(input_data)
        elif "MicroArch" in self.name:
            return self._mock_microarch_agent(input_data)
        elif "RTL" in self.name:
            return self._mock_rtl_agent(input_data)
        elif "Verify" in self.name:
            return self._mock_verify_agent(input_data)
        elif "Synth" in self.name:
            return self._mock_synth_agent(input_data)
        elif "Feedback" in self.name:
            return self._mock_feedback_agent(input_data)
        else:
            return {"error": f"Unknown agent type: {self.name}"}

    def _mock_spec_agent(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Mock spec agent - analyzes algorithm and creates contract."""
        return {
            "name": "TestAlgorithm",
            "description": "16-tap band-pass FIR filter",
            "clock_mhz_target": 200.0,
            "throughput_samples_per_cycle": 1,
            "input_format": {"width": 12, "fractional_bits": 11},
            "output_format": {"width": 16, "fractional_bits": 14},
            "resource_budget": {"lut": 20000, "ff": 40000, "dsp": 40, "bram": 20},
            "verification_config": {"num_samples": 1024, "tolerance_abs": 1e-2}
        }

    def _mock_quant_agent(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Mock quant agent - converts to fixed-point."""
        return {
            "fixed_point_config": {
                "input_width": 12, "input_frac": 11,
                "output_width": 16, "output_frac": 14,
                "coeff_width": 16, "coeff_frac": 15,
                "accumulator_width": 32
            },
            "error_metrics": {
                "max_abs_error": 1.23e-6,
                "rms_error": 5.67e-7,
                "snr_db": 85.2
            },
            "quantized_coefficients": [-0.0071, -0.0138, -0.0109, 0.0112],  # truncated
            "fxp_model_path": "workspace/models/fxp_model.py"
        }

    def _mock_microarch_agent(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Mock microarch agent - designs architecture."""
        return {
            "pipeline_depth": 4,
            "unroll_factor": 1,
            "memory_config": {"use_bram": False, "buffer_depth": 16},
            "dsp_usage_estimate": 16,
            "estimated_latency_cycles": 4,
            "handshake_protocol": "ready_valid"
        }

    def _mock_rtl_agent(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Mock RTL agent - generates SystemVerilog."""
        # Simulate writing RTL files to workspace
        workspace_token = input_data.get('workspace_token')
        if workspace_token:
            # In a real implementation, this would write to the workspace
            print(f"   📝 Generated RTL files in workspace {workspace_token[:8]}...")

        return {
            "rtl_files": ["rtl/bpf16_core.sv", "rtl/bpf16_top.sv", "rtl/bpf16_coeff_mem.sv"],
            "params_file": "rtl/bpf16_params.svh",
            "top_module": "bpf16_top",
            "lint_passed": True,
            "estimated_resources": {"lut": 1847, "ff": 3201, "dsp": 16}
        }

    def _mock_verify_agent(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Mock verify agent - runs verification."""
        return {
            "tests_total": 1024,
            "tests_passed": 1024,
            "all_passed": True,
            "mismatches": [],
            "max_abs_error": 4.56e-7,
            "rms_error": 2.34e-7,
            "functional_coverage": 0.98
        }

    def _mock_synth_agent(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Mock synth agent - estimates synthesis results."""
        return {
            "fmax_mhz": 198.5,
            "timing_met": True,
            "lut_usage": 1847,
            "ff_usage": 3201,
            "dsp_usage": 16,
            "bram_usage": 2,
            "total_power_mw": 45.2,
            "slack_ns": 1.23,
            "reports_path": "synth/reports/"
        }

    def _mock_feedback_agent(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Mock feedback agent - decides whether to retry stages."""
        results = input_data.get('results') if isinstance(input_data, dict) else {}
        synth = (results or {}).get('synth', {})
        verify = (results or {}).get('verify', {})

        if synth and not synth.get('timing_met', True):
            return {
                "action": "retry_synth",
                "target_stage": "synth",
                "guidance": "Timing not met; relax pipeline depth or adjust constraints."
            }

        if verify and not verify.get('all_passed', True):
            return {
                "action": "retry_verify",
                "target_stage": "verify",
                "guidance": "Investigate mismatches and expand test coverage."
            }

        return {"action": "continue"}


async def mock_run_workflow(agent: MockAgent, input_data: Dict[str, Any]) -> Any:
    """Mock workflow runner that simulates agent execution."""
    return await agent.run(input_data)

