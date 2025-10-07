"""
Agent definitions for the ALG2SV pipeline.
Each agent handles a specific stage of the algorithm-to-RTL conversion.
"""

from typing import Dict, Any, List, Optional, Literal
from pydantic import BaseModel, Field
import yaml
import json

from agents import function_tool  # Real OpenAI Agents SDK
from .workspace import read_source, write_artifact, ingest_from_bundle, workspace_manager
from .vivado_integration import run_vivado_synthesis
from .simulator_integration import run_rtl_simulation, generate_testbench
import json
import numpy as np


@function_tool
def list_workspace_files(workspace_token: str) -> str:
    """
    List all files in the workspace to help agents discover RTL files.

    Args:
        workspace_token: Workspace identifier

    Returns:
        JSON string with list of file paths in workspace
    """
    try:
        workspace = workspace_manager.get_workspace(workspace_token)
        if not workspace:
            return json.dumps({"error": "Workspace not found", "success": False})

        files = workspace.list_files()
        return json.dumps({
            "success": True,
            "files": files,
            "rtl_files": [f for f in files if f.endswith('.sv')],
            "workspace_token": workspace_token
        })
    except Exception as e:
        return json.dumps({"error": f"Failed to list files: {str(e)}", "success": False})


@function_tool
def extract_test_vectors(workspace_token: str) -> str:
    """
    Extract test vectors from workspace files for simulation.
    
    Args:
        workspace_token: Workspace identifier
        
    Returns:
        JSON string with input and expected test vectors
    """
    try:
        # Read vectors.py file to get test data
        vectors_result = read_source(workspace_token, "algorithms/bpf16/vectors.py")
        if not vectors_result.get("success"):
            return json.dumps({"error": "Could not read vectors.py", "success": False})
        
        vectors_content = vectors_result["content"]
        
        # Execute the vectors.py code to generate test data
        # This is a simplified approach - in practice you'd want to be more careful
        exec_globals = {"np": np}
        exec_locals = {}
        exec(vectors_content, exec_globals, exec_locals)
        
        # Get the test data
        if "make_input" in exec_locals:
            input_data = exec_locals["make_input"](1024)  # Generate 1024 samples
        else:
            return json.dumps({"error": "make_input function not found", "success": False})
        
        # Run the algorithm to get expected outputs
        if "run_batch" in exec_locals:
            expected_data = exec_locals["run_batch"](input_data)
        else:
            return json.dumps({"error": "run_batch function not found", "success": False})
        
        result = {
            "success": True,
            "input_data": input_data.tolist() if hasattr(input_data, 'tolist') else list(input_data),
            "expected_data": expected_data.tolist() if hasattr(expected_data, 'tolist') else list(expected_data),
            "num_samples": len(input_data)
        }
        
        return json.dumps(result)
        
    except Exception as e:
        return json.dumps({"error": f"Failed to extract test vectors: {str(e)}", "success": False})


# Real RTL simulation tool (replaces placeholder)
@function_tool
def run_simulation(top_module: str, 
                  rtl_files: List[str], 
                  input_data: List[float],
                  expected_data: List[float],
                  simulator: str = "auto") -> str:
    """
    Run real RTL simulation using external simulators.

    Args:
        top_module: Top-level module name
        rtl_files: List of RTL file paths
        input_data: List of input test vectors
        expected_data: List of expected output vectors
        simulator: Simulator to use ('auto', 'modelsim', 'questa', 'vcs', 'iverilog')

    Returns:
        JSON string with simulation results including test pass/fail counts
    """
    return run_rtl_simulation(top_module, rtl_files, input_data, expected_data, simulator)


# Pydantic models for structured outputs
class SpecContract(BaseModel):
    """Hardware contract specification."""
    name: str
    description: str
    clock_mhz_target: float
    throughput_samples_per_cycle: int
    input_format: Dict[str, Any] = Field(description="width and fractional_bits")
    output_format: Dict[str, Any] = Field(description="width and fractional_bits")
    resource_budget: Dict[str, Any] = Field(description="lut, ff, dsp, bram budgets")
    verification_config: Dict[str, Any] = Field(description="test parameters")


class QuantConfig(BaseModel):
    """Fixed-point quantization configuration."""
    fixed_point_config: Dict[str, Any]
    error_metrics: Dict[str, Any]
    quantized_coefficients: List[float]
    fxp_model_path: str


class MicroArchConfig(BaseModel):
    """Micro-architecture configuration."""
    pipeline_depth: int
    unroll_factor: int
    memory_config: Dict[str, Any]
    dsp_usage_estimate: int
    estimated_latency_cycles: int
    handshake_protocol: str


class RTLConfig(BaseModel):
    """RTL generation configuration."""
    rtl_files: List[str]
    params_file: str
    top_module: str
    lint_passed: bool
    estimated_resources: Dict[str, int]


class VerifyResults(BaseModel):
    """Verification results."""
    tests_total: int
    tests_passed: int
    all_passed: bool
    mismatches: List[Dict[str, Any]]
    max_abs_error: float
    rms_error: float
    functional_coverage: float


class SynthResults(BaseModel):
    """Synthesis results."""
    fmax_mhz: float
    timing_met: bool
    lut_usage: int
    ff_usage: int
    dsp_usage: int
    bram_usage: int
    total_power_mw: Optional[float] = None
    slack_ns: float
    reports_path: str


class LintResults(BaseModel):
    """Linting results for SystemVerilog code."""
    syntax_errors: int
    style_warnings: int
    lint_violations: int
    critical_issues: int
    issues_list: List[Dict[str, Any]]
    overall_score: float  # 0-100 quality score
    lint_clean: bool


class SimulateResults(BaseModel):
    """RTL simulation results."""
    test_passed: int
    test_failed: int
    test_total: int
    coverage_percent: float
    timing_violations: int
    simulation_errors: List[str]
    waveform_available: bool
    simulation_time_ms: float


class EvaluateResults(BaseModel):
    """Comprehensive evaluation of the design."""
    overall_score: float  # 0-100
    performance_score: float  # timing, throughput, latency
    resource_score: float  # efficiency vs requirements
    quality_score: float  # code quality, verification completeness
    correctness_score: float  # functional accuracy
    recommendations: List[str]
    bottlenecks: List[str]
    optimization_opportunities: List[str]


class FeedbackDecision(BaseModel):
    """Decision produced by the feedback agent after reviewing stage outputs."""

    action: Literal[
        "continue",
        "retry_spec",
        "retry_quant",
        "retry_microarch",
        "retry_rtl",
        "retry_verify",
        "retry_synth",
        "retry_lint",
        "retry_simulate",
        "retry_evaluate",
        "tune_microarch",
        "abort",
    ] = Field(
        description=(
            "Requested pipeline action. Use retry_<stage> to rerun a specific stage,"
            " tune_microarch to revisit micro-architecture design, or abort to stop the pipeline."
        )
    )
    target_stage: Optional[str] = Field(
        default=None,
        description="Specific stage this decision applies to (e.g., 'synth').",
    )
    guidance: Optional[str] = Field(
        default=None,
        description="Additional instructions or context for the targeted stage retry or adjustment.",
    )
    notes: Optional[List[str]] = Field(
        default=None,
        description="Optional structured notes for logging or downstream analysis.",
    )


# Agent classes are now implemented in mock_agents.py for testing
# When the OpenAI Agents SDK is working, these can be uncommented and used instead

# Global workspace context for tools
_current_workspace_token = None

def set_workspace_context(workspace_token: str):
    """Set the global workspace token for tools."""
    global _current_workspace_token
    _current_workspace_token = workspace_token

# Tool functions that can be called by agents (workspace management)
@function_tool
def read_file_tool(path: str) -> Dict[str, Any]:
    """Tool for reading files from the current workspace.

    Args:
        path: Path to the file to read (relative to workspace root)

    Returns:
        Dict containing file content and metadata
    """
    if _current_workspace_token is None:
        return {"error": "No workspace context set", "success": False}

    return read_source(_current_workspace_token, path)


@function_tool
def write_file_tool(path: str, content: str) -> Dict[str, Any]:
    """Tool for writing files to the current workspace.

    Args:
        path: Path where to write the file (relative to workspace root)
        content: Content to write to the file

    Returns:
        Dict confirming the write operation
    """
    if _current_workspace_token is None:
        return {"error": "No workspace context set", "success": False}

    return write_artifact(_current_workspace_token, path, content)
