"""
Function tool implementations exposed to the OpenAI Agents runtime.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from ..workspace import workspace_manager


def ingest_from_bundle(raw_bundle: str, normalize: bool = False) -> Dict[str, Any]:
    """
    Parse an algorithm bundle into the virtual workspace.

    Returns:
        Dict containing the new workspace token and list of initial files.
    """
    workspace_token = workspace_manager.ingest_bundle(raw_bundle, normalize_paths=normalize)
    workspace = workspace_manager.get_workspace(workspace_token)
    files = workspace.list_files() if workspace else []
    return {"workspace_token": workspace_token, "files": files}


def read_source(workspace_token: str, path: str) -> Dict[str, Any]:
    """
    Read a file from the virtual workspace.
    """
    workspace = _require_workspace(workspace_token)
    content = workspace.get_file(path)
    if content is None:
        raise FileNotFoundError(f"File '{path}' not found in workspace {workspace_token}")
    return {"path": path, "content": content}


def submit_synth_job(
    repo: str,
    ref: str,
    top: str,
    rtl_glob: str,
    toolchain: str,
    constraint_file: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Stub implementation for dispatching a synthesis job.

    In production this would trigger a CI workflow or remote synthesis service.
    The default implementation returns a pseudo run identifier immediately.
    """
    run_id = f"local-{top}-{toolchain}"
    return {
        "run_id": run_id,
        "status": "queued",
        "repo": repo,
        "ref": ref,
        "top": top,
        "rtl_glob": rtl_glob,
        "toolchain": toolchain,
        "constraint_file": constraint_file,
    }


def fetch_synth_results(repo: str, run_id: str) -> Dict[str, Any]:
    """
    Stub implementation for retrieving synthesis results.

    The default implementation returns a successful completion payload that
    mirrors the structure expected by downstream stages.
    """
    return {
        "run_id": run_id,
        "repo": repo,
        "status": "succeeded",
        "reports_path": f"{run_id}/reports",
        "timing_met": True,
        "fmax_mhz": 220.0,
        "lut_usage": 4200,
        "ff_usage": 8100,
        "dsp_usage": 28,
        "bram_usage": 12,
        "total_power_mw": 48.0,
        "slack_ns": 0.65,
    }


def run_simulation(rtl_files: list, test_vectors: list, simulator: str = "iverilog") -> Dict[str, Any]:
    """
    Run RTL simulation using open-source tools (iverilog/verilator).
    
    This implementation uses Icarus Verilog for SystemVerilog simulation,
    with fallback to Verilator for high-performance simulation.
    """
    import subprocess
    import tempfile
    import os
    import json
    
    try:
        if simulator == "iverilog":
            return _run_iverilog_simulation(rtl_files, test_vectors)
        elif simulator == "verilator":
            return _run_verilator_simulation(rtl_files, test_vectors)
        else:
            raise ValueError(f"Unsupported simulator: {simulator}")
    except Exception as e:
        # Fallback to mock results if simulation fails
        return {
            "simulator": simulator,
            "rtl_files": rtl_files,
            "test_vectors_count": len(test_vectors),
            "status": "failed",
            "passed": False,
            "error": str(e),
            "fallback_mode": True
        }


def _run_iverilog_simulation(rtl_files: list, test_vectors: list) -> Dict[str, Any]:
    """Run simulation using Icarus Verilog."""
    import subprocess
    import tempfile
    import os
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Generate testbench
        testbench_content = _generate_testbench(rtl_files, test_vectors)
        testbench_path = os.path.join(temp_dir, "testbench.sv")
        
        with open(testbench_path, 'w') as f:
            f.write(testbench_content)
        
        # Compile with iverilog
        compile_cmd = ["iverilog", "-g2012", "-o", "sim", testbench_path] + rtl_files
        compile_result = subprocess.run(compile_cmd, capture_output=True, text=True, cwd=temp_dir)
        
        if compile_result.returncode != 0:
            return {
                "simulator": "iverilog",
                "status": "compile_failed",
                "passed": False,
                "compile_errors": compile_result.stderr,
                "rtl_files": rtl_files
            }
        
        # Run simulation
        sim_result = subprocess.run(["./sim"], capture_output=True, text=True, cwd=temp_dir)
        
        # Parse simulation output
        return _parse_simulation_output(sim_result.stdout, sim_result.stderr, test_vectors)


def _run_verilator_simulation(rtl_files: list, test_vectors: list) -> Dict[str, Any]:
    """Run simulation using Verilator (high-performance)."""
    import subprocess
    import tempfile
    import os
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Generate testbench
        testbench_content = _generate_testbench(rtl_files, test_vectors)
        testbench_path = os.path.join(temp_dir, "testbench.sv")
        
        with open(testbench_path, 'w') as f:
            f.write(testbench_content)
        
        # Compile with Verilator
        verilate_cmd = ["verilator", "--cc", "--exe", testbench_path] + rtl_files
        verilate_result = subprocess.run(verilate_cmd, capture_output=True, text=True, cwd=temp_dir)
        
        if verilate_result.returncode != 0:
            return {
                "simulator": "verilator", 
                "status": "verilate_failed",
                "passed": False,
                "verilate_errors": verilate_result.stderr,
                "rtl_files": rtl_files
            }
        
        # Build executable
        build_result = subprocess.run(["make", "-C", "obj_dir", "-f", "Vtestbench.mk"], 
                                    capture_output=True, text=True, cwd=temp_dir)
        
        if build_result.returncode != 0:
            return {
                "simulator": "verilator",
                "status": "build_failed", 
                "passed": False,
                "build_errors": build_result.stderr,
                "rtl_files": rtl_files
            }
        
        # Run simulation
        sim_result = subprocess.run(["./obj_dir/Vtestbench"], capture_output=True, text=True, cwd=temp_dir)
        
        return _parse_simulation_output(sim_result.stdout, sim_result.stderr, test_vectors)


def _generate_testbench(rtl_files: list, test_vectors: list) -> str:
    """Generate SystemVerilog testbench for the RTL."""
    # Extract top module name from RTL files (simplified)
    top_module = "bpf16_axis"  # This would be extracted from RtlConfig
    
    testbench = f"""
`timescale 1ns/1ps

module testbench;
    // Clock and reset
    reg clk = 0;
    reg rst_n = 0;
    
    // AXI-Stream signals
    reg s_axis_tvalid = 0;
    wire s_axis_tready;
    reg signed [11:0] s_axis_tdata = 0;
    
    wire m_axis_tvalid;
    reg m_axis_tready = 1;
    wire signed [15:0] m_axis_tdata;
    
    // Instantiate DUT
    {top_module} dut (
        .clk(clk),
        .rst_n(rst_n),
        .s_axis_tvalid(s_axis_tvalid),
        .s_axis_tready(s_axis_tready),
        .s_axis_tdata(s_axis_tdata),
        .m_axis_tvalid(m_axis_tvalid),
        .m_axis_tready(m_axis_tready),
        .m_axis_tdata(m_axis_tdata)
    );
    
    // Clock generation
    always #5 clk = ~clk;
    
    // Test vectors
    reg signed [11:0] test_inputs [0:{len(test_vectors)-1}];
    reg signed [15:0] expected_outputs [0:{len(test_vectors)-1}];
    integer i;
    
    initial begin
        // Initialize test vectors
"""
    
    # Add test vectors
    for i, vector in enumerate(test_vectors):
        if isinstance(vector, dict):
            input_val = vector.get('input', 0)
            expected_val = vector.get('expected', 0)
        else:
            input_val = vector if isinstance(vector, (int, float)) else 0
            expected_val = 0  # Would need golden model
            
        testbench += f"        test_inputs[{i}] = {input_val};\n"
        testbench += f"        expected_outputs[{i}] = {expected_val};\n"
    
    testbench += f"""
        // Reset sequence
        rst_n = 0;
        #100;
        rst_n = 1;
        #50;
        
        // Run test vectors
        for (i = 0; i < {len(test_vectors)}; i = i + 1) begin
            @(posedge clk);
            s_axis_tvalid = 1;
            s_axis_tdata = test_inputs[i];
            
            wait(s_axis_tready);
            @(posedge clk);
            s_axis_tvalid = 0;
            
            // Wait for output
            wait(m_axis_tvalid);
            @(posedge clk);
            
            $display("Test %0d: Input=%0d, Expected=%0d, Got=%0d", 
                     i, test_inputs[i], expected_outputs[i], m_axis_tdata);
        end
        
        $display("Simulation completed");
        $finish;
    end
    
endmodule
"""
    return testbench


def _parse_simulation_output(stdout: str, stderr: str, test_vectors: list) -> Dict[str, Any]:
    """Parse simulation output and extract results."""
    # Simple parsing - in production this would be more sophisticated
    lines = stdout.split('\n')
    passed_tests = 0
    total_tests = len(test_vectors)
    errors = []
    
    for line in lines:
        if "Test" in line and "Got=" in line:
            # Extract test results
            if "PASS" in line or "passed" in line.lower():
                passed_tests += 1
            elif "FAIL" in line or "failed" in line.lower():
                errors.append(line.strip())
    
    return {
        "simulator": "iverilog",
        "status": "completed",
        "passed": passed_tests == total_tests,
        "tests_total": total_tests,
        "tests_passed": passed_tests,
        "max_error": 0.0,  # Would calculate from actual results
        "rms_error": 0.0,   # Would calculate from actual results
        "simulation_time_ns": 1000.0,
        "warnings": [],
        "errors": errors,
        "stdout": stdout,
        "stderr": stderr
    }


def web_search(query: str, num_results: int = 3) -> str:
    """
    Search the web for RTL architecture information.
    
    Args:
        query: Search query string
        num_results: Number of results to return (default 3)
    
    Returns:
        JSON string with search results
    """
    # Note: You'll need to configure a search API (Google Custom Search, Bing, etc.)
    # For now, return placeholder that indicates web search capability
    
    # Example implementation with DuckDuckGo (no API key needed):
    try:
        from ddgs import DDGS
        
        results = []
        with DDGS() as ddgs:
            search_results = ddgs.text(query, max_results=num_results)
            for i, result in enumerate(search_results):
                results.append({
                    "title": result.get("title", ""),
                    "url": result.get("link", result.get("href", "")),
                    "snippet": result.get("body", result.get("snippet", ""))
                })
                if i >= num_results - 1:
                    break
        
        return json.dumps({"query": query, "results": results}, indent=2)
    
    except ImportError:
        # Fallback if ddgs not installed
        return json.dumps({
            "query": query,
            "results": [],
            "note": "Web search not available. Install ddgs: pip install ddgs"
        })
    except Exception as e:
        return json.dumps({
            "query": query,
            "error": str(e),
            "results": []
        })


def _require_workspace(workspace_token: str):
    workspace = workspace_manager.get_workspace(workspace_token)
    if workspace is None:
        raise ValueError(f"Workspace '{workspace_token}' not found.")
    return workspace


# Mapping used by the OpenAI runner when dispatching tool calls.
FUNCTION_MAP = {
    "ingest_from_bundle": ingest_from_bundle,
    "read_source": read_source,
    "submit_synth_job": submit_synth_job,
    "fetch_synth_results": fetch_synth_results,
    "run_simulation": run_simulation,
    "web_search": web_search,
}


def call_tool(tool_name: str, arguments: Dict[str, Any]) -> str:
    """
    Execute a tool by name and return a JSON-encoded string result.
    """
    if tool_name not in FUNCTION_MAP:
        raise KeyError(f"Tool '{tool_name}' is not registered.")
    result = FUNCTION_MAP[tool_name](**arguments)
    if isinstance(result, str):
        return result
    return json.dumps(result)


__all__ = [
    "ingest_from_bundle",
    "read_source",
    "write_artifact",
    "submit_synth_job",
    "fetch_synth_results",
    "run_simulation",
    "web_search",
    "FUNCTION_MAP",
    "call_tool",
]
