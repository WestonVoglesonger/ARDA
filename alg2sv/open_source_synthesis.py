"""
Open-source FPGA synthesis integration for ALG2SV.
Supports Yosys/nextpnr for iCE40/ECP5 and SymbiFlow for Xilinx 7-series.
"""

import os
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional
import re

from agents import function_tool


class OpenSourceSynthesis:
    """Open-source FPGA synthesis integration."""

    def __init__(self):
        self.yosys_available = self._check_yosys()
        self.nextpnr_available = self._check_nextpnr()
        self.icestorm_available = self._check_icestorm()
        self.symbiflow_available = self._check_symbiflow()

    def _check_yosys(self) -> bool:
        """Check if Yosys is available."""
        try:
            result = subprocess.run(['yosys', '--version'],
                                  capture_output=True, text=True, timeout=10)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    def _check_nextpnr(self) -> bool:
        """Check if nextpnr is available."""
        try:
            result = subprocess.run(['nextpnr-ice40', '--help'],
                                  capture_output=True, text=True, timeout=10)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    def _check_icestorm(self) -> bool:
        """Check if IceStorm tools are available."""
        try:
            result = subprocess.run(['icepack', '--version'],
                                  capture_output=True, text=True, timeout=10)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    def _check_symbiflow(self) -> bool:
        """Check if SymbiFlow is available."""
        try:
            result = subprocess.run(['python', '-c', 'import symbiflow'],
                                  capture_output=True, text=True, timeout=10)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    def run_yosys_synthesis(self,
                           rtl_files: List[str],
                           top_module: str,
                           fpga_family: str = "ice40hx8k",
                           project_name: str = "alg2sv_oss_synth") -> Dict[str, Any]:
        """
        Run Yosys synthesis for iCE40 or ECP5 FPGAs.

        Args:
            rtl_files: List of RTL source files
            top_module: Top-level module name
            fpga_family: FPGA family ('ice40hx8k', 'ecp5', etc.)
            project_name: Project name

        Returns:
            Synthesis results
        """
        if not self.yosys_available:
            return {
                "success": False,
                "error": "Yosys not available",
                "status": "yosys_unavailable"
            }

        # Create temporary directory
        with tempfile.TemporaryDirectory(prefix=f"yosys_{project_name}_") as temp_dir:
            try:
                # Generate Yosys script
                yosys_script = self._generate_yosys_script(rtl_files, top_module, fpga_family, temp_dir)

                # Save script
                script_path = os.path.join(temp_dir, f"{project_name}.ys")
                with open(script_path, 'w') as f:
                    f.write(yosys_script)

                # Run Yosys
                print(f"Running Yosys synthesis for {project_name}...")
                result = subprocess.run(
                    ['yosys', script_path],
                    cwd=temp_dir,
                    capture_output=True,
                    text=True,
                    timeout=300  # 5 minute timeout
                )

                if result.returncode != 0:
                    return {
                        "success": False,
                        "error": f"Yosys failed: {result.stderr}",
                        "status": "yosys_error",
                        "yosys_stdout": result.stdout,
                        "yosys_stderr": result.stderr
                    }

                # Parse results (basic implementation)
                return {
                    "success": True,
                    "project_name": project_name,
                    "top_module": top_module,
                    "fpga_family": fpga_family,
                    "synthesis_time_seconds": 0,  # Would need to track this
                    "yosys_stdout": result.stdout,
                    "yosys_stderr": result.stderr,
                    "output_dir": temp_dir,
                    "estimated_resources": self._estimate_resources(fpga_family),
                    "note": "Open-source synthesis completed - bitstream generation requires nextpnr"
                }

            except subprocess.TimeoutExpired:
                return {
                    "success": False,
                    "error": "Yosys synthesis timed out",
                    "status": "timeout"
                }
            except Exception as e:
                return {
                    "success": False,
                    "error": f"Yosys execution failed: {str(e)}",
                    "status": "execution_error"
                }

    def run_symbiflow_synthesis(self,
                               rtl_files: List[str],
                               top_module: str,
                               fpga_family: str = "xc7a100t",
                               project_name: str = "alg2sv_symbiflow") -> Dict[str, Any]:
        """
        Run SymbiFlow synthesis for Xilinx 7-series FPGAs.

        Args:
            rtl_files: List of RTL source files
            top_module: Top-level module name
            fpga_family: FPGA family (e.g., 'xc7a100t')
            project_name: Project name

        Returns:
            Synthesis results
        """
        if not self.symbiflow_available:
            return {
                "success": False,
                "error": "SymbiFlow not available",
                "status": "symbiflow_unavailable"
            }

        return {
            "success": False,
            "error": "SymbiFlow integration not yet implemented",
            "status": "not_implemented",
            "note": "SymbiFlow support requires complex conda environment setup"
        }

    def _generate_yosys_script(self, rtl_files: List[str], top_module: str,
                              fpga_family: str, output_dir: str) -> str:
        """Generate Yosys synthesis script."""
        script = f"""# ALG2SV Yosys Synthesis Script
# Target: {fpga_family}

# Read RTL files
"""

        for rtl_file in rtl_files:
            script += f'read_verilog {rtl_file}\n'

        script += f"""
# Set top module
hierarchy -top {top_module}

# Generic synthesis
proc
opt
fsm
opt
memory
opt
techmap
opt

# FPGA-specific optimization
"""

        if fpga_family.startswith('ice40'):
            script += """
# iCE40 specific optimizations
ice40_opt
"""
        elif fpga_family.startswith('ecp5'):
            script += """
# ECP5 specific optimizations
ecp5_opt
"""

        script += f"""
# Write synthesized netlist
write_json {output_dir}/synthesis.json

# Generate reports
stat
"""

        return script

    def _estimate_resources(self, fpga_family: str) -> Dict[str, int]:
        """Provide resource estimates for different FPGA families."""
        estimates = {
            'ice40hx8k': {
                'lut': 7680,
                'ff': 7680,
                'dsp': 0,  # iCE40 doesn't have DSPs
                'bram': 32
            },
            'ice40up5k': {
                'lut': 5280,
                'ff': 5280,
                'dsp': 0,
                'bram': 30
            },
            'ecp5': {
                'lut': 84000,
                'ff': 84000,
                'dsp': 156,
                'bram': 208
            }
        }

        return estimates.get(fpga_family, estimates['ice40hx8k'])


# Global instance
oss_synthesis = OpenSourceSynthesis()


@function_tool
def run_yosys_synthesis(rtl_files: List[str],
                       top_module: str,
                       fpga_family: str = "ice40hx8k",
                       project_name: str = "alg2sv_oss") -> Dict[str, Any]:
    """
    Function tool for Synth Agent to run Yosys open-source synthesis.
    """
    return oss_synthesis.run_yosys_synthesis(rtl_files, top_module, fpga_family, project_name)


@function_tool
def run_symbiflow_synthesis(rtl_files: List[str],
                           top_module: str,
                           fpga_family: str = "xc7a100t",
                           project_name: str = "alg2sv_symbiflow") -> Dict[str, Any]:
    """
    Function tool for Synth Agent to run SymbiFlow synthesis.
    """
    return oss_synthesis.run_symbiflow_synthesis(rtl_files, top_module, fpga_family, project_name)
