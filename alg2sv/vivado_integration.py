"""
Vivado CLI Integration for ALG2SV
Enables real FPGA synthesis, implementation, and bitstream generation
"""

import os
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import re
import time

class VivadoProject:
    """Manages Vivado project creation and synthesis."""

    def __init__(self, project_name: str, fpga_part: str = "xc7z020clg484-1"):
        self.project_name = project_name
        self.fpga_part = fpga_part
        self.project_dir = None
        self.rtl_files = []
        self.constraint_file = None
        self.top_module = None

    def set_rtl_files(self, rtl_files: List[str]):
        """Set the RTL source files."""
        self.rtl_files = rtl_files

    def set_constraint_file(self, constraint_file: str):
        """Set the constraint file."""
        self.constraint_file = constraint_file

    def set_top_module(self, top_module: str):
        """Set the top-level module name."""
        self.top_module = top_module

    def generate_tcl_script(self, output_dir: str, generate_bitstream: bool = True) -> str:
        """Generate TCL script for Vivado automation."""

        # Create temporary directory for project
        self.project_dir = tempfile.mkdtemp(prefix=f"vivado_{self.project_name}_")
        reports_dir = os.path.join(self.project_dir, "reports")
        os.makedirs(reports_dir, exist_ok=True)

        # TCL script template - avoid f-string conflicts with TCL syntax
        tcl_template = """# ALG2SV Vivado Automation Script
# Generated for project: """ + self.project_name + """

set project_name \"""" + self.project_name + """\"
set project_dir \"""" + self.project_dir + """\"
set fpga_part \"""" + self.fpga_part + """\"
set top_module \"""" + self.top_module + """\"
set reports_dir \"""" + reports_dir + """\"
set output_dir \"""" + output_dir + """\"

puts "Starting ALG2SV Vivado project: $project_name"

# Create project
create_project $project_name $project_dir -part $fpga_part -force
set_property target_language Verilog [current_project]

# Add RTL files
set rtl_files [list \\
"""

        # Add RTL files to TCL
        for rtl_file in self.rtl_files:
            if os.path.exists(rtl_file):
                tcl_template += '    "' + rtl_file + '" \\\n'
            else:
                print(f"Warning: RTL file not found: {rtl_file}")

        tcl_template += "]\n"
        tcl_template += "add_files $rtl_files\n"

        # Add constraint file if exists
        if self.constraint_file and os.path.exists(self.constraint_file):
            tcl_template += """
# Add constraints
add_files -fileset constrs_1 \"""" + self.constraint_file + """\"
"""

        # Set top module
        tcl_template += """
# Set top module
set_property top $top_module [current_fileset]

puts "Project setup complete. Starting synthesis..."

# Launch synthesis
launch_runs synth_1 -jobs 4
wait_on_run synth_1

# Check synthesis status
set synth_progress [get_property PROGRESS [get_runs synth_1]]
puts "Synthesis progress: $synth_progress"

if {[get_property STATUS [get_runs synth_1]] != "synth_design Complete!"} {
    puts "ERROR: Synthesis failed!"
    exit 1
}

puts "Synthesis completed successfully!"

# Launch implementation
puts "Starting implementation..."
launch_runs impl_1 -jobs 4
wait_on_run impl_1

# Check implementation status
set impl_progress [get_property PROGRESS [get_runs impl_1]]
puts "Implementation progress: $impl_progress"

if {[get_property STATUS [get_runs impl_1]] != "route_design Complete!"} {
    puts "ERROR: Implementation failed!"
    exit 1
}

puts "Implementation completed successfully!"

# Generate reports
puts "Generating reports..."
open_run impl_1

# Utilization report
report_utilization -file $reports_dir/utilization.rpt

# Timing reports
report_timing_summary -file $reports_dir/timing_summary.rpt
report_timing -file $reports_dir/timing.rpt -max_paths 10

# Power report (if available)
catch {report_power -file $reports_dir/power.rpt}

puts "Reports generated successfully!"
"""

        if generate_bitstream:
            tcl_template += """
# Generate bitstream
puts "Generating bitstream..."
set_property BITSTREAM.GENERAL.COMPRESS TRUE [current_design]
launch_runs impl_1 -to_step write_bitstream
wait_on_run impl_1

# Check bitstream generation
if {[get_property STATUS [get_runs impl_1]] != "write_bitstream Complete!"} {
    puts "ERROR: Bitstream generation failed!"
    exit 1
}

# Copy bitstream to output directory
set bitstream_src [get_property DIRECTORY [get_runs impl_1]]/$top_module.bit
set bitstream_dst $output_dir/$project_name.bit

if {[file exists $bitstream_src]} {
    file copy -force $bitstream_src $bitstream_dst
    puts "Bitstream generated: $bitstream_dst"
} else {
    puts "ERROR: Bitstream file not found at $bitstream_src"
    exit 1
}
"""

        tcl_template += """
puts "ALG2SV Vivado automation completed successfully!"
"""

        return tcl_template

class VivadoIntegration:
    """Main interface for Vivado CLI integration."""

    def __init__(self, vivado_path: Optional[str] = None):
        self.vivado_path = vivado_path or self._find_vivado()
        self.vivado_available = self._check_vivado()

    def _find_vivado(self) -> str:
        """Find Vivado installation path."""
        # Common Vivado installation paths
        common_paths = [
            "/opt/Xilinx/Vivado/2023.2/bin/vivado",
            "/opt/Xilinx/Vivado/2023.1/bin/vivado",
            "/opt/Xilinx/Vivado/2022.2/bin/vivado",
            "C:\\Xilinx\\Vivado\\2023.2\\bin\\vivado.bat",
            "C:\\Xilinx\\Vivado\\2023.1\\bin\\vivado.bat",
        ]

        for path in common_paths:
            if os.path.exists(path):
                return path

        # Check PATH
        import shutil
        vivado_exe = shutil.which("vivado")
        if vivado_exe:
            return vivado_exe

        return "vivado"  # Assume it's in PATH

    def _check_vivado(self) -> bool:
        """Check if Vivado is available and working."""
        try:
            result = subprocess.run(
                [self.vivado_path, "-version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    def synthesize_design(self,
                         rtl_files: List[str],
                         top_module: str,
                         fpga_part: str = "xc7z020clg484-1",
                         constraint_file: Optional[str] = None,
                         project_name: str = "alg2sv_project",
                         generate_bitstream: bool = True) -> Dict[str, Any]:
        """
        Run complete synthesis flow: synthesis → implementation → bitstream.

        Args:
            rtl_files: List of RTL source files
            top_module: Top-level module name
            fpga_part: FPGA part number
            constraint_file: Optional constraint file
            project_name: Project name
            generate_bitstream: Whether to generate bitstream

        Returns:
            Dict with synthesis results
        """

        if not self.vivado_available:
            return {
                "success": False,
                "error": "Vivado not found or not working",
                "status": "vivado_unavailable"
            }

        # Create project
        project = VivadoProject(project_name, fpga_part)
        project.set_rtl_files(rtl_files)
        project.set_top_module(top_module)
        if constraint_file:
            project.set_constraint_file(constraint_file)

        # Create output directory
        output_dir = tempfile.mkdtemp(prefix=f"vivado_output_{project_name}_")
        os.makedirs(output_dir, exist_ok=True)

        # Generate TCL script
        tcl_script = project.generate_tcl_script(output_dir, generate_bitstream)

        # Save TCL script
        tcl_file = os.path.join(output_dir, f"{project_name}.tcl")
        with open(tcl_file, 'w') as f:
            f.write(tcl_script)

        # Run Vivado
        try:
            print(f"Running Vivado synthesis for {project_name}...")
            start_time = time.time()

            result = subprocess.run(
                [self.vivado_path, "-mode", "batch", "-source", tcl_file],
                cwd=output_dir,
                capture_output=True,
                text=True,
                timeout=1800  # 30 minute timeout
            )

            end_time = time.time()
            synthesis_time = end_time - start_time

            # Parse results
            success = result.returncode == 0
            stdout = result.stdout
            stderr = result.stderr

            results = {
                "success": success,
                "project_name": project_name,
                "top_module": top_module,
                "fpga_part": fpga_part,
                "synthesis_time_seconds": synthesis_time,
                "vivado_stdout": stdout,
                "vivado_stderr": stderr,
                "output_dir": output_dir,
                "tcl_script": tcl_script
            }

            if success:
                # Parse reports
                reports = self._parse_reports(project.project_dir)
                results.update(reports)

                # Check for bitstream
                bitstream_path = os.path.join(output_dir, f"{project_name}.bit")
                if os.path.exists(bitstream_path):
                    results["bitstream_generated"] = True
                    results["bitstream_path"] = bitstream_path
                else:
                    results["bitstream_generated"] = False

            return results

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": "Vivado synthesis timed out",
                "status": "timeout"
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Vivado execution failed: {str(e)}",
                "status": "execution_error"
            }

    def _parse_reports(self, project_dir: str) -> Dict[str, Any]:
        """Parse Vivado reports for utilization and timing."""

        reports_dir = os.path.join(project_dir, "reports")
        results = {}

        # Parse utilization report
        util_file = os.path.join(reports_dir, "utilization.rpt")
        if os.path.exists(util_file):
            results["resource_usage"] = self._parse_utilization(util_file)

        # Parse timing reports
        timing_file = os.path.join(reports_dir, "timing_summary.rpt")
        if os.path.exists(timing_file):
            results["timing"] = self._parse_timing(timing_file)

        return results

    def _parse_utilization(self, util_file: str) -> Dict[str, int]:
        """Parse utilization report."""
        utilization = {}

        try:
            with open(util_file, 'r') as f:
                content = f.read()

            # Extract LUT usage
            lut_match = re.search(r'CLB LUTs\s*\|\s*(\d+)', content)
            if lut_match:
                utilization['lut'] = int(lut_match.group(1))

            # Extract FF usage
            ff_match = re.search(r'CLB Registers\s*\|\s*(\d+)', content)
            if ff_match:
                utilization['ff'] = int(ff_match.group(1))

            # Extract DSP usage
            dsp_match = re.search(r'DSPs\s*\|\s*(\d+)', content)
            if dsp_match:
                utilization['dsp'] = int(dsp_match.group(1))

            # Extract BRAM usage
            bram_match = re.search(r'Block RAM Tile\s*\|\s*(\d+)', content)
            if bram_match:
                utilization['bram'] = int(bram_match.group(1))

        except Exception as e:
            print(f"Warning: Could not parse utilization report: {e}")

        return utilization

    def _parse_timing(self, timing_file: str) -> Dict[str, Any]:
        """Parse timing summary report."""
        timing = {}

        try:
            with open(timing_file, 'r') as f:
                content = f.read()

            # Extract WNS (Worst Negative Slack)
            wns_match = re.search(r'WNS\(ns\)\s*:\s*([-\d.]+)', content)
            if wns_match:
                timing['wns_ns'] = float(wns_match.group(1))

            # Extract TNS (Total Negative Slack)
            tns_match = re.search(r'TNS\(ns\)\s*:\s*([-\d.]+)', content)
            if tns_match:
                timing['tns_ns'] = float(tns_match.group(1))

            # Extract WHS (Worst Hold Slack)
            whs_match = re.search(r'WHS\(ns\)\s*:\s*([-\d.]+)', content)
            if whs_match:
                timing['whs_ns'] = float(whs_match.group(1))

            # Extract THS (Total Hold Slack)
            ths_match = re.search(r'THS\(ns\)\s*:\s*([-\d.]+)', content)
            if ths_match:
                timing['ths_ns'] = float(ths_match.group(1))

            # Check timing met
            timing['met'] = timing.get('wns_ns', -float('inf')) >= 0

        except Exception as e:
            print(f"Warning: Could not parse timing report: {e}")

        return timing

# Global Vivado integration instance
vivado_integration = VivadoIntegration()

def run_vivado_synthesis(rtl_files: List[str],
                        top_module: str,
                        fpga_part: str = "xc7z020clg484-1",
                        constraint_file: Optional[str] = None,
                        project_name: str = "alg2sv_synth") -> Dict[str, Any]:
    """
    Function tool for Synth Agent to run real Vivado synthesis.
    """
    return vivado_integration.synthesize_design(
        rtl_files=rtl_files,
        top_module=top_module,
        fpga_part=fpga_part,
        constraint_file=constraint_file,
        project_name=project_name,
        generate_bitstream=True
    )
