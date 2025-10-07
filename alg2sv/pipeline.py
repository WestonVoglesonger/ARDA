"""
Main ALG2SV pipeline orchestrator.
Runs the complete algorithm-to-RTL conversion workflow.
"""

import asyncio
import json
from typing import Dict, Any, Optional, List
from pathlib import Path

from agents import Runner, Agent, AgentOutputSchema  # Real OpenAI Agents SDK
from .agents import (
    SpecContract, QuantConfig, MicroArchConfig, RTLConfig, VerifyResults, SynthResults,
    LintResults, SimulateResults, EvaluateResults, run_simulation
)
from .vivado_integration import run_vivado_synthesis
from .workspace import workspace_manager, ingest_from_bundle


class ALG2SVPipeline:
    """Main pipeline for algorithm-to-SystemVerilog conversion."""

    def __init__(self, synthesis_backend: str = "auto", fpga_family: Optional[str] = None):
        self.results = {}
        self.current_workspace_token = None  # Global context for tools
        self.synthesis_backend = synthesis_backend
        self.fpga_family = fpga_family

    def _create_agents(self):
        """Create agents with current workspace context."""
        from .agents import read_file_tool, write_file_tool, set_workspace_context

        # Set workspace context for tools
        set_workspace_context(self.current_workspace_token)

        return {
            'spec': Agent(
                name="Spec Agent",
                instructions="""
                Analyze the input algorithm and generate hardware contract specifications.
                Use the read_file_tool to examine algorithm files and metadata in the workspace.

                Focus on:
                - Algorithm interface (step function signature)
                - Data types and ranges
                - Performance requirements from metadata
                - Resource constraints from metadata

                Output structured hardware contract specifications.
                """,
                tools=[read_file_tool],
                output_type=AgentOutputSchema(SpecContract, strict_json_schema=False)
            ),
            'quant': Agent(
                name="Quant Agent",
                instructions="""
                Convert floating-point algorithm to fixed-point representation.
                Use read_file_tool to access algorithm coefficients and specifications.

                Steps:
                1. Read algorithm and coefficient data
                2. Analyze dynamic ranges and precision requirements
                3. Determine optimal fixed-point formats
                4. Compute quantization error metrics
                5. Generate quantization configuration

                Output quantization config with error metrics.
                """,
                tools=[read_file_tool],
                output_type=AgentOutputSchema(QuantConfig, strict_json_schema=False)
            ),
            'microarch': Agent(
                name="MicroArch Agent",
                instructions="""
                Design micro-architecture for the algorithm based on previous specifications.
                Consider the quantization config and performance requirements.

                Design decisions:
                - Pipeline depth for timing closure
                - Parallelism and unroll factors
                - Memory mapping strategy
                - Dataflow optimization
                - Interface protocols (AXI-Stream)

                Output micro-architecture configuration.
                """,
                tools=[],  # Could add calculation tools
                output_type=AgentOutputSchema(MicroArchConfig, strict_json_schema=False)
            ),
            'rtl': Agent(
                name="RTL Agent",
                instructions="""
                Generate synthesizable SystemVerilog code based on all previous specifications.
                Use write_file_tool to create RTL files in the workspace.

                Generate:
                - Core datapath module with fixed-point arithmetic
                - AXI-Stream interfaces for streaming I/O
                - Pipeline registers for timing
                - Parameter files for configuration
                - Coefficient memory initialization

                Ensure code is:
                - Synthesizable (no unsupported constructs)
                - Properly pipelined for timing
                - Lint-clean and well-commented
                """,
                tools=[write_file_tool],
                output_type=AgentOutputSchema(RTLConfig, strict_json_schema=False)
            ),
            'verify': Agent(
                name="Verify Agent",
                instructions="""
                Run functional verification against the golden Python reference.
                Use read_file_tool to access test vectors and RTL files.

                Verification steps:
                1. Load golden reference model and test vectors
                2. Simulate RTL behavior (or estimate)
                3. Compare outputs bit-exact or within tolerance
                4. Report pass/fail with detailed metrics

                Focus on numerical accuracy and streaming behavior.
                """,
                tools=[read_file_tool],
                output_type=AgentOutputSchema(VerifyResults, strict_json_schema=False)
            ),
            'synth': Agent(
                name="Synth Agent",
                instructions=self._get_synthesis_instructions(),
                tools=self._get_synthesis_tools(),
                output_type=AgentOutputSchema(SynthResults, strict_json_schema=False)
            ),
            'lint': Agent(
                name="Lint Agent",
                instructions="""
                Perform comprehensive linting and code quality analysis on generated SystemVerilog RTL.
                Use read_file_tool to access RTL files and analyze them for issues.

                Check for:
                - Syntax errors and compilation issues
                - Style violations (naming, formatting, indentation)
                - Lint violations (unused signals, combinational loops, etc.)
                - Code quality metrics (complexity, maintainability)
                - FPGA-specific best practices
                - Synthesizability issues

                Provide detailed issues list and overall quality score.
                """,
                tools=[read_file_tool],
                output_type=AgentOutputSchema(LintResults, strict_json_schema=False)
            ),
            'simulate': Agent(
                name="Simulate Agent",
                instructions="""
                Run RTL simulation and functional verification.
                Use run_simulation tool for actual simulation (MCP integration) and read_file_tool for analysis.

                Simulation tasks:
                - Run testbench simulation using run_simulation tool
                - Check timing constraints and violations
                - Generate coverage metrics
                - Validate AXI-Stream protocol compliance
                - Test edge cases and error conditions
                - Report simulation errors and failures

                Use the run_simulation function to execute actual RTL simulation.
                Analyze results for correctness and identify any issues.
                """,
                tools=[read_file_tool, run_simulation],
                output_type=AgentOutputSchema(SimulateResults, strict_json_schema=False)
            ),
            'evaluate': Agent(
                name="Evaluate Agent",
                instructions="""
                Perform comprehensive evaluation of the complete ALG2SV pipeline results.
                Analyze all previous stages and provide holistic assessment.

                Evaluate:
                - Overall design quality and completeness
                - Performance vs requirements (timing, throughput, latency)
                - Resource efficiency and utilization
                - Code quality and maintainability
                - Verification completeness and correctness
                - Optimization opportunities and bottlenecks

                Provide actionable recommendations for improvement.
                """,
                tools=[read_file_tool],
                output_type=AgentOutputSchema(EvaluateResults, strict_json_schema=False)
            )
        }

    async def run(self, algorithm_bundle: str) -> Dict[str, Any]:
        """
        Run the complete ALG2SV pipeline.

        Args:
            algorithm_bundle: String containing algorithm files in fence format

        Returns:
            Dict with pipeline results and generated files
        """
        try:
            # Step 1: Ingest algorithm bundle
            print("📥 Ingesting algorithm bundle...")
            ingest_result = ingest_from_bundle(algorithm_bundle)
            if not ingest_result.get('success'):
                raise ValueError(f"Failed to ingest bundle: {ingest_result.get('error')}")

            workspace_token = ingest_result['workspace_token']
            print(f"✅ Created workspace with {ingest_result['count']} files")

            # Set workspace context for tools
            self.current_workspace_token = workspace_token
            self.agents = self._create_agents()

            # Step 2: Run Spec Agent
            print("🔍 Running Spec Agent...")
            spec_result = await self._run_agent_with_context(
                'spec', "Generate hardware contract from algorithm files in workspace"
            )
            self.results['spec'] = spec_result
            print(f"✅ Spec: {spec_result.name} - {spec_result.clock_mhz_target}MHz target")

            # Step 3: Run Quant Agent
            print("🔢 Running Quant Agent...")
            quant_result = await self._run_agent_with_context(
                'quant', f"Convert to fixed-point: {spec_result.input_format}"
            )
            self.results['quant'] = quant_result
            max_error = quant_result.error_metrics.get('max_abs_error', 'N/A')
            print(f"✅ Quant: {len(quant_result.quantized_coefficients)} coeffs, error={max_error}")

            # Step 4: Run MicroArch Agent
            print("🏗️ Running MicroArch Agent...")
            microarch_result = await self._run_agent_with_context(
                'microarch', f"Design architecture for {quant_result.fixed_point_config}"
            )
            self.results['microarch'] = microarch_result
            print(f"✅ MicroArch: {microarch_result.pipeline_depth} stages, {microarch_result.dsp_usage_estimate} DSPs")

            # Step 5: Run RTL Agent
            print("💾 Running RTL Agent...")
            rtl_result = await self._run_agent_with_context(
                'rtl', f"Generate SV for {microarch_result.handshake_protocol} interface"
            )
            self.results['rtl'] = rtl_result
            print(f"✅ RTL: Generated {len(rtl_result.rtl_files)} files, top={rtl_result.top_module}")

            # Step 6: Run Verify Agent
            print("✅ Running Verify Agent...")
            verify_result = await self._run_agent_with_context(
                'verify', f"Verify {rtl_result.top_module} against golden reference"
            )
            self.results['verify'] = verify_result

            if verify_result.all_passed:
                print(f"✅ Verify: {verify_result.tests_passed}/{verify_result.tests_total} tests passed")
            else:
                print(f"⚠️ Verify: {verify_result.tests_passed}/{verify_result.tests_total} tests passed - continuing to synthesis")

            # Step 7: Run Synth Agent (if verification passed)
            print("🔨 Running Synth Agent...")
            synth_context = f"Synthesize {rtl_result.top_module} using {self.synthesis_backend} backend"
            if self.fpga_family:
                synth_context += f" for {self.fpga_family} FPGA"
            synth_result = await self._run_agent_with_context('synth', synth_context)
            self.results['synth'] = synth_result

            # Step 8: Run Lint Agent
            print("🔍 Running Lint Agent...")
            lint_result = await self._run_agent_with_context(
                'lint', f"Lint and analyze quality of {rtl_result.top_module} RTL"
            )
            self.results['lint'] = lint_result
            print(f"✅ Lint: Score {lint_result.overall_score:.1f}/100, {lint_result.critical_issues} critical issues")

            # Step 9: Run Simulate Agent
            print("🎯 Running Simulate Agent...")
            simulate_result = await self._run_agent_with_context(
                'simulate', f"Simulate and test {rtl_result.top_module} functionality"
            )
            self.results['simulate'] = simulate_result
            print(f"✅ Simulate: {simulate_result.test_passed}/{simulate_result.test_total} tests passed")

            # Step 10: Run Evaluate Agent
            print("📊 Running Evaluate Agent...")
            evaluate_result = await self._run_agent_with_context(
                'evaluate', "Comprehensive evaluation of complete ALG2SV pipeline results"
            )
            self.results['evaluate'] = evaluate_result
            print(f"✅ Evaluate: Overall score {evaluate_result.overall_score:.1f}/100")

            # Check final constraints
            budget_check = self._check_resource_budget(spec_result, synth_result)

            if synth_result.timing_met and budget_check['within_budget']:
                print("🎉 Pipeline completed successfully!")
                print(f"   Target: {spec_result.clock_mhz_target}MHz, Achieved: {synth_result.fmax_mhz:.1f}MHz")
                print(f"   Resources: {synth_result.lut_usage} LUTs, {synth_result.ff_usage} FFs, {synth_result.dsp_usage} DSPs")
                return self._create_success_result(workspace_token)
            else:
                issues = []
                if not synth_result.timing_met:
                    issues.append(f"Timing not met: {synth_result.fmax_mhz:.1f}MHz < {spec_result.clock_mhz_target}MHz")
                if not budget_check['within_budget']:
                    issues.append(f"Resource budget exceeded: {budget_check['details']}")
                return self._create_error_result("Synthesis constraints not met", issues)

        except Exception as e:
            return self._create_error_result(f"Pipeline failed: {str(e)}")

    async def _run_agent_with_context(self, agent_name: str, context: str) -> Any:
        """Run an agent with workspace context."""
        agent = self.agents[agent_name]

        # Create input message with context
        # Convert Pydantic objects to dicts for JSON serialization
        serializable_results = {}
        for key, value in self.results.items():
            if hasattr(value, 'model_dump'):  # Pydantic v2
                serializable_results[key] = value.model_dump()
            elif hasattr(value, 'dict'):  # Pydantic v1
                serializable_results[key] = value.dict()
            else:
                serializable_results[key] = value

        input_message = f"""
{context}

Use the available tools to access files in the workspace and previous results: {json.dumps(serializable_results, indent=2)}

Provide your output in the required structured format.
"""

        # Run agent workflow using real OpenAI Agents SDK
        result = await Runner.run(
            starting_agent=agent,
            input=input_message
        )

        # Extract the final output (structured data for agents with output_type)
        if hasattr(result, 'final_output'):
            return result.final_output
        else:
            # Fallback for agents without structured output
            return result.final_output if hasattr(result, 'final_output') else {}

    def _check_resource_budget(self, spec: SpecContract, synth: SynthResults) -> Dict[str, Any]:
        """Check if synthesis results meet resource budget."""
        details = []
        within_budget = True

        for resource, budget in spec.resource_budget.items():
            actual = getattr(synth, f"{resource}_usage", 0)
            if actual > budget:
                within_budget = False
                details.append(f"{resource.upper()}: {actual} > {budget}")

        return {
            "within_budget": within_budget,
            "details": "; ".join(details) if details else "All within budget"
        }

    def _create_success_result(self, workspace_token: str) -> Dict[str, Any]:
        """Create successful pipeline result."""
        workspace = workspace_manager.get_workspace(workspace_token)

        return {
            "success": True,
            "status": "completed",
            "workspace_token": workspace_token,
            "generated_files": workspace.list_files() if workspace else [],
            "results": self.results,
            "summary": {
                "algorithm": self.results.get('spec', {}).name if 'spec' in self.results else "Unknown",
                "target_frequency": self.results.get('spec', {}).clock_mhz_target if 'spec' in self.results else 0,
                "achieved_frequency": self.results.get('synth', {}).fmax_mhz if 'synth' in self.results else 0,
                "resource_usage": {
                    "lut": self.results.get('synth', {}).lut_usage if 'synth' in self.results else 0,
                    "ff": self.results.get('synth', {}).ff_usage if 'synth' in self.results else 0,
                    "dsp": self.results.get('synth', {}).dsp_usage if 'synth' in self.results else 0,
                    "bram": self.results.get('synth', {}).bram_usage if 'synth' in self.results else 0,
                },
                "verification_passed": self.results.get('verify', {}).all_passed if 'verify' in self.results else False
            }
        }

    def _create_error_result(self, message: str, details: Any = None) -> Dict[str, Any]:
        """Create error pipeline result."""
        return {
            "success": False,
            "status": "failed",
            "error": message,
            "details": details,
            "partial_results": self.results
        }


async def run_pipeline(
    algorithm_bundle: str,
    synthesis_backend: str = "auto",
    fpga_family: Optional[str] = None
) -> Dict[str, Any]:
    """
    Convenience function to run the ALG2SV pipeline.

    Args:
        algorithm_bundle: Algorithm files in fence format
        synthesis_backend: Synthesis backend to use ('auto', 'vivado', 'yosys', 'symbiflow')
        fpga_family: FPGA family for synthesis (e.g., 'xc7a100t', 'ice40hx8k')

    Returns:
        Pipeline results
    """
    pipeline = ALG2SVPipeline(synthesis_backend=synthesis_backend, fpga_family=fpga_family)
    return await pipeline.run(algorithm_bundle)


def load_bundle_from_file(filepath: str) -> str:
    """
    Load algorithm bundle from file.

    Args:
        filepath: Path to bundle file

    Returns:
        Bundle content as string
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Bundle file not found: {filepath}")

    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


# Synchronous wrapper for command-line usage
def run_pipeline_sync(
    algorithm_bundle: str,
    synthesis_backend: str = "auto",
    fpga_family: Optional[str] = None
) -> Dict[str, Any]:
    """Synchronous wrapper for the pipeline.

    Args:
        algorithm_bundle: Algorithm bundle string
        synthesis_backend: Synthesis backend to use ('auto', 'vivado', 'yosys', 'symbiflow')
        fpga_family: FPGA family for synthesis (e.g., 'xc7a100t', 'ice40hx8k')

    Returns:
        Pipeline execution results
    """
    try:
        # Try to run in existing event loop (for Jupyter/async contexts)
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # We're in an async context, need to handle differently
            import nest_asyncio
            nest_asyncio.apply()
        return asyncio.run(run_pipeline(algorithm_bundle, synthesis_backend, fpga_family))
    except RuntimeError:
        # No event loop, create one
        return asyncio.run(run_pipeline(algorithm_bundle, synthesis_backend, fpga_family))


# Helper methods for synthesis backend configuration
def _get_synthesis_instructions(self) -> str:
    """Get synthesis instructions based on selected backend."""
    if self.synthesis_backend == "vivado":
        return """
        Run REAL FPGA synthesis using Xilinx Vivado to generate actual implementation results.
        Use run_vivado_synthesis tool to perform actual synthesis, implementation, and bitstream generation.

        Tasks:
        - Identify RTL files from workspace
        - Determine top module and FPGA part from metadata or arguments
        - Run complete Vivado synthesis flow
        - Parse utilization and timing reports
        - Generate bitstream if possible
        - Report actual FPGA resource usage and timing

        Use the run_vivado_synthesis tool to execute real synthesis instead of estimation.
        """
    elif self.synthesis_backend == "yosys":
        return """
        Run FPGA synthesis using Yosys open-source toolchain for iCE40/ECP5 FPGAs.
        Use run_yosys_synthesis tool to perform synthesis and generate bitstreams.

        Tasks:
        - Identify RTL files from workspace
        - Determine FPGA family (iCE40 or ECP5)
        - Run Yosys synthesis with appropriate target
        - Use nextpnr for place and route
        - Generate .bin bitstream using icepack
        - Report resource utilization and timing estimates

        Use the run_yosys_synthesis tool for open-source FPGA synthesis.
        """
    elif self.synthesis_backend == "symbiflow":
        return """
        Run FPGA synthesis using SymbiFlow for Xilinx 7-series FPGAs.
        Use run_symbiflow_synthesis tool for experimental Xilinx synthesis.

        Tasks:
        - Identify RTL files from workspace
        - Configure for Xilinx 7-series FPGA
        - Run SymbiFlow synthesis toolchain
        - Generate .bit bitstream if successful
        - Report resource utilization and timing

        Use the run_symbiflow_synthesis tool for experimental Xilinx synthesis.
        """
    else:  # auto
        return """
        Run FPGA synthesis using available tools (auto-detect best option).
        Check for Vivado first, then try open-source alternatives.

        Tasks:
        - Identify RTL files from workspace
        - Auto-detect FPGA family from metadata
        - Try Vivado synthesis first (if available)
        - Fall back to Yosys/nextpnr for open-source synthesis
        - Generate appropriate bitstream format
        - Report synthesis results and resource usage

        Use run_vivado_synthesis or run_yosys_synthesis based on availability.
        """


def _get_synthesis_tools(self) -> List:
    """Get synthesis tools based on selected backend."""
    if self.synthesis_backend == "vivado":
        from .vivado_integration import run_vivado_synthesis
        return [read_file_tool, run_vivado_synthesis]
    elif self.synthesis_backend == "yosys":
        from .open_source_synthesis import run_yosys_synthesis
        return [read_file_tool, run_yosys_synthesis]
    elif self.synthesis_backend == "symbiflow":
        from .open_source_synthesis import run_symbiflow_synthesis
        return [read_file_tool, run_symbiflow_synthesis]
    else:  # auto
        from .vivado_integration import run_vivado_synthesis
        from .open_source_synthesis import run_yosys_synthesis, run_symbiflow_synthesis
        return [read_file_tool, run_vivado_synthesis, run_yosys_synthesis, run_symbiflow_synthesis]
