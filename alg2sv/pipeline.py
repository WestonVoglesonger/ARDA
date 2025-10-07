"""
Main ALG2SV pipeline orchestrator.
Runs the complete algorithm-to-RTL conversion workflow.
"""

import asyncio
import json
from typing import Dict, Any, Optional, List, Tuple

from agents import Runner, Agent, AgentOutputSchema  # Real OpenAI Agents SDK
from .agents import (
    SpecContract,
    QuantConfig,
    MicroArchConfig,
    RTLConfig,
    VerifyResults,
    SynthResults,
    LintResults,
    SimulateResults,
    EvaluateResults,
    FeedbackDecision,
    run_simulation,
    extract_test_vectors,
    list_workspace_files,
)
from .vivado_integration import run_vivado_synthesis
from .workspace import workspace_manager, ingest_from_bundle


class ALG2SVPipeline:
    """Main pipeline for algorithm-to-SystemVerilog conversion."""

    MAX_STAGE_ATTEMPTS = 3

    def __init__(self, synthesis_backend: str = "auto", fpga_family: Optional[str] = None):
        self.results = {}
        self.current_workspace_token = None  # Global context for tools
        self.synthesis_backend = synthesis_backend
        self.fpga_family = fpga_family
        self.stage_order = [
            'spec',
            'quant',
            'microarch',
            'rtl',
            'verify',
            'synth',
            'lint',
            'simulate',
            'evaluate',
        ]
        self._stage_index_map = {name: idx for idx, name in enumerate(self.stage_order)}
        self.stage_attempts: Dict[str, int] = {}

    def _create_agents(self):
        """Create agents with current workspace context."""
        from .agents import read_file_tool, write_file_tool, set_workspace_context
        # Set workspace context for tools
        set_workspace_context(self.current_workspace_token)

        return {
            'feedback': Agent(
                name="Feedback Agent",
                instructions="""
                Review aggregated pipeline outputs after each stage and decide how the workflow should proceed.

                Actions you may take:
                - `continue`: proceed to the next stage.
                - `retry_<stage>`: request a retry for a specific stage (e.g., retry_synth).
                - `tune_microarch`: revisit the micro-architecture design decisions.
                - `abort`: stop the pipeline due to irrecoverable issues.

                When recommending retries or adjustments, populate the `target_stage` and `guidance` fields with
                concise instructions describing what should change before rerunning. Always output a valid
                FeedbackDecision JSON object.
                """,
                tools=[],
                output_type=AgentOutputSchema(FeedbackDecision, strict_json_schema=False)
            ),
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
                🚨 MANDATORY: You MUST use simulation tools or this pipeline will FAIL!

                CRITICAL REQUIREMENTS:
                1. You MUST call extract_test_vectors(workspace_token) FIRST
                2. You MUST call list_workspace_files(workspace_token) to find RTL files
                3. You MUST derive the top module dynamically from the pipeline results JSON (no hard-coded names)
                4. You MUST call run_simulation() with real data
                5. You MUST return actual simulation results, not estimates

                CONSEQUENCES OF NOT USING TOOLS:
                - Pipeline will be marked as FAILED
                - Verification score will be 0/100
                - All tests will show as failed

                REQUIRED EXACT SEQUENCE:
                ```python
                # Step 1: Get test data (MANDATORY)
                test_data_result = extract_test_vectors(workspace_token)
                test_data = json.loads(test_data_result)
                print(f"Got {test_data['num_samples']} test samples")

                # Step 2: Find RTL files (MANDATORY)
                files_result = list_workspace_files(workspace_token)
                files_data = json.loads(files_result)
                rtl_files = files_data["rtl_files"]
                print(f"Found RTL files: {rtl_files}")

                # Step 3: Determine top module from pipeline results (MANDATORY)
                current_pipeline_results_json = """<paste the pipeline results JSON block from your prompt here>"""
                pipeline_results = json.loads(current_pipeline_results_json)
                rtl_info = pipeline_results.get("rtl", {})
                top_module = rtl_info.get("top_module")
                if not top_module:
                    raise ValueError("Top module missing from RTL stage results")

                # Step 4: Run simulation (MANDATORY)
                sim_result = run_simulation(
                    top_module=top_module,
                    rtl_files=rtl_files,
                    input_data=test_data["input_data"],
                    expected_data=test_data["expected_data"],
                    simulator="auto"
                )
                sim_data = json.loads(sim_result)
                print(f"Simulation completed: {sim_data['success']}")
                ```

                ⚠️ FAILURE TO USE THESE TOOLS = AUTOMATIC PIPELINE FAILURE
                """,
                tools=[read_file_tool, run_simulation, extract_test_vectors, list_workspace_files],
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
                🚨 MANDATORY: You MUST use simulation tools or this pipeline will FAIL!

                CRITICAL REQUIREMENTS:
                1. You MUST call extract_test_vectors(workspace_token) FIRST
                2. You MUST call list_workspace_files(workspace_token) to find RTL files
                3. You MUST derive the top module dynamically from the pipeline results JSON (no hard-coded names)
                4. You MUST call run_simulation() with real data
                5. You MUST return actual simulation results, not estimates

                CONSEQUENCES OF NOT USING TOOLS:
                - Pipeline will be marked as FAILED
                - Simulation score will be 0/100
                - All tests will show as failed

                REQUIRED EXACT SEQUENCE:
                ```python
                # Step 1: Get test data (MANDATORY)
                test_data_result = extract_test_vectors(workspace_token)
                test_data = json.loads(test_data_result)
                print(f"Got {test_data['num_samples']} test samples")

                # Step 2: Find RTL files (MANDATORY)
                files_result = list_workspace_files(workspace_token)
                files_data = json.loads(files_result)
                rtl_files = files_data["rtl_files"]
                print(f"Found RTL files: {rtl_files}")

                # Step 3: Determine top module from pipeline results (MANDATORY)
                current_pipeline_results_json = """<paste the pipeline results JSON block from your prompt here>"""
                pipeline_results = json.loads(current_pipeline_results_json)
                rtl_info = pipeline_results.get("rtl", {})
                top_module = rtl_info.get("top_module")
                if not top_module:
                    raise ValueError("Top module missing from RTL stage results")

                # Step 4: Run simulation (MANDATORY)
                sim_result = run_simulation(
                    top_module=top_module,
                    rtl_files=rtl_files,
                    input_data=test_data["input_data"],
                    expected_data=test_data["expected_data"],
                    simulator="auto"
                )
                sim_data = json.loads(sim_result)
                print(f"Simulation completed: {sim_data['success']}")
                ```

                ⚠️ FAILURE TO USE THESE TOOLS = AUTOMATIC PIPELINE FAILURE
                """,
                tools=[read_file_tool, run_simulation, extract_test_vectors, list_workspace_files],
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
        self.results = {}
        self.stage_attempts = {stage: 0 for stage in self.stage_order}
        stage_guidance: Dict[str, str] = {}

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

            stage_index = 0
            while stage_index < len(self.stage_order):
                stage_name = self.stage_order[stage_index]

                if self.stage_attempts[stage_name] >= self.MAX_STAGE_ATTEMPTS:
                    return self._create_error_result(
                        f"Exceeded retry limit for {stage_name} stage",
                        {"attempts": self.stage_attempts[stage_name], "stage": stage_name},
                    )

                guidance = stage_guidance.get(stage_name)
                context = self._build_stage_context(stage_name, guidance)
                self._log_stage_start(stage_name, self.stage_attempts[stage_name] + 1)

                stage_result = await self._run_agent_with_context(stage_name, context)
                self.results[stage_name] = stage_result
                self.stage_attempts[stage_name] += 1
                self._log_stage_success(stage_name, stage_result)

                decision = await self._run_feedback_agent(stage_name, stage_result)
                next_index, abort_details = self._handle_feedback_decision(
                    stage_name, stage_index, decision, stage_guidance
                )

                if abort_details is not None:
                    print(f"⛔ Feedback agent requested abort: {abort_details}")
                    return self._create_error_result("Pipeline aborted by feedback agent", abort_details)

                stage_index = next_index

            if 'spec' not in self.results or 'synth' not in self.results:
                return self._create_error_result(
                    "Pipeline completed without synthesis results",
                    "Missing spec or synth stage results",
                )

            spec_result = self.results['spec']
            synth_result = self.results['synth']
            budget_check = self._check_resource_budget(spec_result, synth_result)

            synth_timing_met = self._safe_get(synth_result, 'timing_met', False)

            if synth_timing_met and budget_check['within_budget']:
                target_freq = float(self._safe_get(spec_result, 'clock_mhz_target', 0.0))
                achieved_freq = float(self._safe_get(synth_result, 'fmax_mhz', 0.0))
                lut_usage = self._safe_get(synth_result, 'lut_usage', 0)
                ff_usage = self._safe_get(synth_result, 'ff_usage', 0)
                dsp_usage = self._safe_get(synth_result, 'dsp_usage', 0)
                print("🎉 Pipeline completed successfully!")
                print(f"   Target: {target_freq}MHz, Achieved: {achieved_freq:.1f}MHz")
                print(f"   Resources: {lut_usage} LUTs, {ff_usage} FFs, {dsp_usage} DSPs")
                return self._create_success_result(workspace_token)

            issues = []
            achieved_freq = float(self._safe_get(synth_result, 'fmax_mhz', 0.0))
            target_freq = float(self._safe_get(spec_result, 'clock_mhz_target', 0.0))
            if not synth_timing_met:
                issues.append(f"Timing not met: {achieved_freq:.1f}MHz < {target_freq}MHz")
            if not budget_check['within_budget']:
                issues.append(f"Resource budget exceeded: {budget_check['details']}")
            return self._create_error_result("Synthesis constraints not met", issues)

        except Exception as e:
            return self._create_error_result(f"Pipeline failed: {str(e)}")

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
        from .agents import read_file_tool

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

    async def _run_agent_with_context(self, agent_name: str, context: str) -> Any:
        """Run an agent with workspace context."""
        agent = self.agents.get(agent_name)
        if agent is None:
            raise KeyError(f"Agent '{agent_name}' is not registered in the pipeline")

        serializable_results = self._serialize_results()

        input_message = f"""
{context}

Current pipeline results (JSON): {json.dumps(serializable_results, indent=2)}

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

    def _serialize_value(self, value: Any) -> Any:
        """Convert a stage result to a JSON-serializable object."""
        if value is None:
            return None
        if hasattr(value, 'model_dump'):
            return value.model_dump()
        if hasattr(value, 'dict'):
            return value.dict()
        return value

    def _serialize_results(self) -> Dict[str, Any]:
        """Serialize all recorded stage results."""
        return {key: self._serialize_value(val) for key, val in self.results.items()}

    def _build_stage_context(self, stage_name: str, guidance: Optional[str] = None) -> str:
        """Construct context prompt for a specific stage including any feedback guidance."""
        context: str
        if stage_name == 'spec':
            context = "Generate hardware contract from algorithm files in the workspace."
        elif stage_name == 'quant':
            spec = self.results.get('spec')
            spec_input = self._safe_get(spec, 'input_format', {})
            context = f"Convert to fixed-point representation using spec input format {spec_input}."
        elif stage_name == 'microarch':
            quant = self.results.get('quant')
            quant_config = self._safe_get(quant, 'fixed_point_config', {})
            context = f"Design the micro-architecture informed by quantization config: {quant_config}."
        elif stage_name == 'rtl':
            microarch = self.results.get('microarch')
            handshake = self._safe_get(microarch, 'handshake_protocol', 'ready_valid')
            context = f"Generate synthesizable SystemVerilog implementing the {handshake} interface and prior specs."
        elif stage_name == 'verify':
            rtl = self.results.get('rtl')
            top_module = self._safe_get(rtl, 'top_module', 'top')
            context = f"Verify {top_module} against the golden reference using the required simulation tools."
        elif stage_name == 'synth':
            rtl = self.results.get('rtl')
            top_module = self._safe_get(rtl, 'top_module', 'top')
            context = f"Synthesize {top_module} using the {self.synthesis_backend} backend."
            if self.fpga_family:
                context += f" Target FPGA family: {self.fpga_family}."
        elif stage_name == 'lint':
            rtl = self.results.get('rtl')
            top_module = self._safe_get(rtl, 'top_module', 'top')
            context = f"Lint and analyze the quality of {top_module} RTL files."
        elif stage_name == 'simulate':
            rtl = self.results.get('rtl')
            top_module = self._safe_get(rtl, 'top_module', 'top')
            context = f"Run RTL simulations for {top_module} using provided test vectors."
        elif stage_name == 'evaluate':
            context = "Provide a holistic evaluation of all prior stage outputs and identify improvements."
        else:
            context = f"Execute the {stage_name} stage with awareness of prior results."

        if guidance:
            context += f"\n\nFeedback guidance for this stage: {guidance}"

        return context

    def _build_feedback_context(self, stage_name: str, stage_result: Any) -> str:
        """Create the context message for the feedback agent."""
        serialized_stage = self._serialize_value(stage_result)
        attempts_snapshot = {stage: attempts for stage, attempts in self.stage_attempts.items()}
        return f"""
Assess the pipeline progress after completing stage '{stage_name}'.

Stage attempts so far: {json.dumps(attempts_snapshot, indent=2)}
Most recent stage output: {json.dumps(serialized_stage, indent=2)}

Decide whether to continue, retry a stage (use action retry_<stage>), tune_microarch, or abort.
If providing updated instructions, include target_stage and guidance fields.
"""

    async def _run_feedback_agent(self, stage_name: str, stage_result: Any) -> FeedbackDecision:
        """Invoke the feedback agent and normalize its decision output."""
        if 'feedback' not in self.agents:
            return FeedbackDecision(action='continue')

        context = self._build_feedback_context(stage_name, stage_result)
        decision_output = await self._run_agent_with_context('feedback', context)

        if isinstance(decision_output, FeedbackDecision):
            return decision_output

        if isinstance(decision_output, dict):
            try:
                return FeedbackDecision(**decision_output)
            except Exception:
                pass

        if hasattr(FeedbackDecision, 'model_validate'):
            try:
                return FeedbackDecision.model_validate(decision_output)
            except Exception:
                pass

        return FeedbackDecision(action='continue')

    def _prepare_stage_retry(
        self,
        target_stage: str,
        guidance: Optional[str],
        stage_guidance: Dict[str, str],
    ) -> bool:
        """Invalidate downstream results and store guidance for a retry."""
        if target_stage not in self._stage_index_map:
            return False

        self._invalidate_results_from(target_stage)
        if guidance:
            stage_guidance[target_stage] = guidance
        return True

    def _handle_feedback_decision(
        self,
        current_stage: str,
        current_index: int,
        decision: FeedbackDecision,
        stage_guidance: Dict[str, str],
    ) -> Tuple[int, Optional[str]]:
        """Process feedback action and determine the next stage index."""
        if not isinstance(decision, FeedbackDecision):
            return current_index + 1, None

        action = decision.action
        guidance = decision.guidance
        target_stage = decision.target_stage

        next_index = current_index + 1
        abort_details: Optional[str] = None
        retry_prepared = True

        if action == 'abort':
            abort_details = guidance or f"Abort requested after stage '{current_stage}'"
        elif action == 'tune_microarch':
            retry_target = target_stage or 'microarch'
            retry_prepared = self._prepare_stage_retry(retry_target, guidance, stage_guidance)
            if retry_prepared:
                next_index = self._stage_index_map[retry_target]
            target_stage = retry_target
        elif action.startswith('retry'):
            _, _, suffix = action.partition('_')
            retry_target = suffix or target_stage
            if retry_target:
                retry_prepared = self._prepare_stage_retry(retry_target, guidance, stage_guidance)
                if retry_prepared:
                    next_index = self._stage_index_map[retry_target]
                target_stage = retry_target
        else:
            if guidance and target_stage:
                stage_guidance[target_stage] = guidance

        self._log_feedback_action(action, current_stage, target_stage, guidance, abort_details, retry_prepared)
        return next_index, abort_details

    def _invalidate_results_from(self, stage_name: str) -> None:
        """Remove cached results from the specified stage onward."""
        if stage_name not in self._stage_index_map:
            return

        start_idx = self._stage_index_map[stage_name]
        to_remove = [name for name in self.stage_order[start_idx:] if name in self.results]
        for name in to_remove:
            self.results.pop(name, None)

    def _log_stage_start(self, stage_name: str, attempt_no: int) -> None:
        """Log the start of a stage, indicating retry attempts when applicable."""
        messages = {
            'spec': "🔍 Running Spec Agent",
            'quant': "🔢 Running Quant Agent",
            'microarch': "🏗️ Running MicroArch Agent",
            'rtl': "💾 Running RTL Agent",
            'verify': "✅ Running Verify Agent",
            'synth': "🔨 Running Synth Agent",
            'lint': "🔍 Running Lint Agent",
            'simulate': "🎯 Running Simulate Agent",
            'evaluate': "📊 Running Evaluate Agent",
        }
        message = messages.get(stage_name, f"▶️ Running {stage_name.title()} stage")
        if attempt_no > 1:
            message = f"{message} (attempt {attempt_no})"
        print(f"{message}...")

    def _log_stage_success(self, stage_name: str, stage_result: Any) -> None:
        """Log concise success information for a completed stage."""
        if stage_name == 'spec':
            name = self._safe_get(stage_result, 'name', 'Unknown')
            target = self._safe_get(stage_result, 'clock_mhz_target', 'N/A')
            print(f"✅ Spec: {name} - {target}MHz target")
        elif stage_name == 'quant':
            coeffs = self._safe_get(stage_result, 'quantized_coefficients', []) or []
            error_metrics = self._safe_get(stage_result, 'error_metrics', {}) or {}
            max_error = error_metrics.get('max_abs_error', 'N/A')
            print(f"✅ Quant: {len(coeffs)} coeffs, error={max_error}")
        elif stage_name == 'microarch':
            depth = self._safe_get(stage_result, 'pipeline_depth', 'N/A')
            dsp_est = self._safe_get(stage_result, 'dsp_usage_estimate', 'N/A')
            print(f"✅ MicroArch: {depth} stages, {dsp_est} DSPs")
        elif stage_name == 'rtl':
            files = self._safe_get(stage_result, 'rtl_files', []) or []
            top_module = self._safe_get(stage_result, 'top_module', 'unknown')
            print(f"✅ RTL: Generated {len(files)} files, top={top_module}")
        elif stage_name == 'verify':
            tests_total = self._safe_get(stage_result, 'tests_total', 0)
            tests_passed = self._safe_get(stage_result, 'tests_passed', 0)
            if self._safe_get(stage_result, 'all_passed', False):
                print(f"✅ Verify: {tests_passed}/{tests_total} tests passed")
            else:
                print(f"⚠️ Verify: {tests_passed}/{tests_total} tests passed - continuing to synthesis")
        elif stage_name == 'synth':
            fmax = self._safe_get(stage_result, 'fmax_mhz', 'N/A')
            timing_met = self._safe_get(stage_result, 'timing_met', False)
            slack = self._safe_get(stage_result, 'slack_ns', 'N/A')
            status = "timing met" if timing_met else "timing NOT met"
            print(f"✅ Synth: fmax={fmax}MHz, slack={slack}ns ({status})")
        elif stage_name == 'lint':
            score_raw = self._safe_get(stage_result, 'overall_score', 0)
            try:
                score_value = float(score_raw)
            except (TypeError, ValueError):
                score_value = 0.0
            critical = self._safe_get(stage_result, 'critical_issues', 0)
            print(f"✅ Lint: Score {score_value:.1f}/100, {critical} critical issues")
        elif stage_name == 'simulate':
            passed = self._safe_get(stage_result, 'test_passed', 0)
            total = self._safe_get(stage_result, 'test_total', 0)
            print(f"✅ Simulate: {passed}/{total} tests passed")
        elif stage_name == 'evaluate':
            overall_raw = self._safe_get(stage_result, 'overall_score', 0)
            try:
                overall_value = float(overall_raw)
            except (TypeError, ValueError):
                overall_value = 0.0
            print(f"✅ Evaluate: Overall score {overall_value:.1f}/100")
        else:
            print(f"✅ {stage_name.title()} stage completed")

    def _log_feedback_action(
        self,
        action: str,
        current_stage: str,
        target_stage: Optional[str],
        guidance: Optional[str],
        abort_details: Optional[str],
        retry_prepared: bool,
    ) -> None:
        """Emit log messages describing the feedback agent's directive."""
        if action == 'abort':
            message = abort_details or guidance or f"Abort requested after stage '{current_stage}'"
            print(f"⛔ Feedback: {message}")
            return

        if action == 'continue':
            if guidance and target_stage:
                print(f"📝 Feedback: Continue, but update '{target_stage}' with guidance: {guidance}")
            else:
                print("➡️ Feedback: Continue to next stage")
            return

        if action == 'tune_microarch' or action.startswith('retry'):
            stage_label = target_stage or action.split('_', 1)[-1]
            if not retry_prepared:
                print(f"⚠️ Feedback: Requested retry for unknown stage '{stage_label}'. Ignoring request.")
            else:
                guidance_text = f" Guidance: {guidance}" if guidance else ""
                print(f"↩️ Feedback: Retry stage '{stage_label}'.{guidance_text}")
            return

        info = guidance or "No additional guidance provided."
        print(f"ℹ️ Feedback: Action '{action}' after stage '{current_stage}'. {info}")

    def _safe_get(self, obj: Any, attr: str, default: Any = None) -> Any:
        """Safely access attribute or dict key from objects or Pydantic models."""
        if obj is None:
            return default
        if hasattr(obj, attr):
            return getattr(obj, attr)
        if isinstance(obj, dict):
            return obj.get(attr, default)
        return default

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
        from .agents import read_file_tool
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


async def run_pipeline(
    algorithm_bundle: str,
    synthesis_backend: str = "auto",
    fpga_family: Optional[str] = None
) -> Dict[str, Any]:
    """
    Convenience function to run the ALG2SV pipeline.

    Args:
        algorithm_bundle: Algorithm files in fence format
        synthesis_backend: FPGA synthesis backend ('auto', 'vivado', 'yosys', 'symbiflow')
        fpga_family: FPGA family for synthesis (e.g., 'xc7a100t', 'ice40hx8k')

    Returns:
        Pipeline results
    """
    pipeline = ALG2SVPipeline(synthesis_backend=synthesis_backend, fpga_family=fpga_family)
    return await pipeline.run(algorithm_bundle)


def run_pipeline_sync(
    algorithm_bundle: str,
    synthesis_backend: str = "auto",
    fpga_family: Optional[str] = None
) -> Dict[str, Any]:
    """
    Synchronous wrapper for the ALG2SV pipeline.

    Args:
        algorithm_bundle: Algorithm files in fence format
        synthesis_backend: FPGA synthesis backend ('auto', 'vivado', 'yosys', 'symbiflow')
        fpga_family: FPGA family for synthesis (e.g., 'xc7a100t', 'ice40hx8k')

    Returns:
        Pipeline results
    """
    try:
        import nest_asyncio
        nest_asyncio.apply()
        return asyncio.run(run_pipeline(algorithm_bundle, synthesis_backend, fpga_family))
    except RuntimeError:
        # No event loop, create one
        return asyncio.run(run_pipeline(algorithm_bundle, synthesis_backend, fpga_family))


def load_bundle_from_file(filepath: str) -> str:
    """
    Load algorithm bundle from file.

    Args:
        filepath: Path to bundle file

    Returns:
        Bundle content as string
    """
    with open(filepath, 'r') as f:
        return f.read()
