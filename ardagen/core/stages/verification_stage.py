"""
Unified verification stage with three phases: lint, test generation, simulation.
"""

from __future__ import annotations

from typing import Any, Dict, TYPE_CHECKING

from .base import Stage, StageContext
from ...domain import RTLConfig, VerifyResults, LintResults

if TYPE_CHECKING:
    from ...core.strategies import AgentStrategy


class VerificationStage(Stage):
    """
    Unified verification stage with three sequential phases:
    1. Static Analysis (lint with Verilator)
    2. Test Generation (agent generates tests + golden model)
    3. Simulation (agent runs RTL simulation vs golden)
    """

    name = "verification"
    dependencies = ("rtl",)
    output_model = VerifyResults

    def gather_inputs(self, context: StageContext) -> Dict[str, Any]:
        inputs = super().gather_inputs(context)
        if not isinstance(inputs["rtl"], RTLConfig):
            raise TypeError("VerificationStage requires RTLConfig from 'rtl' dependency.")
        return inputs

    def _determine_start_phase(self, context: StageContext) -> int:
        """Determine which phase to start from based on feedback decisions."""
        # Check for feedback guidance in run_inputs
        run_inputs = getattr(context, 'run_inputs', {})
        last_feedback = run_inputs.get('last_feedback')
        if last_feedback and 'guidance' in last_feedback:
            guidance = last_feedback['guidance']
            if guidance and guidance.startswith('retry_from_phase:'):
                phase_name = guidance.split(':', 1)[1]
                phase_map = {
                    'lint': 1,
                    'test_generation': 2,
                    'simulation': 3
                }
                return phase_map.get(phase_name, 1)

        # Default: start from phase 1
        return 1

    async def run(self, context: StageContext, strategy: "AgentStrategy") -> VerifyResults:
        """Execute all three verification phases."""

        rtl_config = context.results["rtl"]
        workspace_token = context.run_inputs.get("workspace_token")

        # Check if we should start from a specific phase due to feedback
        start_phase = self._determine_start_phase(context)

        # PHASE 1: Static Analysis (Lint)
        print("🔍 [Verification Phase 1/3] Running static analysis (lint)...")
        lint_result = self._run_lint_phase(rtl_config, workspace_token)
        
        if lint_result.critical_issues > 0:
            # Critical lint failures - stop immediately
            print(f"❌ Lint failed: {lint_result.critical_issues} critical issues")
            return VerifyResults(
                tests_total=0,
                tests_passed=0,
                all_passed=False,
                mismatches=[{"phase": "lint", "reason": "Critical lint failures"}],
                max_abs_error=0.0,
                rms_error=0.0,
                functional_coverage=0.0,
                confidence=0.0,
                lint_results=lint_result.model_dump()
            )
        
        print(f"✅ Lint passed: {lint_result.overall_score:.1f}/100 score")
        
        # PHASE 2: Test Generation
        print("🔍 [Verification Phase 2/3] Generating test vectors...")
        testgen_context = self._build_testgen_context(context, lint_result)
        # Add feedback information to help the agent know what to focus on
        if start_phase >= 2 and 'last_feedback' in context.run_inputs:
            testgen_context['stage_inputs']['feedback_guidance'] = context.run_inputs['last_feedback'].get('guidance', '')

        test_results = await strategy.run(
            TestGenerationSubStage(),
            testgen_context,
            context.run_inputs
        )

        # Debug: Print the full test generation output
        print(f"🔍 Test generation output: {test_results}")

        if not test_results or test_results.get("test_count", 0) == 0:
            print("❌ Test generation failed")
            return VerifyResults(
                tests_total=0,
                tests_passed=0,
                all_passed=False,
                mismatches=[{"phase": "test_generation", "reason": "No tests generated"}],
                max_abs_error=0.0,
                rms_error=0.0,
                functional_coverage=0.0,
                confidence=0.0,
                lint_results=lint_result.model_dump()
            )
        
        print(f"✅ Generated {test_results['test_count']} test vectors")
        
        # PHASE 3: Simulation
        print("🔍 [Verification Phase 3/3] Running RTL simulation...")
        sim_context = self._build_simulation_context(context, lint_result, test_results)
        # Add feedback information to help the agent know what to focus on
        if start_phase >= 3 and 'last_feedback' in context.run_inputs:
            sim_context['stage_inputs']['feedback_guidance'] = context.run_inputs['last_feedback'].get('guidance', '')

        sim_results = await strategy.run(
            SimulationSubStage(),
            sim_context,
            context.run_inputs
        )

        # Debug: Print the full simulation output
        print(f"🔍 Simulation output: {sim_results}")

        # Aggregate all results
        return VerifyResults(
            tests_total=sim_results.get("tests_total", 0),
            tests_passed=sim_results.get("tests_passed", 0),
            all_passed=sim_results.get("all_passed", False),
            mismatches=sim_results.get("mismatches", []),
            max_abs_error=sim_results.get("max_abs_error", 0.0),
            rms_error=sim_results.get("rms_error", 0.0),
            functional_coverage=sim_results.get("functional_coverage", 0.0),
            confidence=sim_results.get("confidence", 0.0),
            lint_results=lint_result.model_dump()
        )

    def _run_lint_phase(self, rtl_config: RTLConfig, workspace_token: str) -> LintResults:
        """Phase 1: Run Verilator-based linting."""
        from ...tools.lint import lint_rtl_with_verilator
        
        # RTLConfig uses file_paths, not rtl_files
        rtl_files = rtl_config.file_paths or []
        return lint_rtl_with_verilator(rtl_files, workspace_token)

    def _build_testgen_context(self, context: StageContext, lint_result: LintResults) -> Dict[str, Any]:
        """Build context for test generation agent."""
        return {
            "stage_inputs": {
                "rtl": context.results["rtl"].model_dump(),
                "lint_results": lint_result.model_dump()
            },
            "workspace_token": context.run_inputs.get("workspace_token"),
            "observability": context.run_inputs.get("observability"),
            "phase": "test_generation"
        }

    def _build_simulation_context(
        self,
        context: StageContext,
        lint_result: LintResults,
        test_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build context for simulation agent."""
        return {
            "stage_inputs": {
                "rtl": context.results["rtl"].model_dump(),
                "lint_results": lint_result.model_dump(),
                "test_vectors": test_results.get("test_vectors", []),
                "golden_outputs": test_results.get("golden_outputs", [])
            },
            "workspace_token": context.run_inputs.get("workspace_token"),
            "observability": context.run_inputs.get("observability"),
            "phase": "simulation"
        }

    def validate_output(self, output: VerifyResults, context: StageContext) -> None:
        """Enforce verification quality gate."""
        if not output.all_passed or output.tests_passed < output.tests_total:
            raise ValueError(
                f"Verification gate failed: {output.tests_passed}/{output.tests_total} tests passed"
            )


class TestGenerationSubStage(Stage):
    """Sub-stage for test generation phase."""
    name = "test_generation"
    dependencies = ()
    output_model = dict  # Returns dict with test_vectors, golden_outputs, test_count


class SimulationSubStage(Stage):
    """Sub-stage for simulation phase."""
    name = "simulation" 
    dependencies = ()
    output_model = dict  # Returns dict with sim results

