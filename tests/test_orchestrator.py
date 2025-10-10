import asyncio
from typing import Any, Dict, Mapping

import pytest

from ardagen.core import PipelineOrchestrator, PipelineRunResult
from ardagen.core.stages import (
    SpecStage,
    QuantStage,
    MicroArchStage,
    RTLStage,
    StaticChecksStage,
    VerificationStage,
    SynthStage,
    EvaluateStage,
    Stage,
)
from ardagen.core.strategies import AgentStrategy
from ardagen.domain import (
    SpecContract,
    QuantConfig,
    MicroArchConfig,
    RTLConfig,
    LintResults,
    VerifyResults,
    SynthResults,
    EvaluateResults,
)


class StubStrategy(AgentStrategy):
    def __init__(self, outputs: Dict[str, Any]):
        self.outputs = outputs
        self.calls: Dict[str, Dict[str, Any]] = {}

    async def run(
        self,
        stage: Stage,
        stage_inputs: Dict[str, Any],
        run_inputs: Mapping[str, Any],
    ) -> Any:
        self.calls[stage.name] = {
            "inputs": stage_inputs,
            "run_inputs": dict(run_inputs),
        }
        return self.outputs[stage.name]


@pytest.mark.asyncio
async def test_orchestrator_executes_stage_sequence_with_quality_gates():
    spec_output = SpecContract(
        name="Demo",
        description="demo design",
        clock_mhz_target=200.0,
        throughput_samples_per_cycle=1,
        input_format={"width": 16, "fractional_bits": 14},
        output_format={"width": 16, "fractional_bits": 14},
        resource_budget={"lut": 5000, "ff": 6000, "dsp": 20, "bram": 4},
        verification_config={"num_samples": 16},
    )
    quant_output = QuantConfig(
        fixed_point_config={"input_width": 16, "output_width": 16},
        error_metrics={"max_abs_error": 1e-6},
        quantized_coefficients=[0.1, 0.2, 0.3],
        fxp_model_path="model.py",
    )
    microarch_output = MicroArchConfig(
        pipeline_depth=4,
        unroll_factor=2,
        memory_config={"buffer_depth": 8},
        dsp_usage_estimate=8,
        estimated_latency_cycles=6,
        handshake_protocol="ready_valid",
    )
    rtl_output = RTLConfig(
        generated_files={
            "params_svh": "package params; endpackage",
            "algorithm_core_sv": "module core; endmodule",
            "algorithm_top_sv": "module demo_top; endmodule"
        },
        file_paths=["rtl/params.svh", "rtl/core.sv", "rtl/demo_top.sv"],
        rtl_files=["rtl/core.sv"],  # Deprecated field for backward compat
        params_file="rtl/params.svh",
        top_module="demo_top",
        lint_passed=True,
        estimated_resources={"lut": 1200},
    )
    lint_output = LintResults(
        syntax_errors=0,
        style_warnings=0,
        lint_violations=0,
        critical_issues=0,
        issues_list=[],
        overall_score=95.0,
        lint_clean=True,
    )
    verification_output = VerifyResults(
        tests_total=10,
        tests_passed=10,
        all_passed=True,
        mismatches=[],
        max_abs_error=1.0e-6,
        rms_error=5.0e-7,
        functional_coverage=0.95,
    )
    synth_output = SynthResults(
        fmax_mhz=250.0,
        timing_met=True,
        lut_usage=2000,
        ff_usage=3000,
        dsp_usage=12,
        bram_usage=4,
        total_power_mw=45.0,
        slack_ns=0.5,
        reports_path="/tmp/reports",
    )
    evaluate_output = EvaluateResults(
        overall_score=90.0,
        performance_score=92.0,
        resource_score=88.0,
        quality_score=93.0,
        correctness_score=95.0,
        recommendations=[],
        bottlenecks=[],
        optimization_opportunities=[],
    )

    strategy = StubStrategy(
        {
            "spec": spec_output,
            "quant": quant_output,
            "microarch": microarch_output,
            "rtl": rtl_output,
            "static_checks": lint_output,
            "verification": verification_output,
            "synth": synth_output,
            "evaluate": evaluate_output,
        }
    )
    orchestrator = PipelineOrchestrator(
        stages=[
            SpecStage(),
            QuantStage(),
            MicroArchStage(),
            RTLStage(),
            StaticChecksStage(),
            VerificationStage(),
            SynthStage(),
            EvaluateStage(),
        ],
        strategy=strategy,
    )

    result: PipelineRunResult = await orchestrator.run({"bundle": "fake bundle"})

    assert result.stages == [
        "spec",
        "quant",
        "microarch",
        "rtl",
        "static_checks",
        "verification",
        "synth",
        "evaluate",
    ]
    assert result.get("spec") == spec_output
    assert result.get("quant") == quant_output
    assert result.get("microarch") == microarch_output
    assert result.get("rtl") == rtl_output
    assert result.get("static_checks") == lint_output
    assert result.get("verification") == verification_output
    assert result.get("synth") == synth_output
    assert result.get("evaluate") == evaluate_output

    assert strategy.calls["spec"]["run_inputs"]["bundle"] == "fake bundle"
    assert "spec" in strategy.calls["quant"]["inputs"]
    assert "spec" in strategy.calls["microarch"]["inputs"]
    assert "quant" in strategy.calls["microarch"]["inputs"]
    assert "microarch" in strategy.calls["rtl"]["inputs"]
    assert "rtl" in strategy.calls["static_checks"]["inputs"]
    assert "static_checks" in strategy.calls["synth"]["inputs"]
    assert "verification" in strategy.calls["synth"]["inputs"]
