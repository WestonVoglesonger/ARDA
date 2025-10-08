import asyncio
from collections import defaultdict, deque

import pytest

from alg2sv.simplified_pipeline import SimplifiedPipeline
from alg2sv.domain import (
    SpecContract,
    QuantConfig,
    MicroArchConfig,
    RTLConfig,
    SynthResults,
    LintResults,
    VerifyResults,
    EvaluateResults,
    FeedbackDecision,
)


def _sample_bundle() -> str:
    return """``` path=algorithms/sample.py\nprint('hello world')\n```"""


def _default_stage_outputs():
    return {
        'spec': deque([
            SpecContract(
                name="DemoAlgorithm",
                description="Example description",
                clock_mhz_target=200.0,
                throughput_samples_per_cycle=1,
                input_format={"width": 12, "fractional_bits": 11},
                output_format={"width": 16, "fractional_bits": 14},
                resource_budget={"lut": 5000, "ff": 6000, "dsp": 20, "bram": 4},
                verification_config={"num_samples": 16},
            )
        ]),
        'quant': deque([
            QuantConfig(
                fixed_point_config={
                    "input_width": 12,
                    "input_frac": 11,
                    "output_width": 16,
                    "output_frac": 14,
                },
                error_metrics={"max_abs_error": 1.0e-6},
                quantized_coefficients=[0.1, 0.2, 0.3],
                fxp_model_path="models/fxp_model.py",
            )
        ]),
        'microarch': deque([
            MicroArchConfig(
                pipeline_depth=4,
                unroll_factor=1,
                memory_config={"buffer_depth": 8},
                dsp_usage_estimate=8,
                estimated_latency_cycles=6,
                handshake_protocol="ready_valid",
            )
        ]),
        'rtl': deque([
            RTLConfig(
                rtl_files=["rtl/core.sv", "rtl/top.sv"],
                params_file="rtl/params.svh",
                top_module="demo_top",
                lint_passed=True,
                estimated_resources={"lut": 1200, "ff": 2400, "dsp": 8},
            )
        ]),
        'synth': deque([
            SynthResults(
                fmax_mhz=150.0,
                timing_met=False,
                lut_usage=1500,
                ff_usage=2500,
                dsp_usage=10,
                bram_usage=2,
                total_power_mw=55.0,
                slack_ns=-1.2,
                reports_path="synth/run1",
            ),
            SynthResults(
                fmax_mhz=210.0,
                timing_met=True,
                lut_usage=1600,
                ff_usage=2600,
                dsp_usage=10,
                bram_usage=2,
                total_power_mw=52.0,
                slack_ns=0.7,
                reports_path="synth/run2",
            ),
        ]),
        'static_checks': deque([
            LintResults(
                syntax_errors=0,
                style_warnings=1,
                lint_violations=0,
                critical_issues=0,
                issues_list=[],
                overall_score=95.0,
                lint_clean=True,
            )
        ]),
        'verification': deque([
            VerifyResults(
                tests_total=20,
                tests_passed=20,
                all_passed=True,
                mismatches=[],
                max_abs_error=1.0e-6,
                rms_error=5.0e-7,
                functional_coverage=0.95,
            )
        ]),
        'evaluate': deque([
            EvaluateResults(
                overall_score=92.0,
                performance_score=90.0,
                resource_score=88.0,
                quality_score=93.0,
                correctness_score=95.0,
                recommendations=[],
                bottlenecks=[],
                optimization_opportunities=[],
            )
        ]),
    }


def test_pipeline_retries_synth(monkeypatch):
    pipeline = SimplifiedPipeline()
    stage_outputs = _default_stage_outputs()
    feedback_decisions = deque([
        {"action": "continue"},
        {"action": "continue"},
        {"action": "continue"},
        {"action": "continue"},
        {"action": "continue"},
        {"action": "continue"},
        {"action": "retry_synth", "target_stage": "synth", "guidance": "Improve timing."},
        {"action": "continue"},
        {"action": "continue"},
        {"action": "continue"},
        {"action": "continue"},
    ])
    stage_calls = defaultdict(int)

    async def fake_run(self, agent_name: str, context: str):
        stage_calls[agent_name] += 1
        if agent_name == 'feedback':
            decision_data = feedback_decisions.popleft() if feedback_decisions else {"action": "continue"}
            try:
                return FeedbackDecision(**decision_data)
            except Exception:
                return FeedbackDecision(action='continue')

        outputs = stage_outputs[agent_name]
        if len(outputs) > 1:
            result = outputs.popleft()
        else:
            result = outputs[0]
        return result

    monkeypatch.setattr(SimplifiedPipeline, "_run_agent_with_context", fake_run, raising=False)

    result = asyncio.run(pipeline.run(_sample_bundle()))

    assert result["success"] is True
    assert pipeline.stage_attempts['synth'] == 2
    assert stage_calls['synth'] == 2
    assert pipeline.results['synth'].timing_met is True
    assert pipeline.results['static_checks'].overall_score == 95.0


def test_pipeline_feedback_abort(monkeypatch):
    pipeline = SimplifiedPipeline()
    stage_outputs = _default_stage_outputs()
    # Modify verification to fail so that feedback aborts after verification stage
    stage_outputs['verification'] = deque([
        VerifyResults(
            tests_total=10,
            tests_passed=5,
            all_passed=False,
            mismatches=[{"index": 3, "expected": 1.0, "actual": 0.0}],
            max_abs_error=0.1,
            rms_error=0.05,
            functional_coverage=0.6,
        )
    ])

    feedback_decisions = deque([
        {"action": "continue"},
        {"action": "continue"},
        {"action": "continue"},
        {"action": "continue"},
        {"action": "continue"},
        {"action": "abort", "guidance": "Verification failed."},
    ])
    stage_calls = defaultdict(int)

    async def fake_run(self, agent_name: str, context: str):
        stage_calls[agent_name] += 1
        if agent_name == 'feedback':
            decision_data = feedback_decisions.popleft() if feedback_decisions else {"action": "continue"}
            try:
                return FeedbackDecision(**decision_data)
            except Exception:
                return FeedbackDecision(action='continue')

        outputs = stage_outputs[agent_name]
        if len(outputs) > 1:
            result = outputs.popleft()
        else:
            result = outputs[0]
        return result

    monkeypatch.setattr(SimplifiedPipeline, "_run_agent_with_context", fake_run, raising=False)

    result = asyncio.run(pipeline.run(_sample_bundle()))

    assert result["success"] is False
    assert result["error"] == "Pipeline aborted by feedback agent"
    assert 'synth' not in pipeline.results
    assert pipeline.stage_attempts['verification'] == 1
    assert stage_calls['verification'] == 1
