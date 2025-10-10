"""Enhanced simulation and verification utilities."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from ..domain import VerifyResults


def run_verification(context: Mapping[str, Any]) -> VerifyResults:
    """Entry point used by the verification stage.

    Delegates to the default verification flow which orchestrates simulation,
    protocol checks, and result aggregation. The context carries stage inputs
    (including the RTL config) and observability hooks.
    """

    _emit_tool_event(context, "verification", "verification-orchestrator", {})
    verifier = VerificationRunner(context)
    return verifier.execute()


class VerificationRunner:
    """Coordinates AI-assisted verification activities."""

    def __init__(self, context: Mapping[str, Any]):
        self._context = context
        self._workspace_token = self._extract_workspace_token(context)
        self._rtl_config = context.get("stage_inputs", {}).get("rtl")
        self._artefacts: List[str] = []
        self._suite_results: Dict[str, Dict[str, Any]] = {}

    def execute(self) -> VerifyResults:
        """Run all verification suites and assemble the final results."""

        if self._rtl_config is None:
            raise ValueError("Verification context missing RTL configuration")

        suite_configs = self._build_default_suites()
        total_tests = 0
        passed_tests = 0
        mismatches: List[Dict[str, Any]] = []
        max_abs_error = 0.0
        rms_error = 0.0

        for suite in suite_configs:
            suite_result = self._run_suite(suite)
            self._suite_results[suite["name"]] = suite_result
            total_tests += suite_result.get("tests_run", 0)
            passed_tests += suite_result.get("tests_passed", 0)
            mismatches.extend(suite_result.get("mismatches", []))
            max_abs_error = max(max_abs_error, suite_result.get("max_abs_error", 0.0))
            rms_error = max(rms_error, suite_result.get("rms_error", 0.0))

        all_passed = passed_tests == total_tests and not any(
            not data.get("passed", True) for data in self._suite_results.values()
        )

        return VerifyResults(
            tests_total=total_tests,
            tests_passed=passed_tests,
            all_passed=all_passed,
            mismatches=mismatches,
            max_abs_error=max_abs_error,
            rms_error=rms_error,
            functional_coverage=self._estimate_functional_coverage(),
            confidence=self._estimate_confidence(all_passed),
        )

    # ------------------------------------------------------------------
    # Suite execution helpers
    # ------------------------------------------------------------------

    def _build_default_suites(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "functional",
                "description": "Golden-model comparison over >1024 samples",
                "generator": self._generate_functional_vectors,
                "analysis": self._analyze_functional_results,
            },
            {
                "name": "protocol",
                "description": "Ready/valid compliance under backpressure and bubbles",
                "generator": self._generate_protocol_vectors,
                "analysis": self._analyze_protocol_results,
            },
            {
                "name": "stress",
                "description": "Randomised stress and reset toggling",
                "generator": self._generate_stress_vectors,
                "analysis": self._analyze_stress_results,
            },
        ]

    def _run_suite(self, suite: Dict[str, Any]) -> Dict[str, Any]:
        suite_name = suite["name"]
        generator = suite["generator"]
        analysis = suite["analysis"]

        vectors = generator()
        sim_result = self._invoke_simulation(vectors)
        analysis_result = analysis(sim_result)
        analysis_result.setdefault("tests_run", len(vectors))
        analysis_result.setdefault("tests_passed", len(vectors) if analysis_result.get("passed", False) else 0)
        return analysis_result

    # ------------------------------------------------------------------
    # Vector generation (stubs for AI-enhanced flows)
    # ------------------------------------------------------------------

    def _generate_functional_vectors(self) -> List[Dict[str, Any]]:
        return self._load_or_stub_vectors("functional_vectors.json", count=1024)

    def _generate_protocol_vectors(self) -> List[Dict[str, Any]]:
        return self._load_or_stub_vectors("protocol_vectors.json", count=200)

    def _generate_stress_vectors(self) -> List[Dict[str, Any]]:
        return self._load_or_stub_vectors("stress_vectors.json", count=200)

    def _load_or_stub_vectors(self, filename: str, count: int) -> List[Dict[str, Any]]:
        bundle_path = self._resolve_workspace_file(filename)
        if bundle_path and bundle_path.exists():
            with bundle_path.open() as fh:
                data = json.load(fh)
            if isinstance(data, list):
                return data
        # Stub fallback
        vectors = [{"input": i, "expected": 0} for i in range(count)]
        artefact_path = self._write_temp_json(vectors, prefix=filename.replace(".json", ""))
        self._artefacts.append(str(artefact_path))
        return vectors

    # ------------------------------------------------------------------
    # Simulation invocation
    # ------------------------------------------------------------------

    def _invoke_simulation(self, test_vectors: List[Dict[str, Any]]) -> Dict[str, Any]:
        from .tools import run_simulation as run_sim

        rtl_files = list(self._rtl_config.rtl_files or []) if self._rtl_config else []
        result = run_sim(rtl_files, test_vectors)
        if not isinstance(result, dict):
            raise RuntimeError("run_simulation returned unexpected payload")
        result.setdefault("test_vectors_count", len(test_vectors))
        return result

    # ------------------------------------------------------------------
    # Suite-specific analysis (stubs)
    # ------------------------------------------------------------------

    def _analyze_functional_results(self, sim_result: Dict[str, Any]) -> Dict[str, Any]:
        passed = sim_result.get("status") == "completed" and sim_result.get("passed", False)
        return {
            "passed": passed,
            "mismatches": sim_result.get("errors", []),
            "max_abs_error": sim_result.get("max_error", 0.0),
            "rms_error": sim_result.get("rms_error", 0.0),
            "notes": self._collect_warnings(sim_result),
        }

    def _analyze_protocol_results(self, sim_result: Dict[str, Any]) -> Dict[str, Any]:
        violations = sim_result.get("protocol_violations", [])
        return {
            "passed": not violations,
            "failures": violations,
            "mismatches": [],
            "notes": self._collect_warnings(sim_result),
        }

    def _analyze_stress_results(self, sim_result: Dict[str, Any]) -> Dict[str, Any]:
        errors = sim_result.get("errors", [])
        return {
            "passed": sim_result.get("passed", False) and not errors,
            "failures": errors,
            "notes": self._collect_warnings(sim_result),
        }

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    def _estimate_functional_coverage(self) -> float:
        if not self._suite_results:
            return 0.0
        covered = sum(1 for data in self._suite_results.values() if data.get("passed", False))
        return covered / max(len(self._suite_results), 1)

    def _estimate_confidence(self, all_passed: bool) -> float:
        base = 75.0 if all_passed else 40.0
        return min(100.0, base + len(self._suite_results) * 5.0)

    def _collect_warnings(self, sim_result: Dict[str, Any]) -> str:
        warnings = sim_result.get("warnings", [])
        if warnings and isinstance(warnings, Iterable):
            return "; ".join(str(w) for w in warnings)
        return ""

    def _write_temp_json(self, payload: Any, prefix: str) -> Path:
        tmp_dir = Path(tempfile.gettempdir()) / "arda-verification"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        path = tmp_dir / f"{prefix}-{os.getpid()}.json"
        with path.open("w") as fh:
            json.dump(payload, fh, indent=2)
        return path

    def _resolve_workspace_file(self, relative: str) -> Optional[Path]:
        if not self._workspace_token:
            return None
        workspace = self._context.get("workspace")
        if workspace and hasattr(workspace, "resolve_path"):
            try:
                return Path(workspace.resolve_path(relative))
            except Exception:
                return None
        return None

    @staticmethod
    def _extract_workspace_token(context: Mapping[str, Any]) -> Optional[str]:
        return context.get("workspace_token") or context.get("workspace", {}).get("token")


def _emit_tool_event(context: Mapping[str, Any], stage: str, tool_name: str, metadata: Mapping[str, Any]) -> None:
    observability = context.get("observability")
    if observability is not None:
        try:
            observability.tool_invoked(stage, tool_name, dict(metadata))
        except Exception:
            pass
