"""
OpenAI Stage Testing Suite.

Comprehensive tests for all 8 ARDA pipeline stages using real OpenAI API calls.
Tests are independently executable with fixture support, detailed logging,
and interactive retry logic on failures.
"""

import asyncio
import time
from typing import Any, Dict

import pytest

from tests.utils.schema_validator import SchemaValidator, SchemaValidationError
from tests.utils.stage_logger import StageLogger
from tests.utils.retry_handler import RetryHandler
from tests.fixtures.fixture_manager import FixtureManager
from tests.openai_test_config import config
from ardagen.agents.openai_runner import OpenAIAgentRunner


class TestSpecStage:
    """Test spec stage with real OpenAI calls."""
    
    @pytest.mark.parametrize("algorithm", ["conv2d"])
    def test_spec_stage_schema(
        self,
        algorithm: str,
        openai_runner: OpenAIAgentRunner,
        stage_logger: StageLogger,
        retry_handler: RetryHandler,
        conv2d_bundle: str
    ):
        """Test spec stage with real OpenAI call."""
        # Skip live mode tests unless explicitly enabled
        if not config.is_live_mode():
            pytest.skip("Skipping OpenAI test - not in live mode")
        
        test_name = f"test_spec_stage_{algorithm}"
        
        def run_spec_stage():
            # Prepare context for spec stage
            context = {
                "stage": "spec",
                "run_inputs": {"bundle": conv2d_bundle},
                "workspace_token": "test_workspace",
                "workspace_files": []
            }
            
            # Run spec stage
            start_time = time.time()
            result = asyncio.run(openai_runner.run_stage("spec", context))
            duration_ms = (time.time() - start_time) * 1000
            
            return result
        
        # Execute with retry handling
        try:
            result, retry_count, errors = retry_handler.execute_with_retry(
                run_spec_stage, test_name, algorithm, "spec"
            )
            duration_ms = 0  # Duration tracking not implemented in retry handler yet
            
            # Validate schema
            validator = SchemaValidator()
            validated_result = validator.validate_stage_output("spec", result)
            
            # Log test execution
            stage_logger.log_test_execution(
                test_name=test_name,
                algorithm=algorithm,
                stage="spec",
                inputs={"bundle": conv2d_bundle[:100] + "..."},  # Truncate for logging
                outputs=result,
                duration_ms=duration_ms,
                status="passed"
            )
            
            # Basic assertions
            assert isinstance(result, dict)
            assert "name" in result
            assert "clock_mhz_target" in result
            assert "throughput_samples_per_cycle" in result
            assert result["clock_mhz_target"] > 0
            assert result["throughput_samples_per_cycle"] > 0
            
        except Exception as e:
            # Log failed test
            stage_logger.log_test_execution(
                test_name=test_name,
                algorithm=algorithm,
                stage="spec",
                inputs={"bundle": conv2d_bundle[:100] + "..."},
                outputs=None,
                errors=[str(e)],
                status="failed"
            )
            raise


class TestQuantStage:
    """Test quant stage with spec dependency."""
    
    @pytest.mark.parametrize("algorithm", ["conv2d"])
    def test_quant_stage_schema(
        self,
        algorithm: str,
        openai_runner: OpenAIAgentRunner,
        stage_logger: StageLogger,
        retry_handler: RetryHandler,
        conv2d_fixtures: Dict[str, Any],
        conv2d_bundle: str
    ):
        """Test quant stage with spec dependency."""
        # Skip live mode tests unless explicitly enabled
        if not config.is_live_mode():
            pytest.skip("Skipping OpenAI test - not in live mode")
        
        test_name = f"test_quant_stage_{algorithm}"
        
        def run_quant_stage():
            # Prepare context with spec dependency
            context = {
                "stage": "quant",
                "stage_inputs": {
                    "spec": conv2d_fixtures["spec"]
                },
                "run_inputs": {"bundle": conv2d_bundle},
                "workspace_token": "test_workspace",
                "workspace_files": []
            }
            
            # Run quant stage
            start_time = time.time()
            result = asyncio.run(openai_runner.run_stage("quant", context))
            duration_ms = (time.time() - start_time) * 1000
            
            return result
        
        # Execute with retry handling
        try:
            result, retry_count, errors = retry_handler.execute_with_retry(
                run_quant_stage, test_name, algorithm, "quant"
            )
            duration_ms = 0  # Duration tracking not implemented in retry handler yet
            
            # Validate schema
            validator = SchemaValidator()
            validated_result = validator.validate_stage_output("quant", result)
            
            # Log test execution
            stage_logger.log_test_execution(
                test_name=test_name,
                algorithm=algorithm,
                stage="quant",
                inputs={"spec": conv2d_fixtures["spec"], "bundle": conv2d_bundle[:100] + "..."},
                outputs=result,
                duration_ms=duration_ms,
                status="passed"
            )
            
            # Basic assertions
            assert isinstance(result, dict)
            assert "fixed_point_config" in result
            assert "error_metrics" in result
            assert "quantized_coefficients" in result
            
        except Exception as e:
            # Log failed test
            stage_logger.log_test_execution(
                test_name=test_name,
                algorithm=algorithm,
                stage="quant",
                inputs={"spec": conv2d_fixtures["spec"], "bundle": conv2d_bundle[:100] + "..."},
                outputs=None,
                errors=[str(e)],
                status="failed"
            )
            raise


class TestMicroArchStage:
    """Test microarch stage with spec dependency."""
    
    @pytest.mark.parametrize("algorithm", ["conv2d"])
    def test_microarch_stage_schema(
        self,
        algorithm: str,
        openai_runner: OpenAIAgentRunner,
        stage_logger: StageLogger,
        retry_handler: RetryHandler,
        conv2d_fixtures: Dict[str, Any],
        conv2d_bundle: str
    ):
        """Test microarch stage with spec dependency."""
        # Skip live mode tests unless explicitly enabled
        if not config.is_live_mode():
            pytest.skip("Skipping OpenAI test - not in live mode")
        
        test_name = f"test_microarch_stage_{algorithm}"
        
        def run_microarch_stage():
            # Prepare context with spec dependency
            context = {
                "stage": "microarch",
                "stage_inputs": {
                    "spec": conv2d_fixtures["spec"]
                },
                "run_inputs": {"bundle": conv2d_bundle},
                "workspace_token": "test_workspace",
                "workspace_files": []
            }
            
            # Run microarch stage
            start_time = time.time()
            result = asyncio.run(openai_runner.run_stage("microarch", context))
            duration_ms = (time.time() - start_time) * 1000
            
            return result
        
        # Execute with retry handling
        try:
            result, retry_count, errors = retry_handler.execute_with_retry(
                run_microarch_stage, test_name, algorithm, "microarch"
            )
            duration_ms = 0  # Duration tracking not implemented in retry handler yet
            
            # Validate schema
            validator = SchemaValidator()
            validated_result = validator.validate_stage_output("microarch", result)
            
            # Log test execution
            stage_logger.log_test_execution(
                test_name=test_name,
                algorithm=algorithm,
                stage="microarch",
                inputs={"spec": conv2d_fixtures["spec"], "bundle": conv2d_bundle[:100] + "..."},
                outputs=result,
                duration_ms=duration_ms,
                status="passed"
            )
            
            # Basic assertions
            assert isinstance(result, dict)
            assert "pipeline_depth" in result
            assert "unroll_factor" in result
            assert "memory_config" in result
            assert result["pipeline_depth"] > 0
            assert result["unroll_factor"] > 0
            
        except Exception as e:
            # Log failed test
            stage_logger.log_test_execution(
                test_name=test_name,
                algorithm=algorithm,
                stage="microarch",
                inputs={"spec": conv2d_fixtures["spec"], "bundle": conv2d_bundle[:100] + "..."},
                outputs=None,
                errors=[str(e)],
                status="failed"
            )
            raise


class TestArchitectureStage:
    """Test architecture stage with spec, quant, and microarch dependencies."""
    
    @pytest.mark.parametrize("algorithm", ["conv2d"])
    def test_architecture_stage_schema(
        self,
        algorithm: str,
        openai_runner: OpenAIAgentRunner,
        stage_logger: StageLogger,
        retry_handler: RetryHandler,
        conv2d_fixtures: Dict[str, Any],
        conv2d_bundle: str
    ):
        """Test architecture stage with dependencies."""
        # Skip live mode tests unless explicitly enabled
        if not config.is_live_mode():
            pytest.skip("Skipping OpenAI test - not in live mode")
        
        test_name = f"test_architecture_stage_{algorithm}"
        
        def run_architecture_stage():
            # Prepare context with dependencies
            context = {
                "stage": "architecture",
                "stage_inputs": {
                    "spec": conv2d_fixtures["spec"],
                    "quant": conv2d_fixtures["quant"],
                    "microarch": conv2d_fixtures["microarch"]
                },
                "run_inputs": {"bundle": conv2d_bundle},
                "workspace_token": "test_workspace",
                "workspace_files": []
            }
            
            # Run architecture stage
            start_time = time.time()
            result = asyncio.run(openai_runner.run_stage("architecture", context))
            duration_ms = (time.time() - start_time) * 1000
            
            return result
        
        # Execute with retry handling
        try:
            result, retry_count, errors = retry_handler.execute_with_retry(
                run_architecture_stage, test_name, algorithm, "architecture"
            )
            duration_ms = 0  # Duration tracking not implemented in retry handler yet
            
            # Validate schema
            validator = SchemaValidator()
            validated_result = validator.validate_stage_output("architecture", result)
            
            # Log test execution
            stage_logger.log_test_execution(
                test_name=test_name,
                algorithm=algorithm,
                stage="architecture",
                inputs={
                    "spec": conv2d_fixtures["spec"],
                    "quant": conv2d_fixtures["quant"],
                    "microarch": conv2d_fixtures["microarch"],
                    "bundle": conv2d_bundle[:100] + "..."
                },
                outputs=result,
                duration_ms=duration_ms,
                status="passed"
            )
            
            # Basic assertions
            assert isinstance(result, dict)
            assert "architecture_type" in result
            assert "modules" in result
            assert "top_module" in result
            assert isinstance(result["modules"], list)
            assert len(result["modules"]) > 0
            
        except Exception as e:
            # Log failed test
            stage_logger.log_test_execution(
                test_name=test_name,
                algorithm=algorithm,
                stage="architecture",
                inputs={
                    "spec": conv2d_fixtures["spec"],
                    "quant": conv2d_fixtures["quant"],
                    "microarch": conv2d_fixtures["microarch"],
                    "bundle": conv2d_bundle[:100] + "..."
                },
                outputs=None,
                errors=[str(e)],
                status="failed"
            )
            raise


class TestRTLStage:
    """Test RTL stage with architecture dependency."""
    
    @pytest.mark.parametrize("algorithm", ["conv2d"])
    def test_rtl_stage_schema(
        self,
        algorithm: str,
        openai_runner: OpenAIAgentRunner,
        stage_logger: StageLogger,
        retry_handler: RetryHandler,
        conv2d_fixtures: Dict[str, Any],
        conv2d_bundle: str
    ):
        """Test RTL stage with dependencies."""
        # Skip live mode tests unless explicitly enabled
        if not config.is_live_mode():
            pytest.skip("Skipping OpenAI test - not in live mode")
        
        test_name = f"test_rtl_stage_{algorithm}"
        
        def run_rtl_stage():
            # Prepare context with dependencies
            context = {
                "stage": "rtl",
                "stage_inputs": {
                    "spec": conv2d_fixtures["spec"],
                    "quant": conv2d_fixtures["quant"],
                    "microarch": conv2d_fixtures["microarch"],
                    "architecture": conv2d_fixtures["architecture"]
                },
                "run_inputs": {"bundle": conv2d_bundle},
                "workspace_token": "test_workspace",
                "workspace_files": []
            }
            
            # Run RTL stage
            start_time = time.time()
            result = asyncio.run(openai_runner.run_stage("rtl", context))
            duration_ms = (time.time() - start_time) * 1000
            
            return result
        
        # Execute with retry handling
        try:
            result, retry_count, errors = retry_handler.execute_with_retry(
                run_rtl_stage, test_name, algorithm, "rtl"
            )
            duration_ms = 0  # Duration tracking not implemented in retry handler yet
            
            # Validate schema
            validator = SchemaValidator()
            validated_result = validator.validate_stage_output("rtl", result)
            
            # Log test execution
            stage_logger.log_test_execution(
                test_name=test_name,
                algorithm=algorithm,
                stage="rtl",
                inputs={
                    "spec": conv2d_fixtures["spec"],
                    "quant": conv2d_fixtures["quant"],
                    "microarch": conv2d_fixtures["microarch"],
                    "architecture": conv2d_fixtures["architecture"],
                    "bundle": conv2d_bundle[:100] + "..."
                },
                outputs=result,
                duration_ms=duration_ms,
                status="passed"
            )
            
            # Basic assertions
            assert isinstance(result, dict)
            assert "generated_files" in result
            assert "file_paths" in result
            assert "top_module" in result
            assert isinstance(result["generated_files"], dict)
            assert len(result["generated_files"]) > 0
            
        except Exception as e:
            # Log failed test
            stage_logger.log_test_execution(
                test_name=test_name,
                algorithm=algorithm,
                stage="rtl",
                inputs={
                    "spec": conv2d_fixtures["spec"],
                    "quant": conv2d_fixtures["quant"],
                    "microarch": conv2d_fixtures["microarch"],
                    "architecture": conv2d_fixtures["architecture"],
                    "bundle": conv2d_bundle[:100] + "..."
                },
                outputs=None,
                errors=[str(e)],
                status="failed"
            )
            raise


class TestVerificationStage:
    """Test verification stage with RTL dependency."""
    
    @pytest.mark.parametrize("algorithm", ["conv2d"])
    def test_verification_stage_schema(
        self,
        algorithm: str,
        openai_runner: OpenAIAgentRunner,
        stage_logger: StageLogger,
        retry_handler: RetryHandler,
        conv2d_fixtures: Dict[str, Any],
        conv2d_bundle: str
    ):
        """Test verification stage with dependencies."""
        # Skip live mode tests unless explicitly enabled
        if not config.is_live_mode():
            pytest.skip("Skipping OpenAI test - not in live mode")
        
        test_name = f"test_verification_stage_{algorithm}"
        
        def run_verification_stage():
            # Prepare context with dependencies
            context = {
                "stage": "simulation",
                "stage_inputs": {
                    "spec": conv2d_fixtures["spec"],
                    "quant": conv2d_fixtures["quant"],
                    "microarch": conv2d_fixtures["microarch"],
                    "architecture": conv2d_fixtures["architecture"],
                    "rtl": conv2d_fixtures["rtl"]
                },
                "run_inputs": {"bundle": conv2d_bundle},
                "workspace_token": "test_workspace",
                "workspace_files": []
            }

            # Run verification stage with OpenAI agent
            start_time = time.time()
            result = asyncio.run(openai_runner.run_stage("simulation", context))
            duration_ms = (time.time() - start_time) * 1000

            return result
        
        # Execute with retry handling
        try:
            result, retry_count, errors = retry_handler.execute_with_retry(
                run_verification_stage, test_name, algorithm, "simulation"
            )
            duration_ms = 0  # Duration tracking not implemented in retry handler yet
            
            # Validate schema
            validator = SchemaValidator()
            validated_result = validator.validate_stage_output("verification", result)
            
            # Log test execution
            stage_logger.log_test_execution(
                test_name=test_name,
                algorithm=algorithm,
                stage="verification",
                inputs={
                    "spec": conv2d_fixtures["spec"],
                    "quant": conv2d_fixtures["quant"],
                    "microarch": conv2d_fixtures["microarch"],
                    "architecture": conv2d_fixtures["architecture"],
                    "rtl": conv2d_fixtures["rtl"],
                    "bundle": conv2d_bundle[:100] + "..."
                },
                outputs=result,
                duration_ms=duration_ms,
                status="passed"
            )
            
            # Basic assertions - handle both dict and Pydantic objects
            if hasattr(result, 'model_dump'):
                result_dict = result.model_dump()
            else:
                result_dict = result

            assert isinstance(result_dict, dict)
            assert "tests_total" in result_dict
            assert "tests_passed" in result_dict
            assert "all_passed" in result_dict
            assert result_dict["tests_total"] > 0
            assert result_dict["tests_passed"] >= 0
            
        except Exception as e:
            # Log failed test
            stage_logger.log_test_execution(
                test_name=test_name,
                algorithm=algorithm,
                stage="verification",
                inputs={
                    "spec": conv2d_fixtures["spec"],
                    "quant": conv2d_fixtures["quant"],
                    "microarch": conv2d_fixtures["microarch"],
                    "architecture": conv2d_fixtures["architecture"],
                    "rtl": conv2d_fixtures["rtl"],
                    "bundle": conv2d_bundle[:100] + "..."
                },
                outputs=None,
                errors=[str(e)],
                status="failed"
            )
            raise


class TestSynthStage:
    """Test synth stage with verification dependency."""
    
    @pytest.mark.parametrize("algorithm", ["conv2d"])
    def test_synth_stage_schema(
        self,
        algorithm: str,
        openai_runner: OpenAIAgentRunner,
        stage_logger: StageLogger,
        retry_handler: RetryHandler,
        conv2d_fixtures: Dict[str, Any],
        conv2d_bundle: str
    ):
        """Test synth stage with dependencies."""
        # Skip live mode tests unless explicitly enabled
        if not config.is_live_mode():
            pytest.skip("Skipping OpenAI test - not in live mode")
        
        test_name = f"test_synth_stage_{algorithm}"
        
        def run_synth_stage():
            # Prepare context with dependencies
            context = {
                "stage": "synth",
                "stage_inputs": {
                    "spec": conv2d_fixtures["spec"],
                    "quant": conv2d_fixtures["quant"],
                    "microarch": conv2d_fixtures["microarch"],
                    "architecture": conv2d_fixtures["architecture"],
                    "rtl": conv2d_fixtures["rtl"],
                    "verification": conv2d_fixtures["verification"]
                },
                "run_inputs": {"bundle": conv2d_bundle},
                "workspace_token": "test_workspace",
                "workspace_files": []
            }
            
            # Run synth stage
            start_time = time.time()
            result = asyncio.run(openai_runner.run_stage("synth", context))
            duration_ms = (time.time() - start_time) * 1000
            
            return result
        
        # Execute with retry handling
        try:
            result, retry_count, errors = retry_handler.execute_with_retry(
                run_synth_stage, test_name, algorithm, "synth"
            )
            duration_ms = 0  # Duration tracking not implemented in retry handler yet
            
            # Validate schema
            validator = SchemaValidator()
            validated_result = validator.validate_stage_output("synth", result)
            
            # Log test execution
            stage_logger.log_test_execution(
                test_name=test_name,
                algorithm=algorithm,
                stage="synth",
                inputs={
                    "spec": conv2d_fixtures["spec"],
                    "quant": conv2d_fixtures["quant"],
                    "microarch": conv2d_fixtures["microarch"],
                    "architecture": conv2d_fixtures["architecture"],
                    "rtl": conv2d_fixtures["rtl"],
                    "verification": conv2d_fixtures["verification"],
                    "bundle": conv2d_bundle[:100] + "..."
                },
                outputs=result,
                duration_ms=duration_ms,
                status="passed"
            )
            
            # Basic assertions
            assert isinstance(result, dict)
            assert "fmax_mhz" in result
            assert "timing_met" in result
            assert "lut_usage" in result
            assert result["fmax_mhz"] > 0
            
        except Exception as e:
            # Log failed test
            stage_logger.log_test_execution(
                test_name=test_name,
                algorithm=algorithm,
                stage="synth",
                inputs={
                    "spec": conv2d_fixtures["spec"],
                    "quant": conv2d_fixtures["quant"],
                    "microarch": conv2d_fixtures["microarch"],
                    "architecture": conv2d_fixtures["architecture"],
                    "rtl": conv2d_fixtures["rtl"],
                    "verification": conv2d_fixtures["verification"],
                    "bundle": conv2d_bundle[:100] + "..."
                },
                outputs=None,
                errors=[str(e)],
                status="failed"
            )
            raise


class TestEvaluateStage:
    """Test evaluate stage with synth dependency."""
    
    @pytest.mark.parametrize("algorithm", ["conv2d"])
    def test_evaluate_stage_schema(
        self,
        algorithm: str,
        openai_runner: OpenAIAgentRunner,
        stage_logger: StageLogger,
        retry_handler: RetryHandler,
        conv2d_fixtures: Dict[str, Any],
        conv2d_bundle: str
    ):
        """Test evaluate stage with dependencies."""
        # Skip live mode tests unless explicitly enabled
        if not config.is_live_mode():
            pytest.skip("Skipping OpenAI test - not in live mode")
        
        test_name = f"test_evaluate_stage_{algorithm}"
        
        def run_evaluate_stage():
            # Prepare context with dependencies
            context = {
                "stage": "evaluate",
                "stage_inputs": {
                    "spec": conv2d_fixtures["spec"],
                    "quant": conv2d_fixtures["quant"],
                    "microarch": conv2d_fixtures["microarch"],
                    "architecture": conv2d_fixtures["architecture"],
                    "rtl": conv2d_fixtures["rtl"],
                    "verification": conv2d_fixtures["verification"],
                    "synth": conv2d_fixtures["synth"]
                },
                "run_inputs": {"bundle": conv2d_bundle},
                "workspace_token": "test_workspace",
                "workspace_files": []
            }
            
            # Run evaluate stage
            start_time = time.time()
            result = asyncio.run(openai_runner.run_stage("evaluate", context))
            duration_ms = (time.time() - start_time) * 1000
            
            return result
        
        # Execute with retry handling
        try:
            result, retry_count, errors = retry_handler.execute_with_retry(
                run_evaluate_stage, test_name, algorithm, "evaluate"
            )
            duration_ms = 0  # Duration tracking not implemented in retry handler yet
            
            # Validate schema
            validator = SchemaValidator()
            validated_result = validator.validate_stage_output("evaluate", result)
            
            # Log test execution
            stage_logger.log_test_execution(
                test_name=test_name,
                algorithm=algorithm,
                stage="evaluate",
                inputs={
                    "spec": conv2d_fixtures["spec"],
                    "quant": conv2d_fixtures["quant"],
                    "microarch": conv2d_fixtures["microarch"],
                    "architecture": conv2d_fixtures["architecture"],
                    "rtl": conv2d_fixtures["rtl"],
                    "verification": conv2d_fixtures["verification"],
                    "synth": conv2d_fixtures["synth"],
                    "bundle": conv2d_bundle[:100] + "..."
                },
                outputs=result,
                duration_ms=duration_ms,
                status="passed"
            )
            
            # Basic assertions
            # Handle both dict and Pydantic objects
            if hasattr(result, 'model_dump'):
                result_dict = result.model_dump()
            else:
                result_dict = result
            
            assert isinstance(result_dict, dict)
            assert "overall_score" in result_dict
            assert "performance_score" in result_dict
            assert "resource_score" in result_dict
            assert "quality_score" in result_dict
            assert "correctness_score" in result_dict
            assert 0 <= result_dict["overall_score"] <= 100
            
        except Exception as e:
            # Log failed test
            stage_logger.log_test_execution(
                test_name=test_name,
                algorithm=algorithm,
                stage="evaluate",
                inputs={
                    "spec": conv2d_fixtures["spec"],
                    "quant": conv2d_fixtures["quant"],
                    "microarch": conv2d_fixtures["microarch"],
                    "architecture": conv2d_fixtures["architecture"],
                    "rtl": conv2d_fixtures["rtl"],
                    "verification": conv2d_fixtures["verification"],
                    "synth": conv2d_fixtures["synth"],
                    "bundle": conv2d_bundle[:100] + "..."
                },
                outputs=None,
                errors=[str(e)],
                status="failed"
            )
            raise



class TestTestGenerationStage:
    """Test test generation stage that creates test vectors for verification."""

    @pytest.mark.parametrize("algorithm", ["conv2d"])
    @pytest.mark.openai_test_generation
    def test_test_generation_stage_schema(
        self,
        algorithm: str,
        openai_runner: OpenAIAgentRunner,
        stage_logger: StageLogger,
        retry_handler: RetryHandler,
        conv2d_fixtures: Dict[str, Any],
        conv2d_bundle: str
    ):
        """Test test generation stage with RTL dependency."""
        if not config.is_live_mode():
            pytest.skip("Skipping OpenAI test - not in live mode")

        test_name = f"test_test_generation_stage_{algorithm}"

        def run_test_generation_stage():
            context = {
                "stage": "test_generation",
                "attempt": 1,
                "stage_inputs": {
                    "rtl": conv2d_fixtures["rtl"],
                    # Include lint results as they would be available in the pipeline
                    "lint_results": {
                        "syntax_errors": 0,
                        "style_warnings": 2,
                        "lint_violations": 1,
                        "critical_issues": 0,
                        "issues_list": [],
                        "overall_score": 95.0,
                        "lint_clean": True,
                        "confidence": 85.0
                    }
                },
                # Include prior results as the pipeline does - critical for test generation
                "prior_results": {
                    "spec": conv2d_fixtures["spec"],
                    "quant": conv2d_fixtures["quant"],
                    "microarch": conv2d_fixtures["microarch"],
                    "architecture": conv2d_fixtures["architecture"],
                    "rtl": conv2d_fixtures["rtl"]
                },
                "run_inputs": {"bundle": conv2d_bundle},
                "workspace_token": "test_workspace",
                "workspace_files": [],
                "observability": None  # Would be set in real pipeline
            }

            start_time = time.time()
            result = asyncio.run(openai_runner.run_stage("test_generation", context))
            duration_ms = (time.time() - start_time) * 1000

            return result

        result = None
        error_message = ""
        success = False
        duration_ms = 0  # Initialize duration_ms

        try:
            result, retry_count, errors = retry_handler.execute_with_retry(
                run_test_generation_stage, test_name, algorithm, "test_generation"
            )
            duration_ms = 0 # Duration tracking not implemented in retry handler yet

            # Check if there were any errors from the retry handler
            if errors:
                raise RuntimeError(f"Retry handler reported errors: {errors}")

            # Basic validation for test generation output
            assert isinstance(result, dict)
            assert "test_count" in result or "test_vectors" in result
            assert "golden_outputs" in result or "expected_outputs" in result

            # If test_vectors exist, ensure they're properly structured
            if "test_vectors" in result:
                test_vectors = result["test_vectors"]
                assert isinstance(test_vectors, list)
                assert len(test_vectors) <= 5, f"Too many test vectors: {len(test_vectors)}"
                assert len(test_vectors) > 0, "Should have at least 1 test vector"
                if len(test_vectors) > 0:
                    # Check that first test vector has expected structure
                    first_vector = test_vectors[0]
                    assert isinstance(first_vector, dict)
                    assert "input" in first_vector
                    assert isinstance(first_vector["input"], list)
                    assert len(first_vector["input"]) > 0, "Input array should not be empty"

                    # Check golden outputs have matching structure
                    if "golden_outputs" in result:
                        golden_outputs = result["golden_outputs"]
                        assert isinstance(golden_outputs, list)
                        assert len(golden_outputs) == len(test_vectors), "Golden outputs should match test vectors count"
                        if len(golden_outputs) > 0:
                            first_output = golden_outputs[0]
                            assert isinstance(first_output, dict)
                            assert "output" in first_output
                            assert isinstance(first_output["output"], list)
                            assert len(first_output["output"]) > 0, "Output array should not be empty"

            success = True
            print(f"Stage 'test_generation' completed successfully.")

        except Exception as e:
            error_message = str(e)
            print(f"Test failed for stage 'test_generation': {error_message}")
            success = False
            # Re-raise the exception to make the test actually fail
            raise

        finally:
            stage_logger.log_test_execution(
                test_name=test_name,
                algorithm=algorithm,
                stage="test_generation",
                inputs={
                    "rtl": conv2d_fixtures["rtl"],
                    "lint_results": {
                        "syntax_errors": 0,
                        "style_warnings": 2,
                        "lint_violations": 1,
                        "critical_issues": 0,
                        "issues_list": [],
                        "overall_score": 95.0,
                        "lint_clean": True,
                        "confidence": 85.0
                    },
                    "bundle": conv2d_bundle[:100] + "..."
                },
                outputs=result,
                duration_ms=duration_ms,
                status="passed" if success else "failed",
                errors=[error_message] if not success else [],
            )
