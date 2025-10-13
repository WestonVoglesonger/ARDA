import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import os
import pytest
from typing import Any, Dict, Optional

from tests.openai_test_config import config
from tests.utils.stage_logger import StageLogger
from tests.utils.retry_handler import RetryHandler
from tests.fixtures.fixture_manager import FixtureManager
from ardagen.agents.openai_runner import OpenAIAgentRunner


@pytest.fixture(scope="session")
def test_config():
    """Get OpenAI test configuration."""
    return config


@pytest.fixture(scope="session")
def fixture_manager():
    """Get fixture manager instance."""
    return FixtureManager()


@pytest.fixture
def openai_client():
    """Fresh OpenAI client for each test."""
    if not config.is_live_mode():
        pytest.skip("Skipping OpenAI test - not in live mode")
    
    if not config.openai_api_key:
        pytest.skip("Skipping OpenAI test - no API key provided")
    
    try:
        from openai import OpenAI
        client = OpenAI(api_key=config.openai_api_key)
        return client
    except ImportError:
        pytest.skip("Skipping OpenAI test - OpenAI SDK not installed")


@pytest.fixture
def openai_runner():
    """Fresh OpenAI agent runner for each test."""
    if not config.is_live_mode():
        pytest.skip("Skipping OpenAI test - not in live mode")
    
    if not config.openai_api_key:
        pytest.skip("Skipping OpenAI test - no API key provided")
    
    try:
        return OpenAIAgentRunner()
    except Exception as e:
        pytest.skip(f"Skipping OpenAI test - runner creation failed: {e}")


@pytest.fixture
def stage_logger():
    """Logger instance for capturing test data."""
    return StageLogger(config.log_dir)


@pytest.fixture
def retry_handler():
    """Interactive retry handler."""
    return RetryHandler(
        max_retries=config.max_retries,
        retry_delay=config.retry_delay
    )


@pytest.fixture
def conv2d_fixtures(fixture_manager):
    """Load Conv2D fixtures for testing."""
    try:
        return fixture_manager.load_algorithm_fixtures("conv2d")
    except Exception as e:
        pytest.skip(f"Skipping test - Conv2D fixtures not available: {e}")


@pytest.fixture
def conv2d_bundle(fixture_manager):
    """Load Conv2D bundle for testing."""
    try:
        return fixture_manager.load_bundle_fixture("conv2d")
    except Exception as e:
        pytest.skip(f"Skipping test - Conv2D bundle not available: {e}")


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Setup test environment before each test."""
    # Create log directory if it doesn't exist
    os.makedirs(config.log_dir, exist_ok=True)
    
    # Set debug environment variables if configured
    if config.debug_extraction:
        os.environ["ARDA_DEBUG_EXTRACTION"] = "true"
    
    if config.dump_responses:
        os.environ["ARDA_DUMP_OPENAI_RESPONSE"] = "true"
        os.environ["ARDA_DUMP_DIR"] = config.dump_dir


def pytest_configure(config):
    """Configure pytest with OpenAI test markers."""
    config.addinivalue_line(
        "markers", "openai: Tests that call OpenAI API (may incur costs)"
    )
    config.addinivalue_line(
        "markers", "openai_spec: Spec stage tests"
    )
    config.addinivalue_line(
        "markers", "openai_quant: Quant stage tests"
    )
    config.addinivalue_line(
        "markers", "openai_microarch: Microarch stage tests"
    )
    config.addinivalue_line(
        "markers", "openai_architecture: Architecture stage tests"
    )
    config.addinivalue_line(
        "markers", "openai_rtl: RTL stage tests"
    )
    config.addinivalue_line(
        "markers", "openai_verification: Verification stage tests"
    )
    config.addinivalue_line(
        "markers", "openai_test_generation: Test generation stage tests"
    )
    config.addinivalue_line(
        "markers", "openai_synth: Synth stage tests"
    )
    config.addinivalue_line(
        "markers", "openai_evaluate: Evaluate stage tests"
    )
    config.addinivalue_line(
        "markers", "openai_conv2d: Conv2D algorithm tests"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test names."""
    for item in items:
        # Add openai marker to all OpenAI stage tests
        if "test_openai_stages" in str(item.fspath):
            item.add_marker(pytest.mark.openai)
            
            # Add specific stage markers
            if "spec" in item.name:
                item.add_marker(pytest.mark.openai_spec)
            elif "quant" in item.name:
                item.add_marker(pytest.mark.openai_quant)
            elif "microarch" in item.name:
                item.add_marker(pytest.mark.openai_microarch)
            elif "architecture" in item.name:
                item.add_marker(pytest.mark.openai_architecture)
            elif "rtl" in item.name:
                item.add_marker(pytest.mark.openai_rtl)
            elif "verification" in item.name:
                item.add_marker(pytest.mark.openai_verification)
            elif "test_generation" in item.name:
                item.add_marker(pytest.mark.openai_test_generation)
            elif "synth" in item.name:
                item.add_marker(pytest.mark.openai_synth)
            elif "evaluate" in item.name:
                item.add_marker(pytest.mark.openai_evaluate)
            
            # Add algorithm markers
            if "conv2d" in item.name:
                item.add_marker(pytest.mark.openai_conv2d)
