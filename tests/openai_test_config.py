"""
Configuration management for OpenAI tests.

Loads environment variables, defines test modes, and provides configuration
for pytest markers and token usage tracking.
"""

import os
from typing import Dict, Any, Optional
from dotenv import load_dotenv


class OpenAITestConfig:
    """Configuration manager for OpenAI stage tests."""
    
    def __init__(self):
        """Initialize configuration by loading .env file."""
        load_dotenv()
        self._config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from environment variables."""
        return {
            "openai_api_key": os.getenv("OPENAI_API_KEY"),
            "test_mode": os.getenv("ARDA_TEST_MODE", "fixture"),
            "log_dir": os.getenv("ARDA_TEST_LOG_DIR", "tests/logs"),
            "debug_extraction": os.getenv("ARDA_DEBUG_EXTRACTION", "false").lower() == "true",
            "dump_responses": os.getenv("ARDA_DUMP_OPENAI_RESPONSE", "false").lower() == "true",
            "dump_dir": os.getenv("ARDA_DUMP_DIR", "/tmp"),
            "max_retries": int(os.getenv("ARDA_MAX_RETRIES", "3")),
            "retry_delay": float(os.getenv("ARDA_RETRY_DELAY", "1.0")),
            "min_confidence": float(os.getenv("ARDA_MIN_CONFIDENCE", "70.0")),
        }
    
    @property
    def openai_api_key(self) -> Optional[str]:
        """Get OpenAI API key."""
        return self._config["openai_api_key"]
    
    @property
    def test_mode(self) -> str:
        """Get test mode (live, fixture, hybrid)."""
        return self._config["test_mode"]
    
    @property
    def log_dir(self) -> str:
        """Get log directory path."""
        return self._config["log_dir"]
    
    @property
    def debug_extraction(self) -> bool:
        """Whether to enable debug extraction."""
        return self._config["debug_extraction"]
    
    @property
    def dump_responses(self) -> bool:
        """Whether to dump OpenAI responses."""
        return self._config["dump_responses"]
    
    @property
    def dump_dir(self) -> str:
        """Directory for dumping responses."""
        return self._config["dump_dir"]
    
    @property
    def max_retries(self) -> int:
        """Maximum number of retries."""
        return self._config["max_retries"]
    
    @property
    def retry_delay(self) -> float:
        """Delay between retries in seconds."""
        return self._config["retry_delay"]
    
    @property
    def min_confidence(self) -> float:
        """Minimum confidence score threshold."""
        return self._config["min_confidence"]
    
    def is_live_mode(self) -> bool:
        """Check if running in live mode (calls OpenAI API)."""
        return self.test_mode in ["live", "hybrid"]
    
    def is_fixture_mode(self) -> bool:
        """Check if running in fixture mode (uses test data)."""
        return self.test_mode in ["fixture", "hybrid"]
    
    def validate_setup(self) -> Dict[str, Any]:
        """
        Validate test setup and return status.
        
        Returns:
            Dictionary with validation results
        """
        validation = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        # Check API key
        if self.is_live_mode() and not self.openai_api_key:
            validation["valid"] = False
            validation["errors"].append("OPENAI_API_KEY not set in environment")
        
        # Check log directory
        try:
            os.makedirs(self.log_dir, exist_ok=True)
        except Exception as e:
            validation["warnings"].append(f"Cannot create log directory {self.log_dir}: {e}")
        
        # Check dump directory
        if self.dump_responses:
            try:
                os.makedirs(self.dump_dir, exist_ok=True)
            except Exception as e:
                validation["warnings"].append(f"Cannot create dump directory {self.dump_dir}: {e}")
        
        return validation
    
    def get_model_config(self, stage: str) -> Dict[str, Any]:
        """
        Get model configuration for a specific stage.
        
        Args:
            stage: Name of the stage
            
        Returns:
            Model configuration dictionary
        """
        # Default model configurations
        model_configs = {
            "spec": {
                "model": "gpt-4o-mini",
                "temperature": 0.1,
                "max_tokens": 2000
            },
            "quant": {
                "model": "gpt-4o-mini", 
                "temperature": 0.1,
                "max_tokens": 2000
            },
            "microarch": {
                "model": "gpt-4o-mini",
                "temperature": 0.2,
                "max_tokens": 2000
            },
            "architecture": {
                "model": "gpt-4o",
                "temperature": 0.3,
                "max_tokens": 4000
            },
            "rtl": {
                "model": "gpt-4o",
                "temperature": 0.1,
                "max_tokens": 8000
            },
            "verification": {
                "model": "gpt-4o-mini",
                "temperature": 0.1,
                "max_tokens": 2000
            },
            "synth": {
                "model": "gpt-4o-mini",
                "temperature": 0.1,
                "max_tokens": 2000
            },
            "evaluate": {
                "model": "gpt-4o-mini",
                "temperature": 0.2,
                "max_tokens": 2000
            }
        }
        
        return model_configs.get(stage, {
            "model": "gpt-4o-mini",
            "temperature": 0.1,
            "max_tokens": 2000
        })
    
    def get_pytest_markers(self) -> Dict[str, str]:
        """
        Get pytest markers for selective test execution.
        
        Returns:
            Dictionary mapping marker names to descriptions
        """
        return {
            "openai": "Tests that call OpenAI API (may incur costs)",
            "openai_spec": "Spec stage tests",
            "openai_quant": "Quant stage tests", 
            "openai_microarch": "Microarch stage tests",
            "openai_architecture": "Architecture stage tests",
            "openai_rtl": "RTL stage tests",
            "openai_verification": "Verification stage tests",
            "openai_synth": "Synth stage tests",
            "openai_evaluate": "Evaluate stage tests",
            "openai_conv2d": "Conv2D algorithm tests",
            "openai_fft256": "FFT256 algorithm tests",
            "openai_bpf16": "BPF16 algorithm tests",
            "openai_adaptive_filter": "Adaptive filter algorithm tests"
        }


# Global configuration instance
config = OpenAITestConfig()
