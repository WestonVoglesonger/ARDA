"""
Stage output logger for OpenAI tests.

Comprehensive logging for stage execution including inputs, outputs, reasoning,
token usage, and error tracking. Integrates with existing token_usage.json format.
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


class StageLogger:
    """Logs comprehensive test execution data for OpenAI stages."""
    
    def __init__(self, log_dir: Optional[str] = None):
        """Initialize logger with optional custom log directory."""
        self.log_dir = Path(log_dir or os.getenv("ARDA_TEST_LOG_DIR", "tests/logs"))
        self.token_usage_file = Path("token_usage.json")
    
    def log_test_execution(
        self,
        test_name: str,
        algorithm: str,
        stage: str,
        inputs: Dict[str, Any],
        outputs: Any,
        reasoning: Optional[str] = None,
        token_usage: Optional[Dict[str, Any]] = None,
        duration_ms: Optional[float] = None,
        retries: int = 0,
        errors: Optional[list] = None,
        status: str = "passed"
    ) -> Path:
        """
        Log comprehensive test execution data.
        
        Returns:
            Path to the saved log file
        """
        log_entry = {
            "test_name": test_name,
            "algorithm": algorithm,
            "stage": stage,
            "timestamp": datetime.now().isoformat(),
            "status": status,
            "inputs": self._sanitize_inputs(inputs),
            "outputs": self._sanitize_outputs(outputs),
            "reasoning": reasoning,
            "token_usage": token_usage or {},
            "duration_ms": duration_ms,
            "retries": retries,
            "errors": errors or []
        }
        
        # Create organized subdirectory structure: logs/stage/status/
        stage_dir = self.log_dir / stage
        status_dir = stage_dir / status
        status_dir.mkdir(parents=True, exist_ok=True)
        
        # Save individual test log with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = status_dir / f"{algorithm}_{timestamp}.json"
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(log_entry, f, indent=2, ensure_ascii=False)
        
        # Update project-level token usage
        self._update_token_usage(stage, token_usage or {})
        
        return log_file
    
    def _sanitize_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize inputs for logging (remove sensitive data, truncate large content)."""
        sanitized = {}
        for key, value in inputs.items():
            if key == "bundle" and isinstance(value, str) and len(value) > 1000:
                sanitized[key] = value[:1000] + "... [truncated]"
            elif isinstance(value, dict):
                sanitized[key] = self._sanitize_inputs(value)
            else:
                sanitized[key] = value
        return sanitized
    
    def _sanitize_outputs(self, outputs: Any) -> Any:
        """Sanitize outputs for logging."""
        if hasattr(outputs, 'model_dump'):
            return outputs.model_dump()
        elif hasattr(outputs, 'dict'):
            return outputs.dict()
        elif isinstance(outputs, dict):
            return outputs
        else:
            return str(outputs)
    
    def _update_token_usage(self, stage: str, token_data: Dict[str, Any]) -> None:
        """Update project-level token usage tracking."""
        if not token_data:
            return
            
        try:
            # Load existing token usage
            if self.token_usage_file.exists():
                with open(self.token_usage_file, 'r', encoding='utf-8') as f:
                    usage_data = json.load(f)
            else:
                usage_data = {
                    "summary": {
                        "total_prompt_tokens": 0,
                        "total_completion_tokens": 0,
                        "total_tokens": 0,
                        "total_prompt_cost": 0.0,
                        "total_completion_cost": 0.0,
                        "total_cost": 0.0,
                        "total_duration_ms": 0.0,
                        "stage_breakdown": {},
                        "model_breakdown": {},
                        "agent_breakdown": {}
                    },
                    "detailed_records": []
                }
            
            # Update summary
            summary = usage_data["summary"]
            summary["total_tokens"] += token_data.get("total_tokens", 0)
            summary["total_duration_ms"] += token_data.get("duration_ms", 0)
            
            # Update stage breakdown
            if stage not in summary["stage_breakdown"]:
                summary["stage_breakdown"][stage] = {
                    "tokens": 0,
                    "cost": 0.0,
                    "calls": 0,
                    "duration_ms": 0.0
                }
            
            stage_data = summary["stage_breakdown"][stage]
            stage_data["tokens"] += token_data.get("total_tokens", 0)
            stage_data["calls"] += 1
            stage_data["duration_ms"] += token_data.get("duration_ms", 0)
            
            # Add detailed record
            detailed_record = {
                "stage": stage,
                "agent": f"{stage}_agent",
                "model": token_data.get("model", "unknown"),
                "call_type": "test",
                "prompt_tokens": token_data.get("prompt_tokens", 0),
                "completion_tokens": token_data.get("completion_tokens", 0),
                "total_tokens": token_data.get("total_tokens", 0),
                "prompt_cost": token_data.get("prompt_cost", 0.0),
                "completion_cost": token_data.get("completion_cost", 0.0),
                "total_cost": token_data.get("total_cost", 0.0),
                "duration_ms": token_data.get("duration_ms", 0.0),
                "timestamp": time.time()
            }
            
            usage_data["detailed_records"].append(detailed_record)
            
            # Save updated data
            with open(self.token_usage_file, 'w', encoding='utf-8') as f:
                json.dump(usage_data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            print(f"Warning: Failed to update token usage: {e}")
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get summary of all tests in this session."""
        # Find all JSON files in the organized directory structure
        log_files = []
        for stage_dir in self.log_dir.iterdir():
            if stage_dir.is_dir():
                for status_dir in stage_dir.iterdir():
                    if status_dir.is_dir() and status_dir.name in ["passed", "failed"]:
                        log_files.extend(status_dir.glob("*.json"))
        
        summary = {
            "log_dir": str(self.log_dir),
            "total_tests": len(log_files),
            "tests_by_stage": {},
            "tests_by_status": {"passed": 0, "failed": 0},
            "total_duration_ms": 0.0,
            "total_tokens": 0,
            "failed_tests": []
        }
        
        for log_file in log_files:
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                stage = data.get("stage", "unknown")
                status = data.get("status", "unknown")
                
                # Count by stage
                if stage not in summary["tests_by_stage"]:
                    summary["tests_by_stage"][stage] = {"passed": 0, "failed": 0}
                summary["tests_by_stage"][stage][status] += 1
                
                # Count by status
                if status in summary["tests_by_status"]:
                    summary["tests_by_status"][status] += 1
                
                summary["total_duration_ms"] += data.get("duration_ms", 0)
                summary["total_tokens"] += data.get("token_usage", {}).get("total_tokens", 0)
                
                if status == "failed":
                    summary["failed_tests"].append({
                        "file": str(log_file),
                        "test": data.get("test_name"),
                        "stage": stage,
                        "errors": data.get("errors", [])
                    })
                    
            except Exception as e:
                print(f"Warning: Failed to read log file {log_file}: {e}")
        
        return summary
