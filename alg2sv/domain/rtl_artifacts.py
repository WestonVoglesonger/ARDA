"""
SystemVerilog artifact models for ARDA.
"""

from typing import Any, Dict, List
from pydantic import BaseModel


class RTLConfig(BaseModel):
    """RTL generation configuration."""

    rtl_files: List[str]
    params_file: str
    top_module: str
    lint_passed: bool
    estimated_resources: Dict[
        str, Any
    ]  # Allow mixed types (int for counts, str for notes)

    class Config:
        extra = "allow"
