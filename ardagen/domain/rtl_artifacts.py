"""
SystemVerilog artifact models for ARDA.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class RTLConfig(BaseModel):
    """RTL generation configuration and artifacts."""

    # NEW: Embedded file contents
    generated_files: Dict[str, str] = Field(
        description="Generated SystemVerilog file contents keyed by logical name"
    )
    
    # Existing fields
    file_paths: List[str] = Field(description="Paths where files will be written")
    top_module: str
    estimated_resources: Dict[str, int]
    confidence: float = Field(default=80.0, ge=0, le=100, description="Confidence level (0-100%)")
    
    # Optional fields
    lint_passed: bool = False
    params_file: Optional[str] = None
    
    # Deprecated field (for backward compatibility)
    rtl_files: Optional[List[str]] = None

    class Config:
        extra = "allow"
