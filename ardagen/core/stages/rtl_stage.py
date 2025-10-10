"""
RTL generation stage for the ARDA orchestrator.
"""

from __future__ import annotations

from typing import Any, Dict, TYPE_CHECKING

from pydantic import BaseModel
from .base import Stage, StageContext
from ...domain import RTLConfig, MicroArchConfig, QuantConfig, SpecContract

if TYPE_CHECKING:
    from ...core.strategies import AgentStrategy


class RTLStage(Stage):
    """Generate SystemVerilog implementation based on upstream design decisions."""

    name = "rtl"
    dependencies = ("spec", "quant", "microarch", "architecture")
    output_model = RTLConfig

    def gather_inputs(self, context: StageContext) -> Dict[str, Any]:
        inputs = super().gather_inputs(context)
        from ...domain import ArchitectureConfig
        
        if not isinstance(inputs["spec"], SpecContract):
            raise TypeError("RTLStage requires SpecContract from 'spec' dependency.")
        if not isinstance(inputs["quant"], QuantConfig):
            raise TypeError("RTLStage requires QuantConfig from 'quant' dependency.")
        if not isinstance(inputs["microarch"], MicroArchConfig):
            raise TypeError("RTLStage requires MicroArchConfig from 'microarch' dependency.")
        if not isinstance(inputs["architecture"], ArchitectureConfig):
            raise TypeError("RTLStage requires ArchitectureConfig from 'architecture' dependency.")
        return inputs

    async def run(self, context: StageContext, strategy: "AgentStrategy") -> BaseModel:
        """Run the RTL stage and write generated files to workspace."""
        # Call agent (returns RTLConfig with embedded code)
        rtl_config = await super().run(context, strategy)
        
        # Extract and write files to workspace
        workspace_token = context.run_inputs.get("workspace_token")
        if workspace_token and isinstance(rtl_config, RTLConfig):
            self._write_rtl_files(workspace_token, rtl_config)
        
        return rtl_config
    
    def _write_rtl_files(self, workspace_token: str, rtl_config: RTLConfig) -> None:
        """Write generated RTL files dynamically based on keys."""
        from ...workspace import workspace_manager
        
        workspace = workspace_manager.get_workspace(workspace_token)
        if not workspace:
            print(f"⚠️  Workspace {workspace_token} not found")
            return
        
        if not rtl_config.generated_files:
            print(f"⚠️  No files in generated_files")
            return
        
        files_written = 0
        for logical_name, content in rtl_config.generated_files.items():
            # Convert logical name to file path
            # e.g., "conv2d_pe_sv" → "rtl/conv2d_pe.sv"
            path = self._logical_to_physical_path(logical_name)
            
            # Validate
            if not self._validate_rtl_content(logical_name, content):
                print(f"⚠️  Skipping {logical_name}: validation failed")
                continue
            
            # Write
            workspace.add_file(path, content)
            print(f"✓ Wrote {path} ({len(content)} bytes)")
            files_written += 1
        
        print(f"✅ Wrote {files_written} RTL files")
    
    def _logical_to_physical_path(self, logical_name: str) -> str:
        """Convert logical_name to file path."""
        if logical_name.endswith("_svh"):
            filename = logical_name[:-4] + ".svh"
        elif logical_name.endswith("_sv"):
            filename = logical_name[:-3] + ".sv"
        else:
            filename = logical_name + ".sv"
        return f"rtl/{filename}"
    
    def _validate_rtl_content(self, logical_name: str, content: str) -> bool:
        """Validate RTL file content."""
        if not content or len(content) < 100:
            return False
        
        # Check for module/package
        if logical_name.endswith("_svh"):
            if not any(kw in content for kw in ["package ", "parameter ", "typedef "]):
                return False
        else:
            if "module " not in content or "endmodule" not in content:
                return False
            
            if content.count("module ") != content.count("endmodule"):
                return False
        
        return True
