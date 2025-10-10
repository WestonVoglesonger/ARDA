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
    dependencies = ("spec", "quant", "microarch")
    output_model = RTLConfig

    def gather_inputs(self, context: StageContext) -> Dict[str, Any]:
        inputs = super().gather_inputs(context)
        if not isinstance(inputs["spec"], SpecContract):
            raise TypeError("RTLStage requires SpecContract from 'spec' dependency.")
        if not isinstance(inputs["quant"], QuantConfig):
            raise TypeError("RTLStage requires QuantConfig from 'quant' dependency.")
        if not isinstance(inputs["microarch"], MicroArchConfig):
            raise TypeError("RTLStage requires MicroArchConfig from 'microarch' dependency.")
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
        """Write generated RTL files to workspace."""
        from ...workspace import workspace_manager
        
        workspace = workspace_manager.get_workspace(workspace_token)
        if not workspace:
            print(f"WARNING: Could not find workspace with token {workspace_token}")
            return
        
        # Map logical names to file paths
        file_map = {
            "params_svh": "rtl/params.svh",
            "algorithm_core_sv": "rtl/algorithm_core.sv", 
            "algorithm_top_sv": "rtl/algorithm_top.sv"
        }
        
        for logical_name, content in rtl_config.generated_files.items():
            if logical_name in file_map:
                path = file_map[logical_name]
                workspace.add_file(path, content)
                print(f"✓ Wrote {path} ({len(content)} bytes)")
            else:
                print(f"WARNING: Unknown file key '{logical_name}', skipping")
