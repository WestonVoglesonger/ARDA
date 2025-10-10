"""
Test RTL JSON generation approach (embedded code in JSON response).
"""

import pytest
from ardagen.domain.rtl_artifacts import RTLConfig
from ardagen.workspace import Workspace


def test_rtl_config_with_generated_files():
    """Test RTLConfig validates with embedded files."""
    config_data = {
        "generated_files": {
            "params_svh": "// Parameters\npackage params;\nendpackage",
            "algorithm_core_sv": "module core; endmodule",
            "algorithm_top_sv": "module top; endmodule"
        },
        "file_paths": ["rtl/params.svh", "rtl/algorithm_core.sv", "rtl/algorithm_top.sv"],
        "top_module": "top",
        "estimated_resources": {"lut": 5000, "ff": 8000, "dsp": 16},
        "confidence": 85.0
    }
    
    config = RTLConfig(**config_data)
    
    assert config.generated_files["params_svh"].startswith("// Parameters")
    assert config.generated_files["algorithm_core_sv"] == "module core; endmodule"
    assert config.top_module == "top"
    assert len(config.file_paths) == 3


def test_rtl_config_accepts_partial_files():
    """Test RTLConfig accepts generated_files with any keys (schema validation is at API level)."""
    config_data = {
        "generated_files": {
            "params_svh": "// Parameters"
            # Note: The "required" constraint is enforced by OpenAI API schema, not Pydantic
        },
        "file_paths": ["rtl/params.svh"],
        "top_module": "top",
        "estimated_resources": {"lut": 5000},
        "confidence": 85.0
    }
    
    # This should succeed - Pydantic only validates the dict type, not specific keys
    config = RTLConfig(**config_data)
    assert "params_svh" in config.generated_files
    assert len(config.generated_files) == 1


def test_workspace_file_writing():
    """Test writing RTL files to workspace."""
    from ardagen.core.stages.rtl_stage import RTLStage
    
    workspace = Workspace()
    
    rtl_config = RTLConfig(
        generated_files={
            "params_svh": "package params; endpackage",
            "algorithm_core_sv": "module core; endmodule",
            "algorithm_top_sv": "module top; endmodule"
        },
        file_paths=["rtl/params.svh", "rtl/algorithm_core.sv", "rtl/algorithm_top.sv"],
        top_module="top",
        estimated_resources={"lut": 5000, "ff": 8000, "dsp": 16},
        confidence=85.0
    )
    
    stage = RTLStage()
    
    # Manually call the file writing method
    from ardagen.workspace import workspace_manager
    workspace_token = "test_token_123"
    workspace_manager.workspaces[workspace_token] = workspace
    
    stage._write_rtl_files(workspace_token, rtl_config)
    
    # Verify files were written
    assert workspace.get_file("rtl/params.svh") == "package params; endpackage"
    assert workspace.get_file("rtl/algorithm_core.sv") == "module core; endmodule"
    assert workspace.get_file("rtl/algorithm_top.sv") == "module top; endmodule"
    
    # Clean up
    del workspace_manager.workspaces[workspace_token]


def test_rtl_config_backward_compatibility():
    """Test backward compatibility with rtl_files field."""
    config_data = {
        "generated_files": {
            "params_svh": "// Parameters",
            "algorithm_core_sv": "module core; endmodule",
            "algorithm_top_sv": "module top; endmodule"
        },
        "file_paths": ["rtl/params.svh", "rtl/algorithm_core.sv", "rtl/algorithm_top.sv"],
        "rtl_files": ["rtl/params.svh", "rtl/algorithm_core.sv"],  # Old field
        "top_module": "top",
        "estimated_resources": {"lut": 5000, "ff": 8000, "dsp": 16},
        "confidence": 85.0
    }
    
    config = RTLConfig(**config_data)
    
    # Should accept both fields
    assert config.file_paths == ["rtl/params.svh", "rtl/algorithm_core.sv", "rtl/algorithm_top.sv"]
    assert config.rtl_files == ["rtl/params.svh", "rtl/algorithm_core.sv"]


def test_file_content_with_special_characters():
    """Test RTL files with special characters (newlines, quotes, etc.)."""
    verilog_code = '''module top (
    input  wire clk,
    input  wire rst_n,
    output wire [15:0] data_out
);
    // This is a comment with "quotes" and 'apostrophes'
    reg [15:0] internal_reg;
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            internal_reg <= 16'h0000;
        else
            internal_reg <= internal_reg + 1;
    end
    
    assign data_out = internal_reg;
endmodule
'''
    
    config_data = {
        "generated_files": {
            "params_svh": "package params; parameter WIDTH = 16; endpackage",
            "algorithm_core_sv": verilog_code,
            "algorithm_top_sv": "module wrapper; endmodule"
        },
        "file_paths": ["rtl/params.svh", "rtl/algorithm_core.sv", "rtl/algorithm_top.sv"],
        "top_module": "top",
        "estimated_resources": {"lut": 100, "ff": 200, "dsp": 0},
        "confidence": 90.0
    }
    
    config = RTLConfig(**config_data)
    
    # Verify special characters preserved
    assert "clk" in config.generated_files["algorithm_core_sv"]
    assert "rst_n" in config.generated_files["algorithm_core_sv"]
    assert '"quotes"' in config.generated_files["algorithm_core_sv"]
    assert "16'h0000" in config.generated_files["algorithm_core_sv"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

