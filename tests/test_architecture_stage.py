"""
Tests for architecture stage and domain models.
"""

import pytest
from ardagen.domain import ArchitectureConfig, ModuleSpec


def test_architecture_config_validation():
    """Test basic architecture config validation."""
    modules = [
        ModuleSpec(
            name="params_pkg",
            purpose="Parameters and types",
            file_name="params.svh",
            estimated_lines=50,
            inputs=[],
            outputs=[],
            instantiates=[]
        ),
        ModuleSpec(
            name="fir_mac",
            purpose="FIR multiply-accumulate",
            file_name="fir_mac.sv",
            estimated_lines=120,
            inputs=[{"name": "clk", "width": "1", "description": "Clock"}],
            outputs=[{"name": "result", "width": "32", "description": "MAC result"}],
            instantiates=[]
        ),
        ModuleSpec(
            name="fir_top",
            purpose="Top-level integration",
            file_name="fir_top.sv",
            estimated_lines=80,
            inputs=[{"name": "clk", "width": "1", "description": "Clock"}],
            outputs=[{"name": "out", "width": "16", "description": "Output"}],
            instantiates=["fir_mac"]
        ),
    ]
    
    config = ArchitectureConfig(
        architecture_type="pipelined_fir",
        decomposition_rationale="Modular FIR with separate MAC unit",
        modules=modules,
        top_module="fir_top",
        hierarchy_diagram="fir_top -> fir_mac",
        pipeline_stages=5,
        parallelism_factor=1,
        memory_architecture="distributed_regs",
        confidence=90.0
    )
    
    assert len(config.modules) == 3
    assert config.top_module == "fir_top"
    assert config.architecture_type == "pipelined_fir"


def test_circular_dependency_detection():
    """Test that circular dependencies are detected."""
    modules = [
        ModuleSpec(
            name="module_a",
            purpose="Module A",
            file_name="a.sv",
            estimated_lines=100,
            inputs=[],
            outputs=[],
            instantiates=["module_b"]
        ),
        ModuleSpec(
            name="module_b",
            purpose="Module B",
            file_name="b.sv",
            estimated_lines=100,
            inputs=[],
            outputs=[],
            instantiates=["module_a"]  # Circular!
        ),
        ModuleSpec(
            name="top",
            purpose="Top",
            file_name="top.sv",
            estimated_lines=50,
            inputs=[],
            outputs=[],
            instantiates=[]
        ),
    ]
    
    with pytest.raises(ValueError, match="Circular dependency"):
        ArchitectureConfig(
            architecture_type="test",
            decomposition_rationale="Test",
            modules=modules,
            top_module="top",
            hierarchy_diagram="circular",
            pipeline_stages=1,
            parallelism_factor=1,
            memory_architecture="test",
            confidence=50.0
        )


def test_file_count_constraints():
    """Test minimum and maximum file count validation."""
    # Too few modules (< 3)
    with pytest.raises(ValueError):
        ArchitectureConfig(
            architecture_type="test",
            decomposition_rationale="Test",
            modules=[
                ModuleSpec(
                    name="a", purpose="A", file_name="a.sv",
                    estimated_lines=100, inputs=[], outputs=[], instantiates=[]
                ),
                ModuleSpec(
                    name="b", purpose="B", file_name="b.sv",
                    estimated_lines=100, inputs=[], outputs=[], instantiates=[]
                ),
            ],  # Only 2 modules!
            top_module="a",
            hierarchy_diagram="test",
            pipeline_stages=1,
            parallelism_factor=1,
            memory_architecture="test",
            confidence=50.0
        )
    
    # Too many modules (> 15)
    too_many = [
        ModuleSpec(
            name=f"module_{i}",
            purpose=f"Module {i}",
            file_name=f"module_{i}.sv",
            estimated_lines=100,
            inputs=[],
            outputs=[],
            instantiates=[]
        )
        for i in range(20)  # 20 modules!
    ]
    
    with pytest.raises(ValueError):
        ArchitectureConfig(
            architecture_type="test",
            decomposition_rationale="Test",
            modules=too_many,
            top_module="module_0",
            hierarchy_diagram="test",
            pipeline_stages=1,
            parallelism_factor=1,
            memory_architecture="test",
            confidence=50.0
        )


def test_module_spec_basic():
    """Test basic ModuleSpec creation."""
    spec = ModuleSpec(
        name="test_module",
        purpose="Test module for testing",
        file_name="test.sv",
        estimated_lines=100,
        inputs=[
            {"name": "clk", "width": "1", "description": "Clock signal"},
            {"name": "data_in", "width": "16", "description": "Input data"}
        ],
        outputs=[
            {"name": "data_out", "width": "16", "description": "Output data"}
        ],
        parameters=[
            {"name": "WIDTH", "value": "16"}
        ],
        instantiates=["sub_module_a", "sub_module_b"]
    )
    
    assert spec.name == "test_module"
    assert len(spec.inputs) == 2
    assert len(spec.outputs) == 1
    assert len(spec.instantiates) == 2

