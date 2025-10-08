# ARDA Developer Guide

Comprehensive guide for contributors, advanced users, and system integrators.

## Table of Contents

- [Project Structure](#project-structure)
- [Development Setup](#development-setup)
- [Architecture Overview](#architecture-overview)
- [Adding New Agents](#adding-new-agents)
- [Extending Pipeline Stages](#extending-pipeline-stages)
- [Tool Integration](#tool-integration)
- [Testing](#testing)
- [Debugging](#debugging)
- [Performance Optimization](#performance-optimization)

## Project Structure

```
arda/
├── alg2sv/                 # Main package
│   ├── agents/            # AI agent implementations
│   │   ├── __init__.py
│   │   ├── config_loader.py    # Agent configuration loading
│   │   ├── openai_runner.py    # OpenAI Agents SDK integration
│   │   ├── registry.py         # Agent registry and management
│   │   ├── strategies.py       # Agent execution strategies
│   │   └── tools.py           # Function tools for agents
│   ├── bundle_utils.py        # Bundle creation utilities
│   ├── cli.py                 # Command-line interface
│   ├── core/                  # Pipeline orchestration
│   │   ├── __init__.py
│   │   ├── orchestrator.py     # Stage execution coordinator
│   │   ├── stages/            # Individual pipeline stages
│   │   │   ├── __init__.py
│   │   │   ├── base.py        # Base stage class
│   │   │   ├── spec_stage.py
│   │   │   ├── quant_stage.py
│   │   │   ├── microarch_stage.py
│   │   │   ├── rtl_stage.py
│   │   │   ├── simulation_stage.py
│   │   │   ├── synth_stage.py
│   │   │   └── evaluate_stage.py
│   │   └── strategies.py      # Execution strategies
│   ├── domain/                # Data models
│   │   ├── __init__.py
│   │   ├── architecture.py
│   │   ├── contracts.py
│   │   ├── evaluation.py
│   │   ├── feedback.py
│   │   ├── quantization.py
│   │   ├── rtl_artifacts.py
│   │   ├── synthesis.py
│   │   └── verification.py
│   ├── model_config.py        # Model configuration
│   ├── observability/         # Logging and metrics
│   │   ├── __init__.py
│   │   ├── events.py          # Event definitions
│   │   ├── manager.py         # Observability management
│   │   └── tools.py           # Observability tools
│   ├── runtime/               # Runtime components
│   │   ├── __init__.py
│   │   └── agent_runner.py    # Agent runner interface
│   ├── pipeline.py # High-level pipeline runner
│   ├── tools/                 # External tool integrations
│   │   ├── __init__.py
│   │   ├── lint.py            # RTL linting tools
│   │   ├── reporting.py       # Report generation
│   │   ├── simulation.py      # RTL simulation tools
│   │   └── synthesis.py       # FPGA synthesis tools
│   └── workspace.py           # Virtual workspace management
├── docs/                      # Documentation
│   ├── adr/                   # Architecture decision records
│   ├── architecture.md        # Technical architecture
│   ├── user_guide.md          # User documentation
│   └── developer_guide.md     # This file
├── examples/                  # Example bundles
├── generated_rtl/             # Generated RTL output
└── tests/                     # Test suite
```

## Development Setup

### Prerequisites

```bash
# Python environment
python3.8+
pip install -e .

# Optional: OpenAI API key for AI agents
export OPENAI_API_KEY="your-key-here"

# Optional: RTL simulation tools
# Ubuntu/Debian:
sudo apt-get install iverilog verilator

# macOS:
brew install icarus-verilog verilator

# FPGA synthesis (optional)
# Xilinx Vivado or AMD Xilinx tools
# Lattice Diamond or open-source alternatives
```

### Development Installation

```bash
# Clone repository
git clone https://github.com/WestonVoglesonger/ARDA.git
cd ARDA

# Install in development mode (ARDA not yet on PyPI)
pip install -e .

# Install development dependencies
pip install -e ".[dev]"
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test modules
pytest tests/test_pipeline_feedback.py -v

# Run with coverage
pytest --cov=arda tests/
```

## Architecture Overview

### Pipeline Architecture

ARDA uses a modular, stage-based architecture:

```
Algorithm Bundle → [Spec] → [Quant] → [MicroArch] → [RTL] → [Static] → [Verify] → [Synth] → [Evaluate]
                        ↑              ↑              ↑         ↑         ↑         ↑         ↑
                    Feedback Loop ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ←
```

### Key Components

#### 1. Pipeline Orchestrator (`alg2sv/core/orchestrator.py`)
- Coordinates stage execution
- Manages feedback loops and retries
- Handles error recovery and stage dependencies

#### 2. Agent System (`alg2sv/agents/`)
- **OpenAI Runner**: Integrates with OpenAI Agents SDK
- **Registry**: Manages available agents and their configurations
- **Tools**: Function tools callable by agents

#### 3. Stage System (`alg2sv/core/stages/`)
- **Base Stage**: Common functionality for all stages
- **Specialized Stages**: Implementation for each pipeline phase

#### 4. Domain Models (`alg2sv/domain/`)
- **Structured data models** for each pipeline stage
- **Type safety** and validation
- **Serialization** support for observability

## Adding New Agents

### 1. Define Agent Configuration

Add to `agent_configs.json`:

```json
{
  "agents": {
    "my_custom_agent": {
      "name": "My Custom Agent",
      "instructions": "Your custom agent instructions...",
      "model": "gpt-4",
      "tools": [
        {
          "type": "function",
          "name": "my_custom_tool",
          "description": "Description of what this tool does"
        }
      ],
      "output_schema": {
        "custom_field": { "type": "string" },
        "result_value": { "type": "number" }
      }
    }
  },
  "function_tools": {
    "my_custom_tool": {
      "schema": {
        "name": "my_custom_tool",
        "description": "Tool description",
        "parameters": {
          "type": "object",
          "properties": {
            "input_param": { "type": "string" }
          },
          "required": ["input_param"],
          "additionalProperties": false
        }
      }
    }
  }
}
```

### 2. Implement Function Tool

In `alg2sv/agents/tools.py`:

```python
def my_custom_tool(input_param: str) -> Dict[str, Any]:
    """Custom tool implementation."""
    # Your tool logic here
    return {"result": "processed_" + input_param}

# Add to FUNCTION_MAP
FUNCTION_MAP = {
    # ... existing tools
    "my_custom_tool": my_custom_tool,
}
```

### 3. Create Agent Handler

In your custom module:

```python
from alg2sv.agents.registry import AgentRegistry

def my_custom_agent_handler(context: Mapping[str, Any]) -> MyOutputModel:
    """Agent handler that processes context and returns structured output."""
    # Your agent logic here
    return MyOutputModel(custom_field="result", result_value=42.0)
```

### 4. Register Agent

```python
registry = AgentRegistry()
registry.register_stage_agent("my_stage", my_custom_agent_handler)
```

## Extending Pipeline Stages

### Creating a New Stage

1. **Define Domain Model** (`alg2sv/domain/`):
```python
from pydantic import BaseModel

class MyStageOutput(BaseModel):
    """Output model for my custom stage."""
    custom_result: str
    computed_value: float
```

2. **Implement Stage Class** (`alg2sv/core/stages/`):
```python
from .base import Stage, StageContext
from ...domain import MyStageOutput

class MyCustomStage(Stage):
    """Custom pipeline stage."""

    name = "my_stage"
    dependencies = ("previous_stage",)
    output_model = MyStageOutput

    def gather_inputs(self, context: StageContext) -> Dict[str, Any]:
        """Gather inputs from previous stages."""
        inputs = super().gather_inputs(context)
        # Custom input gathering logic
        return inputs

    async def run(self, context: StageContext) -> MyStageOutput:
        """Execute the stage."""
        inputs = self.gather_inputs(context)

        # Stage execution logic
        result = await self.execute_stage_logic(inputs)

        return MyStageOutput(
            custom_result=result["output"],
            computed_value=result["value"]
        )
```

3. **Register with Pipeline**:
```python
# In your pipeline setup
stages = [
    SpecStage(),
    QuantStage(),
    MyCustomStage(),  # Add your custom stage
    RTLStage(),
    # ... other stages
]
```

## Tool Integration

### Adding External Tools

#### RTL Simulation Tools

In `alg2sv/tools/simulation.py`:

```python
def run_custom_simulator(rtl_files: List[str], test_vectors: List[Dict]) -> Dict[str, Any]:
    """Integrate with custom RTL simulator."""
    # Tool integration logic
    return {
        "simulator": "custom_sim",
        "passed": True,
        "results": {...}
    }
```

#### Synthesis Tools

In `alg2sv/tools/synthesis.py`:

```python
def run_custom_synthesis(rtl_files: List[str], constraints: Dict) -> Dict[str, Any]:
    """Integrate with custom FPGA synthesis tool."""
    # Synthesis integration logic
    return {
        "fmax_mhz": 150.0,
        "lut_usage": 5000,
        "timing_met": True
    }
```

### Tool Configuration

Update `agent_configs.json` to use new tools:

```json
{
  "function_tools": {
    "run_custom_simulator": {
      "schema": {
        "name": "run_custom_simulator",
        "parameters": {
          "type": "object",
          "properties": {
            "rtl_files": {"type": "array", "items": {"type": "string"}},
            "test_vectors": {"type": "array", "items": {"type": "object"}}
          },
          "required": ["rtl_files", "test_vectors"],
          "additionalProperties": false
        }
      }
    }
  }
}
```

## Testing

### Test Structure

```python
# tests/test_my_feature.py
import pytest
from alg2sv.core.stages.my_stage import MyCustomStage
from alg2sv.domain import MyStageOutput

def test_my_stage_basic():
    """Test basic stage functionality."""
    stage = MyCustomStage()

    # Setup test context
    context = StageContext(
        run_inputs={"test_param": "value"},
        results={"previous_stage": PreviousStageOutput(...)}
    )

    # Execute stage
    result = await stage.run(context)

    # Assertions
    assert isinstance(result, MyStageOutput)
    assert result.custom_result == "expected_value"
```

### Running Tests

```bash
# All tests
pytest

# Specific test file
pytest tests/test_my_feature.py

# With coverage
pytest --cov=alg2sv --cov-report=html

# Verbose output
pytest -v -s
```

## Debugging

### Debug Logging

Enable detailed logging:

```python
import logging

# Enable debug logging
logging.getLogger('alg2sv').setLevel(logging.DEBUG)

# Run with verbose CLI
arda algorithm.txt --verbose --debug
```

### Pipeline Debugging

#### Stage-by-Stage Debugging

```python
from alg2sv.core.orchestrator import PipelineOrchestrator
from alg2sv.core.stages import SpecStage, QuantStage

# Create minimal pipeline for debugging
stages = [SpecStage(), QuantStage()]
orchestrator = PipelineOrchestrator(stages=stages)

# Run with debugging
async for stage_name, output in orchestrator.run_iter(run_inputs):
    print(f"Stage {stage_name} completed: {output}")
```

#### Agent Debugging

```python
# Enable agent debugging
import os
os.environ['DEBUG'] = 'openai-agents:*'

# Run with agent debugging
arda algorithm.txt --verbose --agent-runner openai
```

### Common Debugging Patterns

#### Context Inspection

```python
def debug_context(stage_name: str, context: StageContext):
    """Debug helper to inspect stage context."""
    print(f"=== {stage_name} Context ===")
    print(f"Run inputs: {context.run_inputs}")
    print(f"Results keys: {list(context.results.keys())}")

    for stage, result in context.results.items():
        print(f"  {stage}: {type(result).__name__}")
        if hasattr(result, 'model_dump'):
            print(f"    Fields: {list(result.model_dump().keys())}")
```

#### Tool Call Debugging

```python
# Patch tool functions for debugging
original_tool = FUNCTION_MAP["my_tool"]

def debug_tool(*args, **kwargs):
    print(f"Tool called: my_tool({args}, {kwargs})")
    result = original_tool(*args, **kwargs)
    print(f"Tool result: {result}")
    return result

FUNCTION_MAP["my_tool"] = debug_tool
```

## Performance Optimization

### Profiling Pipeline Performance

```python
import cProfile
import pstats

# Profile pipeline execution
profiler = cProfile.Profile()
profiler.enable()

# Run pipeline
result = await pipeline.run(algorithm_bundle)

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)  # Top 20 slowest functions
```

### Memory Profiling

```python
import tracemalloc

# Start memory tracing
tracemalloc.start()

# Run pipeline
result = await pipeline.run(algorithm_bundle)

# Show memory usage
current, peak = tracemalloc.get_traced_memory()
print(f"Current memory: {current / 1024 / 1024:.2f} MB")
print(f"Peak memory: {peak / 1024 / 1024:.2f} MB")
```

### Optimization Strategies

#### Agent Performance

```python
# Cache expensive computations
@functools.lru_cache(maxsize=128)
def expensive_computation(params):
    # Expensive logic here
    return result

# Use async processing for I/O bound operations
async def async_heavy_computation():
    # Use asyncio for concurrent operations
    pass
```

#### Memory Optimization

```python
# Process large datasets in chunks
def process_large_dataset_in_chunks(data, chunk_size=1000):
    for i in range(0, len(data), chunk_size):
        chunk = data[i:i + chunk_size]
        # Process chunk
        yield process_chunk(chunk)
```

## API Reference

### Core Classes

#### PipelineAgentRunner

```python
class PipelineAgentRunner(Protocol):
    """Interface for executing pipeline stages."""

    async def run_stage(self, stage: str, context: Mapping[str, Any]) -> Any:
        """Execute a pipeline stage."""

    async def run_feedback(self, context: Mapping[str, Any]) -> FeedbackDecision:
        """Generate feedback decision."""
```

#### Stage Base Class

```python
class Stage(ABC):
    """Base class for pipeline stages."""

    name: str
    dependencies: Tuple[str, ...]
    output_model: Type[BaseModel]

    def gather_inputs(self, context: StageContext) -> Dict[str, Any]:
        """Gather inputs from context."""

    async def run(self, context: StageContext) -> BaseModel:
        """Execute the stage."""

    def validate_output(self, output: BaseModel, context: StageContext) -> None:
        """Validate stage output."""
```

### Key Functions

#### Bundle Creation

```python
from ardagen.bundle_utils import create_bundle

# Create bundle from file
bundle = create_bundle("algorithm.py")

# Create bundle from directory
bundle = create_bundle("my_project/")

# Save to file
bundle = create_bundle("algorithm.py", "algorithm_bundle.txt")
```

#### Agent Execution

```python
from ardagen.agents.openai_runner import OpenAIAgentRunner

runner = OpenAIAgentRunner()
result = await runner.run_stage("rtl", context)
```

## Contributing Guidelines

### Code Style

- **Type hints**: Use comprehensive type annotations
- **Documentation**: Docstrings for all public functions/classes
- **Tests**: Test coverage for new features
- **Linting**: Follow PEP 8 and use black for formatting

### Pull Request Process

1. **Create feature branch**: `git checkout -b feature/my-feature`
2. **Write tests**: Add comprehensive test coverage
3. **Update documentation**: Update relevant docs
4. **Run tests**: Ensure all tests pass
5. **Create PR**: Submit pull request with clear description

### Code Review Checklist

- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] Type hints complete
- [ ] Error handling robust
- [ ] Performance impact considered
- [ ] Security implications reviewed

## Advanced Topics

### Custom Model Configuration

```python
# ardagen/model_config.py
CUSTOM_MODELS = {
    'high_accuracy': 'gpt-4',
    'fast_generation': 'gpt-3.5-turbo',
    'cost_effective': 'gpt-4o-mini'
}
```

### Plugin Architecture

```python
# Custom plugin system
class ARDAPlugin:
    def register_agents(self, registry: AgentRegistry):
        """Register custom agents."""

    def register_tools(self, tool_map: Dict[str, Any]):
        """Register custom tools."""

    def extend_pipeline(self, stages: List[Stage]) -> List[Stage]:
        """Add custom pipeline stages."""
```

### Performance Monitoring

```python
# Custom observability
from ardagen.observability.manager import ObservabilityManager

obs = ObservabilityManager()
obs.stage_started("rtl", {"attempt": 1})
obs.tool_invoked("rtl", "write_artifact", {"path": "rtl/core.sv"})
obs.stage_completed("rtl", rtl_output)
```

## Troubleshooting Development Issues

### Common Development Problems

#### Import Errors
```
ImportError: No module named 'arda'
```
**Solution**: Install in development mode: `pip install -e .`

#### Type Checking Errors
```
error: Incompatible types in assignment
```
**Solution**: Add proper type annotations and use mypy for checking.

#### Test Failures
```
FAILED tests/test_my_feature.py::test_function - AssertionError
```
**Solution**: Update tests to match new behavior, check test isolation.

#### Performance Issues
```
Pipeline execution too slow
```
**Solution**: Profile with cProfile, optimize bottlenecks, consider async processing.

## Getting Help

- **Issues**: [GitHub Issues](https://github.com/WestonVoglesonger/ARDA/issues)
- **Discussions**: [GitHub Discussions](https://github.com/WestonVoglesonger/ARDA/discussions)
- **Documentation**: This developer guide
- **Architecture**: [Technical Architecture](architecture.md)

## Release Process

1. **Version bump**: Update version in `pyproject.toml`
2. **Changelog update**: Add entry to `CHANGELOG.md`
3. **Test run**: Ensure all tests pass
4. **Documentation update**: Update docs for new features
5. **Release commit**: `git commit -m "Release vX.Y.Z"`
6. **Tag creation**: `git tag vX.Y.Z`
7. **Publication**: Publish to PyPI

---

*This developer guide is maintained by the ARDA development team. For questions or suggestions, please open an issue or discussion on GitHub.*
