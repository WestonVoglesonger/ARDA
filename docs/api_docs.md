# ARDA API Documentation

Comprehensive API reference for ARDA's core modules and classes.

## Table of Contents

- [Core Pipeline](#core-pipeline)
- [Agent System](#agent-system)
- [Stage System](#stage-system)
- [Domain Models](#domain-models)
- [Tool Integration](#tool-integration)
- [Bundle Utilities](#bundle-utilities)
- [Observability](#observability)

## Core Pipeline

### PipelineAgentRunner

Interface for executing pipeline stages and feedback decisions.

```python
class PipelineAgentRunner(Protocol):
    """Interface for executing pipeline stages."""

    async def run_stage(self, stage: str, context: Mapping[str, Any]) -> Any:
        """Execute a pipeline stage and return its output."""

    async def run_feedback(self, context: Mapping[str, Any]) -> FeedbackDecision:
        """Generate feedback decision based on current pipeline state."""
```

### OpenAIAgentRunner

OpenAI Agents SDK implementation of PipelineAgentRunner.

```python
class OpenAIAgentRunner(PipelineAgentRunner):
    """Executes ARDA stages using OpenAI's Agents SDK."""

    def __init__(
        self,
        *,
        client: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None,
        fallback_runner: Optional[PipelineAgentRunner] = None,
    ) -> None:
        """Initialize OpenAI agent runner."""

    async def run_stage(self, stage: str, context: Mapping[str, Any]) -> Any:
        """Execute stage using OpenAI agents."""

    async def run_feedback(self, context: Mapping[str, Any]) -> FeedbackDecision:
        """Generate feedback using OpenAI agents."""
```

## Agent System

### AgentRegistry

Manages agent registrations and configurations.

```python
class AgentRegistry:
    """Registry for pipeline agents and their configurations."""

    def register_stage_agent(
        self,
        stage_name: str,
        handler: Callable[[Mapping[str, Any]], Any],
        description: str = "",
    ) -> None:
        """Register a stage agent handler."""

    def register_feedback_agent(
        self,
        handler: Callable[[Mapping[str, Any]], FeedbackDecision],
        description: str = "",
    ) -> None:
        """Register a feedback agent handler."""

    def get_stage_agent(self, stage_name: str) -> Callable:
        """Get registered stage agent handler."""
```

### Agent Configuration

Agent configurations are loaded from `agent_configs.json`:

```json
{
  "agents": {
    "rtl_agent": {
      "name": "RTL Agent",
      "instructions": "Generate synthesizable SystemVerilog...",
      "model": "gpt-4",
      "tools": [...],
      "output_schema": {...}
    }
  },
  "function_tools": {
    "write_artifact": {
      "schema": {
        "name": "write_artifact",
        "parameters": {...}
      }
    }
  }
}
```

## Stage System

### Base Stage Class

```python
class Stage(ABC):
    """Base class for all pipeline stages."""

    name: str
    dependencies: Tuple[str, ...]
    output_model: Type[BaseModel]

    def gather_inputs(self, context: StageContext) -> Dict[str, Any]:
        """Gather inputs from previous stages."""

    async def run(self, context: StageContext) -> BaseModel:
        """Execute the stage and return typed output."""

    def validate_output(self, output: BaseModel, context: StageContext) -> None:
        """Validate stage output against requirements."""
```

### StageContext

Context passed between pipeline stages.

```python
@dataclass
class StageContext:
    """Context information for stage execution."""

    run_inputs: Mapping[str, Any]
    results: Dict[str, Any]
    workspace_token: Optional[str] = None
    observability: Optional[ObservabilityManager] = None
```

### Built-in Stages

#### SpecStage
```python
class SpecStage(Stage):
    """Generate hardware contract from algorithm bundle."""

    name = "spec"
    dependencies = ()
    output_model = SpecContract
```

#### QuantStage
```python
class QuantStage(Stage):
    """Convert floating-point to fixed-point arithmetic."""

    name = "quant"
    dependencies = ("spec",)
    output_model = QuantConfig
```

#### MicroArchStage
```python
class MicroArchStage(Stage):
    """Design micro-architecture and dataflow."""

    name = "microarch"
    dependencies = ("quant",)
    output_model = MicroArchConfig
```

#### RTLStage
```python
class RTLStage(Stage):
    """Generate synthesizable SystemVerilog RTL."""

    name = "rtl"
    dependencies = ("spec", "quant", "microarch")
    output_model = RTLConfig
```

## Domain Models

### Confidence-Based Feedback

All domain models now include a `confidence` field indicating the agent's certainty in the generated results:

```python
class SpecContract(BaseModel):
    """Hardware contract specification."""
    
    name: str
    description: str
    clock_mhz_target: float
    throughput_samples_per_cycle: int
    input_format: Dict[str, Any]
    output_format: Dict[str, Any]
    resource_budget: Dict[str, Any]
    verification_config: Dict[str, Any]
    confidence: float = Field(default=90.0, ge=0, le=100, description="Confidence level (0-100%)")
```

### Confidence Fields

All stage output models include confidence levels:

- **SpecContract**: `confidence: float` - Confidence in hardware specification
- **QuantConfig**: `confidence: float` - Confidence in quantization decisions
- **MicroArchConfig**: `confidence: float` - Confidence in microarchitecture design
- **RTLConfig**: `confidence: float` - Confidence in RTL generation quality
- **LintResults**: `confidence: float` - Confidence in static analysis results
- **VerifyResults**: `confidence: float` - Confidence in verification results
- **SynthResults**: `confidence: float` - Confidence in synthesis results
- **EvaluateResults**: `confidence: float` - Confidence in evaluation summary

### Pipeline Confidence Logic

The pipeline automatically checks confidence levels and only invokes the feedback agent when:

1. Stage confidence < 80% (configurable threshold)
2. Stage execution fails

```python
def _get_stage_confidence(self, stage_name: str) -> Optional[float]:
    """Extract confidence level from stage result."""
    if stage_name not in self.results:
        return None
    
    result = self.results[stage_name]
    if hasattr(result, 'confidence'):
        return result.confidence
    elif isinstance(result, dict) and 'confidence' in result:
        return result['confidence']
    
    return None
```

### SpecContract
Hardware specification contract from algorithm analysis.

```python
class SpecContract(BaseModel):
    """Hardware contract derived from algorithm bundle."""

    name: str
    description: str
    clock_mhz_target: float
    throughput_samples_per_cycle: float
    input_format: Dict[str, Any]
    output_format: Dict[str, Any]
    resource_budget: Dict[str, Any]
    verification_config: Dict[str, Any]
```

### QuantConfig
Fixed-point quantization configuration.

```python
class QuantConfig(BaseModel):
    """Fixed-point arithmetic configuration."""

    fixed_point_config: Dict[str, Any]
    error_metrics: Dict[str, Any]
    quantized_coefficients: List[float]
    fxp_model_path: str
```

### RTLConfig
RTL generation results and metadata.

```python
class RTLConfig(BaseModel):
    """RTL generation artifacts and metadata."""

    rtl_files: List[str]
    params_file: str
    top_module: str
    lint_passed: bool
    estimated_resources: Dict[str, Any]
```

## Tool Integration

### Function Tools

Tools callable by AI agents during execution.

#### File Operations
```python
def read_source(workspace_token: str, path: str) -> Dict[str, Any]:
    """Read file from virtual workspace."""

def write_artifact(workspace_token: str, path: str, content: str) -> Dict[str, Any]:
    """Write file to virtual workspace."""

def ingest_from_bundle(raw_bundle: str, normalize: bool = False) -> Dict[str, Any]:
    """Parse algorithm bundle into workspace."""
```

#### RTL Tools
```python
def run_simulation(rtl_files: List[str], test_vectors: List[Dict], simulator: str = "iverilog") -> Dict[str, Any]:
    """Execute RTL simulation."""

def run_static_checks(rtl_files: List[str]) -> Dict[str, Any]:
    """Run linting and style analysis."""
```

#### Synthesis Tools
```python
def submit_synth_job(repo: str, ref: str, top: str, rtl_glob: str, toolchain: str, constraint_file: Optional[str] = None) -> Dict[str, Any]:
    """Submit FPGA synthesis job."""

def fetch_synth_results(repo: str, run_id: str) -> Dict[str, Any]:
    """Retrieve synthesis results."""
```

## Bundle Utilities

### Bundle Creation

```python
from ardagen.bundle_utils import create_bundle

# Create bundle from single file
bundle = create_bundle("algorithm.py")

# Create bundle from directory
bundle = create_bundle("my_project/")

# Save to file
bundle = create_bundle("algorithm.py", "algorithm_bundle.txt")
```

### BundleCreator Class

```python
class BundleCreator:
    """Creates ARDA bundles from Python files and directories."""

    def create_bundle_from_file(self, file_path: str, output_path: Optional[str] = None) -> str:
        """Convert single Python file to bundle format."""

    def create_bundle_from_directory(self, dir_path: str, output_path: Optional[str] = None) -> str:
        """Scan directory for Python files and create bundle."""

    def _extract_algorithm_metadata(self, content: str) -> List[str]:
        """Extract algorithm metadata from file content."""
```

## Observability

### ObservabilityManager

Central observability and logging management.

```python
class ObservabilityManager:
    """Manages observability events and logging."""

    def stage_started(self, stage: str, metadata: Dict[str, Any]) -> None:
        """Record stage start event."""

    def stage_completed(self, stage: str, output: Any) -> None:
        """Record stage completion event."""

    def tool_invoked(self, stage: str, tool_name: str, metadata: Dict[str, Any]) -> None:
        """Record tool invocation event."""

    def feedback_generated(self, decision: FeedbackDecision) -> None:
        """Record feedback decision event."""
```

### Event Types

```python
@dataclass
class StageEvent:
    """Stage lifecycle event."""

    stage: str
    event_type: str  # "started", "completed", "failed"
    timestamp: datetime
    metadata: Dict[str, Any]

@dataclass
class ToolEvent:
    """Tool invocation event."""

    stage: str
    tool_name: str
    timestamp: datetime
    metadata: Dict[str, Any]
```

## CLI Interface

### Command Line Usage

```bash
# Basic usage
arda <bundle_file> [options]

# Create bundle from files
arda --create-bundle source.py bundle.txt

# Run with verbose output
arda algorithm.txt --verbose

# Extract RTL files
arda algorithm.txt --extract-rtl rtl_output/
```

### CLI Options

| Option | Description | Default |
|--------|-------------|---------|
| `--verbose` | Enable verbose output | `False` |
| `--agent-runner` | Agent runner backend | `"auto"` |
| `--create-bundle` | Create bundle from files | `None` |
| `--extract-rtl` | Extract RTL to directory | `None` |
| `--output` | Save results to JSON | `None` |
| `--synthesis-backend` | Synthesis tool backend | `"auto"` |

## Error Handling

### Exception Hierarchy

```python
class ARDAError(Exception):
    """Base exception for ARDA errors."""

class StageExecutionError(ARDAError):
    """Error during stage execution."""

class ToolExecutionError(ARDAError):
    """Error during tool execution."""

class BundleParseError(ARDAError):
    """Error parsing algorithm bundle."""

class SynthesisError(ARDAError):
    """Error during FPGA synthesis."""
```

### Error Recovery

Stages can define custom error recovery:

```python
class MyStage(Stage):
    async def run(self, context: StageContext) -> MyOutput:
        try:
            # Stage logic here
            return MyOutput(result="success")
        except Exception as e:
            # Custom error recovery
            if self._can_retry(e):
                raise StageRetryError("Retryable error")
            else:
                raise StageExecutionError(self.name, e)
```

## Configuration

### Agent Configuration File

Location: `agent_configs.json`

```json
{
  "agents": {
    "stage_name": {
      "name": "Agent Name",
      "instructions": "Agent instructions...",
      "model": "gpt-4",
      "tools": [...],
      "output_schema": {...}
    }
  },
  "function_tools": {
    "tool_name": {
      "schema": {
        "name": "tool_name",
        "description": "Tool description",
        "parameters": {...}
      }
    }
  }
}
```

### Model Configuration

Location: `arda/model_config.py`

```python
AGENT_MODELS: Dict[str, str] = {
    'spec': 'gpt-4',
    'quant': 'gpt-3.5-turbo',
    'rtl': 'gpt-4',
    # ... other stages
}
```

## Integration Examples

### Custom Agent Integration

```python
from ardagen.agents.registry import AgentRegistry

def my_custom_agent(context: Mapping[str, Any]) -> CustomOutput:
    """Custom agent implementation."""
    # Process context and return structured output
    return CustomOutput(result="processed")

# Register the agent
registry = AgentRegistry()
registry.register_stage_agent("my_stage", my_custom_agent)
```

### External Tool Integration

```python
# In arda/tools/custom_tools.py
def run_custom_eda_tool(files: List[str], options: Dict) -> Dict[str, Any]:
    """Integrate with external EDA tool."""
    # Tool implementation
    return {"status": "completed", "results": {...}}

# Register in agent_configs.json
{
  "function_tools": {
    "run_custom_eda_tool": {
      "schema": {
        "name": "run_custom_eda_tool",
        "parameters": {
          "type": "object",
          "properties": {
            "files": {"type": "array", "items": {"type": "string"}},
            "options": {"type": "object"}
          },
          "required": ["files"],
          "additionalProperties": false
        }
      }
    }
  }
}
```

## Performance Considerations

### Memory Management

```python
# Process large datasets in chunks
async def process_large_rtl_generation(context: StageContext):
    rtl_files = context.results["rtl"].rtl_files

    # Process in chunks to avoid memory issues
    for i in range(0, len(rtl_files), 10):
        chunk = rtl_files[i:i + 10]
        await process_rtl_chunk(chunk)
```

### Async Processing

```python
# Use async for I/O bound operations
async def async_synthesis(synthesis_job: Dict) -> Dict:
    """Async synthesis execution."""
    # Use asyncio for concurrent synthesis jobs
    tasks = [run_synthesis_job(job) for job in jobs]
    results = await asyncio.gather(*tasks)
    return {"results": results}
```

## Testing

### Unit Testing Stages

```python
import pytest
from ardagen.core.stages.rtl_stage import RTLStage

@pytest.mark.asyncio
async def test_rtl_stage():
    """Test RTL stage execution."""
    stage = RTLStage()

    # Setup mock context
    context = StageContext(
        run_inputs={"bundle": "test_bundle"},
        results={
            "spec": SpecContract(...),
            "quant": QuantConfig(...),
            "microarch": MicroArchConfig(...)
        }
    )

    # Execute stage
    result = await stage.run(context)

    # Assertions
    assert isinstance(result, RTLConfig)
    assert len(result.rtl_files) > 0
    assert result.lint_passed is True
```

### Integration Testing

```python
@pytest.mark.integration
async def test_full_pipeline():
    """Test complete pipeline execution."""
    from ardagen.pipeline import Pipeline

    pipeline = Pipeline()
    bundle = create_test_bundle()

    # Run complete pipeline
    result = await pipeline.run(bundle)

    # Assertions
    assert result["success"] is True
    assert "rtl" in result["results"]
    assert "synth" in result["results"]
```

## Debugging

### Debug Logging

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('arda')

# Or enable specific loggers
logging.getLogger('arda.agents').setLevel(logging.DEBUG)
logging.getLogger('arda.core').setLevel(logging.DEBUG)
```

### Pipeline Debugging

```python
# Debug specific stage
from arda.core.orchestrator import PipelineOrchestrator

orchestrator = PipelineOrchestrator(stages=[RTLStage()])
async for stage_name, output in orchestrator.run_iter(context):
    print(f"Stage {stage_name}: {output}")
```

### Agent Debugging

```python
# Enable OpenAI agent debugging
import os
os.environ['DEBUG'] = 'openai-agents:*'

# Run with agent debugging
arda algorithm.txt --verbose --agent-runner openai
```

## Contributing

See [DEVELOPER_GUIDE.md](developer_guide.md) for detailed contribution guidelines.

## Support

For API usage questions:
- [GitHub Issues](https://github.com/WestonVoglesonger/ARDA/issues)
- [GitHub Discussions](https://github.com/WestonVoglesonger/ARDA/discussions)
- [Developer Guide](developer_guide.md)

## Version History

- **v1.0.0**: Initial release with complete AI-powered RTL generation
- **v0.9.0**: OpenAI Responses API integration and bundle utilities
- **v0.8.0**: Modular pipeline architecture and observability
- **v0.7.0**: Initial AI agent integration
- **v0.6.0**: Basic RTL generation from algorithms

---

*This API documentation is automatically maintained and reflects the current ARDA codebase. For the latest updates, check the GitHub repository.*
