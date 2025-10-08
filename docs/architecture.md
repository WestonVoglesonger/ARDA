# ARDA (Automated RTL Design with Agents) Pipeline Revamp Strategy

## Purpose
- Rebrand ALG2SV as **ARDA – Automated RTL Design with Agents** while establishing a maintainable, testable architecture for algorithm-to-SystemVerilog conversion.
- Align software pipeline stages with real FPGA design practices.
- Enable rigorous verification, synthesis, and evaluation with clear quality gates.

## Related Documents
- [ADR 0001: ARDA Pipeline Refactor Strategy](adr/0001-arda-architecture-refactor.md) — formal decision record for the staged migration.

## Current Progress
- Domain models now live under `alg2sv/domain`, with legacy imports re-exported for compatibility.
- A stage-oriented orchestrator and reusable stage modules (`alg2sv/core`) drive the new workflow skeleton with explicit static-check and verification gates.
- Observability manager now emits typed lifecycle/tool events via `alg2sv/observability/{manager,events}.py`.
- The simplified pipeline consumes the orchestrator through a pluggable `PipelineAgentRunner`, backed by an agent registry and tool adapters for linting, simulation, synthesis, and reporting.
- A transitional `arda` namespace shim exposes the package under its future name.

## Audience
- Core ARDA maintainers implementing the refactor.
- Contributors designing agent behaviours, tools, and integrations.
- Stakeholders evaluating roadmap progress toward hardware-accurate results.

## Naming Rationale
**Why ARDA?** The new name captures our agent-driven automation ethos (“Automated RTL Design”) and offers a memorable, lore-inspired identity. Documentation, CLI messaging, and contributor guidance will refer to the system as ARDA going forward, while Python packages will migrate in a later refactor milestone.

## Objectives
- Replace the monolithic `ALG2SVPipeline` with a modular orchestrator and typed stage graph.
- Separate agent prompt/config definitions from execution strategy and tooling adapters.
- Promote reproducible verification and synthesis flows with first-class reporting and observability.
- Provide a migration path that keeps current functionality working during the transition.

## Architectural Overview

### Domain-Driven Package Layout
```
alg2sv/                # Will migrate to arda/ once refactor stabilizes
  core/
    orchestrator.py         # Pipeline engine, stage scheduler, retry policy
    stages/
      base.py               # Stage interface & lifecycle hooks
      spec_stage.py
      quant_stage.py
      microarch_stage.py
      rtl_stage.py
      static_checks_stage.py
      verification_stage.py
      synthesis_stage.py
      evaluation_stage.py
    feedback.py             # Feedback policy, decision types
  domain/
    contracts.py            # SpecContract, device constraints
    quantization.py
    architecture.py
    rtl_artifacts.py
    verification.py
    synthesis.py
    evaluation.py
  agents/
    registry.py             # Agent prompt definitions & tool wiring
    strategies.py           # Interfaces for AgentRunner, Scripted, Manual strategies
  tools/
    workspace/
    lint/
    simulation/
    synthesis/
    reporting/
  observability/
    manager.py              # Centralized logging, tracing, metrics facade
    events.py               # Typed events & payload schemas
```

### Core Concepts
- **Stage**: Encapsulates one FPGA design activity. Declares required inputs, produced artifacts, and quality gates. Implements `prepare()`, `execute()`, `validate()`, `report()`.
- **Orchestrator**: Builds the stage dependency DAG, manages retries, hands feedback to stages, and records observability events.
- **Agent Strategy**: Abstracts how a stage obtains its result (remote LLM agent, local heuristic, scripted fallback). Enables deterministic tests and future non-agent implementations.
- **Artifact Store**: Wraps the existing workspace manager to version inputs/outputs, provide diffable bundles, and expose generated files for downstream tooling.

## Stage Flow Aligned to FPGA Design

| Stage                 | Responsibilities                                                                 | Inputs                              | Outputs / Checks                                          |
|-----------------------|----------------------------------------------------------------------------------|-------------------------------------|-----------------------------------------------------------|
| `SpecStage`           | Parse algorithm bundle, derive I/O, timing targets, device selection.            | Workspace bundle                    | `SpecContract`, device intent                             |
| `QuantStage`          | Produce fixed-point formats, error analysis, coefficient files.                  | `SpecContract`                      | `QuantConfig`, quantized coefficients, error bounds       |
| `MicroArchStage`      | Establish pipeline structure, buffering, parallelism strategy.                   | `SpecContract`, `QuantConfig`       | `MicroArchConfig`, resource expectations                  |
| `RTLGeneationStage`   | Emit synthesizable SystemVerilog, parameter packages, register maps.             | Prior configs                       | `RTLArtifacts`, lint manifest                             |
| `StaticChecksStage`   | Run lint, style, structural analysis before costly runs.                         | `RTLArtifacts`                      | `LintResults`, gating thresholds                          |
| `VerificationStage`   | Generate/consume stimuli, run sims, collect coverage & scoreboard metrics.       | `RTLArtifacts`, vectors, spec       | `VerifyResults`, coverage, waveform links                 |
| `SynthesisStage`      | Dispatch to Vivado/Yosys/SymbiFlow backends with consistent reporting.           | `RTLArtifacts`, constraints         | `SynthesisReport`, timing/power/utilization gates         |
| `EvaluationStage`     | Aggregate metrics vs. requirements, produce scorecard & feedback suggestions.    | All stage reports                   | `EvaluationSummary`, recommended actions                  |

Feedback policies decide when to rerun `Quant`, `MicroArch`, `RTL`, or `Synthesis` based on gate failures.

## Observability and Reporting
- Define typed events: `StageStarted`, `StageCompleted`, `ToolInvoked`, `VerificationMismatch`, `SynthesisFailure`.
- `observability.manager` publishes to existing trace/performance/error tools and optional JSONL logs.
- Stage reports emit machine-readable files (JSON) stored beside generated RTL for post-run dashboards.
- CLI surfaces a concise run summary (targets vs achieved, gate status) and optional `--report-dir` export.

## Tooling Strategy
- Refactor current helpers (`run_rtl_simulation`, `run_vivado_synthesis`, Verilator wrappers) into adapters implementing clear interfaces used by both agents and orchestrator-driven stages.
- Introduce mock adapters for unit tests to avoid external tool dependencies.
- Align test coverage so each stage has dedicated pytest modules exercising success, failure, and retry paths.

## Migration Plan
1. **Document & Contracts**: Move Pydantic models into `alg2sv/domain`, introduce dataclasses (with Pydantic as validation layer).
2. **Core Skeleton**: Implement `Stage` base class and orchestrator. Wrap current pipeline logic to run through orchestrator with minimal behavioural changes.
3. **Tool Adapters**: Extract workspace, lint, simulation, synthesis utilities into `tools/` modules while maintaining API parity.
4. **Verification & Static Checks**: Add new stages for lint/simulation gating; integrate into orchestrator while leaving legacy pipeline available behind a feature flag.
5. **Observability Integration**: Replace ad-hoc logging with structured events; ensure CLI and tests consume new summaries.
6. **Simplified Mode**: Reimplement simplified pipeline as a preset configuration on the new orchestrator (replacing `pipeline.py`).
7. **Retire Legacy Code**: After parity testing, remove `pipeline.py` monolith, consolidate agent orchestration, and update CLI/tests accordingly.
8. **Package Rename**: Rename the top-level `alg2sv` package to `arda` once API consumers are ready, providing deprecation shims where necessary.

## Risks & Mitigations
- **Large Refactor Scope**: Deliver in feature-flagged increments; maintain old pipeline until new path is validated.
- **Tool Availability**: Mock adapters and dependency injection allow running in environments without Vivado/Verilator.
- **Agent Behaviour Variance**: Agent strategy abstraction enables deterministic scripted modes for tests.
- **Team Adoption**: Provide migration guides, code examples, and pair sessions to familiarize contributors with stage APIs.

## Immediate Next Tasks
1. Publish companion ADR describing staged refactor approach and gain stakeholder approval.
2. Extract `SpecContract`, `QuantConfig`, etc. into `alg2sv/domain/` module with unit tests.
3. Prototype `core/stages/base.py` and `core/orchestrator.py`, wiring the current spec → quant → microarch flow as proof of concept.
4. Set up `observability/manager.py` with event definitions and integrate into the prototype stages.
5. Update documentation (README + CLI help) to reference the forthcoming staged pipeline and reporting outputs.
