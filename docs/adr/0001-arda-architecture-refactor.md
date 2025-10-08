# ADR 0001: ARDA Pipeline Refactor Strategy

## Status
- Proposed — pending team review and sign-off

## Context
- The existing `ALG2SVPipeline` implementation has grown into a monolithic orchestrator with tightly coupled agent definitions, tool wiring, and retry logic.
- Verification, linting, and synthesis steps are not first-class quality gates, making it difficult to enforce hardware-readiness or diagnose issues.
- Observability is fragmented; logs, traces, and metrics are emitted ad hoc without a consistent schema or aggregation point.
- Documentation and tooling branding no longer reflect the system’s identity after rebranding to **ARDA (Automated RTL Design with Agents)**.
- We need a foundation that supports deterministic testing, agent strategy experimentation, and future non-agent implementations while keeping the current behaviour available during transition.

## Decision
We will introduce a staged refactor that:

1. **Defines a Modular Core** — Create a `core` package housing a `PipelineOrchestrator`, typed `Stage` abstractions, and feedback policies. Stages expose lifecycle hooks (`prepare`, `execute`, `validate`, `report`) and declare explicit dependencies.
2. **Extracts Domain Models** — Move structured data contracts (spec, quantization, architecture, RTL artifacts, verification, synthesis, evaluation) into a dedicated `alg2sv/domain` module, retaining Pydantic validation but decoupling them from agent plumbing.
3. **Abstracts Agent Strategies** — Separate prompt/tool configuration from execution mechanics. Provide strategy interfaces to swap between OpenAI Agents, scripted mocks, or future deterministic generators.
4. **Modularizes Tool Adapters** — Relocate workspace, lint, simulation, and synthesis helpers under `alg2sv/tools` as reusable adapters consumed by both stages and agents.
5. **Centralizes Observability** — Introduce an `observability.manager` façade that publishes typed events (stage start/end, tool calls, verification mismatches, synthesis failures) to existing tracing/performance utilities and optional JSON exports.
6. **Supports Dual Pipelines During Transition** — Keep the legacy `ALG2SVPipeline` operational behind a compatibility flag while wiring the new orchestrator to parity tests. Once validated, retire the monolith and rename the Python package to `arda`.

## Consequences
### Positive
- Clear stage boundaries enable targeted testing, retries, and feedback-driven iteration.
- Shared domain models reduce duplication and improve schema consistency across agents and tools.
- Observability data becomes structured, enabling dashboards, alerting, and improved debugging.
- Branding aligns with ARDA, reducing confusion for new contributors and users.
- The architecture supports future expansion (e.g., alternate HDLs, formal verification) with minimal churn.

### Negative / Risks
- Refactor scope is sizable; without disciplined milestones the effort may stall.
- Temporary duplication (legacy vs. staged pipeline) increases maintenance overhead until cut-over.
- Moving models and tooling risks breaking existing imports if not carefully shimmed.
- Contributors must learn new abstractions (stages, orchestrator, strategies).

## Milestones & Tasks
1. Extract shared models into `alg2sv/domain` with import shims and unit tests.
2. Implement `core/stages/base.py` and `core/orchestrator.py`, migrating spec → quant → microarch flow first.
3. Wire lint + simulation stages as quality gates ahead of synthesis.
4. Introduce `observability/manager.py` and replace ad hoc logging with structured events.
5. Re-implement simplified pipeline as an orchestrator preset; remove duplicate code path.
6. Rename package namespace to `arda` after parity validation and update downstream tooling.

## Open Questions
- How will we version pipeline presets/configurations for different device families or workflows?
- Do we need an intermediate storage abstraction (e.g., artifact manifests) before renaming the package?
- What automation (CI jobs, pre-commit hooks) should enforce running lint/simulation stages?

## References
- `docs/architecture.md` — high-level vision and migration roadmap.
- `README.md` — updated branding and user-facing documentation.
