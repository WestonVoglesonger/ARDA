"""
Strategy interfaces for executing ARDA stages.
"""

from __future__ import annotations

from typing import Any, Mapping, Protocol, Dict

if False:  # pragma: no cover - for type checkers only
    from .stages import Stage


class AgentStrategy(Protocol):
    """Strategy capable of executing a stage and returning structured output."""

    async def run(
        self,
        stage: "Stage",
        stage_inputs: Dict[str, Any],
        run_inputs: Mapping[str, Any],
    ) -> Any:
        ...
