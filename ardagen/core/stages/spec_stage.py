"""
Spec stage implementation for the ARDA orchestrator.
"""

from __future__ import annotations

from typing import Any, Dict

from .base import Stage, StageContext
from ...domain import SpecContract


class SpecStage(Stage):
    """Derive hardware contract from the input algorithm bundle."""

    name = "spec"
    dependencies: tuple[str, ...] = ()
    output_model = SpecContract

    def gather_inputs(self, context: StageContext) -> Dict[str, Any]:
        inputs = super().gather_inputs(context)
        bundle = context.run_inputs.get("bundle")
        if bundle is None:
            raise KeyError("SpecStage requires 'bundle' in run inputs.")
        inputs["bundle"] = bundle
        return inputs
