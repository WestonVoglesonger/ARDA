"""
Utilities for loading agent configuration metadata used by the OpenAI runner.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional


@lru_cache(maxsize=1)
def load_agent_configs(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load agent configuration JSON describing prompts, tools, and schemas.

    Args:
        config_path: Optional explicit path to the configuration file.  When not
            provided, defaults to `<project_root>/agent_configs.json`.

    Returns:
        Parsed configuration dictionary.
    """
    if config_path is None:
        # Project root is two levels up from this file (alg2sv/agents/)
        config_path = Path(__file__).resolve().parents[2] / "agent_configs.json"

    with config_path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def get_agent_config(agent_name: str, config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Convenience helper to retrieve a single agent definition."""
    configs = load_agent_configs(config_path)
    agents = configs.get("agents", {})
    if agent_name not in agents:
        raise KeyError(f"Agent '{agent_name}' not found in configuration.")
    return agents[agent_name]


def get_function_tool_schema(tool_name: str, config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Retrieve the schema definition for a named function tool."""
    configs = load_agent_configs(config_path)
    tools = configs.get("function_tools", {})
    if tool_name not in tools:
        raise KeyError(f"Function tool '{tool_name}' not defined in configuration.")
    return tools[tool_name]["schema"]


__all__ = ["load_agent_configs", "get_agent_config", "get_function_tool_schema"]
