"""
Pipeline agent runner backed by the OpenAI Agents SDK.
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any, Dict, Mapping, Optional

from ..domain import FeedbackDecision
from ..model_config import get_model_params_for_agent
from ..runtime.agent_runner import PipelineAgentRunner
from .config_loader import (
    get_agent_config,
    get_function_tool_schema,
    load_agent_configs,
)
from .tools import FUNCTION_MAP, call_tool
from .registry import create_default_registry


class OpenAIAgentRunner(PipelineAgentRunner):
    """
    Executes ARDA stages using OpenAI's Agents SDK.

    Summary:
    - Loads agent instructions, tools, and schemas from `agent_configs.json`.
    - Executes each stage via `client.responses.create`, handling tool calls and
      returning structured JSON outputs.
    - Falls back to the deterministic `DefaultAgentRunner` for stages without an
      associated agent definition (e.g., static checks or evaluation).

    Additional context:
    - Function tools are implemented in `alg2sv/agents/tools.py`.
    - Models per stage are controlled by `alg2sv/model_config.py`.
    """

    _STAGE_TO_AGENT = {
        "spec": "spec_agent",
        "quant": "quant_agent",
        "microarch": "microarch_agent",
        "rtl": "rtl_agent",
        "verification": "verify_agent",
        "synth": "synth_agent",
    }

    _FEEDBACK_SCHEMA = {
        "name": "feedback_decision",
        "schema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "continue",
                        "abort",
                        "retry_spec",
                        "retry_quant",
                        "retry_microarch",
                        "retry_rtl",
                        "retry_verification",
                        "retry_synth",
                        "tune_microarch",
                    ],
                },
                "target_stage": {"type": ["string", "null"]},
                "guidance": {"type": ["string", "null"]},
                "notes": {"type": ["array", "null"], "items": {"type": "string"}},
            },
            "required": ["action"],
            "additionalProperties": False,
        },
    }

    def __init__(
        self,
        *,
        client: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None,
        fallback_runner: Optional[PipelineAgentRunner] = None,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "The OpenAI Python SDK is required for OpenAIAgentRunner. "
                "Install it with `pip install openai`."
            ) from exc

        if client is None:
            if not os.getenv("OPENAI_API_KEY"):
                raise RuntimeError(
                    "OPENAI_API_KEY environment variable is required to use OpenAIAgentRunner."
                )
            client = OpenAI()

        self._client = client
        self._configs = config or load_agent_configs()
        self._fallback = fallback_runner or DefaultAgentRunnerFallback()

    async def run_stage(self, stage: str, context: Mapping[str, Any]) -> Any:
        agent_key = self._STAGE_TO_AGENT.get(stage)
        if not agent_key or agent_key not in self._configs.get("agents", {}):
            # Use fallback deterministic implementation
            return await self._fallback.run_stage(stage, context)

        return await asyncio.to_thread(
            self._run_agent_sync,
            stage,
            agent_key,
            dict(context),
        )

    async def run_feedback(self, context: Mapping[str, Any]) -> FeedbackDecision:
        # For now, re-use the OpenAI runner with inline instructions.
        return await asyncio.to_thread(self._run_feedback_sync, dict(context))

    # --------------------------------------------------------------------- #
    # Blocking helpers executed inside `asyncio.to_thread`
    # --------------------------------------------------------------------- #

    def _run_agent_sync(self, stage: str, agent_key: str, context: Dict[str, Any]) -> Any:
        agent_cfg = get_agent_config(agent_key)
        model_params = get_model_params_for_agent(self._model_key_for_stage(stage))
        model = model_params.pop("model")

        tools, tool_map, interpreter_requested = self._build_tool_definitions(agent_cfg)
        instructions = agent_cfg["instructions"]
        if interpreter_requested:
            instructions += (
                "\n\nIMPORTANT: The code interpreter tool is not available in this runtime. "
                "Complete the task using reasoning only and do not request the code interpreter."
            )

        json_schema = self._build_json_schema(agent_key, agent_cfg)
        messages = self._build_messages(instructions, stage, context, json_schema)
        response_format = self._build_response_format(agent_key, json_schema)

        response = self._create_response(
            model=model,
            messages=messages,
            tools=tools if tools else None,
            response_format=response_format,
            model_params=model_params,
        )

        response = self._handle_required_actions(response, context, tool_map)
        output_text = self._extract_output_text(response, stage)
        return json.loads(output_text)

    def _run_feedback_sync(self, context: Dict[str, Any]) -> FeedbackDecision:
        model_params = get_model_params_for_agent("feedback")
        model = model_params.pop("model")
        json_schema = self._FEEDBACK_SCHEMA["schema"]
        messages = self._build_messages(
            "You are the ARDA feedback agent. Review stage results and suggest whether to continue, retry a specific stage, tune microarchitecture, or abort.",
            "feedback",
            context,
            json_schema,
        )
        response = self._create_response(
            model=model,
            messages=messages,
            response_format={"type": "json_schema", "json_schema": self._FEEDBACK_SCHEMA},
            model_params=model_params,
        )
        output_text = self._extract_output_text(response, "feedback")
        data = json.loads(output_text)
        return FeedbackDecision(**data)

    # --------------------------------------------------------------------- #
    # Response / Tool Handling
    # --------------------------------------------------------------------- #

    def _handle_required_actions(
        self,
        response: Any,
        context: Dict[str, Any],
        tool_map: Dict[str, Any],
    ) -> Any:
        """
        Execute tool calls requested by the Responses API until completion.
        """
        while getattr(response, "status", None) == "requires_action":
            required = getattr(response, "required_action", None)
            if not required or not hasattr(required, "submit_tool_outputs"):
                break
            tool_calls = getattr(required.submit_tool_outputs, "tool_calls", None)
            if not tool_calls:
                break
            tool_outputs = []
            for call in tool_calls:
                tool_name = call.function.name
                args = json.loads(call.function.arguments or "{}")
                try:
                    output = self._invoke_tool(tool_name, args, context, tool_map)
                    tool_outputs.append({"tool_call_id": call.id, "output": output})
                except Exception as e:
                    # Add detailed error information for debugging
                    error_msg = f"Tool '{tool_name}' failed: {str(e)}"
                    if "NoneType" in str(e) and "iterable" in str(e):
                        error_msg += f"\nTool arguments: {args}"
                        error_msg += f"\nContext keys: {list(context.keys())}"
                        if "results" in context:
                            error_msg += f"\nResults keys: {list(context['results'].keys())}"
                    tool_outputs.append({"tool_call_id": call.id, "output": f"ERROR: {error_msg}"})

            response = self._client.responses.submit_tool_outputs(
                response_id=response.id,
                tool_outputs=tool_outputs,
            )
        return response

    def _invoke_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        context: Mapping[str, Any],
        tool_map: Dict[str, Any],
    ) -> str:
        if tool_name not in tool_map:
            raise KeyError(f"Tool '{tool_name}' not registered for agent runner.")
        result = tool_map[tool_name](**arguments)
        # Emit observability events when available.
        observability = context.get("observability")
        stage = context.get("stage", tool_name)
        if observability:
            metadata = dict(arguments)
            try:
                observability.tool_invoked(stage, tool_name, metadata)
            except Exception:
                pass
        if isinstance(result, str):
            return result
        return json.dumps(result)

    def _extract_output_text(self, response: Any, stage: str) -> str:
        text = getattr(response, "output_text", None)
        if text:
            return text
        output = getattr(response, "output", None)
        if output:
            for item in output:
                for content in getattr(item, "content", []):
                    text_part = getattr(content, "text", None)
                    if text_part:
                        return text_part
                    if hasattr(content, "output_text"):
                        return content.output_text
        raise RuntimeError(f"Agent response for stage '{stage}' did not include textual output.")

    # --------------------------------------------------------------------- #
    # Message / Tool construction helpers
    # --------------------------------------------------------------------- #

    def _build_messages(
        self,
        instructions: str,
        stage: str,
        context: Dict[str, Any],
        json_schema: Optional[Dict[str, Any]] = None,
    ) -> Any:
        import copy

        sanitized_context = self._sanitize_context(copy.deepcopy(context))
        
        # Add debugging info for RTL stage
        if stage == "rtl":
            quant_data = sanitized_context.get("results", {}).get("quant", {})
            coeffs = quant_data.get("quantized_coefficients")
            if coeffs is None:
                sanitized_context["debug_info"] = {
                    "quant_data_keys": list(quant_data.keys()),
                    "quant_data": quant_data,
                    "warning": "quantized_coefficients is None!"
                }
            else:
                sanitized_context["debug_info"] = {
                    "quantized_coefficients_type": type(coeffs).__name__,
                    "quantized_coefficients_length": len(coeffs) if hasattr(coeffs, '__len__') else 'no length',
                    "first_coeff": coeffs[0] if coeffs else None
                }
        
        schema_instructions = ""
        if json_schema:
            schema_instructions = (
                "\n\nReturn ONLY a JSON object matching this schema:\n```json\n"
                f"{json.dumps(json_schema, indent=2)}\n```"
                "\nDo not include code fences or additional commentary."
            )
        user_content = (
            f"You are executing the '{stage}' stage of the ARDA pipeline. "
            "Use the provided context to produce the required JSON output.\n\n"
            "Context JSON:\n```json\n"
            f"{json.dumps(sanitized_context, indent=2)}\n```"
            f"{schema_instructions}"
        )
        return [
            {"role": "system", "content": instructions},
            {"role": "user", "content": user_content},
        ]

    def _build_tool_definitions(self, agent_cfg: Dict[str, Any]):
        tool_defs = []
        tool_functions = {}
        interpreter_requested = False
        for tool in agent_cfg.get("tools", []):
            if tool["type"] == "function":
                schema = get_function_tool_schema(tool["name"])
                # Flatten structure for Responses API
                function_def = {
                    "name": schema["name"],
                    "parameters": schema.get("parameters", {}),
                }
                if "description" in schema:
                    function_def["description"] = schema["description"]
                if "strict" in schema:
                    function_def["strict"] = schema["strict"]
                tool_defs.append({"type": "function", "name": function_def["name"], "function": function_def})
                tool_functions[tool["name"]] = FUNCTION_MAP[tool["name"]]
            elif tool["type"] == "code_interpreter":
                interpreter_requested = True
                # Code interpreter sessions are not yet supported; skip registering for now.
                continue
        return tool_defs, tool_functions, interpreter_requested

    def _build_json_schema(self, agent_key: str, agent_cfg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        output_schema = agent_cfg.get("output_schema")
        if not output_schema:
            return None
        return {
            "type": "object",
            "properties": output_schema,
            "required": list(output_schema.keys()),
            "additionalProperties": False,
        }

    def _build_response_format(self, agent_key: str, json_schema: Optional[Dict[str, Any]]):
        if not json_schema:
            return None
        return {
            "type": "json_schema",
            "json_schema": {
                "name": f"{agent_key}_response",
                "schema": json_schema,
            },
        }

    def _sanitize_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        context.pop("observability", None)
        return context

    def _model_key_for_stage(self, stage: str) -> str:
        return {
            "verification": "verification",
            "synth": "synth",
            "spec": "spec",
            "quant": "quant",
            "microarch": "microarch",
            "rtl": "rtl",
            "evaluate": "evaluate",
            "static_checks": "static_checks",
        }.get(stage, stage)

    def _create_response(
        self,
        *,
        model: str,
        messages: Any,
        tools: Optional[Any] = None,
        response_format: Optional[Dict[str, Any]] = None,
        model_params: Optional[Dict[str, Any]] = None,
    ) -> Any:
        kwargs = dict(model=model, input=messages)
        if tools:
            kwargs["tools"] = tools
        if model_params:
            kwargs.update(model_params)

        if response_format:
            kwargs["response_format"] = response_format
            try:
                return self._client.responses.create(**kwargs)
            except TypeError as exc:
                if "response_format" not in str(exc):
                    raise
                # Remove unsupported argument and retry with prompt-based enforcement.
                kwargs.pop("response_format", None)
                return self._client.responses.create(**kwargs)
        return self._client.responses.create(**kwargs)


class DefaultAgentRunnerFallback(PipelineAgentRunner):
    """
    Thin wrapper delegating to the existing deterministic runner without
    introducing a circular import.
    """

    def __init__(self) -> None:
        from ..runtime.agent_runner import DefaultAgentRunner

        self._delegate = DefaultAgentRunner()

    async def run_stage(self, stage: str, context: Mapping[str, Any]) -> Any:
        return await self._delegate.run_stage(stage, context)

    async def run_feedback(self, context: Mapping[str, Any]) -> FeedbackDecision:
        return await self._delegate.run_feedback(context)


__all__ = ["OpenAIAgentRunner"]
