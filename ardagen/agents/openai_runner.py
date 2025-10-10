# ardagen/agents/openai_runner.py
"""
Pipeline agent runner backed by the OpenAI Agents SDK.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Any, Dict, Mapping, Optional, Set

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
from .json_parser import ResponseJSONParser
from .response_handler import ResponseHandler


def _dbg(msg: str) -> None:
    if os.getenv("ARDA_DEBUG_EXTRACTION"):
        print(f"DEBUG[extractor]: {msg}")


class OpenAIAgentRunner(PipelineAgentRunner):
    """
    Executes ARDA stages using OpenAI's Agents SDK.

    Notes:
    - With json_schema response_format, the SDK may place JSON in:
        * response.output_parsed
        * output[*].content[*] like:
          {"type":"output_json","content":[{"type":"json_schema","parsed": {...}}]}
    - We first check explicit parsed fields, then block JSON, then a bounded deep scan
      that is hardened to avoid returning the SDK envelope and to return an object,
      not a list, for object-expected stages.
    """

    _STAGE_TO_AGENT = {
        "spec": "spec_agent",
        "quant": "quant_agent",
        "microarch": "microarch_agent",
        "rtl": "rtl_agent",
        "verification": "verify_agent",
        "synth": "synth_agent",
    }

    # -------- Stage key expectations (used by heuristics) -------- #
    _REQ_KEYS: Dict[str, Set[str]] = {
        "spec": {
            "name", "description", "clock_mhz_target", "throughput_samples_per_cycle",
            "input_format", "output_format", "resource_budget", "verification_config"
        },
        "quant": {
            "fixed_point_config", "error_metrics", "quantized_coefficients"
        },
        "microarch": {
            "pipeline_depth", "unroll_factor", "memory_config",
            "estimated_latency_cycles", "handshake_protocol"
        },
        "rtl": {  # flexible, varies by your agent schema
            "verilog", "files", "modules", "top_module", "constraints", "confidence"
        },
        "verification": {
            "testbench", "stimulus", "checkers", "metrics", "confidence"
        },
        "synth": {
            "tcl_script", "timing_constraints", "estimates", "confidence"
        },
        "feedback": {
            "action"  # FeedbackDecision requires 'action'
        },
    }

    # For stages that expect a dict, not a list
    _EXPECTS_OBJECT: Set[str] = {"spec", "quant", "microarch", "rtl", "verification", "synth", "feedback"}

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
        
        # Initialize helper modules
        self._json_parser = ResponseJSONParser(self._REQ_KEYS, self._EXPECTS_OBJECT)
        self._response_handler = ResponseHandler(client)

    async def run_stage(self, stage: str, context: Mapping[str, Any]) -> Any:
        agent_key = self._STAGE_TO_AGENT.get(stage)
        if not agent_key or agent_key not in self._configs.get("agents", {}):
            return await self._fallback.run_stage(stage, context)

        return await asyncio.to_thread(
            self._run_agent_sync,
            stage,
            agent_key,
            dict(context),
        )

    async def run_feedback(self, context: Mapping[str, Any]) -> FeedbackDecision:
        return await asyncio.to_thread(self._run_feedback_sync, dict(context))

    # --------------------------------------------------------------------- #
    # Blocking helpers executed inside `asyncio.to_thread`
    # --------------------------------------------------------------------- #

    def _run_agent_sync(self, stage: str, agent_key: str, context: Dict[str, Any]) -> Any:
        agent_cfg = get_agent_config(agent_key)
        model_params = get_model_params_for_agent(self._model_key_for_stage(stage))
        model = model_params.pop("model")

        tools, tool_map, tool_requirements, interpreter_requested = self._build_tool_definitions(agent_cfg)
        if os.getenv("ARDA_DEBUG_EXTRACTION") and tools:
            try:
                print(f"DEBUG: Tools for stage '{stage}': {json.dumps(tools, indent=2)}")
            except Exception:
                print(f"DEBUG: Tools for stage '{stage}': {tools}")
        instructions = agent_cfg["instructions"]
        if interpreter_requested:
            instructions += (
                "\n\nIMPORTANT: The code interpreter tool is not available in this runtime. "
                "Complete the task using reasoning only and do not request the code interpreter."
            )

        json_schema = self._build_json_schema(agent_key, agent_cfg)
        messages = self._build_messages(instructions, stage, context, json_schema)
        response_format = self._build_response_format(agent_key, json_schema, has_tools=bool(tools))

        # Debug: Log exactly what we're sending to OpenAI
        if os.getenv("ARDA_DEBUG_EXTRACTION") and tools:
            print(f"DEBUG: About to call OpenAI with tools:")
            for tool in tools:
                print(f"  Tool: {tool['name']}")
                if 'function' in tool:
                    func = tool['function']
                    print(f"    Has 'parameters'? {('parameters' in func)}")
                    if 'parameters' in func:
                        params = func['parameters']
                        print(f"    Parameters type: {type(params)}")
                        if isinstance(params, dict):
                            print(f"    Properties empty? {not params.get('properties')}")
                            print(f"    Required empty? {not params.get('required')}")
                            if params.get('properties'):
                                print(f"    Properties keys: {list(params['properties'].keys())}")
                            if params.get('required'):
                                print(f"    Required: {params['required']}")
        
        response = self._create_response(
            model=model,
            messages=messages,
            tools=tools if tools else None,
            response_format=response_format,
            model_params=model_params,
        )

        response = self._response_handler.handle_required_actions(response, context, tool_map, tool_requirements)
        
        # If the agent used tools but didn't return text/JSON, prompt for final response
        if tools and self._response_handler.response_is_empty(response):
            if os.getenv("ARDA_DEBUG_EXTRACTION"):
                print(f"DEBUG: Agent for stage '{stage}' completed tools but returned no output. Prompting for JSON response.")
            response = self._response_handler.prompt_for_final_response(response, stage, json_schema, model, model_params)
        
        # Extract JSON payload from response
        payload = self._json_parser.extract_response_payload(response, stage)
        _dbg(f"{stage}: extracted payload successfully")
        return payload

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
        payload = self._json_parser.extract_response_payload(response, "feedback")
        if not isinstance(payload, dict):
            raise RuntimeError("Feedback agent must return a JSON object.")
        return FeedbackDecision(**payload)

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

        if stage == "rtl":
            quant_data = sanitized_context.get("stage_inputs", {}).get("quant", {})
            coeffs = quant_data.get("quantized_coefficients")
            print(f"DEBUG: RTL stage - stage_inputs keys: {list(sanitized_context.get('stage_inputs', {}).keys())}")
            print(f"DEBUG: RTL stage - quant_data keys: {list(quant_data.keys())}")
            print(f"DEBUG: RTL stage - coeffs type: {type(coeffs)}, value: {coeffs}")
            if coeffs is None:
                sanitized_context["debug_info"] = {
                    "stage_inputs_keys": list(sanitized_context.get("stage_inputs", {}).keys()),
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
        tool_requirements: Dict[str, Dict[str, Any]] = {}
        interpreter_requested = False
        for tool in agent_cfg.get("tools", []):
            if tool["type"] == "function":
                tool_name = tool["name"]
                schema = get_function_tool_schema(tool_name)
                
                # Debug: Log the schema structure
                if os.getenv("ARDA_DEBUG_EXTRACTION"):
                    print(f"DEBUG: Schema for tool '{tool_name}':")
                    print(f"  Keys: {list(schema.keys())}")
                    if "parameters" in schema:
                        params = schema["parameters"]
                        print(f"  Parameters type: {type(params)}")
                        if isinstance(params, dict):
                            print(f"  Parameters keys: {list(params.keys())}")
                            print(f"  Properties: {params.get('properties', {}).keys() if 'properties' in params else 'MISSING'}")
                            print(f"  Required: {params.get('required', [])}")
                
                # Build function definition with full parameter schema
                parameters = schema.get("parameters", {})
                function_def = {
                    "name": schema["name"],
                    "parameters": parameters,
                }
                if "description" in schema:
                    function_def["description"] = schema["description"]
                # NOTE: Disabling strict mode - it appears to cause the Responses API to strip parameter schemas
                # if "strict" in schema:
                #     function_def["strict"] = schema["strict"]
                
                # Validate that parameters has required structure
                if isinstance(parameters, dict):
                    if "properties" not in parameters or not parameters["properties"]:
                        print(f"WARNING: Tool '{tool_name}' has empty or missing properties in parameters schema!")
                    if "required" not in parameters or not parameters["required"]:
                        print(f"WARNING: Tool '{tool_name}' has empty or missing required fields in parameters schema!")
                
                tool_defs.append({"type": "function", "name": function_def["name"], "function": function_def})
                tool_functions[tool_name] = FUNCTION_MAP[tool_name]
                
                required_fields = parameters.get("required", []) if isinstance(parameters, dict) else []
                tool_requirements[tool_name] = {
                    "required": list(required_fields) if isinstance(required_fields, (list, tuple)) else [],
                    "parameters": parameters,
                }
            elif tool["type"] == "code_interpreter":
                interpreter_requested = True
                continue
        return tool_defs, tool_functions, tool_requirements, interpreter_requested

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

    def _build_response_format(self, agent_key: str, json_schema: Optional[Dict[str, Any]], has_tools: bool = False):
        if not json_schema:
            return None
        # Don't use strict json_schema response format for agents that call tools
        # The Responses API has limitations with combining structured outputs and tool calls
        if has_tools:
            return None
        return {
            "type": "json_schema",
            "json_schema": {
                "name": f"{agent_key}_response",
                "schema": json_schema,
                "strict": True,
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
                kwargs.pop("response_format", None)
                return self._client.responses.create(**kwargs)
        return self._client.responses.create(**kwargs)

    # ------------------------------ #
    # Debug / instrumentation helpers
    # ------------------------------ #

    def _debug_dump_response(self, response: Any, stage: str, reason: str = "") -> None:
        if not os.getenv("ARDA_DUMP_OPENAI_RESPONSE"):
            return
        try:
            snapshot: Dict[str, Any] = {
                "type": type(response).__name__,
                "attrs": [],
                "status": getattr(response, "status", None),
            }
            for key in ("output", "output_text", "output_parsed", "message", "messages"):
                val = getattr(response, key, None)
                brief = self._brief(val)
                snapshot["attrs"].append((key, brief))
            if hasattr(response, "model_dump"):
                snapshot["model_dump"] = self._brief(response.model_dump())
            elif hasattr(response, "dict"):
                snapshot["dict"] = self._brief(response.dict())
            dump_dir = os.getenv("ARDA_DUMP_DIR") or "/tmp"
            ts = int(time.time())
            path = os.path.join(dump_dir, f"arda_openai_{stage}_{reason}_{ts}.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(snapshot, f, indent=2, ensure_ascii=False)
            print(f"DEBUG: dumped OpenAI response snapshot to {path}")
        except Exception:
            pass

    @staticmethod
    def _brief(obj: Any, max_len: int = 1000) -> Any:
        try:
            if obj is None:
                return None
            if isinstance(obj, (str, int, float, bool)):
                s = str(obj)
                return s if len(s) <= max_len else s[:max_len] + "…"
            if isinstance(obj, (list, tuple)):
                return [OpenAIAgentRunner._brief(x, max_len=max_len // 2) for x in obj[:20]]
            if isinstance(obj, dict):
                out = {}
                for i, (k, v) in enumerate(list(obj.items())[:20]):
                    out[str(k)] = OpenAIAgentRunner._brief(v, max_len=max_len // 2)
                return out
            text = getattr(obj, "text", None)
            if text and hasattr(text, "value"):
                return {"text.value": str(text.value)[:max_len]}
            value = getattr(obj, "value", None)
            if isinstance(value, str):
                return value[:max_len]
            r = repr(obj)
            return r if len(r) <= max_len else r[:max_len] + "…"
        except Exception:
            return "<unprintable>"


class DefaultAgentRunnerFallback(PipelineAgentRunner):
    def __init__(self) -> None:
        from ..runtime.agent_runner import DefaultAgentRunner
        self._delegate = DefaultAgentRunner()

    async def run_stage(self, stage: str, context: Mapping[str, Any]) -> Any:
        return await self._delegate.run_stage(stage, context)

    async def run_feedback(self, context: Mapping[str, Any]) -> FeedbackDecision:
        return await self._delegate.run_feedback(context)


__all__ = ["OpenAIAgentRunner"]
