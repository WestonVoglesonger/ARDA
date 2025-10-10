# ardagen/agents/openai_runner.py
"""
Pipeline agent runner backed by the OpenAI Agents SDK.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Mapping, Optional, Iterable, List, Tuple, Deque, Set
from collections import deque

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

        response = self._handle_required_actions(response, context, tool_map, tool_requirements)
        
        # If the agent used tools but didn't return text/JSON, prompt for final response
        if tools and self._response_is_empty(response):
            if os.getenv("ARDA_DEBUG_EXTRACTION"):
                print(f"DEBUG: Agent for stage '{stage}' completed tools but returned no output. Prompting for JSON response.")
            response = self._prompt_for_final_response(response, stage, json_schema, model, model_params)
        
        return self._extract_response_payload(response, stage)

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
        payload = self._extract_response_payload(response, "feedback")
        if not isinstance(payload, dict):
            raise RuntimeError("Feedback agent must return a JSON object.")
        return FeedbackDecision(**payload)

    # --------------------------------------------------------------------- #
    # Response / Tool Handling
    # --------------------------------------------------------------------- #

    def _handle_required_actions(
        self,
        response: Any,
        context: Dict[str, Any],
        tool_map: Dict[str, Any],
        tool_requirements: Dict[str, Dict[str, Any]],
    ) -> Any:
        """Handle tool calls when response status is 'requires_action'."""
        response = self._ensure_final_response(response)
        
        # Process tool calls when status is requires_action
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
                raw_args = call.function.arguments or "{}"
                
                # Check for empty/whitespace-only arguments
                if not raw_args.strip() or raw_args.strip() in ("{}", "{\n}", "{\n\t}", "{ }", "{\n \t\t}", "{\n \t}"):
                    requirements = tool_requirements.get(tool_name, {})
                    required_fields = requirements.get("required", [])
                    schema_info = ""
                    if required_fields:
                        schema_info = f"\n\nRequired parameters:\n"
                        params = requirements.get("parameters", {})
                        props = params.get("properties", {})
                        for field in required_fields:
                            field_type = props.get(field, {}).get("type", "string")
                            schema_info += f"  - {field}: {field_type}\n"
                    error_msg = (
                        f"Tool '{tool_name}' called with empty or whitespace-only arguments. "
                        f"You must provide all required parameters.{schema_info}"
                        f"\nRaw arguments received: {repr(raw_args)}"
                    )
                    tool_outputs.append({"tool_call_id": call.id, "output": f"ERROR: {error_msg}"})
                    continue
                
                try:
                    args = json.loads(raw_args)
                except json.JSONDecodeError as e:
                    error_msg = f"Tool '{tool_name}' arguments are invalid JSON: {str(e)}. Raw: {repr(raw_args)}"
                    tool_outputs.append({"tool_call_id": call.id, "output": f"ERROR: {error_msg}"})
                    continue
                
                try:
                    requirements = tool_requirements.get(tool_name, {})
                    required_fields = requirements.get("required", []) or []
                    if required_fields:
                        if not isinstance(args, dict):
                            raise ValueError(
                                f"Tool '{tool_name}' expected object arguments but received {type(args).__name__}."
                            )
                        missing = [
                            field
                            for field in required_fields
                            if field not in args or args[field] in (None, "", [], {})
                        ]
                        if missing:
                            raise ValueError(
                                f"Missing required arguments for tool '{tool_name}': {', '.join(missing)}. "
                                f"Provide values for: {', '.join(required_fields)}."
                            )
                    output = self._invoke_tool(tool_name, args, context, tool_map)
                    tool_outputs.append({"tool_call_id": call.id, "output": output})
                except Exception as e:
                    error_msg = f"Tool '{tool_name}' failed: {str(e)}"
                    if "NoneType" in str(e) and "iterable" in str(e):
                        error_msg += f"\nTool arguments: {args}"
                        error_msg += f"\nContext keys: {list(context.keys())}"
                        if "results" in context:
                            error_msg += f"\nResults keys: {list(context['results'].keys())}"
                    tool_outputs.append({"tool_call_id": call.id, "output": f"ERROR: {error_msg}"})

            if os.getenv("ARDA_DEBUG_EXTRACTION"):
                print(f"DEBUG: Submitting {len(tool_outputs)} tool outputs:")
                for i, out in enumerate(tool_outputs):
                    output_preview = out['output'][:100] if len(out['output']) > 100 else out['output']
                    print(f"  Output {i+1}: call_id={out['tool_call_id'][:20]}... preview={repr(output_preview)}")
            
            response = self._client.responses.submit_tool_outputs(
                response_id=response.id,
                tool_outputs=tool_outputs,
            )
            response = self._ensure_final_response(response)
        return self._ensure_final_response(response)

    def _ensure_final_response(self, response: Any) -> Any:
        terminal_states = {"completed", "failed", "cancelled"}
        pollable_states = {"in_progress", "queued"}

        status = getattr(response, "status", None)
        while status in pollable_states:
            time.sleep(0.5)
            response = self._client.responses.retrieve(response.id)
            status = getattr(response, "status", None)

        if status and status not in terminal_states and status != "requires_action":
            time.sleep(0.5)
            response = self._client.responses.retrieve(response.id)

        return response
    
    def _response_is_empty(self, response: Any) -> bool:
        """Check if response has no textual output."""
        # Check output_text
        output_text = getattr(response, "output_text", None)
        if output_text and output_text.strip():
            return False
        
        # Check output blocks for text content
        output = getattr(response, "output", None)
        if output:
            items = output if isinstance(output, (list, tuple)) else [output]
            for item in items:
                if hasattr(item, "type"):
                    # Skip function calls and reasoning blocks
                    if item.type in ("function_call", "reasoning"):
                        continue
                    # Check for text content
                    if hasattr(item, "text") and item.text:
                        return False
                    if hasattr(item, "content") and item.content:
                        return False
        
        return True
    
    def _prompt_for_final_response(self, previous_response: Any, stage: str, json_schema: Optional[Dict[str, Any]], model: str, model_params: Dict[str, Any]) -> Any:
        """Prompt the agent to provide final JSON response after tool calls."""
        prompt_message = {
            "role": "user",
            "content": (
                "You have successfully completed the tool calls. "
                "Now provide the final JSON response summarizing what you generated. "
                f"Return ONLY a JSON object (no markdown, no code fences) matching this schema:\n```json\n{json.dumps(json_schema, indent=2)}\n```"
            )
        }
        
        # Note: response_format cannot be used with previous_response_id in the Responses API
        # The agent must return plain text JSON that we'll parse
        
        try:
            followup_response = self._client.responses.create(
                model=model,
                input=[prompt_message],
                previous_response_id=previous_response.id,
                **model_params
            )
            return self._ensure_final_response(followup_response)
        except Exception as e:
            if os.getenv("ARDA_DEBUG_EXTRACTION"):
                print(f"DEBUG: Failed to get follow-up response: {e}")
            # Return the original response if follow-up fails
            return previous_response

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

    # --------------------------------------------------------------------- #
    # Extraction
    # --------------------------------------------------------------------- #

    def _extract_response_payload(self, response: Any, stage: str) -> Any:
        # 0) Prefer parsed output at the top level.
        parsed = self._extract_top_level_parsed(response)
        if parsed is not None:
            _dbg(f"{stage}: used top-level parsed")
            return self._coerce_expected_type(parsed, stage)

        # 1) Try known block shapes.
        json_payload = self._extract_output_json(response)
        if json_payload is not None:
            _dbg(f"{stage}: used block JSON (output_json/parsed/json)")
            return self._coerce_expected_type(json_payload, stage)

        # 2) Deep scan (hardened, stage-aware).
        deep = self._deep_scan_for_json(response, stage)
        if deep is not None:
            _dbg(f"{stage}: used deep scan candidate")
            return self._coerce_expected_type(deep, stage)

        # 3) Fall back to text → JSON
        try:
            text_payload = self._extract_output_text(response, stage)
        except Exception:
            self._debug_dump_response(response, stage, reason="no_textual_output")
            raise
        try:
            return self._coerce_expected_type(json.loads(text_payload), stage)
        except (TypeError, json.JSONDecodeError) as exc:
            self._debug_dump_response(response, stage, reason="invalid_json_text")
            raise RuntimeError(
                f"Agent response for stage '{stage}' did not return valid JSON payload."
            ) from exc

    def _coerce_expected_type(self, value: Any, stage: str) -> Any:
        """
        Stages expect objects (dict) not lists. If we somehow got a list:
        - If length 1 and item is dict, use that dict.
        - Otherwise, pick the first/last dict that looks like the stage payload.
        """
        expects_obj = stage in self._EXPECTS_OBJECT
        if not expects_obj:
            return value

        # Already a dict? good.
        if isinstance(value, dict):
            return value

        # If it's a list, try to unwrap
        if isinstance(value, list):
            # Single-element unwrap
            if len(value) == 1 and isinstance(value[0], dict):
                return value[0]
            # Select a plausible dict from the list
            req = self._required_keys_for_stage(stage)
            candidates = [d for d in value if isinstance(d, dict)]
            # Best: one that has any of the required keys
            for d in candidates:
                if any(k in d for k in req):
                    return d
            # Otherwise, any dict at all
            if candidates:
                return candidates[-1]  # last emitted often the final result
        return value  # let downstream raise with a clear type message

    def _required_keys_for_stage(self, stage: str) -> Set[str]:
        return self._REQ_KEYS.get(stage, set())

    def _extract_top_level_parsed(self, response: Any) -> Optional[Any]:
        for key in ("output_parsed", "parsed", "response_parsed"):
            value = getattr(response, key, None)
            if value is None and isinstance(response, dict):
                value = response.get(key)
            if value is not None:
                normalized = self._normalize_json_value(value)
                if normalized is not None:
                    return normalized
        return None

    def _extract_output_json(self, response: Any) -> Optional[Any]:
        for block in self._iter_response_blocks(response):
            candidate = self._extract_json_candidate(block)
            if candidate is not None:
                return candidate
        return None

    def _iter_response_blocks(self, response: Any) -> Iterable[Any]:
        if response is None:
            return []

        candidates: List[Tuple[str, Any]] = []
        output = getattr(response, "output", None)
        if output is None and isinstance(response, dict):
            output = response.get("output")
        if output:
            candidates.append(("output", output))

        if not output:
            for alt in ("message", "messages", "content"):
                val = getattr(response, alt, None)
                if val is None and isinstance(response, dict):
                    val = response.get(alt)
                if val:
                    candidates.append((alt, val))

        for _name, outer in candidates:
            items = outer if isinstance(outer, (list, tuple)) else [outer]
            for item in items:
                yield item
                contents = getattr(item, "content", None)
                if contents is None and isinstance(item, dict):
                    contents = item.get("content")
                if not contents:
                    continue
                if not isinstance(contents, (list, tuple)):
                    contents = [contents]
                for content in contents:
                    yield content

    def _extract_json_candidate(self, block: Any) -> Optional[Any]:
        if block is None:
            return None

        for key in ("output_json", "parsed", "json"):
            value = None
            if isinstance(block, dict):
                value = block.get(key)
            else:
                if key == "json" and getattr(block, "type", None) != "output_json":
                    continue
                value = getattr(block, key, None)
            if value is not None:
                normalized = self._normalize_json_value(value)
                if normalized is not None:
                    return normalized

        # output_json wrapper nesting (newer SDK shapes)
        if (isinstance(block, dict) and block.get("type") == "output_json") or getattr(block, "type", None) == "output_json":
            nested = None
            if isinstance(block, dict):
                nested = block.get("content") or block.get("value") or block.get("data")
            else:
                nested = getattr(block, "content", None) or getattr(block, "value", None) or getattr(block, "data", None)
            if nested is not None:
                if isinstance(nested, (list, tuple)):
                    for node in nested:
                        if isinstance(node, dict) and node.get("type") == "json_schema" and "parsed" in node:
                            norm = self._normalize_json_value(node.get("parsed"))
                            if norm is not None:
                                return norm
                norm = self._normalize_json_value(nested)
                if norm is not None:
                    return norm
        return None

    @staticmethod
    def _normalize_json_value(value: Any) -> Optional[Any]:
        if value is None:
            return None
        if callable(value):
            try:
                value = value()
            except TypeError:
                return None
        try:
            if hasattr(value, "model_dump"):  # pydantic v2
                return value.model_dump()
            if hasattr(value, "dict"):  # pydantic v1
                return value.dict()
            if is_dataclass(value):
                return asdict(value)
        except Exception:
            pass
        if isinstance(value, (bytes, bytearray)):
            value = value.decode("utf-8", errors="ignore")
        if isinstance(value, str):
            s = value.strip()
            if not s:
                return None
            try:
                return json.loads(s)
            except json.JSONDecodeError:
                return None
        return value

    def _extract_output_text(self, response: Any, stage: str) -> str:
        text = getattr(response, "output_text", None)
        if text:
            extracted = self._extract_text_value(text)
            if extracted:
                return extracted
        for block in self._iter_response_blocks(response):
            if isinstance(block, str) and block:
                return block
            if isinstance(block, dict):
                dict_text = self._extract_text_from_dict(block)
                if dict_text:
                    return dict_text
                continue
            text_part = getattr(block, "text", None)
            if text_part:
                text_value = self._extract_text_value(text_part)
                if text_value:
                    return text_value
            output_text = getattr(block, "output_text", None)
            if output_text:
                extracted = self._extract_text_value(output_text)
                if extracted:
                    return extracted
        raise RuntimeError(f"Agent response for stage '{stage}' did not include textual output.")

    @staticmethod
    def _extract_text_value(candidate: Any) -> Optional[str]:
        if isinstance(candidate, str):
            return candidate
        if isinstance(candidate, dict):
            return candidate.get("value") or candidate.get("text")
        value = getattr(candidate, "value", None)
        if value:
            return value
        text = getattr(candidate, "text", None)
        if isinstance(text, str):
            return text
        if text and hasattr(text, "value"):
            return text.value
        return None

    def _extract_text_from_dict(self, content: Dict[str, Any]) -> Optional[str]:
        if "json" in content and content["json"] is not None:
            payload = content["json"]
            if isinstance(payload, (str, bytes, bytearray)):
                return payload.decode() if isinstance(payload, (bytes, bytearray)) else payload
            try:
                return json.dumps(payload)
            except (TypeError, ValueError):
                return str(payload)
        if "output_text" in content and isinstance(content["output_text"], str):
            return content["output_text"]
        text = content.get("text")
        text_value = self._extract_text_value(text)
        if text_value:
            return text_value
        if content.get("type") in {"output_text", "text"}:
            nested = content.get(content["type"])
            nested_value = self._extract_text_value(nested)
            if nested_value:
                return nested_value
        return None

    # ------------------------------ #
    # Deep scan (bounded & hardened)
    # ------------------------------ #

    def _is_envelope_like(self, d: Dict[str, Any]) -> bool:
        """
        Heuristic: OpenAI Responses envelope typically has several of these:
        id, object, created_at, model, status, usage, store, metadata, input, output.
        If we see >=2 of these markers, treat it as a wrapper, not payload.
        """
        if not isinstance(d, dict):
            return False
        markers = {
            "id", "object", "created_at", "model", "status",
            "usage", "store", "metadata", "input", "output", "response_id"
        }
        return len(markers & d.keys()) >= 2

    def _required_keys_for_stage(self, stage: str) -> Set[str]:
        return self._REQ_KEYS.get(stage, set())

    def _looks_like_payload(self, d: Dict[str, Any], stage: str) -> bool:
        """
        Positive heuristic for ARDA stage objects.
        """
        if not isinstance(d, dict):
            return False
        if self._is_envelope_like(d):
            return False

        # Avoid tool call metadata and response artifacts
        tool_indicators = {"function", "arguments", "tool_calls", "call_id", "parameters"}
        response_artifacts = {"annotations", "logprobs", "text", "finish_reason", "index", "object"}
        if any(k in d for k in tool_indicators) or any(k in d for k in response_artifacts):
            return False

        # Check for required keys for this stage
        req = self._required_keys_for_stage(stage)
        if req and any(k in d for k in req):
            return True

        # fallback to global signal if no req keys are configured for this stage
        global_keys = set().union(*self._REQ_KEYS.values())
        return len(global_keys & d.keys()) > 0

    def _deep_scan_for_json(self, response: Any, stage: str) -> Optional[Any]:
        """
        BFS over the response without ever returning the SDK envelope.
        Returns a dict that looks like the stage payload, or a normalized
        value of a 'parsed'/'json' field.
        """
        max_nodes = int(os.getenv("ARDA_OPENAI_WALK_LIMIT", "50000"))
        queue: Deque[Any] = deque()
        seen: Set[int] = set()

        def enqueue(x: Any) -> None:
            try:
                oid = id(x)
                if oid in seen:
                    return
                seen.add(oid)
            except Exception:
                pass
            queue.append(x)

        enqueue(response)
        nodes = 0

        obj_attrs = (
            "output", "output_parsed", "output_text",
            "message", "messages",
            "content", "text", "value",
            "json", "parsed", "data",
            "items", "choices", "parts", "results",
            "tool_calls"
        )

        while queue and nodes < max_nodes:
            nodes += 1
            cur = queue.popleft()

            # Common "json_schema parsed" leaf
            if isinstance(cur, dict) and cur.get("type") == "json_schema" and "parsed" in cur:
                norm = self._normalize_json_value(cur.get("parsed"))
                if isinstance(norm, (dict, list)):
                    return norm

            # Explicit parsed/json fields
            if isinstance(cur, dict):
                for k in ("parsed", "json"):
                    if k in cur:
                        norm = self._normalize_json_value(cur[k])
                        if isinstance(norm, (dict, list)):
                            return norm

            # Prefer dict payloads
            if isinstance(cur, dict) and self._looks_like_payload(cur, stage):
                return cur

            # Lists: do not return the list for object-expected stages.
            if isinstance(cur, list):
                # First try: find a dict in the list that looks like payload
                for d in cur:
                    if isinstance(d, dict) and self._looks_like_payload(d, stage):
                        return d
                # If single element and dict, inspect before returning.
                if len(cur) == 1 and isinstance(cur[0], dict):
                    candidate = cur[0]
                    if self._looks_like_payload(candidate, stage):
                        return candidate
                    enqueue(candidate)
                    continue
                # If stage doesn't strictly require object, you could return list
                # (No stages in _EXPECTS_OBJECT allow this, so we skip returning the list)

            # Expand children
            try:
                if hasattr(cur, "model_dump"):
                    enqueue(cur.model_dump())
                elif hasattr(cur, "dict"):
                    enqueue(cur.dict())
                elif is_dataclass(cur):
                    enqueue(asdict(cur))
            except Exception:
                pass

            if isinstance(cur, dict):
                for k in obj_attrs:
                    if k in cur:
                        enqueue(cur[k])
                for v in list(cur.values())[:32]:
                    enqueue(v)
                continue

            if isinstance(cur, (list, tuple)):
                for v in list(cur)[:64]:
                    enqueue(v)
                continue

            for k in obj_attrs:
                try:
                    if hasattr(cur, k):
                        enqueue(getattr(cur, k))
                except Exception:
                    pass

        return None

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
