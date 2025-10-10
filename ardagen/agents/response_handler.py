"""
Response handling utilities for OpenAI agent responses.

Handles tool call processing, response polling, and validation.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Mapping, Optional


class ResponseHandler:
    """Handles OpenAI agent response processing and tool execution."""
    
    def __init__(self, client: Any):
        """
        Initialize handler with OpenAI client.
        
        Args:
            client: OpenAI client instance
        """
        self._client = client
    
    def handle_required_actions(
        self,
        response: Any,
        context: Dict[str, Any],
        tool_map: Dict[str, Any],
        tool_requirements: Dict[str, Dict[str, Any]],
    ) -> Any:
        """
        Handle tool calls when response status is 'requires_action'.
        
        Args:
            response: OpenAI response object
            context: Pipeline context
            tool_map: Available tools mapping
            tool_requirements: Tool schema requirements
            
        Returns:
            Final response after tool execution
        """
        response = self.ensure_final_response(response)
        
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
            response = self.ensure_final_response(response)
        
        return self.ensure_final_response(response)

    def ensure_final_response(self, response: Any) -> Any:
        """
        Poll response until it reaches a terminal state.
        
        Args:
            response: OpenAI response object
            
        Returns:
            Response in terminal state
        """
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
    
    def response_is_empty(self, response: Any) -> bool:
        """
        Check if response has no textual output.
        
        Args:
            response: OpenAI response object
            
        Returns:
            True if response is empty
        """
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
    
    def prompt_for_final_response(
        self,
        previous_response: Any,
        stage: str,
        json_schema: Optional[Dict[str, Any]],
        model: str,
        model_params: Dict[str, Any]
    ) -> Any:
        """
        Prompt agent to provide final JSON response after tool calls.
        
        Args:
            previous_response: Previous response object
            stage: Pipeline stage name
            json_schema: Expected JSON schema
            model: Model name
            model_params: Model parameters
            
        Returns:
            Final response with JSON payload
        """
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
            return self.ensure_final_response(followup_response)
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
        """
        Invoke a tool function and return result.
        
        Args:
            tool_name: Name of tool to invoke
            arguments: Tool arguments
            context: Pipeline context
            tool_map: Available tools mapping
            
        Returns:
            JSON-encoded tool result
            
        Raises:
            KeyError: If tool not found
        """
        if tool_name not in tool_map:
            raise KeyError(f"Tool '{tool_name}' not registered for agent runner.")
        
        result = tool_map[tool_name](**arguments)
        
        # Log tool invocation if observability is enabled
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

