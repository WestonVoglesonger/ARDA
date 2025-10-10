"""
JSON parsing utilities for OpenAI agent responses.

Handles extraction of JSON payloads from various OpenAI response formats.
"""

from __future__ import annotations

import json
import os
from collections import deque
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Deque


class ResponseJSONParser:
    """Extracts JSON payloads from OpenAI agent responses."""
    
    def __init__(self, required_keys_by_stage: Dict[str, Set[str]], expects_object_stages: Set[str]):
        """
        Initialize parser with stage expectations.
        
        Args:
            required_keys_by_stage: Expected keys for each pipeline stage
            expects_object_stages: Stages that expect dict (not list) responses
        """
        self._req_keys = required_keys_by_stage
        self._expects_object = expects_object_stages
    
    def extract_response_payload(self, response: Any, stage: str) -> Any:
        """
        Extract JSON payload from agent response using multiple strategies.
        
        Tries in order:
        1. Top-level parsed fields
        2. Known block shapes (output_json, parsed, json)
        3. Deep BFS scan
        4. Text extraction + JSON parse
        
        Args:
            response: OpenAI agent response object
            stage: Pipeline stage name for validation
            
        Returns:
            Extracted and coerced payload
            
        Raises:
            RuntimeError: If no valid JSON payload found
        """
        # 0) Prefer parsed output at the top level.
        parsed = self._extract_top_level_parsed(response)
        if parsed is not None:
            return self._coerce_expected_type(parsed, stage)

        # 1) Try known block shapes.
        json_payload = self._extract_output_json(response)
        if json_payload is not None:
            return self._coerce_expected_type(json_payload, stage)

        # 2) Deep scan (hardened, stage-aware).
        deep = self._deep_scan_for_json(response, stage)
        if deep is not None:
            return self._coerce_expected_type(deep, stage)

        # 3) Fall back to text → JSON
        try:
            text_payload = self._extract_output_text(response, stage)
        except Exception:
            raise
        try:
            return self._coerce_expected_type(json.loads(text_payload), stage)
        except (TypeError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"Agent response for stage '{stage}' did not return valid JSON payload."
            ) from exc

    def _coerce_expected_type(self, value: Any, stage: str) -> Any:
        """
        Ensure response matches expected type (dict vs list).
        
        Stages expect objects (dict) not lists. If we got a list:
        - If length 1 and item is dict, use that dict.
        - Otherwise, pick the first/last dict that looks like the stage payload.
        """
        expects_obj = stage in self._expects_object
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
            req = self._req_keys.get(stage, set())
            candidates = [d for d in value if isinstance(d, dict)]
            # Best: one that has any of the required keys
            for d in candidates:
                if any(k in d for k in req):
                    return d
            # Otherwise, any dict at all
            if candidates:
                return candidates[-1]  # last emitted often the final result
        return value  # let downstream raise with a clear type message

    def _extract_top_level_parsed(self, response: Any) -> Optional[Any]:
        """Extract from top-level parsed fields."""
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
        """Extract from output blocks."""
        for block in self._iter_response_blocks(response):
            candidate = self._extract_json_candidate(block)
            if candidate is not None:
                return candidate
        return None

    def _iter_response_blocks(self, response: Any) -> Iterable[Any]:
        """Iterate over response blocks (output, content, etc.)."""
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
        """Extract JSON from a single block."""
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
        """Normalize various formats to plain JSON (dict/list)."""
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
        """Extract text from response for JSON parsing."""
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
        """Extract string from various text representations."""
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
        """Extract text from dict content."""
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

    def _is_envelope_like(self, d: Dict[str, Any]) -> bool:
        """
        Check if dict looks like OpenAI SDK envelope (not payload).
        
        Envelope typically has: id, object, created_at, model, status, etc.
        """
        if not isinstance(d, dict):
            return False
        markers = {
            "id", "object", "created_at", "model", "status",
            "usage", "store", "metadata", "input", "output", "response_id"
        }
        return len(markers & d.keys()) >= 2

    def _looks_like_payload(self, d: Dict[str, Any], stage: str) -> bool:
        """Check if dict looks like an ARDA stage payload."""
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
        req = self._req_keys.get(stage, set())
        if req and any(k in d for k in req):
            return True

        # fallback to global signal if no req keys are configured for this stage
        global_keys = set().union(*self._req_keys.values())
        return len(global_keys & d.keys()) > 0

    def _deep_scan_for_json(self, response: Any, stage: str) -> Optional[Any]:
        """
        BFS traversal to find payload dict without returning SDK envelope.
        
        Bounded to prevent infinite loops on circular structures.
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

