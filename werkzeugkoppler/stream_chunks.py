"""Helpers for OpenAI-compatible streaming chunk rewriting."""

from __future__ import annotations

import json
import uuid
from typing import Any


def append_tool_call_delta(
    tool_calls_by_index: dict[int, dict[str, Any]],
    delta_tool_calls: list[dict[str, Any]],
) -> None:
    """Merge tool-call streaming deltas into complete per-index objects."""
    for tc_delta in delta_tool_calls:
        if not isinstance(tc_delta, dict):
            continue

        index = tc_delta.get("index")
        if not isinstance(index, int):
            continue

        entry = tool_calls_by_index.setdefault(
            index,
            {"id": None, "type": "function", "function": {"name": "", "arguments": ""}},
        )
        if tc_delta.get("id"):
            entry["id"] = tc_delta["id"]
        if tc_delta.get("type"):
            entry["type"] = tc_delta["type"]

        fn_delta = tc_delta.get("function")
        if isinstance(fn_delta, dict):
            fn = entry.setdefault("function", {"name": "", "arguments": ""})
            if fn_delta.get("name"):
                fn["name"] = fn_delta["name"]
            if isinstance(fn_delta.get("arguments"), str):
                fn["arguments"] = str(fn.get("arguments", "")) + fn_delta["arguments"]


def normalized_tool_calls(tool_calls_by_index: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert collected per-index deltas to OpenAI-style tool call objects."""
    normalized: list[dict[str, Any]] = []
    for index in sorted(tool_calls_by_index.keys()):
        tc = tool_calls_by_index[index]
        fn = tc.get("function") or {}
        name = str(fn.get("name") or "").strip()
        if not name:
            continue

        normalized.append(
            {
                "id": tc.get("id") or f"call_{uuid.uuid4().hex}",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": str(fn.get("arguments") or "{}"),
                },
            }
        )
    return normalized


def client_chunk(
    *,
    completion_id: str,
    model: str,
    created: int,
    delta: dict[str, Any],
    finish_reason: str | None = None,
) -> dict[str, Any]:
    """Build a canonical `chat.completion.chunk` payload for clients."""
    return {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }


def rewrite_chunk_identity(
    chunk: dict[str, Any],
    *,
    completion_id: str,
    model: str,
) -> dict[str, Any]:
    """Rewrite upstream identity fields to one stable client stream identity."""
    out = dict(chunk)
    out["id"] = completion_id
    out["model"] = model
    return out


def pick_primary_choice(chunk: dict[str, Any]) -> dict[str, Any] | None:
    """Return the primary choice (index 0 if present) from an upstream chunk."""
    choices = chunk.get("choices")
    if not isinstance(choices, list) or not choices:
        return None

    for choice in choices:
        if isinstance(choice, dict) and choice.get("index") == 0:
            return choice

    first = choices[0]
    return first if isinstance(first, dict) else None


def with_single_primary_choice(chunk: dict[str, Any]) -> dict[str, Any]:
    """Normalize chunk to contain only one deterministic primary choice."""
    primary = pick_primary_choice(chunk)
    if primary is None:
        return {**chunk, "choices": []}

    normalized = dict(primary)
    normalized["index"] = 0
    return {**chunk, "choices": [normalized]}


def compact_json(value: Any, max_len: int = 3000) -> str:
    """Render bounded JSON text for reasoning trace lines."""
    try:
        raw = json.dumps(value, ensure_ascii=False)
    except Exception:
        raw = repr(value)
    if len(raw) > max_len:
        return raw[:max_len] + "...<truncated>"
    return raw


def reasoning_delta_payload(
    *,
    value: Any,
    preferred_field: str | None = None,
) -> dict[str, Any]:
    """Build reasoning delta payload with fixed passthrough behavior."""
    if preferred_field == "reasoning_content":
        return {"reasoning_content": value}
    return {"reasoning": value}


def reasoning_chunk(
    *,
    completion_id: str,
    model: str,
    created: int,
    value: Any,
    preferred_field: str | None = None,
) -> dict[str, Any]:
    """Build a chunk that appends text to the reasoning channel."""
    return client_chunk(
        completion_id=completion_id,
        model=model,
        created=created,
        delta=reasoning_delta_payload(
            value=value,
            preferred_field=preferred_field,
        ),
        finish_reason=None,
    )


def passthrough_non_content_delta_chunk(
    chunk: dict[str, Any],
    *,
    completion_id: str,
    model: str,
) -> dict[str, Any] | None:
    """Rewrite and forward delta fields except content/tool_calls."""
    out = rewrite_chunk_identity(chunk, completion_id=completion_id, model=model)
    choice = pick_primary_choice(out)
    if choice is None:
        return None

    delta = choice.get("delta")
    if not isinstance(delta, dict):
        return None

    passthrough_delta = {k: v for k, v in delta.items() if k not in {"content", "tool_calls"}}
    if not passthrough_delta:
        return None

    choice["delta"] = passthrough_delta
    # Passthrough events are incremental metadata/thinking; final stop is emitted with answer content later.
    choice["finish_reason"] = None
    return out


def is_reasoning_content_item(item: Any) -> bool:
    """Best-effort detection for reasoning-style content blocks."""
    if not isinstance(item, dict):
        return False
    item_type = str(item.get("type") or "").strip().lower()
    if "reason" in item_type:
        return True
    if item_type in {"summary", "summary_text", "thinking"}:
        return True
    return any(key in item for key in ("reasoning", "reasoning_content", "summary"))


def split_content_for_stream(content_value: Any) -> tuple[Any | None, Any | None]:
    """Split delta.content into immediate reasoning part and buffered final part."""
    if content_value is None:
        return None, None
    if isinstance(content_value, str):
        return None, content_value
    if isinstance(content_value, list):
        reasoning_items = [item for item in content_value if is_reasoning_content_item(item)]
        final_items = [item for item in content_value if not is_reasoning_content_item(item)]
        return (reasoning_items or None), (final_items or None)
    if is_reasoning_content_item(content_value):
        return content_value, None
    return None, content_value


def chunk_with_content_delta(
    chunk: dict[str, Any],
    *,
    completion_id: str,
    model: str,
    content_value: Any,
    finish_reason: str | None,
) -> dict[str, Any] | None:
    """Rewrite one chunk identity and replace delta with selected content payload."""
    out = rewrite_chunk_identity(chunk, completion_id=completion_id, model=model)
    choice = pick_primary_choice(out)
    if choice is None:
        return None
    delta = choice.get("delta")
    if not isinstance(delta, dict):
        return None
    new_delta = {"content": content_value}
    if isinstance(delta.get("role"), str):
        new_delta["role"] = delta["role"]
    choice["delta"] = new_delta
    choice["finish_reason"] = finish_reason
    return out


def content_to_thinking_value(content_value: Any) -> str | None:
    """Extract a readable thinking text from content payload shapes."""
    if isinstance(content_value, str):
        return content_value if content_value else None
    if isinstance(content_value, list):
        parts: list[str] = []
        for item in content_value:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str) and text:
                    parts.append(text)
        if parts:
            return "".join(parts)
    return None
