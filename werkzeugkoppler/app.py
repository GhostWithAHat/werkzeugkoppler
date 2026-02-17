"""HTTP application for the werkzeugkoppler gateway.

This module exposes an OpenAI-compatible API and orchestrates:
- upstream OpenAI-compatible chat completions,
- MCP/local tools,
- streaming output with one final assistant answer and a reasoning trace.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import os
import re
import sys
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncGenerator
from urllib.parse import urlparse

try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import JSONResponse, StreamingResponse
except ModuleNotFoundError as exc:  # pragma: no cover
    FastAPI = Any  # type: ignore[assignment]
    HTTPException = Exception  # type: ignore[assignment]
    Request = Any  # type: ignore[assignment]
    JSONResponse = Any  # type: ignore[assignment]
    StreamingResponse = Any  # type: ignore[assignment]
    _FASTAPI_IMPORT_ERROR: ModuleNotFoundError | None = exc
else:
    _FASTAPI_IMPORT_ERROR = None

try:
    from .config import DEFAULT_CONFIG_PATH, GatewayConfig, load_config
    from .logging_utils import setup_logging
    from .mcp_client import MCPError
    from .tool_registry import ToolRegistry
    from .upstream import UpstreamClient
    from .utils import to_bounded_json
except ModuleNotFoundError as exc:  # pragma: no cover
    GatewayConfig = Any  # type: ignore[assignment]
    MCPError = Exception  # type: ignore[assignment]
    ToolRegistry = Any  # type: ignore[assignment]
    UpstreamClient = Any  # type: ignore[assignment]
    load_config = None  # type: ignore[assignment]
    setup_logging = None  # type: ignore[assignment]
    to_bounded_json = None  # type: ignore[assignment]
    _CORE_IMPORT_ERROR: ModuleNotFoundError | None = exc
else:
    _CORE_IMPORT_ERROR = None

LOG = logging.getLogger(__name__)
_TODAY_TOKEN_RE = re.compile(r"\{\s*today\s*\}")


def _extract_bearer_token(request: Request) -> str | None:
    """Extract bearer token from Authorization header if present."""
    auth_header = request.headers.get("authorization")
    if not auth_header:
        return None
    parts = auth_header.strip().split(" ", 1)
    if len(parts) != 2:
        return None
    scheme, token = parts[0].lower(), parts[1].strip()
    if scheme != "bearer" or not token:
        return None
    return token


def _require_gateway_auth(request: Request, cfg: GatewayConfig) -> None:
    """Enforce gateway API key auth when configured."""
    required_key = cfg.service_api_key
    if not required_key:
        return
    provided_key = _extract_bearer_token(request)
    if provided_key != required_key:
        raise HTTPException(status_code=401, detail="Unauthorized")


def _service_bind_addr(service_base_url: str) -> tuple[str, int]:
    """Parse bind host/port from service_base_url."""
    parsed = urlparse(service_base_url)
    if not parsed.hostname or parsed.port is None:
        raise ValueError("service_base_url must include host and port, e.g. http://127.0.0.1:8080")
    return parsed.hostname, parsed.port


def _format_now_german(now: datetime) -> str:
    """Format datetime in German style used for `{today}` replacement."""
    weekdays = [
        "Montag",
        "Dienstag",
        "Mittwoch",
        "Donnerstag",
        "Freitag",
        "Samstag",
        "Sonntag",
    ]
    months = [
        "Januar",
        "Februar",
        "März",
        "April",
        "Mai",
        "Juni",
        "Juli",
        "August",
        "September",
        "Oktober",
        "November",
        "Dezember",
    ]
    return f"{weekdays[now.weekday()]}, {now.day}. {months[now.month - 1]} {now.year}, {now:%H:%M:%S}"


def _replace_today_tokens(content: Any) -> Any:
    """Replace `{today}` placeholder tokens inside message content payloads."""
    now_text = _format_now_german(datetime.now())

    if isinstance(content, str):
        return _TODAY_TOKEN_RE.sub(now_text, content)

    if isinstance(content, list):
        out: list[Any] = []
        for item in content:
            if isinstance(item, dict):
                updated = dict(item)
                if isinstance(updated.get("text"), str):
                    updated["text"] = _TODAY_TOKEN_RE.sub(now_text, updated["text"])
                out.append(updated)
            else:
                out.append(item)
        return out

    return content


def _prepare_messages(messages: list[Any], first_messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Prepend configured first messages and normalize placeholder tokens."""
    combined = [*first_messages, *messages] if first_messages else list(messages)
    prepared: list[dict[str, Any]] = []

    for msg in combined:
        if not isinstance(msg, dict):
            continue
        updated = dict(msg)
        if "content" in updated:
            updated["content"] = _replace_today_tokens(updated.get("content"))
        prepared.append(updated)

    return prepared


def _sse_data(payload: dict[str, Any]) -> bytes:
    """Encode one SSE `data:` event."""
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8")


def _append_tool_call_delta(
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


def _normalized_tool_calls(tool_calls_by_index: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
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


def _client_chunk(
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


def _rewrite_chunk_identity(
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


def _pick_primary_choice(chunk: dict[str, Any]) -> dict[str, Any] | None:
    """Return the primary choice (index 0 if present) from an upstream chunk."""
    choices = chunk.get("choices")
    if not isinstance(choices, list) or not choices:
        return None

    for choice in choices:
        if isinstance(choice, dict) and choice.get("index") == 0:
            return choice

    first = choices[0]
    return first if isinstance(first, dict) else None


def _with_single_primary_choice(chunk: dict[str, Any]) -> dict[str, Any]:
    """Normalize chunk to contain only one deterministic primary choice."""
    primary = _pick_primary_choice(chunk)
    if primary is None:
        return {**chunk, "choices": []}

    normalized = dict(primary)
    normalized["index"] = 0
    return {**chunk, "choices": [normalized]}


def _compact_json(value: Any, max_len: int = 3000) -> str:
    """Render bounded JSON text for reasoning trace lines."""
    try:
        raw = json.dumps(value, ensure_ascii=False)
    except Exception:
        raw = repr(value)
    if len(raw) > max_len:
        return raw[:max_len] + "...<truncated>"
    return raw


def _reasoning_chunk(*, completion_id: str, model: str, created: int, text: str) -> dict[str, Any]:
    """Build a chunk that appends text to the reasoning channel."""
    return _client_chunk(
        completion_id=completion_id,
        model=model,
        created=created,
        delta={"reasoning_content": text},
        finish_reason=None,
    )


def _tool_trace_line(status: dict[str, Any]) -> str | None:
    """Convert tool status events to compact reasoning trace lines."""
    event_type = status.get("type")
    if event_type == "tool_start":
        payload = {"name": status.get("tool"), "arguments": status.get("arguments")}
        return f"\n\n==> tool call `{_compact_json(payload)}`\n"
    if event_type == "tool_end":
        if status.get("ok"):
            payload: Any = status.get("result")
        else:
            payload = {"error": status.get("error"), "is_error": True}
        return f"==> tool result `{_compact_json(payload)}`\n\n"
    return None


def _build_base_payload(
    request_payload: dict[str, Any],
    upstream_default_model: str | None,
) -> tuple[dict[str, Any], list[Any], list[Any]]:
    """Extract and normalize common chat request fields."""
    messages = list(request_payload.get("messages") or [])
    if not isinstance(messages, list):
        raise HTTPException(status_code=400, detail="messages must be an array")

    user_tools = request_payload.get("tools") or []
    blocked_keys = {"messages", "tools", "stream"}
    base_payload = {key: value for key, value in request_payload.items() if key not in blocked_keys}
    if upstream_default_model and not base_payload.get("model"):
        base_payload["model"] = upstream_default_model

    return base_payload, messages, user_tools


def _stable_tool_calls(tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Assign stable tool call IDs for internal orchestration messages."""
    stable: list[dict[str, Any]] = []
    for tc in tool_calls:
        fn = tc.get("function") or {}
        stable.append(
            {
                "id": f"call_{uuid.uuid4().hex}",
                "type": "function",
                "function": {
                    "name": fn.get("name"),
                    "arguments": str(fn.get("arguments") or "{}"),
                },
            }
        )
    return stable


class GatewayService:
    """Runtime container for upstream and tool-registry integrations."""

    def __init__(self, cfg: GatewayConfig) -> None:
        """Initialize service with config-bound clients."""
        self.cfg = cfg
        self.registry = ToolRegistry(cfg)
        self.upstream = UpstreamClient(cfg)
        self._resolved_upstream_default_model: str | None = cfg.upstream_default_model
        self._op_lock = asyncio.Lock()

    async def start(self) -> None:
        """Start background clients and initial tool loading."""
        await self.registry.start()

    async def close(self) -> None:
        """Shut down clients and background resources."""
        await self.registry.close()
        await self.upstream.close()

    async def reload(self, new_cfg: GatewayConfig) -> None:
        """Hot-reload configuration by swapping integrations atomically."""
        old_registry = self.registry
        old_upstream = self.upstream

        new_registry = ToolRegistry(new_cfg)
        new_upstream = UpstreamClient(new_cfg)
        await new_registry.start()

        self.cfg = new_cfg
        self.registry = new_registry
        self.upstream = new_upstream
        self._resolved_upstream_default_model = new_cfg.upstream_default_model

        await old_registry.close()
        await old_upstream.close()

    async def stream_chat(self, request_payload: dict[str, Any]) -> AsyncGenerator[bytes, None]:
        """Run one streaming chat request under the service operation lock."""
        async with self._op_lock:
            async for chunk in _stream_chat(self, request_payload):
                yield chunk

    async def list_models(self) -> dict[str, Any]:
        """Fetch models from upstream with fallback to configured default model."""
        try:
            return await self.upstream.list_models()
        except Exception as exc:
            LOG.warning("Upstream /v1/models failed, fallback to default model: %s", exc)
            fallback_model = await self.resolve_default_model()
            return {
                "object": "list",
                "data": [
                    {
                        "id": fallback_model,
                        "object": "model",
                        "created": int(datetime.now(timezone.utc).timestamp()),
                        "owned_by": "werkzeugkoppler",
                    }
                ],
            }

    async def resolve_default_model(self) -> str:
        """Resolve default upstream model from config or upstream model list."""
        if self._resolved_upstream_default_model:
            return self._resolved_upstream_default_model
        if self.cfg.upstream_default_model:
            self._resolved_upstream_default_model = self.cfg.upstream_default_model
            return self._resolved_upstream_default_model

        try:
            models = await self.upstream.list_models()
            data = models.get("data")
            if isinstance(data, list):
                for entry in data:
                    if isinstance(entry, dict) and entry.get("id"):
                        self._resolved_upstream_default_model = str(entry["id"])
                        LOG.info(
                            "Selected upstream default model dynamically model=%s",
                            self._resolved_upstream_default_model,
                        )
                        return self._resolved_upstream_default_model
        except Exception as exc:
            LOG.warning("Could not auto-detect upstream model: %s", exc)

        raise HTTPException(
            status_code=502,
            detail="No model specified and no upstream_default_model configured; upstream model list is unavailable",
        )

    async def _execute_tool_calls(
        self,
        tool_calls: list[dict[str, Any]],
        status_queue: asyncio.Queue[dict[str, Any]] | None,
    ) -> list[dict[str, Any]]:
        """Execute tool calls with bounded concurrency and emit status events."""
        semaphore = asyncio.Semaphore(max(1, self.cfg.max_tool_concurrency))

        async def run_one(tool_call: dict[str, Any]) -> dict[str, Any]:
            async with semaphore:
                tool_call_id = tool_call.get("id") or f"call_{uuid.uuid4().hex}"
                function = tool_call.get("function") or {}
                tool_name = str(function.get("name", ""))
                raw_args = function.get("arguments") or "{}"

                try:
                    args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                except Exception:
                    args = {"_raw": raw_args}

                if status_queue is not None:
                    await status_queue.put(
                        {
                            "type": "tool_start",
                            "tool": tool_name,
                            "tool_call_id": tool_call_id,
                            "arguments": args,
                        }
                    )

                try:
                    outcome = await self.registry.call_tool_by_openai_name(tool_name, args)
                    result_payload = {"ok": True, "result": outcome.get("result")}
                    if status_queue is not None:
                        await status_queue.put(
                            {
                                "type": "tool_end",
                                "tool": tool_name,
                                "tool_call_id": tool_call_id,
                                "ok": True,
                                "result": outcome.get("result"),
                            }
                        )
                except Exception as exc:
                    result_payload = {"ok": False, "error": str(exc), "is_error": True}
                    if status_queue is not None:
                        await status_queue.put(
                            {
                                "type": "tool_end",
                                "tool": tool_name,
                                "tool_call_id": tool_call_id,
                                "ok": False,
                                "error": str(exc),
                                "is_error": True,
                            }
                        )

                return self.registry.format_tool_message(tool_name, tool_call_id, result_payload)

        return await asyncio.gather(*(run_one(tc) for tc in tool_calls))

    async def resolve_chat_non_stream(
        self,
        request_payload: dict[str, Any],
        status_queue: asyncio.Queue[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Resolve non-streaming chat requests with iterative tool orchestration."""
        base_payload, messages, user_tools = _build_base_payload(request_payload, self.cfg.upstream_default_model)
        if not base_payload.get("model"):
            base_payload["model"] = await self.resolve_default_model()
        messages = _prepare_messages(messages, self.cfg.first_messages)

        mcp_tools = await self.registry.get_openai_tools()
        combined_tools = self._merge_tools(user_tools, mcp_tools)

        for _ in range(max(1, self.cfg.max_tool_loops)):
            upstream_payload = {
                **base_payload,
                "messages": messages,
                "tools": combined_tools,
                "stream": False,
            }
            upstream_response = await self.upstream.chat_completion(upstream_payload)
            choice = (upstream_response.get("choices") or [{}])[0]
            msg = choice.get("message") or {}

            reasoning = msg.get("reasoning") or msg.get("reasoning_content")
            if reasoning and status_queue is not None:
                await status_queue.put({"type": "reasoning", "reasoning": reasoning})

            tool_calls = msg.get("tool_calls") or []
            if not tool_calls:
                return {
                    "mode": "final",
                    "response": upstream_response,
                    "messages": messages,
                    "base_payload": base_payload,
                }

            messages.append(
                {
                    "role": "assistant",
                    "content": msg.get("content"),
                    "tool_calls": tool_calls,
                }
            )
            tool_messages = await self._execute_tool_calls(tool_calls, status_queue)
            messages.extend(tool_messages)

        raise HTTPException(status_code=400, detail="Maximum tool loop iterations reached")

    @staticmethod
    def _merge_tools(user_tools: list[dict[str, Any]], mcp_tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Merge user-specified and discovered tools by unique function name."""
        merged: list[dict[str, Any]] = []
        seen: set[str] = set()

        for tool in [*user_tools, *mcp_tools]:
            if not isinstance(tool, dict):
                continue
            fn = tool.get("function") or {}
            name = str(fn.get("name", ""))
            if not name or name in seen:
                continue
            seen.add(name)
            merged.append(tool)

        return merged


async def _stream_tool_execution_trace(
    service: GatewayService,
    *,
    completion_id: str,
    model: str,
    created: int,
    messages: list[dict[str, Any]],
    assistant_content: str | None,
    tool_calls: list[dict[str, Any]],
) -> AsyncGenerator[bytes, None]:
    """Execute tool calls and stream their start/end trace lines as reasoning."""
    messages.append(
        {
            "role": "assistant",
            "content": assistant_content,
            "tool_calls": tool_calls,
        }
    )

    status_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    tool_task = asyncio.create_task(service._execute_tool_calls(tool_calls, status_queue))

    while not tool_task.done():
        try:
            status = await asyncio.wait_for(status_queue.get(), timeout=0.2)
        except asyncio.TimeoutError:
            continue

        line = _tool_trace_line(status)
        if line:
            yield _sse_data(
                _reasoning_chunk(
                    completion_id=completion_id,
                    model=model,
                    created=created,
                    text=line,
                )
            )

    while not status_queue.empty():
        status = await status_queue.get()
        line = _tool_trace_line(status)
        if line:
            yield _sse_data(
                _reasoning_chunk(
                    completion_id=completion_id,
                    model=model,
                    created=created,
                    text=line,
                )
            )

    messages.extend(await tool_task)


async def _stream_chat(
    service: GatewayService,
    request_payload: dict[str, Any],
) -> AsyncGenerator[bytes, None]:
    """Run streaming chat orchestration with tool loops.

    The client receives:
    - one assistant role start chunk,
    - reasoning deltas (including tool call/result trace),
    - one final assistant answer stream,
    - DONE marker.
    """
    base_payload, messages, user_tools = _build_base_payload(request_payload, service.cfg.upstream_default_model)
    if not base_payload.get("model"):
        base_payload["model"] = await service.resolve_default_model()
    messages = _prepare_messages(messages, service.cfg.first_messages)

    mcp_tools = await service.registry.get_openai_tools()
    combined_tools = service._merge_tools(user_tools, mcp_tools)

    base_payload["n"] = 1
    model = str(base_payload["model"])
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(datetime.now(timezone.utc).timestamp())

    # Emit role chunk immediately to keep UI stream alive.
    yield _sse_data(
        _client_chunk(
            completion_id=completion_id,
            model=model,
            created=created,
            delta={"role": "assistant"},
            finish_reason=None,
        )
    )

    for _ in range(max(1, service.cfg.max_tool_loops)):
        upstream_payload = {
            **base_payload,
            "messages": messages,
            "tools": combined_tools,
            "stream": True,
        }

        assistant_content_parts: list[str] = []
        tool_calls_by_index: dict[int, dict[str, Any]] = {}
        finish_reason: str | None = None
        got_chunks = False
        buffered_content_chunks: list[dict[str, Any]] = []

        async for chunk in service.upstream.stream_chat_completion(upstream_payload):
            got_chunks = True
            single_chunk = _with_single_primary_choice(chunk)
            choice0 = _pick_primary_choice(single_chunk) or {}
            finish_reason = choice0.get("finish_reason") or finish_reason
            delta = choice0.get("delta") or {}

            reasoning_text = delta.get("reasoning_content") or delta.get("reasoning")
            if isinstance(reasoning_text, str) and reasoning_text.strip():
                yield _sse_data(
                    _reasoning_chunk(
                        completion_id=completion_id,
                        model=model,
                        created=created,
                        text=reasoning_text,
                    )
                )

            if isinstance(delta.get("content"), str):
                assistant_content_parts.append(delta["content"])
                buffered_content_chunks.append(single_chunk)

            delta_tool_calls = delta.get("tool_calls")
            if isinstance(delta_tool_calls, list):
                _append_tool_call_delta(tool_calls_by_index, delta_tool_calls)

        if not got_chunks:
            # Fallback for upstreams that ignore streaming.
            fallback_payload = dict(upstream_payload)
            fallback_payload["stream"] = False
            fallback_response = await service.upstream.chat_completion(fallback_payload)
            msg = ((fallback_response.get("choices") or [{}])[0]).get("message") or {}
            tool_calls = msg.get("tool_calls") or []

            if not tool_calls:
                content = msg.get("content") or ""
                yield _sse_data(
                    _client_chunk(
                        completion_id=completion_id,
                        model=model,
                        created=created,
                        delta={"role": "assistant", "content": content},
                        finish_reason="stop",
                    )
                )
                yield b"data: [DONE]\n\n"
                return

            async for trace_chunk in _stream_tool_execution_trace(
                service,
                completion_id=completion_id,
                model=model,
                created=created,
                messages=messages,
                assistant_content=msg.get("content"),
                tool_calls=_stable_tool_calls(tool_calls),
            ):
                yield trace_chunk
            continue

        tool_calls = _normalized_tool_calls(tool_calls_by_index)
        if finish_reason == "tool_calls" and not tool_calls:
            raise HTTPException(
                status_code=502,
                detail="Upstream indicated tool_calls but did not provide tool call payloads in stream",
            )

        if tool_calls:
            async for trace_chunk in _stream_tool_execution_trace(
                service,
                completion_id=completion_id,
                model=model,
                created=created,
                messages=messages,
                assistant_content="".join(assistant_content_parts) or None,
                tool_calls=_stable_tool_calls(tool_calls),
            ):
                yield trace_chunk
            continue

        # Final round: only forward assistant content chunks.
        emitted_answer = False
        for chunk in buffered_content_chunks:
            out = _rewrite_chunk_identity(chunk, completion_id=completion_id, model=model)
            choice = _pick_primary_choice(out) or {}
            delta = choice.get("delta") or {}
            if isinstance(delta.get("content"), str):
                emitted_answer = True
                yield _sse_data(out)

        if not emitted_answer:
            content = "".join(assistant_content_parts).strip()
            if content:
                yield _sse_data(
                    _client_chunk(
                        completion_id=completion_id,
                        model=model,
                        created=created,
                        delta={"role": "assistant", "content": content},
                        finish_reason="stop",
                    )
                )

        yield b"data: [DONE]\n\n"
        return

    raise HTTPException(status_code=400, detail="Maximum tool loop iterations reached")


def create_app(config_path: str | None = None) -> FastAPI:
    """Create and configure the FastAPI application instance."""
    if _CORE_IMPORT_ERROR is not None:
        missing = _CORE_IMPORT_ERROR.name or "unknown"
        raise RuntimeError(
            f"Missing dependency '{missing}'. Install runtime dependencies in your active environment."
        ) from _CORE_IMPORT_ERROR
    if _FASTAPI_IMPORT_ERROR is not None:
        raise RuntimeError(
            "Missing dependency 'fastapi'. Install runtime dependencies in your active environment."
        ) from _FASTAPI_IMPORT_ERROR

    cfg = load_config(config_path)
    setup_logging(cfg.logging)
    service = GatewayService(cfg)

    config_file = Path(config_path or os.getenv("WERKZEUGKOPPLER_CONFIG") or DEFAULT_CONFIG_PATH)
    config_mtime: float | None = config_file.stat().st_mtime if config_file.exists() else None
    reload_task: asyncio.Task[None] | None = None

    async def config_reload_loop() -> None:
        """Poll config file mtime and hot-reload on changes."""
        nonlocal config_mtime
        while True:
            await asyncio.sleep(2.0)
            try:
                mtime = config_file.stat().st_mtime if config_file.exists() else None
                if mtime is None or mtime == config_mtime:
                    continue

                LOG.info("Configuration change detected at %s, reloading...", config_file)
                new_cfg = load_config(str(config_file))
                setup_logging(new_cfg.logging)
                async with service._op_lock:
                    await service.reload(new_cfg)
                config_mtime = mtime
                LOG.info("Configuration reloaded successfully")
            except Exception as exc:
                LOG.warning("Configuration reload failed, keeping current config: %s", exc)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        """Application startup/shutdown lifecycle."""
        nonlocal reload_task
        await service.start()
        reload_task = asyncio.create_task(config_reload_loop())
        try:
            yield
        finally:
            if reload_task:
                reload_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await reload_task
            await service.close()

    app = FastAPI(title="werkzeugkoppler", version="0.1.0", lifespan=lifespan)

    @app.get("/healthz")
    async def healthz() -> JSONResponse:
        """Return service and tool backend health status."""
        async with service._op_lock:
            health = await service.registry.get_health()
        return JSONResponse(
            {
                "service": "werkzeugkoppler",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                **health,
            }
        )

    @app.get("/v1/models")
    async def v1_models(request: Request) -> JSONResponse:
        """OpenAI-compatible model listing endpoint."""
        try:
            _require_gateway_auth(request, service.cfg)
            async with service._op_lock:
                models = await service.list_models()
            return JSONResponse(models)
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"Failed to fetch models: {exc}") from exc

    @app.post("/v1/chat/completions")
    async def v1_chat_completions(request: Request):
        """OpenAI-compatible chat completions endpoint."""
        _require_gateway_auth(request, service.cfg)
        payload = await request.json()
        client_host = getattr(getattr(request, "client", None), "host", None)
        LOG.debug(
            "incoming chat.completions request client=%s payload=%s",
            client_host,
            to_bounded_json(payload),
        )

        stream = bool(payload.get("stream"))
        try:
            if stream:
                return StreamingResponse(service.stream_chat(payload), media_type="text/event-stream")

            async with service._op_lock:
                resolved = await service.resolve_chat_non_stream(payload, status_queue=None)
            return JSONResponse(resolved["response"])
        except MCPError as exc:
            raise HTTPException(status_code=502, detail=f"MCP error: {exc}") from exc
        except Exception as exc:
            if exc.__class__.__name__ == "HTTPStatusError":
                raise HTTPException(status_code=502, detail=f"Upstream HTTP error: {exc}") from exc
            LOG.exception("chat/completions failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return app


def main() -> None:
    """CLI entry point that validates environment and runs uvicorn."""

    def fail(message: str, exit_code: int = 2) -> None:
        """Print startup error and terminate process."""
        print(f"ERROR: {message}", file=sys.stderr)
        raise SystemExit(exit_code)

    parser = argparse.ArgumentParser(description="werkzeugkoppler gateway")
    parser.add_argument("--config", default=None, help="Path to config YAML")
    args = parser.parse_args()

    if _CORE_IMPORT_ERROR is not None:
        missing = _CORE_IMPORT_ERROR.name or "unknown"
        fail(f"Missing dependency '{missing}'. Install runtime dependencies in your active environment.")
    if _FASTAPI_IMPORT_ERROR is not None:
        fail("Missing dependency 'fastapi'. Install runtime dependencies in your active environment.")

    try:
        from pydantic import ValidationError
    except ModuleNotFoundError:
        fail("Missing dependency 'pydantic'. Install runtime dependencies in your active environment.")

    try:
        import uvicorn
    except ModuleNotFoundError:
        fail("Missing dependency 'uvicorn'. Install runtime dependencies in your active environment.")

    try:
        assert load_config is not None
        cfg = load_config(args.config)
    except ValidationError as exc:
        missing = []
        for err in exc.errors():
            if err.get("type") == "missing":
                location = ".".join(str(x) for x in err.get("loc", []))
                missing.append(location)
        if missing:
            fail(
                "Configuration incomplete. Missing required fields: "
                + ", ".join(sorted(set(missing)))
                + ". Provide --config <file> or set env vars "
                + "(WERKZEUGKOPPLER_UPSTREAM_BASE_URL)."
            )
        fail(f"Invalid configuration: {exc}")
    except Exception as exc:
        fail(f"Failed to load configuration: {exc}")

    try:
        app = create_app(args.config)
    except RuntimeError as exc:
        fail(str(exc))
    except Exception as exc:
        fail(f"Failed to create app: {exc}")

    try:
        host, port = _service_bind_addr(cfg.service_base_url)
        uvicorn.run(app, host=host, port=port)
    except Exception as exc:
        fail(f"Server failed to start: {exc}", exit_code=1)


if __name__ == "__main__":
    main()
