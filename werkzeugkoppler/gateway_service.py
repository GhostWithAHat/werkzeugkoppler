"""Gateway service runtime and chat/tool orchestration."""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any, AsyncGenerator

import httpx

try:
    from fastapi import HTTPException
except ModuleNotFoundError:  # pragma: no cover
    HTTPException = Exception  # type: ignore[assignment]

from .chat_handlers import sse_data as _sse_data
from .config import GatewayConfig
from .message_preparation import prepare_messages as _prepare_messages
from .stream_chunks import (
    append_tool_call_delta as _append_tool_call_delta,
    chunk_with_content_delta as _chunk_with_content_delta,
    client_chunk as _client_chunk,
    compact_json as _compact_json,
    content_to_thinking_value as _content_to_thinking_value,
    normalized_tool_calls as _normalized_tool_calls,
    passthrough_non_content_delta_chunk as _passthrough_non_content_delta_chunk,
    pick_primary_choice as _pick_primary_choice,
    reasoning_chunk as _reasoning_chunk,
    rewrite_chunk_identity as _rewrite_chunk_identity,
    split_content_for_stream as _split_content_for_stream,
    with_single_primary_choice as _with_single_primary_choice,
)
from .tool_registry import ToolRegistry
from .upstream import UpstreamClient

LOG = logging.getLogger(__name__)

_TRACE_TOOL_CALL_PREFIX = "==> tool call"
_TRACE_TOOL_RESULT_PREFIX = "==> tool result"
_CONTENT_MIRROR_HEADER = "==> Preparation done. Preview of final assistant.content:"
_CONTENT_MIRROR_MARKER = f"\n\n---\n\n\n{_CONTENT_MIRROR_HEADER}\n\n"
_DEFAULT_FALLBACK_FAKE_MODEL_NAME = "werkzeugkoppler"


def _tool_trace_line(status: dict[str, Any]) -> str | None:
    """Convert tool status events to compact reasoning trace lines."""
    event_type = status.get("type")
    if event_type == "tool_start":
        payload = {"name": status.get("tool"), "arguments": status.get("arguments")}
        return f"\n\n{_TRACE_TOOL_CALL_PREFIX} `{_compact_json(payload)}`\n"
    if event_type == "tool_end":
        if status.get("ok"):
            payload: Any = status.get("result")
        else:
            payload = {"error": status.get("error"), "is_error": True}
        return f"{_TRACE_TOOL_RESULT_PREFIX} `{_compact_json(payload)}`\n\n"
    return None


def _upstream_error_text_from_exception(exc: Exception) -> str:
    """Map upstream exceptions to compact user-facing failure text."""
    text = str(exc).lower()
    if (
        "response payload is not completed" in text
        or "transferencodingerror" in text
        or "not enough data to satisfy transfer length header" in text
    ):
        return "Upstream LLM response payload is not completed"
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code if exc.response is not None else None
        if status == 404:
            return "Upstream LLM model not found"
    return "Connection to upstream LLM failed"


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
        self._last_active_model: str | None = None
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

    def _effective_fake_model_name(self) -> str:
        """Resolve configured fallback fake model name with stable default."""
        configured = (self.cfg.fallback_fake_model_name or "").strip()
        return configured or _DEFAULT_FALLBACK_FAKE_MODEL_NAME

    def _fallback_offline_model_name(self) -> str:
        """Return model name used when upstream is unavailable."""
        if self._last_active_model:
            return self._last_active_model
        return self._effective_fake_model_name()

    def _update_last_active_model(self, model: Any) -> None:
        """Remember last successfully active model id."""
        if isinstance(model, str):
            normalized = model.strip()
            if normalized:
                self._last_active_model = normalized

    @staticmethod
    def _extract_first_model_id(models_payload: dict[str, Any]) -> str | None:
        """Extract first model id from OpenAI-style `/v1/models` payload."""
        data = models_payload.get("data")
        if not isinstance(data, list):
            return None
        for entry in data:
            if isinstance(entry, dict) and entry.get("id"):
                return str(entry["id"]).strip() or None
        return None

    async def stream_chat(self, request_payload: dict[str, Any]) -> AsyncGenerator[bytes, None]:
        """Run one streaming chat request under the service operation lock."""
        async with self._op_lock:
            async for chunk in _stream_chat(self, request_payload):
                yield chunk

    async def list_models(self) -> dict[str, Any]:
        """Fetch models from upstream with fallback to configured default model."""
        try:
            models = await self.upstream.list_models()
            first_model_id = self._extract_first_model_id(models)
            if first_model_id:
                self._update_last_active_model(first_model_id)
            return models
        except Exception as exc:
            fallback_model = self._fallback_offline_model_name()
            LOG.warning("Upstream /v1/models failed, fallback model=%s error=%s", fallback_model, exc)
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
            first_model_id = self._extract_first_model_id(models)
            if first_model_id:
                self._resolved_upstream_default_model = first_model_id
                self._update_last_active_model(first_model_id)
                LOG.info(
                    "Selected upstream default model dynamically model=%s",
                    self._resolved_upstream_default_model,
                )
                return self._resolved_upstream_default_model
        except Exception as exc:
            LOG.warning("Could not auto-detect upstream model: %s", exc)
        fallback_model = self._fallback_offline_model_name()
        LOG.info("Using offline fallback model=%s for default model resolution", fallback_model)
        return fallback_model

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
        messages = _prepare_messages(
            messages,
            self.cfg.init_messages,
            self.cfg.last_user_message_readers,
        )

        mcp_tools = await self.registry.get_openai_tools()
        combined_tools = self._merge_tools(user_tools, mcp_tools)

        for _ in range(max(1, self.cfg.max_tool_loops)):
            upstream_payload = {
                **base_payload,
                "messages": messages,
                "tools": combined_tools,
                "stream": False,
            }
            try:
                upstream_response = await self.upstream.chat_completion(upstream_payload)
            except Exception as exc:
                if self.upstream._is_retryable_connect_error(exc):
                    fail_message = f"{_upstream_error_text_from_exception(exc)}. Retries exhausted."
                    return {
                        "mode": "final",
                        "response": {
                            "id": f"chatcmpl-{uuid.uuid4().hex}",
                            "object": "chat.completion",
                            "created": int(datetime.now(timezone.utc).timestamp()),
                            "model": str(base_payload.get("model") or self._fallback_offline_model_name()),
                            "choices": [
                                {
                                    "index": 0,
                                    "message": {"role": "assistant", "content": fail_message},
                                    "finish_reason": "stop",
                                }
                            ],
                        },
                        "messages": messages,
                        "base_payload": base_payload,
                    }
                raise
            self._update_last_active_model(upstream_response.get("model") or base_payload.get("model"))
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
                    value=line,
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
                    value=line,
                )
            )

    messages.extend(await tool_task)


class _RetryLineTracker:
    """Compact repeated retry notices into a single evolving reasoning line."""

    def __init__(self) -> None:
        self._active_error_text: str | None = None

    def next_retry_event(self, retry_reason: str) -> tuple[bool, str]:
        error_text = _retry_error_text(retry_reason)
        if self._active_error_text == error_text:
            return False, "."
        needs_flush = self._active_error_text is not None
        self._active_error_text = error_text
        return needs_flush, f"{error_text}. Retrying "

    async def flush(self, *, completion_id: str, model: str, created: int) -> AsyncGenerator[bytes, None]:
        if self._active_error_text is not None:
            yield _sse_data(
                _reasoning_chunk(
                    completion_id=completion_id,
                    model=model,
                    created=created,
                    value="\n",
                )
            )
            self._active_error_text = None


def _retry_error_text(retry_reason: str) -> str:
    if retry_reason == "payload_incomplete":
        return "Upstream LLM response payload is not completed"
    if retry_reason == "model_not_found":
        return "Upstream LLM model not found"
    return "Connection to upstream LLM failed"


async def _emit_exhausted_retry_failure(
    *,
    completion_id: str,
    model: str,
    created: int,
    error_text: str,
) -> AsyncGenerator[bytes, None]:
    fail_message = f"{error_text}."
    yield _sse_data(
        _reasoning_chunk(
            completion_id=completion_id,
            model=model,
            created=created,
            value=f"{fail_message} Retries exhausted.\n",
        )
    )
    yield _sse_data(
        _client_chunk(
            completion_id=completion_id,
            model=model,
            created=created,
            delta={"role": "assistant", "content": f"{fail_message} Retries exhausted."},
            finish_reason="stop",
        )
    )
    yield b"data: [DONE]\n\n"


async def _emit_midstream_upstream_failure(
    *,
    completion_id: str,
    model: str,
    created: int,
    error_text: str,
) -> AsyncGenerator[bytes, None]:
    """Close stream gracefully when upstream drops after partial chunks."""
    yield _sse_data(
        _reasoning_chunk(
            completion_id=completion_id,
            model=model,
            created=created,
            value=f"\n==> {error_text}. Upstream stream interrupted.\n",
        )
    )
    yield _sse_data(
        _client_chunk(
            completion_id=completion_id,
            model=model,
            created=created,
            delta={},
            finish_reason="stop",
        )
    )
    yield b"data: [DONE]\n\n"


async def _resolve_fallback_non_stream_round(
    *,
    service: GatewayService,
    upstream_payload: dict[str, Any],
    model: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    # Fallback for upstreams that ignore streaming.
    fallback_payload = dict(upstream_payload)
    fallback_payload["stream"] = False
    fallback_response = await service.upstream.chat_completion(fallback_payload)
    service._update_last_active_model(fallback_response.get("model") or model)
    msg = ((fallback_response.get("choices") or [{}])[0]).get("message") or {}
    tool_calls = msg.get("tool_calls") or []
    return msg, tool_calls


async def _emit_final_stream_answer(
    *,
    completion_id: str,
    model: str,
    created: int,
    safe_preview_mode: bool,
    buffered_content_chunks: list[dict[str, Any]],
    content_emitted_live: bool,
    assistant_content_parts: list[str],
) -> AsyncGenerator[bytes, None]:
    # Final round: only forward buffered assistant content chunks in safe_preview mode.
    emitted_answer = False
    if safe_preview_mode:
        for chunk in buffered_content_chunks:
            out = _rewrite_chunk_identity(chunk, completion_id=completion_id, model=model)
            choice = _pick_primary_choice(out) or {}
            delta = choice.get("delta") or {}
            if "content" in delta:
                emitted_answer = True
                yield _sse_data(out)
    else:
        emitted_answer = content_emitted_live

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


async def _stream_chat(
    service: GatewayService,
    request_payload: dict[str, Any],
) -> AsyncGenerator[bytes, None]:
    """Run streaming chat orchestration with tool loops.

    Stream behavior is controlled via `stream_answer_mode`:
    - `live`: forward assistant content deltas immediately.
    - `safe_preview`: buffer assistant content until the round is complete and
      mirror interim content into reasoning for user-visible progress.
    """
    base_payload, messages, user_tools = _build_base_payload(request_payload, service.cfg.upstream_default_model)
    if not base_payload.get("model"):
        base_payload["model"] = await service.resolve_default_model()
    messages = _prepare_messages(
        messages,
        service.cfg.init_messages,
        service.cfg.last_user_message_readers,
    )

    mcp_tools = await service.registry.get_openai_tools()
    combined_tools = service._merge_tools(user_tools, mcp_tools)

    base_payload["n"] = 1
    model = str(base_payload["model"])
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(datetime.now(timezone.utc).timestamp())
    stream_started = time.monotonic()
    LOG.debug("chat stream start completion_id=%s model=%s", completion_id, model)

    for _ in range(max(1, service.cfg.max_tool_loops)):
        answer_mode = service.cfg.stream_answer_mode or "live"
        safe_preview_mode = answer_mode == "safe_preview"
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
        content_mirror_marker_sent = False
        content_emitted_live = False
        model_marked_active = False
        retry_line_tracker = _RetryLineTracker()

        try:
            async for chunk in service.upstream.stream_chat_completion(
                upstream_payload,
                trace_id=f"{completion_id}:round{_ + 1}",
            ):
                retry_notice = chunk.get("_wk_retry_notice")
                if isinstance(retry_notice, dict):
                    retry_reason = str(retry_notice.get("retry_reason") or "connection_failed")
                    needs_flush, retry_message = retry_line_tracker.next_retry_event(retry_reason)
                    if needs_flush:
                        async for flush_chunk in retry_line_tracker.flush(
                            completion_id=completion_id,
                            model=model,
                            created=created,
                        ):
                            yield flush_chunk
                        _, retry_message = retry_line_tracker.next_retry_event(retry_reason)
                    yield _sse_data(
                        _reasoning_chunk(
                            completion_id=completion_id,
                            model=model,
                            created=created,
                            value=retry_message,
                        )
                    )
                    continue
                async for flush_chunk in retry_line_tracker.flush(
                    completion_id=completion_id,
                    model=model,
                    created=created,
                ):
                    yield flush_chunk
                got_chunks = True
                if not model_marked_active:
                    service._update_last_active_model(chunk.get("model") or model)
                    model_marked_active = True
                single_chunk = _with_single_primary_choice(chunk)
                choice0 = _pick_primary_choice(single_chunk) or {}
                finish_reason = choice0.get("finish_reason") or finish_reason
                delta = choice0.get("delta") or {}

                passthrough_chunk = _passthrough_non_content_delta_chunk(
                    single_chunk,
                    completion_id=completion_id,
                    model=model,
                )
                if passthrough_chunk is not None:
                    yield _sse_data(passthrough_chunk)

                reasoning_content, final_content = _split_content_for_stream(delta.get("content"))
                if reasoning_content is not None:
                    reasoning_chunk = _chunk_with_content_delta(
                        single_chunk,
                        completion_id=completion_id,
                        model=model,
                        content_value=reasoning_content,
                        finish_reason=None,
                    )
                    if reasoning_chunk is not None:
                        yield _sse_data(reasoning_chunk)

                if final_content is not None:
                    if safe_preview_mode:
                        mirrored_thinking = _content_to_thinking_value(final_content)
                        if mirrored_thinking:
                            if not content_mirror_marker_sent:
                                yield _sse_data(
                                    _reasoning_chunk(
                                        completion_id=completion_id,
                                        model=model,
                                        created=created,
                                        value=_CONTENT_MIRROR_MARKER,
                                    )
                                )
                                content_mirror_marker_sent = True
                            yield _sse_data(
                                _reasoning_chunk(
                                    completion_id=completion_id,
                                    model=model,
                                    created=created,
                                    value=mirrored_thinking,
                                )
                            )
                    if isinstance(final_content, str):
                        assistant_content_parts.append(final_content)
                    final_content_chunk = _chunk_with_content_delta(
                        single_chunk,
                        completion_id=completion_id,
                        model=model,
                        content_value=final_content,
                        finish_reason=finish_reason,
                    )
                    if final_content_chunk is not None:
                        if safe_preview_mode:
                            buffered_content_chunks.append(final_content_chunk)
                        else:
                            content_emitted_live = True
                            yield _sse_data(final_content_chunk)

                delta_tool_calls = delta.get("tool_calls")
                if isinstance(delta_tool_calls, list):
                    _append_tool_call_delta(tool_calls_by_index, delta_tool_calls)
        except Exception as exc:
            async for flush_chunk in retry_line_tracker.flush(
                completion_id=completion_id,
                model=model,
                created=created,
            ):
                yield flush_chunk
            if not got_chunks and service.upstream._is_retryable_connect_error(exc):
                async for fail_chunk in _emit_exhausted_retry_failure(
                    completion_id=completion_id,
                    model=model,
                    created=created,
                    error_text=_upstream_error_text_from_exception(exc),
                ):
                    yield fail_chunk
                return
            if got_chunks and service.upstream._is_retryable_connect_error(exc):
                error_text = _upstream_error_text_from_exception(exc)
                LOG.warning(
                    "upstream stream interrupted after partial output completion_id=%s model=%s error=%s",
                    completion_id,
                    model,
                    exc,
                )
                async for fail_chunk in _emit_midstream_upstream_failure(
                    completion_id=completion_id,
                    model=model,
                    created=created,
                    error_text=error_text,
                ):
                    yield fail_chunk
                return
            raise

        if not got_chunks:
            async for flush_chunk in retry_line_tracker.flush(
                completion_id=completion_id,
                model=model,
                created=created,
            ):
                yield flush_chunk
            msg, tool_calls = await _resolve_fallback_non_stream_round(
                service=service,
                upstream_payload=upstream_payload,
                model=model,
            )
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

        async for response_chunk in _emit_final_stream_answer(
            completion_id=completion_id,
            model=model,
            created=created,
            safe_preview_mode=safe_preview_mode,
            buffered_content_chunks=buffered_content_chunks,
            content_emitted_live=content_emitted_live,
            assistant_content_parts=assistant_content_parts,
        ):
            yield response_chunk
        LOG.debug(
            "chat stream done completion_id=%s elapsed=%.3fs",
            completion_id,
            time.monotonic() - stream_started,
        )
        return

    raise HTTPException(status_code=400, detail="Maximum tool loop iterations reached")
