"""Helpers for `/v1/chat/completions` endpoint handling."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncGenerator

try:
    from fastapi import HTTPException, Request
    from fastapi.responses import JSONResponse, StreamingResponse
except ModuleNotFoundError:  # pragma: no cover
    HTTPException = Exception  # type: ignore[assignment]
    Request = Any  # type: ignore[assignment]
    JSONResponse = Any  # type: ignore[assignment]
    StreamingResponse = Any  # type: ignore[assignment]

try:
    from .mcp_client import MCPError
except ModuleNotFoundError:  # pragma: no cover
    MCPError = Exception  # type: ignore[assignment]

from .direct_commands import (
    DEFAULT_DIRECT_COMMAND_MAX_OUTPUT_BYTES,
    DEFAULT_DIRECT_COMMAND_MAX_OUTPUT_LINES,
    direct_command_name,
    direct_command_output_limit_notice,
    extract_latest_user_direct_command,
    format_direct_command_output,
    is_allowed_direct_command,
    run_direct_command,
    stream_direct_command_output,
    strip_direct_command_messages,
    direct_command_output_prefix,
    direct_command_output_suffix,
)
from .stream_chunks import client_chunk

LOG = logging.getLogger(__name__)


def sse_data(payload: dict[str, Any]) -> bytes:
    """Encode one SSE `data:` event."""
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8")


def sse_comment(text: str) -> bytes:
    """Encode one SSE comment/heartbeat event."""
    return f": {text}\n\n".encode("utf-8")


def build_sse_response(stream: AsyncGenerator[bytes, None]) -> StreamingResponse:
    """Build standard SSE response with consistent proxy-safe headers."""
    return StreamingResponse(
        stream,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def build_openai_error_payload(message: str, *, code: str) -> dict[str, Any]:
    """Build OpenAI-style error response payload."""
    return {
        "error": {
            "message": message,
            "type": "invalid_request_error",
            "code": code,
        }
    }


def build_direct_command_chat_response(*, model: str, content: str) -> dict[str, Any]:
    """Build OpenAI-compatible non-streaming chat response payload."""
    created = int(datetime.now(timezone.utc).timestamp())
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
    }


def build_direct_command_stream_events(*, model: str, content: str) -> list[bytes]:
    """Build one-shot SSE payload for direct-command messages."""
    created = int(datetime.now(timezone.utc).timestamp())
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    chunk = client_chunk(
        completion_id=completion_id,
        model=model,
        created=created,
        delta={"role": "assistant", "content": content},
        finish_reason="stop",
    )
    return [sse_data(chunk), b"data: [DONE]\n\n"]


async def stream_direct_command_events(
    *,
    shell_code: str,
    model: str,
    request: Request | None,
    cwd: Path | None = None,
    max_output_lines: int = DEFAULT_DIRECT_COMMAND_MAX_OUTPUT_LINES,
    max_output_bytes: int = DEFAULT_DIRECT_COMMAND_MAX_OUTPUT_BYTES,
) -> AsyncGenerator[bytes, None]:
    """Stream direct-command output as SSE and abort process on client disconnect."""
    created = int(datetime.now(timezone.utc).timestamp())
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"

    def _chunk(content: str, *, finish_reason: str | None, include_role: bool = False) -> bytes:
        delta: dict[str, Any] = {"content": content}
        if include_role:
            delta["role"] = "assistant"
        return sse_data(
            client_chunk(
                completion_id=completion_id,
                model=model,
                created=created,
                delta=delta,
                finish_reason=finish_reason,
            )
        )

    yield _chunk(
        direct_command_output_prefix(shell_code),
        finish_reason=None,
        include_role=True,
    )

    async def _is_disconnected() -> bool:
        return bool(request is not None and await request.is_disconnected())

    async for event in stream_direct_command_output(
        shell_code,
        cwd=cwd,
        is_disconnected=_is_disconnected if request is not None else None,
        max_output_lines=max_output_lines,
        max_output_bytes=max_output_bytes,
    ):
        if event.is_final:
            LOG.info(
                "direct command finished stream=true return_code=%s limit_reason=%s command=%s",
                int(event.return_code or 0),
                event.output_limit_reason or "-",
                shell_code,
            )
            suffix = direct_command_output_suffix(shell_code, return_code=int(event.return_code or 0))
            limit_note = direct_command_output_limit_notice(event.output_limit_reason)
            if limit_note:
                suffix = f"{suffix}\n\n{limit_note}"
            yield _chunk(suffix, finish_reason="stop")
            yield b"data: [DONE]\n\n"
            return
        if event.text:
            yield _chunk(event.text, finish_reason=None)


def strip_direct_command_messages_in_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Remove direct-command messages from payload history before model forwarding."""
    payload_messages = payload.get("messages")
    if not isinstance(payload_messages, list) or not payload_messages:
        return payload
    updated_payload = dict(payload)
    updated_payload["messages"] = strip_direct_command_messages(payload_messages)
    return updated_payload


async def stream_with_keepalive(
    source: AsyncGenerator[bytes, None],
    *,
    keepalive_seconds: float,
    request: Request | None = None,
) -> AsyncGenerator[bytes, None]:
    """Forward stream chunks and emit periodic SSE heartbeats while waiting."""
    started = time.monotonic()
    try:
        emit_keepalive = keepalive_seconds > 0
        poll_seconds = keepalive_seconds if emit_keepalive else 0.5

        iterator = source.__aiter__()
        while True:
            next_item = asyncio.create_task(iterator.__anext__())
            try:
                while not next_item.done():
                    if request is not None and await request.is_disconnected():
                        LOG.debug(
                            "client disconnected, stopping stream elapsed=%.3fs",
                            time.monotonic() - started,
                        )
                        next_item.cancel()
                        with contextlib.suppress(asyncio.CancelledError):
                            await next_item
                        return
                    done, _ = await asyncio.wait({next_item}, timeout=poll_seconds)
                    if done:
                        break
                    if request is not None and await request.is_disconnected():
                        LOG.debug(
                            "client disconnected, stopping stream elapsed=%.3fs",
                            time.monotonic() - started,
                        )
                        next_item.cancel()
                        with contextlib.suppress(asyncio.CancelledError):
                            await next_item
                        return
                    if emit_keepalive:
                        yield sse_comment("keepalive")
                yield next_item.result()
            except StopAsyncIteration:
                return
            except asyncio.CancelledError:
                if not next_item.done():
                    next_item.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await next_item
                raise
            except Exception:
                if not next_item.done():
                    next_item.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await next_item
                raise
    finally:
        cleanup_cancelled = False
        try:
            await asyncio.shield(source.aclose())
        except asyncio.CancelledError:
            cleanup_cancelled = True
        except Exception:
            pass
        LOG.debug("stream wrapper closed elapsed=%.3fs", time.monotonic() - started)
        if cleanup_cancelled:
            raise asyncio.CancelledError


async def handle_direct_command_short_circuit(
    *,
    request: Request,
    payload: dict[str, Any],
    service: Any,
) -> JSONResponse | StreamingResponse | None:
    """Handle direct command request end-to-end without model forwarding."""
    payload_messages = payload.get("messages")
    if not isinstance(payload_messages, list) or not payload_messages:
        return None

    direct_command = extract_latest_user_direct_command(payload_messages)
    if direct_command is None:
        return None
    stream_requested = bool(payload.get("stream"))
    LOG.info("direct command requested stream=%s command=%s", stream_requested, direct_command)

    model_name = str(payload.get("model") or service._fallback_offline_model_name())
    if not is_allowed_direct_command(direct_command, service.cfg.allowed_direct_commands):
        command_name = direct_command_name(direct_command) or direct_command
        LOG.warning("direct command rejected command=%s", command_name)
        allowed_commands_text = "\n".join(service.cfg.allowed_direct_commands)
        error_message = (
            f"`{command_name}` is not an allowed direct command. Allowed commands are:\n"
            f"{allowed_commands_text}"
        )
        if stream_requested:
            events = build_direct_command_stream_events(
                model=model_name,
                content=f"Error: {error_message}",
            )

            async def direct_stream_error() -> AsyncGenerator[bytes, None]:
                for event in events:
                    yield event

            return build_sse_response(direct_stream_error())

        return JSONResponse(
            build_openai_error_payload(
                error_message,
                code="direct_command_not_allowed",
            ),
            status_code=400,
        )

    if stream_requested:

        async def direct_stream() -> AsyncGenerator[bytes, None]:
            async for event in stream_direct_command_events(
                shell_code=direct_command,
                model=model_name,
                request=request,
                cwd=Path.cwd(),
                max_output_lines=int(
                    service.cfg.allowed_direct_commands_max_output_lines
                    or DEFAULT_DIRECT_COMMAND_MAX_OUTPUT_LINES
                ),
                max_output_bytes=int(
                    service.cfg.allowed_direct_commands_max_output_bytes
                    or DEFAULT_DIRECT_COMMAND_MAX_OUTPUT_BYTES
                ),
            ):
                yield event

        return build_sse_response(direct_stream())

    command_output, return_code, output_limit_reason = run_direct_command(
        direct_command,
        cwd=Path.cwd(),
        max_output_lines=int(
            service.cfg.allowed_direct_commands_max_output_lines or DEFAULT_DIRECT_COMMAND_MAX_OUTPUT_LINES
        ),
        max_output_bytes=int(
            service.cfg.allowed_direct_commands_max_output_bytes or DEFAULT_DIRECT_COMMAND_MAX_OUTPUT_BYTES
        ),
    )
    LOG.info(
        "direct command finished stream=false return_code=%s limit_reason=%s command=%s",
        return_code,
        output_limit_reason or "-",
        direct_command,
    )
    content = format_direct_command_output(direct_command, command_output, return_code)
    limit_note = direct_command_output_limit_notice(output_limit_reason)
    if limit_note:
        content = f"{content}\n\n{limit_note}"
    return JSONResponse(build_direct_command_chat_response(model=model_name, content=content))


async def handle_regular_chat_request(
    *,
    request: Request,
    payload: dict[str, Any],
    service: Any,
) -> JSONResponse | StreamingResponse:
    """Handle regular model-backed chat request (streaming or non-streaming)."""
    stream = bool(payload.get("stream"))
    try:
        if stream:
            base_stream = service.stream_chat(payload)
            keepalive_stream = stream_with_keepalive(
                base_stream,
                keepalive_seconds=service.cfg.stream_keepalive_seconds or 0.0,
                request=request,
            )
            return build_sse_response(keepalive_stream)

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
