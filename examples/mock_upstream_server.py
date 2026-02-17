from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

app = FastAPI(title="mock-upstream")


@app.get("/v1/models")
async def models() -> JSONResponse:
    return JSONResponse(
        {
            "object": "list",
            "data": [{"id": "demo-model", "object": "model", "created": int(time.time()), "owned_by": "mock"}],
        }
    )


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    payload = await request.json()
    messages: list[dict[str, Any]] = payload.get("messages") or []
    tools: list[dict[str, Any]] = payload.get("tools") or []
    model = payload.get("model") or "demo-model"
    stream = bool(payload.get("stream"))

    last_user = next((m for m in reversed(messages) if m.get("role") == "user"), {})
    last_tool = next((m for m in reversed(messages) if m.get("role") == "tool"), None)

    if last_tool is None and "add" in (last_user.get("content") or "").lower() and tools:
        tool_name = tools[0].get("function", {}).get("name", "demo__add")
        msg = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": f"call_{uuid.uuid4().hex}",
                    "type": "function",
                    "function": {"name": tool_name, "arguments": json.dumps({"a": 2, "b": 3})},
                }
            ],
            "reasoning": "Need calculator tool for exact result.",
        }
    elif last_tool is not None:
        msg = {
            "role": "assistant",
            "content": f"Ergebnis aus Tool: {last_tool.get('content')}",
        }
    else:
        msg = {"role": "assistant", "content": "Keine Tools notwendig."}

    base = {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(datetime.now(timezone.utc).timestamp()),
        "model": model,
        "choices": [{"index": 0, "message": msg, "finish_reason": "stop"}],
    }

    if not stream:
        return JSONResponse(base)

    async def gen():
        content = msg.get("content") or ""
        chunk = {
            "id": base["id"],
            "object": "chat.completion.chunk",
            "created": base["created"],
            "model": model,
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
        }
        yield f"data: {json.dumps(chunk)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")
