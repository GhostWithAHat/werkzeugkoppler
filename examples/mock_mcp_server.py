from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI(title="mock-mcp")


@app.post("/mcp")
async def mcp(request: Request) -> JSONResponse:
    payload = await request.json()
    method = payload.get("method")
    req_id = payload.get("id")
    params: dict[str, Any] = payload.get("params") or {}

    if method == "tools/list":
        result = {
            "tools": [
                {
                    "name": "add",
                    "description": "Adds two numbers",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "a": {"type": "number"},
                            "b": {"type": "number"},
                        },
                        "required": ["a", "b"],
                    },
                }
            ],
            "nextCursor": None,
        }
    elif method == "tools/call":
        if params.get("name") != "add":
            return JSONResponse(
                {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32602, "message": "unknown tool"}},
                status_code=200,
            )
        args = params.get("arguments") or {}
        a = float(args.get("a", 0))
        b = float(args.get("b", 0))
        result = {
            "content": [{"type": "text", "text": str(a + b)}],
            "structuredContent": {"sum": a + b},
            "isError": False,
        }
    else:
        return JSONResponse(
            {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32601, "message": "method not found"}},
            status_code=200,
        )

    return JSONResponse({"jsonrpc": "2.0", "id": req_id, "result": result})
