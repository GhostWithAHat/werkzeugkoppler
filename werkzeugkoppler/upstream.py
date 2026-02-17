"""Client wrapper for upstream OpenAI-compatible APIs."""

from __future__ import annotations

import json
import logging
from typing import Any, AsyncGenerator

import httpx

from .config import GatewayConfig
from .utils import to_bounded_json

LOG = logging.getLogger(__name__)


class UpstreamClient:
    """Thin async HTTP client for upstream model endpoints."""

    def __init__(self, cfg: GatewayConfig) -> None:
        """Create an upstream client from gateway configuration."""
        self.cfg = cfg
        base_url = cfg.upstream_base_url.rstrip("/")
        timeout = httpx.Timeout(connect=10.0, read=300.0, write=120.0, pool=10.0)
        self._client = httpx.AsyncClient(base_url=base_url, timeout=timeout)

    async def close(self) -> None:
        """Close underlying HTTP resources."""
        await self._client.aclose()

    def _headers(self) -> dict[str, str]:
        """Build authorization headers for upstream calls."""
        headers = {"Content-Type": "application/json"}
        if self.cfg.upstream_api_key:
            headers["Authorization"] = f"Bearer {self.cfg.upstream_api_key}"
        return headers

    async def list_models(self) -> dict[str, Any]:
        """Forward `/v1/models` to upstream."""
        LOG.debug("forwarding upstream request method=GET path=/v1/models")
        response = await self._client.get("/v1/models", headers=self._headers())
        response.raise_for_status()
        return response.json()

    async def chat_completion(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Run non-streaming upstream chat completion."""
        LOG.debug(
            "forwarding upstream request method=POST path=/v1/chat/completions stream=false payload=%s",
            to_bounded_json(payload),
        )
        response = await self._client.post("/v1/chat/completions", headers=self._headers(), json=payload)
        response.raise_for_status()
        return response.json()

    async def stream_chat_completion(self, payload: dict[str, Any]) -> AsyncGenerator[dict[str, Any], None]:
        """Run streaming upstream chat completion and yield decoded chunk objects."""
        req_payload = dict(payload)
        req_payload["stream"] = True
        LOG.debug(
            "forwarding upstream request method=POST path=/v1/chat/completions stream=true payload=%s",
            to_bounded_json(req_payload),
        )
        async with self._client.stream(
            "POST",
            "/v1/chat/completions",
            headers=self._headers(),
            json=req_payload,
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    yield json.loads(data)
                except json.JSONDecodeError:
                    # Tolerate occasional non-JSON lines in malformed streams.
                    continue
