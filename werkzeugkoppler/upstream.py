"""Client wrapper for upstream OpenAI-compatible APIs."""

from __future__ import annotations

import json
import logging
import contextlib
import asyncio
import time
from typing import Any, AsyncGenerator

import httpx

from .config import GatewayConfig
from .json_helpers import to_bounded_json

LOG = logging.getLogger(__name__)


class UpstreamClient:
    """Thin async HTTP client for upstream model endpoints."""

    def __init__(self, cfg: GatewayConfig) -> None:
        """Create an upstream client from gateway configuration."""
        self.cfg = cfg
        self._base_url = cfg.upstream_base_url.rstrip("/")
        self._timeout = httpx.Timeout(connect=10.0, read=300.0, write=120.0, pool=10.0)
        self._client = httpx.AsyncClient(base_url=self._base_url, timeout=self._timeout)

    async def close(self) -> None:
        """Close underlying HTTP resources."""
        await self._client.aclose()

    def _headers(self) -> dict[str, str]:
        """Build authorization headers for upstream calls."""
        headers = {"Content-Type": "application/json"}
        if self.cfg.upstream_api_key:
            headers["Authorization"] = f"Bearer {self.cfg.upstream_api_key}"
        return headers

    def _build_client(self) -> httpx.AsyncClient:
        """Create a fresh upstream HTTP client instance."""
        return httpx.AsyncClient(base_url=self._base_url, timeout=self._timeout)

    def _upstream_connect_retries(self) -> int:
        """Return configured number of retries after first failed request."""
        retries = self.cfg.upstream_connect_retries
        if retries is None:
            return 0
        return int(retries)

    def _upstream_retry_interval_seconds(self) -> float:
        """Return configured wait time between retries in seconds."""
        interval_ms = int(self.cfg.upstream_retry_interval_ms or 0)
        return max(0.0, interval_ms / 1000.0)

    def _upstream_retry_interval_ms(self) -> int:
        """Return configured wait time between retries in milliseconds."""
        interval_ms = int(self.cfg.upstream_retry_interval_ms or 0)
        return max(0, interval_ms)

    @staticmethod
    def _is_incomplete_payload_error(exc: Exception) -> bool:
        """Detect truncated/incomplete upstream response payload errors."""
        text = str(exc)
        lowered = text.lower()
        return (
            "response payload is not completed" in lowered
            or "transferencodingerror" in lowered
            or "not enough data to satisfy transfer length header" in lowered
        )

    @classmethod
    def _is_retryable_connect_error(cls, exc: Exception) -> bool:
        """Decide whether one upstream error should trigger a retry."""
        if isinstance(exc, httpx.TransportError):
            return True
        if isinstance(exc, httpx.HTTPStatusError):
            status = exc.response.status_code if exc.response is not None else None
            if status is None:
                return False
            return status in {404, 429} or status >= 500
        if cls._is_incomplete_payload_error(exc):
            return True
        return False

    async def list_models(self) -> dict[str, Any]:
        """Forward `/v1/models` to upstream."""
        LOG.debug("forwarding upstream request method=GET path=/v1/models")
        response = await self._client.get("/v1/models", headers=self._headers())
        response.raise_for_status()
        return response.json()

    async def chat_completion(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Run non-streaming upstream chat completion."""
        retries = self._upstream_connect_retries()
        retry_delay = self._upstream_retry_interval_seconds()
        attempt = 1
        while True:
            try:
                LOG.debug(
                    "forwarding upstream request method=POST path=/v1/chat/completions stream=false attempt=%s retries=%s payload=%s",
                    attempt,
                    retries,
                    to_bounded_json(payload),
                )
                response = await self._client.post("/v1/chat/completions", headers=self._headers(), json=payload)
                response.raise_for_status()
                return response.json()
            except Exception as exc:
                can_retry = self._is_retryable_connect_error(exc) and (retries < 0 or attempt <= retries)
                if not can_retry:
                    raise
                LOG.warning(
                    "upstream chat request failed attempt=%s retries=%s retry_in=%.3fs error=%s",
                    attempt,
                    retries,
                    retry_delay,
                    exc,
                )
                if retry_delay > 0:
                    await asyncio.sleep(retry_delay)
                attempt += 1

    async def stream_chat_completion(
        self,
        payload: dict[str, Any],
        *,
        trace_id: str | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Run streaming upstream chat completion and yield decoded chunk objects."""
        req_payload = dict(payload)
        req_payload["stream"] = True
        started = time.monotonic()
        tag = trace_id or "-"
        LOG.debug(
            "upstream stream start trace=%s method=POST path=/v1/chat/completions payload=%s",
            tag,
            to_bounded_json(req_payload),
        )
        retries = self._upstream_connect_retries()
        retry_delay = self._upstream_retry_interval_seconds()

        attempt = 1
        while True:
            stream_client = self._build_client()
            response: httpx.Response | None = None
            chunk_count = 0
            try:
                headers = self._headers()
                headers["Connection"] = "close"
                response = await stream_client.send(
                    stream_client.build_request(
                        "POST",
                        "/v1/chat/completions",
                        headers=headers,
                        json=req_payload,
                    ),
                    stream=True,
                )
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        LOG.debug(
                            "upstream stream done marker trace=%s elapsed=%.3fs chunks=%s",
                            tag,
                            time.monotonic() - started,
                            chunk_count,
                        )
                        return
                    try:
                        chunk_count += 1
                        yield json.loads(data)
                    except json.JSONDecodeError:
                        # Tolerate occasional non-JSON lines in malformed streams.
                        continue
                return
            except asyncio.CancelledError:
                LOG.debug(
                    "upstream stream cancelled trace=%s elapsed=%.3fs chunks=%s",
                    tag,
                    time.monotonic() - started,
                    chunk_count,
                )
                raise
            except Exception as exc:
                can_retry = (
                    chunk_count == 0
                    and self._is_retryable_connect_error(exc)
                    and (retries < 0 or attempt <= retries)
                )
                if not can_retry:
                    raise
                retry_delay_ms = self._upstream_retry_interval_ms()
                retry_reason = "connection_failed"
                if self._is_incomplete_payload_error(exc):
                    retry_reason = "payload_incomplete"
                elif isinstance(exc, httpx.HTTPStatusError):
                    status = exc.response.status_code if exc.response is not None else None
                    if status == 404:
                        retry_reason = "model_not_found"
                yield {
                    "_wk_retry_notice": {
                        "retry_number": attempt,
                        "retry_delay_ms": retry_delay_ms,
                        "retry_reason": retry_reason,
                    }
                }
                LOG.warning(
                    "upstream stream connect failed trace=%s attempt=%s retries=%s retry_in=%.3fs error=%s",
                    tag,
                    attempt,
                    retries,
                    retry_delay,
                    exc,
                )
                if retry_delay_ms > 0:
                    await asyncio.sleep(retry_delay_ms / 1000.0)
                attempt += 1
            finally:
                cleanup_cancelled = False
                close_started = time.monotonic()
                if response is not None:
                    try:
                        await asyncio.shield(response.aclose())
                    except asyncio.CancelledError:
                        cleanup_cancelled = True
                    except Exception:
                        pass
                try:
                    await asyncio.shield(stream_client.aclose())
                except asyncio.CancelledError:
                    cleanup_cancelled = True
                except Exception:
                    pass
                LOG.debug(
                    "upstream stream closed trace=%s elapsed=%.3fs close_elapsed=%.3fs chunks=%s",
                    tag,
                    time.monotonic() - started,
                    time.monotonic() - close_started,
                    chunk_count,
                )
                if cleanup_cancelled:
                    raise asyncio.CancelledError
