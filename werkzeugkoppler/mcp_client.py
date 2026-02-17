"""MCP client implementations for HTTP and stdio transports."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

import httpx

from .config import MCPServerConfig

LOG = logging.getLogger(__name__)


class MCPError(Exception):
    """Raised for MCP protocol and transport errors."""


class _TransportAttemptError(Exception):
    """Internal exception used while probing HTTP MCP transport variants."""


@dataclass
class MCPServerState:
    """Basic runtime status tracking for one MCP server."""

    ok: bool = False
    last_error: str | None = None
    last_refresh_at: str | None = None


class MCPClient(ABC):
    """Abstract MCP client interface used by the tool registry."""

    def __init__(self, cfg: MCPServerConfig) -> None:
        """Create a client bound to one server config."""
        self.cfg = cfg
        self.state = MCPServerState()

    @abstractmethod
    async def start(self) -> None:
        """Initialize transport resources."""

    @abstractmethod
    async def close(self) -> None:
        """Release transport resources."""

    @abstractmethod
    async def tools_list(self, cursor: str | None = None) -> dict[str, Any]:
        """List tools for this server, optionally paginated."""

    @abstractmethod
    async def tools_call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Execute one tool call and return MCP result payload."""


class HTTPMCPClient(MCPClient):
    """HTTP MCP client supporting JSON-RPC POST and streamable-http variants."""

    def __init__(self, cfg: MCPServerConfig) -> None:
        """Initialize HTTP transport state."""
        super().__init__(cfg)
        self._client: httpx.AsyncClient | None = None
        self._id = 0
        self._selected_mode: Literal["jsonrpc", "streamable"] | None = None
        self._selected_url: str | None = None
        self._session_id: str | None = None
        self._initialized = False
        self._initialize_lock = asyncio.Lock()

    async def start(self) -> None:
        """Open HTTP client resources."""
        if not self.cfg.url:
            raise ValueError(f"Missing url for MCP server '{self.cfg.server_id}'")
        timeout = httpx.Timeout(
            connect=self.cfg.connect_timeout_seconds,
            read=self.cfg.read_timeout_seconds,
            write=self.cfg.read_timeout_seconds,
            pool=self.cfg.connect_timeout_seconds,
        )
        self._client = httpx.AsyncClient(timeout=timeout, follow_redirects=True)

    async def close(self) -> None:
        """Close HTTP client resources."""
        if self._client:
            await self._client.aclose()

    def _endpoint_candidates(self) -> list[str]:
        """Return endpoint candidates for probing.

        The implementation intentionally only uses the configured URL to avoid
        redirect probing noise.
        """
        assert self.cfg.url is not None
        base = self.cfg.url.strip()
        if not base:
            return []
        return [base]

    def _headers_for_mode(self, mode: Literal["jsonrpc", "streamable"]) -> dict[str, str]:
        """Build MCP HTTP headers.

        Accept always advertises both JSON and SSE to satisfy strict servers.
        """
        _ = mode
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "MCP-Protocol-Version": "2024-11-05",
        }
        if self._session_id:
            headers["mcp-session-id"] = self._session_id
        return headers

    @staticmethod
    def _normalize_id(value: Any) -> str:
        """Normalize JSON-RPC IDs for robust comparisons."""
        return str(value)

    def _extract_rpc_envelope(self, payload: Any, req_id: int | None) -> dict[str, Any] | None:
        """Extract a JSON-RPC envelope from nested payload variants."""
        if not isinstance(payload, dict):
            return None

        if "id" in payload and ("result" in payload or "error" in payload):
            if req_id is None or self._normalize_id(payload["id"]) == self._normalize_id(req_id):
                if "jsonrpc" not in payload:
                    return {"jsonrpc": "2.0", **payload}
                return payload

        for key in ("response", "message", "data"):
            found = self._extract_rpc_envelope(payload.get(key), req_id)
            if found is not None:
                return found

        return None

    async def _read_sse_response(self, response: httpx.Response, req_id: int | None) -> dict[str, Any]:
        """Read and decode one JSON-RPC response from an SSE stream."""
        event_name = "message"
        data_lines: list[str] = []

        def flush_event() -> dict[str, Any] | None:
            nonlocal event_name, data_lines
            if not data_lines:
                event_name = "message"
                return None

            current_event = event_name
            data = "\n".join(data_lines).strip()
            event_name = "message"
            data_lines = []

            if not data or data == "[DONE]":
                return None

            try:
                obj = json.loads(data)
            except json.JSONDecodeError:
                return None

            found = self._extract_rpc_envelope(obj, req_id)
            if found is not None:
                return found

            if current_event == "error" and isinstance(obj, dict):
                return {"jsonrpc": "2.0", "id": req_id or 0, "error": obj}

            return None

        async for line in response.aiter_lines():
            if line == "":
                found = flush_event()
                if found is not None:
                    return found
                continue
            if line.startswith(":"):
                continue
            if line.startswith("event:"):
                event_name = line[6:].strip() or "message"
                continue
            if line.startswith("data:"):
                data_lines.append(line[5:].lstrip())

        found = flush_event()
        if found is not None:
            return found
        raise _TransportAttemptError("SSE response did not contain a JSON-RPC reply")

    @staticmethod
    def _body_preview(raw: str, limit: int = 1000) -> str:
        """Return a bounded response body snippet for logs."""
        if len(raw) > limit:
            return raw[:limit] + "...<truncated>"
        return raw

    def _status_error_detail(
        self,
        mode: Literal["jsonrpc", "streamable"],
        url: str,
        response: httpx.Response,
        body_text: str | None = None,
    ) -> str:
        """Build a detailed error string for failed HTTP attempts."""
        content_type = response.headers.get("content-type", "")
        session = response.headers.get("mcp-session-id", "")
        if body_text is None:
            try:
                body_text = response.text
            except Exception:
                body_text = "<unavailable>"
        return (
            f"{mode} attempt on {url} returned HTTP {response.status_code} "
            f"content_type={content_type!r} mcp_session_id={session!r} "
            f"body={self._body_preview(body_text)!r}"
        )

    def _capture_session_id(self, response: httpx.Response) -> None:
        """Persist session ID from server response if present."""
        new_session = response.headers.get("mcp-session-id")
        if not new_session:
            return
        if new_session != self._session_id:
            self._session_id = new_session
            LOG.info(
                "MCP HTTP session id updated server_id=%s session_id=%s",
                self.cfg.server_id,
                new_session,
            )

    def _parse_rpc_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Validate JSON-RPC envelope and return the result object."""
        if "error" in payload:
            raise MCPError(json.dumps(payload["error"], ensure_ascii=False))
        return payload.get("result", {})

    def _attempt_plan(self) -> list[tuple[Literal["jsonrpc", "streamable"], str]]:
        """Build transport/url attempt order, preferring last successful variant."""
        endpoints = self._endpoint_candidates()
        if not endpoints:
            return []

        plan: list[tuple[Literal["jsonrpc", "streamable"], str]] = []
        if self._selected_mode and self._selected_url:
            plan.append((self._selected_mode, self._selected_url))

        for mode in ("jsonrpc", "streamable"):
            for url in endpoints:
                candidate = (mode, url)
                if candidate not in plan:
                    plan.append(candidate)

        return plan

    def _remember_transport(self, mode: Literal["jsonrpc", "streamable"], url: str) -> None:
        """Persist selected transport variant for future requests."""
        if self._selected_mode == mode and self._selected_url == url:
            return
        self._selected_mode = mode
        self._selected_url = url
        LOG.info(
            "MCP HTTP transport selected server_id=%s mode=%s url=%s",
            self.cfg.server_id,
            mode,
            url,
        )

    async def _rpc_attempt(
        self,
        mode: Literal["jsonrpc", "streamable"],
        url: str,
        req: dict[str, Any],
    ) -> tuple[dict[str, Any], str, Literal["jsonrpc", "streamable"]]:
        """Execute one HTTP request attempt for an RPC call."""
        assert self._client is not None
        headers = self._headers_for_mode(mode)

        if mode == "jsonrpc":
            try:
                response = await self._client.post(url, json=req, headers=headers)
            except httpx.HTTPError as exc:
                raise _TransportAttemptError(f"{mode} attempt on {url} failed: {exc}") from exc

            self._capture_session_id(response)
            if response.history:
                LOG.info(
                    "MCP HTTP redirect followed server_id=%s from=%s to=%s hops=%s",
                    self.cfg.server_id,
                    url,
                    str(response.url),
                    len(response.history),
                )
            if response.status_code >= 400:
                raise _TransportAttemptError(self._status_error_detail(mode, url, response))

            content_type = response.headers.get("content-type", "").lower()
            if "text/event-stream" in content_type:
                payload = await self._read_sse_response(response, req_id=req["id"])
                return payload, str(response.url), "streamable"

            try:
                payload = response.json()
            except json.JSONDecodeError as exc:
                raise _TransportAttemptError("jsonrpc attempt returned non-JSON body") from exc

            found = self._extract_rpc_envelope(payload, req["id"])
            if found is None:
                raise _TransportAttemptError("HTTP response is not a JSON-RPC envelope")
            return found, str(response.url), "jsonrpc"

        try:
            async with self._client.stream("POST", url, json=req, headers=headers) as response:
                self._capture_session_id(response)
                if response.history:
                    LOG.info(
                        "MCP HTTP redirect followed server_id=%s from=%s to=%s hops=%s",
                        self.cfg.server_id,
                        url,
                        str(response.url),
                        len(response.history),
                    )
                if response.status_code >= 400:
                    raw = (await response.aread()).decode("utf-8", errors="replace")
                    raise _TransportAttemptError(self._status_error_detail(mode, url, response, raw))

                content_type = response.headers.get("content-type", "").lower()
                if "text/event-stream" in content_type:
                    payload = await self._read_sse_response(response, req_id=req["id"])
                    return payload, str(response.url), "streamable"

                raw = await response.aread()
                try:
                    payload = json.loads(raw.decode("utf-8"))
                except Exception as exc:
                    raise _TransportAttemptError("streamable-http fallback got non-JSON body") from exc

                found = self._extract_rpc_envelope(payload, req["id"])
                if found is None:
                    raise _TransportAttemptError("streamable-http fallback body not JSON-RPC")
                return found, str(response.url), "jsonrpc"
        except MCPError:
            raise
        except _TransportAttemptError:
            raise
        except httpx.HTTPError as exc:
            raise _TransportAttemptError(f"{mode} attempt on {url} failed: {exc}") from exc

    async def _notify_attempt(
        self,
        mode: Literal["jsonrpc", "streamable"],
        url: str,
        req: dict[str, Any],
    ) -> tuple[str, Literal["jsonrpc", "streamable"]]:
        """Execute one notification request attempt."""
        assert self._client is not None
        headers = self._headers_for_mode(mode)

        if mode == "jsonrpc":
            try:
                response = await self._client.post(url, json=req, headers=headers)
            except httpx.HTTPError as exc:
                raise _TransportAttemptError(f"{mode} attempt on {url} failed: {exc}") from exc

            self._capture_session_id(response)
            if response.history:
                LOG.info(
                    "MCP HTTP redirect followed server_id=%s from=%s to=%s hops=%s",
                    self.cfg.server_id,
                    url,
                    str(response.url),
                    len(response.history),
                )
            if response.status_code >= 400:
                raise _TransportAttemptError(self._status_error_detail(mode, url, response))
            return str(response.url), "jsonrpc"

        try:
            async with self._client.stream("POST", url, json=req, headers=headers) as response:
                self._capture_session_id(response)
                if response.history:
                    LOG.info(
                        "MCP HTTP redirect followed server_id=%s from=%s to=%s hops=%s",
                        self.cfg.server_id,
                        url,
                        str(response.url),
                        len(response.history),
                    )
                if response.status_code >= 400:
                    raw = (await response.aread()).decode("utf-8", errors="replace")
                    raise _TransportAttemptError(self._status_error_detail(mode, url, response, raw))
                return str(response.url), "streamable"
        except _TransportAttemptError:
            raise
        except httpx.HTTPError as exc:
            raise _TransportAttemptError(f"{mode} attempt on {url} failed: {exc}") from exc

    async def _notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        """Send a JSON-RPC notification over the best available HTTP transport."""
        if self._client is None:
            await self.start()

        req = {"jsonrpc": "2.0", "method": method, "params": params or {}}
        errors: list[str] = []
        for mode, url in self._attempt_plan():
            try:
                final_url, final_mode = await self._notify_attempt(mode, url, req)
                self._remember_transport(final_mode, final_url)
                return
            except _TransportAttemptError as exc:
                errors.append(str(exc))
                LOG.debug(
                    "MCP HTTP notification attempt failed server_id=%s method=%s mode=%s url=%s error=%s",
                    self.cfg.server_id,
                    method,
                    mode,
                    url,
                    exc,
                )

        detail = "; ".join(errors) if errors else "no transport attempts executed"
        raise MCPError(f"All MCP HTTP notification attempts failed for {self.cfg.server_id}: {detail}")

    async def _ensure_initialized(self) -> None:
        """Initialize HTTP MCP session lazily."""
        if self._initialized:
            return

        async with self._initialize_lock:
            if self._initialized:
                return

            init_params = {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "werkzeugkoppler", "version": "0.1.0"},
            }
            await self._rpc("initialize", init_params, _allow_reinit_retry=False)

            try:
                await self._notify("notifications/initialized", {})
            except Exception as exc:
                LOG.debug(
                    "MCP notifications/initialized failed server_id=%s error=%s",
                    self.cfg.server_id,
                    exc,
                )

            self._initialized = True
            LOG.info(
                "MCP HTTP session initialized server_id=%s mode=%s url=%s session_id=%s",
                self.cfg.server_id,
                self._selected_mode,
                self._selected_url,
                self._session_id,
            )

    async def _rpc(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        _allow_reinit_retry: bool = True,
    ) -> dict[str, Any]:
        """Execute one JSON-RPC request with transport fallback and reinit retry."""
        if self._client is None:
            await self.start()

        if method not in {"initialize", "notifications/initialized"}:
            await self._ensure_initialized()

        self._id += 1
        req = {
            "jsonrpc": "2.0",
            "id": self._id,
            "method": method,
            "params": params or {},
        }

        errors: list[str] = []
        for mode, url in self._attempt_plan():
            try:
                payload, final_url, final_mode = await self._rpc_attempt(mode, url, req)
                result = self._parse_rpc_payload(payload)
                self._remember_transport(final_mode, final_url)
                return result
            except MCPError:
                raise
            except _TransportAttemptError as exc:
                errors.append(str(exc))
                LOG.debug(
                    "MCP HTTP transport attempt failed server_id=%s method=%s mode=%s url=%s error=%s",
                    self.cfg.server_id,
                    method,
                    mode,
                    url,
                    exc,
                )

        if _allow_reinit_retry and method not in {"initialize", "notifications/initialized"} and self._initialized:
            LOG.info(
                "MCP HTTP session reset and reinitialize server_id=%s after failures method=%s",
                self.cfg.server_id,
                method,
            )
            self._session_id = None
            self._initialized = False
            self._selected_mode = None
            self._selected_url = None
            await self._ensure_initialized()
            return await self._rpc(method, params, _allow_reinit_retry=False)

        detail = "; ".join(errors) if errors else "no transport attempts executed"
        raise MCPError(f"All MCP HTTP transport attempts failed for {self.cfg.server_id}: {detail}")

    async def tools_list(self, cursor: str | None = None) -> dict[str, Any]:
        """List tools for HTTP MCP server."""
        params: dict[str, Any] = {}
        if cursor:
            params["cursor"] = cursor
        return await self._rpc("tools/list", params)

    async def tools_call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Execute a tool call over HTTP MCP."""
        return await self._rpc("tools/call", {"name": name, "arguments": arguments})


class StdioMCPClient(MCPClient):
    """MCP stdio client using framed JSON-RPC messages."""

    def __init__(self, cfg: MCPServerConfig, notify_cb: Callable[[str, str], None] | None = None) -> None:
        """Initialize stdio transport state."""
        super().__init__(cfg)
        self._proc: asyncio.subprocess.Process | None = None
        self._write_lock = asyncio.Lock()
        self._next_id = 0
        self._pending: dict[int, asyncio.Future[dict[str, Any]]] = {}
        self._reader_task: asyncio.Task[None] | None = None
        self._notify_cb = notify_cb

    async def start(self) -> None:
        """Spawn MCP process and start reader loop."""
        if self._proc is not None:
            return
        if not self.cfg.command:
            raise ValueError(f"Missing command for MCP server '{self.cfg.server_id}'")

        self._proc = await asyncio.create_subprocess_exec(
            self.cfg.command,
            *self.cfg.args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        self._reader_task = asyncio.create_task(self._reader_loop())

        # Best-effort initialize for servers requiring an MCP handshake.
        try:
            await self._rpc(
                "initialize",
                {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "werkzeugkoppler", "version": "0.1.0"},
                },
            )
        except Exception as exc:  # pragma: no cover
            LOG.debug("MCP initialize failed for %s: %s", self.cfg.server_id, exc)

    async def close(self) -> None:
        """Stop reader loop and terminate MCP process."""
        if self._reader_task:
            self._reader_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._reader_task
        if self._proc:
            self._proc.terminate()
            await self._proc.wait()

    async def _read_frame(self) -> dict[str, Any]:
        """Read one framed JSON-RPC message from stdout."""
        assert self._proc is not None and self._proc.stdout is not None
        stdout = self._proc.stdout
        content_length = None

        while True:
            line = await stdout.readline()
            if not line:
                raise MCPError("MCP stdio stream ended")
            if line in {b"\r\n", b"\n"}:
                break
            key, _, value = line.decode("utf-8").partition(":")
            if key.lower().strip() == "content-length":
                content_length = int(value.strip())

        if content_length is None:
            raise MCPError("Missing Content-Length in MCP frame")

        raw = await stdout.readexactly(content_length)
        return json.loads(raw.decode("utf-8"))

    async def _send_frame(self, payload: dict[str, Any]) -> None:
        """Write one framed JSON-RPC message to stdin."""
        assert self._proc is not None and self._proc.stdin is not None
        data = json.dumps(payload).encode("utf-8")
        frame = f"Content-Length: {len(data)}\r\n\r\n".encode("utf-8") + data
        self._proc.stdin.write(frame)
        await self._proc.stdin.drain()

    async def _reader_loop(self) -> None:
        """Route incoming responses/notifications to waiting futures and callbacks."""
        try:
            while True:
                msg = await self._read_frame()
                if "id" in msg:
                    req_id = int(msg["id"])
                    future = self._pending.pop(req_id, None)
                    if future and not future.done():
                        future.set_result(msg)
                elif "method" in msg:
                    method = str(msg["method"])
                    if method == "tools/list_changed" and self._notify_cb:
                        self._notify_cb(self.cfg.server_id, method)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            for future in self._pending.values():
                if not future.done():
                    future.set_exception(exc)
            self._pending.clear()
            LOG.exception("MCP reader loop crashed for %s", self.cfg.server_id)

    async def _rpc(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Execute one stdio JSON-RPC request."""
        if self._proc is None:
            await self.start()
        assert self._proc is not None

        async with self._write_lock:
            self._next_id += 1
            req_id = self._next_id
            future: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
            self._pending[req_id] = future
            await self._send_frame(
                {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "method": method,
                    "params": params or {},
                }
            )

        msg = await asyncio.wait_for(future, timeout=self.cfg.read_timeout_seconds)
        if "error" in msg:
            raise MCPError(json.dumps(msg["error"], ensure_ascii=False))
        return msg.get("result", {})

    async def tools_list(self, cursor: str | None = None) -> dict[str, Any]:
        """List tools for stdio MCP server."""
        params: dict[str, Any] = {}
        if cursor:
            params["cursor"] = cursor
        return await self._rpc("tools/list", params)

    async def tools_call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Execute a tool call over stdio MCP."""
        return await self._rpc("tools/call", {"name": name, "arguments": arguments})


@dataclass
class ToolBinding:
    """Mapping from exposed OpenAI tool names to internal backend tools."""

    openai_name: str
    server_id: str
    mcp_name: str
    description: str
    input_schema: dict[str, Any]
    source: Literal["mcp", "action"] = "mcp"


@dataclass
class ServerTools:
    """Container for discovered tools plus generated bindings."""

    tools: list[dict[str, Any]] = field(default_factory=list)
    bindings: list[ToolBinding] = field(default_factory=list)
