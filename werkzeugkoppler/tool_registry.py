"""Tool discovery and dispatch registry.

The registry merges tools discovered from MCP servers with optional local
`actions` tools and exposes a unified OpenAI-tool-facing interface.
"""

from __future__ import annotations

import asyncio
import contextlib
import copy
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import ActionConfig, GatewayConfig, MCPServerConfig
from .local_actions import LocalActionError, LocalActionExecutor
from .mcp_client import HTTPMCPClient, MCPClient, MCPError, StdioMCPClient, ToolBinding
from .utils import to_bounded_json

LOG = logging.getLogger(__name__)


@dataclass
class RefreshStatus:
    """Status snapshot for one refresh operation."""

    server_id: str
    ok: bool
    error: str | None
    refreshed_at: str | None


class ToolRegistry:
    """Thread-safe registry that owns tool cache, bindings, and MCP clients."""

    def __init__(self, cfg: GatewayConfig) -> None:
        """Create a registry for one runtime configuration."""
        self.cfg = cfg
        self._clients: dict[str, MCPClient] = {}
        self._local_actions: dict[str, ActionConfig] = {action.name: action for action in (cfg.actions or [])}
        self._local_action_server_id = "actions"
        self._local_executor = LocalActionExecutor(project_root=Path.cwd())

        self._tool_cache: list[dict[str, Any]] = []
        self._binding_map: dict[str, ToolBinding] = {}
        self._server_errors: dict[str, str | None] = {}
        self._last_refresh_at: str | None = None

        self._lock = asyncio.Lock()
        self._refresh_task: asyncio.Task[None] | None = None
        self._refresh_event = asyncio.Event()

    def _notify(self, _server_id: str, method: str) -> None:
        """React to MCP notifications relevant for registry state."""
        if method == "tools/list_changed":
            self._refresh_event.set()

    async def start(self) -> None:
        """Start clients, perform initial discovery, and launch refresh loop."""
        for server in (self.cfg.mcp_servers or []):
            self._clients[server.server_id] = self._build_client(server)

        for client in self._clients.values():
            try:
                await client.start()
            except Exception as exc:
                self._server_errors[client.cfg.server_id] = str(exc)
                LOG.warning("MCP start failed for %s: %s", client.cfg.server_id, exc)

        await self.refresh_tools()
        self._refresh_task = asyncio.create_task(self._periodic_refresh_loop())

    async def close(self) -> None:
        """Stop background loops and close all clients."""
        if self._refresh_task:
            self._refresh_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._refresh_task

        for client in self._clients.values():
            with contextlib.suppress(Exception):
                await client.close()

    def _build_client(self, server: MCPServerConfig) -> MCPClient:
        """Instantiate the proper MCP client implementation for one server."""
        if server.transport == "http":
            return HTTPMCPClient(server)
        return StdioMCPClient(server, notify_cb=self._notify)

    async def _periodic_refresh_loop(self) -> None:
        """Refresh tools periodically or on explicit notifications."""
        while True:
            try:
                await asyncio.wait_for(self._refresh_event.wait(), timeout=self.cfg.refresh_seconds)
                self._refresh_event.clear()
            except asyncio.TimeoutError:
                pass
            await self.refresh_tools()

    async def refresh_tools(self) -> None:
        """Refresh tool cache from all configured sources."""
        tool_cache: list[dict[str, Any]] = []
        binding_map: dict[str, ToolBinding] = {}

        await self._refresh_mcp_tools(tool_cache, binding_map)
        self._refresh_local_actions(tool_cache, binding_map)

        now = datetime.now(timezone.utc).isoformat()
        async with self._lock:
            self._tool_cache = tool_cache
            self._binding_map = binding_map
            self._last_refresh_at = now

    async def _refresh_mcp_tools(
        self,
        tool_cache: list[dict[str, Any]],
        binding_map: dict[str, ToolBinding],
    ) -> None:
        """Discover tools from all MCP servers and extend cache/bindings."""
        for server_id, client in self._clients.items():
            try:
                tools = await self._load_server_tools(client)
                tool_names = sorted(str(t.get("name", "")).strip() for t in tools if str(t.get("name", "")).strip())
                LOG.info(
                    "MCP tools discovered server_id=%s count=%s tools=%s",
                    server_id,
                    len(tool_names),
                    ", ".join(tool_names) if tool_names else "(none)",
                )

                self._server_errors[server_id] = None
                for tool in tools:
                    mcp_name = str(tool.get("name", "")).strip()
                    if not mcp_name:
                        continue

                    schema = tool.get("inputSchema") or {"type": "object", "properties": {}}
                    mapped_name = self._map_tool_name(server_id, mcp_name)
                    description = str(tool.get("description", "")).strip()

                    tool_cache.append(
                        {
                            "type": "function",
                            "function": {
                                "name": mapped_name,
                                "description": f"{description} (server_id={server_id})".strip(),
                                "parameters": schema,
                            },
                        }
                    )
                    binding_map[mapped_name] = ToolBinding(
                        openai_name=mapped_name,
                        server_id=server_id,
                        mcp_name=mcp_name,
                        description=description,
                        input_schema=schema,
                        source="mcp",
                    )
            except Exception as exc:
                self._server_errors[server_id] = str(exc)
                LOG.warning("MCP refresh failed for %s: %s", server_id, exc)

    def _refresh_local_actions(
        self,
        tool_cache: list[dict[str, Any]],
        binding_map: dict[str, ToolBinding],
    ) -> None:
        """Inject locally configured `actions` tools into cache/bindings."""
        if self._local_actions:
            action_names = sorted(self._local_actions.keys())
            LOG.info(
                "Local action tools discovered server_id=%s count=%s tools=%s",
                self._local_action_server_id,
                len(action_names),
                ", ".join(action_names),
            )

        for action in (self.cfg.actions or []):
            mapped_name = self._map_tool_name(self._local_action_server_id, action.name)
            schema = self._local_executor.input_schema_for_action(action)

            tool_cache.append(
                {
                    "type": "function",
                    "function": {
                        "name": mapped_name,
                        "description": f"{action.description} (server_id={self._local_action_server_id})",
                        "parameters": schema,
                    },
                }
            )
            binding_map[mapped_name] = ToolBinding(
                openai_name=mapped_name,
                server_id=self._local_action_server_id,
                mcp_name=action.name,
                description=action.description,
                input_schema=schema,
                source="action",
            )

    async def _load_server_tools(self, client: MCPClient) -> list[dict[str, Any]]:
        """Load all tools from one MCP server with cursor-based pagination."""
        all_tools: list[dict[str, Any]] = []
        cursor: str | None = None

        while True:
            result = await client.tools_list(cursor=cursor)
            tools = result.get("tools", [])
            if isinstance(tools, list):
                all_tools.extend(tool for tool in tools if isinstance(tool, dict))

            cursor = result.get("nextCursor")
            if not cursor:
                break

        return all_tools

    @staticmethod
    def _map_tool_name(server_id: str, mcp_name: str) -> str:
        """Map server/tool IDs to a collision-free OpenAI tool name."""
        safe_server = "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in server_id)
        safe_tool = "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in mcp_name)
        return f"{safe_server}__{safe_tool}"

    async def get_openai_tools(self) -> list[dict[str, Any]]:
        """Return a deep copy of currently cached OpenAI tool definitions."""
        async with self._lock:
            return copy.deepcopy(self._tool_cache)

    async def resolve_binding(self, openai_name: str) -> ToolBinding | None:
        """Resolve one exposed OpenAI tool name to internal binding."""
        async with self._lock:
            return self._binding_map.get(openai_name)

    async def get_health(self) -> dict[str, Any]:
        """Return aggregated health state of MCP servers and local actions."""
        async with self._lock:
            errors = dict(self._server_errors)
            last_refresh_at = self._last_refresh_at

        server_states: list[dict[str, Any]] = []
        degraded = False
        for server in (self.cfg.mcp_servers or []):
            err = errors.get(server.server_id)
            ok = err is None
            degraded = degraded or not ok
            server_states.append(
                {
                    "server_id": server.server_id,
                    "transport": server.transport,
                    "ok": ok,
                    "error": err,
                }
            )

        if self._local_actions:
            server_states.append(
                {
                    "server_id": self._local_action_server_id,
                    "transport": "local_actions",
                    "ok": True,
                    "error": None,
                    "count": len(self._local_actions),
                }
            )

        return {
            "ok": not degraded,
            "degraded": degraded,
            "last_tool_refresh_at": last_refresh_at,
            "mcp_servers": server_states,
        }

    async def call_tool_by_openai_name(
        self,
        openai_name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """Dispatch a tool call to either local action executor or MCP server."""
        binding = await self.resolve_binding(openai_name)
        if not binding:
            raise MCPError(f"Unknown tool '{openai_name}'")

        if binding.source == "action":
            return await self._call_local_action(binding, openai_name, arguments)
        return await self._call_mcp_tool(binding, openai_name, arguments)

    async def _call_local_action(
        self,
        binding: ToolBinding,
        openai_name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute a local `actions` tool call."""
        action = self._local_actions.get(binding.mcp_name)
        if action is None:
            raise MCPError(f"Unknown local action '{binding.mcp_name}'")

        LOG.info("dispatching local action tool call action=%s openai_tool=%s", action.name, openai_name)
        if LOG.isEnabledFor(logging.DEBUG):
            LOG.debug(
                "local action tool call args action=%s openai_tool=%s args=%s",
                action.name,
                openai_name,
                to_bounded_json(arguments),
            )

        try:
            result = await asyncio.to_thread(self._local_executor.execute_action, action, arguments)
        except LocalActionError as exc:
            raise MCPError(f"Local action error on {action.name}: {exc}") from exc
        except Exception as exc:
            raise MCPError(f"Unexpected local action error on {action.name}: {exc}") from exc

        LOG.info("local action tool call finished action=%s", action.name)
        if LOG.isEnabledFor(logging.DEBUG):
            LOG.debug("local action tool call result action=%s result=%s", action.name, to_bounded_json(result))

        return {
            "server_id": self._local_action_server_id,
            "mcp_tool_name": action.name,
            "result": result,
        }

    async def _call_mcp_tool(
        self,
        binding: ToolBinding,
        openai_name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute one MCP-backed tool call with timeout handling."""
        client = self._clients[binding.server_id]
        timeout = client.cfg.tool_call_timeout_seconds

        LOG.info(
            "dispatching MCP tool call server_id=%s openai_tool=%s mcp_tool=%s timeout=%s",
            binding.server_id,
            openai_name,
            binding.mcp_name,
            timeout,
        )
        if LOG.isEnabledFor(logging.DEBUG):
            LOG.debug(
                "MCP tool call args server_id=%s mcp_tool=%s args=%s",
                binding.server_id,
                binding.mcp_name,
                to_bounded_json(arguments),
            )

        try:
            result = await asyncio.wait_for(client.tools_call(binding.mcp_name, arguments), timeout=timeout)
        except asyncio.TimeoutError as exc:
            LOG.debug(
                "MCP tool call timeout server_id=%s mcp_tool=%s timeout=%s",
                binding.server_id,
                binding.mcp_name,
                timeout,
            )
            raise MCPError(f"Tool timeout on {binding.server_id}/{binding.mcp_name}") from exc
        except Exception:
            LOG.debug(
                "MCP tool call failed server_id=%s mcp_tool=%s",
                binding.server_id,
                binding.mcp_name,
                exc_info=True,
            )
            raise

        LOG.info("MCP tool call finished server_id=%s mcp_tool=%s", binding.server_id, binding.mcp_name)
        if LOG.isEnabledFor(logging.DEBUG):
            LOG.debug(
                "MCP tool call result server_id=%s mcp_tool=%s result=%s",
                binding.server_id,
                binding.mcp_name,
                to_bounded_json(result),
            )

        return {
            "server_id": binding.server_id,
            "mcp_tool_name": binding.mcp_name,
            "result": result,
        }

    def format_tool_message(self, tool_name: str, tool_call_id: str, outcome: dict[str, Any]) -> dict[str, Any]:
        """Format one OpenAI `role=tool` message for upstream continuation."""
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": tool_name,
            "content": json.dumps(outcome, ensure_ascii=False),
        }
