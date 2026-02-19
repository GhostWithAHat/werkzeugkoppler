"""Configuration models and loaders for werkzeugkoppler.

This module defines the runtime configuration schema and how values are loaded
from YAML plus environment variable overrides.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

DEFAULT_CONFIG_PATH = "werkzeugkoppler/config.yaml"


class LoggingConfig(BaseModel):
    """Logging-related configuration."""

    model_config = ConfigDict(populate_by_name=True)

    level: str = "INFO"
    json_logs: bool = Field(default=False, alias="json")


class MCPServerConfig(BaseModel):
    """Configuration for one MCP server endpoint."""

    server_id: str
    transport: Literal["stdio", "http"]
    command: str | None = None
    args: list[str] = Field(default_factory=list)
    url: str | None = None
    connect_timeout_seconds: float = 10.0
    read_timeout_seconds: float = 60.0
    tool_call_timeout_seconds: float = 60.0


class ActionParameterConfig(BaseModel):
    """Parameter specification for one local action input."""

    name: str
    type: Literal[
        "project_file_path",
        "required_env_var",
        "optional_env_var",
        "insecure_string",
    ]
    description: str | None = None
    default: str | None = None


class ActionConfig(BaseModel):
    """Executable local action definition."""

    name: str
    description: str
    command: str
    arguments: list[str] | None = None
    parameters: list[ActionParameterConfig] = Field(default_factory=list)
    run_path: str | None = None
    timeout: int = 60


class LastUserMessageReaderConfig(BaseModel):
    """Executable command definition for last-user-message-based substitution."""

    name: str
    command: str
    arguments: list[str] | None = None
    run_path: str | None = None
    timeout: int = 60


class GatewayConfig(BaseModel):
    """Top-level gateway configuration."""

    model_config = ConfigDict(extra="forbid")

    service_base_url: str = "http://127.0.0.1:8080"
    service_api_key: str | None = None

    upstream_base_url: str
    upstream_api_key: str | None = None
    upstream_default_model: str | None = None
    fallback_fake_model_name: str | None = None
    upstream_connect_retries: int | None = None
    upstream_retry_interval_ms: int | None = None

    mcp_servers_refresh_seconds: int | None = None
    max_tool_concurrency: int | None = None
    max_tool_loops: int | None = None
    stream_keepalive_seconds: float | None = None
    stream_answer_mode: Literal["live", "safe_preview"] | None = None

    mcp_servers: list[MCPServerConfig] = Field(default_factory=list)
    actions: list[ActionConfig] = Field(default_factory=list)
    allowed_direct_commands: list[str] = Field(default_factory=list)
    allowed_direct_commands_max_output_lines: int | None = None
    allowed_direct_commands_max_output_bytes: int | None = None
    last_user_message_readers: list[LastUserMessageReaderConfig] = Field(default_factory=list)
    init_messages: list[dict[str, Any]] = Field(default_factory=list)
    logging: LoggingConfig | None = None

    @model_validator(mode="before")
    @classmethod
    def _map_direct_command_output_limit_aliases(cls, data: Any) -> Any:
        """Support short aliases for direct command output limits."""
        if not isinstance(data, dict):
            return data
        mapped = dict(data)
        alias_map = {
            "max_output_lines": "allowed_direct_commands_max_output_lines",
            "max_output_bytes": "allowed_direct_commands_max_output_bytes",
        }
        for alias, canonical in alias_map.items():
            if alias in mapped and canonical in mapped:
                if mapped[alias] != mapped[canonical]:
                    raise ValueError(f"Both '{alias}' and '{canonical}' are set with different values")
                mapped.pop(alias, None)
                continue
            if alias in mapped:
                mapped[canonical] = mapped.pop(alias)
        return mapped

    @model_validator(mode="after")
    def _validate_service_base_url(self) -> "GatewayConfig":
        """Validate that service_base_url includes host and port."""
        parsed = urlparse(self.service_base_url)
        if not parsed.hostname or parsed.port is None:
            raise ValueError("service_base_url must include host and port, e.g. http://127.0.0.1:8080")
        # Fallback defaults (matching current config.yaml values).
        if self.mcp_servers_refresh_seconds is None:
            self.mcp_servers_refresh_seconds = 300
        if self.upstream_connect_retries is None:
            self.upstream_connect_retries = 0
        if self.upstream_retry_interval_ms is None:
            self.upstream_retry_interval_ms = 1000
        if self.max_tool_concurrency is None:
            self.max_tool_concurrency = 4
        if self.max_tool_loops is None:
            self.max_tool_loops = 8
        if self.stream_keepalive_seconds is None:
            self.stream_keepalive_seconds = 1.0
        if self.stream_answer_mode is None:
            self.stream_answer_mode = "live"
        if self.logging is None:
            self.logging = LoggingConfig()
        if self.allowed_direct_commands_max_output_lines is None:
            self.allowed_direct_commands_max_output_lines = 800
        if self.allowed_direct_commands_max_output_bytes is None:
            self.allowed_direct_commands_max_output_bytes = 80000
        return self

    @field_validator(
        "allowed_direct_commands_max_output_lines",
        "allowed_direct_commands_max_output_bytes",
    )
    @classmethod
    def _validate_positive_output_limits(cls, value: int | None) -> int | None:
        """Ensure direct command output limits are non-negative (0 disables a limit)."""
        if value is None:
            return None
        if value < 0:
            raise ValueError("direct command output limits must be >= 0")
        return value

    @field_validator(
        "mcp_servers",
        "actions",
        "allowed_direct_commands",
        "last_user_message_readers",
        "init_messages",
        mode="before",
    )
    @classmethod
    def _none_to_empty_list(cls, value: Any) -> Any:
        """Treat explicit YAML `null` for list fields as an empty list."""
        if value is None:
            return []
        return value


def _load_yaml(path: str | None) -> dict[str, Any]:
    """Load a YAML file into a dictionary.

    Missing files are treated as empty config for environment-only deployments.
    """
    if not path:
        return {}
    cfg_path = Path(path)
    if not cfg_path.exists():
        return {}
    with cfg_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be object: {path}")
    return data


def _override_from_env(data: dict[str, Any]) -> dict[str, Any]:
    """Apply environment variable overrides on top of file configuration."""
    env_map = {
        "service_base_url": "WERKZEUGKOPPLER_SERVICE_BASE_URL",
        "service_api_key": "WERKZEUGKOPPLER_SERVICE_API_KEY",
        "upstream_base_url": "WERKZEUGKOPPLER_UPSTREAM_BASE_URL",
        "upstream_api_key": "WERKZEUGKOPPLER_UPSTREAM_API_KEY",
        "upstream_default_model": "WERKZEUGKOPPLER_UPSTREAM_DEFAULT_MODEL",
        "mcp_servers_refresh_seconds": "WERKZEUGKOPPLER_MCP_SERVERS_REFRESH_SECONDS",
        "upstream_connect_retries": "WERKZEUGKOPPLER_UPSTREAM_CONNECT_RETRIES",
        "upstream_retry_interval_ms": "WERKZEUGKOPPLER_UPSTREAM_RETRY_INTERVAL_MS",
        "max_tool_concurrency": "WERKZEUGKOPPLER_MAX_TOOL_CONCURRENCY",
        "max_tool_loops": "WERKZEUGKOPPLER_MAX_TOOL_LOOPS",
        "stream_keepalive_seconds": "WERKZEUGKOPPLER_STREAM_KEEPALIVE_SECONDS",
        "stream_answer_mode": "WERKZEUGKOPPLER_STREAM_ANSWER_MODE",
        "logging.level": "WERKZEUGKOPPLER_LOG_LEVEL",
        "logging.json_logs": "WERKZEUGKOPPLER_LOG_JSON",
    }

    out = dict(data)
    out.setdefault("logging", {})

    for key, env_name in env_map.items():
        value = os.getenv(env_name)
        if value is None:
            continue

        if key in {
            "mcp_servers_refresh_seconds",
            "upstream_connect_retries",
            "upstream_retry_interval_ms",
            "max_tool_concurrency",
            "max_tool_loops",
        }:
            out[key] = int(value)
        elif key == "stream_keepalive_seconds":
            out[key] = float(value)
        elif key == "stream_answer_mode":
            out[key] = value.strip().lower()
        elif key == "logging.json_logs":
            out["logging"]["json"] = value.lower() in {"1", "true", "yes", "on"}
        elif key == "logging.level":
            out["logging"]["level"] = value
        else:
            out[key] = value

    return out


def load_config(path: str | None = None) -> GatewayConfig:
    """Load, merge, and validate gateway configuration."""
    final_path = path or os.getenv("WERKZEUGKOPPLER_CONFIG") or DEFAULT_CONFIG_PATH
    raw = _load_yaml(final_path)
    raw = _override_from_env(raw)
    return GatewayConfig.model_validate(raw)
