"""Execution engine for local `actions` tools.

This module implements a hooks_mcp-like action runner that can expose local
commands as OpenAI tools via the tool registry.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
from pathlib import Path
from typing import Any

from .config import ActionConfig, ActionParameterConfig


class LocalActionError(Exception):
    """Raised when an action cannot be validated or executed."""


def _strip_ansi_codes(text: str) -> str:
    """Remove ANSI escape sequences from terminal output."""
    ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    return ansi_escape.sub("", text)


def _process_terminal_output(text: str) -> str:
    """Normalize terminal output by stripping ANSI codes and CR redraws."""
    clean_text = _strip_ansi_codes(text)
    lines = clean_text.split("\n")
    processed: list[str] = []
    for line in lines:
        parts = line.split("\r")
        processed.append(parts[-1] if parts else "")
    return "\n".join(processed).strip()


def _to_lines(text: str) -> list[str]:
    """Return non-empty output lines."""
    if not text:
        return []
    return [line for line in text.splitlines() if line.strip()]


def _resolve_path(path: str, project_root: Path) -> Path:
    """Resolve absolute or project-relative paths safely."""
    untrusted = Path(path)
    if untrusted.is_absolute():
        return untrusted.resolve()
    return (project_root / untrusted).resolve()


def _validate_project_path(path: str, project_root: Path) -> bool:
    """Validate that a path stays inside the project root."""
    try:
        canonical_root = project_root.resolve()
        expanded = os.path.expandvars(os.path.expanduser(path))

        # Keep behavior strict: unresolved environment references are rejected.
        if re.search(r"\$[A-Za-z_][A-Za-z0-9_]*|\$\{[^}]*\}", expanded):
            return False

        candidate = _resolve_path(expanded, canonical_root)
        if hasattr(candidate, "is_relative_to"):
            return candidate.is_relative_to(canonical_root)
        try:
            candidate.relative_to(canonical_root)
            return True
        except ValueError:
            return False
    except Exception:
        return False


def _prepare_parameters(action: ActionConfig, provided: dict[str, Any], project_root: Path) -> dict[str, str]:
    """Resolve action parameters and build environment variable values."""
    env_vars: dict[str, str] = {}
    for param in action.parameters:
        ptype = param.type

        if ptype in {"required_env_var", "optional_env_var"}:
            env_value = os.environ.get(param.name)
            if env_value is not None:
                env_vars[param.name] = env_value
            elif ptype == "required_env_var":
                raise LocalActionError(
                    f"Required environment variable '{param.name}' not set for action '{action.name}'"
                )
            continue

        value = provided.get(param.name, param.default)
        if value is None:
            raise LocalActionError(f"Required parameter '{param.name}' not provided for action '{action.name}'")

        if ptype == "project_file_path":
            path_val = str(value)
            if not _validate_project_path(path_val, project_root):
                raise LocalActionError(
                    f"Invalid path '{path_val}' for parameter '{param.name}' in action '{action.name}'"
                )
            resolved = _resolve_path(path_val, project_root)
            if not resolved.exists():
                raise LocalActionError(
                    f"Path '{path_val}' for parameter '{param.name}' in action '{action.name}' does not exist"
                )
            env_vars[param.name] = path_val
            continue

        if ptype == "insecure_string":
            env_vars[param.name] = str(value)
            continue

        raise LocalActionError(f"Unsupported parameter type '{ptype}' for action '{action.name}'")

    return env_vars


def _substitute_parameters(command_args: list[str], env_vars: dict[str, str]) -> list[str]:
    """Replace `$PARAM` placeholders in command arguments."""
    sorted_params = sorted(env_vars.keys(), key=len, reverse=True)
    substituted: list[str] = []
    for arg in command_args:
        out = arg
        for param_name in sorted_params:
            out = out.replace(f"${param_name}", str(env_vars[param_name]))
        substituted.append(out)
    return substituted


def _build_command_args(action: ActionConfig, env_vars: dict[str, str]) -> list[str]:
    """Build executable argv from action configuration."""
    if action.arguments is not None:
        return _substitute_parameters([action.command, *action.arguments], env_vars)
    try:
        parsed = shlex.split(action.command)
    except ValueError as exc:
        raise LocalActionError(f"Invalid command syntax for action '{action.name}': {exc}") from exc
    return _substitute_parameters(parsed, env_vars)


class LocalActionExecutor:
    """Execute configured local actions and normalize command output."""

    def __init__(self, project_root: Path | None = None) -> None:
        """Create a local action executor rooted at the current project."""
        self.project_root = (project_root or Path(os.getcwd())).resolve()

    def input_schema_for_action(self, action: ActionConfig) -> dict[str, Any]:
        """Generate JSON schema for action tool parameters."""
        params: list[ActionParameterConfig] = []
        for param in action.parameters:
            if param.type in {"required_env_var", "optional_env_var"}:
                continue
            params.append(param)

        return {
            "type": "object",
            "properties": {
                param.name: {
                    "type": "string",
                    "description": param.description or f"Parameter {param.name}",
                }
                for param in params
            },
            "required": [param.name for param in params if param.default is None],
        }

    def execute_action(self, action: ActionConfig, arguments: dict[str, Any]) -> Any:
        """Run a local action and normalize stdout for model-friendly consumption.

        Output behavior is intentionally simple:
        - If stdout is valid JSON: return the parsed JSON value.
        - Otherwise: return a plain list of non-empty stdout lines.
        """
        env_vars = _prepare_parameters(action, arguments, self.project_root)
        command_args = _build_command_args(action, env_vars)

        exec_dir = self.project_root
        if action.run_path:
            if not _validate_project_path(action.run_path, self.project_root):
                raise LocalActionError(f"Invalid run_path '{action.run_path}' for action '{action.name}'")
            exec_dir = (self.project_root / action.run_path).resolve()

        try:
            result = subprocess.run(
                command_args,
                cwd=exec_dir,
                env={**os.environ, **env_vars},
                capture_output=True,
                text=True,
                timeout=action.timeout,
                shell=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise LocalActionError(
                f"Command for action '{action.name}' timed out after {action.timeout} seconds"
            ) from exc
        except Exception as exc:
            raise LocalActionError(f"Failed to execute action '{action.name}': {exc}") from exc

        stdout = _process_terminal_output(result.stdout)
        _ = _process_terminal_output(result.stderr)  # Intentionally ignored in returned payload.

        if stdout:
            try:
                return json.loads(stdout)
            except json.JSONDecodeError:
                pass

        return _to_lines(stdout)
