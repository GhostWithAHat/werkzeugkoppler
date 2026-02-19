"""Message preparation and runtime placeholder handling."""

from __future__ import annotations

import logging
import os
import re
import shlex
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from .direct_commands import strip_direct_command_messages

LOG = logging.getLogger(__name__)

_TODAY_TOKEN_RE = re.compile(r"\{\s*today\s*\}")
_LAST_USER_MESSAGE_READER_RESULT_TOKEN_RE = re.compile(r"\{\s*last_user_message_reader_result\s*:\s*([^\{\}:]+?)\s*\}")


def _format_now_german(now: datetime) -> str:
    """Format datetime in German style used for `{today}` replacement."""
    weekdays = [
        "Montag",
        "Dienstag",
        "Mittwoch",
        "Donnerstag",
        "Freitag",
        "Samstag",
        "Sonntag",
    ]
    months = [
        "Januar",
        "Februar",
        "März",
        "April",
        "Mai",
        "Juni",
        "Juli",
        "August",
        "September",
        "Oktober",
        "November",
        "Dezember",
    ]
    return f"{weekdays[now.weekday()]}, {now.day}. {months[now.month - 1]} {now.year}, {now:%H:%M:%S}"


def _replace_today_tokens(content: Any) -> Any:
    """Replace `{today}` placeholder tokens inside message content payloads."""
    now_text = _format_now_german(datetime.now())

    if isinstance(content, str):
        return _TODAY_TOKEN_RE.sub(now_text, content)

    if isinstance(content, list):
        out: list[Any] = []
        for item in content:
            if isinstance(item, dict):
                updated = dict(item)
                if isinstance(updated.get("text"), str):
                    updated["text"] = _TODAY_TOKEN_RE.sub(now_text, updated["text"])
                out.append(updated)
            else:
                out.append(item)
        return out

    return content


def _substitute_last_message_placeholder(text: str, latest_user_message_text: str) -> str:
    """Replace `$LAST_USER_MESSAGE` token in command/args with latest user message."""
    return text.replace("$LAST_USER_MESSAGE", latest_user_message_text)


def _build_last_user_message_reader_command_args(
    reader: Any, latest_user_message_text: str
) -> list[str]:
    """Build argv for one configured last-user-message reader command."""
    command = _substitute_last_message_placeholder(reader.command, latest_user_message_text)

    if reader.arguments is not None:
        arguments = [_substitute_last_message_placeholder(arg, latest_user_message_text) for arg in reader.arguments]
        return [command, *arguments]

    try:
        parsed_command = shlex.split(command)
    except ValueError as exc:
        raise ValueError(f"Invalid command syntax for last user message reader '{reader.name}': {exc}") from exc
    return parsed_command


def _last_user_message_reader_run_path(reader: Any, project_root: Path) -> Path:
    """Resolve configured last user message reader run path relative to project root."""
    if not reader.run_path:
        return project_root
    run_path = Path(reader.run_path)
    if run_path.is_absolute():
        return run_path
    return (project_root / run_path).resolve()


def _execute_last_user_message_reader(
    reader: Any, latest_user_message_text: str, project_root: Path
) -> str:
    """Execute one configured last user message reader and return trimmed stdout."""
    command_args = _build_last_user_message_reader_command_args(reader, latest_user_message_text)
    run_cwd = _last_user_message_reader_run_path(reader, project_root)
    LOG.info("Executing last user message reader '%s' command=%s", reader.name, command_args)

    try:
        result = subprocess.run(
            command_args,
            cwd=run_cwd,
            capture_output=True,
            text=True,
            timeout=reader.timeout,
            shell=False,
        )
    except subprocess.TimeoutExpired:
        LOG.warning("Last user message reader '%s' timed out after %s seconds", reader.name, reader.timeout)
        return ""
    except Exception:
        LOG.exception("Last user message reader '%s' execution failed", reader.name)
        return ""

    stderr_text = (result.stderr or "").strip()
    if stderr_text:
        LOG.warning("Last user message reader '%s' stderr: %s", reader.name, stderr_text)
    LOG.info("Last user message reader '%s' exit_code=%s", reader.name, result.returncode)
    return (result.stdout or "").strip()


def _replace_last_user_message_reader_result_tokens(
    text: str,
    reader_result_cache: dict[str, str],
) -> str:
    """Replace `{last_user_message_reader_result:<name>}` tokens in one string."""

    def _replacement(match: re.Match[str]) -> str:
        reader_name = match.group(1).strip()
        if not reader_name:
            return ""
        if reader_name not in reader_result_cache:
            LOG.warning("Placeholder references unknown last user message reader '%s'", reader_name)
            return ""
        return reader_result_cache[reader_name]

    return _LAST_USER_MESSAGE_READER_RESULT_TOKEN_RE.sub(_replacement, text)


def _replace_runtime_tokens(
    content: Any,
    *,
    reader_result_cache: dict[str, str],
) -> Any:
    """Replace runtime placeholders (`today`, last-user-message reader results) in message payloads."""
    today_replaced = _replace_today_tokens(content)

    if isinstance(today_replaced, str):
        return _replace_last_user_message_reader_result_tokens(
            today_replaced,
            reader_result_cache=reader_result_cache,
        )

    if isinstance(today_replaced, list):
        out: list[Any] = []
        for item in today_replaced:
            if isinstance(item, dict):
                updated = dict(item)
                if isinstance(updated.get("text"), str):
                    updated["text"] = _replace_last_user_message_reader_result_tokens(
                        updated["text"],
                        reader_result_cache=reader_result_cache,
                    )
                out.append(updated)
            else:
                out.append(item)
        return out

    return today_replaced


def _collect_last_user_message_reader_results(
    last_user_message_readers: list[Any],
    latest_user_message_text: str,
    project_root: Path,
    *,
    execute_reader: Any = _execute_last_user_message_reader,
) -> dict[str, str]:
    """Execute all configured last user message readers and return results by name."""
    results: dict[str, str] = {}
    for reader in last_user_message_readers:
        results[reader.name] = execute_reader(reader, latest_user_message_text, project_root)
    return results


def extract_latest_user_message_text(messages: list[Any]) -> str:
    """Return the latest user message text payload for placeholder substitution."""
    for msg in reversed(messages):
        if not isinstance(msg, dict):
            continue
        if str(msg.get("role") or "").lower() != "user":
            continue

        content = msg.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str):
                        parts.append(text)
            if parts:
                return "\n".join(parts)
    return ""


def prepare_messages(
    messages: list[Any],
    init_messages: list[dict[str, Any]],
    last_user_message_readers: list[Any],
    project_root: Path | None = None,
    *,
    execute_reader: Any = _execute_last_user_message_reader,
) -> list[dict[str, Any]]:
    """Prepend configured init messages and normalize runtime placeholder tokens."""
    messages = strip_direct_command_messages(messages)
    combined = [*init_messages, *messages] if init_messages else list(messages)
    prepared: list[dict[str, Any]] = []
    latest_user_message = extract_latest_user_message_text(messages)
    root = (project_root or Path(os.getcwd())).resolve()
    reader_result_cache = _collect_last_user_message_reader_results(
        last_user_message_readers,
        latest_user_message,
        root,
        execute_reader=execute_reader,
    )

    for msg in combined:
        if not isinstance(msg, dict):
            continue
        updated = dict(msg)
        if "content" in updated:
            updated["content"] = _replace_runtime_tokens(
                updated.get("content"),
                reader_result_cache=reader_result_cache,
            )
        prepared.append(updated)

    return prepared
