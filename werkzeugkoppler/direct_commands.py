"""Direct command parsing, formatting and execution helpers.

This module contains all logic for `@@` direct commands so HTTP orchestration
code can stay focused on request/response handling.
"""

from __future__ import annotations

import asyncio
import fnmatch
import os
import shlex
import signal
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncGenerator, Awaitable, Callable

DIRECT_COMMAND_PREFIX = "@@"
DEFAULT_DIRECT_COMMAND_MAX_OUTPUT_LINES = 2000
DEFAULT_DIRECT_COMMAND_MAX_OUTPUT_BYTES = 40000000
OUTPUT_LIMIT_EXIT_CODE = 124
DIRECT_COMMAND_OUTPUT_LIMIT_LINE_REASON = "line_limit"
DIRECT_COMMAND_OUTPUT_LIMIT_BYTES_REASON = "bytes_limit"
DIRECT_COMMAND_MULTILINE_PREFIX_TEMPLATE = (
    "Executing shell code:\n"
    "\n"
    "```\n"
    "{shell_code}\n"
    "```\n"
    "\n"
    "---\n"
    "Output:\n"
    "\n"
    "```\n"
)
DIRECT_COMMAND_MULTILINE_SUFFIX_TEMPLATE = (
    "\n"
    "```\n"
    "\n"
    "(Return Code {return_code})"
)
DIRECT_COMMAND_SINGLELINE_PREFIX_TEMPLATE = (
    "```\n"
    "$ {shell_code}\n"
)
DIRECT_COMMAND_SINGLELINE_SUFFIX_TEMPLATE = DIRECT_COMMAND_MULTILINE_SUFFIX_TEMPLATE 

@dataclass(frozen=True)
class DirectCommandStreamEvent:
    """One stream event emitted by direct command process execution."""

    text: str
    return_code: int | None = None
    output_limit_reason: str | None = None

    @property
    def is_final(self) -> bool:
        """Return true when this event contains the final process return code."""
        return self.return_code is not None


def extract_message_text(content: Any) -> str:
    """Extract plain text from supported message content shapes."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)
    return ""


def extract_direct_command_text(message_text: str) -> str | None:
    """Return command text after `@@` prefix, or None when not a direct command."""
    trimmed = message_text.strip()
    if not trimmed.startswith(DIRECT_COMMAND_PREFIX):
        return None
    return trimmed[len(DIRECT_COMMAND_PREFIX) :].strip()


def extract_direct_command_from_message(message: Any) -> str | None:
    """Return direct command payload from a user message when present."""
    if not isinstance(message, dict):
        return None
    if str(message.get("role") or "").lower() != "user":
        return None
    return extract_direct_command_text(extract_message_text(message.get("content")))


def extract_latest_user_direct_command(messages: list[Any]) -> str | None:
    """Return direct command from the latest user message, if present."""
    for msg in reversed(messages):
        if not isinstance(msg, dict):
            continue
        if str(msg.get("role") or "").lower() != "user":
            continue
        return extract_direct_command_from_message(msg)
    return None


def strip_direct_command_messages(messages: list[Any]) -> list[Any]:
    """Remove `@@...` user messages and each immediate following assistant message."""
    cleaned: list[Any] = []
    skip_next_assistant = False
    for msg in messages:
        if not isinstance(msg, dict):
            cleaned.append(msg)
            continue

        role = str(msg.get("role") or "").lower()
        if skip_next_assistant and role == "assistant":
            skip_next_assistant = False
            continue
        direct_command = extract_direct_command_from_message(msg)
        if direct_command is not None:
            skip_next_assistant = True
            continue
        cleaned.append(msg)
    return cleaned


def _parse_argv(command_text: str) -> list[str]:
    if not command_text:
        return []
    try:
        return shlex.split(command_text)
    except ValueError:
        return []


def is_allowed_direct_command(command_text: str, allowed_direct_commands: list[str]) -> bool:
    """Check whether first argv token is allowlisted (supports `*` patterns)."""
    argv = _parse_argv(command_text)
    if not argv:
        return False
    command_name = argv[0]
    for allowed in allowed_direct_commands:
        pattern = (allowed or "").strip()
        if not pattern:
            continue
        if fnmatch.fnmatchcase(command_name, pattern):
            return True
    return False


def direct_command_name(command_text: str) -> str:
    """Return first command token for user-facing error messages."""
    argv = _parse_argv(command_text)
    if argv:
        return argv[0]
    return command_text.strip().split(maxsplit=1)[0] if command_text.strip() else ""


def _count_output_lines(
    *,
    total_newlines: int,
    seen_output: bool,
    ends_with_newline: bool,
) -> int:
    return total_newlines + (1 if seen_output and not ends_with_newline else 0)


def _exceeded_output_limits(
    *,
    line_count: int,
    byte_count: int,
    max_output_lines: int,
    max_output_bytes: int,
) -> bool:
    lines_exceeded = max_output_lines > 0 and line_count > max_output_lines
    bytes_exceeded = max_output_bytes > 0 and byte_count > max_output_bytes
    return lines_exceeded or bytes_exceeded


def _output_limit_reason(
    *,
    line_count: int,
    byte_count: int,
    max_output_lines: int,
    max_output_bytes: int,
) -> str | None:
    if max_output_lines > 0 and line_count > max_output_lines:
        return DIRECT_COMMAND_OUTPUT_LIMIT_LINE_REASON
    if max_output_bytes > 0 and byte_count > max_output_bytes:
        return DIRECT_COMMAND_OUTPUT_LIMIT_BYTES_REASON
    return None


def _terminate_process_tree_sync(proc: subprocess.Popen[bytes]) -> None:
    if proc.poll() is not None:
        return
    try:
        if os.name == "nt":
            proc.terminate()
        else:
            os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=2.0)
        return
    except Exception:
        pass

    if proc.poll() is not None:
        return
    try:
        if os.name == "nt":
            proc.kill()
        else:
            os.killpg(proc.pid, signal.SIGKILL)
        proc.wait(timeout=2.0)
    except Exception:
        pass


def run_direct_command(
    command_text: str,
    *,
    cwd: Path | None = None,
    chunk_size: int = 1024,
    max_output_lines: int = DEFAULT_DIRECT_COMMAND_MAX_OUTPUT_LINES,
    max_output_bytes: int = DEFAULT_DIRECT_COMMAND_MAX_OUTPUT_BYTES,
) -> tuple[str, int, str | None]:
    """Execute direct command through shell and return combined stdout/stderr."""
    process = subprocess.Popen(
        command_text,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        shell=True,
        cwd=cwd or Path.cwd(),
        start_new_session=(os.name != "nt"),
    )
    if process.stdout is None:
        raise RuntimeError("Failed to capture direct command output stream")

    chunks: list[bytes] = []
    total_bytes = 0
    total_newlines = 0
    seen_output = False
    ends_with_newline = False
    try:
        while True:
            raw = process.stdout.read(chunk_size)
            if not raw:
                break
            chunks.append(raw)
            total_bytes += len(raw)
            total_newlines += raw.count(b"\n")
            seen_output = True
            ends_with_newline = raw.endswith(b"\n")
            line_count = _count_output_lines(
                total_newlines=total_newlines,
                seen_output=seen_output,
                ends_with_newline=ends_with_newline,
            )
            if _exceeded_output_limits(
                line_count=line_count,
                byte_count=total_bytes,
                max_output_lines=max_output_lines,
                max_output_bytes=max_output_bytes,
            ):
                _terminate_process_tree_sync(process)
                return (
                    b"".join(chunks).decode("utf-8", errors="replace"),
                    OUTPUT_LIMIT_EXIT_CODE,
                    _output_limit_reason(
                        line_count=line_count,
                        byte_count=total_bytes,
                        max_output_lines=max_output_lines,
                        max_output_bytes=max_output_bytes,
                    ),
                )
        return_code = process.wait()
        return b"".join(chunks).decode("utf-8", errors="replace"), int(return_code), None
    finally:
        if process.poll() is None:
            _terminate_process_tree_sync(process)


def direct_command_output_prefix(shell_code: str) -> str:
    """Build static prefix before direct-command output text."""
    if "\n" in shell_code:
        return DIRECT_COMMAND_MULTILINE_PREFIX_TEMPLATE.format(shell_code=shell_code)
    return DIRECT_COMMAND_SINGLELINE_PREFIX_TEMPLATE.format(shell_code=shell_code)


def direct_command_output_suffix(shell_code: str, return_code: int) -> str:
    """Build static suffix after direct-command output text."""
    if "\n" in shell_code:
        return DIRECT_COMMAND_MULTILINE_SUFFIX_TEMPLATE.format(return_code=return_code)
    return DIRECT_COMMAND_SINGLELINE_SUFFIX_TEMPLATE.format(return_code=return_code)


def format_direct_command_output(shell_code: str, combined_output: str, return_code: int) -> str:
    """Format command output for assistant response."""
    prefix = direct_command_output_prefix(shell_code)
    suffix = direct_command_output_suffix(shell_code, return_code)
    normalized_output = combined_output.rstrip("\n")
    return f"{prefix}{normalized_output}{suffix}"


def direct_command_output_limit_notice(limit_reason: str | None) -> str:
    """Build user-facing note shown outside code block for output-limit aborts."""
    if limit_reason == DIRECT_COMMAND_OUTPUT_LIMIT_LINE_REASON:
        return "Command output reached line limit."
    if limit_reason == DIRECT_COMMAND_OUTPUT_LIMIT_BYTES_REASON:
        return "Command output reached bytes limit."
    return ""


async def _terminate_process_tree(proc: asyncio.subprocess.Process) -> None:
    """Terminate process and child process tree for shell-spawned commands."""
    if proc.returncode is not None:
        return
    try:
        if os.name == "nt":
            proc.terminate()
        else:
            # Shell is started in a dedicated session; kill complete process group.
            os.killpg(proc.pid, signal.SIGTERM)
        await asyncio.wait_for(proc.wait(), timeout=2.0)
        return
    except Exception:
        pass

    if proc.returncode is not None:
        return
    try:
        if os.name == "nt":
            proc.kill()
        else:
            os.killpg(proc.pid, signal.SIGKILL)
        await asyncio.wait_for(proc.wait(), timeout=2.0)
    except Exception:
        pass


async def stream_direct_command_output(
    shell_code: str,
    *,
    cwd: Path | None = None,
    is_disconnected: Callable[[], Awaitable[bool]] | None = None,
    chunk_size: int = 1024,
    poll_timeout_seconds: float = 0.2,
    max_output_lines: int = DEFAULT_DIRECT_COMMAND_MAX_OUTPUT_LINES,
    max_output_bytes: int = DEFAULT_DIRECT_COMMAND_MAX_OUTPUT_BYTES,
) -> AsyncGenerator[DirectCommandStreamEvent, None]:
    """Stream command output and emit a final event containing return code."""
    process: asyncio.subprocess.Process | None = None
    workdir = cwd or Path.cwd()
    total_bytes = 0
    total_newlines = 0
    seen_output = False
    ends_with_newline = False
    try:
        process = await asyncio.create_subprocess_shell(
            shell_code,
            cwd=str(workdir),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            start_new_session=(os.name != "nt"),
        )
        if process.stdout is None:
            raise RuntimeError("Failed to capture direct command output stream")

        while True:
            if is_disconnected is not None and await is_disconnected():
                await _terminate_process_tree(process)
                return
            try:
                raw = await asyncio.wait_for(process.stdout.read(chunk_size), timeout=poll_timeout_seconds)
            except asyncio.TimeoutError:
                if process.returncode is not None:
                    break
                continue
            if not raw:
                break
            total_bytes += len(raw)
            total_newlines += raw.count(b"\n")
            seen_output = True
            ends_with_newline = raw.endswith(b"\n")
            line_count = _count_output_lines(
                total_newlines=total_newlines,
                seen_output=seen_output,
                ends_with_newline=ends_with_newline,
            )
            if _exceeded_output_limits(
                line_count=line_count,
                byte_count=total_bytes,
                max_output_lines=max_output_lines,
                max_output_bytes=max_output_bytes,
            ):
                await _terminate_process_tree(process)
                yield DirectCommandStreamEvent(
                    text="",
                    return_code=OUTPUT_LIMIT_EXIT_CODE,
                    output_limit_reason=_output_limit_reason(
                        line_count=line_count,
                        byte_count=total_bytes,
                        max_output_lines=max_output_lines,
                        max_output_bytes=max_output_bytes,
                    ),
                )
                return
            yield DirectCommandStreamEvent(text=raw.decode("utf-8", errors="replace"))

        return_code = await process.wait()
        yield DirectCommandStreamEvent(text="", return_code=return_code)
    finally:
        if process is not None and process.returncode is None:
            await _terminate_process_tree(process)
