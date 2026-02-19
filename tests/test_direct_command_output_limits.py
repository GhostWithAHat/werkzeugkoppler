import asyncio
import shlex
import sys

from werkzeugkoppler.config import GatewayConfig
from werkzeugkoppler.direct_commands import (
    DIRECT_COMMAND_OUTPUT_LIMIT_BYTES_REASON,
    DIRECT_COMMAND_OUTPUT_LIMIT_LINE_REASON,
    OUTPUT_LIMIT_EXIT_CODE,
    direct_command_output_limit_notice,
    run_direct_command,
    stream_direct_command_output,
)


def _python_command(script: str) -> str:
    return f"{shlex.quote(sys.executable)} -c {shlex.quote(script)}"


def test_run_direct_command_aborts_when_line_limit_is_exceeded() -> None:
    cmd = _python_command("for i in range(20): print(i)")
    output, return_code, output_limit_reason = run_direct_command(
        cmd,
        max_output_lines=5,
        max_output_bytes=100000,
    )
    assert return_code == OUTPUT_LIMIT_EXIT_CODE
    assert output_limit_reason == DIRECT_COMMAND_OUTPUT_LIMIT_LINE_REASON
    assert "Command aborted: output exceeded limits" not in output


def test_run_direct_command_aborts_when_byte_limit_is_exceeded() -> None:
    cmd = _python_command("print('x' * 500)")
    output, return_code, output_limit_reason = run_direct_command(
        cmd,
        max_output_lines=100,
        max_output_bytes=50,
    )
    assert return_code == OUTPUT_LIMIT_EXIT_CODE
    assert output_limit_reason == DIRECT_COMMAND_OUTPUT_LIMIT_BYTES_REASON
    assert "Command aborted: output exceeded limits" not in output


def test_stream_direct_command_output_aborts_when_limit_is_exceeded() -> None:
    async def _collect() -> tuple[list[str], int | None]:
        texts: list[str] = []
        final_return_code: int | None = None
        async for event in stream_direct_command_output(
            _python_command("for i in range(20): print(i)"),
            max_output_lines=5,
            max_output_bytes=100000,
        ):
            texts.append(event.text)
            if event.is_final:
                final_return_code = event.return_code
        return texts, final_return_code

    texts, final_return_code = asyncio.run(_collect())
    assert final_return_code == OUTPUT_LIMIT_EXIT_CODE
    assert not any("Command aborted: output exceeded limits" in text for text in texts)


def test_run_direct_command_does_not_abort_when_line_limit_is_zero() -> None:
    cmd = _python_command("for i in range(200): print(i)")
    output, return_code, output_limit_reason = run_direct_command(
        cmd,
        max_output_lines=0,
        max_output_bytes=1000000,
    )
    assert return_code == 0
    assert output_limit_reason is None


def test_run_direct_command_does_not_abort_when_byte_limit_is_zero() -> None:
    cmd = _python_command("print('x' * 100000)")
    output, return_code, output_limit_reason = run_direct_command(
        cmd,
        max_output_lines=1000,
        max_output_bytes=0,
    )
    assert return_code == 0
    assert output_limit_reason is None


def test_direct_command_output_limit_notice_texts() -> None:
    assert direct_command_output_limit_notice(DIRECT_COMMAND_OUTPUT_LIMIT_LINE_REASON) == (
        "Command output reached line limit."
    )
    assert direct_command_output_limit_notice(DIRECT_COMMAND_OUTPUT_LIMIT_BYTES_REASON) == (
        "Command output reached bytes limit."
    )


def test_gateway_config_uses_default_direct_command_output_limits() -> None:
    cfg = GatewayConfig.model_validate(
        {
            "service_base_url": "http://127.0.0.1:10001",
            "upstream_base_url": "http://127.0.0.1:10000",
        }
    )
    assert cfg.allowed_direct_commands_max_output_lines == 8000
    assert cfg.allowed_direct_commands_max_output_bytes == 40000


def test_gateway_config_allows_overriding_direct_command_output_limits() -> None:
    cfg = GatewayConfig.model_validate(
        {
            "service_base_url": "http://127.0.0.1:10001",
            "upstream_base_url": "http://127.0.0.1:10000",
            "allowed_direct_commands_max_output_lines": 42,
            "allowed_direct_commands_max_output_bytes": 1234,
        }
    )
    assert cfg.allowed_direct_commands_max_output_lines == 42
    assert cfg.allowed_direct_commands_max_output_bytes == 1234


def test_gateway_config_accepts_zero_as_unlimited_output_limit() -> None:
    cfg = GatewayConfig.model_validate(
        {
            "service_base_url": "http://127.0.0.1:10001",
            "upstream_base_url": "http://127.0.0.1:10000",
            "allowed_direct_commands_max_output_lines": 0,
            "allowed_direct_commands_max_output_bytes": 0,
        }
    )
    assert cfg.allowed_direct_commands_max_output_lines == 0
    assert cfg.allowed_direct_commands_max_output_bytes == 0


def test_gateway_config_accepts_short_output_limit_aliases() -> None:
    cfg = GatewayConfig.model_validate(
        {
            "service_base_url": "http://127.0.0.1:10001",
            "upstream_base_url": "http://127.0.0.1:10000",
            "max_output_lines": 77,
            "max_output_bytes": 8888,
        }
    )
    assert cfg.allowed_direct_commands_max_output_lines == 77
    assert cfg.allowed_direct_commands_max_output_bytes == 8888
