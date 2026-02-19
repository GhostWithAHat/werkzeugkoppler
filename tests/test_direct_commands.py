from werkzeugkoppler.direct_commands import (
    direct_command_name,
    extract_direct_command_text,
    extract_latest_user_direct_command,
    format_direct_command_output,
    is_allowed_direct_command,
    strip_direct_command_messages,
)


def test_extract_direct_command_text_with_trim() -> None:
    assert extract_direct_command_text(" @@ change_model gpt-4o ") == "change_model gpt-4o"
    assert extract_direct_command_text("frage") is None


def test_is_allowed_direct_command_checks_first_token() -> None:
    allowed = ["change_model", "status"]
    assert is_allowed_direct_command("change_model gpt-4o", allowed) is True
    assert is_allowed_direct_command("rm -rf /", allowed) is False


def test_is_allowed_direct_command_supports_wildcards() -> None:
    allowed = ["/usr/local/bin/*", "apt-*", "*"]
    assert is_allowed_direct_command("/usr/local/bin/query --x", allowed) is True
    assert is_allowed_direct_command("apt-get update", allowed) is True
    assert is_allowed_direct_command("something_else arg", allowed) is True


def test_direct_command_name_extracts_first_token() -> None:
    assert direct_command_name('not_allowed --flag "a b"') == "not_allowed"


def test_extract_latest_user_direct_command_ignores_trailing_assistant() -> None:
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi"},
        {"role": "user", "content": "@@ echo ok"},
        {"role": "assistant", "content": "typing..."},
    ]
    assert extract_latest_user_direct_command(messages) == "echo ok"


def test_extract_latest_user_direct_command_returns_none_when_latest_user_is_not_direct() -> None:
    messages = [
        {"role": "user", "content": "@@ echo old"},
        {"role": "assistant", "content": "done"},
        {"role": "user", "content": "normal question"},
        {"role": "assistant", "content": "typing..."},
    ]
    assert extract_latest_user_direct_command(messages) is None


def test_strip_direct_command_messages_removes_user_and_next_assistant() -> None:
    messages = [
        {"role": "user", "content": "Hallo"},
        {"role": "assistant", "content": "Hi"},
        {"role": "user", "content": " @@ change_model gpt-4o"},
        {"role": "assistant", "content": "old ack"},
        {"role": "tool", "content": "t"},
        {"role": "assistant", "content": "new answer"},
        {"role": "user", "content": "Frage"},
    ]
    cleaned = strip_direct_command_messages(messages)
    assert cleaned == [
        {"role": "user", "content": "Hallo"},
        {"role": "assistant", "content": "Hi"},
        {"role": "tool", "content": "t"},
        {"role": "assistant", "content": "new answer"},
        {"role": "user", "content": "Frage"},
    ]


def test_format_direct_command_output_appends_return_code() -> None:
    formatted = format_direct_command_output("echo hi", "ok\nwarn\n", 0)
    assert formatted == "```\n$ echo hi\nok\nwarn\n```\n(Return Code 0)"


def test_format_direct_command_output_for_multiline_shell_code() -> None:
    formatted = format_direct_command_output("echo a\necho b", "ok\nwarn\n", 3)
    assert formatted == (
        "Executed Shell Code:\n"
        "```\n"
        "$ echo a\necho b\n"
        "\n"
        "```\n"
        "---\n"
        "Output:\n"
        "```\n"
        "ok\nwarn\n"
        "```\n"
        "(Return Code 3)"
    )
