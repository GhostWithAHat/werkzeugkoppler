from pathlib import Path

from werkzeugkoppler.config import LastUserMessageReaderConfig
from werkzeugkoppler.message_preparation import prepare_messages


def test_prepare_messages_replaces_today_and_last_user_message_reader_tokens_with_whitespace(monkeypatch) -> None:
    calls: list[tuple[str, str]] = []

    def fake_execute_last_user_message_reader(
        reader: LastUserMessageReaderConfig, latest_user_message_text: str, project_root: Path
    ) -> str:
        calls.append((reader.name, latest_user_message_text))
        return f"result:{reader.name}:{latest_user_message_text}"

    monkeypatch.setattr(
        "werkzeugkoppler.message_preparation._execute_last_user_message_reader",
        fake_execute_last_user_message_reader,
    )

    init_messages = [
        {
            "role": "system",
            "content": "Date: { today } | A={ last_user_message_reader_result : bio }",
        }
    ]
    messages = [{"role": "user", "content": "letzte nutzernachricht"}]
    last_user_message_readers = [
        LastUserMessageReaderConfig(name="bio", command="/bin/echo", arguments=["$LAST_USER_MESSAGE"])
    ]

    prepared = prepare_messages(messages, init_messages, last_user_message_readers)
    assert len(prepared) == 2
    assert "Date: " in prepared[0]["content"]
    assert "A=result:bio:letzte nutzernachricht" in prepared[0]["content"]
    assert calls == [("bio", "letzte nutzernachricht")]


def test_prepare_messages_uses_latest_user_message_for_last_user_message_reader_replacement(monkeypatch) -> None:
    observed_latest_user_message_text: list[str] = []

    def fake_execute_last_user_message_reader(
        reader: LastUserMessageReaderConfig, latest_user_message_text: str, project_root: Path
    ) -> str:
        observed_latest_user_message_text.append(latest_user_message_text)
        return "ok"

    monkeypatch.setattr(
        "werkzeugkoppler.message_preparation._execute_last_user_message_reader",
        fake_execute_last_user_message_reader,
    )

    init_messages = [{"role": "system", "content": "{last_user_message_reader_result:bio}"}]
    messages = [
        {"role": "user", "content": "erste frage"},
        {"role": "assistant", "content": "antwort"},
        {"role": "user", "content": "zweite frage"},
    ]
    last_user_message_readers = [
        LastUserMessageReaderConfig(name="bio", command="/bin/echo", arguments=["$LAST_USER_MESSAGE"])
    ]

    prepare_messages(messages, init_messages, last_user_message_readers)
    assert observed_latest_user_message_text == ["zweite frage"]


def test_prepare_messages_replaces_text_blocks_in_content_list(monkeypatch) -> None:
    def fake_execute_last_user_message_reader(
        reader: LastUserMessageReaderConfig, latest_user_message_text: str, project_root: Path
    ) -> str:
        return "bio-daten"

    monkeypatch.setattr(
        "werkzeugkoppler.message_preparation._execute_last_user_message_reader",
        fake_execute_last_user_message_reader,
    )

    init_messages = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "Heute: {today}"},
                {"type": "text", "text": "Profil: { last_user_message_reader_result : bio }"},
            ],
        }
    ]
    messages = [{"role": "user", "content": "x"}]
    last_user_message_readers = [
        LastUserMessageReaderConfig(name="bio", command="/bin/echo", arguments=["$LAST_USER_MESSAGE"])
    ]

    prepared = prepare_messages(messages, init_messages, last_user_message_readers)
    content = prepared[0]["content"]
    assert isinstance(content, list)
    assert "Heute: " in content[0]["text"]
    assert content[1]["text"] == "Profil: bio-daten"


def test_prepare_messages_executes_unreferenced_last_user_message_readers(monkeypatch) -> None:
    calls: list[str] = []

    def fake_execute_last_user_message_reader(
        reader: LastUserMessageReaderConfig, latest_user_message_text: str, project_root: Path
    ) -> str:
        calls.append(reader.name)
        return f"result:{reader.name}"

    monkeypatch.setattr(
        "werkzeugkoppler.message_preparation._execute_last_user_message_reader",
        fake_execute_last_user_message_reader,
    )

    init_messages = [{"role": "system", "content": "Nur bio: {last_user_message_reader_result:bio}"}]
    messages = [{"role": "user", "content": "frage"}]
    last_user_message_readers = [
        LastUserMessageReaderConfig(name="bio", command="/bin/echo", arguments=["$LAST_USER_MESSAGE"]),
        LastUserMessageReaderConfig(name="unused", command="/bin/echo", arguments=["$LAST_USER_MESSAGE"]),
    ]

    prepared = prepare_messages(messages, init_messages, last_user_message_readers)
    assert prepared[0]["content"] == "Nur bio: result:bio"
    assert calls == ["bio", "unused"]
