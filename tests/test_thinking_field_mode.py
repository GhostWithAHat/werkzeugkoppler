from werkzeugkoppler.app import (
    _chunk_with_content_delta,
    _content_to_thinking_value,
    _passthrough_non_content_delta_chunk,
    _reasoning_chunk,
    _split_content_for_stream,
)


def _delta(chunk: dict) -> dict:
    return chunk["choices"][0]["delta"]


def test_reasoning_passthrough_keeps_reasoning_field() -> None:
    chunk = _reasoning_chunk(
        completion_id="chatcmpl-demo",
        model="demo-model",
        created=1,
        value="thinking-a",
        preferred_field="reasoning",
    )
    assert _delta(chunk) == {"reasoning": "thinking-a"}


def test_reasoning_passthrough_keeps_reasoning_content_field() -> None:
    chunk = _reasoning_chunk(
        completion_id="chatcmpl-demo",
        model="demo-model",
        created=1,
        value="thinking-b",
        preferred_field="reasoning_content",
    )
    assert _delta(chunk) == {"reasoning_content": "thinking-b"}


def test_reasoning_passthrough_preserves_non_string_values() -> None:
    chunk = _reasoning_chunk(
        completion_id="chatcmpl-demo",
        model="demo-model",
        created=1,
        value={"tokens": ["a", "b"]},
        preferred_field="reasoning_content",
    )
    assert _delta(chunk) == {"reasoning_content": {"tokens": ["a", "b"]}}


def test_passthrough_non_content_delta_keeps_unknown_fields_and_drops_content_and_tool_calls() -> None:
    upstream_chunk = {
        "id": "upstream-id",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "upstream-model",
        "choices": [
            {
                "index": 0,
                "delta": {
                    "reasoning": "r",
                    "foo": {"bar": 1},
                    "content": "will-be-buffered",
                    "tool_calls": [{"index": 0}],
                },
                "finish_reason": "stop",
            }
        ],
    }
    out = _passthrough_non_content_delta_chunk(
        upstream_chunk,
        completion_id="chatcmpl-local",
        model="local-model",
    )
    assert out is not None
    assert out["id"] == "chatcmpl-local"
    assert out["model"] == "local-model"
    assert _delta(out) == {"reasoning": "r", "foo": {"bar": 1}}
    assert out["choices"][0]["finish_reason"] is None


def test_split_content_for_stream_separates_reasoning_blocks_from_final_blocks() -> None:
    content = [
        {"type": "reasoning", "text": "thinking"},
        {"type": "output_text", "text": "final answer"},
    ]
    reasoning, final = _split_content_for_stream(content)
    assert reasoning == [{"type": "reasoning", "text": "thinking"}]
    assert final == [{"type": "output_text", "text": "final answer"}]


def test_chunk_with_content_delta_rewrites_identity_and_sets_content() -> None:
    upstream_chunk = {
        "id": "u1",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "m1",
        "choices": [{"index": 0, "delta": {"content": "x"}, "finish_reason": None}],
    }
    out = _chunk_with_content_delta(
        upstream_chunk,
        completion_id="chatcmpl-local",
        model="local-model",
        content_value=[{"type": "reasoning", "text": "r"}],
        finish_reason=None,
    )
    assert out is not None
    assert out["id"] == "chatcmpl-local"
    assert out["model"] == "local-model"
    assert _delta(out) == {"content": [{"type": "reasoning", "text": "r"}]}


def test_content_to_thinking_value_from_string_and_blocks() -> None:
    assert _content_to_thinking_value("abc") == "abc"
    assert _content_to_thinking_value([{"type": "output_text", "text": "a"}, {"text": "b"}]) == "ab"
