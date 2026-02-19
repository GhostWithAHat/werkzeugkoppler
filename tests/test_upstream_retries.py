import asyncio
import json

import httpx

from werkzeugkoppler.gateway_service import GatewayService
from werkzeugkoppler.config import GatewayConfig
from werkzeugkoppler.upstream import UpstreamClient


def _make_cfg(**overrides: object) -> GatewayConfig:
    raw = {
        "service_base_url": "http://127.0.0.1:10001",
        "upstream_base_url": "http://127.0.0.1:10000",
    }
    raw.update(overrides)
    return GatewayConfig.model_validate(raw)


class _FakeResponse:
    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, object]:
        return {"ok": True}


class _FakeClient:
    def __init__(self, failures_before_success: int) -> None:
        self.failures_before_success = failures_before_success
        self.calls = 0

    async def post(self, *_args, **_kwargs) -> _FakeResponse:
        self.calls += 1
        if self.calls <= self.failures_before_success:
            req = httpx.Request("POST", "http://127.0.0.1:10000/v1/chat/completions")
            raise httpx.ConnectError("upstream down", request=req)
        return _FakeResponse()

    async def aclose(self) -> None:
        return None


class _FakeStreamResponse:
    def raise_for_status(self) -> None:
        return None

    async def aiter_lines(self):
        yield 'data: {"id":"x","choices":[{"index":0,"delta":{"content":"ok"},"finish_reason":"stop"}]}'
        yield "data: [DONE]"

    async def aclose(self) -> None:
        return None


class _FailOnceStreamClient:
    def __init__(self) -> None:
        self.calls = 0

    def build_request(self, *_args, **_kwargs) -> object:
        return object()

    async def send(self, *_args, **_kwargs):
        self.calls += 1
        if self.calls == 1:
            req = httpx.Request("POST", "http://127.0.0.1:10000/v1/chat/completions")
            raise httpx.ConnectError("upstream down", request=req)
        return _FakeStreamResponse()

    async def aclose(self) -> None:
        return None


class _FailOncePayloadIncompleteStreamClient:
    def __init__(self) -> None:
        self.calls = 0

    def build_request(self, *_args, **_kwargs) -> object:
        return object()

    async def send(self, *_args, **_kwargs):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError(
                "Response payload is not completed: <TransferEncodingError: 400, "
                "message='Not enough data to satisfy transfer length header.'>"
            )
        return _FakeStreamResponse()

    async def aclose(self) -> None:
        return None


class _FailOnceModelNotFoundStreamClient:
    def __init__(self) -> None:
        self.calls = 0

    def build_request(self, *_args, **_kwargs) -> object:
        return object()

    async def send(self, *_args, **_kwargs):
        self.calls += 1
        if self.calls == 1:
            req = httpx.Request("POST", "http://127.0.0.1:10000/v1/chat/completions")
            resp = httpx.Response(404, request=req, json={"error": {"message": "Model not found"}})
            raise httpx.HTTPStatusError("Model not found", request=req, response=resp)
        return _FakeStreamResponse()

    async def aclose(self) -> None:
        return None


def test_upstream_connect_retries_defaults_to_zero() -> None:
    cfg = _make_cfg()
    assert cfg.upstream_connect_retries == 0


def test_chat_completion_no_retry_when_upstream_connect_retries_is_zero() -> None:
    client = UpstreamClient(_make_cfg(upstream_connect_retries=0, upstream_retry_interval_ms=0))
    fake = _FakeClient(failures_before_success=1)
    client._client = fake
    try:
        try:
            asyncio.run(client.chat_completion({"model": "x", "messages": []}))
            raise AssertionError("chat_completion should fail on first connect error")
        except httpx.ConnectError:
            pass
        assert fake.calls == 1
    finally:
        asyncio.run(client.close())


def test_chat_completion_retries_infinitely_when_upstream_connect_retries_is_negative() -> None:
    client = UpstreamClient(_make_cfg(upstream_connect_retries=-1, upstream_retry_interval_ms=0))
    fake = _FakeClient(failures_before_success=5)
    client._client = fake
    try:
        response = asyncio.run(client.chat_completion({"model": "x", "messages": []}))
        assert response == {"ok": True}
        assert fake.calls == 6
    finally:
        asyncio.run(client.close())


def test_stream_chat_completion_emits_retry_notice_before_retry() -> None:
    client = UpstreamClient(_make_cfg(upstream_connect_retries=1, upstream_retry_interval_ms=250))
    stream_client = _FailOnceStreamClient()
    client._build_client = lambda: stream_client  # type: ignore[method-assign]

    async def _collect() -> list[dict[str, object]]:
        out: list[dict[str, object]] = []
        async for item in client.stream_chat_completion({"model": "x", "messages": []}):
            out.append(item)
        return out

    try:
        items = asyncio.run(_collect())
        retry_notices = [item for item in items if "_wk_retry_notice" in item]
        assert len(retry_notices) == 1
        assert retry_notices[0]["_wk_retry_notice"] == {
            "retry_number": 1,
            "retry_delay_ms": 250,
            "retry_reason": "connection_failed",
        }
    finally:
        asyncio.run(client.close())


def test_stream_chat_completion_emits_payload_incomplete_retry_notice_before_retry() -> None:
    client = UpstreamClient(_make_cfg(upstream_connect_retries=1, upstream_retry_interval_ms=125))
    stream_client = _FailOncePayloadIncompleteStreamClient()
    client._build_client = lambda: stream_client  # type: ignore[method-assign]

    async def _collect() -> list[dict[str, object]]:
        out: list[dict[str, object]] = []
        async for item in client.stream_chat_completion({"model": "x", "messages": []}):
            out.append(item)
        return out

    try:
        items = asyncio.run(_collect())
        retry_notices = [item for item in items if "_wk_retry_notice" in item]
        assert len(retry_notices) == 1
        assert retry_notices[0]["_wk_retry_notice"] == {
            "retry_number": 1,
            "retry_delay_ms": 125,
            "retry_reason": "payload_incomplete",
        }
    finally:
        asyncio.run(client.close())


def test_stream_chat_completion_retries_on_http_404_model_not_found() -> None:
    client = UpstreamClient(_make_cfg(upstream_connect_retries=1, upstream_retry_interval_ms=10))
    stream_client = _FailOnceModelNotFoundStreamClient()
    client._build_client = lambda: stream_client  # type: ignore[method-assign]

    async def _collect() -> list[dict[str, object]]:
        out: list[dict[str, object]] = []
        async for item in client.stream_chat_completion({"model": "x", "messages": []}):
            out.append(item)
        return out

    try:
        items = asyncio.run(_collect())
        retry_notices = [item for item in items if "_wk_retry_notice" in item]
        assert len(retry_notices) == 1
        assert retry_notices[0]["_wk_retry_notice"]["retry_number"] == 1
        assert retry_notices[0]["_wk_retry_notice"]["retry_delay_ms"] == 10
        assert retry_notices[0]["_wk_retry_notice"]["retry_reason"] == "model_not_found"
    finally:
        asyncio.run(client.close())


def test_stream_chat_handles_exhausted_connect_retries_without_crashing() -> None:
    service = GatewayService(_make_cfg(upstream_connect_retries=0, upstream_retry_interval_ms=0))

    async def _no_tools() -> list[dict[str, object]]:
        return []

    async def _failing_stream(*_args, **_kwargs):
        req = httpx.Request("POST", "http://127.0.0.1:10000/v1/chat/completions")
        raise httpx.ConnectError("All connection attempts failed", request=req)
        yield {}

    service.registry.get_openai_tools = _no_tools  # type: ignore[method-assign]
    service.upstream.stream_chat_completion = _failing_stream  # type: ignore[method-assign]

    async def _collect() -> list[bytes]:
        out: list[bytes] = []
        async for item in service.stream_chat(
            {
                "stream": True,
                "model": "x",
                "messages": [{"role": "user", "content": "hello"}],
            }
        ):
            out.append(item)
        return out

    try:
        events = asyncio.run(_collect())
        assert events[-1] == b"data: [DONE]\n\n"

        data_events = [e for e in events if e.startswith(b"data: {")]
        assert len(data_events) >= 1
        payload = json.loads(data_events[-1][len("data: ") :].decode("utf-8"))
        delta = payload["choices"][0]["delta"]
        assert "Connection to upstream LLM failed. Retries exhausted." in delta["content"]
    finally:
        asyncio.run(service.upstream.close())


def test_stream_chat_retry_line_is_compacted_and_closed_before_content() -> None:
    service = GatewayService(_make_cfg(upstream_connect_retries=5, upstream_retry_interval_ms=0))

    async def _no_tools() -> list[dict[str, object]]:
        return []

    async def _stream_with_retries(*_args, **_kwargs):
        yield {"_wk_retry_notice": {"retry_reason": "connection_failed", "retry_number": 1, "retry_delay_ms": 0}}
        yield {"_wk_retry_notice": {"retry_reason": "connection_failed", "retry_number": 2, "retry_delay_ms": 0}}
        yield {
            "id": "x",
            "model": "x",
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
        }

    service.registry.get_openai_tools = _no_tools  # type: ignore[method-assign]
    service.upstream.stream_chat_completion = _stream_with_retries  # type: ignore[method-assign]

    async def _collect() -> list[bytes]:
        out: list[bytes] = []
        async for item in service.stream_chat(
            {
                "stream": True,
                "model": "x",
                "messages": [{"role": "user", "content": "hello"}],
            }
        ):
            out.append(item)
        return out

    try:
        events = asyncio.run(_collect())
        parsed_payloads: list[dict[str, object]] = []
        for event in events:
            if not event.startswith(b"data: {"):
                continue
            parsed_payloads.append(json.loads(event[len("data: ") :].decode("utf-8")))

        reasoning_values: list[str] = []
        content_values: list[str] = []
        for payload in parsed_payloads:
            delta = payload["choices"][0]["delta"]
            if isinstance(delta, dict):
                reasoning = delta.get("reasoning")
                content = delta.get("content")
                if isinstance(reasoning, str):
                    reasoning_values.append(reasoning)
                if isinstance(content, str):
                    content_values.append(content)

        assert reasoning_values[0] == "Connection to upstream LLM failed. Retrying "
        assert reasoning_values[1] == "."
        assert "\n" in reasoning_values
        assert content_values == ["ok"]
    finally:
        asyncio.run(service.upstream.close())


def test_non_stream_returns_assistant_failure_message_when_retries_exhausted() -> None:
    service = GatewayService(_make_cfg(upstream_connect_retries=0, upstream_retry_interval_ms=0))

    async def _no_tools() -> list[dict[str, object]]:
        return []

    async def _failing_chat_completion(*_args, **_kwargs):
        req = httpx.Request("POST", "http://127.0.0.1:10000/v1/chat/completions")
        resp = httpx.Response(404, request=req, json={"error": {"message": "Model not found"}})
        raise httpx.HTTPStatusError("Model not found", request=req, response=resp)

    service.registry.get_openai_tools = _no_tools  # type: ignore[method-assign]
    service.upstream.chat_completion = _failing_chat_completion  # type: ignore[method-assign]

    async def _run() -> dict[str, object]:
        return await service.resolve_chat_non_stream(
            {
                "stream": False,
                "model": "x",
                "messages": [{"role": "user", "content": "hello"}],
            }
        )

    try:
        resolved = asyncio.run(_run())
        response = resolved["response"]
        assert isinstance(response, dict)
        choices = response.get("choices")
        assert isinstance(choices, list) and choices
        message = choices[0]["message"]
        assert "Upstream LLM model not found. Retries exhausted." in message["content"]
    finally:
        asyncio.run(service.upstream.close())
