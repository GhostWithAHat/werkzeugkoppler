import asyncio

from werkzeugkoppler.config import GatewayConfig
from werkzeugkoppler.gateway_service import GatewayService


def _make_cfg(**overrides: object) -> GatewayConfig:
    raw = {
        "service_base_url": "http://127.0.0.1:10001",
        "upstream_base_url": "http://127.0.0.1:10000",
    }
    raw.update(overrides)
    return GatewayConfig.model_validate(raw)


def test_list_models_falls_back_to_last_active_model_when_upstream_is_down(monkeypatch) -> None:
    service = GatewayService(_make_cfg())
    service._last_active_model = "active-model"

    async def fail_list_models() -> dict:
        raise RuntimeError("down")

    monkeypatch.setattr(service.upstream, "list_models", fail_list_models)
    try:
        models = asyncio.run(service.list_models())
        assert models["data"][0]["id"] == "active-model"
    finally:
        asyncio.run(service.upstream.close())


def test_list_models_falls_back_to_fake_model_when_never_active(monkeypatch) -> None:
    service = GatewayService(_make_cfg(fallback_fake_model_name="offline-fake"))

    async def fail_list_models() -> dict:
        raise RuntimeError("down")

    monkeypatch.setattr(service.upstream, "list_models", fail_list_models)
    try:
        models = asyncio.run(service.list_models())
        assert models["data"][0]["id"] == "offline-fake"
    finally:
        asyncio.run(service.upstream.close())


def test_resolve_default_model_returns_default_fake_model_name_when_upstream_is_unavailable(monkeypatch) -> None:
    service = GatewayService(_make_cfg())

    async def fail_list_models() -> dict:
        raise RuntimeError("down")

    monkeypatch.setattr(service.upstream, "list_models", fail_list_models)
    try:
        resolved = asyncio.run(service.resolve_default_model())
        assert resolved == "werkzeugkoppler"
    finally:
        asyncio.run(service.upstream.close())


def test_list_models_success_updates_last_active_model(monkeypatch) -> None:
    service = GatewayService(_make_cfg())

    async def ok_list_models() -> dict:
        return {"object": "list", "data": [{"id": "online-model", "object": "model"}]}

    monkeypatch.setattr(service.upstream, "list_models", ok_list_models)
    try:
        _ = asyncio.run(service.list_models())
        assert service._last_active_model == "online-model"
    finally:
        asyncio.run(service.upstream.close())
