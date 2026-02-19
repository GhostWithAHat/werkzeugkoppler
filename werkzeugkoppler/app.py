"""HTTP application for the werkzeugkoppler gateway."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import os
import sys
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import JSONResponse
except ModuleNotFoundError as exc:  # pragma: no cover
    FastAPI = Any  # type: ignore[assignment]
    HTTPException = Exception  # type: ignore[assignment]
    Request = Any  # type: ignore[assignment]
    JSONResponse = Any  # type: ignore[assignment]
    _FASTAPI_IMPORT_ERROR: ModuleNotFoundError | None = exc
else:
    _FASTAPI_IMPORT_ERROR = None

try:
    from .chat_handlers import (
        handle_direct_command_short_circuit as _handle_direct_command_short_circuit,
        handle_regular_chat_request as _handle_regular_chat_request,
        strip_direct_command_messages_in_payload as _strip_direct_command_messages_in_payload,
    )
except ModuleNotFoundError as exc:  # pragma: no cover
    _CHAT_HANDLERS_IMPORT_ERROR: ModuleNotFoundError | None = exc
else:
    _CHAT_HANDLERS_IMPORT_ERROR = None

try:
    from .config_reload import ConfigReloadWatcher
except ModuleNotFoundError as exc:  # pragma: no cover
    ConfigReloadWatcher = Any  # type: ignore[assignment]

    _CONFIG_RELOAD_IMPORT_ERROR: ModuleNotFoundError | None = exc
else:
    _CONFIG_RELOAD_IMPORT_ERROR = None

try:
    from .gateway_service import GatewayService
except ModuleNotFoundError as exc:  # pragma: no cover
    GatewayService = Any  # type: ignore[assignment]
    _GATEWAY_SERVICE_IMPORT_ERROR: ModuleNotFoundError | None = exc
else:
    _GATEWAY_SERVICE_IMPORT_ERROR = None

try:
    from .config import DEFAULT_CONFIG_PATH, GatewayConfig, load_config
    from .json_helpers import to_bounded_json
    from .logging_utils import setup_logging
except ModuleNotFoundError as exc:  # pragma: no cover
    GatewayConfig = Any  # type: ignore[assignment]
    load_config = None  # type: ignore[assignment]
    setup_logging = None  # type: ignore[assignment]
    to_bounded_json = None  # type: ignore[assignment]
    _CORE_IMPORT_ERROR: ModuleNotFoundError | None = exc
else:
    _CORE_IMPORT_ERROR = None

LOG = logging.getLogger(__name__)


def _extract_bearer_token(request: Request) -> str | None:
    auth_header = request.headers.get("authorization")
    if not auth_header:
        return None
    parts = auth_header.strip().split(" ", 1)
    if len(parts) != 2:
        return None
    scheme, token = parts[0].lower(), parts[1].strip()
    if scheme != "bearer" or not token:
        return None
    return token


def _require_gateway_auth(request: Request, cfg: GatewayConfig) -> None:
    required_key = cfg.service_api_key
    if not required_key:
        return
    provided_key = _extract_bearer_token(request)
    if provided_key != required_key:
        raise HTTPException(status_code=401, detail="Unauthorized")


def _service_bind_addr(service_base_url: str) -> tuple[str, int]:
    parsed = urlparse(service_base_url)
    if not parsed.hostname or parsed.port is None:
        raise ValueError("service_base_url must include host and port, e.g. http://127.0.0.1:8080")
    return parsed.hostname, parsed.port


def _raise_missing_dependency(exc: ModuleNotFoundError) -> None:
    missing = exc.name or "unknown"
    raise RuntimeError(
        f"Missing dependency '{missing}'. Install runtime dependencies in your active environment."
    ) from exc


def _raise_if_missing_runtime_dependencies() -> None:
    if _CHAT_HANDLERS_IMPORT_ERROR is not None:
        _raise_missing_dependency(_CHAT_HANDLERS_IMPORT_ERROR)
    if _CONFIG_RELOAD_IMPORT_ERROR is not None:
        _raise_missing_dependency(_CONFIG_RELOAD_IMPORT_ERROR)
    if _GATEWAY_SERVICE_IMPORT_ERROR is not None:
        _raise_missing_dependency(_GATEWAY_SERVICE_IMPORT_ERROR)
    if _CORE_IMPORT_ERROR is not None:
        _raise_missing_dependency(_CORE_IMPORT_ERROR)
    if _FASTAPI_IMPORT_ERROR is not None:
        raise RuntimeError(
            "Missing dependency 'fastapi'. Install runtime dependencies in your active environment."
        ) from _FASTAPI_IMPORT_ERROR


@dataclass(slots=True)
class _AppRuntime:
    """Mutable app runtime state shared across routes and lifespan."""

    service: GatewayService
    config_file: Path
    config_watcher: ConfigReloadWatcher | None = None
    reload_task: asyncio.Task[None] | None = None

    async def apply_reloaded_config(self, changed_file: Path) -> None:
        new_cfg = load_config(str(changed_file))
        setup_logging(new_cfg.logging)
        async with self.service._op_lock:
            await self.service.reload(new_cfg)

    @asynccontextmanager
    async def lifespan(self, _app: FastAPI):
        assert self.config_watcher is not None
        await self.service.start()
        self.reload_task = asyncio.create_task(self.config_watcher.run_forever())
        try:
            yield
        finally:
            if self.reload_task:
                self.reload_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self.reload_task
            await self.service.close()

    async def healthz(self) -> JSONResponse:
        health = await self.service.registry.get_health()
        return JSONResponse(
            {
                "service": "werkzeugkoppler",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                **health,
            }
        )

    async def list_models(self, request: Request) -> JSONResponse:
        try:
            _require_gateway_auth(request, self.service.cfg)
            models = await self.service.list_models()
            return JSONResponse(models)
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"Failed to fetch models: {exc}") from exc

    async def chat_completions(self, request: Request):
        _require_gateway_auth(request, self.service.cfg)
        payload = await request.json()
        client_host = getattr(getattr(request, "client", None), "host", None)
        LOG.debug(
            "incoming chat.completions request client=%s payload=%s",
            client_host,
            to_bounded_json(payload),
        )

        direct_response = await _handle_direct_command_short_circuit(
            request=request,
            payload=payload,
            service=self.service,
        )
        if direct_response is not None:
            return direct_response

        payload = _strip_direct_command_messages_in_payload(payload)
        return await _handle_regular_chat_request(
            request=request,
            payload=payload,
            service=self.service,
        )


def _build_runtime(config_path: str | None) -> _AppRuntime:
    assert load_config is not None
    cfg = load_config(config_path)
    setup_logging(cfg.logging)
    service = GatewayService(cfg)
    config_file = Path(config_path or os.getenv("WERKZEUGKOPPLER_CONFIG") or DEFAULT_CONFIG_PATH)
    runtime = _AppRuntime(
        service=service,
        config_file=config_file,
    )
    runtime.config_watcher = ConfigReloadWatcher(
        config_file=runtime.config_file,
        on_reload=runtime.apply_reloaded_config,
        logger=LOG,
    )
    return runtime


def create_app(config_path: str | None = None) -> FastAPI:
    """Create and configure the FastAPI application instance."""
    _raise_if_missing_runtime_dependencies()
    runtime = _build_runtime(config_path)
    app = FastAPI(title="werkzeugkoppler", version="0.1.0", lifespan=runtime.lifespan)

    @app.get("/healthz")
    async def healthz() -> JSONResponse:
        return await runtime.healthz()

    @app.get("/v1/models")
    async def v1_models(request: Request) -> JSONResponse:
        return await runtime.list_models(request)

    @app.post("/v1/chat/completions")
    async def v1_chat_completions(request: Request):
        return await runtime.chat_completions(request)

    return app


def _cli_fail(message: str, exit_code: int = 2) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(exit_code)


def _parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="werkzeugkoppler gateway")
    parser.add_argument("--config", default=None, help="Path to config YAML")
    return parser.parse_args()


def _load_cli_config(config_path: str | None) -> GatewayConfig:
    try:
        from pydantic import ValidationError
    except ModuleNotFoundError:
        _cli_fail("Missing dependency 'pydantic'. Install runtime dependencies in your active environment.")

    assert load_config is not None
    try:
        return load_config(config_path)
    except ValidationError as exc:
        missing = []
        for err in exc.errors():
            if err.get("type") == "missing":
                location = ".".join(str(x) for x in err.get("loc", []))
                missing.append(location)
        if missing:
            _cli_fail(
                "Configuration incomplete. Missing required fields: "
                + ", ".join(sorted(set(missing)))
                + ". Provide --config <file> or set env vars "
                + "(WERKZEUGKOPPLER_UPSTREAM_BASE_URL)."
            )
        _cli_fail(f"Invalid configuration: {exc}")
    except Exception as exc:
        _cli_fail(f"Failed to load configuration: {exc}")
    raise AssertionError("unreachable")


def _load_uvicorn_module() -> Any:
    try:
        import uvicorn
    except ModuleNotFoundError:
        _cli_fail("Missing dependency 'uvicorn'. Install runtime dependencies in your active environment.")
    return uvicorn


def _run_uvicorn(app: FastAPI, cfg: GatewayConfig, uvicorn_module: Any) -> None:
    host, port = _service_bind_addr(cfg.service_base_url)
    uvicorn_module.run(
        app,
        host=host,
        port=port,
        log_config=None,
        # Prevent endless shutdown waits when a stream is stuck in infinite retries.
        timeout_graceful_shutdown=5,
    )


def main() -> None:
    """CLI entry point that validates environment and runs uvicorn."""
    args = _parse_cli_args()

    try:
        _raise_if_missing_runtime_dependencies()
    except RuntimeError as exc:
        _cli_fail(str(exc))

    cfg = _load_cli_config(args.config)
    uvicorn_module = _load_uvicorn_module()

    try:
        app = create_app(args.config)
    except RuntimeError as exc:
        _cli_fail(str(exc))
    except Exception as exc:
        _cli_fail(f"Failed to create app: {exc}")

    try:
        _run_uvicorn(app, cfg, uvicorn_module)
    except Exception as exc:
        _cli_fail(f"Server failed to start: {exc}", exit_code=1)


if __name__ == "__main__":
    main()
