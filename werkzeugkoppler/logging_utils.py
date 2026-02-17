"""Logging setup helpers for werkzeugkoppler."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from .config import LoggingConfig


class JsonLogFormatter(logging.Formatter):
    """Format log records as JSON lines."""

    def format(self, record: logging.LogRecord) -> str:
        """Render one log record as JSON."""
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def setup_logging(cfg: LoggingConfig) -> None:
    """Configure the root logger from runtime configuration."""
    root = logging.getLogger()
    root.setLevel(getattr(logging, cfg.level.upper(), logging.INFO))

    handler = logging.StreamHandler()
    if cfg.json_logs:
        handler.setFormatter(JsonLogFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))

    root.handlers.clear()
    root.addHandler(handler)
