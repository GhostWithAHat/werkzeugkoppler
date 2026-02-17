"""Shared utility helpers for the werkzeugkoppler package."""

from __future__ import annotations

import json
from typing import Any


def to_bounded_json(payload: Any, max_len: int = 8000) -> str:
    """Serialize arbitrary values into bounded JSON-like text for logging.

    The helper never raises and truncates long payloads to keep log lines readable.
    """
    try:
        raw = json.dumps(payload, ensure_ascii=False)
    except Exception:
        raw = repr(payload)
    if len(raw) > max_len:
        return raw[:max_len] + "...<truncated>"
    return raw
