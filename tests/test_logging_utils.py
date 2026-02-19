import logging

from werkzeugkoppler.config import LoggingConfig
from werkzeugkoppler.logging_utils import setup_logging


def test_setup_logging_forces_noisy_third_party_loggers_to_configured_level() -> None:
    noisy = logging.getLogger("httpcore.http11")
    noisy.setLevel(logging.DEBUG)
    noisy.addHandler(logging.StreamHandler())
    noisy.propagate = False

    setup_logging(LoggingConfig(level="INFO", json=False))

    assert logging.getLogger().level == logging.INFO
    assert noisy.level == logging.INFO
    assert noisy.handlers == []
    assert noisy.propagate is True


def test_setup_logging_forces_watchdog_loggers_to_configured_level() -> None:
    noisy = logging.getLogger("watchdog.observers.inotify_buffer")
    noisy.setLevel(logging.DEBUG)
    noisy.addHandler(logging.StreamHandler())
    noisy.propagate = False

    setup_logging(LoggingConfig(level="INFO", json=False))

    assert noisy.level == logging.INFO
    assert noisy.handlers == []
    assert noisy.propagate is True
