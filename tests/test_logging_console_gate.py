"""Audit O5: the resilient wrapper redirects stdout/stderr into crash.log, so the
stderr console handler must be suppressible to avoid duplicating every rotated
line into an unbounded file.  Default (env unset) keeps the console handler.
"""

from __future__ import annotations

import logging

import lumina_quant.utils.logging_utils as logging_utils


def _console_handlers(logger: logging.Logger) -> list[logging.Handler]:
    # The RotatingFileHandler subclasses StreamHandler, so match the console
    # handler by exact type.
    return [h for h in logger.handlers if type(h) is logging.StreamHandler]


def _file_handlers(logger: logging.Logger) -> list[logging.Handler]:
    return [h for h in logger.handlers if isinstance(h, logging.FileHandler)]


def test_console_handler_present_by_default(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("LQ_DISABLE_CONSOLE_LOG", raising=False)
    monkeypatch.setenv("LQ_LOG_DIR", str(tmp_path))
    name = "lumina_test_console_default"
    logging.getLogger(name).handlers.clear()

    logger = logging_utils.setup_logging(name)

    assert _console_handlers(logger), "console handler expected by default"
    assert _file_handlers(logger), "rotating file handler expected"


def test_console_handler_suppressed_when_env_set(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("LQ_DISABLE_CONSOLE_LOG", "1")
    monkeypatch.setenv("LQ_LOG_DIR", str(tmp_path))
    name = "lumina_test_console_suppressed"
    logging.getLogger(name).handlers.clear()

    logger = logging_utils.setup_logging(name)

    # No console handler duplicating output into the wrapper's crash.log ...
    assert not _console_handlers(logger)
    # ... while structured logs are still captured to the rotating file.
    assert _file_handlers(logger)


def test_disabled_helper_parses_truthy_values(monkeypatch) -> None:
    for value in ("1", "true", "TRUE", "yes"):
        monkeypatch.setenv("LQ_DISABLE_CONSOLE_LOG", value)
        assert logging_utils._console_logging_disabled() is True
    for value in ("0", "false", "no", ""):
        monkeypatch.setenv("LQ_DISABLE_CONSOLE_LOG", value)
        assert logging_utils._console_logging_disabled() is False
    monkeypatch.delenv("LQ_DISABLE_CONSOLE_LOG", raising=False)
    assert logging_utils._console_logging_disabled() is False
