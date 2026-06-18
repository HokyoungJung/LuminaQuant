"""Regression tests: Telegram bot token must never reach the logs.

Covers credential-exposure audit finding #3 — the token is embedded in the
request URL, so neither the raw exception (whose ``str`` includes the URL) nor
the HTTP error body may be logged.
"""

from __future__ import annotations

import logging

import lumina_quant.utils.notification as notif_module
from lumina_quant.utils.notification import NotificationManager

_TOKEN = "987654321:AAFAKE-BOT-TOKEN-DO-NOT-LOG"
_CHAT_ID = "42"


class _RaisingRequests:
    def post(self, url, json, timeout):
        # Real requests exceptions stringify to include the full URL (+token).
        raise RuntimeError(f"connection refused to {url}")


class _ErrorResponse:
    status_code = 401

    @property
    def text(self) -> str:
        # Telegram error bodies can echo request context; must not be logged.
        return f"unauthorized for bot{_TOKEN}"


class _ErrorRequests:
    def __init__(self) -> None:
        self.last_url = None

    def post(self, url, json, timeout):
        self.last_url = url
        return _ErrorResponse()


def test_request_exception_does_not_log_token(monkeypatch, caplog) -> None:
    monkeypatch.setattr(notif_module, "requests", _RaisingRequests())
    mgr = NotificationManager(_TOKEN, _CHAT_ID)

    with caplog.at_level(logging.ERROR, logger="NotificationManager"):
        mgr.send_message("hello")

    combined = "\n".join(rec.getMessage() for rec in caplog.records)
    assert _TOKEN not in combined
    assert "RuntimeError" in combined  # type name is logged instead


def test_http_error_does_not_log_token_or_body(monkeypatch, caplog) -> None:
    fake = _ErrorRequests()
    monkeypatch.setattr(notif_module, "requests", fake)
    mgr = NotificationManager(_TOKEN, _CHAT_ID)

    with caplog.at_level(logging.ERROR, logger="NotificationManager"):
        mgr.send_message("hello")

    combined = "\n".join(rec.getMessage() for rec in caplog.records)
    assert _TOKEN not in combined
    assert "401" in combined  # status code is logged
    # The URL still carries the token, but it must not have been logged.
    assert _TOKEN in (fake.last_url or "")


def test_disabled_manager_sends_nothing(monkeypatch) -> None:
    sent = []

    class _Spy:
        def post(self, url, json, timeout):
            sent.append(url)
            raise AssertionError("should not be called when disabled")

    monkeypatch.setattr(notif_module, "requests", _Spy())
    mgr = NotificationManager("", "")  # disabled: no token/chat
    mgr.send_message("hello")
    assert sent == []
