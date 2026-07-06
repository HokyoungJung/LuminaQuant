"""Regression tests: Telegram bot token must never reach the logs, plus the
audit-O3 off-thread bounded-queue delivery contract.

The token is embedded in the request URL, so neither a raw exception (whose
``str``/``.url`` includes the URL) nor an HTTP error body may be logged.
Delivery uses stdlib ``urllib`` (audit O1) and runs on a background worker
(audit O3); the seam ``notification._urlopen`` is patched to avoid real I/O.
"""

from __future__ import annotations

import logging
import threading
import time
import urllib.error

import lumina_quant.utils.notification as notif_module
from lumina_quant.utils.notification import NotificationManager

_TOKEN = "987654321:AAFAKE-BOT-TOKEN-DO-NOT-LOG"
_CHAT_ID = "42"
_URL_WITH_TOKEN = f"https://api.telegram.org/bot{_TOKEN}/sendMessage"


class _OkResponse:
    status = 200

    def __enter__(self) -> _OkResponse:
        return self

    def __exit__(self, *exc) -> bool:
        return False


def _ok_urlopen(request, timeout):
    return _OkResponse()


def test_request_exception_does_not_log_token(monkeypatch, caplog) -> None:
    def _raising_urlopen(request, timeout):
        # URLError.reason can carry the token-bearing URL.
        raise urllib.error.URLError(f"connection refused to {_URL_WITH_TOKEN}")

    monkeypatch.setattr(notif_module, "_urlopen", _raising_urlopen)
    mgr = NotificationManager(_TOKEN, _CHAT_ID)

    with caplog.at_level(logging.ERROR, logger="NotificationManager"):
        mgr.send_message("hello")
        assert mgr.flush(timeout=5.0)

    combined = "\n".join(rec.getMessage() for rec in caplog.records)
    assert _TOKEN not in combined
    assert "URLError" in combined  # type name is logged instead


def test_http_error_does_not_log_token_or_body(monkeypatch, caplog) -> None:
    def _http_error_urlopen(request, timeout):
        # HTTPError carries `.url` (token) and a body that can echo context.
        raise urllib.error.HTTPError(
            url=_URL_WITH_TOKEN,
            code=401,
            msg=f"unauthorized for bot{_TOKEN}",
            hdrs=None,
            fp=None,
        )

    monkeypatch.setattr(notif_module, "_urlopen", _http_error_urlopen)
    mgr = NotificationManager(_TOKEN, _CHAT_ID)

    with caplog.at_level(logging.ERROR, logger="NotificationManager"):
        mgr.send_message("hello")
        assert mgr.flush(timeout=5.0)

    combined = "\n".join(rec.getMessage() for rec in caplog.records)
    assert _TOKEN not in combined
    assert "401" in combined  # status code is logged


def test_disabled_manager_sends_nothing(monkeypatch) -> None:
    sent = []

    def _spy_urlopen(request, timeout):
        sent.append(request)
        raise AssertionError("should not be called when disabled")

    monkeypatch.setattr(notif_module, "_urlopen", _spy_urlopen)
    mgr = NotificationManager("", "")  # disabled: no token/chat
    mgr.send_message("hello")
    assert sent == []


# --------------------------------------------------------------------------
# Audit O3: alert delivery must never run network I/O on the caller thread, and
# the outbox must be a bounded, drop-oldest queue.
# --------------------------------------------------------------------------


def test_delivery_runs_off_caller_thread_with_bounded_timeout(monkeypatch) -> None:
    seen: dict = {}
    caller_thread = threading.current_thread().name

    def _recording_urlopen(request, timeout):
        seen["thread"] = threading.current_thread().name
        seen["timeout"] = timeout
        return _OkResponse()

    monkeypatch.setattr(notif_module, "_urlopen", _recording_urlopen)
    mgr = NotificationManager(_TOKEN, _CHAT_ID)
    mgr.send_message("hi")
    assert mgr.flush(timeout=5.0)

    # Delivered on the dedicated background worker, never the caller.
    assert seen["thread"] == "NotificationSender"
    assert seen["thread"] != caller_thread
    # Short bounded socket timeout.
    assert 0 < seen["timeout"] <= 10.0


def test_send_message_does_not_block_on_slow_network(monkeypatch) -> None:
    release = threading.Event()

    def _slow_urlopen(request, timeout):
        release.wait(2.0)
        return _OkResponse()

    monkeypatch.setattr(notif_module, "_urlopen", _slow_urlopen)
    mgr = NotificationManager(_TOKEN, _CHAT_ID)

    start = time.monotonic()
    mgr.send_message("hi")
    elapsed = time.monotonic() - start
    release.set()  # let the worker finish so the daemon thread doesn't linger

    # The caller returned immediately rather than waiting for the ~2 s send.
    assert elapsed < 0.5


def test_enqueue_drops_oldest_when_full(caplog) -> None:
    # Exercise the bounded drop-oldest outbox directly (no worker draining it).
    mgr = NotificationManager(_TOKEN, _CHAT_ID, max_queue_size=2)

    with caplog.at_level(logging.WARNING, logger="NotificationManager"):
        mgr._enqueue("a")
        mgr._enqueue("b")  # queue now full
        assert mgr._queue.qsize() == 2
        mgr._enqueue("c")  # drops oldest "a", enqueues "c"

    assert mgr._dropped_count == 1
    remaining = []
    while not mgr._queue.empty():
        remaining.append(mgr._queue.get_nowait())
    assert remaining == ["b", "c"]
    combined = "\n".join(rec.getMessage() for rec in caplog.records)
    assert "dropped oldest" in combined


def test_sync_mode_delivers_inline(monkeypatch) -> None:
    calls: list[str] = []

    def _recording_urlopen(request, timeout):
        calls.append(threading.current_thread().name)
        return _OkResponse()

    monkeypatch.setattr(notif_module, "_urlopen", _recording_urlopen)
    mgr = NotificationManager(_TOKEN, _CHAT_ID, async_delivery=False)
    caller_thread = threading.current_thread().name
    mgr.send_message("hi")

    # Synchronous escape hatch: delivered inline on the caller thread.
    assert calls == [caller_thread]


def test_deliverable_reflects_enabled_state() -> None:
    assert NotificationManager(_TOKEN, _CHAT_ID).deliverable() is True
    assert NotificationManager("", "").deliverable() is False
