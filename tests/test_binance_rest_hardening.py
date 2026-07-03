"""Binance REST client hardening (2026-07-03 audit fix #3d).

Covers the client-side throttle, bounded 429 retries honoring Retry-After,
the never-retry 418 (IP auto-ban) rule, and the -1021 clock-offset resync.
"""

import io
import time
import unittest
import urllib.error
import urllib.request
from email.message import Message
from unittest.mock import patch

from lumina_quant.exchanges.binance_futures_client import (
    BinanceFuturesAPIError,
    BinanceFuturesClientConfig,
    BinanceFuturesRESTClient,
)


def _http_error(code, *, body=b"{}", headers=None):
    message = Message()
    for key, value in (headers or {}).items():
        message[key] = value
    return urllib.error.HTTPError(
        url="https://x", code=code, msg=str(code), hdrs=message, fp=io.BytesIO(body)
    )


def _client(**overrides):
    defaults = {
        "api_key": "k",
        "secret_key": "s",
        "min_request_interval_ms": 0.0,
        "sync_server_time": False,
    }
    defaults.update(overrides)
    return BinanceFuturesRESTClient(BinanceFuturesClientConfig(**defaults))


class TestThrottle(unittest.TestCase):
    def test_min_interval_enforced(self):
        client = _client(min_request_interval_ms=40.0)
        client._throttle()
        start = time.monotonic()
        client._throttle()
        self.assertGreaterEqual(time.monotonic() - start, 0.035)

    def test_zero_interval_is_free(self):
        client = _client()
        start = time.monotonic()
        for _ in range(5):
            client._throttle()
        self.assertLess(time.monotonic() - start, 0.05)


class TestHttpErrorPolicy(unittest.TestCase):
    def test_429_retries_with_retry_after_then_succeeds(self):
        client = _client(max_429_retries=2)
        calls = {"n": 0}

        def fake_urlopen(request, timeout=None):
            _ = (request, timeout)
            calls["n"] += 1
            if calls["n"] == 1:
                raise _http_error(429, headers={"Retry-After": "0"})

            class _Resp:
                def __enter__(self):
                    return self

                def __exit__(self, *args):
                    return False

                @staticmethod
                def read():
                    return b'{"ok": true}'

            return _Resp()

        with (
            patch.object(urllib.request, "urlopen", fake_urlopen),
            patch.object(time, "sleep", lambda *_: None),
        ):
            result = client.public_get("/fapi/v1/ping")
        self.assertEqual(result, {"ok": True})
        self.assertEqual(calls["n"], 2)

    def test_429_gives_up_after_bounded_retries(self):
        client = _client(max_429_retries=1)

        def always_429(request, timeout=None):
            _ = (request, timeout)
            raise _http_error(429)

        with (
            patch.object(urllib.request, "urlopen", always_429),
            patch.object(time, "sleep", lambda *_: None),
            self.assertRaises(BinanceFuturesAPIError) as ctx,
        ):
            client.public_get("/fapi/v1/ping")
        self.assertEqual(ctx.exception.status_code, 429)

    def test_418_is_never_retried(self):
        client = _client(max_429_retries=5)
        calls = {"n": 0}

        def ban(request, timeout=None):
            _ = (request, timeout)
            calls["n"] += 1
            raise _http_error(418)

        with (
            patch.object(urllib.request, "urlopen", ban),
            self.assertRaises(BinanceFuturesAPIError) as ctx,
        ):
            client.public_get("/fapi/v1/ping")
        self.assertEqual(calls["n"], 1)
        self.assertEqual(ctx.exception.status_code, 418)
        self.assertIn("auto-ban", str(ctx.exception))


class TestServerTimeSync(unittest.TestCase):
    def test_signed_timestamp_applies_offset(self):
        client = _client(sync_server_time=True)
        with patch.object(
            BinanceFuturesRESTClient,
            "_request",
            return_value={"serverTime": time.time() * 1000.0 + 5_000.0},
        ):
            client._sync_time_offset()
        self.assertAlmostEqual(client._time_offset_ms, 5_000.0, delta=200.0)
        local_ms = time.time() * 1000.0
        self.assertGreater(client._signed_timestamp_ms(), local_ms + 4_000.0)

    def test_minus_1021_triggers_one_resync_retry(self):
        client = _client(sync_server_time=True)
        client._time_synced = True  # pretend already synced with a bad offset
        calls = {"n": 0}

        def flaky(request, timeout=None):
            _ = (request, timeout)
            calls["n"] += 1
            if calls["n"] == 1:
                raise _http_error(400, body=b'{"code": -1021, "msg": "timestamp"}')

            class _Resp:
                def __enter__(self):
                    return self

                def __exit__(self, *args):
                    return False

                @staticmethod
                def read():
                    if calls["n"] == 2:
                        return b'{"serverTime": 1700000000000}'
                    return b'{"ok": true}'

            return _Resp()

        with patch.object(urllib.request, "urlopen", flaky):
            result = client.signed_get("/fapi/v2/account")
        self.assertEqual(result, {"ok": True})
        # 1st: -1021 failure, 2nd: /fapi/v1/time resync, 3rd: successful retry.
        self.assertEqual(calls["n"], 3)


if __name__ == "__main__":
    unittest.main()
