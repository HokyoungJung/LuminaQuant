"""V3 — negative Binance API response-shape tests.

Audit finding V3 (real_money_readiness_audit_20260706): ``_as_float`` coerces
every missing/malformed money-critical field to ``0.0`` with no recorded
fixtures and no negative-shape tests. ``executedQty -> 0`` makes a filled order
look unfilled; ``avgPrice -> 0`` poisons realized PnL and the consecutive-loss
classifier; ``get_balance -> 0`` feeds the C4 equity kill-switch a fake zero; a
Binance "error-in-200" body (HTTP 200 carrying ``{"code": -2011, ...}``) is
booked as a zero-filled order instead of being rejected loudly.

These tests do two things:

1. **Characterize** the exact silent-zero behaviour at tip so any future change
   is a deliberate, reviewed diff (regression anchors), and prove the drift is
   real and untested (the gap the audit flags).
2. **Pin the desired loud-rejection contract** as ``xfail`` — the money-critical
   ``_as_float`` "distinguish absent from 0 / reject in real mode" fix lives in
   ``exchanges/binance_futures_exchange.py`` (NOT owned by this test workstream)
   and has not landed; when it does, these xfails XPASS and must be promoted.

They also lock in the behaviour that IS already safe: garbage numeric strings on
the user-stream frame raise loudly instead of booking a phantom zero fill.
"""

from __future__ import annotations

import pytest

from lumina_quant.exchanges.binance_futures_exchange import BinanceFuturesExchange
from lumina_quant.live.binance_user_stream import BinanceUserStreamClient


def _bare_exchange() -> BinanceFuturesExchange:
    """A BinanceFuturesExchange whose pure normalizers are exercised without the
    network ``connect()``.  ``_normalize_order`` / ``_as_float`` / ``get_balance``
    touch only static helpers (+ ``rest_client`` for balance), never the socket."""
    return object.__new__(BinanceFuturesExchange)


# --------------------------------------------------------------------------- #
# _as_float coercion (the shared root cause)                                  #
# --------------------------------------------------------------------------- #
class TestAsFloatCoercion:
    @pytest.mark.parametrize("bad", [None, "", "abc", "NaNaN", object()])
    def test_missing_or_garbage_coerces_to_default(self, bad):
        # Current behaviour: any absent / non-numeric money value silently becomes
        # the default. This is exactly why executedQty/avgPrice/balance book zeros.
        assert BinanceFuturesExchange._as_float(bad, 7.0) == 7.0

    def test_absent_and_literal_zero_are_indistinguishable(self):
        # The V3 core defect: _as_float cannot tell "field absent" from "field is 0".
        assert BinanceFuturesExchange._as_float(None, 0.0) == BinanceFuturesExchange._as_float(
            "0", 0.0
        )


# --------------------------------------------------------------------------- #
# _normalize_order: missing / malformed money-critical fields                 #
# --------------------------------------------------------------------------- #
class TestNormalizeOrderShape:
    def test_missing_executed_qty_books_a_filled_order_as_unfilled(self):
        ex = _bare_exchange()
        # Exchange says FILLED but omits executedQty -> booked filled=0 (looks unfilled).
        order = ex._normalize_order(
            {"orderId": 5, "symbol": "BTCUSDT", "status": "FILLED", "origQty": "1.0"}
        )
        assert order["status"] == "closed"
        assert order["filled"] == 0.0  # <-- the silent poison V3 flags

    def test_garbage_avg_price_poisons_average_to_zero(self):
        ex = _bare_exchange()
        order = ex._normalize_order(
            {
                "orderId": 6,
                "symbol": "BTCUSDT",
                "status": "FILLED",
                "origQty": "1.0",
                "executedQty": "1.0",
                "avgPrice": "not-a-number",
                "price": "0",
            }
        )
        # avgPrice garbage -> 0, price 0 -> average falls to 0 (poisons realized PnL).
        assert order["average"] == 0.0

    def test_error_in_200_body_is_booked_as_zero_filled_order(self):
        ex = _bare_exchange()
        # A Binance error payload returned with HTTP 200 (the REST client only raises
        # on non-2xx) flows straight into _normalize_order and books zeros silently.
        order = ex._normalize_order({"code": -2011, "msg": "Unknown order sent."})
        assert order["id"] == ""
        assert order["filled"] == 0.0
        assert order["average"] == 0.0
        assert order["status"] == "unknown"


# --------------------------------------------------------------------------- #
# get_balance: missing asset row feeds the equity kill-switch a fake zero      #
# --------------------------------------------------------------------------- #
class TestGetBalanceShape:
    def test_missing_asset_returns_zero(self):
        ex = _bare_exchange()

        class _Client:
            @staticmethod
            def account_balance_v3():
                return [{"asset": "BNB", "availableBalance": "12.5"}]

        ex.rest_client = _Client()
        # USDT row absent -> 0.0 (this is the number that feeds C4 equity reconciliation).
        assert ex.get_balance("USDT") == 0.0

    def test_garbage_balance_field_returns_zero(self):
        ex = _bare_exchange()

        class _Client:
            @staticmethod
            def account_balance_v3():
                return [{"asset": "USDT", "availableBalance": "garbage", "balance": None}]

        ex.rest_client = _Client()
        assert ex.get_balance("USDT") == 0.0


# --------------------------------------------------------------------------- #
# user-stream parse_message: shape robustness                                 #
# --------------------------------------------------------------------------- #
class TestUserStreamParseMessage:
    def test_non_dict_payload_returns_none(self):
        assert BinanceUserStreamClient.parse_message("garbage") is None
        assert BinanceUserStreamClient.parse_message(None) is None

    def test_unknown_event_type_returns_none(self):
        assert BinanceUserStreamClient.parse_message({"e": "depthUpdate"}) is None

    def test_missing_order_object_books_zero_fields(self):
        # ORDER_TRADE_UPDATE with an absent/empty "o" -> every money field 0/empty,
        # but still surfaced as an executionReport (silent-zero, characterization).
        parsed = BinanceUserStreamClient.parse_message({"e": "ORDER_TRADE_UPDATE", "o": {}})
        assert parsed is not None
        assert parsed["event_type"] == "executionReport"
        assert parsed["last_fill_qty"] == 0.0
        assert parsed["cum_fill_qty"] == 0.0
        assert parsed["last_fill_price"] == 0.0
        assert parsed["order_id"] == ""

    @pytest.mark.parametrize("field", ["l", "z", "L"])
    def test_garbage_numeric_field_raises_loudly(self, field):
        # Already-safe behaviour: a non-numeric money field on the frame raises
        # (float("abc") -> ValueError) rather than silently booking a phantom 0 fill.
        payload = {"e": "ORDER_TRADE_UPDATE", "o": {"i": 1, "s": "BTCUSDT", field: "not-a-number"}}
        with pytest.raises(ValueError):
            BinanceUserStreamClient.parse_message(payload)

    def test_empty_string_numeric_field_coerces_to_zero(self):
        # Empty string is the one non-None garbage that still silently -> 0.0.
        payload = {"e": "ORDER_TRADE_UPDATE", "o": {"i": 1, "s": "BTCUSDT", "z": "", "l": ""}}
        parsed = BinanceUserStreamClient.parse_message(payload)
        assert parsed["cum_fill_qty"] == 0.0
        assert parsed["last_fill_qty"] == 0.0


# --------------------------------------------------------------------------- #
# Desired loud-rejection contract (V3 src fix not yet landed)                 #
# --------------------------------------------------------------------------- #
_V3_XFAIL_REASON = (
    "V3 fix not implemented: make _as_float distinguish absent-from-0 and reject "
    "money-critical shape drift loudly in real mode. Owned by "
    "exchanges/binance_futures_exchange.py (outside this test workstream). "
    "When landed, these XPASS and must be promoted to hard assertions."
)
# The exact loud-rejection type is a design choice for the (unlanded) src fix;
# accept any of the concrete error types a money-shape guard would plausibly use.
_LOUD_REJECTION = (ValueError, RuntimeError, KeyError, TypeError, AssertionError)


class TestDesiredLoudRejection:
    @pytest.mark.xfail(reason=_V3_XFAIL_REASON, strict=False)
    def test_error_in_200_body_should_reject_loudly(self):
        ex = _bare_exchange()
        with pytest.raises(_LOUD_REJECTION):
            ex._normalize_order({"code": -2011, "msg": "Unknown order sent."})

    @pytest.mark.xfail(reason=_V3_XFAIL_REASON, strict=False)
    def test_filled_status_with_absent_executed_qty_should_reject_loudly(self):
        ex = _bare_exchange()
        with pytest.raises(_LOUD_REJECTION):
            # FILLED with no executedQty is internally inconsistent: real mode must
            # not book it as filled=0 (a filled order silently disappearing).
            ex._normalize_order(
                {"orderId": 5, "symbol": "BTCUSDT", "status": "FILLED", "origQty": "1.0"}
            )

    @pytest.mark.xfail(reason=_V3_XFAIL_REASON, strict=False)
    def test_missing_balance_asset_should_reject_loudly(self):
        ex = _bare_exchange()

        class _Client:
            @staticmethod
            def account_balance_v3():
                return [{"asset": "BNB", "availableBalance": "12.5"}]

        ex.rest_client = _Client()
        with pytest.raises(_LOUD_REJECTION):
            # A missing USDT row must not feed the equity kill-switch a phantom 0.0.
            ex.get_balance("USDT")
