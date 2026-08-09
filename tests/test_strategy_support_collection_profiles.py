from __future__ import annotations

import urllib.error
from urllib.parse import urlparse
from unittest.mock import patch

from lumina_quant.data_collector import collect_strategy_support_data
from lumina_quant.data_sync import (
    _http_get_json,
    _fetch_open_interest_history,
    _fetch_liquidation_orders,
    _fetch_price_klines,
    sync_futures_feature_points,
)


def test_collect_strategy_support_data_strategy_used_profile_skips_mark_index():
    result = collect_strategy_support_data(
        db_path="data/market_parquet",
        exchange_id="binance",
        symbol_list=["BTC/USDT"],
        since="2025-01-01T00:00:00+00:00",
        execute=False,
        feature_profile="strategy-used",
    )

    assert result["feature_profile"] == "strategy_used"
    assert result["feature_groups"] == {
        "funding": True,
        "mark_index": False,
        "open_interest": True,
        "liquidations": True,
    }
    assert result["features"] == [
        "funding_rate",
        "funding_mark_price",
        "funding_fee_rate",
        "funding_fee_quote_per_unit",
        "open_interest",
        "liquidation_long_qty",
        "liquidation_short_qty",
        "liquidation_long_notional",
        "liquidation_short_notional",
    ]


def test_collect_strategy_support_data_flags_override_profile_defaults():
    result = collect_strategy_support_data(
        db_path="data/market_parquet",
        exchange_id="binance",
        symbol_list=["BTC/USDT"],
        since="2025-01-01T00:00:00+00:00",
        execute=False,
        feature_profile="strategy_used",
        include_mark_index=True,
        include_liquidations=False,
    )

    assert result["feature_groups"] == {
        "funding": True,
        "mark_index": True,
        "open_interest": True,
        "liquidations": False,
    }
    assert "mark_price" in result["features"]
    assert "index_price" in result["features"]
    assert "liquidation_long_qty" not in result["features"]


def test_sync_futures_feature_points_skips_disabled_fetchers_and_persists_enabled_fields():
    captured: dict[str, object] = {}

    def _funding_history(**_kwargs):
        return [{"fundingTime": 1_735_689_600_016, "fundingRate": "0.0001", "markPrice": "50000"}]

    def _price_klines(**_kwargs):
        raise AssertionError("mark/index fetcher should not be called when disabled")

    def _open_interest_history(**_kwargs):
        return [{"timestamp": 1_735_689_900_000, "sumOpenInterestValue": "123456"}]

    def _liquidations(**_kwargs):
        return [{"time": 1_735_690_200_000, "side": "SELL", "origQty": "2", "price": "51000"}]

    def _upsert(
        db_path,
        *,
        exchange,
        symbol,
        rows,
        source,
        backend,
    ):
        captured["db_path"] = db_path
        captured["exchange"] = exchange
        captured["symbol"] = symbol
        captured["rows"] = rows
        captured["source"] = source
        captured["backend"] = backend
        return len(list(rows))

    with (
        patch("lumina_quant.data_sync._fetch_funding_history", side_effect=_funding_history),
        patch("lumina_quant.data_sync._fetch_price_klines", side_effect=_price_klines),
        patch(
            "lumina_quant.data_sync._fetch_open_interest_history",
            side_effect=_open_interest_history,
        ),
        patch("lumina_quant.data_sync._fetch_liquidation_orders", side_effect=_liquidations),
        patch("lumina_quant.data_sync.upsert_futures_feature_points_rows", side_effect=_upsert),
    ):
        stats = sync_futures_feature_points(
            db_path="data/market_parquet",
            exchange_id="binance",
            symbol_list=["BTC/USDT"],
            since_ms=1_735_689_600_000,
            until_ms=1_735_690_300_000,
            include_mark_index=False,
        )

    assert len(stats) == 1
    assert stats[0].symbol == "BTC/USDT"
    assert captured["exchange"] == "binance"
    assert captured["symbol"] == "BTC/USDT"
    rows = list(captured["rows"])
    assert len(rows) == 3
    funding_row = rows[0]
    assert funding_row["timestamp_ms"] == 1_735_689_600_000
    assert funding_row["funding_rate"] == 0.0001
    assert funding_row["funding_fee_rate"] == 0.0001
    assert funding_row["funding_fee_quote_per_unit"] == 5.0
    assert "mark_price" not in funding_row
    assert "index_price" not in funding_row


def test_fetch_price_klines_uses_pair_param_for_index_endpoint():
    captured: dict[str, object] = {}

    def _http_get_json(url, *, params, retries, base_wait_sec):
        captured["url"] = url
        captured["params"] = dict(params)
        return []

    with patch("lumina_quant.data_sync._http_get_json", side_effect=_http_get_json):
        rows = _fetch_price_klines(
            symbol="XAU/USDT",
            price_type="index",
            interval="1m",
            since_ms=1_700_000_000_000,
            until_ms=1_700_000_060_000,
            retries=0,
            base_wait_sec=0.0,
        )

    assert rows == []
    assert str(captured["url"]).endswith("/indexPriceKlines")
    params = dict(captured["params"])
    assert params["pair"] == "XAUUSDT"
    assert "symbol" not in params


def test_fetch_liquidation_orders_returns_empty_on_unavailable_endpoint():
    def _http_get_json(url, *, params, retries, base_wait_sec):
        parsed = urlparse(url)
        _ = (params, retries, base_wait_sec)
        raise RuntimeError(f"HTTP 429 for {parsed.path}")

    with patch("lumina_quant.data_sync._http_get_json", side_effect=_http_get_json):
        rows = _fetch_liquidation_orders(
            symbol="BTC/USDT",
            since_ms=1_700_000_000_000,
            until_ms=1_700_000_060_000,
            retries=0,
            base_wait_sec=0.0,
        )

    assert rows == []


def test_fetch_open_interest_history_chunks_long_ranges():
    calls: list[dict[str, int | str]] = []

    def _http_get_json(url, *, params, retries, base_wait_sec):
        _ = (url, retries, base_wait_sec)
        calls.append(dict(params))
        if len(calls) == 1:
            return []
        if len(calls) == 2:
            return [
                {
                    "timestamp": int(params["startTime"]) + (5 * 60_000),
                    "sumOpenInterestValue": "100.0",
                }
            ]
        return []

    with patch("lumina_quant.data_sync._http_get_json", side_effect=_http_get_json):
        rows = _fetch_open_interest_history(
            symbol="XAU/USDT",
            period="5m",
            since_ms=1_700_000_000_000,
            until_ms=1_700_000_000_000 + (10 * 86_400_000),
            retries=0,
            base_wait_sec=0.0,
        )

    assert len(rows) == 1
    assert len(calls) >= 2
    first_call = calls[0]
    assert first_call["symbol"] == "XAUUSDT"
    assert first_call["period"] == "5m"
    assert int(first_call["endTime"]) == int(first_call["startTime"]) + (500 * 5 * 60_000) - 1
    second_call = calls[1]
    assert int(second_call["startTime"]) == int(first_call["endTime"]) + 1


def test_http_get_json_fails_fast_on_http_400():
    def _urlopen(_target, timeout):
        _ = timeout
        raise urllib.error.HTTPError(
            url="https://example.test",
            code=400,
            msg="Bad Request",
            hdrs=None,
            fp=None,
        )

    with patch("urllib.request.urlopen", side_effect=_urlopen):
        try:
            _http_get_json(
                "https://example.test",
                params={"symbol": "BTCUSDT"},
                retries=3,
                base_wait_sec=0.0,
            )
        except RuntimeError as exc:
            assert "HTTP 400" in str(exc)
        else:
            raise AssertionError("expected RuntimeError")
