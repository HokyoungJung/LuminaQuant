"""Phase 4.4: TickReplayValidator tests.

Uses the committed Phase 0 aggTrades fixture (BTCUSDT 1h ~48 730 trades).
Validates:
- MKT order fill prices are within the bar range
- LMT BUY fills when bar_low < limit_price (strict cross)
- LMT SELL fills when bar_high > limit_price (strict cross)
- LMT orders do NOT fill when the bar doesn't cross the limit
- Fill price for LMT orders equals limit_price exactly

SAFETY: No real exchange orders placed. Public market data only.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_FIXTURE_DIR = _REPO_ROOT / "tests" / "fixtures" / "aggtrades"


def _load_aggtrades():
    """Load the committed aggTrades fixture; skip if missing."""
    try:
        import polars as pl
    except ImportError:
        pytest.skip("polars not available")

    parquets = list(_FIXTURE_DIR.glob("*.parquet"))
    if not parquets:
        pytest.skip(f"No aggTrades fixture in {_FIXTURE_DIR}")
    return pl.read_parquet(parquets[0])


@pytest.fixture(scope="module")
def btcusdt_aggtrades():
    return _load_aggtrades()


@pytest.fixture(scope="module")
def validator(btcusdt_aggtrades):
    from lumina_quant.backtesting.tick_replay_validator import TickReplayValidator

    return TickReplayValidator(btcusdt_aggtrades)


@pytest.fixture(scope="module")
def verdict(validator):
    return validator.validate()


# ── Smoke tests ───────────────────────────────────────────────────────────────


def test_validator_produces_verdict(verdict):
    """validate() returns a TickReplayVerdict with bars from the fixture."""
    assert verdict.bars_count > 0, "Expected at least one 1s bar from fixture"
    assert verdict.lmt_verdict in {"PASS", "FAIL", "SKIP"}
    assert verdict.mkt_verdict in {"PASS", "FAIL", "SKIP"}


def test_bars_count_matches_fixture_duration(verdict):
    """Fixture spans ~1 hour → expect ~3 600 1s bars (±60 for boundary gaps)."""
    assert 3_400 <= verdict.bars_count <= 3_700, f"Expected ~3 600 bars, got {verdict.bars_count}"


# ── MKT verdict ───────────────────────────────────────────────────────────────


def test_mkt_verdict_pass(verdict):
    """MKT orders fill within the bar price range."""
    assert verdict.mkt_verdict == "PASS", f"MKT verdict FAIL — details: {verdict.mkt_cases}"


def test_mkt_fill_price_in_bar_range(verdict):
    """Every MKT case: fill_price is between bar_open and bar_high/low."""
    for case in verdict.mkt_cases:
        assert case.get("passed"), f"MKT case '{case['description']}' failed: {case}"
        assert case["executed_qty"] > 0, f"MKT case '{case['description']}': executed_qty == 0"


# ── LMT verdict ───────────────────────────────────────────────────────────────


def test_lmt_verdict_pass(verdict):
    """LMT orders pass all fill-rule assertions."""
    assert verdict.lmt_verdict == "PASS", "LMT verdict FAIL — details:\n" + "\n".join(
        f"  {c}" for c in verdict.lmt_cases if not c.get("passed")
    )


def test_lmt_buy_fills_when_bar_low_below_limit(btcusdt_aggtrades):
    """BUY LMT fills at least once when limit is inside the price range."""
    from lumina_quant.backtesting.tick_replay_validator import (
        LmtOrderCase,
        TickReplayValidator,
    )
    import polars as pl

    df = btcusdt_aggtrades
    # Aggregate to bars to get price range.
    v = TickReplayValidator(df)
    bars = v._aggregate_to_1s()
    if bars is None or bars.is_empty():
        bars = v._fallback_1s()
    lows = bars["low"].to_numpy()
    highs = bars["high"].to_numpy()

    # Set limit inside the price range so fills should occur.
    # Use median of (min_low, max_high) so it's genuinely in the middle.
    limit = float((lows.min() + highs.max()) / 2.0)
    limit = round(limit / 10.0) * 10.0  # round to $10 for clean price

    result = TickReplayValidator(
        df,
        lmt_cases=[
            LmtOrderCase(
                symbol="BTCUSDT",
                direction="BUY",
                qty=0.01,
                limit_price=limit,
                expect_fill=True,
                description="BUY LMT strict-cross positive case",
            )
        ],
    ).validate()

    assert result.lmt_verdict == "PASS", result.lmt_cases
    case_r = result.lmt_cases[0]
    assert case_r["filled_any"], (
        f"Expected at least one fill for BUY LMT at {limit}; "
        f"price range {lows.min():.1f}-{highs.max():.1f}"
    )


def test_lmt_buy_no_fill_below_market(btcusdt_aggtrades):
    """BUY LMT far below market → zero bars cross → no fill (strict-cross)."""
    from lumina_quant.backtesting.tick_replay_validator import (
        LmtOrderCase,
        TickReplayValidator,
    )

    df = btcusdt_aggtrades
    v_tmp = TickReplayValidator(df)
    bars = v_tmp._aggregate_to_1s()
    if bars is None or bars.is_empty():
        bars = v_tmp._fallback_1s()
    floor_price = float(bars["low"].to_numpy().min()) - 200.0  # well below market

    result = TickReplayValidator(
        df,
        lmt_cases=[
            LmtOrderCase(
                symbol="BTCUSDT",
                direction="BUY",
                qty=0.01,
                limit_price=floor_price,
                expect_fill=False,
                description="BUY LMT strict-cross negative case — below market",
            )
        ],
    ).validate()

    assert result.lmt_verdict == "PASS", result.lmt_cases
    case_r = result.lmt_cases[0]
    assert not case_r["filled_any"], (
        f"BUY LMT at {floor_price:.1f} should not fill; market floor {bars['low'].min():.1f}"
    )
    assert case_r["n_bars_should_fill"] == 0, (
        f"Expected 0 bars with bar_low < {floor_price:.1f}, got {case_r['n_bars_should_fill']}"
    )


def test_lmt_sell_fills_when_bar_high_above_limit(btcusdt_aggtrades):
    """SELL LMT fills at least once when limit is inside the price range."""
    from lumina_quant.backtesting.tick_replay_validator import (
        LmtOrderCase,
        TickReplayValidator,
    )

    df = btcusdt_aggtrades
    v_tmp = TickReplayValidator(df)
    bars = v_tmp._aggregate_to_1s()
    if bars is None or bars.is_empty():
        bars = v_tmp._fallback_1s()
    lows = bars["low"].to_numpy()
    highs = bars["high"].to_numpy()

    limit = float((lows.min() + highs.max()) / 2.0)
    limit = round(limit / 10.0) * 10.0

    result = TickReplayValidator(
        df,
        lmt_cases=[
            LmtOrderCase(
                symbol="BTCUSDT",
                direction="SELL",
                qty=0.01,
                limit_price=limit,
                expect_fill=True,
                description="SELL LMT strict-cross positive case",
            )
        ],
    ).validate()

    assert result.lmt_verdict == "PASS", result.lmt_cases
    assert result.lmt_cases[0]["filled_any"]


def test_lmt_fill_price_equals_limit_price(btcusdt_aggtrades):
    """LMT fill price must equal limit_price exactly (no slippage)."""
    from lumina_quant.backtesting.tick_replay_validator import (
        LmtOrderCase,
        TickReplayValidator,
    )

    df = btcusdt_aggtrades
    v_tmp = TickReplayValidator(df)
    bars = v_tmp._aggregate_to_1s()
    if bars is None or bars.is_empty():
        bars = v_tmp._fallback_1s()
    lows = bars["low"].to_numpy()
    highs = bars["high"].to_numpy()
    limit = float((lows.min() + highs.max()) / 2.0)
    limit = round(limit / 10.0) * 10.0

    result = TickReplayValidator(
        df,
        lmt_cases=[
            LmtOrderCase(
                symbol="BTCUSDT",
                direction="BUY",
                qty=0.01,
                limit_price=limit,
                expect_fill=True,
                description="LMT fill-price-exact test",
            )
        ],
    ).validate()

    case_r = result.lmt_cases[0]
    assert case_r["filled_any"], "Expected at least one fill for exact-price test"
    assert not case_r["wrong_price"], (
        f"LMT fill_price {case_r['fill_price_seen']!r} != limit {limit!r}"
    )
    if case_r["fill_price_seen"] is not None:
        assert abs(case_r["fill_price_seen"] - limit) < 1e-8, (
            f"Fill price {case_r['fill_price_seen']!r} should equal limit {limit!r}"
        )


def test_lmt_sell_no_fill_above_market(btcusdt_aggtrades):
    """SELL LMT far above market → no bars cross → no fill."""
    from lumina_quant.backtesting.tick_replay_validator import (
        LmtOrderCase,
        TickReplayValidator,
    )

    df = btcusdt_aggtrades
    v_tmp = TickReplayValidator(df)
    bars = v_tmp._aggregate_to_1s()
    if bars is None or bars.is_empty():
        bars = v_tmp._fallback_1s()
    ceiling_price = float(bars["high"].to_numpy().max()) + 200.0

    result = TickReplayValidator(
        df,
        lmt_cases=[
            LmtOrderCase(
                symbol="BTCUSDT",
                direction="SELL",
                qty=0.01,
                limit_price=ceiling_price,
                expect_fill=False,
                description="SELL LMT strict-cross negative case — above market",
            )
        ],
    ).validate()

    assert result.lmt_verdict == "PASS", result.lmt_cases
    assert not result.lmt_cases[0]["filled_any"]


# ── Verdict serialisation ─────────────────────────────────────────────────────


def test_verdict_to_dict_is_json_serialisable(verdict):
    """to_dict() produces a JSON-serialisable structure."""
    import json

    d = verdict.to_dict()
    json.dumps(d)  # should not raise
    assert "lmt_verdict" in d
    assert "mkt_verdict" in d
    assert "bars_count" in d


# ── No-fixture fallback ───────────────────────────────────────────────────────


def test_validator_graceful_on_empty_df():
    """Empty aggTrades DataFrame produces SKIP verdict without crashing."""
    try:
        import polars as pl
    except ImportError:
        pytest.skip("polars not available")

    from lumina_quant.backtesting.tick_replay_validator import TickReplayValidator

    empty_df = pl.DataFrame(
        schema={
            "agg_trade_id": pl.Int64,
            "price": pl.Float64,
            "quantity": pl.Float64,
            "timestamp_ms": pl.Int64,
            "is_buyer_maker": pl.Boolean,
            "first_trade_id": pl.Int64,
            "last_trade_id": pl.Int64,
        }
    )
    v = TickReplayValidator(empty_df)
    verdict_empty = v.validate()
    assert verdict_empty.lmt_verdict == "SKIP"
    assert verdict_empty.mkt_verdict == "SKIP"
