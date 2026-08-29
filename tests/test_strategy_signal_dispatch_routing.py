"""Routing tests for the research signal dispatcher (v5 defect fix).

The silent generic-fallback substitution defect: candidates whose
``strategy_class`` had no bespoke handler were scored as one shared 64-bar
momentum proxy with no label — 84/111 classes (every research_only sleeve)
were proxy-scored and their recorded metrics measured the proxy, not the
lane.  These tests pin the fix:

* flag OFF  -> legacy fallback streams, now honestly labelled;
* flag ON   -> unmapped registered classes run as their REAL selves via the
  registry simulator (window-capable sleeves get MARKET_WINDOW feeds);
* unknown classes still fall back (labelled) instead of raising;
* every strategy_class the candidate library emits has a non-fallback route.
"""

from __future__ import annotations

import datetime as dtm
from types import SimpleNamespace

import numpy as np
import pytest

from lumina_quant.strategy_factory import research_runner as rr
from lumina_quant.strategy_factory.candidate_library import build_candidate_manifest
from lumina_quant.strategy_factory.strategy_signal_dispatch import (
    StrategySignalDispatchError,
    StrategySignalDispatcher,
)

_SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT", "ADA/USDT"]
_N = 500
_ROUTE_ON = {"research": {"route_unmapped_registered_strategies": True}}
_STRICT_ROUTE_ON = {"research": {"require_actual_engine_routing": True}}
_NEAR_HIGH_PARAMS = {
    "high_lookback_bars": 200,
    "min_history_bars": 60,
    "rebalance_bars": 42,
    "min_hold_bars": 42,
    "quantile_pct": 0.25,
}


def _lcg_stream(seed: int):
    state = seed & 0x7FFFFFFF
    while True:
        state = (1103515245 * state + 12345) % (1 << 31)
        yield state / float(1 << 31)


def _aligned_panel() -> dict[str, np.ndarray]:
    start = dtm.datetime(2025, 1, 1, tzinfo=dtm.UTC)
    aligned: dict[str, np.ndarray] = {
        "datetime": np.asarray(
            [start + dtm.timedelta(hours=4 * i) for i in range(_N)], dtype=object
        )
    }
    for s_idx, symbol in enumerate(_SYMBOLS):
        rand = _lcg_stream(97 + s_idx)
        closes = [100.0]
        for _ in range(_N - 1):
            closes.append(closes[-1] * (1.0 + (next(rand) - 0.5) * 0.02 + 0.0004))
        close = np.asarray(closes, dtype=float)
        aligned[f"{symbol}:close"] = close
        aligned[f"{symbol}:open"] = close
        aligned[f"{symbol}:high"] = close * 1.004
        aligned[f"{symbol}:low"] = close * 0.996
        aligned[f"{symbol}:volume"] = np.full(_N, 1_000_000.0)
    return aligned


def _crowding_panel(*, warmup_bars: int = 0) -> dict[str, np.ndarray]:
    aligned = _aligned_panel()
    warmup = slice(0, warmup_bars)
    for s_idx, symbol in enumerate(_SYMBOLS):
        offset = float(s_idx + 1)
        aligned[f"{symbol}:funding_rate"] = np.linspace(-0.001, 0.001, _N) * offset
        aligned[f"{symbol}:open_interest"] = np.linspace(1_000.0, 2_000.0, _N) * offset
        aligned[f"{symbol}:liquidation_long_notional"] = np.linspace(10.0, 20.0, _N) * offset
        aligned[f"{symbol}:liquidation_short_notional"] = np.linspace(20.0, 10.0, _N) * offset
        aligned[f"{symbol}:mark_price"] = aligned[f"{symbol}:close"] * 1.0001
        aligned[f"{symbol}:index_price"] = aligned[f"{symbol}:close"] * 0.9999
        for field in (
            "funding_rate",
            "open_interest",
            "liquidation_long_notional",
            "liquidation_short_notional",
            "mark_price",
            "index_price",
        ):
            aligned[f"{symbol}:{field}"][warmup] = np.nan
    return aligned


def _flow_panel() -> dict[str, np.ndarray]:
    aligned = _aligned_panel()
    for s_idx, symbol in enumerate(_SYMBOLS):
        offset = float(s_idx + 1)
        aligned[f"{symbol}:taker_buy_quote_volume"] = np.full(_N, 1_100.0 * offset)
        aligned[f"{symbol}:taker_sell_quote_volume"] = np.full(_N, 900.0 * offset)
        aligned[f"{symbol}:book_depth_imbalance_1pct"] = np.full(_N, 0.2)
        aligned[f"{symbol}:bbo_spread_bps"] = np.full(_N, 2.0)
        aligned[f"{symbol}:liquidation_long_notional"] = np.linspace(10.0, 20.0, _N) * offset
        aligned[f"{symbol}:liquidation_short_notional"] = np.linspace(20.0, 10.0, _N) * offset
    return aligned


def _assert_actual_handler(result) -> None:
    assert result[3]["evaluation_mode"] == "handler"
    assert result[3]["generic_fallback_proxy_count"] == 0


def _signal(klass: str, params: dict, scoring=None):
    candidate = {"strategy_class": klass, "params": params}
    return rr._strategy_signal(
        candidate, aligned=_aligned_panel(), symbols=_SYMBOLS, scoring_config=scoring
    )


@pytest.mark.parametrize(
    ("klass", "field", "missing"),
    (
        *(
            (klass, field, missing)
            for klass in (
                "PerpCrowdingCarryStrategy",
                "CarryTrendFactorRotationStrategy",
                "FundingLiquidationCrowdingFadeStrategy",
                "FundingDislocationTrendCarryStrategy",
            )
            for field in ("mark_price", "index_price")
            for missing in (False, True)
        ),
        *(
            ("BasisSnapbackReversionStrategy", field, missing)
            for field in ("mark_price", "index_price")
            for missing in (False, True)
        ),
    ),
)
def test_strict_crowding_price_support_rejects_missing_or_all_null_without_fallback(
    klass: str, field: str, missing: bool
) -> None:
    aligned = _crowding_panel()
    key = f"BTC/USDT:{field}"
    if missing:
        aligned.pop(key)
    else:
        aligned[key][:] = np.nan

    with pytest.raises(StrategySignalDispatchError, match="missing required support data"):
        rr._strategy_signal(
            {"strategy_class": klass, "params": {}},
            aligned=aligned,
            symbols=_SYMBOLS,
            scoring_config=_STRICT_ROUTE_ON,
        )


@pytest.mark.parametrize(
    "klass",
    (
        "PerpCrowdingCarryStrategy",
        "CarryTrendFactorRotationStrategy",
        "FundingLiquidationCrowdingFadeStrategy",
        "FundingDislocationTrendCarryStrategy",
        "BasisSnapbackReversionStrategy",
    ),
)
def test_strict_crowding_price_support_accepts_warmup_nans_with_finite_support(
    klass: str,
) -> None:
    result = rr._strategy_signal(
        {"strategy_class": klass, "params": {}},
        aligned=_crowding_panel(warmup_bars=24),
        symbols=_SYMBOLS,
        scoring_config=_STRICT_ROUTE_ON,
    )
    _assert_actual_handler(result)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("open_interest", -1.0),
        ("liquidation_long_notional", -1.0),
        ("liquidation_short_notional", -1.0),
        ("mark_price", 0.0),
        ("index_price", -1.0),
        ("funding_rate", np.inf),
    ),
)
def test_strict_crowding_rejects_invalid_support_domain_without_fallback(
    field: str, value: float
) -> None:
    aligned = _crowding_panel()
    aligned[f"BTC/USDT:{field}"][0] = value

    with pytest.raises(StrategySignalDispatchError, match="missing required support data"):
        rr._strategy_signal(
            {"strategy_class": "PerpCrowdingCarryStrategy", "params": {}},
            aligned=aligned,
            symbols=_SYMBOLS,
            scoring_config=_STRICT_ROUTE_ON,
        )


@pytest.mark.parametrize(
    "kind",
    ("missing", "all_null", "infinity", "negative", "disjoint", "post_warmup_hole"),
)
def test_strict_cross_asset_liquidation_rejects_malformed_paired_support_without_fallback(
    kind: str,
) -> None:
    aligned = _crowding_panel()
    long_key = "BTC/USDT:liquidation_long_notional"
    short_key = "BTC/USDT:liquidation_short_notional"
    if kind == "missing":
        aligned.pop(long_key)
    elif kind == "all_null":
        aligned[long_key][:] = np.nan
    elif kind == "infinity":
        aligned[long_key][24] = np.inf
    elif kind == "negative":
        aligned[short_key][24] = -1.0
    elif kind == "disjoint":
        aligned[long_key] = np.r_[np.ones(_N // 2), np.full(_N - (_N // 2), np.nan)]
        aligned[short_key] = np.r_[np.full(_N // 2, np.nan), np.ones(_N - (_N // 2))]
    else:
        aligned[long_key][:24] = np.nan
        aligned[short_key][:24] = np.nan
        aligned[long_key][48] = np.nan

    with pytest.raises(StrategySignalDispatchError, match="missing required support data") as error:
        rr._strategy_signal(
            {"strategy_class": "CrossAssetLiquidationContagionFadeStrategy", "params": {}},
            aligned=aligned,
            symbols=_SYMBOLS,
            scoring_config=_STRICT_ROUTE_ON,
        )

    assert error.value.__cause__ is None


def test_strict_cross_asset_liquidation_accepts_both_leg_nan_prefix_warmup() -> None:
    result = rr._strategy_signal(
        {"strategy_class": "CrossAssetLiquidationContagionFadeStrategy", "params": {}},
        aligned=_crowding_panel(warmup_bars=24),
        symbols=_SYMBOLS,
        scoring_config=_STRICT_ROUTE_ON,
    )

    _assert_actual_handler(result)


def test_cross_asset_liquidation_missing_support_remains_non_strict() -> None:
    aligned = _crowding_panel()
    aligned.pop("BTC/USDT:liquidation_long_notional")

    result = rr._strategy_signal(
        {"strategy_class": "CrossAssetLiquidationContagionFadeStrategy", "params": {}},
        aligned=aligned,
        symbols=_SYMBOLS,
    )

    assert result[3]["evaluation_mode"] == "handler"
    assert "generic_fallback_proxy" not in result[3]
    assert "generic_fallback_proxy_count" not in result[3]
    assert np.all(result[2][0] == 0.0)
    assert result[3]["missing_support_symbols"] == ["BTC/USDT"]


@pytest.mark.parametrize(
    "field",
    (
        "taker_buy_quote_volume",
        "taker_sell_quote_volume",
        "book_depth_imbalance_1pct",
        "bbo_spread_bps",
    ),
)
def test_strict_flow_rejects_all_null_source_route_without_fallback(field: str) -> None:
    aligned = _flow_panel()
    aligned[f"BTC/USDT:{field}"][:] = np.nan

    with pytest.raises(StrategySignalDispatchError, match="missing required support data"):
        rr._strategy_signal(
            {"strategy_class": "FlowImbalanceLiquidationSweepStrategy", "params": {}},
            aligned=aligned,
            symbols=_SYMBOLS,
            scoring_config=_STRICT_ROUTE_ON,
        )


def test_strict_flow_rejects_missing_spread_route_without_fallback() -> None:
    aligned = _flow_panel()
    for symbol in _SYMBOLS:
        aligned.pop(f"{symbol}:bbo_spread_bps")

    with pytest.raises(StrategySignalDispatchError, match="missing required support data"):
        rr._strategy_signal(
            {"strategy_class": "FlowImbalanceLiquidationSweepStrategy", "params": {}},
            aligned=aligned,
            symbols=_SYMBOLS,
            scoring_config=_STRICT_ROUTE_ON,
        )


@pytest.mark.parametrize(
    "field",
    ("liquidation_long_notional", "liquidation_short_notional"),
)
def test_strict_flow_rejects_all_null_liquidation_support_without_fallback(field: str) -> None:
    aligned = _flow_panel()
    aligned[f"BTC/USDT:{field}"][:] = np.nan

    with pytest.raises(StrategySignalDispatchError, match="missing required support data"):
        rr._strategy_signal(
            {"strategy_class": "FlowImbalanceLiquidationSweepStrategy", "params": {}},
            aligned=aligned,
            symbols=_SYMBOLS,
            scoring_config=_STRICT_ROUTE_ON,
        )


def test_strict_flow_executes_with_finite_support_without_fallback() -> None:
    result = rr._strategy_signal(
        {"strategy_class": "FlowImbalanceLiquidationSweepStrategy", "params": {}},
        aligned=_flow_panel(),
        symbols=_SYMBOLS,
        scoring_config=_STRICT_ROUTE_ON,
    )

    _assert_actual_handler(result)


@pytest.mark.parametrize("value", (-1.0001, 1.0001, np.inf))
def test_strict_flow_rejects_invalid_direct_depth_support_without_fallback(value: float) -> None:
    aligned = _flow_panel()
    aligned["BTC/USDT:book_depth_imbalance_1pct"][0] = value

    with pytest.raises(StrategySignalDispatchError, match="missing required support data"):
        rr._strategy_signal(
            {"strategy_class": "FlowImbalanceLiquidationSweepStrategy", "params": {}},
            aligned=aligned,
            symbols=_SYMBOLS,
            scoring_config=_STRICT_ROUTE_ON,
        )


@pytest.mark.parametrize("boundary", (-1.0, 1.0))
def test_strict_flow_accepts_direct_depth_boundaries_without_fallback(boundary: float) -> None:
    aligned = _flow_panel()
    aligned["BTC/USDT:book_depth_imbalance_1pct"] = np.r_[
        np.full(24, np.nan), np.full(_N - 24, boundary)
    ]

    result = rr._strategy_signal(
        {"strategy_class": "FlowImbalanceLiquidationSweepStrategy", "params": {}},
        aligned=aligned,
        symbols=_SYMBOLS,
        scoring_config=_STRICT_ROUTE_ON,
    )

    _assert_actual_handler(result)


def test_strict_flow_accepts_finite_bid_ask_quantity_depth_alternate_without_fallback() -> None:
    aligned = _flow_panel()
    for symbol in _SYMBOLS:
        aligned.pop(f"{symbol}:book_depth_imbalance_1pct")
        aligned[f"{symbol}:best_bid_quantity"] = np.full(_N, 120.0)
        aligned[f"{symbol}:best_ask_quantity"] = np.full(_N, 80.0)

    result = rr._strategy_signal(
        {"strategy_class": "FlowImbalanceLiquidationSweepStrategy", "params": {}},
        aligned=aligned,
        symbols=_SYMBOLS,
        scoring_config=_STRICT_ROUTE_ON,
    )

    _assert_actual_handler(result)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("liquidation_long_notional", -1.0),
        ("liquidation_short_notional", np.inf),
    ),
)
def test_strict_flow_rejects_invalid_liquidation_support_without_fallback(
    field: str, value: float
) -> None:
    aligned = _flow_panel()
    aligned[f"BTC/USDT:{field}"][0] = value
    with pytest.raises(StrategySignalDispatchError, match="missing required support data"):
        rr._strategy_signal(
            {"strategy_class": "FlowImbalanceLiquidationSweepStrategy", "params": {}},
            aligned=aligned,
            symbols=_SYMBOLS,
            scoring_config=_STRICT_ROUTE_ON,
        )


@pytest.mark.parametrize(
    "field_pair",
    (
        ("taker_buy_quote_volume", "taker_sell_quote_volume"),
        ("best_bid_quantity", "best_ask_quantity"),
    ),
)
def test_strict_flow_rejects_disjoint_taker_or_quantity_pairs_without_fallback(
    field_pair: tuple[str, str],
) -> None:
    aligned = _flow_panel()
    first, second = field_pair
    if first == "best_bid_quantity":
        aligned["BTC/USDT:book_depth_imbalance_1pct"][:] = np.nan
    aligned[f"BTC/USDT:{first}"] = np.r_[np.ones(_N // 2), np.full(_N - (_N // 2), np.nan)]
    aligned[f"BTC/USDT:{second}"] = np.r_[np.full(_N // 2, np.nan), np.ones(_N - (_N // 2))]
    with pytest.raises(StrategySignalDispatchError, match="missing required support data"):
        rr._strategy_signal(
            {"strategy_class": "FlowImbalanceLiquidationSweepStrategy", "params": {}},
            aligned=aligned,
            symbols=_SYMBOLS,
            scoring_config=_STRICT_ROUTE_ON,
        )


@pytest.mark.parametrize("kind", ("negative_direct", "disjoint", "crossed", "nonpositive"))
def test_strict_flow_rejects_invalid_bbo_routes_without_fallback(kind: str) -> None:
    aligned = _flow_panel()
    if kind == "negative_direct":
        aligned["BTC/USDT:bbo_spread_bps"][:] = -1.0
    else:
        aligned.pop("BTC/USDT:bbo_spread_bps")
        if kind == "disjoint":
            aligned["BTC/USDT:best_bid_price"] = np.r_[
                np.full(_N // 2, 100.0), np.full(_N - (_N // 2), np.nan)
            ]
            aligned["BTC/USDT:best_ask_price"] = np.r_[
                np.full(_N // 2, np.nan), np.full(_N - (_N // 2), 101.0)
            ]
        elif kind == "crossed":
            aligned["BTC/USDT:best_bid_price"] = np.full(_N, 101.0)
            aligned["BTC/USDT:best_ask_price"] = np.full(_N, 100.0)
        else:
            aligned["BTC/USDT:best_bid_price"] = np.zeros(_N)
            aligned["BTC/USDT:best_ask_price"] = np.full(_N, 100.0)
    with pytest.raises(StrategySignalDispatchError, match="missing required support data"):
        rr._strategy_signal(
            {"strategy_class": "FlowImbalanceLiquidationSweepStrategy", "params": {}},
            aligned=aligned,
            symbols=_SYMBOLS,
            scoring_config=_STRICT_ROUTE_ON,
        )


def test_strict_flow_executes_with_finite_alternate_bbo_without_fallback() -> None:
    aligned = _flow_panel()
    for symbol in _SYMBOLS:
        aligned.pop(f"{symbol}:bbo_spread_bps")
        aligned[f"{symbol}:best_bid_price"] = np.full(_N, 100.0)
        aligned[f"{symbol}:best_ask_price"] = np.full(_N, 100.02)
    result = rr._strategy_signal(
        {"strategy_class": "FlowImbalanceLiquidationSweepStrategy", "params": {}},
        aligned=aligned,
        symbols=_SYMBOLS,
        scoring_config=_STRICT_ROUTE_ON,
    )
    _assert_actual_handler(result)


def test_flag_off_unmapped_classes_fall_back_identically_with_label() -> None:
    r_near = _signal("CrossSectionalNearHighAnchoringStrategy", dict(_NEAR_HIGH_PARAMS))
    r_reb = _signal("RebalancingPremiumHarvestStrategy", {})
    assert r_near[3].get("evaluation_mode") == "generic_fallback_proxy"
    assert r_reb[3].get("evaluation_mode") == "generic_fallback_proxy"
    # The defect signature: two different lanes, byte-identical proxy streams.
    assert np.array_equal(r_near[0], r_reb[0])
    assert np.array_equal(r_near[1], r_reb[1])


def test_flag_on_routes_unmapped_class_through_real_strategy() -> None:
    r_off = _signal("CrossSectionalNearHighAnchoringStrategy", dict(_NEAR_HIGH_PARAMS))
    r_on = _signal("CrossSectionalNearHighAnchoringStrategy", dict(_NEAR_HIGH_PARAMS), _ROUTE_ON)
    assert r_on[3].get("evaluation_mode") == "registry_simulator"
    assert r_on[3].get("event_driven_proxy") is True
    assert not np.array_equal(r_on[0], r_off[0])
    assert int(np.count_nonzero(r_on[2])) > 0


def test_flag_on_real_route_is_params_sensitive() -> None:
    base = _signal("CrossSectionalNearHighAnchoringStrategy", dict(_NEAR_HIGH_PARAMS), _ROUTE_ON)
    wide = _signal(
        "CrossSectionalNearHighAnchoringStrategy",
        dict(_NEAR_HIGH_PARAMS, quantile_pct=0.5),
        _ROUTE_ON,
    )
    assert not np.array_equal(base[2], wide[2])


def test_flag_on_unknown_class_falls_back_with_label() -> None:
    result = _signal("DoesNotExistStrategy", {}, _ROUTE_ON)
    assert result[3].get("evaluation_mode") == "generic_fallback_proxy"


def test_strict_flag_routes_registered_class_and_rejects_unknown_class() -> None:
    result = _signal(
        "CrossSectionalNearHighAnchoringStrategy",
        dict(_NEAR_HIGH_PARAMS),
        _STRICT_ROUTE_ON,
    )
    assert result[3].get("evaluation_mode") == "registry_simulator"
    with pytest.raises(StrategySignalDispatchError):
        _signal("DoesNotExistStrategy", {}, _STRICT_ROUTE_ON)


def test_mapped_handler_path_labelled_handler() -> None:
    result = _signal("VolCompressionVWAPReversionStrategy", {}, _ROUTE_ON)
    assert result[3].get("evaluation_mode") == "handler"


def test_first_bar_has_no_wraparound_position_or_turnover() -> None:
    # The old np.roll wrapped the LAST bar's exposure into bar 0; the fallback
    # proxy holds nonzero exposure at the end of this panel, so a wrap would
    # charge bar-0 return/turnover.  Bar 0 must be positionless now.
    ret, turnover, _exposure, meta = _signal("DoesNotExistStrategy", {})
    assert meta.get("evaluation_mode") == "generic_fallback_proxy"
    assert ret[0] == 0.0
    assert turnover[0] == 0.0


def test_every_candidate_library_class_has_a_non_fallback_route() -> None:
    from lumina_quant.strategies.registry import resolve_strategy_class

    manifest = build_candidate_manifest(
        symbols=(*_SYMBOLS, "DOGE/USDT"), timeframes=("1h", "4h", "1d")
    )
    classes = sorted({str(c["strategy_class"]) for c in manifest["candidates"]})
    unrouted: list[str] = []
    for klass in classes:
        if klass in rr._STRATEGY_SIGNAL_DISPATCHER.handlers:
            continue
        try:
            resolve_strategy_class(klass)
        except Exception:
            unrouted.append(klass)
    assert not unrouted, f"classes with no non-fallback route: {unrouted}"


def test_leadlag_spillover_exposure_is_prefix_stable_no_lookahead() -> None:
    """LeadLagSpillover positions must not depend on FUTURE data (v5 fix).

    The old handler normalised every bar's score by the FULL-SAMPLE follower
    sigma and ``np.roll``-wrapped the tail of the leader-return series into
    the window head, so truncating the panel changed PAST positions.  With the
    expanding sigma and zero-fill shift, the exposure stream over any shared
    prefix is identical.
    """
    candidate = {"strategy_class": "LeadLagSpilloverStrategy", "params": {}}
    aligned_full = _aligned_panel()
    full = rr._strategy_signal(candidate, aligned=aligned_full, symbols=_SYMBOLS)

    cut = 300
    aligned_cut = {key: value[:cut] for key, value in aligned_full.items()}
    trunc = rr._strategy_signal(candidate, aligned=aligned_cut, symbols=_SYMBOLS)

    assert full[3].get("evaluation_mode") == "handler"
    assert int(np.count_nonzero(full[2])) > 0, "fixture must produce trades"
    assert np.allclose(full[2][:cut], trunc[2])


def test_leadlag_spillover_flat_through_sigma_warmup() -> None:
    # Expanding sigma returns 0.0 before ``min_periods`` observations, so the
    # laggard book must be flat through warmup instead of scored against a
    # whole-history sigma from bar 0.
    candidate = {"strategy_class": "LeadLagSpilloverStrategy", "params": {}}
    result = rr._strategy_signal(candidate, aligned=_aligned_panel(), symbols=_SYMBOLS)
    assert np.all(result[2][:31] == 0.0)


def test_strict_pair_handler_propagates_simulator_failure_through_research_dispatch(
    monkeypatch,
) -> None:
    def _failure(*args, **kwargs):
        raise RuntimeError("pair simulator failure")

    monkeypatch.setattr(rr, "_simulate_event_driven_strategy_exposures", _failure)
    params = {"symbol_x": "BTC/USDT", "symbol_y": "ETH/USDT"}

    legacy = _signal("PairSpreadZScoreStrategy", params)
    assert legacy[3]["event_driven_proxy"] is False
    assert legacy[3]["event_driven_proxy_error"] == "pair simulator failure"

    with pytest.raises(StrategySignalDispatchError) as error:
        _signal("PairSpreadZScoreStrategy", params, _STRICT_ROUTE_ON)
    assert isinstance(error.value.__cause__, RuntimeError)
    assert str(error.value.__cause__) == "pair simulator failure"


@pytest.mark.parametrize("klass", ("PairSpreadZScoreStrategy", "LagConvergenceStrategy"))
@pytest.mark.parametrize(
    "params",
    (
        {"symbol_x": "DOGE/USDT", "symbol_y": "ETH/USDT"},
        {"symbol_x": "BTC/USDT", "symbol_y": "DOGE/USDT"},
        {"symbol_x": "ETH/USDT", "symbol_y": "ETH/USDT"},
        {"symbol_x": "", "symbol_y": "ETH/USDT"},
        {"symbol_x": "BTC/USDT", "symbol_y": ""},
        {"symbol_x": None, "symbol_y": "ETH/USDT"},
        {"symbol_x": "BTC/USDT", "symbol_y": None},
    ),
)
def test_strict_pair_handlers_reject_explicit_pair_substitutions(
    klass: str,
    params: dict[str, str | None],
) -> None:
    with pytest.raises(StrategySignalDispatchError) as error:
        _signal(klass, params, _STRICT_ROUTE_ON)

    assert isinstance(error.value.__cause__, ValueError)


@pytest.mark.parametrize("klass", ("PairSpreadZScoreStrategy", "LagConvergenceStrategy"))
@pytest.mark.parametrize("raw_symbol", ("eth/usdt", " ETH/USDT", "ETHUSDT"))
def test_strict_pair_handlers_reject_canonicalizing_raw_pair_tokens_before_pair_logic(
    klass: str,
    raw_symbol: str,
    monkeypatch,
) -> None:
    pair_logic_calls: list[str] = []

    def _unexpected_pair_logic(*args, **kwargs):
        pair_logic_calls.append(klass)
        raise AssertionError("strict pair token validation must precede pair logic")

    monkeypatch.setattr(rr, "_simulate_event_driven_strategy_exposures", _unexpected_pair_logic)
    monkeypatch.setattr(rr, "_lag_convergence_pair_positions", _unexpected_pair_logic)
    monkeypatch.setattr(rr, "_pair_spread_fallback_exposures", _unexpected_pair_logic)

    with pytest.raises(StrategySignalDispatchError) as error:
        _signal(
            klass,
            {"symbol_x": raw_symbol, "symbol_y": "SOL/USDT"},
            _STRICT_ROUTE_ON,
        )

    assert isinstance(error.value.__cause__, ValueError)
    assert pair_logic_calls == []


@pytest.mark.parametrize("klass", ("PairSpreadZScoreStrategy", "LagConvergenceStrategy"))
def test_strict_pair_handlers_accept_exact_distinct_panel_symbols_without_fallback(
    klass: str,
    monkeypatch,
) -> None:
    observed_symbols: list[tuple[str, str]] = []

    def _simulate(*args, **kwargs):
        observed_symbols.append(kwargs["symbols"])
        return np.zeros((2, _N), dtype=float)

    monkeypatch.setattr(rr, "_simulate_event_driven_strategy_exposures", _simulate)
    params = {"symbol_x": "ETH/USDT", "symbol_y": "SOL/USDT"}

    result = _signal(klass, params, _STRICT_ROUTE_ON)

    _assert_actual_handler(result)
    if klass == "PairSpreadZScoreStrategy":
        assert observed_symbols == [("ETH/USDT", "SOL/USDT")]
    else:
        reads: list[str] = []

        class TrackingPanel(dict[str, np.ndarray]):
            def __getitem__(self, key: str) -> np.ndarray:
                reads.append(key)
                return super().__getitem__(key)

        exposures = np.zeros((len(_SYMBOLS), _N), dtype=float)
        rr._apply_lag_convergence_strategy(
            params=params,
            aligned=TrackingPanel(_aligned_panel()),
            symbols=_SYMBOLS,
            n=_N,
            exposures=exposures,
            meta={"_strict_actual_engine": True},
        )
        assert set(reads) == {"ETH/USDT:close", "SOL/USDT:close"}
        assert np.all(exposures[[0, 3, 4, 5]] == 0.0)


def test_public_research_strict_pair_failure_reaches_dispatcher(monkeypatch) -> None:
    def _failure(*args, **kwargs):
        raise RuntimeError("pair simulator failure")

    aligned = _aligned_panel()
    aligned["datetime"] = np.datetime64("2025-01-01T00:00:00.000", "ms") + (
        np.arange(_N) * np.timedelta64(60_000, "ms")
    )
    symbols = ["BTC/USDT", "ETH/USDT"]
    cache = {
        (symbol, "1m"): rr.SeriesBundle(
            symbol=symbol,
            timeframe="1m",
            datetime=aligned["datetime"],
            open=aligned[f"{symbol}:open"],
            high=aligned[f"{symbol}:high"],
            low=aligned[f"{symbol}:low"],
            close=aligned[f"{symbol}:close"],
            volume=aligned[f"{symbol}:volume"],
        )
        for symbol in symbols
    }
    monkeypatch.setattr(rr, "_simulate_event_driven_strategy_exposures", _failure)
    monkeypatch.setattr(
        rr,
        "_load_research_run_resources",
        lambda **kwargs: (cache, {}, {}, {}),
    )

    with pytest.raises(StrategySignalDispatchError) as error:
        rr.run_candidate_research(
            candidates=[
                {
                    "strategy_class": "PairSpreadZScoreStrategy",
                    "symbols": symbols,
                    "strategy_timeframe": "1m",
                    "params": {"symbol_x": symbols[0], "symbol_y": symbols[1]},
                }
            ],
            base_timeframe="1m",
            score_config=_STRICT_ROUTE_ON,
            data_mode="strict",
            allow_csv_fallback=False,
            allow_synthetic_fallback=False,
            min_bundle_bars=2,
        )
    assert isinstance(error.value.__cause__, RuntimeError)
    assert str(error.value.__cause__) == "pair simulator failure"


def test_strict_registry_dispatch_uses_numpy_datetime64_ms_cadence(monkeypatch) -> None:
    observed_window_seconds: list[int] = []

    class _RegisteredWindowStrategy:
        def __init__(self, bars, events, **params):
            self.events = events

        def calculate_signals_window(self, event):
            return None

        def calculate_signals(self, event):
            observed_window_seconds.append(event.window_seconds)

    monkeypatch.setattr(
        "lumina_quant.strategies.registry.resolve_strategy_class",
        lambda strategy_class: _RegisteredWindowStrategy,
    )
    aligned = _aligned_panel()
    start = np.datetime64("2025-01-01T00:00:00.000", "ms")
    aligned["datetime"] = start + np.arange(_N) * np.timedelta64(300_000, "ms")

    result = rr._strategy_signal(
        {"strategy_class": "RegisteredWindowStrategy", "params": {}},
        aligned=aligned,
        symbols=_SYMBOLS,
        scoring_config=_STRICT_ROUTE_ON,
    )

    assert result[3]["evaluation_mode"] == "registry_simulator"
    assert observed_window_seconds == [300] * _N


def test_strict_dispatch_rejects_sparse_utc_grid_before_mapped_handler() -> None:
    calls: list[str] = []

    def handler(params, aligned, symbols, n, exposures, meta) -> None:
        calls.append("handler")
        exposures[:] = 0.25

    dispatcher = StrategySignalDispatcher({"MappedStrategy": handler})
    aligned = _aligned_panel()
    aligned["datetime"][200:] += dtm.timedelta(hours=4)

    with pytest.raises(
        StrategySignalDispatchError,
        match="datetime grid is not positive, regular whole seconds",
    ):
        dispatcher.dispatch(
            {
                "strategy_class": "MappedStrategy",
                "strategy_timeframe": "4h",
                "params": {},
            },
            aligned=aligned,
            symbols=_SYMBOLS,
            require_actual_engine=True,
        )
    assert calls == []

    valid_result = dispatcher.dispatch(
        {
            "strategy_class": "MappedStrategy",
            "strategy_timeframe": "4h",
            "params": {},
        },
        aligned=_aligned_panel(),
        symbols=_SYMBOLS,
        require_actual_engine=True,
    )
    assert calls == ["handler"]
    assert valid_result[3]["evaluation_mode"] == "handler"


def test_strict_dispatch_rejects_numpy_cadence_mismatch_before_registry_router() -> None:
    calls: list[str] = []

    def router(strategy_class, params, aligned, symbols) -> np.ndarray:
        calls.append("router")
        return np.zeros((len(symbols), len(aligned["datetime"])), dtype=float)

    dispatcher = StrategySignalDispatcher({})
    aligned = _aligned_panel()
    start = np.datetime64("2025-01-01T00:00:00.000", "ms")
    aligned["datetime"] = start + np.arange(_N) * np.timedelta64(300_000, "ms")

    with pytest.raises(
        StrategySignalDispatchError,
        match="datetime grid does not match declared strategy timeframe",
    ):
        dispatcher.dispatch(
            {
                "strategy_class": "RegisteredStrategy",
                "strategy_timeframe": "4h",
                "params": {},
            },
            aligned=aligned,
            symbols=_SYMBOLS,
            unmapped_router=router,
            require_actual_engine=True,
        )
    assert calls == []

    valid_result = dispatcher.dispatch(
        {
            "strategy_class": "RegisteredStrategy",
            "strategy_timeframe": "5m",
            "params": {},
        },
        aligned=aligned,
        symbols=_SYMBOLS,
        unmapped_router=router,
        require_actual_engine=True,
    )
    assert calls == ["router"]
    assert valid_result[3]["evaluation_mode"] == "registry_simulator"


def test_strict_dispatch_rejects_conflicting_timeframes_before_registry_router() -> None:
    calls: list[str] = []

    def router(strategy_class, params, aligned, symbols) -> np.ndarray:
        calls.append("router")
        return np.zeros((len(symbols), len(aligned["datetime"])), dtype=float)

    dispatcher = StrategySignalDispatcher({})
    aligned = _aligned_panel()

    with pytest.raises(
        StrategySignalDispatchError,
        match="declared strategy timeframes disagree",
    ):
        dispatcher.dispatch(
            {
                "strategy_class": "RegisteredStrategy",
                "strategy_timeframe": "4h",
                "timeframe": "5m",
                "params": {},
            },
            aligned=aligned,
            symbols=_SYMBOLS,
            unmapped_router=router,
            require_actual_engine=True,
        )
    assert calls == []

    valid_result = dispatcher.dispatch(
        {
            "strategy_class": "RegisteredStrategy",
            "strategy_timeframe": "4h",
            "timeframe": "240m",
            "params": {},
        },
        aligned=aligned,
        symbols=_SYMBOLS,
        unmapped_router=router,
        require_actual_engine=True,
    )
    assert calls == ["router"]
    assert valid_result[3]["evaluation_mode"] == "registry_simulator"


@pytest.mark.parametrize("field", ("strategy_timeframe", "timeframe"))
@pytest.mark.parametrize("value", (None, "", " ", 300))
def test_strict_dispatch_rejects_malformed_declared_timeframe_before_router(
    field: str, value: object
) -> None:
    calls: list[str] = []

    def router(strategy_class, params, aligned, symbols) -> np.ndarray:
        calls.append("router")
        return np.zeros((len(symbols), len(aligned["datetime"])), dtype=float)

    candidate: dict[str, object] = {
        "strategy_class": "RegisteredStrategy",
        "params": {},
        field: value,
    }
    with pytest.raises(
        StrategySignalDispatchError,
        match="declared strategy timeframe is invalid",
    ):
        StrategySignalDispatcher({}).dispatch(
            candidate,
            aligned=_aligned_panel(),
            symbols=_SYMBOLS,
            unmapped_router=router,
            require_actual_engine=True,
        )
    assert calls == []


def test_event_simulator_rejects_cadence_conversion_failure_only_in_strict_mode() -> None:
    observed_window_seconds: list[int] = []

    class _WindowStrategy:
        def __init__(self, bars, events, **params):
            pass

        def calculate_signals_window(self, event):
            return None

        def calculate_signals(self, event):
            observed_window_seconds.append(event.window_seconds)

    aligned = _aligned_panel()
    aligned["datetime"] = np.array([object()] * _N, dtype=object)

    rr._simulate_event_driven_strategy_exposures(
        _WindowStrategy,
        params={},
        aligned=aligned,
        symbols=_SYMBOLS,
    )
    assert observed_window_seconds == [60] * _N

    with pytest.raises(ValueError, match=r"^unable to derive event cadence from datetime input$"):
        rr._simulate_event_driven_strategy_exposures(
            _WindowStrategy,
            params={},
            aligned=aligned,
            symbols=_SYMBOLS,
            require_actual_engine=True,
        )


@pytest.mark.parametrize(
    ("kind", "cause", "corruption_index"),
    [
        pytest.param(
            "later_drift",
            "datetime cadence is not positive, regular whole seconds",
            1,
            id="drift-first-interval",
        ),
        pytest.param(
            "later_drift",
            "datetime cadence is not positive, regular whole seconds",
            _N // 2,
            id="drift-middle-interval",
        ),
        pytest.param(
            "later_drift",
            "datetime cadence is not positive, regular whole seconds",
            _N - 1,
            id="drift-final-interval",
        ),
        pytest.param("nat", "datetime cadence contains NaT", 2),
        pytest.param(
            "repeated",
            "datetime cadence is not positive, regular whole seconds",
            2,
        ),
        pytest.param(
            "descending",
            "datetime cadence is not positive, regular whole seconds",
            2,
        ),
        pytest.param("naive", "datetime cadence requires UTC-aware datetime values", None),
        pytest.param(
            "non_utc",
            "datetime cadence requires UTC-aware datetime values",
            None,
        ),
    ],
)
def test_strict_event_simulator_rejects_every_invalid_datetime_interval(
    kind: str, cause: str, corruption_index: int | None
) -> None:
    class _WindowStrategy:
        def __init__(self, bars, events, **params):
            pass

        def calculate_signals_window(self, event):
            return None

        def calculate_signals(self, event):
            return None

    aligned = _aligned_panel()
    if kind == "later_drift":
        assert corruption_index is not None
        aligned["datetime"][corruption_index] += dtm.timedelta(seconds=1)
    elif kind == "nat":
        assert corruption_index is not None
        aligned["datetime"] = np.asarray(
            [value.replace(tzinfo=None) for value in aligned["datetime"]],
            dtype="datetime64[ns]",
        )
        aligned["datetime"][corruption_index] = np.datetime64("NaT", "ms")
    elif kind == "repeated":
        assert corruption_index is not None
        aligned["datetime"][corruption_index] = aligned["datetime"][corruption_index - 1]
    elif kind == "descending":
        assert corruption_index is not None
        aligned["datetime"][corruption_index] = aligned["datetime"][
            corruption_index - 1
        ] - dtm.timedelta(hours=4)
    elif kind == "naive":
        aligned["datetime"] = np.asarray(
            [value.replace(tzinfo=None) for value in aligned["datetime"]], dtype=object
        )
    else:
        non_utc = dtm.timezone(dtm.timedelta(hours=1))
        aligned["datetime"] = np.asarray(
            [value.astimezone(non_utc) for value in aligned["datetime"]], dtype=object
        )

    with pytest.raises(
        ValueError, match=r"^unable to derive event cadence from datetime input$"
    ) as error:
        rr._simulate_event_driven_strategy_exposures(
            _WindowStrategy,
            params={},
            aligned=aligned,
            symbols=_SYMBOLS,
            require_actual_engine=True,
        )
    assert isinstance(error.value.__cause__, ValueError)
    assert str(error.value.__cause__) == cause


@pytest.mark.parametrize(
    ("signal_symbol", "signal_type", "timestamp_kind", "message"),
    [
        (
            "DOGE/USDT",
            "LONG",
            "current",
            "signal symbol is outside the panel universe: DOGE/USDT",
        ),
        (
            None,
            "LONG",
            "current",
            "signal symbol must be an exact canonical panel symbol",
        ),
        (
            "btc/usdt",
            "LONG",
            "current",
            "signal symbol is not an exact canonical panel symbol: btc/usdt",
        ),
        ("BTC/USDT", "HOLD", "current", "unsupported signal type: HOLD"),
        ("BTC/USDT", "long", "current", "unsupported signal type: long"),
        ("BTC/USDT", " LONG", "current", "unsupported signal type:  LONG"),
        ("BTC/USDT", "LONG", "missing", "signal datetime is missing"),
        (
            "BTC/USDT",
            "LONG",
            "stale",
            "signal datetime does not match current event time",
        ),
        (
            "BTC/USDT",
            "LONG",
            "future",
            "signal datetime does not match current event time",
        ),
    ],
)
def test_strict_registry_rejects_invalid_queued_signals_while_legacy_route_labels_fallback(
    monkeypatch,
    signal_symbol: str | None,
    signal_type: str,
    timestamp_kind: str,
    message: str,
) -> None:
    class _RegisteredWindowStrategy:
        def __init__(self, bars, events, **params):
            self.events = events

        def calculate_signals_window(self, event):
            return None

        def calculate_signals(self, event):
            if timestamp_kind == "stale":
                signal_time = event.time - dtm.timedelta(hours=4)
            elif timestamp_kind == "future":
                signal_time = event.time + dtm.timedelta(hours=4)
            elif timestamp_kind == "missing":
                signal_time = None
            else:
                signal_time = event.time
            self.events.put(
                SimpleNamespace(
                    symbol=signal_symbol,
                    signal_type=signal_type,
                    datetime=signal_time,
                )
            )

    monkeypatch.setattr(
        "lumina_quant.strategies.registry.resolve_strategy_class",
        lambda strategy_class: _RegisteredWindowStrategy,
    )
    with pytest.raises(StrategySignalDispatchError) as error:
        _signal("RegisteredWindowStrategy", {}, _STRICT_ROUTE_ON)
    assert isinstance(error.value.__cause__, ValueError)
    assert str(error.value.__cause__) == message

    legacy = _signal("RegisteredWindowStrategy", {}, _ROUTE_ON)
    expected_mode = (
        "generic_fallback_proxy"
        if signal_symbol in {None, "DOGE/USDT"}
        or signal_type.upper() not in {"LONG", "SHORT", "EXIT"}
        else "registry_simulator"
    )
    assert legacy[3]["evaluation_mode"] == expected_mode


def test_strict_registry_accepts_valid_queued_signals(monkeypatch) -> None:
    class _RegisteredWindowStrategy:
        def __init__(self, bars, events, **params):
            self.events = events

        def calculate_signals_window(self, event):
            return None

        def calculate_signals(self, event):
            self.events.put(
                SimpleNamespace(symbol="BTC/USDT", signal_type="LONG", datetime=event.time)
            )

    monkeypatch.setattr(
        "lumina_quant.strategies.registry.resolve_strategy_class",
        lambda strategy_class: _RegisteredWindowStrategy,
    )

    result = _signal("RegisteredWindowStrategy", {}, _STRICT_ROUTE_ON)
    assert result[3]["evaluation_mode"] == "registry_simulator"
    assert np.all(result[2] > 0.0)
