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

import numpy as np

from lumina_quant.strategy_factory import research_runner as rr
from lumina_quant.strategy_factory.candidate_library import build_candidate_manifest

_SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT", "ADA/USDT"]
_N = 500
_ROUTE_ON = {"research": {"route_unmapped_registered_strategies": True}}
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


def _signal(klass: str, params: dict, scoring=None):
    candidate = {"strategy_class": klass, "params": params}
    return rr._strategy_signal(
        candidate, aligned=_aligned_panel(), symbols=_SYMBOLS, scoring_config=scoring
    )


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
