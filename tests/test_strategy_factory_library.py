from __future__ import annotations

import itertools
import json

from lumina_quant.strategies import factory_candidate_set
from lumina_quant.strategies.factory_candidate_set import (
    build_candidate_set,
    summarize_candidate_set,
)
from lumina_quant.strategies.pair_spread_zscore import (
    PairSpreadZScoreStrategy,
    bounded_pair_retune_params,
)
from lumina_quant.strategies.registry import get_strategy_names
from lumina_quant.strategy_factory import (
    build_article_pipeline_candidates,
    build_article_pipeline_manifest,
    build_binance_futures_candidates,
    build_single_asset_portfolio_sets,
    candidate_library,
    candidate_identity,
    select_diversified_shortlist,
)


def _evenly_spaced_grid_rows(
    param_grid: dict[str, tuple[object, ...]],
    *,
    max_rows: int,
) -> list[dict[str, object]]:
    if not param_grid:
        return [{}]
    keys = list(param_grid.keys())
    rows = [
        dict(zip(keys, values, strict=True))
        for values in itertools.product(*(param_grid[key] for key in keys))
    ]
    if max_rows <= 0 or len(rows) <= max_rows:
        return rows
    if max_rows == 1:
        return [rows[0]]

    step = (len(rows) - 1) / float(max_rows - 1)
    deduped: list[dict[str, object]] = []
    seen: set[str] = set()
    for idx in range(max_rows):
        row = rows[round(idx * step)]
        marker = json.dumps(row, sort_keys=True, separators=(",", ":"))
        if marker in seen:
            continue
        seen.add(marker)
        deduped.append(row)
    return deduped


def test_factory_candidate_set_is_large_and_diverse():
    candidates = build_candidate_set(
        symbols=["BTC/USDT", "ETH/USDT", "XAU/USDT", "XAG/USDT"],
        timeframes=["1s", "1m", "1h"],
    )
    summary = summarize_candidate_set(candidates)

    assert len(candidates) >= 100
    assert "RegimeBreakoutCandidateStrategy" in summary["strategies"]
    assert "VolatilityCompressionReversionStrategy" in summary["strategies"]
    assert "trend_breakout" in summary["families"]
    assert "mean_reversion" in summary["families"]


def test_sample_grid_rows_matches_existing_even_spacing_for_small_grid():
    param_grid = {
        "lookback": (8, 16, 24),
        "threshold": (0.1, 0.2, 0.3),
        "allow_short": (True, False),
    }

    expected = _evenly_spaced_grid_rows(
        param_grid,
        max_rows=5,
    )
    actual = factory_candidate_set._sample_grid_rows(
        param_grid,
        max_rows=5,
    )

    assert actual == expected


def test_build_candidate_set_caps_large_param_grid_without_eager_cartesian_blowup(monkeypatch):
    huge_grid = {f"p{idx}": tuple(range(12)) for idx in range(8)}
    template = factory_candidate_set.StrategyTemplate(
        name="SyntheticHugeGridStrategy",
        family="other",
        symbol_mode="single",
        param_grid=huge_grid,
        tags=("synthetic",),
    )

    monkeypatch.setattr(factory_candidate_set, "_strategy_templates", lambda: [template])

    candidates = build_candidate_set(
        symbols=["BTC/USDT"],
        timeframes=["1m"],
        max_param_rows_per_strategy=5,
    )

    assert len(candidates) == 5
    assert candidates[0]["params"] == {f"p{idx}": 0 for idx in range(8)}
    assert candidates[-1]["params"] == {f"p{idx}": 11 for idx in range(8)}


def test_strategy_factory_library_builds_candidates_and_shortlist():
    rows = build_binance_futures_candidates(
        timeframes=["1h"],
        symbols=["BTC/USDT", "ETH/USDT", "XAU/USDT", "XAG/USDT"],
    )
    assert len(rows) > 20

    mock_candidates = []
    for idx, item in enumerate(rows[:40]):
        as_dict = item.to_dict()
        as_dict["name"] = f"{as_dict['name']}_{idx}"
        as_dict["strategy_timeframe"] = as_dict["timeframe"]
        as_dict["hurdle_fields"] = {"oos": {"pass": True, "score": float(100 - idx)}}
        as_dict["identity"] = candidate_identity(as_dict)
        mock_candidates.append(as_dict)

    shortlist = select_diversified_shortlist(
        mock_candidates,
        mode="oos",
        max_total=12,
        max_per_family=6,
        max_per_timeframe=6,
    )
    assert 1 <= len(shortlist) <= 12
    assert all("shortlist_score" in row for row in shortlist)


def test_mean_reversion_candidate_builder_uses_bounded_5m_15m_slice_only():
    rows = build_binance_futures_candidates(
        timeframes=["1m", "5m", "15m"],
        symbols=["BTC/USDT", "ETH/USDT", "BNB/USDT", "TRX/USDT"],
    )
    volcomp_rows = [
        row for row in rows if row.strategy_class == "VolCompressionVWAPReversionStrategy"
    ]
    leadlag_rows = [row for row in rows if row.strategy_class == "LeadLagSpilloverStrategy"]

    assert {row.timeframe for row in volcomp_rows} == {"5m", "15m"}
    assert {row.timeframe for row in leadlag_rows} == {"5m", "15m"}
    assert all("volcomp_vwap_rev_guarded_" in row.name for row in volcomp_rows)
    assert all(float(row.params["entry_z"]) >= 2.0 for row in volcomp_rows)
    assert all(float(row.params["compression_vol_ratio"]) <= 0.78 for row in volcomp_rows)
    assert all(int(row.params["max_hold_bars"]) <= 36 for row in volcomp_rows)
    assert all(row.params["allow_short"] is False for row in volcomp_rows)


def test_pair_candidate_builder_prunes_5m_and_focuses_15m_pairs():
    rows = build_binance_futures_candidates(
        timeframes=["5m", "15m"],
        symbols=[
            "BTC/USDT",
            "BNB/USDT",
            "TRX/USDT",
            "XAU/USDT",
            "XAG/USDT",
            "XPT/USDT",
            "XPD/USDT",
        ],
    )
    pair_rows = [row for row in rows if row.strategy_class == "PairSpreadZScoreStrategy"]
    pair_set = {tuple(row.symbols) for row in pair_rows}

    assert {row.timeframe for row in pair_rows} == {"15m"}
    assert pair_set == {("BTC/USDT", "TRX/USDT"), ("BNB/USDT", "TRX/USDT")}
    assert all(float(row.params["entry_z"]) >= 2.6 for row in pair_rows)
    assert all(float(row.params["min_correlation"]) >= 0.25 for row in pair_rows)
    assert all(int(row.params["cooldown_bars"]) >= 10 for row in pair_rows)
    assert all(float(row.params["reentry_z_buffer"]) >= 0.35 for row in pair_rows)


def test_pair_candidate_builder_includes_4h_and_1d_rows():
    rows = build_binance_futures_candidates(
        timeframes=["4h", "1d"],
        symbols=[
            "BTC/USDT",
            "ETH/USDT",
            "BNB/USDT",
            "SOL/USDT",
            "TRX/USDT",
            "XAU/USDT",
            "XAG/USDT",
            "XPT/USDT",
            "XPD/USDT",
        ],
    )
    pair_rows = [row for row in rows if row.strategy_class == "PairSpreadZScoreStrategy"]
    pair_timeframes = {row.timeframe for row in pair_rows}
    pair_set = {(row.timeframe, tuple(row.symbols)) for row in pair_rows}

    assert pair_timeframes == {"4h", "1d"}
    assert ("4h", ("BTC/USDT", "ETH/USDT")) in pair_set
    assert ("4h", ("ETH/USDT", "SOL/USDT")) in pair_set
    assert ("4h", ("XAU/USDT", "XAG/USDT")) in pair_set
    assert ("4h", ("BTC/USDT", "XAU/USDT")) in pair_set
    assert ("4h", ("ETH/USDT", "XAU/USDT")) in pair_set
    assert ("4h", ("BNB/USDT", "XAU/USDT")) in pair_set
    assert ("4h", ("BTC/USDT", "XAG/USDT")) in pair_set
    assert ("1d", ("BTC/USDT", "BNB/USDT")) in pair_set
    assert ("1d", ("BTC/USDT", "TRX/USDT")) in pair_set
    assert ("1d", ("XPT/USDT", "XPD/USDT")) in pair_set
    assert ("1d", ("BTC/USDT", "XAU/USDT")) in pair_set
    assert ("1d", ("ETH/USDT", "XAU/USDT")) in pair_set
    assert ("1d", ("BNB/USDT", "XAU/USDT")) in pair_set
    high_tf_rows = [row for row in pair_rows if row.timeframe == "4h"]
    daily_rows = [row for row in pair_rows if row.timeframe == "1d"]

    assert all(float(row.params["min_correlation"]) >= 0.03 for row in high_tf_rows)
    assert all(int(row.params["cooldown_bars"]) >= 3 for row in high_tf_rows)
    assert all(float(row.params["reentry_z_buffer"]) >= 0.12 for row in high_tf_rows)
    assert all(int(row.params["max_hold_bars"]) <= 144 for row in high_tf_rows)
    assert any(float(row.params["entry_z"]) < 1.8 for row in high_tf_rows)
    assert len({int(row.params["lookback_window"]) for row in high_tf_rows}) >= 2

    assert all(float(row.params["min_correlation"]) >= 0.0 for row in daily_rows)
    assert all(int(row.params["cooldown_bars"]) >= 1 for row in daily_rows)
    assert all(float(row.params["reentry_z_buffer"]) >= 0.08 for row in daily_rows)
    assert all(int(row.params["max_hold_bars"]) <= 36 for row in daily_rows)


def test_candidate_library_includes_article_inspired_carry_trend_factor_rotation():
    rows = build_binance_futures_candidates(
        timeframes=["1h", "4h"],
        symbols=["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "TRX/USDT"],
    )
    factor_rows = [row for row in rows if row.strategy_class == "CarryTrendFactorRotationStrategy"]

    assert {row.timeframe for row in factor_rows} == {"1h", "4h"}
    assert len(factor_rows) == 6
    assert all(row.family == "cross_sectional" for row in factor_rows)
    assert all(
        row.symbols == ("BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "TRX/USDT")
        for row in factor_rows
    )
    assert all("carry_trend_factor_rotation_" in row.name for row in factor_rows)
    assert all("article_family:carry-trend-factor-rotation" in row.tags for row in factor_rows)
    assert all(
        row.metadata.get("article_reference") == "quant-company-profit-mechanisms"
        for row in factor_rows
    )
    production_ready = [row for row in factor_rows if row.metadata.get("production_ready") is True]
    assert {row.timeframe for row in production_ready} == {"1h", "4h"}
    assert all(bool(row.params["allow_short"]) is False for row in production_ready)


def test_candidate_library_includes_deep_research_report_leaf_candidates(monkeypatch):
    monkeypatch.setattr(candidate_library, "_has_perp_support_data", lambda: True)

    rows = build_binance_futures_candidates(
        timeframes=["5m", "15m", "1h", "4h"],
        symbols=["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "TRX/USDT"],
    )
    by_class = {row.strategy_class: row for row in rows}

    expected = {
        "FundingDislocationTrendCarryStrategy",
        "VolManagedMomentumCrashGateStrategy",
        "FlowImbalanceLiquidationSweepStrategy",
    }
    assert expected.issubset(by_class)

    report_rows = [row for row in rows if row.strategy_class in expected]
    assert report_rows
    assert all(row.family == "deep_research_leaf" for row in report_rows)
    assert all("deep_research_report_20260608" in row.tags for row in report_rows)
    assert all(
        row.metadata["source_report"] == "desktop-deep-research-report-20260608"
        for row in report_rows
    )
    assert all(row.metadata["leaf_only"] is True for row in report_rows)
    assert all(row.metadata["research_only"] is True for row in report_rows)
    assert all(row.metadata["no_nested_oos_mining"] is True for row in report_rows)
    assert all(row.metadata["requires_fresh_forward_shadow"] is True for row in report_rows)
    assert all(
        row.metadata["deployment_gate"] == "blocked_until_fresh_forward_cost_telemetry_pbo_dsr_psr"
        for row in report_rows
    )

    expected_article_families = {
        "FundingDislocationTrendCarryStrategy": "deep-research-funding-dislocation-trend-carry",
        "VolManagedMomentumCrashGateStrategy": "deep-research-vol-managed-momentum-crash-gate",
        "FlowImbalanceLiquidationSweepStrategy": "deep-research-flow-imbalance-liquidation-sweep",
    }
    for strategy_class, article_family in expected_article_families.items():
        strategy_rows = [row for row in report_rows if row.strategy_class == strategy_class]
        assert strategy_rows
        assert all(
            article_family in row.metadata["article_pipeline_family_ids"] for row in strategy_rows
        )
        assert all(f"article_family:{article_family}" in row.tags for row in strategy_rows)

    flow_rows = [
        row for row in rows if row.strategy_class == "FlowImbalanceLiquidationSweepStrategy"
    ]
    assert flow_rows
    assert all(row.metadata["symbol_scope"] == "major_crypto_only" for row in flow_rows)
    assert all(row.metadata["initial_gross_cap_hint"] == 0.15 for row in flow_rows)


def test_candidate_library_includes_last_day_liquidity_regime_family():
    rows = build_binance_futures_candidates(
        timeframes=["1h"],
        symbols=["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "TRX/USDT"],
    )
    regime_rows = [row for row in rows if row.strategy_class == "LastDayLiquidityRegimeStrategy"]

    assert len(regime_rows) == 2
    assert {row.timeframe for row in regime_rows} == {"1h"}
    assert all(row.family == "cross_sectional" for row in regime_rows)
    assert all("last_day_liquidity_regime_" in row.name for row in regime_rows)
    assert all(int(row.params["momentum_lookback_bars"]) == 24 for row in regime_rows)
    assert any(bool(row.params["illiquid_reversal"]) for row in regime_rows)


def test_candidate_library_includes_abnormal_return_continuation_family():
    rows = build_binance_futures_candidates(
        timeframes=["1d"],
        symbols=["BNB/USDT", "TRX/USDT"],
    )
    event_rows = [row for row in rows if row.strategy_class == "AbnormalReturnContinuationStrategy"]

    assert len(event_rows) == 4
    assert {row.timeframe for row in event_rows} == {"1d"}
    assert all(row.family == "event_alpha" for row in event_rows)
    assert all("abnormal_return_continuation_" in row.name for row in event_rows)


def test_build_article_pipeline_manifest_collects_existing_and_new_article_candidates():
    rows = build_article_pipeline_candidates(
        timeframes=["5m", "15m", "30m", "1h", "4h"],
        symbols=[
            "BTC/USDT",
            "ETH/USDT",
            "BNB/USDT",
            "SOL/USDT",
            "TRX/USDT",
            "XAU/USDT",
            "XAG/USDT",
        ],
        max_per_family=0,
        max_total=0,
    )
    assert rows
    assert all("article_pipeline" in row.tags for row in rows)
    strategy_names = {row.strategy_class for row in rows}
    assert "CompositeTrendStrategy" in strategy_names
    assert "TopCapTimeSeriesMomentumStrategy" in strategy_names
    assert "CarryTrendFactorRotationStrategy" in strategy_names
    assert "FundingLiquidationCrowdingFadeStrategy" in strategy_names
    assert "PairSpreadZScoreStrategy" in strategy_names

    manifest = build_article_pipeline_manifest(
        timeframes=["5m", "15m", "30m", "1h", "4h"],
        symbols=[
            "BTC/USDT",
            "ETH/USDT",
            "BNB/USDT",
            "SOL/USDT",
            "TRX/USDT",
            "XAU/USDT",
            "XAG/USDT",
        ],
        max_per_family=3,
        max_total=0,
    )
    assert manifest["candidate_count"] > 0
    assert manifest["family_counts"]
    assert manifest["article_family_counts"]
    assert "carry-trend-factor-rotation" in manifest["article_family_counts"]
    assert manifest["max_per_family"] == 3


def test_pair_candidate_builder_adds_1h_pair_state_variants():
    rows = build_binance_futures_candidates(
        timeframes=["1h"],
        symbols=[
            "BTC/USDT",
            "ETH/USDT",
            "BNB/USDT",
            "SOL/USDT",
            "TRX/USDT",
        ],
    )
    pair_rows = [row for row in rows if row.strategy_class == "PairSpreadZScoreStrategy"]
    by_name = {row.name: row for row in pair_rows}

    vwap_row = by_name["pair_spread_1h_state_vwap_bnbusdt_trxusdt_2.2_0.55"]
    assert vwap_row.params["vwap_window"] == 72
    assert vwap_row.params["min_volume_ratio"] == 0.20
    assert "pair_state" in vwap_row.tags

    volconv_row = by_name["pair_spread_1h_state_volconv_bnbusdt_trxusdt_2.2_0.55"]
    assert volconv_row.params["vol_lag_bars"] == 2
    assert volconv_row.params["min_vol_convergence"] == 0.60
    assert "pair_state" in volconv_row.tags

    atr_row = by_name["pair_spread_1h_state_atr_bnbusdt_trxusdt_2.2_0.55"]
    assert atr_row.params["atr_window"] == 14
    assert atr_row.params["atr_max_pct"] == 0.04
    assert "pair_state" in atr_row.tags

    adaptive_fast_row = by_name["pair_spread_1h_adaptive_rls_fast_bnbusdt_trxusdt_2.5_0.65"]
    assert adaptive_fast_row.params["hedge_mode"] == "rls"
    assert adaptive_fast_row.params["hedge_forgetting_factor"] == 0.985
    assert adaptive_fast_row.params["take_profit_pct"] == 0.06
    assert "adaptive_hedge" in adaptive_fast_row.tags

    adaptive_stable_row = by_name["pair_spread_1h_adaptive_rls_stable_bnbusdt_trxusdt_2.6_0.70"]
    assert adaptive_stable_row.params["hedge_mode"] == "rls"
    assert adaptive_stable_row.params["hedge_forgetting_factor"] == 0.992
    assert adaptive_stable_row.params["atr_window"] == 14
    assert "focused_followup" in adaptive_stable_row.tags


def test_composite_trend_candidate_builder_uses_explicit_30m_1h_stability_slice():
    rows = build_binance_futures_candidates(
        timeframes=["30m", "1h", "4h", "1d"],
        symbols=[
            "BTC/USDT",
            "ETH/USDT",
            "BNB/USDT",
            "SOL/USDT",
            "XAU/USDT",
            "XAG/USDT",
        ],
    )
    trend_rows = [row for row in rows if row.strategy_class == "CompositeTrendStrategy"]

    assert len(trend_rows) == 9
    assert {row.timeframe for row in trend_rows} == {"30m", "1h"}
    assert all("composite_trend_stable_" in row.name for row in trend_rows)
    assert all("exit_score_cross" in row.params for row in trend_rows)
    assert all("max_signal_strength" in row.params for row in trend_rows)

    one_hour_rows = [row for row in trend_rows if row.timeframe == "1h"]
    assert one_hour_rows
    assert all(row.params["allow_short"] is False for row in one_hour_rows)
    assert all(float(row.params["crowding_block_threshold"]) <= 0.70 for row in one_hour_rows)

    crashguard_row = next(
        row
        for row in trend_rows
        if row.name == "composite_trend_stable_30m_stable_ls_crashguard_ls_0.75_0.45_0.20_0.82"
    )
    assert crashguard_row.params["benchmark_regime_ma"] == 96
    assert crashguard_row.params["benchmark_symbol"] == "BTC/USDT"
    assert crashguard_row.metadata["benchmark_regime_ma"] == 96
    assert "crash_aware" in crashguard_row.tags

    exec_trail_row = next(
        row
        for row in trend_rows
        if row.name == "composite_trend_stable_30m_stable_ls_exec_trail_ls_0.75_0.45_0.20_0.80"
    )
    assert exec_trail_row.params["trail_atr_mult"] == 2.4
    assert exec_trail_row.params["atr_stop_mult"] == 1.8
    assert "execution_risk" in exec_trail_row.tags

    exec_shorthold_row = next(
        row
        for row in trend_rows
        if row.name == "composite_trend_stable_30m_stable_ls_exec_shorthold_ls_0.75_0.45_0.20_0.80"
    )
    assert exec_shorthold_row.params["max_hold_bars"] == 640
    assert exec_shorthold_row.params["exit_score_cross"] == 0.04
    assert "execution_risk" in exec_shorthold_row.tags


def test_pair_spread_default_pair_prefers_xpt_xpd_when_available():
    left, right = PairSpreadZScoreStrategy._resolve_default_pair(
        ["XPT/USDT", "XPD/USDT", "BTC/USDT"]
    )
    assert (left, right) == ("XPT/USDT", "XPD/USDT")


def test_bounded_pair_retune_params_raise_15m_turnover_guards():
    default_params = bounded_pair_retune_params("1h")
    focused_params = bounded_pair_retune_params("15m")

    assert focused_params["lookback_window"] > default_params["lookback_window"]
    assert focused_params["min_correlation"] > default_params["min_correlation"]
    assert focused_params["cooldown_bars"] > default_params["cooldown_bars"]
    assert focused_params["reentry_z_buffer"] > default_params["reentry_z_buffer"]


def test_bounded_pair_retune_params_shorten_high_timeframe_holds_for_oos_realization():
    default_params = bounded_pair_retune_params("1h")
    four_hour = bounded_pair_retune_params("4h")
    daily = bounded_pair_retune_params("1d")

    assert four_hour["lookback_window"] < default_params["lookback_window"]
    assert four_hour["hedge_window"] < default_params["hedge_window"]
    assert four_hour["max_hold_bars"] < default_params["max_hold_bars"]
    assert four_hour["min_correlation"] <= default_params["min_correlation"]

    assert daily["lookback_window"] < four_hour["lookback_window"]
    assert daily["hedge_window"] < four_hour["hedge_window"]
    assert daily["max_hold_bars"] < four_hour["max_hold_bars"]
    assert daily["cooldown_bars"] < four_hour["cooldown_bars"]


def test_pair_candidate_builder_adds_mixed_asset_pairs_only_when_symbols_present():
    rows = build_binance_futures_candidates(
        timeframes=["4h"],
        symbols=["BTC/USDT", "ETH/USDT", "XAU/USDT", "XAG/USDT"],
    )
    pair_rows = [row for row in rows if row.strategy_class == "PairSpreadZScoreStrategy"]
    pair_set = {tuple(row.symbols) for row in pair_rows}

    assert ("BTC/USDT", "XAU/USDT") in pair_set
    assert ("ETH/USDT", "XAU/USDT") in pair_set
    assert ("XAU/USDT", "XAG/USDT") in pair_set
    assert ("BTC/USDT", "XAG/USDT") in pair_set


def test_candidate_library_adds_article_pipeline_provenance_tags_and_metadata():
    rows = build_binance_futures_candidates(
        timeframes=["5m", "15m", "30m", "1h", "4h"],
        symbols=[
            "BTC/USDT",
            "ETH/USDT",
            "BNB/USDT",
            "SOL/USDT",
            "TRX/USDT",
            "XAU/USDT",
            "XAG/USDT",
        ],
    )
    by_name = {row.name: row for row in rows}

    trend_row = next(row for row in rows if row.strategy_class == "CompositeTrendStrategy")
    assert trend_row.metadata["article_pipeline_family_ids"] == [
        "regime-conditioned-composite-trend"
    ]
    assert "article_pipeline" in trend_row.tags
    assert "article_family:regime-conditioned-composite-trend" in trend_row.tags

    volcomp_row = next(
        row for row in rows if row.strategy_class == "VolCompressionVWAPReversionStrategy"
    )
    assert volcomp_row.metadata["article_pipeline_family_ids"] == [
        "vol-compression-break-reversion"
    ]
    assert "article_family:vol-compression-break-reversion" in volcomp_row.tags

    vwap_row = next(row for row in rows if row.strategy_class == "VwapReversionStrategy")
    assert vwap_row.metadata["article_pipeline_family_ids"] == ["intraday-vwap-reversion"]
    assert "article_family:intraday-vwap-reversion" in vwap_row.tags

    std_row = next(row for row in rows if row.strategy_class == "MeanReversionStdStrategy")
    assert std_row.metadata["article_pipeline_family_ids"] == ["single-asset-zscore-reversion"]
    assert "article_family:single-asset-zscore-reversion" in std_row.tags

    residual_std_row = by_name["mean_reversion_std_15m_resid_btc_ls_64_2.00"]
    assert residual_std_row.params["residualize_btc"] is True
    assert residual_std_row.params["btc_symbol"] == "BTC/USDT"
    assert residual_std_row.metadata["residualize_btc"] is True
    assert "btc_beta_neutral" in residual_std_row.tags

    liquidity_row = next(
        row for row in rows if row.strategy_class == "LiquidityShockReversionStrategy"
    )
    assert liquidity_row.metadata["article_pipeline_family_ids"] == ["liquidity-shock-reversion"]
    assert "article_family:liquidity-shock-reversion" in liquidity_row.tags

    session_liquidity_row = next(
        row for row in rows if row.strategy_class == "SessionLiquidityVacuumFadeStrategy"
    )
    assert session_liquidity_row.metadata["article_pipeline_family_ids"] == [
        "session-transition-liquidity-vacuum-fade"
    ]
    assert "article_family:session-transition-liquidity-vacuum-fade" in session_liquidity_row.tags

    funding_crowding_row = next(
        row for row in rows if row.strategy_class == "FundingLiquidationCrowdingFadeStrategy"
    )
    assert funding_crowding_row.metadata["article_pipeline_family_ids"] == [
        "funding-liquidation-crowding-fade"
    ]
    assert "article_family:funding-liquidation-crowding-fade" in funding_crowding_row.tags

    basis_snapback_row = next(
        row for row in rows if row.strategy_class == "BasisSnapbackReversionStrategy"
    )
    assert basis_snapback_row.metadata["article_pipeline_family_ids"] == [
        "basis-snapback-reversion"
    ]
    assert "article_family:basis-snapback-reversion" in basis_snapback_row.tags

    vol_of_vol_row = next(
        row for row in rows if row.strategy_class == "VolOfVolExhaustionFadeStrategy"
    )
    assert vol_of_vol_row.metadata["article_pipeline_family_ids"] == ["vol-of-vol-exhaustion-fade"]
    assert "article_family:vol-of-vol-exhaustion-fade" in vol_of_vol_row.tags

    session_residual_row = next(
        row for row in rows if row.strategy_class == "SessionGatedResidualBasketReversionStrategy"
    )
    assert session_residual_row.metadata["article_pipeline_family_ids"] == [
        "session-gated-residual-basket-reversion"
    ]
    assert "article_family:session-gated-residual-basket-reversion" in session_residual_row.tags

    contagion_row = next(
        row for row in rows if row.strategy_class == "CrossAssetLiquidationContagionFadeStrategy"
    )
    assert contagion_row.metadata["article_pipeline_family_ids"] == [
        "cross-asset-liquidation-contagion-fade"
    ]
    assert "article_family:cross-asset-liquidation-contagion-fade" in contagion_row.tags

    leadlag_row = next(row for row in rows if row.strategy_class == "LeadLagSpilloverStrategy")
    assert leadlag_row.metadata["article_pipeline_family_ids"] == ["lead-lag-regime-spillover"]
    assert "article_family:lead-lag-regime-spillover" in leadlag_row.tags

    mixed_pair_row = by_name["pair_spread_4h_participation_btcusdt_xauusdt_1.6_0.35"]
    assert mixed_pair_row.metadata["article_pipeline_family_ids"] == ["crypto-metal-residual-pairs"]
    assert "article_family:crypto-metal-residual-pairs" in mixed_pair_row.tags

    sector_30m_row = by_name["pair_spread_30m_sector_btcusdt_trxusdt_2.0_0.50"]
    assert sector_30m_row.metadata["article_pipeline_family_ids"] == ["sector-dispersion-reversion"]
    assert "article_family:sector-dispersion-reversion" in sector_30m_row.tags

    crypto_pair_row = by_name["pair_spread_1h_core_btcusdt_trxusdt_1.8_0.45"]
    assert crypto_pair_row.metadata["article_pipeline_family_ids"] == [
        "sector-dispersion-reversion"
    ]
    assert "article_family:sector-dispersion-reversion" in crypto_pair_row.tags

    lag_row = by_name["lag_convergence_4h_metals_core_xauusdt_xagusdt_2_0.018"]
    assert lag_row.metadata["article_pipeline_family_ids"] == ["metals-lag-convergence"]
    assert "article_family:metals-lag-convergence" in lag_row.tags

    breakout_row = by_name["rolling_breakout_30m_loose_lo_48_0.001"]
    assert breakout_row.metadata["article_pipeline_family_ids"] == ["regime-breakout-thrust"]
    assert "article_family:regime-breakout-thrust" in breakout_row.tags

    regime_row = by_name["regime_breakout_30m_trend_guarded_48_0.68"]
    assert regime_row.metadata["article_pipeline_family_ids"] == ["regime-breakout-thrust"]
    assert "article_family:regime-breakout-thrust" in regime_row.tags

    topcap_row = by_name["topcap_tsmom_1h_balanced_16_4_0.015"]
    assert topcap_row.metadata["article_pipeline_family_ids"] == [
        "topcap-rotation-relative-momentum"
    ]
    assert "article_family:topcap-rotation-relative-momentum" in topcap_row.tags

    breadth_row = by_name["breadth_thrust_failure_reversal_30m_balanced_ls_16_0.80"]
    assert breadth_row.metadata["article_pipeline_family_ids"] == [
        "breadth-thrust-failure-reversal"
    ]
    assert "article_family:breadth-thrust-failure-reversal" in breadth_row.tags

    trend_exhaustion_row = by_name["multi_horizon_trend_exhaustion_fade_30m_balanced_ls_16_1.6"]
    assert trend_exhaustion_row.metadata["article_pipeline_family_ids"] == [
        "multi-horizon-trend-exhaustion-fade"
    ]
    assert "article_family:multi-horizon-trend-exhaustion-fade" in trend_exhaustion_row.tags

    residual_basket_row = by_name["residual_basket_reversion_15m_resid_btc_ls_48_1.80"]
    assert residual_basket_row.metadata["article_pipeline_family_ids"] == [
        "cross-sectional-residual-basket-reversion"
    ]
    assert "article_family:cross-sectional-residual-basket-reversion" in residual_basket_row.tags

    vol_regime_row = by_name["volatility_regime_residual_basket_reversion_15m_volcap_ls_48_1.80"]
    assert vol_regime_row.params["btc_vol_fast"] == 12
    assert vol_regime_row.params["btc_vol_ratio_cap"] == 1.15
    assert "volatility_regime" in vol_regime_row.tags

    residual_topcap_row = by_name["topcap_tsmom_1h_resid_btc_16_4_0.010"]
    assert residual_topcap_row.params["residualize_btc"] is True
    assert residual_topcap_row.metadata["residualize_btc"] is True
    assert "residual_momentum" in residual_topcap_row.tags

    factor_neutral_topcap_row = by_name["topcap_tsmom_1h_resid_beta_neutral_24_4_0.008"]
    assert factor_neutral_topcap_row.params["residualize_btc"] is True
    assert factor_neutral_topcap_row.params["residualize_mean"] is True
    assert factor_neutral_topcap_row.metadata["residualize_mean"] is True
    assert "factor_neutral" in factor_neutral_topcap_row.tags

    crashguard_topcap_row = by_name["topcap_tsmom_1h_crashguard_16_4_0.015"]
    assert crashguard_topcap_row.params["benchmark_drawdown_window"] == 48
    assert crashguard_topcap_row.params["benchmark_drawdown_limit"] == 0.08
    assert crashguard_topcap_row.metadata["benchmark_drawdown_window"] == 48
    assert "crash_aware" in crashguard_topcap_row.tags

    tightstop_topcap_row = by_name["topcap_tsmom_1h_exec_tightstop_16_4_0.015"]
    assert tightstop_topcap_row.params["stop_loss_pct"] == 0.05
    assert "execution_risk" in tightstop_topcap_row.tags

    fastrebalance_topcap_row = by_name["topcap_tsmom_1h_exec_fastrebalance_16_2_0.012"]
    assert fastrebalance_topcap_row.params["rebalance_bars"] == 2
    assert fastrebalance_topcap_row.params["signal_threshold"] == 0.012
    assert "execution_risk" in fastrebalance_topcap_row.tags

    takeprofit_topcap_row = by_name["topcap_tsmom_1h_exec_takeprofit_16_4_0.015"]
    assert takeprofit_topcap_row.params["take_profit_pct"] == 0.10
    assert "take_profit" in takeprofit_topcap_row.tags

    tightstop_tp_topcap_row = by_name["topcap_tsmom_1h_exec_tightstop_tp_16_4_0.015"]
    assert tightstop_tp_topcap_row.params["stop_loss_pct"] == 0.05
    assert tightstop_tp_topcap_row.params["take_profit_pct"] == 0.10
    assert "execution_risk" in tightstop_tp_topcap_row.tags
    assert "take_profit" in tightstop_tp_topcap_row.tags

    fastrebalance_tp_topcap_row = by_name["topcap_tsmom_1h_exec_fastrebalance_tp_16_2_0.012"]
    assert fastrebalance_tp_topcap_row.params["rebalance_bars"] == 2
    assert fastrebalance_tp_topcap_row.params["take_profit_pct"] == 0.08
    assert "take_profit" in fastrebalance_tp_topcap_row.tags


def test_candidate_library_generates_lag_convergence_for_xpt_xpd_pairs():
    rows = build_binance_futures_candidates(
        timeframes=["4h", "1d"],
        symbols=["XAU/USDT", "XAG/USDT", "XPT/USDT", "XPD/USDT"],
    )
    lag_rows = [row for row in rows if row.strategy_class == "LagConvergenceStrategy"]
    lag_names = {row.name for row in lag_rows}

    assert lag_rows
    assert {row.timeframe for row in lag_rows} == {"4h", "1d"}
    assert any(tuple(row.symbols) == ("XPT/USDT", "XPD/USDT") for row in lag_rows)
    assert "lag_convergence_4h_metals_core_xptusdt_xpdusdt_2_0.018" in lag_names
    assert "lag_convergence_1d_metals_core_xptusdt_xpdusdt_1_0.012" in lag_names


def test_candidate_library_generates_additional_existing_strategy_families():
    rows = build_binance_futures_candidates(
        timeframes=["5m", "15m", "30m", "1h"],
        symbols=["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "TRX/USDT"],
    )
    names = {row.strategy_class for row in rows}

    assert "VwapReversionStrategy" in names
    assert "MeanReversionStdStrategy" in names
    assert "RollingBreakoutStrategy" in names
    assert "RegimeBreakoutCandidateStrategy" in names
    assert "TopCapTimeSeriesMomentumStrategy" in names


def test_candidate_library_generates_profit_reboot_moonshot_families():
    rows = build_binance_futures_candidates(
        timeframes=["5m", "15m", "1h", "4h"],
        symbols=["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "TRX/USDT"],
    )
    by_name = {row.name: row for row in rows}
    strategy_names = {row.strategy_class for row in rows}

    assert "PanicReboundMeanReversionStrategy" in strategy_names
    assert "SessionFilteredPairCarryStrategy" in strategy_names
    assert "ProfitMoonshotTrendStrategy" in strategy_names
    assert "ProfitMoonshotBreakoutStrategy" in strategy_names
    assert "ProfitMoonshotReversionStrategy" in strategy_names

    panic = by_name["panic_rebound_mr_5m_volume_strict_32_0.025"]
    assert panic.family == "profit_reboot_mean_reversion"
    assert "profit_reboot_20260501" in panic.tags
    assert panic.metadata["article_pipeline_family_ids"] == [
        "profit-reboot-panic-rebound-mean-reversion"
    ]
    assert panic.params["confirmation_bars"] == 3

    pair = by_name["session_filtered_pair_carry_1h_bnbtrx_overlap_2.2_0.50"]
    assert pair.family == "profit_reboot_pair_carry"
    assert pair.symbols == ("BNB/USDT", "TRX/USDT")
    assert pair.params["allowed_session_utc_hours"] == "0,1,8,9,13,14,15,20,21"
    assert pair.metadata["article_pipeline_family_ids"] == [
        "profit-reboot-session-filtered-pair-carry"
    ]

    moonshot = by_name["profit_moonshot_breakout_1h_expansion_48_0.006"]
    assert moonshot.family == "profit_moonshot_breakout"
    assert "profit_moonshot_20260501" in moonshot.tags
    assert "no_timeframe_aggregator" in moonshot.tags
    assert moonshot.metadata["article_pipeline_family_ids"] == [
        "profit-moonshot-range-expansion-breakout"
    ]
    assert moonshot.params["gross_exposure"] == 0.10


def test_candidate_library_generates_alpha101_formula_candidates_with_tuned_overrides():
    rows = build_binance_futures_candidates(
        timeframes=["1h", "4h"],
        symbols=["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "TRX/USDT"],
    )
    alpha_rows = [row for row in rows if row.strategy_class == "Alpha101FormulaStrategy"]

    assert alpha_rows
    assert {row.timeframe for row in alpha_rows} == {"1h", "4h"}
    assert any("alpha_param_overrides" in row.params for row in alpha_rows)
    assert all(row.metadata["alpha_param_override_keys"] for row in alpha_rows)
    assert any(row.metadata["alpha_id"] == 101 for row in alpha_rows)


def test_registry_exposes_new_candidate_strategies():
    names = get_strategy_names()
    assert "RegimeBreakoutCandidateStrategy" in names
    assert "VolatilityCompressionReversionStrategy" in names
    assert "PanicReboundMeanReversionStrategy" in names
    assert "SessionFilteredPairCarryStrategy" in names


def test_shortlist_filters_weak_single_and_assigns_weights():
    candidates = [
        {
            "name": "rsi_1m_candidate",
            "strategy_timeframe": "1m",
            "symbols": ["BTC/USDT"],
            "hurdle_fields": {"oos": {"pass": True, "score": 2.0}},
            "oos": {"return": 0.02, "sharpe": 1.0, "mdd": 0.08, "trades": 19},
            "params": {"rsi_period": 14},
        },
        {
            "name": "pair_z_1m_candidate",
            "strategy_timeframe": "1m",
            "symbols": ["BTC/USDT", "ETH/USDT"],
            "hurdle_fields": {"oos": {"pass": True, "score": 5.5}},
            "oos": {"return": 0.11, "sharpe": 1.4, "mdd": 0.05, "trades": 30},
            "params": {"entry_z": 2.0},
        },
        {
            "name": "carry_trend_factor_rotation_1h_guarded",
            "strategy_class": "CarryTrendFactorRotationStrategy",
            "family": "cross_sectional",
            "strategy_timeframe": "1h",
            "symbols": ["BTC/USDT", "ETH/USDT", "BNB/USDT"],
            "tags": ["cross_sectional", "carry", "momentum", "defensive", "crypto"],
            "hurdle_fields": {"oos": {"pass": True, "score": 4.2}},
            "oos": {"return": 0.09, "sharpe": 1.1, "mdd": 0.06, "trades": 24},
            "params": {"carry_half_life": 8},
        },
    ]

    shortlist = select_diversified_shortlist(
        candidates,
        mode="oos",
        max_total=5,
        max_per_family=5,
        max_per_timeframe=5,
        single_min_score=0.0,
        single_min_return=0.0,
        single_min_sharpe=0.7,
        single_min_trades=20,
        drop_single_without_metrics=True,
        include_weights=True,
    )

    # Weak single strategy is filtered out; mixed strategies remain and get normalized weights.
    assert len(shortlist) == 2
    assert all(float(row.get("portfolio_weight", 0.0)) > 0.0 for row in shortlist)
    total_weight = sum(float(row.get("portfolio_weight", 0.0)) for row in shortlist)
    assert abs(total_weight - 1.0) < 1e-9


def test_shortlist_default_blocks_generic_multi_asset_but_keeps_allowlisted_factor_rotation():
    candidates = [
        {
            "name": "carry_trend_factor_rotation_1h_guarded",
            "strategy_class": "CarryTrendFactorRotationStrategy",
            "family": "cross_sectional",
            "strategy_timeframe": "1h",
            "symbols": ["BTC/USDT", "ETH/USDT", "BNB/USDT"],
            "tags": ["cross_sectional", "carry", "momentum", "defensive", "crypto"],
            "hurdle_fields": {"oos": {"pass": True, "score": 4.2}},
            "oos": {"return": 0.09, "sharpe": 1.1, "mdd": 0.06, "trades": 24},
        },
        {
            "name": "topcap_tsmom_1h_candidate",
            "strategy_class": "TopCapTimeSeriesMomentumStrategy",
            "family": "trend",
            "strategy_timeframe": "1h",
            "symbols": ["BTC/USDT", "ETH/USDT", "BNB/USDT"],
            "hurdle_fields": {"oos": {"pass": True, "score": 6.0}},
            "oos": {"return": 0.12, "sharpe": 1.5, "mdd": 0.05, "trades": 30},
        },
    ]

    shortlist = select_diversified_shortlist(
        candidates,
        mode="oos",
        max_total=5,
        max_per_family=5,
        max_per_timeframe=5,
    )

    assert {row["name"] for row in shortlist} == {"carry_trend_factor_rotation_1h_guarded"}


def test_shortlist_default_keeps_existing_allowlisted_multi_asset_non_trend_families():
    candidates = [
        {
            "name": "leadlag_spillover_15m_0.35_lag3",
            "strategy_class": "LeadLagSpilloverStrategy",
            "family": "intraday_alpha",
            "strategy_timeframe": "15m",
            "symbols": ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "TRX/USDT"],
            "tags": ["leadlag", "cross-asset", "intraday", "alpha"],
            "hurdle_fields": {"oos": {"pass": True, "score": 4.4}},
            "oos": {"return": 0.06, "sharpe": 1.0, "mdd": 0.05, "trades": 25},
        },
        {
            "name": "last_day_liquidity_regime_1h_balanced",
            "strategy_class": "LastDayLiquidityRegimeStrategy",
            "family": "cross_sectional",
            "strategy_timeframe": "1h",
            "symbols": ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT"],
            "tags": ["cross_sectional", "pure_momentum", "liquidity_conditioned", "crypto"],
            "hurdle_fields": {"oos": {"pass": True, "score": 4.0}},
            "oos": {"return": 0.07, "sharpe": 1.1, "mdd": 0.04, "trades": 20},
        },
        {
            "name": "perp_crowding_carry_30m_0.35_0.10",
            "strategy_class": "PerpCrowdingCarryStrategy",
            "family": "carry",
            "strategy_timeframe": "30m",
            "symbols": ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT"],
            "tags": ["carry", "perp", "funding", "crowding"],
            "hurdle_fields": {"oos": {"pass": True, "score": 3.8}},
            "oos": {"return": 0.05, "sharpe": 0.95, "mdd": 0.04, "trades": 18},
        },
    ]

    shortlist = select_diversified_shortlist(
        candidates,
        mode="oos",
        max_total=5,
        max_per_family=5,
        max_per_timeframe=5,
        allow_multi_asset=True,
    )

    assert {row["name"] for row in shortlist} == {
        "leadlag_spillover_15m_0.35_lag3",
        "last_day_liquidity_regime_1h_balanced",
        "perp_crowding_carry_30m_0.35_0.10",
    }


def test_shortlist_suppresses_same_lineage_param_clones_by_default():
    candidates = [
        {
            "name": "composite_trend_30m_a",
            "strategy_class": "CompositeTrendStrategy",
            "family": "trend",
            "strategy_timeframe": "30m",
            "symbols": ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT"],
            "hurdle_fields": {"oos": {"pass": True, "score": 10.0}},
            "oos": {"return": 0.10, "sharpe": 2.0, "mdd": 0.04, "trades": 32},
            "params": {"long_threshold": 0.75},
        },
        {
            "name": "composite_trend_30m_b",
            "strategy_class": "CompositeTrendStrategy",
            "family": "trend",
            "strategy_timeframe": "30m",
            "symbols": ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT"],
            "hurdle_fields": {"oos": {"pass": True, "score": 9.5}},
            "oos": {"return": 0.095, "sharpe": 1.9, "mdd": 0.04, "trades": 30},
            "params": {"long_threshold": 0.60},
        },
        {
            "name": "leadlag_spillover_15m_0.35_lag3",
            "strategy_class": "LeadLagSpilloverStrategy",
            "family": "intraday_alpha",
            "strategy_timeframe": "15m",
            "symbols": ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "TRX/USDT"],
            "tags": ["leadlag", "cross-asset", "intraday", "alpha"],
            "hurdle_fields": {"oos": {"pass": True, "score": 4.4}},
            "oos": {"return": 0.06, "sharpe": 1.0, "mdd": 0.05, "trades": 25},
        },
    ]

    shortlist = select_diversified_shortlist(
        candidates,
        mode="oos",
        max_total=5,
        max_per_family=5,
        max_per_timeframe=5,
        allow_multi_asset=True,
    )

    assert {row["name"] for row in shortlist} == {
        "composite_trend_30m_a",
        "leadlag_spillover_15m_0.35_lag3",
    }


def test_build_single_asset_portfolio_sets_from_shortlist():
    shortlist = [
        {
            "name": "rsi_btc",
            "strategy_timeframe": "1m",
            "symbols": ["BTC/USDT"],
            "shortlist_score": 5.0,
            "oos": {"return": 0.1, "sharpe": 1.2},
        },
        {
            "name": "ma_btc_alt",
            "strategy_timeframe": "1m",
            "symbols": ["BTC/USDT"],
            "shortlist_score": 4.5,
            "oos": {"return": 0.08, "sharpe": 1.0},
        },
        {
            "name": "rsi_eth",
            "strategy_timeframe": "1m",
            "symbols": ["ETH/USDT"],
            "shortlist_score": 4.8,
            "oos": {"return": 0.09, "sharpe": 1.1},
        },
        {
            "name": "pair_btc_eth",
            "strategy_timeframe": "1m",
            "symbols": ["BTC/USDT", "ETH/USDT"],
            "shortlist_score": 6.0,
            "oos": {"return": 0.12, "sharpe": 1.5},
        },
    ]
    sets = build_single_asset_portfolio_sets(shortlist, mode="oos", max_per_asset=2, max_sets=4)
    assert len(sets) >= 1
    top = sets[0]
    assert top["member_count"] == 2
    assert abs(sum(float(row.get("portfolio_weight", 0.0)) for row in top["members"]) - 1.0) < 1e-9


def test_strategy_library_rows_include_tags_and_metadata():
    rows = build_binance_futures_candidates(
        timeframes=["5m", "1h", "4h"],
        symbols=["BTC/USDT", "ETH/USDT", "SOL/USDT", "XAU/USDT", "TRX/USDT"],
    )
    assert rows
    row_dicts = [row.to_dict() for row in rows]
    assert all(isinstance(item["tags"], list) for item in row_dicts)
    assert all(isinstance(item["metadata"], dict) for item in row_dicts)
    assert any("trend-following" in row["tags"] for row in row_dicts)
    assert any("vol_compression" in row["tags"] for row in row_dicts)
    assert any("pair" in row["tags"] for row in row_dicts)
    assert any("leadlag" in row["tags"] for row in row_dicts)
    carry_rows = [row for row in row_dicts if row["family"] == "carry"]
    assert carry_rows
    assert all(row["metadata"].get("data_dependent") is True for row in carry_rows)
    admission_tag_rows = [
        row for row in row_dicts if row["family"] == "cross_sectional" and "carry" in row["tags"]
    ]
    assert all(
        {"cross_sectional", "carry", "momentum"}.issubset(row["tags"]) for row in admission_tag_rows
    )
    assert any(item["metadata"].get("timeframe") is not None for item in row_dicts)
