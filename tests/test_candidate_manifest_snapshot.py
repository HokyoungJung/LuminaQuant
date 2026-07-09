from __future__ import annotations

import hashlib
import json
from unittest.mock import patch


from lumina_quant.strategy_factory import build_candidate_manifest

# Fixed, minimal input set: two symbols (the builder requires >=2 instruments
# so pair/cross-sectional families can materialize) over two timeframes.
FIXED_SYMBOLS = ("BTCUSDT", "ETHUSDT")
FIXED_TIMEFRAMES = ("1h", "4h")

# Byte-pin of the deterministic manifest body (excluding the wall-clock
# ``generated_at`` field), serialized with sorted keys + compact separators so
# the digest is robust to dict-ordering. Captured from current behavior; if the
# rider-builder consolidation changes any candidate, this digest must be
# re-captured intentionally. Last re-pinned for the alpha-pool-expansion-v2
# batch (2026-07-09): only MR2 (RegimeAdaptiveDisagreementEnsembleStrategy, a
# per-symbol >=30m sleeve) materializes in this 2-symbol/1h-4h fixture; the five
# leaves (A1/L1/L2/A2/N4) are 1d and/or cross-sectional >=5-symbol baskets, so
# they do not appear here (exercised instead by the broad-universe tier-guard).
EXPECTED_MANIFEST_SHA256 = "67adabba9b7b89cd429d63b11822ee37ea91604de6d2ce1f33c6d784117015dc"

EXPECTED_CANDIDATE_COUNT = 209

EXPECTED_FAMILY_COUNTS = {
    "breakout": 20,
    "cross_sectional": 2,
    "formulaic_alpha": 6,
    "market_neutral": 38,
    "mean_reversion": 28,
    "momentum": 16,
    "overlay": 8,
    "seasonality": 8,
    "time_series_momentum": 20,
    "trend": 63,
}

EXPECTED_STRATEGY_COUNTS = {
    "AccelerationRiderStrategy": 4,
    "AdaptiveTrendRiderStrategy": 4,
    "AdfGatedReversionRiderStrategy": 4,
    "Alpha101FormulaStrategy": 6,
    "AmihudIlliquidityMomentumRiderStrategy": 4,
    "AvgCorrelationCrashGuardOverlayStrategy": 4,
    "CompositeTrendStrategy": 3,
    "ConfidenceGatedTrendStrategy": 4,
    "CusumChangePointTrendRiderStrategy": 4,
    "DisagreementGatedEnsembleStrategy": 4,
    "GarchInnovationRiderStrategy": 4,
    "IntradaySeasonalMomentumRiderStrategy": 4,
    "KalmanTrendRiderStrategy": 4,
    "LastDayLiquidityRegimeStrategy": 2,
    "LowTurnoverTrendPersistenceStrategy": 4,
    "MomentumCrashDynamicScalingOverlayStrategy": 4,
    "MultiTimeframeTrendEnsembleStrategy": 4,
    "OpeningRangeBreakoutRiderStrategy": 4,
    "OvernightSessionReturnRiderStrategy": 4,
    "PairSpreadZScoreStrategy": 38,
    "PermutationEntropyTrendRiderStrategy": 4,
    "PriceVolumeCorrContinuationStrategy": 8,
    "PullbackTrendContinuationStrategy": 4,
    "RealizedSemivarianceTrendRiderStrategy": 4,
    "RealizedVolTermStructureStrategy": 4,
    "RegimeAdaptiveDisagreementEnsembleStrategy": 4,
    "RegimeBreakoutCandidateStrategy": 2,
    "RollingBreakoutStrategy": 2,
    "RoundNumberBarrierStrategy": 8,
    "SeasonalMicroBreakoutRiderStrategy": 4,
    "SilentVolumeShockResolutionStrategy": 8,
    "SpectralCycleRiderStrategy": 4,
    "SpreadStressLiquidityReversionStrategy": 8,
    "TailIndexRegimeRiderStrategy": 4,
    "VWAPCompressionReversionStrategy": 4,
    "VarianceRatioTrendRiderStrategy": 4,
    "VolOfVolRegimeTrendGateStrategy": 4,
    "VolatilityBreakoutRiderStrategy": 4,
    "VolatilitySqueezeBreakoutRiderStrategy": 4,
    "VolumeClockMomentumRiderStrategy": 4,
    "VpinToxicityRiderStrategy": 4,
}

EXPECTED_TIMEFRAME_COUNTS = {"1h": 111, "4h": 98}

# Every rider-builder produced strategy class. The upcoming consolidation must
# preserve this exact roster (names and per-class candidate counts).
EXPECTED_RIDER_CLASSES = {
    "AccelerationRiderStrategy",
    "AdaptiveTrendRiderStrategy",
    "AdfGatedReversionRiderStrategy",
    "AmihudIlliquidityMomentumRiderStrategy",
    "CusumChangePointTrendRiderStrategy",
    "GarchInnovationRiderStrategy",
    "IntradaySeasonalMomentumRiderStrategy",
    "KalmanTrendRiderStrategy",
    "OpeningRangeBreakoutRiderStrategy",
    "OvernightSessionReturnRiderStrategy",
    "PermutationEntropyTrendRiderStrategy",
    "RealizedSemivarianceTrendRiderStrategy",
    "SeasonalMicroBreakoutRiderStrategy",
    "SpectralCycleRiderStrategy",
    "TailIndexRegimeRiderStrategy",
    "VarianceRatioTrendRiderStrategy",
    "VolatilityBreakoutRiderStrategy",
    "VolatilitySqueezeBreakoutRiderStrategy",
    "VolumeClockMomentumRiderStrategy",
    "VpinToxicityRiderStrategy",
}

EXPECTED_TAGS = {
    "adf",
    "alpha101",
    "amihud_illiquidity",
    "article_family:formulaic-alpha101-research",
    "article_family:regime-breakout-thrust",
    "article_family:regime-conditioned-composite-trend",
    "article_family:sector-dispersion-reversion",
    "article_pipeline",
    "atr",
    "breakout",
    "change_point",
    "conditional_volatility",
    "confidence_gated",
    "continuation",
    "contraction_expansion",
    "correlation_crash_guard",
    "corwin_schultz",
    "cross_sectional",
    "crypto",
    "cusum",
    "cycle",
    "daniel_moskowitz",
    "de_risk",
    "disagreement_gate",
    "ensemble",
    "episodic",
    "execution_risk",
    "extreme_value",
    "factor",
    "flow_toxicity",
    "formulaic",
    "garch",
    "good_bad_volatility",
    "half_life",
    "hill_estimator",
    "illiquidity_premium",
    "information_arrival",
    "intraday_seasonality",
    "jump_filter",
    "kalman",
    "lagged_resolution",
    "last_day_return",
    "left_digit_bias",
    "liquidity_conditioned",
    "liquidity_provision",
    "low_turnover",
    "market_neutral",
    "mean_reversion",
    "meta_gate",
    "micro_signal",
    "microstructure",
    "min_hold",
    "momentum",
    "momentum_crash",
    "multi_horizon",
    "oos-stability",
    "opening_range",
    "overlay",
    "overnight_session",
    "pair",
    "pair_state",
    "per_symbol",
    "periodogram",
    "permutation_entropy",
    "pollet_wilson",
    "predictability_regime",
    "price_volume",
    "psychological_barrier",
    "pullback",
    "pure_momentum",
    "pyramiding",
    "random_walk_rejection",
    "realized_semivariance",
    "regime",
    "regime_gate",
    "return_rider",
    "risk_overlay",
    "round_number",
    "seasonality",
    "session_anchored",
    "signed_jump",
    "single_asset",
    "spectral",
    "spread",
    "spread_stress",
    "state_space",
    "stationarity",
    "subordination",
    "support_resistance",
    "tail_risk",
    "take_profit",
    "term_structure",
    "time_deformation",
    "time_of_day",
    "time_series",
    "time_series_momentum",
    "trailing_stop",
    "trend",
    "trend-following",
    "tsmom",
    "variance_ratio",
    "vol_compression",
    "vol_of_vol",
    "volatility",
    "volatility_squeeze",
    "volume_clock",
    "volume_confirmation",
    "volume_shock",
    "vpin",
    "vwap",
    "zscore",
}


def _build() -> dict:
    with patch(
        "lumina_quant.strategy_factory.candidate_library._has_perp_support_data", return_value=False
    ):
        return build_candidate_manifest(
            timeframes=list(FIXED_TIMEFRAMES),
            symbols=list(FIXED_SYMBOLS),
        )


def _canonical_body(manifest: dict) -> str:
    body = dict(manifest)
    body.pop("generated_at", None)
    return json.dumps(body, sort_keys=True, separators=(",", ":"))


def test_manifest_top_level_shape() -> None:
    manifest = _build()
    assert set(manifest.keys()) == {
        "generated_at",
        "symbol_universe",
        "timeframes",
        "candidate_count",
        "family_counts",
        "strategy_counts",
        "timeframe_counts",
        "candidates",
    }
    # Symbols are canonicalized to slash form.
    assert manifest["symbol_universe"] == ["BTC/USDT", "ETH/USDT"]
    assert manifest["timeframes"] == ["1h", "4h"]
    assert manifest["candidate_count"] == EXPECTED_CANDIDATE_COUNT
    assert manifest["candidate_count"] == len(manifest["candidates"])


def test_manifest_counts_are_pinned() -> None:
    manifest = _build()
    assert manifest["family_counts"] == EXPECTED_FAMILY_COUNTS
    assert manifest["strategy_counts"] == EXPECTED_STRATEGY_COUNTS
    assert manifest["timeframe_counts"] == EXPECTED_TIMEFRAME_COUNTS
    # Counts must be internally consistent with the candidate list.
    assert sum(EXPECTED_FAMILY_COUNTS.values()) == EXPECTED_CANDIDATE_COUNT
    assert sum(EXPECTED_STRATEGY_COUNTS.values()) == EXPECTED_CANDIDATE_COUNT
    assert sum(EXPECTED_TIMEFRAME_COUNTS.values()) == EXPECTED_CANDIDATE_COUNT


def test_rider_builder_roster_is_pinned() -> None:
    manifest = _build()
    rider_classes = {
        c["strategy_class"] for c in manifest["candidates"] if "Rider" in c["strategy_class"]
    }
    assert rider_classes == EXPECTED_RIDER_CLASSES
    # Each rider builder contributes exactly 4 candidates (2 symbols x 2 tf).
    for klass in EXPECTED_RIDER_CLASSES:
        assert manifest["strategy_counts"][klass] == 4


def test_manifest_tag_universe_is_pinned() -> None:
    manifest = _build()
    tags = {t for c in manifest["candidates"] for t in c["tags"]}
    assert tags == EXPECTED_TAGS


def test_candidate_dict_keys_are_pinned() -> None:
    manifest = _build()
    expected_keys = {
        "candidate_id",
        "name",
        "family",
        "strategy_class",
        "strategy",
        "strategy_timeframe",
        "timeframe",
        "symbols",
        "params",
        "notes",
        "tags",
        "metadata",
    }
    for candidate in manifest["candidates"]:
        assert set(candidate.keys()) == expected_keys
        assert candidate["candidate_id"]
        assert candidate["strategy"] == candidate["strategy_class"]
        assert candidate["timeframe"] == candidate["strategy_timeframe"]


def test_manifest_is_deterministic_modulo_timestamp() -> None:
    first = _canonical_body(_build())
    second = _canonical_body(_build())
    assert first == second, "manifest body must be deterministic across calls"


def test_manifest_body_byte_pin() -> None:
    digest = hashlib.sha256(_canonical_body(_build()).encode("utf-8")).hexdigest()
    assert digest == EXPECTED_MANIFEST_SHA256
