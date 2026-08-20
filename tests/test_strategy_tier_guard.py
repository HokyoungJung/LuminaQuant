"""Tier-hint guard + wiring tests for the alpha-hunt meta-spine batch.

THE GUARD (hard CI gate): ``_discover_plugin_strategies`` globs every module in
``strategies/``, and an ``@register``-ed class WITHOUT a ``_STRATEGY_TIER_HINTS``
entry silently resolves to the ``live_default`` tier. Any NEW registration must
therefore land atomically with an explicit tier hint. This test enumerates the
full registry and fails on any registered class that is neither (a) explicitly
tier-hinted, (b) a legacy ``_STRATEGY_MAP`` entry, nor (c) in the FROZEN legacy
snapshot below.

``_LEGACY_UNHINTED_LIVE_DEFAULT`` is a generated, frozen snapshot of the 68
pre-existing un-hinted classes at freeze time (2026-07-03). It is APPEND-ONLY
FOR LEGACY documentation purposes and must NEVER receive new names: a new
strategy belongs in ``_STRATEGY_TIER_HINTS`` (research_only until promoted).
Adding a new class here instead of hinting it defeats the live-safety gate.

Also covers the batch wiring: candidates exist with the right shape, tiers
resolve research_only, classes resolve via the registry map, and the
cross-sectional flow-share candidate follows the HONEST admission route
(excluded from the default shortlist; admitted only with allow_multi_asset).
"""

from __future__ import annotations

from lumina_quant.strategies.registry import (
    _STRATEGY_MAP,
    _STRATEGY_TIER_HINTS,
    get_strategy_map,
    get_strategy_names,
    get_strategy_tier,
)
from lumina_quant.strategy_factory import build_binance_futures_candidates
from lumina_quant.strategy_factory.selection import (
    candidate_mix_type,
    select_diversified_shortlist,
)

# Alpha-hunt meta-spine batch (2026-07-03): every class registered by this batch
# MUST carry an explicit research_only hint (asserted below).
BATCH_STRATEGIES = (
    "DisagreementGatedEnsembleStrategy",
    "CrossSectionalFlowShareRotationStrategy",
    "RegimeRouterConfirmedRotationStrategy",
    # Low-correlation batch 2 (2026-07-03):
    "VpinToxicityRiderStrategy",
    "TailIndexRegimeRiderStrategy",
    "VolumeClockMomentumRiderStrategy",
    # Alpha-pool-expansion-v2 batch (2026-07-09):
    "CrossSectionalNearHighAnchoringStrategy",
    "LowTurnoverTrendPersistenceStrategy",
    "RebalancingPremiumHarvestStrategy",
    "SlowCrossSectionalLeadLagStrategy",
    "StationarityGatedResidualReversionStrategy",
    "RegimeAdaptiveDisagreementEnsembleStrategy",
    # Alpha-pool-expansion-v2b batch (2026-07-09):
    "CrossSectionalCapitalGainsOverhangStrategy",
    "CrossSectionalSeasonalPersistenceStrategy",
    "MomentumCrashDynamicScalingOverlayStrategy",
    "AvgCorrelationCrashGuardOverlayStrategy",
    "SpreadStressLiquidityReversionStrategy",
    "LongRunOverreactionReversalStrategy",
    "CrossSectionalDownsideBetaAsymmetryStrategy",
    "SystematicCoskewnessPremiumStrategy",
    "TrendGatedResidualMomentumStrategy",
    "PriceVolumeCorrContinuationStrategy",
    "CrossSectionalCloseLocationAccumulationStrategy",
    "DownsideTailRiskPremiumStrategy",
    "CrossSectionalRegressionTrendQualityStrategy",
    "CrossSectionalPathConvexityStrategy",
    # Alpha-pool-expansion-v2c batch (2026-07-09):
    "CrossSectionalNearLowRecoveryStrategy",
    "CrossSectionalTimeUnderWaterStrategy",
    "CrossSectionalPriceDelayPremiumStrategy",
    "InformationDiscretenessMomentumStrategy",
    "CrossSectionalIntermediateEchoMomentumStrategy",
    "IdiosyncraticSkewInnovationStrategy",
    "SilentVolumeShockResolutionStrategy",
    "RoundNumberBarrierStrategy",
    "CrossSectionalOffSessionTugOfWarStrategy",
)

# FROZEN legacy snapshot (generated 2026-07-03; 68 names). Append-only for
# legacy documentation — new strategies go in _STRATEGY_TIER_HINTS instead.
_LEGACY_UNHINTED_LIVE_DEFAULT = frozenset(
    {
        "AccelerationRiderStrategy",
        "AdaptiveTrendRiderStrategy",
        "AdfGatedReversionRiderStrategy",
        "AmihudIlliquidityMomentumRiderStrategy",
        "BenchmarkLeadLagContinuationStrategy",
        "BettingAgainstBetaStrategy",
        "BreadthRegimeTrendTimerStrategy",
        "CalendarSeasonalityOverlayStrategy",
        "CarryTrendConfluenceRiderStrategy",
        "ConfidenceGatedTrendStrategy",
        "CrossAssetDiversifiedTrendStrategy",
        "CrossSectionalEquityMomentumStrategy",
        "CrossSectionalShortTermReversalStrategy",
        "CusumChangePointTrendRiderStrategy",
        "DeepLearningForecastGateStrategy",
        "DispersionConditionedReversionStrategy",
        "DonchianAtrTrendStrategy",
        "DualMomentumDefensiveRotationStrategy",
        "DualMomentumIndexRotationStrategy",
        "EquityBenchmarkResidualReversalStrategy",
        "EquityMetalRiskRegimeRotationStrategy",
        "FalseBreakoutReversalStrategy",
        "FundingDislocationTrendCarryStrategy",
        "FundingHarvestCarryStrategy",
        "GarchInnovationRiderStrategy",
        "GoldSilverRatioMeanReversionStrategy",
        "GoldSilverRatioTrendStrategy",
        "HurstRegimeGatedStrategy",
        "IdiosyncraticVolatilityStrategy",
        "IntermarketLeadLagContinuationStrategy",
        "IntradayFlowPressureRiderStrategy",
        "IntradaySeasonalMomentumRiderStrategy",
        "KalmanTrendRiderStrategy",
        "LeveragedTrendTimingRiderStrategy",
        "LiquidationCascadeReversionStrategy",
        "LiquidityShockReversionStrategy",
        "LotterySkewnessStrategy",
        "LowVolatilityMomentumStrategy",
        "MetalEquityDivergenceReversalStrategy",
        "MetalsRelativeValueBasketStrategy",
        "MultiTimeframeTrendEnsembleStrategy",
        "NearHighMomentumStrategy",
        "OpenInterestTrendConfirmationRiderStrategy",
        "OpeningRangeBreakoutRiderStrategy",
        "OpeningRangeContinuationStrategy",
        "OrderBookImbalanceReversionStrategy",
        "OvernightSessionReturnRiderStrategy",
        "PairsSpreadMeanReversionStrategy",
        "PermutationEntropyTrendRiderStrategy",
        "PullbackTrendContinuationStrategy",
        "RealizedSemivarianceTrendRiderStrategy",
        "RealizedVolTermStructureStrategy",
        "ResidualEquityMomentumStrategy",
        "ResidualMomentumRotationStrategy",
        "SeasonalMicroBreakoutRiderStrategy",
        "SelectionGatedMomentumStrategy",
        "SelectionGatedReversionStrategy",
        "SemisLeadLagRotationStrategy",
        "SpectralCycleRiderStrategy",
        "TakerFlowImbalanceContinuationStrategy",
        "TrendEfficiencyMomentumStrategy",
        "VWAPCompressionReversionStrategy",
        "VarianceRatioTrendRiderStrategy",
        "VolManagedMomentumCrashGateStrategy",
        "VolOfVolRegimeTrendGateStrategy",
        "VolatilityBreakoutRiderStrategy",
        "VolatilitySqueezeBreakoutRiderStrategy",
        "VolatilitySqueezeBreakoutStrategy",
    }
)

_CRYPTO_UNIVERSE = [
    "BTC/USDT",
    "ETH/USDT",
    "BNB/USDT",
    "SOL/USDT",
    "TRX/USDT",
    "ADA/USDT",
    "XRP/USDT",
]

_ALL_TIMEFRAMES = ["1s", "1m", "5m", "15m", "30m", "1h", "4h", "1d"]


def _candidates() -> list:
    return build_binance_futures_candidates(timeframes=_ALL_TIMEFRAMES, symbols=_CRYPTO_UNIVERSE)


# --------------------------------------------------------------------------- #
# THE GUARD (hard gate)
# --------------------------------------------------------------------------- #
def test_no_unhinted_registered_strategy() -> None:
    """Every registered class must be hinted, legacy-mapped, or frozen-legacy.

    Membership assertion by design: get_strategy_tier() returns "live_default"
    for BOTH an explicit live_default hint and a silently-defaulted one, so the
    tier VALUE cannot detect a forgotten hint — membership can.
    """
    offenders = [
        name
        for name in get_strategy_names()
        if name not in _STRATEGY_TIER_HINTS
        and name not in _STRATEGY_MAP
        and name not in _LEGACY_UNHINTED_LIVE_DEFAULT
    ]
    assert not offenders, (
        "New @register-ed strategy classes without an explicit _STRATEGY_TIER_HINTS "
        f"entry (they silently become live_default): {offenders}. Add a "
        "research_only hint in the SAME commit as the registration — do NOT "
        "append to the frozen legacy set."
    )


def test_batch_classes_are_explicitly_research_only() -> None:
    for cls in BATCH_STRATEGIES:
        assert cls in _STRATEGY_TIER_HINTS, cls
        assert _STRATEGY_TIER_HINTS[cls] == "research_only", cls
        assert get_strategy_tier(cls) == "research_only", cls
        assert cls not in _LEGACY_UNHINTED_LIVE_DEFAULT, cls


def test_legacy_snapshot_does_not_grow_via_hint_removal() -> None:
    # The frozen names must all still be registered (renames/deletions should
    # prune this set intentionally, not silently).
    names = set(get_strategy_names())
    missing = sorted(_LEGACY_UNHINTED_LIVE_DEFAULT - names)
    assert not missing, f"frozen legacy names no longer registered: {missing}"


# --------------------------------------------------------------------------- #
# H1 fail-safe: unknown-name tier fallback resolves research_only
# --------------------------------------------------------------------------- #
def test_unknown_name_fails_safe_to_research_only() -> None:
    """A never-registered / unhinted name must NOT resolve to a live tier.

    The CI membership guard above blocks unhinted registrations at merge time;
    this asserts the runtime fallback for the path CI cannot see (a module
    dropped onto a box after CI ran) fails SAFE instead of auto-promoting.
    """
    assert get_strategy_tier("SomeUnknownFutureStrategy") == "research_only"


def test_registry_legacy_snapshot_matches_guard_copy() -> None:
    from lumina_quant.strategies.registry import (
        _LEGACY_UNHINTED_LIVE_DEFAULT as registry_legacy,
    )

    assert registry_legacy == _LEGACY_UNHINTED_LIVE_DEFAULT


def test_frozen_legacy_names_stay_live_default() -> None:
    # The H1 fail-safe flip must not change the tier of any pre-contract class.
    for name in sorted(_LEGACY_UNHINTED_LIVE_DEFAULT):
        assert get_strategy_tier(name) == "live_default", name


def test_curated_map_unhinted_names_stay_live_default() -> None:
    for name in sorted(set(_STRATEGY_MAP) - set(_STRATEGY_TIER_HINTS)):
        assert get_strategy_tier(name) == "live_default", name


# --------------------------------------------------------------------------- #
# Batch wiring
# --------------------------------------------------------------------------- #
def test_batch_classes_resolve_via_registry_map() -> None:
    strategy_map = get_strategy_map()
    for cls in BATCH_STRATEGIES:
        assert cls in strategy_map, cls


def test_disagreement_ensemble_candidates_single_asset_ge_30m() -> None:
    rows = [c for c in _candidates() if c.strategy_class == "DisagreementGatedEnsembleStrategy"]
    assert rows, "expected disagreement-ensemble candidates"
    assert {c.timeframe for c in rows} <= {"30m", "1h", "4h", "1d"}
    for c in rows:
        assert candidate_mix_type(c.to_dict()) == "single", c.name
        assert len(c.symbols) == 1, c.symbols
        assert int((c.metadata or {}).get("decision_cadence_seconds", 0)) >= 1800


def test_basket_candidates_cross_sectional_ge_30m() -> None:
    for cls in (
        "CrossSectionalFlowShareRotationStrategy",
        "RegimeRouterConfirmedRotationStrategy",
    ):
        rows = [c for c in _candidates() if c.strategy_class == cls]
        assert rows, cls
        assert {c.timeframe for c in rows} <= {"30m", "1h", "4h", "1d"}, cls
        for c in rows:
            assert c.family == "cross_sectional", c.name
            assert len(c.symbols) == len(_CRYPTO_UNIVERSE), c.name
            assert int((c.metadata or {}).get("decision_cadence_seconds", 0)) >= 1800


def test_flow_share_admission_route_is_honest() -> None:
    """N1 carries NO carry tag: default shortlist excludes it; the data-PC
    handoff admits it explicitly with allow_multi_asset=True."""
    candidates = _candidates()
    flow_rows = [
        c for c in candidates if c.strategy_class == "CrossSectionalFlowShareRotationStrategy"
    ]
    assert flow_rows
    for c in flow_rows:
        assert "carry" not in set(c.tags), c.name

    # The shortlist selector consumes evaluation-row dicts (hurdle/oos fields
    # default to sentinels when absent), so feed candidate dicts.
    rows = [c.to_dict() for c in candidates]
    default_shortlist = select_diversified_shortlist(rows, max_total=len(rows))
    default_classes = {row["strategy_class"] for row in default_shortlist}
    assert "CrossSectionalFlowShareRotationStrategy" not in default_classes

    # Lift the diversification caps: this asserts the ADMISSION GATE passes,
    # not that the candidate out-ranks other (equally metric-less) baskets.
    open_shortlist = select_diversified_shortlist(
        rows,
        allow_multi_asset=True,
        max_total=len(rows),
        max_per_family=len(rows),
        max_per_timeframe=len(rows),
        max_per_lineage=len(rows),
        max_per_symbol_basket=len(rows),
        max_per_family_basket=len(rows),
    )
    open_classes = {row["strategy_class"] for row in open_shortlist}
    assert "CrossSectionalFlowShareRotationStrategy" in open_classes, (
        "flow-share rotation candidate must be admissible under "
        "allow_multi_asset=True (the plan's handoff route)"
    )
