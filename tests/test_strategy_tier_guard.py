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

import queue
import time
from types import SimpleNamespace

import pytest

from lumina_quant.core.engine import TradingEngine
from lumina_quant.core.events import OrderEvent
from lumina_quant.live.trader import LiveTrader
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
    # Alpha-sleeve batch (2026-08-20):
    "CrossSectionalResidualTakerFlowStrategy",
    "BasisFundingGapConvergenceStrategy",
    "OffSessionBasisDislocationStrategy",
    "SalienceTheoryValueStrategy",
    "ProspectTheoryValueStrategy",
    "OpenInterestGrowthPressureStrategy",
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
        "DeepLearningForecastGateStrategy",
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
        "IntermarketLeadLagContinuationStrategy",
        "IntradayFlowPressureRiderStrategy",
        "IntradaySeasonalMomentumRiderStrategy",
        "KalmanTrendRiderStrategy",
        "LeveragedTrendTimingRiderStrategy",
        "LiquidationCascadeReversionStrategy",
        "LiquidityShockReversionStrategy",
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
        "VWAPCompressionReversionStrategy",
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


def _live_order_admission_trader(*, stage="canary", risk_approved=True):
    executed = []
    trader = LiveTrader.__new__(LiveTrader)
    execution_handler = SimpleNamespace(execute_order=executed.append)
    TradingEngine.__init__(
        trader,
        events=None,
        data_handler=SimpleNamespace(get_latest_bar_value=lambda symbol, field: 100.0),
        strategy=get_strategy_map()["RsiStrategy"].__new__(get_strategy_map()["RsiStrategy"]),
        portfolio=SimpleNamespace(
            current_positions={"BTC/USDT": 1.0},
            current_position_legs={},
        ),
        execution_handler=execution_handler,
    )
    trader.config = SimpleNamespace(GO_LIVE_STAGE=stage, POSITION_MODE="ONE_WAY")
    trader.strategy_name = "RsiStrategy"
    trader.risk_manager = SimpleNamespace(
        check_order=lambda event, price, portfolio: (risk_approved, "risk_rejected")
    )
    trader.audit_store = SimpleNamespace(log_risk_event=lambda *args, **kwargs: None)
    trader.run_id = "test"
    trader.logger = SimpleNamespace(warning=lambda *args, **kwargs: None)
    trader._live_readiness_verified = True
    trader._startup_reconciliation_complete = True
    trader._startup_state = "ready"
    trader.materialized_staleness_threshold_seconds = 45
    trader._market_freshness_by_symbol = {"BTC/USDT": (time.time_ns(), 1, 0)}
    trader._materialized_stale_block_active = False
    trader._data_silence_block_active = False
    trader._hard_halt_active = False
    return trader, executed


def _order():
    return OrderEvent("BTC/USDT", "MKT", 1.0, "BUY")


def _current_bar(timestamp_ms):
    return (timestamp_ms, 100.0, 101.0, 99.0, 100.5, 1.0)


def test_live_direct_order_is_denied_before_readiness():
    trader, executed = _live_order_admission_trader()
    trader._live_readiness_verified = False

    trader.process_event(_order())

    assert executed == []


def test_authenticated_reduce_only_flatten_bypasses_only_entry_gates():
    trader, executed = _live_order_admission_trader()
    trader._live_readiness_verified = False
    trader._startup_reconciliation_complete = False
    trader._startup_state = "blocked"
    trader._market_freshness_by_symbol = {}
    trader._materialized_stale_block_active = True
    trader._data_silence_block_active = True
    trader._hard_halt_active = True
    flatten = OrderEvent(
        "BTC/USDT",
        "MKT",
        1.0,
        "SELL",
        reduce_only=True,
        metadata={
            "source": "live_trader_reduce_only_flatten",
            "flatten_quantity": 1.0,
            "flatten_position_side": "",
        },
    )

    trader.process_event(flatten)

    assert executed == [flatten]


def test_forged_reduce_only_flatten_cannot_bypass_entry_gates():
    trader, executed = _live_order_admission_trader()
    trader._hard_halt_active = True
    forged = OrderEvent(
        "BTC/USDT",
        "MKT",
        2.0,
        "SELL",
        reduce_only=True,
        metadata={
            "source": "live_trader_reduce_only_flatten",
            "flatten_quantity": 2.0,
            "flatten_position_side": "",
        },
    )

    trader.process_event(forged)

    assert executed == []


@pytest.mark.parametrize(
    "attribute,value",
    [
        ("_startup_reconciliation_complete", False),
        ("_market_freshness_by_symbol", {}),
        ("_materialized_stale_block_active", True),
    ],
)
def test_live_order_is_denied_for_incomplete_or_stale_startup_state(attribute, value):
    trader, executed = _live_order_admission_trader()
    setattr(trader, attribute, value)

    trader.process_event(_order())

    assert executed == []


@pytest.mark.parametrize("stage", ["research_only", "shadow"])
def test_non_executing_deployment_stages_are_null_order_sinks(stage):
    trader, executed = _live_order_admission_trader(stage=stage)

    trader.process_event(_order())

    assert executed == []


def test_research_only_strategy_tier_is_a_null_order_sink(monkeypatch):
    trader, executed = _live_order_admission_trader()
    research_cls = get_strategy_map()["DisagreementGatedEnsembleStrategy"]
    trader.strategy = research_cls.__new__(research_cls)
    trader.strategy_name = research_cls.__name__

    trader.process_event(_order())

    assert executed == []


def test_unknown_stage_or_strategy_tier_is_denied(monkeypatch):
    trader, executed = _live_order_admission_trader(stage="unreviewed")
    trader.process_event(_order())
    assert executed == []

    monkeypatch.setattr(
        "lumina_quant.strategies.registry.get_strategy_tier",
        lambda strategy_name: "unreviewed",
    )
    trader, executed = _live_order_admission_trader()
    trader.process_event(_order())
    assert executed == []


def test_research_instance_cannot_spoof_a_live_display_name():
    trader, executed = _live_order_admission_trader()
    research_cls = get_strategy_map()["DisagreementGatedEnsembleStrategy"]
    trader.strategy = research_cls.__new__(research_cls)
    # The user-visible/configured label must not choose this instance's tier.
    trader.strategy_name = "RsiStrategy"

    trader.process_event(_order())

    assert executed == []


def test_missing_or_other_symbol_market_evidence_cannot_admit_order():
    trader, executed = _live_order_admission_trader()
    trader._market_freshness_by_symbol = {}
    trader.process_event(_order())
    assert executed == []

    trader, executed = _live_order_admission_trader()
    trader._market_freshness_by_symbol = {"ETH/USDT": (time.time_ns(), 1, 0)}
    trader.process_event(_order())
    assert executed == []


def test_market_window_without_lag_does_not_create_admission_evidence():
    trader, _ = _live_order_admission_trader()
    trader._market_freshness_by_symbol = {}

    trader._record_market_freshness(
        SimpleNamespace(
            bars_1s={"BTC/USDT": ()},
            timestamp_ns=time.time_ns(),
            sequence=1,
            lag_ms=None,
            is_stale=False,
        )
    )

    assert trader._market_freshness_by_symbol == {}


@pytest.mark.parametrize(
    "evidence",
    [(0, 1, 0), (time.time_ns(), 0, 0), (time.time_ns(), 1, 45_001), (1, 1, 0)],
)
def test_stale_market_evidence_cannot_admit_order(evidence):
    trader, _executed = _live_order_admission_trader()
    trader._market_freshness_by_symbol = {"BTC/USDT": evidence}

    trader.process_event(_order())

    assert trader._market_freshness_by_symbol == {}


@pytest.mark.parametrize(
    "bars_1s",
    [
        {},
        {"BTC/USDT": ()},
        {"BTC/USDT": ((1_000, 100.0),)},
        {"BTC/USDT": ((1_000, 100.0, 99.0, 101.0, 100.0, 1.0),)},
    ],
)
def test_empty_or_malformed_current_bar_clears_market_admission_evidence(bars_1s):
    trader, _ = _live_order_admission_trader()
    trader._record_market_freshness(
        SimpleNamespace(
            bars_1s=bars_1s,
            time=1_000,
            event_time_watermark_ms=1_000,
            timestamp_ns=time.time_ns(),
            sequence=2,
            lag_ms=0,
            is_stale=False,
        )
    )

    assert trader._market_freshness_by_symbol == {}


def test_same_symbol_causal_bounded_market_evidence_admits_order():
    trader, executed = _live_order_admission_trader()
    trader._market_freshness_by_symbol = {}
    window_time_ms = int(time.time_ns() // 1_000_000)
    trader._record_market_freshness(
        SimpleNamespace(
            bars_1s={"BTC/USDT": (_current_bar(window_time_ms),)},
            time=window_time_ms,
            event_time_watermark_ms=window_time_ms,
            timestamp_ns=time.time_ns(),
            sequence=1,
            lag_ms=0,
            is_stale=False,
        )
    )

    trader.process_event(_order())

    assert len(executed) == 1


@pytest.mark.parametrize("stage", ["canary", "full"])
def test_canary_and_full_execute_only_after_all_admission_gates(stage):
    trader, executed = _live_order_admission_trader(stage=stage)
    trader.process_event(_order())

    assert len(executed) == 1

    trader.events = queue.Queue()
    trader.events.put(_order())
    trader.process_event(trader.events.get_nowait())

    assert len(executed) == 2

    trader, executed = _live_order_admission_trader(stage=stage, risk_approved=False)
    trader.process_event(_order())

    assert executed == []


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
