from __future__ import annotations

import ast
import json
import queue
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any

import pytest

from lumina_quant.research import alpha_max_evidence as evidence
from lumina_quant.research.alpha_max_engine_runner import (
    AlphaMaxRuntimeContractError,
    _drain_indicator_events,
)
from lumina_quant.strategies.alpha_max_research_sleeves import (
    ResearchOnlyDailyCrossSectionalNearHighAnchoringStrategy,
    ResearchOnlyDailyLowTurnoverTrendPersistenceStrategy,
    ResearchOnlyFourHourFundingHarvestCarryStrategy,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs/research/alpha_max_portfolio_20260711_listing_aware.json"


class _Bars:
    def __init__(self, symbols: tuple[str, ...]) -> None:
        self.symbol_list = list(symbols)


class _Queue:
    def __init__(self) -> None:
        self.items: list[Any] = []

    def put(self, item: Any) -> None:
        self.items.append(item)


class _Aggregator:
    def __init__(self, bars: list[tuple[Any, ...]]) -> None:
        self.bars = list(bars)

    def get_bars(
        self,
        symbol: str,
        timeframe: str,
        lookback_bars: int = 1,
        *,
        n: int | None = None,
    ) -> list[tuple[Any, ...]]:
        del symbol, timeframe
        count = lookback_bars if n is None else n
        return self.bars[-int(count) :]


def _day(day: int) -> datetime:
    return datetime(2026, 1, day, tzinfo=UTC)


def _bar(day: int, close: float) -> tuple[Any, ...]:
    return (_day(day), close, close + 1.0, close - 1.0, close, 1_000.0)


WARMUP_CASES: tuple[tuple[type[Any], tuple[str, ...], int], ...] = (
    (
        ResearchOnlyDailyLowTurnoverTrendPersistenceStrategy,
        ("BTCUSDT",),
        366,
    ),
    (
        ResearchOnlyDailyCrossSectionalNearHighAnchoringStrategy,
        ("ADAUSDT", "AVAXUSDT", "BNBUSDT", "BTCUSDT", "DOGEUSDT"),
        366,
    ),
    (
        ResearchOnlyFourHourFundingHarvestCarryStrategy,
        ("BTCUSDT",),
        64,
    ),
)


@pytest.mark.parametrize(("strategy_cls", "symbols", "minimum"), WARMUP_CASES)
def test_w01_exact_completed_native_history_is_required(
    strategy_cls: type[Any],
    symbols: tuple[str, ...],
    minimum: int,
) -> None:
    strategy = strategy_cls(_Bars(symbols), _Queue())
    assert strategy.minimum_completed_bars == minimum
    strategy._alpha_max_completed_native_count_by_symbol = dict.fromkeys(symbols, minimum - 1)

    with pytest.raises(ValueError, match="insufficient_research_warmup_history"):
        strategy.validate_research_warmup_ready()

    strategy._alpha_max_completed_native_count_by_symbol = dict.fromkeys(symbols, minimum)
    assert strategy.validate_research_warmup_ready() is None


@pytest.mark.parametrize(("strategy_cls", "symbols", "_minimum"), WARMUP_CASES)
def test_w04_w06_capsules_reset_economic_state_and_restore_only_indicators(
    strategy_cls: type[Any],
    symbols: tuple[str, ...],
    _minimum: int,
) -> None:
    source_events = _Queue()
    source = strategy_cls(_Bars(symbols), source_events)
    for state in source._state.values():
        for name, value in {
            "mode": "LONG",
            "entry_price": 123.0,
            "stop_price": 99.0,
            "high_watermark": 140.0,
            "low_watermark": 80.0,
            "last_add_price": 120.0,
            "adds": 2,
            "bars_held": 9,
            "bars_since_exit": 0,
            "score": 7.0,
        }.items():
            if hasattr(state, name):
                setattr(state, name, value)

    capsule = source.get_research_indicator_state()
    for payload in capsule["symbol_state"].values():
        assert payload["mode"] == "OUT"
        assert payload["entry_price"] is None
        assert payload.get("bars_held", 0) == 0
        assert payload.get("score") is None
        if "stop_price" in payload:
            assert payload["stop_price"] is None
            assert payload["adds"] == 0
            assert payload["last_add_price"] is None
            assert payload["high_watermark"] is None
            assert payload["low_watermark"] is None

    serialized = json.dumps(capsule, sort_keys=True, default=str)
    for forbidden in ('"cash"', '"margin"', '"orders"', '"positions"'):
        assert forbidden not in serialized

    restored_events = _Queue()
    restored = strategy_cls(_Bars(symbols), restored_events)
    restored.set_research_indicator_state(capsule)

    assert restored.get_research_indicator_state() == capsule
    assert restored._alpha_max_bound_aggregator is None
    assert source_events.items == []
    assert restored_events.items == []


def test_w03_warmup_queue_discards_only_signals_and_rejects_economic_events() -> None:
    events: queue.Queue[Any] = queue.Queue()
    events.put(SimpleNamespace(type="SIGNAL"))
    events.put(SimpleNamespace(type="SIGNAL"))
    assert _drain_indicator_events(events) == 2

    events.put(SimpleNamespace(type="FILL"))
    with pytest.raises(
        AlphaMaxRuntimeContractError,
        match="alpha_max_warmup_economic_event_forbidden:FILL",
    ):
        _drain_indicator_events(events)


def test_w05_forming_future_poison_cannot_change_indicator_capsule_bytes() -> None:
    symbol = "BTCUSDT"
    first = ResearchOnlyDailyLowTurnoverTrendPersistenceStrategy(_Bars((symbol,)), _Queue())
    second = ResearchOnlyDailyLowTurnoverTrendPersistenceStrategy(_Bars((symbol,)), _Queue())
    first_aggregator = _Aggregator([_bar(1, 100.0), _bar(2, 1_000_000.0)])
    second_aggregator = _Aggregator([_bar(1, 100.0), _bar(2, -1_000_000.0)])
    event = SimpleNamespace(
        type="MARKET_WINDOW",
        time=_day(2),
        event_time_watermark_ms=int(_day(2).timestamp() * 1000),
    )

    first.calculate_signals_window(event, first_aggregator)
    second.calculate_signals_window(event, second_aggregator)

    first_bytes = json.dumps(
        first.get_research_indicator_state(),
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    second_bytes = json.dumps(
        second.get_research_indicator_state(),
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    assert first_bytes == second_bytes
    assert first.get_research_indicator_state()["symbol_state"][symbol]["closes"] == [100.0]


def _report_only_diagnostics_variant(*, poisoned: bool):
    observations = (
        (
            MappingProxyType(
                {
                    "bar_volume": 1_000.0,
                    "equity_before": 10_000.0,
                    "raw_price": 20.0,
                    "requested_qty": 100.0,
                }
            ),
        )
        if poisoned
        else ()
    )
    capacity = evidence.AlphaMaxCapacityDiagnostics(
        observation_count=len(observations),
        capacity_proxy_equity_usdt=(
            MappingProxyType(
                {
                    "minimum": 10_000.0,
                    "p10_type7": 10_000.0,
                    "median_type7": 10_000.0,
                }
            )
            if poisoned
            else None
        ),
        undefined_reason=None if poisoned else "undefined_no_positive_order",
    )
    turnover = evidence.AlphaMaxTurnoverRPTDiagnostics(
        turnover_notional=900.0 if poisoned else 0.0,
        turnover_multiple=0.09 if poisoned else 0.0,
        rpt_bps=0.0 if poisoned else None,
        undefined_reason=None if poisoned else "undefined_zero_turnover",
    )
    zero_by_symbol = MappingProxyType(dict.fromkeys(evidence.ALPHA_MAX_CANDIDATE_SYMBOLS, 0.0))
    return evidence.AlphaMaxRunReportOnlyDiagnostics(
        turnover_rpt=turnover,
        capacity=capacity,
        target_gross_exposure=1.0,
        ending_realized_gross_exposure=0.0,
        ending_realized_gross_undefined_reason=None,
        liquidity_clip_count=9 if poisoned else 0,
        reduce_only_clip_count=7 if poisoned else 0,
        no_fill_attempt_count=5 if poisoned else 0,
        capacity_observations=observations,
        capacity_observation_set_sha256=evidence._sha256_bytes(
            evidence._canonical_json_bytes(
                [dict(value) for value in observations],
                newline=True,
            )
        ),
        ending_market_value_usdt=zero_by_symbol,
        symbol_contribution_usdt=zero_by_symbol,
        contribution_total_usdt=0.0,
        fold_pnl_usdt=0.0,
        reconciliation_residual_usdt=0.0,
    )


def _gate_cell_with_diagnostics(diagnostics):
    run = object.__new__(evidence.AlphaMaxActualEngineRunReceipt)
    for field, value in {
        "report_only_diagnostics": diagnostics,
        "raw_root_set_sha256": "a" * 64,
        "feature_root_set_sha256": "b" * 64,
        "universe_sha256": "c" * 64,
    }.items():
        object.__setattr__(run, field, value)
    fold = object.__new__(evidence.AlphaMaxFoldRunEvidence)
    object.__setattr__(fold, "actual_engine_run", run)

    calendar_sha256 = "d" * 64
    stream = evidence.AlphaMaxPrimaryReturnStream(
        endpoint_timestamps=(datetime(2025, 6, 8, tzinfo=UTC),),
        endpoint_equities=(12_000.0,),
        returns=(0.2,),
        initial_capital=10_000.0,
        periods_per_year=evidence.ALPHA_MAX_PERIODS_PER_YEAR,
        calendar_sha256=calendar_sha256,
    )
    metrics = evidence.AlphaMaxMetricStatistics(
        canonical_metrics=MappingProxyType(
            {
                "total_return": 0.2,
                "cagr": 0.1,
                "sharpe": 1.1,
                "sortino": 1.2,
                "calmar": 0.5,
                "max_drawdown": 0.2,
                "volatility": 0.1,
            }
        ),
        primary_return_stream_sha256="e" * 64,
        streaming_equity_sha256="f" * 64,
        full_event_event_count=1,
        uncapped_full_event_drawdown=0.2,
        full_event_mdd=0.2,
        reporting_4h_mdd=0.2,
        gate_mdd=0.2,
        ruin_detected=False,
        drawdown_duration_endpoints=1,
        drawdown_duration_hours=4,
        value_at_risk_5pct_type7=0.2,
        expected_shortfall_5pct=0.2,
    )
    pre_gate = object.__new__(evidence.AlphaMaxCostCellPreGateEvidence)
    for field, value in {
        "row_id": "component_trend_1x",
        "domain": "validation",
        "nominal_cost_bps": 30,
        "status": "complete",
        "fold_runs": (fold,),
        "combined_primary_return_stream": stream,
        "metric_statistics": metrics,
    }.items():
        object.__setattr__(pre_gate, field, value)
    statistics = evidence.AlphaMaxStatisticalEvidence(
        candidate_ids=("component_trend_1x",),
        input_role="pre_gate_matched_selection_eligible",
        nominal_cost_bps=30,
        calendar_sha256=calendar_sha256,
        variance_across_trials=0.0,
        finite_nonannualized_sharpes=MappingProxyType({"component_trend_1x": 1.0}),
        degenerate_candidate_ids=(),
        dsr_by_candidate=MappingProxyType({"component_trend_1x": 0.95}),
        spa_pvalue_by_candidate=MappingProxyType({"component_trend_1x": 0.01}),
        pbo=0.1,
        dsr_num_trials=evidence.ALPHA_MAX_DSR_NUM_TRIALS,
        dsr_hac_inference=True,
        spa_bootstrap_rounds=2_000,
        spa_block_size=1,
        spa_seed=12_345,
        pbo_n_splits=8,
        prior_trial_key_set_sha256="1" * 64,
        current_trial_key_set_sha256="2" * 64,
    )
    return evidence.build_alpha_max_cost_cell_evidence(
        pre_gate,
        statistical_evidence=statistics,
    )


def test_u47_u48_report_only_mutations_cannot_change_gate_or_rank_bytes() -> None:
    baseline_diagnostics = _report_only_diagnostics_variant(poisoned=False)
    poisoned_diagnostics = _report_only_diagnostics_variant(poisoned=True)
    assert baseline_diagnostics.to_payload() != poisoned_diagnostics.to_payload()
    assert baseline_diagnostics.turnover_rpt != poisoned_diagnostics.turnover_rpt
    assert baseline_diagnostics.capacity != poisoned_diagnostics.capacity

    baseline = _gate_cell_with_diagnostics(baseline_diagnostics)
    poisoned = _gate_cell_with_diagnostics(poisoned_diagnostics)
    assert baseline.gate_input is not None
    assert poisoned.gate_input is not None
    assert baseline.gate_input.to_payload() == poisoned.gate_input.to_payload()

    baseline_rank = evidence.select_alpha_max_prelock_champion((baseline.gate_input,))
    poisoned_rank = evidence.select_alpha_max_prelock_champion((poisoned.gate_input,))
    assert baseline_rank.canonical_bytes == poisoned_rank.canonical_bytes


_SOURCE_PATHS = (
    "src/lumina_quant/alpha_max_process_boundary.py",
    "src/lumina_quant/research/alpha_max_engine_runner.py",
    "src/lumina_quant/backtesting/backtest.py",
    "src/lumina_quant/backtesting/data.py",
    "src/lumina_quant/backtesting/data_windowed_parquet.py",
    "src/lumina_quant/backtesting/execution_model.py",
    "src/lumina_quant/backtesting/execution_sim.py",
    "src/lumina_quant/backtesting/portfolio_backtest.py",
    "src/lumina_quant/portfolio/strategy_quality.py",
)

_EXPECTED_DANGEROUS_SITES = Counter(
    {
        (
            "alpha_max_process_boundary.py",
            "reject_ambient_lq_environment",
            "os.environ",
            None,
        ): 1,
        (
            "alpha_max_engine_runner.py",
            "_alpha_max_prelock_checkpoint_descriptor",
            "os.environ",
            None,
        ): 1,
        (
            "alpha_max_engine_runner.py",
            "_alpha_max_replay_training_component_worker",
            "os.environ",
            None,
        ): 1,
        (
            "alpha_max_engine_runner.py",
            "_alpha_max_indicator_day_checkpoint_descriptor",
            "os.environ",
            None,
        ): 2,
        (
            "backtest.py",
            "Backtest.__init__",
            "get_default_runtime_config",
            None,
        ): 1,
        (
            "backtest.py",
            "Backtest.__init__",
            "os.getenv",
            "LQ__BACKTEST__SKIP_AHEAD_ENABLED",
        ): 1,
        (
            "backtest.py",
            "Backtest._generate_trading_instances",
            "os.getenv",
            "LQ__BACKTEST__DECISION_CADENCE_SECONDS",
        ): 1,
        (
            "data.py",
            "HistoricCSVDataHandler._build_feature_lookup",
            "get_default_runtime_config",
            None,
        ): 1,
        (
            "data_windowed_parquet.py",
            "HistoricParquetWindowedDataHandler.__init__",
            "get_default_runtime_config",
            None,
        ): 1,
        ("execution_sim.py", "_env_flag", "os.getenv", "<dynamic>"): 1,
        (
            "portfolio_backtest.py",
            "Portfolio._check_circuit_breaker",
            "os.getenv",
            "LQ_BACKTEST_SUPPRESS_CIRCUIT_BREAKER_LOGS",
        ): 1,
    }
)

_EXPECTED_DYNAMIC_SITES = Counter(
    {
        (
            "execution_sim.py",
            "SimulatedExecutionHandler.__init__",
            "self._execution_flag",
            (
                "APPLY_LIQUIDITY_CAP_TO_CONDITIONAL_FILLS",
                "apply_liquidity_cap_to_conditional_fills",
            ),
        ): 1,
        (
            "execution_sim.py",
            "SimulatedExecutionHandler.check_open_orders",
            "_env_flag",
            ("LQ_BACKTEST_SUPPRESS_PARTIAL_FILL_LOGS", False),
        ): 2,
        (
            "portfolio_backtest.py",
            "Portfolio.__init__",
            "self._audit_flag",
            (
                "ENFORCE_ORDER_RISK_GATE_IN_BACKTEST",
                "risk",
                "enforce_order_risk_gate_in_backtest",
            ),
        ): 1,
        (
            "portfolio_backtest.py",
            "Portfolio.__init__",
            "self._audit_flag",
            (
                "ATTACH_DEFAULT_PROTECTIVE_STOP",
                "risk",
                "attach_default_protective_stop",
            ),
        ): 1,
        (
            "portfolio_backtest.py",
            "Portfolio.__init__",
            "self._audit_flag",
            (
                "REQUIRE_FUNDING_COVERAGE",
                "execution",
                "require_funding_coverage",
            ),
        ): 1,
        (
            "portfolio_backtest.py",
            "Portfolio.__init__",
            "self._audit_flag",
            ("ENFORCE_REDUCE_ONLY", "execution", "enforce_reduce_only"),
        ): 1,
        (
            "portfolio_backtest.py",
            "Portfolio.__init__",
            "self._audit_flag",
            ("FUNDING_ON_UTC_BOUNDARY", "execution", "funding_on_utc_boundary"),
        ): 1,
        (
            "portfolio_backtest.py",
            "Portfolio.__init__",
            "self._audit_flag",
            ("FUNDING_ENTRY_GUARD", "execution", "funding_entry_guard"),
        ): 1,
    }
)

_EXPECTED_RUNTIME_SOURCE_ATTRIBUTES = frozenset(
    {
        "ALLOW_MARKET_ORDERS",
        "ALLOW_METADATA_RISK_OVERRIDE",
        "APPLY_LIQUIDITY_CAP_TO_CONDITIONAL_FILLS",
        "ATTACH_DEFAULT_PROTECTIVE_STOP",
        "COMMISSION_RATE",
        "DECISION_CADENCE_SECONDS",
        "DEFAULT_ORDER_TYPE",
        "DEFAULT_STOP_LOSS_PCT",
        "EFFECTIVE_POSITION_FRACTION",
        "END_DATE",
        "ENFORCE_ORDER_RISK_GATE_IN_BACKTEST",
        "ENFORCE_REDUCE_ONLY",
        "FUNDING_ENTRY_GUARD",
        "FUNDING_INTERVAL_HOURS",
        "FUNDING_ON_UTC_BOUNDARY",
        "FUNDING_RATE_PER_8H",
        "INITIAL_CAPITAL",
        "LEVERAGE",
        "LIMIT_PRICE_MODE",
        "LIMIT_PRICE_OFFSET_TICKS",
        "LIMIT_TIME_IN_FORCE",
        "LIQUIDATION_BUFFER_RATE",
        "MAINTENANCE_MARGIN_RATE",
        "MAKER_FEE_RATE",
        "MARGIN_MODE",
        "MAX_DAILY_LOSS_PCT",
        "MAX_LEVERAGE",
        "MAX_ORDER_NOTIONAL_PCT",
        "MAX_ORDER_VALUE",
        "MAX_SYMBOL_EXPOSURE_PCT",
        "MIN_TRADE_QTY",
        "PERSIST_OUTPUT",
        "RANDOM_SEED",
        "REQUIRE_FUNDING_COVERAGE",
        "RISK_PER_TRADE",
        "SIM_LATENCY_MAX_BARS",
        "SIM_LATENCY_MIN_BARS",
        "SIM_MAX_BAR_VOLUME_RATIO",
        "SKIP_AHEAD_ENABLED",
        "SLIPPAGE_ADV_QUOTE",
        "SLIPPAGE_IMPACT_COEFFICIENT",
        "SLIPPAGE_IMPACT_MODEL",
        "SLIPPAGE_RATE",
        "SPREAD_RATE",
        "START_DATE",
        "STRATEGY_QUALITY_ENABLED",
        "STRATEGY_QUALITY_MIN_HOLD_BARS",
        "STRATEGY_QUALITY_NO_TRADE_BAND_BPS",
        "SYMBOL_LIMITS",
        "TAKER_FEE_RATE",
        "TARGET_ALLOCATION",
        "TARGET_ALLOCATION_MODE",
        "TIMEFRAME",
    }
)


class _RuntimeSourceInventory(ast.NodeVisitor):
    def __init__(self, filename: str) -> None:
        self.filename = filename
        self.scope: list[str] = []
        self.dangerous_sites: Counter[tuple[str, str, str, object]] = Counter()
        self.dynamic_sites: Counter[tuple[str, str, str, tuple[object, ...]]] = Counter()
        self.runtime_attributes: set[str] = set()

    @property
    def qualified_scope(self) -> str:
        return ".".join(self.scope) or "<module>"

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def _in_runtime_scope(self) -> bool:
        return not (
            self.filename == "strategy_quality.py"
            and self.qualified_scope != "StrategyQualityOverlay.__init__"
        )

    def visit_Call(self, node: ast.Call) -> None:
        name = ast.unparse(node.func)
        if name == "os.getenv":
            argument: object = "<dynamic>"
            if node.args and isinstance(node.args[0], ast.Constant):
                argument = node.args[0].value
            self.dangerous_sites[(self.filename, self.qualified_scope, "os.getenv", argument)] += 1
        if name.endswith("get_default_runtime_config"):
            self.dangerous_sites[
                (
                    self.filename,
                    self.qualified_scope,
                    "get_default_runtime_config",
                    None,
                )
            ] += 1

        dynamic_args: tuple[object, ...] | None = None
        if name.endswith("._audit_flag") or name.endswith("._execution_flag"):
            dynamic_args = tuple(
                item.value if isinstance(item, ast.Constant) else "<dynamic>"
                for item in node.args[1:]
            )
        elif name == "_env_flag":
            dynamic_args = tuple(
                item.value if isinstance(item, ast.Constant) else "<dynamic>" for item in node.args
            )
        if dynamic_args is not None:
            self.dynamic_sites[(self.filename, self.qualified_scope, name, dynamic_args)] += 1
            if (
                name != "_env_flag"
                and dynamic_args
                and isinstance(dynamic_args[0], str)
                and dynamic_args[0].isupper()
            ):
                self.runtime_attributes.add(dynamic_args[0])

        if (
            self._in_runtime_scope()
            and name == "getattr"
            and len(node.args) >= 2
            and ast.unparse(node.args[0]) in {"config", "self.config"}
            and isinstance(node.args[1], ast.Constant)
            and isinstance(node.args[1].value, str)
            and node.args[1].value.isupper()
        ):
            self.runtime_attributes.add(node.args[1].value)
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        expression = ast.unparse(node)
        if expression == "os.environ":
            self.dangerous_sites[(self.filename, self.qualified_scope, "os.environ", None)] += 1
        if (
            self._in_runtime_scope()
            and isinstance(node.ctx, ast.Load)
            and node.attr.isupper()
            and ast.unparse(node.value) in {"config", "self.config"}
        ):
            self.runtime_attributes.add(node.attr)
        self.generic_visit(node)


def test_u41_runtime_source_inventory_matches_allowlist_and_direct_sites() -> None:
    dangerous_sites: Counter[tuple[str, str, str, object]] = Counter()
    dynamic_sites: Counter[tuple[str, str, str, tuple[object, ...]]] = Counter()
    runtime_attributes: set[str] = set()
    for relative_path in _SOURCE_PATHS:
        path = REPO_ROOT / relative_path
        inventory = _RuntimeSourceInventory(path.name)
        inventory.visit(ast.parse(path.read_text(encoding="utf-8"), filename=str(path)))
        dangerous_sites.update(inventory.dangerous_sites)
        dynamic_sites.update(inventory.dynamic_sites)
        runtime_attributes.update(inventory.runtime_attributes)

    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    allowlist = frozenset(config["runtime_contract"]["attribute_allowlist"])

    assert dangerous_sites == _EXPECTED_DANGEROUS_SITES
    assert dynamic_sites == _EXPECTED_DYNAMIC_SITES
    assert frozenset(runtime_attributes) == _EXPECTED_RUNTIME_SOURCE_ATTRIBUTES
    assert allowlist >= _EXPECTED_RUNTIME_SOURCE_ATTRIBUTES
