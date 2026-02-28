"""Strict schema for cost-aware framework experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from lumina_quant.market_data import normalize_timeframe_token


@dataclass(slots=True)
class UniverseConfig:
    assets: list[str]
    filters: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class StrategyConfig:
    name: str
    enabled: bool = True
    signal_params: dict[str, Any] = field(default_factory=dict)
    rebalance_rule: dict[str, Any] = field(default_factory=dict)
    risk_model: dict[str, Any] = field(default_factory=dict)
    portfolio_construction: dict[str, Any] = field(default_factory=dict)
    execution_model: dict[str, Any] = field(default_factory=dict)
    cost_model: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class CostGlobalConfig:
    fees_bps: float = 0.0
    tax_bps: float = 0.0


@dataclass(slots=True)
class RunControlsConfig:
    start_date: str
    end_date: str
    seed: int = 42
    parallelism: int = 1
    exchange: str = "binance"
    market_data_root: str = "data/market_parquet"
    use_synthetic_data: bool = True
    synthetic_bars: int = 480
    initial_capital: float = 100000.0
    ranking_objective: str = "composite"
    ranking_weights: dict[str, float] = field(
        default_factory=lambda: {
            "sharpe": 1.0,
            "total_return": 0.25,
            "drawdown": 0.5,
            "stability": 0.25,
        }
    )


@dataclass(slots=True)
class ExperimentConfig:
    experiment_id: str
    universe: UniverseConfig
    timeframes: list[str]
    strategies: list[StrategyConfig]
    cost_global: CostGlobalConfig
    run_controls: RunControlsConfig

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExperimentConfig:
        required = (
            "experiment_id",
            "universe",
            "timeframes",
            "strategies",
            "cost_global",
            "run_controls",
        )
        missing = [key for key in required if key not in data]
        if missing:
            raise ValueError(f"Experiment config missing required keys: {missing}")

        universe_raw = data.get("universe") or {}
        assets = list(universe_raw.get("assets") or [])
        if not assets:
            raise ValueError("universe.assets must contain at least one asset")

        raw_timeframes = list(data.get("timeframes") or [])
        if not raw_timeframes:
            raise ValueError("timeframes must contain at least one timeframe")
        timeframes = [normalize_timeframe_token(str(token)) for token in raw_timeframes]

        strategy_items = list(data.get("strategies") or [])
        if not strategy_items:
            raise ValueError("strategies must contain at least one strategy block")

        strategies: list[StrategyConfig] = []
        for item in strategy_items:
            if not isinstance(item, dict):
                raise ValueError("Each strategy block must be a mapping")
            name = str(item.get("name") or "").strip()
            if not name:
                raise ValueError("Each strategy block requires a non-empty name")
            strategies.append(
                StrategyConfig(
                    name=name,
                    enabled=bool(item.get("enabled", True)),
                    signal_params=dict(item.get("signal_params") or {}),
                    rebalance_rule=dict(item.get("rebalance_rule") or {}),
                    risk_model=dict(item.get("risk_model") or {}),
                    portfolio_construction=dict(item.get("portfolio_construction") or {}),
                    execution_model=dict(item.get("execution_model") or {}),
                    cost_model=dict(item.get("cost_model") or {}),
                )
            )

        cost_global_raw = dict(data.get("cost_global") or {})
        run_controls_raw = dict(data.get("run_controls") or {})

        run_required = ("start_date", "end_date")
        run_missing = [key for key in run_required if not str(run_controls_raw.get(key) or "").strip()]
        if run_missing:
            raise ValueError(f"run_controls missing required keys: {run_missing}")

        ranking_objective = str(run_controls_raw.get("ranking_objective", "composite")).strip().lower()
        if ranking_objective not in {"composite", "sharpe", "total_return", "drawdown"}:
            ranking_objective = "composite"
        ranking_weights_raw = run_controls_raw.get("ranking_weights") or {}
        default_weights = {
            "sharpe": 1.0,
            "total_return": 0.25,
            "drawdown": 0.5,
            "stability": 0.25,
        }
        ranking_weights = dict(default_weights)
        if isinstance(ranking_weights_raw, dict):
            for key in tuple(default_weights.keys()):
                if key in ranking_weights_raw:
                    ranking_weights[key] = float(ranking_weights_raw.get(key, default_weights[key]))

        return cls(
            experiment_id=str(data.get("experiment_id")),
            universe=UniverseConfig(
                assets=assets,
                filters=dict(universe_raw.get("filters") or {}),
            ),
            timeframes=timeframes,
            strategies=strategies,
            cost_global=CostGlobalConfig(
                fees_bps=float(cost_global_raw.get("fees_bps", 0.0)),
                tax_bps=float(cost_global_raw.get("tax_bps", 0.0)),
            ),
            run_controls=RunControlsConfig(
                start_date=str(run_controls_raw.get("start_date")),
                end_date=str(run_controls_raw.get("end_date")),
                seed=int(run_controls_raw.get("seed", 42)),
                parallelism=max(1, int(run_controls_raw.get("parallelism", 1))),
                exchange=str(run_controls_raw.get("exchange", "binance")),
                market_data_root=str(run_controls_raw.get("market_data_root", "data/market_parquet")),
                use_synthetic_data=bool(run_controls_raw.get("use_synthetic_data", True)),
                synthetic_bars=max(32, int(run_controls_raw.get("synthetic_bars", 480))),
                initial_capital=float(run_controls_raw.get("initial_capital", 100000.0)),
                ranking_objective=ranking_objective,
                ranking_weights=ranking_weights,
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable resolved dictionary."""
        return asdict(self)
