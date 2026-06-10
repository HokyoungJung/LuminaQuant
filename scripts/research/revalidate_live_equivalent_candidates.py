"""Revalidate portfolio candidates only through live-equivalent strategy/backtest paths.

This script is intentionally stricter than artifact/daily-return ranking.  A
candidate may only be selected for live promotion when the same live portfolio
mode can be instantiated by ``ArtifactPortfolioModeStrategy`` and replayed
through the event-driven backtest engine on committed raw-first market data.
Research-only return streams are retained as audit context, but are not allowed
as selection evidence.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from lumina_quant.backtesting.chunked_runner import run_backtest_chunked
from lumina_quant.backtesting.data_windowed_parquet import HistoricParquetWindowedDataHandler
from lumina_quant.backtesting.execution_sim import SimulatedExecutionHandler
from lumina_quant.backtesting.portfolio_backtest import Portfolio
from lumina_quant.configuration import BacktestConfigView, get_default_runtime_config
_lq_rt = get_default_runtime_config()
BacktestConfig = BacktestConfigView(_lq_rt)
BaseConfig = BacktestConfigView(_lq_rt)
from lumina_quant.live_selection import (
    resolve_portfolio_mode_runtime_config,
    supports_live_portfolio_mode,
)
from lumina_quant.market_data import load_data_dict_from_parquet
from lumina_quant.strategies.artifact_portfolio_mode import (
    ArtifactPortfolioModeStrategy,
    resolve_portfolio_mode_definition,
    supported_portfolio_modes,
)
from lumina_quant.symbols import canonical_symbol

REPO_ROOT = Path(__file__).resolve().parents[2]
FOLLOWUP_ROOT = REPO_ROOT / "var/reports/exact_window_backtests/followup_status"
GROUP_ROOT = FOLLOWUP_ROOT / "portfolio_incumbent_autoresearch_grouped"
FULL_UNIVERSE_PATH = (
    GROUP_ROOT / "full_universe_selection_20260426" / "full_universe_selection_latest.json"
)
OUTPUT_DIR = GROUP_ROOT / "live_equivalent_revalidation_20260426"
LIVE_DECISION_PATH = FOLLOWUP_ROOT / "portfolio_live_readiness_decision_latest.json"
DEFAULT_BACKTEST_CHECKPOINT_PATH = OUTPUT_DIR / "live_equivalent_backtest_checkpoint_20260426.json"
MDD_CAP = 0.25
ACTIVE_TRAIN_MDD_CAP = 0.20
ACTIVE_VAL_MDD_CAP = 0.12
MIN_ALPHA_TRAIN_TRADES = 20
MIN_ALPHA_VAL_TRADES = 3
MIN_ALPHA_TRAIN_TOTAL_RETURN = -0.03
MIN_ALPHA_VAL_TOTAL_RETURN = 0.0
MIN_ALPHA_VAL_SHARPE = 0.0
MIN_ALPHA_VAL_SORTINO = 0.0
MIN_USEFUL_ALPHA_OOS_TOTAL_RETURN = 0.008284
MAX_USEFUL_ALPHA_OOS_MDD = 0.001778
MIN_USEFUL_ALPHA_OOS_SHARPE = 1.0
METRIC_KEYS = (
    "total_return",
    "cagr",
    "sharpe",
    "sortino",
    "calmar",
    "max_drawdown",
    "volatility",
)
METRIC_DIAGNOSTIC_KEYS = (
    "raw_total_return",
    "raw_max_drawdown",
    "min_equity",
    "raw_final_equity",
    "equity_breach_count",
    "equity_breach_observed",
)

LIVE_MODE_RESEARCH_ALIASES = {
    "legacy_metric_no_highvol_baseline_raw_score": "legacy_no_highvol_hybrid_mode",
    "source_three_way_regime": "three_way_regime",
    "source_soft_three_way_regime": "soft_three_way_regime",
    "source_static_blend_76_24": "static_blend_76_24",
    "source_balanced_overlay_80_20": "balanced_overlay_80_20",
    "source_pair_tactical_mode": "pair_tactical_mode",
    "source_production_guarded_portfolio": "production_guarded_portfolio",
    "source_state_vwap_pair": "state_vwap_pair",
    "source_wave2_pair": "wave2_pair",
}


@dataclass(frozen=True, slots=True)
class SplitWindow:
    name: str
    start: date
    end: date
    role: str

    def as_payload(self) -> dict[str, str]:
        return {
            "name": self.name,
            "start": self.start.isoformat(),
            "end_inclusive": self.end.isoformat(),
            "role": self.role,
        }


@dataclass(slots=True)
class ModePreflight:
    mode: str
    symbols: list[str]
    cash_weight: float
    component_count: int
    component_summary: list[dict[str, Any]]
    coverage: dict[str, Any]
    status: str
    blocking_reasons: list[str]

    def as_payload(self) -> dict[str, Any]:
        return asdict(self)


def _today_utc() -> date:
    configured = str(getattr(BacktestConfig, "END_DATE", "") or "").strip()
    if configured:
        try:
            return datetime.fromisoformat(configured).date()
        except Exception:
            pass
    return datetime.now(UTC).date()


def _latest_complete_utc_day() -> date:
    configured = str(getattr(BacktestConfig, "END_DATE", "") or "").strip()
    if configured:
        try:
            return datetime.fromisoformat(configured).date()
        except Exception:
            pass
    return datetime.now(UTC).date() - timedelta(days=1)


def _split_windows(oos_end: date | None = None) -> list[SplitWindow]:
    resolved_oos_end = oos_end or _latest_complete_utc_day()
    if resolved_oos_end < date(2026, 3, 1):
        resolved_oos_end = date(2026, 3, 1)
    return [
        SplitWindow("train", date(2025, 1, 1), date(2025, 12, 31), "sanity_filter"),
        SplitWindow("val", date(2026, 1, 1), date(2026, 2, 28), "primary_selection"),
        SplitWindow("oos", date(2026, 3, 1), resolved_oos_end, "report_only"),
    ]


def _date_iter(start: date, end: date) -> list[date]:
    if end < start:
        return []
    days = int((end - start).days)
    return [start + timedelta(days=idx) for idx in range(days + 1)]


def _compact_symbol(symbol: str) -> str:
    return canonical_symbol(symbol).replace("/", "")


def _materialized_day_root(
    *,
    root: Path,
    exchange: str,
    symbol: str,
    timeframe: str,
    day: date,
) -> Path:
    return (
        root
        / "market_data_materialized"
        / str(exchange).lower()
        / _compact_symbol(symbol)
        / f"timeframe={timeframe}"
        / f"date={day.isoformat()}"
    )


def _has_committed_materialized_day(
    *,
    root: Path,
    exchange: str,
    symbol: str,
    timeframe: str,
    day: date,
) -> bool:
    day_root = _materialized_day_root(
        root=root,
        exchange=exchange,
        symbol=symbol,
        timeframe=timeframe,
        day=day,
    )
    manifest_path = day_root / "manifest.json"
    if not manifest_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if str(manifest.get("status") or "").strip().lower() != "committed":
        return False
    files = list(manifest.get("data_files") or [])
    if not files:
        return False
    return all((day_root / str(item)).exists() for item in files)


def _legacy_monthly_files(
    *,
    root: Path,
    exchange: str,
    symbol: str,
    start: date,
    end: date,
) -> list[str]:
    base = root / "market_ohlcv_1s" / str(exchange).lower() / _compact_symbol(symbol)
    if not base.exists():
        return []
    months: list[str] = []
    cursor = date(start.year, start.month, 1)
    stop = date(end.year, end.month, 1)
    while cursor <= stop:
        path = base / f"{cursor.year:04d}-{cursor.month:02d}.parquet"
        if path.exists():
            months.append(str(path))
        if cursor.month == 12:
            cursor = date(cursor.year + 1, 1, 1)
        else:
            cursor = date(cursor.year, cursor.month + 1, 1)
    return months


def _coverage_for_symbol(
    *,
    root: Path,
    exchange: str,
    symbol: str,
    timeframe: str,
    splits: list[SplitWindow],
) -> dict[str, Any]:
    split_payload: dict[str, Any] = {}
    for split in splits:
        days = _date_iter(split.start, split.end)
        present_days = [
            day.isoformat()
            for day in days
            if _has_committed_materialized_day(
                root=root,
                exchange=exchange,
                symbol=symbol,
                timeframe=timeframe,
                day=day,
            )
        ]
        legacy_files = _legacy_monthly_files(
            root=root,
            exchange=exchange,
            symbol=symbol,
            start=split.start,
            end=split.end,
        )
        split_payload[split.name] = {
            "required_day_count": len(days),
            "committed_day_count": len(present_days),
            "missing_day_count": max(0, len(days) - len(present_days)),
            "complete_raw_first": len(days) > 0 and len(present_days) == len(days),
            "first_missing_days": [
                day.isoformat() for day in days if day.isoformat() not in set(present_days)
            ][:5],
            "legacy_monthly_files_present": len(legacy_files),
            "legacy_monthly_files": legacy_files[:5],
        }
    return split_payload


def _mode_preflight(
    *,
    mode: str,
    market_root: Path,
    exchange: str,
    timeframe: str,
    splits: list[SplitWindow],
) -> ModePreflight:
    definition = resolve_portfolio_mode_definition(mode)
    components = [
        {
            "component_id": component.component_id,
            "label": component.label,
            "strategy_class": component.strategy_class,
            "symbols": list(component.symbols),
            "weight": float(component.weight),
        }
        for component in definition.components
    ]
    symbols = list(definition.symbols)
    if float(definition.cash_weight) >= 0.999 and not components:
        return ModePreflight(
            mode=mode,
            symbols=symbols,
            cash_weight=float(definition.cash_weight),
            component_count=len(components),
            component_summary=components,
            coverage={},
            status="eligible_conservative_cash_fallback",
            blocking_reasons=[],
        )

    coverage = {
        symbol: _coverage_for_symbol(
            root=market_root,
            exchange=exchange,
            symbol=symbol,
            timeframe=timeframe,
            splits=splits,
        )
        for symbol in symbols
    }
    blocking_reasons: list[str] = []
    for symbol, split_payload in coverage.items():
        for split_name in ("train", "val"):
            if not bool(dict(split_payload.get(split_name) or {}).get("complete_raw_first")):
                blocking_reasons.append(f"{symbol}:{split_name}_raw_first_incomplete")
    status = (
        "ready_for_live_equivalent_backtest"
        if not blocking_reasons
        else "blocked_missing_raw_first_market_data"
    )
    return ModePreflight(
        mode=mode,
        symbols=symbols,
        cash_weight=float(definition.cash_weight),
        component_count=len(components),
        component_summary=components,
        coverage=coverage,
        status=status,
        blocking_reasons=blocking_reasons,
    )


def _empty_metrics() -> dict[str, float]:
    return dict.fromkeys(METRIC_KEYS, 0.0)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    if not math.isfinite(out):
        return float(default)
    return float(out)


def _metrics_from_equity_totals(values: list[float], *, periods: int) -> dict[str, float]:
    equity = np.asarray([_safe_float(item, 0.0) for item in values], dtype=float)
    equity = equity[np.isfinite(equity)]
    if equity.size < 2 or equity[0] <= 0.0:
        return _empty_metrics()

    raw_total_return = float(equity[-1] / equity[0] - 1.0)
    running_max = np.maximum.accumulate(equity)
    drawdowns = np.where(running_max > 0.0, (running_max - equity) / running_max, 0.0)
    raw_max_drawdown = float(np.max(drawdowns)) if drawdowns.size else 0.0
    min_equity = float(np.min(equity)) if equity.size else 0.0
    final_equity = float(equity[-1]) if equity.size else 0.0
    breach_count = int(np.count_nonzero(equity <= 0.0))
    equity_breach_observed = bool(
        breach_count > 0
        or final_equity <= 0.0
        or raw_total_return <= -1.0
        or raw_max_drawdown > 1.0
    )
    diagnostics = {
        "raw_total_return": raw_total_return,
        "raw_max_drawdown": raw_max_drawdown,
        "min_equity": min_equity,
        "raw_final_equity": final_equity,
        "equity_breach_count": float(breach_count),
        "equity_breach_observed": bool(equity_breach_observed),
    }
    if equity_breach_observed:
        return {
            **_empty_metrics(),
            "total_return": -1.0,
            "cagr": -1.0,
            "calmar": -1.0,
            "max_drawdown": 1.0,
            **diagnostics,
        }

    returns = np.diff(equity) / equity[:-1]
    returns = returns[np.isfinite(returns)]
    total_return = raw_total_return
    if returns.size <= 0:
        return {**_empty_metrics(), "total_return": total_return, **diagnostics}
    periods_per_year = max(1, int(periods))
    mean = float(np.mean(returns))
    std = float(np.std(returns, ddof=1)) if returns.size > 1 else 0.0
    volatility = float(std * math.sqrt(periods_per_year)) if std > 0.0 else 0.0
    sharpe = float((mean / std) * math.sqrt(periods_per_year)) if std > 0.0 else 0.0
    downside = returns[returns < 0.0]
    downside_std = float(np.std(downside, ddof=1)) if downside.size > 1 else 0.0
    sortino = (
        float((mean / downside_std) * math.sqrt(periods_per_year)) if downside_std > 0.0 else 0.0
    )
    max_drawdown = min(raw_max_drawdown, 1.0)
    cagr = (
        float((1.0 + total_return) ** (periods_per_year / returns.size) - 1.0)
        if total_return > -1.0
        else -1.0
    )
    calmar = float(cagr / max_drawdown) if max_drawdown > 1e-12 else 0.0
    return {
        "total_return": total_return,
        "cagr": cagr,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "max_drawdown": max_drawdown,
        "volatility": volatility,
        **diagnostics,
    }


def _scaled_score(metrics: dict[str, Any]) -> float:
    mdd = max(0.0, _safe_float(metrics.get("max_drawdown"), 0.0))
    mdd_headroom = 1.0 - min(max(mdd, 0.0), MDD_CAP) / MDD_CAP
    return 100.0 * (
        0.30 * math.tanh(_safe_float(metrics.get("total_return"), 0.0) / 0.18)
        + 0.30 * math.tanh(_safe_float(metrics.get("sharpe"), 0.0) / 4.0)
        + 0.15 * math.tanh(_safe_float(metrics.get("sortino"), 0.0) / 12.0)
        + 0.15 * math.tanh(_safe_float(metrics.get("calmar"), 0.0) / 80.0)
        + 0.10 * mdd_headroom
    )


def _score_from_split_metrics(metrics: dict[str, dict[str, Any]]) -> dict[str, float | bool]:
    train = dict(metrics.get("train") or {})
    val = dict(metrics.get("val") or {})
    oos = dict(metrics.get("oos") or {})
    train_scaled = _scaled_score(train)
    val_scaled = _scaled_score(val)
    oos_scaled = _scaled_score(oos)
    train_val_mdd_ok = (
        _safe_float(train.get("max_drawdown"), 0.0) <= MDD_CAP
        and _safe_float(val.get("max_drawdown"), 0.0) <= MDD_CAP
    )
    return {
        "selection_score": float(val_scaled + 0.18 * train_scaled),
        "val_scaled_score": float(val_scaled),
        "train_scaled_score": float(train_scaled),
        "oos_scaled_score_report_only": float(oos_scaled),
        "train_val_mdd_ok": bool(train_val_mdd_ok),
    }


def _split_run_by_name(result: dict[str, Any], split_name: str) -> dict[str, Any]:
    for run in list(result.get("split_runs") or []):
        if not isinstance(run, dict):
            continue
        if str(run.get("split") or "") == split_name:
            return dict(run)
    return {}


def _profit_alpha_gate(
    result: dict[str, Any],
    preflight: ModePreflight,
) -> dict[str, Any]:
    """Return explicit alpha/fallback eligibility metadata.

    A live-equivalent backtest with 0 trades or non-positive validation
    performance is useful as a safety result, but it must not outrank an active
    alpha.  Conservative cash remains a fallback only.
    """
    status = str(result.get("status") or preflight.status)
    if status == "eligible_conservative_cash_fallback" or float(preflight.cash_weight) >= 0.999:
        return {
            "selection_role": "fallback",
            "selection_eligible": False,
            "fallback_eligible": True,
            "alpha_blocking_reasons": ["conservative_cash_fallback_not_alpha"],
        }

    metrics = dict(result.get("metrics") or {})
    scores = dict(result.get("scores") or {})
    train = dict(metrics.get("train") or {})
    val = dict(metrics.get("val") or {})
    oos = dict(metrics.get("oos") or {})
    train_run = _split_run_by_name(result, "train")
    val_run = _split_run_by_name(result, "val")
    oos_run = _split_run_by_name(result, "oos")

    train_trade_count = int(_safe_float(train_run.get("trade_count"), 0.0))
    val_trade_count = int(_safe_float(val_run.get("trade_count"), 0.0))
    oos_trade_count = int(_safe_float(oos_run.get("trade_count"), 0.0))
    train_liquidation_count = int(_safe_float(train_run.get("liquidation_count"), 0.0))
    val_liquidation_count = int(_safe_float(val_run.get("liquidation_count"), 0.0))
    oos_liquidation_count = int(_safe_float(oos_run.get("liquidation_count"), 0.0))

    reasons: list[str] = []
    if status != "live_equivalent_validated":
        reasons.append(f"status_not_validated:{status}")
    if not bool(scores.get("train_val_mdd_ok")):
        reasons.append("legacy_train_val_mdd_gate_failed")
    if _safe_float(train.get("max_drawdown"), 0.0) > ACTIVE_TRAIN_MDD_CAP:
        reasons.append("train_mdd_above_active_cap")
    if _safe_float(val.get("max_drawdown"), 0.0) > ACTIVE_VAL_MDD_CAP:
        reasons.append("val_mdd_above_active_cap")
    for split_name, split_metrics in (
        ("train", train),
        ("val", val),
        ("oos", oos),
    ):
        if bool(split_metrics.get("equity_breach_observed")):
            reasons.append(f"{split_name}_equity_breach_observed")
    if train_trade_count < MIN_ALPHA_TRAIN_TRADES:
        reasons.append("train_trade_count_below_min")
    if val_trade_count < MIN_ALPHA_VAL_TRADES:
        reasons.append("val_trade_count_below_min")
    if _safe_float(train.get("total_return"), 0.0) < MIN_ALPHA_TRAIN_TOTAL_RETURN:
        reasons.append("train_total_return_below_floor")
    if _safe_float(val.get("total_return"), 0.0) <= MIN_ALPHA_VAL_TOTAL_RETURN:
        reasons.append("val_total_return_not_positive")
    if _safe_float(val.get("sharpe"), 0.0) <= MIN_ALPHA_VAL_SHARPE:
        reasons.append("val_sharpe_not_positive")
    if _safe_float(val.get("sortino"), 0.0) <= MIN_ALPHA_VAL_SORTINO:
        reasons.append("val_sortino_not_positive")
    if _safe_float(oos.get("total_return"), 0.0) <= MIN_USEFUL_ALPHA_OOS_TOTAL_RETURN:
        reasons.append("oos_return_not_above_0.8284pct_incumbent")
    if _safe_float(oos.get("max_drawdown"), 0.0) >= MAX_USEFUL_ALPHA_OOS_MDD:
        reasons.append("oos_mdd_not_below_funding_guard_shadow")
    if _safe_float(oos.get("sharpe"), 0.0) <= MIN_USEFUL_ALPHA_OOS_SHARPE:
        reasons.append("oos_sharpe_not_above_1.0_success_target")
    if oos_trade_count <= 0:
        reasons.append("oos_trade_count_zero")
    if train_liquidation_count > 0 or val_liquidation_count > 0 or oos_liquidation_count > 0:
        reasons.append("liquidation_observed")

    return {
        "selection_role": "alpha",
        "selection_eligible": not reasons,
        "fallback_eligible": False,
        "alpha_blocking_reasons": reasons,
        "train_trade_count": train_trade_count,
        "val_trade_count": val_trade_count,
        "train_liquidation_count": train_liquidation_count,
        "val_liquidation_count": val_liquidation_count,
        "oos_trade_count": oos_trade_count,
        "oos_liquidation_count": oos_liquidation_count,
        "train_final_equity": _safe_float(train_run.get("final_equity"), 0.0),
        "val_final_equity": _safe_float(val_run.get("final_equity"), 0.0),
        "oos_final_equity": _safe_float(oos_run.get("final_equity"), 0.0),
    }


def _train_fail_fast_reasons(split_run: dict[str, Any]) -> list[str]:
    """Return train-only reasons that make later val/OOS replay wasteful.

    Validation is still the primary selection split, but active alpha promotion
    also requires minimum train return, train trade count, train MDD, and zero
    train liquidations.  Once a candidate violates those train-only invariants,
    additional 1s live-equivalent replay cannot make it deployable.  Full audit
    metrics remain available by leaving fail-fast disabled.
    """
    metrics = dict(split_run.get("metrics") or {})
    reasons: list[str] = []
    if str(split_run.get("status") or "completed") != "completed":
        reasons.append(f"train_status_not_completed:{split_run.get('status')}")
    if int(_safe_float(split_run.get("trade_count"), 0.0)) < MIN_ALPHA_TRAIN_TRADES:
        reasons.append("train_trade_count_below_min")
    if int(_safe_float(split_run.get("liquidation_count"), 0.0)) > 0:
        reasons.append("train_liquidation_observed")
    if _safe_float(metrics.get("total_return"), 0.0) < MIN_ALPHA_TRAIN_TOTAL_RETURN:
        reasons.append("train_total_return_below_floor")
    if _safe_float(metrics.get("max_drawdown"), 0.0) > ACTIVE_TRAIN_MDD_CAP:
        reasons.append("train_mdd_above_active_cap")
    if bool(metrics.get("equity_breach_observed")):
        reasons.append("train_equity_breach_observed")
    return reasons


def _split_to_datetimes(split: SplitWindow) -> tuple[datetime, datetime]:
    return (
        datetime.combine(split.start, time.min).replace(tzinfo=None),
        datetime.combine(split.end, time.max).replace(tzinfo=None),
    )


def _mode_equivalence_key(mode: str) -> str:
    """Stable key for live modes that replay identical component graphs."""
    definition = resolve_portfolio_mode_definition(mode)
    payload = {
        "cash_weight": round(float(definition.cash_weight), 12),
        "watch_symbols": list(definition.watch_symbols),
        "components": [
            {
                "component_id": component.component_id,
                "strategy_class": component.strategy_class,
                "symbols": list(component.symbols),
                "params": component.params,
                "weight": round(float(component.weight), 12),
            }
            for component in definition.components
        ],
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _resolve_revalidation_runtime_config(mode: str) -> dict[str, Any]:
    """Resolve runtime config for live or research-only portfolio modes.

    Some profit-moonshot shadow modes are intentionally not live-selectable yet,
    but are still defined as ArtifactPortfolioModeStrategy modes for raw-first
    engine validation.  Revalidation must be able to audit those modes without
    silently adding them to the live selector allowlist.
    """
    try:
        return dict(resolve_portfolio_mode_runtime_config(mode))
    except ValueError:
        if mode not in supported_portfolio_modes():
            raise
    definition = resolve_portfolio_mode_definition(mode)
    return {
        "portfolio_mode": mode,
        "strategy_name": f"ArtifactPortfolioModeStrategy[{mode}]",
        "strategy_params": {"portfolio_mode": mode},
        "symbols": list(definition.symbols),
        "cash_weight": float(definition.cash_weight),
        "research_only_live_equivalent": True,
    }


def _run_live_equivalent_split(
    *,
    mode: str,
    split: SplitWindow,
    market_root: Path,
    exchange: str,
    timeframe: str,
    chunk_days: int,
    backtest_poll_seconds: int,
    backtest_window_seconds: int,
    equivalence_key: str,
) -> dict[str, Any]:
    runtime_config = _resolve_revalidation_runtime_config(mode)
    symbols = list(runtime_config["symbols"])
    start_dt, end_dt = _split_to_datetimes(split)

    def _loader(chunk_start: datetime, chunk_end: datetime) -> dict[str, Any]:
        _progress_event(
            "live_equivalent_chunk_load",
            mode=mode,
            split=split.name,
            chunk_start=chunk_start.isoformat(),
            chunk_end=chunk_end.isoformat(),
            chunk_days=max(1, int(chunk_days)),
        )
        return load_data_dict_from_parquet(
            str(market_root),
            exchange=str(exchange),
            symbol_list=symbols,
            timeframe=str(timeframe),
            start_date=chunk_start,
            end_date=chunk_end,
            chunk_days=max(1, int(chunk_days)),
            warmup_bars=0,
            data_mode="raw-first",
            staleness_threshold_seconds=None,
        )

    backtest = run_backtest_chunked(
        csv_dir="data",
        symbol_list=symbols,
        start_date=start_dt,
        end_date=end_dt,
        strategy_cls=ArtifactPortfolioModeStrategy,
        strategy_params={"portfolio_mode": mode},
        data_loader=_loader,
        chunk_days=max(1, int(chunk_days)),
        strategy_timeframe=str(BaseConfig.TIMEFRAME),
        data_handler_cls=HistoricParquetWindowedDataHandler,
        execution_handler_cls=SimulatedExecutionHandler,
        portfolio_cls=Portfolio,
        backtest_mode="windowed",
        data_handler_kwargs={
            "backtest_poll_seconds": max(1, int(backtest_poll_seconds)),
            "backtest_window_seconds": max(1, int(backtest_window_seconds)),
        },
        record_history=False,
        track_metrics=True,
        record_trades=False,
    )
    metrics = _metrics_from_equity_totals(
        list(getattr(backtest.portfolio, "_metric_totals", []) or []),
        periods=int(getattr(BacktestConfig, "ANNUAL_PERIODS", 252)),
    )
    return {
        "split": split.name,
        "equivalence_key": equivalence_key,
        "metrics": metrics,
        "trade_count": int(getattr(backtest.portfolio, "trade_count", 0)),
        "final_equity": _safe_float(
            dict(getattr(backtest.portfolio, "current_holdings", {}) or {}).get("total"),
            0.0,
        ),
        "liquidation_count": len(getattr(backtest.portfolio, "liquidation_events", []) or []),
    }


def _run_mode_backtests(
    *,
    preflight: ModePreflight,
    market_root: Path,
    exchange: str,
    timeframe: str,
    splits: list[SplitWindow],
    chunk_days: int,
    backtest_poll_seconds: int,
    backtest_window_seconds: int,
    checkpoint: dict[str, Any] | None = None,
    checkpoint_path: Path | None = None,
    equivalence_cache: dict[tuple[str, str], dict[str, Any]] | None = None,
    fail_fast_alpha_gate: bool = False,
) -> dict[str, Any]:
    if preflight.status == "eligible_conservative_cash_fallback":
        metrics = {split.name: _empty_metrics() for split in splits}
        scores = _score_from_split_metrics(metrics)
        return {
            "mode": preflight.mode,
            "status": preflight.status,
            "metrics": metrics,
            "scores": scores,
            "split_runs": [],
        }
    if preflight.status != "ready_for_live_equivalent_backtest":
        return {
            "mode": preflight.mode,
            "status": preflight.status,
            "metrics": {split.name: _empty_metrics() for split in splits},
            "scores": {
                "selection_score": None,
                "val_scaled_score": None,
                "train_scaled_score": None,
                "oos_scaled_score_report_only": None,
                "train_val_mdd_ok": False,
            },
            "split_runs": [],
            "blocking_reasons": list(preflight.blocking_reasons),
        }

    equivalence_key = _mode_equivalence_key(preflight.mode)
    split_runs: list[dict[str, Any]] = []
    metrics: dict[str, dict[str, float]] = {}
    for split in splits:
        if checkpoint is not None and checkpoint_path is not None:
            cached = _checkpointed_split_run(checkpoint, mode=preflight.mode, split=split)
            if cached is not None:
                split_runs.append(cached)
                metrics[split.name] = dict(cached.get("metrics") or {})
                if equivalence_cache is not None:
                    equivalence_cache.setdefault(
                        (str(cached.get("equivalence_key") or equivalence_key), split.name),
                        dict(cached),
                    )
                _progress_event(
                    "live_equivalent_split_resume", mode=preflight.mode, split=split.name
                )
                continue
        if equivalence_cache is not None:
            source_run = equivalence_cache.get((equivalence_key, split.name))
            if source_run is not None:
                reused_run = {
                    **dict(source_run),
                    "mode": preflight.mode,
                    "status": "completed",
                    "reuse_status": "completed_equivalent_reuse",
                    "equivalent_source_mode": str(
                        source_run.get("mode") or source_run.get("source_mode") or ""
                    ),
                    "source_mode": str(
                        source_run.get("mode") or source_run.get("source_mode") or preflight.mode
                    ),
                    "equivalence_key": equivalence_key,
                }
                split_runs.append(reused_run)
                metrics[split.name] = dict(reused_run.get("metrics") or {})
                if checkpoint is not None and checkpoint_path is not None:
                    _store_checkpointed_split_run(
                        checkpoint,
                        checkpoint_path,
                        mode=preflight.mode,
                        split_run=reused_run,
                    )
                _progress_event(
                    "live_equivalent_split_reuse",
                    mode=preflight.mode,
                    split=split.name,
                    equivalent_source_mode=reused_run.get("equivalent_source_mode"),
                )
                continue
        if split.name == "oos":
            oos_complete = all(
                bool(
                    dict(dict(preflight.coverage.get(symbol) or {}).get("oos") or {}).get(
                        "complete_raw_first"
                    )
                )
                for symbol in preflight.symbols
            )
            if not oos_complete:
                metrics[split.name] = _empty_metrics()
                split_runs.append({"split": split.name, "status": "skipped_oos_data_incomplete"})
                continue
        _progress_event(
            "live_equivalent_split_start",
            mode=preflight.mode,
            split=split.name,
            start=split.start.isoformat(),
            end_inclusive=split.end.isoformat(),
            symbols=preflight.symbols,
        )
        run = _run_live_equivalent_split(
            mode=preflight.mode,
            split=split,
            market_root=market_root,
            exchange=exchange,
            timeframe=timeframe,
            chunk_days=chunk_days,
            backtest_poll_seconds=backtest_poll_seconds,
            backtest_window_seconds=backtest_window_seconds,
            equivalence_key=equivalence_key,
        )
        completed_run = {
            **run,
            "status": "completed",
            "mode": preflight.mode,
            "source_mode": preflight.mode,
        }
        split_runs.append(completed_run)
        metrics[split.name] = dict(run.get("metrics") or {})
        if equivalence_cache is not None:
            equivalence_cache[(equivalence_key, split.name)] = dict(completed_run)
        if checkpoint is not None and checkpoint_path is not None:
            _store_checkpointed_split_run(
                checkpoint,
                checkpoint_path,
                mode=preflight.mode,
                split_run=completed_run,
            )
        _progress_event(
            "live_equivalent_split_complete",
            mode=preflight.mode,
            split=split.name,
            metrics=completed_run.get("metrics", {}),
            trade_count=completed_run.get("trade_count"),
            final_equity=completed_run.get("final_equity"),
        )
        if fail_fast_alpha_gate and split.name == "train":
            train_fail_reasons = _train_fail_fast_reasons(completed_run)
            if train_fail_reasons:
                _progress_event(
                    "live_equivalent_fail_fast_skip",
                    mode=preflight.mode,
                    split=split.name,
                    reasons=train_fail_reasons,
                )
                for remaining in splits:
                    if remaining.name in metrics:
                        continue
                    metrics[remaining.name] = _empty_metrics()
                    split_runs.append(
                        {
                            "split": remaining.name,
                            "status": "skipped_train_alpha_gate_failed",
                            "skip_reasons": train_fail_reasons,
                        }
                    )
                break
    for split in splits:
        metrics.setdefault(split.name, _empty_metrics())
    scores = _score_from_split_metrics(metrics)
    skipped_by_train_gate = any(
        str(run.get("status") or "") == "skipped_train_alpha_gate_failed" for run in split_runs
    )
    if skipped_by_train_gate:
        status = "failed_train_alpha_gate"
    else:
        status = (
            "live_equivalent_validated"
            if bool(scores.get("train_val_mdd_ok"))
            else "failed_train_val_mdd_gate"
        )
    return {
        "mode": preflight.mode,
        "status": status,
        "metrics": metrics,
        "scores": scores,
        "split_runs": split_runs,
    }


def _load_full_universe(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {}
    return payload


def _map_research_candidate_to_mode(name: str) -> str | None:
    token = str(name or "").strip()
    if supports_live_portfolio_mode(token):
        return token
    if token in LIVE_MODE_RESEARCH_ALIASES:
        return LIVE_MODE_RESEARCH_ALIASES[token]
    return None


def _artifact_candidate_reset_rows(full_universe: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    candidates = list(full_universe.get("ranked_clean_candidates") or [])
    for rank, item in enumerate(candidates, start=1):
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        mapped_mode = _map_research_candidate_to_mode(name)
        rows.append(
            {
                "research_rank": rank,
                "name": name,
                "uid": str(item.get("uid") or ""),
                "kind": str(item.get("kind") or ""),
                "category": str(item.get("category") or ""),
                "previous_selection_score": _safe_float(item.get("selection_score"), 0.0),
                "previous_val_total_return": _safe_float(
                    dict(dict(item.get("metrics") or {}).get("val") or {}).get("total_return"),
                    0.0,
                ),
                "previous_val_sharpe": _safe_float(
                    dict(dict(item.get("metrics") or {}).get("val") or {}).get("sharpe"),
                    0.0,
                ),
                "mapped_live_portfolio_mode": mapped_mode or "",
                "reset_status": "requires_live_equivalent_engine_backtest"
                if mapped_mode
                else "research_only_not_live_selectable",
                "selection_eligible_after_reset": False,
            }
        )
    return rows


def _mode_candidate_rows(
    mode_results: list[dict[str, Any]], preflights: dict[str, ModePreflight]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in mode_results:
        mode = str(result.get("mode") or "")
        preflight = preflights[mode]
        scores = dict(result.get("scores") or {})
        metrics = dict(result.get("metrics") or {})
        gate = _profit_alpha_gate(result, preflight)
        row: dict[str, Any] = {
            "mode": mode,
            "status": str(result.get("status") or preflight.status),
            "selection_role": str(gate.get("selection_role") or "alpha"),
            "selection_eligible": bool(gate.get("selection_eligible")),
            "fallback_eligible": bool(gate.get("fallback_eligible")),
            "alpha_blocking_reasons": ";".join(list(gate.get("alpha_blocking_reasons") or [])[:16]),
            "selection_score": scores.get("selection_score"),
            "val_scaled_score": scores.get("val_scaled_score"),
            "train_scaled_score": scores.get("train_scaled_score"),
            "oos_scaled_score_report_only": scores.get("oos_scaled_score_report_only"),
            "train_trade_count": int(_safe_float(gate.get("train_trade_count"), 0.0)),
            "val_trade_count": int(_safe_float(gate.get("val_trade_count"), 0.0)),
            "oos_trade_count": int(_safe_float(gate.get("oos_trade_count"), 0.0)),
            "train_liquidation_count": int(_safe_float(gate.get("train_liquidation_count"), 0.0)),
            "val_liquidation_count": int(_safe_float(gate.get("val_liquidation_count"), 0.0)),
            "oos_liquidation_count": int(_safe_float(gate.get("oos_liquidation_count"), 0.0)),
            "train_final_equity": _safe_float(gate.get("train_final_equity"), 0.0),
            "val_final_equity": _safe_float(gate.get("val_final_equity"), 0.0),
            "oos_final_equity": _safe_float(gate.get("oos_final_equity"), 0.0),
            "cash_weight": float(preflight.cash_weight),
            "symbols": ",".join(preflight.symbols),
            "blocking_reasons": ";".join(preflight.blocking_reasons[:12]),
        }
        for split_name in ("train", "val", "oos"):
            split_metrics = dict(metrics.get(split_name) or {})
            for key in METRIC_KEYS:
                row[f"{split_name}_{key}"] = _safe_float(split_metrics.get(key), 0.0)
            for key in METRIC_DIAGNOSTIC_KEYS:
                if key == "equity_breach_observed":
                    row[f"{split_name}_{key}"] = bool(split_metrics.get(key))
                else:
                    row[f"{split_name}_{key}"] = _safe_float(split_metrics.get(key), 0.0)
        rows.append(row)

    def _status_priority(row: dict[str, Any]) -> int:
        if bool(row.get("selection_eligible")):
            return 0
        if str(row.get("status") or "") == "live_equivalent_validated":
            return 1
        status = str(row.get("status") or "")
        if status == "ready_for_live_equivalent_backtest":
            return 2
        if status == "eligible_conservative_cash_fallback":
            return 3
        if status == "blocked_missing_raw_first_market_data":
            return 4
        return 5

    rows.sort(
        key=lambda row: (
            _status_priority(row),
            -_safe_float(row.get("selection_score"), -1e9),
            str(row.get("mode") or ""),
        )
    )
    return rows


def _best(rows: list[dict[str, Any]], *, predicate) -> dict[str, Any] | None:
    eligible = [row for row in rows if predicate(row)]
    if not eligible:
        return None
    return sorted(
        eligible, key=lambda row: _safe_float(row.get("selection_score"), -1e9), reverse=True
    )[0]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _load_backtest_checkpoint(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"artifact_kind": "live_equivalent_backtest_checkpoint", "split_results": {}}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"artifact_kind": "live_equivalent_backtest_checkpoint", "split_results": {}}
    if not isinstance(payload, dict):
        return {"artifact_kind": "live_equivalent_backtest_checkpoint", "split_results": {}}
    payload.setdefault("artifact_kind", "live_equivalent_backtest_checkpoint")
    if not isinstance(payload.get("split_results"), dict):
        payload["split_results"] = {}
    return payload


def _write_backtest_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload["updated_at"] = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _checkpointed_split_run(
    checkpoint: dict[str, Any],
    *,
    mode: str,
    split: SplitWindow,
) -> dict[str, Any] | None:
    split_results = checkpoint.get("split_results")
    if not isinstance(split_results, dict):
        return None
    by_mode = split_results.get(str(mode))
    if not isinstance(by_mode, dict):
        return None
    row = by_mode.get(str(split.name))
    if not isinstance(row, dict) or str(row.get("status") or "") != "completed":
        return None
    if not isinstance(row.get("metrics"), dict):
        return None
    return dict(row)


def _store_checkpointed_split_run(
    checkpoint: dict[str, Any],
    checkpoint_path: Path,
    *,
    mode: str,
    split_run: dict[str, Any],
) -> None:
    split_name = str(split_run.get("split") or "")
    if not split_name:
        return
    split_results = checkpoint.setdefault("split_results", {})
    if not isinstance(split_results, dict):
        checkpoint["split_results"] = split_results = {}
    by_mode = split_results.setdefault(str(mode), {})
    if not isinstance(by_mode, dict):
        split_results[str(mode)] = by_mode = {}
    by_mode[split_name] = dict(split_run)
    _write_backtest_checkpoint(checkpoint_path, checkpoint)


def _progress_event(event: str, **payload: Any) -> None:
    print(
        json.dumps(
            {
                "event": event,
                "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                **payload,
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )


def _fmt_pct(value: Any) -> str:
    return f"{_safe_float(value, 0.0):+.2%}"


def _fmt_float(value: Any) -> str:
    parsed = _safe_float(value, float("nan"))
    if not math.isfinite(parsed):
        return "n/a"
    return f"{parsed:.4f}"


def _markdown_report(
    payload: dict[str, Any], mode_rows: list[dict[str, Any]], artifact_rows: list[dict[str, Any]]
) -> str:
    recs = dict(payload.get("final_recommendations") or {})
    backtests_executed = bool(payload.get("backtests_executed"))
    validated_count = sum(
        1 for row in mode_rows if str(row.get("status") or "") == "live_equivalent_validated"
    )
    alpha_eligible_count = sum(1 for row in mode_rows if bool(row.get("selection_eligible")))
    ready_count = sum(
        1
        for row in mode_rows
        if str(row.get("status") or "") == "ready_for_live_equivalent_backtest"
    )
    blocked_count = sum(
        1
        for row in mode_rows
        if str(row.get("status") or "") == "blocked_missing_raw_first_market_data"
    )
    lines = [
        "# Live-equivalent candidate revalidation — 2026-04-26",
        "",
        f"Generated: `{payload['generated_at']}`",
        "",
        "## 기준 변경",
        "",
        "- 기존 full-universe/legacy/HYBRID 리포트의 일별 return stream 점수는 이제 **연구 참고값**이다.",
        "- 실투자 후보는 동일한 `ArtifactPortfolioModeStrategy`를 live와 backtest가 같이 사용하고, `SimulatedExecutionHandler`/`Portfolio`를 통과한 이벤트 기반 결과만 selection eligible이다.",
        "- OOS는 계속 report-only이며, selection/tuning/health prior에는 쓰지 않는다.",
        "- cash efficiency는 점수에 넣지 않는다. 또한 0-trade/무수익/현금성 후보는 alpha ranking에서 제외한다.",
        "",
        "## 결론",
        "",
    ]
    best_live = recs.get("best_full_universe_live_equivalent_candidate")
    best_hybrid = recs.get("best_deployable_true_hybrid_candidate")
    fallback = recs.get("best_conservative_fallback_candidate")
    if best_live:
        lines.append(f"- Best full-universe live-equivalent candidate: `{best_live['mode']}`")
    else:
        lines.append(
            "- Best full-universe live-equivalent candidate: `NONE` — profit alpha gate를 통과한 후보가 없다."
        )
    if best_hybrid:
        lines.append(f"- Best deployable true-HYBRID candidate: `{best_hybrid['mode']}`")
    else:
        lines.append(
            "- Best deployable true-HYBRID candidate: `NONE` — dynamic/true HYBRID는 아직 live-equivalent engine validation 미완료다."
        )
    if fallback:
        lines.append(
            f"- Conservative fallback/shadow: `{fallback['mode']}` (`{fallback['status']}`)"
        )
    lines.extend(
        [
            "",
            "## 왜 이전 val return을 그대로 쓰면 안 되는가",
            "",
            "이전 랭킹은 저장된 artifact/일별 return stream을 재조합했다. 거래 엔진의 주문 크기, 수수료/슬리피지, 체결 이벤트, 컴포넌트별 EXIT 처리, live portfolio mode symbol universe를 같은 경로로 강제하지 않았다. 따라서 실투자 승격 기준으로는 모두 재검증 대상이다.",
            "",
            "## Live portfolio mode preflight / revalidation",
            "",
            "| rank | mode | status | alpha | score | val ret | val Sharpe | val MDD | trades train/val | blocker |",
            "|---:|---|---|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for rank, row in enumerate(mode_rows[:30], start=1):
        blocker = (
            row.get("alpha_blocking_reasons")
            or row.get("blocking_reasons")
            or row.get("symbols")
            or ""
        )
        alpha = (
            "yes" if bool(row.get("selection_eligible")) else str(row.get("selection_role") or "no")
        )
        lines.append(
            f"| {rank} | `{row['mode']}` | `{row['status']}` | "
            f"{alpha} | "
            f"{_fmt_float(row.get('selection_score'))} | "
            f"{_fmt_pct(row.get('val_total_return'))} | "
            f"{_fmt_float(row.get('val_sharpe'))} | "
            f"{_fmt_pct(row.get('val_max_drawdown'))} | "
            f"{int(_safe_float(row.get('train_trade_count'), 0.0))}/{int(_safe_float(row.get('val_trade_count'), 0.0))} | "
            f"{blocker} |"
        )
    lines.extend(
        [
            "",
            "## Research artifact reset sample",
            "",
            "| research rank | name | previous score | mapped live mode | reset status |",
            "|---:|---|---:|---|---|",
        ]
    )
    for row in artifact_rows[:40]:
        lines.append(
            f"| {row['research_rank']} | `{row['name']}` | {_fmt_float(row['previous_selection_score'])} | "
            f"`{row['mapped_live_portfolio_mode'] or '-'}` | `{row['reset_status']}` |"
        )
    if alpha_eligible_count:
        coverage_caveat = (
            f"- `{alpha_eligible_count}`개 mode가 live-equivalent engine backtest와 profit alpha gate를 모두 통과했다. "
            "selection eligibility는 양(+)의 validation return/Sharpe/Sortino와 active MDD/trade/liquidation gate 통과 후보에만 부여한다."
        )
    elif blocked_count:
        coverage_caveat = (
            f"- `{blocked_count}`개 live portfolio mode는 아직 train/val raw-first materialized coverage가 부족해서 "
            "`blocked_missing_raw_first_market_data` 상태다."
        )
    elif ready_count and not backtests_executed:
        coverage_caveat = (
            f"- train/val raw-first materialized coverage preflight는 `{ready_count}`개 alpha mode에서 통과했다. "
            "아직 `--execute-backtests`를 실행하지 않았으므로 readiness는 엔진 검증 대기 상태이며 selection evidence는 아니다."
        )
    elif validated_count:
        coverage_caveat = (
            f"- `{validated_count}`개 mode가 train/val live-equivalent engine backtest를 완료했다. "
            "하지만 profit alpha gate를 통과한 후보가 없어 fallback/shadow-only 상태를 유지한다."
        )
    else:
        coverage_caveat = (
            "- train/val live-equivalent alpha 후보는 아직 selection eligible 상태가 아니다."
        )
    lines.extend(
        [
            "",
            "## 명시적 caveats",
            "",
            coverage_caveat,
            "- 이 리포트의 핵심 변경은 `좋아 보이는 연구 점수`를 promotion evidence로 쓰지 않고, live-equivalent engine path를 통과한 후보만 승격시키는 것이다.",
            "- profit alpha gate는 val return/Sharpe/Sortino가 양수이고, train/val 거래 수와 active MDD/liquidation gate를 통과한 후보만 alpha selection eligible로 인정한다.",
            "- OOS는 report-only다. OOS raw-first coverage가 부족한 경우에도 train/val selection score에는 반영하지 않는다.",
        ]
    )
    if ready_count and not backtests_executed:
        lines.append(
            "- 다음 단계는 `--execute-backtests`로 같은 live portfolio mode 후보들을 train/val/OOS 재랭킹하는 것이다."
        )
    return "\n".join(lines) + "\n"


def _write_live_decision(payload: dict[str, Any], path: Path) -> None:
    best_live = dict(
        payload.get("final_recommendations", {}).get("best_full_universe_live_equivalent_candidate")
        or {}
    )
    fallback = dict(
        payload.get("final_recommendations", {}).get("best_conservative_fallback_candidate") or {}
    )
    mode_rows = list(payload.get("mode_candidate_rows") or [])
    ready_count = sum(
        1
        for row in mode_rows
        if str(dict(row).get("status") or "") == "ready_for_live_equivalent_backtest"
    )
    validated_count = sum(
        1 for row in mode_rows if str(dict(row).get("status") or "") == "live_equivalent_validated"
    )
    selected_mode = str(best_live.get("mode") or fallback.get("mode") or "risk_off_mode")
    if best_live:
        decision_state = "live_equivalent_validated_candidate_ready_for_review"
        decision_reason = "An alpha candidate completed live-equivalent train/val engine backtest on committed raw-first data."
        review_status = "candidate_ready_for_deployment_review"
    elif ready_count:
        decision_state = "engine_validation_pending_after_raw_first_preflight"
        decision_reason = (
            "Raw-first train/val preflight is complete for alpha candidates, but no alpha candidate has completed "
            "live-equivalent train/val engine backtest yet."
        )
        review_status = "engine_backtest_required_before_promotion"
    elif validated_count:
        decision_state = "shadow_only_until_profit_alpha_gate_passes"
        decision_reason = (
            "Live-equivalent train/val engine backtests completed, but no alpha candidate passed positive "
            "validation return/Sharpe/Sortino, active MDD, trade-count, and liquidation gates."
        )
        review_status = "fail_closed_no_profitable_alpha_candidate"
    else:
        decision_state = "shadow_only_until_live_equivalent_validation"
        decision_reason = "No alpha candidate has completed live-equivalent train/val engine backtest on committed raw-first data."
        review_status = "fail_closed_cash_or_shadow_only"
    decision = {
        "artifact_kind": "portfolio_live_readiness_decision",
        "generated_at": payload["generated_at"],
        "decision": decision_state,
        "decision_reason": decision_reason,
        "selected_mode": selected_mode,
        "candidate_mode": selected_mode,
        "candidate_key": selected_mode,
        "review_status": review_status,
        "selection_basis": "live_equivalent_revalidation_20260426",
        "source_artifacts": {
            "live_equivalent_revalidation_path": str(
                (OUTPUT_DIR / "live_equivalent_revalidation_latest.json").resolve()
            )
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(decision, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def build_live_equivalent_revalidation(
    *,
    output_dir: Path = OUTPUT_DIR,
    full_universe_path: Path = FULL_UNIVERSE_PATH,
    market_root: Path | None = None,
    exchange: str | None = None,
    timeframe: str = "1s",
    execute_backtests: bool = False,
    update_live_decision: bool = True,
    chunk_days: int = 1,
    backtest_poll_seconds: int = 20,
    backtest_window_seconds: int = 20,
    backtest_checkpoint_path: Path | None = None,
    portfolio_modes: list[str] | None = None,
    fail_fast_alpha_gate: bool = False,
    oos_end: date | None = None,
) -> dict[str, Any]:
    market_root = Path(market_root or str(BaseConfig.MARKET_DATA_PARQUET_PATH))
    exchange = str(exchange or BaseConfig.MARKET_DATA_EXCHANGE).lower()
    checkpoint_path = Path(backtest_checkpoint_path or DEFAULT_BACKTEST_CHECKPOINT_PATH)
    checkpoint = _load_backtest_checkpoint(checkpoint_path) if execute_backtests else None
    splits = _split_windows(oos_end=oos_end)
    full_universe = _load_full_universe(full_universe_path)
    artifact_rows = _artifact_candidate_reset_rows(full_universe)

    if portfolio_modes:
        modes = []
        for raw_mode in portfolio_modes:
            mode = str(raw_mode or "").strip()
            if not mode:
                continue
            if not supports_live_portfolio_mode(mode) and mode not in supported_portfolio_modes():
                raise ValueError(f"unsupported live-equivalent portfolio mode filter: {mode}")
            modes.append(mode)
        modes = sorted(dict.fromkeys(modes))
    else:
        modes = sorted(supported_portfolio_modes())
    preflights = {
        mode: _mode_preflight(
            mode=mode,
            market_root=market_root,
            exchange=exchange,
            timeframe=timeframe,
            splits=splits,
        )
        for mode in modes
    }
    equivalence_cache: dict[tuple[str, str], dict[str, Any]] = {}
    mode_results: list[dict[str, Any]] = []
    for mode, preflight in preflights.items():
        if execute_backtests:
            mode_results.append(
                _run_mode_backtests(
                    preflight=preflight,
                    market_root=market_root,
                    exchange=exchange,
                    timeframe=timeframe,
                    splits=splits,
                    chunk_days=chunk_days,
                    backtest_poll_seconds=backtest_poll_seconds,
                    backtest_window_seconds=backtest_window_seconds,
                    checkpoint=checkpoint,
                    checkpoint_path=checkpoint_path,
                    equivalence_cache=equivalence_cache,
                    fail_fast_alpha_gate=fail_fast_alpha_gate,
                )
            )
        else:
            mode_results.append(
                {
                    "mode": mode,
                    "status": preflight.status,
                    "metrics": {split.name: _empty_metrics() for split in splits},
                    "scores": {},
                    "split_runs": [],
                    "blocking_reasons": list(preflight.blocking_reasons),
                }
            )
    mode_rows = _mode_candidate_rows(mode_results, preflights)
    best_live = _best(
        mode_rows,
        predicate=lambda row: (
            bool(row.get("selection_eligible")) and str(row.get("selection_role") or "") == "alpha"
        ),
    )
    true_hybrid_modes = {
        "hybrid_guarded_mode",
        "legacy_no_highvol_hybrid_mode",
        "retuned_live_portfolio_hybrid_mode",
    }
    best_hybrid = _best(
        mode_rows,
        predicate=lambda row: (
            bool(row.get("selection_eligible"))
            and str(row.get("selection_role") or "") == "alpha"
            and str(row.get("mode")) in true_hybrid_modes
        ),
    )
    fallback = next((row for row in mode_rows if row.get("mode") == "risk_off_mode"), None)
    shadow_queue = [
        row
        for row in mode_rows
        if row.get("mode") != "risk_off_mode"
        and str(row.get("status"))
        in {"blocked_missing_raw_first_market_data", "ready_for_live_equivalent_backtest"}
    ][:10]

    payload: dict[str, Any] = {
        "artifact_kind": "live_equivalent_revalidation",
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "backtests_executed": bool(execute_backtests),
        "selection_policy": {
            "selection_basis": "validation_primary_live_equivalent_engine_backtest_only",
            "research_artifact_streams": "report_only_not_selection_evidence",
            "oos_role": "report_only",
            "cash_efficiency_scored": False,
            "train_val_mdd_cap": MDD_CAP,
            "profit_alpha_gate": {
                "fallback_modes_excluded_from_alpha_ranking": True,
                "active_train_mdd_cap": ACTIVE_TRAIN_MDD_CAP,
                "active_val_mdd_cap": ACTIVE_VAL_MDD_CAP,
                "min_train_trades": MIN_ALPHA_TRAIN_TRADES,
                "min_val_trades": MIN_ALPHA_VAL_TRADES,
                "min_train_total_return": MIN_ALPHA_TRAIN_TOTAL_RETURN,
                "min_val_total_return": MIN_ALPHA_VAL_TOTAL_RETURN,
                "min_val_sharpe": MIN_ALPHA_VAL_SHARPE,
                "min_val_sortino": MIN_ALPHA_VAL_SORTINO,
                "liquidations_allowed": 0,
            },
            "score_formula": "val_scaled_score + 0.18*train_scaled_score; scaled score uses return, Sharpe, Sortino, Calmar, MDD headroom",
        },
        "split_windows": [split.as_payload() for split in splits],
        "execution_model": {
            "strategy_class": "ArtifactPortfolioModeStrategy",
            "backtest_data_handler": "HistoricParquetWindowedDataHandler",
            "execution_handler": "SimulatedExecutionHandler",
            "portfolio": "Portfolio (same implementation reused by live boundary)",
            "market_data_contract": "raw-first committed materialized parquet",
            "market_root": str(market_root),
            "exchange": str(exchange),
            "timeframe": str(timeframe),
            "chunk_days": int(chunk_days),
            "backtest_poll_seconds": int(backtest_poll_seconds),
            "backtest_window_seconds": int(backtest_window_seconds),
            "backtest_checkpoint_path": str(checkpoint_path) if execute_backtests else "",
            "fail_fast_alpha_gate": bool(fail_fast_alpha_gate),
        },
        "preflights": {mode: preflight.as_payload() for mode, preflight in preflights.items()},
        "mode_results": mode_results,
        "mode_candidate_rows": mode_rows,
        "research_artifact_reset_rows": artifact_rows,
        "final_recommendations": {
            "best_full_universe_live_equivalent_candidate": best_live,
            "best_deployable_true_hybrid_candidate": best_hybrid,
            "best_conservative_fallback_candidate": fallback,
            "shadow_queue_requires_market_data_then_engine_backtest": shadow_queue,
            "candidate_reset": "All prior artifact-ranked alpha candidates are demoted until they pass live-equivalent train/val engine backtest.",
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    latest_json = output_dir / "live_equivalent_revalidation_latest.json"
    latest_md = output_dir / "live_equivalent_revalidation_latest.md"
    mode_csv = output_dir / "live_equivalent_revalidation_candidates_20260426.csv"
    artifact_csv = output_dir / "live_equivalent_revalidation_artifact_reset_20260426.csv"
    latest_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    latest_md.write_text(_markdown_report(payload, mode_rows, artifact_rows), encoding="utf-8")
    _write_csv(mode_csv, mode_rows)
    _write_csv(artifact_csv, artifact_rows)
    if update_live_decision:
        _write_live_decision(payload, LIVE_DECISION_PATH)
    return {
        "payload": payload,
        "paths": {
            "json": str(latest_json.resolve()),
            "markdown": str(latest_md.resolve()),
            "mode_csv": str(mode_csv.resolve()),
            "artifact_reset_csv": str(artifact_csv.resolve()),
            "live_decision": str(LIVE_DECISION_PATH.resolve()) if update_live_decision else "",
        },
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--full-universe-path", default=str(FULL_UNIVERSE_PATH))
    parser.add_argument("--market-root", default=str(BaseConfig.MARKET_DATA_PARQUET_PATH))
    parser.add_argument("--exchange", default=str(BaseConfig.MARKET_DATA_EXCHANGE))
    parser.add_argument("--timeframe", default="1s")
    parser.add_argument("--chunk-days", type=int, default=1)
    parser.add_argument("--backtest-poll-seconds", type=int, default=20)
    parser.add_argument("--backtest-window-seconds", type=int, default=20)
    parser.add_argument("--backtest-checkpoint-path", default=str(DEFAULT_BACKTEST_CHECKPOINT_PATH))
    parser.add_argument(
        "--oos-end-date",
        default="",
        help="Inclusive OOS end date override; useful when the latest UTC day is not raw-first complete.",
    )
    parser.add_argument(
        "--portfolio-modes",
        default="",
        help="Comma-separated live portfolio modes to revalidate; defaults to all supported modes.",
    )
    parser.add_argument(
        "--execute-backtests",
        action="store_true",
        help=(
            "Run full event-driven train/val/OOS backtests for modes with complete raw-first coverage. "
            "This can be very slow on 1s exact-window data and is therefore opt-in."
        ),
    )
    parser.add_argument("--no-live-decision-update", action="store_true")
    parser.add_argument(
        "--fail-fast-alpha-gate",
        action="store_true",
        help=(
            "Stop a mode after the train split when train-only alpha-gate "
            "requirements already fail. Use for research iteration; omit when "
            "a full train/val/OOS audit is required."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    result = build_live_equivalent_revalidation(
        output_dir=Path(args.output_dir),
        full_universe_path=Path(args.full_universe_path),
        market_root=Path(args.market_root),
        exchange=str(args.exchange),
        timeframe=str(args.timeframe),
        execute_backtests=bool(args.execute_backtests),
        update_live_decision=not bool(args.no_live_decision_update),
        chunk_days=max(1, int(args.chunk_days)),
        backtest_poll_seconds=max(1, int(args.backtest_poll_seconds)),
        backtest_window_seconds=max(1, int(args.backtest_window_seconds)),
        backtest_checkpoint_path=Path(args.backtest_checkpoint_path),
        portfolio_modes=[
            item.strip() for item in str(args.portfolio_modes or "").split(",") if item.strip()
        ]
        or None,
        fail_fast_alpha_gate=bool(args.fail_fast_alpha_gate),
        oos_end=datetime.fromisoformat(str(args.oos_end_date)).date()
        if str(args.oos_end_date or "").strip()
        else None,
    )
    print(json.dumps(result["paths"], ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
