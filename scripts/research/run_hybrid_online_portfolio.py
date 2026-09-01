"""Deterministic hybrid online portfolio governor over saved sleeve streams.

Inspired by ensemble_strategies hybrid v3.5/v3.6:
- warmup default sleeve
- online default sleeve switching from trailing scores
- exponential weighting
- optional current-health shrinkage

This runner is intentionally thin and deterministic:
- no parameter sweep
- no strategy search
- no market-data backtest
- operates on saved sleeve return streams only
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from lumina_quant.eval.exact_window_runtime import RSSLimitExceeded
from lumina_quant.portfolio_split_contract import (
    FOLLOWUP_ROOT,
    PORTFOLIO_FOLLOWUP_EXPLICIT_BUDGET_BYTES,
    acquire_portfolio_memory_guard,
    memory_policy_payload,
)

_DYN_SPEC = importlib.util.spec_from_file_location(
    "run_causal_dynamic_portfolio",
    Path(__file__).resolve().parent / "run_causal_dynamic_portfolio.py",
)
if _DYN_SPEC is None or _DYN_SPEC.loader is None:
    raise RuntimeError("Failed to load run_causal_dynamic_portfolio helpers")
_DYN = importlib.util.module_from_spec(_DYN_SPEC)
sys.modules[_DYN_SPEC.name] = _DYN
_DYN_SPEC.loader.exec_module(_DYN)

GROUP_ROOT = FOLLOWUP_ROOT / "portfolio_incumbent_autoresearch_grouped"
OUTPUT_DIR = GROUP_ROOT / "portfolio_hybrid_online_current"

HISTORICAL_INPUTS = {
    "soft_three_way_regime": GROUP_ROOT
    / "soft_three_way_market_regime_allocator_current"
    / "soft_three_way_market_regime_allocator_latest.json",
    "three_way_regime": GROUP_ROOT
    / "three_way_market_regime_allocator_current"
    / "three_way_market_regime_allocator_latest.json",
    "static_blend_76_24": GROUP_ROOT / "grouped_incumbent_autoresearch_static_blend_latest.json",
    "incumbent_only": FOLLOWUP_ROOT
    / "portfolio_one_shot_current_opt"
    / "portfolio_optimization_latest.json",
    "pair_tactical_mode": GROUP_ROOT
    / "pair_spread_robustness_cadence_refresh_followup_current"
    / "research_run"
    / "candidate_research_latest.json",
}
REFRESHED_INPUTS = {
    "soft_three_way_regime": GROUP_ROOT
    / "current_switch_validation_current"
    / "refreshed_soft_three_way_allocator_current"
    / "soft_three_way_market_regime_allocator_latest.json",
    "three_way_regime": GROUP_ROOT
    / "current_switch_validation_current"
    / "refreshed_three_way_allocator_current"
    / "three_way_market_regime_allocator_latest.json",
    "static_blend_76_24": GROUP_ROOT
    / "current_switch_validation_current"
    / "refreshed_grouped_static_blend_latest.json",
    "incumbent_only": GROUP_ROOT
    / "current_switch_validation_current"
    / "refreshed_current_one_shot_incumbent_portfolio_latest.json",
    "balanced_overlay_80_20": GROUP_ROOT
    / "current_switch_validation_current"
    / "refreshed_balanced_overlay_strategy_latest.json",
    "pair_tactical_mode": GROUP_ROOT
    / "current_switch_validation_current"
    / "refreshed_pair_fast_exit_candidate_latest.json",
    "production_guarded_portfolio": GROUP_ROOT
    / "portfolio_production_guarded_current"
    / "production_guarded_portfolio_latest.json",
}
OPERATING_SWITCH_PATH = (
    GROUP_ROOT
    / "current_switch_validation_current"
    / "refreshed_operating_switch_current"
    / "portfolio_operating_switch_latest.json"
)

ACTIVE_SLEEVES = (
    "balanced_overlay_80_20",
    "soft_three_way_regime",
    "pair_tactical_mode",
    "production_guarded_portfolio",
)
BENCHMARK_SLEEVES = ("three_way_regime", "static_blend_76_24", "incumbent_only")
PAIR_NAME = "pair_spread_1h_exec_tightstop_tp_fastexit_bnbusdt_trxusdt_2.5_0.75"


@dataclass(frozen=True, slots=True)
class HybridSplitConfig:
    train_start: str = "2025-01-01"
    train_end: str = "2025-12-31"
    val_start: str = "2026-01-01"
    val_end: str = "2026-02-28"
    oos_start: str = "2026-03-01"
    oos_end: str | None = None

    def train_start_date(self) -> date:
        return _parse_split_day(self.train_start)

    def train_end_date(self) -> date:
        return _parse_split_day(self.train_end)

    def val_start_date(self) -> date:
        return _parse_split_day(self.val_start)

    def val_end_date(self) -> date:
        return _parse_split_day(self.val_end)

    def oos_start_date(self) -> date:
        return _parse_split_day(self.oos_start)

    def oos_end_date(self) -> date | None:
        if self.oos_end is None:
            return None
        token = str(self.oos_end).strip()
        if not token:
            return None
        return _parse_split_day(token)

    def split_for_day_key(self, day_key: str) -> str | None:
        day_value = _parse_split_day(day_key)
        if day_value < self.train_start_date():
            return None
        if day_value <= self.train_end_date():
            return "train"
        if self.val_start_date() <= day_value <= self.val_end_date():
            return "val"
        oos_end = self.oos_end_date()
        if day_value >= self.oos_start_date() and (oos_end is None or day_value <= oos_end):
            return "oos"
        return None

    def online_start_date(self, warmup_days: int) -> str:
        return (self.train_start_date() + timedelta(days=max(0, int(warmup_days)))).isoformat()

    def pre_oos_days(self) -> int:
        return int((self.oos_start_date() - self.train_start_date()).days)

    def as_payload(self) -> dict[str, Any]:
        return {
            "train_start": self.train_start_date().isoformat(),
            "train_end_inclusive": self.train_end_date().isoformat(),
            "val_start": self.val_start_date().isoformat(),
            "val_end_inclusive": self.val_end_date().isoformat(),
            "oos_start": self.oos_start_date().isoformat(),
            "oos_end_inclusive": self.oos_end_date().isoformat()
            if self.oos_end_date()
            else "latest",
            "pre_oos_days": self.pre_oos_days(),
        }


@dataclass(slots=True)
class HybridOnlineConfig:
    variant: str = "fixed_default"
    warmup_days: int | None = None
    warmup_ratio: float = 0.60
    lookback_days: int = 13
    default_boost: float = 0.2335751227788591
    sticky_default_bonus: float = 0.09538352712472187
    switch_margin: float = 0.09865009243733545
    score_temperature: float = 0.9070667959024572
    min_positive_score: float = 0.14655939990705838
    pair_score_boost: float = 0.00015054198515658535
    disagreement_threshold: float = 0.22293874953018045
    disagreement_cash_scale: float = 0.856739550695035
    pair_weight_cap: float = 0.17534288208655102
    diversified_weight_cap: float = 0.8757254013166067
    pair_pbo_penalty_scale: float = 1.8475128599670687
    pair_sparsity_penalty_scale: float = 2.732355465712022
    negative_health_floor: float = 0.09432330421001382
    mixed_health_floor: float = 0.5897144969917639
    use_current_health_priors: bool = True


def _json_default(value: Any) -> Any:
    if isinstance(value, (datetime,)):
        return value.isoformat()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _parse_split_day(value: Any) -> date:
    token = str(value or "").strip()
    if not token:
        raise ValueError("missing split day token")
    token = token.split("T", 1)[0]
    return date.fromisoformat(token)


def add_split_config_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    defaults = HybridSplitConfig()
    parser.add_argument("--train-start", default=defaults.train_start)
    parser.add_argument("--train-end", default=defaults.train_end)
    parser.add_argument("--val-start", default=defaults.val_start)
    parser.add_argument("--val-end", default=defaults.val_end)
    parser.add_argument("--oos-start", default=defaults.oos_start)
    parser.add_argument("--oos-end", default=defaults.oos_end)
    return parser


def split_config_from_args(args: argparse.Namespace) -> HybridSplitConfig:
    return HybridSplitConfig(
        train_start=str(args.train_start),
        train_end=str(args.train_end),
        val_start=str(args.val_start),
        val_end=str(args.val_end),
        oos_start=str(args.oos_start),
        oos_end=None if args.oos_end in (None, "") else str(args.oos_end),
    )


def resolve_warmup_days(*, config: HybridOnlineConfig, split_config: HybridSplitConfig) -> int:
    if config.warmup_days is not None:
        return max(0, int(config.warmup_days))
    return max(0, int(np.ceil(split_config.pre_oos_days() * float(config.warmup_ratio))))


def _load_hybrid_config_payload(path: Path) -> dict[str, Any]:
    payload = _load_json(path)
    if "config" in payload and isinstance(payload.get("config"), dict):
        return dict(payload.get("config") or {})
    if "best" in payload and isinstance(dict(payload.get("best") or {}).get("config"), dict):
        return dict((payload.get("best") or {}).get("config") or {})
    if "best_trial" in payload and isinstance(
        dict(payload.get("best_trial") or {}).get("config"), dict
    ):
        return dict((payload.get("best_trial") or {}).get("config") or {})
    return payload


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object in {path}")
    return payload


def _safe_float(value: Any, default: float = 0.0) -> float:
    return float(_DYN._safe_float(value, default))


def _split_index(day_key: str, *, split_config: HybridSplitConfig) -> str | None:
    return split_config.split_for_day_key(day_key)


def _portfolio_return_streams_from_daily(
    dates: list[str],
    daily_returns: list[float],
    *,
    split_config: HybridSplitConfig,
) -> dict[str, list[dict[str, Any]]]:
    streams: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "oos": []}
    for day_key, day_return in zip(dates, daily_returns, strict=True):
        split = _split_index(str(day_key), split_config=split_config)
        if split is None:
            continue
        streams[split].append({"t": f"{day_key}T00:00:00Z", "v": _safe_float(day_return, 0.0)})
    return streams


def _split_metrics_from_streams(
    streams: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[str, float]]:
    return {
        split: _DYN._metrics(
            np.asarray(
                [_safe_float(point.get("v"), 0.0) for point in list(streams.get(split) or [])],
                dtype=float,
            )
        )
        for split in ("train", "val", "oos")
    }


def _payload_daily_map(payload: dict[str, Any]) -> dict[str, float]:
    for key in ("portfolio_daily_return_streams", "portfolio_return_streams", "return_streams"):
        stream_payload = payload.get(key)
        if isinstance(stream_payload, dict):
            merged: dict[str, float] = {}
            for split in ("train", "val", "oos"):
                merged.update(_DYN._daily_compound_stream(list(stream_payload.get(split) or [])))
            if merged:
                return merged
    dates = list(payload.get("dates") or [])
    daily_returns = list(payload.get("daily_returns") or [])
    if dates and daily_returns:
        normalized_dates = []
        for value in dates:
            token = str(value)
            if "T" in token:
                token = token.split("T", 1)[0]
            normalized_dates.append(token)
        return {
            day_key: _safe_float(day_return, 0.0)
            for day_key, day_return in zip(normalized_dates, daily_returns, strict=True)
        }
    raise RuntimeError("Could not resolve daily return streams from payload")


def _payload_daily_streams(
    payload: dict[str, Any],
    *,
    split_config: HybridSplitConfig,
) -> dict[str, list[dict[str, Any]]]:
    daily_map = _payload_daily_map(payload)
    dates = sorted(daily_map)
    returns = [_safe_float(daily_map[day_key], 0.0) for day_key in dates]
    return _portfolio_return_streams_from_daily(dates, returns, split_config=split_config)


def _load_pair_candidate(path: Path, *, refreshed: bool) -> dict[str, Any]:
    payload = _load_json(path)
    if refreshed:
        row = payload
    else:
        candidates = [
            dict(row) for row in list(payload.get("candidates") or []) if isinstance(row, dict)
        ]
        try:
            row = next(row for row in candidates if str(row.get("name")) == PAIR_NAME)
        except StopIteration as exc:
            raise RuntimeError(f"Pair sleeve {PAIR_NAME} not found in {path}") from exc
    return row


def _make_sleeve_row(
    *,
    sleeve_name: str,
    source_payload: dict[str, Any],
    streams: dict[str, list[dict[str, Any]]],
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    split_metrics = _split_metrics_from_streams(streams)
    return {
        "candidate_id": sleeve_name,
        "name": sleeve_name,
        "strategy_class": metadata.get("strategy_class", sleeve_name) if metadata else sleeve_name,
        "strategy_timeframe": metadata.get("timeframe", "1d") if metadata else "1d",
        "family": metadata.get("family", "portfolio") if metadata else "portfolio",
        "symbols": list(metadata.get("symbols") or []) if metadata else [],
        "return_streams": streams,
        "metadata": {
            **(metadata or {}),
            "source_payload_path": str(metadata.get("source_payload_path") or ""),
        },
        "train": dict(split_metrics.get("train") or {}),
        "val": dict(split_metrics.get("val") or {}),
        "oos": dict(split_metrics.get("oos") or {}),
    }


def _merged_daily_map(streams: dict[str, list[dict[str, Any]]]) -> dict[str, float]:
    merged: dict[str, float] = {}
    for split in ("train", "val", "oos"):
        merged.update(_DYN._daily_compound_stream(list(streams.get(split) or [])))
    return merged


def _blend_streams(
    left: dict[str, list[dict[str, Any]]],
    right: dict[str, list[dict[str, Any]]],
    *,
    left_weight: float,
    right_weight: float,
    split_config: HybridSplitConfig,
) -> dict[str, list[dict[str, Any]]]:
    left_map = _merged_daily_map(left)
    right_map = _merged_daily_map(right)
    all_days = sorted(set(left_map) | set(right_map))
    daily_returns = [
        (left_weight * _safe_float(left_map.get(day), 0.0))
        + (right_weight * _safe_float(right_map.get(day), 0.0))
        for day in all_days
    ]
    return _portfolio_return_streams_from_daily(all_days, daily_returns, split_config=split_config)


def _source_sleeve_metrics(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        str(row.get("name")): {
            "family": str(row.get("family") or ""),
            "strategy_class": str(row.get("strategy_class") or ""),
            "timeframe": str(row.get("strategy_timeframe") or ""),
            "symbols": list(row.get("symbols") or []),
            "train": dict(row.get("train") or {}),
            "val": dict(row.get("val") or {}),
            "oos": dict(row.get("oos") or {}),
        }
        for row in rows
    }


def _cash_row(day_keys: list[str]) -> dict[str, Any]:
    return {
        "candidate_id": "risk_off_cash",
        "name": "risk_off_cash",
        "strategy_class": "CashSleeve",
        "strategy_timeframe": "1d",
        "family": "cash",
        "symbols": [],
        "return_streams": _DYN._portfolio_return_streams_from_daily(
            day_keys, [0.0] * len(day_keys)
        ),
        "metadata": {"source_payload_path": ""},
        "train": {"total_return": 0.0, "sharpe": 0.0, "max_drawdown": 0.0},
        "val": {"total_return": 0.0, "sharpe": 0.0, "max_drawdown": 0.0},
        "oos": {"total_return": 0.0, "sharpe": 0.0, "max_drawdown": 0.0},
    }


def _historical_active_rows(
    *,
    split_config: HybridSplitConfig | None = None,
) -> list[dict[str, Any]]:
    split_config = split_config or HybridSplitConfig()
    soft_payload = _load_json(HISTORICAL_INPUTS["soft_three_way_regime"])
    pair_row = _load_pair_candidate(HISTORICAL_INPUTS["pair_tactical_mode"], refreshed=False)
    rows = [
        _make_sleeve_row(
            sleeve_name="soft_three_way_regime",
            source_payload=soft_payload,
            streams=_payload_daily_streams(soft_payload, split_config=split_config),
            metadata={
                "strategy_class": "SoftThreeWayAllocator",
                "timeframe": "1d",
                "family": "portfolio",
                "source_payload_path": str(HISTORICAL_INPUTS["soft_three_way_regime"].resolve()),
            },
        ),
        _make_sleeve_row(
            sleeve_name="pair_tactical_mode",
            source_payload=pair_row,
            streams=_payload_daily_streams(pair_row, split_config=split_config),
            metadata={
                "strategy_class": "PairSpreadZScoreStrategy",
                "timeframe": "1h",
                "family": "market_neutral_pair",
                "symbols": list(pair_row.get("symbols") or []),
                "source_payload_path": str(HISTORICAL_INPUTS["pair_tactical_mode"].resolve()),
            },
        ),
    ]
    balanced_streams = _blend_streams(
        rows[0]["return_streams"],
        rows[1]["return_streams"],
        left_weight=0.8,
        right_weight=0.2,
        split_config=split_config,
    )
    balanced_metrics = {
        split: _DYN._metrics(
            np.asarray([point["v"] for point in balanced_streams[split]], dtype=float)
        )
        for split in ("train", "val", "oos")
    }
    rows.append(
        {
            "candidate_id": "balanced_overlay_80_20",
            "name": "balanced_overlay_80_20",
            "strategy_class": "BalancedOverlayPortfolio",
            "strategy_timeframe": "1d",
            "family": "portfolio_overlay",
            "symbols": [],
            "return_streams": balanced_streams,
            "metadata": {"source_payload_path": "derived:soft_three_way_regime+pair_tactical_mode"},
            **balanced_metrics,
        }
    )
    all_day_keys = sorted(
        set().union(*(_merged_daily_map(row["return_streams"]).keys() for row in rows))
    )
    rows.insert(0, _cash_row(all_day_keys))
    return rows


def _historical_benchmark_rows(
    *,
    split_config: HybridSplitConfig | None = None,
) -> list[dict[str, Any]]:
    split_config = split_config or HybridSplitConfig()
    benchmarks: list[dict[str, Any]] = []
    for sleeve_name in BENCHMARK_SLEEVES:
        payload = _load_json(HISTORICAL_INPUTS[sleeve_name])
        benchmarks.append(
            _make_sleeve_row(
                sleeve_name=sleeve_name,
                source_payload=payload,
                streams=_payload_daily_streams(payload, split_config=split_config),
                metadata={
                    "strategy_class": sleeve_name,
                    "timeframe": "1d",
                    "family": "portfolio",
                    "source_payload_path": str(HISTORICAL_INPUTS[sleeve_name].resolve()),
                },
            )
        )
    return benchmarks


def _refreshed_rows(
    *,
    split_config: HybridSplitConfig | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    split_config = split_config or HybridSplitConfig()
    active: list[dict[str, Any]] = []
    benchmarks: list[dict[str, Any]] = []
    soft_payload = _load_json(REFRESHED_INPUTS["soft_three_way_regime"])
    pair_payload = _load_pair_candidate(REFRESHED_INPUTS["pair_tactical_mode"], refreshed=True)
    balanced_payload = _load_json(REFRESHED_INPUTS["balanced_overlay_80_20"])
    active.append(
        _make_sleeve_row(
            sleeve_name="soft_three_way_regime",
            source_payload=soft_payload,
            streams=_payload_daily_streams(soft_payload, split_config=split_config),
            metadata={
                "strategy_class": "SoftThreeWayAllocator",
                "timeframe": "1d",
                "family": "portfolio",
                "source_payload_path": str(REFRESHED_INPUTS["soft_three_way_regime"].resolve()),
            },
        )
    )
    active.append(
        _make_sleeve_row(
            sleeve_name="balanced_overlay_80_20",
            source_payload=balanced_payload,
            streams=_payload_daily_streams(balanced_payload, split_config=split_config),
            metadata={
                "strategy_class": "BalancedOverlayPortfolio",
                "timeframe": "1d",
                "family": "portfolio_overlay",
                "source_payload_path": str(REFRESHED_INPUTS["balanced_overlay_80_20"].resolve()),
            },
        )
    )
    active.append(
        _make_sleeve_row(
            sleeve_name="pair_tactical_mode",
            source_payload=pair_payload,
            streams=_payload_daily_streams(pair_payload, split_config=split_config),
            metadata={
                "strategy_class": "PairSpreadZScoreStrategy",
                "timeframe": "1h",
                "family": "market_neutral_pair",
                "symbols": list(pair_payload.get("symbols") or []),
                "source_payload_path": str(REFRESHED_INPUTS["pair_tactical_mode"].resolve()),
            },
        )
    )
    production_payload_path = Path(REFRESHED_INPUTS["production_guarded_portfolio"])
    if production_payload_path.exists():
        production_payload = _load_json(production_payload_path)
        active.append(
            _make_sleeve_row(
                sleeve_name="production_guarded_portfolio",
                source_payload=production_payload,
                streams=_payload_daily_streams(production_payload, split_config=split_config),
                metadata={
                    "strategy_class": "ProductionGuardedPortfolio",
                    "timeframe": "1d",
                    "family": "portfolio_overlay",
                    "max_weight_cap": 0.45,
                    "source_payload_path": str(production_payload_path.resolve()),
                },
            )
        )
    for sleeve_name in BENCHMARK_SLEEVES:
        payload = _load_json(REFRESHED_INPUTS[sleeve_name])
        benchmarks.append(
            _make_sleeve_row(
                sleeve_name=sleeve_name,
                source_payload=payload,
                streams=_payload_daily_streams(payload, split_config=split_config),
                metadata={
                    "strategy_class": sleeve_name,
                    "timeframe": "1d",
                    "family": "portfolio",
                    "source_payload_path": str(REFRESHED_INPUTS[sleeve_name].resolve()),
                },
            )
        )
    all_day_keys = sorted(
        set().union(
            *(_merged_daily_map(row["return_streams"]).keys() for row in active + benchmarks)
        )
    )
    active.insert(0, _cash_row(all_day_keys))
    return active, benchmarks


def _health_prior(metrics: dict[str, Any], config: HybridOnlineConfig) -> float:
    total_return = _safe_float(metrics.get("total_return", metrics.get("return")), 0.0)
    sharpe = _safe_float(metrics.get("sharpe"), 0.0)
    if total_return > 0.0 and sharpe > 0.0:
        return 1.0
    if total_return > 0.0 or sharpe > 0.0:
        return config.mixed_health_floor
    return config.negative_health_floor


def _fragility_penalty(row: dict[str, Any], *, config: HybridOnlineConfig) -> float:
    name = str(row.get("name") or "")
    if name == "risk_off_cash":
        return 0.0
    train = dict(row.get("train") or {})
    train_return = _safe_float(train.get("total_return", train.get("return")), 0.0)
    train_trades = _safe_float(train.get("trade_count", train.get("trades")), 0.0)
    if train_return == 0.0 and train_trades == 0.0:
        return 100.0
    oos = dict(row.get("oos") or {})
    pbo = max(0.0, _safe_float(oos.get("pbo"), 0.0))
    active_fold_ratio = _safe_float(oos.get("active_fold_ratio"), 1.0)
    trade_count = _safe_float(oos.get("trade_count", oos.get("trades")), 0.0)
    penalty = 0.0
    penalty += config.pair_pbo_penalty_scale * pbo
    penalty += config.pair_sparsity_penalty_scale * max(0.0, 0.5 - active_fold_ratio)
    if trade_count < 12.0:
        penalty += (12.0 - trade_count) / 12.0
    return penalty


def _softmax(values: dict[str, float], temperature: float) -> dict[str, float]:
    if not values:
        return {}
    temp = max(1e-6, float(temperature))
    ordered = list(values.items())
    arr = np.asarray([value / temp for _, value in ordered], dtype=float)
    finite_mask = np.isfinite(arr)
    if not np.any(finite_mask):
        return {key: 0.0 for key, _ in ordered}
    arr = np.where(finite_mask, arr, -np.inf)
    max_val = np.max(arr[finite_mask])
    exps = np.exp(arr - max_val)
    exps[~finite_mask] = 0.0
    denom = float(np.sum(exps))
    if denom <= 1e-12:
        return {key: 0.0 for key, _ in ordered}
    return {key: float(weight / denom) for (key, _), weight in zip(ordered, exps, strict=True)}


def _apply_caps(raw: dict[str, float], *, caps: dict[str, float]) -> dict[str, float]:
    remaining = dict(raw)
    final: dict[str, float] = {}
    residual = 1.0
    for key, weight in sorted(remaining.items(), key=lambda item: item[1], reverse=True):
        cap = max(0.0, float(caps.get(key, 1.0)))
        assigned = min(cap, max(0.0, weight), residual)
        final[key] = assigned
        residual -= assigned
        if residual <= 1e-12:
            break
    return final


def _historical_default_name(rows: list[dict[str, Any]]) -> str:
    preferred = (
        "soft_three_way_regime",
        "core_mode",
        "balanced_overlay_80_20",
        "balanced_overlay_mode",
        "aggressive_realized_mode",
        "pair_tactical_mode",
    )
    available = {str(row.get("name")) for row in rows}
    for candidate in preferred:
        if candidate in available:
            return candidate
    return "risk_off_cash"


def run_hybrid_online_allocator(
    rows: list[dict[str, Any]],
    *,
    config: HybridOnlineConfig,
    refreshed_health_metrics: dict[str, dict[str, Any]] | None = None,
    split_config: HybridSplitConfig | None = None,
    default_name: str | None = None,
) -> dict[str, Any]:
    split_config = split_config or HybridSplitConfig()
    ordered_days, matrix, _meta = _DYN._build_daily_panel(rows)
    ids = [str(row.get("candidate_id") or row.get("name")) for row in rows]
    name_by_id = {
        str(row.get("candidate_id") or row.get("name")): str(row.get("name") or "") for row in rows
    }
    row_by_id = {str(row.get("candidate_id") or row.get("name")): row for row in rows}
    requested_default = str(default_name or "").strip()
    default_id = requested_default if requested_default in ids else _historical_default_name(rows)
    previous_default_id = default_id
    score_history: list[dict[str, Any]] = []
    allocations: list[dict[str, Any]] = []
    daily_returns: list[float] = []
    split_returns: dict[str, list[float]] = {"train": [], "val": [], "oos": []}

    health_priors = {
        cid: (
            1.0
            if (not config.use_current_health_priors or refreshed_health_metrics is None)
            else _health_prior(refreshed_health_metrics.get(name_by_id[cid], {}), config)
        )
        for cid in ids
    }
    fragility_penalties = {cid: _fragility_penalty(row_by_id[cid], config=config) for cid in ids}
    caps = {
        cid: (
            1.0
            if name_by_id[cid] == "risk_off_cash"
            else (
                _safe_float(
                    dict(row_by_id[cid].get("metadata") or {}).get("max_weight_cap"),
                    config.diversified_weight_cap,
                )
                if dict(row_by_id[cid].get("metadata") or {}).get("max_weight_cap") is not None
                else (
                    config.pair_weight_cap
                    if "pair" in name_by_id[cid]
                    else config.diversified_weight_cap
                )
            )
        )
        for cid in ids
    }

    resolved_warmup_days = max(
        resolve_warmup_days(config=config, split_config=split_config), int(config.lookback_days)
    )
    for idx, day_key in enumerate(ordered_days):
        split = _split_index(day_key, split_config=split_config)
        if split is None:
            continue
        if idx < resolved_warmup_days:
            weights = {default_id: 1.0} if default_id != "risk_off_cash" else {}
            cash_weight = 0.0 if weights else 1.0
            raw_scores = {cid: 0.0 for cid in ids if cid != "risk_off_cash"}
            current_default = default_id if weights else "risk_off_cash"
        else:
            raw_scores: dict[str, float] = {}
            for cid in ids:
                if name_by_id[cid] == "risk_off_cash":
                    continue
                window = np.asarray(matrix[cid][idx - config.lookback_days : idx], dtype=float)
                metrics = _DYN._metrics(window)
                score = _DYN._search_objective(metrics, cash_fraction=0.0)
                score *= health_priors[cid]
                score -= fragility_penalties[cid]
                if (
                    _safe_float(metrics.get("total_return"), 0.0) <= 0.0
                    or _safe_float(metrics.get("sharpe"), 0.0) <= 0.0
                ):
                    score -= 0.5
                raw_scores[cid] = float(score)
            positive_scores = {
                cid: score for cid, score in raw_scores.items() if score > config.min_positive_score
            }
            if not positive_scores:
                weights = {}
                cash_weight = 1.0
                current_default = "risk_off_cash"
            else:
                adjusted_scores = dict(positive_scores)
                pair_id = next(
                    (cid for cid in adjusted_scores if name_by_id[cid] == "pair_tactical_mode"),
                    None,
                )
                if pair_id is not None and config.pair_score_boost > 0.0:
                    adjusted_scores[pair_id] += config.pair_score_boost
                if config.variant == "fixed_default":
                    if previous_default_id in adjusted_scores:
                        current_default = previous_default_id
                    else:
                        current_default = max(adjusted_scores, key=adjusted_scores.get)
                else:
                    if previous_default_id in adjusted_scores:
                        adjusted_scores[previous_default_id] += config.sticky_default_bonus
                    candidate_default = max(adjusted_scores, key=adjusted_scores.get)
                    if (
                        previous_default_id in positive_scores
                        and candidate_default != previous_default_id
                        and adjusted_scores[candidate_default]
                        < positive_scores[previous_default_id] + config.switch_margin
                    ):
                        current_default = previous_default_id
                    else:
                        current_default = candidate_default
                score_weights = _softmax(adjusted_scores, config.score_temperature)
                if current_default in score_weights:
                    score_weights[current_default] += config.default_boost
                total = float(sum(score_weights.values()))
                if total > 1e-12:
                    score_weights = {cid: weight / total for cid, weight in score_weights.items()}
                weights = _apply_caps(score_weights, caps=caps)
                if config.variant == "disagreement_switching" and len(adjusted_scores) >= 2:
                    ranked_scores = sorted(adjusted_scores.values(), reverse=True)
                    top_gap = float(ranked_scores[0] - ranked_scores[1])
                    if top_gap < config.disagreement_threshold:
                        weights = {
                            cid: float(weight) * float(config.disagreement_cash_scale)
                            for cid, weight in weights.items()
                        }
                total_active = float(sum(weights.values()))
                if total_active > 1e-12:
                    weights = {
                        cid: float(weight) / total_active * total_active
                        for cid, weight in weights.items()
                    }
                cash_weight = max(0.0, 1.0 - sum(weights.values()))
        portfolio_ret = sum(
            float(matrix[cid][idx]) * float(weight) for cid, weight in weights.items()
        )
        daily_returns.append(float(portfolio_ret))
        split_returns.setdefault(split, []).append(float(portfolio_ret))
        allocations.append(
            {
                "date": day_key,
                "split": split,
                "default_sleeve": name_by_id.get(current_default, current_default),
                "weights": {name_by_id[cid]: float(weight) for cid, weight in weights.items()},
                "cash_weight": float(cash_weight),
            }
        )
        score_history.append(
            {
                "date": day_key,
                "split": split,
                "selected_default_sleeve": name_by_id.get(current_default, current_default),
                "raw_scores": {name_by_id[cid]: float(score) for cid, score in raw_scores.items()},
                "health_priors": {name_by_id[cid]: float(health_priors[cid]) for cid in raw_scores},
                "fragility_penalties": {
                    name_by_id[cid]: float(fragility_penalties[cid]) for cid in raw_scores
                },
            }
        )
        previous_default_id = current_default if current_default in ids else previous_default_id

    split_metrics = {
        split: _DYN._metrics(np.asarray(values, dtype=float))
        for split, values in split_returns.items()
    }
    all_metrics = _DYN._metrics(np.asarray(daily_returns, dtype=float))
    final_allocation = (
        allocations[-1] if allocations else {"date": None, "weights": {}, "cash_weight": 1.0}
    )
    return {
        "dates": ordered_days,
        "daily_returns": daily_returns,
        "resolved_warmup_days": int(resolved_warmup_days),
        "split_metrics": split_metrics,
        "all_metrics": all_metrics,
        "allocations": allocations,
        "score_history": score_history,
        "final_allocation": final_allocation,
        "health_priors": {
            name_by_id[cid]: float(prior)
            for cid, prior in health_priors.items()
            if name_by_id[cid] != "risk_off_cash"
        },
    }


def _comparison_rows(
    *,
    hybrid_result: dict[str, Any],
    benchmarks: list[dict[str, Any]],
    active_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = [
        {
            "name": "hybrid_online_portfolio",
            "kind": "hybrid",
            **dict((hybrid_result.get("split_metrics") or {}).get("oos") or {}),
        }
    ]
    for row in active_rows + benchmarks:
        rows.append(
            {
                "name": str(row.get("name")),
                "kind": "active"
                if str(row.get("name")) in ACTIVE_SLEEVES or str(row.get("name")) == "risk_off_cash"
                else "benchmark",
                **dict(row.get("oos") or {}),
            }
        )
    return rows


def _scoreboard_markdown(title: str, rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        f"## {title}",
        "",
        "| Sleeve | Kind | OOS return | Sharpe | Max DD |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| `{row['name']}` | {row['kind']} | {_safe_float(row.get('total_return', row.get('return')), 0.0):+.4%} | {_safe_float(row.get('sharpe'), 0.0):.4f} | {_safe_float(row.get('max_drawdown', row.get('mdd')), 0.0):.4%} |"
        )
    return lines


def write_hybrid_online_report(
    *,
    output_dir: Path = OUTPUT_DIR,
    config: HybridOnlineConfig | None = None,
    split_config: HybridSplitConfig | None = None,
) -> dict[str, Any]:
    config = config or HybridOnlineConfig()
    split_config = split_config or HybridSplitConfig()
    output_dir.mkdir(parents=True, exist_ok=True)
    memory_guard = acquire_portfolio_memory_guard(
        run_name="hybrid_online_portfolio",
        output_dir=output_dir,
        input_path=HISTORICAL_INPUTS["soft_three_way_regime"],
        budget_bytes=PORTFOLIO_FOLLOWUP_EXPLICIT_BUDGET_BYTES,
    )
    status = "completed"
    error: str | None = None
    try:
        historical_active = _historical_active_rows(split_config=split_config)
        historical_benchmarks = _historical_benchmark_rows(split_config=split_config)
        refreshed_active, refreshed_benchmarks = _refreshed_rows(split_config=split_config)
        refreshed_health_metrics = {
            row["name"]: dict(row.get("oos") or {})
            for row in refreshed_active + refreshed_benchmarks
        }
        memory_guard.sample(
            event="hybrid_online_loaded",
            context={
                "historical_active_count": len(historical_active),
                "refreshed_active_count": len(refreshed_active),
            },
        )

        historical_config = HybridOnlineConfig(
            **({**asdict(config), "use_current_health_priors": False})
        )
        historical_result = run_hybrid_online_allocator(
            historical_active,
            config=historical_config,
            refreshed_health_metrics=None,
            split_config=split_config,
        )
        memory_guard.sample(
            event="hybrid_online_historical_done",
            context={
                "oos_return": _safe_float(
                    (historical_result.get("split_metrics") or {})
                    .get("oos", {})
                    .get("total_return"),
                    0.0,
                )
            },
        )
        refreshed_result = run_hybrid_online_allocator(
            refreshed_active,
            config=config,
            refreshed_health_metrics=refreshed_health_metrics,
            split_config=split_config,
        )
        memory_guard.sample(
            event="hybrid_online_refreshed_done",
            context={
                "oos_return": _safe_float(
                    (refreshed_result.get("split_metrics") or {})
                    .get("oos", {})
                    .get("total_return"),
                    0.0,
                )
            },
        )
    except RSSLimitExceeded as exc:
        status = "aborted_rss_guard"
        error = str(exc)
        raise
    except Exception as exc:
        status = "failed"
        error = str(exc)
        raise
    finally:
        memory_guard.sample(
            event="hybrid_online_finish", context={"status": status, "error": error}
        )
        memory_summary = memory_guard.finalize(
            status=status,
            error=error,
            context={"output_dir": str(output_dir.resolve())},
        )
        memory_guard.release()

    refreshed_rows = _comparison_rows(
        hybrid_result=refreshed_result,
        benchmarks=refreshed_benchmarks,
        active_rows=refreshed_active,
    )
    refreshed_by_name = {row["name"]: row for row in refreshed_rows}
    pair_alloc_weights = [
        _safe_float((alloc.get("weights") or {}).get("pair_tactical_mode"), 0.0)
        for alloc in list(refreshed_result.get("allocations") or [])
    ]
    hybrid_total_return = _safe_float(
        refreshed_by_name["hybrid_online_portfolio"].get(
            "total_return", refreshed_by_name["hybrid_online_portfolio"].get("return")
        ),
        0.0,
    )
    readiness = {
        "beats_cash_refreshed": bool(
            hybrid_total_return
            > _safe_float(
                refreshed_by_name["risk_off_cash"].get(
                    "total_return", refreshed_by_name["risk_off_cash"].get("return")
                ),
                0.0,
            )
        ),
        "beats_pair_tactical_refreshed": bool(
            hybrid_total_return
            > _safe_float(
                refreshed_by_name["pair_tactical_mode"].get(
                    "total_return", refreshed_by_name["pair_tactical_mode"].get("return")
                ),
                0.0,
            )
        ),
        "beats_balanced_refreshed": bool(
            hybrid_total_return
            > _safe_float(
                refreshed_by_name["balanced_overlay_80_20"].get(
                    "total_return", refreshed_by_name["balanced_overlay_80_20"].get("return")
                ),
                0.0,
            )
        ),
        "max_rss_under_8gib": bool(
            _safe_float(dict(memory_summary).get("peak_rss_bytes"), 0.0) < (8 * 1024 * 1024 * 1024)
        ),
        "pair_cap_respected": bool(
            max(pair_alloc_weights or [0.0]) <= config.pair_weight_cap + 1e-9
        ),
    }
    if "production_guarded_portfolio" in refreshed_by_name:
        readiness["beats_production_guarded_refreshed"] = bool(
            hybrid_total_return
            > _safe_float(
                refreshed_by_name["production_guarded_portfolio"].get(
                    "total_return", refreshed_by_name["production_guarded_portfolio"].get("return")
                ),
                0.0,
            )
        )
    if (
        readiness["beats_cash_refreshed"]
        and readiness["max_rss_under_8gib"]
        and readiness["pair_cap_respected"]
    ):
        readiness["recommended_stage"] = (
            "pilot_candidate" if readiness["beats_pair_tactical_refreshed"] else "guarded_candidate"
        )
    else:
        readiness["recommended_stage"] = "do_not_integrate"

    payload = {
        "artifact_kind": "hybrid_online_portfolio",
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "selection_basis": "deterministic_hybrid_online_governor_over_saved_sleeves",
        "split_windows": split_config.as_payload(),
        "online_policy": {
            "warmup_ratio": float(config.warmup_ratio),
            "warmup_days": int(
                refreshed_result.get("resolved_warmup_days")
                or resolve_warmup_days(config=config, split_config=split_config)
            ),
            "lookback_days": int(config.lookback_days),
            "online_start": split_config.online_start_date(
                refreshed_result.get("resolved_warmup_days")
                or resolve_warmup_days(config=config, split_config=split_config)
            ),
        },
        "memory_policy": memory_policy_payload(
            budget_bytes=PORTFOLIO_FOLLOWUP_EXPLICIT_BUDGET_BYTES
        ),
        "memory_summary": memory_summary,
        "config": asdict(config),
        "current_market_state": dict(
            _load_json(OPERATING_SWITCH_PATH).get("current_market_state") or {}
        ),
        "readiness": readiness,
        "scenarios": {
            "historical_saved_baseline": {
                "active_sleeves": [row["name"] for row in historical_active],
                "benchmark_sleeves": [row["name"] for row in historical_benchmarks],
                "source_sleeve_metrics": _source_sleeve_metrics(
                    historical_active + historical_benchmarks
                ),
                **historical_result,
                "comparison_rows": _comparison_rows(
                    hybrid_result=historical_result,
                    benchmarks=historical_benchmarks,
                    active_rows=historical_active,
                ),
            },
            "refreshed_latest_tail": {
                "active_sleeves": [row["name"] for row in refreshed_active],
                "benchmark_sleeves": [row["name"] for row in refreshed_benchmarks],
                "source_sleeve_metrics": _source_sleeve_metrics(
                    refreshed_active + refreshed_benchmarks
                ),
                **refreshed_result,
                "comparison_rows": refreshed_rows,
            },
        },
        "approval": {
            "primary_refresh_beats_cash": readiness["beats_cash_refreshed"],
            "pair_tactical_is_benchmark_only": True,
        },
    }
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    json_path = output_dir / f"hybrid_online_portfolio_{stamp}.json"
    latest_path = output_dir / "hybrid_online_portfolio_latest.json"
    md_path = output_dir / f"hybrid_online_portfolio_{stamp}.md"
    json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default), encoding="utf-8"
    )
    latest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default), encoding="utf-8"
    )

    hist_rows = payload["scenarios"]["historical_saved_baseline"]["comparison_rows"]
    ref_rows = payload["scenarios"]["refreshed_latest_tail"]["comparison_rows"]
    ref_oos = dict((refreshed_result.get("split_metrics") or {}).get("oos") or {})
    lines = [
        "# Hybrid Online Portfolio",
        "",
        f"- generated_at: `{payload['generated_at']}`",
        f"- memory_log: `{dict(memory_summary).get('rss_log_path')}`",
        f"- peak_rss_mib: `{_safe_float(dict(memory_summary).get('peak_rss_mib'), 0.0):.2f}`",
        f"- current_market_state: favored_group=`{payload['current_market_state'].get('favored_group')}`, trend=`{payload['current_market_state'].get('trend_state')}`, breadth=`{payload['current_market_state'].get('breadth_state')}`, volatility=`{payload['current_market_state'].get('volatility_state')}`, pair_liquidity=`{payload['current_market_state'].get('pair_liquidity_state')}`",
        "",
        "## Deterministic config",
        "",
        "```json",
        json.dumps(payload["config"], indent=2, sort_keys=True),
        "```",
        "",
        "## Split windows",
        "",
        "```json",
        json.dumps(payload["split_windows"], indent=2, sort_keys=True),
        "```",
        "",
        f"- warmup_days: `{payload['online_policy']['warmup_days']}`",
        f"- lookback_days: `{payload['online_policy']['lookback_days']}`",
        f"- online_start: `{payload['online_policy']['online_start']}`",
        "",
        "## Readiness",
        "",
        f"- beats_cash_refreshed: `{readiness['beats_cash_refreshed']}`",
        f"- beats_pair_tactical_refreshed: `{readiness['beats_pair_tactical_refreshed']}`",
        f"- beats_balanced_refreshed: `{readiness['beats_balanced_refreshed']}`",
        *(
            [
                f"- beats_production_guarded_refreshed: `{readiness['beats_production_guarded_refreshed']}`"
            ]
            if "beats_production_guarded_refreshed" in readiness
            else []
        ),
        f"- max_rss_under_8gib: `{readiness['max_rss_under_8gib']}`",
        f"- pair_cap_respected: `{readiness['pair_cap_respected']}`",
        f"- recommended_stage: `{readiness['recommended_stage']}`",
        "",
        f"- refreshed/latest-tail primary approval: hybrid beats cash? `{payload['approval']['primary_refresh_beats_cash']}`",
        f"- pair tactical comparator is benchmark-only? `{payload['approval']['pair_tactical_is_benchmark_only']}`",
        "",
    ]
    lines.extend(_scoreboard_markdown("Historical saved baseline scoreboard", hist_rows))
    lines.extend([""])
    lines.extend(_scoreboard_markdown("Refreshed latest-tail scoreboard", ref_rows))
    lines.extend(
        [
            "",
            "## Refreshed latest-tail final allocation",
            "",
            f"- date: `{refreshed_result['final_allocation'].get('date')}`",
            f"- cash_weight: `{_safe_float(refreshed_result['final_allocation'].get('cash_weight'), 0.0):.2%}`",
        ]
    )
    for sleeve_name, weight in sorted(
        dict(refreshed_result["final_allocation"].get("weights") or {}).items(),
        key=lambda item: item[1],
        reverse=True,
    ):
        lines.append(f"- `{sleeve_name}`: `{_safe_float(weight, 0.0):.2%}`")
    lines.extend(
        [
            "",
            "## Refreshed latest-tail hybrid metrics",
            f"- oos_return: `{_safe_float(ref_oos.get('total_return'), 0.0):+.4%}`",
            f"- oos_sharpe: `{_safe_float(ref_oos.get('sharpe'), 0.0):.4f}`",
            f"- oos_max_dd: `{_safe_float(ref_oos.get('max_drawdown'), 0.0):.4%}`",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "payload": payload,
        "json_path": str(json_path.resolve()),
        "latest_path": str(latest_path.resolve()),
        "md_path": str(md_path.resolve()),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_split_config_arguments(parser)
    parser.add_argument("--warmup-ratio", type=float, default=HybridOnlineConfig().warmup_ratio)
    parser.add_argument("--warmup-days", type=int, default=None)
    parser.add_argument("--lookback-days", type=int, default=HybridOnlineConfig().lookback_days)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--config-json", type=Path, default=None)
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    split_config = split_config_from_args(args)
    config_kwargs = {
        **asdict(HybridOnlineConfig()),
        "warmup_ratio": float(args.warmup_ratio),
        "warmup_days": None if args.warmup_days is None else int(args.warmup_days),
        "lookback_days": int(args.lookback_days),
    }
    if args.config_json is not None:
        config_kwargs.update(_load_hybrid_config_payload(Path(args.config_json).resolve()))
        config_kwargs["warmup_ratio"] = float(args.warmup_ratio)
        config_kwargs["warmup_days"] = None if args.warmup_days is None else int(args.warmup_days)
    config = HybridOnlineConfig(**config_kwargs)
    result = write_hybrid_online_report(
        output_dir=Path(args.output_dir).resolve(),
        config=config,
        split_config=split_config,
    )
    print(result["latest_path"])
    print(result["md_path"])


if __name__ == "__main__":
    main()
