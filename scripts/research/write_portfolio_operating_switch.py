"""Write an operational portfolio-switch recommendation from current market state.

This is a lightweight decision layer on top of the saved allocator artifacts and
operating plan. It recomputes the *current* market regime snapshot from the
latest repaired feature-point data, evaluates the saved market-regime rules on
that latest snapshot, and combines the result with a recent pair-liquidity proxy
derived from raw aggTrades. The result is a concrete recommendation for which
deployment mode to run right now.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import polars as pl

from lumina_quant.portfolio_split_contract import FOLLOWUP_ROOT, ROOT

DEFAULT_GROUP_ROOT = FOLLOWUP_ROOT / "portfolio_incumbent_autoresearch_grouped"
DEFAULT_MARKET_JUDGEMENT_PATH = (
    DEFAULT_GROUP_ROOT
    / "market_regime_judgement_current"
    / "group_market_regime_judgement_latest.json"
)
DEFAULT_SOFT_ALLOCATOR_PATH = (
    DEFAULT_GROUP_ROOT
    / "soft_three_way_market_regime_allocator_current"
    / "soft_three_way_market_regime_allocator_latest.json"
)
DEFAULT_THREE_WAY_ALLOCATOR_PATH = (
    DEFAULT_GROUP_ROOT
    / "three_way_market_regime_allocator_current"
    / "three_way_market_regime_allocator_latest.json"
)
DEFAULT_OPERATING_PLAN_PATH = (
    DEFAULT_GROUP_ROOT
    / "portfolio_candidate_overlay_review_current"
    / "portfolio_operating_plan_latest.json"
)
DEFAULT_BALANCED_STRATEGY_PATH = (
    DEFAULT_GROUP_ROOT
    / "current_switch_validation_current"
    / "refreshed_balanced_overlay_strategy_latest.json"
)
DEFAULT_HYBRID_PORTFOLIO_PATH = (
    DEFAULT_GROUP_ROOT / "portfolio_hybrid_online_current" / "hybrid_online_portfolio_latest.json"
)
DEFAULT_PRODUCTION_GUARDED_PATH = (
    DEFAULT_GROUP_ROOT
    / "portfolio_production_guarded_current"
    / "production_guarded_portfolio_latest.json"
)
DEFAULT_OUTPUT_DIR = DEFAULT_GROUP_ROOT / "portfolio_operating_switch_current"
DEFAULT_MATERIALIZED_ROOT = (
    ROOT / "data" / "market_parquet" / "market_data_materialized" / "binance"
)
DEFAULT_RAW_AGGTRADES_ROOT = (
    ROOT / "data" / "market_parquet" / "market_data_raw_aggtrades" / "binance"
)
DEFAULT_PAIR_SYMBOLS = ("BNB/USDT", "TRX/USDT")
DEFAULT_PAIR_TIMEFRAME = "30m"
DEFAULT_VOLUME_LOOKBACK_DAYS = 7
DEFAULT_FEATURE_LOOKBACK_DAYS = 21
REBOOT_VALIDATION_ROOT = DEFAULT_GROUP_ROOT / "current_switch_validation_current"
REBOOT_MARKET_JUDGEMENT_PATH = (
    REBOOT_VALIDATION_ROOT
    / "refreshed_market_regime_judgement_current"
    / "group_market_regime_judgement_latest.json"
)
REBOOT_SOFT_ALLOCATOR_PATH = (
    REBOOT_VALIDATION_ROOT
    / "refreshed_soft_three_way_allocator_current"
    / "soft_three_way_market_regime_allocator_latest.json"
)
REBOOT_THREE_WAY_ALLOCATOR_PATH = (
    REBOOT_VALIDATION_ROOT
    / "refreshed_three_way_allocator_current"
    / "three_way_market_regime_allocator_latest.json"
)
REBOOT_BALANCED_STRATEGY_PATH = (
    REBOOT_VALIDATION_ROOT / "refreshed_balanced_overlay_strategy_latest.json"
)
REBOOT_SWITCH_VALIDATION_PATH = (
    REBOOT_VALIDATION_ROOT / "refreshed_switch_vs_strategy1_validation_latest.json"
)
REBOOT_OUTPUT_DIR = REBOOT_VALIDATION_ROOT / "refreshed_operating_switch_current"
HYBRID_PROMOTION_MIN_OOS_RETURN_EDGE = 0.0010
HYBRID_PROMOTION_MIN_OOS_SHARPE_EDGE = 0.75
HYBRID_PROMOTION_MAX_VAL_RETURN_GIVEBACK = 0.0100
HYBRID_PROMOTION_MAX_VAL_SHARPE_GIVEBACK = 0.35
HYBRID_PROMOTION_MAX_DRAWDOWN_RATIO = 0.80
HYBRID_PROMOTION_MIN_DRAWDOWN_IMPROVEMENT = 0.0010
# Coverage-adjusted replay supports a tighter mixed/calm hybrid promotion gate,
# but we intentionally keep rounded thresholds with headroom below the exact
# one-shot frontier so small refresh drift does not flip the live default.
HYBRID_PROMOTION_STRONG_OOS_RETURN_EDGE = 0.0050
HYBRID_PROMOTION_STRONG_OOS_SHARPE_EDGE = 2.5
HYBRID_PROMOTION_STRONG_VAL_TOTAL_RETURN = 0.0600
HYBRID_PROMOTION_STRONG_VAL_SHARPE = 3.0
PRODUCTION_PROMOTION_MIN_OOS_RETURN_EDGE = 0.0010
PRODUCTION_PROMOTION_MIN_OOS_SHARPE_EDGE = 0.50

_MARKET_SPEC = importlib.util.spec_from_file_location(
    "run_group_market_regime_judgement",
    Path(__file__).resolve().parent / "run_group_market_regime_judgement.py",
)
if _MARKET_SPEC is None or _MARKET_SPEC.loader is None:
    raise RuntimeError("Failed to load run_group_market_regime_judgement helpers")
_MARKET = importlib.util.module_from_spec(_MARKET_SPEC)
sys.modules[_MARKET_SPEC.name] = _MARKET
_MARKET_SPEC.loader.exec_module(_MARKET)


@dataclass(frozen=True, slots=True)
class SymbolVolumeSignal:
    symbol: str
    as_of_date: str
    latest_available_date: str | None
    stale_days: int | None
    latest_dollar_volume: float
    lookback_mean_dollar_volume: float
    volume_ratio: float | None
    comparison_mode: str
    state: str

    def as_payload(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class OperatingModeDecision:
    mode: str
    allocation: dict[str, float]
    rationale: list[str]

    def as_payload(self) -> dict[str, Any]:
        return asdict(self)


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        numeric = float(value)
    except TypeError, ValueError:
        return float(default)
    if not math.isfinite(numeric):
        return float(default)
    return float(numeric)


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object in {path}")
    return payload


def _profile_as_of_override(profile: str) -> str | None:
    normalized = str(profile or "live_current").strip().lower() or "live_current"
    if normalized != "reboot_validation" or not REBOOT_SWITCH_VALIDATION_PATH.exists():
        return None
    payload = _read_json(REBOOT_SWITCH_VALIDATION_PATH)
    token = str(
        payload.get("latest_common_complete_utc") or payload.get("refresh_cutoff_utc") or ""
    ).strip()
    return token or None


def _parse_as_of_date(value: Any) -> date:
    token = str(value or "").strip()
    if not token:
        raise ValueError("missing as_of date")
    normalized = token.replace("Z", "+00:00")
    if "T" in normalized or " " in normalized:
        return datetime.fromisoformat(normalized).astimezone(UTC).date()
    return date.fromisoformat(normalized)


def _symbol_token(symbol: str) -> str:
    return str(symbol).replace("/", "").upper()


def _latest_commit_files(date_dir: Path) -> list[Path]:
    commits = sorted(date_dir.glob("commit=*"))
    if not commits:
        return []
    commit_dir = commits[-1]
    return sorted(commit_dir.glob("*.parquet"))


def _load_symbol_volume_signal(
    *,
    raw_aggtrades_root: Path,
    symbol: str,
    as_of_date: date,
    lookback_days: int,
) -> SymbolVolumeSignal:
    symbol_root = raw_aggtrades_root / _symbol_token(symbol)
    if not symbol_root.exists():
        return SymbolVolumeSignal(
            symbol=symbol,
            as_of_date=as_of_date.isoformat(),
            latest_available_date=None,
            stale_days=None,
            latest_dollar_volume=0.0,
            lookback_mean_dollar_volume=0.0,
            volume_ratio=None,
            comparison_mode="missing",
            state="missing",
        )

    eligible_days = [
        path
        for path in sorted(symbol_root.glob("date=*"))
        if date.fromisoformat(path.name.split("=", 1)[1]) <= as_of_date
    ]
    if not eligible_days:
        return SymbolVolumeSignal(
            symbol=symbol,
            as_of_date=as_of_date.isoformat(),
            latest_available_date=None,
            stale_days=None,
            latest_dollar_volume=0.0,
            lookback_mean_dollar_volume=0.0,
            volume_ratio=None,
            comparison_mode="missing",
            state="missing",
        )

    selected_days = eligible_days[-max(1, int(lookback_days)) :]
    files = [str(path) for day_dir in selected_days for path in sorted(day_dir.glob("*.parquet"))]
    if not files:
        return SymbolVolumeSignal(
            symbol=symbol,
            as_of_date=as_of_date.isoformat(),
            latest_available_date=None,
            stale_days=None,
            latest_dollar_volume=0.0,
            lookback_mean_dollar_volume=0.0,
            volume_ratio=None,
            comparison_mode="missing",
            state="missing",
        )

    base_scan = (
        pl.scan_parquet(files)
        .select(
            pl.col("timestamp_ms").cast(pl.Int64).alias("timestamp_ms"),
            pl.from_epoch(pl.col("timestamp_ms"), time_unit="ms")
            .dt.replace_time_zone("UTC")
            .alias("datetime"),
            pl.col("price").cast(pl.Float64).fill_null(0.0).alias("price"),
            pl.col("quantity").cast(pl.Float64).fill_null(0.0).alias("quantity"),
        )
        .with_columns(
            (
                pl.col("price").clip(lower_bound=0.0) * pl.col("quantity").clip(lower_bound=0.0)
            ).alias("dollar_volume"),
            pl.col("datetime").dt.date().alias("day"),
            (pl.col("timestamp_ms") % 86_400_000).alias("ms_of_day"),
        )
    )
    daily = (
        base_scan.group_by("day")
        .agg(
            pl.sum("dollar_volume").alias("full_dollar_volume"),
            pl.max("ms_of_day").alias("latest_ms_of_day"),
        )
        .sort("day")
        .collect()
    )
    if daily.is_empty():
        return SymbolVolumeSignal(
            symbol=symbol,
            as_of_date=as_of_date.isoformat(),
            latest_available_date=None,
            stale_days=None,
            latest_dollar_volume=0.0,
            lookback_mean_dollar_volume=0.0,
            volume_ratio=None,
            comparison_mode="missing",
            state="missing",
        )

    latest_day = daily["day"].to_list()[-1]
    latest_cutoff_ms = int(_safe_float(daily["latest_ms_of_day"].to_list()[-1], 0.0))
    partial = (
        base_scan.filter(pl.col("ms_of_day") <= latest_cutoff_ms)
        .group_by("day")
        .agg(pl.sum("dollar_volume").alias("partial_dollar_volume"))
        .sort("day")
        .collect()
    )
    daily = daily.join(partial, on="day", how="left").with_columns(
        pl.col("partial_dollar_volume").fill_null(0.0)
    )

    latest_full_dollar_volume = _safe_float(daily["full_dollar_volume"].to_list()[-1], 0.0)
    latest_partial_dollar_volume = _safe_float(daily["partial_dollar_volume"].to_list()[-1], 0.0)
    full_dollar_volumes = [
        _safe_float(value, 0.0) for value in daily["full_dollar_volume"].to_list()
    ]
    partial_dollar_volumes = [
        _safe_float(value, 0.0) for value in daily["partial_dollar_volume"].to_list()
    ]
    previous_full = full_dollar_volumes[:-1]
    previous_partial = partial_dollar_volumes[:-1]
    if latest_day == as_of_date and previous_partial:
        latest_dollar_volume = latest_partial_dollar_volume
        lookback_mean = float(sum(previous_partial) / len(previous_partial))
        comparison_mode = "same_time_of_day"
    else:
        latest_dollar_volume = latest_full_dollar_volume
        reference_full = previous_full or full_dollar_volumes
        lookback_mean = float(sum(reference_full) / len(reference_full)) if reference_full else 0.0
        comparison_mode = "full_day"
    ratio = None if lookback_mean <= 0.0 else float(latest_dollar_volume / lookback_mean)
    stale_days = int((as_of_date - latest_day).days) if latest_day is not None else None

    if ratio is None:
        state = "missing"
    elif stale_days is not None and stale_days > 2:
        state = "stale"
    elif ratio >= 1.05:
        state = "strong"
    elif ratio <= 0.80:
        state = "weak"
    else:
        state = "normal"

    return SymbolVolumeSignal(
        symbol=symbol,
        as_of_date=as_of_date.isoformat(),
        latest_available_date=latest_day.isoformat() if latest_day is not None else None,
        stale_days=stale_days,
        latest_dollar_volume=latest_dollar_volume,
        lookback_mean_dollar_volume=lookback_mean,
        volume_ratio=ratio,
        comparison_mode=comparison_mode,
        state=state,
    )


def _pair_liquidity_state(signals: list[SymbolVolumeSignal]) -> str:
    states = [signal.state for signal in signals]
    if not states or all(state == "missing" for state in states):
        return "missing"
    if any(state in {"weak", "stale"} for state in states):
        return "weak"
    if all(state == "strong" for state in states):
        return "strong"
    return "normal"


def _load_latest_market_judgement(
    *,
    market_judgement_payload: Mapping[str, Any],
    feature_lookback_days: int,
    as_of_date: date | None = None,
) -> dict[str, Any]:
    selected_rules = [
        dict(rule)
        for rule in list(market_judgement_payload.get("selected_rules") or [])
        if isinstance(rule, Mapping)
    ]
    symbol_universe = [
        str(symbol)
        for symbol in list(market_judgement_payload.get("symbol_universe") or [])
        if str(symbol).strip()
    ]
    if not selected_rules or not symbol_universe:
        return dict(market_judgement_payload.get("current_judgement") or {})

    feature_root = Path(_MARKET.FEATURE_POINT_ROOT).resolve()
    latest_candidates: list[date] = []
    for symbol in symbol_universe:
        symbol_root = feature_root / f"symbol={_symbol_token(symbol)}"
        days = sorted(
            date.fromisoformat(path.name.split("=", 1)[1]) for path in symbol_root.glob("date=*")
        )
        if days:
            latest_candidates.append(days[-1])
    if not latest_candidates:
        return dict(market_judgement_payload.get("current_judgement") or {})

    latest_day = min(latest_candidates)
    if as_of_date is not None:
        latest_day = min(latest_day, as_of_date)
    start_day = (
        latest_day - _MARKET.pd.Timedelta(days=max(8, int(feature_lookback_days))).to_pytimedelta()
    )
    symbol_frames = []
    for symbol in symbol_universe:
        frame, _summary = _MARKET._load_symbol_close_30m_from_feature_points(
            symbol,
            start_day=_MARKET.pd.Timestamp(start_day, tz="UTC"),
            end_day=_MARKET.pd.Timestamp(latest_day, tz="UTC"),
        )
        symbol_frames.append(frame)

    feature_frame = _MARKET._daily_market_feature_frame(symbol_frames)
    if feature_frame.empty:
        return dict(market_judgement_payload.get("current_judgement") or {})
    latest_row = feature_frame.sort_values("date").iloc[-1]
    return dict(_MARKET._current_judgement(latest_row=latest_row, selected_rules=selected_rules))


def _profile_defaults(profile: str) -> dict[str, Any]:
    normalized = str(profile or "live_current").strip().lower() or "live_current"
    defaults = {
        "market_judgement_path": DEFAULT_MARKET_JUDGEMENT_PATH,
        "soft_allocator_path": DEFAULT_SOFT_ALLOCATOR_PATH,
        "three_way_allocator_path": DEFAULT_THREE_WAY_ALLOCATOR_PATH,
        "operating_plan_path": DEFAULT_OPERATING_PLAN_PATH,
        "balanced_strategy_path": DEFAULT_BALANCED_STRATEGY_PATH,
        "hybrid_portfolio_path": DEFAULT_HYBRID_PORTFOLIO_PATH,
        "materialized_root": DEFAULT_MATERIALIZED_ROOT,
        "raw_aggtrades_root": DEFAULT_RAW_AGGTRADES_ROOT,
        "output_dir": DEFAULT_OUTPUT_DIR,
        "market_judgement_mode": "latest",
    }
    if normalized == "reboot_validation":
        defaults.update(
            {
                "market_judgement_path": REBOOT_MARKET_JUDGEMENT_PATH,
                "soft_allocator_path": REBOOT_SOFT_ALLOCATOR_PATH,
                "three_way_allocator_path": REBOOT_THREE_WAY_ALLOCATOR_PATH,
                "balanced_strategy_path": REBOOT_BALANCED_STRATEGY_PATH,
                "output_dir": REBOOT_OUTPUT_DIR,
                "market_judgement_mode": "latest",
            }
        )
    return defaults


def _trend_state(snapshot: Mapping[str, Any]) -> str:
    above_192 = bool(snapshot.get("btc_above_ma192", False))
    above_336 = bool(snapshot.get("btc_above_ma336", False))
    gap_192 = _safe_float(snapshot.get("btc_trend_gap_192"), 0.0)
    gap_336 = _safe_float(snapshot.get("btc_trend_gap_336"), 0.0)
    accel = _safe_float(snapshot.get("btc_trend_accel"), 0.0)
    if above_192 and above_336 and gap_192 > 0.0 and gap_336 > 0.0:
        return "bullish"
    if (not above_192) and (not above_336) and gap_192 < 0.0 and gap_336 < 0.0:
        return "bearish"
    if accel > 0.0 and (above_192 or above_336):
        return "recovery"
    return "mixed"


def _breadth_state(snapshot: Mapping[str, Any]) -> str:
    breadth_96 = _safe_float(snapshot.get("breadth_ma96"), 0.0)
    breadth_192 = _safe_float(snapshot.get("breadth_ma192"), 0.0)
    breadth_delta = _safe_float(snapshot.get("breadth_delta"), 0.0)
    if breadth_96 >= 0.60 and breadth_192 >= 0.50 and breadth_delta >= 0.0:
        return "broad"
    if breadth_96 < 0.40 and breadth_192 < 0.40:
        return "weak"
    return "mixed"


def _volatility_state(snapshot: Mapping[str, Any]) -> str:
    ratio = _safe_float(snapshot.get("basket_vol_ratio"), 1.0)
    if ratio >= 1.15:
        return "stressed"
    if ratio <= 0.75:
        return "calm"
    return "normal"


def _overlay_candidate_health(operating_plan_payload: Mapping[str, Any]) -> dict[str, Any]:
    balanced = dict(
        (operating_plan_payload.get("deployment_modes") or {}).get("balanced_overlay_mode") or {}
    )
    metrics = dict(balanced.get("metrics") or {})
    return {
        "oos_total_return": _safe_float(metrics.get("oos_total_return"), 0.0),
        "oos_sharpe": _safe_float(metrics.get("oos_sharpe"), 0.0),
        "oos_max_drawdown": _safe_float(metrics.get("oos_max_drawdown"), 0.0),
        "validation_objective": _safe_float(metrics.get("validation_objective"), 0.0),
        "healthy": bool(
            _safe_float(metrics.get("oos_total_return"), 0.0) > 0.0
            and _safe_float(metrics.get("oos_sharpe"), 0.0) > 0.0
        ),
    }


def _balanced_strategy_health(
    *,
    operating_plan_payload: Mapping[str, Any],
    balanced_strategy_payload: Mapping[str, Any] | None = None,
    hybrid_source_metrics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    hybrid_source_metrics = dict(hybrid_source_metrics or {})
    if "balanced_overlay_80_20" in hybrid_source_metrics:
        return _allocator_health_from_split_metrics(
            dict(hybrid_source_metrics["balanced_overlay_80_20"])
        )
    if isinstance(balanced_strategy_payload, Mapping):
        metrics = dict((balanced_strategy_payload.get("portfolio_metrics") or {}).get("oos") or {})
        val = dict((balanced_strategy_payload.get("portfolio_metrics") or {}).get("val") or {})
        return {
            "oos_total_return": _safe_float(metrics.get("total_return"), 0.0),
            "oos_sharpe": _safe_float(metrics.get("sharpe"), 0.0),
            "oos_max_drawdown": _safe_float(metrics.get("max_drawdown"), 0.0),
            "validation_objective": _safe_float(
                (balanced_strategy_payload.get("validation_objective"))
                if "validation_objective" in balanced_strategy_payload
                else (
                    (1.0 * _safe_float(val.get("sharpe"), 0.0))
                    + (0.35 * _safe_float(val.get("sortino"), 0.0))
                    + (0.10 * _safe_float(val.get("calmar"), 0.0))
                    + (10.0 * _safe_float(val.get("total_return"), 0.0))
                    - (4.0 * _safe_float(val.get("max_drawdown"), 0.0))
                    - (0.75 * _safe_float(val.get("volatility"), 0.0))
                ),
                0.0,
            ),
            "healthy": bool(
                _safe_float(metrics.get("total_return"), 0.0) > 0.0
                and _safe_float(metrics.get("sharpe"), 0.0) > 0.0
            ),
        }
    return _overlay_candidate_health(operating_plan_payload)


def _allocator_health(payload: Mapping[str, Any]) -> dict[str, Any]:
    return _allocator_health_from_split_metrics(dict(payload.get("split_metrics") or {}))


def _allocator_health_from_split_metrics(split_metrics: Mapping[str, Any] | None) -> dict[str, Any]:
    split_metrics = dict(split_metrics or {})
    val = dict(split_metrics.get("val") or {})
    oos = dict(split_metrics.get("oos") or {})
    return {
        "val_total_return": _safe_float(val.get("total_return"), 0.0),
        "val_sharpe": _safe_float(val.get("sharpe"), 0.0),
        "val_max_drawdown": _safe_float(val.get("max_drawdown", val.get("mdd")), 0.0),
        "oos_total_return": _safe_float(oos.get("total_return"), 0.0),
        "oos_sharpe": _safe_float(oos.get("sharpe"), 0.0),
        "oos_max_drawdown": _safe_float(oos.get("max_drawdown", oos.get("mdd")), 0.0),
        "healthy": bool(
            _safe_float(oos.get("total_return"), 0.0) > 0.0
            and _safe_float(oos.get("sharpe"), 0.0) > 0.0
        ),
    }


def _hybrid_source_sleeve_metrics(
    hybrid_payload: Mapping[str, Any] | None,
    *,
    scenario: str = "refreshed_latest_tail",
) -> dict[str, Any]:
    if not isinstance(hybrid_payload, Mapping):
        return {}
    scenarios = dict(hybrid_payload.get("scenarios") or {})
    scenario_payload = dict(scenarios.get(scenario) or {})
    return {
        str(name): dict(metrics or {})
        for name, metrics in dict(scenario_payload.get("source_sleeve_metrics") or {}).items()
        if str(name).strip()
    }


def _hybrid_portfolio_health(hybrid_payload: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(hybrid_payload, Mapping):
        return {
            "val_total_return": 0.0,
            "val_sharpe": 0.0,
            "oos_total_return": 0.0,
            "oos_sharpe": 0.0,
            "oos_max_drawdown": 0.0,
            "recommended_stage": "",
            "healthy": False,
            "beats_cash_refreshed": False,
            "beats_balanced_refreshed": False,
            "beats_pair_tactical_refreshed": False,
            "pair_cap_respected": False,
            "max_rss_under_8gib": False,
        }
    refreshed = dict(
        dict((hybrid_payload.get("scenarios") or {}).get("refreshed_latest_tail") or {}).get(
            "split_metrics"
        )
        or {}
    )
    val = dict(refreshed.get("val") or {})
    oos = dict(refreshed.get("oos") or {})
    readiness = dict(hybrid_payload.get("readiness") or {})
    stage = str(readiness.get("recommended_stage") or "")
    healthy = bool(
        readiness.get("beats_cash_refreshed")
        and readiness.get("pair_cap_respected")
        and readiness.get("max_rss_under_8gib")
        and _safe_float(oos.get("total_return", oos.get("return")), 0.0) > 0.0
        and _safe_float(oos.get("sharpe"), 0.0) > 0.0
        and _safe_float(val.get("total_return", val.get("return")), 0.0) >= 0.0
        and _safe_float(val.get("sharpe"), 0.0) > 0.0
        and stage in {"guarded_candidate", "pilot_candidate"}
    )
    return {
        "val_total_return": _safe_float(val.get("total_return", val.get("return")), 0.0),
        "val_sharpe": _safe_float(val.get("sharpe"), 0.0),
        "oos_total_return": _safe_float(oos.get("total_return", oos.get("return")), 0.0),
        "oos_sharpe": _safe_float(oos.get("sharpe"), 0.0),
        "oos_max_drawdown": _safe_float(oos.get("max_drawdown", oos.get("mdd")), 0.0),
        "recommended_stage": stage,
        "healthy": healthy,
        "beats_cash_refreshed": bool(readiness.get("beats_cash_refreshed")),
        "beats_balanced_refreshed": bool(readiness.get("beats_balanced_refreshed")),
        "beats_pair_tactical_refreshed": bool(readiness.get("beats_pair_tactical_refreshed")),
        "pair_cap_respected": bool(readiness.get("pair_cap_respected")),
        "max_rss_under_8gib": bool(readiness.get("max_rss_under_8gib")),
    }


def _production_guarded_health(production_payload: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(production_payload, Mapping):
        return {
            "val_total_return": 0.0,
            "val_sharpe": 0.0,
            "oos_total_return": 0.0,
            "oos_sharpe": 0.0,
            "oos_max_drawdown": 0.0,
            "healthy": False,
        }
    metrics = dict(production_payload.get("portfolio_metrics") or {})
    val = dict(metrics.get("val") or {})
    oos = dict(metrics.get("oos") or {})
    return {
        "val_total_return": _safe_float(val.get("total_return", val.get("return")), 0.0),
        "val_sharpe": _safe_float(val.get("sharpe"), 0.0),
        "oos_total_return": _safe_float(oos.get("total_return", oos.get("return")), 0.0),
        "oos_sharpe": _safe_float(oos.get("sharpe"), 0.0),
        "oos_max_drawdown": _safe_float(oos.get("max_drawdown", oos.get("mdd")), 0.0),
        "healthy": bool(
            _safe_float(oos.get("total_return", oos.get("return")), 0.0) > 0.0
            and _safe_float(oos.get("sharpe"), 0.0) > 0.0
            and _safe_float(val.get("total_return", val.get("return")), 0.0) >= 0.0
            and _safe_float(val.get("sharpe"), 0.0) > 0.0
        ),
    }


def _hybrid_balanced_promotion_signal(
    *,
    hybrid_health: Mapping[str, Any] | None,
    balanced_health: Mapping[str, Any] | None,
) -> dict[str, Any]:
    hybrid = dict(hybrid_health or {})
    balanced = dict(balanced_health or {})
    oos_return_edge = _safe_float(hybrid.get("oos_total_return"), 0.0) - _safe_float(
        balanced.get("oos_total_return"), 0.0
    )
    oos_sharpe_edge = _safe_float(hybrid.get("oos_sharpe"), 0.0) - _safe_float(
        balanced.get("oos_sharpe"), 0.0
    )
    val_return_edge = _safe_float(hybrid.get("val_total_return"), 0.0) - _safe_float(
        balanced.get("val_total_return"), 0.0
    )
    val_sharpe_edge = _safe_float(hybrid.get("val_sharpe"), 0.0) - _safe_float(
        balanced.get("val_sharpe"), 0.0
    )
    hybrid_drawdown = _safe_float(hybrid.get("oos_max_drawdown"), 0.0)
    balanced_drawdown = _safe_float(balanced.get("oos_max_drawdown"), 0.0)
    drawdown_ok = (
        balanced_drawdown <= 0.0
        or hybrid_drawdown <= (balanced_drawdown * HYBRID_PROMOTION_MAX_DRAWDOWN_RATIO)
        or hybrid_drawdown <= (balanced_drawdown - HYBRID_PROMOTION_MIN_DRAWDOWN_IMPROVEMENT)
    )
    val_ok = (
        _safe_float(hybrid.get("val_total_return"), 0.0) >= 0.0
        and _safe_float(hybrid.get("val_sharpe"), 0.0) > 0.0
        and val_return_edge >= -HYBRID_PROMOTION_MAX_VAL_RETURN_GIVEBACK
        and val_sharpe_edge >= -HYBRID_PROMOTION_MAX_VAL_SHARPE_GIVEBACK
    )
    oos_ok = (
        oos_return_edge >= HYBRID_PROMOTION_MIN_OOS_RETURN_EDGE
        and oos_sharpe_edge >= HYBRID_PROMOTION_MIN_OOS_SHARPE_EDGE
    )
    profit_first_ok = bool(
        hybrid.get("healthy")
        and hybrid.get("beats_balanced_refreshed")
        and hybrid.get("beats_pair_tactical_refreshed")
        and str(hybrid.get("recommended_stage") or "").strip().lower() == "pilot_candidate"
        and oos_return_edge >= HYBRID_PROMOTION_STRONG_OOS_RETURN_EDGE
        and oos_sharpe_edge >= HYBRID_PROMOTION_STRONG_OOS_SHARPE_EDGE
        and _safe_float(hybrid.get("val_total_return"), 0.0)
        >= HYBRID_PROMOTION_STRONG_VAL_TOTAL_RETURN
        and _safe_float(hybrid.get("val_sharpe"), 0.0) >= HYBRID_PROMOTION_STRONG_VAL_SHARPE
        and drawdown_ok
    )
    promoted = bool(
        hybrid.get("healthy")
        and drawdown_ok
        and (
            ((hybrid.get("beats_balanced_refreshed") or oos_ok) and oos_ok and val_ok)
            or profit_first_ok
        )
    )
    return {
        "promoted": promoted,
        "oos_return_edge": oos_return_edge,
        "oos_sharpe_edge": oos_sharpe_edge,
        "val_return_edge": val_return_edge,
        "val_sharpe_edge": val_sharpe_edge,
        "hybrid_oos_max_drawdown": hybrid_drawdown,
        "balanced_oos_max_drawdown": balanced_drawdown,
        "drawdown_ok": drawdown_ok,
        "val_ok": val_ok,
        "oos_ok": oos_ok,
        "profit_first_ok": profit_first_ok,
    }


def _production_balanced_promotion_signal(
    *,
    production_health: Mapping[str, Any] | None,
    balanced_health: Mapping[str, Any] | None,
) -> dict[str, Any]:
    production = dict(production_health or {})
    balanced = dict(balanced_health or {})
    oos_return_edge = _safe_float(production.get("oos_total_return"), 0.0) - _safe_float(
        balanced.get("oos_total_return"), 0.0
    )
    oos_sharpe_edge = _safe_float(production.get("oos_sharpe"), 0.0) - _safe_float(
        balanced.get("oos_sharpe"), 0.0
    )
    drawdown_ok = _safe_float(production.get("oos_max_drawdown"), 0.0) <= _safe_float(
        balanced.get("oos_max_drawdown"), 0.0
    )
    promoted = bool(
        production.get("healthy")
        and oos_return_edge >= PRODUCTION_PROMOTION_MIN_OOS_RETURN_EDGE
        and oos_sharpe_edge >= PRODUCTION_PROMOTION_MIN_OOS_SHARPE_EDGE
        and drawdown_ok
    )
    return {
        "promoted": promoted,
        "oos_return_edge": oos_return_edge,
        "oos_sharpe_edge": oos_sharpe_edge,
        "production_oos_max_drawdown": _safe_float(production.get("oos_max_drawdown"), 0.0),
        "balanced_oos_max_drawdown": _safe_float(balanced.get("oos_max_drawdown"), 0.0),
    }


def recommend_operating_mode(
    *,
    current_judgement: Mapping[str, Any],
    soft_current_state: Mapping[str, Any],
    hard_current_state: Mapping[str, Any],
    operating_plan_payload: Mapping[str, Any],
    pair_liquidity_state: str,
    balanced_health: Mapping[str, Any] | None = None,
    hybrid_health: Mapping[str, Any] | None = None,
    production_health: Mapping[str, Any] | None = None,
) -> OperatingModeDecision:
    snapshot = dict(current_judgement.get("feature_snapshot") or {})
    favored_group = (
        str(current_judgement.get("favored_group") or "mixed").strip().lower() or "mixed"
    )
    confidence = _safe_float(current_judgement.get("confidence"), 0.0)
    trend_state = _trend_state(snapshot)
    breadth_state = _breadth_state(snapshot)
    volatility_state = _volatility_state(snapshot)
    overlay_health = dict(balanced_health or _overlay_candidate_health(operating_plan_payload))
    soft_health = _allocator_health(
        soft_current_state if "split_metrics" in soft_current_state else {}
    )
    hard_health = _allocator_health(
        hard_current_state if "split_metrics" in hard_current_state else {}
    )
    hybrid = dict(hybrid_health or {})
    production = dict(production_health or {})
    # If raw current_state only was passed, re-read health from payloads is not possible here.
    # Accept optional preattached health blocks.
    if not soft_health["healthy"] and isinstance(
        soft_current_state.get("_allocator_health"), Mapping
    ):
        soft_health = dict(soft_current_state.get("_allocator_health") or {})
    if not hard_health["healthy"] and isinstance(
        hard_current_state.get("_allocator_health"), Mapping
    ):
        hard_health = dict(hard_current_state.get("_allocator_health") or {})

    risk_score = 0.0
    if trend_state == "bearish":
        risk_score += 1.0
    elif trend_state == "bullish":
        risk_score -= 0.5

    if breadth_state == "weak":
        risk_score += 1.0
    elif breadth_state == "broad":
        risk_score -= 0.5

    if volatility_state == "stressed":
        risk_score += 2.0
    elif volatility_state == "calm":
        risk_score -= 0.5

    if pair_liquidity_state == "weak":
        risk_score += 1.0
    elif pair_liquidity_state == "strong":
        risk_score -= 0.5

    soft_incumbent_exposure = _safe_float(
        soft_current_state.get("effective_incumbent_exposure"), 0.0
    )
    soft_autoresearch_exposure = _safe_float(
        soft_current_state.get("effective_autoresearch_exposure"), 0.0
    )
    hard_state = (
        str(hard_current_state.get("state") or hard_current_state.get("selected_state") or "")
        .strip()
        .lower()
    )
    hard_raw_target = str(hard_current_state.get("raw_target_state") or "").strip().lower()

    deployment_modes = dict(operating_plan_payload.get("deployment_modes") or {})
    rationale: list[str] = [
        f"favored_group={favored_group} confidence={confidence:.4f}",
        f"trend_state={trend_state}, breadth_state={breadth_state}, volatility_state={volatility_state}, pair_liquidity_state={pair_liquidity_state}",
        f"soft_exposure incumbent={soft_incumbent_exposure:.4f} autoresearch={soft_autoresearch_exposure:.4f}",
        f"hard_state={hard_state or 'unknown'} raw_target_state={hard_raw_target or 'unknown'}",
        f"balanced_health healthy={bool(overlay_health.get('healthy'))} oos_return={_safe_float(overlay_health.get('oos_total_return'), 0.0):+.4%} oos_sharpe={_safe_float(overlay_health.get('oos_sharpe'), 0.0):.4f}",
        f"soft_health healthy={bool(soft_health.get('healthy'))} oos_return={_safe_float(soft_health.get('oos_total_return'), 0.0):+.4%} oos_sharpe={_safe_float(soft_health.get('oos_sharpe'), 0.0):.4f}",
        f"hard_health healthy={bool(hard_health.get('healthy'))} oos_return={_safe_float(hard_health.get('oos_total_return'), 0.0):+.4%} oos_sharpe={_safe_float(hard_health.get('oos_sharpe'), 0.0):.4f}",
        f"hybrid_health healthy={bool(hybrid.get('healthy'))} stage={hybrid.get('recommended_stage') or 'n/a'!s} beats_balanced={bool(hybrid.get('beats_balanced_refreshed'))} oos_return={_safe_float(hybrid.get('oos_total_return'), 0.0):+.4%} oos_sharpe={_safe_float(hybrid.get('oos_sharpe'), 0.0):.4f}",
        f"production_health healthy={bool(production.get('healthy'))} oos_return={_safe_float(production.get('oos_total_return'), 0.0):+.4%} oos_sharpe={_safe_float(production.get('oos_sharpe'), 0.0):.4f}",
    ]
    hybrid_promotion_signal = _hybrid_balanced_promotion_signal(
        hybrid_health=hybrid,
        balanced_health=overlay_health,
    )
    production_promotion_signal = _production_balanced_promotion_signal(
        production_health=production,
        balanced_health=overlay_health,
    )

    mode: str | None = None
    all_active_unhealthy = (
        not bool(overlay_health.get("healthy"))
        and not bool(soft_health.get("healthy"))
        and not bool(hard_health.get("healthy"))
    )

    if (
        all_active_unhealthy
        and bool(hybrid.get("healthy"))
        and favored_group == "incumbent"
        and trend_state != "bearish"
        and volatility_state != "stressed"
        and pair_liquidity_state != "weak"
    ):
        mode = "hybrid_guarded_mode"
        rationale.append(
            "Legacy active sleeves are degraded, but the hybrid online governor is healthy and current conditions are acceptable -> use hybrid guarded mode."
        )
    elif all_active_unhealthy:
        mode = "risk_off_mode"
        rationale.append(
            "All validated active sleeves are unhealthy -> move to risk-off mode until an active sleeve recovers."
        )
    elif (
        favored_group == "autoresearch"
        and confidence >= 0.65
        and volatility_state != "stressed"
        and pair_liquidity_state != "weak"
        and (hard_raw_target == "autoresearch_55_45" or hard_state == "autoresearch_55_45")
        and bool(hard_health.get("healthy"))
    ):
        mode = "aggressive_realized_mode"
        rationale.append(
            "High-confidence autoresearch regime with non-stressed volatility -> use aggressive realized mode."
        )
    elif (
        favored_group == "autoresearch"
        and confidence >= 0.65
        and not bool(hard_health.get("healthy"))
    ):
        if pair_liquidity_state != "weak" and overlay_health["healthy"]:
            mode = "balanced_overlay_mode"
            rationale.append(
                "Autoresearch regime fired, but hard allocator health is poor -> fall back to balanced overlay mode."
            )
        elif bool(soft_health.get("healthy")):
            mode = "core_mode"
            rationale.append(
                "Autoresearch regime fired, but only the core sleeve remains healthy -> fall back to core mode."
            )
        else:
            mode = "risk_off_mode"
            rationale.append(
                "Autoresearch regime fired, but no validated active sleeve is healthy -> stay risk-off."
            )
    elif favored_group == "incumbent" and confidence >= 0.60:
        if pair_liquidity_state == "weak" and bool(soft_health.get("healthy")):
            mode = "core_mode"
            rationale.append(
                "Incumbent-favored regime but pair liquidity is weak/stale -> remove overlay and stay core."
            )
        elif pair_liquidity_state == "weak":
            mode = "risk_off_mode"
            rationale.append(
                "Incumbent-favored regime but pair liquidity is weak/stale and core is unhealthy -> stay risk-off."
            )
        elif volatility_state == "stressed" and overlay_health["healthy"]:
            mode = "defensive_overlay_mode"
            rationale.append(
                "Incumbent-favored but volatility is stressed -> use the drawdown-reducing defensive overlay."
            )
        elif volatility_state == "stressed" and bool(soft_health.get("healthy")):
            mode = "core_mode"
            rationale.append(
                "Incumbent-favored but volatility is stressed and only core is healthy -> use core mode."
            )
        elif volatility_state == "stressed":
            mode = "risk_off_mode"
            rationale.append(
                "Incumbent-favored but volatility is stressed and no active sleeve is healthy -> stay risk-off."
            )
        elif risk_score >= 3.0:
            if bool(soft_health.get("healthy")):
                mode = "core_mode"
                rationale.append(
                    "Risk score is high under incumbent-favored conditions -> stay in core mode."
                )
            else:
                mode = "risk_off_mode"
                rationale.append(
                    "Risk score is high and the core sleeve is unhealthy -> stay risk-off."
                )
        elif overlay_health["healthy"]:
            mode = "balanced_overlay_mode"
            rationale.append(
                "Incumbent-favored regime with acceptable volatility/liquidity -> keep balanced overlay mode."
            )
        elif bool(soft_health.get("healthy")):
            mode = "core_mode"
            rationale.append(
                "Incumbent-favored regime confirmed, but overlay is unhealthy -> use core mode only."
            )
        else:
            mode = "risk_off_mode"
            rationale.append(
                "Incumbent-favored regime is not enough to override unhealthy active sleeves -> stay risk-off."
            )
    elif favored_group == "mixed":
        if (volatility_state == "stressed" or pair_liquidity_state == "weak") and bool(
            soft_health.get("healthy")
        ):
            mode = "core_mode"
            rationale.append(
                "Mixed regime with stressed volatility or weak pair liquidity -> prefer core mode."
            )
        elif volatility_state == "stressed" or pair_liquidity_state == "weak":
            mode = "risk_off_mode"
            rationale.append(
                "Mixed regime plus stressed volatility/weak liquidity with unhealthy core -> stay risk-off."
            )
        elif (
            bool(hybrid_promotion_signal.get("promoted"))
            and trend_state != "bearish"
            and breadth_state != "weak"
            and volatility_state == "calm"
            and pair_liquidity_state != "weak"
        ):
            mode = "hybrid_guarded_mode"
            if bool(hybrid_promotion_signal.get("profit_first_ok")):
                rationale.append(
                    "Mixed/calm regime and the guarded hybrid clears the performance-first override "
                    f"(Δreturn={_safe_float(hybrid_promotion_signal.get('oos_return_edge'), 0.0):+.4%}, "
                    f"Δsharpe={_safe_float(hybrid_promotion_signal.get('oos_sharpe_edge'), 0.0):+.4f}, "
                    f"val={_safe_float(hybrid.get('val_total_return'), 0.0):+.4%}/{_safe_float(hybrid.get('val_sharpe'), 0.0):.4f}, "
                    f"maxDD {_safe_float(hybrid_promotion_signal.get('hybrid_oos_max_drawdown'), 0.0):.4%} "
                    f"vs {_safe_float(hybrid_promotion_signal.get('balanced_oos_max_drawdown'), 0.0):.4%}) "
                    "-> promote hybrid guarded mode."
                )
            else:
                rationale.append(
                    "Mixed/calm regime and the guarded hybrid materially outperforms balanced "
                    f"(Δreturn={_safe_float(hybrid_promotion_signal.get('oos_return_edge'), 0.0):+.4%}, "
                    f"Δsharpe={_safe_float(hybrid_promotion_signal.get('oos_sharpe_edge'), 0.0):+.4f}, "
                    f"maxDD {_safe_float(hybrid_promotion_signal.get('hybrid_oos_max_drawdown'), 0.0):.4%} "
                    f"vs {_safe_float(hybrid_promotion_signal.get('balanced_oos_max_drawdown'), 0.0):.4%}) "
                    "-> promote hybrid guarded mode."
                )
        elif (
            bool(production_promotion_signal.get("promoted"))
            and trend_state != "bearish"
            and pair_liquidity_state != "weak"
        ):
            mode = "production_guarded_mode"
            rationale.append(
                "Mixed regime and the production-guarded candidate beats balanced on OOS return/sharpe while reducing drawdown "
                f"(Δreturn={_safe_float(production_promotion_signal.get('oos_return_edge'), 0.0):+.4%}, "
                f"Δsharpe={_safe_float(production_promotion_signal.get('oos_sharpe_edge'), 0.0):+.4f}) "
                "-> promote production guarded mode."
            )
        elif overlay_health["healthy"]:
            mode = "balanced_overlay_mode"
            rationale.append(
                "Mixed regime but calm enough for a small overlay -> balanced overlay mode."
            )
        elif bool(soft_health.get("healthy")):
            mode = "core_mode"
            rationale.append("Mixed regime is calm, but the overlay is unhealthy -> use core mode.")
        else:
            mode = "risk_off_mode"
            rationale.append("Mixed regime without any healthy active sleeve -> stay risk-off.")
    else:
        if bool(soft_health.get("healthy")):
            mode = "core_mode"
            rationale.append(
                "Fallback to core mode because aggressive conditions were not confirmed."
            )
        else:
            mode = "risk_off_mode"
            rationale.append(
                "Fallback to risk-off because aggressive conditions were not confirmed and core is unhealthy."
            )

    default_mode_payloads = {
        "risk_off_mode": {"allocation": {"cash": 1.0}},
        "hybrid_guarded_mode": {"allocation": {"hybrid_online_portfolio": 1.0}},
        "production_guarded_mode": {"allocation": {"production_guarded_portfolio": 1.0}},
    }
    allocation = dict(
        (deployment_modes.get(mode) or default_mode_payloads.get(mode) or {}).get("allocation")
        or {}
    )
    return OperatingModeDecision(mode=mode, allocation=allocation, rationale=rationale)


def build_operating_switch_payload(
    *,
    market_judgement_payload: Mapping[str, Any],
    soft_allocator_payload: Mapping[str, Any],
    three_way_allocator_payload: Mapping[str, Any],
    operating_plan_payload: Mapping[str, Any],
    pair_volume_signals: list[SymbolVolumeSignal],
    feature_lookback_days: int,
    as_of_date_override: date | None = None,
    market_judgement_mode: str = "latest",
    balanced_strategy_payload: Mapping[str, Any] | None = None,
    hybrid_portfolio_payload: Mapping[str, Any] | None = None,
    production_guarded_payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    saved_judgement = dict(market_judgement_payload.get("current_judgement") or {})
    normalized_market_judgement_mode = (
        str(market_judgement_mode or "latest").strip().lower() or "latest"
    )
    if normalized_market_judgement_mode == "saved":
        current_judgement = dict(saved_judgement)
    else:
        current_judgement = _load_latest_market_judgement(
            market_judgement_payload=market_judgement_payload,
            feature_lookback_days=max(8, int(feature_lookback_days)),
            as_of_date=as_of_date_override,
        )
    snapshot = dict(current_judgement.get("feature_snapshot") or {})
    soft_current = dict(soft_allocator_payload.get("current_state") or {})
    hard_current = dict(three_way_allocator_payload.get("current_state") or {})
    hybrid_source_metrics = _hybrid_source_sleeve_metrics(hybrid_portfolio_payload)
    soft_current["_allocator_health"] = (
        _allocator_health_from_split_metrics(dict(hybrid_source_metrics["soft_three_way_regime"]))
        if "soft_three_way_regime" in hybrid_source_metrics
        else _allocator_health(soft_allocator_payload)
    )
    hard_current["_allocator_health"] = (
        _allocator_health_from_split_metrics(dict(hybrid_source_metrics["three_way_regime"]))
        if "three_way_regime" in hybrid_source_metrics
        else _allocator_health(three_way_allocator_payload)
    )
    pair_liquidity_state = _pair_liquidity_state(pair_volume_signals)
    decision = recommend_operating_mode(
        current_judgement=current_judgement,
        soft_current_state=soft_current,
        hard_current_state=hard_current,
        operating_plan_payload=operating_plan_payload,
        pair_liquidity_state=pair_liquidity_state,
        balanced_health=_balanced_strategy_health(
            operating_plan_payload=operating_plan_payload,
            balanced_strategy_payload=balanced_strategy_payload,
            hybrid_source_metrics=hybrid_source_metrics,
        ),
        hybrid_health=_hybrid_portfolio_health(hybrid_portfolio_payload),
        production_health=_production_guarded_health(production_guarded_payload),
    )

    return {
        "artifact_kind": "portfolio_operating_switch_recommendation",
        "generated_at": _utc_now_iso(),
        "as_of_date": str(current_judgement.get("date") or snapshot.get("date") or ""),
        "input_paths": {
            "market_judgement": str(market_judgement_payload.get("_path") or ""),
            "soft_allocator": str(soft_allocator_payload.get("_path") or ""),
            "three_way_allocator": str(three_way_allocator_payload.get("_path") or ""),
            "operating_plan": str(operating_plan_payload.get("_path") or ""),
            "hybrid_portfolio": str((hybrid_portfolio_payload or {}).get("_path") or ""),
            "production_guarded": str((production_guarded_payload or {}).get("_path") or ""),
            "market_judgement_mode": normalized_market_judgement_mode,
            "as_of_date_override": as_of_date_override.isoformat()
            if as_of_date_override is not None
            else "",
        },
        "current_market_state": {
            "favored_group": str(current_judgement.get("favored_group") or "mixed"),
            "confidence": _safe_float(current_judgement.get("confidence"), 0.0),
            "trend_state": _trend_state(snapshot),
            "breadth_state": _breadth_state(snapshot),
            "volatility_state": _volatility_state(snapshot),
            "pair_liquidity_state": pair_liquidity_state,
            "feature_snapshot": snapshot,
            "pair_volume_signals": [signal.as_payload() for signal in pair_volume_signals],
            "saved_market_judgement_as_of": str(saved_judgement.get("date") or ""),
            "saved_market_judgement_favored_group": str(
                saved_judgement.get("favored_group") or "mixed"
            ),
            "saved_market_judgement_confidence": _safe_float(
                saved_judgement.get("confidence"), 0.0
            ),
        },
        "allocator_state": {
            "soft_current_state": soft_current,
            "three_way_current_state": hard_current,
            "hybrid_portfolio_health": _hybrid_portfolio_health(hybrid_portfolio_payload),
            "production_guarded_health": _production_guarded_health(production_guarded_payload),
        },
        "recommended_mode": decision.as_payload(),
        "available_modes": dict(operating_plan_payload.get("deployment_modes") or {}),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-profile", choices=("live_current", "reboot_validation"), default="live_current"
    )
    parser.add_argument("--market-judgement-mode", choices=("latest", "saved"), default=None)
    parser.add_argument("--as-of-date", default=None)
    parser.add_argument("--market-judgement-path", type=Path, default=None)
    parser.add_argument("--soft-allocator-path", type=Path, default=None)
    parser.add_argument("--three-way-allocator-path", type=Path, default=None)
    parser.add_argument("--operating-plan-path", type=Path, default=None)
    parser.add_argument("--balanced-strategy-path", type=Path, default=None)
    parser.add_argument("--hybrid-portfolio-path", type=Path, default=None)
    parser.add_argument("--production-guarded-path", type=Path, default=None)
    parser.add_argument("--materialized-root", type=Path, default=None)
    parser.add_argument("--raw-aggtrades-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--pair-timeframe", default=DEFAULT_PAIR_TIMEFRAME)
    parser.add_argument(
        "--pair-volume-lookback-days", type=int, default=DEFAULT_VOLUME_LOOKBACK_DAYS
    )
    parser.add_argument("--feature-lookback-days", type=int, default=DEFAULT_FEATURE_LOOKBACK_DAYS)
    parser.add_argument("--pair-symbol", dest="pair_symbols", action="append", default=[])
    return parser


def _build_markdown(payload: Mapping[str, Any]) -> str:
    state = dict(payload.get("current_market_state") or {})
    decision = dict(payload.get("recommended_mode") or {})
    snapshot = dict(state.get("feature_snapshot") or {})
    input_paths = dict(payload.get("input_paths") or {})
    lines = [
        "# portfolio operating switch recommendation",
        "",
        f"- generated_at: `{payload.get('generated_at')}`",
        f"- as_of_date: `{payload.get('as_of_date')}`",
        "",
        "## Current market state",
        f"- favored_group: `{state.get('favored_group')}`",
        f"- confidence: `{_safe_float(state.get('confidence'), 0.0):.4f}`",
        f"- trend_state: `{state.get('trend_state')}`",
        f"- breadth_state: `{state.get('breadth_state')}`",
        f"- volatility_state: `{state.get('volatility_state')}`",
        f"- pair_liquidity_state: `{state.get('pair_liquidity_state')}`",
        f"- market_judgement_mode: `{input_paths.get('market_judgement_mode')}`",
        f"- as_of_date_override: `{input_paths.get('as_of_date_override')}`",
        f"- saved_market_judgement_as_of: `{state.get('saved_market_judgement_as_of')}`",
        f"- saved_market_judgement_favored_group: `{state.get('saved_market_judgement_favored_group')}`",
        f"- saved_market_judgement_confidence: `{_safe_float(state.get('saved_market_judgement_confidence'), 0.0):.4f}`",
        f"- btc_trend_gap_192: `{_safe_float(snapshot.get('btc_trend_gap_192'), 0.0):.6f}`",
        f"- btc_trend_gap_336: `{_safe_float(snapshot.get('btc_trend_gap_336'), 0.0):.6f}`",
        f"- breadth_ma96: `{_safe_float(snapshot.get('breadth_ma96'), 0.0):.4f}`",
        f"- breadth_ma192: `{_safe_float(snapshot.get('breadth_ma192'), 0.0):.4f}`",
        f"- basket_vol_ratio: `{_safe_float(snapshot.get('basket_vol_ratio'), 0.0):.6f}`",
        "",
        "## Pair liquidity signals",
    ]
    for signal in list(state.get("pair_volume_signals") or []):
        lines.append(
            f"- {signal['symbol']}: state `{signal['state']}` | latest_date `{signal['latest_available_date']}` | comparison `{signal.get('comparison_mode')}` | volume_ratio `{_safe_float(signal.get('volume_ratio'), 0.0):.4f}` | stale_days `{signal.get('stale_days')}`"
        )
    lines += [
        "",
        "## Recommended mode",
        f"- mode: `{decision.get('mode')}`",
        f"- allocation: `{json.dumps(decision.get('allocation') or {}, sort_keys=True)}`",
        "",
        "## Rationale",
        *[f"- {item}" for item in list(decision.get("rationale") or [])],
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    args = _build_parser().parse_args()
    defaults = _profile_defaults(args.artifact_profile)
    market_judgement_path = Path(
        args.market_judgement_path or defaults["market_judgement_path"]
    ).resolve()
    soft_allocator_path = Path(
        args.soft_allocator_path or defaults["soft_allocator_path"]
    ).resolve()
    three_way_allocator_path = Path(
        args.three_way_allocator_path or defaults["three_way_allocator_path"]
    ).resolve()
    operating_plan_path = Path(
        args.operating_plan_path or defaults["operating_plan_path"]
    ).resolve()
    balanced_strategy_path = Path(
        args.balanced_strategy_path or defaults["balanced_strategy_path"]
    ).resolve()
    hybrid_portfolio_path = Path(
        args.hybrid_portfolio_path or defaults["hybrid_portfolio_path"]
    ).resolve()
    production_guarded_path = Path(
        args.production_guarded_path or DEFAULT_PRODUCTION_GUARDED_PATH
    ).resolve()
    raw_aggtrades_root = Path(args.raw_aggtrades_root or defaults["raw_aggtrades_root"]).resolve()
    output_dir = Path(args.output_dir or defaults["output_dir"]).resolve()
    market_judgement_mode = str(args.market_judgement_mode or defaults["market_judgement_mode"])
    as_of_override_token = str(
        args.as_of_date or _profile_as_of_override(args.artifact_profile) or ""
    ).strip()
    as_of_override = _parse_as_of_date(as_of_override_token) if as_of_override_token else None

    market_judgement = _read_json(market_judgement_path)
    market_judgement["_path"] = str(market_judgement_path)
    soft_allocator = _read_json(soft_allocator_path)
    soft_allocator["_path"] = str(soft_allocator_path)
    three_way_allocator = _read_json(three_way_allocator_path)
    three_way_allocator["_path"] = str(three_way_allocator_path)
    operating_plan = _read_json(operating_plan_path)
    operating_plan["_path"] = str(operating_plan_path)
    balanced_strategy = (
        _read_json(balanced_strategy_path) if balanced_strategy_path.exists() else None
    )
    hybrid_portfolio = _read_json(hybrid_portfolio_path) if hybrid_portfolio_path.exists() else None
    if hybrid_portfolio is not None:
        hybrid_portfolio["_path"] = str(hybrid_portfolio_path)
    production_guarded = (
        _read_json(production_guarded_path) if production_guarded_path.exists() else None
    )
    if production_guarded is not None:
        production_guarded["_path"] = str(production_guarded_path)

    if market_judgement_mode == "saved":
        latest_live_judgement = dict(market_judgement.get("current_judgement") or {})
    else:
        latest_live_judgement = _load_latest_market_judgement(
            market_judgement_payload=market_judgement,
            feature_lookback_days=max(8, int(args.feature_lookback_days)),
            as_of_date=as_of_override,
        )
    as_of_date = _parse_as_of_date(dict(latest_live_judgement).get("date") or as_of_override_token)
    pair_symbols = list(args.pair_symbols) or list(DEFAULT_PAIR_SYMBOLS)
    volume_signals = [
        _load_symbol_volume_signal(
            raw_aggtrades_root=raw_aggtrades_root,
            symbol=symbol,
            as_of_date=as_of_date,
            lookback_days=max(2, int(args.pair_volume_lookback_days)),
        )
        for symbol in pair_symbols
    ]

    payload = build_operating_switch_payload(
        market_judgement_payload=market_judgement,
        soft_allocator_payload=soft_allocator,
        three_way_allocator_payload=three_way_allocator,
        operating_plan_payload=operating_plan,
        pair_volume_signals=volume_signals,
        feature_lookback_days=max(8, int(args.feature_lookback_days)),
        as_of_date_override=as_of_override,
        market_judgement_mode=market_judgement_mode,
        balanced_strategy_payload=balanced_strategy,
        hybrid_portfolio_payload=hybrid_portfolio,
        production_guarded_payload=production_guarded,
    )
    payload["available_modes"] = dict(payload.get("available_modes") or {})
    payload["available_modes"].setdefault("risk_off_mode", {"allocation": {"cash": 1.0}})

    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    out_json = output_dir / f"portfolio_operating_switch_{timestamp}.json"
    out_md = output_dir / f"portfolio_operating_switch_{timestamp}.md"
    latest_json = output_dir / "portfolio_operating_switch_latest.json"
    latest_md = output_dir / "portfolio_operating_switch_latest.md"
    out_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default), encoding="utf-8"
    )
    markdown = _build_markdown(payload)
    out_md.write_text(markdown, encoding="utf-8")
    latest_json.write_text(out_json.read_text(encoding="utf-8"), encoding="utf-8")
    latest_md.write_text(markdown, encoding="utf-8")
    print(latest_json)
    print(latest_md)


if __name__ == "__main__":
    main()
