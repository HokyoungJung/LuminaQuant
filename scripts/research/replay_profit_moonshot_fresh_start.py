#!/usr/bin/env python3
"""Fresh-start profit moonshot strategy replay independent of existing incumbents.

This screen intentionally avoids the incumbent ETH shock-reversion and prior
lead-lag sleeves. It starts from refreshed raw-first materialized OHLCV and
Binance support features, then evaluates new cross-sectional/derivatives-context
families under a one-position state machine before any live-equivalent mode can
be considered.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import resource
import sys
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from lumina_quant.configuration import BacktestConfigView, get_default_runtime_config
from lumina_quant.market_data import load_futures_feature_points_from_db
from lumina_quant.portfolio_split_contract import (
    PORTFOLIO_FOLLOWUP_EXPLICIT_BUDGET_BYTES,
    acquire_portfolio_memory_guard,
    memory_policy_payload,
)

BaseConfig = BacktestConfigView(get_default_runtime_config())

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.research.revalidate_live_equivalent_candidates import (  # noqa: E402
    SplitWindow,
    _metrics_from_equity_totals,
    _safe_float,
    _split_windows,
)
from scripts.research.replay_eth_shock_filters import (  # noqa: E402
    _fill_price,
    _materialized_paths,
    _rolling_rms_log_return,
    _rolling_zscore,
    _fmt_float,
    _fmt_pct,
)

DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "var/reports/profit_moonshot_20260501/current_tail_20260507/fresh_overhaul"
)
DEFAULT_SYMBOLS = ("BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "TRX/USDT")
CURRENT_CHAMPION_OOS_RETURN = 0.012181
BASELINE_OOS_RETURN = CURRENT_CHAMPION_OOS_RETURN
SHADOW_OOS_MDD = 0.001778
SUCCESS_SHARPE = 1.0
TARGET_ALLOCATION = 0.008
MAX_ORDER_VALUE = 175.0
HOURLY_PERIODS_PER_YEAR = 365 * 24
RUN_NAME = "profit_moonshot_fresh_start_replay"
CACHE_SCHEMA_VERSION = 2
DEFAULT_PANEL_CACHE_DIR = REPO_ROOT / "var/cache/profit_moonshot_fresh_start"
FEATURE_VALUE_COLUMNS = (
    "funding_rate",
    "open_interest",
    "taker_buy_base_volume",
    "taker_sell_base_volume",
    "taker_buy_quote_volume",
    "taker_sell_quote_volume",
    "liquidation_long_notional",
    "liquidation_short_notional",
)
NON_CALENDAR_SPEC_FAMILIES = frozenset(
    {
        "beta_residual_reversion",
        "calendar_teacher_state_fade",
        "calendar_teacher_state_similarity",
        "dispersion_compression_breakout_unwind",
        "funding_oi_exhaustion_reversal",
        "state_distilled_external_risk_filter",
        "state_distilled_leadership_unwind",
        "state_distilled_crowded_unwind_v2",
        "vol_regime_margin_scaled_momentum",
    }
)
NON_CALENDAR_FIELD_NAMES = (
    "entry_hours",
    "entry_days_of_month",
    "calendar_long_months",
    "calendar_short_months",
)


@dataclass(frozen=True, slots=True)
class FreshSpec:
    name: str
    family: str
    lookback_bars: int
    threshold: float
    hold_bars: int
    cooldown_bars: int
    stop_loss_pct: float
    take_profit_pct: float
    allow_long: bool = True
    allow_short: bool = True
    rv_lookback_bars: int = 24
    max_rv: float = 0.0
    broad_min_abs: float = 0.0
    min_abs_return: float = 0.0
    entry_hours: tuple[int, ...] = ()
    funding_abs_cap: float = 0.0
    funding_rank_min: float = 0.0
    compression_lookback_bars: int = 24
    compression_quantile: float = 0.0
    flow_lookback_bars: int = 0
    flow_threshold: float = 0.0
    flow_persistence_bars: int = 0
    flow_persistence_threshold: float = 0.0
    adaptive_lookback_bars: int = 12
    sharpe_lookback_bars: int = 0
    sharpe_rank_min: float = 0.0
    oi_rank_min: float = 0.0
    long_allocation_scale: float = 1.0
    short_allocation_scale: float = 1.0
    trailing_stop_rv_multiple: float = 0.0
    trailing_stop_floor_pct: float = 0.0
    trailing_stop_cap_pct: float = 0.0
    calendar_long_months: tuple[int, ...] = ()
    calendar_short_months: tuple[int, ...] = ()
    calendar_long_symbol: str = ""
    calendar_short_symbol: str = ""
    primary_symbol: str = ""
    secondary_symbol: str = ""
    entry_days_of_month: tuple[int, ...] = ()
    calendar_veto_resid_z: float = 0.0
    calendar_veto_funding_abs: float = 0.0
    calendar_veto_market_ret_abs: float = 0.0
    calendar_veto_flow_abs: float = 0.0
    spread_hedge_ratio: float = 1.0
    external_risk_max: float = 0.0

    def payload(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "family": self.family,
            "lookback_bars": self.lookback_bars,
            "threshold": self.threshold,
            "hold_bars": self.hold_bars,
            "cooldown_bars": self.cooldown_bars,
            "stop_loss_pct": self.stop_loss_pct,
            "take_profit_pct": self.take_profit_pct,
            "allow_long": self.allow_long,
            "allow_short": self.allow_short,
            "rv_lookback_bars": self.rv_lookback_bars,
            "max_rv": self.max_rv,
            "broad_min_abs": self.broad_min_abs,
            "min_abs_return": self.min_abs_return,
            "entry_hours": list(self.entry_hours),
            "funding_abs_cap": self.funding_abs_cap,
            "funding_rank_min": self.funding_rank_min,
            "compression_lookback_bars": self.compression_lookback_bars,
            "compression_quantile": self.compression_quantile,
            "flow_lookback_bars": self.flow_lookback_bars,
            "flow_threshold": self.flow_threshold,
            "flow_persistence_bars": self.flow_persistence_bars,
            "flow_persistence_threshold": self.flow_persistence_threshold,
            "adaptive_lookback_bars": self.adaptive_lookback_bars,
            "sharpe_lookback_bars": self.sharpe_lookback_bars,
            "sharpe_rank_min": self.sharpe_rank_min,
            "oi_rank_min": self.oi_rank_min,
            "long_allocation_scale": self.long_allocation_scale,
            "short_allocation_scale": self.short_allocation_scale,
            "trailing_stop_rv_multiple": self.trailing_stop_rv_multiple,
            "trailing_stop_floor_pct": self.trailing_stop_floor_pct,
            "trailing_stop_cap_pct": self.trailing_stop_cap_pct,
            "calendar_long_months": list(self.calendar_long_months),
            "calendar_short_months": list(self.calendar_short_months),
            "calendar_long_symbol": self.calendar_long_symbol,
            "calendar_short_symbol": self.calendar_short_symbol,
            "primary_symbol": self.primary_symbol,
            "secondary_symbol": self.secondary_symbol,
            "entry_days_of_month": list(self.entry_days_of_month),
            "calendar_veto_resid_z": self.calendar_veto_resid_z,
            "calendar_veto_funding_abs": self.calendar_veto_funding_abs,
            "calendar_veto_market_ret_abs": self.calendar_veto_market_ret_abs,
            "calendar_veto_flow_abs": self.calendar_veto_flow_abs,
            "spread_hedge_ratio": self.spread_hedge_ratio,
            "external_risk_max": self.external_risk_max,
            "target_allocation": TARGET_ALLOCATION,
            "max_order_value": MAX_ORDER_VALUE,
        }


def _compact(symbol: str) -> str:
    return str(symbol).replace("/", "").upper()


def _column_symbol(column_prefix: str) -> str:
    token = column_prefix.upper()
    if token.endswith("USDT"):
        return f"{token[:-4]}/USDT"
    return token


def _month_iter(start: date, end: date) -> list[date]:
    if end < start:
        return []
    cursor = date(start.year, start.month, 1)
    out: list[date] = []
    while cursor <= end:
        out.append(cursor)
        if cursor.month == 12:
            cursor = date(cursor.year + 1, 1, 1)
        else:
            cursor = date(cursor.year, cursor.month + 1, 1)
    return out


def _month_end(month_start: date) -> date:
    if month_start.month == 12:
        return date(month_start.year + 1, 1, 1) - timedelta(days=1)
    return date(month_start.year, month_start.month + 1, 1) - timedelta(days=1)


def _raw_month_path(*, market_root: Path, exchange: str, symbol: str, month_start: date) -> Path:
    return (
        market_root
        / "market_ohlcv_1s"
        / str(exchange).lower()
        / _compact(symbol)
        / f"{month_start:%Y-%m}.parquet"
    )


def _aggregate_1s_paths_to_hourly(
    *,
    paths: list[str],
    prefix: str,
    start: date,
    end: date,
) -> pl.DataFrame:
    if not paths:
        return pl.DataFrame(
            schema={
                "datetime": pl.Datetime(time_unit="ms"),
                f"{prefix}_open": pl.Float64,
                f"{prefix}_high": pl.Float64,
                f"{prefix}_low": pl.Float64,
                f"{prefix}_close": pl.Float64,
                f"{prefix}_volume": pl.Float64,
            }
        )
    start_dt = datetime.combine(start, datetime.min.time())
    end_dt = datetime.combine(end + timedelta(days=1), datetime.min.time())
    # Collect one month/day at a time.  A single all-period 1s lazy aggregation
    # is fast but peaks around 6.5 GiB for BTC alone on this data set; chunking
    # keeps the whole fresh-start session comfortably below the 8 GiB ceiling.
    return (
        pl.scan_parquet(paths)
        .select(["datetime", "open", "high", "low", "close", "volume"])
        .with_columns(pl.col("datetime").cast(pl.Datetime(time_unit="ms")).alias("datetime"))
        .filter((pl.col("datetime") >= start_dt) & (pl.col("datetime") < end_dt))
        .set_sorted("datetime")
        .group_by_dynamic("datetime", every="1h", period="1h", closed="left", label="left")
        .agg(
            [
                pl.col("open").first().alias(f"{prefix}_open"),
                pl.col("high").max().alias(f"{prefix}_high"),
                pl.col("low").min().alias(f"{prefix}_low"),
                pl.col("close").last().alias(f"{prefix}_close"),
                pl.col("volume").sum().alias(f"{prefix}_volume"),
            ]
        )
        .drop_nulls([f"{prefix}_open", f"{prefix}_close"])
        .collect(engine="streaming")
        .sort("datetime")
    )


def _incomplete_hourly_days(frame: pl.DataFrame, *, start: date, end: date) -> list[date]:
    expected_hours = 24
    observed: dict[date, int] = {}
    if not frame.is_empty() and "datetime" in frame.columns:
        observed = {
            row["date"]: int(row["len"])
            for row in (
                frame.with_columns(pl.col("datetime").dt.date().alias("date"))
                .group_by("date")
                .len()
                .to_dicts()
            )
        }
    missing: list[date] = []
    cursor = start
    while cursor <= end:
        if int(observed.get(cursor, 0)) < expected_hours:
            missing.append(cursor)
        cursor += timedelta(days=1)
    return missing


def _load_daily_materialized_hourly(
    *,
    market_root: Path,
    exchange: str,
    symbol: str,
    prefix: str,
    days: list[date],
) -> list[pl.DataFrame]:
    frames: list[pl.DataFrame] = []
    for day in days:
        frame = _aggregate_1s_paths_to_hourly(
            paths=_materialized_paths(
                market_root=market_root,
                exchange=exchange,
                symbol=symbol,
                timeframe="1s",
                start=day,
                end=day,
            ),
            prefix=prefix,
            start=day,
            end=day,
        )
        if not frame.is_empty():
            frames.append(frame)
        gc.collect()
    return frames


def _load_symbol_hourly(
    *, market_root: Path, exchange: str, symbol: str, start: date, end: date
) -> pl.DataFrame:
    prefix = _compact(symbol).lower()
    frames: list[pl.DataFrame] = []
    for month_start in _month_iter(start, end):
        window_start = max(start, month_start)
        window_end = min(end, _month_end(month_start))
        monthly_path = _raw_month_path(
            market_root=market_root,
            exchange=exchange,
            symbol=symbol,
            month_start=month_start,
        )
        if monthly_path.exists():
            monthly_frame = _aggregate_1s_paths_to_hourly(
                paths=[str(monthly_path)],
                prefix=prefix,
                start=window_start,
                end=window_end,
            )
            frames.append(monthly_frame)
            missing_days = _incomplete_hourly_days(
                monthly_frame, start=window_start, end=window_end
            )
            if missing_days:
                frames.extend(
                    _load_daily_materialized_hourly(
                        market_root=market_root,
                        exchange=exchange,
                        symbol=symbol,
                        prefix=prefix,
                        days=missing_days,
                    )
                )
            gc.collect()
            continue
        frames.append(
            _aggregate_1s_paths_to_hourly(
                paths=_materialized_paths(
                    market_root=market_root,
                    exchange=exchange,
                    symbol=symbol,
                    timeframe="1s",
                    start=window_start,
                    end=window_end,
                ),
                prefix=prefix,
                start=window_start,
                end=window_end,
            )
        )
        gc.collect()
    frames = [frame for frame in frames if not frame.is_empty()]
    if not frames:
        raise RuntimeError(f"no committed raw-first 1s/monthly paths for {symbol} {start}..{end}")
    return (
        pl.concat(frames, how="vertical")
        .sort("datetime")
        .unique(subset=["datetime"], keep="last", maintain_order=True)
    )


def _load_feature_hourly(
    *, market_root: Path, exchange: str, symbol: str, start: date, end: date
) -> tuple[pl.DataFrame, dict[str, Any]]:
    frame = load_futures_feature_points_from_db(
        str(market_root),
        exchange=exchange,
        symbol=symbol,
        start_date=datetime.combine(start, datetime.min.time()),
        end_date=datetime.combine(end + timedelta(days=1), datetime.min.time()),
    )
    compact = _compact(symbol).lower()
    if frame.is_empty():
        return pl.DataFrame({"datetime": []}, schema={"datetime": pl.Datetime(time_unit="ms")}), {
            "symbol": symbol,
            "rows": 0,
            "funding_rows": 0,
            "open_interest_rows": 0,
        }
    aligned = frame
    for column in ("timestamp_ms", *FEATURE_VALUE_COLUMNS):
        if column not in aligned.columns:
            dtype = pl.Int64 if column == "timestamp_ms" else pl.Float64
            aligned = aligned.with_columns(pl.lit(None, dtype=dtype).alias(column))
    aligned = aligned.select(["timestamp_ms", *FEATURE_VALUE_COLUMNS]).filter(
        pl.col("timestamp_ms").is_not_null()
    )
    metadata = {
        "symbol": symbol,
        "rows": int(aligned.height),
        "funding_rows": int(aligned.select(pl.col("funding_rate").is_not_null().sum()).item()),
        "open_interest_rows": int(
            aligned.select(pl.col("open_interest").is_not_null().sum()).item()
        ),
        "taker_flow_rows": int(
            aligned.select(
                (
                    pl.col("taker_buy_quote_volume").is_not_null()
                    | pl.col("taker_sell_quote_volume").is_not_null()
                ).sum()
            ).item()
        ),
    }
    if aligned.is_empty():
        return pl.DataFrame(
            {"datetime": []}, schema={"datetime": pl.Datetime(time_unit="ms")}
        ), metadata
    hourly = (
        aligned.with_columns(
            pl.from_epoch(pl.col("timestamp_ms"), time_unit="ms").alias("datetime")
        )
        .sort("datetime")
        .group_by_dynamic("datetime", every="1h", period="1h", closed="left", label="left")
        .agg(
            [
                pl.col("funding_rate").drop_nulls().last().alias(f"{compact}_funding_rate"),
                pl.col("open_interest").drop_nulls().last().alias(f"{compact}_open_interest"),
                pl.col("taker_buy_base_volume").sum().alias(f"{compact}_taker_buy_base_volume"),
                pl.col("taker_sell_base_volume").sum().alias(f"{compact}_taker_sell_base_volume"),
                pl.col("taker_buy_quote_volume").sum().alias(f"{compact}_taker_buy_quote_volume"),
                pl.col("taker_sell_quote_volume").sum().alias(f"{compact}_taker_sell_quote_volume"),
                pl.col("liquidation_long_notional")
                .sum()
                .alias(f"{compact}_liquidation_long_notional"),
                pl.col("liquidation_short_notional")
                .sum()
                .alias(f"{compact}_liquidation_short_notional"),
            ]
        )
        .sort("datetime")
    )
    return hourly, metadata


def _date_iter(start: date, end: date) -> list[date]:
    if end < start:
        return []
    return [start + timedelta(days=offset) for offset in range((end - start).days + 1)]


def _materialized_source_paths(
    *, market_root: Path, exchange: str, symbol: str, start: date, end: date
) -> list[Path]:
    compact = _compact(symbol)
    paths: list[Path] = []
    for day in _date_iter(start, end):
        day_root = (
            market_root
            / "market_data_materialized"
            / str(exchange).lower()
            / compact
            / "timeframe=1s"
            / f"date={day.isoformat()}"
        )
        manifest_path = day_root / "manifest.json"
        if not manifest_path.exists():
            continue
        paths.append(manifest_path)
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for item in list(manifest.get("data_files") or []):
            path = day_root / str(item)
            if path.exists():
                paths.append(path)
    return paths


def _feature_source_paths(
    *, market_root: Path, exchange: str, symbol: str, start: date, end: date
) -> list[Path]:
    root = (
        market_root
        / "feature_points"
        / f"exchange={str(exchange).lower()}"
        / f"symbol={_compact(symbol)}"
    )
    paths: list[Path] = []
    for day in _date_iter(start, end):
        day_root = root / f"date={day.isoformat()}"
        if day_root.exists():
            paths.extend(sorted(day_root.glob("*.parquet")))
    return paths


def _joined_panel_cache_path(
    *,
    cache_dir: Path,
    market_root: Path,
    exchange: str,
    symbols: list[str],
    start: date,
    end: date,
) -> tuple[Path, dict[str, Any]]:
    source_paths: list[Path] = []
    for symbol in symbols:
        for month_start in _month_iter(start, end):
            monthly_path = _raw_month_path(
                market_root=market_root,
                exchange=exchange,
                symbol=symbol,
                month_start=month_start,
            )
            if monthly_path.exists():
                source_paths.append(monthly_path)
        source_paths.extend(
            _materialized_source_paths(
                market_root=market_root,
                exchange=exchange,
                symbol=symbol,
                start=start,
                end=end,
            )
        )
        source_paths.extend(
            _feature_source_paths(
                market_root=market_root,
                exchange=exchange,
                symbol=symbol,
                start=start,
                end=end,
            )
        )
    digest = hashlib.sha256()
    params = {
        "schema": CACHE_SCHEMA_VERSION,
        "market_root": str(market_root.resolve()),
        "exchange": str(exchange).lower(),
        "symbols": list(symbols),
        "start": start.isoformat(),
        "end": end.isoformat(),
    }
    digest.update(json.dumps(params, sort_keys=True).encode("utf-8"))
    total_bytes = 0
    latest_mtime_ns = 0
    for path in sorted({item.resolve() for item in source_paths}):
        try:
            stat = path.stat()
        except FileNotFoundError:
            continue
        rel = str(path)
        digest.update(rel.encode("utf-8"))
        digest.update(str(int(stat.st_size)).encode("ascii"))
        digest.update(str(int(stat.st_mtime_ns)).encode("ascii"))
        total_bytes += int(stat.st_size)
        latest_mtime_ns = max(latest_mtime_ns, int(stat.st_mtime_ns))
    source_metadata = {
        **params,
        "source_count": len({str(item.resolve()) for item in source_paths if item.exists()}),
        "source_total_bytes": total_bytes,
        "source_latest_mtime_ns": latest_mtime_ns,
        "cache_key": digest.hexdigest(),
    }
    return cache_dir / f"joined_panel_{digest.hexdigest()[:24]}.parquet", source_metadata


def _joined_panel(
    *,
    market_root: Path,
    exchange: str,
    symbols: list[str],
    start: date,
    end: date,
    cache_dir: Path | None = DEFAULT_PANEL_CACHE_DIR,
    refresh_cache: bool = False,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    cache_path: Path | None = None
    cache_metadata: dict[str, Any] = {}
    if cache_dir is not None:
        cache_path, cache_metadata = _joined_panel_cache_path(
            cache_dir=Path(cache_dir),
            market_root=market_root,
            exchange=exchange,
            symbols=symbols,
            start=start,
            end=end,
        )
        metadata_path = cache_path.with_suffix(".json")
        if not refresh_cache and cache_path.exists() and metadata_path.exists():
            panel = pl.read_parquet(cache_path).sort("datetime")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            data_metadata = dict(metadata.get("data_metadata") or {})
            data_metadata["panel_cache"] = {
                **cache_metadata,
                "cache_hit": True,
                "path": str(cache_path),
            }
            return panel, data_metadata

    panels = [
        _load_symbol_hourly(
            market_root=market_root, exchange=exchange, symbol=s, start=start, end=end
        )
        for s in symbols
    ]
    panel = panels[0]
    for frame in panels[1:]:
        panel = panel.join(frame, on="datetime", how="inner")
    feature_meta: list[dict[str, Any]] = []
    for symbol in symbols:
        features, meta = _load_feature_hourly(
            market_root=market_root, exchange=exchange, symbol=symbol, start=start, end=end
        )
        feature_meta.append(meta)
        if not features.is_empty():
            panel = panel.join(features, on="datetime", how="left")
    for symbol in symbols:
        prefix = _compact(symbol).lower()
        for suffix in FEATURE_VALUE_COLUMNS:
            column = f"{prefix}_{suffix}"
            if column not in panel.columns:
                panel = panel.with_columns(pl.lit(None, dtype=pl.Float64).alias(column))
    panel = panel.sort("datetime")
    data_metadata = {"feature_points": feature_meta, "rows": int(panel.height)}
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = cache_path.with_suffix(".tmp.parquet")
        panel.write_parquet(tmp_path)
        tmp_path.replace(cache_path)
        metadata_path = cache_path.with_suffix(".json")
        metadata_path.write_text(
            json.dumps(
                {
                    "data_metadata": data_metadata,
                    "panel_cache": {
                        **cache_metadata,
                        "cache_hit": False,
                        "path": str(cache_path),
                    },
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        data_metadata["panel_cache"] = {
            **cache_metadata,
            "cache_hit": False,
            "path": str(cache_path),
        }
    return panel, data_metadata


def _array_for(arrays: dict[str, Any], key: str) -> np.ndarray | None:
    raw = arrays.get(key)
    if raw is None:
        return None
    return np.asarray(raw, dtype=float)


def _array_value(arrays: dict[str, Any], key: str, idx: int) -> float:
    raw = arrays.get(key)
    if raw is None:
        return np.nan
    return float(raw[idx])


def _calendar_target_match(symbol: str, target: str) -> bool:
    if not target:
        return True
    return _compact(symbol) == _compact(target)


def _resolve_calendar_symbol(symbols: tuple[str, ...], target: str) -> str:
    if not target:
        return ""
    target_compact = _compact(target)
    for symbol in symbols:
        if _compact(symbol) == target_compact:
            return symbol
    return ""


def _non_calendar_spec_calendar_fields(spec: FreshSpec) -> list[str]:
    """Return calendar entry fields that would make a state-distilled spec invalid."""
    if spec.family not in NON_CALENDAR_SPEC_FAMILIES:
        return []
    violations: list[str] = []
    for field_name in NON_CALENDAR_FIELD_NAMES:
        if tuple(getattr(spec, field_name, ()) or ()):
            violations.append(field_name)
    return violations


def _to_float_array(panel: pl.DataFrame, column: str) -> np.ndarray:
    if column not in panel.columns:
        return np.full(panel.height, np.nan, dtype=float)
    return np.asarray(
        [float(v) if v is not None else np.nan for v in panel[column].to_list()], dtype=float
    )


def _pct_change(values: np.ndarray, lookback: int) -> np.ndarray:
    out = np.full(values.shape, np.nan, dtype=float)
    if lookback <= 0:
        return out
    base = values[:-lookback]
    latest = values[lookback:]
    valid = np.isfinite(base) & np.isfinite(latest) & (base > 0.0)
    out[lookback:][valid] = latest[valid] / base[valid] - 1.0
    return out


def _rolling_mean(values: np.ndarray, lookback: int) -> np.ndarray:
    out = np.full(values.shape, np.nan, dtype=float)
    if lookback <= 0:
        return out
    clean = np.where(np.isfinite(values), values, 0.0)
    counts = np.where(np.isfinite(values), 1.0, 0.0)
    csum = np.cumsum(np.insert(clean, 0, 0.0))
    ccnt = np.cumsum(np.insert(counts, 0, 0.0))
    for idx in range(lookback - 1, values.size):
        count = ccnt[idx + 1] - ccnt[idx + 1 - lookback]
        if count > 0:
            out[idx] = float((csum[idx + 1] - csum[idx + 1 - lookback]) / count)
    return out


def _rolling_sum(values: np.ndarray, lookback: int) -> np.ndarray:
    out = np.full(values.shape, np.nan, dtype=float)
    if lookback <= 0:
        return out
    clean = np.where(np.isfinite(values), values, 0.0)
    counts = np.where(np.isfinite(values), 1.0, 0.0)
    csum = np.cumsum(np.insert(clean, 0, 0.0))
    ccnt = np.cumsum(np.insert(counts, 0, 0.0))
    for idx in range(lookback - 1, values.size):
        count = ccnt[idx + 1] - ccnt[idx + 1 - lookback]
        if count > 0:
            out[idx] = float(csum[idx + 1] - csum[idx + 1 - lookback])
    return out


def _ffill(values: np.ndarray) -> np.ndarray:
    out = np.asarray(values, dtype=float).copy()
    last = np.nan
    for idx, value in enumerate(out):
        if math.isfinite(float(value)):
            last = float(value)
        else:
            out[idx] = last
    return out


def _build_arrays(panel: pl.DataFrame, symbols: list[str]) -> dict[str, Any]:
    datetimes = panel["datetime"].to_list()
    arrays: dict[str, Any] = {
        "datetime": datetimes,
        "timestamp": np.asarray(
            [int(dt.replace(tzinfo=UTC).timestamp()) for dt in datetimes], dtype=np.int64
        ),
        "symbols": tuple(symbols),
        "symbol_prefixes": tuple(_symbol_prefix(symbol) for symbol in symbols),
    }
    for column in panel.columns:
        if column.startswith("external_"):
            arrays[column] = _to_float_array(panel, column)
    close_stack: list[np.ndarray] = []
    for symbol in symbols:
        prefix = _compact(symbol).lower()
        for column in ("open", "high", "low", "close", "volume", *FEATURE_VALUE_COLUMNS):
            key = f"{prefix}_{column}"
            arrays[key] = _to_float_array(panel, key)
        close = arrays[f"{prefix}_close"]
        close_stack.append(close)
        for lookback in (3, 6, 12, 24, 48, 72, 168, 336):
            arrays[f"{prefix}_ret_{lookback}h"] = _pct_change(close, lookback)
        for lookback in (24, 48, 72):
            arrays[f"{prefix}_rv_{lookback}h"] = _rolling_rms_log_return(close, lookback)
        arrays[f"{prefix}_rv_24h_mean_72h"] = _rolling_mean(arrays[f"{prefix}_rv_24h"], 72)
        arrays[f"{prefix}_funding_ffill"] = _ffill(arrays[f"{prefix}_funding_rate"])
        arrays[f"{prefix}_open_interest_ffill"] = _ffill(arrays[f"{prefix}_open_interest"])
        buy_quote = arrays[f"{prefix}_taker_buy_quote_volume"]
        sell_quote = arrays[f"{prefix}_taker_sell_quote_volume"]
        flow_den = buy_quote + sell_quote
        arrays[f"{prefix}_flow_imbalance_1h"] = np.divide(
            buy_quote - sell_quote,
            flow_den,
            out=np.full_like(flow_den, np.nan, dtype=float),
            where=np.isfinite(flow_den) & (flow_den > 0.0),
        )
        for lookback in (3, 6, 12, 24):
            buy_sum = _rolling_sum(buy_quote, lookback)
            sell_sum = _rolling_sum(sell_quote, lookback)
            den = buy_sum + sell_sum
            flow = np.divide(
                buy_sum - sell_sum,
                den,
                out=np.full_like(den, np.nan, dtype=float),
                where=np.isfinite(den) & (den > 0.0),
            )
            arrays[f"{prefix}_flow_imbalance_{lookback}h"] = flow
            arrays[f"{prefix}_flow_z_{lookback}h"] = _rolling_zscore(flow, max(24, lookback * 6))
        for lookback in (3, 6, 12, 24):
            oi = arrays[f"{prefix}_open_interest_ffill"]
            oi_delta = np.full_like(oi, np.nan, dtype=float)
            if oi.size > lookback:
                oi_delta[lookback:] = np.divide(
                    oi[lookback:] - oi[:-lookback],
                    lookback,
                    out=np.full_like(oi[lookback:], np.nan, dtype=float),
                    where=np.isfinite(oi[:-lookback]) & np.isfinite(oi[lookback:]),
                )
            arrays[f"{prefix}_oi_delta_{lookback}h"] = oi_delta
    if not close_stack:
        return arrays
    stacked = np.vstack(close_stack)
    market_close = np.nanmean(stacked, axis=0)
    arrays["market_close"] = market_close
    for lookback in (3, 6, 12, 24, 48, 72, 168, 336):
        market_ret = _pct_change(market_close, lookback)
        arrays[f"market_ret_{lookback}h"] = market_ret
        for symbol in symbols:
            prefix = _compact(symbol).lower()
            residual = arrays[f"{prefix}_ret_{lookback}h"] - market_ret
            arrays[f"{prefix}_resid_{lookback}h"] = residual
            arrays[f"{prefix}_resid_z_{lookback}h"] = _rolling_zscore(
                residual, max(24, lookback * 4)
            )
    return arrays


def _join_external_state(
    panel: pl.DataFrame, external_state_csv: str | Path
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Join lagged daily external market-state features to the hourly crypto panel.

    The external CSV is expected to be pre-lagged by the fetch script so the
    feature available for a UTC date only uses information from prior dates.
    This function keeps the join fail-closed: malformed or empty files do not
    create synthetic signal and are reported in metadata instead.
    """
    path = Path(external_state_csv)
    metadata: dict[str, Any] = {
        "path": str(path),
        "enabled": bool(str(external_state_csv).strip()),
        "joined": False,
        "feature_columns": [],
        "rows": 0,
    }
    if not str(external_state_csv).strip():
        return panel, metadata
    if not path.exists():
        metadata["error"] = "external_state_csv_missing"
        return panel, metadata
    try:
        external = pl.read_csv(path, try_parse_dates=True)
    except Exception as exc:  # pragma: no cover - defensive metadata path
        metadata["error"] = f"external_state_csv_read_failed:{exc}"
        return panel, metadata
    date_column = "effective_date" if "effective_date" in external.columns else "date"
    if date_column not in external.columns:
        metadata["error"] = "external_state_csv_missing_date_column"
        return panel, metadata
    feature_columns = [column for column in external.columns if column.startswith("external_")]
    if not feature_columns:
        metadata["error"] = "external_state_csv_missing_external_columns"
        return panel, metadata
    external = (
        external.select(
            [pl.col(date_column).cast(pl.Date).alias("external_join_date"), *feature_columns]
        )
        .unique(subset=["external_join_date"], keep="last")
        .sort("external_join_date")
    )
    joined = (
        panel.with_columns(pl.col("datetime").dt.date().alias("external_join_date"))
        .join(external, on="external_join_date", how="left")
        .drop("external_join_date")
    )
    metadata.update(
        {
            "joined": True,
            "feature_columns": feature_columns,
            "rows": int(external.height),
            "first_effective_date": (
                external["external_join_date"][0].isoformat() if external.height else None
            ),
            "last_effective_date": (
                external["external_join_date"][-1].isoformat() if external.height else None
            ),
        }
    )
    return joined, metadata


def _hour_utc(dt: datetime) -> int:
    return int(dt.replace(tzinfo=UTC).hour)


def _day_of_month_utc(dt: datetime) -> int:
    return int(dt.replace(tzinfo=UTC).day)


def _symbol_prefix(symbol: str) -> str:
    return _compact(symbol).lower()


def _side_allocation_scale(spec: FreshSpec, side: str) -> float:
    if side == "SHORT":
        return max(0.0, float(spec.short_allocation_scale))
    return max(0.0, float(spec.long_allocation_scale))


def _entry_stop_pct(spec: FreshSpec, arrays: dict[str, Any], prefix: str, idx: int) -> float:
    if spec.trailing_stop_rv_multiple <= 0.0:
        return float(spec.stop_loss_pct)
    rv_key = f"{prefix}_rv_{spec.rv_lookback_bars}h"
    rv = float(arrays.get(rv_key, np.full_like(arrays[f"{prefix}_close"], np.nan))[idx])
    if not math.isfinite(rv) or rv <= 0.0:
        return float(spec.stop_loss_pct)
    stop_pct = rv * float(spec.trailing_stop_rv_multiple)
    if spec.trailing_stop_floor_pct > 0.0:
        stop_pct = max(stop_pct, float(spec.trailing_stop_floor_pct))
    if spec.trailing_stop_cap_pct > 0.0:
        stop_pct = min(stop_pct, float(spec.trailing_stop_cap_pct))
    return max(0.0, float(stop_pct))


def _state_exposure_scale(
    spec: FreshSpec, arrays: dict[str, Any], symbol: str, side: str, idx: int
) -> float:
    """Scale signal exposure for explicit vol/regime/margin-buffer research specs."""
    if spec.family != "vol_regime_margin_scaled_momentum":
        return 1.0
    prefix = _symbol_prefix(symbol)
    scale = 1.0
    if spec.max_rv > 0.0:
        rv = _array_value(arrays, f"{prefix}_rv_{spec.rv_lookback_bars}h", idx)
        if not math.isfinite(rv) or rv <= 0.0:
            return 0.0
        scale *= min(1.0, max(0.25, float(spec.max_rv) / rv))
    market_lookback = max(1, int(spec.adaptive_lookback_bars or spec.lookback_bars))
    market_ret = _array_value(arrays, f"market_ret_{market_lookback}h", idx)
    if spec.broad_min_abs > 0.0 and math.isfinite(market_ret):
        adverse = -market_ret if side == "LONG" else market_ret
        if adverse > 0.0:
            scale *= max(0.30, 1.0 - min(0.70, adverse / float(spec.broad_min_abs)))
    if spec.funding_abs_cap > 0.0:
        funding = _array_value(arrays, f"{prefix}_funding_ffill", idx)
        if not math.isfinite(funding):
            return 0.0
        funding_load = abs(funding) / float(spec.funding_abs_cap)
        if funding_load > 1.0:
            scale *= max(0.25, 1.0 / funding_load)
    notional_scale = _side_allocation_scale(spec, side)
    if spec.sharpe_rank_min > 0.0 and notional_scale > spec.sharpe_rank_min:
        scale *= max(0.25, float(spec.sharpe_rank_min) / notional_scale)
    return max(0.0, min(1.0, scale))


def _entry_window_block_reason(spec: FreshSpec, arrays: dict[str, Any], idx: int) -> str:
    dt = arrays["datetime"][idx]
    if spec.entry_hours and _hour_utc(dt) not in spec.entry_hours:
        return "entry_hour_block"
    if spec.entry_days_of_month and _day_of_month_utc(dt) not in spec.entry_days_of_month:
        return "entry_day_block"
    return ""


def _calendar_veto_reason(
    spec: FreshSpec,
    arrays: dict[str, Any],
    *,
    symbol: str,
    side: str,
    prefix: str,
    idx: int,
) -> str:
    if spec.calendar_veto_resid_z > 0.0:
        resid_z = _array_value(arrays, f"{prefix}_resid_z_{spec.lookback_bars}h", idx)
        limit = float(spec.calendar_veto_resid_z)
        if side == "LONG" and math.isfinite(resid_z) and resid_z <= -limit:
            return "calendar_residual_veto"
        if side == "SHORT" and math.isfinite(resid_z) and resid_z >= limit:
            return "calendar_residual_veto"
    if spec.calendar_veto_funding_abs > 0.0:
        funding = _array_value(arrays, f"{prefix}_funding_ffill", idx)
        limit = float(spec.calendar_veto_funding_abs)
        if side == "LONG" and math.isfinite(funding) and funding <= -limit:
            return "calendar_funding_conflict_veto"
        if side == "SHORT" and math.isfinite(funding) and funding >= limit:
            return "calendar_funding_conflict_veto"
    if spec.calendar_veto_market_ret_abs > 0.0:
        market_lookback = max(1, int(spec.adaptive_lookback_bars or spec.lookback_bars))
        market_ret = _array_value(arrays, f"market_ret_{market_lookback}h", idx)
        if math.isfinite(market_ret) and abs(market_ret) >= float(
            spec.calendar_veto_market_ret_abs
        ):
            return "calendar_market_extreme_veto"
    if spec.calendar_veto_flow_abs > 0.0:
        flow_lookback = max(1, int(spec.flow_lookback_bars or 6))
        flow = _array_value(arrays, f"{prefix}_flow_imbalance_{flow_lookback}h", idx)
        limit = float(spec.calendar_veto_flow_abs)
        if side == "LONG" and math.isfinite(flow) and flow <= -limit:
            return "calendar_flow_exhaustion_veto"
        if side == "SHORT" and math.isfinite(flow) and flow >= limit:
            return "calendar_flow_exhaustion_veto"
    _ = symbol
    return ""


def _state_distilled_leadership_unwind_signal(
    spec: FreshSpec, arrays: dict[str, Any], idx: int
) -> tuple[str, str, str]:
    """Select the current leader or crowded laggard without calendar inputs.

    This family is a causal state proxy for the rejected calendar tuple.  The
    old calendar sleeve is treated as a teacher only at the design level: live
    selection uses cross-sectional return/residual rank, flow confirmation,
    market regime, and optional OI/funding crowding.  No month, day, or fixed
    symbol target is consulted.
    """
    blocked = _entry_window_block_reason(spec, arrays, idx)
    if blocked:
        return "", "", blocked

    symbols = tuple(arrays["symbols"])
    prefixes = tuple(
        arrays.get("symbol_prefixes") or tuple(_symbol_prefix(symbol) for symbol in symbols)
    )
    lookback = max(1, int(spec.lookback_bars))
    fast_lookback = max(1, int(spec.adaptive_lookback_bars or spec.lookback_bars))
    flow_lookback = max(1, int(spec.flow_lookback_bars or 6))
    oi_lookback = max(1, int(spec.sharpe_lookback_bars or spec.flow_lookback_bars or 6))
    min_abs_return = max(0.0, float(spec.min_abs_return))
    resid_threshold = max(0.0, float(spec.threshold))
    min_rank_gap = max(0.0, float(spec.sharpe_rank_min))
    market_ret = _array_value(arrays, f"market_ret_{lookback}h", idx)
    fast_market_ret = _array_value(arrays, f"market_ret_{fast_lookback}h", idx)

    states: list[dict[str, float | str]] = []
    for symbol, prefix in zip(symbols, prefixes, strict=True):
        # Treat BTC as the broad market/risk-regime anchor for this family,
        # not as the alt leadership/unwind target.  That keeps the teacher
        # distillation focused on the calendar tuple's alt/major behavior
        # without pinning to a fixed month or a fixed TRX/ETH target.
        if _compact(symbol) == "BTCUSDT":
            continue
        close = _array_value(arrays, f"{prefix}_close", idx)
        if not math.isfinite(close) or close <= 0.0:
            continue
        if spec.max_rv > 0.0:
            rv = _array_value(arrays, f"{prefix}_rv_{spec.rv_lookback_bars}h", idx)
            if not math.isfinite(rv) or rv > float(spec.max_rv):
                continue
        funding = _array_value(arrays, f"{prefix}_funding_ffill", idx)
        if spec.funding_abs_cap > 0.0 and (
            not math.isfinite(funding) or abs(funding) > float(spec.funding_abs_cap)
        ):
            continue
        ret = _array_value(arrays, f"{prefix}_ret_{lookback}h", idx)
        fast_ret = _array_value(arrays, f"{prefix}_ret_{fast_lookback}h", idx)
        resid_z = _array_value(arrays, f"{prefix}_resid_z_{lookback}h", idx)
        flow = _array_value(arrays, f"{prefix}_flow_imbalance_{flow_lookback}h", idx)
        oi_delta = _array_value(arrays, f"{prefix}_oi_delta_{oi_lookback}h", idx)
        if not math.isfinite(ret) or not math.isfinite(resid_z):
            continue
        flow_score = 0.0 if not math.isfinite(flow) else float(flow)
        funding_score = 0.0 if not math.isfinite(funding) else float(funding)
        oi_score = 0.0 if not math.isfinite(oi_delta) else float(oi_delta)
        fast_score = float(fast_ret) if math.isfinite(fast_ret) else float(ret)
        states.append(
            {
                "symbol": symbol,
                "ret": float(ret),
                "fast_ret": fast_score,
                "resid_z": float(resid_z),
                "flow": flow_score,
                "funding": funding_score,
                "oi_delta": oi_score,
                "leadership_score": float(ret)
                + 0.01 * float(resid_z)
                + 0.05 * max(0.0, flow_score),
                "unwind_score": abs(float(ret))
                + 0.01 * abs(float(resid_z))
                + 0.05 * max(0.0, -flow_score)
                + max(0.0, funding_score)
                + 0.10 * max(0.0, oi_score),
            }
        )

    if len(states) < 2:
        return "", "", "state_distilled_universe_too_small"

    leader = max(states, key=lambda row: float(row["leadership_score"]))
    laggard = max(states, key=lambda row: float(row["unwind_score"]))
    strongest_resid = max(float(row["resid_z"]) for row in states)
    weakest_resid = min(float(row["resid_z"]) for row in states)
    rank_gap = strongest_resid - weakest_resid
    if rank_gap < min_rank_gap:
        return "", "", "state_distilled_rank_gap_too_small"

    market_is_bad = (
        math.isfinite(market_ret) and market_ret <= -float(spec.broad_min_abs)
        if spec.broad_min_abs > 0.0
        else math.isfinite(market_ret) and market_ret < 0.0
    )
    market_is_good_enough = (
        not math.isfinite(market_ret)
        or spec.broad_min_abs <= 0.0
        or market_ret >= -float(spec.broad_min_abs)
    )
    fast_market_is_good_enough = not math.isfinite(fast_market_ret) or fast_market_ret >= -float(
        spec.broad_min_abs
    )

    long_ok = (
        spec.allow_long
        and market_is_good_enough
        and fast_market_is_good_enough
        and float(leader["ret"]) >= min_abs_return
        and float(leader["fast_ret"]) >= 0.0
        and float(leader["resid_z"]) >= resid_threshold
        and (spec.flow_threshold <= 0.0 or float(leader["flow"]) >= float(spec.flow_threshold))
    )
    short_crowding_ok = (
        spec.oi_rank_min <= 0.0
        or float(laggard["oi_delta"]) >= float(spec.oi_rank_min)
        or (
            spec.funding_rank_min > 0.0
            and float(laggard["funding"]) >= float(spec.funding_rank_min)
        )
    )
    short_ok = (
        spec.allow_short
        and (
            market_is_bad
            or float(laggard["ret"]) <= -min_abs_return * 1.5
            or float(laggard["flow"]) <= -float(spec.flow_threshold)
        )
        and float(laggard["ret"]) <= -min_abs_return
        and float(laggard["fast_ret"]) <= 0.0
        and float(laggard["resid_z"]) <= -resid_threshold
        and (spec.flow_threshold <= 0.0 or float(laggard["flow"]) <= -float(spec.flow_threshold))
        and short_crowding_ok
    )

    if long_ok and short_ok:
        market_score = float(market_ret) if math.isfinite(market_ret) else 0.0
        long_score = float(leader["leadership_score"]) + max(0.0, 0.5 * market_score)
        short_score = float(laggard["unwind_score"]) + max(0.0, -0.5 * market_score)
        if long_score >= short_score:
            return str(leader["symbol"]), "LONG", ""
        return str(laggard["symbol"]), "SHORT", ""
    if long_ok:
        return str(leader["symbol"]), "LONG", ""
    if short_ok:
        return str(laggard["symbol"]), "SHORT", ""
    return "", "", "signal_missing"


def _residual_dispersion(
    arrays: dict[str, Any], symbols: tuple[str, ...], lookback: int, idx: int
) -> float:
    values: list[float] = []
    for symbol in symbols:
        prefix = _symbol_prefix(symbol)
        resid_z = _array_value(arrays, f"{prefix}_resid_z_{lookback}h", idx)
        if math.isfinite(resid_z):
            values.append(float(resid_z))
    if len(values) < 2:
        return np.nan
    return max(values) - min(values)


def _state_distilled_crowded_unwind_v2_signal(
    spec: FreshSpec, arrays: dict[str, Any], idx: int
) -> tuple[str, str, str]:
    """Short a crowded non-BTC leader when observable state starts unwinding.

    This is the second non-calendar state-distilled pass.  It keeps the old
    calendar tuple as a design teacher only, then trades a current market-state
    mechanism: high residual/return leadership plus funding/OI crowding, with
    either fast-momentum weakening, taker-flow reversal, or residual-rank-gap
    compression.  Month/day/hour entry fields are explicitly rejected.
    """
    if _non_calendar_spec_calendar_fields(spec):
        return "", "", "non_calendar_calendar_field_block"

    symbols = tuple(arrays["symbols"])
    prefixes = tuple(
        arrays.get("symbol_prefixes") or tuple(_symbol_prefix(symbol) for symbol in symbols)
    )
    lookback = max(1, int(spec.lookback_bars))
    fast_lookback = max(1, int(spec.adaptive_lookback_bars or 24))
    flow_lookback = max(1, int(spec.flow_lookback_bars or 6))
    oi_lookback = max(1, int(spec.sharpe_lookback_bars or spec.flow_lookback_bars or 12))
    resid_threshold = max(0.0, float(spec.threshold))
    min_abs_return = max(0.0, float(spec.min_abs_return))
    funding_min = max(0.0, float(spec.funding_rank_min))
    oi_min = max(0.0, float(spec.oi_rank_min))
    flow_reversal_min = max(0.0, float(spec.flow_threshold))
    compression_gap_max = max(0.0, float(spec.sharpe_rank_min))
    fast_market_ret = _array_value(arrays, f"market_ret_{fast_lookback}h", idx)
    broad_ok = (
        spec.broad_min_abs <= 0.0
        or not math.isfinite(fast_market_ret)
        or fast_market_ret <= float(spec.broad_min_abs)
    )

    states: list[dict[str, float | str]] = []
    for symbol, prefix in zip(symbols, prefixes, strict=True):
        if _compact(symbol) == "BTCUSDT":
            continue
        close = _array_value(arrays, f"{prefix}_close", idx)
        if not math.isfinite(close) or close <= 0.0:
            continue
        if spec.max_rv > 0.0:
            rv = _array_value(arrays, f"{prefix}_rv_{spec.rv_lookback_bars}h", idx)
            if not math.isfinite(rv) or rv > float(spec.max_rv):
                continue
        ret = _array_value(arrays, f"{prefix}_ret_{lookback}h", idx)
        fast_ret = _array_value(arrays, f"{prefix}_ret_{fast_lookback}h", idx)
        resid_z = _array_value(arrays, f"{prefix}_resid_z_{lookback}h", idx)
        funding = _array_value(arrays, f"{prefix}_funding_ffill", idx)
        oi_delta = _array_value(arrays, f"{prefix}_oi_delta_{oi_lookback}h", idx)
        flow = _array_value(arrays, f"{prefix}_flow_imbalance_{flow_lookback}h", idx)
        if not math.isfinite(ret) or not math.isfinite(resid_z):
            continue
        states.append(
            {
                "symbol": symbol,
                "ret": float(ret),
                "fast_ret": float(fast_ret) if math.isfinite(fast_ret) else float(ret),
                "resid_z": float(resid_z),
                "funding": float(funding) if math.isfinite(funding) else 0.0,
                "oi_delta": float(oi_delta) if math.isfinite(oi_delta) else 0.0,
                "flow": float(flow) if math.isfinite(flow) else 0.0,
            }
        )

    if len(states) < 2:
        return "", "", "state_crowded_universe_too_small"

    states.sort(key=lambda row: float(row["resid_z"]), reverse=True)
    top_resid = float(states[0]["resid_z"])
    second_resid = float(states[1]["resid_z"])
    leadership_gap = top_resid - second_resid
    candidates: list[tuple[float, str, str]] = []
    for row in states:
        ret = float(row["ret"])
        fast_ret = float(row["fast_ret"])
        resid_z = float(row["resid_z"])
        funding = float(row["funding"])
        oi_delta = float(row["oi_delta"])
        flow = float(row["flow"])
        overheated = ret >= min_abs_return and resid_z >= resid_threshold
        crowding_checks = []
        if funding_min > 0.0:
            crowding_checks.append(funding >= funding_min)
        if oi_min > 0.0:
            crowding_checks.append(oi_delta >= oi_min)
        crowded = any(crowding_checks) if crowding_checks else True
        fast_weakening = fast_ret <= max(0.0, ret * 0.50)
        flow_reversal = flow_reversal_min > 0.0 and flow <= -flow_reversal_min
        gap_compression = compression_gap_max > 0.0 and leadership_gap <= compression_gap_max
        if (
            spec.allow_short
            and broad_ok
            and overheated
            and crowded
            and (fast_weakening or flow_reversal or gap_compression)
        ):
            score = (
                resid_z
                + abs(ret) * 10.0
                + max(0.0, funding) * 1_000.0
                + max(0.0, oi_delta)
                + max(0.0, -flow)
                + (0.25 if gap_compression else 0.0)
            )
            candidates.append((score, str(row["symbol"]), "SHORT"))

    if not candidates:
        return "", "", "signal_missing"
    candidates.sort(reverse=True, key=lambda item: item[0])
    _, symbol, side = candidates[0]
    return symbol, side, ""


def _external_risk_score(arrays: dict[str, Any], idx: int) -> float:
    """Return lagged external risk-off score when an external panel is joined."""
    for key in (
        "external_risk_off_score_lag1",
        "external_risk_off_score",
        "external_vix_z_lag1",
        "external_vix_z",
    ):
        value = _array_value(arrays, key, idx)
        if math.isfinite(value):
            return float(value)
    return 0.0


def _calendar_teacher_state_similarity_signal(
    spec: FreshSpec, arrays: dict[str, Any], idx: int
) -> tuple[str, str, str]:
    """Trade market states that resemble the rejected calendar teacher without dates.

    The invalid calendar tuple is used only as a research teacher: it suggested
    that outsized returns came from a mix of alt leadership, major-coin laggard
    unwind, and risk-regime timing.  This live-valid proxy consumes only
    observable state at the bar: cross-sectional residual rank, slow/fast
    returns, taker flow, funding/OI crowding, broad-market return, realized vol,
    and optional lagged external risk state.  It never reads month/day/hour
    entry fields or fixed calendar target symbols.
    """
    if _non_calendar_spec_calendar_fields(spec):
        return "", "", "non_calendar_calendar_field_block"

    symbols = tuple(arrays["symbols"])
    prefixes = tuple(
        arrays.get("symbol_prefixes") or tuple(_symbol_prefix(symbol) for symbol in symbols)
    )
    lookback = max(1, int(spec.lookback_bars))
    fast_lookback = max(1, int(spec.adaptive_lookback_bars or 24))
    flow_lookback = max(1, int(spec.flow_lookback_bars or 6))
    oi_lookback = max(1, int(spec.sharpe_lookback_bars or spec.flow_lookback_bars or 12))
    min_abs_return = max(0.0, float(spec.min_abs_return))
    resid_threshold = max(0.0, float(spec.threshold))
    min_rank_gap = max(0.0, float(spec.sharpe_rank_min))
    flow_threshold = max(0.0, float(spec.flow_threshold))
    market_ret = _array_value(arrays, f"market_ret_{lookback}h", idx)
    fast_market_ret = _array_value(arrays, f"market_ret_{fast_lookback}h", idx)
    external_risk = _external_risk_score(arrays, idx)

    states: list[dict[str, float | str]] = []
    for symbol, prefix in zip(symbols, prefixes, strict=True):
        if _compact(symbol) == "BTCUSDT":
            continue
        close = _array_value(arrays, f"{prefix}_close", idx)
        if not math.isfinite(close) or close <= 0.0:
            continue
        if spec.max_rv > 0.0:
            rv = _array_value(arrays, f"{prefix}_rv_{spec.rv_lookback_bars}h", idx)
            if not math.isfinite(rv) or rv > float(spec.max_rv):
                continue
        ret = _array_value(arrays, f"{prefix}_ret_{lookback}h", idx)
        fast_ret = _array_value(arrays, f"{prefix}_ret_{fast_lookback}h", idx)
        resid_z = _array_value(arrays, f"{prefix}_resid_z_{lookback}h", idx)
        flow = _array_value(arrays, f"{prefix}_flow_imbalance_{flow_lookback}h", idx)
        funding = _array_value(arrays, f"{prefix}_funding_ffill", idx)
        oi_delta = _array_value(arrays, f"{prefix}_oi_delta_{oi_lookback}h", idx)
        if not math.isfinite(ret) or not math.isfinite(resid_z):
            continue
        flow_score = float(flow) if math.isfinite(flow) else 0.0
        fast_score = float(fast_ret) if math.isfinite(fast_ret) else float(ret)
        funding_score = float(funding) if math.isfinite(funding) else 0.0
        oi_score = float(oi_delta) if math.isfinite(oi_delta) else 0.0
        normalized_ret = float(ret) / max(min_abs_return, 0.001)
        normalized_fast = fast_score / max(min_abs_return, 0.001)
        states.append(
            {
                "symbol": symbol,
                "ret": float(ret),
                "fast_ret": fast_score,
                "resid_z": float(resid_z),
                "flow": flow_score,
                "funding": funding_score,
                "oi_delta": oi_score,
                "long_score": (
                    float(resid_z)
                    + 0.25 * normalized_ret
                    + 0.20 * max(0.0, normalized_fast)
                    + 0.70 * max(0.0, flow_score)
                    - 0.20 * max(0.0, external_risk)
                    - 500.0 * max(0.0, funding_score)
                ),
                "short_score": (
                    -float(resid_z)
                    - 0.25 * normalized_ret
                    + 0.20 * max(0.0, -normalized_fast)
                    + 0.70 * max(0.0, -flow_score)
                    + 0.20 * max(0.0, external_risk)
                    + 0.10 * max(0.0, oi_score)
                    + 500.0 * max(0.0, funding_score)
                ),
                "crowded_unwind_score": (
                    float(resid_z)
                    + 0.25 * max(0.0, normalized_ret)
                    + 0.20 * max(0.0, external_risk)
                    + 0.70 * max(0.0, -flow_score)
                    + 0.10 * max(0.0, oi_score)
                    + 500.0 * max(0.0, funding_score)
                ),
            }
        )

    if len(states) < 2:
        return "", "", "teacher_state_universe_too_small"

    strongest_resid = max(float(row["resid_z"]) for row in states)
    weakest_resid = min(float(row["resid_z"]) for row in states)
    rank_gap = strongest_resid - weakest_resid
    if rank_gap < min_rank_gap:
        return "", "", "teacher_state_rank_gap_too_small"

    broad_ok_for_longs = (
        not math.isfinite(market_ret)
        or spec.broad_min_abs <= 0.0
        or market_ret >= -float(spec.broad_min_abs)
    ) and (not math.isfinite(fast_market_ret) or fast_market_ret >= -float(spec.broad_min_abs))
    if spec.external_risk_max > 0.0 and external_risk > float(spec.external_risk_max):
        broad_ok_for_longs = False
    broad_supports_shorts = (
        math.isfinite(market_ret) and market_ret <= -float(spec.broad_min_abs)
        if spec.broad_min_abs > 0.0
        else math.isfinite(market_ret) and market_ret < 0.0
    ) or external_risk > 0.0

    long_candidates: list[tuple[float, str, str]] = []
    short_candidates: list[tuple[float, str, str]] = []
    for row in states:
        ret = float(row["ret"])
        fast_ret = float(row["fast_ret"])
        resid_z = float(row["resid_z"])
        flow = float(row["flow"])
        funding = float(row["funding"])
        oi_delta = float(row["oi_delta"])
        if (
            spec.allow_long
            and broad_ok_for_longs
            and ret >= min_abs_return
            and fast_ret >= -0.25 * min_abs_return
            and resid_z >= resid_threshold
            and (flow_threshold <= 0.0 or flow >= -0.5 * flow_threshold)
            and (
                spec.funding_abs_cap <= 0.0
                or not math.isfinite(funding)
                or funding <= spec.funding_abs_cap
            )
        ):
            long_candidates.append((float(row["long_score"]), str(row["symbol"]), "LONG"))

        laggard_break = (
            ret <= -min_abs_return
            and fast_ret <= 0.25 * min_abs_return
            and resid_z <= -resid_threshold
        )
        crowded_break = (
            ret >= min_abs_return
            and resid_z >= resid_threshold
            and (
                fast_ret <= max(0.0, ret * 0.35)
                or flow <= -flow_threshold
                or (spec.funding_rank_min > 0.0 and funding >= spec.funding_rank_min)
                or (spec.oi_rank_min > 0.0 and oi_delta >= spec.oi_rank_min)
            )
        )
        if (
            spec.allow_short
            and (broad_supports_shorts or laggard_break or crowded_break)
            and (laggard_break or crowded_break)
            and (flow_threshold <= 0.0 or flow <= flow_threshold)
        ):
            score_key = "crowded_unwind_score" if crowded_break else "short_score"
            short_candidates.append((float(row[score_key]), str(row["symbol"]), "SHORT"))

    candidates = long_candidates + short_candidates
    if not candidates:
        return "", "", "signal_missing"
    candidates.sort(reverse=True, key=lambda item: item[0])
    _, symbol, side = candidates[0]
    return symbol, side, ""


def _calendar_teacher_state_fade_signal(
    spec: FreshSpec, arrays: dict[str, Any], idx: int
) -> tuple[str, str, str]:
    """Fade teacher-like crowded states when train/validation prove the teacher sign decays."""
    symbol, side, reason = _calendar_teacher_state_similarity_signal(spec, arrays, idx)
    if reason:
        return "", "", reason
    if side == "LONG" and spec.allow_short:
        return symbol, "SHORT", ""
    if side == "SHORT" and spec.allow_long:
        return symbol, "LONG", ""
    return "", "", "teacher_fade_side_blocked"


def _state_distilled_external_risk_filter_signal(
    spec: FreshSpec, arrays: dict[str, Any], idx: int
) -> tuple[str, str, str]:
    """Apply lagged external risk-state gating to state-distilled leadership/unwind."""
    symbol, side, reason = _state_distilled_leadership_unwind_signal(spec, arrays, idx)
    if reason:
        return "", "", reason
    risk_max = float(spec.external_risk_max)
    if risk_max <= 0.0:
        return symbol, side, ""
    external_risk = _external_risk_score(arrays, idx)
    if side == "LONG" and external_risk > risk_max:
        return "", "", "external_risk_long_block"
    if side == "SHORT" and external_risk < -risk_max:
        return "", "", "external_risk_short_block"
    return symbol, side, ""


def _candidate_signal(spec: FreshSpec, arrays: dict[str, Any], idx: int) -> tuple[str, str, str]:
    blocked = _entry_window_block_reason(spec, arrays, idx)
    if blocked:
        return "", "", blocked
    symbols = tuple(arrays["symbols"])
    prefixes = tuple(
        arrays.get("symbol_prefixes") or tuple(_symbol_prefix(symbol) for symbol in symbols)
    )
    lookback = int(spec.lookback_bars)
    market_ret = _array_value(arrays, f"market_ret_{lookback}h", idx)
    if spec.family == "state_distilled_leadership_unwind":
        return _state_distilled_leadership_unwind_signal(spec, arrays, idx)
    if spec.family == "state_distilled_crowded_unwind_v2":
        return _state_distilled_crowded_unwind_v2_signal(spec, arrays, idx)
    if spec.family == "state_distilled_external_risk_filter":
        return _state_distilled_external_risk_filter_signal(spec, arrays, idx)
    if spec.family == "calendar_teacher_state_similarity":
        return _calendar_teacher_state_similarity_signal(spec, arrays, idx)
    if spec.family == "calendar_teacher_state_fade":
        return _calendar_teacher_state_fade_signal(spec, arrays, idx)
    if (
        spec.family != "calendar_rotation"
        and spec.broad_min_abs > 0.0
        and (not math.isfinite(market_ret) or abs(market_ret) < spec.broad_min_abs)
    ):
        return "", "", "broad_move_missing"
    signal_month = int(arrays["datetime"][idx].replace(tzinfo=UTC).month)

    candidates: list[tuple[float, str, str]] = []
    veto_reason = ""
    for symbol, prefix in zip(symbols, prefixes, strict=True):
        close = float(arrays[f"{prefix}_close"][idx])
        if not math.isfinite(close) or close <= 0.0:
            continue
        rv_key = f"{prefix}_rv_{spec.rv_lookback_bars}h"
        if spec.max_rv > 0.0:
            rv = _array_value(arrays, rv_key, idx)
            if not math.isfinite(rv) or rv > spec.max_rv:
                continue
        ret = _array_value(arrays, f"{prefix}_ret_{lookback}h", idx)

        if spec.family == "calendar_rotation":
            if signal_month in spec.calendar_long_months:
                if (
                    spec.allow_long
                    and _calendar_target_match(symbol, spec.calendar_long_symbol)
                    and math.isfinite(ret)
                    and ret >= spec.min_abs_return
                ):
                    reason = _calendar_veto_reason(
                        spec, arrays, symbol=symbol, side="LONG", prefix=prefix, idx=idx
                    )
                    if reason:
                        veto_reason = veto_reason or reason
                    else:
                        candidates.append((ret, symbol, "LONG"))
            elif (
                signal_month in spec.calendar_short_months
                and spec.allow_short
                and _calendar_target_match(symbol, spec.calendar_short_symbol)
                and math.isfinite(ret)
                and ret <= -spec.threshold
            ):
                reason = _calendar_veto_reason(
                    spec, arrays, symbol=symbol, side="SHORT", prefix=prefix, idx=idx
                )
                if reason:
                    veto_reason = veto_reason or reason
                else:
                    candidates.append((abs(ret), symbol, "SHORT"))
            continue

        funding = _array_value(arrays, f"{prefix}_funding_ffill", idx)
        if spec.funding_abs_cap > 0.0 and (
            not math.isfinite(funding) or abs(funding) > spec.funding_abs_cap
        ):
            continue
        if spec.family == "residual_reversion":
            resid_z = _array_value(arrays, f"{prefix}_resid_z_{lookback}h", idx)
            if (
                spec.allow_long
                and math.isfinite(resid_z)
                and resid_z <= -spec.threshold
                and ret <= -spec.min_abs_return
            ):
                candidates.append((abs(resid_z), symbol, "LONG"))
            if (
                spec.allow_short
                and math.isfinite(resid_z)
                and resid_z >= spec.threshold
                and ret >= spec.min_abs_return
            ):
                candidates.append((abs(resid_z), symbol, "SHORT"))
        elif spec.family == "residual_momentum":
            resid_z = _array_value(arrays, f"{prefix}_resid_z_{lookback}h", idx)
            if (
                spec.allow_long
                and math.isfinite(resid_z)
                and resid_z >= spec.threshold
                and ret >= spec.min_abs_return
            ):
                candidates.append((abs(resid_z) + abs(ret), symbol, "LONG"))
            if (
                spec.allow_short
                and math.isfinite(resid_z)
                and resid_z <= -spec.threshold
                and ret <= -spec.min_abs_return
            ):
                candidates.append((abs(resid_z) + abs(ret), symbol, "SHORT"))
        elif spec.family == "state_momentum_proxy":
            primary_symbol = spec.primary_symbol or "TRXUSDT"
            secondary_symbol = spec.secondary_symbol
            resid_z = _array_value(arrays, f"{prefix}_resid_z_{lookback}h", idx)
            flow_lookback = max(1, int(spec.flow_lookback_bars or 6))
            flow = _array_value(arrays, f"{prefix}_flow_imbalance_{flow_lookback}h", idx)
            if not math.isfinite(ret) or not math.isfinite(resid_z):
                continue
            regime_lookback = max(0, int(spec.adaptive_lookback_bars or 0))
            regime_ret = np.nan
            regime_resid_z = np.nan
            if regime_lookback > lookback:
                regime_ret = _array_value(arrays, f"{prefix}_ret_{regime_lookback}h", idx)
                regime_resid_z = _array_value(arrays, f"{prefix}_resid_z_{regime_lookback}h", idx)
                if not math.isfinite(regime_ret):
                    continue
            flow_score = 0.0 if not math.isfinite(flow) else flow
            if (
                spec.allow_long
                and _calendar_target_match(symbol, primary_symbol)
                and ret >= spec.min_abs_return
                and resid_z >= spec.threshold
                and (
                    regime_lookback <= lookback
                    or (
                        regime_ret >= spec.min_abs_return
                        and (
                            spec.sharpe_rank_min <= 0.0
                            or (
                                math.isfinite(regime_resid_z)
                                and regime_resid_z >= spec.sharpe_rank_min
                            )
                        )
                    )
                )
                and (spec.flow_threshold <= 0.0 or flow_score >= spec.flow_threshold)
            ):
                candidates.append((abs(resid_z) + abs(ret) + max(0.0, flow_score), symbol, "LONG"))
            if (
                spec.allow_short
                and (not secondary_symbol or _calendar_target_match(symbol, secondary_symbol))
                and ret <= -spec.min_abs_return
                and resid_z <= -spec.threshold
                and (
                    regime_lookback <= lookback
                    or (
                        regime_ret <= -spec.min_abs_return
                        and (
                            spec.sharpe_rank_min <= 0.0
                            or (
                                math.isfinite(regime_resid_z)
                                and regime_resid_z <= -spec.sharpe_rank_min
                            )
                        )
                    )
                )
                and (spec.flow_threshold <= 0.0 or flow_score <= -spec.flow_threshold)
            ):
                candidates.append(
                    (abs(resid_z) + abs(ret) + max(0.0, -flow_score), symbol, "SHORT")
                )
        elif spec.family == "cross_momentum":
            if not math.isfinite(ret) or abs(ret) < spec.min_abs_return:
                continue
            resid_z = _array_value(arrays, f"{prefix}_resid_z_{lookback}h", idx)
            if spec.allow_long and market_ret > spec.broad_min_abs and resid_z >= spec.threshold:
                candidates.append((resid_z, symbol, "LONG"))
            if spec.allow_short and market_ret < -spec.broad_min_abs and resid_z <= -spec.threshold:
                candidates.append((abs(resid_z), symbol, "SHORT"))
        elif spec.family in {"cross_sectional_sharpe_rank", "cross_sectional_sharpe_reversal"}:
            if not math.isfinite(ret):
                continue
            sharpe_lb = max(2, int(spec.sharpe_lookback_bars or spec.lookback_bars))
            window_start = max(0, idx + 1 - sharpe_lb)
            sharpe_scores: list[tuple[str, float]] = []
            for candidate_symbol in symbols:
                candidate_prefix = _symbol_prefix(candidate_symbol)
                key = f"{candidate_prefix}_ret_{sharpe_lb}h"
                if key not in arrays:
                    continue
                hist = np.asarray(arrays[key][window_start : idx + 1], dtype=float)
                finite = np.isfinite(hist)
                if finite.sum() == 0:
                    continue
                vals = hist[finite]
                vol = float(np.std(vals))
                mean_ret = float(np.mean(vals))
                if not math.isfinite(vol) or vol <= 0.0:
                    sharpe_scores.append((candidate_symbol, mean_ret))
                else:
                    sharpe_scores.append((candidate_symbol, mean_ret / vol))
            if len(sharpe_scores) < 2:
                continue
            sharpe_scores.sort(key=lambda item: item[1], reverse=True)
            top = sharpe_scores[0]
            bottom = sharpe_scores[-1]
            rank_spread = float(top[1] - bottom[1])
            if abs(ret) < spec.min_abs_return or rank_spread < spec.sharpe_rank_min:
                continue
            if spec.family == "cross_sectional_sharpe_reversal":
                if spec.allow_short and symbol == top[0]:
                    candidates.append((rank_spread + abs(ret), symbol, "SHORT"))
                if spec.allow_long and symbol == bottom[0]:
                    candidates.append((rank_spread + abs(ret), symbol, "LONG"))
            else:
                if spec.allow_long and symbol == top[0]:
                    candidates.append((top[1] + abs(ret), symbol, "LONG"))
                if spec.allow_short and symbol == bottom[0]:
                    candidates.append((abs(bottom[1]) + abs(ret), symbol, "SHORT"))
        elif spec.family == "funding_carry_fade":
            if not math.isfinite(funding):
                continue
            resid_z = _array_value(arrays, f"{prefix}_resid_z_{lookback}h", idx)
            if (
                spec.allow_short
                and funding >= spec.funding_rank_min
                and resid_z >= spec.threshold
                and ret >= spec.min_abs_return
            ):
                candidates.append((abs(funding) + abs(resid_z) / 100.0, symbol, "SHORT"))
            if (
                spec.allow_long
                and funding <= -spec.funding_rank_min
                and resid_z <= -spec.threshold
                and ret <= -spec.min_abs_return
            ):
                candidates.append((abs(funding) + abs(resid_z) / 100.0, symbol, "LONG"))
        elif spec.family == "funding_carry_momentum":
            if not math.isfinite(funding):
                continue
            resid_z = _array_value(arrays, f"{prefix}_resid_z_{lookback}h", idx)
            if (
                spec.allow_long
                and funding >= spec.funding_rank_min
                and resid_z >= spec.threshold
                and ret >= spec.min_abs_return
            ):
                candidates.append((abs(funding) + abs(resid_z) / 100.0 + abs(ret), symbol, "LONG"))
            if (
                spec.allow_short
                and funding <= -spec.funding_rank_min
                and resid_z <= -spec.threshold
                and ret <= -spec.min_abs_return
            ):
                candidates.append((abs(funding) + abs(resid_z) / 100.0 + abs(ret), symbol, "SHORT"))
        elif spec.family == "funding_oi_carry_fade":
            oi = _array_value(
                arrays, f"{prefix}_oi_delta_{max(int(spec.sharpe_lookback_bars), 1)}h", idx
            )
            if not math.isfinite(funding) or not math.isfinite(oi):
                continue
            if (
                spec.allow_short
                and funding >= spec.funding_rank_min
                and oi >= spec.oi_rank_min
                and ret >= spec.min_abs_return
            ):
                candidates.append((abs(funding) + abs(oi), symbol, "SHORT"))
            if (
                spec.allow_long
                and funding <= -spec.funding_rank_min
                and oi <= -spec.oi_rank_min
                and ret <= -spec.min_abs_return
            ):
                candidates.append((abs(funding) + abs(oi), symbol, "LONG"))
        elif spec.family == "funding_oi_exhaustion_reversal":
            oi = _array_value(
                arrays, f"{prefix}_oi_delta_{max(int(spec.sharpe_lookback_bars), 1)}h", idx
            )
            flow = _array_value(arrays, f"{prefix}_flow_imbalance_{spec.flow_lookback_bars}h", idx)
            fast_ret = _array_value(
                arrays, f"{prefix}_ret_{max(int(spec.adaptive_lookback_bars), 1)}h", idx
            )
            resid_z = _array_value(arrays, f"{prefix}_resid_z_{lookback}h", idx)
            if not all(math.isfinite(value) for value in (funding, oi, fast_ret, resid_z, flow)):
                continue
            long_crowd = funding >= spec.funding_rank_min and oi >= spec.oi_rank_min
            short_crowd = funding <= -spec.funding_rank_min and oi <= -spec.oi_rank_min
            long_exhausted = (
                ret >= spec.min_abs_return
                and resid_z >= spec.threshold
                and (fast_ret <= 0.0 or flow <= -spec.flow_threshold)
            )
            short_exhausted = (
                ret <= -spec.min_abs_return
                and resid_z <= -spec.threshold
                and (fast_ret >= 0.0 or flow >= spec.flow_threshold)
            )
            if spec.allow_short and long_crowd and long_exhausted:
                candidates.append(
                    (abs(funding) * 1_000.0 + abs(oi) + abs(resid_z), symbol, "SHORT")
                )
            if spec.allow_long and short_crowd and short_exhausted:
                candidates.append((abs(funding) * 1_000.0 + abs(oi) + abs(resid_z), symbol, "LONG"))
        elif spec.family == "beta_residual_reversion":
            fast_ret = _array_value(
                arrays, f"{prefix}_ret_{max(int(spec.adaptive_lookback_bars), 1)}h", idx
            )
            flow = _array_value(arrays, f"{prefix}_flow_imbalance_{spec.flow_lookback_bars}h", idx)
            beta = float(spec.spread_hedge_ratio or 1.0)
            beta_residual = ret - beta * market_ret
            if not math.isfinite(beta_residual) or not math.isfinite(fast_ret):
                continue
            flow_score = 0.0 if not math.isfinite(flow) else float(flow)
            if (
                spec.allow_long
                and beta_residual <= -spec.threshold
                and fast_ret >= -spec.min_abs_return
                and (spec.flow_threshold <= 0.0 or flow_score >= spec.flow_threshold)
            ):
                candidates.append((abs(beta_residual) + max(0.0, flow_score), symbol, "LONG"))
            if (
                spec.allow_short
                and beta_residual >= spec.threshold
                and fast_ret <= spec.min_abs_return
                and (spec.flow_threshold <= 0.0 or flow_score <= -spec.flow_threshold)
            ):
                candidates.append((abs(beta_residual) + max(0.0, -flow_score), symbol, "SHORT"))
        elif spec.family == "flow_momentum":
            flow = _array_value(arrays, f"{prefix}_flow_imbalance_{spec.flow_lookback_bars}h", idx)
            if not math.isfinite(ret) or not math.isfinite(flow) or abs(ret) < spec.min_abs_return:
                continue
            if spec.allow_long and ret > 0.0 and flow >= spec.flow_threshold:
                candidates.append((abs(flow) + abs(ret), symbol, "LONG"))
            if spec.allow_short and ret < 0.0 and flow <= -spec.flow_threshold:
                candidates.append((abs(flow) + abs(ret), symbol, "SHORT"))
        elif spec.family == "flow_exhaustion_fade":
            flow = _array_value(arrays, f"{prefix}_flow_imbalance_{spec.flow_lookback_bars}h", idx)
            if not math.isfinite(ret) or not math.isfinite(flow) or abs(ret) < spec.min_abs_return:
                continue
            if spec.allow_short and ret > 0.0 and flow >= spec.flow_threshold:
                candidates.append((abs(flow) + abs(ret), symbol, "SHORT"))
            if spec.allow_long and ret < 0.0 and flow <= -spec.flow_threshold:
                candidates.append((abs(flow) + abs(ret), symbol, "LONG"))
        elif spec.family == "flow_imbalance_persistence":
            flow = _array_value(arrays, f"{prefix}_flow_imbalance_{spec.flow_lookback_bars}h", idx)
            if not math.isfinite(flow) or not math.isfinite(ret) or abs(ret) < spec.min_abs_return:
                continue
            persist = max(2, int(spec.flow_persistence_bars))
            flow_key = f"{prefix}_flow_imbalance_{spec.flow_lookback_bars}h"
            raw_flow_hist = arrays.get(flow_key)
            if raw_flow_hist is None:
                continue
            flow_hist = np.asarray(raw_flow_hist, dtype=float)
            if flow_hist.size < idx + 1:
                continue
            history = flow_hist[max(0, idx + 1 - persist) : idx + 1]
            if history.size < persist:
                continue
            if spec.allow_long and np.all(history >= spec.flow_persistence_threshold):
                candidates.append((abs(flow) + abs(ret), symbol, "LONG"))
            elif spec.allow_short and np.all(history <= -spec.flow_persistence_threshold):
                candidates.append((abs(flow) + abs(ret), symbol, "SHORT"))
        elif spec.family == "flow_imbalance_exhaustion":
            flow = _array_value(arrays, f"{prefix}_flow_imbalance_{spec.flow_lookback_bars}h", idx)
            if not math.isfinite(flow) or not math.isfinite(ret) or abs(ret) < spec.min_abs_return:
                continue
            if (
                spec.allow_long
                and flow <= -spec.flow_persistence_threshold
                and ret <= -spec.min_abs_return
            ):
                candidates.append((abs(flow) + abs(ret), symbol, "LONG"))
            if (
                spec.allow_short
                and flow >= spec.flow_persistence_threshold
                and ret >= spec.min_abs_return
            ):
                candidates.append((abs(flow) + abs(ret), symbol, "SHORT"))
        elif spec.family == "residual_reversion_flow_confirmed":
            resid_z = _array_value(arrays, f"{prefix}_resid_z_{lookback}h", idx)
            flow = _array_value(arrays, f"{prefix}_flow_imbalance_{spec.flow_lookback_bars}h", idx)
            if not math.isfinite(resid_z) or not math.isfinite(flow):
                continue
            if (
                spec.allow_long
                and resid_z <= -spec.threshold
                and ret <= -spec.min_abs_return
                and flow >= spec.flow_threshold
            ):
                candidates.append((abs(resid_z) + abs(flow), symbol, "LONG"))
            if (
                spec.allow_short
                and resid_z >= spec.threshold
                and ret >= spec.min_abs_return
                and flow <= -spec.flow_threshold
            ):
                candidates.append((abs(resid_z) + abs(flow), symbol, "SHORT"))
        elif spec.family == "adaptive_trend":
            trend_key = f"market_ret_{spec.adaptive_lookback_bars}h"
            if trend_key not in arrays:
                continue
            trend = _array_value(arrays, trend_key, idx)
            if not math.isfinite(trend) or abs(trend) < spec.threshold:
                continue
            if spec.allow_long and trend > 0.0 and ret >= spec.min_abs_return:
                candidates.append((abs(trend) + abs(ret), symbol, "LONG"))
            if spec.allow_short and trend < 0.0 and ret <= -spec.min_abs_return:
                candidates.append((abs(trend) + abs(ret), symbol, "SHORT"))
        elif spec.family == "adaptive_trend_fade":
            trend_key = f"market_ret_{spec.adaptive_lookback_bars}h"
            if trend_key not in arrays:
                continue
            trend = _array_value(arrays, trend_key, idx)
            if not math.isfinite(trend) or abs(trend) < spec.threshold:
                continue
            if spec.allow_short and trend > 0.0 and ret >= spec.min_abs_return:
                candidates.append((abs(trend) + abs(ret), symbol, "SHORT"))
            if spec.allow_long and trend < 0.0 and ret <= -spec.min_abs_return:
                candidates.append((abs(trend) + abs(ret), symbol, "LONG"))
        elif spec.family == "compression_breakout":
            comp_key = f"{prefix}_rv_{spec.rv_lookback_bars}h"
            rv = _array_value(arrays, comp_key, idx)
            rv_mean_key = f"{prefix}_rv_24h_mean_72h"
            rv_mean = _array_value(arrays, rv_mean_key, idx)
            if not math.isfinite(rv) or not math.isfinite(rv_mean) or rv_mean <= 0.0:
                continue
            if rv / rv_mean > spec.compression_quantile:
                continue
            if spec.allow_long and ret >= spec.threshold:
                candidates.append((abs(ret), symbol, "LONG"))
            if spec.allow_short and ret <= -spec.threshold:
                candidates.append((abs(ret), symbol, "SHORT"))
        elif spec.family == "compression_breakout_fade":
            comp_key = f"{prefix}_rv_{spec.rv_lookback_bars}h"
            rv = _array_value(arrays, comp_key, idx)
            rv_mean_key = f"{prefix}_rv_24h_mean_72h"
            rv_mean = _array_value(arrays, rv_mean_key, idx)
            if not math.isfinite(rv) or not math.isfinite(rv_mean) or rv_mean <= 0.0:
                continue
            if rv / rv_mean > spec.compression_quantile:
                continue
            if spec.allow_short and ret >= spec.threshold:
                candidates.append((abs(ret), symbol, "SHORT"))
            if spec.allow_long and ret <= -spec.threshold:
                candidates.append((abs(ret), symbol, "LONG"))
        elif spec.family == "compression_expansion_downside_short":
            comp_key = f"{prefix}_rv_{spec.rv_lookback_bars}h"
            rv = _array_value(arrays, comp_key, idx)
            rv_mean_key = f"{prefix}_rv_24h_mean_72h"
            rv_mean = _array_value(arrays, rv_mean_key, idx)
            if not math.isfinite(rv) or not math.isfinite(rv_mean) or rv_mean <= 0.0:
                continue
            if rv / rv_mean > spec.compression_quantile:
                continue
            if spec.allow_short and ret <= -spec.threshold:
                candidates.append((abs(ret), symbol, "SHORT"))
        elif spec.family == "dispersion_compression_breakout_unwind":
            dispersion = _residual_dispersion(arrays, symbols, lookback, idx)
            flow = _array_value(arrays, f"{prefix}_flow_imbalance_{spec.flow_lookback_bars}h", idx)
            resid_z = _array_value(arrays, f"{prefix}_resid_z_{lookback}h", idx)
            if (
                not math.isfinite(dispersion)
                or dispersion > spec.compression_quantile
                or not math.isfinite(flow)
                or not math.isfinite(resid_z)
            ):
                continue
            market_up = market_ret >= spec.broad_min_abs
            market_down = market_ret <= -spec.broad_min_abs
            continuation_long = (
                ret >= spec.threshold and resid_z > 0.0 and flow >= spec.flow_threshold
            )
            continuation_short = (
                ret <= -spec.threshold and resid_z < 0.0 and flow <= -spec.flow_threshold
            )
            unwind_short = ret >= spec.threshold and resid_z > 0.0 and flow <= -spec.flow_threshold
            unwind_long = ret <= -spec.threshold and resid_z < 0.0 and flow >= spec.flow_threshold
            if spec.allow_long and ((market_up and continuation_long) or unwind_long):
                candidates.append((abs(ret) + abs(resid_z) / 10.0 + abs(flow), symbol, "LONG"))
            if spec.allow_short and ((market_down and continuation_short) or unwind_short):
                candidates.append((abs(ret) + abs(resid_z) / 10.0 + abs(flow), symbol, "SHORT"))
        elif spec.family == "vol_regime_margin_scaled_momentum":
            resid_z = _array_value(arrays, f"{prefix}_resid_z_{lookback}h", idx)
            if not math.isfinite(resid_z) or not math.isfinite(ret):
                continue
            if spec.allow_long and ret >= spec.min_abs_return and resid_z >= spec.threshold:
                candidates.append((abs(ret) + abs(resid_z), symbol, "LONG"))
            if spec.allow_short and ret <= -spec.min_abs_return and resid_z <= -spec.threshold:
                candidates.append((abs(ret) + abs(resid_z), symbol, "SHORT"))
    if not candidates:
        return "", "", veto_reason or "signal_missing"
    candidates.sort(reverse=True, key=lambda item: item[0])
    _, symbol, side = candidates[0]
    return symbol, side, ""


def _calendar_spread_signal(
    spec: FreshSpec, arrays: dict[str, Any], idx: int
) -> tuple[str, str, str, str]:
    blocked = _entry_window_block_reason(spec, arrays, idx)
    if blocked:
        return "", "", "", blocked
    symbols = tuple(arrays["symbols"])
    long_symbol = _resolve_calendar_symbol(symbols, spec.calendar_long_symbol)
    short_symbol = _resolve_calendar_symbol(symbols, spec.calendar_short_symbol)
    if not long_symbol or not short_symbol or long_symbol == short_symbol:
        return "", "", "", "calendar_spread_symbol_missing"

    signal_month = int(arrays["datetime"][idx].replace(tzinfo=UTC).month)
    long_prefix = _symbol_prefix(long_symbol)
    short_prefix = _symbol_prefix(short_symbol)
    long_close = _array_value(arrays, f"{long_prefix}_close", idx)
    short_close = _array_value(arrays, f"{short_prefix}_close", idx)
    if (
        not math.isfinite(long_close)
        or long_close <= 0.0
        or not math.isfinite(short_close)
        or short_close <= 0.0
    ):
        return "", "", "", "calendar_spread_price_missing"

    lookback = int(spec.lookback_bars)
    long_ret = _array_value(arrays, f"{long_prefix}_ret_{lookback}h", idx)
    short_ret = _array_value(arrays, f"{short_prefix}_ret_{lookback}h", idx)
    if not math.isfinite(long_ret) or not math.isfinite(short_ret):
        return "", "", "", "signal_missing"
    hedge = max(0.0, float(spec.spread_hedge_ratio))
    spread_ret = long_ret - hedge * short_ret
    threshold = max(0.0, float(spec.threshold))
    min_abs_return = max(threshold, float(spec.min_abs_return))

    if (
        signal_month in spec.calendar_long_months
        and spec.allow_long
        and spread_ret >= threshold
        and (long_ret >= min_abs_return or short_ret <= -min_abs_return)
    ):
        return long_symbol, short_symbol, "LONG_SPREAD", ""
    if (
        signal_month in spec.calendar_short_months
        and spec.allow_short
        and spread_ret <= -threshold
        and (long_ret <= -min_abs_return or short_ret >= min_abs_return)
    ):
        return long_symbol, short_symbol, "SHORT_SPREAD", ""
    return "", "", "", "signal_missing"


def _residual_pair_spread_signal(
    spec: FreshSpec, arrays: dict[str, Any], idx: int
) -> tuple[str, str, str, str]:
    """Select a market-neutral residual mean-reversion pair.

    The rule is intentionally train/validation-selectable and OOS-blind: at a
    bar it longs the most negative cross-sectional residual and shorts the most
    positive residual when both legs exceed the configured z threshold.
    """
    blocked = _entry_window_block_reason(spec, arrays, idx)
    if blocked:
        return "", "", "", blocked
    if not spec.allow_long or not spec.allow_short:
        return "", "", "", "pair_requires_both_sides"

    lookback = int(spec.lookback_bars)
    threshold = max(0.0, float(spec.threshold))
    symbols = tuple(arrays["symbols"])
    prefixes = tuple(
        arrays.get("symbol_prefixes") or tuple(_symbol_prefix(symbol) for symbol in symbols)
    )
    long_leg: tuple[float, str] | None = None
    short_leg: tuple[float, str] | None = None

    for symbol, prefix in zip(symbols, prefixes, strict=True):
        close = _array_value(arrays, f"{prefix}_close", idx)
        if not math.isfinite(close) or close <= 0.0:
            continue
        rv_key = f"{prefix}_rv_{spec.rv_lookback_bars}h"
        if spec.max_rv > 0.0:
            rv = _array_value(arrays, rv_key, idx)
            if not math.isfinite(rv) or rv > spec.max_rv:
                continue
        resid_z = _array_value(arrays, f"{prefix}_resid_z_{lookback}h", idx)
        if not math.isfinite(resid_z):
            continue
        if resid_z <= -threshold and (long_leg is None or resid_z < long_leg[0]):
            long_leg = (resid_z, symbol)
        if resid_z >= threshold and (short_leg is None or resid_z > short_leg[0]):
            short_leg = (resid_z, symbol)

    if long_leg is None or short_leg is None:
        return "", "", "", "signal_missing"
    long_symbol = long_leg[1]
    short_symbol = short_leg[1]
    if long_symbol == short_symbol:
        return "", "", "", "pair_symbol_overlap"
    spread_z = short_leg[0] - long_leg[0]
    min_spread_z = threshold * 2.0
    if spread_z < min_spread_z:
        return "", "", "", "pair_spread_z_too_small"
    return long_symbol, short_symbol, "LONG_SPREAD", ""


def _residual_pair_momentum_spread_signal(
    spec: FreshSpec, arrays: dict[str, Any], idx: int
) -> tuple[str, str, str, str]:
    """Select a market-neutral residual momentum pair.

    This is the inverse of residual mean reversion: it longs the strongest
    positive residual and shorts the weakest negative residual when both legs
    clear the configured z threshold.  It uses the same two-leg execution
    engine, so the only selection input is the current train/validation-visible
    bar state.
    """
    blocked = _entry_window_block_reason(spec, arrays, idx)
    if blocked:
        return "", "", "", blocked
    if not spec.allow_long or not spec.allow_short:
        return "", "", "", "pair_requires_both_sides"

    lookback = int(spec.lookback_bars)
    threshold = max(0.0, float(spec.threshold))
    symbols = tuple(arrays["symbols"])
    prefixes = tuple(
        arrays.get("symbol_prefixes") or tuple(_symbol_prefix(symbol) for symbol in symbols)
    )
    long_leg: tuple[float, str] | None = None
    short_leg: tuple[float, str] | None = None

    for symbol, prefix in zip(symbols, prefixes, strict=True):
        close = _array_value(arrays, f"{prefix}_close", idx)
        if not math.isfinite(close) or close <= 0.0:
            continue
        rv_key = f"{prefix}_rv_{spec.rv_lookback_bars}h"
        if spec.max_rv > 0.0:
            rv = _array_value(arrays, rv_key, idx)
            if not math.isfinite(rv) or rv > spec.max_rv:
                continue
        resid_z = _array_value(arrays, f"{prefix}_resid_z_{lookback}h", idx)
        if not math.isfinite(resid_z):
            continue
        if resid_z >= threshold and (long_leg is None or resid_z > long_leg[0]):
            long_leg = (resid_z, symbol)
        if resid_z <= -threshold and (short_leg is None or resid_z < short_leg[0]):
            short_leg = (resid_z, symbol)

    if long_leg is None or short_leg is None:
        return "", "", "", "signal_missing"
    long_symbol = long_leg[1]
    short_symbol = short_leg[1]
    if long_symbol == short_symbol:
        return "", "", "", "pair_symbol_overlap"
    spread_z = long_leg[0] - short_leg[0]
    min_spread_z = threshold * 2.0
    if spread_z < min_spread_z:
        return "", "", "", "pair_spread_z_too_small"
    return long_symbol, short_symbol, "LONG_SPREAD", ""


def _state_relative_strength_spread_signal(
    spec: FreshSpec, arrays: dict[str, Any], idx: int
) -> tuple[str, str, str, str]:
    """Select a fixed-symbol state spread without month/day calendar inputs.

    The rule is a causal proxy for the rejected TRX calendar spread: it only
    enters when the current bar shows TRX-relative price momentum, residual
    strength, and optional taker-flow confirmation versus the hedge leg.
    """
    blocked = _entry_window_block_reason(spec, arrays, idx)
    if blocked:
        return "", "", "", blocked
    symbols = tuple(arrays["symbols"])
    long_symbol = _resolve_calendar_symbol(symbols, spec.primary_symbol or "TRXUSDT")
    short_symbol = _resolve_calendar_symbol(symbols, spec.secondary_symbol or "ETHUSDT")
    if not long_symbol or not short_symbol or long_symbol == short_symbol:
        return "", "", "", "state_spread_symbol_missing"

    long_prefix = _symbol_prefix(long_symbol)
    short_prefix = _symbol_prefix(short_symbol)
    long_close = _array_value(arrays, f"{long_prefix}_close", idx)
    short_close = _array_value(arrays, f"{short_prefix}_close", idx)
    if (
        not math.isfinite(long_close)
        or long_close <= 0.0
        or not math.isfinite(short_close)
        or short_close <= 0.0
    ):
        return "", "", "", "state_spread_price_missing"

    if spec.max_rv > 0.0:
        long_rv = _array_value(arrays, f"{long_prefix}_rv_{spec.rv_lookback_bars}h", idx)
        short_rv = _array_value(arrays, f"{short_prefix}_rv_{spec.rv_lookback_bars}h", idx)
        if (
            not math.isfinite(long_rv)
            or not math.isfinite(short_rv)
            or max(long_rv, short_rv) > spec.max_rv
        ):
            return "", "", "", "state_spread_rv_cap"

    if spec.funding_abs_cap > 0.0:
        long_funding = _array_value(arrays, f"{long_prefix}_funding_ffill", idx)
        short_funding = _array_value(arrays, f"{short_prefix}_funding_ffill", idx)
        if (
            not math.isfinite(long_funding)
            or not math.isfinite(short_funding)
            or max(abs(long_funding), abs(short_funding)) > spec.funding_abs_cap
        ):
            return "", "", "", "state_spread_funding_cap"

    lookback = int(spec.lookback_bars)
    long_ret = _array_value(arrays, f"{long_prefix}_ret_{lookback}h", idx)
    short_ret = _array_value(arrays, f"{short_prefix}_ret_{lookback}h", idx)
    long_resid_z = _array_value(arrays, f"{long_prefix}_resid_z_{lookback}h", idx)
    short_resid_z = _array_value(arrays, f"{short_prefix}_resid_z_{lookback}h", idx)
    if (
        not math.isfinite(long_ret)
        or not math.isfinite(short_ret)
        or not math.isfinite(long_resid_z)
        or not math.isfinite(short_resid_z)
    ):
        return "", "", "", "signal_missing"

    hedge = max(0.0, float(spec.spread_hedge_ratio))
    spread_ret = long_ret - hedge * short_ret
    resid_gap = long_resid_z - short_resid_z
    threshold = max(0.0, float(spec.threshold))
    min_abs_return = max(threshold, float(spec.min_abs_return))
    min_resid_gap = max(0.0, float(spec.sharpe_rank_min))

    if spec.flow_threshold > 0.0:
        flow_lookback = max(1, int(spec.flow_lookback_bars or 6))
        long_flow = _array_value(arrays, f"{long_prefix}_flow_imbalance_{flow_lookback}h", idx)
        short_flow = _array_value(arrays, f"{short_prefix}_flow_imbalance_{flow_lookback}h", idx)
        if not math.isfinite(long_flow) or not math.isfinite(short_flow):
            return "", "", "", "state_spread_flow_missing"
        flow_gap = long_flow - hedge * short_flow
    else:
        flow_gap = 0.0

    if (
        spec.allow_long
        and spread_ret >= threshold
        and resid_gap >= min_resid_gap
        and (long_ret >= min_abs_return or short_ret <= -min_abs_return)
        and (spec.flow_threshold <= 0.0 or flow_gap >= spec.flow_threshold)
    ):
        return long_symbol, short_symbol, "LONG_SPREAD", ""
    if (
        spec.allow_short
        and spread_ret <= -threshold
        and resid_gap <= -min_resid_gap
        and (long_ret <= -min_abs_return or short_ret >= min_abs_return)
        and (spec.flow_threshold <= 0.0 or flow_gap <= -spec.flow_threshold)
    ):
        return long_symbol, short_symbol, "SHORT_SPREAD", ""
    return "", "", "", "signal_missing"


def _run_calendar_spread_split(
    *,
    spec: FreshSpec,
    arrays: dict[str, Any],
    split: SplitWindow,
    include_equity: bool = False,
    signal_func: Any | None = None,
) -> dict[str, Any]:
    timestamps = arrays["timestamp"]
    start_ts = int(datetime.combine(split.start, datetime.min.time(), tzinfo=UTC).timestamp())
    end_ts = (
        int(
            datetime.combine(
                split.end + timedelta(days=1), datetime.min.time(), tzinfo=UTC
            ).timestamp()
        )
        - 1
    )
    indices = np.flatnonzero((timestamps >= start_ts) & (timestamps <= end_ts))
    if indices.size == 0:
        return {
            "metrics": {},
            "round_trips": 0,
            "fills": 0,
            "reject_counts": {"split_empty": 1},
            "liquidations": 0,
        }

    cash = 10_000.0
    legs: list[dict[str, Any]] = []
    gross_entry_notional = 0.0
    entry_equity = 10_000.0
    bars_held = 0
    cooldown = 0
    equity_history: list[float] = []
    fills = 0
    round_trips = 0
    reject_counts: dict[str, int] = {}

    def record_reject(reason: str) -> None:
        reject_counts[reason] = int(reject_counts.get(reason, 0)) + 1

    def mark_equity_at(idx: int) -> float:
        equity = cash
        for leg in legs:
            prefix = str(leg["prefix"])
            close = _array_value(arrays, f"{prefix}_close", idx)
            if not math.isfinite(close) or close <= 0.0:
                close = float(leg["entry_price"])
            equity += float(leg["signed_qty"]) * close
        return equity

    def close_legs(idx: int) -> None:
        nonlocal bars_held, cash, fills, gross_entry_notional, entry_equity, legs, round_trips
        for leg in legs:
            prefix = str(leg["prefix"])
            close = _array_value(arrays, f"{prefix}_close", idx)
            if not math.isfinite(close) or close <= 0.0:
                close = float(leg["entry_price"])
            high = _array_value(arrays, f"{prefix}_high", idx)
            low = _array_value(arrays, f"{prefix}_low", idx)
            open_ = _array_value(arrays, f"{prefix}_open", idx)
            high_low_vol = (
                max(0.0, (high - low) / open_)
                if math.isfinite(high) and math.isfinite(low) and open_ > 0.0
                else 0.0
            )
            qty_abs = abs(float(leg["signed_qty"]))
            if float(leg["signed_qty"]) > 0.0:
                fill, fee_rate = _fill_price(close, "SELL", high_low_vol=high_low_vol)
                cash += qty_abs * fill - qty_abs * fill * fee_rate
            else:
                fill, fee_rate = _fill_price(close, "BUY", high_low_vol=high_low_vol)
                cash -= qty_abs * fill + qty_abs * fill * fee_rate
        fills += len(legs)
        round_trips += 1
        legs = []
        bars_held = 0
        gross_entry_notional = 0.0
        entry_equity = cash

    def plan_order(symbol: str, action: str, scale: float, idx: int) -> dict[str, Any] | None:
        prefix = _symbol_prefix(symbol)
        close = _array_value(arrays, f"{prefix}_close", idx)
        if not math.isfinite(close) or close <= 0.0 or scale <= 0.0:
            return None
        high = _array_value(arrays, f"{prefix}_high", idx)
        low = _array_value(arrays, f"{prefix}_low", idx)
        open_ = _array_value(arrays, f"{prefix}_open", idx)
        volume = max(0.0, _array_value(arrays, f"{prefix}_volume", idx))
        high_low_vol = (
            max(0.0, (high - low) / open_)
            if math.isfinite(high) and math.isfinite(low) and open_ > 0.0
            else 0.0
        )
        equity = mark_equity_at(idx)
        notional = min(TARGET_ALLOCATION * scale * equity, MAX_ORDER_VALUE * scale)
        raw_qty = math.floor((notional / close) / 0.001) * 0.001
        order_qty = min(raw_qty, volume * 0.10)
        if order_qty * close < 5.0 or order_qty <= 0.0:
            return None
        fill, fee_rate = _fill_price(close, action, high_low_vol=high_low_vol)
        return {
            "symbol": symbol,
            "prefix": prefix,
            "action": action,
            "qty": float(order_qty),
            "fill": float(fill),
            "fee_rate": float(fee_rate),
            "signed_qty": float(order_qty if action == "BUY" else -order_qty),
        }

    spread_signal = signal_func or _calendar_spread_signal
    for idx_raw in indices:
        idx = int(idx_raw)
        if legs:
            bars_held += 1
            spread_return = (mark_equity_at(idx) - entry_equity) / max(1e-9, gross_entry_notional)
            exit_reason = ""
            if spec.stop_loss_pct > 0.0 and spread_return <= -float(spec.stop_loss_pct):
                exit_reason = "stop"
            elif spec.take_profit_pct > 0.0 and spread_return >= float(spec.take_profit_pct):
                exit_reason = "take_profit"
            elif bars_held >= int(spec.hold_bars):
                exit_reason = "max_hold"
            if exit_reason:
                close_legs(idx)
                cooldown = max(0, int(spec.cooldown_bars))

        if not legs:
            if cooldown > 0:
                cooldown -= 1
                record_reject("cooldown")
            else:
                long_symbol, short_symbol, direction, reason = spread_signal(spec, arrays, idx)
                if not long_symbol or not short_symbol or not direction:
                    record_reject(reason or "signal_missing")
                else:
                    hedge = max(0.0, float(spec.spread_hedge_ratio))
                    if direction == "LONG_SPREAD":
                        leg_plans = (
                            (long_symbol, "BUY", max(0.0, float(spec.long_allocation_scale))),
                            (
                                short_symbol,
                                "SELL",
                                max(0.0, float(spec.short_allocation_scale)) * hedge,
                            ),
                        )
                    else:
                        leg_plans = (
                            (long_symbol, "SELL", max(0.0, float(spec.long_allocation_scale))),
                            (
                                short_symbol,
                                "BUY",
                                max(0.0, float(spec.short_allocation_scale)) * hedge,
                            ),
                        )
                    orders = [
                        order
                        for symbol, action, scale in leg_plans
                        if (order := plan_order(symbol, action, scale, idx)) is not None
                    ]
                    if len(orders) != 2:
                        record_reject("fill_or_min_notional")
                    else:
                        gross_entry_notional = sum(
                            abs(float(order["qty"]) * float(order["fill"])) for order in orders
                        )
                        for order in orders:
                            qty = float(order["qty"])
                            fill = float(order["fill"])
                            fee_rate = float(order["fee_rate"])
                            if order["action"] == "BUY":
                                cash -= qty * fill + qty * fill * fee_rate
                            else:
                                cash += qty * fill - qty * fill * fee_rate
                            legs.append(
                                {
                                    "symbol": order["symbol"],
                                    "prefix": order["prefix"],
                                    "signed_qty": order["signed_qty"],
                                    "entry_price": fill,
                                }
                            )
                        fills += len(orders)
                        bars_held = 0
                        entry_equity = mark_equity_at(idx)

        equity_history.append(mark_equity_at(idx))

    if legs and indices.size:
        close_legs(int(indices[-1]))
        equity_history[-1] = cash

    metrics = _metrics_from_equity_totals(equity_history, periods=HOURLY_PERIODS_PER_YEAR)
    payload = {
        "metrics": metrics,
        "round_trips": int(round_trips),
        "fills": int(fills),
        "final_equity": float(equity_history[-1]) if equity_history else 10_000.0,
        "reject_counts": dict(
            sorted(reject_counts.items(), key=lambda item: (-item[1], item[0]))[:8]
        ),
        "liquidations": 0,
    }
    if include_equity:
        payload["equity_history"] = [float(item) for item in equity_history]
    return payload


def _run_split(
    *, spec: FreshSpec, arrays: dict[str, Any], split: SplitWindow, include_equity: bool = False
) -> dict[str, Any]:
    if spec.family == "calendar_spread":
        return _run_calendar_spread_split(
            spec=spec, arrays=arrays, split=split, include_equity=include_equity
        )
    if spec.family == "residual_pair_reversion_spread":
        return _run_calendar_spread_split(
            spec=spec,
            arrays=arrays,
            split=split,
            include_equity=include_equity,
            signal_func=_residual_pair_spread_signal,
        )
    if spec.family == "residual_pair_momentum_spread":
        return _run_calendar_spread_split(
            spec=spec,
            arrays=arrays,
            split=split,
            include_equity=include_equity,
            signal_func=_residual_pair_momentum_spread_signal,
        )
    if spec.family == "state_relative_strength_spread":
        return _run_calendar_spread_split(
            spec=spec,
            arrays=arrays,
            split=split,
            include_equity=include_equity,
            signal_func=_state_relative_strength_spread_signal,
        )
    timestamps = arrays["timestamp"]
    start_ts = int(datetime.combine(split.start, datetime.min.time(), tzinfo=UTC).timestamp())
    end_ts = (
        int(
            datetime.combine(
                split.end + timedelta(days=1), datetime.min.time(), tzinfo=UTC
            ).timestamp()
        )
        - 1
    )
    indices = np.flatnonzero((timestamps >= start_ts) & (timestamps <= end_ts))
    if indices.size == 0:
        return {
            "metrics": {},
            "round_trips": 0,
            "fills": 0,
            "reject_counts": {"split_empty": 1},
            "liquidations": 0,
        }

    cash = 10_000.0
    qty = 0.0
    position_side = "OUT"
    position_symbol = ""
    entry_price = 0.0
    position_stop_loss_pct = 0.0
    position_take_profit_pct = 0.0
    best_price = 0.0
    bars_held = 0
    cooldown = 0
    equity_history: list[float] = []
    fills = 0
    round_trips = 0
    reject_counts: dict[str, int] = {}

    def record_reject(reason: str) -> None:
        reject_counts[reason] = int(reject_counts.get(reason, 0)) + 1

    def mark_equity_at(idx: int) -> float:
        if position_side == "OUT" or not position_symbol:
            return cash
        prefix = _symbol_prefix(position_symbol)
        close = float(arrays[f"{prefix}_close"][idx])
        if not math.isfinite(close) or close <= 0.0:
            close = entry_price
        if position_side == "LONG":
            return cash + qty * close
        if position_side == "SHORT":
            return cash - qty * close
        return cash

    for idx in indices:
        if position_side != "OUT":
            bars_held += 1
            prefix = _symbol_prefix(position_symbol)
            close = float(arrays[f"{prefix}_close"][idx])
            high = float(arrays[f"{prefix}_high"][idx])
            low = float(arrays[f"{prefix}_low"][idx])
            open_ = float(arrays[f"{prefix}_open"][idx])
            high_low_vol = max(0.0, (high - low) / open_) if open_ > 0.0 else 0.0
            exit_reason = ""
            exit_price = close
            if position_side == "LONG":
                best_price = max(best_price, high if math.isfinite(high) else close)
                stop_pct = position_stop_loss_pct
                base_stop = entry_price * (1.0 - stop_pct) if stop_pct > 0.0 else -math.inf
                trail_stop = best_price * (1.0 - stop_pct) if stop_pct > 0.0 else -math.inf
                stop = max(base_stop, trail_stop)
                take = (
                    entry_price * (1.0 + position_take_profit_pct)
                    if position_take_profit_pct > 0.0
                    else math.inf
                )
                if low <= stop:
                    exit_reason = "stop"
                    exit_price = min(open_, stop) if open_ < stop else stop
                elif high >= take:
                    exit_reason = "take_profit"
                    exit_price = max(open_, take) if open_ > take else take
                elif bars_held >= spec.hold_bars:
                    exit_reason = "max_hold"
                if exit_reason:
                    fill, fee_rate = _fill_price(exit_price, "SELL", high_low_vol=high_low_vol)
                    cash += qty * fill - qty * fill * fee_rate
            elif position_side == "SHORT":
                best_price = min(best_price, low if math.isfinite(low) else close)
                stop_pct = position_stop_loss_pct
                base_stop = entry_price * (1.0 + stop_pct) if stop_pct > 0.0 else math.inf
                trail_stop = best_price * (1.0 + stop_pct) if stop_pct > 0.0 else math.inf
                stop = min(base_stop, trail_stop)
                take = (
                    entry_price * (1.0 - position_take_profit_pct)
                    if position_take_profit_pct > 0.0
                    else -math.inf
                )
                if high >= stop:
                    exit_reason = "stop"
                    exit_price = max(open_, stop) if open_ > stop else stop
                elif low <= take:
                    exit_reason = "take_profit"
                    exit_price = min(open_, take) if open_ < take else take
                elif bars_held >= spec.hold_bars:
                    exit_reason = "max_hold"
                if exit_reason:
                    fill, fee_rate = _fill_price(exit_price, "BUY", high_low_vol=high_low_vol)
                    cash -= qty * fill + qty * fill * fee_rate
            if exit_reason:
                position_side = "OUT"
                position_symbol = ""
                qty = 0.0
                entry_price = 0.0
                position_stop_loss_pct = 0.0
                position_take_profit_pct = 0.0
                best_price = 0.0
                bars_held = 0
                cooldown = max(0, int(spec.cooldown_bars))
                round_trips += 1
                fills += 1

        if position_side == "OUT":
            if cooldown > 0:
                cooldown -= 1
                record_reject("cooldown")
            else:
                symbol, side, reason = _candidate_signal(spec, arrays, int(idx))
                if not symbol or not side:
                    record_reject(reason or "signal_missing")
                else:
                    prefix = _symbol_prefix(symbol)
                    close = float(arrays[f"{prefix}_close"][idx])
                    high = float(arrays[f"{prefix}_high"][idx])
                    low = float(arrays[f"{prefix}_low"][idx])
                    open_ = float(arrays[f"{prefix}_open"][idx])
                    volume = max(0.0, float(arrays[f"{prefix}_volume"][idx]))
                    high_low_vol = max(0.0, (high - low) / open_) if open_ > 0.0 else 0.0
                    allocation_scale = _side_allocation_scale(spec, side) * _state_exposure_scale(
                        spec, arrays, symbol, side, int(idx)
                    )
                    notional = min(
                        TARGET_ALLOCATION * allocation_scale * mark_equity_at(int(idx)),
                        MAX_ORDER_VALUE * allocation_scale,
                    )
                    raw_qty = math.floor((notional / close) / 0.001) * 0.001
                    max_fill_qty = volume * 0.10
                    order_qty = min(raw_qty, max_fill_qty)
                    if order_qty * close < 5.0 or order_qty <= 0.0:
                        record_reject("fill_or_min_notional")
                    elif side == "LONG":
                        fill, fee_rate = _fill_price(close, "BUY", high_low_vol=high_low_vol)
                        cash -= order_qty * fill + order_qty * fill * fee_rate
                        qty = order_qty
                        position_symbol = symbol
                        position_side = "LONG"
                        entry_price = fill
                        position_stop_loss_pct = _entry_stop_pct(spec, arrays, prefix, int(idx))
                        position_take_profit_pct = float(spec.take_profit_pct)
                        best_price = fill
                        bars_held = 0
                        fills += 1
                    else:
                        fill, fee_rate = _fill_price(close, "SELL", high_low_vol=high_low_vol)
                        cash += order_qty * fill - order_qty * fill * fee_rate
                        qty = order_qty
                        position_symbol = symbol
                        position_side = "SHORT"
                        entry_price = fill
                        position_stop_loss_pct = _entry_stop_pct(spec, arrays, prefix, int(idx))
                        position_take_profit_pct = float(spec.take_profit_pct)
                        best_price = fill
                        bars_held = 0
                        fills += 1
        equity_history.append(mark_equity_at(int(idx)))

    if position_side != "OUT" and indices.size:
        idx = int(indices[-1])
        prefix = _symbol_prefix(position_symbol)
        close = float(arrays[f"{prefix}_close"][idx])
        if position_side == "LONG":
            fill, fee_rate = _fill_price(close, "SELL", high_low_vol=0.0)
            cash += qty * fill - qty * fill * fee_rate
        else:
            fill, fee_rate = _fill_price(close, "BUY", high_low_vol=0.0)
            cash -= qty * fill + qty * fill * fee_rate
        equity_history[-1] = cash
        round_trips += 1
        fills += 1

    metrics = _metrics_from_equity_totals(equity_history, periods=HOURLY_PERIODS_PER_YEAR)
    payload = {
        "metrics": metrics,
        "round_trips": int(round_trips),
        "fills": int(fills),
        "final_equity": float(equity_history[-1]) if equity_history else 10_000.0,
        "reject_counts": dict(
            sorted(reject_counts.items(), key=lambda item: (-item[1], item[0]))[:8]
        ),
        "liquidations": 0,
    }
    if include_equity:
        payload["equity_history"] = [float(item) for item in equity_history]
    return payload


def _rv_caps(arrays: dict[str, Any], symbols: list[str]) -> list[float]:
    values: list[float] = []
    for symbol in symbols:
        prefix = _symbol_prefix(symbol)
        rv = arrays[f"{prefix}_rv_24h"]
        finite = rv[np.isfinite(rv)]
        if finite.size:
            values.extend([float(np.quantile(finite, q)) for q in (0.55, 0.70, 0.85)])
    return sorted({round(v, 8) for v in values if v > 0.0})[:9]


def _candidate_specs(arrays: dict[str, Any], symbols: list[str]) -> list[FreshSpec]:
    specs: dict[str, FreshSpec] = {}
    rv_caps = _rv_caps(arrays, symbols)
    session_sets = {
        "all": (),
        "asia_us": (0, 1, 2, 13, 14, 15, 16, 20, 21),
        "post_funding": (2, 3, 4, 10, 11, 12, 18, 19, 20),
    }
    for lookback in (6, 12, 24, 48, 72):
        for threshold in (1.25, 1.5, 1.75, 2.0, 2.5):
            for hold in (6, 12, 24, 48, 72):
                for session_label, hours in session_sets.items():
                    name = f"fresh_resid_rev_lb{lookback}_z{str(threshold).replace('.', '')}_h{hold}_{session_label}"
                    specs[name] = FreshSpec(
                        name=name,
                        family="residual_reversion",
                        lookback_bars=lookback,
                        threshold=threshold,
                        hold_bars=hold,
                        cooldown_bars=max(0, hold // 4),
                        stop_loss_pct=0.018,
                        take_profit_pct=0.035,
                        min_abs_return=0.0025 if lookback <= 12 else 0.004,
                        entry_hours=hours,
                    )
        for threshold in (1.0, 1.25, 1.5, 1.75):
            for hold in (12, 24, 48, 96):
                name = f"fresh_xs_mom_lb{lookback}_z{str(threshold).replace('.', '')}_h{hold}"
                specs[name] = FreshSpec(
                    name=name,
                    family="cross_momentum",
                    lookback_bars=lookback,
                    threshold=threshold,
                    hold_bars=hold,
                    cooldown_bars=max(0, hold // 3),
                    stop_loss_pct=0.015,
                    take_profit_pct=0.045,
                    broad_min_abs=0.003 if lookback <= 24 else 0.006,
                    min_abs_return=0.004,
                    allow_short=True,
                )
    for cap in rv_caps:
        for lookback in (12, 24, 48):
            for threshold in (1.25, 1.5, 1.75):
                name = f"fresh_resid_rev_rvcap{int(cap * 1e6)}_lb{lookback}_z{str(threshold).replace('.', '')}"
                specs[name] = FreshSpec(
                    name=name,
                    family="residual_reversion",
                    lookback_bars=lookback,
                    threshold=threshold,
                    hold_bars=48,
                    cooldown_bars=12,
                    stop_loss_pct=0.015,
                    take_profit_pct=0.035,
                    rv_lookback_bars=24,
                    max_rv=cap,
                    min_abs_return=0.003,
                )
    for lookback in (12, 24, 48, 72):
        for threshold in (1.0, 1.25, 1.5, 1.75):
            for hold in (12, 24, 48, 96):
                for scale in (0.5, 1.0, 2.0, 4.0):
                    name = (
                        f"fresh_resid_mom_lb{lookback}_z{str(threshold).replace('.', '')}_"
                        f"h{hold}_sc{str(scale).replace('.', '')}"
                    )
                    specs[name] = FreshSpec(
                        name=name,
                        family="residual_momentum",
                        lookback_bars=lookback,
                        threshold=threshold,
                        hold_bars=hold,
                        cooldown_bars=max(1, hold // 3),
                        stop_loss_pct=0.012,
                        take_profit_pct=0.040,
                        min_abs_return=0.0025 if lookback <= 24 else 0.004,
                        allow_short=True,
                        long_allocation_scale=scale,
                        short_allocation_scale=scale,
                        trailing_stop_rv_multiple=1.4,
                        trailing_stop_floor_pct=0.006,
                        trailing_stop_cap_pct=0.016,
                    )
    for lookback in (12, 24, 48, 72):
        for threshold in (1.0, 1.25, 1.5):
            for funding_min in (0.00005, 0.00010, 0.00015):
                name = f"fresh_funding_fade_lb{lookback}_z{str(threshold).replace('.', '')}_f{int(funding_min * 1e6)}ppm"
                specs[name] = FreshSpec(
                    name=name,
                    family="funding_carry_fade",
                    lookback_bars=lookback,
                    threshold=threshold,
                    hold_bars=24,
                    cooldown_bars=12,
                    stop_loss_pct=0.018,
                    take_profit_pct=0.035,
                    funding_rank_min=funding_min,
                    min_abs_return=0.003,
                    entry_hours=(1, 2, 9, 10, 17, 18),
                )
                name = (
                    f"fresh_funding_mom_lb{lookback}_z{str(threshold).replace('.', '')}_"
                    f"f{int(funding_min * 1e6)}ppm"
                )
                specs[name] = FreshSpec(
                    name=name,
                    family="funding_carry_momentum",
                    lookback_bars=lookback,
                    threshold=threshold,
                    hold_bars=24,
                    cooldown_bars=12,
                    stop_loss_pct=0.014,
                    take_profit_pct=0.040,
                    funding_rank_min=funding_min,
                    min_abs_return=0.003,
                    entry_hours=(1, 2, 9, 10, 17, 18),
                    allow_short=True,
                    trailing_stop_rv_multiple=1.3,
                    trailing_stop_floor_pct=0.006,
                    trailing_stop_cap_pct=0.016,
                )
    for lookback in (6, 12, 24, 48):
        for threshold in (1.25, 1.5, 1.75):
            for funding_min in (0.00005, 0.00010):
                name = f"fresh_funding_oi_fade_lb{lookback}_z{str(threshold).replace('.', '')}_f{int(funding_min * 1e6)}_oi{int((funding_min * 10) * 100)}"
                specs[name] = FreshSpec(
                    name=name,
                    family="funding_oi_carry_fade",
                    lookback_bars=lookback,
                    threshold=threshold,
                    hold_bars=24,
                    cooldown_bars=12,
                    stop_loss_pct=0.016,
                    take_profit_pct=0.032,
                    funding_rank_min=funding_min,
                    oi_rank_min=funding_min * 10.0,
                    sharpe_lookback_bars=12,
                    min_abs_return=0.003 if lookback >= 24 else 0.002,
                    allow_short=True,
                )
    for lookback, fast_lookback, oi_lookback in ((72, 24, 12), (168, 72, 24)):
        for resid_z in (0.75, 1.00):
            for funding_min in (0.00008, 0.00012):
                for oi_min in (0.012, 0.020):
                    for flow_threshold in (0.03, 0.06):
                        name = (
                            f"fresh_funding_oi_exhaustion_rev_lb{lookback}_fast{fast_lookback}_"
                            f"z{int(resid_z * 100):03d}_f{int(funding_min * 1e6)}_"
                            f"oi{int(oi_min * 1000)}_fl{int(flow_threshold * 100)}"
                        )
                        specs[name] = FreshSpec(
                            name=name,
                            family="funding_oi_exhaustion_reversal",
                            lookback_bars=lookback,
                            threshold=resid_z,
                            hold_bars=96,
                            cooldown_bars=24,
                            stop_loss_pct=0.010,
                            take_profit_pct=0.035,
                            min_abs_return=0.010,
                            allow_long=True,
                            allow_short=True,
                            adaptive_lookback_bars=fast_lookback,
                            flow_lookback_bars=6,
                            flow_threshold=flow_threshold,
                            funding_rank_min=funding_min,
                            oi_rank_min=oi_min,
                            sharpe_lookback_bars=oi_lookback,
                            long_allocation_scale=2.5,
                            short_allocation_scale=2.5,
                            trailing_stop_rv_multiple=1.0,
                            trailing_stop_floor_pct=0.005,
                            trailing_stop_cap_pct=0.014,
                        )
    for lookback, fast_lookback in ((72, 24), (168, 72)):
        for beta in (0.80, 1.00, 1.20):
            for threshold in (0.010, 0.016):
                for flow_threshold in (0.00, 0.03):
                    name = (
                        f"fresh_beta_resid_reversion_lb{lookback}_fast{fast_lookback}_"
                        f"b{int(beta * 100)}_thr{int(threshold * 10000)}_"
                        f"fl{int(flow_threshold * 100)}"
                    )
                    specs[name] = FreshSpec(
                        name=name,
                        family="beta_residual_reversion",
                        lookback_bars=lookback,
                        threshold=threshold,
                        hold_bars=96,
                        cooldown_bars=24,
                        stop_loss_pct=0.010,
                        take_profit_pct=0.032,
                        min_abs_return=0.003,
                        allow_long=True,
                        allow_short=True,
                        adaptive_lookback_bars=fast_lookback,
                        flow_lookback_bars=6,
                        flow_threshold=flow_threshold,
                        spread_hedge_ratio=beta,
                        long_allocation_scale=2.0,
                        short_allocation_scale=2.0,
                        trailing_stop_rv_multiple=1.0,
                        trailing_stop_floor_pct=0.005,
                        trailing_stop_cap_pct=0.014,
                    )
    for lookback in (24, 72):
        for threshold in (0.006, 0.012):
            for compression in (0.75, 1.10):
                for flow_threshold in (0.03, 0.06):
                    name = (
                        f"fresh_dispersion_compression_state_lb{lookback}_"
                        f"thr{int(threshold * 10000)}_comp{int(compression * 100)}_"
                        f"fl{int(flow_threshold * 100)}"
                    )
                    specs[name] = FreshSpec(
                        name=name,
                        family="dispersion_compression_breakout_unwind",
                        lookback_bars=lookback,
                        threshold=threshold,
                        hold_bars=72,
                        cooldown_bars=18,
                        stop_loss_pct=0.010,
                        take_profit_pct=0.030,
                        min_abs_return=threshold,
                        allow_long=True,
                        allow_short=True,
                        broad_min_abs=0.002,
                        flow_lookback_bars=6,
                        flow_threshold=flow_threshold,
                        compression_quantile=compression,
                        long_allocation_scale=2.0,
                        short_allocation_scale=2.0,
                        trailing_stop_rv_multiple=1.0,
                        trailing_stop_floor_pct=0.005,
                        trailing_stop_cap_pct=0.014,
                    )
    for lookback, fast_lookback in ((24, 12), (72, 24)):
        for resid_z in (0.75, 1.00):
            for min_return in (0.006, 0.012):
                for max_rv in (0.018, 0.030):
                    name = (
                        f"fresh_vol_regime_margin_scaled_lb{lookback}_fast{fast_lookback}_"
                        f"z{int(resid_z * 100):03d}_ret{int(min_return * 10000)}_"
                        f"rv{int(max_rv * 10000)}"
                    )
                    specs[name] = FreshSpec(
                        name=name,
                        family="vol_regime_margin_scaled_momentum",
                        lookback_bars=lookback,
                        threshold=resid_z,
                        hold_bars=72,
                        cooldown_bars=18,
                        stop_loss_pct=0.010,
                        take_profit_pct=0.035,
                        min_abs_return=min_return,
                        allow_long=True,
                        allow_short=True,
                        adaptive_lookback_bars=fast_lookback,
                        rv_lookback_bars=24,
                        max_rv=max_rv,
                        broad_min_abs=0.012,
                        funding_abs_cap=0.00020,
                        sharpe_rank_min=3.0,
                        long_allocation_scale=5.0,
                        short_allocation_scale=5.0,
                        trailing_stop_rv_multiple=1.1,
                        trailing_stop_floor_pct=0.005,
                        trailing_stop_cap_pct=0.014,
                    )
    for flow_lookback in (3, 6, 12, 24):
        for price_lookback in (3, 6, 12, 24):
            for flow_threshold in (0.03, 0.06, 0.10, 0.15):
                for hold in (6, 12, 24, 48):
                    name = (
                        f"fresh_flow_mom_fl{flow_lookback}_px{price_lookback}_"
                        f"imb{int(flow_threshold * 100)}_h{hold}"
                    )
                    specs[name] = FreshSpec(
                        name=name,
                        family="flow_momentum",
                        lookback_bars=price_lookback,
                        threshold=0.0,
                        hold_bars=hold,
                        cooldown_bars=max(1, hold // 4),
                        stop_loss_pct=0.012,
                        take_profit_pct=0.030,
                        min_abs_return=0.0015 if price_lookback <= 6 else 0.003,
                        flow_lookback_bars=flow_lookback,
                        flow_threshold=flow_threshold,
                    )
                    name = (
                        f"fresh_flow_exhaust_fl{flow_lookback}_px{price_lookback}_"
                        f"imb{int(flow_threshold * 100)}_h{hold}"
                    )
                    specs[name] = FreshSpec(
                        name=name,
                        family="flow_exhaustion_fade",
                        lookback_bars=price_lookback,
                        threshold=0.0,
                        hold_bars=hold,
                        cooldown_bars=max(1, hold // 3),
                        stop_loss_pct=0.014,
                        take_profit_pct=0.028,
                        min_abs_return=0.0015 if price_lookback <= 6 else 0.003,
                        flow_lookback_bars=flow_lookback,
                        flow_threshold=flow_threshold,
                    )
                    name = (
                        f"fresh_flow_imb_persist_fl{flow_lookback}_px{price_lookback}_"
                        f"imb{int(flow_threshold * 100)}_h{hold}"
                    )
                    specs[name] = FreshSpec(
                        name=name,
                        family="flow_imbalance_persistence",
                        lookback_bars=price_lookback,
                        threshold=0.0,
                        hold_bars=hold,
                        cooldown_bars=max(1, hold // 2),
                        stop_loss_pct=0.013,
                        take_profit_pct=0.029,
                        min_abs_return=0.0015 if price_lookback <= 6 else 0.003,
                        flow_lookback_bars=flow_lookback,
                        flow_threshold=flow_threshold,
                        flow_persistence_bars=4,
                        flow_persistence_threshold=flow_threshold,
                    )
                    name = (
                        f"fresh_flow_imb_exhaust_fl{flow_lookback}_px{price_lookback}_"
                        f"imb{int(flow_threshold * 100)}_h{hold}"
                    )
                    specs[name] = FreshSpec(
                        name=name,
                        family="flow_imbalance_exhaustion",
                        lookback_bars=price_lookback,
                        threshold=0.0,
                        hold_bars=hold,
                        cooldown_bars=max(1, hold // 2),
                        stop_loss_pct=0.014,
                        take_profit_pct=0.028,
                        min_abs_return=0.0015 if price_lookback <= 6 else 0.003,
                        flow_lookback_bars=flow_lookback,
                        flow_threshold=flow_threshold,
                        flow_persistence_threshold=flow_threshold,
                    )
    for flow_lookback in (3, 6, 12):
        for price_lookback in (6, 12, 24, 48):
            for threshold in (1.25, 1.5, 1.75):
                for flow_threshold in (0.03, 0.06, 0.10):
                    name = (
                        f"fresh_resid_flow_rev_fl{flow_lookback}_lb{price_lookback}_"
                        f"z{str(threshold).replace('.', '')}_imb{int(flow_threshold * 100)}"
                    )
                    specs[name] = FreshSpec(
                        name=name,
                        family="residual_reversion_flow_confirmed",
                        lookback_bars=price_lookback,
                        threshold=threshold,
                        hold_bars=24,
                        cooldown_bars=8,
                        stop_loss_pct=0.014,
                        take_profit_pct=0.032,
                        min_abs_return=0.0025 if price_lookback <= 12 else 0.004,
                        flow_lookback_bars=flow_lookback,
                        flow_threshold=flow_threshold,
                    )
    for lookback in (12, 24, 48):
        for threshold in (0.0010, 0.0015, 0.0020):
            name = f"fresh_adaptive_trend_lb{lookback}_thr{int(threshold * 10000)}"
            specs[name] = FreshSpec(
                name=name,
                family="adaptive_trend",
                lookback_bars=lookback,
                adaptive_lookback_bars=max(12, lookback),
                threshold=threshold,
                hold_bars=12,
                cooldown_bars=6,
                stop_loss_pct=0.014,
                take_profit_pct=0.028,
                min_abs_return=0.0015 if lookback <= 12 else 0.0025,
                allow_short=True,
                trailing_stop_rv_multiple=1.6,
                trailing_stop_floor_pct=0.006,
                trailing_stop_cap_pct=0.018,
            )
    for lookback in (6, 12, 24, 48):
        for threshold in (0.0010, 0.0015, 0.0020, 0.0030):
            for hold in (6, 12, 24, 48):
                for scale in (0.5, 1.0, 2.0):
                    name = (
                        f"fresh_adaptive_trend_fade_lb{lookback}_thr{int(threshold * 10000)}_"
                        f"h{hold}_sc{str(scale).replace('.', '')}"
                    )
                    specs[name] = FreshSpec(
                        name=name,
                        family="adaptive_trend_fade",
                        lookback_bars=lookback,
                        adaptive_lookback_bars=max(6, lookback),
                        threshold=threshold,
                        hold_bars=hold,
                        cooldown_bars=max(1, hold // 3),
                        stop_loss_pct=0.010,
                        take_profit_pct=0.035,
                        min_abs_return=0.0015 if lookback <= 12 else 0.0025,
                        allow_short=True,
                        long_allocation_scale=scale,
                        short_allocation_scale=scale,
                        trailing_stop_rv_multiple=1.2,
                        trailing_stop_floor_pct=0.005,
                        trailing_stop_cap_pct=0.014,
                    )
    for lookback in (12, 24, 48):
        for rank_min in (0.05, 0.08, 0.12):
            name = f"fresh_cross_sharpe_rank_lb{lookback}_r{int(rank_min * 100)}"
            specs[name] = FreshSpec(
                name=name,
                family="cross_sectional_sharpe_rank",
                lookback_bars=lookback,
                sharpe_lookback_bars=24,
                sharpe_rank_min=rank_min,
                threshold=0.0,
                hold_bars=24,
                cooldown_bars=8,
                stop_loss_pct=0.013,
                take_profit_pct=0.030,
                min_abs_return=0.0025 if lookback <= 24 else 0.0035,
                allow_short=True,
                long_allocation_scale=1.15,
                short_allocation_scale=0.70,
            )
    for lookback in (12, 24, 48, 72):
        for rank_min in (0.03, 0.05, 0.08, 0.12):
            for hold in (12, 24, 48):
                name = f"fresh_cross_sharpe_reversal_lb{lookback}_r{int(rank_min * 100)}_h{hold}"
                specs[name] = FreshSpec(
                    name=name,
                    family="cross_sectional_sharpe_reversal",
                    lookback_bars=lookback,
                    sharpe_lookback_bars=max(12, lookback),
                    sharpe_rank_min=rank_min,
                    threshold=0.0,
                    hold_bars=hold,
                    cooldown_bars=max(1, hold // 3),
                    stop_loss_pct=0.010,
                    take_profit_pct=0.035,
                    min_abs_return=0.0020 if lookback <= 24 else 0.0030,
                    allow_short=True,
                    long_allocation_scale=0.85,
                    short_allocation_scale=0.85,
                    trailing_stop_rv_multiple=1.2,
                    trailing_stop_floor_pct=0.005,
                    trailing_stop_cap_pct=0.014,
                )
    for lookback, fast_lookback in ((72, 24), (168, 72), (336, 168)):
        for resid_z in (0.50, 0.75, 1.0):
            for min_return in (0.006, 0.012, 0.018):
                for hold in (120, 168):
                    for flow_threshold in (0.0, 0.03):
                        for mode_label, allow_long, allow_short, long_scale, short_scale in (
                            ("both", True, True, 5.9, 10.0),
                            ("longonly", True, False, 6.2, 0.0),
                        ):
                            for take in (0.024, 0.045, 0.060):
                                flow_label = (
                                    f"_fl{int(flow_threshold * 100)}" if flow_threshold else ""
                                )
                                name = (
                                    f"fresh_state_distilled_{mode_label}_lb{lookback}_fast{fast_lookback}_"
                                    f"z{int(resid_z * 100):03d}_ret{int(min_return * 10000)}_"
                                    f"h{hold}_ls{int(long_scale * 100)}_ss{int(short_scale * 10)}_"
                                    f"tp{int(take * 10000)}{flow_label}"
                                )
                                specs[name] = FreshSpec(
                                    name=name,
                                    family="state_distilled_leadership_unwind",
                                    lookback_bars=lookback,
                                    threshold=resid_z,
                                    hold_bars=hold,
                                    cooldown_bars=max(0, hold // 4),
                                    stop_loss_pct=0.0,
                                    take_profit_pct=take,
                                    min_abs_return=min_return,
                                    allow_long=allow_long,
                                    allow_short=allow_short,
                                    adaptive_lookback_bars=fast_lookback,
                                    flow_lookback_bars=6,
                                    flow_threshold=flow_threshold,
                                    sharpe_lookback_bars=6,
                                    sharpe_rank_min=max(1.0, resid_z * 1.5),
                                    oi_rank_min=0.02 if flow_threshold > 0.0 else 0.0,
                                    long_allocation_scale=long_scale,
                                    short_allocation_scale=short_scale,
                                    trailing_stop_rv_multiple=0.0,
                                    trailing_stop_floor_pct=0.0,
                                    trailing_stop_cap_pct=0.0,
                                )
    for lookback, fast_lookback, oi_lookback in ((72, 24, 12), (168, 72, 24), (336, 72, 24)):
        for resid_z in (0.35, 0.50, 0.75):
            for min_return in (0.003, 0.006, 0.009):
                for hold in (72, 120, 168):
                    for flow_threshold in (0.0, 0.02):
                        for mode_label, allow_long, allow_short, long_scale, short_scale in (
                            ("both", True, True, 5.9, 8.0),
                            ("longonly", True, False, 6.2, 0.0),
                            ("shortonly", False, True, 0.0, 6.0),
                        ):
                            for external_risk_max in (0.0, 1.25):
                                risk_label = (
                                    f"_xr{int(external_risk_max * 100)}"
                                    if external_risk_max
                                    else ""
                                )
                                name = (
                                    f"fresh_calendar_teacher_state_{mode_label}_lb{lookback}_"
                                    f"fast{fast_lookback}_z{int(resid_z * 100):03d}_"
                                    f"ret{int(min_return * 10000)}_h{hold}_"
                                    f"fl{int(flow_threshold * 100)}{risk_label}"
                                )
                                specs[name] = FreshSpec(
                                    name=name,
                                    family="calendar_teacher_state_similarity",
                                    lookback_bars=lookback,
                                    threshold=resid_z,
                                    hold_bars=hold,
                                    cooldown_bars=max(1, hold // 4),
                                    stop_loss_pct=0.004,
                                    take_profit_pct=0.045,
                                    min_abs_return=min_return,
                                    allow_long=allow_long,
                                    allow_short=allow_short,
                                    adaptive_lookback_bars=fast_lookback,
                                    flow_lookback_bars=6,
                                    flow_threshold=flow_threshold,
                                    sharpe_lookback_bars=oi_lookback,
                                    sharpe_rank_min=max(0.50, resid_z),
                                    funding_rank_min=0.00010,
                                    funding_abs_cap=0.00025,
                                    oi_rank_min=0.012,
                                    broad_min_abs=0.006,
                                    long_allocation_scale=long_scale,
                                    short_allocation_scale=short_scale,
                                    trailing_stop_rv_multiple=0.8,
                                    trailing_stop_floor_pct=0.004,
                                    trailing_stop_cap_pct=0.012,
                                    external_risk_max=external_risk_max,
                                )
    for lookback, fast_lookback in ((168, 72), (336, 168)):
        for resid_z in (0.50, 0.75):
            for min_return in (0.006, 0.012, 0.018):
                for hold in (120, 168):
                    for flow_threshold in (0.0, 0.03):
                        for mode_label, allow_long, allow_short, long_scale, short_scale in (
                            ("both", True, True, 5.9, 10.0),
                            ("longonly", True, False, 6.2, 0.0),
                        ):
                            for take in (0.045, 0.060, 0.075):
                                for external_risk_max in (0.50, 1.00, 1.25, 1.50, 1.75, 2.00):
                                    name = (
                                        f"fresh_state_distilled_ext_{mode_label}_lb{lookback}_"
                                        f"fast{fast_lookback}_z{int(resid_z * 100):03d}_"
                                        f"ret{int(min_return * 10000)}_h{hold}_"
                                        f"tp{int(take * 10000)}_fl{int(flow_threshold * 100)}_"
                                        f"xr{int(external_risk_max * 100)}"
                                    )
                                    specs[name] = FreshSpec(
                                        name=name,
                                        family="state_distilled_external_risk_filter",
                                        lookback_bars=lookback,
                                        threshold=resid_z,
                                        hold_bars=hold,
                                        cooldown_bars=max(0, hold // 4),
                                        stop_loss_pct=0.0,
                                        take_profit_pct=take,
                                        min_abs_return=min_return,
                                        allow_long=allow_long,
                                        allow_short=allow_short,
                                        adaptive_lookback_bars=fast_lookback,
                                        flow_lookback_bars=6,
                                        flow_threshold=flow_threshold,
                                        sharpe_lookback_bars=6,
                                        sharpe_rank_min=max(1.0, resid_z * 1.5),
                                        oi_rank_min=0.02 if flow_threshold > 0.0 else 0.0,
                                        long_allocation_scale=long_scale,
                                        short_allocation_scale=short_scale,
                                        trailing_stop_rv_multiple=0.0,
                                        trailing_stop_floor_pct=0.0,
                                        trailing_stop_cap_pct=0.0,
                                        external_risk_max=external_risk_max,
                                    )
    for lookback, fast_lookback, oi_lookback in ((72, 24, 12), (168, 72, 24), (336, 72, 24)):
        for resid_z in (0.35, 0.50, 0.75):
            for min_return in (0.003, 0.006, 0.009):
                for hold in (72, 120, 168):
                    for flow_threshold in (0.0, 0.02):
                        for external_risk_max in (0.0, 1.25):
                            risk_label = (
                                f"_xr{int(external_risk_max * 100)}" if external_risk_max else ""
                            )
                            name = (
                                f"fresh_calendar_teacher_fade_both_lb{lookback}_"
                                f"fast{fast_lookback}_z{int(resid_z * 100):03d}_"
                                f"ret{int(min_return * 10000)}_h{hold}_"
                                f"fl{int(flow_threshold * 100)}{risk_label}"
                            )
                            specs[name] = FreshSpec(
                                name=name,
                                family="calendar_teacher_state_fade",
                                lookback_bars=lookback,
                                threshold=resid_z,
                                hold_bars=hold,
                                cooldown_bars=max(1, hold // 4),
                                stop_loss_pct=0.004,
                                take_profit_pct=0.045,
                                min_abs_return=min_return,
                                allow_long=True,
                                allow_short=True,
                                adaptive_lookback_bars=fast_lookback,
                                flow_lookback_bars=6,
                                flow_threshold=flow_threshold,
                                sharpe_lookback_bars=oi_lookback,
                                sharpe_rank_min=max(0.50, resid_z),
                                funding_rank_min=0.00010,
                                funding_abs_cap=0.00025,
                                oi_rank_min=0.012,
                                broad_min_abs=0.006,
                                long_allocation_scale=6.0,
                                short_allocation_scale=8.0,
                                trailing_stop_rv_multiple=0.8,
                                trailing_stop_floor_pct=0.004,
                                trailing_stop_cap_pct=0.012,
                                external_risk_max=external_risk_max,
                            )
    for lookback, fast_lookback in ((72, 24), (168, 24), (168, 72)):
        for resid_z in (1.50, 1.75):
            for min_return in (0.012, 0.020):
                for hold in (72, 96):
                    for crowd_label, funding_min, oi_min in (
                        ("fund", 0.00010, 0.0),
                        ("oi", 0.0, 0.020),
                    ):
                        for gap_max in (0.20, 0.40):
                            name = (
                                f"fresh_state_crowded_unwind_v2_lb{lookback}_fast{fast_lookback}_"
                                f"z{int(resid_z * 100):03d}_ret{int(min_return * 10000)}_"
                                f"h{hold}_{crowd_label}_gap{int(gap_max * 100)}"
                            )
                            specs[name] = FreshSpec(
                                name=name,
                                family="state_distilled_crowded_unwind_v2",
                                lookback_bars=lookback,
                                threshold=resid_z,
                                hold_bars=hold,
                                cooldown_bars=max(1, hold // 4),
                                stop_loss_pct=0.006,
                                take_profit_pct=0.040,
                                min_abs_return=min_return,
                                allow_long=False,
                                allow_short=True,
                                adaptive_lookback_bars=fast_lookback,
                                flow_lookback_bars=6,
                                flow_threshold=0.04,
                                funding_rank_min=funding_min,
                                oi_rank_min=oi_min,
                                sharpe_lookback_bars=12,
                                sharpe_rank_min=gap_max,
                                broad_min_abs=0.010,
                                short_allocation_scale=3.0,
                                trailing_stop_rv_multiple=1.1,
                                trailing_stop_floor_pct=0.005,
                                trailing_stop_cap_pct=0.014,
                            )
    for lookback in (72, 168, 336):
        for resid_z in (0.75, 1.0, 1.25):
            for min_return in (0.012, 0.015, 0.018):
                for hold in (96, 120, 168):
                    for long_scale in (4.0, 5.9, 6.2):
                        for short_scale in (6.0, 10.0):
                            for take in (0.018, 0.024, 0.045):
                                for flow_threshold in (0.0, 0.03):
                                    flow_label = (
                                        f"_fl{int(flow_threshold * 100)}" if flow_threshold else ""
                                    )
                                    name = (
                                        f"fresh_state_trx_mom_lb{lookback}_"
                                        f"z{int(resid_z * 100):03d}_ret{int(min_return * 10000)}_"
                                        f"h{hold}_ls{int(long_scale * 100)}_ss{int(short_scale * 10)}_"
                                        f"tp{int(take * 10000)}{flow_label}"
                                    )
                                    specs[name] = FreshSpec(
                                        name=name,
                                        family="state_momentum_proxy",
                                        lookback_bars=lookback,
                                        threshold=resid_z,
                                        hold_bars=hold,
                                        cooldown_bars=max(0, hold // 4),
                                        stop_loss_pct=0.0,
                                        take_profit_pct=take,
                                        min_abs_return=min_return,
                                        allow_long=True,
                                        allow_short=True,
                                        long_allocation_scale=long_scale,
                                        short_allocation_scale=short_scale,
                                        flow_lookback_bars=6,
                                        flow_threshold=flow_threshold,
                                        primary_symbol="TRXUSDT",
                                        secondary_symbol="ETHUSDT",
                                    )
    for lookback in (72, 168):
        for resid_z in (0.50, 0.75, 1.00):
            for min_return in (0.006, 0.012, 0.015):
                for hold in (72, 96, 120, 168):
                    for long_scale in (4.0, 5.9, 6.2, 8.0):
                        for take in (0.018, 0.024, 0.045, 0.060):
                            name = (
                                f"fresh_state_trx_longonly_lb{lookback}_z{int(resid_z * 100):03d}_"
                                f"ret{int(min_return * 10000)}_h{hold}_"
                                f"ls{int(long_scale * 100)}_tp{int(take * 10000)}"
                            )
                            specs[name] = FreshSpec(
                                name=name,
                                family="state_momentum_proxy",
                                lookback_bars=lookback,
                                threshold=resid_z,
                                hold_bars=hold,
                                cooldown_bars=max(0, hold // 4),
                                stop_loss_pct=0.0,
                                take_profit_pct=take,
                                min_abs_return=min_return,
                                allow_long=True,
                                allow_short=False,
                                long_allocation_scale=long_scale,
                                short_allocation_scale=0.0,
                                primary_symbol="TRXUSDT",
                            )
    for fast_lookback, regime_lookback in ((72, 168), (168, 336)):
        for resid_z in (0.75, 1.0):
            for regime_resid_z in (0.0, 0.50, 0.75):
                for min_return in (0.012, 0.015):
                    for hold in (96, 120, 168):
                        for long_scale in (4.0, 5.9, 6.2):
                            for short_scale in (6.0, 10.0):
                                for take in (0.024, 0.045):
                                    for flow_threshold in (0.0, 0.03):
                                        flow_label = (
                                            f"_fl{int(flow_threshold * 100)}"
                                            if flow_threshold
                                            else ""
                                        )
                                        name = (
                                            f"fresh_state_trx_dual_mom_fast{fast_lookback}_reg{regime_lookback}_"
                                            f"z{int(resid_z * 100):03d}_rz{int(regime_resid_z * 100):03d}_"
                                            f"ret{int(min_return * 10000)}_h{hold}_"
                                            f"ls{int(long_scale * 100)}_ss{int(short_scale * 10)}_"
                                            f"tp{int(take * 10000)}{flow_label}"
                                        )
                                        specs[name] = FreshSpec(
                                            name=name,
                                            family="state_momentum_proxy",
                                            lookback_bars=fast_lookback,
                                            threshold=resid_z,
                                            hold_bars=hold,
                                            cooldown_bars=max(0, hold // 4),
                                            stop_loss_pct=0.0,
                                            take_profit_pct=take,
                                            min_abs_return=min_return,
                                            allow_long=True,
                                            allow_short=True,
                                            long_allocation_scale=long_scale,
                                            short_allocation_scale=short_scale,
                                            flow_lookback_bars=6,
                                            flow_threshold=flow_threshold,
                                            adaptive_lookback_bars=regime_lookback,
                                            sharpe_rank_min=regime_resid_z,
                                            primary_symbol="TRXUSDT",
                                            secondary_symbol="ETHUSDT",
                                        )
    for lookback in (72, 168):
        for threshold in (0.006, 0.012, 0.018):
            for resid_gap in (0.50, 1.00):
                for hold in (72, 120, 168):
                    for hedge in (0.5, 1.0):
                        for scale in (2.0, 3.0, 4.0):
                            for take in (0.018, 0.024, 0.060):
                                for flow_threshold in (0.0, 0.03):
                                    flow_label = (
                                        f"_fl{int(flow_threshold * 100)}" if flow_threshold else ""
                                    )
                                    name = (
                                        f"fresh_state_trx_eth_spread_lb{lookback}_"
                                        f"thr{int(threshold * 10000)}_zg{int(resid_gap * 100):03d}_"
                                        f"h{hold}_hr{int(hedge * 100)}_sc{str(scale).replace('.', '')}_"
                                        f"tp{int(take * 10000)}{flow_label}"
                                    )
                                    specs[name] = FreshSpec(
                                        name=name,
                                        family="state_relative_strength_spread",
                                        lookback_bars=lookback,
                                        threshold=threshold,
                                        hold_bars=hold,
                                        cooldown_bars=max(0, hold // 4),
                                        stop_loss_pct=0.006,
                                        take_profit_pct=take,
                                        min_abs_return=threshold,
                                        allow_long=True,
                                        allow_short=True,
                                        long_allocation_scale=scale,
                                        short_allocation_scale=scale,
                                        flow_lookback_bars=6,
                                        flow_threshold=flow_threshold,
                                        sharpe_rank_min=resid_gap,
                                        primary_symbol="TRXUSDT",
                                        secondary_symbol="ETHUSDT",
                                        spread_hedge_ratio=hedge,
                                    )
    calendar_pairs = (
        ("", ""),
        ("TRXUSDT", ""),
        ("BTCUSDT", ""),
        ("ETHUSDT", ""),
        ("TRXUSDT", "ETHUSDT"),
        ("", "ETHUSDT"),
    )
    for long_symbol, short_symbol in calendar_pairs:
        long_label = long_symbol.lower() or "strongest"
        short_label = short_symbol.lower() or "weakest"
        for lookback in (72, 168, 336):
            for threshold in (0.002, 0.005, 0.010, 0.020):
                for hold in (48, 120, 168, 336):
                    for scale in (1.0, 2.0, 4.0, 6.0, 8.0):
                        for stop in (0.0, 0.006):
                            name = (
                                f"fresh_calendar_rot_l{long_label}_s{short_label}_lb{lookback}_"
                                f"thr{int(threshold * 10000)}_h{hold}_sc{str(scale).replace('.', '')}_"
                                f"st{int(stop * 10000)}"
                            )
                            specs[name] = FreshSpec(
                                name=name,
                                family="calendar_rotation",
                                lookback_bars=lookback,
                                threshold=threshold,
                                hold_bars=hold,
                                cooldown_bars=max(0, hold // 4),
                                stop_loss_pct=stop,
                                take_profit_pct=0.0,
                                min_abs_return=threshold,
                                allow_long=True,
                                allow_short=True,
                                long_allocation_scale=scale,
                                short_allocation_scale=scale,
                                calendar_long_months=(3, 4, 5),
                                calendar_short_months=(1, 2),
                                calendar_long_symbol=long_symbol,
                                calendar_short_symbol=short_symbol,
                            )
    for short_symbol in ("", "ETHUSDT"):
        short_label = short_symbol.lower() or "weakest"
        for threshold in (0.012, 0.015, 0.018):
            for hold in (120, 144, 168):
                for long_scale in (5.3, 5.4, 5.6, 5.8, 5.9, 6.0, 6.2):
                    for short_scale in (8.0, 10.0, 12.0):
                        for take in (0.018, 0.024, 0.045, 0.060):
                            name = (
                                f"fresh_calendar_trx_takeprofit_s{short_label}_thr{int(threshold * 10000)}_"
                                f"h{hold}_ls{int(long_scale * 100)}_ss{int(short_scale * 10)}_"
                                f"tp{int(take * 10000)}"
                            )
                            specs[name] = FreshSpec(
                                name=name,
                                family="calendar_rotation",
                                lookback_bars=168,
                                threshold=threshold,
                                hold_bars=hold,
                                cooldown_bars=max(0, hold // 4),
                                stop_loss_pct=0.0,
                                take_profit_pct=take,
                                min_abs_return=threshold,
                                allow_long=True,
                                allow_short=True,
                                long_allocation_scale=long_scale,
                                short_allocation_scale=short_scale,
                                calendar_long_months=(3, 4, 5),
                                calendar_short_months=(1, 2),
                                calendar_long_symbol="TRXUSDT",
                                calendar_short_symbol=short_symbol,
                            )
    veto_sets = (
        ("rz10", {"calendar_veto_resid_z": 1.0}),
        ("rz15", {"calendar_veto_resid_z": 1.5}),
        ("fund100", {"calendar_veto_funding_abs": 0.00010}),
        ("flow6", {"calendar_veto_flow_abs": 0.06, "flow_lookback_bars": 6}),
        ("mkt24", {"calendar_veto_market_ret_abs": 0.04, "adaptive_lookback_bars": 24}),
    )
    for short_symbol in ("", "ETHUSDT"):
        short_label = short_symbol.lower() or "weakest"
        for threshold in (0.015, 0.018):
            for hold in (120, 168):
                for veto_label, veto_kwargs in veto_sets:
                    name = (
                        f"fresh_calendar_trx_veto_{veto_label}_s{short_label}_"
                        f"thr{int(threshold * 10000)}_h{hold}"
                    )
                    specs[name] = FreshSpec(
                        name=name,
                        family="calendar_rotation",
                        lookback_bars=168,
                        threshold=threshold,
                        hold_bars=hold,
                        cooldown_bars=max(0, hold // 4),
                        stop_loss_pct=0.0,
                        take_profit_pct=0.060,
                        min_abs_return=threshold,
                        allow_long=True,
                        allow_short=True,
                        long_allocation_scale=6.0,
                        short_allocation_scale=12.0,
                        calendar_long_months=(3, 4, 5),
                        calendar_short_months=(1, 2),
                        calendar_long_symbol="TRXUSDT",
                        calendar_short_symbol=short_symbol,
                        **veto_kwargs,
                    )
    day_windows = {
        "early": tuple(range(1, 11)),
        "mid": tuple(range(11, 21)),
        "late": tuple(range(21, 32)),
    }
    day_window_sessions = {
        "postfund": (2, 3, 10, 11, 18, 19),
        "asiaus": (0, 1, 2, 13, 14, 15, 16, 20, 21),
    }
    for short_symbol in ("", "ETHUSDT"):
        short_label = short_symbol.lower() or "weakest"
        for day_label, days in day_windows.items():
            for session_label, hours in day_window_sessions.items():
                for hold in (120, 168):
                    name = (
                        f"fresh_calendar_trx_daywin_{day_label}_{session_label}_"
                        f"s{short_label}_thr180_h{hold}"
                    )
                    specs[name] = FreshSpec(
                        name=name,
                        family="calendar_rotation",
                        lookback_bars=168,
                        threshold=0.018,
                        hold_bars=hold,
                        cooldown_bars=max(0, hold // 4),
                        stop_loss_pct=0.0,
                        take_profit_pct=0.060,
                        min_abs_return=0.018,
                        allow_long=True,
                        allow_short=True,
                        long_allocation_scale=6.0,
                        short_allocation_scale=12.0,
                        entry_hours=hours,
                        entry_days_of_month=days,
                        calendar_long_months=(3, 4, 5),
                        calendar_short_months=(1, 2),
                        calendar_long_symbol="TRXUSDT",
                        calendar_short_symbol=short_symbol,
                    )
    for threshold in (0.012, 0.018):
        for hold in (120, 168):
            for hedge in (0.5, 1.0):
                for take in (0.024, 0.060):
                    name = (
                        f"fresh_calendar_spread_trx_eth_hr{int(hedge * 100)}_"
                        f"thr{int(threshold * 10000)}_h{hold}_tp{int(take * 10000)}"
                    )
                    specs[name] = FreshSpec(
                        name=name,
                        family="calendar_spread",
                        lookback_bars=168,
                        threshold=threshold,
                        hold_bars=hold,
                        cooldown_bars=max(0, hold // 4),
                        stop_loss_pct=0.006,
                        take_profit_pct=take,
                        min_abs_return=threshold,
                        allow_long=True,
                        allow_short=True,
                        long_allocation_scale=3.0,
                        short_allocation_scale=3.0,
                        calendar_long_months=(3, 4, 5),
                        calendar_short_months=(1, 2),
                        calendar_long_symbol="TRXUSDT",
                        calendar_short_symbol="ETHUSDT",
                        spread_hedge_ratio=hedge,
                    )
    pair_sessions = {
        "all": (),
        "postfund": (2, 3, 10, 11, 18, 19),
        "asiaus": (0, 1, 2, 13, 14, 15, 16, 20, 21),
    }
    for lookback in (12, 24):
        for threshold in (0.50, 0.75, 1.00, 1.50):
            for hold in (48, 72, 120):
                for scale in (1.0, 2.0, 3.0):
                    for stop in (0.006, 0.010):
                        for take in (0.012, 0.024, 0.040):
                            for session_label, hours in pair_sessions.items():
                                name = (
                                    f"fresh_pair_resid_revert_spread_lb{lookback}_"
                                    f"z{int(threshold * 100):03d}_h{hold}_"
                                    f"sc{str(scale).replace('.', '')}_"
                                    f"st{int(stop * 10000)}_tp{int(take * 10000)}_{session_label}"
                                )
                                specs[name] = FreshSpec(
                                    name=name,
                                    family="residual_pair_reversion_spread",
                                    lookback_bars=lookback,
                                    threshold=threshold,
                                    hold_bars=hold,
                                    cooldown_bars=max(1, hold // 4),
                                    stop_loss_pct=stop,
                                    take_profit_pct=take,
                                    min_abs_return=0.0,
                                    entry_hours=hours,
                                    allow_long=True,
                                    allow_short=True,
                                    long_allocation_scale=scale,
                                    short_allocation_scale=scale,
                                    spread_hedge_ratio=1.0,
                                )
    momentum_sessions = {
        "all": (),
        "asiaus": (0, 1, 2, 13, 14, 15, 16, 20, 21),
    }
    for lookback in (12, 24, 48):
        for threshold in (0.75, 1.00, 1.50, 2.00):
            for hold in (24, 48, 72):
                for scale in (0.5, 1.0, 2.0):
                    for stop in (0.006, 0.010):
                        for session_label, hours in momentum_sessions.items():
                            name = (
                                f"fresh_pair_resid_mom_spread_lb{lookback}_"
                                f"z{int(threshold * 100):03d}_h{hold}_"
                                f"sc{str(scale).replace('.', '')}_st{int(stop * 10000)}_{session_label}"
                            )
                            specs[name] = FreshSpec(
                                name=name,
                                family="residual_pair_momentum_spread",
                                lookback_bars=lookback,
                                threshold=threshold,
                                hold_bars=hold,
                                cooldown_bars=max(1, hold // 3),
                                stop_loss_pct=stop,
                                take_profit_pct=0.030,
                                min_abs_return=0.0,
                                entry_hours=hours,
                                allow_long=True,
                                allow_short=True,
                                long_allocation_scale=scale,
                                short_allocation_scale=scale,
                                spread_hedge_ratio=1.0,
                            )
    for lookback in (6, 12, 24):
        for threshold in (0.006, 0.010, 0.014):
            for comp in (0.55, 0.70, 0.85):
                name = f"fresh_compression_breakout_lb{lookback}_thr{int(threshold * 10000)}_c{int(comp * 100)}"
                specs[name] = FreshSpec(
                    name=name,
                    family="compression_breakout",
                    lookback_bars=lookback,
                    threshold=threshold,
                    hold_bars=24,
                    cooldown_bars=12,
                    stop_loss_pct=0.012,
                    take_profit_pct=0.030,
                    rv_lookback_bars=24,
                    compression_lookback_bars=72,
                    compression_quantile=comp,
                    allow_short=True,
                )
    for lookback in (6, 12, 24):
        for threshold in (0.006, 0.010, 0.014, 0.020):
            for comp in (0.55, 0.70, 0.85):
                for hold in (12, 24, 48):
                    name = (
                        f"fresh_compression_fade_lb{lookback}_thr{int(threshold * 10000)}_"
                        f"c{int(comp * 100)}_h{hold}"
                    )
                    specs[name] = FreshSpec(
                        name=name,
                        family="compression_breakout_fade",
                        lookback_bars=lookback,
                        threshold=threshold,
                        hold_bars=hold,
                        cooldown_bars=max(1, hold // 3),
                        stop_loss_pct=0.010,
                        take_profit_pct=0.035,
                        rv_lookback_bars=24,
                        compression_lookback_bars=72,
                        compression_quantile=comp,
                        allow_short=True,
                        trailing_stop_rv_multiple=1.2,
                        trailing_stop_floor_pct=0.005,
                        trailing_stop_cap_pct=0.014,
                    )
    for lookback in (6, 12, 24):
        for threshold in (0.010, 0.014, 0.020):
            for comp in (0.55, 0.70):
                for hold in (12, 24):
                    for scale in (0.5, 1.0):
                        for stop in (0.006, 0.010):
                            for take in (0.012, 0.025):
                                name = (
                                    f"fresh_compression_downside_short_lb{lookback}_"
                                    f"thr{int(threshold * 10000)}_c{int(comp * 100)}_h{hold}_"
                                    f"sc{str(scale).replace('.', '')}_st{int(stop * 10000)}_tp{int(take * 10000)}"
                                )
                                specs[name] = FreshSpec(
                                    name=name,
                                    family="compression_expansion_downside_short",
                                    lookback_bars=lookback,
                                    threshold=threshold,
                                    hold_bars=hold,
                                    cooldown_bars=max(1, hold // 3),
                                    stop_loss_pct=stop,
                                    take_profit_pct=take,
                                    rv_lookback_bars=24,
                                    compression_lookback_bars=72,
                                    compression_quantile=comp,
                                    allow_long=False,
                                    allow_short=True,
                                    long_allocation_scale=0.0,
                                    short_allocation_scale=scale,
                                    trailing_stop_rv_multiple=1.0,
                                    trailing_stop_floor_pct=0.004,
                                    trailing_stop_cap_pct=0.010,
                                )
    return list(specs.values())


def _evaluate_specs(
    *, specs: list[FreshSpec], arrays: dict[str, Any], splits: list[SplitWindow]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    for spec in specs:
        split_results = {
            split.name: _run_split(spec=spec, arrays=arrays, split=split) for split in splits
        }
        metrics = {
            name: dict(result.get("metrics") or {}) for name, result in split_results.items()
        }
        train = metrics.get("train", {})
        val = metrics.get("val", {})
        oos = metrics.get("oos", {})
        gates = {
            "not_incumbent_or_context_wrapper": not spec.name.startswith(
                "profit_moonshot_hourly_shock"
            ),
            "train_positive": _safe_float(train.get("total_return"), 0.0) > 0.0,
            "val_positive": _safe_float(val.get("total_return"), 0.0) > 0.0,
            "oos_return_beats_incumbent": _safe_float(oos.get("total_return"), 0.0)
            > BASELINE_OOS_RETURN,
            "oos_mdd_beats_shadow": _safe_float(oos.get("max_drawdown"), 1.0) < SHADOW_OOS_MDD,
            "oos_sharpe_gt_1": _safe_float(oos.get("sharpe"), 0.0) > SUCCESS_SHARPE,
            "oos_trades_not_starved": int(split_results["oos"].get("round_trips") or 0) >= 5,
            "liquidations_zero": int(split_results["train"].get("liquidations") or 0) == 0
            and int(split_results["val"].get("liquidations") or 0) == 0
            and int(split_results["oos"].get("liquidations") or 0) == 0,
        }
        replay_survivor = all(
            gates[key]
            for key in (
                "not_incumbent_or_context_wrapper",
                "train_positive",
                "val_positive",
                "oos_return_beats_incumbent",
                "oos_mdd_beats_shadow",
                "oos_trades_not_starved",
                "liquidations_zero",
            )
        )
        success = bool(replay_survivor and gates["oos_sharpe_gt_1"])
        failed = [key for key, ok in gates.items() if not ok]
        row: dict[str, Any] = {
            "name": spec.name,
            "family": spec.family,
            "replay_survivor": bool(replay_survivor),
            "success_candidate": success,
            "failed_gates": ",".join(failed),
            "filters": json.dumps(spec.payload(), sort_keys=True),
            "reject_top": json.dumps(
                split_results["oos"].get("reject_counts") or {}, sort_keys=True
            ),
        }
        for split_name, result in split_results.items():
            split_metrics = dict(result.get("metrics") or {})
            for key in ("total_return", "max_drawdown", "sharpe", "sortino", "volatility"):
                row[f"{split_name}_{key}"] = _safe_float(split_metrics.get(key), 0.0)
            row[f"{split_name}_round_trips"] = int(result.get("round_trips") or 0)
            row[f"{split_name}_fills"] = int(result.get("fills") or 0)
            row[f"{split_name}_liquidations"] = int(result.get("liquidations") or 0)
        rows.append(row)
        results.append(
            {
                "name": spec.name,
                "family": spec.family,
                "filters": spec.payload(),
                "split_results": split_results,
                "gates": gates,
                "failed_gates": failed,
                "replay_survivor": bool(replay_survivor),
                "success_candidate": success,
            }
        )
    rows.sort(
        key=lambda row: (
            not bool(row.get("success_candidate")),
            not bool(row.get("replay_survivor")),
            -_safe_float(row.get("oos_total_return"), 0.0),
            -_safe_float(row.get("oos_sharpe"), 0.0),
            _safe_float(row.get("oos_max_drawdown"), 1.0),
        )
    )
    by_name = {item["name"]: item for item in results}
    results = [by_name[str(row["name"])] for row in rows]
    return rows, results


def _rss_mib() -> float:
    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss or 0)
    if sys.platform == "darwin":
        return peak / (1024.0 * 1024.0)
    return peak / 1024.0


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(
            fp, fieldnames=sorted({key for row in rows for key in row}), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def _markdown(payload: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    top = rows[:15]
    survivors = [r for r in rows if bool(r.get("replay_survivor"))]
    success = [r for r in rows if bool(r.get("success_candidate"))]
    lines = [
        "# Profit moonshot fresh-start overhaul replay",
        "",
        f"Generated: `{payload['generated_at_utc']}`",
        f"OOS end date: `{payload['oos_end_date']}`",
        "",
        "## Intent",
        "",
        "- 기존 ETH shock-reversion incumbent/leadlag/context-wrapper를 쓰지 않고 raw-first data에서 새로 출발했다.",
        "- 신규 후보군: cross-sectional residual reversal, cross-sectional momentum, adaptive trend, cross-sectional Sharpe/rank selector, "
        "funding-carry fade, funding+OI carry fade, taker-flow persistence/exhaustion, non-calendar state-distilled leadership/unwind, "
        "non-calendar crowded-leadership unwind v2, non-calendar TRX state-momentum proxy, "
        "non-calendar TRX/ETH state-relative-strength spread, calendar rotation, calendar-conditioned veto/day-window sleeves, "
        "TRX/ETH calendar spread, compression breakout.",
        "- Replay는 one-position, fee/slippage, 10% bar-volume fill cap, cooldown, stop/take/max-hold, 0.8% target allocation, $175 max order를 강제한다.",
        "",
        "## Gate policy",
        "",
        f"- Success requires OOS return > `{BASELINE_OOS_RETURN:+.4%}`, OOS MDD < `{SHADOW_OOS_MDD:.4%}`, OOS Sharpe > `{SUCCESS_SHARPE:.1f}`, liquidations `0`, and positive train/val.",
        "- Replay survivor는 full live-equivalent raw-first backtest 후보일 뿐이며, sub-1 Sharpe는 성공이 아니다.",
        "",
        "## Result",
        "",
        f"- Specs evaluated: `{len(rows)}`",
        f"- Replay survivors: `{len(survivors)}`",
        f"- Success candidates: `{len(success)}`",
        f"- Peak RSS: `{payload['peak_rss_mib']:.3f} MiB`",
        "",
        "## Top candidates/failures",
        "",
        "| rank | name | family | survivor | success | train ret | val ret | OOS ret | OOS MDD | OOS Sharpe | OOS trips | failed gates |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for idx, row in enumerate(top, start=1):
        lines.append(
            f"| {idx} | `{row['name']}` | `{row['family']}` | {bool(row.get('replay_survivor'))} | {bool(row.get('success_candidate'))} | "
            f"{_fmt_pct(row.get('train_total_return'))} | {_fmt_pct(row.get('val_total_return'))} | {_fmt_pct(row.get('oos_total_return'))} | "
            f"{_fmt_pct(row.get('oos_max_drawdown'))} | {_fmt_float(row.get('oos_sharpe'))} | {int(row.get('oos_round_trips') or 0)} | `{row.get('failed_gates')}` |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
        ]
    )
    if success:
        lines.append(
            "- At least one fresh-start replay candidate earned a one-at-a-time full live-equivalent backtest slot."
        )
    elif survivors:
        lines.append(
            "- Replay survivors exist, but none is a final success until live-equivalent raw-first backtest proves Sharpe > 1.0 and all gates."
        )
    else:
        lines.append(
            "- No fresh-start candidate earned a full live-equivalent slot; do not promote or backtest a random vector-only shape."
        )
    lines.extend(
        [
            "- Blocked/failed families remain recorded in CSV/JSON with failed gates and top reject reasons.",
            "",
        ]
    )
    return "\n".join(lines)


def _split_arg_tokens(raw: Any) -> tuple[str, ...]:
    return tuple(
        token.strip() for token in str(raw or "").replace(";", ",").split(",") if token.strip()
    )


def _filter_specs(
    specs: list[FreshSpec], args: argparse.Namespace
) -> tuple[list[FreshSpec], dict[str, Any]]:
    families = set(_split_arg_tokens(getattr(args, "spec_family", "")))
    name_tokens = _split_arg_tokens(getattr(args, "spec_name_contains", ""))
    max_specs = max(0, int(getattr(args, "max_specs", 0) or 0))
    rejected_non_calendar_specs = {
        spec.name: violations
        for spec in specs
        if (violations := _non_calendar_spec_calendar_fields(spec))
    }
    filtered = [
        spec
        for spec in specs
        if spec.name not in rejected_non_calendar_specs
        and (not families or spec.family in families)
        and (not name_tokens or any(token in spec.name for token in name_tokens))
    ]
    if max_specs > 0:
        filtered = filtered[:max_specs]
    return filtered, {
        "spec_family": sorted(families),
        "spec_name_contains": list(name_tokens),
        "max_specs": max_specs,
        "unfiltered_spec_count": len(specs),
        "filtered_spec_count": len(filtered),
        "rejected_non_calendar_spec_count": len(rejected_non_calendar_specs),
        "rejected_non_calendar_specs": rejected_non_calendar_specs,
    }


def build_payload(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    oos_end = (
        datetime.fromisoformat(str(args.oos_end_date)).date()
        if str(args.oos_end_date or "").strip()
        else date(2026, 5, 6)
    )
    splits = _split_windows(oos_end=oos_end)
    start = min(split.start for split in splits)
    end = max(split.end for split in splits)
    symbols = [item.strip() for item in str(args.symbols).split(",") if item.strip()]
    cache_dir = None if not str(args.panel_cache_dir or "").strip() else Path(args.panel_cache_dir)
    panel, data_metadata = _joined_panel(
        market_root=Path(args.market_root),
        exchange=str(args.exchange),
        symbols=symbols,
        start=start,
        end=end,
        cache_dir=cache_dir,
        refresh_cache=bool(args.refresh_panel_cache),
    )
    panel, external_state_metadata = _join_external_state(
        panel, str(getattr(args, "external_state_csv", "") or "")
    )
    arrays = _build_arrays(panel, symbols)
    unfiltered_specs = _candidate_specs(arrays, symbols)
    specs, spec_filter = _filter_specs(unfiltered_specs, args)
    rows, results = _evaluate_specs(specs=specs, arrays=arrays, splits=splits)
    payload = {
        "artifact_kind": "profit_moonshot_fresh_start_overhaul_replay",
        "generated_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "market_root": str(args.market_root),
        "exchange": str(args.exchange),
        "symbols": symbols,
        "oos_end_date": oos_end.isoformat(),
        "split_windows": [split.as_payload() for split in splits],
        "data_metadata": data_metadata,
        "external_state_metadata": external_state_metadata,
        "spec_filter": spec_filter,
        "gate_policy": {
            "baseline_oos_return": BASELINE_OOS_RETURN,
            "shadow_oos_mdd": SHADOW_OOS_MDD,
            "success_oos_sharpe": SUCCESS_SHARPE,
            "target_allocation": TARGET_ALLOCATION,
            "max_order_value": MAX_ORDER_VALUE,
            "metric_periods_per_year": HOURLY_PERIODS_PER_YEAR,
            "liquidations_required": 0,
        },
        "spec_count": len(specs),
        "replay_survivor_count": sum(1 for row in rows if bool(row.get("replay_survivor"))),
        "success_candidate_count": sum(1 for row in rows if bool(row.get("success_candidate"))),
        "top_results": results[:25],
        "peak_rss_mib": _rss_mib(),
    }
    return payload, rows


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--market-root", default=str(BaseConfig.MARKET_DATA_PARQUET_PATH))
    parser.add_argument("--exchange", default=str(BaseConfig.MARKET_DATA_EXCHANGE))
    parser.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    parser.add_argument("--oos-end-date", default="2026-05-06")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--panel-cache-dir", default=str(DEFAULT_PANEL_CACHE_DIR))
    parser.add_argument("--refresh-panel-cache", action="store_true")
    parser.add_argument(
        "--external-state-csv",
        default="",
        help="Optional lagged daily external market-state CSV with external_* columns.",
    )
    parser.add_argument(
        "--spec-family", default="", help="Comma-separated candidate family allowlist."
    )
    parser.add_argument(
        "--spec-name-contains",
        default="",
        help="Comma-separated substrings; keep matching spec names.",
    )
    parser.add_argument(
        "--max-specs", type=int, default=0, help="Optional cap after spec filters; 0 means no cap."
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "fresh_start_overhaul_replay_latest.json"
    md_path = output_dir / "fresh_start_overhaul_replay_latest.md"
    csv_path = output_dir / "fresh_start_overhaul_replay_candidates.csv"
    memory_guard = acquire_portfolio_memory_guard(
        run_name=RUN_NAME,
        output_dir=output_dir,
        input_path=args.market_root,
        metadata={
            "script": Path(__file__).name,
            "exchange": str(args.exchange),
            "symbols": str(args.symbols),
            "oos_end_date": str(args.oos_end_date),
            "spec_family": str(args.spec_family),
            "spec_name_contains": str(args.spec_name_contains),
            "max_specs": int(args.max_specs),
            "external_state_csv": str(args.external_state_csv),
        },
        budget_bytes=PORTFOLIO_FOLLOWUP_EXPLICIT_BUDGET_BYTES,
    )
    finalized = False
    try:
        memory_guard.checkpoint(
            "start",
            {
                "market_root": str(args.market_root),
                "exchange": str(args.exchange),
                "symbols": str(args.symbols),
                "oos_end_date": str(args.oos_end_date),
                "spec_family": str(args.spec_family),
                "spec_name_contains": str(args.spec_name_contains),
                "max_specs": int(args.max_specs),
                "external_state_csv": str(args.external_state_csv),
            },
        )
        payload, rows = build_payload(args)
        payload["memory_policy"] = memory_policy_payload(
            budget_bytes=PORTFOLIO_FOLLOWUP_EXPLICIT_BUDGET_BYTES
        )
        payload["rss_log_path"] = str(memory_guard.rss_log_path)
        payload["memory_summary_path"] = str(memory_guard.summary_path)
        _write_csv(csv_path, rows)
        memory_guard.checkpoint(
            "artifacts_prepared",
            {
                "spec_count": int(payload["spec_count"]),
                "replay_survivor_count": int(payload["replay_survivor_count"]),
                "success_candidate_count": int(payload["success_candidate_count"]),
            },
        )
        memory_summary = memory_guard.finalize(
            status="completed",
            context={
                "json_path": str(json_path),
                "markdown_path": str(md_path),
                "csv_path": str(csv_path),
                "spec_count": int(payload["spec_count"]),
                "success_candidate_count": int(payload["success_candidate_count"]),
            },
        )
        finalized = True
        memory_summary["summary_path"] = str(memory_guard.summary_path)
        payload["memory_summary"] = memory_summary
        payload["peak_rss_mib"] = max(
            _safe_float(payload.get("peak_rss_mib")),
            _safe_float(memory_summary.get("peak_rss_bytes")) / (1024.0 * 1024.0),
        )
        json_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        md_path.write_text(_markdown(payload, rows) + "\n", encoding="utf-8")
    except Exception as exc:
        if not finalized:
            memory_guard.finalize(
                status="failed", error=str(exc), context={"script": Path(__file__).name}
            )
        raise
    finally:
        memory_guard.release()
    print(
        json.dumps(
            {
                "json": str(json_path),
                "markdown": str(md_path),
                "csv": str(csv_path),
                "spec_count": payload["spec_count"],
                "replay_survivor_count": payload["replay_survivor_count"],
                "success_candidate_count": payload["success_candidate_count"],
                "peak_rss_mib": payload["peak_rss_mib"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
