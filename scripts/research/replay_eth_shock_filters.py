"""Stateful replay screen for ETH 12h shock-reversion filters.

The replay is deliberately narrower than the event-driven backtest.  It reads
committed raw-first 1s materialized parquet, builds completed 1h bars, and then
checks whether candidate entry filters survive a one-position, fee/slippage,
fill-liquidity, cooldown, stop/take-profit, and max-hold state machine before
any full live-equivalent mode is backtested.
"""

from __future__ import annotations

import argparse
import csv
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
_lq_rt = get_default_runtime_config()
BacktestConfig = BacktestConfigView(_lq_rt)
BaseConfig = BacktestConfigView(_lq_rt)
from lumina_quant.market_data import load_futures_feature_points_from_db

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.research.revalidate_live_equivalent_candidates import (  # noqa: E402
    SplitWindow,
    _metrics_from_equity_totals,
    _safe_float,
    _split_windows,
)

DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "var/reports/profit_moonshot_20260501/current_tail_20260506/eth_shock_filter_replay"
)
FEATURE_COLUMNS = (
    "funding_rate",
    "open_interest",
    "taker_buy_base_volume",
    "taker_sell_base_volume",
    "taker_buy_quote_volume",
    "taker_sell_quote_volume",
    "liquidation_long_notional",
    "liquidation_short_notional",
)
BASELINE_OOS_RETURN = 0.008284
FUNDING_GUARD_OOS_SHARPE = 0.111225
FUNDING_GUARD_OOS_MDD = 0.001778


@dataclass(frozen=True, slots=True)
class ReplaySpec:
    name: str
    return_threshold: float
    excluded_hours: tuple[int, ...] = ()
    cooldown_bars: int = 0
    regime_symbols: tuple[str, ...] = ()
    regime_threshold: float = 0.0
    regime_policy: str = "any"
    vol_lookback_bars: int = 0
    max_realized_vol: float = 0.0
    flow_lookback_bars: int = 0
    flow_imbalance_min: float = 0.0
    funding_abs_cap: float = 0.0
    funding_sign_guard: bool = False
    oi_z_min: float = 0.0
    oi_mode: str = ""
    liquidation_lookback_bars: int = 0
    liquidation_z_min: float = 0.0

    def filters_payload(self) -> dict[str, Any]:
        return {
            "return_threshold": self.return_threshold,
            "excluded_hours": list(self.excluded_hours),
            "cooldown_bars": self.cooldown_bars,
            "regime_symbols": list(self.regime_symbols),
            "regime_threshold": self.regime_threshold,
            "regime_policy": self.regime_policy,
            "vol_lookback_bars": self.vol_lookback_bars,
            "max_realized_vol": self.max_realized_vol,
            "flow_lookback_bars": self.flow_lookback_bars,
            "flow_imbalance_min": self.flow_imbalance_min,
            "funding_abs_cap": self.funding_abs_cap,
            "funding_sign_guard": self.funding_sign_guard,
            "oi_z_min": self.oi_z_min,
            "oi_mode": self.oi_mode,
            "liquidation_lookback_bars": self.liquidation_lookback_bars,
            "liquidation_z_min": self.liquidation_z_min,
        }


def _latest_complete_utc_day() -> date:
    configured = str(getattr(BacktestConfig, "END_DATE", "") or "").strip()
    if configured:
        try:
            return datetime.fromisoformat(configured).date()
        except Exception:
            pass
    return datetime.now(UTC).date() - timedelta(days=1)


def _parse_date(raw: str | None) -> date | None:
    token = str(raw or "").strip()
    if not token:
        return None
    return datetime.fromisoformat(token).date()


def _date_iter(start: date, end: date) -> list[date]:
    if end < start:
        return []
    return [start + timedelta(days=offset) for offset in range((end - start).days + 1)]


def _compact_symbol(symbol: str) -> str:
    return str(symbol).replace("/", "").upper()


def _materialized_paths(
    *,
    market_root: Path,
    exchange: str,
    symbol: str,
    timeframe: str,
    start: date,
    end: date,
) -> list[str]:
    paths: list[str] = []
    compact = _compact_symbol(symbol)
    for day in _date_iter(start, end):
        day_root = (
            market_root
            / "market_data_materialized"
            / str(exchange).lower()
            / compact
            / f"timeframe={timeframe}"
            / f"date={day.isoformat()}"
        )
        manifest_path = day_root / "manifest.json"
        if not manifest_path.exists():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if str(manifest.get("status") or "").lower() != "committed":
            continue
        for item in list(manifest.get("data_files") or []):
            path = day_root / str(item)
            if path.exists():
                paths.append(str(path))
    return paths


def _load_hourly_from_raw_first(
    *,
    market_root: Path,
    exchange: str,
    symbol: str,
    start: date,
    end: date,
) -> pl.DataFrame:
    paths = _materialized_paths(
        market_root=market_root,
        exchange=exchange,
        symbol=symbol,
        timeframe="1s",
        start=start,
        end=end,
    )
    if not paths:
        raise RuntimeError(f"no committed raw-first 1s paths for {symbol} {start}..{end}")
    frame = (
        pl.scan_parquet(paths)
        .select(["datetime", "open", "high", "low", "close", "volume"])
        .with_columns(pl.col("datetime").cast(pl.Datetime(time_unit="ms")))
        .sort("datetime")
        .group_by_dynamic("datetime", every="1h", period="1h", closed="left", label="left")
        .agg(
            [
                pl.col("open").first().alias("open"),
                pl.col("high").max().alias("high"),
                pl.col("low").min().alias("low"),
                pl.col("close").last().alias("close"),
                pl.col("volume").sum().alias("volume"),
            ]
        )
        .drop_nulls(["open", "close"])
        .collect()
        .sort("datetime")
    )
    prefix = _compact_symbol(symbol).lower()
    return frame.rename(
        {
            "open": f"{prefix}_open",
            "high": f"{prefix}_high",
            "low": f"{prefix}_low",
            "close": f"{prefix}_close",
            "volume": f"{prefix}_volume",
        }
    )


def _align_feature_columns(frame: pl.DataFrame) -> pl.DataFrame:
    out = frame
    for column in ("timestamp_ms", *FEATURE_COLUMNS):
        if column not in out.columns:
            dtype = pl.Int64 if column == "timestamp_ms" else pl.Float64
            out = out.with_columns(pl.lit(None, dtype=dtype).alias(column))
    return out.select(["timestamp_ms", *FEATURE_COLUMNS])


def _load_eth_feature_hourly(
    *,
    market_root: Path,
    exchange: str,
    start: date,
    end: date,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    frame = load_futures_feature_points_from_db(
        str(market_root),
        exchange=exchange,
        symbol="ETH/USDT",
        start_date=datetime.combine(start, datetime.min.time()),
        end_date=datetime.combine(end + timedelta(days=1), datetime.min.time()),
    )
    if frame.is_empty():
        empty = pl.DataFrame({"datetime": []}, schema={"datetime": pl.Datetime(time_unit="ms")})
        return empty, {"rows": 0, "has_taker_flow": False, "has_liquidation": False}

    aligned = _align_feature_columns(frame).filter(pl.col("timestamp_ms").is_not_null())
    if aligned.is_empty():
        empty = pl.DataFrame({"datetime": []}, schema={"datetime": pl.Datetime(time_unit="ms")})
        return empty, {"rows": 0, "has_taker_flow": False, "has_liquidation": False}

    taker_expr = (
        pl.col("taker_buy_quote_volume").is_not_null()
        | pl.col("taker_sell_quote_volume").is_not_null()
        | pl.col("taker_buy_base_volume").is_not_null()
        | pl.col("taker_sell_base_volume").is_not_null()
    )
    liquidation_expr = (
        pl.col("liquidation_long_notional").is_not_null()
        | pl.col("liquidation_short_notional").is_not_null()
    )
    metadata = {
        "rows": int(aligned.height),
        "has_taker_flow": bool(aligned.select(taker_expr.any()).item()),
        "has_liquidation": bool(aligned.select(liquidation_expr.any()).item()),
        "first_timestamp_utc": datetime.fromtimestamp(
            int(aligned.select(pl.min("timestamp_ms")).item()) / 1000.0, tz=UTC
        ).isoformat(),
        "last_timestamp_utc": datetime.fromtimestamp(
            int(aligned.select(pl.max("timestamp_ms")).item()) / 1000.0, tz=UTC
        ).isoformat(),
    }
    hourly = (
        aligned.with_columns(
            pl.from_epoch(pl.col("timestamp_ms"), time_unit="ms").alias("datetime")
        )
        .sort("datetime")
        .group_by_dynamic("datetime", every="1h", period="1h", closed="left", label="left")
        .agg(
            [
                pl.col("funding_rate").drop_nulls().last().alias("funding_rate"),
                pl.col("open_interest").drop_nulls().last().alias("open_interest"),
                pl.col("taker_buy_quote_volume").sum().alias("taker_buy_quote_volume"),
                pl.col("taker_sell_quote_volume").sum().alias("taker_sell_quote_volume"),
                pl.col("taker_buy_base_volume").sum().alias("taker_buy_base_volume"),
                pl.col("taker_sell_base_volume").sum().alias("taker_sell_base_volume"),
                pl.col("liquidation_long_notional").sum().alias("liquidation_long_notional"),
                pl.col("liquidation_short_notional").sum().alias("liquidation_short_notional"),
            ]
        )
        .sort("datetime")
    )
    return hourly, metadata


def _joined_panel(
    *,
    market_root: Path,
    exchange: str,
    start: date,
    end: date,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    eth = _load_hourly_from_raw_first(
        market_root=market_root, exchange=exchange, symbol="ETH/USDT", start=start, end=end
    )
    btc = _load_hourly_from_raw_first(
        market_root=market_root, exchange=exchange, symbol="BTC/USDT", start=start, end=end
    ).select(["datetime", "btcusdt_close"])
    sol = _load_hourly_from_raw_first(
        market_root=market_root, exchange=exchange, symbol="SOL/USDT", start=start, end=end
    ).select(["datetime", "solusdt_close"])
    features, feature_metadata = _load_eth_feature_hourly(
        market_root=market_root, exchange=exchange, start=start, end=end
    )
    panel = eth.join(btc, on="datetime", how="inner").join(sol, on="datetime", how="inner")
    if not features.is_empty():
        panel = panel.join(features, on="datetime", how="left")
    else:
        for column in FEATURE_COLUMNS:
            panel = panel.with_columns(pl.lit(None, dtype=pl.Float64).alias(column))
    for column in FEATURE_COLUMNS:
        if column not in panel.columns:
            panel = panel.with_columns(pl.lit(None, dtype=pl.Float64).alias(column))
    return panel.sort("datetime"), {"eth_feature_points": feature_metadata}


def _rolling_rms_log_return(close: np.ndarray, lookback: int) -> np.ndarray:
    out = np.full(close.shape, np.nan, dtype=float)
    if lookback <= 1 or close.size < 3:
        return out
    returns = np.full(close.shape, np.nan, dtype=float)
    valid = (close[1:] > 0.0) & (close[:-1] > 0.0)
    returns[1:][valid] = np.log(close[1:][valid] / close[:-1][valid])
    squared = np.where(np.isfinite(returns), returns * returns, np.nan)
    for idx in range(lookback, close.size):
        window = squared[idx - lookback + 1 : idx + 1]
        finite = window[np.isfinite(window)]
        if finite.size:
            out[idx] = float(math.sqrt(float(np.mean(finite))))
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
        total = csum[idx + 1] - csum[idx + 1 - lookback]
        count = ccnt[idx + 1] - ccnt[idx + 1 - lookback]
        out[idx] = float(total) if count > 0 else np.nan
    return out


def _rolling_zscore(values: np.ndarray, lookback: int) -> np.ndarray:
    out = np.full(values.shape, np.nan, dtype=float)
    if lookback <= 2:
        return out
    for idx in range(lookback - 1, values.size):
        window = values[idx + 1 - lookback : idx + 1]
        finite = window[np.isfinite(window)]
        if finite.size < 3:
            continue
        std = float(np.std(finite, ddof=1))
        if std <= 1e-12:
            continue
        out[idx] = float((values[idx] - float(np.mean(finite))) / std)
    return out


def _pct_change(values: np.ndarray, lookback: int) -> np.ndarray:
    out = np.full(values.shape, np.nan, dtype=float)
    if lookback <= 0:
        return out
    base = values[:-lookback]
    latest = values[lookback:]
    valid = np.isfinite(base) & np.isfinite(latest) & (base > 0.0)
    out[lookback:][valid] = latest[valid] / base[valid] - 1.0
    return out


def _to_float_array(panel: pl.DataFrame, column: str) -> np.ndarray:
    if column not in panel.columns:
        return np.full(panel.height, np.nan, dtype=float)
    return np.asarray(
        [float(value) if value is not None else np.nan for value in panel[column].to_list()],
        dtype=float,
    )


def _build_arrays(panel: pl.DataFrame) -> dict[str, Any]:
    datetimes = panel["datetime"].to_list()
    arrays: dict[str, Any] = {
        "datetime": datetimes,
        "timestamp": np.asarray(
            [int(dt.replace(tzinfo=UTC).timestamp()) for dt in datetimes], dtype=np.int64
        ),
    }
    for column in panel.columns:
        if column == "datetime":
            continue
        arrays[column] = _to_float_array(panel, column)
    eth_close = arrays["ethusdt_close"]
    arrays["shock_return_12h"] = _pct_change(eth_close, 12)
    arrays["btc_return_24h"] = _pct_change(arrays["btcusdt_close"], 24)
    arrays["sol_return_24h"] = _pct_change(arrays["solusdt_close"], 24)
    arrays["rv_24h"] = _rolling_rms_log_return(eth_close, 24)
    arrays["rv_48h"] = _rolling_rms_log_return(eth_close, 48)
    arrays["rv_72h"] = _rolling_rms_log_return(eth_close, 72)
    buy = arrays["taker_buy_quote_volume"]
    sell = arrays["taker_sell_quote_volume"]
    if np.nansum(buy) <= 0.0 and np.nansum(sell) <= 0.0:
        buy = arrays["taker_buy_base_volume"] * eth_close
        sell = arrays["taker_sell_base_volume"] * eth_close
    for lookback in (1, 3, 6):
        buy_sum = _rolling_sum(buy, lookback)
        sell_sum = _rolling_sum(sell, lookback)
        total = buy_sum + sell_sum
        imbalance = np.divide(
            buy_sum - sell_sum,
            total,
            out=np.full(total.shape, np.nan, dtype=float),
            where=np.isfinite(total) & (total > 0.0),
        )
        arrays[f"flow_imbalance_{lookback}h"] = imbalance
    oi = arrays["open_interest"]
    arrays["oi_delta"] = np.diff(oi, prepend=np.nan)
    arrays["oi_delta_z_72h"] = _rolling_zscore(arrays["oi_delta"], 72)
    liq_long = arrays["liquidation_long_notional"]
    liq_short = arrays["liquidation_short_notional"]
    arrays["long_liq_z_72h"] = _rolling_zscore(liq_long, 72)
    arrays["short_liq_z_72h"] = _rolling_zscore(liq_short, 72)
    return arrays


def _fill_price(price: float, side: str, *, high_low_vol: float) -> tuple[float, float]:
    slippage = 0.0005 * (2.0 if high_low_vol > 0.01 else 1.0)
    penalty = slippage + 0.0001
    if side == "BUY":
        return price * (1.0 + penalty), 0.001
    return price * (1.0 - penalty), 0.001


def _hour_utc(dt: datetime) -> int:
    return int(dt.replace(tzinfo=UTC).hour)


def _filter_passes(
    spec: ReplaySpec,
    arrays: dict[str, Any],
    idx: int,
    signal: str,
) -> tuple[bool, str]:
    dt = arrays["datetime"][idx]
    if spec.excluded_hours and _hour_utc(dt) in set(spec.excluded_hours):
        return False, "funding_hour_excluded"

    if spec.regime_symbols and spec.regime_threshold > 0.0:
        checks: list[bool] = []
        for symbol in spec.regime_symbols:
            key = "btc_return_24h" if symbol == "BTC/USDT" else "sol_return_24h"
            value = float(arrays[key][idx])
            if not math.isfinite(value):
                return False, f"regime_missing:{symbol}"
            if signal == "LONG":
                checks.append(value <= -spec.regime_threshold)
            else:
                checks.append(value >= spec.regime_threshold)
        blocked = all(checks) if spec.regime_policy == "all" else any(checks)
        if blocked:
            return False, "regime_counterguard"

    if spec.max_realized_vol > 0.0 and spec.vol_lookback_bars > 1:
        key = f"rv_{spec.vol_lookback_bars}h"
        value = float(arrays[key][idx])
        if not math.isfinite(value):
            return False, "realized_vol_missing"
        if value > spec.max_realized_vol:
            return False, "realized_vol_too_high"

    funding = float(arrays["funding_rate"][idx])
    if spec.funding_abs_cap > 0.0:
        if not math.isfinite(funding):
            return False, "funding_missing"
        if abs(funding) > spec.funding_abs_cap:
            return False, "funding_abs_cap"
    if spec.funding_sign_guard:
        if not math.isfinite(funding):
            return False, "funding_missing"
        if signal == "LONG" and funding > spec.funding_abs_cap:
            return False, "funding_sign_long"
        if signal == "SHORT" and funding < -spec.funding_abs_cap:
            return False, "funding_sign_short"

    if spec.flow_lookback_bars > 0 and spec.flow_imbalance_min > 0.0:
        key = f"flow_imbalance_{spec.flow_lookback_bars}h"
        value = float(arrays[key][idx])
        if not math.isfinite(value):
            return False, "taker_flow_missing"
        if signal == "LONG" and value > -spec.flow_imbalance_min:
            return False, "taker_sell_exhaustion_missing"
        if signal == "SHORT" and value < spec.flow_imbalance_min:
            return False, "taker_buy_exhaustion_missing"

    if spec.oi_z_min > 0.0:
        value = float(arrays["oi_delta_z_72h"][idx])
        if not math.isfinite(value):
            return False, "oi_z_missing"
        if spec.oi_mode == "unwind":
            if signal == "LONG" and value > -spec.oi_z_min:
                return False, "oi_long_unwind_missing"
            if signal == "SHORT" and value < spec.oi_z_min:
                return False, "oi_short_crowding_missing"
        elif spec.oi_mode == "flush":
            if value > -spec.oi_z_min:
                return False, "oi_flush_missing"
        elif abs(value) < spec.oi_z_min:
            return False, "oi_abs_z_missing"

    if spec.liquidation_lookback_bars > 0 and spec.liquidation_z_min > 0.0:
        key = "long_liq_z_72h" if signal == "LONG" else "short_liq_z_72h"
        value = float(arrays[key][idx])
        if not math.isfinite(value):
            return False, "liquidation_feature_missing"
        if value < spec.liquidation_z_min:
            return False, "liquidation_exhaustion_missing"

    return True, ""


def _run_split(
    *,
    spec: ReplaySpec,
    arrays: dict[str, Any],
    split: SplitWindow,
    initial_equity: float = 10_000.0,
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
        return {"metrics": {}, "round_trips": 0, "fills": 0, "reject_counts": {"split_empty": 1}}

    cash = float(initial_equity)
    qty = 0.0
    position = "OUT"
    entry_price = 0.0
    bars_held = 0
    cooldown = 0
    equity_history: list[float] = []
    round_trips = 0
    fills = 0
    reject_counts: dict[str, int] = {}

    eth_close = arrays["ethusdt_close"]
    eth_open = arrays["ethusdt_open"]
    eth_high = arrays["ethusdt_high"]
    eth_low = arrays["ethusdt_low"]
    eth_volume = arrays["ethusdt_volume"]
    shock = arrays["shock_return_12h"]

    def mark_equity(price: float) -> float:
        if position == "LONG":
            return cash + qty * price
        if position == "SHORT":
            return cash - qty * price
        return cash

    def record_reject(reason: str) -> None:
        reject_counts[reason] = int(reject_counts.get(reason, 0)) + 1

    for idx in indices:
        close = float(eth_close[idx])
        if not math.isfinite(close) or close <= 0.0:
            continue
        high = float(eth_high[idx])
        low = float(eth_low[idx])
        open_ = float(eth_open[idx])
        volume = max(0.0, float(eth_volume[idx]) if math.isfinite(float(eth_volume[idx])) else 0.0)
        high_low_vol = max(0.0, (high - low) / open_) if open_ > 0.0 else 0.0

        if position != "OUT":
            bars_held += 1
            exit_reason = ""
            exit_price = close
            if position == "LONG":
                stop = entry_price * (1.0 - 0.05)
                take = entry_price * (1.0 + 0.10)
                if low <= stop:
                    exit_reason = "stop"
                    exit_price = min(open_, stop) if open_ < stop else stop
                elif high >= take:
                    exit_reason = "take_profit"
                    exit_price = max(open_, take) if open_ > take else take
                elif bars_held >= 72:
                    exit_reason = "max_hold"
                if exit_reason:
                    fill, fee_rate = _fill_price(exit_price, "SELL", high_low_vol=high_low_vol)
                    cash += qty * fill - qty * fill * fee_rate
            else:
                stop = entry_price * (1.0 + 0.05)
                take = entry_price * (1.0 - 0.10)
                if high >= stop:
                    exit_reason = "stop"
                    exit_price = max(open_, stop) if open_ > stop else stop
                elif low <= take:
                    exit_reason = "take_profit"
                    exit_price = min(open_, take) if open_ < take else take
                elif bars_held >= 72:
                    exit_reason = "max_hold"
                if exit_reason:
                    fill, fee_rate = _fill_price(exit_price, "BUY", high_low_vol=high_low_vol)
                    cash -= qty * fill + qty * fill * fee_rate
            if exit_reason:
                position = "OUT"
                qty = 0.0
                entry_price = 0.0
                bars_held = 0
                cooldown = max(0, int(spec.cooldown_bars))
                round_trips += 1
                fills += 1

        if position == "OUT":
            if cooldown > 0:
                cooldown -= 1
                record_reject("cooldown")
            else:
                shock_return = float(shock[idx])
                signal = ""
                if math.isfinite(shock_return):
                    if shock_return >= spec.return_threshold:
                        signal = "SHORT"
                    elif shock_return <= -spec.return_threshold:
                        signal = "LONG"
                if signal:
                    passed, reason = _filter_passes(spec, arrays, idx, signal)
                    if passed:
                        notional = min(0.008 * mark_equity(close), 175.0)
                        raw_qty = math.floor((notional / close) / 0.001) * 0.001
                        max_fill_qty = volume * 0.10
                        order_qty = min(raw_qty, max_fill_qty)
                        if order_qty * close < 5.0 or order_qty <= 0.0:
                            record_reject("fill_or_min_notional")
                        elif signal == "LONG":
                            fill, fee_rate = _fill_price(close, "BUY", high_low_vol=high_low_vol)
                            cash -= order_qty * fill + order_qty * fill * fee_rate
                            qty = order_qty
                            entry_price = fill
                            position = "LONG"
                            bars_held = 0
                            fills += 1
                        else:
                            fill, fee_rate = _fill_price(close, "SELL", high_low_vol=high_low_vol)
                            cash += order_qty * fill - order_qty * fill * fee_rate
                            qty = order_qty
                            entry_price = fill
                            position = "SHORT"
                            bars_held = 0
                            fills += 1
                    else:
                        record_reject(reason)

        equity_history.append(mark_equity(close))

    if position != "OUT" and indices.size:
        idx = int(indices[-1])
        close = float(eth_close[idx])
        high_low_vol = 0.0
        if position == "LONG":
            fill, fee_rate = _fill_price(close, "SELL", high_low_vol=high_low_vol)
            cash += qty * fill - qty * fill * fee_rate
        else:
            fill, fee_rate = _fill_price(close, "BUY", high_low_vol=high_low_vol)
            cash -= qty * fill + qty * fill * fee_rate
        equity_history[-1] = cash
        round_trips += 1
        fills += 1

    metrics = _metrics_from_equity_totals(
        equity_history, periods=int(getattr(BacktestConfig, "ANNUAL_PERIODS", 252))
    )
    return {
        "metrics": metrics,
        "round_trips": int(round_trips),
        "fills": int(fills),
        "final_equity": float(equity_history[-1]) if equity_history else float(initial_equity),
        "reject_counts": dict(sorted(reject_counts.items(), key=lambda item: (-item[1], item[0]))),
    }


def _vol_caps(arrays: dict[str, Any]) -> list[tuple[int, float, str]]:
    caps: list[tuple[int, float, str]] = []
    for lookback in (24, 48, 72):
        values = arrays[f"rv_{lookback}h"]
        finite = values[np.isfinite(values)]
        if finite.size:
            for quantile in (0.55, 0.70, 0.85):
                caps.append(
                    (lookback, float(np.quantile(finite, quantile)), f"q{int(quantile * 100)}")
                )
    return caps


def _candidate_specs(arrays: dict[str, Any]) -> list[ReplaySpec]:
    specs: list[ReplaySpec] = [
        ReplaySpec(name="replay_base_12h_threshold_100bp", return_threshold=0.010),
        ReplaySpec(
            name="replay_funding_guard_current_threshold_80bp",
            return_threshold=0.008,
            excluded_hours=(0, 1, 8, 9, 16, 17),
        ),
    ]
    funding_sets = {
        "funding_exact": (0, 8, 16),
        "funding_adjacent": (0, 1, 8, 9, 16, 17),
        "funding_wide": (23, 0, 1, 7, 8, 9, 15, 16, 17),
        "post_funding_only_exclusion": (0, 8, 16, 23, 7, 15),
    }
    for label, hours in funding_sets.items():
        specs.append(
            ReplaySpec(
                name=f"replay_{label}_threshold_80bp",
                return_threshold=0.008,
                excluded_hours=tuple(sorted(set(hours))),
            )
        )

    base_hours = (0, 1, 8, 9, 16, 17)
    for prefix, threshold, hours in (
        ("base", 0.010, ()),
        ("funding_guard", 0.008, base_hours),
    ):
        for symbols, policy in (
            (("BTC/USDT",), "any"),
            (("SOL/USDT",), "any"),
            (("BTC/USDT", "SOL/USDT"), "any"),
            (("BTC/USDT", "SOL/USDT"), "all"),
        ):
            for regime_threshold in (0.015, 0.025, 0.035):
                token = "_".join(symbol.split("/", 1)[0].lower() for symbol in symbols)
                specs.append(
                    ReplaySpec(
                        name=f"replay_{prefix}_{token}_{policy}_regime_{int(regime_threshold * 10000)}bp",
                        return_threshold=threshold,
                        excluded_hours=hours,
                        regime_symbols=symbols,
                        regime_threshold=regime_threshold,
                        regime_policy=policy,
                    )
                )
        for lookback, cap, qlabel in _vol_caps(arrays):
            specs.append(
                ReplaySpec(
                    name=f"replay_{prefix}_rv{lookback}_{qlabel}_cap",
                    return_threshold=threshold,
                    excluded_hours=hours,
                    vol_lookback_bars=lookback,
                    max_realized_vol=cap,
                )
            )
        for lookback in (1, 3, 6):
            for imbalance in (0.05, 0.10, 0.15):
                specs.append(
                    ReplaySpec(
                        name=f"replay_{prefix}_taker_flow_{lookback}h_{int(imbalance * 100)}pct",
                        return_threshold=threshold,
                        excluded_hours=hours,
                        flow_lookback_bars=lookback,
                        flow_imbalance_min=imbalance,
                    )
                )
        for cap in (0.00015, 0.00030):
            specs.append(
                ReplaySpec(
                    name=f"replay_{prefix}_funding_abs_{int(cap * 1_000_000)}ppm",
                    return_threshold=threshold,
                    excluded_hours=hours,
                    funding_abs_cap=cap,
                )
            )
            for oi_z in (0.5, 1.0):
                for mode in ("unwind", "flush", "abs"):
                    specs.append(
                        ReplaySpec(
                            name=f"replay_{prefix}_funding_{int(cap * 1_000_000)}ppm_oi_{mode}_z{str(oi_z).replace('.', '')}",
                            return_threshold=threshold,
                            excluded_hours=hours,
                            funding_abs_cap=cap,
                            funding_sign_guard=True,
                            oi_z_min=oi_z,
                            oi_mode=mode,
                        )
                    )
        for liq_z in (0.5, 1.0):
            specs.append(
                ReplaySpec(
                    name=f"replay_{prefix}_liquidation_exhaustion_z{str(liq_z).replace('.', '')}",
                    return_threshold=threshold,
                    excluded_hours=hours,
                    liquidation_lookback_bars=72,
                    liquidation_z_min=liq_z,
                )
            )

    # Focused combinations from the single-filter families above.
    for threshold, prefix, hours in ((0.010, "base", ()), (0.008, "funding_guard", base_hours)):
        for regime_threshold in (0.015, 0.025):
            for lookback, cap, qlabel in _vol_caps(arrays)[:6]:
                specs.append(
                    ReplaySpec(
                        name=f"replay_{prefix}_btc_sol_regime_{int(regime_threshold * 10000)}bp_rv{lookback}_{qlabel}",
                        return_threshold=threshold,
                        excluded_hours=hours,
                        regime_symbols=("BTC/USDT", "SOL/USDT"),
                        regime_threshold=regime_threshold,
                        regime_policy="any",
                        vol_lookback_bars=lookback,
                        max_realized_vol=cap,
                    )
                )
            for flow_lookback in (1, 3):
                specs.append(
                    ReplaySpec(
                        name=f"replay_{prefix}_btc_sol_regime_{int(regime_threshold * 10000)}bp_flow{flow_lookback}h",
                        return_threshold=threshold,
                        excluded_hours=hours,
                        regime_symbols=("BTC/USDT", "SOL/USDT"),
                        regime_threshold=regime_threshold,
                        regime_policy="any",
                        flow_lookback_bars=flow_lookback,
                        flow_imbalance_min=0.05,
                    )
                )

    unique: dict[str, ReplaySpec] = {}
    for spec in specs:
        unique.setdefault(spec.name, spec)
    return list(unique.values())


def _evaluate_specs(
    *,
    specs: list[ReplaySpec],
    arrays: dict[str, Any],
    splits: list[SplitWindow],
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
        oos = metrics.get("oos", {})
        train = metrics.get("train", {})
        val = metrics.get("val", {})
        absolute_gates = {
            "train_positive": _safe_float(train.get("total_return"), 0.0) > 0.0,
            "val_positive": _safe_float(val.get("total_return"), 0.0) > 0.0,
            "oos_return_beats_baseline": _safe_float(oos.get("total_return"), 0.0)
            > BASELINE_OOS_RETURN,
            "oos_sharpe_beats_funding_guard": _safe_float(oos.get("sharpe"), 0.0)
            > FUNDING_GUARD_OOS_SHARPE,
            "oos_mdd_beats_funding_guard": _safe_float(oos.get("max_drawdown"), 1.0)
            < FUNDING_GUARD_OOS_MDD,
            "oos_sharpe_gt_1": _safe_float(oos.get("sharpe"), 0.0) > 1.0,
            "oos_trades_not_starved": int(split_results["oos"].get("round_trips") or 0) >= 5,
        }
        absolute_shape = all(
            absolute_gates[key]
            for key in (
                "train_positive",
                "val_positive",
                "oos_return_beats_baseline",
                "oos_sharpe_beats_funding_guard",
                "oos_mdd_beats_funding_guard",
                "oos_trades_not_starved",
            )
        )
        row: dict[str, Any] = {
            "name": spec.name,
            "replay_survivor": False,
            "absolute_gate_shape": bool(absolute_shape),
            "success_candidate": bool(absolute_shape and absolute_gates["oos_sharpe_gt_1"]),
            "filters": json.dumps(spec.filters_payload(), sort_keys=True),
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
        rows.append(row)
        results.append(
            {
                "name": spec.name,
                "filters": spec.filters_payload(),
                "split_results": split_results,
                "absolute_gates": absolute_gates,
                "replay_relative_gates": {},
                "replay_survivor": False,
                "absolute_gate_shape": bool(absolute_shape),
                "success_candidate": bool(absolute_shape and absolute_gates["oos_sharpe_gt_1"]),
            }
        )
    by_name = {str(row["name"]): row for row in rows}
    replay_base = by_name.get("replay_base_12h_threshold_100bp", {})
    replay_guard = by_name.get("replay_funding_guard_current_threshold_80bp", {})
    replay_return_floor = max(
        _safe_float(replay_base.get("oos_total_return"), 0.0),
        _safe_float(replay_guard.get("oos_total_return"), 0.0),
    )
    replay_sharpe_floor = max(
        _safe_float(replay_base.get("oos_sharpe"), 0.0),
        _safe_float(replay_guard.get("oos_sharpe"), 0.0),
    )
    replay_mdd_floor = min(
        _safe_float(replay_base.get("oos_max_drawdown"), 1.0),
        _safe_float(replay_guard.get("oos_max_drawdown"), 1.0),
    )
    result_by_name = {str(item["name"]): item for item in results}
    for row in rows:
        relative_gates = {
            "train_positive": _safe_float(row.get("train_total_return"), 0.0) > 0.0,
            "val_positive": _safe_float(row.get("val_total_return"), 0.0) > 0.0,
            "oos_return_beats_replay_incumbents": _safe_float(row.get("oos_total_return"), 0.0)
            > replay_return_floor,
            "oos_sharpe_beats_replay_incumbents": _safe_float(row.get("oos_sharpe"), 0.0)
            > replay_sharpe_floor,
            "oos_mdd_beats_replay_incumbents": _safe_float(row.get("oos_max_drawdown"), 1.0)
            < replay_mdd_floor,
            "oos_trades_not_starved": int(row.get("oos_round_trips") or 0) >= 5,
        }
        survivor = all(relative_gates.values())
        row["replay_survivor"] = bool(survivor)
        row["replay_relative_gates"] = json.dumps(relative_gates, sort_keys=True)
        result = result_by_name[str(row["name"])]
        result["replay_relative_gates"] = relative_gates
        result["replay_survivor"] = bool(survivor)
    rows.sort(
        key=lambda row: (
            not bool(row["replay_survivor"]),
            -_safe_float(row.get("oos_total_return"), 0.0),
            -_safe_float(row.get("oos_sharpe"), 0.0),
            _safe_float(row.get("oos_max_drawdown"), 1.0),
        )
    )
    results = [result_by_name[str(row["name"])] for row in rows]
    return rows, results


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)


def _fmt_pct(value: Any) -> str:
    return f"{_safe_float(value, 0.0):+.4%}"


def _fmt_float(value: Any) -> str:
    return f"{_safe_float(value, 0.0):.6f}"


def _markdown(payload: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    survivors = [row for row in rows if bool(row.get("replay_survivor"))]
    success = [row for row in rows if bool(row.get("success_candidate"))]
    lines = [
        "# ETH 12h shock-reversion stateful replay",
        "",
        f"Generated: `{payload['generated_at_utc']}`",
        f"OOS window ends: `{payload['oos_end_date']}`",
        "",
        "## Gate policy",
        "",
        "- Replay survivors are selected only relative to the replayed incumbent and funding-guard shapes; they are **not** final wins.",
        f"- OOS return must beat `{BASELINE_OOS_RETURN:+.4%}`.",
        f"- OOS Sharpe/MDD must beat funding-guard shadow Sharpe `{FUNDING_GUARD_OOS_SHARPE:.6f}` and MDD `{FUNDING_GUARD_OOS_MDD:.4%}`.",
        "- Those absolute thresholds are final live-equivalent backtest gates; Sharpe > 1.0 is required for final success and sub-1 full-backtest survivors are shadow-only.",
        "- Replay enforces one ETH position, 0.8% target allocation, $175 max order value, taker fee/slippage, 10% bar-volume fill cap, cooldown, 5% stop, 10% take-profit, and 72h max hold.",
        "",
        "## Result",
        "",
        f"- Specs evaluated: `{len(rows)}`",
        f"- Replay-relative survivors for one-at-a-time full backtest slots: `{len(survivors)}`",
        f"- Replay rows with absolute final-gate shape and Sharpe>1: `{len(success)}`",
        "",
        "| rank | replay spec | survivor | train ret | val ret | OOS ret | OOS Sharpe | OOS MDD | OOS trips | top OOS rejects |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for rank, row in enumerate(rows[:30], start=1):
        lines.append(
            f"| {rank} | `{row['name']}` | `{bool(row['replay_survivor'])}` | "
            f"{_fmt_pct(row.get('train_total_return'))} | {_fmt_pct(row.get('val_total_return'))} | "
            f"{_fmt_pct(row.get('oos_total_return'))} | {_fmt_float(row.get('oos_sharpe'))} | "
            f"{_fmt_pct(row.get('oos_max_drawdown'))} | {int(row.get('oos_round_trips') or 0)} | "
            f"`{str(row.get('reject_top') or '')[:120]}` |"
        )
    if not survivors:
        lines.extend(
            [
                "",
                "## No replay survivor",
                "",
                "No candidate earned a full live-equivalent backtest slot because every filter failed at least one replay-relative gate before engine validation.",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "## Full-backtest slots",
                "",
                "Run at most one live-equivalent raw-first mode at a time, starting from the first replay survivor above. A replay survivor is only a slot candidate; promotion still requires train/val/OOS engine evidence and the final gates.",
            ]
        )
    return "\n".join(lines) + "\n"


def build_replay(
    *,
    market_root: Path,
    exchange: str,
    output_dir: Path,
    oos_end: date,
) -> dict[str, Any]:
    splits = _split_windows(oos_end=oos_end)
    load_start = min(split.start for split in splits)
    load_end = max(split.end for split in splits)
    panel, data_metadata = _joined_panel(
        market_root=market_root,
        exchange=exchange,
        start=load_start,
        end=load_end,
    )
    arrays = _build_arrays(panel)
    specs = _candidate_specs(arrays)
    rows, results = _evaluate_specs(specs=specs, arrays=arrays, splits=splits)
    max_rss_kb = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    payload: dict[str, Any] = {
        "artifact_kind": "eth_shock_filter_stateful_replay",
        "generated_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "market_root": str(market_root),
        "exchange": str(exchange),
        "oos_end_date": oos_end.isoformat(),
        "split_windows": [split.as_payload() for split in splits],
        "data_metadata": {
            **data_metadata,
            "hourly_rows": int(panel.height),
            "first_hour": str(panel["datetime"][0]) if panel.height else "",
            "last_hour": str(panel["datetime"][-1]) if panel.height else "",
        },
        "resource_usage": {
            "max_rss_mib": round(max_rss_kb / 1024.0, 3),
            "rss_limit_mib": 8192,
        },
        "gate_thresholds": {
            "baseline_oos_return": BASELINE_OOS_RETURN,
            "funding_guard_oos_sharpe": FUNDING_GUARD_OOS_SHARPE,
            "funding_guard_oos_mdd": FUNDING_GUARD_OOS_MDD,
            "success_sharpe_min": 1.0,
            "replay_survivor_policy": "train/val positive plus OOS return/Sharpe/MDD improvement versus replayed base and funding-guard incumbents",
        },
        "candidate_rows": rows,
        "candidate_results": results,
        "replay_survivors": [row for row in rows if bool(row.get("replay_survivor"))],
        "success_candidates": [row for row in rows if bool(row.get("success_candidate"))],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "eth_shock_filter_replay_latest.json"
    csv_path = output_dir / "eth_shock_filter_replay_candidates.csv"
    md_path = output_dir / "eth_shock_filter_replay_latest.md"
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_csv(csv_path, rows)
    md_path.write_text(_markdown(payload, rows), encoding="utf-8")
    return {
        "payload": payload,
        "paths": {
            "json": str(json_path.resolve()),
            "csv": str(csv_path.resolve()),
            "markdown": str(md_path.resolve()),
        },
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--market-root", default=str(BaseConfig.MARKET_DATA_PARQUET_PATH))
    parser.add_argument("--exchange", default=str(BaseConfig.MARKET_DATA_EXCHANGE))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--oos-end-date",
        default="",
        help="Inclusive OOS end date. Defaults to latest complete UTC day.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    oos_end = _parse_date(args.oos_end_date) or _latest_complete_utc_day()
    result = build_replay(
        market_root=Path(args.market_root),
        exchange=str(args.exchange),
        output_dir=Path(args.output_dir),
        oos_end=oos_end,
    )
    print(json.dumps(result["paths"], ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
