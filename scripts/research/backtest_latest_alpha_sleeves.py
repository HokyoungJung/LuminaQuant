"""Backtest the latest alpha-sleeve strategy additions with fail-closed data audits.

The normal ``lq backtest`` CLI is optimized for the current 1s/raw-first execution
contract.  The newly-added alpha sleeves also need quick 1m research validation
over the existing local legacy parquet partitions.  This script keeps the actual
trading simulation on the repository's event-driven ``Backtest``/``Portfolio``/
``ExecutionModel`` stack, but makes the data contract explicit:

* load only local parquet OHLCV via ``MarketDataRepository``;
* never fill missing bars or synthesize prices;
* fail before simulation if timestamp gaps, duplicate bars, bad OHLC, null/NaN,
  or non-positive prices are detected;
* optionally cross-check symbols against official Binance exchangeInfo.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from lumina_quant.backtesting.backtest import Backtest
from lumina_quant.backtesting.data import HistoricCSVDataHandler
from lumina_quant.backtesting.execution_sim import SimulatedExecutionHandler
from lumina_quant.backtesting.portfolio_backtest import Portfolio
from lumina_quant.configuration import get_default_runtime_config
from lumina_quant.market_data import MarketDataRepository, normalize_symbol
from lumina_quant.strategies.registry import resolve_strategy_class


NEW_ALPHA_SLEEVE_STRATEGIES: tuple[str, ...] = (
    # external_alpha_sleeves.py
    "LiquidityShockReversionStrategy",
    "VolManagedMomentumCrashGateStrategy",
    "FundingDislocationTrendCarryStrategy",
    "OpeningRangeContinuationStrategy",
    # robust_alpha_sleeves.py
    "DonchianAtrTrendStrategy",
    "VolatilitySqueezeBreakoutStrategy",
    "TakerFlowImbalanceContinuationStrategy",
    "PairsSpreadMeanReversionStrategy",
    "CrossSectionalShortTermReversalStrategy",
    # adaptive_crypto_alpha_sleeves.py
    "BenchmarkLeadLagContinuationStrategy",
    "ResidualMomentumRotationStrategy",
    "LowVolatilityMomentumStrategy",
    "NearHighMomentumStrategy",
    "FalseBreakoutReversalStrategy",
    # cross_asset_tradfi_alpha_sleeves.py
    "GoldSilverRatioMeanReversionStrategy",
    "GoldSilverRatioTrendStrategy",
    "EquityMetalRiskRegimeRotationStrategy",
    "EquityBenchmarkResidualReversalStrategy",
    "MetalEquityDivergenceReversalStrategy",
)

CORE_CRYPTO_METAL_SYMBOLS: tuple[str, ...] = (
    "BTC/USDT",
    "ETH/USDT",
    "XRP/USDT",
    "BNB/USDT",
    "SOL/USDT",
    "TRX/USDT",
    "DOGE/USDT",
    "ADA/USDT",
    "TON/USDT",
    "AVAX/USDT",
    "XAU/USDT",
    "XAG/USDT",
    "XPT/USDT",
    "XPD/USDT",
)

# Latest local feature-complete sleeve window has the required funding/flow
# columns for BTC/ETH/SOL.  BNB is intentionally excluded from feature-dependent
# sleeves because the local feature partitions have no taker quote volume for it.
FEATURE_SYMBOLS: tuple[str, ...] = ("BTC/USDT", "ETH/USDT", "SOL/USDT")
PAIR_SYMBOLS: tuple[str, ...] = ("BTC/USDT", "ETH/USDT", "XAU/USDT", "XAG/USDT")
METAL_PAIR_SYMBOLS: tuple[str, ...] = ("XAU/USDT", "XAG/USDT")
CROSS_ASSET_SYMBOLS: tuple[str, ...] = (
    "BTC/USDT",
    "ETH/USDT",
    "XAU/USDT",
    "XAG/USDT",
    "XPT/USDT",
    "XPD/USDT",
    "SPY/USDT",
    "QQQ/USDT",
    "AAPL/USDT",
    "MSFT/USDT",
    "NVDA/USDT",
    "TSLA/USDT",
)

STRATEGY_SYMBOLS: dict[str, tuple[str, ...]] = {
    "FundingDislocationTrendCarryStrategy": FEATURE_SYMBOLS,
    "TakerFlowImbalanceContinuationStrategy": FEATURE_SYMBOLS,
    "PairsSpreadMeanReversionStrategy": PAIR_SYMBOLS,
    "GoldSilverRatioMeanReversionStrategy": METAL_PAIR_SYMBOLS,
    "GoldSilverRatioTrendStrategy": METAL_PAIR_SYMBOLS,
    "EquityMetalRiskRegimeRotationStrategy": CROSS_ASSET_SYMBOLS,
    "EquityBenchmarkResidualReversalStrategy": CROSS_ASSET_SYMBOLS,
    "MetalEquityDivergenceReversalStrategy": CROSS_ASSET_SYMBOLS,
    "ResidualMomentumRotationStrategy": CROSS_ASSET_SYMBOLS,
    "LowVolatilityMomentumStrategy": CROSS_ASSET_SYMBOLS,
    "NearHighMomentumStrategy": CROSS_ASSET_SYMBOLS,
    "CrossSectionalShortTermReversalStrategy": CROSS_ASSET_SYMBOLS,
}

BINANCE_EXCHANGE_INFO_URLS: tuple[str, ...] = (
    "https://fapi.binance.com/fapi/v1/exchangeInfo",
    "https://api.binance.com/api/v3/exchangeInfo",
)

TIMEFRAME_MS = 60_000
DEFAULT_START = "2026-05-01T00:00:00Z"
DEFAULT_END = "2026-05-03T00:00:00Z"


@dataclass(frozen=True, slots=True)
class StrategyRunSpec:
    strategy: str
    symbols: tuple[str, ...]


def _parse_datetime(value: str) -> datetime:
    token = str(value or "").strip()
    if not token:
        raise ValueError("empty datetime")
    parsed = datetime.fromisoformat(token.replace("Z", "+00:00"))
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(UTC).replace(tzinfo=None)
    return parsed


def _iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.replace(tzinfo=UTC).isoformat().replace("+00:00", "Z")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except Exception:
        return float(default)
    return parsed if math.isfinite(parsed) else float(default)


def _compact_symbol(symbol: str) -> str:
    return normalize_symbol(symbol).replace("/", "")


def _strategy_specs(selected: list[str] | None = None) -> list[StrategyRunSpec]:
    requested = tuple(selected or NEW_ALPHA_SLEEVE_STRATEGIES)
    specs: list[StrategyRunSpec] = []
    for strategy in requested:
        symbols = STRATEGY_SYMBOLS.get(strategy, CORE_CRYPTO_METAL_SYMBOLS)
        specs.append(StrategyRunSpec(strategy=strategy, symbols=tuple(symbols)))
    return specs


def _official_binance_symbols(timeout_seconds: int = 20) -> dict[str, Any]:
    symbols: set[str] = set()
    errors: dict[str, str] = {}
    counts: dict[str, int] = {}
    for url in BINANCE_EXCHANGE_INFO_URLS:
        try:
            with urllib.request.urlopen(url, timeout=timeout_seconds) as response:
                payload = json.load(response)
            source_symbols = {
                str(item.get("symbol") or "")
                for item in list(payload.get("symbols") or [])
                if str(item.get("symbol") or "")
            }
            symbols.update(source_symbols)
            counts[url] = len(source_symbols)
        except Exception as exc:
            errors[url] = repr(exc)
            counts[url] = 0
    return {
        "source_urls": list(BINANCE_EXCHANGE_INFO_URLS),
        "symbols": sorted(symbols),
        "errors": errors,
        "source_counts": counts,
    }


def _audit_exchange_symbols(
    symbols: list[str],
    *,
    enabled: bool,
    fail_on_error: bool,
) -> dict[str, Any]:
    compact = sorted({_compact_symbol(symbol) for symbol in symbols})
    if not enabled:
        return {
            "enabled": False,
            "status": "skipped",
            "checked_symbols": compact,
            "unmatched_symbols": [],
        }
    official = _official_binance_symbols()
    official_set = set(official["symbols"])
    unmatched = [symbol for symbol in compact if symbol not in official_set]
    status = "pass" if not unmatched and not official["errors"] else "warn"
    if unmatched:
        status = "fail"
    if fail_on_error and official["errors"]:
        status = "fail"
    return {
        "enabled": True,
        "status": status,
        "checked_symbols": compact,
        "matched_count": len([symbol for symbol in compact if symbol in official_set]),
        "unmatched_symbols": unmatched,
        "source_urls": official["source_urls"],
        "source_counts": official["source_counts"],
        "errors": official["errors"],
    }


def _frame_time_bounds(frame: pl.DataFrame) -> tuple[datetime | None, datetime | None]:
    if frame.is_empty() or "datetime" not in frame.columns:
        return None, None
    return frame["datetime"].min(), frame["datetime"].max()


def audit_ohlcv_frame(
    symbol: str,
    frame: pl.DataFrame,
    *,
    max_gap_ratio: float,
) -> dict[str, Any]:
    """Return a fail-closed OHLCV audit for one already-loaded frame."""
    if frame.is_empty():
        return {
            "symbol": symbol,
            "status": "fail",
            "errors": ["empty_frame"],
            "warnings": [],
            "rows": 0,
        }

    required = ("datetime", "open", "high", "low", "close", "volume")
    missing_columns = [column for column in required if column not in frame.columns]
    if missing_columns:
        return {
            "symbol": symbol,
            "status": "fail",
            "errors": ["missing_columns:" + ",".join(missing_columns)],
            "warnings": [],
            "rows": int(frame.height),
        }

    selected = frame.select(required).sort("datetime")
    first_time, last_time = _frame_time_bounds(selected)
    unique_times = int(selected["datetime"].n_unique())
    duplicates = int(selected.height - unique_times)
    expected = 0
    missing_bars = 0
    gap_ratio = 0.0
    if first_time is not None and last_time is not None:
        expected = int((last_time - first_time).total_seconds() * 1000 // TIMEFRAME_MS) + 1
        missing_bars = max(0, expected - unique_times)
        gap_ratio = float(missing_bars / expected) if expected else 0.0

    errors: list[str] = []
    warnings: list[str] = []
    if duplicates:
        errors.append(f"duplicate_timestamps:{duplicates}")
    if gap_ratio > float(max_gap_ratio):
        errors.append(f"missing_1m_bars:{missing_bars}/{expected}")

    numeric = selected.select(
        [
            *[
                pl.col(column).null_count().alias(f"{column}_nulls")
                for column in ("open", "high", "low", "close", "volume")
            ],
            *[
                pl.col(column).is_nan().sum().alias(f"{column}_nans")
                for column in ("open", "high", "low", "close", "volume")
            ],
            *[
                (pl.col(column) <= 0).sum().alias(f"{column}_nonpositive")
                for column in ("open", "high", "low", "close")
            ],
            (pl.col("volume") < 0).sum().alias("volume_negative"),
            (pl.col("volume") == 0).sum().alias("volume_zero"),
            (pl.col("high") < pl.max_horizontal("open", "close"))
            .sum()
            .alias("high_below_open_close"),
            (pl.col("low") > pl.min_horizontal("open", "close"))
            .sum()
            .alias("low_above_open_close"),
            (pl.col("high") < pl.col("low")).sum().alias("high_below_low"),
        ]
    ).row(0, named=True)

    for key, value in numeric.items():
        count = int(value or 0)
        if count <= 0:
            continue
        if key == "volume_zero":
            warnings.append(f"{key}:{count}")
        else:
            errors.append(f"{key}:{count}")

    first_close = _safe_float(selected["close"][0])
    last_close = _safe_float(selected["close"][-1])
    buy_hold_return = (last_close / first_close - 1.0) if first_close > 0 else 0.0
    return {
        "symbol": symbol,
        "status": "pass" if not errors else "fail",
        "errors": errors,
        "warnings": warnings,
        "rows": int(selected.height),
        "unique_timestamps": unique_times,
        "expected_1m_bars": expected,
        "missing_1m_bars": missing_bars,
        "gap_ratio": gap_ratio,
        "start": _iso(first_time),
        "end": _iso(last_time),
        "first_close": first_close,
        "last_close": last_close,
        "buy_hold_return": buy_hold_return,
    }


def _load_and_audit_data(
    repo: MarketDataRepository,
    *,
    exchange: str,
    symbols: tuple[str, ...],
    start: datetime,
    end: datetime,
    max_gap_ratio: float,
) -> tuple[dict[str, pl.DataFrame], list[dict[str, Any]]]:
    data: dict[str, pl.DataFrame] = {}
    audits: list[dict[str, Any]] = []
    for symbol in symbols:
        frame = repo.load_ohlcv(
            exchange=exchange,
            symbol=symbol,
            timeframe="1m",
            start_date=start,
            end_date=end,
        )
        audit = audit_ohlcv_frame(symbol, frame, max_gap_ratio=max_gap_ratio)
        audits.append(audit)
        if audit["status"] == "pass":
            data[symbol] = frame
    return data, audits


def _feature_audit(
    *,
    db_path: str,
    exchange: str,
    symbols: tuple[str, ...],
    start: datetime,
    end: datetime,
    required_features: tuple[str, ...],
) -> dict[str, Any]:
    if not required_features:
        return {"required_features": [], "status": "not_required", "symbols": {}}
    from lumina_quant.market_data import load_futures_feature_points_from_db

    rows: dict[str, Any] = {}
    empty_symbols: list[str] = []
    complete_symbols: list[str] = []
    missing_features_by_symbol: dict[str, list[str]] = {}
    for symbol in symbols:
        frame = load_futures_feature_points_from_db(
            db_path,
            exchange=exchange,
            symbol=symbol,
            start_date=start,
            end_date=end,
        )
        item: dict[str, Any] = {"rows": int(frame.height), "columns": []}
        if frame.is_empty():
            empty_symbols.append(symbol)
            missing_features_by_symbol[symbol] = list(required_features)
        for feature in required_features:
            count = 0
            if feature in frame.columns:
                count = int(frame.select(pl.col(feature).is_not_null().sum()).item())
            item.setdefault("non_null_counts", {})[feature] = count
            if count > 0:
                item["columns"].append(feature)
            else:
                missing_features_by_symbol.setdefault(symbol, []).append(feature)
        if not missing_features_by_symbol.get(symbol):
            complete_symbols.append(symbol)
        rows[symbol] = item
    if not complete_symbols:
        status = "fail"
    elif missing_features_by_symbol:
        status = "warn"
    else:
        status = "pass"
    return {
        "required_features": list(required_features),
        "status": status,
        "empty_feature_symbols": empty_symbols,
        "complete_feature_symbols": complete_symbols,
        "missing_features_by_symbol": missing_features_by_symbol,
        "symbols": rows,
        "note": "FeaturePointLookup forward-fills only up to its 8h stale limit; no unbounded imputation.",
    }


def _run_backtest(
    *,
    strategy: str,
    symbols: tuple[str, ...],
    data: dict[str, pl.DataFrame],
    start: datetime,
    end: datetime,
    annual_periods: int,
) -> dict[str, Any]:
    runtime_config = get_default_runtime_config()
    runtime_config.trading.timeframe = "1m"
    runtime_config.backtest.annual_periods = int(annual_periods)
    runtime_config.backtest.persist_output = False
    strategy_cls = resolve_strategy_class(strategy)
    params: dict[str, Any] = {}
    started = time.perf_counter()
    backtest = Backtest(
        "data",
        list(symbols),
        start,
        HistoricCSVDataHandler,
        SimulatedExecutionHandler,
        Portfolio,
        strategy_cls,
        strategy_params=params,
        end_date=end,
        data_dict=data,
        record_history=False,
        track_metrics=True,
        record_trades=True,
        strategy_timeframe="1m",
        config=runtime_config,
    )
    backtest.simulate_trading(output=False)
    elapsed = time.perf_counter() - started
    totals = list(getattr(backtest.portfolio, "_metric_totals", []) or [])
    first_total = _safe_float(totals[0], 10_000.0) if totals else 10_000.0
    final_total = _safe_float(backtest.portfolio.current_holdings.get("total"), first_total)
    total_return = final_total / first_total - 1.0 if first_total > 0 else 0.0
    fast_stats = dict(backtest.portfolio.output_summary_stats_fast() or {})
    return {
        "status": "pass",
        "strategy": strategy,
        "symbols": list(symbols),
        "start": _iso(start),
        "end": _iso(end),
        "elapsed_seconds": elapsed,
        "initial_equity": first_total,
        "final_equity": final_total,
        "total_return": total_return,
        "trade_count": int(getattr(backtest.portfolio, "trade_count", 0)),
        "market_events": int(getattr(backtest, "market_events", 0)),
        "signals": int(getattr(backtest, "signals", 0)),
        "orders": int(getattr(backtest, "orders", 0)),
        "fills": int(getattr(backtest, "fills", 0)),
        "fast_stats": fast_stats,
    }


def _benchmark_return(audits: list[dict[str, Any]]) -> float:
    for audit in audits:
        if audit.get("status") == "pass":
            return _safe_float(audit.get("buy_hold_return"))
    return 0.0


def _markdown_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Latest alpha-sleeve backtest report",
        "",
        f"- generated_at: `{payload['generated_at']}`",
        f"- data_root: `{payload['data_root']}`",
        f"- exchange: `{payload['exchange']}`",
        "- timeframe: `1m`",
        f"- period: `{payload['start']}` → `{payload['end']}`",
        f"- annual_periods: `{payload['annual_periods']}`",
        f"- strategy_count: `{len(payload['strategy_results'])}`",
        "",
        "## Data integrity summary",
        "",
    ]
    exchange_audit = payload.get("exchange_symbol_audit") or {}
    feature_status_counts: dict[str, int] = {}
    for feature_audit in dict(payload.get("feature_audits") or {}).values():
        status = str(dict(feature_audit or {}).get("status") or "unknown")
        feature_status_counts[status] = feature_status_counts.get(status, 0) + 1
    data_warning_count = 0
    for audits in dict(payload.get("data_audits") or {}).values():
        for audit in list(audits or []):
            if dict(audit or {}).get("warnings"):
                data_warning_count += 1
    lines.extend(
        [
            f"- exchange symbol audit: `{exchange_audit.get('status')}` "
            f"(matched={exchange_audit.get('matched_count', 'n/a')}, "
            f"unmatched={len(exchange_audit.get('unmatched_symbols') or [])})",
            f"- feature audit statuses: `{feature_status_counts}`",
            f"- OHLCV warning rows: `{data_warning_count}`",
            f"- max allowed 1m gap ratio: `{payload['max_gap_ratio']}`",
            "- OHLCV policy: no gap fill, no interpolation, no synthetic rows; any missing bars above threshold fail before simulation.",
            "- Required external features must be present on at least one traded symbol; absent feature columns are not inferred or filled.",
            "- Zero-volume bars are reported as warnings, not imputed.",
            "",
            "## Strategy results",
            "",
            "| Strategy | Symbols | Return | Sharpe | CAGR | MDD | Trades | Signals | Feature audit | Status |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
        ]
    )
    for row in payload["strategy_results"]:
        stats = dict(row.get("fast_stats") or {})
        lines.append(
            "| `{strategy}` | {symbol_count} | {ret:.2%} | {sharpe:.3f} | "
            "{cagr:.2%} | {mdd:.2%} | {trades} | {signals} | `{feature_status}` | `{status}` |".format(
                strategy=row.get("strategy"),
                symbol_count=len(row.get("symbols") or []),
                ret=_safe_float(row.get("total_return")),
                sharpe=_safe_float(stats.get("sharpe")),
                cagr=_safe_float(stats.get("cagr")),
                mdd=_safe_float(stats.get("max_drawdown")),
                trades=int(row.get("trade_count") or 0),
                signals=int(row.get("signals") or 0),
                feature_status=row.get("feature_audit_status", "not_required"),
                status=row.get("status"),
            )
        )
    lines.extend(["", "## Issues and audit notes", ""])
    issues = list(payload.get("issues") or [])
    if not issues:
        lines.append("- No fail-level data integrity or runtime issues remained after this run.")
    else:
        for issue in issues:
            lines.append(
                f"- `{issue.get('severity')}` `{issue.get('scope')}`: {issue.get('message')}"
            )
    lines.extend(["", "## Output files", ""])
    for key, value in sorted(dict(payload.get("output_paths") or {}).items()):
        lines.append(f"- {key}: `{value}`")
    return "\n".join(lines) + "\n"


def _write_outputs(payload: dict[str, Any], output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = str(payload["generated_at"]).replace(":", "").replace("-", "")
    timestamp = timestamp.replace(".", "_").replace("Z", "Z")
    json_path = output_dir / f"latest_alpha_sleeves_backtest_{timestamp}.json"
    md_path = output_dir / f"latest_alpha_sleeves_backtest_{timestamp}.md"
    latest_json = output_dir / "latest_alpha_sleeves_backtest_latest.json"
    latest_md = output_dir / "latest_alpha_sleeves_backtest_latest.md"
    payload["output_paths"] = {
        "timestamped_json": str(json_path),
        "timestamped_markdown": str(md_path),
        "latest_json": str(latest_json),
        "latest_markdown": str(latest_md),
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    json_path.write_text(text + "\n", encoding="utf-8")
    latest_json.write_text(text + "\n", encoding="utf-8")
    markdown = _markdown_report(payload)
    md_path.write_text(markdown, encoding="utf-8")
    latest_md.write_text(markdown, encoding="utf-8")
    return dict(payload["output_paths"])


def run_latest_alpha_sleeve_backtests(args: argparse.Namespace) -> dict[str, Any]:
    os.environ.setdefault("LQ_BACKTEST_SUPPRESS_PARTIAL_FILL_LOGS", "1")
    os.environ.setdefault("LQ_BACKTEST_SUPPRESS_CIRCUIT_BREAKER_LOGS", "1")
    os.environ.setdefault("LQ_AUDIT_BACKTEST", "0")

    start = _parse_datetime(args.start)
    end = _parse_datetime(args.end)
    if end <= start:
        raise ValueError("--end must be after --start")

    specs = _strategy_specs(args.strategy)
    all_symbols = sorted({symbol for spec in specs for symbol in spec.symbols})
    exchange_audit = _audit_exchange_symbols(
        all_symbols,
        enabled=not bool(args.no_exchange_audit),
        fail_on_error=bool(args.fail_on_exchange_audit_error),
    )
    if exchange_audit.get("status") == "fail":
        raise RuntimeError("exchange symbol audit failed: " + json.dumps(exchange_audit))

    repo = MarketDataRepository(args.data_root)
    strategy_results: list[dict[str, Any]] = []
    data_audits: dict[str, list[dict[str, Any]]] = {}
    feature_audits: dict[str, Any] = {}
    issues: list[dict[str, str]] = []

    for spec in specs:
        print(f"[RUN] {spec.strategy} symbols={','.join(spec.symbols)}", flush=True)
        data, audits = _load_and_audit_data(
            repo,
            exchange=args.exchange,
            symbols=spec.symbols,
            start=start,
            end=end,
            max_gap_ratio=float(args.max_gap_ratio),
        )
        data_audits[spec.strategy] = audits
        failed_audits = [audit for audit in audits if audit.get("status") != "pass"]
        if failed_audits:
            message = "data audit failed for " + ",".join(
                str(audit.get("symbol")) for audit in failed_audits
            )
            issues.append({"severity": "fail", "scope": spec.strategy, "message": message})
            strategy_results.append(
                {
                    "status": "fail",
                    "strategy": spec.strategy,
                    "symbols": list(spec.symbols),
                    "error": message,
                    "benchmark_return_first_symbol": _benchmark_return(audits),
                    "fast_stats": {},
                    "trade_count": 0,
                    "signals": 0,
                    "total_return": 0.0,
                }
            )
            if bool(args.fail_fast):
                raise RuntimeError(message)
            continue

        strategy_cls = resolve_strategy_class(spec.strategy)
        required_features = tuple(
            str(item)
            for item in tuple(getattr(strategy_cls, "required_features", ()) or ())
            if str(item)
        )
        feature_audits[spec.strategy] = _feature_audit(
            db_path=args.data_root,
            exchange=args.exchange,
            symbols=spec.symbols,
            start=start,
            end=end,
            required_features=required_features,
        )
        feature_status = str(feature_audits[spec.strategy].get("status") or "")
        if feature_status == "fail":
            message = "required feature audit failed for all symbols"
            issues.append({"severity": "fail", "scope": spec.strategy, "message": message})
            strategy_results.append(
                {
                    "status": "fail",
                    "strategy": spec.strategy,
                    "symbols": list(spec.symbols),
                    "error": message,
                    "benchmark_return_first_symbol": _benchmark_return(audits),
                    "fast_stats": {},
                    "trade_count": 0,
                    "signals": 0,
                    "total_return": 0.0,
                    "feature_audit_status": feature_status,
                }
            )
            if bool(args.fail_fast):
                raise RuntimeError(message)
            continue

        try:
            result = _run_backtest(
                strategy=spec.strategy,
                symbols=spec.symbols,
                data=data,
                start=start,
                end=end,
                annual_periods=int(args.annual_periods),
            )
            result["benchmark_return_first_symbol"] = _benchmark_return(audits)
            result["feature_audit_status"] = feature_status
            strategy_results.append(result)
        except Exception as exc:
            message = repr(exc)
            issues.append({"severity": "fail", "scope": spec.strategy, "message": message})
            strategy_results.append(
                {
                    "status": "fail",
                    "strategy": spec.strategy,
                    "symbols": list(spec.symbols),
                    "error": message,
                    "benchmark_return_first_symbol": _benchmark_return(audits),
                    "fast_stats": {},
                    "trade_count": 0,
                    "signals": 0,
                    "total_return": 0.0,
                }
            )
            if bool(args.fail_fast):
                raise

    generated_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    payload: dict[str, Any] = {
        "artifact_kind": "latest_alpha_sleeves_backtest_report",
        "generated_at": generated_at,
        "git_commit": _git_commit(),
        "data_root": str(Path(args.data_root).resolve()),
        "exchange": str(args.exchange),
        "timeframe": "1m",
        "start": _iso(start),
        "end": _iso(end),
        "annual_periods": int(args.annual_periods),
        "max_gap_ratio": float(args.max_gap_ratio),
        "new_strategy_source_files": [
            "src/lumina_quant/strategies/external_alpha_sleeves.py",
            "src/lumina_quant/strategies/robust_alpha_sleeves.py",
            "src/lumina_quant/strategies/adaptive_crypto_alpha_sleeves.py",
            "src/lumina_quant/strategies/cross_asset_tradfi_alpha_sleeves.py",
            "src/lumina_quant/indicators/alpha_features.py",
        ],
        "exchange_symbol_audit": exchange_audit,
        "data_audits": data_audits,
        "feature_audits": feature_audits,
        "strategy_results": strategy_results,
        "issues": issues,
        "integrity_policy": {
            "no_gap_fill": True,
            "no_interpolation": True,
            "no_synthetic_rows": True,
            "missing_bar_policy": "fail_closed_before_simulation",
            "duplicate_timestamp_policy": "fail_closed_before_simulation",
            "feature_staleness_policy": "FeaturePointLookup bounded <= 8h forward-fill only",
            "execution_model": "repository Backtest + Portfolio + SimulatedExecutionHandler",
        },
    }
    paths = _write_outputs(payload, Path(args.output_dir))
    print(json.dumps({"output_paths": paths, "issues": issues}, indent=2), flush=True)
    return payload


def _git_commit() -> str:
    try:
        import subprocess

        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return ""


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run audited backtests for the latest alpha-sleeve strategy additions."
    )
    parser.add_argument("--data-root", default="data/market_parquet")
    parser.add_argument("--exchange", default="binance")
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--annual-periods", type=int, default=365 * 24 * 60)
    parser.add_argument("--max-gap-ratio", type=float, default=0.0)
    parser.add_argument(
        "--strategy",
        action="append",
        choices=NEW_ALPHA_SLEEVE_STRATEGIES,
        help="Run one strategy; repeat to select multiple. Defaults to all latest additions.",
    )
    parser.add_argument("--output-dir", default="var/reports/latest_new_strategy_backtests")
    parser.add_argument("--no-exchange-audit", action="store_true")
    parser.add_argument("--fail-on-exchange-audit-error", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    payload = run_latest_alpha_sleeve_backtests(args)
    return 1 if payload.get("issues") else 0


if __name__ == "__main__":
    raise SystemExit(main())
