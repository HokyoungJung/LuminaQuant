"""Binance futures strategy factory pipeline.

This script builds a broad strategy-candidate universe and then ranks a
multi-timeframe shortlist from existing research reports.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from lumina_quant.indicators import (
    normalized_true_range_latest,
    rolling_log_return_volatility_latest,
    trend_efficiency_latest,
    volume_shock_zscore_latest,
)

DEFAULT_TOP10_PLUS_METALS: tuple[str, ...] = (
    "BTC/USDT",
    "ETH/USDT",
    "BNB/USDT",
    "SOL/USDT",
    "XRP/USDT",
    "ADA/USDT",
    "DOGE/USDT",
    "TRX/USDT",
    "AVAX/USDT",
    "LINK/USDT",
    "XAU/USDT:USDT",
    "XAG/USDT:USDT",
)

DEFAULT_TIMEFRAMES: tuple[str, ...] = ("1s", "1m", "5m", "15m", "30m", "1h", "4h", "1d")

DEFAULT_PAIR_SET: tuple[tuple[str, str], ...] = (
    ("BTC/USDT", "ETH/USDT"),
    ("BTC/USDT", "BNB/USDT"),
    ("ETH/USDT", "SOL/USDT"),
    ("XAU/USDT:USDT", "XAG/USDT:USDT"),
)

TIMEFRAME_SECONDS: dict[str, int] = {
    "1s": 1,
    "1m": 60,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "4h": 14400,
    "1d": 86400,
}


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _parse_float(value) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    if out != out:
        return None
    return out


def _normalize_symbol(value: str) -> str:
    token = str(value).strip().upper().replace("-", "/").replace("_", "/")
    if "/" not in token and token.endswith("USDT") and len(token) > 4:
        token = f"{token[:-4]}/USDT"
    return token


def _normalize_timeframe(value: str) -> str:
    return str(value).strip().lower()


def _family_from_name(name: str) -> str:
    token = str(name).strip().lower()
    if token.startswith(("topcap_tsmom", "rolling_breakout")):
        return "trend_overlay"
    if token.startswith(("pair_", "lag_convergence", "mean_reversion_std", "vwap_reversion")):
        return "alpha_market_neutral"
    if token.startswith(("rsi_", "moving_average_")):
        return "momentum_mean_reversion"
    return "other"


def _candidate_identity(name: str, symbols: list[str], timeframes: list[str], params: dict) -> str:
    payload = {
        "name": str(name),
        "symbols": [str(symbol) for symbol in symbols],
        "timeframes": [str(token) for token in timeframes],
        "params": dict(params),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha1(encoded.encode("utf-8")).hexdigest()
    return digest[:16]


def _append_candidate(
    out: list[dict],
    *,
    name: str,
    strategy_class: str,
    symbols: list[str],
    timeframes: list[str],
    params: dict,
    source: str,
) -> None:
    clean_symbols = [_normalize_symbol(symbol) for symbol in symbols if str(symbol).strip()]
    clean_timeframes = [_normalize_timeframe(token) for token in timeframes if str(token).strip()]
    row = {
        "candidate_id": _candidate_identity(name, clean_symbols, clean_timeframes, params),
        "name": str(name),
        "strategy_class": str(strategy_class),
        "family": _family_from_name(name),
        "symbols": clean_symbols,
        "timeframes": clean_timeframes,
        "params": dict(params),
        "source": str(source),
    }
    out.append(row)


def _build_candidate_universe(symbols: list[str], timeframes: list[str]) -> list[dict]:
    normalized_symbols = [_normalize_symbol(symbol) for symbol in symbols if str(symbol).strip()]
    normalized_timeframes = [
        _normalize_timeframe(token)
        for token in timeframes
        if _normalize_timeframe(token) in TIMEFRAME_SECONDS
    ]
    if not normalized_timeframes:
        normalized_timeframes = list(DEFAULT_TIMEFRAMES)

    out: list[dict] = []
    topcap_symbol_set = [symbol for symbol in normalized_symbols if symbol.endswith("/USDT")]
    if len(topcap_symbol_set) < 4:
        topcap_symbol_set = list(DEFAULT_TOP10_PLUS_METALS)

    for lookback in (8, 16, 32):
        for rebalance in (4, 8, 16):
            for threshold in (0.015, 0.03):
                for stop in (0.05, 0.08):
                    for book_size in (4, 6):
                        _append_candidate(
                            out,
                            name="topcap_tsmom_factory",
                            strategy_class="TopCapTimeSeriesMomentumStrategy",
                            symbols=topcap_symbol_set,
                            timeframes=normalized_timeframes,
                            params={
                                "lookback_bars": lookback,
                                "rebalance_bars": rebalance,
                                "signal_threshold": threshold,
                                "stop_loss_pct": stop,
                                "max_longs": book_size,
                                "max_shorts": book_size,
                                "min_price": 0.05,
                                "btc_regime_ma": 48,
                                "btc_symbol": "BTC/USDT",
                            },
                            source="factory",
                        )

    for lookback in (24, 48, 96):
        for breakout_buffer in (0.0, 0.002):
            for atr_stop in (1.8, 2.8):
                for allow_short in (True, False):
                    _append_candidate(
                        out,
                        name="rolling_breakout_factory",
                        strategy_class="RollingBreakoutStrategy",
                        symbols=topcap_symbol_set,
                        timeframes=normalized_timeframes,
                        params={
                            "lookback_bars": lookback,
                            "breakout_buffer": breakout_buffer,
                            "atr_window": 14,
                            "atr_stop_multiplier": atr_stop,
                            "stop_loss_pct": 0.02,
                            "allow_short": allow_short,
                        },
                        source="factory",
                    )

    for window in (24, 48, 96):
        for entry_z in (1.2, 2.0, 2.8):
            _append_candidate(
                out,
                name="mean_reversion_std_factory",
                strategy_class="MeanReversionStdStrategy",
                symbols=topcap_symbol_set,
                timeframes=normalized_timeframes,
                params={
                    "window": window,
                    "entry_z": entry_z,
                    "exit_z": 0.4,
                    "stop_loss_pct": 0.02,
                    "allow_short": True,
                },
                source="factory",
            )

    for window in (24, 48, 96):
        for entry_dev in (0.008, 0.016, 0.024):
            _append_candidate(
                out,
                name="vwap_reversion_factory",
                strategy_class="VwapReversionStrategy",
                symbols=topcap_symbol_set,
                timeframes=normalized_timeframes,
                params={
                    "window": window,
                    "entry_dev": entry_dev,
                    "exit_dev": 0.004,
                    "stop_loss_pct": 0.02,
                    "allow_short": True,
                },
                source="factory",
            )

    for period in (7, 14, 21):
        for oversold, overbought in ((20, 70), (30, 80)):
            for allow_short in (True, False):
                _append_candidate(
                    out,
                    name="rsi_factory",
                    strategy_class="RsiStrategy",
                    symbols=topcap_symbol_set,
                    timeframes=normalized_timeframes,
                    params={
                        "rsi_period": period,
                        "oversold": oversold,
                        "overbought": overbought,
                        "allow_short": allow_short,
                    },
                    source="factory",
                )

    for short_window in (8, 14, 21):
        for long_window in (34, 55, 89):
            for allow_short in (True, False):
                _append_candidate(
                    out,
                    name="moving_average_factory",
                    strategy_class="MovingAverageCrossStrategy",
                    symbols=topcap_symbol_set,
                    timeframes=normalized_timeframes,
                    params={
                        "short_window": short_window,
                        "long_window": long_window,
                        "allow_short": allow_short,
                    },
                    source="factory",
                )

    pair_timeframes = [
        token
        for token in normalized_timeframes
        if TIMEFRAME_SECONDS.get(token, 0) <= TIMEFRAME_SECONDS["1h"]
    ]
    if not pair_timeframes:
        pair_timeframes = ["1m", "5m", "15m", "1h"]

    for symbol_x, symbol_y in DEFAULT_PAIR_SET:
        for lookback in (48, 96):
            for entry_z in (1.6, 2.0, 2.4):
                for stop_z in (3.0, 3.8):
                    _append_candidate(
                        out,
                        name=f"pair_{symbol_x.split('/')[0].lower()}_{symbol_y.split('/')[0].lower()}_factory",
                        strategy_class="PairTradingZScoreStrategy",
                        symbols=[symbol_x, symbol_y],
                        timeframes=pair_timeframes,
                        params={
                            "lookback_window": lookback,
                            "hedge_window": lookback * 2,
                            "entry_z": entry_z,
                            "exit_z": 0.25,
                            "stop_z": stop_z,
                            "min_correlation": 0.0,
                            "max_hold_bars": 96,
                            "cooldown_bars": 2,
                            "stop_loss_pct": 0.03,
                        },
                        source="factory",
                    )

    for symbol_x, symbol_y in DEFAULT_PAIR_SET:
        for lag_bars in (2, 4, 8):
            for entry in (0.008, 0.016):
                for stop in (0.05, 0.08):
                    _append_candidate(
                        out,
                        name=(
                            "lag_convergence_"
                            f"{symbol_x.split('/')[0].lower()}_{symbol_y.split('/')[0].lower()}_factory"
                        ),
                        strategy_class="LagConvergenceStrategy",
                        symbols=[symbol_x, symbol_y],
                        timeframes=pair_timeframes,
                        params={
                            "lag_bars": lag_bars,
                            "entry_threshold": entry,
                            "exit_threshold": entry / 4.0,
                            "stop_threshold": stop,
                            "max_hold_bars": 96,
                            "stop_loss_pct": 0.03,
                        },
                        source="factory",
                    )

    return out


def _hurdle_score_from_row(row: dict, mode: str) -> tuple[float, float, bool]:
    hurdle_key = "val" if str(mode).strip().lower() == "live" else "oos"
    fields = ((row.get("hurdle_fields") or {}).get(hurdle_key)) or {}
    score = _safe_float(fields.get("score"), -1_000_000.0)
    excess = _safe_float(fields.get("excess_return"), -1_000_000.0)
    passed = bool(fields.get("pass", False))
    return score, excess, passed


def _resolve_report_paths(pattern: str, limit: int) -> list[Path]:
    files = sorted(Path().glob(pattern), key=lambda item: item.stat().st_mtime)
    if int(limit) > 0:
        files = files[-int(limit) :]
    return files


def _normalize_strategy_row(
    row: dict,
    *,
    timeframe: str,
    source_report: Path,
    mode: str,
    source: str,
) -> dict | None:
    name = str(row.get("name", "")).strip()
    if not name:
        return None

    symbols = [_normalize_symbol(symbol) for symbol in list(row.get("symbols") or []) if symbol]
    params = dict(row.get("params") or {})
    score, excess, passed = _hurdle_score_from_row(row, mode)
    base_score = score if passed else (-1_000_000.0 + excess)
    payload = {
        "name": name,
        "family": _family_from_name(name),
        "strategy_timeframe": _normalize_timeframe(timeframe),
        "symbols": symbols,
        "params": params,
        "hurdle_score": score,
        "hurdle_excess_return": excess,
        "hurdle_pass": passed,
        "base_score": base_score,
        "source_report": str(source_report),
        "source": source,
    }
    identity = hashlib.sha1(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    payload["identity"] = identity[:16]
    return payload


def _load_candidates_from_team_report(path: Path, *, mode: str) -> list[dict]:
    with path.open(encoding="utf-8") as file:
        doc = json.load(file)

    rows: list[dict] = []
    seen: set[str] = set()

    for raw in list(doc.get("selected_team") or []):
        if not isinstance(raw, dict):
            continue
        timeframe = str(raw.get("strategy_timeframe") or raw.get("timeframe") or "")
        normalized = _normalize_strategy_row(
            raw,
            timeframe=timeframe,
            source_report=path,
            mode=mode,
            source="selected_team",
        )
        if normalized is None:
            continue
        ident = str(normalized.get("identity", ""))
        if ident in seen:
            continue
        seen.add(ident)
        rows.append(normalized)

    referenced_reports: set[Path] = set()
    for run_row in list(doc.get("run_rows") or []):
        if not isinstance(run_row, dict):
            continue
        if str(run_row.get("status", "")).strip().lower() != "ok":
            continue
        report_path = Path(str(run_row.get("report_path", "")).strip())
        if not report_path.exists():
            continue
        referenced_reports.add(report_path)

    for ref_path in sorted(referenced_reports):
        try:
            with ref_path.open(encoding="utf-8") as file:
                report_doc = json.load(file)
        except Exception:
            continue

        fallback_tf = str(report_doc.get("timeframe") or "")
        for raw in list(report_doc.get("candidates") or []):
            if not isinstance(raw, dict):
                continue
            normalized = _normalize_strategy_row(
                raw,
                timeframe=fallback_tf,
                source_report=ref_path,
                mode=mode,
                source="run_report",
            )
            if normalized is None:
                continue
            ident = str(normalized.get("identity", ""))
            if ident in seen:
                continue
            seen.add(ident)
            rows.append(normalized)

    return rows


def _symbol_csv_path(data_dir: Path, symbol: str) -> Path | None:
    clean = _normalize_symbol(symbol)
    candidates = [
        data_dir / f"{clean.replace('/', '')}.csv",
        data_dir / f"{clean.replace('/', '_')}.csv",
        data_dir / f"{clean}.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def _load_symbol_ohlcv(path: Path) -> dict[str, list[float]]:
    out = {"high": [], "low": [], "close": [], "volume": []}
    with path.open(encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            lowered = {str(key).strip().lower(): value for key, value in row.items()}
            high = _parse_float(lowered.get("high"))
            low = _parse_float(lowered.get("low"))
            close = _parse_float(lowered.get("close"))
            volume = _parse_float(lowered.get("volume"))
            if high is None or low is None or close is None or volume is None:
                continue
            out["high"].append(high)
            out["low"].append(low)
            out["close"].append(close)
            out["volume"].append(volume)
    return out


def _build_symbol_regime_snapshot(data_dir: Path, symbols: list[str], window: int) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for symbol in symbols:
        path = _symbol_csv_path(data_dir, symbol)
        if path is None:
            continue
        series = _load_symbol_ohlcv(path)
        closes = series["close"]
        highs = series["high"]
        lows = series["low"]
        volumes = series["volume"]
        if len(closes) < max(4, int(window) + 2):
            continue

        out[_normalize_symbol(symbol)] = {
            "symbol": _normalize_symbol(symbol),
            "source_csv": str(path),
            "rolling_log_vol": rolling_log_return_volatility_latest(
                closes,
                window=window,
                annualization=1.0,
            ),
            "normalized_true_range": normalized_true_range_latest(
                highs,
                lows,
                closes,
                window=window,
            ),
            "volume_shock_zscore": volume_shock_zscore_latest(volumes, window=window),
            "trend_efficiency": trend_efficiency_latest(closes, window=window),
        }
    return out


def _mean_metric(rows: list[dict], key: str) -> float | None:
    values: list[float] = []
    for row in rows:
        value = row.get(key)
        if value is None:
            continue
        values.append(float(value))
    if not values:
        return None
    return float(sum(values) / len(values))


def _regime_bias(row: dict, symbol_snapshot: dict[str, dict]) -> float:
    symbols = [_normalize_symbol(symbol) for symbol in list(row.get("symbols") or [])]
    selected = [symbol_snapshot[symbol] for symbol in symbols if symbol in symbol_snapshot]
    if not selected:
        return 0.0

    mean_eff = _mean_metric(selected, "trend_efficiency")
    mean_ntr = _mean_metric(selected, "normalized_true_range")
    mean_vshock = _mean_metric(selected, "volume_shock_zscore")

    family = str(row.get("family", "other"))
    bias = 0.0
    if family == "trend_overlay":
        if mean_eff is not None and mean_eff >= 0.33:
            bias += 0.30
        if mean_ntr is not None and mean_ntr >= 0.010:
            bias += 0.20
        if mean_vshock is not None and mean_vshock >= 1.0:
            bias += 0.10
    elif family == "alpha_market_neutral":
        if mean_eff is not None and mean_eff <= 0.25:
            bias += 0.25
        if mean_ntr is not None and mean_ntr <= 0.009:
            bias += 0.15
        if mean_vshock is not None and abs(mean_vshock) >= 1.5:
            bias += 0.05
    elif family == "momentum_mean_reversion":
        if mean_eff is not None and 0.20 <= mean_eff <= 0.55:
            bias += 0.20
        if mean_vshock is not None and mean_vshock >= 0.5:
            bias += 0.05
    return bias


def _apply_bias(rows: list[dict], symbol_snapshot: dict[str, dict]) -> list[dict]:
    out: list[dict] = []
    for row in rows:
        enriched = dict(row)
        base = _safe_float(enriched.get("base_score"), -1_000_000.0)
        bias = _regime_bias(enriched, symbol_snapshot)
        enriched["regime_bias"] = float(bias)
        enriched["adjusted_score"] = float(base + bias)
        out.append(enriched)
    return out


def _select_shortlist(
    rows: list[dict],
    *,
    max_total: int,
    max_per_family: int,
    max_per_timeframe: int,
    max_per_symbol: int,
    min_score: float,
) -> list[dict]:
    ranked = sorted(rows, key=lambda row: _safe_float(row.get("adjusted_score"), -1_000_000.0), reverse=True)
    selected: list[dict] = []
    family_counts: dict[str, int] = {}
    timeframe_counts: dict[str, int] = {}
    symbol_counts: dict[str, int] = {}
    seen: set[str] = set()

    for row in ranked:
        if len(selected) >= int(max_total):
            break
        score = _safe_float(row.get("adjusted_score"), -1_000_000.0)
        if score < float(min_score):
            continue
        ident = str(row.get("identity", "")).strip()
        if ident and ident in seen:
            continue

        family = str(row.get("family", "other"))
        timeframe = str(row.get("strategy_timeframe", ""))
        if family_counts.get(family, 0) >= int(max_per_family):
            continue
        if timeframe and timeframe_counts.get(timeframe, 0) >= int(max_per_timeframe):
            continue

        symbols = [_normalize_symbol(symbol) for symbol in list(row.get("symbols") or [])]
        if any(symbol_counts.get(symbol, 0) >= int(max_per_symbol) for symbol in symbols):
            continue

        selected.append(row)
        if ident:
            seen.add(ident)
        family_counts[family] = family_counts.get(family, 0) + 1
        if timeframe:
            timeframe_counts[timeframe] = timeframe_counts.get(timeframe, 0) + 1
        for symbol in symbols:
            symbol_counts[symbol] = symbol_counts.get(symbol, 0) + 1

    return selected


def _write_shortlist_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "name",
        "family",
        "strategy_timeframe",
        "symbols",
        "adjusted_score",
        "base_score",
        "regime_bias",
        "hurdle_pass",
        "hurdle_score",
        "hurdle_excess_return",
        "source_report",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "name": row.get("name"),
                    "family": row.get("family"),
                    "strategy_timeframe": row.get("strategy_timeframe"),
                    "symbols": ",".join(list(row.get("symbols") or [])),
                    "adjusted_score": _safe_float(row.get("adjusted_score")),
                    "base_score": _safe_float(row.get("base_score")),
                    "regime_bias": _safe_float(row.get("regime_bias")),
                    "hurdle_pass": bool(row.get("hurdle_pass", False)),
                    "hurdle_score": _safe_float(row.get("hurdle_score")),
                    "hurdle_excess_return": _safe_float(row.get("hurdle_excess_return")),
                    "source_report": row.get("source_report"),
                }
            )


def _write_markdown_report(
    path: Path,
    *,
    universe_count: int,
    sourced_count: int,
    shortlisted: list[dict],
    symbols: list[str],
) -> None:
    family_counts: dict[str, int] = {}
    timeframe_counts: dict[str, int] = {}
    for row in shortlisted:
        family = str(row.get("family", "other"))
        timeframe = str(row.get("strategy_timeframe", ""))
        family_counts[family] = family_counts.get(family, 0) + 1
        if timeframe:
            timeframe_counts[timeframe] = timeframe_counts.get(timeframe, 0) + 1

    lines = [
        "# Futures Strategy Factory Report",
        "",
        f"- Generated at: {datetime.now(UTC).isoformat()}",
        f"- Candidate universe size: {universe_count}",
        f"- Ranked source candidates: {sourced_count}",
        f"- Shortlist size: {len(shortlisted)}",
        "",
        "## Family mix",
        "",
    ]
    for family, count in sorted(family_counts.items()):
        lines.append(f"- {family}: {count}")
    lines.append("")
    lines.append("## Timeframe mix")
    lines.append("")
    for timeframe, count in sorted(
        timeframe_counts.items(),
        key=lambda item: TIMEFRAME_SECONDS.get(item[0], 10**12),
    ):
        lines.append(f"- {timeframe}: {count}")
    lines.append("")
    lines.append("## Top shortlist entries")
    lines.append("")
    for idx, row in enumerate(shortlisted[:12], start=1):
        lines.append(
            f"{idx}. `{row.get('name')}` tf={row.get('strategy_timeframe')} "
            f"score={_safe_float(row.get('adjusted_score')):.3f} "
            f"symbols={','.join(list(row.get('symbols') or []))}"
        )
    lines.append("")
    lines.append("## Recommended commands")
    lines.append("")
    lines.append(
        "```bash\n"
        "uv run python scripts/futures_strategy_factory.py "
        "--max-shortlist 64 --max-report-files 20\n"
        "```"
    )
    lines.append("")
    lines.append(
        "```bash\n"
        "uv run python scripts/run_strategy_team_research.py "
        "--market-type future --mode oos --strategy-set all "
        f"--topcap-symbols {' '.join([_normalize_symbol(symbol) for symbol in symbols])}\n"
        "```"
    )
    lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build and rank Binance futures strategy candidates.")
    parser.add_argument("--symbols", nargs="+", default=list(DEFAULT_TOP10_PLUS_METALS))
    parser.add_argument("--timeframes", nargs="+", default=list(DEFAULT_TIMEFRAMES))
    parser.add_argument("--mode", choices=["oos", "live"], default="oos")
    parser.add_argument("--report-glob", default="reports/strategy_team_research_oos_*.json")
    parser.add_argument("--max-report-files", type=int, default=20)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--regime-window", type=int, default=128)
    parser.add_argument("--max-shortlist", type=int, default=48)
    parser.add_argument("--max-per-family", type=int, default=20)
    parser.add_argument("--max-per-timeframe", type=int, default=8)
    parser.add_argument("--max-per-symbol", type=int, default=12)
    parser.add_argument("--min-score", type=float, default=-10.0)
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    symbols = [_normalize_symbol(symbol) for symbol in list(args.symbols)]
    timeframes = [_normalize_timeframe(token) for token in list(args.timeframes)]

    output_dir = Path(str(args.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")

    candidate_universe = _build_candidate_universe(symbols, timeframes)
    candidate_universe_path = output_dir / f"futures_candidate_universe_{stamp}.json"

    report_paths = _resolve_report_paths(str(args.report_glob), int(args.max_report_files))
    sourced_rows: list[dict] = []
    for path in report_paths:
        sourced_rows.extend(_load_candidates_from_team_report(path, mode=str(args.mode)))

    if not sourced_rows:
        for row in candidate_universe:
            sourced_rows.append(
                {
                    "identity": row.get("candidate_id"),
                    "name": row.get("name"),
                    "family": row.get("family"),
                    "strategy_timeframe": "1h",
                    "symbols": list(row.get("symbols") or []),
                    "params": dict(row.get("params") or {}),
                    "hurdle_score": 0.0,
                    "hurdle_excess_return": 0.0,
                    "hurdle_pass": True,
                    "base_score": 0.0,
                    "source_report": "factory_only",
                    "source": "factory_seed",
                }
            )

    symbol_snapshot = _build_symbol_regime_snapshot(
        Path(str(args.data_dir)),
        symbols,
        window=max(16, int(args.regime_window)),
    )
    biased_rows = _apply_bias(sourced_rows, symbol_snapshot)
    shortlisted = _select_shortlist(
        biased_rows,
        max_total=max(1, int(args.max_shortlist)),
        max_per_family=max(1, int(args.max_per_family)),
        max_per_timeframe=max(1, int(args.max_per_timeframe)),
        max_per_symbol=max(1, int(args.max_per_symbol)),
        min_score=float(args.min_score),
    )

    shortlist_json_path = output_dir / f"futures_shortlist_{stamp}.json"
    shortlist_csv_path = output_dir / f"futures_shortlist_{stamp}.csv"
    shortlist_report_path = output_dir / f"futures_strategy_factory_report_{stamp}.md"

    if not args.dry_run:
        with candidate_universe_path.open("w", encoding="utf-8") as file:
            json.dump(
                {
                    "generated_at": datetime.now(UTC).isoformat(),
                    "symbols": symbols,
                    "timeframes": timeframes,
                    "candidate_count": len(candidate_universe),
                    "candidates": candidate_universe,
                },
                file,
                indent=2,
            )
        with shortlist_json_path.open("w", encoding="utf-8") as file:
            json.dump(
                {
                    "generated_at": datetime.now(UTC).isoformat(),
                    "mode": str(args.mode),
                    "report_glob": str(args.report_glob),
                    "source_report_count": len(report_paths),
                    "source_candidate_count": len(sourced_rows),
                    "symbol_snapshot": symbol_snapshot,
                    "shortlist_count": len(shortlisted),
                    "shortlist": shortlisted,
                },
                file,
                indent=2,
            )
        _write_shortlist_csv(shortlist_csv_path, shortlisted)
        _write_markdown_report(
            shortlist_report_path,
            universe_count=len(candidate_universe),
            sourced_count=len(sourced_rows),
            shortlisted=shortlisted,
            symbols=symbols,
        )

    print("=== Futures Strategy Factory ===")
    print(f"candidate_universe={len(candidate_universe)}")
    print(f"source_reports={len(report_paths)} source_candidates={len(sourced_rows)}")
    print(f"shortlisted={len(shortlisted)}")
    if args.dry_run:
        print("Dry-run mode: output files not written.")
    else:
        print(f"Candidate universe: {candidate_universe_path}")
        print(f"Shortlist JSON   : {shortlist_json_path}")
        print(f"Shortlist CSV    : {shortlist_csv_path}")
        print(f"Factory report   : {shortlist_report_path}")


if __name__ == "__main__":
    main()
