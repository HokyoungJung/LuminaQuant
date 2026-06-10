"""lq data — config-driven market-data collection commands.

All operations are driven by the active RuntimeConfig profile (LQ_CONFIG_PATH).
No code change is needed to switch exchange, symbol universe, or data kinds —
edit the YAML profile and re-run.

Sub-commands
------------
lq data collect     Ensure OHLCV + funding coverage via DataCollector.ensure_ohlcv_coverage()
                    and DataCollector.collect_feature_points().
lq data fast        Fast research refresh — quick incremental OHLCV + feature-point pull.
                    (Replaces the retired lq refresh-data-fast command.)
lq data tick        Collect raw aggTrades tick data (data.kinds must include aggtrades_tick).

Safety: NEVER real orders.  DataCollector is read-only market data.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime


def _resolve_config_path() -> str:
    return os.environ.get("LQ_CONFIG_PATH", "config.yaml")


def _load_rt():
    from lumina_quant.configuration import load_runtime_config
    path = _resolve_config_path()
    try:
        return load_runtime_config(path)
    except FileNotFoundError:
        from lumina_quant.configuration.schema import RuntimeConfig
        return RuntimeConfig()


def _now_ms() -> int:
    return int(datetime.now(UTC).timestamp() * 1000)


def cmd_collect(args: argparse.Namespace) -> int:
    """Ensure OHLCV + feature-point coverage for the configured symbol universe."""
    from lumina_quant.data.collector import DataCollector

    rt = _load_rt()
    print(f"[lq data collect] exchange={rt.storage.market_data_exchange} "
          f"symbols={list(rt.trading.symbols)} kinds={list(rt.data.kinds)}")

    with DataCollector.from_runtime_config(rt) as collector:
        results: dict = {}

        if "ohlcv" in collector.config.kinds:
            print("[lq data collect] collecting OHLCV coverage...")
            stats = collector.ensure_ohlcv_coverage(
                timeframe=str(rt.trading.timeframe),
                since_ms=getattr(args, "since_ms", None),
                until_ms=getattr(args, "until_ms", None),
                force_full=bool(getattr(args, "force_full", False)),
            )
            results["ohlcv"] = stats
            print(f"  ohlcv: {len(stats)} symbol(s) processed")

        if any(k in collector.config.kinds for k in ("funding", "feature_points")):
            since_ms = getattr(args, "since_ms", None) or (_now_ms() - 30 * 24 * 3600 * 1000)
            print("[lq data collect] collecting feature points...")
            fp_result = collector.collect_feature_points(
                since_ms=since_ms,
                feature_profile=str(getattr(args, "feature_profile", "full") or "full"),
            )
            results["feature_points"] = fp_result
            print(f"  feature_points: {fp_result}")

    if bool(getattr(args, "json", False)):
        print(json.dumps(results, indent=2, sort_keys=True, default=str))
    return 0


def cmd_fast(args: argparse.Namespace) -> int:
    """Fast incremental research refresh — replaces lq refresh-data-fast."""
    from lumina_quant.data.collector import DataCollector

    rt = _load_rt()
    print(f"[lq data fast] quick refresh: exchange={rt.storage.market_data_exchange} "
          f"symbols={list(rt.trading.symbols)}")

    with DataCollector.from_runtime_config(rt) as collector:
        # Quick OHLCV top-up (last 7 days)
        lookback_ms = 7 * 24 * 3600 * 1000
        since_ms = _now_ms() - lookback_ms

        if "ohlcv" in collector.config.kinds:
            stats = collector.ensure_ohlcv_coverage(
                timeframe=str(rt.trading.timeframe),
                since_ms=since_ms,
                force_full=False,
            )
            total_rows = sum(s.get("fetched_rows", 0) for s in stats)
            print(f"  ohlcv top-up: {len(stats)} symbol(s), {total_rows} rows fetched")

        if any(k in collector.config.kinds for k in ("funding", "feature_points")):
            fp = collector.collect_feature_points(since_ms=since_ms, feature_profile="full")
            print(f"  feature_points: {fp}")

    print("[lq data fast] done")
    return 0


def cmd_tick(args: argparse.Namespace) -> int:
    """Collect raw aggTrades tick data for a symbol."""
    from lumina_quant.data.collector import DATA_KIND_AGGTRADES_TICK, DataCollector

    rt = _load_rt()
    if DATA_KIND_AGGTRADES_TICK not in list(rt.data.kinds):
        print(
            f"[error] aggtrades_tick not in data.kinds={list(rt.data.kinds)}. "
            "Add 'aggtrades_tick' to data.kinds in your profile YAML.",
            file=sys.stderr,
        )
        return 1

    symbol = str(getattr(args, "symbol", "") or "").strip()
    if not symbol:
        symbols = list(rt.trading.symbols)
        if not symbols:
            print("[error] --symbol required (no symbols in config)", file=sys.stderr)
            return 1
        symbol = symbols[0]
        print(f"[lq data tick] no --symbol; using first configured symbol: {symbol}")

    print(f"[lq data tick] collecting aggTrades for {symbol} ...")
    with DataCollector.from_runtime_config(rt) as collector:
        result = collector.collect_aggtrades_raw(symbol=symbol)

    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="lq data",
        description="Config-driven market-data collection (profile YAML drives all parameters).",
    )
    sub = parser.add_subparsers(dest="subcommand")

    _collect = sub.add_parser("collect", help="Ensure OHLCV + feature-point coverage.")
    _collect.add_argument("--since-ms", type=int, default=None, dest="since_ms",
                          help="Start timestamp in milliseconds (UTC).")
    _collect.add_argument("--until-ms", type=int, default=None, dest="until_ms",
                          help="End timestamp in milliseconds (UTC).")
    _collect.add_argument("--force-full", action="store_true", dest="force_full",
                          help="Force full re-fetch even if data exists.")
    _collect.add_argument("--feature-profile", default="full", dest="feature_profile",
                          help="Feature profile name (default: full).")
    _collect.add_argument("--json", action="store_true",
                          help="Output stats as JSON.")

    sub.add_parser("fast", help="Fast incremental refresh (replaces lq refresh-data-fast).")

    _tick = sub.add_parser("tick", help="Collect raw aggTrades tick data.")
    _tick.add_argument("--symbol", default="", help="Symbol to collect (e.g. BTC/USDT).")

    args = parser.parse_args(argv)

    if not args.subcommand:
        parser.print_help()
        return 0
    if args.subcommand == "collect":
        return cmd_collect(args)
    if args.subcommand == "fast":
        return cmd_fast(args)
    if args.subcommand == "tick":
        return cmd_tick(args)
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
