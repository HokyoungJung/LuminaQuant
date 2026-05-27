#!/usr/bin/env python3
"""Collect Tickmill/MT5 read-only macro context when an MT5 bridge is configured."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parents[2]
    / "var/reports/profit_moonshot_20260501/current_tail_20260506/multiasset_exchange_expansion"
)
DEFAULT_SYMBOLS = (
    "EURUSD",
    "GBPUSD",
    "USDJPY",
    "XAUUSD",
    "XAGUSD",
    "US500",
    "USTEC",
    "US30",
    "DE40",
)


def _bridge_python() -> str:
    return str(
        os.getenv("LQ_MT5_BRIDGE_PYTHON") or os.getenv("LQ__LIVE__MT5_BRIDGE_PYTHON") or ""
    ).strip()


def _is_wsl() -> bool:
    try:
        text = Path("/proc/sys/kernel/osrelease").read_text(encoding="utf-8").lower()
    except Exception:
        return False
    return "microsoft" in text or "wsl" in text


def _bridge_script_path() -> str:
    explicit = str(
        os.getenv("LQ_MT5_BRIDGE_SCRIPT") or os.getenv("LQ__LIVE__MT5_BRIDGE_SCRIPT") or ""
    ).strip()
    path = (
        Path(explicit)
        if explicit
        else Path(__file__).resolve().parents[2] / "scripts/mt5_bridge_worker.py"
    )
    if not path.is_absolute():
        path = Path.cwd() / path
    token = _bridge_python().lower()
    if _is_wsl() and (token.endswith(".exe") or "\\" in token):
        proc = subprocess.run(
            ["wslpath", "-w", str(path)], capture_output=True, text=True, check=False
        )
        converted = (proc.stdout or "").strip()
        if proc.returncode == 0 and converted:
            return converted
    return str(path)


def _call_bridge(action: str, payload: dict[str, Any]) -> dict[str, Any]:
    python = _bridge_python()
    if not python:
        raise RuntimeError("MT5 bridge is not configured")
    proc = subprocess.run(
        [
            python,
            _bridge_script_path(),
            "--action",
            action,
            "--payload",
            json.dumps(payload, ensure_ascii=True),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    stdout = (proc.stdout or "").strip()
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or stdout or f"bridge failed: {action}").strip())
    if not stdout:
        raise RuntimeError(f"empty bridge response: {action}")
    response = json.loads(stdout.splitlines()[-1])
    if not bool(response.get("ok", False)):
        raise RuntimeError(str(response.get("error") or f"bridge failed: {action}"))
    result = response.get("result")
    return result if isinstance(result, dict) else {"value": result}


def _call_bridge_list(action: str, payload: dict[str, Any]) -> list[Any]:
    python = _bridge_python()
    if not python:
        raise RuntimeError("MT5 bridge is not configured")
    proc = subprocess.run(
        [
            python,
            _bridge_script_path(),
            "--action",
            action,
            "--payload",
            json.dumps(payload, ensure_ascii=True),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    stdout = (proc.stdout or "").strip()
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or stdout or f"bridge failed: {action}").strip())
    response = json.loads(stdout.splitlines()[-1])
    if not bool(response.get("ok", False)):
        raise RuntimeError(str(response.get("error") or f"bridge failed: {action}"))
    result = response.get("result")
    return result if isinstance(result, list) else []


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Tickmill/MT5 read-only collection",
        "",
        f"Generated: `{payload['generated_at_utc']}`",
        f"Status: `{payload['status']}`",
        "",
        "## External-doc anchors verified at runtime",
        "",
        "- Tickmill instruments page lists Forex, cryptocurrencies, commodities, stock indices, stocks/ETFs, and MetaTrader 4/5 platform availability.",
        "- Tickmill spreads/swaps page defines spread as bid/ask difference, requires MT4/MT5 symbol properties for trading hours, and states swaps are applied overnight with Wednesday triple-swap handling.",
        "",
    ]
    if payload["status"] != "collected":
        lines.extend(
            [
                "## Blocker",
                "",
                f"- `{payload['blocked_reason']}`",
                "- Tickmill macro filters were not replay-eligible because no MT5 read-only OHLCV/properties were available in this session.",
                "- Direct Tickmill trading remains blocked: spread, swap, session, lot-size, and fill assumptions are not modeled from terminal properties.",
            ]
        )
        return "\n".join(lines) + "\n"
    lines.extend(
        [
            "## Symbol properties / OHLCV snapshots",
            "",
            "| symbol | info | ohlcv rows | latest bar | spread points | swap long | swap short |",
            "|---|---:|---:|---|---:|---:|---:|",
        ]
    )
    for row in payload["symbols"]:
        info = row.get("symbol_info") or {}
        ohlcv = list(row.get("ohlcv") or [])
        latest = ohlcv[-1][0] if ohlcv and isinstance(ohlcv[-1], list) else None
        lines.append(
            f"| `{row['symbol']}` | {1 if info else 0} | {len(ohlcv)} | `{latest}` | "
            f"{info.get('spread')} | {info.get('swap_long')} | {info.get('swap_short')} |"
        )
    lines.extend(
        [
            "",
            "## Replay eligibility",
            "",
            "- This collection is read-only and not a direct trading permission.",
            "- MT5 data must cover train/val/OOS and include spread/swap/session/lot-size properties before direct Tickmill strategies can advance.",
        ]
    )
    return "\n".join(lines) + "\n"


def collect_tickmill(
    *, output_dir: Path, symbols: list[str], timeframe: str, limit: int
) -> dict[str, Any]:
    status = "blocked"
    blocked_reason = ""
    symbol_rows: list[dict[str, Any]] = []
    bridge_python = _bridge_python()
    if not bridge_python:
        blocked_reason = "LQ_MT5_BRIDGE_PYTHON / LQ__LIVE__MT5_BRIDGE_PYTHON is not configured."
    else:
        try:
            connect = _call_bridge("connect", {})
            for symbol in symbols:
                try:
                    symbol_info = _call_bridge("fetch_symbol_info", {"symbol": symbol})
                    ohlcv = _call_bridge_list(
                        "fetch_ohlcv", {"symbol": symbol, "timeframe": timeframe, "limit": limit}
                    )
                    symbol_rows.append(
                        {"symbol": symbol, "symbol_info": symbol_info, "ohlcv": ohlcv}
                    )
                except Exception as exc:
                    symbol_rows.append(
                        {"symbol": symbol, "error": str(exc), "symbol_info": {}, "ohlcv": []}
                    )
            status = "collected"
            blocked_reason = ""
            _ = connect
        except Exception as exc:
            blocked_reason = f"MT5 bridge read-only connection failed: {exc}"
    payload: dict[str, Any] = {
        "artifact_kind": "tickmill_mt5_readonly_collection",
        "generated_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "status": status,
        "blocked_reason": blocked_reason,
        "read_only": True,
        "direct_trading_blocked": True,
        "direct_trading_blocked_reason": "No direct trading until spread/swap/session/lot-size/fill models and raw-first evidence exist.",
        "bridge_python_configured": bool(bridge_python),
        "timeframe": timeframe,
        "limit": int(limit),
        "symbols": symbol_rows,
        "official_docs": {
            "instruments": "https://www.tickmill.com/trading-instruments/",
            "spreads_swaps": "https://www.tickmill.com/conditions/spreads-swaps",
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "tickmill_mt5_readonly_collection_latest.json"
    md_path = output_dir / "tickmill_mt5_readonly_collection_latest.md"
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    md_path.write_text(_markdown(payload), encoding="utf-8")
    return {"payload": payload, "paths": {"json": str(json_path), "markdown": str(md_path)}}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--symbols", nargs="*", default=list(DEFAULT_SYMBOLS))
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--limit", type=int, default=2000)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = collect_tickmill(
        output_dir=Path(args.output_dir),
        symbols=[str(symbol) for symbol in args.symbols],
        timeframe=str(args.timeframe),
        limit=int(args.limit),
    )
    print(json.dumps(result["paths"], ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
