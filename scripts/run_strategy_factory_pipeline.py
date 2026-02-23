"""One-command strategy-factory pipeline for large candidate research.

Workflow:
1) Build candidate manifest (top10 coins + XAU/XAG, 1s~1d configurable).
2) Run strategy-team research orchestrator.
3) Build diversified shortlist payload and markdown summary.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from lumina_quant.strategy_factory import (
    DEFAULT_BINANCE_TOP10_PLUS_METALS,
    DEFAULT_TIMEFRAMES,
    build_research_command,
    build_shortlist_payload,
    extract_saved_report_path,
    render_shortlist_markdown,
    write_candidate_manifest,
)


def _normalize_symbols(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in values:
        token = str(raw).strip().upper().replace("_", "/").replace("-", "/")
        if not token:
            continue
        if "/" not in token and token.endswith("USDT") and len(token) > 4:
            token = f"{token[:-4]}/USDT"
        if token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out


def _normalize_timeframes(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in values:
        token = str(raw).strip().lower()
        if not token:
            continue
        if token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out


def _enforce_1s_base_timeframe(value: str) -> str:
    token = str(value or "").strip().lower() or "1s"
    if token != "1s":
        print(f"[WARN] base-timeframe '{token}' overridden to '1s' for all backtests.")
    return "1s"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run large-scale strategy factory pipeline.")
    parser.add_argument("--db-path", default="data/lq_market.sqlite3")
    parser.add_argument("--backend", default="influxdb", help="Storage backend override (sqlite|influxdb).")
    parser.add_argument("--influx-url", default="")
    parser.add_argument("--influx-org", default="")
    parser.add_argument("--influx-bucket", default="")
    parser.add_argument("--influx-token", default="")
    parser.add_argument("--influx-token-env", default="INFLUXDB_TOKEN")
    parser.add_argument("--output-dir", default="reports/strategy_factory")
    parser.add_argument("--exchange", default="binance")
    parser.add_argument("--market-type", choices=["spot", "future"], default="future")
    parser.add_argument("--mode", choices=["oos", "live"], default="oos")
    parser.add_argument(
        "--strategy-set",
        choices=["all", "crypto-only", "xau-xag-only"],
        default="all",
    )
    parser.add_argument("--base-timeframe", default="1s")
    parser.add_argument("--base-timeframes", nargs="+", default=["1s"])
    parser.add_argument("--timeframes", nargs="+", default=list(DEFAULT_TIMEFRAMES))
    parser.add_argument("--seeds", nargs="+", type=int, default=[20260220, 20260221, 20260222])
    parser.add_argument("--symbols", nargs="+", default=list(DEFAULT_BINANCE_TOP10_PLUS_METALS))
    parser.add_argument("--max-selected", type=int, default=64)
    parser.add_argument("--max-per-family", type=int, default=24)
    parser.add_argument("--max-per-timeframe", type=int, default=12)
    parser.add_argument("--max-runs", type=int, default=0)
    parser.add_argument("--shortlist-max-total", type=int, default=24)
    parser.add_argument("--shortlist-max-per-family", type=int, default=8)
    parser.add_argument("--shortlist-max-per-timeframe", type=int, default=6)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _latest_team_report() -> Path | None:
    candidates = sorted(Path("reports").glob("strategy_team_research_*.json"), reverse=True)
    return candidates[0] if candidates else None


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    args.base_timeframe = _enforce_1s_base_timeframe(args.base_timeframe)
    args.base_timeframes = ["1s"]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    symbols = _normalize_symbols(list(args.symbols))
    if len(symbols) < 2:
        raise SystemExit("Need at least two symbols after normalization.")
    timeframes = _normalize_timeframes(list(args.timeframes))
    if not timeframes:
        raise SystemExit("Need at least one timeframe.")

    manifest_path, manifest = write_candidate_manifest(
        output_dir=output_dir,
        timeframes=timeframes,
        symbols=symbols,
    )
    print(f"[PIPELINE] candidate manifest: {manifest_path}")
    print(f"[PIPELINE] candidate_count={manifest.get('candidate_count', 0)}")

    command = build_research_command(
        db_path=str(args.db_path),
        backend=str(args.backend),
        influx_url=str(args.influx_url),
        influx_org=str(args.influx_org),
        influx_bucket=str(args.influx_bucket),
        influx_token=str(args.influx_token),
        influx_token_env=str(args.influx_token_env),
        exchange=str(args.exchange),
        market_type=str(args.market_type),
        mode=str(args.mode),
        strategy_set=str(args.strategy_set),
        base_timeframe=str(args.base_timeframe),
        base_timeframes=list(args.base_timeframes),
        timeframes=timeframes,
        seeds=[int(seed) for seed in args.seeds],
        topcap_symbols=symbols,
        max_selected=int(args.max_selected),
        max_per_family=int(args.max_per_family),
        max_per_timeframe=int(args.max_per_timeframe),
        max_runs=int(args.max_runs),
        candidate_manifest=str(manifest_path),
    )
    if bool(args.dry_run):
        command.append("--dry-run")

    print("[PIPELINE] research command:")
    print(" ".join(command))

    process = subprocess.run(
        command,
        cwd=str(Path(__file__).resolve().parent.parent),
        check=False,
        capture_output=True,
        text=True,
    )
    if process.stdout:
        print(process.stdout.rstrip())
    if process.stderr:
        print(process.stderr.rstrip(), file=sys.stderr)
    if int(process.returncode) != 0:
        raise SystemExit(int(process.returncode))

    report_path = extract_saved_report_path(process.stdout or "")
    if report_path is None or not report_path.exists():
        report_path = _latest_team_report()
    if report_path is None or not report_path.exists():
        raise SystemExit("Failed to locate strategy_team_research output report.")

    with report_path.open(encoding="utf-8") as file:
        report = json.load(file)

    shortlist_payload = build_shortlist_payload(
        report=report,
        mode=str(args.mode),
        shortlist_max_total=int(args.shortlist_max_total),
        shortlist_max_per_family=int(args.shortlist_max_per_family),
        shortlist_max_per_timeframe=int(args.shortlist_max_per_timeframe),
        manifest_path=manifest_path,
        research_report_path=report_path,
    )

    shortlist_json = output_dir / "strategy_factory_shortlist.json"
    shortlist_md = output_dir / "strategy_factory_shortlist.md"
    shortlist_json.write_text(json.dumps(shortlist_payload, indent=2), encoding="utf-8")
    shortlist_md.write_text(render_shortlist_markdown(shortlist_payload), encoding="utf-8")

    print(f"[PIPELINE] research report: {report_path}")
    print(f"[PIPELINE] shortlist json: {shortlist_json}")
    print(f"[PIPELINE] shortlist md: {shortlist_md}")


if __name__ == "__main__":
    main()
