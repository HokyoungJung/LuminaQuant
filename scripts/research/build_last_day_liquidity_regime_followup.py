"""Build a focused manifest for last-day liquidity-regime follow-up research."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from lumina_quant.strategy_factory import build_binance_futures_candidates
from lumina_quant.strategy_factory.runtime_settings import current_research_market_data_settings

DEFAULT_OUTPUT_DIR = Path(
    "var/reports/exact_window_backtests/followup_status/"
    "portfolio_incumbent_autoresearch_grouped/last_day_liquidity_regime_followup_current"
)
TARGET_STRATEGY = "LastDayLiquidityRegimeStrategy"
TARGET_TIMEFRAME = "1d"


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--timeframe", default=TARGET_TIMEFRAME)
    parser.add_argument("--symbols", nargs="+", default=None)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    settings = current_research_market_data_settings()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_symbols = list(args.symbols or [])
    if not selected_symbols:
        selected_symbols = [
            symbol
            for symbol in list(settings["symbols"])
            if str(symbol).endswith("/USDT")
            and str(symbol).split("/", 1)[0] not in {"XAU", "XAG", "XPT", "XPD"}
        ][:4]
    if len(selected_symbols) < 2:
        raise SystemExit("last-day liquidity regime follow-up requires at least 2 symbols")

    rows = build_binance_futures_candidates(
        symbols=selected_symbols,
        timeframes=[str(args.timeframe)],
    )
    candidates = [
        row.to_dict()
        for row in rows
        if row.strategy_class == TARGET_STRATEGY and row.timeframe == str(args.timeframe)
    ]
    candidates.sort(key=lambda item: str(item.get("name") or ""))

    payload = {
        "artifact_kind": "last_day_liquidity_regime_followup_manifest",
        "generated_at": _utc_now_iso(),
        "candidate_count": len(candidates),
        "strategy_class": TARGET_STRATEGY,
        "strategy_timeframe": str(args.timeframe),
        "symbol_universe": list(selected_symbols),
        "candidates": candidates,
    }

    manifest_path = output_dir / "last_day_liquidity_regime_candidate_manifest_latest.json"
    batch_path = output_dir / "last_day_liquidity_regime_batches_latest.json"
    md_path = output_dir / "last_day_liquidity_regime_manifest_latest.md"
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    batch_path.write_text(
        json.dumps(
            {
                "artifact_kind": "candidate_followup_batches",
                "generated_at": payload["generated_at"],
                "source_manifest": str(manifest_path),
                "batch_count": 1,
                "batches": [
                    {
                        "batch_id": "last_day_liquidity_regime_batch_01",
                        "candidate_ids": [
                            str(item.get("candidate_id") or item.get("name") or "")
                            for item in candidates
                        ],
                        "candidate_names": [str(item.get("name") or "") for item in candidates],
                    }
                ],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    md_path.write_text(
        "\n".join(
            [
                "# last day liquidity regime follow-up manifest",
                "",
                f"- generated_at: `{payload['generated_at']}`",
                f"- candidate_count: `{payload['candidate_count']}`",
                f"- timeframe: `{payload['strategy_timeframe']}`",
                "",
                "## candidates",
                *[
                    f"- `{item['name']}` | longs={item['params'].get('max_longs')} | shorts={item['params'].get('max_shorts')} | signal_threshold={float(item['params'].get('signal_threshold', 0.0)):.3f}"
                    for item in candidates
                ],
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(manifest_path)
    print(batch_path)
    print(md_path)


if __name__ == "__main__":
    main()
