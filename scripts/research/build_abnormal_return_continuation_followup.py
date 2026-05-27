"""Build a focused manifest for abnormal-return continuation follow-up research."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from lumina_quant.strategy_factory import build_binance_futures_candidates

DEFAULT_OUTPUT_DIR = Path(
    "var/reports/exact_window_backtests/followup_status/"
    "portfolio_incumbent_autoresearch_grouped/abnormal_return_continuation_followup_current"
)
TARGET_STRATEGY = "AbnormalReturnContinuationStrategy"
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
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_symbols = list(args.symbols or ["BNB/USDT", "TRX/USDT"])

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
        "artifact_kind": "abnormal_return_continuation_followup_manifest",
        "generated_at": _utc_now_iso(),
        "candidate_count": len(candidates),
        "strategy_class": TARGET_STRATEGY,
        "strategy_timeframe": str(args.timeframe),
        "symbol_universe": list(selected_symbols),
        "candidates": candidates,
    }

    manifest_path = output_dir / "abnormal_return_continuation_candidate_manifest_latest.json"
    md_path = output_dir / "abnormal_return_continuation_manifest_latest.md"
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(
        "\n".join(
            [
                "# abnormal return continuation follow-up manifest",
                "",
                f"- generated_at: `{payload['generated_at']}`",
                f"- candidate_count: `{payload['candidate_count']}`",
                f"- timeframe: `{payload['strategy_timeframe']}`",
                "",
                "## candidates",
                *[
                    f"- `{item['name']}` | entry_z={float(item['params'].get('entry_z', 0.0)):.2f} | hold_bars={int(item['params'].get('hold_bars', 0))} | allow_short={bool(item['params'].get('allow_short', False))}"
                    for item in candidates
                ],
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(manifest_path)
    print(md_path)


if __name__ == "__main__":
    main()
