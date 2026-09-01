"""Offline CLI: build the post-backtest alpha scoreboard (json + markdown).

Runs on the data-bearing PC over exported result rows — no backtests, no
network. Input JSON is a list of candidate/portfolio result rows:

    [
      {
        "id": "kalman_trend_rider_1h_core_ls_btcusdt",
        "family": "trend",
        "strategy_class": "KalmanTrendRiderStrategy",
        "metrics": {"return": 0.42, "sharpe": 1.3, "mdd": 0.11, ...},
        "returns": [0.001, -0.002, ...],
        "turnover": 0.03,
        "trade_count": 57,
        "liquidation_count": 0,
        "bars": 8760
      },
      ...
    ]

``metrics`` (canonical flat keys) takes precedence; missing metrics are
derived from ``returns`` when supplied. Output is deterministic for a given
input (sorted keys, stable tie-breaks, no wall clock).

Usage:
    .venv/bin/python scripts/research/build_alpha_scoreboard.py \
        --input rows.json --output-json scoreboard.json \
        --output-md scoreboard.md [--top-n 10] \
        [--gates '{"max_mdd": 0.3, "min_trades": 5}'] \
        [--weights '{"sharpe": 2.0, "mdd": 1.0, "return": 1.0}']
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lumina_quant.research.alpha_scoreboard import (
    build_scoreboard,
    render_scoreboard_markdown,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="JSON file with result rows")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", default="")
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--gates", default="", help="JSON dict overriding default gates")
    parser.add_argument("--weights", default="", help="JSON dict of composite metric weights")
    parser.add_argument("--title", default="Alpha Scoreboard")
    args = parser.parse_args()

    raw_rows = json.loads(Path(args.input).read_text(encoding="utf-8"))
    if not isinstance(raw_rows, list):
        raise SystemExit("--input must contain a JSON list of result rows")
    gates = json.loads(args.gates) if args.gates else None
    weights = json.loads(args.weights) if args.weights else None

    payload = build_scoreboard(raw_rows, gates=gates, weights=weights, top_n=args.top_n)

    Path(args.output_json).write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    if args.output_md:
        Path(args.output_md).write_text(
            render_scoreboard_markdown(payload, title=args.title), encoding="utf-8"
        )
    print(
        f"scoreboard: {payload['eligible_count']}/{payload['input_count']} eligible; "
        f"top composite: " + ", ".join(item["id"] for item in payload["composite"][:3])
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
