"""Opt-in CLI: screen a research universe into a ranked target pool.

This script is the *composition seam* between the standalone multi-factor
selector (:mod:`lumina_quant.strategy_factory.universe_selection`) and the
existing candidate-research entrypoint. It is purely additive: importing it has
no side effects, and it never touches the default research pipeline unless the
operator explicitly runs it.

Workflow:
    1. Resolve the probe universe / parquet root / exchange from the existing
       runtime settings (falls back to module defaults when config is absent).
    2. Load OHLCV at a cheap probe timeframe (default ``1d``) via the verified
       loader ``market_data.load_data_dict_from_parquet``.
    3. Run ``select_target_pool_from_frames`` to produce a ranked pool plus
       per-symbol factor breakdown.
    4. Print / write the result as JSON.
    5. Optionally feed the pool into ``run_candidate_research`` via the existing
       ``symbol_universe`` kwarg (``--feed-research``).

All data access and I/O live behind ``main()``; the module imports clean.

Example:
    python scripts/research/select_target_pool.py --timeframe 1d --pool-size 8
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

DEFAULT_PROBE_TIMEFRAME = "1d"
DEFAULT_PROTECTED_SYMBOLS: tuple[str, ...] = (
    "BTC/USDT",
    "ETH/USDT",
    "XAU/USDT",
    "XAG/USDT",
    "XPT/USDT",
    "XPD/USDT",
)


def _build_arg_parser() -> argparse.ArgumentParser:
    """Return the configured argument parser for this CLI."""
    parser = argparse.ArgumentParser(
        description="Screen a research universe into a ranked multi-factor target pool.",
    )
    parser.add_argument(
        "--timeframe",
        default=DEFAULT_PROBE_TIMEFRAME,
        help="Probe timeframe used to score symbols (default: 1d).",
    )
    parser.add_argument(
        "--symbols",
        nargs="*",
        default=None,
        help="Explicit symbol universe; defaults to the configured research universe.",
    )
    parser.add_argument(
        "--pool-size",
        type=int,
        default=None,
        help="Override the target pool size (default: config default).",
    )
    parser.add_argument(
        "--protected",
        nargs="*",
        default=list(DEFAULT_PROTECTED_SYMBOLS),
        help="Symbols always kept in the pool (pair anchors).",
    )
    parser.add_argument(
        "--parquet-root",
        default=None,
        help="Override the parquet data root (default: runtime setting).",
    )
    parser.add_argument(
        "--exchange",
        default=None,
        help="Override the exchange (default: runtime setting).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write the ranked pool + breakdown JSON.",
    )
    parser.add_argument(
        "--feed-research",
        action="store_true",
        help="Feed the resulting pool into run_candidate_research(symbol_universe=...).",
    )
    return parser


def _resolve_settings(
    *,
    symbols: Sequence[str] | None,
    parquet_root: str | None,
    exchange: str | None,
) -> dict[str, Any]:
    """Resolve universe/root/exchange from runtime settings with CLI overrides."""
    from lumina_quant.strategy_factory.runtime_settings import (
        current_research_market_data_settings,
    )

    settings = current_research_market_data_settings()
    resolved_symbols = list(symbols) if symbols else list(settings["symbols"])
    return {
        "symbols": resolved_symbols,
        "parquet_root": parquet_root or str(settings["parquet_root"]),
        "exchange": exchange or str(settings["exchange"]),
    }


def _load_frames(
    *,
    parquet_root: str,
    exchange: str,
    symbols: Sequence[str],
    timeframe: str,
) -> dict[str, pl.DataFrame]:
    """Load OHLCV frames for the universe; return ``{}`` when data is absent."""
    from lumina_quant.market_data import load_data_dict_from_parquet

    try:
        return load_data_dict_from_parquet(
            parquet_root,
            exchange=exchange,
            symbol_list=list(symbols),
            timeframe=timeframe,
        )
    except Exception as exc:
        print(f"[select_target_pool] data load failed ({exc!r}); using static fallback")
        return {}


def main(argv: Sequence[str] | None = None) -> int:
    """Run the target-pool selection CLI.

    Args:
        argv: Optional argument vector (defaults to ``sys.argv``).

    Returns:
        Process exit code (``0`` on success).
    """
    from lumina_quant.strategy_factory.universe_selection import (
        UniverseSelectionConfig,
        result_to_dict,
        select_target_pool_from_frames,
    )

    args = _build_arg_parser().parse_args(argv)
    settings = _resolve_settings(
        symbols=args.symbols,
        parquet_root=args.parquet_root,
        exchange=args.exchange,
    )

    config = None
    if args.pool_size is not None:
        config = UniverseSelectionConfig(pool_size=int(args.pool_size))

    loaded = _load_frames(
        parquet_root=settings["parquet_root"],
        exchange=settings["exchange"],
        symbols=settings["symbols"],
        timeframe=str(args.timeframe),
    )

    result = select_target_pool_from_frames(
        loaded,
        config=config,
        fallback_symbols=settings["symbols"],
        protected=args.protected,
    )

    payload = result_to_dict(result)
    payload["meta"] = {
        "timeframe": str(args.timeframe),
        "exchange": settings["exchange"],
        "parquet_root": settings["parquet_root"],
        "universe_size": len(settings["symbols"]),
        "loaded_frames": len(loaded),
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(f"[select_target_pool] wrote {args.output}")

    if args.feed_research:
        from lumina_quant.strategy_factory.research_entrypoints import (
            build_default_candidate_rows,
            run_candidate_research,
        )

        candidates = build_default_candidate_rows(symbols=list(result.pool))
        research = run_candidate_research(
            candidates=candidates,
            symbol_universe=list(result.pool),
        )
        print(
            "[select_target_pool] fed research with pool of "
            f"{len(result.pool)} symbols; report keys: {sorted(research)}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
