"""lq alpha — factor / alpha research CLI (read-only over research modules).

Sub-commands
------------
lq alpha rank      Rank a panel of factors by rank-IC / IC-IR / turnover / decay
                   using :mod:`lumina_quant.research.factor_ic`. Prints a ranking
                   ordered by descending IC-IR.
lq alpha card      Emit a per-factor "factor card" (markdown + JSON) summarising the
                   deterministic IC statistics for a single factor.
lq alpha promote   Run the deterministic alpha-search survivorship workflow
                   (:func:`lumina_quant.research.alpha_search.run_alpha_search`) and
                   promote survivors into a
                   :class:`lumina_quant.research.alpha_candidate_ledger.ResearchLedger`,
                   then report the promoted set from the ledger.

Every sub-command is read-only with respect to the research package: it only calls
public research entry points and never mutates config/schema or research source.
The default input is a deterministic synthetic panel
(:func:`lumina_quant.research.alpha_search.build_synthetic_search_panel`) so the
commands are byte-reproducible offline; a real long-format panel file may be
supplied to ``rank`` / ``card`` via ``--panel``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Module-level defaults (parameters live here, never in configuration/schema.py).
DEFAULT_N_SYMBOLS = 8
DEFAULT_N_PERIODS = 160
DEFAULT_PANEL_SEED = 7
DEFAULT_SIGNAL_STRENGTH = 0.6
DEFAULT_MAX_DECAY_LAG = 5
DEFAULT_PROMOTE_MAX_CANDIDATES = 24
DEFAULT_LABEL_COLUMN = "forward_return"
DEFAULT_TIME_COLUMN = "timestamp"
DEFAULT_SYMBOL_COLUMN = "symbol"


# ---------------------------------------------------------------------------
# Panel helpers (synthetic default; optional file loader for rank/card).
# ---------------------------------------------------------------------------


def _build_synthetic_panel(args: argparse.Namespace):
    from lumina_quant.research.alpha_search import build_synthetic_search_panel

    return build_synthetic_search_panel(
        n_symbols=max(2, int(getattr(args, "n_symbols", DEFAULT_N_SYMBOLS))),
        n_periods=max(4, int(getattr(args, "n_periods", DEFAULT_N_PERIODS))),
        seed=int(getattr(args, "seed", DEFAULT_PANEL_SEED)),
        signal_strength=float(getattr(args, "signal_strength", DEFAULT_SIGNAL_STRENGTH)),
    )


def _panel_to_long(panel) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Flatten a SearchPanel into row-aligned long-format NumPy arrays.

    Each feature grid ``(n_symbols, n_periods)`` becomes one factor column; the
    symbol/timestamp codes are canonical integer ranges so the downstream reducer
    orders them deterministically.
    """
    n_symbols = int(panel.n_symbols)
    n_periods = int(panel.n_periods)
    sym_codes = np.repeat(np.arange(n_symbols, dtype=np.int64), n_periods)
    ts_codes = np.tile(np.arange(n_periods, dtype=np.int64), n_symbols)
    fwd_long = np.ascontiguousarray(panel.forward_returns, dtype=np.float64).reshape(-1)
    names = list(panel.feature_names())
    if names:
        fmat = np.column_stack(
            [np.asarray(panel.features[name], dtype=np.float64).reshape(-1) for name in names]
        )
    else:
        fmat = np.empty((sym_codes.shape[0], 0), dtype=np.float64)
    return sym_codes, ts_codes, fmat, fwd_long, names


def _evaluate_from_synthetic(args: argparse.Namespace, factor_filter: list[str] | None):
    from lumina_quant.research.factor_ic import evaluate_factor_ic_numpy

    panel = _build_synthetic_panel(args)
    sym_codes, ts_codes, fmat, fwd_long, names = _panel_to_long(panel)
    if factor_filter:
        keep = [n for n in factor_filter if n in names]
        if not keep:
            return None, names
        idx = [names.index(n) for n in keep]
        fmat = fmat[:, idx]
        names = keep
    result = evaluate_factor_ic_numpy(
        sym_codes,
        ts_codes,
        fmat,
        fwd_long,
        names,
        max_decay_lag=max(0, int(getattr(args, "max_decay_lag", DEFAULT_MAX_DECAY_LAG))),
    )
    return result, names


def _evaluate_from_file(args: argparse.Namespace, factor_filter: list[str] | None):
    from lumina_quant.research.factor_ic import evaluate_factor_ic

    path = Path(str(args.panel))
    if not path.exists():
        print(f"[error] panel file not found: {path}", file=sys.stderr)
        return None, []
    try:
        import polars as pl

        if path.suffix.lower() in (".parquet", ".pq"):
            frame = pl.read_parquet(path)
        else:
            frame = pl.read_csv(path)
        all_columns = list(frame.columns)
    except Exception as exc:  # pragma: no cover - defensive loader guard
        print(f"[error] failed to read panel: {exc}", file=sys.stderr)
        return None, []

    reserved = {
        str(args.symbol_column),
        str(args.time_column),
        str(args.label_column),
    }
    factor_columns = (
        [n for n in factor_filter if n in all_columns]
        if factor_filter
        else [c for c in all_columns if c not in reserved]
    )
    if not factor_columns:
        print("[error] no factor columns to evaluate", file=sys.stderr)
        return None, []
    result = evaluate_factor_ic(
        frame,
        factor_columns=factor_columns,
        label_column=str(args.label_column),
        time_column=str(args.time_column),
        symbol_column=str(args.symbol_column),
        max_decay_lag=max(0, int(getattr(args, "max_decay_lag", DEFAULT_MAX_DECAY_LAG))),
    )
    return result, factor_columns


def _resolve_evaluation(args: argparse.Namespace, factor_filter: list[str] | None):
    """Dispatch to the file loader when ``--panel`` is set, else synthetic."""
    if getattr(args, "panel", None):
        return _evaluate_from_file(args, factor_filter)
    return _evaluate_from_synthetic(args, factor_filter)


def _ranking_rows(batch) -> list[dict[str, Any]]:
    """Ordered ranking rows (descending IC-IR, factor name tiebreak)."""
    rows = [res.to_dict() for res in batch.results]
    rows.sort(key=lambda row: (-float(row.get("ic_ir", 0.0)), str(row.get("factor", ""))))
    for position, row in enumerate(rows, start=1):
        row["rank"] = position
    return rows


# ---------------------------------------------------------------------------
# rank.
# ---------------------------------------------------------------------------


def cmd_rank(args: argparse.Namespace) -> int:
    """Rank factors by IC / IC-IR / turnover / decay."""
    factor_filter = list(args.factors) if getattr(args, "factors", None) else None
    batch, _names = _resolve_evaluation(args, factor_filter)
    if batch is None:
        return 1
    rows = _ranking_rows(batch)
    top = int(getattr(args, "top", 0) or 0)
    if top > 0:
        rows = rows[:top]

    payload = {
        "artifact_kind": "alpha_factor_ranking",
        "n_rows": int(batch.n_rows),
        "n_symbols": int(batch.n_symbols),
        "n_timestamps": int(batch.n_timestamps),
        "max_decay_lag": int(batch.max_decay_lag),
        "ranking": rows,
    }
    if bool(getattr(args, "json", False)):
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"Factor ranking ({len(rows)} factors, by IC-IR):")
        print(f"  {'rank':>4}  {'factor':<28}  {'ic_mean':>10}  {'ic_ir':>10}  {'turnover':>10}")
        for row in rows:
            print(
                f"  {int(row['rank']):>4}  {row['factor']!s:<28}  "
                f"{float(row['ic_mean']):>10.5f}  {float(row['ic_ir']):>10.5f}  "
                f"{float(row['turnover_mean']):>10.5f}"
            )
    return 0


# ---------------------------------------------------------------------------
# card.
# ---------------------------------------------------------------------------


def _factor_card_markdown(row: dict[str, Any], batch) -> str:
    autocorr = row.get("rank_autocorr", []) or []
    decay_line = ", ".join(f"{float(v):.5f}" for v in autocorr) if autocorr else "n/a"
    lines = [
        f"# Factor Card: {row['factor']}",
        "",
        "| metric | value |",
        "| --- | --- |",
        f"| n_periods | {int(row['n_periods'])} |",
        f"| n_observations | {int(row['n_observations'])} |",
        f"| ic_mean | {float(row['ic_mean']):.6f} |",
        f"| ic_std | {float(row['ic_std']):.6f} |",
        f"| ic_ir | {float(row['ic_ir']):.6f} |",
        f"| ic_positive_ratio | {float(row['ic_positive_ratio']):.6f} |",
        f"| t_stat | {float(row['t_stat']):.6f} |",
        f"| quantile_spread_mean | {float(row['quantile_spread_mean']):.6f} |",
        f"| turnover_mean | {float(row['turnover_mean']):.6f} |",
        "",
        f"Rank decay (lag 1..{int(batch.max_decay_lag)}): {decay_line}",
        "",
        f"Panel: {int(batch.n_symbols)} symbols x {int(batch.n_timestamps)} timestamps "
        f"({int(batch.n_rows)} rows).",
        "",
    ]
    return "\n".join(lines)


def cmd_card(args: argparse.Namespace) -> int:
    """Emit a factor card (markdown + JSON) for a single factor."""
    factor = str(args.factor)
    batch, _names = _resolve_evaluation(args, [factor])
    if batch is None:
        print(f"[error] factor not found in panel: {factor}", file=sys.stderr)
        return 1
    mapping = batch.as_mapping()
    if factor not in mapping:
        print(f"[error] factor not found in panel: {factor}", file=sys.stderr)
        return 1

    row = mapping[factor].to_dict()
    row["rank_autocorr"] = list(mapping[factor].rank_autocorr)
    card_json = {
        "artifact_kind": "alpha_factor_card",
        "factor": factor,
        "n_symbols": int(batch.n_symbols),
        "n_timestamps": int(batch.n_timestamps),
        "n_rows": int(batch.n_rows),
        "max_decay_lag": int(batch.max_decay_lag),
        "metrics": row,
    }
    markdown = _factor_card_markdown(row, batch)

    out_dir = Path(str(getattr(args, "out", None) or ".")).resolve()
    if getattr(args, "out", None):
        out_dir.mkdir(parents=True, exist_ok=True)
        safe = "".join(ch if (ch.isalnum() or ch in "-_.") else "_" for ch in factor)
        json_path = out_dir / f"{safe}.card.json"
        md_path = out_dir / f"{safe}.card.md"
        json_path.write_text(json.dumps(card_json, indent=2, sort_keys=True), encoding="utf-8")
        md_path.write_text(markdown, encoding="utf-8")
        if bool(getattr(args, "json", False)):
            print(json.dumps({"json": str(json_path), "markdown": str(md_path)}, sort_keys=True))
        else:
            print(f"[ok] wrote factor card: {json_path}")
            print(f"[ok] wrote factor card: {md_path}")
        return 0

    if bool(getattr(args, "json", False)):
        print(json.dumps(card_json, indent=2, sort_keys=True))
    else:
        print(markdown)
    return 0


# ---------------------------------------------------------------------------
# promote.
# ---------------------------------------------------------------------------


def cmd_promote(args: argparse.Namespace) -> int:
    """Run the alpha-search survivorship loop and promote survivors to a ledger."""
    from lumina_quant.research.alpha_candidate_ledger import ResearchLedger
    from lumina_quant.research.alpha_search import (
        DEFAULT_SEARCH_SEED,
        run_alpha_search,
    )

    panel = _build_synthetic_panel(args)

    ledger: ResearchLedger | None = None
    ledger_path = getattr(args, "ledger", None)
    if ledger_path:
        ledger = ResearchLedger(Path(str(ledger_path)))

    result = run_alpha_search(
        panel,
        max_candidates=max(1, int(getattr(args, "max_candidates", DEFAULT_PROMOTE_MAX_CANDIDATES))),
        seed=int(getattr(args, "search_seed", DEFAULT_SEARCH_SEED)),
        ledger=ledger,
    )

    payload: dict[str, Any] = {
        "artifact_kind": "alpha_promotion_report",
        "raw_n": int(result.raw_n),
        "evaluated_n": int(result.evaluated_n),
        "promoted_ids": list(result.promoted_ids),
        "promoted_count": len(result.promoted_ids),
        "n_eff": float(result.n_eff),
        "dispersion": float(result.dispersion),
    }
    if ledger is not None:
        payload["ledger"] = ledger.summary()
        payload["promoted_records"] = ledger.query(passed=True)

    if bool(getattr(args, "json", False)):
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(
            f"Alpha promotion: evaluated {result.evaluated_n} / raw {result.raw_n} candidates; "
            f"promoted {len(result.promoted_ids)} "
            f"(n_eff={result.n_eff:.4f}, dispersion={result.dispersion:.6f})."
        )
        for cid in result.promoted_ids:
            print(f"  promoted: {cid}")
        if ledger is not None:
            print(f"Ledger: {ledger.path}")
    return 0


# ---------------------------------------------------------------------------
# Argument parsing.
# ---------------------------------------------------------------------------


def _add_panel_source_args(sub: argparse.ArgumentParser) -> None:
    sub.add_argument("--panel", default=None, help="Long-format panel file (csv/parquet).")
    sub.add_argument("--symbol-column", dest="symbol_column", default=DEFAULT_SYMBOL_COLUMN)
    sub.add_argument("--time-column", dest="time_column", default=DEFAULT_TIME_COLUMN)
    sub.add_argument("--label-column", dest="label_column", default=DEFAULT_LABEL_COLUMN)
    sub.add_argument("--n-symbols", dest="n_symbols", type=int, default=DEFAULT_N_SYMBOLS)
    sub.add_argument("--n-periods", dest="n_periods", type=int, default=DEFAULT_N_PERIODS)
    sub.add_argument("--seed", type=int, default=DEFAULT_PANEL_SEED)
    sub.add_argument(
        "--signal-strength",
        dest="signal_strength",
        type=float,
        default=DEFAULT_SIGNAL_STRENGTH,
    )
    sub.add_argument(
        "--max-decay-lag", dest="max_decay_lag", type=int, default=DEFAULT_MAX_DECAY_LAG
    )
    sub.add_argument("--json", action="store_true", help="Emit JSON.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="lq alpha",
        description="Factor / alpha research inspection (read-only over research modules).",
    )
    sub = parser.add_subparsers(dest="subcommand")

    _rank = sub.add_parser("rank", help="Rank factors by IC / IC-IR / turnover / decay.")
    _add_panel_source_args(_rank)
    _rank.add_argument("--factors", nargs="+", default=None, help="Restrict to these factors.")
    _rank.add_argument("--top", type=int, default=0, help="Keep only the top-N ranked factors.")

    _card = sub.add_parser("card", help="Emit a factor card (markdown + JSON) for one factor.")
    _add_panel_source_args(_card)
    _card.add_argument("--factor", required=True, help="Factor name to card.")
    _card.add_argument("--out", default=None, help="Output directory (stdout if omitted).")

    _promote = sub.add_parser(
        "promote", help="Run the alpha-search survivorship loop and promote survivors."
    )
    _promote.add_argument("--n-symbols", dest="n_symbols", type=int, default=DEFAULT_N_SYMBOLS)
    _promote.add_argument("--n-periods", dest="n_periods", type=int, default=DEFAULT_N_PERIODS)
    _promote.add_argument("--seed", type=int, default=DEFAULT_PANEL_SEED)
    _promote.add_argument(
        "--signal-strength",
        dest="signal_strength",
        type=float,
        default=DEFAULT_SIGNAL_STRENGTH,
    )
    _promote.add_argument(
        "--max-candidates",
        dest="max_candidates",
        type=int,
        default=DEFAULT_PROMOTE_MAX_CANDIDATES,
    )
    _promote.add_argument("--search-seed", dest="search_seed", type=int, default=None)
    _promote.add_argument("--ledger", default=None, help="Ledger JSONL path for promotions.")
    _promote.add_argument("--json", action="store_true", help="Emit JSON.")

    args = parser.parse_args(argv)

    if not args.subcommand:
        parser.print_help()
        return 0
    if args.subcommand == "rank":
        return cmd_rank(args)
    if args.subcommand == "card":
        return cmd_card(args)
    if args.subcommand == "promote":
        if getattr(args, "search_seed", None) is None:
            from lumina_quant.research.alpha_search import DEFAULT_SEARCH_SEED

            args.search_seed = DEFAULT_SEARCH_SEED
        return cmd_promote(args)
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
