"""Strategy-factory helpers for broad quant research runs."""

from .candidate_library import (
    DEFAULT_BINANCE_TOP10_PLUS_METALS,
    DEFAULT_TIMEFRAMES,
    StrategyCandidate,
    build_binance_futures_candidates,
    build_candidate_manifest,
)
from .pipeline import (
    build_research_command,
    build_shortlist_payload,
    extract_saved_report_path,
    render_shortlist_markdown,
    write_candidate_manifest,
)
from .selection import (
    DEFAULT_ROBUST_SCORE_PARAMS,
    DEFAULT_ROBUST_SCORE_WEIGHTS,
    allocate_portfolio_weights,
    build_single_asset_portfolio_sets,
    candidate_identity,
    candidate_mix_type,
    hurdle_score,
    robust_score_from_metrics,
    select_diversified_shortlist,
    strategy_family,
    summarize_shortlist,
)

__all__ = [
    "DEFAULT_BINANCE_TOP10_PLUS_METALS",
    "DEFAULT_ROBUST_SCORE_PARAMS",
    "DEFAULT_ROBUST_SCORE_WEIGHTS",
    "DEFAULT_TIMEFRAMES",
    "StrategyCandidate",
    "allocate_portfolio_weights",
    "build_binance_futures_candidates",
    "build_candidate_manifest",
    "build_research_command",
    "build_shortlist_payload",
    "build_single_asset_portfolio_sets",
    "candidate_identity",
    "candidate_mix_type",
    "extract_saved_report_path",
    "hurdle_score",
    "render_shortlist_markdown",
    "robust_score_from_metrics",
    "select_diversified_shortlist",
    "strategy_family",
    "summarize_shortlist",
    "write_candidate_manifest",
]
