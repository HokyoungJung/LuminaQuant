"""Deterministic alpha discovery loop: generate -> evaluate -> select.

This module is additive research tooling. It closes the loop on top of the C2a
factor-IC evaluator and the C2c survivorship / effective-trials stack:

1.  **generate** -- a deterministic operator-tree enumeration over the formulaic
    operator set (:mod:`lumina_quant.indicators.formulaic_operators`). The *full*
    enumeration count is tracked as ``raw_n`` (an upper bound on the search
    breadth); when it exceeds the search budget a seeded permutation selects a
    reproducible subset while ``raw_n`` still records the total generated.
2.  **evaluate** -- each candidate formula is materialised into a factor panel and
    scored with :func:`lumina_quant.research.factor_ic.evaluate_factor_ic_numpy`
    (rank-IC / IC-IR / turnover / decay). A per-period long/short return series is
    built for every candidate so the family return-correlation matrix (for
    ``N_eff``) is available downstream.
3.  **select** -- every candidate is run through the C2c survivorship gate
    (:func:`lumina_quant.research.survivorship.evaluate_survivorship_gate`) at
    ``num_trials = N_eff`` clamped by ``raw_n``; only survivors are promoted to a
    :class:`lumina_quant.research.alpha_candidate_ledger.ResearchLedger`.
4.  **gated LLM proposer** -- an optional, default-OFF channel accepts frozen /
    seeded proposed candidate specs (never a live call). They are merged into the
    deterministic pool and pass through the *same* survivorship gate, so a
    proposer draft can only be registered if it clears lookahead-free purity and
    the DSR breadth penalty.

Determinism discipline
----------------------
Every step is byte-reproducible: enumeration order is a canonical sort, any
subsampling is a fixed-seed ``np.random.default_rng`` permutation, all float
reductions run in NumPy in a fixed order (reusing the C2a reducer for IC and a
local fixed-order portfolio reducer for candidate returns), and no ``dict`` /
``set`` iteration order is ever observed. Running the loop twice on the same
panel yields the identical promoted set. No scipy / sklearn / eigendecomposition
is used -- only ``research_metrics`` primitives via the survivorship gate.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from lumina_quant.indicators import formulaic_operators as ops
from lumina_quant.research.alpha_candidate_ledger import (
    AlphaCandidateRecord,
    ResearchLedger,
)
from lumina_quant.research.effective_trials import (
    candidate_return_correlation_matrix,
    effective_number_of_trials,
)
from lumina_quant.research.factor_ic import evaluate_factor_ic_numpy
from lumina_quant.research.survivorship import (
    DEFAULT_DSR_THRESHOLD,
    DEFAULT_PBO_THRESHOLD,
    DEFAULT_SPA_PVALUE_THRESHOLD,
    DEFAULT_SPA_SEED,
    empirical_variance_across_trials,
    evaluate_survivorship_gate,
)

__all__ = [
    "DEFAULT_BINARY_OPERATORS",
    "DEFAULT_MAX_CANDIDATES",
    "DEFAULT_SEARCH_SEED",
    "DEFAULT_UNARY_OPERATORS",
    "DEFAULT_WINDOWS",
    "AlphaSearchCandidate",
    "AlphaSearchResult",
    "SearchPanel",
    "generate_candidate_trees",
    "run_alpha_search",
]

# Module-level policy constants. C2b config keys (research.alpha_search.*) may
# override these at call time, but these are the in-module defaults so importing
# the module adds no configuration surface of its own.
DEFAULT_WINDOWS: tuple[int, ...] = (3, 5, 10)
DEFAULT_UNARY_OPERATORS: tuple[str, ...] = ("ts_mean", "ts_stddev", "delta", "ts_rank")
DEFAULT_BINARY_OPERATORS: tuple[str, ...] = ("sub", "div", "ts_corr")
DEFAULT_MAX_CANDIDATES = 256
DEFAULT_SEARCH_SEED = 20260701
_EPS = 1e-12


# ---------------------------------------------------------------------------
# Operator-tree grammar (depth-2 trees over terminal features).
# ---------------------------------------------------------------------------
#
# A tree is represented as an immutable nested tuple so it is hashable, sortable
# and trivially reproducible:
#   ("feat", name)
#   ("u", op_name, child_tree, window)
#   ("b", op_name, left_tree, right_tree, window)


def _formula_str(tree: tuple[Any, ...]) -> str:
    kind = tree[0]
    if kind == "feat":
        return str(tree[1])
    if kind == "u":
        _, op_name, child, window = tree
        return f"{op_name}({_formula_str(child)}, {int(window)})"
    if kind == "b":
        _, op_name, left, right, window = tree
        left_s = _formula_str(left)
        right_s = _formula_str(right)
        if op_name == "sub":
            return f"({left_s} - {right_s})"
        if op_name == "div":
            return f"({left_s} / {right_s})"
        return f"{op_name}({left_s}, {right_s}, {int(window)})"
    raise ValueError(f"alpha_search_unknown_tree_kind:{kind!r}")


@dataclass(frozen=True, slots=True)
class AlphaSearchCandidate:
    """Immutable descriptor for one generated candidate formula."""

    candidate_id: str
    formula: str
    tree: tuple[Any, ...]
    seed: int
    origin: str = "enumerated"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["tree"] = _tree_to_jsonable(self.tree)
        return payload


def _tree_to_jsonable(tree: tuple[Any, ...]) -> list[Any]:
    out: list[Any] = []
    for item in tree:
        if isinstance(item, tuple):
            out.append(_tree_to_jsonable(item))
        else:
            out.append(item)
    return out


@dataclass(frozen=True, slots=True)
class SearchPanel:
    """A deterministic synthetic/real panel for the alpha search loop.

    ``features`` maps a feature name to a ``(n_symbols, n_periods)`` float matrix
    (row = symbol, column = timestamp). ``forward_returns`` has the same shape and
    is the per-period label aligned to each timestamp. ``symbols`` / ``timestamps``
    are 1-D integer/ordinal code arrays used only for deterministic grouping.
    """

    symbols: np.ndarray
    timestamps: np.ndarray
    features: Mapping[str, np.ndarray]
    forward_returns: np.ndarray

    @property
    def n_symbols(self) -> int:
        return int(self.symbols.shape[0])

    @property
    def n_periods(self) -> int:
        return int(self.timestamps.shape[0])

    def feature_names(self) -> tuple[str, ...]:
        return tuple(sorted(self.features))


@dataclass(frozen=True, slots=True)
class AlphaSearchResult:
    """Deterministic summary of one generate -> evaluate -> select run."""

    raw_n: int
    evaluated_n: int
    promoted_ids: tuple[str, ...]
    n_eff: float
    dispersion: float
    records: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_kind": "alpha_search_result",
            "raw_n": int(self.raw_n),
            "evaluated_n": int(self.evaluated_n),
            "promoted_ids": list(self.promoted_ids),
            "n_eff": float(self.n_eff),
            "dispersion": float(self.dispersion),
            "records": [dict(row) for row in self.records],
        }


# ---------------------------------------------------------------------------
# (1) generate.
# ---------------------------------------------------------------------------


def generate_candidate_trees(
    feature_names: Sequence[str],
    *,
    windows: Sequence[int] = DEFAULT_WINDOWS,
    unary_operators: Sequence[str] = DEFAULT_UNARY_OPERATORS,
    binary_operators: Sequence[str] = DEFAULT_BINARY_OPERATORS,
) -> list[tuple[Any, ...]]:
    """Enumerate every depth-2 operator tree in a canonical, deterministic order.

    The returned list is the *full* enumeration (its length is the upper-bound
    ``raw_n``). Ordering is a stable sort on the rendered formula string so it is
    independent of the iteration order of the input collections.
    """
    feats = sorted({str(name) for name in feature_names})
    win = sorted({int(w) for w in windows if int(w) >= 1})
    unary = sorted({str(op) for op in unary_operators})
    binary = sorted({str(op) for op in binary_operators})

    trees: list[tuple[Any, ...]] = []
    for op_name in unary:
        for feat in feats:
            for window in win:
                trees.append(("u", op_name, ("feat", feat), window))
    for op_name in binary:
        for left in feats:
            for right in feats:
                if left == right:
                    continue
                if op_name == "ts_corr":
                    # Symmetric: keep only the canonical (left < right) ordering.
                    if left > right:
                        continue
                    for window in win:
                        trees.append(("b", op_name, ("feat", left), ("feat", right), window))
                else:
                    # Asymmetric (sub / div): the window slot is inert (0).
                    trees.append(("b", op_name, ("feat", left), ("feat", right), 0))

    trees.sort(key=_formula_str)
    return trees


def _select_trees(
    trees: Sequence[tuple[Any, ...]],
    *,
    max_candidates: int,
    seed: int,
) -> list[tuple[Any, ...]]:
    """Cap the pool to ``max_candidates`` via a fixed-seed permutation.

    When the pool already fits it is returned verbatim (canonical order). When it
    overflows, a deterministic ``default_rng(seed)`` permutation picks the subset;
    the survivors are re-sorted into canonical order so downstream indexing is
    reproducible regardless of the permutation draw.
    """
    total = len(trees)
    cap = max(1, int(max_candidates))
    if total <= cap:
        return list(trees)
    rng = np.random.default_rng(int(seed))
    picks = rng.permutation(total)[:cap]
    chosen = [trees[int(i)] for i in np.sort(picks)]
    chosen.sort(key=_formula_str)
    return chosen


# ---------------------------------------------------------------------------
# (2) evaluate -- materialise factor panel + candidate return matrix.
# ---------------------------------------------------------------------------


def _eval_tree_on_symbol(
    tree: tuple[Any, ...],
    feat_series: Mapping[str, pd.Series],
    index: pd.Index,
) -> pd.Series:
    kind = tree[0]
    if kind == "feat":
        return ops.to_series(feat_series[str(tree[1])], index)
    if kind == "u":
        _, op_name, child, window = tree
        child_series = _eval_tree_on_symbol(child, feat_series, index)
        if op_name == "ts_mean":
            return ops.ts_mean_series(child_series, int(window), index=index)
        if op_name == "ts_stddev":
            return ops.ts_stddev_series(child_series, int(window), index=index)
        if op_name == "delta":
            return ops.delta_series(child_series, int(window), index=index)
        if op_name == "ts_rank":
            return ops.ts_rank_series(child_series, int(window), index=index)
        raise ValueError(f"alpha_search_unknown_unary_op:{op_name!r}")
    if kind == "b":
        _, op_name, left, right, window = tree
        left_series = _eval_tree_on_symbol(left, feat_series, index)
        right_series = _eval_tree_on_symbol(right, feat_series, index)
        if op_name == "sub":
            return left_series - right_series
        if op_name == "div":
            denom = right_series.where(right_series.abs() > _EPS)
            return left_series.divide(denom)
        if op_name == "ts_corr":
            return ops.ts_corr_series(left_series, right_series, int(window), index=index)
        raise ValueError(f"alpha_search_unknown_binary_op:{op_name!r}")
    raise ValueError(f"alpha_search_unknown_tree_kind:{kind!r}")


def _average_rank(values: np.ndarray) -> np.ndarray:
    """Fixed-order average ranks with tie handling (matches factor_ic)."""
    n = values.shape[0]
    order = np.argsort(values, kind="stable")
    ranks = np.empty(n, dtype=np.float64)
    sorted_vals = values[order]
    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_vals[j] == sorted_vals[i]:
            j += 1
        avg = (i + j - 1) / 2.0 + 1.0
        ranks[order[i:j]] = avg
        i = j
    return ranks


def _cross_section_portfolio_return(factor_col: np.ndarray, fwd_col: np.ndarray) -> float:
    """Long/short portfolio return: L1-normalised centered-rank weights . fwd.

    Fully deterministic (fixed-order NumPy sums, no BLAS reductions). Non-finite
    factor / label entries drop out of the cross-section.
    """
    mask = np.isfinite(factor_col) & np.isfinite(fwd_col)
    m = int(np.count_nonzero(mask))
    if m < 2:
        return 0.0
    fv = factor_col[mask]
    lb = fwd_col[mask]
    ranks = _average_rank(fv)
    centered = ranks - float(np.sum(ranks)) / m
    l1 = float(np.sum(np.abs(centered)))
    if l1 <= _EPS:
        return 0.0
    weights = centered / l1
    return float(np.sum(weights * lb))


def _series_sharpe(returns: np.ndarray) -> float:
    """Per-period (non-annualised) Sharpe via fixed-order sums."""
    r = returns[np.isfinite(returns)]
    n = int(r.shape[0])
    if n < 2:
        return 0.0
    mean = float(np.sum(r)) / n
    dev = r - mean
    var = float(np.sum(dev * dev)) / (n - 1)
    if var <= _EPS:
        return 0.0
    return mean / math.sqrt(var)


@dataclass(frozen=True, slots=True)
class _EvaluatedCandidate:
    candidate: AlphaSearchCandidate
    ic: float
    icir: float
    turnover: float
    returns: np.ndarray
    sharpe: float


def _evaluate_candidates(
    candidates: Sequence[AlphaSearchCandidate],
    panel: SearchPanel,
) -> list[_EvaluatedCandidate]:
    n_symbols = panel.n_symbols
    n_periods = panel.n_periods
    index = pd.RangeIndex(n_periods)

    # Per-symbol feature series (built once, reused for every candidate).
    per_symbol_features: list[dict[str, pd.Series]] = []
    for s in range(n_symbols):
        per_symbol_features.append(
            {
                name: pd.Series(np.asarray(matrix[s], dtype=float), index=index, dtype=float)
                for name, matrix in panel.features.items()
            }
        )

    fwd = np.ascontiguousarray(panel.forward_returns, dtype=np.float64)

    # Long panel scaffolding: symbol/timestamp codes row-aligned (symbol-major).
    sym_codes = np.repeat(np.arange(n_symbols, dtype=np.int64), n_periods)
    ts_codes = np.tile(np.arange(n_periods, dtype=np.int64), n_symbols)
    fwd_long = fwd.reshape(-1)

    k = len(candidates)
    factor_matrix = np.full((n_symbols * n_periods, k), np.nan, dtype=np.float64)
    # factor_by_ts[col] -> (n_symbols, n_periods) for portfolio returns.
    factor_grids: list[np.ndarray] = []
    for col, cand in enumerate(candidates):
        grid = np.full((n_symbols, n_periods), np.nan, dtype=np.float64)
        for s in range(n_symbols):
            series = _eval_tree_on_symbol(cand.tree, per_symbol_features[s], index)
            arr = np.asarray(series.to_numpy(dtype=float), dtype=np.float64)
            arr = np.where(np.isfinite(arr), arr, np.nan)
            grid[s, : arr.shape[0]] = arr[:n_periods]
        factor_grids.append(grid)
        factor_matrix[:, col] = grid.reshape(-1)

    names = [cand.candidate_id for cand in candidates]
    ic_result = evaluate_factor_ic_numpy(
        sym_codes,
        ts_codes,
        factor_matrix,
        fwd_long,
        names,
    ).as_mapping()

    evaluated: list[_EvaluatedCandidate] = []
    for col, cand in enumerate(candidates):
        grid = factor_grids[col]
        returns = np.empty(n_periods, dtype=np.float64)
        for t in range(n_periods):
            returns[t] = _cross_section_portfolio_return(grid[:, t], fwd[:, t])
        res = ic_result[cand.candidate_id]
        evaluated.append(
            _EvaluatedCandidate(
                candidate=cand,
                ic=float(res.ic_mean),
                icir=float(res.ic_ir),
                turnover=float(res.turnover_mean),
                returns=returns,
                sharpe=_series_sharpe(returns),
            )
        )
    return evaluated


# ---------------------------------------------------------------------------
# Public entry point: generate -> evaluate -> select.
# ---------------------------------------------------------------------------


def run_alpha_search(
    panel: SearchPanel,
    *,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
    seed: int = DEFAULT_SEARCH_SEED,
    windows: Sequence[int] = DEFAULT_WINDOWS,
    unary_operators: Sequence[str] = DEFAULT_UNARY_OPERATORS,
    binary_operators: Sequence[str] = DEFAULT_BINARY_OPERATORS,
    dsr_threshold: float = DEFAULT_DSR_THRESHOLD,
    spa_pvalue_threshold: float = DEFAULT_SPA_PVALUE_THRESHOLD,
    pbo_threshold: float = DEFAULT_PBO_THRESHOLD,
    require_spa: bool = False,
    require_pbo: bool = False,
    spa_seed: int = DEFAULT_SPA_SEED,
    ledger: ResearchLedger | None = None,
    enable_llm_proposer: bool = False,
    llm_proposals: Sequence[AlphaSearchCandidate] | None = None,
) -> AlphaSearchResult:
    """Run the deterministic generate -> evaluate -> select loop.

    Parameters mirror the ``research.alpha_search.*`` config gate. ``ledger``, when
    supplied, receives one :class:`AlphaCandidateRecord` per *surviving* candidate
    (append-only, in canonical order). ``enable_llm_proposer`` (default OFF) merges
    the frozen ``llm_proposals`` specs into the pool; they pass through the same
    survivorship gate and only register if they clear it.

    Returns an :class:`AlphaSearchResult`. ``raw_n`` is the full generated breadth
    (upper bound) even when the evaluated pool was capped by ``max_candidates``.
    """
    trees = generate_candidate_trees(
        panel.feature_names(),
        windows=windows,
        unary_operators=unary_operators,
        binary_operators=binary_operators,
    )
    raw_n = len(trees)

    proposals: list[AlphaSearchCandidate] = []
    if enable_llm_proposer and llm_proposals:
        # Frozen/seeded proposer artifacts merged deterministically (canonical
        # order by formula). Never a live call.
        proposals = sorted(llm_proposals, key=lambda c: c.formula)
        raw_n += len(proposals)

    selected = _select_trees(trees, max_candidates=max_candidates, seed=seed)

    candidates: list[AlphaSearchCandidate] = []
    for idx, tree in enumerate(selected):
        formula = _formula_str(tree)
        candidates.append(
            AlphaSearchCandidate(
                candidate_id=f"as-{idx:04d}",
                formula=formula,
                tree=tree,
                seed=int(seed),
                origin="enumerated",
            )
        )
    for j, proposal in enumerate(proposals):
        candidates.append(
            AlphaSearchCandidate(
                candidate_id=f"llm-{j:04d}",
                formula=proposal.formula,
                tree=proposal.tree,
                seed=int(proposal.seed),
                origin="llm_proposer",
            )
        )

    if not candidates:
        return AlphaSearchResult(
            raw_n=int(raw_n),
            evaluated_n=0,
            promoted_ids=(),
            n_eff=1.0,
            dispersion=0.0,
            records=(),
        )

    evaluated = _evaluate_candidates(candidates, panel)

    return_matrix = np.vstack([ev.returns for ev in evaluated])
    family_sharpes = np.asarray([ev.sharpe for ev in evaluated], dtype=float)
    corr = candidate_return_correlation_matrix(return_matrix)
    n_eff = effective_number_of_trials(corr, raw_n=int(raw_n))
    dispersion = empirical_variance_across_trials(family_sharpes)

    records: list[dict[str, Any]] = []
    promoted_ids: list[str] = []
    for ev in evaluated:
        verdict = evaluate_survivorship_gate(
            ev.returns,
            family_sharpes=family_sharpes,
            correlation_matrix=corr,
            dsr_threshold=dsr_threshold,
            spa_pvalue_threshold=spa_pvalue_threshold,
            pbo_threshold=pbo_threshold,
            require_spa=require_spa,
            require_pbo=require_pbo,
            spa_seed=spa_seed,
        )
        record = AlphaCandidateRecord(
            candidate_id=ev.candidate.candidate_id,
            formula=ev.candidate.formula,
            seed=ev.candidate.seed,
            ic=float(ev.ic),
            icir=float(ev.icir),
            turnover=float(ev.turnover),
            sharpe=float(ev.sharpe),
            deflated_sharpe=float(verdict.deflated_sharpe),
            n_eff=float(verdict.n_eff),
            dispersion=float(verdict.variance_across_trials),
            spa_pvalue=float(verdict.spa_pvalue),
            pbo=float(verdict.pbo),
            raw_n=int(raw_n),
            passed=bool(verdict.passed),
        )
        records.append(record.to_dict())
        if verdict.passed:
            promoted_ids.append(ev.candidate.candidate_id)
            if ledger is not None:
                ledger.append(record)

    return AlphaSearchResult(
        raw_n=int(raw_n),
        evaluated_n=len(evaluated),
        promoted_ids=tuple(promoted_ids),
        n_eff=float(n_eff),
        dispersion=float(dispersion),
        records=tuple(records),
    )


def build_synthetic_search_panel(
    *,
    n_symbols: int = 8,
    n_periods: int = 320,
    seed: int = 7,
    signal_strength: float = 0.6,
) -> SearchPanel:
    """Deterministic synthetic panel with a planted close-momentum edge.

    A subset of the feature space (``ts_mean``/``delta`` of ``close``) genuinely
    predicts ``forward_return`` so the select stage has both true edges and nulls
    to separate. Purely for tests / offline demos; never used by live/backtest.
    """
    rng = np.random.default_rng(int(seed))
    symbols = np.arange(int(n_symbols), dtype=np.int64)
    timestamps = np.arange(int(n_periods), dtype=np.int64)

    # Base random-walk closes + a noisy volume feature.
    steps = rng.standard_normal((n_symbols, n_periods)) * 0.01
    close = 100.0 * np.exp(np.cumsum(steps, axis=1))
    volume = np.abs(rng.standard_normal((n_symbols, n_periods))) * 1000.0 + 500.0
    ret = np.zeros((n_symbols, n_periods), dtype=float)
    ret[:, 1:] = np.diff(close, axis=1) / close[:, :-1]

    # Planted edge: forward return loads on recent short-window momentum.
    momentum = np.zeros((n_symbols, n_periods), dtype=float)
    momentum[:, 3:] = (close[:, 3:] - close[:, :-3]) / close[:, :-3]
    noise = rng.standard_normal((n_symbols, n_periods)) * 0.01
    forward = float(signal_strength) * momentum + noise

    return SearchPanel(
        symbols=symbols,
        timestamps=timestamps,
        features={
            "close": close,
            "volume": volume,
            "ret": ret,
        },
        forward_returns=forward,
    )
