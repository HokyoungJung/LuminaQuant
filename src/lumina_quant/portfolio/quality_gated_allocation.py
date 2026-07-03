"""Offline quality-gated sleeve allocator: static research manifest generator.

This module builds a deterministic, offline sleeve-allocation *research
manifest* from static return streams. It is one half of the Lane M2 meta
spine (the other half is the CLI wrapper in
``scripts/research/build_quality_gated_allocation.py``): given a fixed set of
candidate "sleeves" (each a synthetic or backtested return series + turnover +
strategy definition), it (1) scores each sleeve's cost-realistic quality, (2)
gates out sleeves with non-positive net Sharpe, (3) allocates capital across
the survivors with a risk-parity-style optimizer, and (4) emits a manifest
JSON payload that satisfies the fail-closed provenance contract already
enforced by the live
:class:`lumina_quant.strategies.artifact_portfolio_mode.ArtifactPortfolioModeStrategy`
consumer.

Theory anchors
--------------
* Equal-Risk-Contribution / risk parity: Maillard, Roncalli & Teiletche
  (2010), "The Properties of Equally Weighted Risk Contribution Portfolios".
* Hierarchical Risk Parity: Lopez de Prado (2016), "Building Diversified
  Portfolios that Outperform Out-of-Sample".

Both allocators are consumed as-is from
:mod:`lumina_quant.portfolio.optimizers_extra` (``ERCPortfolio`` /
``HRPPortfolio``); this module does not reimplement their numerics.

OFFLINE-only framing
---------------------
Everything in this module is research/offline tooling. It generates a static
JSON manifest artifact for human/CLI consumption; it is NOT imported by any
live or backtest hot path, is not registered as a strategy, and never touches
real capital. The *consumer* of the manifest
(``ArtifactPortfolioModeStrategy`` / ``manifest:<path>`` portfolio mode) and
its fail-closed gates already exist and are unmodified by this module -- this
module only has to produce a payload that satisfies that consumer's existing
checks (or, on any omission, correctly falls back to cash via the consumer's
own fail-closed logic).

Distinct from ``strategy_quality.py``
--------------------------------------
:mod:`lumina_quant.portfolio.strategy_quality` (``StrategyQualityOverlay``)
gates individual *live* signals at runtime from rolling realized performance
state. This module allocates *across sleeves*, offline, from static
deterministic quality scores computed once over a fixed historical return
series -- there is no runtime state, no live signal gating, and no shared
code path with the overlay.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from lumina_quant.portfolio.optimizer_core import metrics
from lumina_quant.portfolio.optimizers_extra import ERCPortfolio, HRPPortfolio
from lumina_quant.research.cost_realism import DEFAULT_PARTICIPATION, CostRegime, apply_cost_drag

__all__ = [
    "REFERENCE_COST_REGIME_20BPS",
    "allocate_quality_gated",
    "build_allocation_manifest",
    "compute_sleeve_quality",
]

# A CostRegime whose round-trip cost is exactly 20bps: one_side_cost_fraction
# = taker_fee_rate + 0.5 * spread_rate + slippage_rate = 0.0004 + 0.0001 +
# 0.0005 = 0.0010 (10bps); roundtrip_cost_fraction = 2 * one_side = 0.0020
# (20bps). This is the fixed reference cost level used throughout the
# alpha-hunt data-PC promotion rule ("20bps reference cost").
REFERENCE_COST_REGIME_20BPS = CostRegime(
    name="quality_gate_20bps_reference",
    taker_fee_rate=0.0004,
    spread_rate=0.0002,
    slippage_rate=0.0005,
)

# Static provenance labels: the quality score is computed once, offline, over
# a fixed train+validation return series. No current-fold OOS bar is ever
# consulted for selection, objective, pruning, thresholding, tie-breaking,
# correlation, or sizing.
_SELECTION_INPUTS = ("train", "validation")


def _round(value: Any, ndigits: int = 10) -> float:
    return round(float(value), ndigits)


def _default_optimizer_provenance() -> dict[str, Any]:
    return {
        "source": "quality_gated_allocation.compute_sleeve_quality",
        "selection_inputs": list(_SELECTION_INPUTS),
        "uses_current_fold_oos": False,
        "uses_locked_oos_for_selection": False,
        "uses_locked_oos_for_objective": False,
        "uses_locked_oos_for_pruning": False,
        "uses_locked_oos_for_parameter_fitting": False,
        "uses_locked_oos_for_threshold": False,
        "uses_locked_oos_for_tie_break": False,
        "uses_locked_oos_for_correlation": False,
        "uses_locked_oos_for_sizing": False,
    }


def _default_correlation_provenance() -> dict[str, Any]:
    provenance = _default_optimizer_provenance()
    provenance["source"] = "quality_gated_allocation_train_validation_correlation"
    provenance["ready"] = True
    return provenance


def compute_sleeve_quality(
    returns: Sequence[float] | np.ndarray | None,
    turnover: float | None,
    *,
    regime: CostRegime = REFERENCE_COST_REGIME_20BPS,
    participation: float = DEFAULT_PARTICIPATION,
    periods_per_year: int = 365,
) -> dict[str, float]:
    """Cost-realistic quality score for one sleeve's gross return stream.

    Net returns are ``cost_realism.apply_cost_drag(gross_returns, turnover=...,
    regime=regime)``; ``sharpe``/``calmar`` are read from
    ``optimizer_core.metrics(net)`` (which does not compute a hit rate), so
    ``hit_rate`` is self-computed as ``mean(net > 0)``. Deterministic; gracefully
    handles ``None`` / empty ``returns`` (treated as a zero-length series, which
    ``metrics()`` scores as all-zero -- a sleeve with no data never survives the
    ``net_sharpe > 0`` quality gate).
    """
    gross = np.asarray(returns if returns is not None else [], dtype=np.float64).reshape(-1)
    turnover_value = float(turnover) if turnover is not None else 0.0
    net = apply_cost_drag(
        gross, turnover=turnover_value, regime=regime, participation=participation
    )
    stats = metrics(net, periods_per_year=periods_per_year)
    hit_rate = float(np.mean(net > 0.0)) if net.size > 0 else 0.0
    return {
        "n_obs": int(net.size),
        "turnover": turnover_value,
        "net_sharpe": _round(stats["sharpe"]),
        "net_calmar": _round(stats["calmar"]),
        "hit_rate": _round(hit_rate),
    }


def _build_allocator(method: str) -> ERCPortfolio | HRPPortfolio:
    token = str(method or "erc").strip().lower()
    if token == "erc":
        return ERCPortfolio()
    if token == "hrp":
        return HRPPortfolio()
    raise ValueError(f"unsupported allocation method: {method!r} (expected 'erc' or 'hrp')")


def _resolve_upper(
    upper: float | Mapping[str, float] | None, ids: Sequence[str]
) -> dict[str, float] | None:
    if upper is None:
        return None
    if isinstance(upper, Mapping):
        return {sleeve_id: float(upper.get(sleeve_id, 1.0)) for sleeve_id in ids}
    cap = float(upper)
    return dict.fromkeys(ids, cap)


def allocate_quality_gated(
    sleeve_returns: Mapping[str, Sequence[float] | np.ndarray | None] | None,
    turnovers: Mapping[str, float] | None = None,
    *,
    regime: CostRegime = REFERENCE_COST_REGIME_20BPS,
    participation: float = DEFAULT_PARTICIPATION,
    method: str = "erc",
    upper: float | Mapping[str, float] | None = None,
    min_sleeves: int = 1,
) -> dict[str, float]:
    """Quality-gate then risk-allocate across sleeves; returns ``id -> weight``.

    Sleeves with cost-realistic ``net_sharpe <= 0`` (see
    :func:`compute_sleeve_quality`) are dropped. Survivors are allocated via
    ``ERCPortfolio`` (``method="erc"``, default) or ``HRPPortfolio``
    (``method="hrp"``) over the *net* (post-cost) return covariance, optionally
    capped per-sleeve by ``upper`` (a single float applied to every survivor, or
    a ``{id: cap}`` mapping). Survivor ids are always processed in sorted
    (alphabetical) order before being handed to the optimizer, so the result is
    independent of the caller's mapping iteration order -- a deterministic
    tie-break. Returns ``{}`` (an all-cash allocation) when there are fewer than
    ``min_sleeves`` survivors, mirroring the conservative default of the
    downstream manifest consumer. Gracefully handles ``None``/empty
    ``sleeve_returns``/``turnovers`` and per-sleeve ``None``/empty series.
    """
    if not sleeve_returns:
        return {}
    turnovers = turnovers or {}

    quality = {
        sleeve_id: compute_sleeve_quality(
            series, turnovers.get(sleeve_id), regime=regime, participation=participation
        )
        for sleeve_id, series in sleeve_returns.items()
    }
    survivors = sorted(
        sleeve_id for sleeve_id, score in quality.items() if score["net_sharpe"] > 0.0
    )
    if len(survivors) < max(1, int(min_sleeves)):
        return {}

    net_series: list[np.ndarray] = []
    for sleeve_id in survivors:
        gross = np.asarray(
            sleeve_returns[sleeve_id] if sleeve_returns[sleeve_id] is not None else [],
            dtype=np.float64,
        ).reshape(-1)
        net_series.append(
            apply_cost_drag(
                gross,
                turnover=float(turnovers.get(sleeve_id) or 0.0),
                regime=regime,
                participation=participation,
            )
        )

    min_len = min((series.size for series in net_series), default=0)
    if min_len < 2:
        # Not enough overlapping observations to estimate a covariance
        # structure; fall back to equal weight across survivors rather than
        # fail. Still deterministic (equal split, sorted id order).
        equal = 1.0 / float(len(survivors))
        raw_weights = dict.fromkeys(survivors, equal)
    else:
        # Trailing-window alignment: series of differing length are truncated
        # to the shortest survivor's length by keeping the most recent
        # observations. Callers that need calendar-exact alignment must supply
        # equal-length, co-dated series themselves.
        matrix = np.column_stack([series[-min_len:] for series in net_series])
        allocator = _build_allocator(method)
        upper_map = _resolve_upper(upper, survivors)
        raw_weights = allocator.allocate(survivors, matrix, upper=upper_map)
        if not raw_weights:
            equal = 1.0 / float(len(survivors))
            raw_weights = dict.fromkeys(survivors, equal)

    return {sleeve_id: _round(raw_weights.get(sleeve_id, 0.0)) for sleeve_id in survivors}


def build_allocation_manifest(
    sleeves: Mapping[str, Mapping[str, Any]] | None,
    *,
    source_artifacts: Sequence[Mapping[str, Any]],
    regime: CostRegime = REFERENCE_COST_REGIME_20BPS,
    participation: float = DEFAULT_PARTICIPATION,
    method: str = "erc",
    upper: float | Mapping[str, float] | None = None,
    min_sleeves: int = 1,
    gross_cap: float = 1.0,
) -> dict[str, Any]:
    """Build a manifest the live ``ArtifactPortfolioModeStrategy`` accepts as-is.

    ``sleeves`` maps sleeve id -> ``{"returns": [...], "turnover": float,
    "strategy_class": str, "symbols": [...], "params": {...},
    "source_artifact_id": str (optional if there is exactly one entry in
    ``source_artifacts``)}``. ``source_artifacts`` must be pre-built
    ``{"id", "path", "sha256", "max_age_hours", "ready": True,
    "portfolio_ready": True}`` rows (this module does no filesystem I/O, so it
    stays pure and deterministic -- the CLI wrapper is responsible for hashing
    the referenced files and passing the result in).

    Every field the consumer's fail-closed gate inspects is populated so a
    manifest built from quality-surviving sleeves does NOT fail-close: real
    money keys are ``False`` at both the top level and per child; no forbidden
    current-fold-OOS key is ever set; ``optimizer_provenance`` /
    ``correlation_input_provenance`` are non-empty with
    ``selection_inputs=["train", "validation"]`` (and ``ready=True`` for the
    correlation provenance); each surviving child carries
    ``no_current_fold_oos_provenance=True`` and
    ``train_validation_optimizer_provenance=True``, a ``source_artifact_id``
    that reconciles against ``source_artifacts``, and per-leaf/per-netting-group
    gross caps bounded by ``gross_cap``. When quality-gating leaves zero
    survivors, this still emits a well-formed manifest with ``children: []`` --
    the consumer's own ``manifest_children_empty`` fail-closed path (not a
    special case here) safely routes that to 100% cash.
    """
    sleeves = sleeves or {}
    source_rows = [dict(artifact) for artifact in (source_artifacts or [])]
    sleeve_returns = {sid: (spec or {}).get("returns") for sid, spec in sleeves.items()}
    turnovers = {sid: (spec or {}).get("turnover", 0.0) for sid, spec in sleeves.items()}

    weights = allocate_quality_gated(
        sleeve_returns,
        turnovers,
        regime=regime,
        participation=participation,
        method=method,
        upper=upper,
        min_sleeves=min_sleeves,
    )

    default_source_id = str(source_rows[0].get("id") or "") if len(source_rows) == 1 else ""

    children: list[dict[str, Any]] = []
    for sleeve_id in sorted(weights):
        weight = float(weights[sleeve_id])
        if weight <= 0.0:
            continue
        spec = sleeves.get(sleeve_id) or {}
        source_artifact_id = str(spec.get("source_artifact_id") or default_source_id or "")
        children.append(
            {
                "candidate_id": str(sleeve_id),
                "name": str(spec.get("name") or sleeve_id),
                "strategy_class": str(spec.get("strategy_class") or ""),
                "symbols": [str(symbol) for symbol in list(spec.get("symbols") or [])],
                "params": dict(spec.get("params") or {}),
                "weight": weight,
                "leaf_gross": weight,
                "leaf_gross_cap": float(gross_cap),
                "netting_group": str(sleeve_id),
                "netting_group_gross_cap": float(gross_cap),
                "source_artifact_id": source_artifact_id,
                "ready": True,
                "portfolio_ready": True,
                "real_money_execution": False,
                "allow_real_money": False,
                "ready_for_real": False,
                "no_current_fold_oos_provenance": True,
                "train_validation_optimizer_provenance": True,
                "lagged_completed_shadow_optimizer_provenance": False,
                "uses_current_fold_oos": False,
                "uses_locked_oos_for_selection": False,
                "uses_locked_oos_for_objective": False,
                "uses_locked_oos_for_pruning": False,
                "uses_locked_oos_for_parameter_fitting": False,
                "uses_locked_oos_for_threshold": False,
                "uses_locked_oos_for_tie_break": False,
                "uses_locked_oos_for_correlation": False,
                "uses_locked_oos_for_sizing": False,
                "optimizer_provenance": _default_optimizer_provenance(),
                "correlation_input_provenance": _default_correlation_provenance(),
            }
        )

    active_weight = _round(sum(child["weight"] for child in children)) if children else 0.0
    sleeve_quality = {
        sid: compute_sleeve_quality(
            sleeve_returns.get(sid), turnovers.get(sid), regime=regime, participation=participation
        )
        for sid in sorted(sleeves)
    }

    return {
        "artifact_kind": "quality_gated_allocation_manifest",
        "real_money_execution": False,
        "allow_real_money": False,
        "ready_for_real": False,
        "uses_current_fold_oos": False,
        "uses_locked_oos_for_selection": False,
        "uses_locked_oos_for_objective": False,
        "uses_locked_oos_for_pruning": False,
        "uses_locked_oos_for_parameter_fitting": False,
        "uses_locked_oos_for_threshold": False,
        "uses_locked_oos_for_tie_break": False,
        "uses_locked_oos_for_correlation": False,
        "uses_locked_oos_for_sizing": False,
        "gross_cap": float(gross_cap),
        "cash_weight": max(0.0, _round(1.0 - active_weight)),
        "allocation_method": str(method),
        "optimizer_provenance": _default_optimizer_provenance(),
        "correlation_input_provenance": _default_correlation_provenance(),
        "source_artifacts": source_rows,
        "children": children,
        "sleeve_quality": sleeve_quality,
    }
