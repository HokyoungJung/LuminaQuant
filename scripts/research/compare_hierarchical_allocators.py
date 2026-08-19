"""Compare hierarchical / robust allocators on one aligned return panel (data-PC helper).

Input JSON is either ``{"returns": {"id": [r1, r2, ...], ...}}`` or the
quality-gated cell shape (``{"sleeves": {"id": {"returns": [...]}}}``) with the
returns already materialized (common-date aligned NET returns).  For every
method the script prints the weight vector plus three diagnostics that are
independent of any performance claim:

* ``eff_n``   -- effective number of bets ``1 / sum(w^2)``;
* ``div_ratio`` -- diversification ratio ``(w'sigma) / sqrt(w'Sw)``;
* ``max_w``   -- largest single weight.

Deterministic, no filesystem writes unless ``--output`` is given.  Not a
backtest and not a promotion decision -- it only shows how the allocation
family reshapes the same covariance.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from lumina_quant.portfolio import hierarchical as hier
from lumina_quant.portfolio.optimizer_core import ledoit_wolf_shrunk_covariance
from lumina_quant.portfolio.optimizers_extra import erc_weights, hrp_weights_from_returns
from lumina_quant.portfolio.quality_gated_allocation import (
    REFERENCE_COST_REGIME_20BPS,
    _prepare_return_series,
    _validate_materialized_data_contract,
)
from lumina_quant.research.cost_realism import DEFAULT_PARTICIPATION, apply_cost_drag

METHODS: tuple[str, ...] = (
    "equal_weight",
    "inverse_vol",
    "erc",
    "hrp_threshold",
    "hrp_dendrogram",
    "constrained_hrp",
    "herc",
    "nco",
    "wasserstein_dro",
    "graph_inverse_centrality",
)


def _load_returns(path: Path) -> tuple[list[str], np.ndarray]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw = payload.get("returns")
    timestamps = payload.get("return_timestamps") or {}
    net_flags = payload.get("returns_are_net") or {}
    turnovers = payload.get("turnovers") or {}
    if not isinstance(raw, dict):
        sleeves = payload.get("sleeves") or {}
        _validate_materialized_data_contract(sleeves)
        raw = {sid: (spec or {}).get("returns") for sid, spec in sleeves.items()}
        timestamps = {
            sid: (spec or {}).get("return_timestamps")
            for sid, spec in sleeves.items()
            if "return_timestamps" in (spec or {})
        }
        net_flags = {
            sid: bool((spec or {}).get("returns_are_net", False)) for sid, spec in sleeves.items()
        }
        turnovers = {
            sid: float((spec or {}).get("turnover") or 0.0) for sid, spec in sleeves.items()
        }
    ids = sorted(sid for sid, series in raw.items() if isinstance(series, list) and series)
    if len(ids) < 2:
        raise SystemExit("need at least two sleeves/assets with materialized returns")
    prepared, _alignment, _common = _prepare_return_series(raw, timestamps or None)
    min_len = min(prepared[sid].size for sid in ids)
    columns = []
    for sid in ids:
        values = prepared[sid][-min_len:]
        columns.append(
            values
            if bool(net_flags.get(sid, False))
            else apply_cost_drag(
                values,
                turnover=float(turnovers.get(sid) or 0.0),
                regime=REFERENCE_COST_REGIME_20BPS,
                participation=DEFAULT_PARTICIPATION,
            )
        )
    matrix = np.column_stack(columns)
    return ids, matrix


def _weights_for(method: str, ids: list[str], matrix: np.ndarray, *, cap: float) -> np.ndarray:
    n = len(ids)
    cov, _ = ledoit_wolf_shrunk_covariance(matrix)
    if method == "equal_weight":
        return np.full(n, 1.0 / n)
    if method == "inverse_vol":
        vol = np.sqrt(np.clip(np.diag(cov), 1e-18, None))
        inv = 1.0 / vol
        return inv / inv.sum()
    if method == "erc":
        return erc_weights(cov)
    if method == "hrp_threshold":
        return hrp_weights_from_returns(ids, matrix)
    if method == "hrp_dendrogram":
        return hier.hrp_dendrogram_weights(cov)
    if method == "constrained_hrp":
        return hier.hrp_dendrogram_weights(cov, bounds={"lower": 0.0, "upper": cap})
    if method == "herc":
        return hier.herc_weights(cov)
    if method == "nco":
        return hier.nco_weights(cov)
    if method == "wasserstein_dro":
        # radius is in squared-return units; 1e-5 ~ (0.3% daily)^2 ambiguity.
        return hier.wasserstein_dro_weights(cov, radius=1e-5)
    if method == "graph_inverse_centrality":
        return hier.graph_inverse_centrality_weights(cov)
    raise ValueError(method)


def _diagnostics(weights: np.ndarray, cov: np.ndarray) -> dict[str, float]:
    sigma = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    port_var = float(weights @ cov @ weights)
    return {
        "eff_n": float(1.0 / max(1e-18, float(weights @ weights))),
        "div_ratio": float((weights @ sigma) / max(1e-18, np.sqrt(port_var))),
        "max_w": float(weights.max()),
    }


def compare(ids: list[str], matrix: np.ndarray, *, cap: float = 0.30) -> dict[str, Any]:
    cov, _ = ledoit_wolf_shrunk_covariance(matrix)
    out: dict[str, Any] = {}
    for method in METHODS:
        weights = _weights_for(method, ids, matrix, cap=cap)
        out[method] = {
            "weights": {sid: round(float(w), 6) for sid, w in zip(ids, weights)},
            **_diagnostics(weights, cov),
        }
    return out


def run_cell_variants(payload: dict[str, Any]) -> dict[str, Any]:
    """Execute every ``allocator_variants`` row of a materialized cell spec.

    Each variant is routed through :func:`allocate_quality_gated` (same quality
    gate, ``method`` + ``allocator_params``) so the pre-registered comparison set
    is actually RUN, not merely declared.  Requires materialized sleeve returns.
    """
    from lumina_quant.portfolio.quality_gated_allocation import allocate_quality_gated

    sleeves = payload.get("sleeves") or {}
    _validate_materialized_data_contract(sleeves)
    returns = {sid: (spec or {}).get("returns") for sid, spec in sleeves.items()}
    turnovers = {sid: (spec or {}).get("turnover") or 0.0 for sid, spec in sleeves.items()}
    families = {
        sid: str((spec or {}).get("family"))
        for sid, spec in sleeves.items()
        if (spec or {}).get("family") is not None
    }
    allocator = payload.get("allocator") if isinstance(payload.get("allocator"), dict) else {}

    def setting(name: str, default: Any) -> Any:
        return payload[name] if name in payload else allocator.get(name, default)

    returns_are_net = {
        sid: bool((spec or {}).get("returns_are_net", False)) for sid, spec in sleeves.items()
    }
    return_timestamps = {
        sid: (spec or {}).get("return_timestamps")
        for sid, spec in sleeves.items()
        if "return_timestamps" in (spec or {})
    }
    out: dict[str, Any] = {}
    for index, variant in enumerate(payload.get("allocator_variants") or []):
        method = str(variant.get("method") or payload.get("method") or "erc")
        params = dict(variant.get("allocator_params") or {})
        label = f"{index:02d}_{method}"
        try:
            weights = allocate_quality_gated(
                returns,
                turnovers,
                method=method,
                upper=setting("upper", None),
                min_sleeves=int(setting("min_sleeves", 1)),
                turnover_penalty_lambda=float(setting("turnover_penalty_lambda", 0.0)),
                correlation_shrinkage=setting("correlation_shrinkage", None),
                families=families or None,
                family_momentum_window=int(setting("family_momentum_window", 0)),
                family_momentum_tilt_strength=float(setting("family_momentum_tilt_strength", 0.5)),
                family_momentum_tilt_cap=float(setting("family_momentum_tilt_cap", 0.30)),
                min_families=int(setting("min_families", 3)),
                allocator_params=params or None,
                returns_are_net=returns_are_net,
                return_timestamps=return_timestamps or None,
            )
        except (ValueError, RuntimeError) as exc:  # fail-closed variant, keep comparing rows
            out[label] = {"method": method, "allocator_params": params, "error": str(exc)}
            continue
        out[label] = {"method": method, "allocator_params": params, "weights": weights}
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--cap", type=float, default=0.30, help="per-name cap for constrained_hrp")
    parser.add_argument(
        "--variants",
        action="store_true",
        help="also execute every allocator_variants row of the input cell spec",
    )
    args = parser.parse_args(argv)
    ids, matrix = _load_returns(args.input)
    result: dict[str, Any] = compare(ids, matrix, cap=float(args.cap))
    if args.variants:
        payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
        result = {"methods": result, "variants": run_cell_variants(payload)}
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
