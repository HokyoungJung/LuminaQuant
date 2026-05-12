"""Durable factor-card metadata for Alpha Zoo research artifacts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from .crypto_fx_factors import CALENDAR_FIELD_NAMES, FactorSpec


@dataclass(frozen=True, slots=True)
class FactorCard:
    factor: str
    family: str
    market: str
    description: str
    inputs: tuple[str, ...]
    strategy_validity: Mapping[str, Any]
    selection_provenance: Mapping[str, Any]
    metrics: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _calendar_rejections(fields: Sequence[str]) -> list[str]:
    found = sorted(set(fields) & CALENDAR_FIELD_NAMES)
    return [f"calendar_entry_field_forbidden:{field}" for field in found]


def build_factor_card(
    spec: FactorSpec,
    *,
    metrics: Mapping[str, Any] | None = None,
    selected_using_splits: Sequence[str] = ("train", "validation"),
    uses_locked_oos_for_selection: bool = False,
    source_refs: Sequence[str] = (),
) -> FactorCard:
    """Build fail-closed factor metadata for research/promotion gates."""
    selected_splits = tuple(str(item) for item in selected_using_splits)
    rejection_reasons = _calendar_rejections(spec.calendar_fields)
    if uses_locked_oos_for_selection or "locked_oos" in selected_splits or "oos" in selected_splits:
        rejection_reasons.append("locked_oos_used_for_selection")
    if not source_refs:
        rejection_reasons.append("source_refs_missing")
    validity = {
        "pass": not rejection_reasons,
        "calendar_primary": False,
        "calendar_fields": tuple(spec.calendar_fields),
        "causal_state_only": True,
        "lookahead_safe": True,
        "primary_signal_type": "formulaic_state_factor",
        "rejection_reasons": rejection_reasons,
        "source_refs": tuple(source_refs),
    }
    provenance = {
        "selected_using_splits": selected_splits,
        "uses_locked_oos_for_selection": bool(uses_locked_oos_for_selection),
        "locked_oos_role": "gate_report_only",
        "selection_policy": "train_validation_only",
    }
    return FactorCard(
        factor=spec.name,
        family=spec.family,
        market=spec.market,
        description=spec.description,
        inputs=tuple(spec.inputs),
        strategy_validity=validity,
        selection_provenance=provenance,
        metrics=dict(metrics or {}),
    )
