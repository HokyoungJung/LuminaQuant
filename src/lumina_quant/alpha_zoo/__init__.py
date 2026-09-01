"""Crypto/FX formulaic alpha-zoo research primitives."""

from .crypto_fx_factors import (
    FactorSpec,
    assign_time_splits,
    build_crypto_fx_factor_specs,
    compute_factor_frame,
    factor_columns,
    screen_factor_frame,
)
from .evidence import (
    AlphaDescriptor,
    AlphaEvidenceThresholds,
    SplitEvidence,
    alpha_benchmark_evidence,
    attach_evidence_to_screen_payload,
    build_alpha_descriptor,
    rank_ic_series,
    summarize_split_evidence,
)
from .factor_card import FactorCard, build_factor_card

__all__ = [
    "AlphaDescriptor",
    "AlphaEvidenceThresholds",
    "FactorCard",
    "FactorSpec",
    "SplitEvidence",
    "alpha_benchmark_evidence",
    "assign_time_splits",
    "attach_evidence_to_screen_payload",
    "build_alpha_descriptor",
    "build_crypto_fx_factor_specs",
    "build_factor_card",
    "compute_factor_frame",
    "factor_columns",
    "rank_ic_series",
    "screen_factor_frame",
    "summarize_split_evidence",
]
