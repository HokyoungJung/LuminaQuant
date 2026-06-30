from __future__ import annotations

import pandas as pd

from lumina_quant.alpha_zoo.crypto_fx_factors import build_crypto_fx_factor_specs
from lumina_quant.alpha_zoo.evidence import (
    AlphaEvidenceThresholds,
    alpha_benchmark_evidence,
    attach_evidence_to_screen_payload,
    build_alpha_descriptor,
    rank_ic_series,
)


def _factor_panel(periods: int = 10) -> pd.DataFrame:
    rows = []
    splits = ["train"] * 4 + ["validation"] * 4 + ["locked_oos"] * 2
    for idx, split in enumerate(splits[:periods]):
        timestamp = pd.Timestamp("2026-01-01") + pd.Timedelta(hours=idx)
        for rank, symbol in enumerate(("A", "B", "C", "D"), start=1):
            # Perfect positive cross-sectional relationship each period.
            rows.append(
                {
                    "timestamp": timestamp,
                    "symbol": symbol,
                    "split": split,
                    "good_factor": float(rank),
                    "bad_factor": float(5 - rank),
                    "forward_return": float(rank) / 100.0,
                }
            )
    return pd.DataFrame(rows)


def test_rank_ic_series_and_classification_use_train_validation_only() -> None:
    frame = _factor_panel()
    ic = rank_ic_series(frame[frame["split"].eq("train")], factor="good_factor")
    assert ic["rank_ic"].tolist() == [1.0, 1.0, 1.0, 1.0]

    evidence = alpha_benchmark_evidence(
        frame,
        factor="good_factor",
        thresholds=AlphaEvidenceThresholds(min_periods=4, min_abs_t_stat=2.0),
    )

    assert evidence["classification"] == "alive"
    assert evidence["pass"] is True
    assert evidence["selected_using_splits"] == ["train", "validation"]
    assert evidence["uses_locked_oos_for_selection"] is False
    assert evidence["split_evidence"]["locked_oos"]["n_periods"] == 2
    assert "locked_oos_report_only_not_used_for_selection" in evidence["rejection_reasons"]


def test_reversed_alpha_is_detected_without_oos_leakage() -> None:
    evidence = alpha_benchmark_evidence(
        _factor_panel(),
        factor="bad_factor",
        thresholds=AlphaEvidenceThresholds(min_periods=4, min_abs_t_stat=2.0),
    )

    assert evidence["classification"] == "reversed"
    assert evidence["pass"] is True
    assert evidence["requires_inversion"] is True
    assert evidence["signal_direction"] == "invert"
    assert evidence["uses_locked_oos_for_selection"] is False
    assert "candidate_signal_direction_reversed" in evidence["rejection_reasons"]


def test_missing_split_column_fails_closed_instead_of_using_all_rows() -> None:
    frame = _factor_panel().drop(columns=["split"])

    evidence = alpha_benchmark_evidence(
        frame,
        factor="good_factor",
        thresholds=AlphaEvidenceThresholds(min_periods=4, min_abs_t_stat=2.0),
    )

    assert evidence["classification"] == "insufficient"
    assert evidence["pass"] is False
    assert evidence["selected_using_splits"] == ["train", "validation"]
    assert evidence["available_splits"] == []
    assert "split_column_missing_fail_closed" in evidence["rejection_reasons"]


def test_locked_oos_selection_split_overlap_fails_closed_without_leakage() -> None:
    evidence = alpha_benchmark_evidence(
        _factor_panel(),
        factor="good_factor",
        selection_splits=("train", "locked_oos"),
        thresholds=AlphaEvidenceThresholds(min_periods=4, min_abs_t_stat=2.0),
    )

    assert evidence["classification"] == "insufficient"
    assert evidence["pass"] is False
    assert evidence["uses_locked_oos_for_selection"] is True
    assert evidence["forbidden_selection_splits"] == ["locked_oos"]
    assert evidence["selected_using_splits"] == ["train"]
    assert any(
        reason.startswith("locked_oos_selection_splits_forbidden")
        for reason in evidence["rejection_reasons"]
    )


def test_alpha_descriptor_requires_source_refs_and_hashes_payload() -> None:
    spec = build_crypto_fx_factor_specs()[0]
    descriptor = build_alpha_descriptor(spec, source_refs=("unit-test",))
    payload = descriptor.to_dict()

    assert payload["artifact_kind"] == "alpha_descriptor"
    assert payload["name"] == spec.name
    assert len(payload["descriptor_hash"]) == 16


def test_attach_evidence_to_screen_payload_enriches_selected_factors() -> None:
    payload = attach_evidence_to_screen_payload(
        {"selected_factors": [{"factor": "good_factor"}]},
        _factor_panel(),
    )

    assert payload["evidence_policy"] == "train_validation_only_locked_oos_report_only"
    assert payload["selected_factors"][0]["benchmark_evidence"]["classification"] == "alive"
