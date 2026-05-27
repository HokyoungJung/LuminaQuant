from __future__ import annotations

import importlib.util
from pathlib import Path

from lumina_quant.research.candidate_outcome_ledger import (
    CandidateOutcomeLedger,
    CandidateOutcomeRecord,
)
from lumina_quant.research.edge_calibration import calibrate_edge_buckets

_CALIBRATION_SPEC = importlib.util.spec_from_file_location(
    "calibrate_crypto_fx_edges", Path("scripts/research/calibrate_crypto_fx_edges.py")
)
_CALIBRATION_MODULE = importlib.util.module_from_spec(_CALIBRATION_SPEC)
assert _CALIBRATION_SPEC.loader is not None
_CALIBRATION_SPEC.loader.exec_module(_CALIBRATION_MODULE)
build_calibration_payload = _CALIBRATION_MODULE.build_calibration_payload


def test_calibration_allows_positive_bucket_after_shrinkage() -> None:
    records = [
        {
            "candidate_id": "alpha",
            "side": "LONG",
            "symbol": "ETH/USDT",
            "net_pnl_bps": 20.0 + (idx % 3),
        }
        for idx in range(40)
    ]
    result = calibrate_edge_buckets(
        records,
        bucket_fields=("candidate_id", "side", "symbol"),
        parent_fields=("candidate_id", "side"),
        min_bucket_n=30,
        min_lower_edge_bps=0.0,
    )
    calibration = result[("alpha", "LONG", "ETH/USDT")]
    assert calibration.decision.allowed is True
    assert calibration.lower_confidence_edge_bps > 0.0


def test_calibration_blocks_negative_lower_confidence_edge() -> None:
    records = [
        {
            "candidate_id": "alpha",
            "side": "LONG",
            "symbol": "ETH/USDT",
            "net_pnl_bps": -5.0 + (idx % 2),
        }
        for idx in range(40)
    ]
    result = calibrate_edge_buckets(
        records, bucket_fields=("candidate_id", "side", "symbol"), min_bucket_n=10
    )
    calibration = result[("alpha", "LONG", "ETH/USDT")]
    assert calibration.decision.allowed is False
    assert calibration.decision.reason == "lower_confidence_edge_not_positive"


def test_outcome_ledger_roundtrip_and_summary(tmp_path) -> None:
    path = tmp_path / "ledger.jsonl"
    ledger = CandidateOutcomeLedger(path)
    ledger.append(
        CandidateOutcomeRecord(
            candidate_id="alpha",
            split="train",
            symbol="ETH/USDT",
            side="LONG",
            entry_time="2026-01-01T00:00:00Z",
            exit_time="2026-01-01T04:00:00Z",
            barrier_type="take_profit",
            net_pnl_bps=12.5,
            mae_bps=-4.0,
            mfe_bps=20.0,
        )
    )
    ledger.append(
        {
            "candidate_id": "alpha",
            "split": "locked_oos",
            "symbol": "ETH/USDT",
            "side": "LONG",
            "net_pnl_bps": 1.0,
        }
    )
    rows = ledger.read_all()
    assert len(rows) == 2
    summary = ledger.summary()
    assert summary["train_validation_record_count"] == 1
    assert summary["locked_oos_record_count"] == 1
    assert summary["uses_locked_oos_for_selection"] is False


def test_calibration_payload_physically_excludes_locked_oos_records() -> None:
    records = (
        [
            {
                "candidate_id": "alpha",
                "split": "train",
                "side": "LONG",
                "symbol": "ETH/USDT",
                "net_pnl_bps": 20.0,
            }
            for _ in range(20)
        ]
        + [
            {
                "candidate_id": "alpha",
                "split": "validation",
                "side": "LONG",
                "symbol": "ETH/USDT",
                "net_pnl_bps": 18.0,
            }
            for _ in range(20)
        ]
        + [
            {
                "candidate_id": "alpha",
                "split": "locked_oos",
                "side": "LONG",
                "symbol": "ETH/USDT",
                "net_pnl_bps": -9999.0,
            }
            for _ in range(10)
        ]
    )
    payload = build_calibration_payload(
        records,
        ledger_summary={"record_count": len(records)},
        bucket_fields=("candidate_id", "side", "symbol"),
        parent_fields=("candidate_id", "side"),
        min_bucket_n=10,
    )
    assert payload["input_locked_oos_record_count"] == 10
    assert payload["calibration_record_count"] == 40
    assert payload["locked_oos_calibration_record_count"] == 0
    calibration = payload["calibrations"]["alpha|LONG|ETH/USDT"]
    assert calibration["mean_edge_bps"] > 0.0
    assert payload["uses_locked_oos_for_calibration"] is False
