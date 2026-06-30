"""Research-only outcome labeling and calibration utilities."""

from .edge_calibration import BucketCalibration, CalibrationDecision, calibrate_edge_buckets
from .run_card import (
    RunCard,
    RunCardRealityGateError,
    assert_reality_gates_pass,
    build_reality_gates,
    build_research_run_card,
    stable_payload_hash,
    write_run_card,
)
from .triple_barrier import BarrierType, TripleBarrierOutcome, label_triple_barrier

__all__ = [
    "BarrierType",
    "BucketCalibration",
    "CalibrationDecision",
    "RunCard",
    "RunCardRealityGateError",
    "TripleBarrierOutcome",
    "assert_reality_gates_pass",
    "build_reality_gates",
    "build_research_run_card",
    "calibrate_edge_buckets",
    "label_triple_barrier",
    "stable_payload_hash",
    "write_run_card",
]
