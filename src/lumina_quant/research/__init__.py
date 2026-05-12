"""Research-only outcome labeling and calibration utilities."""

from .edge_calibration import BucketCalibration, CalibrationDecision, calibrate_edge_buckets
from .triple_barrier import BarrierType, TripleBarrierOutcome, label_triple_barrier

__all__ = [
    "BarrierType",
    "BucketCalibration",
    "CalibrationDecision",
    "TripleBarrierOutcome",
    "calibrate_edge_buckets",
    "label_triple_barrier",
]
