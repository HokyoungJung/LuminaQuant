"""Research-only outcome labeling and calibration utilities."""

from .alpha_candidate_ledger import AlphaCandidateRecord, ResearchLedger
from .alpha_search import (
    AlphaSearchCandidate,
    AlphaSearchResult,
    SearchPanel,
    build_synthetic_search_panel,
    generate_candidate_trees,
    run_alpha_search,
)
from .edge_calibration import BucketCalibration, CalibrationDecision, calibrate_edge_buckets
from .effective_trials import (
    candidate_return_correlation_matrix,
    effective_number_of_trials,
    raw_number_of_trials,
)
from .execution_attribution import (
    AttributionCostModel,
    ExecutionAttribution,
    ExecutionAttributionReport,
    ExecutionBiasSeverity,
    FillEvent,
    RoundTrip,
    attribute_execution_delta,
    pair_round_trips_fifo,
    run_execution_attribution,
)
from .run_card import (
    RunCard,
    RunCardRealityGateError,
    assert_reality_gates_pass,
    build_reality_gates,
    build_research_run_card,
    stable_payload_hash,
    write_run_card,
)
from .survivorship import (
    SurvivorshipVerdict,
    empirical_variance_across_trials,
    evaluate_survivorship_gate,
)
from .triple_barrier import BarrierType, TripleBarrierOutcome, label_triple_barrier

__all__ = [
    "AlphaCandidateRecord",
    "AlphaSearchCandidate",
    "AlphaSearchResult",
    "AttributionCostModel",
    "BarrierType",
    "BucketCalibration",
    "CalibrationDecision",
    "ExecutionAttribution",
    "ExecutionAttributionReport",
    "ExecutionBiasSeverity",
    "FillEvent",
    "RoundTrip",
    "ResearchLedger",
    "SearchPanel",
    "RunCard",
    "RunCardRealityGateError",
    "SurvivorshipVerdict",
    "TripleBarrierOutcome",
    "assert_reality_gates_pass",
    "attribute_execution_delta",
    "build_reality_gates",
    "build_synthetic_search_panel",
    "build_research_run_card",
    "calibrate_edge_buckets",
    "candidate_return_correlation_matrix",
    "effective_number_of_trials",
    "empirical_variance_across_trials",
    "evaluate_survivorship_gate",
    "generate_candidate_trees",
    "label_triple_barrier",
    "pair_round_trips_fifo",
    "raw_number_of_trials",
    "run_alpha_search",
    "run_execution_attribution",
    "stable_payload_hash",
    "write_run_card",
]
