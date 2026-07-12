from __future__ import annotations

import hashlib
import inspect
import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest

import lumina_quant.research.alpha_max_evidence as alpha_max_evidence
import lumina_quant.research.alpha_max_engine_runner as alpha_max_runner

from lumina_quant.research.alpha_max_engine_runner import (
    ALPHA_MAX_COST_CELL_BPS,
    AlphaMaxAttributionCollector,
    AlphaMaxRuntimeContractError,
    construct_alpha_max_engine,
    create_alpha_max_historical_package,
    create_alpha_max_prelock_bundle,
    orchestrate_alpha_max_status_matrix,
    preflight_alpha_max_runtime_contract,
    seal_alpha_max_manifest_activation,
    validate_alpha_max_engine_activation,
)
from lumina_quant.research.alpha_max_evidence import (
    ALPHA_MAX_CANDIDATE_SYMBOLS,
    AlphaMaxActualEngineRunReceipt,
    AlphaMaxCapsuleReceipt,
    AlphaMaxCostCellEvidence,
    AlphaMaxCostCellPreGateEvidence,
    AlphaMaxFoldRunEvidence,
    AlphaMaxFundingBoundaryResolver,
    AlphaMaxGateInput,
    AlphaMaxManifestReceipt,
    AlphaMaxOrderedFundingLookup,
    AlphaMaxRunReportOnlyDiagnostics,
    AlphaMaxRootSeal,
    AlphaMaxStreamingEquityTracker,
    FeatureRootSpec,
    materialize_alpha_max_manifest,
    select_alpha_max_prelock_champion,
    validate_alpha_max_train_liquidity_buckets,
)
from lumina_quant.backtesting.backtest import Backtest
from lumina_quant.backtesting.data_windowed_parquet import HistoricParquetWindowedDataHandler
from lumina_quant.backtesting.execution_sim import SimulatedExecutionHandler
from lumina_quant.backtesting.portfolio_backtest import Portfolio
from lumina_quant.core.events import OrderEvent, SignalEvent
from lumina_quant.strategies.artifact_portfolio_mode import ArtifactPortfolioModeStrategy


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = (
    REPO_ROOT / "configs/research/alpha_max_portfolio_20260711_listing_aware.json"
).resolve()


@pytest.fixture(autouse=True)
def _clean_lq_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in tuple(os.environ):
        if key.startswith("LQ_"):
            monkeypatch.delenv(key, raising=False)


def _nodes() -> list[dict[str, object]]:
    payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    return payload["current_trial_registry"]["nodes"]


def _owned_root(tmp_path: Path) -> Path:
    root = (tmp_path / "run").resolve()
    (root / "manifests/validation_train_fit").mkdir(parents=True)
    (root / "manifests/prelock_final_refit").mkdir()
    return root


def test_fitted_fixed_weights_use_the_manifest_canonical_precision() -> None:
    preflight = preflight_alpha_max_runtime_contract(str(CONFIG_PATH))
    nodes = alpha_max_runner._alpha_max_current_nodes(preflight)
    observation_count = 252
    component_returns = MappingProxyType(
        {
            "component_carry_1x": tuple(
                0.001 + (index % 7) * 0.0001 for index in range(observation_count)
            ),
            "component_near_high_1x": tuple(
                -0.0005 + (index % 11) * 0.00015 for index in range(observation_count)
            ),
            "component_trend_1x": tuple(
                0.0002 + ((index * 3) % 13) * 0.00012 for index in range(observation_count)
            ),
        }
    )

    fit = alpha_max_runner._alpha_max_fit_weights(
        nodes,
        phase="train",
        calendar=tuple(f"day-{index:03d}" for index in range(observation_count)),
        component_returns=component_returns,
    )

    assert fit.weights_by_row["full_equal_weight_1x"] == {
        "component_carry_1x": 0.3333333333,
        "component_near_high_1x": 0.3333333333,
        "component_trend_1x": 0.3333333333,
    }


def test_status_matrix_has_84_statuses_68_logical_cells_and_816_fold_receipts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, int]] = []
    manifests: dict[str, AlphaMaxManifestReceipt] = {}
    capsules: dict[tuple[str, str], AlphaMaxCapsuleReceipt] = {}

    monkeypatch.setattr(
        "lumina_quant.research.alpha_max_engine_runner.canonical_alpha_max_cost_cell_bytes",
        lambda _value: b"typed-cell\n",
    )

    def bare(cls, **fields):
        value = object.__new__(cls)
        for name, field_value in fields.items():
            object.__setattr__(value, name, field_value)
        return value

    def execute(row, nominal_cost_bps):
        row_id = row["row_id"]
        calls.append((row_id, nominal_cost_bps))
        manifest = manifests.setdefault(
            row_id,
            bare(
                AlphaMaxManifestReceipt,
                sha256=f"{abs(hash(('manifest', row_id))):064x}"[-64:],
            ),
        )
        fold_runs = []
        for index in range(1, 13):
            fold_id = f"validation_w{index:02d}"
            capsule = capsules.setdefault(
                (row_id, fold_id),
                bare(
                    AlphaMaxCapsuleReceipt,
                    prefix_id=fold_id,
                    sha256=f"{abs(hash((row_id, fold_id))):064x}"[-64:],
                ),
            )
            actual = bare(
                AlphaMaxActualEngineRunReceipt,
                split_or_fold_id=fold_id,
                capsule_receipt=capsule,
                manifest_receipt=manifest,
            )
            fold_runs.append(
                bare(
                    AlphaMaxFoldRunEvidence,
                    actual_engine_run=actual,
                )
            )
        pre_gate = bare(
            AlphaMaxCostCellPreGateEvidence,
            fold_runs=tuple(fold_runs),
        )
        return bare(
            AlphaMaxCostCellEvidence,
            row_id=row_id,
            domain="validation",
            nominal_cost_bps=nominal_cost_bps,
            status="complete",
            evidence_tier="actual_engine",
            selection_valid=True,
            pre_gate_evidence=pre_gate,
        )

    result = orchestrate_alpha_max_status_matrix(_nodes(), execute)

    assert len(result.statuses) == 21 * 4
    assert result.engine_cell_count == 17 * 4
    assert len(calls) == 17 * 4
    assert (
        sum(
            len(status.evidence.pre_gate_evidence.fold_runs)
            for status in result.statuses
            if status.engine_constructed
        )
        == 816
    )
    assert {cost for _, cost in calls} == set(ALPHA_MAX_COST_CELL_BPS)
    assert not any(row_id.startswith("incumbent_") for row_id, _ in calls)
    assert not any(row_id.startswith("diagnostic_") for row_id, _ in calls)


def test_production_physical_fold_schedules_are_exact() -> None:
    validation = alpha_max_runner._alpha_max_physical_fold_schedule("validation")
    historical = alpha_max_runner._alpha_max_physical_fold_schedule("historical_exposed_evaluation")

    assert len(validation) == 816
    assert len(historical) == 680
    assert len(set(validation)) == 816
    assert len(set(historical)) == 680
    assert all(
        [fold_id for row, cost, fold_id in validation if row == row_id and cost == nominal]
        == [f"validation_w{index:02d}" for index in range(1, 13)]
        for row_id in alpha_max_runner._ALPHA_MAX_RESOLVABLE_ROWS
        for nominal in ALPHA_MAX_COST_CELL_BPS
    )
    alpha_max_runner._validate_alpha_max_physical_fold_schedule(
        validation,
        domain="validation",
    )
    alpha_max_runner._validate_alpha_max_physical_fold_schedule(
        historical,
        domain="historical_exposed_evaluation",
    )
    for invalid, domain in (
        (validation[:-1], "validation"),
        ((*validation, validation[-1]), "validation"),
        (historical[:-1], "historical_exposed_evaluation"),
        ((*historical, historical[-1]), "historical_exposed_evaluation"),
    ):
        with pytest.raises(
            AlphaMaxRuntimeContractError,
            match="physical_fold_cardinality_mismatch",
        ):
            alpha_max_runner._validate_alpha_max_physical_fold_schedule(
                invalid,
                domain=domain,
            )
    assert all(
        len([fold_id for row, cost, fold_id in historical if row == row_id and cost == nominal])
        == 10
        for row_id in alpha_max_runner._ALPHA_MAX_RESOLVABLE_ROWS
        for nominal in ALPHA_MAX_COST_CELL_BPS
    )


def test_adjacent_feature_root_gap_is_closed_across_split() -> None:
    boundary = datetime(2025, 1, 1, tzinfo=UTC)
    availability_start = MappingProxyType(
        dict.fromkeys(
            alpha_max_runner.ALPHA_MAX_CANDIDATE_SYMBOLS,
            boundary.replace(year=2024),
        )
    )
    availability_end = MappingProxyType(
        dict.fromkeys(
            alpha_max_runner.ALPHA_MAX_CANDIDATE_SYMBOLS,
            boundary.replace(year=2026),
        )
    )

    def seal(root_id: str, *, first: int, last: int, contracted: bool = True):
        entries = tuple(
            SimpleNamespace(
                relative_path=f"symbol={symbol}/date=2025-01-01/part.parquet",
                minimum_timestamp_ms=first,
                maximum_timestamp_ms=last,
            )
            for symbol in alpha_max_runner.ALPHA_MAX_CANDIDATE_SYMBOLS
        )
        value = SimpleNamespace(
            root_id=root_id,
            exchange="binance",
            entries=entries,
            start_utc=boundary if root_id == "right" else boundary.replace(year=2024),
            end_utc=boundary.replace(year=2026) if root_id == "right" else boundary,
        )
        if contracted:
            value.availability_start_by_symbol = availability_start
            value.availability_end_by_symbol = availability_end
        return value

    uncontracted = {
        ("left", "feature"): seal("left", first=1, last=10_000, contracted=False),
        ("right", "feature"): seal(
            "right",
            first=10_000 + 28_801_000,
            last=40_000_000,
            contracted=False,
        ),
    }
    with pytest.raises(
        AlphaMaxRuntimeContractError,
        match="adjacent_feature_root_funding_coverage_incomplete",
    ):
        alpha_max_runner._validate_alpha_max_adjacent_feature_roots(
            uncontracted,
            (("left", "right"),),
        )

    predecessor = seal("left", first=1, last=10_000)
    current = seal("right", first=10_000 + 28_801_000, last=40_000_000)
    seals = {("left", "feature"): predecessor, ("right", "feature"): current}
    alpha_max_runner._validate_alpha_max_adjacent_feature_roots(
        seals,
        (("left", "right"),),
    )

    current.entries = tuple(
        SimpleNamespace(
            relative_path=entry.relative_path,
            minimum_timestamp_ms=10_000 + 28_801_001,
            maximum_timestamp_ms=entry.maximum_timestamp_ms,
        )
        for entry in current.entries
    )
    with pytest.raises(
        AlphaMaxRuntimeContractError,
        match="adjacent_feature_root_funding_coverage_incomplete",
    ):
        alpha_max_runner._validate_alpha_max_adjacent_feature_roots(
            seals,
            (("left", "right"),),
        )


def test_adjacent_feature_roots_accept_only_a_fresh_declared_availability_end() -> None:
    predecessor_start = datetime(2026, 6, 1, tzinfo=UTC)
    boundary = datetime(2026, 6, 24, tzinfo=UTC)
    current_end = datetime(2026, 7, 1, tzinfo=UTC)
    ton_end = datetime(2026, 6, 23, 9, tzinfo=UTC)
    ton_last_ms = int(datetime(2026, 6, 23, 8, tzinfo=UTC).timestamp() * 1000)
    boundary_ms = int(boundary.timestamp() * 1000)
    availability_start = {
        symbol: (
            datetime(2024, 3, 1, 16, tzinfo=UTC)
            if symbol == "TONUSDT"
            else datetime(2022, 12, 31, tzinfo=UTC)
        )
        for symbol in alpha_max_runner.ALPHA_MAX_CANDIDATE_SYMBOLS
    }
    availability_end = {
        symbol: ton_end if symbol == "TONUSDT" else current_end
        for symbol in alpha_max_runner.ALPHA_MAX_CANDIDATE_SYMBOLS
    }

    def entries(*, predecessor: bool, stale_ton: bool = False) -> tuple[SimpleNamespace, ...]:
        rows: list[SimpleNamespace] = []
        for symbol in alpha_max_runner.ALPHA_MAX_CANDIDATE_SYMBOLS:
            if symbol == "TONUSDT" and not predecessor:
                continue
            timestamp_ms = (
                ton_last_ms - (28_801_001 if stale_ton else 0)
                if symbol == "TONUSDT"
                else boundary_ms - 1
                if predecessor
                else boundary_ms
            )
            rows.append(
                SimpleNamespace(
                    relative_path=f"symbol={symbol}/date=2026-06-23/part.parquet",
                    minimum_timestamp_ms=timestamp_ms,
                    maximum_timestamp_ms=timestamp_ms,
                )
            )
        return tuple(rows)

    predecessor = SimpleNamespace(
        exchange="binance",
        entries=entries(predecessor=True),
        start_utc=predecessor_start,
        end_utc=boundary,
        availability_start_by_symbol=availability_start,
        availability_end_by_symbol=availability_end,
    )
    current = SimpleNamespace(
        exchange="binance",
        entries=entries(predecessor=False),
        start_utc=boundary,
        end_utc=current_end,
        availability_start_by_symbol=availability_start,
        availability_end_by_symbol=availability_end,
    )
    seals = {("left", "feature"): predecessor, ("right", "feature"): current}
    alpha_max_runner._validate_alpha_max_adjacent_feature_roots(
        seals,
        (("left", "right"),),
    )

    predecessor.entries = entries(predecessor=True, stale_ton=True)
    with pytest.raises(
        AlphaMaxRuntimeContractError,
        match="adjacent_feature_root_funding_coverage_incomplete",
    ):
        alpha_max_runner._validate_alpha_max_adjacent_feature_roots(
            seals,
            (("left", "right"),),
        )


def test_runtime_read_audit_keeps_different_daily_paths_in_chronological_order() -> None:
    first = ("START_DATE", "END_DATE", "SYMBOLS")
    second = ("START_DATE", "SLIPPAGE_RATE", "END_DATE")

    retained, first_sha = alpha_max_runner._alpha_max_append_runtime_read_audit(
        None,
        first,
    )
    combined, combined_sha = alpha_max_runner._alpha_max_append_runtime_read_audit(
        retained,
        second,
    )

    assert retained == first
    assert combined == (*first, *second)
    assert first_sha != combined_sha
    assert combined_sha == alpha_max_runner._sha256(
        alpha_max_runner._canonical_bytes(list(combined))
    )
    with pytest.raises(
        AlphaMaxRuntimeContractError,
        match="runtime_read_audit_invalid",
    ):
        alpha_max_runner._alpha_max_append_runtime_read_audit(
            combined,
            ("UNSEALED_FIELD",),
        )


def test_run_owned_root_writes_sealed_last_and_never_seals_mismatched_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = (tmp_path / "sealed-last").resolve()
    root = alpha_max_runner._create_alpha_max_run_owned_root(output)
    writes: list[str] = []
    original_write = alpha_max_runner._write_bundle_file

    def recording_write(bundle_root, relative_path, payload):
        writes.append(relative_path)
        return original_write(bundle_root, relative_path, payload)

    monkeypatch.setattr(alpha_max_runner, "_write_bundle_file", recording_write)
    bundle = alpha_max_runner._finalize_alpha_max_run_owned_root(
        root,
        {"report/result.json": b"{}\n"},
        seal_bytes=b'{"sealed":true}\n',
    )
    assert writes[-1] == "SEALED.json"
    assert Path(bundle.seal_path).read_bytes() == b'{"sealed":true}\n'

    mismatch_root = alpha_max_runner._create_alpha_max_run_owned_root(
        (tmp_path / "mismatch").resolve()
    )
    original_write(mismatch_root, "report/result.json", b"first\n")
    with pytest.raises(
        AlphaMaxRuntimeContractError,
        match="run_owned_artifact_mismatch",
    ):
        alpha_max_runner._finalize_alpha_max_run_owned_root(
            mismatch_root,
            {"report/result.json": b"second\n"},
            seal_bytes=b'{"sealed":true}\n',
        )
    assert not (mismatch_root / "SEALED.json").exists()
    alpha_max_runner._cleanup_partial_bundle(mismatch_root)
    assert not mismatch_root.exists()


def test_bundle_immutability_and_cleanup_never_follow_external_symlinks(
    tmp_path: Path,
) -> None:
    external = (tmp_path / "external").resolve()
    external.mkdir(mode=0o700)
    external_file = external / "evidence.txt"
    external_file.write_bytes(b"must remain untouched\n")
    external_file.chmod(0o640)
    external_mode = external.stat().st_mode & 0o777
    external_file_mode = external_file.stat().st_mode & 0o777

    root = (tmp_path / "bundle").resolve()
    root.mkdir(mode=0o700)
    (root / "artifact.json").write_bytes(b"{}\n")
    (root / "escape").symlink_to(external, target_is_directory=True)

    with pytest.raises(AlphaMaxRuntimeContractError, match="bundle_tree_invalid"):
        alpha_max_runner._make_bundle_immutable(root)
    assert external.stat().st_mode & 0o777 == external_mode
    assert external_file.stat().st_mode & 0o777 == external_file_mode
    assert external_file.read_bytes() == b"must remain untouched\n"

    alpha_max_runner._cleanup_partial_bundle(root)
    assert not root.exists()
    assert external.stat().st_mode & 0o777 == external_mode
    assert external_file.stat().st_mode & 0o777 == external_file_mode
    assert external_file.read_bytes() == b"must remain untouched\n"


def test_cleanup_and_claim_release_unlink_hostile_links_without_chmodding_target(
    tmp_path: Path,
) -> None:
    external = (tmp_path / "external-claim.txt").resolve()
    external.write_bytes(b"external claim target\n")
    external.chmod(0o640)
    original_mode = external.stat().st_mode & 0o777

    partial = (tmp_path / "partial").resolve()
    partial.mkdir()
    os.link(external, partial / "hard-linked-artifact")
    (partial / "symlinked-artifact").symlink_to(external)
    alpha_max_runner._cleanup_partial_bundle(partial)
    assert not partial.exists()
    assert external.read_bytes() == b"external claim target\n"
    assert external.stat().st_mode & 0o777 == original_mode

    claim = (tmp_path / ".completion.claim").resolve()
    claim.symlink_to(external)
    alpha_max_runner._release_historical_completion_claim(claim)
    assert not claim.exists()
    assert external.read_bytes() == b"external claim target\n"
    assert external.stat().st_mode & 0o777 == original_mode


def test_bundle_write_is_bound_to_opened_root_when_parent_path_is_swapped(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = (tmp_path / "owned").resolve()
    nested = root / "nested"
    nested.mkdir(parents=True)
    moved = root / "nested-opened"
    external = (tmp_path / "external").resolve()
    external.mkdir()
    original_open = os.open
    swapped = False

    def hostile_open(path, flags, mode=0o777, *, dir_fd=None):
        nonlocal swapped
        if path == "artifact.json" and dir_fd is not None and not swapped:
            swapped = True
            nested.rename(moved)
            nested.symlink_to(external, target_is_directory=True)
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "open", hostile_open)
    alpha_max_runner._write_bundle_file(
        root,
        "nested/artifact.json",
        b'{"owned":true}\n',
    )

    assert swapped is True
    assert not (external / "artifact.json").exists()
    assert (moved / "artifact.json").read_bytes() == b'{"owned":true}\n'


def test_manifest_seal_rejects_hard_linked_target(tmp_path: Path) -> None:
    preflight = preflight_alpha_max_runtime_contract(str(CONFIG_PATH))
    admitted = preflight.candidate_symbols[:5]
    root = _owned_root(tmp_path)
    node = next(row for row in _nodes() if row["row_id"] == "component_trend_1x")
    materialized = materialize_alpha_max_manifest(
        node,
        {"component_trend_1x": 1.0},
        1.0,
        "validation_train_fit",
        str(CONFIG_PATH),
        str(root),
        preflight.candidate_symbols,
        admitted,
        "d" * 64,
    )
    seal = seal_alpha_max_manifest_activation(
        preflight,
        output_root=str(root),
        phase="validation_train_fit",
        manifest_path=materialized.path,
        admitted_symbols=admitted,
    )
    assert seal.manifest_receipt.sha256 == materialized.sha256

    target = Path(materialized.path)
    linked = target.with_suffix(".linked")
    os.link(target, linked)
    with pytest.raises(
        AlphaMaxRuntimeContractError, match="portfolio_manifest_activation_mismatch"
    ):
        seal_alpha_max_manifest_activation(
            preflight,
            output_root=str(root),
            phase="validation_train_fit",
            manifest_path=materialized.path,
            admitted_symbols=admitted,
        )


def test_actual_backtest_construction_binds_all_alpha_runtime_seams(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preflight = preflight_alpha_max_runtime_contract(str(CONFIG_PATH))
    admitted = preflight.candidate_symbols[:5]
    root = _owned_root(tmp_path)
    node = next(row for row in _nodes() if row["row_id"] == "component_trend_1x")
    materialized = materialize_alpha_max_manifest(
        node,
        {"component_trend_1x": 1.0},
        1.0,
        "validation_train_fit",
        str(CONFIG_PATH),
        str(root),
        preflight.candidate_symbols,
        admitted,
        "d" * 64,
    )
    purge_feature_root = (tmp_path / "features-purge").resolve()
    validation_feature_root = (tmp_path / "features-validation").resolve()
    purge_feature_root.mkdir()
    validation_feature_root.mkdir()
    purge_window = preflight.phase_windows["purge"]
    validation_window = preflight.phase_windows["validation"]
    lookup = AlphaMaxOrderedFundingLookup(
        (
            FeatureRootSpec(
                "purge",
                str(purge_feature_root),
                "binance",
                purge_window.start_utc,
                purge_window.end_utc,
                "a" * 64,
                "b" * 64,
            ),
            FeatureRootSpec(
                "validation",
                str(validation_feature_root),
                "binance",
                validation_window.start_utc,
                validation_window.end_utc,
                "c" * 64,
                "d" * 64,
            ),
        )
    )
    resolver = AlphaMaxFundingBoundaryResolver(lookup, admitted)
    timestamp_ms = int(datetime(2025, 6, 8, tzinfo=UTC).timestamp() * 1000)
    data_dict = dict.fromkeys(admitted, ((timestamp_ms, 100.0, 101.0, 99.0, 100.0, 10.0),))

    activation = construct_alpha_max_engine(
        preflight,
        output_root=str(root),
        phase="validation_train_fit",
        manifest_path=materialized.path,
        admitted_symbols=admitted,
        phase_id="validation",
        nominal_cost_bps=30,
        raw_root=str((tmp_path / "raw-validation").resolve()),
        ordered_lookup=lookup,
        funding_resolver=resolver,
        data_dict=data_dict,
    )

    validate_alpha_max_engine_activation(activation)
    assert type(activation.backtest) is Backtest
    assert type(activation.backtest.data_handler) is HistoricParquetWindowedDataHandler
    assert type(activation.backtest.strategy) is ArtifactPortfolioModeStrategy
    assert type(activation.backtest.portfolio) is Portfolio
    assert type(activation.backtest.execution_handler) is SimulatedExecutionHandler
    assert activation.backtest.symbol_list is admitted
    assert activation.backtest.data_handler.symbol_list is admitted
    assert activation.backtest.portfolio.symbol_list is admitted
    assert set(activation.constructor_plan.portfolio_kwargs) == {
        "fill_application_attribution_sink",
        "funding_boundary_resolver",
        "full_event_equity_sink",
        "reporting_sampling_timeframe",
    }
    fill_sink = activation.backtest.portfolio.fill_application_attribution_sink
    equity_sink = activation.backtest.portfolio.full_event_equity_sink
    assert fill_sink.__self__ is activation.attribution_collector
    assert fill_sink.__func__ is AlphaMaxAttributionCollector.record_application
    assert equity_sink.__self__ is activation.full_event_equity_tracker
    assert activation.backtest.portfolio.reporting_sampling_timeframe == "4h"
    assert activation.backtest.execution_handler.pricing_trace_evidence == ()

    # A crossed reporting boundary is priced from the completed native 4h
    # bucket even when sparse raw data has no row at the exact boundary and a
    # later row carries a hostile close.  The inclusive final endpoint settles
    # funding exactly once before it is emitted.
    reporting_start = datetime(2025, 6, 8, tzinfo=UTC)
    reporting_end = reporting_start + timedelta(hours=8)
    completed_close = 101.0
    hostile_later_close = 999.0
    aggregator = alpha_max_runner.TimeframeAggregator(timeframes=["4h"])
    for symbol in admitted:
        aggregator.update_from_1s_batch(
            symbol,
            (
                (
                    int((reporting_start + timedelta(seconds=5)).timestamp() * 1000),
                    100.0,
                    100.0,
                    100.0,
                    100.0,
                    1.0,
                ),
                (
                    int(
                        (reporting_start + timedelta(hours=3, minutes=59, seconds=55)).timestamp()
                        * 1000
                    ),
                    completed_close,
                    completed_close,
                    completed_close,
                    completed_close,
                    1.0,
                ),
                (
                    int((reporting_start + timedelta(hours=4, seconds=5)).timestamp() * 1000),
                    hostile_later_close,
                    hostile_later_close,
                    hostile_later_close,
                    hostile_later_close,
                    1.0,
                ),
            ),
        )
    activation.backtest.timeframe_aggregator = aggregator
    activation.backtest.portfolio.current_holdings["cash"] = 1_000.0
    activation.backtest.portfolio.current_positions[admitted[0]] = 2.0
    funding_boundaries: list[datetime] = []

    def record_funding(_portfolio: Portfolio, boundary: datetime) -> None:
        funding_boundaries.append(boundary)

    monkeypatch.setattr(Portfolio, "_apply_funding", record_funding)
    fanout = alpha_max_runner._AlphaMaxFoldEquityFanout(
        AlphaMaxStreamingEquityTracker(),
        aggregate_scale=1.0,
        reporting_start=reporting_start,
        reporting_end=reporting_end,
    )
    fanout.bind_backtest(activation.backtest)
    fanout.observe(
        (
            (reporting_start + timedelta(hours=4, seconds=5)).timestamp(),
            12_345.0,
        )
    )
    fanout.settle_day_end(reporting_end, settle_funding=True)

    endpoints = fanout.reporting_endpoints
    assert endpoints[0].equity == pytest.approx(1_000.0 + (2.0 * completed_close))
    assert endpoints[0].equity != pytest.approx(1_000.0 + (2.0 * hostile_later_close))
    assert funding_boundaries == [reporting_end]
    with pytest.raises(
        AlphaMaxRuntimeContractError,
        match="reporting_boundary_sequence_invalid",
    ):
        fanout.settle_day_end(reporting_end, settle_funding=True)
    assert funding_boundaries == [reporting_end]


def test_actual_engine_intrabar_liquidation_wipeout_is_terminal_before_open_order_sweep(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preflight = preflight_alpha_max_runtime_contract(str(CONFIG_PATH))
    admitted = preflight.candidate_symbols[:5]
    root = _owned_root(tmp_path)
    node = next(row for row in _nodes() if row["row_id"] == "component_trend_1x")
    materialized = materialize_alpha_max_manifest(
        node,
        {"component_trend_1x": 1.0},
        1.0,
        "validation_train_fit",
        str(CONFIG_PATH),
        str(root),
        preflight.candidate_symbols,
        admitted,
        "d" * 64,
    )
    purge_feature_root = (tmp_path / "features-purge").resolve()
    validation_feature_root = (tmp_path / "features-validation").resolve()
    purge_feature_root.mkdir()
    validation_feature_root.mkdir()
    purge_window = preflight.phase_windows["purge"]
    validation_window = preflight.phase_windows["validation"]
    lookup = AlphaMaxOrderedFundingLookup(
        (
            FeatureRootSpec(
                "purge",
                str(purge_feature_root),
                "binance",
                purge_window.start_utc,
                purge_window.end_utc,
                "a" * 64,
                "b" * 64,
            ),
            FeatureRootSpec(
                "validation",
                str(validation_feature_root),
                "binance",
                validation_window.start_utc,
                validation_window.end_utc,
                "c" * 64,
                "d" * 64,
            ),
        )
    )
    resolver = AlphaMaxFundingBoundaryResolver(lookup, admitted)
    start = datetime(2025, 6, 8, 1, 0, tzinfo=UTC)
    timestamp_ms = int(start.timestamp() * 1000)
    rows: dict[str, tuple[tuple[object, ...], ...]] = {}
    for index, symbol in enumerate(admitted):
        second_low, second_close = 99.0, 100.0
        if index == 0:
            second_low, second_close = 50.0, 100.0
        elif index == 1:
            second_low, second_close = 40.0, 50.0
        elif index == 2:
            second_low, second_close = 80.0, 100.0
        rows[symbol] = (
            (timestamp_ms, 100.0, 101.0, 99.0, 100.0, 1_000_000.0),
            (timestamp_ms + 1000, 100.0, 101.0, second_low, second_close, 1_000_000.0),
        )

    activation = construct_alpha_max_engine(
        preflight,
        output_root=str(root),
        phase="validation_train_fit",
        manifest_path=materialized.path,
        admitted_symbols=admitted,
        phase_id="validation",
        nominal_cost_bps=30,
        raw_root=str((tmp_path / "raw-validation").resolve()),
        ordered_lookup=lookup,
        funding_resolver=resolver,
        data_dict=rows,
    )
    validate_alpha_max_engine_activation(activation)
    backtest = activation.backtest
    execution = backtest.execution_handler
    portfolio = backtest.portfolio
    raw_calls: list[tuple[str, object]] = []
    raw_check = execution.check_open_orders

    def check_open_orders(event) -> None:
        raw_calls.append((event.symbol, event.time))
        raw_check(event)

    seen_fills = []
    monkeypatch.setattr(execution, "check_open_orders", check_open_orders)
    monkeypatch.setattr(backtest, "on_fill", seen_fills.append)

    backtest.process_event(OrderEvent(admitted[0], "MKT", 1.0, "BUY"))
    backtest.process_event(OrderEvent(admitted[1], "MKT", 1000.0, "BUY"))
    backtest.process_event(OrderEvent(admitted[2], "STOP", 1.0, "SELL", stop_price=90.0))
    backtest._run_backtest()

    assert raw_calls == [(symbol, timestamp_ms) for symbol in admitted]
    assert backtest.market_events == 10
    assert backtest.orders == 3
    assert backtest.fills == 4
    assert portfolio.trade_count == 4
    assert len(seen_fills) == 4
    assert [fill.exchange for fill in seen_fills[:2]] == ["BINANCE_SIM", "BINANCE_SIM"]
    assert [fill.exchange for fill in seen_fills[2:]] == [
        "SIM_LIQUIDATION",
        "SIM_LIQUIDATION",
    ]
    assert all(fill.status == "LIQUIDATED" for fill in seen_fills[2:])
    liquidation_metadata = {
        "reason",
        "entry_price",
        "liquidation_price",
        "trigger_price",
        "bar_high",
        "bar_low",
        "close_price",
        "leverage",
        "configured_margin_mode",
        "modeled_margin_mode",
    }
    assert all(liquidation_metadata <= set(fill.metadata or {}) for fill in seen_fills[2:])
    assert all(
        (fill.metadata or {})["reason"] == "maintenance_margin_breach" for fill in seen_fills[2:]
    )

    assert len(portfolio.liquidation_events) == 2
    liquidation_by_symbol = {str(event["symbol"]): event for event in portfolio.liquidation_events}
    intrabar = liquidation_by_symbol[admitted[0]]
    assert set(intrabar) == {
        "bar_high",
        "bar_low",
        "close_price",
        "commission",
        "configured_margin_mode",
        "entry_price",
        "fill_cost",
        "leverage",
        "liquidation_price",
        "modeled_margin_mode",
        "position_qty",
        "reason",
        "symbol",
        "time",
        "trigger_price",
    }
    normalized_liquidations = alpha_max_evidence._alpha_max_normalize_liquidation_events(
        tuple(portfolio.liquidation_events)
    )
    persisted_intrabar = next(
        value.to_payload() for value in normalized_liquidations if value.symbol == admitted[0]
    )
    assert persisted_intrabar["reason"] == "maintenance_margin_breach"
    assert persisted_intrabar["trigger_price"] == pytest.approx(50.0)
    assert persisted_intrabar["bar_high"] == pytest.approx(101.0)
    assert persisted_intrabar["bar_low"] == pytest.approx(50.0)
    assert persisted_intrabar["leverage"] == pytest.approx(3.0)
    assert float(intrabar["close_price"]) > float(intrabar["liquidation_price"])
    assert 50.0 < float(intrabar["liquidation_price"]) < 100.0
    assert portfolio.current_positions[admitted[0]] == pytest.approx(0.0)
    assert portfolio.current_positions[admitted[1]] == pytest.approx(0.0)
    assert admitted[0] not in portfolio._pending_liquidation
    assert admitted[1] not in portfolio._pending_liquidation

    remaining = execution.active_orders
    assert len(remaining) == 1
    assert remaining[0]["symbol"] == admitted[2]
    assert remaining[0]["type"] == "STOP"
    assert all(fill.symbol != admitted[2] for fill in seen_fills)

    traces = execution.pricing_trace_evidence
    applications = activation.attribution_collector.applications
    capacity = execution.capacity_observation_evidence
    assert len(traces) == len(applications) == len(capacity) == 2
    assert {value.equity_before for value in capacity} == {10_000.0}
    assert backtest.fills == len(traces) + len(portfolio.liquidation_events)
    assert portfolio.current_holdings["commission"] == pytest.approx(
        sum(float(fill.commission or 0.0) for fill in seen_fills)
    )

    full_event = activation.full_event_equity_tracker.finalize()
    assert full_event.event_count == 2
    assert full_event.minimum_equity <= 0.0
    assert full_event.ruin_detected is True
    assert full_event.full_event_mdd == 1.0
    assert full_event.uncapped_full_event_drawdown > 1.0


def _native_finalization_coverage_stub(
    completed_keys: list[tuple[str, str]],
    *,
    strategy_class: str = "ResearchOnlyDailyLowTurnoverTrendPersistenceStrategy",
) -> dict[str, object]:
    counts: dict[str, int] = {}
    last: dict[str, str] = {}
    for symbol, key in sorted(completed_keys):
        counts[symbol] = counts.get(symbol, 0) + 1
        last[symbol] = max(last.get(symbol, key), key)
    atomic = strategy_class == "ResearchOnlyDailyCrossSectionalNearHighAnchoringStrategy"
    barrier_keys = sorted({key for _symbol, key in completed_keys}) if atomic else []
    return {
        "adapter_class": strategy_class,
        "native_timeframe": "1d",
        "barrier_mode": "atomic_cross_section" if atomic else "none",
        "completed_native_keys": sorted(completed_keys),
        "completed_native_count_by_symbol": dict(sorted(counts.items())),
        "last_completed_native_key_by_symbol": dict(sorted(last.items())),
        "barrier_pending_keys": barrier_keys,
        "barrier_closed_keys": barrier_keys,
        "barrier_symbol_coverage": {
            key: sorted(symbol for symbol, completed_key in completed_keys if completed_key == key)
            for key in barrier_keys
        },
        "failed_native_keys": {},
        "partial_bucket_error": None,
    }


def test_native_finalization_is_exact_once_and_seals_discarded_signals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class CaptureQueue:
        def __init__(self) -> None:
            self.items: list[object] = []

        def put(self, value: object) -> None:
            self.items.append(value)

        def drain(self) -> list[object]:
            values = list(self.items)
            self.items.clear()
            return values

    component = alpha_max_runner.AlphaMaxExpectedComponent(
        component_id="component_trend_1x",
        strategy_class="ResearchOnlyDailyLowTurnoverTrendPersistenceStrategy",
        symbols=("BTCUSDT", "ETHUSDT"),
        params_bytes=b"{}",
        weight=1.0,
        source_artifact_id="current_trial_registry",
    )
    expected = alpha_max_runner.AlphaMaxExpectedDefinition(
        portfolio_mode="manifest:/sealed/component_trend_1x.json",
        artifact_kind="alpha_max_portfolio_manifest.v1",
        candidate_symbols=("BTCUSDT", "ETHUSDT"),
        admitted_symbols=("BTCUSDT", "ETHUSDT"),
        admission_manifest_sha256="a" * 64,
        gross_cap=1.0,
        cash_weight=0.0,
        allocation_method="static",
        source_path="/sealed/component_trend_1x.json",
        source_sha256="b" * 64,
        components=(component,),
        native_timeframes=("1d",),
    )
    strategy = object.__new__(ArtifactPortfolioModeStrategy)
    strategy.events = alpha_max_runner.FastQueue()
    child_queue = CaptureQueue()
    completed_keys: list[tuple[str, str]] = [
        ("BTCUSDT", "2025-06-07"),
        ("ETHUSDT", "2025-06-07"),
    ]
    child = SimpleNamespace(
        get_native_finalization_evidence=lambda: _native_finalization_coverage_stub(completed_keys)
    )
    strategy._children = [(component, child, child_queue)]
    boundary = datetime(2025, 6, 9, tzinfo=UTC)
    signal = SignalEvent(
        strategy_id="trend",
        symbol="BTCUSDT",
        datetime=datetime(2025, 6, 8, 23, 59, 59),
        signal_type="LONG",
        strength=0.25,
        metadata={"boundary": "final"},
    )
    calls: list[datetime] = []

    def finalize(_strategy: ArtifactPortfolioModeStrategy, watermark: datetime):
        calls.append(watermark)
        completed_keys.extend((("BTCUSDT", "2025-06-08"), ("ETHUSDT", "2025-06-08")))
        child_queue.put(signal)
        return {component.component_id: 2}

    monkeypatch.setattr(
        ArtifactPortfolioModeStrategy,
        "finalize_completed_native_buckets",
        finalize,
    )
    receipt = alpha_max_runner._finalize_alpha_max_native_boundary(
        strategy,
        expected,
        boundary,
        admitted_symbol_count=2,
        require_exact_counts=True,
    )

    expected_payload = alpha_max_runner._alpha_max_boundary_signal_payload(
        component.component_id,
        signal,
    )
    expected_signal_bytes = alpha_max_runner._canonical_bytes(expected_payload) + b"\n"
    assert calls == [boundary]
    assert dict(receipt.finalized_children) == {component.component_id: 2}
    assert receipt.discarded_signal_count == 1
    assert receipt.discarded_signal_sha256 == alpha_max_runner._sha256(expected_signal_bytes)
    assert expected_payload["datetime"] == "2025-06-08T23:59:59Z"
    assert child_queue.items == []
    assert strategy.events.empty()
    coverage = receipt.to_payload()["native_coverage_by_child"][component.component_id]
    assert coverage["finalization_completed_native_keys"] == [
        ["BTCUSDT", "2025-06-08"],
        ["ETHUSDT", "2025-06-08"],
    ]
    assert coverage["finalization_barrier_keys"] == []


def test_native_finalization_receipt_seals_atomic_barrier_coverage_fail_closed() -> None:
    child_id = "component_near_high_1x"
    key = "2025-06-08"
    coverage = {
        "adapter_class": "ResearchOnlyDailyCrossSectionalNearHighAnchoringStrategy",
        "native_timeframe": "1d",
        "barrier_mode": "atomic_cross_section",
        "completed_native_keys": [["BTCUSDT", key], ["ETHUSDT", key]],
        "completed_native_count_by_symbol": {"BTCUSDT": 1, "ETHUSDT": 1},
        "last_completed_native_key_by_symbol": {"BTCUSDT": key, "ETHUSDT": key},
        "barrier_pending_keys": [key],
        "barrier_closed_keys": [key],
        "barrier_symbol_coverage": {key: ["BTCUSDT", "ETHUSDT"]},
        "failed_native_keys": {},
        "partial_bucket_error": None,
        "finalization_completed_native_keys": [["BTCUSDT", key], ["ETHUSDT", key]],
        "finalization_barrier_keys": [key],
    }
    receipt = alpha_max_evidence.build_alpha_max_native_finalization_receipt(
        boundary_utc=datetime(2025, 6, 9, tzinfo=UTC),
        finalized_children={child_id: 1},
        native_coverage_by_child={child_id: coverage},
        discarded_signal_count=0,
        discarded_signal_sha256=hashlib.sha256(b"").hexdigest(),
    )
    assert receipt.to_payload()["native_coverage_by_child"][child_id] == coverage

    for poison in (
        {"barrier_closed_keys": []},
        {"failed_native_keys": {key: "poisoned_barrier"}},
        {"finalization_barrier_keys": []},
    ):
        with pytest.raises(ValueError, match="native_finalization_coverage_invalid"):
            alpha_max_evidence.build_alpha_max_native_finalization_receipt(
                boundary_utc=datetime(2025, 6, 9, tzinfo=UTC),
                finalized_children={child_id: 1},
                native_coverage_by_child={child_id: {**coverage, **poison}},
                discarded_signal_count=0,
                discarded_signal_sha256=hashlib.sha256(b"").hexdigest(),
            )


@pytest.mark.parametrize(
    "poison",
    (
        {"adapter_class": "StubNativeAdapter"},
        {"native_timeframe": "4h"},
        {"barrier_mode": "atomic_cross_section"},
        {
            "completed_native_keys": [("ETHUSDT", "2025-06-07")],
            "completed_native_count_by_symbol": {"ETHUSDT": 1},
            "last_completed_native_key_by_symbol": {"ETHUSDT": "2025-06-07"},
        },
    ),
)
def test_native_finalization_rejects_unbound_adapter_timeframe_mode_and_symbols(
    monkeypatch: pytest.MonkeyPatch,
    poison: dict[str, object],
) -> None:
    component = alpha_max_runner.AlphaMaxExpectedComponent(
        component_id="component_trend_1x",
        strategy_class="ResearchOnlyDailyLowTurnoverTrendPersistenceStrategy",
        symbols=("BTCUSDT",),
        params_bytes=b"{}",
        weight=1.0,
        source_artifact_id="current_trial_registry",
    )
    expected = alpha_max_runner.AlphaMaxExpectedDefinition(
        portfolio_mode="manifest:/sealed/component.json",
        artifact_kind="alpha_max_portfolio_manifest.v1",
        candidate_symbols=("BTCUSDT",),
        admitted_symbols=("BTCUSDT",),
        admission_manifest_sha256="a" * 64,
        gross_cap=1.0,
        cash_weight=0.0,
        allocation_method="static",
        source_path="/sealed/component.json",
        source_sha256="b" * 64,
        components=(component,),
        native_timeframes=("1d",),
    )
    snapshot = {
        **_native_finalization_coverage_stub([("BTCUSDT", "2025-06-07")]),
        **poison,
    }
    strategy = object.__new__(ArtifactPortfolioModeStrategy)
    strategy.events = alpha_max_runner.FastQueue()
    strategy._children = [
        (
            component,
            SimpleNamespace(get_native_finalization_evidence=lambda: snapshot),
            SimpleNamespace(drain=lambda: []),
        )
    ]
    monkeypatch.setattr(
        ArtifactPortfolioModeStrategy,
        "finalize_completed_native_buckets",
        lambda _strategy, _watermark: pytest.fail("unbound snapshot reached finalizer"),
    )

    with pytest.raises(
        AlphaMaxRuntimeContractError,
        match="native_finalization_coverage_binding_mismatch",
    ):
        alpha_max_runner._finalize_alpha_max_native_boundary(
            strategy,
            expected,
            datetime(2025, 6, 9, tzinfo=UTC),
            admitted_symbol_count=1,
            require_exact_counts=True,
        )


@pytest.mark.parametrize(
    ("strategy_class", "finalized_count", "expected_error"),
    (
        (
            "ResearchOnlyDailyLowTurnoverTrendPersistenceStrategy",
            0,
            "native_finalization_invalid",
        ),
        (
            "ResearchOnlyDailyLowTurnoverTrendPersistenceStrategy",
            3,
            "native_finalization_count_mismatch",
        ),
        (
            "ResearchOnlyDailyCrossSectionalNearHighAnchoringStrategy",
            2,
            "native_finalization_count_mismatch",
        ),
    ),
)
def test_native_finalization_rejects_zero_duplicate_and_near_high_count_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    strategy_class: str,
    finalized_count: int,
    expected_error: str,
) -> None:
    component = alpha_max_runner.AlphaMaxExpectedComponent(
        component_id="component",
        strategy_class=strategy_class,
        symbols=("BTCUSDT", "ETHUSDT"),
        params_bytes=b"{}",
        weight=1.0,
        source_artifact_id="current_trial_registry",
    )
    expected = alpha_max_runner.AlphaMaxExpectedDefinition(
        portfolio_mode="manifest:/sealed/component.json",
        artifact_kind="alpha_max_portfolio_manifest.v1",
        candidate_symbols=("BTCUSDT", "ETHUSDT"),
        admitted_symbols=("BTCUSDT", "ETHUSDT"),
        admission_manifest_sha256="a" * 64,
        gross_cap=1.0,
        cash_weight=0.0,
        allocation_method="static",
        source_path="/sealed/component.json",
        source_sha256="b" * 64,
        components=(component,),
        native_timeframes=("1d",),
    )
    strategy = object.__new__(ArtifactPortfolioModeStrategy)
    strategy.events = alpha_max_runner.FastQueue()
    strategy._children = [
        (
            component,
            SimpleNamespace(
                get_native_finalization_evidence=lambda: _native_finalization_coverage_stub(
                    [("BTCUSDT", "2025-06-07"), ("ETHUSDT", "2025-06-07")],
                    strategy_class=strategy_class,
                )
            ),
            SimpleNamespace(drain=lambda: []),
        )
    ]
    monkeypatch.setattr(
        ArtifactPortfolioModeStrategy,
        "finalize_completed_native_buckets",
        lambda _strategy, _watermark: {component.component_id: finalized_count},
    )

    with pytest.raises(AlphaMaxRuntimeContractError, match=expected_error):
        alpha_max_runner._finalize_alpha_max_native_boundary(
            strategy,
            expected,
            datetime(2025, 6, 9, tzinfo=UTC),
            admitted_symbol_count=2,
            require_exact_counts=True,
        )


def test_native_finalization_rejects_non_signal_child_queue_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    component = alpha_max_runner.AlphaMaxExpectedComponent(
        component_id="component_trend_1x",
        strategy_class="ResearchOnlyDailyLowTurnoverTrendPersistenceStrategy",
        symbols=("BTCUSDT",),
        params_bytes=b"{}",
        weight=1.0,
        source_artifact_id="current_trial_registry",
    )
    expected = alpha_max_runner.AlphaMaxExpectedDefinition(
        portfolio_mode="manifest:/sealed/component.json",
        artifact_kind="alpha_max_portfolio_manifest.v1",
        candidate_symbols=("BTCUSDT",),
        admitted_symbols=("BTCUSDT",),
        admission_manifest_sha256="a" * 64,
        gross_cap=1.0,
        cash_weight=0.0,
        allocation_method="static",
        source_path="/sealed/component.json",
        source_sha256="b" * 64,
        components=(component,),
        native_timeframes=("1d",),
    )
    strategy = object.__new__(ArtifactPortfolioModeStrategy)
    strategy.events = alpha_max_runner.FastQueue()
    strategy._children = [
        (
            component,
            SimpleNamespace(
                get_native_finalization_evidence=lambda: _native_finalization_coverage_stub(
                    [("BTCUSDT", "2025-06-07")]
                )
            ),
            SimpleNamespace(drain=lambda: [SimpleNamespace(type="ORDER")]),
        )
    ]
    monkeypatch.setattr(
        ArtifactPortfolioModeStrategy,
        "finalize_completed_native_buckets",
        lambda _strategy, _watermark: {component.component_id: 1},
    )

    with pytest.raises(AlphaMaxRuntimeContractError, match="boundary_event_invalid"):
        alpha_max_runner._finalize_alpha_max_native_boundary(
            strategy,
            expected,
            datetime(2025, 6, 9, tzinfo=UTC),
            admitted_symbol_count=1,
            require_exact_counts=True,
        )


def test_day_boundary_settlement_finalizes_only_the_scoring_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    activation = object.__new__(alpha_max_runner.AlphaMaxEngineActivation)
    object.__setattr__(activation, "backtest", SimpleNamespace(strategy="strategy"))
    object.__setattr__(
        activation,
        "artifact_seal",
        SimpleNamespace(expected_definition="definition"),
    )
    object.__setattr__(activation, "admitted_symbols", ("BTCUSDT",))
    tracker = object.__new__(alpha_max_runner._AlphaMaxFoldEquityFanout)
    settled: list[tuple[datetime, bool]] = []
    finalized: list[tuple[object, ...]] = []
    sentinel = object()

    monkeypatch.setattr(
        alpha_max_runner._AlphaMaxFoldEquityFanout,
        "settle_day_end",
        lambda _tracker, boundary, *, settle_funding: settled.append((boundary, settle_funding)),
    )

    def finalize(*args, **kwargs):
        finalized.append((*args, kwargs))
        return sentinel

    monkeypatch.setattr(alpha_max_runner, "_finalize_alpha_max_native_boundary", finalize)
    internal = datetime(2025, 6, 9, tzinfo=UTC)
    final = internal + timedelta(days=1)

    assert (
        alpha_max_runner._settle_alpha_max_day_boundary(
            activation,
            tracker,
            internal,
            scoring_boundary=False,
        )
        is None
    )
    assert finalized == []
    assert (
        alpha_max_runner._settle_alpha_max_day_boundary(
            activation,
            tracker,
            final,
            scoring_boundary=True,
        )
        is sentinel
    )
    assert settled == [(internal, True), (final, True)]
    assert len(finalized) == 1
    assert finalized[0][0:3] == ("strategy", "definition", final)
    assert finalized[0][3] == {
        "admitted_symbol_count": 1,
        "require_exact_counts": True,
    }


def test_prelock_and_historical_bundles_are_exclusive_immutable_and_duplicate_safe(
    tmp_path: Path,
) -> None:
    prelock_root = (tmp_path / "prelock").resolve()
    stable = {"status/matrix.json": b'{"status":"complete"}\n'}
    prelock = create_alpha_max_prelock_bundle(
        str(prelock_root),
        stable,
        prelock_champion=None,
        selected_candidate_id=None,
    )
    before = {
        path.relative_to(prelock_root): path.read_bytes()
        for path in prelock_root.rglob("*")
        if path.is_file()
    }

    assert Path(prelock.seal_path).is_file()
    assert prelock_root.stat().st_mode & 0o222 == 0
    with pytest.raises(AlphaMaxRuntimeContractError, match="output_root_exists"):
        create_alpha_max_prelock_bundle(
            str(prelock_root),
            stable,
            prelock_champion=None,
            selected_candidate_id=None,
        )

    historical = create_alpha_max_historical_package(
        str(prelock_root),
        str((tmp_path / "historical-1").resolve()),
        {"report.json": b'{"report_only":true}\n'},
        completion_id="evaluation-001",
    )
    after = {
        path.relative_to(prelock_root): path.read_bytes()
        for path in prelock_root.rglob("*")
        if path.is_file()
    }
    assert before == after
    assert Path(historical.seal_path).is_file()
    with pytest.raises(AlphaMaxRuntimeContractError, match="completion_duplicate"):
        create_alpha_max_historical_package(
            str(prelock_root),
            str((tmp_path / "historical-2").resolve()),
            {"report.json": b'{"report_only":true}\n'},
            completion_id="evaluation-001",
        )


def test_historical_gate_requires_exact_complete_prelock_matrix_schema(
    tmp_path: Path,
) -> None:
    statuses: list[dict[str, object]] = []
    for row_id in alpha_max_runner._ALPHA_MAX_CURRENT_ROW_IDS:
        for nominal in ALPHA_MAX_COST_CELL_BPS:
            if row_id in alpha_max_runner._ALPHA_MAX_UNAVAILABLE_ROWS:
                statuses.append(
                    {
                        "capsule_sha256": None,
                        "engine_constructed": False,
                        "manifest_sha256": None,
                        "nominal_cost_bps": nominal,
                        "row_id": row_id,
                        "row_role": "incumbent_unavailable",
                        "selection_eligible": False,
                        "status": "incumbent_replay_unavailable",
                    }
                )
            elif row_id in alpha_max_runner._ALPHA_MAX_DIAGNOSTIC_ROWS:
                statuses.append(
                    {
                        "capsule_sha256": None,
                        "engine_constructed": False,
                        "manifest_sha256": None,
                        "nominal_cost_bps": nominal,
                        "row_id": row_id,
                        "row_role": "track_b_diagnostic",
                        "selection_eligible": False,
                        "status": "diagnostic_report_only",
                    }
                )
            else:
                statuses.append(
                    {
                        "capsule_sha256": "a" * 64,
                        "cell_sha256": "b" * 64,
                        "engine_constructed": True,
                        "manifest_sha256": "c" * 64,
                        "nominal_cost_bps": nominal,
                        "row_id": row_id,
                        "row_role": "resolvable_candidate",
                        "selection_eligible": True,
                        "status": "resolved_engine_cell_complete",
                    }
                )
    valid = {
        "artifact_kind": "alpha_max_matrix_statuses.v1",
        "domain": "validation",
        "engine_cell_count": 68,
        "physical_fold_run_count": 816,
        "status_count": 84,
        "statuses": statuses,
    }
    prelock_payload = {
        "engine_cell_count": 68,
        "physical_fold_run_count": 816,
    }

    def validate(payload: dict[str, object], name: str) -> None:
        matrix_bytes = alpha_max_runner._canonical_bytes(payload) + b"\n"
        root = (tmp_path / name).resolve()
        create_alpha_max_prelock_bundle(
            root,
            {"status/matrix.json": matrix_bytes},
            prelock_champion=None,
            selected_candidate_id=None,
        )
        snapshot = alpha_max_runner._snapshot_bundle_tree(root)
        alpha_max_runner._validate_complete_alpha_max_prelock_matrix(
            snapshot,
            prelock_payload,
        )

    validate(valid, "valid")
    mutations: list[dict[str, object]] = []

    def mutated() -> dict[str, object]:
        return json.loads(json.dumps(valid))

    resolved_index = next(
        index
        for index, value in enumerate(statuses)
        if value["row_id"] in alpha_max_runner._ALPHA_MAX_RESOLVABLE_ROWS
    )
    unavailable_index = next(
        index
        for index, value in enumerate(statuses)
        if value["row_id"] in alpha_max_runner._ALPHA_MAX_UNAVAILABLE_ROWS
    )
    diagnostic_index = next(
        index
        for index, value in enumerate(statuses)
        if value["row_id"] in alpha_max_runner._ALPHA_MAX_DIAGNOSTIC_ROWS
    )
    extra_key = mutated()
    extra_key["statuses"][resolved_index]["attacker"] = True  # type: ignore[index]
    mutations.append(extra_key)
    invalid_hash = mutated()
    invalid_hash["statuses"][resolved_index]["capsule_sha256"] = "not-a-sha"  # type: ignore[index]
    mutations.append(invalid_hash)
    invalid_bool = mutated()
    invalid_bool["statuses"][resolved_index]["selection_eligible"] = "true"  # type: ignore[index]
    mutations.append(invalid_bool)
    invalid_unavailable = mutated()
    invalid_unavailable["statuses"][unavailable_index]["status"] = "attacker_status"  # type: ignore[index]
    mutations.append(invalid_unavailable)
    invalid_diagnostic = mutated()
    invalid_diagnostic["statuses"][diagnostic_index]["row_role"] = "attacker_role"  # type: ignore[index]
    mutations.append(invalid_diagnostic)
    reordered = mutated()
    reordered["statuses"][0], reordered["statuses"][1] = (  # type: ignore[index]
        reordered["statuses"][1],  # type: ignore[index]
        reordered["statuses"][0],  # type: ignore[index]
    )
    mutations.append(reordered)
    top_level_extra = mutated()
    top_level_extra["attacker"] = True
    mutations.append(top_level_extra)

    for index, payload in enumerate(mutations):
        with pytest.raises(
            AlphaMaxRuntimeContractError,
            match="prelock_matrix_incomplete",
        ):
            validate(payload, f"invalid-{index}")


def test_sealed_selection_parser_rejects_coercion_missing_attribution_and_relabeling() -> None:
    gate_inputs = tuple(
        AlphaMaxGateInput(
            row_id=row_id,
            comparison_role="prelock_selection",
            evidence_tier="actual_engine",
            comparison_valid=True,
            nominal_cost_bps=30,
            cumulative_return=0.20 + index / 100.0,
            cagr=0.10 + index / 1000.0,
            calmar=0.50 + index / 1000.0,
            net_sharpe=1.0 + index / 1000.0,
            full_event_mdd=0.20,
            reporting_4h_mdd=0.20,
            dsr=0.95,
            spa_pvalue=0.01,
            pbo=0.10,
            native_data_coverage_complete=True,
            funding_coverage_complete=True,
            hash_valid=True,
            manifest_valid=True,
            reconciliation_complete=True,
            ruin=False,
            raw_root_set_sha256="a" * 64,
            feature_root_set_sha256="b" * 64,
            universe_sha256="c" * 64,
            calendar_sha256="d" * 64,
            seed_schedule_sha256="e" * 64,
        )
        for index, row_id in enumerate(alpha_max_runner._ALPHA_MAX_RESOLVABLE_ROWS)
    )
    selection = select_alpha_max_prelock_champion(gate_inputs)
    parsed = alpha_max_runner._alpha_max_selection_from_bytes(
        selection.canonical_bytes,
        role="prelock_selection",
    )
    assert parsed.prelock_champion == selection.prelock_champion
    assert len(parsed.decisions) == len(alpha_max_runner._ALPHA_MAX_RESOLVABLE_ROWS)
    assert len(parsed.scaling_attributions) == 2

    def mutated() -> dict[str, object]:
        return json.loads(selection.canonical_bytes)

    invalid_payloads: list[dict[str, object]] = []
    extra = mutated()
    extra["attacker"] = True
    invalid_payloads.append(extra)
    string_bool = mutated()
    string_bool["decisions"][0]["eligible"] = "false"  # type: ignore[index]
    invalid_payloads.append(string_bool)
    empty_decisions = mutated()
    empty_decisions["decisions"] = []
    empty_decisions["ranked_candidate_ids"] = []
    empty_decisions["prelock_champion"] = None
    empty_decisions["selected_candidate_id"] = None
    invalid_payloads.append(empty_decisions)
    no_scaling = mutated()
    no_scaling["scaling_attributions"] = []
    invalid_payloads.append(no_scaling)
    wrong_champion = mutated()
    wrong_champion["prelock_champion"] = None
    wrong_champion["selected_candidate_id"] = None
    wrong_champion["historical_evaluation_leader"] = selection.prelock_champion
    invalid_payloads.append(wrong_champion)
    coerced_attribution = mutated()
    coerced_attribution["scaling_attributions"][0]["sibling_gate_eligible"] = "true"  # type: ignore[index]
    invalid_payloads.append(coerced_attribution)

    for payload in invalid_payloads:
        raw = alpha_max_runner._canonical_bytes(payload) + b"\n"
        with pytest.raises(
            AlphaMaxRuntimeContractError,
            match="selection_artifact_invalid",
        ):
            alpha_max_runner._alpha_max_selection_from_bytes(
                raw,
                role="prelock_selection",
            )


def _trend_liquidity_buckets_fixture():
    admitted = ALPHA_MAX_CANDIDATE_SYMBOLS[:5]
    medians = {symbol: float(index + 1) for index, symbol in enumerate(admitted)}
    bucket_by_symbol = {
        symbol: ("weakest" if index < 2 else "middle" if index < 4 else "liquid")
        for index, symbol in enumerate(admitted)
    }
    payload = {
        "admission_computation_sha256": "a" * 64,
        "admitted_symbols": list(admitted),
        "artifact_kind": "alpha_max_train_liquidity_buckets.v1",
        "bucket_by_symbol": bucket_by_symbol,
        "bucket_order": ["weakest", "middle", "liquid"],
        "bucket_rule": "floor(3*ascending_rank_index/admitted_symbol_count)",
        "median_quote_notional_usdt": medians,
        "phase": "train_frozen_report_only",
        "report_only": True,
        "selection_influence": False,
        "symbols_by_bucket": {
            bucket: [symbol for symbol in admitted if bucket_by_symbol[symbol] == bucket]
            for bucket in ("weakest", "middle", "liquid")
        },
        "tie_break": "median_quote_notional_usdt_ascending_then_symbol_ascending",
    }
    return validate_alpha_max_train_liquidity_buckets(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    )


def _exact_trend_liquidity_matrix_fixture(domain: str):
    buckets = _trend_liquidity_buckets_fixture()
    fold_runs = []
    fold_hashes = []
    contributions_by_fold = []
    for fold_index, fold_id in enumerate(alpha_max_runner._alpha_max_fold_ids(domain)):
        fold_hash = hashlib.sha256(f"{domain}:{fold_id}".encode()).hexdigest()
        diagnostics = object.__new__(AlphaMaxRunReportOnlyDiagnostics)
        contributions = {
            symbol: (
                float(fold_index + symbol_index + 1)
                if symbol in buckets.admitted_symbols
                else -1_000_000.0
            )
            for symbol_index, symbol in enumerate(ALPHA_MAX_CANDIDATE_SYMBOLS)
        }
        object.__setattr__(diagnostics, "symbol_contribution_usdt", contributions)
        actual_run = object.__new__(AlphaMaxActualEngineRunReceipt)
        object.__setattr__(actual_run, "report_only_diagnostics", diagnostics)
        fold = object.__new__(AlphaMaxFoldRunEvidence)
        object.__setattr__(fold, "actual_engine_run", actual_run)
        object.__setattr__(fold, "sha256", fold_hash)
        fold_runs.append(fold)
        fold_hashes.append(fold_hash)
        contributions_by_fold.append(contributions)

    pre_gate = object.__new__(AlphaMaxCostCellPreGateEvidence)
    object.__setattr__(pre_gate, "fold_runs", tuple(fold_runs))
    target_cell = object.__new__(AlphaMaxCostCellEvidence)
    object.__setattr__(target_cell, "pre_gate_evidence", pre_gate)
    decoy = SimpleNamespace(pre_gate_evidence=SimpleNamespace(fold_runs=()))
    matrix = alpha_max_runner._AlphaMaxCompletedMatrix(
        domain=domain,
        rows=(),
        cells=MappingProxyType(
            {
                ("component_trend_1x", 20): decoy,
                ("component_trend_1x", 30): target_cell,
                ("full_equal_weight_1x", 30): decoy,
            }
        ),
        status_payload=b"{}\n",
        physical_fold_run_count=len(fold_runs),
        prepared_rows=MappingProxyType({}),
        gross_by_row=MappingProxyType({}),
    )
    return matrix, buckets, tuple(fold_hashes), tuple(contributions_by_fold)


def test_trend_liquidity_falsifier_uses_nominal_30_trend_receipts_and_admitted_symbols() -> None:
    matrix, buckets, fold_hashes, contributions = _exact_trend_liquidity_matrix_fixture(
        "validation"
    )

    payload = json.loads(
        alpha_max_runner._alpha_max_trend_liquidity_falsifier_artifact(matrix, buckets)
    )

    assert payload["row_id"] == "component_trend_1x"
    assert payload["nominal_cost_bps"] == 30
    assert payload["fold_run_sha256s"] == list(fold_hashes)
    assert tuple(payload["symbol_contribution_usdt"]) == buckets.admitted_symbols
    assert payload["symbol_contribution_usdt"] == {
        symbol: sum(fold[symbol] for fold in contributions) for symbol in buckets.admitted_symbols
    }
    assert not set(ALPHA_MAX_CANDIDATE_SYMBOLS[5:]) & set(payload["symbol_contribution_usdt"])


@pytest.mark.parametrize(
    ("domain", "fold_count"),
    (("validation", 12), ("historical_exposed_evaluation", 10)),
)
def test_trend_liquidity_falsifier_preserves_domain_fold_order_and_hashes(
    domain: str,
    fold_count: int,
) -> None:
    matrix, buckets, fold_hashes, _contributions = _exact_trend_liquidity_matrix_fixture(domain)

    payload = json.loads(
        alpha_max_runner._alpha_max_trend_liquidity_falsifier_artifact(matrix, buckets)
    )

    assert payload["domain"] == domain
    assert payload["fold_run_sha256s"] == list(fold_hashes)
    assert len(payload["fold_run_sha256s"]) == fold_count


def test_positive_trend_liquidity_falsifier_is_report_only_and_noncausal() -> None:
    matrix, buckets, _fold_hashes, _contributions = _exact_trend_liquidity_matrix_fixture(
        "validation"
    )

    payload = json.loads(
        alpha_max_runner._alpha_max_trend_liquidity_falsifier_artifact(matrix, buckets)
    )

    assert payload["status"] == "liquidity_falsifier_not_triggered"
    assert payload["rejection_reasons"] == []
    assert payload["report_only"] is True
    assert payload["selection_influence"] is False
    assert "causal" not in payload["status"]


def test_trend_liquidity_falsifier_is_wired_to_prelock_and_historical_artifact_paths() -> None:
    prelock_source = inspect.getsource(alpha_max_runner.run_alpha_max_prelock_process)
    historical_source = inspect.getsource(alpha_max_runner.run_alpha_max_historical_process)

    assert "diagnostics/validation/trend_liquidity_falsifier.json" in prelock_source
    assert "validation_trend_liquidity_falsifier" in prelock_source
    assert (
        "diagnostics/historical_exposed_evaluation/trend_liquidity_falsifier.json"
        in historical_source
    )
    assert "historical_trend_liquidity_falsifier" in historical_source
