"""Fail-closed, read-only validator for frozen ``cost_proof_v1`` evidence.

The validator consumes records only.  It never imports a data source, engine, router,
or order path; every identity used for a gate is recomputed from supplied evidence.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import math
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from lumina_quant.data.symbol_lifecycle import (
    validate_fold_membership_manifest,
    validate_symbol_lifecycle_registry,
)
from lumina_quant.research.router_replay import (
    CANDIDATE_IDS_SHA256,
    evaluate_router_replay,
)
from lumina_quant.research.survivorship import empirical_variance_across_trials
from lumina_quant.strategy_factory.research_metrics import (
    cscv_pbo,
    deflated_sharpe_ratio,
    max_drawdown,
    spa_like_pvalue,
)

SCHEMA = "cost_proof_v1"
COST_LADDER = (10, 15, 20, 30)
CSCV_SPLITS = 8
CANDIDATES = (
    "codex_lagged_leaf_router_grid:h4_avg1_tr-0.02_tmdd0.50_val0.00_vmdd0.25_lagged_plus_val025_exact_unscaled",
    "codex_lagged_leaf_router_grid:h4_avg1_tr-0.02_tmdd0.50_val0.00_vmdd0.25_lagged_plus_val025_fallback_mdd20_cap2",
)
HASH = re.compile(r"^[0-9a-f]{64}$")
EPS = 1e-9
FUNDING_HOURS = {0, 8, 16}
EXTERNAL_ARTIFACTS = (
    "profile",
    "source_data_manifest",
    "router_replay_manifest",
    "router_source_artifact",
    "lifecycle",
    "membership",
    "trial_ledger",
    "producer_source",
    "commit_receipt",
    "router_producer_source",
    "router_commit_receipt",
)
PROVENANCE_ARTIFACTS = {name: f"{name}_sha256" for name in EXTERNAL_ARTIFACTS} | {
    "verifier_source": "verifier_source_sha256"
}


@dataclass(frozen=True, slots=True)
class CostProofReport:
    status: str
    version: str
    reasons: tuple[str, ...]
    candidate_reports: tuple[dict[str, Any], ...]
    selected_candidate_id: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_reports": list(self.candidate_reports),
            "reasons": list(self.reasons),
            "selected_candidate_id": self.selected_candidate_id,
            "status": self.status,
            "version": self.version,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), allow_nan=False, separators=(",", ":"), sort_keys=True)


@dataclass(frozen=True, slots=True)
class ExternalBindings:
    """Authenticated bytes and parsed contracts supplied outside the proof."""

    hashes: Mapping[str, str]
    profile: Mapping[str, Any]
    source_manifest: Mapping[str, Any]
    router_manifest: Mapping[str, Any]
    membership: Mapping[str, Any]
    trial_ledger: Mapping[str, Any]
    market_artifact_hashes: frozenset[str]
    funding_artifact_hashes: frozenset[str]
    market_rows: Mapping[tuple[str, str, str], Mapping[str, Any]]
    funding_rows: Mapping[tuple[str, str, str], Mapping[str, Any]]


def _reject_constant(value: str) -> Any:
    raise ValueError(f"non-finite JSON constant: {value}")


def _no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


class _UniqueSafeLoader(yaml.SafeLoader):
    pass


def _unique_yaml_mapping(
    loader: _UniqueSafeLoader, node: yaml.MappingNode, deep: bool = False
) -> dict[Any, Any]:
    result: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise ValueError(f"duplicate YAML key: {key}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueSafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _unique_yaml_mapping
)


def _json_bytes(raw: bytes) -> Mapping[str, Any]:
    value = json.loads(
        raw.decode("utf-8"),
        object_pairs_hook=_no_duplicates,
        parse_constant=_reject_constant,
    )
    if not isinstance(value, Mapping):
        raise ValueError("JSON root must be an object")
    return value


def _source_contract(
    source: Mapping[str, Any],
) -> tuple[
    frozenset[str],
    frozenset[str],
    dict[tuple[str, str, str], Mapping[str, Any]],
    dict[tuple[str, str, str], Mapping[str, Any]],
]:
    if set(source) != {
        "schema",
        "synthetic_source_count",
        "actual_funding",
        "point_in_time_membership",
        "post_append_strict_receipt_sha256",
        "artifacts",
        "market_rows",
        "funding_rows",
    }:
        raise ValueError("invalid source-data manifest fields")
    if (
        source["schema"] != "cost_proof_source_data_v1"
        or source["synthetic_source_count"] != 0
        or type(source["synthetic_source_count"]) is not int
        or source["actual_funding"] is not True
        or source["point_in_time_membership"] is not True
        or not _hash(source["post_append_strict_receipt_sha256"])
    ):
        raise ValueError("unsafe source-data manifest")
    artifacts = _records(source["artifacts"])
    if not artifacts:
        raise ValueError("empty source-data manifest")
    by_kind: dict[str, set[str]] = {"market": set(), "funding": set()}
    seen: set[str] = set()
    for artifact in artifacts:
        if set(artifact) != {"kind", "artifact_sha256"}:
            raise ValueError("invalid source artifact fields")
        kind = artifact["kind"]
        digest = artifact["artifact_sha256"]
        if kind not in by_kind or not _hash(digest) or digest in seen:
            raise ValueError("invalid or duplicated source artifact")
        seen.add(digest)
        by_kind[kind].add(digest)
    if not by_kind["market"] or not by_kind["funding"]:
        raise ValueError("market and funding artifacts are required")

    market_source_rows = _records(source["market_rows"])
    funding_source_rows = _records(source["funding_rows"])
    if not market_source_rows or funding_source_rows is None:
        raise ValueError("source row collections are invalid")

    market_rows: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for row in market_source_rows:
        if set(row) != {
            "source_row_id",
            "artifact_sha256",
            "symbol",
            "timestamp",
            "prior_mark_price",
            "mark_price",
            "high",
            "low",
        }:
            raise ValueError("invalid market source row fields")
        key = (
            str(row.get("artifact_sha256")),
            str(row.get("source_row_id")),
            str(row.get("symbol")),
        )
        values = [
            _num(row.get(name), positive=True)
            for name in ("prior_mark_price", "mark_price", "high", "low")
        ]
        if (
            key in market_rows
            or key[0] not in by_kind["market"]
            or _string(row, "source_row_id") is None
            or _string(row, "symbol") is None
            or _utc(row.get("timestamp")) is None
            or any(value is None for value in values)
            or float(row["low"]) > min(float(row["prior_mark_price"]), float(row["mark_price"]))
            or float(row["high"]) < max(float(row["prior_mark_price"]), float(row["mark_price"]))
        ):
            raise ValueError("invalid or duplicated market source row")
        market_rows[key] = row

    funding_rows: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for row in funding_source_rows:
        if set(row) != {
            "source_row_id",
            "artifact_sha256",
            "symbol",
            "boundary",
            "observed_rate",
        }:
            raise ValueError("invalid funding source row fields")
        key = (
            str(row.get("artifact_sha256")),
            str(row.get("source_row_id")),
            str(row.get("symbol")),
        )
        if (
            key in funding_rows
            or key[0] not in by_kind["funding"]
            or _string(row, "source_row_id") is None
            or _string(row, "symbol") is None
            or _utc(row.get("boundary")) is None
            or _num(row.get("observed_rate")) is None
        ):
            raise ValueError("invalid or duplicated funding source row")
        funding_rows[key] = row
    return (
        frozenset(by_kind["market"]),
        frozenset(by_kind["funding"]),
        market_rows,
        funding_rows,
    )


def _artifact_bindings(paths: Mapping[str, str | Path]) -> ExternalBindings:
    if set(paths) != set(EXTERNAL_ARTIFACTS):
        raise ValueError("incomplete external artifact bindings")
    concrete = {name: Path(path) for name, path in paths.items()}
    raw = {name: path.read_bytes() for name, path in concrete.items()}
    profile = yaml.load(raw["profile"].decode("utf-8"), Loader=_UniqueSafeLoader)
    if not isinstance(profile, Mapping) or not _profile_ok(profile):
        raise ValueError("profile does not satisfy frozen contract")
    source = _json_bytes(raw["source_data_manifest"])
    market_hashes, funding_hashes, market_rows, funding_rows = _source_contract(source)
    lifecycle = validate_symbol_lifecycle_registry(_json_bytes(raw["lifecycle"]))
    membership = validate_fold_membership_manifest(lifecycle, _json_bytes(raw["membership"]))
    router_report = evaluate_router_replay(
        concrete["router_replay_manifest"],
        source_artifact_path=concrete["router_source_artifact"],
        lifecycle_registry_path=concrete["lifecycle"],
        membership_manifest_path=concrete["membership"],
        combined_profile_path=concrete["profile"],
        producer_source_path=concrete["router_producer_source"],
        commit_receipt_path=concrete["router_commit_receipt"],
    )
    if router_report.status != "PASS":
        raise ValueError("router replay manifest is not authenticated")
    hashes = {name: hashlib.sha256(content).hexdigest() for name, content in raw.items()}
    hashes["verifier_source"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return ExternalBindings(
        hashes=hashes,
        profile=profile,
        source_manifest=source,
        router_manifest=_json_bytes(raw["router_replay_manifest"]),
        membership=membership,
        trial_ledger=_json_bytes(raw["trial_ledger"]),
        market_artifact_hashes=market_hashes,
        funding_artifact_hashes=funding_hashes,
        market_rows=market_rows,
        funding_rows=funding_rows,
    )


def profile_sha256(profile: str | Path | bytes) -> str:
    data = profile if isinstance(profile, bytes) else Path(profile).read_bytes()
    return hashlib.sha256(data).hexdigest()


def candidate_ids_sha256(candidate_ids: tuple[str, ...] = CANDIDATES) -> str:
    return _canonical_sha256(list(candidate_ids))


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _report(
    status: str,
    reasons: list[str],
    candidates: list[dict[str, Any] | None] | None = None,
    selected: str | None = None,
) -> CostProofReport:
    reports = [] if candidates is None else [item for item in candidates if item is not None]
    return CostProofReport(status, SCHEMA, tuple(sorted(set(reasons))), tuple(reports), selected)


def _mapping(value: Any) -> Mapping[str, Any] | None:
    return value if isinstance(value, Mapping) else None


def _records(value: Any) -> list[Mapping[str, Any]] | None:
    if not isinstance(value, list) or not all(isinstance(item, Mapping) for item in value):
        return None
    return value


def _string(record: Mapping[str, Any], key: str) -> str | None:
    value = record.get(key)
    return value if isinstance(value, str) and value else None


def _num(value: Any, *, positive: bool = False) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except TypeError, ValueError, OverflowError:
        return None
    if not math.isfinite(number) or (positive and number <= 0):
        return None
    return number


def _close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=EPS, abs_tol=EPS)


def _utc(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.endswith("Z"):
        return None
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except TypeError, ValueError, OverflowError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() != UTC.utcoffset(parsed):
        return None
    return parsed


def _hash(value: Any) -> bool:
    return isinstance(value, str) and HASH.fullmatch(value) is not None


def _range(value: Any) -> tuple[datetime, datetime] | None:
    record = _mapping(value)
    if record is None:
        return None
    start = _utc(record.get("start"))
    end = _utc(record.get("end"))
    return (start, end) if start is not None and end is not None and start < end else None


def _profile_ok(profile: Any) -> bool:
    root = _mapping(profile)
    if root is None or root.get("profile") != "backtest_cost_realistic":
        return False
    research = _mapping(root.get("research"))
    execution = _mapping(root.get("execution"))
    risk = _mapping(root.get("risk"))
    data = _mapping(root.get("data"))
    live = _mapping(root.get("live"))
    if None in (research, execution, risk, data, live):
        return False
    assert (
        research is not None
        and execution is not None
        and risk is not None
        and data is not None
        and live is not None
    )
    required = {
        "strict_selection_gate": True,
        "use_lockbox_split": True,
        "purge_embargo_bars": 1,
        "single_correlation_discount": True,
        "hac_inference": True,
        "cscv_pbo": True,
        "exposure_normalized_promotion": True,
        "enforce_selection_reject_gate": True,
        "dsr_gate_floor": 0.90,
        "spa_gate_ceiling": 0.05,
        "pbo_gate_ceiling": 0.50,
        "max_cross_trial_pbo": 0.50,
        "route_unmapped_registered_strategies": True,
        "require_actual_engine_routing": True,
        "emit_candidate_overfit_stats": True,
        "portfolio_honest_gate": True,
    }
    if any(
        (research.get(key) is not expected)
        if isinstance(expected, bool)
        else (isinstance(research.get(key), bool) or research.get(key) != expected)
        for key, expected in required.items()
    ):
        return False
    kinds = data.get("kinds")
    adv_quote = _num(execution.get("slippage_adv_quote"))
    return (
        execution.get("slippage_impact_model") == "sqrt_impact"
        and _num(execution.get("slippage_impact_coefficient"), positive=True) is not None
        and adv_quote is not None
        and adv_quote >= 0
        and execution.get("require_funding_coverage") is True
        and execution.get("funding_on_utc_boundary") is True
        and execution.get("funding_interval_hours") == 8
        and type(execution.get("funding_interval_hours")) is int
        and risk.get("attach_default_protective_stop") is True
        and risk.get("enforce_order_risk_gate_in_backtest") is True
        and _num(execution.get("maintenance_margin_rate"), positive=True) is not None
        and _num(execution.get("liquidation_buffer_rate"), positive=True) is not None
        and isinstance(kinds, list)
        and "funding" in kinds
        and live.get("mode") == "paper"
        and live.get("testnet") is True
        and live.get("require_real_enable_flag") is True
        and live.get("allow_market_orders") is False
        and live.get("shadow_live_enabled") is False
    )


def _provenance(
    evidence: Mapping[str, Any], bindings: ExternalBindings | None, errors: list[str]
) -> None:
    provenance = _mapping(evidence.get("provenance"))
    if bindings is None or provenance is None:
        errors.append("missing trusted external bindings")
        return
    names = (*PROVENANCE_ARTIFACTS.values(), "candidate_ids_sha256")
    if set(provenance) != set(names) or any(not _hash(provenance.get(name)) for name in names):
        errors.append("invalid provenance hashes")
        return
    if provenance["candidate_ids_sha256"] != candidate_ids_sha256():
        errors.append("candidate list hash mismatch")
    for artifact, field in PROVENANCE_ARTIFACTS.items():
        if provenance[field] != bindings.hashes.get(artifact):
            errors.append(f"{artifact} SHA mismatch")


def _verify_tapes(scenario: Mapping[str, Any], errors: list[str]) -> tuple[str, str, str] | None:
    signals = _records(scenario.get("signal_position_tape"))
    orders = _records(scenario.get("orders"))
    fills = _records(scenario.get("fills"))
    if signals is None or orders is None or fills is None:
        errors.append("missing recomputable tapes")
        return None
    claimed = (
        scenario.get("signal_tape_sha256"),
        scenario.get("order_tape_sha256"),
        scenario.get("execution_tape_sha256"),
    )
    computed = (_canonical_sha256(signals), _canonical_sha256(orders), _canonical_sha256(fills))
    if not all(_hash(item) for item in claimed) or tuple(claimed) != computed:
        errors.append("claimed tape hash does not match records")
        return None
    return computed


def _layout(folds: list[Mapping[str, Any]]) -> tuple[Any, ...] | None:
    output: list[Any] = []
    for fold in folds:
        fold_id = _string(fold, "fold_id")
        periods = _records(fold.get("periods"))
        if fold_id is None or periods is None:
            return None
        output.append(
            (
                fold_id,
                tuple(
                    (_string(row, "period_id"), row.get("timestamp"), row.get("segment"))
                    for row in periods
                ),
            )
        )
    return tuple(output)


def _economic_tape(folds: list[Mapping[str, Any]]) -> list[dict[str, Any]] | None:
    """Project cost-ladder-invariant engine and safety evidence."""
    period_fields = (
        "period_id",
        "timestamp",
        "segment",
        "expected_funding",
        "gross_pnl",
        "impact_cost",
        "funding_cashflow",
        "active_protective_stop_ids",
        "maintenance_margin_required",
    )
    output: list[dict[str, Any]] = []
    for fold in folds:
        fold_id = _string(fold, "fold_id")
        periods = _records(fold.get("periods"))
        funding = _records(fold.get("funding"))
        stops = _records(fold.get("protective_stops"))
        if fold_id is None or periods is None or funding is None or stops is None:
            return None
        output.append(
            {
                "fold_id": fold_id,
                "router_execution_receipts_sha256": fold.get("router_execution_receipts_sha256"),
                "bar_interval_seconds": fold.get("bar_interval_seconds"),
                "evaluated_range": fold.get("evaluated_range"),
                "validation_range": fold.get("validation_range"),
                "locked_oos_range": fold.get("locked_oos_range"),
                "purge": fold.get("purge"),
                "embargo": fold.get("embargo"),
                "initial_equity": fold.get("initial_equity"),
                "periods": [
                    {field: period.get(field) for field in period_fields} for period in periods
                ],
                "funding": [dict(row) for row in funding],
                "protective_stops": [dict(row) for row in stops],
                "entry_count": fold.get("entry_count"),
                "protective_stop_count": fold.get("protective_stop_count"),
                "liquidation_count": fold.get("liquidation_count"),
                "ruin": fold.get("ruin"),
            }
        )
    return output


def _funding_ids_unique(folds: list[Mapping[str, Any]]) -> bool:
    settlements: set[str] = set()
    sources: set[tuple[str, str]] = set()
    for fold in folds:
        for row in _records(fold.get("funding")) or []:
            settlement, source = _string(row, "settlement_id"), _string(row, "source_row_id")
            artifact = row.get("source_artifact_sha256")
            if settlement is None or source is None or not isinstance(artifact, str):
                return False
            if settlement in settlements or (artifact, source) in sources:
                return False
            settlements.add(settlement)
            sources.add((artifact, source))
    return True


def _exact_fields(record: Mapping[str, Any], fields: set[str]) -> bool:
    return set(record) == fields


def _router_contract(
    bindings: ExternalBindings, candidate_id: str
) -> dict[str, dict[str, Any]] | None:
    folds = _records(bindings.router_manifest.get("folds"))
    if not folds:
        return None
    result: dict[str, dict[str, Any]] = {}
    for fold in folds:
        fold_id = _string(fold, "fold_id")
        selection = _mapping(fold.get("selection"))
        variants = _records(fold.get("variants"))
        if fold_id is None or selection is None or variants is None:
            return None
        variant = next((row for row in variants if row.get("variant_id") == candidate_id), None)
        leaves = _records(selection.get("leaves"))
        receipts = _records(variant.get("execution_receipts")) if variant else None
        locked = _mapping(fold.get("locked_oos"))
        if variant is None or leaves is None or receipts is None or locked is None:
            return None
        symbols: set[str] = set()
        modes: set[str] = set()
        for leaf in leaves:
            traded = leaf.get("traded_symbols")
            if not isinstance(traded, list) or not all(
                isinstance(symbol, str) and symbol for symbol in traded
            ):
                return None
            symbols.update(traded)
        for receipt in receipts:
            mode = _string(receipt, "evaluation_mode")
            if mode not in {"handler", "registry_simulator"}:
                return None
            modes.add(mode)
        result[fold_id] = {
            "execution_receipts_sha256": _canonical_sha256(receipts),
            "locked_oos_range": {
                "start": locked.get("start_utc"),
                "end": locked.get("end_utc"),
            },
            "modes": modes,
            "symbols": symbols,
        }
    return result


def _strict_grid(
    fold: Mapping[str, Any], expected_router: Mapping[str, Any]
) -> tuple[dict[str, datetime], list[Mapping[str, Any]], float] | None:
    evaluated = _range(fold.get("evaluated_range"))
    validation = _range(fold.get("validation_range"))
    locked = _range(fold.get("locked_oos_range"))
    periods = _records(fold.get("periods"))
    interval = fold.get("bar_interval_seconds")
    initial_equity = _num(fold.get("initial_equity"), positive=True)
    if (
        evaluated is None
        or validation is None
        or locked is None
        or not periods
        or isinstance(interval, bool)
        or not isinstance(interval, int)
        or interval <= 0
        or initial_equity is None
        or fold.get("router_execution_receipts_sha256")
        != expected_router["execution_receipts_sha256"]
        or fold.get("locked_oos_range") != expected_router["locked_oos_range"]
    ):
        return None
    step = timedelta(seconds=interval)
    purge = _mapping(fold.get("purge"))
    embargo = _mapping(fold.get("embargo"))
    if purge is None or embargo is None:
        return None
    removed: list[tuple[datetime, datetime, datetime]] = []
    for item in (purge, embargo):
        removed_range = _range(item.get("removed_range"))
        removed_rows = _records(item.get("removed_rows"))
        if (
            item.get("expected_count") != 1
            or type(item.get("expected_count")) is not int
            or removed_range is None
            or removed_range[1] - removed_range[0] != step
            or removed_rows is None
            or len(removed_rows) != 1
            or _utc(removed_rows[0].get("timestamp")) != removed_range[0]
        ):
            return None
        removed.append((removed_range[0], removed_range[1], removed_range[0]))
    if not (
        evaluated[0] == validation[0]
        and validation[1] == removed[0][0]
        and removed[0][1] == removed[1][0]
        and removed[1][1] == locked[0]
        and evaluated[1] == locked[1]
    ):
        return None
    duration = (evaluated[1] - evaluated[0]).total_seconds()
    if duration % interval:
        return None
    expected_grid = [evaluated[0] + index * step for index in range(int(duration // interval))]
    times: dict[str, datetime] = {}
    for period in periods:
        period_id = _string(period, "period_id")
        stamp = _utc(period.get("timestamp"))
        if period_id is None or stamp is None or period_id in times:
            return None
        times[period_id] = stamp
    observed_grid = sorted([*times.values(), removed[0][2], removed[1][2]])
    if (
        observed_grid != expected_grid
        or len(set(observed_grid)) != len(expected_grid)
        or list(times.values()) != sorted(times.values())
    ):
        return None
    return times, periods, initial_equity


def _strict_tapes(scenario: Mapping[str, Any], bindings: ExternalBindings) -> dict[str, Any] | None:
    signals = _records(scenario.get("signal_position_tape"))
    orders = _records(scenario.get("orders"))
    fills = _records(scenario.get("fills"))
    if signals is None or orders is None or fills is None:
        return None
    signal_fields = {
        "period_id",
        "timestamp",
        "symbol",
        "signal",
        "start_position",
        "position",
        "prior_mark_price",
        "mark_price",
        "high",
        "low",
        "gross_pnl",
        "market_data_artifact_sha256",
        "market_source_row_id",
    }
    signal_map: dict[tuple[str, str], Mapping[str, Any]] = {}
    market_rows: set[tuple[str, str, str]] = set()
    signal_sequence: list[tuple[datetime, str, str]] = []
    for signal in signals:
        period_id = _string(signal, "period_id")
        symbol = _string(signal, "symbol")
        artifact = signal.get("market_data_artifact_sha256")
        source_row = _string(signal, "market_source_row_id")
        timestamp = _utc(signal.get("timestamp"))
        values = [
            _num(signal.get(name))
            for name in (
                "signal",
                "start_position",
                "position",
                "prior_mark_price",
                "mark_price",
                "high",
                "low",
                "gross_pnl",
            )
        ]
        if (
            not _exact_fields(signal, signal_fields)
            or period_id is None
            or symbol is None
            or source_row is None
            or timestamp is None
            or artifact not in bindings.market_artifact_hashes
            or any(value is None for value in values)
        ):
            return None
        _, start, _, prior, mark, high, low, gross = values
        assert all(value is not None for value in values)
        source_key = (str(artifact), source_row, symbol)
        source_evidence = bindings.market_rows.get(source_key)
        if (
            (period_id, symbol) in signal_map
            or source_key in market_rows
            or source_evidence is None
            or source_evidence["timestamp"] != signal["timestamp"]
            or any(
                not _close(float(source_evidence[name]), float(signal[name]))
                for name in (
                    "prior_mark_price",
                    "mark_price",
                    "high",
                    "low",
                )
            )
            or prior <= 0
            or mark <= 0
            or low <= 0
            or high <= 0
            or low > min(prior, mark)
            or high < max(prior, mark)
            or not _close(gross, start * (mark - prior))
        ):
            return None
        signal_map[(period_id, symbol)] = signal
        market_rows.add(source_key)
        signal_sequence.append((timestamp, period_id, symbol))
    if signal_sequence != sorted(signal_sequence):
        return None

    order_fields = {
        "order_id",
        "period_id",
        "timestamp",
        "symbol",
        "signed_qty",
        "signed_quote_notional",
        "requested_qty",
        "direction",
        "order_type",
        "time_in_force",
        "is_maker",
        "is_entry",
        "protective_stop_id",
    }
    order_map: dict[str, Mapping[str, Any]] = {}
    order_sequence: list[tuple[datetime, str]] = []
    for order in orders:
        order_id = _string(order, "order_id")
        period_id = _string(order, "period_id")
        symbol = _string(order, "symbol")
        signed_qty = _num(order.get("signed_qty"))
        requested = _num(order.get("requested_qty"), positive=True)
        signed_quote = _num(order.get("signed_quote_notional"))
        direction = _string(order, "direction")
        order_type = _string(order, "order_type")
        stop_id = order.get("protective_stop_id")
        timestamp = _utc(order.get("timestamp"))
        if (
            not _exact_fields(order, order_fields)
            or None in (order_id, period_id, symbol, signed_qty, requested, signed_quote)
            or timestamp is None
            or order_id in order_map
            or (period_id, symbol) not in signal_map
            or signed_qty == 0
            or signed_quote == 0
            or not _close(abs(signed_qty), requested)
            or direction != ("BUY" if signed_qty > 0 else "SELL")
            or not isinstance(order.get("is_maker"), bool)
            or not isinstance(order.get("is_entry"), bool)
            or (order_type == "LMT") != order["is_maker"]
            or _string(order, "time_in_force") is None
            or (stop_id is not None and not isinstance(stop_id, str))
            or (order["is_entry"] and not stop_id)
        ):
            return None
        order_map[order_id] = order
        order_sequence.append((timestamp, order_id))
    if order_sequence != sorted(order_sequence):
        return None

    execution = _mapping(bindings.profile.get("execution"))
    if execution is None:
        return None
    coefficient = _num(execution.get("slippage_impact_coefficient"), positive=True)
    configured_adv = _num(execution.get("slippage_adv_quote"))
    if coefficient is None or configured_adv is None or configured_adv < 0:
        return None
    fill_fields = {
        "fill_id",
        "order_id",
        "period_id",
        "timestamp",
        "symbol",
        "is_entry",
        "signed_qty",
        "requested_qty",
        "direction",
        "fill_price",
        "signed_quote_notional",
        "is_maker",
        "bar_volume",
        "observed_adv_quote",
        "participation",
        "impact_coefficient",
        "sqrt_impact_rate",
        "sqrt_impact_cash_cost",
        "protective_stop_id",
        "protective_stop_source",
    }
    fill_map: dict[str, Mapping[str, Any]] = {}
    filled_orders: set[str] = set()
    deltas: dict[tuple[str, str], float] = {}
    notionals: dict[str, float] = {}
    impact: dict[str, float] = {}
    fill_sequence: list[tuple[datetime, str]] = []
    for fill in fills:
        fill_id = _string(fill, "fill_id")
        order_id = _string(fill, "order_id")
        order = order_map.get(order_id or "")
        quantity = _num(fill.get("signed_qty"))
        requested = _num(fill.get("requested_qty"), positive=True)
        price = _num(fill.get("fill_price"), positive=True)
        signed_quote = _num(fill.get("signed_quote_notional"))
        bar_volume = _num(fill.get("bar_volume"), positive=True)
        observed_adv = _num(fill.get("observed_adv_quote"), positive=True)
        participation = _num(fill.get("participation"), positive=True)
        observed_coefficient = _num(fill.get("impact_coefficient"), positive=True)
        impact_rate = _num(fill.get("sqrt_impact_rate"))
        impact_cash = _num(fill.get("sqrt_impact_cash_cost"))
        timestamp = _utc(fill.get("timestamp"))
        if (
            not _exact_fields(fill, fill_fields)
            or order is None
            or None
            in (
                fill_id,
                order_id,
                quantity,
                requested,
                price,
                signed_quote,
                bar_volume,
                observed_adv,
                participation,
                observed_coefficient,
                impact_rate,
                impact_cash,
                timestamp,
            )
            or fill_id in fill_map
            or order_id in filled_orders
        ):
            return None
        assert all(
            value is not None
            for value in (
                quantity,
                requested,
                price,
                signed_quote,
                bar_volume,
                observed_adv,
                participation,
                observed_coefficient,
                impact_rate,
                impact_cash,
            )
        )
        signal = signal_map[(order["period_id"], order["symbol"])]
        adv = configured_adv if configured_adv > 0 else bar_volume * price
        expected_participation = abs(signed_quote) / adv
        expected_rate = (
            0.0 if fill.get("is_maker") else coefficient * math.sqrt(expected_participation)
        )
        expected_impact = abs(signed_quote) * expected_rate
        if (
            fill.get("period_id") != order["period_id"]
            or fill.get("timestamp") != order["timestamp"]
            or fill.get("symbol") != order["symbol"]
            or fill.get("is_entry") != order["is_entry"]
            or fill.get("is_maker") != order["is_maker"]
            or fill.get("direction") != order["direction"]
            or fill.get("protective_stop_id") != order["protective_stop_id"]
            or not _close(quantity, float(order["signed_qty"]))
            or not _close(requested, float(order["requested_qty"]))
            or not _close(signed_quote, quantity * price)
            or not _close(float(order["signed_quote_notional"]), signed_quote)
            or not float(signal["low"]) <= price <= float(signal["high"])
            or not _close(observed_adv, adv)
            or not _close(participation, expected_participation)
            or not _close(observed_coefficient, coefficient)
            or not _close(impact_rate, expected_rate)
            or not _close(impact_cash, expected_impact)
            or (
                fill["is_entry"]
                and _string(fill, "protective_stop_source") not in {"engine_default", "strategy"}
            )
            or (
                not fill["is_entry"]
                and fill.get("protective_stop_source") is not None
                and _string(fill, "protective_stop_source") is None
            )
        ):
            return None
        key = (str(order["period_id"]), str(order["symbol"]))
        deltas[key] = deltas.get(key, 0.0) + quantity
        period_id = str(order["period_id"])
        notionals[period_id] = notionals.get(period_id, 0.0) + abs(signed_quote)
        impact[period_id] = impact.get(period_id, 0.0) + impact_cash
        fill_map[fill_id] = fill
        filled_orders.add(order_id)
        fill_sequence.append((timestamp, fill_id))
    if filled_orders != set(order_map) or fill_sequence != sorted(fill_sequence):
        return None
    return {
        "signals": signal_map,
        "orders": order_map,
        "fills": fill_map,
        "deltas": deltas,
        "notionals": notionals,
        "impact": impact,
    }


def _strict_stops(
    fold: Mapping[str, Any],
    times: Mapping[str, datetime],
    tape: Mapping[str, Any],
    positions: Mapping[str, tuple[dict[str, float], dict[str, float]]],
) -> bool:
    stop_rows = _records(fold.get("protective_stops"))
    if stop_rows is None:
        return False
    fields = {
        "stop_id",
        "symbol",
        "side",
        "quantity",
        "stop_price",
        "source",
        "activated_period_id",
        "deactivated_period_id",
        "trigger_fill_id",
    }
    stops: dict[str, Mapping[str, Any]] = {}
    allowed_symbols = {symbol for starts, _ in positions.values() for symbol in starts}
    for stop in stop_rows:
        stop_id = _string(stop, "stop_id")
        symbol = _string(stop, "symbol")
        side = _string(stop, "side")
        source = _string(stop, "source")
        activated = _string(stop, "activated_period_id")
        deactivated = stop.get("deactivated_period_id")
        trigger = stop.get("trigger_fill_id")
        quantity = _num(stop.get("quantity"), positive=True)
        price = _num(stop.get("stop_price"), positive=True)
        if (
            not _exact_fields(stop, fields)
            or None in (stop_id, symbol, side, source, activated, quantity, price)
            or stop_id in stops
            or symbol not in allowed_symbols
            or side not in {"BUY", "SELL"}
            or source not in {"engine_default", "strategy"}
            or activated not in times
            or (
                deactivated is not None
                and (
                    not isinstance(deactivated, str)
                    or deactivated not in times
                    or times[deactivated] < times[activated]
                )
            )
            or (
                trigger is not None
                and (
                    not isinstance(trigger, str)
                    or trigger not in tape["fills"]
                    or tape["fills"][trigger].get("protective_stop_id") != stop_id
                    or tape["fills"][trigger].get("is_entry") is not False
                )
            )
        ):
            return False
        stops[stop_id] = stop
    for fill in tape["fills"].values():
        if fill["period_id"] not in times or not fill["is_entry"]:
            continue
        stop = stops.get(str(fill["protective_stop_id"]))
        if (
            stop is None
            or stop["symbol"] != fill["symbol"]
            or stop["activated_period_id"] != fill["period_id"]
            or float(stop["quantity"]) + EPS < abs(float(fill["signed_qty"]))
        ):
            return False
    active_ids: set[str] = set()
    for period_id, (starts, ends) in positions.items():
        stamp = times[period_id]
        active = {
            stop_id: stop
            for stop_id, stop in stops.items()
            if times[str(stop["activated_period_id"])] <= stamp
            and (
                stop["deactivated_period_id"] is None
                or stamp <= times[str(stop["deactivated_period_id"])]
            )
        }
        claimed = next(
            period["active_protective_stop_ids"]
            for period in fold["periods"]
            if period["period_id"] == period_id
        )
        if (
            not isinstance(claimed, list)
            or len(claimed) != len(set(claimed))
            or set(claimed) != set(active)
        ):
            return False
        active_ids.update(active)
        signals = tape["signals"]
        for symbol, end in ends.items():
            matching = [
                (stop_id, stop) for stop_id, stop in active.items() if stop["symbol"] == symbol
            ]
            if abs(end) <= EPS:
                if matching:
                    return False
                continue
            if len(matching) != 1:
                return False
            stop_id, stop = matching[0]
            signal = signals[(period_id, symbol)]
            price = float(stop["stop_price"])
            if (
                float(stop["quantity"]) + EPS < abs(end)
                or (end > 0 and (stop["side"] != "SELL" or price >= signal["mark_price"]))
                or (end < 0 and (stop["side"] != "BUY" or price <= signal["mark_price"]))
            ):
                return False
            start = starts[symbol]
            crossed = (start > 0 and signal["low"] <= price) or (
                start < 0 and signal["high"] >= price
            )
            if crossed:
                trigger = stop.get("trigger_fill_id")
                if trigger is None or abs(end) >= abs(start) - EPS:
                    return False
                if tape["fills"][trigger]["period_id"] != period_id:
                    return False
                if stop_id not in active:
                    return False
    entry_count = sum(
        1 for fill in tape["fills"].values() if fill["period_id"] in times and fill["is_entry"]
    )
    return (
        type(fold.get("protective_stop_count")) is int
        and type(fold.get("entry_count")) is int
        and fold["protective_stop_count"] == len(active_ids)
        and fold["entry_count"] == entry_count
    )


def _strict_funding(
    fold: Mapping[str, Any],
    times: Mapping[str, datetime],
    tape: Mapping[str, Any],
    positions: Mapping[str, tuple[dict[str, float], dict[str, float]]],
    bindings: ExternalBindings,
) -> dict[str, float] | None:
    expected: dict[tuple[str, str], float] = {}
    for period in fold["periods"]:
        period_id = period["period_id"]
        stamp = times[period_id]
        expected_rows: list[dict[str, str]] = []
        if (
            stamp.hour in FUNDING_HOURS
            and stamp.minute == 0
            and stamp.second == 0
            and stamp.microsecond == 0
        ):
            starts, _ = positions[period_id]
            for symbol in sorted(starts):
                if abs(starts[symbol]) <= EPS:
                    continue
                signal = tape["signals"][(period_id, symbol)]
                expected[(period_id, symbol)] = starts[symbol] * float(signal["prior_mark_price"])
                expected_rows.append({"symbol": symbol, "boundary": period["timestamp"]})
        if period.get("expected_funding") != expected_rows:
            return None
    rows = _records(fold.get("funding"))
    if rows is None:
        return None
    fields = {
        "period_id",
        "symbol",
        "settlement_id",
        "source_row_id",
        "source_artifact_sha256",
        "boundary",
        "observed_rate",
        "signed_open_notional",
        "signed_cashflow",
    }
    observed: set[tuple[str, str]] = set()
    settlement_ids: set[str] = set()
    source_ids: set[tuple[str, str, str]] = set()
    cash: dict[str, float] = {}
    for row in rows:
        period_id = _string(row, "period_id")
        symbol = _string(row, "symbol")
        settlement = _string(row, "settlement_id")
        source = _string(row, "source_row_id")
        artifact = row.get("source_artifact_sha256")
        rate = _num(row.get("observed_rate"))
        notional = _num(row.get("signed_open_notional"))
        amount = _num(row.get("signed_cashflow"))
        key = (period_id or "", symbol or "")
        source_key = (str(artifact), source or "", symbol or "")
        source_evidence = bindings.funding_rows.get(source_key)
        if (
            not _exact_fields(row, fields)
            or None in (period_id, symbol, settlement, source, rate, notional, amount)
            or key not in expected
            or key in observed
            or settlement in settlement_ids
            or source_key in source_ids
            or artifact not in bindings.funding_artifact_hashes
            or row.get("boundary")
            != next(
                period["timestamp"]
                for period in fold["periods"]
                if period["period_id"] == period_id
            )
            or source_evidence is None
            or source_evidence["boundary"] != row.get("boundary")
            or not _close(float(source_evidence["observed_rate"]), rate)
            or not _close(notional, expected[key])
            or not _close(amount, -notional * rate)
        ):
            return None
        observed.add(key)
        settlement_ids.add(settlement)
        source_ids.add(source_key)
        cash[period_id] = cash.get(period_id, 0.0) + amount
    return cash if observed == set(expected) else None


def _strict_fold(
    fold: Mapping[str, Any],
    *,
    bps: int,
    expected_router: Mapping[str, Any],
    tape: Mapping[str, Any],
    bindings: ExternalBindings,
) -> (
    tuple[
        set[tuple[str, str]],
        set[str],
        set[str],
        list[str],
        list[tuple[str, str, float, float]],
        float,
    ]
    | None
):
    fold_fields = {
        "fold_id",
        "router_execution_receipts_sha256",
        "bar_interval_seconds",
        "evaluated_range",
        "validation_range",
        "locked_oos_range",
        "purge",
        "embargo",
        "initial_equity",
        "periods",
        "funding",
        "protective_stops",
        "entry_count",
        "protective_stop_count",
        "liquidation_count",
        "ruin",
        "equity",
    }
    grid = _strict_grid(fold, expected_router)
    if (
        not _exact_fields(fold, fold_fields)
        or grid is None
        or fold.get("liquidation_count") != 0
        or type(fold.get("liquidation_count")) is not int
        or fold.get("ruin") is not False
    ):
        return None
    times, periods, initial_equity = grid
    expected_symbols = set(expected_router["symbols"])
    used_signals: set[tuple[str, str]] = set()
    used_orders: set[str] = set()
    used_fills: set[str] = set()
    positions: dict[str, tuple[dict[str, float], dict[str, float]]] = {}
    previous: dict[str, float] = {}
    previous_segment: str | None = None
    for period in periods:
        period_id = str(period["period_id"])
        segment = period.get("segment")
        if segment not in {"validation", "locked_oos"}:
            return None
        signal_keys = {
            symbol for candidate_period, symbol in tape["signals"] if candidate_period == period_id
        }
        if signal_keys != expected_symbols:
            return None
        if segment != previous_segment:
            previous = dict.fromkeys(expected_symbols, 0.0)
        starts: dict[str, float] = {}
        ends: dict[str, float] = {}
        for symbol in expected_symbols:
            signal = tape["signals"][(period_id, symbol)]
            if signal["timestamp"] != period["timestamp"]:
                return None
            start = float(signal["start_position"])
            end = float(signal["position"])
            if not _close(start, previous.get(symbol, 0.0)) or not _close(
                end, start + tape["deltas"].get((period_id, symbol), 0.0)
            ):
                return None
            starts[symbol] = start
            ends[symbol] = end
            used_signals.add((period_id, symbol))
        positions[period_id] = (starts, ends)
        previous = ends
        previous_segment = str(segment)
    for order_id, order in tape["orders"].items():
        if order["period_id"] in times:
            signal = tape["signals"].get((order["period_id"], order["symbol"]))
            if signal is None or order["timestamp"] != signal["timestamp"]:
                return None
            used_orders.add(order_id)
    for fill_id, fill in tape["fills"].items():
        if fill["period_id"] in times:
            used_fills.add(fill_id)
    funding_cash = _strict_funding(
        fold,
        times,
        tape,
        positions,
        bindings,
    )
    if funding_cash is None or not _strict_stops(fold, times, tape, positions):
        return None
    execution = _mapping(bindings.profile.get("execution"))
    if execution is None:
        return None
    maintenance = _num(execution.get("maintenance_margin_rate"), positive=True)
    buffer = _num(execution.get("liquidation_buffer_rate"), positive=True)
    if maintenance is None or buffer is None:
        return None
    period_fields = {
        "period_id",
        "timestamp",
        "segment",
        "expected_funding",
        "gross_pnl",
        "linear_cost",
        "impact_cost",
        "funding_cashflow",
        "net_pnl",
        "prior_equity",
        "equity",
        "gross_exposure_fraction",
        "raw_net_return",
        "exposure_normalized_net_return",
        "position_notional",
        "active_protective_stop_ids",
        "worst_intrabar_equity",
        "maintenance_margin_required",
    }
    prior = initial_equity
    locked_ids: list[str] = []
    returns: list[tuple[str, str, float, float]] = []
    for period in periods:
        if not _exact_fields(period, period_fields):
            return None
        period_id = str(period["period_id"])
        starts, ends = positions[period_id]
        gross = sum(
            float(tape["signals"][(period_id, symbol)]["gross_pnl"]) for symbol in expected_symbols
        )
        linear = tape["notionals"].get(period_id, 0.0) * bps / 10_000
        impact = tape["impact"].get(period_id, 0.0)
        funding = funding_cash.get(period_id, 0.0)
        net = gross - linear - impact + funding
        equity = prior + net
        start_notional = sum(
            abs(starts[symbol] * float(tape["signals"][(period_id, symbol)]["prior_mark_price"]))
            for symbol in expected_symbols
        )
        end_notional = sum(
            abs(ends[symbol] * float(tape["signals"][(period_id, symbol)]["mark_price"]))
            for symbol in expected_symbols
        )
        exposure = max(start_notional, end_notional) / prior
        position_notional = sum(
            ends[symbol] * float(tape["signals"][(period_id, symbol)]["mark_price"])
            for symbol in expected_symbols
        )
        raw = net / prior
        normalized = raw / exposure if exposure > EPS else 0.0
        worst_pnl = 0.0
        maintenance_required = 0.0
        for symbol in expected_symbols:
            signal = tape["signals"][(period_id, symbol)]
            start = starts[symbol]
            if start > 0:
                worst_price = float(signal["low"])
            elif start < 0:
                worst_price = float(signal["high"])
            else:
                continue
            worst_pnl += start * (worst_price - float(signal["prior_mark_price"]))
            maintenance_required += abs(start * worst_price) * (maintenance + buffer)
        worst_equity = prior + worst_pnl
        claimed = [
            _num(period.get(name))
            for name in (
                "gross_pnl",
                "linear_cost",
                "impact_cost",
                "funding_cashflow",
                "net_pnl",
                "prior_equity",
                "equity",
                "gross_exposure_fraction",
                "raw_net_return",
                "exposure_normalized_net_return",
                "position_notional",
                "worst_intrabar_equity",
                "maintenance_margin_required",
            )
        ]
        expected = [
            gross,
            linear,
            impact,
            funding,
            net,
            prior,
            equity,
            exposure,
            raw,
            normalized,
            position_notional,
            worst_equity,
            maintenance_required,
        ]
        if (
            any(value is None for value in claimed)
            or any(
                not _close(float(actual), target)
                for actual, target in zip(claimed, expected, strict=True)
            )
            or prior <= 0
            or equity <= 0
            or worst_equity <= maintenance_required
            or worst_equity <= 0
        ):
            return None
        if exposure <= EPS and any(
            abs(value) > EPS for value in (gross, linear, impact, funding, net, raw)
        ):
            return None
        if period["segment"] == "locked_oos":
            locked_ids.append(period_id)
        prior = equity
        returns.append((period_id, str(period["segment"]), raw, normalized))
    final_equity = _num(fold.get("equity"), positive=True)
    if final_equity is None or not _close(final_equity, prior):
        return None
    locked_gain = sum(math.log1p(raw) for _, segment, raw, _ in returns if segment == "locked_oos")
    return (
        used_signals,
        used_orders,
        used_fills,
        locked_ids,
        returns,
        locked_gain,
    )


def _strict_engine_contract(
    scenario: Mapping[str, Any],
    candidate_id: str,
    bindings: ExternalBindings,
) -> tuple[list[str], list[tuple[list[tuple[str, str, float, float]], float, str]]] | None:
    scenario_fields = {
        "cost_bps",
        "evaluation_modes",
        "generic_fallback_proxy_count",
        "current_fold_oos_input_count",
        "router_replay_manifest_sha256",
        "membership_sha256",
        "signal_position_tape",
        "orders",
        "fills",
        "signal_tape_sha256",
        "order_tape_sha256",
        "execution_tape_sha256",
        "economic_tape_sha256",
        "folds",
    }
    if not _exact_fields(scenario, scenario_fields):
        return None
    tape = _strict_tapes(scenario, bindings)
    router = _router_contract(bindings, candidate_id)
    folds = _records(scenario.get("folds"))
    if tape is None or router is None or not folds:
        return None
    if [fold.get("fold_id") for fold in folds] != list(router):
        return None
    expected_modes = sorted({mode for fold in router.values() for mode in fold["modes"]})
    if scenario.get("evaluation_modes") != expected_modes:
        return None
    used_signals: set[tuple[str, str]] = set()
    used_orders: set[str] = set()
    used_fills: set[str] = set()
    locked_ids: list[str] = []
    parsed_folds: list[tuple[list[tuple[str, str, float, float]], float, str]] = []
    for fold in folds:
        fold_id = str(fold["fold_id"])
        result = _strict_fold(
            fold,
            bps=int(scenario["cost_bps"]),
            expected_router=router[fold_id],
            tape=tape,
            bindings=bindings,
        )
        if result is None:
            return None
        (
            fold_signals,
            fold_orders,
            fold_fills,
            fold_locked,
            fold_returns,
            locked_gain,
        ) = result
        if used_signals & fold_signals or used_orders & fold_orders or used_fills & fold_fills:
            return None
        used_signals.update(fold_signals)
        used_orders.update(fold_orders)
        used_fills.update(fold_fills)
        locked_ids.extend(fold_locked)
        parsed_folds.append((fold_returns, locked_gain, fold_id))
    if (
        used_signals != set(tape["signals"])
        or used_orders != set(tape["orders"])
        or used_fills != set(tape["fills"])
    ):
        return None
    return locked_ids, parsed_folds


def _trial_ledger(
    ledger: Mapping[str, Any], parsed: list[dict[str, Any]]
) -> tuple[np.ndarray, int] | None:
    required = {
        "schema",
        "cost_bps",
        "trial_ids",
        "locked_oos_period_ids",
        "normalized_returns_20bp",
        "raw_trial_count",
        "effective_trial_count",
        "current_fold_oos_input_count",
        "selection_receipt",
        "selection_receipt_sha256",
        "dedup_receipt",
        "dedup_receipt_sha256",
    }
    if (
        set(ledger) != required
        or ledger.get("schema") != "cost_proof_trial_ledger_v1"
        or ledger.get("cost_bps") != 20
        or ledger.get("current_fold_oos_input_count") != 0
        or type(ledger.get("current_fold_oos_input_count")) is not int
    ):
        return None
    ids = ledger.get("trial_ids")
    period_ids = ledger.get("locked_oos_period_ids")
    rows = ledger.get("normalized_returns_20bp")
    raw_count = ledger.get("raw_trial_count")
    effective_count = ledger.get("effective_trial_count")
    selection = _mapping(ledger.get("selection_receipt"))
    dedup = _mapping(ledger.get("dedup_receipt"))
    if (
        not isinstance(ids, list)
        or not all(isinstance(item, str) and item for item in ids)
        or len(ids) != len(set(ids))
        or not isinstance(period_ids, list)
        or len(period_ids) != len(set(period_ids))
        or not all(isinstance(item, str) and item for item in period_ids)
        or len(period_ids) < 16
        or len(period_ids) % CSCV_SPLITS
        or not isinstance(rows, list)
        or len(rows) != len(ids)
        or isinstance(raw_count, bool)
        or not isinstance(raw_count, int)
        or raw_count != len(ids)
        or isinstance(effective_count, bool)
        or not isinstance(effective_count, int)
        or effective_count <= 0
        or effective_count > raw_count
        or selection
        != {
            "candidate_ids": list(CANDIDATES),
            "post_oos_research_variant": True,
            "current_fold_oos_input_count": 0,
        }
        or dedup
        != {
            "input_trial_count": raw_count,
            "effective_trial_count": effective_count,
            "current_fold_oos_input_count": 0,
        }
        or selection["post_oos_research_variant"] is not True
        or type(selection["current_fold_oos_input_count"]) is not int
        or type(dedup["input_trial_count"]) is not int
        or type(dedup["effective_trial_count"]) is not int
        or type(dedup["current_fold_oos_input_count"]) is not int
        or ledger.get("selection_receipt_sha256") != _canonical_sha256(selection)
        or ledger.get("dedup_receipt_sha256") != _canonical_sha256(dedup)
    ):
        return None
    matrix_rows: list[list[float]] = []
    for row in rows:
        if not isinstance(row, list) or len(row) != len(period_ids):
            return None
        numeric = [_num(value) for value in row]
        if any(value is None for value in numeric):
            return None
        matrix_rows.append([float(value) for value in numeric if value is not None])
    by_id = dict(zip(ids, matrix_rows, strict=True))
    for item in parsed:
        if (
            item["locked_ids"] != period_ids
            or by_id.get(item["candidate_id"]) != item["normalized"]
        ):
            return None
    return np.asarray(matrix_rows, dtype=float), raw_count




def _candidate(
    candidate: Mapping[str, Any],
    provenance: Mapping[str, Any],
    bindings: ExternalBindings,
) -> tuple[dict[str, Any] | None, str | None]:
    candidate_id = _string(candidate, "candidate_id")
    scenarios = _records(candidate.get("scenarios"))
    if not _exact_fields(
        candidate,
        {
            "candidate_id",
            "router_replay_manifest_sha256",
            "membership_sha256",
            "scenarios",
        },
    ):
        return None, "invalid candidate fields"
    if (
        candidate_id is None
        or scenarios is None
        or tuple(item.get("cost_bps") for item in scenarios) != COST_LADDER
    ):
        return None, "scenario order/count mismatch"
    if (
        candidate.get("router_replay_manifest_sha256")
        != provenance["router_replay_manifest_sha256"]
        or candidate.get("membership_sha256") != provenance["membership_sha256"]
    ):
        return None, "candidate artifact binding mismatch"
    tapes: tuple[str, str, str] | None = None
    economic_tape_sha256: str | None = None
    layout: tuple[Any, ...] | None = None
    twenty: tuple[list[float], list[float], list[tuple[str, float]], list[float]] | None = None
    locked_ids_20bp: list[str] | None = None
    initial_equities_20bp: tuple[float, ...] | None = None
    for scenario in scenarios:
        modes = scenario.get("evaluation_modes")
        if (
            not isinstance(modes, list)
            or len(modes) != len(set(modes))
            or any(mode not in {"handler", "registry_simulator"} for mode in modes)
        ):
            return None, "invalid evaluation modes"
        if (
            scenario.get("generic_fallback_proxy_count") != 0
            or scenario.get("current_fold_oos_input_count") != 0
            or type(scenario.get("generic_fallback_proxy_count")) is not int
            or type(scenario.get("current_fold_oos_input_count")) is not int
        ):
            return None, "unsafe evaluation evidence"
        if (
            scenario.get("router_replay_manifest_sha256")
            != provenance["router_replay_manifest_sha256"]
            or scenario.get("membership_sha256") != provenance["membership_sha256"]
        ):
            return None, "scenario artifact binding mismatch"
        engine = _strict_engine_contract(scenario, candidate_id, bindings)
        if engine is None:
            return None, "engine ledger does not reconcile"
        locked_ids, parsed_folds = engine
        errors: list[str] = []
        current_tapes = _verify_tapes(scenario, errors)
        if current_tapes is None:
            return None, errors[0]
        if tapes is not None and current_tapes != tapes:
            return None, "same-candidate tape drift"
        tapes = current_tapes
        folds = _records(scenario.get("folds"))
        if not folds:
            return None, "missing folds or fills"
        economic_tape = _economic_tape(folds)
        claimed_economic_sha256 = scenario.get("economic_tape_sha256")
        if (
            economic_tape is None
            or not _hash(claimed_economic_sha256)
            or claimed_economic_sha256 != _canonical_sha256(economic_tape)
        ):
            return None, "economic tape hash does not match records"
        if economic_tape_sha256 is not None and claimed_economic_sha256 != economic_tape_sha256:
            return None, "same-candidate economic tape drift"
        economic_tape_sha256 = claimed_economic_sha256
        if not _funding_ids_unique(folds):
            return None, "reused funding settlement or source"
        ranges = [_range(fold.get("evaluated_range")) for fold in folds]
        if any(item is None for item in ranges):
            return None, "invalid fold ranges"
        concrete_ranges = [item for item in ranges if item is not None]
        if any(
            concrete_ranges[index - 1][1] > concrete_ranges[index][0]
            for index in range(1, len(concrete_ranges))
        ):
            return None, "fold ranges overlap or unordered"
        current_layout = _layout(folds)
        if current_layout is None or (layout is not None and current_layout != layout):
            return None, "scenario fold layout drift"
        layout = current_layout
        if scenario["cost_bps"] == 20:
            raw = [row[2] for fold in parsed_folds for row in fold[0] if row[1] == "locked_oos"]
            normalized = [
                row[3] for fold in parsed_folds for row in fold[0] if row[1] == "locked_oos"
            ]
            validation = [
                row[2] for fold in parsed_folds for row in fold[0] if row[1] == "validation"
            ]
            if not raw or not validation:
                return None, "empty lockbox segment"
            locked_ids_20bp = locked_ids
            initial_equities_20bp = tuple(float(fold["initial_equity"]) for fold in folds)
            twenty = (
                raw,
                normalized,
                [(fold[2], fold[1]) for fold in parsed_folds],
                validation,
            )
    if twenty is None:
        return None, "missing 20bp scenario"
    if locked_ids_20bp is None or initial_equities_20bp is None:
        return None, "missing locked-OOS identity"
    raw, normalized, folds, validation = twenty
    if len(raw) < 16 or len(raw) % CSCV_SPLITS or len(raw) != len(normalized):
        return None, "insufficient locked-OOS data"
    values = np.asarray(raw + normalized, dtype=float)
    if (
        not np.isfinite(values).all()
        or np.std(normalized, ddof=1) <= 0
        or any(value <= -1 for value in raw)
    ):
        return None, "nonfinite, constant, or invalid return"
    return {
        "candidate_id": candidate_id,
        "raw": raw,
        "normalized": normalized,
        "folds": folds,
        "validation": validation,
        "layout": layout,
        "initial_equities": initial_equities_20bp,
        "locked_ids": locked_ids_20bp,
    }, None


def evaluate_cost_proof(
    evidence: Mapping[str, Any], *, bindings: ExternalBindings | None = None
) -> CostProofReport:
    try:
        if bindings is None:
            return _report("STOP", ["missing trusted external bindings"])
        if (
            not isinstance(evidence, Mapping)
            or set(evidence)
            != {
                "schema",
                "candidate_ids",
                "cost_ladder_bps",
                "cscv_splits",
                "provenance",
                "candidates",
            }
            or evidence.get("schema") != SCHEMA
            or evidence.get("candidate_ids") != list(CANDIDATES)
            or candidate_ids_sha256() != CANDIDATE_IDS_SHA256
            or evidence.get("cost_ladder_bps") != list(COST_LADDER)
            or evidence.get("cscv_splits") != CSCV_SPLITS
        ):
            return _report("STOP", ["invalid schema, candidates, ladder, or CSCV split count"])
        errors: list[str] = []
        _provenance(evidence, bindings, errors)
        provenance = _mapping(evidence.get("provenance"))
        candidates = _records(evidence.get("candidates"))
        if (
            candidates is None
            or tuple(_string(candidate, "candidate_id") for candidate in candidates) != CANDIDATES
        ):
            errors.append("candidate order/count mismatch")
            candidates = []
        parsed: list[dict[str, Any]] = []
        if provenance is not None:
            for candidate in candidates:
                result, issue = _candidate(candidate, provenance, bindings)
                if issue is not None:
                    errors.append(f"{_string(candidate, 'candidate_id') or '?'}: {issue}")
                elif result is not None:
                    parsed.append(result)
        if errors:
            return _report("STOP", errors)
        if (
            len(parsed) != len(CANDIDATES)
            or parsed[0]["layout"] != parsed[1]["layout"]
            or parsed[0]["initial_equities"] != parsed[1]["initial_equities"]
        ):
            return _report("STOP", ["invalid aligned family layout"])
        checked_ledger = _trial_ledger(bindings.trial_ledger, parsed)
        if checked_ledger is None:
            return _report("STOP", ["invalid whole-search trial ledger"])
        matrix, raw_trial_count = checked_ledger
        if (
            matrix.shape[0] != raw_trial_count
            or matrix.shape[1] < 16
            or matrix.shape[1] % CSCV_SPLITS
            or not np.isfinite(matrix).all()
            or np.any(np.std(matrix, axis=1, ddof=1) <= 0)
        ):
            return _report("STOP", ["invalid aligned family matrix"])
        pbo = float(cscv_pbo(matrix, n_splits=CSCV_SPLITS))
        if not math.isfinite(pbo) or not 0 <= pbo <= 1:
            return _report("STOP", ["invalid CSCV result"])
        sharpes = np.mean(matrix, axis=1) / np.std(matrix, axis=1, ddof=1)
        variance = empirical_variance_across_trials(sharpes)
        reports: list[dict[str, Any]] = []
        for item in parsed:
            target = np.asarray(item["normalized"], dtype=float)
            dsr = float(
                deflated_sharpe_ratio(
                    target,
                    num_trials=float(raw_trial_count),
                    variance_across_trials=variance,
                    hac_inference=True,
                )
            )
            spa_pvalue = float(spa_like_pvalue(target, seed=12345))
            gate_passed = dsr >= 0.90 and spa_pvalue <= 0.05 and pbo <= 0.50
            raw = item["raw"]
            reasons: list[str] = []
            net = math.prod(1 + value for value in raw) - 1
            mdd = max_drawdown(np.asarray(raw, dtype=float))
            ordered = sorted(item["folds"], key=lambda value: (-value[1], value[0]))
            leave_best = math.expm1(sum(value[1] for value in ordered[1:]))
            positive = [gain for _, gain in item["folds"] if gain > 0]
            if net <= 0:
                reasons.append("20bp net not positive")
            if not math.isfinite(mdd) or mdd > 0.30:
                reasons.append("20bp MDD exceeds .30")
            if leave_best <= 0:
                reasons.append("leave-best-fold-out not positive")
            dominance = max(positive) / sum(positive) if positive else 0.0
            if not positive or dominance >= 0.50:
                reasons.append("positive-fold dominance >= .50")
            if not gate_passed:
                reasons.append("DSR/SPA/PBO gate failed")
            validation = item["validation"]
            validation_return = math.prod(1 + value for value in validation) - 1
            validation_mdd = max_drawdown(np.asarray(validation, dtype=float))
            calmar = validation_return / max(validation_mdd, EPS)
            metrics = {
                "calmar_validation": calmar,
                "dsr": dsr,
                "leave_best_fold_out_return": leave_best,
                "mdd_20bp": mdd,
                "net_20bp": net,
                "pbo": pbo,
                "positive_fold_dominance_share": dominance,
                "spa_pvalue": spa_pvalue,
                "validation_mdd": validation_mdd,
                "validation_return": validation_return,
                "raw_trial_count": float(raw_trial_count),
                "effective_trial_count": float(bindings.trial_ledger["effective_trial_count"]),
            }
            if not all(math.isfinite(value) for value in metrics.values()):
                return _report("STOP", ["nonfinite report metric"])
            reports.append(
                {
                    "candidate_id": item["candidate_id"],
                    "metrics": metrics,
                    "reasons": sorted(set(reasons)),
                    "status": "PASS" if not reasons else "REJECT",
                }
            )
        winners = [report for report in reports if report["status"] == "PASS"]
        if not winners:
            return _report("REJECT", ["no candidate passed"], reports)
        order = {candidate: index for index, candidate in enumerate(CANDIDATES)}
        selected = min(
            winners,
            key=lambda report: (
                -report["metrics"]["calmar_validation"],
                report["metrics"]["mdd_20bp"],
                order[report["candidate_id"]],
            ),
        )
        return _report("PASS", [], reports, selected["candidate_id"])
    except (
        ArithmeticError,
        AttributeError,
        KeyError,
        OverflowError,
        RecursionError,
        TypeError,
        ValueError,
    ):
        return _report("STOP", ["malformed evidence"])


def evaluate_cost_proof_file(
    input_path: str | Path,
    profile_path: str | Path,
    *,
    source_data_manifest_path: str | Path | None = None,
    router_replay_manifest_path: str | Path | None = None,
    router_source_artifact_path: str | Path | None = None,
    lifecycle_path: str | Path | None = None,
    membership_path: str | Path | None = None,
    trial_ledger_path: str | Path | None = None,
    producer_source_path: str | Path | None = None,
    commit_receipt_path: str | Path | None = None,
    router_producer_source_path: str | Path | None = None,
    router_commit_receipt_path: str | Path | None = None,
) -> CostProofReport:
    paths = {
        "profile": profile_path,
        "source_data_manifest": source_data_manifest_path,
        "router_replay_manifest": router_replay_manifest_path,
        "router_source_artifact": router_source_artifact_path,
        "lifecycle": lifecycle_path,
        "membership": membership_path,
        "trial_ledger": trial_ledger_path,
        "producer_source": producer_source_path,
        "commit_receipt": commit_receipt_path,
        "router_producer_source": router_producer_source_path,
        "router_commit_receipt": router_commit_receipt_path,
    }
    if any(path is None for path in paths.values()):
        return _report("STOP", ["missing trusted external bindings"])
    try:
        bindings = _artifact_bindings(paths)  # type: ignore[arg-type]
        evidence = _json_bytes(Path(input_path).read_bytes())
    except (
        OSError,
        RecursionError,
        TypeError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        yaml.YAMLError,
        ValueError,
    ):
        return _report("STOP", ["unreadable evidence or external binding"])
    return evaluate_cost_proof(evidence, bindings=bindings)
