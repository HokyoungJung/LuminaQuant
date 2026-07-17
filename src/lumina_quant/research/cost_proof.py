"""Fail-closed, read-only validator for frozen ``cost_proof_v2`` evidence.

All authority enters through caller-supplied, out-of-band raw-byte SHA-256 roots.
Evidence-produced digest strings are only cross-references; they are never trust roots.
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
)

SCHEMA = "cost_proof_v2"
PROFILE_RAW_SHA256 = "a3e9572365d39e3388c97b8b6c094c0bb9d63a3b1fd6d38c918342b435716950"
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
    "source_run_receipt",
    "search_run_receipt",
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
# The cost commit is deliberately excluded: it is created after canonical evidence
# and binds that evidence hash. Including it would create an unconstructable cycle.
PROVENANCE_ARTIFACTS = {
    name: f"{name}_sha256" for name in EXTERNAL_ARTIFACTS if name != "commit_receipt"
} | {"verifier_source": "verifier_source_sha256"}


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
    source_run_receipt: Mapping[str, Any]
    search_run_receipt: Mapping[str, Any]
    cost_commit: Mapping[str, Any]
    router_manifest: Mapping[str, Any]
    membership: Mapping[str, Any]
    trial_ledger: Mapping[str, Any]
    trial_result_artifacts: Mapping[str, Mapping[str, Any]]
    market_artifact_hashes: frozenset[str]
    funding_artifact_hashes: frozenset[str]
    market_rows: Mapping[tuple[str, str, str], Mapping[str, Any]]
    funding_rows: Mapping[tuple[str, str, str], Mapping[str, Any]]
    router_tapes: Mapping[tuple[str, str, str, int], Mapping[str, Mapping[str, Any]]]
    trusted_roots: Mapping[str, str]


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
    receipt: Mapping[str, Any],
    *,
    manifest_sha256: str,
    artifacts: Mapping[str, tuple[str, int, str, str]],
) -> None:
    fields = {
        "schema",
        "source_run_id",
        "synthetic_source_count",
        "actual_funding",
        "point_in_time_membership",
        "post_append_strict_receipt_sha256",
        "artifacts",
    }
    if (
        set(source) != fields
        or source.get("schema") != "cost_proof_source_data_v2"
        or _string(source, "source_run_id") is None
        or source.get("synthetic_source_count") != 0
        or type(source.get("synthetic_source_count")) is not int
        or source.get("actual_funding") is not True
        or source.get("point_in_time_membership") is not True
    ):
        raise ValueError("unsafe source-data manifest")
    projection = _records(source["artifacts"])
    row_fields = {"kind", "artifact_sha256", "row_count", "min_timestamp_utc", "max_timestamp_utc"}
    if not projection or any(set(row) != row_fields for row in projection):
        raise ValueError("invalid source artifact projection")
    for row in projection:
        digest, parsed = row.get("artifact_sha256"), artifacts.get(row.get("artifact_sha256"))
        if (
            row.get("kind") not in {"market", "funding"}
            or not _hash(digest)
            or parsed is None
            or row["kind"] != parsed[0]
            or type(row.get("row_count")) is not int
            or row["row_count"] != parsed[1]
            or row.get("min_timestamp_utc") != parsed[2]
            or row.get("max_timestamp_utc") != parsed[3]
            or _utc(parsed[2]) is None
            or _utc(parsed[3]) is None
            or parsed[2] > parsed[3]
        ):
            raise ValueError("source artifact count/range drift")
    if (
        len(projection) != len(artifacts)
        or len({row["artifact_sha256"] for row in projection}) != len(projection)
        or not {row["kind"] for row in projection} == {"market", "funding"}
    ):
        raise ValueError("source artifact coverage drift")
    receipt_fields = {
        "schema",
        "source_run_id",
        "manifest_sha256",
        "artifacts",
        "producer_source_sha256",
        "source_commit_sha256",
        "committed_at_utc",
    }
    if (
        set(receipt) != receipt_fields
        or receipt.get("schema") != "cost_proof_source_run_receipt_v1"
        or receipt.get("source_run_id") != source["source_run_id"]
        or receipt.get("manifest_sha256") != manifest_sha256
        or receipt.get("artifacts") != projection
        or not _hash(receipt.get("producer_source_sha256"))
        or not _hash(receipt.get("source_commit_sha256"))
        or _utc(receipt.get("committed_at_utc")) is None
        or source.get("post_append_strict_receipt_sha256") != receipt.get("source_commit_sha256")
    ):
        raise ValueError("source-run receipt binding drift")


def _trusted_roots(roots: Mapping[str, str]) -> dict[str, str]:
    required = {
        "source_data_commit_sha256",
        "search_run_receipt_sha256",
        "cost_proof_commit_sha256",
        "router_source_artifact_sha256",
        "router_commit_receipt_sha256",
    }
    if set(roots) != required or any(not _hash(value) for value in roots.values()):
        raise ValueError("incomplete trusted SHA-256 roots")
    return dict(roots)


def _canonical_artifact(raw: bytes) -> Mapping[str, Any]:
    value = _json_bytes(raw)
    if raw != json.dumps(
        value, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8"):
        raise ValueError("artifact JSON is not canonical")
    return value


def _artifact_bindings(
    paths: Mapping[str, str | Path],
    *,
    market_artifact_paths: Mapping[str, str | Path],
    funding_artifact_paths: Mapping[str, str | Path],
    router_artifact_paths: Mapping[str, str | Path],
    trial_result_artifact_paths: Mapping[str, str | Path],
    evidence_sha256: str,
    trusted_roots: Mapping[str, str],
) -> ExternalBindings:
    if set(paths) != set(EXTERNAL_ARTIFACTS):
        raise ValueError("incomplete external artifact bindings")
    roots = _trusted_roots(trusted_roots)
    concrete = {name: Path(path) for name, path in paths.items()}
    raw = {name: path.read_bytes() for name, path in concrete.items()}
    if hashlib.sha256(raw["profile"]).hexdigest() != PROFILE_RAW_SHA256:
        raise ValueError("profile raw SHA-256 mismatch")
    profile = yaml.load(raw["profile"].decode("utf-8"), Loader=_UniqueSafeLoader)
    if not isinstance(profile, Mapping) or not _profile_ok(profile):
        raise ValueError("profile does not satisfy frozen contract")
    source = _canonical_artifact(raw["source_data_manifest"])
    source_receipt = _canonical_artifact(raw["source_run_receipt"])
    search_receipt = _canonical_artifact(raw["search_run_receipt"])
    if (
        hashlib.sha256(raw["router_source_artifact"]).hexdigest()
        != roots["router_source_artifact_sha256"]
    ):
        raise ValueError("Router source trusted root mismatch")
    if (
        hashlib.sha256(raw["router_commit_receipt"]).hexdigest()
        != roots["router_commit_receipt_sha256"]
    ):
        raise ValueError("Router commit trusted root mismatch")
    if hashlib.sha256(raw["commit_receipt"]).hexdigest() != roots["cost_proof_commit_sha256"]:
        raise ValueError("cost-proof commit trusted root mismatch")
    if hashlib.sha256(raw["search_run_receipt"]).hexdigest() != roots["search_run_receipt_sha256"]:
        raise ValueError("search-run receipt trusted root mismatch")
    if hashlib.sha256(raw["source_run_receipt"]).hexdigest() != roots["source_data_commit_sha256"]:
        raise ValueError("source-data commit trusted root mismatch")
    lifecycle = validate_symbol_lifecycle_registry(_canonical_artifact(raw["lifecycle"]))
    membership = validate_fold_membership_manifest(
        lifecycle, _canonical_artifact(raw["membership"])
    )
    market_raw = {digest: Path(path).read_bytes() for digest, path in market_artifact_paths.items()}
    funding_raw = {
        digest: Path(path).read_bytes() for digest, path in funding_artifact_paths.items()
    }
    if (
        not market_raw
        or not funding_raw
        or any(
            not _hash(digest) or hashlib.sha256(content).hexdigest() != digest
            for digest, content in {**market_raw, **funding_raw}.items()
        )
        or set(market_raw) & set(funding_raw)
    ):
        raise ValueError("invalid source artifact path map")

    def source_rows(
        contents: Mapping[str, bytes], kind: str
    ) -> tuple[dict[tuple[str, str, str], Mapping[str, Any]], dict[str, tuple[str, int, str, str]]]:
        result: dict[tuple[str, str, str], Mapping[str, Any]] = {}
        stats: dict[str, tuple[str, int, str, str]] = {}
        global_keys: set[tuple[str, str]] = set()
        for digest, content in contents.items():
            artifact = _canonical_artifact(content)
            if (
                set(artifact) != {"schema", "rows"}
                or artifact["schema"] != f"cost_proof_{kind}_artifact_v1"
            ):
                raise ValueError("source artifact schema mismatch")
            rows = _records(artifact["rows"])
            if not rows:
                raise ValueError("empty source artifact")
            last: tuple[str, str] | None = None
            for row in rows:
                required = (
                    {
                        "source_row_id",
                        "symbol",
                        "timestamp",
                        "prior_mark_price",
                        "mark_price",
                        "open",
                        "high",
                        "low",
                        "close",
                        "bar_volume_base",
                        "price_tick_size",
                        "quantity_step_size",
                    }
                    if kind == "market"
                    else {"source_row_id", "symbol", "boundary", "observed_rate"}
                )
                time_key = "timestamp" if kind == "market" else "boundary"
                stamp, symbol = _utc(row.get(time_key)), _string(row, "symbol")
                key = (digest, _string(row, "source_row_id") or "", symbol or "")
                chronology = (symbol or "", str(row.get(time_key)))
                if (
                    set(row) != required
                    or key in result
                    or not key[1]
                    or stamp is None
                    or symbol is None
                    or chronology in global_keys
                    or (last is not None and chronology <= last)
                ):
                    raise ValueError("invalid, duplicate, or unordered source row")
                global_keys.add(chronology)
                last = chronology
                numbers = (
                    (
                        "prior_mark_price",
                        "mark_price",
                        "open",
                        "high",
                        "low",
                        "close",
                        "bar_volume_base",
                        "price_tick_size",
                        "quantity_step_size",
                    )
                    if kind == "market"
                    else ("observed_rate",)
                )
                if any(
                    _num(row.get(name), positive=(name != "observed_rate")) is None
                    for name in numbers
                ):
                    raise ValueError("nonfinite source row")
                if kind == "market" and (
                    float(row["low"])
                    > min(
                        float(row["open"]),
                        float(row["close"]),
                        float(row["prior_mark_price"]),
                        float(row["mark_price"]),
                    )
                    or float(row["high"])
                    < max(
                        float(row["open"]),
                        float(row["close"]),
                        float(row["prior_mark_price"]),
                        float(row["mark_price"]),
                    )
                ):
                    raise ValueError("market OHLC containment drift")
                if kind == "funding" and (
                    stamp.hour not in FUNDING_HOURS
                    or stamp.minute
                    or stamp.second
                    or stamp.microsecond
                ):
                    raise ValueError("funding boundary drift")
                result[key] = row
            times = [str(row[time_key]) for key, row in result.items() if key[0] == digest]
            stats[digest] = (kind, len(times), min(times), max(times))
        return result, stats

    market_rows, market_stats = source_rows(market_raw, "market")
    funding_rows, funding_stats = source_rows(funding_raw, "funding")
    _source_contract(
        source,
        source_receipt,
        manifest_sha256=hashlib.sha256(raw["source_data_manifest"]).hexdigest(),
        artifacts=market_stats | funding_stats,
    )
    if [row["artifact_sha256"] for row in source["artifacts"]] != [*market_raw, *funding_raw]:
        raise ValueError("source artifact ordered coverage mismatch")
    router_raw = {digest: Path(path).read_bytes() for digest, path in router_artifact_paths.items()}
    if not router_raw or any(
        not _hash(digest) or hashlib.sha256(content).hexdigest() != digest
        for digest, content in router_raw.items()
    ):
        raise ValueError("invalid Router artifact path map")
    router_report = evaluate_router_replay(
        concrete["router_replay_manifest"],
        source_artifact_path=concrete["router_source_artifact"],
        lifecycle_registry_path=concrete["lifecycle"],
        membership_manifest_path=concrete["membership"],
        combined_profile_path=concrete["profile"],
        producer_source_path=concrete["router_producer_source"],
        commit_receipt_path=concrete["router_commit_receipt"],
        trusted_source_artifact_sha256=roots["router_source_artifact_sha256"],
        trusted_commit_receipt_sha256=roots["router_commit_receipt_sha256"],
        artifact_paths=router_artifact_paths,
    )
    if router_report.status != "PASS":
        raise ValueError("router replay manifest is not authenticated")
    router_commit = _canonical_artifact(raw["router_commit_receipt"])
    index_rows = _records(router_commit.get("artifact_index"))
    if index_rows is None:
        raise ValueError("invalid Router commit artifact index")
    committed_kinds: dict[str, str] = {}
    for row in index_rows:
        if (
            set(row) != {"kind", "sha256"}
            or not _hash(row.get("sha256"))
            or not _string(row, "kind")
        ):
            raise ValueError("invalid Router commit artifact index row")
        digest, kind = row["sha256"], row["kind"]
        if digest in committed_kinds:
            raise ValueError("duplicate Router commit artifact digest")
        committed_kinds[digest] = kind
    cost_kinds = {
        "cost_tape_receipt",
        "cost_signal_position_tape",
        "cost_order_tape",
        "cost_fill_tape",
        "cost_event_tape",
        "cost_base_tape_projection",
    }
    committed_cost = {digest for digest, kind in committed_kinds.items() if kind in cost_kinds}
    if not committed_cost or not committed_cost <= set(router_raw):
        raise ValueError("missing committed Router cost artifact")
    router_tapes: dict[tuple[str, str, str, int], Mapping[str, Mapping[str, Any]]] = {}
    consumed_router: set[str] = set()
    tape_fields = {
        "schema",
        "fold_id",
        "variant_id",
        "selected_label",
        "leaf_id",
        "source_row_sha256",
        "params_sha256",
        "engine_receipt_sha256",
        "signal_receipt_sha256",
        "position_receipt_sha256",
        "tapes",
    }
    expected_kinds = {
        "signal_position": ("signal_position_sha256", "cost_signal_position_tape"),
        "order": ("order_tape_sha256", "cost_order_tape"),
        "fill": ("fill_tape_sha256", "cost_fill_tape"),
        "event": ("event_tape_sha256", "cost_event_tape"),
    }
    router_cost_owners: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    router_cost_receipts: set[str] = set()
    router_folds = _records(_canonical_artifact(raw["router_replay_manifest"]).get("folds"))
    if router_folds is None:
        raise ValueError("invalid Router fold ownership")
    for fold in router_folds:
        fold_id = _string(fold, "fold_id")
        selection = _mapping(fold.get("selection"))
        variants = _records(fold.get("variants"))
        leaves = _records(selection.get("leaves")) if selection is not None else None
        if fold_id is None or variants is None or leaves is None:
            raise ValueError("invalid Router ownership")
        bases = {_string(leaf, "leaf_id"): leaf for leaf in leaves}
        if None in bases or len(bases) != len(leaves):
            raise ValueError("invalid Router selected leaves")
        for variant in variants:
            variant_id = _string(variant, "variant_id")
            executions = _records(variant.get("execution_receipts"))
            if variant_id is None or executions is None or len(executions) != len(leaves):
                raise ValueError("invalid Router execution ownership")
            for execution in executions:
                leaf_id = _string(execution, "leaf_id")
                receipt_digest = execution.get("cost_tape_receipt_sha256")
                base = bases.get(leaf_id)
                key = (fold_id, variant_id, leaf_id or "")
                if (
                    base is None
                    or key in router_cost_owners
                    or not _hash(receipt_digest)
                    or receipt_digest in router_cost_receipts
                ):
                    raise ValueError("duplicate Router cost ownership")
                router_cost_owners[key] = {
                    "base": base,
                    "execution": execution,
                    "selected_label": variant.get("selected_label"),
                }
                router_cost_receipts.add(str(receipt_digest))
    for digest in sorted(committed_cost):
        if committed_kinds[digest] != "cost_tape_receipt":
            continue
        receipt = _canonical_artifact(router_raw[digest])
        if (
            receipt.get("schema") != "router_cost_tape_receipt_v1"
            or set(receipt) != tape_fields
            or not all(
                _string(receipt, key) is not None
                for key in ("fold_id", "variant_id", "leaf_id", "engine_receipt_sha256")
            )
        ):
            raise ValueError("invalid Router cost tape receipt")
        owner = router_cost_owners.get(
            (receipt["fold_id"], receipt["variant_id"], receipt["leaf_id"])
        )
        if (
            owner is None
            or digest not in router_cost_receipts
            or owner["execution"].get("cost_tape_receipt_sha256") != digest
            or receipt.get("source_row_sha256") != owner["base"].get("source_row_sha256")
            or receipt.get("params_sha256") != owner["base"].get("params_sha256")
            or receipt.get("selected_label") != owner["selected_label"]
            or any(
                receipt.get(name) != owner["execution"].get(name)
                for name in (
                    "engine_receipt_sha256",
                    "signal_receipt_sha256",
                    "position_receipt_sha256",
                )
            )
        ):
            raise ValueError("Router cost receipt ownership drift")
        tapes = _records(receipt["tapes"])
        if tapes is None or len(tapes) != len(COST_LADDER):
            raise ValueError("Router cost tape count mismatch")
        consumed_router.add(digest)
        base_commitments: dict[str, str] = {}
        for row, cost_bps in zip(tapes, COST_LADDER, strict=True):
            fields = {
                "cost_bps",
                "signal_position_sha256",
                "order_tape_sha256",
                "fill_tape_sha256",
                "event_tape_sha256",
            }
            if (
                set(row) != fields
                or type(row.get("cost_bps")) is not int
                or row["cost_bps"] != cost_bps
            ):
                raise ValueError("Router cost tape projection mismatch")
            bundle: dict[str, Mapping[str, Any]] = {"receipt": receipt}
            for name, (field, expected_kind) in expected_kinds.items():
                artifact_digest = row.get(field)
                if (
                    not _hash(artifact_digest)
                    or artifact_digest not in committed_cost
                    or committed_kinds.get(artifact_digest) != expected_kind
                ):
                    raise ValueError("Router tape committed-kind mismatch")
                bundle[name] = _canonical_artifact(router_raw[artifact_digest])
                consumed_router.add(artifact_digest)
                artifact = bundle[name]
                artifact_schema = {
                    "signal_position": "router_cost_signal_position_tape_v1",
                    "order": "router_cost_order_tape_v1",
                    "fill": "router_cost_fill_tape_v1",
                    "event": "router_cost_event_tape_v1",
                }[name]
                artifact_fields = {
                    "schema",
                    "cost_cell",
                    "cost_bps",
                    "fold_id",
                    "variant_id",
                    "leaf_id",
                    "engine_receipt_sha256",
                    "sequence",
                    "sequence_sha256",
                    "rows",
                    "rows_sha256",
                    "base_tape_projection_sha256",
                }
                if (
                    set(artifact) != artifact_fields
                    or artifact.get("schema") != artifact_schema
                    or artifact.get("cost_cell") != f"{cost_bps}bps"
                    or type(artifact.get("cost_bps")) is not int
                    or artifact["cost_bps"] != cost_bps
                    or not isinstance(artifact.get("sequence"), list)
                    or not artifact["sequence"]
                    or len(artifact["sequence"]) != len(set(artifact["sequence"]))
                    or any(type(value) is not str or not value for value in artifact["sequence"])
                    or not isinstance(artifact.get("rows"), list)
                    or len(artifact["rows"]) != len(artifact["sequence"])
                    or _canonical_sha256(artifact["sequence"]) != artifact.get("sequence_sha256")
                    or _canonical_sha256(artifact["rows"]) != artifact.get("rows_sha256")
                    or artifact.get("fold_id") != receipt["fold_id"]
                    or artifact.get("variant_id") != receipt["variant_id"]
                    or artifact.get("leaf_id") != receipt["leaf_id"]
                    or artifact.get("engine_receipt_sha256") != receipt["engine_receipt_sha256"]
                ):
                    raise ValueError("Router tape artifact ownership drift")
                base_digest = artifact.get("base_tape_projection_sha256")
                if (
                    not _hash(base_digest)
                    or base_digest not in committed_cost
                    or committed_kinds.get(base_digest) != "cost_base_tape_projection"
                ):
                    raise ValueError("Router base tape committed-kind mismatch")
                base = _canonical_artifact(router_raw[base_digest])
                base_fields = {
                    "schema",
                    "fold_id",
                    "variant_id",
                    "leaf_id",
                    "engine_receipt_sha256",
                    "tape_kind",
                    "projection",
                    "projection_sha256",
                }
                if (
                    set(base) != base_fields
                    or base.get("schema") != "router_cost_base_tape_projection_v1"
                    or base.get("fold_id") != receipt["fold_id"]
                    or base.get("variant_id") != receipt["variant_id"]
                    or base.get("leaf_id") != receipt["leaf_id"]
                    or base.get("engine_receipt_sha256") != receipt["engine_receipt_sha256"]
                    or base.get("tape_kind") != expected_kind
                    or not isinstance(base.get("projection"), list)
                    or not base["projection"]
                    or base["projection"] != artifact["sequence"]
                    or _canonical_sha256(base["projection"]) != base.get("projection_sha256")
                    or (
                        expected_kind in base_commitments
                        and base_commitments[expected_kind] != base_digest
                    )
                ):
                    raise ValueError("Router base tape ownership drift")
                base_commitments[expected_kind] = base_digest
                consumed_router.add(base_digest)
            key = (receipt["variant_id"], receipt["fold_id"], receipt["leaf_id"], cost_bps)
            if key in router_tapes:
                raise ValueError("duplicate Router cost tape commitment")
            router_tapes[key] = bundle
    if (
        not router_tapes
        or consumed_router != committed_cost
        or router_cost_receipts
        != {digest for digest, kind in committed_kinds.items() if kind == "cost_tape_receipt"}
    ):
        raise ValueError("missing or unattributed committed Router cost artifact")
    trial_raw = {
        digest: Path(path).read_bytes() for digest, path in trial_result_artifact_paths.items()
    }
    if not trial_raw or any(
        not _hash(digest) or hashlib.sha256(content).hexdigest() != digest
        for digest, content in trial_raw.items()
    ):
        raise ValueError("invalid trial result artifact path map")
    trial_results = {digest: _canonical_artifact(content) for digest, content in trial_raw.items()}
    search_fields = {
        "schema",
        "trial_ledger_sha256",
        "trial_result_artifacts",
        "candidate_ids",
        "candidate_ids_sha256",
        "profile_sha256",
        "source_manifest_sha256",
        "router_manifest_sha256",
        "lifecycle_sha256",
        "membership_sha256",
        "post_oos_research_variant",
        "post_oos_augment",
        "post_oos_augmentation_count",
        "current_fold_oos_input_count",
        "new_grid_search",
        "recompute_from_json",
        "frozen_at_utc",
        "trial_projection_sha256",
        "validation_period_ids_sha256",
        "locked_oos_period_ids_sha256",
    }
    if (
        set(search_receipt) != search_fields
        or search_receipt.get("schema") != "cost_proof_search_run_receipt_v2"
        or search_receipt.get("trial_ledger_sha256")
        != hashlib.sha256(raw["trial_ledger"]).hexdigest()
        or search_receipt.get("trial_result_artifacts")
        != [{"artifact_sha256": digest} for digest in trial_raw]
        or search_receipt.get("candidate_ids") != list(CANDIDATES)
        or search_receipt.get("candidate_ids_sha256") != candidate_ids_sha256()
        or search_receipt.get("profile_sha256") != hashlib.sha256(raw["profile"]).hexdigest()
        or search_receipt.get("source_manifest_sha256")
        != hashlib.sha256(raw["source_data_manifest"]).hexdigest()
        or search_receipt.get("router_manifest_sha256")
        != hashlib.sha256(raw["router_replay_manifest"]).hexdigest()
        or search_receipt.get("lifecycle_sha256") != hashlib.sha256(raw["lifecycle"]).hexdigest()
        or search_receipt.get("membership_sha256") != hashlib.sha256(raw["membership"]).hexdigest()
        or search_receipt.get("post_oos_research_variant") is not True
        or search_receipt.get("post_oos_augment") is not False
        or type(search_receipt.get("post_oos_augmentation_count")) is not int
        or search_receipt["post_oos_augmentation_count"] != 0
        or type(search_receipt.get("current_fold_oos_input_count")) is not int
        or search_receipt["current_fold_oos_input_count"] != 0
        or search_receipt.get("new_grid_search") is not False
        or search_receipt.get("recompute_from_json") is not False
        or _utc(search_receipt.get("frozen_at_utc")) is None
        or search_receipt.get("trial_projection_sha256")
        != _canonical_artifact(raw["trial_ledger"]).get("trial_projection_sha256")
        or search_receipt.get("validation_period_ids_sha256")
        != _canonical_artifact(raw["trial_ledger"]).get("validation_period_ids_sha256")
        or search_receipt.get("locked_oos_period_ids_sha256")
        != _canonical_artifact(raw["trial_ledger"]).get("locked_oos_period_ids_sha256")
    ):
        raise ValueError("search-run receipt binding drift")
    cost_commit = _canonical_artifact(raw["commit_receipt"])
    commit_fields = {
        "schema",
        "evidence_sha256",
        "profile_sha256",
        "source_manifest_sha256",
        "source_run_receipt_sha256",
        "search_run_receipt_sha256",
        "trial_ledger_sha256",
        "router_manifest_sha256",
        "lifecycle_sha256",
        "membership_sha256",
        "producer_source_sha256",
        "verifier_source_sha256",
        "candidate_ids",
        "candidate_ids_sha256",
        "source_artifacts",
        "trial_result_artifacts",
        "router_tapes",
        "trial_projection_sha256",
        "validation_period_ids_sha256",
        "locked_oos_period_ids_sha256",
        "committed_at_utc",
    }
    router_projection = [
        {
            "variant_id": key[0],
            "fold_id": key[1],
            "leaf_id": key[2],
            "cost_bps": key[3],
            "receipt_sha256": _canonical_sha256(bundle["receipt"]),
            "signal_position_sha256": _canonical_sha256(bundle["signal_position"]),
            "order_sha256": _canonical_sha256(bundle["order"]),
            "fill_sha256": _canonical_sha256(bundle["fill"]),
            "event_sha256": _canonical_sha256(bundle["event"]),
        }
        for key, bundle in sorted(
            router_tapes.items(),
            key=lambda item: (item[0][0], item[0][1], item[0][2], item[0][3]),
        )
    ]
    if (
        set(cost_commit) != commit_fields
        or cost_commit.get("schema") != "cost_proof_commit_v2"
        or cost_commit.get("evidence_sha256") != evidence_sha256
        or cost_commit.get("profile_sha256") != hashlib.sha256(raw["profile"]).hexdigest()
        or cost_commit.get("source_manifest_sha256")
        != hashlib.sha256(raw["source_data_manifest"]).hexdigest()
        or cost_commit.get("source_run_receipt_sha256")
        != hashlib.sha256(raw["source_run_receipt"]).hexdigest()
        or cost_commit.get("search_run_receipt_sha256")
        != hashlib.sha256(raw["search_run_receipt"]).hexdigest()
        or cost_commit.get("trial_ledger_sha256") != hashlib.sha256(raw["trial_ledger"]).hexdigest()
        or cost_commit.get("router_manifest_sha256")
        != hashlib.sha256(raw["router_replay_manifest"]).hexdigest()
        or cost_commit.get("lifecycle_sha256") != hashlib.sha256(raw["lifecycle"]).hexdigest()
        or cost_commit.get("membership_sha256") != hashlib.sha256(raw["membership"]).hexdigest()
        or cost_commit.get("producer_source_sha256")
        != hashlib.sha256(raw["producer_source"]).hexdigest()
        or cost_commit.get("verifier_source_sha256")
        != hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        or cost_commit.get("candidate_ids") != list(CANDIDATES)
        or cost_commit.get("candidate_ids_sha256") != candidate_ids_sha256()
        or cost_commit.get("source_artifacts") != source["artifacts"]
        or cost_commit.get("trial_result_artifacts")
        != [{"artifact_sha256": digest} for digest in trial_raw]
        or cost_commit.get("trial_projection_sha256") != search_receipt["trial_projection_sha256"]
        or cost_commit.get("validation_period_ids_sha256")
        != search_receipt["validation_period_ids_sha256"]
        or cost_commit.get("locked_oos_period_ids_sha256")
        != search_receipt["locked_oos_period_ids_sha256"]
        or cost_commit.get("router_tapes") != router_projection
        or _utc(cost_commit.get("committed_at_utc")) is None
    ):
        raise ValueError("cost-proof commit binding drift")
    hashes = {name: hashlib.sha256(content).hexdigest() for name, content in raw.items()}
    hashes["verifier_source"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return ExternalBindings(
        hashes,
        profile,
        source,
        source_receipt,
        search_receipt,
        _canonical_artifact(raw["commit_receipt"]),
        _canonical_artifact(raw["router_replay_manifest"]),
        membership,
        _canonical_artifact(raw["trial_ledger"]),
        trial_results,
        frozenset(market_raw),
        frozenset(funding_raw),
        market_rows,
        funding_rows,
        router_tapes,
        roots,
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
    if type(value) not in {int, float}:
        return None
    try:
        number = float(value)
    except OverflowError, ValueError:
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
    if (
        parsed.tzinfo is None
        or parsed.utcoffset() != UTC.utcoffset(parsed)
        or parsed.astimezone(UTC).isoformat().replace("+00:00", "Z") != value
    ):
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
    return (
        execution.get("slippage_impact_model") == "sqrt_impact"
        and execution.get("slippage_impact_coefficient") == 0.10
        and execution.get("slippage_adv_quote") == 0.0
        and execution.get("require_funding_coverage") is True
        and execution.get("funding_on_utc_boundary") is True
        and execution.get("funding_interval_hours") == 8
        and type(execution.get("funding_interval_hours")) is int
        and execution.get("maintenance_margin_rate") == 0.005
        and execution.get("liquidation_buffer_rate") == 0.0005
        and risk.get("default_stop_loss_pct") == 0.01
        and risk.get("attach_default_protective_stop") is True
        and risk.get("enforce_order_risk_gate_in_backtest") is True
        and _mapping(root.get("backtest")) is not None
        and root["backtest"].get("leverage") == 3
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


def _verify_tapes(
    scenario: Mapping[str, Any], errors: list[str]
) -> tuple[str, str, str, str] | None:
    tapes = (
        scenario.get("signal_position_tape"),
        scenario.get("orders"),
        scenario.get("fills"),
        scenario.get("events"),
    )
    if any(_records(rows) is None for rows in tapes):
        errors.append("missing recomputable tapes")
        return None
    claimed = (
        scenario.get("signal_tape_sha256"),
        scenario.get("order_tape_sha256"),
        scenario.get("execution_tape_sha256"),
        scenario.get("event_tape_sha256"),
    )
    computed = tuple(_canonical_sha256(rows) for rows in tapes)
    if not all(_hash(item) for item in claimed) or tuple(claimed) != computed:
        errors.append("claimed tape hash does not match records")
        return None
    return tuple(
        _canonical_sha256(
            [{key: value for key, value in row.items() if key != "cost_bps"} for row in rows]
        )
        for rows in tapes
    )


def _router_subset(
    rows: list[Mapping[str, Any]],
    artifact: Mapping[str, Any],
    *,
    fold_id: str,
    candidate_id: str,
) -> set[int] | None:
    """Authenticate one exact downstream Router subset, never a self-hash."""
    committed = _records(artifact.get("rows"))
    sequence = artifact.get("sequence")
    if (
        committed is None
        or not isinstance(sequence, list)
        or len(sequence) != len(committed)
        or len(set(sequence)) != len(sequence)
        or any(not isinstance(value, str) or not value for value in sequence)
    ):
        return None
    expected: dict[str, str] = {}
    for sequence_id, row in zip(sequence, committed, strict=True):
        if (
            set(row)
            != {
                "sequence_id",
                "fold_id",
                "variant_id",
                "leaf_id",
                "engine_receipt_sha256",
                "row_sha256",
            }
            or row.get("sequence_id") != sequence_id
            or row.get("fold_id") != fold_id
            or row.get("variant_id") != candidate_id
            or row.get("leaf_id") != artifact.get("leaf_id")
            or row.get("engine_receipt_sha256") != artifact.get("engine_receipt_sha256")
            or not _hash(row.get("engine_receipt_sha256"))
            or not _hash(row.get("row_sha256"))
        ):
            return None
        expected[sequence_id] = str(row["row_sha256"])
    matched: set[int] = set()
    observed: list[str] = []
    for index, row in enumerate(rows):
        if (
            row.get("fold_id") != fold_id
            or row.get("variant_id") != candidate_id
            or row.get("leaf_id") != artifact.get("leaf_id")
        ):
            continue
        sequence_id = row.get("sequence_id")
        if (
            not isinstance(sequence_id, str)
            or sequence_id not in expected
            or row.get("engine_receipt_sha256") != artifact.get("engine_receipt_sha256")
            or _canonical_sha256(dict(row)) != expected[sequence_id]
        ):
            return None
        observed.append(sequence_id)
        matched.add(index)
    if observed != sequence:
        return None
    return matched


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
        "realized_pnl",
        "unrealized_pnl",
        "inventory_cost_basis",
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
    events = _records(scenario.get("events"))
    if signals is None or orders is None or fills is None or events is None:
        return None
    signal_fields = {
        "cost_bps",
        "sequence_id",
        "fold_id",
        "variant_id",
        "leaf_id",
        "engine_receipt_sha256",
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
            or type(signal.get("cost_bps")) is not int
            or signal["cost_bps"] != scenario.get("cost_bps")
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
        "sequence_id",
        "fold_id",
        "variant_id",
        "leaf_id",
        "engine_receipt_sha256",
        "cost_bps",
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
            or type(order.get("cost_bps")) is not int
            or order["cost_bps"] != scenario.get("cost_bps")
            or order_id in order_map
            or (period_id, symbol) not in signal_map
            or signed_qty == 0
            or signed_quote == 0
            or not _close(abs(signed_qty), requested)
            or direction != ("BUY" if signed_qty > 0 else "SELL")
            or not isinstance(order.get("is_maker"), bool)
            or not isinstance(order.get("is_entry"), bool)
            or (order_type == "LMT") != order["is_maker"]
            or order_type not in {"MKT", "LMT"}
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
        "sequence_id",
        "fold_id",
        "variant_id",
        "leaf_id",
        "engine_receipt_sha256",
        "cost_bps",
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
            or type(fill.get("cost_bps")) is not int
            or fill["cost_bps"] != scenario.get("cost_bps")
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
        source_market = bindings.market_rows.get(
            (
                str(signal["market_data_artifact_sha256"]),
                str(signal["market_source_row_id"]),
                str(signal["symbol"]),
            )
        )
        if source_market is None:
            return None
        tick = float(source_market["price_tick_size"])
        step = float(source_market["quantity_step_size"])
        if (
            not _close(bar_volume, float(source_market["bar_volume_base"]))
            or not _close(price / tick, round(price / tick))
            or not _close(abs(quantity) / step, round(abs(quantity) / step))
        ):
            return None
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
    event_fields = {
        "sequence_id",
        "fold_id",
        "variant_id",
        "leaf_id",
        "engine_receipt_sha256",
        "cost_bps",
        "event_id",
        "event_index",
        "period_id",
        "timestamp",
        "symbol",
        "fill_id",
        "event_type",
    }
    allowed_event_types = {"entry", "reduce", "flatten", "protective_stop_trigger", "liquidation"}
    event_map: dict[str, Mapping[str, Any]] = {}
    fill_events: dict[str, Mapping[str, Any]] = {}
    events_by_period: dict[str, list[Mapping[str, Any]]] = {}
    for event in events:
        event_id, fill_id = _string(event, "event_id"), _string(event, "fill_id")
        stamp = _utc(event.get("timestamp"))
        fill = fill_map.get(fill_id or "")
        event_type = _string(event, "event_type")
        index = event.get("event_index")
        if (
            not _exact_fields(event, event_fields)
            or type(event.get("cost_bps")) is not int
            or event["cost_bps"] != scenario.get("cost_bps")
            or event_id is None
            or fill is None
            or stamp is None
            or event_id in event_map
            or fill_id in fill_events
            or type(index) is not int
            or index < 0
            or event_type not in allowed_event_types
            or event.get("period_id") != fill["period_id"]
            or event.get("timestamp") != fill["timestamp"]
            or event.get("symbol") != fill["symbol"]
            or (event_type == "entry") != bool(fill["is_entry"])
            or (event_type != "entry" and fill["is_entry"] is not False)
        ):
            return None
        event_map[event_id] = event
        fill_events[str(fill_id)] = event
        events_by_period.setdefault(str(event["period_id"]), []).append(event)
    if set(fill_events) != set(fill_map):
        return None
    for period_events in events_by_period.values():
        period_events.sort(key=lambda event: int(event["event_index"]))
        if [event["event_index"] for event in period_events] != list(range(len(period_events))):
            return None
    ordered_events = sorted(
        (event for period_events in events_by_period.values() for event in period_events),
        key=lambda event: (
            str(event["timestamp"]),
            str(event["period_id"]),
            int(event["event_index"]),
        ),
    )
    if events != ordered_events or (not ordered_events and fills):
        return None
    return {
        "signals": signal_map,
        "orders": order_map,
        "fills": fill_map,
        "events": event_map,
        "fill_events": fill_events,
        "events_by_period": events_by_period,
        "deltas": deltas,
        "notionals": notionals,
        "impact": impact,
    }


def _strict_stops(
    fold: Mapping[str, Any],
    times: Mapping[str, datetime],
    tape: Mapping[str, Any],
    positions: Mapping[str, tuple[dict[str, float], dict[str, float]]],
    bindings: ExternalBindings,
) -> bool:
    stop_rows = _records(fold.get("protective_stops"))
    if stop_rows is None:
        return False
    fields = {
        "stop_id",
        "symbol",
        "entry_fill_id",
        "side",
        "quantity",
        "stop_price",
        "source",
        "activated_period_id",
        "deactivated_period_id",
        "trigger_fill_id",
    }
    stops: dict[str, Mapping[str, Any]] = {}
    for stop in stop_rows:
        stop_id = _string(stop, "stop_id")
        entry_id = _string(stop, "entry_fill_id")
        entry = tape["fills"].get(entry_id or "")
        trigger_id = stop.get("trigger_fill_id")
        trigger = tape["fills"].get(trigger_id) if isinstance(trigger_id, str) else None
        deactivated = stop.get("deactivated_period_id")
        quantity, price = (
            _num(stop.get("quantity"), positive=True),
            _num(stop.get("stop_price"), positive=True),
        )
        if (
            not _exact_fields(stop, fields)
            or stop_id is None
            or stop_id in stops
            or entry is None
            or entry.get("is_entry") is not True
            or stop.get("activated_period_id") != entry.get("period_id")
            or _string(stop, "symbol") != entry.get("symbol")
            or _string(stop, "side") != ("SELL" if float(entry["signed_qty"]) > 0 else "BUY")
            or _string(stop, "source") not in {"engine_default", "strategy"}
            or quantity is None
            or price is None
            or quantity + EPS < abs(float(entry["signed_qty"]))
            or stop.get("activated_period_id") not in times
            or (trigger_id is not None and not isinstance(trigger_id, str))
            or (trigger_id is not None and trigger is None)
            or (
                deactivated is not None
                and (not isinstance(deactivated, str) or deactivated not in times)
            )
            or (
                deactivated is not None
                and times[str(deactivated)] < times[str(stop["activated_period_id"])]
            )
            or (
                trigger is not None
                and (
                    trigger.get("is_entry") is not False
                    or trigger.get("protective_stop_id") != stop_id
                    or trigger.get("symbol") != stop.get("symbol")
                    or deactivated != trigger["period_id"]
                    or tape["fill_events"].get(str(trigger_id), {}).get("event_type")
                    != "protective_stop_trigger"
                )
            )
        ):
            return False
        if stop["source"] == "engine_default":
            signal = tape["signals"][(str(entry["period_id"]), str(entry["symbol"]))]
            market = bindings.market_rows.get(
                (
                    str(signal["market_data_artifact_sha256"]),
                    str(signal["market_source_row_id"]),
                    str(entry["symbol"]),
                )
            )
            if market is None:
                return False
            tick, step = float(market["price_tick_size"]), float(market["quantity_step_size"])
            raw = float(entry["fill_price"]) * (0.99 if float(entry["signed_qty"]) > 0 else 1.01)
            expected = (
                math.floor(raw / tick) * tick
                if float(entry["signed_qty"]) > 0
                else math.ceil(raw / tick) * tick
            )
            if not _close(float(stop["stop_price"]), expected) or not _close(
                float(stop["quantity"]) / step, round(float(stop["quantity"]) / step)
            ):
                return False
        stops[stop_id] = stop

    entries = [
        fill
        for fill in tape["fills"].values()
        if fill["period_id"] in times and fill["is_entry"] is True
    ]
    entry_stop_ids = {_string(fill, "protective_stop_id") for fill in entries}
    if (
        None in entry_stop_ids
        or len(entry_stop_ids) != len(entries)
        or set(stops) != entry_stop_ids
        or any(
            stops[str(fill["protective_stop_id"])]["entry_fill_id"] != fill["fill_id"]
            for fill in entries
        )
    ):
        return False
    trigger_stops: dict[str, str] = {}
    ordinary_deactivations: dict[str, str] = {}
    for stop_id, stop in stops.items():
        trigger_id = stop["trigger_fill_id"]
        if trigger_id is not None:
            trigger_fill_id = str(trigger_id)
            trigger_event = tape["fill_events"][trigger_fill_id]
            entry_event = tape["fill_events"][str(stop["entry_fill_id"])]
            if (
                trigger_fill_id in trigger_stops
                or times[str(trigger_event["period_id"])] < times[str(entry_event["period_id"])]
                or (
                    trigger_event["period_id"] == entry_event["period_id"]
                    and int(trigger_event["event_index"]) <= int(entry_event["event_index"])
                )
            ):
                return False
            trigger_stops[trigger_fill_id] = stop_id
            continue
        deactivated = stop["deactivated_period_id"]
        if deactivated is None:
            continue
        matches = [
            event
            for event in tape["events_by_period"].get(str(deactivated), [])
            if event["event_type"] in {"reduce", "flatten", "liquidation"}
            and tape["fills"][str(event["fill_id"])]["protective_stop_id"] == stop_id
            and tape["fills"][str(event["fill_id"])]["symbol"] == stop["symbol"]
        ]
        if len(matches) != 1:
            return False
        deactivation_event = matches[0]
        entry_event = tape["fill_events"][str(stop["entry_fill_id"])]
        if times[str(deactivation_event["period_id"])] < times[str(entry_event["period_id"])] or (
            deactivation_event["period_id"] == entry_event["period_id"]
            and int(deactivation_event["event_index"]) <= int(entry_event["event_index"])
        ):
            return False
        ordinary_deactivations[str(deactivation_event["fill_id"])] = stop_id

    def protected_stop_ids(quantities: Mapping[str, float], active: set[str]) -> bool:
        for symbol, position in quantities.items():
            if abs(position) <= EPS:
                continue
            protected = [stop_id for stop_id in active if stops[stop_id]["symbol"] == symbol]
            if (
                len(protected) != 1
                or stops[protected[0]]["side"] != ("SELL" if position > 0 else "BUY")
                or float(stops[protected[0]]["quantity"]) + EPS < abs(position)
            ):
                return False
        return True

    active: set[str] = set()
    for period in fold["periods"]:
        period_id = str(period["period_id"])
        claimed = period.get("active_protective_stop_ids")
        if not isinstance(claimed, list) or len(claimed) != len(set(claimed)):
            return False
        quantities = dict(positions[period_id][0])
        active_this_period = set(active)
        if not protected_stop_ids(quantities, active):
            return False
        for event in tape["events_by_period"].get(period_id, []):
            fill = tape["fills"][str(event["fill_id"])]
            fill_id = str(fill["fill_id"])
            stop_id = (
                _string(fill, "protective_stop_id") if event["event_type"] == "entry" else None
            )
            if event["event_type"] == "entry":
                if stop_id is None or stop_id not in stops or stop_id in active:
                    return False
                if stops[stop_id]["entry_fill_id"] != fill_id:
                    return False
            elif event["event_type"] == "protective_stop_trigger":
                stop_id = trigger_stops.get(fill_id)
                if stop_id is None or stop_id not in active:
                    return False
            quantities[str(fill["symbol"])] = quantities.get(str(fill["symbol"]), 0.0) + float(
                fill["signed_qty"]
            )
            if event["event_type"] == "entry":
                active.add(str(stop_id))
                active_this_period.add(str(stop_id))
            else:
                deactivated_stop = (
                    trigger_stops.get(fill_id)
                    if event["event_type"] == "protective_stop_trigger"
                    else ordinary_deactivations.get(fill_id)
                )
                if deactivated_stop is not None:
                    if deactivated_stop not in active:
                        return False
                    if abs(quantities.get(str(stops[deactivated_stop]["symbol"]), 0.0)) > EPS:
                        return False
                    active.remove(deactivated_stop)
            if not protected_stop_ids(quantities, active):
                return False
        if set(claimed) != active_this_period:
            return False
        for stop_id in active_this_period:
            stop = stops[stop_id]
            signal = tape["signals"][(period_id, str(stop["symbol"]))]
            entry = tape["fills"][str(stop["entry_fill_id"])]
            crossing = (
                float(entry["signed_qty"]) > 0 and float(signal["low"]) <= float(stop["stop_price"])
            ) or (
                float(entry["signed_qty"]) < 0
                and float(signal["high"]) >= float(stop["stop_price"])
            )
            trigger_id = stop["trigger_fill_id"]
            trigger_event = (
                tape["fill_events"].get(str(trigger_id)) if trigger_id is not None else None
            )
            entry_event = tape["fill_events"][str(stop["entry_fill_id"])]
            if crossing and (
                trigger_event is None
                or trigger_event["period_id"] != period_id
                or (
                    trigger_event["period_id"] == entry_event["period_id"]
                    and int(trigger_event["event_index"]) <= int(entry_event["event_index"])
                )
            ):
                return False
            if (
                not crossing
                and trigger_event is not None
                and trigger_event["period_id"] == period_id
            ):
                return False
    return (
        not active
        and type(fold.get("protective_stop_count")) is int
        and type(fold.get("entry_count")) is int
        and fold["protective_stop_count"] == len(stops)
        and fold["entry_count"]
        == sum(fill["is_entry"] for fill in tape["fills"].values() if fill["period_id"] in times)
    )


def _strict_funding(
    fold: Mapping[str, Any],
    times: Mapping[str, datetime],
    tape: Mapping[str, Any],
    positions: Mapping[str, tuple[dict[str, float], dict[str, float]]],
    bindings: ExternalBindings,
) -> dict[str, float] | None:
    # A held end position must never bridge an unrepresented UTC funding
    # settlement.  The next represented period then supplies both the exact
    # source mark and the funding row through the checks below.
    represented = set(times.values())
    ordered_periods = list(fold["periods"])
    for index, period in enumerate(ordered_periods[:-1]):
        period_id = str(period["period_id"])
        _starts, ends = positions[period_id]
        if not any(abs(quantity) > EPS for quantity in ends.values()):
            continue
        stamp, next_stamp = times[period_id], times[str(ordered_periods[index + 1]["period_id"])]
        boundary = stamp.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(
            hours=(stamp.hour // 8 + 1) * 8
        )
        while boundary < next_stamp:
            if boundary not in represented:
                return None
            boundary += timedelta(hours=8)
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
        bool,
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
        or type(fold.get("liquidation_count")) is not int
        or fold["liquidation_count"] < 0
        or type(fold.get("ruin")) is not bool
    ):
        return None
    times, periods, initial_equity = grid
    expected_symbols = set(expected_router["symbols"])
    used_signals: set[tuple[str, str]] = set()
    used_orders: set[str] = set()
    used_fills: set[str] = set()
    positions: dict[str, tuple[dict[str, float], dict[str, float]]] = {}
    previous: dict[str, float] = dict.fromkeys(expected_symbols, 0.0)
    validation_range = _range(fold["validation_range"])
    locked_range = _range(fold["locked_oos_range"])
    validation_last: str | None = None
    locked_last: str | None = None
    if validation_range is None or locked_range is None:
        return None
    for period in periods:
        period_id = str(period["period_id"])
        stamp = times[period_id]
        segment = (
            "validation"
            if validation_range[0] <= stamp < validation_range[1]
            else "locked_oos"
            if locked_range[0] <= stamp < locked_range[1]
            else None
        )
        if segment is None or period.get("segment") != segment:
            return None
        if segment == "validation":
            if locked_last is not None:
                return None
            validation_last = period_id
        else:
            if validation_last is None:
                return None
            locked_last = period_id
        signal_keys = {
            symbol for candidate_period, symbol in tape["signals"] if candidate_period == period_id
        }
        if signal_keys != expected_symbols:
            return None
        starts: dict[str, float] = {}
        ends: dict[str, float] = {}
        for symbol in expected_symbols:
            signal = tape["signals"][(period_id, symbol)]
            if signal["timestamp"] != period["timestamp"]:
                return None
            start = float(signal["start_position"])
            end = float(signal["position"])
            if not _close(start, previous[symbol]) or not _close(
                end, start + tape["deltas"].get((period_id, symbol), 0.0)
            ):
                return None
            starts[symbol] = start
            ends[symbol] = end
            used_signals.add((period_id, symbol))
        positions[period_id] = (starts, ends)
        previous = ends
    cash_fold = not expected_symbols
    if validation_last is None or locked_last is None:
        return None
    if cash_fold and (
        any(order["period_id"] in times for order in tape["orders"].values())
        or any(fill["period_id"] in times for fill in tape["fills"].values())
        or any(tape["events_by_period"].get(period_id) for period_id in times)
        or fold["funding"]
        or fold["protective_stops"]
        or fold["entry_count"] != 0
        or fold["protective_stop_count"] != 0
        or fold["liquidation_count"] != 0
        or fold["ruin"] is not False
    ):
        return None
    for order_id, order in tape["orders"].items():
        if order["period_id"] in times:
            signal = tape["signals"].get((order["period_id"], order["symbol"]))
            if signal is None or order["timestamp"] != signal["timestamp"]:
                return None
            used_orders.add(order_id)
    for fill_id, fill in tape["fills"].items():
        if fill["period_id"] in times:
            used_fills.add(fill_id)
    liquidation_events = [
        event
        for period_id in times
        for event in tape["events_by_period"].get(period_id, [])
        if event["event_type"] == "liquidation"
    ]
    if len(liquidation_events) != fold["liquidation_count"]:
        return None
    funding_cash = _strict_funding(
        fold,
        times,
        tape,
        positions,
        bindings,
    )
    if funding_cash is None or not _strict_stops(fold, times, tape, positions, bindings):
        return None
    execution = _mapping(bindings.profile.get("execution"))
    if execution is None:
        return None
    maintenance = _num(execution.get("maintenance_margin_rate"), positive=True)
    buffer = _num(execution.get("liquidation_buffer_rate"), positive=True)
    backtest = _mapping(bindings.profile.get("backtest"))
    leverage_limit = _num(backtest.get("leverage")) if backtest is not None else None
    if (
        maintenance is None
        or buffer is None
        or leverage_limit is None
        or not _close(leverage_limit, 3.0)
    ):
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
        "prior_cash_balance",
        "cash_balance",
        "realized_pnl",
        "unrealized_pnl",
        "inventory_cost_basis",
        "gross_exposure_fraction",
        "raw_net_return",
        "exposure_normalized_net_return",
        "position_notional",
        "active_protective_stop_ids",
        "worst_intrabar_equity",
        "maintenance_margin_required",
    }
    cash = initial_equity
    inventory = dict.fromkeys(expected_symbols, (0.0, 0.0))
    locked_ids: list[str] = []
    ruin_seen = False
    returns: list[tuple[str, str, float, float]] = []
    pending_safety_breach = False
    safety_failure_seen = False
    endpoint_closures: dict[str, set[str]] = {}
    endpoint_live_symbols: dict[str, set[str]] = {}
    fills_by_period: dict[str, list[Mapping[str, Any]]] = {
        period_id: [tape["fills"][str(event["fill_id"])] for event in events]
        for period_id, events in tape["events_by_period"].items()
    }
    for period in periods:
        if pending_safety_breach:
            return None
        if not _exact_fields(period, period_fields):
            return None
        period_id = str(period["period_id"])
        starts, ends = positions[period_id]
        if period_id in {validation_last, locked_last}:
            endpoint_closures[period_id] = set()
            endpoint_live_symbols[period_id] = {
                symbol for symbol, quantity in starts.items() if abs(quantity) > EPS
            }
        prior_equity = cash + sum(
            quantity * (float(tape["signals"][(period_id, symbol)]["prior_mark_price"]) - basis)
            for symbol, (quantity, basis) in inventory.items()
        )
        if any(not _close(starts[symbol], inventory[symbol][0]) for symbol in expected_symbols):
            return None
        start_unrealized = prior_equity - cash
        realized = 0.0
        period_linear = 0.0
        period_impact = 0.0
        funding = funding_cash.get(period_id, 0.0)
        event_cash = cash + funding
        min_intrabar_equity = event_cash + sum(
            quantity
            * (
                (
                    float(tape["signals"][(period_id, symbol)]["low"])
                    if quantity > 0
                    else float(tape["signals"][(period_id, symbol)]["high"])
                )
                - basis
            )
            for symbol, (quantity, basis) in inventory.items()
        )
        maintenance_at_min_equity = sum(
            abs(
                quantity
                * (
                    float(tape["signals"][(period_id, symbol)]["low"])
                    if quantity > 0
                    else float(tape["signals"][(period_id, symbol)]["high"])
                )
            )
            * (maintenance + buffer)
            for symbol, (quantity, _) in inventory.items()
        )
        max_state_notional = sum(
            abs(quantity * float(tape["signals"][(period_id, symbol)]["prior_mark_price"]))
            for symbol, (quantity, _) in inventory.items()
        )
        initial_safety_breach = (
            min_intrabar_equity > EPS and min_intrabar_equity <= maintenance_at_min_equity
        ) or (
            min_intrabar_equity > EPS
            and sum(
                abs(
                    quantity
                    * (
                        float(tape["signals"][(period_id, symbol)]["low"])
                        if quantity > 0
                        else float(tape["signals"][(period_id, symbol)]["high"])
                    )
                )
                for symbol, (quantity, _) in inventory.items()
            )
            / min_intrabar_equity
            > leverage_limit + EPS
        )
        pending_safety_breach = pending_safety_breach or initial_safety_breach
        safety_failure_seen = safety_failure_seen or initial_safety_breach
        for fill in fills_by_period.get(period_id, []):
            symbol, delta, price = (
                str(fill["symbol"]),
                float(fill["signed_qty"]),
                float(fill["fill_price"]),
            )
            quantity, basis = inventory[symbol]
            event_realized = 0.0
            if abs(quantity) <= EPS or quantity * delta > 0:
                new_quantity = quantity + delta
                new_basis = (
                    price
                    if abs(quantity) <= EPS
                    else ((abs(quantity) * basis + abs(delta) * price) / abs(new_quantity))
                )
            else:
                closing = min(abs(quantity), abs(delta))
                event_realized = closing * (price - basis) * (1.0 if quantity > 0 else -1.0)
                realized += event_realized
                new_quantity = quantity + delta
                new_basis = (
                    price
                    if quantity * new_quantity < -EPS
                    else (basis if abs(new_quantity) > EPS else 0.0)
                )
            event = tape["fill_events"][str(fill["fill_id"])]
            if pending_safety_breach and event["event_type"] != "liquidation":
                return None
            old_quantity = quantity
            if (
                (
                    event["event_type"] == "entry"
                    and not (abs(old_quantity) <= EPS or old_quantity * delta > 0)
                )
                or (
                    event["event_type"] == "reduce"
                    and not (
                        abs(old_quantity) > EPS
                        and old_quantity * new_quantity > 0
                        and abs(new_quantity) < abs(old_quantity) - EPS
                    )
                )
                or (event["event_type"] == "flatten" and abs(new_quantity) > EPS)
                or (
                    event["event_type"] in {"protective_stop_trigger", "liquidation"}
                    and not (
                        abs(old_quantity) > EPS
                        and old_quantity * new_quantity >= -EPS
                        and abs(new_quantity) < abs(old_quantity) - EPS
                    )
                )
            ):
                return None
            if period_id in {validation_last, locked_last}:
                if abs(old_quantity) > EPS or abs(new_quantity) > EPS:
                    endpoint_live_symbols[period_id].add(symbol)
                if (
                    fill["is_entry"] is False
                    and event["event_type"] in {"flatten", "liquidation"}
                    and abs(old_quantity) > EPS
                    and abs(new_quantity) <= EPS
                ):
                    endpoint_closures[period_id].add(symbol)
            pre_fill_notional = sum(
                abs(
                    current_quantity
                    * (
                        price
                        if current_symbol == symbol
                        else float(tape["signals"][(period_id, current_symbol)]["mark_price"])
                    )
                )
                for current_symbol, (current_quantity, _) in inventory.items()
            )
            max_state_notional = max(max_state_notional, pre_fill_notional)
            immediate_marks = {
                current_symbol: (
                    price
                    if current_symbol == symbol
                    else float(tape["signals"][(period_id, current_symbol)]["mark_price"])
                )
                for current_symbol in expected_symbols
            }
            pre_event_equity = event_cash + sum(
                current_quantity * (immediate_marks[current_symbol] - current_basis)
                for current_symbol, (current_quantity, current_basis) in inventory.items()
            )
            pre_event_maintenance = sum(
                abs(current_quantity * immediate_marks[current_symbol]) * (maintenance + buffer)
                for current_symbol, (current_quantity, _) in inventory.items()
            )
            pre_event_breached = pre_event_equity <= pre_event_maintenance or (
                pre_event_equity > EPS
                and pre_fill_notional / pre_event_equity > leverage_limit + EPS
            )
            if (pre_event_breached and event["event_type"] != "liquidation") or (
                event["event_type"] == "liquidation"
                and not (pre_event_breached or pending_safety_breach)
            ):
                return None
            inventory[symbol] = (new_quantity, new_basis)
            fill_linear = abs(float(fill["signed_quote_notional"])) * bps / 10_000
            fill_impact = float(fill["sqrt_impact_cash_cost"])
            period_linear += fill_linear
            period_impact += fill_impact
            event_cash += event_realized - fill_linear - fill_impact
            immediate_equity = event_cash + sum(
                current_quantity * (immediate_marks[current_symbol] - current_basis)
                for current_symbol, (current_quantity, current_basis) in inventory.items()
            )
            immediate_maintenance = sum(
                abs(current_quantity * immediate_marks[current_symbol]) * (maintenance + buffer)
                for current_symbol, (current_quantity, _) in inventory.items()
            )
            adverse_equity = event_cash + sum(
                current_quantity
                * (
                    (
                        float(tape["signals"][(period_id, current_symbol)]["low"])
                        if current_quantity > 0
                        else float(tape["signals"][(period_id, current_symbol)]["high"])
                    )
                    - current_basis
                )
                for current_symbol, (current_quantity, current_basis) in inventory.items()
            )
            adverse_maintenance = sum(
                abs(
                    current_quantity
                    * (
                        float(tape["signals"][(period_id, current_symbol)]["low"])
                        if current_quantity > 0
                        else float(tape["signals"][(period_id, current_symbol)]["high"])
                    )
                )
                * (maintenance + buffer)
                for current_symbol, (current_quantity, _) in inventory.items()
            )
            adverse_notional = sum(
                abs(
                    current_quantity
                    * (
                        float(tape["signals"][(period_id, current_symbol)]["low"])
                        if current_quantity > 0
                        else float(tape["signals"][(period_id, current_symbol)]["high"])
                    )
                )
                for current_symbol, (current_quantity, _) in inventory.items()
            )
            post_fill_notional = sum(
                abs(
                    current_quantity
                    * (
                        price
                        if current_symbol == symbol
                        else float(tape["signals"][(period_id, current_symbol)]["mark_price"])
                    )
                )
                for current_symbol, (current_quantity, _) in inventory.items()
            )
            max_state_notional = max(max_state_notional, post_fill_notional)
            if adverse_equity <= min_intrabar_equity:
                min_intrabar_equity = adverse_equity
                maintenance_at_min_equity = adverse_maintenance
            immediate_safety_breach = immediate_equity <= immediate_maintenance or (
                immediate_equity > EPS
                and post_fill_notional / immediate_equity > leverage_limit + EPS
            )
            adverse_safety_breach = adverse_equity > EPS and (
                adverse_equity <= adverse_maintenance
                or adverse_notional / adverse_equity > leverage_limit + EPS
            )
            pending_safety_breach = (immediate_safety_breach or adverse_safety_breach) and any(
                abs(quantity) > EPS for quantity, _ in inventory.values()
            )
            safety_failure_seen = (
                safety_failure_seen or immediate_safety_breach or adverse_safety_breach
            )
        if any(not _close(ends[symbol], inventory[symbol][0]) for symbol in expected_symbols):
            return None
        if period_id in {validation_last, locked_last} and (
            any(abs(quantity) > EPS for quantity, _ in inventory.values())
            or endpoint_closures[period_id] != endpoint_live_symbols[period_id]
        ):
            return None
        linear = period_linear
        impact = period_impact
        prior_cash = cash
        cash = event_cash
        unrealized = sum(
            quantity * (float(tape["signals"][(period_id, symbol)]["mark_price"]) - basis)
            for symbol, (quantity, basis) in inventory.items()
        )
        gross = realized + unrealized - start_unrealized
        net = cash + unrealized - prior_equity
        equity = cash + unrealized
        end_notional = sum(
            abs(ends[symbol] * float(tape["signals"][(period_id, symbol)]["mark_price"]))
            for symbol in expected_symbols
        )
        exposure = max(max_state_notional, end_notional) / prior_equity
        position_notional = sum(
            ends[symbol] * float(tape["signals"][(period_id, symbol)]["mark_price"])
            for symbol in expected_symbols
        )
        raw = net / prior_equity
        normalized = raw / exposure if exposure > EPS else 0.0
        worst_equity = min_intrabar_equity
        maintenance_required = maintenance_at_min_equity
        basis_rows = [
            {"symbol": symbol, "quantity": quantity, "average_entry_price": basis}
            for symbol, (quantity, basis) in sorted(inventory.items())
            if abs(quantity) > EPS
        ]
        expected = {
            "gross_pnl": gross,
            "linear_cost": linear,
            "impact_cost": impact,
            "funding_cashflow": funding,
            "net_pnl": net,
            "prior_equity": prior_equity,
            "equity": equity,
            "prior_cash_balance": prior_cash,
            "cash_balance": cash,
            "realized_pnl": realized,
            "unrealized_pnl": unrealized,
            "gross_exposure_fraction": exposure,
            "raw_net_return": raw,
            "exposure_normalized_net_return": normalized,
            "position_notional": position_notional,
            "worst_intrabar_equity": worst_equity,
            "maintenance_margin_required": maintenance_required,
        }
        if (
            any(
                _num(period.get(name)) is None or not _close(float(period[name]), value)
                for name, value in expected.items()
            )
            or period.get("inventory_cost_basis") != basis_rows
            or prior_equity <= 0
            or (fold["ruin"] is False and min_intrabar_equity <= 0)
        ):
            return None
        if min_intrabar_equity <= 0 and period_id != str(periods[-1]["period_id"]):
            return None
        if exposure <= EPS and any(
            abs(value) > EPS for value in (gross, linear, impact, funding, net, raw)
        ):
            return None
        if period["segment"] == "locked_oos":
            locked_ids.append(period_id)
        returns.append((period_id, str(period["segment"]), raw, normalized))
        ruin_seen = ruin_seen or min_intrabar_equity <= 0
    if pending_safety_breach:
        return None
    final_equity = _num(fold.get("equity"))
    if final_equity is None or not _close(
        final_equity,
        cash
        + sum(
            quantity
            * (
                float(tape["signals"][(str(periods[-1]["period_id"]), symbol)]["mark_price"])
                - basis
            )
            for symbol, (quantity, basis) in inventory.items()
        ),
    ):
        return None
    if bool(fold["ruin"]) != ruin_seen:
        return None
    locked_gain = (
        math.prod(1 + raw for _, segment, raw, _ in returns if segment == "locked_oos") - 1
    )
    return (
        used_signals,
        used_orders,
        used_fills,
        locked_ids,
        returns,
        locked_gain,
        safety_failure_seen or bool(fold["liquidation_count"]) or ruin_seen,
    )


def _authenticated_router_tapes(
    scenario: Mapping[str, Any], candidate_id: str, bindings: ExternalBindings
) -> bool:
    names = {
        "signal_position_tape": "signal_position",
        "orders": "order",
        "fills": "fill",
        "events": "event",
    }
    consumed = {name: set() for name in names}
    for (variant, fold_id, _leaf, cost_bps), bundle in bindings.router_tapes.items():
        if variant != candidate_id or cost_bps != scenario.get("cost_bps"):
            continue
        for source_name, artifact_name in names.items():
            rows = _records(scenario.get(source_name))
            if rows is None:
                return False
            subset = _router_subset(
                rows,
                bundle[artifact_name],
                fold_id=fold_id,
                candidate_id=candidate_id,
            )
            if subset is None or consumed[source_name] & subset:
                return False
            consumed[source_name].update(subset)
    return all(
        indices == set(range(len(_records(scenario[name]) or [])))
        for name, indices in consumed.items()
    )


def _strict_engine_contract(
    scenario: Mapping[str, Any],
    candidate_id: str,
    bindings: ExternalBindings,
) -> tuple[list[str], list[tuple[list[tuple[str, str, float, float]], float, str, bool]]] | None:
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
        "events",
        "signal_tape_sha256",
        "order_tape_sha256",
        "execution_tape_sha256",
        "event_tape_sha256",
        "economic_tape_sha256",
        "folds",
    }
    if not _exact_fields(scenario, scenario_fields) or type(scenario.get("cost_bps")) is not int:
        return None
    tape = _strict_tapes(scenario, bindings)
    router = _router_contract(bindings, candidate_id)
    folds = _records(scenario.get("folds"))
    if (
        tape is None
        or router is None
        or not folds
        or not _authenticated_router_tapes(scenario, candidate_id, bindings)
    ):
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
    parsed_folds: list[tuple[list[tuple[str, str, float, float]], float, str, bool]] = []
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
            safety_failure,
        ) = result
        if used_signals & fold_signals or used_orders & fold_orders or used_fills & fold_fills:
            return None
        used_signals.update(fold_signals)
        used_orders.update(fold_orders)
        used_fills.update(fold_fills)
        locked_ids.extend(fold_locked)
        parsed_folds.append((fold_returns, locked_gain, fold_id, safety_failure))
    if (
        used_signals != set(tape["signals"])
        or used_orders != set(tape["orders"])
        or used_fills != set(tape["fills"])
    ):
        return None
    return locked_ids, parsed_folds


def _trial_ledger(
    ledger: Mapping[str, Any], parsed: list[dict[str, Any]], bindings: ExternalBindings
) -> tuple[np.ndarray, list[str], int, int] | None:
    """Validate the closed, authenticated attempted-trial family."""
    fields = {
        "schema",
        "cost_bps",
        "trials",
        "raw_trial_count",
        "effective_trial_count",
        "validation_period_ids",
        "locked_oos_period_ids",
        "validation_period_ids_sha256",
        "locked_oos_period_ids_sha256",
        "trial_projection_sha256",
        "current_fold_oos_input_count",
    }
    if (
        set(ledger) != fields
        or ledger.get("schema") != "cost_proof_trial_ledger_v2"
        or type(ledger.get("cost_bps")) is not int
        or ledger["cost_bps"] != 20
        or ledger.get("current_fold_oos_input_count") != 0
        or type(ledger.get("current_fold_oos_input_count")) is not int
    ):
        return None
    trials = _records(ledger.get("trials"))
    validation_ids, locked_ids = (
        ledger.get("validation_period_ids"),
        ledger.get("locked_oos_period_ids"),
    )
    raw_count, effective_count = ledger.get("raw_trial_count"), ledger.get("effective_trial_count")
    if (
        trials is None
        or not isinstance(validation_ids, list)
        or not isinstance(locked_ids, list)
        or not all(isinstance(value, str) and value for value in validation_ids + locked_ids)
        or len(set(validation_ids)) != len(validation_ids)
        or len(set(locked_ids)) != len(locked_ids)
        or len(locked_ids) < 16
        or len(locked_ids) % CSCV_SPLITS
        or type(raw_count) is not int
        or raw_count != len(trials)
        or type(effective_count) is not int
        or effective_count <= 0
        or effective_count > raw_count
        or ledger.get("validation_period_ids_sha256") != _canonical_sha256(validation_ids)
        or ledger.get("locked_oos_period_ids_sha256") != _canonical_sha256(locked_ids)
    ):
        return None
    trial_fields = {
        "trial_id",
        "ordinal",
        "registered_at_utc",
        "completed_at_utc",
        "status",
        "status_reason",
        "dedup_representative_trial_id",
        "result_artifact_sha256",
    }
    result_fields = {
        "schema",
        "trial_id",
        "ordinal",
        "registered_at_utc",
        "completed_at_utc",
        "status",
        "status_reason",
        "dedup_representative_trial_id",
        "validation_period_ids",
        "locked_oos_period_ids",
        "validation_normalized_returns",
        "locked_oos_normalized_returns",
    }
    frozen = _utc(bindings.search_run_receipt.get("frozen_at_utc"))
    if frozen is None:
        return None
    projection: list[dict[str, Any]] = []
    matrix_rows: list[list[float]] = []
    successful: dict[str, tuple[list[float], list[float]]] = {}
    successful_ids: list[str] = []
    seen_ids: set[str] = set()
    consumed: set[str] = set()
    prior_registered: datetime | None = None
    prior_completed: datetime | None = None
    for ordinal, trial in enumerate(trials):
        trial_id = _string(trial, "trial_id")
        digest = trial.get("result_artifact_sha256")
        status = _string(trial, "status")
        reason = trial.get("status_reason")
        representative = trial.get("dedup_representative_trial_id")
        registered = _utc(trial.get("registered_at_utc"))
        completed = _utc(trial.get("completed_at_utc"))
        artifact = bindings.trial_result_artifacts.get(str(digest))
        if (
            not _exact_fields(trial, trial_fields)
            or trial_id is None
            or trial_id in seen_ids
            or artifact is None
            or type(trial.get("ordinal")) is not int
            or trial["ordinal"] != ordinal
            or type(artifact.get("ordinal")) is not int
            or registered is None
            or completed is None
            or completed < registered
            or completed > frozen
            or (prior_registered is not None and registered < prior_registered)
            or (prior_completed is not None and completed < prior_completed)
            or status not in {"succeeded", "failed", "skipped"}
            or not _hash(digest)
            or digest in consumed
            or set(artifact) != result_fields
            or artifact.get("schema") != "cost_proof_trial_result_v2"
            or any(
                artifact.get(field) != trial.get(field)
                for field in (
                    "trial_id",
                    "ordinal",
                    "registered_at_utc",
                    "completed_at_utc",
                    "status",
                    "status_reason",
                    "dedup_representative_trial_id",
                )
            )
            or artifact.get("validation_period_ids") != validation_ids
            or artifact.get("locked_oos_period_ids") != locked_ids
        ):
            return None
        validation = artifact.get("validation_normalized_returns")
        locked = artifact.get("locked_oos_normalized_returns")
        if not isinstance(validation, list) or not isinstance(locked, list):
            return None
        if status == "succeeded":
            numeric_validation = [_num(value) for value in validation]
            numeric_locked = [_num(value) for value in locked]
            if (
                reason is not None
                or representative != trial_id
                or len(validation) != len(validation_ids)
                or len(locked) != len(locked_ids)
                or any(value is None for value in numeric_validation + numeric_locked)
            ):
                return None
            successful[trial_id] = (
                [float(value) for value in numeric_validation if value is not None],
                [float(value) for value in numeric_locked if value is not None],
            )
            successful_ids.append(trial_id)
            matrix_rows.append(successful[trial_id][1])
        elif status == "failed":
            if (
                reason is None
                or not isinstance(reason, str)
                or not reason
                or validation
                or locked
                or representative is not None
            ):
                return None
        elif (
            not isinstance(reason, str)
            or not reason
            or validation
            or locked
            or not isinstance(representative, str)
            or representative not in successful
        ):
            return None
        seen_ids.add(trial_id)
        consumed.add(str(digest))
        prior_registered, prior_completed = registered, completed
        projection.append(dict(trial))
    if (
        consumed != set(bindings.trial_result_artifacts)
        or ledger.get("trial_projection_sha256") != _canonical_sha256(projection)
        or effective_count != len(successful_ids)
        or not matrix_rows
    ):
        return None
    for item in parsed:
        matching = successful.get(item["candidate_id"])
        if (
            matching is None
            or item["locked_ids"] != locked_ids
            or item["validation_ids"] != validation_ids
            or matching[1] != item["normalized"]
            or matching[0] != item["validation"]
        ):
            return None
    return np.asarray(matrix_rows, dtype=float), successful_ids, raw_count, effective_count


def _whole_family_spa_pvalues(matrix: np.ndarray) -> np.ndarray:
    """Deterministic Hansen-style SPA using shared circular-block bootstrap draws.

    Each successful trial is a member of the null family.  The Hansen
    consistent recentering retains only models not demonstrably poor, while
    every draw re-studentizes its recentered return series.  The common draw
    and max statistic control family-wise promotion; the add-one correction
    prevents a zero finite-bootstrap p-value.
    """
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] < 16:
        raise ValueError("invalid SPA family matrix")
    n = matrix.shape[1]
    means = np.mean(matrix, axis=1)
    sigmas = np.std(matrix, axis=1, ddof=1)
    scale = sigmas / math.sqrt(float(n))
    original_degenerate = scale <= 1e-12
    observed = np.full(matrix.shape[0], 0.0)
    np.divide(means, scale, out=observed, where=~original_degenerate)
    observed[original_degenerate] = np.where(
        means[original_degenerate] > 0,
        math.inf,
        np.where(means[original_degenerate] < 0, -math.inf, 0.0),
    )
    threshold = -math.sqrt(2.0 * math.log(math.log(float(n))))
    null_means = np.where(observed >= threshold, means, 0.0)
    recentered = matrix - null_means[:, None]
    rng = np.random.default_rng(12345)
    block_len = max(1, round(n ** (1.0 / 3.0)))
    block_count = math.ceil(n / block_len)
    bootstrap_draws = 2_000
    exceed = np.zeros(matrix.shape[0], dtype=int)
    for _ in range(bootstrap_draws):
        starts = rng.integers(0, n, size=block_count)
        offsets = (starts[:, None] + np.arange(block_len)[None, :]) % n
        sample = recentered[:, offsets.reshape(-1)[:n]]
        sample_sigmas = np.std(sample, axis=1, ddof=1)
        sample_means = np.mean(sample, axis=1)
        sample_scale = sample_sigmas / math.sqrt(float(n))
        statistics = np.full(matrix.shape[0], 0.0)
        np.divide(
            sample_means,
            sample_scale,
            out=statistics,
            where=sample_scale > 1e-12,
        )
        degenerate_sample = sample_scale <= 1e-12
        statistics[degenerate_sample] = np.where(
            sample_means[degenerate_sample] > 0,
            math.inf,
            np.where(sample_means[degenerate_sample] < 0, -math.inf, 0.0),
        )
        statistics[original_degenerate] = -math.inf
        exceed += float(np.max(statistics)) >= observed
    pvalues = (exceed + 1.0) / (bootstrap_draws + 1.0)
    pvalues[means <= 0] = 1.0
    pvalues[original_degenerate] = 1.0
    return pvalues


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
        or any(type(item.get("cost_bps")) is not int for item in scenarios)
        or tuple(item["cost_bps"] for item in scenarios) != COST_LADDER
    ):
        return None, "scenario order/count mismatch"
    if (
        candidate.get("router_replay_manifest_sha256")
        != provenance["router_replay_manifest_sha256"]
        or candidate.get("membership_sha256") != provenance["membership_sha256"]
    ):
        return None, "candidate artifact binding mismatch"
    tapes: tuple[str, str, str, str] | None = None
    economic_tape_sha256: str | None = None
    layout: tuple[Any, ...] | None = None
    twenty: (
        tuple[list[float], list[float], list[tuple[str, float]], list[float], list[str]] | None
    ) = None
    safety_failure = False
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
        safety_failure = safety_failure or any(fold[3] for fold in parsed_folds)
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
            validation_ids = [
                row[0] for fold in parsed_folds for row in fold[0] if row[1] == "validation"
            ]
            twenty = (
                raw,
                normalized,
                [(fold[2], fold[1]) for fold in parsed_folds],
                validation,
                validation_ids,
            )
    if twenty is None:
        return None, "missing 20bp scenario"
    if locked_ids_20bp is None or initial_equities_20bp is None:
        return None, "missing locked-OOS identity"
    raw, normalized, folds, validation, validation_ids_20bp = twenty
    if len(raw) < 16 or len(raw) % CSCV_SPLITS or len(raw) != len(normalized):
        return None, "insufficient locked-OOS data"
    values = np.asarray(raw + normalized, dtype=float)
    if not np.isfinite(values).all() or np.std(normalized, ddof=1) <= 0:
        return None, "nonfinite, constant, or invalid return"
    return {
        "candidate_id": candidate_id,
        "raw": raw,
        "normalized": normalized,
        "folds": folds,
        "safety_failure": safety_failure,
        "validation_ids": validation_ids_20bp,
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
            or not isinstance(evidence.get("cost_ladder_bps"), list)
            or any(type(value) is not int for value in evidence["cost_ladder_bps"])
            or evidence["cost_ladder_bps"] != list(COST_LADDER)
            or type(evidence.get("cscv_splits")) is not int
            or evidence["cscv_splits"] != CSCV_SPLITS
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
        checked_ledger = _trial_ledger(bindings.trial_ledger, parsed, bindings)
        if checked_ledger is None:
            return _report("STOP", ["invalid whole-search trial ledger"])
        matrix, successful_trial_ids, raw_trial_count, effective_trial_count = checked_ledger
        if (
            matrix.shape[0] != effective_trial_count
            or len(successful_trial_ids) != effective_trial_count
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
        family_spa = _whole_family_spa_pvalues(matrix)
        reports: list[dict[str, Any]] = []
        for item in parsed:
            target = np.asarray(item["normalized"], dtype=float)
            try:
                trial_index = successful_trial_ids.index(item["candidate_id"])
            except ValueError:
                return _report("STOP", ["candidate trial is absent from authenticated family"])
            dsr = float(
                deflated_sharpe_ratio(
                    target,
                    num_trials=float(raw_trial_count),
                    variance_across_trials=variance,
                    hac_inference=True,
                )
            )
            spa_pvalue = float(family_spa[trial_index])
            gate_passed = dsr >= 0.90 and spa_pvalue <= 0.05 and pbo <= 0.50
            raw = item["raw"]
            reasons: list[str] = []
            net = math.prod(1 + value for value in raw) - 1
            mdd = max_drawdown(np.asarray(raw, dtype=float))
            ordered = sorted(item["folds"], key=lambda value: (-value[1], value[0]))
            leave_best = math.prod(1 + value[1] for value in ordered[1:]) - 1
            positive = [gain for _, gain in item["folds"] if gain > 0]
            if item["safety_failure"]:
                reasons.append("liquidation or ruin")
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
                "effective_trial_count": float(effective_trial_count),
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
                report["metrics"]["validation_mdd"],
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
        UnicodeEncodeError,
        ValueError,
    ):
        return _report("STOP", ["malformed evidence"])


def evaluate_cost_proof_file(
    input_path: str | Path,
    profile_path: str | Path,
    *,
    source_data_manifest_path: str | Path | None = None,
    source_run_receipt_path: str | Path | None = None,
    search_run_receipt_path: str | Path | None = None,
    router_replay_manifest_path: str | Path | None = None,
    router_source_artifact_path: str | Path | None = None,
    lifecycle_path: str | Path | None = None,
    membership_path: str | Path | None = None,
    trial_ledger_path: str | Path | None = None,
    producer_source_path: str | Path | None = None,
    commit_receipt_path: str | Path | None = None,
    router_producer_source_path: str | Path | None = None,
    router_commit_receipt_path: str | Path | None = None,
    market_artifact_paths: Mapping[str, str | Path] | None = None,
    funding_artifact_paths: Mapping[str, str | Path] | None = None,
    router_artifact_paths: Mapping[str, str | Path] | None = None,
    trial_result_artifact_paths: Mapping[str, str | Path] | None = None,
    trusted_roots: Mapping[str, str] | None = None,
) -> CostProofReport:
    """Validate v2 proof bytes against explicit out-of-band SHA-256 roots."""
    paths = {
        "profile": profile_path,
        "source_data_manifest": source_data_manifest_path,
        "source_run_receipt": source_run_receipt_path,
        "search_run_receipt": search_run_receipt_path,
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
    if (
        any(path is None for path in paths.values())
        or market_artifact_paths is None
        or funding_artifact_paths is None
        or router_artifact_paths is None
        or trial_result_artifact_paths is None
        or trusted_roots is None
    ):
        return _report("STOP", ["missing trusted external bindings"])
    try:
        evidence_raw = Path(input_path).read_bytes()
        evidence = _canonical_artifact(evidence_raw)
        bindings = _artifact_bindings(
            paths,  # type: ignore[arg-type]
            market_artifact_paths=market_artifact_paths,
            funding_artifact_paths=funding_artifact_paths,
            router_artifact_paths=router_artifact_paths,
            trial_result_artifact_paths=trial_result_artifact_paths,
            evidence_sha256=hashlib.sha256(evidence_raw).hexdigest(),
            trusted_roots=trusted_roots,
        )
    except (
        ArithmeticError,
        OSError,
        RecursionError,
        TypeError,
        UnicodeDecodeError,
        UnicodeEncodeError,
        json.JSONDecodeError,
        yaml.YAMLError,
        ValueError,
    ):
        return _report("STOP", ["unreadable evidence or external binding"])
    return evaluate_cost_proof(evidence, bindings=bindings)
