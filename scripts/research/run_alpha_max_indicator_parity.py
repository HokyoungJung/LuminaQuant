#!/usr/bin/env python3
"""Run the official checkpointed Alpha-Max warmup parity boundary."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType

from lumina_quant.alpha_max_process_boundary import reject_ambient_lq_environment


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
        + b"\n"
    )


def _parse_utc(value: object) -> datetime:
    if type(value) is not str:
        raise ValueError("alpha_max_parity_availability_timestamp_invalid")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if (
        parsed.tzinfo is None
        or parsed.utcoffset() != UTC.utcoffset(parsed)
        or parsed.isoformat().replace("+00:00", "Z") != value
    ):
        raise ValueError("alpha_max_parity_availability_timestamp_invalid")
    return parsed


def _availability(
    preparation: dict[str, object], kind: str
) -> tuple[MappingProxyType, MappingProxyType]:
    availability = preparation.get("availability")
    if type(availability) is not dict or type(availability.get(kind)) is not dict:
        raise ValueError("alpha_max_parity_availability_invalid")
    value = availability[kind]
    starts = value.get("availability_start_by_symbol")
    ends = value.get("availability_end_by_symbol")
    if type(starts) is not dict or type(ends) is not dict or set(starts) != set(ends) or not starts:
        raise ValueError("alpha_max_parity_availability_invalid")
    return (
        MappingProxyType({key: _parse_utc(starts[key]) for key in sorted(starts)}),
        MappingProxyType({key: _parse_utc(ends[key]) for key in sorted(ends)}),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run exact-native Alpha-Max parity with whole-day checkpoints.",
        allow_abbrev=False,
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--feature-root", required=True)
    parser.add_argument("--preparation-manifest", required=True)
    parser.add_argument("--candidate-seal", required=True)
    parser.add_argument("--candidate-seal-sha256", required=True)
    parser.add_argument("--candidate-capsule-sha256", required=True)
    parser.add_argument("--candidate-finalization-sha256", required=True)
    parser.add_argument("--admitted-symbol", action="append", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    reject_ambient_lq_environment()
    args = build_parser().parse_args(argv)

    from lumina_quant.research.alpha_max_evidence import seal_alpha_max_root_tree
    from lumina_quant.research.alpha_max_engine_runner import (
        _AlphaMaxBoundedRawLoader,
        _alpha_max_phase_lookup,
        build_alpha_max_indicator_capsule,
        create_alpha_max_indicator_day_checkpoint_store,
        preflight_alpha_max_runtime_contract,
    )

    preparation_path = Path(args.preparation_manifest).resolve(strict=True)
    preparation_bytes = preparation_path.read_bytes()
    preparation = json.loads(preparation_bytes)
    if type(preparation) is not dict:
        raise ValueError("alpha_max_parity_preparation_manifest_invalid")
    raw_start, raw_end = _availability(preparation, "raw")
    feature_start, feature_end = _availability(preparation, "feature")
    raw = seal_alpha_max_root_tree(
        "warmup",
        "raw",
        args.raw_root,
        availability_start_by_symbol=raw_start,
        availability_end_by_symbol=raw_end,
    )
    feature = seal_alpha_max_root_tree(
        "warmup",
        "feature",
        args.feature_root,
        availability_start_by_symbol=feature_start,
        availability_end_by_symbol=feature_end,
    )
    admitted = tuple(args.admitted_symbol)
    preflight = preflight_alpha_max_runtime_contract(args.config)
    loader = _AlphaMaxBoundedRawLoader(raw, admitted)
    lookup = _alpha_max_phase_lookup({("warmup", "feature"): feature}, "warmup")
    candidate_identity = {
        "path": str(Path(args.candidate_seal).resolve(strict=True)),
        "candidate_seal_sha256": args.candidate_seal_sha256,
        "capsule_sha256": args.candidate_capsule_sha256,
        "finalization_sha256": args.candidate_finalization_sha256,
    }
    store = create_alpha_max_indicator_day_checkpoint_store(
        preflight,
        checkpoint_root=args.checkpoint_root,
        output_root=args.output_root,
        phase="validation_train_fit",
        manifest_path=args.manifest,
        admitted_symbols=admitted,
        phase_id="warmup",
        raw_root=args.raw_root,
        ordered_lookup=lookup,
        watermark=preflight.phase_windows["warmup"].end_utc,
        bounded_raw_loader=loader,
        checkpoint_candidate_identity=candidate_identity,
    )
    capsule = build_alpha_max_indicator_capsule(
        preflight,
        output_root=args.output_root,
        phase="validation_train_fit",
        manifest_path=args.manifest,
        admitted_symbols=admitted,
        phase_id="warmup",
        raw_root=args.raw_root,
        ordered_lookup=lookup,
        watermark=preflight.phase_windows["warmup"].end_utc,
        prior_indicator_capsule=None,
        bounded_raw_loader=loader,
        checkpoint_store=store,
        checkpoint_candidate_identity=candidate_identity,
    )
    payload = {
        "artifact_kind": "alpha_max_checkpointed_indicator_parity_result.v1",
        "capsule_sha256": capsule.capsule_sha256,
        "checkpoint_descriptor_sha256": store.descriptor_sha256,
        "discarded_signal_count": capsule.discarded_signal_count,
        "fill_event_count": capsule.fill_event_count,
        "funding_event_count": capsule.funding_event_count,
        "market_event_count": capsule.market_event_count,
        "native_finalization_sha256": capsule.native_finalization_sha256,
        "order_event_count": capsule.order_event_count,
        "order_routing_enabled": False,
        "partial_output_reusable": False,
        "phase_id": capsule.phase_id,
        "trade_count": capsule.trade_count,
        "windows_processed": capsule.windows_processed,
    }
    sys.stdout.buffer.write(_canonical_bytes(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
