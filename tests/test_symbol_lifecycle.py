from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from lumina_quant.data import validate_fold_membership_manifest
from lumina_quant.data.symbol_lifecycle import (
    build_fold_membership_manifest,
    build_symbol_lifecycle_registry,
    is_symbol_active,
    validate_symbol_lifecycle_registry,
)


SOURCE = {
    "uri": "file:///fixtures/exchange_info.json",
    "retrieved_at_ms": 1_700_000_000_000,
    "payload_sha256": "a" * 64,
}


def _exchange_info() -> dict[str, object]:
    return {
        "symbols": [
            {"symbol": "LATEUSDT", "onboardDate": 150, "deliveryDate": None},
            {"symbol": "BTCUSDT", "onboardDate": 100, "deliveryDate": 200},
        ]
    }


def test_half_open_onboard_and_delivery_boundaries() -> None:
    registry = build_symbol_lifecycle_registry(_exchange_info(), ["BTCUSDT"], SOURCE)
    symbol = registry["symbols"][0]
    assert not is_symbol_active(symbol, 99)
    assert is_symbol_active(symbol, 100)
    assert is_symbol_active(symbol, 199)
    assert not is_symbol_active(symbol, 200)


def test_partial_fold_membership_is_explicit_and_ineligible() -> None:
    registry = build_symbol_lifecycle_registry(_exchange_info(), ["LATEUSDT", "BTCUSDT"], SOURCE)
    manifest = build_fold_membership_manifest(
        registry,
        [{"fold_id": "delivery", "start_ms": 125, "end_ms": 225}],
    )
    fold = manifest["folds"][0]
    assert fold["eligible_symbols"] == []
    assert fold["partial_symbols"] == ["BTCUSDT", "LATEUSDT"]
    assert fold["inactive_symbols"] == []


@pytest.mark.parametrize(
    ("symbols", "message"),
    [(["ABSENT"], "absent"), (["BTCUSDT", "BTCUSDT"], "duplicates")],
)
def test_missing_and_duplicate_requested_symbols_fail_closed(
    symbols: list[str], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        build_symbol_lifecycle_registry(_exchange_info(), symbols, SOURCE)


def test_invalid_provenance_hash_fails_closed() -> None:
    bad_source = {**SOURCE, "payload_sha256": "not-a-hash"}
    with pytest.raises(ValueError, match="SHA-256"):
        build_symbol_lifecycle_registry(_exchange_info(), ["BTCUSDT"], bad_source)


def test_registry_and_fold_ordering_are_deterministic() -> None:
    registry = build_symbol_lifecycle_registry(_exchange_info(), ["LATEUSDT", "BTCUSDT"], SOURCE)
    assert [row["symbol"] for row in registry["symbols"]] == ["BTCUSDT", "LATEUSDT"]
    manifest = build_fold_membership_manifest(
        registry,
        [
            {"fold_id": "later", "start_ms": 300, "end_ms": 400},
            {"fold_id": "earlier", "start_ms": 100, "end_ms": 125},
        ],
    )
    assert [fold["fold_id"] for fold in manifest["folds"]] == ["earlier", "later"]


def test_delivery_date_and_delivery_ms_must_be_explicit() -> None:
    exchange_info = _exchange_info()
    del exchange_info["symbols"][0]["deliveryDate"]  # type: ignore[index]
    with pytest.raises(ValueError, match="deliveryDate is required"):
        build_symbol_lifecycle_registry(exchange_info, ["LATEUSDT"], SOURCE)

    registry = build_symbol_lifecycle_registry(_exchange_info(), ["LATEUSDT"], SOURCE)
    assert is_symbol_active(registry["symbols"][0], 1_000)
    with pytest.raises(ValueError, match="delivery_ms is required"):
        is_symbol_active({"symbol": "LATEUSDT", "onboard_ms": 150}, 1_000)


def test_fold_boundaries_classify_full_partial_and_inactive_symbols() -> None:
    exchange_info = {
        "symbols": [
            {"symbol": "FULL", "onboardDate": 100, "deliveryDate": 200},
            {"symbol": "ONBOARD_START", "onboardDate": 100, "deliveryDate": 150},
            {"symbol": "DELIVERY_START", "onboardDate": 50, "deliveryDate": 100},
            {"symbol": "ONBOARD_END", "onboardDate": 200, "deliveryDate": None},
        ]
    }
    registry = build_symbol_lifecycle_registry(
        exchange_info,
        ["FULL", "ONBOARD_START", "DELIVERY_START", "ONBOARD_END"],
        SOURCE,
    )
    fold = build_fold_membership_manifest(
        registry, [{"fold_id": "boundary", "start_ms": 100, "end_ms": 200}]
    )["folds"][0]
    assert fold["eligible_symbols"] == ["FULL"]
    assert fold["partial_symbols"] == ["ONBOARD_START"]
    assert fold["inactive_symbols"] == ["DELIVERY_START", "ONBOARD_END"]


def test_empty_or_missing_folds_fail_closed() -> None:
    registry = build_symbol_lifecycle_registry(_exchange_info(), ["BTCUSDT"], SOURCE)
    with pytest.raises(ValueError, match="non-empty"):
        build_fold_membership_manifest(registry, [])

    with pytest.raises(ValueError, match="manifest fields"):
        validate_fold_membership_manifest(registry, {"schema_version": 1})


@pytest.mark.parametrize("forgery", ["registry_sha256", "source", "categories", "order"])
def test_fold_manifest_validation_rejects_forged_content(forgery: str) -> None:
    registry = build_symbol_lifecycle_registry(_exchange_info(), ["LATEUSDT", "BTCUSDT"], SOURCE)
    manifest = build_fold_membership_manifest(
        registry,
        [
            {"fold_id": "later", "start_ms": 300, "end_ms": 400},
            {"fold_id": "earlier", "start_ms": 100, "end_ms": 125},
        ],
    )
    forged = json.loads(json.dumps(manifest))
    if forgery == "registry_sha256":
        forged["registry_sha256"] = "0" * 64
    elif forgery == "source":
        forged["source"]["uri"] = "file:///forged.json"
    elif forgery == "categories":
        forged["folds"][0]["eligible_symbols"] = ["LATEUSDT"]
    else:
        forged["folds"].reverse()

    with pytest.raises(ValueError, match="does not match"):
        validate_fold_membership_manifest(registry, forged)


def test_public_fold_manifest_validator_returns_canonical_manifest() -> None:
    registry = build_symbol_lifecycle_registry(_exchange_info(), ["BTCUSDT"], SOURCE)
    manifest = build_fold_membership_manifest(
        registry, [{"fold_id": "fold", "start_ms": 100, "end_ms": 200}]
    )
    assert validate_fold_membership_manifest(registry, manifest) == manifest


def test_cli_writes_source_provenanced_registry_and_fold_manifest(tmp_path: Path) -> None:
    exchange_info_path = tmp_path / "exchange-info.json"
    exchange_info_path.write_text(json.dumps(_exchange_info()), encoding="utf-8")
    folds_path = tmp_path / "folds.json"
    folds_path.write_text(
        json.dumps([{"fold_id": "fold", "start_ms": 100, "end_ms": 200}]), encoding="utf-8"
    )
    output_path = tmp_path / "manifest.json"
    script = Path(__file__).parents[1] / "scripts/research/build_symbol_lifecycle_manifest.py"

    subprocess.run(
        [
            sys.executable,
            str(script),
            "--exchange-info-json",
            str(exchange_info_path),
            "--symbols",
            "BTCUSDT",
            "LATEUSDT",
            "--source-uri",
            "file:///fixtures/exchange-info.json",
            "--retrieved-at-ms",
            "1700000000000",
            "--folds-json",
            str(folds_path),
            "--output-json",
            str(output_path),
        ],
        check=True,
    )

    artifact = json.loads(output_path.read_text(encoding="utf-8"))
    registry = validate_symbol_lifecycle_registry(artifact["registry"])
    assert (
        registry["source"]["payload_sha256"]
        == hashlib.sha256(exchange_info_path.read_bytes()).hexdigest()
    )
    assert artifact["fold_membership"]["folds"][0]["partial_symbols"] == ["LATEUSDT"]


def test_cli_failure_does_not_replace_existing_output(tmp_path: Path) -> None:
    exchange_info_path = tmp_path / "exchange-info.json"
    exchange_info_path.write_text(json.dumps(_exchange_info()), encoding="utf-8")
    folds_path = tmp_path / "folds.json"
    folds_path.write_text("[]", encoding="utf-8")
    output_path = tmp_path / "manifest.json"
    original = b'{"preserve":"this"}\n'
    output_path.write_bytes(original)
    script = Path(__file__).parents[1] / "scripts/research/build_symbol_lifecycle_manifest.py"

    failed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--exchange-info-json",
            str(exchange_info_path),
            "--symbols",
            "BTCUSDT",
            "--source-uri",
            "file:///fixtures/exchange-info.json",
            "--retrieved-at-ms",
            "1700000000000",
            "--folds-json",
            str(folds_path),
            "--output-json",
            str(output_path),
        ],
        check=False,
        capture_output=True,
    )
    assert failed.returncode != 0
    assert output_path.read_bytes() == original
