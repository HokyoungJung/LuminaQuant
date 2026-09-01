from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl
import pytest
from lumina_quant.data import build_fold_membership_manifest
from lumina_quant.market_data import load_strict_ohlcv_route

_SCRIPT = Path(__file__).parents[1] / "scripts/research/validate_research_data_contract.py"
_SPEC = importlib.util.spec_from_file_location(
    "validate_research_data_contract",
    _SCRIPT,
)
assert _SPEC and _SPEC.loader
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

_POINTS = [0, 60_000, 120_000]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _bars(
    points: list[int],
    *,
    close: float = 1.0,
    highs: list[object] | None = None,
) -> pl.DataFrame:
    count = len(points)
    return pl.DataFrame(
        {
            "datetime": [datetime.fromtimestamp(point / 1000, UTC) for point in points],
            "open": [1.0] * count,
            "high": highs if highs is not None else [1.0] * count,
            "low": [1.0] * count,
            "close": [close] * count,
            "volume": [1.0] * count,
        }
    )


def _funding(
    points: list[object],
    *,
    rates: list[object] | None = None,
    sources: list[str] | None = None,
) -> pl.DataFrame:
    count = len(points)
    return pl.DataFrame(
        {
            "timestamp_ms": points,
            "funding_rate": rates if rates is not None else [0.0] * count,
            "source": sources if sources is not None else ["official"] * count,
        },
        strict=False,
    )


def _receipt_path(base: Path, identifier: str) -> Path:
    return base / f"{identifier}.receipt.json"


def _refresh_contract_receipt_hashes(contract_path: Path) -> None:
    contract = _read_json(contract_path)
    for provenance in contract["provenance"]:
        provenance["sha256"] = _sha(contract_path.parent / provenance["receipt_file"])
    _write_json(contract_path, contract)


def _bind_pre_append_receipt(contract_path: Path, receipt: dict[str, Any]) -> None:
    contract = _read_json(contract_path)
    contract["pre_append_receipt_sha256"] = hashlib.sha256(
        _MODULE._canonical_bytes(receipt)
    ).hexdigest()
    _write_json(contract_path, contract)


def _refresh_receipt_payload_hash(
    base: Path,
    contract_path: Path,
    identifier: str,
) -> None:
    receipt_path = _receipt_path(base, identifier)
    receipt = _read_json(receipt_path)
    payload_path = receipt_path.parent / receipt["payload"]["path"]
    receipt["payload"]["sha256"] = _sha(payload_path)
    _write_json(receipt_path, receipt)
    _refresh_contract_receipt_hashes(contract_path)


def _add_funding_wrapper(
    base: Path,
    contract_path: Path,
    identifier: str = "funding-copy",
    *,
    exchange: str = "binance",
) -> None:
    receipt_path = _receipt_path(base, identifier)
    receipt = _read_json(_receipt_path(base, "funding"))
    receipt["provenance_id"] = identifier
    receipt["exchange"] = exchange
    _write_json(receipt_path, receipt)
    contract = _read_json(contract_path)
    contract["provenance"].append(
        {
            "id": identifier,
            "kind": "official_exchange_api_receipt",
            "exchange": exchange,
            "receipt_file": receipt_path.name,
            "sha256": _sha(receipt_path),
        }
    )
    _write_json(contract_path, contract)


def _setup(tmp_path: Path) -> tuple[Path, Path, Path]:
    root = tmp_path / "db"
    root.mkdir()
    source_root = tmp_path / "source"
    source_root.mkdir()
    inventory_sha256 = "a" * 64
    source = {
        "uri": "file://registry",
        "retrieved_at_ms": 0,
        "payload_sha256": "0" * 64,
    }
    registry = {
        "schema_version": 1,
        "source": source,
        "symbols": [
            {"symbol": "BTCUSDT", "onboard_ms": 0, "delivery_ms": None},
            {"symbol": "ETHUSDT", "onboard_ms": 0, "delivery_ms": None},
        ],
    }
    lifecycle = {
        "registry": registry,
        "fold_membership": {
            "schema_version": 1,
            "registry_sha256": _MODULE.registry_sha256(registry),
            "source": source,
            "folds": [
                {
                    "fold_id": "all",
                    "start_ms": 0,
                    "end_ms": 180_000,
                    "eligible_symbols": ["BTCUSDT", "ETHUSDT"],
                    "partial_symbols": [],
                    "inactive_symbols": [],
                }
            ],
        },
    }
    lifecycle_path = tmp_path / "lifecycle.json"
    _write_json(lifecycle_path, lifecycle)
    inventory_path = tmp_path / "inventory.json"
    _write_json(
        inventory_path,
        {
            "artifact_kind": "recovery_inventory_decision",
            "schemaVersion": 1,
            "market_root": str(root.resolve()),
            "pairs": [{"symbol": "BTCUSDT", "timeframe": "1m"}],
            "snapshot_inventory": {
                "before_scans_sha256": inventory_sha256,
                "after_scans_sha256": inventory_sha256,
                "stable_across_scans": True,
            },
            "source_inventory_sha256": inventory_sha256,
            "source_root": str(source_root.resolve()),
            "synthetic_source_contract": {
                "passed": True,
                "selected_root_csv_count": 0,
            },
        },
    )
    funding_path = tmp_path / "funding.json"
    _write_json(
        funding_path,
        [{"symbol": "BTCUSDT", "fundingTime": point, "fundingRate": "0.0"} for point in _POINTS],
    )
    receipts = {
        "inventory": {
            "provenance_kind": "canonical_local_copy",
            "data_kind": "canonical_inventory",
            "source_uri": f"file://{inventory_path}",
            "payload": inventory_path,
            "timeframes": ["1m"],
        },
        "funding": {
            "provenance_kind": "official_exchange_api_receipt",
            "data_kind": "official_funding",
            "source_uri": "https://fapi.binance.com/fapi/v1/fundingRate",
            "payload": funding_path,
            "timeframes": [],
        },
    }
    for identifier, details in receipts.items():
        payload = details["payload"]
        _write_json(
            _receipt_path(tmp_path, identifier),
            {
                "artifact_kind": "research_data_provenance_receipt",
                "schema_version": 1,
                "provenance_id": identifier,
                "provenance_kind": details["provenance_kind"],
                "exchange": "binance",
                "data_kind": details["data_kind"],
                "source_uri": details["source_uri"],
                "symbols": ["BTCUSDT"],
                "timeframes": details["timeframes"],
                "start_ms": 0,
                "end_exclusive_ms": 180_000,
                "captured_at": "2026-01-01T00:00:00+00:00",
                "payload": {"path": payload.name, "sha256": _sha(payload)},
            },
        )
    contract = {
        "artifact_kind": "research_data_contract_manifest",
        "schema_version": 1,
        "contract_id": "contract",
        "lifecycle_manifest_sha256": _sha(lifecycle_path),
        "common_owned_interval": {"start_ms": 0, "end_exclusive_ms": 180_000},
        "session_calendar": "utc_24x7",
        "pre_append_receipt_sha256": None,
        "provenance": [
            {
                "id": "inventory",
                "kind": "canonical_local_copy",
                "exchange": "binance",
                "receipt_file": "inventory.receipt.json",
                "sha256": _sha(_receipt_path(tmp_path, "inventory")),
            },
            {
                "id": "funding",
                "kind": "official_exchange_api_receipt",
                "exchange": "binance",
                "receipt_file": "funding.receipt.json",
                "sha256": _sha(_receipt_path(tmp_path, "funding")),
            },
        ],
        "ohlcv_series": [
            {
                "id": "bars",
                "exchange": "binance",
                "symbol": "BTCUSDT",
                "timeframe": "1m",
                "start_ms": 0,
                "end_exclusive_ms": 180_000,
                "step_ms": 60_000,
                "anchor_ms": 0,
                "physical_layout": "partitioned_ohlcv",
                "provenance_ids": ["inventory"],
                "fold_ids": ["all"],
            }
        ],
        "funding_series": [
            {
                "id": "funding",
                "exchange": "binance",
                "symbol": "BTCUSDT",
                "interval": "perpetual",
                "start_ms": 0,
                "end_exclusive_ms": 180_000,
                "provenance_ids": ["funding"],
                "fold_ids": ["all"],
                "row_sources": {"official": "funding"},
                "schedule": [
                    {
                        "start_ms": 0,
                        "end_exclusive_ms": 180_000,
                        "cadence_ms": 60_000,
                        "first_settlement_ms": 0,
                        "tolerance_ms": 1,
                        "provenance_id": "funding",
                    }
                ],
            }
        ],
    }
    contract_path = tmp_path / "contract.json"
    _write_json(contract_path, contract)
    return root, contract_path, lifecycle_path


def _validate(
    root: Path,
    contract: Path,
    lifecycle: Path,
    bars: pl.DataFrame,
    funding: pl.DataFrame,
    mode: str = "pre_append",
    pre: object = None,
    *,
    physical: pl.DataFrame | None = None,
) -> dict[str, Any]:
    return _MODULE.validate_research_data_contract(
        root,
        contract,
        lifecycle,
        mode,
        pre_append_receipt=pre,
        physical_loader=lambda *_args, **_kwargs: physical if physical is not None else bars,
        ohlcv_loader=lambda *_args, **_kwargs: bars,
        funding_loader=lambda *_args, **_kwargs: funding,
    )


def _valid_receipt(
    tmp_path: Path,
) -> tuple[Path, Path, Path, pl.DataFrame, pl.DataFrame]:
    root, contract, lifecycle = _setup(tmp_path)
    return root, contract, lifecycle, _bars(_POINTS), _funding(_POINTS)


def test_canonical_variable_timeframe_and_preappend_triage(tmp_path: Path) -> None:
    root, contract_path, lifecycle_path, _bars_ok, _funding_ok = _valid_receipt(tmp_path)
    lifecycle = _read_json(lifecycle_path)
    lifecycle["fold_membership"] = build_fold_membership_manifest(
        lifecycle["registry"],
        [{"fold_id": "all", "start_ms": 0, "end_ms": 900_000}],
    )
    _write_json(lifecycle_path, lifecycle)
    for identifier in ("inventory", "funding"):
        receipt_path = _receipt_path(tmp_path, identifier)
        receipt = _read_json(receipt_path)
        receipt["end_exclusive_ms"] = 900_000
        if identifier == "inventory":
            receipt["timeframes"] = ["5m"]
        _write_json(receipt_path, receipt)
    inventory_payload = _read_json(tmp_path / "inventory.json")
    inventory_payload["pairs"][0]["timeframe"] = "5m"
    _write_json(tmp_path / "inventory.json", inventory_payload)
    _write_json(
        tmp_path / "funding.json",
        [
            {"symbol": "BTCUSDT", "fundingTime": point, "fundingRate": "0.0"}
            for point in (0, 300_000, 600_000)
        ],
    )
    _refresh_receipt_payload_hash(tmp_path, contract_path, "inventory")
    _refresh_receipt_payload_hash(tmp_path, contract_path, "funding")
    contract = _read_json(contract_path)
    contract["lifecycle_manifest_sha256"] = _sha(lifecycle_path)
    contract["common_owned_interval"]["end_exclusive_ms"] = 900_000
    contract["ohlcv_series"][0].update(
        {"timeframe": "5m", "step_ms": 300_000, "end_exclusive_ms": 900_000}
    )
    contract["funding_series"][0]["end_exclusive_ms"] = 900_000
    contract["funding_series"][0]["schedule"][0].update(
        {"end_exclusive_ms": 900_000, "cadence_ms": 300_000}
    )
    _write_json(contract_path, contract)
    _refresh_contract_receipt_hashes(contract_path)
    receipt = _validate(
        root,
        contract_path,
        lifecycle_path,
        _bars([0, 300_000, 600_000]),
        _funding([0, 300_000, 600_000]),
        "pre-append",
    )
    assert receipt["mode"] == "pre_append"
    assert receipt["decision_class"] == "triage"
    assert receipt["decision"] == "SAFE_TAIL_APPEND"
    assert receipt["admission_eligible"] is False


def test_preappend_complete_and_suffix_are_triage_not_admission(tmp_path: Path) -> None:
    root, contract, lifecycle, bars, funding = _valid_receipt(tmp_path)
    complete = _validate(root, contract, lifecycle, bars, funding)
    suffix = _validate(root, contract, lifecycle, bars.slice(0, 2), funding.slice(0, 2))
    assert complete["passed"] is True
    assert complete["decision"] == "SAFE_TAIL_APPEND"
    assert complete["admission_eligible"] is False
    assert suffix["passed"] is True
    assert suffix["series"][0]["missing_tail_count"] == 1
    assert suffix["funding"][0]["missing_tail_count"] == 1


@pytest.mark.parametrize(
    ("points", "field", "expected"),
    [
        ([60_000, 120_000], "missing_prefix_count", 1),
        ([0, 120_000], "interior_gap_count", 1),
        ([0, 60_000], "missing_tail_count", 1),
        ([0, 60_000, 90_000, 120_000], "off_grid_count", 1),
        ([-60_000, 0, 60_000, 120_000], "out_of_range_count", 1),
        ([0, 60_000, 60_000, 120_000], "duplicate_count", 1),
        ([60_000, 0, 120_000], "nonmonotone_count", 1),
    ],
)
def test_ohlcv_grid_and_order_failures_are_observable(
    tmp_path: Path,
    points: list[int],
    field: str,
    expected: int,
) -> None:
    root, contract, lifecycle, _bars_ok, funding = _valid_receipt(tmp_path)
    receipt = _validate(root, contract, lifecycle, _bars(points), funding)
    series = receipt["series"][0]
    expected_decision = "SAFE_TAIL_APPEND" if field == "missing_tail_count" else "STOP"
    assert receipt["decision"] == expected_decision
    assert series[field] == expected
    assert series["safe_tail_append"] is False or field == "missing_tail_count"


def test_ohlcv_nonfinite_invalid_values_and_view_parity_fail(tmp_path: Path) -> None:
    root, contract, lifecycle, bars, funding = _valid_receipt(tmp_path)
    nonfinite = _validate(
        root,
        contract,
        lifecycle,
        _bars(_POINTS, highs=[1.0, float("nan"), 1.0]),
        funding,
    )
    invalid = _validate(
        root,
        contract,
        lifecycle,
        _bars(_POINTS, highs=[1.0, 0.0, 1.0]),
        funding,
    )
    mismatch = _validate(
        root,
        contract,
        lifecycle,
        bars,
        funding,
        physical=_bars(_POINTS, close=2.0),
    )
    assert nonfinite["series"][0]["nonfinite_count"] == 1
    assert nonfinite["series"][0]["passed"] is False
    assert invalid["series"][0]["physical_ohlcv_validation"]["passed"] is False
    assert invalid["series"][0]["passed"] is False
    assert mismatch["series"][0]["physical_repository_mismatch_count"] == 1
    assert mismatch["series"][0]["passed"] is False


def test_partitioned_parquet_default_loader_and_unknown_route(tmp_path: Path) -> None:
    root = tmp_path / "market"
    partition = root / "exchange=binance" / "symbol=BTCUSDT" / "timeframe=1m" / "date=1970-01-01"
    partition.mkdir(parents=True)
    frame = _bars([0]).with_columns(pl.col("datetime").dt.replace_time_zone(None))
    frame.write_parquet(partition / "bars.parquet")
    loaded = load_strict_ohlcv_route(
        root,
        storage_route="partitioned_ohlcv",
        exchange="binance",
        symbol="BTCUSDT",
        timeframe="1m",
        start_date=0,
        end_date=0,
    )
    assert loaded.height == 1
    assert loaded.get_column("datetime")[0] == datetime(1970, 1, 1)
    with pytest.raises(ValueError, match="root"):
        load_strict_ohlcv_route(
            tmp_path / "empty",
            storage_route="partitioned_ohlcv",
            exchange="binance",
            symbol="BTCUSDT",
            timeframe="1m",
        )
    empty_root = tmp_path / "empty"
    empty_root.mkdir()
    with pytest.raises(FileNotFoundError):
        load_strict_ohlcv_route(
            empty_root,
            storage_route="partitioned_ohlcv",
            exchange="binance",
            symbol="BTCUSDT",
            timeframe="1m",
        )
    with pytest.raises(ValueError, match="unknown"):
        load_strict_ohlcv_route(
            root,
            storage_route="unknown",
            exchange="binance",
            symbol="BTCUSDT",
            timeframe="1m",
        )


@pytest.mark.parametrize(
    ("mutation", "detail"),
    [
        ("receipt_hash", "provenance receipt hash mismatch"),
        ("origin", "provenance source URI is not an approved origin"),
        ("payload_symbol", "official funding payload semantics"),
        ("payload_interval", "official funding payload semantics"),
        (
            "inventory_root",
            "canonical inventory payload does not bind separated data roots",
        ),
        ("inventory_pair", "canonical inventory omits requested symbol/timeframe"),
        ("inventory_unstable", "canonical inventory snapshot is not stable"),
        ("inventory_synthetic", "synthetic-source exclusion failed"),
    ],
)
def test_provenance_receipt_and_payload_bindings_fail_closed(
    tmp_path: Path,
    mutation: str,
    detail: str,
) -> None:
    root, contract_path, lifecycle_path, bars, funding = _valid_receipt(tmp_path)
    if mutation == "receipt_hash":
        contract = _read_json(contract_path)
        contract["provenance"][0]["sha256"] = "f" * 64
        _write_json(contract_path, contract)
    elif mutation == "origin":
        receipt_path = _receipt_path(tmp_path, "funding")
        receipt = _read_json(receipt_path)
        receipt["source_uri"] = "https://example.invalid/funding"
        _write_json(receipt_path, receipt)
        _refresh_contract_receipt_hashes(contract_path)
    elif mutation == "payload_symbol":
        payload_path = tmp_path / "funding.json"
        payload = _read_json(payload_path)
        payload[0]["symbol"] = "ETHUSDT"
        _write_json(payload_path, payload)
        _refresh_receipt_payload_hash(tmp_path, contract_path, "funding")
    elif mutation == "payload_interval":
        payload_path = tmp_path / "funding.json"
        payload = _read_json(payload_path)
        payload[0]["fundingTime"] = 180_000
        _write_json(payload_path, payload)
        _refresh_receipt_payload_hash(tmp_path, contract_path, "funding")
    elif mutation == "inventory_root":
        inventory_path = tmp_path / "inventory.json"
        inventory = _read_json(inventory_path)
        inventory["market_root"] = str(tmp_path / "other")
        _write_json(inventory_path, inventory)
        _refresh_receipt_payload_hash(tmp_path, contract_path, "inventory")
    elif mutation == "inventory_unstable":
        inventory_path = tmp_path / "inventory.json"
        inventory = _read_json(inventory_path)
        inventory["snapshot_inventory"]["after_scans_sha256"] = "b" * 64
        _write_json(inventory_path, inventory)
        _refresh_receipt_payload_hash(tmp_path, contract_path, "inventory")
    elif mutation == "inventory_synthetic":
        inventory_path = tmp_path / "inventory.json"
        inventory = _read_json(inventory_path)
        inventory["synthetic_source_contract"]["selected_root_csv_count"] = 1
        _write_json(inventory_path, inventory)
        _refresh_receipt_payload_hash(tmp_path, contract_path, "inventory")
    else:
        inventory_path = tmp_path / "inventory.json"
        inventory = _read_json(inventory_path)
        inventory["pairs"] = []
        _write_json(inventory_path, inventory)
        _refresh_receipt_payload_hash(tmp_path, contract_path, "inventory")
    receipt = _validate(root, contract_path, lifecycle_path, bars, funding)
    assert receipt["passed"] is False
    assert receipt["issues"][0]["code"] == "manifest_or_lifecycle_invalid"
    assert detail in receipt["issues"][0]["detail"]


def test_raw_binance_funding_time_and_string_rate_are_accepted(tmp_path: Path) -> None:
    root, contract, lifecycle, bars, funding = _valid_receipt(tmp_path)
    receipt = _validate(root, contract, lifecycle, bars, funding)
    assert receipt["funding"][0]["actual_settlement_count"] == 3
    assert receipt["funding"][0]["expected_settlement_count"] == 3
    assert receipt["funding"][0]["passed"] is True


@pytest.mark.parametrize("forgery", ["eligible", "category", "source", "order"])
def test_d04_forgery_stays_invalid_after_outer_hash_refresh(
    tmp_path: Path,
    forgery: str,
) -> None:
    root, contract_path, lifecycle_path, bars, funding = _valid_receipt(tmp_path)
    lifecycle = _read_json(lifecycle_path)
    fold = lifecycle["fold_membership"]["folds"][0]
    if forgery == "eligible":
        fold["eligible_symbols"] = []
    elif forgery == "category":
        fold["partial_symbols"] = ["BTCUSDT"]
    elif forgery == "source":
        lifecycle["fold_membership"]["source"]["uri"] = "file://forged"
    else:
        fold["eligible_symbols"] = list(reversed(fold["eligible_symbols"]))
    lifecycle["fold_membership"]["registry_sha256"] = _MODULE.registry_sha256(lifecycle["registry"])
    _write_json(lifecycle_path, lifecycle)
    contract = _read_json(contract_path)
    contract["lifecycle_manifest_sha256"] = _sha(lifecycle_path)
    _write_json(contract_path, contract)
    receipt = _validate(root, contract_path, lifecycle_path, bars, funding)
    assert receipt["d04_status"]["passed"] is False
    assert receipt["issues"][0]["code"] == "manifest_or_lifecycle_invalid"


def test_fold_must_own_the_entire_series_interval(tmp_path: Path) -> None:
    root, contract_path, lifecycle_path, bars, funding = _valid_receipt(tmp_path)
    lifecycle = _read_json(lifecycle_path)
    lifecycle["fold_membership"]["folds"][0]["end_ms"] = 120_000
    lifecycle["fold_membership"] = build_fold_membership_manifest(
        lifecycle["registry"],
        [{"fold_id": "all", "start_ms": 0, "end_ms": 120_000}],
    )
    _write_json(lifecycle_path, lifecycle)
    contract = _read_json(contract_path)
    contract["lifecycle_manifest_sha256"] = _sha(lifecycle_path)
    _write_json(contract_path, contract)
    receipt = _validate(root, contract_path, lifecycle_path, bars, funding)
    assert receipt["series"][0]["error"] == ("fold does not own the complete series interval")
    assert receipt["funding"][0]["error"] == ("fold does not own the complete series interval")


@pytest.mark.parametrize(
    ("frame", "field", "expected"),
    [
        (
            _funding(
                [0, 42, 60_000, 120_000],
                rates=[0.0, None, 0.0, 0.0],
                sources=["official", "feature", "official", "official"],
            ),
            "actual_settlement_count",
            3,
        ),
        (
            _funding([0, 60_000, 120_000], rates=[0.0, None, 0.0]),
            "interior_gap_count",
            1,
        ),
        (
            _funding(
                [0, 60_000, 120_000],
                sources=["official", "forged", "official"],
            ),
            "invalid_source_count",
            1,
        ),
        (_funding([1, 60_001, 120_001]), "unexpected_timestamp_count", 0),
        (_funding([2, 60_000, 120_000]), "unexpected_timestamp_count", 1),
        (
            _funding(_POINTS, rates=[0.0, 0.01, 0.0]),
            "funding_rate_mismatch_count",
            1,
        ),
        (_funding([0, 0, 60_000, 120_000]), "duplicate_count", 1),
    ],
)
def test_funding_settlement_accounting_edges(
    tmp_path: Path,
    frame: pl.DataFrame,
    field: str,
    expected: int,
) -> None:
    root, contract, lifecycle, bars, _funding_ok = _valid_receipt(tmp_path)
    receipt = _validate(root, contract, lifecycle, bars, frame)
    result = receipt["funding"][0]
    assert result[field] == expected
    if field == "actual_settlement_count" or (
        field == "unexpected_timestamp_count" and expected == 0
    ):
        assert result["passed"] is True
    else:
        assert result["passed"] is False


@pytest.mark.parametrize(
    "rate",
    [
        True,
        float("inf"),
    ],
)
def test_funding_bool_and_nonfinite_rates_are_rejected(
    tmp_path: Path,
    rate: object,
) -> None:
    root, contract, lifecycle, bars, _funding_ok = _valid_receipt(tmp_path)
    frame = pl.DataFrame(
        {
            "timestamp_ms": [0, 60_000, 120_000],
            "funding_rate": pl.Series(
                "funding_rate",
                [0.0, rate, 0.0],
                dtype=pl.Object,
            ),
            "source": ["official", "official", "official"],
        },
    )
    receipt = _validate(root, contract, lifecycle, bars, frame)
    assert receipt["funding"][0]["passed"] is False
    assert receipt["funding"][0]["nonfinite_count"] == 1


def test_overlapping_funding_tolerance_windows_are_rejected(tmp_path: Path) -> None:
    root, contract_path, lifecycle_path, bars, funding = _valid_receipt(tmp_path)
    contract = _read_json(contract_path)
    contract["funding_series"][0]["schedule"] = [
        {
            "start_ms": 0,
            "end_exclusive_ms": 60_000,
            "cadence_ms": 100_000,
            "first_settlement_ms": 0,
            "tolerance_ms": 49_999,
            "provenance_id": "funding",
        },
        {
            "start_ms": 60_000,
            "end_exclusive_ms": 180_000,
            "cadence_ms": 100_000,
            "first_settlement_ms": 60_000,
            "tolerance_ms": 49_999,
            "provenance_id": "funding",
        },
    ]
    _write_json(contract_path, contract)
    receipt = _validate(root, contract_path, lifecycle_path, bars, funding)
    assert receipt["funding"][0]["error"] == "funding tolerance windows overlap"
    assert receipt["issues"][0]["code"] == "funding_validation_exception"


def test_strict_requires_clean_safe_preappend_receipt(tmp_path: Path) -> None:
    root, contract, lifecycle, bars, funding = _valid_receipt(tmp_path)
    no_pre = _validate(root, contract, lifecycle, bars, funding, "post_append_strict")
    dirty_pre = _validate(
        root,
        contract,
        lifecycle,
        _bars([0, 120_000]),
        funding,
        "pre_append",
    )
    _bind_pre_append_receipt(contract, dirty_pre)
    strict = _validate(
        root,
        contract,
        lifecycle,
        bars,
        funding,
        "post_append_strict",
        dirty_pre,
    )
    assert no_pre["issues"][0]["code"] == "manifest_or_lifecycle_invalid"
    assert dirty_pre["decision"] == "STOP"
    assert strict["issues"][0]["detail"] == ("pre-append receipt is not a clean safe-tail decision")


def test_strict_rejects_historical_drift_and_accepts_suffix(tmp_path: Path) -> None:
    root, contract, lifecycle, bars, funding = _valid_receipt(tmp_path)
    pre = _validate(
        root,
        contract,
        lifecycle,
        bars.slice(0, 2),
        funding.slice(0, 2),
    )
    _bind_pre_append_receipt(contract, pre)
    appended = _validate(
        root,
        contract,
        lifecycle,
        bars,
        funding,
        "post_append_strict",
        pre,
    )
    bar_drift = _validate(
        root,
        contract,
        lifecycle,
        _bars(_POINTS, close=2.0),
        funding,
        "post_append_strict",
        pre,
    )
    funding_drift = _validate(
        root,
        contract,
        lifecycle,
        bars,
        _funding(_POINTS, rates=[1.0, 0.0, 0.0]),
        "post_append_strict",
        pre,
    )
    assert appended["decision"] == "ADMIT"
    assert appended["admission_eligible"] is True
    assert "sealed OHLCV" in bar_drift["series"][0]["error"]
    assert "sealed funding" in funding_drift["funding"][0]["error"]


def test_strict_does_not_accept_an_interior_gap_stop_as_chain(tmp_path: Path) -> None:
    root, contract, lifecycle, bars, funding = _valid_receipt(tmp_path)
    stopped_pre = _validate(
        root,
        contract,
        lifecycle,
        _bars([0, 120_000]),
        funding,
    )
    _bind_pre_append_receipt(contract, stopped_pre)
    strict = _validate(
        root,
        contract,
        lifecycle,
        bars,
        funding,
        "post_append_strict",
        stopped_pre,
    )
    assert stopped_pre["decision"] == "STOP"
    assert stopped_pre["series"][0]["interior_gap_count"] == 1
    assert strict["decision"] == "STOP"
    assert strict["issues"][0]["detail"] == ("pre-append receipt is not a clean safe-tail decision")


def test_extra_official_funding_row_breaks_schedule_bijection(tmp_path: Path) -> None:
    root, contract, lifecycle, bars, funding = _valid_receipt(tmp_path)
    payload_path = tmp_path / "funding.json"
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    payload.append({"symbol": "BTCUSDT", "fundingTime": 30_000, "fundingRate": "0.0"})
    _write_json(payload_path, payload)
    _refresh_receipt_payload_hash(tmp_path, contract, "funding")

    receipt = _validate(root, contract, lifecycle, bars, funding)

    assert receipt["funding"][0]["passed"] is False
    assert (
        receipt["funding"][0]["error"] == "official funding evidence and schedule are not bijective"
    )


def test_unowned_other_symbol_funding_row_stops_contract(tmp_path: Path) -> None:
    root, contract, lifecycle, bars, funding = _valid_receipt(tmp_path)
    payload_path = tmp_path / "funding.json"
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    payload.append({"symbol": "ETHUSDT", "fundingTime": 0, "fundingRate": "0.0"})
    _write_json(payload_path, payload)
    receipt_path = _receipt_path(tmp_path, "funding")
    provenance_receipt = _read_json(receipt_path)
    provenance_receipt["symbols"].append("ETHUSDT")
    provenance_receipt["payload"]["sha256"] = _sha(payload_path)
    _write_json(receipt_path, provenance_receipt)
    _refresh_contract_receipt_hashes(contract)

    receipt = _validate(root, contract, lifecycle, bars, funding)

    assert receipt["funding"][0]["passed"] is True
    assert receipt["issues"] == [{"code": "unowned_official_funding_evidence", "count": 1}]
    assert receipt["decision"] == "STOP"


def test_unreferenced_duplicate_funding_wrapper_stops_contract(tmp_path: Path) -> None:
    root, contract, lifecycle, bars, funding = _valid_receipt(tmp_path)
    _add_funding_wrapper(tmp_path, contract)

    receipt = _validate(root, contract, lifecycle, bars, funding)

    assert receipt["funding"][0]["passed"] is True
    assert receipt["issues"] == [{"code": "unowned_official_funding_evidence", "count": 3}]
    assert receipt["decision"] == "STOP"


def test_shifted_slot_duplicate_wrapper_cannot_alias_events(tmp_path: Path) -> None:
    root, contract_path, lifecycle, bars, funding = _valid_receipt(tmp_path)
    _add_funding_wrapper(tmp_path, contract_path)
    contract = _read_json(contract_path)
    clone = json.loads(json.dumps(contract["funding_series"][0]))
    clone["id"] = "funding-copy"
    clone["provenance_ids"] = ["funding-copy"]
    clone["row_sources"] = {"official": "funding-copy"}
    clone["schedule"][0]["provenance_id"] = "funding-copy"
    clone["schedule"][0]["first_settlement_ms"] = 1
    contract["funding_series"].append(clone)
    _write_json(contract_path, contract)

    receipt = _validate(root, contract_path, lifecycle, bars, funding)

    assert receipt["funding"][0]["passed"] is True
    assert receipt["funding"][1]["error"] == "duplicate semantic funding series"
    assert receipt["decision"] == "STOP"


def test_partially_overlapping_funding_series_cannot_split_common_window(
    tmp_path: Path,
) -> None:
    root, contract_path, lifecycle, bars, funding = _valid_receipt(tmp_path)
    contract = _read_json(contract_path)
    clone = json.loads(json.dumps(contract["funding_series"][0]))
    clone["id"] = "funding-overlap"
    clone["start_ms"] = 60_000
    clone["schedule"][0]["start_ms"] = 60_000
    clone["schedule"][0]["first_settlement_ms"] = 60_000
    contract["funding_series"].append(clone)
    _write_json(contract_path, contract)

    receipt = _validate(root, contract_path, lifecycle, bars, funding)

    assert receipt["funding"][0]["passed"] is True
    assert receipt["funding"][1]["error"] == ("series must use the exact common owned interval")
    assert receipt["decision"] == "STOP"


@pytest.mark.parametrize("mode", ["pre_append", "post_append_strict"])
def test_funding_exchange_relabel_cannot_create_ownership_namespace(
    tmp_path: Path, mode: str
) -> None:
    root, contract_path, lifecycle, bars, funding = _valid_receipt(tmp_path)
    _add_funding_wrapper(tmp_path, contract_path, exchange="binance-shadow")
    contract = _read_json(contract_path)
    clone = json.loads(json.dumps(contract["funding_series"][0]))
    clone["id"] = "funding-shadow"
    clone["exchange"] = "binance-shadow"
    clone["provenance_ids"] = ["funding-copy"]
    clone["row_sources"] = {"official": "funding-copy"}
    clone["schedule"][0]["provenance_id"] = "funding-copy"
    contract["funding_series"].append(clone)
    _write_json(contract_path, contract)
    pre: object = None
    if mode == "post_append_strict":
        pre = {}
        _bind_pre_append_receipt(contract_path, pre)

    receipt = _validate(
        root,
        contract_path,
        lifecycle,
        bars,
        funding,
        mode,
        pre,
    )

    assert receipt["decision"] == "STOP"
    assert receipt["issues"][0] == {
        "code": "manifest_or_lifecycle_invalid",
        "detail": "provenance source URI is not an approved origin",
    }


def test_inventory_exchange_relabel_cannot_create_ownership_namespace(
    tmp_path: Path,
) -> None:
    root, contract_path, lifecycle, bars, funding = _valid_receipt(tmp_path)
    receipt_path = _receipt_path(tmp_path, "inventory-copy")
    provenance_receipt = _read_json(_receipt_path(tmp_path, "inventory"))
    provenance_receipt["provenance_id"] = "inventory-copy"
    provenance_receipt["exchange"] = "binance-shadow"
    _write_json(receipt_path, provenance_receipt)
    contract = _read_json(contract_path)
    contract["provenance"].append(
        {
            "id": "inventory-copy",
            "kind": "canonical_local_copy",
            "exchange": "binance-shadow",
            "receipt_file": receipt_path.name,
            "sha256": _sha(receipt_path),
        }
    )
    clone = json.loads(json.dumps(contract["ohlcv_series"][0]))
    clone["id"] = "bars-shadow"
    clone["exchange"] = "binance-shadow"
    clone["provenance_ids"] = ["inventory-copy"]
    contract["ohlcv_series"].append(clone)
    _write_json(contract_path, contract)

    receipt = _validate(root, contract_path, lifecycle, bars, funding)

    assert receipt["decision"] == "STOP"
    assert receipt["issues"][0] == {
        "code": "manifest_or_lifecycle_invalid",
        "detail": "provenance source URI is not an approved origin",
    }


@pytest.mark.parametrize("mode", ["pre_append", "post_append_strict"])
@pytest.mark.parametrize("alias", ["timestamp", "rate"])
def test_conflicting_funding_payload_aliases_fail_closed(
    tmp_path: Path, mode: str, alias: str
) -> None:
    root, contract, lifecycle, bars, funding = _valid_receipt(tmp_path)
    payload_path = tmp_path / "funding.json"
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    if alias == "timestamp":
        payload[0]["timestamp_ms"] = 1
        expected_detail = "funding timestamp aliases conflict"
    else:
        payload[0]["funding_rate"] = 0.0
        payload[0]["fundingRate"] = "0.01"
        expected_detail = "funding rate aliases conflict"
    _write_json(payload_path, payload)
    _refresh_receipt_payload_hash(tmp_path, contract, "funding")
    pre: object = None
    if mode == "post_append_strict":
        pre = {}
        _bind_pre_append_receipt(contract, pre)

    receipt = _validate(root, contract, lifecycle, bars, funding, mode, pre)

    assert receipt["decision"] == "STOP"
    assert receipt["issues"][0] == {
        "code": "manifest_or_lifecycle_invalid",
        "detail": expected_detail,
    }


def test_matching_funding_payload_aliases_are_unambiguous(tmp_path: Path) -> None:
    root, contract, lifecycle, bars, funding = _valid_receipt(tmp_path)
    payload_path = tmp_path / "funding.json"
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    payload[0]["timestamp_ms"] = payload[0]["fundingTime"]
    payload[0]["funding_rate"] = float(payload[0]["fundingRate"])
    _write_json(payload_path, payload)
    _refresh_receipt_payload_hash(tmp_path, contract, "funding")

    receipt = _validate(root, contract, lifecycle, bars, funding)

    assert receipt["decision"] == "SAFE_TAIL_APPEND"


@pytest.mark.parametrize("mutation", ["missing", "extra", "wrong_id", "zero_count"])
def test_strict_rejects_unbound_or_empty_pre_seals(tmp_path: Path, mutation: str) -> None:
    root, contract, lifecycle, bars, funding = _valid_receipt(tmp_path)
    pre = _validate(root, contract, lifecycle, bars, funding)
    forged = json.loads(json.dumps(pre))
    if mutation == "missing":
        forged["seals"].pop("funding:funding")
    elif mutation == "extra":
        forged["seals"]["ohlcv:extra"] = dict(forged["seals"]["ohlcv:bars"])
    elif mutation == "wrong_id":
        forged["seals"]["ohlcv:bars"]["id"] = "wrong"
    else:
        forged["seals"]["ohlcv:bars"]["row_count"] = 0
    _bind_pre_append_receipt(contract, forged)

    strict = _validate(
        root,
        contract,
        lifecycle,
        bars,
        funding,
        "post_append_strict",
        forged,
    )

    assert strict["decision"] == "STOP"
    assert strict["issues"][0]["code"] == "manifest_or_lifecycle_invalid"
    assert "seal" in strict["issues"][0]["detail"]


@pytest.mark.parametrize("metric_kind", ["ohlcv", "funding"])
def test_strict_rejects_rebound_positive_pre_seal_truncation(
    tmp_path: Path, metric_kind: str
) -> None:
    root, contract, lifecycle, bars, funding = _valid_receipt(tmp_path)
    pre = _validate(root, contract, lifecycle, bars, funding)
    forged = json.loads(json.dumps(pre))
    if metric_kind == "ohlcv":
        forged["series"][0]["rows"] = 1
        seal = forged["seals"]["ohlcv:bars"]
        seal["row_count"] = 1
        seal["row_value_sha256"] = _MODULE._frame_digest(bars.slice(0, 1), _MODULE._OHLCV_COLUMNS)
    else:
        forged["funding"][0]["actual_settlement_count"] = 1
        seal = forged["seals"]["funding:funding"]
        seal["row_count"] = 1
        seal["row_value_sha256"] = _MODULE._frame_digest(
            funding.slice(0, 1), _MODULE._FUNDING_COLUMNS
        )
    _bind_pre_append_receipt(contract, forged)

    strict = _validate(
        root,
        contract,
        lifecycle,
        bars,
        funding,
        "post_append_strict",
        forged,
    )

    assert strict["decision"] == "STOP"
    assert "counts are internally inconsistent" in strict["issues"][0]["detail"]


@pytest.mark.parametrize("metric_kind", ["ohlcv", "funding"])
def test_strict_rejects_metric_and_seal_digest_disagreement(
    tmp_path: Path, metric_kind: str
) -> None:
    root, contract, lifecycle, bars, funding = _valid_receipt(tmp_path)
    pre = _validate(root, contract, lifecycle, bars, funding)
    forged = json.loads(json.dumps(pre))
    if metric_kind == "ohlcv":
        forged["series"][0]["physical_row_value_sha256"] = "0" * 64
        forged["series"][0]["repository_row_value_sha256"] = "0" * 64
    else:
        forged["funding"][0]["settlement_row_value_sha256"] = "0" * 64
    _bind_pre_append_receipt(contract, forged)

    strict = _validate(
        root,
        contract,
        lifecycle,
        bars,
        funding,
        "post_append_strict",
        forged,
    )

    assert strict["decision"] == "STOP"
    assert strict["issues"][0]["detail"] == (
        "pre-append receipt seal identity, count, or digest is invalid"
    )


def test_strict_rejects_extra_pre_metric_fields(tmp_path: Path) -> None:
    root, contract, lifecycle, bars, funding = _valid_receipt(tmp_path)
    pre = _validate(root, contract, lifecycle, bars, funding)
    forged = json.loads(json.dumps(pre))
    forged["series"][0]["unbound_metric"] = 0
    _bind_pre_append_receipt(contract, forged)

    strict = _validate(
        root,
        contract,
        lifecycle,
        bars,
        funding,
        "post_append_strict",
        forged,
    )

    assert strict["decision"] == "STOP"
    assert strict["issues"][0]["detail"] == ("pre-append OHLCV metric schema is invalid")


@pytest.mark.parametrize(
    ("mutation", "expected_detail"),
    [
        ("extra_top", "pre-append receipt schema is invalid"),
        ("extra_input", "pre-append receipt input schema is invalid"),
        ("schema_version", "pre-append receipt identity or mode is invalid"),
        ("admission_eligible", "pre-append receipt identity or mode is invalid"),
        ("decision_class", "pre-append receipt identity or mode is invalid"),
        ("validation_layer", "pre-append receipt identity or mode is invalid"),
        ("d04_status", "pre-append receipt D-04 status is invalid"),
        ("generated_at", "pre-append receipt generation time is invalid"),
        ("future_generated_at", "pre-append receipt generation time is invalid"),
        ("contract_locator", "pre-append receipt manifest chain is invalid"),
        ("lifecycle_locator", "pre-append receipt manifest chain is invalid"),
        ("contract_manifest_sha256", "pre-append receipt manifest chain is invalid"),
        ("db_path", "pre-append receipt manifest chain is invalid"),
        ("contract_core_sha256", "pre-append receipt manifest chain is invalid"),
        ("lifecycle_manifest_sha256", "pre-append receipt manifest chain is invalid"),
    ],
)
def test_strict_rejects_contradictory_pre_receipt_semantics(
    tmp_path: Path, mutation: str, expected_detail: str
) -> None:
    root, contract, lifecycle, bars, funding = _valid_receipt(tmp_path)
    forged = json.loads(json.dumps(_validate(root, contract, lifecycle, bars, funding)))
    if mutation == "extra_top":
        forged["unbound"] = None
    elif mutation == "extra_input":
        forged["input"]["unbound"] = None
    elif mutation == "schema_version":
        forged["schema_version"] = 2
    elif mutation == "admission_eligible":
        forged["admission_eligible"] = True
    elif mutation == "decision_class":
        forged["decision_class"] = "admission"
    elif mutation == "validation_layer":
        forged["validation_layer"] = ["repository_view", "physical"]
    elif mutation == "d04_status":
        forged["d04_status"]["passed"] = 1
    elif mutation == "generated_at":
        forged["generated_at_utc"] = "2026-01-01T00:00:00"
    elif mutation == "future_generated_at":
        forged["generated_at_utc"] = "2099-01-01T00:00:00+00:00"
    elif mutation == "contract_locator":
        forged["input"]["contract_manifest"] = "other-contract.json"
    elif mutation == "lifecycle_locator":
        forged["input"]["lifecycle_manifest"] = "other-lifecycle.json"
    elif mutation == "db_path":
        forged["input"]["db_path"] = "other-db"
    elif mutation == "contract_core_sha256":
        forged["input"]["contract_core_sha256"] = "0" * 64
    elif mutation == "lifecycle_manifest_sha256":
        forged["input"]["lifecycle_manifest_sha256"] = "0" * 64
    else:
        forged["input"]["contract_manifest_sha256"] = "0" * 64
    _bind_pre_append_receipt(contract, forged)

    strict = _validate(
        root,
        contract,
        lifecycle,
        bars,
        funding,
        "post_append_strict",
        forged,
    )

    assert strict["decision"] == "STOP"
    assert strict["issues"][0] == {
        "code": "manifest_or_lifecycle_invalid",
        "detail": expected_detail,
    }


@pytest.mark.parametrize("rows", [-1, "3", 2])
def test_strict_rejects_invalid_or_contradictory_pre_funding_rows(
    tmp_path: Path, rows: object
) -> None:
    root, contract, lifecycle, bars, funding = _valid_receipt(tmp_path)
    forged = json.loads(json.dumps(_validate(root, contract, lifecycle, bars, funding)))
    forged["funding"][0]["rows"] = rows
    _bind_pre_append_receipt(contract, forged)

    strict = _validate(
        root,
        contract,
        lifecycle,
        bars,
        funding,
        "post_append_strict",
        forged,
    )

    assert strict["decision"] == "STOP"
    assert strict["issues"][0]["code"] == "manifest_or_lifecycle_invalid"
    assert "pre.funding.rows" in strict["issues"][0]["detail"] or (
        strict["issues"][0]["detail"] == "pre-append funding counts are internally inconsistent"
    )


@pytest.mark.parametrize("mutation", ["extra", "columns", "boolean", "rows"])
def test_strict_rejects_contradictory_nested_ohlcv_metrics(tmp_path: Path, mutation: str) -> None:
    root, contract, lifecycle, bars, funding = _valid_receipt(tmp_path)
    forged = json.loads(json.dumps(_validate(root, contract, lifecycle, bars, funding)))
    report = forged["series"][0]["physical_ohlcv_validation"]
    if mutation == "extra":
        report["metrics"]["unbound"] = 0
    elif mutation == "columns":
        report["metrics"]["required_columns"] = ["datetime"]
    elif mutation == "boolean":
        report["metrics"]["require_monotonic"] = 1
    else:
        report["rows"] = "3"
    _bind_pre_append_receipt(contract, forged)

    strict = _validate(
        root,
        contract,
        lifecycle,
        bars,
        funding,
        "post_append_strict",
        forged,
    )

    assert strict["decision"] == "STOP"
    assert strict["issues"][0]["code"] == "manifest_or_lifecycle_invalid"
    assert (
        "pre-append OHLCV validation report is inconsistent" in strict["issues"][0]["detail"]
        or "pre.ohlcv.physical_ohlcv_validation.rows" in strict["issues"][0]["detail"]
    )


def test_strict_contract_hash_rejects_mutated_pre_receipt(tmp_path: Path) -> None:
    root, contract, lifecycle, bars, funding = _valid_receipt(tmp_path)
    pre = _validate(root, contract, lifecycle, bars, funding)
    _bind_pre_append_receipt(contract, pre)
    forged = json.loads(json.dumps(pre))
    forged["generated_at_utc"] = "2099-01-01T00:00:00+00:00"

    strict = _validate(
        root,
        contract,
        lifecycle,
        bars,
        funding,
        "post_append_strict",
        forged,
    )

    assert strict["decision"] == "STOP"
    assert strict["issues"][0]["detail"] == ("pre-append receipt hash does not match contract")


def test_unsupported_calendar_is_malformed_and_fail_closed(tmp_path: Path) -> None:
    root, contract_path, lifecycle, bars, funding = _valid_receipt(tmp_path)
    contract = _read_json(contract_path)
    contract["session_calendar"] = "NYSE"
    _write_json(contract_path, contract)

    receipt = _validate(root, contract_path, lifecycle, bars, funding)

    assert receipt["decision"] == "STOP"
    assert receipt["admission_eligible"] is False
    assert receipt["issues"][0] == {
        "code": "manifest_or_lifecycle_invalid",
        "detail": "contract manifest schema is invalid",
    }


def test_duplicate_official_payload_row_is_not_one_slot_evidence(tmp_path: Path) -> None:
    root, contract, lifecycle, bars, funding = _valid_receipt(tmp_path)
    payload_path = tmp_path / "funding.json"
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    payload.append(dict(payload[0]))
    _write_json(payload_path, payload)
    _refresh_receipt_payload_hash(tmp_path, contract, "funding")

    receipt = _validate(root, contract, lifecycle, bars, funding)

    assert (
        receipt["funding"][0]["error"]
        == "official funding evidence does not own each schedule slot"
    )


def test_cloned_ohlcv_series_cannot_alias_semantic_data(tmp_path: Path) -> None:
    root, contract_path, lifecycle, bars, funding = _valid_receipt(tmp_path)
    contract = _read_json(contract_path)
    clone = json.loads(json.dumps(contract["ohlcv_series"][0]))
    clone["id"] = "bars-copy"
    contract["ohlcv_series"].append(clone)
    _write_json(contract_path, contract)

    receipt = _validate(root, contract_path, lifecycle, bars, funding)

    assert receipt["series"][0]["passed"] is True
    assert receipt["series"][1]["error"] == "duplicate semantic OHLCV series"
    assert receipt["decision"] == "STOP"


def test_cloned_funding_series_cannot_alias_semantic_data(tmp_path: Path) -> None:
    root, contract_path, lifecycle, bars, funding = _valid_receipt(tmp_path)
    contract = _read_json(contract_path)
    clone = json.loads(json.dumps(contract["funding_series"][0]))
    clone["id"] = "funding-copy"
    contract["funding_series"].append(clone)
    _write_json(contract_path, contract)

    receipt = _validate(root, contract_path, lifecycle, bars, funding)

    assert receipt["funding"][0]["passed"] is True
    assert receipt["funding"][1]["error"] == "duplicate semantic funding series"
    assert receipt["decision"] == "STOP"


def test_pre_receipt_ghost_metric_and_seal_are_not_contract_owned(tmp_path: Path) -> None:
    root, contract, lifecycle, bars, funding = _valid_receipt(tmp_path)
    pre = _validate(root, contract, lifecycle, bars, funding)
    forged = json.loads(json.dumps(pre))
    forged["series"].append({"id": "ghost", "rows": 1})
    forged["seals"]["ohlcv:ghost"] = {
        "id": "ghost",
        "row_count": 1,
        "row_value_sha256": "0" * 64,
    }
    _bind_pre_append_receipt(contract, forged)

    strict = _validate(
        root,
        contract,
        lifecycle,
        bars,
        funding,
        "post_append_strict",
        forged,
    )

    assert strict["decision"] == "STOP"
    assert strict["issues"][0]["detail"] == (
        "pre-append receipt metrics do not match contract series"
    )


def test_multi_source_funding_handoff_and_source_swap(tmp_path: Path) -> None:
    root, contract_path, lifecycle, bars, _ = _valid_receipt(tmp_path)
    first_payload_path = tmp_path / "funding.json"
    first_payload = json.loads(first_payload_path.read_text(encoding="utf-8"))
    _write_json(first_payload_path, first_payload[:1])
    _refresh_receipt_payload_hash(tmp_path, contract_path, "funding")

    second_payload_path = tmp_path / "funding-second.json"
    _write_json(second_payload_path, first_payload[1:])
    second_receipt_path = _receipt_path(tmp_path, "funding-second")
    second_receipt = _read_json(_receipt_path(tmp_path, "funding"))
    second_receipt["provenance_id"] = "funding-second"
    second_receipt["payload"] = {
        "path": second_payload_path.name,
        "sha256": _sha(second_payload_path),
    }
    _write_json(second_receipt_path, second_receipt)

    contract = _read_json(contract_path)
    contract["provenance"].append(
        {
            "id": "funding-second",
            "kind": "official_exchange_api_receipt",
            "exchange": "binance",
            "receipt_file": second_receipt_path.name,
            "sha256": _sha(second_receipt_path),
        }
    )
    funding_series = contract["funding_series"][0]
    funding_series["provenance_ids"] = ["funding", "funding-second"]
    funding_series["row_sources"] = {
        "source-a": "funding",
        "source-b": "funding-second",
    }
    funding_series["schedule"] = [
        {
            "start_ms": 0,
            "end_exclusive_ms": 60_000,
            "cadence_ms": 60_000,
            "first_settlement_ms": 0,
            "tolerance_ms": 1,
            "provenance_id": "funding",
        },
        {
            "start_ms": 60_000,
            "end_exclusive_ms": 180_000,
            "cadence_ms": 60_000,
            "first_settlement_ms": 60_000,
            "tolerance_ms": 1,
            "provenance_id": "funding-second",
        },
    ]
    _write_json(contract_path, contract)

    correct = _funding(
        _POINTS,
        sources=["source-a", "source-b", "source-b"],
    )
    swapped = _funding(
        _POINTS,
        sources=["source-b", "source-a", "source-b"],
    )
    passed = _validate(root, contract_path, lifecycle, bars, correct)
    stopped = _validate(root, contract_path, lifecycle, bars, swapped)

    assert passed["decision"] == "SAFE_TAIL_APPEND"
    assert stopped["funding"][0]["invalid_source_count"] == 2
    assert stopped["decision"] == "STOP"


def test_cli_help_and_output_inside_db_are_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(sys, "argv", ["validator", "--help"])
    with pytest.raises(SystemExit) as help_exit:
        _MODULE.main()
    assert help_exit.value.code == 0
    assert "--db-path" in capsys.readouterr().out
    root, contract, lifecycle, _bars_ok, _funding_ok = _valid_receipt(tmp_path)
    output = root / "receipt.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validator",
            "--db-path",
            str(root),
            "--contract-manifest",
            str(contract),
            "--lifecycle-manifest",
            str(lifecycle),
            "--mode",
            "pre-append",
            "--output-json",
            str(output),
        ],
    )
    assert _MODULE.main() == 1
    printed = json.loads(capsys.readouterr().out)
    assert printed["issues"][0]["code"] == "internal_or_malformed_failure"
    assert output.exists() is False


def test_cli_valid_stop_is_exit_two_and_malformed_is_exit_one(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root, contract, lifecycle, _bars_ok, _funding_ok = _valid_receipt(tmp_path)
    arguments = [
        "validator",
        "--db-path",
        str(root),
        "--contract-manifest",
        str(contract),
        "--lifecycle-manifest",
        str(lifecycle),
        "--mode",
        "pre-append",
    ]
    monkeypatch.setattr(sys, "argv", arguments)
    assert _MODULE.main() == 2
    assert json.loads(capsys.readouterr().out)["decision"] == "STOP"
    contract.write_text("{", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", arguments)
    assert _MODULE.main() == 1
    assert json.loads(capsys.readouterr().out)["issues"][0]["code"] == (
        "manifest_or_lifecycle_invalid"
    )


def test_cli_atomic_persistence_failure_prints_one_stop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root, contract, lifecycle, _bars_ok, _funding_ok = _valid_receipt(tmp_path)
    output = tmp_path / "receipt.json"

    def fail_atomic(_path: Path, _payload: object) -> None:
        raise OSError("simulated persistence failure")

    monkeypatch.setattr(_MODULE, "_atomic_json", fail_atomic)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validator",
            "--db-path",
            str(root),
            "--contract-manifest",
            str(contract),
            "--lifecycle-manifest",
            str(lifecycle),
            "--mode",
            "pre-append",
            "--output-json",
            str(output),
        ],
    )
    assert _MODULE.main() == 1
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 1
    receipt = json.loads(lines[0])
    assert receipt["decision"] == "STOP"
    assert receipt["issues"][0]["code"] == "internal_or_malformed_failure"
    assert output.exists() is False
