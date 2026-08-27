from __future__ import annotations

import hashlib
import json
import math
import queue
from collections import deque
from dataclasses import FrozenInstanceError
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

import lumina_quant.research.alpha_max_evidence as evidence
from lumina_quant.backtesting.data_windowed_parquet import (
    HistoricParquetWindowedDataHandler,
)
from lumina_quant.backtesting.execution_model import ExecutionModel, ExecutionModelConfig
from lumina_quant.backtesting.portfolio_backtest import Portfolio
from lumina_quant.data.feature_points import FeaturePoint
from lumina_quant.research.alpha_max_evidence import (
    ALPHA_MAX_CANDIDATE_SYMBOLS,
    AlphaMaxFundingBoundaryRequest,
    AlphaMaxFundingBoundaryResolver,
    AlphaMaxOrderedFundingLookup,
    FeatureRootSpec,
    allocate_alpha_max_equal_risk,
    allocate_alpha_max_equal_weight,
    allocate_alpha_max_shrunk_hrp,
    validate_alpha_max_admission_artifact,
    validate_alpha_max_admitted_symbols,
)


_HASH_A = "a" * 64
_HASH_B = "b" * 64


def _symbols_sha256(symbols) -> str:
    canonical = json.dumps(
        list(symbols),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    return hashlib.sha256(canonical).hexdigest()


def _admission_statistics(*, passes: bool) -> dict[str, object]:
    return {
        "daily_quote_notional_day_count": 517,
        "median_quote_notional_usdt": 25_000_000.0 if passes else 19_000_000.0,
        "p10_quote_notional_usdt": 3_000_000.0,
        "consecutive_completed_daily_bars_before_train": 366,
        "readable_monotone_unique_finite_partitions": True,
        "complete_train_daily_keys": True,
        "complete_train_4h_keys": True,
        "causal_funding_coverage_complete": True,
        "unresolved_daily_cross_section_count": 0,
    }


def _admission_payload(
    admitted=ALPHA_MAX_CANDIDATE_SYMBOLS[:5],
) -> dict[str, object]:
    admitted = tuple(admitted)
    admitted_set = set(admitted)
    per_candidate = {}
    for symbol in ALPHA_MAX_CANDIDATE_SYMBOLS:
        passes = symbol in admitted_set
        per_candidate[symbol] = {
            "admitted": passes,
            "reasons": [] if passes else ["median_quote_notional_below_minimum"],
            "statistics": _admission_statistics(passes=passes),
        }
    return {
        "artifact_kind": "alpha_max_train_admission.v1",
        "phase": "train_admission",
        "selection_inputs": ["warmup", "train"],
        "input_root_hashes": {"warmup": _HASH_A, "train": _HASH_B},
        "candidate_symbols": list(ALPHA_MAX_CANDIDATE_SYMBOLS),
        "candidate_symbols_sha256": _symbols_sha256(ALPHA_MAX_CANDIDATE_SYMBOLS),
        "admitted_symbols": list(admitted),
        "admitted_symbols_sha256": _symbols_sha256(admitted),
        "per_candidate": per_candidate,
    }


class _FakeFeatureLookup:
    points_by_path: dict[
        str,
        dict[str, FeaturePoint | None | dict[int, FeaturePoint | None]],
    ] = {}
    calls: list[tuple[str, str, str, int]] = []

    def __init__(self, *, db_path, exchange, start_date, end_date):
        self.db_path = db_path
        self.exchange = exchange
        self.start_date = start_date
        self.end_date = end_date

    def get_latest_point(self, symbol, field, *, timestamp_ms):
        type(self).calls.append((self.db_path, symbol, field, timestamp_ms))
        point = type(self).points_by_path.get(self.db_path, {}).get(symbol)
        if isinstance(point, dict):
            return point.get(timestamp_ms)
        return point


def _spec(tmp_path: Path, root_id: str) -> FeatureRootSpec:
    path = tmp_path / root_id
    path.mkdir(exist_ok=True)
    start, end = evidence._ROOT_INTERVALS[root_id]
    return FeatureRootSpec(
        root_id=root_id,
        path=str(path),
        exchange="binance",
        start_utc=start,
        end_utc=end,
        inventory_sha256=_HASH_A,
        content_sha256=_HASH_B,
    )


def _lookup(monkeypatch, tmp_path: Path, roots=("warmup", "train")):
    _FakeFeatureLookup.points_by_path = {}
    _FakeFeatureLookup.calls = []
    monkeypatch.setattr(evidence, "FeaturePointLookup", _FakeFeatureLookup)
    specs = tuple(_spec(tmp_path, root_id) for root_id in roots)
    return AlphaMaxOrderedFundingLookup(specs), specs


def test_feature_root_spec_is_exact_canonical_and_immutable(tmp_path):
    spec = _spec(tmp_path, "warmup")

    assert spec.path == str((tmp_path / "warmup").resolve())
    assert spec.start_utc == datetime(2022, 12, 31, tzinfo=UTC)
    assert spec.end_utc == datetime(2024, 1, 1, tzinfo=UTC)
    with pytest.raises(FrozenInstanceError):
        spec.root_id = "train"
    with pytest.raises(ValueError, match="must_be_absolute"):
        FeatureRootSpec(
            "warmup",
            "relative",
            "binance",
            spec.start_utc,
            spec.end_utc,
            _HASH_A,
            _HASH_B,
        )
    with pytest.raises(ValueError, match="frozen_bounds_mismatch"):
        FeatureRootSpec(
            "warmup",
            spec.path,
            "binance",
            spec.start_utc + timedelta(seconds=1),
            spec.end_utc,
            _HASH_A,
            _HASH_B,
        )
    with pytest.raises(ValueError, match="exchange"):
        FeatureRootSpec(
            "warmup",
            spec.path,
            "other",
            spec.start_utc,
            spec.end_utc,
            _HASH_A,
            _HASH_B,
        )


def test_ordered_lookup_requires_exact_adjacent_root_contract(monkeypatch, tmp_path):
    monkeypatch.setattr(evidence, "FeaturePointLookup", _FakeFeatureLookup)
    warmup = _spec(tmp_path, "warmup")
    train = _spec(tmp_path, "train")
    purge = _spec(tmp_path, "purge")

    lookup = AlphaMaxOrderedFundingLookup((warmup, train))
    assert lookup.root_specs == (warmup, train)
    assert lookup.ordered_root_ids == ("warmup", "train")
    with pytest.raises(AttributeError, match="immutable"):
        lookup._root_specs = (train,)
    with pytest.raises(ValueError, match="immediately_adjacent"):
        AlphaMaxOrderedFundingLookup((warmup, purge))
    with pytest.raises(ValueError, match="immediately_adjacent"):
        AlphaMaxOrderedFundingLookup((warmup, train, purge))
    with pytest.raises(ValueError, match="immediately_adjacent"):
        AlphaMaxOrderedFundingLookup((train,))


def test_ordered_lookup_selects_newest_causal_cross_boundary_point(monkeypatch, tmp_path):
    lookup, specs = _lookup(monkeypatch, tmp_path)
    boundary_ms = specs[-1].start_timestamp_ms
    _FakeFeatureLookup.points_by_path = {
        specs[0].path: {"BTCUSDT": FeaturePoint(0.0001, boundary_ms - 1)},
        specs[1].path: {"BTCUSDT": None},
    }

    assert lookup.get_latest_point(
        "BTCUSDT", "funding_rate", timestamp_ms=boundary_ms
    ) == FeaturePoint(0.0001, boundary_ms - 1)
    _FakeFeatureLookup.points_by_path[specs[1].path]["BTCUSDT"] = FeaturePoint(
        0.0002, boundary_ms + 2
    )
    assert lookup.get_latest(
        "BTCUSDT", "funding_rate", timestamp_ms=boundary_ms + 2
    ) == pytest.approx(0.0002)
    assert all(call[2] == "funding_rate" for call in _FakeFeatureLookup.calls)


def test_ordered_lookup_reads_actual_adjacent_parquet_roots_causally(tmp_path):
    pl = pytest.importorskip("polars")
    previous = _spec(tmp_path, "purge")
    current = _spec(tmp_path, "validation")
    boundary_ms = current.start_timestamp_ms

    def write_point(spec, timestamp_ms: int, rate: float) -> None:
        timestamp = datetime.fromtimestamp(timestamp_ms / 1000.0, UTC)
        directory = (
            Path(spec.path)
            / "feature_points"
            / "exchange=binance"
            / "symbol=BTCUSDT"
            / f"date={timestamp:%Y-%m-%d}"
        )
        directory.mkdir(parents=True, exist_ok=True)
        pl.DataFrame(
            {
                "timestamp_ms": [timestamp_ms],
                "source_timestamp_ms": [timestamp_ms + 500],
                "funding_rate": [rate],
            }
        ).write_parquet(directory / "part-0.parquet")

    write_point(previous, boundary_ms - 1, 0.0001)
    write_point(current, boundary_ms + 1, 0.0002)
    lookup = AlphaMaxOrderedFundingLookup((previous, current))

    assert (
        lookup.get_latest_point(
            "BTCUSDT",
            "funding_rate",
            timestamp_ms=boundary_ms,
        )
        is None
    )
    assert lookup.get_latest_point(
        "BTCUSDT",
        "funding_rate",
        timestamp_ms=boundary_ms + 499,
    ) == FeaturePoint(0.0001, boundary_ms + 499, boundary_ms - 1)
    assert lookup.get_latest_point(
        "BTCUSDT",
        "funding_rate",
        timestamp_ms=boundary_ms + 501,
    ) == FeaturePoint(0.0002, boundary_ms + 501, boundary_ms + 1)


def test_ordered_lookup_rejects_field_query_bound_stale_future_owned_and_tie_poison(
    monkeypatch, tmp_path
):
    lookup, specs = _lookup(monkeypatch, tmp_path)
    query_ms = specs[-1].start_timestamp_ms

    with pytest.raises(ValueError, match="field_forbidden"):
        lookup.get_latest_point("BTCUSDT", "mark_price", timestamp_ms=query_ms)
    with pytest.raises(ValueError, match="outside_current_root"):
        lookup.get_latest_point("BTCUSDT", "funding_rate", timestamp_ms=query_ms - 1)

    _FakeFeatureLookup.points_by_path = {
        specs[0].path: {
            "BTCUSDT": FeaturePoint(0.1, query_ms - evidence.FEATURE_POINT_MAX_STALE_MS - 1)
        }
    }
    with pytest.raises(ValueError, match="stale"):
        lookup.get_latest_point("BTCUSDT", "funding_rate", timestamp_ms=query_ms)

    _FakeFeatureLookup.points_by_path = {
        specs[1].path: {"BTCUSDT": FeaturePoint(0.1, query_ms + 1)}
    }
    with pytest.raises(ValueError, match="future"):
        lookup.get_latest_point("BTCUSDT", "funding_rate", timestamp_ms=query_ms)

    _FakeFeatureLookup.points_by_path = {
        specs[1].path: {
            "BTCUSDT": FeaturePoint(
                0.1,
                query_ms + evidence._FUNDING_SOURCE_MAX_JITTER_MS + 1,
                query_ms,
            )
        }
    }
    with pytest.raises(ValueError, match="source_timestamp_invalid"):
        lookup.get_latest_point("BTCUSDT", "funding_rate", timestamp_ms=query_ms)

    _FakeFeatureLookup.points_by_path = {
        specs[0].path: {"BTCUSDT": FeaturePoint(0.1, query_ms)},
    }
    with pytest.raises(ValueError, match="outside_owned_root"):
        lookup.get_latest_point("BTCUSDT", "funding_rate", timestamp_ms=query_ms)

    _FakeFeatureLookup.points_by_path = {
        specs[0].path: {"BTCUSDT": FeaturePoint(0.1, query_ms)},
        specs[1].path: {"BTCUSDT": FeaturePoint(0.2, query_ms)},
    }
    with pytest.raises(ValueError, match="equal_timestamp_conflict"):
        lookup.get_latest_point("BTCUSDT", "funding_rate", timestamp_ms=query_ms)


def test_admission_validation_freezes_exact_ten_to_lexicographic_five_or_ten():
    admitted = ALPHA_MAX_CANDIDATE_SYMBOLS[:5]
    payload = _admission_payload(admitted)
    canonical = (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
        + b"\n"
    )

    artifact = validate_alpha_max_admission_artifact(
        canonical, expected_sha256=hashlib.sha256(canonical).hexdigest()
    )
    assert artifact.candidate_symbols is ALPHA_MAX_CANDIDATE_SYMBOLS
    assert artifact.admitted_symbols == admitted
    assert artifact.canonical_bytes == canonical
    assert payload["candidate_symbols_sha256"] == _symbols_sha256(ALPHA_MAX_CANDIDATE_SYMBOLS)
    assert payload["admitted_symbols_sha256"] == _symbols_sha256(admitted)
    with pytest.raises(FrozenInstanceError):
        artifact.sha256 = _HASH_A

    ten = _admission_payload(ALPHA_MAX_CANDIDATE_SYMBOLS)
    assert len(validate_alpha_max_admission_artifact(ten).admitted_symbols) == 10


@pytest.mark.parametrize(
    ("candidates", "admitted", "reason"),
    [
        (ALPHA_MAX_CANDIDATE_SYMBOLS[::-1], ALPHA_MAX_CANDIDATE_SYMBOLS[:5], "candidate"),
        (ALPHA_MAX_CANDIDATE_SYMBOLS, ALPHA_MAX_CANDIDATE_SYMBOLS[:4], "count"),
        (
            ALPHA_MAX_CANDIDATE_SYMBOLS,
            (*ALPHA_MAX_CANDIDATE_SYMBOLS[:4], ALPHA_MAX_CANDIDATE_SYMBOLS[3]),
            "lexicographic_unique",
        ),
        (
            ALPHA_MAX_CANDIDATE_SYMBOLS,
            (*ALPHA_MAX_CANDIDATE_SYMBOLS[:4], "ZZZUSDT"),
            "outside_candidates",
        ),
    ],
)
def test_admitted_symbol_validation_rejects_substitution_and_shrinkage(
    candidates, admitted, reason
):
    with pytest.raises(ValueError, match=reason):
        validate_alpha_max_admitted_symbols(candidates, admitted)


def test_admission_validation_rejects_noncanonical_or_nontrain_inputs():
    base = _admission_payload()
    noncanonical = json.dumps(base, indent=2).encode()
    with pytest.raises(ValueError, match="not_canonical"):
        validate_alpha_max_admission_artifact(noncanonical)
    with pytest.raises(ValueError, match="not_train_only"):
        validate_alpha_max_admission_artifact(dict(base, phase="validation"))
    with pytest.raises(ValueError, match="selection_inputs"):
        validate_alpha_max_admission_artifact(dict(base, selection_inputs=["train", "validation"]))
    with pytest.raises(ValueError, match="not_warmup_train"):
        validate_alpha_max_admission_artifact(
            dict(base, input_root_hashes={"train": _HASH_A, "validation": _HASH_B})
        )


def test_admission_validation_rejects_minimal_membership_only_payload():
    with pytest.raises(ValueError, match="key_set_mismatch"):
        validate_alpha_max_admission_artifact(
            {
                "candidate_symbols": list(ALPHA_MAX_CANDIDATE_SYMBOLS),
                "admitted_symbols": list(ALPHA_MAX_CANDIDATE_SYMBOLS[:5]),
            }
        )


@pytest.mark.parametrize(
    "missing_key",
    [
        "artifact_kind",
        "phase",
        "selection_inputs",
        "input_root_hashes",
        "candidate_symbols_sha256",
        "admitted_symbols_sha256",
        "per_candidate",
    ],
)
def test_admission_validation_rejects_partial_or_extra_schema(missing_key):
    payload = _admission_payload()
    del payload[missing_key]
    with pytest.raises(ValueError, match="key_set_mismatch"):
        validate_alpha_max_admission_artifact(payload)

    payload = _admission_payload()
    payload["validation_metric"] = 1.0
    with pytest.raises(ValueError, match="key_set_mismatch"):
        validate_alpha_max_admission_artifact(payload)


def test_admission_validation_rejects_kind_root_and_symbol_hash_poison():
    payload = _admission_payload()
    payload["artifact_kind"] = "alpha_max_validation_admission.v1"
    with pytest.raises(ValueError, match="artifact_kind_invalid"):
        validate_alpha_max_admission_artifact(payload)

    payload = _admission_payload()
    payload["input_root_hashes"] = {"warmup": _HASH_A, "train": "not-a-hash"}
    with pytest.raises(ValueError, match="train_root_sha256_invalid"):
        validate_alpha_max_admission_artifact(payload)

    payload = _admission_payload()
    payload["input_root_hashes"] = {"warmup": _HASH_A, "train": int("1" * 64)}
    with pytest.raises(ValueError, match="train_root_sha256_invalid"):
        validate_alpha_max_admission_artifact(payload)

    payload = _admission_payload()
    payload["candidate_symbols_sha256"] = "c" * 64
    with pytest.raises(ValueError, match="candidate_symbols_sha256_mismatch"):
        validate_alpha_max_admission_artifact(payload)

    payload = _admission_payload()
    payload["admitted_symbols_sha256"] = "c" * 64
    with pytest.raises(ValueError, match="admitted_symbols_sha256_mismatch"):
        validate_alpha_max_admission_artifact(payload)


def test_admission_validation_rejects_incomplete_candidate_evidence_and_tautological_status():
    payload = _admission_payload()
    del payload["per_candidate"][ALPHA_MAX_CANDIDATE_SYMBOLS[-1]]
    with pytest.raises(ValueError, match="per_candidate_coverage_mismatch"):
        validate_alpha_max_admission_artifact(payload)

    payload = _admission_payload()
    rejected_symbol = ALPHA_MAX_CANDIDATE_SYMBOLS[-1]
    payload["per_candidate"][rejected_symbol]["reasons"] = []
    with pytest.raises(ValueError, match="candidate_reasons_mismatch"):
        validate_alpha_max_admission_artifact(payload)

    payload = _admission_payload()
    admitted_symbol = ALPHA_MAX_CANDIDATE_SYMBOLS[0]
    payload["per_candidate"][admitted_symbol]["statistics"]["causal_funding_coverage_complete"] = (
        False
    )
    payload["per_candidate"][admitted_symbol]["reasons"] = ["incomplete_causal_funding_coverage"]
    with pytest.raises(ValueError, match="candidate_membership_mismatch"):
        validate_alpha_max_admission_artifact(payload)


def test_admission_validation_rejects_statistics_shape_and_stale_whole_artifact_hash():
    payload = _admission_payload()
    symbol = ALPHA_MAX_CANDIDATE_SYMBOLS[0]
    del payload["per_candidate"][symbol]["statistics"]["p10_quote_notional_usdt"]
    with pytest.raises(ValueError, match="statistics_key_set_mismatch"):
        validate_alpha_max_admission_artifact(payload)

    payload = _admission_payload()
    payload["per_candidate"][symbol]["statistics"]["p10_quote_notional_usdt"] = 30_000_000.0
    with pytest.raises(ValueError, match="quote_notional_statistics_invalid"):
        validate_alpha_max_admission_artifact(payload)

    payload = _admission_payload()
    original = (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
        + b"\n"
    )
    payload["input_root_hashes"]["train"] = "c" * 64
    with pytest.raises(ValueError, match="sha256_mismatch"):
        validate_alpha_max_admission_artifact(
            payload,
            expected_sha256=hashlib.sha256(original).hexdigest(),
        )


def _handler(lookup, *, boundary_ms: int, symbols=ALPHA_MAX_CANDIDATE_SYMBOLS[:5]):
    handler = object.__new__(HistoricParquetWindowedDataHandler)
    handler._feature_lookup = lookup
    handler.backtest_window_seconds = 1
    handler._window_rows = {
        symbol: deque([(boundary_ms - 1000, 1.0, 1.0, 1.0, 100.0, 1.0)], maxlen=2)
        for symbol in symbols
    }
    handler._window_row_timestamps_ms = {
        symbol: deque([boundary_ms - 1000], maxlen=2) for symbol in symbols
    }
    return handler


def _execution_model():
    return ExecutionModel(
        ExecutionModelConfig(
            taker_fee_rate=0.0004,
            maker_fee_rate=0.0002,
            slippage_rate=0.0005,
            spread_rate=0.0002,
            leverage=1,
            margin_mode="isolated",
            maintenance_margin_rate=0.005,
            liquidation_buffer_rate=0.0,
            funding_rate_per_8h=0.0,
            funding_interval_hours=8,
            random_seed=42,
        )
    )


class _PortfolioFundingConfig:
    INITIAL_CAPITAL = 1000.0
    TAKER_FEE_RATE = 0.0004
    MAKER_FEE_RATE = 0.0002
    SLIPPAGE_RATE = 0.0005
    SPREAD_RATE = 0.0002
    LEVERAGE = 1
    MARGIN_MODE = "isolated"
    MAINTENANCE_MARGIN_RATE = 0.005
    LIQUIDATION_BUFFER_RATE = 0.0
    FUNDING_RATE_PER_8H = 0.0
    FUNDING_INTERVAL_HOURS = 8
    RANDOM_SEED = 42
    SIM_MAX_BAR_VOLUME_RATIO = 0.1
    MAX_DAILY_LOSS_PCT = 0.05


def _resolver_fixture(monkeypatch, tmp_path):
    lookup, specs = _lookup(monkeypatch, tmp_path)
    boundary_ms = specs[-1].start_timestamp_ms
    admitted = ALPHA_MAX_CANDIDATE_SYMBOLS[:5]
    _FakeFeatureLookup.points_by_path = {
        specs[-1].path: {symbol: FeaturePoint(0.001, boundary_ms) for symbol in admitted}
    }
    handler = _handler(lookup, boundary_ms=boundary_ms, symbols=admitted)
    resolver = AlphaMaxFundingBoundaryResolver(lookup, admitted)
    return resolver, lookup, handler, boundary_ms, admitted


def _production_funding_fixture(monkeypatch, tmp_path, *, include_second_price: bool):
    lookup, specs = _lookup(monkeypatch, tmp_path)
    root_start_ms = specs[-1].start_timestamp_ms
    boundaries = (
        root_start_ms + 8 * 60 * 60 * 1000,
        root_start_ms + 16 * 60 * 60 * 1000,
    )
    admitted = ALPHA_MAX_CANDIDATE_SYMBOLS[:5]
    symbols = admitted[:2]
    _FakeFeatureLookup.points_by_path = {
        specs[-1].path: {
            symbols[0]: {
                boundaries[0]: FeaturePoint(0.001, boundaries[0] - 123),
                boundaries[1]: FeaturePoint(-0.002, boundaries[1] - 456),
            },
            symbols[1]: {
                boundaries[0]: FeaturePoint(0.0005, boundaries[0] - 321),
                boundaries[1]: FeaturePoint(0.001, boundaries[1] - 654),
            },
        }
    }
    raw_rows_by_symbol = {
        symbols[0]: [
            (boundaries[0] - 1000, 1.0, 1.0, 1.0, 100.0, 1.0),
            (boundaries[1] - 1000, 1.0, 1.0, 1.0, 125.0, 1.0),
        ],
        symbols[1]: [
            (boundaries[0] - 1000, 1.0, 1.0, 1.0, 200.0, 1.0),
        ],
    }
    if include_second_price:
        raw_rows_by_symbol[symbols[1]].append((boundaries[1] - 1000, 1.0, 1.0, 1.0, 150.0, 1.0))
    handler = _handler(lookup, boundary_ms=boundaries[0], symbols=symbols)
    handler.symbol_list = list(symbols)
    for symbol, raw_rows in raw_rows_by_symbol.items():
        handler._window_rows[symbol] = deque(raw_rows, maxlen=4)
        handler._window_row_timestamps_ms[symbol] = deque(
            (row[0] for row in raw_rows),
            maxlen=4,
        )
    resolver = AlphaMaxFundingBoundaryResolver(lookup, admitted)
    start = datetime.fromtimestamp(root_start_ms / 1000, UTC)
    latest = datetime.fromtimestamp((boundaries[1] + 1000) / 1000, UTC)
    portfolio = Portfolio(
        handler,
        queue.Queue(),
        start,
        _PortfolioFundingConfig,
        record_history=False,
        track_metrics=False,
        record_trades=False,
        funding_boundary_resolver=resolver,
    )
    portfolio.current_positions[symbols[0]] = 2.0
    portfolio.current_positions[symbols[1]] = -3.0
    for symbol in symbols:
        portfolio._last_funding_ts[symbol] = start.timestamp()
    return portfolio, resolver, handler, symbols, boundaries, latest


def test_funding_resolver_preserves_lookup_and_admitted_tuple_identity(monkeypatch, tmp_path):
    resolver, lookup, _, _, admitted = _resolver_fixture(monkeypatch, tmp_path)
    assert resolver.ordered_lookup is lookup
    assert resolver.admitted_symbols is admitted
    with pytest.raises(AttributeError, match="constructor-bound"):
        resolver._ordered_lookup = lookup
    with pytest.raises(TypeError, match="must_be_tuple"):
        AlphaMaxFundingBoundaryResolver(lookup, list(admitted))


def test_funding_resolver_rejects_outside_domain_before_accessor_or_feature_lookup(
    monkeypatch, tmp_path
):
    resolver, _, _, boundary_ms, _ = _resolver_fixture(monkeypatch, tmp_path)
    _FakeFeatureLookup.calls = []

    with pytest.raises(ValueError, match="outside_admitted_domain"):
        resolver.resolve(
            symbol="XRPUSDT",
            boundary_ms=boundary_ms,
            qty=1.0,
            latest_datetime=datetime.fromtimestamp((boundary_ms + 1000) / 1000, UTC),
            raw_point_accessor=lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("raw accessor was touched")
            ),
        )
    assert _FakeFeatureLookup.calls == []
    assert resolver.ledger == ()


def test_funding_resolver_enforces_exact_bound_accessor_and_causal_points(monkeypatch, tmp_path):
    resolver, _, handler, boundary_ms, admitted = _resolver_fixture(monkeypatch, tmp_path)
    latest = datetime.fromtimestamp((boundary_ms + 1000) / 1000, UTC)
    row = resolver.resolve(
        symbol=admitted[0],
        boundary_ms=boundary_ms,
        qty=2.0,
        latest_datetime=latest,
        raw_point_accessor=handler.get_latest_raw_point,
    )
    assert row.symbol == admitted[0]
    assert row.rate_source_timestamp_ms == boundary_ms
    assert row.price_row_timestamp_ms == boundary_ms - 1000
    assert row.price_close_timestamp_ms == boundary_ms
    assert row.rate == pytest.approx(0.001)
    assert row.price == pytest.approx(100.0)
    assert row.payment is None
    assert resolver.ledger == ()
    with pytest.raises(FrozenInstanceError):
        row.price = 1.0

    with pytest.raises(ValueError, match="raw_accessor"):
        resolver.resolve(
            symbol=admitted[0],
            boundary_ms=boundary_ms + 8 * 60 * 60 * 1000,
            qty=2.0,
            latest_datetime=latest + timedelta(hours=8),
            raw_point_accessor=lambda *args, **kwargs: None,
        )


def test_funding_resolver_preserves_official_source_jitter_in_committed_ledger(
    monkeypatch,
    tmp_path,
) -> None:
    resolver, _, handler, boundary_ms, admitted = _resolver_fixture(monkeypatch, tmp_path)
    _FakeFeatureLookup.points_by_path[resolver.ordered_lookup.current_root.path][admitted[0]] = (
        FeaturePoint(0.001, boundary_ms + 500, boundary_ms)
    )
    latest = datetime.fromtimestamp((boundary_ms + 1000) / 1000, UTC)

    committed = resolver.resolve_batch(
        (AlphaMaxFundingBoundaryRequest(admitted[0], boundary_ms, 2.0, latest),),
        raw_point_accessor=handler.get_latest_raw_point,
        execution_model=_execution_model(),
    )

    assert committed[0].rate_source_timestamp_ms == boundary_ms + 500
    carried = resolver.carry_forward()
    assert carried.ledger == committed


def test_funding_resolver_batch_is_atomic_and_ledger_rows_are_immutable(monkeypatch, tmp_path):
    resolver, _, handler, boundary_ms, admitted = _resolver_fixture(monkeypatch, tmp_path)
    latest = datetime.fromtimestamp((boundary_ms + 1000) / 1000, UTC)
    requests = tuple(
        AlphaMaxFundingBoundaryRequest(symbol, boundary_ms, index + 1.0, latest)
        for index, symbol in enumerate(admitted[:2])
    )

    committed = resolver.resolve_batch(
        requests,
        raw_point_accessor=handler.get_latest_raw_point,
        execution_model=_execution_model(),
    )
    assert resolver.ledger is not committed
    assert resolver.ledger == committed
    assert tuple(row.symbol for row in committed) == tuple(sorted(admitted[:2]))
    assert committed[0].payment == pytest.approx(committed[0].qty * 100.0 * 0.001)
    with pytest.raises(FrozenInstanceError):
        committed[0].payment = 0.0

    before = resolver.ledger
    assert resolver._bound_raw_accessor_owner is handler
    with pytest.raises(AttributeError, match="constructor-bound"):
        resolver._ledger = ()
    with pytest.raises(AttributeError, match="constructor-bound"):
        resolver._bound_raw_accessor_owner = None
    with pytest.raises(AttributeError, match="constructor-bound"):
        resolver._locked = False
    assert resolver.ledger is before
    assert resolver._bound_raw_accessor_owner is handler

    with pytest.raises(ValueError, match="duplicate"):
        resolver.resolve_batch(
            requests,
            raw_point_accessor=handler.get_latest_raw_point,
            execution_model=_execution_model(),
        )
    assert resolver.ledger is before


def test_funding_resolver_failed_multisymbol_batch_mutates_nothing(monkeypatch, tmp_path):
    resolver, _, handler, boundary_ms, admitted = _resolver_fixture(monkeypatch, tmp_path)
    latest = datetime.fromtimestamp((boundary_ms + 1000) / 1000, UTC)
    del handler._window_rows[admitted[1]]
    del handler._window_row_timestamps_ms[admitted[1]]

    with pytest.raises(ValueError, match="funding_boundary_coverage"):
        resolver.resolve_batch(
            (
                AlphaMaxFundingBoundaryRequest(admitted[0], boundary_ms, 1.0, latest),
                AlphaMaxFundingBoundaryRequest(admitted[1], boundary_ms, 1.0, latest),
            ),
            raw_point_accessor=handler.get_latest_raw_point,
            execution_model=_execution_model(),
        )
    assert resolver.ledger == ()
    assert resolver._bound_raw_accessor_owner is None


def test_portfolio_production_seam_seals_and_reconciles_each_alpha_funding_boundary(
    monkeypatch, tmp_path
):
    portfolio, resolver, handler, symbols, boundaries, latest = _production_funding_fixture(
        monkeypatch,
        tmp_path,
        include_second_price=True,
    )

    portfolio._apply_funding(latest)

    ledger = resolver.ledger
    assert tuple((row.symbol, row.boundary_ms) for row in ledger) == (
        (symbols[0], boundaries[0]),
        (symbols[1], boundaries[0]),
        (symbols[0], boundaries[1]),
        (symbols[1], boundaries[1]),
    )
    assert tuple(row.rate_source_timestamp_ms for row in ledger) == (
        boundaries[0] - 123,
        boundaries[0] - 321,
        boundaries[1] - 456,
        boundaries[1] - 654,
    )
    assert tuple(row.price_close_timestamp_ms for row in ledger) == (
        boundaries[0],
        boundaries[0],
        boundaries[1],
        boundaries[1],
    )
    assert tuple(row.rate for row in ledger) == pytest.approx((0.001, 0.0005, -0.002, 0.001))
    assert tuple(row.price for row in ledger) == pytest.approx((100.0, 200.0, 125.0, 150.0))
    payments = tuple(row.payment for row in ledger)
    assert payments == pytest.approx((0.2, -0.3, -0.5, -0.45))
    assert all(type(payment) is float for payment in payments)
    payment_sum = math.fsum(payments)
    assert resolver._bound_raw_accessor_owner is handler
    assert portfolio.total_funding_paid == payment_sum
    assert portfolio.current_holdings["funding"] == payment_sum
    assert portfolio.current_holdings["cash"] == 1000.0 - payment_sum
    assert portfolio.current_holdings["total"] == 1000.0 - payment_sum
    assert {symbol: portfolio._last_funding_ts[symbol] for symbol in symbols} == dict.fromkeys(
        symbols, boundaries[1] / 1000
    )

    ledger_identity = resolver.ledger
    holdings_after_first_application = dict(portfolio.current_holdings)
    portfolio._apply_funding(latest)
    assert resolver.ledger is ledger_identity
    assert portfolio.current_holdings == holdings_after_first_application
    assert portfolio.total_funding_paid == payment_sum


def test_portfolio_production_seam_alpha_funding_failure_rolls_back_ledger_and_cash(
    monkeypatch, tmp_path
):
    portfolio, resolver, _, symbols, _, latest = _production_funding_fixture(
        monkeypatch,
        tmp_path,
        include_second_price=False,
    )
    holdings_before = dict(portfolio.current_holdings)
    anchors_before = {symbol: portfolio._last_funding_ts[symbol] for symbol in symbols}
    empty_ledger = resolver.ledger

    with pytest.raises(ValueError, match="funding_boundary_coverage"):
        portfolio._apply_funding(latest)

    assert resolver.ledger is empty_ledger
    assert resolver._bound_raw_accessor_owner is None
    assert portfolio.current_holdings == holdings_before
    assert portfolio.total_funding_paid == 0.0
    assert {symbol: portfolio._last_funding_ts[symbol] for symbol in symbols} == anchors_before


def _return_matrix(rows=252):
    index = np.arange(rows, dtype=np.float64)
    return np.column_stack(
        (
            np.sin(index / 7.0),
            np.cos(index / 11.0),
            2.0 * np.sin(index / 13.0 + 0.4),
        )
    )


def test_exact_equal_weight_erc_and_shrunk_hrp_wrappers_are_sorted_and_permutation_stable():
    ids = ("component_trend_1x", "component_carry_1x", "component_near_high_1x")
    matrix = _return_matrix()
    permutation = (2, 0, 1)
    permuted_ids = tuple(ids[index] for index in permutation)
    permuted_matrix = matrix[:, permutation]

    equal_weight = allocate_alpha_max_equal_weight(ids, per_component_cap=0.50)
    assert list(equal_weight) == sorted(ids)
    assert set(equal_weight.values()) == {0.3333333333}
    assert 0 <= 1.0 - math.fsum(equal_weight.values()) < 1e-9

    for allocator in (allocate_alpha_max_equal_risk, allocate_alpha_max_shrunk_hrp):
        original = allocator(ids, matrix, per_component_cap=0.50)
        permuted = allocator(permuted_ids, permuted_matrix, per_component_cap=0.50)
        assert original == permuted
        assert list(original) == sorted(ids)
        assert math.fsum(original.values()) == pytest.approx(1.0, abs=1e-9)
        assert max(original.values()) <= 0.50


def test_equal_risk_and_hrp_golden_risk_shapes_and_loo_cap():
    index = np.arange(252, dtype=np.float64)
    orthogonal = np.column_stack((np.sin(index), np.cos(index)))
    high_variance = np.column_stack((np.sin(index), 4.0 * np.cos(index)))
    ids = ("a", "b")

    equal = allocate_alpha_max_equal_risk(ids, orthogonal, per_component_cap=0.70)
    assert equal == pytest.approx({"a": 0.5, "b": 0.5}, abs=0.01)
    unequal = allocate_alpha_max_equal_risk(ids, high_variance, per_component_cap=0.70)
    assert unequal["b"] < unequal["a"]
    hrp = allocate_alpha_max_shrunk_hrp(ids, orthogonal, per_component_cap=0.70)
    assert hrp == pytest.approx({"a": 0.5, "b": 0.5}, abs=0.01)
    assert max(unequal.values()) <= 0.70


@pytest.mark.parametrize(
    "matrix",
    [
        np.ones((251, 2)),
        np.column_stack((np.arange(252), np.ones(252))),
        np.column_stack((np.arange(252), np.full(252, np.nan))),
    ],
)
def test_allocator_wrappers_fail_closed_without_fallback(matrix):
    with pytest.raises(ValueError, match="allocator_fit_invalid"):
        allocate_alpha_max_equal_risk(("a", "b"), matrix, per_component_cap=0.70)
    with pytest.raises(ValueError, match="allocator_fit_invalid"):
        allocate_alpha_max_shrunk_hrp(("a", "b"), matrix, per_component_cap=0.70)
    with pytest.raises(ValueError, match="cap_mismatch"):
        allocate_alpha_max_equal_weight(("a", "b"), per_component_cap=0.50)
