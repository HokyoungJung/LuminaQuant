from __future__ import annotations

from datetime import datetime

import polars as pl
import pytest

from lumina_quant.compute.ohlcv_loader import OHLCVFrameLoader, normalize_ohlcv_frame
from lumina_quant.compute.ohlcv_validation import OHLCVValidationError, validate_ohlcv_frame


def _valid_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "datetime": [datetime(2026, 1, 1), datetime(2026, 1, 2)],
            "open": [10.0, 11.0],
            "high": [12.0, 12.5],
            "low": [9.5, 10.5],
            "close": [11.0, 12.0],
            "volume": [100.0, 120.0],
        }
    )


def test_validate_ohlcv_frame_accepts_valid_data() -> None:
    report = validate_ohlcv_frame(_valid_frame())

    assert report.passed is True
    assert report.rows == 2
    assert report.issues == ()
    assert report.to_dict()["artifact_kind"] == "ohlcv_validation_report"


def test_validate_ohlcv_frame_reports_price_invariant_failures() -> None:
    frame = _valid_frame().with_columns(pl.Series("high", [8.0, 12.5]))

    report = validate_ohlcv_frame(frame)
    codes = {issue.code for issue in report.issues}

    assert report.passed is False
    assert "high_below_low" in codes
    assert "high_below_ohlc_member" in codes


def test_strict_loader_fails_closed_on_invalid_data() -> None:
    frame = _valid_frame().with_columns(pl.Series("volume", [100.0, -1.0]))
    loader = OHLCVFrameLoader(validation="strict")

    with pytest.raises(OHLCVValidationError) as exc_info:
        loader.normalize(frame)

    assert "volume_negative" in str(exc_info.value)
    assert exc_info.value.report.passed is False


def test_strict_validation_rejects_non_datetime_timestamp_dtype() -> None:
    frame = pl.DataFrame(
        {
            "datetime": ["not-a-date", "also-not-a-date"],
            "open": [10.0, 11.0],
            "high": [12.0, 12.5],
            "low": [9.5, 10.5],
            "close": [11.0, 12.0],
            "volume": [100.0, 120.0],
        }
    )

    report = validate_ohlcv_frame(frame)
    codes = {issue.code for issue in report.issues}

    assert report.passed is False
    assert "datetime_dtype_invalid" in codes


def test_strict_loader_validates_non_monotonic_source_before_sorting(tmp_path) -> None:
    frame = pl.DataFrame(
        {
            "datetime": [datetime(2026, 1, 2), datetime(2026, 1, 1)],
            "open": [11.0, 10.0],
            "high": [12.5, 12.0],
            "low": [10.5, 9.5],
            "close": [12.0, 11.0],
            "volume": [120.0, 100.0],
        }
    )
    path = tmp_path / "non_monotonic.csv"
    frame.write_csv(path)

    loader = OHLCVFrameLoader(validation="strict")
    with pytest.raises(OHLCVValidationError) as exc_info:
        loader.load_csv(str(path))

    assert "datetime_not_monotonic" in str(exc_info.value)


def test_validate_ohlcv_frame_checks_symbol_partition_order_and_duplicates() -> None:
    frame = pl.DataFrame(
        {
            "symbol": ["A", "A", "B", "B"],
            "datetime": [
                datetime(2026, 1, 2),
                datetime(2026, 1, 1),
                datetime(2026, 1, 1),
                datetime(2026, 1, 1),
            ],
            "open": [11.0, 10.0, 10.0, 10.0],
            "high": [12.5, 12.0, 12.0, 12.0],
            "low": [10.5, 9.5, 9.5, 9.5],
            "close": [12.0, 11.0, 11.0, 11.0],
            "volume": [120.0, 100.0, 100.0, 100.0],
        }
    )

    report = validate_ohlcv_frame(frame, symbol_column="symbol")
    codes = {issue.code for issue in report.issues}

    assert report.passed is False
    assert "datetime_not_monotonic" in codes
    assert "duplicate_timestamp" in codes


def test_strict_loader_does_not_eager_fallback_on_lazy_failure_by_default(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "valid.csv"
    _valid_frame().write_csv(path)

    def fail_scan_csv(*args: object, **kwargs: object) -> object:
        raise pl.exceptions.ComputeError("lazy ingestion failure")

    monkeypatch.setattr(pl, "scan_csv", fail_scan_csv)

    loader = OHLCVFrameLoader(validation="strict")
    with pytest.raises(pl.exceptions.ComputeError, match="lazy ingestion failure"):
        loader.load_csv(str(path))


def test_strict_loader_eager_fallback_requires_explicit_opt_in(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "valid.csv"
    _valid_frame().write_csv(path)

    def fail_scan_csv(*args: object, **kwargs: object) -> object:
        raise pl.exceptions.ComputeError("lazy ingestion failure")

    monkeypatch.setattr(pl, "scan_csv", fail_scan_csv)

    loader = OHLCVFrameLoader(validation="strict", allow_eager_fallback=True)
    out = loader.load_csv(str(path))

    assert out is not None
    assert out.height == 2


def test_normalize_ohlcv_frame_preserves_legacy_non_strict_behavior() -> None:
    frame = _valid_frame().drop("volume")

    assert normalize_ohlcv_frame(frame) is None
    with pytest.raises(OHLCVValidationError):
        normalize_ohlcv_frame(frame, validation="strict")
