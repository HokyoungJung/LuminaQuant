"""Fail-closed OHLCV data-integrity validation helpers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from itertools import pairwise
from typing import Any

import polars as pl

REQUIRED_OHLCV_COLUMNS: tuple[str, ...] = (
    "datetime",
    "open",
    "high",
    "low",
    "close",
    "volume",
)
_PRICE_COLUMNS = ("open", "high", "low", "close")


@dataclass(frozen=True, slots=True)
class OHLCVValidationIssue:
    """One fail-closed data-integrity issue."""

    code: str
    column: str | None = None
    count: int = 0
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class OHLCVValidationReport:
    """JSON-safe validation report for an OHLCV frame."""

    passed: bool
    rows: int
    issues: tuple[OHLCVValidationIssue, ...]
    metrics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_kind": "ohlcv_validation_report",
            "passed": self.passed,
            "rows": self.rows,
            "issues": [issue.to_dict() for issue in self.issues],
            "metrics": dict(self.metrics),
        }


class OHLCVValidationError(ValueError):
    """Raised when strict OHLCV validation fails."""

    def __init__(self, report: OHLCVValidationReport, *, context: str = "ohlcv") -> None:
        self.report = report
        self.context = str(context or "ohlcv")
        codes = ",".join(issue.code for issue in report.issues) or "unknown"
        super().__init__(f"{self.context}_validation_failed:{codes}")


def _count(frame: pl.DataFrame, expr: pl.Expr) -> int:
    value = frame.select(expr.cast(pl.Int64).sum().alias("count"))["count"][0]
    return int(value or 0)


def _numeric(column: str) -> pl.Expr:
    return pl.col(column).cast(pl.Float64, strict=False)


def _as_python_datetimes(values: Sequence[Any]) -> list[Any]:
    out: list[Any] = []
    for value in values:
        if isinstance(value, datetime):
            out.append(value)
        else:
            out.append(value)
    return out


def _datetime_dtype_is_supported(dtype: Any) -> bool:
    base_type = dtype
    base_getter = getattr(dtype, "base_type", None)
    if callable(base_getter):
        base_type = base_getter()
    return base_type in {pl.Date, pl.Datetime}


def _monotonic_issue_count(
    frame: pl.DataFrame,
    *,
    datetime_column: str,
    symbol_column: str | None,
) -> int:
    if datetime_column not in frame.columns or frame.height < 2:
        return 0
    if symbol_column and symbol_column in frame.columns:
        count = 0
        for part in frame.partition_by(symbol_column, maintain_order=True):
            values = _as_python_datetimes(part[datetime_column].to_list())
            count += sum(1 for left, right in pairwise(values) if right < left)
        return count
    values = _as_python_datetimes(frame[datetime_column].to_list())
    return sum(1 for left, right in pairwise(values) if right < left)


def validate_ohlcv_frame(
    frame: pl.DataFrame | None,
    *,
    required_columns: Sequence[str] = REQUIRED_OHLCV_COLUMNS,
    datetime_column: str = "datetime",
    symbol_column: str | None = None,
    require_monotonic: bool = True,
    require_unique_timestamp: bool = True,
) -> OHLCVValidationReport:
    """Validate OHLCV invariants without mutating the input frame."""
    issues: list[OHLCVValidationIssue] = []
    if frame is None:
        return OHLCVValidationReport(
            passed=False,
            rows=0,
            issues=(OHLCVValidationIssue("frame_missing"),),
            metrics={"required_columns": list(required_columns)},
        )

    rows = int(frame.height)
    missing = [column for column in required_columns if column not in frame.columns]
    for column in missing:
        issues.append(OHLCVValidationIssue("required_column_missing", column=column))
    if rows <= 0:
        issues.append(OHLCVValidationIssue("frame_empty"))
    if missing or rows <= 0:
        return OHLCVValidationReport(
            passed=False,
            rows=rows,
            issues=tuple(issues),
            metrics={"required_columns": list(required_columns)},
        )

    if datetime_column in frame.columns:
        dtype = frame.schema.get(datetime_column)
        if not _datetime_dtype_is_supported(dtype):
            issues.append(
                OHLCVValidationIssue(
                    "datetime_dtype_invalid",
                    column=datetime_column,
                    detail=str(dtype),
                )
            )
        null_count = _count(frame, pl.col(datetime_column).is_null())
        if null_count:
            issues.append(
                OHLCVValidationIssue("datetime_null", column=datetime_column, count=null_count)
            )

    for column in _PRICE_COLUMNS:
        expr = _numeric(column)
        bad_numeric = _count(frame, expr.is_null() | expr.is_nan() | expr.is_infinite())
        if bad_numeric:
            issues.append(OHLCVValidationIssue("price_nonfinite", column=column, count=bad_numeric))
        non_positive = _count(frame, expr <= 0.0)
        if non_positive:
            issues.append(
                OHLCVValidationIssue("price_nonpositive", column=column, count=non_positive)
            )

    volume_expr = _numeric("volume")
    bad_volume = _count(
        frame, volume_expr.is_null() | volume_expr.is_nan() | volume_expr.is_infinite()
    )
    if bad_volume:
        issues.append(OHLCVValidationIssue("volume_nonfinite", column="volume", count=bad_volume))
    negative_volume = _count(frame, volume_expr < 0.0)
    if negative_volume:
        issues.append(
            OHLCVValidationIssue("volume_negative", column="volume", count=negative_volume)
        )

    open_expr = _numeric("open")
    high_expr = _numeric("high")
    low_expr = _numeric("low")
    close_expr = _numeric("close")
    high_low = _count(frame, high_expr < low_expr)
    if high_low:
        issues.append(OHLCVValidationIssue("high_below_low", count=high_low))
    high_below_price = _count(frame, high_expr < pl.max_horizontal(open_expr, close_expr, low_expr))
    if high_below_price:
        issues.append(OHLCVValidationIssue("high_below_ohlc_member", count=high_below_price))
    low_above_price = _count(frame, low_expr > pl.min_horizontal(open_expr, close_expr, high_expr))
    if low_above_price:
        issues.append(OHLCVValidationIssue("low_above_ohlc_member", count=low_above_price))

    if require_unique_timestamp and datetime_column in frame.columns:
        keys = [datetime_column]
        if symbol_column and symbol_column in frame.columns:
            keys.insert(0, symbol_column)
        duplicate_count = int(frame.select(keys).is_duplicated().sum())
        if duplicate_count:
            issues.append(
                OHLCVValidationIssue(
                    "duplicate_timestamp",
                    column=",".join(keys),
                    count=duplicate_count,
                )
            )

    if require_monotonic:
        monotonic_count = _monotonic_issue_count(
            frame,
            datetime_column=datetime_column,
            symbol_column=symbol_column,
        )
        if monotonic_count:
            issues.append(
                OHLCVValidationIssue(
                    "datetime_not_monotonic",
                    column=datetime_column,
                    count=monotonic_count,
                )
            )

    metrics = {
        "required_columns": list(required_columns),
        "price_columns": list(_PRICE_COLUMNS),
        "datetime_column": datetime_column,
        "symbol_column": symbol_column,
        "require_monotonic": bool(require_monotonic),
        "require_unique_timestamp": bool(require_unique_timestamp),
    }
    return OHLCVValidationReport(
        passed=not issues, rows=rows, issues=tuple(issues), metrics=metrics
    )


def assert_valid_ohlcv_frame(
    frame: pl.DataFrame | None,
    *,
    context: str = "ohlcv",
    **kwargs: Any,
) -> OHLCVValidationReport:
    """Validate and raise ``OHLCVValidationError`` when fail-closed checks fail."""
    report = validate_ohlcv_frame(frame, **kwargs)
    if not report.passed:
        raise OHLCVValidationError(report, context=context)
    return report


__all__ = [
    "REQUIRED_OHLCV_COLUMNS",
    "OHLCVValidationError",
    "OHLCVValidationIssue",
    "OHLCVValidationReport",
    "assert_valid_ohlcv_frame",
    "validate_ohlcv_frame",
]
