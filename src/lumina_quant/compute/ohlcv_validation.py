"""Fail-closed OHLCV data-integrity validation helpers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
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


def _count_expr(expr: pl.Expr, alias: str) -> pl.Expr:
    return expr.fill_null(False).cast(pl.UInt32).sum().alias(alias)


def _collect_counts(frame: pl.DataFrame, expressions: Sequence[pl.Expr]) -> dict[str, int]:
    if not expressions:
        return {}
    row = frame.select(list(expressions)).row(0, named=True)
    return {key: int(value or 0) for key, value in row.items()}


def _numeric(column: str) -> pl.Expr:
    return pl.col(column).cast(pl.Float64, strict=False)


def _datetime_dtype_is_supported(dtype: Any) -> bool:
    base_type = dtype
    base_getter = getattr(dtype, "base_type", None)
    if callable(base_getter):
        base_type = base_getter()
    return base_type in {pl.Date, pl.Datetime}


def _monotonic_expr(datetime_column: str, symbol_column: str | None) -> pl.Expr:
    current = pl.col(datetime_column)
    previous = (
        current.shift(1).over(symbol_column) if symbol_column is not None else current.shift(1)
    )
    return current < previous


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

    datetime_present = datetime_column in frame.columns
    datetime_supported = False
    if datetime_present:
        dtype = frame.schema.get(datetime_column)
        datetime_supported = _datetime_dtype_is_supported(dtype)
        if not datetime_supported:
            issues.append(
                OHLCVValidationIssue(
                    "datetime_dtype_invalid",
                    column=datetime_column,
                    detail=str(dtype),
                )
            )

    count_exprs: list[pl.Expr] = []
    if datetime_present:
        count_exprs.append(_count_expr(pl.col(datetime_column).is_null(), "datetime_null"))

    for column in _PRICE_COLUMNS:
        expr = _numeric(column)
        count_exprs.append(
            _count_expr(
                expr.is_null() | expr.is_nan() | expr.is_infinite(),
                f"{column}_nonfinite",
            )
        )
        count_exprs.append(_count_expr(expr <= 0.0, f"{column}_nonpositive"))

    volume_expr = _numeric("volume")
    count_exprs.append(
        _count_expr(
            volume_expr.is_null() | volume_expr.is_nan() | volume_expr.is_infinite(),
            "volume_nonfinite",
        )
    )
    count_exprs.append(_count_expr(volume_expr < 0.0, "volume_negative"))

    open_expr = _numeric("open")
    high_expr = _numeric("high")
    low_expr = _numeric("low")
    close_expr = _numeric("close")
    count_exprs.extend(
        (
            _count_expr(high_expr < low_expr, "high_below_low"),
            _count_expr(
                high_expr < pl.max_horizontal(open_expr, close_expr, low_expr),
                "high_below_ohlc_member",
            ),
            _count_expr(
                low_expr > pl.min_horizontal(open_expr, close_expr, high_expr),
                "low_above_ohlc_member",
            ),
        )
    )

    duplicate_keys: list[str] = []
    if require_unique_timestamp and datetime_present:
        duplicate_keys = [datetime_column]
        if symbol_column and symbol_column in frame.columns:
            duplicate_keys.insert(0, symbol_column)
        count_exprs.append(
            _count_expr(pl.struct(duplicate_keys).is_duplicated(), "duplicate_timestamp")
        )

    if require_monotonic and datetime_present and datetime_supported:
        count_exprs.append(
            _count_expr(
                _monotonic_expr(
                    datetime_column,
                    symbol_column if symbol_column and symbol_column in frame.columns else None,
                ),
                "datetime_not_monotonic",
            )
        )

    counts = _collect_counts(frame, count_exprs)

    null_count = counts.get("datetime_null", 0)
    if null_count:
        issues.append(
            OHLCVValidationIssue("datetime_null", column=datetime_column, count=null_count)
        )

    for column in _PRICE_COLUMNS:
        bad_numeric = counts.get(f"{column}_nonfinite", 0)
        if bad_numeric:
            issues.append(OHLCVValidationIssue("price_nonfinite", column=column, count=bad_numeric))
        non_positive = counts.get(f"{column}_nonpositive", 0)
        if non_positive:
            issues.append(
                OHLCVValidationIssue("price_nonpositive", column=column, count=non_positive)
            )

    bad_volume = counts.get("volume_nonfinite", 0)
    if bad_volume:
        issues.append(OHLCVValidationIssue("volume_nonfinite", column="volume", count=bad_volume))
    negative_volume = counts.get("volume_negative", 0)
    if negative_volume:
        issues.append(
            OHLCVValidationIssue("volume_negative", column="volume", count=negative_volume)
        )

    high_low = counts.get("high_below_low", 0)
    if high_low:
        issues.append(OHLCVValidationIssue("high_below_low", count=high_low))
    high_below_price = counts.get("high_below_ohlc_member", 0)
    if high_below_price:
        issues.append(OHLCVValidationIssue("high_below_ohlc_member", count=high_below_price))
    low_above_price = counts.get("low_above_ohlc_member", 0)
    if low_above_price:
        issues.append(OHLCVValidationIssue("low_above_ohlc_member", count=low_above_price))

    duplicate_count = counts.get("duplicate_timestamp", 0)
    if duplicate_count:
        issues.append(
            OHLCVValidationIssue(
                "duplicate_timestamp",
                column=",".join(duplicate_keys),
                count=duplicate_count,
            )
        )

    monotonic_count = counts.get("datetime_not_monotonic", 0)
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
