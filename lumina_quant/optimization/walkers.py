"""Walk-forward split helpers."""

from __future__ import annotations

from datetime import datetime


def add_months(dt: datetime, months: int) -> datetime:
    """Add months to a datetime while preserving a valid day."""
    year = dt.year + (dt.month - 1 + months) // 12
    month = (dt.month - 1 + months) % 12 + 1
    max_days = [
        31,
        29 if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0) else 28,
        31,
        30,
        31,
        30,
        31,
        31,
        30,
        31,
        30,
        31,
    ]
    day = min(dt.day, max_days[month - 1])
    return datetime(year, month, day)


def build_walk_forward_splits(
    base_start: datetime,
    folds: int,
    train_months: int = 12,
    val_months: int = 6,
    test_months: int = 6,
    step_months: int = 6,
) -> list[dict]:
    """Build rolling walk-forward splits."""
    splits = []
    cursor = base_start
    for i in range(folds):
        train_start = cursor
        train_end = add_months(train_start, train_months)
        val_start = train_end
        val_end = add_months(val_start, val_months)
        test_start = val_end
        test_end = add_months(test_start, test_months)
        splits.append(
            {
                "fold": i + 1,
                "train_start": train_start,
                "train_end": train_end,
                "val_start": val_start,
                "val_end": val_end,
                "test_start": test_start,
                "test_end": test_end,
            }
        )
        cursor = add_months(cursor, step_months)
    return splits
