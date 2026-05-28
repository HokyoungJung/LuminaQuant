"""Standard live-training split policy for Alpha Zoo research refits.

The live runtime consumes frozen artifacts and does not train.  This module
codifies the research/refit policy used to create those frozen artifacts:
hold out the most recent validation window, tune on pre-validation training
only, then perform a final refit on train+validation after selection.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import UTC, datetime, timedelta
from typing import Any


STANDARD_VALIDATION_WEEKS = 8
STANDARD_WARMUP_RATIO = 0.60
STANDARD_BAR_MINUTES = 60
STANDARD_TRAIN_START_UTC = "2025-01-01T00:00:00Z"


@dataclass(frozen=True, slots=True)
class SplitWindow:
    """Inclusive UTC split bounds."""

    start: datetime
    end: datetime
    role: str

    @property
    def enabled(self) -> bool:
        return self.start <= self.end

    def as_payload(self) -> dict[str, Any]:
        return {
            "start": format_utc(self.start),
            "end": format_utc(self.end),
            "role": self.role,
            "enabled": self.enabled,
        }


@dataclass(frozen=True, slots=True)
class StandardLiveTrainingPlan:
    """Train/validation/final-refit plan for live artifact generation."""

    train: SplitWindow
    validation: SplitWindow
    locked_oos: SplitWindow
    validation_weeks: int = STANDARD_VALIDATION_WEEKS
    warmup_ratio: float = STANDARD_WARMUP_RATIO
    bar_minutes: int = STANDARD_BAR_MINUTES
    final_refit_after_selection: bool = True
    test_set_policy: str = "disabled_for_live_final_refit"

    def split_windows(self) -> dict[str, tuple[datetime, datetime]]:
        return {
            "train": (self.train.start, self.train.end),
            "validation": (self.validation.start, self.validation.end),
            "locked_oos": (self.locked_oos.start, self.locked_oos.end),
        }

    def as_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["train"] = self.train.as_payload()
        payload["validation"] = self.validation.as_payload()
        payload["locked_oos"] = self.locked_oos.as_payload()
        payload["selection_fit_inputs"] = ["train"]
        payload["selection_score_inputs"] = ["train", "validation"]
        payload["final_refit_inputs"] = ["train", "validation"]
        payload["locked_oos_used_for_selection"] = False
        payload["locked_oos_used_for_parameter_fitting"] = False
        payload["locked_oos_used_for_objective"] = False
        return payload


def parse_utc(value: str | datetime) -> datetime:
    """Parse an aware UTC datetime from ISO-8601-like input."""
    if isinstance(value, datetime):
        dt = value
    else:
        text = str(value).strip()
        if not text:
            raise ValueError("empty UTC datetime")
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def format_utc(value: datetime) -> str:
    """Render an aware datetime as canonical Z ISO text."""
    return parse_utc(value).isoformat().replace("+00:00", "Z")


def floor_to_bar(value: str | datetime, *, bar_minutes: int = STANDARD_BAR_MINUTES) -> datetime:
    """Floor a timestamp to the latest complete bar boundary."""
    dt = parse_utc(value)
    minutes = max(1, int(bar_minutes))
    total_minutes = dt.hour * 60 + dt.minute
    floored_total = (total_minutes // minutes) * minutes
    floored_hour, floored_minute = divmod(floored_total, 60)
    return dt.replace(
        hour=floored_hour,
        minute=floored_minute,
        second=0,
        microsecond=0,
    )


def latest_complete_bar_start(
    value: str | datetime,
    *,
    bar_minutes: int = STANDARD_BAR_MINUTES,
    completion_tolerance: timedelta = timedelta(seconds=2),
) -> datetime:
    """Return the start timestamp for the latest complete bar.

    The input is a data-coverage timestamp, not a bar label.  If coverage has
    entered a new bar but has not reached that bar's end, the latest complete
    bar is the previous one.  A small tolerance covers raw 1s feeds that often
    end at ``HH:59:58`` rather than exactly ``HH:59:59``.
    """
    dt = parse_utc(value)
    minutes = max(1, int(bar_minutes))
    bar_delta = timedelta(minutes=minutes)
    bar_start = floor_to_bar(dt, bar_minutes=minutes)
    if dt + completion_tolerance >= bar_start + bar_delta:
        return bar_start
    return bar_start - bar_delta


def compute_standard_live_training_plan(
    *,
    data_end_utc: str | datetime,
    train_start_utc: str | datetime = STANDARD_TRAIN_START_UTC,
    validation_weeks: int = STANDARD_VALIDATION_WEEKS,
    warmup_ratio: float = STANDARD_WARMUP_RATIO,
    bar_minutes: int = STANDARD_BAR_MINUTES,
) -> StandardLiveTrainingPlan:
    """Return the standard live-refit train/validation split plan.

    The validation split is the most recent ``validation_weeks`` complete bars
    at ``bar_minutes`` granularity.  No locked test set is reserved for live
    final refits; after selecting on the holdout validation evidence, final
    live parameters are refit on train+validation.
    """
    bar_delta = timedelta(minutes=max(1, int(bar_minutes)))
    validation_end = latest_complete_bar_start(data_end_utc, bar_minutes=bar_minutes)
    validation_start = validation_end - timedelta(weeks=max(1, int(validation_weeks))) + bar_delta
    train_start = floor_to_bar(train_start_utc, bar_minutes=bar_minutes)
    train_end = validation_start - bar_delta
    if train_end < train_start:
        raise ValueError(
            "standard live training split has no train data: "
            f"train_start={format_utc(train_start)} validation_start={format_utc(validation_start)}"
        )
    # Empty by construction; existing artifact schemas keep the key but the
    # standard live-refit policy does not reserve or score a test/OOS set.
    empty_oos_start = validation_end + bar_delta
    empty_oos_end = validation_end
    return StandardLiveTrainingPlan(
        train=SplitWindow(
            start=train_start,
            end=train_end,
            role="parameter_fitting_and_objective_training",
        ),
        validation=SplitWindow(
            start=validation_start,
            end=validation_end,
            role="holdout_selection_and_report",
        ),
        locked_oos=SplitWindow(
            start=empty_oos_start,
            end=empty_oos_end,
            role="disabled_for_live_final_refit_no_test_set_reserved",
        ),
        validation_weeks=max(1, int(validation_weeks)),
        warmup_ratio=float(warmup_ratio),
        bar_minutes=max(1, int(bar_minutes)),
    )
