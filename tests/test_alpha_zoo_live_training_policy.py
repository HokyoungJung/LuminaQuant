from __future__ import annotations

from datetime import UTC, datetime

import pytest

from lumina_quant.alpha_zoo.live_training_policy import (
    compute_standard_live_training_plan,
    format_utc,
    latest_complete_bar_start,
)


def test_latest_complete_bar_does_not_include_incomplete_boundary_bar() -> None:
    assert format_utc(latest_complete_bar_start("2026-05-28T11:00:00Z")) == ("2026-05-28T10:00:00Z")
    assert format_utc(latest_complete_bar_start("2026-05-28T11:42:00Z")) == ("2026-05-28T10:00:00Z")
    assert format_utc(latest_complete_bar_start("2026-05-28T10:59:58Z")) == ("2026-05-28T10:00:00Z")


def test_standard_live_training_plan_uses_latest_8_weeks_and_empty_oos() -> None:
    plan = compute_standard_live_training_plan(
        data_end_utc=datetime(2026, 5, 28, 11, 42, tzinfo=UTC),
        validation_weeks=8,
        bar_minutes=60,
    )

    assert format_utc(plan.validation.end) == "2026-05-28T10:00:00Z"
    assert format_utc(plan.validation.start) == "2026-04-02T11:00:00Z"
    assert format_utc(plan.train.end) == "2026-04-02T10:00:00Z"
    assert plan.locked_oos.enabled is False

    payload = plan.as_payload()
    assert payload["selection_fit_inputs"] == ["train"]
    assert payload["selection_score_inputs"] == ["train", "validation"]
    assert payload["final_refit_inputs"] == ["train", "validation"]
    assert payload["locked_oos_used_for_parameter_fitting"] is False


def test_standard_live_training_plan_rejects_no_train_window() -> None:
    with pytest.raises(ValueError, match="no train data"):
        compute_standard_live_training_plan(
            train_start_utc="2026-05-01T00:00:00Z",
            data_end_utc="2026-05-28T11:00:00Z",
            validation_weeks=8,
        )
