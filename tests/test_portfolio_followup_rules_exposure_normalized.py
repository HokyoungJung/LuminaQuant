from __future__ import annotations

import math

from lumina_quant.portfolio_followup_rules import (
    evaluate_robustness_gates,
    multiple_comparison_delta_floor,
    promotion_gross_exposure,
)


def _payload(
    *,
    oos_total_return: float,
    oos_sharpe: float,
    oos_max_drawdown: float,
    weights: list[dict[str, object]] | None = None,
    train_total_return: float = 0.03,
    val_total_return: float = 0.04,
    train_sharpe: float = 0.6,
    monthly: list[float] | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "train": {
            "total_return": train_total_return,
            "sharpe": train_sharpe,
            "trade_count": 12.0,
            "max_drawdown": 0.05,
        },
        "val": {"total_return": val_total_return, "sharpe": 1.0, "max_drawdown": 0.05},
        "oos": {
            "total_return": oos_total_return,
            "sharpe": oos_sharpe,
            "sortino": 2.0,
            "calmar": 4.0,
            "max_drawdown": oos_max_drawdown,
            "volatility": 0.12,
        },
        "oos_monthly_returns": [
            {"month": f"2026-0{idx}", "total_return": value, "days": 20}
            for idx, value in enumerate(monthly or [0.03, 0.03, 0.03], start=2)
        ],
    }
    if weights is not None:
        payload["weights"] = weights
    return payload


# ---------------------------------------------------------------------------
# promotion_gross_exposure
# ---------------------------------------------------------------------------


def test_promotion_gross_exposure_sums_absolute_weights() -> None:
    payload = {"weights": [{"weight": 0.8}, {"weight": -0.6}, {"weight": 0.4}]}
    assert math.isclose(promotion_gross_exposure(payload), 1.8, rel_tol=1e-12)


def test_promotion_gross_exposure_defaults_to_one_without_weight_evidence() -> None:
    assert promotion_gross_exposure({}) == 1.0
    # Zero-weight rows are treated as no evidence -> fully invested default.
    assert promotion_gross_exposure({"weights": [{"weight": 0.0}]}) == 1.0


def test_promotion_gross_exposure_reads_scalar_fallbacks() -> None:
    assert promotion_gross_exposure({"gross_exposure": 2.5}) == 2.5
    assert promotion_gross_exposure({"active_exposure": 1.3}) == 1.3
    assert promotion_gross_exposure({"metadata": {"gross_leverage": 3.0}}) == 3.0


# ---------------------------------------------------------------------------
# multiple_comparison_delta_floor
# ---------------------------------------------------------------------------


def test_multiple_comparison_floor_is_base_for_single_challenger() -> None:
    assert multiple_comparison_delta_floor([0.9], challenger_count=1) == 0.0
    assert multiple_comparison_delta_floor([0.9], challenger_count=1, base_floor=0.05) == 0.05


def test_multiple_comparison_floor_scales_with_spread_and_k() -> None:
    deltas = [0.10, 0.001]
    mean = sum(deltas) / len(deltas)
    spread = math.sqrt(sum((d - mean) ** 2 for d in deltas) / len(deltas))
    expected = spread * math.sqrt(2.0 * math.log(2.0))
    floor = multiple_comparison_delta_floor(deltas, challenger_count=2)
    assert floor > 0.0
    assert math.isclose(floor, expected, rel_tol=1e-12)


def test_multiple_comparison_floor_grows_with_more_comparisons() -> None:
    deltas = [0.10, 0.02, 0.05, -0.30]
    floor_two = multiple_comparison_delta_floor(deltas, challenger_count=2)
    floor_four = multiple_comparison_delta_floor(deltas, challenger_count=4)
    assert floor_four > floor_two > 0.0


# ---------------------------------------------------------------------------
# evaluate_robustness_gates flag-OFF byte-identity
# ---------------------------------------------------------------------------


def test_gate_flag_off_is_byte_identical() -> None:
    incumbent = _payload(oos_total_return=0.05, oos_sharpe=1.5, oos_max_drawdown=0.06)
    candidate = _payload(oos_total_return=0.09, oos_sharpe=2.1, oos_max_drawdown=0.05)

    legacy = evaluate_robustness_gates(candidate, incumbent)
    explicit_off = evaluate_robustness_gates(
        candidate, incumbent, exposure_normalized_promotion=False
    )

    assert legacy == explicit_off
    # No exposure keys leak into the legacy output.
    assert "exposure_normalized_promotion" not in legacy
    assert "candidate_gross_exposure" not in legacy


# ---------------------------------------------------------------------------
# evaluate_robustness_gates flag-ON corrected behavior
# ---------------------------------------------------------------------------


def test_gate_flag_on_rejects_leverage_bought_superiority() -> None:
    # Incumbent: unlevered (gross 1.0), 5% OOS.
    incumbent = _payload(
        oos_total_return=0.05,
        oos_sharpe=1.5,
        oos_max_drawdown=0.06,
        weights=[{"weight": 0.5}, {"weight": 0.5}],
    )
    # Challenger: 2x gross, 9% raw OOS (beats 5% raw) but only 4.5% per unit
    # of exposure -> should NOT beat the incumbent once normalized. Sharpe
    # relief (>= incumbent + 0.5) keeps every other gate satisfied.
    levered = _payload(
        oos_total_return=0.09,
        oos_sharpe=2.1,
        oos_max_drawdown=0.12,
        weights=[{"weight": 1.0}, {"weight": 1.0}],
    )

    off = evaluate_robustness_gates(levered, incumbent)
    on = evaluate_robustness_gates(levered, incumbent, exposure_normalized_promotion=True)

    # Flag OFF: leverage buys the superiority gate.
    assert off["promotable"] is True
    assert off["oos_total_return_delta"] > 0.0

    # Flag ON: normalized delta is negative -> superiority gate fails.
    assert on["promotable"] is False
    assert "oos_total_return_not_above_incumbent" in on["rejection_reasons"]
    assert math.isclose(on["oos_total_return_delta"], 0.09 / 2.0 - 0.05, rel_tol=1e-12)
    assert on["candidate_gross_exposure"] == 2.0
    assert on["incumbent_gross_exposure"] == 1.0
    assert on["exposure_normalized_promotion"] is True


def test_gate_flag_on_keeps_genuine_unlevered_edge() -> None:
    incumbent = _payload(
        oos_total_return=0.04,
        oos_sharpe=1.5,
        oos_max_drawdown=0.06,
        weights=[{"weight": 1.0}],
    )
    genuine = _payload(
        oos_total_return=0.09,
        oos_sharpe=2.1,
        oos_max_drawdown=0.05,
        weights=[{"weight": 1.0}],
    )

    on = evaluate_robustness_gates(genuine, incumbent, exposure_normalized_promotion=True)

    assert on["promotable"] is True
    assert on["rejection_reasons"] == []
    assert math.isclose(on["oos_total_return_delta"], 0.05, rel_tol=1e-12)
