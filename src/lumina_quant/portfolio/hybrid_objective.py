"""Hybrid online portfolio objective policies.

Generic stream/allocation helpers live in ``optimizer_core``.  This module keeps
hybrid-governor-specific scoring formulas isolated while reusing the shared
locked-OOS policy payload and numeric coercion helpers.
"""

from __future__ import annotations

from typing import Any

from lumina_quant.portfolio.optimizer_core import objective_policy_payload, safe_float

HYBRID_LOCKED_OBJECTIVE_PROFILE = "locked_train_val"
HYBRID_DIAGNOSTIC_OBJECTIVE_PROFILES = ("live_guarded", "train_aware_guarded")
HYBRID_OBJECTIVE_PROFILES = (HYBRID_LOCKED_OBJECTIVE_PROFILE, *HYBRID_DIAGNOSTIC_OBJECTIVE_PROFILES)


def hybrid_online_objective_policy(profile: str) -> dict[str, Any]:
    return objective_policy_payload(
        profile,
        oos_is_objective_input=str(profile) in HYBRID_DIAGNOSTIC_OBJECTIVE_PROFILES,
    )


def hybrid_online_objective_from_payload(payload: dict[str, Any], *, profile: str) -> float:
    refreshed = dict(payload["scenarios"]["refreshed_latest_tail"]["split_metrics"])
    historical = dict(payload["scenarios"]["historical_saved_baseline"]["split_metrics"])
    readiness = dict(payload.get("readiness") or {})
    ref_train = dict(refreshed.get("train") or {})
    ref_val = dict(refreshed.get("val") or {})
    ref_oos = dict(refreshed.get("oos") or {})
    hist_oos = dict(historical.get("oos") or {})
    score = 0.0
    if profile == HYBRID_LOCKED_OBJECTIVE_PROFILE:
        score += 180.0 * safe_float(ref_val.get("total_return", ref_val.get("return")), 0.0)
        score += 12.0 * safe_float(ref_val.get("sharpe"), 0.0)
        score -= 120.0 * safe_float(ref_val.get("max_drawdown", ref_val.get("mdd")), 0.0)
        score += 120.0 * safe_float(ref_train.get("total_return", ref_train.get("return")), 0.0)
        score += 8.0 * safe_float(ref_train.get("sharpe"), 0.0)
        if safe_float(ref_train.get("total_return", ref_train.get("return")), 0.0) < 0.0:
            score -= 50.0
        if safe_float(ref_val.get("total_return", ref_val.get("return")), 0.0) < 0.0:
            score -= 75.0
    elif profile == "train_aware_guarded":
        score += 180.0 * safe_float(ref_oos.get("total_return", ref_oos.get("return")), 0.0)
        score += 8.0 * safe_float(ref_oos.get("sharpe"), 0.0)
        score -= 100.0 * safe_float(ref_oos.get("max_drawdown", ref_oos.get("mdd")), 0.0)
        score += 100.0 * safe_float(ref_val.get("total_return", ref_val.get("return")), 0.0)
        score += 10.0 * safe_float(ref_val.get("sharpe"), 0.0)
        score += 120.0 * safe_float(ref_train.get("total_return", ref_train.get("return")), 0.0)
        score += 10.0 * safe_float(ref_train.get("sharpe"), 0.0)
        score += 15.0 * safe_float(hist_oos.get("total_return", hist_oos.get("return")), 0.0)
        score += 2.0 * safe_float(hist_oos.get("sharpe"), 0.0)
        if safe_float(ref_train.get("total_return", ref_train.get("return")), 0.0) < 0.0:
            score -= 50.0
        if safe_float(ref_val.get("total_return", ref_val.get("return")), 0.0) < 0.0:
            score -= 40.0
    else:
        score += 240.0 * safe_float(ref_oos.get("total_return", ref_oos.get("return")), 0.0)
        score += 10.0 * safe_float(ref_oos.get("sharpe"), 0.0)
        score -= 120.0 * safe_float(ref_oos.get("max_drawdown", ref_oos.get("mdd")), 0.0)
        score += 60.0 * safe_float(ref_val.get("total_return", ref_val.get("return")), 0.0)
        score += 8.0 * safe_float(ref_val.get("sharpe"), 0.0)
        score += 20.0 * safe_float(ref_train.get("total_return", ref_train.get("return")), 0.0)
        score += 3.0 * safe_float(ref_train.get("sharpe"), 0.0)
        score += 20.0 * safe_float(hist_oos.get("total_return", hist_oos.get("return")), 0.0)
        score += 2.0 * safe_float(hist_oos.get("sharpe"), 0.0)
    if not readiness.get("beats_cash_refreshed"):
        score -= 1000.0
    if not readiness.get("pair_cap_respected"):
        score -= 500.0
    return float(score)
