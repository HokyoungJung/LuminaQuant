"""Data-free proof that the anti-overfit HARD selection-reject gate would have
rejected the recorded monthly-refit walk-forward overfit cohort.

Context (overfit_selection_gates 2026-07-08): the recorded artifact
``var/reports/latest_alpha_refresh_20260704_full_walkforward/
full_all_strategy_walkforward_selection_latest.json`` stamped 22 candidates as
``research_selected`` on train+validation only. Their mean validation Sharpe was
strongly positive but the locked-OOS collapsed; 11+ of them are the SAME 9-symbol
4h basket wearing different strategy-class hats. This suite recovers each selected
candidate's VALIDATION-block DSR / SPA / PBO (by joining ``folds[].selected[]`` to
``research_selected_candidates`` on ``candidate_id``, averaging across the folds a
candidate was selected in) and proves, WITHOUT any market data, that:

  * with the strict floors (dsr_gate_floor=0.90, spa_gate_ceiling=0.05,
    pbo_gate_ceiling=0.50, gate enforced) ALL 22 are rejected;
  * the identical-basket clone cluster collapses to <= 2 under basket dedup;
  * with every flag OFF the 22 pass through unchanged, in recorded order
    (byte-identical passthrough);
  * the HONESTY GUARD: the single locked-OOS survivor (candidate_id starting
    ``1f1fd241c12f0bc2``, validation Sharpe 5.409) is ALSO rejected -- at its own
    validation DSR 0.234 -- so the gate rejects on uniform a-priori merit, not by
    cherry-picking the collapsers.

It also pins the achievability of the strict DSR floor (a genuinely strong,
non-overfit synthetic candidate clears 0.90), the cross-trial CSCV/PBO reject
consumer (Step 4), and the fail-closed strict-research env guard (Step 6).
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import statistics
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lumina_quant.configuration.loader import build_runtime_config, load_yaml_config
from lumina_quant.strategy_factory.research_metrics import (
    cross_trial_pbo_rejects_run,
    cscv_pbo,
    deflated_sharpe_ratio,
)
from lumina_quant.strategy_factory.selection import (
    apply_selection_reject_and_dedup,
    candidate_symbol_basket_key,
    hurdle_score,
    passes_dsr_spa_hard_gate,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_ARTIFACT = (
    _REPO_ROOT
    / "var/reports/latest_alpha_refresh_20260704_full_walkforward"
    / "full_all_strategy_walkforward_selection_latest.json"
)

# Strict floors == the honest-research profile values. ``enforce_selection_reject_gate``
# is the master alias that flips the DSR/SPA/PBO soft-score into a binding reject.
_STRICT_PARAMS = {
    "enforce_selection_reject_gate": True,
    "dsr_gate_floor": 0.90,
    "spa_gate_ceiling": 0.05,
    "pbo_gate_ceiling": 0.50,
}
_SURVIVOR_PREFIX = "1f1fd241c12f0bc2"


def _load_artifact() -> dict:
    if not _ARTIFACT.exists():  # pragma: no cover - artifact ships with the repo
        pytest.skip(f"recorded artifact not present: {_ARTIFACT}")
    return json.loads(_ARTIFACT.read_text("utf-8"))


def _mean_validation_by_candidate(artifact: dict) -> dict[str, dict[str, float]]:
    """candidate_id -> mean validation-block metrics across its selected folds."""
    blocks: dict[str, list[dict]] = {}
    for fold in artifact.get("folds", []):
        for sel in fold.get("selected", []):
            cid = sel.get("candidate_id")
            vb = sel.get("validation")
            if cid and isinstance(vb, dict):
                blocks.setdefault(cid, []).append(vb)

    def _mean(entries: list[dict], key: str, default: float) -> float:
        vals = [float(e[key]) for e in entries if e.get(key) is not None]
        return statistics.fmean(vals) if vals else default

    out: dict[str, dict[str, float]] = {}
    for cid, entries in blocks.items():
        out[cid] = {
            "deflated_sharpe": _mean(entries, "deflated_sharpe", 0.0),
            "spa_pvalue": _mean(entries, "spa_pvalue", 1.0),
            "pbo": _mean(entries, "pbo", 1.0),
            "sharpe": _mean(entries, "sharpe", 0.0),
            "return": _mean(entries, "return", 0.0),
        }
    return out


def _selected_candidates_with_validation(artifact: dict) -> list[dict]:
    """The 22 recorded selected candidates, each carrying a genuine ``validation``
    metric block (so ``passes_dsr_spa_hard_gate(..., mode='val')`` reads it)."""
    val_by_id = _mean_validation_by_candidate(artifact)
    rows: list[dict] = []
    for c in artifact["research_selected_candidates"]:
        cid = c["candidate_id"]
        rows.append(
            {
                "candidate_id": cid,
                "name": c.get("name"),
                "strategy_class": c.get("strategy_class"),
                "strategy_timeframe": c.get("strategy_timeframe"),
                "family": c.get("family"),
                "symbols": list(c.get("symbols") or []),
                "rank": c.get("rank"),
                "validation": val_by_id.get(cid, {}),
            }
        )
    return rows


# --------------------------------------------------------------------------- #
# Centerpiece: the recorded overfit cohort.
# --------------------------------------------------------------------------- #
def test_recorded_cohort_has_22_selected_candidates():
    artifact = _load_artifact()
    assert len(artifact["research_selected_candidates"]) == 22


def test_strict_gate_rejects_all_22_recorded_candidates():
    rows = _selected_candidates_with_validation(_load_artifact())
    assert len(rows) == 22
    survivors = apply_selection_reject_and_dedup(
        rows,
        mode="val",
        robust_score_params=_STRICT_PARAMS,
        enabled=True,
    )
    assert survivors == [], "strict DSR/SPA/PBO gate must reject the entire cohort"


def test_recorded_cohort_per_axis_reject_tallies():
    """Sanity tallies the task pins: DSR<0.90 -> 22/22, SPA>0.05 -> 17/22,
    PBO>0.50 -> 3/22 (validation-block means, gate enforced)."""
    rows = _selected_candidates_with_validation(_load_artifact())
    dsr_only = {**_STRICT_PARAMS, "spa_gate_ceiling": 1.0, "pbo_gate_ceiling": 1.0}
    spa_only = {**_STRICT_PARAMS, "dsr_gate_floor": 0.0, "pbo_gate_ceiling": 1.0}
    pbo_only = {**_STRICT_PARAMS, "dsr_gate_floor": 0.0, "spa_gate_ceiling": 1.0}
    dsr_rej = sum(
        not passes_dsr_spa_hard_gate(r, mode="val", robust_score_params=dsr_only) for r in rows
    )
    spa_rej = sum(
        not passes_dsr_spa_hard_gate(r, mode="val", robust_score_params=spa_only) for r in rows
    )
    pbo_rej = sum(
        not passes_dsr_spa_hard_gate(r, mode="val", robust_score_params=pbo_only) for r in rows
    )
    assert (dsr_rej, spa_rej, pbo_rej) == (22, 17, 3)


def test_basket_dedup_collapses_identical_basket_clone_cluster():
    rows = _selected_candidates_with_validation(_load_artifact())
    # The dominant symbol basket (the 9-symbol crypto basket) recurs many times.
    keys = [candidate_symbol_basket_key(r) for r in rows]
    clone_key = max(set(keys), key=keys.count)
    clone_count = keys.count(clone_key)
    assert clone_count >= 11, "expected the identical-basket clone cluster in the recorded cohort"

    # Dedup with the reject gate DISABLED (robust_score_params=None), so the
    # collapse is purely the symbol-basket cap, not the reject gate.
    deduped = apply_selection_reject_and_dedup(
        rows,
        mode="val",
        robust_score_params=None,
        max_per_symbol_basket=2,
        enabled=True,
    )
    deduped_clone = sum(candidate_symbol_basket_key(r) == clone_key for r in deduped)
    assert deduped_clone <= 2, "identical-basket clone cluster must collapse to <= 2"


def _augment_with_scoring_blocks(rows: list[dict]) -> list[dict]:
    """Attach ``train`` + ``oos`` blocks (``mode='val'`` scores off the ``oos``
    block) so ``hurdle_score`` is a meaningful, DISTINCT number per clone, mirroring
    each row's validation Sharpe so the best-scoring clone == highest-val-Sharpe."""
    out: list[dict] = []
    for r in rows:
        vb = r.get("validation") or {}
        out.append(
            {
                **r,
                "train": {"total_return": 0.1, "trade_count": 10},
                "oos": {
                    "sharpe": float(vb.get("sharpe", 0.0)),
                    "return": float(vb.get("return", 0.0)),
                    "pbo": 0.1,
                },
            }
        )
    return out


def test_basket_dedup_keeps_best_scoring_clone_representative():
    """Task-1 ordering fix on the RECORDED cohort: when the identical-basket clone
    cluster collapses under the symbol-basket cap, the survivor(s) are the
    BEST-scoring clones (top-2 by ``hurdle_score``), NOT merely the earliest in
    input order."""
    rows = _augment_with_scoring_blocks(_selected_candidates_with_validation(_load_artifact()))
    keys = [candidate_symbol_basket_key(r) for r in rows]
    clone_key = max(set(keys), key=keys.count)
    clone_members = [r for r in rows if candidate_symbol_basket_key(r) == clone_key]

    expected_top2 = sorted(clone_members, key=lambda r: hurdle_score(r, mode="val"), reverse=True)[
        :2
    ]
    expected_ids = {r["candidate_id"] for r in expected_top2}

    # Worst-score-first input: the best clones sit LAST, so an input-order impl
    # (the bug) would keep the worst two. Only the score-descending fix keeps the
    # best two -> this makes the recorded-cohort assertion a genuine regression guard.
    clones_worst_first = sorted(clone_members, key=lambda r: hurdle_score(r, mode="val"))
    reordered = [r for r in rows if candidate_symbol_basket_key(r) != clone_key]
    reordered += clones_worst_first
    assert {r["candidate_id"] for r in clones_worst_first[:2]} != expected_ids

    deduped = apply_selection_reject_and_dedup(
        reordered,
        mode="val",
        robust_score_params=None,  # reject gate off: isolate the dedup ordering
        max_per_symbol_basket=2,
        enabled=True,
    )
    survivor_ids = {
        r["candidate_id"] for r in deduped if candidate_symbol_basket_key(r) == clone_key
    }
    assert len(survivor_ids) <= 2
    assert survivor_ids == expected_ids, "surviving clones must be the best-scoring representatives"


def test_dedup_ordering_fix_keeps_best_even_when_placed_last():
    """Deterministic, self-contained proof the survivor is the best-scoring clone
    even when the best clones are placed LAST in input order (the exact scenario
    the input-order bug got wrong)."""

    def _clone(cid: str, sharpe: float) -> dict:
        return {
            "candidate_id": cid,
            "symbols": ["BTC/USDT", "ETH/USDT", "SOL/USDT"],
            "strategy_class": cid,  # distinct -> only the symbol-basket cap collapses them
            "family": cid,
            "train": {"total_return": 0.1, "trade_count": 10},
            "oos": {"sharpe": sharpe, "return": 0.02, "pbo": 0.1},
        }

    # Worst-first input order: the two best (c3, c4) are last.
    rows = [_clone("c1", 0.5), _clone("c2", 1.0), _clone("c4", 2.5), _clone("c3", 3.0)]
    deduped = apply_selection_reject_and_dedup(
        rows,
        mode="val",
        robust_score_params=None,
        max_per_symbol_basket=2,
        enabled=True,
    )
    # Best-score-first: c3 (3.0) then c4 (2.5); c1/c2 dropped by the cap of 2.
    assert [r["candidate_id"] for r in deduped] == ["c3", "c4"]
    # The naive input-order impl would instead have kept c1, c2 (the first two).
    assert {"c1", "c2"} != {r["candidate_id"] for r in deduped}


def test_reject_only_no_dedup_preserves_input_order_and_identity():
    """Reject-only (no dedup cap active) must NOT reorder: survivors are the same
    objects in the same input order (only reject-drops removed)."""
    strong = {
        "candidate_id": "keep",
        "symbols": ["BTC/USDT"],
        "validation": {"deflated_sharpe": 0.99, "spa_pvalue": 0.01, "pbo": 0.1},
    }
    weak = {
        "candidate_id": "drop",
        "symbols": ["ETH/USDT"],
        "validation": {"deflated_sharpe": 0.05, "spa_pvalue": 0.4, "pbo": 0.6},
    }
    rows = [strong, weak, dict(strong, candidate_id="keep2")]
    out = apply_selection_reject_and_dedup(
        rows,
        mode="val",
        robust_score_params=_STRICT_PARAMS,  # reject on; NO dedup cap passed
        enabled=True,
    )
    assert [r["candidate_id"] for r in out] == ["keep", "keep2"]
    # Same objects (no copy, no reorder): rows[0] and rows[2] pass straight through.
    assert out[0] is rows[0] and out[1] is rows[2]


def test_all_flags_off_is_byte_identical_passthrough():
    rows = _selected_candidates_with_validation(_load_artifact())
    passthrough = apply_selection_reject_and_dedup(
        rows,
        mode="val",
        robust_score_params=_STRICT_PARAMS,  # floors present but gate NOT enabled
        max_per_symbol_basket=2,
        enabled=False,
    )
    assert [r["candidate_id"] for r in passthrough] == [r["candidate_id"] for r in rows]
    assert len(passthrough) == 22
    # Same objects, same order: strict identity no-op.
    assert all(a is b for a, b in zip(passthrough, rows, strict=True))


def test_honesty_guard_oos_survivor_is_also_rejected():
    """The single locked-OOS survivor is rejected on uniform a-priori merit, not
    cherry-picked away: it fails the SAME validation DSR floor as the collapsers."""
    rows = _selected_candidates_with_validation(_load_artifact())
    survivor = next(r for r in rows if str(r["candidate_id"]).startswith(_SURVIVOR_PREFIX))
    # Its validation DSR is ~0.234 -- far below the 0.90 floor -- despite the
    # strong validation Sharpe (~5.409) that won it selection.
    assert survivor["validation"]["deflated_sharpe"] == pytest.approx(0.234, abs=0.01)
    assert survivor["validation"]["sharpe"] == pytest.approx(5.409, abs=0.01)
    assert (
        passes_dsr_spa_hard_gate(survivor, mode="val", robust_score_params=_STRICT_PARAMS) is False
    )


# --------------------------------------------------------------------------- #
# Step 2 achievability: strict floor is NOT a permanent promote-nothing no-op.
# --------------------------------------------------------------------------- #
def test_strict_dsr_floor_is_achievable_for_a_genuinely_strong_candidate():
    """A clean, non-overfit synthetic candidate clears dsr_gate_floor=0.90 under
    this codebase's DSR computed against the WHOLE-SEARCH trial count (the honest
    multiple-testing basis), so strict selection can still promote real edge."""
    # num_trials = whole-search candidate count (the recorded run evaluated ~1400
    # candidates per fold); the gate must deflate against that, not num_trials=1.
    num_trials = 1400
    rng = np.random.default_rng(20260708)
    returns = 0.0018 + 0.010 * rng.standard_normal(756)
    dsr = deflated_sharpe_ratio(returns, num_trials=num_trials)
    assert dsr >= 0.90, f"genuinely strong candidate should clear the 0.90 floor, got {dsr}"

    strong = {
        "symbols": ["BTC/USDT"],
        "strategy_class": "CleanEdgeStrategy",
        "strategy_timeframe": "1d",
        "validation": {"deflated_sharpe": dsr, "spa_pvalue": 0.01, "pbo": 0.10},
    }
    assert passes_dsr_spa_hard_gate(strong, mode="val", robust_score_params=_STRICT_PARAMS) is True


# --------------------------------------------------------------------------- #
# Step 4: cross-trial CSCV/PBO reject consumer (synthetic-matrix unit test).
# --------------------------------------------------------------------------- #
def _overfit_family_matrix() -> np.ndarray:
    """Anti-persistent family: later blocks anti-correlate with earlier ones, so
    the in-sample winner systematically reverts out-of-sample (high CSCV PBO)."""
    base = np.random.default_rng(11).standard_normal((30, 240))
    out = base.copy()
    out[:, 30:] -= base[:, :-30]
    return out


def _clean_family_matrix() -> np.ndarray:
    """One dominant persistent-alpha candidate among noise (near-zero CSCV PBO)."""
    m = np.random.default_rng(7).standard_normal((40, 240)) * 0.5
    m[0] += 0.30
    return m


def test_cross_trial_pbo_estimates_bracket_the_ceiling():
    assert cscv_pbo(_overfit_family_matrix()) > 0.50
    assert cscv_pbo(_clean_family_matrix()) < 0.50


def test_cross_trial_pbo_rejects_overfit_family_when_enabled():
    assert (
        cross_trial_pbo_rejects_run(
            _overfit_family_matrix(), max_cross_trial_pbo=0.50, enabled=True
        )
        is True
    )
    assert (
        cross_trial_pbo_rejects_run(_clean_family_matrix(), max_cross_trial_pbo=0.50, enabled=True)
        is False
    )


def test_cross_trial_pbo_is_noop_when_disabled_or_permissive():
    overfit = _overfit_family_matrix()
    # Disabled -> never rejects (the estimator is not even run).
    assert cross_trial_pbo_rejects_run(overfit, max_cross_trial_pbo=0.50, enabled=False) is False
    # Permissive ceiling 1.0 -> a probability in [0, 1] can never exceed it.
    assert cross_trial_pbo_rejects_run(overfit, max_cross_trial_pbo=1.0, enabled=True) is False


# --------------------------------------------------------------------------- #
# Step 6: fail-closed strict-research env guard.
# --------------------------------------------------------------------------- #
def test_strict_env_guard_raises_on_permissive_default_config():
    raw = load_yaml_config(config_path=str(_REPO_ROOT / "config.yaml"))
    with pytest.raises(ValueError, match="enforce_selection_reject_gate"):
        build_runtime_config(raw, env={"LUMINA_ENABLE_STRICT_RESEARCH": "true"})


def test_strict_env_guard_passes_on_strict_profile():
    raw = load_yaml_config(config_path=str(_REPO_ROOT / "configs/profiles/research.yaml"))
    rt = build_runtime_config(raw, env={"LUMINA_ENABLE_STRICT_RESEARCH": "1"})
    assert rt.research.enforce_selection_reject_gate is True
    assert rt.research.dsr_gate_floor >= 0.90


def test_strict_env_guard_is_noop_when_env_unset():
    raw = load_yaml_config(config_path=str(_REPO_ROOT / "config.yaml"))
    rt = build_runtime_config(raw, env={})  # never raises
    assert rt.research.enforce_selection_reject_gate is False


# --------------------------------------------------------------------------- #
# E1: config-gated per-candidate SPA/PBO emission into the monthly-refit
# walk-forward runner's validation block (``_candidate_overfit_stats`` /
# ``_evaluate_candidate``). Default OFF -> byte-identical validation block; the
# strict research profile / the explicit flag arms the SPA + approx_pbo emission,
# and the emitted approx_pbo routes through the shared reject gate as the ``pbo``
# alias. Data-free: a synthetic candidate on a deterministic daily return series.
# --------------------------------------------------------------------------- #
_ENGINE_MODULE = "scripts.research.run_alpha_zoo_69_asset_monthly_refit_walkforward"


def _emit_args(**overrides) -> argparse.Namespace:
    base = dict(
        config=None,
        enforce_selection_reject_gate=False,
        dsr_gate_floor=None,
        spa_gate_ceiling=None,
        pbo_gate_ceiling=None,
        dedupe_baskets=False,
        max_per_symbol_basket=None,
        max_per_lineage=None,
        max_per_family_basket=None,
        emit_candidate_overfit_stats=False,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


@pytest.fixture
def engine():
    """The runner module with the E1 emission gate RESET to OFF after each test so
    module-level state cannot leak into the other suites that import it."""
    module = importlib.import_module(_ENGINE_MODULE)
    module._configure_selection_gate(_emit_args())  # start OFF
    try:
        yield module
    finally:
        module._configure_selection_gate(_emit_args())  # reset OFF


def _synthetic_returns(periods: int = 200, *, seed: int = 20260709) -> pd.Series:
    """A deterministic daily return series with a small positive drift (so the SPA
    studentized mean is positive and the p-value is a genuine bootstrap number,
    not the non-positive-mean 1.0 short-circuit)."""
    rng = np.random.default_rng(seed)
    index = pd.date_range("2025-01-01", periods=periods, freq="D")
    values = 0.0015 + 0.012 * rng.standard_normal(periods)
    return pd.Series(values, index=index)


def _synthetic_candidate(engine, returns: pd.Series):
    return engine.CandidateResult(
        family="synthetic",
        candidate_label="synthetic:e1",
        source_profile_id="synthetic",
        row={"symbols": ["BTC/USDT"], "candidate_id": "synthetic-e1"},
        returns=returns,
    )


def _fold(engine, index: pd.DatetimeIndex, *, val_end_pos: int):
    """A single monthly fold whose validation window ends at ``index[val_end_pos]``
    (validation obs == val_end_pos - 100 + 1)."""
    return engine.MonthlyFold(
        fold_id="synthetic",
        refit_at=index[100],
        train=(index[0], index[99]),
        validation=(index[100], index[val_end_pos]),
        locked_oos=(index[val_end_pos + 1], index[-1]),
    )


def test_e1_emit_off_validation_block_is_byte_identical(engine):
    """OFF (default): the emitted validation block is EXACTLY ``_period_metrics``
    output -- no ``spa_pvalue`` / ``approx_pbo`` keys, byte-identical."""
    engine._configure_selection_gate(_emit_args())  # explicit OFF
    returns = _synthetic_returns()
    candidate = _synthetic_candidate(engine, returns)
    fold = _fold(engine, returns.index, val_end_pos=179)  # 80 validation obs

    row = engine._evaluate_candidate(candidate, fold)
    baseline = engine._period_metrics(candidate.returns, fold.validation)

    assert row["validation"] == baseline
    assert "spa_pvalue" not in row["validation"]
    assert "approx_pbo" not in row["validation"]


def test_e1_emit_on_populates_finite_spa_and_pbo_in_unit_interval(engine):
    engine._configure_selection_gate(_emit_args(emit_candidate_overfit_stats=True))
    returns = _synthetic_returns()
    candidate = _synthetic_candidate(engine, returns)
    fold = _fold(engine, returns.index, val_end_pos=179)  # 80 validation obs >= 64

    validation = engine._evaluate_candidate(candidate, fold)["validation"]

    assert validation["bar_count"] >= 64
    for key in ("spa_pvalue", "approx_pbo"):
        value = validation[key]
        assert isinstance(value, float)
        assert math.isfinite(value)
        assert 0.0 <= value <= 1.0
    # The base metric keys are preserved (emission only ADDS fields).
    for key in ("start", "end", "bar_count", "total_return", "mdd", "sharpe"):
        assert key in validation


def test_e1_emit_on_below_min_obs_omits_fields(engine):
    """A validation window with < 64 obs never emits a degenerate 1.0: the fields
    are simply absent (never-raise)."""
    engine._configure_selection_gate(_emit_args(emit_candidate_overfit_stats=True))
    returns = _synthetic_returns()
    candidate = _synthetic_candidate(engine, returns)
    short_fold = _fold(engine, returns.index, val_end_pos=140)  # 41 validation obs < 64

    validation = engine._evaluate_candidate(candidate, short_fold)["validation"]

    assert validation["bar_count"] < 64
    assert validation.get("spa_pvalue") is None
    assert validation.get("approx_pbo") is None


def test_e1_emit_on_is_deterministic_run_twice(engine):
    """Seeded block bootstrap -> two INDEPENDENTLY constructed candidates on the
    same seeded series emit bit-identical spa_pvalue / approx_pbo."""
    engine._configure_selection_gate(_emit_args(emit_candidate_overfit_stats=True))

    first_returns = _synthetic_returns()
    first = engine._evaluate_candidate(
        _synthetic_candidate(engine, first_returns),
        _fold(engine, first_returns.index, val_end_pos=179),
    )["validation"]

    second_returns = _synthetic_returns()
    second = engine._evaluate_candidate(
        _synthetic_candidate(engine, second_returns),
        _fold(engine, second_returns.index, val_end_pos=179),
    )["validation"]

    assert first["spa_pvalue"] == second["spa_pvalue"]
    assert first["approx_pbo"] == second["approx_pbo"]


def test_e1_emitted_approx_pbo_routes_through_selection_as_pbo_alias(engine):
    """The emitted validation block carries NO ``pbo`` key -- only ``approx_pbo`` --
    so the reject gate must read it through the documented ``pbo`` alias
    (``selection.passes_dsr_spa_hard_gate`` -> ``apply_selection_reject_and_dedup``)
    without error, and the emitted value is the DECIDING pbo axis."""
    engine._configure_selection_gate(_emit_args(emit_candidate_overfit_stats=True))
    returns = _synthetic_returns()
    candidate = _synthetic_candidate(engine, returns)
    fold = _fold(engine, returns.index, val_end_pos=179)

    row = engine._evaluate_candidate(candidate, fold)
    row["candidate_id"] = "synthetic-e1"
    assert "pbo" not in row["validation"]  # only the approx_pbo alias is present
    emitted_pbo = row["validation"]["approx_pbo"]

    # DSR floor 0.0 (a missing deflated_sharpe -> 0.0 never rejects) and SPA
    # ceiling 1.0 isolate the PBO axis; a permissive PBO ceiling keeps the row.
    lenient = {
        "enforce_selection_reject_gate": True,
        "dsr_gate_floor": 0.0,
        "spa_gate_ceiling": 1.0,
        "pbo_gate_ceiling": 1.0,
    }
    assert passes_dsr_spa_hard_gate(row, mode="val", robust_score_params=lenient) is True
    survivors = apply_selection_reject_and_dedup(
        [row], mode="val", robust_score_params=lenient, enabled=True
    )
    assert survivors == [row]

    # A PBO ceiling just BELOW the emitted approx_pbo rejects the row -> proves the
    # emitted value (not the missing-key 0.0 default) is what the alias reads.
    assert emitted_pbo > 0.0
    strict = {**lenient, "pbo_gate_ceiling": emitted_pbo - 1e-9}
    assert passes_dsr_spa_hard_gate(row, mode="val", robust_score_params=strict) is False


def test_e1_strict_research_profile_arms_emission(engine):
    """End-to-end plumbing: loading the strict research profile via ``--config``
    arms ``emit_candidate_overfit_stats`` (research.yaml), so the emission fires
    without any explicit CLI flag."""
    engine._configure_selection_gate(
        _emit_args(config=str(_REPO_ROOT / "configs/profiles/research.yaml"))
    )
    assert engine._SELECTION_GATE_STATE["emit_candidate_overfit_stats"] is True

    returns = _synthetic_returns()
    validation = engine._evaluate_candidate(
        _synthetic_candidate(engine, returns),
        _fold(engine, returns.index, val_end_pos=179),
    )["validation"]
    assert "spa_pvalue" in validation
    assert "approx_pbo" in validation

    # Reset guard: a plain (no-config) reconfigure disarms emission again.
    engine._configure_selection_gate(_emit_args())
    assert engine._SELECTION_GATE_STATE["emit_candidate_overfit_stats"] is False
