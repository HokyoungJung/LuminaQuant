"""SELECTION-lane validation-wiring fixes (2026-07-06 viability review).

Covers three opt-in, default-OFF flags and proves that with each flag unset the
code path is byte-identical to today, while the corrected behavior only appears
when the flag is explicitly enabled:

* ``research.strict_selection_gate`` -- promote the deflated-Sharpe / SPA reality
  check from an advisory soft-score to a binding hard reject.
* ``research.use_lockbox_split`` -- carve a never-touched lockbox from the OOS
  tail, rank/gate on validation, and report the lockbox as the sole OOS.
* ``research.purge_embargo_bars`` -- drop N bars between contiguous splits.
"""

from __future__ import annotations

import numpy as np

from lumina_quant.optimization.walkers import build_walk_forward_splits
from lumina_quant.strategy_factory import research_runner as rr
from lumina_quant.strategy_factory import research_run_support as rrs
from lumina_quant.strategy_factory import selection as sel


def _passing_metrics(**overrides):
    metrics = {
        "sharpe": 1.0,
        "return": 0.05,
        "pbo": 0.10,
        "turnover": 1.0,
        "mdd": 0.10,
        "trade_count": 20.0,
        "deflated_sharpe": 0.6,
        "spa_pvalue": 0.10,
    }
    metrics.update(overrides)
    return metrics


# ── research_run_support._resolve_score_config pass-through ────────────────────


def test_resolve_score_config_no_research_key_is_byte_identical():
    baseline = rrs._resolve_score_config(None)
    with_override = rrs._resolve_score_config({"reject_thresholds": {"oos_sharpe_min": 0.5}})
    assert "research" not in baseline
    assert "research" not in with_override


def test_resolve_score_config_passes_research_section_through():
    resolved = rrs._resolve_score_config(
        {"research": {"strict_selection_gate": True, "dsr_gate_floor": 0.2}}
    )
    assert resolved["research"] == {"strict_selection_gate": True, "dsr_gate_floor": 0.2}


def test_resolve_score_config_empty_research_section_not_added():
    resolved = rrs._resolve_score_config({"research": {}})
    assert "research" not in resolved


# ── flag readers ───────────────────────────────────────────────────────────────


def test_research_flag_reads_from_mapping_section_and_defaults():
    cfg = {"research": {"strict_selection_gate": True}}
    assert rr._research_flag(cfg, "strict_selection_gate", False) is True
    assert rr._research_flag(cfg, "dsr_gate_floor", 0.0) == 0.0
    assert rr._research_flag(None, "strict_selection_gate", False) is False
    assert rr._research_flag({}, "strict_selection_gate", False) is False


def test_research_flag_reads_from_object_config():
    from types import SimpleNamespace

    obj = SimpleNamespace(research=SimpleNamespace(strict_selection_gate=True))
    assert rr._research_flag(obj, "strict_selection_gate", False) is True
    flat = SimpleNamespace(strict_selection_gate=True)
    assert rr._research_flag(flat, "strict_selection_gate", False) is True


def test_split_flag_reads_from_mapping():
    assert rr._split_flag({"use_lockbox_split": True}, "use_lockbox_split", False) is True
    assert rr._split_flag({}, "use_lockbox_split", False) is False
    assert rr._split_flag(None, "purge_embargo_bars", 0) == 0


# ── (a) strict_selection_gate: hurdle hard reject ──────────────────────────────


def test_hurdle_fields_flag_off_is_byte_identical():
    train = _passing_metrics()
    val = _passing_metrics()
    # weak DSR + weak SPA but otherwise passing the base gate
    oos = _passing_metrics(deflated_sharpe=-0.5, spa_pvalue=0.99)

    fields_none, passed_none, reject_none = rr._hurdle_fields(train, val, oos, scoring_config=None)
    fields_benign, passed_benign, reject_benign = rr._hurdle_fields(
        train, val, oos, scoring_config={"reject_thresholds": {}}
    )
    # No strict-gate section -> DSR/SPA stay advisory -> candidate still passes.
    assert passed_none is True
    assert reject_none == {}
    assert passed_benign is True
    assert reject_benign == {}
    assert fields_none == fields_benign


def test_strict_gate_rejects_weak_dsr_when_on():
    train = _passing_metrics()
    val = _passing_metrics()
    oos = _passing_metrics(deflated_sharpe=-0.5)
    scoring = {"research": {"strict_selection_gate": True}}
    _, passed, reject = rr._hurdle_fields(train, val, oos, scoring_config=scoring)
    assert passed is False
    assert reject["deflated_sharpe"] == -0.5


def test_strict_gate_rejects_weak_spa_when_on():
    train = _passing_metrics()
    val = _passing_metrics()
    oos = _passing_metrics(spa_pvalue=0.9)
    scoring = {"research": {"strict_selection_gate": True, "spa_gate_ceiling": 0.5}}
    _, passed, reject = rr._hurdle_fields(train, val, oos, scoring_config=scoring)
    assert passed is False
    assert reject["spa_pvalue"] == 0.9


def test_strict_gate_admits_strong_dsr_when_on():
    train = _passing_metrics()
    val = _passing_metrics()
    oos = _passing_metrics(deflated_sharpe=0.8, spa_pvalue=0.01)
    scoring = {"research": {"strict_selection_gate": True}}
    _, passed, reject = rr._hurdle_fields(train, val, oos, scoring_config=scoring)
    assert passed is True
    assert "deflated_sharpe" not in reject
    assert "spa_pvalue" not in reject


# ── (a) strict_selection_gate: selection.py alias ──────────────────────────────


def test_selection_strict_alias_enables_hard_gate():
    resolved = sel._resolve_robust_score_params({"strict_selection_gate": True})
    assert resolved["dsr_spa_hard_gate"] == 1.0
    # absent -> disabled (byte-identical)
    assert sel._resolve_robust_score_params({})["dsr_spa_hard_gate"] == 0.0
    assert sel._resolve_robust_score_params(None)["dsr_spa_hard_gate"] == 0.0


def test_passes_dsr_spa_hard_gate_off_by_default_but_binds_under_strict_alias():
    weak = {"oos": {"deflated_sharpe": -0.5, "spa_pvalue": 0.9}}
    # default: advisory -> always passes
    assert sel.passes_dsr_spa_hard_gate(weak) is True
    # strict alias: binds and rejects the weak candidate
    assert (
        sel.passes_dsr_spa_hard_gate(weak, robust_score_params={"strict_selection_gate": True})
        is False
    )
    strong = {"oos": {"deflated_sharpe": 0.5, "spa_pvalue": 0.01}}
    assert (
        sel.passes_dsr_spa_hard_gate(strong, robust_score_params={"strict_selection_gate": True})
        is True
    )


def test_select_diversified_shortlist_strict_gate_excludes_weak_dsr():
    candidate = {
        "name": "topcap_btc",
        "strategy_class": "CompositeTrendStrategy",
        "strategy_timeframe": "1h",
        "symbols": ["BTC/USDT"],
        "oos": {
            "return": 0.05,
            "sharpe": 1.0,
            "mdd": 0.1,
            "trades": 20,
            "deflated_sharpe": -0.5,
            "spa_pvalue": 0.9,
        },
        "hurdle_fields": {"oos": {"pass": True, "score": 10.0, "excess_return": 0.05}},
    }
    off = sel.select_diversified_shortlist([candidate], mode="oos")
    assert len(off) == 1
    on = sel.select_diversified_shortlist(
        [candidate],
        mode="oos",
        robust_score_params={"strict_selection_gate": True},
    )
    assert on == []


# ── (b) use_lockbox_split: mask carving ────────────────────────────────────────


def _mask(size, sl):
    arr = np.zeros(size, dtype=bool)
    arr[sl] = True
    return arr


def test_apply_lockbox_off_is_noop():
    size = 100
    out = {
        "train": _mask(size, slice(0, 60)),
        "val": _mask(size, slice(60, 80)),
        "oos": _mask(size, slice(80, 100)),
    }
    result = rr._apply_lockbox_and_purge_masks(out, size=size, split={})
    assert set(result.keys()) == {"train", "val", "oos"}
    assert "lockbox" not in result


def test_apply_lockbox_carves_tail_of_oos():
    size = 100
    out = {
        "train": _mask(size, slice(0, 60)),
        "val": _mask(size, slice(60, 80)),
        "oos": _mask(size, slice(80, 100)),
    }
    result = rr._apply_lockbox_and_purge_masks(out, size=size, split={"use_lockbox_split": True})
    assert "lockbox" in result
    oos_idx = np.flatnonzero(result["oos"])
    lock_idx = np.flatnonzero(result["lockbox"])
    # front half stays OOS, tail half is the lockbox; together they repartition
    # the original OOS window without overlap.
    assert oos_idx.tolist() == list(range(80, 90))
    assert lock_idx.tolist() == list(range(90, 100))
    assert not (result["oos"] & result["lockbox"]).any()
    assert not (result["train"] & result["lockbox"]).any()
    assert not (result["val"] & result["lockbox"]).any()


def test_split_masks_from_datetimes_lockbox_via_split_mapping():
    days = np.arange("2024-01-01", "2024-04-10", dtype="datetime64[D]").astype("datetime64[ms]")
    split = {
        "train_start": "2024-01-01",
        "train_end": "2024-02-10",
        "val_start": "2024-02-11",
        "val_end": "2024-03-01",
        "oos_start": "2024-03-02",
        "oos_end": "2024-04-09",
        "strategy_timeframe": "1d",
        "use_lockbox_split": True,
    }
    masks = rr._split_masks_from_datetimes(days, split=split)
    assert "lockbox" in masks
    assert masks["lockbox"].any()
    # lockbox is disjoint from every other split
    for stage in ("train", "val", "oos"):
        assert not (masks[stage] & masks["lockbox"]).any()

    # OFF -> byte-identical (no lockbox key)
    split_off = dict(split)
    del split_off["use_lockbox_split"]
    masks_off = rr._split_masks_from_datetimes(days, split=split_off)
    assert "lockbox" not in masks_off


# ── (b) use_lockbox_split: ranking + hurdle bind on validation ─────────────────


def test_candidate_rank_score_uses_val_under_lockbox():
    row = {
        "val": {"sharpe": 2.0, "return": 0.10, "deflated_sharpe": 0.5, "pbo": 0.1},
        "oos": {"sharpe": -1.0, "return": -0.10, "deflated_sharpe": -0.5, "pbo": 0.9},
        "train": {"total_return": 0.2, "trade_count": 50},
    }
    # No lockbox flag -> ranks on oos (legacy).
    legacy = rr._candidate_rank_score(row)
    row_lock = dict(row)
    row_lock["effective_split"] = {"use_lockbox_split": True}
    locked = rr._candidate_rank_score(row_lock)
    # val is much stronger than oos, so the lockbox-ranked score must be higher.
    assert locked > legacy


def test_hurdle_fields_bind_stage_val_binds_on_validation():
    train = _passing_metrics()
    # validation FAILS the sharpe floor, oos-slot (lockbox) passes
    val = _passing_metrics(sharpe=-0.5)
    reported_oos = _passing_metrics()
    _, passed, reject = rr._hurdle_fields(
        train, val, reported_oos, scoring_config=None, bind_stage="val"
    )
    assert passed is False
    assert reject["oos_sharpe"] == -0.5
    # default bind_stage binds on the (passing) oos slot instead
    _, passed_oos, reject_oos = rr._hurdle_fields(train, val, reported_oos, scoring_config=None)
    assert passed_oos is True
    assert reject_oos == {}


# ── (c) purge_embargo_bars ─────────────────────────────────────────────────────


def test_apply_purge_drops_leading_bars_of_downstream_splits():
    size = 100
    out = {
        "train": _mask(size, slice(0, 60)),
        "val": _mask(size, slice(60, 80)),
        "oos": _mask(size, slice(80, 100)),
    }
    result = rr._apply_lockbox_and_purge_masks(out, size=size, split={"purge_embargo_bars": 3})
    # train keeps all its bars; val and oos lose their leading 3 bars.
    assert np.flatnonzero(result["train"]).tolist() == list(range(0, 60))
    assert np.flatnonzero(result["val"]).tolist() == list(range(63, 80))
    assert np.flatnonzero(result["oos"]).tolist() == list(range(83, 100))


def test_apply_purge_and_lockbox_together():
    size = 100
    out = {
        "train": _mask(size, slice(0, 60)),
        "val": _mask(size, slice(60, 80)),
        "oos": _mask(size, slice(80, 100)),
    }
    result = rr._apply_lockbox_and_purge_masks(
        out, size=size, split={"use_lockbox_split": True, "purge_embargo_bars": 2}
    )
    assert np.flatnonzero(result["val"]).tolist() == list(range(62, 80))
    # oos carved to 80..89 then leading 2 purged -> 82..89
    assert np.flatnonzero(result["oos"]).tolist() == list(range(82, 90))
    # lockbox carved to 90..99 then leading 2 purged -> 92..99
    assert np.flatnonzero(result["lockbox"]).tolist() == list(range(92, 100))


def test_walk_forward_purge_off_is_contiguous():
    from datetime import datetime

    splits = build_walk_forward_splits(base_start=datetime(2024, 1, 1), folds=3)
    for s in splits:
        assert s["val_start"] == s["train_end"]
        assert s["test_start"] == s["val_end"]


def test_walk_forward_purge_inserts_gap():
    from datetime import datetime, timedelta

    bar_seconds = 3600  # 1h bars
    purge = 24  # one day embargo
    splits = build_walk_forward_splits(
        base_start=datetime(2024, 1, 1),
        folds=2,
        purge_embargo_bars=purge,
        bar_seconds=bar_seconds,
    )
    gap = timedelta(seconds=purge * bar_seconds)
    for s in splits:
        assert s["val_start"] == s["train_end"] + gap
        assert s["test_start"] == s["val_end"] + gap


def test_walk_forward_purge_without_bar_seconds_stays_contiguous():
    from datetime import datetime

    # No bar_seconds -> no duration to convert bars, so behavior is unchanged.
    splits = build_walk_forward_splits(
        base_start=datetime(2024, 1, 1), folds=2, purge_embargo_bars=10
    )
    for s in splits:
        assert s["val_start"] == s["train_end"]
        assert s["test_start"] == s["val_end"]
