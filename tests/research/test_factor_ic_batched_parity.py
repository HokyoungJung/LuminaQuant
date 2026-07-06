"""Parity proof: the vectorized batched reducer vs the per-timestamp loop reference.

``reduce_factor_ic`` now dispatches to ``_reduce_factor_ic_batched`` (vectorized
over timestamps per factor) for production use. ``_reduce_factor_ic_loop`` -- the
original per-(factor, timestamp) Python loop -- is kept solely as a permanent
parity oracle. This suite pins ``batched == loop`` bit-for-bit across shapes
chosen to stress every code path that had to be vectorized carefully:

* few symbols, many timestamps (turnover/decay-heavy: long time axis)
* a wide universe (stresses the O(symbols) rank/sum machinery)
* heavy ties (exercises average-rank tie grouping under batching)
* NaN blocks (ragged per-timestamp valid counts -> exercises the
  grouped-length summation path, not just the padding-safe rank path)
* min-obs edge cases (cross-sectional counts hovering right at
  ``min_cross_section`` / the hardcoded quantile-spread floor of 5)
* a genuinely ragged panel (missing rows, not just NaN factor values)
* degenerate shapes: a single timestamp, a single symbol, an empty panel,
  and an all-NaN factor column

All assertions are exact equality (``==``) -- no ``rtol``/``atol`` anywhere --
because NumPy's pairwise summation is provably length-dependent (padding a
row to a common width perturbs ``np.sum`` almost always), so anything less
than bit-identical would mean the batched path is silently computing a
different number, not just a differently-rounded one.
"""

from __future__ import annotations

import numpy as np
import pytest

from lumina_quant.research.factor_ic import (
    FactorICResult,
    _reduce_factor_ic_batched,
    _reduce_factor_ic_loop,
    evaluate_factor_ic_numpy,
    reduce_factor_ic,
)


def _make_panel(
    seed: int,
    n_symbols: int,
    n_timestamps: int,
    n_factors: int,
    *,
    nan_rate: float = 0.0,
    tie_rate: float = 0.0,
    drop_rate: float = 0.0,
) -> dict[str, np.ndarray | list[str]]:
    """Deterministic synthetic long-format panel, seeded via ``default_rng``."""
    rng = np.random.default_rng(seed)
    symbols = np.array([f"SYM{i:04d}" for i in range(n_symbols)])
    timestamps = np.arange(n_timestamps, dtype=np.int64) * 60_000
    sym_col = np.repeat(symbols, n_timestamps)
    ts_col = np.tile(timestamps, n_symbols)
    n_rows = n_symbols * n_timestamps
    factors = rng.standard_normal((n_rows, n_factors))
    if tie_rate > 0.0:
        for col in range(n_factors):
            if rng.random() < tie_rate:
                factors[:, col] = np.round(factors[:, col] * 2.0) / 2.0
    fwd = 0.3 * factors[:, 0] + rng.standard_normal(n_rows) * 0.9
    if tie_rate > 0.0 and rng.random() < tie_rate:
        fwd = np.round(fwd * 3.0) / 3.0

    if nan_rate > 0.0:
        factor_nan_mask = rng.random(factors.shape) < nan_rate
        factors[factor_nan_mask] = np.nan
        fwd_nan_mask = rng.random(n_rows) < nan_rate
        fwd[fwd_nan_mask] = np.nan

    if drop_rate > 0.0:
        keep = rng.random(n_rows) > drop_rate
        sym_col = sym_col[keep]
        ts_col = ts_col[keep]
        factors = factors[keep]
        fwd = fwd[keep]

    names = [f"f{i}" for i in range(n_factors)]
    return {
        "symbol": sym_col,
        "timestamp": ts_col,
        "factors": factors,
        "forward_return": fwd,
        "factor_names": names,
    }


def _assert_bit_identical(loop_result, batched_result, *, label: str) -> None:
    loop_map = loop_result.as_mapping()
    batched_map = batched_result.as_mapping()
    assert set(loop_map) == set(batched_map), label
    assert loop_result.n_rows == batched_result.n_rows, label
    assert loop_result.n_symbols == batched_result.n_symbols, label
    assert loop_result.n_timestamps == batched_result.n_timestamps, label
    assert loop_result.min_cross_section == batched_result.min_cross_section, label
    assert loop_result.max_decay_lag == batched_result.max_decay_lag, label
    for name, loop_res in loop_map.items():
        batched_res: FactorICResult = batched_map[name]
        ctx = f"{label} factor={name}"
        assert loop_res.n_periods == batched_res.n_periods, ctx
        assert loop_res.n_observations == batched_res.n_observations, ctx
        assert loop_res.ic_mean == batched_res.ic_mean, ctx
        assert loop_res.ic_std == batched_res.ic_std, ctx
        assert loop_res.ic_ir == batched_res.ic_ir, ctx
        assert loop_res.ic_positive_ratio == batched_res.ic_positive_ratio, ctx
        assert loop_res.t_stat == batched_res.t_stat, ctx
        assert loop_res.quantile_spread_mean == batched_res.quantile_spread_mean, ctx
        assert loop_res.turnover_mean == batched_res.turnover_mean, ctx
        assert np.array_equal(
            np.asarray(loop_res.rank_autocorr, dtype=np.float64),
            np.asarray(batched_res.rank_autocorr, dtype=np.float64),
        ), ctx
    # Belt-and-suspenders: the full serialized payload must also match exactly.
    assert loop_result.to_dict() == batched_result.to_dict(), label


def _run_parity(panel: dict, *, label: str, **kwargs) -> None:
    loop_result = _reduce_factor_ic_loop(
        panel["symbol"],
        panel["timestamp"],
        panel["factors"],
        panel["forward_return"],
        panel["factor_names"],
        **kwargs,
    )
    batched_result = _reduce_factor_ic_batched(
        panel["symbol"],
        panel["timestamp"],
        panel["factors"],
        panel["forward_return"],
        panel["factor_names"],
        **kwargs,
    )
    _assert_bit_identical(loop_result, batched_result, label=label)


# ---------------------------------------------------------------------------
# Named shapes required by the parity campaign.
# ---------------------------------------------------------------------------


def test_parity_few_symbols_many_timestamps():
    panel = _make_panel(20260703001, n_symbols=5, n_timestamps=500, n_factors=3, nan_rate=0.02)
    _run_parity(panel, label="few_symbols_many_timestamps")


def test_parity_wide_universe():
    panel = _make_panel(20260703002, n_symbols=350, n_timestamps=25, n_factors=4, nan_rate=0.01)
    _run_parity(panel, label="wide_universe")


def test_parity_heavy_ties():
    panel = _make_panel(20260703003, n_symbols=14, n_timestamps=70, n_factors=3, tie_rate=1.0)
    _run_parity(panel, label="heavy_ties")


def test_parity_nan_blocks():
    panel = _make_panel(20260703004, n_symbols=18, n_timestamps=90, n_factors=3, nan_rate=0.2)
    _run_parity(panel, label="nan_blocks")


@pytest.mark.parametrize("min_cross_section", [3, 4, 5, 8])
def test_parity_min_obs_edge_cases(min_cross_section):
    # Small, heavily NaN'd universe so cross-sectional counts frequently sit
    # right at (and below) min_cross_section and the hardcoded quantile-spread
    # floor of 5.
    panel = _make_panel(20260703005, n_symbols=6, n_timestamps=120, n_factors=2, nan_rate=0.45)
    _run_parity(
        panel,
        label=f"min_obs_edge_min_cs={min_cross_section}",
        min_cross_section=min_cross_section,
    )


def test_parity_ragged_missing_rows():
    # Not just NaN factor values -- entire (symbol, timestamp) rows absent,
    # which is the case the dense-scatter construction must reconstruct
    # identically to the loop's boundary-slice approach.
    panel = _make_panel(
        20260703006, n_symbols=24, n_timestamps=60, n_factors=3, nan_rate=0.05, drop_rate=0.25
    )
    _run_parity(panel, label="ragged_missing_rows")


@pytest.mark.parametrize(
    ("top_quantile", "bottom_quantile"), [(0.9, 0.1), (0.6, 0.4), (0.95, 0.05)]
)
def test_parity_quantile_thresholds(top_quantile, bottom_quantile):
    panel = _make_panel(20260703007, n_symbols=16, n_timestamps=80, n_factors=3, nan_rate=0.1)
    _run_parity(
        panel,
        label=f"quantiles_{top_quantile}_{bottom_quantile}",
        top_quantile=top_quantile,
        bottom_quantile=bottom_quantile,
    )


# ---------------------------------------------------------------------------
# Degenerate shapes.
# ---------------------------------------------------------------------------


def test_parity_single_timestamp():
    panel = _make_panel(20260703008, n_symbols=12, n_timestamps=1, n_factors=2)
    _run_parity(panel, label="single_timestamp")


def test_parity_single_symbol():
    panel = _make_panel(20260703009, n_symbols=1, n_timestamps=30, n_factors=2)
    _run_parity(panel, label="single_symbol")


def test_parity_empty_panel():
    panel = {
        "symbol": np.array([], dtype="<U8"),
        "timestamp": np.array([], dtype=np.int64),
        "factors": np.empty((0, 2), dtype=np.float64),
        "forward_return": np.empty((0,), dtype=np.float64),
        "factor_names": ["f0", "f1"],
    }
    _run_parity(panel, label="empty_panel")


def test_parity_all_nan_factor_column():
    panel = _make_panel(20260703010, n_symbols=10, n_timestamps=25, n_factors=1, nan_rate=1.0)
    _run_parity(panel, label="all_nan_factor_column")


# ---------------------------------------------------------------------------
# Randomized sweep: many small-to-moderate shapes with varying NaN/tie/drop
# rates, to catch anything the named shapes above happen not to trigger.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("trial", range(40))
def test_parity_randomized_sweep(trial):
    rng = np.random.default_rng(900_000 + trial)
    n_symbols = int(rng.integers(2, 45))
    n_timestamps = int(rng.integers(2, 70))
    n_factors = int(rng.integers(1, 4))
    nan_rate = float(rng.uniform(0.0, 0.35))
    tie_rate = 1.0 if rng.random() < 0.3 else 0.0
    drop_rate = float(rng.uniform(0.0, 0.2)) if rng.random() < 0.5 else 0.0
    panel = _make_panel(
        900_000 + trial,
        n_symbols=n_symbols,
        n_timestamps=n_timestamps,
        n_factors=n_factors,
        nan_rate=nan_rate,
        tie_rate=tie_rate,
        drop_rate=drop_rate,
    )
    _run_parity(panel, label=f"randomized_trial_{trial}")


# ---------------------------------------------------------------------------
# Dispatch correctness: the public entry point must be the batched path.
# ---------------------------------------------------------------------------


def test_public_entry_point_dispatches_to_batched():
    panel = _make_panel(20260703011, n_symbols=10, n_timestamps=40, n_factors=2, nan_rate=0.1)
    via_public = reduce_factor_ic(
        panel["symbol"],
        panel["timestamp"],
        panel["factors"],
        panel["forward_return"],
        panel["factor_names"],
    )
    via_batched = _reduce_factor_ic_batched(
        panel["symbol"],
        panel["timestamp"],
        panel["factors"],
        panel["forward_return"],
        panel["factor_names"],
    )
    assert via_public.to_dict() == via_batched.to_dict()


# ---------------------------------------------------------------------------
# N2: the (symbol, timestamp) uniqueness precondition is enforced.
#
# The dense-scatter batched path resolves a duplicate coordinate last-write-wins
# while the loop oracle keeps every duplicate, so the two silently diverge on
# such input. reduce_factor_ic (and the public entry points that funnel through
# it) must refuse the panel rather than return an order-dependent number.
# ---------------------------------------------------------------------------


def _panel_with_duplicate_coordinate(seed: int) -> dict:
    """A clean panel with ONE (symbol, timestamp) coordinate duplicated."""
    panel = _make_panel(seed, n_symbols=8, n_timestamps=20, n_factors=2)
    # Append a second row carrying the very first row's coordinate but a
    # different factor/label value -> a genuine duplicate the reducer collides.
    sym = np.concatenate([panel["symbol"], panel["symbol"][:1]])
    ts = np.concatenate([panel["timestamp"], panel["timestamp"][:1]])
    factors = np.concatenate([panel["factors"], panel["factors"][:1] + 0.5], axis=0)
    fwd = np.concatenate([panel["forward_return"], panel["forward_return"][:1] + 0.5])
    return {
        "symbol": sym,
        "timestamp": ts,
        "factors": factors,
        "forward_return": fwd,
        "factor_names": panel["factor_names"],
    }


def test_reduce_factor_ic_rejects_duplicate_coordinates():
    panel = _panel_with_duplicate_coordinate(20260706001)
    with pytest.raises(ValueError, match="duplicate_symbol_timestamp_pairs"):
        reduce_factor_ic(
            panel["symbol"],
            panel["timestamp"],
            panel["factors"],
            panel["forward_return"],
            panel["factor_names"],
        )


def test_public_numpy_entry_rejects_duplicate_coordinates():
    panel = _panel_with_duplicate_coordinate(20260706002)
    with pytest.raises(ValueError, match="duplicate"):
        evaluate_factor_ic_numpy(
            panel["symbol"],
            panel["timestamp"],
            panel["factors"],
            panel["forward_return"],
            panel["factor_names"],
        )


def test_unique_coordinate_panel_still_accepted():
    # Guardrail: the precondition check must be a no-op on the (always unique)
    # cartesian panels the rest of the suite -- and every real caller -- feeds.
    panel = _make_panel(20260706003, n_symbols=9, n_timestamps=15, n_factors=3, nan_rate=0.1)
    result = reduce_factor_ic(
        panel["symbol"],
        panel["timestamp"],
        panel["factors"],
        panel["forward_return"],
        panel["factor_names"],
    )
    assert result.n_rows == 9 * 15


def test_single_row_panel_not_falsely_rejected():
    # n_rows < 2 -> no pair can duplicate; the check must short-circuit cleanly.
    result = reduce_factor_ic(
        np.array(["SYM0"]),
        np.array([0], dtype=np.int64),
        np.array([[1.0, 2.0]], dtype=np.float64),
        np.array([0.1], dtype=np.float64),
        ["f0", "f1"],
    )
    assert result.n_rows == 1


# ---------------------------------------------------------------------------
# X4: the fwd-rank cache (reused across factors sharing a joint finite mask)
# must be bit-identical to the per-factor computation and to the loop oracle.
# ---------------------------------------------------------------------------


def test_x4_shared_mask_multifactor_is_bit_identical():
    # No NaNs -> every factor's joint (factor & fwd) finite mask equals fwd's,
    # so all K factors hit the SAME cache entry: the reuse path is exercised and
    # must still match the loop oracle bit-for-bit.
    panel = _make_panel(20260706004, n_symbols=20, n_timestamps=60, n_factors=6, nan_rate=0.0)
    _run_parity(panel, label="x4_shared_mask_multifactor")


def test_x4_mixed_masks_multifactor_is_bit_identical():
    # NaNs make some factors share a mask and others not -> cache hits AND
    # misses interleave within one reduce call; still bit-identical to the loop.
    panel = _make_panel(20260706005, n_symbols=16, n_timestamps=50, n_factors=5, nan_rate=0.15)
    _run_parity(panel, label="x4_mixed_masks_multifactor")
