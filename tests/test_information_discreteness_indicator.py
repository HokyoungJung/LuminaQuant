"""Closed-form unit tests for the ``information_discreteness`` sign-census numeric."""

from __future__ import annotations

from lumina_quant.indicators.information_discreteness import information_discreteness


def _compound(returns: list[float], p0: float = 100.0) -> list[float]:
    path = [p0]
    for value in returns:
        path.append(path[-1] * (1.0 + value))
    return path[1:]


def test_all_up_days_is_minus_one() -> None:
    # Every formation day positive -> pct_pos = 1, pct_neg = 0, PRET > 0 -> ID = -1.
    closes = _compound([0.003] * 80)
    value = information_discreteness(closes, formation_bars=56, skip_bars=7)
    assert value is not None and abs(value + 1.0) < 1e-9, value


def test_all_down_days_is_minus_one() -> None:
    # Every formation day negative -> pct_neg = 1, pct_pos = 0, PRET < 0 ->
    # sign(-) * (1 - 0) = -1 (a continuous BLEEDER is also low-ID).
    closes = _compound([-0.003] * 80)
    value = information_discreteness(closes, formation_bars=56, skip_bars=7)
    assert value is not None and abs(value + 1.0) < 1e-9, value


def test_jump_driven_path_is_positive() -> None:
    # Flat negative drift + a single large jump inside the formation window:
    # PRET > 0 while most days are down -> ID strongly positive (discrete).
    returns = [-0.001] * 80
    returns[80 - 18] = 0.30
    closes = _compound(returns)
    value = information_discreteness(closes, formation_bars=56, skip_bars=7)
    assert value is not None and value > 0.5, value


def test_zigzag_winner_is_negative() -> None:
    block = [0.012, 0.012, 0.012, 0.012, -0.04]
    closes = _compound([block[i % 5] for i in range(80)])
    value = information_discreteness(closes, formation_bars=56, skip_bars=7)
    assert value is not None and abs(value + 0.607) < 0.02, value


def test_skip_window_is_excluded() -> None:
    # A large jump inside the SKIP window must not enter the sign census.
    returns = [0.002] * 80
    returns[-3] = 0.30  # within the last skip_bars=7 -> excluded
    with_jump = information_discreteness(_compound(returns), formation_bars=56, skip_bars=7)
    without = information_discreteness(_compound([0.002] * 80), formation_bars=56, skip_bars=7)
    assert with_jump == without  # both -1: the skip-window jump is invisible


def test_short_history_returns_none() -> None:
    assert information_discreteness(_compound([0.01] * 20), formation_bars=56, skip_bars=7) is None


def test_flat_formation_returns_none() -> None:
    # A perfectly flat window has no directional PRET -> undefined sign census.
    assert information_discreteness([100.0] * 80, formation_bars=56, skip_bars=7) is None


def test_degenerate_inputs_never_raise() -> None:
    assert information_discreteness([], formation_bars=56, skip_bars=7) is None
    assert information_discreteness([1.0, 2.0], formation_bars=56, skip_bars=7) is None
    assert information_discreteness([float("nan")] * 80, formation_bars=56, skip_bars=7) is None
    assert information_discreteness([-1.0] * 80, formation_bars=56, skip_bars=7) is None
    assert information_discreteness(["x", None, object()], formation_bars=56, skip_bars=7) is None


def test_range_bounded() -> None:
    closes = _compound([0.01 if i % 3 else -0.02 for i in range(80)])
    value = information_discreteness(closes, formation_bars=56, skip_bars=7)
    assert value is not None and -1.0 <= value <= 1.0
