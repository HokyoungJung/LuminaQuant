from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from lumina_quant.alpha_zoo import native_live_signal_backend as native_backend
from lumina_quant.alpha_zoo.optuna_hybrid_signals import (
    debounced_state_signal,
    trailing_state_signal,
)


class _temporary_backend:
    def __init__(self, value: str) -> None:
        self.value = value
        self.previous: str | None = None

    def __enter__(self) -> None:
        self.previous = os.environ.get(native_backend.LIVE_SIGNAL_BACKEND_ENV)
        os.environ[native_backend.LIVE_SIGNAL_BACKEND_ENV] = self.value

    def __exit__(self, *_exc: object) -> None:
        if self.previous is None:
            os.environ.pop(native_backend.LIVE_SIGNAL_BACKEND_ENV, None)
        else:
            os.environ[native_backend.LIVE_SIGNAL_BACKEND_ENV] = self.previous


def test_python_backend_mode_returns_none() -> None:
    values = np.array([True, False, True], dtype=bool)

    assert (
        native_backend.evaluate_debounced_state_native(
            values,
            ~values,
            np.zeros(3, dtype=bool),
            np.zeros(3, dtype=bool),
            side="long_only",
            min_hold_bars=1,
            cooldown_bars=0,
            backend="python",
        )
        is None
    )


def test_debounced_live_signal_rust_matches_python_when_available() -> None:
    if not native_backend.native_backend_available():
        pytest.skip("Rust live-signal backend is not built in this environment")

    rng = np.random.default_rng(20260527)
    length = 2048
    long_entry = pd.Series(rng.random(length) < 0.08)
    long_exit = pd.Series(rng.random(length) < 0.05)
    short_entry = pd.Series(rng.random(length) < 0.07)
    short_exit = pd.Series(rng.random(length) < 0.06)

    with _temporary_backend("python"):
        expected = debounced_state_signal(
            long_entry,
            long_exit,
            short_entry,
            short_exit,
            side="long_short",
            min_hold_bars=5,
            cooldown_bars=3,
        )
    with _temporary_backend("rust"):
        observed = debounced_state_signal(
            long_entry,
            long_exit,
            short_entry,
            short_exit,
            side="long_short",
            min_hold_bars=5,
            cooldown_bars=3,
        )

    assert np.array_equal(observed, expected)


def test_trailing_live_signal_rust_matches_python_when_available() -> None:
    if not native_backend.native_backend_available():
        pytest.skip("Rust live-signal backend is not built in this environment")

    rng = np.random.default_rng(20260528)
    length = 2048
    close = pd.Series(100.0 + np.cumsum(rng.normal(0.0, 0.25, size=length)))
    atr = pd.Series(np.maximum(0.05, rng.lognormal(mean=-2.8, sigma=0.35, size=length)))
    close.iloc[25] = np.nan
    atr.iloc[40] = np.nan
    atr.iloc[41] = 0.0
    long_entry = pd.Series(rng.random(length) < 0.05)
    short_entry = pd.Series(rng.random(length) < 0.05)
    long_exit = pd.Series(rng.random(length) < 0.04)
    short_exit = pd.Series(rng.random(length) < 0.04)

    with _temporary_backend("python"):
        expected = trailing_state_signal(
            close,
            long_entry,
            short_entry,
            long_exit,
            short_exit,
            atr,
            side="long_short",
            min_hold_bars=4,
            cooldown_bars=2,
            trail_atr_mult=2.25,
        )
    with _temporary_backend("rust"):
        observed = trailing_state_signal(
            close,
            long_entry,
            short_entry,
            long_exit,
            short_exit,
            atr,
            side="long_short",
            min_hold_bars=4,
            cooldown_bars=2,
            trail_atr_mult=2.25,
        )

    assert np.array_equal(observed, expected)


def test_live_signal_backend_diagnostics_reports_requested_mode() -> None:
    diagnostics = native_backend.live_signal_backend_diagnostics("python")

    assert diagnostics["requested_backend"] == "python"
    assert diagnostics["resolved_backend"] == "python"
