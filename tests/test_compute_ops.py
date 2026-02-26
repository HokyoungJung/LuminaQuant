from __future__ import annotations

import pandas as pd
from lumina_quant.compute import ops as compute_ops
from lumina_quant.indicators import formulaic_operators as fops
from lumina_quant.indicators import volume as volume_indicators


def test_core_compute_ops_smoke():
    values = [1.0, 2.0, 3.0, 4.0]
    assert compute_ops.delta(values, periods=1) == 1.0
    assert compute_ops.ts_sum(values, window=3) == 9.0
    assert compute_ops.ts_std(values, window=4) is not None
    assert compute_ops.ts_rank(values, window=4) is not None
    assert compute_ops.decay_linear(values, window=4) is not None
    assert compute_ops.signed_power(-2.0, 3.0) == -8.0
    assert compute_ops.clip(10.0, 0.0, 5.0) == 5.0
    assert compute_ops.where(True, 1.0, 2.0) == 1.0
    assert compute_ops.adv([100.0, 110.0], [2.0, 3.0], window=2) == 265.0


def test_adv_series_outputs_expected_rolling_mean():
    close_s = pd.Series([100.0, 110.0, 120.0], dtype=float)
    volume_s = pd.Series([2.0, 3.0, 4.0], dtype=float)
    adv = compute_ops.adv_series(close_s, volume_s, window=2)
    assert round(float(adv.iloc[-1]), 8) == 405.0


def test_formulaic_operator_delta_is_routed_to_compute_ops(monkeypatch):
    monkeypatch.setattr(fops.compute_ops, "delta", lambda values, periods=1: 123.0)
    assert fops.delta([1.0, 2.0, 3.0], periods=1) == 123.0


def test_volume_price_volume_correlation_uses_compute_ops(monkeypatch):
    monkeypatch.setattr(volume_indicators.compute_ops, "ts_corr", lambda *_args, **_kwargs: 0.25)
    assert volume_indicators.price_volume_correlation([1.0, 2.0], [3.0, 4.0], window=2) == 0.25

