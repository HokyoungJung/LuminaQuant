from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.research import run_alpha_zoo_69_asset_profile_optuna_hybrid_refit as module
from scripts.research import run_alpha_zoo_69_asset_optuna_hybrid_refit as broad69


def _stream(symbol: str, anchor: str, notional: float = 1.0) -> broad69.CandidateStream:
    index = pd.date_range("2026-01-01", periods=4, freq="h")
    row = {
        "model_id": f"m_{symbol}",
        "symbol": symbol,
        "asset_group": broad69._asset_group(symbol),
        "family": "trend_pullback_reclaim",
        "notional_fraction": notional,
        "dominant_anchor": anchor,
    }
    returns = pd.Series([0.01, -0.002, 0.003, 0.0], index=index)
    position = pd.Series([1.0, 1.0, 0.0, 0.0], index=index)
    return broad69.CandidateStream(row=row, returns=returns, position=position)


def test_profile_concentration_tracks_domain_anchor_shares() -> None:
    streams = [_stream("BTCUSDT", "crypto_beta_btc"), _stream("SPYUSDT", "us_equity_beta_spy")]

    concentration = module._profile_concentration(streams, np.array([1.0, 0.5]))

    assert concentration["top_symbol"] == "BTCUSDT"
    assert concentration["top_anchor"] == "crypto_beta_btc"
    assert concentration["top_anchor_share"] == pytest.approx(2 / 3)
    assert concentration["asset_group_shares"]["crypto_core"] == pytest.approx(2 / 3)


def test_candidate_objective_penalizes_single_anchor_clone() -> None:
    base = {
        "train_return": 0.10,
        "validation_return": 0.05,
        "train_mdd": 0.02,
        "validation_mdd": 0.02,
        "train_return_per_turnover_proxy_bps": 50.0,
        "validation_return_per_turnover_proxy_bps": 50.0,
        "train_trade_event_count": 100,
        "validation_trade_event_count": 40,
    }
    spec = module.PROFILE_SPECS[0]

    diversified = module._candidate_objective({**base, "dominant_anchor_abs_corr": 0.20}, spec)
    clone = module._candidate_objective({**base, "dominant_anchor_abs_corr": 0.95}, spec)

    assert diversified > clone


def test_params_from_trial_uses_requested_timeframe_and_integer_leverage_cap() -> None:
    class Trial:
        def suggest_categorical(self, name, choices):
            return choices[-1]

        def suggest_int(self, name, low, high, step=1):
            return high

        def suggest_float(self, name, low, high, step=None):
            return high

    params = module._params_from_trial(Trial(), {**module.PROFILE_SPECS[0], "_timeframes": ("2h",)})

    assert params["timeframe"] == "2h"
    assert params["integer_leverage"] == module.PROFILE_SPECS[0]["max_integer_leverage"]
    assert params["family"] == "trend_pullback_reclaim"


def test_split_windows_for_hybrid_preserves_empty_locked_oos() -> None:
    windows = broad69.SplitWindows(
        train=(pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-02")),
        validation=(pd.Timestamp("2026-01-03"), pd.Timestamp("2026-01-04")),
    )

    split_windows = module._split_windows_for_hybrid(windows.as_payload())

    assert split_windows["locked_oos"][0] > split_windows["locked_oos"][1]
