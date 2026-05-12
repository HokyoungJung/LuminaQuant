from __future__ import annotations

import math

import pandas as pd

from lumina_quant.alpha_zoo.crypto_fx_factors import (
    FactorSpec,
    add_forward_return_label,
    assign_time_splits,
    build_crypto_fx_factor_specs,
    compute_factor_frame,
    factor_columns,
    screen_factor_frame,
)
from lumina_quant.alpha_zoo.factor_card import build_factor_card
from lumina_quant.alpha_zoo.operators import delta, ts_rank


def _sample_panel(periods: int = 36) -> pd.DataFrame:
    rows = []
    symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCHF"]
    for t in range(periods):
        ts = pd.Timestamp("2026-01-01") + pd.Timedelta(hours=t)
        for symbol in symbols:
            base = 100.0
            drift = 0.001 * t
            if symbol.startswith("ETH"):
                drift = 0.004 * t
            elif symbol.startswith("SOL"):
                drift = -0.002 * t
            elif symbol == "EURUSD":
                base = 1.10
                drift = -0.0005 * t
            elif symbol == "GBPUSD":
                base = 1.25
                drift = -0.0004 * t
            elif symbol == "AUDUSD":
                base = 0.65
                drift = -0.0003 * t
            elif symbol == "USDJPY":
                base = 145.0
                drift = 0.01 * t
            elif symbol == "USDCHF":
                base = 0.90
                drift = 0.0002 * t
            close = base * (1.0 + drift)
            rows.append(
                {
                    "timestamp": ts,
                    "symbol": symbol,
                    "open": close * 0.999,
                    "high": close * 1.002,
                    "low": close * 0.998,
                    "close": close,
                    "volume": 1000.0 + (10.0 * t),
                    "vwap": close * 0.9995,
                    "funding_rate": 0.0001 * math.sin(t / 3),
                }
            )
    return pd.DataFrame(rows)


def test_operator_delta_and_ts_rank_are_causal_by_symbol() -> None:
    frame = pd.DataFrame({"symbol": ["A", "A", "A", "B", "B", "B"], "value": [1, 3, 6, 10, 5, 1]})
    deltas = delta(frame["value"], 1, by=frame["symbol"]).tolist()
    assert math.isnan(deltas[0])
    assert deltas[1:3] == [2.0, 3.0]
    assert math.isnan(deltas[3])
    assert deltas[4:] == [-5.0, -4.0]
    ranks = ts_rank(frame["value"], 3, by=frame["symbol"])
    assert math.isnan(float(ranks.iloc[1]))
    assert ranks.iloc[2] > 0.8
    assert ranks.iloc[5] < 0.3


def test_crypto_fx_factor_specs_are_calendar_safe_and_broad_enough() -> None:
    specs = build_crypto_fx_factor_specs()
    assert len(specs) >= 50
    assert all(spec.is_calendar_safe for spec in specs)
    assert "fx_usd_strength_12" in factor_columns(specs)
    assert "btc_residual_z_24" in factor_columns(specs)


def test_factor_frame_and_screen_never_use_locked_oos_for_selection() -> None:
    factors = compute_factor_frame(_sample_panel())
    labeled = add_forward_return_label(assign_time_splits(factors), horizon=3)
    payload = screen_factor_frame(labeled, top_n=5)
    assert payload["uses_locked_oos_for_selection"] is False
    assert payload["selected_factors"]
    for row in payload["selected_factors"]:
        assert row["selected_using_splits"] == ["train", "validation"]
        assert row["uses_locked_oos_for_selection"] is False
        assert "locked_oos" in row["split_stats"]


def test_factor_card_fails_closed_on_calendar_or_oos_selection() -> None:
    calendar_spec = FactorSpec(
        name="bad_hour_alpha",
        family="calendar",
        market="crypto",
        inputs=("close",),
        description="invalid fixed hour entry",
        calendar_fields=("hour",),
    )
    card = build_factor_card(calendar_spec, uses_locked_oos_for_selection=True, source_refs=("unit",))
    reasons = card.strategy_validity["rejection_reasons"]
    assert card.strategy_validity["pass"] is False
    assert "calendar_entry_field_forbidden:hour" in reasons
    assert "locked_oos_used_for_selection" in reasons


def test_valid_factor_card_records_gate_only_oos_provenance() -> None:
    spec = build_crypto_fx_factor_specs()[0]
    card = build_factor_card(spec, metrics={"rank_ic": 0.1}, source_refs=("unit",))
    assert card.strategy_validity["pass"] is True
    assert card.selection_provenance["selected_using_splits"] == ("train", "validation")
    assert card.selection_provenance["locked_oos_role"] == "gate_report_only"
