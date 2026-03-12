from __future__ import annotations

from lumina_quant.strategy_factory import build_binance_futures_candidates


def test_4h_pair_focus_includes_metal_and_mixed_pairs_when_symbols_present():
    rows = build_binance_futures_candidates(
        timeframes=["4h"],
        symbols=["BTC/USDT", "ETH/USDT", "XAU/USDT", "XAG/USDT", "XPT/USDT", "XPD/USDT"],
    )
    pair_rows = [
        row
        for row in rows
        if row.strategy_class == "PairSpreadZScoreStrategy"
    ]
    pair_set = {tuple(row.symbols) for row in pair_rows}

    assert ("XAU/USDT", "XAG/USDT") in pair_set
    assert ("XPT/USDT", "XPD/USDT") in pair_set
    assert ("BTC/USDT", "XAU/USDT") in pair_set
    assert ("ETH/USDT", "XAU/USDT") in pair_set
