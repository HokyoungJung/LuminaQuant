from __future__ import annotations

from lumina_quant.strategy_factory.candidate_library import build_binance_futures_candidates
from lumina_quant.symbol_universe import canonicalize_research_symbol


def test_canonicalize_research_symbol_keeps_slash_usdt_form():
    assert canonicalize_research_symbol("BTCUSDT") == "BTC/USDT"
    assert canonicalize_research_symbol("btc-usdt") == "BTC/USDT"
    assert canonicalize_research_symbol("BTC_USDT") == "BTC/USDT"
    assert canonicalize_research_symbol("XAU/USDT:USDT") == "XAU/USDT"
    assert canonicalize_research_symbol("XAG/USDT:USDT") == "XAG/USDT"


def test_candidate_builder_normalizes_symbol_variants_to_canonical_form():
    candidates = build_binance_futures_candidates(
        timeframes=["1m"],
        symbols=["BTCUSDT", "ETH-USDT", "XAU/USDT:USDT", "XAG_USDT"],
    )
    assert candidates

    for row in candidates:
        symbols = row.symbols
        assert all(symbol.endswith("/USDT") for symbol in symbols)
        assert all(":" not in symbol for symbol in symbols)
        assert all("-" not in symbol and "_" not in symbol for symbol in symbols)
