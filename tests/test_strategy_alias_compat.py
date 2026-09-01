from __future__ import annotations

import queue

from lumina_quant.strategies.candidate_vol_compression_reversion import (
    VolatilityCompressionReversionStrategy,
)
from lumina_quant.strategies.leadlag_spillover import LeadLagSpilloverStrategy
from lumina_quant.strategies.registry import (
    get_default_optuna_config,
    get_strategy_map,
    get_strategy_tier,
    resolve_strategy_class,
)


class _BarsMock:
    def __init__(self, symbols: list[str]):
        self.symbol_list = list(symbols)


def test_volatility_compression_strategy_accepts_legacy_alias_kwargs():
    strategy = VolatilityCompressionReversionStrategy(
        _BarsMock(["BTC/USDT"]),
        queue.Queue(),
        compression_vol_ratio=0.55,
        atr_stop_pct=0.01,
        compression_threshold=0.72,
        stop_loss_pct=0.04,
    )

    assert strategy.compression_vol_ratio == 0.72
    assert strategy.atr_stop_pct == 0.04


def test_leadlag_strategy_accepts_legacy_alias_kwargs():
    strategy = LeadLagSpilloverStrategy(
        _BarsMock(["BTC/USDT", "ETH/USDT", "BNB/USDT"]),
        queue.Queue(),
        entry_score=0.25,
        exit_score=0.05,
        entry_spillover=0.61,
        exit_spillover=0.12,
    )

    assert strategy.entry_score == 0.61
    assert strategy.exit_score == 0.12


def test_registry_keeps_alias_strategy_entries_and_optuna_defaults():
    mapping = get_strategy_map()

    assert "PairSpreadZScoreStrategy" in mapping
    assert "LeadLagSpilloverStrategy" in mapping
    assert "VolCompressionVWAPReversionStrategy" in mapping
    assert "VolCompressionVwapReversionStrategy" in mapping
    assert (
        resolve_strategy_class("VolCompressionVWAPReversionStrategy").__name__
        == "VolCompressionVWAPReversionStrategy"
    )
    assert (
        resolve_strategy_class("VolCompressionVwapReversionStrategy").__name__
        == "VolCompressionVwapReversionStrategy"
    )
    assert get_strategy_tier("VolCompressionVWAPReversionStrategy") == "live_opt_in"
    assert get_strategy_tier("VolCompressionVwapReversionStrategy") == "live_opt_in"
    assert get_default_optuna_config("VolCompressionVWAPReversionStrategy")["n_trials"] == 24
    assert get_default_optuna_config("VolCompressionVwapReversionStrategy")["n_trials"] == 24
