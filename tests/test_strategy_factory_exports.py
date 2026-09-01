from __future__ import annotations

from lumina_quant import strategy_factory


def test_strategy_factory_exports_dispatcher_once() -> None:
    exported = strategy_factory.__all__

    assert exported.count("build_default_candidate_rows") == 1
    assert exported.count("run_candidate_research") == 1
    assert exported.count("StrategySignalDispatcher") == 1
    assert callable(strategy_factory.build_default_candidate_rows)
    assert callable(strategy_factory.run_candidate_research)
    assert strategy_factory.StrategySignalDispatcher.__name__ == "StrategySignalDispatcher"
