"""E2E gate for spec AC8: add ONE template file + config edit → registers,
resolves, and runs — with NO edit to framework internals (registry.py).

This is the test whose absence let the decorative plugin registry ship: it
drops a single conforming module into the strategies package, references the
strategy by name through ``config.optimization.strategy``, resolves it through
the same code path the CLI uses, runs a real (CSV-backed) backtest, and asserts
the dropped strategy actually executed (market events processed, signals + fills
produced).  The same dropped file also registers an indicator and a portfolio
plugin to prove registration + resolution for all three required kinds.

The probe module is created in setup and removed in teardown; GLOBAL_REGISTRY
and the discovery cache are restored so no state leaks to other tests.
"""

from __future__ import annotations

import sys
import textwrap
from datetime import datetime
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STRATEGIES_DIR = PROJECT_ROOT / "src" / "lumina_quant" / "strategies"
DATA_DIR = PROJECT_ROOT / "data"

# A user touches exactly these two surfaces: ONE new file + a config name.
PROBE_MODULE = "e2e_dropin_probe"
PROBE_STRATEGY = "E2eDropinProbeStrategy"
PROBE_INDICATOR = "E2eDropinProbeIndicator"
PROBE_PORTFOLIO = "E2eDropinProbePortfolio"

# The conforming template a user would drop in — decorator registration only,
# no edit to registry.py / _STRATEGY_MAP / param_registry.py.
PROBE_SOURCE = textwrap.dedent(
    f'''
    """Drop-in template probe — registers via @register only (spec AC8)."""
    from __future__ import annotations

    from lumina_quant.core.events import SignalEvent
    from lumina_quant.core.plugin_registry import register
    from lumina_quant.strategy import Strategy


    @register("strategy", "{PROBE_STRATEGY}", interface="event_driven")
    class {PROBE_STRATEGY}(Strategy):
        """Goes LONG once per symbol on its first MARKET bar."""

        def __init__(self, bars, events, **params):
            self.bars = bars
            self.events = events
            self.params = dict(params)
            self._entered: set[str] = set()

        def calculate_signals(self, event):
            if getattr(event, "type", None) != "MARKET":
                return
            symbol = str(getattr(event, "symbol", ""))
            if not symbol or symbol in self._entered:
                return
            self._entered.add(symbol)
            self.events.put(
                SignalEvent(
                    strategy_id="e2e_dropin_probe",
                    symbol=symbol,
                    datetime=getattr(event, "time", None),
                    signal_type="LONG",
                    strength=1.0,
                    metadata={{"strategy": "{PROBE_STRATEGY}"}},
                )
            )


    @register("indicator", "{PROBE_INDICATOR}")
    class {PROBE_INDICATOR}:
        pass


    @register("portfolio", "{PROBE_PORTFOLIO}")
    class {PROBE_PORTFOLIO}:
        pass
    '''
).strip()


@pytest.fixture
def dropped_strategy_file(monkeypatch, tmp_path):
    """Create the probe module + a config that references it by name, yield, clean up."""
    probe_path = STRATEGIES_DIR / f"{PROBE_MODULE}.py"
    probe_path.write_text(PROBE_SOURCE + "\n", encoding="utf-8")

    # The ONLY other thing a user edits: config.optimization.strategy = <name>.
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        textwrap.dedent(
            f"""
            optimization:
              strategy: {PROBE_STRATEGY}
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("LQ_CONFIG_PATH", str(config_path))

    try:
        yield config_path
    finally:
        probe_path.unlink(missing_ok=True)
        # Restore global registry + discovery cache so no state leaks.
        from lumina_quant.core import plugin_registry as pr
        from lumina_quant.strategies import registry as reg

        for kind, name in (
            ("strategy", PROBE_STRATEGY),
            ("indicator", PROBE_INDICATOR),
            ("portfolio", PROBE_PORTFOLIO),
        ):
            pr._REGISTRY.get(kind, {}).pop(name, None)
        reg._PLUGIN_DISCOVERY_DONE.discard(PROBE_MODULE)
        sys.modules.pop(f"lumina_quant.strategies.{PROBE_MODULE}", None)


def test_dropfile_strategy_registers_resolves_and_runs(dropped_strategy_file):
    """One dropped file + a config name → discoverable, config-resolvable, runnable."""
    from lumina_quant.backtesting.backtest import Backtest
    from lumina_quant.backtesting.data import HistoricCSVDataHandler
    from lumina_quant.backtesting.execution_sim import SimulatedExecutionHandler
    from lumina_quant.backtesting.portfolio_backtest import Portfolio
    from lumina_quant.cli import backtest as cli_backtest
    from lumina_quant.configuration import get_default_runtime_config
    from lumina_quant.core.plugin_registry import GLOBAL_REGISTRY
    from lumina_quant.strategies import registry as reg

    # 1) Auto-discovery makes the dropped strategy appear in the operational map
    #    with NO edit to registry.py.
    strategy_map = reg.get_strategy_map()
    assert PROBE_STRATEGY in strategy_map, "dropped strategy not auto-discovered"

    # 2) The config name resolves to the dropped class through the registry.
    rt = get_default_runtime_config()
    assert rt.optimization.strategy == PROBE_STRATEGY
    resolved = reg.resolve_strategy_class(rt.optimization.strategy)
    assert resolved.__name__ == PROBE_STRATEGY

    # 3) The CLI strategy-selection path resolves the same class from config.
    setup = cli_backtest._resolve_strategy_setup_detail(log=False)
    assert setup.strategy_cls.__name__ == PROBE_STRATEGY

    # 4) Indicator + portfolio registered from the SAME dropped file resolve too.
    assert GLOBAL_REGISTRY.get("indicator", PROBE_INDICATOR) is not None
    assert GLOBAL_REGISTRY.get("portfolio", PROBE_PORTFOLIO) is not None

    # 5) A real backtest selecting the dropped strategy actually RUNS and trades.
    backtest = Backtest(
        csv_dir=str(DATA_DIR),
        symbol_list=["BTCUSDT"],
        start_date=datetime(2024, 1, 1),
        data_handler_cls=HistoricCSVDataHandler,
        execution_handler_cls=SimulatedExecutionHandler,
        portfolio_cls=Portfolio,
        strategy_cls=setup.strategy_cls,
        strategy_params=setup.strategy_params,
        record_history=True,
        track_metrics=True,
        record_trades=True,
    )
    backtest.simulate_trading(output=False)

    assert backtest.market_events > 0, "backtest did not process any market bars"
    assert backtest.signals > 0, "dropped strategy produced no signals"
    assert backtest.fills > 0, "dropped strategy produced no fills"
