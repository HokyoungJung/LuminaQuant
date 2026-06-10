"""Backward-compat re-export — use ``lumina_quant.configuration.views`` directly.

This module existed as the original home of the engine config views.  It was
renamed to ``views.py`` in Phase 1.  Kept here as a thin re-exporter so that
any code that imported from ``lumina_quant.configuration.compat`` continues to
work without modification.

Deletion gate: Phase 3 (backtest engine) / Phase 5 (live trader) — once the
engines are migrated to typed ``RuntimeConfig`` attrs, both this shim and
``views.py`` can be removed.
"""

from lumina_quant.configuration.views import BacktestConfigView, LiveConfigView

__all__ = ["BacktestConfigView", "LiveConfigView"]
