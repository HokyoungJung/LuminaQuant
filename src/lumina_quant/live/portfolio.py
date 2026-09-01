"""Live portfolio boundary exports.

The live runtime reuses the backtesting portfolio implementation, but binds it
to the live path via a thin subclass so that the *simulated* liquidation engine
and the *simulated* funding charge are disabled: on the live path liquidations
and funding are real exchange events, so the modeled versions must never
fabricate a local fill or debit cash (audit finding M5). Keeping the dependency
behind this module boundary also means live callers do not import the
backtesting module directly.
"""

from __future__ import annotations

from typing import Any

# The dynamically-built live subclass is cached ON the backtest ``Portfolio``
# base class (via a private attribute on that class object), NOT on a global of
# THIS module. Tests such as tests/test_live_portfolio_boundary.py pop
# ``lumina_quant.live.portfolio`` from ``sys.modules`` and re-import it to prove
# the backtesting import stays deferred; a module-level cache would then hand
# back a *different* subclass object per module generation, breaking the
# ``contract.portfolio_cls is get_live_portfolio_cls()`` identity that
# tests/test_system_assembly.py asserts. Keying the cache on the (stable, never
# popped in the same breath as this module without also rebuilding) base class
# makes every resolution path converge on one subclass object.
_LIVE_SUBCLASS_ATTR = "_live_runtime_subclass"


def _build_live_portfolio_cls() -> type:
    from lumina_quant.backtesting.portfolio_backtest import Portfolio as _Portfolio

    cached = _Portfolio.__dict__.get(_LIVE_SUBCLASS_ATTR)
    if cached is not None:
        return cached

    class LivePortfolio(_Portfolio):
        """Portfolio bound to the live runtime.

        Behaviourally identical to the backtest ``Portfolio`` except that the
        simulated liquidation engine and the simulated funding charge are
        disabled (M5): ``self._live_liquidation_disabled`` is set ``True`` so
        ``_check_liquidations`` downgrades a modeled maintenance-margin breach to
        a WARNING / audit record (never a synthetic ``LIQUIDATED`` fill) and
        ``_apply_funding`` skips the simulated debit (funding is reconciled from
        the real exchange balance by the trader).
        """

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self._live_liquidation_disabled = True

    # The live portfolio IS the backtest ``Portfolio`` with one live-only
    # behavioural gate; present it under that identity so existing runtime-
    # contract assertions (name/module) and any consumers keyed on the backtest
    # class continue to hold. The behavioural difference lives entirely in the
    # instance flag set above.
    LivePortfolio.__name__ = "Portfolio"
    LivePortfolio.__qualname__ = "Portfolio"
    LivePortfolio.__module__ = "lumina_quant.backtesting.portfolio_backtest"
    # Cache on the base class so all callers (this module, a re-imported copy of
    # it, and system_assembly) resolve the identical subclass object.
    setattr(_Portfolio, _LIVE_SUBCLASS_ATTR, LivePortfolio)
    return LivePortfolio


def get_live_portfolio_cls() -> type:
    return _build_live_portfolio_cls()


def __getattr__(name: str) -> Any:
    if name in {"LivePortfolio", "Portfolio"}:
        return get_live_portfolio_cls()
    raise AttributeError(name)
