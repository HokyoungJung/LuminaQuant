"""GARCH(1,1) standardized-innovation breakout rider with a jump-exhaustion gate.

One per-symbol single-asset DIRECTIONAL sleeve that subclasses the proven
:class:`_ReturnRiderBase` (ATR trailing stop + pyramiding + vol-target sizing)
and adds a signal family the registry does not carry: entries triggered by the
CONDITIONAL-volatility-standardized return innovation ``z_t = r_t /
sigma_{t|t-1}`` from a deterministic GARCH(1,1) filter.

THEORY / PROVENANCE.  Absorbed from the precious-metal research repo's
ARMA-GARCH extreme band notebook (``GARCH (1).ipynb``), whose
volatility-clustering recursion is re-implemented dependency-free in
:mod:`lumina_quant.indicators.garch` (variance targeting + deterministic
``(alpha, beta)`` grid MLE — same window, same parameters, bit-for-bit).  The
jump-exhaustion gate is the DeepLearning repo's causal ``rolling_diff_p95``
detector: a move beyond ``jump_mult`` times the trailing ``jump_quantile`` of
absolute returns is treated as a blow-off print to stand aside from, not
chase.

ENTRY.  Each accepted decision bar the realized log return is standardized by
the PRIOR bar's one-step-ahead GARCH sigma forecast (strictly causal: the
forecast uses only returns through ``t-1``).  ``z_t >= band_k`` arms a LONG
continuation ride and ``z_t <= -band_k`` a SHORT (vol-clustering: a
conditional-scale shock marks regime-consistent initiative flow), UNLESS the
raw move breaches the jump gate.  Parameters refit every ``refit_every`` bars
on the trailing ``garch_window`` returns; the sigma forecast is recomputed
every bar under the current parameters.  The winner is ridden with the
inherited ATR trailing stop + pyramiding.

DISTINCT from :class:`VolatilityBreakoutRiderStrategy` (Donchian price channel
+ ATR expansion), :class:`VolatilitySqueezeBreakoutRiderStrategy`
(compression release), :class:`RealizedVolTermStructureStrategy` (realized-vol
horizon structure), and the shock-REVERSION sleeves (which fade large moves):
the trigger is a conditional-variance-standardized return innovation from a
recursive GARCH filter — no other registered sleeve estimates conditional
heteroskedasticity.  OHLCV-only, per-symbol single-asset,
``decision_cadence_seconds >= 1800``, ``None``-safe graceful guards (never
raise).  Data-local (no fetching, no artifact writes) and must still be
validated on the data-bearing machine.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any

from lumina_quant.core.plugin_registry import register
from lumina_quant.indicators.common import safe_float, time_key
from lumina_quant.indicators.garch import garch11_fit, garch11_next_variance
from lumina_quant.indicators.rolling_stats import rolling_quantile
from lumina_quant.strategies.external_alpha_sleeves import (
    _EPS,
    _safe_non_negative_int,
    _Snapshot,
)
from lumina_quant.strategies.return_rider_alpha_sleeves import (
    _ReturnRiderBase,
    _RiderCommon,
    _RiderState,
)
from lumina_quant.tuning import HyperParam


@register("strategy", "GarchInnovationRiderStrategy", interface="event_driven")
class GarchInnovationRiderStrategy(_ReturnRiderBase):
    """Ride conditional-scale return shocks: ``|r_t / sigma_{t|t-1}| >= band_k``.

    A deterministic GARCH(1,1) filter (variance targeting + grid MLE, refit
    every ``refit_every`` bars over the trailing ``garch_window`` returns)
    supplies a strictly causal one-step-ahead sigma forecast.  A LONG ride is
    armed when the current bar's log return standardized by that PRIOR
    forecast reaches ``band_k`` (SHORT at ``-band_k`` when ``allow_short``),
    unless the raw move exceeds ``jump_mult`` times the trailing
    ``jump_quantile`` of absolute returns (a blow-off print to stand aside
    from).  The winner is ridden with the inherited ATR trailing stop +
    pyramiding.

    DISTINCT from the price-channel/compression breakout riders and the
    shock-reversion sleeves: the trigger is a GARCH conditional-variance
    standardized innovation — no other sleeve estimates conditional
    heteroskedasticity.  OHLCV-only, per-symbol single-asset,
    ``decision_cadence_seconds >= 1800``.
    """

    decision_cadence_seconds = 1800
    preferred_contract = "market_window"
    uses_timeframe_aggregator = False

    strategy_name = "GarchInnovationRiderStrategy"
    strategy_id = "garch_innovation_rider"

    @classmethod
    def get_param_schema(cls) -> dict[str, HyperParam]:
        return {
            # Bounded highs/lows keep the worst grid/Optuna combination cheap:
            # the fit is a 300-combo grid over an O(garch_window) recursion, so
            # refit_every=8 x garch_window=2048 stays a few ms per decision bar.
            "garch_window": HyperParam.integer("garch_window", default=192, low=48, high=2048),
            "refit_every": HyperParam.integer("refit_every", default=24, low=8, high=4096),
            "band_k": HyperParam.floating("band_k", default=2.5, low=0.5, high=20.0),
            "jump_window": HyperParam.integer("jump_window", default=192, low=16, high=2048),
            "jump_quantile": HyperParam.floating("jump_quantile", default=0.99, low=0.5, high=1.0),
            "jump_mult": HyperParam.floating("jump_mult", default=3.0, low=0.1, high=100.0),
            "trail_atr_mult": HyperParam.floating(
                "trail_atr_mult", default=3.0, low=0.1, high=20.0
            ),
            "atr_period": HyperParam.integer("atr_period", default=14, low=2, high=1024),
            "max_adds": HyperParam.integer("max_adds", default=2, low=0, high=32),
            "add_step_atr": HyperParam.floating("add_step_atr", default=1.0, low=0.0, high=20.0),
            "vol_window": HyperParam.integer("vol_window", default=48, low=2, high=4096),
            "target_vol": HyperParam.floating("target_vol", default=0.02, low=0.0, high=1.0),
            "max_hold_bars": HyperParam.integer("max_hold_bars", default=240, low=1, high=200000),
            "allow_short": HyperParam.boolean("allow_short", default=True, grid=[True, False]),
            "add_alloc_fraction": HyperParam.floating(
                "add_alloc_fraction", default=0.5, low=0.0, high=1.0
            ),
            "target_allocation": HyperParam.floating(
                "target_allocation", default=0.30, low=0.0, high=5.0, tunable=False
            ),
            "max_order_value": HyperParam.floating(
                "max_order_value", default=5000.0, low=0.0, high=1_000_000.0, tunable=False
            ),
            "min_price": HyperParam.floating("min_price", default=0.10, low=0.0, high=1_000_000.0),
        }

    def __init__(self, bars: Any, events: Any, **params: Any) -> None:
        super().__init__(bars, events, **params)
        size = max(self.garch_window, self.jump_window) + 8
        self._returns: dict[str, deque[float]] = {
            symbol: deque(maxlen=size) for symbol in self.symbol_list
        }
        self._garch_params: dict[str, tuple[float, float, float] | None] = dict.fromkeys(
            self.symbol_list
        )
        self._sigma_sq_forecast: dict[str, float | None] = dict.fromkeys(self.symbol_list)
        self._bars_since_refit: dict[str, int] = dict.fromkeys(self.symbol_list, 0)
        self._last_z: dict[str, float | None] = dict.fromkeys(self.symbol_list)
        self._last_return: dict[str, float | None] = dict.fromkeys(self.symbol_list)
        # The sigma^2 that STANDARDIZED the last return (sigma^2_{t|t-1}), kept
        # separate from _sigma_sq_forecast (already advanced to sigma^2_{t+1|t})
        # so emitted metadata stays consistent: z * sigma == last return.
        self._last_sigma_sq_used: dict[str, float | None] = dict.fromkeys(self.symbol_list)

    def _bind_params(self, resolved: dict[str, Any]) -> None:
        self.garch_window = max(48, int(resolved["garch_window"]))
        self.refit_every = max(8, int(resolved["refit_every"]))
        self.band_k = max(0.5, float(resolved["band_k"]))
        self.jump_window = max(16, int(resolved["jump_window"]))
        self.jump_quantile = max(0.5, min(1.0, float(resolved["jump_quantile"])))
        self.jump_mult = max(0.1, float(resolved["jump_mult"]))

    def _common_config(self, resolved: dict[str, Any]) -> _RiderCommon:
        return self._resolve_common(
            resolved,
            extra_window=max(int(resolved["garch_window"]), int(resolved["jump_window"])) + 2,
        )

    def _process_symbol(self, symbol: str, snapshot: _Snapshot) -> None:
        # Advance the GARCH filter BEFORE the base runs ride+entry, mirroring the
        # base time-key dedup so a repeated boundary firing never double-updates.
        # Causality: the sigma used to standardize THIS bar's return is the
        # forecast computed after the PREVIOUS bar; only afterwards is the new
        # return folded in and the next-bar forecast refreshed.
        item = self._state.get(symbol)
        if item is not None:
            key = time_key(snapshot.time)
            if not (key and key == item.last_time_key):
                self._advance_filter(symbol, item, snapshot)
        super()._process_symbol(symbol, snapshot)

    def _advance_filter(self, symbol: str, item: _RiderState, snapshot: _Snapshot) -> None:
        close = safe_float(snapshot.close)
        if close is None or close <= self.min_price:
            return
        closes = list(item.closes)
        if not closes or closes[-1] <= _EPS:
            return
        ret = math.log(close / closes[-1])
        if not math.isfinite(ret):
            return
        prior_sigma_sq = self._sigma_sq_forecast.get(symbol)
        if prior_sigma_sq is not None and prior_sigma_sq > _EPS:
            z = ret / math.sqrt(prior_sigma_sq)
            self._last_z[symbol] = z if math.isfinite(z) else None
            self._last_sigma_sq_used[symbol] = prior_sigma_sq
        else:
            self._last_z[symbol] = None
            self._last_sigma_sq_used[symbol] = None
        self._last_return[symbol] = ret
        history = self._returns.setdefault(
            symbol, deque(maxlen=max(self.garch_window, self.jump_window) + 8)
        )
        history.append(ret)
        window = list(history)[-self.garch_window :]
        counter = self._bars_since_refit.get(symbol, 0) + 1
        params = self._garch_params.get(symbol)
        if params is None or counter >= self.refit_every:
            fitted = garch11_fit(window)
            if fitted is not None:
                self._garch_params[symbol] = fitted
                params = fitted
            counter = 0
        self._bars_since_refit[symbol] = counter
        if params is not None:
            omega, alpha, beta = params
            self._sigma_sq_forecast[symbol] = garch11_next_variance(
                window, omega=omega, alpha=alpha, beta=beta
            )

    def _jump_gate_blocked(self, symbol: str) -> bool:
        """Return whether the current bar's move is a blow-off print to skip."""
        last_ret = self._last_return.get(symbol)
        history = self._returns.get(symbol)
        if last_ret is None or history is None or len(history) < 2:
            return False
        # Threshold from PRIOR bars only (exclude the current return).
        prior_abs = [abs(value) for value in list(history)[:-1]]
        threshold = rolling_quantile(prior_abs, window=self.jump_window, q=self.jump_quantile)
        if threshold is None or threshold <= _EPS:
            return False
        return abs(last_ret) > self.jump_mult * threshold

    def _symbol_for(self, item: _RiderState) -> str | None:
        for symbol, candidate in self._state.items():
            if candidate is item:
                return symbol
        return None

    def _innovation_direction(self, symbol: str) -> str:
        z = self._last_z.get(symbol)
        if z is None or self._jump_gate_blocked(symbol):
            return ""
        if z >= self.band_k:
            return "LONG"
        if z <= -self.band_k:
            return "SHORT"
        return ""

    def _entry_decision(self, item: _RiderState) -> str:
        symbol = self._symbol_for(item)
        if symbol is None:
            return ""
        return self._innovation_direction(symbol)

    def _should_pyramid(self, item: _RiderState) -> bool:
        symbol = self._symbol_for(item)
        if symbol is None:
            return False
        return self._innovation_direction(symbol) == item.mode

    def _entry_metadata(self, item: _RiderState) -> dict[str, Any]:
        symbol = self._symbol_for(item)
        if symbol is None:
            return {"garch_innovation_z": None, "garch_sigma_forecast": None}
        # Emit the sigma that STANDARDIZED the entry return (sigma_{t|t-1}), not
        # the already-advanced next-bar forecast, so z * sigma == entry return.
        sigma_sq = self._last_sigma_sq_used.get(symbol)
        z = self._last_z.get(symbol)
        return {
            "garch_innovation_z": float(z) if z is not None else None,
            "garch_sigma_forecast": (
                math.sqrt(sigma_sq) if sigma_sq is not None and sigma_sq > 0.0 else None
            ),
        }

    def get_state(self) -> dict[str, Any]:
        state = super().get_state()
        state["garch_state"] = {
            symbol: {
                "returns": list(self._returns.get(symbol, [])),
                "params": (
                    list(self._garch_params[symbol])
                    if self._garch_params.get(symbol) is not None
                    else None
                ),
                "sigma_sq_forecast": self._sigma_sq_forecast.get(symbol),
                "sigma_sq_used": self._last_sigma_sq_used.get(symbol),
                "bars_since_refit": int(self._bars_since_refit.get(symbol, 0)),
                "last_z": self._last_z.get(symbol),
                "last_return": self._last_return.get(symbol),
            }
            for symbol in self.symbol_list
        }
        return state

    def set_state(self, state: dict[str, Any]) -> None:
        super().set_state(state)
        raw = state.get("garch_state") if isinstance(state, dict) else None
        if not isinstance(raw, dict):
            return
        for symbol, payload in raw.items():
            if symbol not in self._returns or not isinstance(payload, dict):
                continue
            target = self._returns[symbol]
            target.clear()
            for value in list(payload.get("returns") or [])[-int(target.maxlen or 0) :]:
                parsed = safe_float(value)
                if parsed is not None:
                    target.append(parsed)
            params_raw = payload.get("params")
            if isinstance(params_raw, (list, tuple)) and len(params_raw) == 3:
                try:
                    self._garch_params[symbol] = (
                        float(params_raw[0]),
                        float(params_raw[1]),
                        float(params_raw[2]),
                    )
                except Exception:
                    self._garch_params[symbol] = None
            else:
                self._garch_params[symbol] = None
            self._sigma_sq_forecast[symbol] = safe_float(payload.get("sigma_sq_forecast"))
            self._last_sigma_sq_used[symbol] = safe_float(payload.get("sigma_sq_used"))
            self._bars_since_refit[symbol] = _safe_non_negative_int(payload.get("bars_since_refit"))
            self._last_z[symbol] = safe_float(payload.get("last_z"))
            self._last_return[symbol] = safe_float(payload.get("last_return"))


__all__ = [
    "GarchInnovationRiderStrategy",
]
