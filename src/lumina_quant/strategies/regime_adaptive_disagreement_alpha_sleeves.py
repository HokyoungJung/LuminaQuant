"""Regime-adaptive disagreement gate over the four OHLCV-only components (Lane MR2).

THESIS (variance-reduction / drawdown-smoothing -- NOT a Sharpe-lift claim).
This is a STANDALONE variant of :class:`DisagreementGatedEnsembleStrategy`
(Lane M1) whose only novelty is that the disagreement admission gate is no
longer a fixed constant but is MODULATED by the prevailing volatility regime.
The replication-ratio literature (arXiv 2501.03938 / 2512.12735) and
McLean & Pontiff (2016) caution that an ensemble/combination rarely lifts net
Sharpe ABOVE the best single member out of sample; the honest, measurable
promise of a disagreement gate is instead VARIANCE-REDUCTION and
DRAWDOWN-SMOOTHING (report Calmar / max-drawdown, never "beats the best
sleeve's Sharpe").  A regime-adaptive gate targets exactly that objective:
it is deliberately built to change WHEN the ensemble stands aside as a
function of the regime, so its data-PC evaluation is a MDD/Calmar comparison
against the fixed-gate base, not a Sharpe race.

REGIME DEFINITION (the cleanest deterministic dispersion measure).  "Dispersion
regime" here is the realized-VOLATILITY regime of the symbol's own OHLCV close
stream -- the regime notion of Ang & Bekaert (2002).  It is measured with the
existing pure primitive :func:`~lumina_quant.indicators.alpha_features.
volatility_ratio` = fast-window realized vol / slow-window realized vol.  This
ratio is SCALE-FREE (self-normalizing across symbols and timeframes -- no magic
absolute vol constant) and is centered at ``1.0`` when recent volatility equals
its own trailing baseline.  ``> 1`` is a TURBULENT (high-dispersion) regime
where recent volatility has risen above its baseline; ``< 1`` is a CALM
(low-dispersion) regime where it has fallen below.  It is decoupled from the
INSTANTANEOUS cross-component disagreement coefficient the base sleeve gates on
(a return-magnitude measure versus a directional-agreement measure), so the
modulation is a genuine second signal rather than a restatement of the gate.

GATE MODULATION (monotone; base-identical at regime ratio 1).  The effective
gate is::

    effective_gate = disagreement_gate * clamp(
        1 + gate_sensitivity * (regime_ratio - 1),
        min_gate_factor, max_gate_factor)

so the gate WIDENS (tolerates more disagreement, stands aside less) in a
high-dispersion regime and TIGHTENS (demands more consensus, stands aside more)
in a calm regime -- monotone non-decreasing in ``regime_ratio``, and exactly
the base sleeve's fixed ``disagreement_gate`` when ``regime_ratio == 1`` (or
when the ratio is unavailable on short history: the multiplier falls back to
``1.0``, never-raise).  The economic reading matched to the variance-reduction
thesis: in a CALM regime a marginal, barely-consensual entry mostly adds
whipsaw variance, so the gate tightens it away; in a TURBULENT regime standing
fully aside forfeits the ensemble's diversification/averaging benefit exactly
when drawdowns cluster, so the gate widens to keep the combined view engaged.

Everything else -- the four internal directional components, the inverse-error
adaptive weights, the causal score/error bookkeeping, the composite/entry-band
decision, the ATR trailing-stop ride with pyramiding and vol-targeted sizing,
and all state (de)serialization -- is inherited UNCHANGED from
:class:`DisagreementGatedEnsembleStrategy`; this module NEVER edits that class in
place.  The variant adds no new mutable per-symbol state (the regime ratio is
derived on demand from the inherited close ring buffer), so state round-trips
through the inherited ``get_state``/``set_state`` untouched.

OHLCV-only, per-symbol single-asset, pure Python (no numpy/scipy), never-raise
graceful guards throughout, ``decision_cadence_seconds >= 1800`` (inherited
>=30m floor).  Data-local: no fetching, no artifact writes, no hidden
environment state.  Ships WITHOUT ``@register`` (live-safety: registration and
the research_only tier hint are applied atomically in a later, separate wave).
Must still be validated on the data-bearing machine.
"""

from __future__ import annotations

import math
from typing import Any

from lumina_quant.indicators.alpha_features import volatility_ratio
from lumina_quant.strategies.disagreement_ensemble_alpha_sleeves import (
    DisagreementGatedEnsembleStrategy,
)
from lumina_quant.tuning import HyperParam


class RegimeAdaptiveDisagreementEnsembleStrategy(DisagreementGatedEnsembleStrategy):
    """Disagreement-gated ensemble whose gate widens in turbulent regimes.

    A standalone variant of :class:`DisagreementGatedEnsembleStrategy`: identical
    components/weights/ride machinery, but the disagreement admission gate is
    scaled by the realized-volatility regime ratio (fast/slow realized vol) so
    it tolerates more disagreement when volatility is elevated and demands more
    consensus when it is calm.  Thesis is variance-reduction / drawdown-smoothing
    (Calmar / MDD), NOT a Sharpe-lift over the best single sleeve.  See the module
    docstring.
    """

    strategy_name = "RegimeAdaptiveDisagreementEnsembleStrategy"
    strategy_id = "regime_adaptive_disagreement_ensemble"

    @classmethod
    def get_param_schema(cls) -> dict[str, HyperParam]:
        schema = dict(super().get_param_schema())
        schema.update(
            {
                "regime_fast_window": HyperParam.integer(
                    "regime_fast_window", default=16, low=2, high=4096
                ),
                "regime_slow_window": HyperParam.integer(
                    "regime_slow_window", default=96, low=3, high=8192
                ),
                "gate_sensitivity": HyperParam.floating(
                    "gate_sensitivity", default=1.0, low=0.0, high=20.0
                ),
                "min_gate_factor": HyperParam.floating(
                    "min_gate_factor", default=0.25, low=0.0, high=1.0
                ),
                "max_gate_factor": HyperParam.floating(
                    "max_gate_factor", default=4.0, low=1.0, high=100.0
                ),
            }
        )
        return schema

    def _bind_params(self, resolved: dict[str, Any]) -> None:
        super()._bind_params(resolved)
        self.regime_fast_window = max(2, int(resolved["regime_fast_window"]))
        # slow window must strictly exceed fast so the ratio contrasts two
        # horizons; clamp defensively rather than trusting the resolved value.
        self.regime_slow_window = max(
            self.regime_fast_window + 1, int(resolved["regime_slow_window"])
        )
        self.gate_sensitivity = max(0.0, float(resolved["gate_sensitivity"]))
        self.min_gate_factor = max(0.0, float(resolved["min_gate_factor"]))
        # keep the clamp interval valid (max >= min) even under adversarial params.
        self.max_gate_factor = max(self.min_gate_factor, float(resolved["max_gate_factor"]))

    def _common_config(self, resolved: dict[str, Any]) -> Any:
        # Extend the base's history buffer so the slow realized-vol window
        # (regime_slow_window + 1 returns) always has enough closes.
        return self._resolve_common(
            resolved,
            extra_window=max(
                self.tsmom_window + self.tsmom_lookback,
                self.reversion_window,
                self.donchian_window,
                self.efficiency_period,
                self.regime_slow_window + 1,
            )
            + 2,
        )

    # -- regime-adaptive gate --------------------------------------------------
    def _regime_gate_multiplier(self, regime_ratio: float | None) -> float:
        """Return the monotone gate multiplier for a realized-vol regime ratio.

        ``1.0`` at ``regime_ratio == 1`` (base-identical) and when the ratio is
        unavailable (``None``/non-finite -- the never-raise short-history read);
        rising above ``1`` for turbulent regimes (widen) and falling below for
        calm regimes (tighten), clamped to ``[min_gate_factor, max_gate_factor]``.
        Non-decreasing in ``regime_ratio``.
        """
        if regime_ratio is None or not math.isfinite(regime_ratio):
            return 1.0
        raw = 1.0 + self.gate_sensitivity * (regime_ratio - 1.0)
        return max(self.min_gate_factor, min(self.max_gate_factor, raw))

    def _regime_ratio(self, symbol: str) -> float | None:
        """Return the fast/slow realized-vol ratio from the symbol's closes, or ``None``."""
        item = self._state.get(symbol)
        if item is None:
            return None
        return volatility_ratio(
            list(item.closes),
            fast_window=self.regime_fast_window,
            slow_window=self.regime_slow_window,
        )

    def _effective_disagreement_gate(self, symbol: str) -> float:
        """Return the base disagreement gate scaled by the current volatility regime."""
        return self.disagreement_gate * self._regime_gate_multiplier(self._regime_ratio(symbol))

    def _gated_direction(self, symbol: str) -> str:
        """Return ``"LONG"``/``"SHORT"``/``""`` using the regime-adaptive gate.

        Mirrors the base sleeve's decision but compares the instantaneous
        cross-component disagreement against the REGIME-ADAPTIVE effective gate
        instead of the fixed ``disagreement_gate``.
        """
        state = self._decision_state(symbol)
        if state is None:
            return ""
        disagreement = state["disagreement"]
        gate = self._effective_disagreement_gate(symbol)
        # None disagreement means an undefined coefficient (near-zero consensus
        # mean) -- treated as maximal disagreement, i.e. blocked (conservative).
        if disagreement is None or disagreement > gate:
            return ""
        composite = state["composite"]
        if composite >= self.entry_band:
            return "LONG"
        if composite <= -self.entry_band:
            return "SHORT"
        return ""

    def _entry_metadata(self, item: Any) -> dict[str, Any]:
        metadata = super()._entry_metadata(item)
        symbol = self._symbol_for(item)
        if symbol is not None:
            ratio = self._regime_ratio(symbol)
            metadata["regime_ratio"] = float(ratio) if ratio is not None else None
            metadata["effective_gate"] = float(self._effective_disagreement_gate(symbol))
        else:
            metadata["regime_ratio"] = None
            metadata["effective_gate"] = None
        return metadata


__all__ = [
    "RegimeAdaptiveDisagreementEnsembleStrategy",
]
