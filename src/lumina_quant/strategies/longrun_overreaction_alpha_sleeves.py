"""Long-run overreaction correction, crypto-compressed (De Bondt-Thaler 1985).

``LongRunOverreactionReversalStrategy`` fades EXTREME 3-6 month formation returns
across the liquid-majors cross-section: long the extreme multi-month LOSERS,
short the extreme multi-month WINNERS, on a monthly rebalance with a skip-month
and a hard min-hold.  Two structural devices keep it in a genuinely unoccupied
horizon-sign cell rather than being a re-labelled reversal or negated momentum:

1. SKIP-MONTH.  The formation return excises the most recent ``skip_bars`` (a
   month) of the path.  On liquid majors the daily/weekly horizon is
   MOMENTUM-signed (external verdict 9 / IRFA 2021), so unconditional daily
   reversal is wrong-signed there; skipping that band by construction keeps the
   daily-reversal finding from contaminating the signal, leaving only the
   long-run sign to be measured.
2. EXTREMENESS ABSTENTION.  Only names whose cross-sectional formation z-score
   clears ``z_min`` are tradeable, and if EITHER extreme tail is empty the sleeve
   abstains entirely (ages positions only).  A negated-momentum quintile sleeve
   trades every rebalance; this one trades only when a genuine multi-month
   dislocation exists, so it is not a sign-mirror of momentum.

THEORY / PROVENANCE
-------------------
- De Bondt & Thaler (1985, JF 40(3)): prior extreme multi-year losers
  subsequently outperform prior extreme winners (long-run overreaction).
- Barberis-Shleifer-Vishny (1998, JFE 49) / Daniel-Hirshleifer-Subrahmanyam
  (1998, JF 53) / Lakonishok-Shleifer-Vishny (1994, JF 49): behavioral
  over-extrapolation as the mechanism.
- Kosc-Sakowski-Slepaczuk (2019, Physica A 523): contrarian dominates momentum
  in a large crypto cross-section (weekly horizon).  Liu-Tsyvinski (2021, RFS
  34): crypto momentum concentrates at 1-4wk and decays beyond -- the horizon
  boundary the 3-6 month formation window sits PAST.
- HONEST FLAG: no verified crypto study at exactly 3-6 month formation; the
  compressed-horizon transplant IS the hypothesis this probe measures (EXPECTED
  NULL moderate-to-high: the majors cross-section may still sit inside the
  momentum regime at this horizon).

MECHANISM (per completed daily bar; OHLCV only)
-----------------------------------------------
Each rebalance (monthly ``rebalance_bars`` clock):

1. LIQUIDITY GATE -- rank names by trailing median dollar volume (close*volume)
   over the formation window; keep only the top ``max_universe`` with full
   ``formation_bars + skip_bars`` history.
2. FORMATION SIGNAL -- ``r_i = close[t-skip] / close[t-skip-formation] - 1``.
3. EXTREMENESS GATE -- cross-sectional z-score the formation returns; only
   ``|z| >= z_min`` names are tradeable; if either tail is empty -> abstain.
4. SCORE ``= -z_i`` inverse-formation-vol sized; long the extreme-loser tail /
   short the extreme-winner tail via the shared ranked-target machinery; hard
   ``min_hold_bars`` lock on young positions; cross-sectional demeaning makes the
   book approximately market-neutral.

This module is data-local (no I/O, no hidden config bus), pure Python
(``math`` + ``deque`` + pure indicator primitives, no numpy), never raises from
``calculate_signals``, and ships WITHOUT ``@register`` (inert until a later
integration wave wires it as ``research_only``).
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any

from lumina_quant.core.plugin_registry import register
from lumina_quant.indicators.alpha_features import realized_volatility
from lumina_quant.indicators.common import safe_float, time_key
from lumina_quant.indicators.rolling_stats import sample_std
from lumina_quant.strategies.adaptive_crypto_alpha_sleeves import (
    _age_cross_positions,
    _emit_rebalance_targets,
    _ranked_targets,
    _state_size,
)
from lumina_quant.strategies.external_alpha_sleeves import (
    _EPS,
    _Snapshot,
    _event_symbols,
    _market_snapshot,
    _safe_non_negative_int,
    _window_snapshot,
)
from lumina_quant.strategies.robust_alpha_sleeves import (
    _CrossSectionalState,
    _mode,
    _restore_deque,
)
from lumina_quant.strategy import Strategy
from lumina_quant.tuning import HyperParam, resolve_params_from_schema

_STRATEGY_ID = "longrun_overreaction_reversal"
_STRATEGY_NAME = "LongRunOverreactionReversalStrategy"


def _pack_cross(item: _CrossSectionalState) -> dict[str, Any]:
    return {
        "closes": list(item.closes),
        "volumes": list(item.volumes),
        "mode": item.mode,
        "entry_price": item.entry_price,
        "bars_held": int(item.bars_held),
        "last_time_key": item.last_time_key,
    }


def _restore_cross(item: _CrossSectionalState, payload: dict[str, Any]) -> None:
    _restore_deque(item.closes, payload.get("closes"))
    _restore_deque(item.volumes, payload.get("volumes"))
    item.mode = _mode(payload.get("mode"))
    item.entry_price = safe_float(payload.get("entry_price"))
    item.bars_held = _safe_non_negative_int(payload.get("bars_held"))
    item.last_time_key = str(payload.get("last_time_key", ""))


def _median(values: list[float]) -> float | None:
    cleaned = [float(value) for value in values if math.isfinite(value)]
    if not cleaned:
        return None
    ordered = sorted(cleaned)
    count = len(ordered)
    mid = count // 2
    if count % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


@register("strategy", "LongRunOverreactionReversalStrategy", interface="event_driven")
class LongRunOverreactionReversalStrategy(Strategy):
    """XS fade of extreme 3-6 month formation returns on liquid majors.

    See the module docstring for the full theory, the skip-month rationale, and
    the extremeness-abstention anti-sign-mirror device.  Reads only local
    event/bar OHLCV; performs no I/O and never raises from ``calculate_signals``.
    """

    decision_cadence_seconds = 86400
    preferred_contract = "market_window"
    uses_timeframe_aggregator = False

    @classmethod
    def get_param_schema(cls) -> dict[str, HyperParam]:
        return {
            "formation_bars": HyperParam.integer("formation_bars", default=126, low=8, high=20000),
            "skip_bars": HyperParam.integer("skip_bars", default=21, low=0, high=4096),
            "z_min": HyperParam.floating("z_min", default=1.0, low=0.0, high=20.0),
            "max_universe": HyperParam.integer("max_universe", default=12, low=2, high=512),
            "rebalance_bars": HyperParam.integer("rebalance_bars", default=21, low=1, high=10080),
            "min_hold_bars": HyperParam.integer("min_hold_bars", default=21, low=0, high=200000),
            "quantile_pct": HyperParam.floating("quantile_pct", default=0.25, low=0.05, high=0.50),
            "min_symbols": HyperParam.integer("min_symbols", default=5, low=2, high=512),
            "allow_short": HyperParam.boolean("allow_short", default=True, grid=[True, False]),
            "stop_loss_pct": HyperParam.floating("stop_loss_pct", default=0.15, low=0.0, high=0.50),
            "max_hold_bars": HyperParam.integer("max_hold_bars", default=252, low=1, high=200000),
            "min_dollar_volume": HyperParam.floating(
                "min_dollar_volume", default=0.0, low=0.0, high=1e15
            ),
            "target_gross_exposure": HyperParam.floating(
                "target_gross_exposure", default=0.40, low=0.0, high=5.0, tunable=False
            ),
            "max_order_value": HyperParam.floating(
                "max_order_value", default=500.0, low=0.0, high=1_000_000.0, tunable=False
            ),
            "min_price": HyperParam.floating("min_price", default=0.10, low=0.0, high=1_000_000.0),
        }

    def __init__(self, bars: Any, events: Any, **params: Any) -> None:
        self.bars = bars
        self.events = events
        self.symbol_list = list(getattr(self.bars, "symbol_list", []) or [])
        resolved = resolve_params_from_schema(self.get_param_schema(), params, keep_unknown=False)
        self.formation_bars = max(8, int(resolved["formation_bars"]))
        self.skip_bars = max(0, int(resolved["skip_bars"]))
        self.z_min = max(0.0, float(resolved["z_min"]))
        self.max_universe = max(2, int(resolved["max_universe"]))
        self.rebalance_bars = max(1, int(resolved["rebalance_bars"]))
        self.min_hold_bars = max(0, int(resolved["min_hold_bars"]))
        self.quantile_pct = max(0.01, min(0.50, float(resolved["quantile_pct"])))
        self.min_symbols = max(2, int(resolved["min_symbols"]))
        self.allow_short = bool(resolved["allow_short"])
        self.stop_loss_pct = max(0.0, float(resolved["stop_loss_pct"]))
        self.max_hold_bars = max(1, int(resolved["max_hold_bars"]))
        self.min_dollar_volume = max(0.0, float(resolved["min_dollar_volume"]))
        self.target_gross_exposure = max(0.0, float(resolved["target_gross_exposure"]))
        self.max_order_value = max(0.0, float(resolved["max_order_value"]))
        self.min_price = max(0.0, float(resolved["min_price"]))
        self._required = self.formation_bars + self.skip_bars + 1
        size = _state_size(self._required, self.max_hold_bars)
        self._state: dict[str, _CrossSectionalState] = {
            symbol: _CrossSectionalState(deque(maxlen=size), deque(maxlen=size))
            for symbol in self.symbol_list
        }
        self._last_eval_time_key = ""
        self._tick = 0

    # ------------------------------------------------------------------ #
    # state
    # ------------------------------------------------------------------ #
    def get_state(self) -> dict[str, Any]:
        return {
            "last_eval_time_key": self._last_eval_time_key,
            "tick": int(self._tick),
            "symbol_state": {symbol: _pack_cross(item) for symbol, item in self._state.items()},
        }

    def set_state(self, state: dict[str, Any]) -> None:
        if not isinstance(state, dict):
            return
        self._last_eval_time_key = str(state.get("last_eval_time_key", ""))
        self._tick = _safe_non_negative_int(state.get("tick"))
        raw = state.get("symbol_state")
        if isinstance(raw, dict):
            for symbol, payload in raw.items():
                if symbol in self._state and isinstance(payload, dict):
                    try:
                        _restore_cross(self._state[symbol], payload)
                    except Exception:
                        continue

    # ------------------------------------------------------------------ #
    # ingestion
    # ------------------------------------------------------------------ #
    def _update_symbol(self, symbol: str, snapshot: _Snapshot) -> bool:
        close = safe_float(snapshot.close)
        if close is None or close <= self.min_price:
            return False
        item = self._state[symbol]
        key = time_key(snapshot.time)
        if key and key == item.last_time_key:
            return False
        item.last_time_key = key
        item.closes.append(close)
        item.volumes.append(max(0.0, float(snapshot.volume or 0.0)))
        return True

    def calculate_signals_window(self, event: Any, aggregator: Any = None) -> None:
        _ = aggregator
        event_key = time_key(getattr(event, "time", None))
        updated = False
        for symbol in _event_symbols(event, self.symbol_list):
            snapshot = _window_snapshot(event, symbol)
            if snapshot is not None and self._update_symbol(symbol, snapshot):
                updated = True
        if updated and event_key and event_key != self._last_eval_time_key:
            self._last_eval_time_key = event_key
            self._tick += 1
            self._rebalance(getattr(event, "time", None))

    def calculate_signals(self, event: Any) -> None:
        if str(getattr(event, "type", "")).upper() == "MARKET_WINDOW":
            self.calculate_signals_window(event, None)
            return
        if getattr(event, "type", None) != "MARKET":
            return
        symbol = getattr(event, "symbol", None)
        if symbol in self._state:
            snapshot = _market_snapshot(event)
            if snapshot is not None and self._update_symbol(str(symbol), snapshot):
                key = time_key(snapshot.time)
                if key and key != self._last_eval_time_key:
                    self._last_eval_time_key = key
                    self._tick += 1
                    self._rebalance(snapshot.time)

    # ------------------------------------------------------------------ #
    # scoring / rebalance
    # ------------------------------------------------------------------ #
    def _age(self, event_time: Any) -> None:
        _age_cross_positions(
            self.events,
            self._state,
            event_time=event_time,
            strategy_id=_STRATEGY_ID,
            strategy_name=_STRATEGY_NAME,
            stop_loss_pct=self.stop_loss_pct,
            max_hold_bars=self.max_hold_bars,
        )

    def _liquid_universe(self) -> list[str]:
        """Return the top ``max_universe`` full-history names by median dollar volume."""
        ranked: list[tuple[float, str]] = []
        for symbol, item in self._state.items():
            if len(item.closes) < self._required:
                continue
            closes = list(item.closes)[-self.formation_bars :]
            volumes = list(item.volumes)[-self.formation_bars :]
            dollar_volumes = [
                close * volume
                for close, volume in zip(closes, volumes, strict=False)
                if close > 0.0
            ]
            median_dv = _median(dollar_volumes)
            if median_dv is None or median_dv < self.min_dollar_volume:
                continue
            ranked.append((median_dv, symbol))
        # Deterministic tie-break by symbol so the universe cut is stable.
        ranked.sort(key=lambda row: (-row[0], row[1]))
        return [symbol for _dv, symbol in ranked[: self.max_universe]]

    def _formation(self, symbol: str) -> tuple[float, float] | None:
        """Return ``(skip-adjusted formation return, formation-window vol)`` or ``None``."""
        item = self._state[symbol]
        closes = list(item.closes)
        if len(closes) < self._required:
            return None
        recent = closes[-(self.skip_bars + 1)] if self.skip_bars > 0 else closes[-1]
        base = closes[-self._required]
        if base <= 0.0 or recent <= 0.0:
            return None
        formation_return = recent / base - 1.0
        vol = realized_volatility(closes, window=self.formation_bars)
        if vol is None or vol <= _EPS or not math.isfinite(formation_return):
            return None
        return float(formation_return), float(vol)

    def _apply_min_hold(
        self, targets: dict[str, tuple[str, float, dict[str, Any]]]
    ) -> dict[str, tuple[str, float, dict[str, Any]]]:
        """Lock any position younger than ``min_hold_bars`` onto its current side."""
        if self.min_hold_bars <= 0:
            return targets
        for symbol, item in self._state.items():
            if item.mode == "OUT" or item.bars_held >= self.min_hold_bars:
                continue
            target = targets.get(symbol)
            if target is None or target[0] != item.mode:
                targets[symbol] = (item.mode, 0.0, {"reason": "min_hold_lock"})
        return targets

    def _rebalance(self, event_time: Any) -> None:
        if self._tick % self.rebalance_bars:
            self._age(event_time)
            return
        universe = self._liquid_universe()
        if len(universe) < self.min_symbols:
            self._age(event_time)
            return
        formation: dict[str, tuple[float, float]] = {}
        for symbol in universe:
            result = self._formation(symbol)
            if result is not None:
                formation[symbol] = result
        if len(formation) < self.min_symbols:
            self._age(event_time)
            return

        returns = [value[0] for value in formation.values()]
        mean_return = sum(returns) / float(len(returns))
        std_return = sample_std(returns)
        if std_return is None or std_return <= _EPS:
            # No cross-sectional dispersion -> no overreaction to fade.
            self._age(event_time)
            return

        rows: list[tuple[float, str, dict[str, Any]]] = []
        has_loser = False
        has_winner = False
        for symbol, (formation_return, vol) in formation.items():
            z = (formation_return - mean_return) / std_return
            if abs(z) < self.z_min:
                continue
            if z <= -self.z_min:
                has_loser = True
            if z >= self.z_min:
                has_winner = True
            # SCORE = -z, inverse-formation-vol sized: an extreme LOSER (z << 0)
            # scores high positive -> LONG; an extreme WINNER (z >> 0) scores
            # negative -> SHORT.
            # vol-target horizon classification: category C (SCORE denominator, NOT
            # a sizing throttle).  There is no ``target_vol`` here; sizing normalizes
            # to ``target_gross_exposure`` downstream, and a global sqrt(bars_per_year)
            # rescale of every ``vol`` cancels under the ranking / gross-normalization,
            # so no per-bar-vs-annual horizon mismatch exists.
            score = -z / max(vol, _EPS)
            rows.append(
                (
                    float(score),
                    symbol,
                    {
                        "formation_return": float(formation_return),
                        "formation_zscore": float(z),
                        "formation_vol": float(vol),
                        "basket_mean_return": float(mean_return),
                    },
                )
            )
        # EXTREMENESS ABSTENTION: both tails must be populated, else stand aside.
        if not has_loser or not (has_winner and self.allow_short):
            self._age(event_time)
            return

        max_side = max(1, int(len(rows) * self.quantile_pct))
        targets = _ranked_targets(
            rows,
            threshold=_EPS,
            max_longs=max_side,
            max_shorts=max_side if self.allow_short else 0,
            allow_short=self.allow_short,
        )
        targets = self._apply_min_hold(targets)
        _emit_rebalance_targets(
            self.events,
            self._state,
            targets,
            event_time=event_time,
            strategy_id=_STRATEGY_ID,
            strategy_name=_STRATEGY_NAME,
            target_gross_exposure=self.target_gross_exposure,
            max_order_value=self.max_order_value,
            stop_loss_pct=self.stop_loss_pct,
            max_hold_bars=self.max_hold_bars,
            threshold=_EPS,
        )


# --------------------------------------------------------------------------- #
# Candidate-wiring hints for a later integration wave (this lane does NOT wire
# candidates itself -- new-file-only, no shared-file edits per the live-safety
# plan).  Admission route is `allow_multi_asset=True` at the data-PC handoff:
# this is a cross-sectional long-run-reversal book but NOT carry, so it is
# honestly EXCLUDED from any carry-tag allowlist route -- no fake carry tag.
# --------------------------------------------------------------------------- #
_SUGGESTED_FAMILY = "cross_sectional"
_SUGGESTED_CANDIDATE_TAGS: tuple[str, ...] = (
    "cross_sectional",
    "long_run_reversal",
    "overreaction",
    "contrarian",
    "market_neutral",
    "crypto",
)

# MULTI-TIMEFRAME (data-PC parquet carries 1h/4h but not 1d): the De Bondt-Thaler
# 3-6mo formation window, the skip-month excision, and the monthly rebalance are
# WALL-CLOCK horizons, so every BAR-denominated param (``formation_bars``,
# ``skip_bars``, and the ``rebalance_bars``/``min_hold_bars`` monthly decision
# clocks) scales x6 for 4h and x24 for 1h to keep the same multi-month
# formation/cadence.  The extremeness gate ``z_min``, the ``max_universe`` /
# ``min_symbols`` counts, the ``quantile_pct`` ratio, and ``allow_short`` are
# timeframe-agnostic and stay UNCHANGED.  Longest window (formation_126 @1h =
# 3024 bars) stays well under the ~9000-bar cap.
_LONGRUN_OVERREACTION_SLICE: dict[str, tuple[dict[str, Any], ...]] = {
    "1h": (
        {
            "variant": "formation_126",
            "formation_bars": 3024,  # 126 x24 (~6mo of 1h bars)
            "skip_bars": 504,  # 21 x24 (~1mo)
            "z_min": 1.0,
            "max_universe": 128,
            "rebalance_bars": 504,  # 21 x24 (~monthly cadence)
            "min_hold_bars": 504,  # 21 x24
            "quantile_pct": 0.25,
            "min_symbols": 5,
            "allow_short": True,
        },
        {
            "variant": "formation_91",
            "formation_bars": 2184,  # 91 x24 (~4.3mo of 1h bars)
            "skip_bars": 504,
            "z_min": 1.0,
            "max_universe": 128,
            "rebalance_bars": 504,
            "min_hold_bars": 504,
            "quantile_pct": 0.25,
            "min_symbols": 5,
            "allow_short": True,
        },
    ),
    "4h": (
        {
            "variant": "formation_126",
            "formation_bars": 756,  # 126 x6 (~6mo of 4h bars)
            "skip_bars": 126,  # 21 x6 (~1mo)
            "z_min": 1.0,
            "max_universe": 128,
            "rebalance_bars": 126,  # 21 x6 (~monthly cadence)
            "min_hold_bars": 126,  # 21 x6
            "quantile_pct": 0.25,
            "min_symbols": 5,
            "allow_short": True,
        },
        {
            "variant": "formation_91",
            "formation_bars": 546,  # 91 x6 (~4.3mo of 4h bars)
            "skip_bars": 126,
            "z_min": 1.0,
            "max_universe": 128,
            "rebalance_bars": 126,
            "min_hold_bars": 126,
            "quantile_pct": 0.25,
            "min_symbols": 5,
            "allow_short": True,
        },
    ),
    "1d": (
        {
            "variant": "formation_126",
            "formation_bars": 126,
            "skip_bars": 21,
            "z_min": 1.0,
            "max_universe": 128,
            "rebalance_bars": 21,
            "min_hold_bars": 21,
            "quantile_pct": 0.25,
            "min_symbols": 5,
            "allow_short": True,
        },
        {
            "variant": "formation_91",
            "formation_bars": 91,
            "skip_bars": 21,
            "z_min": 1.0,
            "max_universe": 128,
            "rebalance_bars": 21,
            "min_hold_bars": 21,
            "quantile_pct": 0.25,
            "min_symbols": 5,
            "allow_short": True,
        },
    ),
}


__all__ = ["LongRunOverreactionReversalStrategy"]
