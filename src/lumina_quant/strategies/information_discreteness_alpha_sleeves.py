"""Information-discreteness CONDITIONED cross-sectional momentum sleeve (FIP).

``InformationDiscretenessMomentumStrategy`` conditions a deliberately vanilla
cross-sectional momentum book on the Da, Gurun & Warachka (2014, RFS) "Frog in
the Pan" information-discreteness statistic -- a magnitude-BLIND sign census of
the formation-window daily returns.  Names whose formation path arrived in
CONTINUOUS small same-direction doses (strongly negative ID) are held to be
UNDER-reacted (limited attention under-prices slow information); names whose
path is DISCRETE / jump-driven (ID near zero or positive) are salient and
already priced, so they are EXCLUDED FROM BOTH BOOKS regardless of momentum.

Among the most-continuous fraction of the cross-section the sleeve then takes a
plain momentum book: LONG the top formation-return quantile, SHORT the bottom.
The momentum leg is intentionally vanilla so the conditioning -- not a bespoke
momentum transform -- is what the orthogonality gate isolates.

THEORY / PROVENANCE
-------------------
- Da, Gurun & Warachka (2014, RFS 27(7) 2171-2218): continuous information
  diffuses under limited attention, so momentum is stronger and more persistent
  for continuous-information (low-ID) names; the sign census ``ID`` is the
  magnitude-blind measure of that continuity.
- Hong & Stein (1999); Hirshleifer & Teoh (2003) attention as a
  capacity-constrained resource; Grinblatt & Moskowitz (2004) sign-census
  lineage.
- HONEST FLAGS: no crypto-specific FIP study (an equities transplant); crypto XS
  momentum is SUPPORTED-but-DECAYING.  This lane is a falsification probe with a
  high stated expected null, not a promise of edge.

The load-bearing distinction from every incumbent: ID is a SIGN census -- it is
magnitude-blind.  Kaufman efficiency is net-move / path-length (magnitude), the
signed-R^2 trend-quality fit is magnitude-weighted, and plain momentum / vol are
magnitude-only.  On a flat-drift-plus-single-jump path the momentum / efficiency
/ fit incumbents all treat the name as a clean strong trend, while the sign
census reads it as DISCRETE and excludes it -- the divergence the build gate
pins with the real incumbents.

SIGNAL SPEC
-----------
Per completed WEEKLY ISO decision bar (internal ``_week_key`` clock), per symbol
with >= ``min_history_bars`` closes:

1. ``ID = information_discreteness(closes, formation_bars, skip_bars)`` and the
   formation return ``PRET`` over the SAME window (excluding the ``skip_bars``
   skip-week).
2. ELIGIBILITY: only the most-continuous ``continuity_pct`` fraction of the
   cross-section by ID rank, AND ``ID <= id_max`` -- discrete names are dropped.
3. Among eligibles: LONG the top ``quantile_pct`` by ``PRET``, SHORT the bottom,
   with a ``rank_hysteresis_buffer`` retention band, a hard
   ``min_hold_decisions`` minimum hold, a post-exit ``cooldown_decisions``
   window, and a ``max_hold_decisions`` backstop -- a rank flip inside the hold
   window is suppressed (the proven low-turnover rescue, weekly decision bars).
4. Inverse-realized-vol risk-parity sizing normalized to
   ``target_gross_exposure``.  Self-skips when fewer than ``min_symbols`` names
   carry a valid ID + PRET + vol history.

This module is data-local (no I/O, no hidden configuration bus), pure Python in
the strategy layer (the sign-census numeric lives in
``indicators/information_discreteness.py``), completed-bar, and never raises from
``calculate_signals``.  It ships WITHOUT ``@register`` (inert): registration,
tier hint, and candidate wiring land atomically in the integration wave.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

from lumina_quant.indicators.alpha_features import realized_volatility
from lumina_quant.indicators.common import safe_float, time_key
from lumina_quant.indicators.information_discreteness import information_discreteness
from lumina_quant.strategies.external_alpha_sleeves import (
    _EPS,
    _Snapshot,
    _emit,
    _event_datetime_utc,
    _event_symbols,
    _market_snapshot,
    _safe_non_negative_int,
    _target_metadata,
    _window_snapshot,
)
from lumina_quant.strategy import Strategy
from lumina_quant.tuning import HyperParam, resolve_params_from_schema

_STRATEGY_ID = "information_discreteness_momentum"
_STRATEGY_NAME = "InformationDiscretenessMomentumStrategy"

# A fresh symbol has never traded, so its cooldown is treated as long elapsed --
# the first eligible entry must not be blocked by a phantom startup cooldown.
_COOLDOWN_SATISFIED = 1 << 30


@dataclass(slots=True)
class _State:
    """Per-symbol trailing closes + position / min-hold / cooldown bookkeeping."""

    closes: deque[float]
    mode: str = "OUT"  # OUT / LONG / SHORT
    entry_price: float | None = None
    bars_held: int = 0  # weekly decision bars in the CURRENT position
    bars_since_exit: int = _COOLDOWN_SATISFIED  # weekly decision bars since last exit
    last_bar_key: str = ""  # dedup identical ingested bars
    score: float | None = None


def _coerce_float_list(value: Any) -> list[float]:
    """Best-effort ``list[float]`` coercion that never raises on adversarial input."""
    if not isinstance(value, (list, tuple)):
        return []
    out: list[float] = []
    for item in value:
        parsed = safe_float(item)
        if parsed is not None:
            out.append(parsed)
    return out


def _mode(raw: Any) -> str:
    """Coerce a serialized mode token to one of ``{OUT, LONG, SHORT}``."""
    parsed = str(raw or "OUT").upper()
    return parsed if parsed in {"OUT", "LONG", "SHORT"} else "OUT"


def _formation_return(closes: list[float], formation_bars: int, skip_bars: int) -> float | None:
    """Return the formation-window cumulative return (excluding the skip window)."""
    work = closes[: len(closes) - skip_bars] if skip_bars > 0 else closes
    if len(work) < formation_bars + 1:
        return None
    window = work[-(formation_bars + 1) :]
    base = window[0]
    if base is None or base <= 0.0:
        return None
    return (window[-1] / base) - 1.0


class InformationDiscretenessMomentumStrategy(Strategy):
    """Weekly XS momentum conditioned on the frog-in-the-pan sign census.

    See the module docstring for the full theory, signal spec, and the
    load-bearing distinction from the momentum / efficiency / fit incumbents.
    This class only reads local event/bar closes; it performs no I/O and never
    raises from ``calculate_signals``.
    """

    decision_cadence_seconds = 86400  # daily bars; weekly effective via _week_key
    preferred_contract = "market_window"
    uses_timeframe_aggregator = False

    strategy_name = _STRATEGY_NAME
    strategy_id = _STRATEGY_ID

    @classmethod
    def get_param_schema(cls) -> dict[str, HyperParam]:
        return {
            "formation_bars": HyperParam.integer("formation_bars", default=56, low=8, high=4096),
            "skip_bars": HyperParam.integer("skip_bars", default=7, low=0, high=512),
            "zero_eps": HyperParam.floating(
                "zero_eps", default=1e-6, low=0.0, high=0.10, tunable=False
            ),
            "continuity_pct": HyperParam.floating(
                "continuity_pct", default=0.50, low=0.10, high=1.0
            ),
            "id_max": HyperParam.floating("id_max", default=0.0, low=-1.0, high=1.0),
            "quantile_pct": HyperParam.floating("quantile_pct", default=0.25, low=0.05, high=0.50),
            "vol_window": HyperParam.integer("vol_window", default=30, low=2, high=4096),
            "min_history_bars": HyperParam.integer(
                "min_history_bars", default=70, low=9, high=100000
            ),
            "min_symbols": HyperParam.integer("min_symbols", default=6, low=3, high=512),
            "min_hold_decisions": HyperParam.integer(
                "min_hold_decisions", default=4, low=1, high=200000
            ),
            "cooldown_decisions": HyperParam.integer(
                "cooldown_decisions", default=1, low=0, high=200000
            ),
            "max_hold_decisions": HyperParam.integer(
                "max_hold_decisions", default=52, low=1, high=1000000
            ),
            "rank_hysteresis_buffer": HyperParam.integer(
                "rank_hysteresis_buffer", default=1, low=0, high=512
            ),
            "allow_short": HyperParam.boolean("allow_short", default=True, grid=[True, False]),
            "target_gross_exposure": HyperParam.floating(
                "target_gross_exposure", default=1.0, low=0.0, high=3.0
            ),
            "base_allocation": HyperParam.floating(
                "base_allocation", default=0.20, low=0.0, high=2.0, tunable=False
            ),
            "max_symbol_exposure_pct": HyperParam.floating(
                "max_symbol_exposure_pct", default=0.40, low=0.0, high=2.0, tunable=False
            ),
            "max_order_value": HyperParam.floating(
                "max_order_value", default=400.0, low=0.0, high=1_000_000.0, tunable=False
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
        self.zero_eps = max(0.0, float(resolved["zero_eps"]))
        self.continuity_pct = min(1.0, max(0.05, float(resolved["continuity_pct"])))
        self.id_max = max(-1.0, min(1.0, float(resolved["id_max"])))
        self.quantile_pct = min(0.50, max(0.01, float(resolved["quantile_pct"])))
        self.vol_window = max(2, int(resolved["vol_window"]))
        self.min_symbols = max(3, int(resolved["min_symbols"]))
        self.min_hold_decisions = max(1, int(resolved["min_hold_decisions"]))
        self.cooldown_decisions = max(0, int(resolved["cooldown_decisions"]))
        self.max_hold_decisions = max(self.min_hold_decisions, int(resolved["max_hold_decisions"]))
        self.rank_hysteresis_buffer = max(0, int(resolved["rank_hysteresis_buffer"]))
        self.allow_short = bool(resolved["allow_short"])
        self.target_gross_exposure = max(0.0, float(resolved["target_gross_exposure"]))
        self.base_allocation = max(0.0, float(resolved["base_allocation"]))
        self.max_symbol_exposure_pct = max(0.0, float(resolved["max_symbol_exposure_pct"]))
        self.max_order_value = max(0.0, float(resolved["max_order_value"]))
        self.min_price = max(0.0, float(resolved["min_price"]))
        # Closes needed for the formation window (+ skip) and a valid vol reading.
        self._min_history = max(
            int(resolved["min_history_bars"]),
            self.formation_bars + self.skip_bars + 1,
            self.vol_window + 1,
        )
        history = max(self.formation_bars + self.skip_bars, self.vol_window) + 8
        self._state: dict[str, _State] = {
            symbol: _State(closes=deque(maxlen=history)) for symbol in self.symbol_list
        }
        self._last_decision_week = ""
        self._tick = 0

    # ------------------------------------------------------------------ #
    # state
    # ------------------------------------------------------------------ #
    def get_state(self) -> dict[str, Any]:
        return {
            "last_decision_week": self._last_decision_week,
            "tick": int(self._tick),
            "symbol_state": {
                symbol: {
                    "closes": list(item.closes),
                    "mode": item.mode,
                    "entry_price": item.entry_price,
                    "bars_held": int(item.bars_held),
                    "bars_since_exit": int(item.bars_since_exit),
                    "last_bar_key": item.last_bar_key,
                    "score": item.score,
                }
                for symbol, item in self._state.items()
            },
        }

    def set_state(self, state: dict[str, Any]) -> None:
        if not isinstance(state, dict):
            return
        self._last_decision_week = str(state.get("last_decision_week", ""))
        self._tick = _safe_non_negative_int(state.get("tick"))
        raw = state.get("symbol_state")
        if not isinstance(raw, dict):
            return
        for symbol, payload in raw.items():
            if symbol not in self._state or not isinstance(payload, dict):
                continue
            item = self._state[symbol]
            try:
                target = item.closes
                target.clear()
                maxlen = int(target.maxlen or 0)
                values = _coerce_float_list(payload.get("closes"))
                for value in values[-maxlen:] if maxlen else values:
                    target.append(value)
                item.mode = _mode(payload.get("mode"))
                item.entry_price = safe_float(payload.get("entry_price"))
                item.bars_held = _safe_non_negative_int(payload.get("bars_held"))
                raw_cooldown = payload.get("bars_since_exit")
                item.bars_since_exit = (
                    _safe_non_negative_int(raw_cooldown)
                    if raw_cooldown is not None
                    else _COOLDOWN_SATISFIED
                )
                item.last_bar_key = str(payload.get("last_bar_key", ""))
                item.score = safe_float(payload.get("score"))
            except Exception:
                continue

    # ------------------------------------------------------------------ #
    # ingestion / cadence
    # ------------------------------------------------------------------ #
    def _week_key(self, raw_time: Any) -> str:
        """Bucket a bar timestamp into an ISO ``YYYY-Wnn`` weekly decision key."""
        dt = _event_datetime_utc(raw_time)
        if dt is None:
            return time_key(raw_time)
        iso = dt.isocalendar()
        return f"{int(iso[0]):04d}-W{int(iso[1]):02d}"

    def _update_symbol(self, symbol: str, snapshot: _Snapshot) -> bool:
        item = self._state[symbol]
        key = time_key(snapshot.time)
        if key and key == item.last_bar_key:
            return False
        item.last_bar_key = key
        close = safe_float(snapshot.close)
        if close is None or close <= self.min_price:
            return False
        item.closes.append(float(close))
        return True

    def calculate_signals_window(self, event: Any, aggregator: Any = None) -> None:
        _ = aggregator
        updated = False
        for symbol in _event_symbols(event, self.symbol_list):
            snapshot = _window_snapshot(event, symbol)
            if snapshot is not None and self._update_symbol(symbol, snapshot):
                updated = True
        week = self._week_key(getattr(event, "time", None))
        if updated and week and week != self._last_decision_week:
            self._last_decision_week = week
            self._tick += 1
            self._evaluate(getattr(event, "time", None))

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
                week = self._week_key(snapshot.time)
                if week and week != self._last_decision_week:
                    self._last_decision_week = week
                    self._tick += 1
                    self._evaluate(snapshot.time)

    # ------------------------------------------------------------------ #
    # scoring
    # ------------------------------------------------------------------ #
    def _score(self) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
        """Return ``(id_by_symbol, pret_by_symbol, vol_by_symbol)`` over the book."""
        ids: dict[str, float] = {}
        prets: dict[str, float] = {}
        vols: dict[str, float] = {}
        for symbol, item in self._state.items():
            closes = list(item.closes)
            if len(closes) < self._min_history:
                continue
            id_value = information_discreteness(
                closes,
                formation_bars=self.formation_bars,
                skip_bars=self.skip_bars,
                zero_eps=self.zero_eps,
            )
            if id_value is None:
                continue
            pret = _formation_return(closes, self.formation_bars, self.skip_bars)
            if pret is None:
                continue
            vol = realized_volatility(closes, window=self.vol_window)
            if vol is None or vol <= _EPS:
                continue
            ids[symbol] = float(id_value)
            prets[symbol] = float(pret)
            vols[symbol] = float(vol)
        return ids, prets, vols

    def _eligible(self, ids: dict[str, float]) -> list[str]:
        """Return the most-continuous ``continuity_pct`` names with ``ID <= id_max``."""
        scored = list(ids)
        by_id = sorted(scored, key=lambda symbol: (ids[symbol], symbol))
        n_continuous = max(1, int(len(scored) * self.continuity_pct))
        rank_eligible = set(by_id[:n_continuous])
        return [
            symbol for symbol in by_id if symbol in rank_eligible and ids[symbol] <= self.id_max
        ]

    def _select_book(self, ids: dict[str, float], prets: dict[str, float]) -> dict[str, str | None]:
        """Resolve the desired book: momentum among the continuous eligibles."""
        eligible = self._eligible(ids)
        by_pret = sorted(eligible, key=lambda symbol: (-prets[symbol], symbol))
        count = len(by_pret)
        n_side = max(1, int(count * self.quantile_pct))
        if 2 * n_side > count:
            n_side = count // 2
        buffer = self.rank_hysteresis_buffer
        long_entry: set[str] = set()
        long_hold: set[str] = set()
        short_entry: set[str] = set()
        short_hold: set[str] = set()
        if n_side >= 1:
            long_entry = set(by_pret[:n_side])
            long_hold = set(by_pret[: min(count, n_side + buffer)])
            if self.allow_short:
                short_entry = set(by_pret[count - n_side :])
                short_hold = set(by_pret[max(0, count - n_side - buffer) :])
        # A name cannot be both a long and short hold candidate (thin books).
        contested = long_hold & short_hold
        long_hold -= contested
        short_hold -= contested

        desired: dict[str, str | None] = {}
        for symbol, item in self._state.items():
            cur = item.mode
            if cur in {"LONG", "SHORT"} and item.bars_held >= self.max_hold_decisions:
                desired[symbol] = None
                continue
            if cur in {"LONG", "SHORT"} and item.bars_held < self.min_hold_decisions:
                # Hard min-hold: a would-be flip inside the window is suppressed.
                desired[symbol] = cur
                continue
            if cur == "LONG":
                desired[symbol] = "LONG" if symbol in long_hold else None
            elif cur == "SHORT":
                desired[symbol] = "SHORT" if symbol in short_hold else None
            elif item.bars_since_exit < self.cooldown_decisions:
                desired[symbol] = None
            elif symbol in long_entry:
                desired[symbol] = "LONG"
            elif symbol in short_entry:
                desired[symbol] = "SHORT"
            else:
                desired[symbol] = None
        return desired

    # ------------------------------------------------------------------ #
    # decision (weekly cadence)
    # ------------------------------------------------------------------ #
    def _evaluate(self, event_time: Any) -> None:
        if len(self.symbol_list) < self.min_symbols:
            return
        for item in self._state.values():
            if item.mode in {"LONG", "SHORT"}:
                item.bars_held += 1
            else:
                item.bars_since_exit += 1

        ids, prets, vols = self._score()
        if len(ids) < self.min_symbols:
            self._enforce_max_hold(event_time)
            return
        desired = self._select_book(ids, prets)
        self._emit_targets(desired, ids, prets, vols, event_time)

    def _enforce_max_hold(self, event_time: Any) -> None:
        """Exit only positions past ``max_hold_decisions`` when the book self-skips."""
        for symbol, item in self._state.items():
            if item.mode in {"LONG", "SHORT"} and item.bars_held >= self.max_hold_decisions:
                self._emit_exit(symbol, item, event_time, reason="max_hold")

    def _inverse_vol_weights(
        self, book: dict[str, str], vols: dict[str, float]
    ) -> dict[str, float]:
        inv = {
            symbol: 1.0 / max(vols.get(symbol, 0.0), _EPS)
            for symbol in book
            if vols.get(symbol, 0.0) > _EPS
        }
        total = sum(inv.values())
        if total <= _EPS:
            return {}
        return {symbol: (inv[symbol] / total) * self.target_gross_exposure for symbol in inv}

    def _emit_targets(
        self,
        desired: dict[str, str | None],
        ids: dict[str, float],
        prets: dict[str, float],
        vols: dict[str, float],
        event_time: Any,
    ) -> None:
        book = {symbol: mode for symbol, mode in desired.items() if mode in {"LONG", "SHORT"}}
        weights = self._inverse_vol_weights(book, vols)
        for symbol, item in self._state.items():
            target_mode = desired.get(symbol)
            price = item.closes[-1] if item.closes else None
            if target_mode not in {"LONG", "SHORT"}:
                if item.mode != "OUT":
                    self._emit_exit(symbol, item, event_time, reason="rank_lapsed")
                continue
            score = float(prets.get(symbol, 0.0))
            if item.mode == target_mode:
                item.score = score
                continue
            if item.mode != "OUT":
                _emit(
                    self.events,
                    strategy_id=_STRATEGY_ID,
                    symbol=symbol,
                    event_time=event_time,
                    signal_type="EXIT",
                    price=price,
                    metadata={"strategy": _STRATEGY_NAME, "reason": "side_flip"},
                )
            weight = float(weights.get(symbol, 0.0))
            alloc = max(0.0, self.base_allocation * weight)
            metadata = _target_metadata(
                strategy=_STRATEGY_NAME,
                target_allocation=alloc,
                max_order_value=self.max_order_value,
                score=score,
                target_mode=target_mode,
                information_discreteness=float(ids.get(symbol, 0.0)),
                formation_return=score,
                inverse_vol_weight=weight,
            )
            if self.max_symbol_exposure_pct > 0.0:
                metadata["max_symbol_exposure_pct"] = min(
                    float(metadata.get("max_symbol_exposure_pct", self.max_symbol_exposure_pct)),
                    self.max_symbol_exposure_pct,
                )
            _emit(
                self.events,
                strategy_id=_STRATEGY_ID,
                symbol=symbol,
                event_time=event_time,
                signal_type=target_mode,
                strength=max(0.25, min(3.0, abs(score) * 5.0)),
                price=price,
                metadata=metadata,
            )
            item.mode = target_mode
            item.entry_price = price
            item.bars_held = 0
            item.score = score

    def _emit_exit(self, symbol: str, item: _State, event_time: Any, *, reason: str) -> None:
        price = item.closes[-1] if item.closes else None
        _emit(
            self.events,
            strategy_id=_STRATEGY_ID,
            symbol=symbol,
            event_time=event_time,
            signal_type="EXIT",
            price=price,
            metadata={"strategy": _STRATEGY_NAME, "reason": reason},
        )
        item.mode = "OUT"
        item.entry_price = None
        item.bars_held = 0
        item.bars_since_exit = 0
        item.score = None


# --------------------------------------------------------------------------- #
# Candidate-wiring hints for the integration wave (this lane does NOT wire
# candidates itself -- new-file-only, no shared-file edits).  Cross-sectional
# long-short BOOK: admission route is ``allow_multi_asset=True`` -- ID-conditioned
# momentum, NOT carry, so it is honestly EXCLUDED from any carry tag-superset
# allowlist (no fake carry tag).  Data-PC gates it on RPT >= 10bps per split and
# incremental orthogonal ``factor_ic`` vs the momentum / efficiency / fit
# incumbents PLUS its own unconditioned-momentum ablation (gate-of-the-gate).
# --------------------------------------------------------------------------- #
_SUGGESTED_FAMILY = "cross_sectional"
_SUGGESTED_CANDIDATE_TAGS: tuple[str, ...] = (
    "cross_sectional",
    "momentum",
    "frog_in_the_pan",
    "information_discreteness",
    "sign_census",
    "low_turnover",
    "crypto",
)

# Candidate slice (DAILY bars, internally weekly-sampled: ``formation_bars`` /
# ``skip_bars`` are DAILY bars, ``min_hold_decisions`` are WEEKLY decision bars).
# Cross-sectional long-short book -- vanilla momentum CONDITIONED on the sign
# census; the momentum leg is deliberately plain so the conditioning is isolated.
_INFORMATION_DISCRETENESS_SLICE: dict[str, tuple[dict[str, Any], ...]] = {
    "1d": (
        {
            "variant": "fip_8wk_p50",
            "formation_bars": 56,
            "skip_bars": 7,
            "continuity_pct": 0.50,
            "id_max": 0.0,
            "quantile_pct": 0.25,
            "vol_window": 30,
            "min_hold_decisions": 4,
            "min_symbols": 6,
            "allow_short": True,
            "target_gross_exposure": 1.0,
        },
        {
            "variant": "fip_4wk_p33",
            "formation_bars": 28,
            "skip_bars": 7,
            "continuity_pct": 0.33,
            "id_max": 0.0,
            "quantile_pct": 0.25,
            "vol_window": 30,
            "min_hold_decisions": 4,
            "min_symbols": 6,
            "allow_short": True,
            "target_gross_exposure": 1.0,
        },
    ),
}

__all__ = ["InformationDiscretenessMomentumStrategy"]
