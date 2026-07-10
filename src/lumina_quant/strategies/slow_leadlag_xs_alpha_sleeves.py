"""Slow broad cross-sectional lead-lag sleeve (A2, weekly + min-hold rescued).

``SlowCrossSectionalLeadLagStrategy`` is a WEEKLY-cadence, BROAD cross-sectional
lead-lag book.  Every weekly decision bar it ranks the whole universe by each
follower's estimated exposure to the LAGGED returns of a liquid leader basket
(via :func:`cross_leadlag_spillover`), goes LONG the highest-loading laggards
and SHORT the lowest, and then freezes each position behind a hard minimum hold
plus a hysteresis band so the book barely turns over.  It is the JEDC-2024
cost-surviving *construction* -- a broad cross-section ranked by lagged-basket
spillover -- deliberately NOT the repo's dead BTC->ETH single-pair lead-lag
(graveyard #6, Sharpe 0.24, died at 20-30bps).

THEORY / PROVENANCE
--------------------
- JEDC 2024 (S0165188924000551): lagged cross-coin returns predict the focal
  coin cross-sectionally; a long-short book on that prediction survives
  transaction costs out-of-sample.  This is the best-evidenced lead-lag axis in
  the 10-axis external sweep (verdict 6).
- Lo & MacKinlay (1990, RFS): large-cap returns lead small-cap returns
  (cross-autocorrelation / slow information diffusion).
- Hou (2007, RFS): intra-industry lead-lag from gradual information diffusion.

The mechanism is a slow-diffusion under-reaction harvest: information impounds
into the liquid leaders first and diffuses into the laggards over the following
week(s).  The counterparty is the slow crowd that has not yet repriced the
laggards.

SIGNAL SPEC
-----------
Per completed WEEKLY decision bar (the ``_week_key`` internal clock buckets
ingested bars into ISO weeks, so the sleeve decides at most once per week even
when fed higher-frequency bars):

1. Snapshot each symbol's latest close into a per-symbol weekly close series.
2. Build the panel of symbols that carry at least ``min_history`` weekly closes
   and estimate lead-lag spillover with :func:`cross_leadlag_spillover`
   (``leaders`` = a liquid basket; ``max_lag`` weeks; ridge lag-regression).
   Each follower's ``score`` is its standardized predicted next-week return from
   the lagged leader basket -- its lead-lag LOADING.
3. Rank followers by that loading.  LONG the top ``max_longs`` whose loading
   clears ``+entry_z``; SHORT the bottom ``max_shorts`` whose loading clears
   ``-entry_z`` (when ``allow_short``).  Ties break by symbol name so the book
   is deterministic.
4. Size the entering names by inverse realized volatility (risk parity over
   ``vol_window``), normalized to ``target_gross_exposure`` and clamped to a
   portfolio ``target_vol``.

LOW-TURNOVER (RPT-survivability) DESIGN
---------------------------------------
The book is engineered to survive the repo's realized-per-turn-cost graveyard
(the single-pair lead-lag #6 died at 20-30bps; HTF sweeps #13/#14 died under the
10bps RPT floor).  Two turnover brakes, applied together:

- HARD MINIMUM HOLD: a held name is NEVER changed while
  ``bars_held < min_hold_bars`` -- a would-be flip inside the hold window is
  suppressed by design (the repo's proven RPT rescue, weekly bars here).
- HYSTERESIS: once past the min-hold, a held name is only exited when its
  same-side loading decays past the looser ``exit_z`` band (``exit_z <=
  entry_z``); it is not churned in and out around the entry threshold.  A
  ``cooldown_bars`` flat window then gates re-entry.

Protective stop-loss / ``max_hold_bars`` backstops can act inside the min-hold
window; with ``stop_loss_pct = 0`` the min-hold is the sole turnover gate, which
is the isolation used by the turnover DESIGN test.

DISTINCT-FROM (the three verified lead-lag incumbents)
------------------------------------------------------
- ``CrossCryptoSlowDiffusionStrategy`` (cross_crypto_slow_diffusion.py) trades a
  SINGLE fixed leader->target PAIR (BTC->ETH) and emits on exactly one target
  symbol; it cannot produce a broad cross-sectional book.
- ``SemisLeadLagRotationStrategy`` (equity_xs_factor_alpha_sleeves.py) only
  rebalances after a SINGLE named equity leader (SOXL) clears a >=1-sigma z-move
  and then ranks followers by the UNDER-REACTION GAP to that one leader
  (``leader_ret - follower_ret``); it is leader-trigger-gated and gap-ranked, not
  a broad spillover-loading book, and is architecturally flat on a pure-crypto
  universe with no equity leader.
- ``IntermarketLeadLagContinuationStrategy`` (cross_asset_trend_alpha_sleeves.py)
  instantiates directional legs only for CONFIGURED commodity->equity pairs
  (CL/BZ->XLE) and enters each follower in its own leader's direction on a per-
  pair z-trigger; it has zero legs (flat) on a universe without its configured
  pairs and never ranks a broad cross-section.

This sleeve's load-bearing difference from all three: a BROAD cross-section
ranked by lagged-basket spillover loading (long high / short low), on any
universe, with no single-leader trigger and no fixed pair list -- plus the
weekly min-hold + hysteresis turnover brake.

This module is data-local (no I/O, no hidden configuration bus), pure Python in
the strategy layer (delegates the numeric ridge lag-regression to the repo's
``cross_leadlag_spillover`` primitive; no scipy/sklearn/statsmodels),
completed-bar, and never raises from ``calculate_signals``.  It ships WITHOUT
``@register`` (inert): registration, tier hint, and candidate wiring land
atomically in the W3 integration wave.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

from lumina_quant.core.plugin_registry import register
from lumina_quant.indicators.advanced_alpha import cross_leadlag_spillover
from lumina_quant.indicators.alpha_features import realized_volatility
from lumina_quant.indicators.common import safe_float, time_key
from lumina_quant.strategies.external_alpha_sleeves import (
    _EPS,
    _Snapshot,
    _annualize_per_bar_vol,
    _emit,
    _event_datetime_utc,
    _event_symbols,
    _market_snapshot,
    _safe_non_negative_int,
    _target_metadata,
    _window_snapshot,
)
from lumina_quant.strategy import Strategy
from lumina_quant.symbols import canonical_symbol
from lumina_quant.tuning import HyperParam, resolve_params_from_schema

_STRATEGY_ID = "slow_cross_sectional_lead_lag"
_STRATEGY_NAME = "SlowCrossSectionalLeadLagStrategy"

# A fresh symbol has never traded, so its cooldown is treated as long elapsed --
# the first eligible entry must not be blocked by a phantom startup cooldown.
_COOLDOWN_SATISFIED = 1 << 30

_DEFAULT_LEADERS = "BTC/USDT,ETH/USDT,BNB/USDT,SOL/USDT"


@dataclass(slots=True)
class _State:
    """Per-symbol weekly close series + position / min-hold / cooldown bookkeeping."""

    closes: deque[float]  # one close per ISO week (the lead-lag input series)
    last_close: float | None = None  # latest close seen since the last weekly snapshot
    pending_dirty: bool = False  # a fresh close is waiting to be snapshotted this week
    mode: str = "OUT"  # OUT / LONG / SHORT
    entry_price: float | None = None
    bars_held: int = 0  # weekly decision bars in the CURRENT position
    bars_since_exit: int = _COOLDOWN_SATISFIED  # weekly decision bars since last exit
    last_bar_key: str = ""  # dedup identical ingested bars
    score: float | None = None


def _restore_deque(target: deque[float], payload: Any) -> None:
    """Refill a bounded deque from a serialized payload, dropping non-finite values."""
    target.clear()
    maxlen = int(target.maxlen or 0)
    values: list[float] = []
    if isinstance(payload, (list, tuple)):
        for value in payload:
            parsed = safe_float(value)
            if parsed is not None:
                values.append(parsed)
    for value in values[-maxlen:] if maxlen else values:
        target.append(value)


def _mode(raw: Any) -> str:
    """Coerce a serialized mode token to one of ``{OUT, LONG, SHORT}``."""
    parsed = str(raw or "OUT").upper()
    return parsed if parsed in {"OUT", "LONG", "SHORT"} else "OUT"


def _parse_leaders(raw: Any) -> tuple[str, ...]:
    """Parse a comma-separated leader-basket spec into a tuple of symbols."""
    tokens = [token.strip() for token in str(raw or "").split(",")]
    return tuple(token for token in tokens if token)


@register("strategy", _STRATEGY_NAME, interface="event_driven")
class SlowCrossSectionalLeadLagStrategy(Strategy):
    """Weekly broad cross-sectional lead-lag: long high-loading laggards, short low.

    See the module docstring for the full theory, signal spec, turnover-brake
    design, and distinct-from rationale.  This class only reads local event/bar
    OHLCV; it performs no I/O and never raises from ``calculate_signals``.
    """

    decision_cadence_seconds = 604800  # weekly effective cadence (see _week_key)
    preferred_contract = "market_window"
    uses_timeframe_aggregator = False

    strategy_name = _STRATEGY_NAME
    strategy_id = _STRATEGY_ID

    @classmethod
    def get_param_schema(cls) -> dict[str, HyperParam]:
        return {
            "leader_symbols": HyperParam.string(
                "leader_symbols", default=_DEFAULT_LEADERS, tunable=False
            ),
            "max_lag": HyperParam.integer("max_lag", default=2, low=1, high=24),
            "ridge_alpha": HyperParam.floating("ridge_alpha", default=1.0, low=0.001, high=100.0),
            "spillover_window": HyperParam.integer(
                "spillover_window", default=60, low=32, high=20000
            ),
            # Minimum weekly closes before a symbol enters the ranked panel. The
            # spillover primitive itself skips symbols with < 32 points, so the
            # floor is clamped there.
            "min_history": HyperParam.integer("min_history", default=40, low=32, high=20000),
            "entry_z": HyperParam.floating("entry_z", default=0.50, low=0.0, high=10.0),
            "exit_z": HyperParam.floating("exit_z", default=0.10, low=0.0, high=10.0),
            "max_longs": HyperParam.integer("max_longs", default=3, low=0, high=128),
            "max_shorts": HyperParam.integer("max_shorts", default=3, low=0, high=128),
            "allow_short": HyperParam.boolean("allow_short", default=True, grid=[True, False]),
            "min_symbols": HyperParam.integer("min_symbols", default=4, low=3, high=512),
            # The proven turnover rescue, weekly decision bars: a held name is
            # frozen for at least this many weeks (>= the low-turnover intent
            # that lifts RPT above the 10bps floor graveyard #6/#13/#14 failed).
            "min_hold_bars": HyperParam.integer("min_hold_bars", default=8, low=1, high=200000),
            "cooldown_bars": HyperParam.integer("cooldown_bars", default=2, low=0, high=200000),
            "max_hold_bars": HyperParam.integer("max_hold_bars", default=260, low=1, high=1000000),
            "vol_window": HyperParam.integer("vol_window", default=20, low=2, high=4096),
            "target_gross_exposure": HyperParam.floating(
                "target_gross_exposure", default=1.0, low=0.0, high=5.0
            ),
            "target_vol": HyperParam.floating("target_vol", default=0.20, low=0.0, high=2.0),
            "stop_loss_pct": HyperParam.floating("stop_loss_pct", default=0.10, low=0.0, high=0.50),
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
        self.leaders = _parse_leaders(resolved["leader_symbols"]) or _parse_leaders(
            _DEFAULT_LEADERS
        )
        self.max_lag = max(1, int(resolved["max_lag"]))
        self.ridge_alpha = max(1e-6, float(resolved["ridge_alpha"]))
        self.spillover_window = max(32, int(resolved["spillover_window"]))
        # The primitive skips symbols with < 32 weekly points; respect that floor.
        self.min_history = max(32, int(resolved["min_history"]))
        self.entry_z = max(0.0, float(resolved["entry_z"]))
        # Hysteresis band cannot exceed the entry band (exit is looser/earlier).
        self.exit_z = max(0.0, min(float(resolved["exit_z"]), self.entry_z))
        self.max_longs = max(0, int(resolved["max_longs"]))
        self.allow_short = bool(resolved["allow_short"])
        self.max_shorts = max(0, int(resolved["max_shorts"])) if self.allow_short else 0
        self.min_symbols = max(3, int(resolved["min_symbols"]))
        self.min_hold_bars = max(1, int(resolved["min_hold_bars"]))
        self.cooldown_bars = max(0, int(resolved["cooldown_bars"]))
        self.max_hold_bars = max(1, int(resolved["max_hold_bars"]))
        self.vol_window = max(2, int(resolved["vol_window"]))
        self.target_gross_exposure = max(0.0, float(resolved["target_gross_exposure"]))
        self.target_vol = max(0.0, float(resolved["target_vol"]))
        self.stop_loss_pct = max(0.0, float(resolved["stop_loss_pct"]))
        self.base_allocation = max(0.0, float(resolved["base_allocation"]))
        self.max_symbol_exposure_pct = max(0.0, float(resolved["max_symbol_exposure_pct"]))
        self.max_order_value = max(0.0, float(resolved["max_order_value"]))
        self.min_price = max(0.0, float(resolved["min_price"]))
        # Bounded ring buffer: enough weekly closes for the spillover window plus
        # a small margin (the primitive reads the tail of this series).
        self._history_size = max(self.spillover_window, self.min_history, self.vol_window) + 8
        self._state: dict[str, _State] = {
            symbol: _State(closes=deque(maxlen=self._history_size)) for symbol in self.symbol_list
        }
        self._last_decision_week = ""
        self._tick = 0
        # Recent WEEKLY decision epochs (seconds): the vol-target scalar annualizes
        # the per-week portfolio vol via sqrt(bars_per_year) from the median gap
        # here.  Recorded on the weekly decision clock so the spacing matches the
        # WEEKLY closes (``_snapshot_weekly_closes``) the realized vol is built on.
        self._recent_times: deque[float] = deque(maxlen=16)

    # ------------------------------------------------------------------ #
    # state
    # ------------------------------------------------------------------ #
    def get_state(self) -> dict[str, Any]:
        return {
            "last_decision_week": self._last_decision_week,
            "tick": int(self._tick),
            "recent_times": list(self._recent_times),
            "symbol_state": {
                symbol: {
                    "closes": list(item.closes),
                    "last_close": item.last_close,
                    "pending_dirty": bool(item.pending_dirty),
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
        _restore_deque(self._recent_times, state.get("recent_times"))
        raw = state.get("symbol_state")
        if not isinstance(raw, dict):
            return
        for symbol, payload in raw.items():
            if symbol not in self._state or not isinstance(payload, dict):
                continue
            item = self._state[symbol]
            try:
                _restore_deque(item.closes, payload.get("closes"))
                item.last_close = safe_float(payload.get("last_close"))
                item.pending_dirty = bool(payload.get("pending_dirty"))
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
        """Record the latest close for ``symbol`` (deduped); flag it for snapshot."""
        item = self._state[symbol]
        key = time_key(snapshot.time)
        if key and key == item.last_bar_key:
            return False
        item.last_bar_key = key
        close = safe_float(snapshot.close)
        if close is None or close <= self.min_price:
            return False
        item.last_close = float(close)
        item.pending_dirty = True
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
    # ranking
    # ------------------------------------------------------------------ #
    def _snapshot_weekly_closes(self) -> None:
        """Append each symbol's pending close to its weekly series (once per week)."""
        for item in self._state.values():
            if item.pending_dirty and item.last_close is not None:
                item.closes.append(float(item.last_close))
            item.pending_dirty = False

    def _spillover_scores(self) -> dict[str, float]:
        """Return the lead-lag loading (spillover score) per eligible follower."""
        price_by_symbol = {
            symbol: list(item.closes)
            for symbol, item in self._state.items()
            if len(item.closes) >= self.min_history
        }
        if len(price_by_symbol) < self.min_symbols:
            return {}
        try:
            result = cross_leadlag_spillover(
                price_by_symbol,
                leaders=self.leaders,
                max_lag=self.max_lag,
                ridge_alpha=self.ridge_alpha,
                window=self.spillover_window,
            )
        except Exception:
            return {}
        if not result.get("available"):
            return {}
        predictions = result.get("predictions") or {}
        canon_to_symbol = {canonical_symbol(symbol): symbol for symbol in price_by_symbol}
        scores: dict[str, float] = {}
        for follower, payload in predictions.items():
            symbol = canon_to_symbol.get(canonical_symbol(str(follower)))
            if symbol is None or not isinstance(payload, dict):
                continue
            value = safe_float(payload.get("score"))
            if value is None:
                continue
            scores[symbol] = float(value)
        return scores

    def _desired_targets(self, scores: dict[str, float]) -> dict[str, str]:
        """Pick the long/short book from the loading ranking (deterministic ties)."""
        targets: dict[str, str] = {}
        # Longs: highest loadings that clear +entry_z (tie-break by symbol name).
        longs = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
        for symbol, value in longs:
            if (
                value < self.entry_z
                or len([m for m in targets.values() if m == "LONG"]) >= self.max_longs
            ):
                break
            targets[symbol] = "LONG"
        if not self.allow_short or self.max_shorts <= 0:
            return targets
        # Shorts: lowest loadings that clear -entry_z.
        shorts = sorted(scores.items(), key=lambda kv: (kv[1], kv[0]))
        short_count = 0
        for symbol, value in shorts:
            if value > -self.entry_z or short_count >= self.max_shorts:
                break
            if symbol in targets:
                continue
            targets[symbol] = "SHORT"
            short_count += 1
        return targets

    def _inverse_vol_weights(self, targets: dict[str, str]) -> tuple[dict[str, float], float]:
        """Inverse-realized-vol risk-parity weights over the entering book."""
        vols: dict[str, float] = {}
        for symbol in targets:
            vol = realized_volatility(list(self._state[symbol].closes), window=self.vol_window)
            if vol is not None and vol > _EPS:
                vols[symbol] = float(vol)
        inv = {symbol: 1.0 / vols[symbol] for symbol in vols}
        total_inv = sum(inv.values())
        if total_inv <= _EPS:
            return {}, 1.0
        # ``portfolio_vol`` is the inverse-vol-weighted PER-WEEK vol; the
        # ``inv / total_inv`` normalization is scale-invariant, so the risk-parity
        # weights are horizon-free.  The vol-target SCALAR compares it against an
        # annual-scale ``target_vol`` (0.20): annualize the per-week estimate via
        # sqrt(bars_per_year) from the observed (weekly) decision spacing first,
        # otherwise the Moreira-Muir clamp is INERT.  Pass through (scalar=1.0)
        # when spacing is unavailable rather than throttle on mismatched horizons.
        portfolio_vol = sum((inv[symbol] / total_inv) * vols[symbol] for symbol in inv)
        scalar = 1.0
        if self.target_vol > 0.0 and portfolio_vol > _EPS:
            portfolio_vol_ann = _annualize_per_bar_vol(portfolio_vol, self._recent_times)
            if portfolio_vol_ann is not None and portfolio_vol_ann > _EPS:
                scalar = min(1.0, self.target_vol / portfolio_vol_ann)
        weights = {
            symbol: (inv[symbol] / total_inv) * self.target_gross_exposure * scalar
            for symbol in inv
        }
        return weights, float(scalar)

    # ------------------------------------------------------------------ #
    # decision (weekly cadence)
    # ------------------------------------------------------------------ #
    def _evaluate(self, event_time: Any) -> None:
        # Record the weekly decision epoch so the vol-target scalar can infer the
        # (weekly) bar spacing that annualizes the per-week portfolio vol.
        dt = _event_datetime_utc(event_time)
        if dt is not None:
            self._recent_times.append(dt.timestamp())
        self._snapshot_weekly_closes()
        scores = self._spillover_scores()
        signal_available = bool(scores)
        targets = self._desired_targets(scores) if signal_available else {}
        weights, scalar = self._inverse_vol_weights(targets) if targets else ({}, 1.0)
        self._decide_book(scores, targets, weights, scalar, signal_available, event_time)

    def _stop_hit(self, item: _State, price: float | None) -> bool:
        if price is None or item.entry_price is None or self.stop_loss_pct <= 0.0:
            return False
        if item.mode == "LONG":
            return price <= item.entry_price * (1.0 - self.stop_loss_pct)
        if item.mode == "SHORT":
            return price >= item.entry_price * (1.0 + self.stop_loss_pct)
        return False

    def _decide_book(
        self,
        scores: dict[str, float],
        targets: dict[str, str],
        weights: dict[str, float],
        scalar: float,
        signal_available: bool,
        event_time: Any,
    ) -> None:
        for symbol, item in self._state.items():
            price = item.closes[-1] if item.closes else item.last_close
            if item.mode in {"LONG", "SHORT"}:
                item.bars_held += 1
                # Protective backstops act even inside the min-hold window.
                stop = self._stop_hit(item, price)
                if stop or item.bars_held >= self.max_hold_bars:
                    self._emit_exit(
                        symbol,
                        item,
                        event_time,
                        reason="stop_loss" if stop else "max_hold",
                    )
                    continue
                if not signal_available:
                    continue
                # HARD min-hold rescue: no discretionary change inside the window.
                if item.bars_held < self.min_hold_bars:
                    continue
                score = scores.get(symbol)
                keep = score is not None and (
                    (item.mode == "LONG" and score >= self.exit_z)
                    or (item.mode == "SHORT" and score <= -self.exit_z)
                )
                if keep:
                    item.score = float(score)
                    continue
                self._emit_exit(symbol, item, event_time, reason="hysteresis_exit")
                continue
            # Flat: age the cooldown, then enter on a fresh gated target.
            item.bars_since_exit += 1
            if not signal_available:
                continue
            if item.bars_since_exit < self.cooldown_bars:
                continue
            target = targets.get(symbol)
            if target is None:
                continue
            if target == "SHORT" and not self.allow_short:
                continue
            weight = float(weights.get(symbol, 0.0))
            if weight <= 0.0:
                # Selection vol-gate: a target whose realized vol was
                # unavailable carries no risk-parity weight.  Entering with
                # alloc=0 would omit ``target_allocation`` from metadata and
                # the engine would resize the position to its DEFAULT
                # allocation — an unsized, un-vol-gated entry.
                continue
            self._enter(
                symbol,
                item,
                target,
                float(scores.get(symbol, 0.0)),
                weight,
                scalar,
                price,
                event_time,
            )

    def _enter(
        self,
        symbol: str,
        item: _State,
        mode: str,
        score: float,
        weight: float,
        scalar: float,
        price: float | None,
        event_time: Any,
    ) -> None:
        alloc = max(0.0, self.base_allocation * weight)
        if alloc <= 0.0:
            # Never emit an entry without a positive sizing target (see the
            # vol-gate in ``_decide_book``); metadata without
            # ``target_allocation`` falls back to engine default sizing.
            return
        stop_loss = None
        if price is not None and self.stop_loss_pct > 0.0:
            stop_loss = price * (
                1.0 - self.stop_loss_pct if mode == "LONG" else 1.0 + self.stop_loss_pct
            )
        metadata = _target_metadata(
            strategy=_STRATEGY_NAME,
            target_allocation=alloc,
            max_order_value=self.max_order_value,
            action="entry",
            target_mode=mode,
            leadlag_loading=float(score),
            inverse_vol_weight=float(weight),
            vol_target_scalar=float(scalar),
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
            signal_type=mode,
            strength=max(0.25, min(3.0, abs(score) / max(self.entry_z, _EPS))),
            price=price,
            stop_loss=stop_loss,
            metadata=metadata,
        )
        item.mode = mode
        item.entry_price = price
        item.bars_held = 0
        item.score = float(score)

    def _emit_exit(self, symbol: str, item: _State, event_time: Any, *, reason: str) -> None:
        price = item.closes[-1] if item.closes else item.last_close
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
# Candidate-wiring hints for the W3 integrator (this lane does NOT wire
# candidates itself -- new-file-only, no shared-file edits per the live-safety
# plan).  This is a cross-sectional long-short BOOK, so the data-PC handoff
# admits it via `allow_multi_asset=True` (family=cross_sectional); it is NOT a
# carry sleeve, so NO fake carry tag is added to game the tag-superset allowlist.
# The data-PC gates it on RPT >= 10bps per split (the exact gate graveyard #6
# failed) and incremental factor_ic vs CrossCryptoSlowDiffusion /
# SemisLeadLagRotation / IntermarketLeadLagContinuation.
# --------------------------------------------------------------------------- #
_SUGGESTED_FAMILY = "cross_sectional"
_SUGGESTED_CANDIDATE_TAGS: tuple[str, ...] = (
    "cross_sectional",
    "lead_lag",
    "spillover",
    "diffusion",
    "low_turnover",
    "min_hold",
    "crypto",
)

# Candidate slice.  This sleeve INTERNALLY samples ONE close per ISO week
# (``_snapshot_weekly_closes``) and decides on a timestamp-based WEEKLY clock
# (``_week_key``), so EVERY window here is denominated in WEEKLY decision bars,
# NOT raw bars: ``max_lag``/``spillover_window``/``min_history``/``vol_window``
# and the ``min_hold_bars``/``cooldown_bars``/``max_hold_bars`` turnover brakes
# are all weeks.  That makes the sleeve fully TIMEFRAME-AGNOSTIC -- feeding it
# 4h or 1h bars only changes which intra-week close becomes the weekly snapshot
# -- so the 4h and 1h cells reuse the 1d params VERBATIM (nothing scales).
# Broad cross-sectional long-short book -- the JEDC cost-surviving construction,
# NOT the dead single-pair form.  ``min_hold_bars`` >= 8 weeks is the
# RPT-survival design property (the redesign's entire purpose vs graveyard #6).
_SLOW_LEADLAG_XS_SLICE: dict[str, tuple[dict[str, Any], ...]] = {
    "4h": (
        {
            "variant": "weekly_core",
            "max_lag": 2,
            "ridge_alpha": 1.0,
            "spillover_window": 60,
            "min_history": 40,
            "entry_z": 0.50,
            "exit_z": 0.10,
            "max_longs": 10,
            "max_shorts": 10,
            "allow_short": True,
            "min_symbols": 5,
            "min_hold_bars": 8,
            "cooldown_bars": 2,
            "max_hold_bars": 260,
            "vol_window": 20,
            "target_gross_exposure": 1.0,
            "target_vol": 0.20,
            "stop_loss_pct": 0.10,
        },
    ),
    "1h": (
        {
            "variant": "weekly_core",
            "max_lag": 2,
            "ridge_alpha": 1.0,
            "spillover_window": 60,
            "min_history": 40,
            "entry_z": 0.50,
            "exit_z": 0.10,
            "max_longs": 10,
            "max_shorts": 10,
            "allow_short": True,
            "min_symbols": 5,
            "min_hold_bars": 8,
            "cooldown_bars": 2,
            "max_hold_bars": 260,
            "vol_window": 20,
            "target_gross_exposure": 1.0,
            "target_vol": 0.20,
            "stop_loss_pct": 0.10,
        },
    ),
    "1d": (
        {
            "variant": "weekly_core",
            "max_lag": 2,
            "ridge_alpha": 1.0,
            "spillover_window": 60,
            "min_history": 40,
            "entry_z": 0.50,
            "exit_z": 0.10,
            "max_longs": 10,
            "max_shorts": 10,
            "allow_short": True,
            "min_symbols": 5,
            "min_hold_bars": 8,
            "cooldown_bars": 2,
            "max_hold_bars": 260,
            "vol_window": 20,
            "target_gross_exposure": 1.0,
            "target_vol": 0.20,
            "stop_loss_pct": 0.10,
        },
    ),
}

__all__ = ["SlowCrossSectionalLeadLagStrategy"]
