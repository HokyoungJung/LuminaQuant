"""Cross-sectional dollar-volume TURNOVER-SHARE rotation (flow_share consumer).

``CrossSectionalFlowShareRotationStrategy`` is the first consumer of
``indicators/flow_share.py`` (the cross-sectional turnover-share/z-extremeness
primitives absorbed from the Reference-Price repo's ``amt_ratio_index``
methodology, itself built for trade-flow indices from customs data).  Every bar,
each symbol's per-bar quote (dollar) volume -- ``close * volume`` as a
dollar-volume proxy -- is expressed as its SHARE of the universe's total
turnover via ``cross_sectional_share``.  A rising share means the crowd's
attention/flow is concentrating into a name; a collapsing share means it is
being abandoned; a share that spikes to an extreme and then reverses is a
blow-off.

THEORY / PROVENANCE
--------------------
- High-volume return premium (Gervais, Kaniel & Mingelgrin, 2001): abnormally
  high trading activity precedes a short-horizon return continuation as
  attention/liquidity draws in follow-on demand.
- Attention-driven trading (Barber & Odean, 2008): retail/crowd attention
  concentrates in high-activity names, which the turnover-SHARE (rather than
  the raw volume level) isolates cross-sectionally at every bar.
- Amihud (2002) illiquidity/price-impact framing motivates using dollar
  (quote) volume rather than raw contract volume as the flow proxy, and
  motivates fading a blow-off: a name that has soaked up an extreme share of
  turnover while its price is already reversing is exhausting its liquidity
  runway, not starting a fresh move.

SIGNAL SPEC
-----------
Per completed decision bar (OHLCV including volume), per symbol:

1. ``dollar_volume = close * volume`` (dollar-volume proxy).
2. ``share = cross_sectional_share(dollar_volume, total_dollar_volume)`` across
   the current cross-section; appended to a per-symbol trailing ``shares``
   deque (bounded to ``share_z_window`` bars).
3. ``share_z`` = rolling (time-series) z-score of that symbol's own trailing
   share history (last observation vs. its trailing mean/std).
4. ``extremeness = cdf_extremeness(share_z)`` -- a bounded ``[0, 1]`` anomaly
   gauge of how extreme the current share reading is in either direction.
5. ``rank = dense_rank_desc(current_shares, share)`` -- 1-based dense rank of
   this symbol's current share among the cross-section (diagnostic metadata).
6. ``confirm_return = simple_return(closes, lookback=confirm_window)`` --
   short-horizon return confirmation.

Qualification:

- LONG candidate: rising share (``share_z >= share_z_entry``) AND positive
  short-horizon return confirmation (``confirm_return > 0``).
- SHORT candidate: collapsing share (``share_z <= -share_z_entry``) OR a
  blow-off (``extremeness >= blowoff_extremeness`` AND ``confirm_return < 0``
  -- an extreme share reading, in either direction, paired with price already
  rolling over).

Each side is ranked by ``|share_z|`` and the top ``top_n`` per side are
selected; sizing is inverse-realized-vol risk parity normalized to
``target_gross_exposure`` and scaled by a ``target_vol`` portfolio clamp (same
convention as the sibling cross-sectional books).  A held name EXITs when its
qualifying condition lapses (no longer selected) or ``max_hold_bars``/stop-loss
is hit.  ``share_weighted_return`` (the top-N share-weighted composite return
of the current cross-section) is carried as a diagnostic
``universe_share_weighted_return`` metadata field on every emitted signal --
it does not gate any decision.  The book self-skips (emits nothing) whenever
fewer than ``min_symbols`` names carry sufficient share/return/vol history.

DISTINCT-FROM
-------------
No other registered sleeve ranks the universe by cross-sectional turnover
SHARE: the existing volume-based indicators (``volume_zscore`` and friends) are
PER-SYMBOL level measures, not a share of the cross-section's total flow this
bar.  This is also distinct from ``CrossSectionalShortTermReversalStrategy``
and the equity cross-sectional momentum sleeves (e.g.
``CrossSectionalEquityMomentumStrategy``), which all rank symbols by their
own RETURNS; this book ranks by where turnover is concentrating and only uses
return as a confirmation/blow-off filter, not the primary ranking signal.

This module is data-local (no I/O, no hidden configuration bus), pure Python
(``math``/``deque`` only, no numpy), and never raises from
``calculate_signals``.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

from lumina_quant.core.plugin_registry import register
from lumina_quant.indicators.alpha_features import realized_volatility, simple_return
from lumina_quant.indicators.common import safe_float, time_key
from lumina_quant.indicators.flow_share import (
    cdf_extremeness,
    cross_sectional_share,
    dense_rank_desc,
    share_weighted_return,
)
from lumina_quant.strategies.adaptive_crypto_alpha_sleeves import (
    _age_cross_positions,
    _state_size,
)
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
from lumina_quant.tuning import HyperParam, resolve_params_from_schema

_STRATEGY_ID = "flow_share_rotation"
_STRATEGY_NAME = "CrossSectionalFlowShareRotationStrategy"


@dataclass(slots=True)
class _State:
    closes: deque[float]
    volumes: deque[float]
    shares: deque[float]
    mode: str = "OUT"
    entry_price: float | None = None
    bars_held: int = 0
    last_time_key: str = ""
    score: float | None = None


def _rolling_z(values: list[float]) -> float:
    """Sample z-score of the last value vs its trailing window (0.0 if degenerate)."""
    if len(values) < 2:
        return 0.0
    mean_value = sum(values) / float(len(values))
    variance = sum((value - mean_value) ** 2 for value in values) / float(len(values) - 1)
    sigma = variance**0.5
    if sigma <= _EPS:
        return 0.0
    return float((values[-1] - mean_value) / sigma)


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


@register("strategy", "CrossSectionalFlowShareRotationStrategy", interface="event_driven")
class CrossSectionalFlowShareRotationStrategy(Strategy):
    """Rotate long/short on cross-sectional dollar-volume turnover-share extremes.

    See the module docstring for the full theory, signal spec, and
    distinct-from rationale.  This class only reads local event/bar OHLCV
    (including volume); it performs no I/O and never raises from
    ``calculate_signals``.
    """

    preferred_contract = "market_window"
    uses_timeframe_aggregator = False

    @classmethod
    def get_param_schema(cls) -> dict[str, HyperParam]:
        return {
            "share_z_window": HyperParam.integer("share_z_window", default=60, low=8, high=2000),
            "confirm_window": HyperParam.integer("confirm_window", default=3, low=1, high=500),
            "share_z_entry": HyperParam.floating("share_z_entry", default=1.0, low=0.1, high=5.0),
            "blowoff_extremeness": HyperParam.floating(
                "blowoff_extremeness", default=0.85, low=0.50, high=0.999
            ),
            "top_n": HyperParam.integer("top_n", default=3, low=1, high=50),
            "vol_window": HyperParam.integer("vol_window", default=20, low=2, high=2000),
            "allow_short": HyperParam.boolean("allow_short", default=True, grid=[True, False]),
            "min_symbols": HyperParam.integer("min_symbols", default=4, low=2, high=512),
            "target_gross_exposure": HyperParam.floating(
                "target_gross_exposure", default=1.0, low=0.0, high=3.0
            ),
            "target_vol": HyperParam.floating("target_vol", default=0.20, low=0.0, high=2.0),
            "stop_loss_pct": HyperParam.floating("stop_loss_pct", default=0.08, low=0.0, high=0.50),
            "max_hold_bars": HyperParam.integer("max_hold_bars", default=60, low=0, high=200000),
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
        self.share_z_window = max(8, int(resolved["share_z_window"]))
        self.confirm_window = max(1, int(resolved["confirm_window"]))
        self.share_z_entry = max(0.0, float(resolved["share_z_entry"]))
        self.blowoff_extremeness = max(0.0, min(1.0, float(resolved["blowoff_extremeness"])))
        self.top_n = max(1, int(resolved["top_n"]))
        self.vol_window = max(2, int(resolved["vol_window"]))
        self.allow_short = bool(resolved["allow_short"])
        self.min_symbols = max(2, int(resolved["min_symbols"]))
        self.target_gross_exposure = max(0.0, float(resolved["target_gross_exposure"]))
        self.target_vol = max(0.0, float(resolved["target_vol"]))
        self.stop_loss_pct = max(0.0, float(resolved["stop_loss_pct"]))
        self.max_hold_bars = max(0, int(resolved["max_hold_bars"]))
        self.base_allocation = max(0.0, float(resolved["base_allocation"]))
        self.max_symbol_exposure_pct = max(0.0, float(resolved["max_symbol_exposure_pct"]))
        self.max_order_value = max(0.0, float(resolved["max_order_value"]))
        self.min_price = max(0.0, float(resolved["min_price"]))
        # Minimum trailing share observations before the rolling z-score of
        # share is trusted for ranking (a fraction of the configured window).
        self._share_history_min = max(8, self.share_z_window // 3)
        size = _state_size(self.confirm_window + 1, self.vol_window + 1, self.max_hold_bars)
        self._state: dict[str, _State] = {
            symbol: _State(
                closes=deque(maxlen=size),
                volumes=deque(maxlen=size),
                shares=deque(maxlen=self.share_z_window),
            )
            for symbol in self.symbol_list
        }
        self._last_eval_time_key = ""
        self._tick = 0
        # Recent decision-bar epochs (seconds) for deterministic bar-spacing
        # inference: the vol-target scalar annualizes the per-bar portfolio vol
        # via sqrt(bars_per_year) derived from the median gap here.
        self._recent_times: deque[float] = deque(maxlen=16)

    # ------------------------------------------------------------------ #
    # state
    # ------------------------------------------------------------------ #
    def get_state(self) -> dict[str, Any]:
        return {
            "last_eval_time_key": self._last_eval_time_key,
            "tick": int(self._tick),
            "recent_times": list(self._recent_times),
            "symbol_state": {
                symbol: {
                    "closes": list(item.closes),
                    "volumes": list(item.volumes),
                    "shares": list(item.shares),
                    "mode": item.mode,
                    "entry_price": item.entry_price,
                    "bars_held": int(item.bars_held),
                    "last_time_key": item.last_time_key,
                    "score": item.score,
                }
                for symbol, item in self._state.items()
            },
        }

    def set_state(self, state: dict[str, Any]) -> None:
        if not isinstance(state, dict):
            return
        self._last_eval_time_key = str(state.get("last_eval_time_key", ""))
        self._tick = _safe_non_negative_int(state.get("tick"))
        self._recent_times.clear()
        for value in _coerce_float_list(state.get("recent_times"))[
            -int(self._recent_times.maxlen or 0) :
        ]:
            self._recent_times.append(value)
        raw = state.get("symbol_state")
        if not isinstance(raw, dict):
            return
        for symbol, payload in raw.items():
            if symbol not in self._state or not isinstance(payload, dict):
                continue
            item = self._state[symbol]
            try:
                for attr in ("closes", "volumes"):
                    target = getattr(item, attr)
                    target.clear()
                    maxlen = int(target.maxlen or 0)
                    values = _coerce_float_list(payload.get(attr))
                    for value in values[-maxlen:] if maxlen else values:
                        target.append(value)
                item.shares.clear()
                share_maxlen = int(item.shares.maxlen or 0)
                share_values = [
                    value
                    for value in _coerce_float_list(payload.get("shares"))
                    if 0.0 <= value <= 1.0
                ]
                for value in share_values[-share_maxlen:] if share_maxlen else share_values:
                    item.shares.append(value)
                mode = str(payload.get("mode", "OUT")).upper()
                item.mode = mode if mode in {"OUT", "LONG", "SHORT"} else "OUT"
                item.entry_price = safe_float(payload.get("entry_price"))
                item.bars_held = _safe_non_negative_int(payload.get("bars_held"))
                item.last_time_key = str(payload.get("last_time_key", ""))
                item.score = safe_float(payload.get("score"))
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
                key = time_key(snapshot.time)
                if key and key != self._last_eval_time_key:
                    self._last_eval_time_key = key
                    self._tick += 1
                    self._evaluate(snapshot.time)

    # ------------------------------------------------------------------ #
    # scoring / selection
    # ------------------------------------------------------------------ #
    def _score_and_select(
        self,
    ) -> tuple[dict[str, tuple[str, float, dict[str, Any]]], dict[str, float]]:
        current: dict[str, tuple[float, float]] = {}
        for symbol, item in self._state.items():
            if not item.closes:
                continue
            close = item.closes[-1]
            if close is None or close <= 0.0:
                continue
            volume = item.volumes[-1] if item.volumes else 0.0
            current[symbol] = (close, max(0.0, close * volume))

        total_dv = sum(dv for _close, dv in current.values())
        current_shares: dict[str, float] = {}
        for symbol, (_close, dv) in current.items():
            share = cross_sectional_share(dv, total_dv)
            if share is None:
                continue
            item = self._state[symbol]
            item.shares.append(share)
            current_shares[symbol] = share

        share_values = list(current_shares.values())

        # Diagnostic-only composite: share-weighted 1-bar return of the current
        # cross-section (never gates a decision).
        one_bar_returns: dict[str, float] = {}
        for symbol in current_shares:
            ret = simple_return(list(self._state[symbol].closes), lookback=1)
            if ret is not None:
                one_bar_returns[symbol] = ret
        common = [symbol for symbol in current_shares if symbol in one_bar_returns]
        universe_diag: float | None = None
        if common:
            universe_diag = share_weighted_return(
                [current_shares[symbol] for symbol in common],
                [one_bar_returns[symbol] for symbol in common],
            )

        needed_closes = max(self.confirm_window, self.vol_window) + 1
        longs: list[tuple[float, str, float, dict[str, Any]]] = []
        shorts: list[tuple[float, str, float, dict[str, Any]]] = []
        vols: dict[str, float] = {}
        eligible = 0
        for symbol, share in current_shares.items():
            item = self._state[symbol]
            if len(item.closes) < needed_closes or len(item.shares) < self._share_history_min:
                continue
            vol = realized_volatility(list(item.closes), window=self.vol_window)
            if vol is None or vol <= _EPS:
                continue
            share_z = _rolling_z(list(item.shares))
            extremeness = cdf_extremeness(share_z)
            confirm_ret = simple_return(list(item.closes), lookback=self.confirm_window)
            eligible += 1
            vols[symbol] = float(vol)
            rank = dense_rank_desc(share_values, share)
            meta: dict[str, Any] = {
                "share": float(share),
                "share_z": float(share_z),
                "extremeness": float(extremeness) if extremeness is not None else None,
                "confirm_return": float(confirm_ret) if confirm_ret is not None else None,
                "rank": int(rank) if rank is not None else None,
                "universe_share_weighted_return": (
                    float(universe_diag) if universe_diag is not None else None
                ),
            }
            long_ok = (
                share_z >= self.share_z_entry and confirm_ret is not None and confirm_ret > 0.0
            )
            collapse_ok = share_z <= -self.share_z_entry
            blowoff_ok = (
                extremeness is not None
                and extremeness >= self.blowoff_extremeness
                and confirm_ret is not None
                and confirm_ret < 0.0
            )
            if long_ok:
                longs.append((abs(share_z), symbol, share_z, meta))
            elif collapse_ok or blowoff_ok:
                shorts.append((abs(share_z), symbol, share_z, meta))

        if eligible < self.min_symbols:
            return {}, {}

        longs.sort(key=lambda row: row[0], reverse=True)
        shorts.sort(key=lambda row: row[0], reverse=True)
        targets: dict[str, tuple[str, float, dict[str, Any]]] = {}
        for _metric, symbol, share_z, meta in longs[: self.top_n]:
            targets[symbol] = ("LONG", float(share_z), meta)
        if self.allow_short:
            for _metric, symbol, share_z, meta in shorts[: self.top_n]:
                targets[symbol] = ("SHORT", float(share_z), meta)
        return targets, vols

    def _inverse_vol_weights(
        self,
        targets: dict[str, tuple[str, float, dict[str, Any]]],
        vols: dict[str, float],
    ) -> tuple[dict[str, float], float]:
        inv = {
            symbol: 1.0 / max(vols.get(symbol, 0.0), _EPS)
            for symbol in targets
            if vols.get(symbol, 0.0) > _EPS
        }
        total_inv = sum(inv.values())
        if total_inv <= _EPS:
            return {}, 1.0
        # ``portfolio_vol`` is the inverse-vol-weighted PER-BAR vol; the
        # ``inv / total_inv`` normalization above is scale-invariant (annualizing
        # every vol_i cancels), so the risk-parity weights are horizon-free. The
        # vol-target SCALAR, however, compares ``portfolio_vol`` against an
        # annual-scale ``target_vol`` (~0.20): annualize the per-bar estimate via
        # sqrt(bars_per_year) inferred from observed bar spacing first, otherwise
        # the Moreira-Muir clamp is INERT. When spacing is unavailable we pass
        # through (scalar=1.0) rather than throttle on mismatched horizons.
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
    # aging / emission
    # ------------------------------------------------------------------ #
    def _age(self, event_time: Any) -> None:
        max_hold = self.max_hold_bars if self.max_hold_bars > 0 else (1 << 62)
        _age_cross_positions(
            self.events,
            self._state,  # type: ignore[arg-type]
            event_time=event_time,
            strategy_id=_STRATEGY_ID,
            strategy_name=_STRATEGY_NAME,
            stop_loss_pct=self.stop_loss_pct,
            max_hold_bars=max_hold,
        )

    def _evaluate(self, event_time: Any) -> None:
        if len(self.symbol_list) < self.min_symbols:
            return
        # Record the decision-bar epoch so the vol-target scalar can infer bar
        # spacing (this lane evaluates the ranker EVERY bar -- no coarser
        # rebalance gate -- so this per-bar hook already reflects bar cadence).
        dt = _event_datetime_utc(event_time)
        if dt is not None:
            self._recent_times.append(dt.timestamp())
        # Stops / max-hold run on EVERY evaluation before the ranker so a held
        # name is always protected, independent of this tick's eligibility.
        self._age(event_time)
        targets, vols = self._score_and_select()
        weights, scalar = self._inverse_vol_weights(targets, vols)
        self._emit_targets(targets, weights, scalar, event_time)

    def _emit_targets(
        self,
        targets: dict[str, tuple[str, float, dict[str, Any]]],
        weights: dict[str, float],
        scalar: float,
        event_time: Any,
    ) -> None:
        for symbol, item in self._state.items():
            target = targets.get(symbol)
            price = item.closes[-1] if item.closes else None
            if target is None:
                if item.mode != "OUT":
                    _emit(
                        self.events,
                        strategy_id=_STRATEGY_ID,
                        symbol=symbol,
                        event_time=event_time,
                        signal_type="EXIT",
                        price=price,
                        metadata={"strategy": _STRATEGY_NAME, "reason": "condition_lapsed"},
                    )
                    item.mode = "OUT"
                    item.entry_price = None
                    item.bars_held = 0
                    item.score = None
                continue
            target_mode, score, meta = target
            if item.mode == target_mode:
                item.score = float(score)
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
            if alloc <= 0.0:
                # Zero-alloc entries omit ``target_allocation`` from metadata and
                # the engine resizes them to its DEFAULT allocation; never emit an
                # unsized entry.  (The side-flip EXIT above, if any, already
                # flattened -- keep the bookkeeping consistent with it.)
                if item.mode != "OUT":
                    item.mode = "OUT"
                    item.entry_price = None
                    item.bars_held = 0
                    item.score = None
                continue
            stop_loss = None
            if price is not None and self.stop_loss_pct > 0.0:
                stop_loss = price * (
                    1.0 - self.stop_loss_pct if target_mode == "LONG" else 1.0 + self.stop_loss_pct
                )
            metadata = _target_metadata(
                strategy=_STRATEGY_NAME,
                target_allocation=alloc,
                max_order_value=self.max_order_value,
                score=float(score),
                target_mode=target_mode,
                inverse_vol_weight=weight,
                vol_target_scalar=float(scalar),
                **meta,
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
                strength=max(0.25, min(3.0, abs(score) / max(self.share_z_entry, _EPS))),
                price=price,
                stop_loss=stop_loss,
                metadata=metadata,
            )
            item.mode = target_mode
            item.entry_price = price
            item.bars_held = 0
            item.score = float(score)


# --------------------------------------------------------------------------- #
# Candidate-wiring hints for the W3 integrator (this lane does NOT wire
# candidates itself -- new-file-only, no shared-file edits per the live-safety
# plan).  Admission route is `allow_multi_asset=True` at the data-PC handoff:
# this book is cross-sectional + momentum but NOT carry, so it is honestly
# EXCLUDED from the {cross_sectional, carry, momentum} tag-superset allowlist
# route -- no fake carry tag is added to game that path.
# --------------------------------------------------------------------------- #
_SUGGESTED_FAMILY = "cross_sectional"
_SUGGESTED_CANDIDATE_TAGS: tuple[str, ...] = (
    "cross_sectional",
    "flow_share",
    "turnover_share",
    "attention",
    "momentum",
    "zscore",
    "crypto",
)

_FLOW_SHARE_ROTATION_SLICE: dict[str, tuple[dict[str, Any], ...]] = {
    "30m": (
        {
            "variant": "balanced_lo",
            "share_z_window": 48,
            "confirm_window": 3,
            "share_z_entry": 1.10,
            "blowoff_extremeness": 0.88,
            "top_n": 3,
            "vol_window": 24,
            "min_symbols": 5,
            "target_gross_exposure": 0.60,
            "target_vol": 0.18,
            "stop_loss_pct": 0.05,
            "max_hold_bars": 96,
            "allow_short": False,
        },
        {
            "variant": "guarded_ls",
            "share_z_window": 60,
            "confirm_window": 4,
            "share_z_entry": 1.25,
            "blowoff_extremeness": 0.90,
            "top_n": 2,
            "vol_window": 30,
            "min_symbols": 5,
            "target_gross_exposure": 0.50,
            "target_vol": 0.15,
            "stop_loss_pct": 0.045,
            "max_hold_bars": 72,
            "allow_short": True,
        },
    ),
    "1h": (
        {
            "variant": "balanced_lo",
            "share_z_window": 60,
            "confirm_window": 3,
            "share_z_entry": 1.10,
            "blowoff_extremeness": 0.88,
            "top_n": 3,
            "vol_window": 24,
            "min_symbols": 5,
            "target_gross_exposure": 0.60,
            "target_vol": 0.20,
            "stop_loss_pct": 0.06,
            "max_hold_bars": 72,
            "allow_short": False,
        },
        {
            "variant": "guarded_ls",
            "share_z_window": 72,
            "confirm_window": 4,
            "share_z_entry": 1.25,
            "blowoff_extremeness": 0.90,
            "top_n": 2,
            "vol_window": 30,
            "min_symbols": 5,
            "target_gross_exposure": 0.50,
            "target_vol": 0.16,
            "stop_loss_pct": 0.05,
            "max_hold_bars": 60,
            "allow_short": True,
        },
    ),
    "4h": (
        {
            "variant": "balanced_lo",
            "share_z_window": 42,
            "confirm_window": 2,
            "share_z_entry": 1.05,
            "blowoff_extremeness": 0.86,
            "top_n": 3,
            "vol_window": 18,
            "min_symbols": 5,
            "target_gross_exposure": 0.55,
            "target_vol": 0.22,
            "stop_loss_pct": 0.07,
            "max_hold_bars": 42,
            "allow_short": False,
        },
        {
            "variant": "guarded_ls",
            "share_z_window": 54,
            "confirm_window": 3,
            "share_z_entry": 1.20,
            "blowoff_extremeness": 0.88,
            "top_n": 2,
            "vol_window": 24,
            "min_symbols": 5,
            "target_gross_exposure": 0.45,
            "target_vol": 0.18,
            "stop_loss_pct": 0.06,
            "max_hold_bars": 36,
            "allow_short": True,
        },
    ),
    "1d": (
        {
            "variant": "balanced_lo",
            "share_z_window": 30,
            "confirm_window": 2,
            "share_z_entry": 1.00,
            "blowoff_extremeness": 0.85,
            "top_n": 3,
            "vol_window": 14,
            "min_symbols": 5,
            "target_gross_exposure": 0.50,
            "target_vol": 0.25,
            "stop_loss_pct": 0.08,
            "max_hold_bars": 21,
            "allow_short": False,
        },
        {
            "variant": "guarded_ls",
            "share_z_window": 40,
            "confirm_window": 3,
            "share_z_entry": 1.15,
            "blowoff_extremeness": 0.87,
            "top_n": 2,
            "vol_window": 20,
            "min_symbols": 5,
            "target_gross_exposure": 0.40,
            "target_vol": 0.20,
            "stop_loss_pct": 0.07,
            "max_hold_bars": 15,
            "allow_short": True,
        },
    ),
}


__all__ = ["CrossSectionalFlowShareRotationStrategy"]
