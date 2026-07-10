"""Cross-sectional TradFi-perp off-session tug-of-war sleeve (session decomposition).

``CrossSectionalOffSessionTugOfWarStrategy`` decomposes each TradFi-equity/ETF
perpetual leg's 1h-bar return path, per UTC calendar day, into a CASH-ANCHORED
component (log-return accrued over the core US cash-session hours, UTC
``[cash_start_hour_utc, cash_end_hour_utc)`` = ``[14, 20)``, Mon-Fri) and an
UNANCHORED OFF-SESSION component (all remaining hours -- nights, weekends -- when
the underlying cash market is closed and the perp price is set purely by 24/7
crypto-native flow with no price-discovery anchor).  The two DST-ambiguous
boundary hours (``cash_start-1`` and ``cash_end`` = ``{13, 20}``) are dropped from
BOTH components.

The characteristic is ``TOW = mean(N_d) - mean(D_d)`` over a trailing formation of
``formation_days`` UTC days, where ``N_d`` / ``D_d`` are the day's summed
off-session / cash-session log-returns: how much of the drift accrues in hours
WITHOUT cash-market discovery.  ``TOW`` is cross-sectionally z-scored across the
eligible TradFi legs and residualized (single-regressor Gram-Schmidt, the shared
``cross_sectional_residualize`` primitive) against total trailing-return momentum
z, so the sleeve trades ONLY the decomposition split -- not a momentum tilt in
disguise.  It **SHORTS** the top residual-TOW quantile (persistently pumped by
unanchored off-session flow -- re-anchored downward when cash discovery resumes;
Akbas-Boehmer-Jiang-Koch persistent-overnight-strength-predicts-negatively sign)
and **LONGS** the bottom quantile, inverse-realized-vol sized.

PIVOT NOTE (recorded).  The literal Lou-Polk-Skouras overnight/intraday split is
NOT computable on this feed: Binance ``TRADIFI_PERPETUAL`` legs trade 24/7 through
the same kline pipeline (no session gap; repo-wide there is zero market-hours
handling).  What IS computable from UTC-stamped 1h bars is this cash-vs-unanchored
decomposition, so the axis is harvested through the SLOW cross-sectional
characteristic rather than a session-timed round-trip (which would be structural
cost death).  Positions are held continuously on a weekly ISO-week clock with a
hard ``min_hold_decisions`` floor and a rank-hysteresis band.

THEORY / PROVENANCE
-------------------
- Lou, Polk & Skouras (2019), *JFE* 134(1): "A Tug of War" -- opposing
  overnight/intraday clienteles; here transplanted to the cash-vs-unanchored split.
- Akbas, Boehmer, Jiang & Koch (2022), *JFE*: persistent overnight strength is a
  sentiment proxy that NEGATIVELY predicts the cross-section -- supplies the fade
  sign.
- Bogousslavsky (2021), *JFE* 141(1): infrequent rebalancing and end-of-day
  return patterns.

Counterparty: 24/7 crypto-native perp traders expressing TradFi views while the
underlying cash market is closed -- a structural LPS clientele demanding immediacy
on the only live venue (nights/weekends) with no discovery anchor; the next cash
open plus basis-arb desks re-anchor the perp.  HONEST FLAG: no published
tokenized-equity/TradFi-perp session study exists; off-session moves in
globally-traded names (TSM/ASML/BABA class) can be genuine Asia/Europe-hours
information -> continuation, the declared wrong-sign alternative.

This module is data-local (no I/O, no hidden configuration bus), pure Python
(``math`` only), and never raises from ``calculate_signals``.  It ships WITHOUT
``@register`` (inert until the integration wave wires it).
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Any

from lumina_quant.core.plugin_registry import register
from lumina_quant.indicators.alpha_features import realized_volatility
from lumina_quant.indicators.common import safe_float, time_key
from lumina_quant.indicators.cross_sectional_residualize import cross_sectional_residualize
from lumina_quant.research_universe import (
    BINANCE_TRADFI_EQUITY_SYMBOLS,
    BINANCE_TRADFI_ETF_INDEX_SYMBOLS,
    compact_to_slashed_usdt,
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

_STRATEGY_ID = "offsession_tugofwar"
_STRATEGY_NAME = "CrossSectionalOffSessionTugOfWarStrategy"


def _tradfi_equity_etf_universe() -> frozenset[str]:
    """TradFi equity + ETF/index perp legs, compact AND slashed forms.

    Premarket-only symbols (no cash session) and commodities (~23h futures
    sessions blur the split) are structurally excluded; crypto symbols in the feed
    are simply never members, so they are ingested-but-never-scored.
    """
    compact = (*BINANCE_TRADFI_EQUITY_SYMBOLS, *BINANCE_TRADFI_ETF_INDEX_SYMBOLS)
    members: set[str] = set()
    for symbol in compact:
        members.add(symbol)
        try:
            members.add(compact_to_slashed_usdt(symbol))
        except ValueError:
            continue
    return frozenset(members)


_TRADFI_EQUITY_ETF_UNIVERSE = _tradfi_equity_etf_universe()


@dataclass(slots=True)
class _State:
    # Trailing 1h closes (realized-vol sizing + last price for emission).
    closes: deque[float]
    # Finalized per-UTC-day (cash_log_return, off_log_return) pairs.
    day_returns: deque[tuple[float, float]]
    cur_date: str | None = None
    cur_cash: float = 0.0
    cur_off: float = 0.0
    prev_close: float | None = None
    mode: str = "OUT"
    entry_price: float | None = None
    decisions_held: int = 0
    last_time_key: str = ""
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


def _coerce_day_pairs(value: Any) -> list[tuple[float, float]]:
    """Coerce a serialized ``[[cash, off], ...]`` payload; never raises."""
    if not isinstance(value, (list, tuple)):
        return []
    out: list[tuple[float, float]] = []
    for item in value:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        cash = safe_float(item[0])
        off = safe_float(item[1])
        if cash is None or off is None:
            continue
        out.append((float(cash), float(off)))
    return out


def _cross_z(values: dict[str, float]) -> dict[str, float]:
    """Cross-sectional z-score of ``values`` (0.0 for every symbol if degenerate)."""
    count = len(values)
    if count == 0:
        return {}
    mean_value = sum(values.values()) / float(count)
    variance = sum((value - mean_value) ** 2 for value in values.values()) / float(
        max(1, count - 1)
    )
    sigma = variance**0.5
    if sigma <= _EPS:
        return dict.fromkeys(values, 0.0)
    return {symbol: (value - mean_value) / sigma for symbol, value in values.items()}


def _dispersion(values: list[float]) -> float:
    """Sample standard deviation of a value list (0.0 for < 2 entries)."""
    n = len(values)
    if n < 2:
        return 0.0
    mean_value = sum(values) / float(n)
    variance = sum((value - mean_value) ** 2 for value in values) / float(n - 1)
    return variance**0.5


@register("strategy", "CrossSectionalOffSessionTugOfWarStrategy", interface="event_driven")
class CrossSectionalOffSessionTugOfWarStrategy(Strategy):
    """Long-short XS fade of the persistent cash-vs-unanchored off-session tilt.

    See the module docstring for the full theory, the session-decomposition spec,
    and the distinct-from rationale versus the per-symbol session/overnight riders
    and the cross-sectional equity momentum incumbent.  Reads only local event/bar
    OHLCV, performs no I/O, and never raises from ``calculate_signals``.
    """

    decision_cadence_seconds = 1800
    preferred_contract = "market_window"
    uses_timeframe_aggregator = False

    @classmethod
    def get_param_schema(cls) -> dict[str, HyperParam]:
        return {
            "formation_days": HyperParam.integer("formation_days", default=42, low=5, high=2000),
            # Structurally pinned to NYSE cash hours (external market-structure
            # fact); the DST-ambiguous flanking hours {start-1, end} are dropped.
            "cash_start_hour_utc": HyperParam.integer(
                "cash_start_hour_utc", default=14, low=0, high=23, tunable=False
            ),
            "cash_end_hour_utc": HyperParam.integer(
                "cash_end_hour_utc", default=20, low=1, high=24, tunable=False
            ),
            "quantile_pct": HyperParam.floating("quantile_pct", default=0.25, low=0.02, high=0.50),
            "min_symbols": HyperParam.integer("min_symbols", default=5, low=2, high=512),
            "min_history_days": HyperParam.integer(
                "min_history_days", default=42, low=3, high=100000
            ),
            "min_hold_decisions": HyperParam.integer(
                "min_hold_decisions", default=2, low=0, high=100000
            ),
            "rank_hysteresis_buffer": HyperParam.integer(
                "rank_hysteresis_buffer", default=1, low=0, high=512
            ),
            "vol_window": HyperParam.integer("vol_window", default=168, low=2, high=20000),
            "off_variance_eps": HyperParam.floating(
                "off_variance_eps", default=1e-6, low=0.0, high=1.0, tunable=False
            ),
            "allow_short": HyperParam.boolean("allow_short", default=True, grid=[True, False]),
            "target_gross_exposure": HyperParam.floating(
                "target_gross_exposure", default=1.0, low=0.0, high=3.0
            ),
            "target_vol": HyperParam.floating("target_vol", default=0.20, low=0.0, high=2.0),
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
        configured = list(getattr(self.bars, "symbol_list", []) or [])
        # Universe filter: TradFi equity + ETF legs only.  Everything else
        # (crypto, premarket, commodities) is ingested-but-never-scored -- we
        # simply keep no state for it, so it can never enter a book.
        self.symbol_list = [s for s in configured if s in _TRADFI_EQUITY_ETF_UNIVERSE]
        resolved = resolve_params_from_schema(self.get_param_schema(), params, keep_unknown=False)
        self.formation_days = max(2, int(resolved["formation_days"]))
        self.cash_start_hour_utc = max(0, min(23, int(resolved["cash_start_hour_utc"])))
        self.cash_end_hour_utc = max(1, min(24, int(resolved["cash_end_hour_utc"])))
        self.quantile_pct = min(0.5, max(0.0, float(resolved["quantile_pct"])))
        self.min_symbols = max(2, int(resolved["min_symbols"]))
        self.min_history_days = max(2, int(resolved["min_history_days"]))
        self.min_hold_decisions = max(0, int(resolved["min_hold_decisions"]))
        self.rank_hysteresis_buffer = max(0, int(resolved["rank_hysteresis_buffer"]))
        self.vol_window = max(2, int(resolved["vol_window"]))
        self.off_variance_eps = max(0.0, float(resolved["off_variance_eps"]))
        self.allow_short = bool(resolved["allow_short"])
        self.target_gross_exposure = max(0.0, float(resolved["target_gross_exposure"]))
        self.target_vol = max(0.0, float(resolved["target_vol"]))
        self.base_allocation = max(0.0, float(resolved["base_allocation"]))
        self.max_symbol_exposure_pct = max(0.0, float(resolved["max_symbol_exposure_pct"]))
        self.max_order_value = max(0.0, float(resolved["max_order_value"]))
        self.min_price = max(0.0, float(resolved["min_price"]))
        # Ambiguous (DST) hours dropped from both classes: {start-1, end}.
        self._ambiguous_hours = frozenset(
            {(self.cash_start_hour_utc - 1) % 24, self.cash_end_hour_utc % 24}
        )
        closes_size = max(8, self.vol_window + 8)
        days_size = max(8, self.formation_days + 8)
        self._state: dict[str, _State] = {
            symbol: _State(
                closes=deque(maxlen=closes_size),
                day_returns=deque(maxlen=days_size),
            )
            for symbol in self.symbol_list
        }
        self._last_eval_week: tuple[int, int] | None = None
        # Recent decision-bar epochs (seconds) for deterministic bar-spacing
        # inference: the vol-target scalar annualizes the per-bar portfolio vol
        # via sqrt(bars_per_year) derived from the median gap here.
        self._recent_times: deque[float] = deque(maxlen=16)

    # ------------------------------------------------------------------ #
    # session classification
    # ------------------------------------------------------------------ #
    def _classify_hour(self, dt: Any) -> str:
        """Return ``"CASH"`` / ``"OFF"`` / ``"AMBIGUOUS"`` for a bar's UTC hour."""
        if dt.weekday() >= 5:  # weekend -> fully off-session
            return "OFF"
        hour = dt.hour
        if hour in self._ambiguous_hours:
            return "AMBIGUOUS"
        if self.cash_start_hour_utc <= hour < self.cash_end_hour_utc:
            return "CASH"
        return "OFF"

    # ------------------------------------------------------------------ #
    # state
    # ------------------------------------------------------------------ #
    def get_state(self) -> dict[str, Any]:
        return {
            "last_eval_week": (
                list(self._last_eval_week) if self._last_eval_week is not None else None
            ),
            "recent_times": list(self._recent_times),
            "symbol_state": {
                symbol: {
                    "closes": list(item.closes),
                    "day_returns": [[cash, off] for cash, off in item.day_returns],
                    "cur_date": item.cur_date,
                    "cur_cash": item.cur_cash,
                    "cur_off": item.cur_off,
                    "prev_close": item.prev_close,
                    "mode": item.mode,
                    "entry_price": item.entry_price,
                    "decisions_held": int(item.decisions_held),
                    "last_time_key": item.last_time_key,
                    "score": item.score,
                }
                for symbol, item in self._state.items()
            },
        }

    def set_state(self, state: dict[str, Any]) -> None:
        if not isinstance(state, dict):
            return
        self._recent_times.clear()
        for value in _coerce_float_list(state.get("recent_times"))[
            -int(self._recent_times.maxlen or 0) :
        ]:
            self._recent_times.append(value)
        week = state.get("last_eval_week")
        if isinstance(week, (list, tuple)) and len(week) == 2:
            try:
                self._last_eval_week = (int(week[0]), int(week[1]))
            except Exception:
                self._last_eval_week = None
        else:
            self._last_eval_week = None
        raw = state.get("symbol_state")
        if not isinstance(raw, dict):
            return
        for symbol, payload in raw.items():
            if symbol not in self._state or not isinstance(payload, dict):
                continue
            item = self._state[symbol]
            try:
                item.closes.clear()
                maxlen = int(item.closes.maxlen or 0)
                closes = _coerce_float_list(payload.get("closes"))
                for value in closes[-maxlen:] if maxlen else closes:
                    item.closes.append(value)
                item.day_returns.clear()
                dmax = int(item.day_returns.maxlen or 0)
                pairs = _coerce_day_pairs(payload.get("day_returns"))
                for pair in pairs[-dmax:] if dmax else pairs:
                    item.day_returns.append(pair)
                cur_date = payload.get("cur_date")
                item.cur_date = str(cur_date) if cur_date is not None else None
                item.cur_cash = safe_float(payload.get("cur_cash")) or 0.0
                item.cur_off = safe_float(payload.get("cur_off")) or 0.0
                item.prev_close = safe_float(payload.get("prev_close"))
                mode = str(payload.get("mode", "OUT")).upper()
                item.mode = mode if mode in {"OUT", "LONG", "SHORT"} else "OUT"
                item.entry_price = safe_float(payload.get("entry_price"))
                item.decisions_held = _safe_non_negative_int(payload.get("decisions_held"))
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
        dt = _event_datetime_utc(snapshot.time)
        if dt is None:
            return False
        item.last_time_key = key
        date_str = dt.date().isoformat()
        # Roll the per-UTC-day accumulator when the calendar date advances,
        # finalizing the completed day's (cash, off) split first.
        if item.cur_date is not None and date_str != item.cur_date:
            item.day_returns.append((item.cur_cash, item.cur_off))
            item.cur_cash = 0.0
            item.cur_off = 0.0
        item.cur_date = date_str
        if item.prev_close is not None and item.prev_close > 0.0 and close > 0.0:
            bar_return = math.log(close / item.prev_close)
            if math.isfinite(bar_return):
                cls = self._classify_hour(dt)
                if cls == "CASH":
                    item.cur_cash += bar_return
                elif cls == "OFF":
                    item.cur_off += bar_return
        item.prev_close = close
        item.closes.append(close)
        return True

    def calculate_signals_window(self, event: Any, aggregator: Any = None) -> None:
        _ = aggregator
        updated = False
        for symbol in _event_symbols(event, self.symbol_list):
            snapshot = _window_snapshot(event, symbol)
            if snapshot is not None and self._update_symbol(symbol, snapshot):
                updated = True
        if updated:
            self._maybe_evaluate(getattr(event, "time", None))

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
                self._maybe_evaluate(snapshot.time)

    # ------------------------------------------------------------------ #
    # weekly ISO-week decision clock
    # ------------------------------------------------------------------ #
    def _maybe_evaluate(self, event_time: Any) -> None:
        dt = _event_datetime_utc(event_time)
        if dt is None:
            return
        # Record the decision-bar epoch so the vol-target scalar can infer bar
        # spacing.  This hook fires once per updated symbol, so a monotonic guard
        # keeps ``_recent_times`` to one entry per distinct bar.
        epoch = dt.timestamp()
        if not self._recent_times or epoch > self._recent_times[-1]:
            self._recent_times.append(epoch)
        iso = dt.isocalendar()
        week = (int(iso[0]), int(iso[1]))
        if self._last_eval_week is None:
            self._last_eval_week = week
            return
        if week != self._last_eval_week:
            self._last_eval_week = week
            self._evaluate(event_time)

    # ------------------------------------------------------------------ #
    # scoring / selection
    # ------------------------------------------------------------------ #
    def _residual_scores(
        self,
    ) -> tuple[dict[str, float], dict[str, float], dict[str, dict[str, Any]]]:
        """Return ``(residual_by_symbol, vols, metas)`` for the eligible universe."""
        tow: dict[str, float] = {}
        momentum: dict[str, float] = {}
        vols: dict[str, float] = {}
        for symbol, item in self._state.items():
            if len(item.day_returns) < self.min_history_days:
                continue
            window = list(item.day_returns)[-self.formation_days :]
            if len(window) < self.formation_days:
                continue
            cash_series = [cash for cash, _off in window]
            off_series = [off for _cash, off in window]
            # Per-symbol off-session data-quality probe: a perp pinned off-hours
            # has ~zero dispersion in its daily off-session return -> abstain.
            if _dispersion(off_series) <= self.off_variance_eps:
                continue
            mean_off = sum(off_series) / float(len(off_series))
            mean_cash = sum(cash_series) / float(len(cash_series))
            tow_value = mean_off - mean_cash
            mom_value = sum(cash_series) + sum(off_series)  # total trailing log return
            vol = realized_volatility(item.closes, window=self.vol_window)
            if vol is None or vol <= _EPS:
                continue
            tow[symbol] = float(tow_value)
            momentum[symbol] = float(mom_value)
            vols[symbol] = float(vol)

        if len(tow) < self.min_symbols:
            return {}, {}, {}

        ordered = sorted(tow)
        tow_z = _cross_z(tow)
        mom_z = _cross_z(momentum)
        residual_vec = cross_sectional_residualize(
            [tow_z[symbol] for symbol in ordered],
            [[mom_z[symbol] for symbol in ordered]],
        )
        if residual_vec is None or len(residual_vec) != len(ordered):
            return {}, {}, {}
        # Residual entirely explained by momentum (collinear) -> nothing to trade.
        if _dispersion([float(value) for value in residual_vec]) <= _EPS:
            return {}, {}, {}
        residual = {symbol: float(residual_vec[idx]) for idx, symbol in enumerate(ordered)}
        metas: dict[str, dict[str, Any]] = {
            symbol: {
                "tow": float(tow[symbol]),
                "tow_z": float(tow_z[symbol]),
                "momentum_z": float(mom_z[symbol]),
                "tow_residual": float(residual[symbol]),
            }
            for symbol in ordered
        }
        return residual, vols, metas

    def _score_and_select(
        self,
    ) -> tuple[dict[str, tuple[str, float, dict[str, Any]]], dict[str, float]]:
        residual, vols, metas = self._residual_scores()
        if not residual:
            return {}, {}

        # Ascending by residual (symbol tiebreak).  The FADE sign: the TOP
        # residual-TOW quantile is SHORT (over-pumped off-session), the BOTTOM
        # quantile is LONG.  A held name is retained while its rank stays within
        # the quantile boundary plus the hysteresis buffer.
        ordered = sorted(residual, key=lambda symbol: (residual[symbol], symbol))
        count = len(ordered)
        n_side = max(1, int(self.quantile_pct * count))
        if 2 * n_side > count:
            n_side = count // 2
        if n_side < 1:
            return {}, {}
        buffer = self.rank_hysteresis_buffer
        short_core = set(ordered[count - n_side :])
        long_core = set(ordered[:n_side])
        short_hold = set(ordered[count - min(count, n_side + buffer) :])
        long_hold = set(ordered[: min(count, n_side + buffer)])

        targets: dict[str, tuple[str, float, dict[str, Any]]] = {}
        for symbol in ordered:
            mode = self._state[symbol].mode
            score = residual[symbol]
            meta = metas[symbol]
            if symbol in long_core:
                targets[symbol] = ("LONG", score, meta)
            elif self.allow_short and symbol in short_core:
                targets[symbol] = ("SHORT", score, meta)
            elif mode == "LONG" and symbol in long_hold:
                targets[symbol] = ("LONG", score, meta)
            elif mode == "SHORT" and self.allow_short and symbol in short_hold:
                targets[symbol] = ("SHORT", score, meta)
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
        # every vol_i cancels), so the risk-parity weights are horizon-free.
        # The vol-target SCALAR, however, compares ``portfolio_vol`` against an
        # annual-scale ``target_vol`` (0.20): annualize the per-bar estimate via
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
    # emission
    # ------------------------------------------------------------------ #
    def _evaluate(self, event_time: Any) -> None:
        if len(self.symbol_list) < self.min_symbols:
            return
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
                    # Min-hold floor: a would-be exit inside the hold window is
                    # suppressed (turnover discipline); the position ages instead.
                    if item.decisions_held < self.min_hold_decisions:
                        item.decisions_held += 1
                        continue
                    _emit(
                        self.events,
                        strategy_id=_STRATEGY_ID,
                        symbol=symbol,
                        event_time=event_time,
                        signal_type="EXIT",
                        price=price,
                        metadata={"strategy": _STRATEGY_NAME, "reason": "rank_lapsed"},
                    )
                    item.mode = "OUT"
                    item.entry_price = None
                    item.decisions_held = 0
                    item.score = None
                continue
            target_mode, score, meta = target
            if item.mode == target_mode:
                item.decisions_held += 1
                item.score = float(score)
                continue
            # Min-hold floor: a would-be side-flip inside the hold window is
            # suppressed; the current position is kept until the hold clears.
            if item.mode != "OUT" and item.decisions_held < self.min_hold_decisions:
                item.decisions_held += 1
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
                strength=max(0.25, min(3.0, abs(score))),
                price=price,
                metadata=metadata,
            )
            item.mode = target_mode
            item.entry_price = price
            item.decisions_held = 0
            item.score = float(score)


# --------------------------------------------------------------------------- #
# Candidate-wiring hints for the W3 integrator (this lane does NOT wire
# candidates itself -- new-file-only, no shared-file edits).  Admission route is
# `allow_multi_asset=True` at the data-PC handoff: this book is a pure
# cross-sectional long-short (the momentum component is residualized OUT), so it
# is honestly EXCLUDED from any carry/momentum tag-superset allowlist.
# --------------------------------------------------------------------------- #
_SUGGESTED_FAMILY = "cross_sectional"
_SUGGESTED_CANDIDATE_TAGS: tuple[str, ...] = (
    "cross_sectional",
    "session_decomposition",
    "tug_of_war",
    "tradfi_perp",
    "sentiment_fade",
    "low_turnover",
)

# Candidate slice keyed by the 1h bar cadence this lane consumes; each variant
# dict is a pre-registered sweep cell counted toward N_eff at the data-PC.
_OFFSESSION_TUGOFWAR_SLICE: dict[str, tuple[dict[str, Any], ...]] = {
    "1h": (
        {
            "variant": "tow_42d_fade",
            "formation_days": 42,
            "quantile_pct": 0.25,
            "min_hold_decisions": 2,
            "min_symbols": 5,
            "min_history_days": 42,
            "vol_window": 168,
            "allow_short": True,
            "target_gross_exposure": 1.0,
        },
        {
            "variant": "tow_84d_fade",
            "formation_days": 84,
            "quantile_pct": 0.25,
            "min_hold_decisions": 2,
            "min_symbols": 5,
            "min_history_days": 84,
            "vol_window": 168,
            "allow_short": True,
            "target_gross_exposure": 1.0,
        },
    ),
}

__all__ = ["CrossSectionalOffSessionTugOfWarStrategy"]
