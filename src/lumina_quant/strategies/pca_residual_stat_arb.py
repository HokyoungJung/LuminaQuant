"""PCA-eigenportfolio residual s-score statistical arbitrage (research-only).

LINEAGE / ATTRIBUTION.  Independent adaptation of Avellaneda & Lee (2010),
"Statistical arbitrage in the U.S. equities market" -- PCA eigenportfolio
residuals, an OU fit on the cumulative residual, and the resulting s-score used
as the entry/exit statistic.  The same construction sits in the
statistical-arbitrage research scope publicly associated with the Korean quant
educator 아마추어퀀트 (조성현).  This module is NOT a reproduction of anyone's
book, not an endorsement, and carries no performance claim: it re-derives the
publicly described *rules* from the primary literature and wires them into this
repo's event contract.  research_only.

HYPOTHESIS.  Cross-sectional returns are dominated by a handful of common
factors.  After projecting each name onto the top-``n_factors``
eigenportfolios, the leftover (idiosyncratic) return stream cumulates into a
mean-reverting process.  When a name's cumulative residual sits far BELOW its
own OU long-run mean the s-score is strongly negative and the name is cheap
relative to its factor exposure (buy); far ABOVE and the s-score is strongly
positive (sell).  The edge, if any, is the residual reversion -- not direction.

WHAT THE PUBLIC SOURCE STATES vs. AUTHOR'S CHOICES.
* Public source (Avellaneda-Lee): PCA on the correlation matrix of standardized
  returns; eigenportfolio weights = eigenvector / per-name volatility; per-name
  OLS of returns on the factor returns with an intercept; cumulative residual
  fitted with an AR(1)/OU model; ``s = -m / sigma_eq`` (the auxiliary process is
  centred by construction because the regression carries an intercept); an
  asymmetric open/close threshold ladder (open around |s| ~ 1.25, close longs
  earlier than shorts); a half-life filter that discards names whose residual
  reverts too slowly to be tradable.  All of that lives in
  ``lumina_quant.indicators.stat_arb.pca_residual_sscores``.
* AUTHOR'S choices here: the 60-bar estimation window, the daily decision
  cadence, ``max_longs``/``max_shorts`` book caps, equal-notional sizing off a
  ``gross_cap`` budget, the ``max_hold_bars`` age-out, the ``s_stop`` disaster
  stop, the ``none_tolerance_evals`` grace period before flattening a name whose
  s-score stops being estimable, and the ``require_balanced`` pairing switch.
  Every one of those is a repo-side risk control, not a published rule.

DISTINCTNESS.  This is the repo's deliberate PCA lane: the other cross-sectional
books here are eigen-free (rank / z-score / inverse-vol constructions).  This one
is the only sleeve that eigen-decomposes the cross-sectional correlation matrix
(numpy ``eigh``) and trades the regression RESIDUAL rather than a ranked score.

NO LOOKAHEAD.  The panel ends at the latest COMPLETED bar close; the decision is
a close-to-close rebalance evaluated at that close.  Nothing from a later bar
enters the estimate, and the emitted price is the same close the panel ends on.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Any

from lumina_quant.core.plugin_registry import register
from lumina_quant.indicators.common import safe_float
from lumina_quant.indicators.stat_arb import pca_residual_sscores
from lumina_quant.strategies.external_alpha_sleeves import (
    _EPS,
    _Snapshot,
    _emit,
    _event_symbols,
    _market_snapshot,
    _safe_non_negative_int,
    _target_metadata,
    _window_snapshot,
)
from lumina_quant.strategy import Strategy
from lumina_quant.tuning import HyperParam, resolve_params_from_schema
from lumina_quant.utils.timeutil import utc_epoch_ms

_STRATEGY_ID = "pca_residual_stat_arb"
_STRATEGY_NAME = "PcaResidualStatArbStrategy"
_SIDES = {"LONG", "SHORT"}


@dataclass(slots=True)
class _State:
    """Per-symbol rolling closes plus the open-position bookkeeping."""

    closes: deque[float]
    close_times_ms: deque[int]
    last_time_ms: int | None = None
    side: str = "OUT"
    bars_held: int = 0
    entry_s: float | None = None
    none_streak: int = 0
    cooldown_until_eval: int = -1

    def flatten(self) -> None:
        self.side = "OUT"
        self.bars_held = 0
        self.entry_s = None
        self.none_streak = 0


def _log_returns(closes: list[float]) -> list[float] | None:
    """Close-to-close log returns; ``None`` when any step is degenerate."""
    out: list[float] = []
    for idx in range(1, len(closes)):
        prev, cur = closes[idx - 1], closes[idx]
        if prev <= 0.0 or cur <= 0.0:
            return None
        value = math.log(cur / prev)
        if not math.isfinite(value):
            return None
        out.append(value)
    return out


@register("strategy", _STRATEGY_NAME, interface="event_driven")
class PcaResidualStatArbStrategy(Strategy):
    """Avellaneda-Lee PCA residual s-score book over ``bars.symbol_list``.

    Every new bar the strategy rebuilds a ``lookback_bars x N`` log-return panel
    over the names that carry enough history, hands it to
    ``pca_residual_sscores`` (top-``n_factors`` eigenportfolios, per-name OLS,
    OU fit on the cumulative residual) and trades the resulting s-scores with an
    asymmetric ladder: open LONG below ``-s_open``, open SHORT above ``+s_open``,
    close longs once s recovers above ``-s_close_long`` and shorts once s falls
    below ``+s_close_short``.  Names are equal-notional off a ``gross_cap``
    budget, so the book is dollar-neutral whenever the long and short counts
    match.  research_only -- no performance claim.
    """

    decision_cadence_seconds = 86400
    preferred_contract = "market_window"
    uses_timeframe_aggregator = False

    @classmethod
    def get_param_schema(cls) -> dict[str, HyperParam]:
        return {
            "lookback_bars": HyperParam.integer("lookback_bars", default=60, low=10, high=756),
            "n_factors": HyperParam.integer("n_factors", default=1, low=1, high=20),
            "rebalance_bars": HyperParam.integer("rebalance_bars", default=1, low=1, high=60),
            "min_symbols": HyperParam.integer("min_symbols", default=5, low=3, high=512),
            "min_rows": HyperParam.integer("min_rows", default=30, low=8, high=756),
            "max_half_life_bars": HyperParam.floating(
                "max_half_life_bars", default=0.0, low=0.0, high=756.0
            ),
            "s_open": HyperParam.floating("s_open", default=1.25, low=0.25, high=5.0),
            "s_close_long": HyperParam.floating("s_close_long", default=0.50, low=0.0, high=3.0),
            "s_close_short": HyperParam.floating("s_close_short", default=0.75, low=0.0, high=3.0),
            "s_stop": HyperParam.floating("s_stop", default=0.0, low=0.0, high=10.0),
            "max_longs": HyperParam.integer("max_longs", default=3, low=0, high=50),
            "max_shorts": HyperParam.integer("max_shorts", default=3, low=0, high=50),
            "allow_short": HyperParam.boolean("allow_short", default=True, grid=[True, False]),
            "max_hold_bars": HyperParam.integer("max_hold_bars", default=40, low=0, high=2000),
            "none_tolerance_evals": HyperParam.integer(
                "none_tolerance_evals", default=3, low=0, high=100
            ),
            "gross_cap": HyperParam.floating("gross_cap", default=0.60, low=0.0, high=2.0),
            "require_balanced": HyperParam.boolean(
                "require_balanced", default=False, grid=[True, False]
            ),
            "max_position_allocation": HyperParam.floating(
                "max_position_allocation", default=0.20, low=0.0, high=1.0, tunable=False
            ),
            "max_order_value": HyperParam.floating(
                "max_order_value", default=500.0, low=0.0, high=1_000_000.0, tunable=False
            ),
        }

    def __init__(self, bars: Any, events: Any, **params: Any) -> None:
        self.bars = bars
        self.events = events
        self.symbol_list = list(getattr(bars, "symbol_list", []) or [])
        resolved = resolve_params_from_schema(self.get_param_schema(), params, keep_unknown=False)
        self.lookback_bars = max(10, int(resolved["lookback_bars"]))
        self.n_factors = max(1, int(resolved["n_factors"]))
        self.rebalance_bars = max(1, int(resolved["rebalance_bars"]))
        self.min_symbols = max(3, int(resolved["min_symbols"]))
        self.min_rows = max(8, int(resolved["min_rows"]))
        self.max_half_life_bars = max(0.0, float(resolved["max_half_life_bars"]))
        self.s_open = max(0.0, float(resolved["s_open"]))
        self.s_close_long = max(0.0, float(resolved["s_close_long"]))
        self.s_close_short = max(0.0, float(resolved["s_close_short"]))
        self.s_stop = max(0.0, float(resolved["s_stop"]))
        self.max_longs = max(0, int(resolved["max_longs"]))
        self.max_shorts = max(0, int(resolved["max_shorts"]))
        self.allow_short = bool(resolved["allow_short"])
        self.max_hold_bars = max(0, int(resolved["max_hold_bars"]))
        self.none_tolerance_evals = max(0, int(resolved["none_tolerance_evals"]))
        self.gross_cap = max(0.0, float(resolved["gross_cap"]))
        self.require_balanced = bool(resolved["require_balanced"])
        self.max_position_allocation = max(0.0, float(resolved["max_position_allocation"]))
        self.max_order_value = max(0.0, float(resolved["max_order_value"]))
        size = self.lookback_bars + 2
        self._state = {
            symbol: _State(
                closes=deque(maxlen=size),
                close_times_ms=deque(maxlen=size),
            )
            for symbol in self.symbol_list
        }
        self._eval_count = 0
        self._last_eval_time_ms: int | None = None
        self._pending_time_ms: int | None = None
        self._pending_symbols: set[str] = set()
        self._diagnostics = {
            "incomplete_panel": 0,
            "model_empty": 0,
            "model_width_mismatch": 0,
            "out_of_order_or_duplicate": 0,
            "window_timestamp_mismatch": 0,
        }

    # ------------------------------------------------------------------ state

    def get_state(self) -> dict[str, Any]:
        return {
            "eval_count": int(self._eval_count),
            "last_eval_time_ms": self._last_eval_time_ms,
            "closes": {symbol: list(item.closes) for symbol, item in self._state.items()},
            "close_times_ms": {
                symbol: list(item.close_times_ms) for symbol, item in self._state.items()
            },
            "last_times_ms": {symbol: item.last_time_ms for symbol, item in self._state.items()},
            "none_streaks": {symbol: int(item.none_streak) for symbol, item in self._state.items()},
            "cooldown_until_eval": {
                symbol: int(item.cooldown_until_eval)
                for symbol, item in self._state.items()
                if item.cooldown_until_eval >= 0
            },
            "pending_time_ms": self._pending_time_ms,
            "pending_symbols": sorted(self._pending_symbols),
            "diagnostics": dict(self._diagnostics),
            "positions": {
                symbol: {
                    "side": item.side,
                    "bars_held": int(item.bars_held),
                    "entry_s": item.entry_s,
                }
                for symbol, item in self._state.items()
                if item.side in _SIDES
            },
        }

    def set_state(self, state: dict[str, Any]) -> None:
        if not isinstance(state, dict):
            return
        self._eval_count = _safe_non_negative_int(state.get("eval_count"))
        try:
            last_eval = state.get("last_eval_time_ms")
            self._last_eval_time_ms = None if last_eval is None else max(0, int(last_eval))
        except TypeError, ValueError:
            self._last_eval_time_ms = None
        closes = state.get("closes")
        close_times = state.get("close_times_ms")
        if isinstance(closes, dict):
            for symbol, values in closes.items():
                item = self._state.get(symbol)
                if item is None:
                    continue
                item.closes.clear()
                item.close_times_ms.clear()
                raw_values = list(values or [])[-int(item.closes.maxlen or 0) :]
                raw_times = (
                    list(close_times.get(symbol) or [])[-int(item.close_times_ms.maxlen or 0) :]
                    if isinstance(close_times, dict)
                    else []
                )
                if len(raw_values) != len(raw_times):
                    continue
                for raw_time, value in zip(raw_times, raw_values, strict=True):
                    parsed = safe_float(value)
                    try:
                        parsed_time = int(raw_time)
                    except TypeError, ValueError:
                        item.closes.clear()
                        item.close_times_ms.clear()
                        break
                    if (
                        parsed is None
                        or parsed <= 0.0
                        or parsed_time < 0
                        or (item.close_times_ms and parsed_time <= item.close_times_ms[-1])
                    ):
                        item.closes.clear()
                        item.close_times_ms.clear()
                        break
                    item.closes.append(parsed)
                    item.close_times_ms.append(parsed_time)
                item.last_time_ms = item.close_times_ms[-1] if item.close_times_ms else None
        for item in self._state.values():
            item.side = "OUT"
            item.bars_held = 0
            item.entry_s = None
            item.none_streak = 0
            item.cooldown_until_eval = -1
        positions = state.get("positions")
        if isinstance(positions, dict):
            for symbol, payload in positions.items():
                item = self._state.get(symbol)
                if item is None or not isinstance(payload, dict):
                    continue
                side = str(payload.get("side", "OUT")).upper()
                if side not in _SIDES:
                    continue
                item.side = side
                item.bars_held = _safe_non_negative_int(payload.get("bars_held"))
                item.entry_s = safe_float(payload.get("entry_s"))
        streaks = state.get("none_streaks")
        if isinstance(streaks, dict):
            for symbol, value in streaks.items():
                item = self._state.get(symbol)
                if item is not None:
                    item.none_streak = _safe_non_negative_int(value)
        cooldowns = state.get("cooldown_until_eval")
        if isinstance(cooldowns, dict):
            for symbol, value in cooldowns.items():
                item = self._state.get(symbol)
                if item is not None:
                    try:
                        item.cooldown_until_eval = max(-1, int(value))
                    except TypeError, ValueError:
                        item.cooldown_until_eval = -1
        try:
            pending = state.get("pending_time_ms")
            self._pending_time_ms = None if pending is None else max(0, int(pending))
        except TypeError, ValueError:
            self._pending_time_ms = None
        raw_pending_symbols = state.get("pending_symbols")
        self._pending_symbols = (
            {str(symbol) for symbol in raw_pending_symbols if str(symbol) in self._state}
            if isinstance(raw_pending_symbols, list)
            else set()
        )
        if self._pending_time_ms is None:
            self._pending_symbols.clear()
        diagnostics = state.get("diagnostics")
        if isinstance(diagnostics, dict):
            for key in self._diagnostics:
                self._diagnostics[key] = _safe_non_negative_int(diagnostics.get(key))

    # ------------------------------------------------------------- ingestion

    def _update_symbol(self, symbol: str, snapshot: _Snapshot) -> int | None:
        item = self._state.get(symbol)
        if item is None:
            return None
        close = safe_float(snapshot.close)
        if close is None or close <= 0.0:
            return None
        try:
            timestamp_ms = utc_epoch_ms(snapshot.time)
        except TypeError, ValueError:
            return None
        if type(timestamp_ms) is not int or timestamp_ms < 0:
            return None
        if item.last_time_ms is not None and timestamp_ms <= item.last_time_ms:
            self._diagnostics["out_of_order_or_duplicate"] += 1
            return None
        item.last_time_ms = timestamp_ms
        item.close_times_ms.append(timestamp_ms)
        item.closes.append(float(close))
        return timestamp_ms

    def calculate_signals_window(self, event: Any, aggregator: Any = None) -> None:
        _ = aggregator
        updated_times: list[int] = []
        for symbol in _event_symbols(event, self.symbol_list):
            snapshot = _window_snapshot(event, symbol)
            if snapshot is not None:
                timestamp_ms = self._update_symbol(symbol, snapshot)
                if timestamp_ms is not None:
                    updated_times.append(timestamp_ms)
        if not updated_times:
            return
        if len(set(updated_times)) != 1:
            self._diagnostics["window_timestamp_mismatch"] += 1
            return
        event_time_ms = updated_times[0]
        if event_time_ms == self._last_eval_time_ms:
            return
        self._last_eval_time_ms = event_time_ms
        self._pending_time_ms = None
        self._pending_symbols.clear()
        self._evaluate(getattr(event, "time", None), event_time_ms)

    def calculate_signals(self, event: Any) -> None:
        event_type = str(getattr(event, "type", "")).upper()
        if event_type == "MARKET_WINDOW":
            self.calculate_signals_window(event, None)
            return
        if event_type != "MARKET":
            return
        symbol = str(getattr(event, "symbol", ""))
        if symbol not in self._state:
            return
        snapshot = _market_snapshot(event)
        if snapshot is None:
            return
        timestamp_ms = self._update_symbol(symbol, snapshot)
        if timestamp_ms is None:
            return
        if self._pending_time_ms != timestamp_ms:
            self._pending_time_ms = timestamp_ms
            self._pending_symbols.clear()
        self._pending_symbols.add(symbol)
        if self._pending_symbols != set(self.symbol_list):
            return
        if timestamp_ms == self._last_eval_time_ms:
            return
        self._last_eval_time_ms = timestamp_ms
        self._pending_time_ms = None
        self._pending_symbols.clear()
        self._evaluate(snapshot.time, timestamp_ms)

    # ---------------------------------------------------------------- model

    def _panel(self, expected_time_ms: int) -> tuple[list[str], list[list[float]]]:
        """Aligned ``lookback_bars x N`` log-return panel over eligible names.

        A name is eligible once it carries ``lookback_bars + 1`` closes; the row
        ends at the latest COMPLETED bar close, which is exactly the close this
        rebalance trades against.

        Every selected column contains the same exact timestamp keys and ends at
        ``expected_time_ms``. Sparse names are deterministically removed until a
        complete panel exists; no stale column is admitted.
        """
        needed = self.lookback_bars + 1
        minimum = max(self.min_symbols, self.n_factors + 2)
        observations: dict[str, dict[int, float]] = {}
        for symbol in self.symbol_list:
            item = self._state.get(symbol)
            if (
                item is None
                or len(item.closes) < needed
                or len(item.close_times_ms) != len(item.closes)
                or item.last_time_ms != expected_time_ms
            ):
                continue
            observations[symbol] = dict(zip(item.close_times_ms, item.closes, strict=True))
        active = sorted(observations)
        # ponytail: this deterministic O(n²) sparse-panel reducer is adequate for
        # the bounded research universe; replace with a timestamp incidence index
        # if the universe grows into the thousands.
        while len(active) >= minimum:
            common = set(observations[active[0]])
            for symbol in active[1:]:
                common.intersection_update(observations[symbol])
            ordered = [
                timestamp
                for timestamp in self._state[active[0]].close_times_ms
                if timestamp in common
            ]
            if len(ordered) >= needed and ordered[-1] == expected_time_ms:
                selected_times = ordered[-needed:]
                columns = [
                    [observations[symbol][timestamp] for timestamp in selected_times]
                    for symbol in active
                ]
                return active, [
                    [math.log(column[index] / column[index - 1]) for column in columns]
                    for index in range(1, needed)
                ]
            if len(active) == minimum:
                break
            removal = max(
                active,
                key=lambda candidate: (
                    len(
                        set.intersection(
                            *(set(observations[symbol]) for symbol in active if symbol != candidate)
                        )
                    ),
                    candidate,
                ),
            )
            active.remove(removal)
        self._diagnostics["incomplete_panel"] += 1
        return [], []

    def _scores(self, expected_time_ms: int) -> tuple[dict[str, float | None], int]:
        symbols, rows = self._panel(expected_time_ms)
        if not symbols:
            return {}, 0
        values = pca_residual_sscores(
            rows,
            n_factors=self.n_factors,
            max_half_life_bars=self.max_half_life_bars or None,
            min_rows=self.min_rows,
        )
        if not values:
            self._diagnostics["model_empty"] += 1
            return {}, 0
        if len(values) != len(symbols):
            self._diagnostics["model_width_mismatch"] += 1
            return {}, 0
        return dict(zip(symbols, values, strict=True)), len(symbols)

    def _per_name_allocation(self) -> float:
        """Equal-notional slice of ``gross_cap`` across the full book capacity.

        The denominator is the STATIC capacity ``max_longs + max_shorts`` (not the
        realized count), so a name's notional never depends on how many other
        names happen to be on at the time -- gross rises and falls with the book
        instead of each name being levered up when the book is thin.
        """
        slots = max(1, self.max_longs + self.max_shorts)
        alloc = self.gross_cap / float(slots)
        if self.max_position_allocation > 0.0:
            alloc = min(alloc, self.max_position_allocation)
        return max(0.0, alloc)

    def _exit_reason(self, item: _State, score: float | None) -> str:
        if score is None:
            if self.none_tolerance_evals > 0 and item.none_streak >= self.none_tolerance_evals:
                return "s_unestimable"
        else:
            if self.s_stop > 0.0 and abs(score) >= self.s_stop:
                return "s_stop"
            if item.side == "LONG" and score > -self.s_close_long:
                return "residual_reverted"
            if item.side == "SHORT" and score < self.s_close_short:
                return "residual_reverted"
        if self.max_hold_bars > 0 and item.bars_held >= self.max_hold_bars:
            return "max_hold"
        return ""

    def _evaluate(self, event_time: Any, event_time_ms: int) -> None:
        self._eval_count += 1
        # Age every open name once per BAR (not once per model run) so
        # ``max_hold_bars`` keeps its plain meaning under a coarse
        # ``rebalance_bars``.  ponytail: with rebalance_bars > 1 the age-out
        # can only be ACTED on at the next model run, so a position may
        # overstay by up to rebalance_bars - 1 bars.
        for item in self._state.values():
            if item.side in _SIDES:
                item.bars_held += 1
        if self._eval_count % self.rebalance_bars != 0:
            return
        scores, panel_size = self._scores(event_time_ms)
        # Run the exit ladder even when the panel could not be built at all
        # (universe shrank, history gaps): every held name then scores as
        # unestimable, so ``none_tolerance_evals`` and ``max_hold_bars`` can
        # still flatten the book instead of stranding it.
        closed = self._close_positions(scores, panel_size, event_time)
        if not scores:
            return
        self._open_positions(scores, panel_size, event_time, excluded=closed)

    def _close_positions(
        self, scores: dict[str, float | None], panel_size: int, event_time: Any
    ) -> set[str]:
        planned: dict[str, str] = {}
        for symbol, item in self._state.items():
            if item.side not in _SIDES:
                continue
            score = scores.get(symbol)
            item.none_streak = item.none_streak + 1 if score is None else 0
            reason = self._exit_reason(item, score)
            if reason:
                planned[symbol] = reason
        if self.require_balanced:
            remaining_longs = sorted(
                symbol
                for symbol, item in self._state.items()
                if item.side == "LONG" and symbol not in planned
            )
            remaining_shorts = sorted(
                symbol
                for symbol, item in self._state.items()
                if item.side == "SHORT" and symbol not in planned
            )
            if len(remaining_longs) > len(remaining_shorts):
                for symbol in remaining_longs[len(remaining_shorts) :]:
                    planned[symbol] = "balance_reduction"
            elif len(remaining_shorts) > len(remaining_longs):
                for symbol in remaining_shorts[len(remaining_longs) :]:
                    planned[symbol] = "balance_reduction"
        for symbol in self.symbol_list:
            reason = planned.get(symbol)
            if reason is None:
                continue
            item = self._state[symbol]
            score = scores.get(symbol)
            # ponytail: EXIT is whole-position by portfolio contract (there are no
            # partial exits), so the ladder scales OUT in one step.
            _emit(
                self.events,
                strategy_id=_STRATEGY_ID,
                symbol=symbol,
                event_time=event_time,
                signal_type="EXIT",
                price=item.closes[-1] if item.closes else None,
                metadata={
                    "strategy": _STRATEGY_NAME,
                    "reason": reason,
                    "s_score": score,
                    "n_factors": int(self.n_factors),
                    "panel_size": int(panel_size),
                    "entry_s": item.entry_s,
                    "bars_held": int(item.bars_held),
                },
            )
            item.flatten()
            if reason in {"s_stop", "max_hold"}:
                item.cooldown_until_eval = self._eval_count + 1
        return set(planned)

    def _open_positions(
        self,
        scores: dict[str, float | None],
        panel_size: int,
        event_time: Any,
        *,
        excluded: set[str],
    ) -> None:
        held_long = sum(1 for item in self._state.values() if item.side == "LONG")
        held_short = sum(1 for item in self._state.values() if item.side == "SHORT")
        long_slots = max(0, self.max_longs - held_long)
        short_slots = max(0, self.max_shorts - held_short) if self.allow_short else 0
        candidates = [
            (symbol, float(score))
            for symbol, score in scores.items()
            if score is not None
            and symbol not in excluded
            and self._state[symbol].side == "OUT"
            and self._state[symbol].cooldown_until_eval < self._eval_count
        ]
        # Most negative first for longs, most positive first for shorts; the
        # symbol is the deterministic tiebreak.
        longs = sorted(
            [(symbol, score) for symbol, score in candidates if score < -self.s_open],
            key=lambda pair: (pair[1], pair[0]),
        )[:long_slots]
        shorts = sorted(
            [(symbol, score) for symbol, score in candidates if score > self.s_open],
            key=lambda pair: (-pair[1], pair[0]),
        )[:short_slots]
        if self.require_balanced:
            # Strict pairing: only open in matched long/short counts, so the new
            # notional is dollar-neutral on its own rather than only in aggregate.
            paired = min(len(longs), len(shorts))
            longs, shorts = longs[:paired], shorts[:paired]
        alloc = self._per_name_allocation()
        if alloc <= 0.0:
            # A zero allocation would emit an UNSIZED entry that the engine
            # resizes to its own default; refuse to open instead.
            return
        for side, book in (("LONG", longs), ("SHORT", shorts)):
            for symbol, score in book:
                item = self._state[symbol]
                price = item.closes[-1] if item.closes else None
                metadata = _target_metadata(
                    strategy=_STRATEGY_NAME,
                    target_allocation=alloc,
                    max_order_value=self.max_order_value,
                    s_score=float(score),
                    n_factors=int(self.n_factors),
                    panel_size=int(panel_size),
                    reason="open_long" if side == "LONG" else "open_short",
                )
                _emit(
                    self.events,
                    strategy_id=_STRATEGY_ID,
                    symbol=symbol,
                    event_time=event_time,
                    signal_type=side,
                    strength=max(0.25, min(3.0, abs(score) / max(self.s_open, _EPS))),
                    price=price,
                    metadata=metadata,
                )
                item.side = side
                item.bars_held = 0
                item.entry_s = float(score)
                item.none_streak = 0


__all__ = ["PcaResidualStatArbStrategy"]
