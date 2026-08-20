"""Cross-sectional open-interest growth-pressure sleeve (weekly positioning flow).

``OpenInterestGrowthPressureStrategy`` implements proposal ``oi-growth-pressure-xs``:
growth in futures open interest is a procyclical speculative-demand flow that
POSITIVELY predicts returns at a multi-week horizon (price pressure from capital
inflow into the contract, distinct from past price moves).  Per symbol, the sleeve
tracks the trailing ``delta_oi_window_days`` (7 UTC days, pre-registered) change in
OI NOTIONAL (``open_interest * mark_price`` from the None-tolerant feature-lookup
cascade), normalized by the trailing 7d dollar volume, cross-sectionally z-scored
within the OI-COVERED universe, and residualized (single-regressor Gram-Schmidt via
the shared ``cross_sectional_residualize`` primitive) against the 7d price-momentum
z so the sleeve trades positioning flow, NOT momentum in disguise.  It LONGS the
top residual quintile and SHORTS the bottom quintile on a weekly ISO-week clock
with a hard ``min_hold_decisions`` floor (5 decisions), a rank-hysteresis band,
and inverse-realized-vol sizing.

SIGN PRE-REGISTRATION.  The sign is POSITIVE continuation (Hong-Yogo) at the
1-4 week horizon; the crowding-fade reading (high OI + high funding -> fade) is
explicitly NOT fitted and NOT flipped to post-hoc.  ``DELTA_OI_WINDOW_DAYS_FALLBACK
= 14`` is the single pre-registered fallback variant constant, declared now (module
constant), not tuned later: it is used only if weekly one-way churn exceeds ~30% at
data-PC measurement.

COVERAGE HANDLING (two distinct gates).

1. STEP-0 PRE-BACKTEST KILL GATE (group level, runs BEFORE any backtest spend):
   the OI coverage audit on the data-PC -- fraction of symbol-days since 2025-03
   with OI points present, per group (10 crypto / 100 TradFi); ``>=90%``
   symbol-day coverage per group is required, ``<90%`` coverage on a group kills
   that group's lane immediately (killer-iii pre-check).  The dual-purpose audit
   CLI lives at ``scripts/research/audit_liquidation_feature_coverage.py``.
2. IN-WINDOW PER-SYMBOL COVERAGE FLOOR (runtime, deterministic): a symbol whose
   formation window holds fewer than ``min_oi_coverage`` (0.80) non-None
   OI-notional day samples is EXCLUDED from that week's rank -- deterministic
   exclusion, never interpolation and never a raise.

EXPECTED NULL (verbatim pre-registration): "dOI is (a) too collinear with
contemporaneous price momentum/volume to carry incremental IC once residualized,
or (b) the continuation sign fails on crypto perps where OI growth marks crowding
tops -- residual IC <= 0 either way."

FALSIFYING MEASUREMENT (verbatim pre-registration): "Step 0 (before ANY
backtest): OI coverage audit on data-PC -- fraction of symbol-days since 2025-03
with >=80% of 5m OI points present, per group (10 crypto / 100 TradFi); <90%
coverage on a group kills that group's lane immediately (killer-iii pre-check).
Then the single alpha measurement: validation-forward weekly rank IC of
momentum-residualized dOI z; IC <= 0 at the pre-registered positive sign = dead,
no sign flip allowed."

THEORY / PROVENANCE
-------------------
- Hong & Yogo (2012), *JFE* 105(3): "What does futures market interest tell us
  about the macroeconomy and asset prices?" -- open-interest growth predicts
  futures returns, stronger than past prices, in commodity/financial futures.
- Post-2022 crypto strand: OI/positioning predictability studies in crypto
  futures (2023-2024, weaker/scattered than the equity anchors -- flagged
  honestly).
- Repo-ledger support: the ``OpenInterestConfirmation`` rider already validated
  OI as a live, populated feature stream in this exact store.

NEAREST INCUMBENTS: ``OpenInterestConfirmationRider`` (TS gate: OI confirms the
SAME symbol's price move), ``PerpCrowdingCarry`` (TS funding-led crowding fade),
``DerivativesFlowSqueeze`` (TS event composite).  No registered class produces a
cross-sectional ranking from OI at all; the tradable object here is RELATIVE
positioning flow across contracts, uses no funding input, and is orthogonalized
to price momentum ex-ante.

NEAREST GRAVEYARD: #7 feature-coverage starvation -- pre-empted by the hard
step-0 coverage audit as an explicit kill gate BEFORE spend (OI, unlike
BBO/depth, is a dense 5m column in the store per the data-surface contract);
#13 HTF momentum/crowding sweeps (RPT<10bps) -- pre-empted because the signal is
NOT a price-momentum variant (residualized against momentum by construction) and
cadence is weekly with min-hold, not an HTF sweep.

Every entry stamps ``intended_hold_seconds`` (~2 weeks) so the engine-side
funding-entry guard and hold accounting can read the sleeve's ex-ante horizon.
Feature access is the 3-tier None-tolerant cascade shared with
``strategies/external_alpha_sleeves.py`` (8h staleness respected by the lookup);
a missing feature degrades to a missing day sample, never a raise.  The module
is data-local (no I/O), pure Python/``math`` only, and never raises from
``calculate_signals``.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Any

from lumina_quant.core.plugin_registry import register
from lumina_quant.indicators.alpha_features import realized_volatility
from lumina_quant.indicators.annualization import (
    annualize_per_bar_vol,
    bars_per_year_from_spacing,
)
from lumina_quant.indicators.common import safe_float, time_key
from lumina_quant.indicators.cross_sectional_residualize import cross_sectional_residualize
from lumina_quant.strategies.external_alpha_sleeves import (
    _EPS,
    _Snapshot,
    _emit,
    _event_datetime_utc,
    _event_symbols,
    _extract_feature,
    _market_snapshot,
    _safe_non_negative_int,
    _target_metadata,
    _window_snapshot,
)
from lumina_quant.strategy import Strategy
from lumina_quant.tuning import HyperParam, resolve_params_from_schema

_STRATEGY_ID = "oi_growth_pressure_xs"
_STRATEGY_NAME = "OpenInterestGrowthPressureStrategy"

# Pre-registered formation constants (module-level, NOT tunable grids).
DELTA_OI_WINDOW_DAYS = 7
# Single pre-registered fallback variant (declared now, not tuned later): used
# only if weekly one-way churn exceeds ~30% at data-PC measurement.
DELTA_OI_WINDOW_DAYS_FALLBACK = 14
# Ex-ante intended holding period stamped on every entry (~2 weeks).
INTENDED_HOLD_SECONDS = 14 * 86_400


@dataclass(slots=True)
class _DayRecord:
    """One finalized UTC day: latest OI notional (may be None), $vol, day close."""

    oi_notional: float | None
    dollar_volume: float
    close: float


@dataclass(slots=True)
class _State:
    # Trailing closes (realized-vol sizing + last price for emission).
    closes: deque[float]
    # Finalized per-UTC-day records, oldest -> newest.
    days: deque[_DayRecord]
    cur_date: str | None = None
    cur_oi_notional: float | None = None
    cur_dollar_volume: float = 0.0
    cur_close: float | None = None
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


def _coerce_day_records(value: Any) -> list[_DayRecord]:
    """Coerce a serialized ``[[oi|None, dollar_volume, close], ...]``; never raises."""
    if not isinstance(value, (list, tuple)):
        return []
    out: list[_DayRecord] = []
    for item in value:
        if not isinstance(item, (list, tuple)) or len(item) < 3:
            continue
        dollar_volume = safe_float(item[1])
        close = safe_float(item[2])
        if dollar_volume is None or close is None:
            continue
        out.append(
            _DayRecord(
                oi_notional=safe_float(item[0]),
                dollar_volume=max(0.0, float(dollar_volume)),
                close=float(close),
            )
        )
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


@register("strategy", "OpenInterestGrowthPressureStrategy", interface="event_driven")
class OpenInterestGrowthPressureStrategy(Strategy):
    """Long-short XS continuation on volume-normalized 7d OI-notional growth.

    See the module docstring for the full pre-registration (mechanism, sign,
    EXPECTED NULL, falsifying measurement, coverage gates, and the fallback
    variant constant).  Reads local event/bar OHLCV plus the ``open_interest``
    and ``mark_price`` feature columns through the shared None-tolerant cascade,
    performs no I/O, and never raises from ``calculate_signals``.
    """

    decision_cadence_seconds = 3600
    preferred_contract = "market_window"
    uses_timeframe_aggregator = False
    required_features = ("open_interest", "mark_price")

    @classmethod
    def get_param_schema(cls) -> dict[str, HyperParam]:
        return {
            # Pre-registered 7d primary; the ONLY sanctioned alternative is the
            # declared DELTA_OI_WINDOW_DAYS_FALLBACK=14 module constant.
            "delta_oi_window_days": HyperParam.integer(
                "delta_oi_window_days", default=DELTA_OI_WINDOW_DAYS, low=2, high=60, tunable=False
            ),
            "quantile_pct": HyperParam.floating(
                "quantile_pct", default=0.20, low=0.02, high=0.50, tunable=False
            ),
            "min_symbols": HyperParam.integer("min_symbols", default=5, low=2, high=512),
            "min_history_days": HyperParam.integer(
                "min_history_days", default=DELTA_OI_WINDOW_DAYS + 1, low=2, high=100000
            ),
            # In-window per-symbol OI coverage floor: below it the symbol is
            # deterministically EXCLUDED from that week's rank (see docstring).
            "min_oi_coverage": HyperParam.floating(
                "min_oi_coverage", default=0.80, low=0.0, high=1.0, tunable=False
            ),
            "min_hold_decisions": HyperParam.integer(
                "min_hold_decisions", default=5, low=0, high=100000
            ),
            "rank_hysteresis_buffer": HyperParam.integer(
                "rank_hysteresis_buffer", default=1, low=0, high=512
            ),
            "vol_window": HyperParam.integer("vol_window", default=168, low=2, high=20000),
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
        self.symbol_list = list(getattr(self.bars, "symbol_list", []) or [])
        resolved = resolve_params_from_schema(self.get_param_schema(), params, keep_unknown=False)
        self.delta_oi_window_days = max(2, int(resolved["delta_oi_window_days"]))
        self.quantile_pct = min(0.5, max(0.0, float(resolved["quantile_pct"])))
        self.min_symbols = max(2, int(resolved["min_symbols"]))
        self.min_history_days = max(
            self.delta_oi_window_days + 1, int(resolved["min_history_days"])
        )
        self.min_oi_coverage = min(1.0, max(0.0, float(resolved["min_oi_coverage"])))
        self.min_hold_decisions = max(0, int(resolved["min_hold_decisions"]))
        self.rank_hysteresis_buffer = max(0, int(resolved["rank_hysteresis_buffer"]))
        self.vol_window = max(2, int(resolved["vol_window"]))
        self.allow_short = bool(resolved["allow_short"])
        self.target_gross_exposure = max(0.0, float(resolved["target_gross_exposure"]))
        self.target_vol = max(0.0, float(resolved["target_vol"]))
        self.base_allocation = max(0.0, float(resolved["base_allocation"]))
        self.max_symbol_exposure_pct = max(0.0, float(resolved["max_symbol_exposure_pct"]))
        self.max_order_value = max(0.0, float(resolved["max_order_value"]))
        self.min_price = max(0.0, float(resolved["min_price"]))
        closes_size = max(8, self.vol_window + 8)
        days_size = max(8, max(self.delta_oi_window_days + 1, self.min_history_days) + 8)
        self._state: dict[str, _State] = {
            symbol: _State(
                closes=deque(maxlen=closes_size),
                days=deque(maxlen=days_size),
            )
            for symbol in self.symbol_list
        }
        self._last_eval_week: tuple[int, int] | None = None
        # Recent decision-bar epochs (seconds) for deterministic bar-spacing
        # inference; the vol-target scalar annualizes per-bar portfolio vol via
        # sqrt(bars_per_year) derived from the median gap here.
        self._recent_times: deque[float] = deque(maxlen=16)

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
                    "days": [
                        [record.oi_notional, record.dollar_volume, record.close]
                        for record in item.days
                    ],
                    "cur_date": item.cur_date,
                    "cur_oi_notional": item.cur_oi_notional,
                    "cur_dollar_volume": item.cur_dollar_volume,
                    "cur_close": item.cur_close,
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
                item.days.clear()
                dmax = int(item.days.maxlen or 0)
                records = _coerce_day_records(payload.get("days"))
                for record in records[-dmax:] if dmax else records:
                    item.days.append(record)
                cur_date = payload.get("cur_date")
                item.cur_date = str(cur_date) if cur_date is not None else None
                item.cur_oi_notional = safe_float(payload.get("cur_oi_notional"))
                item.cur_dollar_volume = max(
                    0.0, safe_float(payload.get("cur_dollar_volume")) or 0.0
                )
                item.cur_close = safe_float(payload.get("cur_close"))
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
    def _update_symbol(self, symbol: str, snapshot: _Snapshot, event: Any) -> bool:
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
        # finalizing the completed day's record first (a day WITHOUT any valid
        # OI/mark sample finalizes with ``oi_notional=None`` -- the coverage
        # floor downstream handles it; never interpolated, never raised).
        if item.cur_date is not None and date_str != item.cur_date:
            item.days.append(
                _DayRecord(
                    oi_notional=item.cur_oi_notional,
                    dollar_volume=item.cur_dollar_volume,
                    close=item.cur_close if item.cur_close is not None else close,
                )
            )
            item.cur_oi_notional = None
            item.cur_dollar_volume = 0.0
        item.cur_date = date_str
        item.cur_close = close
        volume = safe_float(snapshot.volume)
        if volume is not None and volume > 0.0:
            item.cur_dollar_volume += close * volume
        # None-tolerant feature cascade: event attr -> feature lookup -> bar value.
        oi = _extract_feature(self.bars, event, symbol, "open_interest")
        mark = _extract_feature(self.bars, event, symbol, "mark_price")
        if oi is not None and mark is not None and oi >= 0.0 and mark > 0.0:
            notional = oi * mark
            if math.isfinite(notional):
                item.cur_oi_notional = notional
        item.closes.append(close)
        return True

    def calculate_signals_window(self, event: Any, aggregator: Any = None) -> None:
        _ = aggregator
        updated = False
        for symbol in _event_symbols(event, self.symbol_list):
            snapshot = _window_snapshot(event, symbol)
            if snapshot is not None and self._update_symbol(symbol, snapshot, event):
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
            if snapshot is not None and self._update_symbol(str(symbol), snapshot, event):
                self._maybe_evaluate(snapshot.time)

    # ------------------------------------------------------------------ #
    # weekly ISO-week decision clock
    # ------------------------------------------------------------------ #
    def _maybe_evaluate(self, event_time: Any) -> None:
        dt = _event_datetime_utc(event_time)
        if dt is None:
            return
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
        """Return ``(residual_by_symbol, vols, metas)`` for the OI-covered universe."""
        pressure: dict[str, float] = {}
        momentum: dict[str, float] = {}
        vols: dict[str, float] = {}
        coverage_by_symbol: dict[str, float] = {}
        window_len = self.delta_oi_window_days + 1
        for symbol, item in self._state.items():
            if len(item.days) < self.min_history_days:
                continue
            window = list(item.days)[-window_len:]
            if len(window) < window_len:
                continue
            samples = [
                (idx, record.oi_notional)
                for idx, record in enumerate(window)
                if record.oi_notional is not None
            ]
            # In-window coverage floor -> deterministic exclusion (documented).
            coverage = len(samples) / float(window_len)
            if coverage < self.min_oi_coverage or len(samples) < 2:
                continue
            first_idx, first_oi = samples[0]
            last_idx, last_oi = samples[-1]
            if last_idx <= first_idx:
                continue
            delta_oi = float(last_oi) - float(first_oi)
            # Normalizer: trailing dollar volume over the formation days
            # (the base day excluded -- it anchors the OI level, not the flow).
            dollar_volume = sum(record.dollar_volume for record in window[1:])
            if dollar_volume <= _EPS:
                continue
            first_close = window[0].close
            last_close = window[-1].close
            if first_close <= 0.0 or last_close <= 0.0:
                continue
            mom_value = math.log(last_close / first_close)
            if not math.isfinite(mom_value):
                continue
            vol = realized_volatility(item.closes, window=self.vol_window)
            if vol is None or vol <= _EPS:
                continue
            raw = delta_oi / dollar_volume
            if not math.isfinite(raw):
                continue
            pressure[symbol] = float(raw)
            momentum[symbol] = float(mom_value)
            vols[symbol] = float(vol)
            coverage_by_symbol[symbol] = float(coverage)

        if len(pressure) < self.min_symbols:
            return {}, {}, {}

        ordered = sorted(pressure)
        pressure_z = _cross_z(pressure)
        mom_z = _cross_z(momentum)
        residual_vec = cross_sectional_residualize(
            [pressure_z[symbol] for symbol in ordered],
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
                "delta_oi_norm": float(pressure[symbol]),
                "delta_oi_z": float(pressure_z[symbol]),
                "momentum_z": float(mom_z[symbol]),
                "oi_residual": float(residual[symbol]),
                "oi_coverage": float(coverage_by_symbol[symbol]),
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

        # Ascending by residual (symbol tiebreak).  CONTINUATION sign
        # (pre-registered POSITIVE, no flip): the TOP residual-dOI quantile is
        # LONG (speculative inflow pressure), the BOTTOM quantile is SHORT.  A
        # held name is retained while its rank stays within the quantile
        # boundary plus the hysteresis buffer.
        ordered = sorted(residual, key=lambda symbol: (residual[symbol], symbol))
        count = len(ordered)
        n_side = max(1, int(self.quantile_pct * count))
        if 2 * n_side > count:
            n_side = count // 2
        if n_side < 1:
            return {}, {}
        buffer = self.rank_hysteresis_buffer
        long_core = set(ordered[count - n_side :])
        short_core = set(ordered[:n_side])
        long_hold = set(ordered[count - min(count, n_side + buffer) :])
        short_hold = set(ordered[: min(count, n_side + buffer)])

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
        # ``portfolio_vol`` is the inverse-vol-weighted PER-BAR vol; annualize it
        # via sqrt(bars_per_year) inferred from the observed bar spacing before
        # comparing against the annual-scale ``target_vol``, otherwise the
        # Moreira-Muir clamp is inert.  Unknown spacing -> pass-through (1.0).
        portfolio_vol = sum((inv[symbol] / total_inv) * vols[symbol] for symbol in inv)
        scalar = 1.0
        if self.target_vol > 0.0 and portfolio_vol > _EPS:
            bars_per_year = bars_per_year_from_spacing(self._recent_times)
            portfolio_vol_ann = annualize_per_bar_vol(portfolio_vol, bars_per_year)
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
            if alloc <= 0.0:
                # v5 zero-alloc trap: a zero/negative computed allocation emits
                # NOTHING (the engine would otherwise resize the entry to its
                # DEFAULT allocation).  If a side-flip EXIT was just emitted,
                # drop to OUT so state matches it.
                if item.mode != "OUT":
                    item.mode = "OUT"
                    item.entry_price = None
                    item.decisions_held = 0
                    item.score = None
                continue
            metadata = _target_metadata(
                strategy=_STRATEGY_NAME,
                target_allocation=alloc,
                max_order_value=self.max_order_value,
                score=float(score),
                target_mode=target_mode,
                inverse_vol_weight=weight,
                vol_target_scalar=float(scalar),
                intended_hold_seconds=INTENDED_HOLD_SECONDS,
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
# Candidate-wiring exports for the orchestrator (this module does NOT wire
# candidates itself -- new-file-only, no shared-file edits).  Admission route is
# ``allow_multi_asset=True`` at the data-PC handoff: this book is a pure
# cross-sectional long-short (the momentum component is residualized OUT).
# --------------------------------------------------------------------------- #
_SUGGESTED_CANDIDATE_TAGS: tuple[str, ...] = (
    "cross_sectional",
    "positioning_flow",
    "open_interest",
    "derivatives_feature",
    "low_turnover",
)

# Pre-registered param variants (2 cells; the 14d cell IS the declared fallback
# constant, not a tuned grid point).  Keyed by the 1h bar cadence this lane
# consumes; each variant dict is a full constructor-param set.
_OI_GROWTH_PRESSURE_SLICE: dict[str, tuple[dict[str, Any], ...]] = {
    "1h": (
        {
            "variant": "oi_growth_7d_continuation",
            "delta_oi_window_days": DELTA_OI_WINDOW_DAYS,
            "quantile_pct": 0.20,
            "min_symbols": 5,
            "min_history_days": DELTA_OI_WINDOW_DAYS + 1,
            "min_oi_coverage": 0.80,
            "min_hold_decisions": 5,
            "rank_hysteresis_buffer": 1,
            "vol_window": 168,
            "allow_short": True,
            "target_gross_exposure": 1.0,
        },
        {
            "variant": "oi_growth_14d_fallback",
            "delta_oi_window_days": DELTA_OI_WINDOW_DAYS_FALLBACK,
            "quantile_pct": 0.20,
            "min_symbols": 5,
            "min_history_days": DELTA_OI_WINDOW_DAYS_FALLBACK + 1,
            "min_oi_coverage": 0.80,
            "min_hold_decisions": 5,
            "rank_hysteresis_buffer": 1,
            "vol_window": 168,
            "allow_short": True,
            "target_gross_exposure": 1.0,
        },
    ),
}

__all__ = [
    "DELTA_OI_WINDOW_DAYS",
    "DELTA_OI_WINDOW_DAYS_FALLBACK",
    "INTENDED_HOLD_SECONDS",
    "OpenInterestGrowthPressureStrategy",
]
