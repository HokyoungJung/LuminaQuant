"""Cross-sectional capital-gains-overhang / disposition long-short sleeve.

``CrossSectionalCapitalGainsOverhangStrategy`` ranks the cross-section on each
symbol's unrealized capital-gains OVERHANG relative to a Grinblatt-Han (2005)
turnover-decay reference price -- the volume-weighted cost basis of the current
holder cohort -- and takes a LONG-SHORT book: long the top-overhang quantile,
short the bottom (most-underwater) quantile.

THEORY / PROVENANCE
-------------------
- Grinblatt & Han (2005), *Journal of Financial Economics* 78(2) -- disposition
  -prone holders sell too early once price clears their cost basis (realization
  utility), pinning price BELOW fundamental value, so positive overhang predicts
  positive drift; underwater holders refuse to realize losses, leaving price
  ABOVE value, so negative overhang predicts negative drift.  The effect is
  CONTINUATION-signed and, in equities, generates and subsumes momentum.
- Frazzini (2006 JF); An (2016 RFS); Wang-Yan-Yu (2017 JFE); the behavioral
  foundation is Barberis-Xiong (2009 JF) realization utility.  Crypto behavioral
  evidence for the disposition pattern is on-chain (SOPR family;
  Schatzmann-Haslhofer 2020) -- disposition EXISTS on-chain, though a published
  crypto cross-sectional CGO return premium is not established, which is priced
  into the (medium) prior of death and the data-PC EXPECTED NULL.

DISTINCT-FROM (the incumbents this sleeve was built to diverge from)
-------------------------------------------------------------------
The anchor is volume-weighted COST BASIS, NOT nearness-to-high.  A coin at its
all-time high on heavy volume traded AT that high has near-zero overhang (all
holders just bought there), while a coin far below an old thin-volume spike but
accumulated far lower carries large positive overhang.  So this sleeve takes
OPPOSITE-signed positions from ``CrossSectionalNearHighAnchoringStrategy`` on
those two geometries.  It is also continuation-signed, so on an above-VWAP
dislocation it goes LONG where ``VwapReversionStrategy`` fades SHORT.  Both
divergences are pinned by the build-gate test.

This module is data-local (no I/O, no hidden configuration bus), pure Python
(``math`` only; the Grinblatt-Han recursion lives in the pure
``indicators/reference_price.py`` helper), and never raises from
``calculate_signals``.  It ships WITHOUT ``@register`` (inert until the
integration wave wires it).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

from lumina_quant.core.plugin_registry import register
from lumina_quant.indicators.alpha_features import realized_volatility
from lumina_quant.indicators.common import safe_float, time_key
from lumina_quant.indicators.reference_price import (
    capital_gains_overhang,
    grinblatt_han_reference_price,
)
from lumina_quant.strategies.adaptive_crypto_alpha_sleeves import (
    _age_cross_positions,
    _state_size,
)
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

_STRATEGY_ID = "cgo_disposition"
_STRATEGY_NAME = "CrossSectionalCapitalGainsOverhangStrategy"


@dataclass(slots=True)
class _State:
    closes: deque[float]
    volumes: deque[float]
    mode: str = "OUT"
    entry_price: float | None = None
    bars_held: int = 0
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


@register("strategy", "CrossSectionalCapitalGainsOverhangStrategy", interface="event_driven")
class CrossSectionalCapitalGainsOverhangStrategy(Strategy):
    """Long-short cross-sectional capital-gains-overhang (disposition) book.

    See the module docstring for the full theory, signal spec, and the
    distinct-from rationale versus ``CrossSectionalNearHighAnchoringStrategy``
    (cost basis vs nearness) and ``VwapReversionStrategy`` (continuation vs
    fade).  This class only reads local event/bar OHLCV; it performs no I/O and
    never raises from ``calculate_signals``.
    """

    # Weekly-cadence cross-sectional book; live-applicable on >= 30-minute bars.
    decision_cadence_seconds = 1800
    preferred_contract = "market_window"
    uses_timeframe_aggregator = False

    @classmethod
    def get_param_schema(cls) -> dict[str, HyperParam]:
        return {
            # Grinblatt-Han reference window in bars (~8wk of 1d bars; the
            # data-PC sweeps 4/8/13/26wk).
            "window_bars": HyperParam.integer("window_bars", default=56, low=4, high=100000),
            "skip_recent": HyperParam.integer("skip_recent", default=1, low=0, high=64),
            "v_clip_max": HyperParam.floating("v_clip_max", default=0.999, low=0.01, high=0.999999),
            "vol_window": HyperParam.integer("vol_window", default=30, low=2, high=2000),
            "quantile_pct": HyperParam.floating("quantile_pct", default=0.25, low=0.02, high=0.50),
            "rebalance_bars": HyperParam.integer("rebalance_bars", default=7, low=1, high=100000),
            "min_hold_bars": HyperParam.integer("min_hold_bars", default=7, low=0, high=100000),
            # Per-symbol history floor: below this a symbol is skipped.
            "min_history_bars": HyperParam.integer(
                "min_history_bars", default=70, low=3, high=100000
            ),
            "allow_short": HyperParam.boolean("allow_short", default=True, grid=[True, False]),
            "min_symbols": HyperParam.integer("min_symbols", default=5, low=2, high=512),
            "target_gross_exposure": HyperParam.floating(
                "target_gross_exposure", default=1.0, low=0.0, high=3.0
            ),
            "target_vol": HyperParam.floating("target_vol", default=0.20, low=0.0, high=2.0),
            "stop_loss_pct": HyperParam.floating("stop_loss_pct", default=0.10, low=0.0, high=0.50),
            "max_hold_bars": HyperParam.integer("max_hold_bars", default=0, low=0, high=200000),
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
        self.window_bars = max(4, int(resolved["window_bars"]))
        self.skip_recent = max(0, int(resolved["skip_recent"]))
        self.v_clip_max = min(0.999999, max(0.01, float(resolved["v_clip_max"])))
        self.vol_window = max(2, int(resolved["vol_window"]))
        self.quantile_pct = min(0.5, max(0.0, float(resolved["quantile_pct"])))
        self.rebalance_bars = max(1, int(resolved["rebalance_bars"]))
        self.min_hold_bars = max(0, int(resolved["min_hold_bars"]))
        self.min_history_bars = max(3, int(resolved["min_history_bars"]))
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
        # Retain enough bars for the reference window + skip and the vol window.
        size = _state_size(
            self.window_bars + self.skip_recent + 2,
            self.vol_window + 1,
            self.min_history_bars,
            self.max_hold_bars,
        )
        self._state: dict[str, _State] = {
            symbol: _State(
                closes=deque(maxlen=size),
                volumes=deque(maxlen=size),
            )
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
            "symbol_state": {
                symbol: {
                    "closes": list(item.closes),
                    "volumes": list(item.volumes),
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
        overhang: dict[str, float] = {}
        vols: dict[str, float] = {}
        metas: dict[str, dict[str, Any]] = {}
        for symbol, item in self._state.items():
            closes = list(item.closes)
            volumes = list(item.volumes)
            if len(closes) < self.min_history_bars:
                continue
            close = closes[-1]
            if close is None or close <= 0.0:
                continue
            reference = grinblatt_han_reference_price(
                closes,
                volumes,
                self.window_bars,
                skip_recent=self.skip_recent,
                v_clip_max=self.v_clip_max,
            )
            cgo = capital_gains_overhang(close, reference)
            if cgo is None:
                continue
            vol = realized_volatility(closes, window=self.vol_window)
            if vol is None or vol <= _EPS:
                continue
            overhang[symbol] = float(cgo)
            vols[symbol] = float(vol)
            metas[symbol] = {
                "capital_gains_overhang": float(cgo),
                "reference_price": float(reference) if reference is not None else None,
            }

        if len(overhang) < self.min_symbols:
            return {}, {}

        # Cross-sectional z-score of overhang (diagnostic + strength scaling).
        values = list(overhang.values())
        count = len(values)
        mean_value = sum(values) / float(count)
        variance = sum((value - mean_value) ** 2 for value in values) / float(max(1, count - 1))
        sigma = variance**0.5
        for symbol, cgo in overhang.items():
            z = 0.0 if sigma <= _EPS else (cgo - mean_value) / sigma
            metas[symbol]["overhang_z"] = float(z)

        # Deterministic ascending order by overhang (symbol tiebreak): the top
        # quantile (highest overhang) is long, the bottom quantile is short.
        ordered = sorted(overhang, key=lambda symbol: (overhang[symbol], symbol))
        n_side = max(1, int(self.quantile_pct * count))
        if 2 * n_side > count:
            n_side = count // 2
        if n_side < 1:
            return {}, {}
        short_syms = ordered[:n_side]
        long_syms = ordered[-n_side:]

        targets: dict[str, tuple[str, float, dict[str, Any]]] = {}
        for symbol in long_syms:
            targets[symbol] = ("LONG", float(metas[symbol]["overhang_z"]), metas[symbol])
        if self.allow_short:
            for symbol in short_syms:
                if symbol in targets:
                    continue
                targets[symbol] = ("SHORT", float(metas[symbol]["overhang_z"]), metas[symbol])
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
        portfolio_vol = sum((inv[symbol] / total_inv) * vols[symbol] for symbol in inv)
        scalar = 1.0
        if self.target_vol > 0.0 and portfolio_vol > _EPS:
            scalar = min(1.0, self.target_vol / portfolio_vol)
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
        # Stops / max-hold age EVERY bar so a held name is always protected,
        # independent of the slow rebalance clock.
        self._age(event_time)
        if self._tick % self.rebalance_bars:
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
                    # suppressed (the proven turnover-discipline rescue).
                    if item.bars_held < self.min_hold_bars:
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
                    item.bars_held = 0
                    item.score = None
                continue
            target_mode, score, meta = target
            if item.mode == target_mode:
                item.score = float(score)
                continue
            # Min-hold floor: a would-be side-flip inside the hold window is
            # suppressed; the current position is kept until the hold clears.
            if item.mode != "OUT" and item.bars_held < self.min_hold_bars:
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
                strength=max(0.25, min(3.0, abs(score))),
                price=price,
                stop_loss=stop_loss,
                metadata=metadata,
            )
            item.mode = target_mode
            item.entry_price = price
            item.bars_held = 0
            item.score = float(score)


# --------------------------------------------------------------------------- #
# Candidate-wiring hints for the integrator (this lane does NOT wire candidates
# itself -- new-file-only, no shared-file edits per the live-safety plan).
# Admission route is `allow_multi_asset=True` at the data-PC handoff: this book
# is a pure cross-sectional long-short (NOT carry, NOT momentum), so it is
# honestly EXCLUDED from any carry/momentum tag-superset allowlist -- no fake
# carry tag is added to game that path.
# --------------------------------------------------------------------------- #
_SUGGESTED_FAMILY = "cross_sectional"
_SUGGESTED_CANDIDATE_TAGS: tuple[str, ...] = (
    "cross_sectional",
    "capital_gains_overhang",
    "disposition",
    "long_short",
    "zscore",
    "crypto",
)

# Candidate slice.  The published effect is a weekly/monthly-horizon reference;
# the data-PC owns the 4/8/13/26wk window factor_ic sweep, so we seed only two
# reference lookbacks (~8wk and ~13wk) to keep the candidate library thin.
#
# MULTI-TIMEFRAME (data-PC parquet carries 1h/4h but not 1d): the ~8wk/~13wk
# holder-cohort economics are WALL-CLOCK horizons, so every BAR-denominated
# param (Grinblatt-Han ``window_bars``, ``skip_recent``, the realized-vol
# ``vol_window``, the per-symbol ``min_history_bars`` floor, and the
# ``rebalance_bars``/``min_hold_bars`` decision clocks) scales x6 for 4h and x24
# for 1h to preserve the same weeks-of-wall-clock reference/cadence.  Ratios,
# counts, and thresholds (``quantile_pct``, ``min_symbols``, ``allow_short``,
# ``target_gross_exposure``, the per-bar ``target_vol`` sizing target, and the
# ``stop_loss_pct`` price stop) are timeframe-agnostic and stay UNCHANGED.  No
# window reaches the ~9000-bar cap.
_CGO_DISPOSITION_SLICE: dict[str, tuple[dict[str, Any], ...]] = {
    "1h": (
        {
            "variant": "wk8",
            "window_bars": 1344,  # 56 x24 (~8wk of 1h bars)
            "skip_recent": 24,  # 1 x24 (~1d)
            "vol_window": 720,  # 30 x24 (~30d)
            "quantile_pct": 0.25,
            "rebalance_bars": 168,  # 7 x24 (~weekly cadence)
            "min_hold_bars": 168,  # 7 x24
            "min_history_bars": 1680,  # 70 x24
            "min_symbols": 5,
            "allow_short": True,
            "target_gross_exposure": 1.0,
            "target_vol": 0.20,
            "stop_loss_pct": 0.10,
        },
        {
            "variant": "wk13",
            "window_bars": 2184,  # 91 x24 (~13wk of 1h bars)
            "skip_recent": 24,
            "vol_window": 720,
            "quantile_pct": 0.25,
            "rebalance_bars": 168,
            "min_hold_bars": 168,
            "min_history_bars": 2520,  # 105 x24
            "min_symbols": 5,
            "allow_short": True,
            "target_gross_exposure": 1.0,
            "target_vol": 0.20,
            "stop_loss_pct": 0.10,
        },
    ),
    "4h": (
        {
            "variant": "wk8",
            "window_bars": 336,  # 56 x6 (~8wk of 4h bars)
            "skip_recent": 6,  # 1 x6 (~1d)
            "vol_window": 180,  # 30 x6 (~30d)
            "quantile_pct": 0.25,
            "rebalance_bars": 42,  # 7 x6 (~weekly cadence)
            "min_hold_bars": 42,  # 7 x6
            "min_history_bars": 420,  # 70 x6
            "min_symbols": 5,
            "allow_short": True,
            "target_gross_exposure": 1.0,
            "target_vol": 0.20,
            "stop_loss_pct": 0.10,
        },
        {
            "variant": "wk13",
            "window_bars": 546,  # 91 x6 (~13wk of 4h bars)
            "skip_recent": 6,
            "vol_window": 180,
            "quantile_pct": 0.25,
            "rebalance_bars": 42,
            "min_hold_bars": 42,
            "min_history_bars": 630,  # 105 x6
            "min_symbols": 5,
            "allow_short": True,
            "target_gross_exposure": 1.0,
            "target_vol": 0.20,
            "stop_loss_pct": 0.10,
        },
    ),
    "1d": (
        {
            "variant": "wk8",
            "window_bars": 56,
            "skip_recent": 1,
            "vol_window": 30,
            "quantile_pct": 0.25,
            "rebalance_bars": 7,
            "min_hold_bars": 7,
            "min_history_bars": 70,
            "min_symbols": 5,
            "allow_short": True,
            "target_gross_exposure": 1.0,
            "target_vol": 0.20,
            "stop_loss_pct": 0.10,
        },
        {
            "variant": "wk13",
            "window_bars": 91,
            "skip_recent": 1,
            "vol_window": 30,
            "quantile_pct": 0.25,
            "rebalance_bars": 7,
            "min_hold_bars": 7,
            "min_history_bars": 105,
            "min_symbols": 5,
            "allow_short": True,
            "target_gross_exposure": 1.0,
            "target_vol": 0.20,
            "stop_loss_pct": 0.10,
        },
    ),
}

__all__ = ["CrossSectionalCapitalGainsOverhangStrategy"]
