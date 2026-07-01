"""Qlib Alpha158-style formulaic factor library (crypto-perp re-expression).

This module is an *additive* net-new factor library that re-expresses the
Qlib ``Alpha158`` handler features (https://github.com/microsoft/qlib) in this
repository's formulaic convention: every factor is a pure, per-symbol OHLCV(+V)
transform that consumes only *trailing* bars and returns the latest scalar
value (mirroring ``formulaic_definitions.AlphaFunctionSpec.callable``).

Scope / de-duplication
----------------------
Alpha158 overlaps with factors already shipped in this repo, so those are
*excluded* to keep this library a strict net-increment:

* The 101 Alpha101 programs already live in
  ``indicators.alpha101`` / ``indicators.formulaic_definitions``.
* The KBAR body/shape ratios ``kmid`` / ``klen`` / ``kup`` / ``klow`` already
  live in ``alpha_zoo.crypto_fx_factors``.

What remains here is the net-new Alpha158 surface:

* KBAR *range-normalised* body/wick ratios and the shift terms
  (``KMID2`` / ``KUP2`` / ``KLOW2`` / ``KSFT`` / ``KSFT2``) that are NOT in the
  crypto_fx panel.
* The current-bar price ratios ``OPEN0`` / ``HIGH0`` / ``LOW0`` / ``VWAP0``.
* The full Alpha158 rolling factor grid over
  :data:`DEFAULT_QLIB158_WINDOWS` (5/10/20/30/60 bars): trend regression,
  quantiles, stochastic, volume pressure, up/down counting, and correlation
  families.

Cross-sectional operators (Qlib ``Rank`` across the universe) are intentionally
NOT used here; ``RANK`` below is a *time-series* trailing percentile rank so the
whole library stays a per-symbol transform, which is the crypto-perp friendly
re-expression requested.

Gating
------
This library is inert by default.  Nothing in the default runtime path imports
it, so the golden backtest / walk-forward numerics are unaffected.  Opting the
factors into a live research/candidate path is gated behind
``strategies.qlib158_formula.enabled`` (default ``False``) in the runtime
config.

Eligibility bar
---------------
The documented IC / ICIR eligibility bar is exposed as module constants
(:data:`MIN_ABS_IC`, :data:`MIN_ICIR`) plus the :class:`Qlib158EligibilityBar`
flag object.  Screening code should call :func:`evaluate_factor_eligibility`
which *marks* every factor with a disposition instead of silently dropping the
ones below the bar.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

_EPS = 1e-12

FieldMap = Mapping[str, Sequence[float]]
ValueFn = Callable[[dict[str, np.ndarray], int], float | None]

# --- Documented IC / ICIR eligibility bar (exposed as flags) -----------------
#: Minimum absolute rank-IC a factor must clear to be considered eligible.
MIN_ABS_IC: float = 0.02
#: Minimum IC information ratio (mean IC / std IC) for eligibility.
MIN_ICIR: float = 0.30

#: Default rolling windows for the Alpha158 rolling grid (bars).
DEFAULT_QLIB158_WINDOWS: tuple[int, ...] = (5, 10, 20, 30, 60)


@dataclass(frozen=True, slots=True)
class Qlib158EligibilityBar:
    """Documented, tunable IC/ICIR gate for the Qlib158 factor surface."""

    min_abs_ic: float = MIN_ABS_IC
    min_icir: float = MIN_ICIR


def evaluate_factor_eligibility(
    *,
    abs_ic: float | None,
    icir: float | None,
    bar: Qlib158EligibilityBar | None = None,
) -> dict[str, object]:
    """Mark a factor against the eligibility bar without silently discarding it.

    Returns a record carrying the raw stats, the active thresholds, a boolean
    ``passes`` flag, and a human-readable ``disposition`` so below-bar factors
    are *shown* rather than dropped.
    """
    active = bar or Qlib158EligibilityBar()
    abs_ic_f = None if abs_ic is None else abs(float(abs_ic))
    icir_f = None if icir is None else float(icir)

    if (
        abs_ic_f is None
        or icir_f is None
        or not math.isfinite(abs_ic_f)
        or not math.isfinite(icir_f)
    ):
        disposition = "insufficient_stats"
        passes = False
    elif abs_ic_f < active.min_abs_ic:
        disposition = "below_ic_bar"
        passes = False
    elif abs(icir_f) < active.min_icir:
        disposition = "below_icir_bar"
        passes = False
    else:
        disposition = "eligible"
        passes = True

    return {
        "abs_ic": abs_ic_f,
        "icir": icir_f,
        "min_abs_ic": float(active.min_abs_ic),
        "min_icir": float(active.min_icir),
        "passes": bool(passes),
        "disposition": disposition,
    }


# --- low-level numeric helpers ----------------------------------------------
def _as_array(values: Sequence[float]) -> np.ndarray:
    return np.asarray(list(values), dtype=float)


def _finite(value: float) -> float | None:
    out = float(value)
    return out if math.isfinite(out) else None


def _safe_div(numer: float, denom: float, *, eps: float = _EPS) -> float | None:
    if abs(denom) <= eps:
        return None
    return _finite(numer / denom)


def _slope(arr: np.ndarray) -> float | None:
    n = arr.size
    if n < 2:
        return None
    x = np.arange(n, dtype=float)
    xm = x.mean()
    denom = float(((x - xm) ** 2).sum())
    if denom <= _EPS:
        return None
    return _finite(float(((x - xm) * (arr - arr.mean())).sum()) / denom)


def _rsqr(arr: np.ndarray) -> float | None:
    n = arr.size
    if n < 2:
        return None
    x = np.arange(n, dtype=float)
    if float(np.std(x)) <= _EPS or float(np.std(arr)) <= _EPS:
        return None
    corr = float(np.corrcoef(x, arr)[0, 1])
    return _finite(corr * corr)


def _resi_last(arr: np.ndarray) -> float | None:
    n = arr.size
    if n < 2:
        return None
    x = np.arange(n, dtype=float)
    xm = x.mean()
    denom = float(((x - xm) ** 2).sum())
    if denom <= _EPS:
        return None
    slope = float(((x - xm) * (arr - arr.mean())).sum()) / denom
    intercept = float(arr.mean()) - slope * xm
    predicted_last = intercept + slope * float(x[-1])
    return _finite(float(arr[-1]) - predicted_last)


def _ts_rank_last(arr: np.ndarray) -> float | None:
    finite = arr[np.isfinite(arr)]
    n = finite.size
    if n == 0:
        return None
    latest = float(arr[-1])
    if not math.isfinite(latest):
        return None
    below = int((finite < latest).sum())
    equal = int((finite == latest).sum())
    return _finite((below + (equal + 1.0) / 2.0) / float(n))


def _diffs(arr: np.ndarray) -> np.ndarray:
    return arr[1:] - arr[:-1]


def _returns(arr: np.ndarray) -> np.ndarray:
    prev = arr[:-1]
    safe_prev = np.where(np.abs(prev) > _EPS, prev, np.nan)
    return arr[1:] / safe_prev - 1.0


# --- field access ------------------------------------------------------------
_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "open": ("open", "o"),
    "high": ("high", "h"),
    "low": ("low", "l"),
    "close": ("close", "c"),
    "volume": ("volume", "vol", "v"),
    "vwap": ("vwap",),
}


def _extract_fields(context: FieldMap, fields: Sequence[str]) -> dict[str, np.ndarray] | None:
    resolved: dict[str, np.ndarray] = {}
    for field in fields:
        raw = None
        for alias in _FIELD_ALIASES.get(field, (field,)):
            if alias in context:
                raw = context[alias]
                break
        if raw is None:
            return None
        arr = _as_array(raw)
        if arr.size == 0:
            return None
        resolved[field] = arr
    return resolved


# --- KBAR (range-normalised) + price-ratio value functions -------------------
def _kmid2(f: dict[str, np.ndarray], _d: int) -> float | None:
    return _safe_div(float(f["close"][-1] - f["open"][-1]), float(f["high"][-1] - f["low"][-1]))


def _kup2(f: dict[str, np.ndarray], _d: int) -> float | None:
    top = float(f["high"][-1] - max(f["open"][-1], f["close"][-1]))
    return _safe_div(top, float(f["high"][-1] - f["low"][-1]))


def _klow2(f: dict[str, np.ndarray], _d: int) -> float | None:
    bottom = float(min(f["open"][-1], f["close"][-1]) - f["low"][-1])
    return _safe_div(bottom, float(f["high"][-1] - f["low"][-1]))


def _ksft(f: dict[str, np.ndarray], _d: int) -> float | None:
    numer = float(2.0 * f["close"][-1] - f["high"][-1] - f["low"][-1])
    return _safe_div(numer, float(f["open"][-1]))


def _ksft2(f: dict[str, np.ndarray], _d: int) -> float | None:
    numer = float(2.0 * f["close"][-1] - f["high"][-1] - f["low"][-1])
    return _safe_div(numer, float(f["high"][-1] - f["low"][-1]))


def _price_ratio(field: str) -> ValueFn:
    def _fn(f: dict[str, np.ndarray], _d: int) -> float | None:
        return _safe_div(float(f[field][-1]), float(f["close"][-1]))

    return _fn


# --- rolling value-function factory ------------------------------------------
def _need(arr: np.ndarray, window: int) -> np.ndarray | None:
    if arr.size < window:
        return None
    return arr[-window:]


def _roc(f, d):
    c = f["close"]
    if c.size < d + 1:
        return None
    return _safe_div(float(c[-1 - d]), float(c[-1]))


def _ma(f, d):
    tail = _need(f["close"], d)
    if tail is None:
        return None
    return _safe_div(float(tail.mean()), float(f["close"][-1]))


def _std(f, d):
    tail = _need(f["close"], d)
    if tail is None or tail.size < 2:
        return None
    return _safe_div(float(np.std(tail, ddof=1)), float(f["close"][-1]))


def _beta(f, d):
    tail = _need(f["close"], d)
    if tail is None:
        return None
    slope = _slope(tail)
    if slope is None:
        return None
    return _safe_div(slope, float(f["close"][-1]))


def _rsqr_fn(f, d):
    tail = _need(f["close"], d)
    if tail is None:
        return None
    return _rsqr(tail)


def _resi(f, d):
    tail = _need(f["close"], d)
    if tail is None:
        return None
    resi = _resi_last(tail)
    if resi is None:
        return None
    return _safe_div(resi, float(f["close"][-1]))


def _max(f, d):
    tail = _need(f["high"], d)
    if tail is None:
        return None
    return _safe_div(float(tail.max()), float(f["close"][-1]))


def _min(f, d):
    tail = _need(f["low"], d)
    if tail is None:
        return None
    return _safe_div(float(tail.min()), float(f["close"][-1]))


def _qtlu(f, d):
    tail = _need(f["close"], d)
    if tail is None:
        return None
    return _safe_div(float(np.quantile(tail, 0.8)), float(f["close"][-1]))


def _qtld(f, d):
    tail = _need(f["close"], d)
    if tail is None:
        return None
    return _safe_div(float(np.quantile(tail, 0.2)), float(f["close"][-1]))


def _rank(f, d):
    tail = _need(f["close"], d)
    if tail is None:
        return None
    return _ts_rank_last(tail)


def _rsv(f, d):
    high = _need(f["high"], d)
    low = _need(f["low"], d)
    if high is None or low is None:
        return None
    lo = float(low.min())
    span = float(high.max()) - lo
    return _safe_div(float(f["close"][-1]) - lo, span)


def _imax(f, d):
    tail = _need(f["high"], d)
    if tail is None:
        return None
    return _finite(float(int(np.argmax(tail))) / float(d))


def _imin(f, d):
    tail = _need(f["low"], d)
    if tail is None:
        return None
    return _finite(float(int(np.argmin(tail))) / float(d))


def _imxd(f, d):
    high = _need(f["high"], d)
    low = _need(f["low"], d)
    if high is None or low is None:
        return None
    return _finite(float(int(np.argmax(high)) - int(np.argmin(low))) / float(d))


def _corr(f, d):
    close = _need(f["close"], d)
    vol = _need(f["volume"], d)
    if close is None or vol is None:
        return None
    log_vol = np.log(np.maximum(vol, 0.0) + 1.0)
    if float(np.std(close)) <= _EPS or float(np.std(log_vol)) <= _EPS:
        return None
    return _finite(float(np.corrcoef(close, log_vol)[0, 1]))


def _cord(f, d):
    close = f["close"]
    vol = f["volume"]
    if close.size < d + 1 or vol.size < d + 1:
        return None
    ret = _returns(close)[-d:]
    safe_prev_vol = np.where(np.abs(vol[:-1]) > _EPS, vol[:-1], np.nan)
    vol_change = np.log(vol[1:] / safe_prev_vol + 1.0)[-d:]
    mask = np.isfinite(ret) & np.isfinite(vol_change)
    if int(mask.sum()) < 2:
        return None
    ret_m = ret[mask]
    vol_m = vol_change[mask]
    if float(np.std(ret_m)) <= _EPS or float(np.std(vol_m)) <= _EPS:
        return None
    return _finite(float(np.corrcoef(ret_m, vol_m)[0, 1]))


def _cntp(f, d):
    close = f["close"]
    if close.size < d + 1:
        return None
    diff = _diffs(close)[-d:]
    return _finite(float((diff > 0.0).mean()))


def _cntn(f, d):
    close = f["close"]
    if close.size < d + 1:
        return None
    diff = _diffs(close)[-d:]
    return _finite(float((diff < 0.0).mean()))


def _cntd(f, d):
    up = _cntp(f, d)
    down = _cntn(f, d)
    if up is None or down is None:
        return None
    return _finite(up - down)


def _sump(f, d):
    close = f["close"]
    if close.size < d + 1:
        return None
    diff = _diffs(close)[-d:]
    pos = float(np.maximum(diff, 0.0).sum())
    denom = float(np.abs(diff).sum())
    return _safe_div(pos, denom)


def _sumn(f, d):
    close = f["close"]
    if close.size < d + 1:
        return None
    diff = _diffs(close)[-d:]
    neg = float(np.maximum(-diff, 0.0).sum())
    denom = float(np.abs(diff).sum())
    return _safe_div(neg, denom)


def _sumd(f, d):
    up = _sump(f, d)
    down = _sumn(f, d)
    if up is None or down is None:
        return None
    return _finite(up - down)


def _vma(f, d):
    tail = _need(f["volume"], d)
    if tail is None:
        return None
    return _safe_div(float(tail.mean()), float(f["volume"][-1]))


def _vstd(f, d):
    tail = _need(f["volume"], d)
    if tail is None or tail.size < 2:
        return None
    return _safe_div(float(np.std(tail, ddof=1)), float(f["volume"][-1]))


def _wvma(f, d):
    close = f["close"]
    vol = f["volume"]
    if close.size < d + 1 or vol.size < d + 1:
        return None
    weighted = np.abs(_returns(close)) * vol[1:]
    tail = weighted[-d:]
    if tail.size < 2:
        return None
    return _safe_div(float(np.std(tail, ddof=1)), float(tail.mean()))


def _vsump(f, d):
    vol = f["volume"]
    if vol.size < d + 1:
        return None
    diff = _diffs(vol)[-d:]
    pos = float(np.maximum(diff, 0.0).sum())
    denom = float(np.abs(diff).sum())
    return _safe_div(pos, denom)


def _vsumn(f, d):
    vol = f["volume"]
    if vol.size < d + 1:
        return None
    diff = _diffs(vol)[-d:]
    neg = float(np.maximum(-diff, 0.0).sum())
    denom = float(np.abs(diff).sum())
    return _safe_div(neg, denom)


def _vsumd(f, d):
    up = _vsump(f, d)
    down = _vsumn(f, d)
    if up is None or down is None:
        return None
    return _finite(up - down)


# --- factor spec assembly ----------------------------------------------------
@dataclass(frozen=True, slots=True)
class Qlib158FactorSpec:
    """Callable Qlib158 factor: pure per-symbol trailing OHLCV transform."""

    name: str
    family: str
    inputs: tuple[str, ...]
    window: int
    description: str
    value_fn: ValueFn

    def evaluate(self, context: FieldMap) -> float | None:
        if not isinstance(context, Mapping) or not context:
            return None
        fields = _extract_fields(context, self.inputs)
        if fields is None:
            return None
        try:
            return self.value_fn(fields, int(self.window))
        except ValueError, ZeroDivisionError, FloatingPointError:
            return None

    @property
    def callable(self) -> Callable[[FieldMap], float | None]:
        return self.evaluate

    def metadata(self) -> dict[str, object]:
        return {
            "name": self.name,
            "family": self.family,
            "inputs": list(self.inputs),
            "window": int(self.window),
            "description": self.description,
        }


# name-stub -> (family, inputs, description-suffix, value_fn)
_ROLLING_FACTORS: tuple[tuple[str, str, tuple[str, ...], str, ValueFn], ...] = (
    ("ROC", "trend", ("close",), "rate of change vs latest close", _roc),
    ("MA", "trend", ("close",), "close moving average / close", _ma),
    ("STD", "volatility", ("close",), "close stddev / close", _std),
    ("BETA", "trend", ("close",), "regression slope / close", _beta),
    ("RSQR", "trend", ("close",), "regression r-squared", _rsqr_fn),
    ("RESI", "trend", ("close",), "regression residual / close", _resi),
    ("MAX", "range", ("high", "close"), "rolling high / close", _max),
    ("MIN", "range", ("low", "close"), "rolling low / close", _min),
    ("QTLU", "range", ("close",), "80% close quantile / close", _qtlu),
    ("QTLD", "range", ("close",), "20% close quantile / close", _qtld),
    ("RANK", "range", ("close",), "trailing percentile rank of close", _rank),
    ("RSV", "range", ("high", "low", "close"), "stochastic RSV", _rsv),
    ("IMAX", "range", ("high",), "argmax(high) / window", _imax),
    ("IMIN", "range", ("low",), "argmin(low) / window", _imin),
    ("IMXD", "range", ("high", "low"), "argmax(high)-argmin(low) / window", _imxd),
    ("CORR", "volume", ("close", "volume"), "corr(close, log volume)", _corr),
    ("CORD", "volume", ("close", "volume"), "corr(return, log volume change)", _cord),
    ("CNTP", "count", ("close",), "fraction of up bars", _cntp),
    ("CNTN", "count", ("close",), "fraction of down bars", _cntn),
    ("CNTD", "count", ("close",), "up minus down bar fraction", _cntd),
    ("SUMP", "count", ("close",), "gain sum / abs-change sum", _sump),
    ("SUMN", "count", ("close",), "loss sum / abs-change sum", _sumn),
    ("SUMD", "count", ("close",), "gain minus loss ratio", _sumd),
    ("VMA", "volume", ("volume",), "volume MA / volume", _vma),
    ("VSTD", "volume", ("volume",), "volume stddev / volume", _vstd),
    ("WVMA", "volume", ("close", "volume"), "return-weighted volume volatility", _wvma),
    ("VSUMP", "volume", ("volume",), "volume gain / abs-change sum", _vsump),
    ("VSUMN", "volume", ("volume",), "volume loss / abs-change sum", _vsumn),
    ("VSUMD", "volume", ("volume",), "volume gain minus loss ratio", _vsumd),
)

# net-new KBAR (range-normalised) + price ratios (window-agnostic; window=1)
_STATIC_FACTORS: tuple[tuple[str, str, tuple[str, ...], str, ValueFn], ...] = (
    ("KMID2", "kbar", ("open", "high", "low", "close"), "body / bar range", _kmid2),
    ("KUP2", "kbar", ("open", "high", "low", "close"), "upper wick / bar range", _kup2),
    ("KLOW2", "kbar", ("open", "high", "low", "close"), "lower wick / bar range", _klow2),
    ("KSFT", "kbar", ("open", "high", "low", "close"), "close shift / open", _ksft),
    ("KSFT2", "kbar", ("open", "high", "low", "close"), "close shift / bar range", _ksft2),
    ("OPEN0", "price", ("open", "close"), "open / close", _price_ratio("open")),
    ("HIGH0", "price", ("high", "close"), "high / close", _price_ratio("high")),
    ("LOW0", "price", ("low", "close"), "low / close", _price_ratio("low")),
    ("VWAP0", "price", ("vwap", "close"), "vwap / close", _price_ratio("vwap")),
)


def _build_specs(
    windows: Sequence[int] = DEFAULT_QLIB158_WINDOWS,
) -> dict[str, Qlib158FactorSpec]:
    specs: dict[str, Qlib158FactorSpec] = {}
    for name, family, inputs, description, fn in _STATIC_FACTORS:
        specs[name] = Qlib158FactorSpec(
            name=name,
            family=family,
            inputs=inputs,
            window=1,
            description=description,
            value_fn=fn,
        )
    for window in windows:
        w = int(window)
        for stub, family, inputs, description, fn in _ROLLING_FACTORS:
            name = f"{stub}{w}"
            specs[name] = Qlib158FactorSpec(
                name=name,
                family=family,
                inputs=inputs,
                window=w,
                description=f"{description} over {w} bars",
                value_fn=fn,
            )
    return specs


QLIB158_FACTOR_SPECS: dict[str, Qlib158FactorSpec] = _build_specs()


def list_qlib158_factor_specs() -> tuple[Qlib158FactorSpec, ...]:
    """Return every factor spec in deterministic name order."""
    return tuple(QLIB158_FACTOR_SPECS[name] for name in sorted(QLIB158_FACTOR_SPECS))


def qlib158_factor_names() -> tuple[str, ...]:
    return tuple(sorted(QLIB158_FACTOR_SPECS))


def get_qlib158_factor_spec(name: str) -> Qlib158FactorSpec:
    spec = QLIB158_FACTOR_SPECS.get(str(name))
    if spec is None:
        raise KeyError(f"Unknown Qlib158 factor: {name!r}")
    return spec


def evaluate_qlib158_factor(name: str, context: FieldMap) -> float | None:
    """Evaluate a single factor by name against a per-symbol OHLCV context."""
    return get_qlib158_factor_spec(name).evaluate(context)


def compute_qlib158_row(context: FieldMap) -> dict[str, float | None]:
    """Evaluate the full factor surface for one symbol context (latest bar)."""
    return {spec.name: spec.evaluate(context) for spec in list_qlib158_factor_specs()}


__all__ = [
    "DEFAULT_QLIB158_WINDOWS",
    "MIN_ABS_IC",
    "MIN_ICIR",
    "QLIB158_FACTOR_SPECS",
    "Qlib158EligibilityBar",
    "Qlib158FactorSpec",
    "compute_qlib158_row",
    "evaluate_factor_eligibility",
    "evaluate_qlib158_factor",
    "get_qlib158_factor_spec",
    "list_qlib158_factor_specs",
    "qlib158_factor_names",
]
