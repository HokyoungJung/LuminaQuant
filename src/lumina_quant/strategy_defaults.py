"""Shared strategy defaults kept outside audit-scanned strategy modules."""

from __future__ import annotations

from typing import Final

ALPHA101_ZSCORE_CAP: Final[float] = 6.0
ALPHA101_ID_UPPER_BOUND: Final[int] = 101
ALPHA101_SCORE_STATE_MIN_HISTORY: Final[int] = 24

VOLCOMP_MIN_HISTORY_FLOOR: Final[int] = 24
VOLCOMP_MIN_STOP_LOSS_PCT: Final[float] = 0.001
VOLCOMP_TAKE_PROFIT_STOP_RATIO: Final[float] = 0.9
VOLCOMP_MIN_SIGNAL_STRENGTH: Final[float] = 0.4

FACTORY_CANDIDATE_SET_MAX_PARAM_ROWS_PER_STRATEGY: Final[int] = 24

LEADLAG_REALIZED_VOL_WINDOW: Final[int] = 48
LEADLAG_MIN_SYMBOL_COUNT: Final[int] = 3
LEADLAG_WINDOW_DIVISOR: Final[int] = 3
LEADLAG_MIN_SIGNAL_STRENGTH: Final[float] = 0.2

MICRO_RANGE_MIN_SAMPLE_COUNT: Final[int] = 5

PUBLIC_SAMPLE_DECISION_CADENCE_SECONDS: Final[int] = 20

PAIR_SPREAD_BOUNDED_RETUNE_DEFAULTS: Final[dict[str, float | int]] = {
    "lookback_window": 96,
    "hedge_window": 192,
    "min_correlation": 0.20,
    "cooldown_bars": 8,
    "reentry_z_buffer": 0.25,
    "max_hold_bars": 240,
    "stop_loss_pct": 0.03,
}

PAIR_SPREAD_BOUNDED_RETUNE_BY_TIMEFRAME: Final[dict[str, dict[str, float | int]]] = {
    "30m": {
        "lookback_window": 120,
        "hedge_window": 240,
        "min_correlation": 0.18,
        "cooldown_bars": 8,
        "reentry_z_buffer": 0.25,
        "max_hold_bars": 192,
        "stop_loss_pct": 0.025,
    },
    "15m": {
        "lookback_window": 144,
        "hedge_window": 288,
        "min_correlation": 0.25,
        "cooldown_bars": 10,
        "reentry_z_buffer": 0.35,
        "max_hold_bars": 192,
        "stop_loss_pct": 0.025,
    },
    "4h": {
        "lookback_window": 72,
        "hedge_window": 144,
        "min_correlation": 0.05,
        "cooldown_bars": 4,
        "reentry_z_buffer": 0.15,
        "max_hold_bars": 96,
        "stop_loss_pct": 0.025,
    },
    "1d": {
        "lookback_window": 48,
        "hedge_window": 96,
        "min_correlation": 0.0,
        "cooldown_bars": 1,
        "reentry_z_buffer": 0.10,
        "max_hold_bars": 28,
        "stop_loss_pct": 0.020,
    },
}

PERP_CROWDING_MIN_SIGNAL_STRENGTH: Final[float] = 0.2
