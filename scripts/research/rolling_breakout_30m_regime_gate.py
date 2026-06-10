"""Build a deterministic ex-ante regime gate for the 30m RollingBreakout sleeve."""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lumina_quant.strategy_factory import research_runner as rr

REPORT_ROOT = Path(__file__).resolve().parents[2] / "var" / "reports" / "exact_window_backtests"
FOLLOWUP_ROOT = REPORT_ROOT / "followup_status"

TARGET_CANDIDATE = "rolling_breakout_30m_guarded_ls_64_0.002"
SCHEMA_VERSION = "4.0"
TARGET_TIMEFRAME = "30m"
REGIME_SIGNAL_LAG_DAYS = 1
DEFAULT_SELECTION_BASIS = "train_val_only"
FULL_SPLIT_SELECTION_BASIS = "full_split"
TRAIN_VAL_SELECTION_BASIS = "train_val_only"
ROLLING_SURVIVAL_THRESHOLDS = {
    "oos_sharpe_min": 1.0,
    "oos_return_min": 0.0,
    "max_pbo": 0.5,
    "val_sharpe_min": 1.0,
    "oos_trade_count_min": 20.0,
    "min_activation_ratio": 0.10,
    "max_activation_ratio": 0.80,
}
TRAIN_VAL_ONLY_SURVIVAL_THRESHOLDS = {
    "val_sharpe_min": 1.0,
    "val_return_min": 0.0,
    "train_return_min": -0.12,
    "min_activation_ratio": 0.10,
    "max_activation_ratio": 0.85,
}
TARGET_RULES: tuple[dict[str, Any], ...] = (
    {
        "rule_id": "basket_vol_ratio_moderate",
        "label": "Moderate basket volatility regime",
        "conditions": ("basket_vol_ratio_moderate",),
    },
    {
        "rule_id": "btc_above_ma192",
        "label": "BTC above 4-day trend filter",
        "conditions": ("btc_above_ma192",),
    },
    {
        "rule_id": "btc_above_ma192_and_breadth_ma96_ge_60",
        "label": "BTC above 4-day trend + 60% basket breadth above 2-day trend",
        "conditions": ("btc_above_ma192", "breadth_ma96_ge_60"),
    },
    {
        "rule_id": "btc_above_ma192_and_breadth_ma192_ge_60",
        "label": "BTC above 4-day trend + 60% basket breadth above 4-day trend",
        "conditions": ("btc_above_ma192", "breadth_ma192_ge_60"),
    },
    {
        "rule_id": "btc_above_ma192_and_breadth_ma96_ge_60_and_vol_expansion",
        "label": "BTC above 4-day trend + breadth + moderate volatility expansion",
        "conditions": (
            "btc_above_ma192",
            "breadth_ma96_ge_60",
            "basket_vol_ratio_moderate",
        ),
    },
    {
        "rule_id": "btc_above_ma336_and_breadth_ma192_ge_60",
        "label": "BTC above 7-day trend + 60% basket breadth above 4-day trend",
        "conditions": ("btc_above_ma336", "breadth_ma192_ge_60"),
    },
    {
        "rule_id": "btc_above_ma192_and_breadth_ma96_ge_60_and_ret96_pos",
        "label": "BTC above 4-day trend + breadth + positive 2-day basket momentum",
        "conditions": ("btc_above_ma192", "breadth_ma96_ge_60", "basket_ret96_pos"),
    },
)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except TypeError, ValueError:
        return float(default)
    if not np.isfinite(out):
        return float(default)
    return out


def _safe_bool(value: Any) -> bool:
    return bool(value)


def _coerce_timestamp(value: Any) -> pd.Timestamp | None:
    if value is None or value == "":
        return None
    if isinstance(value, np.datetime64):
        return pd.Timestamp(value, tz="UTC")
    if isinstance(value, (int, float)):
        return pd.to_datetime(value, unit="ms", utc=True, errors="coerce")
    return pd.to_datetime(value, utc=True, errors="coerce")


def _candidate_timeframe_row(
    decision: Mapping[str, Any], *, candidate: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    target = candidate.strip()
    for timeframe_row in list(decision.get("timeframe_rows") or []):
        best_row = dict(timeframe_row.get("best_row") or {})
        if (
            str(timeframe_row.get("timeframe") or "") == TARGET_TIMEFRAME
            and str(best_row.get("name") or "") == target
            and str(best_row.get("strategy_class") or "") == "RollingBreakoutStrategy"
        ):
            return dict(timeframe_row), best_row
    raise ValueError(f"candidate {target} not found in decision timeframe rows")


def _split_config_from_windows(windows: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "train_start": windows.get("train_start"),
        "train_end": windows.get("train_end_exclusive"),
        "val_start": windows.get("val_start"),
        "val_end": windows.get("val_end_exclusive"),
        "oos_start": windows.get("val_end_exclusive"),
        "oos_end": windows.get("actual_oos_end_exclusive")
        or windows.get("requested_oos_end_exclusive"),
        "strategy_timeframe": TARGET_TIMEFRAME,
        "mode": "exact_dates",
    }


def _normalize_evaluation_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    timestamps = np.asarray(
        [
            np.datetime64(pd.Timestamp(ts).tz_convert("UTC").tz_localize(None), "ms")
            if isinstance(ts, pd.Timestamp)
            else np.datetime64(
                pd.Timestamp(ts).tz_localize(None) if pd.Timestamp(ts).tzinfo else pd.Timestamp(ts),
                "ms",
            )
            for ts in payload.get("timestamps") or []
        ],
        dtype="datetime64[ms]",
    )
    normalized = {
        "timestamps": timestamps,
        "returns_raw": np.asarray(payload.get("returns_raw") or [], dtype=float),
        "turnover": np.asarray(payload.get("turnover") or [], dtype=float),
        "exposure": np.asarray(payload.get("exposure") or [], dtype=float),
        "benchmark_returns": np.asarray(payload.get("benchmark_returns") or [], dtype=float),
        "cost_rate": _safe_float(payload.get("cost_rate"), 0.0005),
        "split_masks": {
            split: np.asarray(list(mask), dtype=bool)
            for split, mask in dict(payload.get("split_masks") or {}).items()
        },
        "bundle_cache": None,
    }
    expected = normalized["timestamps"].size
    for key in ("returns_raw", "turnover", "exposure", "benchmark_returns"):
        if normalized[key].size != expected:
            raise ValueError(f"evaluation payload length mismatch for {key}")
    for split in ("train", "val", "oos"):
        mask = normalized["split_masks"].get(split)
        if mask is None or mask.size != expected:
            raise ValueError(f"evaluation payload missing split mask for {split}")
    return normalized


def _baseline_evaluation(
    candidate_row: Mapping[str, Any],
    *,
    windows: Mapping[str, Any],
    evaluation_payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if evaluation_payload is not None:
        return _normalize_evaluation_payload(evaluation_payload)

    symbols = [str(symbol) for symbol in list(candidate_row.get("symbols") or []) if str(symbol)]
    split_config = rr._resolve_split_config(
        _split_config_from_windows(windows),
        strategy_timeframe=TARGET_TIMEFRAME,
    )
    start = split_config.get("train_start")
    end = split_config.get("oos_end")
    cache, _ = rr._load_bundle_cache(
        symbols=symbols,
        timeframes=[TARGET_TIMEFRAME],
        start_date=start,
        end_date=end,
    )
    bundles = [cache[(symbol, TARGET_TIMEFRAME)] for symbol in symbols]
    aligned = rr._align_bundles(bundles, feature_cache=None)
    if aligned is None:
        raise RuntimeError("unable to align 30m bundles for rolling-breakout regime gate")

    returns_raw, turnover, exposure, _meta = rr._strategy_signal(
        dict(candidate_row),
        aligned=aligned,
        symbols=symbols,
    )
    timestamps = np.asarray(aligned.get("datetime"), dtype="datetime64[ms]")
    split_masks = rr._split_masks_from_datetimes(timestamps, split=split_config)

    benchmark_cache = rr._benchmark_cache(cache, [TARGET_TIMEFRAME])
    benchmark_entry = dict(benchmark_cache.get(TARGET_TIMEFRAME) or {})
    benchmark_datetimes = benchmark_entry.get("datetime")
    benchmark_values = benchmark_entry.get("returns")
    benchmark_returns = rr._align_series_to_timestamps(
        timestamps,
        source_timestamps=np.asarray(
            benchmark_datetimes if benchmark_datetimes is not None else [],
            dtype="datetime64[ms]",
        ),
        values=np.asarray(benchmark_values if benchmark_values is not None else [], dtype=float),
    )
    return {
        "timestamps": timestamps,
        "returns_raw": np.asarray(returns_raw, dtype=float),
        "turnover": np.asarray(turnover, dtype=float),
        "exposure": np.asarray(exposure, dtype=float),
        "benchmark_returns": np.asarray(benchmark_returns, dtype=float),
        "cost_rate": _safe_float(rr._candidate_cost_rate(dict(candidate_row)), 0.0005),
        "split_masks": split_masks,
        "bundle_cache": cache,
    }


def _daily_feature_frame(
    *,
    symbols: list[str],
    windows: Mapping[str, Any],
    bundle_cache: Mapping[tuple[str, str], Any] | None = None,
    feature_frame: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if feature_frame is not None:
        out = feature_frame.copy()
        out["date"] = pd.to_datetime(out["date"], utc=True).dt.floor("D")
        return out.sort_values("date").drop_duplicates("date").reset_index(drop=True)

    cache = bundle_cache
    if cache is None:
        split_config = rr._resolve_split_config(
            _split_config_from_windows(windows),
            strategy_timeframe=TARGET_TIMEFRAME,
        )
        cache, _ = rr._load_bundle_cache(
            symbols=symbols,
            timeframes=[TARGET_TIMEFRAME],
            start_date=split_config.get("train_start"),
            end_date=split_config.get("oos_end"),
        )

    daily_frames: list[pd.DataFrame] = []
    for symbol in symbols:
        bundle = cache[(symbol, TARGET_TIMEFRAME)]
        frame = pd.DataFrame(
            {
                "datetime": pd.to_datetime(bundle.datetime, utc=True),
                "close": np.asarray(bundle.close, dtype=float),
            }
        )
        frame["symbol"] = symbol
        frame["ma96"] = frame["close"].rolling(96).mean()
        frame["ma192"] = frame["close"].rolling(192).mean()
        frame["ma336"] = frame["close"].rolling(336).mean()
        frame["ret96"] = frame["close"].pct_change(96)
        log_ret = np.log(np.clip(frame["close"], 1e-12, np.inf)).diff().fillna(0.0)
        frame["vol48"] = log_ret.rolling(48).std()
        frame["vol192"] = log_ret.rolling(192).std()
        frame["vol_ratio"] = frame["vol48"] / frame["vol192"]
        frame["date"] = frame["datetime"].dt.floor("D")
        daily_frames.append(
            frame.groupby("date", as_index=False).last()[
                ["date", "symbol", "close", "ma96", "ma192", "ma336", "ret96", "vol_ratio"]
            ]
        )

    daily = pd.concat(daily_frames, ignore_index=True)
    close_pivot = daily.pivot(index="date", columns="symbol", values="close")
    ma96_pivot = daily.pivot(index="date", columns="symbol", values="ma96")
    ma192_pivot = daily.pivot(index="date", columns="symbol", values="ma192")
    ma336_pivot = daily.pivot(index="date", columns="symbol", values="ma336")
    ret96_pivot = daily.pivot(index="date", columns="symbol", values="ret96")
    vol_ratio_pivot = daily.pivot(index="date", columns="symbol", values="vol_ratio")

    btc_symbol = "BTC/USDT" if "BTC/USDT" in close_pivot.columns else symbols[0]
    features = pd.DataFrame(index=close_pivot.index.sort_values())
    features["date"] = features.index
    features["btc_above_ma192"] = close_pivot[btc_symbol] > ma192_pivot[btc_symbol]
    features["btc_above_ma336"] = close_pivot[btc_symbol] > ma336_pivot[btc_symbol]
    features["breadth_ma96_ge_60"] = close_pivot.gt(ma96_pivot).mean(axis=1) >= 0.60
    features["breadth_ma192_ge_60"] = close_pivot.gt(ma192_pivot).mean(axis=1) >= 0.60
    basket_ret96 = ret96_pivot.mean(axis=1)
    basket_vol_ratio = vol_ratio_pivot.mean(axis=1)
    features["basket_ret96_pos"] = basket_ret96 > 0.0
    features["basket_vol_ratio_moderate"] = (basket_vol_ratio >= 0.90) & (basket_vol_ratio <= 1.60)
    features["basket_ret96"] = basket_ret96.fillna(0.0).astype(float)
    features["basket_vol_ratio"] = basket_vol_ratio.fillna(0.0).astype(float)
    features["btc_close"] = close_pivot[btc_symbol].ffill().astype(float)
    features["btc_ma192"] = ma192_pivot[btc_symbol].ffill().astype(float)
    features["btc_ma336"] = ma336_pivot[btc_symbol].ffill().astype(float)
    features["breadth_ma96"] = close_pivot.gt(ma96_pivot).mean(axis=1).fillna(0.0).astype(float)
    features["breadth_ma192"] = close_pivot.gt(ma192_pivot).mean(axis=1).fillna(0.0).astype(float)
    return features.reset_index(drop=True)


def _bar_frame(evaluation: Mapping[str, Any]) -> pd.DataFrame:
    timestamps = pd.to_datetime(np.asarray(evaluation["timestamps"]), utc=True)
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "date": timestamps.floor("D"),
            "returns_raw": np.asarray(evaluation["returns_raw"], dtype=float),
            "base_turnover": np.asarray(evaluation["turnover"], dtype=float),
            "base_exposure": np.asarray(evaluation["exposure"], dtype=float),
            "benchmark_returns": np.asarray(evaluation["benchmark_returns"], dtype=float),
        }
    )


def _daily_rule_frame(features: pd.DataFrame, *, conditions: tuple[str, ...]) -> pd.DataFrame:
    out = features.copy()
    if conditions:
        signal_active = np.logical_and.reduce(
            [out[condition].fillna(False).astype(bool).to_numpy() for condition in conditions]
        )
    else:
        signal_active = np.ones(len(out), dtype=bool)
    out["signal_active"] = signal_active
    out["gate_active"] = out["signal_active"].shift(REGIME_SIGNAL_LAG_DAYS, fill_value=False)
    return out


def _gated_bar_frame(evaluation: Mapping[str, Any], daily_rule_frame: pd.DataFrame) -> pd.DataFrame:
    bars = _bar_frame(evaluation)
    merged = bars.merge(
        daily_rule_frame[["date", "signal_active", "gate_active"]],
        on="date",
        how="left",
    )
    merged["signal_active"] = merged["signal_active"].fillna(False).astype(bool)
    merged["gate_active"] = merged["gate_active"].fillna(False).astype(bool)
    merged["gated_exposure"] = np.where(
        merged["gate_active"],
        merged["base_exposure"],
        0.0,
    )
    prev_exposure = np.r_[0.0, merged["gated_exposure"].to_numpy()[:-1]]
    gate_turnover = np.where(merged["gate_active"], merged["base_turnover"], 0.0)
    switch_turnover = np.abs(merged["gated_exposure"].to_numpy() - prev_exposure)
    merged["gated_turnover"] = np.maximum(gate_turnover, switch_turnover)
    gross_returns = np.where(merged["gate_active"], merged["returns_raw"], 0.0)
    merged["gated_return"] = gross_returns - (
        merged["gated_turnover"].to_numpy() * _safe_float(evaluation["cost_rate"], 0.0)
    )
    return merged


def _split_metrics(
    gated_frame: pd.DataFrame,
    *,
    split_masks: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    periods_per_year = int(rr._PERIODS_PER_YEAR.get(TARGET_TIMEFRAME, 365))
    num_trials = len(TARGET_RULES)
    for split in ("train", "val", "oos"):
        mask = np.asarray(split_masks[split], dtype=bool)
        split_frame = gated_frame.loc[mask].copy()
        metrics = rr._compute_metrics(
            split_frame["gated_return"].to_numpy(dtype=float),
            turnover=split_frame["gated_turnover"].to_numpy(dtype=float),
            exposure=split_frame["gated_exposure"].to_numpy(dtype=float),
            benchmark_returns=split_frame["benchmark_returns"].to_numpy(dtype=float),
            periods_per_year=periods_per_year,
            num_trials=num_trials,
        )
        total_days = int(split_frame["date"].nunique())
        gate_days = int(split_frame.loc[split_frame["gate_active"], "date"].nunique())
        metrics.update(
            {
                "gate_days": gate_days,
                "total_days": total_days,
                "activation_ratio": (float(gate_days) / float(total_days))
                if total_days > 0
                else 0.0,
            }
        )
        payload[split] = metrics
    return payload


def _split_streams(
    evaluation: Mapping[str, Any],
    gated_frame: pd.DataFrame,
    *,
    split_masks: Mapping[str, np.ndarray],
) -> dict[str, list[dict[str, Any]]]:
    timestamps = np.asarray(evaluation["timestamps"], dtype="datetime64[ms]")
    streams: dict[str, list[dict[str, Any]]] = {}
    for split in ("train", "val", "oos"):
        mask = np.asarray(split_masks[split], dtype=bool)
        streams[split] = rr._series_to_stream(
            gated_frame.loc[mask, "gated_return"].to_numpy(dtype=float),
            timestamps=timestamps[mask],
        )
    return streams


def _rule_score(
    metrics: Mapping[str, Any],
    *,
    selection_basis: str = DEFAULT_SELECTION_BASIS,
) -> float:
    train = dict(metrics.get("train") or {})
    val = dict(metrics.get("val") or {})
    oos = dict(metrics.get("oos") or {})
    if str(selection_basis).strip().lower() == TRAIN_VAL_SELECTION_BASIS:
        activation_ratio = _safe_float(val.get("activation_ratio"), 0.0)
        return float(
            (2.5 * _safe_float(val.get("sharpe"), 0.0))
            + (12.0 * _safe_float(val.get("return"), 0.0))
            + (2.0 * _safe_float(train.get("return"), 0.0))
            + (0.25 * _safe_float(train.get("sharpe"), 0.0))
            - (
                2.0
                * max(
                    0.0,
                    TRAIN_VAL_ONLY_SURVIVAL_THRESHOLDS["min_activation_ratio"] - activation_ratio,
                )
            )
            - (
                2.0
                * max(
                    0.0,
                    activation_ratio - TRAIN_VAL_ONLY_SURVIVAL_THRESHOLDS["max_activation_ratio"],
                )
            )
        )
    return float(
        (2.5 * _safe_float(oos.get("sharpe"), 0.0))
        + (1.25 * _safe_float(val.get("sharpe"), 0.0))
        + (20.0 * _safe_float(oos.get("return"), 0.0))
        + (10.0 * _safe_float(val.get("return"), 0.0))
        + (2.0 * _safe_float(train.get("return"), 0.0))
        + (0.25 * _safe_float(train.get("sharpe"), 0.0))
        - (2.0 * max(0.0, 0.10 - _safe_float(oos.get("activation_ratio"), 0.0)))
        - (2.0 * max(0.0, 0.10 - _safe_float(val.get("activation_ratio"), 0.0)))
    )


def _survivor_blockers(
    metrics: Mapping[str, Any],
    *,
    selection_basis: str = DEFAULT_SELECTION_BASIS,
) -> list[str]:
    mode = str(selection_basis).strip().lower()
    train = dict(metrics.get("train") or {})
    val = dict(metrics.get("val") or {})
    oos = dict(metrics.get("oos") or {})
    blockers: list[str] = []
    if mode == TRAIN_VAL_SELECTION_BASIS:
        if (
            _safe_float(val.get("sharpe"), 0.0)
            < TRAIN_VAL_ONLY_SURVIVAL_THRESHOLDS["val_sharpe_min"]
        ):
            blockers.append("val_sharpe")
        if (
            _safe_float(val.get("return"), 0.0)
            <= TRAIN_VAL_ONLY_SURVIVAL_THRESHOLDS["val_return_min"]
        ):
            blockers.append("val_return")
        if (
            _safe_float(train.get("return"), 0.0)
            < TRAIN_VAL_ONLY_SURVIVAL_THRESHOLDS["train_return_min"]
        ):
            blockers.append("train_return")
        activation_ratio = _safe_float(val.get("activation_ratio"), 0.0)
        if activation_ratio < TRAIN_VAL_ONLY_SURVIVAL_THRESHOLDS["min_activation_ratio"]:
            blockers.append("activation_ratio_low")
        if activation_ratio > TRAIN_VAL_ONLY_SURVIVAL_THRESHOLDS["max_activation_ratio"]:
            blockers.append("activation_ratio_high")
        return blockers
    if _safe_float(oos.get("sharpe"), 0.0) < ROLLING_SURVIVAL_THRESHOLDS["oos_sharpe_min"]:
        blockers.append("oos_sharpe")
    if _safe_float(oos.get("return"), 0.0) <= ROLLING_SURVIVAL_THRESHOLDS["oos_return_min"]:
        blockers.append("oos_return")
    if _safe_float(oos.get("pbo"), 1.0) > ROLLING_SURVIVAL_THRESHOLDS["max_pbo"]:
        blockers.append("pbo")
    if _safe_float(val.get("sharpe"), 0.0) < ROLLING_SURVIVAL_THRESHOLDS["val_sharpe_min"]:
        blockers.append("val_sharpe")
    if (
        _safe_float(oos.get("trade_count"), 0.0)
        < ROLLING_SURVIVAL_THRESHOLDS["oos_trade_count_min"]
    ):
        blockers.append("oos_trade_count")
    activation_ratio = _safe_float(oos.get("activation_ratio"), 0.0)
    if activation_ratio < ROLLING_SURVIVAL_THRESHOLDS["min_activation_ratio"]:
        blockers.append("activation_ratio_low")
    if activation_ratio > ROLLING_SURVIVAL_THRESHOLDS["max_activation_ratio"]:
        blockers.append("activation_ratio_high")
    return blockers


def _evaluate_rule(
    evaluation: Mapping[str, Any],
    features: pd.DataFrame,
    *,
    rule_id: str,
    label: str,
    conditions: tuple[str, ...],
    selection_basis: str = DEFAULT_SELECTION_BASIS,
) -> dict[str, Any]:
    daily_rule_frame = _daily_rule_frame(features, conditions=conditions)
    gated_frame = _gated_bar_frame(evaluation, daily_rule_frame)
    metrics = _split_metrics(
        gated_frame,
        split_masks=dict(evaluation["split_masks"]),
    )
    hurdle_fields, passed, hard_reject = rr._hurdle_fields(
        metrics["train"],
        metrics["val"],
        metrics["oos"],
    )
    train_val_blockers = _survivor_blockers(
        metrics,
        selection_basis=TRAIN_VAL_SELECTION_BASIS,
    )
    selected_blockers = _survivor_blockers(metrics, selection_basis=selection_basis)
    return {
        "rule_id": rule_id,
        "label": label,
        "conditions": list(conditions),
        "signal_lag_days": REGIME_SIGNAL_LAG_DAYS,
        "selection_basis": selection_basis,
        "selection_uses_oos": str(selection_basis).strip().lower() != TRAIN_VAL_SELECTION_BASIS,
        "score": _rule_score(metrics, selection_basis=selection_basis),
        "metrics": metrics,
        "hurdle_fields": hurdle_fields,
        "pass": _safe_bool(passed),
        "hard_reject_reasons": hard_reject,
        "survivor_blockers": selected_blockers,
        "survives": not selected_blockers,
        "survives_train_val": not train_val_blockers,
        "train_val_survivor_blockers": train_val_blockers,
        "gate_days_total": int(daily_rule_frame["gate_active"].sum()),
        "gated_streams": _split_streams(
            evaluation,
            gated_frame,
            split_masks=dict(evaluation["split_masks"]),
        ),
    }


def build_rolling_breakout_30m_gate(
    decision: Mapping[str, Any],
    candidate: str = TARGET_CANDIDATE,
    *,
    selection_basis: str = DEFAULT_SELECTION_BASIS,
    feature_frame: pd.DataFrame | None = None,
    evaluation_payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    normalized_selection_basis = str(selection_basis).strip().lower() or DEFAULT_SELECTION_BASIS
    if normalized_selection_basis not in {FULL_SPLIT_SELECTION_BASIS, TRAIN_VAL_SELECTION_BASIS}:
        raise ValueError(f"unsupported selection_basis={selection_basis!r}")
    timeframe_row, candidate_row = _candidate_timeframe_row(decision, candidate=candidate)
    windows = dict(timeframe_row.get("windows") or {})
    evaluation = _baseline_evaluation(
        candidate_row,
        windows=windows,
        evaluation_payload=evaluation_payload,
    )
    features = _daily_feature_frame(
        symbols=list(candidate_row.get("symbols") or []),
        windows=windows,
        bundle_cache=evaluation.get("bundle_cache"),
        feature_frame=feature_frame,
    )

    evaluated_rules = [
        _evaluate_rule(
            evaluation,
            features,
            rule_id=str(spec["rule_id"]),
            label=str(spec["label"]),
            conditions=tuple(spec["conditions"]),
            selection_basis=normalized_selection_basis,
        )
        for spec in TARGET_RULES
    ]
    surviving_rules = [rule for rule in evaluated_rules if _safe_bool(rule.get("survives"))]
    candidate_pool = surviving_rules or evaluated_rules
    chosen = max(
        candidate_pool,
        key=lambda item: (
            _safe_float(item.get("score"), 0.0),
            _safe_float(
                (
                    ((item.get("metrics") or {}).get("val") or {}).get("sharpe")
                    if normalized_selection_basis == TRAIN_VAL_SELECTION_BASIS
                    else ((item.get("metrics") or {}).get("oos") or {}).get("sharpe")
                ),
                0.0,
            ),
            _safe_float(
                (
                    ((item.get("metrics") or {}).get("val") or {}).get("return")
                    if normalized_selection_basis == TRAIN_VAL_SELECTION_BASIS
                    else ((item.get("metrics") or {}).get("oos") or {}).get("return")
                ),
                0.0,
            ),
        ),
    )
    survives_train_val = _safe_bool(chosen.get("survives_train_val"))
    selected_survives = _safe_bool(chosen.get("survives"))
    recommended_survives = (
        survives_train_val
        if normalized_selection_basis == TRAIN_VAL_SELECTION_BASIS
        else selected_survives
    )

    gated_candidate_row = {
        **dict(candidate_row),
        "name": str(candidate_row.get("name") or candidate),
        "train": dict((chosen.get("metrics") or {}).get("train") or {}),
        "val": dict((chosen.get("metrics") or {}).get("val") or {}),
        "oos": dict((chosen.get("metrics") or {}).get("oos") or {}),
        "hurdle_fields": dict(chosen.get("hurdle_fields") or {}),
        "pass": _safe_bool(chosen.get("pass")),
        "selection_basis": normalized_selection_basis,
        "survives": selected_survives,
        "survives_train_val": survives_train_val,
        "hard_reject_reasons": dict(chosen.get("hard_reject_reasons") or {}),
        "survivor_blockers": list(chosen.get("survivor_blockers") or []),
        "train_val_survivor_blockers": list(chosen.get("train_val_survivor_blockers") or []),
        "return_streams": dict(chosen.get("gated_streams") or {}),
        "metadata": {
            **dict(candidate_row.get("metadata") or {}),
            "selection_basis": normalized_selection_basis,
            "activation_rule_id": str(chosen.get("rule_id") or ""),
            "activation_rule_conditions": list(chosen.get("conditions") or []),
            "activation_rule_label": str(chosen.get("label") or ""),
            "activation_signal_lag_days": REGIME_SIGNAL_LAG_DAYS,
            "activation_rule_survives": recommended_survives,
            "activation_rule_survives_train_val": survives_train_val,
        },
    }

    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "rolling_breakout_30m_regime_gate",
        "generated_at": datetime.now(UTC).isoformat(),
        "selection_basis": normalized_selection_basis,
        "selection_uses_oos": normalized_selection_basis != TRAIN_VAL_SELECTION_BASIS,
        "candidate_name": str(candidate_row.get("name") or candidate),
        "candidate_id": str(candidate_row.get("candidate_id") or ""),
        "strategy_class": str(candidate_row.get("strategy_class") or ""),
        "timeframe": TARGET_TIMEFRAME,
        "survival_thresholds": dict(
            TRAIN_VAL_ONLY_SURVIVAL_THRESHOLDS
            if normalized_selection_basis == TRAIN_VAL_SELECTION_BASIS
            else ROLLING_SURVIVAL_THRESHOLDS
        ),
        "survives": recommended_survives,
        "survives_train_val": survives_train_val,
        "recommended_action": "activate_conditionally"
        if recommended_survives
        else "do_not_activate",
        "windows": windows,
        "selected_rule": chosen,
        "evaluated_rules": evaluated_rules,
        "baseline_metrics": {
            split: dict(candidate_row.get(split) or {}) for split in ("train", "val", "oos")
        },
        "feature_snapshot": {
            "rows": len(features),
            "start_date": str(features["date"].min()) if not features.empty else "",
            "end_date": str(features["date"].max()) if not features.empty else "",
            "signal_lag_days": REGIME_SIGNAL_LAG_DAYS,
            "median_breadth_ma96": _safe_float(
                features.get("breadth_ma96", pd.Series(dtype=float)).median(), 0.0
            ),
            "median_breadth_ma192": _safe_float(
                features.get("breadth_ma192", pd.Series(dtype=float)).median(), 0.0
            ),
            "median_basket_vol_ratio": _safe_float(
                features.get("basket_vol_ratio", pd.Series(dtype=float)).median(),
                0.0,
            ),
        },
        "gated_candidate_row": gated_candidate_row,
    }


def write_rolling_breakout_30m_gate(
    decision: Mapping[str, Any],
    *,
    candidate: str = TARGET_CANDIDATE,
    report_root: Path | str = REPORT_ROOT,
    run_name: str = "rolling_breakout_30m_gate",
    selection_basis: str = DEFAULT_SELECTION_BASIS,
    feature_frame: pd.DataFrame | None = None,
    evaluation_payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = build_rolling_breakout_30m_gate(
        decision,
        candidate=candidate,
        selection_basis=selection_basis,
        feature_frame=feature_frame,
        evaluation_payload=evaluation_payload,
    )
    followup_root = Path(report_root) / "followup_status"
    followup_root.mkdir(parents=True, exist_ok=True)
    json_path = followup_root / f"{run_name}_latest.json"
    md_path = followup_root / f"{run_name}_latest.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    selected = dict(payload["selected_rule"])
    selected_oos = dict((selected.get("metrics") or {}).get("oos") or {})
    lines = [
        "# rolling breakout 30m regime gate",
        "",
        f"- generated_at: `{payload['generated_at']}`",
        f"- candidate: `{payload['candidate_name']}`",
        f"- selection_basis: `{payload.get('selection_basis')}`",
        f"- selection_uses_oos: `{payload.get('selection_uses_oos')}`",
        f"- selected_rule: `{selected.get('rule_id')}`",
        f"- label: {selected.get('label')}",
        f"- conditions: `{', '.join(selected.get('conditions') or [])}`",
        f"- signal_lag_days: `{selected.get('signal_lag_days')}`",
        f"- survives: `{payload.get('survives')}`",
        f"- survives_train_val: `{payload.get('survives_train_val')}`",
        f"- recommended_action: `{payload.get('recommended_action')}`",
        f"- gated_oos_return: `{_safe_float(selected_oos.get('return'), 0.0):.4%}`",
        f"- gated_oos_sharpe: `{_safe_float(selected_oos.get('sharpe'), 0.0):.3f}`",
        f"- gated_oos_sortino: `{_safe_float(selected_oos.get('sortino'), 0.0):.3f}`",
        f"- gated_oos_calmar: `{_safe_float(selected_oos.get('calmar'), 0.0):.3f}`",
        f"- gated_oos_max_drawdown: `{_safe_float(selected_oos.get('max_drawdown'), 0.0):.4%}`",
        f"- gated_oos_trade_count: `{int(_safe_float(selected_oos.get('trade_count'), 0.0))}`",
        f"- gated_oos_pbo: `{_safe_float(selected_oos.get('pbo'), 0.0):.3f}`",
        "",
        "## evaluated rules",
    ]
    for rule in list(payload.get("evaluated_rules") or []):
        oos = dict((rule.get("metrics") or {}).get("oos") or {})
        lines.append(
            f"- `{rule.get('rule_id')}`: return={_safe_float(oos.get('return'), 0.0):.4%} | "
            f"sharpe={_safe_float(oos.get('sharpe'), 0.0):.3f} | "
            f"pbo={_safe_float(oos.get('pbo'), 0.0):.3f} | "
            f"activation={_safe_float(oos.get('activation_ratio'), 0.0):.2%} | "
            f"survives={rule.get('survives')}"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "payload": payload,
        "json_path": str(json_path.resolve()),
        "md_path": str(md_path.resolve()),
    }


if __name__ == "__main__":
    decision_path = REPORT_ROOT / "exact_window_decision_latest.json"
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    result = write_rolling_breakout_30m_gate(decision)
    print(result["json_path"])
    print(result["md_path"])
