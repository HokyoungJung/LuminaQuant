"""Real-data adapters and outcome-ledger helpers for Crypto/FX Alpha Zoo research."""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from lumina_quant.alpha_zoo.crypto_fx_factors import factor_columns, normalize_symbol
from lumina_quant.research.candidate_outcome_ledger import CandidateOutcomeLedger
from lumina_quant.research.triple_barrier import label_triple_barrier

REQUIRED_OHLCV = ("open", "high", "low", "close", "volume")
OPTIONAL_REAL_FIELDS = (
    "funding_rate",
    "open_interest",
    "taker_buy_base_volume",
    "taker_sell_base_volume",
    "taker_buy_quote_volume",
    "taker_sell_quote_volume",
    "liquidation_long_notional",
    "liquidation_short_notional",
)
LOCKED_OOS_SPLITS = {"locked_oos", "oos"}
TRAIN_VALIDATION_SPLITS = {"train", "validation"}


@dataclass(frozen=True, slots=True)
class RealDataBundle:
    """Normalized long-frame payload plus fail-closed source metadata."""

    frame: pd.DataFrame
    metadata: dict[str, Any]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except Exception:
        return default
    return parsed if math.isfinite(parsed) else default


def _read_parquet_with_polars(path: Path) -> pd.DataFrame:
    try:
        import polars as pl
    except Exception as exc:  # pragma: no cover - depends on optional parquet engines
        raise ImportError(
            "parquet input requires pandas parquet support or installed polars"
        ) from exc
    return pd.DataFrame(pl.read_parquet(path).to_dicts())


def read_tabular_frame(path: str | Path) -> pd.DataFrame:
    """Read CSV/parquet without requiring pandas' optional parquet engine."""
    resolved = Path(path).expanduser().resolve()
    suffix = resolved.suffix.lower()
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(resolved)
    if suffix == ".parquet":
        try:
            return pd.read_parquet(resolved)
        except ImportError:
            return _read_parquet_with_polars(resolved)
    raise ValueError(f"unsupported input format: {resolved}")


def _compact_symbol(symbol: str) -> str:
    return str(symbol).upper().replace("/", "").replace("-", "").replace("_", "")


def _display_symbol_from_prefix(prefix: str) -> str:
    token = _compact_symbol(prefix)
    if token.endswith("USDT") and len(token) > 4:
        return f"{token[:-4]}/USDT"
    if token.endswith("USD") and len(token) > 3:
        return f"{token[:-3]}/USD"
    return token


def discover_latest_current_tail_cache(
    cache_dir: str | Path | None = None,
    *,
    preferred_end: str | None = "2026-05-08",
) -> Path:
    """Find the latest cached profit-moonshot current-tail joined panel."""
    root = Path(cache_dir or (_repo_root() / "var/cache/profit_moonshot_fresh_start"))
    candidates: list[tuple[str, float, Path]] = []
    for meta_path in root.glob("joined_panel_*.json"):
        try:
            payload = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        panel_cache = dict(payload.get("panel_cache") or {})
        panel_path = Path(str(panel_cache.get("path") or meta_path.with_suffix(".parquet")))
        if not panel_path.is_absolute():
            panel_path = (_repo_root() / panel_path).resolve()
        if not panel_path.exists():
            continue
        end = str(panel_cache.get("end") or "")
        mtime = panel_path.stat().st_mtime
        candidates.append((end, mtime, panel_path))
    if not candidates:
        raise FileNotFoundError(f"no joined_panel_*.parquet cache found under {root}")
    if preferred_end:
        preferred = [item for item in candidates if item[0] == preferred_end]
        if preferred:
            return max(preferred, key=lambda item: item[1])[2]
    return max(candidates, key=lambda item: (item[0], item[1]))[2]


def _is_wide_current_tail_frame(frame: pd.DataFrame) -> bool:
    columns = {str(column).lower() for column in frame.columns}
    return (
        ("datetime" in columns or "timestamp" in columns)
        and any(column.endswith("_close") for column in columns)
        and "symbol" not in columns
    )


def _wide_prefixes(columns: list[str]) -> list[str]:
    prefixes = sorted(
        {column[: -len("_close")] for column in columns if column.lower().endswith("_close")}
    )
    return [prefix for prefix in prefixes if f"{prefix}_open" in columns]


def _wide_current_tail_to_long(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    columns = [str(column) for column in frame.columns]
    lower_map = {column.lower(): column for column in columns}
    time_column = lower_map.get("datetime") or lower_map.get("timestamp")
    if time_column is None:
        raise ValueError("wide current-tail panel requires datetime or timestamp column")
    prefixes = _wide_prefixes(columns)
    if not prefixes:
        raise ValueError("wide current-tail panel has no *_open/*_close symbol columns")

    rows: list[pd.DataFrame] = []
    symbol_meta: dict[str, Any] = {}
    for prefix in prefixes:
        symbol = _display_symbol_from_prefix(prefix)
        output = pd.DataFrame({"timestamp": pd.to_datetime(frame[time_column]), "symbol": symbol})
        observed: list[str] = []
        imputed: list[str] = []
        source_columns: dict[str, str] = {}
        for field in REQUIRED_OHLCV:
            source = f"{prefix}_{field}"
            if source in frame.columns:
                output[field] = pd.to_numeric(frame[source], errors="coerce")
                observed.append(field)
                source_columns[field] = source
            elif field == "volume":
                output[field] = 0.0
                imputed.append(field)
            else:
                close_source = f"{prefix}_close"
                output[field] = pd.to_numeric(frame[close_source], errors="coerce")
                imputed.append(field)
        vwap = (output["high"] + output["low"] + output["close"]) / 3.0
        output["vwap"] = vwap
        imputed.append("vwap")
        for field in OPTIONAL_REAL_FIELDS:
            source = f"{prefix}_{field}"
            if source in frame.columns:
                output[field] = pd.to_numeric(frame[source], errors="coerce")
                observed.append(field)
                source_columns[field] = source
        rows.append(output)
        coverage = {
            field: float(output[field].notna().mean())
            if field in output.columns and len(output)
            else 0.0
            for field in (*REQUIRED_OHLCV, *OPTIONAL_REAL_FIELDS)
            if field in output.columns
        }
        symbol_meta[symbol] = {
            "source_prefix": prefix,
            "rows": len(output),
            "observed_fields": sorted(set(observed)),
            "imputed_fields": sorted(set(imputed)),
            "source_columns": source_columns,
            "coverage_ratio_by_field": coverage,
            "required_ohlcv_observed": all(field in observed for field in REQUIRED_OHLCV),
        }
    long_frame = pd.concat(rows, ignore_index=True).sort_values(["timestamp", "symbol"])
    return long_frame, {
        "input_shape": [int(frame.shape[0]), int(frame.shape[1])],
        "input_format": "wide_current_tail_panel",
        "time_column": str(time_column),
        "symbols": sorted(symbol_meta),
        "symbol_coverage": symbol_meta,
    }


def _long_frame_source_metadata(frame: pd.DataFrame) -> dict[str, Any]:
    out = frame.copy()
    out["symbol"] = out["symbol"].astype(str)
    symbol_meta: dict[str, Any] = {}
    for symbol, group in out.groupby("symbol", sort=True):
        observed = [
            field
            for field in (*REQUIRED_OHLCV, *OPTIONAL_REAL_FIELDS, "vwap")
            if field in group.columns
        ]
        imputed = []
        for field in ("vwap",):
            if field not in group.columns:
                imputed.append(field)
        coverage = {
            field: float(pd.to_numeric(group[field], errors="coerce").notna().mean())
            for field in observed
        }
        symbol_meta[str(symbol)] = {
            "rows": len(group),
            "observed_fields": sorted(observed),
            "imputed_fields": sorted(imputed),
            "coverage_ratio_by_field": coverage,
            "required_ohlcv_observed": all(field in observed for field in REQUIRED_OHLCV),
        }
    return {
        "input_shape": [int(frame.shape[0]), int(frame.shape[1])],
        "input_format": "long_symbol_rows",
        "symbols": sorted(symbol_meta),
        "symbol_coverage": symbol_meta,
    }


def _join_external_state(
    frame: pd.DataFrame, external_state_csv: str | Path | None
) -> tuple[pd.DataFrame, dict[str, Any]]:
    raw = str(external_state_csv or "").strip()
    metadata: dict[str, Any] = {
        "enabled": bool(raw),
        "path": raw,
        "joined": False,
        "rows": 0,
        "feature_columns": [],
    }
    if not raw:
        return frame, metadata
    path = Path(raw).expanduser()
    if not path.exists():
        metadata["error"] = "external_state_csv_missing"
        return frame, metadata
    try:
        external = pd.read_csv(path)
    except Exception as exc:  # pragma: no cover - defensive metadata path
        metadata["error"] = f"external_state_csv_read_failed:{exc}"
        return frame, metadata
    date_col = "effective_date" if "effective_date" in external.columns else "date"
    if date_col not in external.columns:
        metadata["error"] = "external_state_csv_missing_date_column"
        return frame, metadata
    features = [column for column in external.columns if str(column).startswith("external_")]
    if not features:
        metadata["error"] = "external_state_csv_missing_external_columns"
        return frame, metadata
    external = external[[date_col, *features]].copy()
    external["_join_date"] = pd.to_datetime(external[date_col], errors="coerce").dt.date
    external = external.dropna(subset=["_join_date"]).drop_duplicates("_join_date", keep="last")
    out = frame.copy()
    out["_join_date"] = pd.to_datetime(out["timestamp"], errors="coerce").dt.date
    out = out.merge(external[["_join_date", *features]], on="_join_date", how="left")
    out = out.drop(columns=["_join_date"])
    metadata.update(
        {
            "joined": True,
            "rows": len(external),
            "feature_columns": features,
            "coverage_ratio_by_feature": {
                feature: float(out[feature].notna().mean()) if len(out) else 0.0
                for feature in features
            },
        }
    )
    return out, metadata


def normalize_real_data_frame(
    frame: pd.DataFrame,
    *,
    source_path: str | Path,
    external_state_csv: str | Path | None = None,
    strict_real_data: bool = False,
) -> RealDataBundle:
    """Normalize wide/long real data and report source coverage fail-closed."""
    if _is_wide_current_tail_frame(frame):
        normalized, coverage = _wide_current_tail_to_long(frame)
    else:
        if "timestamp" not in frame.columns and "datetime" in frame.columns:
            frame = frame.rename(columns={"datetime": "timestamp"})
        required = {"timestamp", "symbol", "close"}
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"long input missing required columns: {', '.join(missing)}")
        normalized = frame.copy()
        normalized["timestamp"] = pd.to_datetime(normalized["timestamp"], errors="coerce")
        coverage = _long_frame_source_metadata(normalized)
    normalized, external_meta = _join_external_state(normalized, external_state_csv)
    normalized = normalized.dropna(subset=["timestamp", "symbol", "close"]).copy()

    symbol_coverage = dict(coverage.get("symbol_coverage") or {})
    required_failures = [
        symbol
        for symbol, item in symbol_coverage.items()
        if not bool(item.get("required_ohlcv_observed"))
    ]
    validity_reasons: list[str] = []
    if required_failures:
        validity_reasons.append("required_real_ohlcv_coverage_missing")
    if normalized.empty:
        validity_reasons.append("normalized_frame_empty")
    if strict_real_data and validity_reasons:
        raise ValueError("real-data coverage failed closed: " + ",".join(validity_reasons))

    split_counts = Counter(
        str(item) for item in normalized.get("split", pd.Series(dtype=str)).dropna()
    )
    metadata = {
        "artifact_kind": "crypto_fx_alpha_zoo_real_data_source_coverage",
        "generated_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "source_path": str(source_path),
        "strict_real_data": bool(strict_real_data),
        "row_count": len(normalized),
        "timestamp_min": str(pd.to_datetime(normalized["timestamp"]).min())
        if len(normalized)
        else None,
        "timestamp_max": str(pd.to_datetime(normalized["timestamp"]).max())
        if len(normalized)
        else None,
        "input": coverage,
        "external_state": external_meta,
        "split_counts_pre_factor": dict(split_counts),
        "strategy_validity": {
            "pass": not validity_reasons,
            "calendar_primary": False,
            "uses_locked_oos_for_selection": False,
            "locked_oos_role": "gate_report_only",
            "rejection_reasons": validity_reasons,
        },
    }
    return RealDataBundle(frame=normalized, metadata=metadata)


def load_real_data_bundle(
    input_path: str | Path | None = None,
    *,
    current_tail_cache: str | Path | None = None,
    external_state_csv: str | Path | None = None,
    strict_real_data: bool = False,
) -> RealDataBundle:
    """Load either explicit input or the latest current-tail cached panel."""
    source = Path(input_path).expanduser().resolve() if input_path else None
    if source is None:
        source = (
            Path(current_tail_cache).expanduser().resolve()
            if current_tail_cache
            else discover_latest_current_tail_cache()
        )
    frame = read_tabular_frame(source)
    return normalize_real_data_frame(
        frame,
        source_path=source,
        external_state_csv=external_state_csv,
        strict_real_data=strict_real_data,
    )


def _regime_bucket(row: pd.Series) -> str:
    for key in ("external_risk_off_score_lag1", "external_risk_off_score"):
        if key in row.index:
            value = _safe_float(row.get(key), default=float("nan"))
            if math.isfinite(value):
                if value >= 0.75:
                    return "risk_off"
                if value <= -0.75:
                    return "risk_on"
                return "neutral"
    return "external_unknown"


def _volatility_bucket(values: pd.Series, value: Any) -> str:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    parsed = _safe_float(value, default=float("nan"))
    if clean.empty or not math.isfinite(parsed):
        return "vol_unknown"
    q33 = float(clean.quantile(0.33))
    q66 = float(clean.quantile(0.66))
    if parsed <= q33:
        return "vol_low"
    if parsed >= q66:
        return "vol_high"
    return "vol_mid"


def build_candidate_outcome_records(
    labeled: pd.DataFrame,
    selected_factors: list[dict[str, Any]],
    *,
    entry_quantile: float = 0.9,
    max_records_per_factor_side_split: int = 120,
    stop_loss_pct: float = 0.025,
    take_profit_pct: float = 0.05,
    max_hold_bars: int = 24,
    fee_bps: float = 4.0,
    slippage_bps: float = 2.0,
) -> list[dict[str, Any]]:
    """Convert selected factor extremes into triple-barrier outcome records."""
    if not selected_factors or labeled.empty:
        return []
    q = min(0.99, max(0.50, float(entry_quantile)))
    max_records = max(1, int(max_records_per_factor_side_split))
    out: list[dict[str, Any]] = []
    data = labeled.sort_values(["symbol", "timestamp"]).copy().reset_index(drop=True)
    if "split" not in data.columns:
        raise ValueError("labeled frame must include split column before ledger generation")
    selected_names = [
        str(row.get("factor")) for row in selected_factors if str(row.get("factor")) in data.columns
    ]
    for factor in selected_names:
        factor_values = pd.to_numeric(data[factor], errors="coerce")
        if factor_values.dropna().empty:
            continue
        for split in ("train", "validation", "locked_oos"):
            split_mask = data["split"].astype(str).eq(split)
            split_values = factor_values[split_mask].dropna()
            if len(split_values) < 10:
                continue
            high_threshold = float(split_values.quantile(q))
            low_threshold = float(split_values.quantile(1.0 - q))
            for symbol, group in data[split_mask].groupby("symbol", sort=True):
                group = group.sort_values("timestamp").copy()
                if len(group) <= max_hold_bars + 1:
                    continue
                closes = pd.to_numeric(group["close"], errors="coerce").to_numpy(dtype=float)
                highs = pd.to_numeric(group.get("high", group["close"]), errors="coerce").to_numpy(
                    dtype=float
                )
                lows = pd.to_numeric(group.get("low", group["close"]), errors="coerce").to_numpy(
                    dtype=float
                )
                values = pd.to_numeric(group[factor], errors="coerce")
                vol_bucket_values = (
                    group["realized_vol_24"] if "realized_vol_24" in group.columns else values.abs()
                )
                side_entries = {
                    "LONG": group.index[values >= high_threshold].tolist(),
                    "SHORT": group.index[values <= low_threshold].tolist(),
                }
                local_index = {global_idx: idx for idx, global_idx in enumerate(group.index)}
                for side, indices in side_entries.items():
                    for global_idx in indices[:max_records]:
                        entry_idx = local_index.get(global_idx)
                        if entry_idx is None or entry_idx >= len(group) - 1:
                            continue
                        try:
                            outcome = label_triple_barrier(
                                closes,
                                entry_index=entry_idx,
                                side=side,
                                stop_loss_pct=stop_loss_pct,
                                take_profit_pct=take_profit_pct,
                                max_hold_bars=max_hold_bars,
                                highs=highs,
                                lows=lows,
                                fee_bps=fee_bps,
                                slippage_bps=slippage_bps,
                                funding_bps_per_bar=0.0,
                            )
                        except ValueError:
                            continue
                        entry_row = group.iloc[entry_idx]
                        exit_row = group.iloc[outcome.exit_index]
                        out.append(
                            {
                                "candidate_id": str(factor),
                                "strategy": "CryptoFxAlphaZooStateStrategy",
                                "split": split,
                                "symbol": str(symbol),
                                "asset": normalize_symbol(symbol),
                                "side": outcome.side,
                                "entry_time": pd.Timestamp(entry_row["timestamp"]).isoformat(),
                                "exit_time": pd.Timestamp(exit_row["timestamp"]).isoformat(),
                                "barrier_type": outcome.barrier_type.value,
                                "net_pnl_bps": float(outcome.net_pnl_bps),
                                "mae_bps": float(outcome.max_adverse_excursion * 10_000.0),
                                "mfe_bps": float(outcome.max_favorable_excursion * 10_000.0),
                                "factor_bucket": "top" if side == "LONG" else "bottom",
                                "regime_bucket": _regime_bucket(entry_row),
                                "volatility_bucket": _volatility_bucket(
                                    vol_bucket_values,
                                    entry_row.get("realized_vol_24", entry_row.get(factor)),
                                ),
                                "feature_snapshot": {
                                    "factor": factor,
                                    "factor_value": _safe_float(entry_row.get(factor)),
                                    "entry_quantile": q,
                                    "selection_split": split,
                                },
                                "costs": {
                                    "fee_bps_round_trip": float(2.0 * fee_bps),
                                    "slippage_bps_round_trip": float(2.0 * slippage_bps),
                                },
                                "strategy_validity": {
                                    "pass": True,
                                    "calendar_primary": False,
                                    "uses_locked_oos_for_selection": False,
                                    "locked_oos_role": "gate_report_only",
                                    "primary_signal_type": "formulaic_state_factor",
                                    "rejection_reasons": [],
                                },
                            }
                        )
    return out


def write_candidate_outcome_ledger(
    path: str | Path, records: list[dict[str, Any]]
) -> dict[str, Any]:
    ledger_path = Path(path)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger_path.write_text("", encoding="utf-8")
    ledger = CandidateOutcomeLedger(ledger_path)
    for record in records:
        ledger.append(record)
    summary = ledger.summary()
    split_counts = Counter(str(row.get("split")) for row in records)
    summary.update(
        {
            "split_counts": dict(split_counts),
            "train_validation_record_count": sum(
                split_counts.get(split, 0) for split in TRAIN_VALIDATION_SPLITS
            ),
            "locked_oos_record_count": sum(
                split_counts.get(split, 0) for split in LOCKED_OOS_SPLITS
            ),
            "strategy": "CryptoFxAlphaZooStateStrategy",
            "calendar_primary": False,
            "uses_locked_oos_for_selection": False,
            "locked_oos_role": "gate_report_only",
        }
    )
    return summary


def summarize_factor_source_coverage(frame: pd.DataFrame) -> dict[str, Any]:
    """Small coverage supplement after factor computation and split assignment."""
    split_counts = Counter(str(item) for item in frame.get("split", pd.Series(dtype=str)).dropna())
    symbol_split_counts: dict[str, dict[str, int]] = defaultdict(dict)
    if "symbol" in frame.columns and "split" in frame.columns:
        for (symbol, split), group in frame.groupby(["symbol", "split"], sort=True):
            symbol_split_counts[str(symbol)][str(split)] = len(group)
    factor_non_null = {
        factor: float(pd.to_numeric(frame[factor], errors="coerce").notna().mean())
        for factor in factor_columns()
        if factor in frame.columns
    }
    return {
        "row_count": len(frame),
        "split_counts": dict(split_counts),
        "symbol_split_counts": dict(symbol_split_counts),
        "factor_non_null_coverage": factor_non_null,
    }
