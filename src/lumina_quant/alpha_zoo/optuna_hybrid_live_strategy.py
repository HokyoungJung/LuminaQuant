"""Paper/testnet live adapter for the frozen Alpha Zoo Optuna hybrid."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lumina_quant.core.events import SignalEvent
from lumina_quant.market_data import normalize_timeframe_token
from lumina_quant.strategy import Strategy

REPO_ROOT = Path(__file__).resolve().parents[3]
ALPHA_V2_ROOT = (
    Path("var")
    / "reports"
    / "profit_moonshot_20260501"
    / "current_tail_20260508"
    / "alpha_v2"
)
DEFAULT_OPTUNA_HYBRID_ARTIFACT = (
    ALPHA_V2_ROOT
    / "alpha_zoo_integer_leverage_optuna_hybrid_decision_20260524"
    / "alpha_zoo_integer_leverage_optuna_hybrid_decision_latest.json"
)
DEFAULT_INTEGER_PORTFOLIO_ARTIFACT = (
    ALPHA_V2_ROOT
    / "alpha_zoo_corr_integer_leverage_portfolio_20260524"
    / "alpha_zoo_corr_integer_leverage_portfolio_latest.json"
)
DEFAULT_SELECTED_PROFILE_ID = "hybrid_v3_5_optuna_three_profile_blend"
PROFILE_IDS = (
    "balanced_mdd12_gross5",
    "growth_mdd20_gross8",
    "aggressive_mdd30_gross10_shadow",
)
TRADED_SYMBOLS = ("ETHUSDT", "SOLUSDT", "TRXUSDT")
WATCH_SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "TRXUSDT")
ROUND_TRIP_COST_BPS = 10.0
RETURN_PER_TURNOVER_THRESHOLD_BPS = 10.0
INTRABAR_RISK_TIMEFRAMES = ("1m", "5m")
INTRABAR_ATR_LOOKBACK = 14
INTRABAR_STOP_ATR_MULT = 2.0
INTRABAR_MIN_STOP_COST_MULT = 12.0
INTRABAR_MAX_STOP_COST_MULT = 80.0


@dataclass(frozen=True, slots=True)
class SourceSleeve:
    model_id: str
    family: str
    symbol: str
    timeframe: str
    side: str
    lookback: int
    allocation_fraction: float
    source_leverage: float
    source_artifact_path: str = ""
    entry_threshold: float = 0.0
    exit_threshold: float = 0.0
    min_hold_bars: int = 0
    cooldown_bars: int = 0
    filter_label: str = "none"
    atr_mult: float = 0.0
    rel_threshold: float = 0.0
    trail_atr_mult: float = 0.0
    base_symbol: str = "BTCUSDT"
    z_entry: float = 0.0


@dataclass(frozen=True, slots=True)
class SourceProfile:
    profile_id: str
    selected_model_ids: tuple[str, ...]
    leverage_map: dict[str, int]


@dataclass(frozen=True, slots=True)
class SleeveDecision:
    signal: int
    completed_key: str
    event_time: Any
    price: float
    diagnostics: dict[str, Any]


@dataclass(frozen=True, slots=True)
class IntrabarProtectionPlan:
    enabled: bool
    source_timeframe: str
    guard_mode: str
    stop_loss: float | None
    take_profit: float | None
    trailing_percent: float | None
    stop_distance_pct: float
    atr_pct: float | None
    notes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class AlphaZooOptunaHybridLiveConfig:
    selected_profile_id: str
    optuna_artifact_path: Path
    integer_artifact_path: Path
    selected_profile: dict[str, Any]
    final_profile_weights: dict[str, float]
    average_profile_weights: dict[str, float]
    best_params: dict[str, Any]
    learned_params: dict[str, Any]
    source_profiles: tuple[SourceProfile, ...]
    source_sleeves: tuple[SourceSleeve, ...]
    watch_symbols: tuple[str, ...]
    governance: dict[str, Any]


class AlphaZooV35HybridAllocator:
    """Pure frozen v3.5 allocation surface for live paper/testnet startup."""

    def __init__(self, config: AlphaZooOptunaHybridLiveConfig) -> None:
        self._config = config

    @property
    def final_profile_weights(self) -> dict[str, float]:
        return dict(self._config.final_profile_weights)

    @property
    def average_profile_weights(self) -> dict[str, float]:
        return dict(self._config.average_profile_weights)

    def profile_weights_for_live(self) -> dict[str, float]:
        """Return frozen post-dampening profile weights from the selected artifact."""
        return self.final_profile_weights

    def allocation_metadata(self) -> dict[str, Any]:
        weights = self.profile_weights_for_live()
        return {
            "hybrid_version": str(self._config.selected_profile.get("hybrid_version") or "v3_5"),
            "selected_profile_id": self._config.selected_profile_id,
            "final_profile_weights": weights,
            "average_profile_weights_train_validation": self.average_profile_weights,
            "profile_weight_sum": float(sum(weights.values())),
            "cash_or_dampened_exposure_fraction": max(0.0, 1.0 - float(sum(weights.values()))),
            "best_params": dict(self._config.best_params),
            "learned_params": dict(self._config.learned_params),
            "optimizer_runtime_dependency": "none_frozen_artifact_only",
        }


def _resolve_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    return candidate.resolve()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object in {path}")
    return payload


def _compact_symbol(symbol: str) -> str:
    return str(symbol or "").replace("/", "").replace("-", "").upper()


def _live_symbol(symbol: str) -> str:
    token = _compact_symbol(symbol)
    if token.endswith("USDT") and len(token) > 4:
        return f"{token[:-4]}/USDT"
    return token


def _symbol_aliases(symbol: str) -> tuple[str, ...]:
    compact = _compact_symbol(symbol)
    live = _live_symbol(compact)
    return tuple(dict.fromkeys([str(symbol), compact, live]))


def _token_float(token: str) -> float:
    return float(str(token).replace("p", "."))


def _expected_model_id(parts: list[Any], prefix: str) -> str:
    text = "_".join(str(part).replace("/", "_").replace(".", "p") for part in parts)
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
    return f"{prefix}_{text}_{digest}".lower()


def _governance_false(payload: dict[str, Any], key: str) -> bool:
    return payload.get(key) is False


def _validate_paper_only_governance(
    payload: dict[str, Any], *, label: str, require_cost: bool = True
) -> None:
    if payload.get("paper_testnet_only") is not True:
        raise ValueError(f"{label} must be paper_testnet_only=true")
    for key in ("ready_for_real", "real_money_execution", "real_execution_allowed"):
        if not _governance_false(payload, key):
            raise ValueError(f"{label} must keep {key}=false")
    if require_cost:
        cost = float(payload.get("research_primary_round_trip_cost_bps", -1.0))
        threshold = float(payload.get("return_per_turnover_threshold_bps", -1.0))
        if not math.isclose(cost, ROUND_TRIP_COST_BPS, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(f"{label} must preserve 10bps round-trip cost")
        if not math.isclose(
            threshold,
            RETURN_PER_TURNOVER_THRESHOLD_BPS,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError(f"{label} must preserve 10bps return-per-turnover threshold")


def _validate_locked_oos_policy(optuna_payload: dict[str, Any]) -> None:
    policy = dict(optuna_payload.get("selection_policy") or {})
    for key in (
        "uses_locked_oos_for_discovery",
        "uses_locked_oos_for_objective",
        "uses_locked_oos_for_parameter_fitting",
        "uses_locked_oos_for_pruning",
        "uses_locked_oos_for_selection",
    ):
        if policy.get(key) is not False:
            raise ValueError(f"Optuna artifact violates locked-OOS policy: {key}")
    selected = dict(optuna_payload.get("selected_optuna_hybrid_profile") or {})
    optuna = dict(selected.get("optuna") or {})
    for key in (
        "uses_locked_oos_for_objective",
        "uses_locked_oos_for_parameter_fitting",
        "uses_locked_oos_for_pruning",
        "uses_locked_oos_for_selection",
    ):
        if optuna.get(key) is not False:
            raise ValueError(f"Selected Optuna profile violates locked-OOS policy: {key}")


def _validate_integer_artifact(integer_payload: dict[str, Any]) -> None:
    review = dict(integer_payload.get("strategy_integrity_review") or {})
    if str(review.get("status") or "").lower() != "pass":
        raise ValueError("integer portfolio strategy_integrity_review must pass")
    token_hits = dict(review.get("calendar_date_rule_check") or {}).get("hits") or []
    if token_hits:
        raise ValueError("integer portfolio artifact contains forbidden date-rule hits")
    locked = dict(review.get("locked_oos_policy_check") or {})
    for key in (
        "uses_locked_oos_for_discovery",
        "uses_locked_oos_for_objective",
        "uses_locked_oos_for_parameter_fitting",
        "uses_locked_oos_for_pruning",
        "uses_locked_oos_for_selection",
    ):
        if locked.get(key) is not False:
            raise ValueError(f"integer portfolio violates locked-OOS policy: {key}")


def _extract_selected_profile(
    optuna_payload: dict[str, Any], selected_profile_id: str
) -> dict[str, Any]:
    selected = dict(optuna_payload.get("selected_optuna_hybrid_profile") or {})
    if selected.get("profile_id") == selected_profile_id:
        return selected
    for row in list(optuna_payload.get("comparison_rows") or []):
        if isinstance(row, dict) and row.get("profile_id") == selected_profile_id:
            return dict(row)
    raise ValueError(f"selected profile not found: {selected_profile_id}")


def _profile_rows_by_id(integer_payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for row in list(integer_payload.get("profile_decision_rows") or []):
        if isinstance(row, dict) and row.get("profile_id"):
            rows[str(row["profile_id"])] = dict(row)
    for key in ("selected_profile", "selected_relaxed_profile", "selected_shadow_profile"):
        row = integer_payload.get(key)
        if isinstance(row, dict) and row.get("profile_id"):
            rows.setdefault(str(row["profile_id"]), dict(row))
    return rows


def _load_corr_rows(integer_payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    corr_path = str(integer_payload.get("source_correlation_artifact") or "").strip()
    if corr_path:
        path = _resolve_path(corr_path)
        if path.exists():
            corr_payload = _read_json(path)
            rows = {}
            for row in list(corr_payload.get("correlation_decision_rows") or []):
                if isinstance(row, dict) and row.get("model_id"):
                    rows[str(row["model_id"])] = dict(row)
            return rows
    rows = {}
    review = dict(integer_payload.get("strategy_integrity_review") or {})
    for row in list(review.get("strategy_rows") or []):
        if isinstance(row, dict) and row.get("model_id"):
            rows[str(row["model_id"])] = dict(row)
    return rows


def _parse_debounced(model_id: str, row: dict[str, Any]) -> SourceSleeve:
    pattern = re.compile(
        r"debrepair_debounced_efficiency_repair_(?P<tf>[^_]+)_(?P<symbol>[^_]+)_"
        r"(?P<side>long_short|short_only|long_only)_lb(?P<lookback>\d+)_"
        r"e(?P<entry>-?[0-9p]+)_x(?P<exit>-?[0-9p]+)_hold(?P<hold>\d+)_"
        r"cool(?P<cool>\d+)_(?P<filter>[^_]+)_(?P<lev>[0-9p]+)x_(?P<alloc>[0-9p]+)_"
    )
    match = pattern.search(model_id)
    if match is None:
        raise ValueError(f"cannot parse debounced source model id: {model_id}")
    return SourceSleeve(
        model_id=model_id,
        family="debounced_momentum_hysteresis_efficiency_repair",
        symbol=_compact_symbol(match.group("symbol")),
        timeframe=normalize_timeframe_token(match.group("tf")),
        side=match.group("side"),
        lookback=int(match.group("lookback")),
        entry_threshold=_token_float(match.group("entry")),
        exit_threshold=_token_float(match.group("exit")),
        min_hold_bars=int(match.group("hold")),
        cooldown_bars=int(match.group("cool")),
        filter_label=match.group("filter"),
        source_leverage=float(row.get("leverage") or _token_float(match.group("lev"))),
        allocation_fraction=float(row.get("allocation_fraction") or _token_float(match.group("alloc"))),
        source_artifact_path=str(row.get("source_artifact_path") or ""),
    )


def _parse_booster(model_id: str, row: dict[str, Any]) -> SourceSleeve:
    pattern = re.compile(
        r"a30fb_booster_rs_chandelier_(?P<tf>[^_]+)_(?P<symbol>[^_]+)_lb(?P<lookback>\d+)_"
        r"atr(?P<atr>[0-9p]+)_rel(?P<rel>[0-9p]+)_trail(?P<trail>[0-9p]+)_"
        r"hold(?P<hold>\d+)_(?P<lev>[0-9p]+)x_(?P<alloc>[0-9p]+)_"
    )
    match = pattern.search(model_id)
    if match is None:
        raise ValueError(f"cannot parse booster source model id: {model_id}")
    return SourceSleeve(
        model_id=model_id,
        family="relative_strength_chandelier_breakout",
        symbol=_compact_symbol(match.group("symbol")),
        timeframe=normalize_timeframe_token(match.group("tf")),
        side="long_short",
        lookback=int(match.group("lookback")),
        atr_mult=_token_float(match.group("atr")),
        rel_threshold=_token_float(match.group("rel")),
        trail_atr_mult=_token_float(match.group("trail")),
        min_hold_bars=int(match.group("hold")),
        cooldown_bars=2,
        exit_threshold=-0.005,
        filter_label="adx15_btc_regime_atr_trail",
        source_leverage=float(row.get("leverage") or _token_float(match.group("lev"))),
        allocation_fraction=float(row.get("allocation_fraction") or _token_float(match.group("alloc"))),
        source_artifact_path=str(row.get("source_artifact_path") or ""),
    )


def _parse_residual(model_id: str, row: dict[str, Any]) -> SourceSleeve:
    pattern = re.compile(
        r"a30fb_asset_diverse_residual_reclaim_(?P<tf>[^_]+)_(?P<symbol>[^_]+)_"
        r"(?P<base>[^_]+)_lb(?P<lookback>\d+)_z(?P<z>[0-9p]+)_hold(?P<hold>\d+)_"
        r"(?P<lev>[0-9p]+)x_(?P<alloc>[0-9p]+)_"
    )
    match = pattern.search(model_id)
    if match is None:
        raise ValueError(f"cannot parse residual source model id: {model_id}")
    return SourceSleeve(
        model_id=model_id,
        family="relative_residual_reclaim",
        symbol=_compact_symbol(match.group("symbol")),
        timeframe=normalize_timeframe_token(match.group("tf")),
        side="long_short",
        lookback=int(match.group("lookback")),
        z_entry=_token_float(match.group("z")),
        exit_threshold=0.05,
        min_hold_bars=int(match.group("hold")),
        cooldown_bars=2,
        filter_label=f"base_{_compact_symbol(match.group('base')).lower()}_market_momentum_guard",
        base_symbol=_compact_symbol(match.group("base")),
        source_leverage=float(row.get("leverage") or _token_float(match.group("lev"))),
        allocation_fraction=float(row.get("allocation_fraction") or _token_float(match.group("alloc"))),
        source_artifact_path=str(row.get("source_artifact_path") or ""),
    )


def _parse_voladj(model_id: str, row: dict[str, Any]) -> SourceSleeve:
    pattern = re.compile(
        r"a30fb_voladj_trend_(?P<tf>[^_]+)_(?P<symbol>[^_]+)_lb(?P<lookback>\d+)_"
        r"z(?P<threshold>[0-9p]+)_hold(?P<hold>\d+)_cool(?P<cool>\d+)_"
        r"(?P<filter>[^_]+)_(?P<lev>[0-9p]+)x_(?P<alloc>[0-9p]+)_"
    )
    match = pattern.search(model_id)
    if match is None:
        raise ValueError(f"cannot parse vol-adjusted source model id: {model_id}")
    return SourceSleeve(
        model_id=model_id,
        family="volatility_adjusted_trend_persistence",
        symbol=_compact_symbol(match.group("symbol")),
        timeframe=normalize_timeframe_token(match.group("tf")),
        side="long_short",
        lookback=int(match.group("lookback")),
        entry_threshold=_token_float(match.group("threshold")),
        exit_threshold=0.25,
        min_hold_bars=int(match.group("hold")),
        cooldown_bars=int(match.group("cool")),
        filter_label=match.group("filter"),
        source_leverage=float(row.get("leverage") or _token_float(match.group("lev"))),
        allocation_fraction=float(row.get("allocation_fraction") or _token_float(match.group("alloc"))),
        source_artifact_path=str(row.get("source_artifact_path") or ""),
    )


def _source_sleeve_from_row(model_id: str, row: dict[str, Any]) -> SourceSleeve:
    if model_id.startswith("debrepair_debounced_efficiency_repair_"):
        return _parse_debounced(model_id, row)
    if model_id.startswith("a30fb_booster_rs_chandelier_"):
        return _parse_booster(model_id, row)
    if model_id.startswith("a30fb_asset_diverse_residual_reclaim_"):
        return _parse_residual(model_id, row)
    if model_id.startswith("a30fb_voladj_trend_"):
        return _parse_voladj(model_id, row)
    raise ValueError(f"unsupported Alpha Zoo source family for live adapter: {model_id}")


def _validate_source_model_id(sleeve: SourceSleeve) -> None:
    if sleeve.family == "debounced_momentum_hysteresis_efficiency_repair":
        expected = _expected_model_id(
            [
                "debounced_efficiency_repair",
                sleeve.timeframe,
                sleeve.symbol,
                sleeve.side,
                f"lb{sleeve.lookback}",
                f"e{sleeve.entry_threshold}",
                f"x{sleeve.exit_threshold}",
                f"hold{sleeve.min_hold_bars}",
                f"cool{sleeve.cooldown_bars}",
                sleeve.filter_label,
                f"{sleeve.source_leverage}x",
                sleeve.allocation_fraction,
            ],
            "debrepair",
        )
    elif sleeve.family == "relative_strength_chandelier_breakout":
        expected = _expected_model_id(
            [
                "booster",
                "rs_chandelier",
                sleeve.timeframe,
                sleeve.symbol,
                f"lb{sleeve.lookback}",
                f"atr{sleeve.atr_mult}",
                f"rel{sleeve.rel_threshold}",
                f"trail{sleeve.trail_atr_mult}",
                f"hold{sleeve.min_hold_bars}",
                f"{sleeve.source_leverage}x",
                sleeve.allocation_fraction,
            ],
            "a30fb",
        )
    elif sleeve.family == "relative_residual_reclaim":
        expected = _expected_model_id(
            [
                "asset_diverse",
                "residual_reclaim",
                sleeve.timeframe,
                sleeve.symbol,
                sleeve.base_symbol,
                f"lb{sleeve.lookback}",
                f"z{sleeve.z_entry}",
                f"hold{sleeve.min_hold_bars}",
                f"{sleeve.source_leverage}x",
                sleeve.allocation_fraction,
            ],
            "a30fb",
        )
    else:
        expected = _expected_model_id(
            [
                "voladj_trend",
                sleeve.timeframe,
                sleeve.symbol,
                f"lb{sleeve.lookback}",
                f"z{sleeve.entry_threshold}",
                f"hold{sleeve.min_hold_bars}",
                f"cool{sleeve.cooldown_bars}",
                sleeve.filter_label,
                f"{sleeve.source_leverage}x",
                sleeve.allocation_fraction,
            ],
            "a30fb",
        )
    if expected != sleeve.model_id:
        raise ValueError(f"source model id parity failed: {sleeve.model_id} != {expected}")


def load_alpha_zoo_optuna_hybrid_live_config(
    *,
    optuna_hybrid_artifact_path: str | Path = DEFAULT_OPTUNA_HYBRID_ARTIFACT,
    integer_portfolio_artifact_path: str | Path = DEFAULT_INTEGER_PORTFOLIO_ARTIFACT,
    selected_profile_id: str = DEFAULT_SELECTED_PROFILE_ID,
) -> AlphaZooOptunaHybridLiveConfig:
    optuna_path = _resolve_path(optuna_hybrid_artifact_path)
    integer_path = _resolve_path(integer_portfolio_artifact_path)
    optuna_payload = _read_json(optuna_path)
    integer_payload = _read_json(integer_path)

    _validate_paper_only_governance(optuna_payload, label="optuna hybrid artifact")
    _validate_paper_only_governance(integer_payload, label="integer portfolio artifact")
    _validate_locked_oos_policy(optuna_payload)
    _validate_integer_artifact(integer_payload)

    selected_profile = _extract_selected_profile(optuna_payload, selected_profile_id)
    for key in ("ready_for_real", "real_money_execution"):
        if selected_profile.get(key) is not False:
            raise ValueError(f"selected Optuna profile must keep {key}=false")
    if selected_profile.get("paper_testnet_candidate") is not True:
        raise ValueError("selected Optuna profile must be paper/testnet only")

    profile_rows = _profile_rows_by_id(integer_payload)
    selected_source_profile_ids = tuple(
        str(item) for item in dict(optuna_payload.get("optuna_hybrid_policy") or {}).get("source_profile_ids", [])
    ) or PROFILE_IDS
    missing = [profile_id for profile_id in selected_source_profile_ids if profile_id not in profile_rows]
    if missing:
        raise ValueError(f"missing source profile rows: {missing}")

    source_profiles: list[SourceProfile] = []
    for profile_id in selected_source_profile_ids:
        row = profile_rows[profile_id]
        raw_map = dict(row.get("leverage_map") or {})
        leverage_map: dict[str, int] = {}
        for symbol, leverage in raw_map.items():
            value = float(leverage)
            if value <= 0.0 or not value.is_integer():
                raise ValueError(f"source leverage must be positive integer: {profile_id}:{symbol}")
            leverage_map[_compact_symbol(symbol)] = int(value)
        models = tuple(str(item) for item in list(row.get("selected_model_ids") or []))
        if not models:
            raise ValueError(f"source profile has no selected models: {profile_id}")
        source_profiles.append(
            SourceProfile(profile_id=profile_id, selected_model_ids=models, leverage_map=leverage_map)
        )

    corr_rows = _load_corr_rows(integer_payload)
    model_ids = tuple(dict.fromkeys(model for profile in source_profiles for model in profile.selected_model_ids))
    source_sleeves: list[SourceSleeve] = []
    for model_id in model_ids:
        row = corr_rows.get(model_id, {"model_id": model_id})
        sleeve = _source_sleeve_from_row(model_id, row)
        _validate_source_model_id(sleeve)
        source_sleeves.append(sleeve)
    sleeve_by_id = {sleeve.model_id: sleeve for sleeve in source_sleeves}
    for profile in source_profiles:
        for model_id in profile.selected_model_ids:
            sleeve = sleeve_by_id[model_id]
            if sleeve.symbol not in profile.leverage_map:
                raise ValueError(
                    f"missing integer leverage for {profile.profile_id}:{sleeve.symbol}:{model_id}"
                )

    weights = {
        str(key): float(value)
        for key, value in dict(selected_profile.get("final_weights") or {}).items()
    }
    if not weights:
        weights = {str(key): float(value) for key, value in dict(selected_profile.get("weights") or {}).items()}
    average_weights = {
        str(key): float(value)
        for key, value in dict(selected_profile.get("average_weights_train_validation") or {}).items()
    }
    if not average_weights:
        average_weights = {str(key): float(value) for key, value in dict(selected_profile.get("weights") or {}).items()}
    for profile_id in selected_source_profile_ids:
        if profile_id not in weights:
            raise ValueError(f"missing frozen final profile weight: {profile_id}")

    watch_symbols = tuple(dict.fromkeys(_live_symbol(symbol) for symbol in WATCH_SYMBOLS))
    governance = {
        "paper_testnet_only": True,
        "ready_for_paper": True,
        "ready_for_real": False,
        "real_money_execution": False,
        "real_execution_allowed": False,
        "research_primary_round_trip_cost_bps": ROUND_TRIP_COST_BPS,
        "return_per_turnover_threshold_bps": RETURN_PER_TURNOVER_THRESHOLD_BPS,
        "locked_oos_role": "gate_report_only",
        "replay_live_notional_parity": bool(integer_payload.get("replay_live_notional_parity")),
    }
    return AlphaZooOptunaHybridLiveConfig(
        selected_profile_id=selected_profile_id,
        optuna_artifact_path=optuna_path,
        integer_artifact_path=integer_path,
        selected_profile=selected_profile,
        final_profile_weights=weights,
        average_profile_weights=average_weights,
        best_params=dict(selected_profile.get("best_params") or {}),
        learned_params=dict(selected_profile.get("learned_params") or {}),
        source_profiles=tuple(source_profiles),
        source_sleeves=tuple(source_sleeves),
        watch_symbols=watch_symbols,
        governance=governance,
    )


def _bars_to_frame(bars: list[Any]) -> pd.DataFrame:
    rows: list[tuple[Any, float, float, float, float, float]] = []
    for bar in bars:
        if isinstance(bar, dict):
            rows.append(
                (
                    bar.get("time") or bar.get("datetime"),
                    float(bar.get("open", 0.0)),
                    float(bar.get("high", 0.0)),
                    float(bar.get("low", 0.0)),
                    float(bar.get("close", 0.0)),
                    float(bar.get("volume", 0.0)),
                )
            )
        elif isinstance(bar, (tuple, list)) and len(bar) >= 6:
            rows.append((bar[0], float(bar[1]), float(bar[2]), float(bar[3]), float(bar[4]), float(bar[5])))
    return pd.DataFrame(rows, columns=["datetime", "open", "high", "low", "close", "volume"])


def completed_bars_only(
    aggregator: Any,
    symbol: str,
    timeframe: str,
    lookback_bars: int,
) -> list[Any]:
    """Return completed bars and drop the active working bar exposed by the aggregator."""
    if aggregator is None:
        return []
    for alias in _symbol_aliases(symbol):
        try:
            bars = list(aggregator.get_bars(alias, timeframe, n=max(2, int(lookback_bars) + 1)))
        except TypeError:
            try:
                bars = list(aggregator.get_bars(alias, timeframe, max(2, int(lookback_bars) + 1)))
            except (AttributeError, KeyError, TypeError, ValueError):
                bars = []
        except (AttributeError, KeyError, ValueError):
            bars = []
        if len(bars) >= 2:
            return bars[:-1]
    return []


def _time_key(value: Any) -> str:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return str(value)


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, lookback: int) -> pd.Series:
    prev_close = close.shift(1)
    true_range = pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return true_range.rolling(lookback).mean()


def _adx_proxy(high: pd.Series, low: pd.Series, close: pd.Series, lookback: int) -> pd.Series:
    prev_high = high.shift(1)
    prev_low = low.shift(1)
    prev_close = close.shift(1)
    up_move = high - prev_high
    down_move = prev_low - low
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0.0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0.0), 0.0)
    true_range = pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    tr_sum = true_range.rolling(lookback).sum().replace(0.0, np.nan)
    plus_di = 100.0 * plus_dm.rolling(lookback).sum() / tr_sum
    minus_di = 100.0 * minus_dm.rolling(lookback).sum() / tr_sum
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    return dx.rolling(lookback).mean()


def _volatility_mask(close: pd.Series, lookback: int, quantile_max: float | None) -> pd.Series:
    if quantile_max is None:
        return pd.Series(True, index=close.index)
    realized = close.pct_change().rolling(max(4, lookback // 2)).std(ddof=1)
    rolling_threshold = realized.rolling(max(24, lookback * 4)).quantile(quantile_max)
    return (realized <= rolling_threshold).fillna(False)


def debounced_state_signal(
    long_entry: pd.Series,
    long_exit: pd.Series,
    short_entry: pd.Series | None = None,
    short_exit: pd.Series | None = None,
    *,
    side: str,
    min_hold_bars: int,
    cooldown_bars: int,
) -> np.ndarray:
    long_entry = long_entry.fillna(False).astype(bool)
    long_exit = long_exit.fillna(False).astype(bool)
    short_entry = pd.Series(False, index=long_entry.index) if short_entry is None else short_entry.fillna(False).astype(bool)
    short_exit = pd.Series(False, index=long_entry.index) if short_exit is None else short_exit.fillna(False).astype(bool)
    out = np.zeros(len(long_entry), dtype=float)
    state = 0.0
    bars_held = 10**9
    cooldown_remaining = 0
    for idx in range(len(long_entry)):
        can_exit = bars_held >= min_hold_bars
        exited = False
        long_exit_now = state > 0 and bool(long_exit.iloc[idx])
        short_exit_now = state < 0 and bool(short_exit.iloc[idx])
        if can_exit and (long_exit_now or short_exit_now):
            state = 0.0
            bars_held = 0
            cooldown_remaining = cooldown_bars
            exited = True
        if state == 0.0:
            if cooldown_remaining > 0:
                cooldown_remaining -= 1
            elif not exited:
                if side in {"long_only", "long_short"} and bool(long_entry.iloc[idx]):
                    state = 1.0
                    bars_held = 0
                elif side in {"short_only", "long_short"} and bool(short_entry.iloc[idx]):
                    state = -1.0
                    bars_held = 0
        out[idx] = state
        if state != 0.0:
            bars_held += 1
    return out


def trailing_state_signal(
    close: pd.Series,
    long_entry: pd.Series,
    short_entry: pd.Series,
    long_exit: pd.Series,
    short_exit: pd.Series,
    atr: pd.Series,
    *,
    side: str,
    min_hold_bars: int,
    cooldown_bars: int,
    trail_atr_mult: float,
) -> np.ndarray:
    close_values = close.astype(float).to_numpy()
    atr_values = atr.astype(float).to_numpy()
    long_entry_values = long_entry.fillna(False).astype(bool).to_numpy()
    short_entry_values = short_entry.fillna(False).astype(bool).to_numpy()
    long_exit_values = long_exit.fillna(False).astype(bool).to_numpy()
    short_exit_values = short_exit.fillna(False).astype(bool).to_numpy()
    signal = np.zeros(len(close_values), dtype=float)
    state = 0.0
    stop = np.nan
    bars_held = 10**9
    cooldown = 0
    for idx, price in enumerate(close_values):
        atr_value = atr_values[idx]
        if not np.isfinite(price) or not np.isfinite(atr_value) or atr_value <= 0.0:
            signal[idx] = state
            if state != 0.0:
                bars_held += 1
            continue
        can_exit = bars_held >= min_hold_bars
        exited = False
        if state > 0.0:
            next_stop = price - trail_atr_mult * atr_value
            stop = next_stop if not np.isfinite(stop) else max(stop, next_stop)
            if can_exit and (long_exit_values[idx] or price < stop):
                state = 0.0
                stop = np.nan
                bars_held = 0
                cooldown = cooldown_bars
                exited = True
        elif state < 0.0:
            next_stop = price + trail_atr_mult * atr_value
            stop = next_stop if not np.isfinite(stop) else min(stop, next_stop)
            if can_exit and (short_exit_values[idx] or price > stop):
                state = 0.0
                stop = np.nan
                bars_held = 0
                cooldown = cooldown_bars
                exited = True
        if state == 0.0:
            if cooldown > 0:
                cooldown -= 1
            elif not exited:
                if side in {"long_only", "long_short"} and long_entry_values[idx]:
                    state = 1.0
                    stop = price - trail_atr_mult * atr_value
                    bars_held = 0
                elif side in {"short_only", "long_short"} and short_entry_values[idx]:
                    state = -1.0
                    stop = price + trail_atr_mult * atr_value
                    bars_held = 0
        signal[idx] = state
        if state != 0.0:
            bars_held += 1
    return signal


def _rolling_zscore(series: pd.Series, lookback: int) -> pd.Series:
    mean = series.rolling(lookback).mean()
    std = series.rolling(lookback).std(ddof=1).replace(0.0, np.nan)
    return (series - mean) / std


def _align_to_frame(series: pd.Series, datetimes: pd.Series) -> pd.Series:
    index = pd.DatetimeIndex(pd.to_datetime(datetimes))
    source = series.copy()
    source.index = pd.DatetimeIndex(pd.to_datetime(source.index))
    if pd.api.types.is_bool_dtype(source):
        aligned_bool = source.reindex(index, fill_value=False).astype(bool)
        return pd.Series(aligned_bool.to_numpy(), index=datetimes.index)
    aligned_numeric = pd.to_numeric(source.reindex(index).ffill(), errors="coerce")
    return pd.Series(aligned_numeric.to_numpy(dtype=float), index=datetimes.index)


def _panel_state(panel: pd.DataFrame, lookback: int) -> dict[str, pd.Series | pd.DataFrame]:
    returns = panel.pct_change().mean(axis=1).fillna(0.0)
    market_index = (1.0 + returns).cumprod()
    momentum = panel / panel.shift(lookback) - 1.0
    ranks = momentum.rank(axis=1, ascending=False, method="first")
    reverse_ranks = momentum.rank(axis=1, ascending=True, method="first")
    return {
        "market_momentum": market_index / market_index.shift(lookback) - 1.0,
        "breadth": (momentum > 0.0).sum(axis=1) / float(len(panel.columns)),
        "dispersion": momentum.std(axis=1, ddof=1),
        "momentum": momentum,
        "ranks": ranks,
        "reverse_ranks": reverse_ranks,
    }


def _merge_close(frame: pd.DataFrame, other: pd.DataFrame, column: str) -> pd.DataFrame:
    close_frame = other[["datetime", "close"]].rename(columns={"close": column})
    return frame.merge(close_frame, on="datetime", how="left").ffill()


def _frame_for(aggregator: Any, symbol: str, timeframe: str, lookback: int) -> pd.DataFrame:
    return _bars_to_frame(completed_bars_only(aggregator, symbol, timeframe, lookback))


def _intrabar_risk_frame_for(
    aggregator: Any,
    symbol: str,
    sleeve_timeframe: str,
    lookback: int,
) -> tuple[pd.DataFrame, str]:
    for timeframe in (*INTRABAR_RISK_TIMEFRAMES, sleeve_timeframe):
        frame = _frame_for(aggregator, symbol, timeframe, lookback)
        if len(frame) >= max(INTRABAR_ATR_LOOKBACK + 2, 4):
            return frame, timeframe
    return pd.DataFrame(), sleeve_timeframe


def _latest_atr_pct(frame: pd.DataFrame) -> float | None:
    if frame.empty or len(frame) < INTRABAR_ATR_LOOKBACK + 2:
        return None
    high = frame["high"].astype(float)
    low = frame["low"].astype(float)
    close = frame["close"].astype(float)
    atr = _atr(high, low, close, INTRABAR_ATR_LOOKBACK)
    latest_atr = float(atr.iloc[-1]) if pd.notna(atr.iloc[-1]) else 0.0
    latest_close = float(close.iloc[-1]) if pd.notna(close.iloc[-1]) else 0.0
    if latest_atr <= 0.0 or latest_close <= 0.0:
        return None
    return float(latest_atr / latest_close)


def _clamp_stop_distance_pct(atr_pct: float | None) -> float:
    floor = (RETURN_PER_TURNOVER_THRESHOLD_BPS / 10_000.0) * INTRABAR_MIN_STOP_COST_MULT
    ceiling = (RETURN_PER_TURNOVER_THRESHOLD_BPS / 10_000.0) * INTRABAR_MAX_STOP_COST_MULT
    candidate = floor if atr_pct is None else max(floor, float(atr_pct) * INTRABAR_STOP_ATR_MULT)
    return float(min(max(candidate, floor), ceiling))


def _evaluate_debounced(
    sleeve: SourceSleeve,
    frame: pd.DataFrame,
    btc_frame: pd.DataFrame,
) -> SleeveDecision | None:
    merged = _merge_close(frame, btc_frame, "btc_close")
    close = merged["close"].astype(float)
    high = merged["high"].astype(float)
    low = merged["low"].astype(float)
    btc_close = merged["btc_close"].astype(float)
    btc_regime_fast = btc_close / btc_close.shift(12) - 1.0
    lookback = sleeve.lookback
    momentum = close / close.shift(lookback) - 1.0
    realized = close.pct_change().rolling(max(4, lookback // 2)).std(ddof=1)
    trend_strength = momentum.abs() / (realized * np.sqrt(float(lookback))).replace(0.0, np.nan)
    adx = _adx_proxy(high, low, close, max(6, lookback))
    if sleeve.filter_label == "none":
        common_filter = pd.Series(True, index=frame.index)
    elif sleeve.filter_label.startswith("low_vol"):
        quantile = 0.65 if "q65" in sleeve.filter_label else 0.55
        common_filter = _volatility_mask(close, lookback, quantile)
    elif sleeve.filter_label.startswith("adx20"):
        common_filter = adx >= 20.0
    elif sleeve.filter_label.startswith("trend_strength2"):
        common_filter = trend_strength >= 2.0
    else:
        common_filter = adx >= 15.0
    long_entry = (momentum > sleeve.entry_threshold) & (btc_regime_fast > -0.02) & common_filter
    short_entry = (momentum < -sleeve.entry_threshold) & (btc_regime_fast < 0.02) & common_filter
    long_exit = (momentum < sleeve.exit_threshold) | (~common_filter)
    short_exit = (momentum > -sleeve.exit_threshold) | (~common_filter)
    signal = debounced_state_signal(
        long_entry,
        long_exit,
        short_entry,
        short_exit,
        side=sleeve.side,
        min_hold_bars=sleeve.min_hold_bars,
        cooldown_bars=sleeve.cooldown_bars,
    )
    if len(signal) == 0:
        return None
    latest = merged.iloc[-1]
    return SleeveDecision(
        signal=int(signal[-1]),
        completed_key=_time_key(latest["datetime"]),
        event_time=latest["datetime"],
        price=float(latest["close"]),
        diagnostics={
            "momentum": float(momentum.iloc[-1]) if pd.notna(momentum.iloc[-1]) else None,
            "btc_regime_fast": float(btc_regime_fast.iloc[-1]) if pd.notna(btc_regime_fast.iloc[-1]) else None,
        },
    )


def _evaluate_booster(
    sleeve: SourceSleeve,
    frame: pd.DataFrame,
    btc_frame: pd.DataFrame,
) -> SleeveDecision | None:
    merged = _merge_close(frame, btc_frame, "btc_close")
    close = merged["close"].astype(float)
    high = merged["high"].astype(float)
    low = merged["low"].astype(float)
    btc_close = merged["btc_close"].astype(float)
    hours = _timeframe_hours(sleeve.timeframe)
    btc_lookback = max(2, int(24 / hours))
    rel = close / btc_close.replace(0.0, np.nan)
    rel_momentum = rel / rel.shift(max(2, int(12 / hours))) - 1.0
    btc_momentum = btc_close / btc_close.shift(btc_lookback) - 1.0
    lookback = sleeve.lookback
    atr = _atr(high, low, close, max(6, lookback))
    roll_high = high.shift(1).rolling(lookback).max()
    roll_low = low.shift(1).rolling(lookback).min()
    mid = (roll_high + roll_low) / 2.0
    adx = _adx_proxy(high, low, close, max(6, lookback // 2))
    common_long = (rel_momentum > sleeve.rel_threshold) & (btc_momentum > -0.015) & (adx >= 15.0)
    common_short = (rel_momentum < -sleeve.rel_threshold) & (btc_momentum < 0.015) & (adx >= 15.0)
    long_entry = (close > roll_high + sleeve.atr_mult * atr) & common_long
    short_entry = (close < roll_low - sleeve.atr_mult * atr) & common_short
    long_exit = (close < mid) | (rel_momentum < -0.005)
    short_exit = (close > mid) | (rel_momentum > 0.005)
    signal = trailing_state_signal(
        close,
        long_entry,
        short_entry,
        long_exit,
        short_exit,
        atr,
        side="long_short",
        min_hold_bars=sleeve.min_hold_bars,
        cooldown_bars=sleeve.cooldown_bars,
        trail_atr_mult=sleeve.trail_atr_mult,
    )
    if len(signal) == 0:
        return None
    latest = merged.iloc[-1]
    return SleeveDecision(
        signal=int(signal[-1]),
        completed_key=_time_key(latest["datetime"]),
        event_time=latest["datetime"],
        price=float(latest["close"]),
        diagnostics={
            "rel_momentum": float(rel_momentum.iloc[-1]) if pd.notna(rel_momentum.iloc[-1]) else None,
            "btc_momentum": float(btc_momentum.iloc[-1]) if pd.notna(btc_momentum.iloc[-1]) else None,
            "adx_proxy": float(adx.iloc[-1]) if pd.notna(adx.iloc[-1]) else None,
        },
    )


def _build_panel(aggregator: Any, timeframe: str, lookback: int) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for symbol in WATCH_SYMBOLS:
        frame = _frame_for(aggregator, symbol, timeframe, lookback)
        if frame.empty:
            return pd.DataFrame()
        frames.append(frame[["datetime", "close"]].assign(symbol=symbol))
    panel = pd.concat(frames).pivot(index="datetime", columns="symbol", values="close")
    return panel.sort_index().dropna(how="any")


def _evaluate_residual(
    sleeve: SourceSleeve,
    frame: pd.DataFrame,
    base_frame: pd.DataFrame,
    panel: pd.DataFrame,
) -> SleeveDecision | None:
    if panel.empty:
        return None
    merged = _merge_close(frame, base_frame, "base_close")
    close = merged["close"].astype(float)
    base_close = merged["base_close"].astype(float)
    ratio = np.log(close / base_close.replace(0.0, np.nan))
    market_mom = _align_to_frame(_panel_state(panel, 24)["market_momentum"], merged["datetime"])
    lookback = sleeve.lookback
    z = _rolling_zscore(ratio, lookback)
    target_mom = close / close.shift(max(4, lookback // 3)) - 1.0
    reclaim_long = (z.shift(1) < -sleeve.z_entry) & (z >= -sleeve.z_entry)
    reclaim_short = (z.shift(1) > sleeve.z_entry) & (z <= sleeve.z_entry)
    long_entry = reclaim_long & (target_mom > -0.005) & (market_mom > -0.02)
    short_entry = reclaim_short & (target_mom < 0.005) & (market_mom < 0.02)
    long_exit = (z > -0.05) | (target_mom < -0.02)
    short_exit = (z < 0.05) | (target_mom > 0.02)
    signal = debounced_state_signal(
        long_entry,
        long_exit,
        short_entry,
        short_exit,
        side="long_short",
        min_hold_bars=sleeve.min_hold_bars,
        cooldown_bars=sleeve.cooldown_bars,
    )
    if len(signal) == 0:
        return None
    latest = merged.iloc[-1]
    return SleeveDecision(
        signal=int(signal[-1]),
        completed_key=_time_key(latest["datetime"]),
        event_time=latest["datetime"],
        price=float(latest["close"]),
        diagnostics={
            "residual_z": float(z.iloc[-1]) if pd.notna(z.iloc[-1]) else None,
            "target_momentum": float(target_mom.iloc[-1]) if pd.notna(target_mom.iloc[-1]) else None,
            "market_momentum": float(market_mom.iloc[-1]) if pd.notna(market_mom.iloc[-1]) else None,
        },
    )


def _evaluate_voladj(
    sleeve: SourceSleeve,
    frame: pd.DataFrame,
    btc_frame: pd.DataFrame,
) -> SleeveDecision | None:
    merged = _merge_close(frame, btc_frame, "btc_close")
    close = merged["close"].astype(float)
    high = merged["high"].astype(float)
    low = merged["low"].astype(float)
    btc_close = merged["btc_close"].astype(float)
    hours = _timeframe_hours(sleeve.timeframe)
    btc_momentum = btc_close / btc_close.shift(max(2, int(12 / hours))) - 1.0
    lookback = sleeve.lookback
    momentum = close / close.shift(lookback) - 1.0
    realized = close.pct_change().rolling(max(4, lookback // 2)).std(ddof=1)
    vol_adjusted = momentum / (realized * np.sqrt(float(lookback))).replace(0.0, np.nan)
    adx = _adx_proxy(high, low, close, max(6, lookback))
    common = adx >= 15.0 if sleeve.filter_label == "adx15" else pd.Series(True, index=frame.index)
    long_entry = (vol_adjusted > sleeve.entry_threshold) & (btc_momentum > -0.02) & common
    short_entry = (vol_adjusted < -sleeve.entry_threshold) & (btc_momentum < 0.02) & common
    long_exit = (vol_adjusted < sleeve.exit_threshold) | (~common)
    short_exit = (vol_adjusted > -sleeve.exit_threshold) | (~common)
    signal = debounced_state_signal(
        long_entry,
        long_exit,
        short_entry,
        short_exit,
        side="long_short",
        min_hold_bars=sleeve.min_hold_bars,
        cooldown_bars=sleeve.cooldown_bars,
    )
    if len(signal) == 0:
        return None
    latest = merged.iloc[-1]
    return SleeveDecision(
        signal=int(signal[-1]),
        completed_key=_time_key(latest["datetime"]),
        event_time=latest["datetime"],
        price=float(latest["close"]),
        diagnostics={
            "vol_adjusted_momentum": float(vol_adjusted.iloc[-1])
            if pd.notna(vol_adjusted.iloc[-1])
            else None,
            "btc_momentum": float(btc_momentum.iloc[-1]) if pd.notna(btc_momentum.iloc[-1]) else None,
            "adx_proxy": float(adx.iloc[-1]) if pd.notna(adx.iloc[-1]) else None,
        },
    )


def _timeframe_hours(timeframe: str) -> float:
    token = normalize_timeframe_token(timeframe)
    if token.endswith("m"):
        return float(token[:-1]) / 60.0
    if token.endswith("h"):
        return float(token[:-1])
    if token.endswith("d"):
        return 24.0 * float(token[:-1])
    return 1.0


class AlphaZooOptunaHybridLiveStrategy(Strategy):
    """Live-compatible paper/testnet adapter for the frozen Alpha Zoo hybrid."""

    strategy_id = "alpha_zoo_optuna_hybrid_live"
    decision_cadence_seconds = 3600
    preferred_contract = "market_window"
    uses_timeframe_aggregator = True
    required_timeframes = ("1m", "5m", "1h", "2h", "4h")
    required_lookbacks = {"1m": 96, "5m": 96, "1h": 96, "2h": 128, "4h": 64}
    required_inputs = ("OHLCV",)
    strategy_validity = {
        "pass": True,
        "primary_signal_type": "artifact_frozen_state_rules",
        "causal_state_only": True,
        "lookahead_safe": True,
        "locked_oos_role": "gate_report_only",
        "ready_for_real": False,
        "real_money_execution": False,
        "real_execution_allowed": False,
        "rejection_reasons": ["paper_testnet_only_requires_live_fill_telemetry_before_real"],
    }

    def __init__(
        self,
        bars: Any,
        events: Any,
        *,
        optuna_hybrid_artifact_path: str | Path = DEFAULT_OPTUNA_HYBRID_ARTIFACT,
        integer_portfolio_artifact_path: str | Path = DEFAULT_INTEGER_PORTFOLIO_ARTIFACT,
        selected_profile_id: str = DEFAULT_SELECTED_PROFILE_ID,
        paper_testnet_only: bool = True,
        allow_real_money: bool = False,
        min_completed_bars: int = 0,
        **_: Any,
    ) -> None:
        if not paper_testnet_only or allow_real_money:
            raise ValueError("AlphaZooOptunaHybridLiveStrategy is paper/testnet-only")
        self.bars = bars
        self.events = events
        self.config = load_alpha_zoo_optuna_hybrid_live_config(
            optuna_hybrid_artifact_path=optuna_hybrid_artifact_path,
            integer_portfolio_artifact_path=integer_portfolio_artifact_path,
            selected_profile_id=selected_profile_id,
        )
        self.allocator = AlphaZooV35HybridAllocator(self.config)
        self.symbol_list = list(getattr(bars, "symbol_list", []) or list(self.config.watch_symbols))
        self.paper_testnet_only = True
        self.ready_for_real = False
        self.real_money_execution = False
        self.real_execution_allowed = False
        self.min_completed_bars = max(0, int(min_completed_bars))
        self._last_completed_key_by_sleeve = {sleeve.model_id: "" for sleeve in self.config.source_sleeves}
        self._last_signal_by_sleeve = {sleeve.model_id: 0 for sleeve in self.config.source_sleeves}
        self._intrabar_guards: dict[str, dict[str, Any]] = {}

    def get_state(self) -> dict[str, Any]:
        return {
            "last_completed_key_by_sleeve": dict(self._last_completed_key_by_sleeve),
            "last_signal_by_sleeve": dict(self._last_signal_by_sleeve),
            "intrabar_guards": dict(self._intrabar_guards),
            "selected_profile_id": self.config.selected_profile_id,
            "paper_testnet_only": True,
            "ready_for_real": False,
            "real_money_execution": False,
            "real_execution_allowed": False,
        }

    def set_state(self, state: dict[str, Any]) -> None:
        if not isinstance(state, dict):
            return
        for key, value in dict(state.get("last_completed_key_by_sleeve") or {}).items():
            if key in self._last_completed_key_by_sleeve:
                self._last_completed_key_by_sleeve[key] = str(value)
        for key, value in dict(state.get("last_signal_by_sleeve") or {}).items():
            if key in self._last_signal_by_sleeve:
                try:
                    parsed = int(value)
                except (TypeError, ValueError):
                    continue
                if parsed in {-1, 0, 1}:
                    self._last_signal_by_sleeve[key] = parsed
        guards = dict(state.get("intrabar_guards") or {})
        self._intrabar_guards = {
            str(key): dict(value)
            for key, value in guards.items()
            if isinstance(value, dict) and key in self._last_signal_by_sleeve
        }

    def calculate_signals(self, event: Any) -> None:
        if getattr(event, "type", None) != "MARKET":
            return
        self._check_intrabar_guards(event)

    def calculate_signals_window(self, event: Any, aggregator: Any) -> None:
        if aggregator is None:
            return
        for sleeve in self.config.source_sleeves:
            decision = self._evaluate_sleeve(aggregator, sleeve)
            if decision is None:
                continue
            if decision.completed_key == self._last_completed_key_by_sleeve.get(sleeve.model_id):
                continue
            self._last_completed_key_by_sleeve[sleeve.model_id] = decision.completed_key
            previous_signal = int(self._last_signal_by_sleeve.get(sleeve.model_id, 0))
            if int(decision.signal) == previous_signal:
                continue
            self._last_signal_by_sleeve[sleeve.model_id] = int(decision.signal)
            self._emit_transition(sleeve, decision, previous_signal, aggregator)
        _ = event

    def _evaluate_sleeve(self, aggregator: Any, sleeve: SourceSleeve) -> SleeveDecision | None:
        lookback = max(96, sleeve.lookback + sleeve.min_hold_bars + 8, self.min_completed_bars)
        frame = _frame_for(aggregator, sleeve.symbol, sleeve.timeframe, lookback)
        if len(frame) < max(8, sleeve.lookback + 2):
            return None
        if sleeve.family == "relative_residual_reclaim":
            base = _frame_for(aggregator, sleeve.base_symbol, sleeve.timeframe, lookback)
            panel = _build_panel(aggregator, sleeve.timeframe, max(96, lookback))
            if len(base) < max(8, sleeve.lookback + 2):
                return None
            return _evaluate_residual(sleeve, frame, base, panel)
        btc = _frame_for(aggregator, "BTCUSDT", sleeve.timeframe, lookback)
        if len(btc) < max(8, sleeve.lookback + 2):
            return None
        if sleeve.family == "debounced_momentum_hysteresis_efficiency_repair":
            return _evaluate_debounced(sleeve, frame, btc)
        if sleeve.family == "relative_strength_chandelier_breakout":
            return _evaluate_booster(sleeve, frame, btc)
        if sleeve.family == "volatility_adjusted_trend_persistence":
            return _evaluate_voladj(sleeve, frame, btc)
        return None

    def target_notional_fraction_for_sleeve(self, sleeve: SourceSleeve) -> float:
        profile_weights = self.allocator.profile_weights_for_live()
        total = 0.0
        for profile in self.config.source_profiles:
            if sleeve.model_id not in profile.selected_model_ids:
                continue
            leverage = int(profile.leverage_map.get(sleeve.symbol, 0))
            if leverage <= 0:
                continue
            total += float(profile_weights.get(profile.profile_id, 0.0)) * float(leverage)
        return float(sleeve.allocation_fraction) * total

    def max_symbol_notional_fraction(self, symbol: str) -> float:
        """Return worst-case all-sleeve notional/equity for one live symbol."""
        compact = _compact_symbol(symbol)
        return float(
            sum(
                self.target_notional_fraction_for_sleeve(sleeve)
                for sleeve in self.config.source_sleeves
                if sleeve.symbol == compact
            )
        )

    def _profile_contributions(self, sleeve: SourceSleeve) -> list[dict[str, Any]]:
        weights = self.allocator.profile_weights_for_live()
        out: list[dict[str, Any]] = []
        for profile in self.config.source_profiles:
            if sleeve.model_id not in profile.selected_model_ids:
                continue
            leverage = int(profile.leverage_map.get(sleeve.symbol, 0))
            weight = float(weights.get(profile.profile_id, 0.0))
            out.append(
                {
                    "profile_id": profile.profile_id,
                    "profile_weight": weight,
                    "integer_leverage": leverage,
                    "weighted_integer_leverage": weight * float(leverage),
                }
            )
        return out

    def _build_intrabar_protection_plan(
        self,
        aggregator: Any,
        sleeve: SourceSleeve,
        decision: SleeveDecision,
        signal_type: str,
    ) -> IntrabarProtectionPlan:
        if signal_type not in {"LONG", "SHORT"}:
            return IntrabarProtectionPlan(
                enabled=False,
                source_timeframe=sleeve.timeframe,
                guard_mode="none_exit_signal",
                stop_loss=None,
                take_profit=None,
                trailing_percent=None,
                stop_distance_pct=0.0,
                atr_pct=None,
                notes=("exit signals clear any active intrabar guard",),
            )
        frame, source_timeframe = _intrabar_risk_frame_for(
            aggregator,
            sleeve.symbol,
            sleeve.timeframe,
            max(INTRABAR_ATR_LOOKBACK * 4, sleeve.lookback + 4),
        )
        atr_pct = _latest_atr_pct(frame)
        stop_pct = _clamp_stop_distance_pct(atr_pct)
        price = max(1e-12, float(decision.price))
        if signal_type == "LONG":
            stop_loss = price * (1.0 - stop_pct)
        else:
            stop_loss = price * (1.0 + stop_pct)
        trailing_percent = stop_pct if sleeve.trail_atr_mult > 0.0 else None
        notes = (
            "paper/testnet local intrabar guard emits component EXIT on stop breach",
            "real exchange-side protective orders still require explicit exchange order support",
            "queue priority is measured as telemetry/proxy, not exactly knowable from exchange APIs",
        )
        return IntrabarProtectionPlan(
            enabled=True,
            source_timeframe=source_timeframe,
            guard_mode="paper_local_or_simulated_component_exit",
            stop_loss=float(stop_loss),
            take_profit=None,
            trailing_percent=trailing_percent,
            stop_distance_pct=stop_pct,
            atr_pct=atr_pct,
            notes=notes,
        )

    def _activate_intrabar_guard(
        self,
        sleeve: SourceSleeve,
        decision: SleeveDecision,
        signal_type: str,
        plan: IntrabarProtectionPlan,
    ) -> None:
        if not plan.enabled or plan.stop_loss is None:
            self._intrabar_guards.pop(sleeve.model_id, None)
            return
        self._intrabar_guards[sleeve.model_id] = {
            "source_model_id": sleeve.model_id,
            "component_id": sleeve.model_id,
            "symbol": _live_symbol(sleeve.symbol),
            "compact_symbol": sleeve.symbol,
            "side": signal_type,
            "stop_loss": float(plan.stop_loss),
            "take_profit": None if plan.take_profit is None else float(plan.take_profit),
            "trailing_percent": None
            if plan.trailing_percent is None
            else float(plan.trailing_percent),
            "highest_price": float(decision.price),
            "lowest_price": float(decision.price),
            "source_timeframe": plan.source_timeframe,
            "activated_at": decision.completed_key,
            "entry_price": float(decision.price),
            "guard_mode": plan.guard_mode,
        }

    def _check_intrabar_guards(self, event: Any) -> None:
        symbol = _compact_symbol(getattr(event, "symbol", ""))
        if not symbol or not self._intrabar_guards:
            return
        high = float(getattr(event, "high", 0.0) or 0.0)
        low = float(getattr(event, "low", 0.0) or 0.0)
        close = float(getattr(event, "close", 0.0) or 0.0)
        if high <= 0.0 or low <= 0.0:
            return
        for model_id, guard in list(self._intrabar_guards.items()):
            if _compact_symbol(str(guard.get("symbol") or guard.get("compact_symbol") or "")) != symbol:
                continue
            side = str(guard.get("side") or "").upper()
            stop_loss = float(guard.get("stop_loss") or 0.0)
            trailing_percent = guard.get("trailing_percent")
            trigger_price = None
            reason = ""
            if side == "LONG":
                guard["highest_price"] = max(float(guard.get("highest_price") or 0.0), high)
                if trailing_percent is not None:
                    stop_loss = max(stop_loss, float(guard["highest_price"]) * (1.0 - float(trailing_percent)))
                    guard["stop_loss"] = stop_loss
                if low <= stop_loss:
                    trigger_price = stop_loss
                    reason = "intrabar_stop_loss_or_trailing_long"
            elif side == "SHORT":
                previous_low = float(guard.get("lowest_price") or close or low)
                guard["lowest_price"] = min(previous_low, low)
                if trailing_percent is not None:
                    stop_loss = min(stop_loss, float(guard["lowest_price"]) * (1.0 + float(trailing_percent)))
                    guard["stop_loss"] = stop_loss
                if high >= stop_loss:
                    trigger_price = stop_loss
                    reason = "intrabar_stop_loss_or_trailing_short"
            if trigger_price is None:
                continue
            self._intrabar_guards.pop(model_id, None)
            self._last_signal_by_sleeve[model_id] = 0
            client_hash = hashlib.sha1(
                f"{model_id}|{getattr(event, 'time', '')}|{reason}".encode()
            ).hexdigest()[:18]
            self.events.put(
                SignalEvent(
                    strategy_id=self.strategy_id,
                    symbol=str(guard.get("symbol")),
                    datetime=getattr(event, "time", None),
                    signal_type="EXIT",
                    strength=1.0,
                    price=float(trigger_price),
                    position_side=side,
                    client_order_id=f"azoh-risk-{client_hash}",
                    metadata={
                        "alpha_zoo_optuna_hybrid_live": True,
                        "paper_testnet_only": True,
                        "ready_for_real": False,
                        "real_money_execution": False,
                        "real_execution_allowed": False,
                        "component_id": model_id,
                        "source_model_id": model_id,
                        "intrabar_protection_enabled": True,
                        "intrabar_exit_reason": reason,
                        "intrabar_trigger_price": float(trigger_price),
                        "intrabar_event_high": high,
                        "intrabar_event_low": low,
                        "intrabar_event_close": close,
                        "target_allocation": 0.0,
                        "target_allocation_mode": "notional_fraction",
                        "sizing_mode": "notional_fraction",
                        "locked_oos_role": "gate_report_only",
                    },
                )
            )

    def _emit_transition(
        self,
        sleeve: SourceSleeve,
        decision: SleeveDecision,
        previous_signal: int,
        aggregator: Any,
    ) -> None:
        if int(decision.signal) > 0:
            signal_type = "LONG"
            position_side = "LONG"
        elif int(decision.signal) < 0:
            signal_type = "SHORT"
            position_side = "SHORT"
        else:
            signal_type = "EXIT"
            position_side = "LONG" if previous_signal > 0 else "SHORT" if previous_signal < 0 else None
        protection = self._build_intrabar_protection_plan(aggregator, sleeve, decision, signal_type)
        if signal_type == "EXIT":
            self._intrabar_guards.pop(sleeve.model_id, None)
        target_notional = self.target_notional_fraction_for_sleeve(sleeve)
        symbol_notional_cap = max(
            target_notional + 0.05,
            self.max_symbol_notional_fraction(sleeve.symbol) * 1.05,
        )
        metadata = {
            "alpha_zoo_optuna_hybrid_live": True,
            "paper_testnet_only": True,
            "ready_for_real": False,
            "real_money_execution": False,
            "real_execution_allowed": False,
            "source_model_id": sleeve.model_id,
            "source_family": sleeve.family,
            "source_symbol": sleeve.symbol,
            "source_timeframe": sleeve.timeframe,
            "source_side": sleeve.side,
            "component_id": sleeve.model_id,
            "source_allocation_fraction": float(sleeve.allocation_fraction),
            "source_research_leverage": float(sleeve.source_leverage),
            "target_notional_fraction": target_notional,
            "target_notional_formula": "allocation_fraction*sum(profile_weight*integer_leverage)",
            "target_allocation": target_notional,
            "target_allocation_mode": "notional_fraction",
            "sizing_mode": "notional_fraction",
            "max_order_value": 0.0,
            "max_order_notional_pct": max(target_notional + 0.05, target_notional * 1.05),
            "max_symbol_exposure_pct": symbol_notional_cap,
            "profile_contributions": self._profile_contributions(sleeve),
            "allocator": self.allocator.allocation_metadata(),
            "round_trip_cost_bps": ROUND_TRIP_COST_BPS,
            "return_per_turnover_threshold_bps": RETURN_PER_TURNOVER_THRESHOLD_BPS,
            "locked_oos_role": "gate_report_only",
            "replay_live_notional_parity": True,
            "completed_bar_key": decision.completed_key,
            "previous_source_signal": int(previous_signal),
            "current_source_signal": int(decision.signal),
            "diagnostics": decision.diagnostics,
            "intrabar_protection_enabled": protection.enabled,
            "intrabar_protection": {
                "guard_mode": protection.guard_mode,
                "source_timeframe": protection.source_timeframe,
                "stop_loss": protection.stop_loss,
                "take_profit": protection.take_profit,
                "trailing_percent": protection.trailing_percent,
                "stop_distance_pct": protection.stop_distance_pct,
                "atr_pct": protection.atr_pct,
                "notes": list(protection.notes),
            },
            "microstructure_telemetry_required": [
                "bbo_spread_bps_at_submit",
                "order_book_depth_or_liquidity_proxy",
                "submit_to_fill_ms",
                "realized_slippage_bps",
                "fee_bps",
                "funding_bps",
                "partial_fill_ratio",
                "cancel_timeout_reject_flags",
            ],
            "queue_priority_model": "proxy_only_exchange_exact_queue_position_unavailable",
        }
        if signal_type in {"LONG", "SHORT"}:
            self._activate_intrabar_guard(sleeve, decision, signal_type, protection)
        client_hash = hashlib.sha1(
            f"{sleeve.model_id}|{decision.completed_key}|{signal_type}".encode()
        ).hexdigest()[:18]
        self.events.put(
            SignalEvent(
                strategy_id=self.strategy_id,
                symbol=_live_symbol(sleeve.symbol),
                datetime=decision.event_time,
                signal_type=signal_type,
                strength=abs(target_notional),
                price=float(decision.price),
                stop_loss=protection.stop_loss,
                take_profit=protection.take_profit,
                position_side=position_side,
                client_order_id=f"azoh-{client_hash}",
                metadata=metadata,
                trailing_percent=protection.trailing_percent,
            )
        )


__all__ = [
    "DEFAULT_INTEGER_PORTFOLIO_ARTIFACT",
    "DEFAULT_OPTUNA_HYBRID_ARTIFACT",
    "DEFAULT_SELECTED_PROFILE_ID",
    "AlphaZooOptunaHybridLiveConfig",
    "AlphaZooOptunaHybridLiveStrategy",
    "AlphaZooV35HybridAllocator",
    "IntrabarProtectionPlan",
    "SourceProfile",
    "SourceSleeve",
    "completed_bars_only",
    "debounced_state_signal",
    "load_alpha_zoo_optuna_hybrid_live_config",
    "trailing_state_signal",
]
