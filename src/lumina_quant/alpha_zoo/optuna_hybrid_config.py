"""Frozen artifact loading for the Alpha Zoo Optuna hybrid live adapter."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lumina_quant.market_data import normalize_timeframe_token


REPO_ROOT = Path(__file__).resolve().parents[3]
ALPHA_V2_ROOT = (
    Path("var") / "reports" / "profit_moonshot_20260501" / "current_tail_20260508" / "alpha_v2"
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
        allocation_fraction=float(
            row.get("allocation_fraction") or _token_float(match.group("alloc"))
        ),
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
        allocation_fraction=float(
            row.get("allocation_fraction") or _token_float(match.group("alloc"))
        ),
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
        allocation_fraction=float(
            row.get("allocation_fraction") or _token_float(match.group("alloc"))
        ),
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
        allocation_fraction=float(
            row.get("allocation_fraction") or _token_float(match.group("alloc"))
        ),
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
    selected_source_profile_ids = (
        tuple(
            str(item)
            for item in dict(optuna_payload.get("optuna_hybrid_policy") or {}).get(
                "source_profile_ids", []
            )
        )
        or PROFILE_IDS
    )
    missing = [
        profile_id for profile_id in selected_source_profile_ids if profile_id not in profile_rows
    ]
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
            SourceProfile(
                profile_id=profile_id, selected_model_ids=models, leverage_map=leverage_map
            )
        )

    corr_rows = _load_corr_rows(integer_payload)
    model_ids = tuple(
        dict.fromkeys(model for profile in source_profiles for model in profile.selected_model_ids)
    )
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
        weights = {
            str(key): float(value)
            for key, value in dict(selected_profile.get("weights") or {}).items()
        }
    average_weights = {
        str(key): float(value)
        for key, value in dict(
            selected_profile.get("average_weights_train_validation") or {}
        ).items()
    }
    if not average_weights:
        average_weights = {
            str(key): float(value)
            for key, value in dict(selected_profile.get("weights") or {}).items()
        }
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


__all__ = [
    "ALPHA_V2_ROOT",
    "DEFAULT_INTEGER_PORTFOLIO_ARTIFACT",
    "DEFAULT_OPTUNA_HYBRID_ARTIFACT",
    "DEFAULT_SELECTED_PROFILE_ID",
    "INTRABAR_ATR_LOOKBACK",
    "INTRABAR_MAX_STOP_COST_MULT",
    "INTRABAR_MIN_STOP_COST_MULT",
    "INTRABAR_RISK_TIMEFRAMES",
    "INTRABAR_STOP_ATR_MULT",
    "PROFILE_IDS",
    "REPO_ROOT",
    "RETURN_PER_TURNOVER_THRESHOLD_BPS",
    "ROUND_TRIP_COST_BPS",
    "TRADED_SYMBOLS",
    "WATCH_SYMBOLS",
    "AlphaZooOptunaHybridLiveConfig",
    "AlphaZooV35HybridAllocator",
    "IntrabarProtectionPlan",
    "SleeveDecision",
    "SourceProfile",
    "SourceSleeve",
    "load_alpha_zoo_optuna_hybrid_live_config",
]
