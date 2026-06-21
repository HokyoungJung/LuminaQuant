from __future__ import annotations

import hashlib
import json
from collections import deque
from datetime import UTC, datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lumina_quant.core.events import SignalEvent
from lumina_quant.portfolio_split_contract import FOLLOWUP_ROOT
from lumina_quant.strategy import Strategy
from lumina_quant.strategies import resolve_strategy_class
from lumina_quant.utils.numeric import safe_float

GROUP_ROOT = FOLLOWUP_ROOT / "portfolio_incumbent_autoresearch_grouped"
REFRESH_ROOT = GROUP_ROOT / "current_switch_validation_current"
HYBRID_PATH = GROUP_ROOT / "portfolio_hybrid_online_current" / "hybrid_online_portfolio_latest.json"
LEGACY_NO_HIGHVOL_HYBRID_PATH = (
    GROUP_ROOT
    / "legacy_metric_live_materialization_20260426"
    / "legacy_metric_live_materialization_latest.json"
)
RETUNED_LIVE_PORTFOLIO_HYBRID_PATH = (
    GROUP_ROOT
    / "live_portfolio_hybrid_retune_20260426"
    / "live_portfolio_hybrid_retune_latest.json"
)
WAVE2_PAIR_PATH = (
    GROUP_ROOT
    / "legacy_metric_live_materialization_20260426"
    / "wave2_pair_live_candidate_latest.json"
)
REFRESHED_INCUMBENT_PATH = (
    REFRESH_ROOT / "refreshed_current_one_shot_incumbent_portfolio_latest.json"
)
REFRESHED_BLEND_PATH = REFRESH_ROOT / "refreshed_grouped_static_blend_latest.json"
REFRESHED_AUTORESEARCH_55_45_PATH = (
    REFRESH_ROOT / "refreshed_autoresearch_pair_55_45_portfolio_latest.json"
)
SOFT_THREE_WAY_ALLOCATOR_PATH = (
    REFRESH_ROOT
    / "refreshed_soft_three_way_allocator_current"
    / "soft_three_way_market_regime_allocator_latest.json"
)
THREE_WAY_ALLOCATOR_PATH = (
    REFRESH_ROOT
    / "refreshed_three_way_allocator_current"
    / "three_way_market_regime_allocator_latest.json"
)
PAIR_TACTICAL_PATH = REFRESH_ROOT / "refreshed_pair_fast_exit_candidate_latest.json"
PRODUCTION_GUARDED_PATH = (
    GROUP_ROOT / "portfolio_production_guarded_current" / "production_guarded_portfolio_latest.json"
)
STATE_VWAP_PAIR_PATH = (
    GROUP_ROOT
    / "portfolio_superiority_dense_pairs_current"
    / "state_vwap_pair_candidate_latest.json"
)
STRICT_AUTORESEARCH_1X_PATH = (
    GROUP_ROOT
    / "strict_blend_76_24_leverage_sweep_rerun_current"
    / "inc_1_auto_1"
    / "strict_autoresearch_portfolio_current"
    / "strict_autoresearch_portfolio_latest.json"
)
DERIVATIVES_FLOW_SQUEEZE_MANIFEST_PATH = (
    Path("var")
    / "reports"
    / "derivatives_flow_squeeze_20260502"
    / "derivatives_flow_squeeze_candidates.json"
)
PRECIOUS_METAL_PAIR_SCREEN_PATH = (
    Path("var")
    / "reports"
    / "profit_moonshot_20260501"
    / "current_tail_20260505"
    / "precious_metal_pair_aggressive"
    / "precious_pair_screen_top.json"
)
DEFAULT_ARTIFACT_PORTFOLIO_MANIFEST_PATH = (
    GROUP_ROOT / "artifact_portfolio_manifest_current" / "artifact_portfolio_manifest_latest.json"
)
MANIFEST_PORTFOLIO_MODE_PREFIX = "manifest:"
MANIFEST_PORTFOLIO_GROSS_CAP = 2.25
_MANIFEST_FORBIDDEN_OOS_KEYS = (
    "uses_current_fold_oos",
    "uses_locked_oos_for_selection",
    "uses_locked_oos_for_objective",
    "uses_locked_oos_for_pruning",
    "uses_locked_oos_for_parameter_fitting",
    "uses_locked_oos_for_threshold",
    "uses_locked_oos_for_tie_break",
    "uses_locked_oos_for_correlation",
    "uses_locked_oos_for_sizing",
)

_LIVE_PORTFOLIO_MODE_ALIASES = {
    "aggressive_realized_mode",
    "hybrid_guarded_mode",
    "legacy_no_highvol_hybrid_mode",
    "retuned_live_portfolio_hybrid_mode",
    "balanced_overlay_mode",
    "defensive_overlay_mode",
    "core_mode",
    "pair_tactical_mode",
    "production_guarded_state_vwap_pair_mode",
    "strict_autoresearch_practical_mode",
    "risk_off_mode",
    # Source sleeves / static blends that can be promoted to live by expanding
    # the same saved strategy rows the research artifacts used.
    "incumbent",
    "incumbent_only",
    "autoresearch_55_45",
    "blend_85_15",
    "static_blend_76_24",
    "production_guarded_portfolio",
    "strict_autoresearch_1x",
    "soft_three_way_regime",
    "three_way_regime",
    "balanced_overlay_80_20",
    "pair_fast_exit",
    "state_vwap_pair",
    "wave2_pair",
    "profit_reboot_adaptive_momentum_mode",
    "profit_reboot_adaptive_momentum_defensive_mode",
    "profit_reboot_adaptive_momentum_short_bias_mode",
    "profit_reboot_panic_rebound_mode",
    "profit_reboot_session_pair_carry_mode",
    "profit_reboot_compression_breakout_mode",
    "profit_moonshot_adaptive_momentum_mode",
    "profit_moonshot_adaptive_momentum_120_mode",
    "profit_moonshot_adaptive_momentum_130_mode",
    "profit_moonshot_adaptive_momentum_140_mode",
    "profit_moonshot_adaptive_momentum_boost_mode",
    "profit_moonshot_adaptive_momentum_governed_mode",
    "profit_moonshot_adaptive_momentum_vol_target_mode",
    "profit_moonshot_adaptive_momentum_vol_target_132_mode",
    "profit_moonshot_adaptive_momentum_asym_dynamic_mode",
    "profit_moonshot_adaptive_momentum_volume_guard_mode",
    "profit_moonshot_momentum_hybrid_return_mode",
    "profit_moonshot_momentum_hybrid_safe_mode",
    "profit_moonshot_momentum_hybrid_core_mode",
    "profit_moonshot_panic_rebound_mode",
    "profit_moonshot_session_pair_carry_mode",
    "profit_moonshot_balanced_mode",
    "profit_moonshot_trend_mode",
    "profit_moonshot_breakout_mode",
    "profit_moonshot_reversion_mode",
    "profit_moonshot_ensemble_mode",
    "derivatives_flow_squeeze_mode",
    "profit_moonshot_derivatives_taker_flow_mode",
    "profit_moonshot_derivatives_taker_flow_sparse_mode",
    "profit_moonshot_leadlag_slow_diffusion_mode",
    "profit_moonshot_leadlag_slow_diffusion_sol_eth_mode",
    "profit_moonshot_leadlag_slow_diffusion_ensemble_mode",
    "profit_moonshot_hourly_shock_reversion_eth_mode",
    "profit_moonshot_hourly_shock_reversion_eth_12h_mode",
    "profit_moonshot_hourly_shock_reversion_eth_12h_dense_mode",
    "profit_moonshot_hourly_shock_reversion_eth_12h_funding_guard_mode",
    "profit_moonshot_hourly_shock_reversion_eth_12h_sol_regime_guard_mode",
    "profit_moonshot_hourly_shock_reversion_eth_12h_taker_flow_guard_mode",
    "profit_moonshot_hourly_shock_reversion_eth_12h_funding_taker_flow_guard_mode",
    "profit_moonshot_filtered_shock_reversion_diversified_mode",
    "profit_moonshot_taker_flow_exhaustion_eth_mode",
    "profit_moonshot_taker_flow_exhaustion_eth_reactive_mode",
    "profit_moonshot_taker_flow_exhaustion_eth_hold_mode",
    "profit_moonshot_taker_flow_exhaustion_eth_slow_momentum_mode",
    "profit_moonshot_precious_metal_pair_aggressive_mode",
    "artifact_manifest_mode",
}
_PROFIT_MODE_UNBOUNDED_CHILD_TARGET_ALLOCATION = 0.02
_PROFIT_MODE_UNBOUNDED_CHILD_MAX_ORDER_VALUE = 250.0


@dataclass(frozen=True, slots=True)
class PortfolioModeComponent:
    component_id: str
    label: str
    strategy_class: str
    symbols: tuple[str, ...]
    params: dict[str, Any]
    weight: float
    source: str


@dataclass(frozen=True, slots=True)
class PortfolioModeDefinition:
    portfolio_mode: str
    components: tuple[PortfolioModeComponent, ...]
    cash_weight: float
    source_artifacts: dict[str, str]
    watch_symbols: tuple[str, ...] = ()

    @property
    def symbols(self) -> list[str]:
        ordered: list[str] = []
        for component in self.components:
            for symbol in component.symbols:
                if symbol not in ordered:
                    ordered.append(symbol)
        if not ordered:
            for symbol in self.watch_symbols:
                if symbol not in ordered:
                    ordered.append(symbol)
        return ordered


class _BarsSubsetProxy:
    def __init__(self, bars: Any, symbols: list[str]) -> None:
        self._bars = bars
        self.symbol_list = list(symbols or [])

    def __getattr__(self, name: str) -> Any:
        return getattr(self._bars, name)


class _SignalCaptureQueue:
    def __init__(self) -> None:
        self._items: deque[Any] = deque()

    def put(self, item: Any) -> None:
        self._items.append(item)

    def drain(self) -> list[Any]:
        out = list(self._items)
        self._items.clear()
        return out


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object in {path}")
    return payload


def _safe_float(value: Any, default: float = 0.0) -> float:
    return safe_float(value, default)


def _ordered_unique(items: list[str]) -> tuple[str, ...]:
    ordered: list[str] = []
    for item in items:
        token = str(item).strip()
        if token and token not in ordered:
            ordered.append(token)
    return tuple(ordered)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_manifest_path(raw_path: Any, *, base_dir: Path | None = None) -> Path:
    path = Path(str(raw_path or "")).expanduser()
    if not path.is_absolute():
        path = (base_dir or Path.cwd()) / path
    return path.resolve()


def _selection_inputs_are_train_validation_or_lagged_shadow_only(inputs: Any) -> bool:
    if not isinstance(inputs, list | tuple):
        return False
    values = [str(item).lower() for item in inputs]
    if not values:
        return False
    allowed = {"train", "validation", "train_validation", "lagged_completed_shadow_oos"}
    return all(value in allowed for value in values)


def _has_forbidden_oos_provenance(payload: dict[str, Any]) -> bool:
    return any(bool(payload.get(key)) for key in _MANIFEST_FORBIDDEN_OOS_KEYS)


def _manifest_fail_closed_definition(
    *, token: str, source_artifacts: dict[str, str], reason: str
) -> PortfolioModeDefinition:
    artifacts = dict(source_artifacts)
    artifacts["manifest_fail_closed_reason"] = str(reason)
    artifacts["manifest_fail_closed_to_cash"] = "true"
    return PortfolioModeDefinition(
        portfolio_mode=token,
        components=(),
        cash_weight=1.0,
        source_artifacts=artifacts,
        watch_symbols=(),
    )


def _validate_manifest_source_artifacts(
    payload: dict[str, Any], *, manifest_path: Path
) -> tuple[dict[str, Path], str | None]:
    artifacts: dict[str, Path] = {}
    base_dir = manifest_path.parent
    raw_artifacts = payload.get("source_artifacts")
    if not isinstance(raw_artifacts, list | tuple):
        return {}, "source_artifacts_not_list"
    for artifact in raw_artifacts:
        if not isinstance(artifact, dict):
            return {}, "source_artifact_not_object"
        artifact_id = str(artifact.get("id") or artifact.get("artifact_id") or "").strip()
        raw_path = artifact.get("path")
        if not artifact_id or not raw_path:
            return {}, "source_artifact_missing_id_or_path"
        source_path = _resolve_manifest_path(raw_path, base_dir=base_dir)
        if not source_path.exists():
            return {}, f"source_artifact_missing:{artifact_id}"
        if not source_path.is_file():
            return {}, f"source_artifact_not_file:{artifact_id}"
        expected_sha = str(artifact.get("sha256") or "").strip()
        if not expected_sha:
            return {}, f"source_artifact_sha_missing:{artifact_id}"
        try:
            actual_sha = _file_sha256(source_path)
        except OSError:
            return {}, f"source_artifact_unreadable:{artifact_id}"
        if actual_sha != expected_sha:
            return {}, f"source_artifact_sha_mismatch:{artifact_id}"
        max_age_hours = _safe_float(artifact.get("max_age_hours"), 0.0)
        if max_age_hours <= 0.0:
            return {}, f"source_artifact_freshness_missing:{artifact_id}"
        mtime = datetime.fromtimestamp(source_path.stat().st_mtime, UTC)
        age_hours = (datetime.now(UTC) - mtime).total_seconds() / 3600.0
        if age_hours > max_age_hours:
            return {}, f"source_artifact_stale:{artifact_id}"
        if artifact.get("ready") is not True or artifact.get("portfolio_ready") is not True:
            return {}, f"source_artifact_not_ready:{artifact_id}"
        artifacts[artifact_id] = source_path
    return artifacts, None


def _manifest_correlation_ok(provenance: Any) -> bool:
    if not isinstance(provenance, dict) or not provenance:
        return False
    if provenance.get("ready") is not True:
        return False
    source = str(provenance.get("source") or "").strip().lower()
    if not source or "oos" in source or "locked" in source:
        return False
    if not _selection_inputs_are_train_validation_or_lagged_shadow_only(
        provenance.get("selection_inputs")
    ):
        return False
    return not _has_forbidden_oos_provenance(provenance)


def _manifest_optimizer_ok(provenance: Any) -> bool:
    if not isinstance(provenance, dict) or not provenance:
        return False
    if not _selection_inputs_are_train_validation_or_lagged_shadow_only(
        provenance.get("selection_inputs")
    ):
        return False
    return not _has_forbidden_oos_provenance(provenance)


def _manifest_definition_from_path(
    manifest_path: Path, *, token: str, source_artifacts: dict[str, str]
) -> PortfolioModeDefinition:
    artifacts = dict(source_artifacts)
    artifacts["artifact_portfolio_manifest_path"] = str(manifest_path)
    if not manifest_path.exists():
        return _manifest_fail_closed_definition(
            token=token, source_artifacts=artifacts, reason="manifest_missing"
        )
    try:
        payload = _read_json(manifest_path)
    except Exception as exc:  # defensive fail-closed boundary for live routing
        return _manifest_fail_closed_definition(
            token=token,
            source_artifacts=artifacts,
            reason=f"manifest_unreadable:{type(exc).__name__}",
        )

    if any(
        bool(payload.get(key))
        for key in ("real_money_execution", "allow_real_money", "ready_for_real")
    ):
        return _manifest_fail_closed_definition(
            token=token, source_artifacts=artifacts, reason="manifest_real_money_enabled"
        )
    if _has_forbidden_oos_provenance(payload):
        return _manifest_fail_closed_definition(
            token=token, source_artifacts=artifacts, reason="manifest_oos_contaminated"
        )
    if not _manifest_optimizer_ok(payload.get("optimizer_provenance")):
        return _manifest_fail_closed_definition(
            token=token, source_artifacts=artifacts, reason="manifest_optimizer_provenance_invalid"
        )
    if not _manifest_correlation_ok(payload.get("correlation_input_provenance")):
        return _manifest_fail_closed_definition(
            token=token,
            source_artifacts=artifacts,
            reason="manifest_correlation_provenance_invalid",
        )

    source_paths, source_error = _validate_manifest_source_artifacts(
        payload, manifest_path=manifest_path
    )
    if source_error is not None:
        return _manifest_fail_closed_definition(
            token=token, source_artifacts=artifacts, reason=source_error
        )
    if not source_paths:
        return _manifest_fail_closed_definition(
            token=token, source_artifacts=artifacts, reason="source_artifacts_empty"
        )
    for artifact_id, artifact_path in source_paths.items():
        artifacts[f"manifest_source_artifact:{artifact_id}"] = str(artifact_path)

    gross_cap = min(
        MANIFEST_PORTFOLIO_GROSS_CAP,
        _safe_float(payload.get("gross_cap"), MANIFEST_PORTFOLIO_GROSS_CAP),
    )
    if gross_cap <= 0.0:
        return _manifest_fail_closed_definition(
            token=token, source_artifacts=artifacts, reason="manifest_gross_cap_non_positive"
        )

    components: list[PortfolioModeComponent] = []
    gross = 0.0
    netting_gross: dict[str, float] = {}
    raw_children = payload.get("children")
    if not isinstance(raw_children, list | tuple):
        return _manifest_fail_closed_definition(
            token=token, source_artifacts=artifacts, reason="manifest_children_not_list"
        )
    if not raw_children:
        return _manifest_fail_closed_definition(
            token=token, source_artifacts=artifacts, reason="manifest_children_empty"
        )
    for index, child in enumerate(raw_children):
        if not isinstance(child, dict):
            return _manifest_fail_closed_definition(
                token=token, source_artifacts=artifacts, reason=f"child_not_object:{index}"
            )
        child_id = str(child.get("candidate_id") or child.get("name") or index)
        if child.get("ready") is not True or child.get("portfolio_ready") is not True:
            return _manifest_fail_closed_definition(
                token=token, source_artifacts=artifacts, reason=f"child_not_ready:{child_id}"
            )
        if child.get("no_current_fold_oos_provenance") is not True:
            return _manifest_fail_closed_definition(
                token=token,
                source_artifacts=artifacts,
                reason=f"child_current_fold_oos_provenance_missing:{child_id}",
            )
        if (
            child.get("train_validation_optimizer_provenance") is not True
            and child.get("lagged_completed_shadow_optimizer_provenance") is not True
        ):
            return _manifest_fail_closed_definition(
                token=token,
                source_artifacts=artifacts,
                reason=f"child_train_validation_optimizer_provenance_missing:{child_id}",
            )
        if any(
            bool(child.get(key))
            for key in ("real_money_execution", "allow_real_money", "ready_for_real")
        ):
            return _manifest_fail_closed_definition(
                token=token,
                source_artifacts=artifacts,
                reason=f"child_real_money_enabled:{child_id}",
            )
        if _has_forbidden_oos_provenance(child):
            return _manifest_fail_closed_definition(
                token=token, source_artifacts=artifacts, reason=f"child_oos_contaminated:{child_id}"
            )
        child_optimizer = child.get("optimizer_provenance")
        if not _manifest_optimizer_ok(child_optimizer):
            return _manifest_fail_closed_definition(
                token=token,
                source_artifacts=artifacts,
                reason=f"child_optimizer_provenance_invalid:{child_id}",
            )
        child_corr = child.get("correlation_input_provenance")
        if not _manifest_correlation_ok(child_corr):
            return _manifest_fail_closed_definition(
                token=token,
                source_artifacts=artifacts,
                reason=f"child_correlation_provenance_invalid:{child_id}",
            )
        source_id = str(child.get("source_artifact_id") or "").strip()
        if source_paths and source_id not in source_paths:
            return _manifest_fail_closed_definition(
                token=token,
                source_artifacts=artifacts,
                reason=f"child_source_unreconciled:{child_id}",
            )
        weight = _safe_float(child.get("weight"), 0.0)
        if not str(child.get("strategy_class") or "").strip():
            return _manifest_fail_closed_definition(
                token=token, source_artifacts=artifacts, reason=f"child_invalid:{child_id}"
            )
        if not isinstance(child.get("symbols"), list | tuple) or not [
            symbol for symbol in child.get("symbols") if str(symbol).strip()
        ]:
            return _manifest_fail_closed_definition(
                token=token, source_artifacts=artifacts, reason=f"child_invalid:{child_id}"
            )
        if not isinstance(child.get("params"), dict):
            return _manifest_fail_closed_definition(
                token=token, source_artifacts=artifacts, reason=f"child_invalid:{child_id}"
            )
        leaf_gross = abs(_safe_float(child.get("leaf_gross"), weight))
        leaf_cap = _safe_float(child.get("leaf_gross_cap"), gross_cap)
        if leaf_gross > leaf_cap + 1e-12:
            return _manifest_fail_closed_definition(
                token=token,
                source_artifacts=artifacts,
                reason=f"child_leaf_gross_cap_breach:{child_id}",
            )
        group = str(child.get("netting_group") or child_id)
        netting_gross[group] = netting_gross.get(group, 0.0) + leaf_gross
        group_cap = _safe_float(child.get("netting_group_gross_cap"), gross_cap)
        if netting_gross[group] > group_cap + 1e-12:
            return _manifest_fail_closed_definition(
                token=token,
                source_artifacts=artifacts,
                reason=f"child_netting_group_cap_breach:{group}",
            )
        if weight <= 0.0:
            continue
        gross += max(abs(weight), leaf_gross)
        if gross > gross_cap + 1e-12:
            return _manifest_fail_closed_definition(
                token=token, source_artifacts=artifacts, reason="manifest_gross_cap_breach"
            )
        child_row = dict(child)
        child_row["weight"] = weight
        try:
            component = _component_from_row(
                child_row,
                weight=weight,
                source=f"{token}:manifest:{source_id or 'inline'}",
            )
        except TypeError, ValueError:
            return _manifest_fail_closed_definition(
                token=token, source_artifacts=artifacts, reason=f"child_invalid:{child_id}"
            )
        components.append(component)

    if not components:
        return _manifest_fail_closed_definition(
            token=token, source_artifacts=artifacts, reason="manifest_no_positive_children"
        )
    cash_weight = _safe_float(payload.get("cash_weight"), max(0.0, 1.0 - gross))
    return PortfolioModeDefinition(
        portfolio_mode=token,
        components=tuple(_merge_components(components)),
        cash_weight=float(max(0.0, min(1.0, cash_weight))),
        source_artifacts=artifacts,
        watch_symbols=(),
    )


def _component_from_row(
    row: dict[str, Any],
    *,
    weight: float,
    source: str,
) -> PortfolioModeComponent:
    strategy_class = str(row.get("strategy_class") or "").strip()
    if not strategy_class:
        raise ValueError(f"component row is missing strategy_class: {row}")
    params = dict(row.get("params") or {})
    return PortfolioModeComponent(
        component_id=str(row.get("candidate_id") or row.get("name") or strategy_class),
        label=str(row.get("name") or strategy_class),
        strategy_class=strategy_class,
        symbols=tuple(
            str(symbol) for symbol in list(row.get("symbols") or []) if str(symbol).strip()
        ),
        params=params,
        weight=float(weight),
        source=source,
    )


def _pair_component(weight: float) -> PortfolioModeComponent:
    row = _read_json(PAIR_TACTICAL_PATH)
    return _component_from_row(row, weight=weight, source="pair_tactical")


def _state_vwap_pair_component(weight: float) -> PortfolioModeComponent:
    row = _read_json(STATE_VWAP_PAIR_PATH)
    return _component_from_row(row, weight=weight, source="state_vwap_pair")


def _wave2_pair_component(weight: float) -> PortfolioModeComponent:
    row = _read_json(WAVE2_PAIR_PATH)
    return _component_from_row(row, weight=weight, source="wave2_pair")


def _profit_reboot_adaptive_momentum_row(variant: str) -> dict[str, Any]:
    symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "TRX/USDT"]
    base_params: dict[str, Any] = {
        "lookback_bars": 360,
        "short_lookback_bars": 24,
        "regime_lookback_bars": 360,
        "volatility_lookback_bars": 60,
        "rebalance_bars": 72,
        "signal_threshold": 0.040,
        "broad_threshold": 0.0,
        "max_longs": 1,
        "max_shorts": 2,
        "gross_exposure": 0.005,
        "max_order_value": 200.0,
        "stop_loss_pct": 0.0,
        "take_profit_pct": 0.0,
        "trailing_exit_pct": 0.0,
        "max_hold_bars": 0,
        "max_realized_vol": 0.0,
        "min_price": 0.10,
        "btc_symbol": "BTC/USDT",
        "risk_off_exit": True,
    }
    if variant == "defensive":
        base_params.update(
            {
                "lookback_bars": 168,
                "short_lookback_bars": 24,
                "regime_lookback_bars": 168,
                "rebalance_bars": 360,
                "signal_threshold": 0.080,
                "gross_exposure": 0.005,
                "max_order_value": 200.0,
                "stop_loss_pct": 0.0,
                "take_profit_pct": 0.0,
                "trailing_exit_pct": 0.0,
                "max_hold_bars": 0,
            }
        )
    elif variant == "short_bias":
        base_params.update(
            {
                "lookback_bars": 168,
                "short_lookback_bars": 72,
                "regime_lookback_bars": 168,
                "rebalance_bars": 360,
                "signal_threshold": 0.080,
                "max_longs": 0,
                "max_shorts": 2,
                "gross_exposure": 0.005,
                "max_order_value": 200.0,
                "stop_loss_pct": 0.0,
                "take_profit_pct": 0.0,
                "trailing_exit_pct": 0.0,
                "max_hold_bars": 0,
            }
        )
    elif variant == "boost":
        # Same signal logic as the current best live-equivalent moonshot
        # candidate, but with a deliberately bounded 1.5x allocation.  Prior
        # validation showed the balanced sleeve had positive validation PnL and
        # enough train-MDD headroom for a small sizing step; keep the params
        # otherwise identical so the follow-up isolates sizing from signal
        # overfitting.
        base_params.update(
            {
                "gross_exposure": 0.0075,
                "max_order_value": 300.0,
            }
        )

    return {
        "candidate_id": f"profit_reboot_adaptive_momentum_{variant}",
        "name": f"profit_reboot_adaptive_momentum_{variant}",
        "strategy_class": "AdaptiveRegimeMomentumStrategy",
        "symbols": symbols,
        "params": base_params,
        "weight": 1.0,
    }


def _profit_moonshot_adaptive_momentum_scaled_row(
    label: str,
    *,
    gross_exposure: float,
    max_order_value: float,
    extra_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row = _profit_reboot_adaptive_momentum_row("balanced")
    row["candidate_id"] = f"profit_moonshot_adaptive_momentum_{label}"
    row["name"] = f"profit_moonshot_adaptive_momentum_{label}"
    params = dict(row.get("params") or {})
    params.update(
        {
            "gross_exposure": float(gross_exposure),
            "max_order_value": float(max_order_value),
        }
    )
    if extra_params:
        params.update(dict(extra_params))
    row["params"] = params
    return row


def _profit_reboot_panic_rebound_row(variant: str) -> dict[str, Any]:
    symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "TRX/USDT"]
    base_params: dict[str, Any] = {
        "history_bars": 160,
        "return_window": 32,
        "volume_window": 32,
        "vwap_window": 24,
        "shock_return_z": 2.0,
        "shock_return_pct": 0.025,
        "volume_z": 1.0,
        "confirmation_bars": 3,
        "min_rebound_pct": 0.006,
        "vwap_recovery_pct": 0.0,
        "stop_loss_pct": 0.018,
        "take_profit_pct": 0.035,
        "trailing_exit_pct": 0.018,
        "max_hold_bars": 18,
        "target_allocation": 0.08,
        "max_order_value": 300.0,
        "min_price": 0.10,
    }
    if variant == "fast":
        base_params.update(
            {
                "return_window": 18,
                "volume_window": 18,
                "vwap_window": 12,
                "shock_return_z": 1.6,
                "shock_return_pct": 0.018,
                "confirmation_bars": 2,
                "min_rebound_pct": 0.004,
                "take_profit_pct": 0.025,
                "max_hold_bars": 10,
            }
        )

    return {
        "candidate_id": f"profit_reboot_panic_rebound_{variant}",
        "name": f"profit_reboot_panic_rebound_{variant}",
        "strategy_class": "PanicReboundMeanReversionStrategy",
        "symbols": symbols,
        "params": base_params,
        "weight": 1.0,
    }


def _profit_reboot_session_pair_carry_row(variant: str) -> dict[str, Any]:
    base_params: dict[str, Any] = {
        "symbol_x": "BNB/USDT",
        "symbol_y": "TRX/USDT",
        "lookback_window": 96,
        "hedge_window": 192,
        "entry_z": 2.2,
        "exit_z": 0.50,
        "stop_z": 3.8,
        "min_correlation": 0.18,
        "max_hold_bars": 72,
        "cooldown_bars": 6,
        "reentry_z_buffer": 0.25,
        "min_z_turn": 0.02,
        "stop_loss_pct": 0.020,
        "take_profit_pct": 0.045,
        "allowed_session_utc_hours": "0,1,8,9,13,14,15,20,21",
        "min_expected_move_pct": 0.0015,
    }
    if variant == "strict":
        base_params.update(
            {
                "entry_z": 2.6,
                "exit_z": 0.65,
                "stop_z": 4.2,
                "min_correlation": 0.24,
                "max_hold_bars": 48,
                "cooldown_bars": 8,
                "min_expected_move_pct": 0.0025,
            }
        )

    return {
        "candidate_id": f"profit_reboot_session_pair_carry_{variant}",
        "name": f"profit_reboot_session_pair_carry_{variant}",
        "strategy_class": "SessionFilteredPairCarryStrategy",
        "symbols": ["BNB/USDT", "TRX/USDT"],
        "params": base_params,
        "weight": 1.0,
    }


def _profit_reboot_compression_breakout_row(variant: str) -> dict[str, Any]:
    symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "TRX/USDT"]
    base_params: dict[str, Any] = {
        "lookback_bars": 48,
        "compression_window": 24,
        "compression_history_bars": 160,
        "compression_percentile": 0.25,
        "breakout_buffer": 0.002,
        "broad_lookback_bars": 24,
        "broad_threshold": 0.0,
        "stop_loss_pct": 0.025,
        "take_profit_pct": 0.060,
        "trailing_exit_pct": 0.030,
        "max_hold_bars": 72,
        "target_allocation": 0.10,
        "max_order_value": 350.0,
        "btc_symbol": "BTC/USDT",
        "min_price": 0.10,
    }
    if variant == "fast":
        base_params.update(
            {
                "lookback_bars": 32,
                "compression_window": 16,
                "compression_history_bars": 96,
                "compression_percentile": 0.30,
                "breakout_buffer": 0.003,
                "broad_lookback_bars": 12,
                "stop_loss_pct": 0.020,
                "take_profit_pct": 0.045,
                "max_hold_bars": 36,
            }
        )

    return {
        "candidate_id": f"profit_reboot_compression_breakout_{variant}",
        "name": f"profit_reboot_compression_breakout_{variant}",
        "strategy_class": "CompressionBreakoutContinuationStrategy",
        "symbols": symbols,
        "params": base_params,
        "weight": 1.0,
    }


def _profit_moonshot_row(strategy: str, variant: str, weight: float = 1.0) -> dict[str, Any]:
    symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "TRX/USDT"]
    class_by_strategy = {
        "trend": "ProfitMoonshotTrendStrategy",
        "breakout": "ProfitMoonshotBreakoutStrategy",
        "reversion": "ProfitMoonshotReversionStrategy",
    }
    params: dict[str, Any] = {
        "lookback_bars": 720,
        "fast_lookback_bars": 120,
        "slow_lookback_bars": 2_880,
        "rebalance_bars": 120,
        "entry_threshold": 0.018,
        "exit_threshold": 0.002,
        "max_longs": 1,
        "max_shorts": 0,
        "gross_exposure": 0.010,
        "max_order_value": 150.0,
        "stop_loss_pct": 0.018,
        "take_profit_pct": 0.050,
        "trailing_exit_pct": 0.020,
        "max_hold_bars": 96,
        "min_price": 0.10,
        "allow_shorts": False,
    }
    if strategy == "trend":
        params.update(
            {
                "lookback_bars": 1_440,
                "fast_lookback_bars": 360,
                "slow_lookback_bars": 10_080,
                "rebalance_bars": 720,
                "entry_threshold": 0.030,
                "gross_exposure": 0.008,
                "breadth_threshold": 0.004,
                "max_hold_bars": 4_320,
            }
        )
    elif strategy == "breakout":
        params.update(
            {
                "lookback_bars": 360,
                "fast_lookback_bars": 60,
                "slow_lookback_bars": 2_880,
                "rebalance_bars": 60,
                "entry_threshold": 0.014,
                "gross_exposure": 0.008,
                "breakout_buffer": 0.004,
                "squeeze_ratio_max": 1.20,
                "volume_z_min": 0.25,
                "max_hold_bars": 720,
            }
        )
    elif strategy == "reversion":
        params.update(
            {
                "lookback_bars": 180,
                "fast_lookback_bars": 30,
                "slow_lookback_bars": 1_440,
                "rebalance_bars": 30,
                "entry_threshold": 1.50,
                "exit_threshold": 0.25,
                "gross_exposure": 0.006,
                "stop_loss_pct": 0.015,
                "take_profit_pct": 0.035,
                "trailing_exit_pct": 0.018,
                "max_hold_bars": 360,
                "return_z_min": 1.75,
                "volume_z_min": 0.50,
                "range_z_min": 0.50,
            }
        )
    else:
        raise ValueError(f"unsupported profit moonshot strategy: {strategy}")
    if variant == "defensive":
        params["gross_exposure"] = min(0.004, float(params["gross_exposure"]))
        params["max_order_value"] = 75.0
        params["entry_threshold"] = float(params["entry_threshold"]) * 1.25
    return {
        "candidate_id": f"profit_moonshot_{strategy}_{variant}",
        "name": f"profit_moonshot_{strategy}_{variant}",
        "strategy_class": class_by_strategy[strategy],
        "symbols": symbols,
        "params": params,
        "weight": float(weight),
    }


def _derivatives_flow_squeeze_row(
    variant: str,
    weight: float,
    *,
    symbols: list[str] | None = None,
    candidate_id: str | None = None,
    params_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    resolved_symbols = symbols or ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "TRX/USDT"]
    params: dict[str, Any] = {
        "lookback_bars": 192,
        "momentum_lookback_bars": 45,
        "short_reclaim_bars": 6,
        "flow_lookback_bars": 90,
        "oi_lookback_bars": 90,
        "liquidation_window_bars": 192,
        "volatility_lookback_bars": 180,
        "evaluation_cadence_bars": 45,
        "continuation_momentum_min": 0.0012,
        "flow_imbalance_min": 0.045,
        "oi_delta_min": 0.00005,
        "oi_delta_z_min": -0.75,
        "max_abs_continuation_funding": 0.0018,
        "liquidation_z_min": 1.8,
        "liquidation_notional_min": 1.0,
        "price_shock_min": 0.0025,
        "reclaim_min": 0.00045,
        "target_allocation": 0.030,
        "max_order_value": 500.0,
        "volatility_target_per_bar": 0.00115,
        "min_volatility_multiplier": 0.50,
        "max_volatility_multiplier": 1.0,
        "volatility_hard_cap": 0.012,
        "funding_overheat_abs": 0.0040,
        "stop_loss_pct": 0.010,
        "take_profit_pct": 0.025,
        "trailing_exit_pct": 0.010,
        "max_hold_bars": 96,
        "min_price": 0.10,
        "allow_short": True,
        "enable_continuation": True,
        "enable_exhaustion": True,
        "allow_ohlcv_flow_proxy": True,
        "allow_volume_oi_proxy": True,
    }
    if variant == "fast_liquidation_reversal":
        params.update(
            {
                "lookback_bars": 96,
                "momentum_lookback_bars": 18,
                "short_reclaim_bars": 3,
                "flow_lookback_bars": 45,
                "oi_lookback_bars": 45,
                "liquidation_window_bars": 96,
                "volatility_lookback_bars": 90,
                "evaluation_cadence_bars": 15,
                "enable_continuation": False,
                "enable_exhaustion": True,
                "liquidation_z_min": 1.45,
                "price_shock_min": 0.0018,
                "reclaim_min": 0.00030,
                "target_allocation": 0.025,
                "max_order_value": 350.0,
                "volatility_target_per_bar": 0.0010,
                "min_volatility_multiplier": 0.50,
                "stop_loss_pct": 0.012,
                "take_profit_pct": 0.025,
                "trailing_exit_pct": 0.012,
                "max_hold_bars": 96,
            }
        )
    elif variant == "basis_flow_continuation":
        params.update(
            {
                "lookback_bars": 360,
                "momentum_lookback_bars": 180,
                "short_reclaim_bars": 18,
                "flow_lookback_bars": 180,
                "oi_lookback_bars": 180,
                "volatility_lookback_bars": 360,
                "evaluation_cadence_bars": 180,
                "continuation_momentum_min": 0.0020,
                "flow_imbalance_min": 0.035,
                "oi_delta_min": 0.00010,
                "oi_delta_z_min": -0.85,
                "enable_continuation": True,
                "enable_exhaustion": False,
                "target_allocation": 0.030,
                "max_order_value": 500.0,
                "volatility_target_per_bar": 0.00135,
                "min_volatility_multiplier": 0.50,
                "stop_loss_pct": 0.014,
                "take_profit_pct": 0.035,
                "trailing_exit_pct": 0.014,
                "max_hold_bars": 144,
            }
        )
    elif variant != "top5_exhaustion_plus_flow":
        raise ValueError(f"unsupported derivatives-flow squeeze variant: {variant}")
    if params_override:
        params.update(dict(params_override))
    resolved_candidate_id = candidate_id or f"dfse_{variant}"
    return {
        "candidate_id": resolved_candidate_id,
        "name": resolved_candidate_id,
        "strategy_class": "DerivativesFlowSqueezeStrategy",
        "symbols": resolved_symbols,
        "params": params,
        "weight": float(weight),
    }


def _portfolio_weight_rows(path: Path) -> list[dict[str, Any]]:
    payload = _read_json(path)
    rows = [dict(item) for item in list(payload.get("weights") or []) if isinstance(item, dict)]
    out: list[dict[str, Any]] = []
    for row in rows:
        normalized = dict(row)
        if "weight_share" in normalized:
            normalized["weight"] = _safe_float(normalized.get("weight_share"), 0.0)
        else:
            normalized["weight"] = _safe_float(normalized.get("weight"), 0.0)
        out.append(normalized)
    cash_weight = _safe_float(payload.get("cash_weight"), 0.0)
    if cash_weight > 1e-12:
        out.append({"candidate_id": "cash", "name": "cash", "weight": cash_weight})
    return out


def _state_weight_rows(path: Path) -> list[dict[str, Any]]:
    payload = _read_json(path)
    state_weights = dict(dict(payload.get("current_state") or {}).get("weights") or {})
    out: list[dict[str, Any]] = []
    for name, weight in state_weights.items():
        out.append(
            {
                "candidate_id": str(name),
                "name": str(name),
                "weight": _safe_float(weight, 0.0),
            }
        )
    return out


def _hybrid_weight_rows(path: Path) -> list[dict[str, Any]]:
    payload = _read_json(path)
    final_allocation = dict(
        dict((payload.get("scenarios") or {}).get("refreshed_latest_tail") or {}).get(
            "final_allocation"
        )
        or {}
    )
    state_weights = dict(final_allocation.get("weights") or {})
    out: list[dict[str, Any]] = []
    for name, weight in state_weights.items():
        out.append(
            {
                "candidate_id": str(name),
                "name": str(name),
                "weight": _safe_float(weight, 0.0),
            }
        )
    cash_weight = _safe_float(final_allocation.get("cash_weight"), 0.0)
    if cash_weight > 1e-12:
        out.append({"candidate_id": "cash", "name": "cash", "weight": cash_weight})
    return out


def _alias_rows(token: str) -> list[dict[str, Any]] | None:
    portfolio_paths = {
        "incumbent": REFRESHED_INCUMBENT_PATH,
        "incumbent_only": REFRESHED_INCUMBENT_PATH,
        "autoresearch_55_45": REFRESHED_AUTORESEARCH_55_45_PATH,
        "blend_85_15": REFRESHED_BLEND_PATH,
        "static_blend_76_24": REFRESHED_BLEND_PATH,
        "production_guarded_portfolio": PRODUCTION_GUARDED_PATH,
        "strict_autoresearch_1x": STRICT_AUTORESEARCH_1X_PATH,
    }
    state_paths = {
        "soft_three_way_regime": SOFT_THREE_WAY_ALLOCATOR_PATH,
        "three_way_regime": THREE_WAY_ALLOCATOR_PATH,
    }
    hybrid_paths = {
        "hybrid_guarded_mode": HYBRID_PATH,
        "legacy_no_highvol_hybrid_mode": LEGACY_NO_HIGHVOL_HYBRID_PATH,
        "retuned_live_portfolio_hybrid_mode": RETUNED_LIVE_PORTFOLIO_HYBRID_PATH,
    }
    synthetic_rows = {
        "core_mode": [
            {
                "candidate_id": "soft_three_way_regime",
                "name": "soft_three_way_regime",
                "weight": 1.0,
            },
        ],
        "balanced_overlay_mode": [
            {
                "candidate_id": "balanced_overlay_80_20",
                "name": "balanced_overlay_80_20",
                "weight": 1.0,
            },
        ],
        "defensive_overlay_mode": [
            {
                "candidate_id": "soft_three_way_regime",
                "name": "soft_three_way_regime",
                "weight": 0.7,
            },
            {"candidate_id": "pair_tactical_mode", "name": "pair_tactical_mode", "weight": 0.3},
        ],
        "aggressive_realized_mode": [
            {"candidate_id": "three_way_regime", "name": "three_way_regime", "weight": 1.0},
        ],
        "risk_off_mode": [
            {"candidate_id": "cash", "name": "cash", "weight": 1.0},
        ],
        "balanced_overlay_80_20": [
            {
                "candidate_id": "soft_three_way_regime",
                "name": "soft_three_way_regime",
                "weight": 0.8,
            },
            {"candidate_id": "pair_fast_exit", "name": "pair_fast_exit", "weight": 0.2},
        ],
        "pair_tactical_mode": [
            {"candidate_id": "pair_fast_exit", "name": "pair_fast_exit", "weight": 1.0},
        ],
        "strict_autoresearch_practical_mode": [
            {
                "candidate_id": "production_guarded_portfolio",
                "name": "production_guarded_portfolio",
                "weight": 0.8,
            },
            {
                "candidate_id": "strict_autoresearch_1x",
                "name": "strict_autoresearch_1x",
                "weight": 0.2,
            },
        ],
        "production_guarded_state_vwap_pair_mode": [
            {
                "candidate_id": "production_guarded_portfolio",
                "name": "production_guarded_portfolio",
                "weight": 0.4,
            },
            {
                "candidate_id": "state_vwap_pair_leaf",
                "name": "state_vwap_pair_leaf",
                "weight": 0.25,
            },
            {"candidate_id": "cash", "name": "cash", "weight": 0.35},
        ],
        "state_vwap_pair": [
            {"candidate_id": "state_vwap_pair_leaf", "name": "state_vwap_pair_leaf", "weight": 1.0},
        ],
        "wave2_pair": [
            {"candidate_id": "wave2_pair_leaf", "name": "wave2_pair_leaf", "weight": 1.0},
        ],
        "pair_fast_exit": [
            {"candidate_id": "pair_fast_exit_leaf", "name": "pair_fast_exit_leaf", "weight": 1.0},
        ],
        "profit_reboot_adaptive_momentum_mode": [
            _profit_reboot_adaptive_momentum_row("balanced"),
        ],
        "profit_reboot_adaptive_momentum_defensive_mode": [
            _profit_reboot_adaptive_momentum_row("defensive"),
        ],
        "profit_reboot_adaptive_momentum_short_bias_mode": [
            _profit_reboot_adaptive_momentum_row("short_bias"),
        ],
        "profit_reboot_panic_rebound_mode": [
            _profit_reboot_panic_rebound_row("balanced"),
        ],
        "profit_reboot_session_pair_carry_mode": [
            _profit_reboot_session_pair_carry_row("balanced"),
        ],
        "profit_reboot_compression_breakout_mode": [
            _profit_reboot_compression_breakout_row("balanced"),
        ],
        "profit_moonshot_adaptive_momentum_mode": [
            _profit_reboot_adaptive_momentum_row("balanced"),
        ],
        "profit_moonshot_adaptive_momentum_120_mode": [
            _profit_moonshot_adaptive_momentum_scaled_row(
                "120", gross_exposure=0.0060, max_order_value=240.0
            ),
        ],
        "profit_moonshot_adaptive_momentum_130_mode": [
            _profit_moonshot_adaptive_momentum_scaled_row(
                "130", gross_exposure=0.0065, max_order_value=260.0
            ),
        ],
        "profit_moonshot_adaptive_momentum_140_mode": [
            _profit_moonshot_adaptive_momentum_scaled_row(
                "140", gross_exposure=0.0070, max_order_value=280.0
            ),
        ],
        "profit_moonshot_adaptive_momentum_boost_mode": [
            _profit_reboot_adaptive_momentum_row("boost"),
        ],
        "profit_moonshot_adaptive_momentum_governed_mode": [
            _profit_moonshot_adaptive_momentum_scaled_row(
                "governed",
                gross_exposure=0.0070,
                max_order_value=280.0,
                extra_params={
                    "broad_threshold": 0.0015,
                    "max_realized_vol": 0.0035,
                    "rebalance_bars": 96,
                },
            ),
        ],
        "profit_moonshot_adaptive_momentum_vol_target_mode": [
            _profit_moonshot_adaptive_momentum_scaled_row(
                "vol_target",
                gross_exposure=0.0075,
                max_order_value=300.0,
                extra_params={
                    # Keep the boost alpha untouched but volatility-manage
                    # sizing during high-realized-volatility decision bars.
                    # This is intentionally a risk governor rather than
                    # another exposure-only ladder step.
                    "volatility_target_per_bar": 0.00125,
                    "min_volatility_exposure_multiplier": 0.55,
                    "max_volatility_exposure_multiplier": 1.0,
                },
            ),
        ],
        "profit_moonshot_adaptive_momentum_vol_target_132_mode": [
            _profit_moonshot_adaptive_momentum_scaled_row(
                "vol_target_132",
                gross_exposure=0.0075,
                max_order_value=300.0,
                extra_params={
                    # Slightly less conservative than vol_target_mode.  The
                    # 1.32 bp/minute target keeps the train drawdown governor
                    # active but aims to clear the +0.40% validation-return
                    # research hurdle without reverting to exposure-only
                    # sizing.
                    "volatility_target_per_bar": 0.00132,
                    "min_volatility_exposure_multiplier": 0.55,
                    "max_volatility_exposure_multiplier": 1.0,
                },
            ),
        ],
        "profit_moonshot_adaptive_momentum_asym_dynamic_mode": [
            _profit_moonshot_adaptive_momentum_scaled_row(
                "asym_dynamic",
                gross_exposure=0.0075,
                max_order_value=300.0,
                extra_params={
                    "long_exposure_multiplier": 1.0,
                    "short_exposure_multiplier": 0.35,
                    "stop_loss_pct": 0.018,
                    "take_profit_pct": 0.055,
                    "trailing_exit_pct": 0.0,
                    "volatility_trailing_multiplier": 7.0,
                    "min_dynamic_trailing_pct": 0.010,
                    "max_dynamic_trailing_pct": 0.040,
                    "max_hold_bars": 1_440,
                    "volume_lookback_bars": 90,
                    "volume_weighted_broad": True,
                    "min_entry_volume_z": -0.50,
                },
            ),
        ],
        "profit_moonshot_adaptive_momentum_volume_guard_mode": [
            _profit_moonshot_adaptive_momentum_scaled_row(
                "volume_guard",
                gross_exposure=0.0065,
                max_order_value=260.0,
                extra_params={
                    "long_exposure_multiplier": 1.15,
                    "short_exposure_multiplier": 0.25,
                    "stop_loss_pct": 0.015,
                    "take_profit_pct": 0.050,
                    "trailing_exit_pct": 0.0,
                    "volatility_trailing_multiplier": 6.0,
                    "min_dynamic_trailing_pct": 0.008,
                    "max_dynamic_trailing_pct": 0.035,
                    "max_hold_bars": 1_440,
                    "volume_lookback_bars": 90,
                    "volume_weighted_broad": True,
                    "min_entry_volume_z": -0.75,
                },
            ),
        ],
        "profit_moonshot_momentum_hybrid_return_mode": [
            {
                "candidate_id": "profit_moonshot_adaptive_momentum_boost_mode",
                "weight": 0.60,
            },
            {
                "candidate_id": "profit_moonshot_adaptive_momentum_vol_target_132_mode",
                "weight": 0.25,
            },
            {
                "candidate_id": "profit_moonshot_adaptive_momentum_governed_mode",
                "weight": 0.15,
            },
        ],
        "profit_moonshot_momentum_hybrid_safe_mode": [
            {
                "candidate_id": "profit_moonshot_adaptive_momentum_boost_mode",
                "weight": 0.35,
            },
            {
                "candidate_id": "profit_moonshot_adaptive_momentum_vol_target_132_mode",
                "weight": 0.35,
            },
            {
                "candidate_id": "profit_moonshot_adaptive_momentum_governed_mode",
                "weight": 0.20,
            },
            {
                "candidate_id": "profit_moonshot_adaptive_momentum_asym_dynamic_mode",
                "weight": 0.10,
            },
        ],
        "profit_moonshot_momentum_hybrid_core_mode": [
            {
                "candidate_id": "profit_moonshot_adaptive_momentum_boost_mode",
                "weight": 0.40,
            },
            {
                "candidate_id": "profit_moonshot_adaptive_momentum_vol_target_132_mode",
                "weight": 0.40,
            },
            {
                "candidate_id": "profit_moonshot_adaptive_momentum_governed_mode",
                "weight": 0.15,
            },
            {
                "candidate_id": "profit_moonshot_adaptive_momentum_asym_dynamic_mode",
                "weight": 0.05,
            },
        ],
        "profit_moonshot_panic_rebound_mode": [
            _profit_reboot_panic_rebound_row("balanced"),
        ],
        "profit_moonshot_session_pair_carry_mode": [
            _profit_reboot_session_pair_carry_row("balanced"),
        ],
        "profit_moonshot_balanced_mode": [
            _profit_moonshot_row("trend", "balanced", weight=0.35),
            _profit_moonshot_row("breakout", "balanced", weight=0.35),
            _profit_moonshot_row("reversion", "balanced", weight=0.30),
        ],
        "profit_moonshot_trend_mode": [
            _profit_moonshot_row("trend", "balanced"),
        ],
        "profit_moonshot_breakout_mode": [
            _profit_moonshot_row("breakout", "balanced"),
        ],
        "profit_moonshot_reversion_mode": [
            _profit_moonshot_row("reversion", "balanced"),
        ],
        "profit_moonshot_ensemble_mode": [
            _profit_moonshot_row("trend", "ensemble", weight=0.40),
            _profit_moonshot_row("breakout", "ensemble", weight=0.35),
            _profit_moonshot_row("reversion", "ensemble", weight=0.25),
        ],
        "profit_moonshot_leadlag_slow_diffusion_mode": [
            {
                "candidate_id": "profit_moonshot_leadlag_btc_eth_2h_8h_slow_diffusion",
                "name": "profit_moonshot_leadlag_btc_eth_2h_8h_slow_diffusion",
                "strategy_class": "CrossCryptoSlowDiffusionStrategy",
                "symbols": ["BTC/USDT", "ETH/USDT"],
                "params": {
                    # Raw-first screen survivor: BTC/USDT -> ETH/USDT, 2h leader
                    # move threshold, 8h fixed hold. Keep sizing conservative;
                    # this is a new alpha-family validation, not a gross-exposure bump.
                    "leader_symbol": "BTC/USDT",
                    "target_symbol": "ETH/USDT",
                    "timeframe": "1h",
                    "lag_bars": 2,
                    "leader_abs_ret_min": 0.015,
                    "target_underreaction_cap": 999.0,
                    "max_hold_bars": 8,
                    "target_allocation": 0.008,
                    "max_order_value": 175.0,
                    "stop_loss_pct": 0.0,
                    "take_profit_pct": 0.0,
                    "allow_short": True,
                },
                "weight": 1.0,
            },
        ],
        "profit_moonshot_leadlag_slow_diffusion_sol_eth_mode": [
            {
                "candidate_id": "profit_moonshot_leadlag_sol_eth_1h_8h_slow_diffusion",
                "name": "profit_moonshot_leadlag_sol_eth_1h_8h_slow_diffusion",
                "strategy_class": "CrossCryptoSlowDiffusionStrategy",
                "symbols": ["SOL/USDT", "ETH/USDT"],
                "params": {
                    # Raw-first screen survivor: SOL/USDT -> ETH/USDT, 1h lag,
                    # 8h fixed hold.  This is a standalone replacement probe
                    # at the same 0.8% target allocation as the weak BTC->ETH
                    # sleeve, not an exposure increase.
                    "leader_symbol": "SOL/USDT",
                    "target_symbol": "ETH/USDT",
                    "timeframe": "1h",
                    "lag_bars": 1,
                    "leader_abs_ret_min": 0.015,
                    "target_underreaction_cap": 999.0,
                    "max_hold_bars": 8,
                    "target_allocation": 0.008,
                    "max_order_value": 175.0,
                    "stop_loss_pct": 0.0,
                    "take_profit_pct": 0.0,
                    "allow_short": True,
                },
                "weight": 1.0,
            },
        ],
        "profit_moonshot_leadlag_slow_diffusion_ensemble_mode": [
            {
                "candidate_id": "profit_moonshot_leadlag_btc_eth_2h_8h_slow_diffusion",
                "name": "profit_moonshot_leadlag_btc_eth_2h_8h_slow_diffusion",
                "strategy_class": "CrossCryptoSlowDiffusionStrategy",
                "symbols": ["BTC/USDT", "ETH/USDT"],
                "params": {
                    "leader_symbol": "BTC/USDT",
                    "target_symbol": "ETH/USDT",
                    "timeframe": "1h",
                    "lag_bars": 2,
                    "leader_abs_ret_min": 0.015,
                    "target_underreaction_cap": 999.0,
                    "max_hold_bars": 8,
                    "target_allocation": 0.008,
                    "max_order_value": 175.0,
                    "stop_loss_pct": 0.0,
                    "take_profit_pct": 0.0,
                    "allow_short": True,
                },
                "weight": 0.60,
            },
            {
                "candidate_id": "profit_moonshot_leadlag_sol_eth_1h_8h_slow_diffusion",
                "name": "profit_moonshot_leadlag_sol_eth_1h_8h_slow_diffusion",
                "strategy_class": "CrossCryptoSlowDiffusionStrategy",
                "symbols": ["SOL/USDT", "ETH/USDT"],
                "params": {
                    # Same raw-first screen family, second survivor:
                    # SOL/USDT -> ETH/USDT, 1h lag, 8h fixed hold.  The
                    # component weights split the same 0.8% ETH target
                    # allocation instead of increasing gross exposure.
                    "leader_symbol": "SOL/USDT",
                    "target_symbol": "ETH/USDT",
                    "timeframe": "1h",
                    "lag_bars": 1,
                    "leader_abs_ret_min": 0.015,
                    "target_underreaction_cap": 999.0,
                    "max_hold_bars": 8,
                    "target_allocation": 0.008,
                    "max_order_value": 175.0,
                    "stop_loss_pct": 0.0,
                    "take_profit_pct": 0.0,
                    "allow_short": True,
                },
                "weight": 0.40,
            },
        ],
        "profit_moonshot_hourly_shock_reversion_eth_mode": [
            {
                "candidate_id": "profit_moonshot_hourly_shock_reversion_eth_4h_48h_stop2",
                "name": "profit_moonshot_hourly_shock_reversion_eth_4h_48h_stop2",
                "strategy_class": "HourlyShockReversionStrategy",
                "symbols": ["ETH/USDT"],
                "params": {
                    # Stateful raw-first screen survivor: fade ETH 4h shocks
                    # on completed 1h bars, hold up to 48h, and cap downside
                    # at 2%.  Sizing stays at the same conservative 0.8%
                    # target allocation as the weak slow-diffusion sleeve.
                    "target_symbol": "ETH/USDT",
                    "timeframe": "1h",
                    "lookback_bars": 4,
                    "return_threshold": 0.006,
                    "max_hold_bars": 48,
                    "target_allocation": 0.008,
                    "max_order_value": 175.0,
                    "stop_loss_pct": 0.02,
                    "take_profit_pct": 0.0,
                    "allow_long": True,
                    "allow_short": True,
                },
                "weight": 1.0,
            },
        ],
        "profit_moonshot_hourly_shock_reversion_eth_12h_mode": [
            {
                "candidate_id": "profit_moonshot_hourly_shock_reversion_eth_12h_72h_stop5_take10",
                "name": "profit_moonshot_hourly_shock_reversion_eth_12h_72h_stop5_take10",
                "strategy_class": "HourlyShockReversionStrategy",
                "symbols": ["ETH/USDT"],
                "params": {
                    # Second stateful screen survivor: fewer ETH events, wider
                    # 12h shock lookback, 72h hold, 5% stop and 10% take-profit.
                    # Keep the same 0.8% allocation cap instead of increasing
                    # gross exposure.
                    "target_symbol": "ETH/USDT",
                    "timeframe": "1h",
                    "lookback_bars": 12,
                    "return_threshold": 0.01,
                    "max_hold_bars": 72,
                    "target_allocation": 0.008,
                    "max_order_value": 175.0,
                    "stop_loss_pct": 0.05,
                    "take_profit_pct": 0.10,
                    "allow_long": True,
                    "allow_short": True,
                },
                "weight": 1.0,
            },
        ],
        "profit_moonshot_hourly_shock_reversion_eth_12h_dense_mode": [
            {
                "candidate_id": "profit_moonshot_hourly_shock_reversion_eth_12h_72h_stop5_take10_dense08",
                "name": "profit_moonshot_hourly_shock_reversion_eth_12h_72h_stop5_take10_dense08",
                "strategy_class": "HourlyShockReversionStrategy",
                "symbols": ["ETH/USDT"],
                "params": {
                    # Raw-first challenger to the 1.0% shock incumbent:
                    # lower the completed-12h ETH shock threshold to 0.8%
                    # while keeping the exact same 0.8% / $175 risk caps.
                    "target_symbol": "ETH/USDT",
                    "timeframe": "1h",
                    "lookback_bars": 12,
                    "return_threshold": 0.008,
                    "max_hold_bars": 72,
                    "target_allocation": 0.008,
                    "max_order_value": 175.0,
                    "stop_loss_pct": 0.05,
                    "take_profit_pct": 0.10,
                    "allow_long": True,
                    "allow_short": True,
                },
                "weight": 1.0,
            },
        ],
        "profit_moonshot_hourly_shock_reversion_eth_12h_funding_guard_mode": [
            {
                "candidate_id": "profit_moonshot_hourly_shock_reversion_eth_12h_72h_stop5_take10_no_funding_hours",
                "name": "profit_moonshot_hourly_shock_reversion_eth_12h_72h_stop5_take10_no_funding_hours",
                "strategy_class": "HourlyShockReversionStrategy",
                "symbols": ["ETH/USDT"],
                "params": {
                    # Follow-up raw-first survivor after the user rejected
                    # Sharpe<1: lower the ETH shock threshold to 0.8% but
                    # block entries during Binance 8h funding settlement
                    # hours and adjacent windows (00/01, 08/09, 16/17 UTC).
                    # This is a condition/filter change, not a gross exposure
                    # increase; risk caps stay at the same 0.8% / $175.
                    "target_symbol": "ETH/USDT",
                    "timeframe": "1h",
                    "lookback_bars": 12,
                    "return_threshold": 0.008,
                    "max_hold_bars": 72,
                    "target_allocation": 0.008,
                    "max_order_value": 175.0,
                    "stop_loss_pct": 0.05,
                    "take_profit_pct": 0.10,
                    "allow_long": True,
                    "allow_short": True,
                    "excluded_entry_hours_utc": "0,1,8,9,16,17",
                },
                "weight": 1.0,
            },
        ],
        "profit_moonshot_hourly_shock_reversion_eth_12h_taker_flow_guard_mode": [
            {
                "candidate_id": "profit_moonshot_hourly_shock_reversion_eth_12h_72h_stop5_take10_taker_flow_1h10",
                "name": "profit_moonshot_hourly_shock_reversion_eth_12h_72h_stop5_take10_taker_flow_1h10",
                "strategy_class": "HourlyShockReversionStrategy",
                "symbols": ["ETH/USDT"],
                "params": {
                    # Replay survivor 2026-05-06: keep the OOS-return-best
                    # 12h shock threshold and same 0.8% / $175 caps, but only
                    # fade shocks when completed raw taker-flow points show
                    # same-direction exhaustion over the just-completed hour.
                    "target_symbol": "ETH/USDT",
                    "timeframe": "1h",
                    "lookback_bars": 12,
                    "return_threshold": 0.01,
                    "max_hold_bars": 72,
                    "target_allocation": 0.008,
                    "max_order_value": 175.0,
                    "stop_loss_pct": 0.05,
                    "take_profit_pct": 0.10,
                    "allow_long": True,
                    "allow_short": True,
                    "flow_confirmation_lookback_bars": 1,
                    "flow_imbalance_min": 0.10,
                },
                "weight": 1.0,
            },
        ],
        "profit_moonshot_hourly_shock_reversion_eth_12h_funding_taker_flow_guard_mode": [
            {
                "candidate_id": "profit_moonshot_hourly_shock_reversion_eth_12h_72h_stop5_take10_funding_taker_flow_1h10",
                "name": "profit_moonshot_hourly_shock_reversion_eth_12h_72h_stop5_take10_funding_taker_flow_1h10",
                "strategy_class": "HourlyShockReversionStrategy",
                "symbols": ["ETH/USDT"],
                "params": {
                    # Replay survivor 2026-05-06 after discarding the cadence
                    # bankruptcy artifact: use the funding-guard 0.8% shock
                    # threshold, but require completed raw taker-flow exhaustion
                    # over the entry hour.  This changes filters only; the same
                    # 0.8% / $175 risk caps remain unchanged.
                    "target_symbol": "ETH/USDT",
                    "timeframe": "1h",
                    "lookback_bars": 12,
                    "return_threshold": 0.008,
                    "max_hold_bars": 72,
                    "target_allocation": 0.008,
                    "max_order_value": 175.0,
                    "stop_loss_pct": 0.05,
                    "take_profit_pct": 0.10,
                    "allow_long": True,
                    "allow_short": True,
                    "excluded_entry_hours_utc": "0,1,8,9,16,17",
                    "flow_confirmation_lookback_bars": 1,
                    "flow_imbalance_min": 0.10,
                },
                "weight": 1.0,
            },
        ],
        "profit_moonshot_hourly_shock_reversion_eth_12h_sol_regime_guard_mode": [
            {
                "candidate_id": "profit_moonshot_hourly_shock_reversion_eth_12h_72h_stop5_take10_sol_regime35",
                "name": "profit_moonshot_hourly_shock_reversion_eth_12h_72h_stop5_take10_sol_regime35",
                "strategy_class": "HourlyShockReversionStrategy",
                "symbols": ["ETH/USDT", "SOL/USDT"],
                "params": {
                    # Replay survivor 2026-05-06: keep the OOS-return-best
                    # ETH 12h shock configuration, but skip entries when the
                    # completed SOL 24h regime move is counter to the intended
                    # fade by more than 3.5%.  Risk caps stay unchanged.
                    "target_symbol": "ETH/USDT",
                    "timeframe": "1h",
                    "lookback_bars": 12,
                    "return_threshold": 0.01,
                    "max_hold_bars": 72,
                    "target_allocation": 0.008,
                    "max_order_value": 175.0,
                    "stop_loss_pct": 0.05,
                    "take_profit_pct": 0.10,
                    "allow_long": True,
                    "allow_short": True,
                    "regime_symbol": "SOL/USDT",
                    "regime_lookback_bars": 24,
                    "counterguard_return_threshold": 0.035,
                },
                "weight": 1.0,
            },
        ],
        "profit_moonshot_precious_metal_pair_aggressive_mode": [
            {
                "candidate_id": "profit_moonshot_pm_pair_xau_xag_1h_z12_96h",
                "name": "profit_moonshot_pm_pair_xau_xag_1h_z12_96h",
                "strategy_class": "TimeframePairZScoreReversionStrategy",
                "symbols": ["XAU/USDT", "XAG/USDT"],
                "params": {
                    # External thesis: hedge-adjusted precious-metal ratio
                    # spreads can mean-revert, but the gold/silver relation is
                    # regime-dependent; use completed 1h bars, rolling hedge
                    # stats, and explicit pair/leg caps rather than a gross
                    # exposure bump.  Screen survivor on current metals data:
                    # train/val/OOS all positive for XAU/XAG.
                    "symbol_x": "XAU/USDT",
                    "symbol_y": "XAG/USDT",
                    "timeframe": "1h",
                    "lookback_window": 96,
                    "hedge_window": 192,
                    "entry_z": 1.2,
                    "exit_z": 0.2,
                    "stop_z": 3.2,
                    "min_correlation": 0.10,
                    "min_abs_beta": 0.02,
                    "max_abs_beta": 8.0,
                    "max_hold_bars": 72,
                    "stop_loss_pct": 0.020,
                    "take_profit_pct": 0.040,
                    "target_allocation": 0.024,
                    "max_order_value": 350.0,
                    "min_entry_volume_x": 0.01,
                    "min_entry_volume_y": 1.0,
                    "allow_long_spread": True,
                    "allow_short_spread": True,
                },
                "weight": 0.65,
            },
            {
                "candidate_id": "profit_moonshot_pm_pair_xpt_xpd_1h_z18_24h",
                "name": "profit_moonshot_pm_pair_xpt_xpd_1h_z18_24h",
                "strategy_class": "TimeframePairZScoreReversionStrategy",
                "symbols": ["XPT/USDT", "XPD/USDT"],
                "params": {
                    # Aggressive secondary PGM sleeve: val/OOS screen-positive
                    # but train-negative, so keep it below the XAU/XAG sleeve
                    # and do not promote unless live-equivalent OOS validates.
                    "symbol_x": "XPT/USDT",
                    "symbol_y": "XPD/USDT",
                    "timeframe": "1h",
                    "lookback_window": 24,
                    "hedge_window": 72,
                    "entry_z": 1.8,
                    "exit_z": 0.2,
                    "stop_z": 3.6,
                    "min_correlation": 0.10,
                    "min_abs_beta": 0.02,
                    "max_abs_beta": 8.0,
                    "max_hold_bars": 48,
                    "stop_loss_pct": 0.025,
                    "take_profit_pct": 0.045,
                    "target_allocation": 0.024,
                    "max_order_value": 350.0,
                    "min_entry_volume_x": 0.01,
                    "min_entry_volume_y": 0.01,
                    "allow_long_spread": True,
                    "allow_short_spread": True,
                },
                "weight": 0.35,
            },
        ],
        "profit_moonshot_filtered_shock_reversion_diversified_mode": [
            {
                "candidate_id": "profit_moonshot_filtered_eth_12h_funding_guard",
                "name": "profit_moonshot_filtered_eth_12h_funding_guard",
                "strategy_class": "HourlyShockReversionStrategy",
                "symbols": ["ETH/USDT"],
                "params": {
                    # Same filtered ETH sleeve as the single-mode guard, but
                    # scaled to keep total portfolio target allocation at 0.8%.
                    "target_symbol": "ETH/USDT",
                    "timeframe": "1h",
                    "lookback_bars": 12,
                    "return_threshold": 0.008,
                    "max_hold_bars": 72,
                    "target_allocation": 0.008,
                    "max_order_value": 175.0,
                    "stop_loss_pct": 0.05,
                    "take_profit_pct": 0.10,
                    "allow_long": True,
                    "allow_short": True,
                    "excluded_entry_hours_utc": "0,1,8,9,16,17",
                },
                "weight": 0.60,
            },
            {
                "candidate_id": "profit_moonshot_filtered_sol_post_funding_reversion",
                "name": "profit_moonshot_filtered_sol_post_funding_reversion",
                "strategy_class": "HourlyShockReversionStrategy",
                "symbols": ["SOL/USDT"],
                "params": {
                    # SOL sleeve from the same raw-first screen: only trade
                    # the post-funding relief hours, so the alpha family is
                    # explicitly conditional rather than exposure-expanded.
                    "target_symbol": "SOL/USDT",
                    "timeframe": "1h",
                    "lookback_bars": 24,
                    "return_threshold": 0.010,
                    "max_hold_bars": 48,
                    "target_allocation": 0.008,
                    "max_order_value": 175.0,
                    "stop_loss_pct": 0.02,
                    "take_profit_pct": 0.04,
                    "allow_long": True,
                    "allow_short": True,
                    "entry_hours_utc": "2,3,4,10,11,12,18,19,20",
                },
                "weight": 0.40,
            },
        ],
        "profit_moonshot_taker_flow_exhaustion_eth_mode": [
            {
                "candidate_id": "profit_moonshot_taker_flow_exhaustion_eth_12h_us_eu_vol_guard",
                "name": "profit_moonshot_taker_flow_exhaustion_eth_12h_us_eu_vol_guard",
                "strategy_class": "TakerFlowExhaustionReversalStrategy",
                "symbols": ["ETH/USDT"],
                "params": {
                    # New alpha family after Sharpe<1 pushback: fade extreme
                    # same-direction ETH taker-flow/price extensions only
                    # during US/EU overlap hours, with funding and realized-vol
                    # gates. Same 0.8% / $175 risk caps; no exposure-only lift.
                    "target_symbol": "ETH/USDT",
                    "flow_lookback_bars": 90,
                    "momentum_lookback_bars": 180,
                    "volatility_lookback_bars": 180,
                    "evaluation_cadence_bars": 180,
                    "flow_imbalance_min": 0.14,
                    "price_extension_min": 0.006,
                    "funding_abs_cap": 0.00015,
                    "max_realized_volatility": 0.008,
                    "entry_hours_utc": "13,14,15,16,17,18,19,20",
                    "target_allocation": 0.008,
                    "max_order_value": 175.0,
                    "volatility_target_per_bar": 0.0011,
                    "min_volatility_multiplier": 0.35,
                    "max_volatility_multiplier": 1.0,
                    "stop_loss_pct": 0.025,
                    "take_profit_pct": 0.050,
                    "trailing_exit_pct": 0.018,
                    "max_hold_bars": 2160,
                    "allow_short": True,
                },
                "weight": 1.0,
            },
        ],
        "profit_moonshot_taker_flow_exhaustion_eth_reactive_mode": [
            {
                "candidate_id": "profit_moonshot_taker_flow_exhaustion_eth_reactive_us_eu_vol_guard",
                "name": "profit_moonshot_taker_flow_exhaustion_eth_reactive_us_eu_vol_guard",
                "strategy_class": "TakerFlowExhaustionReversalStrategy",
                "symbols": ["ETH/USDT"],
                "params": {
                    # Same flow-exhaustion alpha, but evaluate every market
                    # window so 7-day live-equivalent chunks cannot shift the
                    # hourly cadence away from all validation/OOS events.
                    "target_symbol": "ETH/USDT",
                    "flow_lookback_bars": 90,
                    "momentum_lookback_bars": 180,
                    "volatility_lookback_bars": 180,
                    "evaluation_cadence_bars": 1,
                    "flow_imbalance_min": 0.14,
                    "price_extension_min": 0.006,
                    "funding_abs_cap": 0.00015,
                    "max_realized_volatility": 0.008,
                    "entry_hours_utc": "13,14,15,16,17,18,19,20",
                    "target_allocation": 0.008,
                    "max_order_value": 175.0,
                    "volatility_target_per_bar": 0.0011,
                    "min_volatility_multiplier": 0.35,
                    "max_volatility_multiplier": 1.0,
                    "stop_loss_pct": 0.025,
                    "take_profit_pct": 0.050,
                    "trailing_exit_pct": 0.018,
                    "max_hold_bars": 2160,
                    "allow_short": True,
                },
                "weight": 1.0,
            },
        ],
        "profit_moonshot_taker_flow_exhaustion_eth_hold_mode": [
            {
                "candidate_id": "profit_moonshot_taker_flow_exhaustion_eth_12h_hold_us_eu_vol_guard",
                "name": "profit_moonshot_taker_flow_exhaustion_eth_12h_hold_us_eu_vol_guard",
                "strategy_class": "TakerFlowExhaustionReversalStrategy",
                "symbols": ["ETH/USDT"],
                "params": {
                    # Same alpha as the reactive probe, but align exits to the
                    # 12h raw screen horizon; wide stop/take only protects tail
                    # risk without cutting ordinary mean-reversion paths.
                    "target_symbol": "ETH/USDT",
                    "flow_lookback_bars": 90,
                    "momentum_lookback_bars": 180,
                    "volatility_lookback_bars": 180,
                    "evaluation_cadence_bars": 1,
                    "flow_imbalance_min": 0.14,
                    "price_extension_min": 0.006,
                    "funding_abs_cap": 0.00015,
                    "max_realized_volatility": 0.008,
                    "entry_hours_utc": "13,14,15,16,17,18,19,20",
                    "target_allocation": 0.008,
                    "max_order_value": 175.0,
                    "volatility_target_per_bar": 0.0011,
                    "min_volatility_multiplier": 0.35,
                    "max_volatility_multiplier": 1.0,
                    "stop_loss_pct": 0.050,
                    "take_profit_pct": 0.100,
                    "trailing_exit_pct": 0.0,
                    "max_hold_bars": 2160,
                    "allow_short": True,
                },
                "weight": 1.0,
            },
        ],
        "profit_moonshot_taker_flow_exhaustion_eth_slow_momentum_mode": [
            {
                "candidate_id": "profit_moonshot_taker_flow_exhaustion_eth_6h_mom_cooldown_guard",
                "name": "profit_moonshot_taker_flow_exhaustion_eth_6h_mom_cooldown_guard",
                "strategy_class": "TakerFlowExhaustionReversalStrategy",
                "symbols": ["ETH/USDT"],
                "params": {
                    # Cost-aware replay of the strongest raw train/val/OOS
                    # survivor using a slower 6h price-extension window.  The
                    # 12h post-exit cooldown is a risk control, not an exposure
                    # increase: it prevents repeated fills while one exhaustion
                    # episode is still resolving.
                    "target_symbol": "ETH/USDT",
                    "flow_lookback_bars": 90,
                    "momentum_lookback_bars": 360,
                    "volatility_lookback_bars": 180,
                    "evaluation_cadence_bars": 1,
                    "flow_imbalance_min": 0.14,
                    "price_extension_min": 0.006,
                    "funding_abs_cap": 0.00015,
                    "max_realized_volatility": 0.008,
                    "entry_hours_utc": "13,14,15,16,17,18,19,20",
                    "target_allocation": 0.008,
                    "max_order_value": 175.0,
                    "volatility_target_per_bar": 0.0011,
                    "min_volatility_multiplier": 0.35,
                    "max_volatility_multiplier": 1.0,
                    "stop_loss_pct": 0.050,
                    "take_profit_pct": 0.100,
                    "trailing_exit_pct": 0.0,
                    "max_hold_bars": 2160,
                    "cooldown_bars": 2160,
                    "allow_short": True,
                },
                "weight": 1.0,
            },
        ],
        "derivatives_flow_squeeze_mode": [
            _derivatives_flow_squeeze_row("top5_exhaustion_plus_flow", weight=0.55),
            _derivatives_flow_squeeze_row("fast_liquidation_reversal", weight=0.25),
            _derivatives_flow_squeeze_row("basis_flow_continuation", weight=0.20),
        ],
        "profit_moonshot_derivatives_taker_flow_mode": [
            _derivatives_flow_squeeze_row(
                "basis_flow_continuation",
                weight=0.80,
                symbols=["BTC/USDT", "ETH/USDT", "SOL/USDT"],
                candidate_id="profit_moonshot_dfse_top3_taker_flow_continuation",
                params_override={
                    # New alpha evidence must come from raw aggTrade taker-flow
                    # sidecar replay, not an OHLCV directional proxy.
                    "allow_ohlcv_flow_proxy": False,
                    # OI is still unavailable before 2026-03 in Binance history;
                    # retain the documented volume proxy for train/val, while
                    # OOS uses true OI where the feature inventory supports it.
                    "allow_volume_oi_proxy": True,
                    "flow_imbalance_min": 0.025,
                    "oi_delta_min": 0.0,
                    "oi_delta_z_min": -1.25,
                    "target_allocation": 0.012,
                    "max_order_value": 250.0,
                    "volatility_target_per_bar": 0.00105,
                    "min_volatility_multiplier": 0.45,
                    "stop_loss_pct": 0.010,
                    "take_profit_pct": 0.028,
                    "trailing_exit_pct": 0.010,
                    "max_hold_bars": 96,
                    "enable_continuation": True,
                    "enable_exhaustion": False,
                },
            ),
            _derivatives_flow_squeeze_row(
                "fast_liquidation_reversal",
                weight=0.20,
                symbols=["BTC/USDT", "ETH/USDT", "SOL/USDT"],
                candidate_id="profit_moonshot_dfse_top3_liquidation_gap_probe",
                params_override={
                    "allow_ohlcv_flow_proxy": False,
                    "allow_volume_oi_proxy": True,
                    "target_allocation": 0.006,
                    "max_order_value": 125.0,
                    "liquidation_z_min": 1.25,
                    "enable_continuation": False,
                    "enable_exhaustion": True,
                },
            ),
        ],
        "profit_moonshot_derivatives_taker_flow_sparse_mode": [
            _derivatives_flow_squeeze_row(
                "basis_flow_continuation",
                weight=1.0,
                symbols=["BTC/USDT", "ETH/USDT", "SOL/USDT"],
                candidate_id="profit_moonshot_dfse_top3_sparse_taker_flow",
                params_override={
                    # Stricter cadence/threshold candidate to test whether the
                    # raw taker-flow alpha survives costs when overtrading is
                    # reduced. This is deliberately not an exposure increase.
                    "allow_ohlcv_flow_proxy": False,
                    "allow_volume_oi_proxy": True,
                    "evaluation_cadence_bars": 360,
                    "momentum_lookback_bars": 240,
                    "flow_lookback_bars": 240,
                    "oi_lookback_bars": 240,
                    "continuation_momentum_min": 0.0030,
                    "flow_imbalance_min": 0.055,
                    "oi_delta_min": 0.0,
                    "oi_delta_z_min": -0.50,
                    "target_allocation": 0.008,
                    "max_order_value": 175.0,
                    "volatility_target_per_bar": 0.00095,
                    "min_volatility_multiplier": 0.45,
                    "stop_loss_pct": 0.012,
                    "take_profit_pct": 0.030,
                    "trailing_exit_pct": 0.012,
                    "max_hold_bars": 72,
                    "enable_continuation": True,
                    "enable_exhaustion": False,
                },
            ),
        ],
    }
    if token in portfolio_paths:
        return _portfolio_weight_rows(portfolio_paths[token])
    if token in state_paths:
        return _state_weight_rows(state_paths[token])
    if token in hybrid_paths:
        return _hybrid_weight_rows(hybrid_paths[token])
    if token in synthetic_rows:
        return [dict(item) for item in synthetic_rows[token]]
    return None


def _watch_symbols() -> tuple[str, ...]:
    symbols: list[str] = []
    for path in (REFRESHED_INCUMBENT_PATH, REFRESHED_AUTORESEARCH_55_45_PATH, PAIR_TACTICAL_PATH):
        if not path.exists():
            continue
        payload = _read_json(path)
        rows = (
            [payload]
            if isinstance(payload.get("strategy_class"), str)
            else list(payload.get("weights") or [])
        )
        for row in rows:
            if not isinstance(row, dict):
                continue
            for symbol in list(row.get("symbols") or []):
                token = str(symbol).strip()
                if token and token not in symbols:
                    symbols.append(token)
    return tuple(symbols)


def _expand_reference(
    token: str,
    *,
    weight_scale: float,
    source: str,
    stack: tuple[str, ...] = (),
) -> tuple[list[PortfolioModeComponent], float]:
    token = str(token or "").strip()
    if not token or weight_scale <= 0.0:
        return [], 0.0
    if token in stack:
        raise ValueError(f"cyclic artifact portfolio reference: {' -> '.join([*stack, token])}")
    if token in {"cash", "risk_off_cash"}:
        return [], float(weight_scale)
    if token in {"pair_fast_exit_leaf"}:
        return [_pair_component(float(weight_scale))], 0.0
    if token in {"state_vwap_pair_leaf"}:
        return [_state_vwap_pair_component(float(weight_scale))], 0.0
    if token in {"wave2_pair_leaf"}:
        return [_wave2_pair_component(float(weight_scale))], 0.0

    rows = _alias_rows(token)
    if rows is None:
        raise ValueError(f"unsupported artifact portfolio reference: {token}")

    components: list[PortfolioModeComponent] = []
    cash_weight = 0.0
    next_stack = (*stack, token)
    for row in rows:
        row_weight = _safe_float(row.get("weight"), 0.0)
        if row_weight <= 0.0:
            continue
        scaled_weight = float(weight_scale) * row_weight
        if str(row.get("strategy_class") or "").strip():
            components.append(
                _component_from_row(
                    row,
                    weight=scaled_weight,
                    source=f"{source}:{token}",
                )
            )
            continue
        child_token = str(row.get("candidate_id") or row.get("name") or "").strip()
        child_components, child_cash = _expand_reference(
            child_token,
            weight_scale=scaled_weight,
            source=f"{source}:{token}",
            stack=next_stack,
        )
        components.extend(child_components)
        cash_weight += child_cash
    return components, cash_weight


def _merge_components(components: list[PortfolioModeComponent]) -> list[PortfolioModeComponent]:
    merged: dict[str, PortfolioModeComponent] = {}
    for component in components:
        existing = merged.get(component.component_id)
        if existing is None:
            merged[component.component_id] = component
            continue
        merged[component.component_id] = PortfolioModeComponent(
            component_id=component.component_id,
            label=component.label,
            strategy_class=component.strategy_class,
            symbols=component.symbols,
            params=dict(component.params),
            weight=float(existing.weight + component.weight),
            source=f"{existing.source}+{component.source}",
        )
    return sorted(merged.values(), key=lambda item: item.weight, reverse=True)


def _component_param_override(
    *,
    component: PortfolioModeComponent,
    component_index: int,
    overrides: dict[str, Any],
) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for key in (
        "__all__",
        "*",
        component.strategy_class,
        component.component_id,
        str(component_index),
    ):
        raw = overrides.get(key)
        if isinstance(raw, dict):
            merged.update(dict(raw))
    return merged


def _apply_component_param_overrides(
    definition: PortfolioModeDefinition,
    overrides: dict[str, Any] | None,
) -> PortfolioModeDefinition:
    """Return a definition copy with research-only component param overrides."""
    if not isinstance(overrides, dict) or not overrides:
        return definition

    components: list[PortfolioModeComponent] = []
    for idx, component in enumerate(definition.components):
        param_override = _component_param_override(
            component=component,
            component_index=idx,
            overrides=overrides,
        )
        if not param_override:
            components.append(component)
            continue
        params = dict(component.params or {})
        params.update(param_override)
        components.append(
            PortfolioModeComponent(
                component_id=component.component_id,
                label=component.label,
                strategy_class=component.strategy_class,
                symbols=tuple(component.symbols),
                params=params,
                weight=float(component.weight),
                source=component.source,
            )
        )

    return PortfolioModeDefinition(
        portfolio_mode=definition.portfolio_mode,
        components=tuple(components),
        cash_weight=float(definition.cash_weight),
        source_artifacts=dict(definition.source_artifacts),
        watch_symbols=tuple(definition.watch_symbols),
    )


def resolve_portfolio_mode_definition(portfolio_mode: str) -> PortfolioModeDefinition:
    token = str(portfolio_mode or "").strip()
    if not token:
        raise ValueError("portfolio_mode is required")

    source_artifacts = {
        "hybrid_path": str(HYBRID_PATH.resolve()),
        "legacy_no_highvol_hybrid_path": str(LEGACY_NO_HIGHVOL_HYBRID_PATH.resolve()),
        "retuned_live_portfolio_hybrid_path": str(RETUNED_LIVE_PORTFOLIO_HYBRID_PATH.resolve()),
        "refreshed_incumbent_path": str(REFRESHED_INCUMBENT_PATH.resolve()),
        "refreshed_blend_path": str(REFRESHED_BLEND_PATH.resolve()),
        "refreshed_autoresearch_55_45_path": str(REFRESHED_AUTORESEARCH_55_45_PATH.resolve()),
        "soft_three_way_allocator_path": str(SOFT_THREE_WAY_ALLOCATOR_PATH.resolve()),
        "three_way_allocator_path": str(THREE_WAY_ALLOCATOR_PATH.resolve()),
        "pair_tactical_path": str(PAIR_TACTICAL_PATH.resolve()),
        "production_guarded_path": str(PRODUCTION_GUARDED_PATH.resolve()),
        "state_vwap_pair_path": str(STATE_VWAP_PAIR_PATH.resolve()),
        "wave2_pair_path": str(WAVE2_PAIR_PATH.resolve()),
        "strict_autoresearch_1x_path": str(STRICT_AUTORESEARCH_1X_PATH.resolve()),
        "derivatives_flow_squeeze_manifest_path": str(
            DERIVATIVES_FLOW_SQUEEZE_MANIFEST_PATH.resolve()
        ),
        "precious_metal_pair_screen_path": str(PRECIOUS_METAL_PAIR_SCREEN_PATH.resolve()),
        "default_artifact_portfolio_manifest_path": str(
            DEFAULT_ARTIFACT_PORTFOLIO_MANIFEST_PATH.resolve()
        ),
    }

    components: list[PortfolioModeComponent] = []
    cash_weight = 0.0
    watch_symbols: tuple[str, ...] = ()

    if token == "artifact_manifest_mode":
        return _manifest_definition_from_path(
            DEFAULT_ARTIFACT_PORTFOLIO_MANIFEST_PATH.resolve(),
            token=token,
            source_artifacts=source_artifacts,
        )
    if token.startswith(MANIFEST_PORTFOLIO_MODE_PREFIX):
        return _manifest_definition_from_path(
            _resolve_manifest_path(token[len(MANIFEST_PORTFOLIO_MODE_PREFIX) :]),
            token=token,
            source_artifacts=source_artifacts,
        )
    if token == "risk_off_mode":
        cash_weight = 1.0
        watch_symbols = _watch_symbols()
    elif token == "pair_tactical_mode":
        components, cash_weight = _expand_reference(
            "pair_tactical_mode", weight_scale=1.0, source=token
        )
    elif token == "core_mode":
        components, cash_weight = _expand_reference(
            "soft_three_way_regime", weight_scale=1.0, source=token
        )
    elif token == "balanced_overlay_mode":
        components, cash_weight = _expand_reference(
            "balanced_overlay_80_20", weight_scale=1.0, source=token
        )
    elif token == "defensive_overlay_mode":
        soft_components, soft_cash = _expand_reference(
            "soft_three_way_regime", weight_scale=0.7, source=token
        )
        pair_components, pair_cash = _expand_reference(
            "pair_tactical_mode", weight_scale=0.3, source=token
        )
        components.extend(soft_components)
        components.extend(pair_components)
        cash_weight = soft_cash + pair_cash
    elif token == "aggressive_realized_mode":
        components, cash_weight = _expand_reference(
            "three_way_regime", weight_scale=1.0, source=token
        )
    elif token in {
        "hybrid_guarded_mode",
        "legacy_no_highvol_hybrid_mode",
        "retuned_live_portfolio_hybrid_mode",
    }:
        hybrid_paths = {
            "hybrid_guarded_mode": HYBRID_PATH,
            "legacy_no_highvol_hybrid_mode": LEGACY_NO_HIGHVOL_HYBRID_PATH,
            "retuned_live_portfolio_hybrid_mode": RETUNED_LIVE_PORTFOLIO_HYBRID_PATH,
        }
        hybrid_path = hybrid_paths[token]
        hybrid_payload = _read_json(hybrid_path)
        final_allocation = dict(
            dict((hybrid_payload.get("scenarios") or {}).get("refreshed_latest_tail") or {}).get(
                "final_allocation"
            )
            or {}
        )
        sleeve_weights = {
            str(key): _safe_float(value, 0.0)
            for key, value in dict(final_allocation.get("weights") or {}).items()
        }
        cash_weight = _safe_float(final_allocation.get("cash_weight"), 0.0)
        for sleeve_name, sleeve_weight in sleeve_weights.items():
            if sleeve_weight <= 0.0:
                continue
            sleeve_components, sleeve_cash = _expand_reference(
                sleeve_name,
                weight_scale=sleeve_weight,
                source=token,
            )
            components.extend(sleeve_components)
            cash_weight += sleeve_cash
    elif token == "strict_autoresearch_practical_mode":
        components, cash_weight = _expand_reference(
            "strict_autoresearch_practical_mode",
            weight_scale=1.0,
            source=token,
        )
    elif token == "production_guarded_state_vwap_pair_mode":
        components, cash_weight = _expand_reference(
            "production_guarded_state_vwap_pair_mode",
            weight_scale=1.0,
            source=token,
        )
    elif _alias_rows(token) is not None:
        components, cash_weight = _expand_reference(
            token,
            weight_scale=1.0,
            source=token,
        )
    else:
        raise ValueError(f"unsupported live portfolio mode: {token}")

    merged = tuple(_merge_components([item for item in components if item.weight > 1e-12]))
    return PortfolioModeDefinition(
        portfolio_mode=token,
        components=merged,
        cash_weight=float(max(0.0, min(1.0, cash_weight))),
        source_artifacts=source_artifacts,
        watch_symbols=watch_symbols,
    )


def supported_portfolio_modes() -> set[str]:
    return set(_LIVE_PORTFOLIO_MODE_ALIASES)


def _child_uses_timeframe_aggregator(child: Any) -> bool:
    raw = getattr(child, "uses_timeframe_aggregator", False)
    if callable(raw):
        try:
            raw = raw()
        except Exception:
            raw = False
    if bool(raw):
        return True

    required_inputs = getattr(child, "required_inputs", ()) or ()
    try:
        tokens = tuple(required_inputs)
    except TypeError:
        tokens = ()
    return any(str(token).strip().lower() == "aggregator" for token in tokens)


class ArtifactPortfolioModeStrategy(Strategy):
    preferred_contract = "context"

    def __init__(
        self,
        bars,
        events,
        *,
        portfolio_mode: str,
        component_param_overrides: dict[str, Any] | None = None,
    ):
        self.bars = bars
        self.events = events
        self.portfolio_mode = str(portfolio_mode or "").strip()
        self.definition = _apply_component_param_overrides(
            resolve_portfolio_mode_definition(self.portfolio_mode),
            component_param_overrides,
        )
        self.symbol_list = list(self.definition.symbols)
        self.decision_cadence_seconds = 60
        self._children: list[tuple[PortfolioModeComponent, Any, _SignalCaptureQueue]] = []
        required_features: set[str] = set()
        required_timeframes: set[str] = set()
        uses_timeframe_aggregator = False
        for component in self.definition.components:
            strategy_cls = resolve_strategy_class(
                component.strategy_class, default_name=component.strategy_class
            )
            child_queue = _SignalCaptureQueue()
            child_bars = _BarsSubsetProxy(self.bars, list(component.symbols))
            child = strategy_cls(child_bars, child_queue, **dict(component.params))
            child_required_features = getattr(child, "required_features", ()) or ()
            try:
                required_features.update(
                    str(token).strip() for token in child_required_features if str(token).strip()
                )
            except TypeError:
                pass
            child_uses_timeframe_aggregator = _child_uses_timeframe_aggregator(child)
            uses_timeframe_aggregator = uses_timeframe_aggregator or child_uses_timeframe_aggregator
            if child_uses_timeframe_aggregator:
                raw_timeframes = getattr(child, "required_timeframes", ()) or ()
                required_timeframes.update(
                    str(token) for token in raw_timeframes if str(token).strip()
                )
            self._children.append((component, child, child_queue))
        self.uses_timeframe_aggregator = bool(uses_timeframe_aggregator)
        self.required_features = tuple(sorted(required_features))
        self.required_timeframes = tuple(sorted(required_timeframes))

    def get_state(self) -> dict[str, Any]:
        return {
            "portfolio_mode": self.portfolio_mode,
            "children": {
                component.component_id: dict(getattr(child, "get_state", lambda: {})() or {})
                for component, child, _queue in self._children
            },
        }

    def set_state(self, state: dict) -> None:
        if not isinstance(state, dict):
            return
        raw_children = dict(state.get("children") or {})
        for component, child, _queue in self._children:
            child_state = raw_children.get(component.component_id)
            setter = getattr(child, "set_state", None)
            if callable(setter) and isinstance(child_state, dict):
                setter(child_state)

    def _component_client_order_id(
        self, *, component: PortfolioModeComponent, signal: SignalEvent
    ) -> str:
        base = str(signal.client_order_id or "").strip()
        if base:
            return f"{component.component_id[:12]}-{base}"
        token = "|".join(
            [
                component.component_id,
                str(signal.symbol),
                str(signal.signal_type),
                str(signal.datetime),
                str(getattr(signal, "position_side", "") or ""),
            ]
        )
        digest = hashlib.sha256(token.encode("utf-8")).hexdigest()[:24]
        return f"LQPM-{digest}"

    def _forward_child_signal(self, component: PortfolioModeComponent, signal: SignalEvent) -> None:
        metadata = dict(signal.metadata or {})
        is_profit_mode = self.portfolio_mode.startswith(("profit_moonshot_", "profit_reboot_"))
        child_target_allocation = _safe_float(metadata.get("target_allocation"), 0.0)
        child_max_symbol_exposure = _safe_float(metadata.get("max_symbol_exposure_pct"), 0.0)
        if child_target_allocation > 0.0:
            metadata["child_target_allocation"] = child_target_allocation
            metadata["target_allocation"] = child_target_allocation * float(component.weight)
        elif is_profit_mode:
            fallback_target_allocation = _PROFIT_MODE_UNBOUNDED_CHILD_TARGET_ALLOCATION * float(
                component.weight
            )
            metadata["target_allocation"] = fallback_target_allocation
            metadata["max_symbol_exposure_pct"] = fallback_target_allocation
            metadata["portfolio_mode_unbounded_child_target_allocation"] = (
                _PROFIT_MODE_UNBOUNDED_CHILD_TARGET_ALLOCATION
            )

        if child_max_symbol_exposure > 0.0:
            metadata["child_max_symbol_exposure_pct"] = child_max_symbol_exposure
            metadata["max_symbol_exposure_pct"] = child_max_symbol_exposure * float(
                component.weight
            )

        child_max_order_value = _safe_float(metadata.get("max_order_value"), 0.0)
        if child_max_order_value > 0.0:
            metadata["child_max_order_value"] = child_max_order_value
            metadata["max_order_value"] = child_max_order_value * float(component.weight)
        elif is_profit_mode:
            metadata["max_order_value"] = _PROFIT_MODE_UNBOUNDED_CHILD_MAX_ORDER_VALUE * float(
                component.weight
            )
            metadata["portfolio_mode_unbounded_child_max_order_value"] = (
                _PROFIT_MODE_UNBOUNDED_CHILD_MAX_ORDER_VALUE
            )

        metadata.update(
            {
                "portfolio_mode": self.portfolio_mode,
                "component_id": component.component_id,
                "component_label": component.label,
                "component_weight": float(component.weight),
                "target_allocation_scale": float(component.weight),
            }
        )
        forwarded = SignalEvent(
            strategy_id=f"artifact_portfolio_mode::{self.portfolio_mode}",
            symbol=str(signal.symbol),
            datetime=signal.datetime,
            signal_type=str(signal.signal_type),
            strength=float(getattr(signal, "strength", 1.0) or 1.0) * float(component.weight),
            price=getattr(signal, "price", None),
            stop_loss=getattr(signal, "stop_loss", None),
            take_profit=getattr(signal, "take_profit", None),
            position_side=getattr(signal, "position_side", None),
            client_order_id=self._component_client_order_id(component=component, signal=signal),
            time_in_force=getattr(signal, "time_in_force", None),
            metadata=metadata,
            trailing_percent=getattr(signal, "trailing_percent", None),
        )
        self.events.put(forwarded)

    def _drain_child_queue(
        self, component: PortfolioModeComponent, queue: _SignalCaptureQueue
    ) -> None:
        for item in queue.drain():
            if isinstance(item, SignalEvent):
                self._forward_child_signal(component, item)

    def calculate_signals(self, event: Any) -> None:
        for component, child, child_queue in self._children:
            child.calculate_signals(event)
            self._drain_child_queue(component, child_queue)

    def calculate_signals_window(self, event: Any, aggregator: Any) -> None:
        for component, child, child_queue in self._children:
            handler = getattr(child, "calculate_signals_window", None)
            if callable(handler):
                handler(event, aggregator)
            else:
                child.calculate_signals(event)
            self._drain_child_queue(component, child_queue)

    def calculate_signals_context(self, context: Any) -> None:
        for component, child, child_queue in self._children:
            handler = getattr(child, "calculate_signals_context", None)
            if callable(handler):
                handler(context)
            else:
                child.calculate_signals_window(context.event, context.aggregator)
            self._drain_child_queue(component, child_queue)


__all__ = [
    "ArtifactPortfolioModeStrategy",
    "PortfolioModeComponent",
    "PortfolioModeDefinition",
    "resolve_portfolio_mode_definition",
    "supported_portfolio_modes",
]
