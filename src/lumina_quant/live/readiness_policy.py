from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from lumina_quant.configuration.loader import load_runtime_config, load_yaml_config
from lumina_quant.live_selection import infer_strategy_class_name, supports_live_portfolio_mode

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = Path("config.yaml")
DEFAULT_FOLLOWUP_ROOT = Path("var/reports/exact_window_backtests/followup_status")
DEFAULT_REFRESH_JSON = DEFAULT_FOLLOWUP_ROOT / "final_portfolio_validation_data_refresh_latest.json"
DEFAULT_DECISION_JSON = DEFAULT_FOLLOWUP_ROOT / "portfolio_live_readiness_decision_latest.json"
DEFAULT_PREFLIGHT_STALE_MINUTES = 30
DEFAULT_ALPHA_ZOO_OPTUNA_HYBRID_ARTIFACT = (
    Path("var")
    / "reports"
    / "profit_moonshot_20260501"
    / "current_tail_20260508"
    / "alpha_v2"
    / "alpha_zoo_integer_leverage_optuna_hybrid_decision_20260524"
    / "alpha_zoo_integer_leverage_optuna_hybrid_decision_latest.json"
)
DEFAULT_ALPHA_ZOO_INTEGER_PORTFOLIO_ARTIFACT = (
    Path("var")
    / "reports"
    / "profit_moonshot_20260501"
    / "current_tail_20260508"
    / "alpha_v2"
    / "alpha_zoo_corr_integer_leverage_portfolio_20260524"
    / "alpha_zoo_corr_integer_leverage_portfolio_latest.json"
)
_ALPHA_ZOO_OPTUNA_STRATEGY = "AlphaZooOptunaHybridLiveStrategy"

_TRUTHY = {"1", "true", "yes", "on"}
_BOOL_TRUE_TOKENS = {"true", "1", "yes", "on"}
_BOOL_FALSE_TOKENS = {"false", "0", "no", "off"}


def _as_bool_flag(value: Any) -> bool | None:
    """Coerce an external JSON flag to a tri-state bool.

    Maps bool/str/int encodings of truth ({True, 'true', '1', 1, 'yes', 'on'})
    to True and ({False, 'false', '0', 0, 'no', 'off'}) to False; anything else
    (None, absent, unrecognized) returns None so callers can treat it as "not
    asserted".  Used for BOTH positive readiness flags and blocking/governance
    flags so a flag encoded as ``'true'``/1 is honored regardless of JSON typing.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        if value == 1:
            return True
        if value == 0:
            return False
        return None
    if isinstance(value, str):
        token = value.strip().lower()
        if token in _BOOL_TRUE_TOKENS:
            return True
        if token in _BOOL_FALSE_TOKENS:
            return False
    return None


def _flag_is_true(payload: Mapping[str, Any], key: str) -> bool:
    """True only when ``key`` is present and coerces to a truthy flag."""
    return key in payload and _as_bool_flag(payload.get(key)) is True


def _flag_is_false(payload: Mapping[str, Any], key: str) -> bool:
    """True only when ``key`` is present and coerces to a falsy flag."""
    return key in payload and _as_bool_flag(payload.get(key)) is False


class LiveReadinessBlockedError(RuntimeError):
    """Raised when live startup is not ready for the requested execution mode."""

    def __init__(self, *, mode: str, recommended_action: str, payload: dict[str, Any]) -> None:
        self.mode = str(mode or "").strip().lower() or "paper"
        self.recommended_action = str(recommended_action or "block_until_preflight_gaps_closed")
        self.payload = dict(payload or {})
        super().__init__(
            f"Live readiness blocked for mode={self.mode}. "
            f"recommended_action={self.recommended_action}"
        )


@dataclass(frozen=True, slots=True)
class LiveReadinessVerdict:
    generated_at: str
    config_path: str
    refresh_json: str
    decision_json: str
    checks: dict[str, Any]
    latest: dict[str, Any]
    status: dict[str, bool]
    recommended_action: str

    def as_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["artifact_kind"] = "live_readiness_preflight"
        return payload


def _parse_utc(value: str | None) -> datetime | None:
    token = str(value or "").strip()
    if not token:
        return None
    return datetime.fromisoformat(token.replace("Z", "+00:00")).astimezone(UTC)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_repo_artifact_path(path: Path) -> Path:
    candidate = path if path.is_absolute() else (REPO_ROOT / path)
    if candidate.exists():
        return candidate

    try:
        relative = candidate.relative_to(REPO_ROOT)
    except ValueError:
        return candidate

    omx_root = REPO_ROOT / ".omx"
    if not omx_root.exists():
        return candidate

    fallback_matches = sorted(
        (
            matched
            for matched in omx_root.glob(f"team/*/worktrees/*/{relative.as_posix()}")
            if matched.exists()
        ),
        key=lambda matched: matched.stat().st_mtime,
        reverse=True,
    )
    return fallback_matches[0] if fallback_matches else candidate


def _env_truthy(name: str, env: Mapping[str, str] | None = None) -> bool:
    source = env or os.environ
    return str(source.get(name, "") or "").strip().lower() in _TRUTHY


def effective_startup_reconciliation_hard_fail(*, mode: str, configured: bool) -> bool:
    resolved_mode = str(mode or "").strip().lower()
    return bool(configured) or resolved_mode == "real"


def real_mode_explicitly_enabled(
    *, mode: str, require_real_enable_flag: bool, env: Mapping[str, str] | None = None
) -> bool:
    resolved_mode = str(mode or "").strip().lower()
    if resolved_mode != "real":
        return True
    if not bool(require_real_enable_flag):
        return True
    return _env_truthy("LUMINA_ENABLE_LIVE_REAL", env)


def _decision_allows_live_start(decision: Mapping[str, Any]) -> tuple[bool, bool, bool, str]:
    decision_value = str(decision.get("decision", "") or "").strip().lower()
    decision_keep = decision_value == "keep_incumbent"
    selected_reference = str(
        decision.get("selected_mode")
        or decision.get("selected_live_mode")
        or decision.get("candidate_mode")
        or decision.get("candidate_key")
        or ""
    ).strip()
    decision_promote = decision_value in {"promote_candidate", "selected_live_mode"} and bool(
        selected_reference
    )
    decision_allowed = bool(decision_keep or decision_promote)
    return decision_allowed, decision_keep, decision_promote, selected_reference


def _decision_runtime_compatible(
    *,
    decision_allowed: bool,
    decision_keep: bool,
    selected_reference: str,
) -> bool:
    if not decision_allowed:
        return False
    if decision_keep:
        return True
    return bool(
        infer_strategy_class_name(selected_reference)
        or supports_live_portfolio_mode(selected_reference)
    )


def _decision_strategy_params(decision: Mapping[str, Any]) -> dict[str, Any]:
    params = decision.get("strategy_params")
    if not isinstance(params, Mapping):
        params = decision.get("params")
    return dict(params) if isinstance(params, Mapping) else {}


def _is_alpha_zoo_optuna_hybrid_decision(
    decision: Mapping[str, Any],
    *,
    selected_reference: str,
) -> bool:
    explicit_strategy = str(
        decision.get("strategy_name") or decision.get("strategy_class") or ""
    ).strip()
    inferred = infer_strategy_class_name(selected_reference)
    params = _decision_strategy_params(decision)
    return bool(
        explicit_strategy == _ALPHA_ZOO_OPTUNA_STRATEGY
        or inferred == _ALPHA_ZOO_OPTUNA_STRATEGY
        or params.get("optuna_hybrid_artifact_path")
        or params.get("integer_portfolio_artifact_path")
    )


def _explicit_false(payload: Mapping[str, Any], key: str) -> bool:
    return _flag_is_false(payload, key)


def _artifact_governance_veto_checks(
    payloads: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    containers = list(payloads.values())
    post_oos = any(_flag_is_true(item, "post_oos_research_variant") for item in containers)
    fresh_forward = any(_flag_is_true(item, "requires_fresh_forward_shadow") for item in containers)
    clean_promotion = not any(
        _explicit_false(item, "clean_promotion_eligible") for item in containers
    )
    blockers = [
        reason
        for reason, active in (
            ("post_oos_research_variant", post_oos),
            ("requires_fresh_forward_shadow", fresh_forward),
            ("clean_promotion_eligible_false", not clean_promotion),
        )
        if active
    ]
    return {
        "artifact_post_oos_research_variant": post_oos,
        "artifact_requires_fresh_forward_shadow": fresh_forward,
        "artifact_clean_promotion_eligible": clean_promotion,
        "artifact_governance_veto_reasons": blockers,
    }


def _artifact_reference_paths(decision: Mapping[str, Any]) -> list[str]:
    """Collect artifact-path references declared by a non-AlphaZoo decision.

    Inspects both the decision dict and its strategy params for any key ending
    in ``_artifact_path`` plus the ``candidate_artifact`` / ``selected_artifact``
    aliases, returning de-duplicated string paths in declaration order.
    """
    references: list[str] = []
    for source in (_decision_strategy_params(decision), decision):
        for key, value in source.items():
            key_text = str(key)
            if not (
                key_text.endswith("_artifact_path")
                or key_text in {"candidate_artifact", "selected_artifact"}
            ):
                continue
            path_text = str(value or "").strip()
            if path_text and path_text not in references:
                references.append(path_text)
    return references


# Nested profile keys inspected the same way the AlphaZoo path inspects its
# selected sub-profiles, so a clean top-level payload cannot mask a dirty
# selected/strategy profile.
_NESTED_PROFILE_KEYS = ("selected_profile", "selected_live_profile", "strategy_profiles")


def _profile_paper_only(profile: Mapping[str, Any]) -> bool:
    """A nested profile is paper-only when it self-identifies as a paper candidate."""
    return _flag_is_true(profile, "paper_testnet_only") or _flag_is_true(
        profile, "paper_testnet_candidate"
    )


def _payload_blocks_real_money(payload: Mapping[str, Any]) -> bool:
    """A single payload blocks real money if it explicitly asserts anything dirty.

    Mirrors the AlphaZoo veto conditions at payload granularity: a payload blocks
    when it declares paper-only, explicitly negates any of the positive readiness
    flags (``ready_for_real`` / ``real_execution_allowed`` / ``real_money_execution``),
    or trips a governance blocker.  A silent payload (no relevant keys) does not
    block; it simply fails to assert readiness.
    """
    return bool(
        _flag_is_true(payload, "paper_testnet_only")
        or _flag_is_false(payload, "ready_for_real")
        or _flag_is_false(payload, "real_execution_allowed")
        or _flag_is_false(payload, "real_money_execution")
        or _flag_is_true(payload, "post_oos_research_variant")
        or _flag_is_true(payload, "requires_fresh_forward_shadow")
        or _flag_is_false(payload, "clean_promotion_eligible")
    )


def _collect_profile_payloads(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Return nested profile mappings declared by ``payload`` (one level deep)."""
    nested: list[Mapping[str, Any]] = []
    for key in _NESTED_PROFILE_KEYS:
        value = payload.get(key)
        if isinstance(value, Mapping):
            nested.append(value)
        elif isinstance(value, (list, tuple)):
            nested.extend(item for item in value if isinstance(item, Mapping))
    return nested


def _strategy_agnostic_real_money_veto_checks(
    decision: Mapping[str, Any],
) -> dict[str, Any]:
    """Fail-closed real-money veto for non-AlphaZoo live decisions.

    The veto stays True unless EVERY relevant payload is clean and at least one
    payload positively asserts ``ready_for_real``.  This makes the strategy-agnostic
    veto at least as strict as the AlphaZoo sibling:

    * any payload (decision, strategy_params, referenced artifact, or a nested
      selected/strategy profile) that asserts ``paper_testnet_only`` /
      ``paper_testnet_candidate``, explicitly negates a positive readiness flag,
      or trips a governance blocker forces veto=True — a self-attesting decision
      CANNOT override a referenced dirty artifact;
    * absence of an explicit positive ``ready_for_real`` keeps the veto active;
    * a referenced artifact that cannot be read forces veto=True with
      ``artifact_validation_error``.

    Flags are coerced via :func:`_as_bool_flag`, so ``'true'``/1 and
    ``'false'``/0 string/int encodings are honored in both directions.
    """
    params = _decision_strategy_params(decision)
    containers: dict[str, Mapping[str, Any]] = {"decision": decision, "strategy_params": params}
    try:
        for index, reference in enumerate(_artifact_reference_paths(decision)):
            artifact_path = _resolve_repo_artifact_path(Path(reference))
            containers[f"artifact_{index}"] = _read_json(artifact_path)
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        return {
            "artifact_real_money_veto": True,
            "artifact_paper_testnet_only": None,
            "artifact_ready_for_real": None,
            "artifact_real_execution_allowed": None,
            "artifact_real_money_execution": None,
            "artifact_post_oos_research_variant": None,
            "artifact_requires_fresh_forward_shadow": None,
            "artifact_clean_promotion_eligible": None,
            "artifact_governance_veto_reasons": ["artifact_validation_error"],
            "artifact_validation_error": str(exc),
        }

    # Expand to include any nested selected/strategy profiles so a clean
    # top-level payload cannot mask a dirty selected profile.
    items: list[Mapping[str, Any]] = list(containers.values())
    for payload in list(items):
        items.extend(_collect_profile_payloads(payload))

    governance_payloads = {f"payload_{i}": item for i, item in enumerate(items)}
    governance_checks = _artifact_governance_veto_checks(governance_payloads)
    governance_veto = bool(governance_checks["artifact_governance_veto_reasons"])

    paper_only = any(
        _flag_is_true(item, "paper_testnet_only") or _profile_paper_only(item) for item in items
    )
    ready_for_real = any(_flag_is_true(item, "ready_for_real") for item in items)
    real_execution_allowed = any(_flag_is_true(item, "real_execution_allowed") for item in items)
    real_money_execution = any(_flag_is_true(item, "real_money_execution") for item in items)
    any_blocks = any(_payload_blocks_real_money(item) for item in items)

    # Fail-closed: veto unless a positive ready flag is asserted AND no payload
    # blocks (paper-only / explicit negation / governance) AND no governance veto.
    clean_ready = ready_for_real and not any_blocks and not paper_only and not governance_veto
    return {
        "artifact_real_money_veto": not clean_ready,
        "artifact_paper_testnet_only": paper_only,
        "artifact_ready_for_real": ready_for_real,
        "artifact_real_execution_allowed": real_execution_allowed,
        "artifact_real_money_execution": real_money_execution,
        **governance_checks,
        "artifact_validation_error": "",
    }


def _artifact_real_money_veto_checks(
    decision: Mapping[str, Any],
    *,
    selected_reference: str,
) -> dict[str, Any]:
    target = _is_alpha_zoo_optuna_hybrid_decision(
        decision,
        selected_reference=selected_reference,
    )
    if not target:
        return _strategy_agnostic_real_money_veto_checks(decision)

    params = _decision_strategy_params(decision)
    optuna_path = _resolve_repo_artifact_path(
        Path(params.get("optuna_hybrid_artifact_path") or DEFAULT_ALPHA_ZOO_OPTUNA_HYBRID_ARTIFACT)
    )
    integer_path = _resolve_repo_artifact_path(
        Path(
            params.get("integer_portfolio_artifact_path")
            or DEFAULT_ALPHA_ZOO_INTEGER_PORTFOLIO_ARTIFACT
        )
    )
    try:
        optuna = _read_json(optuna_path)
        integer = _read_json(integer_path)
        selected = dict(optuna.get("selected_optuna_hybrid_profile") or {})
        integer_selected = dict(integer.get("selected_profile") or {})
        governance_checks = _artifact_governance_veto_checks(
            {
                "optuna": optuna,
                "integer": integer,
                "selected_optuna_hybrid_profile": selected,
                "selected_integer_profile": integer_selected,
            }
        )
        governance_veto = bool(governance_checks["artifact_governance_veto_reasons"])
        paper_only = bool(
            optuna.get("paper_testnet_only") is True
            and integer.get("paper_testnet_only") is True
            and selected.get("paper_testnet_candidate") is True
        )
        ready_for_real = bool(
            optuna.get("ready_for_real") is True
            and integer.get("ready_for_real") is True
            and selected.get("ready_for_real") is True
        )
        real_execution_allowed = bool(
            optuna.get("real_execution_allowed") is True
            and integer.get("real_execution_allowed") is True
        )
        real_money_execution = bool(
            optuna.get("real_money_execution") is True
            and integer.get("real_money_execution") is True
            and selected.get("real_money_execution") is True
        )
        # Phase 5: canary_execution_allowed — separate artifact flag for canary stage
        canary_execution_allowed = bool(
            optuna.get("canary_execution_allowed") is True
            and integer.get("canary_execution_allowed") is True
        )
        return {
            "artifact_real_money_veto": bool(
                paper_only
                or not ready_for_real
                or not real_execution_allowed
                or not real_money_execution
                or governance_veto
            ),
            "artifact_paper_testnet_only": paper_only,
            "artifact_ready_for_real": ready_for_real,
            "artifact_real_execution_allowed": real_execution_allowed,
            "artifact_real_money_execution": real_money_execution,
            "artifact_canary_execution_allowed": canary_execution_allowed,
            **governance_checks,
            "artifact_validation_error": "",
            "artifact_optuna_hybrid_path": str(optuna_path),
            "artifact_integer_portfolio_path": str(integer_path),
        }
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        return {
            "artifact_real_money_veto": True,
            "artifact_paper_testnet_only": None,
            "artifact_ready_for_real": None,
            "artifact_real_execution_allowed": None,
            "artifact_real_money_execution": None,
            "artifact_post_oos_research_variant": None,
            "artifact_requires_fresh_forward_shadow": None,
            "artifact_clean_promotion_eligible": None,
            "artifact_governance_veto_reasons": ["artifact_validation_error"],
            "artifact_validation_error": str(exc),
            "artifact_optuna_hybrid_path": str(optuna_path),
            "artifact_integer_portfolio_path": str(integer_path),
        }


def _build_live_readiness_verdict(
    *,
    config_path: Path,
    refresh_json: Path,
    decision_json: Path,
    stale_minutes: int,
    runtime_live: Mapping[str, Any],
    runtime_storage: Mapping[str, Any],
    raw_storage: Mapping[str, Any],
    refresh: Mapping[str, Any],
    decision: Mapping[str, Any],
    env: Mapping[str, str] | None = None,
    now: datetime | None = None,
    shadow_parity_ratio: float | None = None,
) -> LiveReadinessVerdict:
    current_time = (now or datetime.now(UTC)).astimezone(UTC)
    cutoff = _parse_utc(str(refresh.get("collection_cutoff_utc") or ""))
    stale_for_minutes = None if cutoff is None else (current_time - cutoff).total_seconds() / 60.0
    refresh_is_stale = bool(stale_for_minutes is None or stale_for_minutes > float(stale_minutes))

    mode = str(runtime_live.get("mode", "") or "").strip().lower()
    paper_mode = mode == "paper"
    real_mode = mode == "real"
    testnet = bool(runtime_live.get("testnet", False))
    require_real_flag = bool(runtime_live.get("require_real_enable_flag", False))
    real_enable_env = real_mode_explicitly_enabled(
        mode=mode,
        require_real_enable_flag=require_real_flag,
        env=env,
    )
    decision_allowed, decision_keep, decision_promote, decision_reference = (
        _decision_allows_live_start(decision)
    )
    decision_runtime_compatible = _decision_runtime_compatible(
        decision_allowed=decision_allowed,
        decision_keep=decision_keep,
        selected_reference=decision_reference,
    )
    artifact_veto_checks = _artifact_real_money_veto_checks(
        decision,
        selected_reference=decision_reference,
    )
    artifact_real_money_veto = bool(artifact_veto_checks.get("artifact_real_money_veto"))

    postgres_dsn_env_name = (
        str(
            runtime_storage.get("postgres_dsn_env", "")
            or raw_storage.get("postgres_dsn_env", "")
            or "LQ_POSTGRES_DSN"
        ).strip()
        or "LQ_POSTGRES_DSN"
    )
    postgres_dsn = (
        str(runtime_storage.get("postgres_dsn", "") or "").strip()
        or str(raw_storage.get("postgres_dsn", "") or "").strip()
        or str((env or os.environ).get(postgres_dsn_env_name, "") or "").strip()
        or str((env or os.environ).get("LQ__STORAGE__POSTGRES_DSN", "") or "").strip()
    )
    postgres_dsn_present = bool(postgres_dsn)

    refresh_completed = str(refresh.get("status", "") or "").strip().lower() == "completed"
    decision_completed = bool(decision.get("decision"))

    ready_for_paper = bool(
        paper_mode
        and testnet
        and require_real_flag
        and refresh_completed
        and decision_completed
        and not refresh_is_stale
        and decision_allowed
        and decision_runtime_compatible
        and postgres_dsn_present
    )
    ready_for_real = bool(
        real_mode
        and not testnet
        and require_real_flag
        and real_enable_env
        and refresh_completed
        and decision_completed
        and not refresh_is_stale
        and decision_allowed
        and decision_runtime_compatible
        and not artifact_real_money_veto
        and postgres_dsn_present
    )

    # Phase 5 go-live stage pipeline — per-stage entry checks (spec R7: free composition).
    # Each stage's checks must pass for that stage; no sequential traversal required.
    #
    # Parity-default semantics (shadow stage):
    #   shadow entry with unmeasured parity (shadow_parity_ratio=None) is BLOCKED by
    #   design (_shadow_parity_ok requires an explicit measured ratio).  The default
    #   parity_ratio=1.0 in ShadowLiveRunner.evaluate_signal_parity (when no strategy
    #   fns are configured) is a "no comparison made" sentinel for the runner itself —
    #   it does NOT flow here as shadow_parity_ratio; the caller must pass an explicit
    #   float from a real measurement to satisfy _shadow_parity_ok.
    #
    #   canary/full entry NEVER depends on shadow_parity_ratio — ready_for_canary and
    #   ready_for_real hinge exclusively on artifact flags (canary_execution_allowed,
    #   real_money_execution) and the LUMINA_ENABLE_LIVE_REAL env gate.  This is
    #   intentional: shadow is the MEASURING stage (testnet-routed, no money at risk);
    #   canary is the DEPLOYING stage (prod endpoint) with its own artifact gate.
    go_live_stage = str(runtime_live.get("go_live_stage", "testnet") or "testnet").strip().lower()
    shadow_parity_min = float(runtime_live.get("shadow_parity_min_ratio", 0.99) or 0.99)
    _shadow_parity_ok = (
        shadow_parity_ratio is not None and float(shadow_parity_ratio) >= shadow_parity_min
    )
    artifact_canary_allowed = bool(
        artifact_veto_checks.get("artifact_canary_execution_allowed", False)
    )
    ready_for_shadow = bool(ready_for_paper and _shadow_parity_ok)
    # canary: prod endpoint (testnet=False), real_enable_env, canary artifact flag,
    # no real-money veto (artifact must be promoted to allow canary).
    # Does NOT gate on shadow_parity_ratio — see parity-default semantics note above.
    ready_for_canary = bool(
        real_mode
        and not testnet
        and require_real_flag
        and real_enable_env
        and refresh_completed
        and decision_completed
        and not refresh_is_stale
        and decision_allowed
        and decision_runtime_compatible
        and artifact_canary_allowed
        and not artifact_real_money_veto
        and postgres_dsn_present
    )

    feature_common_tail = min(
        (
            row.get("last_timestamp_utc")
            for row in list(refresh.get("feature_results") or [])
            if row.get("last_timestamp_utc")
        ),
        default=None,
    )
    recommended_action = (
        "paper_run_allowed"
        if ready_for_paper
        else "shadow_run_allowed"
        if ready_for_shadow
        else "canary_run_allowed"
        if ready_for_canary
        else "real_run_allowed"
        if ready_for_real
        else "block_until_preflight_gaps_closed"
    )

    return LiveReadinessVerdict(
        generated_at=current_time.isoformat(),
        config_path=str(config_path.resolve()),
        refresh_json=str(refresh_json.resolve()),
        decision_json=str(decision_json.resolve()),
        checks={
            "mode": mode,
            "paper_mode": paper_mode,
            "real_mode": real_mode,
            "testnet": testnet,
            "require_real_enable_flag": require_real_flag,
            "real_enable_env": real_enable_env,
            "postgres_dsn_present": postgres_dsn_present,
            "decision_value": str(decision.get("decision", "") or "").strip().lower(),
            "decision_keep_incumbent": decision_keep,
            "decision_promote_candidate": decision_promote,
            "decision_reference": decision_reference,
            "decision_allows_live_start": decision_allowed,
            "decision_runtime_compatible": decision_runtime_compatible,
            **artifact_veto_checks,
            "refresh_completed": refresh_completed,
            "decision_completed": decision_completed,
            "refresh_is_stale": refresh_is_stale,
            "effective_startup_reconciliation_hard_fail": effective_startup_reconciliation_hard_fail(
                mode=mode,
                configured=bool(runtime_live.get("startup_reconciliation_hard_fail", False)),
            ),
            # Phase 5 stage pipeline checks
            "go_live_stage": go_live_stage,
            "shadow_parity_min_ratio": shadow_parity_min,
            "shadow_parity_ratio": shadow_parity_ratio,
            "shadow_parity_satisfied": _shadow_parity_ok,
            "artifact_canary_execution_allowed": artifact_canary_allowed,
        },
        latest={
            "refresh_cutoff_utc": refresh.get("collection_cutoff_utc"),
            "decision": decision.get("decision"),
            "feature_common_tail_utc": feature_common_tail,
            "stale_for_minutes": stale_for_minutes,
        },
        status={
            "ready_for_paper": ready_for_paper,
            "ready_for_shadow": ready_for_shadow,
            "ready_for_canary": ready_for_canary,
            "ready_for_real": ready_for_real,
            # Stage-keyed alias for enforce_live_readiness_from_files stage dispatch
            "ready_for_testnet": ready_for_paper,
            "ready_for_full": ready_for_real,
        },
        recommended_action=recommended_action,
    )


def build_live_readiness_payload(
    *,
    config_path: Path = DEFAULT_CONFIG_PATH,
    refresh_json: Path = DEFAULT_REFRESH_JSON,
    decision_json: Path = DEFAULT_DECISION_JSON,
    stale_minutes: int = DEFAULT_PREFLIGHT_STALE_MINUTES,
    env: Mapping[str, str] | None = None,
    now: datetime | None = None,
    shadow_parity_ratio: float | None = None,
) -> dict[str, Any]:
    effective_env = env or os.environ
    runtime = load_runtime_config(config_path=str(config_path), env=effective_env)
    raw = load_yaml_config(config_path=str(config_path))
    resolved_refresh_json = _resolve_repo_artifact_path(refresh_json)
    resolved_decision_json = _resolve_repo_artifact_path(decision_json)
    refresh = _read_json(resolved_refresh_json)
    decision = _read_json(resolved_decision_json)
    verdict = _build_live_readiness_verdict(
        config_path=config_path,
        refresh_json=resolved_refresh_json,
        decision_json=resolved_decision_json,
        stale_minutes=max(1, int(stale_minutes)),
        runtime_live=asdict(runtime.live),
        runtime_storage=asdict(runtime.storage),
        raw_storage=dict(raw.get("storage") or {}),
        refresh=refresh,
        decision=decision,
        env=effective_env,
        now=now,
        shadow_parity_ratio=shadow_parity_ratio,
    )
    return verdict.as_payload()


def enforce_live_readiness_from_files(
    *,
    mode: str,
    config_path: Path = DEFAULT_CONFIG_PATH,
    refresh_json: Path = DEFAULT_REFRESH_JSON,
    decision_json: Path = DEFAULT_DECISION_JSON,
    stale_minutes: int = DEFAULT_PREFLIGHT_STALE_MINUTES,
    env: Mapping[str, str] | None = None,
    now: datetime | None = None,
    shadow_parity_ratio: float | None = None,
    go_live_stage: str | None = None,
) -> dict[str, Any]:
    """Enforce live readiness for the given mode and stage.

    ``go_live_stage`` selects which status key to check.  When None, the stage
    is inferred from ``mode``: "real" → "ready_for_real", else "ready_for_paper".
    Explicit stage values map directly: "testnet"→paper, "shadow"→shadow,
    "canary"→canary, "full"→real.

    Per-stage entry checks (spec R7): stage selection is free composition —
    operators may set any stage directly.  What is mandatory is that the
    selected stage's entry checks pass at startup.
    """
    payload = build_live_readiness_payload(
        config_path=config_path,
        refresh_json=refresh_json,
        decision_json=decision_json,
        stale_minutes=stale_minutes,
        env=env,
        now=now,
        shadow_parity_ratio=shadow_parity_ratio,
    )
    status = dict(payload.get("status") or {})
    resolved_mode = str(mode or "").strip().lower() or "paper"
    # Stage-aware status key dispatch
    _stage_key_map = {
        "testnet": "ready_for_paper",
        "shadow": "ready_for_shadow",
        "canary": "ready_for_canary",
        "full": "ready_for_real",
    }
    if go_live_stage is not None:
        resolved_stage = str(go_live_stage or "testnet").strip().lower()
        status_key = _stage_key_map.get(resolved_stage, "ready_for_paper")
    else:
        status_key = "ready_for_real" if resolved_mode == "real" else "ready_for_paper"
    allowed = bool(status.get(status_key, False))
    if not allowed:
        raise LiveReadinessBlockedError(
            mode=resolved_mode,
            recommended_action=str(
                payload.get("recommended_action") or "block_until_preflight_gaps_closed"
            ),
            payload=payload,
        )
    return payload


__all__ = [
    "DEFAULT_CONFIG_PATH",
    "DEFAULT_DECISION_JSON",
    "DEFAULT_FOLLOWUP_ROOT",
    "DEFAULT_PREFLIGHT_STALE_MINUTES",
    "DEFAULT_REFRESH_JSON",
    "LiveReadinessBlockedError",
    "LiveReadinessVerdict",
    "build_live_readiness_payload",
    "effective_startup_reconciliation_hard_fail",
    "enforce_live_readiness_from_files",
    "real_mode_explicitly_enabled",
]
