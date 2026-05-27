from __future__ import annotations

import importlib.util
import inspect
import json
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "research" / "build_portfolio_exact_window_freeze.py"
if not MODULE_PATH.exists():
    pytest.skip(
        "build_portfolio_exact_window_freeze script not yet present", allow_module_level=True
    )

SPEC = importlib.util.spec_from_file_location("build_portfolio_exact_window_freeze", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Failed to load build_portfolio_exact_window_freeze module")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

build_portfolio_exact_window_freeze = MODULE.build_portfolio_exact_window_freeze
write_portfolio_exact_window_freeze = MODULE.write_portfolio_exact_window_freeze


_DAY_MS = 86_400_000.0


def _stream(start_ts_ms: float, values: list[float]) -> list[dict[str, float]]:
    return [{"t": start_ts_ms + (idx * _DAY_MS), "v": value} for idx, value in enumerate(values)]


def _row(
    *,
    candidate_id: str,
    name: str,
    family: str,
    strategy_class: str,
    timeframe: str,
    symbols: list[str],
    train_sharpe: float,
    train_deflated_sharpe: float,
    train_return: float,
    val_sharpe: float,
    val_deflated_sharpe: float,
    val_return: float,
    val_pbo: float,
    val_turnover: float,
    oos_sharpe: float,
    oos_return: float,
    promoted: bool = False,
    committee_decision: str = "reject",
    candidate_pool_eligible: bool = False,
) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "name": name,
        "family": family,
        "strategy_class": strategy_class,
        "strategy_timeframe": timeframe,
        "symbols": symbols,
        "train": {
            "sharpe": train_sharpe,
            "deflated_sharpe": train_deflated_sharpe,
            "return": train_return,
            "pbo": 0.10,
            "turnover": 0.12,
            "trade_count": 20,
        },
        "val": {
            "sharpe": val_sharpe,
            "deflated_sharpe": val_deflated_sharpe,
            "return": val_return,
            "pbo": val_pbo,
            "turnover": val_turnover,
            "trade_count": 8,
        },
        "oos": {
            "sharpe": oos_sharpe,
            "return": oos_return,
            "pbo": 0.05,
            "turnover": 0.09,
            "trade_count": 5,
        },
        "committee": {
            "final_decision": committee_decision,
            "technical_score": 0.9 if promoted else 0.2,
        },
        "promoted": promoted,
        "candidate_pool_eligible": candidate_pool_eligible,
        "hurdle_fields": {
            "oos": {
                "pass": promoted,
                "month": "2026-02",
            }
        },
        "return_streams": {
            "train": _stream(1_735_689_600_000.0, [0.0010, 0.0005]),
            "val": _stream(1_767_225_600_000.0, [0.0020, 0.0010]),
            "oos": _stream(1_769_904_000_000.0, [oos_return, oos_return / 2.0]),
        },
        "metadata": {},
    }


def _call_build(
    tmp_path: Path,
    grouped_rows: dict[str, list[dict[str, Any]]],
    *,
    extra_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    params = inspect.signature(build_portfolio_exact_window_freeze).parameters
    flat_rows = [dict(row) for rows in grouped_rows.values() for row in rows]
    pair_payload = {
        "survives": False,
        "coverage_guard": {"pass": False, "observed_total_days": 36, "min_total_days": 60},
    }
    candidate_kwargs: dict[str, Any] = {
        "report_root": tmp_path,
        "output_dir": tmp_path,
        "grouped_rows": grouped_rows,
        "grouped_candidate_rows": grouped_rows,
        "candidate_groups": grouped_rows,
        "rows_by_sleeve": grouped_rows,
        "sleeve_rows": grouped_rows,
        "sleeve_groups": grouped_rows,
        "source_payloads": grouped_rows,
        "component_rows": flat_rows,
        "candidate_rows": flat_rows,
        "rows": flat_rows,
        "pair_payload": pair_payload,
        "pair_spread_payload": pair_payload,
        "pairspread_payload": pair_payload,
    }
    if extra_kwargs:
        candidate_kwargs.update(extra_kwargs)
    call_kwargs = {name: value for name, value in candidate_kwargs.items() if name in params}
    try:
        return build_portfolio_exact_window_freeze(**call_kwargs)
    except TypeError as exc:  # pragma: no cover - assertion payload is the point.
        raise AssertionError(
            "Unable to call build_portfolio_exact_window_freeze with supported kwargs "
            f"{sorted(call_kwargs)} for signature {inspect.signature(build_portfolio_exact_window_freeze)}"
        ) from exc


def _call_write(
    tmp_path: Path,
    grouped_rows: dict[str, list[dict[str, Any]]],
    *,
    extra_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    params = inspect.signature(write_portfolio_exact_window_freeze).parameters
    flat_rows = [dict(row) for rows in grouped_rows.values() for row in rows]
    pair_payload = {
        "survives": False,
        "coverage_guard": {"pass": False, "observed_total_days": 36, "min_total_days": 60},
    }
    candidate_kwargs: dict[str, Any] = {
        "report_root": tmp_path,
        "output_dir": tmp_path,
        "run_name": "test_portfolio_exact_window_freeze",
        "grouped_rows": grouped_rows,
        "grouped_candidate_rows": grouped_rows,
        "candidate_groups": grouped_rows,
        "rows_by_sleeve": grouped_rows,
        "sleeve_rows": grouped_rows,
        "sleeve_groups": grouped_rows,
        "source_payloads": grouped_rows,
        "component_rows": flat_rows,
        "candidate_rows": flat_rows,
        "rows": flat_rows,
        "pair_payload": pair_payload,
        "pair_spread_payload": pair_payload,
        "pairspread_payload": pair_payload,
    }
    if extra_kwargs:
        candidate_kwargs.update(extra_kwargs)
    accepts_kwargs = any(param.kind is inspect.Parameter.VAR_KEYWORD for param in params.values())
    call_kwargs = (
        dict(candidate_kwargs)
        if accepts_kwargs
        else {name: value for name, value in candidate_kwargs.items() if name in params}
    )
    try:
        return write_portfolio_exact_window_freeze(**call_kwargs)
    except TypeError as exc:  # pragma: no cover - assertion payload is the point.
        raise AssertionError(
            "Unable to call write_portfolio_exact_window_freeze with supported kwargs "
            f"{sorted(call_kwargs)} for signature {inspect.signature(write_portfolio_exact_window_freeze)}"
        ) from exc


def _unwrap_payload(payload: dict[str, Any]) -> dict[str, Any]:
    inner = payload.get("payload")
    return dict(inner) if isinstance(inner, dict) else payload


def _selection_basis(payload: dict[str, Any]) -> str:
    payload = _unwrap_payload(payload)
    selection = payload.get("selection")
    selection_meta = selection if isinstance(selection, dict) else {}
    metadata = payload.get("metadata")
    metadata_dict = metadata if isinstance(metadata, dict) else {}
    return str(
        payload.get("selection_basis")
        or selection_meta.get("selection_basis")
        or metadata_dict.get("selection_basis")
        or ""
    )


def _selected_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    payload = _unwrap_payload(payload)
    selection = payload.get("selection")
    selection_meta = selection if isinstance(selection, dict) else {}
    rows = (
        payload.get("selected_team")
        or payload.get("candidates")
        or selection_meta.get("selected_team")
        or selection_meta.get("candidates")
        or []
    )
    return [dict(row) for row in rows if isinstance(row, dict)]


def _selected_ids(payload: dict[str, Any]) -> set[str]:
    return {
        str(row.get("candidate_id") or row.get("name") or "") for row in _selected_rows(payload)
    }


def _rolling_admission_blocked(payload: dict[str, Any]) -> bool:
    payload = _unwrap_payload(payload)
    selection = payload.get("selection")
    selection_meta = selection if isinstance(selection, dict) else {}
    metadata = payload.get("metadata")
    metadata_dict = metadata if isinstance(metadata, dict) else {}
    for container in (payload, selection_meta, metadata_dict):
        value = container.get("rolling_admission_blocked")
        if isinstance(value, bool):
            return value
    return False


def _base_incumbent_grouped_rows() -> dict[str, list[dict[str, Any]]]:
    return {
        "composite_trend_30m": [
            _row(
                candidate_id="composite-anchor",
                name="composite_trend_30m_anchor",
                family="trend",
                strategy_class="CompositeTrendStrategy",
                timeframe="30m",
                symbols=["BTC/USDT"],
                train_sharpe=1.0,
                train_deflated_sharpe=0.5,
                train_return=0.02,
                val_sharpe=1.4,
                val_deflated_sharpe=0.7,
                val_return=0.04,
                val_pbo=0.08,
                val_turnover=0.09,
                oos_sharpe=0.4,
                oos_return=0.02,
            )
        ],
        "topcap_tsmom_1h": [
            _row(
                candidate_id="topcap-anchor",
                name="topcap_tsmom_1h_anchor",
                family="cross_sectional",
                strategy_class="TopCapTimeSeriesMomentumStrategy",
                timeframe="1h",
                symbols=["BTC/USDT", "ETH/USDT", "BNB/USDT"],
                train_sharpe=1.2,
                train_deflated_sharpe=0.6,
                train_return=0.03,
                val_sharpe=1.5,
                val_deflated_sharpe=0.7,
                val_return=0.04,
                val_pbo=0.09,
                val_turnover=0.08,
                oos_sharpe=0.6,
                oos_return=0.025,
            )
        ],
        "regime_breakout_1h": [
            _row(
                candidate_id="regime-anchor",
                name="regime_breakout_1h_anchor",
                family="trend",
                strategy_class="RegimeBreakoutCandidateStrategy",
                timeframe="1h",
                symbols=["BTC/USDT", "ETH/USDT"],
                train_sharpe=1.1,
                train_deflated_sharpe=0.55,
                train_return=0.025,
                val_sharpe=1.45,
                val_deflated_sharpe=0.68,
                val_return=0.038,
                val_pbo=0.10,
                val_turnover=0.07,
                oos_sharpe=0.55,
                oos_return=0.022,
            )
        ],
    }


def _write_incumbent_bundle(
    tmp_path: Path, rows: list[dict[str, Any]]
) -> tuple[Path, dict[str, Any]]:
    bundle_path = tmp_path / "portfolio_one_shot_incumbent_bundle_latest.json"
    weighted_rows = []
    total = max(1, len(rows))
    for row in rows:
        weighted = dict(row)
        weighted["portfolio_weight"] = 1.0 / float(total)
        weighted_rows.append(weighted)
    payload = {
        "artifact_kind": "portfolio_one_shot_incumbent_bundle",
        "selection_basis": "incumbent_saved_one_shot_weights",
        "candidates": weighted_rows,
        "selected_team": weighted_rows,
    }
    bundle_path.write_text(json.dumps(payload), encoding="utf-8")
    return bundle_path, payload


def test_build_portfolio_exact_window_freeze_prefers_validation_only_and_excludes_pair_spread(
    tmp_path: Path,
):
    grouped_rows = {
        "composite_trend_30m": [
            _row(
                candidate_id="composite-val-winner",
                name="composite_trend_30m_candidate_a",
                family="trend",
                strategy_class="CompositeTrendStrategy",
                timeframe="30m",
                symbols=["BTC/USDT"],
                train_sharpe=1.3,
                train_deflated_sharpe=0.7,
                train_return=0.03,
                val_sharpe=1.9,
                val_deflated_sharpe=0.8,
                val_return=0.05,
                val_pbo=0.08,
                val_turnover=0.11,
                oos_sharpe=-9.0,
                oos_return=-0.04,
                promoted=False,
                committee_decision="reject",
                candidate_pool_eligible=False,
            ),
            _row(
                candidate_id="composite-oos-lure",
                name="composite_trend_30m_candidate_b",
                family="trend",
                strategy_class="CompositeTrendStrategy",
                timeframe="30m",
                symbols=["BTC/USDT"],
                train_sharpe=0.4,
                train_deflated_sharpe=0.2,
                train_return=0.01,
                val_sharpe=0.5,
                val_deflated_sharpe=0.1,
                val_return=0.01,
                val_pbo=0.30,
                val_turnover=0.60,
                oos_sharpe=12.0,
                oos_return=0.25,
                promoted=True,
                committee_decision="promote",
                candidate_pool_eligible=True,
            ),
        ],
        "topcap_tsmom_1h": [
            _row(
                candidate_id="topcap-val-winner",
                name="topcap_tsmom_1h_candidate_a",
                family="cross_sectional",
                strategy_class="TopCapTimeSeriesMomentumStrategy",
                timeframe="1h",
                symbols=["BTC/USDT", "ETH/USDT", "BNB/USDT"],
                train_sharpe=1.1,
                train_deflated_sharpe=0.6,
                train_return=0.02,
                val_sharpe=1.6,
                val_deflated_sharpe=0.7,
                val_return=0.04,
                val_pbo=0.07,
                val_turnover=0.09,
                oos_sharpe=0.5,
                oos_return=0.03,
            ),
            _row(
                candidate_id="topcap-oos-lure",
                name="topcap_tsmom_1h_candidate_b",
                family="cross_sectional",
                strategy_class="TopCapTimeSeriesMomentumStrategy",
                timeframe="1h",
                symbols=["BTC/USDT", "ETH/USDT", "BNB/USDT"],
                train_sharpe=0.2,
                train_deflated_sharpe=0.1,
                train_return=0.005,
                val_sharpe=0.4,
                val_deflated_sharpe=0.1,
                val_return=0.008,
                val_pbo=0.40,
                val_turnover=0.55,
                oos_sharpe=8.0,
                oos_return=0.20,
                promoted=True,
                committee_decision="promote",
                candidate_pool_eligible=True,
            ),
        ],
        "pair_spread_4h": [
            _row(
                candidate_id="pair-spread-should-be-excluded",
                name="pair_spread_4h_xpt_xpd_candidate",
                family="market_neutral",
                strategy_class="PairSpreadZScoreStrategy",
                timeframe="4h",
                symbols=["XPT/USDT", "XPD/USDT"],
                train_sharpe=2.0,
                train_deflated_sharpe=1.0,
                train_return=0.05,
                val_sharpe=2.5,
                val_deflated_sharpe=1.2,
                val_return=0.07,
                val_pbo=0.03,
                val_turnover=0.05,
                oos_sharpe=2.0,
                oos_return=0.08,
                promoted=True,
                committee_decision="promote",
                candidate_pool_eligible=True,
            )
        ],
    }

    payload = _call_build(tmp_path, grouped_rows)
    selected = _selected_rows(payload)
    selected_ids = {str(row.get("candidate_id") or row.get("name")) for row in selected}

    assert selected, "expected build_portfolio_exact_window_freeze to emit selected rows"
    assert _selection_basis(payload) == "validation_only"
    assert "composite-val-winner" in selected_ids
    assert "composite-oos-lure" not in selected_ids
    assert "topcap-val-winner" in selected_ids
    assert "topcap-oos-lure" not in selected_ids
    assert "pair-spread-should-be-excluded" not in selected_ids
    assert all(
        str(row.get("strategy_class") or "") != "PairSpreadZScoreStrategy" for row in selected
    )
    assert len(selected) == 2
    optimizer_bundle = dict(_unwrap_payload(payload).get("optimizer_bundle") or {})
    assert len(list(optimizer_bundle.get("candidates") or [])) == 2
    assert len(list(optimizer_bundle.get("selected_team") or [])) == 2

    composite_row = next(
        row for row in selected if str(row.get("candidate_id")) == "composite-val-winner"
    )
    assert (
        composite_row["return_streams"]["oos"][0]["t"]
        == grouped_rows["composite_trend_30m"][0]["return_streams"]["oos"][0]["t"]
    )


def test_build_portfolio_exact_window_freeze_keeps_rollingbreakout_base_row_before_gate_supplement(
    tmp_path: Path,
):
    grouped_rows = {
        "rolling_breakout_30m": [
            _row(
                candidate_id="rolling-base-winner",
                name="rolling_breakout_30m_guarded_base",
                family="trend",
                strategy_class="RollingBreakoutStrategy",
                timeframe="30m",
                symbols=["BTC/USDT"],
                train_sharpe=1.2,
                train_deflated_sharpe=0.6,
                train_return=0.02,
                val_sharpe=1.8,
                val_deflated_sharpe=0.9,
                val_return=0.05,
                val_pbo=0.10,
                val_turnover=0.12,
                oos_sharpe=-4.0,
                oos_return=-0.03,
            ),
            _row(
                candidate_id="rolling-oos-lure",
                name="rolling_breakout_30m_guarded_alt",
                family="trend",
                strategy_class="RollingBreakoutStrategy",
                timeframe="30m",
                symbols=["BTC/USDT"],
                train_sharpe=0.4,
                train_deflated_sharpe=0.2,
                train_return=0.005,
                val_sharpe=0.6,
                val_deflated_sharpe=0.2,
                val_return=0.01,
                val_pbo=0.45,
                val_turnover=0.70,
                oos_sharpe=7.0,
                oos_return=0.20,
                promoted=True,
                committee_decision="promote",
                candidate_pool_eligible=True,
            ),
        ]
    }
    rolling_gate_payload = {
        "survives": True,
        "recommended_action": "activate_conditionally",
        "gated_candidate_row": {
            "candidate_id": "rolling-base-winner",
            "metadata": {
                "activation_rule_id": "basket_vol_ratio_moderate",
                "activation_rule_survives": True,
            },
            "return_streams": {
                "oos": _stream(1_769_904_000_000.0, [0.01, 0.02]),
            },
        },
    }

    payload = _call_build(
        tmp_path,
        grouped_rows,
        extra_kwargs={
            "rolling_gate_payload": rolling_gate_payload,
            "rolling_breakout_gate_payload": rolling_gate_payload,
            "gate_payload": rolling_gate_payload,
        },
    )
    selected = _selected_rows(payload)

    assert len(selected) == 1
    rolling_row = selected[0]
    assert str(rolling_row.get("candidate_id")) == "rolling-base-winner"

    metadata = rolling_row.get("metadata")
    metadata_dict = metadata if isinstance(metadata, dict) else {}
    if metadata_dict:
        assert metadata_dict.get("activation_rule_survives") is True
        assert metadata_dict.get("activation_rule_id") == "basket_vol_ratio_moderate"


def test_write_portfolio_exact_window_freeze_writes_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    grouped_rows = {
        "topcap_tsmom_1h": [
            _row(
                candidate_id="topcap-val-winner",
                name="topcap_tsmom_1h_candidate_a",
                family="cross_sectional",
                strategy_class="TopCapTimeSeriesMomentumStrategy",
                timeframe="1h",
                symbols=["BTC/USDT", "ETH/USDT", "BNB/USDT"],
                train_sharpe=1.0,
                train_deflated_sharpe=0.5,
                train_return=0.02,
                val_sharpe=1.4,
                val_deflated_sharpe=0.7,
                val_return=0.04,
                val_pbo=0.08,
                val_turnover=0.09,
                oos_sharpe=0.4,
                oos_return=0.02,
            )
        ]
    }
    equal_weight_path = tmp_path / "committee_portfolio_followup_latest.json"
    equal_weight_path.write_text(
        json.dumps(
            {
                "generated_at": "2026-03-14T00:00:00Z",
                "selection": [
                    {
                        "name": "equal_weight_component",
                        "strategy_class": "CompositeTrendStrategy",
                        "timeframe": "30m",
                        "symbols": ["BTC/USDT"],
                        "return_streams": {
                            "train": _stream(1_735_689_600_000.0, [0.0010]),
                            "val": _stream(1_767_225_600_000.0, [0.0020]),
                            "oos": _stream(1_769_904_000_000.0, [0.0030]),
                        },
                    }
                ],
                "metrics": {"oos": {"return": 0.03}},
            }
        ),
        encoding="utf-8",
    )
    one_shot_path = tmp_path / "portfolio_optimization_latest.json"
    one_shot_path.write_text(
        json.dumps(
            {
                "generated_at": "2026-03-14T00:00:00Z",
                "portfolio_return_streams": {
                    "train": _stream(1_735_689_600_000.0, [0.0008]),
                    "val": _stream(1_767_225_600_000.0, [0.0015]),
                    "oos": _stream(1_769_904_000_000.0, [0.0025]),
                },
                "portfolio_metrics": {"oos": {"total_return": 0.025}},
                "weights": [{"candidate_id": "topcap-val-winner", "weight": 1.0}],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(MODULE, "EQUAL_WEIGHT_BASELINE_PATH", equal_weight_path)
    monkeypatch.setattr(MODULE, "ONE_SHOT_BASELINE_PATH", one_shot_path)

    result = _call_write(tmp_path, grouped_rows)

    assert Path(result["json_path"]).exists()
    assert Path(result["md_path"]).exists()
    assert Path(result["manifest_json_path"]).exists()
    assert Path(result["manifest_md_path"]).exists()
    payload = json.loads(Path(result["json_path"]).read_text(encoding="utf-8"))
    optimizer_bundle = dict(payload.get("optimizer_bundle") or {})
    assert len(list(optimizer_bundle.get("candidates") or [])) == 1
    assert optimizer_bundle.get("candidates") == payload.get("candidates")
    assert optimizer_bundle.get("selected_team") == payload.get("selected_team")
    baseline_support = dict(payload.get("baseline_support") or {})
    assert (
        dict(baseline_support.get("equal_weight_diagnostic") or {}).get("normalization_method")
        == "rebuilt_from_component_streams_default"
    )
    assert (
        dict(baseline_support.get("one_shot_optimized") or {}).get("normalization_method")
        == "normalized_existing_portfolio_streams"
    )
    manifest_payload = json.loads(Path(result["manifest_json_path"]).read_text(encoding="utf-8"))
    assert manifest_payload["split_windows"]["oos_start"] == MODULE.OOS_START
    assert manifest_payload["pairspread_exclusion"]["excluded"] is True


def test_write_portfolio_exact_window_freeze_falls_back_when_equal_weight_streams_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    grouped_rows = {
        "topcap_tsmom_1h": [
            _row(
                candidate_id="topcap-val-winner",
                name="topcap_tsmom_1h_candidate_a",
                family="cross_sectional",
                strategy_class="TopCapTimeSeriesMomentumStrategy",
                timeframe="1h",
                symbols=["BTC/USDT", "ETH/USDT", "BNB/USDT"],
                train_sharpe=1.0,
                train_deflated_sharpe=0.5,
                train_return=0.02,
                val_sharpe=1.4,
                val_deflated_sharpe=0.7,
                val_return=0.04,
                val_pbo=0.08,
                val_turnover=0.09,
                oos_sharpe=0.4,
                oos_return=0.02,
            )
        ]
    }
    equal_weight_path = tmp_path / "committee_portfolio_followup_latest.json"
    equal_weight_path.write_text(
        json.dumps(
            {
                "generated_at": "2026-03-14T00:00:00Z",
                "selection": [
                    {
                        "name": "equal_weight_component",
                        "strategy_class": "CompositeTrendStrategy",
                        "timeframe": "30m",
                        "symbols": ["BTC/USDT"],
                    }
                ],
                "metrics": {"oos": {"return": 0.03}},
            }
        ),
        encoding="utf-8",
    )
    one_shot_path = tmp_path / "portfolio_optimization_latest.json"
    one_shot_path.write_text(
        json.dumps(
            {
                "generated_at": "2026-03-14T00:00:00Z",
                "portfolio_metrics": {"oos": {"total_return": 0.025}},
                "weights": [{"candidate_id": "topcap-val-winner", "weight": 1.0}],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(MODULE, "EQUAL_WEIGHT_BASELINE_PATH", equal_weight_path)
    monkeypatch.setattr(MODULE, "ONE_SHOT_BASELINE_PATH", one_shot_path)

    result = _call_write(tmp_path, grouped_rows)
    payload = json.loads(Path(result["json_path"]).read_text(encoding="utf-8"))
    baseline_support = dict(payload.get("baseline_support") or {})
    equal_weight = dict(baseline_support.get("equal_weight_diagnostic") or {})

    assert equal_weight.get("normalization_method") == "summary_only_missing_component_streams"
    assert equal_weight.get("combined_streams") == {}
    assert equal_weight.get("split_windows") == {}


def test_incumbent_anchor_mode_keeps_anchor_when_challenger_misses_guard(tmp_path: Path):
    grouped_rows = _base_incumbent_grouped_rows()
    anchor = _row(
        candidate_id="composite-anchor",
        name="composite_trend_30m_anchor",
        family="trend",
        strategy_class="CompositeTrendStrategy",
        timeframe="30m",
        symbols=["BTC/USDT"],
        train_sharpe=1.1,
        train_deflated_sharpe=0.7,
        train_return=0.03,
        val_sharpe=1.4,
        val_deflated_sharpe=0.75,
        val_return=0.032,
        val_pbo=0.10,
        val_turnover=0.08,
        oos_sharpe=0.3,
        oos_return=0.01,
    )
    challenger = _row(
        candidate_id="composite-high-pbo",
        name="composite_trend_30m_high_pbo",
        family="trend",
        strategy_class="CompositeTrendStrategy",
        timeframe="30m",
        symbols=["BTC/USDT"],
        train_sharpe=1.2,
        train_deflated_sharpe=0.72,
        train_return=0.031,
        val_sharpe=1.65,
        val_deflated_sharpe=0.70,
        val_return=0.042,
        val_pbo=0.16,
        val_turnover=0.05,
        oos_sharpe=0.35,
        oos_return=0.012,
    )
    grouped_rows["composite_trend_30m"] = [anchor, challenger]
    incumbent_bundle_path, incumbent_bundle_payload = _write_incumbent_bundle(
        tmp_path,
        [
            anchor,
            grouped_rows["topcap_tsmom_1h"][0],
            grouped_rows["regime_breakout_1h"][0],
        ],
    )

    payload = _call_build(
        tmp_path,
        grouped_rows,
        extra_kwargs={
            "selection_mode": "incumbent_anchor_rolling_gate",
            "incumbent_bundle_path": incumbent_bundle_path,
            "incumbent_bundle_payload": incumbent_bundle_payload,
            "anchor_bundle_path": incumbent_bundle_path,
            "anchor_bundle_payload": incumbent_bundle_payload,
        },
    )

    selected_ids = _selected_ids(payload)
    assert "composite-anchor" in selected_ids
    assert "composite-high-pbo" not in selected_ids
    selection_basis = _selection_basis(payload).lower()
    assert "incumbent" in selection_basis and "anchor" in selection_basis


def test_incumbent_anchor_mode_promotes_local_challenger_when_thresholds_met(tmp_path: Path):
    grouped_rows = _base_incumbent_grouped_rows()
    anchor = _row(
        candidate_id="composite-anchor",
        name="composite_trend_30m_anchor",
        family="trend",
        strategy_class="CompositeTrendStrategy",
        timeframe="30m",
        symbols=["BTC/USDT"],
        train_sharpe=0.10,
        train_deflated_sharpe=0.05,
        train_return=0.01,
        val_sharpe=1.50,
        val_deflated_sharpe=0.70,
        val_return=0.03,
        val_pbo=0.10,
        val_turnover=0.10,
        oos_sharpe=0.25,
        oos_return=0.01,
    )
    challenger = _row(
        candidate_id="composite-train-strong",
        name="composite_trend_30m_train_strong",
        family="trend",
        strategy_class="CompositeTrendStrategy",
        timeframe="30m",
        symbols=["BTC/USDT"],
        train_sharpe=2.0,
        train_deflated_sharpe=1.0,
        train_return=0.03,
        val_sharpe=1.45,
        val_deflated_sharpe=0.69,
        val_return=0.029,
        val_pbo=0.08,
        val_turnover=0.08,
        oos_sharpe=0.30,
        oos_return=0.011,
    )
    grouped_rows["composite_trend_30m"] = [anchor, challenger]
    incumbent_bundle_path, incumbent_bundle_payload = _write_incumbent_bundle(
        tmp_path,
        [
            anchor,
            grouped_rows["topcap_tsmom_1h"][0],
            grouped_rows["regime_breakout_1h"][0],
        ],
    )

    payload = _call_build(
        tmp_path,
        grouped_rows,
        extra_kwargs={
            "selection_mode": "incumbent_anchor_rolling_gate",
            "incumbent_bundle_path": incumbent_bundle_path,
            "incumbent_bundle_payload": incumbent_bundle_payload,
            "anchor_bundle_path": incumbent_bundle_path,
            "anchor_bundle_payload": incumbent_bundle_payload,
        },
    )

    selected_ids = _selected_ids(payload)
    assert "composite-train-strong" in selected_ids
    assert "composite-anchor" not in selected_ids
    selection_basis = _selection_basis(payload).lower()
    assert "incumbent" in selection_basis and "anchor" in selection_basis


def test_incumbent_anchor_mode_admits_fourth_sleeve_only_when_gate_passes(tmp_path: Path):
    grouped_rows = _base_incumbent_grouped_rows()
    rolling_row = _row(
        candidate_id="rolling-train-val-survivor",
        name="rolling_breakout_30m_guarded_ls_64_0.002",
        family="trend",
        strategy_class="RollingBreakoutStrategy",
        timeframe="30m",
        symbols=["BTC/USDT"],
        train_sharpe=0.4,
        train_deflated_sharpe=0.1,
        train_return=-0.04,
        val_sharpe=1.3,
        val_deflated_sharpe=0.2,
        val_return=0.02,
        val_pbo=0.20,
        val_turnover=0.12,
        oos_sharpe=-1.5,
        oos_return=-0.03,
    )
    grouped_rows["rolling_breakout_30m"] = [rolling_row]
    incumbent_bundle_path, incumbent_bundle_payload = _write_incumbent_bundle(
        tmp_path,
        [
            grouped_rows["composite_trend_30m"][0],
            grouped_rows["topcap_tsmom_1h"][0],
            grouped_rows["regime_breakout_1h"][0],
        ],
    )
    rolling_gate_payload = {
        "selection_basis": "train_val_only",
        "survives_train_val": True,
        "survives": True,
        "recommended_action": "activate_conditionally",
        "selected_rule": {
            "rule_id": "btc_above_ma192",
            "survives_train_val": True,
        },
        "gated_candidate_row": {
            "candidate_id": "rolling-train-val-survivor",
            "metadata": {
                "activation_rule_id": "btc_above_ma192",
                "activation_rule_survives": True,
            },
            "return_streams": {
                "train": _stream(1_735_689_600_000.0, [0.0, 0.001]),
                "val": _stream(1_767_225_600_000.0, [0.002, 0.003]),
                "oos": _stream(1_769_904_000_000.0, [0.0, 0.0]),
            },
        },
    }

    payload = _call_build(
        tmp_path,
        grouped_rows,
        extra_kwargs={
            "selection_mode": "incumbent_anchor_rolling_gate",
            "incumbent_bundle_path": incumbent_bundle_path,
            "incumbent_bundle_payload": incumbent_bundle_payload,
            "rolling_gate_payload": rolling_gate_payload,
            "rolling_breakout_gate_payload": rolling_gate_payload,
            "gate_payload": rolling_gate_payload,
        },
    )

    selected = _selected_rows(payload)
    selected_ids = _selected_ids(payload)
    assert len(selected) == 4
    assert "rolling-train-val-survivor" in selected_ids
    selection_basis = _selection_basis(payload).lower()
    assert "incumbent" in selection_basis and "anchor" in selection_basis
    rolling_row = next(
        row for row in selected if str(row.get("candidate_id")) == "rolling-train-val-survivor"
    )
    metadata = dict(rolling_row.get("metadata") or {})
    assert metadata.get("activation_rule_survives") is True
    assert _rolling_admission_blocked(payload) is False


def test_incumbent_anchor_mode_blocks_fourth_sleeve_when_gate_fails(tmp_path: Path):
    grouped_rows = _base_incumbent_grouped_rows()
    grouped_rows["rolling_breakout_30m"] = [
        _row(
            candidate_id="rolling-blocked",
            name="rolling_breakout_30m_guarded_ls_64_0.002",
            family="trend",
            strategy_class="RollingBreakoutStrategy",
            timeframe="30m",
            symbols=["BTC/USDT"],
            train_sharpe=0.5,
            train_deflated_sharpe=0.2,
            train_return=-0.02,
            val_sharpe=1.4,
            val_deflated_sharpe=0.3,
            val_return=0.018,
            val_pbo=0.18,
            val_turnover=0.11,
            oos_sharpe=-2.0,
            oos_return=-0.04,
        )
    ]
    incumbent_bundle_path, incumbent_bundle_payload = _write_incumbent_bundle(
        tmp_path,
        [
            grouped_rows["composite_trend_30m"][0],
            grouped_rows["topcap_tsmom_1h"][0],
            grouped_rows["regime_breakout_1h"][0],
        ],
    )
    rolling_gate_payload = {
        "selection_basis": "train_val_only",
        "survives_train_val": False,
        "survives": False,
        "recommended_action": "do_not_activate",
        "selected_rule": {
            "rule_id": "btc_above_ma192",
            "survives_train_val": False,
        },
    }

    payload = _call_build(
        tmp_path,
        grouped_rows,
        extra_kwargs={
            "selection_mode": "incumbent_anchor_rolling_gate",
            "incumbent_bundle_path": incumbent_bundle_path,
            "incumbent_bundle_payload": incumbent_bundle_payload,
            "rolling_gate_payload": rolling_gate_payload,
            "rolling_breakout_gate_payload": rolling_gate_payload,
            "gate_payload": rolling_gate_payload,
        },
    )

    assert _selected_ids(payload) == {"composite-anchor", "topcap-anchor", "regime-anchor"}
    assert _rolling_admission_blocked(payload) is True
