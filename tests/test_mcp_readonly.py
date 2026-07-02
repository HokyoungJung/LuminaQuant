"""Read-only guarantees for the MCP bridge and the factor-insights surface.

These tests never require the ``mcp`` runtime extra or Node: they introspect the
plain-data tool specs, assert no trading verb is exposed, exercise the read-only
factor-insights payload, and re-affirm the dashboard v2 contract is unchanged.
"""

from __future__ import annotations

import json

import pytest

from lumina_quant.dashboard import mcp_server
from lumina_quant.dashboard.factor_insights_service import (
    build_factor_insights_payload,
    load_factor_insights_payload,
)

# Any tool name a client might mistake for a mutating action.
_MUTATING_TOOL_NAMES = (
    "place_order",
    "submit_order",
    "cancel_order",
    "modify_order",
    "create_order",
    "trade",
    "execute_trade",
    "route_order",
    "buy",
    "sell",
    "liquidate_position",
)


def test_read_only_tool_catalog_has_no_mutating_verbs() -> None:
    names = mcp_server.list_tool_names()
    assert names, "expected at least one read-only tool"
    for name in names:
        lowered = name.lower()
        for forbidden in mcp_server.FORBIDDEN_TOOL_SUBSTRINGS:
            assert forbidden not in lowered, f"tool {name!r} exposes forbidden verb {forbidden!r}"


def test_expected_read_only_tools_present() -> None:
    names = set(mcp_server.list_tool_names())
    assert {
        "get_backtest_overview",
        "get_factor_insights",
        "get_alpha_evidence",
        "list_dashboard_routes",
    } <= names


@pytest.mark.parametrize("name", _MUTATING_TOOL_NAMES)
def test_assert_read_only_rejects_mutating_names(name: str) -> None:
    with pytest.raises(mcp_server.MCPReadOnlyViolationError):
        mcp_server.assert_read_only([name])


def test_assert_read_only_accepts_registered_names() -> None:
    # Must not raise for the actually-registered read-only tools.
    mcp_server.assert_read_only(mcp_server.list_tool_names())


def test_describe_tools_matches_specs() -> None:
    described = mcp_server.describe_tools()
    assert [d["name"] for d in described] == list(mcp_server.list_tool_names())
    for entry in described:
        assert entry["description"].strip()


def test_main_list_tools_smoke_no_mcp_runtime(capsys) -> None:
    exit_code = mcp_server.main(["--list-tools"])
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["read_only"] is True
    assert payload["real_money_execution_enabled"] is False
    catalog_names = [tool["name"] for tool in payload["tools"]]
    assert catalog_names == list(mcp_server.list_tool_names())
    for name in catalog_names:
        for forbidden in mcp_server.FORBIDDEN_TOOL_SUBSTRINGS:
            assert forbidden not in name.lower()


def test_factor_insights_payload_empty_is_wellformed() -> None:
    payload = build_factor_insights_payload()
    assert payload["artifact_kind"] == "dashboard_factor_insights_payload"
    assert payload["status"] == "empty"
    assert payload["real_money_execution_enabled"] is False
    assert payload["ic_heatmap"]["factors"] == []
    assert payload["ic_heatmap"]["cells"] == []
    assert payload["candidate_queue"] == []


def test_factor_insights_heatmap_and_queue_are_deterministic() -> None:
    factor_ic = {
        "artifact_kind": "batch_factor_ic",
        "max_decay_lag": 3,
        "factors": {
            "momentum": {
                "factor": "momentum",
                "n_periods": 20,
                "ic_mean": 0.08,
                "ic_ir": 1.5,
                "ic_positive_ratio": 0.7,
                "t_stat": 3.1,
                "turnover_mean": 0.4,
                "quantile_spread_mean": 0.02,
                "rank_autocorr": [0.9, 0.7, 0.5],
            },
            "reversal": {
                "factor": "reversal",
                "n_periods": 20,
                "ic_mean": -0.02,
                "ic_ir": -0.3,
                "ic_positive_ratio": 0.4,
                "t_stat": -0.9,
                "turnover_mean": 0.8,
                "quantile_spread_mean": -0.01,
                "rank_autocorr": [0.2, 0.1],
            },
        },
    }
    candidates = [
        {"candidate_id": "c2", "strategy": "Momo", "sharpe": 1.2, "status": "pending"},
        {"candidate_id": "c1", "strategy": "Rev", "sharpe": 2.5, "status": "review"},
        {"candidate_id": "c3", "strategy": "Carry", "status": "pending"},
    ]

    first = build_factor_insights_payload(factor_ic=factor_ic, candidate_queue=candidates)
    second = build_factor_insights_payload(factor_ic=factor_ic, candidate_queue=candidates)

    # Determinism: identical inputs -> byte-identical JSON (as_of aside).
    first_body = {k: v for k, v in first.items() if k != "as_of"}
    second_body = {k: v for k, v in second.items() if k != "as_of"}
    assert json.dumps(first_body, sort_keys=True) == json.dumps(second_body, sort_keys=True)

    heatmap = first["ic_heatmap"]
    assert heatmap["factors"] == ["momentum", "reversal"]  # sorted
    assert heatmap["lags"] == ["lag_1", "lag_2", "lag_3"]
    # reversal only had 2 lags -> padded with None to the max lag count.
    assert heatmap["cells"][1] == [0.2, 0.1, None]
    assert heatmap["cells"][0] == [0.9, 0.7, 0.5]

    # Factor ranking ordered by IC-IR descending.
    assert [row["factor"] for row in first["factor_ranking"]] == ["momentum", "reversal"]
    assert first["summary"]["top_factor"] == "momentum"

    # Candidate queue ordered by score (sharpe) descending, id tiebreak.
    queue_ids = [row["candidate_id"] for row in first["candidate_queue"]]
    assert queue_ids == ["c1", "c2", "c3"]  # 2.5, 1.2, then None-score last
    assert first["candidate_queue"][-1]["score"] is None


def test_load_factor_insights_from_artifacts(tmp_path) -> None:
    factor_path = tmp_path / "factor_ic.json"
    factor_path.write_text(
        json.dumps(
            {
                "max_decay_lag": 2,
                "factors": {
                    "value": {
                        "factor": "value",
                        "n_periods": 10,
                        "ic_mean": 0.05,
                        "ic_ir": 0.9,
                        "rank_autocorr": [0.6, 0.3],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    candidate_path = tmp_path / "candidates.json"
    candidate_path.write_text(
        json.dumps([{"candidate_id": "cand", "strategy": "ValueTilt", "sharpe": 1.1}]),
        encoding="utf-8",
    )

    payload = load_factor_insights_payload(
        factor_ic_path=factor_path,
        candidate_queue_path=candidate_path,
    )
    assert payload["status"] == "ok"
    assert payload["ic_heatmap"]["factors"] == ["value"]
    assert payload["candidate_queue"][0]["candidate_id"] == "cand"


def test_dashboard_v2_contract_route_count_unchanged() -> None:
    # The MCP/factor-insights work must not alter the v2 route contract.
    from lumina_quant.dashboard.bridge import build_dashboard_bridge_contract_v2

    contract = build_dashboard_bridge_contract_v2().to_dict()
    assert contract["contract_version"] == 2
    assert len(contract["routes"]) == 12
    factor_routes = [r for r in contract["routes"] if "factor-insights" in r["route"]]
    assert factor_routes == []  # additive route is NOT enumerated in the v2 contract
