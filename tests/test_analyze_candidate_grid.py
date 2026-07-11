from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from scripts.research.analyze_candidate_grid import (
    DEAD,
    INSUFFICIENT,
    NEAR_MISS,
    PASS,
    build_analysis,
    build_identity_key,
    build_kill_list,
    build_shortlist,
    classify_row,
    cross_cost_robustness,
    extract_metrics,
    iter_rows,
    load_grid,
    render_markdown,
    render_tsv,
)


def _row(
    candidate_id: str,
    *,
    strategy_class: str = "momentum",
    family: str = "trend",
    timeframe: str = "1h",
    oos_sharpe: float | None = 1.0,
    dsr: float | None = 0.5,
    oos_return: float | None = 0.2,
    net_edge: float | None = None,
    factor_ic: float | None = None,
    cost_bps: float | None = 10.0,
    hard_reject: bool = False,
    hard_reject_reasons: dict[str, Any] | None = None,
    params: dict[str, Any] | None = None,
    symbols: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    oos: dict[str, Any] = {}
    if oos_sharpe is not None:
        oos["sharpe"] = oos_sharpe
    if dsr is not None:
        oos["deflated_sharpe"] = dsr
    if oos_return is not None:
        oos["total_return"] = oos_return
    if net_edge is not None:
        oos["net_edge"] = net_edge
    if factor_ic is not None:
        oos["factor_ic"] = factor_ic
    meta = dict(metadata or {})
    if cost_bps is not None:
        meta.setdefault("round_trip_cost_bps", cost_bps)
    return {
        "candidate_id": candidate_id,
        "name": candidate_id,
        "strategy_class": strategy_class,
        "family": family,
        "strategy_timeframe": timeframe,
        "timeframe": timeframe,
        "params": params or {"lookback": 20},
        "symbols": symbols or ["BTCUSDT"],
        "oos": oos,
        "hard_reject": hard_reject,
        "hard_reject_reasons": hard_reject_reasons or {},
        "metadata": meta,
    }


# --- classification: the four classes -------------------------------------- #
def test_pass_row_clears_every_clause() -> None:
    detail = classify_row(_row("pass1", dsr=0.4, oos_return=0.3, factor_ic=0.1))
    assert detail["classification"] == PASS
    assert detail["incomplete"] is False
    assert detail["failing_clauses"] == []


def test_pass_uses_oos_return_as_net_edge_proxy_when_net_edge_absent() -> None:
    detail = classify_row(_row("pass2", oos_return=0.15, net_edge=None))
    assert detail["classification"] == PASS
    assert detail["net_edge_source"] == "oos_return_proxy"
    assert math.isclose(detail["net_edge"], 0.15)


def test_explicit_net_edge_wins_over_return_proxy() -> None:
    metrics = extract_metrics(_row("e", net_edge=2.5, oos_return=-0.1))
    assert metrics["net_edge_source"] == "explicit"
    assert math.isclose(metrics["net_edge"], 2.5)


def test_near_miss_single_failing_clause_records_margin() -> None:
    detail = classify_row(_row("nm1", dsr=-0.05, oos_return=0.2))
    assert detail["classification"] == NEAR_MISS
    assert detail["failing_clauses"] == ["dsr"]
    blocker = detail["near_miss_blocker"]
    assert blocker["clause"] == "dsr"
    assert blocker["kind"] == "fail"
    assert math.isclose(blocker["margin"], -0.05)


def test_near_miss_on_factor_ic_only() -> None:
    detail = classify_row(_row("nm2", dsr=0.3, oos_return=0.2, factor_ic=-0.02))
    assert detail["classification"] == NEAR_MISS
    assert detail["near_miss_blocker"]["clause"] == "factor_ic"


def test_dead_two_failing_clauses() -> None:
    detail = classify_row(_row("d1", dsr=-0.1, oos_return=-0.2))
    assert detail["classification"] == DEAD
    assert set(detail["failing_clauses"]) == {"net_edge", "dsr"}


def test_dead_on_hard_reject_gate() -> None:
    detail = classify_row(
        _row(
            "d2",
            dsr=0.4,
            oos_return=0.3,
            hard_reject=True,
            hard_reject_reasons={"oos_sharpe": -0.3},
        )
    )
    assert detail["classification"] == DEAD
    assert detail["hard_reject_reasons"] == {"oos_sharpe": -0.3}


def test_insufficient_records_missing_symbols() -> None:
    detail = classify_row(
        _row(
            "i1",
            hard_reject=True,
            hard_reject_reasons={"insufficient_data": True},
            metadata={"missing_symbols": ["ETHUSDT", "SOLUSDT"]},
        )
    )
    assert detail["classification"] == INSUFFICIENT
    assert detail["missing_symbols"] == ["ETHUSDT", "SOLUSDT"]
    assert detail["incomplete"] is True


def test_insufficient_precedes_hard_reject() -> None:
    detail = classify_row(
        _row(
            "i2",
            hard_reject=True,
            hard_reject_reasons={"insufficient_data": True, "oos_sharpe": -1.0},
        )
    )
    assert detail["classification"] == INSUFFICIENT


# --- incomplete / malformed tolerance -------------------------------------- #
def test_missing_mandatory_metric_marks_incomplete_near_miss() -> None:
    detail = classify_row(_row("mm", dsr=None, oos_return=0.2))
    assert detail["classification"] == NEAR_MISS
    assert detail["incomplete"] is True
    assert detail["missing_clauses"] == ["dsr"]
    assert detail["near_miss_blocker"]["kind"] == "missing"


def test_malformed_rows_never_raise() -> None:
    for malformed in (
        {},
        {"oos": "not-a-dict"},
        {"oos": {"sharpe": "abc"}},
        {"metadata": None, "hard_reject_reasons": None},
    ):
        detail = classify_row(malformed)
        assert detail["classification"] in {PASS, NEAR_MISS, DEAD, INSUFFICIENT}


def test_string_and_bool_metric_coercion() -> None:
    metrics = extract_metrics(_row("s", dsr=None, oos_return=None, metadata={}, net_edge=None))
    assert metrics["dsr"] is None
    detail = classify_row(
        {"oos": {"sharpe": True, "deflated_sharpe": "0.5", "total_return": "0.1"}}
    )
    assert detail["oos_sharpe"] is None  # bool rejected
    assert math.isclose(detail["dsr"], 0.5)


# --- aggregation ----------------------------------------------------------- #
def test_aggregates_counts_best_worst_median() -> None:
    rows = [
        _row("a", oos_sharpe=2.0, dsr=0.9, oos_return=0.3),
        _row("b", oos_sharpe=0.5, dsr=0.2, oos_return=0.1),
        _row("c", oos_sharpe=-0.4, dsr=-0.3, oos_return=-0.2),
    ]
    payload = build_analysis(rows)
    by_class = payload["aggregates"]["by_strategy_class"]["momentum"]
    assert by_class["total"] == 3
    assert by_class["best"]["candidate_id"] == "a"
    assert by_class["worst"]["candidate_id"] == "c"
    assert math.isclose(by_class["median_oos_sharpe"], 0.5)
    assert by_class["counts"][PASS] == 2
    assert by_class["counts"][DEAD] == 1


# --- cross-cost robustness ------------------------------------------------- #
def test_cross_cost_sign_stability_flag() -> None:
    stable = [
        _row("st", cost_bps=10.0, oos_sharpe=1.2),
        _row("st", cost_bps=20.0, oos_sharpe=0.8),
        _row("st", cost_bps=30.0, oos_sharpe=0.4),
    ]
    unstable = [
        _row("un", strategy_class="reversal", cost_bps=10.0, oos_sharpe=0.5),
        _row("un", strategy_class="reversal", cost_bps=30.0, oos_sharpe=-0.6),
    ]
    reports = cross_cost_robustness([classify_row(r) for r in stable + unstable])
    by_class = {r["strategy_class"]: r for r in reports}
    assert by_class["momentum"]["sign_stable"] is True
    assert by_class["momentum"]["monotone_decay"] is True
    assert by_class["momentum"]["cost_cells"] == ["10bps", "20bps", "30bps"]
    assert by_class["reversal"]["sign_stable"] is False


def test_single_cost_identity_omitted_from_robustness() -> None:
    reports = cross_cost_robustness([classify_row(_row("only", cost_bps=10.0))])
    assert reports == []


# --- shortlist ordering + kill-list ---------------------------------------- #
def test_shortlist_ranks_by_dsr_then_sharpe_and_drops_insufficient() -> None:
    rows = [
        _row("low", dsr=0.1, oos_sharpe=0.2),
        _row("high", dsr=0.9, oos_sharpe=1.5),
        _row("mid", dsr=0.5, oos_sharpe=0.8),
        _row("insuff", hard_reject=True, hard_reject_reasons={"insufficient_data": True}),
    ]
    details = [classify_row(r) for r in rows]
    shortlist = build_shortlist(details, cross_cost_robustness(details), top_n=10)
    ids = [entry["candidate_id"] for entry in shortlist]
    assert ids == ["high", "mid", "low"]
    assert shortlist[0]["rank"] == 1
    assert "insuff" not in ids


def test_shortlist_top_n_truncates() -> None:
    rows = [_row(f"c{i}", dsr=float(i) / 10.0, oos_sharpe=float(i)) for i in range(5)]
    details = [classify_row(r) for r in rows]
    shortlist = build_shortlist(details, cross_cost_robustness(details), top_n=2)
    assert [e["candidate_id"] for e in shortlist] == ["c4", "c3"]


def test_kill_list_flags_uniformly_dead_family() -> None:
    rows = [
        _row("dead_a", family="doomed", dsr=-0.2, oos_return=-0.3),
        _row("dead_b", family="doomed", dsr=-0.1, oos_return=-0.4),
        _row("alive", family="healthy", dsr=0.5, oos_return=0.2),
    ]
    kill = build_kill_list([classify_row(r) for r in rows])
    families = {entry["family"]: entry["dead_row_count"] for entry in kill}
    assert families == {"doomed": 2}


# --- envelope ingest + rendering + determinism ----------------------------- #
def test_iter_rows_accepts_multiple_layouts() -> None:
    envelope_candidates = {"candidates": [_row("x"), _row("y")]}
    envelope_rows = {"rows": [_row("z")]}
    combined = {"reports": [{"candidates": [_row("nested")]}]}
    assert len(list(iter_rows(envelope_candidates))) == 2
    assert len(list(iter_rows(envelope_rows))) == 1
    assert len(list(iter_rows(combined))) == 1


def test_directory_ingest_tags_cost_from_filename(tmp_path: Path) -> None:
    cost10 = tmp_path / "grid_10bps.json"
    cost30 = tmp_path / "grid_30bps.json"
    cost10.write_text(json.dumps({"candidates": [_row("f", cost_bps=None)]}), "utf-8")
    cost30.write_text(json.dumps({"candidates": [_row("f", cost_bps=None)]}), "utf-8")
    rows, source = load_grid(tmp_path)
    payload = build_analysis(rows, source=source)
    cost_cells = sorted(payload["aggregates"]["by_cost_cell"].keys())
    assert cost_cells == ["10bps", "30bps"]


def test_build_identity_key_is_cost_independent() -> None:
    a = build_identity_key(_row("a", cost_bps=10.0))
    b = build_identity_key(_row("b", cost_bps=30.0))
    assert a == b


def test_render_outputs_and_determinism() -> None:
    rows = [
        _row("a", oos_sharpe=1.5, dsr=0.7),
        _row("b", oos_sharpe=-0.2, dsr=-0.1, oos_return=-0.1),
    ]
    first = build_analysis(rows)
    second = build_analysis(rows)
    first.pop("generated_at")
    second.pop("generated_at")
    assert first == second
    md = render_markdown(build_analysis(rows))
    assert "# Candidate Grid Analysis" in md
    assert "Second-generation shortlist" in md
    tsv = render_tsv(build_analysis(rows))
    assert tsv.splitlines()[0].startswith("classification\tincomplete")
    assert len(tsv.splitlines()) == 3  # header + two rows


def test_directory_scan_ignores_prior_analyzer_output(tmp_path: Path) -> None:
    grid = tmp_path / "grid_10bps.json"
    grid.write_text(json.dumps({"candidates": [_row("real")]}), "utf-8")
    # A prior run wrote its analysis JSON into the same directory.
    prior = build_analysis([_row("real")])
    (tmp_path / "analysis.json").write_text(json.dumps(prior), "utf-8")
    rows, _ = load_grid(tmp_path)
    assert [r.get("candidate_id") for r in rows] == ["real"]
