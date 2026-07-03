"""Deterministic tests for the post-backtest alpha scoreboard."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from lumina_quant.portfolio.optimizer_core import metrics as core_metrics
from lumina_quant.research.alpha_scoreboard import (
    METRIC_DIRECTIONS,
    build_scoreboard,
    composite_ranking,
    rank_by_metric,
    render_scoreboard_markdown,
    resolve_row,
)


def _row(candidate_id, **metric_overrides):
    metrics = {
        "return": 0.10,
        "cagr": 0.10,
        "sharpe": 1.0,
        "sortino": 1.2,
        "calmar": 0.8,
        "mdd": 0.20,
        "turnover": 0.05,
        "win_rate": 0.5,
    }
    metrics.update(metric_overrides)
    return {
        "id": candidate_id,
        "family": "trend",
        "strategy_class": "X",
        "metrics": metrics,
        "trade_count": 25,
        "liquidation_count": 0,
        "bars": 1000,
    }


# --------------------------------------------------------------------------- #
# resolve_row
# --------------------------------------------------------------------------- #
def test_precomputed_metrics_take_precedence_over_derived():
    returns = [0.01, -0.02, 0.03]
    row = resolve_row({"id": "a", "returns": returns, "metrics": {"sharpe": 9.9}})
    assert row.metrics["sharpe"] == 9.9
    # Derived fields that were NOT supplied still come from the returns series.
    expected = core_metrics(np.asarray(returns, dtype=np.float64))
    assert abs(row.metrics["sortino"] - float(expected["sortino"])) < 1e-12


def test_derived_metrics_match_optimizer_core():
    returns = [0.004, -0.002, 0.006, -0.001, 0.003] * 40
    row = resolve_row({"id": "a", "returns": returns})
    expected = core_metrics(np.asarray(returns, dtype=np.float64))
    for key_row, key_core in (
        ("cagr", "cagr"),
        ("sharpe", "sharpe"),
        ("sortino", "sortino"),
        ("calmar", "calmar"),
        ("mdd", "max_drawdown"),
    ):
        assert abs(row.metrics[key_row] - float(expected[key_core])) < 1e-12
    total = float(np.expm1(np.log1p(np.asarray(returns)).sum()))
    assert abs(row.metrics["return"] - total) < 1e-12


def test_trade_frequency_derived_from_bars():
    row = resolve_row({"id": "a", "trade_count": 50, "bars": 1000})
    assert abs(row.metrics["trade_frequency"] - 0.05) < 1e-12


# --------------------------------------------------------------------------- #
# per-metric ranking
# --------------------------------------------------------------------------- #
def test_rank_direction_higher_and_lower_better():
    rows = [
        resolve_row(_row("a", sharpe=2.0, mdd=0.30)),
        resolve_row(_row("b", sharpe=1.0, mdd=0.10)),
        resolve_row(_row("c", sharpe=3.0, mdd=0.20)),
    ]
    sharpe_table = rank_by_metric(rows, "sharpe")
    assert [item["id"] for item in sharpe_table] == ["c", "a", "b"]
    mdd_table = rank_by_metric(rows, "mdd")
    assert [item["id"] for item in mdd_table] == ["b", "c", "a"]


def test_rank_ties_are_dense_and_deterministic():
    rows = [
        resolve_row(_row("b", sharpe=2.0)),
        resolve_row(_row("a", sharpe=2.0)),
        resolve_row(_row("c", sharpe=1.0)),
    ]
    table = rank_by_metric(rows, "sharpe")
    assert [(item["id"], item["rank"]) for item in table] == [("a", 1), ("b", 1), ("c", 2)]


def test_missing_metric_ranks_last():
    complete = resolve_row(_row("a"))
    incomplete = resolve_row(
        {"id": "b", "metrics": {"sharpe": 5.0}, "trade_count": 10, "bars": 100}
    )
    table = rank_by_metric([complete, incomplete], "calmar")
    assert table[-1]["id"] == "b" and table[-1]["value"] is None
    assert table[-1]["rank"] > table[0]["rank"]


# --------------------------------------------------------------------------- #
# composite
# --------------------------------------------------------------------------- #
def test_composite_dominant_candidate_ranks_first():
    rows = [
        resolve_row(_row("worst", sharpe=0.5, mdd=0.5, cagr=0.02, calmar=0.1)),
        resolve_row(_row("best", sharpe=3.0, mdd=0.05, cagr=0.5, calmar=3.0)),
        resolve_row(_row("mid")),
    ]
    ranking = composite_ranking(rows)
    assert [item["id"] for item in ranking] == ["best", "mid", "worst"]
    assert ranking[0]["composite_score"] == 1.0  # best on every composite metric
    assert ranking[0]["rank"] == 1


def test_composite_weights_change_the_order():
    rows = [
        resolve_row(_row("high_sharpe", sharpe=3.0, mdd=0.40)),
        resolve_row(_row("low_mdd", sharpe=1.0, mdd=0.05)),
    ]
    sharpe_led = composite_ranking(rows, weights={"sharpe": 1.0})
    assert sharpe_led[0]["id"] == "high_sharpe"
    mdd_led = composite_ranking(rows, weights={"mdd": 1.0})
    assert mdd_led[0]["id"] == "low_mdd"


def test_composite_hand_checked_two_metric_case():
    rows = [
        resolve_row(_row("a", sharpe=2.0, mdd=0.30)),
        resolve_row(_row("b", sharpe=1.0, mdd=0.10)),
    ]
    ranking = composite_ranking(rows, weights={"sharpe": 1.0, "mdd": 1.0})
    # a: sharpe pct 1.0, mdd pct 0.0 -> 0.5 ; b: 0.0, 1.0 -> 0.5 ; tie -> id order.
    assert [item["id"] for item in ranking] == ["a", "b"]
    assert all(abs(item["composite_score"] - 0.5) < 1e-12 for item in ranking)


# --------------------------------------------------------------------------- #
# gates + payload + markdown
# --------------------------------------------------------------------------- #
def test_default_gates_exclude_liquidated_and_tradeless():
    rows = [
        _row("clean"),
        {**_row("liquidated"), "liquidation_count": 2},
        {**_row("no_trades"), "trade_count": 0},
    ]
    payload = build_scoreboard(rows)
    assert payload["eligible_count"] == 1
    excluded = {item["id"]: item["reasons"] for item in payload["excluded"]}
    assert "liquidated" in excluded and "max_liquidations" in excluded["liquidated"][0]
    assert "no_trades" in excluded and "min_trades" in excluded["no_trades"][0]


def test_gate_overrides_and_max_mdd():
    rows = [_row("deep", mdd=0.60), _row("shallow", mdd=0.10)]
    payload = build_scoreboard(rows, gates={"max_mdd": 0.30, "max_liquidations": None})
    assert payload["eligible_count"] == 1
    assert payload["excluded"][0]["id"] == "deep"


def test_payload_deterministic_and_markdown_sections():
    rows = [_row("b", sharpe=1.5), _row("a", sharpe=2.5), _row("c", sharpe=0.5)]
    p1 = build_scoreboard(rows)
    p2 = build_scoreboard(list(reversed(rows)))
    assert json.dumps(p1, sort_keys=True) == json.dumps(p2, sort_keys=True)
    md = render_scoreboard_markdown(p1)
    assert "## Composite ranking" in md
    assert "## Top by sharpe" in md
    assert "## Top by mdd" in md
    assert "`a`" in md


def test_cli_roundtrip(tmp_path):
    rows = [_row("a", sharpe=2.0), _row("b", sharpe=1.0)]
    input_path = tmp_path / "rows.json"
    input_path.write_text(json.dumps(rows), encoding="utf-8")
    out_json = tmp_path / "scoreboard.json"
    out_md = tmp_path / "scoreboard.md"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/research/build_alpha_scoreboard.py",
            "--input",
            str(input_path),
            "--output-json",
            str(out_json),
            "--output-md",
            str(out_md),
            "--weights",
            '{"sharpe": 1.0}',
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "2/2 eligible" in result.stdout
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["composite"][0]["id"] == "a"
    first = out_json.read_bytes()
    subprocess.run(
        [
            sys.executable,
            "scripts/research/build_alpha_scoreboard.py",
            "--input",
            str(input_path),
            "--output-json",
            str(out_json),
            "--weights",
            '{"sharpe": 1.0}',
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert out_json.read_bytes() == first  # byte-deterministic


def test_all_direction_metrics_have_known_direction():
    assert set(METRIC_DIRECTIONS) >= {
        "return",
        "cagr",
        "sharpe",
        "sortino",
        "calmar",
        "mdd",
        "turnover",
        "liquidation_count",
        "trade_frequency",
    }
