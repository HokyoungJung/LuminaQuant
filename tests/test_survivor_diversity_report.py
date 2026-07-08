"""Survivor-diversity diagnostic (data-free synthetic verdicts).

Loads ``scripts/research/survivor_diversity_report.py`` by path (the data-PC handoff
script) and asserts its verdict on SYNTHETIC returns:

* 2 highly-correlated survivor streams -> verdict BOUNDED (one low-corr cluster);
* 4 uncorrelated survivor streams -> verdict REAL (four low-corr clusters, low
  crash correlation);
* survivors that de-correlate normally but dump together in benchmark drawdowns ->
  verdict BOUNDED via the crash-period-correlation axis;
* the CSV/JSON loader round-trips into the same verdict.

No market data is required.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "research" / "survivor_diversity_report.py"
)
SPEC = importlib.util.spec_from_file_location("survivor_diversity_report", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Failed to load survivor_diversity_report module")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_two_highly_correlated_streams_are_bounded():
    rng = np.random.default_rng(0)
    base = rng.standard_normal(300) * 0.01
    returns = {"a": base, "b": base * 1.02 + rng.standard_normal(300) * 0.0005}
    report = MODULE.analyze_survivor_diversity(returns)
    assert report["survivor_count"] == 2
    assert report["low_corr_cluster_count"] == 1
    assert report["verdict"] == "BOUNDED"
    assert "insufficient_low_correlation_clusters" in report["verdict_reasons"]


def test_four_uncorrelated_streams_are_real():
    rng = np.random.default_rng(0)
    returns = {key: rng.standard_normal(300) * 0.01 for key in ("a", "b", "c", "d")}
    report = MODULE.analyze_survivor_diversity(returns)
    assert report["survivor_count"] == 4
    assert report["low_corr_cluster_count"] == 4
    assert report["crash_period_correlation"] < report["crash_corr_ceiling"]
    assert report["verdict"] == "REAL"
    assert report["verdict_reasons"] == []


def test_crash_correlation_forces_bounded():
    rng = np.random.default_rng(5)
    n = 300
    streams = {key: rng.standard_normal(n) * 0.01 for key in ("a", "b", "c", "d")}
    benchmark = rng.standard_normal(n) * 0.005
    benchmark[100:140] = -0.03
    common_shock = np.zeros(n)
    common_shock[100:140] = -0.04
    streams = {key: (series * 0.2 + common_shock) for key, series in streams.items()}
    report = MODULE.analyze_survivor_diversity(streams, benchmark=benchmark)
    assert report["crash_bar_count"] >= 8
    assert report["crash_period_correlation"] >= report["crash_corr_ceiling"]
    assert report["verdict"] == "BOUNDED"
    assert "high_crash_period_correlation" in report["verdict_reasons"]


def test_json_loader_round_trips_into_real_verdict(tmp_path):
    rng = np.random.default_rng(0)
    returns = {key: list(rng.standard_normal(300) * 0.01) for key in ("a", "b", "c", "d")}
    path = tmp_path / "survivors.json"
    path.write_text(json.dumps({"returns": returns}))
    loaded, benchmark = MODULE._load_returns(path, benchmark_col="benchmark")
    assert set(loaded) == set(returns)
    assert benchmark is None
    report = MODULE.analyze_survivor_diversity(loaded, benchmark=benchmark)
    assert report["verdict"] == "REAL"


def test_csv_loader_extracts_benchmark_column(tmp_path):
    rng = np.random.default_rng(1)
    n = 64
    rows = []
    for i in range(n):
        rows.append(
            {
                "a": float(rng.standard_normal() * 0.01),
                "b": float(rng.standard_normal() * 0.01),
                "benchmark": float(rng.standard_normal() * 0.005),
            }
        )
    header = "a,b,benchmark\n"
    body = "\n".join(f"{r['a']},{r['b']},{r['benchmark']}" for r in rows)
    path = tmp_path / "survivors.csv"
    path.write_text(header + body + "\n")
    loaded, benchmark = MODULE._load_returns(path, benchmark_col="benchmark")
    assert set(loaded) == {"a", "b"}
    assert benchmark is not None and len(benchmark) == n
