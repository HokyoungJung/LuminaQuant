"""F1 pre-registered momentum + low-beta composite allocation cell (eq-flow v5).

Deterministic, ASCII-only, no ``random`` module. Covers:
  (a) the committed cell JSON shape (3 sleeves, 3 distinct families, provenance,
      allocator block);
  (b) a synthetic end-to-end run of the cell's shape through
      ``allocate_quality_gated`` (valid, positive, stable weights);
  (c) the ``build_quality_gated_allocation --validate-cell-spec`` mode on the
      committed spec (OK) and a mutilated 2-family spec (fails).
"""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lumina_quant.portfolio.quality_gated_allocation import allocate_quality_gated
from scripts.research import build_quality_gated_allocation as cli

CELL_PATH = (
    REPO_ROOT
    / "configs"
    / "research"
    / "allocation_cells"
    / "eqflow_momentum_lowbeta_composite_cell.json"
)


def _load_cell() -> dict:
    return json.loads(CELL_PATH.read_text(encoding="utf-8"))


def test_committed_cell_declares_three_distinct_family_membership() -> None:
    cell = _load_cell()
    assert cell["cell_id"] == "eqflow_momentum_lowbeta_composite_m2"
    assert cell["created"] == "2026-07-10"
    sleeves = cell["sleeves"]
    assert len(sleeves) == 3
    families = [sleeve["family"] for sleeve in sleeves.values()]
    assert len(set(families)) == 3
    assert set(families) == {"momentum", "tradfi_carry", "pair_carry"}
    # Exactly one momentum sleeve paired with two low-beta carry sleeves.
    assert families.count("momentum") == 1
    for sleeve in sleeves.values():
        assert sleeve["returns"] is None
        assert sleeve["turnover"] is None
        provenance = sleeve["returns_source"]
        assert isinstance(provenance, dict) and provenance
    allocator = cell["allocator"]
    assert allocator == {
        "method": "erc",
        "min_families": 3,
        "min_sleeves": 2,
        "turnover_penalty_lambda": 0.0,
    }
    assert isinstance(cell["source_artifacts"], list) and cell["source_artifacts"]


def test_synthetic_end_to_end_cell_shape_allocates_positive_stable_weights() -> None:
    cell = _load_cell()
    sleeve_ids = list(cell["sleeves"])
    families = {sid: cell["sleeves"][sid]["family"] for sid in sleeve_ids}
    allocator = cell["allocator"]

    # Clone the cell's 3-sleeve shape with deterministic synthetic streams; the
    # third sleeve is mildly negatively correlated with the first.
    n = 240
    t = np.linspace(0.0, 8.0 * np.pi, n)
    base = 0.004
    streams = {
        sleeve_ids[0]: (base + 0.003 * np.sin(t)),
        sleeve_ids[1]: (base + 0.003 * np.cos(t)),
        sleeve_ids[2]: (base - 0.0018 * np.sin(t) + 0.0012 * np.cos(2.0 * t)),
    }
    corr = float(np.corrcoef(streams[sleeve_ids[0]], streams[sleeve_ids[2]])[0, 1])
    assert corr < 0.0  # genuinely (mildly) negatively correlated
    sleeve_returns = {sid: series.tolist() for sid, series in streams.items()}
    turnovers = dict.fromkeys(sleeve_ids, 0.02)

    kwargs = dict(
        method=allocator["method"],
        min_sleeves=allocator["min_sleeves"],
        min_families=allocator["min_families"],
        turnover_penalty_lambda=allocator["turnover_penalty_lambda"],
        families=families,
    )
    weights = allocate_quality_gated(sleeve_returns, turnovers, **kwargs)

    assert len(weights) == 3
    assert abs(sum(weights.values()) - 1.0) < 1e-9
    assert all(weight > 0.0 for weight in weights.values())
    # Deterministic: identical inputs -> byte-identical weights across calls.
    again = allocate_quality_gated(sleeve_returns, turnovers, **kwargs)
    assert weights == again


def test_validate_cell_spec_accepts_committed_and_rejects_two_families() -> None:
    cell = _load_cell()
    ok, message = cli.validate_cell_spec(cell)
    assert ok is True
    assert "3 distinct families" in message

    # Mutilate: collapse the pair_carry sleeve into momentum -> 2 distinct families.
    mutilated = copy.deepcopy(cell)
    mutilated["sleeves"]["state_vwap_pair_carry"]["family"] = "momentum"
    bad_ok, bad_message = cli.validate_cell_spec(mutilated)
    assert bad_ok is False
    assert "distinct families" in bad_message


def test_validate_cell_spec_cli_exit_codes(tmp_path: Path) -> None:
    script = str(REPO_ROOT / "scripts" / "research" / "build_quality_gated_allocation.py")
    ok = subprocess.run(
        [sys.executable, script, "--validate-cell-spec", str(CELL_PATH)],
        capture_output=True,
        text=True,
    )
    assert ok.returncode == 0, ok.stderr
    assert "OK:" in ok.stdout

    mutilated = _load_cell()
    del mutilated["sleeves"]["state_vwap_pair_carry"]  # 2 sleeves, 2 families
    bad_path = tmp_path / "mutilated_cell.json"
    bad_path.write_text(json.dumps(mutilated), encoding="utf-8")
    bad = subprocess.run(
        [sys.executable, script, "--validate-cell-spec", str(bad_path)],
        capture_output=True,
        text=True,
    )
    assert bad.returncode == 1, bad.stdout
