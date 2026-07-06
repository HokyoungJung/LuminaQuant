"""Deterministic tests for the alpha scoreboard enablement runner.

Exercises the one-command "load rows -> persist versioned + _latest json/md"
path on a small SYNTHETIC row set (no large data, no backtests, no network).
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

_RUNNER_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "research" / "run_alpha_scoreboard.py"
)
_spec = importlib.util.spec_from_file_location("run_alpha_scoreboard", _RUNNER_PATH)
assert _spec is not None and _spec.loader is not None
runner = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(runner)

_PINNED_VERSION = "20260706T000000Z"


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


def _synthetic_rows_ranked():
    # ``return`` is a reserved keyword, so the return level is set explicitly
    # after construction rather than via a kwarg.
    rows = [
        _row("worst", sharpe=0.5, mdd=0.50, cagr=0.02, calmar=0.1),
        _row("best", sharpe=3.0, mdd=0.05, cagr=0.50, calmar=3.0),
        _row("mid"),
    ]
    rows[0]["metrics"]["return"] = 0.02
    rows[1]["metrics"]["return"] = 0.50
    return rows


# --------------------------------------------------------------------------- #
# load_rows
# --------------------------------------------------------------------------- #
def test_load_rows_from_list_file(tmp_path):
    path = tmp_path / "rows.json"
    path.write_text(json.dumps(_synthetic_rows_ranked()), encoding="utf-8")
    rows = runner.load_rows(path)
    assert {row["id"] for row in rows} == {"worst", "best", "mid"}


def test_load_rows_from_wrapped_object(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"rows": _synthetic_rows_ranked()}), encoding="utf-8")
    rows = runner.load_rows(path)
    assert len(rows) == 3


def test_load_rows_from_directory_concatenates_and_ignores_non_rows(tmp_path):
    src = tmp_path / "rows_dir"
    src.mkdir()
    (src / "a.json").write_text(json.dumps([_row("a")]), encoding="utf-8")
    (src / "b.json").write_text(json.dumps({"candidates": [_row("b")]}), encoding="utf-8")
    # A config blob with no rows must contribute nothing.
    (src / "config.json").write_text(json.dumps({"threshold": 0.3}), encoding="utf-8")
    rows = runner.load_rows(src)
    assert sorted(row["id"] for row in rows) == ["a", "b"]


# --------------------------------------------------------------------------- #
# run_scoreboard: persistence + ranking
# --------------------------------------------------------------------------- #
def test_run_scoreboard_persists_versioned_and_latest(tmp_path):
    result = runner.run_scoreboard(
        _synthetic_rows_ranked(), output_dir=tmp_path, version=_PINNED_VERSION
    )
    paths = result["paths"]
    for key in ("versioned_json", "versioned_md", "latest_json", "latest_md"):
        assert paths[key].exists(), key
    assert paths["versioned_json"].name == f"alpha_scoreboard_{_PINNED_VERSION}.json"
    assert paths["latest_json"].name == "alpha_scoreboard_latest.json"
    # _latest mirrors the versioned artifact byte-for-byte.
    assert paths["latest_json"].read_bytes() == paths["versioned_json"].read_bytes()
    assert paths["latest_md"].read_bytes() == paths["versioned_md"].read_bytes()


def test_run_scoreboard_ranking_is_sane(tmp_path):
    result = runner.run_scoreboard(
        _synthetic_rows_ranked(), output_dir=tmp_path, version=_PINNED_VERSION
    )
    persisted = json.loads(result["paths"]["latest_json"].read_text(encoding="utf-8"))
    assert persisted["version"] == _PINNED_VERSION
    order = [item["id"] for item in persisted["composite"]]
    assert order == ["best", "mid", "worst"]
    assert persisted["composite"][0]["rank"] == 1
    assert persisted["eligible_count"] == 3
    md = result["paths"]["latest_md"].read_text(encoding="utf-8")
    assert "## Composite ranking" in md
    assert f"_version: {_PINNED_VERSION}_" in md


def test_run_scoreboard_is_deterministic_for_pinned_version(tmp_path):
    first = runner.run_scoreboard(
        _synthetic_rows_ranked(), output_dir=tmp_path / "a", version=_PINNED_VERSION
    )
    second = runner.run_scoreboard(
        list(reversed(_synthetic_rows_ranked())),
        output_dir=tmp_path / "b",
        version=_PINNED_VERSION,
    )
    assert first["paths"]["latest_json"].read_bytes() == second["paths"]["latest_json"].read_bytes()
    assert first["paths"]["latest_md"].read_bytes() == second["paths"]["latest_md"].read_bytes()


def test_run_scoreboard_default_version_stamped(tmp_path):
    result = runner.run_scoreboard(_synthetic_rows_ranked(), output_dir=tmp_path)
    stamp = result["version"]
    assert stamp.endswith("Z") and "T" in stamp and len(stamp) == 16
    assert result["paths"]["versioned_json"].name == f"alpha_scoreboard_{stamp}.json"


def test_run_scoreboard_gate_excludes_liquidated(tmp_path):
    rows = _synthetic_rows_ranked()
    rows.append({**_row("blown"), "liquidation_count": 3})
    result = runner.run_scoreboard(rows, output_dir=tmp_path, version=_PINNED_VERSION)
    persisted = result["payload"]
    assert persisted["eligible_count"] == 3
    assert any(item["id"] == "blown" for item in persisted["excluded"])


# --------------------------------------------------------------------------- #
# main / CLI
# --------------------------------------------------------------------------- #
def test_main_persists_leaderboard(tmp_path):
    input_path = tmp_path / "rows.json"
    input_path.write_text(json.dumps(_synthetic_rows_ranked()), encoding="utf-8")
    out_dir = tmp_path / "out"
    rc = runner.main(
        [
            "--input",
            str(input_path),
            "--output-dir",
            str(out_dir),
            "--version",
            _PINNED_VERSION,
        ]
    )
    assert rc == 0
    latest = out_dir / "alpha_scoreboard_latest.json"
    assert latest.exists()
    persisted = json.loads(latest.read_text(encoding="utf-8"))
    assert [item["id"] for item in persisted["composite"]] == ["best", "mid", "worst"]
    assert persisted["source"] == str(input_path)


def test_main_errors_on_empty_input(tmp_path):
    empty = tmp_path / "empty.json"
    empty.write_text(json.dumps([]), encoding="utf-8")
    try:
        runner.main(["--input", str(empty), "--output-dir", str(tmp_path / "out")])
    except SystemExit as exc:
        assert exc.code and "no result rows" in str(exc.code)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected SystemExit on empty input")


def test_cli_subprocess_roundtrip(tmp_path):
    input_path = tmp_path / "rows.json"
    input_path.write_text(json.dumps(_synthetic_rows_ranked()), encoding="utf-8")
    out_dir = tmp_path / "out"
    proc = subprocess.run(
        [
            sys.executable,
            str(_RUNNER_PATH),
            "--input",
            str(input_path),
            "--output-dir",
            str(out_dir),
            "--version",
            _PINNED_VERSION,
        ],
        capture_output=True,
        text=True,
        check=True,
        cwd=str(Path(__file__).resolve().parents[2]),
    )
    assert "3/3 eligible" in proc.stdout
    assert (out_dir / f"alpha_scoreboard_{_PINNED_VERSION}.json").exists()
    assert (out_dir / "alpha_scoreboard_latest.md").exists()
