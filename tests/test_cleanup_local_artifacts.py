import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.dev.cleanup_local_artifacts import discover_candidates, run_cleanup


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x", encoding="utf-8")


def _repo(tmp_path: Path) -> Path:
    (tmp_path / ".git").mkdir()
    return tmp_path


def _candidate_paths(root: Path, **kwargs) -> set[str]:
    return {candidate.relative_path for candidate in discover_candidates(root, **kwargs)}


def test_discovery_preserves_research_notes_data_and_evidence_roots(tmp_path: Path) -> None:
    root = _repo(tmp_path)

    preserved = [
        "data/market_parquet/BTCUSDT.parquet",
        "var/reports/profit_moonshot_20260501/result.json",
        "var/logs/reboot_validation_20260416/01_run.log",
        "docs/research_note/research_note.md",
        ".omx/notepad.md",
        ".omx/context/profit-moonshot.md",
        ".omx/plans/ralplan-profit-moonshot.md",
        ".omx/project-memory.json",
        ".env",
        ".venv/lib/python/site-packages/pkg/__pycache__/kept.pyc",
        "best_optimized_parameters/params.json",
        ".codegraph/codegraph.db",
    ]
    for rel_path in preserved:
        _touch(root / rel_path)

    removable = [
        ".pytest_cache/v/cache/nodeids",
        ".ruff_cache/content",
        "logs/local.log",
        "dist/pkg.whl",
        "lumina_quant.egg-info/PKG-INFO",
        "src/lumina_quant.egg-info/PKG-INFO",
        "src/lumina_quant/__pycache__/module.cpython-311.pyc",
        "scripts/__pycache__/tool.cpython-311.pyc",
        "apps/dashboard_web/.next/build-id",
        "apps/dashboard_web/node_modules/pkg/index.js",
        "apps/dashboard_web/tsconfig.tsbuildinfo",
        "native/rust_metrics/target/debug/libx.rlib",
        "native/rust_rawfirst/target/debug/libx.rlib",
        ".omx/cache/cache.json",
        ".omx/tmp/tmp.txt",
        ".omx/logs/run.log",
    ]
    for rel_path in removable:
        _touch(root / rel_path)

    candidates = _candidate_paths(root)

    assert ".pytest_cache" in candidates
    assert ".ruff_cache" in candidates
    assert "logs" in candidates
    assert "dist" in candidates
    assert "lumina_quant.egg-info" in candidates
    assert "src/lumina_quant.egg-info" in candidates
    assert "src/lumina_quant/__pycache__" in candidates
    assert "scripts/__pycache__" in candidates
    assert "apps/dashboard_web/.next" in candidates
    assert "apps/dashboard_web/node_modules" in candidates
    assert "apps/dashboard_web/tsconfig.tsbuildinfo" in candidates
    assert "native/rust_metrics/target" not in candidates
    assert "native/rust_rawfirst/target" not in candidates
    assert ".omx/cache" in candidates
    assert ".omx/tmp" in candidates
    assert ".omx/logs" in candidates

    for rel_path in preserved:
        assert rel_path not in candidates
    assert ".venv" not in candidates
    assert ".codegraph/codegraph.db" not in candidates


def test_apply_removes_only_generated_local_artifacts(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    safe_delete = [
        "logs/local.log",
        "src/lumina_quant/__pycache__/module.cpython-311.pyc",
        ".omx/logs/run.log",
    ]
    preserved = [
        "data/market_parquet/BTCUSDT.parquet",
        "var/reports/profit_moonshot_20260501/result.json",
        "var/logs/reboot_validation_20260416/01_run.log",
        "docs/research_note/research_note.md",
        ".omx/notepad.md",
        ".env",
    ]
    for rel_path in safe_delete + preserved:
        _touch(root / rel_path)

    summary = run_cleanup(root, apply=True)

    assert summary.applied is True
    assert summary.removed_count == 3
    assert not (root / "logs").exists()
    assert not (root / "src/lumina_quant/__pycache__").exists()
    assert not (root / ".omx/logs").exists()
    for rel_path in preserved:
        assert (root / rel_path).exists()


def test_optional_cleanup_flags_are_explicit(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    _touch(root / ".venv/lib/python/site-packages/pkg.py")
    _touch(root / "native/rust_metrics/target/release/liblumina_metrics.so")
    _touch(root / "native/rust_rawfirst/target/release/liblumina_rawfirst.so")

    default_candidates = _candidate_paths(root)
    explicit_candidates = _candidate_paths(
        root,
        include_venv=True,
        include_native_targets=True,
    )

    assert ".venv" not in default_candidates
    assert "native/rust_metrics/target" not in default_candidates
    assert "native/rust_rawfirst/target" not in default_candidates
    assert ".venv" in explicit_candidates
    assert "native/rust_metrics/target" in explicit_candidates
    assert "native/rust_rawfirst/target" in explicit_candidates
