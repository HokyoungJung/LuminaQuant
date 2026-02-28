from __future__ import annotations

from pathlib import Path


def _read(path: str) -> str:
    return (Path(__file__).resolve().parents[1] / path).read_text(encoding="utf-8")


def test_readme_quickstart_paths_are_repo_consistent():
    text = _read("README.md")
    assert "git clone https://github.com/HokyoungJung/LuminaQuant.git" in text
    assert "cd LuminaQuant" in text
    assert "# cd Quants-agent" in text
    assert "cd lumina-quant" not in text


def test_readme_kr_quickstart_paths_are_repo_consistent():
    text = _read("README_KR.md")
    assert "git clone https://github.com/HokyoungJung/LuminaQuant.git" in text
    assert "cd LuminaQuant" in text
    assert "# cd Quants-agent" in text
    assert "cd lumina-quant" not in text
