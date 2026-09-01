from __future__ import annotations

from pathlib import Path


def test_pyproject_no_longer_requires_ccxt() -> None:
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    text = pyproject.read_text(encoding="utf-8").lower()
    assert "ccxt" not in text


def test_ccxt_exchange_module_removed() -> None:
    exchange_module = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "lumina_quant"
        / "exchanges"
        / "ccxt_exchange.py"
    )
    assert not exchange_module.exists()
