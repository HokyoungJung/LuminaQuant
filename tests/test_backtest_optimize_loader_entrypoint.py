from __future__ import annotations

import re
from pathlib import Path


def _storage_import_block(source: str) -> str:
    match = re.search(
        r"from\s+lumina_quant\.storage\.parquet\s+import\s*\((.*?)\)",
        source,
        flags=re.DOTALL,
    )
    return match.group(1) if match else ""


def test_backtest_cli_imports_owner_loader_entrypoint():
    source = Path("src/lumina_quant/cli/backtest.py").read_text(encoding="utf-8")
    assert "from lumina_quant.market_data import (" in source
    assert "load_data_dict_from_parquet" in source
    storage_block = _storage_import_block(source)
    assert "load_data_dict_from_parquet" not in storage_block


def test_optimize_cli_imports_owner_loader_entrypoint():
    source = Path("src/lumina_quant/cli/optimize.py").read_text(encoding="utf-8")
    assert "from lumina_quant.market_data import (" in source
    assert "load_data_dict_from_parquet" in source
    storage_block = _storage_import_block(source)
    assert "load_data_dict_from_parquet" not in storage_block
