from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


def _load_module():
    root = Path(__file__).resolve().parent.parent
    module_path = root / "scripts" / "run_strategy_factory_pipeline.py"
    spec = importlib.util.spec_from_file_location("strategy_factory_pipeline_script", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load run_strategy_factory_pipeline module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MODULE = _load_module()


class TestStrategyFactoryPipelineScript(unittest.TestCase):
    def test_normalize_symbols(self):
        symbols = MODULE._normalize_symbols([" btcusdt ", "ETH-USDT", "eth/usdt", "XAUUSDT", ""])
        self.assertEqual(symbols, ["BTC/USDT", "ETH/USDT", "XAU/USDT"])

    def test_normalize_timeframes(self):
        timeframes = MODULE._normalize_timeframes(["1M", "1m", "5m", " 5M ", ""])
        self.assertEqual(timeframes, ["1m", "5m"])

    def test_parser_accepts_core_arguments(self):
        parser = MODULE._build_parser()
        args = parser.parse_args(["--db-path", "data/lq_market.sqlite3", "--mode", "oos"])
        self.assertEqual(args.db_path, "data/lq_market.sqlite3")
        self.assertEqual(args.mode, "oos")


if __name__ == "__main__":
    unittest.main()
