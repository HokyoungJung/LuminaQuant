from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


def _load_phase1_module():
    root = Path(__file__).resolve().parent.parent
    module_path = root / "scripts" / "run_phase1_research.py"
    spec = importlib.util.spec_from_file_location("phase1_research_script", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load run_phase1_research module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MODULE = _load_phase1_module()
_build_parser = MODULE._build_parser
_build_sweep_command = MODULE._build_sweep_command
_normalize_symbols = MODULE._normalize_symbols


class TestPhase1ResearchScript(unittest.TestCase):
    def test_normalize_symbols_dedupes_and_formats(self):
        symbols = _normalize_symbols([" btcusdt ", "ETH/USDT", "eth-usdt", "", "SOL_USDT"])
        self.assertEqual(symbols, ["BTC/USDT", "ETH/USDT", "SOL/USDT"])

    def test_build_sweep_command_forwards_topcap_symbols(self):
        parser = _build_parser()
        args = parser.parse_args(["--skip-sync", "--dry-run", "--timeframes", "15m", "1h"])
        cmd = _build_sweep_command(args, ["BTC/USDT", "ETH/USDT"])

        self.assertIn("scripts/timeframe_sweep_oos.py", cmd)
        self.assertIn("--topcap-symbols", cmd)
        self.assertIn("BTC/USDT", cmd)
        self.assertIn("ETH/USDT", cmd)
        self.assertIn("--timeframes", cmd)
        idx = cmd.index("--timeframes")
        self.assertEqual(cmd[idx + 1 : idx + 3], ["15m", "1h"])


if __name__ == "__main__":
    unittest.main()
