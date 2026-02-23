from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


def _load_module():
    root = Path(__file__).resolve().parent.parent
    module_path = root / "scripts" / "run_two_book_research.py"
    spec = importlib.util.spec_from_file_location("two_book_research_script", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load run_two_book_research module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MODULE = _load_module()
_build_parser = MODULE._build_parser
_normalize_risk_share = MODULE._normalize_risk_share
_select_two_book_candidates = MODULE._select_two_book_candidates


class TestTwoBookResearchScript(unittest.TestCase):
    def test_normalize_risk_share_defaults_when_non_positive(self):
        alpha, trend = _normalize_risk_share(0.0, 0.0)
        self.assertEqual(alpha, 0.8)
        self.assertEqual(trend, 0.2)

    def test_normalize_risk_share_scales_positive_values(self):
        alpha, trend = _normalize_risk_share(8.0, 2.0)
        self.assertAlmostEqual(alpha, 0.8, places=6)
        self.assertAlmostEqual(trend, 0.2, places=6)

    def test_select_two_book_candidates_by_prefix_and_hurdle(self):
        candidates = [
            {
                "name": "pair_a",
                "hurdle_fields": {"oos": {"pass": True, "score": 1.2, "excess_return": 0.1}},
            },
            {
                "name": "pair_b",
                "hurdle_fields": {"oos": {"pass": True, "score": 1.8, "excess_return": 0.2}},
            },
            {
                "name": "topcap_tsmom_x",
                "hurdle_fields": {"oos": {"pass": True, "score": 1.4, "excess_return": 0.12}},
            },
        ]
        alpha, trend = _select_two_book_candidates(
            candidates=candidates,
            hurdle_key="oos",
            alpha_prefixes=["pair_"],
            trend_prefixes=["topcap_tsmom"],
        )

        self.assertIsNotNone(alpha)
        self.assertIsNotNone(trend)
        self.assertEqual(alpha["name"], "pair_b")
        self.assertEqual(trend["name"], "topcap_tsmom_x")

    def test_parser_accepts_sweep_report_override(self):
        parser = _build_parser()
        args = parser.parse_args(["--dry-run", "--sweep-report", "reports/x.json"])
        self.assertEqual(args.sweep_report, "reports/x.json")


if __name__ == "__main__":
    unittest.main()
