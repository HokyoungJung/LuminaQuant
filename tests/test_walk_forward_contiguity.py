"""Walk-forward fold-boundary contiguity guards.

Locks the no-data-leak contiguity contract of ``build_walk_forward_splits``:
within every fold the train/val/test windows abut with no gap and no overlap
(``val_start == train_end`` and ``test_start == val_end``).  A silent gap or
overlap at a boundary would let evaluation windows leak across each other, so
these are regression guards rather than feature tests.
"""

import unittest
from datetime import datetime

from lumina_quant.optimization.walkers import add_months, build_walk_forward_splits


class TestWalkForwardContiguity(unittest.TestCase):
    def _assert_fold_contiguous(self, s):
        # Exact intra-fold contiguity: each window starts where the prior ends.
        self.assertEqual(s["val_start"], s["train_end"])
        self.assertEqual(s["test_start"], s["val_end"])

    def _assert_ordered(self, s):
        # No window may run backwards (start must not exceed end).
        self.assertLessEqual(s["train_start"], s["train_end"])
        self.assertLessEqual(s["val_start"], s["val_end"])
        self.assertLessEqual(s["test_start"], s["test_end"])

    def test_boundary_contiguity_default_params(self):
        base = datetime(2024, 1, 1)
        splits = build_walk_forward_splits(
            base_start=base,
            folds=3,
            train_months=12,
            val_months=6,
            test_months=6,
            step_months=6,
        )
        self.assertEqual(len(splits), 3)
        for s in splits:
            self._assert_fold_contiguous(s)
            self._assert_ordered(s)
            # No gap and no overlap at boundaries means the fold covers
            # [train_start, test_end] with no interior holes.
            self.assertLess(s["train_start"], s["train_end"])
            self.assertLess(s["val_start"], s["val_end"])
            self.assertLess(s["test_start"], s["test_end"])

    def test_single_fold(self):
        base = datetime(2024, 1, 1)
        splits = build_walk_forward_splits(base_start=base, folds=1)
        self.assertEqual(len(splits), 1)
        s = splits[0]
        self.assertEqual(s["fold"], 1)
        self.assertEqual(s["train_start"], base)
        self._assert_fold_contiguous(s)
        self._assert_ordered(s)
        # Every window has positive length under default params.
        self.assertLess(s["train_start"], s["train_end"])
        self.assertLess(s["val_start"], s["val_end"])
        self.assertLess(s["test_start"], s["test_end"])

    def test_zero_validation_window_collapses_but_stays_contiguous(self):
        # A zero-length validation window is supported: val collapses to a point
        # (val_start == val_end) yet contiguity is preserved on both boundaries
        # so no data leaks past the collapsed window.
        base = datetime(2024, 1, 1)
        splits = build_walk_forward_splits(base_start=base, folds=2, val_months=0)
        self.assertEqual(len(splits), 2)
        for s in splits:
            self.assertEqual(s["val_start"], s["val_end"])  # zero length
            self._assert_fold_contiguous(s)
            self._assert_ordered(s)
            # Train and test still have positive length.
            self.assertLess(s["train_start"], s["train_end"])
            self.assertLess(s["test_start"], s["test_end"])

    def test_window_lengths_match_month_params(self):
        # Boundary contiguity implies each window spans exactly its month param.
        base = datetime(2024, 1, 1)
        train_months, val_months, test_months = 9, 3, 4
        splits = build_walk_forward_splits(
            base_start=base,
            folds=2,
            train_months=train_months,
            val_months=val_months,
            test_months=test_months,
            step_months=5,
        )
        for s in splits:
            self.assertEqual(s["train_end"], add_months(s["train_start"], train_months))
            self.assertEqual(s["val_end"], add_months(s["val_start"], val_months))
            self.assertEqual(s["test_end"], add_months(s["test_start"], test_months))
            # Contiguity must still hold for arbitrary month params.
            self._assert_fold_contiguous(s)

    def test_fold_step_advances_by_step_months(self):
        # Cross-fold rolling: each fold's anchor advances by exactly step_months.
        # (Folds may overlap by design when span > step; this pins the step, not
        # a no-overlap claim across folds.)
        base = datetime(2024, 1, 1)
        step_months = 6
        splits = build_walk_forward_splits(base_start=base, folds=3, step_months=step_months)
        for i in range(1, len(splits)):
            self.assertEqual(
                splits[i]["train_start"],
                add_months(splits[i - 1]["train_start"], step_months),
            )

    def test_zero_folds_returns_empty(self):
        splits = build_walk_forward_splits(base_start=datetime(2024, 1, 1), folds=0)
        self.assertEqual(splits, [])


if __name__ == "__main__":
    unittest.main()
