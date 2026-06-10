"""Tolerance-based golden comparator (Phase 4.5).

Reads ``baseline/golden/*`` and asserts new outputs are within
``validation.golden_rtol`` (default ``1e-8``, config-tunable).

Divergence procedure
--------------------
When :func:`compare_to_golden` fails, the caller should:

1. Write ``docs/divergences/<artifact>.md`` classifying the cause
   (precision-improvement vs bug).
2. Improvements → update the golden with the new value.
3. Regressions → fix the production code; do NOT accept a new golden.

Phase 4 correctness gate: walk-forward output is compared against
``baseline/golden/walk_forward_results_warmup.json`` (Variant B) at rtol
``1e-8``.  Divergence from Variant A folds 1 & 3 (the ``-999`` sentinels)
is a documented improvement — see ``docs/divergences/walk_forward_no_sentinel.md``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


_REPO_ROOT = Path(__file__).resolve().parents[3]
_GOLDEN_DIR = _REPO_ROOT / "baseline" / "golden"


class GoldenMismatch(AssertionError):
    """Raised when a numeric value exceeds ``rtol`` vs the golden baseline.

    Attributes:
    ----------
    key:
        Dot-separated path to the failing field.
    actual:
        Observed value.
    expected:
        Golden value.
    rtol:
        Tolerance that was applied.
    rel_diff:
        ``abs(actual - expected) / max(abs(expected), 1e-12)``.
    """

    def __init__(
        self,
        key: str,
        actual: float,
        expected: float,
        rtol: float,
        rel_diff: float,
    ) -> None:
        super().__init__(
            f"Golden mismatch at '{key}': "
            f"actual={actual!r} expected={expected!r} "
            f"rel_diff={rel_diff:.3e} > rtol={rtol:.3e}. "
            f"Write docs/divergences/<artifact>.md if this is an improvement."
        )
        self.key = key
        self.actual = actual
        self.expected = expected
        self.rtol = rtol
        self.rel_diff = rel_diff


def _walk_numeric(
    obj: Any,
    prefix: str,
) -> list[tuple[str, float]]:
    """Yield (dotted-path, float) pairs for all numeric leaves in *obj*."""
    results: list[tuple[str, float]] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            results.extend(_walk_numeric(v, f"{prefix}.{k}" if prefix else k))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            results.extend(_walk_numeric(v, f"{prefix}[{i}]"))
    elif isinstance(obj, float):
        results.append((prefix, obj))
    elif isinstance(obj, int) and not isinstance(obj, bool):
        results.append((prefix, float(obj)))
    return results


def compare_to_golden(
    actual: dict,
    golden_path: Path | str,
    *,
    rtol: float | None = None,
    label: str = "",
    skip_sentinel: float = -999.0,
) -> None:
    """Assert all numeric fields in *actual* match the golden within *rtol*.

    Parameters
    ----------
    actual:
        Result dict produced by the new code path.
    golden_path:
        Path to the golden JSON file.  Relative paths are resolved from the
        repo root (``baseline/golden/``).
    rtol:
        Relative tolerance.  Defaults to ``validation.golden_rtol`` from
        the default runtime config (``1e-8``).
    label:
        Optional human-readable label included in ``GoldenMismatch`` messages.
    skip_sentinel:
        Numeric values equal to this in the *golden* are skipped — they
        represent degenerate ``-999`` entries from Variant A that Phase 4
        replaces with real metrics (documented improvement).

    Raises:
    ------
    GoldenMismatch
        On the first field that exceeds *rtol*.
    FileNotFoundError
        If the golden file does not exist.
    """
    if rtol is None:
        try:
            from lumina_quant.configuration import get_default_runtime_config

            rt = get_default_runtime_config()
            rtol = float(rt.validation.golden_rtol)
        except Exception:
            rtol = 1e-8

    golden_path = Path(golden_path)
    if not golden_path.is_absolute():
        golden_path = _GOLDEN_DIR / golden_path

    with golden_path.open() as fh:
        golden = json.load(fh)

    actual_pairs = {k: v for k, v in _walk_numeric(actual, "")}
    golden_pairs = {k: v for k, v in _walk_numeric(golden, "")}

    mismatches: list[str] = []
    for key, expected in golden_pairs.items():
        # Skip degenerate sentinel values (Variant A -999 folds — documented improvement).
        if math.isclose(expected, skip_sentinel, rel_tol=1e-9):
            continue
        if key not in actual_pairs:
            mismatches.append(f"  missing key: '{key}' (expected {expected!r})")
            continue
        observed = actual_pairs[key]
        denom = max(abs(float(expected)), 1e-12)
        rel_diff = abs(float(observed) - float(expected)) / denom
        if rel_diff > float(rtol):
            tag = f" [{label}]" if label else ""
            raise GoldenMismatch(
                key=f"{key}{tag}",
                actual=float(observed),
                expected=float(expected),
                rtol=float(rtol),
                rel_diff=rel_diff,
            )

    if mismatches:
        raise AssertionError(
            f"Golden comparator found {len(mismatches)} missing key(s):\n" + "\n".join(mismatches)
        )


def load_golden(golden_path: Path | str) -> dict:
    """Load and return the raw golden JSON as a dict."""
    golden_path = Path(golden_path)
    if not golden_path.is_absolute():
        golden_path = _GOLDEN_DIR / golden_path
    with golden_path.open() as fh:
        return json.load(fh)


__all__ = [
    "GoldenMismatch",
    "compare_to_golden",
    "load_golden",
]
