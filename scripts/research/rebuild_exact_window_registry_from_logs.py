from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
for candidate in (REPO_ROOT, SRC_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from lumina_quant.eval.exact_window_log_archive import (  # noqa: E402
    write_exact_window_canonical_registry,
    write_exact_window_log_archive,
)


def main() -> int:
    report_root = REPO_ROOT / "var" / "reports" / "exact_window_backtests"
    result = {
        "canonical_registry": write_exact_window_canonical_registry(report_root=report_root),
        "archive": write_exact_window_log_archive(
            log_dir=REPO_ROOT / "logs",
            report_root=report_root,
        ),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
