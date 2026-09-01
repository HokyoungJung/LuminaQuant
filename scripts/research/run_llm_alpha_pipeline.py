from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
for candidate in (REPO_ROOT, SRC_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from lumina_quant.core.memory_budget import DEFAULT_EXECUTION_MEMORY_POLICY  # noqa: E402
from lumina_quant.workflows.alpha_research_pipeline import (  # noqa: E402
    write_alpha_research_pipeline_manifest,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_llm_alpha_pipeline.py",
        description="Write an article-inspired alpha research pipeline manifest for LuminaQuant.",
    )
    parser.add_argument(
        "--report-root",
        default=str(REPO_ROOT / "var" / "reports" / "exact_window_backtests"),
        help="Root directory for exact-window reports.",
    )
    parser.add_argument(
        "--article-url",
        default="https://www.linkedin.com/pulse/ko-managing-real-life-portfolio-based-multi-agent-llms-yeachan-heo-fcmac",
        help="Reference article URL.",
    )
    parser.add_argument(
        "--image-path",
        action="append",
        default=[
            str(REPO_ROOT.parent / "KakaoTalk_20260309_210834478.jpg"),
            str(REPO_ROOT.parent / "KakaoTalk_20260309_210834478_01.jpg"),
        ],
        help="Reference image path. Repeat for multiple paths.",
    )
    parser.add_argument(
        "--total-memory-cap-gib",
        type=float,
        default=DEFAULT_EXECUTION_MEMORY_POLICY.total_memory_cap_gib,
    )
    parser.add_argument(
        "--heavy-run-cap-gib",
        type=float,
        default=DEFAULT_EXECUTION_MEMORY_POLICY.heavy_run_cap_gib,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    result = write_alpha_research_pipeline_manifest(
        report_root=Path(args.report_root).resolve(),
        article_url=str(args.article_url),
        image_paths=[str(item) for item in list(args.image_path or [])],
        total_memory_cap_gib=float(args.total_memory_cap_gib),
        heavy_run_cap_gib=float(args.heavy_run_cap_gib),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
