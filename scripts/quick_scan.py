"""Fast verification scanner for local development.

Profiles:
- quick: fast smoke checks for frequent iteration
- standard: broader targeted suite
- full: complete test suite
"""

from __future__ import annotations

import argparse
import subprocess

QUICK_TESTS: tuple[str, ...] = (
    "tests/test_phase1_research_script.py",
    "tests/test_two_book_research_script.py",
    "tests/test_strategy_team_research_script.py",
    "tests/test_futures_feature_points.py",
    "tests/test_data_sync.py",
    "tests/test_runtime_cache.py",
    "tests/test_live_execution_state_machine.py",
)

STANDARD_TESTS: tuple[str, ...] = (
    "tests/test_native_backend.py",
    "tests/test_optimize_two_stage.py",
    "tests/test_message_bus.py",
    "tests/test_runtime_cache.py",
    "tests/test_replay.py",
    "tests/test_fast_eval.py",
    "tests/test_frozen_dataset.py",
    "tests/test_parity_fast_eval.py",
    "tests/test_system_assembly.py",
    "tests/test_event_clock.py",
    "tests/test_data_handler_prefrozen.py",
    "tests/test_portfolio_fast_stats.py",
    "tests/test_ohlcv_loader.py",
    "tests/test_walk_forward.py",
    "tests/test_execution_protective_orders.py",
    "tests/test_live_execution_state_machine.py",
    "tests/test_lookahead.py",
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run fast verification scan profiles.")
    parser.add_argument(
        "--profile",
        choices=["quick", "standard", "full"],
        default="quick",
        help="Verification profile to run.",
    )
    parser.add_argument("--maxfail", type=int, default=1, help="Pytest maxfail value.")
    parser.add_argument("--with-build", action="store_true", help="Run uv build at the end.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands only.")
    return parser


def _command_plan(profile: str, maxfail: int, with_build: bool) -> list[list[str]]:
    pytest_base = ["uv", "run", "pytest", "-q", "--maxfail", str(max(1, int(maxfail)))]
    cmds: list[list[str]] = [["uv", "run", "ruff", "check", "."]]

    if profile == "quick":
        cmds.append([*pytest_base, *QUICK_TESTS])
    elif profile == "standard":
        cmds.append([*pytest_base, *STANDARD_TESTS])
    else:
        cmds.append([*pytest_base])

    if with_build:
        cmds.append(["uv", "build"])
    return cmds


def _run(cmd: list[str], *, dry_run: bool) -> None:
    print(f"$ {' '.join(cmd)}")
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def main() -> None:
    args = _build_parser().parse_args()
    plan = _command_plan(args.profile, args.maxfail, bool(args.with_build))
    for cmd in plan:
        _run(cmd, dry_run=bool(args.dry_run))


if __name__ == "__main__":
    main()
