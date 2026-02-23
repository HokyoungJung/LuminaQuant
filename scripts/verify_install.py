import platform
import subprocess


def run(cmd):
    print(f"$ {' '.join(cmd)}")
    subprocess.check_call(cmd)


def main():
    print(f"Platform: {platform.platform()}")
    run(["uv", "sync", "--extra", "optimize", "--extra", "dev", "--extra", "live"])
    run(["uv", "run", "ruff", "check", "."])
    run(["uv", "run", "python", "scripts/check_architecture.py"])
    run(
        [
            "uv",
            "run",
            "python",
            "scripts/benchmark_backtest.py",
            "--iters",
            "1",
            "--warmup",
            "0",
            "--output",
            "reports/benchmarks/verify_install.json",
        ]
    )
    run(
        [
            "uv",
            "run",
            "pytest",
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
        ]
    )
    print("Installation and test verification completed.")


if __name__ == "__main__":
    main()
