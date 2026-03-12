import platform
import shutil
import subprocess


def run(cmd):
    print(f"$ {' '.join(cmd)}")
    subprocess.check_call(cmd)


def run_optional(cmd):
    print(f"$ {' '.join(cmd)}")
    return subprocess.run(cmd, check=False)


def main():
    print(f"Platform: {platform.platform()}")
    run(["uv", "sync", "--extra", "optimize", "--extra", "dev", "--extra", "live", "--extra", "gpu"])
    run(["uv", "run", "ruff", "check", "."])
    run(["uv", "run", "python", "scripts/check_architecture.py"])
    run(["uv", "run", "python", "scripts/verify_docs.py"])
    run(
        [
            "uv",
            "run",
            "python",
            "scripts/ci/verify_polars_gpu_runtime.py",
            "--output-json",
            "reports/benchmarks/verify_install_gpu_contract.json",
        ]
    )
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
            "python",
            "scripts/verify_8gb_baseline.py",
            "--benchmark",
            "reports/benchmarks/verify_install.json",
            "--allow-missing-rss-source",
            "--skip-dmesg",
            "--allow-missing-oom-sources",
            "--output",
            "reports/benchmarks/verify_install_8gb_gate.json",
        ]
    )
    print(
        "[verify_install] 8GB gate ran in fallback mode "
        "(RSS/OOM strict checks require --time-log and explicit OOM logs)."
    )
    if shutil.which("nvidia-smi"):
        result = run_optional(
            [
                "uv",
                "run",
                "python",
                "scripts/ci/verify_polars_gpu_runtime.py",
                "--require-gpu",
                "--mode",
                "forced-gpu",
                "--output-json",
                "reports/benchmarks/verify_install_gpu_runtime.json",
            ]
        )
        if result.returncode != 0:
            raise SystemExit(result.returncode)
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
