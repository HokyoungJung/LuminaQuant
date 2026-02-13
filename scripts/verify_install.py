import platform
import subprocess
import sys


def run(cmd):
    print(f"$ {' '.join(cmd)}")
    subprocess.check_call(cmd)


def main():
    print(f"Platform: {platform.platform()}")
    run([sys.executable, "-m", "pip", "install", "."])
    run([sys.executable, "-m", "pip", "install", "ruff"])
    run(["ruff", "format", ".", "--check"])
    run(["ruff", "check", "."])
    run([sys.executable, "scripts/check_architecture.py"])
    run(
        [
            sys.executable,
            "scripts/benchmark_backtest.py",
            "--iters",
            "1",
            "--warmup",
            "0",
            "--output",
            "reports/benchmarks/verify_install.json",
        ]
    )
    run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"])
    print("Installation and test verification completed.")


if __name__ == "__main__":
    main()
