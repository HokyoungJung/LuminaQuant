import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "benchmark_rawfirst_backend.py"

spec = importlib.util.spec_from_file_location("benchmark_rawfirst_backend", MODULE_PATH)
assert spec is not None and spec.loader is not None
benchmark_rawfirst_backend = importlib.util.module_from_spec(spec)
sys.modules["benchmark_rawfirst_backend"] = benchmark_rawfirst_backend
spec.loader.exec_module(benchmark_rawfirst_backend)


def test_python_only_rawfirst_benchmark_writes_json(tmp_path: Path) -> None:
    output = tmp_path / "rawfirst_benchmark.json"

    exit_code = benchmark_rawfirst_backend.main(
        [
            "--backend",
            "python",
            "--trades",
            "500",
            "--seconds",
            "120",
            "--evals",
            "1",
            "--output-json",
            str(output),
        ]
    )

    assert exit_code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "pass"
    assert payload["python"]["output_bars"] == 120
    assert payload["python"]["input_trades_per_second"] > 0
    assert payload["rust"] is None
    assert payload["parity"]["checked"] is False


def test_rawfirst_benchmark_result_has_rust_parity_when_available() -> None:
    result = benchmark_rawfirst_backend.run_benchmark(
        trades=500,
        seconds=120,
        evals=1,
        backend="both",
    )

    assert result.python is not None
    if result.rust_available:
        assert result.status == "pass"
        assert result.rust is not None
        assert result.parity.checked is True
        assert result.parity.passed is True
        assert result.rust_speedup_vs_python is not None
    else:
        assert result.status == "rust_unavailable"
        assert result.rust is None
