import math

from lumina_quant.pipeline import run_backtest_pipeline
from lumina_quant.rust_kernel import run_rust_backtest, rust_kernel_available


def test_rust_kernel_matches_python_backtest() -> None:
    assert rust_kernel_available(), "cargo/native rust kernel must be available in public CI"
    python_result = run_backtest_pipeline("sample_data/sample_ohlcv.csv")
    rust_result = run_rust_backtest("sample_data/sample_ohlcv.csv")
    assert rust_result["trade_count"] == python_result.trade_count
    assert math.isclose(
        rust_result["final_equity"],
        python_result.final_equity,
        rel_tol=0.0,
        abs_tol=1e-9,
    )
    assert math.isclose(
        rust_result["max_drawdown"],
        python_result.max_drawdown,
        rel_tol=0.0,
        abs_tol=1e-12,
    )
