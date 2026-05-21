from lumina_quant.live import paper_summary
from lumina_quant.pipeline import run_paper_live_pipeline


def test_paper_live_pipeline_never_enables_real_orders() -> None:
    result = run_paper_live_pipeline("sample_data/sample_ohlcv.csv")
    summary = paper_summary(result)
    assert summary["mode"] == "paper_replay_only"
    assert summary["order_execution_enabled"] is False
    assert summary["safety"] == {
        "real_order_routing": False,
        "uses_only_local_sample_data": True,
        "credentials_required": False,
    }
