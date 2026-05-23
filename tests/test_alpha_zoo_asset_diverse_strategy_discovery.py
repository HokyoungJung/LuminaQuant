from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "research" / "run_alpha_zoo_asset_diverse_strategy_discovery.py"
SPEC = importlib.util.spec_from_file_location(
    "run_alpha_zoo_asset_diverse_strategy_discovery",
    MODULE_PATH,
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "train_return": 0.24,
        "validation_return": 0.12,
        "locked_oos_return": 0.04,
        "validation_mdd": 0.04,
        "validation_min_half_return": 0.01,
        "validation_trade_event_count": 44,
        "train_return_per_turnover_proxy_bps": 32.0,
        "validation_return_per_turnover_proxy_bps": 36.0,
    }
    row.update(overrides)
    return row


def _gate_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "paper_candidate_gate_pass": True,
        "primary_10bps_promotion_gate_pass": True,
        "ready_for_paper": True,
        "ready_for_real": False,
        "real_money_execution": False,
        "decision": "paper_testnet_candidate_after_fill_preflight",
        "rejection_reasons": [],
    }
    row.update(overrides)
    return row


def test_asset_diverse_score_ignores_locked_oos() -> None:
    strong_oos = _row(locked_oos_return=0.50, locked_oos_return_per_turnover_proxy_bps=100.0)
    weak_oos = _row(locked_oos_return=-0.50, locked_oos_return_per_turnover_proxy_bps=-100.0)

    assert MODULE._asset_diverse_score(strong_oos) == MODULE._asset_diverse_score(weak_oos)


def test_shadow_universe_is_forced_no_promotion() -> None:
    forced = MODULE._force_shadow_only(_gate_row(), "shadow_universe_not_promotion_eligible")

    assert forced["paper_candidate_gate_pass"] is False
    assert forced["primary_10bps_promotion_gate_pass"] is False
    assert forced["ready_for_paper"] is False
    assert forced["ready_for_real"] is False
    assert forced["real_money_execution"] is False
    assert forced["decision"] == "no_promotion_shadow_or_reject"
    assert "shadow_universe_not_promotion_eligible" in forced["rejection_reasons"]


def test_defaults_are_multi_asset_group_paper_only_30m_plus() -> None:
    args = MODULE.parse_args([])
    symbols = MODULE._parse_csv_symbols(args.symbols)
    shadow_symbols = MODULE._parse_csv_symbols(args.shadow_symbols)
    timeframes = MODULE.feedback._validate_timeframes(MODULE._parse_csv_symbols(args.timeframes.lower()))

    assert {MODULE._asset_group(symbol) for symbol in (*symbols, *shadow_symbols)} >= {
        "crypto_major",
        "crypto_high_beta_alt",
        "crypto_payment_alt",
        "crypto_exchange_beta",
        "precious_metal_proxy",
    }
    assert all(MODULE.feedback._timeframe_hours(timeframe) >= 0.5 for timeframe in timeframes)
    assert MODULE.STRATEGY_SCOPE == "single_symbol_cross_asset_conditioned_only"
    assert MODULE.DEFAULT_SYMBOLS == ("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "TRXUSDT")


def test_selected_rows_keep_group_diversity_and_paper_rows() -> None:
    rows = MODULE._rank_rows(
        [
            _row(model_id="btc", asset_group="crypto_major", paper_candidate_gate_pass=False),
            _row(model_id="metal", asset_group="precious_metal_proxy", paper_candidate_gate_pass=False),
            _row(model_id="paper", asset_group="crypto_payment_alt", paper_candidate_gate_pass=True),
        ]
    )
    selected = MODULE._selected_output_rows(rows, top_n=1)

    assert {row["model_id"] for row in selected} == {"btc", "metal", "paper"}
