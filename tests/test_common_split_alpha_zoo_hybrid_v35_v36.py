from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts" / "research" / "run_common_split_alpha_zoo_hybrid_v35_v36.py"
HYBRID_PATH = ROOT / "scripts" / "research" / "run_profit_moonshot_hybrid_v35_v36_fixed_inputs.py"
CALIBRATOR_PATH = ROOT / "scripts" / "research" / "calibrate_crypto_fx_edges.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


RUNNER = _load(RUNNER_PATH, "run_common_split_alpha_zoo_hybrid_v35_v36")
HYBRID = _load(HYBRID_PATH, "run_profit_moonshot_hybrid_v35_v36_fixed_inputs_common_test")
CALIBRATOR = _load(CALIBRATOR_PATH, "calibrate_crypto_fx_edges_common_test")


def _calibration_record(split: str, pnl_bps: float) -> dict[str, object]:
    return {
        "candidate_id": "factor_a",
        "side": "LONG",
        "symbol": "BTC/USDT",
        "regime_bucket": "risk_on",
        "volatility_bucket": "vol_mid",
        "factor_bucket": "top",
        "split": split,
        "net_pnl_bps": pnl_bps,
    }


def test_apply_common_split_uses_exact_calendar_boundaries() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": [
                "2024-12-31T23:00:00Z",
                "2025-01-01T00:00:00Z",
                "2025-12-31T23:00:00Z",
                "2026-01-01T00:00:00Z",
                "2026-02-28T23:00:00Z",
                "2026-03-01T00:00:00Z",
                "2026-05-06T23:00:00Z",
                "2026-05-07T00:00:00Z",
            ],
            "symbol": ["BTC/USDT"] * 8,
            "close": range(8),
        }
    )

    out = RUNNER.apply_common_split(frame)

    assert out["timestamp"].astype(str).tolist() == [
        "2025-01-01 00:00:00",
        "2025-12-31 23:00:00",
        "2026-01-01 00:00:00",
        "2026-02-28 23:00:00",
        "2026-03-01 00:00:00",
        "2026-05-06 23:00:00",
    ]
    assert out["split"].tolist() == ["train", "train", "validation", "validation", "locked_oos", "locked_oos"]


def test_split_bounded_forward_labels_do_not_cross_into_future_split() -> None:
    base = pd.DataFrame(
        {
            "timestamp": [
                "2025-12-31T21:00:00Z",
                "2025-12-31T22:00:00Z",
                "2025-12-31T23:00:00Z",
                "2026-01-01T00:00:00Z",
            ],
            "symbol": ["BTC/USDT"] * 4,
            "close": [100.0, 101.0, 102.0, 999_999.0],
        }
    )
    poisoned = base.copy()
    poisoned.loc[poisoned.index[-1], "close"] = 1.0

    labeled = RUNNER.add_split_bounded_forward_return_label(RUNNER.apply_common_split(base), horizon=1)
    labeled_poisoned = RUNNER.add_split_bounded_forward_return_label(RUNNER.apply_common_split(poisoned), horizon=1)

    train = labeled[labeled["split"].eq("train")].reset_index(drop=True)
    train_poisoned = labeled_poisoned[labeled_poisoned["split"].eq("train")].reset_index(drop=True)
    # The last train row would otherwise look into validation and is masked.
    assert pd.isna(train.loc[2, "forward_return"])
    # Earlier train labels remain same-split and are invariant to validation/OOS poisoning.
    assert train.loc[:1, "forward_return"].tolist() == train_poisoned.loc[:1, "forward_return"].tolist()


def test_factor_screen_selection_ignores_locked_oos_poison() -> None:
    rows: list[dict[str, object]] = []
    for split in ("train", "validation", "locked_oos"):
        for idx in range(30):
            sign = 1.0 if idx >= 15 else -1.0
            rows.append(
                {
                    "split": split,
                    "factor_good": sign,
                    "factor_bad": -sign,
                    "forward_return": sign,
                }
            )
    clean = pd.DataFrame(rows)
    poisoned = clean.copy()
    locked = poisoned["split"].eq("locked_oos")
    poisoned.loc[locked, "factor_good"] = -poisoned.loc[locked, "factor_good"]
    poisoned.loc[locked, "factor_bad"] = -poisoned.loc[locked, "factor_bad"]
    poisoned.loc[locked, "forward_return"] = -poisoned.loc[locked, "forward_return"]

    clean_screen = RUNNER.screen_factor_frame(clean, factors=["factor_good", "factor_bad"], top_n=2)
    poisoned_screen = RUNNER.screen_factor_frame(poisoned, factors=["factor_good", "factor_bad"], top_n=2)

    assert [row["factor"] for row in clean_screen["selected_factors"]] == [
        row["factor"] for row in poisoned_screen["selected_factors"]
    ]
    assert clean_screen["selected_factors"][0]["factor"] == "factor_good"


def test_edge_calibration_signature_ignores_locked_oos_poison() -> None:
    train_val = [_calibration_record("train", 25.0) for _ in range(4)] + [
        _calibration_record("validation", 20.0) for _ in range(4)
    ]
    clean = [*train_val, *[_calibration_record("locked_oos", -500.0) for _ in range(8)]]
    poisoned = [*train_val, *[_calibration_record("locked_oos", 5000.0) for _ in range(8)]]
    kwargs = {
        "ledger_summary": {"record_count": len(clean)},
        "bucket_fields": ("candidate_id", "side", "symbol", "regime_bucket", "volatility_bucket", "factor_bucket"),
        "parent_fields": ("candidate_id", "side"),
        "min_bucket_n": 1,
        "confidence_z": 0.0,
        "min_lower_edge_bps": 0.0,
        "max_tail_loss_bps": 250.0,
    }

    clean_payload = CALIBRATOR.build_calibration_payload(clean, **kwargs)
    poisoned_payload = CALIBRATOR.build_calibration_payload(poisoned, **kwargs)

    assert clean_payload["calibrated_edges_for_strategy"] == poisoned_payload["calibrated_edges_for_strategy"]
    assert clean_payload["locked_oos_calibration_record_count"] == 0
    assert poisoned_payload["locked_oos_calibration_record_count"] == 0


def test_timestamp_hash_uses_unique_split_timestamp_index() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": ["2025-01-01T00:00:00Z", "2025-01-01T00:00:00Z", "2026-03-01T00:00:00Z"],
            "symbol": ["BTC/USDT", "ETH/USDT", "BTC/USDT"],
            "close": [1.0, 2.0, 3.0],
        }
    )
    single_symbol = frame[frame["symbol"].eq("BTC/USDT")].copy()

    assert RUNNER._timestamp_index_hash(RUNNER.apply_common_split(frame)) == RUNNER._timestamp_index_hash(
        RUNNER.apply_common_split(single_symbol)
    )


def test_hybrid_alpha_stream_applies_common_replay_manifest_instead_of_fractional_split() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": [
                "2025-01-01T00:00:00Z",
                "2026-01-01T00:00:00Z",
                "2026-03-01T00:00:00Z",
                "2026-05-07T00:00:00Z",
            ],
            "symbol": ["BTC/USDT"] * 4,
            "close": [1.0, 2.0, 3.0, 4.0],
        }
    )

    out = HYBRID._apply_common_split_contract(frame, RUNNER.COMMON_SPLIT_CONTRACT)

    assert out["split"].tolist() == ["train", "validation", "locked_oos"]
    assert str(out["timestamp"].iloc[-1]) == "2026-03-01 00:00:00"


def test_alpha_rows_use_top_level_integer_grid_results() -> None:
    payload = {
        "integer_grid_results": [
            {
                "leverage": 6,
                "deployable_success": True,
                "liquidation_audit": {
                    "split_status": {"locked_oos": {"liquidation_count": 0, "minimum_margin_buffer": 123.0}}
                },
                "split_metrics": {
                    "locked_oos": {
                        "total_return": 0.2,
                        "max_drawdown": 0.05,
                        "trade_count": 7,
                    }
                },
            }
        ],
        "trade_split_periods": {"locked_oos": {"start_timestamp": "2026-03-01T01:00:00Z"}},
    }
    periods = {
        split: {"start_timestamp": f"{split}-start", "end_timestamp": f"{split}-end"}
        for split in RUNNER.SPLIT_ORDER
    }

    rows = RUNNER._alpha_rows_from_replay(
        payload,
        candidate="alpha",
        family="alpha_family",
        manifest_periods=periods,
    )
    locked = next(row for row in rows if row["split"] == "locked_oos")

    assert locked["total_return"] == 0.2
    assert locked["liquidation_count"] == 0
    assert locked["minimum_margin_buffer"] == 123.0
    assert locked["deployable_success"] is True


def test_hybrid_rows_use_split_periods_as_effective_active_window() -> None:
    hybrid_payload = {
        "split_periods": {
            "train": {"start_timestamp": "2025-01-01T00:00:00Z", "end_timestamp": "2025-12-31T23:00:00Z"},
            "validation": {"start_timestamp": "2026-01-01T00:00:00Z", "end_timestamp": "2026-02-28T23:00:00Z"},
            "locked_oos": {"start_timestamp": "2026-03-01T00:00:00Z", "end_timestamp": "2026-05-06T23:00:00Z"},
        }
    }
    item = {"splits": {"locked_oos": {"total_return": 0.1}}, "selection_provenance": {}}

    rows = RUNNER._hybrid_rows(item, candidate="hybrid", hybrid_payload=hybrid_payload)
    locked = next(row for row in rows if row["split"] == "locked_oos")

    assert locked["active_start_timestamp"] == "2026-03-01T00:00:00Z"
    assert locked["active_end_timestamp"] == "2026-05-06T23:00:00Z"
