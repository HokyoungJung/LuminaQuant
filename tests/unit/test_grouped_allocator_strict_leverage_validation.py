import importlib.util
import json
import math
import sys
from pathlib import Path

MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "research"
    / "run_grouped_allocator_strict_leverage_validation.py"
)
SPEC = importlib.util.spec_from_file_location(
    "run_grouped_allocator_strict_leverage_validation",
    MODULE_PATH,
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_resolve_portfolio_candidates_uses_source_components(tmp_path: Path) -> None:
    detail_path = tmp_path / "detail.json"
    detail_path.write_text(
        json.dumps(
            [
                {
                    "candidate_id": "cand-1",
                    "name": "alpha",
                    "strategy_timeframe": "1h",
                    "symbols": ["BTC/USDT"],
                    "params": {"lookback": 20},
                }
            ]
        ),
        encoding="utf-8",
    )
    portfolio_payload = {
        "source_components": [
            {
                "candidate_id": "cand-1",
                "name": "alpha",
                "artifact_path": str(detail_path),
            }
        ]
    }

    resolved = MODULE._resolve_portfolio_candidates(portfolio_payload=portfolio_payload)

    assert len(resolved) == 1
    assert resolved[0]["candidate_id"] == "cand-1"
    assert resolved[0]["params"]["lookback"] == 20


def test_resolve_portfolio_candidates_falls_back_to_strict_cache(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cache_dir = tmp_path / "strict_validation_cache"
    cache_dir.mkdir(parents=True)
    cache_path = cache_dir / "cached.json"
    cache_path.write_text(
        json.dumps(
            {
                "candidates": [
                    {
                        "candidate_id": "cand-2",
                        "name": "beta",
                        "strategy_timeframe": "1h",
                        "symbols": ["ETH/USDT"],
                        "params": {"lookback": 48, "leverage": 3},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(MODULE._validation, "DEFAULT_STRICT_VALIDATION_CACHE_DIR", cache_dir)

    portfolio_payload = {
        "source_components": [
            {
                "candidate_id": "cand-2",
                "name": "beta",
                "artifact_path": str(tmp_path / "missing.json"),
            }
        ]
    }

    resolved = MODULE._resolve_portfolio_candidates(portfolio_payload=portfolio_payload)

    assert len(resolved) == 1
    assert resolved[0]["candidate_id"] == "cand-2"
    assert resolved[0]["params"]["lookback"] == 48
    assert resolved[0]["params"]["leverage"] == 3


def test_apply_group_leverage_adds_root_and_param_fields() -> None:
    candidates = [
        {
            "candidate_id": "cand-1",
            "name": "alpha",
            "params": {"lookback": 20},
        }
    ]

    leveraged = MODULE._apply_group_leverage(candidates, leverage=9)

    assert leveraged[0]["leverage"] == 9
    assert leveraged[0]["params"]["leverage"] == 9
    assert "leverage" not in candidates[0]
    assert "leverage" not in candidates[0]["params"]


def test_apply_candidate_level_leverage_to_rows_transforms_streams() -> None:
    rows = [
        {
            "candidate_id": "cand-1",
            "name": "alpha",
            "params": {"leverage": 3},
            "return_streams": {
                "train": [
                    {"datetime": "2025-01-01T00:00:00Z", "t": "2025-01-01T00:00:00Z", "v": 0.01}
                ],
                "val": [
                    {"datetime": "2026-01-01T00:00:00Z", "t": "2026-01-01T00:00:00Z", "v": 0.02}
                ],
                "oos": [
                    {"datetime": "2026-02-01T00:00:00Z", "t": "2026-02-01T00:00:00Z", "v": 0.03}
                ],
            },
        }
    ]

    leveraged, liquidation_counts = MODULE._apply_candidate_level_leverage_to_rows(rows)

    assert liquidation_counts == {"cand-1": 0}
    assert math.isclose(leveraged[0]["return_streams"]["train"][0]["v"], 0.03)
    assert math.isclose(leveraged[0]["return_streams"]["val"][0]["v"], 0.06)
    assert math.isclose(leveraged[0]["return_streams"]["oos"][0]["v"], 0.09)
    assert leveraged[0]["metadata"]["candidate_level_leverage"] == 3


def test_apply_candidate_level_leverage_preserves_epoch_millisecond_timestamps() -> None:
    rows = [
        {
            "candidate_id": "cand-epoch-ms",
            "name": "epoch-ms",
            "params": {"leverage": 2},
            "return_streams": {
                "train": [
                    {"t": 1735689600000, "v": 0.01},
                    {"t": 1735776000000, "v": -0.02},
                ],
                "val": [],
                "oos": [],
            },
        }
    ]

    leveraged, _ = MODULE._apply_candidate_level_leverage_to_rows(rows)
    train_stream = leveraged[0]["return_streams"]["train"]

    assert train_stream[0]["datetime"].startswith("2025-01-01T00:00:00")
    assert train_stream[1]["datetime"].startswith("2025-01-02T00:00:00")

    evaluation = MODULE._validation.evaluate_saved_weight_portfolio(
        [{**leveraged[0], "_saved_weight": 1.0}]
    )
    daily_train = evaluation["portfolio_daily_return_streams"]["train"]

    assert len(daily_train) == 2
    assert daily_train[0]["datetime"].startswith("2025-01-01T00:00:00")
    assert daily_train[1]["datetime"].startswith("2025-01-02T00:00:00")


def test_build_blend_payload_combines_group_streams() -> None:
    incumbent_payload = {
        "source_portfolio_path": "/tmp/incumbent.json",
        "portfolio_metrics": {
            "train": {"turnover": 0.0},
            "val": {"turnover": 0.0},
            "oos": {"turnover": 0.0},
        },
        "portfolio_return_streams": {
            "train": [{"t": "2025-01-01T00:00:00Z", "v": 0.10}],
            "val": [{"t": "2026-01-01T00:00:00Z", "v": 0.02}],
            "oos": [{"t": "2026-02-01T00:00:00Z", "v": 0.03}],
        },
    }
    autoresearch_payload = {
        "source_portfolio_path": "/tmp/autoresearch.json",
        "portfolio_metrics": {
            "train": {"turnover": 0.0},
            "val": {"turnover": 0.0},
            "oos": {"turnover": 0.0},
        },
        "portfolio_return_streams": {
            "train": [{"t": "2025-01-01T00:00:00Z", "v": -0.10}],
            "val": [{"t": "2026-01-01T00:00:00Z", "v": 0.00}],
            "oos": [{"t": "2026-02-01T00:00:00Z", "v": 0.01}],
        },
    }

    payload = MODULE._build_blend_payload(
        incumbent_payload=incumbent_payload,
        autoresearch_payload=autoresearch_payload,
        blend_weight=0.85,
    )

    train_stream = payload["portfolio_return_streams"]["train"]
    oos_stream = payload["portfolio_return_streams"]["oos"]
    assert math.isclose(train_stream[0]["v"], 0.07)
    assert math.isclose(oos_stream[0]["v"], 0.027)
    assert payload["best_weights"]["current_one_shot_incumbent"] == 0.85
    assert math.isclose(payload["best_weights"]["autoresearch_pair_55_45"], 0.15)


def test_apply_allocator_state_leverage_to_payload_updates_split_metrics(monkeypatch) -> None:
    allocator_payload = {
        "split_metrics": {"train": {"total_return": 0.01}},
        "states": [
            {
                "date": "2025-01-01T00:00:00+00:00",
                "split_group": "train",
                "state": "blend_85_15",
                "return": 0.01,
            },
            {
                "date": "2025-01-02T00:00:00+00:00",
                "split_group": "oos",
                "state": "autoresearch_55_45",
                "return": 0.02,
            },
        ],
        "current_state": {
            "date": "2025-01-02T00:00:00+00:00",
            "split_group": "oos",
            "state": "autoresearch_55_45",
            "return": 0.02,
        },
    }

    def _fake_apply_state_leverage(frame, *, leverage_by_state):
        tuned = frame.copy()
        tuned["leveraged_return"] = [
            row.base_return * leverage_by_state[row.state] for row in tuned.itertuples(index=False)
        ]
        tuned["leverage"] = [leverage_by_state[row.state] for row in tuned.itertuples(index=False)]
        tuned["segment_equity"] = [1.0, 1.1]
        tuned["segment_floor"] = [0.0, 0.0]
        tuned["liquidated"] = [False, False]
        return tuned, {"blend_85_15": 0, "autoresearch_55_45": 0, "incumbent": 0}

    monkeypatch.setattr(
        MODULE._leverage_tuning, "_apply_state_leverage", _fake_apply_state_leverage
    )
    monkeypatch.setattr(
        MODULE._three_way,
        "_metrics_by_split",
        lambda frame, return_col: {
            "train": {
                "total_return": float(frame.loc[frame["split_group"] == "train", return_col].sum())
            },
            "oos": {
                "total_return": float(frame.loc[frame["split_group"] == "oos", return_col].sum())
            },
        },
    )

    updated = MODULE._apply_allocator_state_leverage_to_payload(
        allocator_payload=allocator_payload,
        leverage_by_state={"blend_85_15": 25, "autoresearch_55_45": 1, "incumbent": 9},
    )

    assert updated["unleveraged_split_metrics"] == {"train": {"total_return": 0.01}}
    assert math.isclose(updated["split_metrics"]["train"]["total_return"], 0.25)
    assert math.isclose(updated["split_metrics"]["oos"]["total_return"], 0.02)
    assert updated["states"][0]["applied_leverage"] == 25
    assert math.isclose(updated["states"][0]["return"], 0.25)
    assert updated["current_state"]["state"] == "autoresearch_55_45"
