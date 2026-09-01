from __future__ import annotations

from lumina_quant.dashboard import exact_window_service


def test_empty_exact_window_payload_tracks_reason() -> None:
    payload = exact_window_service.empty_exact_window_payload(reason="missing_bundle")

    assert payload["status"] == "missing_bundle"
    assert payload["timeframes"] == []
    assert payload["top_candidates"] == []
    assert payload["portfolio_weights"] == []


def test_load_exact_window_summary_payload_normalizes_bundle(monkeypatch) -> None:
    monkeypatch.setattr(
        exact_window_service,
        "load_exact_window_bundle",
        lambda root=None: {
            "root": "/tmp/exact-window",
            "run_root": "/tmp/exact-window/run-123",
            "warnings": ["stale follow-up artifact"],
            "summary": {
                "generated_at": "2026-03-25T00:00:00Z",
                "candidate_count": 42,
                "evaluated_count": 39,
                "promoted_count": 2,
                "btc_beating_candidate_count": 3,
                "provisional_candidate_count": 1,
                "candidate_pool_count": 4,
                "windows": {
                    "train_start": "2025-01-01T00:00:00Z",
                    "val_start": "2026-01-01T00:00:00Z",
                },
                "execution_profile": {
                    "requested_timeframes": ["15m", "1h"],
                    "requested_symbols": ["BTC/USDT", "ETH/USDT"],
                    "low_ram_profile": True,
                },
                "notes": {
                    "selection_basis": "validation_only",
                    "operator_note": "retain latest exact-window bundle",
                },
                "best_per_strategy": [
                    {
                        "timeframe": "15m",
                        "candidate_id": "cand-1",
                        "name": "alpha_15m",
                        "family": "trend",
                        "promoted": True,
                        "hard_reject_reasons": {"pbo": 0.5},
                        "oos": {
                            "return": 0.12,
                            "sharpe": 1.8,
                            "max_drawdown": 0.04,
                            "trade_count": 28,
                        },
                    }
                ],
                "portfolio": {
                    "construction_basis": "best_per_strategy_fallback",
                    "oos": {
                        "total_return": 0.08,
                        "sharpe": 1.3,
                        "max_drawdown": 0.05,
                    },
                    "weights": [
                        {
                            "name": "alpha_15m",
                            "timeframe": "15m",
                            "weight": 0.6,
                            "family": "trend",
                            "oos_return": 0.12,
                            "oos_sharpe": 1.8,
                        }
                    ],
                },
            },
            "decision": {
                "next_action": "promote_candidate",
                "promoted_total": 2,
                "total_evaluated": 39,
                "max_peak_rss_mib": 2048.5,
                "valid_strategy_found": True,
                "timeframe_rows": [
                    {
                        "timeframe": "15m",
                        "candidate_id": "cand-1",
                        "name": "alpha_15m",
                        "family": "trend",
                        "promoted": True,
                        "reject_reason_counts": [{"rejection_reason": "pbo", "count": 1}],
                        "oos": {
                            "return": 0.12,
                            "sharpe": 1.8,
                            "max_drawdown": 0.04,
                            "trade_count": 28,
                        },
                    }
                ],
                "timeframes": ["15m", "1h"],
            },
            "memory_evidence": {
                "status": "completed",
                "peak_rss_mib": 1536.25,
                "soft_limit_mib": 2918.4,
                "hard_limit_mib": 3891.2,
            },
        },
    )

    payload = exact_window_service.load_exact_window_summary_payload()

    assert payload["status"] == "ok"
    assert payload["generated_at"] == "2026-03-25T00:00:00Z"
    assert payload["summary"]["candidate_count"] == 42
    assert payload["summary"]["requested_timeframes"] == ["15m", "1h"]
    assert payload["decision"]["next_action"] == "promote_candidate"
    assert payload["memory"]["peak_rss_mib"] == 1536.25
    assert payload["timeframes"][0]["candidate_id"] == "cand-1"
    assert payload["timeframes"][0]["reject_reasons"] == ["pbo"]
    assert payload["top_candidates"][0]["oos_sharpe"] == 1.8
    assert payload["portfolio_weights"][0]["weight"] == 0.6
    assert payload["warnings"] == ["stale follow-up artifact"]


def test_load_exact_window_summary_payload_handles_missing_summary(monkeypatch) -> None:
    monkeypatch.setattr(
        exact_window_service,
        "load_exact_window_bundle",
        lambda root=None: {
            "root": "/tmp/exact-window",
            "run_root": "/tmp/exact-window/run-123",
            "summary": None,
        },
    )

    payload = exact_window_service.load_exact_window_summary_payload()

    assert payload["status"] == "missing_summary"
    assert payload["root"] == "/tmp/exact-window"
