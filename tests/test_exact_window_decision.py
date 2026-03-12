from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from lumina_quant.eval.exact_window_decision import write_exact_window_decision_bundle


def _ts(year: int, month: int, day: int) -> int:
    return int(datetime(year, month, day, tzinfo=UTC).timestamp() * 1000)


def _detail(
    *,
    candidate_id: str,
    timeframe: str,
    strategy_class: str,
    val_sharpe: float,
    val_return: float,
    oos_sharpe: float,
    oos_return: float,
    val_stream: list[tuple[int, int, int, float]] | None = None,
    oos_stream: list[tuple[int, int, int, float]] | None = None,
    hard_reject_reasons: dict[str, float] | None = None,
) -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "name": f"{strategy_class}_{timeframe}_{candidate_id}",
        "family": "test_family",
        "strategy_class": strategy_class,
        "strategy_timeframe": timeframe,
        "val": {
            "return": val_return,
            "sharpe": val_sharpe,
            "deflated_sharpe": max(val_sharpe - 0.1, 0.0),
            "pbo": 0.05,
            "turnover": 0.12,
            "mdd": 0.03,
        },
        "oos": {
            "return": oos_return,
            "sharpe": oos_sharpe,
            "trade_count": 12,
            "mdd": 0.08,
        },
        "hurdle_fields": {
            "train": {"pass": True},
            "val": {"pass": True},
            "oos": {"pass": False},
        },
        "hard_reject_reasons": dict(hard_reject_reasons or {}),
        "return_streams": {
            "val": [
                {"t": _ts(year, month, day), "v": value}
                for year, month, day, value in (val_stream or [(2026, 1, 15, 0.03)])
            ],
            "oos": [
                {"t": _ts(year, month, day), "v": value}
                for year, month, day, value in (oos_stream or [(2026, 2, 15, oos_return)])
            ],
        },
        "metadata": {},
    }


def _write_slice(
    root: Path,
    relative_dir: str,
    *,
    generated_at: str,
    requested_timeframes: list[str],
    details: list[dict[str, object]],
    peak_rss_mib: float,
    monthly_thresholds: dict[str, dict[str, float]] | None = None,
    promoted_strategy_classes: set[str] | None = None,
) -> Path:
    out_dir = root / relative_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    best_per_strategy = []
    by_strategy: dict[str, dict[str, object]] = {}
    for row in details:
        strategy_class = str(row["strategy_class"])
        current = by_strategy.get(strategy_class)
        if current is None or float((row["val"] or {}).get("sharpe", 0.0)) > float((current["val"] or {}).get("sharpe", 0.0)):
            by_strategy[strategy_class] = row
    for row in by_strategy.values():
        copied = json.loads(json.dumps(row))
        copied["promoted"] = str(copied.get("strategy_class")) in set(promoted_strategy_classes or set())
        copied["validation_hurdle_pass"] = True
        best_per_strategy.append(copied)
    summary = {
        "generated_at": generated_at,
        "windows": {
            "actual_max_timestamp": "2026-03-07T10:00:00+00:00",
            "requested_oos_end_exclusive": "2026-03-09T00:00:00+00:00",
        },
        "execution_profile": {"requested_timeframes": requested_timeframes},
        "evaluated_count": len(details),
        "promoted_count": 0,
        "best_per_strategy": best_per_strategy,
        "monthly_thresholds": monthly_thresholds
        or {
            "2026-01": {"btc_buy_hold_return": 0.01, "threshold": 0.02},
            "2026-02": {"btc_buy_hold_return": 0.01, "threshold": 0.02},
            "2026-03": {"btc_buy_hold_return": 0.04, "threshold": 0.04},
        },
    }
    (out_dir / "exact_window_suite_summary_latest.json").write_text(json.dumps(summary), encoding="utf-8")
    (out_dir / "exact_window_candidate_details_latest.json").write_text(json.dumps(details), encoding="utf-8")
    (out_dir / "exact_window_fail_analysis_latest.json").write_text(
        json.dumps({"counts_by_rejection_reason": []}),
        encoding="utf-8",
    )
    (out_dir / "exact_window_memory_evidence_latest.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "peak_rss_mib": peak_rss_mib,
                "budget_mib": 1024.0,
                "soft_limit_mib": 768.0,
                "hard_limit_mib": 896.0,
                "rss_log_path": str(out_dir / "exact_window_rss_latest.jsonl"),
            }
        ),
        encoding="utf-8",
    )
    return out_dir


def test_write_exact_window_decision_bundle_consolidates_latest_timeframe_slices(tmp_path: Path):
    _write_slice(
        tmp_path,
        "timeframe_1m",
        generated_at="2026-03-08T12:58:19+00:00",
        requested_timeframes=["1m"],
        details=[
            _detail(
                candidate_id="1m-best",
                timeframe="1m",
                strategy_class="VolCompressionVWAPReversionStrategy",
                val_sharpe=1.8,
                val_return=0.05,
                oos_sharpe=-4.2,
                oos_return=-0.03,
                hard_reject_reasons={"oos_sharpe": -4.2},
            ),
            _detail(
                candidate_id="1m-worse",
                timeframe="1m",
                strategy_class="VolCompressionVWAPReversionStrategy",
                val_sharpe=0.8,
                val_return=0.01,
                oos_sharpe=-9.0,
                oos_return=-0.08,
                hard_reject_reasons={"trade_count": 2.0},
            ),
        ],
        peak_rss_mib=128.0,
    )
    _write_slice(
        tmp_path,
        "stale_5m",
        generated_at="2026-03-08T10:00:00+00:00",
        requested_timeframes=["5m"],
        details=[
            _detail(
                candidate_id="5m-stale",
                timeframe="5m",
                strategy_class="PairSpreadZScoreStrategy",
                val_sharpe=0.3,
                val_return=0.00,
                oos_sharpe=-12.0,
                oos_return=-0.12,
                hard_reject_reasons={"max_drawdown": 0.9},
            )
        ],
        peak_rss_mib=64.0,
    )
    new_5m_dir = _write_slice(
        tmp_path,
        "timeframe_5m/resume_5m_20260309/5m",
        generated_at="2026-03-09T09:00:00+00:00",
        requested_timeframes=["5m"],
        details=[
            _detail(
                candidate_id="5m-best",
                timeframe="5m",
                strategy_class="VolCompressionVWAPReversionStrategy",
                val_sharpe=1.4,
                val_return=0.04,
                oos_sharpe=-1.5,
                oos_return=-0.01,
                hard_reject_reasons={"oos_sharpe": -1.5},
            ),
            _detail(
                candidate_id="5m-alt",
                timeframe="5m",
                strategy_class="VolCompressionVWAPReversionStrategy",
                val_sharpe=0.4,
                val_return=0.01,
                oos_sharpe=-2.5,
                oos_return=-0.02,
                hard_reject_reasons={"oos_sharpe": -2.5},
            ),
        ],
        peak_rss_mib=256.0,
    )
    _write_slice(
        tmp_path,
        "timeframes_1h_4h_1d_refresh/followup_high_20260309/1h-4h-1d",
        generated_at="2026-03-09T09:16:00+00:00",
        requested_timeframes=["1h", "4h", "1d"],
        details=[
            _detail(
                candidate_id="1h-best",
                timeframe="1h",
                strategy_class="CompositeTrendStrategy",
                val_sharpe=1.1,
                val_return=0.03,
                oos_sharpe=-0.6,
                oos_return=-0.01,
                hard_reject_reasons={"oos_sharpe": -0.6},
            ),
            _detail(
                candidate_id="4h-best",
                timeframe="4h",
                strategy_class="PairSpreadZScoreStrategy",
                val_sharpe=0.9,
                val_return=0.02,
                oos_sharpe=0.0,
                oos_return=0.0,
                hard_reject_reasons={"pbo": 0.7},
            ),
            _detail(
                candidate_id="1d-best",
                timeframe="1d",
                strategy_class="PerpCrowdingCarryStrategy",
                val_sharpe=0.7,
                val_return=0.01,
                oos_sharpe=-0.1,
                oos_return=-0.005,
                hard_reject_reasons={"trade_count": 3.0},
            ),
        ],
        peak_rss_mib=512.0,
    )

    bundle = write_exact_window_decision_bundle(tmp_path)
    payload = bundle["payload"]

    assert payload["total_evaluated"] == 7
    assert payload["promoted_total"] == 0
    assert payload["btc_beating_candidate_total"] == 5
    assert payload["three_month_two_pct_candidate_total"] == 0
    assert payload["provisional_candidate_total"] == 5
    assert payload["candidate_pool_total"] == 5
    assert payload["next_action"] == "review_candidate_pool_candidates"
    assert payload["timeframes"] == ["1m", "5m", "1h", "4h", "1d"]
    assert payload["common_actual_max_timestamp"] == "2026-03-07T10:00:00+00:00"
    assert payload["max_peak_rss_mib"] == 512.0
    reject_counts = {row["rejection_reason"]: row["count"] for row in payload["reject_counts_all_rows"]}
    assert reject_counts["oos_sharpe"] == 4
    assert reject_counts["trade_count"] == 2
    assert reject_counts["pbo"] == 1
    assert "max_drawdown" not in reject_counts

    timeframe_rows = {row["timeframe"]: row for row in payload["timeframe_rows"]}
    assert timeframe_rows["5m"]["best_row"]["candidate_id"] == "5m-best"
    assert timeframe_rows["5m"]["summary_path"] == str((new_5m_dir / "exact_window_suite_summary_latest.json").resolve())
    assert timeframe_rows["1m"]["monthly_hurdle_outcomes"]["validation"]
    assert timeframe_rows["1m"]["monthly_hurdle_outcomes"]["validation_btc_pass"] is True
    assert timeframe_rows["1m"]["memory_evidence"]["peak_rss_mib"] == 128.0

    assert bundle["json_latest"].exists()
    assert bundle["json_path"].exists()
    assert bundle["strict_json_latest"].exists()
    assert bundle["strict_pass_json_latest"].exists()
    strict_payload = json.loads(bundle["strict_json_latest"].read_text(encoding="utf-8"))
    strict_pass_payload = json.loads(bundle["strict_pass_json_latest"].read_text(encoding="utf-8"))
    assert strict_payload["count"] == 0
    assert strict_pass_payload["count"] == 0
    latest_payload = json.loads(bundle["json_latest"].read_text(encoding="utf-8"))
    assert latest_payload["total_evaluated"] == 7


def test_write_exact_window_decision_bundle_tracks_recent_three_month_two_pct_candidates(tmp_path: Path):
    _write_slice(
        tmp_path,
        "timeframe_1h",
        generated_at="2026-03-09T10:00:00+00:00",
        requested_timeframes=["1h"],
        details=[
            _detail(
                candidate_id="1h-three-month",
                timeframe="1h",
                strategy_class="CompositeTrendStrategy",
                val_sharpe=1.2,
                val_return=0.025,
                oos_sharpe=0.1,
                oos_return=0.047,
                val_stream=[(2026, 1, 15, 0.025)],
                oos_stream=[(2026, 2, 15, 0.026), (2026, 3, 5, 0.021)],
                hard_reject_reasons={"oos_sharpe": 0.1},
            )
        ],
        peak_rss_mib=96.0,
        monthly_thresholds={
            "2026-01": {"btc_buy_hold_return": 0.03, "threshold": 0.03},
            "2026-02": {"btc_buy_hold_return": 0.01, "threshold": 0.02},
            "2026-03": {"btc_buy_hold_return": 0.05, "threshold": 0.05},
        },
    )

    payload = write_exact_window_decision_bundle(tmp_path)["payload"]
    row = payload["timeframe_rows"][0]
    best = row["best_row"]

    assert payload["promoted_total"] == 0
    assert payload["btc_beating_candidate_total"] == 0
    assert payload["three_month_two_pct_candidate_total"] == 1
    assert payload["provisional_candidate_total"] == 1
    assert payload["candidate_pool_total"] == 1
    assert payload["next_action"] == "review_candidate_pool_candidates"
    assert best["three_month_two_pct_candidate"] is True
    assert best["btc_beating_candidate"] is False
    assert row["monthly_hurdle_outcomes"]["recent_three_month_two_pct_pass"] is True


def test_write_exact_window_decision_bundle_penalizes_zero_trade_best_rows(tmp_path: Path):
    zero_trade = _detail(
        candidate_id="4h-zero-trade",
        timeframe="4h",
        strategy_class="PairSpreadZScoreStrategy",
        val_sharpe=3.0,
        val_return=0.03,
        oos_sharpe=0.0,
        oos_return=0.0,
        hard_reject_reasons={"trade_count": 0.0},
    )
    zero_trade["oos"] = {
        "return": 0.0,
        "sharpe": 0.0,
        "trade_count": 0.0,
        "mdd": 0.01,
    }
    live_candidate = _detail(
        candidate_id="4h-live-candidate",
        timeframe="4h",
        strategy_class="PairSpreadZScoreStrategy",
        val_sharpe=2.1,
        val_return=0.015,
        oos_sharpe=1.2,
        oos_return=0.02,
        hard_reject_reasons={"pbo": 0.5},
    )

    _write_slice(
        tmp_path,
        "timeframe_4h",
        generated_at="2026-03-09T10:30:00+00:00",
        requested_timeframes=["4h"],
        details=[zero_trade, live_candidate],
        peak_rss_mib=144.0,
    )

    payload = write_exact_window_decision_bundle(tmp_path)["payload"]
    row = payload["timeframe_rows"][0]

    assert row["timeframe"] == "4h"
    assert row["best_row"]["candidate_id"] == "4h-live-candidate"
    assert row["best_row"]["timeframe_selection_score"] > row["best_row"]["validation_score"]


def test_write_exact_window_decision_bundle_writes_strict_pass_artifact_separately(tmp_path: Path):
    _write_slice(
        tmp_path,
        "timeframe_30m",
        generated_at="2026-03-09T10:00:00+00:00",
        requested_timeframes=["30m"],
        details=[
            _detail(
                candidate_id="strict-30m",
                timeframe="30m",
                strategy_class="CompositeTrendStrategy",
                val_sharpe=1.5,
                val_return=0.05,
                oos_sharpe=2.1,
                oos_return=0.04,
            )
        ],
        peak_rss_mib=120.0,
        promoted_strategy_classes={"CompositeTrendStrategy"},
    )

    bundle = write_exact_window_decision_bundle(tmp_path)
    strict_payload = json.loads(bundle["strict_json_latest"].read_text(encoding="utf-8"))
    strict_pass_payload = json.loads(bundle["strict_pass_json_latest"].read_text(encoding="utf-8"))
    strategy = strict_payload["strategies"][0]
    strict_pass_strategy = strict_pass_payload["strategies"][0]

    assert strict_payload["count"] == 1
    assert strategy["qualification"] == "strict_pass"
    assert strategy["strategy_class"] == "CompositeTrendStrategy"
    assert bundle["strict_md_latest"].exists()
    assert bundle["strict_pass_md_latest"].exists()
    assert strict_pass_payload["count"] == 1
    assert strict_pass_strategy["qualification"] == "strict_pass"
