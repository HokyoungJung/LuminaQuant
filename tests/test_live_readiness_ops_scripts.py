from __future__ import annotations

import importlib.util
import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import lumina_quant.live.readiness_policy as readiness_policy

ROOT = Path(__file__).resolve().parents[1]


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


PREFLIGHT = _load(
    ROOT / "scripts" / "ops" / "live_readiness_preflight.py", "live_readiness_preflight"
)
STOP = _load(ROOT / "scripts" / "ops" / "request_live_stop.py", "request_live_stop")
SLIPPAGE = _load(ROOT / "scripts" / "ops" / "summarize_fill_slippage.py", "summarize_fill_slippage")


def test_live_readiness_preflight_reports_ready_for_paper(tmp_path: Path) -> None:
    fresh_cutoff = (
        (datetime.now(UTC) - timedelta(minutes=5))
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "storage:",
                '  postgres_dsn: "postgresql://demo"',
                "live:",
                '  mode: "paper"',
                "  testnet: true",
                "  require_real_enable_flag: true",
            ]
        ),
        encoding="utf-8",
    )
    refresh = tmp_path / "refresh.json"
    refresh.write_text(
        json.dumps(
            {
                "status": "completed",
                "collection_cutoff_utc": fresh_cutoff,
                "feature_results": [{"last_timestamp_utc": fresh_cutoff}],
            }
        ),
        encoding="utf-8",
    )
    decision = tmp_path / "decision.json"
    decision.write_text(json.dumps({"decision": "keep_incumbent"}), encoding="utf-8")

    payload = PREFLIGHT.build_preflight_payload(
        config_path=config_path,
        refresh_json=refresh,
        decision_json=decision,
        stale_minutes=10_000,
    )

    assert payload["checks"]["paper_mode"] is True
    assert payload["checks"]["testnet"] is True
    assert payload["checks"]["postgres_dsn_present"] is True
    assert payload["status"]["ready_for_paper"] is True
    assert payload["status"]["ready_for_real"] is False


def test_live_readiness_preflight_reports_ready_for_real(monkeypatch, tmp_path: Path) -> None:
    fresh_cutoff = (
        (datetime.now(UTC) - timedelta(minutes=5))
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "storage:",
                '  postgres_dsn: "postgresql://demo"',
                "live:",
                '  mode: "real"',
                "  testnet: false",
                "  require_real_enable_flag: true",
            ]
        ),
        encoding="utf-8",
    )
    refresh = tmp_path / "refresh.json"
    refresh.write_text(
        json.dumps(
            {
                "status": "completed",
                "collection_cutoff_utc": fresh_cutoff,
                "feature_results": [{"last_timestamp_utc": fresh_cutoff}],
            }
        ),
        encoding="utf-8",
    )
    decision = tmp_path / "decision.json"
    decision.write_text(json.dumps({"decision": "keep_incumbent"}), encoding="utf-8")
    monkeypatch.setenv("LUMINA_ENABLE_LIVE_REAL", "true")

    payload = PREFLIGHT.build_preflight_payload(
        config_path=config_path,
        refresh_json=refresh,
        decision_json=decision,
        stale_minutes=10_000,
    )

    assert payload["checks"]["paper_mode"] is False
    assert payload["checks"]["real_enable_env"] is True
    assert payload["status"]["ready_for_paper"] is False
    assert payload["status"]["ready_for_real"] is True
    assert payload["recommended_action"] == "real_run_allowed"


def test_live_readiness_preflight_accepts_explicit_promoted_candidate(tmp_path: Path) -> None:
    fresh_cutoff = (
        (datetime.now(UTC) - timedelta(minutes=5))
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "storage:",
                '  postgres_dsn: "postgresql://demo"',
                "live:",
                '  mode: "paper"',
                "  testnet: true",
                "  require_real_enable_flag: true",
            ]
        ),
        encoding="utf-8",
    )
    refresh = tmp_path / "refresh.json"
    refresh.write_text(
        json.dumps(
            {
                "status": "completed",
                "collection_cutoff_utc": fresh_cutoff,
                "feature_results": [{"last_timestamp_utc": fresh_cutoff}],
            }
        ),
        encoding="utf-8",
    )
    decision = tmp_path / "decision.json"
    decision.write_text(
        json.dumps(
            {
                "decision": "selected_live_mode",
                "selected_mode": "MovingAverageCrossStrategy",
                "candidate_key": "moving_average_cross",
            }
        ),
        encoding="utf-8",
    )

    payload = PREFLIGHT.build_preflight_payload(
        config_path=config_path,
        refresh_json=refresh,
        decision_json=decision,
        stale_minutes=10_000,
    )

    assert payload["checks"]["decision_allows_live_start"] is True
    assert payload["checks"]["decision_promote_candidate"] is True
    assert payload["checks"]["decision_runtime_compatible"] is True
    assert payload["status"]["ready_for_paper"] is True


def test_alpha_zoo_optuna_hybrid_real_mode_is_vetoed_by_artifact_flags(
    monkeypatch, tmp_path: Path
) -> None:
    fresh_cutoff = (
        (datetime.now(UTC) - timedelta(minutes=5))
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "storage:",
                '  postgres_dsn: "postgresql://demo"',
                "live:",
                '  mode: "real"',
                "  testnet: false",
                "  require_real_enable_flag: true",
            ]
        ),
        encoding="utf-8",
    )
    refresh = tmp_path / "refresh.json"
    refresh.write_text(
        json.dumps(
            {
                "status": "completed",
                "collection_cutoff_utc": fresh_cutoff,
                "feature_results": [{"last_timestamp_utc": fresh_cutoff}],
            }
        ),
        encoding="utf-8",
    )
    decision = tmp_path / "decision.json"
    decision.write_text(
        json.dumps(
            {
                "decision": "selected_live_mode",
                "selected_mode": "alpha_zoo_integer_leverage_optuna_hybrid",
                "strategy_name": "AlphaZooOptunaHybridLiveStrategy",
                "strategy_params": {
                    "optuna_hybrid_artifact_path": str(
                        ROOT
                        / "var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/"
                        "alpha_zoo_integer_leverage_optuna_hybrid_decision_20260524/"
                        "alpha_zoo_integer_leverage_optuna_hybrid_decision_latest.json"
                    ),
                    "integer_portfolio_artifact_path": str(
                        ROOT
                        / "var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/"
                        "alpha_zoo_corr_integer_leverage_portfolio_20260524/"
                        "alpha_zoo_corr_integer_leverage_portfolio_latest.json"
                    ),
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("LUMINA_ENABLE_LIVE_REAL", "true")

    payload = PREFLIGHT.build_preflight_payload(
        config_path=config_path,
        refresh_json=refresh,
        decision_json=decision,
        stale_minutes=10_000,
    )

    assert payload["checks"]["decision_runtime_compatible"] is True
    assert payload["checks"]["artifact_real_money_veto"] is True
    assert payload["checks"]["artifact_paper_testnet_only"] is True
    assert payload["checks"]["artifact_ready_for_real"] is False
    assert payload["status"]["ready_for_real"] is False
    assert payload["recommended_action"] == "block_until_preflight_gaps_closed"


def test_alpha_zoo_optuna_hybrid_paper_mode_can_pass_with_testnet(tmp_path: Path) -> None:
    fresh_cutoff = (
        (datetime.now(UTC) - timedelta(minutes=5))
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "storage:",
                '  postgres_dsn: "postgresql://demo"',
                "live:",
                '  mode: "paper"',
                "  testnet: true",
                "  require_real_enable_flag: true",
            ]
        ),
        encoding="utf-8",
    )
    refresh = tmp_path / "refresh.json"
    refresh.write_text(
        json.dumps(
            {
                "status": "completed",
                "collection_cutoff_utc": fresh_cutoff,
                "feature_results": [{"last_timestamp_utc": fresh_cutoff}],
            }
        ),
        encoding="utf-8",
    )
    decision = tmp_path / "decision.json"
    decision.write_text(
        json.dumps(
            {
                "decision": "selected_live_mode",
                "selected_live_mode": "alpha_zoo_integer_leverage_optuna_hybrid",
                "strategy_name": "AlphaZooOptunaHybridLiveStrategy",
            }
        ),
        encoding="utf-8",
    )

    payload = PREFLIGHT.build_preflight_payload(
        config_path=config_path,
        refresh_json=refresh,
        decision_json=decision,
        stale_minutes=10_000,
    )

    assert payload["checks"]["artifact_real_money_veto"] is True
    assert payload["status"]["ready_for_paper"] is True
    assert payload["status"]["ready_for_real"] is False


def test_live_readiness_preflight_honors_runtime_env_mode_override(
    monkeypatch, tmp_path: Path
) -> None:
    fresh_cutoff = (
        (datetime.now(UTC) - timedelta(minutes=5))
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "storage:",
                '  postgres_dsn: "postgresql://demo"',
                "live:",
                '  mode: "paper"',
                "  testnet: true",
                "  require_real_enable_flag: true",
            ]
        ),
        encoding="utf-8",
    )
    refresh = tmp_path / "refresh.json"
    refresh.write_text(
        json.dumps(
            {
                "status": "completed",
                "collection_cutoff_utc": fresh_cutoff,
                "feature_results": [{"last_timestamp_utc": fresh_cutoff}],
            }
        ),
        encoding="utf-8",
    )
    decision = tmp_path / "decision.json"
    decision.write_text(json.dumps({"decision": "keep_incumbent"}), encoding="utf-8")
    monkeypatch.setenv("LQ__LIVE__MODE", "real")
    monkeypatch.setenv("LQ__LIVE__TESTNET", "false")
    monkeypatch.setenv("LUMINA_ENABLE_LIVE_REAL", "true")

    payload = PREFLIGHT.build_preflight_payload(
        config_path=config_path,
        refresh_json=refresh,
        decision_json=decision,
        stale_minutes=10_000,
    )

    assert payload["checks"]["mode"] == "real"
    assert payload["status"]["ready_for_real"] is True


def test_live_readiness_preflight_falls_back_to_latest_worktree_decision_artifact(
    monkeypatch, tmp_path: Path
) -> None:
    fresh_cutoff = (
        (datetime.now(UTC) - timedelta(minutes=5))
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "storage:",
                '  postgres_dsn: "postgresql://demo"',
                "live:",
                '  mode: "paper"',
                "  testnet: true",
                "  require_real_enable_flag: true",
            ]
        ),
        encoding="utf-8",
    )
    refresh = tmp_path / "refresh.json"
    refresh.write_text(
        json.dumps(
            {
                "status": "completed",
                "collection_cutoff_utc": fresh_cutoff,
                "feature_results": [{"last_timestamp_utc": fresh_cutoff}],
            }
        ),
        encoding="utf-8",
    )
    decision_rel = Path("var/reports/exact_window_backtests/followup_status/decision.json")
    worktree_decision = (
        tmp_path / ".omx" / "team" / "demo" / "worktrees" / "worker-1" / decision_rel
    )
    worktree_decision.parent.mkdir(parents=True, exist_ok=True)
    worktree_decision.write_text(json.dumps({"decision": "keep_incumbent"}), encoding="utf-8")
    monkeypatch.setattr(readiness_policy, "REPO_ROOT", tmp_path)

    payload = PREFLIGHT.build_preflight_payload(
        config_path=config_path,
        refresh_json=refresh,
        decision_json=decision_rel,
        stale_minutes=10_000,
    )

    assert payload["status"]["ready_for_paper"] is True
    assert payload["decision_json"] == str(worktree_decision.resolve())


def test_request_live_stop_touches_stop_file(tmp_path: Path) -> None:
    stop_path = tmp_path / "lq.stop"
    assert not stop_path.exists()
    code = STOP.main(["--stop-file", str(stop_path)])
    assert code == 0
    assert stop_path.exists()


def test_summarize_fill_slippage_from_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "fills.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "symbol": "BTC/USDT",
                        "side": "BUY",
                        "fill_price": 101.0,
                        "metadata": {"reference_price": 100.0},
                    }
                ),
                json.dumps(
                    {
                        "symbol": "BTC/USDT",
                        "side": "SELL",
                        "fill_price": 99.0,
                        "metadata": {"reference_price": 100.0, "timeout_flag": True},
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )

    rows = SLIPPAGE._read_jsonl(path)
    summary = SLIPPAGE.build_summary(rows)

    assert summary["overall"]["count"] == 2
    assert summary["overall"]["timeout_count"] == 1
    assert summary["by_symbol"]["BTC/USDT"]["median_slippage_bps"] == 100.0
