from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from lumina_quant.core.memory_budget import DEFAULT_EXECUTION_MEMORY_POLICY


@dataclass(frozen=True)
class ResearchFamily:
    family_id: str
    thesis: str
    execution_style: str
    target_timeframes: tuple[str, ...]
    target_universe: tuple[str, ...]
    rationale: str
    preferred_metrics: tuple[str, ...]


@dataclass(frozen=True)
class PipelineStage:
    stage_id: str
    objective: str
    outputs: tuple[str, ...]


@dataclass(frozen=True)
class ExecutionPolicy:
    total_memory_cap_gib: float
    heavy_run_cap_gib: float
    heavy_run_parallelism: int
    light_worker_parallelism: int
    duplicate_policy: str


DEFAULT_METRICS: tuple[str, ...] = (
    "return",
    "cagr",
    "sharpe",
    "sortino",
    "calmar",
    "max_drawdown",
    "volatility",
    "trade_count",
    "trade_density",
    "turnover",
    "win_rate",
    "avg_trade",
    "exposure",
    "deflated_sharpe",
    "pbo",
    "spa_pvalue",
    "benchmark_corr",
    "rolling_sharpe_min",
    "stability",
    "peak_rss_mib",
)

DEFAULT_FAMILIES: tuple[ResearchFamily, ...] = (
    ResearchFamily(
        family_id="crypto-metal-residual-pairs",
        thesis="Crypto and metals can express partially independent mean-reverting dislocations.",
        execution_style="rule_based_pair_spread",
        target_timeframes=("30m", "1h", "4h"),
        target_universe=("BTC/USDT", "ETH/USDT", "BNB/USDT", "XAU/USDT", "XAG/USDT"),
        rationale="Build on current BTC/XAG watchlist progress while keeping execution deterministic.",
        preferred_metrics=(
            "oos_return",
            "oos_sharpe",
            "oos_pbo",
            "oos_trade_count",
            "oos_max_drawdown",
        ),
    ),
    ResearchFamily(
        family_id="sector-dispersion-reversion",
        thesis="Within-crypto sector baskets can mean-revert after short-lived dispersion shocks.",
        execution_style="cross_sectional_residual_reversion",
        target_timeframes=("15m", "30m", "1h"),
        target_universe=("BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "TRX/USDT"),
        rationale="Matches the article's emphasis on many small-capacity ensemble alphas.",
        preferred_metrics=(
            "oos_return",
            "oos_sortino",
            "oos_trade_count",
            "oos_turnover",
            "oos_stability",
        ),
    ),
    ResearchFamily(
        family_id="lead-lag-regime-spillover",
        thesis="Lead/lag effects are worth keeping only when gated by regime and liquidity conditions.",
        execution_style="regime_gated_signal",
        target_timeframes=("5m", "15m", "30m"),
        target_universe=("BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT"),
        rationale="Use LLM-style hypothesis generation but keep final execution rules explicit and auditable.",
        preferred_metrics=(
            "oos_return",
            "oos_sharpe",
            "oos_trade_count",
            "rolling_sharpe_min",
            "oos_max_drawdown",
        ),
    ),
    ResearchFamily(
        family_id="liquidity-shock-reversion",
        thesis="Short-horizon overreactions around liquidity droughts can revert when participation normalizes.",
        execution_style="event_triggered_mean_reversion",
        target_timeframes=("5m", "15m"),
        target_universe=("BTC/USDT", "ETH/USDT", "BNB/USDT"),
        rationale="Targets small-capacity intraday alpha consistent with the screenshots' discussion.",
        preferred_metrics=(
            "oos_return",
            "oos_avg_trade",
            "oos_trade_count",
            "oos_pbo",
            "peak_rss_mib",
        ),
    ),
    ResearchFamily(
        family_id="metals-lag-convergence",
        thesis="Short-history precious-metals pairs need lagged momentum convergence rules instead of wide z-score mean reversion only.",
        execution_style="lagged_momentum_convergence",
        target_timeframes=("4h", "1d"),
        target_universe=("XAU/USDT", "XAG/USDT", "XPT/USDT", "XPD/USDT"),
        rationale="Targets XPT/XPD and XAU/XAG overlap windows where pair z-score can undertrade despite usable directional-relative structure.",
        preferred_metrics=(
            "oos_return",
            "oos_sharpe",
            "oos_trade_count",
            "oos_pbo",
            "oos_max_drawdown",
        ),
    ),
    ResearchFamily(
        family_id="regime-breakout-thrust",
        thesis="Breakout sleeves should survive only when trend and volatility regime filters confirm the thrust.",
        execution_style="regime_filtered_breakout",
        target_timeframes=("30m", "1h"),
        target_universe=("BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "TRX/USDT"),
        rationale="Adds trend-following breadth without relying solely on the existing composite trend sleeve.",
        preferred_metrics=(
            "oos_return",
            "oos_sharpe",
            "oos_calmar",
            "oos_trade_count",
            "oos_max_drawdown",
        ),
    ),
    ResearchFamily(
        family_id="single-asset-zscore-reversion",
        thesis="Simple rolling z-score reversion is still useful as a low-complexity sleeve when crowding and trend overlays are absent.",
        execution_style="single_asset_mean_reversion",
        target_timeframes=("15m", "30m"),
        target_universe=(
            "BTC/USDT",
            "ETH/USDT",
            "BNB/USDT",
            "SOL/USDT",
            "TRX/USDT",
            "XAU/USDT",
            "XAG/USDT",
        ),
        rationale="Provides a low-complexity baseline mean-reversion family for the automated research loop.",
        preferred_metrics=(
            "oos_return",
            "oos_sharpe",
            "oos_sortino",
            "oos_trade_count",
            "oos_max_drawdown",
        ),
    ),
    ResearchFamily(
        family_id="intraday-vwap-reversion",
        thesis="Intraday VWAP dislocations can revert repeatedly when turnover and stop rules are kept bounded.",
        execution_style="vwap_deviation_reversion",
        target_timeframes=("5m", "15m"),
        target_universe=("BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "TRX/USDT", "XAG/USDT"),
        rationale="Adds a classic execution-friendly reversion sleeve aligned with the article's many-small-alpha framing.",
        preferred_metrics=(
            "oos_return",
            "oos_sharpe",
            "oos_trade_count",
            "oos_turnover",
            "oos_max_drawdown",
        ),
    ),
    ResearchFamily(
        family_id="topcap-rotation-relative-momentum",
        thesis="Cross-sectional top-cap rotation can express relative-value dispersion without needing pair-specific hedges.",
        execution_style="cross_sectional_relative_momentum",
        target_timeframes=("1h", "4h"),
        target_universe=("BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "TRX/USDT"),
        rationale="Adds a lower-implementation-risk relative-value sleeve to complement pair-spread and lag-convergence stat-arb families.",
        preferred_metrics=(
            "oos_return",
            "oos_sharpe",
            "oos_trade_count",
            "oos_max_drawdown",
            "oos_pbo",
        ),
    ),
    ResearchFamily(
        family_id="vol-compression-break-reversion",
        thesis="Volatility compression followed by failed breaks can create repeatable short-term mean reversion.",
        execution_style="volatility_compression_reversion",
        target_timeframes=("5m", "15m", "1h"),
        target_universe=("BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XAG/USDT"),
        rationale="Extends existing vol-compression ideas with richer diagnostics instead of one-off tuning.",
        preferred_metrics=(
            "oos_return",
            "oos_sharpe",
            "oos_sortino",
            "oos_trade_count",
            "oos_max_drawdown",
        ),
    ),
    ResearchFamily(
        family_id="regime-conditioned-composite-trend",
        thesis="Trend sleeves remain useful when paired with explicit regime filters and crowding-aware guards.",
        execution_style="composite_trend_with_regime_filter",
        target_timeframes=("30m", "1h", "4h"),
        target_universe=("BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT"),
        rationale="Preserve the current strict anchor while broadening the surrounding ensemble.",
        preferred_metrics=("oos_return", "oos_sharpe", "oos_calmar", "oos_pbo", "oos_trade_count"),
    ),
)

DEFAULT_STAGES: tuple[PipelineStage, ...] = (
    PipelineStage(
        stage_id="research-brief",
        objective="Convert external references into auditable research hypotheses and family tags.",
        outputs=("reference_notes", "family_manifest", "metric_checklist"),
    ),
    PipelineStage(
        stage_id="candidate-materialization",
        objective="Map hypotheses into deterministic strategy families, parameters, and metadata tags.",
        outputs=("candidate_family_specs", "timeframe_priority", "asset_mix_tags"),
    ),
    PipelineStage(
        stage_id="memory-safe-backtest-queue",
        objective="Serialize heavy runs and attach no-rerun signatures plus RSS evidence.",
        outputs=("run_registry", "queue_plan", "memory_budget"),
    ),
    PipelineStage(
        stage_id="validation-reporting",
        objective="Score candidates with richer diagnostics than Sharpe alone and surface them in dashboards.",
        outputs=("rich_metrics", "scenario_panels", "failure_reason_matrix"),
    ),
    PipelineStage(
        stage_id="portfolio-review",
        objective="Build deployment/watchlist scenario comparisons and archive final decisions.",
        outputs=("deployment_scenarios", "watchlist_summary", "git_ready_report"),
    ),
)


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def default_reference_notes(*, article_url: str, image_paths: list[str]) -> list[dict[str, Any]]:
    return [
        {
            "source": "linkedin_article",
            "path": article_url,
            "insights": [
                "LLMs are most useful for signal discovery and hypothesis generation, not direct discretionary execution.",
                "Robust evaluation needs regime breadth, drawdown context, and participation diagnostics rather than headline returns alone.",
                "Real-life deployment benefits from many small-capacity ensemble sleeves rather than a single dominant strategy.",
            ],
        },
        {
            "source": "screenshot_commentary",
            "path": image_paths[0] if image_paths else None,
            "insights": [
                "A human-tuned research harness can resemble an LLM workflow but still rely on classical alpha-research process.",
                "Small-capital crypto stat-arb at mid/high frequency can be plausible when execution burden is modest.",
            ],
        },
        {
            "source": "screenshot_followup",
            "path": image_paths[1] if len(image_paths) > 1 else None,
            "insights": [
                "Skepticism should focus on robustness across regimes, not just a single favorable period.",
                "Drawdown and regime-specific diagnostics are mandatory for evaluating article-inspired claims.",
            ],
        },
    ]


def build_alpha_research_pipeline_manifest(
    *,
    output_root: Path,
    article_url: str,
    image_paths: list[str],
    total_memory_cap_gib: float = DEFAULT_EXECUTION_MEMORY_POLICY.total_memory_cap_gib,
    heavy_run_cap_gib: float = DEFAULT_EXECUTION_MEMORY_POLICY.heavy_run_cap_gib,
    heavy_run_parallelism: int = DEFAULT_EXECUTION_MEMORY_POLICY.heavy_run_parallelism,
    light_worker_parallelism: int = DEFAULT_EXECUTION_MEMORY_POLICY.light_worker_parallelism,
) -> dict[str, Any]:
    policy = ExecutionPolicy(
        total_memory_cap_gib=total_memory_cap_gib,
        heavy_run_cap_gib=heavy_run_cap_gib,
        heavy_run_parallelism=heavy_run_parallelism,
        light_worker_parallelism=light_worker_parallelism,
        duplicate_policy="skip_if_signature_exists_in_exact_window_run_registry_jsonl_unless_forced",
    )
    payload = {
        "generated_at": _now_iso(),
        "schema_version": "1.0",
        "label": "article-inspired llm alpha research pipeline",
        "references": default_reference_notes(article_url=article_url, image_paths=image_paths),
        "thesis": [
            "Use LLM-style research orchestration for hypothesis discovery only.",
            "Keep trading execution rule-based and reproducible.",
            "Expand many partially uncorrelated sleeves and evaluate them with richer metrics plus regime diagnostics.",
        ],
        "families": [asdict(item) for item in DEFAULT_FAMILIES],
        "stages": [asdict(item) for item in DEFAULT_STAGES],
        "metric_checklist": list(DEFAULT_METRICS),
        "execution_policy": asdict(policy),
        "operating_rules": [
            "Count memory against the global across all active sessions/services/workers total, not per process.",
            "Allow at most one heavy backtest run at a time.",
            "Record every finished run in the registry before considering a rerun.",
            "Prefer saved artifacts over recomputation for dashboards and deployment panels.",
        ],
        "recommended_outputs": {
            "pipeline_manifest_json": str(
                (output_root / "alpha_research_pipeline_latest.json").resolve()
            ),
            "pipeline_manifest_md": str(
                (output_root / "alpha_research_pipeline_latest.md").resolve()
            ),
            "signature_registry": str(
                (output_root.parent / "exact_window_run_registry.jsonl").resolve()
            ),
            "canonical_registry_snapshot": str(
                (output_root.parent / "exact_window_backtest_registry_latest.json").resolve()
            ),
            "recovered_run_archive": str(
                (
                    output_root.parent / "followup_status" / "backtest_log_archive_latest.json"
                ).resolve()
            ),
            "deployment_scenarios": str(
                (
                    output_root.parent / "followup_status" / "deployment_scenarios_latest.json"
                ).resolve()
            ),
        },
        "suggested_work_items": {
            "light_parallel": [
                "expand candidate metadata and research-family tags",
                "surface rich metrics and registry status in dashboard",
                "build saved deployment / scenario artifact panels",
            ],
            "heavy_serial": [
                "bounded exact-window reruns for newly added families",
                "mixed/metals follow-up reruns under one-active-job policy",
            ],
        },
    }
    return payload


def write_alpha_research_pipeline_manifest(
    *,
    report_root: Path,
    article_url: str,
    image_paths: list[str],
    total_memory_cap_gib: float = DEFAULT_EXECUTION_MEMORY_POLICY.total_memory_cap_gib,
    heavy_run_cap_gib: float = DEFAULT_EXECUTION_MEMORY_POLICY.heavy_run_cap_gib,
) -> dict[str, Any]:
    output_root = report_root / "pipeline"
    output_root.mkdir(parents=True, exist_ok=True)
    payload = build_alpha_research_pipeline_manifest(
        output_root=output_root,
        article_url=article_url,
        image_paths=image_paths,
        total_memory_cap_gib=total_memory_cap_gib,
        heavy_run_cap_gib=heavy_run_cap_gib,
    )
    json_path = output_root / "alpha_research_pipeline_latest.json"
    md_path = output_root / "alpha_research_pipeline_latest.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    lines = [
        "# article-inspired llm alpha research pipeline",
        "",
        f"- generated_at: `{payload['generated_at']}`",
        f"- total_memory_cap_gib: `{payload['execution_policy']['total_memory_cap_gib']}`",
        f"- heavy_run_cap_gib: `{payload['execution_policy']['heavy_run_cap_gib']}`",
        f"- heavy_run_parallelism: `{payload['execution_policy']['heavy_run_parallelism']}`",
        "",
        "## thesis",
    ]
    lines.extend(f"- {item}" for item in payload["thesis"])
    lines.extend(["", "## strategy families"])
    for family in payload["families"]:
        lines.append(
            f"- `{family['family_id']}` | exec={family['execution_style']} | tf={', '.join(family['target_timeframes'])} | rationale={family['rationale']}"
        )
    lines.extend(["", "## operating rules"])
    lines.extend(f"- {item}" for item in payload["operating_rules"])
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "json_path": str(json_path.resolve()),
        "md_path": str(md_path.resolve()),
        "family_count": len(payload["families"]),
        "metric_count": len(payload["metric_checklist"]),
    }


__all__ = [
    "DEFAULT_FAMILIES",
    "DEFAULT_METRICS",
    "DEFAULT_STAGES",
    "build_alpha_research_pipeline_manifest",
    "default_reference_notes",
    "write_alpha_research_pipeline_manifest",
]
