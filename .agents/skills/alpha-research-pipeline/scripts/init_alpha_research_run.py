#!/usr/bin/env python3
"""Create a LuminaQuant alpha-discovery research run skeleton."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DEFAULT_REPO = Path("/home/hoky/Quants-agent/LuminaQuant")
SKILL_DIR = Path(__file__).resolve().parents[1]
AUDIT_JSON = SKILL_DIR / "references" / "scientific-agent-skills-audit.json"

METRICS = [
    "return_10bps",
    "return_15bps",
    "return_20bps",
    "max_drawdown",
    "sharpe",
    "sortino",
    "calmar",
    "turnover",
    "gross_exposure",
    "trade_count",
    "win_rate",
    "benchmark_correlation",
    "rolling_stability",
]

FAIL_CLOSED = [
    "missing_source_artifact",
    "sha_mismatch",
    "data_gap",
    "stale_market_data",
    "exchange_status_mismatch",
    "oos_contamination",
    "gross_cap_breach",
    "turnover_cap_breach",
    "real_money_enabled",
    "cost_stress_failure",
    "lookahead_or_survivorship_risk",
]

CANDIDATE_TEMPLATES: list[dict[str, Any]] = [
    {
        "slug": "vol-managed-tsmom-crash-filter",
        "lane": "literature_time_series",
        "hypothesis": "Volatility-managed time-series momentum with drawdown/crash-state suppression improves net 20bps performance versus raw momentum.",
        "mechanism": "Trend persistence survives costs only when exposure is reduced during realized-volatility spikes and adverse crash states.",
        "source_skill_inspiration": [
            "literature-review",
            "paper-lookup",
            "statsmodels",
            "statistical-analysis",
        ],
        "features": [
            "multi-horizon returns",
            "realized volatility",
            "drawdown state",
            "cost-adjusted trend strength",
        ],
        "parameters": {
            "lookbacks": [60, 240, 1440],
            "vol_halflife": [60, 240],
            "crash_cutoff": "train_selected",
        },
        "disconfirming_evidence": [
            "edge disappears at 20bps",
            "only one symbol contributes",
            "drawdown filter selected on OOS",
        ],
    },
    {
        "slug": "cross-timeframe-alignment",
        "lane": "time_series",
        "hypothesis": "Trades taken only when 1m/15m/1h states align have lower churn and better net expectancy than single-timeframe signals.",
        "mechanism": "Multi-scale agreement filters microstructure noise and avoids whipsaw regimes.",
        "source_skill_inspiration": ["aeon", "experimental-design", "polars"],
        "features": [
            "1m state",
            "15m state",
            "1h state",
            "alignment score",
            "disagreement duration",
        ],
        "parameters": {"state_windows": [20, 80, 240], "min_alignment": [0.5, 0.75, 1.0]},
        "disconfirming_evidence": [
            "turnover unchanged",
            "missed-trend opportunity cost dominates",
            "alignment threshold overfits validation",
        ],
    },
    {
        "slug": "vol-squeeze-breakout-confirmation",
        "lane": "time_series_microstructure",
        "hypothesis": "Low-volatility squeeze followed by volume/trade-intensity confirmation predicts continuation after costs.",
        "mechanism": "Compressed volatility plus participation expansion marks inventory imbalance resolution.",
        "source_skill_inspiration": ["aeon", "exploratory-data-analysis", "statistical-analysis"],
        "features": [
            "realized volatility percentile",
            "range compression",
            "volume z-score",
            "trade intensity",
            "breakout direction",
        ],
        "parameters": {"squeeze_quantile": [0.1, 0.2], "confirmation_window": [5, 20, 60]},
        "disconfirming_evidence": [
            "false breakouts dominate",
            "spread/slippage removes edge",
            "signal only works in one market month",
        ],
    },
    {
        "slug": "anomaly-reversion-band",
        "lane": "time_series_anomaly",
        "hypothesis": "Large deviations outside forecast/rolling bands mean-revert when liquidity is normal but continue when liquidity is stressed.",
        "mechanism": "Separate temporary dislocations from informed jumps using liquidity state.",
        "source_skill_inspiration": ["aeon", "timesfm-forecasting", "statsmodels", "shap"],
        "features": [
            "forecast residual",
            "rolling z-score",
            "liquidity state",
            "spread proxy",
            "post-event drift",
        ],
        "parameters": {"band": [2.0, 2.5, 3.0], "liquidity_gate": "train_selected"},
        "disconfirming_evidence": [
            "tail losses dominate",
            "forecast model leaks future",
            "liquidity gate uses unavailable live data",
        ],
    },
    {
        "slug": "motif-pre-breakout",
        "lane": "time_series_similarity",
        "hypothesis": "Recurring pre-breakout motifs in returns/volume state forecast short-horizon continuation more reliably than scalar indicators.",
        "mechanism": "Micro-patterns encode order-flow build-up before price expansion.",
        "source_skill_inspiration": ["aeon", "scikit-learn", "scientific-critical-thinking"],
        "features": [
            "shapelet/motif distance",
            "volume motif",
            "volatility motif",
            "future breakout label",
        ],
        "parameters": {"motif_length": [20, 60, 120], "k_neighbors": [5, 20, 50]},
        "disconfirming_evidence": [
            "nearest-neighbor lookup crosses validation boundary",
            "motif labels unstable",
            "candidate budget correction rejects result",
        ],
    },
    {
        "slug": "cointegration-residual-reversion",
        "lane": "econometrics_pairs",
        "hypothesis": "Stable cointegration residuals across selected crypto pairs mean-revert net of borrow/funding/spread costs.",
        "mechanism": "Shared risk factor dislocations close when residual z-score is extreme and beta is stable.",
        "source_skill_inspiration": ["statsmodels", "statistical-analysis", "experimental-design"],
        "features": [
            "hedge ratio",
            "residual z-score",
            "half-life",
            "beta stability",
            "funding differential",
        ],
        "parameters": {
            "formation_window": [1440, 4320],
            "entry_z": [1.5, 2.0, 2.5],
            "exit_z": [0.0, 0.5],
        },
        "disconfirming_evidence": [
            "cointegration selected using future",
            "residual half-life too slow",
            "spread/funding flips expectancy",
        ],
    },
    {
        "slug": "var-lead-lag-rotation",
        "lane": "econometrics_lead_lag",
        "hypothesis": "Lead assets with statistically stable lagged predictive relation can time laggard exposure under embargoed walk-forward evaluation.",
        "mechanism": "Information diffuses across correlated crypto assets at short horizons.",
        "source_skill_inspiration": ["statsmodels", "networkx", "statistical-analysis"],
        "features": [
            "lagged returns",
            "VAR coefficients",
            "lead-lag p-values",
            "edge stability",
            "network centrality",
        ],
        "parameters": {"lags": [1, 5, 20], "stability_window": [1440, 4320]},
        "disconfirming_evidence": [
            "relationship vanishes out-of-sample",
            "multiple testing explains significance",
            "latency/cost exceeds edge",
        ],
    },
    {
        "slug": "rolling-beta-residual-momentum",
        "lane": "econometrics_factor",
        "hypothesis": "Residual momentum after removing market/beta exposure is more diversifying than raw momentum.",
        "mechanism": "Idiosyncratic drift persists while market beta noise is hedged out.",
        "source_skill_inspiration": ["statsmodels", "scikit-learn", "shap"],
        "features": ["market beta", "residual return", "residual volatility", "factor exposure"],
        "parameters": {"beta_window": [240, 1440], "residual_momentum_window": [60, 240]},
        "disconfirming_evidence": [
            "residualization increases turnover",
            "beta estimate unstable",
            "same market crash exposure remains",
        ],
    },
    {
        "slug": "meta-selector-quality-gate",
        "lane": "ml_meta_selection",
        "hypothesis": "A leakage-audited meta-selector can select when existing alpha_zoo sleeves are likely to work, improving net portfolio stability.",
        "mechanism": "Regime features predict sleeve-specific edge persistence and suppress weak contexts.",
        "source_skill_inspiration": [
            "scikit-learn",
            "shap",
            "scientific-critical-thinking",
            "experimental-design",
        ],
        "features": [
            "regime state",
            "recent sleeve PnL",
            "volatility",
            "correlation crowding",
            "turnover",
        ],
        "parameters": {
            "model_family": ["logistic", "random_forest", "hist_gradient_boosting"],
            "threshold": "train_validation_only",
        },
        "disconfirming_evidence": [
            "SHAP highlights future-return leakage",
            "selector only improves validation",
            "suppression misses all large winners",
        ],
    },
    {
        "slug": "negative-control-leakage-sentinel",
        "lane": "validation_sentinel",
        "hypothesis": "Negative-control shuffled/lag-broken features should fail; if they pass, the pipeline is leaking or overfitting.",
        "mechanism": "A sentinel candidate protects the alpha factory from false discovery.",
        "source_skill_inspiration": [
            "scientific-critical-thinking",
            "statistical-analysis",
            "scikit-learn",
        ],
        "features": ["shuffled labels", "future-shifted banlist", "lag-broken controls"],
        "parameters": {"shuffle_seed_count": 10, "lag_breaks": [1, 5, 20]},
        "disconfirming_evidence": [
            "negative controls pass gates",
            "candidate registry lacks correction for tested variants",
        ],
    },
    {
        "slug": "asset-graph-centrality-momentum",
        "lane": "graph_alpha",
        "hypothesis": "Momentum from central/leader assets propagates to peripheral assets with lower immediate crowding.",
        "mechanism": "Correlation/lead-lag network topology captures delayed risk-on/risk-off transmission.",
        "source_skill_inspiration": ["networkx", "statsmodels", "scientific-visualization"],
        "features": [
            "lead-lag graph",
            "pagerank",
            "community",
            "leader momentum",
            "peripheral lag",
        ],
        "parameters": {"edge_threshold": "train_selected", "rebalance_window": [240, 1440]},
        "disconfirming_evidence": [
            "graph instability",
            "edge threshold selected on OOS",
            "leader signal indistinguishable from market beta",
        ],
    },
    {
        "slug": "correlation-cluster-rotation",
        "lane": "graph_portfolio",
        "hypothesis": "Within-cluster winner/loser rotation improves diversification without increasing gross exposure.",
        "mechanism": "Clusters share risk but differ in short-term relative strength and liquidity.",
        "source_skill_inspiration": ["networkx", "umap-learn", "pymoo"],
        "features": [
            "correlation cluster",
            "cluster momentum",
            "within-cluster rank",
            "cluster volatility",
        ],
        "parameters": {"cluster_window": [1440, 4320], "max_per_cluster": [1, 2, 3]},
        "disconfirming_evidence": [
            "cluster assignment unstable",
            "diversification vanishes in stress",
            "turnover too high",
        ],
    },
    {
        "slug": "us-liquidity-crypto-beta-filter",
        "lane": "macro_external",
        "hypothesis": "U.S. liquidity/rates proxies identify regimes where crypto beta/trend exposure should be reduced or expanded.",
        "mechanism": "Liquidity shocks modulate speculative asset risk appetite and leverage constraints.",
        "source_skill_inspiration": [
            "database-lookup",
            "usfiscaldata",
            "statsmodels",
            "literature-review",
        ],
        "features": [
            "Treasury cash balance",
            "rates",
            "DXY/proxy",
            "liquidity impulse",
            "crypto beta",
        ],
        "parameters": {"macro_lag_days": [1, 3, 7], "risk_multiplier": [0.0, 0.5, 1.0]},
        "disconfirming_evidence": [
            "macro data timestamp unavailable live",
            "signal frequency mismatch",
            "crypto beta filter selected on current OOS",
        ],
    },
    {
        "slug": "rates-dollar-risk-off-guard",
        "lane": "macro_external",
        "hypothesis": "Rates/dollar risk-off regimes reduce long-biased crypto alpha expectancy and should gate exposure.",
        "mechanism": "Macro tightening increases discount-rate/liquidity pressure on high-beta assets.",
        "source_skill_inspiration": ["database-lookup", "research-lookup", "statistical-analysis"],
        "features": [
            "rate level",
            "rate change",
            "USD proxy",
            "risk-off state",
            "trend sleeve exposure",
        ],
        "parameters": {"macro_window": [5, 20, 60], "gate_strength": [0.25, 0.5, 0.75]},
        "disconfirming_evidence": [
            "macro proxy unavailable or stale",
            "lag choice overfit",
            "gating reduces convex winners",
        ],
    },
    {
        "slug": "book-imbalance-state",
        "lane": "microstructure",
        "hypothesis": "Order-book imbalance state improves short-horizon entry timing for existing sleeves when feature_points are available.",
        "mechanism": "Near-touch depth imbalance reveals short-lived pressure before price update.",
        "source_skill_inspiration": ["polars", "aeon", "simpy", "scientific-critical-thinking"],
        "features": [
            "book imbalance",
            "spread",
            "depth",
            "last trade direction",
            "latency/freshness",
        ],
        "parameters": {"imbalance_window": [1, 5, 20], "freshness_ms": "config_bound"},
        "disconfirming_evidence": [
            "feature unavailable in live path",
            "latency invalidates fill",
            "post-only/funding costs erase edge",
        ],
    },
    {
        "slug": "trade-intensity-reversal",
        "lane": "microstructure",
        "hypothesis": "Extreme short-window trade intensity after a price jump predicts temporary reversal unless higher timeframe trend confirms.",
        "mechanism": "Liquidity-taking exhaustion mean-reverts when not supported by broader trend.",
        "source_skill_inspiration": ["exploratory-data-analysis", "statsmodels", "aeon"],
        "features": ["aggtrade intensity", "signed volume", "jump size", "higher timeframe trend"],
        "parameters": {"intensity_z": [2.0, 3.0], "reversion_horizon": [5, 20, 60]},
        "disconfirming_evidence": [
            "unavailable signed flow",
            "extreme events continue not revert",
            "fees dominate scalps",
        ],
    },
    {
        "slug": "cost-aware-no-trade-filter",
        "lane": "execution_cost",
        "hypothesis": "Explicit no-trade zones based on expected edge versus spread/slippage reduce turnover and improve 20bps net return.",
        "mechanism": "Small forecast edges are consumed by costs; abstention improves realized edge quality.",
        "source_skill_inspiration": ["pymoo", "simpy", "statistical-analysis"],
        "features": ["expected edge", "spread proxy", "slippage proxy", "turnover", "fill quality"],
        "parameters": {"edge_cost_multiplier": [1.0, 1.5, 2.0], "cooldown": [0, 5, 20]},
        "disconfirming_evidence": [
            "filter selected on cost-stress OOS",
            "turnover reduction misses winners",
            "cost estimate stale",
        ],
    },
    {
        "slug": "pareto-sleeve-blend",
        "lane": "ensemble_portfolio",
        "hypothesis": "Pareto-selected sleeve blends improve return/MDD/turnover trade-off versus single best sleeve.",
        "mechanism": "Weakly correlated alphas compound better when weights penalize drawdown, turnover, and fragility.",
        "source_skill_inspiration": ["pymoo", "networkx", "statistical-analysis"],
        "features": ["sleeve returns", "sleeve MDD", "correlation", "turnover", "gross"],
        "parameters": {
            "weight_grid": "train_validation_only",
            "objectives": ["return", "mdd", "turnover", "stability"],
        },
        "disconfirming_evidence": [
            "same sleeve dominates all PnL",
            "weights chosen from current OOS",
            "gross/turnover cap breach",
        ],
    },
    {
        "slug": "bayesian-regime-sizer",
        "lane": "bayesian_risk",
        "hypothesis": "Sizing by posterior regime/edge uncertainty reduces drawdown while retaining most positive expectancy.",
        "mechanism": "Uncertain edge estimates should receive lower capital until evidence accumulates.",
        "source_skill_inspiration": ["pymc", "statistical-power", "scientific-critical-thinking"],
        "features": ["posterior edge", "regime probability", "uncertainty", "drawdown state"],
        "parameters": {"credible_edge_threshold": [0.55, 0.65], "max_size_multiplier": [0.5, 1.0]},
        "disconfirming_evidence": [
            "posterior calibrated on OOS",
            "uncertainty model too slow",
            "sizing adds no benefit over vol target",
        ],
    },
    {
        "slug": "signal-decay-hazard-exit",
        "lane": "survival_exit",
        "hypothesis": "Time-to-decay hazard models improve exits for breakout/trend candidates versus fixed holding windows.",
        "mechanism": "Alpha half-life varies by regime and signal age; hazard-aware exits cut stale exposure.",
        "source_skill_inspiration": ["scikit-survival", "statsmodels", "statistical-analysis"],
        "features": ["signal age", "unrealized PnL", "regime", "volatility", "hazard label"],
        "parameters": {"max_age": [20, 60, 240], "hazard_cutoff": "train_validation_only"},
        "disconfirming_evidence": [
            "censoring mishandled",
            "exit rule overfits",
            "holding-period reduction only lowers winners",
        ],
    },
    {
        "slug": "weekday-session-seasonality",
        "lane": "seasonality",
        "hypothesis": "Intraday/session/weekday seasonality interacts with volatility state to create low-turnover exposure gates.",
        "mechanism": "Liquidity and participant mix vary predictably across sessions, changing cost-adjusted alpha.",
        "source_skill_inspiration": [
            "statistical-analysis",
            "exploratory-data-analysis",
            "experimental-design",
        ],
        "features": ["hour", "weekday", "session", "realized volatility", "turnover"],
        "parameters": {"session_map": "pre_registered", "min_trade_count": "power_checked"},
        "disconfirming_evidence": [
            "sample size too low",
            "DST/timezone bug",
            "seasonality disappears after multiple-testing correction",
        ],
    },
    {
        "slug": "funding-basis-extreme",
        "lane": "carry_funding",
        "hypothesis": "Extreme funding/basis states predict reversal or continuation depending on trend and crowding context.",
        "mechanism": "Crowded leverage creates carry pressure and liquidation risk; context determines sign.",
        "source_skill_inspiration": [
            "statsmodels",
            "scientific-brainstorming",
            "statistical-analysis",
        ],
        "features": ["funding rate", "basis proxy", "trend", "volatility", "crowding proxy"],
        "parameters": {
            "funding_percentile": [0.05, 0.1, 0.9, 0.95],
            "context_gate": "train_selected",
        },
        "disconfirming_evidence": [
            "funding data missing",
            "settlement timing leak",
            "borrow/funding costs erase return",
        ],
    },
    {
        "slug": "liquidity-shock-rebound",
        "lane": "liquidity_event",
        "hypothesis": "Large liquidity shocks followed by spread normalization produce short rebound windows.",
        "mechanism": "Temporary liquidity withdrawal causes overshoot that reverts as depth returns.",
        "source_skill_inspiration": ["aeon", "simpy", "scientific-critical-thinking"],
        "features": [
            "spread shock",
            "depth drop",
            "volume burst",
            "normalization speed",
            "rebound return",
        ],
        "parameters": {"shock_z": [2.0, 3.0], "normalization_window": [5, 20]},
        "disconfirming_evidence": [
            "shock data not live-fresh",
            "rebound unavailable after realistic fills",
            "tail losses too large",
        ],
    },
    {
        "slug": "return-dispersion-rotation",
        "lane": "cross_sectional",
        "hypothesis": "High cross-sectional return dispersion identifies regimes where rotation beats market beta exposure.",
        "mechanism": "Dispersion signals asset-specific flows and relative opportunities.",
        "source_skill_inspiration": ["polars", "statsmodels", "pymoo"],
        "features": ["cross-sectional dispersion", "rank momentum", "rank reversal", "market beta"],
        "parameters": {"rank_window": [60, 240], "dispersion_gate": "train_selected"},
        "disconfirming_evidence": [
            "universe changes drive result",
            "rank signal too costly",
            "beta explains PnL",
        ],
    },
    {
        "slug": "entropy-crowding-state",
        "lane": "regime_crowding",
        "hypothesis": "Low entropy/crowded return states precede higher reversal risk and should downweight trend sleeves.",
        "mechanism": "Crowded one-way behavior becomes fragile when flow diversity collapses.",
        "source_skill_inspiration": ["scientific-brainstorming", "statistical-analysis", "shap"],
        "features": [
            "return entropy",
            "cross-asset concentration",
            "correlation",
            "trend exposure",
        ],
        "parameters": {"entropy_window": [60, 240, 1440], "crowding_cutoff": "train_selected"},
        "disconfirming_evidence": [
            "entropy is proxy for volatility only",
            "cutoff overfit",
            "downweight misses persistent trends",
        ],
    },
    {
        "slug": "cusum-varratio-state",
        "lane": "regime_state",
        "hypothesis": "CUSUM/variance-ratio states distinguish trending from mean-reverting windows for dynamic sleeve routing.",
        "mechanism": "State tests detect structural path behavior better than fixed indicators.",
        "source_skill_inspiration": ["statsmodels", "statistical-analysis", "experimental-design"],
        "features": [
            "CUSUM state",
            "variance ratio",
            "trend sleeve signal",
            "reversion sleeve signal",
        ],
        "parameters": {"test_window": [60, 240, 1440], "state_threshold": "train_selected"},
        "disconfirming_evidence": [
            "test statistics unstable",
            "routing adds churn",
            "state threshold selected post hoc",
        ],
    },
    {
        "slug": "dominant-asset-transfer-shadow",
        "lane": "transfer_learning",
        "hypothesis": "Alphas learned on dominant/liquid assets can transfer to similar assets only when shadow evidence confirms asset-specific robustness.",
        "mechanism": "Liquid leaders provide cleaner signal templates but transfer requires completed-shadow validation.",
        "source_skill_inspiration": [
            "arbor",
            "experimental-design",
            "scientific-critical-thinking",
        ],
        "features": ["source asset signal", "target asset similarity", "shadow PnL", "liquidity"],
        "parameters": {"similarity_metric": "pre_registered", "shadow_min_days": 30},
        "disconfirming_evidence": [
            "transfer chosen from current OOS",
            "target liquidity insufficient",
            "shadow sample too small",
        ],
    },
    {
        "slug": "news-filing-sentiment-optional",
        "lane": "text_alt_data_optional",
        "hypothesis": "Time-stamped text/news/filing sentiment can gate risk exposure if publication-time provenance is exact.",
        "mechanism": "Public information shocks change risk appetite and asset-specific flows.",
        "source_skill_inspiration": [
            "transformers",
            "research-lookup",
            "database-lookup",
            "scientific-critical-thinking",
        ],
        "features": [
            "timestamped sentiment",
            "source reliability",
            "asset mapping",
            "publication lag",
        ],
        "parameters": {
            "lag_minutes": "source_contract",
            "sentiment_threshold": "train_validation_only",
        },
        "disconfirming_evidence": [
            "publication timestamps unreliable",
            "source not available live",
            "NLP model hallucination/proxy leakage",
        ],
    },
    {
        "slug": "geospatial-commodity-risk-optional",
        "lane": "geospatial_alt_data_optional",
        "hypothesis": "Geospatial/weather/supply-chain signals may improve commodity-linked crypto/TradFi sleeves only with auditable timestamps.",
        "mechanism": "Physical-world constraints influence commodity and macro risk premia.",
        "source_skill_inspiration": ["geomaster", "geopandas", "database-lookup"],
        "features": ["geospatial event", "weather anomaly", "commodity proxy", "timestamp lag"],
        "parameters": {"event_lag": "source_contract", "exposure_gate": "train_validation_only"},
        "disconfirming_evidence": [
            "not relevant to current universe",
            "data license/provenance invalid",
            "timestamp lag too slow",
        ],
    },
]


def git_sha(repo: Path) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    except Exception:
        return "unknown"


def git_status(repo: Path) -> list[str]:
    try:
        out = subprocess.check_output(["git", "status", "--short"], cwd=repo, text=True)
    except Exception:
        return []
    return [line for line in out.splitlines() if line.strip()]


def read_backtest_start(repo: Path) -> str | None:
    config = repo / "config.yaml"
    if not config.exists():
        return None
    in_backtest = False
    for line in config.read_text(errors="ignore").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if not line.startswith(" ") and stripped.endswith(":"):
            in_backtest = stripped == "backtest:"
            continue
        if in_backtest and stripped.startswith("start_date:"):
            return stripped.split(":", 1)[1].strip().strip('"').strip("'")
    return None


def candidate_from_template(template: dict[str, Any], idx: int, run_date: str) -> dict[str, Any]:
    return {
        "candidate_id": f"alpha_{run_date}_{idx:02d}_{template['slug']}",
        "lane": template["lane"],
        "status": "proposed_pre_registered",
        "hypothesis": template["hypothesis"],
        "mechanism": template["mechanism"],
        "source_skill_inspiration": template["source_skill_inspiration"],
        "allowed_selection_data": ["train_window", "validation_window", "lagged_completed_shadow"],
        "forbidden_data": ["locked_current_oos", "current_unfinished_live", "future_data"],
        "features": template["features"],
        "parameters": template["parameters"],
        "pre_registered_metrics": METRICS,
        "fail_closed_conditions": FAIL_CLOSED,
        "cost_stress_bps": [10, 15, 20],
        "disconfirming_evidence": template["disconfirming_evidence"],
        "implementation_notes": "Implement only after data contract and experiment design are accepted for this run.",
        "decision": "pending",
    }


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--run-id", help="Default: alpha_skill_bootstrap_<UTC timestamp>")
    parser.add_argument("--max-candidates", type=int, default=len(CANDIDATE_TEMPLATES))
    parser.add_argument("--force", action="store_true", help="Overwrite an existing run directory.")
    args = parser.parse_args()

    repo = args.repo_root.resolve()
    now = datetime.now(UTC)
    run_stamp = now.strftime("%Y%m%dT%H%M%SZ")
    run_date = now.strftime("%Y%m%d")
    run_id = args.run_id or f"alpha_skill_bootstrap_{run_stamp}"
    out_dir = repo / "var" / "reports" / "alpha_discovery" / run_id
    if out_dir.exists() and not args.force:
        raise SystemExit(f"Run directory already exists: {out_dir} (use --force to overwrite)")
    out_dir.mkdir(parents=True, exist_ok=True)

    max_candidates = max(1, min(args.max_candidates, len(CANDIDATE_TEMPLATES)))
    candidates = [
        candidate_from_template(t, i + 1, run_date)
        for i, t in enumerate(CANDIDATE_TEMPLATES[:max_candidates])
    ]
    audit = json.loads(AUDIT_JSON.read_text(encoding="utf-8")) if AUDIT_JSON.exists() else {}
    backtest_start = read_backtest_start(repo) or "2025-01-01"

    manifest = {
        "run_id": run_id,
        "created_at_utc": now.isoformat(),
        "repo_root": str(repo),
        "repo_git_sha": git_sha(repo),
        "git_status_short_at_init": git_status(repo),
        "skill": "alpha-research-pipeline",
        "skill_dir": str(SKILL_DIR),
        "scientific_agent_skills_source_commit": audit.get("source_commit"),
        "reviewed_scientific_skill_count": audit.get("reviewed_skill_count"),
        "real_money_execution": False,
        "allow_real_money": False,
        "ready_for_real": False,
    }

    candidate_registry = {
        "schema_version": 1,
        "run_id": run_id,
        "created_at_utc": now.isoformat(),
        "candidate_count": len(candidates),
        "candidate_budget": {
            "max_implemented_first_batch": min(8, len(candidates)),
            "max_family_variants_before_correction": 5,
            "requires_multiple_testing_accounting": True,
        },
        "global_forbidden_data": ["locked_current_oos", "current_unfinished_live", "future_data"],
        "real_money_execution": False,
        "allow_real_money": False,
        "ready_for_real": False,
        "candidates": candidates,
    }

    experiment_design = {
        "schema_version": 1,
        "run_id": run_id,
        "created_at_utc": now.isoformat(),
        "purpose": "Pre-register LuminaQuant alpha discovery before implementation or result inspection.",
        "windows": {
            "train_window": {
                "start": backtest_start,
                "end": "latest_available_minus_120d",
                "may_select": ["features", "parameters", "candidate_family"],
            },
            "validation_window": {
                "start": "latest_available_minus_120d",
                "end": "latest_available_minus_30d",
                "may_select": ["candidate_admission", "thresholds_if_pre_registered"],
            },
            "lagged_completed_shadow": {
                "start": "latest_available_minus_30d",
                "end": "latest_completed_data_minus_execution_lag",
                "may_select": ["shadow_watch_only"],
            },
            "locked_current_oos": {
                "use": "report_only_after_all_selection_is_frozen",
                "forbidden_for": [
                    "threshold_selection",
                    "sleeve_selection",
                    "weight_selection",
                    "tie_breaks",
                    "promotion_without_prior_gates",
                ],
            },
        },
        "embargo": {
            "required": True,
            "minimum_bars": "max(feature_lookback, label_horizon)",
            "notes": "Set concrete embargo per implemented candidate.",
        },
        "benchmarks": [
            "cash",
            "incumbent_portfolio",
            "raw_strategy_without_new_filter",
            "negative_control",
        ],
        "metrics": METRICS,
        "cost_stress_bps": [10, 15, 20],
        "reject_thresholds": {
            "return_20bps": "> 0 and beats incumbent unless explicitly diagnostic-only",
            "max_drawdown": "not materially worse than incumbent and under policy cap",
            "turnover": "under pre-registered cap or lower than baseline for filters",
            "gross_exposure": "under configured policy cap",
            "data_blockers": "zero fail-closed blockers",
            "leakage_review": "clean",
        },
        "multiple_testing": {
            "family_budget_required": True,
            "method": "pre-register candidate family count; use FDR/Bonferroni or report deflated Sharpe/SPA where available",
        },
        "real_money_execution": False,
        "allow_real_money": False,
        "ready_for_real": False,
    }

    quality_gate = {
        "schema_version": 1,
        "run_id": run_id,
        "created_at_utc": now.isoformat(),
        "required_before_implementation": [
            "candidate_registry.json exists",
            "experiment_design.json exists",
            "data contract inspected",
            "no real money flags",
        ],
        "required_before_decision": [
            "focused tests pass",
            "cost stress at 10/15/20bps recorded",
            "statistical audit complete",
            "leakage/critical-thinking review complete",
            "decision artifact cites files, commands, hashes, metrics",
        ],
        "promotion_allowed": False,
        "promotion_default": "no-promotion",
        "real_money_execution": False,
        "allow_real_money": False,
        "ready_for_real": False,
    }

    run_plan_md = f"""# LuminaQuant Alpha Discovery Run: {run_id}

Created: `{now.isoformat()}`
Repo: `{repo}`
Git SHA: `{manifest["repo_git_sha"]}`
Scientific-agent-skills reviewed: `{audit.get("reviewed_skill_count", "unknown")}` at `{audit.get("source_commit", "unknown")}`

## First actions

1. Inspect `candidate_registry.json` and keep only candidates supported by current data.
2. Fill concrete train/validation/fresh-forward dates in `experiment_design.json` before heavy runs.
3. Run data contract checks and focused tests.
4. Implement at most `{candidate_registry["candidate_budget"]["max_implemented_first_batch"]}` candidates in the first batch.
5. Validate with `validate_alpha_research_run.py --run-dir {out_dir}` before any decision.

## Safety

- real-money flags are false.
- locked/current OOS is forbidden for selection.
- failed data/source/provenance checks require fail-closed `no-promotion`.
"""

    write_json(out_dir / "run_manifest.json", manifest)
    write_json(out_dir / "candidate_registry.json", candidate_registry)
    write_json(out_dir / "experiment_design.json", experiment_design)
    write_json(out_dir / "quality_gate_receipt.json", quality_gate)
    if audit:
        write_json(out_dir / "scientific_skill_audit_snapshot.json", audit)
    (out_dir / "run_plan.md").write_text(run_plan_md, encoding="utf-8")

    print(
        json.dumps(
            {"run_id": run_id, "run_dir": str(out_dir), "candidate_count": len(candidates)},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
