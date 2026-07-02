"""Typed runtime configuration schema."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from lumina_quant.research_universe import BINANCE_EXTENDED_RESEARCH_SYMBOLS_SLASHED

_DEFAULT_DEEP_LEARNING_MODELS: tuple[str, ...] = ("FITS", "CycleNet", "CMamba", "PatchTST")


@dataclass(slots=True)
class SystemConfig:
    """System-level runtime settings."""

    log_level: str = "INFO"


@dataclass(slots=True)
class TradingConfig:
    """Shared trading settings."""

    # Defaults to the full research universe (crypto + metals + tradfi perps),
    # which grows automatically as instruments are added to
    # BINANCE_EXTENDED_RESEARCH_SYMBOLS_SLASHED in lumina_quant.research_universe.
    symbols: list[str] = field(
        default_factory=lambda: list(BINANCE_EXTENDED_RESEARCH_SYMBOLS_SLASHED)
    )
    timeframes: list[str] = field(
        default_factory=lambda: ["1s", "1m", "5m", "15m", "30m", "1h", "4h", "1d"]
    )
    timeframe: str = "1m"
    initial_capital: float = 10000.0
    target_allocation: float = 0.1
    target_allocation_mode: str = "legacy_notional_cap"
    min_trade_qty: float = 0.001


@dataclass(slots=True)
class RiskConfig:
    """Shared risk settings."""

    risk_per_trade: float = 0.005
    max_daily_loss_pct: float = 0.03
    max_total_margin_pct: float = 0.50
    max_symbol_exposure_pct: float = 0.25
    max_order_value: float = 5000.0
    max_order_notional_pct: float = 0.0
    max_total_notional_pct: float = 0.0
    default_stop_loss_pct: float = 0.01
    max_intraday_drawdown_pct: float = 0.03
    max_rolling_loss_pct_1h: float = 0.05
    freeze_new_entries_on_breach: bool = True
    auto_flatten_on_breach: bool = False
    # Phase 5 kill-switch envelope: non-bypassable risk caps
    max_position_size_pct: float = 1.0  # (0, 1] — per-symbol position size as fraction of capital
    consecutive_loss_halt_count: int = (
        5  # > 0 — freeze new entries after N consecutive realized losses
    )
    # Audit-hardening (fix/audit-hardening) risk controls.
    #   allow_metadata_risk_override: when False (default, SAFE) signal-metadata risk
    #     overrides may only LOWER a config ceiling — any override that would RAISE
    #     leverage / target_allocation / max_symbol_exposure_pct / max_order_value /
    #     max_order_notional_pct above its config value is clamped back to the config
    #     ceiling.  Set True only to restore the legacy unclamped behavior.
    allow_metadata_risk_override: bool = False
    #   max_leverage: absolute leverage ceiling applied to metadata leverage overrides
    #     when allow_metadata_risk_override is False.  0.0 => clamp to the configured
    #     run leverage instead (metadata may not raise leverage at all).
    max_leverage: float = 0.0
    #   attach_default_protective_stop: when True, a signal that carries no stop_loss is
    #     given a synthetic protective stop at default_stop_loss_pct from entry so no
    #     position runs naked.  Default False preserves existing backtest numerics
    #     (enable on the backtest machine to measure the cost).
    attach_default_protective_stop: bool = False
    #   hard_drawdown_flatten_pct: > 0 => a hard de-risk tier above the soft FREEZE
    #     threshold; when intraday drawdown exceeds this fraction ALL open positions are
    #     flattened even if auto_flatten_on_breach is False.  0.0 => disabled (legacy).
    hard_drawdown_flatten_pct: float = 0.0
    #   enforce_order_risk_gate_in_backtest: when True the backtest order path runs the
    #     same RiskManager.check_order gate as live, so one enforcement path governs both.
    #     Default False preserves the golden baseline (enable on the backtest machine).
    enforce_order_risk_gate_in_backtest: bool = False


@dataclass(slots=True)
class ExecutionConfig:
    """Shared execution model settings."""

    maker_fee_rate: float = 0.0002
    taker_fee_rate: float = 0.0004
    spread_rate: float = 0.0002
    slippage_rate: float = 0.0005
    funding_rate_per_8h: float = 0.0
    funding_interval_hours: int = 8
    maintenance_margin_rate: float = 0.005
    liquidation_buffer_rate: float = 0.0005
    max_bar_volume_ratio: float = 0.1
    compute_backend: str = "gpu"
    gpu_mode: str = "gpu"
    gpu_vram_gb: float = 8.0
    # Audit-hardening (fix/audit-hardening) fill realism.
    #   slippage_impact_model: "flat" (default, golden-stable) keeps the legacy
    #     size-blind slippage; "sqrt_impact" adds a square-root market-impact term
    #     scaled by order participation so large/leveraged orders pay more.
    slippage_impact_model: str = "flat"
    #   slippage_impact_coefficient: impact strength (bps-per-sqrt-participation) for the
    #     sqrt_impact model.  Only used when slippage_impact_model == "sqrt_impact".
    slippage_impact_coefficient: float = 0.0
    #   slippage_adv_quote: reference average daily traded value (quote ccy) used as the
    #     participation denominator for sqrt_impact.  0.0 => fall back to per-bar volume.
    slippage_adv_quote: float = 0.0
    #   require_funding_coverage: when True a leveraged backtest fails / flags instead of
    #     silently charging 0.0 funding when funding feature data is absent.  Default
    #     False preserves current behavior (enable on the backtest machine).
    require_funding_coverage: bool = False


@dataclass(slots=True)
class StorageConfig:
    """Storage settings for audit and exports."""

    backend: str = "parquet-postgres"
    postgres_dsn: str = ""
    postgres_dsn_env: str = "LQ_POSTGRES_DSN"
    market_data_parquet_path: str = "data/market_parquet"
    market_data_exchange: str = "binance"
    wal_max_bytes: int = 268435456
    wal_compact_on_threshold: bool = True
    wal_compaction_interval_seconds: int = 3600
    collector_periodic_enabled: bool = True
    collector_poll_seconds: int = 2
    collector_bootstrap_lookback_hours: int = 24
    materializer_periodic_enabled: bool = True
    materializer_poll_seconds: int = 5
    materializer_base_timeframe: str = "1s"
    materializer_required_timeframes: list[str] = field(
        default_factory=lambda: ["1s", "1m", "5m", "15m", "30m", "1h", "4h", "1d"]
    )
    export_csv: bool = True


@dataclass(slots=True)
class BacktestExternalConfig:
    """External backtest/optimize data source settings."""

    source_kind: str = "csv"
    root_path: str = ""
    symbol_map: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class BacktestRuntimeConfig:
    """Backtest-only settings."""

    start_date: str = "2024-01-01"
    end_date: str | None = None
    mode: str = "windowed"
    commission_rate: float = 0.001
    slippage_rate: float = 0.0005
    annual_periods: int = 252
    risk_free_rate: float = 0.0
    risk_free_mode: str = "us_treasury_constant"
    risk_free_tenor: str = "3m"
    risk_free_annual: float = 0.0
    risk_free_series_path: str = ""
    sortino_target_mode: str = "same_as_rf"
    sortino_target_annual: float = 0.0
    random_seed: int = 42
    persist_output: bool = True
    leverage: int = 3
    margin_mode: str = "isolated"
    poll_seconds: int = 20
    window_seconds: int = 20
    decision_cadence_seconds: int = 20
    chunk_days: int = 2
    chunk_warmup_bars: int = 0
    skip_ahead_enabled: bool = True
    backtest_poll_seconds: int | None = None
    backtest_window_seconds: int | None = None
    backtest_decision_seconds: int | None = None
    market_window_parity_v2_enabled: bool | None = None
    data_source: str = "auto"
    external: BacktestExternalConfig = field(default_factory=BacktestExternalConfig)


@dataclass(slots=True)
class LiveExchangeConfig:
    """Live exchange settings."""

    driver: str = "binance_futures"
    name: str = "binance"
    market_type: str = "future"
    position_mode: str = "HEDGE"
    margin_mode: str = "isolated"
    leverage: int = 3


@dataclass(slots=True)
class LiveExternalConfig:
    """External live data source settings."""

    source_kind: str = "jsonl"
    path: str = ""
    poll_seconds: int = 2
    allow_stale_seconds: int = 45
    schema: str = "market_window_v1"
    symbol_map: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class LivePolymarketConfig:
    """Polymarket-specific live configuration."""

    host: str = "https://clob.polymarket.com"
    gamma_host: str = "https://gamma-api.polymarket.com"
    data_host: str = "https://data-api.polymarket.com"
    market_ws_url: str = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
    user_ws_url: str = "wss://ws-subscriptions-clob.polymarket.com/ws/user"
    chain_id: int = 137
    asset_ids: list[str] = field(default_factory=list)
    private_key_env: str = "POLYMARKET_PRIVATE_KEY"
    api_key_env: str = "POLYMARKET_API_KEY"
    api_secret_env: str = "POLYMARKET_API_SECRET"
    api_passphrase_env: str = "POLYMARKET_API_PASSPHRASE"
    funder: str = ""
    signature_type: int = 0
    allow_real_execution: bool = False


@dataclass(slots=True)
class LiveRuntimeConfig:
    """Live-only settings."""

    mode: str = "paper"
    require_real_enable_flag: bool = True
    market_data_source: str = "committed"
    order_state_source: str = "polling"
    shadow_live_enabled: bool = False
    # Phase 5 go-live gate pipeline (R7: free stage composition, per-stage entry checks)
    go_live_stage: str = "testnet"  # testnet | shadow | canary | full
    kill_switch_enabled: bool = True  # always-on; validate.py REJECTS False
    canary_position_fraction: float = 0.10  # position sizing fraction during canary stage (0, 1]
    # shadow_parity_min_ratio / shadow_parity_window_bars govern the shadow-stage entry
    # check.  The caller must pass an explicit measured parity_ratio (from
    # ShadowLiveRunner.evaluate_signal_parity) to readiness_policy; an unmeasured ratio
    # (None) blocks shadow entry by design — the default=1.0 in the runner is a
    # "no comparison made" sentinel for the runner itself, not a bypass of the gate.
    # canary/full entry NEVER depends on parity — it gates on artifact flags
    # (canary_execution_allowed, real_money_execution) + LUMINA_ENABLE_LIVE_REAL.
    # Shadow is the measuring stage (testnet-routed, no money at risk);
    # canary is the deploying stage with its own artifact gate.
    shadow_parity_min_ratio: float = 0.99  # minimum signal agreement ratio for shadow entry
    shadow_parity_window_bars: int = 1000  # bars to evaluate parity over
    reconciliation_poll_fallback_enabled: bool = True
    book_ticker_enabled: bool = False
    default_order_type: str = "LMT"
    allow_market_orders: bool = False
    limit_price_mode: str = "one_tick_worse"
    limit_price_offset_ticks: int = 1
    limit_price_tick_fallback: float = 0.0
    limit_time_in_force: str = "GTC"
    protective_order_style: str = "limit"
    startup_reconciliation_hard_fail: bool = False
    main_loop_error_retry_limit: int = 3
    main_loop_error_window_seconds: int = 60
    poll_interval: int = 20
    poll_seconds: int = 20
    window_seconds: int = 20
    decision_cadence_seconds: int = 20
    materialized_staleness_threshold_seconds: int = 45
    materialized_staleness_alert_cooldown_seconds: int = 60
    live_poll_seconds: int | None = None
    ingest_window_seconds: int | None = None
    order_timeout: int = 10
    heartbeat_interval_sec: int = 30
    reconciliation_interval_sec: int = 30
    # Audit-hardening (fix/audit-hardening): reject limit orders validated against a
    # cached best-bid/offer older than this many seconds (fail-closed during websocket
    # stalls / dislocations).  Live-only — no backtest impact.  0.0 => disabled (legacy).
    max_bbo_age_seconds: float = 2.0
    exchange: LiveExchangeConfig = field(default_factory=LiveExchangeConfig)
    external: LiveExternalConfig = field(default_factory=LiveExternalConfig)
    polymarket: LivePolymarketConfig = field(default_factory=LivePolymarketConfig)
    symbol_limits: dict[str, dict[str, float]] = field(default_factory=dict)
    api_key: str = ""
    secret_key: str = ""
    telegram_bot_token: str | None = None
    telegram_chat_id: str | None = None
    testnet: bool | None = None
    mt5_magic: int = 234000
    mt5_deviation: int = 20
    mt5_bridge_python: str = ""
    mt5_bridge_script: str = "scripts/mt5_bridge_worker.py"
    mt5_bridge_use_wslpath: bool = True
    market_window_parity_v2_enabled: bool | None = None


@dataclass(slots=True)
class MarketWindowConfig:
    """Shared MARKET_WINDOW contract and rollout controls."""

    parity_v2_enabled: bool = False
    metrics_log_path: str = "logs/live/market_window_metrics.ndjson"


@dataclass(slots=True)
class OptimizationRuntimeConfig:
    """Optimization-only settings."""

    method: str = "OPTUNA"
    strategy: str = "RsiStrategy"
    optuna: dict[str, Any] = field(default_factory=dict)
    grid: dict[str, Any] = field(default_factory=dict)
    walk_forward_folds: int = 3
    overfit_penalty: float = 0.5
    max_workers: int = 1
    persist_best_params: bool = False
    validation_days: int = 30
    oos_days: int = 30


@dataclass(slots=True)
class PromotionGateConfig:
    """Promotion gate defaults and strategy-specific override profiles."""

    days: int = 14
    max_order_rejects: int = 0
    max_order_timeouts: int = 0
    max_reconciliation_alerts: int = 0
    max_critical_errors: int = 0
    require_alpha_card: bool = False
    strategy_profiles: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass(slots=True)
class MemoryConfig:
    """Memory usage limits."""

    cap_gb: float = 8.0


@dataclass(slots=True)
class ValidationConfig:
    """Validation and testing contracts."""

    golden_rtol: float = 1e-8


_VALID_DATA_KINDS: frozenset[str] = frozenset(
    {"ohlcv", "funding", "feature_points", "aggtrades_tick"}
)


@dataclass(slots=True)
class DataConfig:
    """Data-collection settings.

    ``kinds`` is the explicit list of data kinds the collector will fetch.
    Valid values: ohlcv, funding, feature_points, aggtrades_tick.
    ``aggtrades_tick`` is validation-only (testnet/paper/replay; no live orders).
    ``tick_path`` overrides the parquet root for raw aggTrades archives.
    """

    kinds: list[str] = field(default_factory=lambda: ["ohlcv", "funding", "feature_points"])
    tick_path: str = ""


@dataclass(slots=True)
class DeepLearningRuntimeConfig:
    """External DeepLearning training/prediction pipeline settings."""

    enabled: bool = False
    repo_path: str = "../../DeepLearning"
    config_path: str = "configs/config.yaml"
    test_config_path: str = "configs/config_test.yaml"
    dataset_path: str = ""
    feature_matrix_path: str = "var/data/deeplearning/features"
    feature_format: str = "parquet"
    prediction_path: str = "var/data/deeplearning/predictions"
    manifest_path: str = "var/data/deeplearning/pipeline_manifest.json"
    training_state_path: str = "var/data/deeplearning/training_state.json"
    metrics_path: str = "var/data/deeplearning/metrics"
    models: list[str] = field(default_factory=lambda: list(_DEFAULT_DEEP_LEARNING_MODELS))
    ensemble_models: list[str] = field(default_factory=list)
    freq: str = ""
    target_suffix: str = "close"
    seq_len: int = 120
    pred_len: int = 30
    total_pred: int = 120
    gpus: str = "[]"
    hpo_enabled: bool = False
    hpo_trials: int = 20
    gpu_parallel_jobs: int = 1
    gpu_parallel_min_free_vram_gb: float = 2.0
    gpu_parallel_max_utilization_pct: float = 85.0
    hpo_top_trial_fraction: float = 0.10
    hpo_min_top_trials: int = 5
    hpo_max_train_val_gap: float = 0.15
    hpo_boundary_tolerance: float = 0.02
    chronological_sanity_check: bool = True
    chronological_sanity_val_fraction: float = 0.20
    chronological_sanity_max_relative_drop: float = 0.25
    partial_hpo_trials: int = 5
    epochs: int = 0
    epochs_per_trial: int = 0
    force_train: bool = False
    resume: bool = True
    run_test_after_fit: bool = False
    no_upload: bool = True
    strict_source_models: bool = True
    min_model_coverage: int = 2
    entry_threshold_bps: float = 10.0
    exit_threshold_bps: float = 2.0
    min_model_agreement: float = 0.75
    max_dispersion_bps: float = 80.0
    min_confidence: float = 0.0
    target_allocation: float = 0.05
    stop_loss_pct: float = 0.0
    take_profit_pct: float = 0.0


@dataclass(slots=True)
class StrategyQualityConfig:
    """Portfolio-level strategy quality overlays for existing strategies."""

    enabled: bool = False
    edge_gate_enabled: bool = True
    min_expected_edge_bps: float = 8.0
    edge_cost_buffer_bps: float = 3.0
    allow_unknown_edge: bool = True
    regime_router_enabled: bool = True
    regime_lookback_bars: int = 60
    trend_return_bps: float = 35.0
    panic_return_bps: float = 120.0
    low_liquidity_volume_ratio: float = 0.20
    position_sizing_enabled: bool = True
    target_vol_per_bar: float = 0.006
    min_position_scale: float = 0.20
    max_position_scale: float = 1.0
    turnover_budget_enabled: bool = True
    max_daily_turnover_pct: float = 3.0
    exit_overlay_enabled: bool = True
    atr_window_bars: int = 20
    atr_stop_mult: float = 2.0
    atr_take_profit_mult: float = 3.0
    trailing_atr_mult: float = 1.5
    min_stop_bps: float = 20.0
    max_stop_bps: float = 500.0
    strategy_health_enabled: bool = True
    health_window_trades: int = 20
    min_health_scale: float = 0.25
    loss_cooldown_bars: int = 12
    profit_moonshot_conflict_cooldown_bars: int = 3
    pair_min_correlation: float = 0.35


@dataclass(slots=True)
class PortfolioConfig:
    """Opt-in portfolio allocation-method settings.

    ``allocation_method`` selects the portfolio construction routine.  The
    default ``"legacy"`` preserves the current behavior — NO code routes through
    the extra pure-numpy optimizers (ERC / MaxDiversification / MeanVariance /
    HRP in ``portfolio.optimizers_extra``) unless a non-legacy method is chosen,
    so the golden backtest / walk-forward numerics stay byte-identical by
    default.  All other fields are inert tuning knobs for those optimizers and
    have no effect while ``allocation_method == "legacy"``.
    """

    allocation_method: str = "legacy"
    erc_max_iter: int = 10000
    erc_tol: float = 1e-10
    mv_risk_aversion: float = 5.0
    pgd_max_iter: int = 500
    pgd_step: float = 0.05
    hrp_corr_threshold: float = 0.60
    cov_window: int = 0


@dataclass(slots=True)
class AlphaSearchConfig:
    """Opt-in gate for the deterministic alpha discovery search loop.

    ``enabled`` defaults to ``False`` — the generate -> evaluate -> select loop in
    ``lumina_quant.research.alpha_search`` is inert net-new research tooling that
    no default runtime path imports, so the golden backtest / walk-forward
    numerics stay byte-identical.  Flip this only to run the offline alpha
    discovery loop.  The remaining fields expose the loop's deterministic search
    budget and survivorship thresholds as tunable overrides (defaults mirror the
    in-module constants in ``research.alpha_search``); ``enable_llm_proposer``
    gates the frozen/seeded LLM proposer draft channel (still fed through the same
    deterministic survivorship gate — never a live dependency).
    """

    enabled: bool = False
    max_candidates: int = 256
    seed: int = 20260701
    dsr_threshold: float = 0.95
    require_spa: bool = False
    require_pbo: bool = False
    enable_llm_proposer: bool = False


@dataclass(slots=True)
class ResearchConfig:
    """Opt-in research-only tooling toggles.

    All fields default to the current behavior; nothing in the default runtime
    path imports the gated research kernels.  ``execution_attribution_enabled``
    gates the offline execution-attribution kernel
    (``research.execution_attribution``): when ``False`` (default) no code
    constructs or runs it, so backtest / walk-forward numerics stay
    byte-identical.  ``alpha_search`` gates the deterministic alpha discovery
    loop and is likewise inert by default.
    """

    execution_attribution_enabled: bool = False
    alpha_search: AlphaSearchConfig = field(default_factory=AlphaSearchConfig)


@dataclass(slots=True)
class Qlib158FormulaConfig:
    """Opt-in gate for the Qlib158-style formulaic factor library.

    ``enabled`` defaults to ``False`` — the factor definitions in
    ``lumina_quant.indicators.qlib158`` are inert net-new transforms that no
    default runtime path imports, so the golden backtest / walk-forward
    numerics stay byte-identical.  Flip this only to opt the mirror factors
    into a research / candidate-library path.  ``min_abs_ic`` / ``min_icir``
    expose the documented IC / ICIR eligibility bar as tunable flags (defaults
    mirror ``qlib158.MIN_ABS_IC`` / ``qlib158.MIN_ICIR``).
    """

    enabled: bool = False
    min_abs_ic: float = 0.02
    min_icir: float = 0.30


@dataclass(slots=True)
class StrategiesConfig:
    """Opt-in strategy-family toggles (all inert by default)."""

    qlib158_formula: Qlib158FormulaConfig = field(default_factory=Qlib158FormulaConfig)


@dataclass(slots=True)
class RuntimeConfig:
    """Full runtime configuration bundle."""

    system: SystemConfig = field(default_factory=SystemConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    data: DataConfig = field(default_factory=DataConfig)
    deep_learning: DeepLearningRuntimeConfig = field(default_factory=DeepLearningRuntimeConfig)
    strategy_quality: StrategyQualityConfig = field(default_factory=StrategyQualityConfig)
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)
    backtest: BacktestRuntimeConfig = field(default_factory=BacktestRuntimeConfig)
    live: LiveRuntimeConfig = field(default_factory=LiveRuntimeConfig)
    optimization: OptimizationRuntimeConfig = field(default_factory=OptimizationRuntimeConfig)
    market_window: MarketWindowConfig = field(default_factory=MarketWindowConfig)
    promotion_gate: PromotionGateConfig = field(default_factory=PromotionGateConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    research: ResearchConfig = field(default_factory=ResearchConfig)
    strategies: StrategiesConfig = field(default_factory=StrategiesConfig)
