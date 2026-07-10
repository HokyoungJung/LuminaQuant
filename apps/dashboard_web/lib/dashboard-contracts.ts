export interface BridgeErrorBody {
  detail?: string;
  error?: string;
  ok?: boolean;
}

export interface RunSummary {
  run_id: string;
  mode: string;
  status: string;
  strategy: string;
  started_at: string | null;
}

export interface EquityWindow {
  start: string | null;
  end: string | null;
  points_total: number;
  points_returned: number;
  truncated: boolean;
}

export interface OverviewMetric {
  key: string;
  label: string;
  value: number | string | null;
}

export interface OverviewPoint {
  timestamp: string;
  equity?: number;
  drawdown?: number;
}

export interface OverviewPayload {
  as_of: string;
  summary_metrics: OverviewMetric[];
  performance_metrics: {
    cagr?: number;
    annualized_volatility?: number;
    sharpe_ratio?: number;
    sortino_ratio?: number;
    calmar_ratio?: number;
    max_drawdown?: number;
  };
  recent_runs: Array<{
    run_id: string;
    mode: string;
    status: string;
    strategy: string;
    started_at: string | null;
  }>;
  workflow_jobs: WorkflowJobRecord[];
  equity_curve: Array<OverviewPoint & { equity: number }>;
  drawdown_curve: Array<OverviewPoint & { drawdown: number }>;
  equity_window?: EquityWindow;
  source: {
    mode: string;
    backend: string;
    status: string;
    run_id?: string;
  };
}

export interface WorkflowJobRecord {
  job_id: string;
  workflow: string;
  status: string;
  requested_mode: string;
  strategy: string;
  run_id: string;
  started_at: string | null;
  ended_at: string | null;
}

export interface WorkflowJobsPayload {
  as_of?: string;
  jobs: WorkflowJobRecord[];
  status: string;
}

export interface CutoverGateEvidence {
  label: string;
  detail: string;
  status: 'available' | 'guarded';
}

export interface CutoverGate {
  defaultLauncher: 'next';
  launcherStatus: 'available' | 'guarded';
  readyRoutes: string[];
  evidence: CutoverGateEvidence[];
  remainingGate: string;
}

export interface RiskHealthPayload {
  as_of: string;
  run_id: string;
  summary: {
    risk_event_count: number;
    heartbeat_count: number;
    order_state_count: number;
  };
  risk_events: Array<{ event_time: string | null; reason: string | null }>;
  heartbeats: Array<{ heartbeat_time: string | null; status: string | null }>;
  order_states: Array<{ event_time: string | null; symbol: string | null; state: string | null; message: string | null }>;
  status: string;
}

export interface AlphaEvidencePayload {
  artifact_kind: string;
  as_of: string;
  advisory_only: boolean;
  real_money_execution_enabled: boolean;
  summary: {
    alpha_count: number;
    run_card_count: number;
    classifications: Record<string, number>;
    reality_gates: Record<string, { passed: boolean; observations: number }>;
    live_readiness_action?: string | null;
  };
  evidence: Array<Record<string, unknown>>;
  run_cards: Array<Record<string, unknown>>;
  live_readiness: Record<string, unknown>;
  source: {
    mode: string;
    status: string;
    [key: string]: unknown;
  };
}

export interface ExactWindowPayload {
  as_of: string;
  generated_at: string | null;
  status: string;
  error: string | null;
  root: string;
  run_root: string;
  summary: {
    candidate_count: number;
    evaluated_count: number;
    promoted_count: number;
    btc_beating_candidate_count: number;
    provisional_candidate_count: number;
    candidate_pool_count: number;
    requested_timeframes: string[];
    requested_symbols: string[];
    low_ram_profile: boolean;
  };
  decision: {
    next_action: string;
    promoted_total: number;
    total_evaluated: number;
    max_peak_rss_mib: number | null;
    valid_strategy_found: boolean;
  };
  memory: {
    status: string;
    peak_rss_mib: number | null;
    soft_limit_mib: number | null;
    hard_limit_mib: number | null;
  };
  portfolio: {
    construction_basis: string;
    oos_return: number | null;
    oos_sharpe: number | null;
    oos_max_drawdown: number | null;
  };
  time_window: Record<string, string>;
  timeframes: Array<{
    timeframe: string;
    candidate_id: string;
    name: string;
    family: string;
    promoted: boolean;
    oos_return: number | null;
    oos_sharpe: number | null;
    oos_max_drawdown: number | null;
    trade_count: number | null;
    reject_reasons: string[];
  }>;
  top_candidates: Array<{
    timeframe: string;
    candidate_id: string;
    name: string;
    family: string;
    promoted: boolean;
    oos_return: number | null;
    oos_sharpe: number | null;
    oos_max_drawdown: number | null;
    trade_count: number | null;
    reject_reasons: string[];
  }>;
  portfolio_weights: Array<{
    name: string;
    timeframe: string;
    family: string;
    weight: number | null;
    oos_return: number | null;
    oos_sharpe: number | null;
  }>;
  notes: Array<{ label: string; value: string }>;
  warnings: string[];
}

export interface PerformancePricePayload {
  as_of: string;
  run_id: string;
  status: string;
  source: {
    mode?: string;
    backend?: string;
    status?: string;
    run_id?: string;
  };
  summary_metrics: OverviewMetric[];
  performance_metrics: OverviewPayload['performance_metrics'];
  equity_curve: Array<OverviewPoint & { equity: number }>;
  drawdown_curve: Array<OverviewPoint & { drawdown: number }>;
  equity_window?: EquityWindow;
  /** Symbol behind benchmark_curve (primary run symbol); null when unknown. */
  benchmark_symbol?: string | null;
  benchmark_curve: Array<{ timestamp: string; price: number }>;
  /** Cumulative funding in quote currency (NOT a rate/fraction). */
  funding_curve: Array<{ timestamp: string; funding: number }>;
  trade_markers: Array<{
    timestamp: string;
    symbol: string;
    direction: string;
    price: number;
    quantity: number;
    realized_pnl: number;
    realized_return_pct: number | null;
    position_after: number;
  }>;
  runs?: RunSummary[];
}

export interface ExecutionAnalyticsPayload {
  as_of: string;
  run_id: string;
  status: string;
  summary: {
    buy_fills: number;
    sell_fills: number;
    avg_qty: number;
    avg_notional: number;
    total_commission: number;
    avg_trade_return_pct: number;
    best_trade_pnl: number;
    worst_trade_pnl: number;
    win_streak_max: number;
    loss_streak_max: number;
    win_streak_avg: number;
    loss_streak_avg: number;
    holding_time_avg_sec: number;
    long_trades: number;
    long_win_rate: number;
    short_trades: number;
    short_win_rate: number;
    order_count: number;
    closed_trade_count: number;
  };
  direction_breakdown: Array<{
    direction: string;
    closed_trades: number;
    win_rate: number;
  }>;
  order_status: Array<{ status: string; count: number }>;
  recent_closed_trades: Array<{
    timestamp: string;
    symbol: string;
    close_side: string;
    realized_pnl: number;
    realized_return_pct: number | null;
    holding_sec: number | null;
  }>;
  runs?: RunSummary[];
}

export interface MarketDataPayload {
  as_of: string;
  run_id: string;
  status: string;
  market_context: {
    symbol?: string;
    timeframe?: string;
    timeframe_clamped?: boolean;
    exchange?: string;
    strategy?: string;
    market_db_path?: string;
    source?: string;
  };
  summary_metrics: OverviewMetric[];
  recent_bars: Array<{
    timestamp: string;
    open?: number | null;
    high?: number | null;
    low?: number | null;
    close?: number | null;
    volume?: number | null;
  }>;
  indicator_summary: OverviewMetric[];
  warnings: string[];
  runs?: RunSummary[];
  symbols?: string[];
}

export interface OptimizationInsightsPayload {
  as_of: string;
  run_id: string;
  status: string;
  summary_metrics: OverviewMetric[];
  stage_breakdown: Array<{
    stage: string;
    count: number;
    median_sharpe: number | null;
    median_robustness: number | null;
  }>;
  top_candidates: Array<{
    created_at: string | null;
    run_id: string;
    stage: string;
    sharpe: number | null;
    train_sharpe: number | null;
    robustness_score: number | null;
    cagr: number | null;
    mdd: number | null;
    params: Record<string, unknown>;
  }>;
  best_candidate: {
    created_at: string | null;
    run_id: string;
    stage: string;
    sharpe: number | null;
    train_sharpe: number | null;
    robustness_score: number | null;
    cagr: number | null;
    mdd: number | null;
    params: Record<string, unknown>;
  } | null;
}

export interface RawDataPayload {
  as_of: string;
  run_id: string;
  status: string;
  context: {
    run_id?: string;
    source?: string;
    market?: string;
  };
  frame_summaries: Array<{
    label: string;
    rows: number;
    columns: number;
  }>;
  previews: Array<{
    label: string;
    columns: string[];
    rows: Array<Record<string, unknown>>;
  }>;
  runs?: RunSummary[];
}

export interface ReportExportPayload {
  as_of: string;
  run_id: string;
  status: string;
  filenames: {
    json?: string;
    markdown?: string;
  };
  json_report: {
    title?: string;
    generated_at?: string;
    run_id?: string;
    strategy?: string;
    mode?: string;
    status?: string;
    period_start?: string | null;
    period_end?: string | null;
    total_return?: number | string | null;
    latest_equity?: number | string | null;
    realized_pnl?: number;
    closed_trade_count?: number;
    risk_event_count?: number;
    heartbeat_count?: number;
    performance_metrics?: OverviewPayload['performance_metrics'];
  };
  markdown_report: string;
  cutover_gate: {
    default_launcher?: string;
    status?: string;
    evidence?: string[];
  };
  runs?: RunSummary[];
}

export interface FactorInsightsPayload {
  artifact_kind: string;
  as_of: string;
  advisory_only: boolean;
  real_money_execution_enabled: boolean;
  status: string;
  summary: {
    factor_count: number;
    candidate_count: number;
    lag_count: number;
    top_factor: string | null;
    top_factor_ic_ir: number | null;
  };
  ic_heatmap: {
    factors: string[];
    lags: string[];
    cells: Array<Array<number | null>>;
    ic_mean: Record<string, number | null>;
    ic_ir: Record<string, number | null>;
  };
  factor_ranking: Array<{
    factor: string;
    ic_mean: number | null;
    ic_ir: number | null;
    ic_positive_ratio: number | null;
    t_stat: number | null;
    turnover_mean: number | null;
    quantile_spread_mean: number | null;
    n_periods: number;
  }>;
  candidate_queue: Array<{
    candidate_id: string;
    strategy: string;
    status: string;
    score: number | null;
    sharpe: number | null;
    robustness_score: number | null;
    submitted_at: string | null;
  }>;
  source: Record<string, unknown>;
}
