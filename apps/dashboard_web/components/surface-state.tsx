'use client';

import type { ReactNode } from 'react';

export interface SurfaceStateProps {
  status?: string | null;
  error?: string | null;
  /** Human-readable surface name for fallback copy, e.g. 'performance & price'. */
  surface: string;
  onRetry?: () => void;
}

const STATUS_GUIDANCE: Record<string, ReactNode> = {
  missing_dsn: (
    <>
      Postgres DSN is not configured. Set <code className="code-chip">LQ_POSTGRES_DSN</code> or{' '}
      <code className="code-chip">storage.postgres_dsn</code> in config.yaml, then refresh.
    </>
  ),
  no_runs: <>No runs recorded yet — run a backtest/walkforward and refresh.</>,
  no_market_data: (
    <>No market bars found for this run — ingest OHLCV data for the configured symbol/timeframe, then refresh.</>
  ),
  no_optimization_results: (
    <>No optimization results recorded yet — run an optimization workflow, then refresh.</>
  ),
  no_execution_data: (
    <>No execution records for this run yet — fills and order telemetry appear once the strategy trades.</>
  ),
  empty: (
    <>No artifacts configured yet for this surface — generate or point the service at artifacts, then refresh.</>
  ),
  run_not_found: (
    <>The selected run id was not found — pick another run from the selector.</>
  ),
};

/**
 * Actionable empty/error panel for bridge-backed surfaces.  Renders nothing
 * when the surface is healthy so pages can mount it unconditionally.
 */
export function SurfaceState({ status, error, surface, onRetry }: SurfaceStateProps) {
  if (error) {
    return (
      <div className="surface-state surface-state-error" role="alert">
        <p className="surface-state-title">Failed to load {surface}</p>
        <p className="surface-state-detail">{error}</p>
        {onRetry ? (
          <button type="button" className="action-button" onClick={onRetry}>
            Retry
          </button>
        ) : null}
      </div>
    );
  }
  if (!status || status === 'ok') {
    return null;
  }
  const guidance = STATUS_GUIDANCE[status] ?? (
    <>
      The {surface} surface reported status <code className="code-chip">{status}</code>.
    </>
  );
  return (
    <div className="surface-state surface-state-info" role="status">
      <p className="surface-state-detail">{guidance}</p>
    </div>
  );
}
