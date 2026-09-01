'use client';

import { useMemo } from 'react';

import { TimeSeriesChart } from '@/components/charts/time-series-chart';
import { PageContextBar } from '@/components/page-context-bar';
import { SurfaceState } from '@/components/surface-state';
import type { OverviewPayload } from '@/lib/dashboard-contracts';
import {
  formatCompactTimestamp,
  formatMetricValue,
  formatNumber,
  pnlClass,
  statusPillClass,
} from '@/lib/format';
import { useBridgeFetch } from '@/lib/use-bridge-fetch';

const OVERVIEW_URL = '/api/python/dashboard/overview';

interface HeadlineTile {
  key: string;
  label: string;
  value: number | string | null | undefined;
  className: string;
}

function toChartPoints(
  rows: Array<{ timestamp: string }>,
  pick: (row: { timestamp: string }) => number | undefined,
): Array<{ t: number; v: number }> {
  return rows
    .map((row) => ({ t: Date.parse(row.timestamp), v: pick(row) ?? Number.NaN }))
    .filter((point) => Number.isFinite(point.t) && Number.isFinite(point.v));
}

export function OverviewRuntime() {
  const { payload, error, loading, refetch, lastFetchedAt } = useBridgeFetch<OverviewPayload>(
    OVERVIEW_URL,
    'overview request failed',
  );

  const curves = useMemo(() => {
    if (!payload) {
      return { equity: [], drawdown: [] };
    }
    return {
      equity: toChartPoints(payload.equity_curve, (row) => (row as { equity?: number }).equity),
      drawdown: toChartPoints(
        payload.drawdown_curve,
        (row) => (row as { drawdown?: number }).drawdown,
      ),
    };
  }, [payload]);

  if (!payload) {
    return (
      <>
        <PageContextBar lastFetchedAt={lastFetchedAt} onRefresh={refetch} loading={loading} />
        <SurfaceState error={error || undefined} surface="overview" onRetry={refetch} />
        {!error ? <p>Loading overview…</p> : null}
      </>
    );
  }

  const status = payload.source.status;
  const totalReturnEntry = payload.summary_metrics.find((metric) => metric.key === 'total_return');
  const totalReturn = totalReturnEntry?.value ?? null;
  const perf = payload.performance_metrics;
  const tiles: HeadlineTile[] = [
    {
      key: 'total_return',
      label: 'Total Return',
      value: totalReturn,
      className: typeof totalReturn === 'number' ? pnlClass(totalReturn) : '',
    },
    { key: 'cagr', label: 'CAGR', value: perf.cagr, className: '' },
    { key: 'sharpe_ratio', label: 'Sharpe', value: perf.sharpe_ratio, className: '' },
    { key: 'sortino_ratio', label: 'Sortino', value: perf.sortino_ratio, className: '' },
    { key: 'calmar_ratio', label: 'Calmar', value: perf.calmar_ratio, className: '' },
    {
      key: 'max_drawdown',
      label: 'Max Drawdown',
      value: perf.max_drawdown,
      className: perf.max_drawdown ? 'pnl-neg' : '',
    },
    {
      key: 'annualized_volatility',
      label: 'Ann. Vol',
      value: perf.annualized_volatility,
      className: '',
    },
  ];

  const window = payload.equity_window;
  const windowNote = window
    ? [
        `Metrics over full run · ${formatNumber(window.points_total, { digits: 0 })} points`,
        window.start && window.end
          ? `${formatCompactTimestamp(window.start)} → ${formatCompactTimestamp(window.end)}`
          : null,
        window.truncated ? 'covering the newest 200,000 rows' : null,
      ]
        .filter(Boolean)
        .join(' · ')
    : null;

  const recentRuns = payload.recent_runs;
  const recentJobs = payload.workflow_jobs.slice(0, 10);

  return (
    <>
      <PageContextBar
        asOf={payload.as_of}
        runId={payload.source.run_id}
        status={status}
        lastFetchedAt={lastFetchedAt}
        onRefresh={refetch}
        loading={loading}
      />
      <SurfaceState status={status} error={error || undefined} surface="overview" onRetry={refetch} />

      <section className="section-card">
        <div className="section-header">
          <div>
            <p className="eyebrow">Headline</p>
            <h3>Run performance</h3>
          </div>
          {windowNote ? <div className="metric-badge">{windowNote}</div> : null}
        </div>
        <div className="metric-grid">
          {tiles.map((tile) => (
            <article key={tile.key}>
              <span>{tile.label}</span>
              <strong className={tile.className || undefined}>
                {formatMetricValue(tile.value, { key: tile.key, label: tile.label })}
              </strong>
            </article>
          ))}
        </div>
      </section>

      <section className="section-card">
        <div className="section-header">
          <div>
            <p className="eyebrow">Equity</p>
            <h3>Equity curve</h3>
          </div>
        </div>
        <TimeSeriesChart
          series={[{ id: 'equity', label: 'Equity', points: curves.equity, kind: 'area' }]}
          yKind="currency"
          emptyLabel="No equity points recorded yet."
        />
      </section>

      <section className="section-card">
        <div className="section-header">
          <div>
            <p className="eyebrow">Risk</p>
            <h3>Drawdown</h3>
          </div>
        </div>
        <TimeSeriesChart
          series={[{ id: 'drawdown', label: 'Drawdown', points: curves.drawdown, kind: 'area' }]}
          yKind="percent"
          emptyLabel="No drawdown points recorded yet."
        />
      </section>

      <section className="section-card">
        <div className="section-header">
          <div>
            <p className="eyebrow">Runs</p>
            <h3>Recent runs</h3>
          </div>
          <div className="metric-badge">{recentRuns.length} runs</div>
        </div>
        {recentRuns.length > 0 ? (
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Run ID</th>
                  <th>Mode</th>
                  <th>Status</th>
                  <th>Strategy</th>
                  <th>Started</th>
                </tr>
              </thead>
              <tbody>
                {recentRuns.map((run) => (
                  <tr key={run.run_id}>
                    <td>
                      <code className="run-chip">{run.run_id}</code>
                    </td>
                    <td>{run.mode}</td>
                    <td>
                      <span className={statusPillClass(run.status)}>{run.status}</span>
                    </td>
                    <td>{run.strategy || 'n/a'}</td>
                    <td>{formatCompactTimestamp(run.started_at)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p>No runs recorded yet.</p>
        )}
      </section>

      <section className="section-card">
        <div className="section-header">
          <div>
            <p className="eyebrow">Jobs</p>
            <h3>Recent jobs</h3>
          </div>
          <div className="metric-badge">{recentJobs.length} jobs</div>
        </div>
        {recentJobs.length > 0 ? (
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Workflow</th>
                  <th>Status</th>
                  <th>Mode</th>
                  <th>Strategy</th>
                  <th>Run ID</th>
                  <th>Started</th>
                </tr>
              </thead>
              <tbody>
                {recentJobs.map((job) => (
                  <tr key={job.job_id}>
                    <td>{job.workflow}</td>
                    <td>
                      <span className={statusPillClass(job.status)}>{job.status}</span>
                    </td>
                    <td>{job.requested_mode || 'n/a'}</td>
                    <td>{job.strategy || 'n/a'}</td>
                    <td>{job.run_id ? <code className="run-chip">{job.run_id}</code> : 'n/a'}</td>
                    <td>{formatCompactTimestamp(job.started_at)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p>No workflow jobs recorded yet.</p>
        )}
      </section>
    </>
  );
}
