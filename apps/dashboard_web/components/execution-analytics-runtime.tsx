'use client';

import { useState } from 'react';

import { PageContextBar } from '@/components/page-context-bar';
import { RunSelector } from '@/components/run-selector';
import { SurfaceState } from '@/components/surface-state';
import type { ExecutionAnalyticsPayload } from '@/lib/dashboard-contracts';
import {
  formatCompactTimestamp,
  formatCurrency,
  formatDuration,
  formatMetricValue,
  formatNumber,
  pnlClass,
} from '@/lib/format';
import { useBridgeFetch } from '@/lib/use-bridge-fetch';

const ENDPOINT = '/api/python/dashboard/execution-analytics';

/** Humanize a holding duration in seconds via the shared formatter. */
function formatHoldingTime(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined || !Number.isFinite(seconds)) {
    return 'n/a';
  }
  return formatDuration(seconds * 1000);
}

/** Map an order lifecycle status onto the shared status-pill palette. */
function orderStatusPillClass(status: string): string {
  const normalized = status.toLowerCase();
  if (/reject|fail|expire/.test(normalized)) {
    return 'status-failed';
  }
  if (normalized.includes('cancel')) {
    return 'status-degraded';
  }
  if (normalized.includes('partial')) {
    return 'status-running';
  }
  if (/fill|closed|done/.test(normalized)) {
    return 'status-ok';
  }
  return 'status-running';
}

interface SummaryTile {
  label: string;
  value: string;
  className?: string;
}

interface TileGroup {
  title: string;
  tiles: SummaryTile[];
}

function buildTileGroups(summary: ExecutionAnalyticsPayload['summary']): TileGroup[] {
  return [
    {
      title: 'Fills',
      tiles: [
        { label: 'Buy fills', value: formatMetricValue(summary.buy_fills, { key: 'buy_fills' }) },
        { label: 'Sell fills', value: formatMetricValue(summary.sell_fills, { key: 'sell_fills' }) },
        { label: 'Orders', value: formatMetricValue(summary.order_count, { key: 'order_count' }) },
        { label: 'Avg fill size', value: formatMetricValue(summary.avg_qty, { key: 'avg_qty' }) },
        { label: 'Avg notional', value: formatCurrency(summary.avg_notional) },
        { label: 'Total commission', value: formatCurrency(summary.total_commission) },
      ],
    },
    {
      title: 'Outcomes',
      tiles: [
        {
          label: 'Closed trades',
          value: formatMetricValue(summary.closed_trade_count, { key: 'closed_trade_count' }),
        },
        {
          label: 'Avg trade return',
          value: formatMetricValue(summary.avg_trade_return_pct, { key: 'avg_trade_return_pct' }),
          className: pnlClass(summary.avg_trade_return_pct),
        },
        {
          label: 'Best trade PnL',
          value: formatCurrency(summary.best_trade_pnl),
          className: pnlClass(summary.best_trade_pnl),
        },
        {
          label: 'Worst trade PnL',
          value: formatCurrency(summary.worst_trade_pnl),
          className: pnlClass(summary.worst_trade_pnl),
        },
        { label: 'Avg holding time', value: formatHoldingTime(summary.holding_time_avg_sec) },
      ],
    },
    {
      title: 'Streaks',
      tiles: [
        {
          label: 'Longest win streak',
          value: formatMetricValue(summary.win_streak_max, { key: 'win_streak_max' }),
        },
        {
          label: 'Longest loss streak',
          value: formatMetricValue(summary.loss_streak_max, { key: 'loss_streak_max' }),
        },
        { label: 'Avg win streak', value: formatNumber(summary.win_streak_avg, { digits: 1 }) },
        { label: 'Avg loss streak', value: formatNumber(summary.loss_streak_avg, { digits: 1 }) },
      ],
    },
    {
      title: 'Long vs Short',
      tiles: [
        { label: 'Long trades', value: formatMetricValue(summary.long_trades, { key: 'long_trades' }) },
        {
          label: 'Long win rate',
          value: formatMetricValue(summary.long_win_rate, { key: 'long_win_rate' }),
        },
        { label: 'Short trades', value: formatMetricValue(summary.short_trades, { key: 'short_trades' }) },
        {
          label: 'Short win rate',
          value: formatMetricValue(summary.short_win_rate, { key: 'short_win_rate' }),
        },
      ],
    },
  ];
}

export function ExecutionAnalyticsRuntime() {
  const [selectedRunId, setSelectedRunId] = useState('');
  const url = selectedRunId ? `${ENDPOINT}?run_id=${encodeURIComponent(selectedRunId)}` : ENDPOINT;
  const { payload, error, loading, refetch, lastFetchedAt } =
    useBridgeFetch<ExecutionAnalyticsPayload>(url, 'execution analytics request failed');

  const runs = payload?.runs ?? [];
  const showBody =
    payload !== null && (payload.status === 'ok' || payload.status === 'no_execution_data');

  return (
    <div className="page-stack">
      <PageContextBar
        asOf={payload?.as_of}
        runId={payload?.run_id}
        status={payload?.status}
        lastFetchedAt={lastFetchedAt}
        onRefresh={refetch}
        loading={loading}
      />
      {runs.length > 0 ? (
        <RunSelector
          runs={runs}
          value={selectedRunId || payload?.run_id || ''}
          onChange={setSelectedRunId}
        />
      ) : null}
      <SurfaceState
        status={payload?.status}
        error={error || null}
        surface="execution analytics"
        onRetry={refetch}
      />
      {payload === null && !error ? <p>Loading execution analytics…</p> : null}

      {showBody && payload !== null ? (
        <>
          {buildTileGroups(payload.summary).map((group) => (
            <section key={group.title}>
              <p className="eyebrow">{group.title}</p>
              <div className="metric-grid">
                {group.tiles.map((tile) => (
                  <article key={tile.label}>
                    <span>{tile.label}</span>
                    <strong className={tile.className || undefined}>{tile.value}</strong>
                  </article>
                ))}
              </div>
            </section>
          ))}

          <div className="card-grid">
            <article className="feature-card">
              <div className="feature-header">
                <h4>Direction breakdown</h4>
                <span className="metric-badge">{payload.direction_breakdown.length} rows</span>
              </div>
              <div className="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>Direction</th>
                      <th>Closed trades</th>
                      <th>Win rate</th>
                    </tr>
                  </thead>
                  <tbody>
                    {payload.direction_breakdown.map((row) => (
                      <tr key={row.direction}>
                        <td>{row.direction}</td>
                        <td>{formatMetricValue(row.closed_trades, { key: 'closed_trades' })}</td>
                        <td>{formatMetricValue(row.win_rate, { key: 'win_rate' })}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </article>

            <article className="feature-card">
              <div className="feature-header">
                <h4>Order status</h4>
                <span className="metric-badge">{payload.order_status.length} states</span>
              </div>
              <div className="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>Status</th>
                      <th>Count</th>
                    </tr>
                  </thead>
                  <tbody>
                    {payload.order_status.map((row) => (
                      <tr key={row.status}>
                        <td>
                          <span className={`status-pill ${orderStatusPillClass(row.status)}`}>
                            {row.status}
                          </span>
                        </td>
                        <td>{formatMetricValue(row.count, { key: 'count' })}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </article>
          </div>

          <section>
            <p className="eyebrow">Recent closed trades</p>
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Closed at</th>
                    <th>Symbol</th>
                    <th>Side</th>
                    <th>Realized PnL</th>
                    <th>Return</th>
                    <th>Holding time</th>
                  </tr>
                </thead>
                <tbody>
                  {payload.recent_closed_trades.length > 0 ? (
                    payload.recent_closed_trades.map((trade) => (
                      <tr key={`${trade.timestamp}-${trade.symbol}-${trade.realized_pnl}`}>
                        <td>{formatCompactTimestamp(trade.timestamp)}</td>
                        <td>{trade.symbol}</td>
                        <td>{trade.close_side}</td>
                        <td className={pnlClass(trade.realized_pnl) || undefined}>
                          {formatCurrency(trade.realized_pnl)}
                        </td>
                        <td className={pnlClass(trade.realized_return_pct) || undefined}>
                          {formatMetricValue(trade.realized_return_pct, {
                            key: 'realized_return_pct',
                          })}
                        </td>
                        <td>{formatHoldingTime(trade.holding_sec)}</td>
                      </tr>
                    ))
                  ) : (
                    <tr>
                      <td colSpan={6}>No closed trades recorded for this run yet.</td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </section>
        </>
      ) : null}
    </div>
  );
}
