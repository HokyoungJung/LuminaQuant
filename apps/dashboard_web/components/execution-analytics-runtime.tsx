'use client';

import type { ExecutionAnalyticsPayload } from '@/lib/dashboard-contracts';
import { useBridgeFetch } from '@/lib/use-bridge-fetch';

export function ExecutionAnalyticsRuntime() {
  const { payload, error } = useBridgeFetch<ExecutionAnalyticsPayload>(
    '/api/python/dashboard/execution-analytics',
    'execution analytics bridge failed',
  );

  if (error) {
    return <p>{error}</p>;
  }
  if (payload === null) {
    return <p>Loading execution analytics…</p>;
  }
  if (payload.status !== 'ok' && payload.status !== 'no_execution_data') {
    return <p>No execution analytics payload available yet.</p>;
  }

  const summaryEntries = Object.entries(payload.summary);

  return (
    <div className="page-stack">
      <div className="metric-grid">
        {summaryEntries.slice(0, 8).map(([key, value]) => (
          <article key={key}>
            <span>{key}</span>
            <strong>{String(value)}</strong>
          </article>
        ))}
      </div>

      <div className="card-grid">
        <article className="feature-card">
          <div className="feature-header">
            <h4>Direction Breakdown</h4>
            <span className="status-pill status-available">{payload.direction_breakdown.length} rows</span>
          </div>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Direction</th>
                  <th>Closed Trades</th>
                  <th>Win Rate</th>
                </tr>
              </thead>
              <tbody>
                {payload.direction_breakdown.map((row) => (
                  <tr key={row.direction}>
                    <td>{row.direction}</td>
                    <td>{row.closed_trades}</td>
                    <td>{row.win_rate}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </article>

        <article className="feature-card">
          <div className="feature-header">
            <h4>Order Status</h4>
            <span className="status-pill status-available">{payload.order_status.length} states</span>
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
                    <td>{row.status}</td>
                    <td>{row.count}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </article>
      </div>

      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Closed Trade Time</th>
              <th>Symbol</th>
              <th>Side</th>
              <th>Realized PnL</th>
              <th>Return %</th>
              <th>Holding Seconds</th>
            </tr>
          </thead>
          <tbody>
            {payload.recent_closed_trades.length > 0 ? (
              payload.recent_closed_trades.map((trade) => (
                <tr key={`${trade.timestamp}-${trade.symbol}-${trade.realized_pnl}`}>
                  <td>{trade.timestamp}</td>
                  <td>{trade.symbol}</td>
                  <td>{trade.close_side}</td>
                  <td>{trade.realized_pnl}</td>
                  <td>{trade.realized_return_pct ?? 'n/a'}</td>
                  <td>{trade.holding_sec ?? 'n/a'}</td>
                </tr>
              ))
            ) : (
              <tr>
                <td colSpan={6}>No closed trades were available for the latest run.</td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
