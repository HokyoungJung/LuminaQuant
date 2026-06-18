'use client';

import type { PerformancePricePayload } from '@/lib/dashboard-contracts';
import { buildSparklinePath, formatMetricValue } from '@/lib/format';
import { useBridgeFetch } from '@/lib/use-bridge-fetch';

export function PerformancePriceRuntime() {
  const { payload, error } = useBridgeFetch<PerformancePricePayload>(
    '/api/python/dashboard/performance-price',
    'performance-price bridge failed',
  );

  if (error) {
    return <p>{error}</p>;
  }
  if (payload === null) {
    return <p>Loading performance and price telemetry…</p>;
  }
  if (payload.status !== 'ok') {
    return <p>No performance/price payload available yet.</p>;
  }

  const equityPath = buildSparklinePath(payload.equity_curve.map((point) => point.equity));
  const drawdownPath = buildSparklinePath(payload.drawdown_curve.map((point) => point.drawdown));
  const benchmarkPath = buildSparklinePath(payload.benchmark_curve.map((point) => point.price));
  const fundingPath = buildSparklinePath(payload.funding_curve.map((point) => point.funding));

  return (
    <div className="page-stack">
      <div className="metric-grid">
        {payload.summary_metrics.slice(0, 6).map((metric) => (
          <article key={metric.key}>
            <span>{metric.label}</span>
            <strong>{formatMetricValue(metric.value)}</strong>
          </article>
        ))}
      </div>

      <div className="card-grid">
        {equityPath ? (
          <article className="feature-card">
            <div className="feature-header">
              <h4>Equity Curve</h4>
              <span className="status-pill status-available">{payload.equity_curve.length} points</span>
            </div>
            <svg viewBox="0 0 420 120" role="img" aria-label="Equity curve preview">
              <path d={equityPath} fill="none" stroke="currentColor" strokeWidth="3" />
            </svg>
          </article>
        ) : null}
        {drawdownPath ? (
          <article className="feature-card">
            <div className="feature-header">
              <h4>Drawdown Curve</h4>
              <span className="status-pill status-available">{payload.drawdown_curve.length} points</span>
            </div>
            <svg viewBox="0 0 420 120" role="img" aria-label="Drawdown curve preview">
              <path d={drawdownPath} fill="none" stroke="currentColor" strokeWidth="3" />
            </svg>
          </article>
        ) : null}
        {benchmarkPath ? (
          <article className="feature-card">
            <div className="feature-header">
              <h4>Benchmark Price</h4>
              <span className="status-pill status-available">{payload.benchmark_curve.length} points</span>
            </div>
            <svg viewBox="0 0 420 120" role="img" aria-label="Benchmark price preview">
              <path d={benchmarkPath} fill="none" stroke="currentColor" strokeWidth="3" />
            </svg>
          </article>
        ) : null}
        {fundingPath ? (
          <article className="feature-card">
            <div className="feature-header">
              <h4>Funding Trace</h4>
              <span className="status-pill status-available">{payload.funding_curve.length} points</span>
            </div>
            <svg viewBox="0 0 420 120" role="img" aria-label="Funding preview">
              <path d={fundingPath} fill="none" stroke="currentColor" strokeWidth="3" />
            </svg>
          </article>
        ) : null}
      </div>

      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Performance Metric</th>
              <th>Value</th>
            </tr>
          </thead>
          <tbody>
            {Object.entries(payload.performance_metrics).map(([key, value]) => (
              <tr key={key}>
                <td>{key}</td>
                <td>{formatMetricValue(value)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Trade Marker Time</th>
              <th>Symbol</th>
              <th>Direction</th>
              <th>Price</th>
              <th>Qty</th>
              <th>Realized PnL</th>
              <th>Position After</th>
            </tr>
          </thead>
          <tbody>
            {payload.trade_markers.length > 0 ? (
              payload.trade_markers.map((marker) => (
                <tr key={`${marker.timestamp}-${marker.direction}-${marker.price}`}>
                  <td>{marker.timestamp}</td>
                  <td>{marker.symbol}</td>
                  <td>{marker.direction}</td>
                  <td>{marker.price}</td>
                  <td>{marker.quantity}</td>
                  <td>{marker.realized_pnl}</td>
                  <td>{marker.position_after}</td>
                </tr>
              ))
            ) : (
              <tr>
                <td colSpan={7}>No recent trade markers were available for the latest run.</td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
