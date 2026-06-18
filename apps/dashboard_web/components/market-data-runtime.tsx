'use client';

import type { MarketDataPayload } from '@/lib/dashboard-contracts';
import { formatMetricValue } from '@/lib/format';
import { useBridgeFetch } from '@/lib/use-bridge-fetch';

export function MarketDataRuntime() {
  const { payload, error } = useBridgeFetch<MarketDataPayload>(
    '/api/python/dashboard/market-data',
    'market data bridge failed',
  );

  if (error) {
    return <p>{error}</p>;
  }
  if (payload === null) {
    return <p>Loading market data parity payload…</p>;
  }
  if (payload.status !== 'ok' && payload.status !== 'no_market_data') {
    return <p>No market data payload available yet.</p>;
  }

  return (
    <div className="page-stack">
      <div className="metric-grid">
        {payload.summary_metrics.map((metric) => (
          <article key={metric.key}>
            <span>{metric.label}</span>
            <strong>{formatMetricValue(metric.value)}</strong>
          </article>
        ))}
      </div>

      <section className="section-card">
        <div className="section-header">
          <div>
            <p className="eyebrow">Market context</p>
            <h3>Active symbol and strategy</h3>
          </div>
          <div className="metric-badge">{payload.market_context.strategy ?? 'unknown'}</div>
        </div>
        <div className="metric-grid">
          <article>
            <span>Symbol</span>
            <strong>{payload.market_context.symbol ?? 'n/a'}</strong>
          </article>
          <article>
            <span>Timeframe</span>
            <strong>{payload.market_context.timeframe ?? 'n/a'}</strong>
          </article>
          <article>
            <span>Exchange</span>
            <strong>{payload.market_context.exchange ?? 'n/a'}</strong>
          </article>
          <article>
            <span>Data source</span>
            <strong>{payload.market_context.source ?? 'n/a'}</strong>
          </article>
        </div>
      </section>

      <section className="section-card">
        <div className="section-header">
          <div>
            <p className="eyebrow">Indicator parity</p>
            <h3>Guarded market-view summary</h3>
          </div>
          <div className="metric-badge">{payload.indicator_summary.length} items</div>
        </div>
        <div className="metric-grid">
          {payload.indicator_summary.map((metric) => (
            <article key={metric.key}>
              <span>{metric.label}</span>
              <strong>{formatMetricValue(metric.value)}</strong>
            </article>
          ))}
        </div>
        {payload.warnings.length > 0 ? (
          <ul className="guidance-list">
            {payload.warnings.map((warning) => (
              <li key={warning}>{warning}</li>
            ))}
          </ul>
        ) : null}
      </section>

      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Timestamp</th>
              <th>Open</th>
              <th>High</th>
              <th>Low</th>
              <th>Close</th>
              <th>Volume</th>
            </tr>
          </thead>
          <tbody>
            {payload.recent_bars.length > 0 ? (
              payload.recent_bars.map((bar) => (
                <tr key={bar.timestamp}>
                  <td>{bar.timestamp}</td>
                  <td>{formatMetricValue(bar.open)}</td>
                  <td>{formatMetricValue(bar.high)}</td>
                  <td>{formatMetricValue(bar.low)}</td>
                  <td>{formatMetricValue(bar.close)}</td>
                  <td>{formatMetricValue(bar.volume)}</td>
                </tr>
              ))
            ) : (
              <tr>
                <td colSpan={6}>No recent market bars were available for the configured context.</td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
